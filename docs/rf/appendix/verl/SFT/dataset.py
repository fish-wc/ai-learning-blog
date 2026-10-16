# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
SFT 数据集 —— verl/learning/sft/dataset.py

【这一步在做什么】
把一条 messages 对话 -> token id 序列 + 标签(labels)，其中:
  - input_ids: 整段对话的 token(包含 user 和 assistant)
  - labels:   和 input_ids 等长，"要学的位置"放真实 token id，"不学的位置"放 -100

为什么用 -100？这是 HuggingFace CrossEntropyLoss 的约定(ignore_index=-100)：
模型内部会把 labels 左移一位做 next-token 预测，遇到 -100 的位置不算 loss。
所以我们只要标好"哪些 token 是 assistant 的回答内容"，模型就只在这些位置算损失。

------ 和 verl 原版的对比(消融说明) ------
verl 的 MultiTurnSFTDataset (verl/utils/dataset/multiturn_sft_dataset.py) 做得更复杂:
  1. 逐条 message 单独 apply_chat_template，再拼接(需要剥掉重复的系统提示词)
  2. 产出 loss_mask(0/1)，不用 -100
  3. 损失函数里手动 log_softmax + 左移 mask 对齐
  4. 支持 no_padding(嵌套张量/jagged)以省显存
本最小版把这些都砍掉，用最朴素的方式:
  - 整段对话一次性 apply_chat_template 得到 input_ids
  - 用"前缀增量 tokenize"定位每个 assistant 回答的 token 区间，把该区间的 labels 设成真实 id
  - 右填充 + 用 HF 自带的 labels/-100 机制(模型内部自动 shift)
这样代码最短、最好懂，是 SFT "最原始"的形态。后续模块(见 README 路线图)再逐步加回 verl 的能力。
"""
import os

import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_int_list(x):
    """apply_chat_template(tokenize=True) 在不同 transformers 版本可能返回 list / BatchEncoding，统一成 list[int]。"""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    return list(x)


def build_input_and_labels(messages, tokenizer, max_length):
    """把 messages 转成 (input_ids, labels) 两个 list[int]。

    定位 assistant 回答 token 的方法(前缀增量法，直观但 O(n^2)，n 是轮数；
    gsm8k 是单轮，开销可忽略):
      - full_ids        = tokenize(全部 messages, add_generation_prompt=False)
      - 对第 i 条 assistant 消息:
            prompt_ids  = tokenize(messages[:i], add_generation_prompt=True)   # 到 <|im_start|>assistant\n 为止
            end_ids     = tokenize(messages[:i+1], add_generation_prompt=False) # 到这条 assistant 结束
          那么 [len(prompt_ids), len(end_ids)) 就是这条 assistant 回答内容(含结尾 <|im_end|>)的 token，
          把这些位置的 labels 设成真实 token。
      - 其余位置 labels=-100(不学)。
    """
    # 整段对话的 token
    full_ids = _to_int_list(
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True
        )
    )

    labels = [-100] * len(full_ids)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # 到"assistant 生成提示"为止的 token(含 <|im_start|>assistant\n)，这部分不算 loss
        prompt_ids = _to_int_list(
            tokenizer.apply_chat_template(
                messages[:i], add_generation_prompt=True, tokenize=True
            )
        )
        # 到"这条 assistant 结束"为止的 token
        end_ids = _to_int_list(
            tokenizer.apply_chat_template(
                messages[: i + 1], add_generation_prompt=False, tokenize=True
            )
        )
        start = len(prompt_ids)
        end = len(end_ids)
        # 把 assistant 回答内容(含结尾标记)对应的位置设成真实 id
        for j in range(start, min(end, len(full_ids))):
            labels[j] = full_ids[j]

    # 截断到 max_length(从右截断，保留开头；gsm8k 很短一般不会超)
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    return full_ids, labels


class SFTDataset(Dataset):
    """最小 SFT 数据集：读 parquet(messages 列) -> input_ids + labels。

    对应 verl 的 MultiTurnSFTDataset，但只保留最基础功能：
    读 parquet、tokenize、构造 labels、右填充对齐。
    """

    def __init__(self, parquet_file, tokenizer, max_length=512, max_samples=-1):
        df = pd.read_parquet(parquet_file)
        # parquet 里的 messages 列可能是 numpy array 嵌套，转成 python list
        messages_col = df["messages"].tolist()
        self.messages = [list(m) for m in messages_col]
        if max_samples > 0:
            self.messages = self.messages[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            # Qwen2.5 默认 pad_token 可能未设，用 eos 兜底
            self.pad_token_id = tokenizer.eos_token_id
        print(f"[SFTDataset] 加载 {len(self.messages)} 条, max_length={max_length}, "
              f"pad_token_id={self.pad_token_id}")

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        messages = self.messages[idx]
        input_ids, labels = build_input_and_labels(messages, self.tokenizer, self.max_length)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def make_collator(pad_token_id):
    """构造一个 collator：把一个 batch 的变长样本右填充到等长。

    对应 verl 的 SFTTensorCollator。verl 默认用 no_padding(嵌套张量)省显存，
    这里为了好懂用最朴素的右填充:
      - input_ids 右填充 pad_token_id
      - labels 右填充 -100(填充位置不算 loss)
      - attention_mask: 真实 token=1, 填充=0
    """

    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    return collate_fn
