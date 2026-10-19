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

==================== 【新增·模块4】remove_padding / 打包(packing) ====================
在模块3 基础上继续改动:
  - 保留原 make_collator(右填充版, 作为对照/可切换),
  - 新增 make_packing_collator: 把一个 micro-batch 的变长样本拼成单条 (1,T) 序列,
    额外产出 position_ids(每样本内从 0 重置) + 4D 块对角因果注意力掩码(1,1,T,T)。
  - 原理: 右填充时 pad 位是"废算"(虽然不算 loss 但前向算了一遍注意力),
    打包后所有 token 都有效, 省掉 pad 的浪费; 块对角掩码保证各样本互不可见,
    等价于各样本独立前向, 但少了一堆 padding。
意义: verl 默认就开 remove_padding(用 flash_attn_varlen + cu_seqlens 做 O(n) 变长注意力),
本最小版环境没装 flash_attn, 退用 4D 块对角掩码(物化 O(T^2) 但 0.5B/短序列完全够),
数学上和 verl 的 no_padding 路径等价。详见 make_packing_collator 注释。

==================== 【新增·模块5】动态 batch (max_token_len_per_gpu) ====================
在模块4 基础上继续改动:
  - SFTDataset 预 tokenize 并缓存(顺便得到每条样本长度 self.lengths)。
  - 新增 TokenBudgetBatchSampler: 按 token 预算切 batch —— 累积样本直到总 token 接近
    max_token_len_per_gpu 就 yield 一个 batch。每个 batch 的 token 总量大致恒定。
意义: 固定 micro_batch_size(按样本数切)时, 样本长度不齐 -> 有的 batch 短(显存浪费)、
有的 batch 碰到超长样本要 pad 到 max_length(更浪费)。改成按 token 预算切, 每个 batch
都把显存填到差不多的水平 -> 利用率稳定, 不再受长度不齐影响。
verl 对应: trainer.max_token_len_per_gpu(配合 remove_padding 一起用)。
注意: 这个动态 batch 只在 remove_padding=true 时启用(packing 才能容纳变长 batch);
remove_padding=false 时仍走固定 micro_batch_size + 右填充(模块4 的对照路径)。
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
        # 【新增·模块5】预 tokenize 并缓存(原本是 __getitem__ 里惰性 tokenize, 现在 __init__ 一次性做完)。
        # 好处: 1) 采样器能直接拿到每条长度 self.lengths 做按 token 预算切分;
        #       2) 不再重复 tokenize(原本采样器查长度要 tokenize 一次, __getitem__ 又 tokenize 一次)。
        self.cache = []
        self.lengths = []
        for msg in self.messages:
            ids, lab = build_input_and_labels(msg, tokenizer, max_length)
            self.cache.append((ids, lab))
            self.lengths.append(len(ids))
        print(f"[SFTDataset] 加载 {len(self.messages)} 条, max_length={max_length}, "
              f"pad_token_id={self.pad_token_id}, "
              f"token 长度: min={min(self.lengths)} max={max(self.lengths)} 均值={sum(self.lengths)/len(self.lengths):.0f}")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        # 【新增·模块5】从缓存取, 不再现场 tokenize
        ids, lab = self.cache[idx]
        return {
            "input_ids": ids,
            "labels": lab,
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


# 【新增·模块4】打包(packing) collator —— remove_padding 的最小实现
def make_packing_collator():
    """把一个 micro-batch 的变长样本**拼接**成单条 (1, T) 序列(不再右填充)。

    对应 verl 的 remove_padding / no_padding 路径
    (verl/utils/dataset/sft_tensor_collator.py 的 remove_padding=True 分支)。

    【和右填充(make_collator)的区别】
    右填充: 把 B 个样本各自 pad 到 max_len, 得 (B, max_len)。pad 位前向照算(虽不算 loss),
    且 max_len 取决于本 batch 最长样本, 短样本 pad 一大堆 -> 浪费算力 + 浪费显存。
    打包:   把 B 个样本首尾相接拼成 (1, T), T = sum(各样本长度)。没有 pad,
    全部 token 都参与且有效; 用 4D 块对角因果掩码让各样本互不可见 -> 数学等价于各自独立前向。

    【verl 原版怎么做的】
    verl 用 flash_attn_varlen: 只需喂一个 cu_seqlens(每样本起始偏移), flash 内部 O(n) 分块,
    不物化 (T,T) 掩码。本环境没装 flash_attn, 退而物化一个 (1,1,T,T) 的块对角因果掩码:
      - 每个样本块内是因果下三角(可看自己及之前)
      - 块间全 -inf(互不可见)
    内存 O(T^2), 但 0.5B + 短序列(如 gsm8k T≈1600)完全吃得下, 且结果和 verl 数值等价。

    【边界处 loss 为什么自动正确】
    packed 序列里样本 i 的末尾(<|im_end|>, label=真实id, 要学)紧接样本 i+1 的开头
    (<|im_start|>system, label=-100, 不学)。HF 内部 loss 做 shift: logits[t] 预测 label[t+1]。
    于是在边界处 logits[样本i末] 去预测 label[样本i+1首] = -100 -> 自动不算 loss, 无需特判。
    """

    def collate_fn(batch):
        # 1) 拼接所有样本的 input_ids / labels, 记录每样本长度(用来切块)
        packed_ids, packed_labels, lengths = [], [], []
        for b in batch:
            packed_ids.extend(b["input_ids"])
            packed_labels.extend(b["labels"])
            lengths.append(len(b["input_ids"]))
        T = len(packed_ids)
        if T == 0:
            raise ValueError("打包 collator 收到空 batch")
        input_ids = torch.tensor([packed_ids], dtype=torch.long)    # (1, T)
        labels = torch.tensor([packed_labels], dtype=torch.long)    # (1, T)

        # 2) position_ids: 每个样本内从 0 重置。
        #    因为注意力是块对角的(样本间互不可见), 位置编码也必须"各自为政"重置,
        #    否则样本 i+1 会接着样本 i 的绝对位置编号 -> RoPE 位置乱套。
        pos = []
        for L in lengths:
            pos.extend(range(L))
        position_ids = torch.tensor([pos], dtype=torch.long)        # (1, T)

        # 3) 4D 块对角因果注意力掩码 (1,1,T,T)。
        #    传给 SDPA 当作 additive bias: 0=可看, -inf=不可看。
        #    dtype 必须 = query dtype(bfloat16), 否则 SDPA 报 "invalid dtype for bias"。
        mask = torch.full((1, 1, T, T), float("-inf"), dtype=torch.bfloat16)
        offset = 0
        for L in lengths:
            # 块内因果下三角(含对角)置 0: 样本 i 的第 a 个 token 能看本样本的第 0..a 个
            tri = torch.tril(torch.ones((L, L), dtype=torch.bool))  # True=对角及以下
            mask[0, 0, offset:offset + L, offset:offset + L].masked_fill_(tri, 0.0)
            offset += L

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,        # 【新增·模块4】packing 模式额外产出
            "attention_mask": mask,             # 【新增·模块4】4D 块对角掩码(取代 2D padding mask)
        }

    return collate_fn


# 【新增·模块5】按 token 预算切 batch 的采样器 —— 动态 batch 的核心
class TokenBudgetBatchSampler:
    """每个 batch 累积样本直到总 token 数接近 max_token_len_per_gpu, 再 yield。

    对应 verl 的 max_token_len_per_gpu 机制(verl/trainer/.../sft_trainer 走的就是这套):
      - 固定 micro_batch_size(按样本数切): 样本长度不齐时, 短样本 batch 显存空跑,
        长样本 batch 还得 pad 到 max_length 更浪费。
      - 按 token 预算切: 每个 batch 塞到差不多 max_token_len_per_gpu 个 token,
        显存/算力利用率恒定, 不受长度不齐影响 —— 这也是 remove_padding 的天然搭档
        (packing 能容纳任意大小的 batch, 才让"按 token 切"有意义)。

    贪心策略: 顺序遍历打乱后的下标, 只要加进去不超预算就加; 超了就 yield 当前 batch 开新 batch。
    单条样本超预算时它自成一批(无法再切)。__len__ 给的是估算 batch 数(供调度器用)。
    """

    def __init__(self, lengths, max_token_len_per_gpu, shuffle=True, drop_last=False, seed=42):
        self.lengths = list(lengths)
        self.max_token_len = max_token_len_per_gpu
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """每 epoch 用不同 seed 打乱(对应 verl/DataLoader 的 set_epoch)。"""
        self.epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            idx = torch.randperm(n, generator=g).tolist()
        else:
            idx = list(range(n))
        # 贪心累积
        batch, batch_tokens = [], 0
        for i in idx:
            L = self.lengths[i]
            if batch and batch_tokens + L > self.max_token_len:
                yield batch
                batch, batch_tokens = [], 0
            batch.append(i)
            batch_tokens += L
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        # 估算: 总 token / 预算(贪心实际数会因打乱顺序略有出入, 但量级一致, 够调度器用)
        total = sum(self.lengths)
        return max(1, (total + self.max_token_len - 1) // self.max_token_len)
