# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
准备 gsm8k_sft 数据 —— verl/learning/sft/prepare_data.py

【这一步在做什么】
verl 原版用 examples/data_preprocess/gsm8k_multiturn_sft.py 把 GSM8k 转成 parquet。
这里把它"消融"到最小：下载 gsm8k -> 每条数据拼成 messages(user/assistant) -> 存成 parquet。

输出:
    data/train.parquet   列: messages  -> [{role:"user",content:...},{role:"assistant",content:...}]
    data/test.parquet

为什么用 messages 格式？因为 verl 的 MultiTurnSFTDataset 读的就是这个列(由 data.messages_key 指定)。
我们沿用同样的格式，方便后面逐步往 verl 靠拢。
"""
import argparse
import os
import re

import pandas as pd


def extract_solution(answer_str: str) -> str:
    """从 gsm8k 答案里抠出 #### 后面的数字，作为最终答案。"""
    m = re.search(r"#### (\-?[0-9\.\,]+)", answer_str)
    assert m is not None, f"无法从答案中提取数字: {answer_str}"
    return m.group(0).split("#### ")[1].replace(",", "")


def build_messages(question: str, answer: str):
    """把一条 gsm8k 样本转成 chat messages 格式。

    user:      原始问题 + 一句"逐步思考"提示(和 verl 原版一致)
    assistant: 原始答案(含思维链和 #### 数字)
    """
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    return [
        {"role": "user", "content": f"{question} {instruction}"},
        {"role": "assistant", "content": answer},
    ]


def load_gsm8k_split(split: str):
    """通过 modelscope 下载 gsm8k(服务器无外网，只能走 modelscope 镜像)。"""
    from modelscope.msdatasets import MsDataset

    # AI-ModelScope/gsm8k 是 openai/gsm8k 的 modelscope 镜像，subset='main'
    ds = MsDataset.load(
        "AI-ModelScope/gsm8k", subset_name="main", split=split, trust_remote_code=True
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--max_train", type=int, default=-1, help="只取前 N 条训练数据，-1 表示全部")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    rows = []
    for split, fname in [("train", "train.parquet"), ("test", "test.parquet")]:
        print(f"[{split}] 下载 gsm8k ...")
        ds = load_gsm8k_split(split)
        n = len(ds)
        if split == "train" and args.max_train > 0:
            n = min(n, args.max_train)
        records = []
        for i in range(n):
            ex = ds[i]
            q = ex["question"]
            a = ex["answer"]
            records.append({"messages": build_messages(q, a)})
        df = pd.DataFrame(records)
        out = os.path.join(args.data_dir, fname)
        df.to_parquet(out)
        print(f"[{split}] 写入 {len(df)} 条 -> {out}")
        if split == "test":
            # 顺带打印一条样例方便人眼检查格式
            print("  样例 messages:")
            for m in records[0]["messages"]:
                print(f"    {m['role']}: {m['content'][:80]}...")
            rows = records

    # 校验: 确认能读回来
    chk = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    print(f"\n校验: 读回 {len(chk)} 条, 列={list(chk.columns)}")


if __name__ == "__main__":
    main()
