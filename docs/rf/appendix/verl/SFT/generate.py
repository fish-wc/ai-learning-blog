# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
微调效果测试 —— verl/learning/sft/generate.py

【这一步在做什么】
拿 gsm8k 测试集的若干道题，分别用 "微调前(原始模型)" 和 "微调后(我们训出来的模型)" 生成回答，
并对比、统计正确率，让你直观看到 SFT 的效果。

做法很朴素:
  - 用 tokenizer.apply_chat_template 拼 prompt(add_generation_prompt=True)
  - model.generate 贪心解码
  - 用正则抠出 #### 后面的数字，和标准答案比对
"""
import argparse
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text: str):
    """从生成文本里抠 #### 后面的数字；抠不到返回 None。"""
    m = re.search(r"####\s*(\-?[0-9\.\,]+)", text)
    if m is None:
        return None
    return m.group(1).replace(",", "").strip()


def load_test_questions(test_parquet, n=10):
    df = pd.read_parquet(test_parquet)
    items = []
    for _, row in df.head(n).iterrows():
        msgs = list(row["messages"])
        # messages: [user(含问题), assistant(含答案)]
        user_msg = msgs[0]["content"]
        gold_answer_raw = msgs[1]["content"]
        gold = extract_answer(gold_answer_raw)
        items.append({"question": user_msg, "gold": gold})
    return items


def generate_answer(model, tokenizer, question, max_new_tokens=256):
    """对一道题生成回答。"""
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # 贪心，结果可复现
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # 只取新生成的部分(去掉 prompt)
    new_tokens = out[0, prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def evaluate(model_path, questions, label, device="cuda"):
    print(f"\n========== 加载模型: {label} ==========")
    print(f"  路径: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    correct = 0
    total = len(questions)
    for i, q in enumerate(questions):
        gen = generate_answer(model, tokenizer, q["question"])
        pred = extract_answer(gen)
        ok = (pred is not None) and (q["gold"] is not None) and (pred == q["gold"])
        if ok:
            correct += 1
        # 只详细打印前 3 条，其余只统计
        if i < 3:
            print(f"\n--- 题 {i+1} ---")
            print(f"问题: {q['question'][:100]}...")
            print(f"标准答案: {q['gold']}")
            print(f"模型生成(截断显示):\n{gen[:400]}")
            print(f"抽取答案: {pred}  ->  {'✓ 正确' if ok else '✗ 错误'}")
    acc = correct / total if total else 0.0
    print(f"\n[{label}] 正确率: {correct}/{total} = {acc:.1%}")
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default=os.path.join(os.path.dirname(__file__), "data/test.parquet"))
    p.add_argument("--base_model", default="/home/zzy/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct",
                   help="微调前的原始模型")
    p.add_argument("--ft_model", default=os.path.join(os.path.dirname(__file__), "checkpoints/sft-qwen0.5b-gsm8k"),
                   help="微调后的模型目录")
    p.add_argument("--num_questions", type=int, default=20)
    args = p.parse_args()

    questions = load_test_questions(args.test_file, n=args.num_questions)
    print(f"加载 {len(questions)} 道测试题")

    base_acc = evaluate(args.base_model, questions, label="微调前(原始模型)")
    ft_acc = evaluate(args.ft_model, questions, label="微调后(SFT)")

    print("\n" + "=" * 50)
    print(f"对比结论:")
    print(f"  微调前正确率: {base_acc:.1%}")
    print(f"  微调后正确率: {ft_acc:.1%}")
    print(f"  提升: {ft_acc - base_acc:+.1%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
