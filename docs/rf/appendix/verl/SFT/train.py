# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
最小 SFT 训练 —— verl/learning/sft/train.py

【这一步在做什么】
把"模型 + 优化器 + 学习率调度 + 训练循环"组装起来，这是 SFT 最原始的骨架:

    for batch in dataloader:
        out  = model(input_ids, attention_mask, labels)   # 前向，HF 内部自动 shift + 算 CE loss
        loss = out.loss / grad_accum_steps                 # 梯度累积
        loss.backward()                                    # 反向
        if 是累积的最后一步:
            clip_grad_norm_(model, max_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

------ 和 verl 原版的对比(消融说明) ------
verl 的 SFTTrainer (verl/trainer/sft_trainer.py) 在这之上加了很多工程能力:
  - 分布式: torch.distributed + FSDP 分片 + DP all-reduce(这里全砍掉，单卡)
  - 引擎抽象: TrainingWorker -> FSDPEngine，loss 在 engine 里用 logprobs_from_logits 手算
              (这里直接用 model(...).loss，让 HF 内部算，最直观)
  - 动态 batch / remove_padding: 省显存(这里用固定右填充，最简单)
  - TensorDict + StatefulDataLoader: 统一数据协议 + 断点续训(这里用普通 DataLoader)
  - CheckpointHandler: 多备份/恢复/LoRA 元信息(这里只存最终权重)
  - Tracking(wandb 等): 日志(这里只 print)
这些能力在 README 路线图里会逐个加回，每一步说清"为什么需要、解决什么问题"。
"""
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import SFTDataset, make_collator


def parse_args():
    p = argparse.ArgumentParser()
    # 模型
    p.add_argument("--model_path", default="/home/zzy/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct")
    # 数据
    p.add_argument("--train_file", default=os.path.join(os.path.dirname(__file__), "data/train.parquet"))
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_train_samples", type=int, default=1000, help="取子集加速，-1=全部")
    # 训练(参数都尽量小，不把服务器跑崩)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=4, help="梯度累积步数，等效 batch = micro * accum")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    # 输出
    p.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "checkpoints/sft-qwen0.5b-gsm8k"))
    p.add_argument("--log_every", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 tokenizer 和 模型
    #    对应 verl FSDPEngine._build_module: AutoModelForCausalLM.from_pretrained(...)
    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).to("cuda")
    model.train()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 2. 数据集 + dataloader
    #    对应 verl 的 _build_dataset + _build_dataloader
    train_ds = SFTDataset(
        args.train_file, tokenizer,
        max_length=args.max_length, max_samples=args.max_train_samples,
    )
    collate_fn = make_collator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    train_dl = DataLoader(
        train_ds, batch_size=args.micro_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, drop_last=True,
    )

    # 3. 优化器 + 学习率调度
    #    对应 verl 的 build_optimizer + _build_lr_scheduler (constant/cosine)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )
    print(f"总优化步数: {total_steps} (每 epoch {steps_per_epoch} 步, 等效 batch={args.micro_batch_size*args.grad_accum_steps})")

    # 4. 训练循环(带梯度累积 + bf16 autocast + 梯度裁剪)
    #    对应 verl SFTTrainer.fit() 的 for epoch / for step 主循环
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for micro_idx, batch in enumerate(train_dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            # 前向: 传 labels，HF 会内部做 shift + CrossEntropyLoss(ignore_index=-100)
            # 这就是 verl 里 "logprobs_from_logits + sft_loss" 的最简等价
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = out.loss / args.grad_accum_steps

            loss.backward()  # 反向，梯度累积

            is_accum_last = (micro_idx + 1) % args.grad_accum_steps == 0
            if is_accum_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    lr = lr_scheduler.get_last_lr()[0]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss={out.loss.item():.4f} lr={lr:.2e} "
                          f"grad_norm 已裁剪 max={args.max_grad_norm}")

    # 5. 保存最终模型(只存权重，不存优化器，省磁盘)
    #    对应 verl 的 ckpt_handler.save_checkpoint，但极简
    print(f"保存模型到 {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("训练完成。")


if __name__ == "__main__":
    main()
