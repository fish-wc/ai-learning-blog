# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置版 SFT 训练 —— verl/learning/sft/code/2_config_system/train.py

==================== 【新增·模块1】用配置系统替换 argparse ====================
本文件在 1_base_sft/train.py 的基础上改动:
  - 删掉 parse_args() 里十几个 argparse.add_argument
  - 改为: 读 config.yaml 默认值 + 命令行 key.path=value 覆盖 -> SFTConfig dataclass
  - 训练逻辑完全不变, 只是把 args.X 换成 cfg.group.X
  - 开头打印"最终生效配置"(配置即实验记录, 方便复现/对比)

意义: 参数从"扁平命令行"变成"分组 + 可存档 + 可覆盖", 详见 config.py 顶部说明。
训练循环本身(模型/优化器/梯度累积/bf16/裁剪)和基础版一字不差, 故下面只标注改动处。
"""
import math
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# 【新增·模块1】引入配置系统
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SFTConfig, load_config, config_to_yaml_str
from dataset import SFTDataset, make_collator


# 【新增·模块1】解析命令行: --config <yaml> + 其余 key.path=value 覆盖
def parse_cli():
    args = sys.argv[1:]
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    overrides = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--config":
            yaml_path = args[i + 1]
            i += 2
        elif "=" in a and not a.startswith("--"):
            overrides.append(a)
            i += 1
        else:
            i += 1  # 忽略未知参数
    return load_config(yaml_path, overrides), overrides


def main():
    # 【新增·模块1】配置来自 yaml + 覆盖, 不再是 argparse
    cfg, overrides = parse_cli()
    torch.manual_seed(cfg.trainer.seed)
    os.makedirs(cfg.trainer.output_dir, exist_ok=True)

    # 【新增·模块1】打印最终生效的配置(含命令行覆盖), 方便复现
    print("=" * 60)
    print("最终生效配置(配置即实验记录):")
    if overrides:
        print("  命令行覆盖:", overrides)
    print(config_to_yaml_str(cfg))
    print("=" * 60)

    # 1. 加载 tokenizer 和 模型
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "fp16": torch.float16}[cfg.model.dtype]
    print(f"加载模型: {cfg.model.path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code,
    ).to("cuda")
    model.train()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 2. 数据集 + dataloader
    train_ds = SFTDataset(
        cfg.data.train_file, tokenizer,
        max_length=cfg.data.max_length, max_samples=cfg.data.max_train_samples,
    )
    collate_fn = make_collator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.trainer.micro_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, drop_last=True,
    )

    # 3. 优化器 + 学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )
    steps_per_epoch = math.ceil(len(train_dl) / cfg.trainer.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.trainer.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.optim.warmup_steps, num_training_steps=total_steps,
    )
    print(f"总优化步数: {total_steps} (每 epoch {steps_per_epoch} 步, 等效 batch={cfg.trainer.micro_batch_size*cfg.trainer.grad_accum_steps})")

    # 4. 训练循环(与基础版完全一致)
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(cfg.trainer.epochs):
        for micro_idx, batch in enumerate(train_dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.trainer.bf16):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = out.loss / cfg.trainer.grad_accum_steps
            loss.backward()

            is_accum_last = (micro_idx + 1) % cfg.trainer.grad_accum_steps == 0
            if is_accum_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.trainer.log_every == 0 or global_step == 1:
                    lr = lr_scheduler.get_last_lr()[0]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss={out.loss.item():.4f} lr={lr:.2e} "
                          f"grad_norm 已裁剪 max={cfg.optim.max_grad_norm}")

    # 5. 保存模型
    # 【新增·模块1】save_model 开关: 验证配置时可设 false 省磁盘
    if cfg.trainer.save_model:
        print(f"保存模型到 {cfg.trainer.output_dir}")
        model.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(cfg.trainer.output_dir)
    else:
        print("save_model=false, 跳过保存(仅验证配置/流程)")
    print("训练完成。")


if __name__ == "__main__":
    main()
