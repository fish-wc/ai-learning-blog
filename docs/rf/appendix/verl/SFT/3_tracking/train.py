# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置 + 日志追踪版 SFT 训练 —— verl/learning/sft/code/3_tracking/train.py

==================== 【新增·模块1】用配置系统替换 argparse ====================
本文件在 1_base_sft/train.py 的基础上改动:
  - 删掉 parse_args() 里十几个 argparse.add_argument
  - 改为: 读 config.yaml 默认值 + 命令行 key.path=value 覆盖 -> SFTConfig dataclass
  - 训练逻辑完全不变, 只是把 args.X 换成 cfg.group.X
  - 开头打印"最终生效配置"(配置即实验记录, 方便复现/对比)

意义: 参数从"扁平命令行"变成"分组 + 可存档 + 可覆盖", 详见 config.py 顶部说明。
训练循环本身(模型/优化器/梯度累积/bf16/裁剪)和基础版一字不差, 故下面只标注改动处。

==================== 【新增·模块2】用 Tracking 替换 print ====================
在模块1 基础上继续改动:
  - 训练里不再 print(loss...), 改为 tracker.log({"train/loss":..,"train/lr":..,"train/grad_norm":..}, step)
  - 指标同时落 console(打印) + file(jsonl 文件, 可事后回看/画曲线)
  - 顺手捕获 clip_grad_norm_ 返回的真实 grad_norm 作为指标(基础版只是裁剪没记录)
  - 训练结束 tracker.finish() 关闭文件
意义: 指标持久化 + 可插拔后端。详见 tracking.py 顶部说明。
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
# 【新增·模块2】引入追踪系统
from tracking import Tracking
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

    # 【新增·模块2】创建追踪器: 后端来自 cfg.trainer.logger, 日志写到 outputs/logs/
    # 对应 verl: tracking = Tracking(project_name, experiment_name, default_backend=cfg.trainer.logger, config=cfg)
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "logs")
    tracker = Tracking(
        project_name=cfg.trainer.project_name,
        experiment_name=cfg.trainer.experiment_name,
        default_backend=list(cfg.trainer.logger),
        config=config_to_yaml_str(cfg),
        output_dir=logs_dir,
    )

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
                # 【新增·模块2】捕获真实 grad_norm(裁剪前的范数), 作为指标记录/监控
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.max_grad_norm
                ).item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.trainer.log_every == 0 or global_step == 1:
                    lr = lr_scheduler.get_last_lr()[0]
                    # 【新增·模块2】指标走 Tracking: 同时落 console(打印) + file(jsonl)
                    # 对应 verl: tracking.log(data=metrics, step=global_step)
                    metrics = {
                        "train/loss": out.loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                    }
                    tracker.log(metrics, step=global_step)

    # 5. 保存模型
    # 【新增·模块1】save_model 开关: 验证配置时可设 false 省磁盘
    if cfg.trainer.save_model:
        print(f"保存模型到 {cfg.trainer.output_dir}")
        model.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(cfg.trainer.output_dir)
    else:
        print("save_model=false, 跳过保存(仅验证配置/流程)")
    # 【新增·模块2】收尾追踪器(关闭 jsonl 文件等), 对应 verl tracking.finish()
    tracker.finish()
    print("训练完成。")


if __name__ == "__main__":
    main()
