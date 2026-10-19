# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置 + 日志追踪 + 梯度检查点 + remove_padding + 动态 batch 版 SFT 训练 —— verl/learning/sft/code/6_dynamic_bsz/train.py

==================== 【新增·模块1】用配置系统替换 argparse ====================
(见 2_config_system/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块2】用 Tracking 替换 print ====================
(见 3_tracking/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块3】梯度检查点 ====================
(见 4_grad_checkpoint/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块4】remove_padding / 打包(packing) ====================
(见 5_remove_padding/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块5】动态 batch (max_token_len_per_gpu) ====================
在模块4 基础上继续改动:
  - remove_padding=true 时, DataLoader 改用 batch_sampler=TokenBudgetBatchSampler
    (按 token 预算切 batch, 每批塞到约 max_token_len_per_gpu 个 token), 不再用固定 micro_batch_size。
  - 每 epoch 调 sampler.set_epoch(epoch) 换打乱种子。
  - remove_padding=false 时仍走固定 micro_batch_size + 右填充(模块4 的对照路径, 不变)。
意义: 固定按样本数切 batch 时, 样本长度不齐 -> 短 batch 显存空跑、长 batch 要 pad 到 max 更浪费;
按 token 预算切 -> 每个 batch 显存利用率恒定, 不受长度不齐影响。这也是 remove_padding 的天然搭档
(packing 能容纳任意大小的 batch, "按 token 切"才有意义)。对应 verl trainer.max_token_len_per_gpu。
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
from dataset import SFTDataset, make_collator, make_packing_collator, TokenBudgetBatchSampler   # 【新增·模块4/5】多导入 make_packing_collator, TokenBudgetBatchSampler


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

    # 【新增·模块3】梯度检查点(gradient checkpointing)
    # 原理: 反向传播时不再保存每层中间激活, 而是在需要时重算前向片段 -> 用时间换显存(省激活)。
    # 0.5B 不开也行, 但模型一大(3B+)反向存激活会爆显存, 这是标配。对应 verl enable_gradient_checkpointing。
    if cfg.model.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()   # 内部会设 model.config.use_cache=False
        model.enable_input_require_grads()      # 让输入嵌入需梯度(检查点的配套, 否则首层输入无梯度)
        print("已开启梯度检查点(用重算换激活显存)")

    # 2. 数据集 + dataloader
    train_ds = SFTDataset(
        cfg.data.train_file, tokenizer,
        max_length=cfg.data.max_length, max_samples=cfg.data.max_train_samples,
    )
    # 【新增·模块4】按 remove_padding 选 collator: 打包(无 pad) 或 右填充(对照)
    # 【新增·模块5】packing 模式下, DataLoader 改用 batch_sampler=TokenBudgetBatchSampler
    #   (按 token 预算切 batch, 不再用固定 micro_batch_size); 右填充模式仍走固定 micro_batch_size。
    batch_sampler = None
    if cfg.trainer.remove_padding:
        collate_fn = make_packing_collator()
        batch_sampler = TokenBudgetBatchSampler(
            train_ds.lengths,
            max_token_len_per_gpu=cfg.trainer.max_token_len_per_gpu,
            shuffle=True, drop_last=True, seed=cfg.trainer.seed,
        )
        print(f"collator: packing + 动态 batch(remove_padding=true, "
              f"max_token_len_per_gpu={cfg.trainer.max_token_len_per_gpu}, "
              f"估算 {len(batch_sampler)} 个 micro-batch/epoch)")
        train_dl = DataLoader(
            train_ds, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=2,
        )
    else:
        collate_fn = make_collator(tokenizer.pad_token_id or tokenizer.eos_token_id)
        print("collator: 右填充 + 固定 micro_batch_size(remove_padding=false, 与之前模块一致)")
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
    # 【新增·模块5】等效 batch 不再是 micro_batch_size*accum(动态 batch 下样本数不定),
    # 改用 token 预算 * accum 表征(每步处理的 token 量大致恒定)
    equiv_batch_desc = (f"~{cfg.trainer.max_token_len_per_gpu*cfg.trainer.grad_accum_steps} tokens/step"
                        if cfg.trainer.remove_padding else f"{cfg.trainer.micro_batch_size*cfg.trainer.grad_accum_steps} samples/step")
    print(f"总优化步数: {total_steps} (每 epoch {steps_per_epoch} 步, 等效 batch={equiv_batch_desc})")

    # 4. 训练循环(与基础版完全一致)
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(cfg.trainer.epochs):
        # 【新增·模块5】动态 batch 模式下, 每 epoch 换打乱种子(对应 verl DataLoader.set_epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        for micro_idx, batch in enumerate(train_dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.trainer.bf16):
                # 【新增·模块4】packing 模式下 attention_mask 是 4D 块对角因果掩码(不再是 2D padding mask),
                # 并多传 position_ids(每样本内从 0 重置)。右填充模式时没有 position_ids, 走原路径。
                fwd_kwargs = dict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                if "position_ids" in batch:   # packing collator 才会产出 position_ids
                    fwd_kwargs["position_ids"] = batch["position_ids"]
                out = model(**fwd_kwargs)
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
