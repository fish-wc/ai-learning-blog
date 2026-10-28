# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置 + 日志追踪 + 梯度检查点 + remove_padding + 动态 batch + 手算 loss + 分布式进程组 版 SFT 训练
—— verl/learning/sft/code/8_distributed_init/train.py

==================== 【新增·模块1】用配置系统替换 argparse ====================
(见 2_config_system/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块2】用 Tracking 替换 print ====================
(见 3_tracking/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块3】梯度检查点 ====================
(见 4_grad_checkpoint/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块4】remove_padding / 打包(packing) ====================
(见 5_remove_padding/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块5】动态 batch (max_token_len_per_gpu) ====================
(见 6_dynamic_bsz/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块6】手算 log-prob + 统一 sft_loss ====================
(见 7_manual_loss/train.py 顶部说明, 此处保持不变)

==================== 【新增·模块7】分布式进程组 (torch.distributed) ====================
在模块6 基础上继续改动:
  - 启动方式: 从 `python train.py` 改为 `torchrun --nproc_per_node=N train.py`(也保留单进程回退)。
  - main() 开头调 initialize_global_process_group() 建 NCCL 组, 拿 (local_rank, rank, world_size)。
  - device = f'cuda:{local_rank}'; model.to(device) / batch.to(device) 都走本 rank 的卡。
  - rank-aware 控制流: 只有 rank0 打印配置 / 创建 Tracking / 写日志 / 存模型(对应 verl if self.rank==0)。
  - TokenBudgetBatchSampler 加 num_replicas=world_size, rank=rank 做数据分片(对应 verl DistributedSampler)。
  - 收尾 destroy_global_process_group()(所有 rank 都调, 对应 verl run_sft 末尾)。
意义: 建立 verl 一切分布式能力的前置通信骨架(进程组 + rank/world_size + rank0 控制流 + 数据分片)。
【故意停在的中间态】本模块只建组、没同步梯度(没 DDP/FSDP 包 model) -> 各 rank 各自 backward+step,
  模型会发散。这是"没有 DDP 时多卡各自为政"的中间态, 用来理解模块8 DDP 的必要性。
对应 verl: verl/utils/distributed.py 的 initialize_global_process_group; sft_trainer.py 的 run_sft
  (init->fit->destroy 三段式) + SFTTrainer.__init__ 的 rank0 守卫 + _build_dataloader 的 DistributedSampler。
"""
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# 【新增·模块1】引入配置系统
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SFTConfig, load_config, config_to_yaml_str
# 【新增·模块2】引入追踪系统
from tracking import Tracking
from dataset import SFTDataset, make_collator, make_packing_collator, TokenBudgetBatchSampler   # 【新增·模块4/5】多导入 make_packing_collator, TokenBudgetBatchSampler
# 【新增·模块6】引入手算 loss: logprobs_from_logits + sft_loss
from loss import sft_loss
# 【新增·模块7】引入分布式进程组初始化
from dist import initialize_global_process_group, destroy_global_process_group, is_rank0


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

    # 【新增·模块7】分布式进程组初始化 —— torchrun 启动时建 NCCL 组, 单进程回退
    #   对应 verl run_sft(): initialize_global_process_group(); trainer=SFTTrainer(config); trainer.fit()
    local_rank, rank, world_size = initialize_global_process_group(
        backend=cfg.trainer.backend, timeout_second=36000,
    )
    # 用真实 world_size 覆盖 config 占位值(verl 也是运行时从 env 读, 不信 yaml)
    cfg.trainer.world_size = world_size
    # 【新增·模块7】device = 本 rank 的卡。单进程时 local_rank=0 -> cuda:0。
    device = f"cuda:{local_rank}"
    # 【新增·模块7】模型 init 用同一个 seed(各 rank 一致) -> 模块8 DDP 梯度同步才有意义
    #   (否则同步的是不同初始化的梯度, 毫无意义)
    torch.manual_seed(cfg.trainer.seed)
    os.makedirs(cfg.trainer.output_dir, exist_ok=True)
    # 【新增·模块7】rank0 判定: 只有 rank0 打印配置 / 创建 Tracking / 存模型
    #   对应 verl sft_trainer.py L77 `if self.rank == 0: print(self.config)`
    is_main = is_rank0()

    # 【新增·模块1】打印最终生效的配置(含命令行覆盖), 方便复现 —— 【模块7】仅 rank0 打印
    if is_main:
        print("=" * 60)
        print("最终生效配置(配置即实验记录):")
        if overrides:
            print("  命令行覆盖:", overrides)
        print(config_to_yaml_str(cfg))
        print("=" * 60)
        print(f"[dist] world_size={world_size}, rank={rank}, local_rank={local_rank}, "
              f"device={device}, backend={cfg.trainer.backend}")

    # 【新增·模块2】创建追踪器: 后端来自 cfg.trainer.logger, 日志写到 outputs/logs/
    # 【新增·模块7】只在 rank0 创建 Tracking —— FileBackend 用 open(filepath,'w') 独占写,
    #   两个进程同时 'w' 打开同一 jsonl 会互相截断/乱码。非 rank0 的进程 tracker=None。
    #   对应 verl: tracking = Tracking(project_name, experiment_name, default_backend=cfg.trainer.logger, config=cfg)
    tracker = None
    if is_main:
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
    if is_main:
        print(f"加载模型: {cfg.model.path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)
    # 【新增·模块7】model.to(device) 取代 .to("cuda"): 落到本 rank 的卡(必须在 set_device 之后, dist.py 已做)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code,
    ).to(device)
    model.train()
    if is_main:
        print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 【新增·模块3】梯度检查点(gradient checkpointing)
    # 原理: 反向传播时不再保存每层中间激活, 而是在需要时重算前向片段 -> 用时间换显存(省激活)。
    # 0.5B 不开也行, 但模型一大(3B+)反向存激活会爆显存, 这是标配。对应 verl enable_gradient_checkpointing。
    if cfg.model.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()   # 内部会设 model.config.use_cache=False
        model.enable_input_require_grads()      # 让输入嵌入需梯度(检查点的配套, 否则首层输入无梯度)
        if is_main:
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
        # 【新增·模块7】TokenBudgetBatchSampler 加 num_replicas=world_size, rank=rank 做数据分片
        #   对应 verl _build_dataloader: DistributedSampler(num_replicas=dp_size, rank=dp_rank)
        batch_sampler = TokenBudgetBatchSampler(
            train_ds.lengths,
            max_token_len_per_gpu=cfg.trainer.max_token_len_per_gpu,
            shuffle=True, drop_last=True, seed=cfg.trainer.seed,
            num_replicas=world_size, rank=rank,
        )
        if is_main:
            print(f"collator: packing + 动态 batch(remove_padding=true, "
                  f"max_token_len_per_gpu={cfg.trainer.max_token_len_per_gpu}, "
                  f"world_size={world_size}, 估算每 rank {len(batch_sampler)} 个 micro-batch/epoch)")
        train_dl = DataLoader(
            train_ds, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=2,
        )
    else:
        collate_fn = make_collator(tokenizer.pad_token_id or tokenizer.eos_token_id)
        # 【新增·模块7】右填充路径用 torch DistributedSampler 分片(对应 verl _build_dataloader)
        if world_size > 1:
            dist_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True, seed=cfg.trainer.seed,
            )
        else:
            dist_sampler = None
        if is_main:
            print("collator: 右填充 + 固定 micro_batch_size(remove_padding=false, 与之前模块一致)")
        train_dl = DataLoader(
            train_ds, batch_size=cfg.trainer.micro_batch_size, shuffle=(dist_sampler is None),
            sampler=dist_sampler, collate_fn=collate_fn, num_workers=2, drop_last=True,
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
    # 【新增·模块7】只有 rank0 打印(非 rank0 静默, 避免日志重复 world_size 份)
    if is_main:
        print(f"总优化步数: {total_steps} (每 epoch {steps_per_epoch} 步, 等效 batch={equiv_batch_desc})")

    # 4. 训练循环(与基础版完全一致) —— 【模块7】无 DDP/FSDP: 各 rank 各自 backward+step,
    #    梯度不跨卡同步(中间态, 见顶部说明); 只有 rank0 写日志。
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(cfg.trainer.epochs):
        # 【新增·模块5】动态 batch 模式下, 每 epoch 换打乱种子(对应 verl DataLoader.set_epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        # 【新增·模块7】右填充路径用 DistributedSampler 时, 每 epoch 要 set_epoch
        #   (否则各 epoch 用同一打乱顺序, 对应 verl DataLoader.set_epoch 的另一半)
        if world_size > 1 and not cfg.trainer.remove_padding:
            train_dl.sampler.set_epoch(epoch)
        for micro_idx, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.trainer.bf16):
                # 【新增·模块4】packing 模式下 attention_mask 是 4D 块对角因果掩码(不再是 2D padding mask),
                # 并多传 position_ids(每样本内从 0 重置)。右填充模式时没有 position_ids, 走原路径。
                fwd_kwargs = dict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                if "position_ids" in batch:   # packing collator 才会产出 position_ids
                    fwd_kwargs["position_ids"] = batch["position_ids"]
                out = model(**fwd_kwargs)   # 【模块6】不传 labels -> out.loss 为 None, 只取 out.logits 手算
                # 【新增·模块6】手算 sft_loss: logprobs_from_logits(shift) + loss_mask 加权平均
                # 等价于之前 out.loss(HF 内部 shift+CrossEntropy+ignore_index=-100), 但全透明。
                raw_loss = sft_loss(out.logits, batch["input_ids"], batch["loss_mask"])
                loss = raw_loss / cfg.trainer.grad_accum_steps   # 梯度累积: 每步只回传 1/accum
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
                        "train/loss": raw_loss.item(),   # 【模块6】手算 sft_loss 的值(未除 accum)
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                    }
                    # 【新增·模块7】只有 rank0 写日志(tracker 在非 rank0 上为 None, 直接调会 AttributeError)。
                    #   对应 verl sft_trainer 的 `if self.rank == 0: tracking.log(...)`
                    if is_main and tracker is not None:
                        tracker.log(metrics, step=global_step)

    # 5. 保存模型
    # 【新增·模块1】save_model 开关: 验证配置时可设 false 省磁盘
    # 【新增·模块7】只有 rank0 存模型 + 打印 —— 非 rank0 存盘会各 rank 互相覆盖写冲突
    #   (且本中间态各 rank 模型已发散, 存非 rank0 的没意义)。对应 verl `if self.rank == 0: save_pretrained`
    if is_main and cfg.trainer.save_model:
        print(f"保存模型到 {cfg.trainer.output_dir}")
        model.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(cfg.trainer.output_dir)
    elif is_main:
        print("save_model=false, 跳过保存(仅验证配置/流程)")
    # 【新增·模块2】收尾追踪器(关闭 jsonl 文件等), 对应 verl tracking.finish()
    # 【新增·模块7】只有 rank0 有 tracker 才 finish; 非 rank0 的 tracker=None 跳过
    if is_main and tracker is not None:
        tracker.finish()
    if is_main:
        print("训练完成。")
    # 【新增·模块7】销毁进程组(所有 rank 都调, 对应 verl run_sft 末尾 destroy_global_process_group)。
    #   不调的话 NCCL communicator 不释放, torchrun 进程挂住不退出。
    destroy_global_process_group()


if __name__ == "__main__":
    main()
