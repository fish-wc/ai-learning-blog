# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
结构化配置 + 日志追踪 + 梯度检查点 + remove_padding + 动态 batch + 手算 loss + 分布式进程组 + 数据并行(DDP) + FSDP 模型分片 版 SFT 训练
—— verl/learning/sft/code/10_fsdp/train.py

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

==================== 【新增·模块8】数据并行 (DP): DDP + no_sync 梯度累积 + *dp_size 全局 loss ====================
在模块7 基础上继续改动(数据分片已由模块7 的 TokenBudgetBatchSampler 完成, 本模块只动"模型/梯度"侧):
  - 模型用 DDP 包装(world_size>1 时): 反向时自动 all-reduce 梯度并 /world_size 平均(对应 verl 引擎 DDP/FSDP 包装)。
  - 梯度累积用 no_sync: 非最后一个 micro 进 model.no_sync() 关闭跨 rank 同步, 只在最后一个 micro 的
    backward 触发一次 all-reduce(对应 verl _gradient_sync_context / use_no_sync_for_gradient_accumulation=True)。
  - *dp_size 全局 loss 技巧(对应 verl/workers/utils/losses.py:49): 每个 optimizer-step 先 all_reduce(SUM) 算
    【全局】batch_num_tokens, 再用 sft_loss_dp = local_loss_sum/global_num_tokens*dp_size。*dp_size 抵消 DDP
    的 /world_size 平均 -> 等效梯度 = 真全局 per-token 均值 loss 的梯度(解决 mean-of-means 偏差)。
  - 训练循环改成"按 optimizer-step 预取 grad_accum 个 micro 一组"(对应 verl forward_backward_batch +
    prepare_micro_batches), 因为全局 batch_num_tokens 要在进 micro 循环前 all_reduce 算好。
  - 日志的全局 loss: all_reduce(local_loss_sum, SUM)/global_num_tokens = 全局 per-token 均值(仅 rank0 记)。
  - grad_norm 不再单独 all_reduce: DDP 反向时已同步梯度, clip_grad_norm_ 算的是同步后的全局一致范数。
  - 存模型只在 rank0 且走 model.module(DDP 包装后底层模型在 .module)。
意义: 多卡真正"并行"——每卡见不重叠数据, 梯度跨卡同步平均, 等效于单卡大 batch 训练; *dp_size 让 loss
  数值与梯度在 token 数不齐时仍精确(模块7 的"各自为政发散"中间态在此被解决)。
对应 verl: sft_trainer.py 的 DistributedSampler + engine forward_backward_batch(all_reduce(batch_num_tokens))
  + losses.py 的 *dp_size + _gradient_sync_context(no_sync) + engine_workers._postprocess_output(all_reduce loss, AVG)。

==================== 【新增·模块9】FSDP 模型分片 (FullyShardedDataParallel) ====================
在模块8 基础上继续改动(*dp_size / no_sync / 全局 loss / min-count 防死锁 全沿用, 只换"模型怎么包"):
  - 模型包装从 DDP 换成 FSDP(auto_wrap_policy={Qwen2DecoderLayer} + MixedPrecision(bf16/fp32) +
    FULL_SHARD + sync_module_states + use_orig_params + forward_prefetch)。对应 verl _build_fsdp_module。
  - FSDP 把参数/梯度/优化器状态分片到各卡: 前向 all-gather 拼回, 反向 reduce-scatter 分走 ->
    单卡放不下的大模型能跑(DDP 只是整模型复制不省显存)。
  - no_sync 梯度累积沿用模块8 的 maybe_no_sync(FSDP 也有 no_sync, 对应 verl _gradient_sync_context)。
  - *dp_size 技巧沿用 sft_loss_dp —— 经实测验证 FSDP 与 DDP 一样对梯度做 /world_size 平均(/tmp/fsdp_reduce_probe.py:
    两 rank 不同输入, FSDP 后各 rank 权重完全一致 == 两单进程 delta 的平均), 故 *dp_size 抵消仍正确。
  - grad_norm: FSDP 下各 rank 只持梯度 shard, torch.nn.utils.clip_grad_norm_ 返回的是【局部 shard 范数】
    (实测 = 全局/√world), 需手动 all_reduce(SUM of local²) 再 sqrt 才是全局范数(对应 verl worker 的 grad_norm all_reduce)。
  - 存模型: FSDP 要 FSDP.summon_full_params 把分片拼回才能 save_pretrained(verify save_model=false 规避)。
意义: 把"模型整份复制(DDP)"升级为"参数/梯度/优化器状态分片(FSDP)", 显存占用从 O(模型) 降到 O(模型/world_size),
  为单卡放不下的大模型训练铺路。对应 verl/workers/engine/fsdp/transformer_impl.py。
"""
import functools
import itertools
import math
import os
import sys
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# 【新增·模块9】FSDP 相关导入 —— 替换模块8 的 DDP(DDP 只复制模型不省显存; FSDP 分片省显存)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer   # 【新增·模块9】auto_wrap 的目标层

# 【新增·模块1】引入配置系统
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SFTConfig, load_config, config_to_yaml_str
# 【新增·模块2】引入追踪系统
from tracking import Tracking
from dataset import SFTDataset, make_collator, make_packing_collator, TokenBudgetBatchSampler   # 【新增·模块4/5】多导入 make_packing_collator, TokenBudgetBatchSampler
# 【新增·模块6】引入手算 loss: logprobs_from_logits + sft_loss
# 【新增·模块8】多导入 sft_loss_dp(*dp_size 技巧的数据并行版 loss)
from loss import sft_loss, sft_loss_dp
# 【新增·模块7】引入分布式进程组初始化
from dist import initialize_global_process_group, destroy_global_process_group, is_rank0


# 【新增·模块8】DDP 梯度累积辅助: 非最后一个 micro 用 model.no_sync() 关闭跨 rank 梯度 all-reduce,
#   只在最后一个 micro 的 backward 触发一次同步+平均。对应 verl:
#   verl/workers/engine/fsdp/transformer_impl.py:706-724 _gradient_sync_context
#   (use_no_sync_for_gradient_accumulation=True, is_last_micro_batch ? 正常 : module.no_sync())。
#   非 DDP 模型(单进程)或最后一个 micro 都返回 nullcontext(单进程无需同步, 最后一步要同步)。
def maybe_no_sync(model, is_last_micro):
    if is_last_micro or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


# 【新增·模块9】FSDP 全局 grad_norm + 全局裁剪。
#   FSDP 下各 rank 只持梯度 shard, torch.nn.utils.clip_grad_norm_ 返回的是【局部 shard 范数】
#   (实测 = 全局/√world, 见 /tmp/fsdp_reduce_probe.py), 不能直接当全局范数用。
#   正确做法: 各 rank 算本 shard 的平方和, all_reduce(SUM) 得全局平方和, sqrt 得全局范数,
#   再按 max_norm 全局统一缩放。对应 verl worker 层对 grad_norm 的 all_reduce。
def fsdp_grad_norm_and_clip(model, max_norm, world_size, device):
    local_sq = torch.zeros(1, device=device, dtype=torch.float32)
    for p in model.parameters():
        if p.grad is not None:
            local_sq += p.grad.detach().float().pow(2).sum()
    if world_size > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    global_norm = local_sq.sqrt().item()
    # 全局裁剪: 所有 rank 用同一 scale 缩放自己的 shard(各 shard 同比例 -> 裁剪后全局范数 == max_norm)
    clip_coef = min(1.0, max_norm / (global_norm + 1e-6))
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
    return global_norm


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

    # 【新增·模块9】模型包装: cfg.fsdp.enabled 时用 FSDP 取代模块8 的 DDP。
    #   必须在 gradient_checkpointing_enable + enable_input_require_grads 之后包(同模块8: 检查点配套)。
    #   对应 verl/workers/engine/fsdp/transformer_impl.py:_build_fsdp_module。
    full_params = sum(p.numel() for p in model.parameters())   # wrap 前的全量(供分片证据对比)
    if world_size > 1 and cfg.fsdp.enabled:
        # (1) auto_wrap_policy: verl 用 transformer_auto_wrap_policy + 模型 _no_split_modules(Qwen2DecoderLayer),
        #     见 verl/utils/fsdp_utils.py:get_fsdp_wrap_policy。最小版直接用 Qwen2DecoderLayer 类。
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2DecoderLayer},
        )
        # (2) MixedPrecision: bf16 训练。对应 verl _build_fsdp_module 的 MixedPrecision(param/reduce/buffer_dtype)
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,   # 参数用 bf16 存/算(显存减半)
            reduce_dtype=torch.float32,   # 梯度 reduce 用 fp32(数值稳, 对应 verl reduce_dtype=fp32)
            buffer_dtype=torch.bfloat16,
        )
        # (3) sharding_strategy: reshard_after_forward=True -> FULL_SHARD(ZeRO-3); False -> SHARD_GRAD_OP(ZeRO-2)
        sharding_strategy = (ShardingStrategy.FULL_SHARD if cfg.fsdp.reshard_after_forward
                             else ShardingStrategy.SHARD_GRAD_OP)
        # (4) CPUOffload: verl 在 actor 训练强制关(param_offload 配合梯度累积会算错, transformer_impl.py:434)
        cpu_offload = CPUOffload(offload_params=True) if cfg.fsdp.param_offload else None
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),            # 模型已在 cuda; device_id 负责 sharding 初始化
            sync_module_states=cfg.fsdp.sync_module_states,   # rank0 权重广播给其余 rank
            forward_prefetch=cfg.fsdp.forward_prefetch,
            use_orig_params=cfg.fsdp.use_orig_params,          # 保留原参数名, clip/save 更友好
            cpu_offload=cpu_offload,
        )
        # 【新增·模块9】分片证据: FSDP 下 model.parameters() 是各 rank 的 flat shard, 约为总量/world_size
        shard_params = sum(p.numel() for p in model.parameters())
        if is_main:
            print(f"已用 FSDP 包装模型(dp_size={world_size}, strategy={sharding_strategy.value}); "
                  f"全量参数 {full_params/1e6:.1f}M, 各 rank shard {shard_params/1e6:.1f}M "
                  f"(期望 ~{full_params/world_size/1e6:.0f}M = 全量/world_size)")
    elif world_size > 1:
        # 【模块8 路径保留作对照】cfg.fsdp.enabled=false 时退回 DDP
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=cfg.trainer.find_unused_parameters,
        )
        if is_main:
            print(f"已用 DDP 包装模型(dp_size={world_size}, find_unused_parameters={cfg.trainer.find_unused_parameters}) "
                  f"[fsdp.enabled=false, 模块8 对照]")
    dp_size = world_size   # = verl 的 data_parallel_size; *dp_size 抵消 DDP/FSDP 的 /world_size 平均(实测两者都平均, 见 README)

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

    # 4. 训练循环 —— 【新增·模块8】按 optimizer-step 预取 grad_accum 个 micro 一组(对应 verl
    #    forward_backward_batch + prepare_micro_batches): 先 all_reduce 算【全局】batch_num_tokens,
    #    再 micro 循环用 sft_loss_dp(*dp_size 技巧) + no_sync 梯度累积, 最后一步 backward 触发 DDP 同步。
    #    (模块7 是"逐 micro 流式 backward + loss/accum", 本模块改成"按组预取"是因为全局 batch_num_tokens
    #     要在进 micro 循环前 all_reduce 算好。)
    global_step = 0
    for epoch in range(cfg.trainer.epochs):
        # 【新增·模块5】动态 batch 模式下, 每 epoch 换打乱种子(对应 verl DataLoader.set_epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        # 【新增·模块7】右填充路径用 DistributedSampler 时, 每 epoch 要 set_epoch
        #   (否则各 epoch 用同一打乱顺序, 对应 verl DataLoader.set_epoch 的另一半)
        if world_size > 1 and not cfg.trainer.remove_padding:
            train_dl.sampler.set_epoch(epoch)
        # 【新增·模块8·修复DP死锁】各 rank 的 micro 数可能不同(动态 batch + drop_last + 样本长度不齐),
        #   若两 rank 的 optimizer-step 数不一致, all_reduce 调用次数不匹配 -> NCCL 死锁(本模块实测踩到)。
        #   修法: 各 rank 数自己本 epoch 的 micro 数, all_reduce(MIN) 取公共 micro 数, 两 rank 都只跑
        #   common_opt_steps = common_micro // grad_accum 步(余数丢弃, 每组必满 grad_accum 个 micro)。
        #   对应 verl prepare_micro_batches 的 same_micro_num_in_dp=True(保证各 dp rank 的 micro 数相同)。
        if batch_sampler is not None:
            local_micro_count = sum(1 for _ in batch_sampler)   # 精确数本 rank 本 epoch 的 micro 数
        else:
            local_micro_count = len(train_dl)                     # 右填充: 固定 batch_size, len 精确
        if world_size > 1:
            mc = torch.tensor([float(local_micro_count)], dtype=torch.float32, device=device)
            dist.all_reduce(mc, op=dist.ReduceOp.MIN)
            common_micro = int(mc.item())
        else:
            common_micro = local_micro_count
        common_opt_steps = common_micro // cfg.trainer.grad_accum_steps

        dl_iter = iter(train_dl)
        for _step in range(common_opt_steps):
            # 预取一组 micro(= 一个 optimizer step 要处理的 micro 数), 对应 verl prepare_micro_batches。
            # common_opt_steps 保证本 rank 有足够 micro, 这里 islice 必返回 grad_accum 个(无空组)。
            micro_group = list(itertools.islice(dl_iter, cfg.trainer.grad_accum_steps))

            # 【新增·模块8】全局 batch_num_tokens: 跨 rank SUM(必须 float32, bf16 下 357->356 的坑, 见模块6)。
            #   对应 verl forward_backward_batch: batch_num_tokens = data['loss_mask'].sum();
            #   torch.distributed.all_reduce(batch_num_tokens, op=SUM, group=dp_group)。
            #   取 shift 后的 trained token 数(loss_mask[:, 1:]), 与 sft_loss_dp 的分母口径一致。
            local_n = sum(int(b["loss_mask"][:, 1:].sum().item()) for b in micro_group)
            g_n = torch.tensor([float(local_n)], dtype=torch.float32, device=device)
            if world_size > 1:
                dist.all_reduce(g_n, op=dist.ReduceOp.SUM)
            global_num_tokens = g_n.item()

            optimizer.zero_grad()
            local_loss_sum_log = 0.0   # 累积本 rank 这组 micro 的 -sum(logp*mask), 用于算全局 loss(仅日志)
            for m, micro in enumerate(micro_group):
                micro = {k: v.to(device) for k, v in micro.items()}
                is_last = (m == len(micro_group) - 1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.trainer.bf16):
                    # 【新增·模块4】packing 模式下 attention_mask 是 4D 块对角因果掩码(不再是 2D padding mask),
                    # 并多传 position_ids(每样本内从 0 重置)。右填充模式时没有 position_ids, 走原路径。
                    fwd_kwargs = dict(
                        input_ids=micro["input_ids"],
                        attention_mask=micro["attention_mask"],
                        use_cache=False,
                    )
                    if "position_ids" in micro:   # packing collator 才会产出 position_ids
                        fwd_kwargs["position_ids"] = micro["position_ids"]
                    # 【新增·模块8】forward 必须走 DDP 包装层(model(**kwargs), 不是 model.module)——
                    #   DDP.forward 会登记 grad reduction 准备态, 否则反向钩子不触发、梯度不同步。
                    #   4D 块对角掩码作为 attention_mask 经 DDP.forward 转发到底层模型, 无副作用。
                    out = model(**fwd_kwargs)   # 【模块6】不传 labels -> out.loss 为 None, 只取 out.logits 手算
                    # 【新增·模块8】*dp_size 技巧: loss = local_loss_sum/global_num_tokens*dp_size
                    #   不除 grad_accum(分母 global_num_tokens 已是整组 optimizer-step 的全局 token 总数,
                    #   天然把"一组 micro 求和"归一化掉)。对应 verl/workers/utils/losses.py:49。
                    loss, lsum = sft_loss_dp(
                        out.logits, micro["input_ids"], micro["loss_mask"],
                        global_num_tokens, dp_size,
                    )
                # 【新增·模块8】非最后 micro 进 model.no_sync() 关闭跨 rank 同步; 最后一个 backward 触发
                #   DDP all-reduce(avg) 把【累积】梯度同步+平均。对应 verl _gradient_sync_context。
                with maybe_no_sync(model, is_last):
                    loss.backward()
                local_loss_sum_log += lsum.item()   # lsum 已 detach, 不进反传图

            # 最后 micro 的 backward 已触发 DDP/FSDP 同步, 此刻梯度全局一致(DDP: 全量; FSDP: 分片)
            # 【新增·模块2】grad_norm:
            #   DDP/单进程: clip_grad_norm_ 算的是已同步的全量梯度范数(全局一致)。
            #   【新增·模块9】FSDP: 各 rank 只持梯度 shard, clip_grad_norm_ 返回【局部 shard 范数】(=全局/√world),
            #     需 fsdp_grad_norm_and_clip 手动 all_reduce(SUM of local²) sqrt 得全局范数 + 全局裁剪
            #     (对应 verl worker 层对 grad_norm 的 all_reduce)。
            if isinstance(model, FSDP):
                grad_norm = fsdp_grad_norm_and_clip(
                    model, cfg.optim.max_grad_norm, world_size, device,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.max_grad_norm
                ).item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # 【新增·模块8】全局 loss(仅 rank0 记): all_reduce(local_loss_sum, SUM)/global_num_tokens
            #   = 全局 per-token 均值(对应 verl engine_workers._postprocess_output 的 all_reduce(loss, AVG))。
            #   必须 .item() 后再 all_reduce, 不能让反传图穿过 all_reduce(否则梯度被重复同步)。
            if world_size > 1:
                lsum_t = torch.tensor([local_loss_sum_log], dtype=torch.float32, device=device)
                dist.all_reduce(lsum_t, op=dist.ReduceOp.SUM)
                global_loss = lsum_t.item() / global_num_tokens
            else:
                global_loss = local_loss_sum_log / global_num_tokens

            if global_step % cfg.trainer.log_every == 0 or global_step == 1:
                lr = lr_scheduler.get_last_lr()[0]
                # 【新增·模块2】指标走 Tracking: 同时落 console(打印) + file(jsonl)
                # 对应 verl: tracking.log(data=metrics, step=global_step)
                metrics = {
                    "train/loss": global_loss,        # 【模块8】全局 per-token 均值(不再是单 rank 的 mean)
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,     # 全局一致(DDP 已同步梯度)
                    "train/epoch": epoch,
                    "train/global_tokens": global_num_tokens,   # 【模块8】本 optimizer-step 的全局 trained token 数
                }
                # 【新增·模块7】只有 rank0 写日志(tracker 在非 rank0 上为 None, 直接调会 AttributeError)。
                #   对应 verl sft_trainer 的 `if self.rank == 0: tracking.log(...)`
                if is_main and tracker is not None:
                    tracker.log(metrics, step=global_step)

    # 5. 保存模型
    # 【新增·模块1】save_model 开关: 验证配置时可设 false 省磁盘
    # 【新增·模块7】只有 rank0 存模型 + 打印。对应 verl `if self.rank == 0: save_pretrained`
    # 【新增·模块8】DDP 包装后底层模型在 model.module, 存盘要走它。
    # 【新增·模块9】FSDP 模型存盘要先 summon_full_params 把分片拼回原参数(verl 走 FSDP.state_dict / save);
    #   不 summon 直接 save_pretrained 只会存到 shard/缺参数。verify 设 save_model=false 规避。
    if is_main and cfg.trainer.save_model:
        print(f"保存模型到 {cfg.trainer.output_dir}")
        if isinstance(model, FSDP):
            # FSDP: summon 把所有 nested FSDP unit 的分片拼回全量, 再走底层 model.module 存盘
            with FSDP.summon_full_params(model, writeback=False):
                base_model = model.module if hasattr(model, "module") else model
                base_model.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
        else:
            base_model = model.module if isinstance(model, DDP) else model
            base_model.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
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
