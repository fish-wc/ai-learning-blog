# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
引擎抽象层 —— verl/learning/sft/code/11_engine_abstraction/engine.py

==================== 【新增·模块10】BaseEngine + FSDPEngine + AutoModelEngine + EngineRegistry ====================
在模块9(10_fsdp)基础上继续改动(**纯结构重构, 数值语义不变**):
  - 模块9 把"FSDP 包裹 + 前向 + 反向 + no_sync 梯度累积 + *dp_size 全局 loss + clip + step"全堆在 train.py 的 main() 主循环里。
  - 本模块把这些"机制"抽进 BaseEngine / 具体引擎, train.py 只剩 init 进程组 + worker.fit()。

【这一层管什么 / 不管什么】
  管(机制, loss-agnostic): model/optimizer/lr_scheduler 的构建、FSDP 包裹、forward_backward_batch 的
    微批 + no_sync 循环、global_num_tokens 的跨卡 all_reduce(SUM)、grad_norm 全局归约 + 裁剪、optimizer.step。
    引擎**不 import sft_loss** —— loss_fn 由 worker 注入, forward_step 只负责把 logits 喂给注入的 loss_fn。
  不管(留给 worker): loss 的具体公式(由 worker 注入 sft_loss_dp)、外层 epoch/micro 调度、指标后处理与日志、
    min-count 防死锁(决定跑几个 optimizer step)、模型存盘。这就是 verl 的分工: 引擎管"怎么算", worker 管"算什么 + 怎么汇报"。

【这块的意义】
verl 靠这层抽象让同一套 TrainingWorker 能挂 fsdp / megatron / automodel 不同后端 —— 后续学 megatron/torch.compile
时只需新增一个 Engine 子类(填 forward_backward_batch / optimizer_step / initialize 三个机制方法), worker 与主循环不动。
本模块用 fsdp + automodel 两个后端演示这个抽象: 同一个 worker, 改一行 cfg.trainer.engine_backend 就换引擎。

【和 verl 原版的对应】
  BaseEngine          ~ verl/workers/engine/base.py:BaseEngine(train_batch 骨架 :113, optimizer_zero_grad/step,
                        lr_scheduler_step/forward_backward_batch/forward_step/initialize/train_mode 接口)
  FSDPEngine          ~ verl/workers/engine/fsdp/transformer_impl.py:FSDPEngineWithLMHead
                        (_build_fsdp_module :106 / forward_backward_batch :728 / _gradient_sync_context :706 /
                         optimizer_step :792 用 module.clip_grad_norm_ 全局归约)
  AutoModelEngine     ~ verl/workers/engine/automodel/transformer_impl.py:AutomodelEngineWithLMHead
                        (同接口的非 FSDP 后端 —— 证明 BaseEngine 抽象能挂多后端)
  EngineRegistry      ~ verl/workers/engine/base.py:339 EngineRegistry(register+new+get_engine_cls, 按 backend 选引擎)
  loss 注入点         ~ verl forward_step 里 loss_function(model_output=raw_output, data=micro_batch, dp_group=...)
                        (fsdp/transformer_impl.py:1597) —— 这里简化成 loss_fn(logits, input_ids, loss_mask,
                        global_num_tokens, dp_size), 引擎只把 logits + DP 上下文喂给注入的 loss_fn, 不认识 loss 长啥样。

【简化项】(相对 verl)
  (a) verl EngineRegistry 按 (model_type, backend, device, vendor) 四维查表 + Ray dispatch; 这里压成单维 backend dict + 直接 Python 调用(单机 torchrun, 无 Ray/DeviceMesh)。
  (b) verl 的 offload/train_mode/eval_mode/checkpoint/save_checkpoint/get_per_tensor_param_shard 等大量方法省略或留占位, 只保留 SFT 训练必需子集。
  (c) MFU 不在引擎算(留到 worker._postprocess, 引擎只暴露 n_full_params)。
"""
import functools
import math
import os
from abc import abstractmethod
from contextlib import nullcontext, contextmanager

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer   # auto_wrap 的目标层(沿用模块9 已验证导入路径)


# 【新增·模块10】BaseEngine: 机制层基类。对应 verl/workers/engine/base.py:BaseEngine。
#   持有 model/optimizer/lr_scheduler/n_full_params, 定义 train_batch 骨架; 子类填 initialize /
#   forward_step / optimizer_step / _gradient_sync_context 四个机制方法。
class BaseEngine:
    def __init__(self, cfg, local_rank, world_size):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device("cuda", local_rank)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.n_full_params = 0   # 全量参数数(供 worker 算 MFU; FSDP 下 model.parameters() 只返回 shard, 故单独存全量)

    # ---- verl BaseEngine 同名接口(只保留 SFT 必需子集) ----
    def initialize(self):
        """加载模型 + 梯度检查点 + 后端包裹(FSDP/DDP/裸模型) + 优化器。子类实现。"""
        raise NotImplementedError

    @contextmanager
    def train_mode(self):
        """对应 verl BaseEngine.train_mode: 进 train 模式, 退出时不动(本模块无 eval, 全程 train)。"""
        self.model.train()
        yield

    def optimizer_zero_grad(self):
        """对应 verl BaseEngine.optimizer_zero_grad。"""
        self.optimizer.zero_grad()

    def lr_scheduler_step(self):
        """对应 verl BaseEngine.lr_scheduler_step; 返回当前 lr(供日志)。"""
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]

    def build_scheduler(self, total_steps):
        """构建 cosine 学习率调度器。单独拎出来(不在 initialize 里)是因为 total_steps 依赖
        dataloader 长度(数据侧), 由 worker 先建好 dataloader 算出 total_steps 再注入引擎 ——
        对应 verl 里 scheduler 的 total_steps 也来自 trainer 算好的 num_training_steps。"""
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.optim.warmup_steps,
            num_training_steps=total_steps,
        )

    # ---- 核心: train_batch 骨架(对应 verl BaseEngine.train_batch, base.py:113-132) ----
    #   固定模板: zero_grad -> forward_backward_batch -> optimizer_step -> 注入 grad_norm。
    #   这是"一次 optimizer step"的骨架, 子类只填 forward_backward_batch / optimizer_step 的机制。
    def train_batch(self, micro_batches, loss_fn):
        self.optimizer_zero_grad()
        out = self.forward_backward_batch(micro_batches, loss_fn)
        grad_norm = self.optimizer_step()           # clip + step, 返回全局 grad_norm
        out["metrics"]["grad_norm"] = grad_norm
        return out

    def forward_backward_batch(self, micro_batches, loss_fn):
        """微批循环 + no_sync 延迟同步 + 全局 batch_num_tokens all_reduce。对应 verl
        FSDPEngine.forward_backward_batch (fsdp/transformer_impl.py:728-781) 的**微批循环部分**。

        【与 verl 的一个结构差异(须留意)】verl 的 forward_backward_batch 收**一个** mini-batch(data TensorDict),
          在引擎**内部**调 prepare_micro_batches(:741) 把它切成 micro_batch 再循环(:753) —— 切分是引擎的职责。
          本模块把切分**上提到 worker**: worker.fit 用 itertools.islice 预取 grad_accum 个 micro 一组(worker.py,
          注释里标了"对应 verl prepare_micro_batches"), 再把**已切好**的 micro_batches 传给 engine.train_batch。
          故本引擎的 forward_backward_batch 只对应 verl :728-781 里 prepare_micro_batches **之后**的循环段
          (all_reduce batch_num_tokens + 微批 no_sync + backward), 不含切分。这样切分逻辑留在 worker,
          引擎更纯(只算不切); 代价是边界与 verl 不完全对齐 —— 读 verl 源码时注意 prepare_micro_batches 在引擎内。

        【关键·数值等价模块9】这里**完全复刻模块9 train.py 的 micro 循环语义**:
          1. 进 micro 循环前 all_reduce(SUM) 算【全局】batch_num_tokens(float32, 避 bf16 舍入坑);
          2. 每个 micro 调 sft_loss_dp = local_loss_sum/global_num_tokens*dp_size(由注入的 loss_fn 算),
             **不除 grad_accum**(分母 global_num_tokens 已是整组 optimizer-step 的全局 token 总数, 天然归一化);
          3. 非 last micro 进 model.no_sync() 延迟跨卡梯度同步, 最后一个 micro 的 backward 触发一次 reduce-scatter。
        故 loss 轨迹与模块9 逐位一致(纯重构, 计算语义未改)。
        """
        dp_size = self.world_size   # *dp_size 技巧: 抵消 DDP/FSDP 反向时对梯度的 /world_size 平均(实测两者都平均, 见模块9 README)
        # (1) 全局 batch_num_tokens: 跨 rank SUM(必须 float32! bf16 下 357->356 的坑, 见模块6)。
        #   对应 verl forward_backward_batch: batch_num_tokens = data['loss_mask'].sum();
        #   torch.distributed.all_reduce(batch_num_tokens, op=SUM, group=dp_group)。
        #   取 shift 后的 trained token 数(loss_mask[:, 1:]), 与 sft_loss_dp 的分母口径一致。
        #   micro_batches 此时还在 CPU(DataLoader 产出), 在 CPU 上 sum 再搬卡, 与模块9 line 379 一致。
        local_n = sum(int(mb["loss_mask"][:, 1:].sum().item()) for mb in micro_batches)
        g_n = torch.tensor([float(local_n)], dtype=torch.float32, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(g_n, op=dist.ReduceOp.SUM)
        global_num_tokens = g_n.item()

        # (2) micro 循环: no_sync 延迟同步 + 注入 loss_fn(引擎不认识 loss, 只把 logits 喂给它)
        n = len(micro_batches)
        local_loss_sum_log = 0.0   # 累积本 rank 这组 micro 的 -sum(logp*mask)(detach 值), 供 worker 算全局 loss 日志
        for i, mb in enumerate(micro_batches):
            is_last = (i == n - 1)
            # 非 last micro 用 _gradient_sync_context(no_sync) 延迟 reduce-scatter = 梯度累积; last 正常同步
            sync_ctx = nullcontext() if is_last else self._gradient_sync_context(False)
            with sync_ctx:
                loss, lsum_val = self.forward_step(mb, loss_fn, global_num_tokens, dp_size)
                loss.backward()   # 不除 grad_accum —— sft_loss_dp 的 global_num_tokens 分母已归一化(见 loss.py 注释)
            local_loss_sum_log += lsum_val
        return {"loss": local_loss_sum_log, "num_tokens": global_num_tokens, "metrics": {}}

    @abstractmethod
    def forward_step(self, micro_batch, loss_fn, global_num_tokens, dp_size):
        """子类实现: 出 logits -> 调注入的 loss_fn(logits, input_ids, loss_mask, global_num_tokens, dp_size)。
        返回 (loss_tensor[带梯度], local_loss_sum_float[detach, 供日志])。"""
        raise NotImplementedError

    def _gradient_sync_context(self, is_last):
        """非 last micro 的梯度同步上下文(默认不延迟)。对应 verl _gradient_sync_context (:706)。"""
        return nullcontext()

    @abstractmethod
    def optimizer_step(self):
        """clip_grad + optimizer.step, 返回全局 grad_norm。子类实现。对应 verl BaseEngine.optimizer_step。"""
        raise NotImplementedError


# 【新增·模块10】_MixinLMHead: forward_step 共用实现(LM head 后端的 logits -> loss_fn)。
#   对应 verl FSDPEngineWithLMHead.forward_step (fsdp/transformer_impl.py:1570):
#     raw_output = self.module(**model_inputs, use_cache=False)
#     loss, metrics = loss_function(model_output=raw_output, data=micro_batch, dp_group=...)
#   这里把 verl 的 loss_function(model_output=, data=, dp_group=) 简化成
#     loss_fn(logits, input_ids, loss_mask, global_num_tokens, dp_size) -> (loss_tensor, local_loss_sum)
#   引擎只负责"前向拿 logits + 把 logits 和 DP 上下文喂给注入的 loss_fn", 不认识 loss 公式(loss 注入)。
class _MixinLMHead:
    def forward_step(self, micro_batch, loss_fn, global_num_tokens, dp_size):
        mb = {k: v.to(self.device) for k, v in micro_batch.items()}
        # packing 模式下 attention_mask 是 4D 块对角因果掩码, 并多传 position_ids(每样本内从 0 重置);
        # 右填充模式没有 position_ids, 走原路径。沿用模块4/9 的 forward 约定。
        fwd_kwargs = dict(
            input_ids=mb["input_ids"],
            attention_mask=mb["attention_mask"],
            use_cache=False,   # 不传 labels -> out.loss 为 None, 只取 out.logits 手算(模块6 约定)
        )
        if "position_ids" in mb:   # packing collator 才产出 position_ids
            fwd_kwargs["position_ids"] = mb["position_ids"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.trainer.bf16):
            out = self.model(**fwd_kwargs)   # 必须走包装层(model 而非 model.module): DDP/FSDP.forward 登记梯度同步准备态
            loss, lsum = loss_fn(out.logits, mb["input_ids"], mb["loss_mask"], global_num_tokens, dp_size)
        return loss, lsum.item()   # lsum 已 detach, 不进反传图; backward 在 forward_backward_batch 里(outside autocast, 同模块9)


# 【新增·模块10】FSDPEngine: FSDP 后端, 复用模块9 的 FSDP 分片设置。对应 verl FSDPEngineWithLMHead。
class FSDPEngine(_MixinLMHead, BaseEngine):
    def initialize(self):
        cfg = self.cfg
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "fp16": torch.float16}[cfg.model.dtype]
        if self.local_rank == 0:
            print(f"加载模型: {cfg.model.path}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code,
        ).to(self.device)
        if self.local_rank == 0:
            print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        # 【模块3 沿用】梯度检查点(必须在 FSDP 包裹之前开, 同模块8/9: 检查点配套)
        if cfg.model.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            if self.local_rank == 0:
                print("已开启梯度检查点(用重算换激活显存)")

        # 【模块9 沿用】FSDP 包裹: FULL_SHARD + bf16 混合精度 + transformer_auto_wrap(Qwen2DecoderLayer) +
        #   use_orig_params + sync_module_states + forward_prefetch。对应 verl _build_fsdp_module。
        full_params = sum(p.numel() for p in model.parameters())   # wrap 前全量(供分片证据对比 + MFU)
        # (1) auto_wrap_policy: verl 用 transformer_auto_wrap_policy + _no_split_modules(Qwen2DecoderLayer)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2DecoderLayer},
        )
        # (2) MixedPrecision: bf16 存/算, reduce 用 fp32(数值稳)
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        # (3) sharding_strategy: reshard_after_forward=True -> FULL_SHARD(ZeRO-3); False -> SHARD_GRAD_OP(ZeRO-2)
        sharding_strategy = (ShardingStrategy.FULL_SHARD if cfg.fsdp.reshard_after_forward
                             else ShardingStrategy.SHARD_GRAD_OP)
        # (4) CPUOffload: verl actor 训练强制关(param_offload 配合梯度累积会算错, transformer_impl.py:434)
        cpu_offload = CPUOffload(offload_params=True) if cfg.fsdp.param_offload else None
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            sync_module_states=cfg.fsdp.sync_module_states and self.world_size > 1,   # 单卡无需广播(避免 sync warning); 多卡 rank0 权重广播给其余 rank
            forward_prefetch=cfg.fsdp.forward_prefetch,
            use_orig_params=cfg.fsdp.use_orig_params,          # 保留原参数名, clip/save 更友好
            cpu_offload=cpu_offload,
        )
        # 分片证据: FSDP 下 model.parameters() 是各 rank 的 flat shard, 约为总量/world_size
        shard_params = sum(p.numel() for p in model.parameters())
        if self.local_rank == 0:
            print(f"已用 FSDP 包装模型(dp_size={self.world_size}, strategy={sharding_strategy.value}); "
                  f"全量参数 {full_params/1e6:.1f}M, 各 rank shard {shard_params/1e6:.1f}M "
                  f"(期望 ~{full_params/self.world_size/1e6:.0f}M = 全量/world_size)")
        self.n_full_params = full_params   # 存全量供 worker 算 MFU(FSDP 下 model.parameters() 只返 shard)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
            betas=tuple(cfg.optim.betas),
        )
        self.model.train()

    def _gradient_sync_context(self, is_last):
        # verl _gradient_sync_context (fsdp/transformer_impl.py:706): 非 last micro 用 module.no_sync() 延迟 reduce-scatter。
        #   与模块9 的 maybe_no_sync 等价: is_last 或模型无 no_sync 时返回 nullcontext。
        if is_last or not hasattr(self.model, "no_sync"):
            return nullcontext()
        return self.model.no_sync()

    def optimizer_step(self):
        # 【模块9 沿用】FSDP 下各 rank 只持梯度 shard, torch.nn.utils.clip_grad_norm_ 返回【局部 shard 范数】
        #   (实测 = 全局/√world, 见 /tmp/fsdp_reduce_probe.py), 需手动 all_reduce(SUM of local²) sqrt 得全局范数 + 全局裁剪。
        #   对应 verl worker 层对 grad_norm 的 all_reduce(verl FSDPEngine 用 module.clip_grad_norm_ 自带全局归约, 等价)。
        #   这里沿用模块9 已验证的 fsdp_grad_norm_and_clip, 保证与模块9 数值逐位一致。
        grad_norm = self._fsdp_grad_norm_and_clip(self.cfg.optim.max_grad_norm)
        self.optimizer.step()
        return grad_norm

    def _fsdp_grad_norm_and_clip(self, max_norm):
        """手动算全局 grad_norm + 全局裁剪(模块9 的 fsdp_grad_norm_and_clip 原样搬进引擎)。"""
        local_sq = torch.zeros(1, device=self.device, dtype=torch.float32)
        for p in self.model.parameters():
            if p.grad is not None:
                local_sq += p.grad.detach().float().pow(2).sum()
        if self.world_size > 1:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        global_norm = local_sq.sqrt().item()
        # 全局裁剪: 所有 rank 用同一 scale 缩放自己的 shard(各 shard 同比例 -> 裁剪后全局范数 == max_norm)
        clip_coef = min(1.0, max_norm / (global_norm + 1e-6))
        if clip_coef < 1.0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(clip_coef)
        return global_norm


# 【新增·模块10】AutoModelEngine: 非 FSDP 后端(单卡/对照)。对应 verl AutomodelEngineWithLMHead
#   (automodel/transformer_impl.py) —— 同接口的另一后端, 证明 BaseEngine 抽象能挂多后端。
#   注意: 此后端**不包 DDP/FSDP**, 多卡时会各自为政(等价模块7 的中间态), 仅作单卡 smoke / 对照后端用。
#   真要数据并行请用 fsdp 后端(FSDP 自带 DP 梯度同步, subsume DDP)。
class AutoModelEngine(_MixinLMHead, BaseEngine):
    def initialize(self):
        cfg = self.cfg
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "fp16": torch.float16}[cfg.model.dtype]
        if self.local_rank == 0:
            print(f"加载模型(automodel 后端, 不分片): {cfg.model.path}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code,
        ).to(self.device)
        if self.local_rank == 0:
            print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        if cfg.model.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        self.n_full_params = sum(p.numel() for p in model.parameters())
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
            betas=tuple(cfg.optim.betas),
        )
        self.model.train()

    def _gradient_sync_context(self, is_last):
        # 裸模型无 DDP/FSDP, 无跨 rank 同步可延迟; 梯度累积靠纯梯度求和(单卡时正确)。
        return nullcontext()

    def optimizer_step(self):
        # verl AutomodelEngine.optimizer_step: 普通 clip_grad_norm_(单卡/DDP 同步后梯度, 直接得全局范数)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.optim.max_grad_norm,
        ).item()
        self.optimizer.step()
        return grad_norm


# 【新增·模块10】EngineRegistry: 极简版, 对应 verl engine/base.py:339 的 EngineRegistry。
#   verl 按 (model_type, backend, device, vendor) 四维查表 + Ray dispatch; 这里压成单维 {backend: cls} dict。
#   register(backend) 装饰器注册; new(backend, *a, **kw) 按 backend 查表实例化。
class EngineRegistry:
    _registry = {}

    @classmethod
    def register(cls, backend):
        def deco(engine_cls):
            cls._registry[backend] = engine_cls
            return engine_cls
        return deco

    @classmethod
    def new(cls, backend, *a, **kw):
        assert backend in cls._registry, (
            f"unknown engine backend {backend!r} (registered: {list(cls._registry)})"
        )
        return cls._registry[backend](*a, **kw)

    @classmethod
    def get_engine_cls(cls, backend):
        return cls._registry.get(backend)


EngineRegistry.register("fsdp")(FSDPEngine)
EngineRegistry.register("automodel")(AutoModelEngine)
