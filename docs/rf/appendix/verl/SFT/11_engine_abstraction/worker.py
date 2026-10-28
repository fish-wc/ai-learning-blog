# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
TrainingWorker —— verl/learning/sft/code/11_engine_abstraction/worker.py

==================== 【新增·模块10】TrainingWorker(外循环 + loss 注入 + 指标后处理) ====================
在模块9(10_fsdp)基础上继续改动(**纯结构重构, 数值语义不变**):
  - 模块9 的训练 for 循环、FSDP clip、all_reduce、Tracking 构造与 log 全堆在 train.py 的 main() 一锅烩。
  - 本模块把它们上提到 worker: train.py 只管 init 进程组 + worker.fit()。

【这一层管什么 / 不管什么】
  管: 持有一个 engine(经 EngineRegistry 选后端建)、注入 loss_fn(sft_loss_dp)、跑外层 epoch/micro/accum 循环、
    min-count 防死锁(决定跑几个 optimizer step)、指标后处理(loss all_reduce + MFU)、rank0 落日志、模型存盘。
  不管(交给 engine): 前向/反向/no_sync/clip/step 的具体机制。
这就是 verl 的 TrainingWorker —— 引擎管"怎么算", worker 管"算什么(loss)+怎么汇报(指标/MFU)+怎么调度(外循环)"。

【和 verl 原版的对应】
  TrainingWorker.__init__     ~ verl/workers/engine_workers.py:167 (EngineRegistry.new 选后端建 engine)
  set_loss_fn / loss 注入     ~ engine_workers.py:200 + sft_trainer.py:163,175
                                (SFTTrainer 里 self.loss_fn=partial(sft_loss); training_client.set_loss_fn(loss_fn=...))
  train_batch 包 engine       ~ engine_workers.py:374 (worker.train_batch 包 engine.train_batch + _postprocess_output)
  _postprocess(all_reduce+MFU)~ engine_workers.py:212 _postprocess_output (:229 loss all_reduce AVG 等价、:260 FlopsCounter MFU)
  fit / 外循环                ~ verl/trainer/sft_trainer.py 的 SFTTrainer.fit 主循环
"""
import itertools
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP   # _save_model 里 isinstance 判定 FSDP 后端
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from config import config_to_yaml_str
from tracking import Tracking
from dataset import SFTDataset, make_collator, make_packing_collator, TokenBudgetBatchSampler
from loss import sft_loss_dp   # 【新增·模块10】注入给 engine 的 loss_fn(*dp_size 技巧的数据并行版)
from engine import EngineRegistry


class TrainingWorker:
    def __init__(self, cfg, local_rank, world_size):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        # 全局 rank(单节点 == local_rank; 多节点 != local_rank)。DataLoader 分片按全局 rank 切(对应 verl dp_rank)。
        self.rank = dist.get_rank() if world_size > 1 else 0
        self.device = torch.device("cuda", local_rank)
        self.is_rank0 = (self.rank == 0)

        # 【新增·模块10】loss 注入: worker 持有 loss_fn, engine loss-agnostic。
        #   对应 verl sft_trainer.py:163 self.loss_fn=partial(sft_loss) + :175 set_loss_fn(loss_fn=self.loss_fn)。
        #   这里注入 sft_loss_dp(模块8/9 的 *dp_size 版), engine.forward_step 调它拿 (loss, local_loss_sum)。
        self.loss_fn = sft_loss_dp

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.path, trust_remote_code=cfg.model.trust_remote_code,
        )
        # 先建 dataloader: scheduler 的 total_steps 依赖 dataloader 长度(数据侧), 算好后再注入 engine。
        self._build_dataloader()

        # 【新增·模块10】经 EngineRegistry 选后端建引擎 —— 对应 verl EngineRegistry.new(model_type, backend=strategy)。
        #   worker 不直接 new 模型, 而是通过 Registry 选后端建引擎; 换 backend 即换引擎, worker 其余代码不动。
        self.engine = EngineRegistry.new(cfg.trainer.engine_backend, cfg, local_rank, world_size)
        self.engine.initialize()                 # 加载模型 + 梯度检查点 + FSDP 包裹 + 优化器
        self.engine.build_scheduler(self.total_steps)   # cosine scheduler(total_steps 由 dataloader 算出)

        self._build_tracker()

    def set_loss_fn(self, fn):
        """对应 verl set_loss_fn (engine_workers.py:200)。"""
        self.loss_fn = fn

    def _build_dataloader(self):
        """沿用模块4-9 的数据管线: packing(TokenBudgetBatchSampler 按 token 预算 + rank 分片) 或 右填充(固定 micro_batch_size)。"""
        cfg = self.cfg
        train_ds = SFTDataset(
            cfg.data.train_file, self.tokenizer,
            max_length=cfg.data.max_length, max_samples=cfg.data.max_train_samples,
        )
        self.batch_sampler = None   # packing 模式下的 TokenBudgetBatchSampler(供 fit 里 min-count 数 micro)
        self.sampler = None         # 右填充 DP 模式下的 DistributedSampler
        if cfg.trainer.remove_padding:
            collate_fn = make_packing_collator()
            # 【模块7 沿用】TokenBudgetBatchSampler 加 num_replicas=world_size, rank=rank 做数据分片
            #   (同 seed + rank 取不同 strided 切片 -> 不重叠; 对应 verl DistributedSampler(num_replicas, rank))
            self.batch_sampler = TokenBudgetBatchSampler(
                train_ds.lengths,
                max_token_len_per_gpu=cfg.trainer.max_token_len_per_gpu,
                shuffle=True, drop_last=True, seed=cfg.trainer.seed,
                num_replicas=self.world_size, rank=self.rank,
            )
            self.train_dl = DataLoader(
                train_ds, batch_sampler=self.batch_sampler, collate_fn=collate_fn, num_workers=2,
            )
            if self.is_rank0:
                print(f"collator: packing + 动态 batch(remove_padding=true, "
                      f"max_token_len_per_gpu={cfg.trainer.max_token_len_per_gpu}, "
                      f"world_size={self.world_size}, 估算每 rank {len(self.batch_sampler)} 个 micro-batch/epoch)")
        else:
            collate_fn = make_collator(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if self.world_size > 1:
                self.sampler = DistributedSampler(
                    train_ds, num_replicas=self.world_size, rank=self.rank,
                    shuffle=True, drop_last=True, seed=cfg.trainer.seed,
                )
            self.train_dl = DataLoader(
                train_ds, batch_size=cfg.trainer.micro_batch_size,
                shuffle=(self.sampler is None), sampler=self.sampler,
                collate_fn=collate_fn, num_workers=2, drop_last=True,
            )
            if self.is_rank0:
                print("collator: 右填充 + 固定 micro_batch_size(remove_padding=false, 与之前模块一致)")
        # total_steps 供 cosine scheduler: 每 epoch 的 optimizer-step 数 * epochs(上限; min-count 可能少跑余数)
        steps_per_epoch = math.ceil(len(self.train_dl) / cfg.trainer.grad_accum_steps)
        self.total_steps = steps_per_epoch * cfg.trainer.epochs
        if self.is_rank0:
            equiv = (f"~{cfg.trainer.max_token_len_per_gpu*cfg.trainer.grad_accum_steps} tokens/step"
                     if cfg.trainer.remove_padding else f"{cfg.trainer.micro_batch_size*cfg.trainer.grad_accum_steps} samples/step")
            print(f"总优化步数: {self.total_steps} (每 epoch {steps_per_epoch} 步, 等效 batch={equiv})")

    def _build_tracker(self):
        """【模块2 沿用】Tracking: console + file jsonl。只在 rank0 建(FileBackend 独占写, 多进程同时写会乱码)。"""
        if not self.is_rank0:
            self.tracker = None
            return
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.tracker = Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.experiment_name,
            default_backend=list(self.cfg.trainer.logger),
            config=config_to_yaml_str(self.cfg),
            output_dir=logs_dir,
        )

    def fit(self):
        """外层 epoch / mini-batch 循环, 调 engine.train_batch。对应 verl SFTTrainer.fit。"""
        cfg = self.cfg
        eng = self.engine
        accum = cfg.trainer.grad_accum_steps
        global_step = 0

        if self.is_rank0:
            shard_p = sum(p.numel() for p in eng.model.parameters()) / 1e6
            print(f"架构: TrainingWorker -> {type(eng).__name__} -> {type(eng.model).__name__} "
                  f"(backend={cfg.trainer.engine_backend}, 全量 {eng.n_full_params/1e6:.1f}M, "
                  f"每 rank 持 {shard_p:.1f}M 参数)")

        with eng.train_mode():   # 全程 train 模式(对应 verl BaseEngine.train_mode)
            for epoch in range(cfg.trainer.epochs):
                # 【模块5 沿用】每 epoch 换打乱种子(packing 的 batch_sampler / 右填充的 DistributedSampler 都要 set_epoch)
                if self.batch_sampler is not None:
                    self.batch_sampler.set_epoch(epoch)
                elif self.world_size > 1 and self.sampler is not None:
                    self.sampler.set_epoch(epoch)

                # 【模块8 沿用·修复DP死锁】各 rank 的 micro 数可能不同(动态 batch + drop_last + 样本长度不齐),
                #   若两 rank 的 optimizer-step 数不一致, engine.forward_backward_batch 里的 all_reduce(g_n) 调用次数不匹配
                #   -> NCCL 死锁。修法: 各 rank 数自己本 epoch 的 micro 数, all_reduce(MIN) 取公共 micro 数, 两 rank 都只跑
                #   common_opt_steps = common_micro // accum 步(余数丢弃, 每组必满 accum 个 micro)。
                #   对应 verl prepare_micro_batches 的 same_micro_num_in_dp=True。
                if self.batch_sampler is not None:
                    local_micro_count = sum(1 for _ in self.batch_sampler)   # 精确数本 rank 本 epoch 的 micro 数
                else:
                    local_micro_count = len(self.train_dl)                    # 右填充: 固定 batch_size, len 精确
                if self.world_size > 1:
                    mc = torch.tensor([float(local_micro_count)], dtype=torch.float32, device=self.device)
                    dist.all_reduce(mc, op=dist.ReduceOp.MIN)
                    common_micro = int(mc.item())
                else:
                    common_micro = local_micro_count
                common_opt_steps = common_micro // accum

                dl_iter = iter(self.train_dl)
                for _step in range(common_opt_steps):
                    # 预取一组 micro(= 一个 optimizer step 要处理的 micro 数), 对应 verl prepare_micro_batches。
                    micro_group = list(itertools.islice(dl_iter, accum))

                    t0 = time.time()
                    # 【新增·模块10】一次 optimizer step 全交给 engine.train_batch(骨架: zero_grad ->
                    #   forward_backward_batch[all_reduce g_n + 微批 no_sync 循环] -> optimizer_step[clip+step])。
                    out = eng.train_batch(micro_group, self.loss_fn)
                    lr = eng.lr_scheduler_step()   # lr 调度(对应 verl worker 调 BaseEngine.lr_scheduler_step; 不在 train_batch 骨架里)
                    dt = time.time() - t0

                    global_step += 1
                    metrics = self._postprocess(out, dt, epoch, lr)   # all_reduce loss + MFU, rank0 落日志
                    if self.is_rank0 and self.tracker is not None and (
                        global_step % cfg.trainer.log_every == 0 or global_step == 1
                    ):
                        self.tracker.log(metrics, step=global_step)

        self._save_model()
        if self.is_rank0 and self.tracker is not None:
            self.tracker.finish()
        if self.is_rank0:
            print("训练完成。")

    def _postprocess(self, out, delta_time, epoch, lr):
        """指标后处理: 对应 verl TrainingWorker._postprocess_output (engine_workers.py:212)。
        (1) loss 跨卡 all_reduce(SUM) 再 /global_num_tokens = 全局 per-token 均值(对应 verl :229 loss all_reduce);
        (2) grad_norm 已在 engine.optimizer_step 全局归约(FSDP 手动 all_reduce), 不再重复 all_reduce(verl 明确 pop grad_norm);
        (3) MFU 估算(对应 verl :260 FlopsCounter.estimate_flops)。
        """
        # (1) 全局 loss: out["loss"] 是本 rank 这组 micro 的 local_loss_sum(累计), all_reduce(SUM) 跨 rank
        #   再 /global_num_tokens = 全局 per-token 均值。注意是 SUM 不是 AVG —— global_num_tokens 已含所有 rank 的 token,
        #   SUM(local_loss_sum)/global_num_tokens 才是全局均值; AVG 会多除一次 world_size(模块9 用 SUM, 这里沿用)。
        #   必须 float32 再 all_reduce(模块6 的 bf16 舍入坑: 357->356)。
        loss_t = torch.tensor([out["loss"]], dtype=torch.float32, device=self.device)
        global_num_tokens = out["num_tokens"]   # engine 已 all_reduce(SUM) 好的全局 token 数
        if self.world_size > 1:
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        global_loss = loss_t.item() / global_num_tokens

        grad_norm = out["metrics"]["grad_norm"]   # 已全局归约, 不重复 all_reduce
        metrics = {
            "train/loss": global_loss,
            "train/lr": lr,
            "train/grad_norm": grad_norm,
            "train/epoch": epoch,
            "train/global_tokens": global_num_tokens,
        }
        # (3) MFU: 每 token 约 6*n_param FLOPs(2x 前向 + 1x 反向≈3x 前向, 6≈2*3); 除以 (dt * world_size * 单卡峰值 FLOPS)。
        #   对应 verl FlopsCounter.estimate_flops(精确分层估算); 这里用 6*n_param 近似。RTX3090 bf16 峰值约 35.6 TFLOPS。
        n_param = self.engine.n_full_params
        est_flops = 6.0 * float(n_param) * float(global_num_tokens)
        peak = 35.6e12 * self.world_size
        metrics["train/mfu"] = est_flops / (delta_time * peak) if delta_time > 0 else 0.0
        return metrics

    def _save_model(self):
        """【模块9 沿用】FSDP 保存: summon_full_params(rank0_only) 拼回全量, rank0 落盘; automodel 直接 save_pretrained。"""
        cfg = self.cfg
        if not cfg.trainer.save_model:
            if self.is_rank0:
                print("save_model=false, 跳过保存(仅验证配置/流程)")
            return
        if self.is_rank0:
            print(f"保存模型到 {cfg.trainer.output_dir}")
        eng_model = self.engine.model
        if isinstance(eng_model, FSDP):
            # 【模块9 坑】FSDP 模型参数是分片 DTensor, 不能直接 save_pretrained。
            #   summon_full_params(rank0_only=True, writeback=False): 仅 rank0 拼回全量, 非 rank0 不参与 collective(不挂);
            #   所有 rank 都进 with(上下文进入/退出是安全的), 但只在 rank0 落盘。verl 走 FSDP.state_dict, 等价。
            with FSDP.summon_full_params(eng_model, writeback=False, rank0_only=True):
                if self.is_rank0:
                    base = eng_model.module if hasattr(eng_model, "module") else eng_model
                    base.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
                    self.tokenizer.save_pretrained(cfg.trainer.output_dir)
        else:
            # automodel / 单卡: 直接存(底层模型在 .module 若曾被 DDP 包, 这里 automodel 不包, 直接 eng_model)
            if self.is_rank0:
                base = eng_model.module if hasattr(eng_model, "module") else eng_model
                base.save_pretrained(cfg.trainer.output_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(cfg.trainer.output_dir)
        if self.world_size > 1:
            dist.barrier()
