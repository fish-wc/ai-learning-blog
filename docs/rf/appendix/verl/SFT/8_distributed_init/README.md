# 模块 7：分布式进程组 (torch.distributed)

> 在 `7_manual_loss`（模块6）基础上增量添加。对应 README 路线图【阶段三·模块7】。
> 这是**阶段三（分布式）的第一个模块**，建立 verl 一切分布式能力的前置通信骨架。

## 这一模块加了什么

- **新建 `dist.py`**（对应 `verl/utils/distributed.py`）：
  - `initialize_global_process_group(backend, timeout)`：读 torchrun 注入的 `RANK/WORLD_SIZE/LOCAL_RANK`，`dist.init_process_group(backend="nccl")` 建 NCCL 进程组，`torch.cuda.set_device(local_rank)` 落到本 rank 的卡，返回 `(local_rank, rank, world_size)`。保留单进程回退（`RANK` 不在 env 时返回 `(0,0,1)`，方便 `python train.py` 烟测）。
  - `destroy_global_process_group()`：所有 rank 收尾调，释放 NCCL communicator（不调的话 torchrun 进程挂住不退出）。
  - `is_rank0() / get_world_size() / get_rank()`：rank 感知的小工具，对应 verl 里到处用的 `if self.rank == 0`。
  - **环境兜底**：模块顶部 `os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")`（见下方"踩过的坑"）。
- **`config.py` / `config.yaml`**：`TrainerConfig` 加 `world_size: int = 1`（torchrun 注入，运行时用真实值覆盖）和 `backend: str = "nccl"`。对应 verl 从 env 读 world_size 而非信 yaml。
- **`dataset.py`**：`TokenBudgetBatchSampler` 加 `num_replicas / rank` 两个参数。`__iter__` 用同一 seed 生成**全局 randperm**，再按 `rank` 取不重叠的一段（`shard_idx = global_idx[rank*per:(rank+1)*per]`），**yield 全局下标**（不是局部下标，否则所有 rank 都取 `train_ds[0..per-1]` → 数据重复）。对应 verl `_build_dataloader` 的 `DistributedSampler(num_replicas=dp_size, rank=dp_rank)`。右填充路径改用 `torch.utils.data.DistributedSampler`。
- **`train.py`**：
  - `main()` 开头调 `initialize_global_process_group()` 拿 `(local_rank, rank, world_size)`；`cfg.trainer.world_size = world_size`（用真实值覆盖占位）；`device = f"cuda:{local_rank}"`；`torch.manual_seed(seed)`（各 rank 同 seed，为模块8 DDP 梯度同步铺路）。
  - **rank-aware 控制流**：只有 rank0 打印配置 / 创建 `Tracking` / 写日志 / 存模型（`is_main = is_rank0()`）。非 rank0 的 `tracker=None`，避免两个进程同时 `'w'` 打开同一 jsonl 互相截断。
  - `model.to(device)` / `batch.to(device)` 都走本 rank 的卡（取代 `.to("cuda")`）。
  - 右填充路径每 epoch `train_dl.sampler.set_epoch(epoch)`（DistributedSampler 的要求，否则各 epoch 打乱一致）。
  - 收尾 `destroy_global_process_group()`（所有 rank 调）。

## 和上一模块(7_manual_loss)的区别

数据流（packing、动态 batch、4D 掩码、手算 sft_loss、梯度检查点、追踪）**全不变**；只换了**怎么启动 / 谁来跑**这一层：
- 模块6：`python train.py` 单进程，一张卡，一份完整数据，梯度和日志都自然只有一份。
- 模块7：`torchrun --nproc_per_node=N train.py` 起 N 个进程，建 NCCL 进程组，每个 rank 拿 `(local_rank, rank, world_size)`，各跑一张卡、各见不重叠的数据分片，只有 rank0 写日志/存盘。

**故意停在的中间态（关键）**：本模块只建进程组、**没同步梯度**（没 DDP/FSDP 包 model）。各 rank 在各自的数据分片上各自 `backward + step`，模型会**发散**。这正是"没有 DDP 时多卡各自为政"的中间态，用来理解模块8（DDP 梯度同步）的必要性。

## 这块的意义

1. **建立分布式前置骨架**：进程组 + `rank/world_size` + `local_rank` 选卡 + rank0 控制流 + 数据分片——verl 一切分布式能力（DP/FSDP/SP/Engine 抽象）都长在这套骨架上。先把骨架立起来，后续模块只往里加"梯度同步""参数分片""引擎抽象"。
2. **数据并行的数据侧**：`TokenBudgetBatchSampler(num_replicas, rank)` 让每个 rank 见不重叠子集——这是"数据并行"的真正含义（不是"多卡各跑一份相同数据"）。为模块8 DDP 梯度同步铺好数据侧。
3. **rank0 控制流**：日志/存盘/打印只在 rank0 做，避免 N 份重复日志、N 个进程抢写同一文件。对应 verl `sft_trainer.py` 的 `if self.rank == 0`。
4. **三段式生命周期**：`init -> fit -> destroy`，对应 verl `run_sft()` 的 `initialize_global_process_group() -> trainer.fit() -> destroy_global_process_group()`。

> **verl 对应**：`verl/utils/distributed.py`（`initialize/destroy_global_process_group`、`get_nccl_backend`）；`verl/trainer/sft_trainer.py` 的 `run_sft`（三段式）+ `SFTTrainer.__init__` 的 rank0 守卫 + `_build_dataloader` 的 `DistributedSampler`。
> **简化点**：verl backend = `f'cpu:gloo,{device}:{nccl_backend}'`（CPU+GPU 双组，给 vLLM 权重同步等用），本学习版只用单一 nccl backend；verl 用 Ray + `DIST_INIT_METHOD` 注入 init_method，本版用 torchrun 注入的 env（`init_process_group` 自动读）。

## 验证

### 1. 单进程烟测（回退路径：`RANK` 不在 env）

```bash
cd verl/learning/sft/code/8_distributed_init
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 \
    trainer.experiment_name=qwen0.5b-dist-init-single
```
- `[dist] world_size=1, rank=0, local_rank=0, device=cuda:0`（回退路径：没建进程组）。
- loss 0.595→0.354（6 步），grad_norm 20.25→5.16，与模块6 一致（单进程路径代码不变）。

### 2. 两卡 torchrun（真正的分布式路径）

```bash
cd verl/learning/sft/code/8_distributed_init
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
/home/zzy/.conda/envs/easysteer/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29500 train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 \
    trainer.experiment_name=qwen0.5b-dist-init-demo
```
- `[dist] world_size=2, rank=0, local_rank=0, device=cuda:0, backend=nccl`（进程组建好）。
- **rank0 的 loss 与单进程逐位一致**：step1 `0.5949621200561523`、step2 `0.6323273181915283`（与单进程完全相同）。
  - 这同时证明两件事：(a) 数据分片正确——rank0 取 `global_idx[0:per]`，是单进程轨迹的**前缀**，所以前几步见同样的样本、同样的顺序；(b) **梯度没有同步**——若 DDP 在平均 rank0+rank1 的梯度，rank0 的 loss 会偏离单进程。逐位一致正是"无 DDP、各 rank 各自为政"的签名。
- `估算每 rank 12 个 micro-batch/epoch`（单进程是 23）→ 每个 rank 见约一半数据，分片生效。
- 指标落 `outputs/logs/gsm8k-sft/qwen0.5b-dist-init-demo.jsonl`，**只有 2 条**（rank0 写，非 rank0 的 `tracker=None` 不写）→ rank0 守卫正确。
- 进程退出后 GPU 显存全释放（`nvidia-smi` 各卡 ~15MiB），无 NCCL 挂死 → `destroy_global_process_group()` 正常收尾。

## 踩过的坑

### 坑1（环境·必设）：CUDA_DEVICE_ORDER 导致 rank1 落到 11GB 的 2080Ti 上 OOM

机器有 3 张卡：GPU0=3090(24GB)、GPU1=2080Ti(11GB)、GPU2=3090(24GB)。本想用 `CUDA_VISIBLE_DEVICES=0,2` 取两张 3090。但 **PyTorch 默认 `CUDA_DEVICE_ORDER=FASTEST_FIRST`**：按速度排序，两张 3090 排前（index 0、1）、2080Ti 排后（index 2）。于是 `CUDA_VISIBLE_DEVICES=0,2` 实际取的是 **{3090, 2080Ti}**——`local_rank=1` 落到 11GB 的 2080Ti，backward 时 OOM（`GPU 1 has a total capacity of 10.57 GiB`）。

之前 `/tmp/dist_smoke.py` 烟测没暴露这个坑，因为它只做个 `all_reduce` 不分配显存，哪张卡都跑得动。

**修法**：`CUDA_DEVICE_ORDER=PCI_BUS_ID`（按 PCI 总线序 = `nvidia-smi` 序），则 `0,2` = {3090, 3090}。已在 `dist.py` 顶部 `os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")` 兜底（用 setdefault 不覆盖用户显式设置），后续模块都继承。验证：

```
nvidia-smi:  0=01:00.0(3090)  1=02:00.0(2080Ti)  2=83:00.0(3090)
PCI_BUS_ID + CUDA_VISIBLE_DEVICES=0,2 -> cuda:0=3090(24GB), cuda:1=3090(24GB)  ✓
（不加 PCI_BUS_ID)                    -> cuda:0=3090(24GB), cuda:1=2080Ti(11GB) ✗
```

### 坑2（显存）：默认 `max_token_len_per_gpu=8192` 在无 flash_attn 时 OOM

本环境没装 flash_attn，packing 路径退用 4D 块对角掩码（物化 `(1,1,T,T)`）。`T=8192` 时：① dense float bias 把 SDPA 顶下 memory-efficient kernel，注意力权重物化成 `(1,14,8192,8192)`；② LM head logits `1×8192×151936` + float32 `log_softmax` 各 ~5GB。单卡 24GB 装不下。

**修法**：烟测用 `trainer.max_token_len_per_gpu=2048`（与模块6 验证命令一致；模块6 的默认 8192 同样从未跑通）。生产环境装 flash_attn 后 `8192` 才可行——这正是 verl 默认开 flash_attn_varlen 的原因。

> **bf16 求和坑**（从模块6 继承，本模块仍生效）：`loss_mask.sum()` 必须在 float32 里算，bf16 尾数 7 位会把 >256 的整数舍入（357→356）→ loss 偏 ~0.28%。`loss.py` 的 `sft_loss` 已用 `shift_mask = loss_mask[:, 1:].float()` 修好。模块8 做 `all_reduce(loss)` 时同理：必须在 float32 上 reduce，不能在 bf16 上。
