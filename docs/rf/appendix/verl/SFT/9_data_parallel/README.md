# 模块 8：数据并行 (DP) —— DDP + no_sync 梯度累积 + *dp_size 全局 loss

> 在 `8_distributed_init`（模块7）基础上增量添加。对应 README 路线图【阶段三·模块8】。
> 数据分片(采样器侧)已由模块7 完成, 本模块只动**模型/梯度**侧: 把"各自为政"变成"真正并行"。

## 这一模块加了什么

- **`loss.py` 新增 `sft_loss_dp`**（对应 `verl/workers/utils/losses.py:49`）：精确复刻 verl 的 `*dp_size` 技巧
  `loss = local_loss_sum / global_num_tokens * dp_size`。`global_num_tokens` 是**全局** trained token 总数（跨 rank 跨 micro，调用方先 `all_reduce(SUM)` 算好），`dp_size=world_size`。`*dp_size` 抵消 DDP 反向时对梯度的 `/world_size` 平均 → 等效梯度 = 真全局 per-token 均值 loss 的梯度。返回 `(loss, local_loss_sum.detach())`，第二个供日志累积算全局 loss。保留模块6 的 `sft_loss`（单卡/对照）。
- **`train.py`**：
  - **DDP 包装**（`world_size>1` 时）：`model = DDP(model, device_ids=[local_rank], find_unused_parameters=...)`，必须在 `gradient_checkpointing_enable + enable_input_require_grads` **之后**包（DDP 要求首层输入有梯度才能触发反向钩子）。`dp_size = world_size`。
  - **`maybe_no_sync(model, is_last)` 辅助**：非最后一个 micro 进 `model.no_sync()` 关闭跨 rank 梯度同步, 只在最后一个 micro 的 backward 触发一次 `all-reduce(avg)`。对应 verl `_gradient_sync_context`（`use_no_sync_for_gradient_accumulation=True`）。
  - **训练循环重构成"按 optimizer-step 预取 `grad_accum` 个 micro 一组"**（对应 verl `forward_backward_batch` + `prepare_micro_batches`）：因为全局 `batch_num_tokens` 要在进 micro 循环前 `all_reduce(SUM)` 算好。每组：算本 rank 的 local token 数 → `all_reduce(SUM)` 得 `global_num_tokens`（float32！）→ micro 循环 `sft_loss_dp` + `maybe_no_sync` backward → `clip_grad_norm_` + step。
  - **全局 loss 日志**（仅 rank0）：`all_reduce(local_loss_sum, SUM)/global_num_tokens` = 全局 per-token 均值（对应 verl `_postprocess_output` 的 `all_reduce(loss, AVG)`）。必须在 `.item()` 后再 all_reduce，不能让反传图穿过 `all_reduce`。
  - **grad_norm 不单独 all_reduce**：DDP 反向已同步梯度, `clip_grad_norm_` 算的是同步后的全局一致范数（对应 verl 注释 "grad_norm 不再 all_reduce, 因为裁剪时梯度已同步"）。
  - **DP 死锁修复**：各 rank 数本 epoch micro 数 → `all_reduce(MIN)` 取公共 micro 数 → 两 rank 都只跑 `common_opt_steps = common_micro // grad_accum` 步（对应 verl `same_micro_num_in_dp=True`）。
  - **存模型**：DDP 包装后底层模型在 `model.module`, 走它存盘（`base_model = model.module if isinstance(model, DDP) else model`）。
- **`config.py` / `config.yaml`**：加 `find_unused_parameters: bool = False`（对应 verl 引擎配置；Qwen2 前向所有参数都被用, 默认 False）。

## 和上一模块(8_distributed_init)的区别

数据侧（进程组、`TokenBudgetBatchSampler` 的 `num_replicas/rank` 分片、rank0 控制流、`CUDA_DEVICE_ORDER` 兜底）**全不变**；只换了**梯度怎么同步、loss 怎么归一化**：
- 模块7：模型**不**包 DDP, 各 rank 在各自数据分片上各自 `backward+step`, 梯度**不跨卡同步** → 模型发散（故意的中间态）。
- 模块8：模型包 DDP, 反向时梯度自动 `all-reduce` 并 `/world_size` 平均；`no_sync` 让一组 `grad_accum` 个 micro 只同步一次；`*dp_size` 让 loss 在 token 数不齐时仍精确等于全局 per-token 均值 → 各 rank 模型始终一致, 等效于单卡大 batch 训练。

## 这块的意义

1. **多卡真正"并行"**：每卡见不重叠数据, 梯度跨卡同步平均 = 等效单卡大 batch。模块7 的"各自为政发散"在此被解决。
2. **`*dp_size` 让 loss 数值与梯度在 token 数不齐时仍精确**：朴素 DDP（每 rank 算自己的 per-token 均值再让 DDP 平均梯度）得到的是 **mean-of-means**, 当各 rank/micro 的 trained token 数不齐（动态 batch 下常态）时 ≠ 真全局 per-token 均值, 长样本多的 rank 权重被低估。`*dp_size` 技巧让梯度精确等于全局 per-token 均值的梯度（数学见 `loss.py` 注释）。
3. **`no_sync` 省通信**：一组 `grad_accum` 个 micro 只在最后一步同步一次, 而非每 micro 都 `all-reduce`（通信量降 `grad_accum` 倍）。
4. **模块7→模块8 的对比是理解 DDP 的关键**：模块7 rank0 的 loss 与单进程**逐位一致**（证明无同步）；模块8 rank0 的 loss **偏离**单进程（证明梯度已同步, 模型见了全部数据）——见下方验证。

> **verl 对应**：`sft_trainer.py` 的 `DistributedSampler`；engine `forward_backward_batch`（`all_reduce(batch_num_tokens, SUM)`）；`verl/workers/utils/losses.py:49` 的 `*dp_size`；`_gradient_sync_context`（`no_sync`）；`engine_workers._postprocess_output`（`all_reduce(loss, AVG)`）；`prepare_micro_batches`（`same_micro_num_in_dp`）。
> **简化点**：verl 不用 `torch.nn.DDP`, 用自研 FSDP/Megatron 引擎（梯度 SUM-reduce, 不平均, 故不需 `*dp_size` 抵消——直接除全局 token 数）。本学习版用 `DDP`（梯度平均）, 故需 `*dp_size` 抵消 `/world_size`。模块9（FSDP）会换成 verl 风格的引擎。

## 验证

### 1. 单进程烟测（`dp_size=1` 回退路径）

```bash
cd verl/learning/sft/code/9_data_parallel
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 trainer.grad_accum_steps=2 \
    trainer.experiment_name=qwen0.5b-dp-single
```
- 不包 DDP, `dp_size=1`, `sft_loss_dp` 退化等价于全局 per-token 均值（分母 = 本组所有 micro 的 token 和）。
- 12 步, `train/global_tokens` ~2100/step（1 rank × 2 micro）, loss 0.62→0.37 趋势下降后随 LR 升到 1e-4 变噪（200 样本 + 高 LR 的正常表现）。

### 2. 两卡 DDP torchrun（真正的数据并行）

```bash
cd verl/learning/sft/code/9_data_parallel
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
/home/zzy/.conda/envs/easysteer/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29500 train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 trainer.grad_accum_steps=2 \
    trainer.experiment_name=qwen0.5b-dp-demo
```
- `已用 DDP 包装模型(dp_size=2, find_unused_parameters=False)`。
- **`train/global_tokens` ~4250/step ≈ 恰好 2× 单进程的 ~2100** → 跨 rank `all_reduce(SUM)` token 数正确（2 rank × 2 micro × ~1050 trained token）。
- **rank0 loss 偏离单进程**：step1 `0.5817`（单进程 `0.6191`）—— 与模块7 的"逐位一致"形成对比。模块7 一致 = 无同步；模块8 偏离 = **梯度已同步**, 模型更新反映了**两 rank 的全部数据**而非 rank0 的前缀。这正是 DDP 数据并行的本质。
- loss 0.58→0.41（step4）后随 LR 升高变噪（同单进程）, grad_norm 23.5→9.0。
- 5 个 optimizer step（min-count 把估算的 6 步裁成两 rank 公共的 5 步, 无死锁）。
- 指标落 `outputs/logs/gsm8k-sft/qwen0.5b-dp-demo.jsonl`, **只有 5 条**（rank0 写）→ rank0 + all_reduce(loss) 守卫正确。
- 退出后 GPU 显存全释放, 无 NCCL 挂死。

## 踩过的坑

### 坑1（致命）：DP 死锁 —— 各 rank micro 数不一致

初版用 `while True: micro_group = islice(dl_iter, grad_accum); if not micro_group: break` 流式取 micro。但动态 batch + `drop_last` + 样本长度不齐 → **两 rank 的 micro 数差 1**, 于是两 rank 的 `all_reduce(g_n)` 调用次数不匹配 → NCCL `all_reduce` 互相等待 → **死锁**（2-GPU run 挂死超 600s, 进程卡在 NCCL 不可中断, 需 `kill -9 <pid>` 清）。

**修法**：epoch 开头各 rank 数自己本 epoch 的 micro 数（`sum(1 for _ in batch_sampler)`）, `all_reduce(MIN)` 取公共 micro 数, 两 rank 都只跑 `common_opt_steps = common_micro // grad_accum` 步（余数丢弃, 每组必满 `grad_accum` 个 micro）。对应 verl `prepare_micro_batches` 的 `same_micro_num_in_dp=True`。修后两 rank 同步退出, 无死锁。

**教训**：任何**集体通信**（`all_reduce`/`broadcast` 等）的调用次数必须在所有 rank 上一致, 否则死锁。数据并行的数据分片若不能保证各 rank 步数相同, 就得显式对齐（取 min 或 pad）。

### 坑2（继承自模块7）：`CUDA_DEVICE_ORDER=PCI_BUS_ID`
不加会 `CUDA_VISIBLE_DEVICES=0,2` 取到 {3090, 2080Ti}, rank1 落 11GB 卡 OOM。`dist.py` 已 `setdefault` 兜底, 命令行也显式带上。详见模块7 README。

### 坑3（继承自模块6）：`all_reduce` 必须 float32
`batch_num_tokens` 来自 `loss_mask.sum()`, bf16 下 357 会被舍入成 356（bf16 尾数 7 位）。代码里 `torch.tensor([float(local_n)], dtype=torch.float32)`。`*dp_size` 的全局 loss 日志同理 float32。

### 坑4：DDP + 梯度检查点必须 `enable_input_require_grads`
DDP 要求首层输入有梯度才能触发反向钩子；梯度检查点把首层激活重算, 没有 `input_require_grads` 会报 "Expected to mark a variable"。模块3 已设, 顺序是 `gradient_checkpointing_enable → enable_input_require_grads → DDP 包装`。

### 坑5：forward 必须走 DDP 包装层
`model(**fwd_kwargs)`（DDP.forward）, 不是 `model.module(**fwd_kwargs)`。DDP.forward 登记 grad reduction 准备态, 直接调 `module.forward` 会让反向钩子不触发、梯度不同步。4D 块对角掩码作为 `attention_mask` 经 DDP.forward 转发到底层模型, 无副作用。
