# 模块 9：FSDP 模型分片 (FullyShardedDataParallel)

> 在 `9_data_parallel`（模块8）基础上增量添加。对应 README 路线图【阶段三·模块9】。
> 把"模型整份复制(DDP)"升级为"参数/梯度/优化器状态分片(FSDP)", 显存占用从 O(模型) 降到 O(模型/world_size)。

## 这一模块加了什么

- **`config.py` 新增 `FSDPConfig`**（对应 verl engine_config 的 fsdp 块）：`enabled / param_offload / optimizer_offload / reshard_after_forward / forward_prefetch / use_orig_params / sync_module_states`。挂入 `SFTConfig.fsdp`, `load_config` 的 `setdefault` 加 `'fsdp'`。
- **`config.yaml`** 新增 `fsdp:` 块默认值。
- **`train.py`**：
  - **模型包装从 DDP 换成 FSDP**（`cfg.fsdp.enabled=true` 时；`false` 退回模块8 DDP 作对照）。对应 verl `_build_fsdp_module`：
    - `auto_wrap_policy = transformer_auto_wrap_policy({Qwen2DecoderLayer})`（verl 用 `_no_split_modules`，Qwen2 即 `Qwen2DecoderLayer`）。
    - `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=bf16)`（对应 verl 同名配置，reduce 用 fp32 保数值稳）。
    - `sharding_strategy = FULL_SHARD`（`reshard_after_forward=True` → ZeRO-3；`False` → `SHARD_GRAD_OP` ZeRO-2）。
    - `sync_module_states=True`（rank0 权重广播给其余 rank）、`use_orig_params=True`（保留原参数名，clip/save 友好）、`forward_prefetch`、`cpu_offload`（verify 关）。
  - **分片证据打印**：wrap 前 `full_params=494M`, wrap 后 `shard_params=247M ≈ 全量/world_size`。
  - **`fsdp_grad_norm_and_clip` 辅助**：FSDP 下各 rank 只持梯度 shard, 手动 `all_reduce(SUM of local²)` → sqrt 得**全局**范数, 再全局统一缩放裁剪（对应 verl worker 层的 grad_norm all_reduce）。
  - **存模型**：`with FSDP.summon_full_params(model): model.module.save_pretrained(...)`（不 summon 直接存只存到 shard/缺参数）。
  - **沿用模块8 不变**：`*dp_size` 技巧（`sft_loss_dp`）、`maybe_no_sync`（FSDP 也有 `no_sync`）、全局 loss `all_reduce(SUM)`、min-count 防死锁、rank0 控制流。

## 和上一模块(9_data_parallel)的区别

数据侧 + loss 侧（`*dp_size`、`no_sync`、全局 loss、min-count、rank0 控制流）**全不变**；只换**模型怎么包、grad_norm 怎么算、模型怎么存**：
- 模块8（DDP）：整模型在每张卡上复制一份, 反向 `all-reduce` 梯度并平均；`clip_grad_norm_` 直接得全局范数（梯度已全量同步）；存盘走 `model.module`。
- 模块9（FSDP）：参数/梯度/优化器状态**分片**到各卡, 前向 all-gather 拼回、反向 reduce-scatter 分走；`clip_grad_norm_` 只返回**局部 shard 范数**, 需手动 `all_reduce(SUM of local²)` 得全局范数；存盘需 `summon_full_params` 拼回。

## 这块的意义

1. **省显存**：DDP 每卡存完整模型（参数+梯度+优化器状态 = ~16×参数字节 for AdamW fp32）；FSDP 分片后每卡只存 `1/world_size`。0.5B 两卡看不出压力, 但 7B+/单卡放不下时 FSDP 是关键（这正是 verl 默认用 FSDP 引擎的原因）。
2. **FSDP ≈ DDP 数值等价**：FSDP 与 DDP 一样对梯度做 `/world_size` **平均**（见下方"关键发现"），故 `*dp_size` 技巧同样适用, 训练轨迹与 DDP 几乎重合（见验证：loss 逐位接近）。FSDP 不是"另一种算法", 只是"DDP 的省显存实现"。
3. **`no_sync` 在 FSDP 下同构**：FSDP 也有 `no_sync()`, 把 reduce-scatter 推迟到累积窗口最后一步（对应 verl `_gradient_sync_context`），和模块8 DDP 的 `no_sync` 语义一致, 只是底层通信原语不同。

## 关键发现（实测）：FSDP 对梯度做 /world_size 平均, 与 DDP 一致

PyTorch 的 reduce-scatter 默认是 SUM, 故一度怀疑 FSDP 是**求和**（若是求和, `*dp_size` 技巧会过度补偿 `world_size` 倍）。用 `/tmp/fsdp_reduce_probe.py` 实测（两 rank **不同输入**, backward+一步 SGD 后比较权重）：
- 两 rank 的 post-step 权重**完全一致** → FSDP 确实同步了梯度（不是"各自为政"）。
- 该共同权重 delta == **两单进程 delta 的平均**（不是求和）→ FSDP **平均**梯度, 与 DDP 一致。

故 `sft_loss_dp` 的 `*dp_size` 抵消 `/world_size` 平均在 FSDP 下仍正确, 模块9 直接沿用模块8 的 loss 不改。这与 verl 一致——verl 的 `*dp_size` 在 DDP/FSDP/automodel 后端统一使用。

> **verl 对应**：`verl/workers/engine/fsdp/transformer_impl.py` 的 `_build_fsdp_module`（FSDP 构造）、`_gradient_sync_context`（`no_sync`）、`forward_backward_batch`（`all_reduce(batch_num_tokens)`）；`verl/utils/fsdp_utils.py` 的 `get_fsdp_wrap_policy`（`_no_split_modules`）；`verl/workers/utils/losses.py:49`（`*dp_size`, DDP/FSDP 共用）。
> **简化点**：不做 HSDP（2D mesh）、不做 fsdp2（`fully_shard` composable）、不做 meta device `param_init_fn`（模型已从磁盘加载）、不开 CPUOffload（verify 关, 仅留开关）、save 路径用 `summon_full_params`（verl 用 `FSDP.state_dict`）。

## 验证

### 1. 单进程烟测（`world_size=1` → raw 模型路径, 不包 FSDP）

```bash
cd verl/learning/sft/code/10_fsdp
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 trainer.grad_accum_steps=2 \
    trainer.experiment_name=qwen0.5b-fsdp-single
```
- `world_size=1` 时 `world_size>1 and cfg.fsdp.enabled` 为假, 不包 FSDP（走 raw 模型路径, 等价模块8 单进程）。验证 FSDP/Qwen2 导入无误 + loss 流程不变。

### 2. 两卡 FSDP torchrun

```bash
cd verl/learning/sft/code/10_fsdp
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
/home/zzy/.conda/envs/easysteer/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29510 train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 trainer.grad_accum_steps=2 \
    trainer.experiment_name=qwen0.5b-fsdp-demo
```
- **分片证据**：`已用 FSDP 包装模型(dp_size=2, strategy=1); 全量参数 494.0M, 各 rank shard 247.0M (期望 ~247M = 全量/world_size)` —— 参数被对半分片（DDP 会是 494M）。`strategy=1` = `FULL_SHARD`。
- **FSDP loss ≈ DDP loss**（与模块8 逐步对比）：

  | step | 模块8 DDP loss | 模块9 FSDP loss | global_tokens |
  |---|---|---|---|
  | 1 | 0.5817 | 0.5824 | 4256 |
  | 2 | 0.5966 | 0.5966 | 4339 |
  | 3 | 0.5076 | 0.5083 | 4242 |
  | 4 | 0.4126 | 0.4109 | 4001 |
  | 5 | 0.4666 | 0.4622 | 4139 |

  逐位接近（差在第 3-4 位小数, 来自 FSDP `MixedPrecision(bf16/fp32)` vs DDP `autocast` 的舍入差异）→ **FSDP 与 DDP 数值等价**, 证明 `*dp_size` 在 FSDP 下正确（FSDP 平均梯度, 同 DDP）。
- `global_tokens` 与 DDP 完全一致（4256/4339/...）→ `all_reduce(SUM)` token 数在 FSDP 下同样正确。
- `grad_norm` 为 float32 全局值（`23.4457` vs 模块8 的 `23.5`——后者 `clip_grad_norm_.item()` 舍入, 前者手动 float32 精确）, 量级与 DDP 一致。
- 5 步无死锁（min-count 防死锁从模块8 沿用）。指标落 `outputs/logs/gsm8k-sft/qwen0.5b-fsdp-demo.jsonl`（5 条, rank0 写）。退出后 GPU 显存全释放。

## 踩过的坑

### 坑1（关键）：FSDP 下 `clip_grad_norm_` 返回局部 shard 范数, 不是全局
`torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm)` 作用于各 rank 的 flat shard, 返回的是**本 rank 局部范数**（实测 = 全局/√world, 见 `/tmp/fsdp_reduce_probe.py`：identical 输入下 ratio 0.71≈1/√2）。直接当全局范数用会偏小、裁剪失真。
**修法**：`fsdp_grad_norm_and_clip` 手动算各 shard 平方和 → `all_reduce(SUM)` → sqrt 得全局范数 → 全局统一 `clip_coef` 缩放各 shard（同比例 → 裁剪后全局范数 == max_norm）。对应 verl worker 层对 grad_norm 的 all_reduce。
（注：verl 的 `*dp_size` 抵消的是 `/world_size` 平均, 与范数无关; 范数仍需单独 all_reduce。）

### 坑2：FSDP 平均梯度（不是求和）—— 经实测确认
PyTorch reduce-scatter 默认 SUM, 故怀疑 FSDP 求和（若求和, `*dp_size` 会过度补偿）。实测（两 rank 不同输入, 比较一步 SGD 后权重）证明 FSDP **平均**梯度（post-step 权重 == 两单进程 delta 的平均）, 与 DDP 一致。故 `*dp_size` 沿用不改。详见上方"关键发现"。

### 坑3：CPUOffload 配合梯度累积会算错
verl `transformer_impl.py:434` 明确注释 actor 训练强制关 `param_offload`（CPUOffload + 梯度累积出错）。verify 里 `param_offload=false`、`optimizer_offload=false`, 开关留着只为对照 verl 配置完整性。

### 坑4：FSDP 存盘必须 `summon_full_params`
不 summon 直接 `save_pretrained` 只存到 shard/缺参数。正确：`with FSDP.summon_full_params(model): model.module.save_pretrained(...)`。verify 设 `save_model=false` 规避（真要存时用此模式, verl 走 `FSDP.state_dict`）。

### 坑5（继承）：`CUDA_DEVICE_ORDER=PCI_BUS_ID` + bf16 sum float32 + DP 死锁 min-count
分别见模块7、模块6、模块8 README。FSDP 路径全部继承, 不再赘述。
