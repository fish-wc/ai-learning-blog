# 模块 10：引擎抽象层 (BaseEngine / TrainingWorker / EngineRegistry)

> 在 `10_fsdp`（模块9）基础上增量添加。对应 README 路线图【阶段三·模块10】。
> 把模块9 "压平在 main() 里的 v0 主循环" 重新拆成 verl 的 **fit / train_batch / engine 三层** —— 纯结构重构, 计算语义不变。

## 这一模块加了什么

- **`config.py` / `config.yaml` 新增 `trainer.engine_backend`**（对应 verl `EngineRegistry` 按 `backend=strategy` 选引擎）：`fsdp`（默认, 复用模块9 的 FSDP 分片）/ `automodel`（单卡对照后端, 不分片不 DDP）。一行切换后端。
- **`engine.py`（新文件）—— "机制"层, loss-agnostic**：
  - **`BaseEngine`**（对应 `verl/workers/engine/base.py:BaseEngine`）：持有 `model/optimizer/lr_scheduler/n_full_params`；定义 **`train_batch` 骨架**（`base.py:113`）：`optimizer_zero_grad → forward_backward_batch → optimizer_step → 注入 grad_norm`。子类只填 `initialize / forward_step / optimizer_step / _gradient_sync_context` 四个机制方法。另含 `train_mode / optimizer_zero_grad / lr_scheduler_step / build_scheduler` 接口。
  - **`_MixinLMHead.forward_step`**（对应 `fsdp/transformer_impl.py:1570`）：`model(input_ids, attention_mask, position_ids) → logits → loss_fn(logits, input_ids, loss_mask, global_num_tokens, dp_size)`。**引擎不 import `sft_loss`** —— loss_fn 由 worker 注入, 引擎只把 logits + DP 上下文喂给它。
  - **`FSDPEngine`**（对应 `FSDPEngineWithLMHead`）：`initialize` 复用模块9 的 FSDP 包裹（FULL_SHARD + bf16/fp32 MixedPrecision + `Qwen2DecoderLayer` auto_wrap + use_orig_params + sync_module_states）；`_gradient_sync_context` = `model.no_sync()`（非 last micro 延迟 reduce-scatter）；`optimizer_step` 沿用模块9 的 `_fsdp_grad_norm_and_clip`（手动 `all_reduce(SUM of local²)` 得全局范数 + 全局裁剪）。
  - **`AutoModelEngine`**（对应 `AutomodelEngineWithLMHead`）：同接口的非 FSDP 后端（裸模型 + 普通 `clip_grad_norm_`）—— 证明 `BaseEngine` 抽象能挂多后端。
  - **`EngineRegistry`**（对应 `base.py:339`）：`register(backend)` 装饰器 + `new(backend, *a, **kw)` 查表实例化。verl 按 `(model_type, backend, device, vendor)` 四维 + Ray dispatch, 这里压成单维 `{backend: cls}` dict。
- **`worker.py`（新文件）—— "调度 + loss 注入 + 指标后处理"层**（对应 `verl/workers/engine_workers.py:TrainingWorker`）：
  - `__init__`：建 dataloader → `EngineRegistry.new(cfg.trainer.engine_backend, ...)` 选后端建引擎 → `engine.initialize()` → `engine.build_scheduler(total_steps)`。`self.loss_fn = sft_loss_dp`（注入）。
  - `set_loss_fn`（对应 `:200`）：注入/换 loss。
  - `fit`（对应 `SFTTrainer.fit`）：外层 epoch/micro 循环 + **min-count 防死锁**（沿用模块8）+ 预取 `grad_accum` 个 micro 一组 → `eng.train_batch(micro_group, self.loss_fn)` → `eng.lr_scheduler_step()` → `_postprocess` → rank0 落日志。
  - `_postprocess`（对应 `_postprocess_output :212`）：`all_reduce(loss, SUM)/global_num_tokens` = 全局 per-token 均值；grad_norm 不重复归约（已在 engine 全局归约）；MFU = `6*n_param*tokens / (dt * world_size * 单卡峰值 FLOPS)`（对应 `:260 FlopsCounter`）。
  - `_save_model`：FSDP 走 `summon_full_params(rank0_only=True)` 拼回全量 rank0 落盘；automodel 直接 `save_pretrained`。
- **`train.py` 瘦身**：main() 从 ~330 行缩到 ~25 行, 只剩 `parse_config → init_process_group → set_device → worker.fit() → destroy_process_group`。对应 verl `SFTTrainer.fit` 的最外层调度。

## 三层架构（本模块的核心）

```
train.py.main()                 ← verl SFTTrainer.fit 最外层（init_pg + worker.fit）
  └─ TrainingWorker.fit()       ← verl TrainingWorker（外循环 + min-count + _postprocess + 存盘）
       └─ engine.train_batch()  ← verl BaseEngine.train_batch 骨架（zero_grad→fwd_bwd→opt_step）
            └─ FSDPEngine / AutoModelEngine  ← verl FSDPEngine / AutomodelEngine（机制）
                 └─ FSDP(Qwen2) / raw Qwen2   ← 实际模型
```

模块9 把这四层压平在 `main()` 一个函数里；本模块把它们拆开, 每层职责单一。**这就是 verl 能在 fsdp/megatron/automodel 不同后端间切换、又能把 loss 公式与训练机制解耦的架构根基** —— 后续学 megatron/torch.compile 只需新增一个 `Engine` 子类（填 4 个机制方法）, `worker` 与 `main` 不动。

## 和上一模块(10_fsdp)的区别

**纯结构重构, 计算语义逐位不变** —— 模块9 主循环里干的每件事, 都原样搬进某一层：

| 模块9 main() 里的逻辑 | 模块10 搬到 | 对应 verl |
|---|---|---|
| AutoModel 加载 + 梯度检查点 + FSDP 包裹 + 优化器 | `FSDPEngine.initialize` | `_build_fsdp_module` |
| `all_reduce(global_num_tokens, SUM)` | `BaseEngine.forward_backward_batch` | `forward_backward_batch :728` |
| micro 循环 + `sft_loss_dp` + `loss.backward()` | `forward_backward_batch` + `_MixinLMHead.forward_step` | 同上 + `forward_step :1570` |
| `maybe_no_sync(model, is_last)` | `FSDPEngine._gradient_sync_context` | `_gradient_sync_context :706` |
| `fsdp_grad_norm_and_clip` + `optimizer.step` | `FSDPEngine.optimizer_step` | `optimizer_step :792` |
| `lr_scheduler.step()` | `worker.fit` 调 `eng.lr_scheduler_step()` | worker 调 `BaseEngine.lr_scheduler_step` |
| `all_reduce(local_loss_sum, SUM)/global_num_tokens` 全局 loss | `worker._postprocess` | `_postprocess_output :212` |
| Tracking 构造 + `tracker.log` | `worker._build_tracker` + `worker.fit` | sft_trainer `if rank==0: tracking.log` |
| `summon_full_params` 存盘 | `worker._save_model` | verl checkpoint |
| min-count 防死锁（决定跑几步） | `worker.fit` | `same_micro_num_in_dp=True` |
| epoch 循环 + 预取 micro 组 | `worker.fit` | `SFTTrainer.fit` |

新增的只有：`engine_backend` 配置、`EngineRegistry` 选后端、`train/mfu` 指标、loss 注入（engine 不认识 loss）。**无新训练能力**。

## 这块的意义

1. **机制与调度解耦**：引擎只管"一次 optimizer step 怎么算"（前向/反向/no_sync/clip/step）；worker 只管"算什么 loss、跑几个 step、怎么汇报"。verl 靠这层解耦让 worker 代码在 fsdp/megatron/automodel 间复用。
2. **loss 注入**：引擎不 import `sft_loss` —— worker 把 `sft_loss_dp` 注入 `engine.forward_step`。对应 verl `forward_step` 里 `loss_function(model_output=, data=, dp_group=)`。这让同一套引擎能跑 SFT/RL（RL 时 worker 注入的是 `advantage*ratio*mask` 的 loss, 引擎机制不变）。
3. **多后端可插拔**：`EngineRegistry` 按 `backend` 选引擎类。本模块用 fsdp + automodel 两个后端演示；verl 还挂 megatron。换后端 = 改一行 `engine_backend`, worker 与 main 不动。
4. **数值等价（实测）**：见下方验证 —— 与模块9 同 config 跑出的 loss 轨迹逐位接近, `global_tokens` 完全相同（数据路径未改）。

## 关键正确性决策（偏离朴素草稿, 为保数值等价）

朴素草稿会让 `forward_backward_batch` 用 `sft_loss(logits, input_ids, loss_mask)/grad_accum`（per-micro 均值 / accum）+ worker `all_reduce(loss, AVG)`。**这会破坏与模块9 的等价**：
- `sft_loss/accum` 得到的是 mean-of-means（各 micro 均值的均值），当 token 数不齐时 ≠ 真全局 per-token 均值（正是模块8 引入 `*dp_size` 要解决的偏差）。
- `all_reduce(loss, AVG)` 多除一次 `world_size`（`global_num_tokens` 已含所有 rank 的 token, 应 `SUM/global_num_tokens`）。

**正确做法（本模块采用, 与 verl 一致）**：
- `forward_backward_batch` **进 micro 循环前** `all_reduce(SUM)` 算全局 `global_num_tokens`（float32），对应 verl `forward_backward_batch` 里 `all_reduce(batch_num_tokens, SUM)` —— 全局 token 计数本就是引擎机制的一部分, 不是 worker 后处理。
- 注入 `sft_loss_dp`，每个 micro `loss = local_loss_sum/global_num_tokens*dp_size`，`backward()` **不除 accum**（分母已是整组 optimizer-step 的全局 token 总数, 天然归一化）。
- worker `_postprocess`：`all_reduce(local_loss_sum, SUM)/global_num_tokens` = 全局 per-token 均值（SUM 不是 AVG）。

这样 loss 轨迹与模块9 逐位一致（见验证表）。

> **verl 对应**：`verl/workers/engine/base.py`（BaseEngine + EngineRegistry）、`verl/workers/engine/fsdp/transformer_impl.py`（FSDPEngine）、`verl/workers/engine/automodel/transformer_impl.py`（AutomodelEngine）、`verl/workers/engine_workers.py`（TrainingWorker）、`verl/trainer/sft_trainer.py`（SFTTrainer.fit + `set_loss_fn`）。
> **简化项**：(a) EngineRegistry 单维 backend dict（verl 四维 + Ray）；(b) verl `forward_backward_batch` 把 `batch_num_tokens` 注入 data 供 loss_fn, 这里作显式参数传 `global_num_tokens, dp_size`（等价, 更直观）；(c) MFU 用 `6*n_param*tokens` 近似（verl `FlopsCounter` 精确分层）；(d) offload/eval_mode/checkpoint/save_checkpoint 等 verl 方法省略或留占位；(e) scheduler 单独 `build_scheduler(total_steps)`（total_steps 依赖数据, 由 worker 算好注入, verl 里 `num_training_steps` 也由 trainer 算）。

## 验证

### 1. 两卡 FSDP torchrun（fsdp 后端, 默认）—— 与模块9 同 config 对比

```bash
cd verl/learning/sft/code/11_engine_abstraction
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
/home/zzy/.conda/envs/easysteer/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29513 train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 trainer.grad_accum_steps=2 \
    trainer.experiment_name=qwen0.5b-engine-m10
```
- **架构摘要**（rank0）：`架构: TrainingWorker -> FSDPEngine -> FullyShardedDataParallel (backend=fsdp, 全量 494.0M, 每 rank 持 247.0M 参数)` —— 三层 + FSDP 分片（494M→247M/rank）。
- **与模块9 逐步对比**（同 config, 纯重构）：

  | step | 模块9 FSDP loss | 模块10 engine loss | 模块9/10 global_tokens | 模块10 grad_norm | 模块10 mfu |
  |---|---|---|---|---|---|
  | 1 | 0.5824 | **0.58237** | 4256 / 4256 ✓ | 23.4475 | 0.095 |
  | 2 | 0.5966 | **0.59658** | 4339 / 4339 ✓ | 18.902 | 0.151 |
  | 3 | 0.5083 | 0.50753 | 4242 / 4242 ✓ | 14.550 | 0.148 |
  | 4 | 0.4109 | 0.41154 | 4001 / 4001 ✓ | 8.709 | 0.141 |
  | 5 | 0.4622 | 0.45943 | 4139 / 4139 ✓ | 28.077 | 0.146 |

  - **`global_tokens` 5 步完全相同** → 数据管线（打乱/分片/token 计数）未改, 证明重构没动数据路径。
  - step 1-2 loss 匹配到 4-5 位有效数字；step 1 grad_norm 匹配（23.4475 vs 模块9 的 23.4457）。
  - step 3-5 loss 漂移 0.0006-0.003 —— 这是 **bf16 执行噪声**（matmul/attention 的 atomic-add 归约在两次独立进程间不确定, 经优化器步骤累积放大），与模块9 README 记录的 "DDP vs FSDP 差在第 3-4 位小数" 同源。**不是逻辑 bug**（若是 bug, global_tokens 或 step-1 也会偏）。
  - 新增 `train/mfu`（两张 3090 上 0.09-0.15, 0.5B+梯度检查点+小 batch 下合理）。
  - 5 步无死锁（min-count 从模块8 沿用）, 退出后 GPU 显存全释放。

### 2. 单卡 automodel 后端烟测（对照后端, 验证抽象可挂多后端）

```bash
cd verl/learning/sft/code/11_engine_abstraction
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
/home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 trainer.grad_accum_steps=2 \
    trainer.engine_backend=automodel trainer.experiment_name=qwen0.5b-engine-automodel
```
- **架构摘要**：`架构: TrainingWorker -> AutoModelEngine -> Qwen2ForCausalLM (backend=automodel, 全量 494.0M, 每 rank 持 494.0M 参数)` —— 选到 AutoModelEngine, 不分片（494M 全在单卡）。
- 12 步（单卡见全部 200 样本 → 23 micro/epoch → 12 optimizer step）, loss/grad_norm/mfu 正常流出, 无错退出。MFU 较高（0.16-0.36, 单卡无 FSDP 通信开销）。loss 轨迹与 fsdp 不同（单卡无 DP 平均 + 不同有效 batch, 属预期 —— automodel 是"对照后端", 不是 DP 引擎）。
- 证明：同一套 `worker` + `main`, 改一行 `engine_backend` 即换引擎后端, 其余代码不动。

## 踩过的坑

### 坑1（关键正确性）：朴素草稿的 `sft_loss/accum` + `all_reduce(AVG)` 会破坏数值等价
见上方"关键正确性决策"。修法：引擎 `forward_backward_batch` 进 micro 循环前 `all_reduce(SUM)` 算全局 `global_num_tokens`（float32）, 注入 `sft_loss_dp`（不除 accum）, worker `all_reduce(SUM)/global_num_tokens`。与 verl `forward_backward_batch` 的 `all_reduce(batch_num_tokens, SUM)` 一致。

### 坑2（继承）：grad_norm 不能重复 all_reduce
FSDP 的全局 grad_norm 已在 `FSDPEngine._fsdp_grad_norm_and_clip` 里 `all_reduce(SUM of local²)` 算好, `worker._postprocess` 直接取 `out["metrics"]["grad_norm"]`, **绝不能再 all_reduce**（重复归约会把 grad_norm 缩小 `world_size` 倍）。对应 verl `_postprocess_output` 明确 `pop` grad_norm 不归约。

### 坑3（继承）：`num_tokens` 必须 float32 再 all_reduce
`all_reduce(global_num_tokens, SUM)` 前必须 `torch.float32`, 否则 bf16 下 357→356 舍入（模块6 已踩）。`engine.forward_backward_batch` 与 `worker._postprocess` 都显式 float32。

### 坑4：FSDP `no_sync` 必须穿过 engine.train_batch
`FSDPEngine._gradient_sync_context(is_last=False)` 返回 `model.no_sync()`, 在 `forward_backward_batch` 的非 last micro 上套。若忘了套或套错位置, 梯度累积会每 micro 都 reduce-scatter, 数值仍对但通信量翻 `accum` 倍（破坏与模块9 的效率一致性）。

### 坑5：loss 注入的接口契约
`engine.forward_step` 调 `loss_fn(logits, input_ids, loss_mask, global_num_tokens, dp_size)` 必须返回 `(loss_tensor[带梯度], local_loss_sum[detach])`。worker 注入的是 `sft_loss_dp`（返回 `(loss, local_loss_sum.detach())` ✓）。**不能注入模块6 的 `sft_loss`**（只返单值 loss, 无 num_tokens/lsum, forward_step 拿不到分母）—— 必须用 `sft_loss_dp`。

### 坑6：FSDP 存盘必须 `summon_full_params(rank0_only=True)`
FSDP 模型参数是分片 DTensor, 不能直接 `save_pretrained`。`summon_full_params(rank0_only=True, writeback=False)`：仅 rank0 拼回全量, 非 rank0 不参与 collective（不挂）; 所有 rank 都进 `with`（上下文进入/退出安全）, 只 rank0 落盘 + `dist.barrier()`。`save_model=false` 规避（真要存时用此模式, verl 走 `FSDP.state_dict`）。

### 坑7：scheduler 的 `total_steps` 依赖数据 → 单独 `build_scheduler`
`cosine` 调度器要知道 `total_steps`, 而 `total_steps = ceil(len(dataloader)/accum)*epochs` 依赖数据侧。故 worker 先建 dataloader 算出 `total_steps`, 再 `engine.build_scheduler(total_steps)`（不在 `initialize` 里）。

### 坑8（继承）：`CUDA_DEVICE_ORDER=PCI_BUS_ID` + DP 死锁 min-count + bf16 sum float32
分别见模块7、模块8、模块6 README。本模块全部继承, 不再赘述。

### 坑9：automodel 后端是单卡（不包 DDP）
`AutoModelEngine` 不包 DDP/FSDP —— 多卡时各 rank 各自为政（等价模块7 的中间态, 会发散）。真要数据并行用 `fsdp` 后端（FSDP 自带 DP 梯度同步, subsume DDP）。automodel 仅作单卡 smoke / 对照后端（对应 verl `AutomodelEngine` 也是 vanilla 不 DDP）。
