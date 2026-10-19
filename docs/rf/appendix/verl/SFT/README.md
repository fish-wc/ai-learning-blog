# SFT 消融学习路线图

> 本目录是一份 **最最小、最原始** 的 SFT(Supervised Fine-Tuning) 实现，用来对照学习
> verl 框架里 SFT 的完整实现。verl 的 SFT 功能全面但工程层很厚，本路线图把它逐层拆开：
> 从你手上这份"只有骨架"的 v0 出发，**一个模块一个模块地加**，每一步都讲清楚
> "为什么需要加、要解决什么问题、对应 verl 哪个文件"。
>
> 目标：读完这份路线图，你能理解 verl SFT 每一层存在的必要性，而不是被一堆代码淹没。

---

## 0. 起点：本目录的 v0(最基础形态)有什么、没有什么

**已包含的最小成分（SFT 的本质骨架）：**

| 成分 | 本目录文件 | 做什么 |
|------|-----------|--------|
| 数据准备 | `prepare_data.py` | 下载 gsm8k → 拼 messages → 存 parquet |
| 数据集 + 掩码 | `dataset.py` | 读 parquet → apply_chat_template → input_ids + labels(-100) → 右填充 collator |
| 训练循环 | `train.py` | 加载模型 → AdamW → cosine 调度 → 梯度累积 → bf16 autocast → 梯度裁剪 → 存最终权重 |
| 效果测试 | `generate.py` | 微调前后各生成一遍 → 抠 `####` 答案 → 算正确率对比 |

**刻意没有的（verl 有、v0 砍掉的）成分：**

分布式(FSDP/DP)、嵌套张量 remove_padding、动态 batch、手算 log-prob 的 loss、
Hydra 配置、TensorDict 协议、CheckpointHandler、StatefulDataLoader 断点续训、
wandb 追踪、profiler、LoRA/QAT、torch.compile、多模态、引擎抽象层。

> **核心思想**：v0 只回答"SFT 到底在干什么"——
> 给模型看 `user` 的提问，让它在 `assistant` 回答的那些 token 上做 next-token 预测，
> 用交叉熵逼近日后能生成这样的回答。其余一切工程都是为了**更大、更快、更稳、更通用**地做这件事。

---

## 1. v0 的损失掩码是怎么工作的（务必先理解这步）

```
对话:  <|im_start|>system ... <|im_end|> <|im_start|>user 问题 <|im_end|> <|im_start|>assistant\n 回答 <|im_end|>
labels: -100 -100 ... -100               -100 ...           -100      回答的token  -100→真实id
                                                                                     (含 <|im_end|>)
```

- `input_ids`：整段对话的 token（system + user + assistant）。
- `labels`：和 input_ids 等长。**只有 assistant 的"回答内容 + 结尾 `<|im_end|>`"位置是真实 token id，
  其余全部是 `-100`**。
- HF 模型内部拿到 `labels` 后会自动 **左移一位** 做 next-token 预测，并对 `-100` 位置 `ignore_index` 不算 loss。

所以模型只学一件事：**在 assistant 该回答的地方，生成和标准答案一样的 token**。

> 对比 verl：verl 不用 `-100`，而是产出 `loss_mask`(0/1)，并在 `sft_loss` 里**手动**左移 mask、手动算
> `log_softmax(logits).gather` 再求和。两者数学等价，但 verl 的写法是为了和 RL(PPO) 共用 loss 接口、
> 并支持 remove_padding(模块 6 讲)。v0 用 HF 内置 `labels` 是"最原始"的形态。

---

## 2. 路线图：从 v0 逐步加到 verl 的程度

下面 5 个阶段、共 18 个模块。建议按顺序加，因为后面的模块往往依赖前面的。
每个模块格式：**解决什么问题 → 加什么 → verl 对应文件**。

### 阶段一：把"玩具脚本"工程化（可维护性）

#### 模块 1：结构化配置系统
- **解决什么问题**：v0 用 `argparse` 把超参硬塞进命令行，参数一多就乱、没法组合、没法继承默认值。
- **加什么**：用 dataclass / OmegaConf(Hydra) 定义配置 schema，支持 yaml 组合覆盖。
  比如把 `model / data / optim / trainer` 拆成几块 yaml，`defaults` 组合。
- **verl 对应**：`verl/trainer/config/sft_trainer_engine.yaml` + `verl/utils/config.py` 的 `omega_conf_to_dataclass`。
- ✅ **已实现** → `code/2_config_system/`（用 dataclass + PyYAML + CLI 覆盖实现配置系统本质；环境无 hydra/omegaconf，故未用 hydra，注释里对照说明了 hydra 额外提供组合/插值/multirun）。验证：覆盖生效、配置驱动训练跑通。

#### 模块 2：日志与实验追踪
- **解决什么问题**：v0 只 `print` loss，跑完就没了，没法回看曲线、没法对比多次实验。
- **加什么**：抽象一个 `Tracking` 类，把指标推到 console / wandb / tensorboard。
- **verl 对应**：`verl/utils/tracking.py`，`sft_trainer.py` 里 `tracking.log(data=metrics, step=...)`。
- ✅ **已实现** → `code/3_tracking/`（`tracking.py` 实现 console + file(jsonl) 后端，接口与 verl 一致）。验证：指标落 jsonl 文件可回看，含真实 grad_norm，见 `code/3_tracking/outputs/logs/`。

---

### 阶段二：显存与算力效率（让大模型也跑得动）

#### 模块 3：梯度检查点 (Gradient Checkpointing)
- **解决什么问题**：反向传播要存所有中间激活，模型一大显存就爆。v0 没开，0.5B 还行，3B+ 就吃紧。
- **加什么**：`model.gradient_checkpointing_enable()`，用重算换显存（省激活、多花一点时间）。
- **verl 对应**：`verl/trainer/config/model/hf_model.yaml` 的 `enable_gradient_checkpointing`。
- ✅ **已实现** → `code/4_grad_checkpoint/`（`config.py` 加 `enable_gradient_checkpointing` 字段，`train.py` 模型加载后调 `gradient_checkpointing_enable() + enable_input_require_grads()`）。验证：loss 0.609→0.332 与模块2 一致（重算路径数学等价），见 `code/4_grad_checkpoint/outputs/logs/`。

#### 模块 4：remove_padding / 嵌套张量 (Nested Jagged Tensor)
- **解决什么问题**：v0 用**右填充**把一个 batch 拉到等长。如果 batch 里有 1 条 500 token + 7 条 50 token，
  那 7 条都被填到 500，白白算/存 450 个 pad 位置——浪费算力和显存。
- **加什么**：用 `torch.nested.as_nested_tensor(..., layout=torch.jagged)` 把变长样本"打包"成嵌套张量，
  不做任何 padding；attention 走 `flash_attn_varlen` 按真实长度算。
- **verl 对应**：`verl/utils/dataset/dataset_utils.py` 的 `SFTTensorCollator.collate_variable_batch`；
  引擎里 `prepare_model_inputs` 对 `.values()` 拍平。**这是 verl 默认 SFT 模式(`pad_mode: no_padding`)**。
- ✅ **已实现** → `code/5_remove_padding/`（`dataset.py` 新增 `make_packing_collator`：拼成 `(1,T)` + `position_ids` 每样本重置 + 4D 块对角因果掩码；本环境无 flash_attn，退用 4D 掩码物化 O(T²) 但 0.5B/短序列够用，数学等价于 verl 的 `no_padding` 路径）。验证：loss 0.607→0.331 与右填充(0.609→0.332)一致 → 打包等价性通过；开发期 `/tmp` 隔离性 de-risk 测得跨样本 logits diff=0.00e+00，见 `code/5_remove_padding/outputs/logs/`。

#### 模块 5：动态 batch (max_token_len_per_gpu)
- **解决什么问题**：固定 `batch_size` 时：短样本 batch 浪费额度，长样本 batch 可能 OOM。
  理想是"每个 batch 的总 token 数尽量接近显存上限"，而不是"每 batch 固定条数"。
- **加什么**：按 `max_token_len_per_gpu` 动态决定每步塞多少条（token 总数守恒），配 `use_dynamic_bsz`。
- **verl 对应**：`sft_trainer_engine.yaml` 的 `data.use_dynamic_bsz` / `data.max_token_len_per_gpu`；
  引擎 `prepare_micro_batches` 里的切分逻辑。
- ✅ **已实现** → `code/6_dynamic_bsz/`（`dataset.py` 新增 `TokenBudgetBatchSampler` 按贪心累积 token 到预算即切 batch，`SFTDataset` 改预 tokenize+缓存以暴露 `lengths`；`train.py` packing 模式下用 `batch_sampler=` 取代固定 `micro_batch_size`）。验证：实测每批样本数 7-11 不等但 token 总数恒贴 2048 预算；loss 0.595→0.351，见 `code/6_dynamic_bsz/outputs/logs/`。

#### 模块 6：手算 log-prob + 统一 sft_loss
- **解决什么问题**：v0 直接 `model(..., labels=).loss`，简单但和 RL/PPO 的 loss 路径完全分叉，
  也无法在 remove_padding(嵌套张量)上直接用 HF 的 CE。
- **加什么**：自己取 `logits` → `logprobs_from_logits(logits, labels=左移后的input_ids)` 得每 token log-prob；
  再 `sft_loss = -masked_sum(log_prob, 左移的loss_mask) / batch_num_tokens`。
- **verl 对应**：`verl/workers/utils/losses.py` 的 `sft_loss`；`verl/utils/torch_functional.py` 的 `logprobs_from_logits`。
  > 这一步是把 SFT 和 RL 的"actor 更新"统一成同一个接口，是 verl 设计的关键。
- ✅ **已实现** → `code/7_manual_loss/`（新建 `loss.py`：`logprobs_from_logits` 用 logsumexp 融合省显存 + `sft_loss` 做 shift+mask 加权；`dataset.py` 把 `labels`(-100) 全量换成 `loss_mask`(0/1)；`train.py` 不再传 labels，用 `out.logits` 手算 loss）。验证：同一前向上 `sft_loss` 与 `F.cross_entropy` 逐位一致（diff=0.00e+00）；训练 loss 0.595→0.354，见 `code/7_manual_loss/outputs/logs/`。**踩坑记录**：`loss_mask.sum()` 不能在 bf16 里做（357 会舍入成 356），改 float32 后等价性才成立。

---

### 阶段三：分布式（从单卡到多卡/多机）

#### 模块 7：分布式进程组 (torch.distributed)
- **解决什么问题**：v0 单卡单进程。模型/数据一大，一张卡放不下或太慢。
- **加什么**：`initialize_global_process_group()`，建立全局 rank/world_size，`torchrun` 多进程启动。
- **verl 对应**：`verl/utils/distributed.py`，`sft_trainer.py` 的 `run_sft`。

#### 模块 8：数据并行 (DP)
- **解决什么问题**：多卡各跑一部分数据，但每个 batch 的 loss/梯度要跨卡平均才算"全局 batch"。
- **加什么**：`DistributedSampler` 切分数据；`all_reduce(loss, AVG)` / `all_gather` 跨 DP 组平均指标和梯度。
  注意 loss 里那个 `* dp_size` 就是为了配合"按 token 平均再 all-reduce 平均"得到正确全局均值。
- **verl 对应**：`sft_trainer.py` 的 `DistributedSampler`、`_get_batch_seqlens`；
  引擎 `forward_backward_batch` 里 `all_reduce(batch_num_tokens)`。

#### 模块 9：FSDP 模型分片
- **解决什么问题**：单卡放不下整个模型+优化器+梯度。DP 只是复制模型，没省显存。
- **加什么**：用 FSDP/FSDP2 把模型参数/梯度/优化器状态**分片**到各卡，前向时按需 all-gather 拼回、
  反向后 reduce-scatter 分走；还可 `param_offload`/`optimizer_offload` 卸到 CPU。
- **verl 对应**：`verl/workers/engine/fsdp/transformer_impl.py` 的 `FSDPEngineWithLMHead`，
  `_build_fsdp_module`、`_gradient_sync_context`(梯度同步延迟=梯度累积)。

#### 模块 10：引擎抽象层 (TrainingWorker / BaseEngine)
- **解决什么问题**：上面 FSDP 的 forward/backward/step 跟 Megatron/automodel 后端细节不同，
  但训练循环想要的接口是一致的(zero_grad→forward→backward→clip→step→lr_step)。
- **加什么**：抽 `BaseEngine`(定义 `train_batch` 骨架) + `TrainingWorker`(Ray worker，封装 loss 注入、
  指标后处理、MFU 计算)，后端用 `EngineRegistry` 选择(fsdp/fsdp2/megatron/automodel)。
- **verl 对应**：`verl/workers/engine/base.py`、`verl/workers/engine_workers.py`。
  > v0 的 train.py 主循环 = verl 的 `SFTTrainer.fit` + `TrainingWorker.train_batch` + `BaseEngine.train_batch` 三层压平后的样子。

---

### 阶段四：训练治理（长训练的稳定性）

#### 模块 11：CheckpointHandler
- **解决什么问题**：v0 只在最后存一次。长训练中断会全丢；想保留多个 checkpoint 回滚也不行。
- **加什么**：按 `save_freq` 周期存 ckpt(模型+优化器+调度器+步数)，`max_ckpt_to_keep` 滚动清理旧 ckpt，
  `resume_mode` 支持从指定路径恢复，并记 LoRA 元信息。
- **verl 对应**：`verl/utils/checkpoint.py` 的 `CheckpointHandler`。

#### 模块 12：StatefulDataLoader 断点续训
- **解决什么问题**：恢复训练时，不仅模型权重要恢复，**dataloader 也得恢复到中断的那一步**，
  否则会重复/漏数据。普通 DataLoader 没有"接着读"的能力。
- **加什么**：用 `torchdata.stateful_dataloader.StatefulDataLoader`，它的迭代状态可序列化、可恢复。
- **verl 对应**：`sft_trainer.py` 里 `StatefulDataLoader` + `resume_global_step` 那段注释。

#### 模块 13：验证循环
- **解决什么问题**：v0 只看 train loss 下降，看不出是否过拟合、泛化如何。
- **加什么**：按 `test_freq` 在 val 集上 `infer_batch`(只前向不算梯度)，记 `val/loss`。
- **verl 对应**：`sft_trainer.py` 的 `fit` 里 `is_valid_step` 那段验证逻辑。

#### 模块 14：TensorDict 数据协议
- **解决什么问题**：v0 用裸 dict 传 batch。当要同时带张量数据 + 一堆非张量元信息(use_remove_padding、
  pad_mode、global_token_num、lr_scheduler 标志……)时，普通 dict 跨分布式传递/访问很别扭。
- **加什么**：用 `TensorDict` + `NonTensorData` 统一装"张量+元信息"，跨进程传递、按 key 取用一致。
- **verl 对应**：`verl/utils/tensordict_utils.py`，`sft_trainer.py` 里 `tu.get_tensordict(...)`。
  > 这一层看着"虚"，但它是 verl 把 SFT/RL/多引擎统一的数据总线，少了它后面模块会处处特判。

---

### 阶段五：进阶能力（锦上添花 / 特定场景）

#### 模块 15：LoRA / QAT 高效微调
- **解决什么问题**：全量微调要更新所有参数。只想花很少参数适配新任务 → LoRA；量化训练 → QAT。
- **加什么**：模型构建时挂 LoRA adapter；checkpoint 存 adapter + 元信息(r/alpha/task_type)。
- **verl 对应**：`sft_trainer.py` 的 `_get_lora_train_meta`；模型构建里的 LoRA 分支。

#### 模块 16：torch.compile 编译加速
- **解决什么问题**：Python 解释开销 + 算子融合机会没吃满。
- **加什么**：`engine.use_torch_compile`，编译前向/反向图。
- **verl 对应**：`config/engine/fsdp.yaml` 的 `use_torch_compile`。

#### 模块 17：多模态 (processor / images / videos)
- **解决什么问题**：图文 SFT 时 input 不仅有文本 token，还有图像/视频。
- **加什么**：dataset 支持 `images`/`videos` 列，用 `processor` 而非纯 tokenizer，position_ids 走 RoPE 4D。
- **verl 对应**：`multiturn_sft_dataset.py` 里 `image_key`/`video_key`、`_build_messages` 的占位符替换。

#### 模块 18：逐轮 tokenize + system prompt 剥离
- **解决什么问题**：v0 用"前缀增量 tokenize"定位 assistant 区间(单轮够用、多轮 O(n²) 且有系统提示
  重复问题)。verl 的做法是**每条 message 单独 tokenize 再拼接**，并剥掉重复的系统提示词前缀，
  保证拼接结果 == 整段 tokenize 的结果(有 sanity check)。
- **加什么**：`extract_system_prompt_and_generation` 算出共享的 system 前缀和 generation 提示，
  逐条 message 渲染后去前缀、设 loss_mask。
- **verl 对应**：`verl/utils/dataset/multiturn_sft_dataset.py` 的 `_process_single_message`；
  `verl/utils/tokenizer/chat_template.py` 的 `extract_system_prompt_and_generation`。

---

## 3. 一句话串起来：verl SFT 为什么长那样

> v0 是"SFT 的内核"：**让模型在 assistant 回答的 token 上学 next-token 预测**。
> verl 的所有外层，都是为了让这个内核能 **跑更大的模型(阶段二/三)**、
> **跑得更稳更久(阶段四)**、**适配更多场景(阶段五)** 而加的工程脚手架。
> 你每加一个模块，就问一句"它解决了 v0 的哪个痛点"——这就是消融学习的意义。

---

## 4. 复现本目录 v0 的方法

```bash
# 环境: conda activate easysteer (verl 环境当前为空; 本代码只依赖 torch/transformers/datasets/pyarrow)
cd verl/learning/sft

# 1) 下数据(走 modelscope 镜像，因为服务器无外网)
python prepare_data.py

# 2) 训练(单卡，参数都很小)
CUDA_VISIBLE_DEVICES=0 python train.py

# 3) 看微调前后效果对比
CUDA_VISIBLE_DEVICES=0 python generate.py --num_questions 20
```

> 默认模型路径指向 modelscope 缓存里的 `Qwen/Qwen2.5-0.5B-Instruct`；
> 训练用 1000 条子集、2 epoch、等效 batch=32、lr=1e-4 cosine，约 60 步，单卡几分钟跑完。
