全部完成！下面是最终交付报告。

## ✅ 交付总览

我在 [verl/learning/sft/](verl/learning/sft/) 放了一份**最最小、最原始的 SFT 实现**，并验证跑通。

### 文件结构

| 文件 | 作用 | 对应 verl |
|---|---|---|
| [prepare_data.py](verl/learning/sft/prepare_data.py) | 下载 gsm8k → 拼 messages → 存 parquet | `examples/data_preprocess/gsm8k_multiturn_sft.py` |
| [dataset.py](verl/learning/sft/dataset.py) | tokenize + 构造 labels(-100 掩码) + 右填充 collator | `MultiTurnSFTDataset` + `SFTTensorCollator` |
| [train.py](verl/learning/sft/train.py) | 模型+AdamW+cosine+梯度累积+bf16+裁剪+存权重 | `SFTTrainer.fit` + `FSDPEngine` 压平 |
| [generate.py](verl/learning/sft/generate.py) | 微调前后各生成一遍，抠 `####` 算正确率对比 | （verl 没有，专为"查看效果"加） |
| [README.md](verl/learning/sft/README.md) | **学习路线图**（18 模块/5 阶段，markdown 文档） | — |

**消融思路**：把 verl 的分布式(FSDP/DP)、嵌套张量 remove_padding、动态 batch、手算 log-prob、Hydra 配置、TensorDict、CheckpointHandler、StatefulDataLoader、wandb 全部砍掉，只留 SFT 内核——**让模型在 assistant 回答 token 上做 next-token 预测**，用 HF 内置 `labels`(-100) 让模型自动 shift+算 CE。

### 验证结果（都实测过了）

1. **掩码正确性**：非 `-100` 标签数 = `144 - 83 = 61`，恰好等于 assistant 回答 token 数；被学习的 token 解码出来正好是回答内容，`<|im_start|>assistant\n` 头部被正确排除 ✅
2. **训练跑通**：全量 7473 条 × 3 epoch = 702 步，loss 从 **0.679 → 0.067**，单卡几分钟跑完 ✅
3. **微调前后效果对比**（30 题）：

| | 微调前 | 微调后 |
|---|---|---|
| 正确率 | 3.3% (1/30) | **23.3% (7/30)** |
| 输出格式 | 啰嗦英文+LaTeX，不输出 `####` | 完整 gsm8k 格式 `<<calc>>` + `#### 答案<|im_end|>` |
| 提升 | — | **+20.0%（7 倍）** ✅ |

SFT 完美学会了回答格式与风格，并带来真实准确率提升。

### 学习路线图（README.md 核心）

路线图从 v0 出发，**逐个模块**加回 verl 的程度，每个都讲清"**解决什么问题 → 对应 verl 哪个文件**"：

- **阶段一 工程化**：① 配置系统(OmegaConf) ② 日志追踪(Tracking)
- **阶段二 显存/效率**：③ 梯度检查点 ④ remove_padding 嵌套张量 ⑤ 动态 batch ⑥ 手算 log-prob+sft_loss(与 RL 统一)
- **阶段三 分布式**：⑦ 进程组 ⑧ 数据并行 ⑨ FSDP 分片 ⑩ 引擎抽象层
- **阶段四 训练治理**：⑪ CheckpointHandler ⑫ StatefulDataLoader 断点续训 ⑬ 验证循环 ⑭ TensorDict 协议
- **阶段五 进阶**：⑮ LoRA/QAT ⑯ torch.compile ⑰ 多模态 ⑱ 逐轮 tokenize+系统提示剥离

## ⚠️ 环境说明（你指定的环境有问题，我做了替换）

| 问题 | 解决 |
|---|---|
| `conda activate verl` 环境是**空的**（没装 torch） | 改用 `easysteer` 环境（torch 2.10 + transformers 4.57 + datasets + modelscope 齐全） |
| **无外网**（huggingface.co 不可达） | 走 **modelscope.cn** 镜像下载 Qwen2.5-0.5B 和 gsm8k |
| `/home` 分区 **100% 满**（815G） | 清了 pip 缓存腾出 8.7G；checkpoint 只存权重不存优化器 |

> 代码本身是独立的（只用 torch/transformers/datasets 标准库，**不依赖 verl 包**），所以换环境不影响。复现命令见 [README.md](verl/learning/sft/README.md) 末尾。

## 如何运行

```bash
cd verl/learning/sft
/home/zzy/.conda/envs/easysteer/bin/python prepare_data.py          # 下数据
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py          # 训练
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python generate.py --num_questions 30   # 看效果
```

如果你想继续推进路线图里某个模块（比如先加 **remove_padding** 或 **FSDP 分布式**），告诉我从哪个开始，我帮你在 v0 基础上增量实现并讲清楚。