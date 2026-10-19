阶段一两个模块都已完成并验证通过。

## ✅ 阶段一交付总览

按你的要求，每个模块一个独立可运行文件夹，**拷贝基础代码 + 增量添加 + `【新增·模块N】` 标注每处改动、对照说明与基础版的区别和意义**。

### 模块1：结构化配置系统 → [code/2_config_system/](verl/learning/sft/code/2_config_system/)
| 文件 | 改动 |
|---|---|
| [config.py](verl/learning/sft/code/2_config_system/config.py) | **新增** — dataclass 配置 schema（`ModelConfig/DataConfig/OptimConfig/TrainerConfig/SFTConfig`）+ 读 yaml + `key.path=value` CLI 覆盖 + `config_to_yaml_str` 打印生效配置 |
| [config.yaml](verl/learning/sft/code/2_config_system/config.yaml) | **新增** — 分组默认值（对应 verl `sft_trainer_engine.yaml`） |
| [train.py](verl/learning/sft/code/2_config_system/train.py) | 删 argparse → 改用 `load_config()`；`args.X`→`cfg.group.X`；开头打印生效配置；加 `save_model` 开关 |
| [generate.py](verl/learning/sft/code/2_config_system/generate.py) | 路径默认值改从 config 读（配置即"单一事实源"） |

**验证**：覆盖生效（`data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false`），类型正确转换（bool/int/float/list），配置驱动训练跑通 7 步。

### 模块2：日志与实验追踪 → [code/3_tracking/](verl/learning/sft/code/3_tracking/)
| 文件 | 改动 |
|---|---|
| [tracking.py](verl/learning/sft/code/3_tracking/tracking.py) | **新增** — 最小 `Tracking` 类（console + file jsonl 后端，`log()/finish()` 接口与 verl 一致） |
| [config.py](verl/learning/sft/code/3_tracking/config.py) | TrainerConfig 加 `project_name/experiment_name/logger` 字段 |
| [config.yaml](verl/learning/sft/code/3_tracking/config.yaml) | 加 `project_name/logger: [console, file]` |
| [train.py](verl/learning/sft/code/3_tracking/train.py) | `print(loss)`→`tracker.log({train/loss, train/lr, train/grad_norm}, step)`；捕获真实 grad_norm；末尾 `tracker.finish()` |

**验证**：指标同时落 console（打印）+ jsonl 文件。**可查看的输出文件** → [outputs/logs/gsm8k-sft/qwen0.5b-tracking-demo.jsonl](verl/learning/sft/code/3_tracking/outputs/logs/gsm8k-sft/qwen0.5b-tracking-demo.jsonl)，内容：
```json
{"step": 1, "ts": "...", "data": {"train/loss": 0.609, "train/lr": 1e-05, "train/grad_norm": 23.5, "train/epoch": 0.0}}
{"step": 6, "ts": "...", "data": {"train/loss": 0.332, "train/lr": 6e-05, "train/grad_norm": 7.25, "train/epoch": 0.0}}
```
grad_norm 23.5→7.25 递减、loss 0.61→0.33，训练在学。

## 关键设计决策

**为什么没用 hydra/omegaconf**：easysteer 环境没装（磁盘也紧）。我用 dataclass + PyYAML 手写配置系统的**本质**——这更符合消融"看本质"的精神，注释里也对照说明了 verl 的 hydra 额外提供什么（`defaults` 组合、`${}` 插值、multirun、`@hydra.main`）。环境不变的话，阶段三之后真要复刻 hydra 时再装也行（~10MB）。

## 复现命令
```bash
conda activate easysteer
cd verl/learning/sft/code/2_config_system   # 或 3_tracking
# 训练(小参数验证, 不存模型省磁盘)
CUDA_VISIBLE_DEVICES=0 python train.py data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false trainer.log_every=1
# 3_tracking 可多加: trainer.experiment_name=xxx  改 jsonl 文件名
```

README 路线图阶段一两个模块已标注 ✅ 实现进度并指向对应代码文件夹。**下一步是阶段二模块3：梯度检查点**（`model.gradient_checkpointing_enable()`），需要时我继续按同样方式做 `code/4_grad_checkpoint/`。