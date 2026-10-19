# 模块 3：梯度检查点 (Gradient Checkpointing)

> 在 `3_tracking` 基础上增量添加。对应 README 路线图【阶段二·模块3】。

## 这一模块加了什么

- `config.py` 的 `ModelConfig` 加字段 `enable_gradient_checkpointing: bool = True`（对应 verl 的 `model.enable_gradient_checkpointing`）。
- `config.yaml` 的 `model` 块加 `enable_gradient_checkpointing: true`。
- `train.py` 模型加载后：
  ```python
  model.gradient_checkpointing_enable()   # 内部设 use_cache=False
  model.enable_input_require_grads()      # 让输入嵌入需梯度(检查点配套)
  ```

## 和上一模块(3_tracking)的区别

只多了"开梯度检查点"这几行，训练循环、配置、追踪全不变。loss 曲线和模块2 几乎一致（0.609→0.332），说明重算路径数学等价、只是换了显存策略。

## 这块的意义

反向传播默认要**保存每一层的中间激活**，模型一大（3B+）激活存储就爆显存。
梯度检查点的做法：**反向时丢弃中间激活、需要时重算对应前向片段**，用一点时间换大幅显存（省激活存储，约省到 √层数量级）。
0.5B 不开也够，但这是大模型训练的标配开关，所以提前加上。

> **verl 对应**：`verl/trainer/config/model/hf_model.yaml` 的 `enable_gradient_checkpointing`，在引擎构建模型时同样调 `gradient_checkpointing_enable()`。

## 验证

```bash
conda activate easysteer
cd verl/learning/sft/code/4_grad_checkpoint
CUDA_VISIBLE_DEVICES=0 python train.py data.max_train_samples=200 trainer.epochs=1 \
    trainer.save_model=false trainer.log_every=1 trainer.experiment_name=qwen0.5b-gradckpt-demo
```
- 控制台打印 `已开启梯度检查点(用重算换激活显存)`。
- 指标落 `outputs/logs/gsm8k-sft/qwen0.5b-gradckpt-demo.jsonl`，loss/grad_norm 正常下降。
