# 模块 5：动态 batch (max_token_len_per_gpu)

> 在 `5_remove_padding` 基础上增量添加。对应 README 路线图【阶段二·模块5】。

## 这一模块加了什么

- `dataset.py`：
  - `SFTDataset` 改为 `__init__` 里**预 tokenize 并缓存**（原先是 `__getitem__` 惰性 tokenize），顺便得到每条长度 `self.lengths`。
  - 新增 `TokenBudgetBatchSampler`：按 token 预算切 batch——累积样本直到总 token 接近 `max_token_len_per_gpu` 就 yield 一个 batch。
- `config.py` 的 `TrainerConfig` 加 `max_token_len_per_gpu: int = 8192`（对应 verl 同名配置）。
- `config.yaml` 加 `max_token_len_per_gpu: 8192`。
- `train.py`：`remove_padding=true` 时 `DataLoader` 改用 `batch_sampler=TokenBudgetBatchSampler`（不再用固定 `micro_batch_size`）；每 epoch 调 `sampler.set_epoch(epoch)` 换打乱种子。`remove_padding=false` 时仍走固定 `micro_batch_size` + 右填充（对照路径不变）。

## 和上一模块(5_remove_padding)的区别

collator、4D 掩码、position_ids、梯度检查点全不变；只换了 **batch 怎么切**这一环：
- 模块4：固定 `micro_batch_size=8`（按**样本数**切），每批恰好 8 条。
- 模块5：按 **token 预算**切，每批样本数不固定，但 token 总量恒定。

实测（200 条 gsm8k，`max_token_len_per_gpu=2048`）：
```
每批样本数: [10, 8, 8, 7, 8, 9, 9, 11, 7, 8, 8, 8, 8, 9, 7, 8, ...]   # 7~11 不等
每批 token 数: [2021, 1914, 1929, 1889, 1931, 1923, 1799, 2019, ...]   # 都贴近 2048
```
样本数在 7-11 间漂移，但 token 总量始终贴着 2048 预算 —— 这就是 `max_token_len_per_gpu` 的效果。

## 这块的意义

**固定按样本数切的问题**：gsm8k 样本长度 129-480 不齐。按 `micro_batch_size=8` 切：
- 碰到一批短样本（8×130=1040 tokens）→ 显存大头空着、算力浪费。
- 碰到一批长样本（8×480=3840 tokens）→ 显存吃紧，还得右填充对齐更长。

**按 token 预算切的好处**：每个 batch 都把显存填到差不多的水平（≈`max_token_len_per_gpu`）→ **利用率恒定**，不再受样本长度不齐影响。这也是 `remove_padding` 的天然搭档：packing 能容纳任意大小的 batch，"按 token 切"才有意义；右填充模式下每批还得 pad 到本批最长，"按 token 切"就没了意义。

> **verl 对应**：`trainer.max_token_len_per_gpu`。verl 的 dataloader 按这个预算切，再配合 `flash_attn_varlen` 做 O(n) 变长注意力。本最小版用 `TokenBudgetBatchSampler`（贪心累积）+ 4D 块对角掩码达到同样的"每批 token 恒定"效果，思路一致、实现更朴素。

## 验证

```bash
cd verl/learning/sft/code/6_dynamic_bsz
# 200 样本 + 小预算(2048)让步数够看
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 \
    trainer.experiment_name=qwen0.5b-dynbsz-demo
```
- 控制台打印 `collator: packing + 动态 batch(... max_token_len_per_gpu=2048, 估算 23 个 micro-batch/epoch)`。
- loss 0.595→0.351（6 步），grad_norm 20.25→5.41，正常下降。
- 指标落 `outputs/logs/gsm8k-sft/qwen0.5b-dynbsz-demo.jsonl`。
- 对照（切回固定 batch）：`trainer.remove_padding=false`（走模块4 的右填充路径）。

## 验证动态 batch 行为(单独跑采样器)

```bash
CUDA_VISIBLE_DEVICES="" /home/zzy/.conda/envs/easysteer/bin/python -c "
import sys; sys.path.insert(0,'.')
from transformers import AutoTokenizer
from dataset import SFTDataset, TokenBudgetBatchSampler
tok = AutoTokenizer.from_pretrained('/home/zzy/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct', trust_remote_code=True)
ds = SFTDataset('/home/zzy/verl/learning/sft/data/train.parquet', tok, max_length=512, max_samples=200)
s = TokenBudgetBatchSampler(ds.lengths, 2048, shuffle=True, drop_last=True, seed=42)
sizes=[len(b) for b in s]; print('每批样本数', sizes)
"
```
输出每批样本数在 7-11 间漂移，证明 batch 大小确实在动态变化。
