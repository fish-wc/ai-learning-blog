# 模块 6：手算 log-prob + 统一 sft_loss

> 在 `6_dynamic_bsz` 基础上增量添加。对应 README 路线图【阶段二·模块6】。

## 这一模块加了什么

- **新建 `loss.py`**：
  - `logprobs_from_logits(logits, labels)`：每个位置取"实际 token"的 log 概率。用 `logsumexp` 融合写法（`gathered - logsumexp`），不物化完整 `(B,T,V)` 的 `log_softmax`，省显存；bf16 输入上转 float32 算 log，数值更稳。对应 `verl/utils/torch_dtypes.py` 的 `logprobs_from_logits`。
  - `sft_loss(logits, labels, loss_mask)`：`-(log_prob * loss_mask).sum() / loss_mask.sum()`。内部做 shift（`logits[:, :-1]` 预测 `labels[:, 1:]`，mask 同步 shift）。对应 `verl/trainer/.../sft_loss`。
- **`dataset.py`**：
  - `build_input_and_labels` → **`build_input_and_loss_mask`**：返回 `loss_mask`（0/1）取代 `labels`（-100）。`loss_mask[i]=1` 表示 `input_ids[i]` 是要学的 assistant token。
  - `SFTDataset.__getitem__` 返回 `{input_ids, loss_mask}`（不再有 `labels`）。
  - 两个 collator 都拼 `loss_mask`（右填充补 0 / packing 直接拼）。
- **`train.py`**：forward **不再传 `labels`**（`out.loss` 为 None），改用 `out.logits` 手算：`raw_loss = sft_loss(out.logits, input_ids, loss_mask)`。metrics 用 `raw_loss.item()`。

## 和上一模块(6_dynamic_bsz)的区别

数据流（packing、动态 batch、4D 掩码、position_ids）、梯度检查点、追踪全不变；只换了 **loss 怎么算**这一环：
- 模块5：`labels`(-100) 喂进 model，用 HF 内部的 `out.loss`（黑盒：内部 shift + CrossEntropy + ignore_index=-100）。
- 模块6：`loss_mask`(0/1) 不喂 labels，自己 `logprobs_from_logits` + `sft_loss`（全透明）。

## 这块的意义

1. **摆脱 HF loss 黑盒**：shift、选位、平均每一步都看得见。`-100` 是 `CrossEntropyLoss` 的约定，只在"算 CE"时有用；`log_prob + loss_mask` 这套对 **SFT/PTX/RL 都统一**——RL 里 `advantage * ratio * mask` 的 `mask` 就是 `loss_mask`，提前换好为后续学 PPO/GRPO 铺路。
2. **是 RL 的前置基础**：PPO/GRPO 要拿"旧/新策略对同一批 token 的 log-prob"算 ratio，那时根本没有 `labels=-100`，全靠 `logprobs_from_logits(logits, input_ids)` + `loss_mask`。本模块就是把这套工具先在 SFT 上跑通。

> **verl 对应**：`verl/utils/torch_dtypes.py` 的 `logprobs_from_logits`；`verl/utils/dataset/multiturn_sft_dataset.py` 用 `loss_mask`(0/1) 不用 -100；SFT/PTX/RL 共用同一套 `sft_loss`。

## 等价性验证(关键)

在同一份 `out.logits` 上对比三种算法（不训练，纯前向）：

| 算法 | loss 值 |
|---|---|
| (A) `F.cross_entropy`（HF `out.loss` 等价路径） | 0.6092605591 |
| (B) 本模块 `sft_loss`（logsumexp 版） | 0.6092605591 |
| (C) `F.log_softmax` + gather 手算（对照） | 0.6092605591 |

`|A - B| = 0.00e+00` —— **手算 `sft_loss` 与 HF 内部 loss 逐位一致**，证明数学等价、实现正确。

### 踩过的坑：bf16 求和丢精度

初版 `sft_loss` 里 `shift_mask = loss_mask.to(shift_logits.dtype)` 把 mask 转成 bf16，于是 `shift_mask.sum()`（数 trained token 数）在 bf16 里算。但 **bf16 尾数只有 7 位，整数 >256 就不能精确表示**——357 个 trained token 会被舍入成 356，分母错 0.28% → loss 整体偏大 ~1.7e-3（初版 (B) 给 0.61097 而非 0.60926）。

修法：mask 用 **float32** 求和（`shift_mask = loss_mask[:, 1:].float()`）。修后 (B)==(A) 逐位一致。这是个很典型的"混合精度训练"坑：**累加/计数类操作别在 bf16 里做**，要么 float32 要么用整数 dtype。

## 验证

```bash
cd verl/learning/sft/code/7_manual_loss
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.max_token_len_per_gpu=2048 \
    trainer.experiment_name=qwen0.5b-manualloss-demo
```
- loss 0.595→0.354（6 步），grad_norm 20.25→5.16，正常下降。
- step1 loss 0.5949621200 与模块5（HF `out.loss`）的 0.5949620604 仅差 6e-8（kernel 级舍入）→ 手算 loss 与 HF 内部 loss 实质等价。
- 指标落 `outputs/logs/gsm8k-sft/qwen0.5b-manualloss-demo.jsonl`。

## 等价性自测脚本

```bash
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python -c "
import sys; sys.path.insert(0,'.')
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import SFTDataset, make_packing_collator
from loss import sft_loss
tok = AutoTokenizer.from_pretrained('/home/zzy/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct', trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained('/home/zzy/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct', dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
ds = SFTDataset('/home/zzy/verl/learning/sft/data/train.parquet', tok, max_length=512, max_samples=4)
raw = make_packing_collator()([ds[0],ds[1],ds[2],ds[3]])
batch = {k: v.to('cuda') for k, v in raw.items()}
labels_hf = torch.where(batch['loss_mask'].bool(), batch['input_ids'], torch.full_like(batch['input_ids'], -100))
with torch.no_grad():
    out = m(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], position_ids=batch['position_ids'])
    V = out.logits.size(-1)
    hf = F.cross_entropy(out.logits[...,:-1,:].reshape(-1,V).float(), labels_hf[...,1:].reshape(-1), ignore_index=-100, reduction='mean').item()
    man = sft_loss(out.logits, batch['input_ids'], batch['loss_mask']).item()
print(f'HF={hf:.10f}  手算={man:.10f}  diff={abs(hf-man):.2e}')
"
# 期望输出: HF=0.609... 手算=0.609... diff=0.00e+00
```
