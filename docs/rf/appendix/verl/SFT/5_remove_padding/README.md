## 凝练
packing拼序列省 padding，但样本会互相 attend
cu_seqlens记录每个样本的累计起止边界
flash_attn_varlen传边界，kernel自动切片
packing+varlen 配套：又省算力又不串题
packing拼序列省算力，但要配varlen一
用 cu_seqlens卡样本边界，否则串题



四步配置
面试问"packing后attention怎么办"，按这四步答
第1步，先做 packing，短序列拼接成定长块（如4096），按max_len切
第2步，算 cu_seqlens，每个块记录[o,len1,len1+len2,..]累计边界
第3步，调 varlen 接口，传 cu_seqlens + max_seqlen 给 flash_attn_varlen
第4步，校验 mask，可视化attention mask为块对角，确认无越界。

面试金句：packing把序列拼一起省padding，但必须配varlen attention，用cu_seqlens卡住样本边界，否则串题。

---

# 模块 4：remove_padding / 打包 (Packing)

> 在 `4_grad_checkpoint` 基础上增量添加。对应 README 路线图【阶段二·模块4】。

## 这一模块加了什么

- `dataset.py` 新增 `make_packing_collator()`（保留原 `make_collator` 右填充版作对照）：
  - 把一个 micro-batch 的 B 个变长样本**首尾相接**拼成单条 `(1, T)` 序列（`T = Σ各样本长度`），不再右填充。
  - 额外产出 `position_ids`：每个样本内从 0 重置（块对角注意力下，位置编码必须"各自为政"）。
  - 物化一个 `(1,1,T,T)` **4D 块对角因果注意力掩码**：块内下三角（可看自己及之前）、块间 `-inf`（互不可见）。`dtype=bfloat16`（必须等于 query dtype，否则 SDPA 报 `invalid dtype for bias`）。
- `config.py` 的 `TrainerConfig` 加 `remove_padding: bool = True`（对应 verl 的 `remove_padding` 开关）。
- `config.yaml` 加 `remove_padding: true`。
- `train.py` 按 `cfg.trainer.remove_padding` 选 collator；forward 时 packing 模式多传 `position_ids`，`attention_mask` 此时是 4D。

## 和上一模块(4_grad_checkpoint)的区别

训练循环、优化器、梯度检查点、追踪全不变；只换了**数据怎么喂进模型**这一环：
- 之前：B 个样本各自右填充到 `max_len` → `(B, max_len)`，pad 位是"废算"。
- 现在：B 个样本拼成 `(1, T)`，没有 pad，全部 token 有效。
loss 曲线和模块3 几乎一致（0.607→0.331 vs 0.609→0.332），证明两者数学等价，只是省了 pad 的浪费。

## 这块的意义

**右填充的浪费**：每个 micro-batch 取最长样本作 `max_len`，短样本 pad 一大堆。pad 位虽然 `label=-100` 不算 loss，但**前向照样算了一遍注意力/FFN**，白白烧算力和显存。样本长度越不齐、越长，浪费越大。

**打包(packing)的做法**：把变长样本首尾相接拼成长序列，用**块对角掩码**让各样本互不可见 → 等价于各自独立前向，但一个 batch 里没有任何 pad token。全部算力都花在真实 token 上。

**边界处 loss 为什么自动正确**：packed 序列里样本 i 末尾（`<|im_end|>`，`label=真实id`，要学）紧接样本 i+1 开头（`<|im_start|>system`，`label=-100`，不学）。HF 内部 loss 做 shift：`logits[t]` 预测 `label[t+1]`，于是边界处 `logits[样本i末]` 去预测 `label[样本i+1首]=-100` → 自动不算 loss，无需特判。

> **verl 对应**：`verl/utils/dataset/sft_tensor_collator.py` 的 `remove_padding=True` 分支。verl 用 **`flash_attn_varlen` + `cu_seqlens`** 做 O(n) 变长注意力（只喂每样本起始偏移，flash 内部分块，不物化 `(T,T)` 掩码）。本环境没装 flash_attn，退用 4D 块对角掩码（物化 O(T²)，但 0.5B + gsm8k 短序列 T≈1600 完全够，结果数值等价）。这就是 verl 里 `torch.nested.as_nested_tensor(..., layout=torch.jagged)` + `flash_attn_varlen` 那套东西的最小替身。

## 验证

```bash
# 用 easysteer env 的 python 直接跑(conda activate 在子 shell 里 PATH 没切干净)
cd verl/learning/sft/code/5_remove_padding
CUDA_VISIBLE_DEVICES=0 /home/zzy/.conda/envs/easysteer/bin/python train.py \
    data.max_train_samples=200 trainer.epochs=1 trainer.save_model=false \
    trainer.log_every=1 trainer.experiment_name=qwen0.5b-packing-demo
```
- 控制台打印 `collator: packing(remove_padding=true, 拼成 (1,T) + 4D 块对角掩码)`。
- loss 0.607→0.331，与模块3（右填充 0.609→0.332）一致 → 打包等价性验证通过。
- 指标落 `outputs/logs/gsm8k-sft/qwen0.5b-packing-demo.jsonl`。
- 切回右填充对照：`trainer.remove_padding=false`（走原 `make_collator`）。

## 已验证的隔离性(开发期 de-risk)

实现这个模块前，先用 `/tmp/test_packing2.py` 做了隔离性 de-risk：把 2 个样本拼进 `(1,T)`，用 4D 块对角掩码前向，对比"拼在一起跑" vs "各自独立跑"的 logits → 跨样本位置 diff 为 `0.00e+00`。证明块对角掩码确实让样本互不可见，数学等价于独立前向。（该测试脚本不在模块目录里，仅作开发期验证。）



