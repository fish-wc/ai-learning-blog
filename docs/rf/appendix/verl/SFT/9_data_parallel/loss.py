# Copyright 2024 Bytedance Ltd. and/or its affiliates (minimal-learning edition)
"""
手算 log-prob + 统一 SFT 损失 —— verl/learning/sft/code/7_manual_loss/loss.py

==================== 【新增·模块6】手算 log-prob + 统一 sft_loss ====================
在模块5 基础上继续改动:
  - 之前把 labels(-100) 直接喂进 model, 用 HF 内部的 out.loss(它内部做 shift + CrossEntropy)。
  - 本模块改为: model 只拿 input_ids(不算 loss, out.loss 为 None), 我们用 out.logits
    自己算 log 概率 + sft_loss。
意义:
  1. 摆脱 HF 内部 loss 的"黑盒": 自己做 shift、自己用 loss_mask 选位置、自己平均,
     每一步都看得见, 是理解 PPO/GRPO 里 "old_logprob / new_logprob" 的前置基础
     (RL 里要拿旧/新策略对同一批 token 的 log-prob 算 ratio, 那时根本没有 labels=-100, 全靠 loss_mask)。
  2. loss_mask(0/1) 比 labels(-100) 更通用: -100 是 CrossEntropyLoss 的约定, 只在"算 CE"时有用;
     而 log_prob + loss_mask 这套对 SFT/PTX/RL 都统一(RL 里 advantage * ratio * mask 的 mask 就是它)。
对应 verl: verl/utils/torch_dtypes.py 的 logprobs_from_logits, verl/trainer/.../sft_loss。

==================== 【新增·模块8】数据并行版 sft_loss_dp (*dp_size 技巧) ====================
在模块6 的 sft_loss 基础上新增 sft_loss_dp: 精确复刻 verl/workers/utils/losses.py:49 的
  loss = -masked_sum(logp, mask) / batch_num_tokens * dp_size
  - batch_num_tokens 是【全局】trained token 总数(跨 rank 跨 micro, 调用方先 all_reduce(SUM) 算好)
  - dp_size = world_size, *dp_size 抵消 DDP 反向时对梯度的 /world_size 平均 -> 等效梯度 = 真全局 per-token 均值
  - 不除 grad_accum(分母已是整组 optimizer-step 的全局 token 总数, 天然归一化)
保留模块6 的 sft_loss 不动(单卡/对照用)。对应 verl: verl/workers/utils/losses.py 的 sft_loss。
"""
import torch
import torch.nn.functional as F


# 【新增·模块6】logprobs_from_logits: 每个位置取"实际下一个 token"的 log 概率
def logprobs_from_logits(logits, labels):
    """log_softmax(logits)[labels] —— 但用 logsumexp 融合写法省显存。

    数学等价于:
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    区别: log_softmax 会物化完整 (..., V) 张量(词表 15w+ 时很大), 这里只物化 logsumexp
    (..., 1) 和 gather 出来的 (...,), 显存省一个数量级。verl 在长序列/大词表也走这条路。

    logits: (..., V), labels: (...) 都是"要预测的 token id"。
    返回: (..., ) 每个位置取真实 token 的 log 概率。注意本函数**不做 shift**, shift 由调用方做。
    bf16 输入时上转 float32 算 log, 数值更稳(log_softmax 在 bf16 下会丢精度)。
    """
    logits = logits.float()                                       # 上转 f32, 数值稳
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)     # (..., 1)
    gathered = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (...) 真实 token 的 logit
    return gathered - logsumexp.squeeze(-1)                       # logit - logsumexp = log_softmax[真实token]


# 【新增·模块6】sft_loss: 对"要学的位置"取交叉熵, 按 trained token 数平均
def sft_loss(logits, labels, loss_mask):
    """统一 SFT 损失: -(log_prob * loss_mask).sum() / loss_mask.sum()。

    对应 verl/trainer/.../sft_loss。和 HF 内部 out.loss 的结果数值等价:
      - HF: CrossEntropy(shift_logits, shift_labels, ignore_index=-100, reduction='mean')
            = -sum(log_softmax(shift_logits)[shift_labels] for 非忽略位) / 非忽略位数
      - 本函数: shift 后用 loss_mask 选位(等价于"非忽略位"), 同样的加权求和 / 计数。
      区别只在: HF 用 -100 当哨兵, 这里用 0/1 的 loss_mask 当哨兵 —— 后者对 RL/PTX 更通用。

    shift 细节: 模型做 next-token 预测, logits[t] 预测的是 token t+1, 所以
      shift_logits = logits[:, :-1, :]   # 用 0..T-2 的 logits
      shift_labels = labels[:, 1:]       # 预测 1..T-1 的 token
      shift_mask   = loss_mask[:, 1:]    # 这些预测位置里哪些要算 loss

    logits: (B, T, V), labels: (B, T)=实际 input_ids, loss_mask: (B, T) [0/1]。
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    # 【注意】mask 用 float32 而非 bf16: 求和(数 trained token 数)必须在 float32 里做,
    # 否则 bf16 下 357 这种数会被舍入成 356(bf16 尾数只有 7 位, 整数 >256 就不精确),
    # 分母错 0.3% -> loss 整体偏大 ~1e-3。logp 本身也是 float32, 相乘同 dtype 无额外开销。
    shift_mask = loss_mask[:, 1:].float()
    logp = logprobs_from_logits(shift_logits, shift_labels)       # (B, T-1) float32
    num_tokens = shift_mask.sum().clamp(min=1.0)                  # trained token 数, float32 精确
    return -(logp * shift_mask).sum() / num_tokens


# 【新增·模块8】数据并行版 sft_loss: 精确复刻 verl 的 *dp_size 技巧。
# 对应 verl/workers/utils/losses.py:49  loss = -masked_sum(logp, mask) / batch_num_tokens * dp_size
#
# 【和模块6 sft_loss 的区别】
# 模块6 的 sft_loss 返回"本 micro 的 per-token 均值"(= local_loss_sum / local_num_tokens),
# 每张卡各自 backward, DDP 把梯度 /world_size 平均 -> 得到的是【各 rank 均值的均值】(mean-of-means)。
# 当各 rank/micro 的 trained token 数不齐(动态 batch 下常态)时, mean-of-means ≠ 真全局 per-token 均值,
# 会引入偏差(长样本多的 rank 权重被低估)。
#
# 本函数改成 verl 的做法: 拿【全局】batch_num_tokens(跨 rank 跨 micro, 调用方先 all_reduce(SUM) 算好)
# 当分母, 再 *dp_size 抵消 DDP 反向时对梯度的 /world_size 平均 -> 等效梯度 = 全局 per-token 均值 loss 的梯度。
# 数学(含 grad_accum 的 no_sync 累积):
#   每 micro: loss_m = local_loss_sum_m / global_num_tokens * dp_size, backward(非最后 micro 进 no_sync)
#   本 rank 累积梯度 = (dp_size/global_num_tokens) * sum_m grad(local_loss_sum_m)
#   DDP 最后一步 all-reduce(avg=/dp_size) -> (1/global_num_tokens) * sum_rank sum_m grad(local_loss_sum_m)
#                                       = grad( 全局 loss_sum / global_num_tokens ) = 真全局 per-token 均值梯度 ✓
#
# 【不除 grad_accum_steps】分母 global_num_tokens 已是整组 optimizer-step(所有 micro+所有 rank)的 token 总数,
# 天然把"一组 micro 求和"归一化掉, 不需要再 /grad_accum(模块6 那样除是因为分母是单 micro 的)。
# 对应 verl sft_loss 同样不除 num_micro: 每个 micro 贡献 local_loss_sum/batch_num_tokens*dp_size,
# sum_micro 后 DDP avg = 全局均值。
#
# 关键: backward 只经过 local_loss_sum(本 rank 张量), **不经过 all_reduce**; 跨 rank 同步交给 DDP。
# 日志用的全局 loss 由调用方对 local_loss_sum.detach() 再 all_reduce(SUM) 算, 不能让反传图穿过 all_reduce。
def sft_loss_dp(logits, labels, loss_mask, global_num_tokens, dp_size):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = loss_mask[:, 1:].float()                          # float32! bf16 下 357->356 的坑(模块6)
    logp = logprobs_from_logits(shift_logits, shift_labels)        # (B, T-1) float32
    local_loss_sum = -(logp * shift_mask).sum()                    # 本 rank 本 micro 的 -sum(logp*mask), 带梯度
    # global_num_tokens 由调用方 all_reduce(SUM) 算好, 当【常量】进梯度图(不反传 through all_reduce)
    loss = local_loss_sum / global_num_tokens * dp_size
    return loss, local_loss_sum.detach()   # 第二个返回值供日志累积算全局 loss(已 detach, 不进反传图)
