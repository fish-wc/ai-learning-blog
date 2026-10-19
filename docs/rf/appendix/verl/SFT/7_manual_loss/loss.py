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
