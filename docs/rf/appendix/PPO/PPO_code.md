# 从经典 PPO 到 LLM PPO：读懂五个核心变量、KL 惩罚、Flatten 与 Token 对齐

> **系列：强化学习与大模型后训练学习笔记**
>
> **Day 6：别读论文，读代码**
>
> 关键词：Policy Gradient、PPO、RLHF、TRL、KL Divergence、GAE、Causal Attention、Tensor Shape

## 一、前言：为什么我要研究 LLM PPO？

在学习强化学习的过程中，我发现一个非常有意思的问题。

经典 PPO（Proximal Policy Optimization）通常处理的是游戏环境。例如 CartPole：

```text
state
  ↓
Actor
  ↓
action
  ↓
Environment
  ↓
reward
```

但是当 PPO 被应用到大语言模型（LLM）时，情况似乎发生了变化。

因为语言模型的动作是生成 Token，而且下一个 Token 的预测依赖于之前所有的 Token。

这让我产生了几个问题：

1. 经典 PPO 中可以将 `[batch, timestep]` 展平（Flatten），LLM PPO 是否也可以？
2. 如果展平 Token，Transformer 会不会失去时序信息？
3. LLM PPO 中的 `old_log_probs`、`rewards`、`values`、`advantages` 和 `policy_loss` 分别代表什么？
4. 为什么 RLHF PPO 还要额外引入 KL Penalty？它和 PPO Clipping 有什么区别？
5. 为什么 TRL 的源码经常出现 `context_length - 1 : -1`，而不是直接从 `context_length` 开始？

这些问题看似独立，实际都指向同一个核心：

**如何将大语言模型的自回归生成过程，转化成强化学习中的 State、Action、Reward 和 Value？**

本文将从代码与 Tensor Shape 的角度，把这些问题串起来。

---

## 二、先理解 LLM 中的 State 和 Action

在分析代码之前，我们需要明确一个最基本的问题：

**大语言模型进行强化学习时，什么是 State？什么是 Action？**

### 2.1 经典强化学习

以 CartPole 为例，某个时间步的状态可能是：

```python
state = [
    cart_position,
    cart_velocity,
    pole_angle,
    pole_velocity,
]
```

Actor 根据这个状态选择动作：

```python
action = actor(state)
```

这里，每个 State 本身就是一条完整的 Observation。

因此，我们通常可以直接将多个 State 组成 Batch，送入 Actor。

### 2.2 LLM 中的 State

假设用户输入：

```text
中国的首都是
```

模型生成：

```text
北京。
```

为了方便理解，假设 Response 被分成三个 Token：

```text
y0 = 北
y1 = 京
y2 = 。
```

那么整个生成过程可以被拆成三个强化学习时间步。

**第一个时间步：**

```text
State:
中国的首都是

Action:
北
```

**第二个时间步：**

```text
State:
中国的首都是 北

Action:
京
```

**第三个时间步：**

```text
State:
中国的首都是 北 京

Action:
。
```

因此，对于 LLM：

$$
s_t=(x,y_{<t})
$$

$$
a_t=y_t
$$

其中：

* \(x\)：Prompt。
* \(y_{<t}\)：在第 \(t\) 个 Token 之前已经生成的 Token。
* \(s_t\)：生成当前 Token 之前的完整上下文。
* \(a_t\)：当前选择的 Token。

策略则表示为：

$$
\pi_\theta(a_t|s_t)
=P_\theta(y_t|x,y_{<t})
$$

这意味着：

**LLM 的 State 不是当前 Token，而是 Prompt 加上之前所有已经生成的 Token。**

这是后面理解 Flatten、Causal Mask 和 Value 对齐的前提。

---

## 三、LLM PPO 的五个核心变量

我最初学习 PPO 时，重点追踪的是五个变量：

```text
old_log_probs
rewards
values
advantages
policy_loss
```

但它们并不是简单的线性依赖关系。

更准确的数据流是：

```text
                 Rollout
                    │
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
    Old Policy                 Critic
       │                         │
       ▼                         ▼
 old_log_probs                values
       │                         │
       │                     rewards
       │                         │
       │                         ▼
       │                    advantages
       │                         │
       │                         │
       │   Current Policy        │
       │          │              │
       │          ▼              │
       └─── new_log_probs        │
                  │              │
                  ▼              │
                ratio            │
                  │              │
                  └──────┬───────┘
                         ▼
                 PPO Clipped Loss
                         │
                         ▼
                    policy_loss
```

### 3.1 首先建立 Tensor Shape

定义：

| 符号  | 含义                                 |
| --- | ---------------------------------- |
| `B` | Batch Size，即一批 Prompt/Response 的数量 |
| `P` | Prompt 的填充后长度                      |
| `R` | Response 的填充后长度                    |
| `S` | 完整序列长度，`S = P + R`                 |
| `V` | Vocabulary Size                    |
| `H` | Hidden Size                        |

那么典型的 Token-Level LLM PPO 中：

| 变量                | Shape    | 含义                                           |
| ----------------- | -------- | -------------------------------------------- |
| `queries`         | `[B, P]` | Prompt Token IDs                             |
| `responses`       | `[B, R]` | 旧策略生成的 Response                              |
| `query_responses` | `[B, S]` | Prompt + Response                            |
| `old_log_probs`   | `[B, R]` | 采样时每个 Response Token 的 Log Probability       |
| `ref_log_probs`   | `[B, R]` | Reference Policy 对同一 Token 的 Log Probability |
| `rewards`         | `[B, R]` | 每个时间步的训练奖励                                   |
| `values`          | `[B, R]` | Critic 对各个决策状态的价值预测                          |
| `advantages`      | `[B, R]` | 每个 Token Action 的优势估计                        |
| `new_log_probs`   | `[B, R]` | 当前 Actor 对旧 Response 的重新评分                   |
| `ratio`           | `[B, R]` | 当前策略与采样策略的概率比                                |
| `policy_loss`     | `[]`     | 最终用于反向传播的标量损失                                |

注意：

经典 PPO 中，CleanRL 常用 `[T, B]` 存储 Rollout；LLM PPO 更常看到 `[B, R]`。

**维度顺序并不是算法的本质区别，真正重要的是每个维度代表什么。**

### 3.2 old_log_probs：当时为什么选这个 Token？

假设在某个 State 下，旧策略认为：

```text
P(北 | 中国的首都是) = 0.6
```

那么：

```python
old_log_prob = log(0.6)
```

注意，`old_log_probs` 保存的不是整个 Vocabulary 上的概率分布，而是实际采样 Token 对应的对数概率。

对于一个 Response：

```text
北 京 。
```

可以对应：

```text
Token:        北       京       。
old_logp:    log p0   log p1   log p2
```

因此：

```text
old_log_probs.shape = [B, R]
```

为什么需要保存它？

因为 PPO 更新 Actor 时，要知道当前策略相对于采样策略发生了多大变化。

### 3.3 rewards：模型获得了多少反馈？

经典强化学习中，奖励由环境提供。

在常见的 RLHF PPO 中，奖励主要来自两部分：

1. Reward Model 对回答的评价。
2. 当前策略相对于 Reference Model 的 KL 惩罚。

Reward Model 通常可以给整条 Response 一个分数：

```text
score.shape = [B]
```

但 PPO 需要计算每个 Token 时间步的 Advantage。

因此，训练过程中通常会构造：

```text
rewards.shape = [B, R]
```

后面会详细解释它如何构造。

### 3.4 values：Critic 认为这个状态值多少钱？

Critic 学习的是状态价值函数：

$$
V_\phi(s_t)
\approx
\mathbb{E}\left[
\sum_{k=t}^{T-1}\gamma^{k-t}r_k
\mid s_t
\right]
$$

简单理解：

> 当前已经生成了这些 Token，接下来继续生成，我预计还能获得多少累计奖励？

例如：

```text
State:
中国的首都是 北
```

Critic 预测：

```text
V(state) = 0.8
```

这不是当前 Token 的即时 Reward，而是从当前 State 出发的未来回报预测。

因此：

```text
values.shape = [B, R]
```

每个 Response 时间步对应一个 State Value。

### 3.5 advantages：这个 Token 是否值得鼓励？

优势函数的理论定义是：

$$
A^\pi(s_t,a_t)
=
Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

它衡量的是：

> 在当前 State 下，选择这个 Action，相比该 State 的平均预期表现好多少？

因此：

```text
Advantage > 0
```

意味着应该倾向于提高这个 Token Action 的概率。

```text
Advantage < 0
```

意味着应该倾向于降低这个 Token Action 的概率。

实际 PPO 通常使用 GAE（Generalized Advantage Estimation）估计 Advantage。

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t
=
\delta_t+\gamma\lambda A_{t+1}
$$

这里先写的是非终止时间步的形式；真实代码需要在 Episode 结束时停止 Bootstrap 和 GAE 递推。

所以：

```text
rewards + values + next_values
                ↓
               GAE
                ↓
            advantages
               [B,R]
```

Advantage 已经计算完成后，Actor 才能利用它判断应该增加还是减少某个 Token 的概率。

---

## 四、LLM PPO 到底能不能 Flatten？

这是我从经典 PPO 迁移到 LLM PPO 时，遇到的第一个重要问题。

答案是：

> **可以 Flatten，但必须区分计算发生在哪个阶段。**

### 4.1 为什么经典 PPO 可以直接 Flatten State？

在 CartPole 中，假设：

```text
states.shape = [T, B, obs_dim]
```

可以直接：

```python
states = states.reshape(T * B, obs_dim)
```

因为每条 Observation 本身已经完整描述了当前 State。

Actor 不需要再根据其他时间步重建它。

### 4.2 为什么 LLM 不能直接 Flatten 原始 Token？

对于 LLM：

```text
State 0 = Prompt

State 1 = Prompt + y0

State 2 = Prompt + y0 + y1
```

如果直接：

```python
responses.reshape(-1)
```

然后把每个 Token 当作独立 State 送入 Transformer，就会丢失对应的 Prefix。

例如，要计算：

$$
P(y_2|x,y_0,y_1)
$$

但我们只提供：

```text
y2
```

显然无法正确计算。

### 4.3 Transformer 是怎么解决这个问题的？

关键在于 **Causal Attention**。

我们无需显式构造：

```text
state0 = prompt

state1 = prompt + y0

state2 = prompt + y0 + y1
```

并分别执行三次完整前向传播。

而是直接输入：

```text
prompt + y0 + y1 + y2
```

假设：

```text
input_ids.shape = [B, S]
```

Transformer 输出：

```text
hidden_states.shape = [B, S, H]
```

由于 Causal Mask 的存在，每个位置只能看到自己以及此前的 Token。

因此，经过正确的位置对齐后，一次 Forward 就能同时计算：

```text
P(y0 | prompt)

P(y1 | prompt, y0)

P(y2 | prompt, y0, y1)
```

这与自回归模型逐 Token 生成时使用的条件概率是一致的。

**Causal Transformer 帮助我们并行计算了多个 State 下的 Action Probability。**

### 4.4 new_log_probs 不是重新生成 Response

这里还有一个很容易理解错的地方。

PPO Update 阶段，Actor 并不会重新采样一条新的 Response，再拿新 Response 与旧 Response 比较。

它使用的是原来的：

```text
prompt + old_response
```

然后在当前策略下重新计算旧 Token 的概率。

这一过程可以理解为 Teacher Forcing。

```text
Rollout：

Old Actor
    ↓
逐 Token 采样
    ↓
Response = [y0,y1,y2]
    ↓
保存 old_log_probs
```

```text
Update：

Prompt + 原 Response
        ↓
   Current Actor
        ↓
重新计算每个旧 Token 的 logprob
        ↓
    new_log_probs
```

因此：

```text
new_log_probs.shape = [B, R]
```

只有固定相同的 State-Action 对，才能计算 PPO 所需的概率比。

### 4.5 三个阶段，Flatten 的规则不同

| 阶段                       | 能否直接 Flatten Token 维 | 原因               |
| ------------------------ | -------------------- | ---------------- |
| Transformer Forward 之前   | 通常不能                 | 需要 Prefix 上下文    |
| GAE 计算之前                 | 不能直接打乱               | 需要沿时间维递推，并处理终止边界 |
| PPO Per-Token Loss 已经计算后 | 可以                   | 每个位置的训练信号已经准备完成  |

对于 GAE，如果直接将两条 Response 拼接后做连续递推：

```text
Response A:
a0 a1 a2

Response B:
b0 b1 b2
```

错误地当成：

```text
a0 a1 a2 b0 b1 b2
```

就可能使 Response B 的未来奖励流入 Response A。

这显然不是我们想要的。

因此，GAE 必须正确保留每条轨迹的终止信息。

而当我们已经得到：

```text
old_log_probs [B,R]
new_log_probs [B,R]
advantages    [B,R]
response_mask [B,R]
```

就可以计算：

```python
ratio = torch.exp(
    new_log_probs - old_log_probs
)

loss1 = -advantages * ratio

loss2 = -advantages * torch.clamp(
    ratio,
    1 - epsilon,
    1 + epsilon,
)

per_token_loss = torch.maximum(loss1, loss2)

policy_loss = per_token_loss[response_mask].mean()
```

因为此时各个 Token 的 State 信息已经被用于计算 Log Probability 和 Advantage。

我们只是在对训练损失做聚合。

这里需要特别记住：

**Flatten 本身不是问题；问题是 Flatten 后，后续计算是否仍然拥有它需要的上下文和时间边界。**

---

## 五、为什么 logits 和 values 都要使用 `context_length - 1 : -1`？

这个索引是我认为 LLM PPO 最值得彻底理解的细节。

### 5.1 先构造一个例子

假设：

```text
Prompt:
q0 q1 q2 q3

Response:
y0 y1 y2
```

因此：

```text
P = 4
R = 3
S = 7
```

完整输入：

```text
Index:    0  1  2  3  4  5  6
Token:   q0 q1 q2 q3 y0 y1 y2
```

现在的问题是：

**哪个位置的 Hidden State 对应生成 y0 之前的状态？**

### 5.2 State 与 Hidden State 的关系

因为 Causal Attention：

```text
h0:
看到 q0

h1:
看到 q0 q1

h2:
看到 q0 q1 q2

h3:
看到 q0 q1 q2 q3

h4:
看到 q0 q1 q2 q3 y0

h5:
看到 q0 q1 q2 q3 y0 y1

h6:
看到 q0 q1 q2 q3 y0 y1 y2
```

而 RL State 定义为：

```text
s0 = prompt

s1 = prompt + y0

s2 = prompt + y0 + y1
```

所以：

```text
s0 → h3
s1 → h4
s2 → h5
```

推广一下：

$$
h_{P+t-1}
\leftrightarrow
s_t
$$

这意味着：

```text
values[:, 0]
```

应该对应：

```text
ValueHead(h3)
```

而不是：

```text
ValueHead(h4)
```

因为 `h4` 已经看到了 `y0`，表示的是下一个 State。

### 5.3 logits 为什么也要提前一个位置？

因为 Causal Language Model 做的是 Next Token Prediction。

模型在位置 `i` 的 Logits 用来预测位置 `i+1` 的 Token。

也就是说：

```text
logits[3] → 预测 y0

logits[4] → 预测 y1

logits[5] → 预测 y2

logits[6] → 预测 Response 后面的下一个 Token
```

PPO 的 Action 只有：

```text
y0 y1 y2
```

所以只需要：

```text
logits[3]
logits[4]
logits[5]
```

对应 Python Slice：

```python
logits[:, 3:6]
```

也就是：

```python
logits[:, context_length - 1 : -1]
```

注意 Python 切片右边界不包含在内。

所以：

```text
3:-1
```

在这里选中：

```text
3, 4, 5
```

### 5.4 为什么 values 也使用相同的 Slice？

对于每个时间步，我们需要：

```text
State: s_t

Action: y_t

Value: V(s_t)

Log Probability: log π(y_t | s_t)
```

而：

```text
ValueHead(h3)
```

和：

```text
LMHead(h3)
```

恰好都基于相同的决策 State：

```text
完整 Prompt
```

区别只是输出目标：

```text
                   h3
                    │
             Prompt State
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
       LM Head             Value Head
          │                   │
          ▼                   ▼
   P(y0 | Prompt)         V(Prompt)
```

所以相同的索引可以同时用于 Actor 和 Critic。

完整对应关系如下：

| 位置    | 对应的 State      | Value   | Logits 预测的 Action |
| ----- | -------------- | ------- | ----------------- |
| `P-1` | `prompt`       | `V(s0)` | `y0`              |
| `P`   | `prompt+y0`    | `V(s1)` | `y1`              |
| `P+1` | `prompt+y0+y1` | `V(s2)` | `y2`              |

这是理解 LLM Actor-Critic 的关键：

**同一个决策位置的 Hidden State，可以通过不同的 Head 分别输出 Policy 和 Value。**

需要说明的是，实际工程中 Actor 和 Critic 也可以使用不同 Backbone。上面的图表达的是位置与语义上的对齐关系，并不要求两个模型一定共享 Hidden State。

### 5.5 对应的 PyTorch 代码

```python
# 假设：
# query_response: [B, P + R]
# context_length = P

# 将完整序列送入 Actor
output = actor(query_response)

# 整个 Vocabulary 上的预测分数
# Shape: [B, P + R, V]
full_logits = output.logits

# 截取每个 Response Action 对应的决策位置
# Shape: [B, R, V]
response_logits = full_logits[
    :, context_length - 1 : -1, :
]

# Response Token IDs
# Shape: [B, R]
responses = query_response[:, context_length:]

# 从 Vocabulary 中取出实际 Action 的 logprob
# Shape: [B, R]
new_log_probs = torch.log_softmax(
    response_logits, dim=-1
).gather(
    dim=-1,
    index=responses.unsqueeze(-1)
).squeeze(-1)
```

Value 的对齐：

```python
# Critic 对完整输入的价值预测
# Shape: [B, P + R]
full_values = critic(query_response)

# 每个 Action 执行之前的 State Value
# Shape: [B, R]
values = full_values[
    :, context_length - 1 : -1
]
```

这里假设 Critic 已输出 `[B, S]` 的标量预测。如果输出是 `[B, S, 1]`，还需要去掉最后一维。

上述 Slice 成立的前提是：Prompt 区域具有统一的物理长度 `P`，并正确处理了 Attention Mask 与 Position IDs。

在存在 Left Padding、变长 Response、EOS 和截断的真实训练中，还必须进一步处理有效位置和终止边界。

---

## 六、KL Penalty 是什么？为什么需要它？

现在回到 Reward。

这是 LLM PPO 和简单游戏环境 PPO 的另一个重要区别。

### 6.1 为什么只最大化 Reward Model 分数不够？

假设有一个经过 SFT 的模型：

```text
Reference Policy
```

它已经具备不错的语言能力。

然后，我们使用 Reward Model 对生成回答打分。

PPO 的目标是找到获得更高奖励的回答。

但问题是：

**Reward Model 并不是完美的人类偏好函数。**

如果模型过度优化 Reward Model，就可能找到评分器的漏洞。

例如，模型可能学会使用一些更容易获得高分、却未必真正改善回答质量的表达模式。

这种现象被称为 Reward Hacking。

因此，我们希望模型提高 Reward 的同时，不要无限偏离原本的语言分布。

### 6.2 Reference Policy 是什么？

Reference Policy 通常是 RL 训练开始前冻结的 SFT 模型副本。

记为：

$$
\pi_{\mathrm{ref}}
$$

正在更新的 Actor：

$$
\pi_\theta
$$

Reference Policy 不负责直接评价回答质量。

它主要用于提供一个概率分布上的参考基准。

于是，典型的 KL 正则化优化目标可以写成：

$$
\max_\theta\;
\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}
\left[
R_{\mathrm{RM}}(x,y)
\right]
-
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|x)
\|
\pi_{\mathrm{ref}}(\cdot|x)
\right)
$$

其中：

* \(R_{\mathrm{RM}}\)：Reward Model 的评分。
* \(D_{\mathrm{KL}}\)：衡量两个分布的差异。
* \(\beta\)：KL 惩罚系数。

直观理解：

```text
Reward Model:
希望模型获得更高的回答评分

KL Penalty:
不希望模型为了追求评分而偏离参考策略太远
```

### 6.3 为什么不直接计算整个 Vocabulary 的 KL？

理论上的单步 KL Divergence 是：

$$
D_{\mathrm{KL}}
(\pi_\theta\|\pi_{\mathrm{ref}})
=
\sum_a
\pi_\theta(a|s)
\log
\frac{\pi_\theta(a|s)}
{\pi_{\mathrm{ref}}(a|s)}
$$

这里需要考虑该 State 下整个 Vocabulary 的概率分布。

而在 PPO Rollout 中，我们已经从当前策略采样了 Token。

因此，一种常见的做法是使用 Sampled KL Estimator。

对于实际采样 Token：

$$
k_t
=
\log\pi_{\mathrm{old}}(y_t|s_t)
-
\log\pi_{\mathrm{ref}}(y_t|s_t)
$$

这里使用 \(\pi_{\mathrm{old}}\)，因为本轮奖励是根据 Rollout 时的策略和采样数据构造的。

它是 KL 的单样本估计量；对来自该策略的 Action 取期望时，得到相应 State 下的 Forward KL。

于是：

```python
sampled_kl = old_log_probs - ref_log_probs
```

Shape：

```text
[B, R]
```

再构造 KL Reward：

```python
kl_reward = -beta * sampled_kl
```

### 6.4 一个简单的数值例子

假设：

```text
Old Policy:
P(action) = 0.6

Reference Policy:
P(action) = 0.3
```

那么：

$$
k_t
=
\log\frac{0.6}{0.3}
=
\log 2
\approx 0.6931
$$

如果：

$$
\beta=0.1
$$

那么：

$$
r_t^{KL}
=
-0.1\times0.6931
\approx-0.0693
$$

这个 Token 在本次采样中的 KL Reward 为负。

但需要特别注意：

**单个 Token 的 Sampled KL 可以为负，而真正的 KL Divergence 非负。**

因为这里计算的是单样本估计，而不是对整个 Action 分布求期望之后的结果。

TRL 中还可以看到其他 KL 估计器，例如 `k3`。本文主要使用最容易理解的 `k1` 形式。

### 6.5 KL Reward 如何和 Reward Model Score 结合？

假设生成：

```text
北 京 。
```

Reward Model 对整条回答给出：

```text
score = 1.0
```

那么一种简化的 Token-Level Reward 构造方式是：

```text
Token:      北         京         。
Reward:   -βKL0      -βKL1      -βKL2 + 1.0
```

最终：

```text
rewards.shape = [B, R]
```

这样，前面的 Token 虽然没有直接获得 Reward Model 的最终评分，但 GAE 可以沿时间向前传递后续奖励信号。

这就是序列生成中的 Credit Assignment（信用分配）。

**工程注意：** 上面的写法是便于理解的教学版本。实际 TRL 实现需要额外处理 EOS、PAD、末端 Value 和 Reward 放置位置；特定版本可能在终止边界位置放置 Score，并使用额外的 Padding Mask。不能在不了解终止边界定义的情况下机械地把 Score 加到某个固定索引。

---

## 七、KL Penalty 和 PPO Clipping 有什么区别？

这是面试中非常容易混淆的两个概念。

### 7.1 PPO Clipping

PPO 更新时计算：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\mathrm{old}}(a_t|s_t)
}
$$

也就是：

```python
ratio = torch.exp(
    new_log_probs - old_log_probs
)
```

PPO 的 Clipped Objective 是：

$$
L_t^{CLIP}
=
\min
\left(
r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
$$

通常在 PyTorch 中转成最小化形式：

```python
unclipped_loss = -ratio * advantages

clipped_loss = (
    -torch.clamp(ratio, 1-epsilon, 1+epsilon)
    * advantages
)

policy_loss = torch.maximum(
    unclipped_loss,
    clipped_loss,
).mean()
```

这里的 Clipping 主要限制单次 PPO 更新的代理目标，防止模型从本轮 Rollout Policy 偏离得过于激进。

但需要注意：

**PPO Clipping 并不严格保证实际策略概率比一定落在裁剪区间内。** 它裁剪的是优化目标，而不是强行把所有概率变化限制在某个范围内。

### 7.2 KL Penalty

KL Penalty 比较的是：

```text
Current / Rollout Policy
           vs
Reference Policy
```

它更关注模型相对长期参考策略的偏离。

即使每次 PPO 更新的变化都很小，多次更新以后，策略仍然可能离最初的 SFT 模型越来越远。

因此，这两种机制解决的问题不同。

| 机制           | 比较对象                                 | 主要目的              |
| ------------ | ------------------------------------ | ----------------- |
| PPO Clipping | Current Policy vs Old Rollout Policy | 控制本轮策略更新的幅度       |
| KL Penalty   | Policy vs Reference Policy           | 防止整个后训练过程过度偏离参考分布 |

可以将 PPO Clipping 理解为短期更新约束，将 Reference KL 理解为长期参考正则化。

两者可以同时使用，并不冲突。

---

## 八、GAE 为什么不能直接打乱 Token？

有了 Reward 之后，我们还需要计算 Advantage。

GAE 的递推关系为：

$$
\delta_t
=
r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)
$$

$$
A_t
=
\delta_t+
\gamma\lambda(1-d_t)A_{t+1}
$$

其中 \(d_t\) 表示当前时间步之后是否真正终止。对于仅达到采样长度上限的截断轨迹，是否 Bootstrap 需要依据训练实现的终止语义处理。

假设 Response 为：

```text
y0 y1 y2
```

奖励为：

```text
r0 r1 r2
```

GAE 要从后向前递推：

```text
A2
 ↑
A1
 ↑
A0
```

因为较早的 Token 可能间接影响最终获得的 Reward。

但如果把不同 Response 直接拼成一条轨迹，GAE 就可能跨越 Response 边界传播奖励。

所以：

```text
rewards    [B,R]
values     [B,R]
advantages [B,R]
```

在计算 GAE 时必须保留每条 Response 的时间顺序与终止边界。

等到 Advantage 已经计算完成以后：

```text
advantages[b,t]
```

就已经表示该 State-Action 对应的训练信号。

此时便可以与相同位置的 `new_log_probs` 一起进入 PPO Loss。

---

## 九、最后一步：为什么必须处理 Padding Mask？

真实的 LLM Batch 中，各条 Response 的长度通常不同。

例如：

```text
Response A:
a0 a1 EOS PAD PAD

Response B:
b0 b1 b2  b3 EOS
```

有效 Action Mask 为：

```text
A:
1 1 1 0 0

B:
1 1 1 1 1
```

其中 EOS 本身是已经生成的 Action，通常应参与有效 Token 计算；EOS 后面的 PAD 不参与。

如果直接：

```python
policy_loss = per_token_loss.reshape(-1).mean()
```

可能将 PAD 位置也计入 Loss。

正确的 Token-Level Mean 可以写成：

```python
policy_loss = per_token_loss[response_mask].mean()
```

等价于：

$$
L_{\mathrm{policy}}
=
\frac{
\sum_{b,t}m_{b,t}L_{b,t}
}{
\sum_{b,t}m_{b,t}
}
$$

其中 \(m_{b,t}\) 是有效 Token Mask。

但这里还有一个值得注意的细节：

**Token Mean 与 Sequence Mean 不一定相同。**

假设：

```text
Response A:
有效 Token Loss = [1, 3]

Response B:
有效 Token Loss = [4]
```

按 Token 平均：

$$
L_{\mathrm{token}}
=
\frac{1+3+4}{3}
\approx2.667
$$

先按 Sequence 平均，再平均：

$$
L_{\mathrm{sequence}}
=
\frac{
(1+3)/2+4
}{2}
=3
$$

前者使较长的 Response 拥有更多 Token 权重。

后者使每条 Response 的权重相同。

因此，阅读 PPO 源码时，不能只看 Flatten 以后 Shape 是否正确，还需要看 **Loss Reduction 所表达的加权语义**。

---

## 十、动手验证：一个微型 LLM PPO 数据流

前面讲了很多概念，现在用一个只依赖 PyTorch 的小实验验证它们。

这个实验不训练真正的大语言模型，而是构造一个微型 Causal Transformer，用来观察：

```text
old_log_probs
rewards
values
advantages
policy_loss
```

以及：

```text
Transformer Forward
       ↓
context_length - 1 : -1
       ↓
Response Token Logprobs
       ↓
GAE
       ↓
PPO Loss
```

实验刻意省略了真实 PPO 的环境采样、Reward Model 训练、Critic Loss、参数调度等工程内容。

其中 Response Token 是模拟数据，而不是从旧策略真实采样的 Rollout，因此这是一段 **Shape、位置对齐与损失计算的验证代码，不应直接当成完整的 On-Policy PPO 训练器**。

### 10.1 核心代码

运行环境：

```bash
pip install torch
```

代码如下：

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================
# 1. 定义一个微型 Causal Transformer
# ==============================================

class TinyCausalActorCritic(nn.Module):
    def __init__(self, vocab_size=24, max_len=24, hidden_size=16):
        super().__init__()

        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_size)

        # Position Embedding
        self.position_emb = nn.Embedding(max_len, hidden_size)

        # 单层 Transformer，仅用于演示时序依赖
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=2,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )

        self.backbone = nn.TransformerEncoder(
            layer, num_layers=1
        )

        # Actor Head：预测下一个 Token 的分布
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Critic Head：预测当前 State 的 Value
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids):
        B, S = input_ids.shape

        # 为每个位置构造 Position Embedding
        positions = torch.arange(S, device=input_ids.device)

        h = (
            self.token_emb(input_ids)
            + self.position_emb(positions)[None]
        )

        # 上三角 Mask：禁止关注未来 Token
        future_mask = torch.triu(
            torch.ones(
                S, S,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )

        # 保留完整序列进行 Transformer Forward
        h = self.backbone(h, mask=future_mask)

        # [B,S,V]：每个位置预测下一个 Token
        logits = self.lm_head(h)

        # [B,S]：每个位置预测 State Value
        values = self.value_head(h).squeeze(-1)

        return logits, values


# ==============================================
# 2. 一次教学版 PPO 数据流
# ==============================================

def run_case(B, P, R, lengths):
    torch.manual_seed(2026)

    # 创建旧策略
    old_model = TinyCausalActorCritic()

    # 冻结的参考策略
    ref_model = copy.deepcopy(old_model)

    # 模拟旧策略与参考策略之间产生了一点差异
    with torch.no_grad():
        old_model.lm_head.weight[1].add_(0.1)

    # 当前准备更新的策略
    current_model = copy.deepcopy(old_model)

    # 关闭 Dropout，保证位置对齐测试可复现
    for model in (old_model, ref_model, current_model):
        model.eval()

    # 随机构造 Prompt 和 Response Token
    prompt = torch.randint(1, 24, (B, P))
    response = torch.randint(1, 24, (B, R))

    # 每条 Response 的有效长度
    lengths = torch.tensor(lengths)

    # 有效 Token Mask：[B,R]
    mask = torch.arange(R)[None, :] < lengths[:, None]

    # 无效位置用 0 模拟 PAD
    response = response.masked_fill(~mask, 0)

    # 完整输入：[B,P+R]
    input_ids = torch.cat((prompt, response), dim=1)

    # 从完整序列的 logits 中计算 Response Logprobs
    def selected_logprob(full_logits):

        # [B,P+R,V] -> [B,R,V]
        response_logits = full_logits[:, P-1:-1, :]

        # 先计算 Vocabulary 上的 Log Softmax
        all_logprobs = F.log_softmax(
            response_logits, dim=-1
        )

        # 只提取实际 Response Token 的 Logprob
        # [B,R,V] -> [B,R]
        return all_logprobs.gather(
            -1, response.unsqueeze(-1)
        ).squeeze(-1)

    # ==========================================
    # 3. Rollout 数据：旧 Logprob、Ref Logprob、Value
    # ==========================================

    with torch.no_grad():

        old_logits, old_full_values = old_model(input_ids)
        ref_logits, _ = ref_model(input_ids)

        # [B,R]
        old_log_probs = selected_logprob(old_logits)

        # [B,R]
        ref_log_probs = selected_logprob(ref_logits)

        # 对齐每个 Action 执行前的 State Value
        # [B,P+R] -> [B,R]
        values = old_full_values[:, P-1:-1]

        # ======================================
        # 4. 验证 Causal Forward 的位置对齐
        # ======================================

        for t in range(R):

            # 只保留生成第 t 个 Token 之前的 Prefix
            prefix = input_ids[:, :P+t]

            # 单独计算该 Prefix 的最后一个决策位置
            prefix_logits, prefix_values = old_model(prefix)

            # 完整 Forward 在对应位置的输出
            # 应与单独计算 Prefix 的输出一致
            assert torch.allclose(
                old_logits[:, P+t-1],
                prefix_logits[:, -1],
                atol=1e-5,
            )

            assert torch.allclose(
                old_full_values[:, P+t-1],
                prefix_values[:, -1],
                atol=1e-5,
            )

        # ======================================
        # 5. 构造 KL Reward
        # ======================================

        beta = 0.05
        gamma = 0.99
        gae_lambda = 0.95
        epsilon = 0.2

        # Sampled KL：[B,R]
        sampled_kl = old_log_probs - ref_log_probs

        # KL Reward：[B,R]
        rewards = (-beta * sampled_kl).masked_fill(
            ~mask, 0
        )

        # 模拟 Reward Model 对整条回答的评分
        scores = torch.linspace(0.4, 1.0, B)

        # 将 Score 加到每条 Response 的最后有效 Token
        rewards.scatter_add_(
            1,
            (lengths - 1)[:, None],
            scores[:, None],
        )

        # ======================================
        # 6. 计算 GAE
        # ======================================

        advantages = torch.zeros_like(rewards)

        # 每条 Response 独立维护 GAE
        next_gae = torch.zeros(B)

        # 从最后一个 Token 向前遍历
        for t in reversed(range(R)):

            # 下一位置是否仍属于同一有效 Response
            continues = (
                mask[:, t+1].float()
                if t+1 < R
                else torch.zeros(B)
            )

            # 下一状态 Value
            next_v = (
                values[:, t+1]
                if t+1 < R
                else torch.zeros(B)
            )

            # TD Error
            delta = (
                rewards[:, t]
                + gamma * continues * next_v
                - values[:, t]
            )

            # GAE 反向递推
            next_gae = (
                delta
                + gamma * gae_lambda * continues * next_gae
            )

            # 只保存有效位置
            advantages[:, t] = next_gae * mask[:, t]

    # ==========================================
    # 7. PPO Update：重新评分旧 Response
    # ==========================================

    optimizer = torch.optim.Adam(
        current_model.parameters(),
        lr=1e-2,
    )

    # 仍然输入完整的 Prompt + 原 Response
    new_logits, _ = current_model(input_ids)

    # [B,R]
    new_log_probs = selected_logprob(new_logits)

    # 当前策略 / 旧策略概率比
    ratio = torch.exp(new_log_probs - old_log_probs)

    # PPO 未裁剪损失
    loss1 = -advantages * ratio

    # PPO 裁剪损失
    loss2 = (
        -advantages
        * ratio.clamp(1-epsilon, 1+epsilon)
    )

    # 每个 Token 的 PPO Loss：[B,R]
    per_token_loss = torch.maximum(loss1, loss2)

    # 只选择有效 Token，最终得到 Scalar
    policy_loss = per_token_loss[mask].mean()

    # ==========================================
    # 8. 验证 Shape 与反向传播
    # ==========================================

    assert old_log_probs.shape == (B, R)
    assert rewards.shape == (B, R)
    assert values.shape == (B, R)
    assert advantages.shape == (B, R)
    assert policy_loss.ndim == 0
    assert torch.isfinite(policy_loss)

    # 初始 Current Model 和 Old Model 相同
    # 因此有效位置上的 Ratio 应当接近 1
    assert torch.allclose(
        ratio[mask],
        torch.ones_like(ratio[mask]),
        atol=1e-5,
    )

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # 验证更新后的策略概率确实发生变化
    with torch.no_grad():
        changed_logits, _ = current_model(input_ids)
        changed_ratio = torch.exp(
            selected_logprob(changed_logits) - old_log_probs
        )

        assert not torch.allclose(
            changed_ratio[mask],
            torch.ones_like(changed_ratio[mask]),
            atol=1e-5,
        )

    print(
        f"B={B}, P={P}, R={R}, "
        f"old_log_probs={tuple(old_log_probs.shape)}, "
        f"advantages={tuple(advantages.shape)}, "
        f"policy_loss={tuple(policy_loss.shape)}"
    )


# ==============================================
# 9. 多组测试：包括最短序列和变长 Response
# ==============================================

run_case(2, 4, 3, [3, 1])
run_case(1, 1, 1, [1])
run_case(3, 2, 5, [5, 3, 1])
run_case(2, 3, 4, [4, 4])
```

### 10.2 这个实验验证了什么？

最值得关注的是下面几个结论。

**第一，完整序列的 Causal Forward 与逐个 Prefix 评分能够对齐。**

代码中比较：

```python
old_logits[:, P+t-1]
```

与：

```python
prefix_logits[:, -1]
```

两者应该一致。

这验证了我们之前的判断：

$$
h_{P+t-1}\leftrightarrow s_t
$$

**第二，Transformer Forward 之前不能丢掉 Prefix。**

代码始终使用：

```python
input_ids.shape = [B, P+R]
```

进行 Actor Forward。

它没有先将 Response Token 打散成独立的输入。

**第三，GAE 使用序列内部的时间关系。**

我们保留：

```text
advantages.shape = [B,R]
```

并借助 Mask 阻止不同 Response 的奖励互相传播。

**第四，PPO Loss 可以在有效 Token 上进行聚合。**

```python
policy_loss = per_token_loss[mask].mean()
```

最终得到：

```text
policy_loss.shape = []
```

即一个标量。

这个实验最重要的意义，不是训练出了什么模型，而是将 State、Action、Log Probability、Value、Reward、Advantage 和 Loss 对应到了具体 Tensor。

---

## 十一、TRL 是什么？如何阅读真实 PPO 源码？

TRL 的全称是 **Transformers Reinforcement Learning**。

它是 Hugging Face 提供的大模型后训练库，与 Transformers 生态集成，可以用于：

* Supervised Fine-Tuning（SFT）。
* Reward Modeling。
* Direct Preference Optimization（DPO）。
* Group Relative Policy Optimization（GRPO）。
* Proximal Policy Optimization（PPO）等。

GitHub 地址：

https://github.com/huggingface/trl

对于阅读 PPO 源码，我建议不要一开始就研究训练配置、分布式训练和日志系统。

应当按照数据流寻找关键代码。

### 11.1 第一站：找到 Rollout

关注：

```text
query_responses
responses
logprobs
ref_logprobs
values
scores
```

检查每个变量的 Shape。

尤其关注：

```python
ref_logits = ref_output.logits[
    :, context_length - 1 : -1
]
```

和：

```python
value = full_value[
    :, context_length - 1 : -1
]
```

观察为什么这两个 Slice 可以对齐同一组 State-Action 对。

### 11.2 第二站：找到 KL Reward

重点追踪：

```text
old_logprobs
ref_logprobs
      │
      ▼
sampled_kl
      │
      ▼
non_score_reward
      │
      ├──────── score
      │           │
      └─────┬─────┘
            ▼
          rewards
```

除了计算公式，还要看 Reward Model 的 Score 被加到了什么索引，以及终止边界如何表示。

### 11.3 第三站：找到 GAE

关注：

```text
rewards
values
nextvalues
advantages
returns
```

观察：

```python
for t in reversed(range(...)):
```

并检查：

* 是否正确处理终止 Token？
* 是否需要对截断轨迹 Bootstrap？
* 是否在计算 Advantage 时避免使用无效 Padding？
* `returns` 是否由 `advantages + values` 构造？

### 11.4 第四站：找到 PPO Update

这是本次学习的终点。

重点观察：

```python
new_logprobs
ratio
pg_losses
pg_losses2
pg_loss
```

还应该检查：

```text
minibatch 使用什么维度进行随机打乱？
```

在 TRL 的 PPO 实现中，可以看到它对 Batch 中的 Sequence 索引进行 Shuffle，然后仍然将完整 Sequence 送入 Actor。

这样可以保持 Prefix 上下文，再计算 Token-Level PPO Loss。

### 11.5 一个版本提醒

TRL 更新比较频繁，PPO 的模块路径和配置接口发生过迁移。

为了方便复现本文的源码阅读过程，建议先使用固定版本：

[TRL v0.25.0：ppo_trainer.py](https://github.com/huggingface/trl/blob/v0.25.0/trl/trainer/ppo_trainer.py)

该版本包含本文讨论的：

```text
context_length - 1 : -1
sampled KL
GAE
sequence minibatch
PPO clipped loss
```

当前官方 PPO 文档展示的入口为：

```python
from trl.experimental.ppo import PPOConfig, PPOTrainer
```

但实际使用时，应以安装版本的文档与源码为准，不要直接混用不同版本的导入路径和配置参数。

---

## 十二、我在理解 LLM PPO 时最容易犯的几个错误

### 错误一：认为整个 Response 是 PPO 唯一的 Action

在本文讨论的 Token-Level PPO 中：

```text
每个生成 Token = 一个 Action
```

整条 Response 是一条 Trajectory。

虽然 Reward Model 经常对整条 Response 评分，但策略梯度通常可以在 Token Level 计算。

### 错误二：认为 new_log_probs 来自重新采样的新 Response

实际上，PPO Update 使用旧 Response 重新评分。

因此：

```text
Old State-Action
```

和：

```text
New State-Action
```

应该对应相同的样本，只是策略参数不同。

### 错误三：认为 Flatten 一定破坏时序信息

Flatten 是否安全取决于数据处于哪个计算阶段。

如果 Context 已经通过 Causal Transformer 编码到 Hidden State，相关 Log Probability 已经计算出来，那么 Flatten 这些 Token-Level 结果并不会重新抹掉之前的上下文计算。

但如果在 Transformer Forward 之前把 Token 拆成独立输入，就可能丢失 State 所需的 Prefix。

### 错误四：认为 KL Penalty 和 PPO Clipping 是同一个东西

PPO Clipping 使用的是：

```text
Current Policy vs Old Policy
```

KL Penalty 使用的是：

```text
Policy vs Reference Policy
```

二者优化目的不同。

### 错误五：认为 `values[:, t]` 表示执行第 t 个 Token 之后的 Value

在本文使用的定义中：

$$
V(s_t)
$$

表示执行当前 Action 之前的 State Value。

因此：

```text
values[:, t]
```

对应：

```text
prompt + response[:t]
```

而不是：

```text
prompt + response[:t+1]
```

这也是为什么 Value 与 Logits 在 Token PPO 中能够使用同一个决策位置进行对齐。

---

## 十三、总结：我真正理解了什么？

经过这次代码学习，我认为 LLM PPO 可以归纳成三条非常重要的认识。

### 1. LLM PPO 仍然遵循经典强化学习的 State-Action 结构

对于自回归语言模型：

$$
s_t=(prompt,y_{<t})
$$

$$
a_t=y_t
$$

Transformer 使用 Causal Attention，并行计算多个时间步的条件概率，并没有改变 PPO 的核心目标。

### 2. Flatten 的关键不是 Shape，而是依赖关系

在 LLM PPO 中：

```text
Transformer Forward 之前：
需要保留完整 Prefix

GAE 计算过程中：
需要保留时间依赖和终止边界

PPO Loss Reduction：
可以聚合有效 Token
```

真正重要的是确定：

**当前计算是否还需要使用 Token 之间的时序依赖？**

### 3. 同一个决策位置连接了 Actor 和 Critic

对于第 \(t\) 个 Response Token：

$$
h_{P+t-1}\leftrightarrow s_t
$$

通过 Actor Head：

$$
\mathrm{LMHead}(h_{P+t-1})
\rightarrow
\pi_\theta(a_t|s_t)
$$

通过 Value Head：

$$
\mathrm{ValueHead}(h_{P+t-1})
\rightarrow
V_\phi(s_t)
$$

再结合：

```text
old_log_probs
new_log_probs
rewards
values
advantages
```

得到：

```text
                      Prompt
                        │
                        ▼
                 Sample Response
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
          Old Policy            Reference
             │                     │
             ▼                     ▼
        old_logprobs          ref_logprobs
             │                     │
             └──────────┬──────────┘
                        ▼
                   KL Penalty
                        │
                        │      Reward Model
                        │            │
                        │            ▼
                        │           Score
                        │            │
                        └──────┬─────┘
                               ▼
                            rewards
                               │
                             values
                               │
                               ▼
                              GAE
                               │
                               ▼
                          advantages
                               │
                     Current Actor
                               │
                               ▼
                         new_log_probs
                               │
                               ▼
                     PPO Probability Ratio
                               │
                               ▼
                     PPO Clipped Objective
                               │
                               ▼
                          policy_loss
                               │
                               ▼
                         backward()
```

最终，PPO 的核心思想依然没有改变：

> **Advantage 告诉策略某个 Action 是否值得鼓励，Probability Ratio 描述当前策略相对于旧策略发生了怎样的变化，PPO Clipping 则通过代理目标抑制过于激进的更新。**

在 LLM 场景下，我们还通过 Reference KL 正则化，限制模型为了追求 Reward Model 评分而过度偏离参考策略。

这次学习让我意识到，阅读强化学习源码时，比记住某个公式更重要的是能够回答：

> **这个变量从哪里来？它代表什么？Shape 为什么是这样？它在后续计算中如何被使用？**

当这些问题都能回答清楚时，复杂的 PPO 代码就逐渐变成了一条可以理解、可以验证的数据流。

---

## 参考文献

以下资料包括 PPO 原始论文、RLHF 相关论文、官方文档和真实源码。本文中的教学代码为便于理解而编写的简化示例，不是对这些项目完整训练器的直接复现。

**[1] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.**

PPO 的原始论文，介绍 Clipped Surrogate Objective 等核心方法。

* https://arxiv.org/abs/1707.06347

**[2] Huang, S., et al. CleanRL — Proximal Policy Optimization.**

适合入门经典 PPO 的单文件实现，用于理解 Rollout、GAE、Flatten 和 PPO Policy Loss。

* 官方文档：https://docs.cleanrl.dev/rl-algorithms/ppo/
* 源码：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

**[3] Hugging Face. TRL — Transformers Reinforcement Learning.**

大模型后训练框架及 PPO Trainer 官方文档。

* GitHub：https://github.com/huggingface/trl
* PPO 文档：https://huggingface.co/docs/trl/main/en/ppo_trainer

**[4] Hugging Face. TRL PPOTrainer Source Code, v0.25.0.**

本文重点参考的工程源码版本，包含 Token-Level Log Probability、Value、KL Reward、GAE、Minibatch 和 PPO Loss 的实现。

* https://github.com/huggingface/trl/blob/v0.25.0/trl/trainer/ppo_trainer.py

**[5] Huang, S., et al. (2023). The N Implementation Details of RLHF with PPO.**

详细介绍 RLHF PPO 的工程细节，包括 KL Penalty、Reward、Value Head、Padding、训练稳定性和实现差异。

* https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo

**[6] Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback.**

InstructGPT 论文，介绍使用人类反馈、Reward Model 和 PPO 进行语言模型对齐的经典流程。

* https://arxiv.org/abs/2203.02155

**[7] Hugging Face Transformers. Causal Language Modeling.**

解释自回归语言模型的 Next Token Prediction 和 Causal Attention，是理解 Logits Shift 的基础。

* https://huggingface.co/docs/transformers/v4.46.0/en/tasks/language_modeling

**[8] Schulman, J. Approximating KL Divergence.**

讨论 KL Divergence 的不同采样估计器，可用于进一步理解 TRL 中的 `k1`、`k3` 等实现。

* http://joschu.net/blog/kl-approx.html

---

*本文属于个人强化学习与大模型后训练学习笔记系列。目标不是追求 SOTA，而是通过公式、代码、Tensor Shape 和可验证的实验，逐步建立对 RLHF 算法及其工程实现的理解。*
