# Day 11｜从 PPO 到 GRPO：不需要 Value Model，大模型为什么依然能够通过强化学习提升推理能力？

> **学习系列：LLM Post-Training / Reinforcement Learning**
>
> 核心论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
>
> 本文目标：从 PPO 出发，理解 GRPO 为什么需要对同一个 Prompt 采样多个 Response、为什么使用组内相对优势、为什么能够移除 Value Model，并从零实现一个可运行的极简 GRPO Loss。

---

## 一、引言：GRPO 到底解决了什么问题？

在大语言模型的后训练阶段，我们希望模型不仅能够模仿已有的高质量答案，还能够通过反馈不断提升自己的推理能力。

例如，我们向模型提出一个数学问题：

```text
Prompt:
求解方程 2x + 3 = 7
```

模型可能生成不同的回答：

```text
Response 1:
2x = 4，因此 x = 2。
正确。

Response 2:
2x = 7 - 3 = 4，x = 4。
错误。

Response 3:
x = 7 - 3 = 4。
错误。

Response 4:
无法解答。
```

如果存在一个 Reward Function，能够判断这些答案的质量，那么我们自然希望：

* 高质量 Response 在未来更容易被生成。
* 低质量 Response 在未来更不容易被生成。

然而，这里存在一个重要的问题：

**当模型生成一条 Response 并得到 Reward 后，它怎么知道这个 Reward 算好还是算坏？**

假设模型得到：

$$
r=0.8
$$

这个分数到底意味着什么？

对于一道特别困难的题目，如果模型通常只能得到 0.2，那么 0.8 显然非常优秀。

但对于一道非常简单的题目，如果模型通常能够得到 0.95，那么 0.8 反而不太理想。

因此，真正决定模型应该如何更新的，不仅是 Reward 的绝对数值，更重要的是：

> **这个 Response 的表现，相对于当前模型在相同任务上的正常表现，到底好多少或者差多少？**

这就是 Advantage（优势函数）试图回答的问题。

GRPO 的核心思想也由此产生：

**不再额外训练一个 Value Model 来预测正常表现，而是让模型对同一个问题生成多个答案，通过这些答案之间的相对比较，估计一个 Baseline。**

这也是 DeepSeekMath 论文提出 GRPO 的关键动机。[1]

---

## 二、理解 GRPO 之前，先理解 Advantage

### 2.1 Reward 和 Advantage 有什么区别？

Reward 衡量的是：

> 这次回答获得了多少奖励？

Advantage 衡量的是：

> 这次回答比预期表现好多少？

在强化学习中，优势函数的标准定义为：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

其中：

* \(Q^\pi(s,a)\)：在状态 \(s\) 选择动作 \(a\) 后的预期回报。
* \(V^\pi(s)\)：在状态 \(s\) 按照当前策略行动的预期回报。
* \(A^\pi(s,a)\)：选择动作 \(a\) 相对于平均表现的优势。

为了建立直觉，可以先把它简化成：

$$
\boxed{A=\text{实际表现}-\text{预期表现}}
$$

也就是：

$$
A=R-b
$$

其中 \(b\) 是 Baseline（基线）。

例如：

| 实际 Reward | Baseline | Advantage | 含义   |
| --------- | -------- | --------- | ---- |
| 0.9       | 0.5      | +0.4      | 高于预期 |
| 0.5       | 0.5      | 0         | 符合预期 |
| 0.2       | 0.5      | -0.3      | 低于预期 |

我们希望训练过程产生这样的倾向：

$$
A>0 \Rightarrow \text{提高对应动作的概率}
$$

$$
A<0 \Rightarrow \text{降低对应动作的概率}
$$

$$
A=0 \Rightarrow \text{没有该 Advantage 带来的更新信号}
$$

这里需要注意：

**Advantage 并不是 Reward 的另一个名字。**

Reward 提供评价，Advantage 提供相对于基线的比较信号。

而 GRPO 最重要的创新，就发生在 Baseline 的构造方式上。

### 2.2 PPO 的 Baseline 从哪里来？

在常见的 Actor-Critic PPO 实现中，存在两个重要的可训练部分：

**Policy Model（Actor）**

负责决定在当前状态下应该生成哪个 Token。

**Value Model（Critic）**

负责预测当前状态后续可能获得的回报。

对于大语言模型而言，可以把 Value Model 理解为：

```text
Prompt:
求解 2x + 3 = 7

已经生成：
2x = 7 - 3
x =

       ↓

Value Model

       ↓

预测：
从这个前缀继续生成，
最终大概能够获得多少 Reward？
```

于是，可以通过实际回报和预测 Value 之间的差异估计 Advantage：

$$
A_t\approx G_t-V_\psi(s_t)
$$

这里 \(G_t\) 表示从当前时间步开始的 Return。

实际 PPO 往往结合 GAE（Generalized Advantage Estimation）来估计 Advantage，而不是简单地使用最终 Reward 减 Value。[2][3]

### 2.3 PPO 为什么需要 Value Model？

一个很重要的原因是降低策略梯度估计的方差。

假设模型生成了一条回答，得到 Reward = 0.8。

如果缺少合适的 Baseline，就很难判断这个结果到底是超出预期，还是低于预期。

Value Model 可以学习当前状态的预期回报，让更新信号更有针对性。

但是，在 LLM 训练场景中，这也带来了额外的成本：

1. Value Model 可能与 Policy Model 规模相近。
2. 需要额外的显存、计算资源来训练 Critic。
3. 很多语言模型任务只在完整回答结束后才能获得 Reward。
4. 从最终结果出发，为每一个中间 Token 学习准确的 Value 并不容易。

于是，DeepSeekMath 提出了一个重要的问题：

**如果 Value Model 的一个核心作用是提供 Baseline，那么这个 Baseline 能不能直接从采样结果中计算出来？**

GRPO 的答案是：可以。

---

## 三、GRPO 最核心的思想：让同一道题的多个答案互相比较

GRPO，全称：

**Group Relative Policy Optimization（组相对策略优化）**。

我们先不看复杂公式，只看整个过程。

假设模型面对同一道数学题，一次生成四个 Response：

```text
                  Prompt q
                      |
                      v
                Old Policy
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Response 1  Response 2  Response 3 ...
          |           |           |
          v           v           v
       Reward 1    Reward 2    Reward 3 ...
          |           |           |
          +-----------+-----------+
                      |
                      v
             Group Statistics
               Mean / Std
                      |
                      v
             Relative Advantage
                      |
                      v
              PPO-style Update
```

对于同一个 Prompt，我们从旧策略中采样：

$$
o_1,o_2,\ldots,o_G\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)
$$

这里：

* \(q\)：一个 Prompt。
* \(o_i\)：第 \(i\) 个生成的 Response。
* \(G\)：同一组 Response 的数量。
* \(\pi_{\theta_{\mathrm{old}}}\)：采样时使用的旧 Policy。

然后通过 Reward Function 得到：

$$
r_1,r_2,\ldots,r_G
$$

例如：

$$
\mathbf r=[1.0,0.8,0.2,0.0]
$$

这些分数可以理解为某种假设的回答质量评分。

现在，我们不再让额外的神经网络预测 Baseline。

而是直接计算：

$$
\mu_r=\frac{1}{G}\sum_{i=1}^G r_i
$$

这个组内平均 Reward，就可以近似表示：

> 当前 Policy 在解决这个 Prompt 时的平均表现。

数学上可以理解为：

$$
\boxed{\mu_r\approx\mathbb E_{o\sim\pi_{\mathrm{old}}(\cdot|q)}[R(q,o)]}
$$

这就是 GRPO 用组内统计替代 Critic Baseline 的基本思路。

### 3.1 为什么必须对同一个 Prompt 采样多个 Response？

因为 GRPO 希望比较的是：

$$
R(q,o_i)-\mathbb E[R|q]
$$

也就是：

**同一问题下，这条回答相对于模型平均表现的好坏。**

如果一个 Prompt 只采样一条 Response，那么只有一个 Reward：

$$
\mathbf r=[0.8]
$$

它的组内均值也是 0.8。

因此：

$$
r-\mu_r=0
$$

标准差也为 0。

这时，我们无法通过组内比较判断它是否比其他回答更好。

注意，这并不意味着所有强化学习算法都必须对一个 Prompt 采样多次。

而是说：

**GRPO 所采用的组内相对 Advantage，需要多个样本才能提供有效的比较信息。**

### 3.2 为什么不能把不同 Prompt 的 Response 放在一起比较？

我们来看一个例子。

假设 Prompt A 是一道简单题：

$$
R_A=[1,1,1,0.8]
$$

它的组内均值：

$$
\mu_A=0.95
$$

对于 Reward 为 0.8 的回答：

$$
0.8-0.95<0
$$

这意味着该回答低于当前模型在这道题上的平均表现。

再假设 Prompt B 是一道难题：

$$
R_B=[0,0,0,0.8]
$$

它的组内均值：

$$
\mu_B=0.2
$$

对于 Reward 为 0.8 的回答：

$$
0.8-0.2>0
$$

同样得到 0.8 分，但它比当前模型解决 Prompt B 时的平均水平好得多。

因此：

| Prompt | Reward | 相对表现  | Advantage 符号 |
| ------ | ------ | ----- | ------------ |
| A：简单题  | 0.8    | 低于组平均 | 负            |
| B：困难题  | 0.8    | 高于组平均 | 正            |

这说明：

**同一个 Reward，在不同 Prompt 的上下文中可能代表完全不同的学习信号。**

GRPO 通过 Prompt 内部分组比较，避免直接使用整个 Batch 的统一平均 Reward 作为基线，从而在一定程度上减少不同任务难度造成的干扰。

---

## 四、从公式真正理解 Group-Relative Advantage

DeepSeekMath 论文在 Outcome Supervision 场景中使用的 Advantage 为：

$$
\boxed{
A_i=\frac{r_i-\mu_r}{\sigma_r}
}
$$

其中：

$$
\mu_r=\frac1G\sum_{i=1}^{G}r_i
$$

$$
\sigma_r=\sqrt{\frac1G\sum_{i=1}^{G}(r_i-\mu_r)^2}
$$

这里采用总体标准差的定义，方便后续和 NumPy 示例保持一致。

原论文使用的是 `std` 记号，没有在该公式中明确规定具体的标准差实现约定。

### 4.1 第一步：减去 Mean

$$
r_i-\mu_r
$$

这一部分决定 Advantage 的正负。

* 比组内平均 Reward 高，Advantage 为正。
* 比组内平均 Reward 低，Advantage 为负。
* 等于组内平均 Reward，Advantage 为零。

也就是说，减去 Mean 的作用是：

**建立一个相对评价基准。**

### 4.2 第二步：除以 Standard Deviation

为什么还要除以标准差？

考虑两组 Reward：

$$
R_A=[100,80,20,0]
$$

$$
R_B=[1,0.8,0.2,0]
$$

这两组分数的相对结构完全一样，只是 Reward 的数值尺度不同。

如果只减去 Mean，两组产生的 Advantage 大小会差很多。

如果继续除以标准差：

$$
A_i=\frac{r_i-\mu_r}{\sigma_r}
$$

就能使它们具有相同的标准化结果。

这实际上就是统计学中的 **Z-Score Standardization**。

减均值解决的是：

> 谁比平均水平好？

除标准差解决的是：

> 不同 Reward 尺度下，更新信号如何更具有可比性？

不过，标准化也有代价。

当某一组 Reward 的差异极小时，除以很小的标准差可能放大评分噪声。

因此，标准差归一化虽然能统一尺度，却并不意味着在所有场景下都一定更稳定。

### 4.3 手算一次 Advantage

假设：

$$
\mathbf r=[1.0,0.8,0.2,0.0]
$$

计算均值：

$$
\mu_r=\frac{1.0+0.8+0.2+0.0}{4}=0.5
$$

计算标准差：

$$
\sigma_r=
\sqrt{
\frac{
(1-0.5)^2+
(0.8-0.5)^2+
(0.2-0.5)^2+
(0-0.5)^2
}{4}
}
$$

得到：

$$
\sigma_r\approx0.4123
$$

计算每条 Response 的 Advantage：

$$
A_1=\frac{1.0-0.5}{0.4123}\approx1.2127
$$

$$
A_2=\frac{0.8-0.5}{0.4123}\approx0.7276
$$

$$
A_3=\frac{0.2-0.5}{0.4123}\approx-0.7276
$$

$$
A_4=\frac{0.0-0.5}{0.4123}\approx-1.2127
$$

最终：

| Response   | Reward | Advantage | 更新倾向   |
| ---------- | -----: | --------: | ------ |
| Response 1 |    1.0 |   +1.2127 | 增加生成概率 |
| Response 2 |    0.8 |   +0.7276 | 增加生成概率 |
| Response 3 |    0.2 |   -0.7276 | 降低生成概率 |
| Response 4 |    0.0 |   -1.2127 | 降低生成概率 |

对于非退化的标准化结果，有：

$$
\sum_{i=1}^{G}A_i=0
$$

这里的直觉非常重要：

> **GRPO 不只是告诉模型哪些答案是好的，而是让模型学习在同一问题下偏向表现更好的生成轨迹。**

这种更新倾向并不意味着每一步优化都能独立、精确地提高某条 Response 的概率，因为不同 Response 共享模型参数。

但它提供了优化的基本方向。

---

## 五、GRPO 为什么能够不使用 Value Model？

现在我们可以回到最初的问题。

PPO 中，Critic 学习：

$$
V_\psi(s_t)
$$

它预测从当前 Token 前缀继续生成的预期回报。

GRPO 则使用：

$$
\mu_r=\frac1G\sum_{i=1}^{G}r_i
$$

近似估计同一 Prompt 下的平均回报。

两者都试图提供比较基准，但来源不同。

| 维度            | Actor-Critic PPO     | GRPO                     |
| ------------- | -------------------- | ------------------------ |
| Baseline 来源   | 学习得到的 Value Function | 同一 Prompt 的组内 Reward 统计  |
| 是否额外训练 Critic | 需要                   | 不需要                      |
| Advantage 信息  | 可以依赖 Token 前缀状态      | Outcome GRPO 基于整条回答的相对结果 |
| 额外成本          | 训练 Value Model       | 同一 Prompt 多次 Rollout     |
| 主要挑战          | Value 估计与 Critic 训练  | 组内 Reward 区分度、采样成本及估计噪声  |

可以把两者的区别概括为：

```text
PPO:

Rollout
   |
   v
Reward / Return
   |
   v
Learned Value Model
   |
   v
Advantage Estimation
   |
   v
Policy Optimization
```

```text
GRPO:

Same Prompt
   |
   v
Sample Multiple Responses
   |
   v
Group Rewards
   |
   v
Mean / Standard Deviation
   |
   v
Relative Advantage
   |
   v
Policy Optimization
```

因此，我认为理解 GRPO 最重要的一句话是：

**GRPO 并不是认为 Baseline 不重要，而是用采样得到的组内统计 Baseline，替代了需要额外训练的 Critic。**

当然，这两种 Baseline 不能完全等价。

PPO 的 Critic 能根据不同 Token 前缀估计 Value。

而在最简单的 Outcome GRPO 中，同一条 Response 的所有 Token 通常使用相同的 Advantage：

$$
\hat A_{i,t}=A_i
$$

也就是说，一条回答即使只有中间某一步推理有问题，Outcome GRPO 也不能仅凭最终 Reward 精确确定错误发生在哪个 Token。

这是它在节省 Critic 成本的同时，需要面对的 Credit Assignment（信用分配）问题。

需要补充的是，DeepSeekMath 也研究了 Process Supervision GRPO，通过对推理步骤进行评分获得更细粒度的监督信息。[1]

---

## 六、理解 Clipped Objective：GRPO 如何真正更新 Policy？

到目前为止，我们已经知道了每条 Response 的 Advantage。

但是：

**知道哪个 Response 好，还不代表知道应该怎样更新模型参数。**

GRPO 继承了 PPO-style Clipped Surrogate Objective。

### 6.1 什么是 Probability Ratio？

假设一条 Response 在旧 Policy 下的概率为：

$$
\pi_{\theta_{\mathrm{old}}}(o_i|q)
$$

在新 Policy 下的概率为：

$$
\pi_\theta(o_i|q)
$$

我们定义：

$$
\rho_i=
\frac{\pi_\theta(o_i|q)}
{\pi_{\theta_{\mathrm{old}}}(o_i|q)}
$$

解释一下：

当：

$$
\rho_i>1
$$

意味着新 Policy 提高了这条 Response 的概率。

当：

$$
\rho_i<1
$$

意味着新 Policy 降低了这条 Response 的概率。

而：

$$
\rho_i=1
$$

意味着新旧 Policy 对这条 Response 给出的概率相同。

### 6.2 为什么实际代码使用 Log Probability？

大语言模型的完整 Response 由多个 Token 组成。

其概率等于逐 Token 条件概率的乘积：

$$
\pi_\theta(o|q)=
\prod_{t=1}^{|o|}
\pi_\theta(o_t|q,o_{<t})
$$

直接相乘容易出现数值下溢。

因此，通常使用 Log Probability：

$$
\log\pi_\theta(o|q)
=
\sum_{t=1}^{|o|}
\log\pi_\theta(o_t|q,o_{<t})
$$

于是：

$$
\boxed{
\rho_i=
\exp(
\log\pi_{\mathrm{new}}
-
\log\pi_{\mathrm{old}}
)
}
$$

注意，这里写的是**将完整 Response 视为一个 Action 的教学简化版本**。

真正的 DeepSeekMath GRPO 使用的是逐 Token Probability Ratio，后面会给出原论文形式。

### 6.3 为什么不能直接最大化 \(\rho_i A_i\)？

假设某条 Response 的 Advantage 大于零。

那么我们希望增加它的生成概率。

如果直接最大化：

$$
\rho_i A_i
$$

模型可能通过过度增大某条高 Advantage Response 的概率来提高优化目标，导致 Policy 更新幅度过大。

为此，PPO 引入了 Clipping 机制：

$$
\boxed{
J_i=
\min\left(
\rho_i A_i,
\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i
\right)
}
$$

例如：

$$
\epsilon=0.2
$$

那么：

$$
\operatorname{clip}(\rho_i,0.8,1.2)
$$

会将参与第二项计算的 Ratio 截断在对应区间内。

但这里有一个特别容易理解错的地方：

**Clipping 并不是把 Policy 的实际概率变化硬性限制在 0.8 到 1.2 之间。**

它是在 Surrogate Objective 中，削弱某些过度更新所带来的额外优化收益。

### 6.4 为什么要取 Min？

我们分两种情况讨论。

**情况一：Advantage 为正**

$$
A_i>0
$$

说明这条 Response 表现较好。

我们希望提高它的概率。

但是，当：

$$
\rho_i>1+\epsilon
$$

继续提高其概率，不应该无限制地获得更大的 Surrogate Objective。

所以取：

$$
\min(\rho_i A_i,\operatorname{clip}(\rho_i)A_i)
$$

会使这一方向上的收益不再继续增加。

**情况二：Advantage 为负**

$$
A_i<0
$$

说明这条 Response 表现较差。

我们希望降低它的概率。

但如果：

$$
\rho_i<1-\epsilon
$$

说明它的概率已经被降低很多。

继续降低这一概率，也不应该无限制地提高目标函数。

因此，这个方向上的优化收益同样会被截断。

可以整理成：

| Advantage | Ratio 变化          | Clipped Objective 的作用 |
| --------- | ----------------- | --------------------- |
| 正         | 超过 \(1+\epsilon\) | 不再奖励进一步增加概率           |
| 负         | 低于 \(1-\epsilon\) | 不再奖励进一步降低概率           |
| 正         | 过度降低概率            | 保留这种不利变化的惩罚           |
| 负         | 过度增加概率            | 保留这种不利变化的惩罚           |

这才是 PPO Clipping 的关键：

**限制过度有利的 Surrogate 改进，但不会自动抹除不利变化带来的损失。**

### 6.5 GRPO Loss 为什么要加负号？

GRPO 的目标是最大化：

$$
J=\frac1G\sum_{i=1}^{G}J_i
$$

但常见优化器通常通过梯度下降最小化 Loss。

因此定义：

$$
\boxed{L=-J}
$$

这样：

$$
\min L
\Longleftrightarrow
\max J
$$

需要注意，这个 Loss 是策略优化的代理目标，不是分类任务里的交叉熵或回归均方误差。

因此，它可以为正、为负或者为零。

不能仅凭 Loss 的正负判断模型是否训练成功。

---

## 七、回到 DeepSeekMath：真正的 GRPO Objective

前面为了方便理解，将完整 Response 当成一个 Action。

但原论文并不是这样计算 Probability Ratio 的。

DeepSeekMath 中，对于 Response \(o_i\) 的第 \(t\) 个 Token：

$$
\rho_{i,t}(\theta)=
\frac{
\pi_\theta(o_{i,t}|q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})
}
$$

在 Outcome Supervision 中：

$$
\hat A_{i,t}=
\frac{r_i-\mu_r}{\sigma_r}
$$

即同一条 Response 的所有 Token 共用归一化后的 Outcome Advantage。

原论文的 GRPO Objective 可以表示为：

$$
\begin{aligned}
J_{\mathrm{GRPO}}(\theta)
=\mathbb E\Bigg[
\frac1G\sum_{i=1}^{G}
\frac1{|o_i|}
\sum_{t=1}^{|o_i|}
\Big(
&\min\big(
\rho_{i,t}\hat A_{i,t},\\
&\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)
\hat A_{i,t}
\big)\\
&-\beta D_{\mathrm{KL}}
(\pi_\theta\|\pi_{\mathrm{ref}})
\Big)
\Bigg]
\end{aligned}
$$

其中期望针对采样的 Prompt 和旧策略生成的 Response 计算，KL 项在对应 Token 位置估计。[1]

这里出现了一个新概念：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

### 7.1 为什么还需要 Reference Model？

GRPO 希望提高高 Reward Response 的概率。

但是，如果 Policy 只追求 Reward，就可能出现 Reward Hacking 或过度偏离原有模型分布的现象。

因此，引入 Reference Policy：

$$
\pi_{\mathrm{ref}}
$$

作为一个参考分布。

KL Divergence 用于度量当前 Policy 与 Reference Policy 的差异。

目标中：

$$
-\beta D_{\mathrm{KL}}
$$

的作用是：

**在提高 Reward 的同时，对偏离 Reference Policy 的行为施加惩罚。**

这里需要区分三个模型角色：

* **Old Policy**：产生当前 Rollout 的策略，也用于计算 Probability Ratio。
* **New Policy**：正在优化的策略。
* **Reference Policy**：提供 KL 正则化参考的策略。

Old Policy 和 Reference Policy 的作用不同，不能混为一谈。

另外：

**GRPO 不需要 Value Model，不代表它不需要 Reward Function。**

Reward 可以由学习得到的 Reward Model 提供，也可以来自数学答案验证、代码执行测试等规则化评分。

DeepSeekMath 原论文使用了 Reward Model，而本文接下来使用手工构造的 Reward，仅为了理解算法。

---

## 八、动手实现：从零写一个极简 GRPO Loss

接下来，我们不使用 Transformer，也不加载任何语言模型。

只使用：

```python
rewards = [...]
old_log_probs = [...]
new_log_probs = [...]
```

完成以下流程：

```text
Rewards
   |
   v
Group Mean / Std
   |
   v
Advantages
   |
   v
New / Old Probability Ratio
   |
   v
Clipped Surrogate Objective
   |
   v
GRPO Loss
```

为了让重点集中在算法本身，我们做两个简化：

1. 将一条完整 Response 当成一个 Action。
2. 暂时省略 KL Regularization。

所以，下面实现的是 **GRPO 核心机制的教学版本**，不是 DeepSeekMath 的完整训练代码。

### 8.1 环境准备

只需要 NumPy：

```bash
pip install numpy
```

### 8.2 完整代码

将以下代码保存为：

`grpo_demo.py`

```python
import numpy as np


def grpo_loss(
    rewards,
    old_log_probs,
    new_log_probs,
    clip_eps=0.2,
    std_eps=1e-8,
):
    """
    教学版 GRPO Loss。

    假设：
    1. 所有 Response 来自同一个 Prompt。
    2. 一条完整 Response 视作一个 Action。
    3. 不包含 KL Regularization。
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float64)
    new_log_probs = np.asarray(new_log_probs, dtype=np.float64)

    # 检查输入
    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError(
            "同一组至少需要 2 个 response，且输入必须是一维数组"
        )

    if (
        old_log_probs.shape != rewards.shape
        or new_log_probs.shape != rewards.shape
    ):
        raise ValueError(
            "rewards、old_log_probs、new_log_probs 的长度必须一致"
        )

    if not all(
        np.all(np.isfinite(x))
        for x in (rewards, old_log_probs, new_log_probs)
    ):
        raise ValueError("输入不能包含 NaN 或 Inf")

    if not 0 <= clip_eps < 1 or std_eps <= 0:
        raise ValueError(
            "需要 0 <= clip_eps < 1 且 std_eps > 0"
        )

    # Step 1: Group-relative advantage
    reward_mean = float(rewards.mean())
    reward_std = float(rewards.std(ddof=0))

    if reward_std < std_eps:
        advantages = np.zeros_like(rewards)
    else:
        advantages = (
            rewards - reward_mean
        ) / reward_std

    # Step 2: Probability ratio
    ratios = np.exp(
        new_log_probs - old_log_probs
    )

    if not np.all(np.isfinite(ratios)):
        raise ValueError(
            "log probability 差值过大，导致 ratio 溢出"
        )

    # Step 3: Clipping
    clipped_ratios = np.clip(
        ratios,
        1 - clip_eps,
        1 + clip_eps,
    )

    # Step 4: Clipped surrogate objective
    surrogate1 = ratios * advantages
    surrogate2 = clipped_ratios * advantages

    per_response_objective = np.minimum(
        surrogate1,
        surrogate2,
    )

    # Step 5: Convert maximization to minimization
    objective = float(
        per_response_objective.mean()
    )

    return {
        "mean": reward_mean,
        "std": reward_std,
        "advantages": advantages,
        "ratios": ratios,
        "clipped_ratios": clipped_ratios,
        "per_response_objective": per_response_objective,
        "objective": objective,
        "loss": -objective,
    }


if __name__ == "__main__":
    rewards = [1.0, 0.8, 0.2, 0.0]

    old_log_probs = [
        -2.2, -2.0, -2.4, -1.9
    ]

    new_log_probs = [
        -2.0, -2.1, -2.0, -2.3
    ]

    result = grpo_loss(
        rewards,
        old_log_probs,
        new_log_probs,
    )

    for key, value in result.items():
        print(f"{key}: {np.round(value, 4)}")

    # Test 1: 所有 Reward 相同
    for same_rewards in (
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ):
        r = grpo_loss(
            same_rewards,
            old_log_probs,
            new_log_probs,
        )
        assert np.all(r["advantages"] == 0)
        assert r["loss"] == 0

    # Test 2: 只有一个 Response 正确
    r = grpo_loss(
        [1, 0, 0, 0],
        old_log_probs,
        new_log_probs,
    )
    assert r["advantages"][0] > 0
    assert np.all(r["advantages"][1:] < 0)

    # Test 3: Reward 缩放不影响相对优势
    a = grpo_loss(
        [1.0, 0.8, 0.2, 0.0],
        old_log_probs,
        new_log_probs,
    )

    b = grpo_loss(
        [100.0, 80.0, 20.0, 0.0],
        old_log_probs,
        new_log_probs,
    )

    np.testing.assert_allclose(
        a["advantages"],
        b["advantages"],
        rtol=1e-12,
        atol=1e-12,
    )

    # Test 4: 新旧 Policy 相同
    r = grpo_loss(
        rewards,
        old_log_probs,
        old_log_probs,
    )

    np.testing.assert_allclose(
        r["objective"],
        0.0,
        atol=1e-14,
    )

    # Test 5: 非法输入
    for args in (
        ([1], [-2], [-2]),
        ([1, 0], [-2], [-2, -3]),
        ([1, float("nan")], [-2, -3], [-2, -3]),
    ):
        try:
            grpo_loss(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"未拦截错误输入：{args}"
            )

    # Test 6: 检查 Clipping 的两个方向
    hi = grpo_loss(
        [1, 0],
        [-2, -2],
        [-1.4, -2.6],
    )

    np.testing.assert_allclose(
        hi["per_response_objective"],
        [1.2, -0.8],
        atol=1e-12,
    )

    print(
        "PASS: constant rewards, mixed rewards, "
        "reward scaling, unchanged policy, "
        "invalid inputs, clipping"
    )
```

### 8.3 运行结果

执行：

```bash
python grpo_demo.py
```

主要输出为：

```text
mean: 0.5

std: 0.4123

advantages:
[ 1.2127  0.7276 -0.7276 -1.2127]

ratios:
[1.2214 0.9048 1.4918 0.6703]

clipped_ratios:
[1.2    0.9048 1.2    0.8   ]

per_response_objective:
[ 1.4552  0.6584 -1.0855 -0.9701]

objective: 0.0145

loss: -0.0145
```

最后还会输出：

```text
PASS: constant rewards, mixed rewards,
reward scaling, unchanged policy,
invalid inputs, clipping
```

说明这些测试用例均成功通过。

---

## 九、逐条分析：GRPO Loss 究竟在表达什么？

只看代码输出，可能仍然不知道每一个数值代表什么。

下面重点分析其中两条 Response。

### 9.1 Response 1：好答案的概率提高了，但已经超过 Clipping 阈值

已知：

$$
A_1\approx1.2127
$$

因此，这是一个相对较好的 Response。

它的 Log Probability 从：

$$
-2.2\rightarrow-2.0
$$

所以：

$$
\rho_1=e^{-2.0-(-2.2)}
$$

$$
\rho_1=e^{0.2}\approx1.2214
$$

说明这条 Response 在新 Policy 下的概率相对于旧 Policy 增加了约 22.14%。

但是，我们设置：

$$
\epsilon=0.2
$$

于是：

$$
\operatorname{clip}(1.2214,0.8,1.2)=1.2
$$

对应两项：

$$
\rho_1A_1\approx1.4813
$$

$$
\operatorname{clip}(\rho_1)A_1\approx1.4552
$$

最终取更小值：

$$
J_1\approx1.4552
$$

这里的含义是：

**虽然这是一条优质回答，但当它的概率已经增长较多时，不再通过继续扩大这个方向上的 Ratio 来增加 Clipped Objective。**

### 9.2 Response 3：一个较差的答案，为什么概率反而增加了？

这是更值得分析的情况。

已知：

$$
A_3\approx-0.7276
$$

说明该 Response 相对于组内平均水平表现较差。

我们原本希望降低它的概率。

但它的 Log Probability 从：

$$
-2.4\rightarrow-2.0
$$

所以：

$$
\rho_3=e^{0.4}\approx1.4918
$$

这意味着：

**一个低于平均水平的 Response，在新 Policy 下反而变得更容易生成。**

于是：

$$
\rho_3A_3
\approx-1.0855
$$

而截断后的对应值约为：

$$
1.2\times(-0.7276)\approx-0.8731
$$

因为取的是：

$$
\min(-1.0855,-0.8731)
$$

最终：

$$
J_3\approx-1.0855
$$

注意，最终选择的是没有被截断的、更负的那一项。

这说明：

**Clipping 并不会把所有超出区间的 Ratio 都强行变成边界值后用于最终目标。**

对于负 Advantage 而言，如果模型错误地大幅提高了这条 Response 的概率，这种不利变化带来的惩罚依然会保留。

这也是理解 PPO 和 GRPO 的 Clipped Objective 时最容易忽略的细节。

---

## 十、最重要的边界实验：所有 Reward 都相同，会怎么样？

现在进入我认为最能体现 GRPO 局限性的实验。

### 10.1 情况一：所有 Response 都正确

假设：

```python
rewards = [1, 1, 1, 1]
```

那么：

$$
\mu_r=1
$$

$$
\sigma_r=0
$$

根据标准化公式：

$$
A_i=\frac{r_i-\mu_r}{\sigma_r}
$$

直接计算会遇到：

$$
\frac00
$$

这是未定义的。

因此，在实际实现中需要处理这一边界情况。

本教程采用：

```python
if reward_std < std_eps:
    advantages = np.zeros_like(rewards)
```

于是：

$$
A=[0,0,0,0]
$$

在省略 KL 的情况下：

$$
J=0
$$

$$
L=0
$$

### 10.2 情况二：所有 Response 都错误

假设：

```python
rewards = [0, 0, 0, 0]
```

同样有：

$$
\mu_r=0
$$

$$
\sigma_r=0
$$

处理后：

$$
A=[0,0,0,0]
$$

因此，同样没有组内相对 Advantage 信号。

这可能有点反直觉。

**明明模型四次都回答错误，为什么却没有有效的相对学习信号？**

因为 GRPO 需要通过比较不同 Response 的奖励差异来确定优化方向。

如果所有回答全部错误：

```text
Response 1: Wrong
Response 2: Wrong
Response 3: Wrong
Response 4: Wrong
```

仅凭这些相同的 Reward，GRPO 无法判断哪一个回答更值得鼓励。

这不是说这些答案都很好，而是说：

**组内比较并没有提供区分它们的信息。**

需要注意：这里说的是简化版的 Policy Surrogate。

在原论文的完整目标中，即使 Advantage 全为零，只要 KL 惩罚项非零，仍然可能存在由 KL Regularization 带来的更新信号。

### 10.3 情况三：只有一个 Response 正确

假设：

```python
rewards = [1, 0, 0, 0]
```

这一次，组内就出现了明显差异。

于是：

```text
Response 1:
Reward = 1
Advantage > 0

Response 2:
Reward = 0
Advantage < 0

Response 3:
Reward = 0
Advantage < 0

Response 4:
Reward = 0
Advantage < 0
```

模型终于获得了相对比较信号：

> Response 1 比其他 Response 更值得强化。

### 10.4 从这个实验中得到的启发

对于只有正确和错误两种结果的 Reward：

```text
[1, 1, 1, 1]
全部正确
没有组内区分度

[0, 0, 0, 0]
全部错误
没有组内区分度

[1, 0, 0, 0]
出现正确与错误的差异
具有组内比较信号

[1, 0, 1, 0]
出现正确与错误的差异
具有组内比较信号
```

这会带来一个很重要的训练启发：

**对于基于二元正确性 Reward 的 GRPO，当前模型有一定成功率、但又不能稳定完成的题目，通常更容易产生有用的组内比较信号。**

太简单的题目，模型可能每次都答对。

太困难的题目，模型可能每次都答错。

两种情况都可能缺少组内 Reward 差异。

因此，GRPO 和训练数据的难度分布、探索能力、采样数量有密切关系。

这也让我们看到了 Curriculum Learning（课程学习）和动态采样策略可以发挥作用的地方。

---

## 十一、再进一步：为什么 Multiple Sampling 也是一种 Exploration？

假设模型面对一道从未解决过的困难数学题。

它可能生成：

```text
Response 1:
错误推理路径
Reward = 0

Response 2:
错误推理路径
Reward = 0

Response 3:
正确推理路径
Reward = 1

Response 4:
错误推理路径
Reward = 0
```

如果只生成一条 Response，模型可能恰好没有探索到正确答案。

但是，通过同一个 Prompt 的多次采样，模型有机会尝试不同的推理路径。

一旦采样到了更好的推理轨迹：

$$
A_3>0
$$

策略优化就可以增加这类生成轨迹获得更高概率的倾向。

整个过程可以概括为：

$$
\boxed{
\text{Exploration}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Reinforcement}
}
$$

换成 LLM 的语言：

```text
Sample reasoning paths
          |
          v
Evaluate the responses
          |
          v
Compute relative advantages
          |
          v
Reinforce better trajectories
          |
          v
Updated policy
```

这也解释了为什么 GRPO 特别容易与数学、代码等具有可验证结果的任务结合。

不过需要强调：

**Multiple Sampling 并不能凭空创造模型完全不会生成的推理能力。**

如果当前 Policy 几乎不可能探索到有效答案，那么增加采样数量也未必足够。

模型初始能力、Reward 质量以及探索多样性，仍然十分重要。

---

## 十二、几个容易被忽略的技术细节

### 12.1 GRPO 没有 Value Model，是不是就不需要 Reward Model？

不是。

GRPO 只是去掉了额外训练的 Critic。

它仍然需要为 Response 提供 Reward。

Reward 可以来自学习得到的评分模型，也可以来自可验证规则。

例如，代码生成任务可以使用测试用例通过率作为 Reward。

### 12.2 GRPO 的组内均值就是 Value Function 吗？

不是严格等价。

GRPO 的组内均值是基于当前 Rollout 的样本估计。

而 Value Function 通常学习给定状态下的期望回报。

前者依赖有限采样，后者可以针对不同的状态或 Token 前缀输出预测。

组内均值可以充当 Baseline，但并不意味着它包含 Critic 的全部状态信息。

### 12.3 所有 Reward 相同，是不是整个 GRPO 都停止学习？

不一定。

如果某一组所有 Reward 相同，采用零 Advantage 处理后，这一组没有 Outcome-Relative Policy Signal。

但其他有区分度的组仍然可以参与更新。

而完整目标中的 KL 等其他项也可能产生梯度。

### 12.4 Clipping 能确保 Policy 的概率变化永远不超过 20% 吗？

不能。

\(\epsilon=0.2\) 是 Clipped Surrogate Objective 中的阈值，并不是对新旧 Policy Probability Ratio 的严格硬约束。

实际 Ratio 仍然可能超过对应区间。

### 12.5 如果 Loss 等于零，是不是就没有梯度？

不一定。

例如，当：

```python
new_log_probs = old_log_probs
```

所有 Ratio 等于 1。

对于非退化的组内标准化 Advantage：

$$
\frac1G\sum_i A_i=0
$$

因此，在我们的极简实现中：

$$
J=\frac1G\sum_i A_i=0
$$

但这并不代表在真实的可微 Policy Model 中：

$$
\nabla_\theta J=0
$$

因为不同 Response 的 Log Probability 对模型参数的梯度通常并不相同。

**Loss 的当前数值为零，不等于它对参数的导数为零。**

要注意，本文使用 NumPy 计算静态数值，并没有定义真正的可训练模型或自动微分计算图。

因此，代码实现的是 Loss 数值计算，不是完整的神经网络训练流程。

### 12.6 GRPO 是不是一种没有代价的 PPO 改进？

也不是。

GRPO 移除了 Critic 的训练成本，但仍然需要对同一 Prompt 生成多个 Rollout。

组大小、Response 长度以及奖励评估成本，都会影响实际训练资源消耗。

同时，较少的组内样本也可能导致 Baseline 估计噪声较大。

因此，它本质上是一种工程与统计上的取舍：

**以组采样与相对比较为基础，换取不必额外训练 Value Model 的优化方式。**

---

## 十三、用一张表总结 PPO 与 GRPO

| 比较维度                    | PPO（常见 Actor-Critic 版本）     | GRPO                                |
| ----------------------- | --------------------------- | ----------------------------------- |
| Policy Optimization     | PPO-style Clipped Objective | PPO-style Clipped Objective         |
| Advantage 来源            | Value Function、GAE 等        | 同 Prompt 组内相对 Reward                |
| Critic                  | 需要训练                        | 不需要额外训练                             |
| Group Sampling          | 并非核心要求                      | 核心机制                                |
| Reward                  | Reward Model 或环境反馈          | Reward Model 或规则奖励                  |
| KL Regularization       | LLM PPO 中常见                 | DeepSeekMath 原始目标包含                 |
| Token Credit Assignment | 可利用状态级 Value 信息             | Outcome 版本共享整条 Response 的 Advantage |
| 主要优势                    | 可以利用更细粒度的状态 Value 信息        | 避免额外训练 Critic                       |
| 主要限制                    | Critic 成本与 Value 估计误差       | 组采样成本、相对奖励区分度和信用分配问题                |

---

## 十四、最后总结：我真正理解的 GRPO

如果只看公式：

$$
A_i=\frac{r_i-\mu_r}{\sigma_r}
$$

可能会觉得 GRPO 就是一次简单的标准化。

但是，在理解 PPO、Advantage 和 Baseline 之后，就会发现 GRPO 的关键不在于标准化公式本身，而在于：

**它改变了 Advantage Baseline 的获取方式。**

PPO 使用一个可学习的 Critic 来估计预期回报。

GRPO 则针对相同 Prompt 采样多个 Response，使用组内 Reward 统计构造相对 Advantage。

然后继续使用 PPO-style Clipped Objective 来优化 Policy，并在 DeepSeekMath 原始目标中引入 KL Regularization 控制与 Reference Policy 的偏离。

因此，我会把 GRPO 概括为：

$$
\boxed{
\begin{aligned}
\text{GRPO}
={}&\text{Group-Relative Advantage}\\
&+\text{PPO-style Policy Optimization}\\
&-\text{Learned Critic}
\end{aligned}
}
$$

这里的减号表示移除 Critic 组件，而不是实际的数值减法；原始完整目标还包含 KL 正则化。

如果用一句更直观的话总结：

> **GRPO 通过让模型对同一个问题生成多个答案，并依据它们之间的相对奖励差异进行策略优化，在不额外训练 Value Model 的情况下，为大语言模型提供了强化学习所需的 Advantage 信号。**

而通过今天的实验，我还学到了三个重要的事实：

1. **Reward 高不等于 Advantage 高。** Advantage 取决于相对 Baseline 的表现。
2. **GRPO 的效果依赖 Reward 区分度。** 如果所有 Response 得到相同的 Reward，组内相对优势就无法区分它们。
3. **Clipping 并不直接禁止 Policy 发生较大的概率变化。** 它通过修改 Surrogate Objective，限制某些过度有利更新所带来的额外收益。

到这里，我认为自己终于可以把 GRPO 的核心逻辑串起来了：

```text
同一个 Prompt
      |
      v
采样多个 Response
      |
      v
计算每条 Response 的 Reward
      |
      v
计算 Group Mean / Std
      |
      v
得到 Group-Relative Advantage
      |
      v
计算 New / Old Probability Ratio
      |
      v
PPO-style Clipped Objective
      |
      v
KL Regularization（完整版本）
      |
      v
更新 Policy Model
```

这也是我理解 DeepSeekMath 中 GRPO 的起点。

真正掌握一个算法，并不是能够背下它的公式，而是能够回答：

**这个公式解决了什么问题？为什么要这样设计？如果去掉其中某一项，会发生什么？**

GRPO 让我进一步认识到，强化学习不只是奖励正确答案，而是需要设计一种机制，把奖励转换成有意义、可优化的策略更新信号。

---

## 参考文献

**[1] Shao, Z., et al. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.

* 论文地址：https://arxiv.org/abs/2402.03300
* 论文 PDF：https://arxiv.org/pdf/2402.03300
* 原始项目：https://github.com/deepseek-ai/DeepSeek-Math
* 重点阅读：Section 4.1.1（From PPO to GRPO）、Section 4.1.2（Outcome Supervision RL with GRPO）
* 主要参考内容：GRPO 的提出动机、组内相对 Advantage、完整目标函数、Value Model 替代机制及 KL Regularization。

**[2] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

* 论文地址：https://arxiv.org/abs/1707.06347
* 论文 PDF：https://arxiv.org/pdf/1707.06347
* 重点阅读：Section 3（Clipped Surrogate Objective）
* 主要参考内容：Probability Ratio、Clipped Objective、策略更新稳定性以及 PPO 优化目标。

**[3] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).** *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438.

* 论文地址：https://arxiv.org/abs/1506.02438
* 主要参考内容：Advantage Estimation、Value Function 与 GAE 的基本思想。
