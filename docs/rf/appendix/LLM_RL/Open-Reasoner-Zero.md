# Day 13｜Open-Reasoner-Zero：PPO 真的过时了吗？从 Critic 到 Reasoning RL 的 Training Dynamics

> **论文精读系列 · Reasoning RL**
>
> 核心关键词：Open-Reasoner-Zero、PPO、GRPO、GAE、Critic、Credit Assignment、Training Dynamics

## 一、引言：为什么这篇论文值得读？

在大语言模型的强化学习（Reinforcement Learning，RL）领域，随着 DeepSeek-R1 的出现，GRPO（Group Relative Policy Optimization）逐渐成为 Reasoning RL 中备受关注的训练算法。

GRPO 最吸引人的地方之一，就是它不再需要 PPO 中的 Critic（价值模型），而是通过比较同一个问题的多个回答来估计 Advantage。

这似乎带来了一个很自然的判断：

既然 GRPO 可以省掉 Critic，那么 PPO 是不是已经过时了？

然而，NeurIPS 2025 论文 **Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model** 给出了一个值得重新思考的答案。[1]

作者采用了一套非常简洁的训练方案：

* 直接从 Base Model 开始强化学习，不需要预先进行 SFT。
* 使用传统 PPO 进行策略优化。
* 使用 GAE 估计 Advantage，并设置 \(\gamma=1,\lambda=1\)。
* 使用基于最终答案正确性的 Rule-based Reward。
* 不使用额外的 KL Regularization。

尽管方法相对简单，ORZ 仍然成功展示了从 0.5B 到 32B 模型规模的 Reasoning RL 训练效果。

更重要的是，作者通过实验研究了 Critic 在长推理轨迹中的作用，特别是它如何学习识别重复生成等退化模式，以及如何影响 Advantage Estimation。

这篇论文真正值得我们学习的，不是“PPO 比 GRPO 更强”，而是一个更加底层的问题：

**Reasoning RL 的训练效果，究竟是由算法名称决定的，还是由整个训练系统产生的 Learning Signal 和 Training Dynamics 决定的？**

本文将围绕三个问题展开：

1. PPO 和 GRPO 在 Reasoning RL 中究竟有什么区别？
2. Critic 为什么可能对长链推理有帮助？
3. 为什么说理解 Training Dynamics 比单纯比较算法名称更重要？

---

## 二、从一个数学问题出发：Reasoning RL 究竟在优化什么？

理解这篇论文前，需要先弄清楚，大语言模型的强化学习究竟在做什么。

### 2.1 把语言模型看成一个 Policy

假设我们希望模型解决下面的问题：

> 解方程：\(2x+4=10\)。

模型可能生成：

```text
Question:
Solve 2x + 4 = 10.

Response:
First, subtract 4 from both sides.

2x = 6

Then divide both sides by 2.

x = 3

<answer>3</answer>
```

在传统监督微调（SFT）中，我们通常会提供一个标准回答，然后通过交叉熵损失让模型学习生成对应的 Token。

但在 Outcome-based Reasoning RL 中，我们可以采用另一种训练方式：

**不要求模型复现某一条标准推理过程，而是允许模型自己探索，只根据最终答案给予奖励。**

例如：

```text
                    Math Question
                          |
                          v
                     Policy Model
                          |
                          v
                 Generate Reasoning
                          |
                          v
                    Final Answer
                          |
                          v
                   Rule-based Verifier
                          |
                  +-------+-------+
                  |               |
                Correct         Wrong
                  |               |
                  v               v
                R = 1           R = 0
```

这里有三个非常重要的概念。

**State（状态）**

对于自回归语言模型，状态可以理解为当前输入问题和此前已经生成的 Token：

$$
s_t=(q,a_0,a_1,\ldots,a_{t-1})
$$

**Action（动作）**

模型在当前状态下生成的下一个 Token：

$$
a_t\sim\pi_\theta(\cdot\mid s_t)
$$

**Reward（奖励）**

通过 Verifier 对完整回答进行评价：

$$
R=
\begin{cases}
1,&\text{答案正确}\\
0,&\text{答案错误}
\end{cases}
$$

于是，一个完整回答就构成了一条 Trajectory（轨迹）：

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_{T-1},a_{T-1})
$$

模型训练的目标，是让能够获得高 Reward 的生成行为更有可能发生。

### 2.2 真正的困难：只有最终 Reward，怎样评价中间的推理？

假设模型生成了一段很长的推理：

```text
Step 1: 正确理解题意
Step 2: 正确建立方程
Step 3: 正确进行代数变换
Step 4: 计算时出现错误
Step 5: 沿着错误结果继续推导
Step 6: 得到错误答案
```

最终：

$$
R=0
$$

这时候问题来了。

如果我们仅仅知道最终答案错误，那么该如何判断：

* 哪些 Token 值得鼓励？
* 哪些 Token 应该降低生成概率？
* 错误是从哪里开始出现的？
* 整条回答是否都应该受到相同程度的惩罚？

这就是强化学习中的 **Credit Assignment（信用分配）** 问题。

它是理解 PPO、GRPO，以及 ORZ 中 Critic 价值的重要入口。

需要注意，Credit Assignment 不一定意味着精确找到“第一个错误 Token”。对于只有最终结果奖励的训练，真正困难的是如何从稀疏的结果信号中估计各个动作的学习价值。

---

## 三、Advantage：强化学习为什么不直接使用 Reward？

既然已经有了 Reward，为什么不能直接根据 Reward 更新模型？

原因是：**Reward 只能告诉我们结果怎么样，却不能充分说明这个结果相对于预期有多好。**

### 3.1 一个直观例子

假设模型遇到了两个问题。

**问题 A：**

一道非常简单的加法题，模型通常都能答对。

**问题 B：**

一道非常困难的数学证明相关问题，模型通常很难得到正确结果。

现在，两道题都回答正确，Reward 都为 1。

但从强化学习的角度来看，这两次成功提供的信息可能并不相同。

对于问题 A，模型本来就有很高概率答对。

对于问题 B，模型原本成功的概率很低，这次正确回答可能更加值得关注。

因此，我们希望引入一个 Baseline（基线），衡量当前结果相对于预期的好坏。

这就引出了 Advantage。

### 3.2 Advantage 的定义

强化学习中的 Advantage Function 定义为：

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

其中：

* \(Q^\pi(s_t,a_t)\)：在当前状态下采取动作 \(a_t\)，之后继续按照策略 \(\pi\) 运行的预期回报。
* \(V^\pi(s_t)\)：在当前状态下按照策略 \(\pi\) 继续行动的预期回报。

因此：

$$
\boxed{Advantage=Action\ Value-State\ Value}
$$

它回答的是：

> 在当前状态下，采取这个动作，相比按照当前策略的平均表现，究竟好多少？

一般可以这样理解：

* \(A>0\)：动作的结果比预期更好，倾向于提高其概率。
* \(A<0\)：动作的结果比预期更差，倾向于降低其概率。
* \(A\approx0\)：动作没有体现出明显优于或差于基线的证据。

这里需要区分：

**理论上的 Advantage** 是 \(Q^\pi-V^\pi\)，而实际训练中的 Advantage 通常是通过采样和估计算法获得的近似值。

ORZ 使用的正是 GAE（Generalized Advantage Estimation）。[3]

---

## 四、Critic 到底是什么？为什么 PPO 需要它？

这是本文最核心的部分之一。

### 4.1 Actor 和 Critic 的分工

在 Actor-Critic 架构中，我们通常有两个模型。

**Actor（策略模型）**

$$
\pi_\theta(a_t\mid s_t)
$$

负责决定：

> 当前应该生成哪个 Token？

**Critic（价值模型）**

$$
V_\phi(s_t)
$$

负责预测：

> 给定当前生成到这里的上下文，如果之后继续按照当前策略生成，最终预计能获得多大的回报？

可以将它们理解为：

```text
                   Current State
                        |
            +-----------+-----------+
            |                       |
            v                       v
          Actor                   Critic
            |                       |
            v                       v
      Choose Next Token      Estimate State Value
            |                       |
            +-----------+-----------+
                        |
                        v
                 Policy Optimization
```

在 ORZ 中，Actor 和 Critic 使用不同的训练参数，并且 Critic 具有用于预测 Value 的输出头。[1]

### 4.2 Critic 预测的究竟是什么？

假设最终 Reward 只有 0 或 1。

理想情况下：

$$
V^\pi(s_t)=\mathbb{E}_\pi[R\mid s_t]
$$

由于 Reward 是二值的，因此它可以被理解为：

**从当前状态继续生成，最终回答正确的预期概率。**

例如，假设某条推理路径存在如下状态：

| State   | 当前推理状态   | 假设的 Value |
| ------- | -------- | --------: |
| \(s_0\) | 刚开始解题    |      0.40 |
| \(s_1\) | 正确建立方程   |      0.65 |
| \(s_2\) | 找到关键数学关系 |      0.82 |
| \(s_3\) | 开始出现逻辑问题 |      0.55 |
| \(s_4\) | 陷入大量重复文本 |      0.10 |

注意，这些数字只是为了帮助理解而构造的例子，并不是 ORZ 的实际实验输出。

它展示了一个重要特点：

**Critic 的输出可以随着推理上下文变化，而不是只给整个回答一个固定分数。**

不过要注意，Critic 预测的是 Value，而不是直接判断当前 Token 在数学上是否正确。

实际训练中的神经网络 Value 预测也未必严格限制在 0 到 1 的区间内。

### 4.3 Critic 没有中间步骤标签，为什么还能学习？

这是一个非常重要的问题。

ORZ 的 Reward 只来自最终答案，并没有为每个推理步骤提供人工标注。

那么 Critic 凭什么能够学习状态价值？

答案是：

**Critic 可以把最终 Reward 作为不同中间状态的监督目标。**

在 ORZ 的特殊设置中，其 Value Loss 可以写为：

$$
L_V(\phi)
=
\frac12
\mathbb{E}_{\tau,t}
\left[
(V_\phi(s_t)-R)^2
\right]
$$

也就是说，对于一次生成的多个中间状态，Critic 都使用这条轨迹最终获得的 Reward 作为回归目标。

假设模型反复生成不同的推理轨迹：

```text
Trajectory A:
合理分析 → 正确推导 → 正确答案
R = 1

Trajectory B:
合理分析 → 错误计算 → 错误答案
R = 0

Trajectory C:
开始重复 → 持续重复 → 未能正确作答
R = 0

Trajectory D:
发现错误 → 重新推导 → 正确答案
R = 1
```

经过大量样本训练后，Critic 可以学习不同状态特征与最终成功概率之间的统计关系。

从数学上说，在模型容量足够、数据分布合适、优化充分等理想条件下，最小化均方误差的最优预测是：

$$
V^*(s)=\mathbb{E}[R\mid s]
$$

这就是为什么仅有 Outcome Reward，Value Model 仍然有可能学习到有用的状态信息。

但这里有一个非常重要的边界：

**Critic 学习到的是状态与最终回报之间的预测关系，并不代表它真正知道哪个 Token 是导致失败的根本原因。**

这一点会在后文分析 ORZ 的 repetition 实验时再次出现。

---

## 五、GAE：ORZ 为什么设置 \(\gamma=1,\lambda=1\)？

理解了 Critic，接下来就可以理解它如何参与 Advantage Estimation。

### 5.1 GAE 的基本公式

GAE 的定义是：

$$
\hat A_t^{GAE(\gamma,\lambda)}
=
\sum_{k=0}^{T-t-1}
(\gamma\lambda)^k\delta_{t+k}
$$

其中 TD Error 为：

$$
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

两个超参数分别是：

**\(\gamma\)：Discount Factor（折扣因子）**

控制未来 Reward 对当前状态的影响程度。

**\(\lambda\)：GAE 参数**

控制估计中不同时间跨度 TD Error 的权重，并影响 Bias-Variance Trade-off。

在一般强化学习问题中，\(\gamma\) 和 \(\lambda\) 往往小于 1。

但 ORZ 选择：

$$
\boxed{\gamma=1,\quad\lambda=1}
$$

### 5.2 为什么 \(\gamma=1\)？

Reasoning RL 的一个特殊之处，是推理长度可能非常长。

假设最终只有一个 Terminal Reward。

如果使用：

$$
\gamma=0.99
$$

那么距离最终奖励很远的 Token，其折扣权重可能非常小。

例如：

$$
0.99^{1000}\approx4.32\times10^{-5}
$$

这意味着在这种折扣设置下，距离最终奖励 1000 步的未来结果，其直接折扣贡献已经很弱。

因此，较小的 \(\gamma\) 可能产生不希望看到的长度偏好，使模型更倾向于较早结束推理。

ORZ 作者认为长推理任务需要充分考虑长期依赖，因此选择 \(\gamma=1\)。[1]

这相当于不因为 Reward 距离当前 Token 很远，就自动降低它的重要性。

但 \(\gamma=1\) 本身不会保证模型一定愿意进行高质量的长推理。生成长度还受到 Reward、数据分布、采样方式和优化策略等因素影响。

### 5.3 为什么 \(\lambda=1\)？

\(\lambda\) 主要影响 Advantage 估计中的 Bias-Variance Trade-off。

一般情况下：

* 较小的 \(\lambda\)：更多依赖 Bootstrapping，通常降低方差，但可能引入更多估计偏差。
* 较大的 \(\lambda\)：更多依赖完整采样回报，降低 Bootstrapping 引入的偏差，但可能增加方差。

ORZ 作者认为，在大规模 Rollout 数据支持下，可以通过更多采样来缓解方差问题。

因此，选择 \(\lambda=1\)，让 Advantage 估计充分利用完整的未来回报。

注意，\(\lambda=1\) 并不意味着实际 Advantage 估计完全没有误差。采样噪声、Critic 的函数逼近误差仍然可能存在。

### 5.4 关键推导：GAE 为什么可以变成 \(R-V(s_t)\)？

这是 ORZ 最值得手动推导的公式。

首先，当：

$$
\gamma=1,\quad\lambda=1
$$

GAE 变成：

$$
\hat A_t
=
\sum_{k=t}^{T-1}\delta_k
$$

将 TD Error 展开：

$$
\hat A_t
=
\sum_{k=t}^{T-1}
\left[
r_k+V_\phi(s_{k+1})-V_\phi(s_k)
\right]
$$

把 Reward 和 Value 分开：

$$
\hat A_t
=
\sum_{k=t}^{T-1}r_k
+
\sum_{k=t}^{T-1}
\left[
V_\phi(s_{k+1})-V_\phi(s_k)
\right]
$$

注意第二项：

$$
\begin{aligned}
&V(s_{t+1})-V(s_t)\\
+{}&V(s_{t+2})-V(s_{t+1})\\
+{}&V(s_{t+3})-V(s_{t+2})\\
&\cdots\\
+{}&V(s_T)-V(s_{T-1})
\end{aligned}
$$

中间的 Value 会相互抵消。

这在数学上称为 **Telescoping Sum（望远镜求和）**。

最终得到：

$$
\hat A_t
=
\sum_{k=t}^{T-1}r_k
+
V(s_T)-V(s_t)
$$

由于 ORZ 只在终止时提供 Reward：

$$
\sum_{k=t}^{T-1}r_k=R
$$

并且终止状态：

$$
V(s_T)=0
$$

因此：

$$
\boxed{\hat A_t=R-V_\phi(s_t)}
$$

推导完成。

进一步，Value Target 为：

$$
V_t^{target}
=
\hat A_t+V_\phi(s_t)
=
R
$$

所以 Critic 的训练目标也被简化为：

$$
\boxed{
L_V=
\frac12\mathbb E[(V_\phi(s_t)-R)^2]
}
$$

这意味着 ORZ 的 GAE 在该设置下，退化为一种使用完整 Terminal Return 的 Monte Carlo 风格 Advantage Estimation。

### 5.5 这个公式究竟意味着什么？

$$
\hat A_t=R-V_\phi(s_t)
$$

可以理解为：

> Advantage = 最终实际得到的结果 − 当前状态下原本预计得到的结果。

假设某个状态的 Critic 预测：

$$
V(s_t)=0.2
$$

但最终回答正确：

$$
R=1
$$

则：

$$
A_t=0.8
$$

说明这条采样轨迹的最终结果明显好于该状态下的预期。

反过来，Critic 预测：

$$
V(s_t)=0.8
$$

但最终回答错误：

$$
R=0
$$

则：

$$
A_t=-0.8
$$

说明这次结果明显低于预期。

此时应该记住一个核心区别：

**Reward 是整条 Trajectory 的最终结果，而 Value 是与当前 State 有关的预测。**

因此，即使同一条回答只有一个 Reward，不同 Token 位置的 Advantage 也可能不同。

这就是 ORZ 使用 Critic 的重要动机之一。

---

## 六、PPO 和 GRPO 究竟有什么区别？

现在我们可以真正比较两种算法了。

### 6.1 PPO：使用学习得到的 Value Baseline

ORZ 使用 PPO，通过 Critic 估计：

$$
V_\phi(s_t)
$$

再计算：

$$
\hat A_t=R-V_\phi(s_t)
$$

因此 Advantage 可以随 Token 位置变化。

但 PPO 还需要考虑一个问题：

如果 Advantage 为正，就不断增大对应 Token 的概率，模型会不会一次更新得太激进？

这就需要 PPO 的 Clipped Objective。[4]

$$
L_{\text{PPO}}(\theta)
=
\mathbb E_t
\left[
\min
\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

其中：

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\text{old}}}(a_t\mid s_t)
}
$$

\(\rho_t\) 表示当前 Policy 与采样时旧 Policy 对同一个 Token 的概率比值。

Clipping 的直观作用是：

**限制过度改变 Token 概率所带来的目标函数收益，避免策略更新过于激进。**

在 ORZ 中：

$$
\epsilon=0.2
$$

这里要特别区分两个概念：

* **PPO Clipping**：限制策略更新目标中的概率比。
* **KL Regularization**：额外限制策略与参考模型之间的分布偏离。

ORZ 不使用额外的 KL Regularization，但仍然保留 PPO Clipping。

所以：

**No KL Regularization 不等于完全没有约束的策略更新。**

### 6.2 GRPO：不训练 Critic，直接比较一组回答

GRPO 最重要的变化，是不再需要一个独立的 Critic 网络。[5]

假设给同一道题采样四条回答：

```text
Question Q
    |
    +-- Response 1 --> Reward = 1
    |
    +-- Response 2 --> Reward = 0
    |
    +-- Response 3 --> Reward = 1
    |
    +-- Response 4 --> Reward = 0
```

GRPO 可以通过组内的 Reward 均值和标准差构造 Advantage：

$$
\hat A_i^{GRPO}
=
\frac{
R_i-\operatorname{mean}(R_1,\ldots,R_G)
}{
\operatorname{std}(R_1,\ldots,R_G)
}
$$

这里 \(G\) 表示同一个 Prompt 下采样的回答数量。

在这个例子中，组内 Reward 的均值为：

$$
\mu=0.5
$$

标准差为：

$$
\sigma=0.5
$$

于是得到：

| Response   | Reward | GRPO Advantage |
| ---------- | -----: | -------------: |
| Response 1 |      1 |             +1 |
| Response 2 |      0 |             -1 |
| Response 3 |      1 |             +1 |
| Response 4 |      0 |             -1 |

所以 GRPO 的基本思路是：

**不需要学习这个 State 的绝对价值，只需要判断当前回答相对于同组其他回答表现如何。**

在经典 Outcome-supervised GRPO 中，同一条 Response 的各个 Token 通常共享相同的组相对 Advantage。

当然，GRPO 依然会使用 PPO 风格的概率比及 Clipping 目标进行优化。

因此，GRPO 并不是彻底推翻 PPO，而是对 PPO 一类策略优化方法的改造。

### 6.3 一个容易忽略的问题：如果一组回答全错呢？

假设：

```text
Response 1 --> R = 0
Response 2 --> R = 0
Response 3 --> R = 0
Response 4 --> R = 0
```

此时：

$$
\operatorname{mean}(R)=0
$$

并且：

$$
\operatorname{std}(R)=0
$$

原始归一化公式会遇到除零问题。

实际实现通常会使用数值稳定处理；对于上述简单组相对估计，如果所有 Reward 相同，那么 Advantage 可以全部置为 0。

这意味着：

**这组回答无法提供组内好坏的相对区分信号。**

如果额外配置了 KL 等损失项，这些项仍然可能产生更新；这里讨论的只是组相对 Advantage 所提供的 Policy Gradient 信号。

反过来，在使用 Critic 的 PPO 中，即使一组回答的最终 Reward 都是 0，只要：

$$
V_\phi(s_t)\neq0
$$

仍然可能产生非零的：

$$
A_t=0-V_\phi(s_t)
$$

这就是两种 Baseline 的一个差异。

但 Critic 也不是万能的。如果它最终认为这些状态的成功概率都接近 0，那么对应 Advantage 也会趋近于 0。

### 6.4 PPO 和 GRPO 的核心对比

| 比较维度                        | PPO（ORZ 设置）            | 经典 Outcome GRPO           |
| --------------------------- | ---------------------- | ------------------------- |
| 是否需要 Critic                 | 需要                     | 不需要                       |
| Advantage 来源                | \(R-V(s_t)\)           | 组内 Reward 标准化             |
| 是否可以对不同 Token 给不同 Advantage | 可以                     | 通常共享同一 Response Advantage |
| 是否需要多个 Rollout              | 可以使用多个                 | 依赖组内多个采样                  |
| Policy 更新                   | PPO Clipping           | PPO 风格 Clipping           |
| 主要额外成本                      | Critic 训练与显存           | 组采样及对应 Rollout 成本         |
| 潜在优势                        | 状态相关的 Value Estimation | 避免训练 Value Model          |
| 潜在问题                        | Value 估计误差与训练成本        | 组内缺少有效 Reward 差异时信号不足     |

需要特别强调：

**PPO 不意味着只能采样一条回答，GRPO 也不意味着不需要 Clipping。**

二者在实际 LLM RL 系统中的区别，不能仅仅简化成“采样数量不同”。

最关键的一个区别是 Advantage Baseline 的构造方式。

---

## 七、Open-Reasoner-Zero 的训练方案到底有多简单？

理解完算法，我们再回到 ORZ 的完整训练系统。

### 7.1 核心训练流程

ORZ 的主要流程可以概括为：

```text
                Base Language Model
                         |
                         v
                   Sample Prompts
                         |
                         v
                  Policy Rollouts
                         |
                         v
                 Generated Responses
                         |
                         v
                  Rule-based Reward
                         |
                         v
                     R = 0 / 1
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Critic Network           Old Policy
             |                       |
             v                       |
          V(s_t)                     |
             |                       |
             v                       |
      A_t = R - V(s_t)               |
             |                       |
             +-----------+-----------+
                         |
                         v
                 PPO Clipped Update
                         |
                         v
                    New Policy
                         |
                         v
                       Repeat
```

ORZ 的关键设计包括：

| 模块                | ORZ 采用的方案                |
| ----------------- | ------------------------ |
| 初始模型              | Qwen2.5 Base 等           |
| 预训练后的 SFT 冷启动     | 核心 Reasoner-Zero 方案不需要   |
| 强化学习算法            | PPO                      |
| Advantage 估计      | GAE                      |
| Discount Factor   | \(\gamma=1\)             |
| GAE 参数            | \(\lambda=1\)            |
| Reward            | Binary Rule-based Reward |
| 额外 Format Reward  | 不使用                      |
| KL Regularization | 不使用                      |
| Entropy Bonus     | 不使用                      |
| PPO Clip Range    | 0.2                      |

来源：论文 Section 2 和 Supplementary Material Section B。[1]

### 7.2 ORZ 并不是只在算法上做减法

论文附录还有一些特别值得关注的实现设置：

* 每轮采样 128 个不同的 Prompt。
* 每个 Prompt 生成 64 条 Response。
* 每轮对应一次 Policy 优化更新。
* Critic 每轮执行 12 次优化更新。
* 对 Advantage 进行 Batch-level Normalization。

这样一轮生成会得到：

$$
128\times64=8192
$$

条采样 Trajectory。

这里非常值得注意：

**即使使用 PPO，ORZ 仍然为同一个 Prompt 生成多个 Response。**

所以“多次采样”不是 GRPO 独有的特点。

两者真正的差异，在于如何利用这些 Rollout 估计 Advantage，以及如何组织训练。

另外，Critic 并不是免费获得的。ORZ 为 Critic 配置了独立参数和专门的优化步骤。

因此，是否引入 Critic，本质上仍然是一种计算资源与训练信号质量之间的权衡。

### 7.3 为什么只使用 Binary Reward？

ORZ 的 Reward 只检查最终答案是否正确。

例如，从生成内容的 `<answer>...</answer>` 中提取答案，再与标准答案进行匹配。

如果正确：

$$
R=1
$$

否则：

$$
R=0
$$

它没有额外给：

```text
写出 <think>        +0.1
思考过程很长         +0.2
使用特定反思词       +0.1
推理格式漂亮         +0.1
```

这样的辅助奖励。

作者发现，使用设计好的 Prompt 和简单结果奖励，Base Model 也能逐渐学习需要的输出格式。

这给我们一个工程启示：

**Reward 不一定越复杂越好。**

复杂 Reward 可能带来更多调参成本，也可能出现 Reward Hacking，即模型学会利用奖励函数漏洞，而不是真正提升任务能力。

不过，简单 Reward 也有代价。

最终答案正确不代表中间每一步推理都正确；答案匹配器也可能存在错误或覆盖不足。因此，Rule-based Reward 是否可靠，仍然取决于任务本身是否容易验证，以及 Verifier 的质量。

---

## 八、论文最值得关注的实验：Critic 能识别 Repetition 吗？

前面讲了大量概念，现在终于可以理解 ORZ 最有意思的实验之一。

### 8.1 什么是 Repetition Collapse？

在长链推理训练中，模型有时会产生大量重复内容。

例如：

```text
We can simplify the expression:

x = 26 / 51

Therefore:
x = 26 / 51

Again:
x = 26 / 51

Thus:
x = 26 / 51

x = 26 / 51
x = 26 / 51
...
```

模型没有继续推进有效推理，而是陷入重复生成。

这种现象不仅浪费 Token，还可能让模型无法正常完成回答。

在一些训练过程中，重复问题会变得越来越严重，甚至造成生成质量的明显退化。

### 8.2 ORZ Figure 5 观察到了什么？

在论文 Figure 5 中，作者分析了 Critic 对推理状态的 Value Prediction。

他们发现，Critic 会对具有明显重复模式的状态给出相对较低的 Value，而对连贯的正常推理上下文给出较高的 Value。

这说明，Critic 可能从大量训练轨迹中学到了：

> 某些重复状态往往意味着后续获得正确答案的概率较低。

这并不是因为 Critic 被明确告知“重复是错误行为”。

而是因为它在预测最终 Reward 的过程中，发现了这些状态与后续结果之间的统计关系。

### 8.3 更关键的实验：PPO Advantage vs GRPO Advantage

作者进一步分析了重复片段对应的 Advantage。

实验方法大致为：

1. 识别一条生成轨迹中开始出现显著重复模式的位置。
2. 将其之后的 Token 视为具有重复模式的 Token。
3. 计算 PPO 对这些 Token 给出的 Advantage。
4. 在同一批 Token 上计算假设使用 GRPO 时的组相对 Advantage。
5. 比较两种 Advantage 的分布差异。

注意，这里 PPO 的 Advantage 还经过了 Batch-level Normalization。

Figure 5 显示：

**在 ORZ-7B 大部分训练迭代中，PPO 对重复 Token 给出的 Advantage，比假设使用 GRPO 得到的 Advantage 更低、更偏负。**

作者认为，这说明 Critic 可以提供更有区分度的信号，从而有助于抑制退化生成模式。[1]

### 8.4 论文还做了直接的稳定性对照

除了 Figure 5 中的假设 GRPO Advantage 比较，论文附录 Figure 8 还给出了 ORZ-7B 设置下的 PPO 与 GRPO 训练稳定性对比。

作者观察到：

* PPO 在实验期间维持了相对稳定的 Reward。
* GRPO 在大约 240 个训练步骤附近出现明显不稳定。
* GRPO 的 Truncate Rate 和 Repeat Score 随之显著升高。

这为论文关于 Critic 和训练稳定性的讨论提供了额外实验证据。

但这仍然是特定训练配置下的实验结果，不意味着任何 GRPO 实现都会出现相同问题。

### 8.5 一个必须弄清楚的细节：Value 低，不等于 Advantage 一定更负

这是阅读 ORZ 时非常容易产生的误解。

我们已经知道：

$$
A_t=R-V(s_t)
$$

假设一条回答最终失败：

$$
R=0
$$

在正常推理状态下：

$$
V(s_{\text{normal}})=0.8
$$

那么：

$$
A_{\text{normal}}=-0.8
$$

而在重复状态下：

$$
V(s_{\text{repeat}})=0.1
$$

则：

$$
A_{\text{repeat}}=-0.1
$$

结果发现：

**重复状态的 Value 更低，但这个失败样本中对应的 Advantage 反而没有那么负。**

为什么？

因为 Critic 已经认为这个状态的成功希望很小。

失败本身并没有大幅低于预期。

这说明我们不能简单地写出以下推理：

```text
Critic 发现 repetition
        ↓
Value 很低
        ↓
Advantage 一定更负
        ↓
直接强烈惩罚 repetition
```

这个因果链在数学上并不总是成立。

同样，如果一个被预测为低 Value 的状态最终获得了 \(R=1\)，那么：

$$
A_t=1-V(s_t)
$$

反而可能产生较大的正 Advantage。

因此，**Critic 并不是一个直接给错误 Token 贴负面标签的分类器。**

它提供的是 State-dependent Baseline。

其对策略更新的影响取决于 Reward、Value、采样轨迹分布、Advantage Normalization 和 PPO 优化过程的共同作用。

ORZ Figure 5 证明的是：

> 在作者分析的训练数据和配置中，使用 Critic 的 PPO 最终对重复 Token 产生了更负的平均 Advantage 信号。

而不是证明：

> 只要 Critic 给重复状态低 Value，它就必然强烈惩罚这些 Token。

理解这个区别，才能真正从数学公式过渡到 Training Dynamics，而不是停留在直觉解释上。

---

## 九、Training Dynamics：为什么它比算法名字更重要？

这篇论文给我的一个重要启发是：

**评估 Reasoning RL 训练，不应该只看最后的 Benchmark Score。**

模型的训练过程本身，也包含大量有价值的信息。

### 9.1 Reward 在上涨，模型就一定变好了吗？

不一定。

Reward 上涨可能意味着模型能力增强，也可能受到训练数据难度、采样分布和 Reward 设计的影响。

如果 Reward 本身有漏洞，模型甚至可能通过 Reward Hacking 获得更高分数。

因此，除了 Training Reward，还需要关注独立验证集上的表现。

### 9.2 Response Length 越长越好吗？

也不一定。

在 Reasoning RL 中，Response Length 增长可能表示模型开始进行更深入的探索和自我检查。

但同样可能表示：

* 模型出现无效重复。
* 模型无法及时终止生成。
* 模型产生大量没有贡献的中间步骤。
* 模型越来越倾向于输出冗长回答。

因此：

$$
\boxed{\text{Longer Reasoning}\neq\text{Better Reasoning}}
$$

在 ORZ 的 Figure 2 中，作者展示了不同模型规模训练时 Reward 和 Response Length 的变化趋势。

值得注意的是，ORZ-32B 的 Response Length 存在一定波动，但 Reward 仍然持续提升。

这说明：

**Response Length 暂时波动，并不能单独证明训练已经 Collapse。**

### 9.3 真正值得监控哪些指标？

如果自己实现一个 Reasoning RL 训练框架，我认为至少应该监控以下指标。

| 指标                     | 主要观察什么                  | 潜在异常                   |
| ---------------------- | ----------------------- | ---------------------- |
| Training Reward        | 当前采样的平均回报               | 长时间停滞、突然下降             |
| Eval Accuracy          | 独立测试集上的正确率              | 训练提升但泛化退化              |
| Response Length        | 生成 Token 数量             | 异常暴涨或骤降                |
| Truncate Rate          | 达到长度限制而被截断的比例           | 大量回答无法完成               |
| Repeat Score           | 重复生成程度                  | 出现 Repetition Collapse |
| Policy Entropy         | 输出分布的不确定性               | 探索能力异常下降               |
| Approx. KL             | 新旧策略的分布变化               | 策略更新漂移过大               |
| Clip Fraction          | PPO 更新中被 Clipping 影响的比例 | 更新可能过于激进               |
| Critic Loss            | Value 预测误差              | Critic 难以拟合 Reward     |
| Advantage Distribution | 更新信号的均值和分布              | 大量极端值或异常变化             |
| Group Reward Variance  | 同题采样的结果差异               | GRPO 缺少有效组内对比信号        |

不同训练框架对这些指标的计算定义可能不同，因此跨实验比较时需要保持口径一致。

关键不在于把 Dashboard 做得多复杂，而在于能够回答：

> 当前模型是如何变强的？又是从什么时候开始出现不稳定迹象的？

### 9.4 ORZ 的 Ablation 给了哪些启发？

论文 Figure 3 对几个关键设计进行了消融实验。

**第一，GAE 的 \(\lambda\) 不是固定答案。**

在作者的设置中：

$$
\lambda=1
$$

相比：

$$
\lambda=0.95
$$

获得了更好的 Reward Progression 和 Response Length Dynamics。

这说明经典 RL 中经常使用的超参数，不一定在长链推理训练中仍然最合适。

**第二，KL Regularization 并不是始终必需。**

作者对比了不同 KL 方案，发现在其 ORZ-7B 设置下，不使用额外 KL 正则反而获得了更好的训练表现。

但这不能推广为“所有 LLM RL 都应该移除 KL”。

KL 可能在其他模型、Reward 设置或分布偏移条件下发挥重要作用。

**第三，Data Scale 和 Data Quality 非常关键。**

作者比较了约 7.5k 的 MATH Train 数据和约 57k 的 ORZ Curated Dataset。

实验中，更大且更加多样化的数据支持了更持续的训练改进，而较小数据集更早出现性能平台期。

需要注意，数据规模和数据组成同时发生了变化，因此不能把所有效果都归因于样本数量。

此外，官方 GitHub 后续发布了扩展数据，包含原来的 57k、额外约 72k，以及用于困难样本训练的约 13k 子集。[2]

阅读论文和代码时，应区分论文中的具体消融配置与后来仓库发布的完整数据版本。

---

## 十、论文的效果到底怎么样？应该如何解读？

ORZ 最核心的主张，是证明简单的 PPO Recipe 也能支持大规模 Reasoning RL。

论文 Table 1 给出了 ORZ-32B 与 DeepSeek-R1-Zero-Qwen-32B 的部分结果。[1]

| Benchmark    | DeepSeek-R1-Zero-Qwen-32B | ORZ-32B |
| ------------ | ------------------------: | ------: |
| AIME 2024    |                      47.0 |    48.1 |
| MATH500      |                      91.6 |    92.2 |
| GPQA Diamond |                      55.0 |    55.5 |

从表格可以看到，ORZ 在这些报告指标上略高于对应的 DeepSeek-R1-Zero-Qwen-32B。

作者还报告，在使用相同 Qwen2.5-32B Base Model 的背景下，ORZ 使用了约十分之一的训练步骤。

但这里要保持一个清醒的认识：

**Training Steps 更少，不等于严格意义上 Training FLOPs、GPU Hours 或实际训练成本也按相同比例减少。**

不同系统的每一步可能采用不同 Rollout 数量、序列长度和优化方式。

同样，Benchmark 的差异也不能单独证明 PPO 比 GRPO 更优，因为训练数据、Reward、工程实现等变量并不完全一致。

这篇论文真正的意义，是给出了一个强有力的反例：

**不使用 GRPO，也可以依靠传统 PPO 构建有效的大规模 Reasoning RL 系统。**

与此同时，论文还在其他模型规模、模型家族以及混合任务设置中验证了该训练方案。

不过，作者也指出，对更加广泛的代码推理和 Test-time Scaling 等方向仍然需要进一步研究。

---

## 十一、动手验证：用 Python 理解 GAE 和 GRPO Advantage

仅仅阅读公式，可能仍然难以真正建立直觉。

这里使用一个完全不依赖深度学习框架的小实验，验证前面介绍的核心数学关系。

**注意：这只是 Advantage Estimation 的教学示例，并不是完整的 PPO 或 GRPO 模型训练代码。**

```python
from math import sqrt, isclose


def gae(rewards, values, gamma=1.0, lam=1.0):
    """
    计算 Generalized Advantage Estimation。

    rewards: 每个 step 的 reward
    values: 每个 state 的 value，最后一个是终止状态 value
    """
    if not rewards or len(values) != len(rewards) + 1:
        raise ValueError(
            "rewards must be nonempty; values must include terminal value"
        )

    advantages = [0.0] * len(rewards)
    running = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        delta = (
            rewards[t]
            + gamma * values[t + 1]
            - values[t]
        )

        running = delta + gamma * lam * running
        advantages[t] = running

    return advantages


def grpo_advantages(group_rewards):
    """
    简化版 Outcome-based GRPO Advantage。
    使用总体标准差，并处理全同 Reward 的情况。
    """
    if not group_rewards:
        raise ValueError("group_rewards must be nonempty")

    mean = sum(group_rewards) / len(group_rewards)

    variance = sum(
        (r - mean) ** 2 for r in group_rewards
    ) / len(group_rewards)

    std = sqrt(variance)

    if std < 1e-12:
        return [0.0] * len(group_rewards)

    return [
        (r - mean) / std
        for r in group_rewards
    ]


if __name__ == "__main__":
    values = [0.4, 0.6, 0.2, 0.0]

    # 实验一：最终答对
    correct = gae([0, 0, 1], values)

    # 实验二：最终答错
    wrong = gae([0, 0, 0], values)

    # 实验三：GRPO 有正确、错误回答
    group = grpo_advantages([1, 0, 1, 0])

    # 实验四：GRPO 全部回答错误
    zero_group = grpo_advantages([0, 0, 0, 0])

    # 实验五：折扣因子小于 1
    discounted = gae([0, 0, 1], values, 0.9, 1.0)

    assert all(isclose(a, b) for a, b in zip(
        correct, [0.6, 0.4, 0.8]
    ))

    assert all(isclose(a, b) for a, b in zip(
        wrong, [-0.4, -0.6, -0.2]
    ))

    assert all(isclose(a, b) for a, b in zip(
        group, [1, -1, 1, -1]
    ))

    assert zero_group == [0.0, 0.0, 0.0, 0.0]

    assert all(isclose(a, b) for a, b in zip(
        discounted, [0.41, 0.30, 0.8]
    ))

    print("Correct:", correct)
    print("Wrong:", wrong)
    print("GRPO:", group)
    print("All wrong:", zero_group)
    print("Discounted GAE:", discounted)
```

运行后，核心结果为：

```text
Correct: [0.6, 0.4, 0.8]
Wrong: [-0.4, -0.6, -0.2]
GRPO: [1.0, -1.0, 1.0, -1.0]
All wrong: [0.0, 0.0, 0.0, 0.0]
Discounted GAE: [0.41, 0.3, 0.8]
```

我们重点理解三个现象。

**现象一：同一条回答中的 Advantage 可以不同。**

虽然整条回答的最终 Reward 相同，但 Critic Value 不同，因此 PPO 得到了不同的 Advantage。

**现象二：经典 Outcome GRPO 在同一条回答内部不直接区分 Token 的 Advantage。**

它的组相对信号来自整条回答的 Reward 与组内其他回答的对比。

**现象三：当组内 Reward 全相同时，组相对 Advantage 失去有效区分能力。**

这说明不同 Advantage Estimation 方法，并不是单纯换一个数学公式。

它们实际上决定了：

**我们如何从采样结果中构造可以用于训练的 Learning Signal。**

---

## 十二、如果我来做工程实验，应该怎样比较 PPO 和 GRPO？

读完 ORZ，我认为下一步不应该直接争论“哪种算法更好”，而应该提出可验证的实验问题。

例如：

> 在相同模型、数据、Reward 和尽量相同计算预算下，引入 Critic 是否真的能够提高 Reasoning RL 的稳定性？

一个相对合理的实验思路是：

### 实验 A：固定训练条件

尽量保持以下变量一致：

* 相同 Base Model。
* 相同的训练和验证数据。
* 相同的 Rule-based Verifier。
* 相同生成长度限制与采样参数。
* 相同或者可比较的 Rollout 计算预算。
* 相同的训练评价标准。

### 实验 B：比较不同 Advantage Estimation

实验组：

```text
Group A:
PPO + Learned Critic + GAE

Group B:
GRPO + Group-relative Advantage
```

如果进一步研究 Critic 的贡献，还可以引入不同的 Baseline、Value 网络规模或者其他 Advantage 估计方式。

### 实验 C：重点观察训练过程

除了最终 Accuracy，更应该记录：

```text
Reward Curve
Response Length Curve
KL Curve
Entropy Curve
Advantage Distribution
Critic Prediction
Repeat Score
Truncate Rate
```

尤其可以进一步分析：

* Repetition 开始前后，Value 如何变化？
* 不同最终 Reward 下，重复 Token 的 Advantage 分布是否不同？
* Group Reward 全同的比例是否随着训练增加？
* Critic 的训练误差与 Policy Stability 是否相关？
* 引入 Critic 后获得的收益，能否抵消额外的计算成本？

当然，想要严格判断因果关系，还需要控制超参数、随机种子，以及 Actor-Critic 和 GRPO 不同实现之间的其他变量。

这种实验设计比单纯对比一次 Benchmark Score，更有助于理解算法为什么有效。

---

## 十三、从求职面试角度，最值得掌握的三个问题

### Q1：PPO 已经有 Critic 了，为什么 GRPO 还要把 Critic 去掉？

因为 Critic 的训练存在实际成本。

尤其当模型规模很大时，额外训练一个 Value Model 会增加参数、显存和优化开销。

GRPO 利用同一 Prompt 下多个 Rollout 的相对 Reward 来构造 Advantage，从而避免学习独立的 Value Function。

但代价是，组相对 Advantage 主要依赖组内采样结果的差异。

所以它是一种工程 Trade-off，而不是对 PPO 的全面替代。

### Q2：为什么 ORZ 把 GAE 设置为 \(\gamma=1,\lambda=1\)？

因为 Reasoning RL 往往具有很长的推理序列，以及稀疏的 Terminal Reward。

设置 \(\gamma=1\)，可以避免远距离 Reward 被指数折扣削弱。

设置 \(\lambda=1\)，可以充分利用完整的采样回报，减少 Bootstrapping 引入的偏差。

在只有 Terminal Reward 的条件下，GAE 最终简化为：

$$
\boxed{\hat A_t=R-V_\phi(s_t)}
$$

从而使 Advantage 具有非常直观的“实际结果减去预期结果”的解释。

### Q3：为什么说 Critic 能改善 Credit Assignment，但又不能精确指出错误 Token？

因为 Critic 学习的是：

$$
V^\pi(s)=\mathbb E[R\mid s]
$$

它预测当前 State 下的未来回报，而不是直接判断每个推理步骤的因果正确性。

因此，Critic 可以为不同状态提供不同的 Baseline，让 Advantage 估计具有状态相关性。

但低 Value 不必然意味着对应 Token 会获得更负的 Advantage，也不代表这个 Token 是失败的根本原因。

ORZ 的实验说明，在作者的训练配置中，这种状态相关估计确实与更有利于抑制重复模式的 Advantage 分布有关。

它提供了 Critic 价值的实验证据，但不是完整的因果证明。

---

## 十四、总结：Open-Reasoner-Zero 真正教会了我什么？

回到最初的问题：

### PPO 真的过时了吗？

没有。

ORZ 证明了，在设计合理的 Reward、Advantage Estimation 和训练系统支持下，传统 PPO 仍然可以成为有效的 Reasoning RL 方法。

### Reasoning RL 一定要使用 GRPO 吗？

不一定。

GRPO 的重要优势在于不需要独立训练 Critic，能够简化训练系统。

但这是一种资源成本与学习信号构造方式之间的权衡，而不是所有场景下的最优解。

### Critic 究竟有没有价值？

有潜在的、不可忽视的价值。

Critic 可以学习 State-dependent Value，从而影响不同 Token 位置的 Advantage 估计。

ORZ 的实验进一步表明，在作者研究的长推理训练设置中，Critic 对重复模式的预测与更加稳定的 Advantage 信号有关。

但 Critic 的效果也依赖 Value 估计质量、训练数据和整体优化过程，并不能保证总是优于 Critic-free 方法。

### 最重要的收获：从 Algorithm Thinking 转向 Training Dynamics Thinking

以前读强化学习论文，很容易形成这样的思维方式：

```text
PPO
  ↓
GRPO
  ↓
New Algorithm
  ↓
Better Benchmark
```

仿佛算法的演进，就是不断使用更新的名字替代旧算法。

但 ORZ 让我意识到，更合理的思维方式应该是：

```text
                   Reasoning RL
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
    Exploration        Reward        Credit Assignment
        |                |                |
        v                v                v
    Rollouts          Verifier       Advantage Estimation
                                          |
                               +----------+----------+
                               |                     |
                               v                     v
                         PPO + Critic              GRPO
                               |                     |
                               +----------+----------+
                                          |
                                          v
                                  Policy Optimization
                                          |
                                          v
                                   Training Dynamics
                                          |
                        +-----------------+-----------------+
                        |                 |                 |
                        v                 v                 v
                      Reward           Stability         Generation
                                                           Quality
```

一个成功的 Reasoning RL 系统，需要同时解决：

**Exploration：** 如何让模型探索足够多样且有效的推理路径？

**Reward Design：** 如何保证奖励能够真实反映我们希望模型学到的能力？

**Credit Assignment：** 如何将稀疏的最终反馈转化为有价值的学习信号？

**Policy Optimization：** 如何在提高正确回答概率的同时，避免策略更新不稳定？

**Training Dynamics：** 如何持续观察模型训练中的行为变化，并及时识别 Collapse、Repetition 和其他异常？

最终我认为，Open-Reasoner-Zero 最值得记住的一句话是：

> **在 Reasoning RL 中，决定训练成败的往往不是某个算法名称，而是整个训练系统能否持续产生可靠的 Learning Signal，并保持健康的 Training Dynamics。**

PPO 与 GRPO 不应该被简单理解为新旧算法之间的竞争。

它们更应该被看成两种不同的 Credit Assignment 与训练成本权衡方案。

真正理解这些机制后，面对未来新的 RL 算法，我们才能提出比“它是否刷新了 Benchmark”更加深入的问题：

**它到底改变了什么？为什么这样的改变会影响训练行为？它的收益又来自哪里？**

这才是我希望通过 Day 13 建立的工程思维。

---

## 参考文献

[1] Hu, J., Zhang, Y., Han, Q., Jiang, D., Zhang, X., & Shum, H.-Y. (2025). **Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model.** *Advances in Neural Information Processing Systems (NeurIPS 2025).*

* [NeurIPS 2025 正式论文页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ed873d79e7c268c020c4b4db13a2812a-Abstract-Conference.html)
* [论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/ed873d79e7c268c020c4b4db13a2812a-Paper-Conference.pdf)
* 重点阅读：Section 2（RL Algorithm）、Section 3.2（Ablation Study）、Section 3.3（Critic Analysis）、Supplementary Material Section C.1 和 D。

[2] Open-Reasoner-Zero Authors. **Open-Reasoner-Zero Official Repository.** GitHub.

* https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
* 包含训练代码、配置、数据集和模型权重。
* 可重点阅读 `playground/` 目录中的训练配置，以及 `orz/ppo/` 中的 PPO 实现。

[3] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). **High-Dimensional Continuous Control Using Generalized Advantage Estimation.**

* https://arxiv.org/abs/1506.02438
* GAE 的原始论文，理解 Discount Factor、TD Error 和 Bias-Variance Trade-off 的基础资料。

[4] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms.**

* https://arxiv.org/abs/1707.06347
* PPO 原始论文，重点理解 Probability Ratio 和 Clipped Surrogate Objective。

[5] Shao, Z., et al. (2024). **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.**

* https://arxiv.org/abs/2402.03300
* GRPO 的重要原始来源，可重点阅读 Group Relative Policy Optimization 及 Outcome Supervision 相关部分。

[6] DeepSeek-AI, et al. (2025). **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.**

* https://arxiv.org/abs/2501.12948
* 理解 Reasoner-Zero、强化学习驱动的推理能力提升，以及 DeepSeek-R1 训练体系的重要参考文献。
