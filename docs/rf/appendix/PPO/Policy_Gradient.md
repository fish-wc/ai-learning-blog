---
title: "PPO 学习笔记：Policy Gradient 到底在优化什么？"
date: 2026-09-10
categories:
  - Reinforcement Learning
tags:
  - Policy Gradient
  - REINFORCE
  - Baseline
  - Advantage
  - PPO
math: true
---

# PPO 学习笔记：Policy Gradient 到底在优化什么？从 Log-Derivative Trick 到 REINFORCE + Baseline

但真正开始啃 PPO 之后，我发现一个问题：

> 如果连 Policy Gradient 为什么成立都没有搞明白，那么 PPO 里面的 probability ratio、advantage、clip，其实都只能靠背。

所以我决定先暂时把 PPO 放到一边，从它最底层的地基开始：

1. Policy Gradient 到底在优化什么？
2. Reward 明明不可导，为什么 Policy 还能通过梯度下降训练？




# 1. PPO究竟想优化什么？

简述：Policy Gradient 优化的不是某一次 reward，而是**当前 Policy 所产生的 trajectory 的期望回报**。


强化学习和监督学习有一个非常重要的区别。监督学习通常有一个明确的 loss：

$$
L(\theta)
$$

然后直接计算：

$$
\nabla_\theta L(\theta)
$$

例如神经网络做分类时：

```text
input
  ↓
neural network
  ↓
prediction
  ↓
cross entropy loss
```

整个计算图通常都是可微的，于是我们可以一路反向传播。但是强化学习不是这样，Agent 做的是：

```text
state
  ↓
policy
  ↓
action
  ↓
environment
  ↓
new state / reward
```

Environment 很可能根本不可微。

例如：

- 游戏环境；
- 机器人和真实世界交互；
- 用户给出的评分；
- 人类偏好；
- 一个只返回 0/1 的规则系统；
- LLM 生成完整回答之后得到的标量 reward。

所以我们不能简单地写：

$$
\frac{\partial R}{\partial \theta}
$$

然后从 reward 一路反向传播到 policy。Policy Gradient 的关键思想就是：

>**我根本不需要对 Reward 求导。** 

我们真正要优化的是：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
[R(\tau)]
$$

其中：

- $\theta$：Policy 的参数；
- $\pi_\theta$：当前 Policy；
- $\tau$：一整条 trajectory；
- $R(\tau)$：这条 trajectory 最终获得的 return。

也就是说：

> Policy Gradient 优化的不是某一次 reward，而是**当前 Policy 所产生的 trajectory 的期望回报**。


---

# 2. 什么是 trajectory？
简述：trajectory是状态和动作的记录。

一条 trajectory 可以写成：

$$
\tau =
(s_0,a_0,r_1,s_1,a_1,r_2,\cdots,s_T)
$$

例如对于一个游戏 Agent：

```text
看到当前位置 s0
    ↓
选择动作 a0
    ↓
环境进入 s1
    ↓
选择动作 a1
    ↓
环境进入 s2
    ↓
...
    ↓
游戏结束
```

对于 LLM，其实也完全可以这样理解。

假设模型正在生成：

```text
强化学习是一种...
```

那么：

- state $s_t$：当前已经生成的 token；
- action $a_t$：下一枚 token；
- policy $\pi_\theta(a_t|s_t)$：模型对下一枚 token 的概率分布；
- trajectory $\tau$：完整生成的一段文本；
- reward $R(\tau)$：最终给这段回答的评分。

所以 LLM 本质上也可以看成一个 sequence decision making problem。

---

# 3. 一个容易混淆的记号：$\pi_\theta(\tau)$

简述：$\pi_\theta(\tau)$，即$p_\theta(\tau)$，即一条轨迹$\tau$发生的概率，即在当前状态$a_t$下进入状态$s_t$的概率 乘以 在$a_t$和$s_t$下进入状态$s_{t+1}$的概率，从初始状态$s_0$开始累乘。
备注：这里认为是马尔科夫过程，即当前状态仅与上一个状态相关。

很多教程会写：

$$
\tau \sim \pi_\theta
$$

或者甚至直接写：

$$
\pi_\theta(\tau)
$$

严格来说，更准确的记号应该是：

$$
p_\theta(\tau)
$$

因为一条 trajectory 出现的概率并不完全由 Policy 决定。它还包含 environment transition。

对于一个 MDP（马尔科夫决策过程，这里需要具备马尔科夫性的相关知识才能看懂）：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

这里：

$$
\rho_0(s_0)
$$

是初始状态分布，

$$
\pi_\theta(a_t|s_t)
$$

是 Agent 的策略（比如，模型对下一枚 token 的概率分布），而：

$$
P(s_{t+1}|s_t,a_t)
$$

是环境的状态转移概率。依据马尔科夫性，当前状态的转移只与上一个状态有关，$s_{t+1}$这个状态只由上一个状态$s_t$和上一个状态采取的策略$a_t$所决定。

这三个东西共同决定：

> 一条 trajectory 到底有多大概率发生。


---

# 4. Policy Gradient 的目标函数

我们希望最大化期望回报：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
[R(\tau)]
$$

即，如果奖励函数$R(\tau)$比较大时，轨迹$\tau =
(s_0,a_0,r_1,s_1,a_1,r_2,\cdots,s_T)$出现的概率对应增大，反之减小。

$\mathbb{E}_{\tau \sim p_\theta(\tau)}$表示要使用概率$p_\theta(\tau)$来乘，把 expectation 展开：

$$
J(\theta)
=
\sum_\tau
p_\theta(\tau)R(\tau)
$$

即，对于数学期望这个概念而言，是指的函数数值和其出现的概率相乘之后，穷尽所有的乘值进行累加得到。这里函数是$R(\tau)$，对应的概率是发生轨迹$\tau$的概率，即$p_\theta(\tau)$。

连续情况则写成积分：

$$
J(\theta)
=
\int
p_\theta(\tau)R(\tau)d\tau
$$

接下来问题来了：

$$
\nabla_\theta J(\theta)
$$

是怎么算的。

---

# 5. 第一个关键步骤：对期望求导

对于函数一阶导而言是指的函数的趋势，对于趋势为零的点，则说明是极值点，所以从：

$$
J(\theta)
=
\int
p_\theta(\tau)R(\tau)d\tau
$$

出发。

对 $\theta$ 求梯度：

$$
\nabla_\theta J(\theta)
=
\nabla_\theta
\int
p_\theta(\tau)R(\tau)d\tau
$$

先积分再求导可以交换成先求导再积分，可以把梯度移动到积分里面：

$$
=
\int
\nabla_\theta
\left[
p_\theta(\tau)R(\tau)
\right]
d\tau
$$

对于一条已经确定的 trajectory $\tau$，它得到的 $R(\tau)$ 并不显式依赖 Policy 参数 $\theta$。

所以：

$$
=
\int
R(\tau)
\nabla_\theta p_\theta(\tau)
d\tau
$$

注意这一行。

我们根本没有出现：

$$
\nabla_\theta R(\tau)
$$

我们求导的是：

$$
\nabla_\theta p_\theta(\tau)
$$

也就是：

> **Policy 参数发生变化之后，这条 trajectory 出现的概率会怎么变化？**

这就是 Policy Gradient 最核心的视角转换。

---

# 6. Log-Derivative Trick

接下来就是整个 Policy Gradient 最重要的数学技巧：

## Log-Derivative Trick

对于任意概率分布 $p_\theta(x)$：

$$
\nabla_\theta \log p_\theta(x)
=
\frac{
\nabla_\theta p_\theta(x)
}{
p_\theta(x)
}
$$

于是：

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)
\nabla_\theta \log p_\theta(x)
$$

把它代回刚才的式子：

$$
\nabla_\theta J(\theta)
=
\int
R(\tau)
\nabla_\theta p_\theta(\tau)
d\tau
$$

得到：

$$
\nabla_\theta J(\theta)
=
\int
R(\tau)
p_\theta(\tau)
\nabla_\theta
\log p_\theta(\tau)
d\tau
$$

注意，这里依据数学期望的定义，函数值乘以该函数值出现的概率，然后穷举求和就是期望，所以这里可以写回期望的数学形式：

$$
\int p_\theta(\tau)f(\tau)d\tau
$$

实际上就是：

$$
\mathbb E_{\tau\sim p_\theta}[f(\tau)]
$$

因此，目标函数对于参数$\theta$求导可以等效为如下式子：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)
\nabla_\theta
\log p_\theta(\tau)
\right]
}
$$

这就是我们一开始想得到的结果。

---

# 7. 但是 $\log p_\theta(\tau)$ 又是什么？

前面已经得到：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_t
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

取 log：

$$
\log p_\theta(\tau)
=
\log\rho_0(s_0)
+
\sum_t
\log\pi_\theta(a_t|s_t)
+
\sum_t
\log P(s_{t+1}|s_t,a_t)
$$

接下来对 $\theta$ 求导。

在标准 Policy Gradient 假设下：

- 初始状态分布不由 Policy 参数决定；
- Environment transition 不由 Policy 参数决定。

所以：

$$
\nabla_\theta \log\rho_0(s_0)=0
$$

以及：

$$
\nabla_\theta
\log P(s_{t+1}|s_t,a_t)
=
0
$$

最后只剩：

$$
\nabla_\theta
\log p_\theta(\tau)
=
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

因此：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
R(\tau)
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

这就是最经典的 trajectory-level Policy Gradient。

---

# 8. Reward 不可导，为什么 Policy 还能训练？

概述：**Policy Gradient 不是在对 Reward 求梯度，而是在对“产生不同 Reward 的概率分布”求梯度。**

现在正面回答这个问题。

假设只有两个动作：

```text
action 1 → reward = 10
action 2 → reward = 0
```

Reward 本身就是两个常数，没必要求导。

假设智能体执行了两个动作$a_1$和$a_2$，每个动作的发生都有对应的概率如下：

$$
P(a_1)=\pi_\theta(a_1)
$$

$$
P(a_2)=\pi_\theta(a_2)
$$

那么期望 reward 是：

$$
J(\theta)
=
10\pi_\theta(a_1)
+
0\pi_\theta(a_2)
$$

求导：

$$
\nabla_\theta J
=
10\nabla_\theta\pi_\theta(a_1)
$$

所以根本不需要：

$$
\nabla_\theta 10
$$

只需要知道：

> 怎样修改 $\theta$，才能提高“拿到 10 分的 action 被选中”的概率。

（重点）所以 Policy Gradient 的完整逻辑其实是：

```text
参数 θ 发生变化
    ↓
Policy 的 action probability 发生变化
    ↓
不同 trajectory 出现的概率发生变化
    ↓
高 reward trajectory 出现得更多
    ↓
Expected Reward 上升
```

因此：

> **Policy Gradient 不是在对 Reward 求梯度，而是在对“产生不同 Reward 的概率分布”求梯度。**

Reward 在这里扮演的是一个**权重**：

```text
这个 trajectory reward 很高
        ↓
增加它出现的概率

这个 trajectory reward 很低
        ↓
减少它出现的概率
```

这就是 Policy Gradient。

---

# 9. 为什么偏偏要取 log？

概述：对目标函数$J(\theta)$求导数的时候，使用对数导数的技巧，在数学表达式上可以重新引入轨迹$\tau$的概率表达式，进而可以将**积分形式的数学表达式等效为数学期望形式的表达式**，进而可以使用蒙特卡洛模拟来进行近似求值。

可能还会有一个疑问，为什么不能直接用：

$$
\nabla_\theta\pi_\theta(a|s)
$$

为什么大家都写：

$$
\nabla_\theta\log\pi_\theta(a|s)
$$

最直接的原因是 log-derivative trick：

$$
\nabla_\theta p
=
p\nabla_\theta\log p
$$

它成功把：

$$
\nabla p
$$

转换成了：

$$
p\times\nabla\log p
$$

这样积分里面就重新出现了概率分布 $p$：

$$
\int
p(\tau)
R(\tau)
\nabla\log p(\tau)d\tau
$$

于是它可以被写成 expectation：

$$
\mathbb E[
R\nabla\log p
]
$$

而 expectation 最大的好处就是：

> **可以通过 Monte Carlo sampling 来近似。**

我们不需要知道所有 trajectory，让当前 Policy 真正跑 100 条 trajectory：

$$
\tau_1,\tau_2,\cdots,\tau_{100}
$$

然后直接计算：

$$
\hat g
=
\frac{1}{100}
\sum_{i=1}^{100}
R(\tau_i)
\nabla_\theta\log p_\theta(\tau_i)
$$

就能够近似真实梯度。这就是 Policy Gradient 能够真正实现成算法的关键。


---


# 10. 总结

记忆：写好目标函数$ J(\theta)=\mathbb E_{\tau\sim p_\theta}
[R(\tau)] $，打开期望，求导运算，对数求导技巧做符号替换，大概概率表达式化简。

真正需要记住的是这条链，先明确优化目标是**使得奖励高的轨迹发生的概率更大**：

$$
J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
[R(\tau)]
$$

↓  要求优化目标的极值，就要求一阶导数。

$$
\nabla_\theta J
=
\int
R(\tau)
\nabla_\theta p_\theta(\tau)d\tau
$$

↓ 使用概率求导的对数导数技巧，保留数学表达式中的概率表达式$p$，以此来便于将积分的函数表达式等效为求期望的函数表达式，如此可以使用蒙特卡洛方法求近似值。

$$
\nabla p
=
p\nabla\log p
$$

↓ 将积分的形式的函数值转换为数学期望形式的表达式。

$$
\nabla_\theta J
=
\mathbb E
\left[
R(\tau)
\nabla_\theta\log p_\theta(\tau)
\right]
$$

↓ 把轨迹$\tau$的概率$ p_\theta(\tau)$展开，表达式里面省去除无用部分。因为初始状态$\rho_0(s_0)$ 和 Environment 与 $\theta$ 无关，所以等效于常数求导：

$$
\nabla_\theta\log p_\theta(\tau)
=
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

↓

最终代入轨迹$\tau$的概率表达式$ p_\theta(\tau)$：

$$
\boxed{
\nabla_\theta J
=
\mathbb E
\left[
R(\tau)
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
}
$$

简述：

> **让高 Reward trajectory 更容易发生，让低 Reward trajectory 更不容易发生。**

---


# 参考文献

1. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. Advances in Neural Information Processing Systems 12.

<!-- 2. Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning, 8, 229–256. DOI: 10.1007/BF00992696.

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd Edition. MIT Press. -->
