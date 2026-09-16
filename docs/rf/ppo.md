---
title: "PPO 学习笔记（一）：Policy Gradient 到底在优化什么？从 Log-Derivative Trick 到 REINFORCE + Baseline"
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

# PPO 学习笔记（一）：Policy Gradient 到底在优化什么？从 Log-Derivative Trick 到 REINFORCE + Baseline

最近开始系统学习 PPO。

但真正开始啃 PPO 之后，我发现一个问题：

> 如果连 Policy Gradient 为什么成立都没有搞明白，那么 PPO 里面的 probability ratio、advantage、clip，其实都只能靠背。

所以我决定先暂时把 PPO 放到一边，从它最底层的地基开始：

1. Policy Gradient 到底在优化什么？
2. Reward 明明不可导，为什么 Policy 还能通过梯度下降训练？
3. REINFORCE 是怎么从 Policy Gradient 推出来的？
4. 为什么减去一个 baseline 不会改变梯度的期望？
5. 为什么 baseline 又能够降低 variance？
6. 这些东西最后和 PPO 到底是什么关系？



# 1. PPO究竟想优化什么？

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

简述：$\pi_\theta(\tau)$，即$p_\theta(\tau)$，即一条轨迹$\tau$发生的概率，即在当前状态$a_t$下进入状态$s_t$的概率 乘以 在$a_t$和$s_t$下进入状态$s_{t+1}$的概率。

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

概述一下：对目标函数$J(\theta)$求导数的时候，使用对数导数的技巧，在数学表达式上可以重新引入轨迹$\tau$的概率表达式，进而可以将**积分形式的数学表达式等效为数学期望形式的表达式**，进而可以使用蒙特卡洛模拟来进行近似求值。

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


# 12. 到这里应该真正记住什么？

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

# 13. REINFORCE 到底是什么？

前面得到：

$$
\nabla_\theta J
=
\mathbb E
\left[
R(\tau)
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
$$

但是 expectation 无法精确计算。

怎么办？

答案非常暴力：

> Sample。

我们按照当前 Policy：

$$
\pi_\theta
$$

跑出若干条 trajectory：

$$
\tau^{(1)},\tau^{(2)},\cdots,\tau^{(N)}
$$

然后使用样本平均估计：

$$
\nabla_\theta J
\approx
\frac{1}{N}
\sum_{i=1}^{N}
R(\tau^{(i)})
\sum_t
\nabla_\theta
\log\pi_\theta
(a_t^{(i)}|s_t^{(i)})
$$

这已经基本就是 REINFORCE 了。

因此从现代视角来看：

> **REINFORCE 本质上就是 Monte Carlo Policy Gradient。**

---

# 14. 为什么实际 REINFORCE 使用 Return-to-Go更新策略？

简述：当前的动作只带来未来的收益，不会影响已有的动作带来的收益。

针对一条最原始的 trajectory 写法是：

$$
R(\tau)
\sum_t
\nabla\log\pi(a_t|s_t)
$$

也就是说，每个 action 都乘整条 trajectory 的 reward。这里强调针对一条具体的轨迹，计算其具体的梯度影响，所以外层的期望表达式临时去除了。

但这里有一个问题。

假设：

```text
t = 0     选择 action a0, 得到 r1
t = 1     选择 action a1, 得到 r2
t = 2     选择 action a2，得到 r3
t = 3     选择 action a3, 得到 r4
t = 4     选择 action a4, 得到 r5
```

$a_2$ 怎么可能影响已经发生的 $r_1,r_2$ ？

所以对于 $a_t$ 来说，真正需要考虑的是从当前时刻开始的 future return：

$$
G_t
=
\sum_{k=t}^{T-1}
\gamma^{k-t}r_{k+1}
$$

代入$G_t$于是可以写成：

$$
\boxed{
\nabla_\theta J
\approx
\sum_t
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

这里假设每一个动作$a_t$的奖励都能够被环境量化出来。详细解释在[Return-to-go](./ppo/Return-to-go)。

---

# 15. REINFORCE 到底在干什么？

简述：对于结果好的action $a_t$，让其发生的概率更高，对于不好的$a_t$让其发生的概率降低。

看这个式子（$\propto$表示正相关的意思）：

$$
\Delta\theta
\propto
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

假设：

$$
G_t > 0
$$

那么做 gradient ascent：

$$
\theta
\leftarrow
\theta
+
\alpha
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

相当于：

> 提高刚才这个 action 的 log probability。

如果：

$$
G_t < 0
$$

则会反过来降低这个 action 的概率。

所以可以把 REINFORCE 理解成一句极其朴素的话：

> **结果好，就让刚才做过的事情以后更容易发生；结果差，就让它以后更不容易发生。**

数学只是把这句话变成了一个无偏的 gradient estimator。

---

# 16. 但是 REINFORCE 有一个致命问题：Variance 很大

简述：action太多，导致耽搁action的影响被稀释。

假设 Agent 做了 100 个 action。

最后得到了：

$$
R=103.7
$$

然后你告诉模型：

```text
刚才所有 action 都乘 103.7 更新。
```

问题就来了。

到底：

- 哪个 action 真正做得好？
- 哪个 action 做得差？
- reward 高是因为当前 action，还是其他 action？
- reward 的绝对大小是不是本来就很大？

Monte Carlo return 是一个噪声非常大的信号。

因此虽然 REINFORCE 的期望方向是对的，但实际每一次采样得到的梯度可能紊乱，即$\text{high variance}$ 。

---

# 17. 解决方案： Baseline 

简述：不改变梯度的期望。

于是我们把：

$$
G_t
$$

换成：

$$
G_t-b(s_t)
$$

Policy Gradient 变成：

$$
\boxed{
\nabla_\theta J
\approx
\sum_t
\left(
G_t-b(s_t)
\right)
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

这里的：

$$
b(s_t)
$$

就叫 baseline。

$V^\pi(s_t)$：在状态 $s_t$，按照策略 $\pi$ 能拿到的**期望未来总收益**，用作 baseline，减去它之后得到优势，用来更新策略梯度，减少方差。

最常见的选择就是：

$$
b(s_t)=V^\pi(s_t)
$$

于是：

$$
G_t-V^\pi(s_t)
$$

就在估计 Advantage。

---

# 18. 为什么减 baseline 不会改变期望？



我们需要证明：

$$
\mathbb E
\left[
b(s)
\nabla_\theta
\log\pi_\theta(a|s)
\right]
=
0
$$

固定一个 state $s$。

因为 baseline $b(s)$ 不依赖采样出的 action，所以：

$$
\mathbb E_{a\sim\pi}
\left[
b(s)
\nabla_\theta
\log\pi_\theta(a|s)
\right]
$$

可以把 $b(s)$ 提出来：

$$
=
b(s)
\sum_a
\pi_\theta(a|s)
\nabla_\theta
\log\pi_\theta(a|s)
$$

利用：

$$
\pi_\theta(a|s)
\nabla_\theta\log\pi_\theta(a|s)
=
\nabla_\theta\pi_\theta(a|s)
$$

于是：

$$
=
b(s)
\sum_a
\nabla_\theta
\pi_\theta(a|s)
$$

把梯度移到外面：

$$
=
b(s)
\nabla_\theta
\sum_a
\pi_\theta(a|s)
$$

而概率之和永远等于 1：

$$
\sum_a
\pi_\theta(a|s)
=
1
$$

所以：

$$
=
b(s)\nabla_\theta 1
$$

最终：

$$
\boxed{
\mathbb E
\left[
b(s)
\nabla_\theta
\log\pi_\theta(a|s)
\right]
=
0
}
$$

因此：

$$
\mathbb E
\left[
(G_t-b(s_t))
\nabla\log\pi
\right]
$$

和：

$$
\mathbb E
\left[
G_t\nabla\log\pi
\right]
$$

具有完全相同的期望。

也就是说：

> **Baseline 改变了每一次 gradient sample，却没有改变平均意义上的真实梯度方向。**

---

# 19. 一个特别容易产生的误解

千万不要把原因理解成：

$$
\mathbb E[G-b]
=
\mathbb E[G]
$$

这显然不一定成立。

例如：

$$
G=10,\quad b=8
$$

那么：

$$
G-b=2
$$

当然已经发生变化。

真正为 0 的东西是：

$$
\mathbb E
\left[
b(s)
\nabla\log\pi(a|s)
\right]
$$

而不是：

$$
\mathbb E[b(s)]
$$

这两件事情完全不同。

Baseline 能够不改变梯度期望，依赖的是 score function 的一个关键性质：

$$
\boxed{
\mathbb E_{a\sim\pi}
[
\nabla_\theta\log\pi_\theta(a|s)
]
=
0
}
$$

这个公式非常值得记住。

后面很多 Policy Gradient 推导都会再次遇到它。

---

# 20. 为什么 baseline 可以降低 variance？

现在再看：

$$
(G_t-b(s_t))
\nabla\log\pi(a_t|s_t)
$$

假设某个 state 本来就是一个特别容易拿高分的 state。

例如：

```text
不管采取什么 action

reward 大概都在 100 左右
```

现在有两个动作：

```text
action A → 101
action B → 99
```

如果直接使用 reward：

```text
A → +101
B → +99
```

看起来两个 action 都应该被疯狂加强。

但真正重要的信息其实是：

```text
在这个 state 下：

A 比平均水平好一点
B 比平均水平差一点
```

假设：

$$
V(s)=100
$$

减掉 baseline 后：

```text
A → 101 - 100 = +1
B → 99 - 100 = -1
```

学习信号立刻清楚很多。


它回答的不是：

> “这个 action 最终能拿多少 reward？”

而是：

> **“在当前这个 state 下，这个 action 相比我的正常水平到底好多少？”**

---


# 22. 从 REINFORCE 到 Advantage


现在整个公式已经逐渐变成熟悉的样子：

最开始：

$$
R(\tau)
\nabla\log\pi
$$

然后变成 return-to-go：

$$
G_t
\nabla\log\pi
$$

加入 baseline：

$$
(G_t-b(s_t))
\nabla\log\pi
$$

如果：

$$
b(s_t)=V^\pi(s_t)
$$

那么：

$$
G_t-V^\pi(s_t)
$$

就是 advantage 的 Monte Carlo estimate。

于是：

$$
\boxed{
\nabla_\theta J
\approx
\sum_t
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

看到这个公式，PPO 已经离我们非常近了。

---

# 23. 用一个 Softmax Multi-Armed Bandit 了解一下 Policy Gradient 工作

假设四个 action 的真实 expected reward 分别是：

$$
[0.2,\;0.5,\;1.0,\;0.7]
$$

因此第三个 action 是最优动作。

但是 Agent 并不知道这些数字。

Policy 使用四个 logit：

$$
\theta_1,\theta_2,\theta_3,\theta_4
$$

通过 softmax 得到：

$$
\pi_\theta(a=i)
=
\frac{
e^{\theta_i}
}{
\sum_j e^{\theta_j}
}
$$

初始时：

$$
\theta=[0,0,0,0]
$$

因此：

$$
\pi=[0.25,0.25,0.25,0.25]
$$

然后不断：

```text
根据 Policy 采样 action
        ↓
Environment 返回 reward
        ↓
计算 REINFORCE gradient
        ↓
更新 θ
        ↓
重新计算 action probability
```

代码见[代码](./ppo/multi-armed%20bandit)。

<!-- 代码如下。

```python
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(0)

# 四个老虎机臂真实的 expected reward
# Agent 并不知道这些值，只能通过不断尝试获得 noisy reward
true_means = np.array([0.2, 0.5, 1.0, 0.7])

num_actions = len(true_means)

# Policy 参数：softmax logits
theta = np.zeros(num_actions)

learning_rate = 0.05

# running baseline
baseline = 0.0
baseline_lr = 0.05

steps = 3000

prob_history = []


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


for step in range(steps):

    # 1. 当前 Policy
    probs = softmax(theta)

    # 2. 根据 Policy 采样 action
    action = rng.choice(num_actions, p=probs)

    # 3. Environment 返回 noisy reward
    reward = rng.normal(
        loc=true_means[action],
        scale=1.0
    )

    # 4. 计算 grad log pi(a)
    #
    # 对 softmax：
    #
    # d log pi(a) / d theta_k
    # =
    # 1[k == a] - pi(k)
    #
    grad_log_pi = -probs.copy()
    grad_log_pi[action] += 1.0

    # 5. REINFORCE + baseline
    #
    # 注意：先使用“旧 baseline”计算 advantage，
    # 再使用当前 reward 更新 baseline。
    advantage = reward - baseline

    # Gradient Ascent
    theta += learning_rate * advantage * grad_log_pi

    # 6. 更新 running baseline
    baseline += baseline_lr * (reward - baseline)

    prob_history.append(softmax(theta))


prob_history = np.array(prob_history)

for action in range(num_actions):
    plt.plot(
        prob_history[:, action],
        label=f"Action {action}: mean={true_means[action]}"
    )

plt.xlabel("Training Step")
plt.ylabel("Action Probability")
plt.title("REINFORCE on a Softmax Multi-Armed Bandit")
plt.legend()
plt.show()
```

---

# 24. 这段代码最重要的不是结果，而是这一行

核心更新只有：

```python
theta += learning_rate * advantage * grad_log_pi
```

对应：

$$
\theta
\leftarrow
\theta
+
\alpha
(R-b)
\nabla_\theta
\log\pi_\theta(a)
$$

现在来看 softmax 的梯度：

$$
\frac{\partial\log\pi(a=i)}
{\partial\theta_k}
=
\mathbb 1[k=i]-\pi(k)
$$

如果：

$$
R-b>0
$$

说明：

> 这一次 action 的表现比平均水平好。

那么 chosen action 对应的 logit 会增加。

它的 probability 随之增加。

反过来，如果：

$$
R-b<0
$$

说明：

> 这个 action 比正常水平差。

那么它的 probability 会下降。

因此经过大量采样以后，第三个 action：

$$
\mu_3=1.0
$$

获得高 reward 的次数更多。

它得到正向 Policy Gradient 更新的概率也更高。

最终：

$$
\pi(a=2)
$$

会逐渐接近 1。

这就是第一次真正“看到”：

> **Policy Gradient 在工作。**

--- -->

# 25. 为什么一次 reward 高，不代表 action 一定会被永久加强？

简述：因为reward有噪音，导致单次采样梯度有噪音。

这里还有一个非常重要的统计视角。

假设：

```text
Action A true mean reward = 1
Action B true mean reward = 0
```

由于 reward 有噪声，某一次可能出现：

```text
A → -0.3
B → +2.1
```

那么这一次 REINFORCE 甚至会：

```text
降低 A 的概率
提高 B 的概率
```

是不是说明算法错了？

不是。

Policy Gradient 从来没有承诺：

> 每一次 gradient estimate 都是正确的。

它保证的是：

> **在 expectation 上，gradient estimator 指向提高 expected reward 的方向。**

因此，对于单次trajtory的梯度$\hat g$的期望而言，和目标函数期望一样：

$$
\mathbb E[\hat g]
=
\nabla_\theta J
$$

但不意味着，单词trajtory的梯度和目标函数的梯度一样：

$$
\hat g
\neq
\nabla_\theta J
$$

每一次采样都会有 noise。

这就是我们一直讨论 variance 的原因。

---

# 26. REINFORCE 最核心的两个性质

到这里，可以把 REINFORCE 理解成两个关键词：

## 第一：Unbiased

在相应假设下：

$$
\mathbb E[\hat g]
=
\nabla_\theta J
$$

平均意义上方向正确。

## 第二：High Variance

但是：

$$
\mathrm{Var}(\hat g)
$$

可能非常大。

所以实际强化学习的发展史，很大一部分其实都在解决：

> **怎样在尽量保持正确优化方向的同时，把 Policy Gradient 的 variance 降下来？**

Baseline 是其中最基础的一步。

Actor-Critic 又进一步用 learned value function 来提供更好的学习信号。

后面的：

- Advantage；
- TD；
- GAE；
- TRPO；
- PPO；

都可以沿着这条主线继续理解。

---

# 27. 后序PPO的学习思路

这一部分不是 Sutton 1999 的内容，而是为了帮助建立后续 PPO 的知识连接。

我们现在已经得到 Policy Gradient 的核心形式：

$$
\boxed{
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

PPO 并没有推翻这个东西。

PPO 仍然是 Policy Gradient。

它真正要解决的是另外一个问题：

> **如果我用同一批数据，把 Policy 一口气更新得太远怎么办？**

PPO 定义：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

然后构造 surrogate objective（代理目标）。

最简单的版本可以先理解成：

$$
L(\theta)
=
\mathbb E_t
[
r_t(\theta)\hat A_t
]
$$

对 $r_t$ 求梯度：

$$
\nabla_\theta r_t(\theta)
=
r_t(\theta)
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

因此：

$$
\nabla_\theta L
=
\mathbb E_t
\left[
r_t(\theta)
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
$$

刚开始更新时：

$$
\theta=\theta_{\text{old}}
$$

所以：

$$
r_t(\theta)=1
$$

于是又变回：

$$
\boxed{
\mathbb E_t
[
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
]
}
$$

也就是我们今天学的 Policy Gradient。

PPO 后面的 clipping：

$$
\operatorname{clip}
(
r_t(\theta),
1-\epsilon,
1+\epsilon
)
$$

可以先粗略理解成：

> Policy Gradient 告诉我们**往哪里走**，PPO clipping 开始限制我们**一次不要走太远**。

所以学习顺序应该是：

```text
Expected Return
      ↓
Log-Derivative Trick
      ↓
Policy Gradient
      ↓
REINFORCE
      ↓
Baseline
      ↓
Advantage
      ↓
Actor-Critic
      ↓
GAE
      ↓
Importance Sampling / Probability Ratio
      ↓
PPO
```

如果前面的东西没懂，直接看最后的 clip objective，很容易感觉 PPO 是突然从天上掉下来的一个公式。

但顺着这条路线走，会发现 PPO 的每一个组件其实都有明确的来源。

---

# 28. 本次学习最重要的几个知识点巩固

## Reward 本身不可导，为什么 Policy 还能训练？

简述版：因为对于梯度策略的目标函数求导，不需要对Reward求导，Reward在这里仅仅只是体现为一个权重系数的作用，奖励高的轨迹出现的概率增大，让奖励低的轨迹出现的概率减小，以此为目标去更新参数$\theta$。

答：Policy Gradient 不需要对 Reward 本身求导。

我们的目标是：

$$
J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
[R(\tau)]
$$

Policy 参数 $\theta$ 改变的是 trajectory 的概率分布：

$$
p_\theta(\tau)
$$

而不是直接改变某一个已经发生的 reward。

利用 log-derivative trick：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
$$

可以得到：

$$
\nabla_\theta J
=
\mathbb E
[
R(\tau)
\nabla_\theta
\log p_\theta(\tau)
]
$$

因此 reward 只需要作为一个 scalar weight：

```text
高 reward trajectory
→ 提高其 log probability

低 reward trajectory
→ 降低其 log probability
```

整个过程完全不要求：

$$
\nabla R
$$

存在。

---

## 为什么梯度式子里面减去baseline不改变梯度更新的期望

简述版：因为baseline不依赖采样的action，可以单独提出来，和策略action的概率的梯度相乘之后的和，通过等式变换恒等为0。这里并不是指目标函数的期望不变，而是指的目标函数梯度的期望不变。

为什么：

$$
(R-b)
\nabla\log\pi
$$

不会改变 Policy Gradient 的 expectation？

因为：

$$
\mathbb E_{a\sim\pi}
[
b(s)\nabla\log\pi(a|s)
]
$$

等于：

$$
b(s)
\nabla
\sum_a\pi(a|s)
$$

而：

$$
\sum_a\pi(a|s)=1
$$

可以理解为，在状态s下采取所有策略的概率的和为1。所以：

$$
b(s)\nabla1=0
$$

---

## 那为什么 baseline 又能降低 variance？

简述版：一般baseline取的是当前状态state所能带来的期望收益，对于一个action带来的收益是建立在状态state的基础上带来的，减去state的期望收益，可以衡量出一个具体的action带来的相对收益。

因为它把：

```text
absolute return
```

转变成更有意义的：

```text
relative return
```

即：

> 当前 action 相比这个 state 下的正常表现到底好多少。



于是 Policy 不再单纯学习：

> “这个 action 得了 100 分。”

而是在学习：

> “在本来就能得 98 分的情况下，这个 action 多贡献了 2 分。”

后者显然是更干净的 credit assignment signal（信用分配信号）。

---

# 30. 最后，用一句话分别理解这些概念

**Policy Gradient**

> 改变 Policy 参数，让高回报行为出现的概率增加。

**Log-Derivative Trick**

> 把“概率的梯度”变成“概率 × log probability 的梯度”，从而把梯度写成可以采样估计的 expectation。

**REINFORCE**

> 用 Monte Carlo return 对 Policy Gradient 进行采样估计。

**Baseline**

> 不改变期望梯度，只重新定义“什么叫表现好”，从而降低 gradient estimator 的 variance。

**Advantage**

> 当前 action 相比这个 state 下的正常表现到底好多少。

**PPO**

> 仍然沿着 Policy Gradient 的方向优化，但对每次 Policy 改变的幅度加以约束。

---

# 31. 真正需要建立的直觉

学完以后，我现在会把强化学习里的 Policy Gradient 想象成这样：

```text
Policy 生成行为
      ↓
Environment 只需要告诉我：
“这次结果有多好”
      ↓
我不需要知道 Environment 怎么求导
      ↓
只需要知道：
“这次行为是由我的 Policy 以多大概率生成的”
      ↓
Reward × ∇ log π
      ↓
调整 Policy 的概率分布
```

不需要知道：

> Reward 是怎样计算出来的。

只需要能够：

1. Sample；
2. Evaluate；
3. Compute log probability；
4. Backpropagate through Policy。

这也是为什么到了大语言模型时代，Policy Gradient 依然如此重要。

模型可以生成离散 token，

Environment 可以包含用户、规则、工具甚至整个真实世界，

Reward 可以来自一个完全不可微的过程，

但只要最终能得到一个 scalar feedback：

$$
R
$$

Policy Gradient 就提供了一条优化 Policy 的道路（标量反馈）。

---

# 参考文献

1. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. Advances in Neural Information Processing Systems 12.

2. Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning, 8, 229–256. DOI: 10.1007/BF00992696.

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd Edition. MIT Press.
