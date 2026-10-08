---
title: "PPO 学习笔记： REINFORCE"
date: 2026-09-24
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

1. REINFORCE 是怎么从 Policy Gradient 推出来的？
2. 为什么减去一个 baseline 不会改变梯度的期望？
3. 为什么 baseline 又能够降低 variance？
4. 这些东西最后和 PPO 到底是什么关系？

# 1. REINFORCE 到底是什么？

前面[Policy Gradient](/docs/rf/appendix/PPO/Policy_Gradient.md)得到：

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

# 2. 为什么实际 REINFORCE 使用 Return-to-Go更新策略？

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

# 3. REINFORCE 到底在干什么？

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

# 4. 但是 REINFORCE 有一个致命问题：Variance 很大

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

# 5. 解决方案： Baseline 

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

# 6. 为什么减 baseline 不会改变期望？

概述：期望只是整体减掉常数，梯度的期望不变，所以不影响期望对应的策略更新方向。

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

# 7. 一个特别容易产生的误解

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

# 8. 为什么 baseline 可以降低 variance？

概述：回报里包含状态本身的固有价值和动作带来的增量。baseline（通常是 V (s)）就是状态的平均回报。减去 baseline，剔除掉状态自带的那部分公共分量，只保留动作好坏的相对信号，**相当于把信号做中心化**，样本波动变小，方差降低；同时因为 baseline 不依赖动作，梯度期望不变，不会引入偏差。

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
