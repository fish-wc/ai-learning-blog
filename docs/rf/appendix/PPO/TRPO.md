---
title: "TRPO：为什么强化学习中的策略不能一次更新太大？"
date: 2026-10-08
categories:
  - Reinforcement Learning
tags:
  - TRPO
math: true
---

# TRPO：为什么强化学习中的策略不能一次更新太大？

> **Trust Region Policy Optimization**
>
> 关键词：Policy Gradient、Surrogate Objective、KL Divergence、Trust Region、Policy Collapse
>
> 前置知识：Policy Gradient、Advantage Function、基础概率论

## 一、引言：Policy Gradient 存在什么问题？

在学习 Policy Gradient（策略梯度）时，我们知道它的核心思想是：

**通过梯度上升，让策略更倾向于选择 Advantage 为正的动作，减少选择 Advantage 为负的动作。**

策略参数的更新公式为：

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
\alpha\nabla_\theta J(\theta_{\text{old}})
$$

其中：

* \(\theta\)：策略神经网络的参数。
* \(J(\theta)\)：当前策略的期望累计奖励（Expected Return）。
* \(\alpha\)：学习率。
* \(\nabla_\theta J(\theta)\)：期望回报关于策略参数的梯度。

这个公式告诉我们应该**朝哪个方向更新策略**。

但是，它没有直接回答另一个重要的问题：

**我们每次到底应该更新多大？**

假设现在有一个机器人，正在学习如何向前行走。

通过采样，算法发现：

> 身体适当前倾，有助于机器人更快地向前移动。

于是，Policy Gradient 会倾向于提高机器人向前倾斜的程度。

如果更新比较小：

* 原来向前倾斜 \(5^\circ\)。
* 更新后向前倾斜 \(6^\circ\)。
* 机器人可能走得更快。

但如果一次更新过大：

* 原来向前倾斜 \(5^\circ\)。
* 更新后向前倾斜 \(35^\circ\)。
* 机器人可能直接失去平衡。

于是我们发现：

**梯度方向正确，并不意味着沿着这个方向走任意远都正确。**

这是因为梯度只提供当前位置附近的局部信息。

可以把梯度想象成：

> 我站在山坡上，发现向东走是下坡。

但是，这并不能证明：

> 向东走一百公里，仍然一直是下坡。

对于普通的梯度优化，这已经是一个需要注意的问题。

但在强化学习中，还有一个更加特殊、更加危险的问题：

**Policy 的变化，会反过来改变 Agent 未来访问的状态分布。**

TRPO（Trust Region Policy Optimization）正是为了缓解这个问题而提出的。

它的核心思想可以概括为：

> **在尽可能提高策略表现的同时，限制新策略与旧策略之间的差异，避免一次过大的策略更新破坏已有的学习成果。**

---

## 二、为什么强化学习中的大步更新更加危险？

### 2.1 先理解：训练数据从哪里来？

假设当前策略是：

$$
\pi_{\text{old}}
$$

我们让 Agent 使用这个策略与环境交互，收集一批轨迹：

$$
\tau = (s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

然后利用这些数据计算 Advantage：

$$
A^{\pi_{\text{old}}}(s,a)
=
Q^{\pi_{\text{old}}}(s,a)
-
V^{\pi_{\text{old}}}(s)
$$

这里：

* \(Q^{\pi_{\text{old}}}(s,a)\)：在状态 \(s\) 执行动作 \(a\)，之后继续遵循旧策略时的期望回报。
* \(V^{\pi_{\text{old}}}(s)\)：从状态 \(s\) 开始一直遵循旧策略时的期望回报。
* \(A^{\pi_{\text{old}}}(s,a)\)：动作 \(a\) 相比旧策略在状态 \(s\) 下的平均表现，好了多少或差了多少。

注意一个重要细节：

**这些训练数据来自旧策略，而且 Advantage 评价的是按照旧策略继续行动时的结果。**

我们还没有真正运行过更新后的策略：

$$
\pi_{\text{new}}
$$

因此，新策略究竟会访问哪些状态，以及在这些状态下表现如何，我们并不完全清楚。

### 2.2 策略变化，会导致状态分布变化

考虑一个简单的路径规划问题。

旧策略访问的状态可能是：

```text
旧策略 π_old

起点 s0
   |
   v
安全路径 s1
   |
   v
安全路径 s2
   |
   v
终点（获得奖励）
```

通过反复采样，Agent 已经比较熟悉这条路径。

此时，如果我们对策略做一次较大的更新，新策略可能变成：

```text
新策略 π_new

起点 s0
   |
   v
陌生路径 x1
   |
   v
陌生路径 x2
   |
   v
危险区域 x3
   |
   v
任务失败
```

这里最重要的变化，不仅是 Agent 选择的动作不同了。

更关键的是：

**Agent 访问的状态也发生了改变。**

我们可以使用状态访问分布（State Visitation Distribution）描述这一点。

旧策略对应：

$$
d_{\pi_{\text{old}}}(s)
$$

新策略对应：

$$
d_{\pi_{\text{new}}}(s)
$$

如果策略只发生小幅改变，那么在一定条件下，我们通常希望：

$$
d_{\pi_{\text{new}}}(s)
\approx
d_{\pi_{\text{old}}}(s)
$$

但如果策略发生剧烈变化，就不能再认为两个状态分布足够接近。

这意味着：

我们原来基于旧策略收集的数据，可能已经无法准确反映新策略的真实表现。

### 2.3 这和监督学习有什么不同？

在典型的监督学习中，我们优化：

$$
\mathcal L(\theta)
=
\mathbb E_{(x,y)\sim D}
[\ell(f_\theta(x),y)]
$$

这里的数据分布 \(D\) 通常是相对固定的。

我们更新模型参数，一般不会直接改变原有训练数据的生成方式。

但是强化学习不同。

强化学习的交互过程是：

```text
策略 Policy
     |
     v
选择动作 Action
     |
     v
影响环境 Environment
     |
     v
产生下一状态 Next State
     |
     v
影响未来采样的数据分布
```

也就是说：

$$
\boxed{
\theta
\rightarrow
\pi_\theta
\rightarrow
a
\rightarrow
s'
\rightarrow
d_{\pi_\theta}
}
$$

策略参数不仅影响模型输出，还会影响未来训练数据的分布。

因此，强化学习中的策略优化存在一个特殊的困难：

> **当我们使用旧策略的数据去优化新策略时，如果新旧策略相差太大，那么优化目标就可能不再准确反映新策略的真实回报。**

这也是理解 TRPO 的第一把钥匙。

---

## 三、为什么需要 Surrogate Objective？

要理解 TRPO，必须先区分两个概念：

1. 我们真正想优化的目标：**True Objective**。
2. 我们实际能够利用旧数据优化的目标：**Surrogate Objective（代理目标）**。

### 3.1 真正想优化的是 Expected Return

强化学习的最终目标是最大化期望累计奖励：

$$
J(\pi)
=
\mathbb E_{\tau\sim\pi}
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\right]
$$

其中 \(\gamma\in(0,1)\) 为折扣因子。

我们希望：

$$
J(\pi_{\text{new}})
>
J(\pi_{\text{old}})
$$

换句话说：

**新策略在环境中运行之后，应该比旧策略获得更高的期望回报。**

那如何判断新策略究竟提高了多少？

这里需要使用一个重要的理论结果：Performance Difference Lemma（性能差异引理）。

### 3.2 Performance Difference Lemma

在相同环境和初始状态分布下，定义归一化折扣状态访问分布：

$$
d_\pi(s)
=
(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^t
P(s_t=s\mid\pi)
$$

那么，有：

$$
\boxed{
J(\pi_{\text{new}})
-
J(\pi_{\text{old}})
=
\frac{1}{1-\gamma}
\mathbb E_{\substack{
s\sim d_{\pi_{\text{new}}}\\
a\sim\pi_{\text{new}}(\cdot|s)
}}
\left[
A^{\pi_{\text{old}}}(s,a)
\right]
}
$$

这个公式非常重要。

我们可以逐项理解它：

* \(J(\pi_{\text{new}})-J(\pi_{\text{old}})\)：新旧策略之间的真实回报差异。
* \(d_{\pi_{\text{new}}}\)：新策略实际访问的状态分布。
* \(\pi_{\text{new}}(a|s)\)：新策略在状态 \(s\) 下选择动作 \(a\) 的概率。
* \(A^{\pi_{\text{old}}}(s,a)\)：相对于旧策略，该动作的 Advantage。

公式告诉我们：

> 想知道新策略究竟提高了多少，就需要考察新策略会访问哪些状态，以及它在这些状态下选择的动作相对于旧策略有多好。

注意，这里用的是：

$$
\boxed{d_{\pi_{\text{new}}}}
$$

而不是旧策略的状态访问分布。

### 3.3 问题：我们没有新策略的采样数据

在当前这一步策略更新中，我们手里的数据来自：

$$
\pi_{\text{old}}
$$

因此我们比较容易估计：

$$
d_{\pi_{\text{old}}}(s)
$$

但我们还没有让新策略真正执行大量交互。

于是，TRPO 所依据的局部近似思想是：

**暂时使用旧策略的状态访问分布，代替新策略的状态访问分布。**

也就是：

$$
d_{\pi_{\text{new}}}(s)
\approx
d_{\pi_{\text{old}}}(s)
$$

由此构造一个代理目标：

$$
\boxed{
L_{\pi_{\text{old}}}(\pi_{\text{new}})
=
J(\pi_{\text{old}})
+
\frac{1}{1-\gamma}
\mathbb E_{\substack{
s\sim d_{\pi_{\text{old}}}\\
a\sim\pi_{\text{new}}(\cdot|s)
}}
\left[
A^{\pi_{\text{old}}}(s,a)
\right]
}
$$

这就是 Surrogate Objective。

与真正的目标相比，它主要做了一件事：

| 对比项       | True Objective               | Surrogate Objective          |
| --------- | ---------------------------- | ---------------------------- |
| 状态访问分布    | 新策略 \(d_{\pi_{\text{new}}}\) | 旧策略 \(d_{\pi_{\text{old}}}\) |
| 动作分布      | 新策略 \(\pi_{\text{new}}\)     | 新策略 \(\pi_{\text{new}}\)     |
| Advantage | \(A^{\pi_{\text{old}}}\)     | \(A^{\pi_{\text{old}}}\)     |
| 含义        | 真实策略表现                       | 基于旧状态分布的局部近似                 |

需要强调：

**Surrogate Objective 并不是随便构造的近似。**

TRPO 原论文指出，在旧策略的位置，它和真实目标具有相同的函数值与一阶梯度：

$$
L_{\pi_{\text{old}}}(\pi_{\text{old}})
=
J(\pi_{\text{old}})
$$

以及：

$$
\left.
\nabla_\theta
L_{\pi_{\text{old}}}(\pi_\theta)
\right|_{\theta=\theta_{\text{old}}}
=
\left.
\nabla_\theta J(\pi_\theta)
\right|_{\theta=\theta_{\text{old}}}
$$

这意味着：

**在旧策略附近，Surrogate Objective 可以提供正确的一阶局部优化方向。**

但这里隐藏了最关键的限制：

> 一阶局部正确，不代表当新策略距离旧策略很远时，代理目标仍然可靠。

这正是 TRPO 原论文 Equation (3) 和 Equation (4) 想表达的重要思想。[1]

---

## 四、用一个具体例子理解：为什么 Surrogate 变好，真实 Return 反而下降？

前面的结论有点抽象。

下面构造一个非常简单的环境，直接用数字证明这个现象。

### 4.1 环境设计

假设 Agent 在状态 \(s_0\) 有两个选择：

```text
             s0（起点）
             /       \
          safe       enter
           |            |
       奖励 0           s1
                       /  \
                    good   bad
                     |      |
                   +10     -10
```

其中：

* 选择 `safe`：立即结束，奖励为 \(0\)。
* 选择 `enter`：进入状态 \(s_1\)，当前奖励为 \(0\)。
* 在 \(s_1\) 选择 `good`：获得 \(+10\) 并结束。
* 在 \(s_1\) 选择 `bad`：获得 \(-10\) 并结束。

设置折扣因子：

$$
\gamma=0.9
$$

为了模拟神经网络参数共享造成的动作概率联动，我们使用一个参数 \(q\) 控制策略：

$$
\pi_q(\text{enter}|s_0)=q
$$

$$
\pi_q(\text{good}|s_1)=1-q
$$

因此：

**当 \(q\) 变大时，Agent 更倾向于进入 \(s_1\)，但进入后选择好动作的概率反而降低。**

这可以理解为一个简化的参数共享模型：不同状态的动作概率不是完全独立调节的。

### 4.2 计算真实 Return

在状态 \(s_1\)，预期奖励为：

$$
(1-q)\times10+q\times(-10)
=
10-20q
$$

Agent 以概率 \(q\) 进入 \(s_1\)，所以真实期望回报为：

$$
\begin{aligned}
J(q)
&=
q\cdot\gamma\cdot(10-20q)\\
&=
9q(1-2q)
\end{aligned}
$$

假设旧策略：

$$
q_{\text{old}}=0.1
$$

那么：

$$
J(0.1)
=
9\times0.1\times(1-0.2)
=
0.72
$$

现在观察它的梯度：

$$
J'(q)=9(1-4q)
$$

因此：

$$
J'(0.1)=5.4>0
$$

梯度告诉我们：

**在当前位置，增大 \(q\) 是有利的。**

这并没有错。

例如，适当地更新到：

$$
q_{\text{new}}=0.2
$$

真实回报就会变为：

$$
J(0.2)=1.08
$$

确实比原来更好。

但是，如果我们更新得非常猛烈：

$$
q_{\text{new}}=0.9
$$

那么：

$$
J(0.9)
=
9\times0.9\times(1-1.8)
=
-6.48
$$

策略的真实表现反而大幅下降了。

### 4.3 Surrogate Objective 会怎样评价这次更新？

回忆一下：

Surrogate Objective 在计算时，将状态访问分布固定在旧策略上。

对于这个例子，将旧策略下的 Advantage 代入代理目标，可以得到：

$$
L_{q_{\text{old}}}(q)
=
0.72+5.4(q-0.1)
$$

由于这个例子里的动作概率与 \(q\) 呈线性关系，因此代理目标恰好是旧策略处的切线。

于是我们得到：

| 策略参数 \(q\) | Surrogate Objective | 真实 Return |
| ---------- | ------------------: | --------: |
| 0.1（旧策略）   |                0.72 |      0.72 |
| 0.2（小幅更新）  |                1.26 |      1.08 |
| 0.9（大幅更新）  |                5.04 |     -6.48 |

现在，问题就非常清楚了。

当 \(q\) 从 \(0.1\) 更新到 \(0.9\) 时：

* Surrogate Objective 从 \(0.72\) 增加到 \(5.04\)。
* 真实 Return 却从 \(0.72\) 下降到 \(-6.48\)。

**代理目标认为策略变得更好，但真实执行后，策略表现却严重退化。**

### 4.4 错误究竟出在哪里？

旧策略进入 \(s_1\) 的概率只有：

$$
P_{\text{old}}(s_1)=0.1
$$

但新策略进入 \(s_1\) 的概率变成：

$$
P_{\text{new}}(s_1)=0.9
$$

也就是说：

新策略会频繁进入一个旧策略很少访问的状态。

但是 Surrogate Objective 仍然使用旧策略的状态访问频率评价新策略。

因此，它低估了新策略在 \(s_1\) 中做出坏动作所产生的整体负面影响。

这里还有一个值得注意的事实：

**这个例子即使使用精确的旧策略 Advantage，依然会出现 Surrogate Objective 与真实 Return 严重不一致的情况。**

问题不仅来自 Advantage 估计不准，还来自状态访问分布的变化。

这并不代表每次梯度更新都会产生这种结果，而是说明：

> **当新旧策略距离过大时，仅仅优化 Surrogate Objective，并不能保证真实回报得到改善。**

这就是我们需要 Trust Region 的根本原因。

---

## 五、Trust Region：只在可信区域内优化

现在我们已经发现：

Surrogate Objective 是一个局部近似。

那么自然就会想到：

**既然它只在旧策略附近可靠，为什么不限制每次策略更新的范围呢？**

这就是 Trust Region（信赖域）的思想。

### 5.1 什么是 Trust Region？

假设我们站在一个山坡上。

我们只能根据附近的地形判断哪个方向更好，而无法准确知道远处的地形。

那么，一个合理的优化方式是：

1. 根据当前位置的梯度，找到有希望的方向。
2. 只允许自己在附近一个可信区域内移动。
3. 到达新位置之后，重新获取信息，再继续优化。

这就是 Trust Region。

对于策略优化，它对应：

$$
\max_{\pi_{\text{new}}}
L_{\pi_{\text{old}}}(\pi_{\text{new}})
$$

同时增加限制：

$$
D(\pi_{\text{old}},\pi_{\text{new}})
\leq\delta
$$

其中：

* \(D\)：新旧策略之间的差异度量。
* \(\delta\)：允许策略发生多大变化的阈值。

换句话说：

**尽可能提升代理目标，但不允许新策略一次偏离旧策略太远。**

那么问题来了：

应该使用什么方式衡量新旧策略之间的差异？

TRPO 选择了一个重要工具：

**KL Divergence（KL 散度）。**

---

## 六、KL Divergence：衡量新旧策略改变了多少

### 6.1 KL Divergence 是什么？

在离散动作空间中，KL 散度的定义为：

$$
\boxed{
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_x
P(x)\log\frac{P(x)}{Q(x)}
}
$$

放到策略优化中：

$$
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot|s)
\Vert
\pi_{\text{new}}(\cdot|s)
\right)
=
\sum_a
\pi_{\text{old}}(a|s)
\log
\frac{\pi_{\text{old}}(a|s)}
{\pi_{\text{new}}(a|s)}
$$

不需要一开始就从信息论角度理解这个公式。

现在只需要建立一个直觉：

> KL 散度可以用来衡量两个动作概率分布之间的差异。两个分布越接近，KL 越接近 0。

这里特别注意：

KL 散度严格来说不是数学意义上的距离，因为它一般不满足对称性：

$$
D_{\mathrm{KL}}(P\Vert Q)
\neq
D_{\mathrm{KL}}(Q\Vert P)
$$

### 6.2 用具体数字理解 KL

假设 Agent 在某个状态有两个动作：

$$
\pi_{\text{old}}
=
(0.5,0.5)
$$

**情况一：小幅更新**

$$
\pi_{\text{new}}
=
(0.55,0.45)
$$

代入公式：

$$
D_{\mathrm{KL}}
(\pi_{\text{old}}\Vert\pi_{\text{new}})
\approx0.005
$$

KL 非常小，说明动作概率分布变化不大。

**情况二：大幅更新**

$$
\pi_{\text{new}}
=
(0.9,0.1)
$$

此时：

$$
D_{\mathrm{KL}}
(\pi_{\text{old}}\Vert\pi_{\text{new}})
\approx0.511
$$

KL 明显增大。

说明新策略的行为与旧策略之间已经出现较大差异。

在 TRPO 中，我们利用 KL 来控制每次策略更新的幅度。

### 6.3 为什么限制 KL，而不是限制参数变化？

一个很自然的问题是：

为什么不用：

$$
\|\theta_{\text{new}}-\theta_{\text{old}}\|_2
\leq\epsilon
$$

直接约束神经网络参数变化？

原因是：

**参数变化的大小，并不能直接代表策略行为变化的大小。**

同样大小的参数变化，在不同参数位置、不同网络结构下，可能对动作概率产生非常不同的影响。

例如：

* 参数只改变一点，动作概率可能几乎不变。
* 参数也只改变一点，但某个状态下的动作概率可能发生剧烈变化。

我们真正想控制的不是：

> 神经网络的权重改变了多少？

而是：

> Agent 面对同样的状态时，其动作选择行为改变了多少？

KL 直接比较动作概率分布，因此更符合策略优化的目的。

这也与后续 Natural Policy Gradient 所采用的信息几何思想有关。

---

## 七、TRPO 的核心优化目标

理解了 Surrogate Objective 和 KL Divergence，现在就可以把 TRPO 的核心公式组合起来。

### 7.1 TRPO 优化什么？

在最大化期望奖励的记号下，TRPO 的实际核心优化问题可以写成：

$$
\boxed{
\begin{aligned}
\max_\theta\quad&
L_{\theta_{\text{old}}}(\theta)
\\[6pt]
\text{s.t.}\quad&
\mathbb E_{s\sim d_{\pi_{\text{old}}}}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot|s)
\Vert
\pi_\theta(\cdot|s)
\right)
\right]
\leq\delta
\end{aligned}
}
$$

这个公式分为两个部分。

**第一部分：优化 Surrogate Objective**

$$
\max_\theta L_{\theta_{\text{old}}}(\theta)
$$

希望新策略在旧策略数据构造的代理目标上表现得更好。

**第二部分：限制平均 KL 散度**

$$
\mathbb E_{s\sim d_{\pi_{\text{old}}}}
\left[
D_{\mathrm{KL}}
(
\pi_{\text{old}}(\cdot|s)
\Vert
\pi_\theta(\cdot|s)
)
\right]
\leq\delta
$$

希望新策略相对于旧策略的平均行为变化不要超过指定阈值。

二者结合起来就是：

> **在 KL 约束允许的范围内，寻找能够最大化 Surrogate Objective 的新策略。**

### 7.2 为什么是 Average KL？

这里需要区分原论文中的理论结果和实际算法。

在理论分析中，TRPO 考虑最大 KL：

$$
D_{\mathrm{KL}}^{\max}
(
\pi_{\text{old}},
\pi_{\text{new}}
)
=
\max_s
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot|s)
\Vert
\pi_{\text{new}}(\cdot|s)
\right)
$$

它要求：

**在所有状态下，新旧策略之间的 KL 都不能太大。**

这种约束具有更强的理论意义。

原论文证明，在一定假设下，真实目标与代理目标之间的偏差可以通过最大策略散度控制。[1]

但实际的状态空间可能非常巨大，甚至是连续的。

我们无法轻易遍历所有状态并找到其中最大的 KL。

因此，TRPO 实际采用一个更容易估计的近似：

$$
\bar D_{\mathrm{KL}}
=
\mathbb E_{s\sim d_{\pi_{\text{old}}}}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot|s)
\Vert
\pi_{\text{new}}(\cdot|s)
\right)
\right]
$$

也就是：

**只考虑旧策略访问的状态分布下，平均意义上的 KL 差异。**

这种方法更适合实际训练。

但是，也要注意：

**平均 KL 约束不能保证每个状态下的 KL 都很小。**

所以，TRPO 实际算法中的稳定性保证并不等同于理论中的严格单调改进保证。

另外，本文采用的是 TRPO 原论文中的 \(\pi_{\text{old}}\Vert\pi_{\text{new}}\) 方向。有些教学资料或实现采用相反的 KL 方向，两者一般不严格相等，阅读时应注意其定义。

---

## 八、Importance Sampling：怎样用旧策略的数据评估新策略？

前面还有一个没有解决的问题。

在 Surrogate Objective 中，我们需要计算：

$$
\mathbb E_{a\sim\pi_{\text{new}}(\cdot|s)}
[
A^{\pi_{\text{old}}}(s,a)
]
$$

但我们采样时，动作实际上来自：

$$
a\sim\pi_{\text{old}}(\cdot|s)
$$

这时可以使用 Importance Sampling（重要性采样）。

在满足适当的概率支持条件时：

$$
\mathbb E_{a\sim\pi_{\text{new}}}
[A(s,a)]
=
\mathbb E_{a\sim\pi_{\text{old}}}
\left[
\frac{\pi_{\text{new}}(a|s)}
{\pi_{\text{old}}(a|s)}
A(s,a)
\right]
$$

定义概率比值：

$$
\boxed{
r_\theta(s,a)
=
\frac{\pi_\theta(a|s)}
{\pi_{\text{old}}(a|s)}
}
$$

它表示：

**对于相同的状态和动作，新策略赋予这个动作的概率，是旧策略的多少倍。**

例如：

如果旧策略选择某动作的概率是 \(0.2\)，新策略变成 \(0.4\)，那么：

$$
r_\theta(s,a)=2
$$

这意味着新策略选择该动作的概率是旧策略的两倍。

因此，我们可以把 Surrogate Objective 中与策略参数有关的部分写成：

$$
\boxed{
L^{\mathrm{PG}}(\theta)
=
\mathbb E_{\substack{
s\sim d_{\pi_{\text{old}}}\\
a\sim\pi_{\text{old}}
}}
\left[
r_\theta(s,a)
A^{\pi_{\text{old}}}(s,a)
\right]
}
$$

这里省略了与 \(\theta\) 无关的常数，以及不影响最优解的正比例因子。

这个公式非常重要。

因为后续学习 PPO 时，我们还会遇到：

$$
r_\theta(s,a)A^{\pi_{\text{old}}}(s,a)
$$

现在可以提前理解它的意义：

> 使用重要性采样比值，对旧策略收集的动作样本重新加权，从而估计新策略在旧状态分布上的代理目标。

但必须强调：

**重要性采样这里只修正了动作分布，并没有自动修正新旧策略之间的状态访问分布差异。**

因此，我们仍然需要控制策略更新幅度。

---

## 九、TRPO 如何实现约束优化？

理解算法思想后，可以简单了解实际求解流程。

TRPO 的目标是：

$$
\max_\theta L(\theta)
$$

满足：

$$
\bar D_{\mathrm{KL}}
(\pi_{\text{old}},\pi_\theta)
\leq\delta
$$

这是一个带约束的优化问题。

常见的普通梯度上升不能直接保证每次更新都满足 KL 约束。

因此，TRPO 使用局部近似。

设：

$$
\Delta\theta
=
\theta-\theta_{\text{old}}
$$

在旧参数附近，代理目标可以使用一阶 Taylor 展开：

$$
L(\theta)
\approx
L(\theta_{\text{old}})
+
g^T\Delta\theta
$$

其中：

$$
g=\nabla_\theta L(\theta_{\text{old}})
$$

KL 约束则可以使用二阶近似：

$$
\bar D_{\mathrm{KL}}
\approx
\frac12
\Delta\theta^TF\Delta\theta
$$

这里的 \(F\) 与 Fisher Information Matrix 有关，描述局部策略分布对参数变化的敏感程度。

于是，优化问题变成：

$$
\boxed{
\begin{aligned}
\max_{\Delta\theta}\quad&
g^T\Delta\theta\\
\text{s.t.}\quad&
\frac12
\Delta\theta^TF\Delta\theta
\leq\delta
\end{aligned}
}
$$

它的更新方向与下面的表达式有关：

$$
\Delta\theta\propto F^{-1}g
$$

这就是 Natural Policy Gradient 的核心思想之一。

### Conjugate Gradient 在做什么？

对于大型神经网络，显式计算并求逆 Fisher 矩阵通常非常昂贵。

因此，TRPO 使用 Conjugate Gradient（共轭梯度）近似求解：

$$
Fx=g
$$

从而得到更新方向。

再利用 KL 约束对步长进行缩放，并使用 Backtracking Line Search（回溯线搜索）检查候选更新。

候选更新需要通过相应的代理目标改进条件和采样 KL 约束，否则就缩小步长重新尝试。

所以，实际训练的大致流程是：

```text
使用旧策略采样
       |
       v
估计 Advantage
       |
       v
构造 Surrogate Objective
       |
       v
计算局部优化方向
       |
       v
根据 KL 约束缩放步长
       |
       v
Backtracking Line Search
       |
       v
接受满足条件的候选策略
       |
       v
重新采样，继续训练
```

对于第一次学习 TRPO，暂时不用深入 Conjugate Gradient 的具体实现。

只需要记住：

**TRPO 利用近似二阶信息找到合适的更新方向，再通过 KL 约束和线搜索，尽量避免策略一次发生过大的变化。**

---

## 十、为什么 RL 不能像监督学习一样一直猛降 Loss？

现在可以回答最初的问题了。

首先，需要澄清：

**监督学习也不能无限制地使用大步长优化。**

如果学习率过大，监督学习同样可能不收敛；即使训练 Loss 降低，也不代表测试集性能一定提升。

但强化学习还有一个额外问题。

对于典型的监督学习：

$$
\mathcal L(\theta)
=
\mathbb E_{(x,y)\sim D}
[\ell(f_\theta(x),y)]
$$

训练数据分布 \(D\) 通常不会因为参数更新而立即发生改变。

但在强化学习中：

$$
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
[R(\tau)]
$$

轨迹本身由当前策略决定。

当参数更新时：

$$
\theta_{\text{old}}
\rightarrow
\theta_{\text{new}}
$$

策略随之改变：

$$
\pi_{\text{old}}
\rightarrow
\pi_{\text{new}}
$$

进一步导致：

$$
d_{\pi_{\text{old}}}
\rightarrow
d_{\pi_{\text{new}}}
$$

因此：

**优化目标所依赖的数据分布，也会随着策略改变而变化。**

而我们手里的训练 batch 仍然来自旧策略。

如果策略改变太大：

* 旧策略的采样数据可能不再具有代表性。
* 旧策略下的 Advantage 无法充分反映新策略实际访问状态后的整体表现。
* Surrogate Objective 可能继续改善，但真实 Expected Return 可能下降。

因此，在 RL 中不能简单地认为：

> Policy Loss 越低，新策略就一定越好。

对 TRPO 来说，更合理的思想是：

> **在旧策略附近优化代理目标，并利用 KL 散度限制行为变化，使代理目标保持在相对可信的范围内。**

这就是 TRPO 和普通 Policy Gradient 之间的重要区别。

---

## 十一、TRPO 是否能够保证训练永远不会崩溃？

不能。

这是理解 TRPO 时特别容易出现的误区。

TRPO 的理论基础提供了在一定条件下约束策略更新、控制真实目标与代理目标误差的方法。

但实际实现中存在很多近似：

1. Advantage 通常是通过有限样本估计的，并不完全准确。
2. 实际使用平均 KL，而不是理论分析中的最大 KL。
3. 神经网络函数近似、有限样本和数值优化都会引入误差。
4. 线搜索检查的是样本上的代理目标改进与 KL 约束，并不直接检查真实期望回报。

所以：

**TRPO 能够提高策略更新的稳定性，但并不能保证实际训练中的每一次更新都使真实回报上升。**

Trust Region 更适合理解为一种控制更新风险的方法，而不是绝对的安全保证。

---

## 十二、从 TRPO 到 PPO：为什么接下来要学习 Clipping？

到目前为止，我们已经知道：

TRPO 希望解决的问题是：

> 如何利用旧策略收集的数据，尽可能提高策略表现，同时避免更新过大？

它采用的方案是：

$$
\max_\theta L^{\mathrm{PG}}(\theta)
$$

并且限制：

$$
\bar D_{\mathrm{KL}}
\leq\delta
$$

但代价是优化过程比较复杂。

实际 TRPO 通常需要使用：

* Fisher/KL 的局部二阶信息。
* Conjugate Gradient。
* 步长缩放和 Backtracking Line Search。

因此，后来的 PPO（Proximal Policy Optimization）尝试使用更简单的一阶优化方法达到类似的目的。[4]

其中，PPO-Clip 的核心思路是：

**通过对概率比值施加 Clipping，减少代理目标对过大策略变化的鼓励。**

它不需要像 TRPO 一样显式求解复杂的二阶约束优化问题。

但要注意：

**PPO-Clip 并不意味着新旧策略之间的 KL 被严格限制在某个范围之内。**

它是在优化目标中抑制过大的策略变化，而不是天然提供一个硬性的 KL 约束。

因此，理解 TRPO 之后再学习 PPO，就可以很自然地建立如下知识链条：

```text
Policy Gradient
      |
      v
发现更新过大可能导致性能退化
      |
      v
Surrogate Objective 只是局部近似
      |
      v
TRPO：用 KL 约束策略变化
      |
      v
更新稳定性有所改善，但实现复杂
      |
      v
PPO：用更简单的一阶方法限制更新激励
```

---

## 十三、常见误区与面试问题

### Q1：为什么 Policy Gradient 的梯度方向正确，最终性能却可能变差？

因为梯度只反映局部的一阶变化趋势。

在当前策略附近，沿梯度方向进行足够小的更新可能提高期望回报，但较大的更新可能超出这个局部近似的有效范围。

对于 RL，策略更新还可能改变状态访问分布，进一步增加优化难度。

### Q2：TRPO 为什么需要 Surrogate Objective？

因为新策略的真实 Expected Return 依赖于新策略访问的状态分布，而当前训练数据通常由旧策略产生。

TRPO 因此构造一个使用旧策略状态访问分布的代理目标，使我们能够利用已有样本进行策略优化。

### Q3：为什么 Surrogate Objective 不能无限优化？

因为它与真实目标只在旧策略的位置具有相同的函数值和一阶梯度。

随着新策略远离旧策略，状态访问分布可能发生明显变化，代理目标和真实回报之间的误差也可能增大。

### Q4：TRPO 为什么使用 KL 而不是参数距离？

因为我们真正想限制的是 Agent 行为分布的变化，而不是神经网络参数本身的变化。

参数距离并不能稳定、直接地反映动作概率分布之间的差异，而 KL 能够比较新旧策略的动作概率分布。

### Q5：TRPO 的核心创新是什么？

TRPO 将策略优化构造成一个受约束的优化问题：

**在最大化局部 Surrogate Objective 的同时，利用 KL Divergence 限制新旧策略之间的行为差异。**

它还结合了单调性能改进的理论分析，并设计了适用于参数化策略的实际近似求解方法。

---

## 十四、总结：真正理解 TRPO，需要记住什么？

回顾整个学习过程：

**第一步：Policy Gradient 只能提供局部优化方向。**

它告诉我们怎样调整动作概率可能提高期望回报，但无法保证任意大的更新都有效。

**第二步：策略改变会引起状态访问分布改变。**

这使得新策略的真实回报不能简单通过旧策略采集的数据完全确定。

**第三步：Surrogate Objective 使用旧策略的状态访问分布。**

它在旧策略附近具有正确的一阶优化性质，但当新旧策略相距很远时，可能与真实回报发生明显偏离。

**第四步：TRPO 利用 KL Divergence 建立 Trust Region。**

要求每次更新不能让策略行为改变过大，以此提高局部近似的可靠性。

**第五步：通过受约束的策略优化，尽可能提高训练稳定性。**

TRPO 并不保证实际训练绝不会退化，但它为控制 Policy Gradient 的更新风险提供了一套具有理论动机的方法。

### 一句话总结

> **TRPO 要解决的不是“怎样找到更大的梯度”，而是“怎样在当前采样数据仍然可信的范围内，安全地利用梯度改进策略”。**
>
> 它通过最大化 Surrogate Objective，并使用 KL Divergence 限制新旧策略的差异，从而缓解过大策略更新导致真实回报下降的问题。

如果面试中被问到：

**“为什么强化学习不能像普通监督学习一样一直猛降 Loss？”**

可以这样回答：

> 在典型监督学习中，模型优化所使用的数据分布相对固定。但在强化学习中，策略决定动作，动作影响环境，进而改变后续状态访问分布。
>
> Policy Gradient 使用旧策略采集的数据构造代理目标，它只能在当前策略附近可靠地反映真实回报的变化。如果一次更新过大，新策略的状态访问分布就可能显著偏离旧策略，导致 Surrogate Objective 虽然改善，真实 Return 却下降。
>
> TRPO 因此引入 Trust Region，通过 KL Divergence 限制新旧策略之间的行为差异，让策略尽量在代理目标可信的局部区域内优化。

到这里，TRPO 最核心的思想就串联起来了：

**Policy Gradient 告诉我们往哪里走，TRPO 则进一步尝试回答每一步应该走多远。**

---

## 参考文献

[1] Schulman, J., et al. (2015). **Trust Region Policy Optimization**. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37.

* 论文主页：https://proceedings.mlr.press/v37/schulman15.html
* 论文 PDF：https://proceedings.mlr.press/v37/schulman15.pdf
* arXiv：https://arxiv.org/abs/1502.05477
* **推荐阅读**：Introduction、Section 2（Performance Difference 与 Surrogate Objective）、Section 3（Monotonic Improvement）、Section 4–5（TRPO 的实际优化目标及样本估计）。重点关注原论文 Equation (1)–(4)、(10)、(12)–(15)。

[2] Kakade, S. M., & Langford, J. (2002). **Approximately Optimal Approximate Reinforcement Learning**. *International Conference on Machine Learning (ICML)*.

* 论文 PDF：https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf
* 说明：介绍 Conservative Policy Iteration，与 TRPO 的策略改进理论基础密切相关。

[3] OpenAI. **Spinning Up in Deep RL — Trust Region Policy Optimization**.

* 官方教程：https://spinningup.openai.com/en/latest/algorithms/trpo.html
* 说明：提供 TRPO 的核心公式、KL 约束、局部二阶近似、Conjugate Gradient 和 Backtracking Line Search 的解释，适合与原论文对照学习。

[4] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms**. *arXiv:1707.06347*.

* 论文：https://arxiv.org/abs/1707.06347
* OpenAI PPO 教程：https://spinningup.openai.com/en/latest/algorithms/ppo.html
* 说明：用于进一步理解 PPO 如何沿着 TRPO 的稳定策略更新思想，提出更加简单的优化方法。
