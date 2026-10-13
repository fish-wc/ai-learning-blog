---
title: "REINFORCE 到底是什么？"
date: 2026-10-11
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

# PPT 制作脚本

## 第1页，要讲清楚的1个问题：REINFORCE 到底是什么？

要讲明白该问题，需要拆解为几个子问题。
1. 对数求导策略是什么？（对数求导策略是Policy Gradient推导出REINFORCE的关键数学技巧）
2. REINFORCE 是怎么从 Policy Gradient 推出来的？（关键数学推导）
3. 理解REINFORCE 到底是什么？

## 第2页，什么是对数求导策略

假设有概率分布$p(x)$，对于任意概率分布 $p$ 的对数 $\log p$ 进行求导：

$$
\nabla \log p
=
\frac{
\nabla p
}{
p
}
$$

将分母 $p$ 乘到等式另一边，于是：

$$
\nabla p
=
p
\nabla \log p
$$

对数求导策略，是对 对数求导，然后通过等式变换，可以将原来的概率分布求导的表达式$\nabla p$ 重新引入概率分布表达式 $p$ 。这样的好处在于，对于连续分布，可以将积分的表达式等效变换为期望的表达式，比如：
$$
\int \nabla p d x=\int p \nabla \log p dx = \mathbb E_{x \sim p}[\log p]
$$

> 为什么要将积分表达式转换为期望表达式？
>
> 因为，对于期望表达式的求值，可以通过Monte Carlo 采样来进行近似求解。
> 所以，将积分表达式转换为期望表达式，是为了近似求解。

## 第3页，Policy Gradient 章节回顾

回顾上一章节[policy gradient](./Policy_Gradient.md)，策略梯度的优化目标，希望环境给轨迹的奖励的期望最大，轨迹的出现满足概率分布$p_\theta(\tau)$，表达式如下：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
[R(\tau)]
$$

然后，将期望表达式展开为积分表达式，然后对参数$\theta$求导，奖励函数与参数$\theta$无关，移动到求导外侧，最终得到梯度表达式：

$$
\nabla_{\theta} J(\theta)
=
\int
R(\tau)
\nabla_\theta p_\theta(\tau)
d\tau
$$

## 第4页，REINFORCE 从 Policy Gradient 推导的过程：利用对数求导策略，将梯度的积分表达式转换为期望表达式。

由对数求导技巧可知：

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)
\nabla_\theta \log p_\theta(x)
$$

所以，梯度的表达式可以等效为：

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

然后再将积分的表达式写成期望的表达式：

$$

\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)
\nabla_\theta
\log p_\theta(\tau)
\right]

$$


## 第5页，理解REINFORCE 到底是什么？

这里将梯度的积分表达式转换为期望表达式之后，可以通过Mento Carlo采样求取梯度的近似值。

比如，基于当前策略跑100条trajectory:

$$
\tau_1,\tau_2,\cdots,\tau_{100}
$$

每条策略都能有环境的奖励反馈$R(\tau_i)$。
对于概率分布函数$p_\theta(\tau)$而言，是一个连续函数，函数里面有$\theta$和$\tau$两个变量，对于变量$\tau$而言，通过$\tau_i$固定下来，那么$p_\theta(\tau_i)$就是关于变量$\theta$的连续函数，自然可以求导计算梯度，最终得到：

$$
\hat g
=
\frac{1}{100}
\sum_{i=1}^{100}
R(\tau_i)
\nabla_\theta\log p_\theta(\tau_i)
$$

用这个式子来近似求解真实梯度。

所以，**REINFORCE 本质上就是 Monte Carlo Policy Gradient。**

## 第6页，过渡页，说明接下来是额外的知识

补充几个REINFORCE的重要知识点：
1. 为什么实际 REINFORCE 使用 Return-to-Go更新策略？
2. 为什么要减去一个baseline？
3. 减去baseline之后为什么梯度的期望不变。


## 第7页，对于梯度的表达式而言，将概率分布的导数$\log p_\theta(\tau)$展开化简，以便于后序知识点讨论

对于$p_{\theta}(\tau)$而言，指的是这条轨迹的概率，轨迹的概率由每一个状态下采取的动作策略$\pi_\theta(a_t|s_t)$和 该状态与动作下的状态转移$P(s_{t+1}|s_t,a_t)$决定，所以从初始状态累乘过来：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_t
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

取对数$\log$ ：

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

由于是对参数$\theta$求导，而初始的状态与参数$\theta$无关，状态的转换$P(s_{t+1}|s_t,a_t)$也是由环境所决定的，与参数$\theta$也是无关的，所以对参数$\theta$求导都是0，所以得到：

$$
\nabla_\theta
\log p_\theta(\tau)
=
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

带入最开始的目标梯度公式得到：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]

$$

## 第8页，为什么实际 REINFORCE 使用 Return-to-Go更新策略？

Return-to-go更新策略：即未来回报更新策略。

针对一条轨迹trajectory而言，结合蒙特卡洛采样求近似值的思想，将如上简化的梯度表达式用到这里，单条轨迹$\tau$贡献的梯度的写法可以写成：

$$
R(\tau)
\sum_t
\nabla\log\pi(a_t|s_t)
$$

通过这个表达式可以理解，对于每一个action而言，都会乘以整条轨迹的奖励。也就是说，每个动作$a_t$的奖励都是一样的，都是$R(\tau)$，明显这是不合理的。

所以，需要设计一个机制，用来单独度量一个动作$a_t$的环境奖励是多少。

而对于一个动作的奖励，不仅有当前立马能够得到反馈的奖励$r$，还有当前动作对于未来的影响$r_{future}$，需要设计一种机制，将动作$a_t$的当前奖励和未来奖励的累计影响给表示出来，于是就有：

$$
G_t
=
\sum_{k=t}^{T-1}
\gamma^{k-t}r_{k}
$$

表示，从当前时间$t$开始，通过衰减因子$\gamma$的形式，粗略的计算一个当前影响和未来影响的累计值。理论上，衰减因子$\gamma$应该越往后越小。

于是，对于每个动作$a_t$所贡献的梯度，从原来的：

$$
R(\tau)

\nabla\log\pi(a_t|s_t)
$$

变成了：

$$
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

于是，梯度的计算也就约等于：

$$

\nabla_\theta J
\approx
\sum_t
G_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)

$$

后面统一用$G_t$表示当前动作$a_t$的奖励。

## 第9页，当前梯度表达式存在的问题：动作间的差异变小。方差大。

在一个状态$s_t$下，采取动作$a_t$，会有一个综合奖励$G_t$。但是存在一个明显的问题，如果当前状态比较好，对于不同的动作$a_{ti}$而言，都能有一个不错的综合收益$G_t$，那么不同动作$a_t$所带来的差异就会变得很小。

比如状态$s_t$下，三个动作收益分别是[99,100,101]，收益都很高，但是相对差异被缩小了。于是梯度更新的时候，不能有效的增益更好的动作。

对于一组数据，减去一个固定值，方差不变，因为方差反应一组数据的离散程度。但是整个题目公式还要乘以$\nabla_\theta \log\pi_\theta(a_t|s_t)$ ,所以会导致方差平方倍数的扩大或者缩小，导致**单个动作带来的梯度数值的方差**变得更大。所以，如果能够降低系数$G_t$，那么也就能降低方差。

补充方差公式：
$ (\mathrm{Var}(X) = \mathbb{E}\Big[\big(X-\mathbb{E}[X]\big)^2\Big]) $


$ (\mathrm{Var}[c\cdot X]=c^2\mathrm{Var}[X]) $

## 第10页，解决方案：在奖励$G_t$上减去一个baseline

把$G_t$变成$G_t - b(s_t)$，对于$b(s_t)$的取值而言，通常取当前状态$s_t$下采取动作综合收益的期望$V^\pi(s_t)$ 。

理解为，当前动作$a_t$的收益$G_t$扣除当前状态的平均收益$V^\pi(s_t)$，这样，增益更大的动作会得到相对更多的梯度更新，增大这类动作的概率。

于是梯度公式可以先写成：

$$
\nabla_\theta J
\approx
\sum_t
\left(
G_t-b(s_t)
\right)
\nabla_\theta
\log\pi_\theta(a_t|s_t)

$$


## 第11页，减去一个baseline为什么不改变梯度的期望

这个是通过数学推导证明出来的，对于梯度公式包含$b(s)$的部分：

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

因为 baseline $b(s)$ 不依赖采样出的 action，所以：

$$
\mathbb E_{a\sim\pi}
\left[
b(s)
\nabla_\theta
\log\pi_\theta(a|s)
\right]
$$

展开期望，并把$b(s)$提出，有：

$$
=
b(s)
\sum_a
\pi_\theta(a|s)
\nabla_\theta
\log\pi_\theta(a|s)
$$

仍然利用对数求导策略进行等式替换有：

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

所以，求导为0，所以得证：

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

## 第12页，总结

1.对数求导策略是什么？
答：针对一个对数求导的等式变换$\nabla \log p = \frac{\nabla p}{p}$ 。这里用来重新引入概率分布p，实现梯度表达式积分形式和期望形式的等效变换。这样才能使用Menta Carlo采样来近似求梯度。
2.REINFORCE 是怎么从 Policy Gradient 推出来的？
答：使用梯度求导策略，将梯度的积分表达式转换为期望表达式即可。
3.理解REINFORCE 到底是什么？
答：REINFORCE本质是Mente Carlo Policy Gradient

补充几个REINFORCE的重要知识点：
1.为什么实际 REINFORCE 使用 Return-to-Go更新策略？
答：最开始的梯度公式可以理解为使用整个轨迹$\tau$的奖励来奖励每一次动作$a_t$，但是明显每个动作的奖励应该不一样。所以需要设计一种机制，来度量一个动作的奖励。而一个动作的奖励体现在当前时间以及之后，所以需要设计一种机制，用来度量未来累计的奖励，所以这里采用未来回报奖励$G_t$。
2.为什么要减去一个baseline？
答：因为同一个状态$s_t$下，动作$a_t$的奖励可能基数比较大，动作之间的差异提现不出来。所以减去一个基数，能够扩大动作之前的相对奖励差异。这里一般减去当前状态$s_t$下的综合收益的期望$V^\pi(s_t)$。由于梯度公式里面还有额外乘一项对数概率的导数，所以减去一个值之后，一个动作带来梯度的变化**方差变小**。
3.减去baseline之后为什么梯度的期望不变。
答：因为含有$b(s_t)$这一项**与动作$a_t$无关**，对于一个动作而言，可以通过数学证明$b(s)\sum_a\nabla_\theta\pi_\theta(a|s)$的期望是0。