---
order: 59
---

# 强化学习

这里记录从 Policy Gradient、REINFORCE 到 PPO 的学习过程。

对于强化学习的理解：智能体和马尔科夫决策过程进行交互的过程。


# 笔记目录 - 近端策略估计

目录原则：按照递逻辑进关系进行排列。

## Policy Gradient
[详细笔记](./appendix/PPO/Policy_Gradient.md)。[PPT制作脚本](./appendix/PPO/Policy_Gradient_script.md)。[ppt](./appendix/PPO/Policy_Gradient_到底在优化什么.html)。[PPT讲稿](./appendix/PPO/Policy_Gradient_讲稿.md)。

1.明确Policy Gradient 到底在优化什么？
答：**Policy Gradient** : Policy Gradient 优化的不是某一次 reward，而是**当前 Policy 所产生的 trajectory 的期望回报**。
2.Reward 不可导，为什么 Policy 还能训练？
答：**Policy Gradient 不是在对 Reward 求梯度，而是在对“产生不同 Reward 的概率分布”求梯度**。

## REINFORCE
[详细笔记](./appendix/PPO/REINFORCE.md)。[PPT制作脚本](./appendix/PPO/REINFORCE_script.md)。[ppt](./appendix/PPO/REINFORCE.html)。

概述：REINFORCE 本质是基于蒙特卡洛采样对梯度求近似值。
说明：为什么可以基于蒙特卡洛采样对梯度求近似，需要从策略梯度进行一系列公式推导。详细推导见[详细笔记](./appendix/PPO/REINFORCE.md)。
需要理解的几个知识点：

1.对数求导策略是什么？
答：针对一个对数求导的等式变换$\nabla \log p = \frac{\nabla p}{p}$ 。这里用来重新引入概率分布p，实现梯度表达式积分形式和期望形式的等效变换。这样才能使用Menta Carlo采样来近似求梯度。
2.REINFORCE 是怎么从 Policy Gradient 推出来的？
答：使用梯度求导策略，将梯度的积分表达式转换为期望表达式即可。
3.理解REINFORCE 到底是什么？
答：REINFORCE本质是Mente Carlo Policy Gradient

4.为什么实际 REINFORCE 使用 Return-to-Go更新策略？
答：最开始的梯度公式可以理解为使用整个轨迹$\tau$的奖励来奖励每一次动作$a_t$，但是明显每个动作的奖励应该不一样。所以需要设计一种机制，来度量一个动作的奖励。而一个动作的奖励体现在当前时间以及之后，所以需要设计一种机制，用来度量未来累计的奖励，所以这里采用未来回报奖励$G_t$。
5.为什么要减去一个baseline？
答：因为同一个状态$s_t$下，动作$a_t$的奖励可能基数比较大，动作之间的差异提现不出来。所以减去一个基数，能够扩大动作之前的相对奖励差异。这里一般减去当前状态$s_t$下的综合收益的期望$V^\pi(s_t)$。由于梯度公式里面还有额外乘一项对数概率的导数，所以减去一个值之后，一个动作带来梯度的变化**方差变小**。
6.减去baseline之后为什么梯度的期望不变。
答：因为含有$b(s_t)$这一项**与动作$a_t$无关**，对于一个动作而言，可以通过数学证明$b(s)\sum_a\nabla_\theta\pi_\theta(a|s)$的期望是0。


# 笔记目录 - 基础模型

## transformer