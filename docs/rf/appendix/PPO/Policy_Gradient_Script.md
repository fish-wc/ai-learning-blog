---
title: "PPO 学习笔记：Policy Gradient 到底在优化什么？"
date: 2026-10-08
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


# PPT 制作目录脚本

## 第1页：明确这份ppt的主旨是要讲清楚2个问题

#### 问题1：Policy Gradient策略梯度在优化什么？

#### 问题2：奖励函数与环境有关，不能求导，为什么还能通过梯度下降训练。

## 第2页：ppt导航栏写上，针对的是问题1


首先放置优化目标的公式，并附上说明，然后将后文推理需要用到的符号一一陈列，最后得出结论。

真正要优化的是：

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

然后得出主旨：Policy Gradient 优化的不是某一次 reward，而是**当前 Policy 所产生的 trajectory 的期望回报**。


## 第3页：针对问题2 ，需要进行公式推导，推导出求导与奖励函数无关，奖励函数只起到缩放作用。这里只陈列推导过程中会出现的必要符号。

#### 状态和动作的记录trajectory $\tau$。

一条 trajectory 可以写成：

$$
\tau =
(s_0,a_0,r_1,s_1,a_1,r_2,\cdots,s_T)
$$


#### 一条轨迹的概率 $\pi_\theta(\tau)$，从概率的形式被记为$p_\theta(\tau)$

$\pi_\theta(\tau)$，即$p_\theta(\tau)$，即一条轨迹$\tau$发生的概率，即在当前状态$a_t$下进入状态$s_t$的概率 乘以 在$a_t$和$s_t$下进入状态$s_{t+1}$的概率，从初始状态$s_0$开始累乘。

这里认为是马尔科夫过程，即当前状态仅与上一个状态相关，所以有如下轨迹概率的式子：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
$$

## 第4页，开始推导目标函数 $J(\theta)$

#### 原始目标函数：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
[R(\tau)]
$$

注意，之前写法是轨迹$\tau$满足概率分布$\pi_\theta$，这里是一样的意思，只是把概率分布的写法从策略$\pi$的写法写成了概率的写法$p_\theta(\tau)$。

#### 需要将期望形式展开才能进行求导运算，这里认为是连续的概率分布，将期望展开的时候写成积分的写法。

写成积分：

$$
J(\theta)
=
\int
p_\theta(\tau)R(\tau)d\tau
$$

## 第5页，这里是具体求导过程

#### 需要对目标函数求导，求导变量是参数$\theta$，得到梯度，然后才能才能进行参数的梯度更新

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

注意这一行。式子里面根本没有出现对奖励函数的求导，因为奖励函数明显是与环境相关的，与梯度不相关。因此得到结论。**Policy Gradient 不是在对 Reward 求梯度，而是在对“产生不同 Reward 的概率分布”求梯度。**

## 第6页，总结

#### 针对问题1：Policy Gradient策略梯度在优化什么？

简述：Policy Gradient 优化的不是某一次 reward，而是**当前 Policy 所产生的 trajectory 的期望回报**。

#### 针对问题2：奖励函数与环境有关，不能求导，为什么还能通过梯度下降训练。

简述：**Policy Gradient 不是在对 Reward 求梯度，而是在对“产生不同 Reward 的概率分布”求梯度。**

结论：策略梯度目标，**让高 Reward trajectory 更容易发生，让低 Reward trajectory 更不容易发生。**