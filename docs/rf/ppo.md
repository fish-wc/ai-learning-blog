---
title: "PPO 学习笔记
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

# PPO 学习笔记

### 梯度策略在优化什么？
概述：Policy Gradient 优化的不是某一次 reward，而是[当前 Policy 所产生的 trajectory 的期望回报](/docs/rf/appendix/PPO/Policy_Gradient.md)，优化目标是**使得奖励高的轨迹发生的概率更大**，奖励reward只起到一个权重放缩的作用。

### 为什么要使用Log-Derivative Trick？

概述：对目标函数$J(\theta)$求导数的时候，使用对数导数的技巧，在数学表达式上可以重新引入轨迹$\tau$的概率表达式，进而可以[将积分形式的数学表达式等效为数学期望形式的表达式，进而可以使用蒙特卡洛模拟来进行近似求值](/docs/rf/appendix/PPO/Policy_Gradient.md)。

# REINFORCE 到底是什么？

概述：REINFORCE 本质上就是 Monte Carlo Policy Gradient。