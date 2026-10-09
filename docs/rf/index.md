---
order: 59
---

# 强化学习

这里记录从 Policy Gradient、REINFORCE 到 PPO 的学习过程。

对于强化学习的理解：智能体和马尔科夫决策过程进行交互的过程。

## 笔记列表

- [PPO 学习笔记（一）：Policy Gradient 到底在优化什么？](./ppo)
- [PPO 学习笔记（二）：Advantage 与 GAE —— 一场 Bias 和 Variance 之间的博弈](./advantage)
- [强化学习基础测试](./rf-learn)

> `docs/rf/` 是源码目录，网页链接不需要写 `docs/`，也不需要写 `.md` 后缀。


# 笔记目录

目录原则：按照递逻辑进关系进行排列。

## 明确Policy Gradient 到底在优化什么？

概述：**Policy Gradient** : Policy Gradient 优化的不是某一次 reward，而是**当前 Policy 所产生的 trajectory 的期望回报**。

[详细笔记](./appendix/PPO/Policy_Gradient.md)。[PPT制作脚本](./appendix/PPO/Policy_Gradient_script.md)。[ppt](./appendix/PPO/Policy_Gradient_到底在优化什么.html)。[PPT讲稿](./appendix/PPO/Policy_Gradient_讲稿.md)。

##  Reward 不可导，为什么 Policy 还能训练？

概述：**Policy Gradient 不是在对 Reward 求梯度，而是在对“产生不同 Reward 的概率分布”求梯度**。

[详细笔记](./appendix/PPO/Policy_Gradient.md)。[PPT制作脚本](./appendix/PPO/Policy_Gradient_script.md)。[ppt](./appendix/PPO/Policy_Gradient_到底在优化什么.html)。[PPT讲稿](./appendix/PPO/Policy_Gradient_讲稿.md)。