# Return-to-go（往回走的回报 / 后续回报，$G_t$）
一句话核心：
> $G_t$ = **从时刻 $t$ 往后，未来能拿到的所有折扣奖励之和**；
> 动作 $a_t$ **只能影响它之后的奖励，不能改变过去已经拿到的奖励**。

我们先对比原始错误版本，再看正确REINFORCE，看懂为什么要改成 return-to-go。

## 1. 原始朴素版本（不合理的版本）
原始思路：整条轨迹总回报 $R(\tau) = r_1+r_2+r_3+r_4$
梯度写成：
$$
\nabla_\theta J \approx \sum_{t=0}^{T-1} R(\tau)\cdot \nabla_\theta \log\pi(a_t|s_t)
$$
意思：**轨迹里每一个动作 $a_t$，都乘上整条轨迹的总奖励**。

拿你举的例子：
- $t=0$：$r_1$
- $t=1$：$r_2$
- $t=2$：动作 $a_2$
- $t=3$：$r_3$
- $t=4$：$r_4$

按这个朴素公式：
$a_2$ 的梯度项 = $(r_1+r_2+r_3+r_4)\cdot \nabla\log\pi(a_2|s_2)$

👉 问题来了：
$a_2$ 是**在 $t=2$ 才选出来的动作**。
$r_1,r_2$ 在 $t=0,t=1$ 就已经发生了，**$a_2$ 根本不可能影响之前已经拿到的奖励！**
把历史奖励 $r_1,r_2$ 算到 $a_2$ 的头上，属于错误归因（credit assignment 信用分配问题）。

> 类比：你打游戏，第5分钟做的操作，不该为前2分钟已经拿到/丢掉的分负责。

## 2. Return-to-go $G_t$（正确REINFORCE）
$$
\boldsymbol{G_t = \sum_{k=t}^{T-1} \gamma^{k-t}\, r_{k+1}}
$$
- $t$：当前动作时刻
- 求和从 $k=t$ 开始，**只包含 t 之后的奖励**
- $\gamma \in [0,1]$：折扣因子，越遥远未来的奖励权重越低；$\gamma=1$ 就是无折扣
- $G_t$：**从t时刻往后，剩下还能拿到的回报，中文叫「后续回报/待获回报」，英文 return-to-go**

套进你的例子（假设$\gamma=1$）：
- $G_0 = r_1+r_2+r_3+r_4$ （$t=0$动作，对全部奖励负责）
- $G_1 = r_2+r_3+r_4$ （$t=1$动作，只对$t=1$之后奖励负责）
- $G_2 = r_3+r_4$ （$t=2$动作$a_2$，**只负责 $r_3,r_4$，不再背 $r_1,r_2$ 的锅！**）
- $G_3 = r_4$
- $G_4 = 0$（最后一步没有后续奖励）

✅ 现在REINFORCE梯度公式：
$$
\nabla_\theta J \approx \sum_{t=0}^{T-1} G_t \cdot \nabla_\theta \log\pi(a_t|s_t)
$$

> 核心思想：Credit Assignment（信用分配）
> 动作 $a_t$ 只对**它之后发生的奖励**分配功劳/惩罚；过去的奖励和当前动作无关，剔除掉。

## 3. 简单推导（为什么可以这么替换）
目标函数：
$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} r_{t+1}\right]
$$
REINFORCE的核心等式（对数似然技巧，policy gradient trick）：
$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[
\left(\sum_{t=0}^{T-1} r_{t+1}\right)
\sum_{t=0}^{T-1}\nabla_\theta\log\pi(a_t|s_t)
\right]
$$
把两个求和展开，交换顺序：
$$
=\mathbb{E}\left[
\sum_{t=0}^{T-1} \nabla_\theta\log\pi(a_t|s_t) \sum_{k=0}^{T-1} r_{k+1}
\right]
$$
把内层求和拆成两块：$k<t$（过去奖励）和 $k\ge t$（未来奖励）
$$
=\mathbb{E}\left[
\sum_{t=0}^{T-1} \nabla_\theta\log\pi(a_t|s_t)
\left( \underbrace{\sum_{k=0}^{t-1}r_{k+1}}_{\text{过去，与}a_t\text{独立}}
+ \underbrace{\sum_{k=t}^{T-1}r_{k+1}}_{G_t,\text{未来回报}}
\right)
\right]
$$

关键一步期望化简：
$\sum_{k=0}^{t-1}r_{k+1}$ 是**t时刻之前已经发生的随机量，和 $a_t$ 独立**，期望等于0。
> 直观理解：过去奖励是已经确定的，$a_t$ 无法改变过去，所以这一项对更新没有贡献，可以直接扔掉。

于是只剩下：
$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[
\sum_{t=0}^{T-1} G_t \cdot \nabla_\theta\log\pi(a_t|s_t)
\right]
$$
这就是REINFORCE！

## 4. 直观对比
朴素版本：
$$
\sum_t (r_1+r_2+r_3+r_4)\nabla\log\pi(a_t|s_t)
$$
每个动作都背负全部奖励，归因错误，方差巨大。

REINFORCE return-to-go版本：
$$
\sum_t G_t \nabla\log\pi(a_t|s_t)
$$
每个动作只对**之后的奖励**负责，归因正确，方差显著下降。

## 5. 补充小细节
1. $\gamma$ 折扣因子
$G_t = r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\dots$
未来奖励打折扣，代表智能体更看重眼前奖励，远期奖励权重更低。

2. 为什么叫 return-to-go？
return = 回报；to-go = 剩下还没拿到的。直译：**剩下待获取的回报**。

3. 一个常见坑
$G_t$ 是**采样得到的轨迹样本估计值**，不是期望，所以policy gradient本身是蒙特卡洛估计，方差依然偏高，后面才有baseline（减去$V(s_t)$）来进一步压低方差：
$$
\nabla_\theta J \approx \sum_t \big(G_t - V(s_t)\big)\nabla_\theta\log\pi(a_t|s_t)
$$

---

你可以看看现在两点是否清楚：
1. return-to-go 的含义；
2. 推导里为什么「过去奖励项期望为0，可以删掉」。
如果这部分没问题，我们可以继续：REINFORCE加baseline，或者PPO里怎么用advantage（优势函数 $A_t=G_t-V(s_t)$）。