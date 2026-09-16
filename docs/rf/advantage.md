---
title: "PPO 学习笔记（二）：Advantage 与 GAE —— 一场 Bias 和 Variance 之间的博弈"
date: 2026-09-15
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


# PPO 学习笔记（二）：Advantage 与 GAE —— 一场 Bias 和 Variance 之间的博弈

上一篇笔记里，我们从 Policy Gradient 的最底层公理出发，一路推到了：

\[
\boxed{
\nabla_\theta J
\approx
\sum_t
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
\]

也就是说，只要能给出一个合理的 \(\hat A_t\)（Advantage 的估计值），我们就能更新 Policy。

但是上次笔记里，\(\hat A_t\) 只是很粗糙地写成：

\[
G_t - V^\pi(s_t)
\]

也就是 Monte Carlo return 减掉一个 baseline。

这次要读的论文《High-Dimensional Continuous Control Using Generalized Advantage Estimation》（下面简称 GAE 论文），其实就是在回答一个非常具体的问题：

> \(\hat A_t\) 到底应该怎么估计，才能既不太"瞎猜"（bias 小），又不太"抖"（variance 小）？

今天不追求把论文里的每一个推导都抠一遍，只抓住核心思路：

1. Advantage 到底是什么；
2. TD residual \(\delta_t\) 是什么，为什么它本身就是一个 Advantage 的估计；
3. GAE 是怎么从"多个候选估计量"里，通过加权平均构造出来的；
4. 为什么这一整套东西的本质是 **bias-variance tradeoff**；
5. 自己写一个 `compute_gae()`。

---

## 1. 先回到 Advantage 的定义

上一篇笔记里已经埋下了这个概念，现在把它正式请出来。

定义 state-value function：

\[
V^\pi(s_t) := \mathbb{E}\left[\sum_{l=0}^{\infty} r_{t+l}\right]
\]

也就是说：

> 站在状态 \(s_t\)，如果接下来一直按照策略 \(\pi\) 走下去，平均能拿到多少总回报。

再定义 state-action value function：

\[
Q^\pi(s_t, a_t) := \mathbb{E}\left[\sum_{l=0}^{\infty} r_{t+l}\right]
\]

区别在于，\(Q^\pi\) 多"固定"了一件事：**在 \(s_t\) 这个状态下，第一步动作已经被指定为 \(a_t\) 了**，后面才继续按照 \(\pi\) 走。

于是 Advantage 的定义就非常自然：

\[
\boxed{
A^\pi(s_t, a_t) := Q^\pi(s_t, a_t) - V^\pi(s_t)
}
\]

翻译成人话：

> 在状态 \(s_t\) 下，specifically 选择动作 \(a_t\)，比"随便按照策略 \(\pi\) 的平均水平"要好多少。

这其实就是上一篇笔记里 baseline 那一节的自然延伸——只不过上次我们说的 baseline 是"随便选一个不依赖 action 的函数 \(b(s)\)"，而这里直接把最标准的选择钉死成了 \(V^\pi(s)\)，于是 \(G_t - b(s_t)\) 就升级成了一个有名字的东西：Advantage。

**为什么 Advantage 是最自然的学习信号？**

GAE 论文里有一句话说得很清楚：Policy Gradient 的方向，本质上应该是"提高比平均水平好的动作的概率，降低比平均水平差的动作的概率"。而 Advantage 的正负号，恰好就是在回答"这个动作到底是不是比平均水平好"。所以理论上，把 \(\Psi_t\)（上次笔记里 REINFORCE 更新公式里 reward 那个位置）换成 \(A^\pi(s_t,a_t)\)，是**方差最小的选择之一**。

问题来了：\(A^\pi(s_t,a_t)\) 我们根本不知道，因为它依赖真实的 \(Q^\pi\) 和 \(V^\pi\)，而这两个东西本身就需要被估计。

这篇论文剩下的所有内容，其实都是在回答一句话：

> 我们手头没有真正的 Advantage，只能用各种方式去**近似**它，那么应该怎么近似，才能兼顾"准不准"（bias）和"稳不稳"（variance）？

---

## 2. 第一个关键工具：TD residual \(\delta_t\)

假设我们已经训练出了一个近似的 value function \(V\)（不一定等于真实的 \(V^\pi\)），定义 TD residual：

\[
\boxed{
\delta_t := r_t + \gamma V(s_{t+1}) - V(s_t)
}
\]

这个式子看着眼熟，其实我们在做监督学习/强化学习基础的时候大概率见过它，它就是 TD(0) 更新里的那个误差项。

**但这里有一个更有意思的视角**：如果 \(V\) 恰好就是真实的 \(V^{\pi,\gamma}\)（论文里引入了 discount \(\gamma\) 之后带下标的版本，这里可以先不纠结记号细节，把它当作"精确的 value function"），那么：

\[
\mathbb{E}_{s_{t+1}}\left[\delta_t\right]
= \mathbb{E}_{s_{t+1}}\left[r_t + \gamma V^\pi(s_{t+1})\right] - V^\pi(s_t)
= Q^\pi(s_t,a_t) - V^\pi(s_t)
= A^\pi(s_t,a_t)
\]

也就是说：

> **当 \(V\) 是准确的时候，单步的 TD residual \(\delta_t\)，它的期望恰好就等于真实的 Advantage。**

这是一个非常漂亮的结论。它意味着 \(\delta_t\) 本身就可以直接拿来当作 \(\hat A_t\) 用——这就是 GAE 家族里最简单的一个特例（后面会看到，这正是 \(\lambda = 0\) 的情况）。

但是这里藏着一个前提条件：**\(V = V^\pi\)**。而实际训练中，我们的 value function 网络几乎不可能完全准确。一旦 \(V\) 不准，\(\delta_t\) 就会带有 bias。

不过换来的好处是：\(\delta_t\) 只涉及一步的随机性（一个 reward + 一次 bootstrap），所以它的 variance 非常小。

到这里，我们已经隐约看到了那个贯穿全文的主题：

- 用真实的、完整的 Monte Carlo return 做 Advantage 估计 → variance 大，但如果做法正确 bias 小；
- 用单步的 TD residual 做 Advantage 估计 → variance 小，但依赖 \(V\) 的准确性，容易有 bias。

GAE 要做的事情，就是在这两个极端之间，找一个"可调节的中间地带"。

---

## 3. 从 1 步到 \(k\) 步：一族 Advantage 估计量

论文的做法非常巧妙：先不要满足于"只用一步 \(\delta_t\)"，而是把多个 \(\delta\) 累加起来，构造出一整个家族的估计量。

定义：

\[
\hat A_t^{(1)} := \delta_t = -V(s_t) + r_t + \gamma V(s_{t+1})
\]

\[
\hat A_t^{(2)} := \delta_t + \gamma\delta_{t+1} = -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})
\]

\[
\hat A_t^{(k)} := \sum_{l=0}^{k-1}\gamma^l\delta_{t+l} = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1}r_{t+k-1} + \gamma^k V(s_{t+k})
\]

这一串式子看起来复杂，但其实是一个望远镜求和（telescoping sum），展开之后会发现中间的 \(V\) 项全部抵消掉了，只留下"前 \(k\) 步真实 reward + 第 \(k\) 步之后用 \(V\) 做 bootstrap"。

**这其实就是我们熟悉的 \(n\)-step return，只不过又减去了一个 baseline \(V(s_t)\)。**

规律非常直观：

- \(k\) 越小：用真实 reward 的步数越少，主要依赖 \(V(s_{t+k})\) 这个"猜测"，如果 \(V\) 不准，bias 就大；但因为随机性来源少，variance 小。
- \(k\) 越大：用真实 reward 的步数越多，越不依赖 \(V\) 准不准；但是叠加了更多步的随机性，variance 越来越大。

当 \(k \to \infty\) 时：

\[
\hat A_t^{(\infty)} = \sum_{l=0}^{\infty}\gamma^l\delta_{t+l} = -V(s_t) + \sum_{l=0}^{\infty}\gamma^l r_{t+l}
\]

这正是我们上一篇笔记里最原始的写法：**Monte Carlo return 减掉 value function baseline**。它对 \(V\) 的准确性完全不敏感（不管 \(V\) 准不准，这个估计量在期望意义上都是对的），但 variance 也是全家族里最大的。

到这里，bias-variance tradeoff 已经不再是一句抽象的口号，而是变成了一个非常具体的旋钮：**这个旋钮就是 \(k\)。**

---

## 4. GAE：与其选一个 \(k\)，不如把所有 \(k\) 都要了

选一个固定的 \(k\) 总归是不优雅的——为什么必须是 5 步而不是 6 步？

论文的处理方式是：**对所有 \(k\) 的估计量做一个指数加权平均**，权重由一个新引入的参数 \(\lambda \in [0,1]\) 控制：

\[
\hat A_t^{GAE(\gamma,\lambda)} := (1-\lambda)\left(\hat A_t^{(1)} + \lambda \hat A_t^{(2)} + \lambda^2 \hat A_t^{(3)} + \cdots\right)
\]

这一步和 TD(\(\lambda\)) 里那个经典的"资格迹（eligibility trace）"思想是同一个套路：不是非黑即白地选一步或者全部步，而是让近的步数权重大，远的步数权重按 \(\lambda\) 指数衰减。

把这个式子按 \(\delta\) 重新整理（论文里给出了完整展开，本质上是等比数列求和），最后会神奇地化简成一个极其简洁的形式：

\[
\boxed{
\hat A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
}
\]

这就是整篇论文最核心的一行公式。看起来简单到有点不可思议——它不过是把 TD residual \(\delta_t\) 按照 \((\gamma\lambda)^l\) 做了一个几何衰减求和，但它背后其实是刚才那一整族 \(\hat A_t^{(k)}\) 的加权平均结果。

**两个特殊情况值得单独看一眼：**

\[
GAE(\gamma, 0): \quad \hat A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

也就是只用最短的一步——**低方差，但只要 \(V\) 不准就会有 bias**。

\[
GAE(\gamma, 1): \quad \hat A_t = \sum_{l=0}^{\infty}\gamma^l\delta_{t+l} = \sum_{l=0}^{\infty}\gamma^l r_{t+l} - V(s_t)
\]

也就是完整的 Monte Carlo return 减 baseline——**不管 \(V\) 准不准都不会引入额外 bias，但方差全家族最大**。

而 \(0 < \lambda < 1\) 时，我们就得到了一个"介于两者之间"的估计量，用一个连续的旋钮，在 bias 和 variance 之间连续地滑动。

**这也是为什么这篇笔记开头我反复强调那一句话**：今天真正要理解的不是公式怎么推出来的，而是——

> \(\lambda\) 这个参数，本质上就是在"要不要相信我手头这个可能不准的 value function \(V\)"和"要不要忍受 Monte Carlo return 的巨大方差"之间做一个连续的权衡。

---

## 5. \(\gamma\) 和 \(\lambda\)：两个旋钮，分工不同

论文里特意强调了一点，容易被初学者混在一起：\(\gamma\) 和 \(\lambda\) 虽然都出现在同一个公式里，但它们做的事情并不一样。

- \(\gamma\)：本质上决定了 value function \(V^{\pi,\gamma}\) 这个"目标"本身的尺度——即使 \(V\) 学得完全准确，只要 \(\gamma < 1\)，就已经在系统性地忽略"太遥远的未来"，这本身就会给 Policy Gradient 引入 bias（哪怕不涉及任何估计误差）。
- \(\lambda\)：只有当 \(V\) 本身不准确的时候，\(\lambda < 1\) 才会引入额外的 bias；如果 \(V\) 恰好是精确的，那么不管 \(\lambda\) 取多少，GAE 都是无偏的。

论文的实验（cart-pole、3D 双足/四足机器人行走）里也验证了这一点：最优的 \(\lambda\) 往往比最优的 \(\gamma\) 要小很多，直觉上可以理解为——\(\lambda\) 造成的偏差"更便宜"，所以可以放心地把它调得更激进一点来换取方差的下降。

---

## 6. 手写一个 `compute_gae()`

理解到这里，其实已经可以脱离公式，直接把它写成代码了。核心就是那个望远镜求和 \(\sum_l (\gamma\lambda)^l \delta_{t+l}\)，用**从后往前递推**的方式实现最高效：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: list[float]，长度为 T，第 t 个元素是 r_t
    values:  list[float]，长度为 T+1，第 t 个元素是 V(s_t)，
             values[T] 是最后一个 state 的 bootstrap 值（如果是终止状态，可以直接置 0）
    返回：
        advantages: list[float]，长度为 T，即每个时间步的 \hat A_t^{GAE}
        returns:    list[float]，长度为 T，advantages + values[:-1]，
                    可以直接用来做 value function 的回归目标
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0

    # 从最后一步往前递推：
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    # A_t = delta_t + (gamma * lam) * A_{t+1}
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns
````

**这里最关键的一行是**：

```python
gae = delta + gamma * lam * gae
```

它其实就是把 (\hat A_t^{GAE} = \delta_t + (\gamma\lambda)\hat A_{t+1}^{GAE}) 这个递推关系直接翻译成代码——这本质上是无穷级数 (\sum_l(\gamma\lambda)^l\delta_{t+l}) 的一个等价递推形式（因为 (\hat A_{t+1}^{GAE}=\sum_l(\gamma\lambda)^l\delta_{t+1+l})），所以完全不需要真的展开无穷项去求和，从后往前一次线性扫描就能算出所有时间步的 Advantage，这也是几乎所有 PPO 开源实现（比如 OpenAI Baselines、Stable-Baselines3）里 `compute_gae` 或 `compute_returns` 函数的标准写法。

简单验证一下退化情况：

* 令 `lam = 0`：`gae = delta`，也就是每一步都只用当前的 (\delta_t)，对应 (GAE(\gamma,0))；
* 令 `lam = 1`：递推变成 `gae = delta + gamma * gae`，展开之后就是完整的折扣 Monte Carlo return 减去 baseline，对应 (GAE(\gamma,1))。

和公式完全对得上。

---

## 7. 今天真正要带走的东西

如果只记一句话，那就是：

> **Advantage 告诉 Policy Gradient"这个动作比平均水平好多少"；但真实 Advantage 算不出来，只能用近似估计量代替，而所有近似估计量都要在"信任 value function 换低方差"和"信任真实采样换低偏差"之间做权衡，GAE 只是提供了一个连续调节这个权衡的旋钮 (\lambda)。**

回到我们上一篇笔记结尾埋下的伏笔：

[
\nabla_\theta J \approx \sum_t \hat A_t \nabla_\theta \log\pi_\theta(a_t|s_t)
]

这里的 (\hat A_t)，在真正的 PPO 实现里，几乎从来不是简单的 (G_t - V(s_t))，而是这次笔记里推出来的：

[
\hat A_t = \hat A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
]

至此，PPO 目标函数里那个 (\hat A_t) 的来龙去脉，就已经完整闭环了。下一篇笔记，就可以正式回到 PPO 本身——probability ratio (r_t(\theta)) 和 clip，这两个东西终于可以在一个扎实的地基上被理解，而不是死记硬背。

---

## 参考文献

1. Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. International Conference on Learning Representations (ICLR 2016). arXiv:1506.02438.
2. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. Advances in Neural Information Processing Systems 12.
3. Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning, 8, 229–256. DOI: 10.1007/BF00992696.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
5. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd Edition. MIT Press.
6. Konda, V. R., & Tsitsiklis, J. N. (2003). *On Actor-Critic Algorithms*. SIAM Journal on Control and Optimization, 42(4), 1143–1166.

```
