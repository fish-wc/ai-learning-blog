---
title: "Advantage + GAE——策略梯度中的 Bias-Variance Tradeoff"
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


**Advantage 是“这一步到底比预期好多少” → TD residual 是“只往前看一步得到的惊喜程度” → GAE 是“把未来很多步的惊喜按照距离加权回来” → λ 决定你相信真实轨迹更多，还是相信 Critic 更多。**

---

# 1. Advantage 到底是什么？

概述：Advantage描述状态$s_t$条件下当前动作$a_t$相比于正常水平好多少。

先看公式：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

比如在玩一个游戏，现在角色站在状态 \(s\)。在这里，现在有很多可能的操作：

> 左走、右走、跳跃、攻击……

按照当前 policy \(\pi\) 正常操作下去，从这里出发，**平均预计最终能拿 100 分**。

那么：

$$
V^\pi(s)=100
$$

现在偏偏选择了动作：

$$
a=\text{向右跳}
$$

选择这个动作以后，再按照 policy 玩下去，预计能拿：

$$
Q^\pi(s,a)=130
$$

所以：

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
=
130-100
=
30
$$

这个 \(+30\) 的含义特别重要：

> **在当前这个状态下，做动作 \(a\)，比我平时预计的表现好 30。**

所以 Advantage 可以翻译成：

$$
\boxed{
Advantage = 实际选择这个动作有多好 - 在这里通常有多好
}
$$

因此：

* \(A(s,a)>0\)：这个动作比平均选择好；
* \(A(s,a)<0\)：这个动作比平均选择差；
* \(A(s,a)\approx0\)：这个动作没什么特别。

---

# 2. 为什么 Policy Gradient 需要 Advantage？

Policy Gradient 是在做：

$$
\nabla_\theta J(\theta)
\approx
\mathbb E[
\nabla_\theta\log\pi_\theta(a_t|s_t)
A_t
]
$$

先不要管严谨推导。

如果：

$$
A_t>0
$$

说明：

> “在 \(s_t\) 做 \(a_t\) 比预期好。”

那就：

> **提高以后选择 \(a_t\) 的概率。**

如果：

$$
A_t<0
$$

说明：

> “这个动作比预期差。”

那就：

> **降低以后选择它的概率。**

所以 Actor 真正需要的其实不是：

> “这局游戏最终拿了多少分？”

而更接近：

> **“刚才这个 action，到底比正常水平好多少？”**


---

# 3. 出现问题：根本不知道真正的 Advantage

概述：GAE 解决Advantage怎么估计的问题。

理论上：

$$
A(s_t,a_t)
=
Q(s_t,a_t)-V(s_t)
$$

但实际训练时，真正的 \(Q\) 不知道。

所以必须：

$$
\text{estimate } A_t
$$

于是问题从：

> Advantage 是什么？

变成：

> **怎样估计 Advantage？**

而这就是 GAE 真正解决的问题。

---

# 4. 第一种暴力办法：采样完整轨迹

比如：

```text
s0 --a0--> s1 --a1--> s2 --a2--> terminal
      r0         r1         r2
```

从 \(t\) 开始实际拿到的 discounted return：

$$
G_t
=
r_t+\gamma r_{t+1}
+\gamma^2r_{t+2}+\cdots
$$

那么自然可以写：

$$
\hat A_t
=
G_t-V(s_t)
$$

意思是：

> 我原本预计从这里能拿 \(V(s_t)\) 分，
> 最后实际上拿到了 \(G_t\) 分，
> 两者的差就是“惊喜程度”。

比如：

$$
V(s_t)=100
$$

最后实际：

$$
G_t=130
$$

那么：

$$
\hat A_t=30
$$

非常合理。

但它有一个大问题：

> variance 很大。

---

# 5. 为什么 Monte Carlo 的 variance 很大？

这是今天第一个真正需要建立直觉的地方。

假设同样在：

$$
(s_t,a_t)
$$

做完全一样的 action。

第一次 trajectory：

```text
a_t
 ↓
...
敌人突然左走
...
reward = 150
```

第二次：

```text
a_t
 ↓
...
敌人突然右走
...
reward = 70
```

第三次：

```text
a_t
 ↓
...
发生另一个随机事件
...
reward = 110
```

明明最开始：

$$
(s_t,a_t)
$$

完全一样。

但因为后面环境随机、policy 采样随机，最终：

$$
G_t
$$

可能变化巨大。

于是：

$$
G_t-V(s_t)
$$

也会非常抖。

也就是说，你在问：

> “\(a_t\) 到底是不是一个好动作？”

结果答案却受到**后面几十步随机事件**影响。

这就是 high variance。

---

# 6. 那能不能不要等那么久？

当然可以。

这就到了 TD。

我们已经训练了一个 Critic：

$$
V(s)
$$

它负责预测：

> “从这个状态继续走，未来大概还能拿多少 reward？”

那么站在 \(s_t\)，执行 \(a_t\)，得到：

$$
r_t
$$

来到：

$$
s_{t+1}
$$

我们就可以说：

> 我已经知道眼前拿了 \(r_t\)，至于后面几十步，不等它真的发生了，直接问 Critic。

于是：

$$
Q(s_t,a_t)
\approx
r_t+\gamma V(s_{t+1})
$$

所以：

$$
A(s_t,a_t)
=
Q(s_t,a_t)-V(s_t)
$$

自然得到：

$$
\boxed{
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
}
$$

这就是 TD residual。

---

# 7. TD residual 到底在说什么？

我非常建议你以后看到

$$
\delta_t
$$

脑子里不要读：

> temporal difference residual

而读：

> **“这一步发生之后，我有多惊喜？”**

因为：

$$
V(s_t)
$$

是：

> 行动之前，我预计未来值多少钱。

而：

$$
r_t+\gamma V(s_{t+1})
$$

是：

> 行动之后，根据刚发生的事情重新估计，现在看来值多少钱。

所以：

$$
\delta_t
=
\underbrace{r_t+\gamma V(s_{t+1})}_{行动之后重新估计}
-
\underbrace{V(s_t)}_{行动之前的预期}
$$

因此：

$$
\boxed{
\delta_t = 新信息到来以后，现实比原来的预期好多少
}
$$

这其实就是一个 **prediction error**。

---

# 8. 为什么 TD variance 小？

因为它没有等整个 trajectory。

Monte Carlo：

$$
r_t+\gamma r_{t+1}
+\gamma^2r_{t+2}
+\cdots
$$

后面所有随机性都进来了。

而 TD：

$$
r_t+\gamma V(s_{t+1})
$$

只观察一步，剩下的直接让 Critic：

$$
V(s_{t+1})
$$

概括掉。

于是随机性少很多。

所以：

$$
\boxed{\text{TD → lower variance}}
$$

但天下没有免费的午餐。

问题变成：

> **如果 Critic 自己就是错的呢？**

---

# 9. Bias 就从这里来了

假设真实情况：

$$
V^\pi(s_{t+1})=100
$$

但你的神经网络刚开始训练，非常菜：

$$
V_\phi(s_{t+1})=60
$$

那么：

$$
r_t+\gamma V_\phi(s_{t+1})
$$

当然也是偏的。

也就是说，我们为了避免等待真实未来，使用了一个**预测出来的未来**。

这种操作叫：

$$
\boxed{\text{bootstrapping}}
$$

核心就是：

> **用自己的估计继续构造新的估计。**

它减少了 variance。

代价是：

> Critic 估错了，会把错误带进 advantage。

于是增加 bias。

这就是你今天真正应该牢牢记住的一对关系：

$$
\boxed{
\text{更多 bootstrapping}
\Rightarrow
\text{更依赖 }V
\Rightarrow
\text{低 variance / 更可能有 bias}
}
$$

反过来：

$$
\boxed{
\text{更多真实 reward}
\Rightarrow
\text{更少依赖 }V
\Rightarrow
\text{低 bias / 高 variance}
}
$$

论文提出 GAE 的核心动机正是这种权衡。([arXiv][1])

---

# 10. 现在 GAE 就几乎呼之欲出了

现在我们手上有两个极端。

一个是：

$$
\delta_t
$$

只看一步。

稳定，但太相信 Critic。

另一个是：

$$
G_t-V(s_t)
$$

一直看到 episode 结束。

更相信真实 trajectory，但非常 noisy。

于是一个很自然的问题出现：

> **我为什么非得二选一？**

能不能：

> 看一点真实 future reward，然后适当相信 Critic？

当然可以。

于是有：

```text
1-step
2-step
3-step
4-step
...
Monte Carlo
```

比如：

$$
\hat A_t^{(1)}
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

再多看一步：

$$
\hat A_t^{(2)}
=
r_t+\gamma r_{t+1}
+\gamma^2V(s_{t+2})
-V(s_t)
$$

再多看一步：

$$
\hat A_t^{(3)}
=
r_t+\gamma r_{t+1}
+\gamma^2r_{t+2}
+\gamma^3V(s_{t+3})
-V(s_t)
$$

一直下去。

你会发现一个非常漂亮的连续变化：

```text
1-step ────────────────────────────── Monte Carlo
   ↑                                       ↑
更依赖 V                               更依赖真实 reward
   ↑                                       ↑
低 variance                            高 variance
高 bias                                低 bias
```

这才是 GAE 的灵魂。

---

# 11. λ 登场

GAE 的想法可以粗略理解成：

> **既然不知道到底应该看未来几步，那干脆把不同 horizon 的估计混起来。**

然后让：

$$
\lambda
$$

控制：

> 我愿意把多远的未来算进当前 advantage？

最后整理以后得到：

$$
\boxed{
A_t^{GAE}
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+\cdots
}
$$

也就是：

$$
\boxed{
A_t^{GAE}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
}
$$

今天真的不需要继续沉迷它是怎么严格推出来的。

你需要**看懂它在干什么。**

---

# 12. 把公式直接画成人话

假设：

$$
\gamma=0.99,\qquad \lambda=0.95
$$

那么：

$$
\gamma\lambda=0.9405
$$

GAE：

$$
A_t
=
\delta_t
+0.9405\delta_{t+1}
+0.9405^2\delta_{t+2}
+0.9405^3\delta_{t+3}
+\cdots
$$

也就是：

```text
现在的 TD error       ████████████████████
下一步 TD error       ███████████████████
再下一步              █████████████████
再下一步              ████████████████
再下一步              ██████████████
...
```

离现在越远：

$$
(\gamma\lambda)^l
$$

越小。

所以 GAE 实际上是在说：

> **未来发生的事情也可以解释今天这个 action 好不好，但是离今天越远，我越不确定它是不是应该归功于今天，所以权重逐渐降低。**

这个直觉非常重要。

---

# 13. λ = 0 会发生什么？

代进去：

$$
A_t^{GAE}
=
\delta_t
+
0\delta_{t+1}
+
0\delta_{t+2}
+\cdots
$$

所以：

$$
\boxed{A_t^{GAE}=\delta_t}
$$

完全变成 one-step TD。

也就是说：

> **我几乎完全相信 Critic，不想等待未来真实 trajectory。**

特点：

$$
\boxed{\text{low variance, potentially higher bias}}
$$

---

# 14. λ → 1 呢？

这时候：

$$
A_t^{GAE}
=
\delta_t
+
\gamma\delta_{t+1}
+
\gamma^2\delta_{t+2}
+\cdots
$$

把：

$$
\delta_t
=
r_t+\gamma V_{t+1}-V_t
$$

展开：

$$
\begin{aligned}
A_t
=&
(r_t+\gamma V_{t+1}-V_t)\\
&+\gamma(r_{t+1}+\gamma V_{t+2}-V_{t+1})\\
&+\gamma^2(r_{t+2}+\gamma V_{t+3}-V_{t+2})
+\cdots
\end{aligned}
$$

观察 \(V\)：

$$
+\gamma V_{t+1}
$$

下一项出现：

$$
-\gamma V_{t+1}
$$

抵消。

然后：

$$
+\gamma^2V_{t+2}
$$

又和后面的：

$$
-\gamma^2V_{t+2}
$$

抵消。

最后几乎全消掉。

剩：

$$
r_t+\gamma r_{t+1}
+\gamma^2r_{t+2}
+\cdots
-V(s_t)
$$

也就是：

$$
\boxed{
A_t
=
G_t-V(s_t)
}
$$

Monte Carlo advantage。

所以你突然就能看到：

$$
\lambda=0
$$

和：

$$
\lambda=1
$$

其实是同一条光谱的两端。这个端点关系也是理解 GAE 最有价值的方式之一。([Stanford University][2])

---

# 15. 所以 GAE 到底是什么？

如果面试官问：

> What is GAE?

我希望你第一反应**不是背公式**。

而是：

> GAE is a method for estimating the advantage by exponentially combining TD residuals, where \(\lambda\) controls how much we rely on bootstrapped value estimates versus sampled future rewards.

然后脑子里马上出现：

```text
λ = 0                                      λ = 1

TD(0) -------------------------------- Monte Carlo
  │                                           │
  │                                           │
more bootstrap                         less bootstrap
trust critic                           trust trajectory
  │                                           │
lower variance                         higher variance
potentially more bias                  potentially less bias
```

这就是今天真正需要带走的东西。

---

# 16. 有一个很容易被讲错的细节

这里我特意提醒你，因为你以后写博客最好不要简单写成：

> “TD 本身天然有 bias。”

更准确地说：

**如果 \(V\) 就是真实的 \(V^\pi\)，one-step TD residual 本身可以是正确的 advantage estimator。**

实践中的 bias 问题主要来自：

$$
V_\phi(s)\neq V^\pi(s)
$$

也就是 Critic 是一个函数逼近器。

所以真正的问题是：

> **你愿意多大程度相信一个不完美的 Critic？**

这比机械背：

> TD = high bias

理解得深得多。

而原 GAE 论文还讨论了 \(\gamma\) 本身对估计所引入的 bias，因此严格讨论时，\(\gamma\) 与 \(\lambda\) 的角色也不应该完全混为一谈。([DOI][3])

---

# 17. 最后再看 compute_gae()，你会发现它简单得离谱

数学公式：

$$
A_t
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+\cdots
$$

把后面括起来：

$$
A_t
=
\delta_t
+
\gamma\lambda
[
\delta_{t+1}
+\gamma\lambda\delta_{t+2}
+\cdots
]
$$

括号里面是什么？

就是：

$$
A_{t+1}
$$

所以：

$$
\boxed{
A_t
=
\delta_t
+
\gamma\lambda A_{t+1}
}
$$

这就是代码为什么要**倒着算**。

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = [0.0] * T

    gae = 0.0

    for t in reversed(range(T)):
        delta = (
            rewards[t]
            + gamma * values[t + 1]
            - values[t]
        )

        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages
```

现在不要只是看代码。

逐行翻译：

```python
delta = rewards[t] + gamma * values[t + 1] - values[t]
```

就是：

> **这一步现实比 Critic 原来的预期好多少？**

然后：

```python
gae = delta + gamma * lam * gae
```

就是：

> **当前惊喜 + 一部分未来惊喜。**

而：

```python
lam
```

控制：

> **未来那些惊喜，我到底愿意让它们影响今天多少？**

到这里，你其实已经理解 GAE 的核心了。

真实实现还需要处理 episode terminal / truncation，例如：

```python
non_terminal = 1.0 - done[t]

delta = (
    reward[t]
    + gamma * value[t + 1] * non_terminal
    - value[t]
)

gae = (
    delta
    + gamma * lam * non_terminal * gae
)
```

不过我建议你**第一遍先写上面的最小版本**，确认理解以后再处理 `done`、bootstrap value、batch/tensor shape。这会清楚很多。

---

# 18. 今天可以用一个问题检查自己到底懂没懂

假设我告诉你：

> Critic 非常非常准。

你应该马上想到：

> 那么我可以更放心地 bootstrap，较小的 \(\lambda\) 可能也能得到不错的 advantage estimation，而且 variance 更低。

反过来：

> Critic 非常垃圾。

你应该想到：

> 太依赖 \(V(s_{t+1})\) 会把 Critic 的错误带进 advantage，因此可能希望更多依赖真实 trajectory，但 variance 会增加。

如果这两个判断你已经可以**不看公式自己说出来**，那你今天最重要的目标已经完成了。

最后把整章压缩成一句适合放在 GitHub 博客开头的话：

$$
\boxed{
\text{GAE 的本质不是一个复杂公式，而是在“相信 Critic”与“相信采样轨迹”之间做连续权衡。}
}
$$

其中，**更相信 Critic → 更多 bootstrapping → 通常更低 variance，但更受 value approximation error 影响；更相信 trajectory → 更少 bootstrapping → 通常 bias 更小，但 variance 更高。** GAE 用 \(\lambda\) 把这两个极端连接起来。([arXiv][1])

你今天写博客时，我很建议就按 **「Advantage → 为什么难估计 → Monte Carlo → TD residual → bias/variance → GAE → 20 行实现」** 这个故事线写。它比照着论文公式顺序抄更能体现你是真的理解了。

[1]: https://arxiv.org/abs/1506.02438?utm_source=chatgpt.com "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
[2]: https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture12-reasoning-part1.pdf?utm_source=chatgpt.com "cs224n-2026-lecture12-reasoning-part1.pptx"
[3]: https://doi.org/10.48550/arxiv.1506.02438?utm_source=chatgpt.com "[1506.02438] High-Dimensional Continuous Control Using Generalized Advantage Estimation"


# 参考文献

[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438?utm_source=chatgpt.com)
