

# 22. 从 REINFORCE 到 Advantage


现在整个公式已经逐渐变成熟悉的样子：

最开始：

$$
R(\tau)
\nabla\log\pi
$$

然后变成 return-to-go：

$$
G_t
\nabla\log\pi
$$

加入 baseline：

$$
(G_t-b(s_t))
\nabla\log\pi
$$

如果：

$$
b(s_t)=V^\pi(s_t)
$$

那么：

$$
G_t-V^\pi(s_t)
$$

就是 advantage 的 Monte Carlo estimate。

于是：

$$
\boxed{
\nabla_\theta J
\approx
\sum_t
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

看到这个公式，PPO 已经离我们非常近了。

---

# 23. 用一个 Softmax Multi-Armed Bandit 了解一下 Policy Gradient 工作

假设四个 action 的真实 expected reward 分别是：

$$
[0.2,\;0.5,\;1.0,\;0.7]
$$

因此第三个 action 是最优动作。

但是 Agent 并不知道这些数字。

Policy 使用四个 logit：

$$
\theta_1,\theta_2,\theta_3,\theta_4
$$

通过 softmax 得到：

$$
\pi_\theta(a=i)
=
\frac{
e^{\theta_i}
}{
\sum_j e^{\theta_j}
}
$$

初始时：

$$
\theta=[0,0,0,0]
$$

因此：

$$
\pi=[0.25,0.25,0.25,0.25]
$$

然后不断：

```text
根据 Policy 采样 action
        ↓
Environment 返回 reward
        ↓
计算 REINFORCE gradient
        ↓
更新 θ
        ↓
重新计算 action probability
```

代码见[代码](./ppo/multi-armed%20bandit)。

<!-- 代码如下。

```python
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(0)

# 四个老虎机臂真实的 expected reward
# Agent 并不知道这些值，只能通过不断尝试获得 noisy reward
true_means = np.array([0.2, 0.5, 1.0, 0.7])

num_actions = len(true_means)

# Policy 参数：softmax logits
theta = np.zeros(num_actions)

learning_rate = 0.05

# running baseline
baseline = 0.0
baseline_lr = 0.05

steps = 3000

prob_history = []


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


for step in range(steps):

    # 1. 当前 Policy
    probs = softmax(theta)

    # 2. 根据 Policy 采样 action
    action = rng.choice(num_actions, p=probs)

    # 3. Environment 返回 noisy reward
    reward = rng.normal(
        loc=true_means[action],
        scale=1.0
    )

    # 4. 计算 grad log pi(a)
    #
    # 对 softmax：
    #
    # d log pi(a) / d theta_k
    # =
    # 1[k == a] - pi(k)
    #
    grad_log_pi = -probs.copy()
    grad_log_pi[action] += 1.0

    # 5. REINFORCE + baseline
    #
    # 注意：先使用“旧 baseline”计算 advantage，
    # 再使用当前 reward 更新 baseline。
    advantage = reward - baseline

    # Gradient Ascent
    theta += learning_rate * advantage * grad_log_pi

    # 6. 更新 running baseline
    baseline += baseline_lr * (reward - baseline)

    prob_history.append(softmax(theta))


prob_history = np.array(prob_history)

for action in range(num_actions):
    plt.plot(
        prob_history[:, action],
        label=f"Action {action}: mean={true_means[action]}"
    )

plt.xlabel("Training Step")
plt.ylabel("Action Probability")
plt.title("REINFORCE on a Softmax Multi-Armed Bandit")
plt.legend()
plt.show()
```

---

# 24. 这段代码最重要的不是结果，而是这一行

核心更新只有：

```python
theta += learning_rate * advantage * grad_log_pi
```

对应：

$$
\theta
\leftarrow
\theta
+
\alpha
(R-b)
\nabla_\theta
\log\pi_\theta(a)
$$

现在来看 softmax 的梯度：

$$
\frac{\partial\log\pi(a=i)}
{\partial\theta_k}
=
\mathbb 1[k=i]-\pi(k)
$$

如果：

$$
R-b>0
$$

说明：

> 这一次 action 的表现比平均水平好。

那么 chosen action 对应的 logit 会增加。

它的 probability 随之增加。

反过来，如果：

$$
R-b<0
$$

说明：

> 这个 action 比正常水平差。

那么它的 probability 会下降。

因此经过大量采样以后，第三个 action：

$$
\mu_3=1.0
$$

获得高 reward 的次数更多。

它得到正向 Policy Gradient 更新的概率也更高。

最终：

$$
\pi(a=2)
$$

会逐渐接近 1。

这就是第一次真正“看到”：

> **Policy Gradient 在工作。**

--- -->

# 25. 为什么一次 reward 高，不代表 action 一定会被永久加强？

简述：因为reward有噪音，导致单次采样梯度有噪音。

这里还有一个非常重要的统计视角。

假设：

```text
Action A true mean reward = 1
Action B true mean reward = 0
```

由于 reward 有噪声，某一次可能出现：

```text
A → -0.3
B → +2.1
```

那么这一次 REINFORCE 甚至会：

```text
降低 A 的概率
提高 B 的概率
```

是不是说明算法错了？

不是。

Policy Gradient 从来没有承诺：

> 每一次 gradient estimate 都是正确的。

它保证的是：

> **在 expectation 上，gradient estimator 指向提高 expected reward 的方向。**

因此，对于单次trajtory的梯度$\hat g$的期望而言，和目标函数期望一样：

$$
\mathbb E[\hat g]
=
\nabla_\theta J
$$

但不意味着，单词trajtory的梯度和目标函数的梯度一样：

$$
\hat g
\neq
\nabla_\theta J
$$

每一次采样都会有 noise。

这就是我们一直讨论 variance 的原因。

---

# 26. REINFORCE 最核心的两个性质

到这里，可以把 REINFORCE 理解成两个关键词：

## 第一：Unbiased

在相应假设下：

$$
\mathbb E[\hat g]
=
\nabla_\theta J
$$

平均意义上方向正确。

## 第二：High Variance

但是：

$$
\mathrm{Var}(\hat g)
$$

可能非常大。

所以实际强化学习的发展史，很大一部分其实都在解决：

> **怎样在尽量保持正确优化方向的同时，把 Policy Gradient 的 variance 降下来？**

Baseline 是其中最基础的一步。

Actor-Critic 又进一步用 learned value function 来提供更好的学习信号。

后面的：

- Advantage；
- TD；
- GAE；
- TRPO；
- PPO；

都可以沿着这条主线继续理解。

---

# 27. 后序PPO的学习思路

这一部分不是 Sutton 1999 的内容，而是为了帮助建立后续 PPO 的知识连接。

我们现在已经得到 Policy Gradient 的核心形式：

$$
\boxed{
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
}
$$

PPO 并没有推翻这个东西。

PPO 仍然是 Policy Gradient。

它真正要解决的是另外一个问题：

> **如果我用同一批数据，把 Policy 一口气更新得太远怎么办？**

PPO 定义：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

然后构造 surrogate objective（代理目标）。

最简单的版本可以先理解成：

$$
L(\theta)
=
\mathbb E_t
[
r_t(\theta)\hat A_t
]
$$

对 $r_t$ 求梯度：

$$
\nabla_\theta r_t(\theta)
=
r_t(\theta)
\nabla_\theta
\log\pi_\theta(a_t|s_t)
$$

因此：

$$
\nabla_\theta L
=
\mathbb E_t
\left[
r_t(\theta)
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\right]
$$

刚开始更新时：

$$
\theta=\theta_{\text{old}}
$$

所以：

$$
r_t(\theta)=1
$$

于是又变回：

$$
\boxed{
\mathbb E_t
[
\hat A_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
]
}
$$

也就是我们今天学的 Policy Gradient。

PPO 后面的 clipping：

$$
\operatorname{clip}
(
r_t(\theta),
1-\epsilon,
1+\epsilon
)
$$

可以先粗略理解成：

> Policy Gradient 告诉我们**往哪里走**，PPO clipping 开始限制我们**一次不要走太远**。

所以学习顺序应该是：

```text
Expected Return
      ↓
Log-Derivative Trick
      ↓
Policy Gradient
      ↓
REINFORCE
      ↓
Baseline
      ↓
Advantage
      ↓
Actor-Critic
      ↓
GAE
      ↓
Importance Sampling / Probability Ratio
      ↓
PPO
```

如果前面的东西没懂，直接看最后的 clip objective，很容易感觉 PPO 是突然从天上掉下来的一个公式。

但顺着这条路线走，会发现 PPO 的每一个组件其实都有明确的来源。

---

# 28. 本次学习最重要的几个知识点巩固

## Reward 本身不可导，为什么 Policy 还能训练？

简述版：因为对于梯度策略的目标函数求导，不需要对Reward求导，Reward在这里仅仅只是体现为一个权重系数的作用，奖励高的轨迹出现的概率增大，让奖励低的轨迹出现的概率减小，以此为目标去更新参数$\theta$。

答：Policy Gradient 不需要对 Reward 本身求导。

我们的目标是：

$$
J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
[R(\tau)]
$$

Policy 参数 $\theta$ 改变的是 trajectory 的概率分布：

$$
p_\theta(\tau)
$$

而不是直接改变某一个已经发生的 reward。

利用 log-derivative trick：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
$$

可以得到：

$$
\nabla_\theta J
=
\mathbb E
[
R(\tau)
\nabla_\theta
\log p_\theta(\tau)
]
$$

因此 reward 只需要作为一个 scalar weight：

```text
高 reward trajectory
→ 提高其 log probability

低 reward trajectory
→ 降低其 log probability
```

整个过程完全不要求：

$$
\nabla R
$$

存在。

---

## 为什么梯度式子里面减去baseline不改变梯度更新的期望

简述版：因为baseline不依赖采样的action，可以单独提出来，和策略action的概率的梯度相乘之后的和，通过等式变换恒等为0。这里并不是指目标函数的期望不变，而是指的目标函数梯度的期望不变。

为什么：

$$
(R-b)
\nabla\log\pi
$$

不会改变 Policy Gradient 的 expectation？

因为：

$$
\mathbb E_{a\sim\pi}
[
b(s)\nabla\log\pi(a|s)
]
$$

等于：

$$
b(s)
\nabla
\sum_a\pi(a|s)
$$

而：

$$
\sum_a\pi(a|s)=1
$$

可以理解为，在状态s下采取所有策略的概率的和为1。所以：

$$
b(s)\nabla1=0
$$

---

## 那为什么 baseline 又能降低 variance？

简述版：一般baseline取的是当前状态state所能带来的期望收益，对于一个action带来的收益是建立在状态state的基础上带来的，减去state的期望收益，可以衡量出一个具体的action带来的相对收益。

因为它把：

```text
absolute return
```

转变成更有意义的：

```text
relative return
```

即：

> 当前 action 相比这个 state 下的正常表现到底好多少。



于是 Policy 不再单纯学习：

> “这个 action 得了 100 分。”

而是在学习：

> “在本来就能得 98 分的情况下，这个 action 多贡献了 2 分。”

后者显然是更干净的 credit assignment signal（信用分配信号）。

---

# 30. 最后，用一句话分别理解这些概念

**Policy Gradient**

> 改变 Policy 参数，让高回报行为出现的概率增加。

**Log-Derivative Trick**

> 把“概率的梯度”变成“概率 × log probability 的梯度”，从而把梯度写成可以采样估计的 expectation。

**REINFORCE**

> 用 Monte Carlo return 对 Policy Gradient 进行采样估计。

**Baseline**

> 不改变期望梯度，只重新定义“什么叫表现好”，从而降低 gradient estimator 的 variance。

**Advantage**

> 当前 action 相比这个 state 下的正常表现到底好多少。

**PPO**

> 仍然沿着 Policy Gradient 的方向优化，但对每次 Policy 改变的幅度加以约束。

---

# 31. 真正需要建立的直觉

学完以后，我现在会把强化学习里的 Policy Gradient 想象成这样：

```text
Policy 生成行为
      ↓
Environment 只需要告诉我：
“这次结果有多好”
      ↓
我不需要知道 Environment 怎么求导
      ↓
只需要知道：
“这次行为是由我的 Policy 以多大概率生成的”
      ↓
Reward × ∇ log π
      ↓
调整 Policy 的概率分布
```

不需要知道：

> Reward 是怎样计算出来的。

只需要能够：

1. Sample；
2. Evaluate；
3. Compute log probability；
4. Backpropagate through Policy。

这也是为什么到了大语言模型时代，Policy Gradient 依然如此重要。

模型可以生成离散 token，

Environment 可以包含用户、规则、工具甚至整个真实世界，

Reward 可以来自一个完全不可微的过程，

但只要最终能得到一个 scalar feedback：

$$
R
$$

Policy Gradient 就提供了一条优化 Policy 的道路（标量反馈）。


# 参考文献

1. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. Advances in Neural Information Processing Systems 12.