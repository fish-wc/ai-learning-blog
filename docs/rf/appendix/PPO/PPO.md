# 从 Policy Gradient 到 PPO：理解了 Clip 到底在 Clip 什么

> PPO（Proximal Policy Optimization，近端策略优化）是目前强化学习中应用最广泛的策略优化算法之一。
>
> 以前学习 PPO 时，我一直觉得它的核心公式：

\[
L^{clip}
=
\mathbb{E}
[
\min
(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
)
]
\]

像一个人为设计出来的复杂技巧。

但是当我真正理解：

- Advantage 决定动作应该增加还是降低概率；
- Ratio 衡量策略变化幅度；
- Clip 限制策略更新不要走得太远；
- Min 保证错误方向的更新仍然受到惩罚；

之后，PPO 的整个设计逻辑就变得非常自然。

本文记录我对 PPO-Clip 的理解过程。

---

# 1. 为什么需要 PPO？

在强化学习中，我们希望学习一个策略：

\[
\pi_\theta(a|s)
\]

其中：

- \(s\)：当前状态（state）
- \(a\)：选择的动作（action）
- \(\theta\)：神经网络参数

策略的目标是：

> 找到一个策略，使长期累计奖励最大。

最经典的方法是 Policy Gradient：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}
[
\nabla_\theta
\log \pi_\theta(a_t|s_t)
A_t
]
\]


其中：

\[
A_t
\]

叫做 Advantage（优势函数）。

它衡量：

> 当前动作相比平均水平到底好不好。

通常：

\[
A(s,a)=Q(s,a)-V(s)
\]

其中：

- \(Q(s,a)\)：执行动作后的价值
- \(V(s)\)：当前状态平均价值


因此：

如果：

\[
A_t>0
\]

说明：

> 这个动作比预期更好，未来应该提高选择概率。


如果：

\[
A_t<0
\]

说明：

> 这个动作表现较差，未来应该降低选择概率。


所以：

\[
\boxed{
A_t决定策略更新方向
}
\]

---

# 2. 传统 Policy Gradient 存在的问题

理论上：

如果一个动作很好：

\[
A_t>0
\]

那么我们不断增加：

\[
\pi_\theta(a_t|s_t)
\]

即可。


但是实际训练中存在一个问题：

## 策略更新可能过大


例如：

原策略：

\[
\pi_{old}(a|s)=0.1
\]


经过一次梯度更新：

变成：

\[
\pi_\theta(a|s)=0.9
\]


虽然这个动作可能确实很好，但是：

一次更新让概率从 10% 提升到 90%。


这会导致：

- 策略变化过快；
- 新策略偏离旧策略太远；
- 之前采样的数据失效；
- 训练不稳定。


因此 PPO 的核心思想：

> 策略可以更新，但是不要一次改变太多。

这也是 Proximal（近端）的含义。

---

# 3. PPO 中最重要的公式：Probability Ratio


PPO 引入：

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{old}}(a_t|s_t)
}
\]


这个公式表示：

> 当前新策略相比旧策略，对这个动作概率改变了多少。


例如：

旧策略：

\[
\pi_{\theta_{old}}(a|s)=0.5
\]


新策略：

\[
\pi_\theta(a|s)=0.6
\]


那么：

\[
r_t
=
\frac{0.6}{0.5}
=
1.2
\]


说明：

> 新策略让这个动作概率增加了 20%。


因此：

| Ratio | 含义 |
|-|-|
| \(r_t=1\) | 策略没有变化 |
| \(r_t>1\) | 更喜欢这个动作 |
| \(r_t<1\) | 降低这个动作概率 |

所以：

\[
\boxed{
r_t衡量策略改变程度
}
\]


---

# 4. 为什么 PPO 使用 \(r_tA_t\)

简述：\(r_t\) 用来做重要性采样，复用旧轨迹数据，配合优势 \(A_t\)，让模型增加好动作概率、减少坏动作概率。

如果暂时不考虑 Clip：

PPO 的目标：

\[
L(\theta)
=
\mathbb{E}[r_tA_t]
\]


我们分析一下这个公式。


---

## 情况1：优势为正

假设：

\[
A_t=2
\]


说明：

这个动作很好。


如果：

\[
r_t=1.2
\]


那么：

\[
r_tA_t
=
1.2\times2
=
2.4
\]


目标变大。


优化器会继续增加：

\[
r_t
\]


也就是：

增加：

\[
\pi_\theta(a_t|s_t)
\]


因此：

\[
\boxed{
A_t>0
\Rightarrow
增加动作概率
}
\]


---

## 情况2：优势为负


假设：

\[
A_t=-2
\]


说明：

这个动作不好。


如果：

\[
r_t=0.8
\]


那么：

\[
r_tA_t
=
0.8\times(-2)
=
-1.6
\]


相比：

\[
-2
\]

目标变大。


因为 PPO 最大化目标函数。


因此优化器会倾向：

降低这个动作概率。


即：

\[
\boxed{
A_t<0
\Rightarrow
降低动作概率
}
\]


所以：

\[
r_tA_t
\]

已经天然包含：

- 好动作增加概率；
- 坏动作降低概率。


那么为什么还需要 Clip？

---

# 5. PPO 为什么需要 Clip？

假设：

\[
A_t>0
\]


动作很好。


优化器发现：

提高概率能够增加目标。


于是：

\[
r_t
\]

不断增加。


例如：

\[
r_t=1.5
\]


那么：

\[
r_tA_t
\]

会继续变大。


优化器会认为：

> 增加越多越好。


但是：

一次更新过大，会导致：

- 新旧策略差异过大；
- 数据利用效率下降；
- 训练震荡。


所以 PPO 引入：

\[
clip(r_t,1-\epsilon,1+\epsilon)
\]


例如：

\[
\epsilon=0.2
\]


那么：

ratio 被限制：

\[
0.8\le r_t\le1.2
\]


---

# 6. PPO-Clip 完整目标函数


最终：

\[
L^{clip}
=
\mathbb E
[
\min
(
r_tA_t,
clip(r_t,1-\epsilon,1+\epsilon)A_t
)
]
\]


其中：

第一项：

\[
r_tA_t
\]


表示：

正常策略梯度更新。


第二项：

\[
clip(r_t)A_t
\]


表示：

限制策略变化范围。


而：

\[
min
\]

非常关键。


---

# 7. 四种情况理解 PPO


这是理解 PPO 最重要的一张表。

| Advantage | Ratio变化 | PPO行为 | 原因 |
|-|-|-|-|
| \(A>0\) | 增大 | 鼓励 | 好动作，提高概率 |
| \(A>0\) | 增大太多 | Clip | 防止策略改变过快 |
| \(A<0\) | 减小 | 鼓励 | 坏动作，降低概率 |
| \(A<0\) | 减小太多 | Clip | 防止概率下降过猛 |


下面逐个推导。

---

# 8. 情况一：\(A>0\)，Ratio 增大


假设：

\[
A=2
\]


\[
r=1.1
\]


因为：

\[
1.1<1.2
\]


没有触发 Clip。


目标：

\[
rA=1.1\times2=2.2
\]


比原来：

\[
2
\]

更大。


所以：

继续增加概率。


---

# 9. 情况二：\(A>0\)，Ratio 增大太多


假设：

\[
r=1.5
\]


那么：

未裁剪：

\[
1.5\times2=3
\]


裁剪：

\[
clip(1.5)=1.2
\]


得到：

\[
1.2\times2=2.4
\]


最终：

\[
min(3,2.4)
=
2.4
\]


发现：

继续增加 ratio：

\[
1.5\rightarrow2
\]


目标不会继续增加。


因此：

PPO 不再奖励。


这就是：

\[
\boxed{
A>0,r>1+\epsilon
\rightarrow Clip
}
\]


---

# 10. 情况三：\(A<0\)，Ratio降低


假设：

\[
A=-2
\]


动作不好。


如果：

\[
r=0.9
\]


那么：

\[
rA
=
0.9\times(-2)
=
-1.8
\]


相比：

\[
-2
\]


目标提高。


所以：

继续降低动作概率。


---

# 11. 情况四：\(A<0\)，Ratio降低太多


假设：

\[
r=0.5
\]


未裁剪：

\[
0.5\times(-2)
=
-1
\]


看起来非常好。


但是：

PPO认为：

> 你降低太多了。


于是：

\[
clip(0.5)=0.8
\]


得到：

\[
0.8\times(-2)
=
-1.6
\]


最终：

\[
min(-1,-1.6)
=
-1.6
\]


因此继续降低：

\[
r=0.5\rightarrow0.3
\]


不会继续获得收益。


---

# 12. Min 为什么如此重要？


很多人第一次学习 PPO 会误解：

> Clip 是把 ratio 强行限制在区间里面。


实际上不是。


PPO 并不会禁止：

\[
r>1+\epsilon
\]


或者：

\[
r<1-\epsilon
\]


它只是：

> 当策略朝正确方向变化太多时，不再给予额外奖励。


但是：

如果策略朝错误方向变化：

PPO 仍然保留惩罚。


因此：

\[
\boxed{
min保证PPO更加保守
}
\]


它只限制：

"好得过头"

而不会掩盖：

"坏的更新"

---

# 13. PPO 的一句话理解


如果把 PPO 公式翻译成人话：

\[
L^{clip}
=
min(rA,clip(r)A)
\]


就是：

> Advantage 告诉我应该喜欢还是讨厌这个动作；
>
> Ratio 告诉我现在喜欢程度改变了多少；
>
> Clip 防止我一次改变太激进；
>
> Min 保证错误方向仍然受到惩罚。


最终：

\[
\boxed{
PPO=
方向正确+
更新适度
}
\]


---

# 14. PPO代码中的对应关系


实际实现：

```python
ratio = torch.exp(
    new_log_prob - old_log_prob
)

surr1 = ratio * advantage


surr2 = torch.clamp(
    ratio,
    1-eps,
    1+eps
) * advantage


policy_loss = -torch.min(
    surr1,
    surr2
).mean()
````

为什么使用：

```python
exp(new_log_prob-old_log_prob)


因为：

$$
e^{log\pi_\theta-log\pi_{old}}
$$

根据：

$$
log(a)-log(b)
=
log(\frac ab)
$$

得到：

$$
=
\frac{\pi_\theta}{\pi_{old}}
$$

也就是：

$$
r_t
$$

```



---

# 参考文献

[1] Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O.
**Proximal Policy Optimization Algorithms**.
arXiv preprint arXiv:1707.06347, 2017.

[https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

[2] Sutton R S, Barto A G.
**Reinforcement Learning: An Introduction (2nd Edition)**.
MIT Press, 2018.

[http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

[3] Williams R J.
**Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning**.
Machine Learning, 1992, 8:229–256.

[4] OpenAI Spinning Up Documentation.
**Proximal Policy Optimization (PPO)**.

[https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

[5] Hugging Face Deep Reinforcement Learning Course.
**Proximal Policy Optimization (PPO)**.

[https://huggingface.co/learn/deep-rl-course](https://huggingface.co/learn/deep-rl-course)


