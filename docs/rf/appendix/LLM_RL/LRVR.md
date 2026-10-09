# RLVR 真的让 LLM 学会了新的推理能力吗？——从 pass@1 与 pass@k 重新理解 Reasoning

> **LLM Reasoning 学习笔记 · Day 14**
>
> 核心关键词：RLVR、Base Model、pass@1、pass@k、Sampling Efficiency、Reasoning Capacity、Distillation
>
> 论文：*Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*
>
> 会议：NeurIPS 2025 Main Conference（Best Paper Runner-Up）
>
> 原文：[NeurIPS 官方论文](https://papers.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html) | [PDF](https://papers.neurips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf) | [项目主页](https://limit-of-rlvr.github.io/)

---

## 一、引言：RL 训练后数学题做得更好了，就代表模型学会了新的推理能力吗？

近年来，随着 DeepSeek-R1 等推理模型的出现，强化学习（Reinforcement Learning，RL）逐渐成为提升大语言模型推理表现的重要方法。

特别是 RLVR（Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习），在数学、代码等能够自动判断答案正确性的任务上取得了非常显著的效果。

一个常见的训练过程是：

```text
Pretraining
    │
    ▼
Base Model
    │
    ▼
RLVR Training
    │
    ▼
Reasoning Model
    │
    ▼
更高的数学 / 代码 Benchmark 分数
```

看到模型训练后的表现，我们很容易形成这样的理解：

> Base Model 原本不会解决某些问题，通过 RLVR 训练，模型学会了新的推理方法，因此推理能力得到了提升。

但这里隐藏了一个非常关键的问题：

**模型更容易回答正确，和模型获得原本不存在的推理能力，真的是一回事吗？**

2025 年，Yang Yue 等人在论文 *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 中，对这一假设提出了挑战。

论文的核心发现是：

* RLVR 通常能显著提高模型一次采样得到正确答案的概率。
* 但当允许模型进行大量采样时，Base Model 在不少实验中反而能解决更多不同的问题。
* 这些现象表明，当前一些 RLVR 方法可能主要是在放大 Base Model 已经能够生成的正确推理路径，而不一定创造了全新的推理模式。

这并不意味着 RLVR 没有价值。

恰恰相反，**RLVR 可以让模型从“偶尔能做对”，变成“经常能做对”。**

但是，我们需要重新思考：

> 我们所说的推理能力提升，究竟指的是正确答案的生成概率提升，还是模型能够解决的问题范围扩大？

要理解这个问题，首先必须理解两个指标：

$$
\boxed{\mathrm{pass@1}\quad vs\quad \mathrm{pass@k}}
$$

---

## 二、理解 pass@1：模型一次回答正确的概率

### 2.1 LLM 不是只会生成一条固定答案

假设我们给模型一道数学题：

$$
x^2-5x+6=0
$$

模型可能生成不同的回答：

```text
题目：x² - 5x + 6 = 0

Path A:
因式分解为 (x-2)(x-3)=0
答案为 x=2 或 x=3
✅ 正确

Path B:
因式分解时出现符号错误
得到 x=-2 或 x=-3
❌ 错误

Path C:
使用求根公式
得到 x=2 或 x=3
✅ 正确

Path D:
中间计算错误
得到错误答案
❌ 错误
```

这是因为语言模型学习的是一个条件概率分布：

$$
\pi_\theta(y|x)
$$

其中：

* \(x\)：输入的问题（Prompt）
* \(y\)：模型生成的完整回答，包括可能的推理过程
* \(\theta\)：模型参数
* \(\pi_\theta(y|x)\)：给定问题 \(x\) 时，生成回答 \(y\) 的概率

不同的推理路径具有不同的生成概率。

例如：

| 推理路径   | 是否正确 | 采样概率 |
| ------ | ---- | ---- |
| Path A | ✅    | 5%   |
| Path B | ❌    | 50%  |
| Path C | ✅    | 5%   |
| Path D | ❌    | 40%  |

这是一个人为构造的概率分布，仅用于解释概念。

从这个分布中采样一次，得到正确答案的概率为：

$$
p=5\%+5\%=10\%
$$

这里有一个很容易混淆的地方。

当我们说：

> 模型对这道题的正确率只有 10%。

并不代表模型脑海中只掌握了这道题 10% 的知识。

更准确的理解应该是：

> **在当前 Prompt、采样策略以及模型参数下，随机生成一次回答，有 10% 的概率得到正确结果。**

模型并非完全没有正确的推理路径，只是正确路径被采样到的概率比较低。

### 2.2 pass@1 的含义

pass@1 指的是：

**对于一道题，只从模型采样一个答案，判断它是否正确。**

如果某道题单次采样正确的概率为 \(p\)，那么：

$$
\mathrm{pass@1}(x)=p
$$

对于包含 \(N\) 道题的 Benchmark，可以理解为：

$$
\mathrm{pass@1}
=
\frac{1}{N}
\sum_{i=1}^{N}p_i
$$

其中 \(p_i\) 表示第 \(i\) 道题单次采样正确的概率。

因此，pass@1 衡量的是模型在指定采样设置下的**平均单次作答表现**。

实际使用中，这个指标非常重要。

因为大部分用户不会为了一个普通问题，让模型独立生成几百个答案，再从中挑选一个正确答案。

用户更关心：

> 我问一次，模型能否直接给出一个可靠答案？

需要注意的是，pass@1 通常对应随机采样下的单次成功率，并不必然等于使用 Greedy Decoding（贪心解码）生成的确定性答案准确率。

---

## 三、理解 pass@k：给模型更多次机会，会发生什么？

pass@k 与 pass@1 最大的区别在于：

**我们允许模型对同一道题独立采样 \(k\) 次，只要其中至少一个答案正确，就认为这道题通过。**

假设让模型回答五次：

```text
Sample 1  ❌
Sample 2  ❌
Sample 3  ❌
Sample 4  ✅
Sample 5  ❌
```

那么：

$$
\mathrm{pass@5}=1
$$

对于这道题来说，虽然模型只有一次回答正确，但我们已经证明：

在这五次尝试中，模型确实生成过一个正确答案。

### 3.1 pass@k 的数学推导

假设对于同一道题：

* 每次采样相互独立
* 模型参数与采样条件保持不变
* 单次采样正确的概率为 \(p\)

那么：

一次失败的概率：

$$
P(\text{失败})=1-p
$$

连续 \(k\) 次全部失败的概率：

$$
P(\text{全部失败})=(1-p)^k
$$

因此，至少有一次成功的概率为：

$$
\boxed{
\mathrm{pass@k}(x)=1-(1-p)^k
}
$$

这是理解整篇论文最基础的公式。

假设单次正确概率只有：

$$
p=0.1
$$

那么：

| 采样次数     | 至少一次成功的概率 |
| -------- | --------- |
| pass@1   | 10%       |
| pass@10  | 65.13%    |
| pass@100 | 99.997%   |

观察这个结果。

同一个模型，完全不需要修改参数，仅仅通过增加采样次数，就能够显著提高“至少找到一个正确答案”的概率。

这说明：

**单次回答经常出错，不代表模型无法生成正确答案。**

但是需要强调，这里计算的是“至少生成一次正确答案”的概率，不代表系统一定有能力从多个候选答案中识别出正确答案。

pass@k 通常隐含了一个能够验证候选答案正确性的评测条件。

### 3.2 从一道题扩展到整个 Benchmark

不同问题的难度不同，模型对每道题的单次成功概率也不同。

因此，整个数据集上的 pass@k 应该表示为：

$$
\boxed{
\mathrm{pass@k}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[1-(1-p_i)^k\right]
}
$$

这一点尤其重要。

我们不能简单地先计算整个数据集的平均正确率，再直接代入单道题的公式。

原因是：

$$
\frac{1}{N}\sum_i[1-(1-p_i)^k]
$$

通常不等于：

$$
1-\left(1-\frac{1}{N}\sum_i p_i\right)^k
$$

**pass@k 不仅取决于整体平均正确率，还取决于正确概率如何分布在不同题目上。**

而这正是后面理解 RLVR 争议的关键。

### 3.3 论文中实际如何估计 pass@k？

在真实实验中，我们并不知道每道题精确的 \(p_i\)。

所以论文沿用了代码生成评测中的无偏估计方法。

假设对一道题采样 \(n\) 次，其中有 \(c\) 个正确答案，并且 \(k\leq n\)：

$$
\boxed{
\widehat{\mathrm{pass@k}}
=
1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
}
$$

它的直觉是：

从 \(n\) 个采样结果中选出 \(k\) 个，计算这 \(k\) 个答案全部错误的概率，再用 1 减去它。

对不同题目分别估计，再求平均，就得到 Benchmark 上的 pass@k。

相比直接依靠很少量的重复实验，这种方式可以降低估计方差。

该方法来自 Chen 等人的 *Evaluating Large Language Models Trained on Code*，也是 HumanEval 等代码评测工作的重要基础。

---

## 四、整篇论文最核心的问题：pass@1 提升，代表 Reasoning Capacity 提升吗？

现在我们已经具备了理解论文的全部基础。

假设有两个模型：

* Base Model：预训练模型
* RLVR Model：从 Base Model 出发，经过 RLVR 训练得到的模型

在正常推理时：

$$
\mathrm{pass@1}_{RLVR}
>
\mathrm{pass@1}_{Base}
$$

我们可以合理地认为：

RLVR 让模型变得更加实用了。

但能不能进一步说：

> RLVR 创造了 Base Model 原本不具备的推理能力？

不能直接这么判断。

因为这涉及两个不同的概念。

### 4.1 Sampling Efficiency：采样效率

Sampling Efficiency 关心的是：

> 对于模型能够生成的正确答案，它有多大的概率把正确答案采样出来？

例如，同一道题：

$$
p_{Base}=0.01
$$

经过 RLVR 后：

$$
p_{RLVR}=0.5
$$

正确率从 1% 提高到了 50%。

即使正确推理方法在训练前已经存在，这种提升也非常有价值。

我们可以说：

**RLVR 提高了模型采样正确推理路径的效率。**

### 4.2 Reasoning Capacity：推理能力范围

Reasoning Capacity 则试图回答另一个问题：

> 模型到底具备解决哪些问题、生成哪些有效推理模式的能力？

这篇论文将大 \(k\) 下的 pass@k 作为推理能力边界（Reasoning Capacity Boundary）的经验近似。

思路是：

如果允许模型进行大量采样，那么一些原本概率很低的正确推理路径也有机会出现。

于是我们可以比较：

* Base Model 在大量尝试后能解决哪些问题？
* RLVR Model 在大量尝试后能解决哪些问题？
* RLVR 是否解决了 Base Model 始终无法解决的新问题？

这就不再只是比较正常情况下哪个模型更容易回答正确。

而是开始研究模型能够覆盖的可解问题集合。

### 4.3 为什么两个指标可能给出相反结论？

这一点特别容易产生误解。

对于**同一道题、同一个固定的单次成功概率 \(p\)**，随着 \(k\) 增大：

$$
1-(1-p)^k
$$

一定是单调不下降的。

而且，如果 RLVR 对每一道题都提高了成功概率：

$$
p_i^{RLVR}\geq p_i^{Base}
$$

那么 RLVR 的 pass@k 不可能被 Base Model 反超。

所以，如果我们观察到两条曲线发生反超，就意味着 RLVR 并不是对所有问题的成功概率都进行了统一提升。

它可能提高了部分问题的正确概率，同时压低了另一些问题的正确概率。

下面我们用一个完整例子来理解。

---

## 五、一个能真正解释论文现象的例子

假设 Benchmark 只有四道题：A、B、C、D。

### 5.1 训练之前：Base Model

Base Model 对四道题的单次成功概率分别是：

| 题目 | 单次成功概率 |
| -- | ------ |
| A  | 2%     |
| B  | 2%     |
| C  | 2%     |
| D  | 2%     |

它的特点是：

每道题都不容易做对，但每道题都有一定概率找到正确答案。

所以：

$$
\mathrm{pass@1}_{Base}=2\%
$$

### 5.2 训练之后：RLVR Model

假设 RLVR 训练改变了概率分布：

| 题目 | Base Model | RLVR Model |
| -- | ---------- | ---------- |
| A  | 2%         | 60%        |
| B  | 2%         | 60%        |
| C  | 2%         | 0.01%      |
| D  | 2%         | 0.01%      |

RLVR 大幅强化了 A、B 两道题上的有效推理行为。

但与此同时，C、D 两道题的正确路径变得非常难以采样。

这时：

$$
\mathrm{pass@1}_{RLVR}=30.005\%
$$

它的单次正确率显著高于 Base Model。

如果我们只看 pass@1，一定会认为 RLVR 取得了巨大的成功。

### 5.3 现在增加采样次数

使用前面推导的公式：

$$
\mathrm{pass@k}
=
\frac{1}{4}
\sum_{i=1}^{4}[1-(1-p_i)^k]
$$

得到以下结果：

| 指标       | Base Model | RLVR Model  |
| -------- | ---------- | ----------- |
| pass@1   | 2.00%      | **30.005%** |
| pass@10  | 18.29%     | **50.04%**  |
| pass@100 | **86.74%** | 50.50%      |

这里的数字是人为设计的示例，并非论文实验结果。

发生了什么？

当 \(k=1\) 时：

RLVR 显著占优，因为它能够非常高效地解决 A、B 两道题。

但当 \(k=100\) 时：

Base Model 在四道题上都已经有较高概率至少找到一个正确答案。

RLVR Model 却仍然很难采样出 C、D 两道题的正确路径。

因此，Base Model 的 pass@100 反而更高。

这意味着：

**提高平均正确率，不代表扩大了可解决问题的范围。**

### 5.4 用 Python 亲自验证

以下代码不依赖第三方库，可以直接运行：

```python
def pass_at_k(probs, k):
    if k < 1 or not probs or any(p < 0 or p > 1 for p in probs):
        raise ValueError("Invalid inputs")
    return sum(1 - (1 - p) ** k for p in probs) / len(probs)

base = [0.02, 0.02, 0.02, 0.02]
rlvr = [0.6, 0.6, 0.0001, 0.0001]

for k in (1, 10, 100):
    print(
        f"k={k:3d} | "
        f"Base={pass_at_k(base, k):.2%} | "
        f"RLVR={pass_at_k(rlvr, k):.2%}"
    )
```

输出：

```text
k=  1 | Base=2.00% | RLVR=30.00%
k= 10 | Base=18.29% | RLVR=50.04%
k=100 | Base=86.74% | RLVR=50.50%
```

需要注意：

在这个示例中，RLVR 对 C、D 的正确概率并不是严格的零。

因此，如果采样次数无限增加，两个模型最终都可能以接近 100% 的概率覆盖四道题。

这也是为什么我们不能把有限 \(k\) 下的 pass@k 直接理解成模型在数学意义上的绝对能力边界。

但在有限的计算资源下，两者所能有效覆盖的问题集合可以有很大差异。

**这正是论文希望研究的现象。**

---

## 六、论文到底做了什么实验？

理解前面的例子之后，再来看论文实验就容易多了。

研究者没有只比较 Base Model 与 RLVR Model 的一次作答正确率。

他们把 \(k\) 从较小的值逐步增加到几十、几百，部分实验甚至达到 1024 次采样。

他们希望通过大量采样，观察模型能够解决的问题覆盖范围。

### 6.1 实验覆盖了哪些任务？

论文覆盖了三类重要任务：

| 任务类别 | 代表模型                  | Benchmark                             |
| ---- | --------------------- | ------------------------------------- |
| 数学推理 | Qwen2.5、LLaMA 3.1 等   | GSM8K、MATH500、Minerva、AIME24、Olympiad |
| 代码生成 | CodeR1、DeepCoder 相关模型 | LiveCodeBench、HumanEval+              |
| 视觉推理 | Qwen2.5-VL            | MathVista、MathVision                  |

在数学实验中，作者主要比较从预训练 Base Model 直接进行 RLVR 训练得到的模型，即 Zero-RL 设置。

而在代码与视觉推理实验中，部分模型使用已经经过指令微调的模型作为起点。

这里的 Base Model 应理解为**对应 RLVR 训练之前的起始模型**，并不意味着每一组实验的起点都是纯预训练模型。

此外，作者也进行了多种 RLVR 算法的对比，包括 PPO、GRPO、Reinforce++、RLOO、ReMax、DAPO。

这一设计的意义是：

研究者希望观察到的是一种相对普遍的训练现象，而不是某个特定模型或者某个 RL 算法的偶然结果。

### 6.2 发现一：小 k 时，RLVR 普遍更强

当 \(k\) 比较小时，特别是 \(k=1\) 时：

$$
\mathrm{pass@1}_{RLVR}
>
\mathrm{pass@1}_{Base}
$$

这一现象并不令人意外。

RLVR 正是通过奖励信号，提高模型生成正确结果的倾向。

因此，训练后的模型通常能够更频繁地采样到正确路径。

这说明 RLVR 在改善模型实际作答表现方面确实有效。

### 6.3 发现二：k 增大以后，Base Model 开始追上甚至反超

论文 Figure 2 展示了多个数学任务上的 pass@k 曲线。

总体趋势是：

```text
Small k:

RLVR Model > Base Model

          │
          │ 增加采样次数
          ▼

Large k:

Base Model ≥ RLVR Model
（论文所测试的多组实验中）
```

也就是说：

RLVR Model 虽然更容易在少数几次尝试中生成正确答案，但 Base Model 在大量采样后，往往可以覆盖更广泛的问题。

例如，作者报告，在 Minerva Benchmark 的一组 32B 模型实验中，当 \(k=128\) 时，Base Model 的 pass@k 比 RLVR Model 高约 9 个百分点。

这说明，即便 RLVR 让模型整体更容易回答正确，也不意味着它能够解决更多种类的难题。

不过，需要特别强调：

**论文的结论对应其测试的模型、任务、训练配置和采样预算，不应直接推广为所有 RL 训练都具有这个结果。**

### 6.4 发现三：部分问题在 RLVR 后反而更难被解决

作者进一步比较了 Base Model 与 RLVR Model 所能解决的问题集合。

下面的数据来自论文 Table 2：

| 问题类别             | AIME24（k=1024） | MATH500（k=128） |
| ---------------- | -------------: | -------------: |
| Base 和 RLVR 都能解决 |          63.3% |          92.4% |
| 只有 Base 能解决      |          13.3% |           3.6% |
| 只有 RLVR 能解决      |           0.0% |           1.0% |
| 两者都不能解决          |          23.3% |           3.0% |

这里的“能解决”是指在给定采样预算内，至少出现一个通过验证的答案。

这组数据特别值得关注。

以 AIME24 为例：

在这次实验中，有一部分题目，Base Model 通过大量采样能够找到正确答案，而 RLVR Model 没有找到。

反过来，RLVR Model 没有覆盖 Base Model 无法覆盖的新题目。

这与前面四道题的例子非常相似。

作者据此认为：

**在这些实验中，RLVR 更像是改善了已有可解问题的采样效率，而没有扩大可解问题的覆盖范围。**

### 6.5 发现四：随着 RL 继续训练，覆盖范围可能进一步收窄

论文还分析了不同训练阶段的模型。

一个值得注意的现象是：

$$
\text{RL Training Steps}\uparrow
$$

对应：

$$
\mathrm{pass@1}\uparrow
$$

但部分实验同时出现：

$$
\mathrm{pass@256}\downarrow
$$

这说明，随着模型越来越擅长生成能够获得奖励的答案，它在有限采样预算内探索其他有效推理路径的能力，反而可能下降。

这一结果引出了一个非常重要的问题：

> 强化学习是否在提高模型的平均正确率时，也牺牲了部分推理路径的多样性？

---

## 七、为什么 RLVR 可能导致这样的结果？

要理解这个现象，需要回到 RLVR 的训练目标。

### 7.1 RLVR 优化的是什么？

在 RLVR 中，模型会根据输入问题生成完整回答。

然后 Verifier（验证器）判断答案是否正确。

对于最简单的二元奖励：

$$
r(x,y)=
\begin{cases}
1,& \text{答案正确}\\
0,& \text{答案错误}
\end{cases}
$$

模型希望最大化期望奖励：

$$
\boxed{
J(\theta)
=
\mathbb E_{x\sim D}
\left[
\mathbb E_{y\sim\pi_\theta(\cdot|x)}
[r(x,y)]
\right]
}
$$

其中：

* \(D\) 是训练问题的分布
* \(\pi_\theta\) 是模型的生成策略
* \(r(x,y)\) 是对生成回答的奖励

这个目标意味着：

模型应该尽可能提高得到正奖励回答的概率。

但是，需要注意：

**奖励通常判断的是答案是否正确，而不一定检查每个中间推理步骤是否逻辑严密。**

例如，一道数学题可能存在多种推导方式。

只要最后得到的答案正确，它就可能得到正奖励。

### 7.2 RLVR 可能主要是在重新分配概率

考虑下面这个抽象例子。

训练之前：

```text
Base Model

错误路径 A：45%
错误路径 B：35%
正确路径 C：10%
正确路径 D：10%
```

RLVR 训练之后：

```text
RLVR Model

错误路径 A：10%
错误路径 B：10%
正确路径 C：75%
正确路径 D：5%
```

这里的数字仅用于说明可能的分布变化。

RLVR 显著提高了正确路径 C 的概率。

但正确路径 D 的概率反而降低。

虽然正确回答的总概率提高了，但部分正确推理路径变得更难采样。

从单次成功率来看，这是巨大的进步。

从探索不同推理路径的角度来看，却可能出现多样性下降。

因此，这篇论文提出了一个非常有启发性的理解方式：

$$
\boxed{
\text{RLVR}
\approx
\text{Reward-guided Probability Reshaping}
}
$$

即：

**RLVR 的一个重要作用，是根据奖励信号重新塑造模型的输出概率分布。**

它可以提高高奖励路径的概率，使正确结果更容易被采样到。

但是，这种训练不保证每一条有价值的推理路径都会获得更高的概率。

### 7.3 这和 Mode Collapse 有什么关系？

我们可以从 Mode Collapse（模式坍缩）的角度理解这种现象。

当模型越来越偏向某些高奖励输出时，其他输出模式可能受到抑制。

例如：

```text
训练之前：

Path A ███████
Path B ██████
Path C █████
Path D ████
Path E ███

训练之后：

Path A ███████████████████
Path B ███████
Path C █
Path D ▏
Path E ▏
```

这不代表所有 RLVR 训练都会发生严重的 Mode Collapse。

但论文的实验表明：

**当前某些 RLVR 配置确实可能出现采样效率提升与推理覆盖范围收缩并存的现象。**

从优化角度看，这并不矛盾。

因为最大化期望奖励和保持探索多样性，本身就不是完全相同的目标。

---

## 八、论文如何进一步验证“正确路径可能早就存在于 Base Model”？

只看 pass@k 曲线还不够。

曲线反超能够说明采样分布发生了变化，但不能直接证明 RLVR 生成的推理路径都来自 Base Model。

所以作者进行了进一步分析。

### 8.1 Solvable-Problem Coverage Analysis

第一种方法是比较两种模型实际解决的问题集合。

可以将问题分为四类：

```text
                    RLVR

                 成功      失败
              ┌────────┬────────┐
Base   成功   │ 都成功  │ Base独有│
              ├────────┼────────┤
       失败   │ RL独有  │ 都失败  │
              └────────┴────────┘
```

如果 RLVR 真正扩大了模型的可解问题覆盖范围，我们希望看到足够多的 RL 独有成功案例。

但论文中的若干实验发现：

Base 独有成功案例比较多，而 RL 独有成功案例很少。

这说明在给定采样预算下：

$$
S_{RLVR}^{(k)}
\approx
S_{Base}^{(k)}
\text{ 的子集}
$$

这里：

$$
S^{(k)}
$$

表示模型在最多 \(k\) 次采样中实际覆盖的问题集合。

这是经验上的近似关系，而非对无限采样条件下集合包含关系的数学证明。

### 8.2 Perplexity Analysis：困惑度分析

作者还使用 Perplexity（PPL，困惑度）分析模型对推理路径的概率分配。

对于一段包含 \(T\) 个 Token 的回答 \(y\)，其困惑度可以写为：

$$
\mathrm{PPL}_{\pi}(y|x)
=
\exp\left(
-\frac{1}{T}
\sum_{t=1}^{T}
\log \pi(y_t|x,y_{<t})
\right)
$$

直观理解：

PPL 越低，代表模型平均而言越容易预测出这段文本中的 Token。

作者采用 Base Model 来评估 RLVR Model 生成的回答。

核心问题是：

> 如果把 RLVR 模型生成的推理轨迹交给 Base Model，Base Model 会不会认为这是一段非常陌生、极不可能生成的文本？

论文发现，RLVR 生成的一些推理轨迹在 Base Model 下同样具有较低的困惑度，并且训练推进过程中存在相应变化趋势。

这支持了一种解释：

这些推理路径并不是 RLVR 完全从零发现的，而可能已经处在 Base Model 相对容易到达的输出区域中。

不过要注意：

**低 PPL 只能说明 Base Model 对这些 Token 序列赋予了较高的平均预测概率，不能单独严格证明两种模型拥有完全相同的推理能力。**

因此，论文的论证依赖的是 pass@k、覆盖集合、PPL 和部分推理轨迹人工检查等证据的组合。

---

## 九、为什么 Distillation 反而可能扩大 Reasoning Boundary？

论文还讨论了一个很有价值的对照实验：

**RLVR 与 Distillation（知识蒸馏）有什么不同？**

### 9.1 RLVR：从自己的探索与奖励反馈中学习

典型流程：

```text
Student / Base Model
         │
         ▼
  自己生成 Rollout
         │
         ▼
    Verifier 打分
         │
         ▼
  根据 Reward 更新参数
         │
         ▼
       RLVR Model
```

这里的训练数据主要由当前模型自身生成。

虽然 Verifier 的正确性反馈也是外部信息，但模型不一定直接获得一条完整的、来自更强模型的正确推理示范。

因此，当 Base Model 极难探索到某种有效推理路径时，RLVR 可能很难从中获得足够的正奖励信号。

### 9.2 Distillation：从更强 Teacher 获取推理示范

Distillation 的流程则不同：

```text
Stronger Teacher Model
          │
          ▼
  生成高质量 Reasoning
          │
          ▼
    推理轨迹训练数据
          │
          ▼
     Student Model
          │
          ▼
   Distilled Model
```

例如：

Student Model 原本极难生成某种数学证明方式。

但 Teacher Model 能够生成完整的推导过程。

Student 可以通过监督学习，直接学习 Teacher 提供的示范。

这相当于给 Student 提供了额外的信息来源。

### 9.3 论文的对比实验

作者比较了 Qwen2.5-Math-7B 相关模型、RLVR 训练模型，以及 DeepSeek-R1-Distill-Qwen-7B。

论文 Figure 7 显示：

在对应实验中，蒸馏模型的 pass@k 曲线能够持续高于 Base Model。

作者据此认为，蒸馏有可能扩大模型的经验推理能力边界，而不仅仅是提高已有路径的采样效率。

因此，我们可以从信息来源的角度理解两者：

| 维度             | RLVR        | Distillation     |
| -------------- | ----------- | ---------------- |
| 主要训练信号         | 自身生成结果 + 奖励 | Teacher 提供的示范    |
| 是否需要更强 Teacher | 不一定         | 通常需要             |
| 主要优化方向         | 提高期望奖励      | 学习 Teacher 的输出分布 |
| 是否可能扩展推理覆盖     | 取决于探索与训练机制  | 可借助外部推理示范扩展      |
| 论文对应实验结果       | 主要提升采样效率    | 可提高大 k 下的覆盖      |

需要强调的是：

**这种差异并不能证明 RLVR 原理上永远无法产生新能力，也不能证明所有蒸馏过程都能扩展能力边界。**

它说明的是：外部推理示范与自身奖励反馈，可能以不同方式影响模型的学习过程。

---

## 十、如何批判性理解这篇论文？

我认为，真正理解这篇论文，不能只记住：

> RLVR does not create new reasoning capabilities.

更重要的是知道，这个结论成立需要哪些前提，以及它还有哪些值得讨论的地方。

### 10.1 large-k pass@k 不等于真正的能力上界

假设某道题正确路径的生成概率为：

$$
p=10^{-12}
$$

这个概率虽然极低，但并不严格等于零。

即使我们采样 1024 次，也几乎不可能得到正确答案。

因此：

$$
\mathrm{pass@1024}\approx0
$$

不能证明：

$$
p=0
$$

反过来，如果 \(p>0\)，从数学上看：

$$
\lim_{k\to\infty}[1-(1-p)^k]=1
$$

所以，如果把“能力存在”简单定义成某条有限推理路径具有非零生成概率，那么对于使用 Softmax 的语言模型，这个定义往往缺乏足够的区分度。

更合理的理解是：

> 大 k 下的 pass@k 衡量的是在特定计算预算、Prompt 和采样条件下，模型可以有效访问的可解问题范围。

这是一种操作性、经验性的能力定义。

而不是模型全部潜在推理能力的严格数学刻画。

### 10.2 正确答案不一定来自正确推理

这是第二个重要问题。

假设模型生成：

```text
第一步：错误计算
第二步：错误推导
第三步：碰巧得到正确答案
```

如果 Verifier 只检查最终答案，那么这个回答仍然可能得到正奖励。

也就是说：

$$
\text{Correct Final Answer}
\not\Rightarrow
\text{Correct Reasoning Process}
$$

这会对 pass@k 产生影响。

因为当 \(k\) 变得非常大时，模型可能通过错误推理、猜测或偶然计算得到正确结果。

原论文并没有完全忽视这个问题。

作者对部分困难数学题的 Chain-of-Thought 进行了人工检查，并在代码任务中利用测试验证结果。

不过，人工检查并不覆盖所有生成结果，因此仍然存在测量局限。

后续研究 *Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs* 进一步提出了 CoT-Pass@K，要求最终答案和推理过程都正确。

该研究报告，在其测试条件下，RLVR 可以提升 CoT-Pass@K，即使传统 pass@k 的结论不完全一致。

这表明：

**我们选择怎样定义和测量推理能力，会直接影响对 RLVR 的判断。**

### 10.3 其他研究也报告了不同结论

值得注意的是，另一篇 NeurIPS 2025 论文：

*ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models*

提出了不同的实验观察。

ProRL 通过延长 RL 训练、引入 KL 控制、参考策略重置及更多样的任务，报告了某些设置下 RL 模型在较大 k 的评测中仍然超过 Base Model 的现象。

这说明：

RLVR 能否扩展推理边界，可能与以下因素有关：

* 训练时间及计算资源
* 探索策略
* KL 约束与策略更新方式
* 任务分布与难度
* Base Model 本身的能力
* 对推理正确性的评估方法

因此，比较成熟的研究立场应该是：

> 2025 年这篇反方论文对当时常见 RLVR 设置提出了重要的经验性质疑，但它没有证明强化学习在所有条件下都不可能产生新的推理能力。

科学研究的价值，不只是给出一个结论。

更重要的是让我们知道：

**哪些假设需要被重新检验，以及什么实验能够区分不同解释。**

---

## 十一、重新理解 RLVR：它到底是在学习，还是在搜索？

现在我们可以回到最开始的问题。

传统的直觉可能是：

```text
Base Model 不会解题
         │
         ▼
        RLVR
         │
         ▼
模型学会全新的解题方法
```

但这篇论文提出了另一种可能：

```text
           Base Model
               │
               ▼
     已有多种潜在推理路径
               │
       ┌───────┴───────┐
       │               │
   高频率路径       低频率路径
       │               │
       └───────┬───────┘
               │
               ▼
     RLVR + Reward Signal
               │
               ▼
       重新分配生成概率
               │
       ┌───────┴───────┐
       │               │
 正确路径更易采样   某些路径更难采样
       │               │
       ▼               ▼
    pass@1 ↑      大 k 覆盖未必 ↑
```

可以将这类现象理解为：

$$
\boxed{
\text{Capability Elicitation}
}
$$

即：

**把模型已有但难以稳定表现出来的能力激发出来。**

而我们原本设想的：

$$
\boxed{
\text{Capability Acquisition}
}
$$

指的是模型通过训练获得此前无法有效实现的新能力。

两者并不是非此即彼的绝对分类。

一次 RL 训练可能同时包含已有行为的强化、对奖励反馈的学习，以及对新问题的泛化。

真正需要研究的是：

这些变化分别占了多大比重，以及是否出现了可验证的新推理策略。

---

## 十二、这篇论文对实际 LLM 工程有什么启发？

从工程角度来看，我们不能因为一篇论文质疑 RLVR 的能力扩展效果，就认为 RLVR 不值得使用。

事实上，采样效率提升本身就是巨大的价值。

### 12.1 对实际产品：pass@1 的价值依然非常高

假设模型需要回答用户的数学问题。

如果 Base Model 必须采样很多次才有机会找到正确答案，而 RLVR Model 单次采样就有较高正确率，那么 RLVR 可以帮助系统：

* 减少重复推理的成本
* 降低生成多个候选答案的需求
* 提高用户第一次获得正确答案的概率
* 改善有限延迟预算下的整体体验

当然，实际推理成本还取决于输出 Token 数、采样策略、验证器成本等因素，因此不能简单将采样次数减少直接等同于等比例的总成本下降。

### 12.2 对搜索型推理系统：需要关注多样性

对于数学搜索、程序合成、自动定理证明等任务，我们有时更关心：

> 在一定计算预算内，模型能不能探索到足够多不同的有效解法？

这时，除了 pass@1，还应该关注：

* 大 k 下的 pass@k
* 可解决问题覆盖率
* 不同推理路径的多样性
* 正确推理链的有效性
* 采样成本与答案筛选成本

一个 pass@1 很高的模型，不一定就是搜索场景中最好的候选生成模型。

### 12.3 对 RL 研究：探索机制比单纯提高奖励更值得关注

论文还引出了一个更深层的研究方向：

如果当前 RLVR 主要是在已有的高概率区域内强化正确答案，我们应该如何鼓励模型探索新的有效路径？

可能的方向包括：

* 更有效的 Exploration Strategy
* 保持多样性的 Policy Optimization
* 更长时间的 RL Training
* 使用 Teacher Distillation 改善初始分布
* 多轮 Agent-Environment Interaction
* 结合过程监督与结果监督

这些方向并不是本文已经证明有效的统一方案，而是从论文争议中自然延伸出的研究问题。

---

## 十三、面试中如何回答：Does RLVR Actually Improve Reasoning Ability?

如果面试官问：

> Does reinforcement learning with verifiable rewards actually improve the reasoning ability of LLMs?

一个比较普通的回答是：

> Yes. RLVR improves mathematical reasoning performance because models achieve higher benchmark scores after RL training.

这个回答并没有错，但它只讨论了模型性能。

我们可以给出一个更完整的回答。

### 推荐回答

> **I think we should distinguish reasoning performance from reasoning capacity.**
>
> RLVR clearly improves practical reasoning performance, especially pass@1, because it increases the probability of generating correct responses.
>
> However, a NeurIPS 2025 paper, *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, found that base models can match or even outperform their RL-trained counterparts when pass@k is evaluated with a large sampling budget.
>
> This suggests that current RLVR methods may primarily reweight existing reasoning trajectories rather than consistently expand the reasoning capabilities accessible from the base model.
>
> But I wouldn't conclude that RL can never produce new reasoning capabilities. Large-k pass@k is only an empirical proxy for reasoning coverage, and other research has reported different results under prolonged RL training or more refined reasoning metrics.
>
> Therefore, I would distinguish **capability elicitation from capability acquisition**, and investigate both sampling efficiency and reasoning coverage when evaluating RLVR.

### 这个回答体现了哪些理解？

首先，你知道 pass@1 和 pass@k 衡量的并不是完全相同的东西。

其次，你能够区分模型的实际表现与可访问的推理能力范围。

再次，你理解 RLVR 的概率重分配机制，以及它可能对探索多样性产生的影响。

最后，你能够正确看待论文的实证范围，而不是将特定实验结果绝对化。

这也是我认为阅读这篇论文最大的价值。

---

## 十四、为什么这篇论文会自然地把我们引向 Base Model？

读完这篇论文之后，一个新的问题会自然出现：

> 如果 RLVR 生成的许多有效推理模式，在训练前就已经可以由 Base Model 生成，那么这些推理模式最初是从哪里来的？

我们通常把大语言模型的训练过程理解为：

```text
Pretraining
    │
    ▼
Base Model
    │
    ▼
SFT / Distillation
    │
    ▼
RLVR
    │
    ▼
Reasoning Model
```

但是，这篇论文提醒我们：

不能把 Reasoning Model 最终表现出的全部能力，都简单归因于最后的 RLVR 阶段。

Base Model 通过大规模预训练，可能已经形成了许多有用的表示、知识结构以及推理行为。

这些能力在训练前不一定能够被稳定调用。

后训练阶段可能改变的是：

* 哪些行为更容易出现
* 哪些回答更符合目标
* 哪些推理路径更受奖励
* 模型能否更可靠地完成特定任务

因此，真正值得继续探索的问题是：

1. Base Model 为什么会具备潜在的推理能力？
2. Next-token Prediction 如何让模型学会复杂的推理结构？
3. Pretraining Data 与 Model Scale 如何影响推理能力的形成？
4. SFT、Distillation、RLVR 分别对模型能力产生了哪些影响？
5. 什么样的实验证据，才能证明模型真正获得了新的推理策略？

这些问题最终会引向 LLM 训练中更底层的主题：

**Pretraining、Representation Learning、In-context Learning，以及 Pretraining 与 Post-training 的能力分工。**

---

## 十五、总结：今天最值得记住的三个结论

### 结论一：pass@1 与 pass@k 需要分开理解

$$
\boxed{
\mathrm{pass@1}
\rightarrow
\text{Single-sample Performance}
}
$$

$$
\boxed{
\mathrm{pass@k}
\rightarrow
\text{Finite-budget Solution Coverage}
}
$$

pass@1 反映模型一次采样生成正确结果的表现。

大 k 下的 pass@k 则帮助我们观察模型在较多尝试下能够覆盖哪些问题。

后者并不是严格的无限采样能力上界。

### 结论二：提高正确答案的概率，不等于必然创造新的推理能力

$$
\boxed{
\text{Sampling Efficiency}\uparrow
\not\Rightarrow
\text{Reasoning Capacity}\uparrow
}
$$

RLVR 可能通过重新分配输出概率，提高已有正确推理路径的采样效率。

这种提升本身非常有价值，但不一定意味着可解问题范围扩大。

### 结论三：RLVR 的能力扩展问题仍然存在研究争议

这篇 NeurIPS 2025 论文揭示了当时常见 RLVR 设置下的重要限制。

但后续研究也提示我们，训练策略、探索预算和评测指标，都可能影响结论。

因此，与其简单地说“RL 不能创造新的推理能力”，不如记住：

**RLVR 的性能提升可以被确认，但其中有多少来自已有能力的激发，又有多少来自新能力的获得，需要更严格的实验来区分。**

最后，用一句话概括我对这篇论文的理解：

> **RLVR 的一个重要价值，是把 Base Model 原本偶尔能够做到的事情，变成更稳定、更容易做到的事情；但这种可靠性的提升，并不能自动证明模型的推理能力边界被扩展了。**

这也是从 RLVR 走向 Base Model 研究时，最值得带走的思考方式。

---

## 参考文献

**[1] Yue, Y., Chen, Z., Lu, R., et al. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? NeurIPS 2025.**

* 论文主页：https://papers.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html
* 论文 PDF：https://papers.neurips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf
* arXiv：https://arxiv.org/abs/2504.13837
* 项目主页：https://limit-of-rlvr.github.io/

本文主要参考论文第 2 节的评测指标、第 3 节的实验结果、第 4 节的机制分析，以及附录中的 pass@k 估计与推理轨迹检查。

**[2] NeurIPS (2025). Announcing the NeurIPS 2025 Best Paper Awards.**

* 官方公告：https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/

用于核实主论文获得 NeurIPS 2025 Best Paper Runner-Up 的信息及评奖委员会对论文贡献的评价。

**[3] Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code.**

* arXiv：https://arxiv.org/abs/2107.03374

介绍代码生成评测中的 pass@k 指标、无偏估计方法以及重复采样对模型性能评估的重要性。

**[4] Liu, M., Diao, S., Lu, X., et al. (2025). ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models. NeurIPS 2025.**

* 论文主页：https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html

讨论延长 RL 训练及改进训练策略之后，是否能够扩展模型在大 k 条件下的推理能力边界，为主论文提供不同的实证视角。

**[5] Wen, X., Liu, Z., Zheng, S., et al. (2025). Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs.**

* arXiv：https://arxiv.org/abs/2506.14245

提出 CoT-Pass@K 指标，强调不能只根据最终答案正确与否判断模型是否真正生成了有效的推理过程，并从这一角度重新审视 RLVR 的作用。

---

*本文是 LLM Reasoning 系统学习系列 Day 14 的阅读笔记，重点在于通过概率分布、采样指标和实验设计理解 RLVR 的作用与局限。文中的简化示例用于辅助推导，不代表论文原始实验数据。*
