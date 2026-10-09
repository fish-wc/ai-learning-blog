# Day 20｜DoReMi：大模型预训练的数据配比，为什么不能拍脑袋？

> **学习主题**：LLM Pretraining · Data Mixture · Proxy Model · Excess Loss · Group DRO · Reweighting
>
> **核心问题**：有限的训练预算下，应该如何决定不同类型数据的采样比例？
>
> **论文**：[DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)（NeurIPS 2023）

## 1. 引言：数据越多，就应该训练得越多吗？

假设我们正在预训练一个大语言模型，手里有四类数据：

| Domain    | 数据类型 | 原始数据占比 |
| --------- | ---- | -----: |
| Web       | 网页文本 |    70% |
| Wikipedia | 百科知识 |     5% |
| Books     | 书籍   |    15% |
| Code      | 代码   |    10% |

现在有一个问题：

**如果总训练预算固定，我们应该按照什么比例使用这些数据？**

最直观的方案是按照原始数据量采样：

```text
Web        70%
Wikipedia   5%
Books      15%
Code       10%
```

但仔细想想，这个方案隐含了一个假设：

> 不同来源的数据，每个 token 对模型训练的价值都差不多。

这个假设未必成立。

例如：

* Web 数据量大，但可能存在大量重复、低信息密度或者噪声内容。
* Wikipedia 数据较少，但可能包含密度较高的事实性知识。
* Code 数据可能有助于模型学习程序结构和逻辑关系。
* Books 数据可能有助于模型学习长篇叙事和复杂语言结构。

这里并不是说 Wikipedia 一定比 Web 更有价值，而是说：**原始数据量并不能直接代表数据的训练收益。**

那么，我们是否可以手动将比例调整成：

```text
Web        40%
Wikipedia  20%
Books      20%
Code       20%
```

当然可以。

但新的问题又出现了：

**为什么是 40% 而不是 30%？为什么 Wikipedia 应该占 20%？这些比例有客观依据吗？**

如果依赖人工经验，就很难系统性地回答。

DoReMi 研究的正是这个问题：

> 如何在不知道具体下游任务的情况下，利用较小的模型自动寻找更合适的预训练数据配比，并将其迁移到大模型训练中？

这也是我理解 DoReMi 的第一条主线：

**数据配比优化，本质上是有限训练预算下的数据资源分配问题。**

---

## 2. 什么是 Domain Mixture？

理解 DoReMi 之前，需要先搞清楚三个概念：

* Domain
* Domain Mixture
* Sampling Weight

### 2.1 Domain：按照什么维度划分数据？

在语言模型预训练中，Domain 可以理解为具有某种共同属性的数据集合。

例如，按照数据来源进行划分：

```text
Pretraining Dataset
│
├── Web
├── Wikipedia
├── Books
└── Code
```

也可以按照内容类型划分：

```text
Training Dataset
│
├── Mathematics
├── Programming
├── Dialogue
└── Science
```

一个 Domain 不一定对应某种严格定义的能力。在 DoReMi 原论文中，Domain 主要按照数据来源进行划分。

需要注意，Domain 的划分粒度会影响最终的数据配比结果。

例如 Web 本身还可以细分为：

```text
Web
├── News
├── Forums
├── Tutorials
├── Blogs
└── Other
```

如果把所有网页文本放进同一个 Domain，那么 DoReMi 只能统一调整这个大类的权重，而不能直接识别其中哪些网页更值得训练。

### 2.2 Domain Mixture：训练数据的混合分布

假设有 \(K\) 个 Domain：

$$
D_1,D_2,\ldots,D_K
$$

为每个 Domain 分配一个权重：

$$
\alpha=(\alpha_1,\alpha_2,\ldots,\alpha_K)
$$

满足：

$$
\alpha_i\geq 0,\qquad \sum_{i=1}^{K}\alpha_i=1
$$

其中，\(\alpha_i\) 表示从第 \(i\) 个 Domain 中采样的概率。

例如：

$$
\alpha=(0.4,0.2,0.2,0.2)
$$

意味着：

* Web：40%
* Wikipedia：20%
* Books：20%
* Code：20%

从概率分布的角度，可以写为：

$$
P_\alpha(x)=\sum_{i=1}^{K}\alpha_iP_i(x)
$$

其中：

* \(P_i(x)\)：第 \(i\) 个 Domain 内部的数据分布。
* \(\alpha_i\)：第 \(i\) 个 Domain 的混合权重。
* \(P_\alpha(x)\)：最终的训练数据分布。

一个直观的理解是：

**先按照 \(\alpha\) 选择 Domain，再从该 Domain 中抽取训练数据。**

这里还有一个重要的工程细节：如果不同样本的长度差异很大，那么按照样本数量采样和按照 token 数量采样，最终获得的 token 配比并不一定相同。

因此，讨论 pretraining mixture 时，应明确采用什么统计单位。

### 2.3 Sampling Weight 为什么等价于分配训练预算？

假设我们固定训练 token 总量为 \(N\)，并按照 token 级别的 Domain Mixture 进行采样。

那么第 \(i\) 个 Domain 获得的期望训练 token 数为：

$$
N_i=N\alpha_i
$$

这意味着，修改 \(\alpha_i\) 就是在改变不同 Domain 获得的训练 token 预算。

在训练总量、batch size、模型结构等条件保持一致的情况下，不同 Domain 对优化目标和梯度更新的贡献也会随之发生变化。

因此，我更倾向于从下面这条链路理解 Domain Mixture：

```text
Domain Mixture
      ↓
Sampling Probability
      ↓
Training Token Budget
      ↓
Gradient Contribution
      ↓
Model Capability
```

这里需要强调：Sampling Weight 并不严格等于算力占比，也不严格等于梯度大小。

但在训练设置可比的前提下，它是控制不同数据来源参与优化的重要手段。

**所以，Domain Mixture 不是简单地“调整数据比例”，而是在控制优化器把有限的训练资源分配到哪里。**

---

## 3. 为什么不能直接根据 Loss 调整比例？

既然原始数据量不可靠，那么我们能不能根据模型的 Loss 来调整权重？

一个自然的想法是：

> 哪个 Domain 的 Loss 最大，说明模型在哪个 Domain 上表现最差，那就多训练哪个 Domain。

例如：

| Domain    | Training Loss |
| --------- | ------------: |
| Wikipedia |           2.0 |
| Books     |           3.0 |
| Web       |           6.0 |

如果只看 Loss，Web 显然最难。

于是我们可能决定大幅增加 Web 的采样权重。

但这个逻辑存在一个关键问题：

$$
\boxed{\text{High Loss}\neq\text{High Learning Value}}
$$

### 3.1 为什么 High Loss 不等于 High Value？

假设某个 Domain 包含大量随机字符、无意义文本或者高度不可预测的内容。

模型可能一直在这些数据上产生较高的 Loss。

原因未必是训练不充分，也可能是：

**这些数据本身具有较高的不可约不确定性。**

即使继续投入大量训练预算，模型也不一定能够显著降低 Loss。

从信息论角度来说，某些 Domain 的条件熵可能比较高，因此其理论上能够达到的最优预测损失也较高。

如果简单采用：

$$
\alpha_i\propto L_i
$$

就可能让高噪声、高熵的 Domain 长期占据大量训练预算。

与此同时，另一些当前 Loss 没有那么高、但仍然具有较大学习空间的 Domain，反而没有获得足够资源。

这就是只按照 Loss 选择数据容易出现的问题。

### 3.2 真正应该关注的是什么？

我认为应该区分三个概念：

**Difficulty：数据当前有多难？**

可以使用当前模型的 Loss 作为一种衡量信号。

**Learnability：这些数据能否被模型有效学习？**

一个 Domain 当前 Loss 高，不代表继续训练就一定能够降低。

**Marginal Utility：再增加训练预算能够带来多少收益？**

这才更接近我们真正想知道的东西。

可以抽象写成：

$$
\text{Marginal Utility}_i
\approx
\frac{\Delta \text{Performance}}{\Delta \text{Compute}_i}
$$

但实际训练中，要直接准确估计每个 Domain 的边际收益非常困难。

因为这可能需要在不同配比下反复训练模型，再观察整体性能变化。

DoReMi 提供了一种更便宜的替代思路：

**不直接估计真实的 Marginal Utility，而是利用 Reference Model 构造 Excess Loss，寻找当前模型相对于参考水平仍然存在较大学习差距的 Domain。**

---

## 4. DoReMi 的关键创新：Excess Loss

这是整篇论文最值得理解的部分。

### 4.1 为什么需要 Reference Model？

假设只使用一个模型，我们只能观察到：

$$
L_{\text{current}}(x)
$$

即模型在样本 \(x\) 上的 Loss。

但是，我们不知道这个 Loss 高究竟是因为：

1. 数据本身难以预测。
2. 模型目前还没有充分学会。

于是 DoReMi 引入一个 Reference Model，记为：

$$
p_{\mathrm{ref}}
$$

Reference Model 通过预先在某个初始数据配比上训练获得。

它不是一个能够给出理论最优 Loss 的 Oracle，而是提供一个实际可达到的参考水平。

接下来，再训练一个 Proxy Model：

$$
p_\theta
$$

让这两个模型在同一个样本上进行比较。

### 4.2 Excess Loss 是什么？

对于样本 \(x\)，定义：

$$
\boxed{
\Delta\ell(x)=
\ell_\theta(x)-\ell_{\mathrm{ref}}(x)
}
$$

其中：

* \(\ell_\theta(x)\)：Proxy Model 的 Loss。
* \(\ell_{\mathrm{ref}}(x)\)：Reference Model 的 Loss。
* \(\Delta\ell(x)\)：Excess Loss，即相对参考模型的损失差距。

如何理解这个公式？

可以把它理解成：

> 一个已经训练过的参考模型能够达到某种水平，而当前 Proxy Model 距离这个参考水平还有多远？

这和直接观察 Loss 的区别非常大。

### 4.3 用三个例子理解 Excess Loss

假设三个 Domain 的平均 token Loss 如下。

| Domain      | Proxy Loss | Reference Loss | Excess Loss |
| ----------- | ---------: | -------------: | ----------: |
| A：高熵数据      |        5.1 |            5.0 |         0.1 |
| B：简单数据      |        1.1 |            1.0 |         0.1 |
| C：尚未充分学习的数据 |        2.6 |            2.0 |         0.6 |

**Domain A：Loss 很高，但 Excess Loss 很低。**

$$
5.1-5.0=0.1
$$

Proxy Model 和 Reference Model 的 Loss 都很高。

这可能意味着该 Domain 本身就比较难预测，继续投入大量预算也未必带来相应收益。

不过，这只是一种可能的解释，并不能仅凭两个 Loss 就证明数据不可学习。

**Domain B：Loss 很低，Excess Loss 也很低。**

$$
1.1-1.0=0.1
$$

Proxy Model 已经接近 Reference Model 的表现。

该 Domain 当前没有表现出很大的相对学习差距。

**Domain C：Loss 中等，但 Excess Loss 很高。**

$$
2.6-2.0=0.6
$$

Reference Model 已经达到较低的 Loss，而 Proxy Model 仍然存在明显差距。

这说明至少存在一个参考模型，能够在该 Domain 上达到更好的预测水平。

因此，C 对 Proxy Model 而言存在更明显的学习空间。

这就是 Excess Loss 提供的额外信息。

### 4.4 Excess Loss 的真正含义

DoReMi 想寻找的，并不单纯是：

> 当前最难的数据。

而更接近：

> 相对于一个已训练参考模型，当前模型还有哪些 Domain 没有学到位？

我会用一句话概括：

$$
\boxed{\text{Learnable but not sufficiently learned}}
$$

即：

**具有可学习空间，但当前模型尚未充分学习的数据。**

不过，这里必须保留一个严格的边界：

Excess Loss 是一种相对学习差距的代理信号，而不是真实数据价值的精确度量。

高 Excess Loss 不一定意味着增加训练预算就能获得最大的下游收益。

它仍然受到 Reference Model、Proxy Model、训练过程和 Domain 划分方式的影响。

---

## 5. Group DRO：如何自动调整 Domain Weights？

现在我们已经知道如何计算 Excess Loss。

接下来的问题是：

**如何根据不同 Domain 的 Excess Loss 自动调整权重？**

DoReMi 使用的是 Group Distributionally Robust Optimization，简称 **Group DRO**。

### 5.1 为什么需要 Robust Optimization？

普通语言模型训练，通常希望最小化训练数据分布上的平均 Loss：

$$
\min_\theta \sum_{i=1}^{K}\alpha_iL_i(\theta)
$$

这意味着：

如果某个 Domain 的权重非常大，那么优化器就会更加关注这个 Domain。

但对于通用语言模型来说，我们通常不知道模型未来会面对什么样的下游任务。

某个下游任务可能更依赖代码，另一个可能更依赖百科知识。

因此，只优化某个预先指定的平均训练分布，不一定能够获得良好的跨 Domain 表现。

Group DRO 提供了另一种思路：

**不要只关心平均表现，还要关注表现最差的 Domain。**

DoReMi 进一步将这种最坏情况优化应用到 Excess Loss 上。

### 5.2 DoReMi 的 Min-Max Objective

设第 \(i\) 个 Domain 的平均 Excess Loss 为：

$$
E_i(\theta)
=
L_i(\theta)-L_i(\mathrm{ref})
$$

DoReMi 的核心优化目标可以写成：

$$
\boxed{
\min_\theta
\max_{\alpha\in\Delta^K}
\sum_{i=1}^{K}\alpha_iE_i(\theta)
}
$$

其中：

* \(\theta\)：Proxy Model 的参数。
* \(\alpha\)：Domain Weights。
* \(\Delta^K\)：所有合法 Domain 权重组成的概率单纯形。
* \(E_i(\theta)\)：第 \(i\) 个 Domain 的平均 Excess Loss。

这个公式虽然看起来复杂，但可以拆成两个相互竞争的角色。

**第一个角色：\(\max_\alpha\)，不断寻找当前最大的短板。**

如果某个 Domain 的 Excess Loss 最大，就会倾向于增加该 Domain 的权重。

**第二个角色：\(\min_\theta\)，努力降低被强调 Domain 的 Loss。**

Proxy Model 根据当前权重执行梯度更新，试图缩小对应的 Excess Loss。

因此，整个过程可以理解成：

```text
       Domain Weight Optimizer
                  │
                  ▼
        寻找 Excess Loss 较高的域
                  │
                  ▼
            增加 Domain Weight
                  │
                  ▼
       Proxy Model 更关注该域
                  │
                  ▼
          尝试降低该域 Loss
                  │
                  ▼
           重新计算 Excess Loss
                  │
                  └──────→ 循环
```

这里有一个有趣的数学性质：

当 \(\alpha\) 可以在概率单纯形上自由选择时，

$$
\max_{\alpha\in\Delta^K}
\sum_{i=1}^{K}\alpha_iE_i
=
\max_i E_i
$$

也就是说，内层最大化本质上是在寻找 Excess Loss 最大的 Domain。

不过实际训练是交替执行参数更新和权重更新，而不是在每一步都直接把全部权重分配给单个 Domain。

### 5.3 Reweighting 在数学上是如何实现的？

DoReMi 使用 Exponentiated Gradient 方法更新权重。

简化后的更新公式为：

$$
\widetilde{\alpha}_{t,i}
=
\alpha_{t-1,i}
\exp(\eta\widehat E_{t,i})
$$

其中：

* \(\eta\)：权重更新步长。
* \(\widehat E_{t,i}\)：当前 minibatch 估计的 Domain Excess Loss。
* \(\widetilde{\alpha}_{t,i}\)：尚未归一化的新权重。

然后进行归一化：

$$
\alpha_{t,i}
=
\frac{\widetilde{\alpha}_{t,i}}
{\sum_j\widetilde{\alpha}_{t,j}}
$$

直观来说：

* Excess Loss 较大的 Domain，权重增长得更快。
* Excess Loss 较小的 Domain，相对权重会下降。
* 每次更新后，所有权重之和仍然为 1。

### 5.4 手算一次 Domain Weight 更新

假设有三个 Domain，初始权重完全相同：

$$
\alpha^{(0)}
=
\left(
\frac13,\frac13,\frac13
\right)
$$

对应的 Excess Loss 为：

$$
E=(0.2,0.1,0.6)
$$

取：

$$
\eta=1
$$

暂时不考虑 smoothing。

根据：

$$
\alpha_i^{(1)}
\propto
\alpha_i^{(0)}e^{E_i}
$$

归一化后得到：

$$
\boxed{
\alpha^{(1)}
\approx
(0.294,\ 0.266,\ 0.439)
}
$$

从结果中可以观察到：

| Domain |  初始权重 | 更新后权重 |
| ------ | ----: | ----: |
| A      | 33.3% | 29.4% |
| B      | 33.3% | 26.6% |
| C      | 33.3% | 43.9% |

Domain C 的 Excess Loss 最大，因此获得了更高的权重。

**权重不是人工决定的，而是根据训练过程中观察到的学习差距动态更新的。**

### 5.5 为什么还需要 Clipping 和 Smoothing？

论文中的实际算法还考虑了两个问题。

**第一，Token-level Clipping。**

DoReMi 在更新 Domain Weights 时，先计算每个 token 的 Loss 差值，再进行非负截断：

$$
\Delta\ell_j^+(x)
=
\max\left(
\ell_{\theta,j}(x)-\ell_{\mathrm{ref},j}(x),
0
\right)
$$

随后在 Domain 内按 token 求平均，得到当前的 \(\widehat E_{t,i}\)。

需要注意：

**先对每个 token 截断再求平均，不等于先求平均再截断。**

这样做的一个原因是适配非负损失形式的 Group DRO 权重更新。

**第二，Domain Weight Smoothing。**

为了避免权重过度集中，算法在归一化后混入一小部分均匀分布：

$$
\boxed{
\alpha_{t,i}
=
(1-c)
\frac{\widetilde{\alpha}_{t,i}}
{\sum_j\widetilde{\alpha}_{t,j}}
+
\frac{c}{K}
}
$$

其中 \(c\) 是 smoothing 系数。

论文实现使用了：

$$
c=10^{-3}
$$

这可以让每个 Domain 保留一个很小的非零权重。

### 5.6 用 Python 理解一次权重更新

下面是一段简化的代码，只演示 Domain Weight 的 Exponentiated Gradient 更新，不包含完整的 Proxy Model 训练。

```python
import math

def update_domain_weights(
    alpha,
    excess,
    eta=1.0,
    smoothing=1e-3
):
    # excess 已经过 token-level clipping 和 domain aggregation
    if not alpha or len(alpha) != len(excess):
        raise ValueError("alpha and excess must be nonempty and the same length")

    if any(not math.isfinite(w) or w <= 0 for w in alpha):
        raise ValueError("all input weights must be finite and positive")

    if any(not math.isfinite(x) or x < 0 for x in excess):
        raise ValueError("excess must be finite and nonnegative")

    if not (
        math.isfinite(eta)
        and eta >= 0
        and 0 <= smoothing <= 1
    ):
        raise ValueError("invalid hyperparameters")

    # log-space 计算，避免 exp 溢出
    logits = [
        math.log(w) + eta * x
        for w, x in zip(alpha, excess)
    ]

    pivot = max(logits)
    raw = [math.exp(v - pivot) for v in logits]

    z = math.fsum(raw)
    k = len(alpha)

    return [
        (1 - smoothing) * v / z + smoothing / k
        for v in raw
    ]


alpha = [1 / 3] * 3
excess = [0.2, 0.1, 0.6]

# 关闭 smoothing，以便对照前面的手算示例
new_alpha = update_domain_weights(
    alpha,
    excess,
    smoothing=0.0
)

print([round(v, 3) for v in new_alpha])
# [0.294, 0.266, 0.439]
```

这段代码最值得关注的是：

```python
math.log(w) + eta * x
```

它对应的就是：

$$
\alpha_i\exp(\eta E_i)
$$

代码采用 log-space 形式，避免数值过大时直接计算指数带来的溢出问题。

但要记住：这只是 DoReMi 的一个权重更新组件，真正的算法还包括 Reference Model、Proxy Model 和梯度训练过程。

---

## 6. 为什么还要专门训练 Proxy Model？

到这里，我们已经知道：

Reference Model 提供参考 Loss，Proxy Model 提供当前 Loss，Group DRO 利用两者的差距学习 Domain Weights。

那么问题来了：

**为什么不直接在最终的大模型上运行 Group DRO？**

答案是：成本。

### 6.1 直接使用大模型搜索配比太昂贵

假设最终需要训练一个非常大的语言模型。

如果要比较不同的数据配比：

```text
Mixture A → Train Large Model
Mixture B → Train Large Model
Mixture C → Train Large Model
Mixture D → Train Large Model
```

每一次完整训练都会消耗大量算力。

如果 Domain 数量较多，权重搜索空间还会进一步扩大。

因此，直接依靠大模型搜索数据配比并不经济。

### 6.2 Proxy Model 本质上是数据配比的探测器

DoReMi 将这个流程改成：

```text
               Original Dataset
                       │
                       ▼
                 Initial Mixture
                       │
                       ▼
             Small Reference Model
                       │
                       ▼
             Small Proxy Model
               + Group DRO
                       │
                       ▼
              Optimized Mixture
                       │
                       ▼
                Large Model
```

Reference Model 和 Proxy Model 规模较小。

它们主要承担数据配比探索的工作，而最终大模型可以使用这些已经优化过的权重进行常规训练。

### 6.3 为什么小模型找到的比例能够给大模型使用？

这里存在一个关键假设：

> 较小模型上观察到的不同 Domain 的相对学习需求，在一定程度上可以迁移到更大规模的模型。

例如，小模型发现某个 Domain 存在较大的相对学习差距。

DoReMi 假设增加该 Domain 的训练比例，也有可能改善更大模型的训练效率和泛化表现。

但是，这并不是严格的数学保证。

不同规模的模型可能具有不同的学习能力、优化行为以及对数据的利用效率。

因此，**Proxy Model 发现的权重是一种可以迁移尝试的优化结果，而不是适用于所有模型规模的绝对最优配比。**

---

## 7. DoReMi 的完整训练流程

论文中的 DoReMi 可以拆成三个阶段。

### Step 1：训练 Reference Model

首先选择一组初始权重：

$$
\alpha_{\mathrm{ref}}
$$

例如使用原始数据配比或均匀配比。

然后训练一个较小的 Reference Model：

$$
p_{\mathrm{ref}}
$$

这个模型为后续计算 Excess Loss 提供参照。

### Step 2：训练 Proxy Model，同时学习 Domain Weights

接下来从头训练一个与 Reference Model 规模和结构相同的 Proxy Model。

训练过程中：

1. 按照均匀的 Domain 分布构造 minibatch。
2. 使用 Proxy Model 和 Reference Model 计算 token-level Loss。
3. 计算经过 clipping 的 per-domain Excess Loss。
4. 使用 Exponentiated Gradient 更新 Domain Weights。
5. 根据更新后的权重构造加权训练目标，更新 Proxy Model 参数。
6. 重复以上过程。

这里有一个特别容易理解错误的细节。

**原论文的 Proxy 阶段不是简单地每次按照新的 \(\alpha\) 对 Domain 重新采样。**

它在计算 Domain 权重时，采用均匀 Domain 采样构造 minibatch，再通过加权损失来控制不同 Domain 对 Proxy Model 梯度的贡献。

也就是说：

**Proxy 阶段主要是在训练目标上进行 Domain Reweighting。**

而最终训练大模型时，才根据优化得到的 Domain Weights 对数据进行重新采样。

### Step 3：平均权重并训练 Large Model

训练过程中，每一步都会得到新的：

$$
\alpha_t
$$

但最终输出并不是最后一步的权重。

DoReMi 对训练轨迹中的 Domain Weights 求平均：

$$
\boxed{
\bar{\alpha}
=
\frac{1}{T}\sum_{t=1}^{T}\alpha_t
}
$$

然后使用：

$$
P_{\bar{\alpha}}
$$

作为最终大模型的训练数据分布。

这意味着，DoReMi 最终真正需要保留的产物是：

$$
\boxed{\bar{\alpha}}
$$

而不是 Proxy Model 本身。

在最终的大模型训练阶段，可以不再使用 Reference Model 和 Proxy Model，而是按照优化后的采样分布执行标准预训练。

因此，可以把 DoReMi 理解成：

**位于大模型训练之前的数据配比优化过程。**

---

## 8. DoReMi 真的有效吗？从论文实验看结果

前面都是算法原理。

但我们还需要回答一个更实际的问题：

**通过小模型优化 Domain Weights，真的能够帮助大模型训练吗？**

论文在 The Pile 和 GLaM 两个数据集上进行了实验。

### 8.1 小模型探索，大模型受益

在 The Pile 实验中：

* Reference Model：280M 参数。
* Proxy Model：280M 参数。
* Main Model：8B 参数。

作者使用小模型找到的数据配比来训练 8B 模型。

论文报告，相对于使用 The Pile 默认权重训练的 Baseline：

**第一，平均 one-shot 下游任务准确率提高了 6.5 个百分点。**

注意这里是 percentage points，而不是相对提高 6.5%。

**第二，达到 Baseline 最终准确率水平所需的训练步数减少到了约 1/2.6。**

论文报告的对比是：

* Baseline：约 200K steps。
* DoReMi：约 75K steps 达到相应准确率水平。

这里说的是达到某个准确率水平所需的训练步数，并不等价于端到端实际运行时间一定缩短相同比例。

**第三，优化 Domain Weights 的额外计算开销相对较小。**

在该实验设置中，训练两个 280M 小模型的计算量约为训练 8B Main Model 所需 FLOPs 的 8%。

这些结果说明：在论文的实验条件下，改进数据配比可以显著提升训练效率。

### 8.2 一个反直觉的结果：Web 数据反而被大幅 Upweight

还记得最开始的例子吗？

我们直觉上可能认为：

> Web 数据噪声比较多，所以应该降低 Web 的采样比例。

但是原论文在 The Pile 上得到的结果并不是这么简单。

下面摘取论文 Table 1 中的部分数据：

| Domain    | Baseline Weight | DoReMi Weight |
| --------- | --------------: | ------------: |
| Pile-CC   |          11.21% |        60.57% |
| Wikipedia |           9.19% |         6.99% |
| GitHub    |           4.27% |         1.79% |
| ArXiv     |          10.52% |         0.36% |

可以看到：

**Pile-CC 这个网页文本 Domain，反而被大幅增加了权重。**

而 Wikipedia、GitHub 和 ArXiv 的权重有所降低。

这非常值得思考。

它说明 DoReMi 并不是简单地：

```text
网页数据   → 降权
学术数据   → 加权
代码数据   → 加权
```

DoReMi 也没有内置一套固定的“优质 Domain 排名”。

它根据特定数据集、模型和训练设置中的 Excess Loss 信号，寻找更合适的配比。

因此，**我们不能把人类对 Domain 质量的主观判断，直接等同于这些数据在特定训练阶段的边际训练收益。**

论文还观察到，即使某些 Domain 被 Downweight，最终模型在 The Pile 各 Domain 的 perplexity 仍然可以得到改善。

一个可能的解释是，不同 Domain 之间存在知识和表示的迁移效应，因此增加某类数据的训练，不一定只对该类数据有好处。

不过，论文也将这种解释作为可能的机制，而不是一个对所有数据集都成立的定律。

---

## 9. Iterated DoReMi：如果 Reference Model 自己就有偏差怎么办？

前面讲到，DoReMi 使用 Reference Model 作为参照。

但这里存在一个问题：

**如果 Reference Model 本身是根据不合理的数据配比训练出来的，那么它的 Loss 能否作为可靠参照？**

例如，初始配比严重低估了某些 Domain 的训练需求。

那么 Reference Model 可能在这些 Domain 上表现较差。

这会影响之后得到的 Excess Loss 信号，也可能影响最终配比。

为此，论文进一步提出了 Iterated DoReMi。

其思路是：

```text
Initial Mixture
      │
      ▼
  DoReMi Round 1
      │
      ▼
   Mixture v1
      │
      ▼
重新训练 Reference Model
      │
      ▼
  DoReMi Round 2
      │
      ▼
   Mixture v2
      │
      ▼
     ...
```

也就是将上一轮优化得到的 Domain Weights，用作下一轮 Reference Model 的训练配比。

直到权重逐渐稳定。

在论文的 GLaM 实验中，作者观察到 Iterated DoReMi 在较少的轮次内能够趋于稳定。

这说明：

**DoReMi 不仅可以从初始配比开始优化，还可以通过迭代更新 Reference Model 来进一步修正配比。**

但迭代也意味着额外的训练成本，而且不能保证任何场景下迭代越多就一定越好。

---

## 10. 联系 Post-training：Sampling Weight 其实也在解决同一类问题

这一部分是我学习 DoReMi 时最希望形成的知识迁移。

虽然论文研究的是 Pretraining，但它背后的 Data Mixture 思想，与 Post-training 中的 Sampling Weight 有很明显的联系。

### 10.1 Post-training 为什么也需要 Sampling Weight？

假设我们进行 SFT，拥有以下几类训练数据：

| Data Type  | 数据特点      |
| ---------- | --------- |
| General QA | 数量较多的通用问答 |
| Math       | 数学推理      |
| Coding     | 代码相关任务    |
| Safety     | 安全相关数据    |

如果直接按照原始样本数量采样，那么数据量最大的 General QA 可能占据大量训练步骤。

但实际目标可能不仅是改善通用问答，还需要保持数学、代码和安全方面的表现。

因此，我们可能设计：

```python
sampling_weight = {
    "general": 0.35,
    "math": 0.30,
    "coding": 0.20,
    "safety": 0.15,
}
```

这组比例只是一个假设性的例子，并不代表经过实验验证的最优方案。

它体现的是：

**即使数据量不同，也可以通过控制 Sampling Weight，主动调整不同类型数据在训练中的参与程度。**

### 10.2 Pretraining 和 Post-training 的共性

两者可以用相似的数学形式表达。

假设有 \(K\) 个训练 Domain：

$$
\mathcal{L}(\theta;w)
=
\sum_{i=1}^{K}w_i
\mathbb{E}_{x\sim P_i}
[\ell_\theta(x)]
$$

其中：

* \(w_i\)：第 \(i\) 类数据的训练权重。
* \(P_i\)：对应的数据分布。
* \(\ell_\theta(x)\)：当前训练任务的损失函数。

在 Pretraining 中，这些 Domain 可能是：

```text
Web / Books / Wikipedia / Code
```

在 Post-training 中，则可能是：

```text
General / Math / Coding / Safety
```

虽然使用的数据、Loss 和优化目标不同，但存在相通的思想：

$$
\boxed{
\text{Data Mixture}
\longrightarrow
\text{Training Resource Allocation}
}
$$

也就是说，Sampling Weight 并不只是一个 DataLoader 参数。

它在一定程度上决定了：

**哪些数据在训练过程中获得更多优化机会。**

### 10.3 两者有什么重要区别？

不能因为两者数学形式相似，就认为它们完全相同。

| 维度   | Pretraining Mixture       | Post-training Weighting |
| ---- | ------------------------- | ----------------------- |
| 数据类型 | 大规模通用文本                   | 指令、推理、偏好等数据             |
| 常见目标 | 语言建模与通用表示学习               | 指令遵循、任务能力、对齐等           |
| 损失函数 | 通常为 Next-token Prediction | 依训练阶段而不同                |
| 权重依据 | 数据来源、Loss、跨域表现等           | 能力指标、数据质量、任务需求等         |
| 主要风险 | 分布失衡、数据冗余、跨域泛化不足          | 能力遗忘、过拟合、能力间相互影响等       |

例如，在 Post-training 中，即使某个任务的 Loss 已经很低，也不能仅凭这一点就大幅降低其采样权重。

因为它可能承担着能力保持的作用。

对于 Safety 等数据，还可能存在额外的约束和评估要求。

因此，我认为正确的迁移方式并不是：

> 直接把 DoReMi 原封不动搬到 Post-training。

而应该是：

> 借鉴 DoReMi 使用训练反馈信号优化数据配比的思想，结合 Post-training 的实际能力目标和验证指标，设计更系统的数据加权策略。

### 10.4 如果要将这种思想用于真实训练，我会怎么做？

假设我负责优化一个 SFT 数据集的 Sampling Weight。

可以按照以下步骤设计实验。

**Step 1：建立 Domain 划分。**

明确 General、Math、Coding、Safety 等数据的边界，记录数据量、token 数量、质量和重复情况。

**Step 2：建立 Baseline。**

至少比较：

* 按原始数据量采样。
* 均匀 Domain Sampling。
* 当前人工经验配比。

**Step 3：构建独立评估集。**

不仅观察整体平均 Loss，还需要按 Domain 记录评估结果。

例如数学正确率、代码任务表现、通用问答质量以及安全相关指标。

**Step 4：分析不同 Domain 的学习状态。**

观察：

* 训练和验证 Loss 的变化。
* 不同能力的提升与退化。
* 数据加权后的跨任务影响。
* 特定 Domain 是否存在过拟合。

**Step 5：根据训练信号优化权重。**

可以从小范围人工搜索开始，也可以尝试参考 DoReMi 的 Reference Model、Proxy Model 和 Excess Loss 思想。

**Step 6：做受控对比实验。**

保持模型、训练 token 预算、优化器等条件尽可能一致，只改变 Sampling Weight。

这样才能更清楚地判断性能变化究竟是不是由 Data Mixture 引起的。

如果要进一步证明方法有效，还应在不同随机种子和独立测试集上验证结果。

---

## 11. DoReMi 有哪些局限？

理解一篇论文，不仅要理解它为什么有效，还需要理解它在哪些情况下可能失效。

### 11.1 Domain 划分是人为设定的

DoReMi 主要优化的是 Domain-level Weights。

如果某个 Domain 内部存在明显的质量差异，统一调整它的权重可能不够精细。

例如同属 Web 的高质量技术文章与低质量重复网页，就可能需要不同的处理方式。

这意味着 DoReMi 并不能代替数据清洗、去重和细粒度质量控制。

### 11.2 Excess Loss 并不等于真实 Marginal Utility

DoReMi 使用的是：

$$
L_i(\theta)-L_i(\mathrm{ref})
$$

但我们真正关心的可能是：

$$
\frac{\Delta\text{Downstream Performance}}
{\Delta\text{Training Compute}}
$$

这两个量并不等价。

某个 Domain 的 Excess Loss 大，不代表它一定能为目标任务带来最大的提升。

因此，最终仍然需要通过模型训练和评估来验证新的配比。

### 11.3 Reference Model 会影响结果

Reference Model 的初始配比、训练程度和模型能力都会影响 Excess Loss。

因此，DoReMi 优化的是相对于特定参考水平的学习差距，而不是一个绝对的数据价值排名。

Iterated DoReMi 可以缓解部分问题，但也不能彻底消除参考模型带来的依赖。

### 11.4 Proxy Model 到 Large Model 的迁移存在边界

小模型和大模型可能具有不同的训练动态。

某个 Domain 对小模型很有帮助，并不意味着它对所有规模的大模型都有同样的收益。

论文在其实验条件下验证了这种迁移的有效性，但并未证明其在任意模型规模和训练预算下都成立。

### 11.5 配比可能依赖训练预算

一个比较容易忽略的问题是：

**最合适的数据配比可能随着训练阶段和总训练预算发生变化。**

训练初期模型可能需要更多某类基础数据，而训练后期可能对其他数据产生更高需求。

DoReMi 的权重本身也是随着训练过程动态变化的。

因此，将不同训练阶段得到的权重简单看作永久不变的最优比例，并不严谨。

---

## 12. 如果面试官问我 DoReMi，我会怎么回答？

### Q1：DoReMi 解决的是什么问题？

DoReMi 解决的是语言模型预训练中的 Data Mixture Optimization 问题。

由于不同 Domain 的数据量不等于它们的训练价值，直接按照原始数据比例采样不一定合理。

DoReMi 利用较小的 Reference Model 和 Proxy Model，通过 Group DRO 优化各 Domain 的训练权重，再将优化后的配比用于更大模型的预训练。

### Q2：为什么不能直接对 Loss 最大的 Domain 增加采样权重？

因为 High Loss 不一定代表 High Learning Value。

某些 Domain 可能由于较高的条件熵或数据噪声而始终难以预测。

如果只根据 Loss 加权，可能会把过多训练预算分配给收益有限的 Domain。

DoReMi 使用 Proxy Loss 与 Reference Loss 的差值，即 Excess Loss，作为相对学习差距的信号，以降低仅根据绝对 Loss 判断数据价值的局限。

### Q3：Reference Model 和 Proxy Model 分别起什么作用？

Reference Model 提供一个已经训练得到的 Loss 参考水平。

Proxy Model 在 Group DRO 目标下训练，其 Loss 与 Reference Loss 的差距用于动态更新 Domain Weights。

最终要使用的是训练过程中获得的平均 Domain Weights，而不是直接将 Proxy Model 作为最终模型。

### Q4：DoReMi 和 Post-training 的 Sampling Weight 有什么联系？

我认为两者都属于 Data Mixture Optimization 的范畴。

它们都在解决如何将有限训练预算分配给不同类型数据的问题。

区别在于，DoReMi 主要面向 Pretraining 的 Domain-level Reweighting，而 Post-training 往往需要考虑更具体的任务能力、遗忘风险、指令遵循和安全约束。

因此，可以迁移它的方法论，但不能直接假设相同算法在所有训练阶段都最优。

---

## 13. 总结：我从 DoReMi 中真正学到了什么？

学习 DoReMi 之前，我可能会把数据配比理解为一个经验参数：

> Web 多一点，Code 少一点，Math 再补一点。

但学习这篇论文之后，我更愿意把它理解成：

**一个需要通过训练反馈进行优化的资源配置问题。**

整篇论文可以压缩为一条逻辑链：

```text
原始数据量
    ≠
最优训练配比
    │
    ▼
不能只凭经验或 Loss 决定权重
    │
    ▼
使用 Reference Model 提供参考水平
    │
    ▼
计算 Proxy Model 的 Excess Loss
    │
    ▼
使用 Group DRO 动态调整权重
    │
    ▼
平均训练过程中的 Domain Weights
    │
    ▼
得到 Optimized Data Mixture
    │
    ▼
使用优化后的配比训练 Large Model
```

对我而言，最重要的是下面五点：

1. **数据的原始占比不等于最优训练占比。** 数据量仅仅描述资源有多少，不代表它应该获得多少训练预算。

2. **High Loss 不等于 High Learning Value。** 当前难以预测的数据，不一定值得投入更多算力。

3. **Excess Loss 提供相对学习差距的信号。** 通过 Reference Model 和 Proxy Model 的对比，可以获得比单独观察 Loss 更丰富的信息。

4. **Group DRO 将数据配比变成了可优化的变量。** 不再完全依赖人工设置权重，而是根据训练反馈动态调整。

5. **Pretraining Domain Mixture 和 Post-training Sampling Weight 具有相通的方法论。** 它们都与如何分配有限训练资源有关，但需要结合各自训练阶段的目标和约束使用。

最后，用一句话概括我对 DoReMi 的理解：

> **DoReMi 的价值不只是提供了一套自动调整 Domain Weights 的算法，更重要的是，它展示了如何借助小模型的训练反馈，为更大模型的数据资源分配提供可操作的优化信号。**

这也是 Data-Centric AI 中一个很值得持续学习的方向：

不仅关注如何改进模型结构和优化算法，也关注如何让模型在有限预算下学习到更合适的数据。

---

## 参考文献

**[1] DoReMi 原论文（核心参考）**

Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*. Advances in Neural Information Processing Systems (NeurIPS 2023).

* arXiv：https://arxiv.org/abs/2305.10429
* NeurIPS 论文原文：https://proceedings.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf
* 官方论文页面：https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html

本文对 DoReMi 三阶段训练流程、Excess Loss、Min-Max Objective、权重更新算法及论文实验数据的介绍，主要依据该论文第 2～4 节及相关附录。

**[2] DoReMi 公开 PyTorch 实现**

Sang Michael Xie. *DoReMi: Domain Reweighting with Minimax Optimization*.

* GitHub：https://github.com/sangmichaelxie/doremi

用于进一步理解 Domain Weight 更新、训练数据采样及实验复现。

**[3] Stanford CRFM：DoReMi 作者技术解读**

Xie, S. M. (2023). *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*. Stanford Center for Research on Foundation Models.

* https://crfm.stanford.edu/2023/09/14/doremi

适合结合原论文理解算法动机、完整流程与实验结果。

**[4] Group DRO 相关论文**

Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*. ICLR 2020.

* arXiv：https://arxiv.org/abs/1911.08731
* GitHub：https://github.com/kohpangwei/group_DRO

用于进一步理解 Group DRO 的优化目标和算法背景。

**[5] The Pile 数据集论文**

Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*.

* arXiv：https://arxiv.org/abs/2101.00027

用于了解 DoReMi 实验采用的多 Domain 预训练数据集背景。

---

*注：文章中的四类数据配比例子、A/B/C Excess Loss 示例、Python 权重更新演示及 Post-training 采样权重均为教学示例，并非原论文实验数据；第 8 节中的实测权重和性能结论来自 DoReMi 原论文。*
