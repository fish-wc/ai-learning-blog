# Day 16｜Transformer：从训练工程视角理解 Attention、长上下文与 GQA

> **学习主题**：Transformer / Self-Attention / FLOPs / Memory / FlashAttention / KV Cache / MHA / MQA / GQA
>
> **学习目标**：不只是记住 Attention 的公式，而是能够从张量形状和计算过程出发，理解 Transformer 为什么昂贵，以及现代大语言模型如何优化训练和推理效率。
>
> **本文核心问题**：
>
> 1. Sequence Length 从 4K 增加到 8K，Attention FLOPs 为什么大约变成 4 倍？
> 2. Attention 的显存为什么可能按平方增长？FlashAttention 又解决了什么？
> 3. 为什么 Long Context 如此昂贵？
> 4. MHA、MQA、GQA 有什么区别？它们如何影响 KV Cache？

---

## 一、从一个问题开始：为什么上下文长度翻倍，成本却不只是翻倍？

假设我们正在训练一个 Transformer 模型。

原本模型一次处理：

```text
Sequence Length = 4096 tokens
```

现在希望它能够处理更长的文本：

```text
Sequence Length = 8192 tokens
```

直觉上，我们可能会认为：

> 输入 Token 数量增加了 2 倍，那么计算量和显存消耗应该也增加 2 倍。

然而，对于标准的 Dense Self-Attention，这个直觉并不完全正确。

在其他条件不变时：

| 指标                     | 4K → 8K  | 原因                   |
| ---------------------- | -------- | -------------------- |
| Attention 核心 FLOPs     | 约 4 倍    | 与序列长度平方成正比           |
| 朴素 Attention 中间矩阵显存    | 4 倍      | 矩阵形状为 N × N          |
| KV Cache 显存            | 2 倍      | 与缓存的 Token 数量成正比     |
| 整个 Transformer 的 FLOPs | 不一定是 4 倍 | 还存在随 N 线性增长的计算       |
| 实际训练时间                 | 不一定是 4 倍 | 还受到硬件、Kernel、通信等因素影响 |

**这里最重要的不是记住倍数，而是理解：为什么不同部分的增长规律不同？**

要回答这个问题，需要重新理解 Attention 的计算过程。

## 二、Attention 到底在计算什么？

Transformer 中最核心的计算公式之一是：

$$
\boxed{
Attention(Q,K,V)=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
}
$$

这就是 Scaled Dot-Product Attention，最早在《Attention Is All You Need》中系统提出 [1]。

第一次看到这个公式，很容易把它当作需要记忆的数学表达式。

但从工程视角看，我们更应该问：

* Q、K、V 分别是什么？
* 它们的 Tensor Shape 是什么？
* 每一步矩阵运算产生多大的中间结果？
* 哪一步最消耗算力和显存？

### 2.1 Q、K、V 分别代表什么？

假设有一句话：

> 小猫跳上沙发，因为它想睡觉。

当模型处理“它”这个 Token 时，需要从上下文获取相关信息。

这里可以用一种直观的方式理解 Q、K、V：

| 名称 | 全称    | 直观含义                 |
| -- | ----- | -------------------- |
| Q  | Query | 当前 Token 想寻找什么信息     |
| K  | Key   | 每个 Token 提供什么可供匹配的特征 |
| V  | Value | 每个 Token 可以贡献什么信息    |

可以把 Attention 想象成一个动态的信息检索过程：

**第一步：查询**

“它”生成一个 Query，表示当前需要从上下文获取的信息。

**第二步：匹配**

这个 Query 与上下文中各个 Token 的 Key 进行相似度计算。

例如，可以想象得到如下注意力分布（仅用于解释，不代表真实模型输出）：

```text
当前 Query：「它」

上下文 Token       Attention Weight

小猫                 0.65
跳上                 0.05
沙发                 0.15
因为                 0.03
它                   0.07
想                   0.03
睡觉                 0.02
```

**第三步：读取信息**

Attention 根据这些权重，对不同 Token 的 Value 进行加权求和。

这样，当前 Token 就获得了来自上下文的信息。

注意，真实模型中的 Q、K、V 是通过可学习的线性投影得到的向量，Attention 权重也不必直接对应人类理解的语法关系。

因此，可以把 Attention 理解成：

> **让每一个 Token，根据当前上下文，动态决定从其他 Token 中读取哪些信息，以及读取多少。**

### 2.2 为什么需要三个矩阵？

假设输入为：

$$
X\in\mathbb{R}^{N\times D}
$$

其中：

* \(N\)：Sequence Length，即 Token 数量。
* \(D\)：Hidden Size，即每个 Token 的隐藏状态维度。

Transformer 会通过不同的线性投影生成：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

这三个矩阵的投影参数不同，学习的功能也不同。

接下来，我们暂时只观察一个 Attention Head，并假设 Q、K、V 的维度都是 \(d_h\)：

$$
Q,K,V\in\mathbb{R}^{N\times d_h}
$$

现在开始逐步拆解 Attention 公式。

### 2.3 第一步：计算 QKᵀ

$$
S=QK^T
$$

矩阵形状：

$$
(N\times d_h)(d_h\times N)
$$

结果：

$$
\boxed{S\in\mathbb{R}^{N\times N}}
$$

这里出现了整个 Attention 计算中最重要的结构：

**N × N 的 Attention Score Matrix。**

为什么结果是 \(N\times N\)？

因为每一个 Query Token，都要与每一个 Key Token 计算一次匹配分数。

假设只有 4 个 Token：

```text
                  Key

             K1  K2  K3  K4

Query   Q1    ×   ×   ×   ×
        Q2    ×   ×   ×   ×
        Q3    ×   ×   ×   ×
        Q4    ×   ×   ×   ×

总共 16 个匹配位置
```

如果变成 8 个 Token：

```text
                    Key

             K1 K2 K3 K4 K5 K6 K7 K8

Query   Q1    ×  ×  ×  ×  ×  ×  ×  ×
        Q2    ×  ×  ×  ×  ×  ×  ×  ×
        Q3    ×  ×  ×  ×  ×  ×  ×  ×
        Q4    ×  ×  ×  ×  ×  ×  ×  ×
        Q5    ×  ×  ×  ×  ×  ×  ×  ×
        Q6    ×  ×  ×  ×  ×  ×  ×  ×
        Q7    ×  ×  ×  ×  ×  ×  ×  ×
        Q8    ×  ×  ×  ×  ×  ×  ×  ×

总共 64 个匹配位置
```

可以发现：

$$
4^2=16
$$

$$
8^2=64
$$

Token 数量增加 2 倍，匹配矩阵的面积增加了 4 倍。

**这就是标准 Dense Self-Attention 存在平方复杂度的根本原因。**

补充一点：GPT 类 Decoder-only 模型通常采用 Causal Attention，也就是当前 Token 不能关注未来 Token。

因此，真正允许访问的位置是下三角区域，数量为：

$$
\frac{N(N+1)}{2}
$$

但这仍然是 \(O(N^2)\) 的增长规律。朴素实现也可能分配完整的 \(N\times N\) 矩阵。

### 2.4 第二步：为什么要除以 √d？

Attention 公式不是直接计算：

$$
softmax(QK^T)
$$

而是：

$$
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

这里的 \(d_k\) 是 Query 和 Key 的向量维度。

为什么需要缩放？

因为在一些常见的统计假设下，向量维度越大，点积结果的方差也越大。

如果点积数值过大，经过 Softmax 后，概率分布容易变得非常尖锐。

例如：

```text
Softmax 输入：

[1, 2, 3]

与

[10, 20, 30]
```

第二组的输出会更加接近 One-Hot 分布。

当 Softmax 过度饱和时，一些位置的梯度可能变得非常小，不利于优化。

因此，除以 \(\sqrt{d_k}\) 的重要目的，就是控制点积分数的尺度，让 Softmax 的数值和梯度行为更加稳定。

### 2.5 第三步：Softmax 和 V 在做什么？

通过：

$$
P=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

得到 Attention Probability Matrix：

$$
P\in\mathbb{R}^{N\times N}
$$

Softmax 是按 Query 对应的每一行计算的，每行权重之和为 1。

随后：

$$
O=PV
$$

形状为：

$$
(N\times N)(N\times d_h)
$$

最终得到：

$$
\boxed{O\in\mathbb{R}^{N\times d_h}}
$$

也就是说：

> Attention 先计算每个 Token 应该关注谁，再根据关注程度汇总对应的 Value 信息。

至此，我们已经可以把 Attention 理解成两次核心矩阵乘法：

```text
Q [N, d]       Kᵀ [d, N]
       \       /
        \     /
         QKᵀ
          |
          v
       S [N, N]
          |
       Scale
          |
       Softmax
          |
          v
       P [N, N]       V [N, d]
             \        /
              \      /
                P × V
                  |
                  v
              O [N, d]
```

接下来就可以开始计算成本了。

---

## 三、Attention FLOPs 为什么是 O(N²d)？

FLOPs 指 Floating Point Operations，即浮点运算次数。

它是描述模型计算工作量的重要指标。

不过需要注意：

**FLOPs 不等于实际运行时间。**

实际性能还会受到 GPU 利用率、显存带宽、Kernel 实现、通信等因素影响。

### 3.1 QKᵀ 需要多少计算？

假设：

$$
Q\in\mathbb{R}^{N\times d_h}
$$

$$
K^T\in\mathbb{R}^{d_h\times N}
$$

输出：

$$
S\in\mathbb{R}^{N\times N}
$$

也就是需要计算 \(N^2\) 个分数。

每个分数都是一次长度为 \(d_h\) 的 Dot Product。

如果将一次乘法和一次加法分别计算为一个 FLOP，那么一次点积大约需要：

$$
2d_h
$$

次浮点运算。

因此：

$$
FLOPs_{QK^T}\approx2N^2d_h
$$

### 3.2 Attention × V 需要多少计算？

接下来：

$$
O=PV
$$

其中：

$$
P\in\mathbb{R}^{N\times N}
$$

$$
V\in\mathbb{R}^{N\times d_h}
$$

同样是一项矩阵乘法：

$$
FLOPs_{PV}\approx2N^2d_h
$$

所以，对于一个 Attention Head，两次核心矩阵乘法的 FLOPs 约为：

$$
\boxed{
FLOPs_{Attention}\approx4N^2d_h
}
$$

这里忽略了 Softmax、缩放、Mask 等操作，并且没有考虑 Causal Kernel 跳过无效位置带来的常数差异。

如果加入 Batch Size 和 Query Head 数量：

$$
\boxed{
FLOPs_{Attention}\approx
4Bh_qN^2d_h
}
$$

其中：

* \(B\)：Batch Size。
* \(h_q\)：Query Head 数量。
* \(N\)：Sequence Length。
* \(d_h\)：Head Dimension。

当其他参数固定时：

$$
\boxed{FLOPs_{Attention}\propto N^2}
$$

### 3.3 从 4K 增加到 8K，会发生什么？

假设：

$$
N_1=4096
$$

$$
N_2=8192
$$

Attention Score Matrix 的元素数量分别为：

$$
4096^2=16,777,216
$$

$$
8192^2=67,108,864
$$

两者的比例：

$$
\frac{8192^2}{4096^2}
=
\left(\frac{8192}{4096}\right)^2
=4
$$

因此：

$$
\boxed{
Attention\ FLOPs:\ 4K\rightarrow8K\approx4\times
}
$$

这是在其他参数不变时，标准 Dense Attention 二次计算部分的结论。

### 3.4 一个容易被忽略的细节：整个 Transformer 并非一定增加 4 倍

Transformer 不只有 Attention。

一个典型 Transformer Block 还包含：

```text
Transformer Block
│
├── Q / K / V Linear Projection
│
├── Attention
│   ├── QKᵀ
│   ├── Softmax
│   └── Attention × V
│
├── Output Projection
│
├── Feed Forward Network / MLP
│
└── Normalization
```

对于固定的 Hidden Size：

* Q/K/V Projection：通常随 \(N\) 线性增长。
* Output Projection：通常随 \(N\) 线性增长。
* MLP：通常随 \(N\) 线性增长。
* Attention 核心矩阵乘法：随 \(N^2\) 增长。

因此，可以粗略表示为：

$$
FLOPs_{Block}
\approx
c_1ND^2+c_2N^2D
$$

这里的 \(c_1,c_2\) 是与模型结构和实现有关的常数。

所以：

> **Sequence Length 翻倍，Attention 的二次计算部分约增加 4 倍，但整个 Transformer 的 FLOPs 不一定精确增加 4 倍。**

还有一个训练工程细节：

如果保持每个 Step 的 Batch Size 不变，上述 Attention FLOPs 约为 4 倍。

但如果保持每个 Step 的 Token 总量 \(B\times N\) 不变，那么当 Sequence Length 翻倍时，Batch Size 就需要减半。

此时：

$$
BN^2\rightarrow
\frac{B}{2}(2N)^2
=2BN^2
$$

Attention 核心 FLOPs 总量反而约是原来的 2 倍。

这也是为什么讨论训练成本时，一定要说明：

**到底固定的是 Batch Size，还是每个 Step 的 Token 数量？**

---

## 四、Attention Memory 为什么也会按平方增长？

前面我们分析了计算量。

现在考虑 GPU 显存。

在朴素 Attention 实现中，需要显式生成：

$$
S=QK^T
$$

它的形状是：

$$
S\in\mathbb{R}^{N\times N}
$$

因此，仅仅存储一份 Attention Score Matrix，就需要：

$$
Memory_{Score}=N^2\times BytesPerElement
$$

对于 Batch Size 为 \(B\)、Query Head 数为 \(h_q\) 的情况：

$$
\boxed{
Memory_{Score}=Bh_qN^2s
}
$$

其中 \(s\) 是每个元素的存储字节数。

### 4.1 用 BF16 计算一个具体例子

假设：

* Batch Size = 1。
* Attention Head = 1。
* 数据按 BF16 存储。
* 每个 BF16 元素占 2 Bytes。

当 Sequence Length = 4096：

$$
Memory=4096^2\times2
$$

得到：

$$
\boxed{32\ MiB}
$$

当 Sequence Length = 8192：

$$
Memory=8192^2\times2
$$

得到：

$$
\boxed{128\ MiB}
$$

因此：

$$
\frac{128}{32}=4
$$

一个 Head 的 Attention Matrix 就增长了 4 倍。

如果有 32 个 Query Heads：

| Sequence Length |  单 Head | 32 Heads |
| --------------- | ------: | -------: |
| 4K              |  32 MiB |    1 GiB |
| 8K              | 128 MiB |    4 GiB |

注意：这里的数字仅表示**单层、Batch Size 为 1、存储一份 BF16 Attention Matrix** 的理论数据量。

它不是模型实际训练时的峰值显存。

真实训练还需要考虑：

* 模型参数。
* 梯度。
* Optimizer States。
* Q/K/V 和其他中间激活。
* Backward 需要保存或重算的数据。
* 不同 Kernel 的计算精度及工作空间。

另外，某些实现会使用 FP32 累积或中间存储，因此也不能简单认为所有 Attention 中间张量都一定按 BF16 保存。

### 4.2 为什么这个问题在训练时特别严重？

训练不仅有 Forward，还需要 Backward。

为了计算梯度，Backward 需要访问一些 Forward 中产生的信息。

如果使用朴素 Attention 实现，并且需要保存完整的 Attention Probability Matrix，那么大量 \(N\times N\) 中间数据就会带来明显的显存压力。

这会导致：

1. Sequence Length 受到显存限制。
2. Batch Size 可能被迫减小。
3. GPU 内存读写开销增加。
4. 长上下文训练更难高效扩展。

不过，这里必须补充一个重要的工程优化：

**FlashAttention。**

---

## 五、FlashAttention：为什么 Attention 不一定需要保存完整的 N×N 矩阵？

如果我们仅仅记住：

> Attention Memory Complexity = O(N²)

那么对现代大模型的训练工程理解还不够完整。

因为 \(O(N^2)\) 的中间矩阵显存需求，是朴素 Attention 实现的重要特征，而不是所有精确 Attention 实现都必须承担的代价。

FlashAttention [2] 正是为了解决这个问题而提出的。

### 5.1 传统 Attention 的问题

一个朴素的计算过程可以表示成：

```text
GPU HBM
  |
  | 读取 Q、K
  v
计算 QKᵀ
  |
  v
写入完整的 N×N Score Matrix
  |
  v
读取 Score Matrix
  |
  v
计算 Softmax
  |
  v
生成 Attention Probabilities
  |
  v
读取 Probability Matrix 和 V
  |
  v
计算 Attention Output
```

这带来两个成本：

**第一：显存容量**

巨大的 \(N\times N\) 中间矩阵会占用大量空间。

**第二：显存带宽**

GPU 不仅需要执行浮点运算，还需要在 HBM 和计算单元之间搬运数据。

即使 GPU 具有很高的理论 FLOPs，如果计算频繁等待内存读写，也无法充分发挥计算性能。

这就是为什么：

> GPU 算力很强，并不意味着所有矩阵运算都一定很快。

### 5.2 FlashAttention 的核心思想

FlashAttention 的重要思路是：

**不把完整的 N×N Attention Matrix 写入 HBM，而是通过分块计算和 Online Softmax 得到最终结果。**

可以用下面的过程理解：

```text
传统 Attention

Q、K
  ↓
完整 N×N Matrix
  ↓
Softmax
  ↓
乘 V
  ↓
Output


FlashAttention

Q、K、V
  ↓
拆分成小块
  ↓
在快速片上存储中计算局部结果
  ↓
使用 Online Softmax 更新归一化统计量
  ↓
逐块累积输出
  ↓
Output
```

GPU 中存在不同层级的存储。

例如：

* HBM：容量较大，但访问代价更高。
* SRAM / Shared Memory / Registers：更靠近计算单元，容量较小，但适合高速数据复用。

FlashAttention 利用 Tiling，让中间计算尽可能在片上完成，减少大量不必要的 HBM 读写。

### 5.3 FlashAttention 是否把 O(N²) 变成了 O(N)？

**没有。**

这是非常重要的区别。

对于标准 Exact Dense Attention：

$$
Compute=O(N^2d_h)
$$

FlashAttention 仍然需要完成这些 Token 之间的核心 Attention 运算。

因此，它并没有把 Dense Attention 的算术复杂度直接降低到 \(O(N)\)。

但它避免了完整 \(N\times N\) Attention Matrix 的朴素存储方式，使中间存储需求和 HBM IO 开销显著下降。

可以这样理解：

| 维度                        | 朴素 Attention    | FlashAttention              |
| ------------------------- | --------------- | --------------------------- |
| Dense Attention 核心计算      | O(N²d)          | O(N²d)                      |
| 是否显式保存完整 Attention Matrix | 通常需要            | 不需要                         |
| HBM IO 开销                 | 较大              | 显著优化                        |
| 中间显存                      | 可能存在 O(N²) 矩阵   | 避免完整 O(N²) 矩阵存储             |
| 计算结果                      | Exact Attention | Exact Attention（存在正常浮点舍入差异） |

因此，我认为正确的工程理解应该是：

> **FlashAttention 的核心价值不是取消 Attention 的平方计算，而是重新组织计算过程，减少显存中间存储和 GPU 内存层级之间的数据搬运。**

这也意味着：

Sequence Length 从 4K 增加到 8K，Attention 的二次计算工作量仍然约增加 4 倍，但实际训练峰值显存和运行时间不一定按 4 倍增长。

---

## 六、为什么 Long Context 昂贵？要区分 Prefill 和 Decoding

理解完训练中的 Attention，接下来把视角转向大语言模型推理。

对于 GPT 类自回归语言模型，推理通常可以分成两个阶段：

### 6.1 Prefill：处理输入 Prompt

假设输入：

```text
请阅读下面这篇很长的文章，并总结它的主要观点。
```

在 Prefill 阶段，模型会处理整个输入序列。

对于长度为 \(N\) 的 Prompt，Self-Attention 需要计算序列内部的注意力关系。

因此，标准 Dense Attention 的核心工作量仍然与 \(N^2\) 有关。

```text
Prefill

N 个 Query Tokens
        ×
N 个 Key Tokens
        ↓
Attention Work ∝ N²
```

这也是长 Prompt 处理成本较高的重要原因之一。

### 6.2 Decoding：逐个生成 Token

Prefill 之后，模型开始生成回答。

例如：

```text
Token 1：深
Token 2：度
Token 3：学
Token 4：习
...
```

自回归生成的特点是：

> 每次生成下一个 Token，都要利用已经存在的上下文信息。

假设已经有 8192 个历史 Token。

下一步生成时，新的 Query 需要与历史 Key 进行 Attention。

但这一次，Query 只有一个新 Token。

因此，使用 KV Cache 的单 Token Decoding 可以理解成：

$$
Q\in\mathbb{R}^{1\times d_h}
$$

$$
K^T\in\mathbb{R}^{d_h\times N}
$$

于是：

$$
QK^T\in\mathbb{R}^{1\times N}
$$

这时，每个 Head 的 Attention 分数不再是 \(N\times N\)，而是：

$$
1\times N
$$

所以在单 Token Decoding 时，Attention 核心计算量与历史长度近似呈线性关系：

$$
\boxed{Compute_{DecodeStep}=O(Nd_h)}
$$

这里暂时只考虑一个 Head。

不过，随着生成继续进行，历史 Token 越来越多，每一步需要访问的历史 K/V 也越来越多。

**单步是 O(N)，不意味着生成很长一段文本的累计成本也是 O(N)。**

---

## 七、KV Cache：为什么推理不能每次都重新计算历史 K/V？

现在考虑一个问题。

假设模型已经处理了：

```text
I love machine learning
```

现在准备生成下一个 Token。

模型需要利用已有上下文中的 K 和 V。

当模型再生成一个 Token 时，之前的 K/V 并不会因为新 Token 的出现而需要全部重新计算。

这是 Causal Transformer 的一个重要性质：

**历史 Token 不依赖未来 Token。**

因此，历史 K/V 可以复用。

### 7.1 不使用 KV Cache

如果每生成一个新 Token，都重新处理全部历史序列：

```text
Step 1:
I

Step 2:
I love

Step 3:
I love machine

Step 4:
I love machine learning
```

大量历史计算会被重复执行。

这显然非常浪费。

### 7.2 使用 KV Cache

KV Cache 的思路很直接：

> 将历史 Token 已经计算好的 K 和 V 保存下来，在后续 Decoding 时直接复用。

```text
                KV Cache
          ┌──────────────────┐
          │ Historical K     │
          │ Historical V     │
          └──────────────────┘
                   |
                   v
New Token → New Q、K、V
              |
              ├── New K/V 加入 Cache
              |
              └── New Q 与 Cached K 做 Attention
                         |
                         v
                      输出结果
```

这样，每生成一个新的 Token：

1. 只需要计算新 Token 的 Q/K/V。
2. 复用历史 Token 的 K/V。
3. 当前 Query 对所有可访问的历史 Key 计算 Attention。
4. 根据注意力权重汇总历史 Value。
5. 将新的 K/V 加入缓存。

KV Cache 的机制可以参考 Hugging Face 官方文档 [5]。

### 7.3 KV Cache 的 Memory Complexity 是多少？

假设：

* \(B\)：Batch Size。
* \(L\)：Transformer Layers 数量。
* \(N\)：缓存的 Sequence Length。
* \(h_{kv}\)：KV Heads 数量。
* \(d_h\)：每个 KV Head 的维度。
* \(s\)：每个元素所占的 Bytes。

则 KV Cache 的原始数据量可以估算为：

$$
\boxed{
Memory_{KV}=
2BLNh_{kv}d_hs
}
$$

其中最前面的 2 表示：

$$
K+V
$$

注意 KV Cache 保存的是每个 Token 的 Key 和 Value，而不是任意两个 Token 之间的 Attention Score。

因此：

$$
\boxed{Memory_{KV}\propto N}
$$

这意味着：

当 Sequence Length 从 4K 增加到 8K：

$$
\frac{8192}{4096}=2
$$

所以：

$$
\boxed{KV\ Cache\ Memory\approx2\times}
$$

这个结论非常重要。

**Attention Matrix 和 KV Cache 不是同一个东西。**

| 对象                  | 存储内容                | 随 N 增长 |
| ------------------- | ------------------- | ------ |
| 朴素 Attention Matrix | Token 之间的匹配分数或概率    | O(N²)  |
| KV Cache            | 每个历史 Token 的 K/V 向量 | O(N)   |

我们不能因为 Attention 的复杂度是 \(O(N^2)\)，就认为 KV Cache 也一定是 \(O(N^2)\)。

此外，KV Cache 虽然减少了重复计算，但每次 Decoding 通常还需要读取历史 K/V。

所以，长上下文推理的另一个重要瓶颈是：

**Memory Bandwidth。**

而这正是 MQA 和 GQA 要重点解决的问题。

---

## 八、MHA、MQA、GQA：为什么要让多个 Query Head 共享 KV？

现在开始理解 Transformer 中三个很重要的概念：

* MHA：Multi-Head Attention。
* MQA：Multi-Query Attention。
* GQA：Grouped-Query Attention。

理解它们之前，需要记住一件事：

> **它们的重要区别，是 Query Heads 与 Key/Value Heads 之间如何对应。**

### 8.1 MHA：每个 Query Head 都有独立的 KV Head

传统 Multi-Head Attention 可以理解为：

```text
Q1  → K1  V1
Q2  → K2  V2
Q3  → K3  V3
Q4  → K4  V4
...
Q32 → K32 V32
```

假设：

$$
h_q=32
$$

那么：

$$
h_{kv}=32
$$

也就是：

$$
\boxed{h_q=h_{kv}}
$$

每个 Query Head 都有对应的 Key Head 和 Value Head。

这样，不同 Attention Heads 能够学习不同的投影和信息交互方式。

但对于自回归推理来说，缺点也很明显：

**每一层都需要为所有 KV Heads 缓存历史 Token 的 K/V。**

当模型层数较多、上下文很长、并发请求很多时，KV Cache 可能占据大量显存。

### 8.2 MQA：所有 Query Heads 共享一组 KV

Noam Shazeer 在《Fast Transformer Decoding: One Write-Head is All You Need》中提出了 Multi-Query Attention [3]。

核心思路是：

> Query Heads 仍然有很多个，但所有 Query Heads 共享同一组 K/V。

例如：

```text
Q1  ─┐
Q2   │
Q3   │
Q4   │
...  ├──→ K1、V1
Q32 ─┘
```

此时：

$$
h_q=32
$$

$$
h_{kv}=1
$$

因此：

$$
\boxed{MQA:\ h_{kv}=1}
$$

与 32 KV Heads 的 MHA 相比：

$$
\frac{Memory_{MQA}}{Memory_{MHA}}
=
\frac{1}{32}
$$

也就是说，在其他条件相同的情况下：

**MQA 的 KV Cache 数据量只有 MHA 的 1/32。**

这对 Decoding 非常重要。

因为 KV Cache 变小，不仅可以节省显存，也可以减少读取历史 K/V 所需的显存带宽。

但问题是：

所有 Query Heads 都使用相同的 K/V 投影表示，可能对模型的表达能力和最终质量产生影响。

需要注意，这是一种可能的权衡，而不是说 MQA 一定会导致模型质量下降。

具体效果需要依赖模型结构、训练方式和实际任务评估。

### 8.3 GQA：MHA 和 MQA 之间的折中方案

既然：

* MHA 有 32 个 KV Heads。
* MQA 只有 1 个 KV Head。

那么可不可以使用中间数量的 KV Heads？

例如：

$$
h_q=32
$$

$$
h_{kv}=8
$$

答案就是：

**Grouped-Query Attention（GQA）。**

GQA 将 Query Heads 分组，每一组共享一组 K/V。

例如：

```text
Q1  ─┐
Q2   │
Q3   ├──→ K1、V1
Q4  ─┘

Q5  ─┐
Q6   │
Q7   ├──→ K2、V2
Q8  ─┘

...

Q29 ─┐
Q30  │
Q31  ├──→ K8、V8
Q32 ─┘
```

这里：

$$
\frac{32}{8}=4
$$

表示每 4 个 Query Heads 共用一组 KV。

因此，在相同的层数、序列长度和 Head Dimension 下：

$$
\frac{Memory_{GQA}}{Memory_{MHA}}
=
\frac{8}{32}
=
\frac14
$$

也就是：

**GQA 的 KV Cache 数据量只有对应 MHA 的 25%。**

GQA 的目标，是在推理效率与模型质量之间取得更好的平衡。

Ainslie 等人的 GQA 论文 [4] 展示了：在其研究设置中，GQA 可以取得接近 MHA 的质量，同时获得接近 MQA 的推理性能。

但这并不意味着所有 GQA 模型都能保证相同的效果。

### 8.4 三者放在一起比较

假设 Query Heads 固定为 32：

| 结构  | Query Heads | KV Heads | KV Cache 相对大小 |
| --- | ----------: | -------: | ------------: |
| MHA |          32 |       32 |          100% |
| GQA |          32 |        8 |           25% |
| MQA |          32 |        1 |        3.125% |

可以用一句话记住：

$$
\boxed{MHA\rightarrow GQA\rightarrow MQA}
$$

代表：

**K/V 被越来越多的 Query Heads 共享。**

而不是 Query Heads 数量不断减少。

这里还有一个容易忽视的细节：

GQA 并不会自动将 Attention 的核心 \(O(N^2)\) 计算变成 \(O(N)\)。

因为每个 Query Head 依然需要完成自己的 Attention 计算。

减少 KV Heads 主要能够减少：

* K/V Projection 的一部分计算和参数量。
* KV Cache 显存。
* Decoding 时读取 K/V 的显存带宽。

所以不能简单地说：

> GQA 把 KV Cache 降低到 1/4，Attention FLOPs 和推理延迟也一定降低到 1/4。

实际加速取决于具体负载和实现。

---

## 九、动手算一次：4K、8K、32K、128K 的 KV Cache 有多大？

前面已经理解了公式。

但如果想真正建立训练和推理工程直觉，最好亲自计算一组数据。

假设我们有一个**用于教学的模型配置**：

| 参数                 |   数值 |
| ------------------ | ---: |
| Transformer Layers |   32 |
| Query Heads        |   32 |
| GQA KV Heads       |    8 |
| Head Dimension     |  128 |
| KV Cache Precision | BF16 |
| Bytes Per Element  |    2 |
| Batch Size         |    1 |

注意，这只是一组计算示例，不代表某个特定模型的完整结构。

### 9.1 先计算 8K Context 的 GQA KV Cache

公式：

$$
Memory_{KV}=2BLNh_{kv}d_hs
$$

代入：

$$
B=1
$$

$$
L=32
$$

$$
N=8192
$$

$$
h_{kv}=8
$$

$$
d_h=128
$$

$$
s=2
$$

因此：

$$
Memory_{KV}
=
2\times1\times32\times8192\times8\times128\times2
$$

计算结果：

$$
1,073,741,824\ Bytes
$$

换算成 GiB：

$$
\frac{1,073,741,824}{1024^3}
=
1\ GiB
$$

所以：

$$
\boxed{8K\ GQA\ KV\ Cache=1\ GiB}
$$

这里只计算原始 KV 数据，不包括模型权重、其他激活、Allocator 额外开销以及缓存管理元数据。

### 9.2 计算不同 Context Length

保持其他条件不变：

| Context Length | MHA（32 KV） | GQA（8 KV） | MQA（1 KV） |
| -------------- | ---------: | --------: | --------: |
| 4K             |      2 GiB |   512 MiB |    64 MiB |
| 8K             |      4 GiB |     1 GiB |   128 MiB |
| 32K            |     16 GiB |     4 GiB |   512 MiB |
| 128K           |     64 GiB |    16 GiB |     2 GiB |

这里：

* 1 MiB = \(1024^2\) Bytes。
* 1 GiB = \(1024^3\) Bytes。
* 4K、8K、32K、128K 分别按 4096、8192、32768、131072 Tokens 计算。

从这张表可以看出两个现象。

**现象一：固定 Attention 结构，Context Length 翻倍，KV Cache 翻倍。**

例如 GQA：

```text
4K    → 512 MiB
8K    → 1 GiB
16K   → 2 GiB
32K   → 4 GiB
64K   → 8 GiB
128K  → 16 GiB
```

**现象二：固定 Context Length，减少 KV Heads 可以显著节省显存。**

例如 128K：

```text
MHA：64 GiB

GQA：16 GiB

MQA： 2 GiB
```

在这个例子中，单个请求的 MHA KV Cache 就可能达到非常大的规模。

如果还要支持多个并发请求，KV Cache 带来的显存压力会继续增加。

这也是为什么推理系统会非常关心 KV Cache 管理、KV Cache 量化、Paged KV Cache 和共享 K/V 等技术。

### 9.3 用 Python 验证 KV Cache 计算

我们可以写一个简单的 Python 函数，快速估算不同配置的原始 KV Cache 大小。

```python
def kv_cache_gib(
    layers,
    seq_len,
    kv_heads,
    head_dim,
    batch=1,
    bytes_per_element=2,
):
    values = [
        layers,
        seq_len,
        kv_heads,
        head_dim,
        batch,
        bytes_per_element,
    ]

    if any(not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("All parameters must be positive integers")

    # K 和 V 各需要一份存储
    elements = (
        batch
        * layers
        * seq_len
        * kv_heads
        * head_dim
        * 2
    )

    total_bytes = elements * bytes_per_element

    return total_bytes / (1024 ** 3)


lengths = [4096, 8192, 32768, 131072]

for name, kv_heads in [
    ("MHA", 32),
    ("GQA", 8),
    ("MQA", 1),
]:
    sizes = [
        kv_cache_gib(32, n, kv_heads, 128)
        for n in lengths
    ]

    print(name, sizes)
```

输出：

```text
MHA [2.0, 4.0, 16.0, 64.0]

GQA [0.5, 1.0, 4.0, 16.0]

MQA [0.0625, 0.125, 0.5, 2.0]
```

单位都是 GiB。

这个小工具还可以进一步扩展，加入：

* FP16 / BF16 / FP8 / INT8 等 KV 存储精度。
* 不同 Batch Size。
* 不同 Transformer Layers。
* 不同 KV Heads。
* 不同模型的特殊 Attention 结构。

不过要记住，它估算的是规则全上下文 KV Cache 的原始数据量。

实际模型可能采用 Sliding Window Attention、混合 Attention、量化缓存等方法，使缓存结构和大小与这里的公式不同。

---

## 十、FlashAttention 和 GQA 到底在优化什么？

到这里，我们已经理解了两类重要的 Attention 优化技术。

但是它们解决的问题并不完全相同。

可以从两个维度看。

### 10.1 FlashAttention：优化 Attention 计算过程

FlashAttention 更关注：

> 如何更加高效地执行 Attention，减少大量中间结果的显存存储和 HBM IO。

它主要通过：

* Tiling。
* Kernel Fusion。
* Online Softmax。
* 更好的片上数据复用。
* 避免显式存储完整 \(N\times N\) 矩阵。

来改善效率。

### 10.2 GQA / MQA：优化 KV 存储与读取

GQA 和 MQA 更关注：

> 是否真的需要给每个 Query Head 保存一组独立的 K/V？

通过多个 Query Heads 共享 K/V，减少：

* KV Cache 显存容量。
* K/V 相关的数据搬运。
* Decoding 时的显存带宽压力。

### 10.3 一张表总结两者的区别

| 优化技术           | 核心问题                   | 主要收益                   |
| -------------- | ---------------------- | ---------------------- |
| FlashAttention | Attention 中间矩阵与 HBM IO | 减少中间显存和内存访问            |
| GQA            | KV Heads 太多            | 降低 KV Cache 与 K/V 带宽需求 |
| MQA            | 进一步减少 KV Heads         | 更小的 KV Cache           |
| KV Cache       | 历史 K/V 重复计算            | 复用历史 K/V，加速 Decoding   |

所以，现代 Transformer 系统可以同时使用 FlashAttention 和 GQA。

它们不是互相替代的关系，而是可以互相配合的优化手段。

---

## 十一、用训练工程视角重新理解 Long Context

现在我们可以把前面的知识串起来。

**为什么 Long Context 昂贵？**

不能只回答：

> 因为 Token 数量增加了。

更准确的答案需要区分训练、Prefill 和 Decoding。

### 11.1 在训练阶段

Sequence Length 增加后：

$$
N\uparrow
$$

标准 Dense Attention 的核心计算：

$$
FLOPs\propto N^2
$$

因此，长上下文可能带来显著的算力开销。

如果采用显式保存 Attention Matrix 的朴素实现，中间显存也可能按平方增长。

FlashAttention 可以缓解中间显存和 HBM IO 问题，但不会消除 Dense Attention 本身的二次算术工作。

### 11.2 在 Prefill 阶段

对于长 Prompt，模型需要处理大量 Token 之间的关系。

因此：

$$
Compute_{Prefill}\propto N^2
$$

这里讨论的是 Dense Attention 的核心部分，而不是整个模型的全部计算。

### 11.3 在 Decoding 阶段

有了 KV Cache 后，每一步只新增一个 Query Token。

对于已有的 \(N\) 个历史 Token：

$$
Compute_{AttentionPerStep}\propto N
$$

但 KV Cache 也随上下文长度增长：

$$
Memory_{KV}\propto N
$$

同时，每一步通常还需要访问更多历史 K/V。

所以：

**上下文越长，单步 Decoding 的历史 Attention 计算量和 KV Cache 读取需求通常也越大。**

### 11.4 从头到尾的工程因果链

```text
                 Long Context
                      |
                      v
               Sequence Length ↑
                      |
          ┌───────────┴─────────────┐
          |                         |
          v                         v
   Training / Prefill           Decoding
          |                         |
          v                         v
   Dense Attention N²         KV Cache grows
          |                         |
    ┌─────┴─────┐              ┌────┴────┐
    |           |              |         |
    v           v              v         v
  FLOPs ↑   Naive Memory ↑   VRAM ↑   KV Read ↑
    |           |              |         |
    |           v              └────┬────┘
    |      FlashAttention           |
    |      Reduces IO              GQA / MQA
    |      and storage             |
    |                               v
    |                       Fewer KV Heads
    |
    v
Quadratic compute remains
```

这里需要强调：

FlashAttention 和 GQA 都是重要优化，但都不能保证任意长的上下文都能够以固定成本运行。

---

## 十二、几个特别容易理解错的地方

### 误区一：Sequence Length 翻倍，整个模型 FLOPs 一定翻 4 倍

不准确。

正确理解：

**标准 Dense Attention 的二次计算部分约翻 4 倍，但整个模型还有很多随 N 线性增长的操作。**

另外，Batch Size 是否变化也会影响每个 Step 的总计算量。

### 误区二：Attention Memory 永远是 O(N²)

不准确。

对于显式保存完整 Attention Matrix 的朴素实现，中间存储存在 \(O(N^2)\) 项。

但 FlashAttention 等算法不需要把完整矩阵写入 HBM，因此实际的 Attention 中间显存占用可以明显降低。

### 误区三：KV Cache 是 O(N²)

错误。

KV Cache 保存的是历史 Token 的 K/V 向量：

$$
Memory_{KV}\propto N
$$

它不是完整的 Attention Score Matrix。

### 误区四：GQA 把 KV Cache 降低到 1/4，模型推理就一定快 4 倍

错误。

GQA 确实能够减少 KV Cache 和 K/V 读取需求。

但是实际推理延迟还受到：

* 模型权重读取。
* Attention 运算。
* GPU 计算利用率。
* Kernel 实现。
* Batch Size。
* Prompt Length。
* 输出长度。

等因素影响。

因此，KV Cache 缩小 4 倍，不等于端到端推理一定加速 4 倍。

### 误区五：FlashAttention 和 GQA 解决的是同一个问题

不准确。

FlashAttention 主要优化 Attention 计算方式和 IO。

GQA 主要通过 KV 共享减少 KV Cache 与相关带宽开销。

两者可以同时使用。

---

## 十三、面试中如何回答这几个问题？

### Q1：为什么 Transformer 的 Self-Attention 具有 O(N²) 复杂度？

因为标准 Dense Self-Attention 中，每个 Query Token 都需要与所有可访问的 Key Token 计算匹配分数。

对于长度为 \(N\) 的序列，Attention Score Matrix 的形状是 \(N\times N\)。

每次 Dot Product 的维度为 \(d_h\)，因此核心计算复杂度为：

$$
O(N^2d_h)
$$

### Q2：Sequence Length 从 4K 增加到 8K，FLOPs 和 Memory 如何变化？

在其他参数固定的情况下：

* Dense Attention 的二次 FLOPs 约增加 4 倍。
* 朴素 Attention Matrix 的数据量增加 4 倍。
* KV Cache 的数据量增加 2 倍。

但整个模型的 FLOPs、实际训练显存和运行时间不一定遵循完全相同的倍数关系。

### Q3：为什么 FlashAttention 能够降低显存？

因为 FlashAttention 使用 Tiling 和 Online Softmax，避免显式保存完整的 \(N\times N\) Attention Matrix。

同时，它减少 GPU HBM 与片上存储之间的数据搬运，从而降低中间显存需求并改善执行效率。

但 Exact Dense Attention 的核心算术复杂度仍然是 \(O(N^2d)\)。

### Q4：MHA、MQA、GQA 的核心区别是什么？

三者的核心区别在于 KV Heads 的共享程度。

* MHA：每个 Query Head 对应独立 KV Head。
* MQA：所有 Query Heads 共享一个 KV Head。
* GQA：将 Query Heads 分组，每组共享一个 KV Head。

因此，MQA 和 GQA 可以减少 KV Cache 和 Decoding 时的 K/V Memory Bandwidth 需求。

### Q5：为什么 GQA 对 Long Context 推理特别重要？

因为 KV Cache 的数据量与 Sequence Length 和 KV Heads 数量成正比：

$$
Memory_{KV}\propto Nh_{kv}
$$

当上下文变长时，KV Cache 会不断增长。

GQA 通过减少 KV Heads，降低 KV Cache 的容量需求及相关的显存带宽压力，让长上下文推理更容易实现较高的资源利用率。

---

## 十四、总结：从公式思维转换到训练工程思维

以前看到 Transformer，我们可能首先想到经典的网络结构图：

```text
Embedding
   ↓
Attention
   ↓
Feed Forward
   ↓
Output
```

但从训练与推理工程视角，我们应该进一步思考：

```text
Transformer
     |
     v
Attention(Q, K, V)
     |
     ├── QKᵀ
     |     |
     |     └── N × N
     |           |
     |           ├── Quadratic FLOPs
     |           |
     |           └── Naive Quadratic Storage
     |
     ├── FlashAttention
     |     |
     |     └── Improve Memory IO
     |
     └── Autoregressive Decoding
           |
           v
         KV Cache
           |
           ├── Linear in Context Length
           |
           └── MHA / GQA / MQA
                 |
                 └── Reduce KV Sharing Cost
```

我认为，这一节最值得记住的并不是某个公式，而是三个不同的工程问题：

**第一：计算复杂度。**

为什么 Token 越多，需要计算的 Attention 关系就越多？

因为：

$$
QK^T:N\times N
$$

因此 Dense Attention 的计算具有平方增长特征。

**第二：显存与 IO。**

为什么完成相同数学计算，不同 Kernel 的实际效率差异可能很大？

因为实际 GPU 性能不仅取决于计算量，还取决于数据如何在不同内存层级之间移动。

FlashAttention 就是一个非常典型的 IO-aware 算法优化案例。

**第三：推理阶段的 KV Cache。**

为什么长上下文不仅影响 Prefill，也影响后续 Token 的生成效率？

因为历史 K/V 需要缓存和读取。

随着上下文增长，KV Cache 和相关显存带宽需求持续增加。

MHA、GQA、MQA 的设计差异，正是围绕 K/V 共享程度展开的。

最终，我希望自己能够真正建立下面这条工程直觉：

> **Long Context 昂贵，不只是因为模型需要处理更多 Token，而是因为标准 Dense Attention 需要处理随序列长度平方增长的 Token 交互。FlashAttention 通过优化计算过程与 IO 降低中间存储开销；而在自回归推理中，KV Cache 虽然避免了历史 K/V 的重复计算，却带来了随上下文长度线性增长的显存与带宽需求。MHA、GQA 和 MQA 的核心区别，就是如何通过不同程度的 K/V 共享，在模型质量与推理效率之间进行权衡。**

从理解 Attention 的数学公式，到理解 FLOPs、Memory、IO、KV Cache 和模型架构设计之间的关系，才算真正开始从训练工程的视角理解 Transformer。

---

## 参考文献

以下优先引用原始论文、正式会议论文以及官方技术文档。

**[1] Vaswani, A., et al. (2017). Attention Is All You Need.**

* 发表会议：NeurIPS 2017
* 论文地址：https://arxiv.org/abs/1706.03762
* 主要参考内容：Transformer 架构、Scaled Dot-Product Attention、Multi-Head Attention 及 Self-Attention 复杂度。

**[2] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.**

* 发表会议：NeurIPS 2022
* 论文地址：https://arxiv.org/abs/2205.14135
* 主要参考内容：FlashAttention、Tiling、IO-aware Algorithm、GPU HBM 与 SRAM 的数据访问优化。

**[3] Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need.**

* 论文地址：https://arxiv.org/abs/1911.02150
* 主要参考内容：Multi-Query Attention、增量解码中的 K/V 共享、推理阶段的 Memory Bandwidth 优化。

**[4] Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.**

* 发表会议：EMNLP 2023
* 论文地址：https://arxiv.org/abs/2305.13245
* ACL Anthology：https://aclanthology.org/2023.emnlp-main.298/
* 主要参考内容：Grouped-Query Attention、MHA/MQA/GQA 的关系，以及模型质量和推理效率之间的权衡。

**[5] Hugging Face Transformers Documentation. How Caching Works.**

* 官方文档：https://huggingface.co/docs/transformers/main/cache_explanation
* 主要参考内容：KV Cache 的工作原理、Cache Tensor Shape、Causal Attention 和自回归生成流程。

**[6] PyTorch Documentation. torch.nn.functional.scaled_dot_product_attention.**

* 官方文档：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
* 主要参考内容：Scaled Dot-Product Attention 的工程实现、优化 Kernel 选择及 GQA 支持。

---

*Day 16 学习完成：Transformer 不只是一个神经网络架构，也是一系列关于计算复杂度、显存容量、内存带宽与训练/推理效率的工程权衡。*
