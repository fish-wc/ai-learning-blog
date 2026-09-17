# LLM 训练中的交叉熵损失：从 Logits 到 Loss

在大语言模型训练中，我们经常会看到类似这样的代码：

```python
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),
    targets.flatten()
)
```

代码只有一行，但这一行背后实际上完成了：

```text
Logits
  ↓
Softmax
  ↓
得到每个 Token 的预测概率
  ↓
找到正确 Token 对应的概率
  ↓
计算 -log(p)
  ↓
对所有有效 Token 求平均
  ↓
得到最终 Loss
```

因此，如果想真正理解 LLM 是怎么训练的，就必须理解 **Cross Entropy Loss（交叉熵损失）**。

---

## 1. 模型首先输出的不是概率，而是 Logits

假设现在 Vocabulary 中只有 4 个 Token：

| Token ID | Token |
| -------- | ----- |
| 0        | A     |
| 1        | B     |
| 2        | C     |
| 3        | EOS   |

现在模型需要预测下一个 Token。

假设正确答案是：

```text
C
```

即：

```python
target = 2
```

模型经过 Transformer 后，输出：

```text
[0, ln2, ln4, 0]
```

约等于：

```text
[0, 0.693, 1.386, 0]
```

这些数并不是概率，而叫：

```text
Logits
```

可以暂时把 Logit 理解成：

> 模型对于每一个 Token 的“原始打分”。

Logit 越大，模型越倾向于选择这个 Token。

但是：

```text
[0, 0.693, 1.386, 0]
```

不能直接理解成：

```text
0%
69.3%
138.6%
0%
```

因为它还不是概率。

我们首先需要经过 Softmax。

---

# 2. Softmax：把 Logits 转换成概率

Softmax 的公式：

$$
p_i =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

其中：

* \(z_i\)：第 \(i\) 个 Token 的 Logit
* \(p_i\)：第 \(i\) 个 Token 对应的预测概率

对于刚才：

```text
Logits:

[0, ln2, ln4, 0]
```

首先进行指数运算：

$$
e^0=1
$$

$$
e^{\ln 2}=2
$$

$$
e^{\ln 4}=4
$$

$$
e^0=1
$$

因此：

```text
[1, 2, 4, 1]
```

然后求和：

$$
1+2+4+1=8
$$

因此概率分别为：

```text
A   = 1 / 8 = 0.125

B   = 2 / 8 = 0.250

C   = 4 / 8 = 0.500

EOS = 1 / 8 = 0.125
```

即：

| Token | Probability |
| ----- | ----------: |
| A     |       12.5% |
| B     |       25.0% |
| C     |       50.0% |
| EOS   |       12.5% |

其中正确答案：

```text
C
```

模型给出的概率是：

$$
P(C)=0.5
$$

---

# 3. Cross Entropy 到底在计算什么？

交叉熵的公式是：

$$
L
=
-\sum_i y_i\log p_i
$$

其中：

* \(y_i\)：真实标签
* \(p_i\)：模型预测概率

假设正确答案是：

```text
C
```

真实标签可以表示成 One-Hot：

```text
A    B    C    EOS

0    0    1     0
```

因此：

$$
y=[0,0,1,0]
$$

模型预测：

$$
p=[0.125,0.25,0.5,0.125]
$$

代入：

$$
L
=
-
(
0\log0.125
+
0\log0.25
+
1\log0.5
+
0\log0.125
)
$$

因为只有正确答案对应的位置：

```text
C
```

的 \(y_i=1\)，所以最终实际上只剩：

$$
L=-\log0.5
$$

计算得到：

$$
L\approx0.693
$$

所以：

> 对于单分类问题，Cross Entropy 本质上可以理解成：
>
> \(-\log(\text{正确答案的预测概率})\)

---

# 4. 为什么要取负对数？

理解这一点非常重要。

假设模型给正确答案的概率不同：

| 正确答案概率 \(p\) | \(-\ln(p)\) |
| -----------: | ----------: |
|         0.99 |       0.010 |
|         0.90 |       0.105 |
|         0.80 |       0.223 |
|         0.50 |       0.693 |
|         0.25 |       1.386 |
|         0.10 |       2.303 |
|         0.01 |       4.605 |

可以观察到：

```text
正确答案概率越高
        ↓
Loss 越小
```

例如：

```text
P(correct) = 99%

Loss ≈ 0.01
```

说明模型预测得非常好。

而：

```text
P(correct) = 1%

Loss ≈ 4.605
```

说明模型几乎完全没有预测正确。

因此 Cross Entropy 实际是在告诉模型：

> “真正正确的那个 Token，你到底给了它多大的概率？”

---

# 5. 为什么不用 `1 - p` 直接作为 Loss？

可能会有一个疑问：

如果正确答案概率是：

```text
0.8
```

为什么不用：

```text
Loss = 1 - 0.8 = 0.2
```

而非要使用：

$$
-\log(0.8)
$$

其中一个重要原因是：

**Log Loss 会对“非常自信但预测错误”的情况给予更大的惩罚。**

例如：

```text
正确答案概率 = 0.5

Loss = 0.693
```

但是：

```text
正确答案概率 = 0.01

Loss = 4.605
```

如果模型极度自信地把真正正确的答案概率压到接近 0：

```text
p \to 0
```

那么：

$$
-\log p \to +\infty
$$

惩罚会越来越大。

这使得模型不仅需要：

```text
预测正确
```

还需要：

```text
给正确答案足够高的概率
```

---

# 6. LLM 中不是只预测一个 Token

前面的例子只有一个目标：

```text
预测 C
```

但 GPT 的任务是：

> 对序列中的每一个位置预测下一个 Token。

例如：

```text
我 喜欢 学习 AI
```

Tokenize 后假设为：

```text
[我, 喜欢, 学习, AI]
```

训练时构造：

```text
Input:

[我, 喜欢, 学习]

Target:

[喜欢, 学习, AI]
```

对齐：

```text
Input          Target

我       →     喜欢

喜欢     →     学习

学习     →     AI
```

所以模型实际上需要计算三个 Loss：

$$
L_1
=
-\log P(\text{喜欢}|\text{我})
$$

$$
L_2
=
-\log P(\text{学习}|\text{我，喜欢})
$$

$$
L_3
=
-\log P(\text{AI}|\text{我，喜欢，学习})
$$

最终将它们进行平均：

$$
L
=
\frac{
L_1+L_2+L_3
}{3}
$$

这就是一条 Sequence 的语言模型 Loss。

---

# 7. 用一个具体 SFT 样本计算 Loss

假设 Instruction Fine-Tuning 数据：

```text
Instruction:
2 + 2 等于多少？

Response:
4
```

为了方便说明，假设经过 Tokenizer 后：

```text
[10, 11, 12, 13]
```

分别对应：

```text
10 → Instruction
11 → 2+2?
12 → Response
13 → 4
```

再添加：

```text
99 → EOS
```

得到：

```text
[10, 11, 12, 13, 99]
```

构造 Input 和 Target：

```text
Input:

[10, 11, 12, 13]


Target:

[11, 12, 13, 99]
```

模型实际上需要完成：

```text
10
↓
预测 11


10, 11
↓
预测 12


10, 11, 12
↓
预测 13


10, 11, 12, 13
↓
预测 99
```

也就是：

```text
Instruction
↓
2+2?


Instruction + 2+2?
↓
Response


Instruction + 2+2? + Response
↓
4


完整回答
↓
EOS
```

---

# 8. 假设模型给正确 Token 的概率如下

假设四个位置：

| Target     | 正确 Token 概率 |
| ---------- | ----------: |
| `2+2?`     |        0.50 |
| `Response` |        0.25 |
| `4`        |        0.80 |
| `EOS`      |        0.50 |

分别计算：

## 第一个位置

$$
L_1
=
-\ln0.5
=
0.693
$$

## 第二个位置

$$
L_2
=
-\ln0.25
=
1.386
$$

## 第三个位置

$$
L_3
=
-\ln0.8
=
0.223
$$

## 第四个位置

$$
L_4
=
-\ln0.5
=
0.693
$$

因此：

$$
L
=
\frac{
0.693+1.386+0.223+0.693
}{4}
$$

得到：

$$
L
\approx0.749
$$

这就是这条训练数据的平均 Token Loss。

---

# 9. 一个 Batch 的 Loss 又怎么计算？

假设：

```text
Batch Size = 2
```

有两条数据。

第一条：

```text
4 个有效 Target Token
```

对应 Loss：

```text
0.693
1.386
0.223
0.693
```

第二条：

```text
5 个有效 Target Token
```

Loss：

```text
1.386
0.693
0.693
0.223
0.693
```

那么整个 Batch 有：

```text
4 + 5 = 9
```

个有效 Token。

最终：

$$
L_{batch}
=
\frac{
\sum_{i=1}^{9}L_i
}{9}
$$

也就是：

> 默认情况下，语言模型的 Cross Entropy Loss 通常是对整个 Batch 中所有有效 Token 的 Loss 求平均。

并不是简单：

```text
Sample 1 Loss
+
Sample 2 Loss
----------------
       2
```

而更准确地说是：

```text
所有有效 Token 的 Loss 总和
─────────────────────────
所有有效 Token 的数量
```

---

# 10. Padding 为什么不能参与 Loss？

假设 Batch 中长度不一样：

```text
Sample 1:
[A, B, C, EOS]

Sample 2:
[D, E, EOS]
```

Padding 后：

```text
Sample 1:
[A, B, C, EOS]

Sample 2:
[D, E, EOS, PAD]
```

如果直接训练：

```text
E → EOS
EOS → PAD
```

模型就会被要求学习：

```text
预测 PAD
```

但是 PAD 只是为了让 Tensor Shape 一致，没有真实语义。

所以通常会修改 Target：

```text
[D, E, EOS, -100]
```

PyTorch：

```python
CrossEntropyLoss(ignore_index=-100)
```

会忽略：

```text
target = -100
```

的位置。

因此：

```text
真实 Token
→ 计算 Loss

Padding Token
→ 不计算 Loss
```

---

# 11. `-100` 本身不是一个 Token

这一点也需要注意。

```text
-100
```

不是 Vocabulary 中的特殊 Token。

它只是 PyTorch 中约定的一个：

```text
ignore_index
```

例如原始 Target：

```text
[11, 12, 13, 99, 99]
```

假设最后一个 `99` 是 Padding：

```text
[11, 12, 13, 99, -100]
```

Cross Entropy 计算：

```text
11   → 算 Loss
12   → 算 Loss
13   → 算 Loss
99   → 算 Loss
-100 → 忽略
```

其中第一个：

```text
99
```

是 EOS，所以需要训练模型学会：

```text
回答结束
↓
生成 EOS
```

而最后一个只是 Padding，所以忽略。

---

# 12. 为什么 PyTorch 不需要手动写 Softmax？

实际训练代码通常是：

```python
loss = torch.nn.functional.cross_entropy(
    logits,
    targets
)
```

而不是：

```python
probabilities = torch.softmax(logits)

loss = torch.nn.functional.cross_entropy(
    probabilities,
    targets
)
```

原因是：

> `CrossEntropyLoss` 直接接收 Raw Logits。

它内部已经完成了与：

```text
LogSoftmax
+
Negative Log Likelihood
```

等价的计算。

所以正确写法通常是：

```python
logits = model(inputs)

loss = F.cross_entropy(
    logits,
    targets
)
```

不要提前调用：

```python
softmax()
```

否则反而是不正确的使用方式。

---

# 13. LLM 的 Logits Shape 是什么？

假设：

```text
Batch Size = B
Sequence Length = T
Vocabulary Size = V
```

模型输入：

```text
inputs.shape
=
[B, T]
```

经过 Transformer：

```python
logits = model(inputs)
```

输出：

```text
logits.shape
=
[B, T, V]
```

为什么？

因为：

```text
B
↓
Batch 中有 B 条数据


T
↓
每条数据有 T 个 Token 位置


V
↓
每个位置都需要预测 Vocabulary 中
所有 Token 的概率
```

所以：

```text
[B, T, V]
```

可以理解成：

```text
每一条数据
 ×
每一个 Token 位置
 ×
整个 Vocabulary 的预测分数
```

---

# 14. 为什么代码中要 Flatten？

常见代码：

```python
loss = F.cross_entropy(
    logits.flatten(0, 1),
    targets.flatten()
)
```

原来：

```text
logits:

[B, T, V]
```

Flatten 后：

```text
[B × T, V]
```

Target：

```text
[B, T]
```

Flatten：

```text
[B × T]
```

于是问题就变成：

```text
一共有 B × T 个分类问题

每个分类问题：
从 V 个 Token 中
选出正确的那个 Token
```

例如：

```text
B = 2
T = 5
V = 100
```

原来：

```text
logits.shape = [2, 5, 100]
```

Flatten：

```text
[10, 100]
```

Target：

```text
[2, 5]
```

Flatten：

```text
[10]
```

也就是说：

```text
一共有 10 个 Token 预测位置

每个位置：
100 选 1
```

其中如果一个 Target 是：

```text
-100
```

那么这个位置就会被忽略。

---

# 15. Cross Entropy 如何推动模型学习？

假设某一个位置：

```text
正确答案 = C
```

模型预测：

```text
A    0.125
B    0.250
C    0.500
EOS  0.125
```

真实标签：

```text
[0, 0, 1, 0]
```

对于：

```text
Softmax + Cross Entropy
```

有一个非常重要的梯度结果：

$$
\frac{\partial L}{\partial z_i}
=
p_i-y_i
$$

因此：

```text
A:

0.125 - 0
= +0.125


B:

0.250 - 0
= +0.250


C:

0.500 - 1
= -0.500


EOS:

0.125 - 0
= +0.125
```

也就是：

```text
Gradient:

[+0.125, +0.250, -0.500, +0.125]
```

Optimizer 更新参数：

$$
\theta_{new}
=
\theta_{old}
-
lr \cdot
\frac{\partial L}{\partial\theta}
$$

因为正确 Token `C` 对应的梯度是负数：

```text
-0.500
```

减去负数：

```text
C 的 Logit ↑
```

而错误 Token：

```text
A、B、EOS
```

梯度为正：

```text
它们的 Logit ↓
```

最终产生：

```text
正确 Token Logit ↑
        ↓
正确 Token Probability ↑
        ↓
Cross Entropy Loss ↓
```

这就是模型训练最核心的过程。

---

# 16. 为什么训练过程中 Loss 会逐渐下降？

假设一开始：

```text
正确 Token Probability = 0.10
```

那么：

$$
Loss=-\ln0.1
\approx2.303
$$

经过梯度下降：

```text
正确 Token Probability
0.10
↓
0.30
↓
0.50
↓
0.80
↓
0.95
```

对应：

```text
Loss

2.303
↓
1.204
↓
0.693
↓
0.223
↓
0.051
```

所以训练日志中的：

```text
Loss = 4.2
Loss = 3.1
Loss = 2.0
Loss = 1.1
Loss = 0.7
```

本质上说明：

> 模型正在逐渐给训练数据中的正确 Token 分配更高的概率。

---

# 17. Cross Entropy 的完整训练链路

最终，可以把整个过程串起来：

```text
Token IDs
   ↓

Transformer
   ↓

Logits
[B, T, V]
   ↓

Softmax
   ↓

每个位置的 Vocabulary Probability
   ↓

找到 Target 对应 Token 的概率
   ↓

-log(P(correct token))
   ↓

忽略 Target = -100 的位置
   ↓

所有有效 Token 求平均
   ↓

Batch Loss
   ↓

loss.backward()
   ↓

计算：

∂Loss
──────
∂θ
   ↓

optimizer.step()
   ↓

更新模型参数
   ↓

提高正确 Token 的概率
   ↓

Loss 下降
```

---

# 18. 最核心的数学公式

如果模型正确答案为：

$$
y
$$

模型预测该 Token 的概率为：

$$
P(y)
$$

那么单个 Token 的 Loss：

$$
\boxed{
L=-\log P(y)
}
$$

对于一个长度为 \(T\) 的 Sequence：

$$
\boxed{
L
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(y_t|y_{<t})
}
$$

如果存在 Padding，只对有效位置计算：

$$
\boxed{
L
=
-\frac{1}{|\mathcal V|}
\sum_{t\in\mathcal V}
\log
P(y_t|y_{<t})
}
$$

其中：

$$
\mathcal V
=
\{t\mid y_t\neq-100\}
$$

---

# 19. 一句话理解 Cross Entropy

如果只记住一句话：

> **LLM 中的 Cross Entropy Loss，本质上就是看模型给“真正正确的下一个 Token”分配了多大的概率；正确 Token 的概率越高，Loss 越小。**

进一步可以记成：

```text
正确 Token Probability ↑

        ↓

-log(P) ↓

        ↓

Loss ↓
```

训练的过程，本质上就是不断调整模型参数，使：

```text
P(correct next token)
```

越来越高。

---

# 20. 总结

理解 LLM 中的交叉熵，可以按照下面几个步骤逐层理解：

```text
① Transformer 输出 Logits

② Softmax 把 Logits 转成概率

③ 找到正确 Target Token 对应的概率

④ 计算：

   -log(P(correct token))

⑤ 对所有有效 Token 的 Loss 求平均

⑥ Padding 对应的 -100 不参与 Loss

⑦ loss.backward() 计算梯度

⑧ optimizer.step() 更新参数

⑨ 正确 Token 的预测概率提高

⑩ 下一轮 Loss 下降
```

因此：

```python
loss = F.cross_entropy(logits, targets)
```

虽然只有一行代码，但它连接了整个语言模型训练过程中非常关键的几步：

```text
模型预测
→ 概率分布
→ 衡量错误程度
→ 计算梯度
→ 更新参数
```

而真正理解 Cross Entropy 之后，再去看 SFT、Pretraining、Response-only Loss、Padding Mask 等概念，就会发现：

> 它们底层其实都围绕同一个问题展开：**哪些 Token 应该被当成正确答案，并参与这个 `-log P(correct token)` 的计算。**
