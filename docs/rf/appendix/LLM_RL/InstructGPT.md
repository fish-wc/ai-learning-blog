# Day 8｜InstructGPT 详解：从 SFT、Reward Model 到 PPO，真正理解 RLHF 的完整训练流程

> **学习主题**：InstructGPT / RLHF / Reward Model / Preference Learning / PPO / Reference Model / KL Divergence
>
> **核心论文**：Training language models to follow instructions with human feedback（Ouyang et al., 2022）
>
> **本文目标**：不把 InstructGPT 当成几个算法名词的组合，而是从问题出发，理解为什么需要每个训练阶段、每个模型分别承担什么职责，以及整套方法如何将人类偏好转化为语言模型的行为。

---

## 一、引言：为什么需要 InstructGPT？

在学习大语言模型时，我们已经知道，GPT 类模型的预训练目标是 Next Token Prediction（预测下一个 Token）。

给定一段文本：

> 北京是中国的……

模型尝试预测下一个 Token，可能是“首都”。

在大量文本上训练之后，模型逐渐掌握语言规律、知识关联和一些推理能力。

但是，这里存在一个根本问题：

**一个善于预测下一个 Token 的模型，不一定是一个善于理解用户意图、遵循用户指令的助手。**

例如，用户提出：

> 请用一个生活中的例子，向没有机器学习基础的人解释什么是过拟合。

一个经过预训练的模型可能输出：

**回答 A：**

> 过拟合是指机器学习模型在训练数据上表现优异，但由于泛化能力不足，在测试数据上表现不佳的现象。

这个回答并没有明显的事实错误，但是它并没有很好地遵循“没有机器学习基础”和“生活中的例子”这两个要求。

另一个回答可能是：

**回答 B：**

> 你可以把过拟合想象成一个只会背答案的学生。
>
> 他把往年试卷的答案背得滚瓜烂熟，遇到原题就能拿高分。但考试稍微换一种问法，他就不会做了。
>
> 机器学习中的过拟合也是类似的情况：模型过于依赖训练数据中的细节，却没有学会能够应用到新数据上的规律。

显然，对于这个用户，回答 B 更符合需求。

我们希望语言模型不仅能够生成合理的文字，还能理解：

* 用户真正希望完成什么任务；
* 什么样的回答对当前用户更有帮助；
* 如何遵守用户提出的格式、风格和内容要求；
* 如何避免虚构事实、不必要的有害内容等问题。

这就是 InstructGPT 所要解决的核心问题之一：**如何让语言模型的行为更符合人类意图？**

OpenAI 在 2022 年的 InstructGPT 论文中，提出了一套通过人类反馈对语言模型进行微调的方法。

它使用的核心技术是：

**RLHF（Reinforcement Learning from Human Feedback），即基于人类反馈的强化学习。**

本文重点不是分析论文的 Benchmark，而是理解它背后的完整训练机制。

---

## 二、先从全局理解 InstructGPT

### 2.1 整体训练流水线

如果把大语言模型的预训练也计算在内，InstructGPT 可以概括为：

$$
\boxed{
\text{Pretrain}
\rightarrow
\text{SFT}
\rightarrow
\text{Reward Model}
\rightarrow
\text{PPO}
}
$$

不过需要说明：这个公式表达的是学习和训练的主要顺序，并不意味着每一步都简单地将上一个模型的输出作为下一个模型的输入。

实际上，Reward Model 和 PPO Policy 是两条相互配合的训练分支。

整体结构可以理解为：

```text
                  Pretrained Language Model
                              |
                              v
                     SFT (监督微调)
                              |
                         SFT Model
                              |
           +------------------+------------------+
           |                                     |
           v                                     v
    收集模型生成的回答                       初始化 Policy
           |                                     |
           v                                     |
       人类偏好排序                                |
           |                                     |
           v                                     v
    训练 Reward Model                       PPO 优化
           |                                     ^
           |                                     |
           +---------> 提供 Reward --------------+
                                                 |
    Frozen Reference Model ----> KL Penalty ------+
                                                 |
                                                 v
                                           更新 Policy
                                                 |
                                                 v
                                         InstructGPT
```

我们可以先用一句话概括每个阶段。

| 阶段             | 核心目标       | 解决的问题                  |
| -------------- | ---------- | ---------------------- |
| Pretrain       | 学习语言与知识    | 模型是否具备基础语言能力           |
| SFT            | 模仿高质量人工回答  | 模型是否知道应该如何遵循指令         |
| Reward Model   | 学习人类偏好     | 如何判断两个回答哪个更好           |
| PPO            | 根据奖励优化生成策略 | 如何让模型更倾向于生成高质量回答       |
| Reference + KL | 约束策略偏移     | 如何避免模型为了获得高奖励而偏离原有合理行为 |

这里最重要的思想是：

> **SFT 负责教模型如何回答，Reward Model 负责评价回答，PPO 负责根据评价改进回答，Reference Model 和 KL 负责约束优化过程。**

下面逐步拆解。

---

## 三、阶段一：Pretrain——模型为什么还不够像一个助手？

### 3.1 预训练到底在优化什么？

对自回归语言模型来说，给定一个 Token 序列：

$$
x_1,x_2,\ldots,x_T
$$

模型通过前面的 Token 预测下一个 Token。

其训练损失通常写作：

$$
\mathcal{L}_{pretrain}
=
-\sum_{t=1}^{T}
\log P_\theta(x_t\mid x_{<t})
$$

其中：

* \(\theta\)：模型参数；
* \(x_t\)：当前位置的目标 Token；
* \(x_{<t}\)：当前位置之前的 Token；
* \(P_\theta\)：模型预测的条件概率分布。

训练目标就是：

**提高真实文本中下一个 Token 的预测概率。**

随着训练数据不断增加，模型会学习许多有价值的能力。

但是，预测文本和遵循指令并不是完全相同的任务。

例如，用户问：

> 什么是梯度下降？

预训练模型可能学到很多种文本形式：

* 教科书中的数学定义；
* 论坛中的技术讨论；
* 论文中的公式推导；
* 问答网站中的简单解释；
* 代码和注释。

这些都是可能的续写模式。

但用户真正想要的是哪一种？

预训练目标本身并没有明确告诉模型：

> 当前正在和一个用户交互，你应该优先满足用户的任务需求。

因此，我们需要进一步调整模型的行为。

### 3.2 一个需要明确的认识

这并不意味着预训练模型完全不会遵循指令。

经过充分预训练，模型可能已经具备相当多的理解和执行指令的能力。

问题在于，这些能力不一定能够稳定地被用户调用。

因此，InstructGPT 并不是从零开始教模型所有能力，而是通过后续训练，使模型更稳定地表现出符合用户需求的行为。

---

## 四、阶段二：SFT——先告诉模型什么是好的回答

SFT 的全称是：

**Supervised Fine-Tuning，监督微调。**

### 4.1 SFT 训练数据是什么？

SFT 使用人工编写的高质量示范数据。

每条数据通常包含：

$$
(x,y^*)
$$

其中：

* \(x\)：用户输入的 Prompt；
* \(y^*\)：人工编写的示范回答。

例如：

**Prompt：**

> 请用小学生能够理解的方式解释什么是神经网络。

**Human Demonstration：**

> 你可以把神经网络想象成一个正在学习认东西的小朋友。
>
> 一开始，他可能分不清猫和狗。但是看过很多图片之后，他会逐渐注意到耳朵、脸型等特征，学会更好地区分它们。

模型通过这些示范学习：

> 当用户提出这样的要求时，一个好的助手通常应该如何回答。

### 4.2 SFT 的训练目标

SFT 仍然使用监督学习。

给定 Prompt 和示范回答，希望模型提高生成该回答的概率。

因此，损失函数可以写成：

$$
\mathcal{L}_{SFT}(\theta)
=
-\mathbb{E}_{(x,y^*)\sim D_{SFT}}
\left[
\log \pi_\theta(y^*\mid x)
\right]
$$

其中：

* \(D_{SFT}\)：监督微调数据集；
* \(\pi_\theta\)：待训练的语言模型；
* \(y^*\)：人工示范回答。

将序列概率展开，可以得到：

$$
\log \pi_\theta(y^*\mid x)
=
\sum_{t=1}^{T}
\log \pi_\theta(y_t^*\mid x,y_{<t}^*)
$$

也就是说：

**SFT 本质上还是在进行 Token 级别的概率学习，只是训练数据变成了符合指令要求的高质量回答。**

### 4.3 为什么有了 SFT 还需要 RLHF？

这是理解 InstructGPT 的第一个关键问题。

SFT 的核心是：

> 让模型模仿示范回答。

但开放式语言任务往往没有唯一正确答案。

例如：

> 向初学者解释什么是 KL Divergence。

我们可能得到几种回答：

**回答 A：**

> KL 散度用于衡量两个概率分布之间的差异。

**回答 B：**

> KL 散度的定义为：

$$
D_{KL}(P\|Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

**回答 C：**

> 假设你原本认为一枚硬币正反面出现的概率相同，但另一位朋友认为这枚硬币更容易出现正面。KL 散度可以用来描述：如果真实情况遵循你的概率判断，而你却使用朋友的判断，会产生多少额外的信息代价。

对于初学者来说，回答 C 可能更容易理解。

但需要注意：

**A、B、C 不一定存在绝对意义上的正确与错误。**

它们只是对于当前任务的适用程度不同。

传统 SFT 面临两个问题：

1. 人类需要为每个 Prompt 编写高质量示范，数据收集成本较高。
2. 当存在多个合理答案时，单纯模仿一个示范回答，无法充分利用“哪个回答更好”的相对偏好信息。

因此，我们希望引入另一种监督信号：

> 不要求人类每次都写出理想答案，而是让人类比较多个候选回答，判断哪个更好。

这就是 Preference Learning 的起点。

---

## 五、阶段三：Preference Pair——如何把人类偏好变成训练数据？

### 5.1 什么是 Preference Pair？

Preference Pair 指一组带有偏好关系的回答。

给定一个 Prompt：

$$
x
$$

模型生成两个候选回答：

$$
y_1,\quad y_2
$$

由人类标注者比较它们的质量。

假设人类更喜欢 \(y_1\)，我们可以记录：

$$
y_1 \succ y_2
$$

这里的 \(\succ\) 表示“更受偏好”。

为了便于训练，一般使用：

$$
\boxed{
(x,y_w,y_l)
}
$$

其中：

* \(x\)：Prompt；
* \(y_w\)：Preferred / Chosen Response；
* \(y_l\)：Rejected Response。

\(w\) 表示 winner，\(l\) 表示 loser。

例如：

```text
Prompt:
请向小学生解释什么是过拟合。

Chosen:
就像一个只会背考试答案的学生，
遇到新题目时可能就不会做了。

Rejected:
过拟合是经验风险最小化过程中，
模型泛化误差增加的现象。
```

这里并不是说 Rejected 一定是错误的。

我们只能说：

> 对于这个 Prompt，Chosen 比 Rejected 更符合标注时采用的评价标准。

这是理解偏好数据非常重要的一点。

**Preference Pair 表达的是相对偏好，而不是绝对真理。**

### 5.2 InstructGPT 实际如何收集偏好？

InstructGPT 不只是要求标注员比较两个回答。

根据原论文第 3.5 节，标注员会对同一个 Prompt 的多个回答进行排序，候选数量 \(K\) 通常为 4 到 9。

例如，同一个问题生成四个回答：

$$
A,\ B,\ C,\ D
$$

人类排序结果为：

$$
A\succ C\succ B\succ D
$$

那么可以构造以下偏好对：

$$
(A,C)
$$

$$
(A,B)
$$

$$
(A,D)
$$

$$
(C,B)
$$

$$
(C,D)
$$

$$
(B,D)
$$

因此，当排序中不存在并列时，\(K\) 个候选回答最多可以产生：

$$
\binom{K}{2}
=
\frac{K(K-1)}{2}
$$

个 Preference Pair。

这里还存在一个值得关注的训练细节。

同一个 Prompt 产生的偏好对往往高度相关，因为不同 Pair 可能反复使用同一个回答。

InstructGPT 的作者发现，如果将这些 Pair 完全打散进行训练，容易造成 Reward Model 过拟合。

因此，他们将同一次排序产生的比较关系放在一起处理，提高了训练效率，也改善了模型的泛化表现。

这说明：

**Preference 数据不仅要关注数量，还需要关注数据之间的相关性。**

---

## 六、Reward Model：让机器学习人类喜欢什么样的回答

现在我们已经有了 Preference Pair。

接下来的问题是：

> 如何让计算机自动判断一个回答是否符合人类偏好？

这就是 Reward Model 的作用。

### 6.1 Reward Model 是什么？

Reward Model（RM，奖励模型）可以表示为：

$$
\boxed{
r_\theta(x,y)\rightarrow \mathbb{R}
}
$$

输入：

$$
\text{Prompt + Response}
$$

输出：

$$
\text{Scalar Reward}
$$

即一个标量分数。

例如，对于同一个 Prompt：

| 候选回答       | RM 输出的示例分数 |
| ---------- | ---------: |
| Response A |        2.5 |
| Response B |        0.8 |
| Response C |       -1.2 |

这些分数是用于解释原理的假设数据。

如果模型学到了合理的偏好关系，我们希望：

$$
r_\theta(x,A)
>
r_\theta(x,B)
>
r_\theta(x,C)
$$

需要特别注意：

**RM 不是一个能够准确衡量回答绝对质量的万能评分器。**

它学习的是训练数据中的人类偏好规律。

某个回答得到 2.5 分，并不意味着其真实质量具有一个客观的“2.5”刻度。

在这种 Pairwise Reward Modeling 中，回答之间的分数差异通常比单个分数的绝对数值更重要。

### 6.2 Reward Model 的模型结构

从架构角度看，可以把 RM 理解为一个经过改造的语言模型。

普通语言模型：

```text
Prompt
   |
Transformer
   |
LM Head
   |
Next Token Logits
```

Reward Model：

```text
Prompt + Response
        |
    Transformer
        |
    Reward Head
        |
   Scalar Reward
```

具体实现中，可以利用 Transformer 对输入序列提取的表示，通过一个 Reward Head 输出分数。

InstructGPT 论文第 3.5 节将其概括为从 SFT 模型去掉语言建模输出层，改为输出标量 Reward。

但有一个容易忽略的论文实现细节：附录 C.2 指出，实验最终采用的 6B RM 实际初始化自经过多个公开 NLP 任务微调的 GPT-3 模型。作者同时发现，从 GPT-3 或 SFT 模型初始化也能取得相近结果。

因此，不应该认为：

> RM 必须与 Policy 使用完全相同的参数初始化。

真正关键的是：

**RM 学习的是人类偏好，而不是继续预测下一个 Token。**

### 6.3 Reward Model 怎么训练？

这是本文最重要的公式之一。

我们已经有了一条偏好数据：

$$
(x,y_w,y_l)
$$

人类标注结果为：

$$
y_w\succ y_l
$$

Reward Model 分别输出：

$$
r_\theta(x,y_w)
$$

$$
r_\theta(x,y_l)
$$

希望满足：

$$
r_\theta(x,y_w)>r_\theta(x,y_l)
$$

但仅仅写出这个不等式，还不能直接进行梯度优化。

因此，需要将 Reward 差值映射为偏好概率。

InstructGPT 使用了基于 Bradley-Terry 模型的成对偏好建模方法。

定义：

$$
P_\theta(y_w\succ y_l\mid x)
=
\frac{
e^{r_\theta(x,y_w)}
}{
e^{r_\theta(x,y_w)}
+
e^{r_\theta(x,y_l)}
}
$$

将分子分母同时除以 \(e^{r_\theta(x,y_w)}\)：

$$
P_\theta(y_w\succ y_l\mid x)
=
\frac{1}{
1+e^{-(r_\theta(x,y_w)-r_\theta(x,y_l))}
}
$$

这正是 Sigmoid 函数：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

因此：

$$
\boxed{
P_\theta(y_w\succ y_l\mid x)
=
\sigma\left(
r_\theta(x,y_w)-r_\theta(x,y_l)
\right)
}
$$

这个公式非常值得理解。

它并不是直接预测一个回答绝对有多好，而是在预测：

> 在给定 Prompt 的情况下，模型认为人类更可能选择哪个回答？

### 6.4 从偏好概率推导 RM Loss

既然人类已经告诉我们：

$$
y_w\succ y_l
$$

那么我们希望模型预测这一偏好事件的概率越高越好。

因此采用负对数似然：

$$
\mathcal{L}_{RM}(\theta)
=
-\log P_\theta(y_w\succ y_l\mid x)
$$

代入刚才的概率表达式：

$$
\boxed{
\mathcal{L}_{RM}(\theta)
=
-\log\sigma\left(
r_\theta(x,y_w)-r_\theta(x,y_l)
\right)
}
$$

这就是 Reward Model 的核心训练损失。

对数据集取期望，可写为：

$$
\mathcal{L}_{RM}(\theta)
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log\sigma(
r_\theta(x,y_w)-r_\theta(x,y_l)
)
\right]
$$

InstructGPT 原论文 Equation (1) 还考虑了同一个 Prompt 下多个候选回答形成的全部 Pair，对这些比较损失进行平均。

### 6.5 不背公式，真正理解这个 Loss

定义 Reward 差值：

$$
\Delta r=r_w-r_l
$$

现在分三种情况讨论。

**情况一：两个回答得分相同**

$$
r_w=r_l
$$

则：

$$
\Delta r=0
$$

$$
P(y_w\succ y_l)=\sigma(0)=0.5
$$

模型没有表现出明确的偏好。

**情况二：Winner 得分更高**

$$
r_w>r_l
$$

则：

$$
\Delta r>0
$$

$$
\sigma(\Delta r)>0.5
$$

模型预测的偏好方向与人类标注一致。

随着 Reward 差值增大，偏好概率进一步提高，Loss 会下降。

**情况三：Loser 得分更高**

$$
r_w<r_l
$$

则：

$$
\Delta r<0
$$

$$
\sigma(\Delta r)<0.5
$$

说明模型把人类不喜欢的回答排在了前面。

这时 Loss 会比较大。

梯度下降将推动模型修正分数关系。

为了更清楚地理解梯度方向，对 Loss 关于 \(\Delta r\) 求导：

$$
\frac{\partial\mathcal{L}_{RM}}
{\partial\Delta r}
=
\sigma(\Delta r)-1
$$

由于 Sigmoid 的输出处于 0 和 1 之间，因此：

$$
\frac{\partial\mathcal{L}_{RM}}
{\partial\Delta r}<0
$$

在最小化 Loss 时，优化过程会倾向于增大 \(\Delta r\)。

也就是：

$$
r_w-r_l\uparrow
$$

最终使模型更倾向于给人类偏好的回答更高的分数。

**这就是 Reward Model 如何从 Preference Pair 中学习人类偏好的核心机制。**

### 6.6 Reward Model 学到的究竟是什么？

现在可以总结：

```text
Human Preference
       |
       v
Preference Pairs
       |
       v
Reward Model Training
       |
       v
A Learned Reward Function
```

原本无法直接参与梯度优化的人类主观判断，被转化成了一个可计算的标量奖励函数。

因此：

$$
\boxed{
\text{Human Preference}
\rightarrow
\text{Learned Reward Function}
}
$$

但这里必须强调：

**Reward Model 是人类偏好的近似模型，不是人类偏好本身。**

这一点将直接引出后面 Reference Model 和 KL 的必要性。

---

## 七、为什么训练完 Reward Model，还需要 PPO？

现在我们已经有了一个 RM。

对于任意：

$$
(x,y)
$$

它都可以输出：

$$
r_\theta(x,y)
$$

那么问题来了：

> 既然 RM 已经知道什么回答更好，为什么不直接使用 RM？

因为 Reward Model 主要负责评价，而不是生成。

我们希望最终得到的是一个能够生成高质量回答的语言模型。

可以把两者想象成：

* **Reward Model 是裁判。**
* **Policy Model 是选手。**

裁判知道什么表现更好，并不意味着选手已经具备相应的能力。

我们还需要一个过程，让选手根据裁判的反馈不断改进。

这就是强化学习阶段的作用。

### 7.1 能不能直接生成多个答案，再让 RM 选择？

当然可以。

例如：

```text
User Prompt
    |
    v
Generate N Responses
    |
    v
Reward Model Scores
    |
    v
Choose the Best Response
```

这类思路通常称为 Best-of-N。

但是，如果每次请求都需要生成大量候选答案并进行评分，就会增加推理计算开销和延迟。

PPO 则尝试在训练阶段，将 Reward Model 所表达的偏好逐渐融入生成模型的参数。

让模型未来不需要每次都生成大量候选答案，也更有可能直接生成高质量回答。

因此：

> **Reward Model 负责提供优化信号，PPO 负责利用这个信号更新生成模型。**

---

## 八、PPO：如何让语言模型主动生成更好的回答？

PPO 的全称是：

**Proximal Policy Optimization，近端策略优化。**

它是一种 Policy Gradient 强化学习算法。

这里先不深入推导所有 PPO 数学细节，而是重点理解它在 InstructGPT 中扮演的角色。

### 8.1 将文本生成理解为强化学习

传统强化学习可以抽象为：

```text
Agent
  |
  v
Choose Action
  |
  v
Environment
  |
  v
Receive Reward
  |
  v
Update Policy
```

对应到语言模型：

```text
Language Model (Policy)
          |
          v
     Generate Tokens
          |
          v
    Complete Response
          |
          v
      Reward Model
          |
          v
        Reward
          |
          v
      PPO Update
```

在这个过程中：

| 强化学习概念  | 语言模型中的含义               |
| ------- | ---------------------- |
| Agent   | 语言模型                   |
| State   | Prompt + 当前已生成的 Token  |
| Action  | 选择下一个 Token            |
| Policy  | 语言模型的 Token 概率分布       |
| Reward  | RM 对生成结果的评价，并可结合 KL 惩罚 |
| Episode | 对一个 Prompt 完成一次回答生成    |

定义状态：

$$
s_t=(x,y_{<t})
$$

动作：

$$
a_t=y_t
$$

Policy：

$$
\pi_\phi(a_t\mid s_t)
=
\pi_\phi(y_t\mid x,y_{<t})
$$

其中 \(\phi\) 表示当前可训练 Policy 的参数。

完整回答为：

$$
y=(y_1,y_2,\ldots,y_T)
$$

Reward Model 对完整回答给出：

$$
r_\theta(x,y)
$$

InstructGPT 将这一交互描述为一个以完整回答为单位给出结果的 Bandit 环境。工程上仍然可以把生成过程展开为 Token 级决策，并利用逐 Token 的 KL 惩罚和 Value Model 来进行 PPO 优化。

### 8.2 PPO 想优化什么？

暂时忽略 KL 等额外约束，最直观的目标是：

$$
\boxed{
\max_\phi
\mathbb{E}_{x\sim D,\,
y\sim\pi_\phi(\cdot\mid x)}
[r_\theta(x,y)]
}
$$

含义是：

> 对于训练 Prompt，调整 Policy 参数，让模型生成的回答获得更高的预期 Reward。

需要特别注意：

这里的 \(y\) 不是人工写好的标准答案，而是当前 Policy 自己采样生成的回答。

因此，PPO 与 SFT 的训练过程存在明显区别。

**SFT：**

```text
Human-written Response
         |
         v
   Supervised Loss
         |
         v
   Update Policy
```

**PPO：**

```text
Policy Generates Response
         |
         v
    Reward Model
         |
         v
    Reward Signal
         |
         v
    Policy Gradient
         |
         v
    Update Policy
```

在 PPO 阶段，并不是每次生成回答后都让人类重新评分。

通常是先用人工偏好数据训练 RM，再在强化学习过程中使用这个 RM 提供自动化奖励信号。

### 8.3 为什么 PPO 还需要 Value Model？

这里需要补充一个容易被忽略的角色：

**Value Model（价值模型）。**

Reward Model 主要评价一个完整的 Prompt-Response。

而 Value Model 试图预测：

> 从当前生成状态继续下去，预计能够获得多少 Return？

可以表示为：

$$
V_\psi(s_t)
$$

PPO 使用 Advantage 来判断一个动作相对于基线表现得如何。

直观地看：

$$
A_t\approx G_t-V_\psi(s_t)
$$

其中：

* \(G_t\)：从当前时刻计算的 Return；
* \(V_\psi(s_t)\)：Value Model 预测的 Return；
* \(A_t\)：Advantage。

如果：

$$
A_t>0
$$

说明这个动作的结果比价值基线预期的更好。

如果：

$$
A_t<0
$$

说明它相对较差。

实际 PPO 可以采用 GAE 等更复杂的 Advantage 估计方法。

这里先记住：

* Reward Model：负责提供对生成结果的奖励评价。
* Value Model：负责估计状态价值，帮助 Policy 更稳定地学习。

InstructGPT 的实现中，Value Function 从 RM 初始化，但二者在强化学习中的职责并不相同。

### 8.4 PPO 为什么叫 Proximal？

PPO 希望利用 Reward 更新模型，但不希望一次参数更新导致策略变化过大。

一个经典的 PPO-Clip 目标是：

$$
\mathcal{J}_{clip}(\phi)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\phi)A_t,\,
\operatorname{clip}
(\rho_t(\phi),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

其中：

$$
\rho_t(\phi)
=
\frac{
\pi_\phi(a_t\mid s_t)
}{
\pi_{old}(a_t\mid s_t)
}
$$

这里涉及两个策略：

* \(\pi_\phi\)：正在优化的 Policy；
* \(\pi_{old}\)：采样数据时使用的旧 Policy。

PPO-Clip 通过截断目标函数中概率比率的作用，限制过大的策略更新收益，从而改善训练稳定性。

需要注意：**PPO clipping 并不意味着实际策略变化被严格限制在某个固定范围内。** 它是一种优化目标层面的近似约束。

到这里，我们似乎已经得到完整流程：

$$
\text{SFT}
\rightarrow
\text{RM}
\rightarrow
\text{PPO}
$$

但接下来会出现 RLHF 最值得警惕的问题。

---

## 九、Reward Hacking：为什么不能只追求更高的 Reward？

假设我们直接优化：

$$
\max_\phi
\mathbb{E}[r_\theta(x,y)]
$$

直觉上：

> Reward 越高，回答应该越好。

但这个结论有一个隐藏前提：

$$
\text{Reward Model}
=
\text{Real Human Preference}
$$

事实上，这个等式通常不成立。

RM 只是从有限的人类标注数据中学习到的近似评价函数。

### 9.1 一个直观的例子

假设 Reward Model 在训练中观察到：

> 很多优质回答都比较详细。

它可能将回答长度与回答质量建立较强关联。

于是，Policy 在优化过程中可能发现：

> 只要我把答案写得越来越长，就容易获得更高的 Reward。

结果可能出现：

* 简单问题也输出很长的答案；
* 重复说明相同内容；
* 堆砌专业术语；
* 为获得高分而偏离用户真正需要的回答风格。

例如，用户只想知道：

> Python 中如何获取列表长度？

正常情况下，一句：

`使用 len(my_list) 即可。`

就足够了。

但某个存在偏差的 RM 可能偏好更冗长的解释。

于是 Policy 反复生成没有必要的详细内容。

最终：

$$
\text{RM Score}\uparrow
$$

却不一定意味着：

$$
\text{Human Satisfaction}\uparrow
$$

这种利用奖励函数缺陷获取高分的现象，属于：

**Reward Hacking（奖励投机）。**

更一般地说，持续优化不完美的 Reward Model，还可能导致：

**Reward Model Overoptimization（奖励模型过度优化）。**

### 9.2 为什么模型会主动寻找 RM 的漏洞？

因为优化过程只知道当前定义的目标函数。

假如目标是：

$$
\max_\phi r_\theta(x,y)
$$

那么只要某种行为可以提高 RM 分数，优化过程就可能逐步强化这种行为。

但 RM 所学到的规律不一定能够在所有可能回答上准确反映人类偏好。

尤其当 Policy 开始生成与 RM 训练数据差异很大的回答时，RM 的评价可能更加不可靠。

因此，我们面临新的问题：

> **如何允许 Policy 根据 Reward 改进，同时避免它为了追求 Reward 而发生过大的行为偏移？**

InstructGPT 的一个关键设计是：

$$
\boxed{
\text{Reference Model}+\text{KL Penalty}
}
$$

---

## 十、Reference Model：为什么还需要一个冻结的 SFT 模型？

### 10.1 Reference Model 是什么？

在 PPO 阶段，我们可以从 SFT 后的模型构造两个角色。

**Policy Model：**

$$
\pi_\phi
$$

这是需要继续优化的模型。

**Reference Model：**

$$
\pi_{ref}
$$

这是冻结的参考模型。

从训练开始后：

* Policy 的参数会通过 PPO 不断更新；
* Reference 的参数保持不变。

可以理解为：

```text
              SFT Model
                  |
          +-------+-------+
          |               |
          v               v
     Policy Model     Reference Model
      Trainable           Frozen
          |
          v
       PPO Update
```

为什么需要冻结的 Reference？

因为我们希望能够持续衡量：

> 当前 Policy 的生成行为相对于最初的 SFT Policy 发生了多大变化？

如果 Reference 也跟着 Policy 一起变化，那么比较基准本身就在移动，无法稳定地提供相对初始行为的约束。

### 10.2 RM 和 Reference Model 的本质区别

这是一个特别容易混淆的问题。

| 对比项      | Reward Model      | Reference Model              |
| -------- | ----------------- | ---------------------------- |
| 主要作用     | 评价回答的偏好质量         | 提供稳定的策略比较基准                  |
| 主要输入     | Prompt + Response | Prompt + 当前生成前缀              |
| 主要输出     | Scalar Reward     | Token 概率分布 / Log Probability |
| 在 PPO 阶段 | 通常冻结              | 冻结                           |
| 解决的问题    | 哪个回答更好            | 当前 Policy 偏离初始 Policy 多远     |

一句话概括：

> **Reward Model 告诉 Policy 应该往哪里优化，Reference Model 帮助约束它不要离原来的合理行为太远。**

但是 Reference Model 本身不会判断回答是否正确，也不是另一个独立的人类偏好裁判。

它提供的是一个概率分布基准。

而量化这种分布差异，正是 KL Divergence 的作用。

---

## 十一、KL Divergence：为什么它能约束 Policy？

KL Divergence 的全称是：

**Kullback-Leibler Divergence，KL 散度。**

### 11.1 KL 在衡量什么？

对于两个离散概率分布：

$$
P(x),Q(x)
$$

KL 散度定义为：

$$
\boxed{
D_{KL}(P\|Q)
=
\sum_x
P(x)\log\frac{P(x)}{Q(x)}
}
$$

它衡量的是：

> 当数据遵循分布 \(P\)，却使用分布 \(Q\) 进行描述时，会产生多少额外的期望对数代价？

从信息论角度看，它也是一种相对熵。

这里需要注意几个性质：

1. \(D_{KL}(P\|Q)\geq 0\)。
2. 当两个分布相同时，KL 为 0。
3. KL 通常不是对称的。
4. KL 不是严格意义上的数学距离。

也就是说：

$$
D_{KL}(P\|Q)
\neq
D_{KL}(Q\|P)
$$

一般成立。

因此，KL 可以衡量概率分布差异，但它不是普通的欧氏距离。

### 11.2 在 RLHF 中比较哪两个分布？

在 InstructGPT 中，我们关心：

$$
D_{KL}
\left(
\pi_\phi(\cdot\mid x)
\|
\pi_{ref}(\cdot\mid x)
\right)
$$

其中：

* \(\pi_\phi\)：当前 PPO Policy；
* \(\pi_{ref}\)：冻结的 Reference Policy。

它回答的问题是：

> 对于同一个 Prompt，当前模型生成各种回答的概率分布，相对于参考模型发生了多大的变化？

KL 越大，表示两者的分布偏移通常越明显。

所以，我们可以在奖励优化目标中增加 KL 惩罚。

### 11.3 RLHF 的 KL-Regularized Objective

一个常见的简化形式是：

$$
\boxed{
\max_\phi
\mathbb{E}_{x,\,
y\sim\pi_\phi(\cdot\mid x)}
\left[
r_\theta(x,y)
\right]
-
\beta
\mathbb{E}_{x}
\left[
D_{KL}
\left(
\pi_\phi(\cdot\mid x)
\|
\pi_{ref}(\cdot\mid x)
\right)
\right]
}
$$

其中：

$$
r_\theta(x,y)
$$

表示 Reward Model 的评分。

$$
D_{KL}(\pi_\phi\|\pi_{ref})
$$

表示 Policy 相对于 Reference 的策略差异。

而：

$$
\beta
$$

是 KL 惩罚系数。

所以整个目标可以理解为：

$$
\boxed{
\text{Optimization Objective}
=
\text{Reward}
-
\beta\cdot\text{KL}
}
$$

这一公式非常重要。

它同时表达了两个目标：

**第一：提高回答质量。**

通过提高 RM 的预期评分，让模型更符合人类偏好。

**第二：避免无约束的策略偏移。**

通过对偏离参考模型的行为施加成本，缓解 Reward Model 被过度利用的问题。

### 11.4 KL 惩罚系数 β 的作用

KL 惩罚并不是越大越好。

**当 \(\beta\) 很小时：**

Policy 可以更加自由地追求高 Reward。

但这种自由可能带来更大的策略偏移，也可能提高 Reward Hacking 的风险。

**当 \(\beta\) 很大时：**

Policy 更难偏离 Reference Model。

但与此同时，模型可能无法充分利用人类偏好信号改善行为。

因此，这里存在一种权衡：

$$
\boxed{
\text{Preference Optimization}
\quad\text{vs.}\quad
\text{Policy Deviation}
}
$$

KL 系数决定了两者之间的相对权重。

它不是越小越好，也不是越大越好，而是需要根据任务、Reward 质量和训练表现进行调节。

### 11.5 为什么 KL 惩罚能缓解 Reward Hacking？

我们可以从分布变化的角度理解。

Reward Model 通常是在一定数据分布上训练出来的。

例如，RM 在 SFT Policy 及相关 Policy 生成的回答上收集人类比较数据。

因此，它对这些数据附近的回答，往往具有更直接的训练依据。

但如果 PPO 不断提高 Reward，导致 Policy 生成与原始分布差异很大的回答：

$$
\pi_\phi
\rightarrow
\text{Different Response Distribution}
$$

那么 Reward Model 可能开始在缺少足够监督数据的区域做出预测。

这时，Policy 更有可能利用 RM 的错误偏好。

Reference KL 通过惩罚过大的分布偏移，帮助约束这种现象。

可以把它理解成：

> RM 是一个不完美的评分标准。我们希望模型按照它改进，但不希望模型为了获得高分而走到评分标准完全不可靠的区域。

需要注意：**KL 只是一种正则化手段，并不能从理论上保证完全杜绝 Reward Hacking。**

如果 RM 在 Reference 附近本身就存在偏差，或者约束强度不足，问题仍然可能发生。

---

## 十二、从公式再深入一步：为什么可以使用 Per-token KL？

这是从“听懂概念”走向“看懂 RLHF 训练实现”的关键。

InstructGPT 实际上采用了 Per-token KL Penalty。

为什么完整回答上的概率比值，可以转化成 Token 级别的计算？

### 12.1 自回归语言模型的概率分解

一个完整回答：

$$
y=(y_1,y_2,\ldots,y_T)
$$

在自回归模型中的概率为：

$$
\pi_\phi(y\mid x)
=
\prod_{t=1}^{T}
\pi_\phi(y_t\mid x,y_{<t})
$$

对两边取对数：

$$
\log\pi_\phi(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\phi(y_t\mid x,y_{<t})
$$

Reference Model 同理：

$$
\log\pi_{ref}(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_{ref}(y_t\mid x,y_{<t})
$$

因此，两者的 Log Probability Ratio 为：

$$
\log
\frac{
\pi_\phi(y\mid x)
}{
\pi_{ref}(y\mid x)
}
=
\sum_{t=1}^{T}
\log
\frac{
\pi_\phi(y_t\mid x,y_{<t})
}{
\pi_{ref}(y_t\mid x,y_{<t})
}
$$

于是我们可以定义每个 Token 上的采样 Log Ratio：

$$
k_t
=
\log\pi_\phi(y_t\mid x,y_{<t})
-
\log\pi_{ref}(y_t\mid x,y_{<t})
$$

那么完整回答对应的 Log Ratio 为：

$$
\boxed{
k(x,y)=\sum_{t=1}^{T}k_t
}
$$

这就是 Per-token KL Penalty 在实现层面的重要依据。

### 12.2 采样 Log Ratio 与真正 KL 的关系

这里有一个容易被忽略的数学细节。

对于固定 Prompt：

$$
\mathbb{E}_{y\sim\pi_\phi}
\left[
\log
\frac{\pi_\phi(y\mid x)}
{\pi_{ref}(y\mid x)}
\right]
=
D_{KL}
\left(
\pi_\phi(\cdot\mid x)
\|
\pi_{ref}(\cdot\mid x)
\right)
$$

也就是说：

**对当前 Policy 采样的回答取期望时，Log Probability Ratio 对应 KL 散度。**

但是：

单次采样得到的 \(k_t\) 或 \(k(x,y)\) 可能为负数。

这并不违反：

$$
D_{KL}\geq 0
$$

因为 KL 非负说的是分布上的期望，而不是每一个采样结果都非负。

### 12.3 加上 KL 后的奖励

对于一次采样得到的完整回答，可以构造如下奖励：

$$
\boxed{
R(x,y)
=
r_\theta(x,y)
-
\beta
\sum_{t=1}^{T}
\left[
\log\pi_\phi(y_t\mid x,y_{<t})
-
\log\pi_{ref}(y_t\mid x,y_{<t})
\right]
}
$$

其中：

* RM 对完整回答提供质量评价；
* Token 级 Log Ratio 提供策略偏移信号；
* \(\beta\) 控制惩罚强度。

因此，在实现时我们通常需要：

1. 使用当前 Policy 生成回答；
2. 计算生成 Token 在当前 Policy 下的 Log Probability；
3. 计算相同 Token、相同生成前缀在 Reference Policy 下的 Log Probability；
4. 计算相应的 KL 惩罚；
5. 将其与 RM Reward 结合；
6. 使用 PPO 更新 Policy。

这个过程解释了为什么 RLHF 训练系统中需要同时保留 Policy 和 Reference Model。

---

## 十三、为什么 PPO 已经有 Clipping，还需要 Reference KL？

这是一个非常值得准备的面试问题。

乍看之下：

* PPO Clipping：不希望 Policy 更新太大；
* Reference KL：不希望 Policy 偏移太大。

两者好像在做相同的事情。

其实它们约束的对象不同。

### 13.1 PPO Clipping 比较的是当前策略和旧策略

PPO 中存在：

$$
\rho_t(\phi)
=
\frac{
\pi_\phi(a_t\mid s_t)
}{
\pi_{old}(a_t\mid s_t)
}
$$

这里：

$$
\pi_{old}
$$

通常是生成当前训练数据时使用的策略快照。

PPO Clipping 主要用于控制当前一轮策略优化中的更新行为。

### 13.2 Reference KL 比较的是当前策略和固定参考策略

Reference KL 使用：

$$
D_{KL}
\left(
\pi_\phi\|\pi_{ref}
\right)
$$

其中：

$$
\pi_{ref}
$$

在 RLHF 训练过程中保持冻结。

它关心的是：

> 经过多轮优化之后，当前 Policy 累积偏离初始参考策略多少？

### 13.3 一个形象的类比

假设一个人每天都只能向前迈出很小的一步。

这类似于 PPO Clipping 所提供的局部更新约束。

但是，如果他每天都朝同一个方向走，即使每一步很小，经过很多天以后，也可能距离最初的位置非常远。

Reference KL 则类似于持续检查：

> 现在距离出发点有多远？

所以：

**PPO Clipping：关注局部更新稳定性。**

**Reference KL：关注相对于固定参考策略的整体偏移。**

两者并不是相互替代的关系。

| 对比项              | PPO Clipping          | Reference KL             |
| ---------------- | --------------------- | ------------------------ |
| 比较对象             | 当前 Policy 与旧 Policy   | 当前 Policy 与冻结的 Reference |
| 主要目的             | 限制过大的局部更新收益           | 限制相对初始策略的分布偏移            |
| Reference 是否持续不变 | Old Policy 会随数据采样过程更新 | Reference 通常保持冻结         |
| 主要解决的问题          | 优化稳定性                 | 策略偏移与 Reward 过度优化        |

还需要澄清：

**PPO 算法本身并不要求一定存在 SFT Reference Model。**

Reference KL 是 InstructGPT 这类 KL-Regularized RLHF 方法中的额外设计，不是所有 PPO 任务都必须具备的组件。

---

## 十四、InstructGPT 实际还使用了 PPO-ptx

如果只学习：

$$
\text{Reward}-\beta\text{KL}
$$

已经能够理解经典 InstructGPT RLHF 的主体。

但仔细阅读原论文，还会看到：

**PPO-ptx。**

为什么还需要它？

### 14.1 什么是 Alignment Tax？

在使用人类偏好进行对齐时，模型在用户关心的指令任务上可能表现更好。

但同时，它在某些其他任务上的能力可能退化。

例如，InstructGPT 论文观察到，RLHF 微调可能导致一些公开 NLP 数据集上的表现下降。

这种因对齐过程而产生的性能代价，被论文称为：

**Alignment Tax。**

这里需要区分：

提高当前偏好分布下的表现，并不自动意味着模型在所有任务上都会进步。

### 14.2 PPO-ptx 怎么做？

PPO-ptx 的核心思想是：

> 在 PPO 优化期间，额外混入一部分原始预训练数据上的语言建模训练。

论文给出的目标函数可以写为：

$$
\begin{aligned}
\mathcal{J}(\phi)
=&\ 
\mathbb{E}_{x,y}
\left[
r_\theta(x,y)
-
\beta
\log
\frac{\pi_\phi(y\mid x)}
{\pi_{ref}(y\mid x)}
\right]
\\
&+
\gamma
\mathbb{E}_{z\sim D_{pretrain}}
\left[
\log\pi_\phi(z)
\right]
\end{aligned}
$$

其中：

* 第一项：强化学习奖励与 KL 正则；
* 第二项：预训练数据上的语言建模目标；
* \(\gamma\)：控制预训练目标的权重。

其动机是：

**在优化人类偏好的同时，尽可能减少部分原有语言能力和任务表现的退化。**

这也说明，Reference KL 和预训练数据混合并不是完全相同的手段。

Reference KL 约束分布相对参考模型的偏移。

PPO-ptx 则额外通过预训练数据提供语言建模训练信号。

根据论文，若只称为 PPO，预训练混合项的系数设为 0；而论文中未特别说明时，InstructGPT 通常指 PPO-ptx 模型。

---

## 十五、把整个 InstructGPT 训练过程完整串起来

现在重新梳理一遍完整流程。

### Step 1：Pretraining

使用大规模文本数据训练语言模型：

$$
\min_\theta \mathcal{L}_{pretrain}
$$

得到具备基础语言能力的模型：

$$
\pi_{base}
$$

**训练目标：预测 Token。**

### Step 2：Supervised Fine-Tuning

收集人工示范：

$$
D_{SFT}=\{(x,y^*)\}
$$

优化：

$$
\min_\theta\mathcal{L}_{SFT}
$$

得到：

$$
\pi_{SFT}
$$

**训练目标：模仿高质量指令回答。**

### Step 3：Preference Data Collection

对同一个 Prompt 生成多个候选回答，由人类进行比较与排序。

构造：

$$
D_{pref}=\{(x,y_w,y_l)\}
$$

**数据目标：将人类的相对偏好转化为监督信号。**

### Step 4：Reward Model Training

使用 Preference Pair 训练 RM：

$$
\mathcal{L}_{RM}
=
-\mathbb{E}
\left[
\log\sigma(r_w-r_l)
\right]
$$

得到：

$$
r_\theta(x,y)
$$

**训练目标：让人类更偏好的回答获得更高评分。**

### Step 5：Initialize PPO Policy and Reference

从 SFT 后的策略初始化 PPO Policy。

同时保留冻结的 Reference Policy：

$$
\pi_\phi\leftarrow\pi_{SFT}
$$

$$
\pi_{ref}\leftarrow\pi_{SFT}
$$

其中：

* \(\pi_\phi\) 可训练；
* \(\pi_{ref}\) 冻结。

还需要 PPO 使用的 Old Policy 快照，以及用于估计 Return 的 Value Model。

### Step 6：Generate Responses

对于训练 Prompt：

$$
x\sim D_{prompt}
$$

由当前 Policy 生成：

$$
y\sim\pi_\phi(\cdot\mid x)
$$

### Step 7：Calculate Reward and KL

Reward Model 评价：

$$
r_\theta(x,y)
$$

Reference Model 提供概率基准：

$$
\log\pi_{ref}(y\mid x)
$$

结合当前 Policy：

$$
\log\pi_\phi(y\mid x)
$$

得到包含 KL 惩罚的训练奖励。

### Step 8：PPO Update

使用 Return、Value Model 和 Advantage 等信息构造 PPO 更新目标。

优化：

$$
\pi_\phi
$$

使其更倾向于产生高 Reward、同时不过度偏离 Reference 的回答。

然后继续采样、计算奖励和更新策略。

最终得到经过 RLHF 优化的指令遵循模型。

### 一张表总结训练中的模型

| 模型              | 主要职责                     | PPO 阶段是否更新   |
| --------------- | ------------------------ | ------------ |
| Policy Model    | 生成回答，接受强化学习优化            | 是            |
| Reward Model    | 评价生成结果                   | 通常否          |
| Reference Model | 提供固定的概率分布基准              | 否            |
| Old Policy      | 提供 PPO 概率比率的采样基准         | 按 PPO 采样轮次刷新 |
| Value Model     | 估计 Return，辅助计算 Advantage | 是            |

注意，工程实现不一定将这些角色全部部署为完全独立的模型副本。

但从概念和职责上，必须能够区分它们。

---

## 十六、几个最容易理解错误的地方

### 误区一：Reward Model 是一个判断对错的模型

不准确。

Reward Model 主要学习人工偏好数据中的相对排序关系。

它可能对正确性、帮助性、风格、指令遵循程度等多种因素进行隐式综合判断。

它不保证每次评分都等同于事实正确性。

### 误区二：Preference Pair 中的 Rejected 一定是错误答案

不正确。

Rejected 只表示在给定 Prompt 和标注标准下，相对于 Chosen 更不受偏好。

两个回答可能都正确，只是一个更适合当前用户。

### 误区三：PPO 是直接利用人工分数进行训练

在 InstructGPT 中，通常不是每轮 PPO 都依赖人工实时打分。

人工偏好先训练出 Reward Model，再由 RM 为 PPO 提供自动化奖励。

### 误区四：RM Reward 越高，实际回答质量就一定越高

不正确。

Reward Model 是不完美的人类偏好代理。

在优化过程中，可能发生 Reward Hacking 和 Reward Overoptimization。

因此，需要独立的人类评估，以及 KL 等正则化手段。

### 误区五：Reference Model 和 Reward Model 是同一种模型

不正确。

RM 评价回答。

Reference Model 输出概率分布，用于衡量策略差异。

### 误区六：PPO Clipping 已经足够，所以不需要 KL

不正确。

PPO Clipping 与 Reference KL 约束的是不同的策略比较关系。

但反过来也不能说所有 PPO 都必须使用 Reference KL。

### 误区七：KL 可以保证 Policy 永远不会出现异常行为

不正确。

KL 可以限制相对于参考分布的偏移，但无法保证 RM 本身没有偏差，也不能提供绝对的安全性保证。

### 误区八：RLHF 就等于 PPO

不正确。

RLHF 是利用人类反馈来优化模型行为的一类训练范式。

InstructGPT 使用了 Reward Modeling + PPO 的经典路线。

但 PPO 并不是利用人类偏好训练语言模型的唯一优化方法。

---

## 十七、进一步思考：InstructGPT 真正的价值是什么？

在我看来，InstructGPT 最重要的贡献之一，不是某一个单独的 Loss Function，而是建立了一条可以工程化实施的人类偏好优化链路。

它解决了一个非常现实的问题：

> 当我们无法为复杂的语言任务定义唯一标准答案时，如何让模型朝人类认为更好的方向优化？

整套方法完成了三个关键转换。

### 第一个转换：从人类示范到语言模型行为

$$
\text{Human Demonstrations}
\rightarrow
\text{SFT Policy}
$$

通过高质量示范，让模型学习助手应该如何回答问题。

### 第二个转换：从人类偏好到奖励函数

$$
\text{Human Preferences}
\rightarrow
\text{Preference Pairs}
\rightarrow
\text{Reward Model}
$$

通过比较数据，把主观偏好变成可以计算的优化信号。

### 第三个转换：从奖励函数到生成策略

$$
\text{Reward Function}
\rightarrow
\text{PPO Optimization}
\rightarrow
\text{Improved Policy}
$$

通过强化学习，让模型更倾向于主动产生高质量回答。

但是，InstructGPT 也揭示了对齐方法的一项重要局限：

**我们真正关心的是人类实际需求，但训练中往往只能优化其代理目标。**

这个代理目标包括：

* 人工示范数据；
* 偏好标注；
* Reward Model；
* KL 正则；
* 有限的评估指标。

这些信号都不能完美代表所有用户的真实意图。

而且，偏好标注还受到标注者、任务说明、数据分布和评价标准的影响。

因此，InstructGPT 学习到的是特定数据和标注流程所表达的人类偏好，而不是放之四海而皆准的“人类价值函数”。

这也解释了为什么模型对齐不仅是一个优化算法问题，同时还是一个数据质量、反馈设计、泛化能力和评估可靠性问题。

---

## 十八、面试中应该如何回答 InstructGPT 相关问题？

### Q1：请解释 InstructGPT 的训练流程。

**回答：**

InstructGPT 从预训练语言模型出发，通过 SFT、Reward Modeling 和 PPO 三个主要阶段进行对齐。

首先，SFT 使用人工示范数据训练模型，使其具备更好的指令遵循能力。

其次，让人类对多个候选回答进行比较，利用 Preference Pair 训练 Reward Model，使其能够预测人类更偏好哪个回答。

最后，使用 PPO 优化 SFT Policy，使生成结果获得更高的 RM Reward。

由于直接优化不完美的 RM 可能导致 Reward Hacking，训练中还会使用冻结的 Reference Model，通过 KL 惩罚约束策略偏移。

### Q2：Reward Model 的 Loss 是什么？

**回答：**

Reward Model 使用基于成对偏好的排序损失：

$$
\mathcal{L}_{RM}
=
-\mathbb{E}
\left[
\log\sigma
\left(
r_\theta(x,y_w)-r_\theta(x,y_l)
\right)
\right]
$$

其核心思想是让人类更偏好的回答获得更高 Reward。

通过 Sigmoid 将 Reward 差值映射成偏好概率，再利用负对数似然优化模型。

### Q3：为什么 SFT 后还需要 RLHF？

**回答：**

SFT 可以让模型模仿人工示范，但开放式语言任务通常存在多个合理答案。

相比要求人类为每个 Prompt 编写唯一的示范回答，偏好比较能够提供不同回答之间的相对质量信息。

RLHF 利用这些信息训练 RM，再通过策略优化使模型更倾向于生成符合人类偏好的回答。

### Q4：为什么 PPO 阶段需要 Reference Model？

**回答：**

Reference Model 通常是冻结的 SFT Policy，用作概率分布基准。

PPO 会持续改变 Policy，如果只优化 RM Reward，模型可能逐渐偏离原有行为，利用 RM 的缺陷获取高分。

因此，可以通过当前 Policy 与 Reference Policy 的 KL 散度对策略变化施加惩罚，缓解 Reward Model Overoptimization。

### Q5：PPO Clipping 与 Reference KL 有什么区别？

**回答：**

PPO Clipping 比较的是当前策略与采样数据时使用的 Old Policy，主要关注局部策略更新的稳定性。

Reference KL 比较的是当前 Policy 与冻结参考模型，主要约束相对初始策略的分布偏移。

两者解决的问题不同，可以同时使用。

### Q6：Reward Model 和 Value Model 有什么区别？

**回答：**

Reward Model 负责根据 Prompt 和 Response 输出奖励评分，用来近似人类偏好。

Value Model 负责估计某个生成状态的预期 Return，辅助计算 Advantage，提高 PPO 训练的稳定性。

两者虽然可能使用相似的 Transformer 结构，但训练目标和使用方式不同。

---

## 十九、总结：用一句话理解整个 RLHF Pipeline

经过这一轮学习，我认为可以用下面这条主线理解 InstructGPT：

$$
\boxed{
\text{Pretrain}
\rightarrow
\text{SFT}
\rightarrow
\text{Reward Model}
\rightarrow
\text{PPO}
}
$$

**Pretrain：**

让模型学习语言和知识。

**SFT：**

通过人工示范，教模型如何像一个助手一样回答问题。

**Reward Model：**

通过 Preference Pair，将人类偏好转化为可以学习的奖励信号。

**PPO：**

利用 Reward Model 提供的反馈，持续优化模型的生成策略。

**Reference Model + KL：**

为策略优化提供一个固定基准，缓解模型为了追求高 Reward 而产生的过度偏移。

从优化目标来看，最值得记住的是：

$$
\boxed{
\max_\pi
\quad
\mathbb{E}[r(x,y)]
-
\beta D_{KL}(\pi\|\pi_{ref})
}
$$

其中：

$$
\text{Reward}
$$

鼓励模型向更符合人类偏好的方向改进。

而：

$$
\text{KL Penalty}
$$

限制模型相对于参考策略的过度偏移。

所以整套方法最终可以概括为：

> **通过人类示范建立基础行为，通过偏好比较学习奖励标准，通过强化学习改进生成策略，再通过 KL 正则约束优化过程。**

我认为，真正理解 InstructGPT，并不是记住 SFT、RM 和 PPO 这些名词，而是能够解释：

1. 为什么开放式语言任务需要偏好学习？
2. 为什么偏好比较可以训练出 Reward Model？
3. 为什么只有 Reward Model 还无法直接得到更好的生成模型？
4. 为什么直接最大化 Reward 存在风险？
5. 为什么需要 Reference Model 和 KL？
6. 为什么 PPO Clipping 与 Reference KL 不能混为一谈？

当这些问题都能够独立回答时，就基本建立起了对经典 InstructGPT/RLHF 流水线的系统理解。

接下来值得进一步学习的是 **PPO 的具体优化机制**，包括 Policy Gradient、Advantage、Value Function、GAE 和 PPO-Clip。

有了本文的整体框架，再去阅读 RLHF 训练代码，才能真正理解每个模型、每个概率比率以及每个 Loss 的作用。

---

## 参考文献

**[1] Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback.**

InstructGPT 原始论文，本文最主要的参考来源。重点阅读 Figure 2、Section 3.1、Section 3.5、Equation (1)、Equation (2) 和 Appendix C。

* 论文地址：https://arxiv.org/abs/2203.02155
* PDF：https://arxiv.org/pdf/2203.02155

**[2] OpenAI. (2022). Aligning language models to follow instructions.**

OpenAI 官方技术介绍，解释了为什么需要通过人类反馈改进模型的指令遵循能力，以及 InstructGPT 的训练方法、实验发现和局限性。

* 官方文章：https://openai.com/index/instruction-following/

**[3] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms.**

PPO 原始论文。用于理解 PPO 的 Policy Gradient 优化思想、Surrogate Objective 和 Clipping 机制。

* 论文地址：https://arxiv.org/abs/1707.06347
* PDF：https://arxiv.org/pdf/1707.06347

**[4] Stiennon, N., Ouyang, L., Wu, J., et al. (2020). Learning to summarize from human feedback.**

较早将人类偏好比较、Reward Modeling 和强化学习结合应用于语言模型文本摘要任务的代表性工作，也是 InstructGPT 的重要方法基础之一。

* 论文地址：https://arxiv.org/abs/2009.01325
* PDF：https://arxiv.org/pdf/2009.01325

---

*本文为大语言模型系统学习系列 Day 8 的学习笔记，主要基于 InstructGPT 原论文和相关原始研究整理。文中的教学示例用于解释训练原理，不代表论文中的真实样本或实验输出。*
