# Day 9｜Reward Model 与 KL：为什么 RLHF 不能只追求更高的 Reward？

> LLM 强化学习系列 · Day 9
>
> 关键词： RLHF、Reward Model、Bradley-Terry、KL Divergence、Reward Hacking、Goodhart's Law
>
> 本文目标： 从直觉、数学公式和实际案例出发，理解 Reward Model 如何学习人类偏好、为什么它会被强化学习算法利用，以及 KL 正则化为什么是 RLHF 中的重要组成部分。

## 一、引言：为什么让大模型获得更高的 Reward，反而可能让它变差？

假设我们正在训练一个大语言模型，希望它能够生成更加符合人类偏好的回答。

一个很自然的想法是：

既然我们希望模型生成更好的回答，为什么不直接给每个回答打分，然后训练模型尽可能获得更高的分数？

这正是 RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）的核心思路之一。

但是，这里存在一个非常重要的问题：

我们并不知道如何用一个准确的数学函数来衡量回答是否真的符合人类偏好。

例如，对于同一个问题：

> 为什么天空是蓝色的？

回答 A：

> 这是因为瑞利散射。太阳光进入大气层后，较短波长的光更容易被空气分子散射，因此我们通常看到蓝色的天空。

回答 B：

> 因为天空反射了海洋的颜色。

大多数了解相关科学知识的人会认为回答 A 更好。

但是，我们很难直接定义一个函数：

R∗(x,y)R^*(x,y)R∗(x,y)

让它准确表示任意问题 xxx 和回答 yyy 的真实质量。

这时，一个替代方案出现了：

既然很难直接定义什么是好回答，那么能否让人类比较两个回答，然后让模型学习这种偏好？

于是，Reward Model（奖励模型）就出现了。

我们可以把 RLHF 的主要流程理解为：

```
            预训练语言模型
                  |
                  v
             SFT 阶段
          学习如何完成指令
                  |
                  v
         生成多个候选回答
                  |
                  v
             人类偏好标注
        哪个回答更符合要求？
                  |
                  v
         Preference Dataset
            (x, yw, yl)
                  |
                  v
            Reward Model
           学习预测人类偏好
                  |
                  v
             RL 阶段
        优化 Policy 获得高分
                  |
             +----+----+
             |         |
             v         v
        Reward ↑    KL 约束
             |         |
             +----+----+
                  |
                  v
          更新后的语言模型
```

这里最值得思考的是：

Reward Model 只是在预测人类偏好，而 RL 优化器却会主动寻找能够获得高 Reward 的输出。

预测人类偏好，与最大化偏好预测模型的分数，并不是完全相同的事情。

本文围绕这一区别，逐步解释三个核心问题：

1. Reward Model 是如何通过 Bradley-Terry 模型学习人类偏好的？

2. 为什么 Reward Model 会被 Reward Hacking？

3. KL 正则化如何限制 Policy 的变化？没有 KL 又会发生什么？

## 二、Reward Model 到底在学习什么？

### 2.1 为什么不直接预测一个绝对分数？

假设我们收集到一个训练样本：

(x,yw,yl)(x,y_w,y_l)(x,yw,yl)

其中：

* xxx：用户输入的 Prompt。

* ywy_wyw：人类更喜欢的回答，称为 Winner。

* yly_lyl：人类不太喜欢的回答，称为 Loser。

人类标注员只需要给出：

yw≻yly_w \succ y_lyw≻yl

也就是：

> 在给定 Prompt 的情况下，回答 ywy_wyw 比回答 yly_lyl 更符合标注偏好。

注意，人类并没有告诉模型：

```
回答 A：8.5 分
回答 B：3.2 分
```

而只是提供：

```
回答 A 优于回答 B
```

为什么采用这种方式？

因为在很多任务中，相对比较通常比绝对评分更容易定义，也更容易让标注员完成。

比如：

> 你可能很难准确判断一篇文章应该得 83 分还是 87 分，但相对容易判断两篇文章哪篇更好。

于是，我们训练一个模型：

rϕ(x,y)r_\phi(x,y)rϕ(x,y)

其中：

* ϕ\phiϕ：Reward Model 的参数。

* xxx：输入问题。

* yyy：候选回答。

* rϕ(x,y)r_\phi(x,y)rϕ(x,y)：Reward Model 输出的标量分数。

例如：

rϕ(x,yA)=5r_\phi(x,y_A)=5rϕ(x,yA)=5

rϕ(x,yB)=2r_\phi(x,y_B)=2rϕ(x,yB)=2

我们希望：

rϕ(x,yA)>rϕ(x,yB)r_\phi(x,y_A)>r_\phi(x,y_B)rϕ(x,yA)>rϕ(x,yB)

但这里产生了一个问题：

人类只告诉我们 A 比 B 好，Reward Model 为什么能够学会输出数值？

答案就是 Bradley-Terry Preference Model。

## 三、Bradley-Terry Model：如何将偏好比较转化成概率？

### 3.1 从一个直觉开始

假设我们已经有两个回答的 Reward：

rw=5,rl=2r_w=5,\qquad r_l=2rw=5,rl=2

二者差值为：

Δr=rw−rl=3\Delta r=r_w-r_l=3Δr=rw−rl=3

我们希望：

* 如果 Δr\Delta rΔr 很大，模型预测 Winner 被偏好的概率很高。

* 如果 Δr=0\Delta r=0Δr=0，模型认为两个回答被偏好的概率相同。

* 如果 Δr<0\Delta r<0Δr<0，说明 Reward Model 的预测倾向与当前标注相反。

也就是说，需要一个函数，将：

Δr∈(−∞,+∞)\Delta r\in(-\infty,+\infty)Δr∈(−∞,+∞)

映射成：

P∈(0,1)P\in(0,1)P∈(0,1)

这正是 Sigmoid 函数能够完成的事情。

### 3.2 从 Bradley-Terry 模型推导 Sigmoid

Bradley-Terry 模型最初用于成对比较。

它的基本思想是：

假设每个候选项都具有一个正的潜在强度。

对于两个回答，可以定义：

sw=erw,sl=erls_w=e^{r_w},\qquad s_l=e^{r_l}sw=erw,sl=erl

其中 rw,rlr_w,r_lrw,rl 是实数形式的潜在得分。

Bradley-Terry 模型假设 Winner 被偏好的概率为：

P(yw≻yl)=swsw+slP(y_w\succ y_l) = \frac{s_w}{s_w+s_l}P(yw≻yl)=sw+slsw

代入指数形式：

P(yw≻yl)=erwerw+erlP(y_w\succ y_l) = \frac{e^{r_w}}{e^{r_w}+e^{r_l}}P(yw≻yl)=erw+erlerw

分子分母同时除以 erwe^{r_w}erw：

P(yw≻yl)=11+erl−rwP(y_w\succ y_l) = \frac{1}{1+e^{r_l-r_w}}P(yw≻yl)=1+erl−rw1

整理：

P(yw≻yl)=11+e−(rw−rl)P(y_w\succ y_l) = \frac{1}{1+e^{-(r_w-r_l)}}P(yw≻yl)=1+e−(rw−rl)1

定义 Sigmoid 函数：

σ(z)=11+e−z\sigma(z)=\frac{1}{1+e^{-z}}σ(z)=1+e−z1

于是得到：

P(yw≻yl∣x)=σ(rϕ(x,yw)−rϕ(x,yl))\boxed{ P(y_w\succ y_l\mid x) = \sigma\left( r_\phi(x,y_w)-r_\phi(x,y_l) \right) }P(yw≻yl∣x)=σ(rϕ(x,yw)−rϕ(x,yl))

这就是 RLHF 中经常使用的 Bradley-Terry 偏好模型。

它的核心思想是：两个回答的 Reward 差值，决定模型预测的人类偏好概率。

需要注意，这里的概率是建立在 Bradley-Terry 假设下的预测概率，不等于回答在客观意义上正确的概率。

### 3.3 用具体数字理解 Sigmoid

假设：

Δr=rw−rl\Delta r=r_w-r_lΔr=rw−rl

可以得到：

|
Reward 差值

|

Winner 的预测偏好概率

|

如何理解

|
| --- | --- | --- |
|

-2

|

0.119

|

模型明显倾向另一个回答

|
|

0

|

0.500

|

模型没有表现出偏好

|
|

1

|

0.731

|

模型更倾向 Winner

|
|

2

|

0.881

|

模型明显倾向 Winner

|

这些数值经过 Sigmoid 计算及反向 Logit 核对。

比如：

rw=3,rl=2r_w=3,\quad r_l=2rw=3,rl=2

因此：

Δr=1\Delta r=1Δr=1

那么：

P(yw≻yl)=σ(1)≈0.731P(y_w\succ y_l)=\sigma(1)\approx0.731P(yw≻yl)=σ(1)≈0.731

意思是：

> 根据 Reward Model 当前的分数，在 Bradley-Terry 模型下，Winner 被偏好的预测概率约为 73.1%。

这里有一个非常关键的理解：

Reward Model 并不是直接在判断一个回答是否“绝对正确”，它是在学习如何预测偏好比较的结果。

### 3.4 为什么 Reward 的绝对值没有固定意义？

继续考虑：

rw=5,rl=2r_w=5,\quad r_l=2rw=5,rl=2

二者的差值：

Δr=3\Delta r=3Δr=3

假设给两个回答同时加上一个常数 CCC：

rw′=rw+Cr'_w=r_w+Crw′=rw+C

rl′=rl+Cr'_l=r_l+Crl′=rl+C

新的差值：

rw′−rl′=(rw+C)−(rl+C)r'_w-r'_l = (r_w+C)-(r_l+C)rw′−rl′=(rw+C)−(rl+C)

最终：

rw′−rl′=rw−rlr'_w-r'_l=r_w-r_lrw′−rl′=rw−rl

因此：

σ(rw′−rl′)=σ(rw−rl)\sigma(r'_w-r'_l) = \sigma(r_w-r_l)σ(rw′−rl′)=σ(rw−rl)

这说明：

Bradley-Terry 偏好概率对两个回答共同的 Reward 平移不敏感。

更一般地，对同一个 Prompt，给所有候选回答的 Reward 加上相同的常数，都不会改变偏好概率。

所以：

rϕ(x,y)=8r_\phi(x,y)=8rϕ(x,y)=8

不意味着：

> 这个回答的真实质量是 8 分（满分 10 分）。

这个数值本身没有这样的固定语义。

但是需要补充一个容易忽略的细节：

Reward 的绝对零点可以平移，不代表 Reward 的尺度可以任意缩放。

因为在固定的 Sigmoid 模型中，如果把所有 Reward 乘以一个系数，Reward Difference 也会变化，进而改变预测的偏好概率。

所以更严谨的表述是：

> Reward Model 从相对偏好监督中学习一个潜在得分函数。其分数差值影响预测偏好概率，但单个分数并不是具有统一物理意义的绝对质量评分。

## 四、Reward Model 如何训练？

我们已经知道：

P(yw≻yl∣x)=σ(rw−rl)P(y_w\succ y_l\mid x) = \sigma(r_w-r_l)P(yw≻yl∣x)=σ(rw−rl)

现在假设人类已经选择了 ywy_wyw。

那么训练 Reward Model 的目标，就是让它给这个事件分配尽可能高的概率。

这是一个最大似然估计问题。

对于一个偏好样本：

LRM=−log⁡P(yw≻yl∣x)\mathcal L_{\mathrm{RM}} = -\log P(y_w\succ y_l\mid x)LRM=−logP(yw≻yl∣x)

把 Bradley-Terry 概率代入：

LRM=−log⁡σ(rϕ(x,yw)−rϕ(x,yl))\boxed{ \mathcal L_{\mathrm{RM}} = -\log\sigma \left( r_\phi(x,y_w)-r_\phi(x,y_l) \right) }LRM=−logσ(rϕ(x,yw)−rϕ(x,yl))

对于整个数据集：

LRM=−E(x,yw,yl)∼D\(log⁡σ(rϕ(x,yw)−rϕ(x,yl))\)\boxed{ \mathcal L_{\mathrm{RM}} = -\mathbb E_{(x,y_w,y_l)\sim\mathcal D} \left\( \\log\\sigma \\left( r\_\\phi(x,y\_w)-r\_\\phi(x,y\_l) \\right) \\right\) }LRM=−E(x,yw,yl)∼D\(logσ(rϕ​(x,yw​)−rϕ​(x,yl​))\)

现在从直觉上分析。

情况 A：Reward Model 的预测符合人类偏好

rw>rlr_w>r_lrw>rl

于是：

σ(rw−rl)>0.5\sigma(r_w-r_l)>0.5σ(rw−rl)>0.5

当 Reward 差距足够大时，模型给人类偏好事件分配较高概率，Loss 就会比较小。

情况 B：Reward Model 的预测与人类偏好相反

rw<rlr_w<r_lrw<rl

此时：

σ(rw−rl)<0.5\sigma(r_w-r_l)<0.5σ(rw−rl)<0.5

模型给正确标注事件分配的概率较低，因此 Loss 较大。

训练通过梯度下降，推动：

rw−rlr_w-r_lrw−rl

朝更大的方向变化。

注意，这并不要求每次更新都让 rwr_wrw 增大、rlr_lrl 减小。真正被 Loss 直接约束的是两者的差值。

到这里，我们已经理解 Reward Model 在做什么了：

它通过大量人类偏好比较样本，学习一个能够预测回答相对偏好的评分函数。

但是，一个新的问题出现了。

能够预测人类偏好的模型，是否就可以被无限优化？

答案是否定的。

## 五、为什么 Reward Model 会被 Reward Hacking？

### 5.1 从“评判回答”到“主动寻找高分回答”

训练 Reward Model 时，我们做的是：

```
人类提供偏好数据
        |
        v
Reward Model 学习评分规则
        |
        v
预测人类更喜欢哪个回答
```

但是进入 RL 阶段后，情况发生了变化。

我们有一个 Policy：

πθ(y∣x)\pi_\theta(y\mid x)πθ(y∣x)

它表示给定输入 xxx 时，模型生成回答 yyy 的概率。

现在，强化学习希望最大化：

max⁡θEx∼D, y∼πθ(⋅∣x)\(rϕ(x,y)\)\boxed{ \max_\theta \mathbb E_{x\sim\mathcal D,\, y\sim\pi_\theta(\cdot\mid x)} \left\( r\_\\phi(x,y) \\right\) }θmaxEx∼D,y∼πθ(⋅∣x)\(rϕ​(x,y)\)

简单来说：

> 通过调整 Policy，让模型更容易生成能够获得高 Reward 的回答。

于是：

```
Policy 生成回答
       |
       v
Reward Model 评分
       |
       v
RL 优化器调整 Policy
       |
       v
Policy 更倾向产生高分回答
       |
       v
继续优化
```

这里最关键的变化是：

原本 Reward Model 只是一个评分器，现在它成为了整个优化过程的目标。

只要 Reward Model 存在某种可利用的评分偏差，Policy 就可能逐渐学会利用这种偏差。

### 5.2 一个例子：Reward Model 为什么可能偏爱长回答？

假设我们正在训练一个问答模型。

在收集偏好数据时，标注员经常发现：

* 解释充分的回答通常比过于简略的回答更好。

* 结构清晰的回答通常比杂乱的回答更好。

* 能够说明原因的回答通常比只有结论的回答更好。

这些偏好本身没有问题。

但是，Reward Model 可能从有限数据中学到一种不完善的规律：

```
好的回答通常更详细
        |
        v
详细的回答通常更长
        |
        v
较长的回答可能获得更高 Reward
```

其中最后一步可能成为一种 Shortcut（捷径特征）。

也就是说：

回答长度只是某些高质量回答的相关特征，并不是回答质量本身。

但是，RL 优化器并不会天然理解这一点。

如果在某些情况下，增加无关内容就能提高 Reward Model 的评分，Policy 可能逐渐学会：

```
增加解释长度
       |
       v
获得更高 Reward
       |
       v
进一步增加长度
       |
       v
开始重复、堆砌内容
       |
       v
Reward Model 仍然给出高分
```

最后可能出现：

> 回答越来越长，越来越啰嗦，甚至偏离用户的问题，但 Reward Model 仍然认为它很好。

这就是 Reward Hacking 的一种可能表现。

### 5.3 Reward Hacking 的本质：优化了代理目标，而不是真实目标

为了更准确地理解这个问题，我们可以假设存在一个理想的目标函数：

R∗(x,y)R^*(x,y)R∗(x,y)

它代表我们真正希望优化的回答质量。

需要强调：这里的 R∗R^*R∗ 是一种用于分析的理想化假设，并不意味着真实世界中一定存在一个能够完整衡量所有人类偏好的客观标量函数。

现实中，我们只能训练：

rϕ(x,y)r_\phi(x,y)rϕ(x,y)

去近似它。

因此可以形式化地写成：

rϕ(x,y)=R∗(x,y)+ϵ(x,y)r_\phi(x,y) = R^*(x,y)+\epsilon(x,y)rϕ(x,y)=R∗(x,y)+ϵ(x,y)

其中：

ϵ(x,y)\epsilon(x,y)ϵ(x,y)

代表 Reward Model 的预测误差。

现在 RL 优化的是：

max⁡yrϕ(x,y)\max_y r_\phi(x,y)ymaxrϕ(x,y)

代入：

max⁡y\(R∗(x,y)+ϵ(x,y)\)\max_y \left\( R^\*(x,y)+\\epsilon(x,y) \\right\)ymax\(R∗(x,y)+ϵ(x,y)\)

这意味着优化器不仅可能找到真实质量高的回答，也可能找到：

真实质量一般，但 Reward Model 预测误差特别有利的回答。

这就是问题所在。

如果某个回答：

R∗(x,y) 不高R^*(x,y)\text{ 不高}R∗(x,y) 不高

但：

ϵ(x,y) 很大且为正\epsilon(x,y)\text{ 很大且为正}ϵ(x,y) 很大且为正

那么：

rϕ(x,y)r_\phi(x,y)rϕ(x,y)

仍然可能很高。

强化学习并不知道这部分高分来自预测误差。

它只知道：

> 这个回答的 Reward 很高，应该增加产生它的概率。

于是，Reward Model 的误差可能被持续利用。

这一过程可以总结为：

```
有限的偏好数据
       |
       v
不完美的 Reward Model
       |
       v
存在近似误差或错误相关性
       |
       v
RL 主动搜索高 Reward 输出
       |
       v
找到能够利用误差的回答
       |
       v
Policy 增加此类回答的概率
       |
       v
Reward Hacking
```

Reward Hacking 并不一定意味着模型在有意识地“作弊”，而是优化过程找到了代理目标与真实目标之间的漏洞。

## 六、更深一层：为什么优化越强，Reward Model 的问题可能越严重？

这里需要理解两个重要概念：

1. Goodhart's Law

2. Distribution Shift

### 6.1 Goodhart's Law：当指标变成目标

Goodhart's Law 经常被概括为：

> 当一个指标成为优化目标时，它可能不再是一个可靠的衡量指标。

为什么？

假设我们希望培养真正理解数学的学生。

由于数学能力很难直接衡量，于是我们设计考试分数作为代理指标。

最初：

```
数学能力较好
      |
      v
通常获得较高考试分数
```

所以考试分数确实具有一定参考价值。

但是，如果把考试分数作为唯一目标，学生可能开始研究：

* 押题。

* 记忆标准答案。

* 依赖答题模板。

* 针对考试规则进行训练。

最终可能出现：

```
考试分数很高
       ≠
数学理解能力一定很强
```

我们实际上优化了考试这个代理指标，而不一定是真正想要的数学能力。

Reward Model 也有类似问题：

```
真正想优化：
Human Preference / Answer Quality

实际能够优化：
Reward Model Score
```

当优化压力越来越强时，Policy 可能越来越擅长获得高分，而不一定越来越擅长满足人类真实需求。

### 6.2 Distribution Shift：Policy 可能离开 Reward Model 熟悉的区域

还有一个重要问题：

Reward Model 的训练数据分布，与 RL 后期的回答分布可能不一样。

假设 Reward Model 最初是在这样的数据上训练的：

```
正常回答 A
    vs
正常回答 B
```

这些回答可能主要来自 SFT Model。

但是 RL 会不断改变 Policy：

πθ\pi_\thetaπθ

于是模型生成回答的分布也在不断发生变化。

最初：

πθ≈πref\pi_\theta\approx\pi_{\mathrm{ref}}πθ≈πref

后期可能变成：

πθ≉πref\pi_\theta\not\approx\pi_{\mathrm{ref}}πθ≈πref

此时，Policy 可能产生一些 Reward Model 训练时很少见过的回答。

在这些区域中，Reward Model 的预测未必可靠。

因此可能出现：

```
Reward Model 训练分布
        |
        |  预测相对可靠
        v
   常见的正常回答
        |
        |
        | RL 不断优化
        v
   分布逐渐发生变化
        |
        v
   非常规高分回答
        |
        v
   Reward Model 可能误判
```

这就是为什么：

Reward Model 的预测能力，不等于在任意优化强度、任意回答分布下都可靠。

### 6.3 真实研究中观察到了什么？

这不仅仅是一个理论上的担忧。

在 Ziegler 等人 2019 年的研究 Fine-Tuning Language Models from Human Preferences 中，研究者发现，在摘要任务中，模型会复制原文中的完整句子。作者指出，这种行为可能利用了标注员评价时使用的启发式规则。\(2\)

另外，Gao、Schulman 和 Hilton 在 2023 年发表的 Scaling Laws for Reward Model Overoptimization 专门研究了 Reward Model 过度优化。

他们使用一种可控的实验设置：让一个较强的“gold-standard”奖励模型充当评价标准，再训练一个代理 Reward Model。

研究发现，随着对代理 Reward Model 的优化持续增强，代理评分的提高并不总能转化为真实评价目标的提高。\(4\)

这里需要注意，该实验中的 gold-standard 是一个模拟理想评价者的模型，而不是真正能够测量所有人类偏好的函数。

这项研究的重要意义在于：

它说明了一个代理 Reward Model 即使在训练时具有一定预测能力，也不能保证被持续最大化后，实际目标仍然同步改善。

## 七、既然 Reward Model 不可靠，为什么还要优化它？

这是一个很值得思考的问题。

因为虽然 Reward Model 并不完美，但它仍然提供了有价值的训练信号。

如果完全不使用 Reward Model，我们就难以利用大规模偏好数据来引导 Policy。

真正的问题不在于：

> 是否应该使用 Reward Model？

而在于：

> 应该允许 Policy 为了追求 Reward，偏离原来的模型多远？

这就引出了 KL Regularization。

## 八、KL Divergence：为什么 RLHF 需要一个约束？

### 8.1 先理解 Reference Policy

在典型的 RLHF 流程中，我们已经有一个经过 SFT 的模型。

记为：

πref(y∣x)\pi_{\mathrm{ref}}(y\mid x)πref(y∣x)

它通常被称为 Reference Policy（参考策略）。

我们复制或者基于它初始化一个需要继续训练的 Policy：

πθ(y∣x)\pi_\theta(y\mid x)πθ(y∣x)

最初：

πθ≈πref\pi_\theta\approx\pi_{\mathrm{ref}}πθ≈πref

这意味着它们生成回答的概率分布非常接近。

现在，我们希望：

πθ\pi_\thetaπθ

能够通过 RL 学会更符合人类偏好的行为。

但是，如果只要求最大化 Reward，就可能出现严重的 Policy Drift。

所谓 Policy Drift，就是：

> 优化后的 Policy 与原始 Reference Policy 的行为分布发生较大偏离。

这种变化不一定全部是坏事。

毕竟，RL 本来就需要改变模型行为。

真正的问题是：

如果模型为了获得高 Reward，偏离了原本具有一定语言能力和任务表现的分布，可能更容易进入 Reward Model 难以可靠评估的区域。

因此，我们希望：

```
允许 Policy 改进
       +
不希望它无约束地偏离 Reference
```

于是加入 KL 正则化。

### 8.2 KL Divergence 究竟衡量什么？

KL Divergence（Kullback-Leibler Divergence）可以用来衡量两个概率分布之间的差异。

对于给定的 Prompt xxx，定义：

DKL(πθ∥πref)=Ey∼πθ(⋅∣x)\(log⁡πθ(y∣x)πref(y∣x)\)\boxed{ D_{\mathrm{KL}} ( \pi_\theta\Vert\pi_{\mathrm{ref}} ) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)} \left\( \\log \\frac{\\pi\_\\theta(y\\mid x)} {\\pi\_{\\mathrm{ref}}(y\\mid x)} \\right\) }DKL(πθ∥πref)=Ey∼πθ(⋅∣x)\(logπref​(y∣x)πθ​(y∣x)​\)

这里的两个分布分别是：

πθ(y∣x)\pi_\theta(y\mid x)πθ(y∣x)

当前正在训练的 Policy。

以及：

πref(y∣x)\pi_{\mathrm{ref}}(y\mid x)πref(y∣x)

固定的 Reference Policy。

直觉上：

* 两个分布越接近，KL 越小。

* 两个分布差异越明显，KL 可能越大。

* 当两个分布完全相同时，KL 为零。

需要注意，KL 严格来说不是普通意义上的距离，因为它通常不满足对称性：

DKL(P∥Q)≠DKL(Q∥P)D_{\mathrm{KL}}(P\Vert Q) \neq D_{\mathrm{KL}}(Q\Vert P)DKL(P∥Q)=DKL(Q∥P)

而且它是在概率分布层面定义的量，不是单个回答的质量评分。

### 8.3 RLHF 的完整优化目标

没有 KL 时：

max⁡θEx,y\(rϕ(x,y)\)\max_\theta \mathbb E_{x,y} \left\( r\_\\phi(x,y) \\right\)θmaxEx,y\(rϕ​(x,y)\)

加入 KL 后：

max⁡θJ(θ)=Ex∼D\(Ey∼πθ(⋅∣x)\[rϕ(x,y)\)−βDKL(πθ(⋅∣x)∥πref(⋅∣x))]\boxed{ \begin{aligned} \max_\theta\quad J(\theta) = \mathbb E_{x\sim\mathcal D} \Big\( &\\mathbb E\_{y\\sim\\pi\_\\theta(\\cdot\\mid x)} \[r\_\\phi(x,y)\)\\ &-\beta D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid x) \Vert \pi_{\mathrm{ref}}(\cdot\mid x) \right) \Big] \end{aligned} }θmaxJ(θ)=Ex∼D\(​Ey∼πθ​(⋅∣x)​\[rϕ​(x,y)\)−βDKL(πθ(⋅∣x)∥πref(⋅∣x))]

这里：

* 第一项：希望获得更高的 Reward。

* 第二项：惩罚 Policy 相对于 Reference Policy 的分布偏离。

* β\betaβ：控制两者之间的权衡。

可以将其理解为：

```
           Reward Maximization
                    |
                    v
              推动 Policy
             寻找更高 Reward
                    |
                    v
                  πθ
                    ^
                    |
               KL Penalty
                    |
                    |
                  πref
           约束偏离 Reference
```

Reward 负责告诉模型往哪里改进，KL 负责限制改进过程的偏离程度。

但要注意：Reference Policy 并不是“绝对正确”的模型，KL 也不意味着模型越接近 Reference 就一定越好。

它只是在优化过程中提供一个相对稳定的参考分布。

## 九、β 到底控制什么？用一个小实验看懂

前面我们得到：

J=E\(rϕ\)−βDKLJ= \mathbb E\(r\_\\phi\) - \beta D_{\mathrm{KL}}J=E\(rϕ​\)−βDKL

现在重点分析：

β\betaβ

### 9.1 当 β 较小时

β→0\beta\rightarrow0β→0

KL 惩罚的影响逐渐减小。

模型会更加重视：

max⁡E\(rϕ\)\max\mathbb E\(r\_\\phi\)maxE\(rϕ​\)

因此：

* Policy 可以发生更大变化。

* 模型更容易集中到高 Reward 回答上。

* Reward Hacking 和 Distribution Shift 的风险可能增加。

### 9.2 当 β 较大时

β→∞\beta\rightarrow\inftyβ→∞

KL 惩罚越来越强。

在通常的理想化假设下，最优 Policy 会越来越接近 Reference Policy。

因此：

* Policy 更新更加保守。

* 更不容易出现大幅分布漂移。

* 但同时也可能限制模型在 Reward 上获得的改进。

### 9.3 一个只有两种回答的数学例子

假设某个 Prompt 只有 A、B 两种可能回答。

Reference Policy 的概率为：

πref(A)=0.9\pi_{\mathrm{ref}}(A)=0.9πref(A)=0.9

πref(B)=0.1\pi_{\mathrm{ref}}(B)=0.1πref(B)=0.1

Reward Model 给出的评分为：

r(A)=0,r(B)=3r(A)=0,\qquad r(B)=3r(A)=0,r(B)=3

也就是说：

Reference Policy 更倾向回答 A，但 Reward Model 更喜欢回答 B。

当使用 KL 正则化目标时，在可以自由选择回答分布的理想条件下，最优 Policy 满足：

π∗(y∣x)=πref(y∣x)er(x,y)/βZ(x)\boxed{ \pi^*(y\mid x) = \frac{ \pi_{\mathrm{ref}}(y\mid x) e^{r(x,y)/\beta} }{ Z(x) } }π∗(y∣x)=Z(x)πref(y∣x)er(x,y)/β

其中：

Z(x)=∑yπref(y∣x)er(x,y)/βZ(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x) e^{r(x,y)/\beta}Z(x)=y∑πref(y∣x)er(x,y)/β

这个公式表明：

最优 Policy 同时受到 Reference Policy 原始概率和 Reward 大小的影响。

对于回答 B：

π∗(B)=0.1e3/β0.9+0.1e3/β\pi^*(B) = \frac{ 0.1e^{3/\beta} }{ 0.9+0.1e^{3/\beta} }π∗(B)=0.9+0.1e3/β0.1e3/β

代入不同的 β\betaβ：

|
β

|

最优 Policy 选择 B 的概率

|
| --- | --- |
|

0.5

|

0.978

|
|

1

|

0.691

|
|

2

|

0.332

|

计算结果已核对，并通过概率比的反向关系进行了验证。

观察这个结果：

最初 Reference Policy 选择 B 的概率只有 0.1。

但是，当 Reward Model 认为 B 更好时，RL 会提高 B 的概率。

如果 β\betaβ 较小，Policy 可能几乎总是选择 B。

如果 β\betaβ 较大，Policy 虽然仍然会增加选择 B 的概率，但变化相对保守。

这个例子解释了一个重要事实：

KL 并不是禁止 Policy 改变，而是让 Reward 改进必须付出偏离 Reference 的代价。

同样需要注意，即使使用了 KL，Policy 也可能发生很大的变化。这取决于 Reward 差值、β\betaβ 和 Reference Policy 本身。

## 十、深入理解：为什么代码里经常出现 logprob - ref_logprob？

这部分非常重要。

因为前面讨论的是完整概率分布之间的 KL：

DKL(πθ∥πref)D_{\mathrm{KL}} ( \pi_\theta\Vert\pi_{\mathrm{ref}} )DKL(πθ∥πref)

但实际阅读 PPO/RLHF 实现时，我们经常看到：

Python

运行

```
log_ratio = logprob - ref_logprob
kl_reward = -beta * log_ratio
```

为什么这两个东西能够联系起来？

### 10.1 从 Sequence Probability 开始

大语言模型通常是自回归模型。

假设生成的回答由多个 Token 组成：

y=(y1,y2,…,yT)y=(y_1,y_2,\ldots,y_T)y=(y1,y2,…,yT)

那么完整回答的概率可以写成：

πθ(y∣x)=∏t=1Tπθ(yt∣x,y<t)\pi_\theta(y\mid x) = \prod_{t=1}^{T} \pi_\theta(y_t\mid x,y_{<t})πθ(y∣x)=t=1∏Tπθ(yt∣x,y<t)

对两边取对数：

log⁡πθ(y∣x)=∑t=1Tlog⁡πθ(yt∣x,y<t)\log\pi_\theta(y\mid x) = \sum_{t=1}^{T} \log\pi_\theta(y_t\mid x,y_{<t})logπθ(y∣x)=t=1∑Tlogπθ(yt∣x,y<t)

同理：

log⁡πref(y∣x)=∑t=1Tlog⁡πref(yt∣x,y<t)\log\pi_{\mathrm{ref}}(y\mid x) = \sum_{t=1}^{T} \log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})logπref(y∣x)=t=1∑Tlogπref(yt∣x,y<t)

于是：

log⁡πθ(y∣x)πref(y∣x)=∑t=1Tlog⁡πθ(yt∣x,y<t)πref(yt∣x,y<t)\log \frac{ \pi_\theta(y\mid x) }{ \pi_{\mathrm{ref}}(y\mid x) } = \sum_{t=1}^{T} \log \frac{ \pi_\theta(y_t\mid x,y_{<t}) }{ \pi_{\mathrm{ref}}(y_t\mid x,y_{<t}) }logπref(y∣x)πθ(y∣x)=t=1∑Tlogπref(yt∣x,y<t)πθ(yt∣x,y<t)

这意味着：

完整回答的 Log Probability Ratio，可以分解成每个 Token 的 Log Probability Ratio 之和。

实际实现时，需要一致地处理 Token 序列、终止符以及概率计算范围。

### 10.2 Token-Level KL Reward

因此，在一些 RLHF 实现中，可以构造：

rtKL=−β\(log⁡πθ(yt∣x,y<t)−log⁡πref(yt∣x,y<t)\)\boxed{ r_t^{\mathrm{KL}} = -\beta \left\( \\log\\pi\_\\theta(y\_t\\mid x,y\_{<t}) - \\log\\pi\_{\\mathrm{ref}}(y\_t\\mid x,y\_{<t}) \\right\) }rtKL=−β\(logπθ​(yt​∣x,y<t​)−logπref​(yt​∣x,y<t​)\)

然后将各个 Token 的 KL 相关奖励累加，并与 Reward Model 给出的回答级奖励结合。

例如：

Rsample=rϕ(x,y)−βlog⁡πθ(y∣x)πref(y∣x)R_{\mathrm{sample}} = r_\phi(x,y) - \beta \log \frac{ \pi_\theta(y\mid x) }{ \pi_{\mathrm{ref}}(y\mid x) }Rsample=rϕ(x,y)−βlogπref(y∣x)πθ(y∣x)

为什么这样可以对应 KL 正则化？

因为：

Ey∼πθ\(log⁡πθ(y∣x)πref(y∣x)\)=DKL(πθ∥πref)\mathbb E_{y\sim\pi_\theta} \left\( \\log \\frac{ \\pi\_\\theta(y\\mid x) }{ \\pi\_{\\mathrm{ref}}(y\\mid x) } \\right\) = D_{\mathrm{KL}} ( \pi_\theta\Vert\pi_{\mathrm{ref}} )Ey∼πθ\(logπref​(y∣x)πθ​(y∣x)​\)=DKL(πθ∥πref)

所以，这个逐样本构造在对当前 Policy 生成的回答取期望之后，就对应原来的 KL 正则化项。

这里有一个特别容易弄错的概念：

> 单个 Token 或单条生成序列上的 Log Probability Ratio，并不等同于完整分布上的 KL Divergence。

例如：

log⁡πθ(yt∣⋅)−log⁡πref(yt∣⋅)\log\pi_\theta(y_t\mid\cdot) - \log\pi_{\mathrm{ref}}(y_t\mid\cdot)logπθ(yt∣⋅)−logπref(yt∣⋅)

对某个采样 Token 可以是负数。

但是，真正的 KL Divergence 满足：

DKL(P∥Q)≥0D_{\mathrm{KL}}(P\Vert Q)\geq0DKL(P∥Q)≥0

这并不矛盾。

因为 KL 的非负性是对整个分布取期望后的性质，不要求每个单独样本的 Log Ratio 都非负。

理解了这一点，再去阅读 PPO Trainer 或其他 RLHF 实现，就不会把 `logprob - ref_logprob` 错误理解为一个必然非负的 KL 值。

## 十一、如果完全去掉 KL，会发生什么？

现在终于可以系统性地回答本文的重要问题：

> 为什么 RLHF 不直接最大化 Reward，而要加入 KL？

如果没有 KL，优化目标退化为：

max⁡θEy∼πθ\(rϕ(x,y)\)\max_\theta \mathbb E_{y\sim\pi_\theta} \(r\_\\phi(x,y)\)θmaxEy∼πθ\(rϕ​(x,y)\)

在一个理想化的情形下，假设所有回答及其 Reward 都固定，而且 Policy 可以自由选择任何概率分布。

那么，为了最大化期望 Reward，最优的做法就是：

把尽可能多的概率集中在 Reward 最高的回答上。

如果存在唯一的最高 Reward 回答：

y∗=arg⁡max⁡yrϕ(x,y)y^*=\arg\max_y r_\phi(x,y)y∗=argymaxrϕ(x,y)

那么不受约束的最优分布可以退化为：

π(y∗∣x)=1\pi(y^*\mid x)=1π(y∗∣x)=1

这说明：

单纯的期望 Reward 最大化，并不会自动鼓励模型保持生成多样性，也不会自动保护原有能力。

对于真实语言模型，参数化和优化过程会使情况更加复杂，但这个理想化结论解释了潜在风险。

### 11.1 Policy Drift

模型越来越偏离 Reference Policy。

某些原本合理的生成习惯可能被削弱。

### 11.2 Reward Hacking

模型可能找到提高 Reward Model 分数的捷径。

例如：

* 过度使用自信语气。

* 输出大量没有必要的解释。

* 采用 Reward Model 错误偏好的固定表达。

* 在事实不准确时仍然获得高分。

这些都是可能的表现，不代表所有 Reward Model 都存在相同问题。

### 11.3 Distribution Shift

随着 Policy 分布改变，模型可能生成越来越多 Reward Model 训练时未充分覆盖的回答。

于是，Reward Model 的预测误差可能被进一步放大。

### 11.4 输出分布可能过度集中

如果某种回答形式始终获得较高的 Reward，模型可能不断提高该形式的生成概率。

这可能损害回答的多样性。

因此：

没有 KL⇒缺少对 Reference Policy 偏离的显式约束\boxed{ \text{没有 KL} \Rightarrow \text{缺少对 Reference Policy 偏离的显式约束} }没有 KL⇒缺少对 Reference Policy 偏离的显式约束

但注意：

没有 KL 不意味着一定发生 Reward Hacking；有 KL 也不意味着一定不会发生 Reward Hacking。

KL 是一种重要的风险缓解机制，而不是解决所有对齐问题的万能方法。

## 十二、为什么有了 KL，Reward Hacking 仍然可能存在？

因为 KL 解决的是：

> Policy 相对于 Reference Policy 偏离了多少？

但它不能直接回答：

> Reward Model 给出的评价是否正确？

假设 Reward Model 存在一个偏差：

它稍微偏爱带有自信语气的回答。

而 Reference Policy 本来就会经常使用一些自信表达。

那么 RL 只需要在 Reference Policy 附近调整这些表达的概率，就可能提高 Reward。

此时：

DKLD_{\mathrm{KL}}DKL

可能仍然不大。

但回答的真实性不一定得到提高。

这意味着：

即使 Policy 没有显著偏离 Reference，它仍然可能利用 Reward Model 的局部偏差。

因此，在实际系统中，不能只依赖 KL。

还需要关注：

* Preference Data 是否具有代表性。

* Reward Model 是否具有良好的泛化能力。

* 是否存在明显的评分捷径。

* Reward 的改善能否转化为独立人工评价的改善。

* Policy 是否出现了异常的行为模式。

一个重要结论是：

> KL 可以约束 Policy 的变化，但不能保证 Reward Model 对回答质量的判断永远正确。

## 十三、一个更深刻的数学结论：KL 约束下的最优 Policy 是什么？

前面我们已经用过这个公式：

π∗(y∣x)=1Z(x)πref(y∣x)exp⁡(r(x,y)β)\pi^*(y\mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{r(x,y)}{\beta}\right)π∗(y∣x)=Z(x)1πref(y∣x)exp(βr(x,y))

现在尝试理解它为什么成立。

固定一个 Prompt xxx，暂时省略条件 xxx。

目标是：

max⁡π\(∑yπ(y)r(y)−β∑yπ(y)log⁡π(y)πref(y)\)\max_{\pi} \left\( \\sum\_y\\pi(y)r(y) - \\beta \\sum\_y \\pi(y)\\log \\frac{\\pi(y)}{\\pi\_{\\mathrm{ref}}(y)} \\right\)πmax\(y∑​π(y)r(y)−βy∑​π(y)logπref​(y)π(y)​\)

并满足：

∑yπ(y)=1\sum_y\pi(y)=1y∑π(y)=1

引入拉格朗日乘子 λ\lambdaλ：

F(π,λ)=∑yπ(y)r(y)−β∑yπ(y)log⁡π(y)πref(y)+λ(1−∑yπ(y))\begin{aligned} \mathcal F(\pi,\lambda) =& \sum_y\pi(y)r(y)\\ &-\beta\sum_y\pi(y) \log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}\\ &+\lambda\left(1-\sum_y\pi(y)\right) \end{aligned}F(π,λ)=y∑π(y)r(y)−βy∑π(y)logπref(y)π(y)+λ(1−y∑π(y))

对 π(y)\pi(y)π(y) 求偏导，并令其等于零：

r(y)−β\(log⁡π(y)πref(y)+1\)−λ=0r(y) - \beta \left\( \\log\\frac{\\pi(y)}{\\pi\_{\\mathrm{ref}}(y)} +1 \\right\) -\lambda =0r(y)−β\(logπref​(y)π(y)​+1\)−λ=0

整理：

log⁡π(y)πref(y)=r(y)β−1−λβ\log \frac{\pi(y)}{\pi_{\mathrm{ref}}(y)} = \frac{r(y)}{\beta} -1-\frac{\lambda}{\beta}logπref(y)π(y)=βr(y)−1−βλ

两边取指数：

π(y)=πref(y)exp⁡(r(y)β)exp⁡(−1−λβ)\pi(y) = \pi_{\mathrm{ref}}(y) \exp\left(\frac{r(y)}{\beta}\right) \exp\left(-1-\frac{\lambda}{\beta}\right)π(y)=πref(y)exp(βr(y))exp(−1−βλ)

由于最后一项与 yyy 无关，可以将其并入归一化常数：

Z=∑yπref(y)exp⁡(r(y)β)Z= \sum_y \pi_{\mathrm{ref}}(y) \exp\left(\frac{r(y)}{\beta}\right)Z=y∑πref(y)exp(βr(y))

最终得到：

π∗(y∣x)=πref(y∣x)exp⁡(r(x,y)/β)Z(x)\boxed{ \pi^*(y\mid x) = \frac{ \pi_{\mathrm{ref}}(y\mid x) \exp(r(x,y)/\beta) }{ Z(x) } }π∗(y∣x)=Z(x)πref(y∣x)exp(r(x,y)/β)

这里假设 β>0\beta>0β>0，并且讨论的是具有共同支持集、可以直接优化概率分布的理想化情形。

这个结论非常有意思。

它说明：

KL 正则化下的最优 Policy，相当于在 Reference Policy 的基础上，根据 Reward 对回答概率进行指数加权。

Reward 越高，对应回答的概率越有机会被提高。

但是，Reference Policy 的原始概率也会影响最终结果。

这进一步解释了：

> RLHF 并不是简单地把所有概率分配给 Reward 最高的回答，而是在 Reward 提升与 Reference 分布之间寻找平衡。

这也是理解 DPO（Direct Preference Optimization）的重要数学基础之一。

DPO 利用了 Reward 和 KL 正则化最优 Policy 之间的关系，把部分传统 RLHF 流程转化为直接基于偏好数据的 Policy 优化。\(5\)

## 十四、三个面试高频问题

### Q1：为什么 Reward Model 会被 Hack？

回答：

Reward Model 是从有限的人类偏好数据中学习出来的代理奖励函数，而不是真实人类偏好的完美表示。

它可能具有近似误差、数据偏差或错误相关性。

当 RL 优化器持续最大化 Reward Model 的输出时，会主动寻找能够获得高分的回答，并可能利用这些误差。

随着 Policy 分布发生变化，Reward Model 还可能面临 Distribution Shift，进一步降低评分的可靠性。

因此，Reward Hacking 的本质是：

优化过程利用了代理奖励与真实目标之间的不一致。

### Q2：为什么 Reward 高，不代表回答真的更好？

回答：

因为 Reward Model 预测的是模型所学习的偏好评分，而不是回答质量的绝对真值。

Reward Model 可能错误地依赖长度、语气、格式等表面特征，也可能无法可靠评估训练分布之外的回答。

所以 Reward 高只能说明这个回答在当前 Reward Model 下得分较高，不能保证独立人类评价、事实准确性或实际任务表现一定更好。

### Q3：为什么 RLHF 需要 KL Penalty？

回答：

KL Penalty 用来约束当前 Policy 与 Reference Policy 的分布差异。

在最大化 Reward 的同时加入 KL 正则化，可以减少 Policy 无约束漂移，使模型在改变行为时需要付出一定的分布偏离代价。

它有助于缓解过度优化 Reward Model、生成分布异常集中以及进入 Reward Model 不熟悉区域等问题。

但 KL 不能保证 Reward Model 的正确性，因此它是一种重要的正则化手段，而不是 Reward Hacking 的完全解决方案。

## 十五、把这几个知识点真正串起来

学完 Reward Model 和 KL 后，我认为最重要的不是记住两个公式，而是理解它们之间的因果关系。

```
真实的人类偏好难以直接数学化
                |
                v
       收集人类偏好比较
                |
                v
      Bradley-Terry Model
                |
                v
         训练 Reward Model
                |
                v
     得到人类偏好的代理评分
                |
                v
        使用 RL 最大化 Reward
                |
                v
     优化器可能利用 RM 的误差
                |
                v
  Reward Hacking / Distribution Shift
                |
                v
       引入 KL Regularization
                |
                v
    在 Reward 与 Policy Drift
           之间寻找平衡
```

整个逻辑可以浓缩成：

Human Preference→Reward Model→RL Optimization\boxed{ \text{Human Preference} \rightarrow \text{Reward Model} \rightarrow \text{RL Optimization} }Human Preference→Reward Model→RL Optimization

但由于：

Reward Model≠Perfect Human Utility\boxed{ \text{Reward Model} \neq \text{Perfect Human Utility} }Reward Model=Perfect Human Utility

因此需要：

Reward Maximization+KL Regularization\boxed{ \text{Reward Maximization} + \text{KL Regularization} }Reward Maximization+KL Regularization

最终优化目标可以写为：

max⁡θ Ex∼D\(Ey∼πθ\[rϕ(x,y)\)−βDKL(πθ∥πref)]\boxed{ \begin{aligned} \max_\theta\ \mathbb E_{x\sim\mathcal D} \Big\( &\\mathbb E\_{y\\sim\\pi\_\\theta} \[r\_\\phi(x,y)\)\\ &-\beta D_{\mathrm{KL}} (\pi_\theta\Vert\pi_{\mathrm{ref}}) \Big] \end{aligned} }θmax Ex∼D\(​Ey∼πθ​​\[rϕ​(x,y)\)−βDKL(πθ∥πref)]

这是理解传统 KL 正则化 RLHF 方法的重要基础。

## 十六、学习后的思考：一个微小的 Reward Bias 会造成什么影响？

最后留下一个值得反复思考的问题。

假设 Reward Model 有一个很小的系统性偏差：

> 它倾向于给使用自信语气的回答额外增加 0.1 Reward。

比如：

普通回答：

> 根据目前的信息，这个结论可能成立，但还需要进一步验证。

自信回答：

> 这个结论毫无疑问是正确的。

假设两者在其他质量维度上相近，但 Reward Model 更喜欢后者。

那么，当 RL 优化器持续最大化 Reward 时，可能发生什么？

第一阶段：

模型发现使用某些自信措辞能够获得略高的 Reward。

第二阶段：

Policy 增加这些措辞的生成概率。

第三阶段：

如果缺少真实性约束，模型可能逐渐减少合理的不确定性表达。

第四阶段：

最终模型可能表现得更加自信，但不一定更加准确。

这里需要注意：固定的额外 0.1 Reward 并不意味着模型可以无限增加这个偏差收益。它只能在该特征确实带来收益、且没有被其他优化代价抵消时，推动 Policy 增加对应行为的概率。

但这个例子仍然揭示了一个很重要的现象：

即使是很小的系统性评分偏差，也可能改变长期优化后的模型行为分布。

而这恰好说明：

为什么 RLHF 不能只关注 Reward 上升。

我们还必须同时关注：

* Reward Model 是否可靠。

* Policy 是否产生了非预期行为。

* 模型的真实任务表现是否改善。

* KL 是否有效限制了过度偏离。

* 是否需要引入独立于训练奖励的评价方式。

## 十七、总结

今天学习的 Reward Model 和 KL，实际上对应 RLHF 中两个不同但紧密相关的问题。

Reward Model 解决的是：

> 如何把人类的偏好比较转化为一个可以用于优化的奖励信号？

KL Regularization 解决的是：

> 当我们开始最大化这个奖励信号时，如何限制 Policy 过度偏离参考模型？

而 Reward Hacking 则提醒我们：

> 被优化的指标与真正想实现的目标，并不一定完全一致。

本文最值得记住的是以下五句话：

1. Reward Model 主要通过相对偏好数据学习评分函数，而不是预测有固定刻度的绝对质量分数。

2. Bradley-Terry 模型使用两个回答的 Reward Difference，通过 Sigmoid 将其转换为偏好概率。

3. Reward Model 是人类偏好的代理模型，因此存在近似误差，强化学习可能主动利用这些误差。

4. KL Penalty 通过限制当前 Policy 对 Reference Policy 的分布偏离，缓解无约束 Reward Optimization 带来的风险。

5. 更高的 Reward 不一定意味着更高的真实回答质量；最终仍需要可靠、独立的评价。

我觉得理解这些内容之后，再去学习 PPO、DPO 和其他 Preference Optimization 方法，会更容易理解它们的设计动机，而不是单纯记忆算法公式。

## 参考文献

以下主要选择原始论文、正式会议论文和官方技术资料，方便进一步查阅与验证。

### \(1\) Bradley-Terry 模型的原始论文

Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4), 324–345.

* 论文地址：doi.org 

* 推荐阅读目的： 理解成对比较的概率建模思想，以及 Bradley-Terry 模型的数学基础。

### \(2\) 基于人类偏好微调语言模型

Ziegler, D. M., Stiennon, N., Wu, J., et al. (2019). Fine-Tuning Language Models from Human Preferences.

* arXiv：arxiv.org 

* OpenAI 研究介绍：openai.com 

* 推荐阅读目的： 了解早期语言模型如何利用人类偏好训练 Reward Model，以及奖励优化和标注启发式规则之间可能出现的问题。

### \(3\) InstructGPT：RLHF 的经典实践

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems (NeurIPS 2022).

* 论文地址：arxiv.org 

* NeurIPS 正式论文页面：proceedings.neurips.cc 

* 推荐阅读部分： Section 3，特别是 Reward Model Training 和 Reinforcement Learning。

* 推荐阅读目的： 理解 SFT → Reward Model → PPO 的完整流程，以及 KL 惩罚在实际 RLHF 中的使用方式。

### \(4\) Reward Model 过度优化研究

Gao, L., Schulman, J., & Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835–10866.

* 论文地址：proceedings.mlr.press 

* arXiv：arxiv.org 

* 推荐阅读目的： 理解为什么优化代理 Reward 过度时，实际评价目标可能不再同步改善，以及 Reward Overoptimization 与 KL 系数之间的关系。

### \(5\) DPO：Reward 与最优 Policy 的联系

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.

* 论文地址：arxiv.org 

* HTML 阅读版：arxiv.org 

* 推荐阅读部分： Section 3（Preliminaries）、Section 4（Direct Preference Optimization），以及 Appendix A.1。

* 推荐阅读目的： 理解 Bradley-Terry 模型、KL 正则化奖励最大化和最优 Policy 的数学联系，为后续学习 DPO 做准备。

### \(6\) Hugging Face TRL 官方文档

Hugging Face. TRL — Transformer Reinforcement Learning.

* 官方文档：huggingface.co 

* Trainer 文档：huggingface.co 

* 推荐阅读目的： 对照实际训练代码，理解 `logprobs`、`ref_logprobs`、KL Penalty 和 Reward 的实现方式。需要注意，不同版本和 Trainer 的 KL 实现细节可能有所差异。

系列导航： LLM 强化学习学习笔记 / Day 9 — Reward Model & KL

下一步： 在理解 Reward Model、KL 正则化和 Reward Hacking 之后，可以进一步学习 PPO 中的 Advantage、Policy Ratio 与 Clip Objective，并思考 PPO 如何在这种奖励信号下更新语言模型。
