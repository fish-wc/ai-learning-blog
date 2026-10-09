# Day 19｜FineWeb：什么才是真正的高质量预训练数据？

> **学习主题**：LLM Pretraining · Data Curation · FineWeb
> **核心关键词**：Common Crawl、Filtering、Deduplication、Data Quality、Ablation
> **核心问题**：什么叫高质量预训练数据？如何通过可测量的指标和实验，证明一套数据处理 Pipeline 是有效的？

## 一、引言：为什么预训练数据不是越多越好？

在学习大语言模型（LLM）的过程中，我们经常听到这样一句话：

> 数据质量决定了模型能力的上限。

但这句话其实有一个问题：**什么是数据质量？**

假设我们从互联网上收集了两份数据：

* **Dataset A**：包含大量网页，其中有重复文章、广告、导航菜单、低信息密度内容。
* **Dataset B**：规模更小，但经过正文提取、语言过滤、去重和质量筛选。

如果只看数据规模，Dataset A 显然更大。

但如果用同一个模型、相同的训练 Token 数，分别在 A 和 B 上训练，结果发现 B 的模型在知识理解、常识推理等任务上表现更好，那么我们就有理由认为：

**在当前训练目标和预算下，Dataset B 提供了更有效的学习信号。**

这正是 FineWeb 想解决的问题。

2024 年，Hugging Face 团队发布了 [FineWeb](https://arxiv.org/abs/2406.17557)，一个从 Common Crawl 中构建的大规模英文预训练数据集。

原始论文中的规模是：

* FineWeb：约 15T Tokens。
* 数据来源：96 个 Common Crawl 快照。
* FineWeb-Edu：从 FineWeb 中进一步筛选得到的约 1.3T 教育类 Tokens。

与单纯发布数据集不同，FineWeb 最值得学习的地方在于：

**它通过大量受控实验（Ablation），系统地研究了哪些数据处理方法真的能够提升模型性能。**

本文只关注五件事情：

1. Data Source：原始数据从哪里来？
2. Filtering：如何过滤明显无效的数据？
3. Deduplication：怎样处理重复数据？
4. Quality：如何将数据质量转换为可测量指标？
5. Ablation：如何证明一个过滤策略真的有效？

> **版本说明**：本文讨论的是 NeurIPS 2024 论文对应的原始 FineWeb（15T Tokens）。官方数据集后来扩展到 18.5T Tokens 以上，因此不同版本的规模统计可能存在差异。[1][3]

---

## 二、Data Source：FineWeb 的原始数据从哪里来？

### 2.1 Common Crawl：互联网数据的原材料

FineWeb 的原始数据来自 [Common Crawl](https://commoncrawl.org/)。

Common Crawl 是一个公开网页抓取项目，持续收集互联网网页，并提供开放的数据归档。

但需要注意：

**抓取到一个网页，并不意味着获得了一篇适合训练语言模型的文章。**

例如，网页 HTML 中可能包含：

```text
Home
Products
Login
Accept Cookies

<script>...</script>

Advertisement
Advertisement

What is machine learning?

Machine learning is a field of artificial
intelligence that enables systems to learn
patterns from data.

Privacy Policy
Contact Us
```

如果把这些内容直接作为训练数据，会有什么问题？

模型不只会学习机器学习相关知识，还会反复看到：

* 页面导航和按钮；
* 广告与网站模板；
* Cookie 提示；
* 隐私政策；
* 各种与文章正文无关的文本。

这些内容并非全部没有价值，但当它们大规模重复出现时，就可能浪费训练预算。

因此，预训练数据处理的第一步通常不是过滤低质量文章，而是：

**先从网页中正确提取正文。**

### 2.2 为什么 FineWeb 选择 WARC，而不是直接使用 WET？

Common Crawl 提供不同的数据格式。

| 格式   | 含义                     | 特点                |
| ---- | ---------------------- | ----------------- |
| WARC | Web ARChive            | 包含原始网页 HTML 等抓取内容 |
| WET  | WARC Encapsulated Text | 已经过文本提取的网页内容      |

直接使用 WET 比较方便，因为已经省去了部分 HTML 解析工作。

但 FineWeb 发现，Common Crawl 默认的文本提取方式可能保留过多网页模板内容。

因此，FineWeb 从 WARC 出发，使用 **Trafilatura** 重新提取网页正文。

团队在 `CC-MAIN-2019-18` 数据上进行了对比：

| 文本提取方式             |      处理后的数据规模 |
| ------------------ | ------------: |
| Common Crawl WET   | 约 254B Tokens |
| WARC + Trafilatura | 约 200B Tokens |

尽管 Trafilatura 提取出的 Token 更少，但在随后相同处理条件下的模型训练对比中，表现更好。[2]

为什么？

因为 WET 中多出来的内容，不少是导航菜单、网页模板等低价值文本。

这说明：

**一个好的数据提取器，不应该以提取最多的文字为目标，而应该尽量保留正文、减少无关噪声。**

当然，这并不意味着 Trafilatura 在所有场景中都一定最优。文本提取本身存在计算成本，具体方案仍然需要实验验证。

---

## 三、Filtering：如何过滤不适合训练的数据？

完成正文提取以后，数据依然存在大量质量问题。

例如：

```text
Document A:
Deep learning models learn representations
from large-scale training data.

Document B:
BUY NOW BUY NOW BUY NOW
FREE FREE FREE FREE

Document C:
这是一篇中文机器学习教程。
```

对于一个以英文网页为目标的预训练数据集：

* Document A：可以考虑保留。
* Document B：重复性强，信息密度低。
* Document C：不符合当前英文语料目标。

这里要理解一个重要概念：

**Filtering 不是凭感觉判断文章是否“高级”，而是使用可以批量计算的规则过滤数据。**

### 3.1 FineWeb 的基础过滤规则

FineWeb 主要采用以下基础过滤方式。[1][2]

| 过滤步骤                 | 技术或指标                              | 目的           |
| -------------------- | ---------------------------------- | ------------ |
| URL Filtering        | URL Blocklist                      | 过滤已知不合适的网站来源 |
| Language Filtering   | fastText Language Probability      | 保留目标语言       |
| Quality Filtering    | MassiveText / Gopher Quality Rules | 去除明显异常的文本    |
| Repetition Filtering | 重复行、重复 N-gram 等统计特征                | 过滤内部高度重复的文档  |

其中一个具体阈值是：

```text
English Language Score >= 0.65
```

这是什么意思？

FineWeb 使用 fastText 语言识别模型，估计文档属于英文的概率或置信分数。

假设有两篇文档：

```text
Document A:
English Score = 0.93

Document B:
English Score = 0.41
```

对于目标为英文的数据集：

```text
Document A -> Keep
Document B -> Drop
```

这就把“是不是英文”转化为了一个可计算的指标。

经过基础过滤以后，FineWeb 从原始网页文本中得到了大约 **36T Tokens**。

需要强调的是：`0.65` 是 FineWeb 在特定语料与模型配置下采用的阈值，不是适用于所有语言和所有数据集的通用常数。

### 3.2 Filtering 的本质是什么？

我们可以把过滤器看成一个函数：

$$
F(d)\in\{0,1\}
$$

其中：

* \(d\)：一篇文档；
* \(F(d)=1\)：保留文档；
* \(F(d)=0\)：丢弃文档。

例如：

```text
Raw Document
      |
      v
Language Filter
      |
      v
Repetition Filter
      |
      v
Quality Filter
      |
      v
Keep / Drop
```

但这里会出现一个重要问题：

**规则能够帮助我们发现明显异常的数据，却不代表规则本身一定合理。**

例如，一篇代码教程可能包含大量特殊符号。

如果简单地认为“特殊符号多就是垃圾”，就可能误删大量有价值的代码数据。

因此，Filtering 存在一个非常重要的权衡：

* 过滤太松：会保留大量噪声。
* 过滤太严：可能误删有效数据，损害多样性。

究竟怎样选择？

答案不是继续依赖直觉，而是进行 **Ablation**。

在讨论实验之前，我们还需要理解 FineWeb 中一个非常反直觉的发现。

---

## 四、Deduplication：为什么重复数据不是删得越多越好？

### 4.1 重复数据为什么有害？

互联网存在大量重复内容。

例如，一篇技术文章可能同时出现在：

```text
Original Blog
      |
      +--> Repost Website
      |
      +--> Mirror Website
      |
      +--> Content Aggregator
```

假设训练数据中有以下样本：

```text
A A A A A A A A B C
```

虽然我们拥有十个样本位置，但模型看到的内容分布是：

```text
A: 80%
B: 10%
C: 10%
```

此时，A 的影响被过度放大。

这会带来几个问题：

1. 大量训练 Token 被重复内容占据。
2. 训练数据的主题分布可能被扭曲。
3. 模型看到的不同知识和表达方式减少。
4. 高度重复的文本可能增加模型记忆训练样本的风险。

所以，去重的目标并不只是减少文件大小。

**去重真正想做的是控制重复内容对训练分布的影响。**

### 4.2 FineWeb 使用 MinHash 进行近重复检测

有两种常见的重复数据。

**第一种：完全重复。**

```text
A:
Machine learning is a subset of AI.

B:
Machine learning is a subset of AI.
```

这可以通过文本 Hash 很容易地检测。

**第二种：近似重复。**

```text
A:
Machine learning is a subset of AI.
Published in 2023.

B:
Machine learning is a subset of AI.
Published in 2024.
```

两篇文档并不完全相等，但高度相似。

FineWeb 使用 **MinHash** 来检测这种近重复文档。

可以简单理解为：

```text
Document
   |
   v
Split into Word 5-grams
   |
   v
Compute MinHash Signatures
   |
   v
Find Similar Documents
   |
   v
Deduplicate
```

这里的 5-gram 指连续五个词构成的片段。

例如：

```text
deep learning models can learn
learning models can learn useful
models can learn useful representations
```

两个文档共享越多这样的片段，通常说明它们在表面文本上越相似。

FineWeb 使用了以下配置：

* Word 5-grams；
* 112 个 MinHash Hash Functions；
* 14 个 Buckets，每个包含 8 个 Hashes；
* 目标是高效发现大约 75% 及以上相似度的文档。

需要注意，75% 并非严格的分界线。

对于相似度为 \(s\) 的文档，在该 MinHash 配置下，候选匹配概率近似为：

$$
P(\text{match})=1-(1-s^8)^{14}
$$

例如，当相似度为 0.75 时，匹配概率约为 77%。

这说明 MinHash 是一种概率性的近重复检测技术，而不是简单的“相似度大于阈值就必然删除”。[2]

### 4.3 FineWeb 的反直觉实验：全局去重反而不理想

一开始，研究人员认为：

> 既然重复数据不好，那就把所有 Common Crawl 数据放在一起，全局去重。

于是他们进行了跨 Crawl 的迭代 MinHash 去重。

结果：

```text
Base Filtering
约 36T Tokens
      |
      v
Global / Iterative MinHash
      |
      v
约 4T Tokens
```

数据规模大幅下降。

直觉上，这应该得到一个更加“纯净”的数据集。

但实验结果却并不理想：

在相同训练 Token 预算下，这种激进的跨 Crawl 去重方式相对于未去重数据几乎没有明显改善，整体表现也落后于 RefinedWeb。[2]

为什么会这样？

我们考虑一个简单的例子。

假设三个年份的抓取结果如下：

```text
2021:
优秀教程 A
优秀教程 B
垃圾网页 X

2022:
优秀教程 A
优秀教程 B
垃圾网页 Y

2023:
优秀教程 A
优秀教程 B
垃圾网页 Z
```

如果进行全局去重：

```text
A -> 保留一次
B -> 保留一次

X -> 保留
Y -> 保留
Z -> 保留
```

结果：

```text
A
B
X
Y
Z
```

高质量但经常被转载的内容被大量删除。

而一些独一无二、但质量不好的网页，却被保留下来。

需要说明：这是用于理解原理的简化示例，并不代表 FineWeb 已证明所有重复内容都优质、所有独特内容都低质。

FineWeb 的实验观察是：

**激进去重可能使剩余数据中低质量、分布异常的内容占比上升。**

团队也在早期 Common Crawl 快照的分析中发现，全局去重后留下的部分文本包含更多广告、关键词列表和格式异常内容。

### 4.4 FineWeb 最终采用 Per-Crawl Deduplication

经过对比，团队最终选择：

**分别对每个 Common Crawl Snapshot 进行去重。**

```text
Crawl 1 -> MinHash Dedup
Crawl 2 -> MinHash Dedup
Crawl 3 -> MinHash Dedup
...
                 |
                 v
            Merge Data
```

也就是说：

* 同一个 Crawl 内部尽量去除近重复文档。
* 不强制把不同 Crawl 之间的相似内容全部清除。

实验中的结果可以总结为：

| 去重方案           |         数据规模 | 实验观察                        |
| -------------- | -----------: | --------------------------- |
| 跨 Crawl 全局迭代去重 |  约 4T Tokens | 效果不理想                       |
| 每个 Crawl 独立去重  | 约 20T Tokens | 明显优于前者，达到与 RefinedWeb 接近的表现 |

注意：这里的 20T 还不是最终的 FineWeb，后面仍然需要做额外的质量过滤。

**这部分最重要的结论是：**

> 去重不是追求最低重复率，而是在减少无意义重复的同时，尽量保留对模型有价值的数据分布。

换句话说：

$$
\text{Less Duplication}\not\Rightarrow\text{Better Model}
$$

**去重策略的好坏，最终仍然需要通过训练实验判断。**

---

## 五、Quality：如何将“数据质量”变成具体指标？

这是我认为 FineWeb 最值得深入理解的部分。

因为我们终于要回答：

**“高质量”究竟如何测量？**

### 5.1 为什么不能只依赖人工判断？

假设有一份万亿 Token 级别的数据集。

我们不可能人工阅读所有文档，然后判断：

```text
这篇不错 -> Keep
那篇一般 -> Drop
```

所以必须采用能够自动计算的指标。

例如：

* 文档包含多少行？
* 平均每行有多少字符？
* 有多少行以标点符号结束？
* 是否出现大量重复行？
* 短行占比是否异常？
* 语言识别分数是多少？

这些指标不一定能直接衡量“文章是否有知识价值”，但有可能识别某些具有明显格式特征的垃圾数据。

### 5.2 FineWeb 是如何找到有效过滤规则的？

FineWeb 没有直接拍脑袋设定所有阈值，而是采用了一个系统性的实验过程：

```text
准备表现较好与较差的数据版本
                 |
                 v
计算 50+ 文本统计特征
                 |
                 v
比较特征的统计分布
                 |
                 v
寻找差异明显的特征
                 |
                 v
观察直方图并提出过滤阈值
                 |
                 v
在候选数据上应用 Filter
                 |
                 v
训练小模型进行 Ablation
                 |
                 v
只保留经验证有效的规则
```

其中使用了 **Wasserstein Distance** 来辅助比较特征分布。

可以把它直观理解成：

> 如果两批数据在某个指标上的分布差异很大，那么这个指标可能值得进一步研究。

但“分布差异大”不代表这个指标一定能够判断质量。

研究人员还必须继续检查样本，并通过模型训练验证。

最终，在提出的一批候选规则中，FineWeb 找到了三个特别有效的自定义统计过滤器。[1][2]

### 5.3 三个关键质量指标

| 指标      | 计算方式                | FineWeb 论文报告的丢弃条件 |
| ------- | ------------------- | ----------------- |
| 标点结束行比例 | 以终止标点结束的行数 / 总行数    | ≤ 0.12            |
| 重复行字符比例 | 重复行涉及的字符数 / 文档字符数   | ≥ 0.10            |
| 短行比例    | 长度小于 30 字符的行数 / 总行数 | ≥ 0.67            |

这些阈值来自原始论文的消融实验；复现时还需对照相应版本的 DataTrove 实现及其具体边界条件。

下面逐个解释。

#### 指标一：Lines Ending with Punctuation Ratio

考虑两篇文档。

**Document A：**

```text
Deep learning is a subset of machine learning.
It uses neural networks to learn representations.
These representations are useful for many tasks.
```

**Document B：**

```text
Home
Login
Register
Products
Free Download
Contact
```

A 中的大部分行都以句号结束。

B 中则几乎没有。

我们可以定义：

$$
R_{\text{punct}}=
\frac{\text{以终止标点结尾的行数}}
{\text{非空行总数}}
$$

这个指标可以帮助识别某些导航栏、关键词列表和网页模板。

FineWeb 最终采用的阈值是：

```text
punctuation_line_ratio <= 0.12
=> Drop
```

但这里有一个值得思考的问题：

代码、诗歌、表格也可能没有很多句号。

所以这个指标**只是具有统计意义的代理特征，并不是衡量文章真实性或知识价值的绝对标准。**

#### 指标二：Duplicated-Line Character Ratio

假设某篇文章长这样：

```text
Buy now and save money.
Buy now and save money.
Buy now and save money.
Buy now and save money.

This product is available today.
```

前面的大部分内容都在重复。

我们可以通过重复行所涉及的字符比例，估计一篇文档有多少内容被重复占据。

FineWeb 使用的过滤条件是：

```text
duplicate_line_char_ratio >= 0.10
=> Drop
```

这个规则想识别的是：

**表面上文字很多，但其中大量内容可能是重复模板或重复段落的文档。**

#### 指标三：Short-Line Ratio

假设有一篇网页：

```text
Home
About
Docs
Blog
Community
Pricing
Support
FAQ
Contact
```

这种文本可能不是自然语言正文，而是网页布局信息。

我们定义：

$$
R_{\text{short}}=
\frac{\text{长度小于30字符的行数}}
{\text{非空行总数}}
$$

FineWeb 的规则为：

```text
short_line_ratio >= 0.67
=> Drop
```

也就是当文档中很大比例的行都特别短时，将它视为需要过滤的候选。

同样，这个规则也可能误伤列表、诗歌或者其他非标准排版内容。

这正是为什么要进行 Ablation，而不能单纯追求规则覆盖率。

### 5.4 更严格的过滤不一定更好

FineWeb 还复现和研究了 C4 数据集的一系列过滤规则。

其中一个规则是 Terminal Punctuation Filter。

简单理解，就是对行末标点进行较严格的要求。

实验发现：

* 单独使用这个较严格的过滤器，可以删除约 30% 的 Tokens，并带来一定效果提升。
* 但保留其他 C4 质量过滤规则、去掉这个过于激进的规则时，整体表现反而更好，且仅删除约 7% 的 Tokens。[2]

这说明：

> 一个特征与高质量相关，并不意味着把它的过滤阈值调得越严格越好。

FineWeb 后来使用三个自定义统计规则，一起过滤掉约 22% 的 Tokens，并在实验中取得进一步改善。

最终，原始 FineWeb 的数据规模约为：

```text
Base Filtering
       |
       v
36T Tokens
       |
       v
Per-Crawl MinHash Dedup
       |
       v
20T Tokens
       |
       v
Selected C4 Filters
+
FineWeb Custom Quality Filters
       |
       v
15T Tokens
```

这张图表示论文中的主要研究和处理阶段。生产实现中，部分过滤和去重操作的实际执行顺序可以为了工程效率进行调整。

到这里，我们已经从互联网原始网页，得到了一份经过系统处理的预训练数据集。

不过，还有一个问题没有解决：

**这些规则主要识别的是文本的表面特征，它们并不能直接判断文章是否具有教育价值。**

---

## 六、FineWeb-Edu：怎样衡量文本的知识价值？

前面三个规则可以检测：

```text
这篇文章是否有很多短行？
这篇文章是否严重重复？
这篇文章是否像网页导航菜单？
```

但无法直接回答：

```text
这篇文章能否有效地教会模型某个知识？
```

例如：

**Document A：**

```text
Photosynthesis is the process by which plants
convert light energy into chemical energy...
```

**Document B：**

```text
My favorite thing about summer is spending
time with friends and visiting new places...
```

两篇文章都可能是结构完整、没有严重重复的自然语言文本。

但它们提供的学习信号不一样。

A 可能具有比较明确的科学教育价值。

B 则可能提供生活常识、日常表达和叙事语言方面的学习信号。

因此，不能简单说 A 永远比 B 更有价值。

如果目标是提升知识问答和科学推理能力，A 可能更符合当前需求。

这就引出了 FineWeb-Edu。

### 6.1 使用大模型给数据打教育价值分数

FineWeb-Edu 采用了一个更接近语义层面的筛选方法：

**让较强的大模型帮助判断文档的教育价值。**

基本过程如下：

```text
FineWeb Documents
        |
        v
Llama-3-70B-Instruct
        |
        v
Educational Quality Score
        |
        v
0 / 1 / 2 / 3 / 4 / 5
```

FineWeb 团队让 Llama-3-70B-Instruct 为约 500K 篇文档生成教育价值评分。

但问题是：

对万亿 Token 规模的数据全部调用一个 70B 大模型，成本非常高。

怎么办？

团队又把这些 LLM 生成的分数作为监督信号，用来训练一个更便宜的分类模型。

```text
Sample Documents
       |
       v
Teacher LLM Annotation
       |
       v
Educational Scores
       |
       v
Train Small Quality Classifier
       |
       v
Apply to Massive Dataset
```

官方报告，该分类器在以阈值 3 定义的教育内容二分类验证任务上，F1 Score 约为 82%。需要注意，验证时主要使用 Llama 3 的标签作为参考，并不代表它与人类客观判断有 82% 的绝对一致性。[2][7]

### 6.2 FineWeb-Edu 的过滤结果

FineWeb-Edu 选择：

```text
Educational Score >= 3
=> Keep
```

得到的数据规模约为：

```text
FineWeb
15T Tokens
      |
      v
Educational Quality Classifier
      |
      v
Score >= 3
      |
      v
FineWeb-Edu
1.3T Tokens
```

也就是说，原始数据中约九成被筛掉了。

但是，在 MMLU、ARC、OpenBookQA 等知识和推理相关任务上，FineWeb-Edu 的训练效果显著改善。

团队还报告，在其特定实验设置中，与 C4、Dolma 相比，FineWeb-Edu 达到相近 MMLU 表现所需要的训练 Tokens 可以减少约一个数量级。[2]

这揭示了一个非常重要的概念：

**Token Efficiency（Token 效率）。**

我们不仅关心：

> 模型最终能达到多高的性能？

还可以关心：

> 达到某个性能目标，究竟需要多少训练 Tokens？

如果 Dataset A 需要更多 Tokens 才能达到相同的任务表现，而 Dataset B 使用更少 Tokens 就能做到，那么 B 对当前任务可能有更高的训练效率。

### 6.3 为什么教育价值阈值不是越高越好？

这里还有一个值得注意的实验结果。

FineWeb-Edu 测试了更高的教育价值阈值。

随着过滤更严格，知识和推理类任务可能继续改善。

但是，HellaSwag 和 PIQA 等常识类任务的表现可能下降。

这很容易理解：

如果模型看到的大部分内容都偏向教材、科学解释和知识问答，那么它可能减少接触其他类型自然语言的机会。

例如：

* 日常表达；
* 生活常识；
* 情境描述；
* 故事和叙事；
* 非正式语言。

因此：

$$
\text{Educational Quality}\neq\text{Overall Data Quality}
$$

教育价值是数据质量的一个维度，但不是唯一维度。

**好的预训练数据分布，需要与模型最终希望获得的能力相匹配，同时兼顾信息价值和多样性。**

---

## 七、Ablation：怎样证明一个数据处理步骤真的有效？

这是 FineWeb 整篇工作的核心方法论。

我们已经知道，FineWeb 使用了很多规则：

* Language Filter；
* Repetition Filter；
* MinHash Dedup；
* C4 Quality Filters；
* Custom Quality Filters；
* Educational Quality Classifier。

但有一个问题：

**我们如何知道模型变强究竟来自哪个步骤？**

答案是 Ablation Study，即消融实验。

### 7.1 什么是 Ablation？

假设我们提出一个新的质量过滤器：

```text
Filter X:
Remove documents with excessive short lines.
```

我们想知道它是否有效。

可以准备两份数据集：

```text
Dataset A:
Base Pipeline

Dataset B:
Base Pipeline + Filter X
```

然后训练两个模型。

必须尽量保持以下条件相同：

| 实验条件                    | Model A   | Model B   |
| ----------------------- | --------- | --------- |
| 模型结构                    | 相同        | 相同        |
| 参数规模                    | 相同        | 相同        |
| Tokenizer               | 相同        | 相同        |
| Optimizer / LR Schedule | 相同        | 相同        |
| 训练 Token 预算             | 相同        | 相同        |
| 评估任务与方法                 | 相同        | 相同        |
| 训练数据                    | Dataset A | Dataset B |

这样，主要变化就集中在 Filter X 对训练数据的影响上。

如果：

```text
Model A:
Benchmark Score = S_A

Model B:
Benchmark Score = S_B
```

那么我们可以观察：

$$
\Delta S=S_B-S_A
$$

如果多项代表性任务上出现稳定改善，我们就获得了支持这个过滤器有效的实验证据。

但如果提升很小，还必须考虑随机种子、采样差异、测量噪声等因素。

不能因为某一次分数更高，就立刻认为一个 Filter 有效。

### 7.2 FineWeb 如何进行 Ablation？

FineWeb 使用了较小的模型进行快速实验。

原始实验中的典型配置包括：

* 模型参数量：约 1.82B；
* 架构：Llama-style；
* Tokenizer：GPT-2；
* 常见的快速 Ablation 训练预算：约 28B Tokens；
* 更长的验证训练：最高约 350B Tokens。

评估任务覆盖：

* MMLU；
* ARC；
* HellaSwag；
* PIQA；
* OpenBookQA；
* CommonSenseQA 等。[1][2]

为什么不直接训练一个巨大的模型？

因为数据处理策略可能需要反复修改。

如果每测试一个 Filter，都训练一次超大模型，实验成本会非常高。

因此，FineWeb 使用小模型观察较早出现的训练信号，再通过更大训练预算验证关键结论。

不过，小模型实验也是一种代理评估：小规模训练中有效的策略，不保证在所有模型规模和数据规模下都同样有效。

### 7.3 Ablation 应该观察哪些指标？

我认为，实际做数据工程时，至少需要记录两类指标。

**第一类：数据 Pipeline 指标。**

| 指标                              | 作用        |
| ------------------------------- | --------- |
| Input Tokens                    | 输入数据规模    |
| Retained Tokens                 | 处理后剩余数据规模 |
| Token Retention Ratio           | 数据保留比例    |
| Language Score Distribution     | 检查语言分布    |
| Duplicate / Near-Duplicate Rate | 检查重复情况    |
| Quality Feature Distribution    | 检查过滤特征    |
| Domain / Topic Coverage         | 检查数据多样性   |

**第二类：模型训练与评估指标。**

| 指标                           | 作用            |
| ---------------------------- | ------------- |
| Validation Loss              | 观察模型拟合效果      |
| Benchmark Scores             | 观察目标任务性能      |
| Tokens to Target Performance | 衡量达到目标性能所需数据量 |
| Performance Across Seeds     | 检查结果稳定性       |
| Category-wise Performance    | 检查是否牺牲其他能力    |

其中，Validation Loss 和 Perplexity 可以作为诊断指标，但不能单独作为数据质量的最终判断标准。

因为在某个验证分布上更容易预测，不一定意味着模型在目标下游任务上更有能力。

### 7.4 我认为最值得学习的实验思路

如果由我设计一个简单的数据过滤实验，我会这样做：

```text
Step 1:
建立一个 Baseline Dataset

Step 2:
提出一个新的过滤规则

Step 3:
生成 Filtered Dataset

Step 4:
统计数据保留率和特征分布变化

Step 5:
使用相同模型与训练预算分别训练

Step 6:
在固定 Benchmark Suite 上评估

Step 7:
比较性能提升、下降及稳定性

Step 8:
决定是否保留过滤规则
```

这里还有一个容易忽略的细节：

**固定训练 Token 数，不等于实验的所有潜在偏差都被消除了。**

例如，如果过滤后数据集太小，模型不得不重复采样同一批数据，那么训练时的重复率可能又发生变化。

所以还需要记录采样策略、数据覆盖度和实际 Epoch 情况。

从这个角度看，数据工程并不是简单地写几个清洗脚本。

它更像是：

**提出假设 → 修改数据分布 → 训练验证 → 接受或拒绝假设。**

---

## 八、动手理解：用 Python 实现一个极简质量过滤器

为了真正理解这些指标如何落地，可以先实现一个教学版的 Quality Filter。

这里只展示 FineWeb 论文中三个统计特征的基本思想：

* 标点结束行比例；
* 重复行字符比例；
* 短行比例。

**注意：以下是用于学习指标计算的简化代码，不是 FineWeb/DataTrove 官方实现。** 它省略了正式 Pipeline 中的其他过滤规则，重复行的统计方式也作了简化，不能用于严格复现论文结果。

```python
from collections import Counter


def measure(text: str):
    lines = [
        s.strip()
        for s in text.splitlines()
        if s.strip()
    ]

    if not lines:
        return None

    count = Counter(lines)
    chars = sum(len(s) for s in lines)

    return {
        "punct_ratio": (
            sum(s.endswith((".", "!", "?")) for s in lines)
            / len(lines)
        ),
        "repeat_char_ratio": (
            sum(len(s) for s in lines if count[s] > 1)
            / chars
        ),
        "short_line_ratio": (
            sum(len(s) < 30 for s in lines)
            / len(lines)
        ),
    }


def keep(text: str) -> bool:
    m = measure(text)

    return m is not None and (
        m["punct_ratio"] > 0.12
        and m["repeat_char_ratio"] < 0.10
        and m["short_line_ratio"] < 0.67
    )


nav = "Home\nSign in\nPricing"

prose = (
    "Photosynthesis transforms sunlight into chemical energy in plants.\n"
    "This process supports ecosystems and helps plants grow."
)

print(keep(nav))    # False
print(keep(prose))  # True
```

从这个例子可以看到，所谓“质量过滤”，在工程上可以被实现为：

```text
Document
    |
    v
Calculate Features
    |
    v
Compare with Thresholds
    |
    v
Keep / Drop
```

但请注意：

**这段代码只能演示规则如何计算，不能证明这些规则对模型训练一定有效。**

要证明它们有效，还需要使用保留前后的语料进行真正的模型训练与 Ablation。

这正是 FineWeb 与普通数据清洗教程之间最大的区别。

---

## 九、最终回答：什么叫高质量预训练数据？

经过 FineWeb 的学习，我认为不能这样定义：

> 高质量预训练数据就是内容准确、重复较少、质量比较高的数据。

这个回答有两个问题：

第一，“质量比较高”没有明确的测量方式。

第二，即使重复率下降、网页更加干净，也不能直接证明模型一定会变强。

一个更有工程意义的定义是：

> **高质量预训练数据，是针对明确的模型训练目标，经过可量化的正文提取、语言过滤、结构质量过滤、重复控制以及必要的语义质量筛选后，在固定模型结构和训练预算下，能够提供更有效学习信号，使模型在代表性下游任务上取得更好表现，或者使用更少训练 Tokens 达到相同性能水平的数据分布。**

这个定义可以拆成三个层次。

### 第一层：Measurable——质量必须可以测量

例如：

| 质量维度   | 可测量指标                         |
| ------ | ----------------------------- |
| 语言一致性  | Language Identification Score |
| 文本结构   | Punctuation Line Ratio        |
| 文本格式   | Short-Line Ratio              |
| 文档内部重复 | Repeated-Line Character Ratio |
| 跨文档重复  | MinHash / Near-Duplicate Rate |
| 教育价值   | Educational Quality Score     |
| 数据覆盖度  | Topic / Domain Distribution   |

这些指标能够帮助我们观察数据特征。

但单一指标的数值好坏，不等于最终模型质量。

### 第二层：Pipeline——质量必须能够通过工程流程实现

以原始 FineWeb 的思路为例：

```text
Common Crawl (WARC)
          |
          v
URL Filtering
          |
          v
Trafilatura Text Extraction
          |
          v
Language Filtering
          |
          v
Base Quality / Repetition Filtering
          |
          v
Per-Crawl MinHash Deduplication
          |
          v
Selected C4 Filters
          |
          v
FineWeb Custom Quality Filters
          |
          v
FineWeb
          |
          +----------------------+
          |                      |
          v                      v
 General Pretraining     Educational Classifier
                                 |
                                 v
                         FineWeb-Edu
```

随后：

```text
Candidate Dataset
        |
        v
Controlled Model Training
        |
        v
Downstream Evaluation
        |
        v
Ablation Comparison
        |
        v
Validate Pipeline Decisions
```

只有一套可重复执行的数据处理流程，才能把抽象的数据质量要求转变成实际工程能力。

### 第三层：Eval-driven——质量最终要由模型表现验证

真正的问题不应该只是：

```text
我们删除了多少垃圾数据？
数据重复率下降了多少？
过滤后的文本是否更加规范？
```

而应该进一步问：

```text
在相同训练预算下：

模型知识能力是否提高？
模型推理能力是否提高？
常识能力是否下降？
不同任务之间是否出现明显权衡？
达到目标性能需要的 Tokens 是否减少？
```

也就是说：

**数据处理指标负责描述和约束 Pipeline，模型评估负责验证其实际收益。**

还需要注意，Benchmark 性能不是质量的全部。数据中的隐私风险、有害内容、版权问题、分布偏差和评估数据污染，也需要单独检查。

一个数据集即使能够提升模型分数，也不代表它在安全性、合法性和代表性方面没有问题。

因此，“高质量”始终是一个与训练目标、评估方式和约束条件相关的概念，而不是所有场景下都成立的绝对标签。

---

## 十、总结：FineWeb 给我的五个核心启发

### 1. 更多 Tokens 不一定意味着更多有效知识

网页正文提取本身就可能大幅改变数据质量。

有时候减少网页模板和无关内容，比单纯增加数据规模更有价值。

### 2. 去重不是越激进越好

FineWeb 最有启发性的发现之一，就是跨 Crawl 的激进去重可能损害数据分布。

重复率下降不等于模型性能提高。

### 3. 高质量需要可计算的代理指标

语言分数、重复行比例、短行比例、MinHash 相似性和教育价值分数，都能让数据处理从抽象判断变成可执行的工程流程。

但它们只是代理指标，不是质量本身。

### 4. 不存在适用于所有目标的单一质量阈值

FineWeb-Edu 说明，更严格的教育价值过滤可能改善知识类任务，却损害部分常识任务。

因此，数据筛选要与目标能力相匹配。

### 5. Ablation 是数据质量决策的关键

不能因为一个过滤器“看起来合理”，就认定它有效。

必须通过受控训练实验，检验其对目标任务的影响。

**我认为，FineWeb 最值得带走的不是某一个固定阈值，而是一种数据工程思维：**

```text
Define Training Objectives
           |
           v
Measure Data Properties
           |
           v
Design Filtering / Dedup Strategy
           |
           v
Run Controlled Ablations
           |
           v
Measure Downstream Performance
           |
           v
Keep Validated Improvements
```

最终，我们要优化的并不是“数据看起来有多干净”，而是：

> **在给定训练目标与计算预算下，如何让每一个用于训练的 Token，尽可能提供有效的学习信号。**

这就是我从 FineWeb 中理解的“高质量预训练数据”。

---

## 参考文献

**[1] FineWeb 原始论文**

Penedo, G., Kydlíček, H., Ben Allal, L., et al. (2024). *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. NeurIPS 2024, Datasets and Benchmarks Track.

* Paper: https://arxiv.org/abs/2406.17557
* NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf

**[2] Hugging Face 官方 FineWeb 技术报告**

Hugging Face. (2024). *FineWeb: Decanting the Web for the Finest Text Data at Scale*.

详细介绍了 FineWeb 的数据提取、去重实验、C4 过滤规则、自定义质量指标与 FineWeb-Edu 的构建方法。

* https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

**[3] FineWeb 官方数据集**

HuggingFaceFW. *FineWeb Dataset*.

包含数据集说明、数据来源、处理流程及数据版本信息。

* https://huggingface.co/datasets/HuggingFaceFW/fineweb

**[4] DataTrove：FineWeb 数据处理框架**

Hugging Face. *DataTrove*.

用于大规模预训练数据过滤、去重、统计和处理的开源框架。FineWeb 的处理 Pipeline 可以参考官方示例。

* GitHub：https://github.com/huggingface/datatrove
* FineWeb Pipeline：https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py
* Quality Filter 实现：https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/fineweb_quality_filter.py

**[5] FineWeb-Edu 官方数据集**

HuggingFaceFW. *FineWeb-Edu Dataset*.

提供基于教育价值评分筛选得到的预训练文本。

* https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

**[6] Common Crawl**

Common Crawl Foundation. *Open Repository of Web Crawl Data*.

FineWeb 的原始网页数据来源。

* https://commoncrawl.org/

**[7] FineWeb-Edu Classifier**

HuggingFaceFW. *FineWeb-Edu Classifier*.

用于评估网页文本教育价值的模型，可用于进一步理解基于 LLM 标注的语义质量筛选。

* https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier

**[8] RefinedWeb：FineWeb 的重要前置工作**

Penedo, G., et al. (2023). *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*.

* https://arxiv.org/abs/2306.01116
