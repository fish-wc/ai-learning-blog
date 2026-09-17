# 从 JD 反推学习路线：面向腾讯犀牛鸟 / 大模型研究型岗位的「1 天、1 周、1 个月、半年」学习规划

> 本文更新时间：2026 年 9 月
> 适合人群：研一～研三，希望投递大模型算法、研究型算法、AI Infra、Agent、多模态等方向的同学。
>
> **说明**：背景中提到的是“快手人才计划”，但给出的岗位描述实际是 **腾讯犀牛鸟人才计划 / TEG**。因此本文严格按照给出的腾讯岗位信息与 2026 年公开的犀牛鸟相关资料进行规划。如果后续拿到快手真正的 JD，可以按照本文相同的方法重新做一次能力映射。

---

## 1. 在学习之前，先回答一个问题：这个岗位到底想要什么人？

拿到一个岗位 JD 后，我以前最容易犯的错误是：

> JD 里写“大模型”，那我就去学大模型；
> JD 里写“Transformer”，那我就去看 Transformer。

但真正有效的方法应该反过来：

> **岗位希望最终得到什么样的人？为了成为这种人，我分别需要补哪些能力？**

这次给出的岗位描述比较特殊，因为它并没有像普通算法岗一样罗列：

* 熟悉 Python/C++；
* 熟悉 PyTorch；
* 掌握 Transformer；
* 有大模型训练经验；
* 有论文者优先。

它实际上是一个偏 **科研人才培养** 的岗位入口。

从腾讯公开的 2026 年犀牛鸟精英人才计划申请指南来看，计划面向大三及以上本科生、硕士生和博士生，但申请者通常需要在论文、大型科研项目或国家级/学术组织竞赛奖励等方面至少具备一类代表性经历。培养方式也不是单纯“来公司实习写业务代码”，而是校企双导师、产业问题、科研探索、论文/专利和应用落地相结合。

所以，如果是硕士研究生，**学历阶段本身可以匹配犀牛鸟精英人才计划**。需要区分的是岗位描述中提到的“中国电子学会—腾讯博士生科研激励计划”，这一子计划明确面向在读博士研究生，不能与面向本科生、硕士生和博士生的精英人才计划混为一谈。([CIE China][1])

更重要的是，2026 年公开的腾讯犀牛鸟精英人才计划研究课题已经覆盖了非常完整的一张 AI 技术地图，包括：

1. 多模态理解与生成；
2. 语言与音频大模型；
3. 大模型数据、训练与推理优化；
4. Agentic AI 与 AI 智能体；
5. 生成式检索与推荐；
6. 大模型应用研究；
7. 基础平台技术与安全。

其中又进一步涉及 VLM、视频理解与生成、AI Coding、预训练、强化学习、Reasoning、Agent、GUI Agent、大模型推理系统、推荐系统、因果推断、数据库、AI 芯片等大量具体研究问题。

这意味着一件非常重要的事情：

> **准备这种岗位，绝对不是把所有 AI 技术都学一遍。**

真正合理的路线应该是一个 **T 型结构**：

```text
                         你的研究方向
                              │
                              │
                              │
                ┌─────────────┴─────────────┐
                │       一个方向深入        │
                │  LLM / Agent / Infra ... │
                │                           │
════════════════╪═══════════════════════════╪══════════════
 数学 → 深度学习 → PyTorch → Transformer → LLM → 科研方法
                   通用基础能力
```

也就是说：

> **底层知识要宽，研究方向要深。**

---

# 2. 我到底要掌握哪些知识？

如果把整个能力体系压缩成一张图，我认为至少应该包含下面七层。

| 层级          | 需要掌握的东西                                                 | 最终应该达到什么程度            |
| ----------- | ------------------------------------------------------- | --------------------- |
| 数学基础        | 线性代数、概率统计、微积分、优化                                        | 能看懂论文公式，而不是专门研究数学证明   |
| 深度学习        | MLP、反向传播、优化器、Normalization、Embedding                    | 能解释训练为什么有效、为什么失败      |
| PyTorch     | Tensor、Autograd、Module、Dataset、训练循环                     | 能独立写模型，而不是只会调 API     |
| Transformer | Attention、FFN、Residual、LayerNorm、RoPE、Mask              | 能从零写一个简化版 Transformer |
| LLM         | Tokenizer、预训练、SFT、LoRA、RLHF/DPO、RAG、Agent               | 理解完整生命周期              |
| AI Systems  | GPU、显存、混合精度、DDP/FSDP、TP/PP、KV Cache、Serving             | 理解模型为什么“跑得动、跑得快”      |
| Research    | Paper Reading、Baseline、Ablation、Evaluation、Reproduction | 能真正提出和验证一个研究问题        |

这里有一个非常关键的学习原则：

> **不要把“看懂”当成“学会”。**

我现在更愿意把掌握程度分成四级：

```text
Level 1：我听说过它
        ↓
Level 2：我能解释它
        ↓
Level 3：我能实现它
        ↓
Level 4：我能修改它，并设计实验验证自己的想法
```

真正的研究型岗位，希望看到的通常是 **Level 3～Level 4**。

---

# 3. 如果我只有一天：先建立整个大模型世界的地图

如果明天突然通知我：

> “后天腾讯面试。”

这时候千万不要尝试一天读十篇论文。

一天的目标只有一个：

> **把知识地图建立起来。**

你至少要知道每个技术处于整个系统的什么位置。

---

## 3.1 上午：搞懂 Transformer 到底是什么

先搞明白：

```text
文本
 ↓
Tokenizer
 ↓
Token ID
 ↓
Embedding
 ↓
Transformer × N
 ↓
Linear
 ↓
Softmax
 ↓
下一个 Token
```

重点学习：

* Tokenizer 是什么；
* Embedding 是什么；
* Query / Key / Value 是什么；
* Self-Attention 为什么能够建立 Token 之间的关系；
* 为什么 Attention 要除以 `sqrt(d_k)`；
* Multi-Head Attention 在做什么；
* FFN 的作用；
* Residual Connection 的作用；
* LayerNorm 的作用；
* Causal Mask 为什么能够保证自回归生成；
* Position Encoding / RoPE 为什么存在。

Transformer 的经典起点仍然是 *Attention Is All You Need*。它第一次系统提出了完全基于 Attention 的 Transformer 架构。([arXiv][2])

一天之内不需要把论文每一个公式全部推导出来。

至少做到：

> 给你一张 Transformer 图，你可以从输入一直讲到输出。

---

# 4. 下午：搞懂一个大模型是怎么“出生”的

建议建立下面这条流水线：

```text
                     大规模文本
                         │
                         ↓
                   Data Cleaning
                         │
                         ↓
                     Tokenizer
                         │
                         ↓
                    Pretraining
                         │
                         ↓
                    Base Model
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
             SFT                   RAG
              ↓
      Preference Alignment
        RLHF / DPO / RL
              ↓
          Chat / Reasoning
              ↓
             Agent
              ↓
          Deployment
       vLLM / Serving
```

只要把这张图真正搞清楚，就已经比“只知道怎么调用 ChatGPT API”前进了一大步。

---

## 4.1 Pretraining

预训练最简单的理解就是：

> 给模型前面的 Token，让模型预测下一个 Token。

例如：

```text
输入：
今 天 天 气

目标：
天 气 很 好
```

大量这样的训练最终让模型学会语言结构、知识和一定程度上的推理模式。

GPT-3 展示了随着模型规模扩大，大语言模型 Few-shot 能力可以明显增强；Scaling Laws 和后续 Chinchilla 工作则进一步研究了模型参数量、数据量和计算量之间的关系，尤其说明了“模型越大越好”并不是完整结论，数据和计算预算的配比同样重要。([arXiv][3])

---

## 4.2 SFT

预训练后的模型只是：

> “会续写。”

但我们希望它：

> “会回答问题。”

于是需要：

```text
Instruction
+
Response
```

这样的数据继续训练。

这就是：

> Supervised Fine-Tuning。

---

## 4.3 RLHF / DPO

SFT 解决的是：

> “怎么回答。”

Alignment 进一步解决：

> “什么回答更好。”

InstructGPT 是理解 RLHF 非常经典的工作，而 DPO 后来提出了一个更简洁的偏好优化方案，不需要传统 RLHF 中完整的 reward model + PPO 训练流程。([arXiv][4])

现阶段准备面试，不需要立刻手写 PPO。

但至少应该能够解释：

```text
Pretrain
   ↓
SFT
   ↓
Preference Data
   ↓
RLHF / DPO
```

每一步究竟想解决什么问题。

---

# 5. 晚上：理解 RAG、Agent 和 AI Infra

## 5.1 RAG

大模型的问题之一是：

```text
模型参数中的知识
        ↓
可能过时
可能不存在
可能记错
```

于是可以：

```text
User Query
    ↓
Retriever
    ↓
Documents
    ↓
Prompt + Documents
    ↓
LLM
    ↓
Answer
```

这就是最基本的 Retrieval-Augmented Generation。

RAG 的早期经典工作将参数化语言模型与外部非参数知识库结合起来，希望改善知识密集型任务中的知识访问和来源问题。([arXiv][5])

---

## 5.2 Agent

如果 RAG 是：

> **让模型查资料。**

那么 Agent 更进一步：

> **让模型做事情。**

最简单的 Agent Loop 可以理解成：

```text
Observe
  ↓
Think
  ↓
Choose Tool
  ↓
Execute
  ↓
Observe Result
  ↓
Think Again
```

ReAct 工作就是这一思路非常经典的代表之一：让语言模型将推理与行动交替进行。([arXiv][6])

真正学习 Agent 时，千万不要只学：

```python
agent = create_agent(...)
```

真正值得研究的是：

* Tool Use；
* Planning；
* Memory；
* Reflection；
* Multi-Agent；
* Agent Evaluation；
* Long-horizon Task；
* RL for Agent。

这也是 2026 年腾讯犀牛鸟公开研究方向中非常明显的一条主线，包括长期记忆、自演化 Agent、GUI Agent、视觉 Agent、多智能体等。

---

# 6. 如果只有一天，最后应该达到什么程度？

一天结束时，不要求你会训练一个 70B 模型。

但下面这些问题至少应该可以讲明白：

```text
Transformer 为什么需要 Attention？

Q、K、V 分别是什么意思？

为什么 Decoder 需要 Causal Mask？

LLM 是怎么预训练出来的？

Pretrain、SFT 和 DPO 有什么区别？

LoRA 为什么可以减少微调参数？

RAG 和 Fine-tuning 有什么区别？

Agent 和普通 LLM Application 有什么区别？

KV Cache 是干什么的？

FlashAttention 优化了什么？

DDP、TP、PP 大概有什么区别？

模型训练和模型推理分别有哪些性能瓶颈？
```

如果这些问题能够建立起概念联系，一天的任务就完成了。

---

# 7. 如果我有一个星期：从“听说过”进入“真正做过”

一周时间开始以后，学习策略就必须改变。

> **50% 时间学习，50% 时间写代码。**

不再允许连续七天只看视频。

建议按照下面的路线执行。

| Day   | 学习主题                    | 必须完成的实践                            |
| ----- | ----------------------- | ---------------------------------- |
| Day 1 | PyTorch + 深度学习          | 手写完整训练循环                           |
| Day 2 | Attention + Transformer | 手写 Self-Attention                  |
| Day 3 | Language Model          | 训练一个 Tiny GPT                      |
| Day 4 | SFT + LoRA + DPO        | LoRA 微调一个小模型                       |
| Day 5 | RAG + Agent             | 实现最小 RAG / Tool Agent              |
| Day 6 | AI Infra                | 学 DDP、KV Cache、FlashAttention、vLLM |
| Day 7 | Research                | 阅读论文 + 复现实验 + 写博客                  |

---

# 8. Day 1：先把 PyTorch 基础补扎实

一定要真正理解：

```python
model = Model()
optimizer = AdamW(...)
criterion = CrossEntropyLoss()

for batch in dataloader:
    optimizer.zero_grad()

    output = model(batch.x)

    loss = criterion(output, batch.y)

    loss.backward()

    optimizer.step()
```

不要觉得这个代码太简单。

后面几乎所有大模型训练框架，本质都在这个循环上不断扩展：

```text
普通训练
   ↓
Mixed Precision
   ↓
Gradient Accumulation
   ↓
Data Parallel
   ↓
Tensor Parallel
   ↓
Pipeline Parallel
   ↓
ZeRO / FSDP
   ↓
大规模集群训练
```

因此第一步不是学 DeepSpeed，而是先真正理解：

> 一个模型到底是怎么被训练起来的。

如果深度学习基础还不够稳定，《动手学深度学习》依然是非常适合系统补基础的材料，因为它把数学、代码和实验结合在一起。([Dive into Deep Learning][7])

---

# 9. Day 2：自己写一个 Transformer

不要直接：

```python
AutoModelForCausalLM.from_pretrained(...)
```

至少自己实现一次：

```text
Embedding
   ↓
MultiHeadAttention
   ↓
Add + Norm
   ↓
FFN
   ↓
Add + Norm
   ↓
Linear
```

最好做到：

```python
class MultiHeadAttention(nn.Module):
    ...

class FeedForward(nn.Module):
    ...

class TransformerBlock(nn.Module):
    ...

class GPT(nn.Module):
    ...
```

完成之后你会发现：

> 很多以前看起来特别神秘的大模型技术，本质都是在 Transformer Block 上做修改。

这个认知非常重要。

---

# 10. Day 3：训练自己的 Tiny GPT

这一天非常建议做一个小项目：

> **MiniGPT-from-Scratch**

不用追求参数规模。

10M、30M，甚至更小都可以。

重点是完整走一次：

```text
Data
 ↓
Tokenizer
 ↓
Dataset
 ↓
Transformer
 ↓
Training
 ↓
Loss Curve
 ↓
Checkpoint
 ↓
Sampling
```

项目 README 里至少记录：

* 数据集；
* 参数量；
* Hidden Size；
* Head 数；
* Layer 数；
* Learning Rate；
* Batch Size；
* Token 数；
* 最终 Loss；
* 生成案例。

做到这里，你已经从：

> “我学过 Transformer”

变成：

> “我自己训练过 Transformer。”

对求职来说，这两句话的含金量完全不同。

---

# 11. Day 4：SFT + LoRA + DPO

这一天开始接触真正的大模型训练工具链。

建议学习：

```text
Full Fine-tuning
        │
        ├── LoRA
        │
        └── QLoRA
```

重点理解 LoRA 的思想：

> 不直接更新完整的大权重矩阵，而是在更新量中引入低秩矩阵，只训练少量新增参数。

LoRA 论文展示了这种低秩适配方式可以大幅降低需要训练的参数量和显存开销，因此非常适合学生阶段进行微调实验。([arXiv][8])

这一阶段推荐开始真正使用：

```text
PyTorch
Hugging Face Transformers
Datasets
PEFT
Accelerate
```

Hugging Face 的 Transformers 项目已经成为现代开源模型训练和推理生态的重要基础设施之一，其官方课程也覆盖 Transformers、Datasets、Tokenizers、Accelerate 等常用组件。([GitHub][9])

---

# 12. Day 5：做一个“真正能评测”的 RAG

不要做：

```text
PDF
 ↓
Vector DB
 ↓
LLM
 ↓
Demo
```

然后宣布：

> “我做完 RAG 了。”

真正值得学习的是：

```text
Query
 ↓
Retriever
 ↓
Recall@K
 ↓
Reranker
 ↓
Context
 ↓
Generator
 ↓
Answer
 ↓
Evaluation
```

重点研究三个问题：

### Question 1

Retriever 有没有检索到正确文档？

### Question 2

正确文档已经给模型了，模型有没有正确使用？

### Question 3

最终答案如何评价？

这样你的项目就从：

> “Demo 项目”

开始接近：

> “Research Project”。

---

# 13. Day 6：第一次认真学习 AI Infra

如果目标是大模型方向，我非常建议至少理解一些 AI Infra。

因为模型最终逃不开三个东西：

```text
算力
显存
通信
```

首先学习：

```text
Data Parallel
Tensor Parallel
Pipeline Parallel
FSDP / ZeRO
Mixed Precision
Gradient Checkpointing
```

PyTorch 官方的 Distributed 文档将 DDP、FSDP、Tensor Parallel、Pipeline Parallel 等都纳入了当前的大规模训练工具体系。([PyTorch Documentation][10])

Megatron-LM 则是理解大模型 Tensor Parallel、Pipeline Parallel 等训练技术的重要工程项目；ZeRO 的核心目标之一则是通过切分训练状态减少数据并行过程中的显存冗余。([arXiv][11])

推理侧至少学习：

```text
KV Cache
Continuous Batching
PagedAttention
Quantization
Speculative Decoding
```

vLLM 的 PagedAttention 借鉴虚拟内存分页的思想管理动态 KV Cache，重点改善推理服务中的显存利用率和吞吐问题。([arXiv][12])

还有一个面试中很值得掌握的知识：

> **FlashAttention 并不是“近似 Attention”。**

它仍然计算精确 Attention，核心优化点是减少 GPU HBM 与片上 SRAM 之间的数据搬运，使 Attention 更加 IO-aware。([arXiv][13])

---

# 14. Day 7：第一次进行“研究训练”

找一篇论文。

不要只是：

> 阅读 → 做笔记。

应该：

```text
论文提出什么问题？
        ↓
以前的方法为什么不够？
        ↓
作者提出什么方法？
        ↓
为什么理论上可能有效？
        ↓
实验怎么设计？
        ↓
用了哪些 Baseline？
        ↓
有没有 Ablation？
        ↓
结果真的支持作者结论吗？
        ↓
还有什么问题没解决？
```

然后尝试：

> 改一个变量。

例如：

```text
LoRA Rank = 8
        ↓
LoRA Rank = 16
        ↓
LoRA Rank = 32
```

观察：

```text
Train Loss
Validation Loss
GPU Memory
Training Time
Evaluation Score
```

这时候，你已经开始从：

> “程序员思维”

向：

> “研究者思维”

转换。

---

# 15. 如果有一个月：不要继续横向扩张，而要开始建立深度

一个月以后最大的风险是：

```text
Transformer 学一点
RAG 学一点
Agent 学一点
Diffusion 学一点
vLLM 学一点
CUDA 学一点
推荐系统学一点
```

最后：

> 什么都知道，但什么都讲不深。

因此，一个月的规划应该变成：

```text
Week 1：夯实基础
Week 2：完成 LLM 全流程
Week 3：选择一个研究方向
Week 4：做一次真正的 Reproduction
```

---

# 16. 第一周：Transformer From Scratch

目标：

> 不依赖高级 Transformer API，实现一个最小语言模型。

知识：

```text
Tokenizer
Embedding
Self-Attention
Multi-Head Attention
RoPE
LayerNorm
FFN
Residual
Causal Mask
Cross Entropy
AdamW
Warmup
Gradient Clipping
```

产出：

```text
01-transformer-from-scratch.md
```

以及：

```text
mini-transformer/
├── tokenizer.py
├── attention.py
├── transformer.py
├── train.py
├── generate.py
└── README.md
```

---

# 17. 第二周：完整走一次 LLM Pipeline

学习：

```text
Pretraining
 ↓
SFT
 ↓
LoRA
 ↓
Preference Learning
 ↓
DPO
 ↓
Evaluation
 ↓
Inference
```

这里开始学习：

* Data Collator；
* Instruction Dataset；
* Packing；
* Gradient Accumulation；
* BF16 / FP16；
* Checkpoint；
* PEFT；
* Generation Config；
* Evaluation。

建议用一个小型开源模型跑完整流程，而不是挑战超大模型。

---

# 18. 第三周：必须选方向

这是整条学习路线中最关键的一步。

从腾讯 2026 年公开研究课题来看，其研究覆盖已经非常广，因此最合理的方法不是“全部掌握”，而是根据自己的科研经历选择一个主方向。

如果目前没有特别明确的方向，可以考虑下面几条路线。

| 方向                        | 核心知识                                     | 推荐项目               |
| ------------------------- | ---------------------------------------- | ------------------ |
| LLM Training              | Pretrain、SFT、DPO、RL、Data                 | 小模型完整训练            |
| Agent / RAG               | Retrieval、Tool、Memory、Planning、RL        | Research Agent     |
| AI Infra                  | DDP、FSDP、TP、PP、Kernel、Serving            | LLM Benchmark      |
| Multimodal                | VLM、Vision Encoder、Cross Modal Alignment | Document VQA       |
| Data / Evaluation         | Data Cleaning、Synthetic Data、Benchmark   | LLM Eval Framework |
| Generative Recommendation | Recommendation + Transformer + LLM       | Generative RecSys  |

如果没有明显历史包袱，我会采用：

```text
LLM 基础
   +
一个差异化方向
```

例如：

```text
LLM + Agent
```

或者：

```text
LLM + AI Infra
```

这样既不会丢掉通用基础，又能逐渐建立自己的技术标签。

---

# 19. 如果选择 LLM Training，需要继续学什么？

路线大概是：

```text
Tokenizer
   ↓
Data Pipeline
   ↓
Transformer Architecture
   ↓
Pretraining
   ↓
Scaling Law
   ↓
Distributed Training
   ↓
SFT
   ↓
Preference Learning
   ↓
RL / Reasoning
```

重点论文路线可以按照：

```text
Transformer
   ↓
GPT-3
   ↓
Scaling Laws
   ↓
Chinchilla
   ↓
InstructGPT
   ↓
LoRA
   ↓
DPO
```

不要追求：

> “一天读完一篇。”

而应该问：

> 这篇论文解决了上一代方法的什么问题？

最终建立论文之间的关系。

---

# 20. 如果选择 Agent，需要继续学什么？

不要把学习路线写成：

```text
LangChain
→ LangGraph
→ AutoGen
```

这些是框架。

真正的问题是：

```text
LLM
 ↓
Reasoning
 ↓
Planning
 ↓
Tool Use
 ↓
Memory
 ↓
Environment
 ↓
Feedback
 ↓
Learning
```

深入以后继续研究：

* Tool Learning；
* Function Calling；
* Long-term Memory；
* Planning；
* Multi-Agent；
* GUI Agent；
* Web Agent；
* Agent RL；
* Agent Evaluation。

腾讯 2026 年的研究课题中已经直接出现视觉 Agent、GUI Agent、自演化 Agent、多智能体和长期记忆等方向，因此 Agent 已经不是单纯的应用层 Demo，而是明确的研究主题。

---

# 21. 如果选择 AI Infra，需要继续学什么？

推荐从一张显存账单开始。

训练一个参数量为 `N` 的模型：

```text
Parameters
Gradients
Optimizer States
Activations
```

分别需要多少显存？

然后一步一步理解：

```text
为什么需要 FP16 / BF16？
        ↓
为什么需要 Gradient Checkpointing？
        ↓
为什么需要 DDP？
        ↓
为什么需要 ZeRO / FSDP？
        ↓
为什么需要 Tensor Parallel？
        ↓
为什么需要 Pipeline Parallel？
```

再进入推理：

```text
Prefill
Decode
KV Cache
Batching
PagedAttention
Quantization
Speculative Decoding
```

最后再去看：

```text
PyTorch Distributed
Megatron-LM
DeepSpeed
vLLM
```

这个顺序比一上来读几万行框架源码有效得多。

---

# 22. 第四周：完成第一次论文复现

一个月结束时，我认为最重要的成果不是：

> “看完了 30 篇论文。”

而是：

> **真正复现过一篇。**

完整流程应该包括：

```text
Read Paper
    ↓
Understand Method
    ↓
Build Baseline
    ↓
Reproduce
    ↓
Compare Results
    ↓
Ablation
    ↓
Failure Analysis
    ↓
Write Report
```

最终形成一篇博客：

```text
《从零复现 XXX：论文思想、代码实现、实验结果与踩坑记录》
```

这种文章的价值远高于：

```text
《XXX 论文阅读笔记》
```

因为前者能够真正证明：

> 你做过。

---

# 23. 三个月应该做到什么？

如果坚持三个月，我会把目标设成：

```text
Month 1
建立完整知识体系

        ↓

Month 2
复现 2～3 篇经典论文

        ↓

Month 3
在已有工作上提出自己的小改进
```

第三个月开始做：

```text
Baseline
+
My Method
+
Ablation
+
Evaluation
```

哪怕最终结果是：

> “我的方法没有效果。”

这仍然是一次非常好的研究训练。

因为研究真正训练的是：

```text
提出问题
↓
提出假设
↓
设计实验
↓
分析结果
↓
修正假设
```

而不是：

> 每一次实验都必须 SOTA。

---

# 24. 六个月应该做到什么？

到了半年，学习目标应该发生明显变化。

不要再以：

> “我学了多少东西”

作为衡量指标。

开始以：

> **“我产出了什么？”**

衡量自己。

理想状态下应该逐渐形成：

```text
一个主研究方向

+

一个有质量的科研项目

+

一个可以公开的 GitHub Repository

+

一系列技术博客

+

至少一次完整论文复现

+

至少一次自己的实验创新

+

开源贡献 / Issue / PR
```

如果能够参与实验室论文，则进一步尝试：

```text
Research Question
       ↓
Experiment
       ↓
Paper
```

这也更加贴近犀牛鸟计划真正强调的“科研探索 + 产业真实问题 + 应用落地”的培养方式。

---

# 25. 我会给自己安排这样一个 12 周路线

| 时间      | 目标                          |
| ------- | --------------------------- |
| Week 1  | PyTorch + 深度学习              |
| Week 2  | Transformer From Scratch    |
| Week 3  | Tiny GPT                    |
| Week 4  | SFT + LoRA + DPO            |
| Week 5  | RAG / Agent / Infra 三选一深入   |
| Week 6  | 阅读该领域 5～8 篇经典论文             |
| Week 7  | 选择一篇论文复现                    |
| Week 8  | 完成 Baseline                 |
| Week 9  | 完成 Reproduction             |
| Week 10 | Ablation + Failure Analysis |
| Week 11 | 尝试一个自己的改进                   |
| Week 12 | 整理 GitHub + 博客 + 面试表达       |

三个月之后，简历里就不应该只剩：

```text
熟悉 Transformer
熟悉 PyTorch
了解大语言模型
```

而应该出现：

```text
实现了什么
训练了什么
复现了什么
优化了什么
实验结果如何
为什么这么设计
```

---

# 26. GitHub 博客应该怎么和学习结合？

如果目标是求职，我不建议把博客做成：

```text
今天学了 Softmax
明天学了 Attention
后天学了 Transformer
```

这样的流水账。

博客真正应该做的是：

> **公开自己的知识体系与技术能力。**

推荐仓库结构：

```text
blog/
│
├── fundamentals/
│   ├── autograd.md
│   ├── optimizer.md
│   └── normalization.md
│
├── transformer/
│   ├── attention-from-scratch.md
│   ├── rope.md
│   └── kv-cache.md
│
├── llm/
│   ├── pretraining.md
│   ├── lora.md
│   ├── dpo.md
│   └── scaling-law.md
│
├── agent/
│   ├── rag.md
│   ├── react.md
│   └── agent-memory.md
│
├── infra/
│   ├── ddp.md
│   ├── zero.md
│   ├── flash-attention.md
│   └── vllm.md
│
├── paper-reading/
│
├── reproductions/
│
└── projects/
```

随着时间推移，这个仓库本身就会变成你的：

> **公开技术履历。**

---

# 27. 每篇博客都使用同一个模板

以后我会建议自己所有技术博客都尽可能回答下面几个问题。

## 27.1 What：它是什么？

一句话解释概念。

---

## 27.2 Why：为什么需要它？

以前的方法有什么问题？

---

## 27.3 How：它怎么解决？

讲思想。

---

## 27.4 Math：数学上是什么？

把核心公式讲清楚。

---

## 27.5 Code：代码怎么实现？

提供最小实现。

---

## 27.6 Experiment：效果怎么样？

给实验。

---

## 27.7 Failure：什么时候不好用？

讨论限制。

---

## 27.8 Interview：面试会怎么问？

整理问题。

---

## 27.9 Reference

论文、官方文档、代码仓库全部列出来。

这样写出来的博客，不只是给别人看的。

半年以后自己回来重新阅读，也能重新把知识捡起来。

---

# 28. 推荐优先完成的博客系列

如果让我从现在开始建立整个博客，我会按照下面的顺序：

```text
01｜深度学习训练到底发生了什么？
02｜反向传播与 Autograd 到底是什么？
03｜从矩阵运算理解 Self-Attention
04｜从零实现 Multi-Head Attention
05｜从零实现 Transformer
06｜从零训练一个 Tiny GPT
07｜Tokenizer 到底在做什么？
08｜大模型 Pretraining 到底在训练什么？
09｜LoRA 为什么只训练很少参数也有效？
10｜SFT、RLHF 和 DPO 到底有什么区别？
11｜RAG 到底解决了什么问题？
12｜Agent 到底比普通 LLM 多了什么？
13｜KV Cache 为什么能加速推理？
14｜FlashAttention 到底优化了什么？
15｜DDP、FSDP、TP、PP 到底有什么区别？
16｜vLLM 为什么能够提高推理吞吐？
17｜第一次完整复现一篇 LLM 论文
```

如果这 17 篇真的都能自己理解、实现并写清楚，大模型基础已经会发生非常明显的变化。

---

# 29. 学到什么程度才算“真正掌握”？

以后每学习一个知识点，可以问自己五个问题。

例如：

> LoRA。

### 第一关：能不能不用术语解释？

为什么需要 LoRA？

---

### 第二关：能不能画图？

LoRA 到底插在哪里？

---

### 第三关：能不能解释公式？

为什么：

[
W' = W + BA
]

其中：

[
B \in \mathbb{R}^{d\times r}
]

[
A \in \mathbb{R}^{r\times k}
]

可以降低训练参数量？

---

### 第四关：能不能实现？

不用 PEFT，自己实现一个：

```python
class LoRALinear(nn.Module):
    ...
```

---

### 第五关：能不能设计实验？

比较：

```text
Full Fine-tuning
LoRA rank=8
LoRA rank=16
LoRA rank=32
```

分析：

```text
Accuracy
GPU Memory
Trainable Parameters
Training Time
```

当五关全部通过的时候：

> 才算真的学过。

---

# 30. 面试前必须能够回答的核心问题

如果投递的是大模型相关科研/算法岗位，我会至少检查自己能否完整回答下面这些问题：

1. Self-Attention 中为什么要除以 `sqrt(d_k)`？
2. Multi-Head Attention 为什么需要多个 Head？
3. Decoder 为什么需要 Causal Mask？
4. RoPE 如何编码相对位置信息？
5. Pre-LN 和 Post-LN 有什么区别？
6. Adam / AdamW 的核心思想是什么？
7. Pretraining、SFT、RLHF 和 DPO 分别解决什么问题？
8. LoRA 为什么有效？
9. RAG 和 Fine-tuning 应该分别在什么时候使用？
10. Agent 的 Tool Use、Memory 和 Planning 分别解决什么问题？
11. DDP、FSDP、TP 和 PP 分别切分了什么？
12. KV Cache 为什么会占用大量显存？
13. FlashAttention 为什么可以加速 Attention？
14. PagedAttention 为什么能够提升大模型 Serving 效率？
15. 如果一个实验比 Baseline 提高 1%，怎么证明提升来自你的方法而不是随机波动？
16. Ablation Study 应该怎么设计？
17. 一个新方法应该选择哪些 Baseline？
18. 如果实验结果和论文不同，你会如何定位问题？

最后几个问题尤其重要。

因为研究岗真正想判断的，往往已经不是：

> 你记住了多少名词？

而是：

> **你到底会不会做实验、分析问题和进行研究。**

---

# 31. 学习过程中最应该避免的几个坑

### 坑一：疯狂收藏课程

```text
收藏 ≠ 学习
```

看 20 门课程不如真正完成一个项目。

---

### 坑二：一直追最新论文

每天：

```text
今天 DeepSeek
明天 Agent
后天 World Model
大后天 Diffusion
```

最后基础反而没补起来。

前期应该优先建立：

> **经典知识的稳定结构。**

---

### 坑三：一上来训练超大模型

学生阶段最重要的是理解机制。

一个：

```text
30M Transformer
```

如果完全是自己写、自己训练、自己分析，

往往比：

```text
调用框架 LoRA 一个 70B 模型
```

更有学习价值。

---

### 坑四：Agent 只会调框架

真正研究：

```text
Planning
Memory
Tool Learning
Evaluation
RL
```

而不是 API。

---

### 坑五：只有代码，没有实验

科研项目 README 不能只有：

```bash
python main.py
```

至少应该有：

```text
Research Question
Baseline
Method
Dataset
Metric
Experiment
Ablation
Conclusion
```

---

# 32. 最终，我希望半年以后自己变成什么样？

不是：

> “我学过大模型。”

而是能够非常清楚地介绍自己：

```text
我掌握 Transformer 和大语言模型的基础原理；

我自己实现并训练过一个 Transformer Language Model；

我完成过 SFT / LoRA / DPO 等训练实验；

我深入研究的是 XXX 方向；

我复现过 XXX 工作；

我针对 XXX 问题做过 XXX 实验；

我的代码、实验记录和技术博客全部公开在 GitHub；
```

这时候：

> GitHub 不再只是一个代码仓库。

它开始变成：

```text
学习记录
+
技术能力
+
科研能力
+
个人作品集
+
公开技术履历
```

这才是我认为“边学习、边做博客”真正有价值的地方。

---

# 33. 一句话总结整条学习路线

如果只有一天：

> **建立地图。**

如果只有一周：

> **跑通代码。**

如果有一个月：

> **选择方向。**

如果有三个月：

> **复现论文。**

如果有半年：

> **形成自己的研究问题与公开成果。**

最终不要让自己停留在：

```text
I know it.
```

而应该一步一步走到：

```text
I understand it.
        ↓
I can implement it.
        ↓
I can reproduce it.
        ↓
I can improve it.
        ↓
I can explain why.
```

对于研究型 AI 岗位来说，这五步，大概就是从：

> **“学习 AI”**

逐渐走向：

> **“真正开始做 AI 研究”。**

---

# 参考文献与学习资料

## 1. 腾讯犀牛鸟相关资料

1. **腾讯高校合作：开放申请｜2026 腾讯犀牛鸟精英人才计划**。2026 年 1 月。([University Relations, Tencent][14])
2. **《2026 年度腾讯犀牛鸟精英人才计划申请指南》**。其中包括申请对象、成果要求、联合培养模式等。
3. **《2026 年度腾讯犀牛鸟精英人才计划研究课题及导师介绍》**。研究方向覆盖多模态、语言模型、模型训练与推理、Agent、生成式搜索推荐、大模型应用及基础平台等方向。
4. **中国电子学会：2026 年度“中国电子学会—腾讯博士生科研激励计划（混元大模型专项）”申报征集通知**。重点方向包括大语言模型与预训练、多模态、Agent/数字人、AI Infra 与训推优化、可信数据构建与评测。([CIE China][1])
5. **Tencent Hunyuan 官方开源组织**，可用于持续跟踪腾讯混元的模型、代码与研究工作。GitHub：`Tencent-Hunyuan`。([GitHub][15])

## 2. Transformer 与大语言模型基础

6. Vaswani et al. **Attention Is All You Need**. NeurIPS 2017. arXiv:1706.03762. ([arXiv][2])
7. Brown et al. **Language Models are Few-Shot Learners**. NeurIPS 2020. arXiv:2005.14165. ([arXiv][16])
8. Kaplan et al. **Scaling Laws for Neural Language Models**. arXiv:2001.08361. ([arXiv][3])
9. Hoffmann et al. **Training Compute-Optimal Large Language Models**. arXiv:2203.15556. ([arXiv][17])
10. Kingma & Ba. **Adam: A Method for Stochastic Optimization**. arXiv:1412.6980. ([arXiv][18])

## 3. Fine-tuning 与 Alignment

11. Ouyang et al. **Training Language Models to Follow Instructions with Human Feedback**. arXiv:2203.02155. ([arXiv][4])
12. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**. arXiv:2106.09685. ([arXiv][8])
13. Rafailov et al. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. arXiv:2305.18290. ([arXiv][19])

## 4. RAG 与 Agent

14. Lewis et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. arXiv:2005.11401. ([arXiv][5])
15. Yao et al. **ReAct: Synergizing Reasoning and Acting in Language Models**. arXiv:2210.03629. ([arXiv][6])

## 5. 大模型训练与推理系统

16. Shoeybi et al. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. arXiv:1909.08053. ([arXiv][11])
17. Rajbhandari et al. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. arXiv:1910.02054. ([arXiv][20])
18. Dao et al. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. arXiv:2205.14135. ([arXiv][13])
19. Kwon et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention**. SOSP 2023 / arXiv:2309.06180. ([arXiv][12])
20. **vLLM**，高吞吐 LLM 推理与 Serving 开源项目。GitHub：`vllm-project/vllm`。([GitHub][21])
21. **NVIDIA Megatron-LM / Megatron Core**，用于学习大规模 Transformer 训练以及 TP、PP、DP、EP 等并行技术。GitHub：`NVIDIA/Megatron-LM`。([GitHub][22])
22. **PyTorch Distributed Overview**，PyTorch 官方分布式训练文档，包括 DDP、FSDP、TP、PP 等内容。([PyTorch Documentation][10])

## 6. 系统性课程与开源学习资料

23. Stanford **CS336: Language Modeling from Scratch**. 课程从数据处理、Transformer 实现、训练一直覆盖到 Evaluation，非常适合作为深入学习大模型的课程主线。([Stanford CS336][23])
24. Zhang, Lipton, Li, Smola. **Dive into Deep Learning / 动手学深度学习**。非常适合作为深度学习基础补全资料。([Dive into Deep Learning][7])
25. **Hugging Face Course**。覆盖 Transformers、Datasets、Tokenizers、Accelerate 等现代大模型工具链。([Hugging Face][24])
26. **Hugging Face Transformers**。GitHub：`huggingface/transformers`，适合从模型使用逐步进入源码阅读。([GitHub][9])

[1]: https://www.cie.org.cn/list_43/16226.html?utm_source=chatgpt.com "2026年度“中国电子学会-腾讯博士生科研激励计划（混元大模型专项）”申报征集通知-中国电子学会"
[2]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"
[3]: https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com "Scaling Laws for Neural Language Models"
[4]: https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with human feedback"
[5]: https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[6]: https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com "ReAct: Synergizing Reasoning and Acting in Language Models"
[7]: https://d2l.ai/?utm_source=chatgpt.com "Dive into Deep Learning — Dive into Deep Learning 1.0.3 documentation"
[8]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
[9]: https://github.com/huggingface/transformers?utm_source=chatgpt.com "GitHub - huggingface/transformers: 🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text, vision, audio, and multimodal models, for both inference and training. · GitHub"
[10]: https://docs.pytorch.org/tutorials/beginner/dist_overview.html?utm_source=chatgpt.com "PyTorch Distributed Overview — PyTorch Tutorials 2.14.0+cu130 documentation"
[11]: https://arxiv.org/abs/1909.08053?utm_source=chatgpt.com "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
[12]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[13]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[14]: https://ur.tencent.com/article/1516?utm_source=chatgpt.com "开放申请 | 2026腾讯犀牛鸟精英人才计划 - 腾讯高校合作"
[15]: https://github.com/tencent-hunyuan?utm_source=chatgpt.com "Tencent-Hunyuan · GitHub"
[16]: https://arxiv.org/abs/2005.14165?utm_source=chatgpt.com "Language Models are Few-Shot Learners"
[17]: https://arxiv.org/abs/2203.15556?utm_source=chatgpt.com "Training Compute-Optimal Large Language Models"
[18]: https://arxiv.org/abs/1412.6980?utm_source=chatgpt.com "Adam: A Method for Stochastic Optimization"
[19]: https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[20]: https://arxiv.org/abs/1910.02054?utm_source=chatgpt.com "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
[21]: https://github.com/vllm-project/vllm?utm_source=chatgpt.com "GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs · GitHub"
[22]: https://github.com/nvidia/megatron-lm?utm_source=chatgpt.com "GitHub - NVIDIA/Megatron-LM: Ongoing research training transformer models at scale · GitHub"
[23]: https://cs336.stanford.edu/spring2025/?utm_source=chatgpt.com "Stanford CS336 | Language Modeling from Scratch (2025)"
[24]: https://huggingface.co/docs/course/chapter1/1?utm_source=chatgpt.com "Introduction · Hugging Face"
