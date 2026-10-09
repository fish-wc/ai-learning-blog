---
title: "Agent 与 LLM Workflow：架构边界、Agent Loop 及工具使用范式"
date: 2026-10-09
categories:
  - Agent
tags:
  - ReAct
---

# ReAct 论文精读：语言推理与工具行动的协同机制

> **论文**：ReAct: Synergizing Reasoning and Acting in Language Models
> **会议**：ICLR 2023
> **研究主题**：LLM Agent、Reasoning、Tool Use、Sequential Decision Making
> **阅读范围**：Introduction、Section 2（Method）、Figure 1（交互轨迹），并结合实验结果分析其有效性与局限。

## 1. 核心问题：为什么需要耦合 Reasoning 与 Acting？

ReAct [1] 的核心研究问题是：**如何使语言模型在序列决策过程中，将内部推理与外部环境交互统一起来？**

此前，两类方法分别强调不同的能力：

* **Chain-of-Thought（CoT）**：通过中间推理步骤完成复杂任务，但在没有外部信息获取机制时，推理依赖模型已有知识，错误事实可能沿推理链传播。
* **Act-only**：根据历史交互生成环境动作，能够利用外部反馈，但缺少显式的语言推理步骤来组织任务目标、抽取关键信息与整合多步证据。

ReAct 将两者结合，形成双向的信息流：

**Reasoning to Act**：推理用于分解目标、制定计划、选择动作以及处理异常。

**Acting to Reason**：行动用于访问外部环境，获取新信息，并使后续推理建立在观察结果之上。

因此，ReAct 的贡献并非简单地让 LLM 具备工具调用能力，而是建立了一种**语言推理与环境反馈交替驱动的决策范式**。

## 2. Method：ReAct 的形式化建模

### 2.1 扩展动作空间

论文首先将智能体与环境的交互表示为序列决策过程。

在时刻 \(t\)，智能体接收环境观察 \(o_t \in \mathcal O\)，根据上下文 \(c_t\) 选择动作 \(a_t \in \mathcal A\)：

$$
a_t \sim \pi(a_t \mid c_t)
$$

其中：

$$
c_t=(o_1,a_1,\ldots,o_{t-1},a_{t-1},o_t)
$$

传统 Act-only 方法直接学习或生成从历史上下文到环境动作的映射。当任务涉及复杂的信息整合、长期规划或异常恢复时，这种映射可能难以直接建立。

ReAct 的关键改动是扩展动作空间：

$$
\boxed{\hat{\mathcal A}=\mathcal A\cup\mathcal L}
$$

其中：

* \(\mathcal A\)：能够作用于外部环境的动作空间。
* \(\mathcal L\)：自然语言空间，用于生成 Thought（推理轨迹）。
* \(\hat{\mathcal A}\)：同时包含环境动作与语言推理的扩展动作空间。

扩展后，模型既可以执行语言层面的推理，也可以发起环境动作。

### 2.2 Thought 与 Action 的状态转移差异

两类动作对环境及上下文的影响不同。

**Thought：语言推理**

当 \(\hat a_t\in\mathcal L\) 时，模型生成语言推理文本，不改变外部环境，也不会产生新的环境观察：

$$
c_{t+1}=(c_t,\hat a_t)
$$

Thought 的作用是将任务分解、事实提取、计划调整等中间信息显式写入上下文，以支持后续决策。

**Action：环境交互**

当 \(\hat a_t\in\mathcal A\) 时，模型选择环境动作，由外部执行器执行，并接收环境反馈。

为便于理解，可将这一过程抽象为：

$$
o_{t+1}=\operatorname{Env}(\hat a_t)
$$

$$
c_{t+1}=(c_t,\hat a_t,o_{t+1})
$$

这里的关键区别是：

**Thought 更新模型的上下文；Action 通过外部执行器作用于环境，并将 Observation 引入上下文。**

上述 Action 转移公式是对论文交互过程的简化表示，并非原文逐字给出的方程。

由此，ReAct 形成如下闭环：

```text
Task
  │
  ▼
Thought ──► 决策 / 计划 / 信息整合
  │
  ▼
Action ───► 工具调用或环境操作
  │
  ▼
Observation ──► 获取环境反馈
  │
  ├── 信息不足：更新上下文，继续决策
  │
  └── 任务完成：Finish
```

需要强调：这是一种决策交互范式，而不是必须固定执行的三步协议。

### 2.3 Thought 的生成频率并非固定

原论文根据任务特点采用不同的 Thought 分布策略：

* **知识密集型任务**（HotpotQA、FEVER）：使用较密集的 Thought–Action–Observation 交替结构，以支持多跳推理与证据整合。
* **交互式决策任务**（ALFWorld、WebShop）：采用较稀疏的 Thought，在目标变化、关键决策或异常出现时进行语言推理，避免为每一次环境操作都生成冗余文本。

这一设计说明，ReAct 并不依赖固定的思考频率，而是允许推理粒度随任务需求变化。

### 2.4 ReAct 是否依赖模型训练？

ReAct 的主要实验使用冻结的 PaLM-540B，通过 Few-shot In-context Learning，引导模型生成推理和行动交错的轨迹。

在知识问答任务中，作者为 HotpotQA 和 FEVER 分别构造了 6 条与 3 条少样本示例。

因此，**经典 ReAct 首先是一种推理时的提示与交互组织方法，而非必须修改模型参数的训练算法。**

不过，论文也进一步研究了使用生成轨迹微调较小模型的效果，不能将 ReAct 简单等同于“完全不涉及训练”。

## 3. 真实轨迹分析：从工具反馈到动态决策

选取原论文 Figure 1(1d) 的 HotpotQA 案例 [1]。

### 3.1 任务定义

原始问题：

> Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?

即：除了 Apple Remote，还有什么设备或控制方式可以操作它最初设计用于控制的程序？

该问题需要完成两跳信息检索：

$$
\text{Apple Remote}
\xrightarrow{\text{原始控制对象}}
\text{Front Row}
$$

$$
\text{Front Row}
\xrightarrow{\text{其他控制方式}}
\text{Keyboard Function Keys}
$$

论文设计了三个可用操作：

| Action           | 功能                                    |
| ---------------- | ------------------------------------- |
| `Search[entity]` | 搜索 Wikipedia 实体；存在时返回页面前五句，否则返回相似实体候选 |
| `Lookup[string]` | 在当前页面中查找包含指定字符串的后续句子                  |
| `Finish[answer]` | 提交答案并终止任务                             |

这里的工具接口是研究中的简化 Wikipedia API，并非通用搜索引擎。

### 3.2 逐步拆解交互轨迹

以下依据 Figure 1(1d) 整理。Thought 使用语义概括，Action 保留原论文中的调用形式。

| 步骤 | Thought（决策）               | Action（调用）                       | Observation（反馈）                              |
| -- | ------------------------- | -------------------------------- | -------------------------------------------- |
| 1  | 确定 Apple Remote 最初控制的程序   | `Search[Apple Remote]`           | Apple Remote 最初用于控制 Front Row media center   |
| 2  | 已知目标是 Front Row，继续检索其控制方式 | `Search[Front Row]`              | 未找到精确实体，但返回包含 `Front Row (software)` 的相似页面列表 |
| 3  | 根据候选实体修正查询，消除名称歧义         | `Search[Front Row (software)]`   | 返回 Front Row 软件页面，论文展示的反馈内容经过截断              |
| 4  | 综合已获得的信息，确定最终答案           | `Finish[keyboard function keys]` | 终止任务，无后续检索反馈                                 |

这条轨迹包含三次搜索调用、对应的三次环境观察，以及最后一次任务终止动作。

### 3.3 关键分析：Observation 如何改变决策？

第二次检索：

```text
Action: Search[Front Row]

Observation:
Could not find [Front Row].
Similar: ...
[Front Row (software)], ...
```

该反馈包含两类信息：

1. **否定信息**：当前检索词未能定位目标条目。
2. **候选信息**：工具提供了更准确的实体名称。

模型随后执行：

```text
Thought:
Front Row is not found.
I need to search Front Row (software).

Action:
Search[Front Row (software)]
```

这里体现了一个关键决策过程：

$$
\text{检索未命中}
\rightarrow
\text{候选实体识别}
\rightarrow
\text{查询修正}
\rightarrow
\text{再次检索}
$$

需要区分的是，`Front Row` 未命中属于**检索层面的实体匹配失败**，并不代表工具执行异常。

更重要的是，后续动作受到 Observation 的影响，而不只是执行预先确定的查询列表。

### 3.4 与 Act-only 的严格对照

仅凭上述查询修正过程，还不足以证明 ReAct 的独特性。

原论文 Figure 1(1c) 中，Act-only 实际上也执行了相同的三次搜索：

```text
Search[Apple Remote]
Search[Front Row]
Search[Front Row (software)]
Finish[yes]
```

尽管 Act-only 也能够利用反馈修正搜索词，但最终错误地提交了 `yes`。

相对而言，ReAct 在第三次观察后生成了显式的信息整合步骤，并提交：

```text
Finish[keyboard function keys]
```

**这表明 ReAct 的价值不只是查询修正，而是在多步交互中引入显式语言推理，辅助事实关联、目标跟踪和答案合成。**

Act-only 具有一定的反馈适应能力；ReAct 并不独占这种能力。两者的核心差异在于是否显式利用语言推理组织环境信息与决策过程。

同时还应注意：Figure 1 中第三次 Observation 的文本被截断，图中并未完整展示支持最终答案的原始证据。因此，这条轨迹能够说明模型的决策方式，但不能仅凭被截断的文本独立核实最终答案的事实依据。

这也揭示了一个方法论问题：

**可读的推理轨迹不等于经过验证的证据链；模型生成的 Thought 也不能直接等同于其内部决策机制的完整、忠实解释。**

## 4. 实验结果：ReAct 是否始终优于 CoT？

论文基于 PaLM-540B 进行了知识密集型任务实验，部分结果如下 [1, Table 1]：

| 方法             | HotpotQA EM (%) | FEVER Acc (%) |
| -------------- | --------------: | ------------: |
| Act-only       |            25.7 |          58.9 |
| CoT            |            29.4 |          56.3 |
| ReAct          |            27.4 |          60.9 |
| ReAct → CoT-SC |        **35.1** |          62.0 |
| CoT-SC → ReAct |            34.2 |      **64.6** |

其中，CoT-SC 表示采用 Self-Consistency 的 Chain-of-Thought 方法，箭头表示论文设计的策略切换方向。

实验表明：

**第一，ReAct 相比 Act-only 在两个任务上均有提升。**

这支持了显式语言推理有助于环境动作决策及最终答案整合的结论。

**第二，ReAct 并非在所有任务上优于 CoT。**

在 HotpotQA 中，ReAct 的 EM 低于 CoT；但在 FEVER 上，ReAct 取得更高的分类准确率。

这说明工具交互与语言推理之间存在权衡。外部检索可以提供事实依据，但也可能引入不相关结果、额外决策步骤和错误传播路径。

**第三，推理与检索的混合策略能够带来进一步提升。**

论文设计了两类回退机制：

* `ReAct → CoT-SC`：当 ReAct 在指定步数内未能完成任务时，回退到 CoT-SC。
* `CoT-SC → ReAct`：当 CoT-SC 的答案一致性不足时，转而执行外部检索。

其思想是根据任务中的信息需求，在内部知识推理和外部知识获取之间动态切换。

需要注意，混合方法会涉及额外采样或工具调用，因此性能提升不应脱离推理成本进行比较。

## 5. 局限性与工程启示

### 5.1 ReAct 不保证推理正确

ReAct 通过外部信息增强事实依据，但并不保证：

* 检索结果一定相关或真实。
* Thought 对 Observation 的解释一定正确。
* Agent 一定能够跳出重复搜索或错误决策循环。
* 最终答案一定能够从可见证据中得到严格支持。

论文的错误分析也指出，信息不足的搜索结果以及重复生成 Thought 和 Action 都可能导致任务失败。

因此，**Observation 的引入提高了外部事实可用性，但不等于完成了事实验证。**

### 5.2 工程实现需要明确模型与执行器的边界

在实际 Agent 系统中，可以将 ReAct 抽象为三个模块：

| 模块                      | 职责                     |
| ----------------------- | ---------------------- |
| LLM Policy              | 根据任务上下文生成下一步决策与工具调用请求  |
| Tool Executor           | 解析、校验并执行工具调用，处理超时与异常   |
| State / Context Manager | 保存必要的历史观察、证据、任务进度与终止状态 |

模型负责提出 Action，但真实工具调用由执行器完成。

因此，生产环境中通常还需要额外的控制机制：

**工具边界**：使用明确的工具 Schema、参数校验、权限约束，避免执行不合法或越权动作。

**执行约束**：设置最大迭代次数、超时、重试次数与异常退出机制，限制无效循环和资源消耗。

**证据管理**：记录工具输入、原始返回、信息来源以及最终结论之间的对应关系，避免将模型自行推断的内容误记为工具证据。

**安全隔离**：将外部工具输出视为不可信输入，防止检索内容中的恶意指令改变 Agent 的执行权限与任务目标。

**可观测性**：记录动作、参数、反馈、决策摘要与终止原因，而不必依赖完整内部推理文本作为系统日志。

这些属于从 ReAct 机制延伸出的工程设计要求，而不是原论文已经完整解决的问题。

### 5.3 ReAct 与 Toolformer 的区别

ReAct 与 Toolformer [2] 都关注语言模型的工具使用能力，但研究重点不同。

| 维度   | ReAct              | Toolformer          |
| ---- | ------------------ | ------------------- |
| 核心问题 | 如何通过推理与行动交替完成任务    | 如何让模型学习何时、如何调用 API  |
| 主要方法 | Few-shot 推理与行动轨迹提示 | 自监督生成与筛选 API 调用训练数据 |
| 核心关注 | 运行时的多步决策和环境反馈      | 模型层面的工具调用能力学习       |
| 工具反馈 | 用于更新后续推理与决策        | 用于训练数据构建及后续文本预测     |

两者并非相互替代的技术路线。

ReAct 侧重**推理时的决策组织机制**；Toolformer 侧重**工具使用能力的学习过程**。

### 5.4 ReAct 与 Agent 架构的关系

Anthropic 在 *Building Effective Agents* [3] 中区分了两类系统：

* **Workflow**：通过预先定义的代码路径组织模型与工具调用。
* **Agent**：由模型动态决定执行步骤和工具使用方式。

从这一角度看，ReAct 为后者提供了一个典型的运行时决策范式。

但 ReAct 并不意味着所有任务都应该交给自主 Agent。

对于步骤明确、路径稳定、结果易验证的任务，固定 Workflow 通常更容易控制成本和错误。

当任务具有开放性，且必须根据环境反馈动态决定后续步骤时，ReAct 式交互循环才更能体现价值。

## 6. 总结

ReAct 的核心贡献是将语言推理纳入智能体的序列决策过程，使模型能够在任务执行过程中交替进行信息整合、环境交互与决策更新。

从方法层面看，ReAct 通过扩展动作空间，使 Thought 成为影响后续行动的显式上下文；从交互层面看，Observation 为后续决策提供外部信息；从实验层面看，它验证了显式推理与环境行动协同的价值，同时也暴露了检索依赖、循环失败与推理灵活性方面的限制。

我认为，理解 ReAct 最重要的不是记住 `Thought → Action → Observation` 的结构，而是理解以下区别：

> **工具调用解决的是模型如何获取外部信息；ReAct 进一步研究的是模型如何依据这些信息，持续调整自己的决策。**

对于 Agent 工程而言，ReAct 提供的是一种决策机制基础，而可靠的智能体系统还需要结合工具执行约束、证据验证、状态管理和终止控制，才能将这一机制转化为可维护、可评估的实际应用。

---

## 参考文献

[1] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). **ReAct: Synergizing Reasoning and Acting in Language Models.** *International Conference on Learning Representations (ICLR 2023).*
论文：https://arxiv.org/abs/2210.03629
PDF：https://arxiv.org/pdf/2210.03629
项目与代码：https://react-lm.github.io/

[2] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). **Toolformer: Language Models Can Teach Themselves to Use Tools.** *Advances in Neural Information Processing Systems 36 (NeurIPS 2023).*
论文：https://arxiv.org/abs/2302.04761
会议论文：https://proceedings.neurips.cc/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html

[3] Anthropic. (2024, December 19). **Building Effective Agents.** *Anthropic Engineering.*
原文：https://www.anthropic.com/engineering/building-effective-agents
