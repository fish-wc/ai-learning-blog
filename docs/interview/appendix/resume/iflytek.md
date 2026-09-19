---
title: "简历准备"
date: 2026-09-19
categories:
  - resume
tags:
  - resume
math: false
---



# 从 Terminal Agent 数据到 SFT：一次大模型 Agent 后训练实践梳理

> 本文从工程视角梳理 Terminal Agent 后训练的一条完整链路：
> **任务构建 → 环境验证 → Agent Rollout → 轨迹筛选 → SFT 数据构造 → 训练 → 评测 → Badcase 回流。**
>
> 重点在于理解：**一个能够真正操作终端的 Agent，是如何通过数据和训练逐步获得工具使用、环境反馈理解以及多步执行能力的。**

---

## 一、真正需要记住的核心信息

这 12 条是“项目骨架”。不是逐字背，而是保证任何一条都能展开讲 1～3 分钟。

1. **一句话项目介绍**：我的工作主要是把 Terminal 任务构造成可执行、可验证的环境，让 Teacher Agent rollout 出多轮交互轨迹，再经过自动验证和质量筛选转成 SFT 数据，训练后通过固定 Harness 评测，并根据 badcase 继续迭代数据。

2. **Terminal Agent 和普通 SFT 的区别**：普通 SFT 更像 `instruction → answer`；Terminal Agent 学的是 `task + history + environment observation → next action`，核心是与环境持续交互。

3. **一个标准任务至少包含什么**：Instruction、Docker Environment、Oracle/Reference Solution、Verifier/Test。Docker 保证隔离和可复现；Oracle 证明任务可解；Verifier 判断 Agent 最终是否真的成功。

4. **Oracle 和 Verifier 不一样**：Oracle 回答“这道题能不能做出来”；Verifier 回答“Agent 当前有没有做出来”。最好还做 NOP check：什么都不做必须失败。

5. **Harness 是什么**：它是模型和环境之间的执行层，负责构造上下文、调用 LLM、解析 tool call、执行 shell、回填 stdout/stderr、维护历史、控制 timeout/max turns、运行 verifier 和记录 trajectory。记住：`Agent ≠ Model`。

6. **Trajectory 是什么**：不是最终答案，而是完整的 `Task → Action → Observation → Action → Observation → ... → Verifier`。Harbor 当前的 ATIF 规范也把 trajectory 定义成包含 agent reasoning、actions 和 observations 的完整交互历史，并面向调试、SFT、RL 共用。([GitHub][1])

7. **为什么一个 Task 要多次 Rollout**：模型具有随机性，同一道题可以得到成功、失败、错误恢复和不同路径的数据。因此“3 万条”更合理地理解为 trajectory/episode 数，而不是 3 万个独立任务。

8. **Success 不等于高质量数据**：成功轨迹可能靠 shortcut 或碰巧做对。真正有训练价值的是比较清晰的 `Inspect → Act → Verify`，以及发生错误以后根据 observation 做 Recovery。

9. **失败轨迹不能一刀切**：Infrastructure failure 要排除；死循环式失败通常没价值；Recoverable failure 很有价值，因为它能教模型如何读 stderr、修正策略并恢复执行。

10. **Loss Mask 怎么理解**：标准 Agent SFT 一般把 environment/tool observation 当上下文，主要监督 assistant/action token，而不是训练模型自己“生成 shell 输出”。当前 TRL 直接支持 `assistant_only_loss=True`。([Hugging Face][2])

11. **实验为什么要固定 Harness**：Agent 最终效果不仅取决于模型，还取决于 prompt、tool schema、max turns、timeout、sampling、environment 和 verifier。因此比较不同 checkpoint 时必须尽量只改变一个变量。

12. **19%→61% 应该怎么解释**：这是项目级结果，不要直接声称“我的方法贡献了 42 个百分点”。真正证明某个数据策略有效的是控制变量实验和 ablation；同时训练任务和 benchmark 必须隔离，防止 contamination。

只过这一行：

> **Task 标准化 → Docker → Oracle/NOP → Teacher Rollout → Trajectory → Verifier → Cleaning → Inspect-Act-Verify / Recovery → SFT + Loss Mask → Fixed Harness Eval → Badcase → Data Iteration。**


---

## 1. 我们到底在训练什么？

简述：学习基于task、history、observation去判断next action的能力，即 $ P(a_t \mid task, history, observation_{1:t-1}) $。

普通的指令微调通常可以抽象成：

```text
User Instruction
      ↓
Model
      ↓
Assistant Answer
```

例如：

```text
User:
解释一下什么是快速排序。

Assistant:
快速排序是一种基于分治思想的排序算法……
```

但 Terminal Agent 明显不一样。

它面对的不是一个静态问题，而是一个会随着操作不断变化的环境。

例如用户要求：

```text
修复当前 Python 项目，使所有测试通过。
```

Agent 的执行过程可能是：

```text
查看目录
   ↓
运行测试
   ↓
观察报错
   ↓
阅读相关代码
   ↓
修改文件
   ↓
再次运行测试
   ↓
发现新的问题
   ↓
继续修改
   ↓
最终验证
```

因此，一个 Terminal Agent 真正需要学习的不是：

> 给定问题，生成一个答案。

而是：

> **根据当前环境状态，决定下一步应该执行什么动作。**

更准确地说，它学习的是：

```text
Task
  +
History
  +
Environment Observation
        ↓
Next Action
```

即：

$$
P(a_t \mid task, history, observation_{1:t-1})
$$

这也是整个项目最核心的出发点。

---

# 2. Terminal Agent 的核心不是“回答”，而是“交互”

简述：一次完整的Terminal Agent执行包含Observation → Action → Verification。

一次完整的 Terminal Agent 执行通常包含三类信息：

```text
Observation → Action → Verification
```

例如：

```text
Agent:
先运行测试看看问题在哪里。

Action:
pytest -q

Observation:
2 failed, 18 passed
AssertionError: ...

Agent:
错误出现在 parser.py，先看一下对应实现。

Action:
sed -n '1,200p' src/parser.py

Observation:
...

Agent:
修改代码。

Action:
...

Observation:
...

Agent:
重新运行测试。

Action:
pytest -q

Observation:
20 passed
```

可以把它进一步抽象成：

```text
Inspect
   ↓
Act
   ↓
Observe
   ↓
Reason / Recover
   ↓
Act
   ↓
Verify
```

因此，Terminal Agent 的训练数据也不能只是：

```text
问题 → 最终答案
```

而应该尽可能保存完整的：

```text
任务
→ 模型决策
→ 工具调用
→ 环境反馈
→ 下一次决策
→ ...
→ 最终结果
```

这段完整的交互历史通常称为 **Trajectory（轨迹）**。

---

# 3. 整个后训练 Pipeline 长什么样？

整体流程可以概括为：

```text
Raw Tasks
   ↓
任务标准化
   ↓
Docker 环境构建
   ↓
Oracle / Verifier 验证
   ↓
Teacher Agent Rollout
   ↓
Raw Trajectories
   ↓
自动验证 + 数据清洗
   ↓
轨迹质量分析
   ↓
SFT 数据构造
   ↓
模型训练
   ↓
固定 Harness 评测
   ↓
Badcase 分析
   ↓
下一轮数据策略
```

真正做项目以后会发现：

> **模型训练本身只是其中一个环节，数据和环境往往占据了更多工作量。**

下面逐步展开。

---

# 4. 第一步：把任务变成“真的可以执行”

假设我们拿到一个原始任务：

```text
修复项目中的 bug，使全部测试通过。
```

只有这一句话是不够的。

为了让 Agent 真正执行，我们还需要：

```text
任务描述
+
初始文件系统
+
依赖环境
+
测试程序
+
成功判定标准
```

因此，一个完整的 Terminal Task 通常可以理解为：

```text
task/
├── instruction
├── environment/
│   └── Dockerfile
├── solution/
│   └── reference_solution.sh
└── tests/
    └── verifier.sh
```

文件名可以不同，但逻辑基本一致。

其中有三个特别重要的部分。

---

## 4.1 Environment：Agent 在哪里执行？

Terminal Agent 会：

* 创建文件；
* 删除文件；
* 安装依赖；
* 修改配置；
* 编译代码；
* 运行程序。

这些操作具有很强的副作用。

因此不能让不同 Agent 共享同一个运行环境。

一种自然的做法是：

```text
Task A → Container A
Task B → Container B
Task C → Container C
```

每个任务从一个相对干净的 Docker 环境开始。

这样做有两个好处：

### 隔离

Task A 的操作不会污染 Task B。

### 可复现

同一个任务可以重新启动一个新的 container，再执行一次。

对于 Agent 数据来说，**可复现性非常重要**。

因为我们最终希望得到的不是：

> “某一次偶然跑成功的数据。”

而是：

> “在相同初始环境下能够稳定重现的执行轨迹。”

---

# 5. Oracle 和 Verifier 分别解决什么问题？

## 5.1 Oracle：证明任务是可以完成的

Oracle 可以理解为：

> 一套已知能够完成任务的参考解法。

例如：

```bash
sed -i 's/xxx/yyy/' src/parser.py
pytest -q
```

Oracle 最重要的意义其实不是：

> 给模型提供标准答案。

而是帮助我们验证：

> **这个任务本身到底是不是好的。**

假设 10 个 Teacher Agent 全部失败。

原因可能有两类。

第一类：

```text
模型不会做
```

第二类：

```text
Docker 镜像坏了
依赖版本发生变化
测试程序写错
任务本身不可解
timeout 设置太短
```

如果 Oracle 也跑不通，那么问题很可能并不在模型。

因此一个很自然的 Task Validation 流程是：

```text
Build Container
      ↓
Run Oracle
      ↓
Run Verifier
      ↓
PASS
```

通过oracle 任务才适合进入后续数据生产。

---

## 5.2 Verifier：判断 Agent 最后到底有没有完成任务

Oracle 回答：

> 这个任务能不能完成？

Verifier 回答：

> Agent 现在到底完成没有？

例如任务是：

```text
修复代码，使所有测试通过。
```

Verifier 最简单可以是：

```bash
pytest -q
```

最终：

```text
exit code = 0
```

则认为任务成功。

这里有一个非常重要的原则：

> **不要仅依赖模型自己说“我完成了”。**

例如模型最后输出：

```text
问题已经修复，所有测试应该可以通过。
```

这不代表它真的完成了任务。

Agent 的成功应该尽量由：

```text
Environment State
+
Programmatic Verifier
```

决定。

而不是由语言模型自己判断。

---

# 6. 一个容易忽略的检查：什么都不做能不能通过？

除了 Oracle Pass，还应该考虑一个很有意思的问题：

> 如果 Agent 什么都不做，Verifier 会不会直接通过？

例如：

```text
Fresh Container
      ↓
No Operation
      ↓
Verifier
```

理论上应该：

```text
FAIL
```

如果什么都没做就成功，那么可能说明：

* 初始环境本身已经满足目标；
* verifier 写错了；
* task 与 verifier 不一致。

这种任务没有什么训练价值。

所以，一个比较可靠的任务至少应该满足：

```text
Oracle → PASS
NOP    → FAIL
```

---

# 7. 第二步：让 Teacher Agent 真正去做任务

当任务环境通过验证以后，就可以开始产生 trajectory。

这一步通常叫：

**Rollout**

可以理解为：

> 让某一个 Agent 从初始环境开始完整执行一次任务。

一个简单架构是：

```text
                 Task Queue
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   Container A  Container B  Container C
        ↓            ↓            ↓
      Agent        Agent        Agent
        ↓            ↓            ↓
            Teacher Model
```

其中 Agent Harness 负责把模型和环境连接起来。

---

# 8. Harness 到底是什么？

简述：**连接 LLM 和真实工具环境的一层执行框架。**

Harness 可以理解成：

> **连接 LLM 和真实工具环境的一层执行框架。**

模型本身只能生成 token。

它不能直接运行：

```bash
pytest
```

因此 Harness 通常负责：

```text
构造模型输入
     ↓
调用 LLM
     ↓
解析 Tool Call
     ↓
执行 Shell Command
     ↓
捕获 stdout / stderr
     ↓
返回 Observation
     ↓
继续调用 LLM
     ↓
...
     ↓
调用 Verifier
     ↓
保存 Trajectory
```

因此有一个很重要的概念：

```text
Model ≠ Agent
```

更接近：

```text
Agent
=
Model
+
Prompt
+
Tool Protocol
+
Harness
+
Environment
```

所以做模型评测的时候，必须尽可能固定 Harness。

否则最终结果可能并不是模型能力本身发生变化。

---

# 9. 一条 Trajectory 到底保存什么？

简述：思考执行观察。

一条比较完整的轨迹可能是：

```text
User Task
   ↓
Assistant Thought / Action
   ↓
Tool Call
   ↓
Observation
   ↓
Assistant Action
   ↓
Tool Call
   ↓
Observation
   ↓
...
   ↓
Final Answer
   ↓
Verifier Result
```

在数据层面，可以抽象成：

```json
{
  "task_id": "task_001",
  "messages": [
    {
      "role": "user",
      "content": "Fix the project..."
    },
    {
      "role": "assistant",
      "content": "先运行测试。",
      "tool_calls": ["pytest -q"]
    },
    {
      "role": "tool",
      "content": "2 failed, 18 passed..."
    },
    {
      "role": "assistant",
      "content": "查看对应代码...",
      "tool_calls": ["cat src/parser.py"]
    }
  ],
  "reward": 1
}
```

真正重要的是：

> **Trajectory 不是简单文本，而是一次环境状态不断变化的过程记录。**

---

# 10. 为什么一个 Task 要 Rollout 多次？

因为 LLM 的行为不是完全确定的。

同一个任务，三次 rollout 可能完全不同。

第一次：

```text
inspect
→ 定位问题
→ 修改
→ verify
→ success
```

第二次：

```text
直接猜
→ 修改错误
→ 重复尝试
→ timeout
```

第三次：

```text
inspect
→ 修改
→ 遇到新 error
→ recovery
→ verify
→ success
```

因此：

```text
一个 Task
≠
一条 Trajectory
```

更常见的是：

```text
Task
├── Teacher A / Rollout 1
├── Teacher A / Rollout 2
├── Teacher B / Rollout 1
├── Teacher B / Rollout 2
└── ...
```

因此数万条 trajectory 并不意味着有数万个独立任务。

数据规模通常来自：

```text
Unique Tasks
×
Teacher Models
×
Repeated Rollouts
×
Different Sampling Configurations
```

---

# 11. 第三步：成功的 Trajectory 不一定是好的训练数据

这是整个项目中我认为最值得理解的一点。

假设两个 Agent 最终都成功了。

## Trajectory A

```text
查看错误日志
↓
定位相关文件
↓
阅读代码
↓
修改
↓
执行测试
↓
发现问题
↓
继续修改
↓
最终验证
```

## Trajectory B

```text
直接修改文件
↓
直接执行几个命令
↓
碰巧成功
```

从 verifier 看：

```text
A = success
B = success
```

但是从训练角度看，两条轨迹的价值完全不同。

Trajectory A 教给模型的是：

> 遇到未知任务时，应该如何利用环境信息逐步解决问题。

Trajectory B 更可能只是：

> 某一个任务上的偶然操作序列。

因此：

> **Outcome Quality 和 Training Quality 不是一回事。**

这也是为什么 Agent 后训练不能只做：

```text
成功 → 保留
失败 → 删除
```

而应该进一步研究过程质量。

---

# 12. Inspect → Act → Verify

一种比较直观的高质量轨迹结构是：

```text
Inspect
   ↓
Act
   ↓
Verify
```

例如修改一个 Python 文件。

低质量行为：

```text
直接修改 parser.py
↓
直接修改 utils.py
↓
宣布完成
```

高质量行为：

```text
运行测试
↓
读取 error
↓
查看 parser.py
↓
针对 error 修改
↓
再次运行测试
↓
检查结果
```

这里至少包含三个关键能力：

### Inspect

执行动作之前，先获取足够环境信息。

### Act

动作应该和刚才观察到的信息相关。

### Verify

修改以后主动检查结果，而不是默认自己做对了。

因此我们可以围绕这些过程建模一些“轨迹特征”。

例如：

```text
是否在重要动作之前观察相关文件？
是否读取了 stderr？
修改代码以后是否运行测试？
出现错误以后有没有调整方案？
是否反复执行同一个无效命令？
最终完成前有没有 verification？
```

这些特征不一定需要非常复杂的模型。

很多时候：

> **简单规则 + 自动统计 + 人工抽样**

已经可以帮助发现大量数据问题。

---

# 13. 失败 Trajectory 有没有价值？

简述：异常恢复能力本身就是 Agent 很重要的一种能力。


> 有，但不能直接全部作为 SFT 正样本。

需要区分不同失败类型。

---

## 13.1 无价值失败

例如：

```text
执行 command A
↓
失败
↓
再次执行 command A
↓
失败
↓
再次执行 command A
↓
timeout
```

这种轨迹通常没有太大教学价值。

---

## 13.2 Infrastructure Failure

例如：

```text
Docker 启动失败
API Timeout
网络异常
磁盘错误
模型服务错误
```

这甚至不能算模型失败。

如果不单独处理，会错误地低估模型能力。

---

## 13.3 Recoverable Failure

这一类反而很有价值。

例如：

```text
pip install package==1.0
↓
Dependency Conflict
↓
读取错误信息
↓
重新分析版本关系
↓
安装 package==1.2
↓
Success
```

这里第一次 action 虽然失败了，但模型表现出了：

```text
Error
   ↓
Observation
   ↓
Reasoning
   ↓
Recovery
```

真实世界中的 Agent 不可能永远第一次就做对。

因此：

> **异常恢复能力本身就是 Agent 很重要的一种能力。**

所以有价值的不是“失败”本身，而是：

> **模型如何利用失败后的环境反馈继续解决问题。**

---

# 14. 第四步：把 Trajectory 转成 SFT 数据

完成轨迹筛选以后，就进入模型训练数据构造阶段。

一条 Agent SFT 数据通常包含：

```text
system
user
assistant
tool
assistant
tool
assistant
...
```

但这里有一个很重要的问题：

> 所有 token 都应该计算 Loss 吗？

通常不是。

---

# 15. Loss Mask：模型到底应该学习哪些 Token？

简述：**模型主要学习 Assistant 的决策**。

考虑下面这段数据：

```text
User:
修复代码。

Assistant:
先运行测试。

Tool:
2 failed, 18 passed

Assistant:
查看 parser.py。

Tool:
...
```

其中：

```text
2 failed, 18 passed
```

并不是模型生成的。

它来自真实环境。

因此一种很常见的 SFT 方式是：

```text
System             Mask
User               Mask
Assistant           Loss
Tool Observation    Mask
Assistant           Loss
Tool Observation    Mask
Assistant           Loss
```

也就是说：

> **Observation 用来作为条件信息，但模型主要学习 Assistant 的决策。**

模型真正需要优化的是：

```text
看到这个环境反馈以后
下一步应该做什么？
```

而不是：

```text
自己生成一个假的 Shell 输出。
```

这也是理解 Agent SFT 非常关键的一点。

---

# 16. Multi-Teacher 为什么有意义？

简述：**研究不同 Teacher 产生的数据分布有什么差异，以及这种差异是否能够迁移到 Student。**

直觉上似乎应该：

> 永远使用最强的 Teacher。

但实际不一定这么简单。

例如一个非常强的 Teacher 可能：

```text
几乎不 inspect
↓
直接定位问题
↓
一步解决
```

这种轨迹虽然非常高效，但对于能力较弱的 Student 来说，不一定容易模仿。

另一个 Teacher 可能：

```text
先检查
↓
逐步定位
↓
尝试
↓
遇到错误
↓
恢复
↓
最终完成
```

它自己的 benchmark 分数可能没有第一个 Teacher 高。

但它产生的 trajectory 反而可能包含更多：

```text
环境理解
工具使用
错误恢复
验证过程
```

因此应该区分两个概念：

```text
Teacher Quality
≠
Teaching Quality
```

我们真正关心的是：

> **哪种 Teacher 生成的数据，对 Student 的 downstream performance 最有帮助。**

所以 Multi-Teacher 实验的目的并不是单纯比较：

> 哪个大模型最强。

而是比较：

> **不同 Teacher 产生的数据分布有什么差异，以及这种差异是否能够迁移到 Student。**

---

# 17. 为什么还要做任务难度分桶？

简述：**研究什么样的数据分布最适合当前阶段的 Student。**

不同任务对模型的价值并不相同。

一种很自然的 difficulty proxy 可以来自：

```text
Teacher Success Rate
Trajectory Length
Tool Call Count
Runtime
Oracle Steps
需要修改的文件数量
是否需要编译
是否包含复杂依赖
```

例如：

```text
Easy:
大部分 Teacher 都可以稳定完成。

Medium:
一部分能够完成。

Hard:
大多数 Teacher 都失败。
```

然后就可以设计不同的数据配比：

```text
Recipe A
Easy   50%
Medium 40%
Hard   10%

Recipe B
Easy   30%
Medium 50%
Hard   20%
```

训练两个 checkpoint。

最终在相同评测环境下比较。

这里真正研究的是：

> **什么样的数据分布最适合当前阶段的 Student。**

而不是简单认为：

> 越难的数据越好。

---

# 18. 数据实验最重要的是控制变量

假设我们想研究：

> Teacher A 和 Teacher B 谁的数据更适合 Student。

错误实验：

```text
Teacher A：20k 条
Teacher B：80k 条
```

最后 B 更好。

这不能证明：

```text
Teacher B 的数据质量更高
```

也可能只是：

```text
B 的数据更多
```

因此尽量应该控制：

```text
数据数量
Token Budget
任务分布
Base Model
Learning Rate
Training Steps
Batch Size
Harness
Eval Set
Decoding Parameters
```

然后只改变：

```text
Teacher Source
```

同理，如果比较：

```text
高质量轨迹
vs
随机轨迹
```

最好也尽可能保持：

```text
任务数量
总 Token
难度分布
```

一致。

这才叫：

**Controlled Ablation**

---

# 19. 第五步：SFT 训练

TODO 探究一下怎么设置Loss Mask

进入训练阶段以后，一个比较合理的算法实习工作边界通常是：

```text
准备数据
↓
设置数据配比
↓
设置 Loss Mask
↓
配置 Sequence Length
↓
配置训练 Recipe
↓
提交 Job
↓
查看 Loss / Checkpoint
↓
运行 Evaluation
```

而不是：

```text
自己实现整个分布式训练系统
```

大规模模型底层通常还涉及：

```text
DP
TP
PP
EP
ZeRO / FSDP
通信库
设备调度
故障恢复
Checkpoint 系统
```

这些通常由统一训练平台负责。

对于做数据和后训练的算法工程而言，更核心的问题是：

补充：训练recipe ，一般有batch size、学习率、epoch、warmup、权重衰减、梯度累积。

> **训练什么数据，以什么方式训练，以及训练以后能力到底发生了什么变化。**

---


# 20. 为什么“19% → 61%”不能直接理解成某一个方法带来的提升？

假设最终结果：

```text
Baseline
19%

Final Checkpoint
61%
```

很容易产生一个错误结论：

> 某一个数据筛选方法带来了 42 个百分点提升。

实际上一个完整项目期间可能同时发生：

```text
新增数据
Teacher 变化
数据清洗
Loss Mask 调整
样本配比调整
Training Step 变化
Prompt 优化
Harness 修复
其他训练数据加入
```

因此：

```text
Project-level Improvement
```

和：

```text
Method-level Contribution
```

必须分开。

真正判断某个方法有没有价值，要看：

**Ablation Experiment**

例如：

```text
Random Data
→ 35%

Filtered Data
→ 41%
```

或者：

```text
Without Recovery Data
→ 38%

With Recovery Data
→ 42%
```

这里的差异才比较能够反映某个具体数据策略的作用。

所以：

> **总成绩说明整个项目最终做到什么程度；消融实验才说明某个方法到底有没有贡献。**

---

# 21. 最后一步：Badcase 如何重新变成训练数据？

TODO 详细知识在"20260512 分析BrowseComp.docx"里面，有空整理一下。

最终评测结束以后，并不是只记录：

```text
Success Rate = 61%
```

更有价值的是分析剩下的失败任务。

例如：

```text
任务理解错误
↓
没有 Inspect 环境

---

理解正确
↓
执行 Command 错误

---

Command 失败
↓
没有读取 stderr

---

读取错误
↓
不会 Recovery

---

修改完成
↓
没有 Verify

---

Verify Fail
↓
仍然宣布完成
```

然后把失败类型做成 taxonomy。

例如发现大量问题来自：

```text
错误发生以后不会 recovery
```

下一轮可以增加：

```text
Recoverable Failure Trajectories
```

如果发现：

```text
修改文件以前很少 inspect
```

下一轮可以增加：

```text
Inspect-heavy Trajectories
```

于是整个项目就形成一个真正的数据闭环：

```text
Data
 ↓
SFT
 ↓
Evaluation
 ↓
Badcase
 ↓
Failure Taxonomy
 ↓
Data Strategy
 ↓
New Data
 ↓
SFT
```

这其实是 Agent 后训练里非常核心的一种工作方式。

---

# 22. 这项工作的核心到底是什么？

刚开始接触 Terminal Agent 时，很容易觉得：

> 核心就是调用更强的大模型生成更多数据。

但真正做完整个流程以后，会发现问题远没有这么简单。

一个高质量 Agent 后训练系统至少需要解决四个问题。

---

## 22.1 Task 是否可靠？

如果任务环境本身坏了，后面的模型训练都没有意义。

因此需要：

```text
Docker
Oracle
Verifier
Environment Reproducibility
```

---

## 22.2 Trajectory 是否可信？

一个成功结果，不代表这是一条好的训练数据。

因此需要：

```text
Automatic Verification
Trajectory Cleaning
Inspect-Act-Verify Analysis
Recovery Analysis
```

---

## 22.3 数据是否真的能迁移到 Student？

Teacher 自己强，不代表它生成的数据一定更适合 Student。

因此需要研究：

```text
Teacher Source
Difficulty
Success / Failure
Trajectory Quality
Sampling Ratio
Loss Mask
```

---

## 22.4 提升到底来自哪里？

Final Score 只能说明项目结果。

想知道某一个策略有没有效果，需要：

```text
Controlled Experiment
+
Ablation
+
Held-out Evaluation
```

---

# 23. 最后总结

简述：**Agent 后训练本质上是一项围绕“环境、轨迹和反馈”展开的数据工程问题。**

如果把整个 Terminal Agent 后训练流程压缩成一句话，我会这样描述：

> **先把真实终端任务构造成可执行、可验证的task数据，让 Teacher Agent 在环境中产生完整交互轨迹，再从大量轨迹中筛选真正具有教学价值的“观察—行动—验证—恢复”过程，将其转换为 SFT 数据训练 Student，最后通过固定环境和 Harness 做隔离评测，并利用 Badcase 继续指导下一轮数据构造。**

因此，这项工作的核心并不是简单的：

```text
Generate More Data
```

而更接近：

```text
Build Reliable Tasks
        +
Collect Real Interactions
        +
Select Teachable Trajectories
        +
Train
        +
Evaluate
        +
Iterate
```











