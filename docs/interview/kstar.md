---

title: "从岗位 JD 反推学习路线：大模型推理/训练优化工程师应该学什么？"
date: 2026-09-16
tags:

- AI Infra
- LLM Inference
- CUDA
- vLLM
- AI Compiler
- Triton
- MLIR
- 求职

categories:
- AI Infra

---



# 从岗位 JD 反推学习路线：大模型推理/训练优化工程师应该学什么？

> 我的目标不是在简历上多写几个 `CUDA`、`vLLM`、`TensorRT-LLM` 这样的关键词，而是真正理解：
>
> **一个大模型从 PyTorch 代码，到计算图，到 GPU Kernel，再到线上推理服务，中间到底发生了什么？我们又能在哪些地方让它变得更快？**

---

# 1. 从招聘要求开始：这个岗位到底在做什么？

我当前关注的岗位是：

> **大模型推理 / 训练优化工程师**

招聘要求里出现了很多关键词：

```text
Python / C++
Linux
PyTorch / TensorFlow
vLLM
TensorRT-LLM
MLC-LLM
CUDA / ROCm
Nsight Systems
Nsight Compute
GPU Kernel
算子融合
图优化
Codegen
XLA
TVM
MLIR
LLVM
```

第一次看到这些东西时，很容易产生一个感觉：

> 每一个东西都很大，到底应该先学哪个？

实际上，这些技术并不是互相独立的。

它们共同组成了一条完整的 **AI Infra 技术链路**。

可以粗略画成下面这样：

```text
                     大模型训练 / 推理代码
                             │
                             ▼
                  PyTorch / TensorFlow
                             │
                     Graph Capture
                             │
                             ▼
                 FX Graph / StableHLO / IR
                             │
                    Graph Optimization
                     Pass / Fusion
                             │
                             ▼
             TorchInductor / XLA / TVM / MLIR
                             │
                         Codegen
                             │
                             ▼
              Triton / CUDA / ROCm / LLVM
                             │
                             ▼
                        GPU Kernel
                             │
                             ▼
                 NVIDIA / AMD / ASIC
                             │
                             ▼
             Nsight Systems / Nsight Compute
                             │
                             ▼
                  找瓶颈 → 优化 → Benchmark
```

而在大模型推理场景中，上面还会再增加一层：

```text
用户请求
   │
   ▼
LLM Serving System
   │
   ├── Scheduler
   ├── Continuous Batching
   ├── KV Cache Management
   ├── Prefix Cache
   ├── Speculative Decoding
   └── Distributed Inference
   │
   ▼
vLLM / TensorRT-LLM / MLC-LLM
   │
   ▼
Compiler + Runtime + GPU Kernel
```

所以我现在对这个岗位的理解可以浓缩成一句话：

> **让一个深度学习模型，尤其是大模型，在特定硬件上跑得更快、更省显存、更高吞吐，并且最终能够稳定地部署成服务。**

这才是这份 JD 背后真正想招的人。

---

# 2. 为什么这个岗位同时要求 vLLM、CUDA 和编译器？

这是我一开始最困惑的地方。

为什么一个岗位既要求：

```text
vLLM
```

又要求：

```text
CUDA
```

甚至还希望你懂：

```text
MLIR / LLVM / TVM / XLA
```

答案是：

> 因为性能问题可能出现在整个计算栈的任何一层。

举一个非常简单的例子。

假设现在一个模型推理很慢。

第一种可能是：

```text
请求调度不合理
↓
GPU 吃不满
↓
吞吐低
```

那么可能需要优化：

```text
Continuous Batching
Scheduler
Chunked Prefill
```

这是 **vLLM / Serving System** 层的问题。

---

第二种可能是：

```text
KV Cache 占用了大量显存
↓
能同时处理的请求数量太少
↓
吞吐上不去
```

那么需要优化：

```text
KV Cache Management
PagedAttention
Prefix Caching
KV Cache Quantization
```

vLLM 最经典的 PagedAttention 工作就是从这个问题出发：传统 KV Cache 管理存在碎片和冗余的问题，PagedAttention 借鉴操作系统虚拟内存中的分页思想管理 KV Cache。原始论文报告，在其实验配置下，相比当时的系统能够显著提高吞吐。

---

第三种可能是：

```text
模型中存在大量小 Operator
↓
Kernel Launch 很多
↓
频繁读取 / 写回显存
↓
性能差
```

那么可能需要：

```text
Operator Fusion
Graph Fusion
```

例如：

```text
Add
 ↓
LayerNorm
 ↓
Activation
```

原本可能需要多个 kernel。

经过 fusion：

```text
Fused Kernel
```

就有机会减少中间结果在显存中的读写。

Triton 官方教程中的 Fused Softmax 就是非常典型的学习案例：通过融合原本需要多次 GPU memory round-trip 的操作，理解为什么算子融合可以改善 bandwidth-bound workload。

---

第四种可能是：

```text
Kernel 本身写得不好
```

比如：

```text
Global Memory Access 不连续
Shared Memory 使用不合理
Bank Conflict
Register Pressure 太大
Occupancy 太低
Warp Divergence
```

这时候就必须进入：

```text
CUDA Kernel
```

甚至：

```text
PTX
SASS
```

这一层。

---

第五种可能是：

```text
框架没有自动发现可以优化的计算模式
```

那么就可能需要写：

```text
Compiler Pass
Pattern Rewrite
Graph Rewrite
Lowering
Codegen
```

这就是：

```text
MLIR
LLVM
TVM
XLA
```

存在的意义。

MLIR 官方将 Pass 定义为 transformation / optimization 的基础设施；LLVM 中大量优化同样由 Pass 完成。

所以最终会发现：

> **vLLM、CUDA、Triton、MLIR 并不是四个方向。**
>
> **它们是一个性能优化工程师从上到下看到的四个不同层次。**

---

# 3. 我需要建立什么样的能力结构？

如果把这个岗位需要的知识画成一棵树，我认为大概是下面这样：

```text
AI Infra
│
├── ① 基础能力
│   ├── Python
│   ├── C++
│   ├── Linux
│   ├── 数据结构与算法
│   └── 计算机体系结构基础
│
├── ② 深度学习基础
│   ├── PyTorch
│   ├── Transformer
│   ├── Attention
│   ├── GEMM
│   ├── LayerNorm / RMSNorm
│   ├── RoPE
│   └── MoE
│
├── ③ LLM Inference
│   ├── Prefill / Decode
│   ├── KV Cache
│   ├── Continuous Batching
│   ├── PagedAttention
│   ├── Prefix Cache
│   ├── Quantization
│   ├── Speculative Decoding
│   └── Distributed Inference
│
├── ④ LLM Serving Engine
│   ├── vLLM
│   ├── TensorRT-LLM
│   └── MLC-LLM
│
├── ⑤ GPU Programming
│   ├── CUDA Programming Model
│   ├── Thread / Warp / Block
│   ├── Memory Hierarchy
│   ├── Shared Memory
│   ├── Tensor Core
│   ├── CUDA Stream
│   ├── CUDA Graph
│   └── Kernel Optimization
│
├── ⑥ GPU Kernel DSL
│   ├── Triton
│   └── CUTLASS / CuTe
│
├── ⑦ Performance Analysis
│   ├── Nsight Systems
│   ├── Nsight Compute
│   ├── Roofline Model
│   ├── Compute Bound
│   └── Memory Bound
│
├── ⑧ AI Compiler
│   ├── torch.compile
│   ├── TorchDynamo
│   ├── TorchInductor
│   ├── IR
│   ├── SSA
│   ├── Pass
│   ├── Graph Rewrite
│   ├── Fusion
│   ├── Lowering
│   └── Codegen
│
├── ⑨ Compiler Framework
│   ├── MLIR
│   ├── LLVM
│   ├── TVM
│   └── XLA
│
└── ⑩ Distributed Training
    ├── DDP
    ├── FSDP / ZeRO
    ├── TP
    ├── PP
    ├── CP
    ├── EP
    ├── NCCL
    └── Communication / Computation Overlap
```

看起来非常多。

所以最重要的问题不是：

> 我要不要学这些？

答案显然是都需要慢慢接触。

真正的问题是：

> **学习顺序是什么？**

---

# 4. 我的学习原则：先建立全局，再逐层下钻

我准备采用一种类似计算机网络学习中的方法：

> **先知道整个系统怎么工作，再深入其中的一层。**

也就是：

```text
先建立地图
    ↓
能够跑起来
    ↓
能够 Benchmark
    ↓
能够发现性能问题
    ↓
能够解释为什么慢
    ↓
能够修改代码
    ↓
能够证明真的变快了
```

而不是：

```text
CUDA 第一章
CUDA 第二章
CUDA 第三章
……
LLVM 第一章
LLVM 第二章
……
```

后者很容易学几个月以后依然不知道这些知识到底有什么用。

性能优化最重要的是一个闭环：

```text
Baseline
   ↓
Benchmark
   ↓
Profile
   ↓
发现 Bottleneck
   ↓
提出 Hypothesis
   ↓
Optimization
   ↓
Benchmark Again
   ↓
判断优化是否有效
```

以后我的所有项目都应该遵循这个流程。

---

# 5. 如果我只有一天：建立整张 AI Infra 地图

如果只有一天，我不会去写复杂 CUDA Kernel。

一天时间真正需要解决的问题是：

> **这个行业到底在优化什么？**

一天结束以后，我至少应该可以解释下面这些问题：

```text
1. Transformer 推理过程中发生了什么？
2. Prefill 和 Decode 有什么区别？
3. KV Cache 为什么存在？
4. 为什么 KV Cache 会成为推理瓶颈？
5. Continuous Batching 是什么？
6. PagedAttention 解决什么问题？
7. 什么叫 Operator Fusion？
8. GPU 为什么适合深度学习？
9. CUDA Kernel 是什么？
10. 什么叫 Compute Bound / Memory Bound？
11. 编译器为什么可以加速深度学习？
12. vLLM、TensorRT-LLM、Triton、CUDA、MLIR 分别处在哪一层？
```

## 上午：搞懂一次 LLM 推理

重点理解：

```text
Prompt
 ↓
Tokenizer
 ↓
Embedding
 ↓
Transformer Layer × N
 ↓
Attention
 ↓
FFN
 ↓
Logits
 ↓
Sampling
 ↓
Next Token
```

然后一定要理解两个阶段：

```text
Prefill
```

和：

```text
Decode
```

### Prefill

假设输入：

```text
"请解释一下什么是 CUDA"
```

模型第一次需要同时处理整个 Prompt。

这一步通常包含比较大的矩阵运算，因此具有较高的计算密度。

### Decode

接下来：

```text
生成 Token 1
生成 Token 2
生成 Token 3
……
```

每生成一个 Token，都需要再执行一次模型。

但我们显然不希望每次重新计算之前所有 Token 的：

```text
Key
Value
```

于是就产生了：

```text
KV Cache
```

因此：

> **Prefill 与 Decode 是理解 LLM 推理优化的第一道门。**

在很多典型在线推理配置中，Prefill 更容易表现出较高的计算密度，而小 batch Decode 则更容易受到权重读取、KV Cache 等内存访问影响。这里不能机械地记成“Prefill 一定 Compute Bound、Decode 一定 Memory Bound”，具体瓶颈仍然要根据 batch size、序列长度、模型结构和硬件通过 profiler 判断。

---

## 下午：认识 vLLM

首先不看源码。

只需要理解几个核心概念：

```text
Request
   ↓
Scheduler
   ↓
Batch
   ↓
Model Executor
   ↓
GPU
```

重点学习：

```text
KV Cache
PagedAttention
Continuous Batching
Prefix Caching
Chunked Prefill
```

vLLM 当前官方项目本身已经把 PagedAttention、continuous batching、chunked prefill、prefix caching、CUDA/HIP Graph、量化、优化 attention/GEMM kernel、`torch.compile` 图优化以及多种分布式策略放进了同一个推理系统里。

这其实很好地说明了：

> **LLM 推理优化本身就是一个系统问题，而不是单个 CUDA Kernel 的问题。**

---

## 晚上：认识 GPU + Compiler

GPU 部分只需要理解：

```text
CPU
vs
GPU
```

以及：

```text
Thread
Warp
Block
Grid
SM
Global Memory
Shared Memory
Register
```

官方 CUDA Programming Guide 本身也是从 Programming Model 开始，再逐渐进入 GPU 编程和高级特性。

Compiler 部分理解：

```text
Python
 ↓
Graph
 ↓
IR
 ↓
Optimization
 ↓
Codegen
 ↓
GPU Kernel
```

以 PyTorch 为例，可以先形成：

```text
PyTorch Program
      ↓
TorchDynamo
      ↓
FX Graph
      ↓
TorchInductor
      ↓
Triton / C++
      ↓
GPU / CPU
```

的认知。

PyTorch 官方资料中也明确说明，`torch.compile` 背后包括 TorchDynamo、AOTAutograd/AOTDispatcher 和 TorchInductor；TorchInductor 会针对 GPU 等后端生成优化代码。

### 一天结束后的产出

不要只说：

```text
今天学习了 vLLM。
```

而应该写一篇博客：

```text
《大模型推理优化到底在优化什么？从 PyTorch 到 GPU Kernel 的完整链路》
```

这篇文章就是整个系列的目录。

---

# 6. 如果我有一周：完成第一次“推理优化闭环”

一周的目标应该从：

> 我知道这些名词。

升级为：

> **我真正跑过一个 LLM Serving System，并且测过性能。**

---

## Day 1：Transformer Inference

学习：

```text
Transformer
Attention
GEMM
RMSNorm
RoPE
KV Cache
Prefill
Decode
```

需要回答：

> 为什么生成第 N 个 Token 时不需要重新计算前 N-1 个 Token 的 K/V？

然后手算或者写代码估算：

```text
KV Cache Memory
≈
2
× num_layers
× num_kv_heads
× head_dim
× sequence_length
× bytes_per_element
× batch_size
```

理解这个公式非常重要。

它会直接告诉我们：

```text
Sequence Length ↑
Batch Size ↑
```

为什么会迅速吃掉 GPU Memory。

---

## Day 2：vLLM + PagedAttention

目标：

```text
把一个模型真正部署起来。
```

例如：

```text
Qwen / Llama 等小规模模型
```

然后尝试：

```text
Transformers inference
vs
vLLM inference
```

重点不是“谁快”。

而是思考：

```text
为什么可能更快？
```

阅读：

> **Efficient Memory Management for Large Language Model Serving with PagedAttention**

也就是 vLLM 的经典工作。

这篇论文最值得学习的并不只是一个 Attention 算法，而是一种很重要的系统思想：

```text
操作系统 Virtual Memory
        ↓
Page
        ↓
迁移到 KV Cache Management
```

也就是说：

> 很多 AI Infra 创新，其实来自传统系统知识和 AI workload 的结合。

---

## Day 3：CUDA 基础

开始真正写 CUDA。

至少实现：

```text
Vector Add
```

理解：

```cpp
__global__
blockIdx
threadIdx
blockDim
gridDim
```

然后继续理解：

```text
SIMT
Warp
Memory Coalescing
Shared Memory
Synchronization
```

此时不要急着追求“写 FlashAttention”。

真正重要的是搞清楚：

> 一个 CUDA Kernel 到底是怎么运行在 GPU 上的？

---

## Day 4：Triton

Triton 非常适合作为：

```text
PyTorch
```

与：

```text
CUDA
```

之间的桥梁。

官方教程本身就按照比较合理的顺序提供了：

```text
Vector Addition
Fused Softmax
Matrix Multiplication
Layer Normalization
Fused Attention
```

等例子。

我准备按照：

```text
Vector Add
   ↓
Softmax
   ↓
Matrix Multiplication
   ↓
RMSNorm / LayerNorm
```

逐渐学习。

其中最值得认真做的是：

```text
Fused Softmax
```

因为这个例子能够真正开始理解：

> **为什么 Operator Fusion 能提高性能？**

---

## Day 5：性能分析

这是整个学习过程中非常重要的一天。

需要认识两个工具：

```text
Nsight Systems
Nsight Compute
```

简单理解：

### Nsight Systems

看的是：

```text
整个应用
```

例如：

```text
CPU Timeline
CUDA API
Kernel Launch
GPU Timeline
Memcpy
Synchronization
```

更像：

> **宏观性能分析。**

Nsight Systems 官方文档也建议针对 performance-critical region 聚焦 profiling，而不是毫无目的地收集整个应用的海量数据。

### Nsight Compute

看的是：

```text
某一个 CUDA Kernel
```

例如：

```text
Memory Throughput
SM Throughput
Occupancy
Warp Stall
Cache
Register
```

更像：

> **Kernel 显微镜。**

NVIDIA 对 Nsight Compute 的定位就是交互式 CUDA kernel profiler，并提供详细的性能指标。

---

## Day 6：torch.compile + AI Compiler

这一天不要上来学 LLVM。

先从自己熟悉的 PyTorch 开始。

写：

```python
model = torch.compile(model)
```

然后思考：

```text
为什么一行代码可以让模型变快？
```

继续研究：

```text
TorchDynamo
 ↓
FX Graph
 ↓
TorchInductor
 ↓
Triton
 ↓
GPU
```

重点理解几个编译器概念：

```text
Graph
IR
Pass
Pattern
Rewrite
Fusion
Lowering
Codegen
```

这一天结束以后应该可以解释：

> Graph Optimization 和 Kernel Optimization 到底有什么区别？

---

## Day 7：完成第一个 Mini Project

项目：

```text
LLM Inference Optimization Lab
```

目录可以设计成：

```text
llm-inference-optimization/
│
├── baseline/
│   └── transformers_inference.py
│
├── vllm/
│   └── benchmark.py
│
├── triton/
│   └── fused_softmax.py
│
├── profiling/
│   ├── nsight_systems/
│   └── nsight_compute/
│
├── results/
│   └── benchmark.md
│
└── README.md
```

实验至少记录：

```text
Input Length
Output Length
Concurrency
Batch Size
GPU Memory
Throughput
Latency
TTFT
TPOT / ITL
```

其中：

```text
TTFT = Time To First Token
```

表示：

> 用户等待第一个 Token 的时间。

而：

```text
TPOT = Time Per Output Token
```

或者常见的：

```text
ITL = Inter-Token Latency
```

更能体现 Decode 阶段的体验。

不要只写：

```text
优化后快了 20%。
```

而应该记录：

```text
Hardware:
GPU:

Model:

Input tokens:
Output tokens:

Concurrency:

Before:
TTFT:
TPOT:
Throughput:

After:
TTFT:
TPOT:
Throughput:

Profiler Evidence:

Analysis:
```

TensorRT-LLM 官方同样提供专门的 benchmark 工具，并区分 maximum throughput 等场景，这说明“如何正确 Benchmark”本身就是推理优化工程的一部分。

---

# 7. 如果我有一个月：开始真正进入 AI Infra

如果有一个月，我不会选择：

```text
同时深入 CUDA + vLLM + LLVM + TVM + TensorRT
```

因为最终大概率是哪一个都没有真正掌握。

我的安排会是：

```text
第 1 周：LLM Inference
第 2 周：CUDA + Triton + Profiling
第 3 周：AI Compiler
第 4 周：Distributed Training + 综合项目
```

---

# 8. 第一周：吃透 LLM Inference

这一周解决：

> **一个线上 LLM Serving Engine 到底是怎么工作的？**

学习：

```text
Transformer Inference
Prefill / Decode
KV Cache
Continuous Batching
PagedAttention
Prefix Caching
Chunked Prefill
Quantization
Speculative Decoding
Tensor Parallelism
Pipeline Parallelism
```

然后开始读：

```text
vLLM
```

源码。

但不要：

```text
git clone vllm
↓
打开源码
↓
从第一行开始看
```

更好的方式是带问题读源码。

比如：

```text
一个 HTTP Request 是怎么进入 Engine 的？
```

继续追：

```text
Request
 ↓
Scheduler
 ↓
KV Cache Allocation
 ↓
Model Runner
 ↓
Attention Backend
 ↓
CUDA Kernel
```

TensorRT-LLM 的架构文档同样把 Model Engine、Decoder 和 Scheduler 等部分明确划分出来，其中 Scheduler 需要负责资源分配以及请求执行决策。

这说明 Scheduler、KV Cache、Executor 并不是只属于某个框架的“实现细节”，而是现代 LLM Serving System 的核心抽象。

### 第一周建议博客

```text
01. 大模型推理中的 Prefill 和 Decode 到底是什么？
02. 一文搞懂 KV Cache
03. PagedAttention 为什么可以提高 LLM Serving 吞吐？
04. Continuous Batching 到底解决了什么问题？
05. 从一次请求看懂 vLLM 的执行流程
```

---

# 9. 第二周：CUDA + Triton + GPU Performance

这一周的目标是：

> **开始真正建立 GPU 性能直觉。**

需要系统学习：

```text
Thread
Warp
Block
Grid
SM
```

然后学习 Memory Hierarchy：

```text
Register
   ↓
Shared Memory
   ↓
L1 Cache
   ↓
L2 Cache
   ↓
Global Memory / HBM
```

核心问题是：

> 为什么 GPU 计算这么快，但一个 Kernel 仍然可能很慢？

原因往往并不是：

```text
计算不够快
```

而是：

```text
数据没喂进去。
```

所以需要建立：

```text
Arithmetic Intensity
```

的概念。

简单理解：

```text
Arithmetic Intensity
=
计算量 / 数据搬运量
```

Nsight Compute 官方的 Roofline 分析也是把 Arithmetic Intensity 与 GPU 的 Peak Compute、Memory Bandwidth 放到一起判断 Kernel 更偏向 Memory Bound 还是 Compute Bound。

这会成为之后分析所有 Kernel 的基本工具。

---

## 这一周建议写三个 Kernel

### Kernel 1：Vector Add

学习：

```text
Thread Mapping
Memory Access
```

### Kernel 2：Softmax

学习：

```text
Reduction
Shared Memory
Fusion
Numerical Stability
```

### Kernel 3：Matrix Multiplication

学习：

```text
Tiling
Shared Memory
Data Reuse
Tensor Core
```

矩阵乘是整个 AI Infra 的核心。

因为大量 Transformer 计算最终都会落到：

```text
GEMM
```

Triton 官方 Matrix Multiplication 教程专门讨论了 block-level matmul、多维指针计算、L2 cache locality 和 autotuning。

更深入以后再进入：

```text
CUTLASS
CuTe
```

CUTLASS 本质上提供了一套高性能 CUDA 线性代数抽象，可以用于构建和定制 GEMM 等高性能 kernel。当前 CUTLASS 项目也在进一步提供 Python-native DSL 来降低高性能 kernel 开发门槛。

---

# 10. 第三周：AI Compiler

这是整个岗位里门槛比较高、同时也很值得长期积累的一部分。

但我不会直接从 LLVM 开始。

我的顺序是：

```text
torch.compile
    ↓
FX Graph
    ↓
TorchInductor
    ↓
Triton
    ↓
MLIR
    ↓
LLVM
```

先理解：

> **深度学习编译器为什么存在？**

假设 PyTorch 写：

```python
y = relu(x + bias)
```

从用户视角就是两行数学运算。

编译器看到的可能是：

```text
aten.add
   ↓
aten.relu
```

然后进行：

```text
Pattern Match
      ↓
Graph Rewrite
      ↓
Fusion
```

最终：

```text
Fused Kernel
```

所以 Compiler Optimization 本质上是在回答：

> **能不能自动把用户写的高层程序转换成更适合硬件执行的程序？**

---

## 编译器至少需要掌握的 7 个概念

```text
1. IR
2. SSA
3. Pass
4. Pattern Matching
5. Rewrite
6. Lowering
7. Codegen
```

### IR

Intermediate Representation：

```text
中间表示
```

程序不会直接：

```text
Python → GPU
```

而可能经过：

```text
Python
 ↓
High Level IR
 ↓
Low Level IR
 ↓
LLVM IR
 ↓
PTX
 ↓
SASS
```

---

### Pass

可以简单理解为：

> **对 IR 做一次分析或者转换。**

例如：

```text
Constant Folding
Dead Code Elimination
Operator Fusion
Layout Transformation
```

MLIR 官方把 Pass 作为 transformation / optimization 的基本基础设施，并提供 PassManager 组织优化 pipeline。

---

### Dialect

这是 MLIR 很重要的思想。

不同抽象层可以拥有不同的：

```text
Operation
Type
Attribute
```

例如：

```text
Tensor Dialect
GPU Dialect
LLVM Dialect
```

MLIR 官方文档中，Dialect 就是扩展 MLIR 生态、定义新的 Operation / Attribute / Type 的核心机制。

因此可以：

```text
High Level ML Operation
         ↓
        Pass
         ↓
GPU Dialect
         ↓
        Pass
         ↓
LLVM Dialect
```

不断 Lowering。

这就是：

> **多层 IR。**

---

# 11. 第四周：补齐训练优化与分布式系统

这个 JD 不只是：

```text
LLM Inference
```

还明确包含：

```text
Training Optimization
```

所以训练侧不能完全忽略。

至少需要理解：

```text
Data Parallel
Tensor Parallel
Pipeline Parallel
Context Parallel
Expert Parallel
FSDP
ZeRO
NCCL
```

---

## Data Parallel

最直观：

```text
GPU 0 → Model Copy → Batch 0
GPU 1 → Model Copy → Batch 1
GPU 2 → Model Copy → Batch 2
GPU 3 → Model Copy → Batch 3
```

然后同步：

```text
Gradient
```

这里就会涉及：

```text
AllReduce
```

NCCL 提供的正是多 GPU 场景中的 AllReduce、AllGather、ReduceScatter、AllToAll 等通信原语。

---

## ZeRO / FSDP

普通 Data Parallel 每张卡都复制：

```text
Parameters
Gradients
Optimizer States
```

会产生大量显存冗余。

ZeRO 的核心思路就是：

```text
Shard
```

ZeRO 论文从消除 data/model parallel training 中的内存冗余出发；PyTorch FSDP 则直接提供参数分片的数据并行机制。

---

## Tensor Parallel

把一个大矩阵拆开：

```text
Linear Layer
```

例如：

```text
W
```

拆给不同 GPU。

经典 Megatron-LM 工作系统化探索了大规模 Transformer 中的模型并行；后续工作进一步组合了 Tensor、Pipeline 和 Data Parallelism。

---

## Pipeline Parallel

把模型：

```text
Layer 0
Layer 1
Layer 2
...
Layer 80
```

拆到不同 GPU：

```text
GPU 0 → Layer 0~19
GPU 1 → Layer 20~39
GPU 2 → Layer 40~59
GPU 3 → Layer 60~79
```

需要进一步处理：

```text
Pipeline Bubble
Micro Batch
Scheduling
```

---

## Context / Expert Parallel

随着：

```text
Long Context
MoE
```

越来越常见，还需要理解：

```text
Context Parallel
Expert Parallel
```

目前 Megatron Core 官方把：

```text
DP
TP
PP
CP
EP
FSDP
```

都作为可以组合的大模型训练并行策略。

这里现阶段不需要做到源码级。

一个月阶段的目标是：

> **能够解释每种 Parallelism 到底切了哪个维度，以及为什么需要通信。**

---

# 12. 一个月以后，我希望自己的能力变成什么样？

不是：

```text
我学过 CUDA。
我学过 vLLM。
我学过 MLIR。
```

而是面对一个问题：

> 模型推理为什么慢？

我能够开始按照下面的方式思考：

```text
                性能下降
                   │
          ┌────────┴────────┐
          │                 │
         CPU               GPU
          │                 │
    Scheduling?       ┌─────┴─────┐
    Python?           │           │
    Tokenizer?     Compute      Memory
                     │            │
                    GEMM      HBM Bandwidth
                 Tensor Core     Cache
                                KV Cache
                                   │
                             Memory Access
```

进一步：

```text
GPU Utilization 低？
      ↓
是不是请求太少？
是不是 Scheduler 有问题？
是不是 Kernel Launch Gap 很大？

GPU Utilization 高但性能仍然低？
      ↓
看 Kernel

Kernel 慢？
      ↓
Nsight Compute

Memory Bound？
      ↓
减少 Memory Traffic
Fusion
Tiling
Cache
Data Reuse

Compute Bound？
      ↓
Tensor Core
Precision
Algorithm
Instruction Throughput
```

这才是真正的：

> **Performance Engineering 思维。**

---

# 13. 3～6 个月长期学习路线

一个月只能建立基本能力。

真正形成求职竞争力，需要继续往下沉。

我的长期路线准备按照：

```text
Month 1
LLM Inference + CUDA 入门
        ↓
Month 2
CUDA / Triton / Profiling
        ↓
Month 3
vLLM Source Code
        ↓
Month 4
AI Compiler
        ↓
Month 5
Distributed Training
        ↓
Month 6
Open Source Contribution
```

推进。

---

# 14. Month 2：成为一个“会分析 Kernel”的人

这一阶段重点：

```text
CUDA
Triton
Nsight
```

深入：

```text
Memory Coalescing
Shared Memory
Bank Conflict
Occupancy
Register Pressure
Warp Divergence
Async Copy
Tensor Core
CUDA Stream
CUDA Graph
```

目标不是：

> 写 100 个 CUDA demo。

而是：

> 同一个 Kernel，我能从 Version 1 优化到 Version 5，并解释每一步为什么更快。

例如：

```text
Naive Matmul
      ↓
Tiled Matmul
      ↓
Shared Memory
      ↓
Register Blocking
      ↓
Vectorized Load
      ↓
Tensor Core
```

每一个版本必须记录：

```text
Latency
Bandwidth
FLOPS
Occupancy
Memory Throughput
```

这样最终博客不是：

```text
CUDA 学习笔记
```

而是：

```text
《从 200 GFLOPS 到 10 TFLOPS：一步一步优化 CUDA Matrix Multiplication》
```

这种文章对于求职明显更有价值。

---

# 15. Month 3：真正读 vLLM

前面已经会：

```text
LLM Inference
CUDA
Profiling
```

这时候再读 vLLM 源码会舒服很多。

重点追踪：

```text
Request
 ↓
Scheduler
 ↓
KV Cache Manager
 ↓
Model Runner
 ↓
Attention
 ↓
Kernel
```

阅读源码的时候，我准备坚持一个原则：

> **不按照文件读源码，按照问题读源码。**

例如：

```text
问题 1：
一个 Request 怎么进入 Scheduler？

问题 2：
Scheduler 怎么决定这一步跑谁？

问题 3：
KV Cache block 怎么分配？

问题 4：
Prefill 与 Decode 如何被调度？

问题 5：
Attention Backend 怎么选择？

问题 6：
Tensor Parallel 在哪里发生？

问题 7：
CUDA Graph 在哪里使用？
```

然后形成文章：

```text
《vLLM 源码阅读（一）：一次 Request 的生命周期》
《vLLM 源码阅读（二）：Scheduler》
《vLLM 源码阅读（三）：KV Cache Manager》
《vLLM 源码阅读（四）：Model Runner》
```

这时候博客就开始真正形成技术深度。

---

# 16. Month 4：真正进入 Compiler

这个阶段开始：

```text
MLIR
LLVM
TVM
XLA
```

但仍然不要四个同时学。

建议先：

```text
MLIR
```

因为它更贴近现代 AI Compiler 中：

```text
Multi-Level IR
Dialect
Pass
Lowering
```

这些思想。

然后了解：

```text
LLVM
```

主要理解：

```text
LLVM IR
Basic Block
SSA
Pass
Backend
Code Generation
```

再结合：

```text
TVM / XLA
```

理解真正的 ML Compiler Pipeline。

例如 XLA:GPU 的官方架构本身就是从 StableHLO 等高层表示逐步产生 GPU code，并包含 LLVM/PTX 与 TritonIR 等代码生成路径。

StableHLO 则承担着 ML Framework 与 ML Compiler 之间的高层可移植表示角色。

最终希望自己能看懂：

```text
Frontend
   ↓
Graph IR
   ↓
High-Level Optimization
   ↓
Tensor IR
   ↓
Loop / Tile
   ↓
GPU Mapping
   ↓
LLVM IR
   ↓
PTX
```

而不是只会说：

> MLIR 是编译器。

---

# 17. Month 5：Distributed Training

这个阶段重点学习：

```text
NCCL
DDP
FSDP
ZeRO
TP
PP
CP
EP
```

但重点不是背定义。

而是分析：

```text
Compute
vs
Communication
```

例如训练：

```text
Backward
   ↓
Gradient Ready
   ↓
AllReduce
```

如果：

```text
Communication
```

和：

```text
Backward Computation
```

可以 overlap：

```text
Compute ────────────────────────
      Communication ────────────
```

整体时间就可能下降。

因此以后真正重要的问题会逐渐变成：

```text
通信量是多少？

什么时候发生通信？

走 NVLink 还是 PCIe / 网络？

能不能 overlap？

哪个 collective 最贵？

TP degree 应该设多大？
```

这已经从：

```text
Deep Learning
```

进入：

```text
Distributed Systems + HPC
```

的范畴。

---

# 18. Month 6：开始尝试 Open Source Contribution

这个岗位明确把：

```text
二次开发能力
开源社区贡献经历
```

作为加分项。

因此我希望最终不只是：

```text
Fork vLLM
```

而是尝试真正产生：

```text
Issue
PR
Benchmark
Bug Fix
Documentation
Test
Performance Optimization
```

第一次 Contribution 没必要挑战：

```text
重新设计 Scheduler
```

可以从：

```text
Documentation
Test Case
Small Bug
Error Message
Benchmark Script
```

开始。

然后逐渐进入：

```text
Kernel
Optimization
Feature
```

对于求职来说：

```text
Merged PR
```

最大的价值不只是 GitHub 页面上多了一条绿色记录。

而是面试的时候可以说：

```text
我在阅读 XXX 项目源码时发现了 XXX 问题。

我使用 XXX 方法定位到了问题。

Profiler 显示主要 Bottleneck 是 XXX。

因此我修改了 XXX。

最终 XXX 指标从 A 变成 B。

后来这个修改提交到了上游。
```

这已经是一套非常完整的工程故事。

---

# 19. 我认为最值得做的三个项目

如果时间有限，我不会做十个 Demo。

我更希望认真完成三个项目。

---

## Project 1：LLM Serving Benchmark

项目：

```text
vLLM Inference Benchmark
```

研究：

```text
Input Length
Output Length
Concurrency
Batch Size
```

对于：

```text
TTFT
TPOT
Throughput
GPU Memory
```

的影响。

最后画出：

```text
Concurrency
     ↓
Throughput

Input Length
     ↓
TTFT

Batch Size
     ↓
TPOT
```

然后解释：

> 为什么出现这样的趋势？

---

## Project 2：GPU Kernel Optimization

实现：

```text
RMSNorm / Softmax / GEMM
```

三个版本：

```text
PyTorch
Triton
CUDA
```

比较：

```text
Latency
Bandwidth
Speedup
```

然后：

```text
Nsight Compute
```

分析。

最终标题甚至可以写成：

```text
《RMSNorm 到底慢在哪里？从 PyTorch 到 Triton 再到 CUDA》
```

---

## Project 3：Mini AI Compiler

实现一个非常简单的：

```text
Graph Optimizer
```

假设 IR：

```text
MatMul
  ↓
Add
  ↓
ReLU
```

实现 Pattern：

```text
MatMul + Add
```

转换成：

```text
FusedLinear
```

即使这个 Compiler 非常简单，也能够真正理解：

```text
IR
Pass
Pattern
Rewrite
Fusion
```

比看几十篇 Compiler PPT 有效得多。

---

# 20. 我的 GitHub 应该怎么组织？

如果最终目标还有：

> 建立自己的技术影响力

那么我不希望 GitHub 只是：

```text
一堆 LeetCode
一堆课程作业
一堆 notebook
```

更希望变成一个完整的：

```text
AI Infra Knowledge Base
```

例如：

```text
AI-Infra-Notes/
│
├── README.md
│
├── roadmap/
│   └── ai-infra-roadmap.md
│
├── 01-llm-inference/
│   ├── prefill-and-decode.md
│   ├── kv-cache.md
│   ├── paged-attention.md
│   └── continuous-batching.md
│
├── 02-vllm/
│   ├── architecture.md
│   ├── scheduler.md
│   ├── kv-cache-manager.md
│   └── model-runner.md
│
├── 03-cuda/
│   ├── gpu-architecture.md
│   ├── memory-hierarchy.md
│   ├── shared-memory.md
│   └── cuda-optimization.md
│
├── 04-triton/
│   ├── vector-add.md
│   ├── softmax.md
│   └── matmul.md
│
├── 05-performance/
│   ├── nsight-systems.md
│   ├── nsight-compute.md
│   └── roofline-model.md
│
├── 06-ai-compiler/
│   ├── torch-compile.md
│   ├── ir.md
│   ├── mlir.md
│   └── llvm.md
│
├── 07-distributed-training/
│   ├── ddp.md
│   ├── fsdp.md
│   ├── tensor-parallel.md
│   └── pipeline-parallel.md
│
└── projects/
    ├── llm-serving-benchmark/
    ├── triton-kernels/
    └── mini-ai-compiler/
```

这样面试官点进 GitHub 后看到的是：

```text
一个持续围绕 AI Infra 深入的人
```

而不是：

```text
一个最近临时看了几个 CUDA 教程的人。
```

---

# 21. 博客到底应该怎么写？

我希望以后每一篇性能优化博客都遵循一个模板。

## 1. Problem

首先回答：

> 我们遇到了什么问题？

---

## 2. Baseline

最朴素的方法是什么？

例如：

```python
torch.softmax(x)
```

---

## 3. Bottleneck

为什么它可能慢？

```text
Memory?
Compute?
Launch Overhead?
Synchronization?
```

---

## 4. Principle

背后的计算机原理是什么？

例如：

```text
Memory Hierarchy
Arithmetic Intensity
Tiling
```

---

## 5. Optimization

做了什么修改？

---

## 6. Benchmark

拿数据说话：

```text
Before
After
```

---

## 7. Profiling

拿：

```text
Nsight
```

证明：

> 为什么变快了？

---

## 8. Trade-off

讨论：

```text
什么时候有效？
什么时候无效？
有什么代价？
```

---

## 9. Source Code

提供完整代码。

---

## 10. References

给出：

```text
Paper
Official Documentation
Source Code
```

这样博客才不只是：

> 学习笔记。

而会逐渐变成：

> **技术研究记录。**

---

# 22. 什么叫“真正会 vLLM”？

我觉得不能因为：

```bash
pip install vllm
```

然后：

```python
LLM(...)
```

就写：

```text
熟悉 vLLM
```

至少应该可以解释：

```text
PagedAttention 为什么存在？

Continuous Batching 为什么有效？

KV Cache 怎么管理？

Scheduler 在调度什么？

Prefill / Decode 有什么差异？

Concurrency 为什么影响吞吐？

Tensor Parallel 怎么工作？

Prefix Cache 的收益来自哪里？

vLLM 的性能瓶颈怎么定位？
```

再进一步：

```text
能够读 Scheduler 源码
能够修改一个模块
能够 Benchmark
能够 Profile
```

这时候再写：

```text
熟悉 vLLM
```

会更有底气。

---

# 23. 什么叫“真正会 CUDA”？

同理：

会写：

```cpp
__global__ void add(...)
```

只能算：

> CUDA Hello World。

真正进入性能工程至少需要理解：

```text
Thread / Warp / Block
Memory Hierarchy
Coalescing
Shared Memory
Bank Conflict
Occupancy
Register Pressure
Warp Divergence
Streams
Tensor Core
```

并且可以：

```text
Naive Kernel
      ↓
Profile
      ↓
发现 Bottleneck
      ↓
Optimization
      ↓
Profile Again
```

因此 CUDA 的学习单位不应该是：

```text
“今天学了 Shared Memory。”
```

而应该是：

```text
“今天通过 Shared Memory 把一个 Kernel 从 X us 优化到了 Y us，
并且通过 Nsight Compute 证明 DRAM traffic 出现了变化。”
```

---

# 24. 什么叫“懂 AI Compiler”？

不是：

> 我知道 LLVM。

而是能解释：

```text
为什么需要 IR？

为什么需要多层 IR？

什么是 SSA？

什么是 Pass？

什么是 Graph Rewrite？

什么是 Pattern Matching？

什么叫 Lowering？

Fusion 在哪一层做？

Codegen 到底生成什么？
```

然后能够自己：

```text
写一个简单 Pass
```

或者：

```text
实现一次 Pattern Rewrite
```

这才算真正跨进 Compiler 的门。

MLIR 的 Dialect + Pass + Conversion，以及 LLVM 的 IR + Pass，是非常适合逐渐建立这种认知的两套体系。

---

# 25. 有哪些坑我暂时不应该踩？

在这条学习路线里，我认为最危险的事情就是：

> **什么都想学。**

尤其不要一开始就：

```text
CUDA
ROCm
LLVM
MLIR
TVM
XLA
TensorRT
vLLM
SGLang
CUTLASS
DeepSpeed
Megatron
```

一起开。

结果通常是：

```text
每个项目都知道名字
↓
每个 README 都看过
↓
面试官追问两层
↓
不会了
```

我更倾向：

```text
vLLM
  +
CUDA
  +
Triton
  +
Nsight
```

作为第一条主线。

然后延伸：

```text
torch.compile
     ↓
MLIR / LLVM
```

再补：

```text
Distributed Training
```

最终形成：

```text
                    AI Infra
                       │
        ┌──────────────┴──────────────┐
        │                             │
     System                       Compiler
        │                             │
      vLLM                         MLIR
        │                             │
    Scheduler                       Pass
    KV Cache                       Codegen
        │                             │
        └─────────── GPU ─────────────┘
                     │
                CUDA / Triton
                     │
                   Nsight
```

这条路线会更连贯。

---

# 26. 最终学习路线总结

如果时间被压缩到：

## 只有 1 天

目标：

> **建立地图。**

掌握：

```text
Transformer Inference
Prefill / Decode
KV Cache
PagedAttention
Continuous Batching
CUDA 基本模型
Compiler Pipeline
```

不要追求代码深度。

---

## 只有 1 周

目标：

> **跑通完整流程。**

做到：

```text
部署一个 LLM
使用 vLLM
做 Benchmark
写简单 CUDA
写 Triton Kernel
使用 Nsight
理解 torch.compile
```

最后形成：

```text
一个 LLM Inference Optimization Mini Project
```

---

## 有 1 个月

目标：

> **形成真正的 AI Infra 入门能力。**

完成：

```text
Week 1  LLM Inference
Week 2  CUDA + Triton + Profiling
Week 3  AI Compiler
Week 4  Distributed Training + Project
```

最终至少有：

```text
1 个完整项目
5~10 篇技术博客
1 套 Benchmark
1 套 Profiling 数据
```

---

## 有 3～6 个月

目标：

> **形成求职壁垒。**

继续深入：

```text
vLLM Source Code
CUDA Optimization
Triton
CUTLASS
Nsight
torch.compile
MLIR
LLVM
NCCL
Megatron
```

然后：

```text
Open Source Contribution
```

---

# 27. 我真正想成为的是什么样的人？

写到这里，我发现自己真正需要追求的并不是：

```text
“我会多少框架？”
```

而应该是：

> **当一个 AI workload 很慢的时候，我有没有能力一步一步找到它为什么慢。**

可能是：

```text
Scheduler
```

可能是：

```text
KV Cache
```

可能是：

```text
GPU Utilization
```

可能是：

```text
Memory Bandwidth
```

可能是：

```text
Kernel
```

可能是：

```text
Communication
```

也可能是：

```text
Compiler 没有做 Fusion
```

真正的 AI Infra / Performance Engineer 应该能够不断沿着：

```text
Application
    ↓
Framework
    ↓
Graph
    ↓
Compiler
    ↓
Runtime
    ↓
Kernel
    ↓
Hardware
```

往下定位。

然后：

```text
Measure
   ↓
Understand
   ↓
Optimize
   ↓
Measure Again
```

这也是我接下来学习这套技术栈最核心的方法论。

> **不要为了学习 CUDA 而学习 CUDA。**
>
> **不要为了学习编译器而学习编译器。**
>
> **去找一个真实的性能问题，然后一路追到底。**

如果最终能够做到这一点，我认为自己才算真正迈进了：

```text
AI Infra
```

的大门。

---

# 参考资料

下面是这套学习路线主要参考的论文、官方文档以及开源项目。阅读顺序不需要严格按照编号，可以随着学习进度逐步阅读。

## LLM Inference / Serving

1. **Woosuk Kwon et al. — Efficient Memory Management for Large Language Model Serving with PagedAttention**, SOSP 2023，arXiv:2309.06180。vLLM / PagedAttention 的经典论文，建议作为 LLM Serving 必读论文。

2. **vLLM Project — `vllm-project/vllm`**。官方开源项目，当前实现涉及 PagedAttention、Continuous Batching、Chunked Prefill、Prefix Caching、CUDA/HIP Graph、量化、优化 Kernel、`torch.compile` 以及多种分布式推理能力。

3. **NVIDIA TensorRT-LLM — Architecture Overview**。适合学习现代 LLM Runtime 中 Model Engine、Decoder、Scheduler、KV Cache 等组件如何组织。

4. **TensorRT-LLM Benchmarking / Performance Guide**。适合学习推理系统 Benchmark 的设计方法以及 throughput workload。

5. **MLC-LLM Documentation**。MLC-LLM 将机器学习编译与跨平台 LLM 部署结合，适合理解 TVM / compiler-based deployment 路线。

---

## Attention / Kernel Optimization

6. **Tri Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, arXiv:2205.14135。学习 IO-aware algorithm、GPU memory hierarchy 与 Attention 优化的重要论文。

7. **Triton Official Tutorials**。建议按照 Vector Addition → Fused Softmax → Matrix Multiplication → LayerNorm → Fused Attention 的顺序实践。

8. **Triton — Fused Softmax Tutorial**。理解 Operator Fusion、memory traffic 与 bandwidth-bound kernel 的优秀入门材料。

9. **Triton — Matrix Multiplication Tutorial**。涉及 block matmul、cache locality 和 autotuning，是学习高性能 GEMM 很好的过渡材料。

10. **NVIDIA CUTLASS — `NVIDIA/cutlass`**。高性能 CUDA GEMM 与 Tensor Core 编程的重要开源项目，适合 CUDA/Triton 之后继续深入。

---

## CUDA / GPU Performance

11. **NVIDIA CUDA Programming Guide**。CUDA 学习最重要的一手资料，涵盖 Programming Model、GPU Programming、异步执行、多 GPU 等内容。

12. **NVIDIA GPU Performance Background User's Guide**。介绍 GPU 架构、执行模型、Arithmetic Intensity 以及常见性能限制，适合建立 GPU Performance Engineering 直觉。

13. **NVIDIA Nsight Systems User Guide**。学习系统级 Timeline、CUDA API、CPU/GPU interaction 与应用级性能分析。

14. **NVIDIA Nsight Compute Documentation / Profiling Guide**。学习 Kernel Profiling、Occupancy、Memory Workload、Roofline 等 GPU Kernel 性能指标。

---

## PyTorch Compiler / AI Compiler

15. **PyTorch `torch.compile` Documentation**。学习 PyTorch 编译入口以及 graph capture / compilation 的基本行为。

16. **PyTorch 2 / TorchDynamo / TorchInductor 相关官方资料**。用于理解 PyTorch Program → TorchDynamo → FX Graph → TorchInductor → Triton/C++ 的完整编译链路。

17. **MLIR Language Reference / Dialect Documentation**。学习 Operation、Type、Attribute、Dialect、多层 IR 等 MLIR 核心思想。

18. **MLIR Pass Infrastructure**。学习 Pass、PassManager、Analysis、Transformation Pipeline。

19. **LLVM — Writing an LLVM Pass**。理解传统编译器优化中 Pass 的基本实现方式。

20. **Apache TVM Documentation**。学习深度学习编译器如何进行模型导入、IR 表示、优化和代码生成。

21. **OpenXLA — XLA:GPU Architecture Overview**。理解 StableHLO、XLA GPU Pipeline、LLVM/PTX、TritonIR 等组件之间的关系。

22. **OpenXLA — StableHLO Specification**。了解 ML Framework 与 ML Compiler 之间高层 IR / portability layer 的设计。

---

## Distributed Training

23. **Mohammad Shoeybi et al. — Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, arXiv:1909.08053。Tensor Model Parallelism 的经典工作。

24. **Deepak Narayanan et al. — Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**, arXiv:2104.04473。系统讨论 Tensor、Pipeline、Data Parallelism 的组合与大规模训练。

25. **Samyam Rajbhandari et al. — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, arXiv:1910.02054。理解参数、梯度和 Optimizer State 分片的重要论文。

26. **NVIDIA Megatron-LM / Megatron Core**。目前官方实现包含 TP、PP、DP、CP、EP、FSDP、Mixed Precision 等大规模 Transformer 训练能力。

27. **NVIDIA NCCL Documentation**。学习 AllReduce、AllGather、ReduceScatter、AllToAll 等 GPU Collective Communication 原语。

28. **PyTorch FullyShardedDataParallel Documentation**。用于理解 PyTorch FSDP 参数分片及其与 ZeRO 思想之间的联系。

---

> **最后给未来的自己留一句话：**
>
> 学 AI Infra 最容易陷入的误区，是收藏越来越多的框架、论文和课程，却从来没有真正定位过一次性能问题。
>
> 所以接下来的学习，我希望始终坚持：
>
> ```text
> Run it.
> Measure it.
> Profile it.
> Understand it.
> Optimize it.
> Measure it again.
> ```
>
> **性能优化最终不是背知识点，而是建立一套发现问题、解释问题、解决问题的能力。**
