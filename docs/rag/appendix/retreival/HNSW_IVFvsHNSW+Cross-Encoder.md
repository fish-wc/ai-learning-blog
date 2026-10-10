---

## 2.4 HNSW（Hierarchical Navigable Small World）分层可导航小世界图

HNSW 是一种基于图结构的 ANN（Approximate Nearest Neighbor）索引。

与 IVF 不同，HNSW **不依赖 K-Means 聚类**，而是通过构建多层近邻图，实现高效的向量搜索。

HNSW 的核心思想：

> 使用上层稀疏图进行全局导航，快速定位目标区域；使用底层相对稠密的近邻图进行精细搜索。

其结构类似于跳表（Skip List）：

- 上层：节点数量少，负责长距离导航；
- 下层：节点数量多，负责局部精细搜索；
- Layer 0：包含全部向量节点，是最终近邻检索的主要执行层。

需要注意：

**HNSW 底层虽然比上层稠密，但并不是完全图（Complete Graph）。**

每个节点只与有限数量的邻居建立连接，从而控制索引的内存开销。

---

### 2.4.1 HNSW 分层结构

HNSW 由多个独立的 NSW（Navigable Small World）图组成。

假设索引包含：

```text
Layer 3        A
               |
Layer 2    A ------- B
           |         |
Layer 1    A --- B --- C --- D
           |     |     |     |
Layer 0    A-B-C-D-E-F-G-H-I-J
```

以上仅为概念示意，竖线表示同一节点在不同层的对应关系，并非显式存储的跨层图边。

#### 核心层级规则

**规则 1：Layer 0 包含全部节点**

所有插入 HNSW 的向量，都会出现在 Layer 0。

Layer 0 是整个索引中节点数量最多的一层。

**规则 2：高层节点是低层节点的子集**

如果节点存在于 Layer $l$，那么该节点必然存在于：

$$
0,1,\dots,l-1
$$

所有下层。

因此：

$$
V_l \subseteq V_{l-1}
$$

其中 $V_l$ 表示第 $l$ 层的节点集合。

**规则 3：节点最高层由随机采样决定**

HNSW 的上层节点并不是通过聚类选出的中心点。

当新节点插入时，算法会随机采样该节点的最高层：

$$
l_q =
\left\lfloor
-\ln(U)\cdot m_L
\right\rfloor
$$

其中：

- $U \sim \mathcal U(0,1)$：均匀随机变量；
- $m_L$：控制层级分布的参数；
- $l_q$：当前节点能够到达的最高层。

随着层数增加，节点数量通常呈指数下降趋势。

因此：

- 大多数节点只存在于 Layer 0；
- 少量节点进入 Layer 1；
- 更少节点进入 Layer 2；
- 极少数节点进入更高层。

高层节点承担全局导航的作用。

> **关键结论：**
>
> HNSW 的分层来自随机层级分配，而不是对底层节点再次执行聚类。
>
> 层与层之间不需要建立独立的跨层近邻边。搜索时通过同一个节点在不同层的对应关系完成下降。

---

### 2.4.2 HNSW 核心超参数

HNSW 的主要超参数如下：

| 参数 | 含义 | 主要影响 |
|---|---|---|
| $M$ | 非底层节点的最大邻居连接数参数 | 图连通性、内存开销 |
| $M_0$ | Layer 0 的最大邻居连接数 | 底层搜索能力 |
| $ef_{\text{construction}}$ | 构建索引时的候选搜索池大小 | 建图质量、构建耗时 |
| $ef_{\text{search}}$ | 查询阶段的候选搜索池大小 | Recall、查询延迟 |
| $m_L$ | 随机层级分布参数 | 图的层数和稀疏程度 |

常见配置中：

$$
M_0 \approx 2M
$$

实际取值取决于具体实现。

其中，最重要的两个参数是：

- $ef_{\text{construction}}$
- $ef_{\text{search}}$

前者主要决定建图质量，后者主要决定在线检索的 Recall 与延迟权衡。

---

### 2.4.3 HNSW 节点插入（建图）流程

HNSW 支持增量插入。

与 IVF 先执行全局 K-Means 聚类不同，HNSW 可以逐个添加向量，并更新图结构。

假设需要插入新向量：

$$
q
$$

#### Step 1：随机决定节点最高层

通过随机层级分布采样：

$$
l_q
$$

假设：

$$
l_q = 2
$$

则该节点需要出现在：

```text
Layer 2
Layer 1
Layer 0
```

但不会出现在 Layer 3 或更高层。

---

#### Step 2：从全局入口开始导航

HNSW 维护一个全局入口节点：

```text
entry point
```

入口节点通常位于当前索引的最高层。

新节点从最高层开始执行贪心搜索。

在当前层：

1. 计算新节点与入口节点的距离；
2. 遍历入口节点的邻居；
3. 如果发现距离新节点更近的邻居，则移动到该邻居；
4. 重复上述过程，直到无法继续找到更近节点。

得到当前层的局部最近节点后，将其作为下一层的搜索入口。

---

#### Step 3：高于新节点最高层时，仅执行搜索

假设：

```text
Global Max Layer = 4

New Node Max Layer = 2
```

那么：

```text
Layer 4 → Greedy Search
Layer 3 → Greedy Search
Layer 2 → Search + Connect
Layer 1 → Search + Connect
Layer 0 → Search + Connect
```

在 Layer 4 和 Layer 3：

**只搜索，不建立新边。**

因为新节点并不存在于这两层。

---

#### Step 4：在当前层搜索候选近邻

当下降至：

$$
l \leq l_q
$$

时，开始建立新节点的近邻连接。

算法使用候选池：

$$
ef_{\text{construction}}
$$

执行更充分的局部图搜索。

此时不是简单沿着单一路径贪心移动，而是维护候选集合，探索多个可能的近邻节点。

候选池越大，通常越有机会找到高质量邻居。

---

#### Step 5：选择邻居并建立双向连接

从候选集合中选择最多 $M$ 个邻居（Layer 0 可使用更大的连接数限制）。

需要注意：

**HNSW 并不一定简单选择距离最近的 M 个节点。**

常见实现会使用启发式邻居选择策略：

- 优先考虑距离新节点较近的候选；
- 同时保留具有不同方向或空间覆盖能力的邻居；
- 避免邻居过度集中在同一个局部区域。

这样可以减少图搜索陷入局部区域的概率。

建立连接：

```text
New Node <----> Neighbor A
         <----> Neighbor B
         <----> Neighbor C
```

连接一般是双向的。

如果某个已有节点的邻居数量超过上限，需要执行邻居裁剪，重新选择保留的连接。

---

#### Step 6：逐层下降

重复：

```text
Candidate Search
       |
       v
Neighbor Selection
       |
       v
Bidirectional Connection
       |
       v
Next Lower Layer
```

直到 Layer 0。

如果新节点的最高层超过原有全局最高层，则更新：

- 全局最高层；
- 全局入口节点。

至此完成插入。

---

### 2.4.4 HNSW 在线检索流程

假设输入：

$$
q = E_q(query)
$$

需要返回：

$$
TopK
$$

个近邻向量。

#### Step 1：从最高层入口节点开始

从全局最高层的 entry point 出发。

根据向量距离，沿着近邻图不断移动至距离 query 更近的节点。

例如：

```text
Entry
  |
  v
Node A
  |
  v
Node F
  |
  v
Node K
```

这种导航方式通常称为 Greedy Search。

---

#### Step 2：逐层下降

在当前层找到局部较优入口后，下降至下一层。

```text
Layer 3
   |
Greedy Search
   |
   v
Layer 2
   |
Greedy Search
   |
   v
Layer 1
   |
Greedy Search
   |
   v
Layer 0
```

越接近底层：

- 节点越多；
- 图的局部连接越丰富；
- 搜索结果越精细。

---

#### Step 3：Layer 0 执行候选集搜索

到达 Layer 0 后，使用：

$$
ef_{\text{search}}
$$

控制搜索候选集合的规模。

通常要求：

$$
ef_{\text{search}} \geq K
$$

其中 $K$ 为最终需要返回的近邻数量。

与高层单路径贪心搜索不同，底层会维护多个候选节点，并持续扩展可能的近邻。

---

#### Step 4：返回 Top-K

从搜索得到的候选集合中，根据向量距离进行排序：

$$
TopK =
\operatorname{NearestK}(q,C)
$$

其中：

$$
C
$$

表示候选集合。

最终返回距离 query 最近的 $K$ 个节点。

---

### 2.4.5 ef_search 的影响

$ef_{\text{search}}$ 是 HNSW 最重要的在线调参之一。

| ef_search | Recall | 查询延迟 | 搜索开销 |
|---|---|---|---|
| 较小 | 通常较低 | 较低 | 较小 |
| 较大 | 通常较高 | 较高 | 较大 |

例如：

```text
ef_search = 20
```

表示较小的搜索候选池。

而：

```text
ef_search = 200
```

允许算法探索更多候选节点，通常可以提升 Recall，但也会增加计算成本。

需要注意，实际 Recall 与延迟还受到以下因素影响：

- 向量维度；
- 数据分布；
- M 参数；
- 建图质量；
- 距离度量方式。

---

### 2.4.6 HNSW 的优势与局限

#### 优势

**1. 检索效率高**

通过高层全局导航与底层局部搜索，避免对全部向量进行暴力扫描。

在合适的数据分布和参数下，HNSW 通常具有接近对数级增长的经验搜索表现。

**2. 召回质量较高**

通过多层图导航与候选池搜索，可以实现较高 Recall。

**3. 支持增量插入**

新节点可以直接加入已有图结构，无需每次重新训练全局聚类中心。

**4. 不依赖 K-Means 聚类**

能够适应较复杂的向量空间分布。

#### 局限

**1. 内存开销较高**

除了存储向量，还需要存储大量邻接关系。

```text
Vector Storage
      +
Graph Edges
      =
HNSW Index Memory
```

**2. 构建成本较高**

插入每个节点时，都需要执行近邻搜索与图结构维护。

**3. 删除与更新需要额外管理**

虽然支持增量插入，但频繁删除和大规模更新可能影响图结构质量。

某些实现需要定期优化或重建索引。

---

## 2.5 IVF vs HNSW

IVF 与 HNSW 都属于 ANN 索引，但内部机制完全不同。

| 对比维度 | IVF | HNSW |
|---|---|---|
| 底层结构 | 聚类中心 + 倒排列表 | 多层近邻图 |
| 核心思想 | 先定位簇，再搜索簇内向量 | 沿图导航寻找近邻 |
| 构建过程 | K-Means 聚类 + 向量分桶 | 节点逐个插入 + 动态建边 |
| 在线检索 | 选择 nprobe 个簇并扫描候选 | 高层导航 + 底层候选搜索 |
| 主要参数 | nlist、nprobe | M、ef_construction、ef_search |
| 内存开销 | IVFFlat 通常较低 | 通常较高 |
| 数据分布依赖 | 较依赖聚类质量 | 不依赖聚类中心 |
| 增量插入 | 训练完成后可以添加向量 | 天然支持动态插入 |
| 数据分布漂移 | 可能需要重新训练聚类中心 | 可增量维护，但仍可能需要优化 |
| 典型实现 | FAISS IVFFlat、IVFPQ | FAISS HNSW、hnswlib、Qdrant |

需要注意：

**IVF 并非完全不支持增量更新。**

例如，IVFFlat 在训练好聚类中心后，可以继续添加新向量。

但当数据分布发生明显变化时，原先的聚类中心可能不再适合新数据，从而影响检索效果。

因此：

- 静态、大规模、内存敏感的场景，可以优先考虑 IVF；
- 动态插入频繁、对 Recall 和查询延迟要求较高的场景，可以重点评估 HNSW。

最终选型仍需结合向量规模、维度、更新频率、内存预算及实际性能测试。

---

# 3. Rerank 阶段：Cross-Encoder 原理

Retrieval 阶段已经利用 Bi-Encoder 与 ANN 索引，从海量文档中快速找出 Top-K 候选。

但是：

**向量相似，不一定代表事实相关。**

例如：

```text
Query:
苹果公司的创始人是谁？

Document A:
苹果公司由史蒂夫·乔布斯、
史蒂夫·沃兹尼亚克和罗纳德·韦恩共同创立。

Document B:
苹果公司发布了新款 iPhone。

Document C:
苹果是一种常见的水果。
```

Bi-Encoder 可以捕捉整体语义相似性，但可能无法充分区分：

- 实体一致性；
- 具体关系；
- 否定表达；
- 时间条件；
- 精细的事实匹配。

因此，需要更强的相关性模型对候选进行重新排序。

这就是：

**Cross-Encoder Reranker。**

---

## 3.1 Bi-Encoder 与 Cross-Encoder 的本质差异

### Bi-Encoder

Bi-Encoder 分别编码 Query 和 Document：

$$
\boldsymbol q = E_q(q)
$$

$$
\boldsymbol d = E_d(d)
$$

然后计算：

$$
s(q,d) =
\cos(\boldsymbol q,\boldsymbol d)
$$

其最大优势是：

**Document Embedding 可以离线预计算。**

在线阶段只需要：

```text
Query
  |
  v
Query Encoder
  |
  v
Query Embedding
  |
  v
ANN Search
  |
  v
Top-K Documents
```

但 Query 与 Document 在编码阶段互不交互。

---

### Cross-Encoder

Cross-Encoder 不再独立编码 Query 和 Document。

它将二者拼接为一个序列，送入同一个 Transformer：

```text
[CLS] Query Tokens [SEP] Document Tokens [SEP]
```

例如：

```text
[CLS]
苹果公司的创始人是谁？
[SEP]
苹果公司由乔布斯等人共同创立。
[SEP]
```

在 Transformer 的 Self-Attention 中：

**Query Token 可以直接关注 Document Token。**

Document Token 也可以关注 Query Token。

因此，模型可以学习跨文本的细粒度语义关系。

---

## 3.2 Cross-Encoder 的 Token 级交互

对于标准 Transformer Self-Attention：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

注意力计算为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

其中：

- $X$：Query 与 Document 拼接后的 Token 表征；
- $Q$：Query Matrix；
- $K$：Key Matrix；
- $V$：Value Matrix；
- $d_k$：Key 向量维度。

由于 Query 和 Document 被拼接在同一个序列中，Self-Attention 矩阵包含：

```text
                  Key Tokens

               Query    Document
             +-------------------+
Query        |  Q-Q   |   Q-D    |
Tokens       |        |          |
             +-------------------+
Document     |  D-Q   |   D-D    |
Tokens       |        |          |
             +-------------------+
```

其中：

- Q-Q：Query 内部 Token 交互；
- D-D：Document 内部 Token 交互；
- Q-D：Query 对 Document 的注意力；
- D-Q：Document 对 Query 的注意力。

**Q-D 和 D-Q 是 Cross-Encoder 能够进行跨文本细粒度匹配的重要原因。**

相比之下，Bi-Encoder 通常在两个独立的编码过程中完成特征提取。

因此，Cross-Encoder 可以更直接地学习：

- 哪些 Document Token 回答了 Query；
- 哪些实体与 Query 对齐；
- 是否存在否定关系；
- 是否满足时间或条件约束；
- Query 与 Document 是否具有真实的事实相关性。

---

## 3.3 Cross-Encoder 前向打分流程

典型的 BERT Cross-Encoder 使用以下输入格式：

```text
[CLS] q_t1 q_t2 ... [SEP] d_t1 d_t2 ... [SEP]
```

### Step 1：联合编码

将 Query 与 Document 拼接：

$$
X = [q;d]
$$

输入 Transformer：

$$
H = \operatorname{Transformer}(X)
$$

得到全部 Token 的上下文表示。

---

### Step 2：提取序列级特征

对于典型的 BERT 分类架构，可以使用：

$$
h_{\text{CLS}}
$$

作为整个输入序列的聚合特征。

该特征经过多层 Self-Attention，已经融合了 Query 与 Document 的交互信息。

---

### Step 3：相关性打分

通过一个线性分类头：

$$
z(q,d)
=
Wh_{\text{CLS}}+b
$$

其中：

- $W$：可学习的权重；
- $b$：偏置；
- $z(q,d)$：相关性 Logit。

对于采用二分类训练的模型，可以使用 Sigmoid：

$$
s(q,d)
=
\sigma(z(q,d))
$$

也就是：

$$
s(q,d)
=
\frac{1}{1+e^{-z(q,d)}}
$$

得到：

$$
s(q,d)\in(0,1)
$$

分数越大，通常表示模型判断 Query 与 Document 越相关。

需要注意：

> 并非所有 Cross-Encoder 都使用 Sigmoid。
>
> 一些 Reranker 直接输出未归一化的 Logit 或排序分数。排序任务通常更关注候选之间的相对得分，而非分数本身是否具有概率意义。

---

## 3.4 Cross-Encoder 为什么不能用于全库检索？

这是 Cross-Encoder 最重要的工程限制。

对于 Bi-Encoder：

$$
\boldsymbol d = E_d(d)
$$

可以提前计算并存储。

因此检索时只需要计算 Query Embedding：

$$
\boldsymbol q = E_q(q)
$$

然后搜索 ANN 索引。

但是，对于 Cross-Encoder：

$$
s(q,d)=\operatorname{CrossEncoder}(q,d)
$$

由于 Query 与 Document 需要联合编码，因此：

**每一组 Query-Document 都需要单独计算相关性分数。**

假设知识库有：

$$
N
$$

个 Document。

若使用 Cross-Encoder 全库扫描，需要执行：

$$
N
$$

次 Query-Document 联合推理。

这一成本通常难以接受。

因此实际系统一般采用：

```text
Knowledge Base
      |
      v
Bi-Encoder + ANN
      |
      v
Top-K Candidates
      |
      v
Cross-Encoder
      |
      v
Reranked Top-N
      |
      v
LLM
```

其中：

$$
N \ll K \ll |\mathcal D|
$$

这里：

- $|\mathcal D|$：知识库中的文档总量；
- $K$：召回阶段返回的候选数量；
- $N$：重排后最终保留的文档数量。

Cross-Encoder 只对 Top-K 候选进行精细打分。

---

## 3.5 Cross-Encoder 训练范式

Cross-Encoder 的训练目标通常围绕：

**让相关文档获得更高分数，让不相关文档获得更低分数。**

常见训练方式包括：

1. Pointwise：将 Query-Document 对视为独立样本；
2. Pairwise：优化两个候选文档之间的相对顺序。

---

### 3.5.1 Pointwise：BCE Loss

训练数据形式：

$$
(q,d,y)
$$

其中：

$$
y\in\{0,1\}
$$

表示：

- $y=1$：Document 与 Query 相关；
- $y=0$：Document 与 Query 不相关。

模型预测：

$$
s(q,d)=\sigma(z(q,d))
$$

使用 Binary Cross Entropy：

$$
\mathcal L_{\text{BCE}}
=
-\left[
y\log s(q,d)
+
(1-y)\log(1-s(q,d))
\right]
$$

训练目标：

- 正样本分数尽可能高；
- 负样本分数尽可能低。

这种方式适合学习单个 Query-Document 对的相关性。

---

### 3.5.2 Pairwise：排序损失

Pairwise 不再独立判断每个文档是否相关，而是比较两个候选文档。

训练样本：

$$
(q,d^+,d^-)
$$

其中：

- $d^+$：相关文档；
- $d^-$：不相关文档。

目标是：

$$
s(q,d^+) > s(q,d^-)
$$

一种常见 Pairwise Logistic Loss 为：

$$
\mathcal L_{\text{pair}}
=
-\log
\sigma
\left(
s(q,d^+)-s(q,d^-)
\right)
$$

其中 $s$ 可以是模型输出的原始排序分数。

该损失鼓励：

$$
s(q,d^+) - s(q,d^-)
$$

尽可能增大。

也就是说：

**训练重点从绝对相关性判断，转向相对排序质量。**

这种训练方式与 Rerank 的最终目标更加直接一致。

---

## 3.6 Cross-Encoder 的工程定位

Cross-Encoder 并不是替代 Bi-Encoder，而是作为第二阶段的精排模型。

两者分工如下：

| 对比维度 | Bi-Encoder | Cross-Encoder |
|---|---|---|
| 编码方式 | Query / Document 独立编码 | Query / Document 联合编码 |
| Token 级交互 | 编码阶段无跨文本交互 | 存在跨文本 Self-Attention |
| 文档离线预计算 | 支持 | 不能独立预计算最终配对表示 |
| ANN 检索 | 支持 | 不直接支持 |
| 单次配对计算成本 | 通常较低 | 通常较高 |
| 适合的候选规模 | 百万级及以上检索 | 少量候选精排 |
| 核心目标 | 高 Recall | 高排序精度 |
| 典型任务 | Retrieval | Rerank |

两阶段组合：

$$
\operatorname{TopK}
=
\operatorname{ANN}(E_q(q),\mathcal D)
$$

$$
\operatorname{TopN}
=
\operatorname{Rerank}
\left(
q,\operatorname{TopK}
\right)
$$

其中：

$$
N<K
$$

最终：

$$
\operatorname{Answer}
=
\operatorname{LLM}
\left(
q,\operatorname{TopN}
\right)
$$

---

## 3.7 Rerank 无法弥补 Recall 损失

这是理解两阶段 RAG Pipeline 的关键。

假设真正相关的文档是：

```text
Document A
```

但是 Retrieval 阶段只返回：

```text
Document B
Document C
Document D
```

那么：

**无论 Cross-Encoder 多么强大，都不可能重新找回 Document A。**

因为 Document A 根本没有进入候选集合。

从集合角度看：

$$
\mathcal D_{\text{rerank}}
\subseteq
\mathcal D_{\text{retrieval}}
$$

重排阶段只能在召回候选集合内部重新排序。

因此：

> **Retrieval 决定相关证据能否进入候选集合；Rerank 决定候选证据能否被准确排序。**

这也是为什么两阶段系统通常需要同时优化：

- Retrieval Recall@K；
- Rerank 的 MRR、NDCG 等排序指标；
- 最终进入 LLM 上下文的证据质量。

在智能体长期记忆系统中，Rerank 可以进一步减少语义相似但事实不匹配的记忆片段进入上下文的概率。

但它并不能独立保证事实正确，也无法完全消除 LLM 幻觉。

---