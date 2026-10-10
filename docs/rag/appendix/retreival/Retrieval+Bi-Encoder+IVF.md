# RAG 两阶段流水线：Retrieval + Rerank 原理、ANN 索引与 Cross-Encoder

> 摘要：RAG 采用 **召回-重排两阶段架构（Retrieval + Rerank）**，本质是在大规模知识库约束下，平衡检索吞吐量与相关性判别精度。
>
> 召回阶段基于 Bi-Encoder 稠密嵌入结合 ANN（Approximate Nearest Neighbor）近似检索，优先保证 Recall；重排阶段使用 Cross-Encoder 对少量候选进行细粒度相关性打分，修正向量相似带来的假阳性。
>
> 本文聚焦两阶段底层建模，展开 IVF、HNSW 索引与 Cross-Encoder 的内部机制，同时讨论该范式在智能体长期记忆模块中的延伸价值。

---

# 1. 两阶段 Pipeline 的设计动机

在面向智能体的长期记忆检索场景中，知识库规模可达到百万级甚至更大规模的文本片段。

如果直接使用强相关性模型对整个知识库进行逐对匹配：

\[
Query \times Document
\]

其推理成本不可接受。

因此现代 RAG 系统通常拆分为两个目标不同的阶段：

---

## 1.1 Retrieval（召回）

目标：

> 最大化 Recall

特点：

- 粗粒度检索
- 快速筛选候选集合
- 允许存在噪声

典型流程：

1. 使用 Bi-Encoder 将 query 和 document 映射到共享向量空间；
2. 利用 ANN 索引快速搜索近邻向量；
3. 返回 Top-K 候选文档。

核心约束：

> **不能遗漏真实相关文档。**

---

## 1.2 Rerank（重排）

目标：

> 最大化 Precision

特点：

- 细粒度相关性判断；
- 仅处理 Retrieval 阶段返回的小规模候选集合；
- 使用更强模型重新排序。

典型流程：

```
Query
  |
  v
Retriever
  |
Top-K candidates
  |
  v
Cross-Encoder Reranker
  |
Top-N documents
  |
  v
LLM Generation
```

---

两阶段之间存在一个核心矛盾：

- Bi-Encoder 可以提前计算 document embedding，因此适合百万级检索；
- 但由于 query 与 document 被独立编码，无法捕获 token 级交互；
- Cross-Encoder 可以实现 query-document 深度交互，但无法提前计算 document 表征。

因此：

| 模型 | 优势 | 劣势 |
|---|---|---|
| Bi-Encoder | 支持大规模 ANN 检索 | 语义交互能力弱 |
| Cross-Encoder | 相关性判断精准 | 无法扫描海量文档 |

---

# 2. Retrieval 阶段：Bi-Encoder 嵌入与 ANN 索引

## 2.1 Bi-Encoder 向量空间约束

Bi-Encoder 通常采用双塔结构：

$$
\boldsymbol q = E_q(q)
$$

$$
\boldsymbol d = E_d(d)
$$

其中：

- $E_q$：Query Encoder
- $E_d$：Document Encoder

相似度通常采用余弦距离：

$$
sim(\boldsymbol q,\boldsymbol d)
=
\cos(\boldsymbol q,\boldsymbol d)
=
\frac{\boldsymbol q^T\boldsymbol d}
{\|\boldsymbol q\|\|\boldsymbol d\|}
$$


> **关键结论：**
>
> 文档离线 embedding 与在线 query embedding 必须使用同一套模型权重。
>
> 本质原因：
>
> \[
> \boldsymbol q,\boldsymbol d
> \]
>
> 必须处于同一个语义向量空间。
>
> 如果两个向量由不同模型产生，则向量分布相互独立，距离度量无法表达真实语义关系。

理论上可以通过额外线性映射进行空间对齐，但工程成本高、泛化能力弱，因此工业系统很少采用。


---

## 2.2 ANN（Approximate Nearest Neighbor）

在离线阶段：

1. 对所有 document chunk 计算 embedding；
2. 将向量存储进入索引。


在线阶段：

1. 对 query 编码；
2. 使用 ANN 索引搜索近邻。


如果采用暴力搜索：

\[
O(N)
\]

当数据库达到百万、千万甚至十亿规模时无法接受。


ANN 的思想：

> 用少量 Recall 损失换取数量级的速度提升。


典型复杂度：

\[
O(\log N)
\]


常见 ANN 方法：

- IVF（Inverted File）
- HNSW（Hierarchical Navigable Small World）
- PQ（Product Quantization）

---

# 2.3 IVF（Inverted File）倒排文件索引

IVF 中的“倒排”来源于传统文本搜索中的倒排思想：

传统文本倒排：

```
term → posting list(document IDs)
```

IVF：

```
cluster centroid → inverted list(vector IDs)
```


普通存储：

```
doc_id → vector
```


IVF 建立：

```
centroid → 所属向量集合
```

因此称为倒排文件索引。


---

## 离线建索引

步骤：

### Step 1：K-Means 聚类

对全部 document embedding 执行聚类：

得到：

\[
n_{list}
\]

个 cluster centroid。


### Step 2：向量分桶

每个向量被分配到最近 centroid：

```
vector
   |
   v
nearest centroid
   |
   v
inverted list
```


---

## 在线检索流程

假设 query embedding 为：

\[
q
\]


### Step 1

计算 query 与所有 centroid 的距离：

选择最近：

\[
n_{probe}
\]

个 cluster。


### Step 2

只搜索这些 cluster：

```
selected clusters
        |
        v
candidate vectors
```


### Step 3

对候选向量执行精确距离计算：

返回 Top-K。


---

## IVF 参数

核心参数：

\[
n_{probe}
\]


影响：

| n_probe | Recall | 延迟 |
|-|-|-|
| 增大 | 提升 | 增加 |
| 减小 | 降低 | 降低 |


如果：

- n_probe 太小：

容易错过跨 cluster 的近邻；

- n_probe 太大：

接近暴力搜索。


---

## IVF 优缺点

### 优点

- 内存开销低；
- 索引结构简单；
- 适合静态大规模数据库。


### 缺点

- 强依赖聚类质量；
- 数据分布不均时 Recall 下降明显；
- 动态更新能力较弱。


主流实现：

- FAISS IVFFlat
- FAISS IVFPQ
