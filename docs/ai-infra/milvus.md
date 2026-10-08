---
title: "从 FAISS 到 Milvus:我在 RAG 项目里踩过的向量检索坑"
date: 2026-09-22
tags: ["向量数据库", "Milvus", "FAISS", "RAG", "大模型工程化"]
target: [
    "FAISS 和 Milvus 的本质区别(库 vs 系统;单机内存 vs 分布式持久化服务)",
    "Milvus 的核心概念(Collection/Partition/Segment、Schema、一致性级别)",
    "常见索引类型和距离度量的选择逻辑,以及为什么某个场景选某个索引",
    "一个亲手做过的具体细节(哪怕很小,比如我们用的是 HNSW,ef_search 设了多少,做过什么调优)"
]

---

# 从 FAISS 到 Milvus:在 RAG 项目里踩过的向量检索坑


这篇笔记着重梳理:**FAISS 为什么会遇到瓶颈、Milvus 解决的是什么问题、它的架构长什么样、索引和距离度量该怎么选**。目标是让第一次接触向量数据库的人,读完能理解"这东西到底在解决什么问题"。

---

## 1. 先搞清楚问题:为什么向量检索需要专门的系统

大模型 / RAG 场景下,我们把文本、图片等非结构化数据通过 Embedding 模型映射成高维向量,检索的本质是:给定一个 query 向量 \(q\),在海量向量集合 \(\{v_1, v_2, \dots, v_n\}\) 中找出与 \(q\) 最相似的 top-k 个向量。

最直接的做法是暴力计算 \(q\) 与所有 \(v_i\) 的距离,再排序取前 k 个,这叫 **Flat / 暴力搜索**,时间复杂度是 \(O(n \cdot d)\)(\(d\) 是向量维度)。当 \(n\) 达到百万、千万级别时,这个计算量在实时问答场景里是不可接受的,于是才有了**近似最近邻搜索(Approximate Nearest Neighbor, ANN)** 算法,以及围绕 ANN 算法构建的工程系统。

## 2. FAISS 是什么:一个"算法库",不是一个"服务"

[FAISS](https://github.com/facebookresearch/faiss)(Facebook AI Similarity Search)本质上是 Meta 开源的一个 **C++ 算法库**,提供了各种高效的 ANN 索引结构(Flat、IVF、HNSW、PQ 等)和向量运算能力,可以通过 Python 绑定直接在内存里跑起来。

它的优点非常突出:

- **速度快**:针对 CPU/GPU 做了大量底层优化,单机检索性能极强;
- **算法全**:几乎覆盖了主流 ANN 索引类型,学术界很多新算法也会先在 FAISS 上验证;
- **轻量**:不需要额外部署服务,`pip install faiss-cpu` 几行代码就能跑通一个检索 demo。

但正因为它是一个"库"而不是"系统",在从**原型验证**走向**生产环境**时,会暴露出几个结构性短板:

| 维度 | FAISS 的局限 |
|---|---|
| **持久化** | 索引默认在内存中,需要自己写逻辑做 save/load,没有内建的数据落盘与恢复机制 |
| **动态更新** | 大多数高性能索引(如 IVF、HNSW)对"增量插入"和"删除"支持有限,频繁增删会导致索引退化,需要定期全量重建 |
| **分布式扩展** | 原生不支持水平扩展,数据量超过单机内存上限就很难处理 |
| **服务化能力** | 没有内建的多租户、鉴权、多副本高可用、标量字段过滤等"数据库该有的东西" |
| **一致性/事务** | 没有一致性级别的概念,并发读写时的可见性由使用者自己保证 |


所以更准确的说法是:**FAISS 解决的是"向量相似度计算"这一个算法问题,而 Milvus 解决的是"向量数据的存储、管理、检索"这一整套系统工程问题**。当项目还在验证 RAG 效果好不好的阶段,FAISS 完全够用,甚至更简单直接;但当项目需要支撑线上服务、数据要持续更新、要多个业务方共享同一份知识库时,就需要一个真正的"向量数据库"。
















## 3. Milvus 整体架构:为什么它是"[云原生](/docs/interview/appendix/resume/云原生.md)"的向量数据库

Milvus 从 2.0 版本开始采用了**存储计算分离**的云原生架构,核心思路是把"接入层""计算层""存储层"拆开,每一层都可以独立扩缩容。简化理解可以分成四层:

![milvus向量数据库](/docs/interview/appendix/resume/fig/milvus_architecture_2_6.png)

[几个关键设计点](/docs/interview/appendix/resume/milvus_arch.md):

- **日志即数据(Log as Data)**:写入请求先进入消息队列(Pulsar/Kafka),再由 Data Node 消费落盘到对象存储,这样天然具备了持久化和可回放能力,也是分布式一致性的基础;
- **计算与存储分离**:Query Node 只负责"算",数据本身存在对象存储里,这意味着计算节点可以无状态地扩缩容,故障恢复也更快;
- **元数据集中管理**:etcd 存 Collection Schema、分片信息等元数据,各个协调节点(Coordinator)基于元数据做任务调度。

这套架构解决的正是 FAISS 缺失的那部分能力:持久化、水平扩展、高可用。代价是复杂度上升——你需要部署和运维一整套分布式系统(当然 Milvus 也提供了 **Standalone 单机模式**,用 Docker 一条命令就能跑起来,适合中小规模场景,这也是很多团队从 FAISS 迁移过来时的第一步)。

## 4. 核心概念:Collection、Partition、Segment

理解 Milvus 的数据组织方式,是能不能讲清楚"具体怎么用"的关键:

- **Collection(集合)**:相当于关系型数据库里的"表",需要预先定义 Schema(字段、向量维度、主键等),是数据管理的基本单元;
- **Partition(分区)**:Collection 内部的逻辑分片,比如按时间、按业务线分区,可以缩小查询范围、提升检索效率;
- **Segment(段)**:数据实际存储的物理单元,分为可写的 Growing Segment 和已封存的 Sealed Segment,索引是构建在 Segment 粒度上的,这也是 Milvus 能做增量写入的关键——新数据先进 Growing Segment(此时可能用 Brute-force 或临时索引应急),积累到一定量后封存并异步构建正式索引。

这一点和 FAISS 的区别很直观:FAISS 里"索引"就是整个数据集的一个内存对象,增删数据往往意味着要重建索引;而 Milvus 通过 Segment 机制,把"持续写入"和"高效检索"这两个天然冲突的需求解耦开了。

## 5. 索引类型与距离度量:该怎么选

### 5.1 常见索引类型

| 索引类型 | 原理简述 | 适用场景 |
|---|---|---|
| **FLAT** | 暴力全量比较,召回率 100% | 数据量小(几万级以内)、对精度要求极高 |
| **IVF_FLAT** | 先用聚类(倒排文件,Inverted File)把向量分桶,查询时只在最近的若干个桶内做精确搜索 | 中等规模数据,精度和速度的平衡方案 |
| **IVF_PQ** | 在 IVF 基础上对向量做乘积量化(Product Quantization)压缩,牺牲一定精度换取更小的内存占用 | 超大规模数据、内存受限场景 |
| **HNSW** | 基于分层可导航小世界图(Hierarchical Navigable Small World),通过图结构做贪心搜索 | 对查询延迟极敏感、内存充足的场景,是目前最主流的高性能选择之一 |

### 5.2 距离度量该用哪个

Milvus 支持多种度量方式,最常用的三种:

**欧氏距离(L2)**:

\[
d(q, v) = \sqrt{\sum_{i=1}^{d} (q_i - v_i)^2}
\]

**内积(Inner Product, IP)**:

\[
\text{sim}(q, v) = \sum_{i=1}^{d} q_i \cdot v_i
\]

**余弦相似度(Cosine)**:

\[
\cos(q, v) = \frac{\sum_{i=1}^{d} q_i v_i}{\sqrt{\sum_{i=1}^{d} q_i^2} \cdot \sqrt{\sum_{i=1}^{d} v_i^2}}
\]

一个容易在面试里被问到、也容易讲错的点是:**如果向量已经做过 L2 归一化(即 \(\|v\|=1\)),那么内积和余弦相似度在数学上是等价的**,因为此时余弦相似度公式里的分母恒为 1。这也是为什么很多 Embedding 模型(比如常见的 BGE、text-embedding 系列)官方推荐"先归一化再用 IP 度量",这样可以省掉一次除法运算,在大规模检索场景下有实际的性能收益。

选择建议:文本语义检索场景(RAG 最常见的情形)通常用**归一化向量 + IP** 或直接用 **Cosine**;如果向量的模长本身携带语义信息(比如某些推荐场景),才需要用 L2。

## 6. 一次检索请求发生了什么(简化版查询流程)

1. 客户端通过 SDK 把 query 向量和过滤条件(可选的标量字段过滤,比如 `WHERE category == "tech"`)发给 **Proxy**;
2. Proxy 从 **Query Coord** 获取路由信息,把请求分发到持有相关 Segment 的 **Query Node**;
3. 每个 Query Node 在自己负责的 Segment 上并行执行 ANN 搜索,返回局部 top-k;
4. Proxy 汇总各个 Query Node 返回的局部结果,做全局排序,返回最终 top-k 给客户端。

这个"分片并行检索 + 汇总"的模式,和分布式搜索引擎(比如 Elasticsearch)的查询模式是同一套思路,也是 Milvus 能做水平扩展的根本原因——数据分布在多个 Segment / 多个 Query Node 上,查询天然可以并行。

## 7. Milvus vs FAISS:该怎么选,怎么讲清楚"迁移"这件事

简述：**先用轻量方案验证可行性,再用系统化方案支撑生产化**。

| 维度 | FAISS | Milvus |
|---|---|---|
| 定位 | 算法库 | 分布式向量数据库(系统) |
| 部署 | 无需部署,进程内嵌入 | 需要部署服务(支持 Standalone / Cluster 两种模式) |
| 数据规模 | 受限于单机内存 | 支持水平扩展,可处理十亿级向量 |
| 增删改 | 支持较弱,增删代价高 | 原生支持 Upsert/Delete,基于 Segment 机制解耦写入与索引构建 |
| 标量过滤 | 不支持(需要自己额外维护映射) | 原生支持向量检索 + 标量条件联合过滤 |
| 一致性 | 无此概念,单机内存天然强一致 | 提供 Strong / Bounded / Session / Eventually 多级一致性可选 |
| 适用阶段 | 算法验证、小规模离线实验 | 生产环境、多业务方共享的线上服务 |

**如果要在面试里讲清楚这次"迁移",建议按这个逻辑组织回答:**

1. **背景**:FAISS 阶段验证了 RAG 检索链路可行,但业务上线后数据持续增长、需要支持增量更新和多路并发访问,内存型单机方案出现瓶颈;
2. **选型对比**:调研了 Milvus / Weaviate / Qdrant 等方案,基于社区活跃度、与现有技术栈的兼容性(比如是否已有 K8s 部署经验)、以及对超大规模场景的验证案例,选择了 Milvus;
3. **具体工作**:Schema 设计(哪些字段建索引、向量维度、主键策略)、索引类型选择(为什么选 HNSW 或 IVF)、一致性级别选择(比如问答场景对实时性要求没那么极致,可以用 Bounded Staleness 换取更好的吞吐)、迁移过程中的数据双写/校验方案;
4. **效果**:哪怕是定性的("支持了增量数据接入""检索延迟稳定在多少毫秒"),都比空泛地说"更好用了"要有说服力。

如果你目前对第 3 点里的具体参数(比如实际用的 `nlist`、`ef_search` 取值,或者一致性级别到底选的哪个)还答不上来,建议回头找一下当时项目里 Milvus 的 Collection 配置和索引参数,哪怕数值记不准,也要能讲出"当时是怎么权衡的"这个逻辑,这比记住某个数字更重要。

## 8. 一个最小可跑的例子(pymilvus)

```python
from pymilvus import MilvusClient

# 1. 连接(Standalone 模式,本地或远程一个 endpoint 即可)
client = MilvusClient(uri="http://localhost:19530")

# 2. 创建 Collection(768 维,对应常见 Embedding 模型输出维度)
client.create_collection(
    collection_name="doc_chunks",
    dimension=768,
    metric_type="COSINE",   # 归一化后的语义检索场景常用
)

# 3. 插入数据(向量 + 元数据字段,便于后续做标量过滤)
data = [
    {"id": 1, "vector": [0.1, 0.2, ...], "text": "...", "source": "handbook.pdf"},
    {"id": 2, "vector": [0.05, 0.3, ...], "text": "...", "source": "faq.md"},
]
client.insert(collection_name="doc_chunks", data=data)

# 4. 检索:向量相似度 + 标量过滤联合查询
results = client.search(
    collection_name="doc_chunks",
    data=[query_vector],
    limit=5,
    filter='source == "handbook.pdf"',
    output_fields=["text", "source"],
)
```
## 9. 小结

这次重新梳理下来,我对"为什么要从 FAISS 迁移到 Milvus"这件事的理解,已经从"听说 Milvus 更适合生产环境"这种模糊印象,变成了能讲清楚具体是哪几个工程维度(持久化、增量更新、水平扩展、标量过滤、一致性)驱动了这个选择。这也是我认为技术选型类的经历,在面试里最值得深挖、也最容易讲出深度的地方——不是记住某个工具的功能列表,而是能讲清楚"没有它会遇到什么具体问题"。


参考文献
Milvus 官方文档 — Architecture Overview, https://milvus.io/docs/architecture_overview.md
Milvus 官方文档 — Index, https://milvus.io/docs/index.md
Milvus 官方文档 — Consistency Level, https://milvus.io/docs/consistency.md
Wang, J. et al. "Milvus: A Purpose-Built Vector Data Management System." SIGMOD 2021.
Facebook Research — FAISS Wiki, https://github.com/facebookresearch/faiss/wiki
Malkov, Y. A., & Yashunin, D. A. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE TPAMI, 2018.
Jégou, H., Douze, M., & Schmid, C. "Product Quantization for Nearest Neighbor Search." IEEE TPAMI, 2011.


## 术语表 terms

**RTT** : [Round-Trip Time](/docs/ai-infra/appendix/vector_db/milvus/RTT.md)，往返时间。指数据包从发送端发出，抵达接收端，再携带**应答**返回发送端的总网络耗时，单位通常为ms。Woodpecker访问远端对象存储存在RTT开销，因此采用批量写入减少网络请求次数，降低整体写入延迟。


**MemoryBuffer** ：[Woodpecker内嵌模式使用的内存缓冲区](/docs/ai-infra/appendix/vector_db/milvus/MemoryBuffer&QuorumBuffer.md)。写入数据先暂存在单机内存，攒批后一次性上传对象存储。只有数据成功写入对象存储，才返回写入确认。若节点宕机，内存中尚未刷入对象存储的数据会丢失。

**QuorumBuffer** ：[Woodpecker独立服务模式的仲裁内存缓冲区](/docs/ai-infra/appendix/vector_db/milvus/MemoryBuffer&QuorumBuffer.md)。基于Raft协议，数据同步至多节点内存副本；当多数（quorum）节点内存写入成功，即向客户端返回写入成功，后台异步持久化到对象存储。单节点故障不会丢失已确认写入的数据，写入延迟更低，需要多节点集群支撑。

**Raft一致性协议**：[Raft是分布式一致性算法，用于保证集群多节点数据一致，可容忍少量节点故障](/docs/ai-infra/appendix/vector_db/milvus/Raft.md)。拆分为三大核心机制：Leader选举、日志复制、安全性。集群节点有三种状态：Leader、Follower、Candidate。依靠任期(Term)和多数派(Quorum)仲裁机制：写请求需要复制到半数以上节点确认，才算提交成功。

**Remote Procedure Call，远程过程调用**：[是一种通信机制，允许程序像调用本地函数一样调用远端机器上的服务函数，屏蔽底层网络通信细节](/docs/ai-infra/appendix/vector_db/milvus/RPC.md)。Milvus集群内部节点之间通信采用gRPC（基于Protobuf二进制序列化，性能高）。Raft协议中节点间投票、日志复制消息也基于RPC实现。

**Woodpecker 的零磁盘架构 Zero-Disk** ： [Woodpecker 本身不使用节点本地磁盘持久化日志数据，日志直接写入远端对象存储（MinIO/S3），本地磁盘只做临时buffer缓存，不需要用来永久保存日志](/docs/ai-infra/appendix/vector_db/milvus/磁盘架构.md)。


**墓碑**：[墓碑就是逻辑删除标记](/docs/ai-infra/appendix/vector_db/milvus/墓碑.md)。删除向量时，不会直接抹掉对象存储里原始向量，而是单独记录一条「这个主键已经被删」的标记，这个标记就叫 Tombstone（墓碑），Milvus里存放在Delta Log增量删除日志里。查询阶段过滤掉标记数据；Compaction阶段永久清理。


**云原生**：[云原生是一套架构理念与技术栈](/docs/ai-infra/appendix/vector_db/milvus/云原生.md)，应用从设计之初就面向云环境（公有云/私有云/混合云）开发、部署、运维，充分利用云的弹性、分布式能力，而不是把传统单机应用简单搬到云上。四大核心技术，容器，容器编排，微服务，DevOps+持续交付（CI/CD）。

**DevOps, Development & Operations**：[开发（Dev）+运维（Ops）](/docs/ai-infra/appendix/vector_db/milvus/DevOps.md)，不是一个工具，是一套**文化、流程、工程方法论**，目标是打通**开发团队和运维团队**的壁垒，让软件**更快、更稳定、更频繁地交付上线**。


**CNCF，Cloud Native Computing Foundation（云原生计算基金会）**：[隶属于Linux基金会，负责托管、孵化云原生开源项目](/docs/ai-infra/appendix/vector_db/milvus/CNCF.md)，项目分为沙盒、孵化、毕业三个阶段。Milvus、K8s、etcd均为CNCF毕业项目。


**分布式 KV 存储**：[多机器组成的键值数据库](/docs/ai-infra/appendix/vector_db/milvus/分布式KV存储.md)，**可靠存取少量元数据、配置信息**，etcd 就是最典型的一款。
