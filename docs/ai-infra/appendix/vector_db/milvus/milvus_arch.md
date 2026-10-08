---
title: "Milvus 架构图"
date: 2026-09-23
tags: ["向量数据库", "Milvus"]

---

# Milvus 架构图通俗解析
Milvus 是云原生向量数据库，**计算与存储分离**。整张图从左到右，分为：**客户端接入层 → 协调控制层 → 工作计算节点层 → 持久化存储层**。

> 先记住4个SQL缩写（图里的箭头标记）
> - **DDL/DCL**：建集合、删集合、权限管理（库表结构操作）
> - **DML**：插入、删除向量数据（写数据）
> - **DQL**：向量相似度搜索、查询（读数据）

![milvus向量数据库](/docs/interview/appendix/resume/fig/milvus_architecture_2_6.png)

## 1. Client SDK + Proxy（最左侧：接入层）
- **Client SDK**：应用程序入口，业务代码通过SDK发送向量写入/向量检索请求。
- **Proxy（代理节点，多副本）**：Milvus的网关，接收SDK请求，做参数校验、路由分发。
  - DDL(Data Definition Language，**数据定义语言**) / DCL(Data Control Language，**数据控制语言**)（建表删表）→ 转发给 Coordinator
  - DML（Data Manipulation Language，**数据操纵语言**，插入向量）→ 转发给 Streaming Nodes
  - DQL（Data Query Language，数据查询语言，向量搜索）→ 转发给 Query Nodes
> Proxy是无状态的，可以水平扩容，承担负载均衡。

## 2. Coordinator（协调器，集群大脑，控制平面）
接收Proxy发来的DDL/DCL请求。
职责：
1. 维护整个集群拓扑：管理所有节点注册、健康检查（图里`Registration`箭头指向etcd元存储）
2. 全局任务调度、分配分片、生成全局时间戳，保证集群元数据一致性
3. 下发管理指令给 Workers 工作节点（`Management`箭头）

> 同一时刻集群内只有1个活跃Coordinator，主从切换保证高可用。

## 3. Workers 工作节点（计算层，虚线框内，3类节点）
### ① Streaming Node 流节点（写入实时数据）
分片shard级别的“小脑”，**负责增量实时写入的数据**
1. 接收Proxy的DML插入请求，把写入操作追加到WAL（预写日志，图中`Append`），保证写入不丢数据。
2. 实时可查：刚插入还没固化的**增长数据(growing data)**，由Streaming Node直接提供查询。
3. 当segment（数据段）大小达到阈值，触发`Flush`：把内存里的增量数据打包，写入对象存储，变成**封存历史数据(sealed data)**。
4. 订阅（Subscription）：把flush完成的消息通知给Query Node，让Query Node感知新的历史段。
5. 故障恢复：节点宕机后，依靠WAL日志做数据恢复（`Recovery`箭头）

### ② Data Node 数据节点（后台索引&合并任务）
不对外提供查询，是后台任务节点：
- 监听Streaming Node的flush事件，读取对象存储里的封存segment
- 执行**建索引、Compaction（段合并，清理删除数据、合并小segment）**，完成后把索引文件写回对象存储。

### ③ Query Node 查询节点（向量检索主力）
负责**已经固化到对象存储的历史封存数据**的向量搜索：
1. 根据Coordinator指令，从对象存储加载segment数据和索引（`Segment Loading`）
2. 接收Proxy的DQL查询请求，执行向量相似度检索
3. 一次向量查询，会同时查两处：
    - Streaming Node：查询**新插入还没flush的实时增量数据**
    - Query Node：查询**已经落盘建索引的历史数据**
    最后合并两边结果返回给客户端。

## 4. Durable Storage 持久化存储（最右侧，三类存储，解耦设计）
### Meta Storage（元存储：etcd）
保存集群所有元信息：集合schema、分片信息、节点注册信息、数据段元数据。
etcd提供强一致、事务能力，Coordinator把集群状态存在这里。

### WAL（预写日志存储：Pulsar / Kafka / Woodpecker）
写入时先写WAL，再写内存。作用：**宕机恢复，保证写入原子性，消息流订阅**。
Streaming Node的所有插入操作，Append追加写入WAL。

### Object Storage 对象存储（MinIO / S3 / Azure Blob）
最终数据落地的地方：存放向量原始数据、向量索引文件。
> Milvus计算节点（Query/Data）本身不持久保存数据，只按需加载对象存储里的文件，**计算存储分离**，弹性扩缩容。

# 完整流程
## 写入流程（插入向量）
1. 业务SDK调用insert，请求发到Proxy
2. Proxy路由，转发给对应分片的Streaming Node
3. Streaming Node先Append写入WAL（保证不丢），写入内存，此时这条向量**立刻可以被检索**
4. 积累到阈值，Streaming Node执行Flush，把内存数据打包成segment写入对象存储
5. Data Node感知新segment，拉取数据，构建向量索引、执行compaction，索引写回对象存储

## 查询流程（向量相似度搜索）
1. SDK发起向量search，请求到Proxy
2. Proxy同时分发请求：
   - 发给Streaming Node：查询还未flush的实时增量数据
   - 发给Query Node：查询已经落盘、建好索引的历史segment
3. Query Node如果还没加载对应segment，会从对象存储加载数据+索引
4. 两边检索结果合并，返回给客户端

# 架构设计亮点总结
1. **冷热数据分离查询**：实时热数据由Streaming Node处理；历史冷数据由Query Node+对象存储承载。
2. **存算分离**：计算节点（Proxy/Streaming/Query/Data）本身不存永久数据，数据全在对象存储。计算资源和存储资源可以独立扩容。
3. **WAL保证写入可靠性**，etcd保证元数据强一致。
4. 微服务模块化，不同组件可以单独扩缩容：查询压力大就加Query Node；写入压力大就加Streaming Node。
