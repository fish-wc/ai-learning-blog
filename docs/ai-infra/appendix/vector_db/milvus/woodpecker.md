---
title: "Milvus 架构图"
date: 2026-09-23
tags: ["向量数据库", "Milvus","Woodpecker"]

---


# Woodpecker（啄木鸟）
一句话：**Milvus 自研的云原生WAL（预写日志）组件，同时充当集群内部消息流，Milvus 2.6可选，Milvus3.x默认启用，用来替代Kafka/Pulsar**

![架构](/docs\ai-infra\appendix\vector_db\milvus\figs\woodpecker.jpeg)

## 核心设计亮点：零磁盘架构（Zero-Disk）
- 日志主体**直接写入对象存储（S3/MinIO）**，不需要依赖本地磁盘持久化日志
- 日志元数据交给 **etcd** 保存
- 不再需要独立部署Kafka/Pulsar，减少一套中间件运维，简化集群部署

## 两种部署模式
1. **内嵌模式 Embedded（2.6默认）**
    直接跑在StreamingNode进程内部，轻量。采用批量写入，适合高吞吐场景，写入延迟会高一点（MinIO后端默认200ms批间隔）
2. **独立服务模式 Service（Milvus3.0+）**
    Woodpecker独立部署，支持QuorumBuffer三副本法定写，低延迟，适合高写入压力大集群；一次RTT完成副本复制，就近直读优化网络开销

## 在Milvus数据流里的作用
DML写入流程：
`SDK → Proxy → StreamingNode → Woodpecker(WAL)`
1. Insert / Delete（带墓碑tombstone）请求，先写入Woodpecker日志，日志落对象存储后，返回客户端写入成功
2. DataNode消费Woodpecker日志流：生成Growing Segment，满足条件Flush，变成Sealed Segment上传对象存储
3. QueryNode消费同一条Woodpecker日志流，实时感知增量数据、删除墓碑，更新内存检索数据
> 也就是说：Woodpecker **两件事一起干**
>  WAL预写日志：宕机后可以重放日志恢复数据，保证写入持久化
>  消息队列：作为数据流总线，给DataNode、QueryNode分发增量DML事件

## 和旧方案Pulsar/Kafka对比（面试高频）
- 旧方案：额外独立部署消息队列，运维重；需要本地磁盘存消息
- Woodpecker：Milvus内置，日志直接落对象存储；**每个Shard对应一条独立的Woodpecker日志流**，和Shard一一对应

## 关键补充
- 日志是Append-only（只追加），不会修改旧日志内容
- 支持日志压缩，过期日志可以清理
- 元数据（日志分片、消费位点checkpoint）全部存在etcd

## Markdown笔记文本（直接放进`ai-infra/milvus/terms.md`）
> **Woodpecker（啄木鸟）**：Milvus自研云原生WAL预写日志，兼具消息队列能力，Milvus2.6引入，3.x默认启用，替代Kafka/Pulsar。采用零磁盘架构，日志持久化到对象存储，元数据由etcd管理。每个Shard绑定一条独立Woodpecker日志流；DML写入先写入Woodpecker，DataNode、QueryNode消费日志流实现增量同步与故障恢复。支持内嵌模式、独立服务模式。

---
### 面试追问
Q：为什么Milvus要自研Woodpecker，不用Kafka？
A：降低运维复杂度，去掉独立中间件；适配向量库Shard模型；支持Timetick临时信号跳过持久化；云原生场景直接复用对象存储，不需要本地盘，适合弹性扩缩容。

Q：Woodpecker日志存在哪里？
A：日志数据存对象存储，日志元数据、消费位点存在etcd。
