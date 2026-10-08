---
title: "Milvus 架构图"
date: 2026-09-23
tags: ["向量数据库", "Milvus","Segment"]

---


# Segment（段）

概述：Segment是Milvus最小数据单元，向量数据、索引都封装在 Segment 里面，分为Growing（内存可写入）与Sealed（刷盘密封）。Flush将Growing转为Sealed并存入对象存储；Compaction合并多个小Segment。QueryNode加载Sealed Segment至内存执行向量检索。

> Collection（集合） → Shard（分片） → **Segment（段）**

## 两种状态的 Segment
### 1. Growing Segment（增量段 / 可写入段）
- 处于**可写入状态**，接收新的 Insert / Delete 数据
- 存放在 **DataNode 的内存**，**还没有刷到对象存储**
- 不构建向量索引；只能做暴力检索，查询性能差
- 当满足条件（内存上限、定时、手动flush）触发 Flush

### 2. Sealed Segment（密封段）
- Flush 之后：Growing Segment 打包，上传到对象存储，**不再接收任何新写入**
- 在对象存储上保存：向量原始数据、标量字段、删除墓碑、后续构建的向量索引文件
- QueryNode 会把 Sealed Segment 下载加载到内存，用向量索引做高效相似检索

## Segment 的生命周期（完整链路）
1. 写入数据，进入 DataNode，写入 **Growing Segment（内存）**
2. 触发 Flush：Growing → Sealed，文件上传对象存储，etcd 更新元数据
3. QueryCoord 感知新 Sealed Segment，通知 QueryNode **从对象存储拉取文件加载进内存**，提供检索
4. 持续写入产生大量小 Sealed Segment，后台 Compaction：把多个小Segment合并成一个大的新Segment，清理掉带删除标记的数据
5. Compaction完成：新大Segment上传对象存储；旧小Segment保留一段时间后后台删除

## 关联
1. **Shard vs Segment**
Shard：水平分片，负责写入路由，一个Shard对应一条独立WAL日志流；
Segment：Shard内部的数据文件单元，承载真实向量。
> 一个Shard可以有多个Growing + 多个Sealed Segment。

2. **Segment 和 对象存储**
只有 **Sealed Segment** 的文件保存在对象存储；Growing只在内存。

3. **Segment 和 QueryNode**
QueryNode只加载Sealed Segment到内存做检索；Growing Segment不会被QueryNode加载。

## 补充几个细节
- Segment 里面不仅存向量，还存标量字段、主键、删除标记（Milvus删除不是原地删，是打墓碑）
- Compaction 的目的就是合并大量细碎小Segment：Segment越多，检索时要扫描的索引越多，查询延迟越高
- Segment 的元数据（ID、状态、对象存储路径）全部存在 etcd；向量本体在对象存储


