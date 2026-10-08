---
title: "etcd"
date: 2026-09-23
tags: ["向量数据库", "Milvus","etcd"]
---

# etcd
名字由来：`/etc`（Linux 存放配置的目录）+ **d**istributed → distributed `/etc`，分布式配置目录，读作 /ˈɛtsiːdiː/。
> 一句话定义：**基于Raft共识算法的强一致分布式KV存储**，CNCF项目，K8s原生元数据存储。

## 在 Milvus 里的核心作用
1. **存储Milvus全部元数据（最核心）**
    - 集合、分区、字段schema、索引信息
    - Segment元数据（segment属于哪个集合、状态、位置）
    - 消息消费checkpoint、用户角色权限信息
    > 注意：**向量原始数据不存etcd，向量数据存在对象存储MinIO/S3**，etcd只存少量元信息。
2. **服务注册与健康检查**
    Proxy、QueryNode、DataNode、Coordinator启动后向etcd注册；节点挂掉etcd自动感知，Coordinator做故障转移。
3. **分布式协调：选主、分布式锁**
    DataCoord/QueryCoord等组件的leader选举，保证集群同一时间只有一个主Coordinator干活；
4. **Watch机制**
    Coordinator监听etcd元数据变更，集合/索引创建删除时实时感知。