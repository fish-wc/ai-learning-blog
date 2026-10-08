# 1. 对象存储 Object Storage
## 概念
对象存储是一种**非层次化**的数据存储范式，和我们熟悉的本地文件系统（目录/文件树）不一样。
它把数据打包成**对象（Object）**：对象 = 数据本身 + 对象元数据 + 唯一标识（object key），存放在桶（Bucket）里面。
> 典型产品：MinIO（Milvus本地部署常用）、AWS S3、阿里云OSS、腾讯COS

### Milvus 里的作用
Milvus 把**Segment向量文件、索引文件**全部放在对象存储。
- 优点：无限扩容、低成本、高持久化，适合存海量向量；
- 缺点：延迟偏高，不适合频繁随机读写；
- 区分：
  - etcd：存**元数据**（集合、segment信息、节点状态）
  - 对象存储：存**真实向量数据、索引文件**
  - Woodpecker(WAL)：存增量写入日志流

> 一句话：对象存储就是Milvus用来海量持久化向量和索引的地方。

# 2. Flush（刷盘）
## Milvus 定义
**Flush：把内存中Growing Segment（增量、可写入的段）的数据，持久化写到对象存储，然后把这个Segment状态变为Sealed（密封，不再接收新写入）**
1. 写入进来的数据先放在内存Growing Segment；
2. 触发Flush（手动/内存阈值/定时）；
3. 内存数据打包，上传对象存储；
4. Segment密封，不再接受新数据；新写入会开辟新的Growing Segment。

⚠️ 重点区分：
- Flush **不会删除内存里的数据**，只是把数据备份到对象存储，QueryNode还可以继续查询它；
- Flush只是持久化，**不做合并、不构建索引**；索引构建一般在Sealed Segment上。

# 3. Compaction（合并压缩）
## Milvus 定义
随着持续写入，会产生大量小的Sealed Segment（还有Delete墓碑标记）。
**Compaction：后台异步任务，读取多个小Segment，合并、清理删除数据，生成更少、更大的新Segment，之后删除旧小Segment。**

### 为什么需要Compaction
1. 大量小Segment会导致检索变慢：查询要扫描很多文件，IO开销大；
2. 处理删除：向量删除不是原地删，是写墓碑标记，compaction才会真正把删掉的数据剔除；
3. 减少对象存储文件数量，优化元数据管理。

### Compaction vs Flush（面试高频对比）
- Flush：内存 → 对象存储，单个Segment密封，**不合并**；触发快，偏持久化；
- Compaction：多个Sealed小Segment → 合并成大Segment，清理删除数据；后台任务，比较耗资源。

# 4. Query（Milvus中的DQL查询）
Milvus两类查询接口，都属于DQL：
1. **search：向量相似检索**
给定查询向量，在Segment向量索引里做近邻搜索，返回TopK向量。支持搭配标量过滤。
2. **query：标量查询**
按主键、字段过滤条件查询，**不做向量相似度计算**，类似数据库select。

### QueryNode 的工作逻辑
QueryNode负责加载Sealed Segment的索引/数据到内存，对外提供search和query能力。
查询流程：
1. Proxy收到DQL请求；
2. Proxy路由，把请求下发到负责对应shard的QueryNode；
3. QueryNode检索本地加载的Segment，返回局部结果；
4. Proxy汇总所有shard结果，做排序，返回最终TopK。

---
# 极简PPT短句版（直接复制）
- 对象存储：以对象为单元的分布式存储，Milvus存放Segment向量与索引文件。
- Flush：将内存增量Segment持久化写入对象存储，密封Segment，不合并数据。
- Compaction：后台异步合并多个小Segment，清理已删除数据，减少文件数量，提升检索性能。
- Query：Milvus DQL操作，分为向量检索search和标量查询query，由QueryNode执行。

如果你需要，我可以把前面所有术语整合成一张Milvus核心名词对照表，放进开题PPT。