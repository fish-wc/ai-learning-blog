# 1. Shard（分片）
## 全称：没有英文缩写，单词本身就是 `shard`，含义：碎片、分片
> Milvus 中：**Shard 是集合（Collection）的水平拆分单元**，一个集合可以拆成多个 Shard，实现写入和查询的并行扩展。

### Milvus 里 Shard 的要点
1. **拆分规则**：按主键哈希路由。写入一条向量，对主键做哈希，决定这条数据落到哪个 Shard。
2. 一个 Shard 对应**一条独立的 WAL 流（Woodpecker）**。
   > 这是 Milvus 2.x/3.x 的关键：每个shard有独立的日志流，shard之间写入互不阻塞。
3. Shard 包含多个 Segment：
   - 数据不断写入Shard → 生成增长的增量 Segment（Sealed/Growing）
   - 后台 Compaction 把小 Segment 合并成大 Segment
4. 读写流程：
   - 写入：Proxy 根据主键哈希路由到对应Shard的日志流
   - 查询：Proxy 把查询请求**广播到所有Shard**，每个shard本地检索，最后合并结果
5. 对比区分（面试高频）
   - **Shard：水平拆分，数据按哈希分到不同流，用于扩容写入吞吐**
   - **Segment：Shard内部的数据文件单元，向量索引构建在Segment上，用于检索**

> 一句话记忆：Collection（集合）→ 分成多个 Shard；每个Shard内部由很多Segment组成。

---