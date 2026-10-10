# RAG两阶段流水线：Retrieval + Rerank 原理、ANN索引与Cross-Encoder
> 摘要：RAG采用**召回-重排两阶段架构**，本质是在大规模知识库约束下，平衡检索吞吐量与相关性判别精度。召回阶段基于Bi-Encoder稠密嵌入结合ANN近似检索，优先保证Recall；重排阶段使用Cross-Encoder对少量候选做细粒度相关性打分，修正向量相似带来的假阳性。本文聚焦两阶段底层建模，展开IVF、HNSW索引与Cross-Encoder的内部机制，同时讨论智能体记忆模块中该范式的延伸价值。

## 1 两阶段Pipeline的设计动机
在面向智能体的长期记忆检索场景中，知识库规模可达百万级文本片段。若直接使用强相关性模型对全库做逐对打分，推理开销不可接受。因此系统拆分为两个目标异构的阶段：
1. **Retrieval（召回）**：粗粒度检索，目标最大化Recall。使用Bi-Encoder将query与文档映射至共享向量空间，借助ANN索引快速返回Top-K候选；允许混入噪声，核心约束是**不能丢失真实相关文档**。
2. **Rerank（重排）**：细粒度排序，目标最大化Precision。仅对召回输出的少量候选集合执行联合相关性判别，过滤假阳性、重新排序，选取Top-N送入LLM生成。

两阶段的核心矛盾：Bi-Encoder只能编码全局向量，无法建模query与文档token之间的细粒度交互；Cross-Encoder可以完成跨文本token级注意力匹配，但无法预计算文档侧表征，不支持海量库扫描。

## 2 Retrieval阶段：Bi-Encoder嵌入与ANN索引
### 2.1 Bi-Encoder向量空间约束
Bi-Encoder采用双塔结构：
$$
\boldsymbol q = E_q(q),\quad \boldsymbol d = E_d(d)
$$
相似度由向量空间距离度量，常用余弦相似度：
$$
sim(\(\boldsymbol q,\boldsymbol d\))=\cos(\boldsymbol q,\boldsymbol d)=\frac{\boldsymbol q^\top \boldsymbol d}{\|\boldsymbol q\|\|\boldsymbol d\|}
$$
> 关键结论：文档离线嵌入与在线query向量化**必须使用同一套模型权重**。其本质要求$\boldsymbol q,\boldsymbol d$处于**同一个语义向量空间**；若使用不同模型，两个向量分布相互独立，相似度不具备语义意义。理论上可通过额外线性映射做空间对齐，但工程代价大、泛化弱，工业场景极少采用。

离线阶段，所有chunk的向量预先计算存入索引；在线阶段仅编码query向量，执行ANN检索。ANN是一类近似最近邻算法的统称：牺牲少量召回精度，规避暴力检索$\(O(N)\)$复杂度，将检索开销降至$O(\log N)$，支撑百万乃至十亿规模向量库。

### 2.2 IVF（Inverted File）倒排文件索引
IVF名字中的**倒排**，源自与传统文本倒排索引相同的数据范式：
- 传统倒排：`term → posting list(文档ID列表)`
- IVF向量倒排：`聚类中心centroid → inverted list(簇内向量列表)`

正向存储为`doc_id → vector`；IVF建立的是从聚类中心指向所属向量集合的映射，因此被命名为倒排文件索引。

#### 离线建索引
1. 在全量文档向量上执行K-Means聚类，得到$n_{list}$个聚类中心；
2. 每个向量分配至距离最近的centroid，追加进该centroid对应的倒排列表。

#### 在线检索
1. 计算query向量与全部centroid的距离，选取最近$n_{probe}$个簇；
2. 仅加载这$n_{probe}$个倒排列表，对桶内向量做精确距离计算；
3. 全局排序返回Top-K。

$n_{probe}$是核心调参：取值越大，打开更多簇，Recall提升，延迟上升；取值过小则易丢失跨簇近邻。IVF优点是内存开销低；缺点依赖聚类质量，向量分布不均匀时召回显著退化。FAISS的IVFFlat/IVFPQ是该索引的主流实现。

### 2.3 HNSW（Hierarchical Navigable Small Worlds）分层可导航小世界图
HNSW属于基于图的ANN索引，**无聚类操作**；底层为完整稠密图，上层为稀疏路标层，用于快速空间导航。

#### 核心层规则
- Layer 0（底层）：存储**全部向量节点**，图最稠密；
- Layer $\(l>0\)$：仅包含部分节点；若节点存在于Layer $l$，则该节点必然存在于所有下层$0,1,\dots,l$；
- 上层节点**不是从底层事后采样聚类代表点**：节点插入的瞬间，通过几何分布随机采样确定该节点的最高存在层：
$$
l_q = \lfloor -\frac{\ln(\mathcal U(0,1))}{\text{ml}} \rfloor
$$
`ml`控制分层稀疏度；层数越高，节点被抽中概率指数衰减。大部分节点仅存在于Layer0，极少数节点晋升至高层作为全局路标。

> 每层为独立的NSW（Navigable Small World）图，**层与层之间不存在图边**；检索时仅将上层搜索结果作为下一层的搜索起点，属于算法跳转逻辑。

#### 节点插入（建图）流程
超参定义：
- $M$：非0层节点最大边数；$M_0$：Layer0最大边数，一般$M_0 \ge M$；
- $ef\_construction$：建图阶段候选近邻池大小。

1. 对新向量$q$，采样得到最高层$l_q$；
2. 从HNSW全局入口节点、全局最高层开始逐层向下遍历；高于$l_q$的层仅执行贪心搜索，**不建立边**，更新搜索入口点；
3. 下降至$l_q, l_q-1,\dots,0$：
    - 以上一层得到的入口点为起点，在当前层图中贪心搜索，收集$ef\_construction$个候选近邻；
    - 选出距离$q$最近$M$个节点，建立双向无向边；
    - 对被连接邻居，若边数超限，则裁剪掉距离最远的边；
    - 更新入口点，继续下降一层；
4. 若$l_q$大于当前全局最高层，更新全局入口节点与全局最高层。

#### 在线检索流程
1. 以全局顶层入口节点为起点，逐层贪心搜索，不断向离query更近节点移动；
2. 逐层下降直至Layer0；
3. 在底层稠密图上使用$ef\_search$候选池执行局部贪心搜索；
4. 候选池内精确距离排序，返回TopK。

`\(ef\_search\) ≥ TopK`；ef越大召回越高，延迟越高。HNSW优势是鲁棒性强、支持增量插入；代价是邻接边带来额外内存开销，是Milvus/Qdrant等向量数据库的默认索引。

### 2.4 IVF vs HNSW
|维度|IVF|HNSW|
|---|---|---|
|底层结构|聚类中心 + 倒排列表|多层独立NSW无向图|
|离线构建|K-Means聚类，向量分桶|节点逐个插入，动态建边|
|检索逻辑|选择nprobe个簇，桶内暴力扫描|高层路标导航，底层局部图贪心遍历|
|内存开销|低，仅存储向量+少量centroid|高，向量+邻接边|
|向量分布鲁棒性|弱，依赖聚类假设|强，无聚类前置假设|
|增量更新|IVFFlat增量困难，IVFPQ有量化损失|天然支持增量插入|

## 3 Rerank阶段：Cross-Encoder原理
Cross-Encoder是两阶段流水线的重排核心，与Bi-Encoder最本质差异：**不独立编码query与文档，将文本拼接后送入同一个Transformer做联合编码**。

输入格式：
```
[CLS] q_t1 q_t2 ... [SEP] d_t1 d_t2 ... [SEP]
```
拼接序列全部token参与自注意力计算，允许query侧token与文档侧token直接交互、对齐，捕捉指代、否定、细粒度语义匹配，解决Bi-Encoder“向量相似但事实无关”的假阳性。

### 前向打分流程
1. 拼接序列经过多层Transformer编码；
2. 取`[CLS]`位置隐状态$h_{\texttt{CLS}}$，经过全连接层输出logit；
3. Sigmoid归一化得到0~1相关性分数：
$$
s(\(\boldsymbol q,\boldsymbol d\))=\sigma(W\cdot h_{\texttt{CLS}}+b)
$$
每一对$(\(\boldsymbol q,\boldsymbol d\))$都需要独立执行一次模型推理，**文档向量无法预计算**，因此仅可作用于召回后的少量候选集合。

### 训练范式
主流两种目标：
1. 二分类BCE Loss：样本$(\(\boldsymbol q,\boldsymbol d\),y)$，$y\in\{0,1\}$标记是否相关。
$$
L_{BCE}=-\big[y\log s(\(\boldsymbol q,\boldsymbol d\))+(1-y)\log(1-s(\(\boldsymbol q,\boldsymbol d\)))\big]
$$
2. Pairwise排序损失：输入$(\(\boldsymbol q,\boldsymbol d\)_+,d_-)$，约束模型满足$s(\(\boldsymbol q,\boldsymbol d\)_+)>s(\(\boldsymbol q,\boldsymbol d\)_-)$，优化相对排序。

### 工程定位
Rerank无法弥补召回阶段的Recall损失：若相关文档未被召回，Cross-Encoder无法凭空找回。在智能体记忆系统中，Rerank用于对候选记忆片段做证据筛选，降低幻觉、提升事实对齐。

## 4 面向智能体记忆的拓展思考
智能体的记忆模块本质是动态RAG：记忆片段持续新增、查询为智能体内部推理生成，查询分布偏移显著。
- 召回层：HNSW的增量插入特性更适配动态记忆；混合检索（BM25 + Dense Embedding）可兼顾实体关键词与语义；
- 重排层：除Cross-Encoder外，LLM pairwise rerank可处理复杂逻辑相关性，但推理成本更高，适合小候选集；
- 权衡：当知识库持续膨胀，ANN索引的建库、内存、检索延迟会成为智能体长期记忆的主要瓶颈。

## 参考文献
[1] Malkov Y, Yashunin D. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018. arXiv:1603.09320
[2] Jégou H, Douze M, Schmid C. Product quantization for nearest neighbor search[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011.
[3] Nogueira R, Cho K. Passage Re-ranking with BERT[J]. arXiv preprint arXiv:1901.04085, 2019.
[4] Karpukhin V, Oguz B, Goyal N, et al. Dense Passage Retrieval for Open-Domain Question Answering[J]. EMNLP, 2020.
[5] Johnson J, Douze M, Jégou H. Billion-scale similarity search with GPUs[J]. IEEE Transactions on Big Data, 2019. (FAISS)
[6] Gao Y, Xiong Y, Gao Z, et al. Retrieval-Augmented Generation for Large Language Models: A Survey[J]. arXiv preprint arXiv:2312.10997, 2024.
