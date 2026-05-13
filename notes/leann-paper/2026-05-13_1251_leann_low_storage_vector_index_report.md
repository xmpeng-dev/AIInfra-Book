# LEANN：低存储开销向量索引
# LEANN: A Low-Storage Overhead Vector Index

> **arXiv/DOI:** [arXiv:2506.08276](https://arxiv.org/abs/2506.08276) | **PDF:** [arXiv PDF](https://arxiv.org/pdf/2506.08276)  
> **发表信息:** arXiv v2, 2025-11-25；论文 PDF 标注 Under Review，代码仓库 README 标注 MLSys 2026  
> **机构:** UC Berkeley, CUHK, Amazon Web Services, UC Davis  
> **代码:** [yichuan-w/LEANN](https://github.com/yichuan-w/LEANN)  
> **领域:** Vector Search · Approximate Nearest Neighbor Search · RAG · On-device AI · Storage-efficient Index  
> **核心贡献:** LEANN 通过 graph-based selective recomputation、two-level search、dynamic batching 和 high-degree preserving graph pruning，把 76GB/60M passages RAG 数据库的向量索引存储从 HNSW 的 188GB 降到 4GB，同时在 NQ 上保持 25.5% downstream accuracy，并把端到端延迟控制在 23.34s。

---

## 一、问题分析 (Problem Analysis)

### 1.1 研究背景 (Research Background)

**领域现状**:

- RAG、推荐、个人助理、内容搜索都越来越依赖 embedding-based vector search。非结构化对象先被编码为高维向量，再通过近似最近邻搜索 (ANNS) 找 top-k 相似对象。
- 精确最近邻需要线性扫描高维向量，真实系统通常使用 IVF、HNSW、NSG、Vamana/DiskANN 等近似索引。
- 图索引通常有最好的召回-延迟表现，但存储成本高：既要存高维 dense embeddings，也要存邻接表/层级图等 index metadata。
- 本地 RAG / 个人设备场景越来越重要，因为它能保护隐私、支持离线访问、避免上传私有数据到云端，但个人设备的磁盘/RAM 容量往往不够承载大规模向量库。

**核心挑战**:

- **embedding 体积过大**：论文的 RPJ-Wiki 设置中，76GB 原始文本切成 60M 个 256-token passages 后，Contriever 生成的 768 维 FP32 embedding 达到 173GB。
- **图索引 metadata 也不可忽略**：HNSW 在同一设置下还需要 15GB metadata；一个 64 邻居节点只存邻居 ID 就约 256 bytes，已经是典型 1KB 文本 chunk 的 25%。
- **强压缩会损伤召回和 RAG 准确率**：PQ 若要把 173GB embedding 压到约 5GB，需要约 35x 压缩，量化误差会让下游准确率甚至低于 BM25。
- **低存储方案常常牺牲延迟**：IVF-Recompute 不存 embedding，但需要 O(sqrt(N)) 级别的 embedding 重算；在 60M passages 上可比 LEANN 慢到两个数量级。

**研究动机**:

论文抓住了一个关键系统事实：在很多 RAG workload 中，LLM generation 是端到端延迟主导项。Table 1 中 Qwen3-4B 在 RTX 4090 上生成耗时约 20.9s，而 HNSW 检索只要 0.05s。既然生成阶段已经很慢，系统可以把少量检索延迟换成巨大的索引存储节省，这对本地 RAG、冷数据检索、个人知识库非常有价值。

### 1.2 问题定义 (Problem Definition)

**具体问题**:

- **输入**：原始数据对象集合，例如文本 chunks、邮件、图片或网页；查询对象；embedding 模型；存储预算。
- **输出**：query 的 top-k 近似最近邻，用于 RAG 或语义搜索。
- **约束**：索引总存储远小于原始 dense vector index；召回和下游准确率接近 HNSW；检索延迟在 RAG 端到端流程中可接受。
- **评估指标**：索引存储大小、Recall@3、检索延迟、RAG 端到端延迟、Exact Match/F1、更新延迟、构建峰值存储。

**问题形式化**:

向量检索目标是对向量集合 \(X = \{x_1, x_2, ..., x_N\} \subset R^d\) 和查询 \(q\)，返回 top-k 最近集合 \(S_q\)：

\[
|S_q| = k,\quad ||q-x_i|| \le ||q-x_j||,\ \forall x_i \in S_q,\ x_j \in X \setminus S_q
\]

近似搜索质量用：

\[
Recall@K = |S_q \cap S'_q| / k
\]

图压缩问题可写为：在图 \(G_1\) 的 metadata 存储不超过预算 \(B\)、准确率不低于阈值 \(\tau\) 的条件下，最小化搜索中需要重算 embedding 的节点数：

\[
\min T(G_1)=\sum_{i=1}^{ef}|V_i|
\]

\[
s.t.\quad Space(G_1)=\sum_{v \in V(G_1)}deg(v)\cdot s_{edge} \le B,\quad Acc(G_1)\ge\tau
\]

其中每条邻居 ID 以 4 bytes 存储，\(ef\) 是 graph search queue size。

### 1.3 解决方案 (Solution)

**核心思路**:

LEANN 的核心不是“把向量压得更狠”，而是“不再保存完整向量”。它保留一个高度压缩的图结构和 PQ approximate embedding table，在查询时沿 graph 做 best-first traversal。对候选节点先用便宜的 PQ approximate distance 过滤，再只对少数有希望的节点调用本地 embedding model 重算 exact embedding。图结构本身再通过保留高出度 hub 节点、裁剪低价值边来降 metadata。

**方法概述**:

1. **离线构建**：对数据 chunk 计算 embedding，构建 HNSW-style proximity graph；随后丢弃 dense embeddings，只保存压缩后的 graph adjacency lists 和高压缩 PQ table。
2. **在线查询**：query 先被编码为 query embedding；LEANN 在 pruned graph 上搜索。每步先用 PQ approximate distance 排候选，再按 reranking ratio \(\alpha\) 选择 top \(\alpha\%\) 候选做 exact embedding recomputation。
3. **动态 batching**：不是每访问一个节点就单独跑 embedding model，而是跨多个 graph exploration steps 收集待重算节点，凑到目标 batch size 后统一送入 GPU。
4. **图剪枝**：保留 top \(\beta\%\) 高度节点作为 graph navigation hubs；多数普通节点只保留较小出度 \(m\)，hub 节点保留较大出度 \(M\)。
5. **受限存储构建与更新**：用 k-means soft assignment 把数据分片、分片建图再合并，降低构建阶段峰值存储；新增节点通过缓存、简化 neighbor selection 和 delayed insertion 降低更新成本。

**技术细节**:

*Two-Level Search with Hybrid Distance*:

- **功能**：减少需要 exact embedding recomputation 的节点数。
- **实现**：维护 exact queue (EQ) 和 approximate queue (AQ)。邻居先进入 AQ；从 AQ 取 top \(\alpha\%\) 且不在 EQ 的候选做 exact recomputation，再放入 EQ 继续图遍历。
- **创新**：不像 DiskANN 式“全程 approximate search 后只 rerank top candidates”，LEANN 把 approximate 和 exact distance 交错使用，避免高压缩 PQ 误差把图遍历带偏。

*Dynamic Batching*:

- **功能**：提高 embedding recomputation 的 GPU 利用率。
- **实现**：放松 best-first search 的严格顺序，跨多个探索 step 收集待重算节点，达到目标 batch size，例如 64，再批量编码。
- **创新**：承认图搜索顺序会有轻微 staleness，但换来显著吞吐提升；论文消融显示总加速平均到 1.8x，峰值 2.0x。

*High-Degree Preserving Graph Pruning*:

- **功能**：压缩 graph metadata，同时尽量不破坏 navigability。
- **实现**：用节点度数作为重要性 proxy，保留 top \(\beta\%\) 高度节点，普通节点出度限制为 \(m=M/5\)，hub 节点可保留 \(M\) 条边。论文经验上 top 2% high-degree nodes 很关键。
- **创新**：不是随机删边，也不是统一降低最大出度，而是保留图搜索中被频繁访问的 hub backbone。

*Storage-Efficient Index Build*:

- **功能**：避免“构建时必须先落盘完整 embedding”的峰值存储问题。
- **实现**：先对小样本 k-means 得到 centroids；每个 passage sequential embedding 后分配到最近两个 centroid；每个 shard 重新计算 embedding 建图后立刻丢弃；最后合并 shard graph。
- **创新**：在 Appendix D 中，15 shards 带来约 5x 构建峰值存储下降，k-means shard 的 graph quality 接近原始 HNSW。

*Efficient Update*:

- **功能**：支持动态 add/delete。
- **实现**：对 add 使用距离缓存和简化 RNG pruning，把单点插入复杂度从 \(O(M\cdot efC + efC^2 + M^3)\) 降到 \(O(M\cdot efC)\)。对 delete 使用 soft delete flag，查询结果过滤 inactive nodes，删除比例超过阈值再后台 rebuild。
- **创新**：把“只存图不存向量”导致的更新重算开销控制住。

**算法/架构描述**:

LEANN 的数据面可以理解为三层：

- **原始对象层**：保存文本/图像等原始数据，用于按需重算 embedding。
- **轻量近似层**：PQ table 保存高压缩 approximate embeddings，用于 cheap distance estimation。
- **图导航层**：pruned HNSW-style graph 保存邻接关系；高度节点保留更多边，低度节点被压缩。

查询路径是：query embedding -> graph entry point -> approximate distance filter -> exact recomputation batch -> exact queue traversal -> exact reranking -> 返回 top-k。

---

## 二、实验效果 (Experimental Results)

### 2.1 实验设置 (Experimental Setup)

| Item | Details |
|---|---|
| Datastore | RPJ-Wiki，约 76GB 原始 Wikipedia 文本；切成 256-token chunks |
| Scale | 60M passages，Contriever 生成 768-dimensional embeddings，总 embedding 约 173GB |
| QA Benchmarks | NQ, TriviaQA, GPQA, HotpotQA |
| Extra Datasets | FinanceBench, Enron Email Corpus, LAION |
| Baselines | HNSW, IVF, DiskANN, IVF-Disk, IVF-Recompute, PQ Compression, BM25 |
| Metrics | Storage size, Recall@3, retrieval latency, end-to-end RAG latency, EM/F1, update latency |
| Hardware | RTX 4090 workstation, 32GB RAM, 1TB disk, WSL2；AWS EC2 M1 Mac, Apple M1 Ultra, 128GB RAM, 512GB EBS |
| Models | Contriever embeddings；Qwen3-4B text generation；Qwen2.5-VL-7B-Instruct for multimodal workload |

### 2.2 主要结果 (Main Results)

**NQ + 76GB RPJ-Wiki 上的核心对比**:

| Method | Downstream Accuracy (%) | Storage Size (GB) | Index Metadata (GB) | Vectors (GB) | E2E Latency (s) | Search (s) | Generation (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 18.3 | 59 | - | - | 21.36 | 0.03 | 21.33 |
| HNSW | 25.5 | 188 | 15 | 173 | 20.95 | 0.05 | 20.90 |
| PQ | 17.9 | 20 | 15 | 5 | 25.45 | 4.53 | 20.92 |
| **LEANN** | **25.5** | **4** | **2** | **2** | **23.34** | **2.48** | **20.86** |

**解读**:

- LEANN 的存储只有 HNSW 的 \(4/188 \approx 2.1\%\)，即约 47x 更小；论文摘要和结论概括为 up to 50x reduction。
- LEANN 和 HNSW 的 downstream accuracy 都是 25.5%，说明在 90% Recall@3 目标下，RAG 准确率没有因大幅降存储而下降。
- 检索延迟从 HNSW 的 0.05s 增加到 2.48s，但端到端只从 20.95s 增到 23.34s。这个 trade-off 是 LEANN 面向 RAG 而非 ultra-low-latency serving 的核心前提。
- PQ 的存储也不大，但准确率 17.9% 低于 BM25 的 18.3%，说明高倍率量化会破坏语义检索质量。

**RTX 4090 上达到 90% Recall@3 的检索延迟**:

| Dataset | Generation (s) | Method | Retrieval (s) | Retrieval / Total |
|---|---:|---|---:|---:|
| NQ | 20.86 | HNSW | 0.05 | 0.20% |
| NQ | 20.86 | DiskANN | 0.03 | 0.10% |
| NQ | 20.86 | IVF-Recompute | 307.61 | 93.60% |
| NQ | 20.86 | **LEANN** | **2.48** | **10.60%** |
| GPQA | 69.60 | HNSW | 0.04 | 0.06% |
| GPQA | 69.60 | **LEANN** | **1.12** | **1.60%** |
| TriviaQA | 17.17 | HNSW | 0.04 | 0.20% |
| TriviaQA | 17.17 | **LEANN** | **2.96** | **14.70%** |
| HotpotQA | 23.28 | HNSW | 0.05 | 0.20% |
| HotpotQA | 23.28 | **LEANN** | **7.12** | **23.40%** |

论文正文称 LEANN 在 Table 2/3 中通常给端到端流程增加低于 20% 的检索开销；HotpotQA 表中为 23.4%，是一个值得注意的边界案例。原因是 HotpotQA 需要 multi-hop reasoning，而实验只做 single-hop retrieval，搜索路径更复杂。

**个人数据集上相对 HNSW 的存储节省**:

| Dataset | Generation (s) | LEANN Retrieval (s) | Overhead (%) | Storage Savings vs HNSW |
|---|---:|---:|---:|---:|
| FinanceBench | 46.0 | 1.5 | 3 | 97% |
| Enron | 22.3 | 1.9 | 8 | 98% |
| LAION | 6.6 | 1.6 | 20 | 97% |

**下游 RAG 准确率**:

- Figure 4 比较 BM25、PQ、HNSW、LEANN 在 NQ、TriviaQA、GPQA、HotpotQA 上的 EM/F1。
- 论文文字给出的关键结论是：LEANN 在所有数据集上都优于 BM25 和 PQ；EM 相对 BM25 最高提升 11.8%，相对 PQ 最高提升 11.3%；F1 相对 BM25 最高提升 12.0%，相对 PQ 最高提升 11.1%。
- 当 HNSW 和 LEANN 都调到 90% Recall@3 时，LEANN 的下游准确率匹配 HNSW。

### 2.3 消融实验 (Ablation Study)

| Component / Configuration | Result | Notes |
|---|---:|---|
| Two-level search | 平均 1.4x speedup，最高 1.6x | 减少需要重算 embedding 的节点 |
| Two-level + dynamic batching | 平均 1.8x speedup，最高 2.0x | 提高 GPU 利用率；HotpotQA 收益最大 |
| High-degree preserving pruning | 边数减半后接近原始 HNSW | 原图 average degree 18，剪枝后 9 |
| Random Prune | 最多需要 1.8x 更多重算节点 | 随机删边破坏 navigability |
| Small M | 最多需要 5.8x 更多重算节点 | 94%/96% recall 目标下甚至失败 |
| Add operation optimization | 最高 63.3x speedup | 复杂度从含 \(efC^2\)、\(M^3\) 项降到 \(O(M\cdot efC)\) |
| Sharded index construction | 15 shards 约 5x 峰值存储下降 | k-means sharding 接近原始 HNSW graph quality |
| Smaller embedding model | GTE-small 带来 2.3x speedup | 在 2M datastore 上准确率与 Contriever 差距 2% 内 |
| Exact embedding cache | 缓存 10% embedding 得到 1.5x speedup | Cache hit rate 最高 41.9% |
| Latency breakdown | embedding recomputation 约 76.0% | Text + PQ lookup 8.0%，tokenize + distance 16.1% |

**消融结论**:

- LEANN 的最关键工程瓶颈是 embedding recomputation；two-level search、dynamic batching、小模型和 hot embedding cache 都围绕这个瓶颈展开。
- 图剪枝的关键不是“删边数量”，而是“保留导航 hub”。同样把 average degree 从 18 降到 9，随机删边和统一 Small M 都明显更差。
- 构建和更新不是论文主打卖点，但它们决定方案能否真正部署：如果只看查询阶段，可能会忽略构建峰值存储和动态数据问题。

---

## 三、业界类似方案 (Industry Similar Solutions)

### 3.1 方案对比表 (Solution Comparison Table)

| Solution | Year | Core Idea | Advantages | Disadvantages | Fit vs LEANN |
|---|---:|---|---|---|---|
| HNSW | 2018 | Hierarchical proximity graph + best-first search | 高召回、低延迟、工程成熟 | 存 full embeddings + graph metadata，RAM/disk 压力大 | LEANN 继承图搜索优势，但丢弃 full embeddings |
| FAISS IVF/PQ | 2017+ | 聚类倒排或 product quantization | 构建成熟，压缩简单 | 高压缩下召回/下游准确率下降 | LEANN 使用 PQ 只做 early filtering，不让 PQ 单独决定最终路径 |
| DiskANN / Vamana | 2019 | SSD-resident graph + PQ in memory | 高吞吐、适合大规模 ANN | 仍存 full vectors/sector-aligned layout，存储可很大 | LEANN 以重算换存储，适合低存储而非极致 QPS |
| SPANN / disk-based vector search | 2021+ | 内存索引 + SSD 存向量或 posting lists | 降 DRAM，支持大数据 | 瓶颈转为 SSD I/O，仍依赖保存向量 | LEANN 避免保存 exact embeddings |
| EdgeRAG / IVF-Recompute | 2024 | 查询时重算 embedding，通常基于 IVF | 存储极低 | IVF 需要 O(sqrt(N)) 级重算，延迟高 | LEANN 用 graph traversal 将重算规模降到 O(log N) 经验量级 |
| RabitQ / aggressive quantization | 2024 | 更强的 vector quantization | 存储压缩直接 | 高压缩仍有精度-召回 trade-off | LEANN 把 quantization 限定为候选过滤信号 |
| ObjectBox / MicroNN | 2025 附近 | 面向移动/边缘设备的向量检索 | on-device 友好 | 多数仍保存 embeddings | LEANN 更激进地把 exact embeddings 外移到 recomputation |

### 3.2 技术路线对比 (Technical Approach Comparison)

**路线 A：存 full embeddings，优化搜索路径**

- 代表：HNSW、NSG、Vamana/DiskANN、FAISS IndexHNSWFlat。
- 核心思路：把向量和图都保留下来，通过图连边或倒排结构减少距离计算。
- 优点：检索快，服务端成熟，适合低延迟高 QPS。
- 缺点：向量本体和 metadata 都占空间；个人设备或冷数据 lake 不划算。

**路线 B：压缩 embeddings**

- 代表：PQ、OPQ、scalar quantization、RabitQ。
- 核心思路：用较少 bits 表示每个向量，保留 compressed vectors。
- 优点：兼容传统索引，压缩比可控。
- 缺点：高压缩率下距离估计误差大；论文 Table 1 中 PQ 在 20GB 存储下准确率 17.9%，低于 BM25。

**路线 C：放到 SSD / disk-based ANN**

- 代表：DiskANN、SPANN、OpenSearch k-NN disk-based vector search RFC。
- 核心思路：内存只保存导航或压缩结构，full vectors/graph 放 SSD。
- 优点：降 DRAM，适合 billion-scale 服务端。
- 缺点：总存储未必低；I/O layout 很关键；DiskANN 在论文设置中因为 4KB sector padding 和额外 PQ 达到 270GB。

**路线 D：按需重算 embeddings**

- 代表：EdgeRAG, IVF-Recompute, LEANN。
- 核心思路：不保存 exact embeddings，而是在查询时用原始对象重新跑 embedding model。
- 优点：存储可以大幅下降。
- 缺点：延迟、算力、能耗转移到查询阶段；如果生成不慢或 QPS 很高，代价可能不可接受。

### 3.3 本文定位 (This Paper's Position)

- **相对 HNSW/DiskANN**：LEANN 不追求最低单次检索延迟，而是把索引存储从数百 GB 级降到数 GB 级，适合本地 RAG 和冷数据。
- **相对 PQ**：LEANN 不信任高压缩 PQ 的最终排序，只把 PQ 作为 cheap candidate filter，再用 exact recomputation 修正排序。
- **相对 IVF-Recompute/EdgeRAG**：LEANN 用 proximity graph 降低重算节点数，避免 IVF-Recompute 的 O(sqrt(N)) 级重算开销。
- **独特贡献**：把“embedding 重算”与“图索引导航”结合起来，并通过 hub-preserving pruning 让 graph metadata 也随之压缩。

### 3.4 推荐进一步阅读 (Recommended Further Reading)

| Paper / Project | Reason |
|---|---|
| HNSW: Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs | 理解 LEANN 基础图搜索结构 |
| DiskANN / Vamana | 对比 SSD-resident graph 与 LEANN recomputation 的存储-延迟取舍 |
| FAISS documentation and papers | 了解 IVF、PQ、HNSW 的工程实现 |
| EdgeRAG / IVF-Recompute | 对比 embedding online generation 路线 |
| RabitQ | 理解高压缩 quantization 的精度边界 |
| [OpenSearch k-NN disk-based vector search RFC](https://github.com/opensearch-project/k-NN/issues/1779) | 工程系统如何把 disk-based ANN 接入搜索引擎 |
| [LEANN GitHub repository](https://github.com/yichuan-w/LEANN) | 复现实验和查看当前工程形态 |

---

## 四、全文翻译 / 逐节译读 (Full Translation)

> 以下按论文原始结构做中文译读。公式、算法名和关键技术词保留英文；参考文献列表不逐条翻译。

### 摘要 (Abstract)

基于 embedding 的向量搜索支撑了推荐系统、RAG 等重要应用。它依赖向量索引实现高效搜索，但这些索引必须保存高维 embedding 和大量 metadata，总体大小可能是原始数据的数倍。高存储开销让向量搜索很难部署在个人设备或超大数据集上。

LEANN 的解决办法是：不保存所有 exact embeddings，而是在查询时按需重算；同时压缩先进的 proximity graph index，并尽量保持搜索准确率。LEANN 在真实 benchmark 上能用很小一部分存储，例如原始数据的 5%，保持高质量向量搜索；相对传统索引最高减少 50x index size，并在 RAG 应用中保持 SOTA 准确率和可比端到端延迟。

### 1. 引言 (Introduction)

基础模型推动 embedding model 变强，向量搜索成为内容搜索、个人助理、问答等应用的核心功能。对象被映射到高维向量空间后，语义相似对象的向量距离更近；query 也被编码成向量，再检索 top-k 相似向量。由于精确搜索代价太高，系统通常采用 ANNS，用 recall 衡量近似结果包含 ground-truth top-k 的比例。

论文首先用 Table 1 说明：在 RAG 问答中，HNSW 这类向量搜索能显著超过 BM25；但同一 76GB 文本库需要 173GB embeddings 和 15GB HNSW metadata。对本地 RAG 来说，这个成本过高。PQ 可以压缩 embedding，但要把 173GB 压到约 5GB 需要 35x 压缩，误差会让准确率低于 BM25。

关键观察是：RAG 中生成阶段经常主导端到端延迟。既然 Qwen3-4B 生成已经超过 20s，检索从毫秒增加到几秒并不一定不可接受。因此论文提出问题：能否设计一个大幅降低存储、保持准确率、并满足宽松延迟要求的向量索引？

LEANN 的答案有两个洞察。第一，图索引每个查询只访问少量节点，因此不必保存所有 embedding，可以沿图搜索时按需重算。第二，图中高出度 hub 节点被访问得更多，对导航更重要，所以图剪枝应保留 hub，而不是随机删边或统一限制所有节点出度。

### 2. 背景 (Background)

向量搜索的目标是在 \(N\) 个 \(d\) 维向量中找到离 query 最近的 top-k。RAG 一般要求较高 recall，例如不低于 0.9。索引的存储由两部分组成：vectors 和 metadata。IVF 通过聚类减少扫描范围；proximity graph 通过相似向量连边，再做 best-first traversal。图索引通常距离计算更少、召回-延迟表现更好。

图搜索维护一个大小为 \(ef\) 的 priority queue。每次取距离 query 最近且尚未访问的节点，展开其邻居，计算邻居与 query 的距离，再尝试插入 queue。\(ef\) 是质量旋钮，越大 recall 越高但计算越多。经验上图索引能用 O(log N) 级别的节点访问得到高 recall。

### 3. LEANN 概览 (LEANN Overview)

离线阶段，LEANN 对数据 chunk 计算 embedding 并构建 graph index；之后应用 high-degree preserving pruning，丢弃 dense embeddings，仅保留 pruned graph 和 PQ-compressed embedding table。若用户指定峰值存储预算，则用 graph partitioning/sharded build 顺序构建和合并 shards。

在线阶段，query 到来后，LEANN 在 pruned graph 上搜索。它先用 PQ approximate embeddings 计算便宜距离，再对最有希望的候选做 exact embedding recomputation。动态 batching 会把多个 graph hops 中的候选合并成 batch，提高 GPU 利用率。最后，系统用 exact distance 对访问节点排序并返回 top-k。

LEANN 的存储主要是两部分：CSR 格式 graph adjacency lists，规模约 \(O(N|D|)\)；以及比原始 FP32 embeddings 小 100x 的 PQ table，规模约 \(O(4N \cdot dim/100)\)。论文声称二者合起来相对 conventional dense indexes 最多节省 50x。

### 4. 基于图的重算 (Graph-based Recomputation)

**Two-Level Search** 的动机是：单纯用高压缩 PQ 跑图搜索会被量化误差带偏；先搜索后 rerank top-100 也无法补救遗漏的真邻居。因此 LEANN 交错使用 approximate 和 exact distance。每步先把邻居放入 AQ，用 PQ 算 approximate distance；再从 AQ 取 top \(\alpha\%\) 候选做 exact recomputation，进入 EQ，之后由 EQ 指导图遍历。

**Dynamic Batching** 的动机是：逐节点重算 embedding GPU 利用率很低，即使按当前节点邻居 batch，也受节点 degree 限制。LEANN 放松严格 best-first 顺序，跨多个 step 收集候选，达到目标 batch size 后统一重算。这样牺牲一点搜索顺序新鲜度，换来更高吞吐。

### 5. 紧凑图结构 (Compact Graph Structure)

虽然 LEANN 丢弃 exact embeddings，但 graph metadata 仍可能很大。论文把问题形式化为：在存储预算 \(B\) 和准确率阈值 \(\tau\) 下，最小化搜索中需要重算的节点数量。

随机删边和统一降低 degree limit 都会破坏图连通性。观察 HNSW 图后，作者发现高出度节点访问概率更高，是 graph traversal 的导航 hub。因此 LEANN 保留 top \(\beta\%\) 高度节点的更多连接；多数节点只保留较小度数 \(m\)，经验设置 \(m=M/5\)，并通过 offline profiling 选择 \(M\)。

### 6. 索引构建与更新 (Index Building and Update)

朴素构建需要先计算并保存所有 embeddings，构建完再丢弃，会导致峰值存储过高。LEANN 使用 sharded merging pipeline：先用小样本 k-means 得到 centroids；每个对象被 embedding 后分配到最近两个 centroids，随后丢弃 embedding；每个 shard 独立重算 embedding 并建图；最后合并 shard graphs。每个 passage 属于两个 shards，有助于保持全局连通性。

更新方面，LEANN 对单点 add 做缓存和简化 neighbor selection，把复杂度降到 \(O(M\cdot efC)\)。删除采用 soft deletion：节点打 inactive flag，但图结构保留，查询时仍可穿过 deleted nodes，最终结果过滤 inactive entries。删除比例过高时再后台 rebuild。

### 7. 实验 (Evaluation)

实验主数据集是 76GB RPJ-Wiki，切成 60M passages，生成 173GB Contriever embeddings。问答 benchmark 包含 NQ、TriviaQA、GPQA、HotpotQA；额外测试 FinanceBench、Enron 和 LAION。硬件包括 RTX 4090 + 32GB RAM，以及 AWS EC2 M1 Mac。

主要结果是：LEANN 在 Table 1 中用 4GB 存储达到 HNSW 的 25.5% downstream accuracy，而 HNSW 需要 188GB。LEANN 检索延迟更高，但生成阶段主导端到端耗时，因此整体 RAG 延迟仍可接受。Table 3 中，LEANN 在 FinanceBench、Enron、LAION 上相对 HNSW 节省 97%-98% 存储。

消融显示：two-level search 平均 1.4x 加速；加入 dynamic batching 后平均 1.8x、最高 2.0x；high-degree preserving pruning 在边数减半后接近原始 HNSW；随机删边和 Small M 都明显更差；add operation 优化最高 63.3x 加速；缓存 10% exact embeddings 带来 1.5x 加速。

### 8. 相关工作 (Related Work)

资源受限向量搜索有几类路线。DiskANN 等 disk-based systems 把 full vectors/graph 放 SSD，用压缩向量在内存导航；Starling、FusionANNS、AiSAQ、LM-DiskANN 继续降低 I/O 或 DRAM 成本。EdgeRAG 也做 online embedding generation，但基于 IVF，重算开销更高。PQ、RabitQ 等压缩向量，但紧预算下会损失准确率。ObjectBox、MicroNN 等面向个人设备优化，但通常仍保存 all embeddings。LEANN 的差异在于把按需重算、pruned graph 和图遍历优化一起使用。

### 9. 结论 (Conclusion)

LEANN 解决的是高维 embedding search 在本地和低存储场景中的索引膨胀问题。它通过 two-level search、dynamic batching、high-degree preserving pruning、storage-efficient build 和 update pipeline，使索引小于原始数据 5%，并相对现有方法最多减少 50x 存储，同时保持高 recall 和较低 RAG 端到端延迟。

### 附录要点 (Appendix Highlights)

- RNG pruning 用相对邻域图规则去掉三角形中的长边，是现代 graph ANN 常见稀疏化方法。
- Add operation 的 naive complexity 包含 \(O(M\cdot efC + efC^2 + M^3)\)，通过 cache 和简化 RNG 降到 \(O(M\cdot efC)\)。
- Mac 平台上 HNSW/IVF 因 OOM 省略；LEANN 仍能跑，但 HotpotQA retrieval overhead 达到 50.1%，说明低算力平台上的复杂查询有明显边界。
- GTE-small 替换 Contriever 在 2M datastore 上带来 2.3x latency speedup，准确率保持在 2% 以内，说明模型选择对 LEANN 很关键。
- Latency breakdown 表明 embedding recomputation 占 76%，未来优化应优先围绕模型推理、batching、overlap 和 cache。

---

## 五、评价与局限

### 5.1 我认为最有价值的点

- **系统取舍非常清晰**：它没有把自己包装成“所有向量检索都更快”，而是明确针对 RAG/个人设备/冷数据：用几秒检索换几十倍存储。
- **设计有组合性**：PQ 只做过滤，exact recomputation 做校正，图剪枝保留 hub，dynamic batching 补 GPU 利用率；每个组件都针对前一个组件带来的副作用。
- **实验抓住了真实部署痛点**：不仅测 query latency，还测 storage size、generation-dominated E2E latency、构建峰值存储、update、Mac 平台。

### 5.2 需要谨慎的地方

- **不适合极低延迟/高 QPS 服务端检索**：HNSW/DiskANN 的检索是几十毫秒以内，LEANN 常常是秒级。
- **前提是原始对象可访问且 embedding model 可本地运行**：如果原始数据读取慢、模型推理慢、tokenization 慢，优势会被吃掉。
- **能耗和并发成本没有充分讨论**：把存储换成计算后，电力、GPU 占用和多用户并发会成为新瓶颈。
- **召回 ground truth 使用 exact search proxy**：这是 ANN 常见做法，但不等于真实问答 relevance label。
- **soft delete 会积累垃圾节点**：论文建议超过阈值后台 rebuild，但具体策略留给未来工作。
- **HotpotQA/Mac 结果显示边界明显**：复杂 multi-hop 或低算力平台上，检索 overhead 可超过正文中“通常低于 20%”的直觉。

### 5.3 适用场景判断

**适合**:

- 本地个人知识库：邮件、文件、浏览器历史、聊天记录、代码库。
- 冷数据 RAG：偶尔查询的大规模日志、文档湖、归档资料。
- 隐私敏感场景：不希望把数据或 embeddings 上传云端。
- 生成很慢的 agentic/RAG workflow：检索几秒不主导总延迟。

**不适合**:

- 搜索引擎级高 QPS、p99 latency 严格的在线服务。
- 查询远多于构建且有充足内存/SSD 的服务端向量库。
- 原始数据无法保留或 embedding 模型推理成本极高的场景。
- 需要频繁硬删除且不能后台 rebuild 的动态索引。

---

## 附录 (Appendix)

### A. 术语表 (Glossary)

| English Term | Chinese Translation | Explanation |
|---|---|---|
| ANNS | 近似最近邻搜索 | 用近似结果换低延迟的 top-k 向量检索 |
| HNSW | 分层可导航小世界图 | 高性能 graph-based ANN 索引 |
| Proximity Graph | 近邻图 | 节点是向量，边连接相似向量 |
| PQ | 乘积量化 | 将向量分块后量化压缩 |
| Recall@K | K 召回率 | 返回结果包含 ground-truth top-k 的比例 |
| Reranking Ratio \(\alpha\) | 重排比例 | LEANN 中从 AQ 取多少比例候选做 exact recomputation |
| Dynamic Batching | 动态批处理 | 跨图搜索步骤收集待重算节点批量推理 |
| High-degree Hub | 高度枢纽节点 | 图中出度高、搜索中经常被访问的导航节点 |
| CSR | 压缩稀疏行格式 | 保存图邻接表的紧凑格式 |
| Soft Delete | 软删除 | 标记节点 inactive，但保留图结构 |

### B. 复现检查清单 (Reproducibility Checklist)

- [x] Code open-sourced: Yes, [GitHub](https://github.com/yichuan-w/LEANN)
- [x] Data described: RPJ-Wiki, NQ, TriviaQA, GPQA, HotpotQA, FinanceBench, Enron, LAION
- [x] Hardware described: RTX 4090 workstation and AWS EC2 M1 Mac
- [x] Baseline configurations partially provided: Appendix C.1 includes HNSW M=30/efConstruction=128, IVF nlist=sqrt(N)=8192, DiskANN M=60/efConstruction=128, PQ compressed to 5GB
- [x] Evaluation protocol provided: binary search over ef to reach target Recall@3; average latency over 20 random queries
- [ ] Random seeds: Not provided in original
- [ ] Full hyperparameters for all system knobs: Partially provided; some values chosen by offline profiling, not fully enumerated
- [ ] Exact scripts for all paper figures: Code repository exists, but this report did not verify one-command figure reproduction

### C. 关键信息来源

- Paper: [LEANN: A Low-Storage Vector Index](https://arxiv.org/abs/2506.08276)
- Code: [yichuan-w/LEANN](https://github.com/yichuan-w/LEANN)
- Related engineering context: [OpenSearch k-NN disk-based vector search RFC](https://github.com/opensearch-project/k-NN/issues/1779)
