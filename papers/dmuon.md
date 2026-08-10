# DMuon：近 Adam 开销的高效分布式 Muon 训练
# DMuon: Efficient Distributed Muon Training with Near-Adam Overhead

> **arXiv/DOI:** [2606.27153v1](https://arxiv.org/abs/2606.27153) · **PDF:** https://arxiv.org/pdf/2606.27153  
> **发表信息:** arXiv preprint（2026-06-25）  
> **机构:** X Square Robot Team（中国）  
> **代码:** https://github.com/X-Square-Robot/dmuon  
> **领域:** 分布式训练 · 优化器系统 · Muon · FSDP/ZeRO · Newton-Schulz  
> **核心贡献:** **Owner-centric 分布式 Muon** + Gram-space 对称 NS kernel + 实测代价 MILP 负载均衡 + 分层通信 pipeline，使端到端 step time 平均仅比 AdamW **+2%**，相对 naive gather-then-compute Muon 优化器步加速 **6.85–163×**、端到端 **1.48–3.01×**。

---

## 一、问题分析 (Problem Analysis)

### 1.1 研究背景 (Research Background)

**领域现状** (Current State of the Field):
- 自 AdamW 以来，大模型训练长期依赖 **逐元素（element-wise）** 优化器；参数 shard 与 optimizer step 一一对应是 ZeRO/FSDP/DeepSpeed 的隐含契约。
- **Muon**（MomentUm Orthogonalized by Newton-Schulz, Jordan et al. 2024）对 2D 权重矩阵的 momentum 梯度做 Newton-Schulz 迭代，近似 matrix sign / 正交因子 $UV^\top$，使更新奇异值谱趋近均匀。
- Moonlight（Kimi）报告 Muon 在 compute-optimal pretraining 下约 **2× token 效率** vs AdamW；Kimi-K2、DeepSeek-V4 等已在生产级训练采用 Muon。
- 分布式训练栈（ZeRO、FSDP2/HSDP、Tensor Parallelism）按 **storage/semantic 轴** 切分矩阵，optimizer 在 local shard 上独立更新。

**核心挑战** (Core Challenges):
- **粒度失配（granularity mismatch）**：Newton-Schulz 每步需 $XX^\top$，耦合矩阵 **全部行**；无法在 row-wise shard 上局部计算，必须先 **materialize 完整 reduced gradient**。
- **Naive gather-then-compute**：每 rank all-gather 全矩阵 → 各 rank **冗余执行相同 NS** → 优化器计算量 × DP 宽度；实测 vanilla distributed Muon 可 **>2× forward+backward 总时间**。
- **VLA/机器人模型更敏感**：相比 LLM，VLA 训练 context 短、F+B 占 step 比例小，optimizer overhead 更难摊销（Wall-OSS、Pi0 等）。
- **NS 本身计算重**：$k=5$ 步 NS，每步大矩阵乘法；rectangular 矩阵 $m \ll n$ 时直接在 $m \times n$ 空间迭代代价 $O(m^2 n)$。

**研究动机** (Research Motivation):
- Muon 的 **算法收益**（更快收敛、更少 token）已被验证，但 **系统部署成本** 抵消 wall-clock 优势。
- 需要 **不改变 Muon 更新语义** 的前提下，把 per-step 开销压到 **near-AdamW**，且 **drop-in** 现有 FSDP2 训练管线（无需改框架源码）。

### 1.2 问题定义 (Problem Definition)

**具体问题** (Specific Problem):

- **输入**：sharded 训练中的 2D 权重 $W \in \mathbb{R}^{m \times n}$ 及其各 rank 梯度 shard $\{g_r\}$；momentum $M_t$；DP/TP mesh。
- **输出**：与 synchronous full-matrix Muon **数学等价** 的 $W_{t+1} = W_t - \eta \cdot \mathrm{NS}_k(M_t)$。
- **约束**：
  - 保留 FSDP2 式 transient materialization（峰值内存不爆炸）；
  - 与 TP 可组合（矩阵在 TP 组内仍 sharded）；
  - 非矩阵参数（embedding、bias 等）仍走 host stack 的 AdamW。
- **评估指标**：optimizer step time、end-to-end step time、相对 AdamW 的 $\Delta_A$、scaling 至 256 GPU。

**问题形式化** (Problem Formalization):

Synchronized Muon 需先聚合梯度：
\[
M_t = \text{momentum}\left(\frac{1}{D}\sum_{r=1}^{D} g_r\right), \quad W_{t+1} = W_t - \eta \cdot \mathrm{NS}_k(M_t)
\]

Newton-Schulz 迭代（式 2）：
\[
X_{i+1} = aX_i + b X_i X_i^\top X_i + c (X_i X_i^\top)^2 X_i
\]

Naive 分布式代价：每矩阵每步 **$O(D \cdot \text{comm}(mn) + D \cdot \text{NS}(m,n))$**（materialization + 冗余 NS）。

DMuon 目标：每矩阵 **单次 owner-side NS** + 通信与 F+B **overlap**，使
\[
T_{\text{step}}^{\text{DMuon}} \approx T_{\text{step}}^{\text{AdamW}} \cdot (1 + \epsilon), \quad \epsilon \approx 2\%
\]

### 1.3 解决方案 (Solution)

**核心思路** (Core Idea):

**每个矩阵参数指定唯一 owner rank**：owner 持有 authoritative 参数与 optimizer state，接收 reduced full-matrix gradient，**执行一次** Gram-space Newton-Schulz；非 owner 仅在 F+B 时 transient materialize 参数。通信采用 **分层（intra-node / inter-node）+ pipeline overlap**；NS 采用 **Gram 空间递推 + SYRK 对称 kernel + shape batching + DSL autotune**；owner 分配通过 **实测代价 MILP** 最小化 makespan。

**方法概述** (Method Overview):

1. **Setup（一次性）**：`dedicate_params(model, mesh)` → MILP/greedy owner 分配；owner 分配 $W^{(p)}, M^{(p)}$；非 owner 换 zero-size placeholder。
2. **Forward**：分层 broadcast（inter-node lookahead + intra-node just-in-time）materialize 层参数到 packed buffer；与 forward compute pipeline overlap。
3. **Backward**：同样 pipeline materialize；gradient reduce 到 owner（Avg）；与相邻层 backward overlap。
4. **Optimizer（owner-only）**：Gram NS$_k$ → 更新 $W^{(p)}$；非矩阵参数走 sharded AdamW。
5. **Publish（async）**：owner 更新后异步 broadcast 新权重；下一步仅在消费层 pre-forward hook 等待。

**技术细节** (Technical Details):

*Owner-Centric Communication*:
- **Function:** 消除冗余 NS；将 all-rank collective 转为 owner↔many 非对称通信。
- **Implementation:** XOR owner-slot layout（式 3）：$\mathrm{gpu}(w)=w\bmod 8$，$\mathrm{node}(w)=(w\bmod 4)\oplus(\lfloor w/8\rfloor\bmod 4)$；分散 concurrent collective 争用。
- **Innovation:** Forward inter-node 提前 broadcast + intra-node 延迟；Backward 重排为 $\mathrm{bcast}_{inter}\!\to\!\mathrm{bcast}_{intra}\!\to\!\mathrm{compute}\!\to\!\mathrm{reduce}_{intra}\!\to\!\mathrm{reduce}_{inter}$ 并与相邻层 overlap。

*Gram Newton-Schulz*:
- **Function:** 降 NS 算术复杂度与 kernel 效率。
- **Implementation:** Gram 递推 $G_{i+1}=P_i G_i P_i$，$G_i=X_i X_i^\top \in \mathbb{R}^{m\times m}$；$m<n$ 时 $O(m^3)$ vs $O(m^2 n)$。SYRK 只算下三角 + epilogue 重建上三角；小矩阵 shape batching；TileLang/CUTE DSL autotune + persistent cache。
- **Innovation:** 对称 Gram kernel 贡献 **48%** optimizer speedup；fp16 NS + fp32 master weight update。

*Computation-Aware Load Balancing*:
- **Function:** 避免 owner straggler。
- **Implementation:** 按 shape 分组；benchmark 各 $(s,b)$ 的实测 $c_{s,b}$；MILP（式 5）最小化 $\max_r \sum c_{s,b} x_{s,b,r}$；搜索空间过大时 greedy fallback。
- **Innovation:** 考虑 batching/autotune 效应，不用 FLOPs 解析 proxy；贡献 **32%** speedup。

*TP Composition*:
- **Function:** 与 Megatron TP 共存。
- **Implementation:** DP owner 内再 designate TP owner；TP owner gather gradient slices → full NS → scatter update slices。
- **Innovation:** TP 处理 **仅 confined to optimizer step**；F+B broadcast/reduce 不改 host TP 路径。

**算法/架构描述** (Algorithm/Architecture Description):

```
对比（Figure 3）:

Vanilla Muon-AG (每 rank):
  all_gather(G) → NS_k(G) on ALL ranks  [冗余 × D]

DMuon (每 matrix):
  reduce_to_owner(G) → NS_k(G) on OWNER only → async publish(W')

单 step 四阶段（Algorithm 1）:
  Forward mat.  →  Backward reduce  →  Owner GramNS  →  Async publish
       ↑___________________pipeline overlap___________________↑
```

用户 API（3 行）：
```python
import dmuon
dmuon.dedicate_params(model, mesh)
opt = dmuon.Muon(model, lr=0.02, ns_steps=5, adamw_lr=1e-3)
```

实现：~10K 行 Python + custom CUDA kernels；与 stock FSDP2 组合，无框架源码修改。

---

## 二、实验效果 (Experimental Results)

### 2.1 实验设置 (Experimental Setup)

| Item | Details |
|------|---------|
| Workloads | **Wall-OSS**（VLA/具身基础模型）、**Pi0**（机器人策略）、**Wall-WM**（世界模型）、**Qwen2.5-7B**（LLM） |
| Baselines | **AdamW**（主 baseline）；**Muon-AG**（vanilla gather-then-compute distributed Muon） |
| Metrics | Optimizer step time (ms)、End-to-end step time (ms)、$\Delta_A$ vs AdamW、Speedup vs Muon-AG |
| Hardware | **A800-SXM4-80GB**；8 GPU/node NVLink + **200 Gb/s IB** 跨节点；bf16 |
| Scale | 8 / 16 / 32 / 64 / 128 / **256** GPUs |
| NS config | Polar Express 系数，$k=5$；NS 在 fp16，master weight fp32 |

### 2.2 主要结果 (Main Results)

**核心性能指标** (Core Performance Metrics):

**vs AdamW（$\Delta_A$，端到端 step time 相对增量）**

| Model | GPUs | DMuon Step (ms) | AdamW Step (ms) | $\Delta_A$ |
|-------|------|-----------------|-----------------|------------|
| Wall-OSS | 8–256 | 1359–1519 | 1324–1496 | **0.7%–2.7%** |
| Pi0 | 8–256 | 1597–1648 | 1498–1637 | **0.6%–6.6%** |
| Wall-WM | 8–256 | 2787–3011 | 2539–2915 | **3.3%–17.6%** |
| Qwen2.5-7B | 8–256 | 2636–2850 | 2590–2844 | **0.2%–4.8%** |

**全文结论：平均端到端 step time overhead 在 AdamW 的 +2% 以内。**

**vs Muon-AG（Table 1 精选）**

| Model | GPUs | Optim Speedup | E2E Speedup | DMuon Optim (ms) | Muon-AG Optim (ms) |
|-------|------|---------------|-------------|------------------|---------------------|
| Wall-OSS | 256 | **109.44×** | **1.88×** | 18 | 1977 |
| Pi0 | 256 | **93.43×** | **1.62×** | 14 | 1308 |
| Wall-WM | 256 | **96.89×** | **3.01×** | 64 | 6265 |
| Qwen2.5-7B | 256 | **163.82×** | **2.18×** | 22 | 3604 |

**范围汇总（Abstract / §5.1）**：
- 端到端 step：**1.48× – 3.01×** vs Muon-AG
- Optimizer step：**6.85× – 163.00×** vs Muon-AG

**关键发现** (Key Findings):
- DMuon 将 Muon 的 **可扩展系统开销** 移除，剩余 vs AdamW 的差距主要是 **最大矩阵的单次 NS 不可消除关键路径**（算法固有成本）。
- **256 GPU scaling**：Wall-OSS optimizer 从 Muon-AG 1977ms → DMuon 18ms；E2E 2857ms → 1519ms（仍仅比 AdamW 1496ms 慢 1.5%）。
- **小 GPU 数时 DMuon 可快于 AdamW**：Muon 允许更大 batch（同内存），Wall-OSS 8 GPU 时 DMuon 1359ms vs AdamW 1324ms 但吞吐更高（Figure 8）；随 scale 增大该优势收敛。
- **Wall-WM $\Delta_A$ 偏高**（8 GPU 17.6%）：WM 模型 optimizer 占比更大，但 vs Muon-AG 仍有 2×+ E2E 加速。

### 2.3 消融实验 (Ablation Study)

**Table 2：Wall-OSS-0.5 @ 128 GPU，optimizer-step speedup 归因**

| Configuration | Share of speedup | Notes |
|---------------|------------------|-------|
| Symmetric Gram kernel | **48%** | SYRK 半算术 × 5 NS steps |
| Owner scheduling & load balancing | **32%** | 消除 $D×$ 冗余 NS + MILP 防 straggler |
| Auto-tuning & NS batching | **16%** | shape-specific kernel cache + 小矩阵 batch |
| **合计 → E2E vs AdamW** | **+2% avg** | 端到端额外开销 |

**Ownership strategy ablation（§4）**：
| Strategy | 效果 |
|----------|------|
| `load_balance`（MILP） | 生产默认 |
| `round_robin` | 忽略 shape 代价差异 |
| `rank0`（全矩阵 rank0 owner） | 极端 straggler，用于 ablation |

**消融结论** (Ablation Conclusions):
- **对称 Gram kernel 是单一最大杠杆**（48%），其次 owner+LB（32%）。
- 去掉 load balancing 会导致大矩阵集中到少数 owner → optimizer step 由最慢 rank 决定。
- 单 GPU 仍有 ~2× optimizer speedup（kernel 优化），但无法获得分布式通信优化收益（§5.3 limitation）。

---

## 三、业界类似方案 (Industry Similar Solutions)

### 3.1 方案对比表 (Solution Comparison Table)

| Solution | Year | Core Idea | Advantages | Disadvantages | Performance |
|----------|------|-----------|------------|---------------|-------------|
| Moonlight Muon | 2025 | ZeRO-1 式 distributed Muon PoC | 证明 LLM 可 scale | gather-then-compute；非 production runtime | ~2× token efficiency vs AdamW |
| Muon-AG (vanilla) | — | 每 rank all-gather + 本地 NS | 语义正确、实现简单 | 冗余 NS × D；>2× F+B | 本文主要负 baseline |
| Distributed Shampoo | 2023 | owner-compute + all-gather direction | 矩阵优化器分布式先例 | Shampoo 状态更大 | 不同 optimizer |
| Canzona | 2026 | $\alpha$-balanced DP + TP micro-group scheduling | Megatron 栈 256 GPU 1.57× E2E | 非 Muon 专用 | Qwen3-32B |
| MatrixFSDP | arXiv'26 | 改 ZeRO-3 shard 布局：2D weight 整矩阵 owner | 无 optimizer-step matrix collective | 需改 FSDP shard 语义；uneven layout | 64 A100 4.2× optim step vs FSDP2-Muon |
| **DMuon** | 2026 | **FSDP2 外挂 owner runtime + Gram NS + MILP LB** | **3 行 drop-in；+2% vs AdamW；开源** | 不改 Muon 算法；大矩阵 NS 仍有关键路径 | **1.48–3.01× E2E vs Muon-AG** |

### 3.2 技术路线对比 (Technical Approach Comparison)

**路线A：Gather-then-Compute（冗余计算）**
- Representative works: Moonlight PoC, naive Muon-AG, 社区 FSDP Muon PR
- Core idea: 每步 all-gather 全梯度 → 每 rank 跑相同 NS → 保留 local shard。
- Pros and cons: (+) 正确、易实现；(−) NS 计算 × DP 宽度；通信 + 计算双重复。

**路线B：Owner-Compute + 改 Shard 布局（MatrixFSDP）**
- Representative works: MatrixFSDP (arXiv:2607.05895), Distributed Shampoo
- Core idea: 从 **参数分片策略** 入手，让 full matrix 自然落在 owner 上；F+B 仍 reshard。
- Pros and cons: (+) optimizer step 无 matrix collective；(−) 需深度改 FSDP/ checkpoint；内存布局 uneven。

**路线C：Owner-Compute + 通信 Runtime（本文 DMuon）**
- Representative works: DMuon, Canzona（部分重叠）
- Core idea: **不改 host FSDP 分片**；外挂 owner 状态 + 分层 pipeline 通信 + overlap。
- Pros and cons: (+) drop-in、exact Muon 语义、生产验证（Wall-OSS/WALL-WM）；(−) 额外通信 runtime 复杂度；Wall-WM 小集群 $\Delta_A$ 仍可达 17%。

### 3.3 本文定位 (This Paper's Position)

- **Improvement over Approach A**: optimizer step **6.85–163×**；消除 $D×$ 冗余 NS；通信 pipeline overlap。
- **Improvement over Approach B**: **无需修改 FSDP2 源码或 shard 契约**；与 TP/HSDP 正交组合。
- **Unique contributions**: XOR fine-grained owner layout + F/B 双阶段 pipeline；**实测代价 MILP**（非 FLOPs proxy）；Gram SYRK + shape autotune 生产 kernel stack。

### 3.4 推荐进一步阅读 (Recommended Further Reading)

| Paper | Reason |
|-------|--------|
| [Muon (Jordan et al.)](https://kellerjordan.github.io/posts/muon/) | 算法原典：NS 迭代、spectral scaling |
| Moonlight (arXiv:2502.16982) | Muon LLM scale 的首个系统证明 |
| MatrixFSDP (arXiv:2607.05895) | 另一条「改 shard 而非改 runtime」的 Muon 系统路线 |
| Gram Newton-Schulz (Dao et al. 2026) | DMuon NS kernel 的 Gram 空间理论基础 |
| Polar Express (2026) | DMuon 默认 NS 系数集 |
| [`veScale FSDP`](./vescale-fsdp.md) | 理解 FSDP2/DTensor 分片契约的背景 |

---

## 四、全文翻译 (Full Translation)

> 以下为论文主要内容的中文翻译，保持原有结构。技术术语首次出现标注英文原文。

### 摘要 (Abstract)

基于矩阵正交化的优化器以 Muon 为代表，已在多种现代深度学习 workload 中展现强收敛行为。矩阵感知（matrix-aware）更新为传统逐元素优化提供了有吸引力的替代，尤其当模型架构在规模与异构性上持续增长时。然而，围绕逐元素优化器假设构建的当代分布式训练基础设施，与 Muon 等矩阵级优化器 poorly matched——其更新耦合整个权重矩阵并需代价高昂的 Newton-Schulz 迭代。Vanilla Muon 实现的开销 **超过前向与反向 pass 总和的 2 倍**。为缩小差距，我们提出 **DMuon**——开源分布式 Muon 实现，作为 **drop-in 模块** 集成到现有训练管线，**无需框架级修改**。在具身基础模型与 LLM 训练 workload 上，DMuon 实现 **1.48×–3.01×** 端到端 step 加速与 **6.85×–163.00×** 优化器步加速，将 per-step 延迟带到 **near-AdamW** 水平。

### 1. 引言 (Introduction)

大模型训练开始超越自 AdamW 以来主导的逐元素优化范式。Muon 对每个权重矩阵的 momentum 聚合梯度应用 Newton-Schulz 迭代，产生奇异值被驱动向近均匀谱的更新。Moonlight 报告约 **2× compute efficiency**；Kimi-K2、DeepSeek-V4 在生产级训练采用 Muon。算法 promise 已确立，**部署成本** 仍是重大 practical obstacle。

现代分布式栈在 each parameter 的 shard 上执行 optimizer step。隐含契约：optimizer 更新规则是 **element-wise**——AdamW 对 shard 更新无需知其他 shard。Muon 打破该契约：Newton-Schulz 在 **完整权重矩阵** 上操作，必须先 **重建矩阵**，引入逐元素优化器所无的通信。该通信出现在 **每个 optimizer step、每个矩阵参数**，随模型规模与分布式宽度 scaling。Distributed Muon 代价可 **rival 或超过 F+B 总和**。

对具身模型训练，惩罚尤其显著：VLA 的 temporal context 远短于 LLM/VLM pretraining，F+B 占 step 比例小，optimizer overhead 更难摊销。

**贡献：**
1. 细粒度通信优化：owner-centric 策略，最小化通信开销，保留 exact semantics。
2. Shape-adaptive 执行栈：batched Gram NS + 对称感知 kernel + DSL autotune。
3. 计算感知负载均衡：实测代价 MILP owner 分配。
4. Drop-in 模块：与 host 框架/并行策略正交。
5. 生产级验证：Wall-OSS-0.5、WALL-WM 等真实训练。

### 2. 背景与动机 (Background and Motivation)

#### 2.1 Muon 与 Newton-Schulz

对 $W \in \mathbb{R}^{m \times n}$，$W_{t+1} = W_t - \eta \cdot \mathrm{NS}_k(M_t)$。每 NS 步（式 2）主要为大型矩阵乘法；$k=5$ 通常足够；可在降精度执行（只需保 singular subspace）。

#### 2.2 分片训练抽象

- **ZeRO**：flatten 参数 buffer 再 shard；边界由 storage 定，单矩阵可跨 rank。
- **FSDP2**：每参数 DTensor 沿 leading dim shard；边界对齐 tensor 结构；HSDP 为 $D_{shard} \times D_{replica}$ mesh。
- **TP**：沿 semantic 轴（head/hidden）shard；partition 是模型定义一部分。

#### 2.3 问题与机遇

**粒度失配**：训练系统暴露 matrix pieces，Muon 需 full reduced gradient $M = \frac{1}{D}\sum g_r$ 再跑 NS。

**Gather-then-compute** 的两项代价：
1. **Matrix materialization**：每步重建 full matrix；通信量 ∝ 矩阵大小 × 宽度。
2. **Replicated orthogonalization**：每 rank 跑相同 NS；优化器计算 × ranks。

Owner-like execution 可缓解 replication，但通信与执行 overhead 成为新瓶颈——DMuon 用 multi-level co-design 解决。

### 3. 系统设计 (System Design)

#### 3.1 设计概览

DMuon 每矩阵指定 **single owner rank**；owner 维护 authoritative state 并 **compute Muon update once**；非 owner 仅在 F+B 需要时 materialize。

四阶段：(1) Parameter materialization；(2) Gradient routing；(3) Owner-side Muon update；(4) Async publication。

#### 3.2 Owner-Centric 通信优化

**3.2.1 Fine-Grained Weight Layout**

XOR 映射（式 3）将 consecutive matrices 分散到不同 inter-node column，减少 collective contention（Figure 4）。

**3.2.2 Forward Overlap**

Inter-node broadcast **lookahead** 提前发起；intra-node broadcast 延迟到接近 consumption。Layer $l$ compute 时 materialize $l+1$ 并 launch 更远层 inter-node publication（Figure 5a）。保留 FSDP transient materialization 的低峰值内存。

**3.2.3 Backward Overlap**

相邻层 backward 的 materialize 与 reduce **独立**，pipeline 调度：while reducing layer $l$ gradient，broadcast layer $l-1$ parameters。Intra-node bcast/reduce 有序避免 interference；inter-node 因 layout 可并发（Figure 5b）。

#### 3.3 高效 Gram Newton-Schulz

采用 Dao et al. Gram NS：$G_{i+1}=P_i G_i P_i$，$G_i=X_i X_i^\top$，$m<n$ 时 $O(m^3)$ vs $O(m^2 n)$。

- **Batching**：小矩阵 occupancy 不足时按 shape 分组 batched iteration（Figure 7：$1024×1024$ batch 16 可达 3×/matrix）。
- **SYRK 对称路径**：只算 $G$ 下三角，epilogue 重建上三角 + fuse 逐元素操作；算术近减半。
- **Autotune**：TileLang/CUTE DSL 生成 schedule 变体；首次 benchmark 后 persistent cache（Figure 6）。

#### 3.4 计算感知负载均衡

按 shape 分组；benchmark 得 $c_{s,b}$；MILP（式 5）最小化 owner makespan；变量超 $S_{thr}$ 时 greedy fallback。训练期间 shape 固定，**仅 init 一次**。

#### 3.5 DMuon Training Step

Algorithm 1：Setup → Forward（$\mathcal{S}_{bc}$ materialize + forward）→ Backward（materialize + backward + $\mathcal{S}_{rd}$ reduce）→ Owner GramNS → Async publish。语义：每 owner 收到与 synchronous Muon 相同的 $\bar{g}^{(p)}$，应用相同 NS update。

### 4. 实现 (Implementation)

~10K 行 Python + custom kernels；三 API：`dedicate_params`, `Muon`, state-dict accessors。与 stock FSDP2 组合。

**TP**：DP owner 内 designate TP owner；gather TP gradient slices → NS → scatter slices。**Non-owner placeholder**：zero-size tensor 保留 module graph traversal。**NS fp16**，master weight fp32；Polar Express 默认系数。

### 5. 实验 (Evaluation)

A800-80GB，bf16，8–256 GPU。四模型 Table 1 结果见 §2.2。平均 **+2% vs AdamW**；vs Muon-AG **1.48–3.01× E2E, 6.85–163× optim**。Table 2 组件分解：Symmetric Gram 48%，Owner+LB 32%，Autotune+batching 16%。

**Limitation**：数学等价 reformulation，不改 update rule；单 GPU 无分布式通信优化收益（仍有 ~2× optim kernel speedup）。

### 6–7. 相关工作与结论 (Related Work & Conclusion)

DMuon 位于 matrix-aware optimizer、分布式实现、NS hardware-aware kernel 三线交叉。不改 Muon 算法，解决 **系统问题**。与 MatrixFSDP/Canzona/Distributed Shampoo 同属 owner-compute 家族但 **针对 FSDP2/HSDP/TP drop-in**。

结论：DMuon 使 Muon 成为 **practical drop-in AdamW replacement**，平均端到端 **within 2% of AdamW**。

### 参考文献 (References)

关键引用择译：
- Jordan et al., Muon — 原始矩阵正交化优化器
- Moonlight — Muon LLM 可扩展性
- Rajbhandari et al., ZeRO — 分片训练基础
- Zhao et al., FSDP2 — PyTorch DTensor 分片
- Dao et al., Gram Newton-Schulz — Gram 空间 NS
- Gupta et al., Shampoo — 矩阵预条件优化器先驱

---

## 附录 (Appendix)

### A. 术语表 (Glossary)

| English Term | Chinese Translation | Explanation |
|--------------|---------------------|-------------|
| Muon | — | MomentUm Orthogonalized by Newton-Schulz |
| Newton-Schulz (NS) | 牛顿-舒尔茨迭代 | 近似 matrix sign / 正交因子 |
| Gram Newton-Schulz | Gram 空间 NS | 在 $XX^\top$ 空间递推，降复杂度 |
| Element-wise optimizer | 逐元素优化器 | AdamW 等；shard 独立更新 |
| Granularity mismatch | 粒度失配 | 分片 vs 全矩阵 NS 的结构冲突 |
| Owner rank | 属主 rank | 每矩阵唯一负责 NS 的 rank |
| Muon-AG | — | Vanilla all-gather then compute baseline |
| FSDP2 / HSDP | — | PyTorch 原生分片数据并行 |
| SYRK | 对称秩-k 更新 | 只算 $XX^\top$ 下三角 |
| MILP | 混合整数线性规划 | Owner 分配 makespan 优化 |
| $\Delta_A$ | 相对 AdamW 开销 | $(Step_{DMuon}-Step_{AdamW})/Step_{AdamW}$ |

### B. 复现检查清单 (Reproducibility Checklist)

- [x] Code open-sourced: **Yes** — https://github.com/X-Square-Robot/dmuon
- [x] Data available: 模型权重/训练配置见 X-Square-Robot/wall-x 等
- [x] Hyperparameters complete: **Partial** — `lr=0.02, ns_steps=5, adamw_lr=1e-3` 在 README；各模型 full config 在 wall-x
- [ ] Random seeds: 原文未强调
- [x] Hardware requirements: A800-80GB×N（8/node NVLink + 200G IB）；bf16；FSDP2 mesh

### C. Limitations · Our take

**局限**
- **不改 Muon 算法**：最大矩阵 NS 仍是 irreducible critical path；无法 beat AdamW optim step 绝对时间。
- **Wall-WM 小集群 $\Delta_A$ 高**：8 GPU 时 +17.6% vs AdamW（WM 优化器占比大）。
- **Init 开销**：MILP profiling + autotune 在首次见 shape 时发生（训练期 amortize）。
- **仅矩阵参数用 Muon**：embedding/bias/LayerNorm 仍 AdamW（Hybrid MuonW 模式）。
- **与 MatrixFSDP 路线竞争**：后者改 shard 语义、前者改 runtime；长期可能收敛或共存。

**Our take**
- DMuon 把 Muon 从「算法 curiosity」推到 **production drop-in**：3 行代码 + FSDP2 兼容，对 VLA/LLM 训练栈直接可用。
- **Owner-centric + pipeline 通信** 与 MoE combine overlap（[`moe-tile-signaling.md`](./moe-tile-signaling.md)）同属「矩阵级操作 vs 分片 runtime 失配」问题族，但 DMuon 解决的是 **optimizer step** 而非 forward A2A。
- **48% speedup 来自 SYRK** 说明：Muon 系统优化的主战场是 **kernel 算术效率**，而非仅通信；这与 Comet/MegaScale 强调 comm overlap 形成对照。
- 若已用 MatrixFSDP，需评估 DMuon 的 drop-in 优势是否超过改 shard 的一次性迁移成本；新 project 更倾向 DMuon（零 FSDP 源码改动）。
