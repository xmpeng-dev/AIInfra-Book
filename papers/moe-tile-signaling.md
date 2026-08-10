# 基于 Tile 级信号与调度的 MoE 细粒度计算-通信重叠
# Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts

> **arXiv/DOI:** [2607.19539v1](https://arxiv.org/abs/2607.19539) · **DOI:** [10.1145/3832810.3832907](https://doi.org/10.1145/3832810.3832907) · **PDF:** https://arxiv.org/pdf/2607.19539  
> **发表信息:** ICPP 2026（Singapore, Sep 28–Oct 1）  
> **机构:** Linnaeus University（瑞典）  
> **代码:** 原文未提供  
> **领域:** MoE 推理 · 计算-通信 overlap · Expert Parallelism · GPU kernel 协同  
> **核心贡献:** 用 **remote-owner-aligned 行布局 + rank-wide persistent GEMM(producer) + NVSHMEM segment 传输(consumer)** 在 **第二个 all-to-all(combine)** 路径上做 tile 级 overlap；4×A100 上相对 FasterMoE 端到端最高 **2.64×**、MoE 层最高 **2.74×**，且不侵入底层算子/通信原语。

---

## 一、问题分析 (Problem Analysis)

### 1.1 研究背景 (Research Background)

**领域现状** (Current State of the Field):
- MoE 通过稀疏激活在几乎不增加 per-token 计算的前提下扩展模型容量，已成为万亿参数 LLM 的关键构件（Switch Transformer、GShard 等）。
- 分布式 MoE 采用 **Expert Parallelism (EP)**：专家跨 GPU 分片，每层需两次 **all-to-all**——dispatch（把 token 发到专家所在 rank）和 combine（把专家输出送回 token 源 rank）。
- 实测表明，MoE 层中 all-to-all 通信可占 **近一半** 总执行时间；GPU 算力增速远快于互联带宽，通信阶段 GPU 空转愈发严重。
- 既有 overlap 方案分两大路线：**分解式 chunk 流水线**（FasterMoE、Tutel、PipeMoE 等）和 **kernel 融合**（Comet、CCFuser、FlashMoE 等）；近期还有 **tile 级 signaling**（Hong et al. EuroSys'26、T3 ASPLOS'24、TileLink MLSys'25）。

**核心挑战** (Core Challenges):
- **粗粒度分解的粒度失配**：NCCL 等 collective 要求 contiguous buffer，chunk 往往沿单一 tensor 维切分，与 GEMM tile 结构不对齐；小 chunk 伤 tensor-core 效率，chunk 间还需 host 侧同步。
- **传统 MoE 执行模式**：专家 GEMM 全部完成后才启动 combine all-to-all，通信完全暴露在关键路径上。
- **MoE combine 的动态路由**：每个输出行必须回到其 token 的源 rank，行→rank 映射运行时确定，比 TP 等静态通信模式更难做细粒度 overlap。
- **Producer/Consumer 资源争用**：计算 kernel 与通信 kernel 默认共享 SM，partition 不当会导致 back-pressure 或计算吞吐下降。

**研究动机** (Research Motivation):
- 分解式方法实现简单但难以达到 tile 级 overlap；融合式方法 overlap 效率高但需定制 barrier、跨 rank atomic、per-target kernel 特化，工程侵入性大。
- 本文借鉴 tile-level signaling 思想，但 **保持 compute 与 communication 为独立 persistent kernel**，通过 device-resident flag 协调，兼顾细粒度 overlap 与实现可移植性。
- 聚焦 **MoE 推理前向** 中 **第二个 all-to-all（return path）** 与 expert GEMM 的重叠；反向训练留作 future work。

### 1.2 问题定义 (Problem Definition)

**具体问题** (Specific Problem):

- **输入**：各 rank 在 dispatch 后收到的 token（经路由后的 expert 输入）、gate 权重、本地专家权重；路由元数据决定每行输出去向。
- **输出**：各 rank 上 per-token 的 gate 加权 expert 输出（combine + scale 之后）。
- **优化目标**：在 **不改变语义** 的前提下，缩短 MoE 层中 **expert GEMM + 第二个 all-to-all** 的关键路径延迟。
- **约束**：
  - 不修改底层 GEMM 主循环结构（基于 CUTLASS）；
  - 不修改 NCCL/NVSHMEM 原语语义；
  - 需支持运行时动态路由（不同 router mode：balanced / moderate_skew / stress_skew）；
  - FP16 tensor-core 精度下与串行 baseline 数值一致（在 rounding noise 范围内）。
- **评估指标**：operator 级延迟（GEMM + 第二 A2A）、MoE 层延迟、端到端前向延迟、吞吐（MTokens/sec）、overlap ratio（第二 A2A 被 GEMM 隐藏的比例）。

**问题形式化** (Problem Formalization):

MoE 层第 $l$ 层在 rank $r$ 上，设 dispatch 后本地有 $E$ 个专家，每个专家 $e$ 的 GEMM 为 $C_{M_e \times N} = A_{M_e \times K} \times B_{K \times N}$。combine 需将输出行 $i$ 写回 owner rank $\text{owner}(i)$。

传统执行：
\[
T_{\text{serial}} = T_{\text{GEMM}}(\{e\}_{e=1}^{E}) + T_{\text{A2A-combine}}
\]

本文目标：
\[
T_{\text{overlap}} = T_{\text{persistent-producer}} \parallel T_{\text{persistent-consumer}} \ll T_{\text{serial}}
\]

其中 producer 按 tile schedule 逐 tile 计算并在 epilogue 发布 ready signal；consumer 在 segment（一个或多个 $tb_M$ 高度的 row band）就绪后立即通过 NVSHMEM put 发起 remote write。

### 1.3 解决方案 (Solution)

**核心思路** (Core Idea):

把 MoE combine 路径重构为 **producer-consumer 协同**：rank 上所有本地专家共用一个 **persistent GEMM kernel（producer）**，在 CUTLASS epilogue 中对每个完成的 output tile 设置 device-resident ready flag；同时在独立 SM 分区上运行 **persistent communication kernel（consumer）**，轮询 segment 就绪状态并通过 NVSHMEM 发起 contiguous remote write。关键前提是 **remote-owner-aligned row layout**——重排 GEMM 输入行，使每个 output tile 只对应单一 destination rank，从而 consumer 无需 per-row 路由逻辑，每个 segment 一次 put 即可。

**方法概述** (Method Overview):

1. **Expert problem construction & combine plan（离线/launch 前）**
   - 对每个本地专家，按 owner rank 分组并重排输入行：**remote rows first, local rows last**；
   - 在 owner-rank 边界插入 zero-padding 使下一组 tile-aligned（$tb_M$ 对齐）；
   - 构建 **remote-first tile schedule**：remote tile 全部排在 local tile 之前；remote 行多的专家优先；
   - 预计算 **combine plan**：per-tile 的 `dest_rank`、`remote_buffer_offset`、`valid_row_count` 扁平数组。

2. **Rank-wide persistent GEMM（producer）**
   - 单 kernel 覆盖 rank 上所有专家的 tile worklist（非 classical grouped GEMM 的矩阵拼接）；
   - 每个 persistent CTA 按 strided 模式取 tile；专家切换仅 register-level 指针替换，无 re-launch；
   - CUTLASS epilogue 在 tile store 后 thread-fence + 写 ready flag。

3. **Segment-granular NVSHMEM transfer（consumer）**
   - 传输粒度为 **row band**（$tb_M$ 行 × 全宽 $N$ 的 contiguous 条带），而非单 tile（仅 $tb_N$ 列，内存 strided）；
   - 首/末 segment 各 1 个 row band（尽早启动、避免 communication tail）；中间 segment 合并 $x$ 个 row band（$mgb$ 参数，默认 1 或 2）以饱和带宽曲线。

4. **Producer-Consumer co-scheduling**
   - 两 kernel 在不同 non-blocking stream；consumer **更高 stream priority**；
   - consumer 占用 $cCTA$ 个 SM（默认 14/108），producer 用剩余 SM；persistent kernel 消除 SM 争用；
   - 全程 device-resident 协调，critical path 无 host 同步。

**技术细节** (Technical Details):

*Remote-owner-aligned Row Layout*:
- **Function:** 保证每个 output tile 对应唯一 destination rank。
- **Implementation:** 每专家内按 remote rank 分组 contiguous 排列；边界 padding 至 $tb_M$ 倍数；padding 行参与 GEMM 但不传输。
- **Innovation:** 把 MoE 动态 combine 路由 **前移到 layout 阶段**，使 communication kernel 退化为 O(1) 查表 + contiguous put。

*Remote-first Tile Schedule*:
- **Function:** 最大化 overlap 窗口——尽早产出需通信的数据。
- **Implementation:** 专家按 remote row-band 数降序；每专家 remote tile 全部完成后才排 local tile。
- **Innovation:** 与 persistent rank-wide kernel 结合，一次 launch 覆盖全部专家，消除 per-expert kernel launch 开销。

*Combine Plan*:
- **Function:** 避免 transfer 时 per-row 查路由。
- **Implementation:** 遍历 layout 生成 device-resident 数组：`dest[i]`、`offset[i]`、`nvalid[i]`；跨专家协调同一 peer rank 的 buffer offset 不重叠。
- **Innovation:** tile index → 三元组数组读，consumer 无分支路由。

*Segment Granularity*:
- **Function:** 平衡「尽早开始传输」与「带宽饱和」。
- **Implementation:** 8 SM consumer 在 ~1 MiB 达 87 GB/s 峰值；4 SM 在 ~3 MiB 达 67 GB/s；单 tile 仅 32–64 KiB 远低于饱和点。
- **Innovation:** 首段 1 band 快速启动 + 中段 $x$ band 合并 + 末段 1 band 防 tail。

*SM Partitioning*:
- **Function:** 隔离 producer/consumer 资源，避免 contention。
- **Implementation:** $cCTA \in \{2,4,...,24\}$ 可调；默认 14。
- **Innovation:** 揭示「少给 communication SM 反而更慢」的反直觉——$cCTA=2$ 时可达 baseline 的 **1.91×** 延迟。

**算法/架构描述** (Algorithm/Architecture Description):

```
MoE 层 combine 路径（单 rank）:

[Combine Plan 构建]     [Persistent Producer]          [Persistent Consumer]
  routing → layout  →   rank-wide GEMM tiles  ←──→   poll ready flags
  → tile schedule       remote-first order              NVSHMEM put segments
  → flat metadata       epilogue: signal per tile       higher stream priority
                        lower stream priority           cCTA SMs dedicated

时间线（理想 overlap）:

Producer:  [remote tiles expert A][remote tiles expert B]...[local tiles]
Consumer:       [seg1 put][seg2 put][seg3 put]...[last put]
                 ↑ 与 producer 并行，overlap ratio 可达 71.9%–99.9%
```

实现栈：PyTorch 2.6.0 · CUDA 12.1 · CUTLASS 3.9 · NVSHMEM 3.6.5；dispatch 仍用 NCCL，combine overlap 用 NVSHMEM。

---

## 二、实验效果 (Experimental Results)

### 2.1 实验设置 (Experimental Setup)

| Item | Details |
|------|---------|
| Datasets / Models | 三个合成 MoE 模型：**M-GPT**（MoE 仅在 block 11，B=8,S=1024）、**M-BERT**（block 2/5/8/11，B=32,S=512）、**M-Trans-xl**（全部 12 block，B=16,S=512）；均 topk=2，64 experts（4 GPU × 16/GPU） |
| Baselines | **FasterMoE**、**Tutel**、**Megatron-CUTLASS**（GroupGEMM）、**Megatron-TE**（Transformer Engine）；另与 PyTorch 串行 baseline（cuBLAS GEMM + NCCL A2A）对比 |
| Metrics | E2E 前向延迟、MoE 层延迟、operator 延迟、吞吐（MTokens/sec）、overlap ratio、正确性（相对误差） |
| Hardware | 单节点 **4× NVIDIA A100**（108 SM，40 GB HBM）；NVLink 4 lane × 25 GB/s ≈ **100 GB/s** 单向 peer 带宽 |
| Software | CUDA 12.1, NCCL 2.29.3, PyTorch 2.6.0 |
| Shape sweep | $tpr \in \{8192,16384\}$, $M=2\times tpr$, $K=2048$, $N \in \{4096,8192\}$, $E \in \{4,8,16,32,64\}$, router: balanced / moderate_skew / stress_skew, $cCTA \in [2,24]$, $mgb \in \{1,2\}$ |

### 2.2 主要结果 (Main Results)

**核心性能指标** (Core Performance Metrics):

**vs 四个 SoA 系统（Figure 4，相对 FasterMoE 归一化）**

| 场景 | E2E 加速 (vs FasterMoE) | MoE 层加速 (vs FasterMoE) | 备注 |
|------|-------------------------|----------------------------|------|
| M-GPT | **1.57×** | **2.65×** | 全面优于四个 baseline |
| M-BERT | **1.66×** | **1.78×** | 全面优于四个 baseline |
| M-Trans-xl | **2.64×** | **2.74×** | vs FasterMoE 最高；但 **略慢于** Tutel / Megatron-CUTLASS / Megatron-TE |

**vs Tutel（M-Trans-xl）**：E2E 1.35×（M-GPT）、1.15×（M-BERT）；M-Trans-xl 上本文略逊——因 hidden dim=512 最小，通信量小、overlap 固定开销占比高。

**Expert 数量 sweep（Figure 5，单 MoE 层 microbenchmark，Dim=1024）**

| vs Baseline | 加速范围 |
|-------------|----------|
| FasterMoE | **1.30× – 5.33×** |
| Tutel | 1.77× – 2.16× |
| Megatron-CUTLASS | 1.23× – 1.45× |
| Megatron-TE | 1.35× – 1.76× |
| Overlap ratio | **71.9% – 99.9%** |

**vs 串行 PyTorch baseline（Table 2 配置，$N=8192$）**

| 层级 | balanced | moderate_skew | stress_skew |
|------|----------|---------------|-------------|
| Operator 峰值 | **2.97×** | **2.94×** | **3.01×** |
| MoE 层峰值 | **1.78×** | **1.73×** | **1.77×** |
| 吞吐峰值 | **1.78×** | **1.73×** | **1.77×** |

**关键发现** (Key Findings):
- tile 级 signaling 在 **full model** 集成后仍有效；E2E 加速低于 MoE 层加速符合 Amdahl 定律（非 MoE 部分不变）。
- **$E$ 越大 overlap 收益越大**：更多 expert → 更重 combine 通信 → 传统方案 post-GEMM tail 更长；本文 overlap ratio 随 $E$ 升高。
- **M-Trans-xl 上输给 Megatron 系**：hidden dim 512 → 通信体积小、per-tile 有效计算少 → 细粒度 overlap 固定开销占比高。
- **SM partition 极度敏感**：$cCTA=2$ 在所有 shape/router 下均慢于 baseline（stress_skew + $M=32768$ + $E=4$ 达 **1.91× baseline 延迟**）；最优区间 **$cCTA \in [10,20]$**，默认 14。
- **正确性**：1440 组配置中 max 相对误差 **$1.913 \times 10^{-3}$**（容差 $8 \times 10^{-3}$），仅 1 次 stress_skew 迭代初检失败但重跑通过（视为 transient nondeterminism）。

### 2.3 消融实验 (Ablation Study)

原文无独立 named ablation table，但以下对比可视为组件/参数消融：

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full method ($cCTA=14$, $mgb=1$) | 最优区间 | 默认配置 |
| $cCTA=2$ | 比 baseline 慢 up to **1.91×** | consumer 资源不足，back-pressure 崩溃 overlap |
| $cCTA=4$ | 部分配置 > baseline | 重负载下仍不足 |
| $cCTA \in [10,20]$ | 通常最优 | 计算/通信资源平衡 |
| $mgb=1$ vs $mgb=2$ |  workload-dependent | 1 band ≈ 1–2 MiB；2 band ≈ 2–4 MiB；无 uniformly best |
| w/o remote-first schedule | 原文未单独报告 | 设计动机：缩短 first transferable segment 等待 |
| w/o row layout padding | 无法 tile-aligned combine | 破坏 single-dest tile 性质 |
| Padding overhead | 最多 $(W-1)(tb_M-1)=765$ 行/rank（4-rank, $tb_M \le 256$） | 占 Table 2 问题规模几 %；stress_skew 最大 |

**消融结论** (Ablation Conclusions):
- **SM partition 是最关键超参**：communication consumer 过少比不做 overlap 更糟。
- **Segment 大小 ($mgb$) 需 empirical tuning**：带宽饱和 vs 等待更多 tile 的 trade-off。
- **Remote-owner-aligned layout 的 padding 开销** 远小于 overlap 收益。

---

## 三、业界类似方案 (Industry Similar Solutions)

### 3.1 方案对比表 (Solution Comparison Table)

| Solution | Year | Core Idea | Advantages | Disadvantages | Performance |
|----------|------|-----------|------------|---------------|-------------|
| FasterMoE | 2022 | pipeline degree=2 的 token 维 chunk A2A+GEMM | 实现简单 | 粗粒度、host 同步、小 chunk 伤 GEMM | 本文 E2E 1.57–2.64× 更快 |
| Tutel | 2023 | 自适应 pipeline degree + 2D hierarchical A2A | 生产可用、`a2a_ffn_overlap_degree` 可调 | 仍 chunk 级；小 hidden dim 上更优 | M-Trans-xl 上 Tutel 略胜本文 |
| Comet | MLSys'25 | shared-tensor 分解 + 单 kernel thread-block 专用化 | 86.5% 通信隐藏；1.96× 单层 | 侵入式融合 kernel；需 per-shape 特化 | 融合路线代表；见 [`comet.md`](./comet.md) |
| MegaScale-MoE | EuroSys'26 | 节点内 SP+EP + intra-op tile fuse + device barrier | 1440×H800 生产级；训练 fwd+bwd | 需定制 scatter/gather + 融合 kernel | 见 [`megascale-moe.md`](./megascale-moe.md) |
| FlashMoE | NeurIPS'25 | 整层 MoE 单 kernel（dispatch+GEMM+combine） | 极致融合 | 工程复杂度最高 | 单 kernel 哲学极端 |
| Hong et al. (signaling) | EuroSys'26 | tile signaling + reordering（通用 GEMM+A2A） | 本文直接借鉴 signaling 原则 | 非 MoE 专用 layout | 本文 cite 为方法论 predecessor |
| **This Paper** | ICPP'26 | **独立 persistent producer/consumer + remote-owner layout + segment NVSHMEM** | **非侵入、保留 CUTLASS/NCCL 栈；overlap ratio 71–99%** | **仅 fwd inference；4 GPU 规模；M-Trans-xl 输给 Megatron 系** | **2.64× E2E / 2.74× MoE layer vs FasterMoE** |

### 3.2 技术路线对比 (Technical Approach Comparison)

**路线A：Chunk 流水线（Decomposition-based Pipelining）**
- Representative works: FasterMoE, Tutel, PipeMoE, ScheMoE, MPipeMoE, Centauri, Domino
- Core idea: 沿 token 维切分输入，不同 chunk 的 GEMM 与 A2A 在不同 CUDA stream 上流水线执行。
- Pros and cons: (+) 框架易集成；(−) chunk 与 GEMM tile 不对齐；NCCL 需 contiguous buffer；chunk 间 host 同步；小 chunk 降低 tensor-core 利用率。

**路线B：Kernel 融合（Fusion-based）**
- Representative works: Comet, CCFuser, FlashMoE, Punniyamurthy et al., MegaScale-MoE intra-op kernels
- Core idea: 计算与通信融合为单 kernel 或在 thread-block 内动态分配 compute/comm 角色。
- Pros and cons: (+) overlap 粒度最细、可消除 launch 开销；(−) 需 custom barrier、跨 rank atomic、per-target 优化；维护成本高。

**路线C：Signaling + 独立 Kernel（本文路线）**
- Representative works: T3 (ASPLOS'24), Hong et al. (EuroSys'26), TileLink (MLSys'25), **本文**
- Core idea: GEMM epilogue 发 tile-ready signal；独立 communication kernel 消费；可选 SM partition。
- Pros and cons: (+) 不修改 GEMM main loop / 不融合 comm 原语；CUTLASS epilogue 集成轻量；(−) 两 persistent kernel + MoE 专用 layout 预处理；需 tuning $cCTA$/$mgb$。

### 3.3 本文定位 (This Paper's Position)

- **Improvement over Approach A**: 达到 tile/segment 级 overlap（非 chunk 级）；无 inter-chunk host 同步；rank-wide persistent kernel 消除 per-expert launch；operator 峰值 **~3×** vs 串行 baseline。
- **Improvement over Approach B**: 避免 intrusive fusion；保留 CUTLASS profiler 选优 + 标准 NVSHMEM put；实现复杂度显著低于 Comet/FlashMoE。
- **Unique contributions**: **MoE combine 专用的 remote-owner-aligned row layout** 使动态 A2A 退化为 contiguous segment put；**remote-first tile schedule** 最大化 overlap 窗口；系统的 **SM partition sensitivity** 分析（$cCTA=2$ 反例）。

### 3.4 推荐进一步阅读 (Recommended Further Reading)

| Paper | Reason |
|-------|--------|
| [Comet (MLSys'25)](./comet.md) | 同问题（MoE A2A+GEMM overlap）的融合 kernel 路线；对比 non-intrusive signaling 路线的 trade-off |
| [MegaScale-MoE (EuroSys'26)](./megascale-moe.md) | 生产级 MoE 训练中 tile-level device barrier + GroupedGEMM fuse 的工业实现 |
| Hong et al. (EuroSys'26, arXiv:2504.19519) | 本文 signaling 方法论来源；通用 GEMM+A2A reordering |
| Tutel (MLSys'23) | 自适应 chunk pipeline baseline；理解 `a2a_ffn_overlap_degree` |
| FlashMoE (NeurIPS'25) | 单 kernel 整层 MoE 的极端融合设计 |

---

## 四、全文翻译 (Full Translation)

> 以下为论文全文的中文翻译，保持原有段落结构。技术术语首次出现标注英文原文。

### 摘要 (Abstract)

混合专家（Mixture-of-Experts, MoE）架构在不成比例增加计算成本的情况下提升模型容量，已成为将大语言模型（LLM）扩展到万亿参数规模的关键构建块。高效部署 MoE 模型依赖跨多 GPU 的分布式执行，其中每个 MoE 层涉及两次 all-to-all 通信：将 token dispatch 到专家所在 rank，以及将专家输出返回到其源 rank。传统 MoE 实现在专家计算完成后才启动 return all-to-all，使通信延迟暴露在关键路径上并降低 GPU 利用率。我们提出一种通过 tile 级信号与调度，将专家计算与第二次 all-to-all 重叠的细粒度方法。我们的 producer-consumer 协同设计包含：(1) 覆盖 rank 上所有本地专家的 persistent 计算 kernel（producer），消除重复 kernel launch 开销并优先处理 remote-critical tile；(2) 在少量专用流式多处理器（Streaming Multiprocessors, SMs）分区上运行的 persistent 通信 kernel（consumer），在 tile 就绪时发起 segment 粒度传输。该协同设计避免对底层计算算子或通信原语做侵入性修改，便于在多 GPU 系统上提升分布式 MoE 执行效率。在 4×A100 平台上，针对三个 MoE 模型与四个最先进 MoE 系统的评估表明，我们的方法实现最高 **2.64×** 端到端加速和 **2.74×** MoE 层加速。与传统非 overlap baseline 相比，我们的方法在 varying GEMM shape、router mode 和广泛的 producer/consumer SM 分区下，一致提升算子级和 MoE 层级性能，同时保持正确性。

### 1. 引言 (Introduction)

现代机器学习（ML）模型的计算需求快速增长，驱动因素包括模型规模增大、序列变长以及多模态模型兴起。近年来大语言模型参数规模在数年内从数十亿增长到万亿（Fedus et al., 2022; Llama Team, 2025）。在 dense transformer block 中，每个 token 走相同的 dense 计算路径，因此 per-token 计算随模型规模线性增长。随着模型持续增大，模型容量与 active per-token 计算之间的耦合成为主要可扩展性瓶颈。

在不成比例增加 active 计算的前提下扩展模型容量的压力，推动了 sparse MoE 架构。MoE 层将每个 token 路由到仅一部分专家，从而增加模型容量同时限制 per-token 计算量。但在分布式 MoE 执行中，专家通常通过 expert parallelism 分区到各 GPU，token 路由引入大量 all-to-all 通信开销，约占 **总执行时间近一半**（Zhang et al., 2025）。硬件趋势进一步放大该问题：GPU 计算吞吐增速远快于互联带宽，加速器在通信阶段空闲时间比例持续上升。

缓解通信开销的自然方法是让通信与依赖的计算 overlap。若干基于分解的方法（Wang et al., 2022; Chen et al., 2024a; Jangda et al., 2022; Wang et al., 2024; PyTorch, 2024; Jiang et al., 2024）将 collective 与 surrounding GEMM 切分为 chunk 并在不同 stream 上流水线化。MoE 专用系统（Hwang et al., 2023; He et al., 2022; Shi et al., 2024, 2023; PyTorch, 2024; Zhang et al., 2023）沿 token 维应用类似思想，将专家计算与 all-to-all 流水线化。这些粗粒度 overlap 方法易于在现有框架中实现，但在小 chunk 尺寸下往往无法充分利用 tensor-core 效率，且 chunk 间存在不可忽视的 host 侧同步成本。此外，NCCL 等 GPU 通信库的 collective API 通常要求 contiguous buffer，分解往往限于单一 tensor 维度，所得 chunk 与 GEMM tile 结构不对齐。因此仅靠分解难以实现 tile 级 overlap。另一大瓶颈是 host 侧 kernel launch 开销。Kernel 融合方法（Punniyamurthy et al., 2024; Aimuyo et al., 2025; Wang et al., 2025; Zhang et al., 2025）将计算与通信融合为单 kernel，从而消除 host 侧编排（Punniyamurthy et al., 2024）和冗余 launch（Aimuyo et al., 2025）。这些方案普遍提升 overlap 效率，但通常需要侵入性工程，包括 custom barrier、跨 rank atomic 协议和 per-target kernel 特化。近期研究（Hong et al., 2026; Pati et al., 2024）使用 tile 级 signaling 从已完成的 GEMM 输出区域发起通信，实现更细粒度的计算-通信 overlap。我们的设计采用该 signaling 原则，同时保持计算与通信 kernel 分离。

本文面向分布式 **MoE 推理**，提出通过 tile 级 device-resident signal 协调 GEMM（producer）与通信（consumer）kernel 的方法。设计与评估 **仅覆盖前向执行**；支持训练反向留作 future work。Producer 在 tile 完成时发出 signal，consumer 在对应 tile 就绪后立即为一个或多个 row band（row band 是跨越完整输出宽度的连续输出行条带，见 Section 3.2.1 与 Figure 3(b)）发起 segment 粒度传输，无需 host 侧同步。与分解式方法相比，该设计保持 tensor-core 效率，既无 inter-chunk 同步也无 chunked 执行的 launch 开销；同时避免融合方案的实现复杂度。具体地，我们在不相交的 SM 分区上 launch 两个 persistent kernel：可调比例的 SM Dedicated 给通信 kernel，其余运行 GEMM。该分区使各 kernel 拥有无干扰的计算资源，persistent kernel 与 device-resident signal 使 host 与 barrier 同步脱离关键路径。将该设计用于 MoE 层的 return path（专家输出投影后第二次 all-to-all）比 TP 等静态通信模式更具挑战——每个专家输出行必须回到 originally 持有其输入 token 的 rank，行到 rank 映射运行时确定。我们的关键进展是 **remote-owner-aligned row layout**，使每个输出 tile 对齐单一 destination rank，consumer 对每个 segment 发起一次 contiguous remote write，通信 kernel 内无需 per-row 路由逻辑（Hong et al., 2026）。

**贡献总结：**
1. MoE return path 的细粒度计算-通信 overlap：signaling 机制 + persistent producer/consumer 在不相交 SM 上协同。
2. Communication-aware remote-owner-aligned row layout：每 tile 映射单一 destination rank；producer 可优先 remote-critical tile；consumer 每 segment 一次 contiguous write。
3. Rank-wide GEMM kernel：单 persistent producer 覆盖 rank 上全部本地专家，消除 repeated launch。
4. 跨模型与配置的全面评估：三 MoE 模型、三路由分布、多 GEMM shape、广泛 SM 分区；operator/MoE 层/E2E 加速、SM 争用、正确性验证。

### 2. 背景 (Background)

#### 2.1 MoE 架构

Transformer 模型中，MoE 层用多个子层（experts）和 gating（router）网络替代标准 FFN（Lepikhin et al., 2021）。每个 token 的 router 通过概率分布选择 top-k 专家，层输出为 gate 加权专家输出之和。因每 token 仅激活少数专家，active 计算远小于同等参数量的 dense FFN——这是 MoE 可扩展容量的基础。

分布式设置中，专家分布在各 GPU（rank）。MoE 层前向包括：(1) Routing；(2) 第一次 all-to-all（dispatch）；(3) 专家计算；(4) 第二次 all-to-all（combine）；(5) 加权归约（scale）。两次 all-to-all 是主要通信瓶颈。本文聚焦 **专家计算与第二次 all-to-all 的 overlap**。

#### 2.2 GEMM Kernel 与 Epilogue Signaling

GEMM $C_{M\times N}=A_{M\times K}\times B_{K\times N}$ 是神经网络主导运算。现代 GPU GEMM 将输出矩阵分解为矩形 output tile；thread block（CTA）通常一次计算一个 tile，沿 $K$ 维迭代，warp 用 tensor-core 做 MAC，寄存器累加后在 epilogue 写 global memory。Tile 结构是 GPU GEMM 性能核心：揭示输出矩阵并行性、改善 shared memory/register 复用、提供自然调度单元。对本工作关键的是，CTA 独立调度，单个 output tile 可在整个 GEMM 完成前就绪，允许下游消费 ready tile。

CUTLASS 等库通过 template 暴露 tile shape、layout、pipeline、epilogue。GEMM kernel = main loop + epilogue（输出转换、逐元素操作、tile store）。Store 是 tile 在内存中可见之处，故 epilogue 是发布 tile 级完成 signal 的自然位置。Epilogue signaling 下，producer 在 store 每个 tile 后设置 device-resident flag，consumer kernel 可 poll 这些 flag 并在其他 tile 仍在计算时处理 ready tile。开销为每 tile 少量额外指令，不扰动 main loop。

#### 2.3 Device-Initiated Communication

分布式 GPU 工作负载 increasingly 依赖 device-initiated communication：传输在 CUDA kernel 内发起而非 host launch。NVSHMEM 通过对称 heap 暴露 one-sided put/get，使 rank 上 kernel 可直接写 peer 对称 GPU 内存。从而可将 MoE all-to-all 实现为单 CUDA kernel 对 tile 发起 remote write，而非 host-driven collective 链。本文用该原语将 MoE return path 的第二次 all-to-all 实现为由 per-tile signal 驱动的 consumer kernel。

#### 2.4 并发 Kernel 与 SM 分区

CUDA stream 允许多 kernel 并发，但默认共享 SM 资源并争用。将各 kernel launch 为 persistent kernel（固定 CTA 数，每 CTA 驻留 SM 直至 kernel 退出）可使两 kernel 在不相交 SM 子集运行，消除 SM 级干扰。本文用该机制在 small SM partition 上运行 communication consumer，详见 Section 3.3.3。

### 3. 系统设计 (System Design)

#### 3.1 概览

Figure 1–2 展示分布式 MoE 执行流设计。聚焦 dispatch 后专家计算的 **第二次 all-to-all**。GEMM 输入每行对应一个 routed token copy，下文 interchangeably 使用。设计包含：

1. **Expert problem construction & combine plan**：rank-wide GEMM 输入采用「remote tokens first, local tokens last」layout；remote 行按 destination rank 分组；owner-rank 边界插入 alignment padding；构建 remote-bound output 最早产出的 tile schedule；combine plan 将路由元数据预解析为 flat per-tile device 数组。

2. **Overlapped execution**：Producer 为覆盖 rank 上全部本地专家的 persistent GEMM，epilogue 标记 ready tile；Consumer 为 persistent kernel，在 disjoint SM 上 poll ready segment 并通过 non-blocking NVSHMEM put 传输；两 kernel 在不同 stream，consumer 更高 priority，全程 device flag 协调。

#### 3.2 Expert Problem Construction and Combine Plan

该阶段（Figure 1 左框）准备 rank-wide persistent GEMM 的输入、执行顺序与传输元数据。各专家保留独立 $A,B,C$；rank-wide 性通过 unified tile worklist 而非矩阵拼接实现。

**3.2.1 Remote-owner-aligned Row Layout**

细粒度 overlap 有效当且仅当每个 completed output tile 映射到 **单一 destination rank**。Dispatch 后各本地专家输入行来自多 peer rank，无重排则单 tile 可含指向不同 rank 的行，破坏 single-destination 性质。我们 impose remote-owner-aligned layout：每本地专家内，(1) 指向 remote owner 的行优先，按 owner rank contiguous 分组；(2) 指向 local rank 的行最后；(3) 每个 owner-rank 边界若行数非 $tb_M$ 倍数，用 zero-filled 行 pad 至 tile-aligned offset。Padding 行参与 GEMM 但不传输。Figure 2(b) 示例 4-rank、2 experts/rank。Padding 仅影响 owner 边界 trailing 不足 $tb_M$ 的行；完整 $tb_M$ 高度 row band 无需 padding；local 组无需 padding。此后每 remote tile 有静态已知 destination，tile index → destination 查找 trivial。

**3.2.2 Remote-first Tile Schedule**

给定 aligned layout，构建 rank-wide tile schedule：(i) 全部 remote tile 先于 local tile（仅 remote 需通信）；(ii) remote 行多的专家优先，使最大传输尽早进入通信阶段并 minimize expert 切换。Figure 2(c) 示例 rank 1：expert 3 remote tile 先（更重 remote load），再 expert 2 remote tile，最后全部 local tile。Algorithm 1 总结 layout 与 schedule 构造。

**3.2.3 Combine Plan**

Consumer 返回每 tile 到 owner rank 需知：destination rank、remote receive buffer 写偏移、valid（非 padding）行数。Transfer 时解析需 inspect 每 tile 行并查 gate 阶段路由元数据，增加 per-row 开销。Combine plan 遍历前述 layout 构建 flat device 数组：`dest_rank`、`remote_offset`、`valid_rows`，按 tile id 索引，每 tile 三次 array read。Under remote-owner-aligned layout，每 tile valid rows 共享单一 owner rank，plan 将 tile 分为 remote-publishable 或 local-only；对每 remote-publishable row band 分配 destination rank receive buffer 的全局写偏移，跨本地专家协调同 peer 的非重叠 placement。Local-only tile 无需 remote transfer。

#### 3.3 Overlapped Execution

**3.3.1 Rank-Wide Persistent GEMM with Tile-Level Signaling**

Producer 为 rank 上 **单 persistent GEMM kernel**，按 rank-wide tile schedule 处理全部 tile，而非 per-expert launch。Scheduled item 配对 expert index 与 tile 坐标；per-expert device view 提供 $A,B,C$ 指针与维度。多 persistent CTA，每 SM 一 CTA；CTA $i$ 处理 tile $i, i+g, i+2g, ...$。每 tile CTA 取 device view、register-level 复制 base param block、仅覆写 expert-varying 字段后调用 GEMM tile routine；专家切换开销 negligible。Shared-memory staging 跨 tile/expert 复用。

Signaling 在 CUTLASS epilogue 内：main-loop MAC 完成后，CTA thread-fence 保证 global visibility，再 publish ready flag。构成 producer 与 consumer 并发执行的接口。

**3.3.2 Communication Granularity**

Signaling 标记每 tile ready，但 per-tile 传输导致严重 fragmentation（Hong et al., 2026）。传输粒度由带宽利用率与 row layout 下 GEMM 输出结构决定。Figure 3(a)：8 SM consumer ~1 MiB 达 87 GB/s 峰值，4 SM ~3 MiB 达 67 GB/s；单 tile 通常 32–64 KiB，远低于饱和。Motivation：粒度大于单 tile。Aligned layout 下，$tb_M$ 高度 row band 内所有行返回同一 destination rank，是 natural transfer unit；单 tile 仅覆盖 $tb_N$ 列，是 row band 的 partial-width strided slice，需多次 copy；row band 跨全宽 $N$，owner-uniform 且 contiguous，单次 remote write 即可。

因此以 **segment** 粒度传输：首 segment 保持单 row band 以便最早启动；末 segment 亦单 row band 减少 communication tail；中间 segment 合并 $x$ 个 consecutive row band（$x$ 可调），使 payload 落在 Figure 3(a) 饱和区。最优 $x$ empirical 且 workload-dependent；$x=1$（~1–2 MiB）与 $x=2$（~2–4 MiB）无 uniformly best。

**3.3.3 Producer-Consumer Co-Scheduling**

Producer 发布内容与 consumer 传输粒度已定义；剩余问题是如何共享 GPU 使 GEMM 与 transfer 真正并行。Producer 与 consumer 为两 independent persistent kernel，依赖：(1) stream priority——consumer 更高，因 delay producer tile 主要影响 GEMM tail，delay consumer transfer 直接进入通信 critical path 并 compound；(2) SM partition——consumer 占 $cCTA$ SM  effectively reserved，producer 用 stable 剩余 SM budget；(3) device-memory coordination——驻留后全程 device state，critical path 无 host。整体将 serialized compute-then-communicate 转为 continuously overlapped execution。

#### 3.4 实现 (Implementation)

CUDA runtime 暴露给 PyTorch 2.6.0；CUDA 12.1, CUTLASS 3.9, NVSHMEM 3.6.5。Persistent GEMM 基于 CUTLASS templated GEMM，profiler lookup table 选 per-shape 最优配置。Following EVT (Chen et al., 2024b)，tile signaling 集成于 epilogue。Consumer 为 separate persistent kernel，通 NVSHMEM 通信。

### 4. 实验 (Experiments)

#### 4.1 实验设置

**平台**：单节点 4×A100，NVLink 4 lane×25 GB/s ≈ 100 GB/s 单向 peer；108 SM，40 GB HBM；每 GPU 一 rank。CUDA 12.1, NCCL 2.29.3, PyTorch 2.6.0。NCCL 负责 dispatch 与 baseline 通信；combine overlap 用 NVSHMEM+CUTLASS。

**Baselines**：FasterMoE, Megatron-CUTLASS (GroupGEMM), Megatron-TE, Tutel。

**Workload**：三 MoE 模型（Table 1）；topk=2，64 experts，4 GPU；另 sweep $E \in \{4,...,64\}$。Table 2 shape sweep：$tpr$, $M$, $K$, $N$, router mode, $cCTA$, $mgb$。

#### 4.2 MoE 模型性能

**4.2.1 E2E 与 MoE 层（Figure 4）**

相对 FasterMoE：M-GPT E2E **1.57×**、MoE 层 **2.65×**；M-BERT **1.66×** / **1.78×**；M-Trans-xl **2.64×** / **2.74×**。M-Trans-xl 上 E2E 略慢于 Tutel/Megatron 系——hidden dim 最小（512），通信量线性于 dim，可隐藏通信少，per-tile 有效工作少，细粒度 overlap 固定开销占比大。E2E 加速低于 MoE 层加速符合 Amdahl（非 MoE 部分不变）。

**4.2.2 Expert 数量（Figure 5）**

单 MoE 层 microbenchmark：相对 FasterMoE **1.30–5.33×**，Tutel **1.77–2.16×**，Megatron-CUTLASS **1.23–1.45×**，Megatron-TE **1.35–1.76×**。Overlap ratio **71.9–99.9%**。$E$ 越大 benefit 越大。Dim=1024 时亦低于 Megatron-CUTLASS/TE——更大 hidden 提供足够 per-tile work 摊销 fixed overhead。

#### 4.3 可扩展性（Table 2, $N=8192$）

Operator 峰值：**2.97×** (balanced), **2.94×** (moderate_skew), **3.01×** (stress_skew)。MoE 层峰值：**1.78×**, **1.73×**, **1.77×**。吞吐峰值同 MoE 层（latency 反比）。

**4.3.4 资源争用（Figure 9）**

$cCTA$ 从 2 到 24 sweep。$cCTA=2$ 所有 shape/router 慢于 baseline，stress_skew+$M=32768$+$E=4$ 达 **1.91× baseline**。$cCTA=4$ 部分 >1.0。最低延迟通常 $cCTA \in [10,20]$。Router skew 放大 penalty：balanced 最小，stress_skew 最大。说明有效 overlap 需 resource-aware，非单纯 maximize compute SM。

#### 4.4 正确性

相对 baseline 比较最终 per-token 输出；共享 routing、dispatch、scale；差异仅 overlapped expert compute + second A2A。容差 $8\times10^{-3}$；1440 checks 全通过（stress_skew 1 iteration 初检失败但 rerun pass）；max $|\varepsilon|_{rel}=1.913\times10^{-3}$。

#### 4.5 开销

Padding 最多 $(W-1)(tb_M-1)$ 行/rank；4-rank、$tb_M\le256$ 最多 765 行，占 Table 2 规模几 %。Balanced routing padding 最少，stress_skew 最大。Overlap 收益仍远超该 cost。

### 5. 相关工作 (Related Work)

分解式方法（CoCoNet, Wang et al., Centauri, Domino, PyTorch async TP, MegaScale 等）将 GEMM 与 collective 切 chunk 流水线。MoE 粗粒度 pipelining（FasterMoE, Tutel, PipeMoE, ScheMoE, MPipeMoE）沿 token 维 chunk。融合式（Punniyamurthy, Comet, CCFuser, FlashMoE）将 comm 与 comp 融为单 kernel，fine-grained overlap 强但 software complexity 高。

### 6. 结论 (Conclusion)

我们提出 MoE 细粒度计算-通信 overlap 设计，通过 tile-level signaling 与 scheduling 将第二次 all-to-all 隐藏在专家计算后。三组件协同：remote-owner-aligned layout；communication-aware tile schedule 的 persistent rank-wide producer；consumer 在 segment 就绪时转发。三 MoE 模型、多 problem size、router 分布、SM 分区下相对 SoA 与串行 baseline 有 substantial 加速，输出匹配 baseline 至 FP16 rounding。Future work：训练反向、自适应 SM partition、更多并行 regime、更大规模机器。

### 参考文献 (References)

关键引用标题择译：
- FasterMoE: 大规模动态预训练模型的建模与优化
- Tutel: 规模化自适应 MoE
- Comet: MoE 细粒度计算-通信 overlap
- FlashMoE: 单 kernel 快速分布式 MoE
- Hong et al.: 通过 signaling 与 reordering 的高效可适配 overlap

完整 bibliography 见原文。

---

## 附录 (Appendix)

### A. 术语表 (Glossary)

| English Term | Chinese Translation | Explanation |
|--------------|---------------------|-------------|
| Mixture-of-Experts (MoE) | 混合专家 | 稀疏激活的多专家 FFN 层 |
| Expert Parallelism (EP) | 专家并行 | 专家权重跨 GPU 分片 |
| All-to-all (A2A) | 全互换通信 | MoE dispatch/combine 的核心 collective |
| Tile | 分块 | GEMM 输出矩阵的矩形计算单元 |
| Row band | 行带 | $tb_M$ 行 × 全宽 $N$ 的 contiguous 输出行条带 |
| Segment | 段 | 一个或多个 row band 组成的传输单元 |
| Producer / Consumer | 生产者 / 消费者 | 计算 GEMM kernel / 通信 NVSHMEM kernel |
| Persistent kernel | 持久 kernel | 固定 CTA 数、CTA 驻留 SM 直至完成的 kernel |
| Epilogue signaling | 尾声信号 | CUTLASS GEMM epilogue 中发布 tile-ready flag |
| Remote-owner-aligned layout | 远程属主对齐布局 | 按 token 源 rank 重排 GEMM 输入行 |
| NVSHMEM | — | GPU device-initiated one-sided 通信库 |
| $tb_M$ | Tile M 维度 | GEMM tile 在 $M$ 维的高度 |
| $cCTA$ | Consumer CTA 数 | 分配给 communication kernel 的 SM 数（1 CTA/SM） |
| $mgb$ | Middle segment row bands | 中间传输 segment 合并的 row band 数 |
| Overlap ratio | 重叠率 | 第二 A2A 被 expert compute 隐藏的时间比例 |

### B. 复现检查清单 (Reproducibility Checklist)

- [ ] Code open-sourced: **No**（原文未提供）
- [ ] Data available: N/A（合成 MoE 模型，配置见 Table 1–2）
- [ ] Hyperparameters complete: **Partial**（$cCTA=14$, $mgb \in \{1,2\}$ 默认；CUTLASS config 来自 profiler lookup）
- [ ] Random seeds: **Yes**（正确性实验 identical seeds）
- [ ] Hardware requirements: 4× NVIDIA A100 NVLink 节点；CUDA 12.1, PyTorch 2.6.0, CUTLASS 3.9, NVSHMEM 3.6.5, NCCL 2.29.3

### C. Limitations · Our take

**局限**
- **仅前向推理**：无 backward，无法直接用于训练。
- **规模有限**：4 GPU 单节点；未评 multi-node、更大 EP 宽度。
- **Hidden dim 敏感**：M-Trans-xl（dim=512）上输给 Megatron/Tutel——小通信量场景 fixed overhead 主导。
- **Layout padding 开销**：stress_skew 下 zero-row padding 增加无效 GEMM 计算（虽 excluded from transfer）。
- **Tuning 负担**：$cCTA$、$mgb$、CUTLASS tile config 需 per-workload 调优；无 adaptive selector（future work）。
- **无开源**：复现需自实现 CUTLASS epilogue signaling + NVSHMEM consumer + combine plan。

**Our take**
- 与 [`comet.md`](./comet.md) / [`uniep/`](./uniep/README.md) 同属 MoE combine overlap 问题，但本文走 **「非融合双 persistent kernel + MoE 专用 layout」** 中间路线——比 Tutel chunk pipeline 细，比 Comet/FlashMoE 侵入性低，适合作为 **CUTLASS epilogue hook + NVSHMEM consumer** 的 reference design。
- **Remote-owner-aligned layout** 是 MoE 动态 combine 做 tile overlap 的关键 enabler，与 Comet 的 shared-tensor 维度分解异曲同工但更显式。
- **$cCTA=2$ 比 baseline 慢 1.91×** 的实验很有价值：说明 overlap 不是「communication SM 越少越好」，production 需 runtime adaptive partition。
- 若扩展到训练，需处理 backward 中 combine/dispatch 的双向依赖；可参考 MegaScale-MoE 的 intra-op fuse 或 DisagMoE 的 AF-Pipe。
