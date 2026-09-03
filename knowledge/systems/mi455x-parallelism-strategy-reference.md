# MI455X / Helios 并行策略参考点（以 NVL72 为对标）

> **用途**：为 MI455X / Helios（CDNA5，72 卡单 scale-up 域）选择并行策略提供有据可查的参考点。
> **主要信源**：NVIDIA 官方 Megatron-Core MoE 论文 [2603.07685](https://arxiv.org/abs/2603.07685)（已有深读笔记 [`papers/megatron-core-moe.md`](../../papers/megatron-core-moe.md)，本文按 GB200/GB300 NVL72 相关内容重新提取）、Megatron-LM 主线 GTP 设计文档、AMD Helios 发布规格、[`papers/ultraep/`](../../papers/ultraep/README.md)。
> **建立日期**：2026-09-03
> **状态**：MI455X 侧全部为**推断**，未经实测。所有"Helios 列"的数字都要标记为待验证。

---

## 0. 一句话结论

**NVL72 的经验是：72 卡单域把"通信墙"直接用硬件拓扑消掉了，瓶颈随之移到 CPU 启动开销与计算效率。** 所以 MI455X 上真正该投的不是 DeepEP 类通信隐藏，而是 **HIP graph 捕获 + kernel 融合 + CPU/NUMA 绑定**，以及把省下来的显存换成更浅的流水线。

原文（§9.2.2）：

> On GB200 (NVL72), EP64 stays entirely within the NVLink domain. HybridEP fully utilizes the 1.8 TB/s bidirectional bandwidth **without requiring communication overlap. The communication wall is effectively resolved by hardware topology alone, shifting the dominant bottleneck to compute efficiency.**
>
> On GB200, where NVL72 already eliminates the communication bottleneck, **CPU overhead becomes the dominant constraint**; the host cannot launch kernels fast enough to keep the GPU saturated.

---

## 1. 硬件对标

| | GB200 NVL72 | GB300 NVL72 | **MI455X / Helios** |
|---|---|---|---|
| 架构 | Blackwell | Blackwell Ultra | **CDNA5 / TSMC 2nm** |
| scale-up 域 | 72 GPU | 72 GPU | **72 GPU** |
| 每卡 HBM | 192 GB | 288 GB (HBM3E) | **432 GB HBM4** |
| 每卡显存带宽 | ~8 TB/s | — | **19.6 或 23.3 TB/s**（AMD 两处口径不一致，待澄清） |
| 每卡 scale-up 带宽 | 1.8 TB/s 双向（NVLink5） | 1.8 TB/s | **3.6 TB/s 双向**（36×400 GbE UALoE） |
| 机架 scale-up 聚合 | 130 TB/s（单向）/ 260 TB/s（双向） | — | **260 TB/s** |
| 机架 scale-out | 800 Gb/s per GPU（CX-8） | — | **43 TB/s / 72 ≈ 597 GB/s per GPU** |
| 域内拓扑 | 单跳全互联 | 单跳全互联 | **单跳全互联（UALoE，12 交换机 × 21.6 TB/s ≈ 259 TB/s）** |
| 每卡 FP8 稠密 | — | — | **1.4 EF / 72 ≈ 19.4 PFLOPS** |
| 每卡 FP4 稠密 | — | — | **40 PFLOPS** |
| 互连标准 | 私有 NVLink | 私有 | **UALink over Ethernet + Ultra Ethernet（开放）** |
| 机架功耗 | — | — | **225–246 kW**（推测 >2 kW/GPU） |

### 关键判断

**scale-up 带宽大概持平，不要拿它当差异化。** AMD 报 3.6 TB/s/GPU、260 TB/s/机架；NVIDIA 报 1.8 TB/s/GPU、130 TB/s（另一处又说 all-reduce 可达 260 TB/s，说明 130 是单向）。**两边口径未对齐，2× 的推论不成立。** 决定性旁证：AMD 官方对比 Vera Rubin NVL72 时声称 **+15% 算力、+50% HBM 容量、+50% scale-out 带宽——唯独没有声称 scale-up 带宽优势**。这个沉默基本可以认定 scale-up 侧持平。

**真正的差异化是 HBM 容量：432 GB vs GB200 的 192 GB（2.25×）、vs GB300 的 288 GB（1.5×）。** 下面所有策略推断都建立在这一条上。

**带宽层次是 6:1**（域内 3.6 TB/s vs 跨机架 597 GB/s per GPU），和 NVL72 的层次结构同构。所以 NVL72 的映射原则可以直接借。

---

## 2. NVIDIA 的并行策略决策程序（原文 §9.1.2 的 5 条 Guideline）

这是 NVL72 类硬件上最直接可借的东西——一个显式的决策顺序。

| # | Guideline | 内容 | 对 MI455X 的适用性 |
|---|---|---|---|
| **G1** | 最小化模型并行、最大化数据并行 | TP/EP/PP/CP 在不 OOM 的前提下**尽量小**；用 distributed optimizer 把优化器状态切到 DP 上腾显存 | **直接适用，且 Helios 上更激进**——432 GB 让"尽量小"的下限更低 |
| **G2** | 把 EP 和 TP 的通信关在 scale-up 域内 | 确保 `EP × TP ≤ 域大小`（无 MNNVL 时通常是 8）。**超出域时优先用 PP，而不是把 TP/EP 拉across 节点** | 直接适用，域大小 = 72 |
| **G3** | 用 PP 做跨节点扩展 | PP ≥ 2 时开 VPP 降气泡；**VPP 越大气泡越小但 P2P 通信越多，中间值通常最好**；各 VPP rank 负载要均衡 | 适用，但 Helios 上 PP 可能更浅甚至为 1（见 §4） |
| **G4** | 专家层优先用 EP 而非 TP | EP 的优势：GEMM 更大更高效、通信量比 TP 小、计算图更简单易 overlap、**`EP = num_experts` 时本地 token permutation 被完全消除**。实例：Mixtral-8×7B 上 EP8×TP1 优于 EP4×TP2 | **直接适用**，这条是 Parallel Folding 的核心 |
| **G5** | 长序列开 CP | 序列 ≥ 8K 才开；< 4K 时 CP 开销大于收益；效率取决于通信能否与计算重叠 | 直接适用 |

**NVL72 实例（原文 §9.1.2 结尾）**：256 专家的 MoE 模型 → 按 G4，Parallel Folding 设 **expert TP = 1**（每个专家跑在单卡上，GEMM 效率最大化）→ 按 G2，**EP64 完整落在域内** → 按 G1，剩下的由显存预算决定 attention 层的 TP 和 PP，**DP 填满其余**。

---

## 3. 可复现的实测配置（原文 Table 11 + Table 20 + B.2）

这是最有价值的参考点：**同一个模型在大域和小域上收敛到根本不同的配置**。

| 模型 | 系统 | 卡数 | seq | 精度 | 并行配置 | TFLOPS/GPU | tokens/s/GPU |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 685B | **GB300** | 256 | 4k | MXFP8 | `TP1 PP4 CP1 EP64 VPP4 MBS1 GBS8192` | **1233** | 4730 |
| DeepSeek-V3 | **GB200** | 256 | 4k | MXFP8 | `TP1 PP4 CP1 EP64 VPP4 MBS1 GBS8192` | **1048** | 4020 |
| DeepSeek-V3 | GB200 | 256 | 4k | BF16 | `TP1 PP8 CP1 EP32 VPP4 MBS1 GBS4096` | 857 | 3298 |
| DeepSeek-V3 | H100 | 1024 | 4k | FP8-BLK | `TP2 PP4 CP1 EP64 VPP4 MBS1 GBS8192` | 368 | 1412 |
| Qwen3-235B | **GB300** | 256 | 4k | MXFP8 | `TP1 PP4 CP1 EP64 VPP06 MBS2 GBS3072` | **974** | 6583 |
| Qwen3-235B | GB200 | 256 | 4k | MXFP8 | `TP1 PP4 CP1 EP64 VPP06 MBS3 GBS3072` | 919 | 6212 |
| Qwen3-235B | GB200 | 256 | 4k | BF16 | `TP1 PP4 CP1 EP32 VPP12 MBS1 GBS8192` | 750 | 5100 |
| Qwen3-235B | H100 | 256 | 4k | BF16 | `TP2 PP8 CP1 EP32 VPP04 MBS1 GBS2048` | 320 | 2132 |
| Qwen3-235B | GB300 | 128 | **128k** | MXFP8 | `TP4 PP4 CP4 EP32 VPP12 MBS1 GBS1024` | 1150 | 1556 |

> ⚠ 论文内部有一处不一致：Table 17 写 H100 的 DeepSeek-V3 是 TP/PP/EP = `2/8/64`，Appendix B Table 20 写 `TP2 PP4 CP1 EP64`。引用时以 Table 20（Appendix）为准并注明。
> ⚠ 全部配置使用 **force-balanced routing**（强制均衡路由），即这些数字**不含真实路由不均衡的代价**。作者自己也说这些是"best-found"而非全局最优。

### 逐特性开关（原文 B.2）

| | DeepSeek-V3 GB300 | DeepSeek-V3 GB200 | DeepSeek-V3 H100 |
|---|---|---|---|
| Dispatcher | HybridEP | HybridEP | DeepEP |
| Recompute | **无** | `mlp` | `up_proj, mlp` |
| 1F1B a2a overlap | ON | **OFF** | ON |
| CUDA Graphs | `attn, moe_router, moe_preprocess` | 同左 | **OFF** |

| | Qwen3-235B GB300 | Qwen3-235B GB200 | Qwen3-235B H100 |
|---|---|---|---|
| Dispatcher | HybridEP | HybridEP | DeepEP |
| Recompute | — | `moe_act, layernorm` | `moe_act, layernorm` |
| 1F1B a2a overlap | **OFF** | ON | ON |
| CUDA Graphs | `attn, moe_router, moe_preprocess` | 同左 | 同左 |

**读法**：大域 + 大显存 → 重算可以关到几乎没有（GB300 上 DeepSeek-V3 **完全不重算**）、通信 overlap 可以关掉（GB200 上 OFF）、但 **CUDA Graph 必须开**。小域 → 反过来：必须 overlap、必须重算、Graph 关掉。

---

## 4. 大域到底值多少：两组量化数据

### 4.1 EP all-to-all 占训练时间的比例（原文 §4.2.1）

> 优化前 EP all-to-all 通常占训练时间 20–60%。**EP 留在 NVLink 域内时（DeepSeek-V3 EP64 on GB200 NVL72）约 20%；EP 跨节点时（EP64 on H100）升到 40–60%。**

全套优化后可压到 **< 10%**（原文 §4.2 结尾、§4.4）。

### 4.2 EP 规模化的延迟曲线（原文 Table 7）

hidden 7168、seq 4096、256 专家，单位 µs：

| 阶段 | EP | GB200 HybridEP | GB200 all-to-all | H100 HybridEP | H100 all-to-all |
|---|---|---|---|---|---|
| dispatch | 8 | 391 | 735 | 661 | 1265 |
| | 16 | 578 | 743 | 1485 | 5774 |
| | 32 | 612 | 769 | 3064 | 8059 |
| | 64 | **675** | **930** | **4626** | **9164** |
| combine | 8 | 353 | 741 | 624 | 1277 |
| | 16 | 527 | 765 | 1688 | 5628 |
| | 32 | 646 | 758 | 3088 | 7815 |
| | 64 | **744** | **827** | **4398** | **8727** |

**从这张表能推出两个对 MI455X 极其重要的结论：**

**（a）域内 EP 扩展几乎是平的。** GB200 上 dispatch 从 EP8 到 EP64（EP 度 8×）只从 391 → 675 µs，即 **1.73×**；H100 上同样跨度是 661 → 4626 µs，即 **7.0×**。combine 侧同理（2.11× vs 7.05×）。**这是"把 EP 关在 72 卡域内"这件事的全部价值所在**，也是 Helios 策略的最强单一论据。

**（b）域内 dispatcher 的选择远没那么重要。** GB200 EP64 上 HybridEP vs 朴素 all-to-all 只差 **1.38×**（675 vs 930）；而在 EP8 上差 1.88×，在 H100 EP16 上差 **3.9×**。**换言之：域越大、EP 越宽，专用 dispatcher 的边际收益越小。**

→ 这直接支持"**不要先移植 DeepEP，先测 UALoE 上朴素单跳 all-to-all 的裸性能**"。DeepEP 的价值是隐藏 RDMA 延迟，域内没有 RDMA。

---

## 5. 翻译到 MI455X / Helios 的策略推断

全部**未经实测**，需要 Projection 先扫、再上机验证。

### 5.1 起始配置建议

以 DeepSeek-V3 685B 为例。NVIDIA 在 192 GB（GB200）上收敛到 `TP1 PP4 CP1 EP64 VPP4`；Helios 有 **432 GB（2.25×）**，所以：

| 维度 | GB200 实测 | **Helios 推断** | 理由 |
|---|---|---|---|
| expert TP | 1 | **1** | G4，且 `EP=64` 时每卡 4 个专家、无本地 permutation |
| attention TP | 1 | **1**（可试 2） | G1 最小化；432 GB 不需要 TP 省显存 |
| EP | 64 | **64** | G2，64 ≤ 72；每卡恰好 4 个专家 |
| **PP** | **4** | **2 或 1（重点验证项）** | 2.25× 显存 → 流水线可更浅。GB200 靠 192 GB 从 PP8 降到 PP4，Helios 应能再降一档 |
| VPP | 4 | 随 PP 定；PP=1 时无意义 | G3 |
| CP | 1 | 1（长上下文另论） | G5，seq 4k 不开 |
| 重算 | `mlp` | **趋近于无** | GB300 在 288 GB 上已做到完全不重算；432 GB 更宽裕 |
| Graph 捕获 | CUDA Graph 三处 | **必须开**（HIP graph） | 见 §5.2 |
| a2a overlap | OFF | **先测再定** | GB200 上 OFF、GB300 上 ON，说明这条与显存/算力比值有关而非单调 |

**`PP=1` 是最值得验证的一条。** 如果成立，MoE 优化博客里两个最痛的问题同时消失而非被优化：调优后边界 rank 仍有 22.0% / 11.7% 的气泡；PP1 用到 88% HBM 而 PP15 只 23%（同等工作量跨 rank 64 个百分点落差）。两者都是 PP 的产物。

容量佐证：DeepSeek-V3 671B 按 16 B/param（bf16 参数+梯度 + fp32 master/m/v）约 **10.7 TB**，一个 Helios 机架 31 TB —— **一个机架装得下完整副本含优化器状态，占约 35%**，剩约 20 TB 给激活。

### 5.2 最重要的策略转向：瓶颈会移到 CPU 侧

NVL72 的教训是**通信墙被拓扑消掉之后，CPU 启动开销成为主约束**。且 FP8/FP4 会加剧这一点——原文 §9.2.3 第 3 条："FP8 shifts bottlenecks: FP8 accelerates GEMMs and reduces memory, **but amplifies CPU overhead as the dominant bottleneck**. CUDA Graphs, kernel fusions, and CPU/NUMA binding become essential."

MI455X 上 FP4 是 40 PFLOPS/卡，这个效应只会更强。对应动作：

1. **全迭代 HIP graph 捕获**——这是 roadmap 里的 C3-7，**应从"路线图后段"提前到前段**。NVIDIA 的做法是**部分捕获**：把 attention / router / MoE 预处理捕成静态图，动态的专家计算留在图外。这个"部分捕获"的划分方式可以直接照搬，比追求全迭代捕获更现实。
2. **kernel 融合**：router fusion、permute fusion、MLA RoPE fusion。
3. **CPU/NUMA 绑定**：按 local rank 检测 GPU/NUMA 拓扑并用 `numactl` 绑定。Helios 每托盘 4 卡 + 1 颗 96 核 EPYC Venice + 1 TB DDR，NUMA 拓扑和 8 卡 OAM 节点完全不同，**这块需要重做**。
4. **sync-free 执行**：device-initiated Grouped GEMM + 估上界预分配的 sync-free dispatch，消除 dropless MoE 里的 host-device 同步（代价是额外显存）。Primus 已有 Sync-Free MoE stage 0–3，需要按 CDNA5 重标定。

### 5.3 显存换性能的两条具体路径

**（a）优化器状态卸载。** GB200 借 NVLink-C2C 把优化器状态卸到 CPU：省 **15–20 GB（占 32.1 GB 优化器+权重预算的 47–62%）**，每迭代只多 **0.1–0.2 s**。Helios 每托盘 1 颗 EPYC + 1 TB DDR，但 **UALoE 不是 C2C，主机侧带宽路径完全不同**，这条能不能成立要单独测。相关：SuperOffload(ASPLOS'26) 明确把 MI300A 列为目标平台，但 GraceAdam 只为 Grace 实现。

**（b）通信 kernel 的 SM 占用。** Megatron-FSDP 用持久化双缓冲 + NCCL User Buffer Registration 做到零拷贝，**把通信 kernel 的 SM 占用从 8–32 个降到 1–4 个**（NVLink 系统上）。另有非均匀分片（把模块内所有参数扁平化拼接后非均匀切，使分片边界对齐通信缓冲布局），在 Llama3 405B 上降约 **10%** 通信开销。

这一条与我们已知的两个发现互证：Tessera 实测 **EP 通信 kernel 占约 20 SM 会导致 10–20% 减速**；Motif 3 独立佐证"规模上消除专家权重 All-Gather 优于隐藏它"。**RCCL 侧有没有 UBR 等价物、能否把通信 SM 占用压到个位数，是一个值得单独立项的问题。**

### 5.4 长上下文：分层 CP（原文 §6.3）

NVIDIA 的分层 CP 建议可直接借：

- **P2P CP**（环状交换 KV）通信天然与 SDPA 重叠；**all-to-all CP** 把张量从 seq 分片转成 head 分片再做 SDPA；**TP** 额外切 linear 权重（省参数显存）但在 linear 层引入额外 collective
- all-to-all CP 与 TP 都让 SDPA 跑在 head 分片上 → 每分片子序列更长 → **SDPA kernel 效率更高**
- **节点内优先 TP**（通信快、省显存收益大）；**跨节点优先 P2P CP**（TP 通信开销涨而 P2P 重叠仍有效）
- 推荐起点：**域内 all-to-all CP + TP，跨域 P2P CP**
- 还有 **Dynamic-CP**：按 microbatch 自适应选 CP 大小，与序列打包方案联合决定

对 Helios 的含义：72 卡域内"节点内"的定义被放大了 9 倍，**原本"跨节点才用 P2P CP"的边界应该外推到机架边界**。GB300 上 128k 序列的实测配置是 `TP4 PP4 CP4 EP32`——注意这里 TP 升到 4、CP 开到 4，与 4k 序列的 `TP1 PP4 EP64` 完全不同。

### 5.5 动态专家放置：ECHO 的做法

原文 §4.3 的 ECHO：前向时 planner 识别热专家，产出两个东西——热专家映射（哪些专家克隆到哪些空闲槽位）和更新后的路由映射（把溢出 token 重定向到克隆体）；Expert Dispatch 用**基于 HybridEP 的 sync-free 通信**把热专家权重拷到空槽；token 同时路由到本体和克隆体；反向时 Expert Gradient Dispatch 从克隆体收集梯度并归约回本体以保证一致性；克隆体算完即弃以省显存。

推理侧的对应物是 Wide-EP 的 **EPLB**：把热专家与冷专家重分布，**权重更新非阻塞地插在两次 forward 之间**，容器化设计使专家流入流出不破坏 CUDA graph。

对 Helios：72 卡单域内迁移是单跳，很便宜，所以这类重均衡的性价比比 8 卡时代高得多。但**注意与 §5.2 的 graph 捕获有直接张力**——EPLB 的"容器化以免破坏 graph"就是在解这个矛盾，EEP 也指出 graph 烘焙的路由元数据是弹性的障碍。**这两件事必须显式排序设计，不能各自推进。**

反方向的风险：72 卡的 blast radius 是 8 卡的 9 倍，加上每卡 >2 kW / 机架 245 kW 带来的热变异，**一张慢卡或一个热专家的影响面大得多**。UltraEP（[2606.04101](https://arxiv.org/abs/2606.04101)，**明确点名 AMD Helios**）做的正是机架级逐 microbatch 逐层精确均衡：不均衡度 1.3–4 → 1.0，训练 1.42× / serving 1.56×。⚠ 但 DODOCO 在 5 个真实 MoE checkpoint 上推翻了两个前提：路由不均衡无法在系统层完全纠正、mock-token benchmark 不代表生产路由。

### 5.6 低精度 wgrad reduce-scatter：该打哪一层

这条线的起因是 GTP 把权重 all-gather 压到原生 MXFP8/NVFP4 之后，**BF16 的 wgrad RS 反而变成每权重通信预算的 64%（bf16 RS）到 78%（fp32 RS）**。按分布式优化器下"每权重每步 ≈ param AG + wgrad RS"算：MXFP8 param(1 B) + BF16 wgrad(2 B) → wgrad 占 **67%**；NVFP4 param(0.5 B) + BF16 wgrad(2 B) → **80%**。压到 5 位后，NVFP4 配置下每权重总通信 2.5 B → **1.125 B（2.2×）**；保守到 8 位仍有 1.67×。**这是 per-weight 通信预算里最大的未开采储量。**

但 Helios 的拓扑对"该怎么做"给出了两个反直觉的约束。

**（a）机架内不需要"部分和感知"那套机制。** 72 卡全非阻塞下，带宽最优的 RS 不是 ring 而是 **direct / one-shot RS**：每 rank 把第 j 块直接发给 owner j，owner 本地累加 72 份。用 DynamiQ 的语言这是**深度 1、扇入 72 的星形 in-arborescence**，线上量与 ring 相同（每卡收发 `(n-1)/n·d`）但一跳完成、中间零重压缩，sink 可以用任意精度免费累加。代入其误差界：所有子树大小为 1，MSE 从 ring 的 `O(εSM²n³)` / butterfly 的 `O(εSM²n²)` 退化到 `O(εSM²n)`。**fabric 免费解决了溢出与误差累积问题，`decompress_accumulate_recompress` 一次都不会被调用。** 这恰好就是 AGoQ 的结构。

**（b）更关键的是带宽比值——这一层做不划算。** 这类方案本质是**用 HBM 流量换线上流量**：每坐标线上省 `ΔW = 2.75·AR` 字节，额外 HBM 付 `ΔH = 18 + 7.875·AR` 字节。盈亏平衡要求 `B_HBM / B_wire > ΔH/ΔW`，`AR = 71/72` 时门槛是 **9.50**：

| 环境 | HBM 带宽 | 每卡线速 | 比值 | 判定 |
|---|---|---|---|---|
| 论文 testbed（A6000） | 768 GB/s | 12.5 GB/s | **61.4** | 盈利 6.5× 门槛 |
| MI355X 单节点 XGMI | 8 TB/s | ~1.075 TB/s | **7.4** | 略亏（0.78×） |
| **MI455X 机架内（UALoE）** | 19.6–23.3 TB/s | 3.6 TB/s | **5.4–6.5** | **亏（0.57–0.68×）** |
| **MI455X 跨机架** | 19.6–23.3 TB/s | 0.597 TB/s | **32.8–39.0** | **盈利 3.5–4.1× 门槛** |

（HBM 取 AMD 两处口径的上下界；结论对取哪个不敏感。若两侧都改双向口径，绝对值变但方向不变。）

**结论：低精度 wgrad RS 应该瞄准跨机架 DP 路径，而不是机架内。** 这也从一个完全独立的方向解释了为什么论文的加速比不可外推——它的 testbed 比值 61，MI455X 机架内只有 5.4–6.5，差一个数量级。

**（c）6:1 的层次让多跳问题变小而不是变大。** 覆盖 DP 规模 N 所需的跨机架深度是 `log₂(N/72)` 而非 `log₂N`——DP=8192 只需约 114 个机架、butterfly 深度 7，而论文那些 582×/1283× 的优势是在**扁平** n=8192（深度 13）下取的。**Helios 的层级结构已经免费砍掉了 6–7 层**，所以"部分和感知"相对"MXFP8 + 逐跳重标定"的边际价值远小于 headline。

**（d）跳数无关、两层都成立的部分才是真正可借的。** DynamiQ 消融里贡献最大的一项是**变量位宽分配**（3.5–5.1×，单项超过其余三项之和），它纯粹利用梯度偏斜——LLaMA 约 20%、Gemma 约 30% 的 super-group 范数比中位数低数量级，而 MXFP8 的 8.5 位处处等价。另有层次化 scale（约 30%，group=16 + UINT8 scale，元数据开销与 MXFP8 的 chunk-32 + BF16 scale 相同但粒度细一倍）和相关舍入（约 35%，在深度 1 下**更强**，因为 72 个 leaf 同时喂一个 sink 正是经典 DME 场景）。

**（e）⚠ 重要修正：A2A + 本地 FP32 归约这个"结构"上游已经出货了，缺口在格式而不在结构。**

不要把 A2A 重构当成我们的差异化点。已在 `fb7ba4c83` 上验证：

- `megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py` **存在**（4207 字节）
- GTP 文档 §2.6 有 `--gtp-remat-reduce-scatter-with-fp32-accumulation`，原文：*"A ring reduce-scatter rounds the partial sum at every one of its `N-1` hops, so BF16 gradient error compounds with the axis size (≈`√N` for gradient-like data...). This flag replaces it with an **all-to-all plus one local FP32 sum**, eliminating that accumulation error **for the same bytes on the wire**."*
- 还有 DP 轴的孪生 flag `--ddp-reduce-scatter-with-fp32-accumulation`，两者独立
- §2.7 `--gtp-remat-nccl-ub` 走 NVLS，`multimem.ld_reduce` 对 BF16 输入用 `.acc::f32` **在交换机内以 FP32 累加**，且对称池优先级高于 A2A 路径（自动 bypass）
- 更早的先例：**ZeRO++ 的 qgZ（[2306.10209](https://arxiv.org/abs/2306.10209)，2023）** 已发表同一结构，还多了 2-hop 层级化与 slice reordering。AGoQ 引用了 ZeRO++ 却从未提及 qgZ

注意那句 "for the same bytes on the wire"——**上游这个 flag 修的是精度，不是带宽。** 所以：

**GTP 把 wgrad RS 留在 BF16 的真正原因不是溢出，而是通信库里没有块缩放数据类型。** NCCL 只有 `ncclFloat8e4m3/e5m2`；MXFP8/NVFP4 的 AllReduce 是 issue #2199 的**未实现** RFE，难点被列为 "block-scaled datatype semantics, cross-rank scaling/alignment"。加上 NVL72 上 NVLS 已经免费给了他们 FP32 精确 RS，只剩字节问题，优先级自然低于把 AG 侧做稳。

**而更根本的数值障碍是尾数宽度，不是动态范围。** block scale 锁住指数，但把 W 个数相加需要约 `log₂W` 位额外尾数——W=72 时 **6.2 位**，而 FP8 E4M3 只有 3 位、FP4 只有 1 位。正确的表述是**"累加器必须永远比线上格式宽"**，而不是"FP8 会溢出"。

**（f）MX 格式在加法下不封闭——这不是精度问题而是类型系统问题。** 三层独立成立的原因：① **和的 block scale 是数据相关的**，必须由和的 amax 决定，而那要先算出和；② 即使 scale 相同，尾数也放不下（同上 6.2 位 vs 3/1 位）；③ ring 下误差双重放大——`W−1` 次 requantize 按 `√(W−1)` 累积（W=72 → √71 ≈ 8.4），而 MXFP8 单次相对误差约 2⁻⁴ vs BF16 的 2⁻⁸ 差 16×，**乘起来原生 MXFP8 ring RS 比 BF16 ring RS 差约 135×**。

推论：**A2A + 本地 FP32 归约对 MX 不是优化而是唯一正确结构**（量化往返压到 1 次，误差与轴宽解耦——这正是 W=72 可行而 ring 灾难的原因）；分片边界必须 pad 到 32（MXFP8）/16（NVFP4），否则块被劈开两边都无法反量化，这就是 GTP `pad_for_alignment` 存在的理由；scale 元数据必须同行，线上实际是 **1.03125**（MXFP8）和 **0.5625**（NVFP4）B/elem，不是 1.0/0.5。

**（g）所以我们该建的形态是：上游已验证的结构 + 低精度块缩放线上格式 + DynamiQ 的位分配。** 具体：`按（上一步的）F_j 分位宽` → `零均值化 + 重排` → **`压缩后 A2A`** → `owner 本地 FP32 累加` → **一次下转** → 结束。

**关键简化：ZeRO-3 / GTP 只需要 reduce-scatter，AllGather 那条腿和它那次 re-quantize 都不存在——不要照抄 AGoQ 的 Fig. 5，抄它的左半边。**

**（h）AMD 没有 NVLS，反而让这条路成为必需品而不是可选项。** GB200 上 NVLS 在交换机内做 FP32 归约，A2A 路径只是备选（对称池优先）；Helios 没有等价物，**A2A 重构是拿到 FP32 精确归约的唯一途径**。所以它在 Helios 上比在 GB200 上更有价值——但要清楚这部分是**追平**，差异化在格式那一层。

**（i）硬边界：flat 1-hop A2A 只能用在 ≤72 的域内轴上。** ZeRO++ 踩过这个坑：flat 1-hop A2A 跨节点时通信量 blow up（`M·N/Z` vs 层级化的 `M/Z`，N 倍差距），而 ring RS 按 rank 序排环天然层级化。**Helios 让 W=72 整个装进单跳非阻塞域、没有节点边界，flat A2A 不需任何层级化修补，跳数从 71 降到 1——同时是字节最优、跳数最优、数值最优。** 但跨机架（6:1）必须上 ZeRO++ 的 2-hop。NVIDIA 正是这么做的：GTP64 ≤ NVL72。

W=72、N = 4096×14336 = 58.72M、单向 1.8 TB/s 的估算：BF16 ring RS 115.8 MB → 64.3 µs；MXFP8 A2A 59.7 MB → 33.2 µs；NVFP4 32.6 MB → 18.1 µs。本地 FP32 归约约 63 MB HBM 流量、无转置，70% 带宽约 9 µs / 35% 约 18 µs。**MXFP8 合计 42–51 µs，赢 1.26–1.53×；归约若能重叠则 1.94×；NVFP4 是 1.8–2.4×。** 注意 1.26× 与 1.94× 的差别全押在能否重叠上，也就是押在 C3-4（RCCL overlap）上。

**（j）一个更大的目标，但它与本文推荐的主力配置冲突——需要显式取舍。** AGoQ 的 Table 5 显示 **TP/SP 的 all-gather / reduce-scatter 占逐层墙钟 forward 的 69%、backward 的 59%**，而它一个字节都没动。SP 的 activation reduce-scatter 是货真价实的"在通信中做加法"，同一套论证一字不改地适用，**目标比 DP 梯度归约大 3–6 倍**。

⚠ **但注意矛盾**：本文 §5.1 推荐的主力配置是 **TP=1**（对标 GB200/GB300 实测），而 `megatron/training/arguments.py:1435-1441` 会在 `TP==1` 时**静默关闭 sequence parallel**。**TP=1 时这块通信本身就是零，这个"3–6 倍的机会"不存在。** 所以两者互斥：

- **走 TP=1 主力路线** → TP/SP 的 ag/rs 归零（这本身就是收益，也是 TP 退化的最强论据），低精度通信只剩 DP/GTP 轴可做
- **只有在 TP>1 的场景下这个机会才成立** → 即长上下文（§5.4，GB300 上 128k 实测是 `TP4 PP4 CP4 EP32`）和稠密模型（无 EP 可替代）

**结论：把它定位成"长上下文与稠密路线的专项"，而不是主力 MoE 路线的后段。** 排期上应晚于域内 wgrad RS，且立项前先确认目标 workload 的 TP 是否 >1。

**（k）AGoQ 本身的数字要打折看。** 52% 显存节省的归因是：8-bit Adam 8.4 GB、激活量化 13.0 GB、**梯度量化只有 2.4 GB（占总节省 5.2%）**；1.34× 加速来自 LLaMA2-13B / 64 GPU / 80K 序列，而同一张表显示重算层数从 R=10 降到 R=0——**加速几乎全部来自激活省显存换掉重算**，且该配置 DP = 64/(4×8) = **2**，梯度归约只有两个参与方，重构基本没起作用。另有多处内部矛盾（Table 7 表头 samples/sec 但三行 AGoQ 全最低，反推是作者把"基线÷AGoQ"当成了加速比；Table 4 的 1 GB 行物理不可能；Eq. 21 与正文方向相反）。**它的 3.4× 全部来自位宽而非结构，且从未测原生低精度 RS 的对照组。**

**一个论文没做但便宜的改进**：第一步的轻量 all-reduce 虽只占 0.125 位/坐标，却是主 RS 之前的**串行同步点**；而论文自己的数据显示 vNMSE 跨训练步基本平坦（即 `F_j` 高度自相关）。**直接用上一步的 `F_j` 定这一步的位宽，可以把这个 collective 从关键路径上完全拿掉**，代价只是一步陈旧度。

**⚠ 立项前必须先证伪的假设**：整条线建立在"生产 MoE 梯度的 super-group 范数偏斜程度与 LLaMA-1B 微调梯度相当"之上。**而 MoE 梯度结构上就不一样**——专家权重只从被路由到的 token 拿梯度。论文 §8 提到未激活专家可给 0 位，暗示这可能是**更大**的机会，但零实验。所以第一步应该是：从 Primus 真实 run dump 出 DSv3 / Llama 的梯度快照，画 vNMSE-vs-位宽曲线叠 MXFP8。**若 ≤6 位处打不过 MXFP8 就停**——这个证伪只要两周，却决定后面 7 周做不做。

**⚠ 与 AMD 自家结论的交叉校验**：Penn State + AMD 在 MI355X 上实测 **wgrad 量化是收敛退化的主因**（到 ppl 3.3 的 token 开销 Fprop 8–9% → +Dgrad 10–11% → **+Wgrad 26–27%**），且**随机舍入与随机 Hadamard 在全流水下不收敛、确定性 Hadamard 才救回来**。相关随机舍入不等同于随机 Hadamard，但这条说明 wgrad 路径对注入随机性异常敏感。收敛验证必须跑三方 A/B：相关随机舍入 / 独立随机舍入 / 确定性最近邻。

**（l）不要从"几何变换"那条路起步。** GIFT 的保真度分析是扎实的（FP8 单步往返 RelL2 **−67.4%**，且对角近似几乎无效 → 有用的几何是**跨维耦合**的，不是逐维 scale 能抓的），但它的端到端结论站不住：**同一张表里直接欧氏 FP8 是 −10.79%，GIFT 只有 −7.6%，即比"什么几何都不做"慢 3.19 个百分点**；质量优势也经不起头对头（600M 上欧氏赢 8 项 / GIFT 赢 6 项，14 任务均值欧氏 0.5186 > FP32 0.5060 > GIFT 0.5032）；而且基线是 FP32 而非 BF16——换成 BF16 后直接 FP8 只剩 **+3.9%**、GIFT 只剩 **+0.4%**。论文还缺了最关键的一个消融：**"欧氏 + 误差反馈"**（GIFT 分支有 EF 和局部 scale，欧氏基线两样都没有），很可能这一项就把差距填平了。

顺带一个论文没写、但会在长 horizon 上咬人的问题：**误差反馈缓冲累积在变换坐标系里，而坐标系每 50 步刷新一次**——刷新时旧基的残差被加到新基的梯度上、再用新的 `L_A^⊤` 映射回去，**这是基不匹配，会注入正比于 `(L_A^new − L_A^old)^⊤ R` 的伪更新**。加上无阻尼的病态求逆，正好是慢性发散的形状。这与上面 MXFP4 那条"确定性才收敛"互为呼应。

**（m）该先做的两件便宜事。** 一是 **`确定性 Hadamard + FP8 wgrad RS`** 作为对照组——kernel 在 ROCm TE 里已有，一次消掉低秩因子维护、跨 rank 一致性、坐标刷新、EF 重定基这四个工程负担，而且最符合 AMD 自家的 MXFP4 发现。二是**先做不带几何的纯 FP8 wgrad RS，并补上论文省掉的"欧氏 + EF"消融**——如果它就填平了差距，我们既做完了又比 GIFT 快。

**（n）更好的地基是 SDP4Bit，不是这三篇里的任何一篇**：collective 选对了、在 Megatron 内、开源、验到 6.7B/18B，而且它的**两级结构正是 Helios 需要的形态**——域内 BF16 免费归约，只对跨机架的 `1/k` 分片做量化，算力开销和 EF 显存同时降到 `1/k`。这与 §5.6(b) 的带宽比值结论（机架内亏、跨机架盈）和 (i) 的结构结论（flat A2A 只在 ≤72 域内）完全吻合。

**（o）如果真要做几何，它在 reduce-scatter 上反而比在 all-reduce 上更合适**：变换是**右乘、逐行独立**，所以沿 `d_out` 的行分片天然封闭，map-back 只在收到的 `1/N` 行上做，不需要额外 collective。纯数学、零 NVIDIA 依赖、不碰 GEMM 路径、不需要 TE，RCCL 已原生支持 FP8 e4m3/e5m2 和 reduce-scatter。三个前提：分片边界要按行对齐（改 `ParamAndGradBuffer`）；`A` 必须跨 rank 一致（这是正确性问题，论文没写）；**TP 下最麻烦——GIFT 选中的层全是 `fc2`，其 `d_in` 正是 row-parallel 被切的那一维**，出路是对 `[d_out, 32]` 的 sketch 做一次 TP all-reduce，额外流量约 0.4%。实现要点：变换的 flops 只占前反向的 **0.03%**，那 3.19 个百分点全是 HBM 往返——所以要把「EF 累加 → 变换 → amax → 量化」融成一遍、梯度只读一次；且别物化 `A`（`d_in=28672` 时是 3.3 GB），用随机化 range finder。

**（p）Muon 线的一个可测推论**：GIFT 其实没有分析 Muon（只是用了它，无 Muon vs AdamW 对照）。但机制上：Newton-Schulz 压平奇异谱，会放大小奇异值方向的噪声；单一欧氏 scale 让相对误差在那里最糟，而白化把噪声塑造成 `∝A`、恰好均衡相对误差——这正是 NS 这类尺度不变算子想要的。**推论：低精度梯度通信在 Muon 下应比 AdamW 更脆弱，这个对照值得单独跑。** 另外 owner placement（DMuon / MatrixFSDP）让完整矩阵整块落在单 rank 上，**直接消掉 (o) 的全部分片问题**，两者正交可叠加。

详见 [`papers/dynamiq.md`](../../papers/dynamiq.md)、[`papers/agoq.md`](../../papers/agoq.md)、[`papers/mxfp4-pretraining.md`](../../papers/mxfp4-pretraining.md)。

---

## 6. 需要修正的一处既有判断

之前记录"zero-bubble / DualPipe 在上游 Megatron 是 0 命中"，这个说法需要收窄。

代码层面仍然成立：`rg -ri 'dualpipe|zero.?bubble' megatron/ docs/` 在 `fb7ba4c83` 上确实 0 命中，upstream 没有 zero-bubble / DualPipe **调度族**的实现。

但论文 §4.2.3 明确写道，它的 1F1B all-to-all overlap 方案"**conceptually is a DualPipe-like bidirectional schedule built on top of standard 1F1B while preserving Megatron-Core compatibility**"——即合并相邻 microbatch 的前反向、跨 CUDA stream 交错计算与 all-to-all kernel，让一个 microbatch 的反向 a2a 与另一个的前向 attention/MLP 重叠。

**所以准确表述是：上游有 DualPipe 式的双向重叠思想（目标是隐藏通信），但没有 zero-bubble / DualPipe 的调度实现（目标是消除气泡）。** Primus 的 31 个 scheduler 文件仍然是差异化，但护城河比原先记录的窄，且如果 Helios 上 `PP=1` 成立，其适用范围会进一步收窄到"跨机架 PP"和"单机架放不下的模型"。

---

## 7. 待验证清单（按优先级）

| # | 待测项 | 为什么关键 | 怎么测 |
|---|---|---|---|
| 1 | **RCCL 在 UALoE 上的 per-op 启动延迟** | 决定 GTP 类逐权重预取策略在 AMD 上是否成立（GTP 放弃 bucket 换逐权重，代价是大量小 collective） | micro-benchmark，扫 message size 与 op 数 |
| 2 | **UALoE 单跳 all-to-all 裸性能**（对标 Table 7） | 决定要不要移植 DeepEP。GB200 上 EP64 时专用 dispatcher 只领先朴素 a2a 1.38× | 复刻 hidden 7168 / seq 4096 / 256 专家 / EP 8→64 的扫点 |
| 3 | **`PP=1` 或 `PP=2` 是否可行** | 直接删掉气泡与显存倾斜两个问题 | Projection 先扫（已有 node/pod 两级建模），再上机 |
| 4 | **RCCL 有无 NCCL UBR 等价物**，通信 kernel SM 占用能否压到个位数 | Tessera 实测 20 SM 通信 kernel 致 10–20% 减速 | rocprof 数通信 kernel 的 CU 占用 |
| 5 | **CDNA5 上现有 kernel 的可移植性** | "十余年来最大架构改动"，FlyDSL / HK tile / CK 配置 / hipBLASLt 调优缓存都不保证平移 | 拿现有 micro-benchmark 全套重跑 |
| 6 | **Helios NUMA 拓扑下的 CPU 绑定** | 每托盘 4 卡 + 1 颗 96 核 EPYC，与 8 卡 OAM 节点完全不同 | 重做 bindpcie 等价物 |
| 7 | 主机侧卸载路径带宽（UALoE ≠ NVLink-C2C） | 决定优化器状态卸载能否复现 GB200 的 47–62% 节省 | 测 H2D/D2H 有效带宽与重叠度 |
| 8 | **CDNA5 的 copy engine 能否打满 UALoE** | Primus 现有 `hipMemcpyDeviceToDeviceNoCU` 路径（零 CU param all-gather）的全部价值建立在此。MI355X 的 XGMI 与 UALoE 的 3.6 TB/s 不是一个量级 | SDMA vs RCCL 零 CTA 两条路径在 UALoE 上对打，扫 message size |
| 9 | **SDMA 路径的 launch 开销在大域上是否反转** | 每 bucket 发 `world_size` 个 `hipMemcpyAsync`（DP64 = 63 peer + 1 local，摊在 8 条流上）+ 2 次 barrier。**§5.2 说大域上瓶颈本就移到 CPU/launch**，SM 不再是瓶颈后这些 launch 可能变成新瓶颈 | 对比 SDMA 多 memcpy vs RCCL 单 collective kernel，扫 world_size 8→72 |
| 10 | 非均匀分片（Megatron-FSDP 那项）值多少 | **这是 NVIDIA 有、Primus 完全没有的唯一一项**（Llama3 405B 上 −10% 通信）。Primus 仍用 MCore 的 `shard_buffer` 均匀 bucket 分片 | 先量当前分片边界与通信缓冲不对齐带来的额外搬运量 |
| 11 | **reduce-scatter 侧的零 CU 路径** | 现状：param AG 已零 CU，但梯度 RS 仍跑 RCCL kernel 且是 BF16，**同时是带宽瓶颈和 SM 瓶颈**。`NoCU` 全树只出现 1 处 | 与 §5.6 的低精度 RS 一并设计，不要分两条线做 |

---

## 8. 相关索引

- 论文深读：[`papers/megatron-core-moe.md`](../../papers/megatron-core-moe.md)（[2603.07685](https://arxiv.org/abs/2603.07685)）、[`papers/ultraep/`](../../papers/ultraep/README.md)（点名 Helios）、[`papers/moe-parallel-folding.md`](../../papers/moe-parallel-folding.md)（[2504.14960](https://arxiv.org/abs/2504.14960)，打破 EP ≤ DP）、[`papers/tessera.md`](../../papers/tessera.md)（20 SM 发现）、[`papers/motif-3.md`](../../papers/motif-3.md)（消除优于隐藏、逐层选 CP）
- 低精度通信线（wgrad RS 前沿）：[`papers/gift.md`](../../papers/gift.md)、[`papers/dynamiq.md`](../../papers/dynamiq.md)、[`papers/agoq.md`](../../papers/agoq.md)
- 硬件规格：[`../hardware/gpu-comparison.md`](../hardware/gpu-comparison.md)
- Primus roadmap：[`../../notes/primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md`](../../notes/primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md)
- 行业全景：[`industry-training-optimization-2026.md`](./industry-training-optimization-2026.md)、[`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md)
