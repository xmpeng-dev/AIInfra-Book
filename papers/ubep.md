# UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods

> [arXiv 2607.06202](https://arxiv.org/abs/2607.06202) (v2, 2026-07, cs.DC) · **SIGCOMM'26**（2026-08-17~21, Denver）· DOI [10.1145/3789240.3829183](https://doi.org/10.1145/3789240.3829183)
> 单位：**南京大学**（State Key Lab for Novel Software Technology）+ **华为**。一作 Yipeng Liu 在南大，共同通讯 `jzheng@nju.edu.cn`（第一）/ `huzhongzhe@huawei.com`（第二）；19 位作者里南大 9 / 华为 10 —— **联合工作，不是华为主导**
> 硬件：华为 **CM384 超节点**（两层 UB switch，UGAS 统一地址空间，单链路 ~392 GB/s）· 实测 16 台 Ascend server = **256 NPU die**
> 软件：CANN 栈上约 **10K 行 Ascend C** · 端到端走 vLLM-Ascend v0.14.0
> 模型：Qwen3-30B / GLM-4.7 358B / DeepSeek-R1 671B / DeepSeek-V3.2 685B
> **口径警告：全文只评测推理解码（算子延迟 + TPOT），没有任何训练实验。**

## TL;DR

**这是"不写 megakernel、改写通信库"这条路线目前最强的一篇**，而且它在 §6 里点名批评了我们这条路线。

它把 MoE 的 All-to-All 从 BSP（阶段 + 全局 barrier）改成**依赖驱动的细粒度任务**，三个设计：kernel decomposition（AIV 分组并行跑独立子任务）、hierarchical token-level scheduling（同时均衡负载与 hop 距离）、**Data-as-Flag**（把同步信号塞进数据负载，靠 512B 原子写实现隐式同步）。dispatch 算子 **34.7%–52.4%**，端到端 P99 TPOT **≤11.1%**。

**最该记住的三个数**：

1. **MoE 通信占每 token 延迟约 50%，但只占实际硬件执行时间约 20%**（§5.5 profiling）——剩下 30 个百分点是依赖阻塞与运行时开销。这是"该重排执行、而不是加带宽"最干净的第三方证据，可以直接引用进我们的 megakernel 论证。
2. **`SyncAll` 单项占总执行时间约 15%**（Insight 2）——他们管这叫 **synchronization tax**：带宽涨一个数量级之后，同步原语的相对成本变成主导。这正是 cco Technique 3.2「fence 预算」原则缺的那个量化。
3. **只要一半 AIV 就能打满带宽，24→48 个通信核几乎没有收益**（Insight 1，Figure 3b）——多出来的核全在 barrier 上等。这直接支持 MonolithEP 的 WG 角色分区逻辑：通信角色给到带宽饱和点就够，剩下的 CU 全部划给 compute。

**必须诚实面对的一点**：论文 §6 明确把自己和 fused persistent kernel 对立起来——"Unlike fused persistent kernels that often rely on **rigid resource allocation** and **software-managed completion polling**, UBEP uses receiver-driven data signals and instruction-level atomics"。这两条批评对 MonolithEP 都成立（我们的 224:16:16 就是静态刚性划分，我们的 ready flag 就是软件轮询）。见 §6.2。

## 1. Problem

### 1.1 超节点把瓶颈从"传输"换成了"同步"

| | 传统集群（IB/RoCE v2） | CM384 超节点（UB） |
|---|---|---|
| 带宽 | 50 GB/s | **392 GB/s**（涨一个数量级） |
| 语义 | 显式消息传递 | **UGAS**：统一全局地址空间，coherent load/store |
| 后果 | 传输时间长，**同步开销被掩盖** | dispatch 迅速打满带宽，**阶段级同步成为主导约束** |

MoE 通信平均占总执行时间 ~47%（论文引 Comet 等 [24,49]），所以这件事值得单独重做一层库。

### 1.2 三个瓶颈（论文的立论骨架）

1. **BSP-style Serialization。** dispatch 被切成 Init → Token-Sending → SetFlag & Verification → CalCumSum → Token-Reordering 五阶段，阶段间插全局 barrier（`SyncAll`）。同一阶段内所有 AIV 必须干同一类活，快的等慢的。**没有依赖关系的子任务也被强行串起来。**
2. **Synchronization Tax。** flag / barrier / kernel launch 这些控制面原语，在超低延迟 fabric 上吃掉端到端预算的可观比例（`SyncAll` ≈ 15%）。论文说这是此前没有在 EPCL 语境下被量化过的。
3. **Topology-Agnostic Scheduling。** UGAS 给了"逻辑上平坦"的地址空间，但物理上是两层交换，**NUMA 在集群尺度上回来了**。现有实现只按 token 数均衡：

| 访问层级 | 延迟 | 归一化 |
|---|---|---|
| Intra-NPU（本地 NPU 跨 die HBM） | 218 ns | 1.0× |
| One-Hop（穿一层交换） | 929 ns | 4.3× |
| Two-Hop（穿两层交换） | **2500 ns** | **11.5×** |

只均衡 token 数、不管 hop 距离，会**反过来制造 straggler**——这是 Insight 3，也是本文最容易被低估的一条。

### 1.3 现有 EPCL 的定位（论文 Table 1）

| 库 | 网络 | 目标架构 | Pipeline | 同步模型 | 调度粒度 |
|---|---|---|---|---|---|
| DeepEP | IB+NVLink | 非超节点 | Inter+Intra | BSP | Kernel |
| UCCL-EP | 异构 | 非超节点 | Inter+Intra | BSP | Kernel |
| Hybrid-EP | IB+NVLink | 单层超节点 | Inter+Intra | **ASP** | Warp |
| CANN EP（基线） | UB | 两层（base） | Intra-only | BSP | Kernel |
| **UBEP** | UB | 两层（opt） | Intra-only | **ASP** | **Core（AIV）** |

## 2. Contribution

- 刻画 BSP 式 EPCL 在超节点上的瓶颈，提出并量化 **synchronization tax**。
- 三个机制：**token 级 kernel 分解**、**层次化 token 调度**（负载 × hop 距离联合优化）、**Data-as-Flag** 同步协议。
- 在生产 CM384（256 die）上实测，A2A 延迟 ↓ ≤52.4%，端到端 TPOT ↓ ≤11.1%。

## 3. Method

### 3.1 Kernel Decomposition（破串行）

不再要求所有 AIV 走同一条任务序列，而是**按角色分组**：

- 多数 AIV 干 **Token-Sending**；
- 少数 AIV 干 **TokenCnt-Sending**，metadata 到齐后算前缀和，得到 per-expert 的 HBM 偏移供 Token-Reordering 用。

于是 **TokenCnt 的处理延迟被 Token-Sending 的通信延迟盖住**，且不占传输带宽。分组比例是个受约束优化问题：counts 组 ∝ 专家数，sending 组 ∝ batch × top-k（代价模型在 Appendix A，实测在 Appendix C.3）。

唯一的真依赖（Token-Reordering 要等 CalCumSum 的地址）改成**点对点数据信号**：CalCumSum 的 AIV 把地址写进全局共享内存，其他 AIV 轮询该偏移，一变就立刻开工。**没有 `SyncAll`。**

### 3.2 Hierarchical Token-level Scheduling（破 straggler）

硬件加速的 mapper，1 µs 内出调度，两个部件：

1. **Expert Remapping via Logical Matrix Transposition**：构造 AIV × expert 的虚拟矩阵，**按列主序读**，让每个 AIV 拿到的 one-hop / two-hop 专家是均匀混合的 —— 把期望通信延迟同质化。
2. **Token-level Load Partition**：全局 token 序列切等长片，每 AIV 一片，任意两 AIV 负载差 ≤ 1 个 token。因为 remap 后 per-expert token 数不均，查"第 i 个 token 属于哪个专家"是个前缀和搜索，用**四分搜索 + `VectorCount` SIMD 指令并行取三个 pivot**，4–5 步收敛、`<1 µs`。

作者说这套可推广到任意 hop 层数，只要每层延迟可区分。

### 3.3 Data-as-Flag（破同步税）

核心前提：**UB fabric 保证 512B 原子写**（写整块原子可见、后续读被串行化）。三个变体，是一条 payload 效率 vs 流水粒度的权衡链：

| 变体 | 做法 | payload 效率 | 代价 |
|---|---|---|---|
| **TFF**（Token-Flag Fusion） | 512B DataBlock = 32B flag + 480B payload，原子写一次 | 480/512 = 93.75% | 发送侧无 barrier；但 flag 占带宽 |
| **DC**（Data-for-Checksum） | 不带 flag，发完整批后单独发累积 checksum | 高 | **要等整批 checksum**，破坏 token 级流水 |
| **SP**（Sentinel Polling） | 接收侧预填哨兵值，发送侧发裸 payload，接收侧比对指定 32B 是否变化 | 理论 100% | 需消费后重置 buffer；**payload 撞哨兵会死锁** |

SP 的死锁他们靠"保留一个正常计算不可能产出的 BF16/FP16 位型"来消除，并在模型初始化时校验权重与激活不会产生它；32B 哨兵在均匀分布下碰撞概率 2⁻²⁵⁶。

论文给了每个变体的 happens-before 论证（TFF：原子性保证 flag 与 payload 同时可见；DC：checksum 写充当 per-batch barrier；SP：原子写保证接收侧不会看到半块）。

## 4. Experiments

**Setup**：生产 CM384，16 Ascend server / 256 NPU die。基线是**同硬件同拓扑的 CANN EP**；DeepEP 只作为 H800 上的"协议能力参照"列在 Table 4，**不是 apples-to-apples 对比**。

| 项目 | 结果 |
|---|---|
| dispatch 算子（EP=64，1 expert/NPU，扫 batch size） | **↓34.7%–52.4%** vs CANN EP |
| 有效带宽（同硬件同拓扑，隔离算法收益） | **+35.3%–40.8%** |
| BS 敏感性（128 rank，BS 8→64） | 平均 **~46.4%**；BS=128 掉到 35.9%（通信量变大，收益被摊薄） |
| 端到端 P99 TPOT | **↓≤11.1%**（每模型 ≥100 个 decoding step） |

**消融 1（调度）**：64 NPU / 256 expert / top-8 / 四个 16-NPU 节点 + 一层 Tier-2 交换 / 8 个热专家，看单节点 48 个 AIV 里 28 个代表性 AIV 的最大传输延迟：

| 配置 | 最大 AIV 延迟 | 问题 |
|---|---|---|
| CANN EP | **62.2 µs** | 热专家把发送集中到少数 AIV |
| UBEP w/o mapping | 48.1 µs | 负载均了，但拓扑无关 → 大多数传输穿 Tier-2 |
| **UBEP** | **43.5 µs** | 负载与 hop 联合均衡，近似均匀 |

**消融 2（同步）**：TFF / DC / SP 三个变体在不同同步粒度与 rank 数下，相对 CANN EP 的 stop-and-wait **降 31.0%–57.1%**。

## 5. Limitations

**作者自陈的可移植性边界（Table 5）** —— 这张表对我们特别有用：

| 依赖 | 用在哪 | 没有它怎么办 |
|---|---|---|
| **512B 原子写** | Data-as-Flag | 原子块更小 → payload 效率下降；**完全没有原子写 → 退化到 DC 或显式 fence/ack** |
| **AIV 级并发** | kernel 分解与 overlap | GPU 上对应 warp / thread-block 级专精，含 NVIDIA 式 persistent kernel |
| **多层 fabric（带宽均匀、延迟随 hop 变）** | 层次化调度 | 更平坦的 fabric 上退化为纯 token 负载均衡，抗 straggler 变弱 |

**我补的方法学风险**：

- **只有推理，且只有 decode。** 指标是算子延迟与 TPOT，端到端跑 vLLM-Ascend 的 decoding step。**训练侧一次没测**；反向的 dgrad/wgrad 依赖链更长、张量更大，串行化只会更重（对我们是好消息，但论文没给证据）。
- **基线单一。** 主对比只有 CANN EP（同为华为栈）。DeepEP 在不同硬件上，不能横比。所以"比 SOTA 强"这个结论其实是"比自家上一代强"。
- **收益随通信量增大而衰减**（BS=128 从 46.4% 掉到 35.9%）。这是同步税模型的自然推论：传输时间变长，软件开销重新被掩盖 —— 意味着**在训练侧（batch 远大于 decode）收益可能显著小于 52.4%**。这点必须在引用时说清。
- SP 的哨兵安全性依赖"正常计算不会产出该位型"，靠初始化时校验保证；这是个**全局假设**，一旦某层数值范围变化就要重新校验。

## 6. Our take

### 6.1 可以直接拿走的四件事

1. **50% / 20% 那个数**（§5.5）——MoE 通信占 per-token 延迟一半，却只占硬件执行时间两成。这是外部、生产超节点上的证据，说明**瓶颈是依赖阻塞与运行时开销，不是带宽**。已回写进 [`../knowledge/systems/industry-training-optimization-2026.md`](../knowledge/systems/industry-training-optimization-2026.md) §1.4 读法。
2. **通信核的边际收益在带宽饱和点归零**（Insight 1：24→48 核几乎无提升）。给 MonolithEP 的角色分区一个先验：`COMM_DISPATCH` 只要够打满 XGMI 就行，多给就是浪费 CU。我们应该照着这个思路测一遍 MI355X 上"打满 XGMI 写带宽需要几个 WG"，把 224:16:16 从拍脑袋变成实测。
3. **Sentinel Polling 是最便宜的可移植技巧**：接收侧预填哨兵、发送侧发裸 payload、接收侧比对。**不需要 512B 原子性**（只需要单块写不被观察到半写），payload 效率 100%，还省掉一整套 flag 传输。值得在我们的 IPC dispatch 上做一版对照实验，替掉现在的 `atomicAdd_system` claim slot + ready flag。
4. **hop 距离要进调度判据**。CM384 的 218/929/2500 ns 在我们单节点 8×MI355X 上没有直接对应物（节点内 XGMI 是单层全连接），但两个地方会撞上：跨节点分层 a2a（ROCmoe Phase 4），以及**片内 XCD/L2 的 NUMA**（参见 [`./swizzled-head-first-attention.md`](./swizzled-head-first-attention.md)）。AMD Helios 这类 rack-scale 上来之后，Insight 3 会变成一等问题。

### 6.2 它对我们路线的批评，得正面回答

论文 §6 的原话把 UBEP 和 fused persistent kernel 对立起来，两条批评：

| 批评 | 对 MonolithEP 成立吗 | 我们的回应 |
|---|---|---|
| **rigid resource allocation**（刚性资源划分） | **成立**。`COMPUTE:COMM_DISPATCH:TAIL_COMBINE = 224:16:16` 是编译期定死的 | 这是 ROCmoe P4「静态调度 + 显式预算」的**主动选择**，理由是 AMD 的 stream/ACE 语义让动态调度不可控（见 `rocmoe_DESIGN.md` §1.2）。但"静态"不等于"一个比例走天下"——应该按 roofline 出配置，而不是一个常数 |
| **software-managed completion polling**（软件轮询完成） | **成立**。我们的 `dispatch_expert_ready` 就是 pull-based 轮询（cco Technique 3.6） | UBEP 的 Data-as-Flag 严格更优：**把 flag 消掉**，让数据本身携带完成语义。这是我们能抄的，且不需要放弃 megakernel |

**结论：这两条批评不是"megakernel 路线错了"，而是"megakernel 里的同步原语可以更好"。** 把 Data-as-Flag（尤其 SP 变体）搬进 super-kernel，两条路线是叠加而非互斥的 —— UBEP 自己也承认它的核心思想在 GPU 上"maps to warp- or thread-block-level specialization, including NVIDIA-style persistent kernels"。

### 6.3 待验证的 AMD 侧问题

1. **CDNA 的原子写粒度是多少？** Data-as-Flag 的 TFF 变体吃 512B 原子性。gfx942/gfx950 的 cacheline 是 128B，`__hip_atomic_store` 没有 512B 保证。**先测清楚"多大的远程写不会被观察到半写"**，决定我们能用 TFF 还是只能上 DC/SP。这是抄这篇之前的第一道门。
2. **收益在训练规模下还剩多少？** 他们的收益随通信量增大衰减；训练的 token 量远大于 decode。需要我们自己在 MonolithEP 的 dispatch 上量一遍"同步开销占比"，看 15% 的同步税在训练侧是多少。
3. **`SyncAll` 的 AMD 对应物开销**：我们的 `compute_barrier`（AGENT scope）与跨 rank flag（SYSTEM scope）各占多少？现在只有 fence 次数预算，没有占比。

### 6.4 相关笔记

- [`./hyperparallel-moe.md`](./hyperparallel-moe.md) —— 同一硬件（昇腾）、同一思路（编译期静态调度 + AIV 单边通信），但走的是**算子融合**而非通信库。两篇合起来是昇腾阵营的完整答卷。
- [`./uniep/README.md`](./uniep/README.md) —— fused megakernel 路线的代表，**被 UBEP 引用为 [53]**；两篇正好是同一问题的两条解法。
- [`./comet.md`](./comet.md) —— UBEP 引它作 "MoE 通信占 ~47%/50%" 的来源 [49]。
- [`./x-stage.md`](./x-stage.md) —— remote store 发出到远端可见之间的软件可见阶段；和 Data-as-Flag 的 happens-before 论证是同一层问题。
- [`./disagmoe/README.md`](./disagmoe/README.md) —— UBEP §6 讨论的 AFD / M2N 通信形态，DisagMoE 的 AF-Pipe 是同一模式。
- [`./perseus.md`](./perseus.md) —— 多节点 megakernel 的 `put-with-signal` 语义；UBEP 结尾呼吁的 Write-with-Notify / Remote-Sync-Write 正是这条线。
