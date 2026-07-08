# DisagMoE:注意力-FFN 解耦的 AF-Pipe 计算通信重叠 MoE 训练
# DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism

> **arXiv:** [2605.11005](https://arxiv.org/abs/2605.11005) (2026-05-10) · **机构:** ByteDance Seed + Univ. of Washington + Cornell
> **平台:** 16 节点 × 8 H800（NVLink 400GB/s + CX-7 8×400GbE）
> **领域:** MoE 训练 · attention-FFN 解耦 · pipeline overlap · roofline 资源分配
> **核心贡献:** 把 **attention 与 FFN 拆到不相交的 GPU 组**（A-Worker 走 DP、F-Worker 走 EP），用 **AF-Pipe** 多阶段流水把 all-to-all 变成 many-to-many（M2N/N2M）并当作**一等流水阶段**与两侧计算 overlap，再用 **compute-comm roofline + MILP** 自适应给两组分 GPU/NIC。up to **1.81× vs Megatron-1F1B**、**1.34× vs SOTA overlap（Tutel/Comet/DualPipe）**。

---

## 一、问题分析

### 1.1 背景与瓶颈
- Large-EP 下 expert 跨节点分片，dispatch/combine 两次 all-to-all 卡在 attention↔FFN 之间的关键路径。64×H800 profiling：通信占训练步 **up to 50%**；EP 跨节点越多，all-to-all 占比从单节点 22% 升到 8 节点 **78%**；top-k 越大占比越高。

### 1.2 现有 overlap 的天花板
- **op 级 chunk overlap**（Tutel/Comet）：overlap 窗口被 FFN 计算量上界卡住，且 FlashAttention 难完整切块 → 留下无法隐藏的通信尾巴。
- **micro-batch 级 pipeline**（Lancet/DualPipe）：仍面临**结构性失衡**——attention FLOPs 随 seq **二次**增长，FFN 与 all-to-all **线性**增长。
- **serving 侧 AFD**（MegaScale-Infer/StepFun）与 **HeterMoE**（训练 AFD）：每组要存该组件全部层参数 + 训练态 → 规模上 OOM（HeterMoE 仅到 ~4.3B）。且固定 compute-comm 比例，限制 overlap 设计空间。

### 1.3 关键洞察（roofline，§3.3）
Qwen3 8 节点：seq 4K→32K 时 attention 计算占比 28.4%→50.3%、其 compute/comm 比 1.08→2.78；FFN 占比降、比值几乎不变（0.73→0.77）。**attention 算术强度随 seq 涨得快、先进入 compute-bound；FFN 一直 comm-bound**。整体被 FFN 的通信拖住 → 对称地给全系统加带宽无法同时把两者推到 compute roof。

---

## 二、方法

### 2.1 解耦放置（§4.1）
按组件类型切 Transformer：group `g` of 组件 `c∈{A,F}` 拿层 `{g+kp}`（每 p 层取一层）。A-Worker 组内 DP 复制 dense，F-Worker 组内 EP 分片 expert。每组可持有同类型多层 → 在显存约束下最大化利用率、支持更大 MoE。

### 2.2 AF-Pipe（§4.2）
- **时分复用流水**：一个 batch 切 micro-batch，前向 hidden_states 从 A→F 传，反向梯度 F→A 传，两组持续活跃、几乎无 idle。
- **M2N 阶段边界**：传统 all-to-all → many-to-many（M2N/N2M），**当作一等流水阶段**与两侧计算对齐 overlap；把 P2P + combine 融进一个统一通信阶段，通信量约 **↓1/k**。warm-up bubble：AF-Pipe ≈ baseline 的 **1/4**（去掉 P2P 延迟 + 两个 all-to-all 融成一个 overlapped M2N）。
- **异步三流**：forward / backward / communication 三流协调，稳态交错计算与通信。

### 2.3 自适应资源分配（§4.3）
把 GPU/NIC 在 A/F 两组间的划分建成 **compute-comm roofline + MILP**：两组算力峰值随各自 GPU 数缩放、共享带宽。两阶段决策：① GPU split 最小化瓶颈阶段 latency `max(T_a,T_f)`（消 bubble）；② 在此基础上 NIC split 最大化 MFU（把带宽 slack 侧的 compute roof 抬起）。MILP 出 roofline seed，再 profile 局部搜索精调。

---

## 三、实验效果

**设置**：16 节点 × 8 H800；模型 DeepSeek-MoE / GPT-OSS-120B / Qwen3-235B；seq 4K–32K；EP=16。Baseline：Megatron-1F1B、Tutel、Comet、DualPipe。

| 维度 | 结果 |
|---|---|
| vs Megatron-1F1B | **1.59–1.81×** |
| vs Tutel / Comet | 1.2–1.5× |
| vs DualPipe | 1.05–1.13× |
| 非重叠通信削减 | vs Tutel −88% / vs Comet −75% / vs DualPipe −45% |
| 最优 A:F 资源比 | 随 seq 变：16K 时 **16:10**（1.56× vs baseline / 1.29× vs 均分 16:16）；4K 时均分最优 |
| top-k / EP 扫描 | 1.08–1.92× vs baseline |
| virtual stage | ≤16 提升吞吐（减 bubble），>16 OOM |

---

## 四、业界定位

| 路线 | 代表 | 缺陷 / DisagMoE 改进 |
|---|---|---|
| op 级 chunk overlap | Tutel / Comet | overlap 窗口被 FFN 上界卡；DisagMoE 做 module 级 overlap |
| micro-batch pipeline | Lancet / DualPipe | attention/FFN compute-comm 比失衡致残余尾巴 |
| serving AFD | MegaScale-Infer / StepFun | 训练不适用（每组存全层参数 OOM） |
| 训练 AFD | HeterMoE | 仅 ~4.3B，固定 comp-comm 比；DisagMoE 用 AF-Pipe + 自适应分配扩到数千亿 |

**独特贡献**：把 serving 的 AFD 思路真正带到大规模训练（AF-Pipe 解 OOM + module 级 overlap）；roofline+MILP 自适应分 GPU/NIC 对付 attention-FFN 结构性失衡。

---

## 五、局限与复现
- 假设 pretraining 固定形状（seq/micro-batch），动态形状（如 RL）需在线重分配。
- 单一 pipeline 深度 p 两组共用；非对称深度是 future work。
- 6K Python + 2K C++，基于 Megatron-LM（PyTorch 2.6），M2N 用 GPUDirect + GPUCopy。

## 六、对 monolith-moe / rocmoe 的启示（Our take）

DisagMoE 和我们是**互补的两条路**：我们在单节点把 attention+MoE 融进一个 super-kernel（fuse），它反过来把 attention 与 FFN **拆开**放不同 GPU 组（disaggregate）。但它的 roofline 洞察对我们极有用。

| DisagMoE | 我们（monolith-moe / rocmoe） | 关系 |
|---|---|---|
| **attention 二次 / FFN+comm 线性 → 结构性 compute-comm 失衡** | 我们只在 MoE 层内 overlap，不含 attention | ⭐ 解释了为什么 attention 侧不该硬塞进同一 overlap 预算 |
| roofline + MILP 自适应分 GPU/NIC | `comm_ratio` 分 CU（单一 knob） | ⭐ 可借鉴 roofline 判据决定 comm_ratio,而非纯 sweep |
| all-to-all → M2N 一等流水阶段，通信 ↓1/k | dispatch/combine 融进 super-kernel | 思路不同（拆 vs 融），但「通信当一等公民调度」一致 |
| 跨机架多节点、IB 带宽是主约束 | 单节点 XGMI（带宽充裕） | 他们的收益主要来自跨节点带宽稀缺,我们场景 overlap 收益天花板更低（见 MegaScale-MoE 的 R 判据） |

**两条最有用的结论：**
1. **attention/FFN 的 compute-comm 比是结构性失衡**——这佐证我们只做 MoE 层内 overlap 是对的（attention 是 compute-bound，不该占通信预算）；若未来要把 attention 也纳入，得像 DisagMoE 那样分开算 roofline。
2. **roofline+MILP 决定资源划分** 比我们纯 sweep `comm_ratio` 更有原则——可以先用 roofline 算出 comm/comp 该分多少 CU 的先验点，再局部 sweep 精调，省掉盲搜。
3. 它的收益（1.34–1.81×）主要来自**跨节点 IB 带宽稀缺**；我们单节点 XGMI 带宽充裕，这提醒 overlap 的绝对收益上限受带宽/算力比制约（与 MegaScale-MoE 的 `R≈3/2·h_ffn·bw/peak` 判据一致）。

> 相关：[`../megascale-moe.md`](../megascale-moe.md)（同为 ByteDance，overlap 的 R 判据）、[`../../notes/monolith-moe/README.md`](../../notes/monolith-moe/README.md)。

---

*据 arXiv:2605.11005 全文（2026-05-10）整理于 2026-07-07。HTML：[`disagmoe.html`](./disagmoe.html)。*
