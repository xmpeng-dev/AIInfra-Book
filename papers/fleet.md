# Fleet: 面向多芯粒 GPU 的分层任务抽象与 Megakernel

> **来源:** [arXiv:2604.15379](https://arxiv.org/abs/2604.15379) · PDF: https://arxiv.org/pdf/2604.15379  
> **机构:** AMD Research  
> **场景:** AMD Instinct MI350X 上单卡 dense LLM decode（Qwen3-8B bf16）  
> **关联:** 基于 Mirage Persistent Kernel (MPK)；与 HipKittens、Hazy megakernel、FlashFormer 同赛道

---

## Bibliographic metadata

- **Authors:** Sangeeta Chowdhary, Ryan Swann, Sean Siddens, Muhammad Osama, Stephen Neuendorffer, Alexandru Dutu, Karthik Sangaiah, Sandeepa Bhuyan, Samuel Bayliss, Ganesh Dasika（AMD Research）
- **Venue / year:** arXiv preprint（2604.15379，2026）
- **Identifiers:** https://arxiv.org/abs/2604.15379
- **Code / data:** 文中未给出公开仓库链接；实现基于 Mirage MPK 扩展

---

## One-paragraph thesis（一段话摘要）

现代 GPU 普遍采用多芯粒（chiplet）封装，每个 XCD 拥有私有 L2，但 CUDA/HIP 仍按「单体 GPU + 统一 L2」的扁平层次暴露线程→wavefront→workgroup→grid，无法在编程模型中表达「绑定到某一 chiplet / 其 L2」的工作与数据亲和性。Fleet 提出四层任务模型（wavefront / CU / **Chiplet** / device），其中 **Chiplet-task** 把同一 XCD 上所有 worker 绑在一起，通过 per-chiplet 软件调度与协作式 weight tiling，在 persistent megakernel 内保留跨算子的 L2 状态。作者在 MI350X 上对 Qwen3-8B decode 评测：相对 vLLM（eager）小 batch 延迟约 **1.3–1.5×** 更低；在 bs=32/64 时 M-major 协作遍历将 L2 命中率从约 39% 提到 **51% / 61%**，HBM 读流量最多降 **37%**，相对 chiplet 无感 Mirage MPK 基线约 **1.27–1.30×**。

---

## Problem & motivation（问题与动机）

### Task / setting

- **任务:** 自回归 **decode**（非 prefill）；单卡、dense Qwen3-8B（hidden 4096，36 层，FFN 12288，GQA 32Q/8KV），bf16。
- **硬件:** AMD Instinct **MI350X**（8 XCD × 32 CU，每 XCD 4 MB 私有 L2，256 MB Infinity Cache/MALL，288 GB HBM3，>5 TB/s HBM 带宽）。论文未在 NVIDIA Blackwell 上评测。

### Gap vs prior work

| 维度 | 现状 | Fleet 要补的洞 |
|------|------|----------------|
| 编程模型 | L2 在 chiplet GPU 上物理分区，HIP 仍当 device-scope 统一缓存 | 缺少与 **chiplet / 分区 L2** 对齐的抽象 |
| 服务框架 | vLLM / SGLang：算子级 kernel + hipBLASLt；graph 按 batch size 特化 | 跨 kernel L2 被 barrier 刷掉；~250 次 launch/token/36 层 |
| Persistent megakernel | Mirage MPK、Hazy megakernel、FlashFormer 等多面向 **单体 NVIDIA L2** | 未系统处理 **跨 XCD 无 L2 共享** 导致的重复 HBM 流量 |
| 单 kernel 优化 | HipKittens、CUTLASS persistent GEMM、block swizzling | 优化 **单次 launch**；不覆盖整层/整模型任务图与 chiplet 级事件同步 |

### Assumptions / scope

- 单层权重 working set **368 MB**（bf16），远超单 XCD 4 MB L2；优化重点是 **权重 HBM 流量**，非 activation（bs=128 时 activation ~1 MB）。
- Linear（GEMM）占 decode 时间约 **95%**；attention ~5%。
- 默认 load 为 **wave scope**，权重在各 XCD L2 **独立缓存、无跨 XCD 一致性**；跨 XCD 可见需 `buffer_wbl2` + invalidate。
- 评测 **未覆盖** 多卡 TP、prefill、MoE、Blackwell；bs≥64 时 Fleet 仍慢于 vLLM（缺 K-split 等）。

---

## Main contributions（主要贡献）

1. **Chiplet-task 抽象：** 将「一个 XCD 上全部 worker（MI350 上 31/32 CU）+ 显式 L2 预算」作为一等任务粒度，与 wavefront-task / CU-task / device-task 组成四层层次，映射到寄存器/LDS/分区 L2/HBM。
2. **Chiplet 感知的协作式 GEMM tiling：** N-split 把输出列分到 8 个 XCD；**M-major 窗口遍历** 让同 XCD 多 worker 按相同顺序读同一 weight 列，把 L2 miss 转为 hit；配合 **cache-streaming**（sc1=1, nt=1）权重 load 与 activation NT store 的三档 cache 策略。
3. **分层事件同步：** XCD 内用 L2-local 计数，仅每 XCD 最后一个 worker 做一次 `buffer_wbl2` + GPU-scope 全局事件，相对 per-worker 全局 fence 大幅减少跨芯粒一致性流量（线性层每事件最多 8 次 fence）。
4. **基于 Mirage MPK 的 persistent runtime：** 每 XCD 一个 scheduler workgroup + 其余 worker；Chiplet-task **广播** 到同 XCD 所有 worker，CU-task **round-robin** 到各 CU。
5. **端到端证据：** 在 MI350X + Qwen3-8B 上相对 vLLM 与 chiplet-unaware Mirage MPK 给出可分解的延迟、L2 命中率、HBM 流量与 roofline 分析。

---

## Method（方法）

### Core idea

把 transformer decode 的一层编译成 **device 级任务图**（事件依赖），在 **单个 persistent HIP kernel** 内由每 XCD 的 scheduler 派发。GEMM 等内存主导算子落在 **Chiplet-task**：8 个 Chiplet-task 各算 `[M, N/8]` 子矩阵，同 XCD 内 31 个 worker 用 **strided tile index** 协作，共享该 XCD 的 4 MB L2。

```
HIP 扁平模型          Fleet 四层（与硬件边界对齐）
─────────────────────────────────────────────
Wavefront      ↔    Wavefront-task  (SiLU, residual, RoPE)
Workgroup/CU   ↔    CU-task         (RMSNorm, 单头 attention)
  (缺失)       ↔    Chiplet-task    (GEMM 分区 + L2 协作)  ← 核心新增
Grid/Device    ↔    Device-task     (8 Chiplet-task 拼完整算子)
```

### 关键机制

**1) 标准调度 vs Fleet 调度（Figure 2）**

- 标准 block 调度：同 XCD 上不同 worker 可能做 **不同 GEMM 列** → L2 互相挤占、权重重复从 HBM/LLC 拉取。
- Fleet：列方向 **N-split 到 XCD**；同 XCD 上 worker **协同读同一 weight 分区**。

**2) M-tile vs M-split（消融）**

| 模式 | 行为 | 作用 |
|------|------|------|
| **M-tile** | 所有 8 XCD 处理相同 `m_tiles=⌈B/T_M⌉`；同 XCD 内先扫 M-tile 再进下一 N 列 | 启用 **协作 weight reuse**；预期 L2 hit ≈ `(R-1)/R`, `R=min(W, m_tiles)` |
| **M-split** | 每 XCD 分配不相交 M-tile；无跨 worker weight 共享 | 仅隔离 **Chiplet-task 调度开销** 的收益 |

**3) 遍历顺序（Figure 3）**

- **N-major：** 连续 tile 沿 N 前进 → 并发 worker 读 **不同 weight 列** → L2 压力大。
- **M-major（采用）：** 沿 activation 行 / 共享 weight 列推进 → 时间局部性，首 worker 拉 HBM，后续 hit L2。

**4) 算子融合**

- 将 SiLU 融进 gate+up 的 Chiplet-task，去掉中间 buffer 的 L2 写回；bs=1 时 L2 hit 从 ~9.4% 提到 ~17.4%（与融到 CU-task 类似，说明收益主要来自 **少一次 L2 往返**）。

**5) N-split vs K-split**

- **N-split + 协作 L2：** 小 batch（bs 1–16）占优（调度 + persistent 为主）。
- **K-split（Stream-K 类）：** 大 batch（bs≥32）算力 bound 时 hipBLASLt 更优；Fleet 尚未在 megakernel 内集成 K-split，故 bs=64 输给 vLLM。

**6) 分层同步（Figure 5）**

1. 任务描述符队列：launch 前写好，只读，无锁。  
2. Scheduler→worker：同 XCD，device-scope 写本地 L2 队列。  
3. Worker↔worker（Chiplet-task）：XCD-local 原子计数，无 fence。  
4. XCD→全局：仅 **每 XCD 最后一个 worker** `buffer_wbl2` + `flat_atomic_add` 更新全局事件；scheduler 轮询 `G[e]` 再派发下游。

**7) 任务图规模（bs=1，单层）**

- 标准 Mirage 分解：~**1407** tasks（每 GEMM 96–256 CU-tasks）。  
- Fleet：**543** tasks（每 GEMM **8** Chiplet-tasks）；SiLU 已融入 gate+up。

### Complexity / compute

- Scheduler 占用 **8/256 CU（3.1%）**；声称轻量控制下 worker 很少等任务。  
- Persistent kernel **所有 task 类型单函数编译** → 寄存器压力高，**每 SIMD 仅 1 wave**，无 wave 切换隐藏延迟 → L2 miss 直接 stall MFMA。  
- 论文给出 L2 聚合带宽 ~**100 TB/s**（8 XCD）vs HBM **5.3 TB/s**。

---

## Experiments & evidence（实验与证据）

### Setup

- **硬件:** 单卡 MI350X。  
- **Baselines:** (1) **Mirage MPK** 移植 MI350，2D tiling、无 chiplet 协作；(2) **vLLM 0.17.2** on ROCm，`--enforce-eager`（无 graph），hipBLASLt GEMM。  
- **指标:** Decode-only **TPOT**（ms/token）；64 prompt tokens + 1024 output tokens；GPU 内 `s_memrealtime` 计时。  
- **工具:** HIP timing、cycle counter、rocprofiler-sdk（L2/HBM）。

### 端到端 TPOT（Figure 6，Qwen3-8B）

| Batch | vLLM | Mirage MPK | Fleet M-tile | Fleet M-split | 备注 |
|-------|------|------------|--------------|---------------|------|
| 1 | 10.51 ms | 7.83 ms | **6.82** | **6.73** | Mirage 已 1.34× vLLM；Fleet ~1.54× vLLM |
| 16 | — | — | ~1.13–1.16× Mirage | 同左 | 无协作 reuse，`m_tiles=1` |
| 32 | ~11–12（平台期） | 15.62 ms | **12.35** (51.0% L2) | 13.37 (39.5%) | M-tile 1.27× Mirage；仍 1.06× vLLM |
| 64 | ~11–12 | 24.10 ms | **18.61** (61.4% L2) | 23.40 | M-tile 1.30× Mirage；**慢于** vLLM |

**分解结论（作者）：**

- **bs 1–16：** M-tile ≈ M-split；收益 ≈ **Chiplet-task 调度**（8 次派发 vs 96–256 CU 派发），L2/HBM 几乎不变。  
- **bs 32+：** M-tile 的 **协作 weight tiling** 主导；bs=64 HBM 读 6203 GB → 3925 GB（**-37%**）。

### L2 / HBM（Table 4，1024 decode tokens）

- bs=32：Mirage L2 **38.9%** → Fleet M-tile **51.0%**；HBM Rd **0.82×** Mirage。  
- bs=64：Mirage **39.0%** → M-tile **61.4%**；HBM Rd **0.63×**。  
- 解析模型：`L2_hit_weight = 1 - 1/min(W, m_tiles)`；bs=1 仍有 ~17% hit 来自 **fused SiLU** 等。

### Roofline（Figure 7）

- 名义 AI = batch size B；有效 `AI_eff = B / (1 - L2_hit)`。  
- bs=32：51% hit → AI 32→65，向 ridge（~245）靠近。

### Per-GEMM 权重（Table 5）

- gate_up 每 XCD 分区 **24 MB** >> 4 MB L2，但 **活跃 K-chunk tile** ~1 MB（31×32 KB）可驻留 L2 → M-major 仍高 hit。

---

## Limitations & risks（局限与风险）

### Stated limitations（论文自述）

- 仅 **Qwen3-8B dense**、**单卡 MI350**；无 multi-GPU TP、prefill。  
- bs≥32 仍缺 **K-split / attention 优化**，大 batch 输给 vLLM。  
- Chiplet-task 图目前靠 **手工 Mirage 输入**，无编译器自动 super-optimization（需 chiplet 代价模型）。  
- Register 压力限制 occupancy；megakernel 固有权衡。

### Unstated / methodological risks

- vLLM 使用 **enforce-eager**，可能低估工业界 graph+capture 路径；与 Mirage/Fleet 的对比偏「公平 megakernel vs 最保守 vLLM」。  
- 未开源代码，复现依赖 Mirage 生态与 AMD 内部移植细节。  
- Blackwell「全局 coherent L2」与 AMD「非 coherent 分区 L2」行为不同，Chiplet-task 的可移植性需单独验证。  
- Scheduler 3.1% CU + 极短 task 时调度开销可能被低估。

---

## Positioning vs related work（相关工作定位）

| 方向 | 代表工作 | Fleet 差异 |
|------|----------|------------|
| Whole-model megakernel | Hazy megakernel, FlashFormer | 它们面向 **统一 L2**；Fleet 针对 **分区 L2 chiplet** |
| Compiler + MPK | Mirage MPK (Cheng et al.) | Fleet **扩展** MPK：Chiplet-task、协作 tiling、两级事件 |
| AMD kernel 库 | HipKittens | 单 kernel XCD grouping；Fleet 是 **跨算子 persistent 任务图** |
| NVIDIA | Thread block clusters | 粒度小于 chiplet/L2；Fleet 用 **纯软件** 协调整 XCD L2 |
| Serving | vLLM, SGLang | 算子级 + 库 GEMM；难表达 **应用级 L2 局部性** + 零 launch 融合 |

**Suggested reading next**

- Mirage Persistent Kernel — `2512.22219`  
- HipKittens — `2511.08083`  
- Hazy «Look ma, no bubbles» megakernel  
- FlashFormer — `2505.22758`  
- CDNA4 scope / `buffer_wbl2`（AMD ISA guide）

---

## Our take（与我们工作的关联）

1. **与 MI355 / MonolithEP / RocMoE 的直接共鸣：** 论文把 chiplet GPU 讲成 **机内 NUMA**——这正是我们在 XGMI + 分区 L2 上做 dispatch/combine、super-kernel 时要显式管理的层次。Chiplet-task ≈ 「绑定到一个 XCD + 其 4 MB L2 的协作单元」，与 **按 XCD 分 tile、L2 驻留权重、减少 HBM** 的优化方向一致。  
2. **与 megakernel / Mirage 路线：** Fleet 证明 **软件 scheduler 替代硬件 block dispatcher** 在 decode 上可拿回 1.3–1.5×；代价是寄存器并集、单 wave occupancy。我们若做 fused MoE 或整层 persistent kernel，应把 **Chiplet-task + 分层 fence** 当作一等设计项，而不是事后 swizzle。  
3. **可借鉴的具体技术：** M-major 窗口遍历、权重 streaming load + activation NT、SiLU 融入 GEMM Chiplet-task、**M-split 消融** 方法论、rocprofiler L2/HBM 表。  
4. **已知缺口（论文也承认）：** 大 batch 需要 **K-split / CK auto-tune**；多卡 TP 与 Fleet 的 N-split **正交可组合** 但未测。我们 bench turbo MoE / EP 时应分开报 **调度收益 vs L2 协作收益**。  
5. **行动项：** 在 `knowledge/hardware/` 或 MI355 skill 中交叉引用分区 L2 + scope bits；若 Mirage/Primus 有 MPK 路径，评估能否表达 Chiplet-task 或手写等价调度。

---

## Glossary（术语表）

| 术语 | 含义 |
|------|------|
| XCD | AMD Accelerator Complex Die，MI350 上 8 个，各 32 CU + 4 MB L2 |
| Chiplet-task | 绑定单一 XCD、协调分区 L2 上数据与 worker 的任务 |
| M-tile | 沿 batch 维划分的 GEMM tile 组，`m_tiles=⌈B/T_M⌉` |
| MALL / Infinity Cache | MI350 上 256 MB 最后一级 on-package cache（L2 victim） |
| MPK | Mirage Persistent Kernel |
| TPOT | Time Per Output Token，decode 每 token 延迟 |
| scope (SC1, SC0) | CDNA 内存指令一致性范围：wave / workgroup / device / system |

---

## Open questions & reproducibility checklist

**Questions**

- Mirage + Fleet 何时开源？与 ROCm/vLLM 集成的产品路径？  
- MoE / FP8 / TP>1 下 Chiplet-task 图是否仍 8× 少于 CU-task？  
- 与 **Primus / AITER** 内置 GEMM 的 K-split 能否在同一 megakernel 内切换策略？  
- Blackwell 上 Chiplet-task 是否映射到 GPC 集群还是整 die L2？

**Repro checklist**

| 项 | 状态 |
|----|------|
| 论文 PDF / arXiv | 有 |
| 代码 | 未声明公开 |
| 模型权重 | Qwen3-8B（公开模型） |
| 硬件 | MI350X（单卡） |
| 超参 / tile | 16×64×256 CK GEMM tiles；cache-streaming 权重 |
| 随机种子 | N/A（推理 benchmark） |
| 计时方法 | `s_memrealtime` in-kernel |
