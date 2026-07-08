# MegaScale-MoE:生产级大规模 MoE 训练的通信高效系统
# MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

> **arXiv:** [2505.11432](https://arxiv.org/abs/2505.11432) (v3) · **DOI:** [10.1145/3767295.3769325](https://doi.org/10.1145/3767295.3769325) · **PDF:** https://arxiv.org/pdf/2505.11432
> **发表:** EuroSys 2026（Edinburgh, Apr 27–30）· **机构:** ByteDance Seed + 北京大学
> **领域:** MoE 训练 · 通信-计算 overlap · 并行策略 · 通信压缩
> **核心贡献:** 把每个 MoE 层**约束在单节点内（NVLink）**，用 **SP(attention)+EP(FFN) 通信高效并行 + inter/intra-op comm-compute overlap + 通信压缩**，在 **1440 张 H800** 上训 **352B MoE** 达 **1.41M tokens/s，比 Megatron-LM 快 1.88×**。

> **重读修正（2026-07-07）：** 旧版笔记（2026-03-07，arXiv「待公开」时期）基于会议标题**大量脑补**，把本文写成「万卡 + 生产容错 + 拓扑感知路由 + 分层 EP + 42% MFU + <30s 故障恢复 + METIS 图分区伪代码」——**这些都不是本文的贡献**。据 arXiv:2505.11432 v3 全文：本文**不做容错、不做拓扑感知路由、不做分层 EP**；三大真实支柱是 **① 通信高效并行(SP+EP，MoE 层锁在节点内) ② inter/intra-op 通信-计算 overlap ③ 通信压缩(FP32→BF16 / FP8 A2A)**。本页据全文完全重写。

---

## 一、问题分析

### 1.1 研究背景

- ByteDance 常态化在**数千卡**上训数千亿参数 LLM，边际效率提升即可省大量算力与时间。
- MoE 靠稀疏激活让 FLOPs 亚线性增长，训练成本比同质量 dense 模型低一个量级 —— 但**系统视角下通信成了关键瓶颈**。
- **实测：在 Hopper 上训某内部模型，通信占前向 43.6%、占整训练 32%。** 两个成因：① MoE 参数更大 → 需更多 GPU 做模型并行 → 通信更多；② 稀疏计算需要在 fwd/bwd 各加 **2 次 all-to-all**（dispatch + combine）来分发/聚合 token。
- **硬件趋势加剧失衡**：算力增速远快于带宽（Figure 1），加上低精度训练进一步压缩计算时间 → 通信占比更突出；仅把 TP 扩到多节点，通信占比就常 >50%。

### 1.2 问题定义

**目标：** 在数千卡上高效训练数千亿～万亿参数 MoE，把通信开销压到接近零而不牺牲收敛。

**关键洞察（贯穿全文）：** MoE 与 dense 的架构差异是**层内（intra-layer）**的，也是通信开销的主要来源。因此 **MegaScale-MoE 把每个 MoE 层约束在单个节点内**，用高带宽 NVLink，避免既有系统常见的**跨节点 EP**。节点间用 pipeline parallelism 分参数、并 overlap 不同 micro-batch 的通信。

符号（Table 1）：$b$ micro-batch、$s$ seq、$h$ hidden、$n$ 模型并行度(TP/SP/EP)、$m$ query 头/kv 头之比、$k$ top-k。

### 1.3 解决方案

三条支柱：

**支柱一 · 通信高效并行（§3，降通信量）**
- **Attention 用 SP（DeepSpeed-Ulysses 式序列并行）替代 TP**。TP 需沿关键路径 all-gather / reduce-scatter 激活，通信量 $2bsh(n-1)/n$；SP 降到 $2bsh(n-1)/n\times(2+2/m)/n$，在 NVLink domain=8 时约为 TP 的 **1/4**。SP 复制而非切分 attention 权重，但靠**分层通信**（Figure 5 / Appendix A.1），SP 的参数同步开销与 TP 实测仅差 0.3–3.1%；额外显存仅 +1.2–5.4%（MoE 显存主要被 expert 参数占）。
  - 也评估过 CP（context parallelism），因 causal mask 负载不均、zigzag 也难完美均衡而放弃。
- **FFN 用 EP 替代 TP**。TP 切 expert 的 hidden 维伤 GEMM 效率；EP 每卡保留完整 expert 计算。EP 通信 $2k/n\times bsh(n-1)/n$ vs TP $2bsh(n-1)/n$，相对优劣看 $k/n$。**自适应通信模式**：当 top-k > n 时，用 **all-gather + 本地 scatter + reduce-scatter 替代 all-to-all**（ring 式、只与邻居通信，比 A2A 高效；Mixtral-8×7B 上 top-k>6 时 AG 式更快）。用 **自研 CUDA scatter/gather 算子**（预算 token→行映射）替代 `torch.scatter_add/gather`。
- 负载均衡：辅助 loss + token dropping，按「同 GPU 上的 expert 组」（类 DeepSeek-V2）算 balance loss 与容量。

**支柱二 · 通信-计算 overlap（§4，把通信藏到接近零）**
- 把每个 MoE 层的 fwd/bwd **拆成独立的计算/通信算子 GPU kernel**（不依赖 `torch.autograd` 的整块反向），获得细粒度调度自由。
- **Inter-operator overlap**：手工「holistic scheduling」在不同 CUDA stream 上异步跑，重排算子把通信藏进无依赖计算。配合 **selective activation rematerialization（SAR）**——前向只保留「重算贵」的激活，反向重算/重通信「内存贵」的部分，且把重算 overlap 掉。单 MoE 层激活从 $(2n+2k+3kf+12+5/m)bsh/n$ 降到 $(2kf+4+2/m)bsh/n$，**激活内存 ↓~50% 而不掉速**。
- **Intra-operator overlap**（关键路径上的通信，如 dispatch 后必须等 token 才能算）：把通信切成 tile、对齐 GPU 计算 pattern、**融进 compute kernel**；用**device memory barrier + tile 级通知**去掉 host 干预。两类 kernel：
  - Attention：`A2A+GEMM` / `GEMM+A2A`（用专用 copy engine 搬数、全部 SM 留给计算；tile 到达即通知 GEMM 续算；用 **swizzling** 对齐通信 tile 到达节奏与计算节奏；给通信分配**少量 SM**）。
  - FFN：`AG+scatter+GroupedGEMM` / `GroupedGEMM+gather+RS`。因 GroupedGEMM 需 token shuffle，先按 expert 再按 source rank 排序，**让每个计算 tile 只依赖少数（甚至单个）source rank**，减少等待、避免重复加载 expert 权重。

**支柱三 · 通信压缩（§5）**
- **DP**：BF16 混合精度训练里，梯度同步从 FP32 降到 BF16 —— 但用 **all-to-all 收集分片 + FP32 本地归约**（而非 ring reduce-scatter），避免 BF16 累加的精度损失；配 in-place buffer 防峰值显存增长。梯度通信 **↓50%**，精度损失可忽略。
- **FP8 训练**：把 TP 的 BF16 reduce-scatter 换成 **FP8 all-to-all**（E4M3）+ FP32 归约；前向 per-token 量化、反向 per-channel + token 维 group（group size 128）量化，保持与 BF16 loss 对齐。

---

## 二、实验效果

### 2.1 实验设置

| 项 | 详情 |
|---|---|
| 硬件 | NVIDIA **H800**（989 TFLOPS / 80GB / 3.4TB/s / NVLink 400GB/s）；另测 A100、H20 |
| Baseline | **Megatron-LM**（commit f1f03922），双方都开 MegaScale 的 DP/PP overlap，PP=15，公平同 global batch |
| 主模型 | **Internal-352B**（60 层，$h$=4096，32 头，$m$=4，$h_{ffn}$=14336，**32 experts，top-3**），seq=8192，vocab=65536 |
| 其他模型 | Mixtral-8×7B / 8×22B、Hunyuan-Large、Phi-3.5-MoE、DeepSeekMoE（Table 2） |
| 配置 | MegaScale-MoE 用节点内 SP+EP；Megatron-LM 用节点内 TP |

### 2.2 主要结果

**强扩展（352B，固定 global batch=720，Table 3）：**

| #GPUs | Megatron-LM 吞吐 (tok/s) | MegaScale-MoE 吞吐 (tok/s) | 加速 |
|---|---|---|---|
| 240 | 151.1k | 272.9k | **1.81×** |
| 480 | 301.1k | 498.6k | 1.65× |
| 720 | 430.5k | 740.1k | 1.72× |
| 960 | 550.2k | 963.8k | 1.77× |
| **1440** | 746.6k | **1407.7k** | **1.88×** |

- 训 1T tokens 的时间：1440 卡从 Megatron 的 15.50 天降到 **8.22 天**。
- MegaScale-MoE 的 MFU 随卡数从 32.48% 降到 27.89%（固定 batch → micro-batch 变少 → pipeline bubble 增多，属预期）。

**弱扩展（batch 随卡数 480→1440 从 360→1080）：** 加速 **1.74–1.79×**；MegaScale-MoE 近线性（吞吐仅降 0.2%），Megatron 降 2.74%。

**跨 GPU（Mixtral-8×7B，32 卡 H800/H20/A100）：** MFU 最高 **1.58×** 优于 Megatron；attention+GroupedGEMM 仅占一层约 1/3 时间，其余是通信与其他算子。

### 2.3 消融

**系统性拆解（352B，240 卡，batch 720，Table 5）：**

| 配置 | 归一化吞吐 | 增量 |
|---|---|---|
| baseline（TP for attn+FFN，无 overlap） | 1.00 | — |
| + SP+EP 通信高效并行 | 1.13 | **+13%** |
| + inter-operator overlap | 1.22 | **+9%** |
| + intra-operator overlap | 1.28 | **+6%** |

**逐组件：**
- **并行策略**：SP+EP 比 TP+TP 高 **14.9–32.9% MFU**（跨 7 个模型）；SP 额外显存仅 1.2–5.4%；SP vs TP 参数同步时间仅差 0.3–3.1%。
- **intra-op overlap**：通信+计算合计时间降 **1.2–4.7×**；训练迭代时间降 **7.1–12.9%**。
- **SAR**：Mixtral-8×7B/8×22B 激活内存分别 ↓45.5%/57.2%，整体显存 ↓21.3%/35%，性能差异 <0.5%。
- **DP 通信压缩**：BF16 A2A + FP32 归约的 loss 曲线与 FP32 reduce-scatter 几乎重合。
- **收敛**：35B 从头训 + 176B 续训，BF16 与 FP8 loss 均稳定一致。

---

## 三、业界类似方案

### 3.1 方案对比

| 方案 | 核心思路 | 与本文关系 |
|---|---|---|
| **Megatron-LM** | 3D 并行（TP/PP/DP），节点内 TP | baseline；本文构建其上 |
| **DeepSpeed-MoE** | 分层 all-to-all + 压缩，训练+推理 | 早期 MoE 系统，跨节点 EP |
| **Tutel** | 运行时自适应并行 + 分层 A2A | 动态切换在数千亿参数下 overhead 大 |
| **DeepSeek-V3（DeepEP+DualPipe）** | DeepEP 跨节点 A2A（限 4 节点）+ DualPipe overlap | 最强对照，见 §3.3 |
| **COMET**（[`./comet.md`](./comet.md)） | 单融合 kernel 内 tile 级 overlap（同团队 intra-op 手法同源） | 本文 intra-op overlap 引用其 swizzling/tile barrier |
| **MegaScale-MoE**（本文） | MoE 层锁节点内 + SP+EP + inter/intra-op overlap + 压缩 | 生产级 1.88× |

### 3.2 技术路线对比

- **路线 A：跨节点 EP + 限制路由**（DeepSeek-V3 DeepEP）——DeepEP 因跨节点 IB 带宽低，把 token dispatch 限最多 4 节点以保持跨节点通信量恒定，**牺牲路由灵活性**。
- **路线 B：MoE 层锁节点内 + 分层 PP**（本文）——每层在 NVLink 域内，**可路由到任意 top-k expert**，无跨节点 token dispatch。
- **overlap 手法**：DualPipe 用跨 micro-batch 的 PP overlap，需存 **2× 参数**；MegaScale-MoE 的 overlap 发生在**单个 micro-batch 的 fwd/bwd 内**，**无额外显存**，且不依赖 PP。

### 3.3 本文定位

- **相对 Megatron-LM**：SP 替 TP（attention 通信 ↓~4×）、EP 替 TP（不伤 GEMM 效率）、加 inter/intra-op overlap + 压缩 → 1.88×。
- **相对 DeepSeek-V3**：不限路由（intra-node 任意 top-k）、overlap 无 2× 参数代价。
- **独特贡献**：把「MoE 差异是层内」这一洞察落成「MoE 层锁节点内」的系统设计；给出 **scale-up 判据 $R\approx\frac{3}{2}h_{ffn}\times\frac{bandwidth}{peak}$**（>1 即可 overlap，且与 expert 数/top-k/hidden/并行度/输入无关，只由 expert 中间维、算力峰值、带宽决定）。

### 3.4 推荐进一步阅读

| 论文 | 理由 |
|---|---|
| COMET（MLSys'25，[`./comet.md`](./comet.md)） | 同团队 intra-op 融合 overlap 的单 kernel 版，方法同源 |
| MegaScale（NSDI'24） | 同团队 dense LLM 万卡训练前作，DP/PP overlap 基础 |
| DeepSeek-V3 技术报告 | DeepEP + DualPipe 的对照路线 |
| FLUX / TileLink（同团队） | intra-op 融合 kernel 的底层原语 |

---

## 四、关键章节精译（摘要 + scale-up 判据）

**摘要。** 我们提出 MegaScale-MoE，一个为大规模 MoE 模型高效训练量身打造的生产系统。MoE 是把 LLM 扩到空前规模、提升模型性能的有力架构，但既有 MoE 训练系统随模型规模上升与硬件演进而效率退化。认识到高效通信在 MoE 训练中的关键作用，MegaScale-MoE 为每个 MoE 层的 attention 与 FFN 定制通信高效并行策略，并在 **inter- 与 intra-operator 两个层面**整体性地 overlap 通信与计算；此外用**调整通信模式后的低精度通信压缩**进一步提效。在 1440 张 NVIDIA Hopper GPU 上训练 352B MoE 模型时，吞吐达 **1.41M tokens/s，比 Megatron-LM 提效 1.88×**。

**§7 scale-up 判据。** 对含 MoE 的 SwiGLU 结构，计算/通信时间比
$$R=\frac{\text{comp\_time}}{\text{comm\_time}}\approx\frac{3}{2}\times h_{ffn}\times\frac{bandwidth}{peak}.$$
要 overlap 有效须 $R>1$。两点洞察：① $R$ 与 expert 数、top-k、hidden、并行度、输入大小**都无关**，算法参数选择灵活；② $R$ 只由 **expert 中间维、算力峰值、通信带宽**决定 —— 固定硬件下只要 expert 维足够大，就能在保持训练效率的前提下扩大 MoE。

**§7 经验。** 已部署于生产，承担公司大部分大规模 MoE 训练；支持万亿参数、单任务 >10000 GPU、跑数月（Figure 20：200B 总参/20B 激活的真实生产任务，>1万卡、多万亿 token、loss 稳定收敛）。FP8：SwiGLU 扩大数值范围 → 用 per-token 量化、并把 gating 权重乘法挪到 FC2 输出之后减小量化误差；多精度 optimizer 直接以 FP8 存参数、FP32 存 main 参数，省显存并把 DP 的参数 all-gather 通信减半。

---

## 五、局限与复现清单

**局限：**
- **依赖高带宽节点内 NVLink**：核心前提是「MoE 层锁节点内」；一旦 expert 并行必须跨出 NVLink 域到 RDMA 级带宽，$R>1$ 能否维持是公开问题（§7 自述）。
- **holistic scheduling 靠手工**：算子执行序、comm/comp 并发度、给通信分几个 SM 都是人工调；自动化留作 future work。
- **MoE 计算算子仍是 straggler 源**：GroupedGEMM 的 `cuFuncSetAttribute` 资源控制引入同步延迟；动态形状张量致显存碎片；gating 的众多小算子受 CPU jitter 影响造成 pipeline bubble。
- 代码未开源（生产系统）。

**复现清单：**
- [ ] 代码开源：**否**（ByteDance 生产系统）
- [ ] 模型配置：Table 2 给全（可在开源 Mixtral 等上复现思路）
- [ ] 硬件：H800/H20/A100 均测；核心收益依赖 NVLink 域
- [ ] 关键实现：SP+EP 并行、inter/intra-op overlap kernel（A2A+GEMM 等）、device-memory barrier、通信压缩

---

## 六、对 monolith-moe / rocmoe 的启示（Our take）

MegaScale-MoE 是我们 super-kernel 在**「host/collective/op 级 overlap」维度上的最强生产级对照**。它证明 comm-compute overlap 在生产里能稳拿 1.88×，也给了我们几把可直接搬的尺子。

| MegaScale-MoE 的招 | 我们的对应（monolith-moe / rocmoe） | 关系 |
|---|---|---|
| **intra-op：A2A+GEMM / GroupedGEMM+gather+RS 融进 compute kernel** | 我们把整个前向 5 phase 融成一个 persistent super-kernel | 同类思想；他们半层一 kernel + device barrier，我们整层一 kernel + scoreboard |
| **device-memory barrier + tile 级通知，去 host 干预** | 64-bit `block_ready` 位图 + receiver-pull | 收敛结论一致：tile/block 级就绪信号、无 host 介入 |
| **给通信分少量 SM、swizzling 对齐到达节奏** | `comm_ratio`（分 CU 给 scatter）+ XOR swizzle | 直接对应；他们的「comm SM 数调到 comm≈comp latency」= 我们 comm_ratio 甜点 |
| **按 source rank 排序让 tile 只依赖少数 rank** | receiver-pull：block 就绪即算 | 同源，都把「等全部 token」降成「等这个 tile 的 token」 |
| **SP 替 TP（attention 通信 ↓4×）** | 我们目前不动 attention（EP8 MoE 层） | 正交；若做 attention 可借 |
| **通信压缩 FP32→BF16 / FP8 A2A** | backlog：FP8/mxfp8 weights | 同类正交杠杆，砍通信/HBM 流量 |
| **SAR：只存重算贵的激活，overlap 掉重算** | Phase 2.1 SwiGLU pre-compute + decomposed backward | 思想一致 |

**三条最有用的结论：**
1. **scale-up 判据 $R\approx\frac{3}{2}h_{ffn}\times\frac{bandwidth}{peak}$ 可直接套 MI355X**：拿 XGMI 带宽 + MI355X BF16/FP8 peak + DSV3 $h_{ffn}$ 代入，能先验判断我们 super-kernel 在给定 shape 下 overlap 到底有没有净收益上限——这正是我们「512 t/g 赢、2048/8192 t/g 输」现象的一个理论解释器，值得算一遍写进 rocmoe README。
2. **「MoE 层锁节点内」是他们的前提，也正是我们的现状**：我们本就在单节点 8×MI355X / XGMI 上做，等于站在他们认为最优的通信域里；差异化在于我们把 op 级 overlap 再推进到 **in-kernel chunk 级**（他们 intra-op 已经很接近，但仍是「半层一 kernel + copy engine」，不是整前向一个 persistent kernel）。
3. **他们的 1.88× 是 op/collective 级 overlap 的生产上限参照**：我们要证明 in-kernel 融合能在 AMD/XGMI 上超过这个「op 级」天花板，否则不如直接移植 MegaScale-MoE 式方案。这条要写进项目的价值主张。

> 相关笔记：[`../notes/monolith-moe/README.md`](../notes/monolith-moe/README.md)（super-kernel 调优时间线、comm_ratio / launch_bounds CU 隔离 / XOR swizzle）、[`./comet.md`](./comet.md)（同团队 intra-op 融合 kernel）。

---

*据 arXiv:2505.11432 v3 全文完全重写于 2026-07-07（原脑补版笔记 2026-03-07 已作废）。HTML 版：[`megascale-moe.html`](./megascale-moe.html)。*
