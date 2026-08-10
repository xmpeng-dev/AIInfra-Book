# 训练/推理系统优化 arXiv 全景（2025–2026）

> **定位：** 从 arXiv 近期工作中筛选**训练工程优化**相关论文，含可迁移到训练的推理系统工作。  
> **更新时间：** 2026-07-31  
> **筛选标准：** 大厂/顶尖名校优先；有量化数据；与 comm-compute overlap / MoE / 优化器 / kernel fusion 相关。  
> **详细笔记：** 见 [`../../papers/README.md`](../../papers/README.md)；单篇深读用 `paper-deep-analysis` skill。
> **最新增量：** 2026-05~08 的扫描结果单独成文，见 [`arxiv-digest-2026-08.md`](./arxiv-digest-2026-08.md)（27 篇新论文 + 5 篇精读）。

---

## 0. 快速导航

| 你的瓶颈 | 优先看 |
|----------|--------|
| MoE all-to-all 隐藏 | UniEP · Comet · DisagMoE · FlashOverlap · TileLink |
| Muon/Shampoo 分布式 | [DMuon](../../papers/dmuon.md) · Canzona · MatrixFSDP |
| 变长序列 + 万卡 | ByteScale · ByteRobust |
| 推理思路回灌训练 | MegaScale-Infer→DisagMoE · Event Tensor · CODA |
| NCCL 争用 / overlap 调参 | Lagom · Resource-aware Overlap |
| FP8 MoE | DeepSeek-V3 报告 · FP8-Flow-MoE |
| Transformer 非 GEMM 融合 | CODA |

**已有深读笔记（📒）** 链到 `papers/`；**待读（⭐）** 建议下一步细读。

---

## 1. MoE 通信 / overlap（最活跃）

| 论文 | arXiv | 机构 | 核心 | 关键数据 | 笔记 |
|------|-------|------|------|----------|------|
| MegaScale-MoE | [2505.11432](https://arxiv.org/abs/2505.11432) | ByteDance + 北大 | 节点内 SP+EP；inter/intra-op tile fuse；通信压缩 | 1440×H800 352B **1.88×** vs Megatron | 📒 [megascale-moe.md](../../papers/megascale-moe.md) |
| DisagMoE | [2605.11005](https://arxiv.org/abs/2605.11005) | **ByteDance Seed** + UW + Cornell | Attn/FFN **GPU 组解耦** + AF-Pipe M2N 流水 | **1.8×** @ 128×H800 | 📒 [disagmoe/](../../papers/disagmoe/README.md) |
| UniEP | [2604.19241](https://arxiv.org/abs/2604.19241) | ByteDance Seed | Dispatch+GEMM / GEMM+Combine **单 MegaKernel** | vs COMET **1.03–1.38×** | 📒 [uniep/](../../papers/uniep/README.md) |
| Comet | [2502.19811](https://arxiv.org/abs/2502.19811) | ByteDance Seed | shared-tensor 分解 + thread-block 专用化单 kernel | 单层 **1.96×** / E2E **1.71×** | 📒 [comet.md](../../papers/comet.md) |
| UltraEP | [2606.04101](https://arxiv.org/abs/2606.04101) | 北大主导 + 小红书 | Rack-scale exact-load 均衡 + RSN 通信 | 训练 **1.42×**；达理想均衡 **94.3%** | 📒 [ultraep/](../../papers/ultraep/README.md) |
| MoE Tile Signaling | [2607.19539](https://arxiv.org/abs/2607.19539) | Linnaeus U | persistent producer/consumer + tile epilogue signal | E2E **2.64×**（**fwd only**） | 📒 [moe-tile-signaling.md](../../papers/moe-tile-signaling.md) |
| AutoOverlap | arXiv'26 | — | Triton 编译器自动 kernel 内 overlap | avg **1.3×**, max **4.7×** | 📒 [autooverlap/](../../papers/autooverlap/README.md) |
| **MoE-Hub** | [2605.05888](https://arxiv.org/abs/2605.05888) | **SJTU** (ISCA'26) | 硬件-软件协同：destination-agnostic GPU hub | 单层 **1.4–3.08×** | ⭐ 待读 |
| FlowMoE | NeurIPS'25 | — | 统一流水线 + chunk 优先级 | **-57%** 训练时间 | 📒 [flowmoe.md](../../papers/flowmoe.md) |
| Piper | arXiv'26 | — | AMD Frontier：PP×EP + 拓扑感知 A2A | **2–3.5×** MFU | 📒 [piper/](../../papers/piper/README.md) |

**技术演进：**

```
chunk 流水线 (Tutel/FasterMoE)
    → tile signaling (FlashOverlap / moe-tile-signaling)
    → 单 kernel 融合 (Comet / UniEP / FlashMoE)
    → 解耦式 EP (MegaScale-Infer → DisagMoE)
    → 硬件加速控制面 (MoE-Hub)
```

---

## 2. 通用 comm-compute overlap

| 论文 | arXiv | 机构 | 核心 | 关键数据 | 笔记 |
|------|-------|------|------|----------|------|
| **FlashOverlap** | [2504.19519](https://arxiv.org/abs/2504.19519) | **清华 + Infinigence** (EuroSys'26) | tile signal + reorder；不侵入 GEMM mainloop | overlap **1.65×** | ⭐⭐ [GitHub](https://github.com/infinigence/FlashOverlap) |
| **TileLink** | [2503.20313](https://arxiv.org/abs/2503.20313) | **ByteDance Seed** (MLSys'25) | tile-centric primitive 编译 overlap kernel | **1.17–20.76×** | ⭐⭐ [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) |
| **Lagom** | [2602.20656](https://arxiv.org/abs/2602.20656) | 清华系 (AutoCCL 延续) | NCCL 参数 co-tune 平衡 comp/comm 争用 | FSDP **1.07–1.33×** | ⭐ |
| Resource-aware Overlap | [2606.09200](https://arxiv.org/abs/2606.09200) | — | SM occupancy shaping + stream priority | **-25.5%** 延迟 | ⭐ |
| Hong et al. (signaling) | [2504.19519](https://arxiv.org/abs/2504.19519) | 清华 | 通用 GEMM+A2A reordering（FlashOverlap 前身） | EuroSys'26 | 见 FlashOverlap |

**与已有笔记关系：** [moe-tile-signaling.md](../../papers/moe-tile-signaling.md) 与 FlashOverlap 同族（tile epilogue signal）；[comet.md](../../papers/comet.md) 是融合 kernel 路线。

---

## 3. 分布式矩阵优化器（Muon / Shampoo / SOAP）

| 论文 | arXiv | 机构 | 路线 | 关键数据 | 笔记 |
|------|-------|------|------|----------|------|
| **DMuon** | [2606.27153](https://arxiv.org/abs/2606.27153) | X Square Robot | FSDP2 **外挂 owner runtime** + Gram SYRK NS | E2E avg **+2%** vs AdamW | 📒 [dmuon.md](../../papers/dmuon.md) |
| **Canzona** | [2602.06079](https://arxiv.org/abs/2602.06079) | **阿里 Qwen** | Megatron 栈；DP α-balanced + TP async micro-group | E2E **1.57×**, optim **5.8×** | ⭐⭐ 待读 |
| **MatrixFSDP** | [2607.05895](https://arxiv.org/abs/2607.05895) | Pittsburgh/Google/清华 | **改 ZeRO-3 shard 布局**；optim step 无 matrix collective | optim **4.2×→54.6×**(1→8 节点) / E2E **1.37×→2.15×** | 📒 [matrixfsdp.md](../../papers/matrixfsdp.md) |

**三条系统路线对比：**

| 路线 | 代表 | 改什么 | 优点 | 缺点 |
|------|------|--------|------|------|
| A. 改 runtime | DMuon | 外挂 owner + pipeline 通信 | FSDP2 drop-in（3 行） | MILP/autotune 调参 |
| B. 改 Megatron 并行 | Canzona | DP/TP owner 解耦 | Qwen 生产栈验证 | 绑 Megatron |
| C. 改 FSDP 分片 | MatrixFSDP | 2D weight 整矩阵 owner | optim step 零 matrix collective | uneven layout；改 shard 语义 |

**算法背景：** [Muon vs AdamW 对比](../../papers/dmuon.md)（见 dmuon.md §1.1）；Moonlight [2502.16982](https://arxiv.org/abs/2502.16982) 证明 Muon LLM scale 可行。

---

## 4. 长上下文 / 大规模训练基础设施

| 论文 | arXiv | 机构 | 核心 | 关键数据 |
|------|-------|------|------|----------|
| **ByteScale** | [2502.21231](https://arxiv.org/abs/2502.21231) | ByteDance + 北大 | Hybrid DP：动态统一 DP+CP，变长序列 | 16384 GPU **7.89×** |
| **ByteRobust** | [2509.16293](https://arxiv.org/abs/2509.16293) | ByteDance | 万卡容错 / checkpoint / 故障定位 | 9600 GPU **97% ETTR** |
| **Decoupled DiLoCo** | [2604.21428](https://arxiv.org/abs/2604.21428) | Google DeepMind | 异步 learner + quorum；超越 SPMD | 故障环境 zero downtime |
| **DeepSeek-V3 硬件洞察** | [2505.09343](https://arxiv.org/abs/2505.09343) | DeepSeek | MLA + FP8 + DeepEP/DeepGEMM | 2048×H800 |

**开源组件：** [DeepEP](https://github.com/deepseek-ai/DeepEP) · [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)

---

## 5. Kernel / 算子融合（训练向）

| 论文 | arXiv | 机构 | 核心 | 训练适用性 |
|------|-------|------|------|------------|
| **CODA** | [2605.19269](https://arxiv.org/abs/2605.19269) | **MIT + Meta + Princeton** | Transformer block → **GEMM-epilogue 程序（含 backward）** | ⭐⭐⭐ 直接面向训练 |
| MoEBlaze | MLSys'26 | — | routed token 不物化 + SwiGLU 融合 | 📒 [moeblaze.md](../../papers/moeblaze.md) |
| FP8-Flow-MoE | [2511.02302](https://arxiv.org/abs/2511.02302) | — | 无 cast 的 FP8 MoE 数据流 | 对标 DeepSeek-V3 |

**CODA 开源：** https://github.com/HanGuo97/coda-kernels

---

## 6. 推理系统 → 训练迁移地图

| 推理论文 | arXiv | 机构 | 推理做了什么 | 训练迁移 | 难度 | 笔记 |
|----------|-------|------|-------------|----------|------|------|
| MegaScale-Infer | [2504.02263](https://arxiv.org/abs/2504.02263) | ByteDance | Attn/FFN 解耦 + M2N + ping-pong | → **DisagMoE 已实现** | 中 | 📒 [megascale-infer.md](../../papers/megascale-infer.md) |
| MPK (Mirage) | [2512.22219](https://arxiv.org/abs/2512.22219) | — | 编译器自动生成 multi-GPU megakernel | 需 backward/autograd | 高 | — |
| Event Tensor / ETC | [2604.13327](https://arxiv.org/abs/2604.13327) | **CMU + NVIDIA** | 动态 shape megakernel；TP RS+GEMM **1.4×** | MoE/TP overlap 可复用 | 中高 | ⭐ |
| Ada-MK | [2605.11581](https://arxiv.org/abs/2605.11581) | 工业界 | MLIR DAG + TensorRT-LLM plugin | decode IO-bound；训练 prefill 收益小 | 低 | — |
| Fleet | arXiv'26 | — | Chiplet megakernel + L2 tiling | AMD；需 persistent backward | 高 | 📒 [fleet.md](../../papers/fleet.md) |
| MoE Tile Signaling | [2607.19539](https://arxiv.org/abs/2607.19539) | — | persistent producer/consumer | 论文 **explicitly 留 backward** | 中 | 📒 |
| FlashMoE | [2506.04667](https://arxiv.org/abs/2506.04667) | NeurIPS'25 | 整层 MoE 单 kernel | backward 依赖链复杂 | 中 | — |
| UniEP | [2604.19241](https://arxiv.org/abs/2604.19241) | ByteDance | 阶段 MegaKernel | **已做 training** | 中 | 📒 |

### 迁移规律（经验总结）

1. **Disaggregation（Attn/FFN 分池）**  
   推理 MegaScale-Infer → 训练 DisagMoE，已被验证。

2. **Tile signaling / persistent kernel**  
   fwd 成熟（FlashOverlap、moe-tile-signaling）；训练需补 **backward tile schedule + gradient routing**。

3. **Megakernel 编译器（MPK / Event Tensor）**  
   推理领先 1–2 年；训练卡点：autograd + 动态 shape + checkpoint。

4. **GEMM-epilogue（CODA）**  
   **最现实的训练迁移路径**——不碰 whole-model megakernel，直接 fuse fwd+bwd epilogue。

---

## 7. 按机构索引

> **更全的机构视角见 [`industry-training-optimization-2026.md`](./industry-training-optimization-2026.md)**（2026-03~08，作者单位逐篇核实，含华为/无问芯穹/美团/B站等本表未覆盖的单位）。

| 机构 | 近期代表作 | 主攻方向 |
|------|-----------|----------|
| **ByteDance Seed** | MegaScale-MoE, Comet, TileLink, UniEP, ByteScale, ByteRobust, MegaScale-Omni, DITRON, **DisagMoE** | MoE 训练全栈 → 编译器化 |
| **DeepSeek** | V3 报告, DeepEP, DeepGEMM, DualPipe | FP8 MoE + EP 通信 |
| **Alibaba (Qwen)** | Canzona, RollArt, QUADS | Muon/Megatron 生产栈 + RL 训练系统 |
| **Huawei (Ascend)** | HiFloat4(预训练/RL), UBEP, StrataCL, CommFuse, HyperParallel-MoE | FP4 训练 + 超节点通信库 |
| **Google DeepMind** | Decoupled DiLoCo, Orbax | 异步/容错 pretraining + checkpoint |
| **Meta** | CODA (with MIT), **HCCL / MTIA 300** | 训练 kernel 抽象 + 通信硬件卸载 |
| **CMU** | Event Tensor (Mowry/Jia/Chen) | 动态 megakernel 编译 |
| **清华 / SJTU** | FlashOverlap, MoE-Hub, Yu Wang 组 | overlap + 体系结构 |
| **MIT** | CODA | Transformer 训练 fusion |
| **Moonshot/Kimi** | Moonlight Muon | 算法 + ZeRO-1 Muon PoC |
| **X Square Robot** | DMuon | FSDP2 drop-in Muon |

---

## 8. 优先阅读队列（相对已有笔记的缺口）

已有 📒 笔记覆盖 MoE overlap 主线 + DMuon。**建议下一步细读：**

| 优先级 | 论文 | 理由 |
|--------|------|------|
| P0 | [FlashOverlap](https://arxiv.org/abs/2504.19519) | 通用 tile overlap 方法论（EuroSys'26）；训练 TP/FSDP 直接受益 |
| P0 | [Canzona](https://arxiv.org/abs/2602.06079) | 阿里 Megatron 栈 Muon；与 DMuon 对照 |
| P0 | [MatrixFSDP](https://arxiv.org/abs/2607.05895) | ZeRO-3 + Muon 另一条系统路线 |
| P1 | [CODA](https://arxiv.org/abs/2605.19269) | MIT/Meta 训练 epilogue fusion（含 backward） |
| P1 | [MoE-Hub](https://arxiv.org/abs/2605.05888) | SJTU ISCA'26；EP 硬件/软件协同 |
| P2 | [Event Tensor](https://arxiv.org/abs/2604.13327) | CMU 动态 megakernel；若做 super-kernel |
| P2 | [ByteScale](https://arxiv.org/abs/2502.21231) | 变长序列 + 16384 GPU |
| P2 | [Lagom](https://arxiv.org/abs/2602.20656) | 低成本 NCCL co-tune 叠加 |

细读完成后用 `paper-deep-analysis` skill 写入 `papers/<slug>.md` 并回写 [`papers/README.md`](../../papers/README.md)。

---

## 9. 问题 → 论文速查

```
训练瓶颈                         优先看的 arXiv 线
──────────────────────────────────────────────────────────
MoE A2A 隐藏                    UniEP / Comet / DisagMoE / FlashOverlap
Muon 分布式                     DMuon 📒 / Canzona / MatrixFSDP
变长序列 + 万卡                  ByteScale / ByteRobust
推理思路回灌训练                 MegaScale-Infer→DisagMoE / Event Tensor / CODA
NCCL 争用调优                   Lagom / Resource-aware Overlap
FP8 MoE                         DeepSeek-V3 报告 / FP8-Flow-MoE
非 GEMM 算子融合                 CODA
MoE 内存                        MoEBlaze 📒 / MemFine 📒
AMD 集群 MoE                    Piper 📒 / UltraEP 📒
```

---

## 10. 相关索引

- 已有深读笔记清单：[`papers/README.md`](../../papers/README.md)
- MoE 100+ 篇分类：[`../moe/paper-landscape.md`](../moe/paper-landscape.md)
- MoE 训练 arXiv 早期整理：[`../moe/recent-arxiv.md`](../moe/recent-arxiv.md)
- TorchTitan 跨版本 diff：[`torchtitan-diff-2025-10-vs-2026-04.md`](./torchtitan-diff-2025-10-vs-2026-04.md)

---

## 变更 log

| 日期 | 变更 |
|------|------|
| 2026-07-31 | 初版：arXiv 2025–2026 训练/推理迁移全景，含机构索引与优先队列 |
