# hk-attn-bwd — HipKittens attention backward 切入 Primus

> **目标**: 以 attention backward 为切口，把 HipKittens 的内核经由 Primus-Turbo 接进 Primus 训练栈，同时积累「AMD attention backward 专家能力」这一可迁移资产
> **平台**: MI355X (gfx950, CDNA4)
> **栈**: ROCm / HipKittens / Primus-Turbo / Megatron
> **范围**: Llama / Qwen 系 **GQA** backward。**不含 DeepSeek MLA**（另见 [`megaattn`](../megaattn/README.md)）
> **工作仓库**: `~/workspace/hipKitten/`（`kernels/cdna4/attn/` + `training/llama/`）
> **对接仓库**: `~/workspace/Primus/`（Primus-Turbo 算子层，`use_turbo_attention` 开关）

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | **on-hold 2026-08-12 11:54 — 三轮证据后判定 HK 在 dense attention backward 上边际价值很低**。①论文「AITER 只有 SoTA 30%」的基线实为 MI300X 级软件；②MI355X 上 AITER backward 已达 **902–1047**，持平/反超 HK 论文的 1024；③DSV3 实测：**HK 源码层跑不了 MLA 形状**（单一 `ATTN_D=128`，无 D_QK/D_V），且配置修好后 **turbo ≈ TE 差 <2%**，剩余空间在配置层不在内核层。详见 [§0 / §0.1](./2026-08-12_1006_hk_primus_attn_bwd_plan.md);**收口动作**: P0' 查 16-head 反向机制、P1' 补测 GQA 比例，两者均无发现即关闭 |
| ⚠ 前提修正 | 及格线由「打败 272–403」上调至 **~1050**。方向性反转：非 causal 长序列上 MI355X 的 **BWD 反超 B200**，落后的是 **FWD**（FlyDSL 地盘） |
| MLA 出范围（已确认） | HK `training/llama/csrc/attn_fwd_causal.cpp:22` 为 `constexpr int ATTN_D = 128`，K/V shared tile 复用同一常量。与 FA2 / 原生 aiter / TE AOTriton 同属「不支持 qk≠v 头维」失败类别 |
| 仅存的待查格子 | 16 heads（TP=8 每卡份额）反向掉到 **~655 TFLOPS**。但机制存疑——绝对时间仅 1.31 ms，更可能是 dQ 原子归约 / launch 固定开销占比放大，而非调度不良。**先 rocprof 定性** |
| 切口依据 | backward 是 attention 里寄存器压力最高处（dQ/dK/dV 累加器共存）——正是 HK 独有的 register pinning 的用武之地，也是 FlyDSL「交给 LLVM 分配」设计最吃亏处。**结构性缺口，非偶然** |
| 通道 | Primus-Turbo 算子层 + `use_turbo_attention` 开关；先例为 FlyDSL 供给 FP8 GEMM（Primus commit `fa391f32`） |
| 已有资产 | HK 的 `training/llama/llama/models/attentions/` 本身就是可插拔 attention 后端抽象，并排放着 `aiter.py` 对照组，与 Turbo 开关同构，M0 几乎不用搭台 |
| M0 门槛 | HK vs AITER vs TE/CK 在**Primus 生产形状**上的 backward 性能表；**不占优则就地重估** |
| 主要障碍 | HK 内核走编译期模板（`ATTN_B/ATTN_H/ATTN_H_KV/ATTN_N`），batch 固定；生产要变长与多形状 |

## 形状缺口（M0 的输入）

| 模型 | H_Q | H_KV | GQA 比 | head_dim | seq | 状态 |
|---|---|---|---|---|---|---|
| Llama3.1-8B | 32 | 8 | 4:1 | 128 | 2048 / max 131072 | 待验证 |
| Llama3.1-70B | 64 | 8 | 8:1 | 128 | max 131072 | 待验证（**HK Makefile 默认值恰为此形状**） |
| Qwen3-235B-A22B | 64 | 4 | 16:1 | 128 | 4096 | 待验证 |
| DeepSeek-V3 | 128 | MLA | — | qk/v 128 | 4096 | **不在范围** |

HK `setup_kernels.sh` 当前只构建 `B=8 H=16 H_KV=16 N=2048`（MHA），与生产形状均不匹配；但 Makefile 默认 `H=64/H_KV=8` 说明 GQA 路径本身支持。

## 阶段计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | HK vs AITER vs TE/CK backward 基准，用生产形状 | 拿到性能表；不占优则叫停 |
| M1 | 形状覆盖：GQA 4:1/8:1/16:1 × seq 2048/4096/8192，解掉 batch 编译期固定 | 生产形状全绿 + 与 AITER 数值对拍 |
| M2 | 接入 Primus-Turbo，挂 `use_turbo_attention` | 端到端训练步时下降可度量 |
| M3 | XCD 感知调度 + register pinning 在 backward 的结合 | 相对 M1 有增量；目标可发表 |

## 风险

| 风险 | 对冲 |
|---|---|
| AMD 投入方向在 FlyDSL（官方 / 781 commits / 已进 Primus 致谢），HK 是研究产物 | **资产定义为「能力」而非代码库**；长期可把 register pinning 作为逃生舱反向提案给 FlyDSL |
| HK 编译期模板与生产变长需求冲突 | 即 M1 主要工程量，也是最难替代的积累 |
| HK 主干不完整（FP6 只在 `fp6_experimental` 分支） | 不把论文全部结果当主干可复现 |

## 文件索引

| 文件 | 内容 |
|---|---|
| [2026-08-12_1006_hk_primus_attn_bwd_plan.md](./2026-08-12_1006_hk_primus_attn_bwd_plan.md) | 缺口分析（FlyDSL 无 backward 的证据、HK 资产盘点、Turbo 通道与 FlyDSL 先例）、形状缺口表、四阶段计划、风险与对冲 |

## 相关

- [`papers/hipkittens.md`](../../papers/hipkittens.md) · [全文中译](../../papers/hipkittens-zh.md) — 切口的论文依据
- [`papers/swizzled-head-first-attention.md`](../../papers/swizzled-head-first-attention.md) — M3 的先行工作（backward 只有 1.10×，作者留白）
- [`knowledge/systems/primus-pipeline-runtime-megatron-integration.md`](../../knowledge/systems/primus-pipeline-runtime-megatron-integration.md) — Primus 接入机制
- [`megaattn/`](../megaattn/README.md) — MLA 方向，与本项目互补不重叠
