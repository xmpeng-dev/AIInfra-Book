# ante — Agent-Native Training Engineering on CDNA

> **目标**：构建面向 AI Agent 设计的 CDNA 训练工程环境，使通用 coding agent 能完成从端到端训练性能问题下钻到 GPU kernel 修改的完整闭环。
> **平台**：MI355X（CDNA4, gfx950）
> **栈**：ROCm / HipKittens / PyTorch
> **工作仓库**：`~/workspace/ante/`（开题分析见 `docs/proposal.md`）
> **相关仓库**：`~/workspace/hipKitten/`（kernel 底座）· `~/workspace/Primus/`（双臂对照的 A 臂）· `~/workspace/FlyDSL/`（外环骨架参考）

## 状态

**开题阶段。** 尚无实验。

## 核心命题

训练层与 kernel 层的 agent 瓶颈不同，因此两层需要各自的 agent-native 设计。训练层原则采纳既有验证；**kernel 层的四条原则由本项目提出**：反馈延迟即第一约束 · 机器状态是一等输出 · 动作空间先闭合后开放 · 归因优先于结果。

核心研究问题：**当模型预训练语料里几乎没有该领域知识时，外部供给的结构化证据能否替代它？**

## 阶段判据

| 阶段 | 证明 | 判据 |
|---|---|---|
| S1 | 接缝成立 | 改一个 tile 尺寸触碰文件数 = 1；`Edit-to-Verdict` 进入分钟级 |
| S2 | 可改性的不可替代价值 | 交付黑盒后端结构上做不到的融合（GEMM epilogue 融 FP8 quant）；不要求打赢在位实现 |
| S3 | 证据能否替代预训练知识 | `Ground-Truth Recovery Rate`；四格消融 |
| S4 | 跨层归因成立 | L3 题型，双臂对照 |
| S5 | 复利成立 | `Knowledge Reuse Gain` 单调下降 |

## 时间线

| 日期 | 事件 | 记录 |
|---|---|---|
| 2026-08-17 | 开题分析定稿，repo `ante` 建立 | [`ante/docs/proposal.md`](../../../ante/docs/proposal.md) |

## 下一步

细化 design docs 1–4（算子契约 · 证据 schema · kernel 隔离编译 · 闭合动作空间）。它们共同决定 agent 在 kernel 层能否动手，是其余部分的前提。索引见 `ante/docs/design/README.md`。

## 已核实的技术前提

| 项 | 结论 | 来源 |
|---|---|---|
| HipKittens CDNA4 覆盖 | `gemm`（bf16fp32/fp8fp32/mxfp8）· `attn`（gqa ± causal ± backwards）· `layernorm` · `rotary` · `softmax` · `torch_scaled` | `hipKitten/kernels/cdna4/` |
| **CDNA4 无 grouped GEMM / MoE kernel** | 唯一 `gemm_expert.cpp` 在 `kernels/udna1/.../gfx1250/`，非 MI355X | 同上 |
| 编译期模板固定 | `ATTN_B/H/H_KV/N/D` 均为 `constexpr` | `hipKitten/training/llama/csrc/attn_fwd_causal.cpp` |
| 无跨 WG 同步 | `ops/warp/` 只有 `memory`/`register`/`shared`，`ops/group/` 只有 `memory`；子树内 `__hip_atomic_*` 与 `threadfence` 零出现 | `hipKitten/include/cdna4/ops/` |

## 相关

- [`papers/hipkittens.md`](../../papers/hipkittens.md) —— 底座数字与机器状态清单的出处
- [`notes/hk-attn-bwd/README.md`](../hk-attn-bwd/README.md) —— kernel 底座立项，S2 的消费者分析
- [`knowledge/systems/industry-training-optimization-2026.md`](../../knowledge/systems/industry-training-optimization-2026.md) —— 行业动向
