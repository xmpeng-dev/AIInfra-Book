# moe-train — Compact MoE training runtime (PithTrain-inspired rewrite)

> **目标**: 继承 PithTrain 的 schedule-aware protocol + V-PP 调度 + context 注入，重写为 **AMD-first、topology-aware、BackendBundle 驱动** 的 MoE 预训练 runtime
> **平台**: 8× MI355X (gfx950) 优先；保留 NVIDIA backend 路径但不第一版优化
> **参考实现**: [`3rd/pith-train/`](../../3rd/pith-train/)（只借架构，不 fork）
> **关联资产**: `notes/rocmoe/`（FUSED_MOE）、`notes/primus-moe/`（执行层战略）、Primus-Turbo / AITER

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | **设计 2026-08-19 — v2 (Plan IR) 架构设计成文，repo 未创建** |
| 核心取舍 | 保留 DualPipeV bubble 削减 + phase 分解；`comm_stream` → 可替换 `OverlapEngine`；schedule 代码 → `StepPlan` IR（纯函数 + golden 快照）；融合粒度钉在 **stage-pair** |
| 第一版验收 | 单 node MI355X BF16 Qwen3-30B-A3B loss 曲线 vs PithTrain parity |
| 工作名 | `moe-train`（repo 名待定） |

## 设计核心（v2 = v1 的演进，非替换）

| # | PithTrain | v1 | v2 |
|---|---|---|---|
| 1 | stage2/3/4 散落 + RCCL + `comm_stream` | `MoEExecutionUnit` + `OverlapPolicy` enum | **`StepPlan` IR + 四个可互换 `OverlapEngine`** |
| 2 | `training.Linear` 注入 | `BackendBundle`（gemm / attn / moe） | 同 + `Capability` 启动探测 |
| 3 | mesh 内层 CP/EP = NVLink 假设 | `ParallelSpec.ep_scope` topology-aware | 同 + `platform/` arch 事实层 |
| 4 | 同步 checkpoint + memmap shuffle | async ckpt + ShardDataset | 同 + run `manifest.json` |
| 5 | 手写 `resize_(0)` + 注释保证内存安全 | — | **plan liveness 生成 `free` op** |
| 6 | `schedule_simulator.py` 重实现 schedule | — | **executor / estimator / cost 共享同一 plan** |
| 7 | Python-native，禁 in-tree C++ | — | **接受 `kernels/hip/`（C3 强制），每核带 reference** |

## 阶段计划（v2）

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | `plan/` + `SerialEngine` + Rocm BF16 backend | 单 node ep=8 loss 降；plan golden 通过 |
| M1 | DualPipeV plan + liveness free | loss 曲线 vs PithTrain（< 5e-3）；plan 不变式全绿 |
| M2 | cost model + memory estimator（共享 plan）+ manifest | peak memory 预测误差 < 10% |
| M3a | `SdmaPeerEngine` | rocprof 证明 overlap 且未抢 CU |
| M3b | `FusedStageEngine`（HIP，两对 fusion group） | T=8192 MoE 段优于 Serial |
| M4 | FP8 `f8f6f4` + async ckpt + 多节点 pp>1 | FP8/BF16 parity；resharding 测试 |

## 文件索引

| 文件 | 内容 |
|---|---|
| [2026-08-19_1229_moe_train_runtime_v2_plan_ir_architecture_design.md](./2026-08-19_1229_moe_train_runtime_v2_plan_ir_architecture_design.md) | **v2（当前）**：三条 AMD 硬约束、`StepPlan` IR、四个 `OverlapEngine`、EP-invariant 引擎契约、优势与代价论证、里程碑 |
| [2026-08-19_1046_moe_train_runtime_v1_architecture_design.md](./2026-08-19_1046_moe_train_runtime_v1_architecture_design.md) | v1：DNA、四层结构、protocol/backend/unit/policy、里程碑、开放问题 |

## 相关

- [PithTrain 源码](../../3rd/pith-train/)
- [RocMoE v2 架构](../rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md)
- [Primus 执行层战略](../primus-moe/2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md)

三条硬约束的证据来源（v2 §2）：

- C1 RCCL 侧流不 overlap：[`notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md`](../monolith-moe/2026-04-14_rccl_overlap_analysis.md)
- C2 整层融合训练规模翻盘：[`notes/monolith-moe/`](../monolith-moe/)、[`notes/peer-tiles/`](../peer-tiles/)
- C2 stage-pair 融合有效：[`notes/MegaMoeFlydsl/`](../MegaMoeFlydsl/)
- C3 AMD 库缺口：[`knowledge/kernels/fp8-expert-gemm.md`](../../knowledge/kernels/fp8-expert-gemm.md)、[`knowledge/libraries/`](../../knowledge/libraries/)

## 维护

- 架构大版本用 `YYYY-MM-DD_HHMM_*_architecture_design.md`；实现进展用 `progress-note` skill。
- 更新本 README 的 **状态** 与 **文件索引**  whenever 新设计/里程碑落地。
