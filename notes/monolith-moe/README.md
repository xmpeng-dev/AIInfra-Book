# monolith-moe — MoE Super-Kernel + Comm-Compute Overlap (MI355X)

> **目标**：在 8×MI355X 上做 MoE 层的 **CCO（Communication-Computation Overlap）**，把 GEMM 计算与 All-to-All 通信融合在同一个 persistent kernel 内部，跨过 RCCL multi-stream 无法 overlap 的架构限制
> **平台**：8×MI355X (gfx950, XGMI 全互联)
> **栈**：PyTorch 2.12+rocm7.1 / ROCm 7.2, Triton (ByteDance fork), triton-distributed 3.4.0, primus_turbo CK
> **目标模型**：DeepSeek-V3 671B（256E EP8→32 local, top_k=8, H=7168, F=2048）

## 状态

| 维度 | 值 |
|---|---|
| Hand-written Grouped GEMM (DSV3 grouped, MOe shape) | **GateUP 530T / Down 520T** vs CK 1050T / 960T （**0.50×** —— 阻塞 4.8ms 端到端目标）|
| 上一代 micro-bench (M=128 单形状) | FC1 497T / FC2 498T （vs CK micro 376T，+32%）但**不代表 grouped 性能**|
| Layout C 设计 | 5 种 IPC buffer layout 对比完成，Layout C 最优（XGMI ~95% 效率 + 无 gather） |
| RCCL multi-stream 路 | ❌ 架构限制，否决 |
| TileFlow rocSHMEM 移植 | ❌ device bitcode 与 gfx950 不兼容 |
| HIP IPC + Triton system atomics | ❌ Triton gfx950 system-scope atomic codegen 不完整 |
| 选定方向 | **HIP C++ IPC**，参考 DeepEP `atomicAdd_system + ld_volatile_global` |
| 项目状态 | 阶段性收尾（2026-04-14 后切到 gpt-oss / mlperf-llama） |

## 进展时间线

| 日期 | 里程碑 | 关键数字 | 来源 note |
|---|---|---|---|
| 2026-04-09 | TileFlow tile-level overlap 调研（Comet 移植到 AMD） | stream-based overlap PASS；rocSHMEM tile path 受 ROCm bitcode 阻塞 | [tileflow_mi355x](./2026-04-09_tileflow_mi355x_tile_overlap_analysis.md) |
| 2026-04-10 | **MFMA Grouped GEMM v4c → production** | FC1 247 → 497T (+101%)，**vs CK +32% / +46%** | [mfma_gemm](./2026-04-10-mfma-gemm-optimization.md) |
| 2026-04-10 | MonolithMoE Layout C 重构 + Pack phase 优化 | OPT-1~8 理论 ~10× pack 加速，端到端预估 4.83 ms (1.75× vs baseline 8.47 ms) | [monolithmoe_layout_c_pack](./2026-04-10_monolithmoe_layout_c_pack_optimization.md) |
| 2026-04-13 | MoE comm+compute overlap 总结 | RCCL multi-stream 4× 反慢；rocSHMEM bitcode 阻塞；HIP IPC + Triton sys atomic hang | [moe_comm_overlap](./2026-04-13_moe_comm_overlap_analysis.md) |
| 2026-04-14 | IPC buffer layout (A–E) 全面对比 | Layout C 胜出 | [data_layout](./2026-04-14_data_layout_analysis.md) |
| 2026-04-14 | RCCL 无法 overlap 根因（架构层面） | host-initiated + collective semantics 决定 | [rccl_overlap](./2026-04-14_rccl_overlap_analysis.md) |
| 2026-04-14 | Tile-level overlap 数学可行性分析 | overlap 收益条件 + 临界点 | [tile_overlap](./2026-04-14_tile_overlap_analysis.md) |
| 2026-04-14 | MoE E2E baseline + overlap 上限估算 | Small 1.29× / DSV3 推理 1.13× / DSV3 训练 1.08×（70% overlap 假设） | [moe_e2e](./2026-04-14_moe_e2e_performance_benchmark.md) |
| 2026-05-08 | **CK Grouped GEMM 实现深度解析（vs v8a 530T）** | 拆解 6 项叠乘优化（Stream-K / 3-stage prefetch / LDS swizzle / 精确 waitcnt / C-shuffle / persistent counter），估算 530T → 860T 闭合路线 | [ck_deep_dive](./2026-05-08_ck_implementation_deep_dive.md) |

## 下一步（按 ROI）

| 优先级 | 方向 | 说明 |
|---|---|---|
| P0 | **3-stage prefetch + Stream-K** | 单点最大 GEMM 收益（530T → 860T 估算），DSV3 4.8ms 目标的前置依赖；详见 [ck_deep_dive](./2026-05-08_ck_implementation_deep_dive.md) |
| P0 | HIP C++ 实现 IPC 通信 kernel | 参考 DeepEP，`atomicAdd_system` 替代 RCCL/Triton（已有 `csrc/ipc_utils.cuh`，super-kernel 在用） |
| P1 | LDS XOR swizzle 替换 PAD=4 + 拿掉 K-step barrier | 解锁 prefetch 深度，与 P0 联动 |
| P1 | 自适应 M_TILE / 多版本派发 | DSV3 sparse 形状下 +20–40%，README Phase 2 TODO |
| P2 | `rocprof` 实测 Pack OPT-1~8 | 验证理论 10× 加速 |
| P2 | C-shuffle epilogue | +3–5%，FC2 影响小 |
| P3 | 跨节点 ROCShmem + Persistent Kernel | IB 场景下 comm 更长，overlap 收益更大 |

## 文件索引

| 主题 | 文件 |
|---|---|
| GEMM kernel | [`2026-04-10-mfma-gemm-optimization.md`](./2026-04-10-mfma-gemm-optimization.md), [`2026-05-08_ck_implementation_deep_dive.md`](./2026-05-08_ck_implementation_deep_dive.md) |
| Super-kernel + Layout | [`2026-04-10_monolithmoe_layout_c_pack_optimization.md`](./2026-04-10_monolithmoe_layout_c_pack_optimization.md), [`2026-04-14_data_layout_analysis.md`](./2026-04-14_data_layout_analysis.md) |
| Overlap 调研 / 排障 | [`2026-04-09_tileflow_mi355x_tile_overlap_analysis.md`](./2026-04-09_tileflow_mi355x_tile_overlap_analysis.md), [`2026-04-13_moe_comm_overlap_analysis.md`](./2026-04-13_moe_comm_overlap_analysis.md), [`2026-04-14_rccl_overlap_analysis.md`](./2026-04-14_rccl_overlap_analysis.md) |
| 可行性 + baseline | [`2026-04-14_tile_overlap_analysis.md`](./2026-04-14_tile_overlap_analysis.md), [`2026-04-14_moe_e2e_performance_benchmark.md`](./2026-04-14_moe_e2e_performance_benchmark.md) |

> 周报视角：[`weekly-reports/2026-04-14_weekly_report_0409_0414.md`](../weekly-reports/2026-04-14_weekly_report_0409_0414.md)

## 维护约定

每写一篇 note → 同时回写本 README 的 **进展时间线** 和 **下一步**。
