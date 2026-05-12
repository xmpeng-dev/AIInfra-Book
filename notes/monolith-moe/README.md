# monolith-moe — MoE Super-Kernel + Comm-Compute Overlap (MI355X)

> **目标**：在 8×MI355X 上做 MoE 层的 **CCO（Communication-Computation Overlap）**，把 GEMM 计算与 All-to-All 通信融合在同一个 persistent kernel 内部，跨过 RCCL multi-stream 无法 overlap 的架构限制
> **平台**：8×MI355X (gfx950, XGMI 全互联)
> **栈**：PyTorch 2.12+rocm7.1 / ROCm 7.2, Triton (ByteDance fork), triton-distributed 3.4.0, primus_turbo CK
> **目标模型**：DeepSeek-V3 671B（256E EP8→32 local, top_k=8, H=7168, F=2048）

## 状态

| 维度 | 值 |
|---|---|
| **End-to-end super-kernel (DSV3 SPARSE, 8×MI355X)** | **7.95 ms / 363.1 TFLOPS / 1.07× vs PyTorch+RCCL 8.466 ms**（2026-05-12 P2 batched compute） |
| End-to-end super-kernel (TILE-FIT) | **4.31 ms / 191.4 TFLOPS**（2026-05-12 P2） |
| 调优脉络（DSV3 SPARSE wall） | 24.0 → 15.21 → 14.86 → 14.04 → **8.80**（P0 flat+small tile）→ **7.95**（P2 batched compute） |
| Hand-written Grouped GEMM (DSV3 grouped) | GateUP 530T / Down 520T vs CK 1050T / 960T（**0.50×**，super-kernel 内嵌 MFMA 上限） |
| Layout C 设计 | 5 种 IPC buffer layout 对比完成，Layout C 最优（XGMI ~95% 效率 + 无 gather） |
| 选定方向 | **HIP C++ IPC**（已实现），参考 DeepEP `atomicAdd_system + ld_volatile_global` |
| 项目状态 | **重新激活中**（2026-05-12 起 P0/P1/P2 多轮优化已落地） |

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
| 2026-05-09 → 05-12 | **P0 全程：super-kernel 24.0 → 8.80 ms（项目里最大一跃）** | 四步串联：`wgs/CU=1 ratio=0.25` (24.0→15.21) → per-src wait_flag (→14.86) → per-src compute-barrier coalescing (→14.04) → **flat tile dispatch + adaptive small-tile (→8.80)**；compute_barrier_1 8.55 → 0.29 ms；FC1+FC2 11.19 → 6.11 ms (−45 %) | [p0_full_arc](./2026-05-12_1230_super_kernel_p0_full_arc_24_to_8p8ms.md) |
| 2026-05-12 13:10 | **P1 tail-pipelining 失败 + critical-path 修正** | per-pair round-robin DSV3 8.80 → 9.98 ms regress；hybrid 持平；确认 wall = max(per-WG kernel_total) = compute，tail 不在临界路径 | [p1_tail_failed](./2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md) |
| 2026-05-12 13:35 | **P2 Batched (e, src, mi, ni) compute + 单相 FC1/FC2** | 24 cross-WG barrier → 3；DSV3 **8.80 → 7.95 ms（1.07× vs PyTorch+RCCL，首次反超）**；TILE-FIT **5.07 → 4.31 ms（−15 %）**；L2 weight reuse 假设证伪，真实收益来自 barrier 合并 | [p2_batched_compute](./2026-05-12_1335_batched_compute_p2_single_phase.md) |
| 2026-05-12 14:55 | **A1 per-expert pipelined compute 失败** | 把外层换成 `for e in epg` + per-expert 3 barrier；DSV3 SPARSE 7.95 → 21.6 ms (+172 %)，TILE-FIT 4.31 → 4.56 ms；并行度从 192-wide 摊到 ~30-wide + 96 vs 3 barrier；已 revert 到 P2 (`f50bb43`)；复盘 P1/A1 同款诊断陷阱 | [a1_pipelined_failed](./2026-05-12_1455_per_expert_pipelined_a1_failed_lost_flat_tile_parallelism.md) |
| 2026-05-12 15:30 | **A2 M-concat tile dispatch 失败** | 物理 gather + per-expert (e, mi, ni) flat tile + 4 barrier；DSV3 SPARSE 7.95 → 9.03 ms (+13.6 %)，TILE-FIT 4.31 → 6.75 ms (+56.6 %)；tile 数从 8192 → 1024（DSV3）/ 512 → 128（TILE-FIT < 192 WGs）→ WG 利用率坍塌；L2 reuse 假设在 P2 时已经成立（隐式），A2 是浪费；已 revert；P1/A1/A2 三连同根：MI355X compute 是 parallelism-bound，不要再动 compute 排布 | [a2_concat_failed](./2026-05-12_1530_m_concat_a2_failed_parallelism_collapse.md) |
| 2026-05-12 16:35 | **P0a 3-stage prefetch 失败 — workload 不是 HBM-latency-bound** | CK-style 3 LDS buffer + 提前 2 tile issue + 精确 `vmcnt(N)`；50/50 PASS、DSV3 / TILE-FIT 全 ratio 与 P2 在 ±1% 噪声内等价；hipcc `-Rpass-analysis` 显示 P2 已经 256 VGPR + 240 B/lane scratch + 1 wave/SIMD，3-stage 同资源；1-stage prefetch 已把 HBM 完全盖住，gap 不在这；patch 存档 `benchmarks/results/3stage_prefetch.patch`，已 revert；下一步先 profile MFMA pipeline 再决定 swizzle / waitcnt / pre-permute | [p0a_3stage_failed](./2026-05-12_1635_3stage_prefetch_p0a_failed_not_hbm_latency_bound.md) |

## 下一步（按 ROI）

| 优先级 | 方向 | 说明 |
|---|---|---|
| **P0** | **Profile `mfma_gemm_tile_t` 内部**（PROFILE_DECL or perf-counters） | A1/A2/P0a 三连失败的共同教训：**不 profile 直接动 GEMM 主循环 ≈ 50 % 失败率**；先确认 357T → 1050T 的 700T gap 是在 dependency stall 还是 LDS bank conflict 还是 syncthreads；估算 1d |
| **P0** | **FP8 / mxfp8 weights for FC1 / FC2** | A2 失败后唯一剩下的 DSV3 大杠杆：HBM weight 流量直接 ÷2，与 layout 重排正交（不影响 tile 数 / WG 利用率）。MI355X 原生支持 mxfp8 MFMA。预期 DSV3 FC 5.86 → ~3 ms（−2.8 ms） |
| **P0** | **TILE-FIT scatter 总时间下降**（pack+scatter 合并 / per-rank XGMI link 并行 / skip empty pair early） | A1 / A2 证伪「让 wait/L2 reuse 帮忙」的路径；要省 wall 必须真正缩短 scatter 的物理时间，目标 TILE-FIT −1 ms |
| P1 | K-step 间去掉 `lgkmcnt(0)` 隐式 barrier | 需要 NSTAGES ≥ 2 LDS buffer（已满足），CK 实测 +3–7 % |
| P1 | LDS XOR swizzle 替 PAD=4 | +5–10 %（与 K-step barrier 联动）；详见 [ck_deep_dive](./2026-05-08_ck_implementation_deep_dive.md) |
| P1 | C-shuffle epilogue | +3–5 %，FC2 影响小，可与 FP8 联动 |
| P1 | Weight pre-permutation to MFMA fragment layout | 把 weight 离线重排成 MFMA fragment 序，去掉 LDS write + 减少 ds_read 仲裁；与 FP8 正交 |
| P2 | Work-stealing tile counter（atomic 替代 round-robin） | 吸收 T_e 不均，−0.2~0.5 ms |
| P2 | ~~3-stage prefetch~~ | 已证伪（[p0a_3stage_failed](./2026-05-12_1635_3stage_prefetch_p0a_failed_not_hbm_latency_bound.md)）：workload 不吃 HBM 延迟隐藏 |
| P3 | 重启 tail WG 并行化（compute 真比 tail 快之后） | 必须先把 atomic spin 换成 LDS-cached flag，否则像 P1 一样回退 |
| P3 | 跨节点 ROCShmem + Persistent Kernel | IB 场景下 comm 更长，overlap 收益更大 |

## 文件索引

| 主题 | 文件 |
|---|---|
| GEMM kernel | [`2026-04-10-mfma-gemm-optimization.md`](./2026-04-10-mfma-gemm-optimization.md), [`2026-05-08_ck_implementation_deep_dive.md`](./2026-05-08_ck_implementation_deep_dive.md) |
| Super-kernel + Layout | [`2026-04-10_monolithmoe_layout_c_pack_optimization.md`](./2026-04-10_monolithmoe_layout_c_pack_optimization.md), [`2026-04-14_data_layout_analysis.md`](./2026-04-14_data_layout_analysis.md) |
| Overlap 调研 / 排障 | [`2026-04-09_tileflow_mi355x_tile_overlap_analysis.md`](./2026-04-09_tileflow_mi355x_tile_overlap_analysis.md), [`2026-04-13_moe_comm_overlap_analysis.md`](./2026-04-13_moe_comm_overlap_analysis.md), [`2026-04-14_rccl_overlap_analysis.md`](./2026-04-14_rccl_overlap_analysis.md) |
| 可行性 + baseline | [`2026-04-14_tile_overlap_analysis.md`](./2026-04-14_tile_overlap_analysis.md), [`2026-04-14_moe_e2e_performance_benchmark.md`](./2026-04-14_moe_e2e_performance_benchmark.md) |
| Super-kernel 调优 (E2E) | [`2026-05-12_1230_super_kernel_p0_full_arc_24_to_8p8ms.md`](./2026-05-12_1230_super_kernel_p0_full_arc_24_to_8p8ms.md), [`2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md`](./2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md), [`2026-05-12_1335_batched_compute_p2_single_phase.md`](./2026-05-12_1335_batched_compute_p2_single_phase.md), [`2026-05-12_1455_per_expert_pipelined_a1_failed_lost_flat_tile_parallelism.md`](./2026-05-12_1455_per_expert_pipelined_a1_failed_lost_flat_tile_parallelism.md), [`2026-05-12_1530_m_concat_a2_failed_parallelism_collapse.md`](./2026-05-12_1530_m_concat_a2_failed_parallelism_collapse.md), [`2026-05-12_1635_3stage_prefetch_p0a_failed_not_hbm_latency_bound.md`](./2026-05-12_1635_3stage_prefetch_p0a_failed_not_hbm_latency_bound.md) |

> 周报视角：[`weekly-reports/2026-04-14_weekly_report_0409_0414.md`](../weekly-reports/2026-04-14_weekly_report_0409_0414.md)

## 维护约定

- **每一次优化都要写一篇 note**（即使是失败实验或临界路径修正），命名格式
  `YYYY-MM-DD_HHMM_<topic_slug>.md`，遵循 `progress-note` skill 模板。
- 每写一篇 note → 同时回写本 README 的 **进展时间线**、**状态** 和 **下一步**。
- 老 note 不改（除非加 superseded 横幅），项目当前状态以本 README 为单一真源。
