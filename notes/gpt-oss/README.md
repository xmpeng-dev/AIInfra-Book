# gpt-oss — GPT-OSS-20B MLPerf 调优 (MI355X)

> **目标**：1×8 MI355X 上把 GPT-OSS-20B MLPerf training submission 时间压到最低
> **平台**：1×8 AMD Instinct MI355X (gfx950)
> **栈**：Primus + Megatron-LM, ROCm 7.x, FP8 hybrid GEMM + bf16 weight

## 状态

| 维度 | 值 |
|---|---|
| 当前最佳 E2E | **9963 s (B11_full)** vs base 10320 s, val 3.3247 ✓ |
| 稳态 step / TFLOP | **775 ms / 533 TFLOP/s/GPU** (mbs=2 路径) |
| 已验证落地 | B11 = Triton RMSNorm + bf16 grad reduce; Tier 1A V1+V2 fused (residual+RMSNorm) 80-iter smoke −1.04% / −0.93% |
| 项目状态 | active（2026-04-28 最近一笔） |

详细阅读顺序、故事线、配置 diff 见 [`2026-04-20_gptoss_INDEX.md`](./2026-04-20_gptoss_INDEX.md)。

## 进展时间线

| 日期 | 里程碑 | 关键数字 | 来源 note |
|---|---|---|---|
| 2026-04-19 | 首个 ✅ E2E 收敛 run | RESULT 10375 s · val 3.3345 · step 862 ms / 479 TFLOP/s | [01](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md) |
| 2026-04-20 | B10 自写 Triton RMSNorm | step 796 → 770 ms (−3.3%), RMSNorm 31% → 4.6% | [02](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md), [03](./2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md) |
| 2026-04-20 | B11 `grad_reduce_in_bf16` | step 770 → 713 ms (−7.45%), 释放 36 GB | [04](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |
| 2026-04-20 | DeepSeek-V3 / Qwen3 yaml 借鉴 | 推荐 8 行 diff，估 −12% E2E | [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| 2026-04-20 | **B11 全量收敛验证** | RESULT **9963 s**, val 3.3247 ✓ | [06](./2026-04-20_gptoss_06_B11_full_convergence_report.md) |
| 2026-04-20 | 优化项总览表 | B0/B10/B11/B12/B13 ROI 排序 | [07](./2026-04-20_gptoss_07_optimization_summary_table.md) |
| 2026-04-21 | mbs=4/gbs=32 6 阶累积链 | step −18.8% / TFLOPs +24%; NUMA off −7.7%; Triton RMSNorm ROI 归零 | [08](./2026-04-21_gptoss_08_mbs4_optimization_chain.md), [09](./2026-04-21_gptoss_09_M4_C5_profile_bottleneck.md) |
| 2026-04-21 | MoE FP8 grouped GEMM 诊断 + postmortem | 问题定位 | [10](./2026-04-21_gptoss_10_moe_fp8_grouped_gemm_diagnosis.md), [11](./2026-04-21_gptoss_11_moe_fp8_GG_actual_run_postmortem.md) |
| 2026-04-21 | Comm 优化 B 系列 | 见 note | [12](./2026-04-21_gptoss_12_comm_optimization_B_series.md), [13](./2026-04-21_gptoss_13_attn_bwd_recompile_postmortem.md) |
| 2026-04-23 | HSDP-2 ❌ 负结果 | DP=8 下 +23.8% 慢 | [14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md) |
| 2026-04-24 | EP=1 单步 trace 校准 + 优化栈重排 | 暴露 comm ≈46 ms; stream-0 elementwise tax = 197 ms (17.4%) | [15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md), [16](./2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md) |
| 2026-04-24 | **Tier 1A V1** fused (residual+RMSNorm) Triton kernel | 80-iter smoke −1.04% step / +1.04% TFLOP/s, 0 NaN | [17](./2026-04-24_gptoss_17_fused_residual_rmsnorm_impl.md), [18](./2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md) |
| 2026-04-24 | **Tier 1A V2** 跨层 ADD#2 + 下一层 input_layernorm 融合 | 80-iter smoke −0.93% step / +0.93% TFLOP/s, 0 NaN; V3 q/k_norm 不落 patch | [19](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md) |
| 2026-04-25 | `use_turbo_grouped_mlp` env-var truthy fix | yaml `"false"` 被当 truthy 的 bug | [20](./2026-04-25_gptoss_20_use_turbo_grouped_mlp_envvar_truthy_fix.md), [21](./2026-04-25_gptoss_21_turbo_grouped_mlp_trace_evidence.md) |
| 2026-04-25 | turbo sync-free MoE 全 stage 实测 | 仅 stage 1 +0.5%，stage 2/3 反慢 | [22](./2026-04-25_gptoss_22_sync_free_moe_stage_audit.md) |
| 2026-04-25 | Tier 1B Triton SwiGLU no-cat bwd verify | — | [23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md) |
| 2026-04-25 | V2 fused residual+norm 800-iter A/B verdict | — | [24](./2026-04-25_gptoss_24_fused_residual_norm_v2_800iter_verdict.md) |
| 2026-04-27 | RMSNorm 优化 wave 过夜 time-to-quality A/B | — | [25](./2026-04-27_gptoss_25_overnight_rmsnorm_wave_timetotarget.md) |
| 2026-04-28 | MLPerf v6.0 合法 baseline 复测 + 修订 note 25 wave 收益 | — | [27](./2026-04-28_gptoss_27_mlperf_legal_baseline.md) |
| 2026-05-07 | **B200 step #17 baseline 建立**（rank2+rank3 平均）+ note 16 头注硬件更正 | step **966.75 ms / 32 sample**; SendRecv 224 ms (23%) + RS 192 + AG 154 + GEMM 141; compute-busy 48.5%, NCCL hidden 24% | [28](./2026-05-07_gptoss_28_b200_baseline_step17.md) |

## 下一步（按 ROI）

| 优先级 | 方向 | 来源 |
|---|---|---|
| ✅ | Tier 1A V1/V2 800-iter normal-LR 收敛终判 | [18](./2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md), [19](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md) |
| ★★ | Tier 1B SwiGLU bwd 去 cat (−0.5~0.9%) + Tier 1C `direct_copy` 排查 (−0.5~0.9%) | [16](./2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md), [23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md) |
| ★★ | Tier 2 — Optimizer tail HIP-graph 包 (最后 40 ms 单流串行) | [15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ★★ | 修 regress：把 B1+B2（`ddp_average_in_collective` + `bucket_size=100M`）落回 baseline | [12](./2026-04-21_gptoss_12_comm_optimization_B_series.md), [14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md) |
| ★★ | 推荐 yaml diff 落到下次 submission run | [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★ | Tier 4 — FMHA bwd 调优 (aiter::fmha_bwd tile / sliding-window) | [15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ★★ | **B200 trace 重抓**（active step 1 个 / 自然退出，恢复 6 个 rank 缺失的 GPU 段）→ 多 rank 求平均 baseline 替换 §3 的 rk2+rk3 平均 | [28](./2026-05-07_gptoss_28_b200_baseline_step17.md) |
| ★★ | **MoE expert 负载失衡定位**（rank2 vs rank3 GEMM 差 67 ms / SendRecv 差 120 ms） | [28](./2026-05-07_gptoss_28_b200_baseline_step17.md) |
| ★★ | 写一份**严格 B200↔MI355X comparison note**（同 yaml + 同 patch + 多 rank 平均） | [28](./2026-05-07_gptoss_28_b200_baseline_step17.md) |
| ⛔ | ~~继续追 DDP/HSDP/NCCL 旋钮~~（暴露 comm 仅 46 ms） | [14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md), [15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |

## 命名约定

`YYYY-MM-DD_gptoss_NN_<topic>.md`，`NN` 是**逻辑顺序**而非写作时间。详见 [INDEX](./2026-04-20_gptoss_INDEX.md#命名约定)。

## 维护约定

每写一篇 note → 同时更新本 README 的 **进展时间线** 和 **下一步**；累积变化大时再回写 [INDEX](./2026-04-20_gptoss_INDEX.md) 的故事线。
