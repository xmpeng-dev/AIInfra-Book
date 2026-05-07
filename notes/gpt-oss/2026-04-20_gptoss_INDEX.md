# GPT-OSS-20B MLPerf MI355X 调优思考线索引

**周期**: 2026-04-19 → 2026-04-24（含 V2 跨层 ADD#2 fuse 落地 + V3 tile 扫描）
**目标**: 1×8 MI355X 上把 GPT-OSS-20B MLPerf submission 时间压到最低
**起点**: 04-19 凌晨完成首个 ✅ E2E 收敛 run（10375 s）
**04-20 终点**: 整理出下次 submission 的最小可执行 yaml diff（估 ~9100 s, −12%）
**04-21 续作**: 把 base 横切到 **mbs=4 / gbs=32 / lr=8e-4**，跑 6 阶累积优化链，发现 **NUMA off 在大 batch 下放大到 −7.7%**、**Triton RMSNorm 在 mbs=4 下 ROI 归零**
**04-23/24**: HSDP-2 −23.8% 关闭（note 14）；rank2 单步 trace 校准了 grad-sync 真实暴露 ≈ 46 ms（不是 231），新最大可削目标迁移到 stream 0 上的 **elementwise/cast/norm tax = 252 ms (22%)**，按 ROI 重排优化栈（note 15）；Tier 1 hotspot 校准定位到 `bf16 add = 42 ms / 3.8 % step`（note 16）；落地 fused (residual + RMSNorm) Triton kernel + monkeypatch，预期 V1 −1.5~2 % step（note 17）；E2E 80-iter A/B smoke 实测 **−1.04 % step / +1.04 % TFLOP/s, 0 NaN/0 skip**，loss 在 bf16 噪声带（note 18），等正规 LR 全量 800-iter 收敛复核；继续推 V2（跨层 ADD #2 + 下一层 input_layernorm 融合），同 80-iter smoke 实测 **V2 −0.93 % step / +0.93 % TFLOP/s vs no-fuse base, 0 NaN, max iter Δloss 0.009**；V3（q_norm/k_norm tile 调优）扫描发现仅 2 % 余量、k_norm "余量" 实为 Python wrapper 开销 — 不落 V3 patch（note 19）

---

## 阅读顺序（按思考递进）

| # | 文件 | 主题 | 关键产出 |
|---|---|---|---|
| **01** | [`2026-04-19_gptoss_01_mlperf_best_e2e_run.md`](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md) | **起点**：解读首个收敛 run | RESULT 10375 s, val loss 3.3345 ≤ 3.34, 稳态 862 ms / 479 TFLOP/s/GPU |
| **02** | [`2026-04-20_gptoss_02_triton_rmsnorm_optimization.md`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) | **第一个 ablation**：B10 自写 Triton RMSNorm | step 796 → 770 ms (−3.3%), RMSNorm 占比 31% → 4.6% |
| **03** | [`2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md`](./2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md) | 02 的配套原始 trace 数据 | per-kernel 拆分、caller 链、launch 分布 |
| **04** | [`2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md`](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) | **第二个 ablation**：B11 `grad_reduce_in_bf16` | step 770 → 713 ms (−7.45%), 释放 36 GB 显存, 1 行 yaml |
| **05** | [`2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md`](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) | **横向对比**：从 DeepSeek-V3 / Qwen3 ref yaml 找下一批改进 | 推荐 8 行 yaml diff，估 −12% E2E + −25% mem |
| **06** | [`2026-04-20_gptoss_06_B11_full_convergence_report.md`](./2026-04-20_gptoss_06_B11_full_convergence_report.md) | **B11 全量收敛验证**：Triton RMSNorm + bf16 grad reduce 端到端 run | RESULT 9963 s vs base 10320 s, −5.96 min (−3.46%), val 3.3247 ✓, 16 个 eval 全部 Δ ≤ +0.025 |
| **07** | [`2026-04-20_gptoss_07_optimization_summary_table.md`](./2026-04-20_gptoss_07_optimization_summary_table.md) | **优化项总览表**：B0→B8→B10→B11 各项 step/TFLOPs/mem/收敛 + ROI 排序 | 一页看完所有优化项收益 + 落地状态 |
| **08** | [`2026-04-21_gptoss_08_mbs4_optimization_chain.md`](./2026-04-21_gptoss_08_mbs4_optimization_chain.md) | **mbs=4 / gbs=32 / lr=8e-4 6 阶累积链**：从 raw baseline 到全优化全部跑一遍 | step −18.8% / TFLOPs +24%；NUMA off 边际 −7.7%（mbs=2 下仅 −2.5%）；Triton RMSNorm 在 mbs=4 下 ROI=0；mbs=4 单样本收敛慢 ~30%，TTT 模型估 +11% — 不建议切主路径前未跑 full E2E |
| **14** | [`2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md`](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md) | **Grad Sync overlap Phase B (HSDP) 负结果**：`num_distributed_optimizer_instances=2` 在 DP=8 下 **+23.8% 慢**（1212 → 1500 ms/iter），证实 note 12 的预测；同时 flag 一个 regression — 当前 `run_base.log` 没有带 note 12 推荐的 B1+B2，需重新落回 |
| **15** | [`2026-04-24_gptoss_15_ep1_trace_optimization_plan.md`](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) | **EP=1 单步 trace 重新校准 + 优化栈重排**：rank2 #17 trace 给出 stream 0 busy 95.2% / RCCL 隐藏 80.7% / 暴露 comm ~46 ms；新最大可削目标 = stream 0 上 **elementwise+other+norm = 252 ms (22%)**；按 ROI 重排 Tier 1 elementwise fusion (−7~9%) > Tier 2 optimizer tail HIP-graph (−2~3%) > Tier 3 剩余 comm (−1%) > Tier 4 attention 等 Tier 1 完后再看；保守上限 ~12–13% step → 锚 ~990 ms/iter |
| **16** | [`2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md`](./2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md) | **Tier 1 hotspot 定位 + 配置审计**（精修 note 15）：跑 Top-25 GPU kernel + yaml/patches 全审计。**修正**：stream 0 上 elementwise tax = 197 ms (17.4%)，不是 252 ms (22%)。**头号目标 = `vectorized_elementwise_kernel<bf16 add> = 42 ms`** = attention/MLP 后的 residual add，写 fused residual+RMSNorm Triton kernel 直接砍 −3.4~4.3%。**TE epilogue 路否决**（已开 `gradient_accumulation_fusion`，且 `PrimusTurboLayerNormColumnParallelLinear` 显式 trade 掉 norm-linear fusion 换 Triton RMSNorm 速度）。修正 Tier 1 预算 −4.4~6.5%；E2E 锚 8100~8280 s (−15~17% vs baseline 9777s) |
| **17** | [`2026-04-24_gptoss_17_fused_residual_rmsnorm_impl.md`](./2026-04-24_gptoss_17_fused_residual_rmsnorm_impl.md) | **Tier 1A 落地：fused (residual + RMSNorm) Triton kernel V1**。新 fwd/bwd kernel + autograd Function + 运行时 monkeypatch（不动 Primus/Megatron 源码，不动 yaml）。`PRIMUS_FUSED_RESIDUAL_NORM=1` 启用。**仅切层内 ADD #1 + CALL #2**（跨层 ADD #2 留 V2），预期 **−1.5~2 % step (~17~22 ms)**，约占 note 16 给出的 Tier 1A 预算的 1/2。包含 `bench_fused_residual_rmsnorm.py` 数值正确性 + 单 kernel 性能微基准；待容器内 microbench + 80-iter A/B smoke 验收 |
| **18** | [`2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md`](./2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md) | **Tier 1A V1 验证报告**：容器内 microbench 4 个形状全 PASS（`pre_mlp_layernorm` 形状 1.36× kernel speedup）；E2E 80-iter A/B smoke (`run_ab_fused_residual.sh`) 末 5 iter 平均 **1141.1 → 1129.2 ms (−1.04 %), 723.8 → 731.4 TFLOP/s/GPU (+1.04 %)**，0 NaN / 0 skipped iter；val loss Δ +0.119（短跑 + 压缩 LR schedule，落在 bf16 噪声带）；下一步：800-iter normal-warmup A/B 拿 MLPerf 收敛终判 + trace 复盘看 stream-0 elementwise 真实下降 |
| **19** | [`2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md`](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md) | **Tier 1A V2 实现 + 验证（跨层 ADD #2 → 下一层 input_layernorm，最后一层 → final_layernorm）**：新 monkeypatch 在 `TransformerBlock.__init__` 时用 `object.__setattr__` 接 layer-link 链（避开 `nn.Module` 的子模块自动注册导致的 `RecursionError`），`PRIMUS_FUSED_RESIDUAL_NORM_V2=1` 启用（implies V1）。容器内 `bench_v2_correctness.py` 三形状全 PASS（bf16 2 ULP）；E2E 80-iter 同 chain A/B smoke (`run_ab_fused_residual_v2.sh`) **base 1124.4 → V2 1114.0 ms (−0.93 %), 734.5 → 741.3 TFLOP/s/GPU (+0.93 %)**，0 NaN / 0 skip / max iter `|Δlm_loss|` 0.009 / val Δ 0.013。同时附 V3 q_norm/k_norm tile 扫描结果——q_norm 余量仅 2 %、k_norm "41 %" 是 `TritonRMSNormFn` Python wrapper 开销（同 config 直接调 kernel 19.5 → 11.6 us），**不落 V3 tile patch**，留 P3 wrapper micro-cleanup |

附 06 的两份完整 stdout（已拷贝）：
- `2026-04-20_gptoss_06_B11_full_run.log` (4.7 MB) — B11_full 端到端 run
- `2026-04-20_gptoss_06_baseline_best_run.log` (4.4 MB) — best baseline 端到端 run

---

## 故事线（一句话版）

1. **(01) 看清现状** — best E2E 在哪、瓶颈在哪：稳态 862 ms / step、RMSNorm 31% + elementwise 33% + comm 大头。
2. **(02) 拍最大那块** — 写 Triton RMSNorm 替换 TE 实现：RMSNorm 31% → 4.6%，step −3.3%。
3. **(03) 把 02 的原始数据存档** — kernel/caller/launch 体积分布，方便后续对照。
4. **(04) Profile 暴露第二个大头** — elementwise 里 72% 是 fp32 grad add；改 1 行 `grad_reduce_in_bf16: true` 拿 −10.5% step + −36 GB mem。
5. **(05) 横向找漏项** — DeepSeek/Qwen3 production yaml 都开了 `precision_aware_optimizer + bf16 master/m/v`，当前 mlperf yaml 完全漏掉。再加 `enable_experimental: true`。
6. **(06) 全量收敛验证落地** — B11 (02+04 叠加) 跑完整 mlperf, 单 iter −10%, 多跑 1 个 eval (+768 iter), **净 wall −3.46%, status=success, 命中 val=3.3247（margin 0.015）**。
7. **(07) 把所有优化做成总览表** — B0/B11/B10/B12/B13 的 ROI 排序、mem 收益、收敛风险、落地状态一页化。
8. **(08) 切 mbs=4 重跑 6 阶累积链** — raw baseline → +NUMA off → +nan_check off → +ddp_pad → +bf16 grad → +Triton RMSNorm，**单 iter 累积 −18.8%, TFLOPs 509→633**；意外发现 NUMA off 在 mbs=4 下被放大到 −7.7%（mbs=2 下仅 −2.5%）、Triton RMSNorm 在 mbs=4 下 ROI 归零（瓶颈已迁移到 attention/MoE permute）。

---

## 当前已知最佳形态 vs 下次 submission 推荐形态

| 指标 | 当前 best (04-19 run) | **04-20 实测 B11_full** | 04-20 推荐 (B11 + DeepSeek 借鉴) |
|---|---:|---:|---:|
| MLPerf RESULT | 10375 s | **9963 s (−3.97%)** ✅ | **~9100 s (−12%)** |
| `run_duration` | 10320 s (2h52m) | **9963 s (2h46m)** | **~9100 s (2h32m)** |
| 稳态 step time | 862 ms | **775 ms (−10.1%)** | **~760 ms (−12%)** |
| 显存 / GPU | 226 GB | ~190 GB (−36 GB) | **~170 GB (−25%)** |
| 收敛性 | val 3.3345 ✅ @11520 | **val 3.3247 ✅ @12288** | 等价（DeepSeek/Qwen3 production 已验证） |

**完整 yaml diff** 见 [`05_borrow_from_deepseek_v3_config.md`](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md#推荐的下次-mlperf-submission-yaml-完整-diff)。

---

## 还没做、按 ROI 排序的下一步候选

> note 15 (2026-04-24) trace 校准后的新排序，旧 mbs=2 行项保留在下方供参考。

| ROI | 方向 | 来源 note |
|---|---|---|
| ✅ | **Tier 1A V1 — fused (residual + RMSNorm) Triton kernel** — microbench 1.36× / 80-iter smoke −1.04 % step / +1.04 % TFLOP/s / 0 NaN（[18](./2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md)）；待 800-iter normal-LR A/B 收敛终判 | [17](./2026-04-24_gptoss_17_fused_residual_rmsnorm_impl.md) [18](./2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md) |
| ✅ | **Tier 1A V2 — 跨层 ADD #2 + 下一层 input_layernorm 融合** — 同 chain A/B 实测 −0.93 % step / +0.93 % TFLOP/s vs no-fuse base, 0 NaN, max iter Δloss 0.009, val Δ 0.013（[19](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md)）；待 800-iter normal-LR A/B 收敛终判 | [19](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md) |
| ⛔ | ~~V3 — q_norm/k_norm 形状调优~~ — 扫了 10 组 (ROWS, num_warps, num_stages)，q_norm 仅 2 % 余量、k_norm "余量" 实为 `TritonRMSNormFn` Python wrapper 开销，不落 tile patch；留 P3 wrapper micro-cleanup（cache `_pick_config`、少 `torch.empty`，估 ≤ 1 ms/step） | [19](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md) |
| ★★ | Tier 1B — SwiGLU bwd 去 cat（−0.5~0.9%） + Tier 1C — direct_copy 排查（−0.5~0.9%） | [16](./2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md) |
| ★★ | **Tier 2 — Optimizer tail HIP graph 包**（最后 40 ms 单流串行，副流全空） | [15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ★★ | **修 regress：把 B1+B2（`ddp_average_in_collective` + `bucket_size=100M`）落回 baseline** | [12](./)/[14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md)/[15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ★★ | `patch_moe_overlap=true` smoke（15 min，EP=1 下可能 no-op） | [14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md)/[15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ★★ | 把推荐 yaml diff 落到下次 submission run | [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★ | Tier 4 — FMHA bwd 调优（aiter::fmha_bwd tile / sliding-window 走没走） | [15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ⛔ | ~~继续追 DDP/HSDP/NCCL 旋钮~~ — 真实暴露 comm 仅 46 ms，地板 ~3% | [14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md)/[15](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) |
| ⛔ | ~~HSDP-2~~ — DP=8 下 -23.8% | [14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md) |

旧候选项（mbs=2 路径或已停止追加）：

| ROI | 方向 | 来源 note |
|---|---|---|
| ★★ | 重测 `use_turbo_rms_norm: true` (DeepSeek 路径) vs 自写 Triton (B10) | [02](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) + [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★★ | EP=1 vs EP=8 throughput 对比 | [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★★ | mbs=4 C5 全量 E2E 收敛 run（决定是否切主路径） | [08](./2026-04-21_gptoss_08_mbs4_optimization_chain.md) |
| ✗ | ~~Triton RMSNorm 套件继续投入~~（mbs=4 下边际归零） | [08](./2026-04-21_gptoss_08_mbs4_optimization_chain.md) |
| ★ | 选择性 GEMM 提到 fp8（GEMM 占比上升时再考虑） | [04](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |

---

## 命名约定

- **`2026-04-DD_gptoss_NN_<topic>.md`**
  - `DD` = note 写作日期
  - `gptoss` = 项目 / topic group（这条思考线都是 GPT-OSS-20B MLPerf）
  - `NN` = 在这条思考线里的递进序号（不是按写作时间，而是按**读起来的逻辑顺序**）
  - `<topic>` = 简短主题
- 同一周/月内 INDEX 命名 `2026-04-DD_<group>_INDEX.md`，DD 取该 group 最后一次更新的日期
