# GPT-OSS-20B MLPerf MI355X — 优化项总览表

**周期**: 2026-04-19 → 2026-04-20
**配置**: 1×8 MI355X, fp8 hybrid GEMM + bf16 weight, mbs=2 / gbs=16, tp=pp=ep=1
**baseline**: B0 = 04-19 best E2E run (MLPerf RESULT 10375 s, val 3.3345 ✓)

---

## 1. 累加阶梯对照 (768 iter + 2 次 eval, 同 seed=1234, 同 data)

**累加定义**：B0 是裸基线；B11 = B0 + `grad_reduce_in_bf16`；B10 = B11 + Triton RMSNorm 全覆盖 (MLP 位点 `te.pytorch.RMSNorm` monkey-patch + linear_qkv 位点 `PrimusTurboLayerNormColumnParallelLinear` 替换)。

| # | 配置 (累加) | step (ms, 稳态) | Δ vs B0 | TFLOPs/GPU | eval@384 loss | eval@768 loss | 风险 |
|---:|---|---:|---:|---:|---:|---:|---|
| **B0** | 裸基线 (TE RMSNorm + fp32 grad) | 863.2 | — | 478 | 5.4895 | 4.7258 | — |
| **B11** | + `grad_reduce_in_bf16: true` | **790.1** | **−8.47 %** | **522** | 5.4690 | 4.7012 | 0（loss 略更优） |
| **B10** | + Triton RMSNorm 套件全开 | **776.8** | **−9.99 %** | **531** | 5.4764 | 4.7146 | 0（loss parity） |

**解读**：

- **B0 → B11**: 单 flag `grad_reduce_in_bf16: true` 贡献 **−73.1 ms (−8.5 %)**，eval loss 无劣化（甚至略低），是单位工程量收益最高的项。
- **B11 → B10**: Triton RMSNorm 套件在 bf16-grad 基础上再省 **−13.3 ms (−1.5 %)**——因为 B11 已把 elementwise 大头（fp32 main_grad add）砍掉，RMSNorm 绝对值变小，Triton 实现的相对加速不变但放到更短 step 里看百分比更小。
- **eval loss**: B11/B10 两个优化叠加后 eval@768 与 B0 相差 ≤ 0.012 (PPL ≤ 1.5 %)，在 eval_iters=32 随机起点的固有噪声内，视作 **loss parity**。

数据来源: `/home/xiaompen/mlperf/run_ablation_B14_{B0,B11,B10}.log`, yaml: `gpt_oss_20B-pretrain-fp8-ablation-B14_{B0,B11,B10}.yaml`

---

## 2. 全量端 (E2E MLPerf run, 完整收敛, status=success)

| # | 配置 | 命中 iter | 命中 val_loss | run_duration | Δ vs B0 | tokens/s | 备注 |
|---:|---|---:|---:|---:|---:|---:|---|
| **B0** | 04-19 best run | 11520 | 3.3345 ✓ | 10320.5 s (2h52m00s) | — | ~20572 | 当前 best |
| **B11_full** | Triton RMSNorm + bf16 grad reduce | 12288 | **3.3247 ✓** | **9962.99 s (2h46m03s)** | **−5.96 min (−3.46%)** | ~22982 | margin 0.0153, 16 个 eval Δ ≤ +0.025 |

**为什么 E2E 收益 (−3.46%) 比单 iter 收益 (−10%) 小**：bf16 grad noise 让收敛多跑 +768 iter (+6.7%)，但单 iter 提速更大，净 wall 仍负。

---

## 3. 优化项 ROI 排序

| 排名 | 优化项 | 工程量 | E2E 收益 | mem 收益 | 风险 | 已落地 |
|:---:|---|---|---:|---:|---|:---:|
| 1 | `grad_reduce_in_bf16: true` (1 行 yaml) | 0.5 d | **−8.47 % step / 实测 −3.46 % E2E** | **−36 GB** | 0 (实测, loss parity) | ✅ B11 |
| 2 | Triton RMSNorm 自研 + linear_qkv 全覆盖 | 3 d | **−1.68 % step (B11→B10 边际) / −10 % step (B0→B10 累加)** | 0 | 0 (实测, loss parity) | ✅ B10 |
| 3 | DeepSeek/Qwen3 借鉴 (`precision_aware_optimizer + bf16 master/m/v + experimental`) | 0.5 d | **实测 +1.4 % step 回归 (B12) / +1.9 % (B12.1 仅 bf16 moments)** | **−10~15 GB** | 高 (ROCm 额外 bf16↔fp32 cast 抵消收益) | ❌ 放弃 (无 comm 收益) |
| 4 | DDP `bucket_size` 调优 (B11 切 bf16 后默认 40 M elements) | 0.5 d | **实测 small/large/xlarge 均 +1 ~ +2 % 回归 (B13)** | 0 | 低 | ❌ 放弃 (comm 已 100 % overlap, 无空间) |
| 5 | mbs 2→4 (依赖 #1+#3 释放的 ~50 GB) | 1 d | 估 −5-8% | +30 GB | 中 | ⏳ 待验 |
| 6 | `use_turbo_attention` (B12 候选, 依赖 #1 释放 mem) | 1 d | 估 −2-3% | 中 | 中 | ⏳ 待验 |
| 7 | 自研 fused Triton norm+cast_fp8 | 2-3 d | 仅 ~0.2% step | 0 | 高 | ❌ 放弃 (microbench split 仅 −3.6%, ROI 太差) |
| 8 | 手工融合 SwiGLU+cast | 2 d | 0 (inductor 已融合) | 0 | — | ❌ 不必要 (profile 证伪) |

---

## 4. 累积曲线（端到端 step time, 768 iter 稳态）

```
B0    █████████████████████████████████████████████████████  863.2 ms  (basis,    478 TFLOPs/GPU)
B11   ██████████████████████████████████████████████████      790.1 ms  (−8.47 %,  522 TFLOPs/GPU)   [+ grad_reduce_in_bf16]
B10   █████████████████████████████████████████████████       776.8 ms  (−9.99 %,  531 TFLOPs/GPU)   [+ Triton RMSNorm 套件]
下一步 ████████████████████████████████████████████            ~740 ms   (−14 %,   ~555 TFLOPs/GPU)   [+ MoE permute/unpermute 优化 (估)]
```

```
单 iter:   B0  → B11         −8.47 %   (实测)
单 iter:   B11 → B10         −1.68 %   (实测, Triton RMSNorm 套件边际收益)
单 iter:   B0  → B10         −9.99 %   (实测, 累加)
E2E wall:  B0  → B11_full    −3.46 %  (实测, 含 bf16 多 +768 iter 抵消)
```

---

## 5. 验证矩阵

| 优化项 | profile 验证 | E2E 收敛验证 | 落地配置文件 |
|---|:---:|:---:|---|
| Triton RMSNorm + linear_qkv 路由 (B10) | ✅ | ✅ (随 B11_full 同跑) | `transformer_engine_spec_provider.py` + `triton_rmsnorm.py` + yaml `use_turbo_rms_norm: true` |
| `grad_reduce_in_bf16` (B11) | ✅ | ✅ (B11_full, 16 个 eval) | yaml override `grad_reduce_in_bf16: true` |
| DeepSeek/Qwen3 借鉴 | ✅ B12/B12.1 | ❌ 回归 +1.4 ~ +1.9 % | 放弃, 说明见 [`05_borrow_from_deepseek_v3_config.md`](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| DDP `bucket_size` 调优 | ✅ B13 small/large/xlarge | ❌ 回归 +1 ~ +2 % | 放弃 (comm 已 100 % overlap) |

---

## 6. 关联 notes

| 主题 | 文件 |
|---|---|
| 起点 (B0 解读) | [`01_mlperf_best_e2e_run`](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md) |
| Triton RMSNorm 详解 | [`02_triton_rmsnorm_optimization`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) |
| RMSNorm 原始 trace 数据 | [`03_triton_rmsnorm_report_raw`](./2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md) |
| bf16 grad reduce 详解 | [`04_grad_reduce_bf16_optimization`](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |
| DeepSeek/Qwen3 借鉴推荐 | [`05_borrow_from_deepseek_v3_config`](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| **B11 全量收敛 final 报告** | [`06_B11_full_convergence_report`](./2026-04-20_gptoss_06_B11_full_convergence_report.md) |
| 总览索引 | [`INDEX`](./2026-04-20_gptoss_INDEX.md) |
