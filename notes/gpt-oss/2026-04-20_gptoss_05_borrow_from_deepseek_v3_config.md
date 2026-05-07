# GPT-OSS-20B MLPerf 借鉴 DeepSeek-V3 FP8 调优配置清单

**日期**: 2026-04-20
**承接**: [`2026-04-19_gptoss_01_mlperf_best_e2e_run.md`](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md)
**目标**: 给当前 1×8 MI355X GPT-OSS-20B MLPerf 提交 yaml 找出可直接借鉴的优化项
**对比对象**:
- 主参考: `/home/xiaompen/Primus-dev/examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml`
- 辅参考: `qwen3_30B_A3B-FP8-pretrain.yaml`、`gpt_oss_120B-FP8-pretrain.yaml`、`gpt_oss_20B-FP8-pretrain.yaml` (upstream)
- 当前: `/home/xiaompen/mlperf/gpt_oss_20B-pretrain-fp8.yaml`

---

## TL;DR — 推荐 yaml diff（5 项 8 行，零代码改动）

```yaml
# === 借鉴 1: B11 (昨天 ablation 已验证 −10% step) ===
grad_reduce_in_bf16: true

# === 借鉴 2: DeepSeek-V3 / Qwen3-30B 的 precision-aware DistOpt ===
# 把 master grad + Adam m/v 全部从 fp32 降到 bf16
# 估再省 ~16 GB optimizer state / GPU
use_precision_aware_optimizer: true
main_grads_dtype: bf16
exp_avg_dtype: bf16
exp_avg_sq_dtype: bf16

# === 借鉴 3: 解锁 experimental code path ===
# DeepSeek / Qwen3 ref 都开了，包含 fused RoPE / attention 的若干优化分支
enable_experimental: true
```

**叠加预估**:
| 指标 | 当前 best (E2E) | 推荐 diff 之后 |
|---|---:|---:|
| step time | 862 ms | **~760 ms** (−12%) |
| `run_duration` | 10320 s (2h52m) | **~9100 s (2h32m)** |
| 显存 / GPU | 226 GB | **~170 GB** (−25%) |
| 收敛风险 | — | **无新增**（DeepSeek + Qwen3 production 已验证） |

---

## 关键差异速览

| 配置项 | 当前 mlperf yaml | DeepSeek-V3 FP8 ref | Qwen3-30B-A3B ref | 借鉴优先级 |
|---|---|---|---|---|
| `grad_reduce_in_bf16` | false | (未显式设) | (未显式设) | ★★★（B11 已验证） |
| `use_precision_aware_optimizer` | (默认 false) | **true** | **true** | ★★★ |
| `main_grads_dtype` | torch.float32 | **bf16** | **bf16** | ★★★ |
| `exp_avg_dtype` | torch.float32 | **bf16** | **bf16** | ★★★ |
| `exp_avg_sq_dtype` | torch.float32 | **bf16** | **bf16** | ★★★ |
| `enable_experimental` | false | **true** | **true** | ★★ |
| `use_turbo_rms_norm` | false（yaml 注释里禁掉） | **true** | false（注 "bug"） | ★★（需复测） |
| `use_turbo_deepep` | false | **true** | **true** | ★★（仅 EP=8 时） |
| `turbo_sync_free_moe_stage` | 0 | 1 | 1 | ★★（仅 EP=8 时） |
| `turbo_deepep_num_cu` | 64 | 80 | 80 | ★（EP=8 用 80） |
| `moe_router_dtype` | fp32 ✅ | fp32 ✅ | fp32 ✅ | 已对齐 |
| `recompute_granularity` | None | full + `recompute_layer_ids` | full + block(5) | ★（仅 mbs↑ 时） |
| `pipeline_model_parallel_layout` | (PP=1, N/A) | 自定义首末 1 层 | (PP=1, N/A) | N/A |
| `multi_latent_attention` | false ✅ | true (DeepSeek 专用) | false ✅ | N/A |

---

## ★★★ 强烈建议借鉴 — `Precision-Aware Optimizer`（最大漏项）

DeepSeek-V3 FP8 ref 和 Qwen3-30B-A3B FP8 ref 都默认开了，**当前 mlperf yaml 完全没有**：

```yaml
use_precision_aware_optimizer: true
main_grads_dtype: bf16
exp_avg_dtype: bf16
exp_avg_sq_dtype: bf16
```

### 与 B11 的关系（互补，不冲突）

| | B11 (`grad_reduce_in_bf16`) | DeepSeek ref (`precision_aware_optimizer`) |
|---|---|---|
| 改动位置 | DDP grad bucket / reduce-scatter | DistOpt 内部 master state |
| 数据类型 | grad accumulation + wire 降 bf16 | master grad + Adam m / v 降 bf16 |
| 主要收益 | step time −10%（comm + elementwise 一起降） | optimizer state mem −50% |
| 是否互斥 | **完全兼容，建议一起开** | |

### 显存账（GPT-OSS-20B 21B 参数总量）

| state | fp32 (当前) | bf16 (借鉴后) | 节省 |
|---|---:|---:|---:|
| master grad | 84 GB | 42 GB | 42 GB |
| Adam m | 84 GB | 42 GB | 42 GB |
| Adam v | 84 GB | 42 GB | 42 GB |
| **合计** | **252 GB** | **126 GB** | **126 GB（全集群，÷8 GPU = 16 GB/GPU）** |

注：以上是全集群账，因为 `use_distributed_optimizer: true`，optimizer state 已经 shard 过；每 GPU 实际省的就是 ~16 GB。

### 收敛性

- DeepSeek-V3、Qwen3-30B-A3B、Qwen3-235B-A22B 的 Primus production yaml 全都默认开
- Megatron-LM 上游已稳定支持，与 `use_distributed_optimizer: true` 兼容（已开 ✅）
- DistOpt 在 step 时仍有 fp32 master copy 做精确 cast，业界标准的 bf16 + DistOpt 配方

### 预期收益

- 显存: 226 → ~170 GB / GPU（−25%）
- optimizer step 时间: ~5–10 ms 小幅下降（mem-bandwidth bound 的 Adam update 体积砍半）
- step time: 主要靠 B11 拿（−10%），这个再 −1% 上下

---

## ★★ 中等优先级

### `enable_experimental: true` + `apply_rope_fusion: true`

```yaml
enable_experimental: true   # 当前 false，要加
apply_rope_fusion: true     # 当前已 true ✅
```

`enable_experimental: true` 在 Megatron 里解锁一组 experimental code paths（包括 fused RoPE 的某些后端、某些 attention 优化路径）。DeepSeek / Qwen3 都默认开，无负面报告。GPT-OSS-20B 也用 RoPE，没理由不开。

**风险**：低。如果实测 regression 直接关掉。
**预期收益**：小幅 attention / RoPE 路径加速，5–15 ms / step 量级，需 trace 确认。

### `use_turbo_rms_norm: true`（带条件，需复测）

DeepSeek-V3 ref **直接开了** ✅；Qwen3-30B ref 注释 "bug" 关掉了。

当前 mlperf yaml 注释里写：
> "even after fixing the DDP double-Param bug + contig wrap … enabling this still regressed B0 baseline ~22% (1018 ms/iter, 404 TFLOP/s vs 795 ms, 519 TFLOP/s). The fused RMSNorm absorbs scheduling slack used to overlap NCCL grad-reduce on this stack."

但昨天 B10 ablation（参考 [`2026-04-20_gptoss_02_triton_rmsnorm_optimization.md`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md)）用**自写的 Triton RMSNorm**反而 +3.3%。两者实现路径不同：
- `use_turbo_rms_norm`: Primus 内置的 turbo 版（DeepSeek 上正收益）
- B10 Triton RMSNorm: 我们自己的实现（GPT-OSS 上正收益）

**建议动作**：在 1×8 fp8 hybrid + B11 + `precision_aware_optimizer` 路径下重新 ablate `use_turbo_rms_norm: true`，看 DeepSeek 的修复 + B11 释放的 slack 是不是已经把 GPT-OSS 路径上的 regression 解掉了。
- 如果还是 regression → 保留我们自己的 B10 Triton 版本
- 如果已经 +3% 以上 → 去掉自写 Triton 版本，用上游 turbo（更易维护）

### DeepEP / Sync-Free MoE — 仅当 EP=8 才适用

DeepSeek / Qwen3 调优 yaml 都开了：

```yaml
use_turbo_deepep: true
turbo_sync_free_moe_stage: 1
turbo_deepep_num_cu: 80     # ep8 推荐 80, ep16-64 推荐 32
moe_shared_expert_overlap: false
moe_router_dtype: fp32      # DeepEP only supports fp32 probs (已对齐 ✅)
```

**关键判断点**：当前实际跑的 EP 数是多少？
- yaml 默认 `expert_model_parallel_size: 8`
- 但 `config_MI355X_1x8x1_tp1pp1ep1_gbs16_fp8.sh` 里 `export PRIMUS_EP=1` **覆盖**了 yaml
- log 里 `world_size=8 / TP=1 / PP=1`，配 EP=1 → 实际 **8 路 DP，无 expert parallel**

判断：
- 当前 EP=1 → DeepEP 没有意义（没有 alltoall expert routing 的开销可降）
- 如果改成 EP=8（每个 expert 在不同 GPU 上） → DeepEP 才值得开

**建议动作**：
1. 先确认 mlperf 提交规则下用 EP=1 还是 EP=8 更好（需要再做一次 EP=1 vs EP=8 的 throughput 对比）
2. 如果决定走 EP=8，再开 DeepEP 这一组

### `recompute_granularity` — 仅当 mbs 提升后才考虑

DeepSeek-V3 用 `recompute_granularity: full` + 自定义 `recompute_layer_ids`（21 层选择性重算），Qwen3-30B 用 `recompute_method: block, recompute_num_layers: 5`，gpt_oss_120B 用 `block, num=4`。

当前 mlperf 路径：
- mbs=2，显存 226/288 GiB，**还有 60+ GB 余量**
- B11 + precision_aware optimizer 之后再省 ~50 GB → 余量 ~110 GB
- 此时加 recompute 纯减速，**不建议开**

**唯一的应用场景**：如果决定把 mbs 从 2 提到 4（gbs 同步 32 或保持 16 调 grad_accum），到时候显存可能不够，再考虑加 `recompute_granularity: full + recompute_layer_ids: "..."` 选择性重算几个最重的层。

---

## 不建议借鉴

| 项 | DeepSeek 是否用 | 不借鉴原因 |
|---|---|---|
| `pipeline_model_parallel_layout` | ✅ (PP=16) | 我们 PP=1，无关 |
| `multi_latent_attention: true` | ✅ | DeepSeek 架构专用，GPT-OSS 用普通 GQA |
| `mock_data: true` | ✅ (perf benchmark only) | MLPerf 提交必须 false（已 false ✅） |
| `pipeline_model_parallel_size: 16` | ✅ | 我们 1 节点 8 GPU 不需要 PP |
| `recompute_layer_ids` 显式列表 | ✅ | 当前显存够，不需要 |

---

## 推荐执行顺序

| 步骤 | 改动 | 预期 step time | 预期 mem/GPU | 风险 |
|---|---|---:|---:|---|
| baseline (E2E best) | — | 862 ms | 226 GB | 已知好 |
| **+ B11** | `grad_reduce_in_bf16: true` | ~770 ms (−11%) | ~190 GB | 已 ablation 验证 |
| **+ DeepSeek precision-aware optimizer** | 4 行 yaml | ~760 ms (−1% 额外) | **~170 GB** | DeepSeek/Qwen3 production 已验证 |
| **+ enable_experimental** | 1 行 yaml | ~755 ms (−0.5% 额外) | ~170 GB | 低（实测有问题就关） |
| **重测 use_turbo_rms_norm** | ablation only | TBD | TBD | 中（先单独 ablation，不直接进 submission） |
| EP=1 vs EP=8 sweep | env 改 | TBD | TBD | 中（需要 throughput 对比） |
| mbs 2→4 + 选择性 recompute | 多行 | TBD | TBD | 高（要重新 sweep LR）|

---

## 推荐的下次 MLPerf submission yaml 完整 diff

只在 `gpt_oss_20B-pretrain-fp8.yaml` 的 `overrides:` 块里加这 8 行（其他不变）：

```diff
       # mixed-precision
       attention_softmax_in_fp32: false
+
+      # === Borrowed from DeepSeek-V3 / Qwen3-30B FP8 reference (2026-04-20) ===
+      # B11 ablation verified: −10% step time, −36 GB mem
+      grad_reduce_in_bf16: true
+
+      # DeepSeek/Qwen3 production default: bf16 master grad + Adam m/v
+      # Estimated: −16 GB optimizer state / GPU, small optimizer step speedup
+      use_precision_aware_optimizer: true
+      main_grads_dtype: bf16
+      exp_avg_dtype: bf16
+      exp_avg_sq_dtype: bf16
+
+      # Unlock experimental code paths (fused RoPE / attention variants)
+      enable_experimental: true
```

**预期 MLPerf submission 结果**:
- `RESULT,GPT_OSS_20B,,~9100,AMD` (vs current 10375)
- `run_duration` ~9100 s ≈ **2h 32min**（vs current 2h 52min）
- 0 nan / 0 skip 维持
- val loss ≤ 3.34 维持（数值稳定性等价于 bf16 + DistOpt 业界标准配方）

---

## 参考

- 当前最佳 E2E run 总结: [`2026-04-19_gptoss_01_mlperf_best_e2e_run.md`](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md)
- B11 (`grad_reduce_in_bf16`) ablation: [`2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md`](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md)
- B10 (Triton RMSNorm) ablation: [`2026-04-20_gptoss_02_triton_rmsnorm_optimization.md`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md)
- DeepSeek-V3 FP8 ref yaml: `/home/xiaompen/Primus-dev/examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml`
- Qwen3-30B-A3B FP8 ref yaml: `/home/xiaompen/Primus-dev/examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml`
- 当前 mlperf yaml: `/home/xiaompen/mlperf/gpt_oss_20B-pretrain-fp8.yaml`
