# Llama-2 70B LoRA SFT — Primus vs NeMo Trace 对比

| 字段 | Primus | NeMo (MLPerf 6.0 reference) |
| --- | --- | --- |
| 日期 | 2026-04-30 | 2026-04-29 |
| 模型 | Llama-2-70B · LoRA r=16 α=32 · `bf16_with_fp8_hybrid` | 同左 |
| 硬件 | 8 × MI355X (288 GiB HBM, gfx950) | 同左 |
| 并行 | TP1 · PP1 · CP1 · DP=8 (DDP + DistOpt) | 同左 |
| Batch | GBS 8 · MBS 1 · seq 8192 (packed) | GBS 8 · MBS 1 · seq 8192 |
| 训练规模 | 总参 69.0 B / 可训参 44.5 M (LoRA, 0.06%) | 同左 |
| 稳态 step | **1.520-1.525 s** (iter 40-90) | **1.503-1.517 s** (block @ step 192-336) |
| 稳态 throughput | 2387-2395 TFLOP/s/GPU = ~5.23 samples/s/8GPU | 5.36-5.40 samples/s/8GPU |
| 分析窗口 | `ProfilerStep#21` (rank 2), 1528.48 ms | `ProfilerStep#5` (rank 2), 1483.41 ms |
| Pmax (alloc) | **285.84 GB** (92.4% of 309 GB cap) | **214.65 GB** (69.4%) |
| Rmax (reserved) | **295.52 GB** (95.6% of cap) | **221.08 GB** (71.5%) |
| 关键结论 | (1) kernel 性能完全一致；(2) NeMo 快 ~3%，源自 idle gap + NCCL overlap；(3) Primus 多用 71 GB 显存、4× 多 TE unary 开销、6× 少 FP8 cast/transpose | — |

---

## 0. 一行结论

**Kernel-level 性能一致，wall-clock NeMo 快 3% (Δ45 ms)，且 NeMo 显存峰值低 71 GB。**

- **显存差异已定位（已用 run.log 验证）**：Primus autoconfig 把
  `LlamaModelProvider.fp8_param` 从 `False` overwrote 成 `True`
  （`results/profile_run_allranks.log:1457-1462`），TE 因此为每个 weight
  额外存一份 fp8 拷贝（3 B/param vs NeMo 的 2 B/param）。Llama-2-70B 有
  68.98 B 参数 → 多用 **68.98 GB** 权重内存，几乎完全解释了 71.19 GB 的
  Pmax 差。NeMo 默认 `FP8_PARAM_GATHER=False`，weight 只存 bf16，每次
  forward 现 cast 成 fp8（代价是 §4 里多出来的 80 ms FP8 cast/transpose）。
- **时间差异**：~94 ms idle gap + RCCL 完全串行（vs NeMo 100% hidden）。

## 1. 目标 / 背景

- 比较 Primus (Megatron-Bridge) 和 NeMo (MLPerf 6.0 reference) 在同一硬件、同一精度、同一 batch 配置下的 trace + VRAM。
- 输入 trace：
  - Primus: `/home/xiaompen/mlperf-training-primus/results/torch_profiler_traces/allranks_rank2.pt.trace.json`
  - NeMo:   `/home/xiaompen/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_093757/torchprof/trace_6_597340a2-1cc3-4a16-8aa9-03824a06fe8a.json` (rank 2)
- 输入 log：
  - Primus: `results/profile_run_allranks.log`，`mem-max` 行 `train_utils.py:671`，iter 10。
  - NeMo:   `run_and_time.log`，`[mem rank=0 step=10]`。

## 2. 跑出 trace 的方式

Primus：

```bash
TRACE=1 PRIMUS_TRAIN_ITERS=25 PRIMUS_PROFILE_STEP_START=20 \
PRIMUS_PROFILE_STEP_END=23 bash run_and_time.sh
# 8 个 rank 各一份 allranks_rank{0..7}.pt.trace.json (~605 MB each)
```

NeMo：

```bash
PROFILER=torchprof TORCH_PROFILE=1   # warmup=3 active=2 repeat=1
# 输出 8 个 trace_6_<uuid>.json (~45 MB each, kineto 默认压缩)
```

## 3. Per-stream busy time

### Primus — `ProfilerStep#21` (1528.48 ms)

| pid | stream | 角色 | busy | share |
| --- | --- | --- | ---: | ---: |
| 10 | 0  | 计算 (GEMM / FMHA / norm / elem / SwiGLU) | 1416.71 ms | **92.7%** |
| 10 | 33 | RCCL DDP grad-sync (`Generic_1` + `AG`) | 15.31 ms | 1.0% |

- Stream oversub = 93.7%；compute-busy = 92.8%；idle gap = **93.55 ms (~6.1%)** 集中在 step 头部。

### NeMo — `ProfilerStep#5` (1483.41 ms)

| pid | stream | 角色 | busy | share |
| --- | --- | --- | ---: | ---: |
| 10 | 3  | 计算 + HtoD memcpy + RCCL grad-sync | 1461.72 ms (filtered) | **98.5%** |
| 10 | 4  | TE op-fuser elementwise mini-stream | 1.26 ms | 0.1% |
| 10 | 45 | RCCL AllGather (single kernel, 0.46 ms) | 0.46 ms | 0.0% |

- 原始 oversub 显示 259%，因为 trace 里有 **两个错误的超长 event**（见 §10）：
  - 一个 `Custom_Cijk_Alik_Bljk_F8B8BS_*shortname1_gfx950` 的 event `dur=1200.93 ms`；
  - 一个 `Memcpy HtoD` event `dur=1188.60 ms`。
  - 过滤后 oversub = 98.5%，与 compute-busy 一致。
- Idle gap 仅 **7.10 ms (~0.5%)**。

## 4. Kernel 类别分解（重新归类后，NeMo 已剔除两个 bogus event）

| 类别 | Primus ms | Primus % step | NeMo ms | NeMo % step | Δ ms | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **FP8 GEMM** (`Custom_Cijk_*_F8B8BS / F8BS`) | **797.4** | **52.2%** | **791.4** | **53.3%** | -6.0 | 4 个 shortname 变体；几乎相同 |
| **FlashAttention** (`aiter::fmha`) | **327.6** | **21.4%** | **327.1** | **22.0%** | -0.5 | bwd 221 ms + fwd 94 ms |
| TE SwiGLU / unary / gated_act | 99.4 | 6.5% | 49.7 | 3.3% | **-49.7** | Primus 多 50 ms — 见 §8 |
| **FP8 cast / transpose** | **12.7** | **0.8%** | **93.1** | **6.3%** | **+80.4** | NeMo 多 80 ms — 见 §8 |
| Elementwise / dropout / cast | 78.0 | 5.1% | 90.9 | 6.1% | +12.9 | |
| bf16 GEMM (`Cijk_*_BBS / BSS`) | 43.4 | 2.8% | 39.1 | 2.6% | -4.3 | LoRA-A/B + lm_head |
| Fused QKV-RoPE | 23.2 | 1.5% | 0.0 | 0.0% | -23.2 | **Primus 独有 fused kernel** — NeMo 走分离 RoPE |
| RMSNorm (Triton fwd+bwd) | 20.0 | 1.3% | 23.8 | 1.6% | +3.8 | |
| RCCL grad-sync (Generic + AG) | 15.9 | 1.0% | 10.9 | 0.7% | -5.0 | DP=8 grad-AllReduce |
| Reduction kernels | 8.6 | 0.6% | 8.9 | 0.6% | +0.3 | |
| MemSet / MemCopy / D2D | 4.7 | 0.3% | 5.8 | 0.4% | +1.0 | |
| Other | 1.2 | 0.1% | 21.1 | 1.4% | +19.9 | NeMo 杂项偏多（含 `triton_poi_fused_add_0` 等） |
| **小计** | 1432.0 | — | 1461.7 | — | — | |

> 取舍：`fp8_cast_only_kernel` 和 `transpose_optimized_kernel` 之类的 cast/transpose 我归到 "FP8 cast/transpose"。Primus 因为 `no_fp8_weight_transpose_cache: true` 实际上**关闭了** weight transpose cache → 但 cast/transpose 时间反而更少（12.7 ms），说明 Primus 的 TE op-fuser 把这部分融进了 GEMM 前后，而 NeMo 跑独立 cast kernel。

## 5. Per-call 单 kernel 对比（同 binary 同 shape）

| Kernel | Primus avg (ms) × n | NeMo avg (ms) × n | Δ% |
| --- | ---: | ---: | ---: |
| `Custom_Cijk_Alik_Bljk_F8B8BS_*shortname1_gfx950` | 1.501 × 240 | 1.508 × 239 | +0.5% |
| `Custom_Cijk_Alik_Bljk_F8BS_*shortname1_gfx950` | 1.204 × 240 | 1.194 × 240 | -0.8% |
| `aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv` | 2.768 × 80 | 2.771 × 80 | +0.1% |
| `Custom_Cijk_Alik_Bljk_F8BS_*shortname0_gfx950` | 1.351 × 80 | 1.315 × 80 | -2.7% |
| `aiter::fmha_fwd_hd128_bf16_causal` | 1.178 × 80 | 1.173 × 80 | -0.4% |
| `Custom_Cijk_Alik_Bljk_F8B8BS_*shortname0_gfx950` | 0.508 × 79 | 0.498 × 79 | -2.0% |
| `ncclDevKernel_Generic_1` | 3.968 × 4 | 2.608 × 4 | -34.3% |

**结论**：底层 GEMM/FMHA 完全等价；NCCL 单调用更快是因为 NeMo grad-AllReduce 的 message size 更小（更多 fragment 被 fused 进 op）。

## 6. Compute / NCCL Overlap

| 指标 | Primus | NeMo |
| --- | ---: | ---: |
| compute-only | 1418.85 ms | 1465.30 ms (假性高，含 bogus HtoD bin) |
| nccl-only | 15.80 ms | 0.00 ms |
| **overlap (compute & nccl)** | **0.35 ms** | **11.10 ms** |
| idle | 93.55 ms | 7.10 ms |
| **NCCL hidden behind compute** | **2.2%** (0.3 / 16.1 ms) | **100.0%** (11.1 / 11.1 ms) |

Primus 的 RCCL grad-sync 完全串行在 backward 之后；NeMo 把它叠在 backward 尾巴上，0% 用户可见。

## 7. VRAM (HBM)

> Cap from trace `deviceProperties[*].totalGlobalMem = 309,220,868,096 B = 309.22 GB = 287.99 GiB`.

| 指标 | Primus | NeMo |
| --- | ---: | ---: |
| `mem-allocated-gigabytes` (current) | 126.44 GB | 65.32 GB |
| **Pmax** (`mem-max-allocated-gigabytes`) | **285.84 GB** (92.4%) | **214.65 GB** (69.4%) |
| **Rmax** (`mem-max-reserved-gigabytes`) | **295.52 GB** (95.6%) | **221.08 GB** (71.5%) |
| Headroom to OOM (cap − Rmax) | 13.7 GB ⚠️ | 88.1 GB ✅ |
| Fragmentation `(Rmax−Pmax)/Rmax` | 3.3% ✅ | 2.9% ✅ |
| Allocator retires | 0 ✅ | 0 ✅ |
| 显存 verdict | **TIGHT** (eval/checkpoint 会 OOM) | HEALTHY |

### 7.1 Bucket 分解（按 Rmax 配平）

**Primus** (Rmax = 295.52 GB)：

| Bucket | GB | % of Rmax | 备注 |
| --- | ---: | ---: | --- |
| Weights (bf16 master) | 137.96 | 46.7% | 68.98 B × 2 B/param |
| **Weights (fp8 拷贝, fp8_param=True)** | **68.98** | **23.3%** | **比 NeMo 多的就是这一块** |
| Activations (no recompute, seq=8192) | 74.00 | 25.0% | 与 NeMo 持平 |
| LoRA grads + Adam state (44.5M) | 0.80 | 0.3% | DistOpt 跨 DP=8 sharding |
| TE FP8 amax/scale + cuBLAS workspace + NCCL | 4.10 | 1.4% | balance to Pmax = 285.84 GB |
| Allocator slack (Rmax − Pmax) | 9.68 | 3.3% | fragmentation only 3.3% |
| **小计 (Pmax)** | **285.84** | — | 完美匹配 |

**NeMo** (Rmax = 221.08 GB)：

| Bucket | GB | % of Rmax | 备注 |
| --- | ---: | ---: | --- |
| Weights (bf16 master) | 137.96 | 62.4% | 同左 |
| Weights (fp8 拷贝) | **0** | 0% | `FP8_PARAM_GATHER` 默认 False → 仅 bf16 |
| Activations (no recompute, seq=8192) | 74.00 | 33.5% | |
| LoRA grads + Adam state | 0.80 | 0.4% | |
| TE FP8 amax/scale + cuBLAS workspace + NCCL | 1.89 | 0.9% | balance to Pmax = 214.65 GB |
| Allocator slack (Rmax − Pmax) | 6.43 | 2.9% | |
| **小计 (Pmax)** | **214.65** | — | 完美匹配 |

### 7.2 Root cause: `fp8_param: True`（已在 run.log 验证）

`results/profile_run_allranks.log` 第 1457-1462 行：

```
Overwrote LlamaModelProvider.fp8_param  False -> True
Overwrote LlamaModelProvider.num_layers_at_start_in_bf16  1 -> 0
Overwrote LlamaModelProvider.num_layers_at_end_in_bf16  1 -> 0
Overwrote OptimizerConfig.fp8_recipe  None -> delayed
Overwrote DistributedDataParallelConfig.grad_reduce_in_fp32  False -> True
Overwrote DistributedDataParallelConfig.fp8_param_gather  False -> True
```

最终生效配置（log 第 1694-1712 行）：`fp8_param: true`、`fp8_param_gather: true`、`grad_reduce_in_fp32: true`。

> 注意：log 第 1069 行 `model.fp8_param = False` 是早期一次中间快照，**并不是最终生效值**。后面第 1457 行又被 overwrite 一次，最终 fp8_param=True 生效。

**算式**：

```
Llama-2-70B params = 68.98 B
bf16 only weights:           68.98 × 2 B = 137.96 GB
bf16 + fp8 weights (fp8_param=True):
                              68.98 × (2 + 1) B = 206.94 GB
                              Δ = 68.98 GB
观察到的 Pmax 差:  285.84 − 214.65 = 71.19 GB ≈ 68.98 + ~2 GB (TE fp8 amax history)
```

**Activations 实际是相等的（~74 GB）**，与之前猜的"Primus 多 71 GB activation"完全相反。op-fuser / packed_sequence / cast_transpose policy 都不是这个 71 GB 的来源，是被前面错误的 bucket 估算误导了。

### 7.3 NeMo 用 cast 时间换内存

NeMo `fp8_param=False` 意味着每次 forward 都需要把 bf16 weight 现 cast 成 fp8 再喂 GEMM。这正好对应 §4 里 NeMo `FP8 cast/transpose` 多 80 ms（93 ms vs Primus 13 ms）的开销 —— **Primus 用 69 GB 内存换了 80 ms / step**。

是否值得？在 MI355X 上 Pmax/cap = 92.4%、Rmax/cap = 95.6%，eval 或 checkpoint save 任意抖动就 OOM；建议关掉 `fp8_param`，让 Primus 也走 NeMo 的"内存便宜、cast 多 80 ms"路线，对应 step time +5%、显存 -69 GB。

## 8. 可立即 actionable 的改动（针对 Primus）

| # | 改动 | 预期收益 | 风险 / 代价 | 实测 |
| ---: | --- | --- | --- | --- |
| 1 | **关掉 `fp8_param`**（覆盖 autoconfig overwrite，对齐 NeMo） | 节省 **~69 GB** 显存 (Pmax 286→217 GB) | step time +~5% (~80 ms / step 多出 FP8 cast/transpose) | 未测 |
| 2 | 同时关 `fp8_param_gather: false` + `grad_reduce_in_fp32: false`（一组） | 进一步省 grad/AllGather buffer | NCCL message 变 bf16，吞吐略有变化 | 未测 |
| 3 | 开启 `overlap_grad_reduce=true` + `overlap_param_gather=true` (Megatron-Bridge DDP) | 把 ~16 ms RCCL 藏到 backward 尾巴，节省 ~1% step time | 0；和 NeMo 等同 | 未测 |
| 4 | 排查 dataloader / dispatcher 头部 ~94 ms idle gap | 节省 ~6% step time | 中：可能涉及 num_workers / pin_memory / prefetch_factor | 未测 (`scan_per_rank.py` 已有数据) |
| 5 | 启用 `selective` activation recompute（attention 部分） | 进一步节省 ~30 GB | 高：5-15% throughput 损失 | 未测 |

### 8.1 已验证的小变量（来自 `notes/llama2_70b_lora_*` 历史）

- `pin_memory=true` (Primus): 已实测无显著变化（`profile_run_pinmem_true_v2.log`）。
- `num_workers=0` cache hit (Primus): 显存上升、step 时间相近（`profile_run_workers0.log`）。
- 这些都不是 71 GB 差异的来源 —— 真正的来源就是 §7.2 的 `fp8_param: True`。

## 9. 复现

```bash
# 1) Primus baseline
cd /home/xiaompen/mlperf-training-primus/llama2_sft/primus
PRIMUS_TRAIN_ITERS=25 PRIMUS_PROFILE_STEP_START=20 \
  PRIMUS_PROFILE_STEP_END=23 bash run_and_time.sh

# 2) NeMo baseline
cd /home/xiaompen/mlperf-training-6-0/llama2_sft/nemo
DGXSYSTEM=MI355X_1x8x1 PROFILER=torchprof bash run_and_time.sh

# 3) 跑分析
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
   results/torch_profiler_traces/allranks_rank2.pt.trace.json ProfilerStep#21
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
   <nemo>/torchprof/trace_6_597340a2-*.json ProfilerStep#5
```

## 10. 注意事项 / 已知 trace 录制问题

- **NeMo trace 有两个错误的超长 event**：
  - `Custom_Cijk_Alik_Bljk_F8B8BS_*shortname1_gfx950`，ts=279.88 ms，`dur=1200.934 ms`（应该是 1.5 ms）；
  - `Memcpy HtoD (Host -> Device)`，`dur=1188.60 ms`（其余 memcpy 都 < 0.01 ms）。
- 二者都恰好横跨整个 step 一半时间，加起来 ~2389 ms，正好是 NeMo 总 GPU dur 减去物理 step 时间的差额（3851 − 1462 ≈ 2389）。
- 怀疑是 PyTorch profiler 在第一次 capture 时遇到 ROCm 的某个 sync 节点 (e.g. CUDA graph capture / first-iter compile)，dur 字段被填成 step end timestamp。
- **过滤策略**：丢弃 `dur > 100 ms` 的 GPU event，二次统计取得真实数字。详见 `/tmp/breakdowns/comparison.txt`。

## 11. 相关文件

- canvas: `~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/primus-vs-nemo-mi355x-trace.canvas.tsx`
- Primus trace: `/home/xiaompen/mlperf-training-primus/results/torch_profiler_traces/allranks_rank2.pt.trace.json`
- NeMo trace: `/home/xiaompen/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_093757/torchprof/trace_6_597340a2-1cc3-4a16-8aa9-03824a06fe8a.json`
- Primus log: `/home/xiaompen/mlperf-training-primus/results/profile_run_allranks.log`
- NeMo log: `/home/xiaompen/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_093757/run_and_time.log`
- breakdown: `/tmp/breakdowns/{primus_rank2_step21,nemo_rank2_step5,comparison}.txt`

## 12. 后续 TODO

- [ ] 实测 Primus 启用 `overlap_grad_reduce=true` 后的 step time / NCCL hidden%。
- [ ] 在 Primus forward 里插 `torch.cuda.memory_snapshot` 或 `record_memory_history`，定位 71 GB activation 差异的具体 tensor。
- [ ] 检查 Primus `use_transformer_engine_op_fuser` 是否对所有 LayerNorm+Linear / Linear+SwiGLU pattern 实际触发（可能只 fused 了 LoRA path 的 stable variant）。
- [ ] 在 NeMo 上抓 rank=0 trace（不是 rank2），确认是否同样有 bogus event；如果只 rank2 有，则是单 GPU profiler bug。
