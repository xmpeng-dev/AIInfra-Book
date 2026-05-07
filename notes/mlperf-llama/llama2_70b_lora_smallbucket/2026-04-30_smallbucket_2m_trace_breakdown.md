# Llama-2-70B LoRA SFT — `ddp.bucket_size = 2_000_000` Trace 分析

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-04-30 |
| 模型 | Llama-2-70B · LoRA SFT · `bf16_with_fp8_hybrid` (LoRA r=16 α=32) |
| 硬件 | 8 × MI355X (288 GiB HBM, gfx950) |
| 并行 | TP1 · PP1 · CP1 · EP1 · DP=8（纯 DDP + DistOpt） |
| Batch | GBS 8 · MBS 1 · seq 8192 (packed) |
| 训练规模 | 总参 69.0 B / 可训参 44.5 M (LoRA, 0.06%) |
| 实测 | iter-20 step **1.526 s** · **2385.8 TFLOP/s/GPU** · 25-iter wall 220 s |
| 分析窗口 | `ProfilerStep#21` (steady state), 1528.47 ms |
| 关键结论 | (1) bucket_size override 对 LoRA + DistOpt **无效**——RCCL 仍只发 2 个 kernel；(2) FP8 GEMM 50.5%、FMHA 20.9% → compute-bound 健康；(3) 唯一明显 idle 是 head-of-step 75 ms loader/launch；(4) **VRAM 95.6% reserved（TIGHT）** —— 与 baseline 等同。|

---

## 1. 目标 / 背景

- **Motivation**：上一轮 baseline (`num_workers0_cachehit`) 的 trace 显示 RCCL grad-sync 整段 65.8 ms 全堆在 step 末尾 1450–1525 ms，0% overlap。Hypothesis：把 Megatron 默认 `bucket_size=134_217_728` 调小成 `2_000_000`，让 grad bucket 切成 ~22 份，backward 早期就能 fire，让通信藏到计算里。
- **预期收益**：~75 ms / step (≈5%)，从 1525 → 1450 ms，TFLOP/s 2391 → ~2515。
- **真实结果**：step time 1522.7 → 1526.2 ms（噪声内）；TFLOP 2391 → 2386。**完全没改善**。

## 2. 跑 baseline + 抓 trace

代码改动（`llama2_sft/primus/src/llama2_custom.py`）：

```python
ddp=DistributedDataParallelConfig(
    ...
    overlap_grad_reduce=True,
    overlap_param_gather=True,
    use_distributed_optimizer=True,
    ...
    bucket_size=2_000_000,   # was implicit-default 134_217_728
),
```

跑法（已经修好的 `start_docker.sh` bind mount，这次免 `docker cp`）：

```bash
docker exec -d xm-primus bash -c '
  cd /home/xiaompen/mlperf-training-primus/llama2_sft/primus &&
  bash run.sh > /results/profile_run_smallbucket.log 2>&1
'
```

`profile_step_start=20, profile_step_end=23, profile_ranks=[0]`，`PRIMUS_TRAIN_ITERS=25` 短跑出 trace。

产物：

| 文件 | 内容 |
| --- | --- |
| `results/torch_profiler_traces/smallbucket_2m.pt.trace.json` | 610.1 MB Kineto trace, rank 0 |
| `results/profile_run_smallbucket.log` | 完整训练日志 (`train_utils.py:671` 显存) |
| `notes/smallbucket_2m/breakdown_step21.txt` | `full_breakdown.py` 完整输出 |

## 3. Per-stream busy time

| pid | stream | 角色 | busy | share | n_kern |
| --- | ---: | --- | ---: | ---: | ---: |
| 8 | 0  | Compute（GEMM / FMHA / norm / elem / activation） | 1373.7 ms | **89.9%** | 4990 |
| 8 | 33 | RCCL DDP grad-sync (`ncclDevKernel_Generic_1`)    | 54.0 ms | 3.5% | **2** |

- 流 oversubscription = **93.4%**（两条流 busy 求和 / step 时长）。
- Idle gap 98.0 ms (~6.4%)，其中 75 ms 集中在 step 头 0–76 ms。
- **关键观察：stream 33 只发 2 个 RCCL kernel** —— 跟 baseline 一样。说明 `bucket_size` 调整**没有让 MCore 的 ParamAndGradBuffer 切出更多 bucket**。

## 4. Kernel 类别分解（重新归类后）

> 默认 `full_breakdown.py` 的 `cat_kernel` 把 `Custom_Cijk_*` 全划进 `other`。
> 手动把 4 个 FP8 GEMM kernel 重新归到 GEMM。

| 类别 | ms | % step | 备注 |
| --- | ---: | ---: | --- |
| **FP8 GEMM** (`Custom_Cijk_*_F8*BS_*`) | **771.4** | **50.5%** | 4 个 shortname 变体 |
| **FlashAttention** (`aiter::fmha_*`) | **319.3** | **20.9%** | bwd 217.6 + fwd 89.5 + odo 7.9 + dk_dv 8.5 |
| Elementwise / dropout | 124.3 | 8.1% | 含 `vectorized_elementwise_kernel`、`fused_dropout` |
| TE SwiGLU (gated_act + dgated_act) | 67.4 | 4.4% | silu fwd + bwd |
| **RCCL grad-sync** | 53.8 | **3.5%** | overlap 0.3% |
| bf16 GEMM (`Cijk_Ailk_*BBS/BSS_*`) | 39.8 | 2.6% | LoRA-A/B 等小矩阵 |
| TE unary | 32.9 | 2.2% | identity helper |
| Fused QKV-RoPE (fwd+bwd) | 23.0 | 1.5% | |
| FP8 cast / transpose | 21.2 | 1.4% | `transpose_optimized` + `_cast_transpose_triton` |
| RMSNorm (triton fwd+bwd) | 19.9 | 1.3% | |
| Reduction | 8.5 | 0.6% | |
| MemCopy / D2D | 4.7 | 0.3% | |
| 累计 | 1486.2 | 97.3% | step 1528.47 ms（剩余 = idle 98 ms / oversub） |

## 5. Top-10 单 kernel

| # | ms | % step | kernel | 桶 |
| ---: | ---: | ---: | --- | --- |
| 1 | 348.05 | 22.8% | `Custom_Cijk_Alik_Bljk_F8B8BS … shortname1` | GEMM fwd |
| 2 | 281.10 | 18.4% | `Custom_Cijk_Alik_Bljk_F8BS … shortname1`   | GEMM bwd |
| 3 | 209.76 | 13.7% | `aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv` | Attention bwd |
| 4 | 103.42 | 6.8% | `Custom_Cijk_Alik_Bljk_F8BS … shortname0`   | GEMM |
| 5 | 89.55  | 5.9% | `aiter::fmha_fwd_hd128_bf16_causal`         | Attention fwd |
| 6 | 54.06  | 3.5% | `ncclDevKernel_Generic_1` (RCCL grad sync)  | Collective |
| 7 | 42.31  | 2.8% | `vectorized_elementwise_kernel<CUDAFunctor_add bf16>` | Elementwise |
| 8 | 38.84  | 2.5% | `Custom_Cijk_Alik_Bljk_F8B8BS … shortname0` | GEMM |
| 9 | 34.57  | 2.3% | `transformer_engine::gated_act_kernel<silu>` | Activation |
| 10 | 32.78 | 2.1% | `transformer_engine::dgated_act_kernel<silu>` | Activation |

## 6. Compute / NCCL Overlap

| 指标 | 值 |
| --- | ---: |
| compute-only 时长 | 1376.30 ms |
| nccl-only 时长 | 54.10 ms |
| overlap (compute & nccl) | **0.15 ms** |
| idle | 98.00 ms |
| **NCCL hidden behind compute** | **0.3% (0.1 / 54.2 ms)** |

跟 baseline 一致 —— bucket override 对 overlap **没有任何影响**。

## 7. VRAM (HBM)

> rank 0，`train_utils.py:671` after iter 10。
> trace `deviceProperties[*].totalGlobalMem = 309.22 GB = 287.98 GiB`.

| 指标 | 值 | 阈值判定 |
| --- | ---: | --- |
| Reserved peak (Rmax) | 295.52 GB → **95.6%** | TIGHT (≥95%) |
| Allocated peak (Pmax) | 285.84 GB → **92.4%** | warning |
| Headroom to OOM | 13.7 GB | warning (<15) |
| Fragmentation | 3.3% | excellent |
| Allocator retires | 0 | OK |

### 7.1 Bucket 分解（估算，校准到 Pmax = 285.84 GB）

| Bucket | GB | % of Pmax | 备注 |
| --- | ---: | ---: | --- |
| Weights (bf16 + fp8 hybrid) | 120.0 | 42.0% | 70B params, mixed precision |
| Activations (no recompute) | 145.0 | 50.7% | seq 8192 × 80 layers × bf16 |
| LoRA grads + Adam state | 0.8 | 0.3% | 44.5M trainable, distributed across DP=8 |
| TE FP8 caches / cuBLAS / NCCL bufs | 12.0 | 4.2% | ~10-15 GB typical |
| Allocator slack (Rmax − Pmax) | 9.68 | n/a | 不计入 Pmax |
| 未归类 | -1.68 | -0.6% | sanity gap < 5 GB ✓ |

跟 baseline VRAM 完全一样 —— LoRA + 同 batch shape，bucket 改动**对显存无影响**。

## 8. 可立即 actionable 的改动

| # | 改动 | 预期收益 | 风险 / 代价 | 实测结果 |
| ---: | --- | --- | --- | --- |
| 1 | `ddp.bucket_size = 2_000_000` (本次) | ~5%, hide RCCL 75 ms | 无 | **失败** —— RCCL 仍发 2 个 kernel；step 1522.7 → 1526.2 ms 噪声内 |
| 2 | gate per-step `.item()` 在 `step % log_interval == 0`（消除 ~660 ms/step CPU stall 的核心来源） | ~5–10%（消 head 75 ms idle 同时缓解 D2H 瓶颈） | 无；不影响 mlperf logging | 待测 |
| 3 | 缓存 8192² causal mask 跨 step（避免 `aten::tril/ones/fill_`） | ~30 ms / step | 仅在 seq 长度固定时有效 | 待测 |
| 4 | 试 `use_distributed_optimizer=False` + `bucket_size=2M` | 让 buffer 真切出多 bucket → RCCL overlap | DistOpt 关掉后梯度 mem 翻倍 (Adam state per-rank)，可能 OOM | 待测 |
| 5 | 拆 LoRA adapter 到多个 param_group（每层一组？）| 强制 buffer 多 bucket | 可能改变 optimizer behavior | 待测 |
| 6 | selective activation recompute | -30~50 GB Pmax → 显存 95%→79% | -5~15% TFLOP | 待测 |

### 8.1 失败实验（保留作为反例）

`bucket_size=2_000_000` 在 LoRA + DistOpt 下**没有效果**。

**根因（推测）**：

- MCore 的 `ParamAndGradBuffer` 按 `(dtype, requires_grad, param_group)` 分桶；
- LoRA 所有 trainable adapter 都是 bf16，且都在同一个 optimizer param_group（默认配置）；
- 这些 adapter 总参 44.5 M < `bucket_size=2_000_000` 的 element 单位...等等，这里要确认单位是不是真的是 elements；
- 即便 `bucket_size` 设了，最终 ParamAndGradBuffer 还是会把 44.5 M 全塞 1 个 bucket（因为同 group + 同 dtype）；
- DistOpt 拿到 1 个 bucket 就发 1 个 ReduceScatter（实际 trace 里 nccl_generic 53.8 ms + nccl_ag 0.22 ms = 2 kernels = 1 个 RS + 1 个 AG，匹配 distopt 模式）。

**下一步实验设计**：先确认 `bucket_size` 单位（grep `_pad_buckets_for_high_nccl_busbw` 旁边的注释或调用点），再决定走 #4 还是 #5。

## 9. 复现与产物

```bash
# 1) Edit dataset_cfg / ddp config in llama2_custom.py
# 2) docker cp 已经改成 bind mount —— 直接编辑 host 上的 .py 即可
docker exec -d xm-primus bash -c '
  cd /home/xiaompen/mlperf-training-primus/llama2_sft/primus &&
  bash run.sh > /results/profile_run_smallbucket.log 2>&1
'

# 3) 跑 kernel 分析
docker exec xm-primus bash -c '
  /opt/venv/bin/python /home/xiaompen/mlperf-training-primus/.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    /results/torch_profiler_traces/smallbucket_2m.pt.trace.json \
    "ProfilerStep#21"
' > notes/smallbucket_2m/breakdown_step21.txt 2>&1

# 4) 抓显存
grep "mem-max-" results/profile_run_smallbucket.log | tail -3
```

## 10. 相关文件

- canvas: `~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/llama2-70b-lora-mi355x-smallbucket-trace.canvas.tsx`
- trace: `results/torch_profiler_traces/smallbucket_2m.pt.trace.json`
- log: `results/profile_run_smallbucket.log`
- breakdown: `notes/smallbucket_2m/breakdown_step21.txt`
- baseline canvas（对比用）: `~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/llama2-70b-lora-mi355x-pipeline.canvas.tsx`

## 11. 后续 TODO

- [ ] 确认 MCore `bucket_size` 字段单位是 element 还是 byte（看源码 `param_and_grad_buffer.py`）。
- [ ] 实验 #2：在 train loop 把 `loss_scale.item()` 等 D2H scalar 用 `step % log_interval == 0` gate（这是当前真正的最大 lever）。
- [ ] 实验 #4：`use_distributed_optimizer=False` + small bucket（先确认 VRAM headroom 够 13.7 GB / Adam state 翻倍）。
- [ ] 实验 #5：手动拆 LoRA adapter 到 N 个 param_group。
- [ ] 把 `bucket_size=2_000_000` 从 `llama2_custom.py` 回退（这次无效，留着会让人误以为有改善）。
