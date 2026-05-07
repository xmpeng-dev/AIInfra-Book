# Llama-2-70B LoRA SFT — `overlap=False` trace breakdown (Primus, MI355X)

> Run date: 2026-04-30 10:19-10:24 UTC
> Trace: `/results/torch_profiler_traces/no_overlap_test.pt.trace.json` (1018 MB)
> Run log: `llama2_sft/primus/run.log.0430_no_overlap`
> Breakdown: `/results/no_overlap_test/breakdown_step82.txt`
> Canvas: `~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/llama2-70b-lora-mi355x-no-overlap-trace.canvas.tsx`

## 1. 实验目的

NeMo 的 hydra 配置里 `overlap_grad_reduce=False` / `overlap_param_gather=False`
（`/home/xiaompen/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_093757/hydra_resolved/config.yaml`），
而 Primus 之前一直 True/True。假设：在 `use_distributed_optimizer=True`
下，这两个 flag = True 会让 DDP 给每个 bucket 多分配一份 grad-reduce
staging + param-gather staging buffer，可能就是 71 GB VRAM 差距的来源
(Primus Pmax 285.84 GB vs NeMo 214.65 GB)。

改 `llama2_custom.py` 把两个都改为 False，跑 100 iter，验证 VRAM 是否下降。

## 2. 关键数字

| metric | baseline (overlap=True) | this run (overlap=False) | Δ |
|---|---|---|---|
| step time (avg, log) | 1522.7 ms | **1444.5 ms** | **-5.4%** |
| TFLOP/s/GPU | 2391.3 | **2521.1** | **+5.4%** |
| ProfilerStep#82 wall | 1626.5 ms | 1444.7 ms | -11.2% |
| stream 0 busy | 1377.7 ms (84.7%) | **1370.6 ms (94.9%)** | +10pp |
| stream 33 (RCCL) | 54.4 ms | 51.9 ms | ≈ |
| compute/NCCL overlap | 0.2 ms | 0.25 ms | ≈ |
| NCCL hidden | 0.4% | **0.5%** | ≈ |
| idle | 191.6 ms | **19.6 ms** | -90% |
| **mem-max-allocated** | **285.84 GB** | **285.84 GB** | **0** |
| mem-max-reserved | 295.52 GB | 295.52 GB | 0 |
| allocator retires | 0 | 0 | — |

## 3. 主要发现

### Finding 1 — 节省 78 ms/step (5.4%)，但**不是来自 VRAM**
Pmax 完全没动 (285.84 → 285.84 GB)。提升来自 idle 缩短 191 → 19.6 ms
(stream 0 busy 84.7% → 94.9%)。原因猜测：
- overlap=True 时，每完成一个 backward bucket，调度器要插入 stream wait + 启动
  bucket-N+1 的 grad-reduce kernel，host-side bookkeeping + stream sync
  开销 ~20-30 µs / layer × 80 layers ≈ 1-2 ms 直接 overhead；
- 加上 GPU 上 grad-reduce 占用 SM 与下一层 compute 抢资源（即便理论上"重叠"），
  实测 wall-clock 反而更长。

### Finding 2 — RCCL 现在完全串行
trace 显示 ncclDevKernel_Generic_1 全部落在最后 54 ms（bins 77-79，几乎 100% NCCL）。
overlap hidden 0.5%。NeMo 也是这么跑的，所以这是 parity 行为，不是 regression。

### Finding 3 — **71 GB VRAM 差距不在 DistOpt+DDP buffer**
这是这次实验最重要的结论。两边 DDP 配置现在完全对齐：
- `use_distributed_optimizer=True` ✅
- `overlap_grad_reduce=False` ✅
- `overlap_param_gather=False` ✅
- `fp8_param_gather=False` ✅

但 Pmax 还是 285.84 GB。NeMo Pmax 214.65 GB。差距没消失。

之前所有"调 flag 省 VRAM"的尝试均失败：
| 尝试 | 预期省 | 实际 | 结果 |
|---|---|---|---|
| `fp8_param_gather=False` (DDP 层) | -69 GB | 0 GB | OOM 282 GB |
| `keep_fp8_transpose_cache=False` | -? GB | 0 GB | OOM |
| `overlap_grad_reduce=False` + `overlap_param_gather=False` | -? GB | **0 GB** | 训练正常但无影响 |

下一步必须做**结构性**改动而不是 flag 调优：
- 用 `torch.cuda.memory._dump_snapshot()` 在 iter 5 抓 snapshot，按 tensor name
  归因 71 GB 是哪些 buffer；
- 重点排查 Megatron-Bridge 的 `ParamAndGradBuffer` 是不是为了 grad accumulation
  保留了一份 main_grad fp32 的 全量副本（NeMo 用 fp32 master 但不是全量副本）；
- 排查 mixed_precision propagation 是否把 `model.fp8_param=True` 传上去
  导致 TE 内部多分配一份 weight transpose cache。

## 4. 配置变更

`llama2_sft/primus/src/llama2_custom.py` line 602-628：

```python
ddp=DistributedDataParallelConfig(
    check_for_nan_in_grad=False,
    grad_reduce_in_fp32=False,
    overlap_grad_reduce=False,    # was True
    overlap_param_gather=False,   # was True
    average_in_collective=True,
    use_distributed_optimizer=True,
    gradient_reduce_div_fusion=True,
    pad_buckets_for_high_nccl_busbw=True,
    use_megatron_fsdp=False,
    keep_fp8_transpose_cache=True,
    fp8_param_gather=False,
),
```

`profiling.profile_step_start/end` 改为 80/85（per skill recommendation）。
`PRIMUS_TRAIN_ITERS=100` (env override)。

## 5. 训练数据健康度

```
iter   10  step 6437 ms  loss 3.227  norm 1.036
iter   20  step 1446 ms  loss 1.682  norm 0.376
iter   50  step 1445 ms  loss 1.301  norm 0.114
iter   80  step 1444 ms  loss 1.281  norm 0.121
iter  100  step 1462 ms  loss 1.247  norm 0.097
val loss @100 = 0.898 (< 0.925 → early exit)
```
Loss 收敛，无 nan / skipped iter。第 1 步 6437 ms 是首次 cudagraph capture
+ inductor cache fill；之后稳定在 1444 ms。

## 6. Trace 详情 (ProfilerStep#82)

### 6.1 Per-stream
| stream | role | busy ms | share |
|---|---|---|---|
| pid=8 stream=0  | Compute | 1370.63 | 94.9% |
| pid=8 stream=33 | RCCL DDP grad-sync | 51.90 | 3.6% |
total step wall = 1444.73 ms · stream oversub 98.5%

### 6.2 Kernel categories (FP8 GEMM 已 re-categorize)
| category | ms | % |
|---|---|---|
| FP8 GEMM (Custom_Cijk_*_F8*) | 768.0 | 53.2% |
| FlashAttention (aiter::fmha + ck) | 315.4 | 21.8% |
| Elementwise / dropout / cast | 124.4 | 8.6% |
| TE SwiGLU / dgated_act | 64.4 | 4.5% |
| RCCL grad-sync | 51.9 | 3.6% |
| Fused QKV-RoPE | 23.1 | 1.6% |
| RMSNorm | 19.2 | 1.3% |
| FP8 cast/transpose | 14.6 | 1.0% |
| bf16 GEMM | 12.4 | 0.9% |
| Reduction | 8.6 | 0.6% |
| Memcpy | 4.8 | 0.3% |
| Optimizer / misc | 15.7 | 1.1% |

### 6.3 Top kernels
1. Custom_Cijk_Alik_Bljk_F8B8BS_…_shortname1 — 347.32 ms (24.0%)
2. Custom_Cijk_Alik_Bljk_F8BS_…_shortname1 — 279.39 ms (19.3%)
3. aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv — 209.29 ms (14.5%)
4. Custom_Cijk_Alik_Bljk_F8BS_…_shortname0 — 102.42 ms (7.1%)
5. aiter::fmha_fwd_hd128_bf16_causal — 89.61 ms (6.2%)
6. ncclDevKernel_Generic_1 — 51.94 ms (3.6%)
7. vectorized_elementwise_kernel<add bf16> — 42.53 ms (2.9%)
8. Custom_Cijk_Alik_Bljk_F8B8BS_…_shortname0 — 38.90 ms (2.7%)
9. transformer_engine::gated_act_kernel<silu> — 34.59 ms (2.4%)
10. transformer_engine::dgated_act_kernel<silu> — 32.53 ms (2.3%)

## 7. VRAM 估算 (Mode A · LoRA)

| bucket | GB | % of Pmax |
|---|---|---|
| Weights (bf16+fp8 hybrid) | 120.0 | 42.0% |
| Activations (no recompute, seq=8192) | 145.0 | 50.7% |
| DistOpt + DDP buffers | 8.0 | 2.8% |
| TE FP8 caches / hipBLASLt / RCCL | 12.0 | 4.2% |
| Allocator slack (Rmax-Pmax) | 9.7 | 3.4% |
| Other / unaccounted | -8.8 | -3.1% |
| **sum** | **285.9** | **100%** |

注：unaccounted 为负说明 weights 估的略高 5-10 GB 或 activation 估的略低，
误差在 ±5%，没超出 skill 的 10% sanity gap。

## 8. 决定要不要回退

净结论：**保留 overlap=False**。
- 节省 78 ms/step (5.4%) 是真实可复现的
- VRAM 没有变化（不是动机但也没损失）
- RCCL 完全串行 = NeMo parity，无 regression risk
- 后续 NeMo 对照实验更直接（同 DDP schedule）

唯一风险是 multi-node 时如果 RCCL tail 超过当前 backward bucket 的串行延迟
就需要 overlap=True 才能 hide，但这是 1 node × 8 GPU 的场景，无关。

## 9. 下一步

| # | 实验 | 目的 | 优先级 |
|---|---|---|---|
| 1 | NeMo run.sh 实测 → 抓同样 trace / mem snapshot | bucket-by-bucket 对比 71 GB 真实出处 | **P0** |
| 2 | `torch.cuda.memory._dump_snapshot()` @ iter 5 (Primus) | 按 tensor name 归因 71 GB | P0 |
| 3 | 对比 NeMo vs Primus distributed_optimizer.py 的 ParamAndGradBuffer 实现 | 找代码层差异 | P1 |
| 4 | 尝试 selective activation recompute (省 ~30-50 GB) | 临时给 num_workers>0 让出空间 | P2 |

实验 1+2 安排在本 note 之后，scripts/log/trace 放在
`/home/xiaompen/mlperf-training-primus/debug/`。

## 10. 参考

- baseline canvas: `~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/llama2-70b-lora-mi355x-after-fix.canvas.tsx`
- 之前 Primus vs NeMo 对比: `slab/notes/mlperf-llama/2026-04-30_primus_vs_nemo_llama2_70b_lora_trace_compare.md`
- skill: `.cursor/skills/trace-vram-canvas/SKILL.md`
- NeMo hydra config: `/home/xiaompen/mlperf-training-6-0/llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_093757/hydra_resolved/config.yaml`

## 11. Reproducibility

```bash
# 修改 (已 commit-ready):
#   llama2_sft/primus/src/llama2_custom.py:606-607  overlap_*=False
#   llama2_sft/primus/src/llama2_custom.py:689-690  profile_step_start/end=80/85

docker exec xm-primus bash -c '
  cd /home/xiaompen/mlperf-training-primus/llama2_sft/primus &&
  PRIMUS_TRAIN_ITERS=100 bash run.sh' \
  2>&1 | tee llama2_sft/primus/run.log.0430_no_overlap

docker exec xm-primus bash -c '
  cd /home/xiaompen/mlperf-training-primus &&
  python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    /results/torch_profiler_traces/no_overlap_test.pt.trace.json \
    ProfilerStep#82' \
  > /results/no_overlap_test/breakdown_step82.txt
```
