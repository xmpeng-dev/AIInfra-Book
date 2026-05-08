# GPT-OSS-20B B200 baseline (step #17, rank2+rank3 平均)

**日期**: 2026-05-07
**硬件**: NVIDIA B200 × 8 (sm100), 1 节点
**Trace 来源**: `torch_trace_base_20260423_223318/torchprof/primus-megatron-exp[gpt_oss_20b_nvidia]-rank[*].pt.trace.json`
**ProfilerStep**: #17（4 个 active step 16–19 中 wall time 最对齐的一个）
**关联**: note `16`（误把这份 trace 当 MI355X 做了一次 Tier 1 审计 — 见 §0 更正）；note `15`（MI355X EP=1 单步 trace 校准）

## 0. 校正：note 16 头里 "MI355X" 是错的

note 16 表头写"硬件: MI355X × 8"。本次重新检查 trace 文件名 `primus-megatron-exp[gpt_oss_20b_**nvidia**]-rank[*]`，且 Top-25 kernel 全部是 `nvjet_sm100_*` / `cudnn_generated_fort_native_sdpa_sm100_*` / `ncclDevKernel_*`，**这是 B200 (sm100) 的 trace**。note 16 §"一手数据" 那张 Top-25 表里的 `Cijk_*` / `aiter::fmha` / `ck_tile` / RCCL 是另一份 MI355X trace 的数据，被错混进去了，**已废弃，不要再引用**。本 note 是 B200 一侧的正式 baseline。

## 1. TL;DR

- **B200 step #17 wall time = 966.75 ms ± 0.10 ms**（8 个 rank 全 EP=1 同步 ≤ 0.3 ms 抖动）。
- 仅 rank 2 / rank 3 两份 trace 的 GPU 段未被截断，**baseline = (rank2 + rank3) / 2**。
- compute / NCCL 拆分（avg）：**compute-busy 48.5 %**，**NCCL hidden 24 %**，**stream oversub 106 %**（多流并行有效）。
- 最大头：**`ncclDevKernel_SendRecv` ≈ 224 ms (23 %)**，其次 RS 192 ms (20 %) + AG 154 ms (16 %) + GEMM 141 ms (15 %) + elementwise 88 ms (9 %) + attn 64 ms (7 %)。
- **rank2 vs rank3 GEMM 差 67 ms / SendRecv 差 120 ms**（见 §4），这是真实 MoE per-expert token 不均；**只看一个 rank 会高估 SendRecv、低估 GEMM**。

## 2. 8 个 rank ProfilerStep 全表（CPU 侧，全部完好）

| rank | trace size | step 16 (ms) | **step 17 (ms)** | step 18 (ms) | step 19 (ms) | 17–19 mean | GPU 段完整 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 329 MB | 1046.13 | 966.85 | 957.63 | 978.00 | 967.49 | ❌ |
| 1 | 308 MB | 1044.82 | 966.82 | 957.64 | 976.70 | 967.05 | ❌ |
| 2 | 403 MB | 1046.04 | **966.76** | 957.75 | 976.53 | 967.01 | ✅ |
| 3 | 407 MB | 1046.17 | **966.64** | 957.80 | 976.46 | 966.97 | ✅ |
| 4 | 259 MB | 1046.13 | 966.85 | 957.71 | 975.25 | 966.60 | ❌ |
| 5 | 229 MB | 1046.13 | 966.81 | 957.64 | 976.66 | 967.04 | ❌ |
| 6 | 235 MB | 1046.10 | 966.73 | 957.71 | 976.67 | 967.04 | ❌ |
| 7 | 294 MB | 1047.55 | 966.57 | 957.15 | 976.80 | 966.84 | ❌ |
| **mean** | — | 1046.13 | **966.75** | 957.63 | 976.63 | 967.01 | — |
| **min/max Δ** | — | 2.73 | **0.28** | 0.65 | 2.75 | 0.89 | — |

**采用值**：`B200_step17_baseline = 966.75 ms / 32 sample`（gbs=32, mbs=4, DP=8, TP=PP=EP=1）。

step 16 比 17–19 慢 ~80 ms（cudnn / cuBLAS autotune + cuda graph capture 尾巴），**不进 baseline**。

## 3. GPU baseline（rank2 + rank3 平均）

### 3.1 Per-stream busy time

| stream | avg ms | avg % step | rk2 ms | rk3 ms | 角色 |
|---:|---:|---:|---:|---:|---|
| 7 (compute) | **573.76** | **59.4** | 617.44 | 530.08 | 主 stream：GEMM/attn/norm/elem + SendRecv |
| 55 (NCCL #1) | 254.31 | 26.3 | 257.51 | 251.11 | 跨节点 RS+AG |
| 51 (NCCL #2) | 79.46 | 8.2 | 79.97 | 78.94 | 同上 |
| 173 (副 GEMM) | 42.80 | 4.4 | 34.89 | 50.70 | grouped/dense GEMM 副流 |
| 174 (副 GEMM) | 28.37 | 2.9 | 19.80 | 36.93 | 同 |
| 175 (副 GEMM) | 20.41 | 2.1 |  8.12 | 32.69 | 同 |
| 31 (housekeeping) | 17.39 | 1.8 | 17.43 | 17.34 | small reductions / memcpy |
| 176 |  4.76 | 0.5 |  0.45 |  9.07 | small GEMM |
| 27 |  0.31 | 0.0 |  0.31 |  0.30 | mgmt |
| **sum** | **1021.55** | **105.7** | 1035.94 | 1007.16 | stream 真实并行 |

stream oversub 106 % → **副流真实在跑**（不是只挂在 stream 7 上串行）。

### 3.2 GPU 类别 baseline（按 ms 由大到小）

| 类别 | avg ms | avg % step | rk2 ms | rk3 ms | rk3−rk2 |
|---|---:|---:|---:|---:|---:|
| nccl_a2a (SendRecv) | **223.82** | **23.2** | 283.60 | 164.04 | −119.6 |
| nccl_rs (ReduceScatter) | 191.62 | 19.8 | 193.16 | 190.08 | −3.1 |
| nccl_ag (AllGather) | 153.91 | 15.9 | 162.62 | 145.19 | −17.4 |
| gemm | 140.67 | 14.5 | 107.06 | 174.27 | +67.2 |
| elementwise | 87.89 | 9.1 | 78.90 | 96.88 | +18.0 |
| attn_kernel | 64.13 | 6.6 | 64.11 | 64.14 | +0.0 |
| moe_dispatch | 41.14 | 4.3 | 33.31 | 48.96 | +15.7 |
| other | 39.25 | 4.1 | 33.69 | 44.81 | +11.1 |
| norm | 30.13 | 3.1 | 30.27 | 29.99 | −0.3 |
| nccl_ar | 19.60 | 2.0 | 19.75 | 19.44 | −0.3 |
| optimizer (TE multi-tensor Adam) | 14.34 | 1.5 | 14.32 | 14.35 | 0.0 |
| reduction | 10.30 | 1.1 | 10.34 | 10.26 | −0.1 |
| memcpy | 4.72 | 0.5 | 4.76 | 4.68 | −0.1 |
| fp8_cast | 0.05 | 0.0 | 0.05 | 0.05 | 0.0 |
| **跨 stream 求和** | **1021.55** | **105.7** | 1035.94 | 1007.16 | — |

### 3.3 Compute / NCCL overlap

| 指标 | avg | rank 2 | rank 3 |
|---|---:|---:|---:|
| compute-only ms | 341.58 | 295.30 | 387.85 |
| nccl-only ms | 411.80 | 479.70 | 343.90 |
| overlap (compute & nccl) ms | 127.33 | 121.45 | 133.20 |
| idle ms | 86.08 | 70.40 | 101.75 |
| **compute-busy %** | **48.5** | 43.1 | 53.9 |
| **NCCL hidden behind compute %** | **24.0** | 20.2 | 27.9 |

### 3.4 Top-11 GPU kernel baseline（avg by dur）

| avg ms | rk2 ms | rk3 ms | kernel |
|---:|---:|---:|---|
| **223.82** | 283.60 | 164.04 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)` |
| 191.62 | 193.16 | 190.08 | `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` |
| 153.91 | 162.62 | 145.19 | `ncclDevKernel_AllGather_RING_LL` |
| 42.87 | 42.87 | 42.87 | `cudnn_..._sdpa_sm100_flash_bprop_f16_..._128x128x64` (FMHA bwd) |
| 38.47 | 23.64 | 53.29 | `nvjet_sm100_qrsss_256x256_128x4_2x1_2cta_v_bz_NTT` (MoE GEMM) |
| 32.92 | 23.22 | 42.62 | `nvjet_sm100_qrtst_128x256_128x6_2x1_2cta_v_bz_NNT` (MoE GEMM) |
| 26.83 | 19.26 | 34.40 | `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_TNT` (MoE GEMM) |
| 21.35 | 15.54 | 27.15 | `triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0` (SwiGLU fused) |
| 19.89 | 19.88 | 19.90 | `cudnn_..._sdpa_sm100_flash_fprop_..._128x128x64` (FMHA fwd) |
| 19.60 | 19.75 | 19.44 | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` |
| 14.34 | 14.32 | 14.35 | `multi_tensor_apply_kernel<...AdamFunctor<float, float>>` (TE Adam) |

## 4. rank2 vs rank3 失衡 — 解读 + 使用注意

step wall 完全对齐（差 0.12 ms），但 GPU 内：
- rank 2 多花 **~120 ms** 在 SendRecv，少花 **~67 ms** 在 GEMM、~16 ms 在 moe_dispatch、~18 ms 在 elementwise；
- rank 3 反过来；
- compute-busy 差 10.8 个百分点（43.1 vs 53.9），但 wall 一样 → **rank 2 多出来的时间全部花在等集合通信上**。

yaml 写 `tp1pp1ep1`、纯 DP=8。EP=1 理论上无 MoE all-to-all，但 trace 里 SendRecv 占 16–28 % step，可能性：

- **(a)** 实际 EP 不是 1（被 shell config 或 Megatron 默认值覆盖）；
- **(b)** 这些 SendRecv 来自 **HSDP/FSDP 的 ring 实现**（NCCL 在 RING 算法下用 SendRecv 拼 RS/AG）。

依据：脚本默认 `SPLIT_NCCL_BY_CPU=1`，本次 0 个被重标 → c10d 注释里没有显式 `c10d::send/recv`，**(b) 更可能**。

GEMM 差 67 ms **不能用 ring-NCCL 解释**——只可能是 **MoE token-per-expert 在两个 rank 上分布不均**（router 输出依赖输入），是真实的 expert 负载失衡。

**因此本 note 的 baseline 表（§3）必须用 rank2+rank3 平均**。任何后续 A/B 都要至少抓 2 个 rank（推荐 rank 2 + rank 3 或 rank 2 + rank 4）才能区分"算法变更带来的差"与"DP rank 之间的天然方差"。

## 5. 复现命令

```bash
# 8 个 rank 并行解析（每个 rank 取 ProfilerStep#17）
mkdir -p /tmp/tier1_b200 && cd /tmp/tier1_b200
for r in 0 1 2 3 4 5 6 7; do
  f=$(ls /home/xiaompen/primus-megatron-exp\[gpt_oss_20b_nvidia\]-rank\[$r\].*.pt.trace.json)
  python3 /home/xiaompen/mlperf-training/.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    "$f" ProfilerStep#17 > rank${r}.txt 2>&1 &
done
wait
```

输出（已保存）：
- `/tmp/tier1_b200/rank{0..7}.txt` — 每 rank 一份 raw breakdown
- rank 2 / rank 3 各 200+ 行（含 80-bin 时间序列），其余 6 个文件只有 step 表 + `[warn] tail truncated: premature EOF`

## 6. 为什么 6 个 rank 的 GPU 段缺失 + 下次怎么避免

trace 文件尾部都是 `cat: "python_function"` 的 CPU stack，结尾**没有 `]}`**——Kineto 在 CUPTI buffer 序列化阶段被打断。size 也明显不一致（229–407 MB），rank 4–7 比 rank 2–3 小 100–180 MB。

**修不了已截断的 GPU 段**（物理缺失，不是 parser 问题）。下次抓 trace：

1. 把 active step 从 4 缩到 1（只保 step #17）：
   ```yaml
   profile_step_start: 17
   profile_step_end: 18
   ```
   单 rank 文件应能落在 80–100 MB。
2. `with_stack: false`、`record_shapes: false`（已是默认，确认即可）。
3. 抓前确认 `tensorboard_dir` 所在盘 ≥ 10 GB 余量。
4. **不要 SIGINT**：让 `train_iters` 自然退出，profiler 才有机会 `flush` CUPTI buffer。

## 7. 与 MI355X (note 16) 的可比性

**不可直接比**，两份 trace 配置不同：

| 维度 | B200（本 note） | MI355X（note 16 头注，但表数据已废） |
|---|---|---|
| step wall (#17) | **966.75 ms** | 1129.33 ms（声称） |
| FP8 recipe | 看 yaml `bf16` 路径 + cudnn flash | hybrid + ck_v3 FMHA |
| Adam | TE `multi_tensor_adam` | `multi_tensor_apply_kernel<AdamFunctor>` |
| residual add 独立 launch | 否（已被 cudnn/TE/inductor 吸进去） | **是**（42 ms `vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>`） |
| SwiGLU fused kernel | `triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0`（21 ms avg） | `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1`（30.7 ms） |

**Tier 1A 关键结论**：B200 上**没有 42 ms 独立 residual add**，note 16 §A "fused residual+RMSNorm" 那个 −3.4~4.3 % 收益**只在 MI355X 上成立**，**不能 port 到 B200**。Tier 1B "SwiGLU bwd 去 cat" 跨硬件成立（B200 21 ms / MI355X 31 ms 都带 `cat`）。

要做严格 B200 vs MI355X A/B，必须：
- 同一份 yaml + 同一个 patch 集合
- 抓同一个 step（推荐 #17）
- 至少 2 个 rank 平均（避免 §4 那种失衡）
- 单独写一个 comparison note，不要再混进 note 16

## 8. 相关文件

- 8 个 rank trace（symlink）：`small_llm_moe_pretraining/primus/run-trace/torch_trace_base_20260423_223318/torchprof/`
- 实体文件：`/home/xiaompen/primus-megatron-exp[gpt_oss_20b_nvidia]-rank[*].*.pt.trace.json`
- 解析脚本：`.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py`
- raw 输出：`/tmp/tier1_b200/rank{0..7}.txt`
- yaml: `small_llm_moe_pretraining/primus/gpt_oss_20B-pretrain-fp8.yaml`
- 关联 notes: `15` (MI355X EP=1 trace 校准), `16` (Tier 1 audit — 头注硬件需更正), `17`/`19` (fused residual+RMSNorm 落地)
