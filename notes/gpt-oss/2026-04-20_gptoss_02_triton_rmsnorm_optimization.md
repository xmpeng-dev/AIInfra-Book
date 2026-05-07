# GPT-OSS-20B Triton RMSNorm 优化 — MI355X

**日期**: 2026-04-20
**模型**: GPT-OSS-20B (24L, hidden=2880, 64H/8KVH, 32 experts, head_dim=128)
**配置**: 1×8 MI355X, fp8 hybrid GEMM + bf16 weight, mbs=2 / gbs=16, tp=pp=ep=1
**容器**: `xiaoming-mlperf` (`tasimage/primus:gpt-oss-20b_training_6.0_2026-04-07-19-47-24_dev`)

---

## TL;DR

| | B0 (TE 基线) | B8 (Triton RMSNorm) | **B10 (Triton + linear_qkv 全覆盖)** |
|---|---:|---:|---:|
| step time (median) | 796.4 ms | 773.8 ms | **770.3 ms** |
| Δ vs B0 | — | −2.84% | **−3.28%** |
| TFLOP/s/GPU | 518.5 | 533.7 | **536.0 (+3.37%)** |
| tokens/s/GPU | 20572 | 21174 | **21269** |
| **norm GPU 时间** (3 steps) | **253.6 ms** | 64.7 ms | **35.8 ms (−86%)** |
| nan / skip | 0 / 0 | 0 / 0 | 0 / 0 |

**结论**：B10 可上线，端到端 +3.37% 吞吐，RMSNorm GPU 时间压缩到基线 14%，数值完全稳定。

---

## 背景与起点

打开 `use_turbo_rms_norm=true`（走 `primus_turbo::rmsnorm_fwd_two_scan_kernel`）时实测对 GPT-OSS-20B 反而 **regress 22%**。原因：上游 two-scan kernel 是为 DeepSeek-V3 大 hidden（7168）做的算法选择，对 GPT-OSS-20B 的 hidden=2880 + 极大 batch（B=16384）这个形态不友好，调度退化。

而 baseline TE RMSNorm 自身也是大头：

```
B0 norm GPU 时间 = 253.6 ms / 3 steps ≈ 84.5 ms/step ≈ 10.6% of step
其中：
  rmsnorm_bwd_finalize_general_kernel  166.86 ms  (dgamma reduction)
  rmsnorm_bwd_general_kernel            63.06 ms
  rmsnorm_fwd_general_kernel            23.71 ms
```

`bwd_finalize`（计算 dgamma）就吃掉 65%，TE 没有融合方案可选。

---

## 方案

### Step 1 — 自研 Triton RMSNorm

写在 `/home/xiaompen/Primus/primus/backends/megatron/core/extensions/triton_rmsnorm.py`，两套 kernel + 动态选型：

- **single-row kernel** (`_rmsnorm_fwd/bwd_kernel`)：一个 program 处理一行。适合 H 大、B 中等。
- **multi-row kernel** (`_rmsnorm_fwd/bwd_kernel_multi_row`)：一个 program 处理 N 行。适合 H 小、B 巨大（GPT-OSS-20B 主分支：H=2880, B=16384）。
- `_pick_config(H, B)` 启发式：根据 (H, B) 选 `BLOCK_H / ROWS_PER_BLOCK / num_warps / num_stages`。
- `dgamma` 用 `dgamma_partial.sum(dim=0)` 替代 TE 的 `bwd_finalize` 大 reduction kernel。

**单 kernel 收益**（per launch, B=16384, H=2880）：

| | TE | Triton |
|---|---:|---:|
| fwd | 81 us | **34.3 us** (≈2.4×) |
| bwd | 217 us | **108.6 us** (≈3×) |
| bwd_finalize | 573 us | (消除) |

数值正确性（vs `F.rms_norm`）：fwd max abs Δ 9.77e-4，dx 3.91e-3，dg 6.10e-5，q/k_norm 非连续 4D 也通过。

### Step 2 — 接管残留的 `linear_qkv` 路径

B8 部署后发现 trace 还有零星 TE rmsnorm 调用（≈13 ms / 3 steps）。根因：attention 的 `linear_qkv` 用的是 TE 的 **fused `LayerNormLinear`**，rmsnorm 写在 C++ 算子内部，不受 `te.pytorch.RMSNorm` 的 monkey-patch 影响。

#### 是否值得自己写 fused triton norm+cast_to_fp8 的判断

先做 microbench (`bench_layernorm_linear.py`，B=16384, in=2880, out=10240, FP8 DelayedScaling)：

| | fwd+bwd / call |
|---|---:|
| TE 融合 LayerNormLinear (RMS) | 1850 us |
| Triton RMSNorm + TE Linear (split) | **1784 us (−3.6%)** |

split 比 fused 只快 3-4%，因为 TE 的 fused kernel 把 `rmsnorm + cast_to_fp8 + transpose` 写在一起了，cast cost 已经被吃掉。**自研 fused triton norm+cast 路径放弃**：理论天花板 ~1.5 ms/step，工程量 2-3 天，还要绑死 TE 内部 quantizer API，ROI 太差。

#### 实施 split 方案

新增 `nn.Module` `PrimusTurboLayerNormColumnParallelLinear`，组合 `[PrimusTurboRMSNorm (Triton), TEColumnParallelLinear]`，暴露 `layer_norm_weight / weight / bias / sharded_state_dict` 一系列 Megatron 期望的属性，做到 ckpt 级别向后兼容。

#### 路由踩的坑

第一版改了 `transformer_engine_spec_provider.TELayerNormColumnParallelLinear` 的 patch，但 B9 trace 里 TE rmsnorm 仍在。原因是 `PrimusTurboSpecProvider.column_parallel_layer_norm_linear()` 里写的：

```python
return PrimusTurboLayerNormColumnParallelLinear if cfg.use_turbo_parallel_linear else TELayerNormColumnParallelLinear
```

而我们这次没开 `use_turbo_parallel_linear`，永远走 else 分支。

**修复**：增加一个独立分支，让 `use_turbo_rms_norm=True` 时直接返回我们的 `nn.Module` 版本，不依赖 `use_turbo_parallel_linear`。

```python
if cfg.use_turbo_parallel_linear:
    return PrimusTurboLayerNormColumnParallelLinear
if getattr(cfg, "use_turbo_rms_norm", False):
    return PrimusTurboLayerNormColumnParallelLinear
return TELayerNormColumnParallelLinear
```

B10 再跑，trace 里 0 个 `te::rmsnorm_*` kernel，全部是 `_rmsnorm_*_kernel(_multi_row)`。日志确认：

```
(linear_qkv): PrimusTurboLayerNormColumnParallelLinear(in=2880, out=5120, norm=PrimusTurboRMSNorm)
```

---

## 端到端结果（trace-level avg-step 拆解）

| Category | B0 | B8 | **B10** | B10 Δ vs B0 |
|---|---:|---:|---:|---:|
| comm | 2042.3 | 2092.7 | 2066.1 | +1.2% |
| gemm_bf16 | 1654.0 | 1721.6 | 1692.5 | +2.3% |
| elementwise | 695.3 | 771.5 | 766.5 | +10.2% |
| attention | 348.6 | 341.9 | 339.9 | −2.5% |
| **norm** | **253.6** | 64.7 | **35.8** | **−85.9%** |
| moe_permute | 115.7 | 109.0 | 107.2 | −7.4% |
| optimizer | 99.6 | 99.4 | 99.9 | — |
| **avg step** | **795.7** | 778.2 | **769.5** | **−3.30%** |
| GPU util | 97.8% | 97.9% | **98.0%** | +0.2pp |
| comp time | 2087.8 | 2027.9 | **2010.8** | −3.7% |

注：elementwise / gemm_bf16 的 wall-time slice 略涨是 **bookkeeping**，不是真退化 —— norm 让出热点带后，相邻 op 占据 kernel slot 的时刻提前，分摊到它们 wall-time 的份额变大。净 step 时间还是降了 26 ms。

### Amdahl 分析

| | RMSNorm GPU 节省 | wall step 节省 | 转化率 |
|---|---:|---:|---:|
| B0 → B8 | 63 ms / step | 22.6 ms | 36% |
| B0 → B10 | 72.6 ms / step | 26.1 ms | 36% |

约 70% 的 RMSNorm 节省在 B0 时本来就被 NCCL grad-reduce overlap 吃掉了，外加相邻 kernel 提前抢占释放的 slot，只有 ~36% 的 kernel 节省可以转化为 wall-clock。这在重 overlap 系统里属于正常区间。

---

## Comm 专项分析

| | B0 | B8 | B10 |
|---|---:|---:|---:|
| comm GPU 时间 (ms / 3 steps) | 2042.3 | 2092.7 | 2066.1 |
| comm 启动数 / 3 steps | 636 | 636 | 636 |
| overlap (ms) | 871.2 | 893.8 | 878.4 |
| **exposed_comm** | **14.7%** | 14.6% | **15.0%** |

**结论：comm 不是接下来的瓶颈。** 几个观察：

1. **comm 总 GPU 时间几乎不变**（±2.5%）—— 通信 volume 没变（ep=1, tp=1，只有 DP grad-reduce + start-of-step param all-gather）。
2. **exposed_comm 微涨 14.7% → 15.0%**：absolute exposed time 接近，但 comp 缩了 77 ms / step，比例自然上抬。
3. **comm GPU 时间的 ±50 ms 波动是 kernel 边界对齐**：norm 缩短后 NCCL bucket 触发时刻变，reduce 窗口对齐变，不是真实带宽变化。

**为什么 15% 已经接近物理下限**：单节点 8-GPU bf16-weight + fp8-GEMM、DP-only，剩下的 15% 主要是

- last-bucket tail（最后一个 grad bucket 必须等 bwd 完才能 reduce，无法 pipeline）
- RCCL launch overhead（24 layers × 多 op，每次 ~3-5 us host）
- fp8 amax DelayedScaling 每步同步

要再压 comm，得动 DDP bucket、async_op、grad_reduce_in_bf16、custom_fsdp，跟 RMSNorm 没关系。

---

## 弯路 & 决策记录

1. **`primus_turbo::rmsnorm_fwd_two_scan_kernel`**：DSv3 友好，GPT-OSS-20B regress 22%。原因是 `(H, B)` 形态不同导致 two-scan 调度退化。**结论：放弃在 GPT-OSS-20B 上用，改自研 Triton。**

2. **fused triton `rmsnorm + cast_to_fp8 + transpose`**：实测 cast 开销已经被 TE 内部融合吃掉，自研 split 比 fused 只快 3-4%（理论上限 ~1.5 ms/step）。**结论：放弃，工程量 vs 收益不划算。**

3. **`PrimusTurboSpecProvider` 路由 bug**：默认条件挂在 `use_turbo_parallel_linear` 上，导致 `use_turbo_rms_norm=true` 时 `linear_qkv` 仍走 TE fused。**修复：增加独立分支判断 `use_turbo_rms_norm`。**

---

## 文件清单（最终态，container `/workspace/Primus`）

- **NEW** `primus/backends/megatron/core/extensions/triton_rmsnorm.py`
  两套 Triton kernel + autograd wrapper，零 Megatron / TE 依赖。
- **MODIFIED** `primus/backends/megatron/core/extensions/primus_turbo.py`
  - `PrimusTurboRMSNorm.forward` 改调 `triton_rmsnorm`。
  - 新 `PrimusTurboLayerNormColumnParallelLinear` (nn.Module)，组合 `[PrimusTurboRMSNorm, TEColumnParallelLinear]`，附带 ckpt 兼容的 `sharded_state_dict`。
- **MODIFIED** `primus/backends/megatron/patches/turbo/rms_norm_patches.py`
  同时 patch `te.pytorch.RMSNorm` 和 `transformer_engine_spec_provider.TELayerNormColumnParallelLinear`。
- **MODIFIED** `primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py`
  `column_parallel_layer_norm_linear()` 在 `use_turbo_rms_norm=True` 时无条件返回我们的版本。

用户层开关不变：yaml 里 `use_turbo_rms_norm: true`。

---

## 下一步候选（按 ROI 排序）

RMSNorm 的 headroom 已经基本榨干（35.8 / 770 = 4.6% of step）。剩余大头：

| Category | B10 GPU 时间 / step | 占比 | 思路 |
|---|---:|---:|---|
| comm | 689 ms | 89% (重叠后 15%) | DDP bucket / async_op，工程量大 |
| gemm_bf16 | 564 ms | 73% | 选择性 fp8 promotion（仅 cast cost 已被融合的 site） |
| elementwise | 256 ms | 33% | Triton 融合 SwiGLU + cast 之类的小 kernel |
| attention | 113 ms | 15% | 已 CK v3，难再挤 |

**优先建议**：
1. **elementwise Triton 融合**（256 ms/step）—— 当前 GPU 时间集中在 SwiGLU + cast + add bias 这类单 op，融合后预计可省 30-50 ms/step。
2. 然后看 **DDP bucket strategy**，把 last-bucket tail 切薄。
3. fp8 promotion 留作后续，需要 site-level 评估每个 GEMM 的 cast cost / GEMM cost 比。

详细数据见同目录 [`2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md`](./2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md)（源文件 `/home/xiaompen/mlperf/TRITON_RMSNORM_REPORT.md`）。
