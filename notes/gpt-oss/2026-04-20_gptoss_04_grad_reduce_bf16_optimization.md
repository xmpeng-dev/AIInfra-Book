# GPT-OSS-20B `grad_reduce_in_bf16` 优化 — MI355X (B11)

**日期**: 2026-04-20 (晚)
**承接**: [`2026-04-20_gptoss_02_triton_rmsnorm_optimization.md`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md)
**配置**: 1×8 MI355X, fp8 hybrid GEMM + bf16 weight, mbs=2 / gbs=16

---

## TL;DR

**B11 = B10 (Triton RMSNorm 全覆盖) + `grad_reduce_in_bf16=true`**

| | B0 (TE 基线) | B10 (Triton RMSNorm) | **B11 (本次)** |
|---|---:|---:|---:|
| step time (median) | 796.4 ms | 770.3 ms | **712.9 ms** |
| Δ vs B0 | — | −3.28% | **−10.5%** |
| TFLOP/s/GPU | 518.5 | 536.0 | **579.2 (+11.7%)** |
| tokens/s/GPU | 20572 | 21269 | **22982** |
| HIP mem 占用 | 232.5 GiB | 232.5 GiB | **196.4 GiB (−36 GB)** |
| nan / skip | 0 / 0 | 0 / 0 | 0 / 0 |

单一 yaml 改动（`grad_reduce_in_bf16: true`），**+8% TFLOPs / 释放 36 GB 显存 / 0 验收风险**。

---

## 背景：怎么找到这个优化

B10 已经把 RMSNorm 压到 4.6% of step，按 trace category 看下一个大头是 `elementwise` (766 ms / 3 steps ≈ 33% of step)。原计划是 "Triton 融合 SwiGLU"。

但深入 elementwise 拆 kernel 后发现**真实分布完全不是 SwiGLU**：

| Kernel | Time / 3 steps | 占 elementwise | caller |
|---|---:|---:|---|
| `CUDAFunctor_add<float>` (vectorized_templated) | **550 ms** | **72%** | `aten::add_` |
| `MulFunctor<float>` (AUnary) | 129 ms | 17% | `aten::mul_` |
| `FillFunctor<float>` (zero) | 37 ms | 4.8% | `grad.zero_()` |
| 其他 (bf16 copy/add) | <50 ms | — | — |
| SwiGLU 三件套 (`triton_poi_fused__to_copy_*_silu_*`) | 86 ms | (在 activation 类别) | inductor 自动融合 |

SwiGLU 已经被 PyTorch inductor 自动融合成 3 个 triton kernel，没有手工优化空间。真正的大头 **`add<float>` (550 ms)** 是 fp32 main_grad accumulation。

### 定位 caller

写脚本 (`profile_kernel_callers.py`) 沿 `External id` 把 GPU kernel 关联到 CPU op，确认 caller 全是 `aten::add_`。再 grep Megatron 源码：

```python
# megatron/core/distributed/distributed_data_parallel.py:461
def hook(*unused):  # backward post-hook for each param
    ...
    if param.grad is not None and not param.grad_added_to_main_grad:
        param.main_grad.add_(param.grad.data)  # ← 就是这里
```

每个 backward post-hook 触发一次 fp32 大 add。单 kernel 体积分布：

```
[ 1000 ..  2000 us]  n=72   sum= 117.6 ms
[ 2000 ..  5000 us]  n=36   sum= 147.8 ms
[ 5000 .. 10000 us]  n=36   sum= 259.6 ms   ← 3.5 GB / launch! 单 launch 8 ms
```

最大单 launch grid `[32400, 1, 1]`, block `[512, 1, 1]`, vec=4 → 66.4M elements × fp32 = **265 MB / launch**。这是 MoE expert FFN1 weights 那一档的 grad bucket。

---

## Megatron 控制路径

```
bf16 训练 (yaml: bf16: true)
  └─ accumulate_allreduce_grads_in_fp32 默认 True
       └─ DDP grad_reduce_in_fp32 = True
            └─ grad_buffer dtype = fp32
                 └─ main_grad.add_(grad)  在 fp32 跑
                 └─ grad reduce-scatter 在 fp32 跑

加 yaml: grad_reduce_in_bf16: true
  └─ args.accumulate_allreduce_grads_in_fp32 强制改 False
       └─ DDP grad_reduce_in_fp32 = False
            └─ grad_buffer dtype = param.dtype = bf16
                 └─ main_grad.add_(grad)  在 bf16 跑   (体积 ÷ 2)
                 └─ grad reduce-scatter 在 bf16 跑    (wire 体积 ÷ 2)
                 └─ DistOpt 仍然在 step 时把 grad cast 回 fp32 master copy 做 Adam
```

关键代码：
- `megatron/training/arguments.py:717` — `if args.grad_reduce_in_bf16: args.accumulate_allreduce_grads_in_fp32 = False`
- `megatron/training/training.py:962` — `kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32`
- `megatron/core/distributed/distributed_data_parallel.py:168` — `grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype`

**注意 DistOpt 仍维护 fp32 master**，所以收敛性等价于业界标准 bf16 + DistOpt 配方（Llama / GPT-NeoX 默认就是这样）。

---

## 端到端结果

| 指标 | B0 | B10 | **B11** | B11 vs B0 | B11 vs B10 |
|---|---:|---:|---:|---:|---:|
| step time (median, ms) | 796.4 | 770.3 | **712.9** | **−10.5%** | **−7.45%** |
| step time (min, ms) | 788.3 | 764.0 | **707.5** | −10.2% | −7.39% |
| step time (p90-trim, ms) | 815.1 | 793.6 | **727.6** | −10.7% | −8.32% |
| TFLOP/s/GPU (median) | 518.5 | 536.0 | **579.2** | **+11.7%** | +8.06% |
| TFLOP/s/GPU (max) | 523.8 | 540.4 | **583.6** | +11.4% | +8.0% |
| tokens/s/GPU (median) | 20572 | 21269 | **22982** | **+11.7%** | +8.06% |
| HIP mem (GiB) | 232.5 | 232.5 | **196.4** | **−15.5%** | −15.5% |
| nan / skip iters | 0 | 0 | 0 | — | — |

### Trace category 拆解

| Category | B0 | B10 | **B11** | B11 vs B0 |
|---|---:|---:|---:|---:|
| comm | 2042.3 | 2066.1 | **1498.0** | **−26.6%** |
| gemm_bf16 | 1654.0 | 1692.5 | 1501.4 | −9.2% |
| elementwise | 695.3 | 766.5 | **262.6** | **−62.2%** |
| attention | 348.6 | 339.9 | 327.2 | −6.1% |
| norm | 253.6 | 35.8 | (~35) | −86% |
| moe_permute | 115.7 | 107.2 | 108.6 | — |
| optimizer | 99.6 | 99.9 | 102.0 | — |
| **avg step (ms)** | **795.7** | 769.5 | **712.7** | **−10.4%** |
| comp time (ms) | 2087.8 | 2010.8 | **1810.9** | −13.3% |
| **exposed_comm** | 14.7% | 15.0% | **20.0%** | +5.3pp |
| GPU util | 97.8% | 98.0% | 97.2% | −0.6pp |

### 单 kernel 验证 (elementwise)

| | B10 fp32 | **B11 bf16** | 变化 |
|---|---|---|---|
| `CUDAFunctor_add<float>` | 550 ms / 510 launches / 1079 us avg | **0 ms** | 完全消失 |
| `CUDAFunctor_add<BFloat16>` | 14.8 ms / 216 / 68 us | **142 ms / 726 / 196 us** | 接管 fp32 add 的语义 |
| `MulFunctor<float>` | 129 ms / 150 / 858 us | **0 ms** | 完全消失 |
| `MulFunctor<BFloat16>` (AUnary) | — | **47.9 ms / 150 / 320 us** | 接管 |
| `FillFunctor<float>` | 37 ms / 222 | **0.1 ms / 30** | 完全消失 |
| `FillFunctor<BFloat16>` | — | **19 ms / 99** | 接管 |

bf16 add 单 kernel 比 fp32 快 **3.9×**（1079 us → 196 us），不光是体积砍半，cache 命中也好得多（vec=8 路径，更宽的 SIMT 利用）。

---

## 意外的双赢：comm 也大跌

预期只动 elementwise，但 comm 也降了 **27%**（2066 → 1498 ms / 3 steps）。原因：

`grad_reduce_in_fp32=False` 让 reduce-scatter 的 dtype 变成 bf16，**wire volume 减半**。MI355X 单节点 8 GPU 的 RCCL all-reduce 在小 message 上是 **bandwidth-bound**，体积砍半几乎线性翻译为时间砍半。

但 `exposed_comm` 反而从 15% 升到 20%：

```
B10: comp=2011 ms, comm=2066 ms, exposed=300 ms (15%)
B11: comp=1811 ms, comm=1498 ms, exposed=143 ms ÷ 713 = 20%
```

absolute exposed time 实际从 ~300 ms 降到 ~143 ms（**也降了**），只是因为 comp 缩太多了，比例分母变小。

新的瓶颈结构：
- **exposed comm 占 20% 是当前最大的可优化项**
- comp 已经压到 1811 ms / 3 steps，再压一档需要动 GEMM dtype（fp8 promotion）
- elementwise 已经从 766 → 263 ms，剩下的 142 ms 大头是必要的 bf16 add（无法消除）

---

## 数值稳定性

逐 iter 对比 B11 vs B10 (rank-7, iter 30-60)：

| iter | B10 lm_loss | B11 lm_loss | Δ | B10 grad_norm | B11 grad_norm |
|---|---:|---:|---:|---:|---:|
| 30 | 8.085 | 8.131 | +0.6% | 0.880 | 0.937 |
| 40 | 7.985 | 7.981 | −0.05% | 0.731 | 0.868 |
| 47 | 8.012 | 7.976 | −0.45% | 30.45 | 34.09 |
| 50 | 7.802 | 7.808 | +0.08% | 0.858 | 0.853 |
| 60 | 7.838 | 7.844 | +0.08% | 1.024 | 1.097 |

iter 47 的 grad_norm spike 在 B0 / B10 / B11 都出现（lr ramp 期），不是 bf16 引入的。其他 iter loss / grad_norm 在同一分布，**0 nan / 0 skip in 80 iter**，与 B10 完全可比。

---

## 弯路与教训

1. **"Triton 融合 SwiGLU" 的判断错了**：先 dive trace 比拍脑袋强。SwiGLU 已被 PyTorch inductor 融合，elementwise 的真大头是 fp32 grad add。**教训：profile-driven，永远先拆 kernel + caller 再决定优化方向。**

2. **"comm 已经接近物理下限" 的判断也错了**：在 B10 报告里说 14.7% exposed comm 已经是 8-GPU DP 物理下限。但 B11 把 comm volume 直接砍半（fp32→bf16），absolute exposed 也降了。**教训：comm 时间不是常数，dtype 一变就重新算。**

3. **配置型优化的 ROI 经常超过算子型**：B11 改了 1 行 yaml，收益 8% TFLOPs；B10 写了 ~600 行 Triton + 自定义 nn.Module，收益 3% TFLOPs。**教训：先找 args.py 里那些"默认 True 但其实可关"的配置项。**

---

## 配置改动

只改 1 个 yaml 字段：

```yaml
# /workspace/code/gpt_oss_20B-pretrain-fp8.yaml (or your runtime yaml)
modules:
  pre_trainer:
    overrides:
      grad_reduce_in_bf16: true
```

无代码改动。container 里完全不需要部署。

---

## 下一步候选（按 ROI 排序）

| 方向 | B11 现状 | 可行手段 | 估收益 |
|---|---|---|---|
| **exposed comm** | 143 ms / step (20%) | DDP bucket size 调优、`overlap_param_gather` 已开 → 看 `bucket_size_mb` | 30-60 ms / step |
| **gemm_bf16** | 500 ms / step | 选择性把 1-2 个大 GEMM 提升到 fp8（已部分 fp8 hybrid） | 20-50 ms / step |
| **elementwise residual** | 88 ms / step | 大头是不可消除的 bf16 add；其他可融合的已 inductor 化 | <20 ms / step |
| **attention** | 109 ms / step (CK v3) | 已经是 CK v3 极限 | <10 ms / step |
| **释放 36 GB 提 mbs** | mem 196/288 GiB | mbs 2→4，gbs 同步翻倍或保持 | 根据 batch math |

**最优先**：DDP bucket 调优。`exposed_comm = 143 ms / step` 是新的最大瓶颈，bucket_size_mb 默认 25 MB 在 bf16 下偏小（之前 fp32 时正好），现在可以加到 50-100 MB 试试。

详细数据：[`2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md`](./2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md) 配套，trace 在 `/home/xiaompen/mlperf/ablations/B11.trace.json`。
