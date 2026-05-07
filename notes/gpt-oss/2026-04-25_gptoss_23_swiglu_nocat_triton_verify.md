# GPT-OSS-20B / MI355X — Tier 1B Verification: Triton MoE SwiGLU no-cat backward

**日期**：2026-04-25
**仓库**：`/home/xiaompen/mlperf-training/small_llm_moe_pretraining/primus`
**前情**：
- [gptoss_15](2026-04-24_gptoss_15_ep1_trace_optimization_plan.md) 把 EP=1
  trace 上 stream 0 的 bwd "elementwise tax" 列成 Tier 1 候选；
- [gptoss_16](2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md) 把这条 tax
  拆成两个具体 kernel：`vectorized_elementwise<CUDAFunctor_add<bf16>>`
  (~42 ms, 残差加) 和
  `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` (~30 ms,
  GroupedMLP `chunk + silu + mul + cat/split` 的 Inductor 复合体)；
- [gptoss_18](2026-04-24_gptoss_18_fused_residual_rmsnorm_verify.md) /
  [gptoss_19](2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md)
  把 Tier 1A（fused residual+RMSNorm V1/V2）签字，留下来的就是这条 SwiGLU 的
  cat/split 链路。

本 note 给 Tier 1B 收尾：用一对 Triton kernel 把 SwiGLU + probs 的
fwd/bwd 直接写进 `[N, 2H]` 的 dx，不再落 `cat`，也不再被 Inductor 缝合成
那个长名字的 `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1`。

> 备注：本 note 占用 gptoss_23 槽位，因为 22 已经被
> [`2026-04-25_gptoss_22_sync_free_moe_stage_audit.md`](2026-04-25_gptoss_22_sync_free_moe_stage_audit.md)
> 占用。

---

## 0. TL;DR

| 指标 | base (NOCAT=0) | Triton no-cat (NOCAT=1) | Δ |
|---|---:|---:|---:|
| **ProfilerStep#17 wall** | 1125.03 ms | **1101.88 ms** | **-23.15 ms (-2.06 %)** |
| **stream 0 busy**        |  599.89 ms |  **575.92 ms** | **-23.97 ms (-4.0 %)** |
| **stream 0 elementwise** |  102.01 ms |   **77.93 ms** | **-24.08 ms (-23.6 %)** |
| **elementwise category** |  159.12 ms |  **133.72 ms** | **-25.40 ms (-16.0 %)** |
| **trainer per-iter @ 20**| 1200.3 ms  | **1165.1 ms**  | **-35.2 ms (-2.93 %)** |
| **TFLOP/s/GPU @ iter 20**|  688.0     |  **708.8**     | **+20.8 (+3.0 %)** |
| **lm_loss @ iter 20**    |  9.385678  |  9.385141      | -5.4e-4 (bf16 噪声) |
| **grad_norm @ iter 20**  |  1.685     |  1.686         | +1e-3 (bf16 噪声) |
| NaN / skipped iters      | 0 / 0      | 0 / 0          | — |
| Microbench fwd+bwd ([16384, 8192] bf16) | 0.852 ms | **0.349 ms** | **2.44×** |
| Microbench bf16 ULP error | — | 0.0625 (1 ULP) | OK |

**结论**：

1. ✅ **trace 端实锤**：`triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1`
   30.6 ms 在 Top-25 中**消失**，被新 `_swiglu_probs_bwd_kernel` (15.6 ms)
   取代；stream 0 上 elementwise 类下降 -23.6 %。
2. ✅ **wall 实锤**：trace 内 ProfilerStep#17 -2.06 %，trainer 自报每 iter
   -2.93 %（trainer 用的窗口比 trace step 大一点，差值合理）。
3. ✅ **数值在 bf16 噪声带内**：iter 20 lm_loss / grad_norm 与 base 相差
   `5e-4 / 1e-3` 量级，0 NaN / 0 skipped。
4. ✅ **回滚零成本**：env `PRIMUS_MOE_SWIGLU_NOCAT=0` 即恢复原路径，patch 在
   `gated_linear_unit && activation_func is F.silu` 之外自动 no-op。

→ 已经把
[`config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh`](../../../small_llm_moe_pretraining/primus/config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh)
里的默认值翻成 `:-1`。

---

## 1. 解决的问题

GroupedMLP 默认的 `activation_func_with_probs` 走的是 PyTorch 端
`gate, up = torch.chunk(x, 2, dim=-1)` → `F.silu(gate) * up * probs`。

正向 Inductor 还能把它折成一两个 kernel；但是反向被缝成一个超长名字：

```
triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1
```

这个 kernel 有几个特点（参见 gptoss_16 §2 / §3.2）：

- 名字里的 `cat`/`split` 揭示它在内部做了 `torch.cat([d_gate, d_up], dim=-1)`
  的物化 —— 即在反向再走一次 `[N, H] || [N, H] → [N, 2H]` 的 IO；
- 完全跑在 stream 0；
- 24-iter trace 的稳态步内 **总共 30.6 ms**，是 stream 0 上 elementwise 类
  里**最大**的单 kernel；
- 与 `chunk` 在 fwd 端制造的若干 `direct_copy_kernel_cuda` /
  `vectorized_elementwise<AUnaryFunctor>` 联动，是 EP=1 路径上 stream 0 排
  bwd 抽不出空闲的主因之一。

我们要做的就是**让 dx 一次写完 `[N, 2H]`，不再落 cat**。

---

## 2. 实现

### 2.1 Triton kernel

`patches/moe_swiglu_nocat.py` 内新增两个 JIT kernel：

- `_swiglu_probs_fwd_kernel`：每个 program 处理一行 `[H]`，从 `[N, 2H]` 的 X
  里 stride-aware 地分别 load `gate = X[:, :H]` 和 `up = X[:, H:]`，再 load
  `probs[N]`，算 `silu(gate) * up * prob` 并写 `Y[N, H]`。fp32 中间，
  bf16 落盘。
- `_swiglu_probs_bwd_kernel`：同样按行 program，重算 `sig = sigmoid(gate)`、
  `silu = gate * sig`、`d_silu = sig * (1 + gate * (1 - sig))`，
  然后**直接把** `d_gate`、`d_up` 写到 `DX[:, :H]` / `DX[:, H:]`（对，就是同
  一个 `dx` buffer 的两半），再做行内 `tl.sum` 算 `d_probs`，写 `DP[N]`。
  没有任何 `torch.cat` / `torch.split`，**全程不在 stream 0 上多分配中间
  buffer**。

`BLOCK_H` 用 `_pick_config(H)` 按 next-pow2 选：

| H 区间 | BLOCK_H | num_warps | num_stages |
|---|---:|---:|---:|
| ≤ 256 | next_pow2(H) | 4 | 2 |
| ≤ 1024 | next_pow2(H) | 8 | 2 |
| > 1024 | next_pow2(H) | 8 | 3 |

GPT-OSS-20B 的 expert hidden 是 `H=4096`（即 `[N, 8192]`），落到第 3 档。

### 2.2 autograd 接口

`_SwiGLUWithProbsNoCatFn(torch.autograd.Function)`：

- `forward(x, probs)`：保存 `(x, p)`，记录 `(block_h, num_warps, num_stages,
  probs_shape, probs_dtype)` 到 `ctx`，启 `_swiglu_probs_fwd_kernel`。
- `backward(grad_out)`：分配 `dx = empty_like(x)` (`[N, 2H]`)、
  `dp = empty(N)`，启 `_swiglu_probs_bwd_kernel`，返回 `(dx, dp.view(probs_shape).to(probs_dtype))`。
- 非 CUDA 时落到纯 PyTorch fallback，仅供单测/排查；训练路径必走 Triton。

### 2.3 安装钩子

`install()` monkeypatch `megatron.core.transformer.moe.experts.GroupedMLP.__init__`：

```python
def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    cfg = getattr(self, "config", None)
    if cfg is None: return
    if not getattr(cfg, "gated_linear_unit", False): return
    if getattr(cfg, "activation_func", None) is not F.silu: return
    self.activation_func_with_probs = _swiglu_with_probs_nocat
```

三个 guard：

1. `cfg is None` → 不是我们要 patch 的 GroupedMLP，跳过；
2. `gated_linear_unit == False` → 不是 GLU，本 kernel 不适用；
3. `activation_func is not F.silu` → 不是 SiLU（比如 GELU/SwiGLU 变体）也跳。

只要任一不满足，patch silent no-op，跑回 Megatron 原 path。

`install()` 入口由 `train.py:install_runtime_patches()` 调用，env 关掉时
（`PRIMUS_MOE_SWIGLU_NOCAT=0`）`install()` 直接 `_log("disabled ...")` 返回
`False`，不动 GroupedMLP 的 `__init__`。

---

## 3. Microbench

跑 `python3 patches/bench_moe_swiglu_nocat.py`（容器内），形状取
`x.shape=[16384, 8192]`、`probs.shape=[16384, 1]`、bf16：

```
max|y_base - y_nocat|  = 0.062500   # 1 bf16 ULP @ |y|≈O(1)
max|dx_base - dx_nocat|= 0.062500
fwd+bwd per iter: baseline=0.852 ms, nocat=0.349 ms, speedup=2.443x
```

ULP 误差：`0.0625 = 2^-4`，是 bf16 在 |y|≈1 量级的单步舍入幅度。fwd 出与
dx 都打到这个量级，没有出现幅度级别的偏差。

---

## 4. 端到端 trace A/B

两条同型 24-iter run，唯一差异是 `PRIMUS_MOE_SWIGLU_NOCAT`：

- base：`PRIMUS_MOE_SWIGLU_NOCAT=0`
  → trace `output/.../tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].1777109104088101229.pt.trace.json` (09:25 UTC)
- triton：`PRIMUS_MOE_SWIGLU_NOCAT=1`
  → trace `output/.../tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].1777108565684118753.pt.trace.json` (09:16 UTC)

复用 `.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py`，窗口
`ProfilerStep#17`（trace 内的第一次 #17，覆盖一个完整 fwd+bwd+optimizer step）。

### 4.1 Per-stream busy time（stream 0 = 计算主流）

| stream | base busy | triton busy | Δ |
|---|---:|---:|---:|
| 0 (compute main) | 599.89 ms | **575.92 ms** | **-23.97 ms (-4.0 %)** |
| 11 (NCCL) | 227.39 | 236.10 | +8.71 |
| 13 (副 GEMM) | 210.58 | 211.43 | +0.85 |
| 14 (副 GEMM) | 221.40 | 214.29 | -7.11 |
| 15 (副 GEMM) | 181.98 | 193.00 | +11.02 |
| 16 (副 GEMM) | 207.54 | 216.76 | +9.22 |
| 4 (其他)   |  57.11 |  55.80 |  -1.31 |

stream 0 单流减 24 ms，副 stream 总和 +22 ms（NCCL/副 GEMM 因为 stream 0
腾出空当被排得更紧），ProfilerStep wall 因此净减 23.15 ms。

### 4.2 GPU kernel 类别 breakdown

| 类别 | base ms | triton ms | Δ ms | Δ% |
|---|---:|---:|---:|---:|
| gemm | 948.37 | 962.40 | +14.03 | +1.5 % |
| attn_kernel | 201.28 | 202.70 | +1.42 | +0.7 % |
| **elementwise** | **159.12** | **133.72** | **-25.40** | **-16.0 %** |
| nccl_generic | 117.95 | 127.07 | +9.12 | +7.7 % |
| nccl_ag | 96.96 | 97.22 | +0.26 | +0.3 % |
| other | 51.22 | 50.92 | -0.30 | -0.6 % |
| norm | 41.74 | 40.85 | -0.89 | -2.1 % |
| moe_dispatch | 27.95 | 27.96 | +0.01 | — |
| optimizer | 16.73 | 16.52 | -0.21 | -1.3 % |
| reduction | 15.94 | 15.81 | -0.13 | -0.8 % |
| memcpy | 11.88 | 11.14 | -0.74 | -6.2 % |

`gemm` / `nccl` 微微涨是因为副 stream 排得更紧（参见 §4.1），这部分是
**被隐藏在 wall 里的 overlap 增益**，不会反映到 wall。`elementwise` 直接
-25.4 ms，**就是这一发 patch 全部正面收益的来源**。

### 4.3 Top kernel 对比（stream 0 上的 silu/cat/split 链路）

| kernel | base ms | triton ms | Δ |
|---|---:|---:|---:|
| `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` | **30.61** | **不在 top-25 (≤ 18)** | **消失** |
| `_swiglu_probs_bwd_kernel` (新增) | — | **15.64** | +15.64 |
| `_rmsnorm_bwd_residual_kernel` | 24.30 | 23.38 | -0.92 |
| `vectorized_elementwise<CUDAFunctor_add<bf16>>` (residual ADD) | 40.69 | 39.90 | -0.79 |
| `vectorized_elementwise<AUnaryFunctor<bf16>>` | 23.97 | 23.27 | -0.70 |
| `elementwise_kernel_manual_unroll` direct_copy | 22.35 | 22.18 | -0.17 |

净账：**30.61 (旧 cat/split kernel)** − **15.64 (新 _swiglu_probs_bwd)** =
**-15.0 ms** 在 stream 0 上的 GLU 反向单 kernel；其余 -10.4 ms elementwise
来自原本伴随 cat 的 `chunk` / `direct_copy` 短 kernel 不再起。

### 4.4 trainer 自报指标（同一 24-iter run，iter 20 行）

| | base | triton no-cat | Δ |
|---|---:|---:|---:|
| elapsed time per iter (ms) | 1200.3 | **1165.1** | **-35.2 (-2.93 %)** |
| TFLOP/s/GPU | 688.0 | **708.8** | **+20.8 (+3.0 %)** |
| tokens/s/GPU | 27298.9 | **28124.2** | **+825 (+3.0 %)** |
| lm_loss | 9.385678 | 9.385141 | -5.4e-4 |
| grad_norm | 1.685 | 1.686 | +1e-3 |
| NaN / skipped iters | 0 / 0 | 0 / 0 | — |

trainer 自报 -35.2 ms / iter > trace 单 step -23.2 ms 是因为 trainer 的
"elapsed time" 包含 Python 端 host overhead 和 grad-sync 等待，trace 的
ProfilerStep 只算 GPU 那一段；patch 减掉的 stream 0 占用顺带让 host 等
GPU 的时间也缩短了。

### 4.5 与之前 Python autograd 版的对照

之前 [gptoss_16 §6.3] 试过纯 Python `torch.autograd.Function` 写法，trace 反
**回归** +91.6 ms stream 0 / +86.2 ms elementwise / +8.1 % wall（dispatcher
overhead 抵消了 cat 节省）。这次走 Triton kernel，把 dispatcher 完全压
进一个 GPU launch，才把 Python 版本反向退化的部分翻回来再加 -2 % 的真收
益。**所以本 patch 必须走 Triton/C++ kernel 级别，不能停在 Python autograd**
（已经废弃 Python 版本，仅留 CPU fallback 用于本机 debug）。

---

## 5. 默认开启与回滚

### 5.1 默认开启

`config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh`:

```bash
export PRIMUS_MOE_SWIGLU_NOCAT="${PRIMUS_MOE_SWIGLU_NOCAT:-1}"
```

### 5.2 回滚（不动 patch 文件）

```bash
export PRIMUS_MOE_SWIGLU_NOCAT=0   # 下次 train.py 启动生效，本进程不变
```

`install()` 在 `_enabled() == False` 时直接打 `[moe_swiglu_nocat] disabled
(set PRIMUS_MOE_SWIGLU_NOCAT=1 to enable)` 后退出，**不修改** GroupedMLP，
**不增加任何 GPU launch**。

### 5.3 自适应 no-op

即便 env=1，下列任一情况 patch 也会自动 no-op（fallback 到 Megatron 原路径）：

- `cfg is None`（不是 GroupedMLP 而是其他实现）
- `cfg.gated_linear_unit == False`
- `cfg.activation_func is not F.silu`（比如 GELU、SwiGLU 变体）

所以 patch 在非 GLU + SiLU 的模型上是绝对无害的，可以放心默认开。

### 5.4 硬下线（永久禁用）

把 `src/train.py` 里的：

```python
import moe_swiglu_nocat
moe_swiglu_nocat.install()
```

注释掉即可。

---

## 6. 残余风险

| 风险 | 现状 | 处置 |
|---|---|---|
| 长跑收敛 | 24-iter smoke 显示 lm_loss 与 base 同噪声带，但没跑到 800-iter 收敛窗 | 与 Tier 1A V1/V2 合并跑一次 800-iter A/B 时一并签字（待办） |
| 形状外延 | 仅在 `H=4096`（GPT-OSS-20B expert hidden）和 16k 行规模做了 microbench | `_pick_config` 已分 3 档；未覆盖到的 H 自动选最大档，BLOCK_H 用 next-pow2，最差是 occupancy 不满，不会算错 |
| 与 Tier 1A V2 的相互影响 | 已经在与 V1 同时开启的状态下验证；V2 默认仍关 | 等 V2 默认开时再补一发 1A_V2 + 1B 同开的 trace |
| Inductor 路径与本 patch 的耦合 | 现在我们用自己的 Triton kernel 顶替 Inductor 缝合体；如果未来 PyTorch / Megatron 改了 GroupedMLP 接口，patch 会 import 失败但 try/except 会 swallow，训练继续走原路径 | 加 install 失败的告警（已经有 `[MLPerf Train] moe-swiglu-nocat patch install skipped: ...`） |

---

## 7. 留下来的可复用脚手架

- `patches/moe_swiglu_nocat.py`：Triton kernel + autograd + install hook
- `patches/bench_moe_swiglu_nocat.py`：单 GPU microbench（数值 + 速度）
- 默认行为由 `config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh` 控制
- trace 复盘：复用 `.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py`

---

## 8. 一行小结

> Triton 版 MoE SwiGLU no-cat backward 把 stream 0 上 30.6 ms 的
> `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` 替换成
> 15.6 ms 的 `_swiglu_probs_bwd_kernel`，trace `ProfilerStep#17` -2.06 %
> wall / -4.0 % stream 0 / -16 % elementwise，trainer per-iter -2.93 % /
> +3.0 % TFLOP·s⁻¹·GPU⁻¹，loss 与 grad_norm 在 bf16 噪声带内，0 NaN /
> 0 skipped。默认值已翻成 `PRIMUS_MOE_SWIGLU_NOCAT=1`，env=0 一键回滚。
