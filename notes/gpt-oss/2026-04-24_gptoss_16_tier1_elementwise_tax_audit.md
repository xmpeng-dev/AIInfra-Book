# GPT-OSS-20B EP=1 — Tier 1 elementwise/cast/norm tax 审计 + 优化点定位

**日期**: 2026-04-24
**硬件**: MI355X × 8，TP1 PP1 EP1，DP=8，GBS=32 / MBS=4，FP8 hybrid
**Trace**: `…/run-trace/torch_trace_base_20260423_223318/torchprof/...rank[2].1777002476000591444.pt.trace.json`
**ProfilerStep**: #17（rank 2），1129.33 ms / 32 sample
**关联**: note `15`（trace 校准 + 优化栈重排），yaml/config/patches 现状审计

## TL;DR — 三个发现 + 一个修正

1. **note 15 把"elementwise tax"算大了。** Trace 里 elementwise=160 ms 是**全 stream 求和**；
   stream 0（critical path）上实际只有 elementwise 104.5 + other 53.1 + norm 39.2 = **196.9 ms = 17.4 % step**，
   不是 252 ms / 22 %。Tier 1 的天花板要从 7–9 % 下调到 **5–7 %**。
2. **Tier 1 真正能打的是一个 kernel**：
   `vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>` = **42.4 ms (3.8 % step)** —
   是 attention/MLP 之后的 **residual add**，独立 launch。当前 PrimusTurboRMSNorm 不收 residual。
   **写 fused residual+RMSNorm Triton kernel → 直接砍 ~42 ms ≈ 3.7 %。这是 Tier 1 最大头**。
3. **TE epilogue 那条路已经被 Primus 自己掀掉了**——`PrimusTurboLayerNormColumnParallelLinear`
   patch 显式 trade 掉了 TE 的 norm+GEMM fusion（`one launch -> two launches`），换 Triton RMSNorm
   per-launch 时间。**回去开 TE epilogue 反而会变慢**（patch 自带 microbench：fused TE 1850us vs
   split Triton+TELinear 1784us）。所以 note 15 写的"检查 fuse_wgrad_accumulation /
   return_layernorm_output"这一项**不再适用**——`gradient_accumulation_fusion: true` 已经开了，
   norm-linear 融合是被有意识地放弃的。
4. **`triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` = 30.7 ms (2.7 %)** 已经是
   inductor 融合产物（cast + cat + mul + silu + silu_bwd 五合一）。但是名字里那个 `cat` 暗示
   MoE bwd 在跑 `torch.cat([gate_silu, up])`，**直接写一个吃掉 cat 的 SwiGLU bwd Triton kernel**
   还能拿 5–10 ms。这是 Tier 1 第二档。

修正后的 Tier 1 现实预算：

| 来源 | 现状 | 可拿 | 备注 |
|---|---:|---:|---|
| residual add 融入 RMSNorm | 42 ms 独立 launch | **−40 ms (−3.5 %)** | Triton 改 50 行 |
| SwiGLU bwd 去 cat | 31 ms（已 fused 但带 cat） | **−5~10 ms** | 改 inductor pattern 或写专用 kernel |
| direct_copy / .contiguous() | 22 ms | −5~10 ms | 排查 SBHD↔BSHD、view→contig 调用点 |
| AUnaryFunctor (bf16 unary) | 23 ms | −0~5 ms | 多半是 scale/cast，难单独融 |
| **合计 Tier 1 现实预算** | — | **−50~65 ms ≈ 4.4–5.7 % step** | 比 note 15 估的 7–9 % 低 |

## 一手数据：Top-25 GPU kernel（rank 2，step #17，1129.33 ms）

完整输出在 `/tmp/tier1/breakdown_rank2_step17.txt`。摘要：

| ms | % step | kernel | 类型 |
|---:|---:|---|---|
| 242.08 | 21.4 | `ncclDevKernel_Generic_1` | RCCL DDP（已被覆盖到 80.7%） |
| 238.77 | 21.1 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_..._MT192x192x64_MI16x16x1` | grouped MoE GEMM (副流) |
| 234.28 | 20.7 | `Cijk_Ailk_Bljk_BBS_BH_..._MT256x256x64_MI16x16x1` | grouped MoE GEMM (副流) |
| 185.67 | 16.4 | `Cijk_Alik_Bljk_BBS_BH_..._MT256x256x64_MI16x16x1` | grouped MoE GEMM bwd (副流) |
| 125.91 | 11.1 | `aiter::fmha_bwd_hd64_bf16_causal_a16_rtne_recompile` | FMHA bwd（attention 长杆） |
| 46.54 | 4.1 | `Cijk_Alik_Bljk_F8B8BS_BH_..._MT256x256x128` | dense FP8 GEMM bwd |
| 45.78 | 4.1 | `Cijk_Ailk_Bjlk_BBS_BH_..._MT128x64x32` | dense GEMM |
| 43.96 | 3.9 | `ck_tile::FmhaFwd...gfx950_t` | FMHA fwd |
| **42.36** | **3.8** | **`vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>`** | **bf16 residual add — Tier 1 头号目标** |
| 30.71 | 2.7 | `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` | inductor SwiGLU bwd（已 fused，带 cat） |
| 29.43 | 2.6 | `Cijk_Alik_Bljk_BBS_BH_..._MT32x16x256` | small GEMM |
| 26.90 | 2.4 | `ck_tile::FmhaBw...gfx950_t` | FMHA bwd（次要变体） |
| 26.38 | 2.3 | `_rmsnorm_bwd_kernel` | **Primus Triton RMSNorm bwd（已优化）** |
| 24.19 | 2.1 | `Cijk_Ailk_Bljk_..._MT64x16x128` | small GEMM |
| 23.25 | 2.1 | `Cijk_Ailk_Bjlk_..._MT192x192x32` | small GEMM |
| **23.13** | **2.0** | **`vectorized_elementwise_kernel<AUnaryFunctor<bf16,bf16,bf16,...>>`** | **bf16 unary（scale/cast）** |
| **22.20** | **2.0** | **`elementwise_kernel_manual_unroll<128,8, direct_copy_kernel_cuda>`** | **direct_copy（contiguous/view）** |
| 20.85 | 1.8 | `Cijk_Alik_Bljk_F8BS_BH_..._MT256x256x128` | dense FP8 GEMM |
| 16.79 | 1.5 | `Cijk_Ailk_Bjlk_..._MT256x256x64` | dense GEMM |
| **16.63** | **1.5** | **`multi_tensor_apply_kernel<...AdamFunctor<float,float,...>>`** | **optimizer step（Tier 2）** |
| 16.42 | 1.5 | `Cijk_Alik_Bljk_..._MT16x16x256` | small GEMM |
| 13.18 | 1.2 | `Cijk_Ailk_Bjlk_..._MT128x192x32` | small GEMM |
| 11.43 | 1.0 | `Cijk_Ailk_Bjlk_..._MT64x64x32` | small GEMM |
| 10.92 | 1.0 | `reduce_kernel<128,4, sum_functor<float>>` | dgamma sum（rmsnorm bwd reduction） |
| 10.00 | 0.9 | `_unpermute_kernel` | MoE unpermute（与 permute_fusion 分开） |

### Per-stream 0 类别拆分（关键路径）

```
stream 0 busy 604.70 ms (53.5% step)
  attn_kernel    203.21 ms   ← Tier 4
  gemm           127.23 ms   ← dense fwd+bwd（grouped 在副流）
  elementwise    104.54 ms   ← 主要 Tier 1 肉
  other           53.10 ms   ← autograd / housekeeping
  norm            39.22 ms   ← Triton RMSNorm（已优化）
  moe_dispatch    28.08 ms   ← permute / topk / unpermute
  optimizer       16.63 ms   ← Tier 2
  reduction       16.01 ms   ← dgamma + 其它 sum
```

stream 0 上 elementwise + other + norm = **196.86 ms ≈ 17.4 % step**（修正 note 15 的 22 %）。

## 配置审计：哪些 fusion 已经开了，哪些还没

### `gpt_oss_20B-pretrain-fp8.yaml` 关键开关

| 项 | 当前值 | 含义 | 评估 |
|---|---|---|---|
| `gradient_accumulation_fusion` | **true** | TE Linear 的 `fuse_wgrad_accumulation=True` | ✅ 已开（note 15 这一项 obsolete） |
| `apply_rope_fusion` | **true** | RoPE Triton 融合 | ✅ |
| `moe_permute_fusion` | **true** | Megatron permute fusion | ✅ |
| `moe_router_fusion` | **true** | router topk + softmax 融合 | ✅ |
| `moe_grouped_gemm` | **true** | 走 grouped GEMM | ✅ |
| `moe_use_legacy_grouped_gemm` | **true** | legacy 路径，由 PRIMUS_TRITON_GG_PATCH 接管 | ✅（间接） |
| `cross_entropy_loss_fusion` | **true** | TE fused CE | ✅ |
| `cross_entropy_fusion_impl` | te | TE 实现 | ✅ |
| `enable_primus_turbo` | **true** | 总开关 | ✅ |
| `use_turbo_rms_norm` | **true** | → PrimusTurboRMSNorm (Triton) + LayerNormColumnParallelLinear 替换 | ✅（详见下） |
| `use_turbo_attention` | false | 跳过 turbo attention | 故意（SWA 兼容） |
| `use_turbo_grouped_mlp` | false | 跳过 turbo GG | 故意（让 PRIMUS_TRITON_GG patch 接管） |
| `grad_reduce_in_bf16` | **true** | bf16 grad reduce（B11 已落） | ✅ |
| `ddp_pad_buckets_for_high_nccl_busbw` | **true** | DDP bucket pad | ✅ |
| `overlap_grad_reduce` / `overlap_param_gather` | **true** | DDP 重叠 | ✅ |
| `ddp_average_in_collective` | (未设) | note 12 B1，应为 true | ⚠️ regress（note 14 已 flag） |
| `ddp_bucket_size` | (未设) | note 12 B2，应为 100_000_000 | ⚠️ regress |

### Shell config (`config_MI355X_..._gbs32_fp8.sh`) 关键 env

```
PRIMUS_FP8_RECIPE=hybrid
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
PRIMUS_GRAD_REDUCE_IN_BF16=true
USE_TURBO_RMS_NORM=true

# Triton GG 全开
PRIMUS_TRITON_GG_PATCH=1
PRIMUS_TRITON_GG_FORCE_TRITON=1
PRIMUS_TRITON_GG_V2=1
PRIMUS_TRITON_GG_EP1_HEURISTIC=0
PRIMUS_TRITON_GG_EP1_ONLY=1

# NVTE FP8 cast transpose 优化
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
NVTE_USE_CAST_TRANSPOSE_TRITON=0

# FMHA 后端
NVTE_FMHA_BACKEND_DENSE_BWD=ck_v3
NVTE_CK_USES_BWD_V3=0  # global off, per-call patch 打开
```

绝大多数已知融合**都已经开了**。

### `patches/turbo_rms_norm.patch` 现状（关键引文）

```python
# PrimusTurboRMSNorm.forward
def forward(self, x):
    gamma = self.weight
    if getattr(self, "zero_centered_gamma", False):
        gamma = gamma + 1
    return _triton_rmsnorm(x, gamma, self.eps)   # ← 不收 residual
```

```
# PrimusTurboLayerNormColumnParallelLinear.__doc__
Trade-off vs the fused TE kernel:
  * Loses TE's "norm + GEMM" fusion (one launch -> two launches).
  * Gains the Triton fwd / bwd kernels which are 2-3x faster per launch
    on the GPT-OSS-20B shapes (see PrimusTurboRMSNorm docstring) and a
    cheap fp32 sum(dim=0) dgamma reduction that replaces TE's
    slow rmsnorm_bwd_finalize_general_kernel.
  * Net: -188.9 ms RMSNorm GPU time / 3 steps (B0 vs B8) measured
    end-to-end, plus another ~13 ms / 3 steps freed by routing this
    site too.
```

→ 结论：**RMSNorm 路径已经被仔细调过**，进一步优化必须从"功能扩展"入手（吃掉外部的 add/cast），
不能从"换实现"入手。

### `patches/triton_grouped_gemm_v2.py` 现状

代码量 735 行，但功能上：
- monkeypatch `MoE.experts.GroupedMLP.forward`，在 forward 前后采样 shape、强制走 Triton 后端、
  挂 EP1 heuristic。
- **不**做 permute→cast→GG fusion。permute 仍在前一个 kernel，cast 仍在 TE 内部，
  GG 走 Triton。

→ 结论：note 15 写的"PRIMUS_TRITON_* 是否覆盖 permute → cast → grouped_gemm chain？"答案是
**没有**。但 trace 显示 moe_dispatch 总共只有 28 ms，permute/unpermute 加起来也就 ~10 ms，
这个 chain 的 fusion 即使做完上限也只有 ~10 ms。**ROI 不如打 residual add**。

## Tier 1 优化点：按 ROI 排序

### A. **fused residual + RMSNorm Triton kernel**（−3.5 %，**MUST DO**）

#### 现状

`vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>` = 42.36 ms / step。在 24-layer GPT-OSS-20B
里每层 fwd 2 个 + bwd 2 个 add = 96 次，trace 显示每次 ~440 us 在 stream 0 上独立 launch。

调用 site（推断 + 需 `with_stack` trace 复核）：
- **fwd**: `attn_out → residual_add → rmsnorm → mlp/moe_in`（24 层 × 2 个 = 48 次）
- **bwd**: 对应 grad path 上的 add（48 次）

PrimusTurboRMSNorm 当前签名 `forward(self, x)` 不收 residual。

#### 改动

1. **kernel 层** (`patches/triton_rmsnorm.py`):
   ```python
   @triton.jit
   def _rmsnorm_fwd_kernel_with_residual(
       X_ptr, R_ptr, G_ptr, Y_ptr, X_PLUS_R_ptr, RSTD_ptr, ...,
       HAS_RESIDUAL: tl.constexpr,
   ):
       x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
       if HAS_RESIDUAL:
           r = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
           x = x + r
           tl.store(xpr_ptrs, x.to(...), mask=mask)   # save x+r for resid path
       var = tl.sum(x * x, axis=0) / H
       ...
   ```
   bwd 同理（residual grad = grad_x，不需要额外 dy）。

2. **module 层** (`PrimusTurboRMSNorm.forward`):
   ```python
   def forward(self, x, residual=None):
       gamma = self.weight + (1 if self.zero_centered_gamma else 0)
       if residual is None:
           return _triton_rmsnorm(x, gamma, self.eps)
       y, x_plus_r = _triton_rmsnorm_with_residual(x, residual, gamma, self.eps)
       return y, x_plus_r
   ```

3. **调用点接入**：Megatron 的 transformer block 在 `mcore/transformer_block.py` 里走
   `pre_mlp_layernorm`、`pre_attention_layernorm` 等 hook。不需要改 Megatron，写一个
   `patches/megatron_fused_residual_norm.patch` monkeypatch 这两个 hook 即可，类似
   `turbo_rms_norm.patch` 的写法。或者更简洁的做法：把 add 合并放在 `TransformerLayer.forward`
   一处，挂在 `PrimusTurboRMSNorm` 上。

#### 期望

- 删除 42.4 ms add → ~0 ms（融在 norm 内部，shared-mem 复用 x）
- bwd add 也类似（~部分 in 30 ms 的 fused silu chain 之外，~6-10 ms）
- **保守估 −38 ms ≈ −3.4 % step**
- **乐观估 −48 ms ≈ −4.3 % step**（含 bwd 那部分）

### B. **SwiGLU bwd 去 cat**（−0.5~1 %，二档）

#### 现状

`triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` = 30.71 ms。
名字里 `cat` 暗示 inductor 把 `torch.cat([d_gate, d_up], dim=-1)` 编进了 SwiGLU bwd。
这个 cat 写出会有 ~30% 的额外 store traffic（因为最后还得拆开喂给 grouped GEMM bwd）。

#### 改动方向

- 选项 1：直接写一个 Primus Triton kernel 替换 inductor 产物，输出两个独立 tensor
  （`d_gate`, `d_up`）而不是 cat 后 split。
- 选项 2：让上游 grouped GEMM bwd 直接吃 cat 后的 buffer（避免再 split）。
- 调用点：Megatron MoE expert MLP 的 bwd 路径，`primus_turbo` 的 GroupedMLP 模块。

#### 期望

- −5 ~ −10 ms ≈ −0.5–0.9 % step
- 工作量 1–2 天

### C. **`direct_copy_kernel_cuda` 排查**（−0.5–1 %）

#### 现状

22.2 ms 在做 `.contiguous()` / view-to-contig。来源高度怀疑：

- `megatron_te_bshd_layout.patch` 已经把 SBHD→BSHD 转换收掉了一部分，但可能还有残留。
- MoE permute 后的 `permuted_local_hidden_states` 经常被 `.contiguous()` 一次。
- DDP bucket pack/unpack 可能在拷贝 grad bf16 cast 后的 buffer。

#### 改动方向

- 跑一份带 `record_shapes=True, with_stack=True` 的短 trace（5 step），grep
  `aten::contiguous` 的 caller 链，定位 top-3 site。
- 多数 case 可以通过 stride-aware kernel 直接消化 strided input，省掉 `.contiguous()`。

#### 期望

- −5 ~ −10 ms ≈ −0.5–0.9 % step
- 工作量 1 天定位 + 1 天改

### D. **`AUnaryFunctor<bf16,bf16,bf16, ...>`**（−0–0.4 %）

#### 现状

23.13 ms，pattern `bf16 → bf16 → bf16`，3 个相同 dtype。最可能是：

- `attention_softmax_in_fp32: false` 路径上的 bf16 scale `q * (1/sqrt(d))`
- 或 RoPE 后某个 element-wise mul
- 或 grad reduce 路径的 scale

#### 改动方向

- 单独融到上游 GEMM 的 epilogue 工程量大，ROI 低。
- **建议先放着**，等 A+B+C 落完再看排序。

## TE epilogue 这条路为什么放弃

note 15 写的 "检查 fuse_wgrad_accumulation / return_layernorm_output 把 cast→silu→matmul 收进
TE epilogue" 这条路在本审计里**已经被否决**：

1. `gradient_accumulation_fusion: true` 已经开 → TE Linear 的 `fuse_wgrad_accumulation=True`
   是激活的。
2. `return_layernorm_output` 是 TE LayerNormLinear fused 路径的一部分；但
   `PrimusTurboLayerNormColumnParallelLinear` patch **显式拆掉**了这个 fusion
   （`one launch -> two launches`），换 Triton RMSNorm。patch 的 microbench 表明拆开比融合**快 4 %**：
   ```
   fused TE (RMS) FP8 fwd+bwd  : 1850 us
   Triton + TELinear FP8 f+b   : 1784 us
   ```
3. `cast → silu → matmul` 这条 chain 在 GPT-OSS-20B 里：
   - dense MLP 走 `gate_proj/up_proj`（TE Linear）→ silu → `down_proj`（TE Linear），
     silu 已经被 TE 的 swiglu kernel 吸进去了。
   - MoE expert 走 grouped GEMM → silu → grouped GEMM；silu 体现在 inductor 的
     `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` 里（已 fused）。

→ 没有"裸的" cast→silu→matmul 三连暴露在 trace 里。**这一条 obsolete，从 Tier 1 拿掉**。

## RMSNorm residual 这一条是不是已经覆盖了

note 15 写的 "primus.rms_norm 是否带 residual 参数？没开就开"——审计结论：

**没带，也没"开关"可开。** 当前 `PrimusTurboRMSNorm.forward(self, x)` 只接受单 input，
没有 residual 形参。**必须新增 kernel + 改 module 接口 + 改调用点**才能拿到这部分收益。
属于 **A 项的具体落地动作**。

## 修正后的执行顺序

```
Day 0–1 (本 note 已完成)
  [√] Top-25 GPU kernel + per-stream 拆分
  [√] yaml/config/patches 现状审计
  [√] Tier 1 重新预算

Day 1–3
  [ ] A. fused residual + RMSNorm Triton kernel（fwd+bwd 各加 2 个 kernel）
        - 写 kernel
        - 写 module patch
        - 写 monkeypatch hook
        - 80-iter A/B 验证 ≥ −3.0 % step

Day 4
  [ ] C-定位. 跑 1 个 with_stack=True 短 trace，定位 direct_copy 来源 top-3

Day 5–6
  [ ] B. SwiGLU bwd 去 cat（如果 A 落地干净，再做 B）
  [ ] C. direct_copy 消除（若定位结果可行）

Day 7
  [ ] Tier 1 整体回归：80-iter 长 A/B vs run_base，目标 −5 % step

Day 8+
  [ ] 进入 Tier 2 (HIP graph optimizer step)
```

## 修正后的 Tier 1 现实预算（替换 note 15 §"Tier 1"）

| 项 | 工程量 | 期望 step Δ |
|---|---|---:|
| A. fused residual + RMSNorm | 2-3 天 | **−3.4 ~ −4.3 %** |
| B. SwiGLU bwd 去 cat | 1-2 天 | −0.5 ~ −0.9 % |
| C. direct_copy 消除 | 2 天 | −0.5 ~ −0.9 % |
| D. AUnaryFunctor 融合 | 暂缓 | 0 ~ −0.4 % |
| **合计 Tier 1** | **5–7 天** | **−4.4 ~ −6.5 % step** |

锚到绝对值：1129 ms → **1056–1078 ms / step**（vs note 15 估 990 ms）。
对应 E2E 7680 iter ≈ **8100–8280 s**（vs baseline 9777 s, **−15~17 %**）。
note 15 的"−12~13 % step"目标稍偏乐观，**修正为 −10~12 %**（含 Tier 1 + Tier 2 + B1+B2 regress 修复）。

## 文件 / 命令复现

```bash
# 一手 trace 解析（0.8s）
cd /home/xiaompen/mlperf-training/b200 && \
  python3 full_breakdown.py \
    "/home/xiaompen/mlperf-training/small_llm_moe_pretraining/primus/run-trace/torch_trace_base_20260423_223318/torchprof/primus-megatron-exp[gpt_oss_20b]-rank[2].1777002476000591444.pt.trace.json" \
    ProfilerStep#17 > /tmp/tier1/breakdown_rank2_step17.txt
```

- 输出（206 行）：`/tmp/tier1/breakdown_rank2_step17.txt`
- yaml: `small_llm_moe_pretraining/primus/gpt_oss_20B-pretrain-fp8.yaml`
- shell config: `small_llm_moe_pretraining/primus/config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh`
- 关键 patch 1: `small_llm_moe_pretraining/primus/patches/turbo_rms_norm.patch`（348 行）
- 关键 patch 2: `small_llm_moe_pretraining/primus/patches/triton_rmsnorm.py`（240 行）
- 关键 patch 3: `small_llm_moe_pretraining/primus/patches/triton_grouped_gemm_v2.py`（735 行）
- 关联 notes: `15`（trace 校准），`14`（HSDP 负 + B1+B2 regress flag），`12`（comm B 系列）
