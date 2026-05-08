# Tier 1A FP8 epilogue fuse — 调研 verdict & 重新规划

**日期**: 2026-05-07
**目标**: 把 trace 里 71.6 ms / step (7.7%) 的 FP8 量化链 (cast + amax + scale) 下推到 hipBLASLt epilogue
**结果**: **原始计划失效, 但发现的真相反而更清楚**。这条路有一半 TE 已经做了, 另一半不是 hipBLASLt epilogue 问题, 是 `fp8_recipe` 选型问题。
**关联**: [note 28](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md) §6 Tier 1A、[note 29](./2026-05-07_gptoss_29_baseline_redo_clean_window.md)

## TL;DR

原本以为是: `bf16 GEMM 输出 → reduce_row(absmax) → unary cast → fp8` 这条链可以全部塞到 hipBLASLt epilogue 里, 一锤定音省 50 ms。**但代码读完发现完全不是这样**：

1. **TE 的 ROCm GEMM (rocm_gemm.cu:1111) 已经使用 `HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER`** —— 也就是 GEMM **输出** D 的 amax 早就 fuse 进 hipBLASLt epilogue 了, trace 里**根本看不到**"GEMM 输出 amax"这条 kernel。
2. **trace 里 71.6 ms 的 `unary_kernel` / `reduce_row_kernel` / `amax_kernel`** 全部是 GEMM **输入侧** prep —— `quantize_fp8(a, ...)` 在每个 GEMM **之前**重新算 input amax + cast 到 fp8。**输入侧的 amax 不能塞进同一个 GEMM 的 epilogue (因为还没开始算)**, 所以 hipBLASLt epilogue 这条路对这 71.6 ms 不适用。
3. **真正能拿的肉**是切 `fp8_recipe`：当前 yaml 用 `tensorwise` (current scaling, 每步重算 input amax), Primus 默认 `delayed` (用历史 amax 跨步复用)。**但 yaml 注释链显示 `use_turbo_parallel_linear` 只支持 `tensorwise/blockwise/mxfp8`, 不支持 `delayed`** —— 切到 delayed 会失去 turbo Linear 的其他优化, 净收益不一定为正。
4. **最低成本验证**：跑一次 24-iter A/B (yaml 把 `fp8_recipe: tensorwise` 改成 `delayed`), 看 step time 实测变化 ± kernel breakdown。如果 delayed 下 amax/cast 类 kernel 消失但 step 反而变慢 ≥ 5 ms, 就坐实 turbo Linear 在其他维度的收益更大, **Tier 1A 在不动 primus_turbo 源码情况下没有低成本入口**, 直接降级。

**当前 verdict**: ⚠️ **partial / blocked**. 不动源码做 A/B 验证, 决定是否值得做 Plan B (给 primus_turbo Float8Linear 加 delayed scaling 支持, 这是中等工程量)。

## 1. 调研事实

### 1.1 ROCm / hipBLASLt 的 FP8 epilogue 支持矩阵 (ROCm 7.2.0)

| 能力 | 支持 | API |
|---|---|---|
| GEMM **输出** D_amax 写到指定 buffer (在 epilogue 里算) | ✅ | `HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER` (`hipblaslt.h:1111`) + `setAmaxD()` (`hipblaslt-ext.hpp:281`) |
| FP8 D 矩阵直接输出 (D 类型设成 FP8 + scale 给 hipBLASLt) | ✅ | `setScaleD()` + D dtype = FP8 |
| Per-tensor scale A/B/C/D | ✅ | `setScaleA/B/C/D` |
| Standalone amax 算子 (GEMM 之外的 fused amax+cast) | ✅ | `hipblaslt-ext-op.h:138` |
| **GEMM 输入** A/B 的 amax 在 epilogue 里算 | ❌ | 设计上不可能 — A/B 是输入, GEMM 没开始算前算不了 amax |

**关键结论**: hipBLASLt 库层完全 OK，**输出侧已经能 fuse, 输入侧本质上做不到**。

### 1.2 TE 的现状 (`/workspace/deps/TransformerEngine/transformer_engine/common/gemm/rocm_gemm.cu`)

```cpp
// 第 985 行: 拿到 output amax buffer
void *D_amax = outputD->amax.dptr;

// 第 1111 行: 把 D_amax 设进 hipBLASLt operationDesc
NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
    operationDesc, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &D_amax, sizeof(D_amax)));
```

→ TE 调 hipBLASLt 时, **每次 FP8 GEMM 都顺带把 D 的 amax 写到一个固定 buffer**。trace 里因此**根本不会出现 `D_amax` 计算的独立 kernel**。这一半 TE 早做完了, 我们之前误以为能再拿一把肉是错判。

### 1.3 primus_turbo 的现状

- `primus_turbo/pytorch/modules/linear_fp8.py:Float8Linear.forward()` 调 `gemm_fp8(x, weight, ...)` (在 `pytorch/ops/gemm_fp8.py`)
- `gemm_fp8` 流程：

```python
a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)  # ← input cast + amax
b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)  # ← weight cast + amax
out = gemm_fp8_impl(a_fp8, a_scale_inv, ..., default_backend=BackendType.HIPBLASLT.value)  # ← FP8 GEMM (走 hipBLASLt)
```

- `quantize_fp8(...)` → `quantize_fp8_tensorwise_impl(x, out_dtype, scale)` → Triton kernel 算 input amax + cast。这就是 trace 里 `primus_turbo::reduce_row_kernel<AbsMaxOp>` (25.40 ms) 和 `primus_turbo::unary_kernel<...QuantTensorwiseScalePtrOp...>` (31.27 ms) 的来源。
- **每步 144 次 GEMM × 每次 2 个 input (a + b) → 288 次 quantize_fp8 调用 (但 weight 通常缓存, 实际 launch 次数较少)**。

### 1.4 trace 71.6 ms 的真实归属

| kernel | ms | 性质 | 能 fuse 进 GEMM epilogue 吗 |
|---|---:|---|---|
| `primus_turbo::unary_kernel<bf16 → fp8_e4m3>` | 31.27 | **input cast** (GEMM 前) | ❌ 输入侧, 不行 |
| `primus_turbo::reduce_row_kernel<AbsMaxOp, bf16>` | 25.40 | **input amax** (GEMM 前) | ❌ 输入侧, 不行 |
| `transformer_engine::amax_kernel<16, true, bf16>` | 8.49 | **input amax** (TE Linear 那一支, GEMM 前) | ❌ 输入侧, 不行 |
| `cast_transpose_optimized_kernel<bf16, fp8_e4m3>` | 10.59 | **input cast + transpose** (GEMM 前, 给需要 transposed weight 的 dgrad/wgrad) | ❌ 输入侧, 不行 |
| 杂项 (scale compute, clamp) | ~6 | input prep | ❌ |

**所有 71.6 ms 都是 GEMM 输入侧 prep**。下推到该 GEMM 自己的 epilogue 在物理上不可能。

## 2. 真正可行的 attack 路径

### Plan A: 切 `fp8_recipe: tensorwise → delayed` (最低成本 A/B)

- **Yaml 改动**: 1 行 (`gpt_oss_20B-pretrain-fp8.yaml:55`)
- **机制**: TE delayed scaling 用一个 history buffer 存过去 N 步 amax, 每步取 max 当 scale, **跳过当步 amax 计算**。
- **预期收益**:
  - `transformer_engine::amax_kernel` 8.49 ms → ~0
  - `primus_turbo::reduce_row AbsMaxOp` 25.40 ms → ~0 (如果 primus_turbo 也尊重 delayed; 需要核实)
  - `unary_kernel` cast 31.27 ms 仍在 (cast 还需要 — 只是 scale 来源换了)
  - 总收益 ~25-34 ms / step (2.7-3.6%)
- **风险**: Primus 代码 `modules/trainer/megatron/utils.py:488` 显示 `support_fp8_recipe = ["tensorwise", "blockwise", "mxfp8"]` 是 **`use_turbo_parallel_linear` 路径下的限制**。切 `delayed` 会强制走 TE 原生 Linear, **失去 turbo Linear 的其他优化** (cast_transpose 融合策略可能有变, weight cache 行为变, 等)。
- **实验**: 跑 24-iter trace + 80-iter loss 曲线对比。决定指标:
  - kernel breakdown: `unary_kernel` / `reduce_row_kernel` / `amax_kernel` / `cast_transpose_optimized_kernel` 是否真的下降
  - step time 总值 (考虑 turbo Linear 损失)
  - lm loss 是否漂移 > 5%

### Plan B: 给 primus_turbo Float8Linear 加 delayed scaling (中等工程)

- **目标**: 保留 turbo Linear 的所有其他优化, 同时跳过每步 input amax 计算
- **Sketch**:
  - `Float8QuantConfig` 新增 `recipe: Recipe.DELAYED`
  - `Float8Linear` 维护一个 `amax_history: torch.Tensor` (per-tensor, length 1024 by default)
  - `forward()` 先用 `amax_history.max()` 当 scale 直接 cast (`unary_kernel` 仍跑, 但跳过 `reduce_row_kernel` 的 amax 算)
  - 反向算出 `amax(input)` 写进 history (用 GEMM 的 D_amax fuse, 不另外起 reduce_row)
- **预期收益**: ~25 ms (`reduce_row_kernel`), 保留所有 turbo 收益
- **工程量**: 改 `primus_turbo/pytorch/modules/linear_fp8.py` + `quantization.py` + 新增 amax history 管理. 估 1-2 天 + A/B 验证.
- **风险**: amax history length 选择影响数值稳定性, MLPerf 合规要审 (delayed scaling 数值跟 current scaling 不完全等价)

### Plan C: FP8 chaining (大工程, 暂存)

- **目标**: 让 GEMM_n 的 D 直接输出 FP8 (用预先算好的 scale), GEMM_{n+1} 直接读 FP8 input
- **要求**: 跨 GEMM 边界传 scale, 改 `gemm_fp8_impl` API, 改所有调用点
- **预期收益**: cast (31 ms) + cast_transpose (10.6 ms) 砍掉, 留下 amax (但用 D_amax history 跨步)
- **工程量**: 大. 推迟到 Plan A/B 验证后再考虑.

### Plan D: 不动 (默认)

- 接受 71.6 ms 是 `tensorwise` recipe 的内禀代价
- 把优化重心转到 [note 28 §6](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#6-单点优化排序roi--难度) 其他 Tier (Tier 0 grouped GEMM, Tier 4 FMHA bwd, Tier X 多流恢复)
- 等 ROCm 8.x 或 primus_turbo 上游加 delayed scaling 支持

## 3. 决策矩阵

| Plan | 工程量 | 期望收益 | 风险 | 推荐顺序 |
|---|---|---|---|---|
| A. yaml 切 delayed | 1 行 | -25 ~ -34 ms (但可能被 turbo 损失抵消) | 中 (turbo Linear 失效, MLPerf 数值审) | **★ 先做** |
| B. primus_turbo 加 delayed | 1-2 天 | -25 ms (保 turbo 收益) | 中 (数值合规审) | A 验证完且收益正才做 |
| C. FP8 chaining | 1 周+ | -40 ms | 大 (跨 API 改) | A/B 都通过且仍要榨油再考虑 |
| D. 不动 | 0 | 0 | 0 | A 失败时回到这条 |

## 4. 推荐立刻做的实验 (Plan A 验证)

### 4.1 改动 (单行)

```yaml
# small_llm_moe_pretraining/primus/gpt_oss_20B-pretrain-fp8.yaml: 55
- fp8_recipe: tensorwise
+ fp8_recipe: delayed
```

注意: 这里 yaml 是模型路径而非 Primus 默认值, 可能还要看 `use_turbo_parallel_linear` 是否 force `tensorwise`。如果 Primus 在加载时 assert 拒绝, 就要先临时关掉 turbo 强制走 TE 原生 Linear。

### 4.2 跑 trace 对比 (24-iter, 跟 note 29 同口径)

```bash
docker exec -w /home/xiaompen/mlperf-training/small_llm_moe_pretraining/primus \
    xm-mlperf bash -c 'bash run.sh'

cd /home/xiaompen/mlperf-training
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    'small_llm_moe_pretraining/primus/output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].*.pt.trace.json' \
    ProfilerStep#17
```

### 4.3 对比指标 (所有都要拿到, 缺一不可)

| metric | baseline (note 29 #17) | delayed | Δ ms | 决定 |
|---|---:|---:|---:|---|
| step wall | 938.97 | ? | ? | 总 step 是涨是跌 |
| `primus_turbo::unary_kernel` | 31.27 | ? | ? | input cast 是否还在 (delayed 仍需 cast) |
| `primus_turbo::reduce_row_kernel` | 25.40 | ? | ? | input amax 是否消失 |
| `transformer_engine::amax_kernel` | 8.49 | ? | ? | TE Linear 那一支 amax |
| `cast_transpose_optimized_kernel` | 10.59 | ? | ? | TE 那一侧 cast |

### 4.4 决策门

- **绿灯**: amax/reduce_row 类总和减少 ≥ 25 ms **且** step wall 减少 ≥ 15 ms **且** 80-iter lm_loss 漂移 ≤ 1% → 直接合并 yaml 改动, 收回 Tier 1A 部分肉
- **黄灯**: kernel 减少了但 step wall 没显著降 → turbo Linear 损失抵消, **走 Plan B** (给 primus_turbo 加 delayed 支持)
- **红灯**: kernel 没减或 loss 大幅漂移 → 退回, 标 Tier 1A 在当前栈不可行, 转 Tier 0 / Tier 4

## 5. 这次调研的修正点 (写给将来的自己)

1. **永远先读 backend 源码再列优化 plan**: 这次原始 [note 28 Tier 1A](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#tier-1--elementwise--norm-融合--目标-2--4-step已收割大半剩余主要是-fp8-量化链) 的描述 ("把 amax 和 scale 收到上一个 GEMM 出口, 把 cast 收到下一个 GEMM 入口") 是把 input/output 侧的 amax 混在一起说, 误导了自己 ~3 小时调研后才看清楚。
2. **trace 里看 `xxx_kernel` 在 GEMM 之前还是之后**: 用 SKILL.md 的 `full_breakdown.py` 看 time-binned mix, 或者直接在 trace JSON 里按时间戳找单个 GEMM 的前后 N ms 的 kernel, 确认 prep 还是 post 流。这次没做这步, 直接基于"trace 显示有这些 kernel"就推断"它们能塞 epilogue", 步骤跳了。
3. **TE 是 editable install (`/workspace/deps/TransformerEngine/`)**, 改 .py 直接生效, 改 .cu 要重编。Plan B 改 primus_turbo 也是同样 editable, 工程量评估按这个走。

## 6. 不动 yaml 也能做的小修

调研过程中顺手发现两处可以独立优化, 跟 Tier 1A 主线分开做：

### 6.1 `cast_transpose_optimized_kernel` 10.59 ms × 144 — 已经融合, 不动

`cast_transpose_optimized_kernel` 是 TE 自带的 cast + transpose 融合 (一次 launch 干两件事), trace 里看到的就是融合后的样子。**这是 TE 的优化已经落地**, 没法再融。

### 6.2 0-byte memcpy + scale recompute 杂项 ~6 ms

trace 里 `memcpy` 类 4.32 ms 大部分是 0-byte probe (可忽略), scale recompute 是 fp8 recipe 的固定开销。这部分数值已经很小, 不优先碰。

## 7. 后续 TODO

| # | 任务 | 优先级 | 阻塞条件 |
|---|---|---|---|
| 1 | Plan A 验证: yaml 切 delayed, 跑 24-iter trace + 80-iter loss A/B | ★★★ | 无, 立刻可做 |
| 2 | Plan B 实现: primus_turbo Float8Linear 加 delayed scaling | ★★ | Plan A 黄灯时启动 |
| 3 | 修正 [note 28 §6 Tier 1A](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#tier-1--elementwise--norm-融合--目标-2--4-step已收割大半剩余主要是-fp8-量化链) 描述, 把"下推 epilogue"改成"切 fp8_recipe + 维持 turbo 优化" | ★ | 本 note 完成后可做 |
| 4 | 修正 §6 ROI 表的 Tier 1A 预期收益 (-5% → -2.5 ~ 3.5%, 范围更窄) | ★ | 同上 |

## 8. 文件 / 代码清单

读过的关键文件 (供下次复盘):

- `/opt/rocm/include/hipblaslt/hipblaslt.h` — epilogue enum, MATMUL_DESC 字段
- `/opt/rocm/include/hipblaslt/hipblaslt-ext.hpp` — `setAmaxD` / `setScaleD` 等 ext API
- `/workspace/deps/TransformerEngine/transformer_engine/common/gemm/rocm_gemm.cu:1037-1170` — TE 调 hipBLASLt 完整设置, 已经用 `AMAX_D_POINTER`
- `/workspace/deps/TransformerEngine/transformer_engine/common/recipe/current_scaling.cu:62` — `amax_kernel` 实现 (TE 一侧)
- `/opt/venv/lib/python3.12/site-packages/primus_turbo/pytorch/modules/linear_fp8.py` — `Float8Linear` 入口
- `/opt/venv/lib/python3.12/site-packages/primus_turbo/pytorch/ops/gemm_fp8.py` — `gemm_fp8 → quantize_fp8 + gemm_fp8_impl`
- `/opt/venv/lib/python3.12/site-packages/primus_turbo/pytorch/ops/quantization.py` — `quantize_fp8 → quantize_fp8_tensorwise_impl`
- `/workspace/Primus/primus/configs/modules/megatron/trainer_base.yaml:53` — `fp8_recipe: delayed` (Primus 默认, **被本 yaml 覆盖**)
- `/workspace/Primus/primus/modules/trainer/megatron/utils.py:488` — `use_turbo_parallel_linear` 限制
- `/workspace/Primus/primus/backends/megatron/patches/te_patches/delayed_scaling_patches.py` — Primus 的 TE delayed scaling 补丁 (有现成的 path 走)
