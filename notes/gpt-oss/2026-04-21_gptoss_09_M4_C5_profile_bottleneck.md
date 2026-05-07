# GPT-OSS-20B MLPerf MI355X — M4_C5 (mbs=4 / 全 5 优化叠) torch.profiler 瓶颈分析

**周期**: 2026-04-21
**配置**: 1×8 MI355X, fp8 hybrid + bf16 weight, **mbs=4 / gbs=32 / lr=8e-4**, tp=pp=ep=1, 80 iter, profiler @ iter 50/51/52, rank-0
**Trace**: `/home/xiaompen/mlperf/profile_traces/M4_C5/M4_C5_rank0.pt.trace.json` (260 MB, 977k events)
**关联**: 上游优化链分析见 [`08_mbs4_optimization_chain`](./2026-04-21_gptoss_08_mbs4_optimization_chain.md)；mbs=2 baseline profile 见 `/home/xiaompen/mlperf/PROFILE_REPORT.md`
**说明**: profile run 在 lr-warmup 中段（iter 50, lr≈3.1e-4），但单步算子结构 / 并行 / fusion 路径与稳态完全一致；wall ≈ 1217 ms（链稳态 1320 ms 略高，是 lr 走完后的 bf16 elementwise 稍多 + amax 历史更新的微差）。

---

## 1. 端到端 per-iter 时间分解（3 个 profiler step 平均）

| 指标 | C5 (mbs=4) | B0 (mbs=2 baseline, 04-19) | 备注 |
|---|---:|---:|---|
| wall / iter | **1217 ms** | 800 ms | 单步 +52%（但塞了 2× 样本 → samples/sec +18%） |
| GPU active (merged across streams) | **1192 ms** | 697 ms | utilization 98.0% |
| GPU bubble (wall − merged) | **25 ms (2.0%)** | 52 ms (6.5%) | 比 mbs=2 时减半，硬件已经被填得更满 |
| compute kernels busy（不含 comm） | 1066 ms | 697 ms | +53% |
| comm kernels busy (rccl) | **512 ms (42% wall)** | 670 ms (42%) | mbs=4 下绝对 comm 比 mbs=2 baseline **少了 158 ms** ← bf16 grad reduce + ddp_pad 的功劳 |
| compute ∩ comm overlap | 200 ms (16.4%) | 284 ms (35.5%) | overlap 比例下降是因为分子（comm）变小 |
| **exposed comm (comm-only)** | **49 ms (4.0%)** | 51 ms (6.4%) | **绝对值已经压到只有 4%**，几乎没空间了 |
| idle (无 kernel 区) | **99 ms (8.2%)** | 52 ms (6.5%) | 唯一变大的一项 — **`.item()` host sync 拉长** |

**可优化"理论天花板"** ≈ exposed comm + idle ≈ **148 ms / iter ≈ 12% throughput**，和 mbs=2 时（103 ms / 12-13%）几乎相同；硬件饱和度已经到顶，剩余收益必须靠**砍掉计算本身**。

---

## 2. GPU 时间按类别分布（per iter）

| 类别 | 3 iter 合 (ms) | per iter (ms) | 占 wall | 启动 / iter | 同 mbs=2 B0 (per iter) | 趋势 |
|---|---:|---:|---:|---:|---:|---:|
| **gemm_bf16** | 2804 | **935** | **76.8 %** | 2687 | 558 ms | **+68%（首要瓶颈）** |
| comm (rccl) | 1535 | 512 | 42.0% | 212 | 670 ms | **−24%（grad reduce bf16 收益）** |
| attention | 604 | 201 | 16.5% | 144 | 116 ms | +73% |
| other (sort_chunks / FxGraph / topk / aiter RoPE / 其他 fused) | 438 | 146 | 12.0% | 747 | 78 ms | +87% |
| elementwise | 357 | 119 | 9.8% | 1184 | 232 ms | **−49%（bf16 grad accumulate 收益）** |
| moe_permute | 211 | 70 | 5.8% | 96 | 39 ms | +79% |
| activation (Triton SwiGLU) | 148 | 49 | 4.0% | 72 | 25 ms | +96% |
| optimizer | 100 | 33 | 2.7% | 127 | 33 ms | 0% (gbs=32 与 gbs=16 主权重数量相同) |
| **norm (Triton RMSNorm)** | 89 | **30** | **2.4 %** | 194 | 84 ms | **−64%（Triton RMSNorm 收益）** |
| cast_fp8 | 87 | 29 | 2.4% | 192 | 17 ms | +71% |
| reduce / loss / indexing | 45 | 15 | 1.2% | 175 | — | — |

> 各类别在多 stream 上并行，类别和 ≈ 6418 ms 远 > GPU 活跃 3577 ms，平均 **~1.8 stream 并行**（mbs=2 B0 时是 ~2.4，因为 mbs=4 把单 stream 喂得更满，并行度自然下降）。

**MoE 调度真实开销** = `_permute_kernel` (90) + `_unpermute_kernel` (121) + `_sort_chunks_by_idxs_kernel` (113, 在 "other") + `_sort_chunks_by_map_kernel` (56, 在 "other") = **380 ms / 3 iter ≈ 127 ms / iter (10.4% wall)**，比表面看到的 5.8% 高近一倍。

---

## 3. Top-10 单 kernel（按 GPU 总时间，3 iter 合）

| # | 时间 (ms) | 占比 | 启动 | avg (us) | kernel | 归属 |
|---:|---:|---:|---:|---:|---|---|
| 1 | **767.3** | **12.0%** | 318 | 2413 | `ncclDevKernel_Generic_1` | comm |
| 2 | **728.2** | **11.4%** | 666 | 1093 | `Cijk_…_BBS_…MT192x192x64_MI16x16x1_…` | **MoE bf16 GEMM (fwd dispatch)** |
| 3 | **724.4** | **11.3%** | 588 | 1232 | `Cijk_…_BBS_…MT256x256x64_MI16x16x1_CMS_LDSB0_DTLA1_DTLB1` | **MoE bf16 GEMM (大 tile)** |
| 4 | **582.0** | **9.1%** | 580 | 1003 | `Cijk_Alik_Bljk_BBS_…MT256x256x64_…` (T variant) | **MoE bf16 GEMM (bwd, transposed)** |
| 5 | 452.7 | 7.1% | 150 | 3018 | `nccl:reduce_scatter_tensor_coalesced` | comm (DDP grad reduce) |
| 6 | **344.7** | **5.4%** | 36 | **9575** | `aiter::fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile` | **attention bwd（最重，且仍带 `_recompile` 后缀）** |
| 7 | 313.3 | 4.9% | 150 | 2089 | `nccl:all_gather_into_tensor_coalesced` | comm (param all-gather) |
| 8 | 149.1 | 2.3% | 1605 | 93 | `Cijk_…MT128x64x32_MI16x16x1` | small-tile bf16 GEMM (MoE 路由后小 batch) |
| 9 | **147.5** | **2.3%** | 288 | 512 | `Cijk_Alik_Bljk_F8B8BS_…MT256x256x128_…` | **FP8 GEMM (LayerNormLinear, qkv/proj only)** |
| 10 | 146.0 | 2.3% | 726 | 201 | `vectorized_elementwise_kernel<add<bf16>>` | **bf16 grad accumulate（C4 已切到 bf16）** |

**关键观察 — Top-3 全部是 MoE bf16 GEMM (#2+#3+#4 = 2034 ms / 3 iter ≈ 678 ms / iter ≈ 56% wall)。**

---

## 4. 五个核心瓶颈与画像

### 4.1 ⭐ MoE grouped GEMM 仍走 bf16 (Top 1, 56% wall)

- 三个 Cijk Tensile bf16 内核合计 **678 ms / iter（56% wall）**。
- FP8 GEMM 类（`*_F8B8BS_*` 与 `*_F8BS_*`）合计仅 **207 ms / 3 iter ≈ 69 ms / iter (5.7% wall)** —— 即只有 LayerNormLinear 的 qkv/proj 走 FP8，**MoE 32 个 expert 的 grouped GEMM 整条路径仍是 bf16**。
- yaml 锁死的两条线：`moe_use_legacy_grouped_gemm: true` + `use_turbo_grouped_mlp: false`。
- **理论收益**：FP8 GEMM ≈ bf16 GEMM 的 1.7× 吞吐（MI355 spec），即 678 ms × (1−1/1.7) ≈ **节省 280 ms / iter ≈ −23 % wall**。即使打 60% 折扣（schema 切换 + alignment + 激活 cast 损耗），仍能拿到 −12 % 到 −15 %。
- **这是 mbs=4 路径下唯一的"大头"优化机会**，比所有其他优化加起来都大。
- 风险：(a) `enable_primus_turbo + use_turbo_grouped_mlp + turbo_sync_free_moe_stage=2` 需要一次完整收敛 ablation；(b) 与 `moe_use_legacy_grouped_gemm: true` 互斥，必须一起切；(c) MXFP8 路径 vs delayed amax 还要再选一次。

### 4.2 ⭐ Attention bwd 仍带 `_recompile` 后缀 (Top 6, 9.4% wall)

- `aiter::fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile` 单次 **9.6 ms × 12 调用 / iter ≈ 115 ms / iter (9.4% wall)**。
- `_recompile` 后缀提示 backward 仍在用 JIT-recompile 变体，不是缓存的 AOT 内核。
- forward (`ck_tile::FmhaFwdKernel`) 已经走 ck_v3 + gfx950 (#11，43.5 ms / iter)，最快路径。
- yaml/env 已经设了 `NVTE_FMHA_BACKEND_DENSE_BWD=ck_v3` + `NVTE_CK_USES_BWD_V3=1`，但实际运行时 dense bwd 没切过去 —— 可能是 hd64 case 在 ck_v3 builder 里没注册，或者 SWA window=128 触发 fallback。
- 当前 trace 里 ck_tile bwd 也存在 (`FmhaBwdDQDKDVKernel` 87 ms / 3 iter = 29 ms / iter)，只是只覆盖 SWA layer；**dense layer 的 bwd 还是 aiter recompile 路径**。
- **理论收益**：aiter recompile → ck_v3 大约能省 30–40 ms / iter（−2.5 ~ −3.3 %）。代价低（只是 build/env），值得调查。

### 4.3 MoE permute / unpermute / sort_chunks（合 10.4% wall）

- 真实 dispatcher 开销 = `_permute_kernel` 30 + `_unpermute_kernel` 40 + `_sort_chunks_by_idxs_kernel` 38 + `_sort_chunks_by_map_kernel` 19 = **127 ms / iter (10.4% wall)**。
- 当前 `moe_permute_fusion: true` + `moe_token_dispatcher_type: alltoall` + `moe_use_legacy_grouped_gemm: true`。
- 进一步压缩需要 `turbo_sync_free_moe_stage: 2` 或 `3`，但与 4.1 强耦合（必须一起切到 PrimusTurboGroupedMLP）。
- **预估和 4.1 一起拿**：sync_free_moe 能再省 30–50 ms / iter（−2.5 ~ −4 %）。

### 4.4 Triton SwiGLU bwd 的 fused_cat (4.0% wall)

- `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1` 单次 **1.28 ms × 24 调用 / iter ≈ 31 ms / iter (2.5% wall)**。
- 这是 SwiGLU 反向的 chain（cat + silu + silu_backward + cast），现在已经被 torch.compile 融到一个 Triton kernel 里。
- 短期没有明显改进路径；如果切到 `use_turbo_grouped_mlp` (4.1)，整条 MLP 反向会被 grouped GEMM 自己的 cast/写回吃掉，这部分会一并消失。

### 4.5 ⭐ Host-side `.item()` 拉出 99 ms / iter 的 idle (8.2% wall)

- CPU-op 表里 `aten::item` + `aten::_local_scalar_dense` 共 **3979 ms / 3 iter ≈ 1326 ms / iter** 的 CPU 自时间 —— 等于强制 device→host 同步。
- 实际 GPU bubble (idle) 是 99 ms / iter (8.2%)，比 mbs=2 B0 (52 ms, 6.5%) 多 **+47 ms** —— 这是 mbs=4 唯一回退的指标。
- 来源：(a) FP8 delayed recipe amax history 更新；(b) `L2NormFunctor` 在 trace 里 369 次（grad-clip norm）；(c) loss / grad_norm 的 `.item()` 转 host 用于 logging。
- `check_for_nan_in_loss_and_grad: false` 已经关；下一档收益要么改 logging 频率（`MLLOG_TRAIN_LOSS_LOG_FREQ` 32 → 256），要么换 MXFP8 (`fp8_recipe: blockwise` + `NVTE_ROCM_ENABLE_MXFP8=1`) 去掉 delayed amax 同步。
- **预估**：+log_freq + MXFP8 大概能砍 30–40 ms idle (−2.5 ~ −3.3 %)。

---

## 5. 与 mbs=2 baseline 的瓶颈结构对比

| 维度 | mbs=2 B0 (04-19) | mbs=4 C5 (本次) | 解读 |
|---|---|---|---|
| **首要瓶颈** | comm 36% + bf16 GEMM 30% | **bf16 GEMM 77%** | mbs=4 把 comm 用 bf16 grad reduce + ddp_pad 削到 21%，而 GEMM 因为 batch 翻倍变成绝对统治 |
| 第二瓶颈 | fp32 grad add 12.5% | attention 16.5% + MoE dispatch 10% | mbs=4 下 attention seq 在 bwd_recompile 里更显眼 |
| 第三瓶颈 | RMSNorm 4.5% | host sync (idle) 8.2% | RMSNorm 已经被 Triton 干掉，剩下的是 .item() 同步 |
| **GPU util** | 87% | **98%** | 多 11 pp，已逼近上限 |
| **exposed comm** | 6.4% | 4.0% | overlap 已经几乎打满 |
| **bubble (no-kernel)** | 6.5% | 8.2% | 唯一退化项，全部来自 host sync |

**结论：mbs=4 把"通信-计算耦合"变成"纯计算驱动"**——优化方向必须从"压通信 / overlap" 转向"砍 GEMM / cast / sync"。

---

## 6. 下一步优化优先级（mbs=4 路径下重排）

| 排名 | 优化 | 预估 wall ↓ | 改动 | 风险 | 说明 |
|---:|---|:---:|---|:---:|---|
| **P0** | **MoE FP8 grouped GEMM**：`use_turbo_grouped_mlp: true` + `moe_use_legacy_grouped_gemm: false` + `turbo_sync_free_moe_stage: 2` | **−12 ~ −15 %** (~150-180 ms) | 3 行 yaml + PrimusTurbo build 验证 | **中**（数值要 1 次收敛 ablation） | **唯一的"大头"，其他全部加起来都不如它** |
| P1 | Dense attention bwd 切 ck_v3，去掉 `_recompile` | −2.5 ~ −3.3 % (~30-40 ms) | env / build flag | 低 | 检查 `NVTE_FMHA_BACKEND_DENSE_BWD=ck_v3` 在 hd64 是否真的生效；可能要 PrimusTurbo `use_turbo_attention: true` 触发 |
| P2 | 降低 host sync：MXFP8 (`NVTE_ROCM_ENABLE_MXFP8=1` + `fp8_recipe: blockwise`) + `MLLOG_TRAIN_LOSS_LOG_FREQ` 32→256 | −2.5 ~ −3.3 % (~30-40 ms) | env + 1 行 yaml | 中 (MXFP8 收敛要 ablation) | mbs=4 的 idle 比 mbs=2 多 47 ms，全部从 sync 来 |
| P3 | MoE sync-free dispatcher (`turbo_sync_free_moe_stage: 2/3`) | −2.5 ~ −4 % (~30-50 ms) | yaml 1 行 | 中 | 与 P0 强耦合，**必须和 P0 一起做** |
| P4 | 缩短暴露通信 / bucket 调优：`overlap_param_gather_with_optimizer_step` 等 Megatron 高阶 flag | < −1 % | yaml | 低 | exposed comm 已经只有 4%，空间几乎封顶 |
| **不做** | **Triton RMSNorm 进一步优化** | < −0.1 % | — | — | norm 总占比已降到 2.4%，再优化没有空间 (04-21 ablation 已确认) |
| **不做** | **bf16 grad accumulate 进一步优化** | < −0.5 % | — | — | C4 已落地，elementwise 已从 232ms → 119ms |

**乐观叠加上限** = P0 + P1 + P2 + P3 ≈ **−18 ~ −24 %**，wall 1217 → **925-1000 ms**，对应 TFLOPs/GPU **633 → 770-830**，samples/sec 24.2 → **29-32**。

**保守现实** ≈ P0 单点拿 −10%，其他叠加 50% 折扣 → 1217 → **1085 ms (-11%)**，TFLOPs/GPU 633 → 710。

---

## 7. 一行总结

> **C5 已经把通信 / elementwise / RMSNorm 三类瓶颈全部解决（comm 暴露压到 4%，elementwise −49%，norm −64%），剩下 77% 的 wall 被 bf16 MoE grouped GEMM 占据 —— 下一步的收益只能从"把 MoE 切到 FP8 grouped GEMM"这一项里出，其他所有候选加起来 < 8%。**

---

## 8. 复现命令

```bash
# 1. 跑 C5 profile（约 5-7 分钟，写 trace 到容器内）
bash /home/xiaompen/mlperf/ablations/M4_chain/run_profile_C5.sh

# 2. 把 trace 拷到 host
docker cp 'xiaoming-mlperf:/workspace/code/output/amd/root/M4_C5_profile/tensorboard/<trace>.pt.trace.json' \
          /home/xiaompen/mlperf/profile_traces/M4_C5/M4_C5_rank0.pt.trace.json

# 3. 三件套分析
cd /home/xiaompen/mlperf
python3 profile_analyze.py profile_traces/M4_C5/M4_C5_rank0.pt.trace.json | tee profile_traces/M4_C5/analyze.txt
python3 profile_overlap.py  profile_traces/M4_C5/M4_C5_rank0.pt.trace.json | tee profile_traces/M4_C5/overlap.txt
python3 profile_dive.py     profile_traces/M4_C5/M4_C5_rank0.pt.trace.json comm gemm_bf16 elementwise attention moe_permute norm cast_fp8 optimizer activation other | tee profile_traces/M4_C5/dive.txt
```

## 9. 关联资产

| 资产 | 路径 |
|---|---|
| profile yaml | `/home/xiaompen/mlperf/ablations/M4_chain/M4_C5_profile.yaml` |
| profile runner | `/home/xiaompen/mlperf/ablations/M4_chain/run_profile_C5.sh` |
| trace JSON (260 MB) | `/home/xiaompen/mlperf/profile_traces/M4_C5/M4_C5_rank0.pt.trace.json` |
| analyze 输出 | `/home/xiaompen/mlperf/profile_traces/M4_C5/{analyze,overlap,dive}.txt` |
| 上游优化链分析 | [`08_mbs4_optimization_chain`](./2026-04-21_gptoss_08_mbs4_optimization_chain.md) |
| mbs=2 baseline profile (B0) | `/home/xiaompen/mlperf/PROFILE_REPORT.md` |
