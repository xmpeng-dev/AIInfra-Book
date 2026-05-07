# GPT-OSS-20B MLPerf MI355X — Attention bwd "_recompile" 调查 postmortem (M4_C7 / C-series)

**周期**: 2026-04-21
**起点**: [`09_M4_C5_profile_bottleneck`](./2026-04-21_gptoss_09_M4_C5_profile_bottleneck.md) §4.2 P1 项 — "Dense attention bwd 仍带 `_recompile` 后缀，估计切到 ck_v3 可省 30–40 ms / iter"
**结论 (一行)**: **note 09 P1 的前提是错的** — `_recompile` 只是 codegen 命名约定，trace 中那个 kernel 已经是 gfx950 上的 AOT v3 fast path，dense attn bwd 已经跑到 hd64 理论 bf16 算力的 70%；唯一可调的 `NVTE_CK_IS_V3_ATOMIC_FP32=0` 反而**慢了 13.6%（+165 ms）**。**所以 C 阶段不产生新 baseline，M4_C7 (1218.9 ms) 仍是当前最优**，下一步直接进入 D（mbs=8）评估。

---

## 1. note 09 hypothesis 拆解

note 09 的判断链:

> Top-6 kernel: `aiter::fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile`，9.6 ms × 12 调用 / iter ≈ 115 ms / iter (9.4 % wall)。`_recompile` 后缀提示 backward 仍在用 JIT-recompile 变体，不是缓存的 AOT 内核。理论收益: aiter recompile → ck_v3 大约能省 30–40 ms / iter。

实际 trace 拆解 (3 iter 平均，本次复读，与 note 09 一致):

| 类型 | kernel | 调用 / iter | 时间 / iter | 占 wall |
|---|---|---:|---:|---:|
| Dense fwd (24 layers 中 12 dense) | `ck_tile::FmhaFwdKernel` | 12 | 21.8 ms | 1.8 % |
| SWA   fwd (24 layers 中 12 SWA)   | `ck_tile::FmhaFwdKernel` | 12 | 21.8 ms | 1.8 % |
| **Dense bwd (12 layers, hd64 causal)** | **`aiter::fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile`** | **12** | **114.9 ms** | **9.4 %** |
| SWA   bwd (12 layers, hd64 SWA)    | `ck_tile::FmhaBwdDQDKDVKernel` | 12 | 29.1 ms | 2.4 % |
| Bwd preprocess (OGradDotO)         | `ck_tile::FmhaBwdOGradDotO` | 24 | 5.0 ms | 0.4 % |
| Bwd postprocess (ConvertQGrad)     | `ck_tile::FmhaBwdConvertQGrad` | 24 | 4.0 ms | 0.3 % |
| Bwd RNG/uncached helper            | `aiter::kn_entry_1c_sbhd_uncached<OpUncachedBwd>` | 48 | 4.7 ms | 0.4 % |
| **合计 attention bwd**             | — | — | **~158 ms** | **~13 %** |

注意:
- 24 个 layer = 12 dense + 12 SWA (`window_attn_skip_freq` 1/0 交替)
- dense bwd 全部走 aiter，SWA bwd 全部走 ck_tile，与设计预期 (`NVTE_FMHA_BACKEND_DENSE_BWD=ck_v3` / `NVTE_FMHA_BACKEND_SWA_BWD=ck`) 一致 — **没有路径错配**

---

## 2. `_recompile` 后缀真实含义 — 不是 JIT recompile

证据来自 aiter 源码:

### 2.1 静态 AOT 配置表中已经把 `_recompile` 当作正常变体名注册

`/workspace/deps/aiter/hsa/gfx950/fmha_v3_bwd/fmha_bwd_dqdkdv.csv` 的 hd64 行 (节选):

```
bf16,64,64,1,1,1,0,0,2,32,192,_ZN5aiter48fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompileE,bwd_hd64_bf16_causal_a32_rtz_pssk.co
bf16,64,64,1,0,0,0,0,2,32,192,_ZN5aiter43fmha_bwd_hd64_bf16_causal_a16_rtz_recompileE       ,bwd_hd64_bf16_causal_a16_rtz.co
```

- 列分别是 `dtype, hdim_q, hdim_v, mask, atomic32, pssk, pddv, mode, bf16_cvt, ts_qo, ts, knl_name, co_name`
- `co_name` 已经是预编译 `.co` 文件 (`bwd_hd64_bf16_causal_a32_rtz_pssk.co`)，**不会触发 hipModule JIT**
- 只要 dtype + hdim + mask + atomic32 + pssk + pddv + bf16_cvt 一对就直接调用 → 这才是 trace 里看到的那个 kernel

### 2.2 `_recompile` 后缀仅表示该变体来自 aiter 的 "recompile" codegen 路径

`/workspace/deps/aiter/csrc/cpp_itfs/mha_bwd.cu:87-93` 的判断:

```cpp
if(cfg.atomic32 == 0 &&
   ((arch_id == "gfx942") || (el.first.find("recompile") != std::string::npos)))
{
    tmp_ts_kv = 64;
}
```

- 这里用 `find("recompile") != npos` 做 dispatch micro-arch 调整 (改 `tmp_ts_kv`)，**没有任何 "需要 JIT 编译" 的语义**
- gfx950 上 hd64 的所有 bf16 kernel 全部带 `_recompile` 后缀 — 这是 aiter team 给 hd64 codegen 路径起的标签，与 hd128 / hd192 区分

### 2.3 trace 直接证据 — kernel grid 是预编译尺寸，不是 JIT 后的随机尺寸

```
ts=1738732508994 dur=9312 us name=aiter::fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile
  device=2 stream=0 grid=[22,64,4] block=[256,1,1]
```

- 12 次连续调用都是同一个 grid `[22,64,4]` / block `[256,1,1]` / 9.3-9.7 ms — **如果是 JIT 第一次会有 100ms+ 编译时间**，trace 里完全没有

### 2.4 父 CPU op 是 TE 的 `FusedAttnFunc.backward`

```
correlation=33726 → cpu_op=FusedAttnFuncBackward → external_id=5913
父调用链: TE FusedAttnFunc.backward
       → ck_fused_attn_bwd.cpp:676   aiter::mha_bwd(uses_bwd_v3=true, is_v3_atomic_fp32=true)
       → aiter mha_bwd.cu fmha_v3_bwd
       → 命中 dqdkdv.csv 中的 bf16/hd64/causal/a32_pssk 行
       → 启动 _ZN5aiter48fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompileE  (来自 .co)
```

- `NVTE_CK_USES_BWD_V3=1` 和 `NVTE_CK_IS_V3_ATOMIC_FP32=1` 已经把 `uses_bwd_v3=true` 传到 aiter 层（[`fused_attn_ck.cpp:851`](file:///workspace/deps/TransformerEngine/transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp)）
- **路径已经是最优 v3 了**，note 09 推测的 "ck_v3 还能再降一档" 不存在 — gfx950 hd64 没有更快的 v3 变体

---

## 3. 已有 v3 path 的算力检查 — 已经 70 % peak

```
mbs=4, seq=8192, nhead_q=64, nhead_k=8 (GQA), hd=64, causal
单层 fwd  FLOPs = 4·mbs·seq²·nhead_q·hd·0.5 = 2.20 TFLOPs
单层 bwd  FLOPs ≈ 4× fwd                    = 8.80 TFLOPs

MI355X bf16 peak ≈ 1300 TFLOPs/s
@100% util → 6.77 ms / kernel
@ 75% util → 9.02 ms / kernel
@ 50% util → 13.5 ms / kernel
观测值     → 9.6  ms / kernel  ⇒ ≈ 70 % of peak
```

hd64 因为 head dim 小 → 单 thread 计算密度低、HBM 带宽开销占比高，**70 % 已经是接近 hd64 在 MI355 上的 hardware ceiling**。要再快只能改 hd（模型结构变更，不可行）。

---

## 4. 仅剩的可调旋钮 (实测)

| 旋钮 | 候选位置 | 可达变体 | 风险 | 实测 |
|---|---|---|---|---|
| `NVTE_CK_IS_V3_ATOMIC_FP32=0` | `mha_bwd.cu` → 选 a16 path | `bwd_hd64_bf16_causal_a16_rtz.co` (无 atomic-fp32, 无 dq_convert post) | dQ accumulator 改 fp16，可能损失收敛精度 | **见下** |
| `use_turbo_attention: true`   | `transformer_engine_spec_provider.py:104` → `PrimusTurboAttention` | pt.ops.flash_attn_func | **不兼容 SWA**：`primus_turbo.py:368` `assert config.window_size is None` — 会启动直接报错 | 已排除 |

---

## 5. C-series 实验结果

短跑 80 iter，tail-30 mean step time，全部基于 **M4_C7 baseline (=C5 + B1 + B2)**：

| Stage | yaml | 额外环境 | tail-30 mean | Δ vs Cbase | TFLOPs/GPU |
|---|---|---|---:|---:|---:|
| M4_C7_Cbase | M4_C7.yaml | — | **1217.2 ms** | — | 686 |
| M4_C7_C1    | M4_C7.yaml | `NVTE_CK_IS_V3_ATOMIC_FP32=0` | 1382.2 ms | **+165.0 ms (+13.6 %)** | 591 |

C1 显著回退，说明 a32_pssk 路径不仅是 "更精确"，**自身的 main kernel 也比 a16 快**（在 hd64 上 a32 的 dQdKdV 主 kernel 用 atomic 写 dq_acc，省掉了 fp16 atomic 路径上的额外 sync）。结论: a16 路径完全不可用。

> Cbase 1217.2 ms vs note 12 中记录的 M4_C7 = 1218.9 ms，差 1.7 ms，落在热漂移 ±10 ms 范围内 → baseline 完全可复现。

---

## 6. 结论与下一步

### 6.1 确定性结论

- ✅ **note 09 P1 项作废**：`_recompile` 不是 JIT 标志，dense bwd 已经跑在 gfx950 v3 hd64 path 上
- ✅ Dense attn bwd 在 hd64 上已经达到 ~70 % 理论算力上限
- ❌ `NVTE_CK_IS_V3_ATOMIC_FP32=0` 不可用（-13.6 % 严重回退）
- ❌ `use_turbo_attention=true` 与模型 SWA 互斥，无法启用
- ⚖️ **不创建 M4_C8** —— M4_C7 (1218.9 ms / 633 TFLOPs/GPU) 仍然是当前最优 baseline

### 6.2 仍有意义的小动作（独立条目）

| 动作 | 预期 | 风险 | 备注 |
|---|---|---|---|
| `aiter::OpUncachedBwd` 是否能合并？ | -3-5 ms | 低 | 4.7 ms / iter 占比已经很小，可以暂不投入 |
| SWA bwd 切 aotriton (`NVTE_FMHA_BACKEND_SWA_BWD=aotriton`) | ~10 ms | 低-中 | 当前 SWA bwd 29 ms / iter；aotriton 可能更快或更慢，可独立做一组短跑 |

### 6.3 优先级重排（仍指向 D，不是 P1）

| 排名 | 优化 | 预估 wall ↓ | 状态 |
|---:|---|:---:|---|
| **P0** | **mbs=8 重测 + OOM check (D)** | -1.5 ~ -2 % per-step + 同步降 host sync 占比 | 待启动 |
| P1 | host sync MXFP8 / 调 log_freq (note 09 §4.5) | -2.5 ~ -3.3 % | 待启动 |
| ~~P2~~ | ~~attn bwd 切 ck_v3~~ | ~~原假设作废~~ | **CLOSED — 无收益空间** |

---

## 7. 关联资产

| 资产 | 路径 |
|---|---|
| C-chain runner | `/home/xiaompen/mlperf/ablations/M4_chain/run_C_chain.sh` |
| Cbase log      | `/home/xiaompen/mlperf/ablations/M4_chain/logs/M4_C7_Cbase.log` |
| C1 log         | `/home/xiaompen/mlperf/ablations/M4_chain/logs/M4_C7_C1.log` |
| 上游 profile note | [`09_M4_C5_profile_bottleneck`](./2026-04-21_gptoss_09_M4_C5_profile_bottleneck.md) |
| Comm 优化 (B1+B2 ⇒ C7) | [`12_comm_optimization_B_series`](./2026-04-21_gptoss_12_comm_optimization_B_series.md) |
| aiter dispatch 表 | `/workspace/deps/aiter/hsa/gfx950/fmha_v3_bwd/fmha_bwd_dqdkdv.csv` |
| aiter dispatch 代码 | `/workspace/deps/aiter/csrc/cpp_itfs/mha_bwd.cu` |
| TE → aiter binding | `/workspace/deps/TransformerEngine/transformer_engine/common/ck_fused_attn/src/ck_fused_attn_bwd.cpp:676` |
| PrimusTurboAttention | `/workspace/Primus/primus/backends/megatron/core/extensions/primus_turbo.py:304` |
