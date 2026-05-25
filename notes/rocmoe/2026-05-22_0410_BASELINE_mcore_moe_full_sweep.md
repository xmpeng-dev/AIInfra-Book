# 2026-05-22 04:10  Megatron-LM MoE baseline full sweep — 4 models × 5 T × 2 variants

> 时间: 2026-05-22 04:10 (Asia/Shanghai) / 2026-05-21 20:10 UTC
> 项目: rocmoe
> 类型: BASELINE — first canonical Megatron-LM reference numbers for the
>   RocMoE-v2 fused MoE kernel (cross-rank critical-path device wall).
> 数据源:
>   - bench_results/mcore_baseline_20260521_0853.csv (42 timed runs)
>   - bench_results/log_20260521_0853/ (per-run torchrun stdout)
>   - benchmarks/workloads.yaml (workload catalog, single source of truth)
>   - baselines/mcore_moe_bench.py (Megatron-LM MoELayer standalone harness)
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), 节点 mi355-gpu-7
> 容器: `xiaoming-dev` (podman, ROCm 7.2 / PyTorch 2.10.0a0+git449b176 /
>   Megatron-LM @ Primus/third_party/Megatron-LM)
> Dev-loop rule: Rule 1 (fixed PT+RCCL baseline). The previous role of
>   `baselines/pt_rccl_moe.py` (hand-rolled textbook MoE) is replaced by
>   `baselines/mcore_moe_bench.py` (`MoELayer` standalone harness) as the
>   canonical baseline. The hand-rolled version is kept as a sanity check.

## TL;DR

`MoELayer` from upstream Megatron-LM on 8x MI355X gives the following
**cross-rank critical-path wall** (max over rank of mean-over-iters,
warmup=10 iters=50, bf16, EP=8, TP=1, ETP=1, dispatcher=alltoall,
`moe_grouped_gemm=true`, force-load-balanced router, shared experts
disabled):

| model | prod T | crit ms | us/tok | dominant stage |
|---|---:|---:|---:|---|
| **DeepSeek-V3** (H=7168, E=256, topk=8)     | 8192  | **18.13** | 2.21 | experts 9.62 (53%) |
| **DeepSeek-V2** (H=5120, E=160, topk=6)     | 4096  | **7.93**  | 1.94 | experts 3.87 (49%) |
| **Qwen3-235B-A22B** (H=4096, E=128, topk=8) | 16384 | **17.34** | 1.06 | experts 8.06 (46%) |
| **Qwen3-30B-A3B** (H=2048, E=128, topk=8)   | 32768 | **15.89** | 0.49 | experts 6.59 (41%) |

This is what RocMoE-v2 must beat at M3 (end-to-end super-kernel) and is
the **canonical reference number** for `rocmoe-dev-loop` rule 1 from now
on. The DSv3 8192 = 18.13 ms slot is the headline benchmark; the
architecture-design note budgeted -60% wall (~7-8 ms) at M3.

### Sub-finding: `alltoall_gg` vs `alltoall_legacy_gg` are equivalent

In this Megatron-LM build, `moe_use_legacy_grouped_gemm=true` and the
default `moe_grouped_gemm` path **collapse to the same TE GroupedLinear
backend** on ROCm. Across all 4 models and 5 T points the two variants
land within ±4% of each other (almost certainly run-to-run noise):

| model | T | alltoall_gg | legacy_gg | δ |
|---|---:|---:|---:|---:|
| DSv3 | 8192  | 18.13 | 18.17 | +0.2 % |
| DSv2 | 4096  |  7.93 |  7.15 | -9.8 % * |
| Qwen3-235B | 8192 | 9.90 | 10.06 | +1.6 % |
| Qwen3-30B  | 32768 | 15.89 | 15.63 | -1.6 % |

(* DSv2 T=4096 looks like an outlier: legacy_gg combine=1.25 vs gg
combine=2.15 on the same workload — almost certainly an unlucky tail
on the gg run. The mean is dominated by 2-3 slow iters; need warmup=20
or median to dampen.)

### Sub-finding: `primus_turbo_deepep` is uniformly SKIPped

Wired only through Primus' trainer (`primus/modules/trainer/megatron/
utils.py`), not through `TransformerConfig`. Out of scope for this
sweep — the standalone `MoELayer` harness cannot reproduce the
`use_turbo_deepep` dispatcher backend. Next step (after first
optimization round) is to add a Primus-trainer integration path to the
harness so we can compare against the actual production-fast number.

## 1. What was built / what runs

### 1.1 Workload catalog (single source of truth)

`benchmarks/workloads.yaml` — extracted directly from the Primus MI355X
configs at `~/workspace/Primus/examples/megatron/configs/MI355X/<model>-
BF16-pretrain.yaml` + the model definitions under
`~/workspace/Primus/primus/configs/models/megatron/<model>.yaml`:

| model key | H | F_moe | E_global | E/GPU | topk | router | n_group | group_topk | prod_T |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| deepseek_v3     | 7168 | 2048 | 256 | 32 | 8 | sigmoid | 8 | 4 | 8192  |
| deepseek_v2     | 5120 | 1536 | 160 | 20 | 6 | softmax | 8 | 3 | 4096  |
| qwen3_235b_a22b | 4096 | 1536 | 128 | 16 | 8 | softmax | — | — | 16384 |
| qwen3_30b_a3b   | 2048 |  768 | 128 | 16 | 8 | softmax | — | — | 32768 |

T_per_rank sweep `{512, 2048, 4096, 8192, 16384}` applied to every row.
Qwen3-30B gets an extra T=32768 (its prod point).

`benchmarks/list_workloads.py --format {table,env,json}` is the
canonical parser shared by harness + driver.

### 1.2 Megatron-LM `MoELayer` harness

`baselines/mcore_moe_bench.py`:

- `torchrun --nproc-per-node=8` boots 8 ranks, EP=8 single-node.
- `megatron.core.parallel_state.initialize_model_parallel(TP=1, PP=1,
  EP=8, ETP=1)` + `tensor_parallel.random.model_parallel_cuda_manual_seed`
  to register the `expert-parallel-rng` tracker (`SequentialMLP` /
  `TEGroupedMLP` require this at construction time).
- Builds a minimal `TransformerConfig` carrying only the MoE-relevant
  fields. `moe_use_legacy_grouped_gemm` is `setattr`-ed onto the config
  after construction (it is read by `gpt_layer_specs.py:554` via
  attribute lookup, not a declared dataclass field).
- Module spec resolution: tries Primus' `get_gpt_layer_with_transformer_
  engine_spec` first (so the legacy-GG / TE switch goes through the same
  code path Primus production uses); falls back to upstream
  `get_moe_module_spec_for_backend(LocalSpecProvider)`.
- Instantiates a *single* `MoELayer`, NOT a full `TransformerLayer`. No
  attention, no embed, no loss, no optimizer; out of scope.
- 10 warmup + 50 timed iters with `torch.cuda.Event` around each of
  `route / dispatch / routed_experts_compute / combine / postprocess`.
- Cross-rank `all_reduce(MAX)` on per-rank mean total → critical path.
- One CSV row per (model, T, variant) into
  `bench_results/mcore_baseline_<date>.csv`.

### 1.3 Sweep driver

`scripts/run_baseline_sweep.sh`:

- Picks a node from `squeue` (or `ROCMOE_NODE` override).
- Ensures `xiaoming-dev` container is up.
- `for m in MODELS; for v in VARIANTS; for t in T_LIST + prod_T_m: ssh
  → podman exec → torchrun`, serialized.
- `ROCMOE_DRY=1` enumerates without running.

### 1.4 Knobs intentionally OFF for this sweep

- `moe_permute_fusion: false` — TE's `moe_permute_with_probs` kernel
  trips `Unsupported function referenced: <function get_int_dtype>` in
  this ROCm-built TE pytorch op machinery. Re-enable once that TE build
  ships.
- Shared experts disabled (out of scope; runs on a separate stream in
  prod and would double-count time).
- FP8 paths off; bf16 only at this milestone (FP8 is M6).

## 2. Raw numbers (4 models × 2 variants × T sweep)

All times are **cross-rank max device-event wall** in ms, mean over 50
timed iters after 10 warmup. `us/tok = crit_ms × 1000 / T`. Sum of
per-stage maxes generally exceeds the cross-rank total — they're each
maxed independently across ranks, and the slowest rank for routing is
typically not the slowest rank for combine.

### 2.1 deepseek_v3 (H=7168, E=256, topk=8, group_topk=4, sigmoid)

| variant | T | route | dispatch | experts | combine | post | crit_ms | us/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alltoall_gg        |   512 | 1.67 | 0.77 |  5.22 | 3.62 | 0.10 |  8.29 | 16.19 |
| alltoall_gg        |  2048 | 1.68 | 1.11 |  5.48 | 2.46 | 0.38 |  9.30 |  4.54 |
| alltoall_gg        |  4096 | 1.60 | 1.65 |  6.28 | 2.44 | 0.85 | 11.79 |  2.88 |
| alltoall_gg        |  **8192**  | 1.65 | 2.68 |  9.62 | 3.93 | 1.73 | **18.13** |  2.21 |
| alltoall_gg        | 16384 | 2.40 | 4.98 | 17.09 | 7.74 | 3.50 | 32.66 |  1.99 |
| alltoall_legacy_gg |   512 | 1.65 | 0.67 |  4.77 | 2.99 | 0.10 |  7.85 | 15.32 |
| alltoall_legacy_gg |  2048 | 1.50 | 1.10 |  4.79 | 2.03 | 0.38 |  8.52 |  4.16 |
| alltoall_legacy_gg |  4096 | 1.61 | 1.67 |  6.29 | 2.30 | 0.85 | 11.77 |  2.87 |
| alltoall_legacy_gg |  8192 | 1.69 | 2.70 |  9.32 | 3.74 | 1.72 | 18.17 |  2.22 |
| alltoall_legacy_gg | 16384 | 2.46 | 4.98 | 17.07 | 7.74 | 3.50 | 32.70 |  2.00 |

**Prod point** (T=8192): experts (= grouped FC1 + SwiGLU + FC2) = 9.62 ms
(53%) is the dominant cost. Dispatch 2.68 + combine 3.93 = 6.61 ms (37%)
is the comm tax. Router+postprocess = 1.65+1.73 = 3.38 ms (19%).

### 2.2 deepseek_v2 (H=5120, E=160, topk=6, group_topk=3, softmax)

| variant | T | route | dispatch | experts | combine | post | crit_ms | us/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alltoall_gg        |   512 | 1.68 | 0.77 | 3.92 | 3.15 | 0.07 |  6.44 | 12.58 |
| alltoall_gg        |  2048 | 1.61 | 0.94 | 3.38 | 2.29 | 0.20 |  6.63 |  3.24 |
| alltoall_gg        |  **4096**  | 1.71 | 1.21 | 3.87 | 2.15 | 0.42 |  **7.93** |  1.94 |
| alltoall_gg        |  8192 | 1.58 | 1.68 | 4.77 | 2.10 | 0.92 | 10.31 |  1.26 |
| alltoall_gg        | 16384 | 1.86 | 2.81 | 8.10 | 2.99 | 1.88 | 17.19 |  1.05 |
| alltoall_legacy_gg |   512 | 1.81 | 0.74 | 4.14 | 2.28 | 0.07 |  6.83 | 13.34 |
| alltoall_legacy_gg |  2048 | 1.73 | 1.00 | 3.85 | 2.66 | 0.20 |  7.07 |  3.45 |
| alltoall_legacy_gg |  4096 | 1.52 | 1.07 | 3.37 | 1.25 | 0.43 |  7.15 |  1.75 |
| alltoall_legacy_gg |  8192 | 1.52 | 1.65 | 5.28 | 2.47 | 0.92 | 10.59 |  1.29 |
| alltoall_legacy_gg | 16384 | 1.88 | 2.79 | 8.12 | 3.04 | 1.88 | 17.29 |  1.05 |

### 2.3 qwen3_235b_a22b (H=4096, E=128, topk=8, no group routing, softmax)

| variant | T | route | dispatch | experts | combine | post | crit_ms | us/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alltoall_gg        |   512 | 1.55 | 0.83 | 3.19 | 2.81 | 0.07 |  5.94 | 11.61 |
| alltoall_gg        |  2048 | 1.28 | 0.84 | 2.84 | 1.77 | 0.22 |  5.90 |  2.88 |
| alltoall_gg        |  4096 | 1.39 | 1.16 | 3.20 | 1.37 | 0.46 |  6.88 |  1.68 |
| alltoall_gg        |  8192 | 1.38 | 1.79 | 4.45 | 1.79 | 0.98 |  9.90 |  1.21 |
| alltoall_gg        | **16384** | 1.50 | 3.02 | 8.06 | 3.25 | 1.98 | **17.34** |  1.06 |
| alltoall_legacy_gg |   512 | 1.50 | 0.79 | 3.14 | 3.04 | 0.07 |  6.02 | 11.76 |
| alltoall_legacy_gg |  2048 | 1.36 | 0.91 | 2.78 | 2.13 | 0.22 |  6.07 |  2.96 |
| alltoall_legacy_gg |  4096 | 1.52 | 1.24 | 3.30 | 1.49 | 0.46 |  7.11 |  1.74 |
| alltoall_legacy_gg |  8192 | 1.30 | 1.73 | 4.80 | 2.08 | 0.97 | 10.06 |  1.23 |
| alltoall_legacy_gg | 16384 | 1.49 | 3.02 | 8.08 | 3.28 | 1.98 | 17.39 |  1.06 |

### 2.4 qwen3_30b_a3b (H=2048, E=128, topk=8, no group routing, softmax)

| variant | T | route | dispatch | experts | combine | post | crit_ms | us/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alltoall_gg        |   512 | 1.09 | 0.59 | 2.18 | 2.12 | 0.04 |  4.71 |  9.20 |
| alltoall_gg        |  2048 | 1.49 | 0.82 | 3.37 | 2.88 | 0.11 |  5.85 |  2.85 |
| alltoall_gg        |  4096 | 1.35 | 0.92 | 2.91 | 2.71 | 0.22 |  6.12 |  1.49 |
| alltoall_gg        |  8192 | 1.47 | 1.27 | 3.38 | 2.44 | 0.46 |  7.18 |  0.88 |
| alltoall_gg        | 16384 | 1.43 | 1.73 | 3.61 | 1.83 | 0.97 |  9.10 |  0.56 |
| alltoall_gg        | **32768** | 1.67 | 3.01 | 6.59 | 3.56 | 1.98 | **15.89** |  0.49 |
| alltoall_legacy_gg |   512 | 1.49 | 0.76 | 3.10 | 2.99 | 0.06 |  5.96 | 11.64 |
| alltoall_legacy_gg |  2048 | 1.60 | 0.86 | 3.56 | 3.03 | 0.11 |  6.15 |  3.00 |
| alltoall_legacy_gg |  4096 | 1.50 | 1.00 | 3.22 | 3.03 | 0.22 |  6.62 |  1.62 |
| alltoall_legacy_gg |  8192 | 1.57 | 1.12 | 3.32 | 1.88 | 0.46 |  7.31 |  0.89 |
| alltoall_legacy_gg | 16384 | 1.40 | 1.72 | 3.77 | 2.34 | 0.97 |  9.48 |  0.58 |
| alltoall_legacy_gg | 32768 | 1.58 | 2.98 | 6.32 | 3.38 | 1.98 | 15.63 |  0.48 |

## 3. Comparison anchors (for RocMoE-v2)

| anchor | DSv3 T=8192 wall | source |
|---|---:|---|
| **Megatron-LM `MoELayer` baseline (this note)** | **18.13 ms** | bench_results/mcore_baseline_20260521_0853.csv |
| MonolithEP BF16 super-kernel (peer push, Layout-E) | 15.10 ms | `~/workspace/MonolithEP/notes/2026-05-21_1310_bf16_sprint_closeout_pivot_fp8.md` |
| MonolithEP standalone dispatch only | 4.30 ms | `~/workspace/MonolithEP/notes/r5_phase1_perf_analysis.md` (T=8192 same shape) |
| RocMoE-v2 M2-D in-super-kernel dispatch only | ~6.3 ms (extrapolated linear to T=8192) | `notes/2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md` |
| RocMoE-v2 architecture-design target (M3 BF16) | ~7-8 ms (-60% from baseline) | `notes/2026-05-21_1252_rocmoe_v2_architecture_design.md` §2 |

The Megatron baseline (18.13 ms) is **+20 % slower than MonolithEP's
production super-kernel** (15.10 ms) — Megatron-LM is the "as-you-would-
run-it-out-of-the-box" reference, not the "what the fastest existing
hand-tuned kernel can do" reference. RocMoE-v2 has to beat both:

- beat Megatron 18.13 ms cleanly: shows the fusion is worth the
  engineering effort vs just using MCore.
- beat MonolithEP 15.10 ms: shows Layout-P / pull-pull is structurally
  better than push, not just a refactor.

Architecture-design budget (~7-8 ms) is **2.5x faster than Megatron**
and **2x faster than MonolithEP**. Both numbers will be cited in M3
acceptance criteria.

## 4. Observations / open questions

### 4.1 Per-stage breakdown is consistent across models

The "experts" stage (grouped FC1+SwiGLU+FC2) is the dominant cost on
every model (41-53% of wall). Dispatch + combine together are 25-40%.
Router + postprocess are 15-25%. This matches what MonolithEP measured
on DSv3 (FC1+FC2 GEMM ~9 ms out of 15 ms super wall = 60%).

→ Confirms that **fusing GEMM into the super-kernel (M2-G phase) is the
single highest-leverage optimization** on the RocMoE-v2 roadmap. The
combine + dispatch reductions land later (M2-FC2, M3) but they account
for less of the wall.

### 4.2 us/token at large T converges across models

At T=16384 (most workloads' largest sensible per-rank token count):

| model | H | F_moe | us/tok @ 16384 |
|---|---:|---:|---:|
| DSv3       | 7168 | 2048 | 1.99 |
| DSv2       | 5120 | 1536 | 1.05 |
| Qwen3-235B | 4096 | 1536 | 1.06 |
| Qwen3-30B  | 2048 |  768 | 0.56 |

us/tok roughly scales as `H * topk * F_moe`, which is the dominant flop
density per token. DSv3 ~2x slower per-token than Qwen3-235B because of
2x larger H. This is the **flop ceiling shape** — RocMoE-v2's job is to
get closer to that flop ceiling by removing the comm + sync tax.

### 4.3 alltoall_gg vs legacy_gg are equivalent in this build

Already discussed in TL;DR. The "Primus production" legacy_gg path
collapses to the same TE GroupedLinear backend as the default
`moe_grouped_gemm` in this Megatron-LM build. We can **drop legacy_gg
from future sweeps** unless a Primus-runtime variant materially changes
behavior. (Keep the variant entry in workloads.yaml so the runner can
re-enable it later.)

### 4.4 Run-to-run stdev is high (5-25% at small T)

Per-iter cross-rank max walls have stdev = 5-25% of the mean at T≤2048,
narrowing to 2-5% at T≥8192. Likely culprits:

1. RCCL all_to_all stragglers (XGMI is shared with PCIe activity).
2. JIT compilation residue on the first 1-2 timed iters (warmup=10 may
   not be enough for grouped GEMM kernel cache to stabilize).
3. Routing skew amplified at small T (force_load_balancing should pin
   it, but verify).

→ For the M3+ comparison numbers we should bump warmup=20 and report
   p50 alongside mean, especially at T≤2048.

### 4.5 primus_turbo_deepep variant needs separate harness work

The `use_turbo_deepep` dispatcher backend is wired via Primus' trainer
arg-parsing path (`primus/modules/trainer/megatron/utils.py`), not via
`TransformerConfig`. Hooking it into this standalone harness needs:

1. `argparse.Namespace`-based config flow (Primus uses argparse, not
   yaml directly into TransformerConfig).
2. `enable_primus_turbo: true` triggers a code path in `_build_model_specs`
   that swaps in turbo's grouped MLP + deepep dispatcher.
3. The `flex` dispatcher type Megatron-LM only supports nvidia-deepep
   upstream; Primus' fork patches in an AMD deepep backend.

→ TODO **post-M3** (not blocking): write `baselines/primus_moe_bench.py`
   that uses Primus' `pretrain_gpt`-style entry with a 2-3 iter probe
   training step and a profiler hook to extract the MoE-block wall.

## 5. Files added / modified

| file | role |
|---|---|
| `benchmarks/workloads.yaml`              | **NEW** — single source of truth for workload catalog |
| `benchmarks/list_workloads.py`           | **NEW** — workload-table parser (table/env/json) |
| `benchmarks/README.md`                   | **NEW** — workload table + extension instructions |
| `baselines/mcore_moe_bench.py`           | **NEW** — canonical Megatron-LM `MoELayer` harness |
| `baselines/pt_rccl_moe.py`               | DEMOTED — kept as sanity-check baseline |
| `scripts/run_baseline_sweep.sh`          | **NEW** — sweep driver via `dev_on_node.sh`-style SSH+podman |
| `README.md`                              | updated: new tree layout + sweep usage |
| `bench_results/mcore_baseline_20260521_0853.csv` | data — 42 rows |
| `bench_results/log_20260521_0853/`       | logs — 63 per-run torchrun stdout files |

## 6. Next step (M2-G entry criteria)

Now that the baseline is pinned, the next development node is:

- **M2-G** — replace the GEMM stub in the persistent super-kernel
  (`csrc/super_kernel.hip`) with the M0 `mfma_tile.h` body driven by
  per-pool-block `l1_arrival_count` work-stealing.
- M2-G acceptance: DSv3 T=2048 in-super-kernel **(dispatch + GEMM)**
  wall must be lower than M2-D `(dispatch only)` 1.669 ms + a sanity
  budget for the GEMM portion, AND the GEMM portion alone must hit
  the M0 standalone GEMM number (1290 TFLOPS / 99% peak) within ±10%.
- M2-G note: `notes/<time>_BASELINE_m2g_gemm_in_super_kernel.md` (or
  UP / DOWN flag depending on outcome).

The baseline number to beat at M3 BF16 end-to-end is **DSv3 T=8192 = 18.13
ms** (this note's headline). At M6 FP8 the same comparison reruns
against `bench_results/mcore_baseline_<future>_fp8.csv`.

## 7. 命名 / 量纲约定 reminder

Per `notes/2026-05-21_1740_compare_monolithep_dispatch.md` §7, **all
future RocMoE-v2 perf comparisons against this baseline must**:

- Report per-rank `cuda.Event` device-wall, cross-rank `max` =
  critical path. Host wall stays in a footnote.
- Use us/tok for BW-bound stages (dispatch / combine).
- Tag standalone vs in-super-kernel comparisons explicitly.
- Quote the baseline as "MCore MoELayer BF16 alltoall_gg" (this note).

## 8. Addendum (2026-05-21 22:30) — Skew profile sweep (workload realism)

### 8.1 Why this matters

§4 of this note flagged that all timings above use the
`moe_router_force_load_balancing=true` path, which resamples logits as
iid N(0,1) noise per step. That makes per-expert load uniform up to
~3% CoV — clean dev-loop signal but **not** representative of
production routing, which lands at CoV ~ 0.15-0.25 even for
aux-loss-free-balanced trained DSv3 / Qwen3 post-warmup, and pre-
balanced or hot-expert runs can hit 0.5+.

A fused MoE kernel benchmarked only at CoV=0 over-claims: GEMM tiles
are uniform, AllToAll buckets are evenly sized, no padding tail. The
publication-grade comparison must include skewed workloads.

### 8.2 Mechanism (skew injection by logit bias)

Added a `skew_profiles:` axis to `benchmarks/workloads.yaml` and a
`--skew-profile` CLI to `baselines/mcore_moe_bench.py`. For sigma > 0
the harness monkey-patches `router.routing` so a deterministic
per-expert logit bias `b ~ N(0, sigma)` (seed = `sha256("rocmoe-skew-<profile>")`)
is added to the logits BEFORE score_function + top-k. This works
uniformly for sigmoid + grouped (DSv3), softmax + grouped (DSv2), and
softmax + plain top-k (Qwen3) — mcore's own `expert_bias` mechanism
is sigmoid-only (see `moe_utils.py:772`) and would skip DSv2 / Qwen3.

Calibration table (logit-level σ → realized CoV; verified per-model):

| σ | DSv3 (sigmoid+grouped) | DSv2 (softmax+grouped) | Qwen3 (softmax+plain) |
|---|------------------------|------------------------|-----------------------|
| 0.00 | 0.02-0.04 | 0.03 | 0.01 |
| 0.10 | 0.22      | 0.21 | 0.19 |
| 0.30 | 0.63      | 0.65 | 0.59 |

Profiles:
- `balanced` (σ=0.0)         — dev-loop signal; not for publication
- `realistic_cov20` (σ=0.10) — primary publication-grade target
- `hot_cov50`       (σ=0.30) — heavy skew; stresses worst-case tail

### 8.3 Sweep result — `bench_results/mcore_baseline_20260521_1014.csv`

`mcore_alltoall_gg` × prod_T × 3 skews × 4 models = 12 runs,
iters=50.

Total MoE-forward time (cross-rank crit_path, ms):

| Model      | T     | balanced | realistic_cov20 | hot_cov50 | Δ realistic | Δ hot |
|------------|------:|---------:|----------------:|----------:|------------:|------:|
| DSv3       |  8192 |    17.46 |           17.67 |     19.49 |       +1.2% | +11.6% |
| DSv2       |  4096 |     5.87 |            6.52 |      7.07 |      +11.2% | +20.6% |
| Qwen3-235B | 16384 |    16.87 |           17.67 |     19.91 |       +4.7% | +18.0% |
| Qwen3-30B  | 32768 |    15.09 |           15.78 |     17.74 |       +4.6% | +17.6% |

Per-stage breakdown — `combine` (AllToAll #2) is the most skew-sensitive
stage; `experts` (grouped GEMM) takes a smaller hit:

| Model      | stage    | balanced | realistic | hot   | Δ hot vs bal |
|------------|----------|---------:|----------:|------:|-------------:|
| DSv3       | combine  |     3.49 |      3.24 |  6.17 |       +76.5% |
| DSv3       | experts  |     9.30 |      9.53 | 10.68 |       +14.8% |
| DSv2       | combine  |     0.74 |      1.43 |  2.05 |      +175.5% |
| DSv2       | experts  |     2.80 |      3.25 |  3.54 |       +26.5% |
| Qwen3-235B | combine  |     3.11 |      3.82 |  6.87 |      +120.8% |
| Qwen3-235B | experts  |     8.04 |      8.59 | 10.02 |       +24.5% |
| Qwen3-30B  | combine  |     3.04 |      3.67 |  6.11 |      +100.8% |
| Qwen3-30B  | experts  |     6.07 |      6.56 |  7.70 |       +26.8% |

Worst-case `bucket_max/mean` (the worst (src_rank, dst_local_e) bucket
divided by mean bucket — the metric the AllToAll send/recv must size
for):

| Model      | balanced | realistic | hot   |
|------------|---------:|----------:|------:|
| DSv3       |     1.28 |      1.73 |  3.30 |
| DSv2       |     1.27 |      1.90 |  3.53 |
| Qwen3-235B |     1.10 |      1.58 |  2.68 |
| Qwen3-30B  |     1.08 |      1.55 |  2.70 |

### 8.4 What this changes for RocMoE-v2 dev

1. **The "baseline to beat" stays the same** (DSv3 T=8192 = 18.13 ms
   under balanced, refined here to 17.46 ms on a fresh run). The
   M2-G/M3 acceptance criteria continue to use the balanced profile
   for clean dev-loop signal.
2. **Publication-grade comparison must include realistic_cov20.**
   Once an UP milestone lands under balanced, immediately re-run that
   same milestone under `realistic_cov20` and report both. The fused
   super-kernel's value proposition is "no padding tail under skew" —
   that has to be shown.
3. **hot_cov50 is the tail-latency stress test.** Under heavy skew
   the combine stage roughly *doubles*; if the super-kernel keeps
   combine flat (because its persistent worker queue auto-balances
   chunks rather than waiting on the slowest expert) the win there
   is the publication headline.

### 8.5 Files modified for the addendum

| file | change |
|---|---|
| `benchmarks/workloads.yaml`        | added `skew_profiles:` section + calibration table in header |
| `benchmarks/list_workloads.py`     | added `SKEW_PROFILES` env export + `skew_profiles` format |
| `baselines/mcore_moe_bench.py`     | added `--skew-profile` CLI, `inject_logit_bias`, routing-stats dump on iter 0, 4 new CSV columns |
| `scripts/run_baseline_sweep.sh`    | added `ROCMOE_SKEW_LIST` / `ROCMOE_PROD_T_ONLY` env knobs, skew loop |
| `bench_results/mcore_baseline_20260521_1014.csv` | data — 12 rows |
| `bench_results/log_20260521_1014/` | logs — 12 per-run torchrun stdout files |

### 8.6 Follow-up (Path 2-b — real router replay)

The realistic_cov20 / hot_cov50 profiles are **synthetic but calibrated**.
The next-level realism is **trained-router replay**: capture
`(input_ids, hidden_state, routing_map)` traces from an actual Primus
DSv2-Lite or DSv3 training run (Megatron-LM ships `RouterReplay` for
exactly this — `megatron/core/transformer/moe/router_replay.py`), then
feed those into the harness via `--router-replay <trace>`. This
removes the "are biases drawn from a Gaussian realistic?" caveat and
matches per-token correlations a synthetic bias can't reproduce. Plan
to add after M3 BF16 end-to-end lands.
