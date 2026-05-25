# [BASELINE] Super-kernel bench harness + multi-skew correctness scaffold

- **When**: 2026-05-22, 09:00 → 09:45 (UTC+8)
- **Where**: `mi355-gpu-7` (AMD Instinct MI355X, gfx950, 8 GPUs), inside
  the `xiaoming-dev` container (ROCm 7.2 / PyTorch 2.10), node held by
  SLURM job `27091.batch`. Build host: `aac14-slurm-controller-p`.
- **What this milestone did**: design and land a regression-grade
  benchmark harness for the RoCMoE super-kernel itself (sister to the
  Megatron-LM baseline harness), plus a multi-iter / multi-skew
  correctness scaffold so future M2-G / M2-COMB rounds keep producing
  bit-exact dispatch outputs while their own bodies land.
- **Flag**: `BASELINE`. No perf comparison (the kernel itself is
  unchanged from M2-D BASELINE 2026-05-21 19:30) — this is harness-only.
- **CSV schema invariant**: super-kernel rows now share the exact column
  layout of `mcore_moe_bench.py` so a single plot pipeline can compare
  `mcore_alltoall_legacy_gg` vs `rocmoe_m2d` (and later `rocmoe_m2g`,
  `rocmoe_m2comb`) per (model, T_per_rank, skew_profile) point.

---

## 1. Problem the harness solves

After the [M2-D BASELINE landing](./2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md)
the super-kernel has a real dispatch body, and the [Megatron-LM baseline
sweep](./2026-05-22_0410_BASELINE_mcore_moe_full_sweep.md) generates
publication-grade timings against the four production MoE workloads
(DSv3 / DSv2 / Qwen3-235B / Qwen3-30B) at three skew profiles
(`balanced` / `realistic_cov20` / `hot_cov50`). What was missing:

1. **No regression bench for the super-kernel itself** — every M2 round
   so far reads the in-kernel cycle log manually
   (`bench_super_dispatch.hip` is a single-stage isolation bench, not a
   row-emitting forward bench).
2. **No CSV schema shared between super-kernel and baseline** — comparing
   `rocmoe_m2d dispatch_ms` against `mcore_alltoall_legacy_gg dispatch_ms`
   today requires hand-aligning two different output formats.
3. **No multi-iter correctness validation** — `test_super_kernel_dispatch`
   runs once on a fresh sym buffer. Multi-iter back-to-back launches
   exercise the persistent grid's barrier ping-pong + counter-state
   discipline, and that has never been tested.
4. **No multi-skew correctness validation** — `hot_cov50` produces
   `bucket_max/mean ≈ 6.3x` (vs ≈ 1.4x at `balanced`), which exercises
   slot-overflow paths the existing tests bypass.

## 2. What was added

| file                                    | purpose                                  |
|-----------------------------------------|------------------------------------------|
| `benchmarks/bench_super_kernel.hip`     | Unified forward bench. CSV schema = mcore_moe_bench.py. One row per (model, T, skew_profile). Variant tag carries the milestone (`rocmoe_m2d` -> `rocmoe_m2g` -> `rocmoe_m2comb`). |
| `benchmarks/bench_routing.h` (extended) | Adds `Dist::PlainTopK` (Qwen3 family, softmax + plain top-k) and a logit-bias skew injection (sigma -> deterministic per-expert N(0, sigma) bias added BEFORE the score function) calibrated against the mcore harness. Same calibration table: `sigma 0.0/0.10/0.30 -> CoV ≈ 0.04/0.21/0.73`. |
| `tests/test_super_kernel_e2e.hip`       | Multi-iter (1+3 back-to-back) and multi-skew (3 profiles) correctness scaffold with explicit M2-G / M2-COMB plug-in points. |
| `scripts/run_super_kernel_sweep.sh`     | Sister of `run_baseline_sweep.sh`. Same node-pick / yaml-enumeration / env-knob shape. |
| `benchmarks/list_workloads.py` (ext.)   | Emits `M_<model>_*` per-model fields + `SIGMA_<profile>` per-skew sigmas in `--format env`, so the new shell driver can flatten everything into flat CLI args for the HIP harness. |
| `CMakeLists.txt`                        | Registers `bench_super_kernel`, `test_super_kernel_e2e`. |
| `benchmarks/README.md` + `README.md`    | Documents the harness + sweep flow. |

## 3. CSV schema bridge (super-kernel <-> mcore baseline)

The super-kernel doesn't have a separate "router fwd" stage (the topk
indices are pre-computed on host) and folds postprocess (weighted sum)
into combine. Stages map onto the persistent kernel's per-WG cycle log
as follows:

| `mcore_moe_bench.py` column | super-kernel source                                 | populated since |
|-----------------------------|-----------------------------------------------------|-----------------|
| `route_ms`                  | 0.0 (host-precomputed `topk_idx`; out of scope)     | always          |
| `dispatch_ms`               | DISPATCH role bucket 0 -> 6 median (per-WG, rank 0) | M2-D            |
| `experts_ms`                | GEMM role bucket 0 -> 6 median                      | M2-G (today: 0) |
| `combine_ms`                | TAIL_COMBINE role bucket 0 -> 6 median              | M2-COMB (today: 0) |
| `postprocess_ms`            | 0.0 (folded into combine)                           | always          |
| `total_ms_*` (rank 0)       | hipEventElapsed on rank 0                           | always          |
| `crit_path_ms`              | hipEventElapsed cross-rank max, p50 over iters      | always          |
| `us_per_token`              | `crit_path_ms * 1e3 / T`                            | always          |

**Milestone-aware blanking**: stub roles still log a role-start /
role-end wall (they sit through the kernel's grid_sync + cross-rank
barriers). The harness recognises milestone tags via a substring match
on the variant string (`m2g` -> GEMM real, `m2comb` -> COMBINE real)
and blanks stub-role walls in the CSV so rows are honest. Raw values
are still printed to stdout for diagnostic transparency.

## 4. Multi-iter / multi-skew test design

`test_super_kernel_e2e.hip` runs each of three skew profiles
(`balanced` / `realistic_cov20` / `hot_cov50`) with both `n_iters=1`
(sanity, mirrors `test_super_kernel_dispatch`) and `n_iters=3`
(back-to-back). It verifies that the FINAL iter's outputs match the
CPU reference bit-exact.

Important nuance discovered during bring-up: the persistent kernel
uses `atomic_add_agent<u32>` on `l1_arrival_count` and never resets
it (the `B5 workspace_clean` phase is reserved for M2-COMB). Without
an explicit between-iter reset the `n_iters=3` check fails because
`l1_arrival_count` accumulates 3x the per-iter value. The test now
calls a small `reset_counters()` helper between iters that zeroes
`{l1_arrival_count, expert_recv_count, send_done_flag,
barrier_signal}` — this is a stand-in for the future B5 body and
documents exactly what counters M2-COMB needs to clear.

Hot-skew specifically exposes overflow paths balanced runs don't hit:
at sigma=0.30 the realised `bucket_max / mean` is 6.3x (vs 1.4x
balanced), and the symmetric workspace's `max_recv_per_e_per_src`
must be sized accordingly. The harness defaults `--max-recv-factor`
to 8 (covers worst observed bucket with 27% headroom) and lowers it
to 4 for balanced-only runs. The fatal "`max_recv` < observed bucket
max" check in the harness gives a one-line diagnosis if a future
skew profile pushes past the cap.

## 5. Plug-in points for future milestones

Both the harness and the test were designed so M2-G / M2-COMB land
without reshaping the harness:

| milestone | new code                                | what extends                                    |
|-----------|-----------------------------------------|-------------------------------------------------|
| M2-G      | real GEMM body in `super_kernel.hip`    | `bench_super_kernel`: variant tag `rocmoe_m2g`  |
|           |                                         | -> milestone_flags() returns gemm_real=true     |
|           |                                         | -> CSV `experts_ms` populated automatically     |
|           | + `expect_gemm()` body in test_e2e      | bit-exact FC1+SwiGLU+FC2 vs CPU (abs<5e-2 / rel<2 %) |
| M2-FC2    | real FC2_PUSH body                      | (covered by combine; no separate body)          |
| M2-COMB   | real TAIL_COMBINE body                  | variant tag `rocmoe_m2comb` -> combine_real=true |
|           |                                         | -> CSV `combine_ms` populated                    |
|           | + `expect_combine()` body in test_e2e   | bit-exact `out[t,h] = sum_k topk_wts[t,k] * fc2_out[slot]` |
|           | + B5 `workspace_clean` real body        | reset_counters() in test e2e becomes a no-op:    |
|           |                                         | future iters self-reset inside the kernel        |

`test_super_kernel_e2e`'s top-level driver iterates `(skew, n_iters) in
{balanced, realistic_cov20, hot_cov50} x {1, 3}` and calls
`expect_dispatch()` (today's only check) plus the future `expect_gemm`
and `expect_combine` plug-in stubs. Filling those stubs is the entire
correctness work for M2-G / M2-COMB.

## 6. Verification

```text
# build + ctest, mi355-gpu-7
# 9/9 pass (added test_super_e2e_small)

# smoke run, deepseek_v3 T=2048 dsv3
balanced        : crit=3.58 ms  dispatch=3.78 ms  CoV=0.04  bucket_max/mean=1.41x
realistic_cov20 : crit=3.91 ms  dispatch=4.04 ms  CoV=0.22  bucket_max/mean=1.97x
hot_cov50       : crit=4.52 ms  dispatch=4.19 ms  CoV=0.73  bucket_max/mean=6.27x
```

Numbers are tabulated under the `rocmoe_m2d` variant in
`bench_results/rocmoe_super_<date>.csv`. Skew calibration matches the
mcore Python harness within stochastic-sample noise (bench_routing's
LCG vs torch.randn produce the same distribution shape but different
draws; CoV/bucket-ratio numbers are within 5-10 % of the calibration
table in `workloads.yaml`).

## 7. Next steps

- **M2-G** — replace the GEMM stub with the M0 `mfma_tile.h` body
  driven by per-pool-block `l1_arrival_count` work-stealing. At this
  point B4 re-enters the production kernel, dispatch peer-reads start
  overlapping with FC1 MFMA bursts, and `bench_super_kernel` starts
  producing meaningful `experts_ms` numbers automatically (no harness
  change needed; just bump the variant tag to `rocmoe_m2g`).
- **CI add-on** — the e2e test currently runs the small (8 ranks, 32
  experts, T=256) configuration in ctest. A follow-up adds a
  production-shape variant (DSv3, T=2048) that runs on demand
  (`ROCMOE_E2E_PROD=1 bash scripts/dev_on_node.sh test`).
- **Comparison plot** — extend the existing baseline HTML generator to
  consume two CSVs (mcore + rocmoe) and render side-by-side bars per
  (model, T, skew) point. Trivial once a `rocmoe_m2g` row exists.

## 8. Files (canonical paths)

```
benchmarks/bench_super_kernel.hip
benchmarks/bench_routing.h        (extended: PlainTopK + skew profiles)
benchmarks/list_workloads.py      (extended: M_<model>_*, SIGMA_*)
benchmarks/README.md              (extended: super-kernel section)
tests/test_super_kernel_e2e.hip
scripts/run_super_kernel_sweep.sh
CMakeLists.txt                    (extended: 2 new targets)
README.md                         (extended: quick-start)
```
