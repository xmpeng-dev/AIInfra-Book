# [BASELINE] M2-D — Dispatch body lands inside the persistent super-kernel

- **When**: 2026-05-21, 18:30 → 19:30 (UTC+8)
- **Where**: `mi355-gpu-7` (AMD Instinct MI355X, gfx950, 8 GPUs), inside the
  `xiaoming-dev` container (ROCm 7.2 / PyTorch 2.10), node held by SLURM
  job `27091.batch`.
- **What this milestone did**: replace `stub_dispatch_body` in the M2 BASELINE
  super-kernel with the real M1b receiver-pull body, validate bit-exact
  against the same CPU reference that `test_dispatch` uses, and benchmark
  the in-super-kernel dispatch wall against the M1b standalone wall.
- **Flag**: `BASELINE` for the new "dispatch wall when inside the persistent
  super-kernel" axis. The headline is **+100 µs flat across T-sweep vs
  the M1b standalone wall** — that's the cost of the cross-rank
  `phase_barrier<8>` B1 + grid_syncs that the persistent grid pays once
  per launch.

## Context entering M2-D

The M2 BASELINE skeleton already had:

- a persistent `<<<256, 512>>>` grid that role-decodes each WG into
  `DISPATCH / GEMM / FC2_PUSH / TAIL_COMBINE`,
- 5 cross-rank `phase_barrier<8>` calls (B1, B4, B5) wrapped between
  intra-grid `grid_sync<>` arrivals,
- four stub bodies that did nothing but `noop_busy_wait(smoke_iters)`,
- ordering smoke test that passed at zero-config (no MoEConfig in the
  args).

What M2-D had to do: drop in the real dispatch — sender (Stage A) +
B1 + receiver (Stage B) — without forking the implementation. The
receiver-pull is exactly the M1b code (`csrc/include/rocmoe/dispatch_body.h`),
so the M1b -19 % win must carry over.

## What landed in code

- `csrc/include/rocmoe/types.h`:
  - moved `kSubWGs` (the M1b-tuned dispatch sub-WG count) into types.h.
    Previously it lived in `dispatch.h` and `kNDispatchWGs` (in types.h)
    was stale at 32 from when `kSubWGs` was 4 — now both derive from
    the same constant. Added a `static_assert` that pins
    `kNDispatchWGs == kMaxRanks * kSubWGs` so the persistent grid's
    DISPATCH role sees the exact same `(partner, sub_wg)` mapping the
    standalone dispatch kernel uses.
  - bumped `ROCMOE_N_DISPATCH_WGS` default from 32 → 64
    (= `kMaxRanks * kSubWGs`) and `ROCMOE_N_GEMM_WGS` from 216 → 184
    so `kNTotalWGs` stays at 256.
- `csrc/include/rocmoe/dispatch.h`: kept `kDispatchWGs` as an alias of
  `kNDispatchWGs` for backward compat; deleted the local `kSubWGs`
  definition.
- `csrc/include/rocmoe/super_kernel.h`: extended `SuperKernelArgs` v0 → v1
  with `cfg` (`MoEConfig`) and `num_actual_tokens`, so the dispatch body
  has its layout math without a separate args struct. New
  `launch_super_skeleton_kernel` decl alongside the production
  `launch_super_kernel` (see split rationale below).
- `csrc/super_kernel.hip`:
  - real `dispatch_role_stage_a` and `dispatch_role_stage_b` thin
    wrappers around `dispatch_sender_stage` and `dispatch_receiver_stage`,
    same `(partner, sub_wg)` decode as the standalone kernel.
  - **kernel split** — production `rocmoe_super_kernel` only runs Stage
    A → B1 → Stage B; debug-friendly `rocmoe_super_skeleton_kernel`
    keeps the full 5-phase scaffold (B1 + B4 + B5) for the smoke test.
    The reason for the split is the headline performance bug below.
- `csrc/launcher.{h,cpp}`: `launch_on_device` was already wired in M2
  BASELINE; here it now correctly references `SuperKernelArgs` (which
  includes `cfg`).
- `tests/test_super_kernel_dispatch.hip` (new) — bit-exact vs the same
  CPU reference `test_dispatch` uses. Verifies pool / topk_wts /
  pool_src_meta / l1_arrival_count layouts are identical between
  standalone and in-super-kernel runs.
- `tests/test_super_kernel_skeleton.hip` — switched from
  `launch_super_kernel` to `launch_super_skeleton_kernel`; otherwise
  unchanged.
- `benchmarks/bench_super_dispatch.hip` (new) — measures dispatch wall
  when the body runs inside the persistent grid. Uses the same per-rank
  `hipEvent` pair as `bench_dispatch` for apples-to-apples comparison
  + dumps per-stage cycle telemetry (`StageA→B1`, `StageB→B4`, `B4→B5`).
- `CMakeLists.txt` — adds the new test (ctest #8 `test_super_dispatch`)
  and the new bench.

## Where the headline performance bug came from

First implementation put both the dispatch role bodies AND the
post-Stage-B B4 / TAIL_COMBINE / B5 code in the same kernel function.
Result at T=2048:

| variant                                                  | dev wall | vs M1b standalone |
|----------------------------------------------------------|----------|-------------------|
| M1b standalone dispatch                                  | 1.57 ms  | (BASELINE)        |
| M2-D super-kernel (full B1+B4+B5, stubs idle)            | **5.66 ms** | **+261 % (DOWN)** |
| M2-D super-kernel (B4+B5 + stubs stripped, B1 only)      | **1.67 ms** | **+6 % (BASELINE)** |
| ↑ same kernel, **with full grid (256 WGs incl. stubs)**, just no B4/B5 | **1.67 ms** | **+6 %** |
| ↑ same kernel, **with shrunk grid (64 dispatch WGs only)**, full B4/B5 still in | 5.66 ms | **+261 %** |

The two "shrink one axis at a time" rows are the diagnostic. They show:

1. The 4× regression is **not** stub-WG spin contention on
   `grid_sync<2>` — shrinking the grid to dispatch-only WGs left the
   wall at 5.66 ms.
2. The 4× regression **is** caused by the B4 + B5 + TAIL_COMBINE code
   sharing the same kernel function as Stage B's hot loop —
   stripping that post-loop code (which executes *after* the receiver
   pull, with no semantic dependency on it) brought the wall back to
   M1b + 6 %.

Per-stage cycle telemetry from rank 0 with full B4/B5 in:

```
DISPATCH role: total=6337 us | StageA->B1=187 us | StageB->B4=6099 us | B4->B5=53 us
```

→ Stage B ran in 6099 µs vs the ~1500 µs the same code took standalone.
The extra 4.5 ms appears entirely inside the receiver-pull's inner loop,
not anywhere the new barriers actually execute.

**Diagnosis** (most likely): when B4 + B5 + TAIL_COMBINE bodies are
present in the same `__global__` function, the HIP compiler allocates
substantially more VGPRs across the function and / or grows the
function's I-cache footprint. The receiver pull's inner loop
(`cooperative_b128_copy`) has very tight pressure — when wave occupancy
drops or instruction fetches start missing the I-cache, XGMI peer-read
latency stops being hidden and the kernel runs much slower despite
doing identical work. This is the same family of compiler-level
performance pothole that
[`mi355_hardware_aware`](../.cursor/skills/mi355_hardware_aware/SKILL.md)
warns about for CDNA4 — `__launch_bounds__(kWGSize, 1)` does not
itself cap VGPR usage; it only sets the minimum WGs/CU target.

**Engineering response**: split the kernel. The production
`rocmoe_super_kernel` only contains the Stage A + B1 + Stage B code
that actually runs in M2-D. The full 5-phase scaffold lives as a
separate `rocmoe_super_skeleton_kernel` that the smoke test calls.
Each new role in M2-G / M2-FC2 / M2-COMB will land its own barrier
inside the production kernel **at the moment the role's real body is
also there** — that way the receiver pull is never sharing a function
with code that's just dead weight.

## Bench results (final, with split kernels)

`bench_super_dispatch` measures the production `rocmoe_super_kernel`
device wall via per-rank `hipEvent`s, on 8× MI355X. M1b numbers
re-measured this round with the same harness for apples-to-apples.

| T    | M1b standalone | **M2-D in super-kernel** | Δ (abs) | Δ (%)  |
|------|---------------:|-------------------------:|--------:|-------:|
| 512  | 0.432 ms       | **0.509 ms**             | +0.077 ms | +18 %  |
| 1024 | 0.798 ms       | **0.889 ms**             | +0.091 ms | +11 %  |
| 2048 | 1.568 ms       | **1.669 ms**             | +0.101 ms | +6.4 % |
| 4096 | 3.086 ms       | **3.219 ms**             | +0.133 ms | +4.3 % |

The absolute Δ is **flat at ~100 µs** across the T sweep — the
persistent grid pays the same fixed scaffold cost regardless of token
count. Component breakdown:

- B1 cross-rank `phase_barrier<8>` ≈ 55-60 µs (one-time per launch)
- 2 × `grid_sync<>` around B1 ≈ 30-40 µs
- Stage A sender (very small at low T, ~30 % of stage A→B1 at T=4096)

The 6 % regression at the DSv3 production point (T=2048) is a real
artifact of the persistent-grid scaffold and disappears once
`launch_super_kernel` is the sole launch for the entire MoE forward
(M2-G/-FC2/-COMB will eliminate the standalone dispatch launch + 3
follow-on launches the PT+RCCL baseline pays today, more than
compensating).

## Correctness

`ctest` is now 8/8 green:

- 5 `test_gemm_*` tile shapes (M0)
- `test_dispatch_smoke` (M1, standalone dispatch bit-exact)
- `test_super_skeleton` (M2 BASELINE, 5-phase scaffold ordering)
- `test_super_dispatch` (M2-D, in-super-kernel dispatch bit-exact)

The new `test_super_dispatch` compares pool / topk_wts / pool_src_meta /
l1_arrival_count between the in-super-kernel run and the same CPU
reference `test_dispatch` uses. Per-(le, src_rank) bucket multiset
comparison sorts by encoded `src_meta` to absorb the sender's
non-deterministic intra-bucket slot ordering, identical to the
standalone test's scheme.

Verified bit-exact at four shapes:
- `(R=8, E=32, topk=4, T=256, H=256, block_m=32)` (the ctest entry)
- `(R=8, E=32, topk=8, T=512,  H=7168, block_m=32)`
- `(R=8, E=32, topk=8, T=1024, H=7168, block_m=32)`
- `(R=8, E=32, topk=8, T=2048, H=7168, block_m=32)` ← DSv3 production

## Why this is BASELINE and not UP / DOWN

`UP` would mean better than M1b standalone wall — the +100 µs scaffold
tax means it isn't, on a *standalone-launch* comparison axis.
`DOWN` would imply the M2-D code is worse on the M2 BASELINE axis (it
isn't — there was no in-super-kernel dispatch number before this).

This is a **new measurement axis**: dispatch wall when fused into the
persistent grid. It establishes the baseline that the next milestones
will be measured against. Specifically:

- M2-G (real GEMM body) is expected to overlap with Stage B's pull
  (chunk-level CCO via per-pool-block `l1_arrival_count` scoreboard).
  The in-super-kernel dispatch wall reported here is the worst case
  before any overlap; M2-G should *reduce* the effective dispatch
  contribution to total forward latency by overlapping it with FC1.
- M2-COMB will reintroduce B5 inside the production kernel only when
  TAIL_COMBINE has a real body that needs it. The split-kernel design
  means today's hot path doesn't pay for it.

## Engineering rules applied this round

Per the `rocmoe-dev-loop` skill:

1. **Build/run on allocated cluster nodes inside the container** —
   used `bash scripts/dev_on_node.sh build / test / raw "..."` exclusively;
   when investigating the regression I shrank the grid via a per-build
   `-DROCMOE_N_GEMM_WGS=0 -DROCMOE_N_TAIL_COMBINE_WGS=0` rather than
   patching the constants in tree, so the diagnostic was reproducible
   without leaving uncommitted state.
2. **Per-phase test cases** — `test_super_dispatch` is the M2-D phase
   test. The skeleton smoke test stays in place and continues to
   validate the 5-phase ordering scaffold via `launch_super_skeleton_kernel`,
   so removing B4/B5 from the production hot path didn't lose any
   sync-correctness coverage.
3. **Test before flagging** — every code revision in this round went
   through `bash scripts/dev_on_node.sh build && test`, with the
   regression caught BEFORE writing any flag.
4. **Note with explicit flag** — this file. Flag = `BASELINE` (new
   measurement axis), with the `DOWN` 5.66 ms result along with the
   `BASELINE` 1.67 ms result both fully recorded; the regression's
   diagnosis (kernel-fusion VGPR / I-cache pressure) is captured so a
   future me / a colleague doesn't re-discover it the hard way when
   adding more roles.

## Files touched this round

- `csrc/include/rocmoe/types.h` — `kSubWGs` moves here; `kNDispatchWGs`
  derives from it; `kNGemmWGs` 216 → 184; static_assert added.
- `csrc/include/rocmoe/dispatch.h` — `kSubWGs` removed (now alias
  via `kNDispatchWGs`); old `kDispatchWGs` kept as alias.
- `csrc/include/rocmoe/super_kernel.h` — `SuperKernelArgs` v1
  (adds `cfg` + `num_actual_tokens`); `launch_super_skeleton_kernel`
  decl.
- `csrc/super_kernel.hip` — added `dispatch_role_stage_a/b` wrappers
  around `dispatch_body.h`; split into `rocmoe_super_kernel`
  (production hot path: Stage A → B1 → Stage B only) and
  `rocmoe_super_skeleton_kernel` (full 5-phase scaffold for the smoke
  test); detailed comment on the kernel split rationale.
- `tests/test_super_kernel_skeleton.hip` — switched to
  `launch_super_skeleton_kernel`.
- `tests/test_super_kernel_dispatch.hip` (new, 304 LOC).
- `benchmarks/bench_super_dispatch.hip` (new, 235 LOC) — per-rank
  `hipEvent` pair + per-stage cycle telemetry on rank 0.
- `CMakeLists.txt` — adds `test_super_dispatch` ctest + `bench_super_dispatch`.

## Reproducing

```
bash scripts/dev_on_node.sh build
bash scripts/dev_on_node.sh test                                       # 8/8 PASS

# Standalone dispatch wall (M1b)
bash scripts/dev_on_node.sh raw "build/bench_dispatch       8 32 8 2048 7168 32"

# Same dispatch body, in the persistent super-kernel (M2-D)
bash scripts/dev_on_node.sh raw "build/bench_super_dispatch 8 32 8 2048 7168 32"
```

## Next directions

1. **M2-G** — drop the M0 `mfma_tile.h` GEMM into `stub_gemm_body`,
   driven by per-pool-block `l1_arrival_count` work-stealing (one
   pool block = one MFMA tile's worth of input rows). At this point
   B4 comes back in the production kernel because GEMM's output
   needs to gate FC2 push. Expect the receiver pull to start
   overlapping with FC1's MFMA work — that's where the dispatch
   wall measured here gets *hidden*, not just paid.
2. **M2-FC2** — keep FC2 push inline in the GEMM epilogue at first;
   only split into a dedicated FC2_PUSH role if profiling shows
   benefit.
3. **M2-COMB** — bring back B5 alongside the real TAIL_COMBINE body.
   Watch for VGPR pressure in the production kernel as more roles
   land; if Stage B's wall starts climbing again, split functions
   with `__attribute__((noinline))` to keep their compilation
   contexts isolated.
