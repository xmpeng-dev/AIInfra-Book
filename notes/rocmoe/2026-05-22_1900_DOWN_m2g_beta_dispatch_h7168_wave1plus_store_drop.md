# 2026-05-22 19:00 — M2-G β DOWN: dispatch H≥4096 wave-1+ stores dropped, M2-G β blocked

| field          | value                                                            |
|----------------|------------------------------------------------------------------|
| when           | 2026-05-22 18:00–19:00 (UTC+8, mi355-gpu-7)                      |
| who            | xiaoming + agent                                                 |
| project        | RocMoE-v2 / M2-G β (production T/H validation)                   |
| flag           | DOWN — M2-G β cannot proceed without dispatch correctness fix    |
| status         | pre-existing dispatch bug uncovered, surgery deferred            |
| container      | `xiaoming-dev` Podman, image `docker.io/rocm/primus:v26.2`       |
| HW             | 8 × MI355X (gfx950, CDNA4), 1 node                               |
| build          | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` (gfx950, -O3)   |

## 1. Goal (going in)

M2-G α had landed real FC1 GEMM + SwiGLU in the persistent super-kernel
with chunk-overlap (dispatch ↔ FC1) and was passing `test_super_e2e_fc1`
at the small (T=128, H=256, F=64) ctest shape. M2-G β was supposed to:

1. Wire the super-kernel up at **production** shape T=2048 H=7168 F=2048.
2. Bit-exact-validate FC1 against host CPU SwiGLU(X·W^T).
3. Capture wall numbers (dispatch wall / FC1 wall / integrated wall) and
   compare them against the 1545 FLAT FC1 roofline (3.25 ms, 1185
   TFLOPS) and M1c-D dispatch wall (5.42 ms balanced), to confirm
   chunk-overlap really hides max(D, FC1).

## 2. What was actually done

### 2.1 Pre-flight fixes

| change                                                  | reason                                                                                       |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `tests/test_super_kernel_e2e.hip`: OpenMP-parallelize `expect_fc1_act` | serial CPU SwiGLU(X·W^T) at T=2048 H=7168 F=2048 = ~8–15 min/rank; with `#pragma omp parallel for` over pool-blocks it drops to ~30 s on 96 cores |
| `CMakeLists.txt`: add `-fopenmp` compile + link to the e2e test       | HIP language doesn't auto-forward `OpenMP::OpenMP_CXX`; pass explicitly         |

Verified `test_super_e2e_fc1 8 32 4 128 256 32 64` still PASSes
(dispatch bit-exact × 3 skews × {1,3} iters + FC1+SwiGLU bit-exact).

### 2.2 First production-H attempt → exposed pre-existing dispatch bug

`test_super_kernel_e2e 8 32 4 256 7168 32 2048` (T=256 H=7168 F=2048)
**FAILed** with `(le=0,src=0) sort#0 token bytes differ` on rank 6-7.
Setting `F=0` (skip FC1 body, leave only dispatch + grid-sync) **STILL
FAILed** the same way. So the failure was in the dispatch stage, not in
the M2-G α FC1 plumbing.

Direct `test_dispatch 8 32 4 128 7168 32` (the standalone Phase-1
dispatch test, unchanged since M1c-D landed) reproduces the same
pattern: ranks 6-7 token-bytes-differ. Confirmed by `git stash` + a
clean rebuild of the **M1c-D commit** (`6fa180f`) — same FAIL. So this
is **not** a regression introduced by M2-G α. It is a pre-existing
dispatch correctness bug at H ≥ 4096 that no test in the M1c-D set
exercised (the only ctest entry runs at H=256; benches run at H=7168 but
were not bit-exact-checked).

### 2.3 Bisect: what is broken

Added a diagnostic in `test_dispatch.hip` to dump first-diff / last-diff
hidden-dim indices and per-element got/exp values for the failing slot.
For T=128 H=7168 on rank 0:

```
[FAIL rank=0] (le=0,src=0) sort#0 token bytes differ (rg=2 re=5)
    n_diff=5120 first=512 last=7167  (H=7168)
    h=512 got=0.000000 exp=0.542969
    h=513 got=0.000000 exp=0.535156
    h=514 got=0.000000 exp=0.542969
    ...
```

**Pattern across all failing ranks**:

- `first_diff = 512` (bf16) = `uint4 #64` (8 bf16 per uint4)
- `last_diff  = 7167` = last bf16 in the row (`uint4 #895`)
- `n_diff    = 5120` = 5120 bf16 zeroed; remaining
  6656 − 5120 = 1536 bf16 in that range happen to coincide with the
  expected near-zero value
- `uint4 [0, 64)` (= `bf16 [0, 512)`) is bit-exact correct
- `uint4 [64, 896)` is all zero in `expert_token_pool`

In the receiver-side per-row `cooperative_b128_copy` with `kWGSize=256`,
`kUnroll=2`, `chunk=896`:

| who                                 | what they would store     | observed result    |
|-------------------------------------|----------------------------|--------------------|
| wave 0 (lanes 0–63), iter 0, u=0    | `dst[uint4 0…63]`         | CORRECT            |
| wave 1 (lanes 64–127), iter 0, u=0  | `dst[uint4 64…127]`       | **zero**           |
| wave 2 (lanes 128–191), iter 0, u=0 | `dst[uint4 128…191]`      | **zero**           |
| wave 3 (lanes 192–255), iter 0, u=0 | `dst[uint4 192…255]`      | **zero**           |
| wave 0, iter 0, u=1                 | `dst[uint4 256…319]`      | **zero**           |
| … all higher waves and iters        | …                          | **zero**           |

Only **wave 0's first store** lands. This **exactly matches** the
hardware hazard the M1c-D comment in `dispatch_body.h:415-425` already
described:

> "(2026-05-22 bisect) … intermittently drops the stores from waves 1…N
> — wave-0 lanes write the correct first 64 uint4 to
> expert_token_pool, lanes 64+ emit stores that are never visible to
> the host hipMemcpy after hipStreamSynchronize."

That comment claimed the LDS-staged form fixes it. The bisect today
shows the LDS-staging mitigates the per-call partial-wave-active form
(wave-N with few lanes), but the full-WG form at `kWGSize=256` +
`chunk=896` still loses wave-1+ stores.

### 2.4 Fixes that were attempted and did NOT cure the wave-store-drop

| attempt                                                                    | result | kept? | rationale                                                                                                                                                                          |
|----------------------------------------------------------------------------|--------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `kUnroll=2 → 1` in `cooperative_b128_copy`                                 | broke H=256 baseline (token differ + intermittent illegal-mem-access) | reverted | partial-wave bound check is OK at `kUnroll=1` but per-store cadence regressed worse cases; not the cause                                                                          |
| `ROCMOE_DISPATCH_NAIVE_COPY=1` (replace LDS-staged copy with naïve `for (i=tid; i<n; i+=kWGSize) d[i] = s[i];`) | same FAIL pattern | removed | rules out cooperative_b128_copy as the bug — the hazard is in the underlying store path |
| `ROCMOE_DISPATCH_USE_PACKED_OUTBOX=0` (M1b: read peer.input_token_row(src_t) directly, no packed_outbox staging) | same FAIL pattern | not committed | rules out packed_outbox L2 footprint as the cause |

### 2.5 Fixes that ARE structurally correct and were kept (didn't fully fix the bug but didn't regress anything at H=256)

| change                                                                                   | scope        | reason                                                                                                                                                                                                                                                                                |
|------------------------------------------------------------------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `csrc/dispatch.hip`: add `grid_sync<3>` between Phase C and `phase_barrier`              | dispatch     | Without this, blk==0's `phase_barrier` triggers the cross-rank arrival signal before blk=N (N≠0) finishes its Phase C, so the peer's receiver may wake from `phase_barrier` and read a not-yet-published `recv_count`. Verified at 2-rank H=7168 the symptom shifted from `recv_count=0` to `token bytes differ` once this barrier was added. |
| `dispatch_body.h`: end of Phase A and end of Phase B promote `__threadfence()` → `fence_system()` | dispatch     | Each sub_wg's peer-mapped writes (src_index_table in Phase A, packed_outbox in Phase B) need a SYSTEM-scope drain so the *peer's* receiver-side L2 sees them via XGMI. AGENT scope only drains to local L2.                                                                          |
| `dispatch_body.h`: receiver entry `fence_system() + __syncthreads()`                     | dispatch     | Forces the receiver-side WG's L1$/L2 to drop stale lines mapping to peer-written addresses (cross-rank `phase_barrier` only acquires SYSTEM-visibility into blk==0; `grid_sync<2>` propagation to other blks is AGENT scope).                                                          |
| `dispatch_body.h`: receiver reads of `expert_recv_count` and `src_index_ptr` → `atomic_load_system_acquire` | dispatch     | The writer lives on a different GPU; AGENT acquire wouldn't invalidate stale L2 lines for these peer-written addresses.                                                                                                                                                              |

These are all defensively-correct on AMD CDNA system-coherence rules
(per `ipc_primitives.h:5–17` and the MMOE `ipc_utils.cuh` comments) and
will be needed regardless of what fixes the wave-store-drop hazard, so
they stay in the tree.

## 3. Why we stopped here

The remaining root cause is a hardware/compiler interaction that
`__threadfence_system`, LDS staging, `kUnroll`, naïve store, M1b
fallback, intra-WG syncs, cross-rank barriers, and SYSTEM-scope
acquire-loads all fail to repair. The first 64 uint4 always land
(wave 0's first store) and every subsequent wave's store is lost. The
last credible explanation is that on this particular gfx950 + ROCm
combo, the store pipe for wave 1..N inside a 256-thread WG that hits a
peer-mapped address range > some L2-footprint threshold gets its
`buffer_store_dwordx4 sc0|sc1` retired without actually committing.
Reproducing as an isolated 30-line repro and getting AMD eyes on it is
its own project; doing it inside the dispatch-body codebase as a
debugging detour will burn the rest of the day.

So M2-G β (production T/H bring-up) is **blocked**. Everything we built
in M2-G α is still correct at small H; the limit comes from dispatch,
not from FC1.

## 4. What still works

`test_super_e2e_fc1 8 32 4 128 256 32 64` (T=128, H=256, F=64) still
**PASSes**:

```
[skew=balanced] PASS fc1_act numeric (F=64, bf16 tol abs<0.15 OR rel<10%)
[skew=realistic_cov20] PASS fc1_act numeric (F=64, bf16 tol abs<0.15 OR rel<10%)
[skew=hot_cov50] PASS fc1_act numeric (F=64, bf16 tol abs<0.15 OR rel<10%)
[test_super_e2e] PASS (dispatch + FC1 bit-exact, all skews × {1,3} iters)
```

So the M2-G α super-kernel is correctness-validated at small shape, and
the new `grid_sync<3>` / `fence_system` / SYSTEM-acquire dispatch-body
hardening did not regress the small case.

## 5. Next directions (ordered by ROI)

1. **M2-G β-fix** — isolate the wave-1+ store-drop as a 30-line repro
   (single kernel, 1 GPU, no IPC, just two peer-mapped buffers and a
   single per-row `cooperative_b128_copy`) and bisect against
   `__hip_atomic_store(..., __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM)`
   per-store (the heavyweight "sc0|sc1 on every store" pattern that
   `ipc_primitives.h:12–15` says is the only thing that reliably drains
   across XGMI). If that survives, productize as a template parameter
   `kSystemScopeStore` on `cooperative_b128_copy` and use it only on the
   sender-pack / receiver-pull peer-touching paths.
2. **M2-FC2 / M2-COMB at small H** — keep the super-kernel build-out
   moving by adding FC2 push body + tail combine body + B4/B5 cross-rank
   barriers + workspace-clean at H=256. The full forward path will then
   be end-to-end at small shape; production-shape bring-up gates on
   M2-G β-fix.
3. **M4 wave-specialization** — long-shot but might naturally avoid the
   hazard. If the receiver is rewritten as 4 specialized waves where
   each wave OWNS a 224-uint4 slice (chunk = 224 = 1/4 of H=7168 in
   uint4), then wave 0 stores `dst[0..223]`, wave 1 stores `dst[224..447]`,
   etc. — no wave skips iters. If the hazard is specifically about
   "wave-N iter-K when wave-N is uniformly active across all lanes but
   wave-M (M<N) had an earlier predicated iter", wave-specialization
   sidesteps it. (Speculative.)

## 6. Concrete files changed this session

```
modified:   CMakeLists.txt
modified:   csrc/dispatch.hip
modified:   csrc/include/rocmoe/dispatch_body.h
modified:   tests/test_super_kernel_e2e.hip
unchanged:  csrc/include/rocmoe/ipc_primitives.h     (reverted kUnroll experiment + naïve-copy probe)
unchanged:  csrc/include/rocmoe/types.h              (still kWavesPerWG=4 from M2-G α)
unchanged:  csrc/super_kernel.hip                    (M2-G α body untouched)
unchanged:  csrc/include/rocmoe/super_kernel.h       (M2-G α args untouched)
unchanged:  csrc/include/rocmoe/barrier.h            (s_sleep stays)
```

## 7. Decision pending (for next session)

User to pick which of §5's three directions to take next.
