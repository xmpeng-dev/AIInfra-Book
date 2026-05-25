# Cycle 6: re-profile at new optimum (256x256x64 + ratio=0.250)

**When**: 2026-05-14 00:48 (UTC+8)
**Where**: `mi355-gpu-26` / `xiaoming-dev` container
**Script**: `benchmarks/cycle6_profile_new_optimum.sh`
**Raw log**: `benchmarks/results/cycle6_profile_new_optimum_20260513_1647.txt`

## Why this profile run

Cycles 2 + 3 + 5 settled on **tile=256x256x64, comm_ratio=0.250** as the
LDS-frontier optimum. Cycle 1's profile was at the OLD baseline
(128x128x128 + ratio=0.180). We need to know what the *new* phase
breakdown looks like to decide where Cycle 7+ should focus.

## Numbers

### T_src=2048 — wall 14.78 ms, 781 effective_TFLOPS

```
compute WGs [64..192) kernel_total=14782 us
  dispatch_src_ready_wait :  4620 us  (31.3 %)
  fc1_tiles               :  4258 us  (28.8 %)
  swiglu_precompute       :   314 us  ( 2.1 %)
  fc2_tiles               :  2828 us  (19.1 %)
  copy_to_combine         :  2180 us  (14.7 %)
  barriers (1+swiglu+2+3) :   576 us  ( 3.9 %)
comm WGs    [0..64)  total=2421 us  (sort 314 + scatter 2415)
tail WGs    [192..256) gather_combine_phase = 23148 us
```

### T_src=8192 — wall 56.85 ms, 812 effective_TFLOPS

```
compute WGs [64..192) kernel_total=56853 us
  dispatch_src_ready_wait : 17520 us  (30.8 %)
  fc1_tiles               : 16805 us  (29.6 %)
  swiglu_precompute       :  1194 us  ( 2.1 %)
  fc2_tiles               : 10882 us  (19.1 %)
  copy_to_combine         :  8303 us  (14.6 %)
  barriers (1+swiglu+2+3) :  2141 us  ( 3.8 %)
comm WGs    [0..64)  total=8891 us  (sort 1087 + scatter 8873)
tail WGs    [192..256) gather_combine_phase = 89430 us
```

### GEMM-internal breakdown (compute WG, summed across all FC1+FC2 calls)

|  | T_src=2048 | T_src=8192 | C1 (old) 8192 |
|---|---|---|---|
| gemm_total                | 6051 us | 23884 us | 32390 us |
| hbm_issue                 | 35.2 % | 35.6 % | 46.0 % |
| wait_vm                   |  2.0 % |  1.4 % |  2.4 % |
| sync_per_ktile            |  1.1 % |  1.1 % |  2.7 % |
| mfma_inner                | 41.0 % | 41.4 % | 32.0 % |

The GEMM moved from **HBM-bound** (46% issue, 32% MFMA) to closer to
**compute-bound** (36% issue, 41% MFMA). Cycle 2's tile growth bought us
~10pp of MFMA fraction. Wait_vm dropped from 2.4 % to 1.4 % -- prefetch is
basically perfect now.

## Phase ratios are remarkably scale-invariant

Comparing T_src=2048 and T_src=8192 percentages:

| phase | 2048 % | 8192 % | scaling |
|---|---|---|---|
| dispatch_wait      | 31.3 | 30.8 | linear |
| fc1                | 28.8 | 29.6 | linear |
| swiglu_pc          |  2.1 |  2.1 | linear |
| fc2                | 19.1 | 19.1 | linear |
| copy_to_combine    | 14.7 | 14.6 | linear |
| barriers           |  3.9 |  3.8 | linear |

So whatever we improve at one scale, will scale to both.

## The biggest single phase is still dispatch_wait

**dispatch_src_ready_wait = 30% of wall time at both scales.**

This was 26.6 % at cycle 1 (ratio=0.180); going to ratio=0.250 + larger tile
SHRUNK the wall time (67 -> 57 ms) but the *absolute* dispatch_wait basically
didn't change (17.9 -> 17.5 ms). So adding more comm_wgs (46 -> 64) doesn't
help wall scatter time -- it's bottlenecked by **source-side scatter
completion**, not receive-side capacity.

The comm WGs themselves only spend 8.9 ms total per iter (out of 56.85 ms
wall). They're done WAY before compute finishes dispatch_wait. So the
17.5 ms compute wait must be: kernel start -> last peer's `dispatch_src_ready`
signal arrives. That means the SLOWEST peer's scatter takes ~17.5 ms wall on
its own GPU.

Why is a peer's scatter taking 17.5 ms when comm WG total per iter is 8.9 ms?
Possible reasons:

1. **Tail latency between peers.** All 8 ranks launch via IPC barrier, but
   xGMI/HBM contention causes some to lag.
2. **Source-side HBM bandwidth saturation.** Each rank's comm WGs read
   T_src * top_k * H = 8192 * 8 * 7168 * 2B = ~937 MB from HBM and write
   the same volume via xGMI. HBM read at 1.5 TB/s -> ~0.6 ms. xGMI write
   at ~7 * 50 GB/s = 350 GB/s -> ~2.7 ms. So per-rank scatter wall should be
   ~3 ms, not 17.5 ms.
3. The 8.9 ms "comm WG total" in profile must be **per-WG** time, not wall
   time across the 64 comm WGs. With 64 comm WGs cooperating, real wall
   time of scatter = 8.9 ms / (64 / num_ops_per_scatter). Hmm, hard to tell.

## What about copy_to_combine (14.6 %)?

This is the FC2 -> combine_buf -> peer A2A path. 8.3 ms at T_src=8192.

It runs on COMPUTE WGs (not comm), so it's on the critical path of compute
phase. Cycle 7+ candidate: overlap with FC2 tail (per-tile combine).

## What about the gather_combine on tail WGs?

Tail WGs report 89 ms total time for gather_combine. That's >1.5x the kernel
wall. Either:
- the kernel actually takes 89 ms but bench reports 57 ms (timing bug)
- or tail WGs idle a lot and "kernel_total" includes idle time before
  they start

Looking at bench timing source: it uses hipEventElapsedTime between two
events on the stream surrounding the kernel launch. So 56.85 ms is the
true wall time. Therefore tail WGs MUST have been idle ~32 ms before their
combine-receive work arrives. That makes sense: gather_combine_phase
waits for peers' FC2 results to arrive (via xGMI), and peers' FC2 is the
LAST phase of their kernel (~57 - 8 = 49 ms after kernel start).

So tail WGs aren't on critical path -- they soak up combine-arrival latency
that already overlaps with my own compute.

## Plan: Cycle 7 = chunked FC1 (kill dispatch_wait)

The largest single addressable phase is **dispatch_wait = 17.5 ms**. The
ideal (zero dispatch_wait) would save 30 % off the wall.

The current kernel:
1. Polls all 8 `dispatch_src_ready[src]` flags up front (single thread, WG 0).
2. After ALL 8 ready, builds the (e, src) pair_tile_offset table on WG 0.
3. All compute WGs round-robin over the global tile table.
4. Single FC1->SwiGLU barrier, then FC2.

The proposed chunked design:
1. Poll src=0 ready. As soon as it's ready, build pair_tile_offset for src=0.
2. Compute WGs start FC1 on src=0 (round-robin its tiles).
3. WG 0 continues polling src=1, src=2, ... and extending pair_tile_offset.
4. The compute WGs check the current pair count (atomically) and grab more
   work as it becomes available.
5. Eventually all 8 srcs are processed, then barrier, then continue.

**Expected save**: src=0 ready arrives at T0; src=7 ready at T0+17.5 ms.
With chunked: FC1 on src=0 starts at T0 and takes ~16.8/8 = 2.1 ms.
By the time the last src=7 arrives at T0+17.5 ms, srcs 0..6 are nearly done.
Wall time: ~T0 + 17.5 + 2.1 = T0 + 19.6 ms for combined dispatch+FC1.
Vs current: T0 + 17.5 + 16.8 = T0 + 34.3 ms.
**Save**: ~14.7 ms at T_src=8192 (26 % of wall).

If we achieve this, T_src=8192 latency: 56.85 - 14.7 = **42.15 ms** (1.36×
worse than current 56). vs PyTorch 18.64 still 2.26x slower, but on path to
M1 (parity).

This is a real kernel refactor. Estimate effort: 1-2 hours of careful work
on `super_kernel_compute_phase()`.

## Action items

- [x] Cycle 6: re-profile at new optimum
- [ ] **Cycle 7** (next): implement chunked-FC1 via `-DMOE_CHUNKED_FC1` flag
   - Step 1: build per-src pair_tile_offset (vs current global)
   - Step 2: compute WGs grab tiles per-src as ready
   - Step 3: drop the upfront "poll all 8 srcs" wait
- [ ] Cycle 8: overlap copy_to_combine with FC2 tail
- [ ] Cycle 9: investigate gemm_internal "remainder = prologue/epilogue +
    sample overhead" (~21 % of GEMM time is unattributed)
