# Cycle 7: chunked FC1 -- *negative result* in single-node bench

**When**: 2026-05-14 01:00 (UTC+8)
**Where**: `mi355-gpu-26` / `xiaoming-dev` container
**Code**: `csrc/fused_moe_super_kernel.hip` lines 1823-2138 (`MOE_CHUNKED_FC1`)
**Binaries**: `bench_sk_chunked_b{2,4,8,16}`

## Hypothesis (from Cycle 6 profile)

Cycle 6 showed dispatch_src_ready_wait = 17.5 ms (30.8 % of wall) at
T_src=8192, even at the new optimum tile/ratio.  If even part of that wait
is the "slow-src tail" of an asymmetric scatter, then a chunked FC1 design
that starts FC1 work on early-ready experts (while still waiting for
straggler srcs) should hide a chunk of those 17.5 ms.

Expected save: 5-15 ms at T_src=8192 if peers are skewed by ~few ms each.

## Implementation

Added `#if defined(MOE_CHUNKED_FC1)` block that processes experts in
batches of `MOE_CHUNK_FC1_BATCH` (default 8).  For each batch:
1. Poll `dispatch_expert_ready[src * epg + e]` for all srcs of all
   experts in the batch.
2. Build per-batch pair info (B * 8 pairs).
3. Round-robin all 128 compute WGs over the B*8*8=B*64 tiles.
4. Cross-WG barrier so batches march in lockstep (preserves L2 weight
   reuse).

Original path preserved under `#else`; default build is unchanged.

## Sweep results (single-node bench, ratio=0.250, tile=256x256x64)

| Variant      | T_src=2048 ms | T_src=8192 ms |
|---|---|---|
| **Original (no chunk)** | **14.43** | **56.08** |
| Chunked B=1 | 27.50 | -- (catastrophic granularity) |
| Chunked B=2 | 19.48 | 55.95 |
| Chunked B=4 | 14.98 | 56.16 |
| Chunked B=8 | 14.87 | 56.09 |
| Chunked B=16 | 14.83 | 56.17 |

**Net**: no improvement at any batch size.  At T_src=2048 the chunked
path is slightly *worse* by 0.4-1.0 ms (per-batch overhead).

## Why chunked doesn't help in single-node bench

The bench launches all 8 ranks on the SAME node via shared peer access.
They all start the persistent kernel after a single `moe_ipc_barrier_kernel`,
which gates kernel start to within microseconds.  Then all 8 ranks'
comm WGs scatter to each other in parallel; given matched HBM/xGMI
bandwidth per rank, all 8 srcs become ready to me at nearly the same
time (within ~1 ms).  So there's no "slow-src tail" for chunked FC1 to
hide.

The 17.5 ms dispatch_wait in profile is **not skew** -- it's the wall-clock
time of the parallel scatter itself, which all ranks do simultaneously.
The compute WG's wait equals the wall-clock scatter duration, not the
sum of individual peer latencies.

## Where chunked WOULD help (theory)

In a real multi-node training environment where:
- Different ranks land on different host machines.
- Some ranks share xGMI fabric, others go over Infinity Fabric / NIC.
- gate-softmax / topk compute time varies between ranks (gate weights
  differ, accumulator paths differ).
- Other concurrent training kernels (attention, embedding) may run
  slightly different durations.

In such cases, the per-rank kernel launch time can be skewed by a few
ms.  The chunked FC1 path would let us start work on early ranks'
data while waiting for the slow rank.  We can't easily reproduce this
in the single-node bench.

We keep the chunked code in-tree under the `MOE_CHUNKED_FC1` compile
flag so it can be enabled if/when we observe per-rank skew in actual
training.

## Action items

- [x] Cycle 7: implement chunked FC1; confirm no-op in bench; keep
       gated under compile flag.
- [ ] **Cycle 8** (next): overlap `copy_to_combine` (14.6 % of wall,
       8.3 ms at T_src=8192) with FC2 tail.  Per-pair copy as each
       (src, e) FC2 finishes.
- [ ] Cycle 9: try `swiglu_precompute` overlap with the *next* FC1
       batch (currently a hard barrier sits between them).
- [ ] Cycle 10: investigate GEMM "unattributed remainder" (~20 % of
       GEMM time).  Bench bucket-12 only counts hot loop; what's the
       prologue/epilogue cost?  Tools like rocprof / omnitrace might
       help.

## Where we stand vs the 1.8x goal

| shape | PyTorch+RCCL | SK current | gap | save needed for 1.8x |
|---|---|---|---|---|
| T_src=2048 | 9.05 ms | 14.43 ms (1.59x) | -37 % | get to 5.0 ms |
| T_src=8192 | 18.64 ms | 56.08 ms (3.01x) | -67 % | get to 10.4 ms |

To close the 8192 gap by 5.4x, **the GEMM phase needs to roughly
double in throughput** (from 812 to ~1500-1700 effective_TFLOPS) AND
the dispatch_wait must become a non-issue.  Single-node bench will
expose only the GEMM side of this; multi-node training may help on
dispatch_wait.

Next focus: find the missing ~20 % "remainder" in GEMM internal
profile (prologue/epilogue) and the copy_to_combine overlap.
