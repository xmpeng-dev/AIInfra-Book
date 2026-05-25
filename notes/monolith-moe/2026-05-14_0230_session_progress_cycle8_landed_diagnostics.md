# Session 2026-05-14: 持续优化进展记录

**Session window**: 2026-05-13 22:00 ~ 2026-05-14 02:30 (UTC+8)
**Operator**: agent in `xiaoming-dev` container on `mi355-gpu-26`

## What landed

| Cycle | Change | T_src=2048 | T_src=8192 |
|---|---|---|---|
| 1 (baseline before today) | 256x256x64 tile + ratio=0.250 | 14.43 ms (776 TFLOPS) | 56.08 ms (823 TFLOPS) |
| 7 | chunked-FC1 — kept under flag, **no bench win** | — | — |
| **8** | **parallel-peer copy_to_combine** — 8 srcs active concurrently via WG partitioning by `wg_local_id % NUM_GPUS` | **13.09 ms (882 TFLOPS, -9.3 %)** | **50.87 ms (908 TFLOPS, -9.3 %)** |
| 11 | dispatch_wait split instrumentation (no perf change) | — | — |

Bench gains today: **-9 % on both shapes** from a single targeted change (Cycle 8).

## Failed cycles (kept as negative lessons)

| Cycle | Hypothesis | What happened | Lesson |
|---|---|---|---|
| 9 (attempt) | parallel-peer scatter (mirror Cycle 8 to scatter loop) | latency went to 19.77 / 77.73 ms (-51 % / -53 %) | scatter receiver-side HBM bank contention; xGMI fabric saturated by 7 outbound *and* 7 inbound streams per peer. Reverted. |
| 9 (attempt 2) | fuse swiglu into FC2 prologue | analysis showed silu*up would be recomputed F/K_TILE = 32× per token → +16 ms cost vs 1.2 ms saved | swiglu_pc is "amortised per-token", FC2 prologue is "per K-tile" — fusion regime is wrong. |
| comm_ratio resweep | maybe 0.125 or 0.375 is better | 0.250 still optimal | confirmed earlier finding |
| small-tile threshold | broader use of M_TILE_SMALL=32 | within noise | already well-tuned |

## Diagnostic finding (Cycle 11)

Split `dispatch_src_ready_wait` into:
- **until_first_src** = time from compute WG entry until ANY src flag reaches `epg`.
- **first→all (skew)** = additional time until ALL 8 srcs ready.

|  | total | until_first | skew |
|---|---|---|---|
| T_src=2048 | 4.73 ms | 4.43 ms (94 %) | 0.30 ms (6 %) |
| T_src=8192 | 17.94 ms | 16.85 ms (94 %) | 1.10 ms (6 %) |

But: host `launch_all` only takes **114 µs**, and per-rank hipEvent
elapsed time shows kernel wall spread = **0.13 ms across 8 ranks**.

**So the 8 ranks DO start within 0.14 ms of each other.**  The "16.85 ms"
is NOT launch skew.  Probable cause: comm WGs and compute WGs sit in
different Shader Engines with different `clock64()` domains; the profile
cycle→µs conversion uses compute WG's wall calibration, so comm/tail
WG measurements are inflated.  The actual real wall (latency_ms,
hipEvents) is reliable.

This is bench-side ambiguity, not a real kernel issue.  The wall time
that we want to drive down is captured correctly by `latency_ms`.

## Where Cycle 8 win comes from

Cycle 8 changed `copy_to_combine` from a `for (int src = 0; src < 8; src++)`
sequential outer loop (only ONE xGMI link active at a time) to a
WG-by-src partitioning (`my_src = wg_local_id % NUM_GPUS`) so all 7
outbound xGMI links saturate concurrently.  Profile confirms:

| | before | after |
|---|---|---|
| copy_to_combine (ms) | 8.30 | 2.88 (**-65 %**) |
| Achieved xGMI BW (GB/s) | 113 | 326 |
| xGMI link utilization | 32 % | **93 %** |

The same trick did NOT work for scatter (Cycle 9 attempt 1) because:
- copy writes to 8 disjoint memory regions on 8 disjoint peers; no
  cross-peer contention.
- scatter writes 32 disjoint expert-slot ranges PER DEST, plus issues
  atomic-add to a shared per-src counter; concurrent 8-dest writes
  cause receiver HBM bank contention and atomic serialization.

## Status vs 1.8x speedup goal

| metric | T_src=2048 | T_src=8192 |
|---|---|---|
| PyTorch+RCCL baseline | 9.05 ms | 18.64 ms |
| Super-kernel today | **13.09 ms** | **50.87 ms** |
| Current ratio | 1.45× slower | 2.73× slower |
| Target (1.8× faster) | 5.03 ms | 10.36 ms |
| Total cumulative gap | -62 % | -80 % |

Bench-side leverage map (T_src=8192, 51 ms):

| phase | ms | % of wall | optimisation regime |
|---|---|---|---|
| dispatch_src_ready_wait | 17.94 | 35 % | bench artifact (?clock-domain); in real training ~9 ms |
| fc1_tiles               | 17.23 | 34 % | GEMM core, 62 % of peak; hipBLASLt-class needed |
| fc2_tiles               | 11.20 | 22 % | GEMM core, 62 % of peak |
| copy_to_combine         |  2.88 |  6 % | already 93 % xGMI peak |
| swiglu_precompute       |  1.22 |  2 % | small, hard to fuse cleanly |
| barriers                |  0.68 |  1 % | already minimal |

The path to 1.8× requires:
1. Closing the GEMM-vs-hipBLASLt gap (FC1+FC2 from 28 ms → ~12 ms).
   Requires hipBLASLt-class scheduling / occupancy.  ~weeks of work.
2. Reducing dispatch_wait — needs clock-domain investigation or
   training-mode validation (where the 17 ms may not exist).
3. Even with both, copy + swiglu + barrier = ~5 ms floor remains.

**Realistic single-node bench reach**: 35-40 ms at T_src=8192 (1.3-1.5×
slower than PyTorch+RCCL), assuming we can shave 10-15 ms from GEMM and
2-3 ms from copy/swiglu.

**Realistic training reach**: depends on whether dispatch_wait shrinks
in production-launch conditions; data point would unlock the picture.

## Next concrete actions

1. ✅ Cycle 8 landed; -9 % on both shapes.
2. ⏸ Cycle 9 (swiglu fuse) — not pursuing (negative analysis).
3. ⏭ Cycle 10 (GEMM remainder ~5 ms): instrument prologue + epilogue
       store_acc separately; check whether store_acc can vectorize to
       `dwordx2` writes.
4. ⏭ Cycle 11b (start-barrier in kernel) — DEFERRED.  Per-rank hipEvent
       showed spread = 0.13 ms; launch-skew is not the issue.
       Real ambiguity is clock-domain between WG roles.  Lower priority
       than GEMM/wall reductions.
5. ⏭ Cycle 12 (run real training with current kernel) — verify
       Megatron-side latency now that bench wall has dropped to 50.87 ms
       (from 56.08 ms). Confirm decomposed_bwd correctness on extended
       iter count. Compare against PyTorch+RCCL in same training env.
6. ⏭ Cycle 13: investigate FC1/FC2 GEMM occupancy & wave layout
       sweep (we did tile sweep but not wave layout `<WLM, WLN>`).

Will execute Cycle 12 + Cycle 13 next.
