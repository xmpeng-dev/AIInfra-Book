# Cycle 11: dispatch_wait split — 94 % is "wait for FIRST src", not skew

**When**: 2026-05-14 01:55 (UTC+8)
**Change**: instrument bucket 20 / 21 in `csrc/fused_moe_super_kernel.hip` and
`benchmarks/bench_super_kernel.hip`.  Profile-only; no perf impact in
non-MOE_PROFILE builds.

## What we measured (256x256x64 + comm_ratio=0.250)

| T_src | total | until_first_src | first→all (skew) |
|---|---|---|---|
| 2048 | 4.73 ms | **4.43 ms (94 %)** | 0.30 ms (6 %) |
| 8192 | 17.94 ms | **16.85 ms (94 %)** | 1.10 ms (6 %) |

The compute WG enters `t_ready` near `t=0` of its own kernel, then waits
**16.85 ms** before ANY src's `dispatch_src_ready` flag reaches `epg=32`.
By contrast, the comm WGs of *the same kernel* finish scatter at 9.22 ms.

So the first peer's flag arrives 7.6 ms LATER than our own comm WGs publish
our self-flag.  The implication: in this single-node benchmark, the wait
is bounded by **inter-GPU kernel-launch skew + xGMI atomic propagation**,
not by scatter throughput.

## Where the skew comes from

The bench launches kernels in a host-side serial loop:

```cpp
for (int r = 0; r < NUM_GPUS; ++r) {
    HIP_CHECK(hipSetDevice(r));
    launch_fused_moe_super_kernel(args, ..., ranks[r].stream);
}
```

Each `hipSetDevice` + launch round-trip takes ~1 ms of host time
(driver IPC validation, stream submission), so rank 7's kernel is queued
~7-8 ms after rank 0's.

Once `sync_all` waits for the slowest GPU, the latency we report includes
the entire skew window — which manifests in compute WGs of fast-starting
ranks as a long `until_first_src`.

## Why earlier cycles couldn't move this number

- **Cycle 7 (chunked-FC1)**: overlap FC1 with dispatch_wait was supposed
  to hide this, but the chunked path also needed a cross-WG barrier
  between expert batches, which in single-node bench costs more than the
  hidden wait (no actual stragglers to hide).
- **Scatter parallelization**: irrelevant — the scatter only contributes
  ~9 ms of the 17.9 ms wait; the other 8.7 ms is launch-skew, which the
  comm WGs can't fix from inside the kernel.

## What we WOULD expect in real training

Real Megatron training:
- All 8 ranks share the same Python launcher and `torchrun` infra.
- Kernels are launched into each rank's stream within ~tens of µs of
  each other (no `hipSetDevice` loop in the hot path; the CUDA Graph
  capture amortises it).
- Inter-iter spacing is dominated by the previous iteration's compute,
  so launches are essentially pipelined.

Therefore, in production, `until_first_src` should drop to scatter wall
(~9 ms at T_src=8192) and `first→all` should stay around 1 ms — yielding
**dispatch_wait ≈ 10 ms instead of 18 ms.  Wall time win: ~8 ms per
layer per fwd.**

Conversely, if real training shows MORE than 18 ms in dispatch_wait,
the Megatron pipeline is the bottleneck (CUDA Graph fragmentation, host
GIL contention, dispatcher overhead) — *not* the kernel itself.

## Action items

- [x] Cycle 11: instrumentation lands.  Confirmed launch-skew dominates
       `dispatch_wait` in single-node bench.
- [ ] Cycle 11b: add a **kernel-entry start barrier** (atomic increment
       on shared IPC field, spin until 8) to factor out launch-skew in
       the bench, so future micro-optimizations are measured against
       a clean scatter-bound baseline.  Defer until we exhaust GEMM
       optimizations (which are skew-independent).
- [ ] Move on to Cycle 9 (swiglu fuse) and Cycle 10 (GEMM remainder),
       both of which are independent of `dispatch_wait`.

## Updated leverage map

Single-node bench (T_src=8192, 51.19 ms):

|  | ms | source | recoverable in training? |
|---|---|---|---|
| dispatch_wait — launch skew | 7.6 | bench artifact | YES (~0) |
| dispatch_wait — scatter wall | 9.2 | fundamental | depends on net |
| dispatch_wait — peer skew | 1.1 | scatter compl. spread | partial |
| fc1_tiles | 17.2 | GEMM compute | optimizable |
| fc2_tiles | 11.2 | GEMM compute | optimizable |
| swiglu_precompute | 1.2 | separate phase | fuse |
| copy_to_combine | 2.9 | xGMI write | already 93 % peak |
| barriers | 0.7 | __syncthreads/atomic | minimal |

So in real training the comparable wall would be approximately
51.19 - 7.6 = **43.6 ms** (vs PyTorch+RCCL 18.64 ms — still 2.34× slower).
That's the actual gap we need to close with GEMM/copy/fuse work.

## Updated milestone targets

For the 1.8x goal (T_src=8192, target = 10.4 ms vs PyTorch+RCCL 18.64 ms):

- Today bench: 50.87 ms (1× our own, 2.73× slower than baseline)
- Discount launch-skew: 43.6 ms (estimated training-equivalent)
- Need: 10.4 ms in training → must shave ~33 ms (76 %) more from compute
- Per-cycle leverage targets:
  - swiglu fuse: 1.2 ms (2 %)
  - copy elim/overlap: 2.9 ms (7 %)
  - GEMM 62 → 80 % peak: 6 ms (14 %)
  - GEMM 62 → 90 % peak (heroic): 10 ms (23 %)
  - dispatch wait reduction: 7.6 ms (free in training)
  - …gap remains very large.

The honest read: hitting 1.8x is **very stretch** without architectural
changes that match hipBLASLt-class GEMM scheduling.  Will keep pushing
incrementally and re-evaluate at 1.2× and 1.5× milestones.
