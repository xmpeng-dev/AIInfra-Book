# Cycle 2: GEMM tile sweep — winner 256×256×64 unlocks +10% TFLOPS

**When**: 2026-05-13 23:55 ~ 2026-05-14 00:05 (UTC+8)
**Where**: `mi355-gpu-26` / `xiaoming-dev` podman container / single-GPU bench (no RCCL)
**Build**: hipcc gfx950, ROCm 6.4, `-DMOE_K_TILE=K -DMOE_M_TILE=M -DMOE_N_TILE=N`
**Knobs**: `epg=32`, `topk=8`, `hidden=7168`, `ffn=2048`, `num-cus=256`, `wgs-per-cu=1`, `comm-ratio=0.180`
**Iters**: warmup=5, iters=30
**Script**: `benchmarks/cycle2_tile_sweep.sh`
**Raw log**: `benchmarks/results/cycle2_tile_sweep_20260513_1559.txt`

## What I changed
The super-kernel today compiles with `MOE_M_TILE=MOE_N_TILE=MOE_K_TILE=128` by default
(`csrc/fused_moe_super_kernel.hip`). Cycle 1 showed FC1/FC2 GEMMs eat ~56% of the kernel
and are 3× slower than rocBLAS. So I built 8 variants and ran them on the two real
training shapes (T_src=2048 = 4-layer val, T_src=8192 = production 8192-context).

## Results (training-relevant shapes, single-GPU bench, no comms)

| tile (M×N×K) | T_src=2048 ms | TFLOPS | T_src=8192 ms | TFLOPS | vs base @8192 |
|---|---|---|---|---|---|
| **128×128×128** (base) | 16.91 | 683 | 64.32 | 718 | — |
| 256×128×64  | 18.26 | 632 | 67.78 | 681 | −5.4% |
| 128×256×64  | 18.32 | 630 | 66.99 | 689 | −4.1% |
| **256×256×64**  | **15.94** | **724** | **58.37** | **791** | **+9.3%** ✓ |
| 128×128×64  | 17.97 | 642 | 67.57 | 683 | −5.0% |
| 256×256×32  | FAILED | — | FAILED | — | — |
| 64×128×128  | 19.71 | 586 | 75.15 | 614 | −16.8% |
| 64×256×128  | FAILED | — | FAILED | — | — |

## Why 256×256×64 wins

1. **Large output tile (256×256) amortizes K-loop overhead.** Each tile now does
   `256*256*K=16384*K` MACs per epilogue write. Base 128×128 did `16384*K`-quarter
   per write — 4× more epilogue / scheduling overhead.
2. **K=64 keeps LDS budget tight.** A-tile (256·64·2) + B-tile (64·256·2) = 64KB,
   with double-buffer = 128KB → fits the 160KB per-WG LDS. K=128 at 256×256 would
   need 256KB → won't fit.
3. **MFMA 16×16×16 saturates better at large M_e.** At T_src=8192, expected
   `tokens/expert ≈ 8192·8/256 ≈ 256` per source rank → after combine each expert
   sees ~2k tokens. 256×256 tile = single tile covers a whole expert's chunk;
   no row-tail under-utilization.

## Why others lose

- **Asymmetric M256×N128 or M128×N256**: epilogue still aligned to 128, no
  scheduling win over symmetric base. K=64 alone (no tile growth) just halves the
  pipeline stages → worse.
- **256×256×32**: K=32 < MFMA_K * 2 stages → pipeline starved, compile or runtime
  failure (need to investigate, suspect LDS swizzle constraint).
- **64×*×128**: 64-row tile means single waves per tile and FC2 can't hide
  latency on 256 CUs.

## Impact on the goal (1.8× vs PyTorch+RCCL)

| shape | PyTorch+RCCL ms | SK base (128³) ms | SK 256×256×64 ms | gap |
|---|---|---|---|---|
| T_src=2048 (val)     | 9.05  | 16.91 (1.87×) | **15.94 (1.76×)** | still **−43%** |
| T_src=8192 (prod)    | 18.64 | 64.32 (3.45×) | **58.37 (3.13×)** | still **−68%** |

So one tile-sweep cycle bought us 6–9% on each shape, but we're still far from
parity. Next levers (in priority order):

1. **K_TILE sweep at fixed 256×256** (Cycle 4). With K=64 winning, but base K=128
   trailing only by 10%, maybe K=96 or K=160 with a different MFMA stride hits
   the sweet spot of memory-throughput-vs-arithmetic.
2. **MFMA 32×32×16 instead of 16×16×16** (Cycle 5). 256×256 is a 4×4 grid of
   32×32 MFMA tiles → 16 MFMAs/output-tile vs 256 with 16×16. Far less issue
   density and register pressure.
3. **comm_ratio re-sweep** (Cycle 3). Communication-vs-compute split was tuned
   for the slower base kernel; the faster GEMM may now make comm the bottleneck.

## Action items (next cycles)

- [x] Cycle 1 done: profile-enabled bench → fc1 34% / fc2 22% / dispatch_wait 27% / copy 13%
- [x] Cycle 2 done: tile sweep → 256×256×64 winner, +9.3% @ T_src=8192
- [ ] **Cycle 3** (next): comm_ratio fine sweep on 256×256×64 — 0.10/0.13/0.15/0.18/0.20/0.225
- [ ] Cycle 4: K_TILE refinement {64, 96, 128, 160, 192} at fixed 256×256
- [ ] Cycle 5: MFMA 16×16 → 32×32 at fixed 256×256×64
- [ ] Cycle 6: even larger M_TILE for T_src=8192 (384×256, 512×128?) — LDS-limited
- [ ] Cycle 7: phase pipelining (FC2 per-tile-ready, not whole-FC1-barrier)

## Reference: build commands

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx950 -I csrc \
    -DMOE_M_TILE=256 -DMOE_N_TILE=256 -DMOE_K_TILE=64 \
    -o benchmarks/results/bin/bench_sk_m256_256_64 \
    benchmarks/bench_super_kernel.hip
```

The new "champion" binary lives at:
`benchmarks/results/bin/bench_sk_m256_256_64`

I'll promote it to the default by also patching the header default
(`MOE_M_TILE=256, MOE_N_TILE=256, MOE_K_TILE=64`) at the end of cycle 5 once
MFMA upgrade lands, to avoid churning the build between every cycle.
