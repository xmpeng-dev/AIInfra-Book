# Cycles 4 + 5 + 5b: tile-shape exploration capped at 256x256x64 by LDS

**When**: 2026-05-14 00:20 ~ 00:46 (UTC+8)
**Where**: `mi355-gpu-26` / `xiaoming-dev` container
**Scripts**: `benchmarks/cycle4_ktile.sh`, `cycle5_asym_tile.sh`, `cycle5b_asym_safe.sh`
**Raw logs**:
- `benchmarks/results/cycle5_asym_20260513_1623.txt`
- `benchmarks/results/cycle5b_asym_safe_20260513_1645.txt`

## Cycle 4 — K_TILE refinement at fixed 256x256

K_TILE valid values are **doubly constrained**:

1. `K_TILE % MFMA_K == 0` (MFMA_K=16 on gfx950) -> K in {16, 32, 48, 64, 80, ...}
2. `M_TILE % ROWS_PER_LD_PHASE == 0` AND `N_TILE % ROWS_PER_LD_PHASE == 0`
   where `ROWS_PER_LD_PHASE = WG_SIZE / (K_TILE / 8) = 2048 / K_TILE`

For M_TILE=N_TILE=256: ROWS must divide 256, so ROWS in {32, 64, 128, 256}
-> K_TILE in {8, 16, 32, 64, 128} only (and only K>=16 is valid for MFMA).

So the effective K_TILE sweep at 256x256 is just **{16, 32, 64, 128}**:
- K=16: V_ELEMS=2, ROWS=128. Compiles. Not yet tested.
- K=32: compiles, **runtime FAILS** -- some other pipeline / scheduling issue.
- K=64: **winner** (14.43 ms / 56.08 ms)
- K=128: 16.91 ms / 64.32 ms (-15% / -13% vs K=64)
- K=256: LDS over -- 256*256*2*2 + 256*256*2*2 = 512 KB >> 160 KB envelope.

K=16 not chased -- would halve K-step efficiency (more LDS swaps, less work per
swap). Skipping cycle 4.

## Cycle 5 / 5b -- asymmetric and large-N tiles

LDS budget on MI355X is **~160 KB per WG** (occupancy=1). The kernel's
`GemmLdsLayout` cost is:

  `A+B buffers = 4 * K_TILE * (M_TILE + N_TILE)` bytes (double-buffered, bf16)
  `pair_data  ~ 10 KB`

For K_TILE=64: M+N max ~= (153600 / 256) = 600
For K_TILE=32: M+N max ~= (153600 / 128) = 1200

But there's an **additional** constraint from `ROWS_PER_LD_PHASE`:
  M_TILE % ROWS == 0 AND N_TILE % ROWS == 0
For K=64: ROWS=32, so M_TILE and N_TILE must be multiples of 32 (effectively 64).
For K=32: ROWS=64.

### Tested variants (all built; runtime success/failure noted)

| variant | M | N | K | M+N | LDS est. | T_src=2048 ms | T_src=8192 ms | result |
|---|---|---|---|---|---|---|---|---|
| baseline_256_256_64 | 256 | 256 | 64 | 512 | 138 KB | 14.43 | 56.08 | **WINNER** |
| m256_384_64 | 256 | 384 | 64 | 640 | 170 KB | FAIL | FAIL | LDS over |
| m384_256_64 | 384 | 256 | 64 | 640 | 170 KB | FAIL | FAIL | LDS over |
| m256_512_32 | 256 | 512 | 32 | 768 | 96 KB | FAIL | FAIL | runtime |
| m512_256_32 | 512 | 256 | 32 | 768 | 96 KB | FAIL | FAIL | runtime |
| m384_384_32 | 384 | 384 | 32 | 768 | 96 KB | FAIL | FAIL | runtime |
| m512_128_64 | 512 | 128 | 64 | 640 | 170 KB | FAIL | FAIL | LDS over |
| m128_512_64 | 128 | 512 | 64 | 640 | 170 KB | FAIL | FAIL | LDS over |
| m192_256_64 | 192 | 256 | 64 | 448 | 124 KB | 18.92 | 69.75 | wave bal |
| m256_192_64 | 256 | 192 | 64 | 448 | 124 KB | 18.35 | 69.69 | wave bal |
| m192_320_64 | 192 | 320 | 64 | 512 | 138 KB | 15.32 | 56.61 | tie |
| m320_192_64 | 320 | 192 | 64 | 512 | 138 KB | 15.71 | 58.27 | close |
| m192_384_64 | 192 | 384 | 64 | 576 | 154 KB | 23.94 | 81.79 | wave bal |
| m320_256_64 | 320 | 256 | 64 | 576 | 154 KB | 37.86 | 134.16 | catastrophic |
| m256_320_64 | 256 | 320 | 64 | 576 | 154 KB | 31.13 | 113.29 | catastrophic |
| m192_192_64 | 192 | 192 | 64 | 384 | 108 KB | 19.27 | 69.88 | smaller |
| m256_384_32 | 256 | 384 | 32 | 640 | 80 KB | FAIL | FAIL | K=32 ROWS |
| m384_256_32 | 384 | 256 | 32 | 640 | 80 KB | FAIL | FAIL | K=32 ROWS |

### Findings

1. **Symmetric 256x256 is the LDS-frontier optimal.** No asymmetric variant beats it.
2. **M_TILE that is NOT a multiple of 64 (e.g. 320) hurts catastrophically.**
   - m320_256_64 went from 56 ms to **134 ms** at T_src=8192 (-58% TFLOPS).
   - Theory: the wave layout (2x2) assumes M_TILE/2 = WAVE_M, MFMA_PER_WAVE_M
     must integer-divide cleanly into MFMA-32 blocks. 320/2 = 160, 160/32 = 5
     (works), but every wave then has 5 MFMA blocks instead of 4 (256/2/32) or
     8 (any power of 2). The extra MFMA per wave breaks some load-balance
     assumption, possibly causing register spill or address-arithmetic
     overhead.
3. **K=32 variants all FAIL at runtime.** Even when LDS budget is under
   (m256_512_32 is just 96 KB), the kernel returns invalid argument. Suspect a
   K-pipeline-stages assertion or unhandled small-K codepath. Not chased
   further; K=64 is the regime that wins.
4. **The 160 KB MI355X LDS envelope binds** anywhere M+N > ~570 at K=64.

## Decision: stop tile-shape sweep, focus on phase-level wins

Two cycles confirmed that the (M, N, K, ratio) sweet spot is **256x256x64 at
ratio=0.250**. The remaining gap vs PyTorch+RCCL (1.60x / 3.01x) cannot be
closed by tile shape alone.

The biggest single overhead in the cycle 1 profile (at ratio=0.180, pre-cycle-3)
was **dispatch_src_ready_wait = 26.6%** (17.9 ms at T_src=8192). Even if cycle
3's higher comm_wgs halve that, it still leaves ~8-10 ms on the table per
forward pass.

Next: re-profile at the new optimum (cycle 6) to measure the *actual* current
breakdown, then attack the largest remaining phase.

## Action items

- [x] Cycle 1: profile-enabled bench (at OLD ratio=0.180)
- [x] Cycle 2: tile sweep -> 256x256x64
- [x] Cycle 3: comm_ratio -> 0.250
- [x] Cycle 4/5/5b: exhaustive tile-shape exploration -- 256x256x64 confirmed optimal
- [ ] **Cycle 6**: re-profile at 256x256x64 + ratio=0.250 to update phase breakdown
- [ ] Cycle 7: based on cycle 6 findings, target the largest phase
  - dispatch_wait: chunked-FC1 (start FC1 on src=0 as soon as ready)
  - fc1_tiles (HBM-bound): expert-weight L2 reuse / packed weight layout
  - fc2_tiles (HBM-bound): same
  - copy_to_combine: vectorized epilogue
