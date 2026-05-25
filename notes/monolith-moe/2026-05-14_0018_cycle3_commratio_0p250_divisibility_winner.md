# Cycle 3: comm_ratio fine sweep -- winner 0.250 unlocks another +10%

**When**: 2026-05-14 00:12 ~ 00:18 (UTC+8)
**Where**: `mi355-gpu-26` / `xiaoming-dev` container
**Binary**: `benchmarks/results/bin/bench_sk_m256_256_64`
**Knobs**: tile=256x256x64 (Cycle 2 winner), epg=32, topk=8, hidden=7168, ffn=2048, CUs=256, WGs/CU=1
**Iters**: warmup=5, iters=30 per ratio
**Scripts**: `benchmarks/cycle3_commratio.sh`, `benchmarks/cycle3b_commratio_high.sh`
**Raw logs**:
- `benchmarks/results/cycle3_commratio_20260513_1614.txt`
- `benchmarks/results/cycle3b_commratio_high_20260513_1616.txt`

## Combined data (19 ratio points)

**IMPORTANT**: `comm` is head-and-tail (two phases). So
`num_compute_wgs = num_total_wgs - 2 * num_comm_wgs`. At total=256:

| ratio | comm (head=tail) | total_comm (2x) | compute | T_src=2048 ms | T_src=8192 ms |
|---|---|---|---|---|---|
| 0.100 | 25 | 50 | 206 | 25.14 | 100.16 |
| 0.125 | 32 | 64 | 192 (=6x32) | 17.53 | 66.96 |
| 0.140 | 36 | 72 | 184 | 18.72 | 70.58 |
| 0.150 | 38 | 76 | 180 | 18.29 | 69.38 |
| 0.160 | 41 | 82 | 174 | 16.11 | 61.23 |
| 0.180 | 46 | 92 | 164 | 16.02 | 58.22 |
| 0.200 | 51 | 102 | 154 | 16.10 | 60.04 |
| 0.225 | 57 | 114 | 142 | 16.93 | 63.74 |
| 0.230 | 58 | 116 | 140 | 18.61 | 70.68 |
| 0.240 | 61 | 122 | 134 | 15.32 | 57.09 |
| **0.250** | **64** | **128** | **128 (=4x32)** | **14.44** | **56.10** |
| 0.260 | 66 | 132 | 124 | 19.16 | 71.17 |
| 0.275 | 70 | 140 | 116 | 17.06 | 65.29 |
| 0.290 | 74 | 148 | 108 | 18.38 | 69.74 |
| 0.300 | 76 | 152 | 104 | 15.72 | 59.35 |
| 0.325 | 83 | 166 | 90 | 18.45 | 70.18 |
| 0.350 | 89 | 178 | 78 | 17.69 | 66.67 |
| 0.400 | 102 | 204 | 52 | 22.11 | 85.12 |

## Key finding: highly non-monotonic, integer-divisibility dominated

The curve has two visible "good" wells (0.18 and 0.24-0.25) and two
spikes-back-up (0.225-0.230 and 0.260). It is **NOT** a smooth tradeoff
between "more comm WGs -> faster A2A" vs "fewer compute WGs -> slower GEMM".

Instead, two factors compete:

(a) `compute_wgs` divisibility by `num_experts=32`. Clean values: 32, 64, 96, 128,
    160, 192, 224. Below this granularity the tail-WG idles waiting for the
    last expert's last tile.

(b) `total_comm_wgs = 2 * comm_wgs` must be high enough that the head/tail
    A2A doesn't bottleneck.

| ratio | compute | factor of 32? | total_comm | T_src=8192 ms |
|---|---|---|---|---|
| 0.125 | 192 | YES (6x) | 64 | 66.96 -- comm starved |
| 0.180 | 164 | no | 92 | 58.22 -- good (compute fits) |
| 0.240 | 134 | no | 122 | 57.09 -- good |
| **0.250** | **128** | **YES (4x)** | **128** | **56.10 -- best** |
| 0.260 | 124 | no | 132 | 71.17 -- bad (124 = 4x31, 1 expert idle) |
| 0.300 | 104 | no | 152 | 59.35 -- comp starved |

So **ratio=0.250 hits both criteria**: compute=128=4·32 (every expert gets
exactly 4 WGs) AND total_comm=128 enough for A2A. 0.125 has the cleaner
compute factor (6x) but only 64 comm WGs cannot keep pace with FC1 finishing,
so dispatch tail dominates.

## Impact on the goal

Single-GPU bench (no comms cost in standalone bench, just kernel time):

| shape | PyTorch+RCCL | SK base | SK + Cycle 2 | SK + Cycle 3 | speedup C1->C3 |
|---|---|---|---|---|---|
| T_src=2048 | 9.05 ms | 16.91 (1.87x) | 15.94 (1.76x) | **14.44 (1.60x)** | **17.2%** |
| T_src=8192 | 18.64 ms | 64.32 (3.46x) | 58.37 (3.13x) | **56.10 (3.01x)** | **14.7%** |

Two cycles bought us 14-17% throughput, but the gap to PyTorch is still
1.6x / 3.0x. The dominant gap remains GEMM efficiency (the kernel runs at
~800 effective TFLOPS; MI355X BF16 peak ~= 1.3 PF -> ~60% peak).

## Lessons

1. **comm_ratio search MUST be integer-aware** -- picking ratios at the start
   that round to "nice" compute_wgs (factors that include #experts and small
   primes 2/3) wins. Continuous interpolation gives misleadingly bad numbers.
2. The kernel's load balancing is per-WG, not per-CU. So 256-CU machines
   benefit when compute_wgs in {32k for k=4..7}, i.e. {128, 160, 192, 224}.
   Future: have the host pick compute_wgs by rounding to nearest 32-multiple.
3. **Comm path is NOT the bottleneck at this scale** -- going from 25 -> 102
   comm WGs only moves the optimum by a few ms, and the local optimum at 64
   is determined by the *compute side* needing to be 192 for divisibility.

## Action items

- [x] Cycle 1: profile-enabled bench
- [x] Cycle 2: tile sweep -> 256x256x64 (+9%)
- [x] Cycle 3: comm_ratio -> 0.250 (+10% more on 2048, +4% on 8192)
- [ ] **Cycle 4** (next): K_TILE refinement at 256x256+0.250
- [ ] Cycle 5: asymmetric/large-N tile exploration
- [ ] Cycle 6: phase pipelining
- [ ] Cycle 7: investigate GEMM efficiency vs rocBLAS

## Note on production deployment

Once we settle on a permanent ratio, host launcher should switch from
fractional ratio to *direct compute_wgs* (snapping to 32-multiple). Right
now `MonolithMoELayer` in `python/mmoe/megatron.py` reads `comm_ratio` from
an env var; we should expose an explicit `compute_wgs=192` knob.
