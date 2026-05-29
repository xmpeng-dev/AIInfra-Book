# Turbo MoE forward + backward breakdown — MI355X EP8

Per-stage latency breakdown of the Primus-turbo MoE pipeline
(dispatch + grouped fc1 + swiglu + grouped fc2 + combine), measured on a
single MI355X node with EP=8 (1 node × 8 GPUs, no TP). The numbers below
correspond to **one transformer layer's MoE submodule, one micro-batch**,
reported separately for forward and backward, with the routing table
that Megatron's `moe_router_force_load_balancing` produces.

All data was collected by `bench_turbo_moe_e2e.py --mode fwd_bwd` inside
the `xiaoming-dev` container on `mi355-gpu-7` (Slurm job `13780`,
re-generated 2026-05-28 with the backward-pass breakdown). The wrapper
that swept all four configs is `run_all_models.sh`. Canonical
configuration: `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON`, no autotune
(see *Autotune experiment* below for why).

## Layout

```
moe_perf/turbo/                       # this directory
├── README.md                         # this file
├── bench_turbo_moe_e2e.py            # standalone benchmark (only depends on primus_turbo + torch)
├── run_all_models.sh                 # 4-model sweep driver (Triton-pinned)
├── bench-script-notes.md             # per-stage table format and CLI knobs
├── *.csv                             # canonical Triton-baseline tables — see Files below
├── build_backend_comparison.py       # render Triton/HIPBLASLT/CK comparison from archive_backends/
├── archive_backends/{triton,hipblaslt,ck}/  # full sweep per backend, no autotune (see Backend comparison)
├── archive_triton_baseline/          # raw Triton CSVs (mirror of the canonical ones, with -triton suffix)
├── archive_autotune_experiment/      # CSVs from PRIMUS_TURBO_AUTO_TUNE=1 run, kept for comparison
└── archive_fwd_only/                 # historical forward-only sweep before bwd timing was added
```

## Files

| File | Routing | Shape `(H, ffn, E, K)` | Tokens swept per rank | Production hot point |
|---|---|---|---|---|
| `deepseek-v2-lite-spread-breakdown.csv` | spread | (2048, 1408, 64, 6) | 4k, 8k, 16k, 32k, 49k | **49,152** (mbs=12 × seq=4096) |
| `deepseek-v3-spread-breakdown.csv` | spread | (7168, 2048, 256, 8) | 1k, 2k, 4k, 8k, 16k | **8,192**  (mbs=2  × seq=4096) |
| `qwen3-30b-a3b-spread-breakdown.csv` | spread | (2048,  768, 128, 8) | 2k, 4k, 8k, 16k, 32k | **32,768** (mbs=8  × seq=4096) |
| `qwen3-235b-a22b-spread-breakdown.csv` | spread | (4096, 1536, 128, 8) | 2k, 4k, 8k, 16k, 32k | **16,384** (mbs=4  × seq=4096) |
| `deepseek-v3-cluster-fullsweep.csv` | cluster (legacy) | (7168, 2048, 256, 8) | 32 → 16k, full sweep | — |
| `deepseek-v3-cluster-breakdown.csv` | cluster (legacy) | same as above | subset | — |
| `deepseek-v3-spread-breakdown-legacy.csv` | spread (legacy, includes BS=256) | same | 256 → 16k | — |

The three `*legacy*.csv` files are kept for reference; the canonical
breakdown for each model is the corresponding `*-spread-breakdown.csv`
(mirror of `archive_backends/triton/<model>-spread-breakdown.csv` —
top-level files are the Triton baseline).

The per-backend full sweep used by the *Per-model breakdown* and
*Backend comparison summary* sections lives in
`archive_backends/{triton,hipblaslt,ck}/<model>-spread-breakdown.csv`
(same column layout, same models, no autotune). Regenerate the
Markdown tables in those sections by running:

```bash
python3 slab/notes/moe_perf/turbo/build_backend_comparison.py
```

## Routing

The "spread" runs are what production training actually sees under
`moe_router_force_load_balancing: true` — each token routes to `topk`
experts that map to **`topk` distinct EP ranks**, so the all-to-all
volume scales as the full `num_tokens × topk × hidden × 2 B`.

The "cluster" runs concentrate all `topk` experts of a token on a single
EP rank. They make the dispatch all-to-all collapse to a single remote
peer per token, which artificially compresses dispatch latency (~6×
faster than real). They were the default early in the sweep and are
preserved here only as the starting point of the analysis. **Do not
quote cluster-routing numbers as Primus performance.**

## Per-model breakdown — spread routing, BF16

Each model has **seven** tables: forward breakdown for each grouped-GEMM backend (Triton / HIPBLASLT / CK), backward breakdown for each backend, and a step-time comparison across the three. All rows are spread routing, no autotune, `--sync-free-stage 2`, BF16 weights + activations. The ★ row in each table marks that model's production token count per rank (`mbs × seq_length`). Bold numbers in the breakdown tables highlight the PROD row.

Column semantics (forward): `Time (us)` is the e2e CUDA-event wall time. `Compute (TFLOPS)` is FC1+FC2 matmul throughput against `Time (us)` (backward counts the 2× FLOPs from `grad_input + grad_weight` GEMMs). `fc1 / swiglu / fc2` are `mean_us / TFLOPS`; per-stage TFLOPS uses each kernel's own wall-time and its theoretical FLOPs (fwd fc1 = `4·N·H·F`, swiglu = `5·N·F`, fc2 = `2·N·H·F`, `N = num_tokens × topk`; bwd = 2× fwd FLOPs). `Global Memory (GB/s)` is the forward-only e2e-averaged DRAM traffic. `all_kernels (us)` is the sum of per-stage CUDA-event medians; the residual vs `Time (us)` is launch-queue overhead. In the backward table, `sort` is a residual (e2e backward minus the sum of all other stages) because PyTorch leaf-tensor hooks fire at unpredictable times.

Raw CSVs live in `archive_backends/{triton,hipblaslt,ck}/`. This whole section is regenerated by `python3 build_backend_comparison.py`.

### DeepSeek-V2-Lite

`experts=64 top_k=6 hidden=2048 intermediate=1408 ep_size=8 local_experts=8`

#### Forward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2699.8 | 157.5 | 277 | 10.5 | 671.8 | 497.6 / 569.6 | 49.4 / 3.5 | 176.8 / 801.6 | 942.6 | 266.1 | 2614.8 |
| 8,192 | 3187.4 | 266.8 | 426 | 10.9 | 895.7 | 735.2 / 771.1 | 90.0 / 3.8 | 331.5 / 855.1 | 666.8 | 386.7 | 3116.8 |
| 16,384 | 5589.4 | 304.3 | 461 | 10.5 | 1673.2 | 1161.2 / 976.5 | 169.9 / 4.1 | 666.0 / 851.3 | 1233.8 | 608.8 | 5523.3 |
| 32,768 | 10906.7 | 311.9 | 460 | 10.1 | 3231.4 | 2269.2 / 999.3 | 344.2 / 4.0 | 1440.7 / 787.0 | 2396.2 | 1145.1 | 10837.0 |
| **49,152 ★** | **16270.3** | **313.6** | **459** | **11.0** | **4794.3** | **3563.5 / 954.6** | **526.6 / 3.9** | **2092.0 / 813.0** | **3567.5** | **1646.6** | **16201.5** |

#### Forward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 3985.9 | 106.7 | 188 | 10.4 | 811.4 | 786.9 / 360.2 | 103.2 / 1.7 | 464.3 / 305.3 | 1203.3 | 429.0 | 3808.6 |
| 8,192 | 3866.3 | 220.0 | 351 | 11.0 | 891.7 | 933.5 / 607.3 | 91.5 / 3.8 | 495.5 / 572.1 | 832.9 | 512.9 | 3769.0 |
| 16,384 | 5870.4 | 289.7 | 439 | 9.8 | 1684.4 | 1171.6 / 967.8 | 167.8 / 4.1 | 693.9 / 817.0 | 1390.1 | 652.7 | 5770.3 |
| 32,768 | 11302.7 | 301.0 | 444 | 10.1 | 3246.6 | 2326.0 / 975.0 | 376.8 / 3.7 | 1263.4 / 897.5 | 2607.2 | 1312.5 | 11142.7 |

#### Forward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2455.6 | 173.2 | 305 | 12.2 | 574.5 | 600.8 / 471.8 | 49.2 / 3.5 | 209.9 / 675.2 | 551.6 | 344.9 | 2343.1 |
| 8,192 | 3440.8 | 247.2 | 395 | 11.2 | 902.0 | 927.5 / 611.3 | 88.4 / 3.9 | 410.3 / 690.8 | 627.1 | 400.5 | 3367.1 |
| 16,384 | 6165.6 | 275.9 | 418 | 10.5 | 1673.9 | 1566.0 / 724.1 | 168.8 / 4.1 | 834.5 / 679.4 | 1221.6 | 620.5 | 6095.9 |
| 32,768 | 12057.7 | 282.1 | 416 | 10.5 | 3241.2 | 3047.0 / 744.3 | 345.7 / 4.0 | 1797.9 / 630.7 | 2402.2 | 1143.1 | 11987.6 |
| **49,152 ★** | **17880.1** | **285.4** | **417** | **11.7** | **4808.2** | **4742.4 / 717.3** | **514.6 / 4.0** | **2528.2 / 672.7** | **3551.1** | **1653.6** | **17809.8** |

#### Backward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2454.1 | 346.5 | 8.8 | 521.7 | 624.7 / 907.6 | 64.5 / 5.4 | 427.6 / 663.0 | 682.9 | 123.9 | 2454.1 |
| 8,192 | 3711.0 | 458.3 | 9.1 | 671.7 | 1195.4 / 948.5 | 125.0 / 5.5 | 783.3 / 723.8 | 702.9 | 223.6 | 3711.0 |
| 16,384 | 7248.3 | 469.3 | 8.9 | 1333.0 | 2435.5 / 931.1 | 259.0 / 5.3 | 1463.6 / 774.7 | 1335.9 | 412.4 | 7248.3 |
| 32,768 | 14231.2 | 478.1 | 8.9 | 2567.8 | 4799.0 / 945.1 | 534.1 / 5.2 | 2957.9 / 766.7 | 2590.4 | 773.1 | 14231.2 |
| **49,152 ★** | **21255.2** | **480.1** | **9.0** | **3838.4** | **7054.2 / 964.4** | **810.7 / 5.1** | **4579.4 / 742.8** | **3839.0** | **1124.5** | **21255.2** |

#### Backward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 3792.9 | 224.2 | 29.6 | 574.7 | 1224.9 / 462.8 | 170.3 / 2.0 | 798.2 / 355.1 | 782.8 | 212.3 | 3792.9 |
| 8,192 | 5110.5 | 332.8 | 12.8 | 871.6 | 1406.3 / 806.3 | 229.2 / 3.0 | 1056.7 / 536.5 | 1200.7 | 333.2 | 5110.5 |
| 16,384 | 8190.4 | 415.3 | 23.6 | 1545.6 | 2451.4 / 925.1 | 345.0 / 4.0 | 1677.0 / 676.1 | 1587.6 | 560.2 | 8190.4 |
| 32,768 | 14919.3 | 456.0 | 38.9 | 2945.2 | 4455.5 / 1018.0 | 670.9 / 4.1 | 2927.8 / 774.5 | 2923.4 | 957.6 | 14919.3 |

#### Backward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2265.4 | 375.4 | 18.4 | 394.5 | 590.0 / 961.0 | 63.2 / 5.5 | 482.1 / 588.0 | 588.8 | 128.4 | 2265.4 |
| 8,192 | 3781.8 | 449.7 | 9.1 | 701.2 | 1134.8 / 999.2 | 124.6 / 5.6 | 886.3 / 639.7 | 705.5 | 220.4 | 3781.8 |
| 16,384 | 7442.6 | 457.0 | 9.0 | 1367.0 | 2343.2 / 967.8 | 256.9 / 5.4 | 1701.9 / 666.2 | 1337.5 | 427.1 | 7442.6 |
| 32,768 | 14718.7 | 462.2 | 8.9 | 2623.1 | 4771.1 / 950.6 | 538.9 / 5.1 | 3407.4 / 665.5 | 2591.1 | 778.2 | 14718.7 |
| **49,152 ★** | **21872.2** | **466.6** | **9.0** | **3862.1** | **6935.3 / 981.0** | **810.1 / 5.1** | **5263.5 / 646.3** | **3847.9** | **1144.3** | **21872.2** |

#### Step time — Triton vs HIPBLASLT vs CK

Forward + Backward wall time per BS. **Δ** columns compare against Triton at the same BS.

| BS | Triton fwd | Triton bwd | Triton step | HBLAS fwd | HBLAS bwd | HBLAS step | HBLAS Δ | CK fwd | CK bwd | CK step | CK Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2,700 | 2,454 | **5,154** | 3,986 | 3,793 | 7,779 | +50.9% | 2,456 | 2,265 | 4,721 | -8.4% |
| 8,192 | 3,187 | 3,711 | **6,898** | 3,866 | 5,110 | 8,977 | +30.1% | 3,441 | 3,782 | 7,223 | +4.7% |
| 16,384 | 5,589 | 7,248 | **12,838** | 5,870 | 8,190 | 14,061 | +9.5% | 6,166 | 7,443 | 13,608 | +6.0% |
| 32,768 | 10,907 | 14,231 | **25,138** | 11,303 | 14,919 | 26,222 | +4.3% | 12,058 | 14,719 | 26,776 | +6.5% |
| 49,152 ★ | 16,270 | 21,255 | **37,525** | — | — | — | — | 17,880 | 21,872 | 39,752 | +5.9% |

### DeepSeek-V3

`experts=256 top_k=8 hidden=7168 intermediate=2048 ep_size=8 local_experts=32`

#### Forward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2131.8 | 338.5 | 1590 | 11.3 | 406.8 | 530.3 / 907.1 | 26.4 / 3.2 | 312.7 / 769.1 | 588.9 | 185.9 | 2062.3 |
| 2,048 | 3553.3 | 406.1 | 1114 | 10.6 | 723.8 | 1020.1 / 943.1 | 40.9 / 4.1 | 543.3 / 885.5 | 834.6 | 309.9 | 3483.3 |
| 4,096 | 6438.5 | 448.3 | 792 | 10.7 | 1337.2 | 1923.9 / 1000.1 | 76.5 / 4.4 | 1119.4 / 859.4 | 1352.7 | 551.0 | 6371.4 |
| **8,192 ★** | **12410.3** | **465.1** | **595** | **10.7** | **2559.4** | **3924.1 / 980.7** | **152.4 / 4.4** | **2120.5 / 907.4** | **2525.8** | **1050.1** | **12343.0** |
| 16,384 | 24149.1 | 478.1 | 495 | 11.5 | 4987.8 | 7782.6 / 989.0 | 314.7 / 4.3 | 3980.6 / 966.8 | 4980.0 | 2024.1 | 24081.3 |

#### Forward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 5421.6 | 133.1 | 625 | 49.1 | 643.8 | 1386.2 / 347.0 | 278.4 / 0.3 | 1379.3 / 174.4 | 804.6 | 638.9 | 5180.3 |
| 2,048 | 5784.6 | 249.5 | 684 | 11.8 | 734.4 | 1875.3 / 513.0 | 46.6 / 3.6 | 1036.3 / 464.2 | 1605.5 | 373.1 | 5683.1 |
| 4,096 | 7609.5 | 379.3 | 670 | 10.8 | 1344.8 | 2428.0 / 792.5 | 77.2 / 4.3 | 1416.0 / 679.4 | 1664.4 | 557.3 | 7498.5 |
| **8,192 ★** | **12988.4** | **444.4** | **568** | **11.0** | **2563.3** | **4188.5 / 918.8** | **156.0 / 4.3** | **1826.0 / 1053.7** | **2909.0** | **1189.1** | **12842.8** |
| 16,384 | 22784.9 | 506.7 | 524 | 11.5 | 4998.5 | 6094.2 / 1262.9 | 316.7 / 4.2 | 3431.2 / 1121.6 | 5578.9 | 2171.9 | 22602.7 |

#### Forward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2300.7 | 313.6 | 1473 | 11.6 | 400.5 | 725.3 / 663.2 | 26.0 / 3.2 | 389.4 / 617.7 | 435.7 | 225.1 | 2213.6 |
| 2,048 | 3907.7 | 369.3 | 1013 | 10.8 | 726.4 | 1277.5 / 753.1 | 41.1 / 4.1 | 680.4 / 707.0 | 802.2 | 300.0 | 3838.4 |
| 4,096 | 7141.4 | 404.2 | 714 | 10.6 | 1338.5 | 2462.4 / 781.4 | 76.2 / 4.4 | 1337.8 / 719.2 | 1322.9 | 525.9 | 7074.2 |
| **8,192 ★** | **13821.5** | **417.6** | **534** | **10.6** | **2560.3** | **4930.3 / 780.5** | **149.9 / 4.5** | **2527.0 / 761.4** | **2536.6** | **1039.7** | **13754.3** |
| 16,384 | 26953.2 | 428.3 | 443 | 11.4 | 4987.2 | 9762.0 / 788.4 | 307.2 / 4.4 | 4808.6 / 800.3 | 4971.3 | 2036.9 | 26884.5 |

#### Backward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 3568.5 | 404.4 | 9.1 | 534.7 | 1651.7 / 582.5 | 51.2 / 3.3 | 793.9 / 605.9 | 386.4 | 141.4 | 3568.5 |
| 2,048 | 5531.7 | 521.8 | 9.1 | 735.0 | 2555.3 / 753.0 | 74.5 / 4.5 | 1271.9 / 756.4 | 653.8 | 232.0 | 5531.7 |
| 4,096 | 9859.8 | 585.5 | 9.1 | 1359.8 | 4495.5 / 856.0 | 126.0 / 5.3 | 2197.8 / 875.5 | 1241.3 | 430.3 | 9859.8 |
| **8,192 ★** | **18327.7** | **629.9** | **9.0** | **2567.6** | **8093.4 / 951.0** | **255.5 / 5.3** | **4199.2 / 916.4** | **2405.2** | **797.8** | **18327.7** |
| 16,384 | 34777.8 | 663.9 | 9.2 | 5108.1 | 14786.8 / 1041.0 | 530.5 / 5.1 | 8082.8 / 952.2 | 4719.8 | 1540.6 | 34777.8 |

#### Backward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 6435.1 | 224.3 | 18.5 | 701.8 | 2254.3 / 426.8 | 192.4 / 0.9 | 1966.0 / 244.7 | 854.2 | 448.0 | 6435.1 |
| 2,048 | 7432.0 | 388.3 | 43.3 | 1149.8 | 3057.9 / 629.2 | 119.0 / 2.8 | 1882.7 / 511.0 | 905.9 | 273.4 | 7432.0 |
| 4,096 | 11198.5 | 515.5 | 28.5 | 1613.6 | 4412.1 / 872.2 | 182.8 / 3.7 | 2909.0 / 661.4 | 1564.7 | 487.8 | 11198.5 |
| **8,192 ★** | **17771.0** | **649.6** | **46.5** | **2807.1** | **6543.2 / 1176.3** | **348.4 / 3.9** | **4399.3 / 874.8** | **2698.5** | **928.0** | **17771.0** |
| 16,384 | 32719.1 | 705.7 | 27.7 | 5743.0 | 11955.4 / 1287.6 | 595.5 / 4.5 | 7671.1 / 1003.3 | 5018.8 | 1707.6 | 32719.1 |

#### Backward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 3693.3 | 390.7 | 9.0 | 456.6 | 1786.1 / 538.7 | 47.7 / 3.5 | 909.9 / 528.7 | 353.5 | 130.5 | 3693.3 |
| 2,048 | 5684.7 | 507.7 | 9.2 | 766.5 | 2659.0 / 723.6 | 78.9 / 4.3 | 1294.5 / 743.2 | 654.3 | 222.3 | 5684.7 |
| 4,096 | 9998.1 | 577.4 | 9.0 | 1400.4 | 4619.1 / 833.1 | 130.4 / 5.1 | 2195.8 / 876.3 | 1240.7 | 402.7 | 9998.1 |
| **8,192 ★** | **18477.8** | **624.8** | **9.1** | **2704.4** | **8082.5 / 952.2** | **252.9 / 5.3** | **4237.6 / 908.1** | **2406.8** | **784.5** | **18477.8** |
| 16,384 | 34870.5 | 662.2 | 8.9 | 5168.7 | 14829.6 / 1038.0 | 527.7 / 5.1 | 8086.9 / 951.7 | 4724.2 | 1524.4 | 34870.5 |

#### Step time — Triton vs HIPBLASLT vs CK

Forward + Backward wall time per BS. **Δ** columns compare against Triton at the same BS.

| BS | Triton fwd | Triton bwd | Triton step | HBLAS fwd | HBLAS bwd | HBLAS step | HBLAS Δ | CK fwd | CK bwd | CK step | CK Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,132 | 3,568 | **5,700** | 5,422 | 6,435 | 11,857 | +108.0% | 2,301 | 3,693 | 5,994 | +5.2% |
| 2,048 | 3,553 | 5,532 | **9,085** | 5,785 | 7,432 | 13,217 | +45.5% | 3,908 | 5,685 | 9,592 | +5.6% |
| 4,096 | 6,439 | 9,860 | **16,298** | 7,609 | 11,198 | 18,808 | +15.4% | 7,141 | 9,998 | 17,139 | +5.2% |
| 8,192 ★ | 12,410 | 18,328 | **30,738** | 12,988 | 17,771 | 30,759 | +0.1% | 13,821 | 18,478 | 32,299 | +5.1% |
| 16,384 | 24,149 | 34,778 | **58,927** | 22,785 | 32,719 | 55,504 | -5.8% | 26,953 | 34,870 | 61,824 | +4.9% |

### Qwen3-30B-A3B

`experts=128 top_k=8 hidden=2048 intermediate=768 ep_size=8 local_experts=16`

#### Forward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2766.9 | 55.9 | 179 | 10.7 | 665.5 | 309.1 / 333.5 | 49.8 / 1.3 | 183.3 / 281.1 | 1138.2 | 292.0 | 2648.6 |
| 4,096 | 2808.1 | 110.1 | 299 | 10.7 | 674.3 | 355.1 / 580.6 | 44.3 / 2.8 | 183.6 / 561.5 | 1157.1 | 303.5 | 2728.6 |
| 8,192 | 3211.2 | 192.6 | 475 | 10.1 | 926.9 | 449.2 / 917.8 | 79.8 / 3.2 | 302.3 / 681.9 | 920.8 | 455.7 | 3144.8 |
| 16,384 | 5742.3 | 215.4 | 505 | 10.3 | 1740.2 | 830.1 / 993.5 | 143.1 / 3.5 | 611.8 / 673.9 | 1525.0 | 816.6 | 5677.1 |
| **32,768 ★** | **11227.8** | **220.3** | **504** | **10.6** | **3414.1** | **1643.2 / 1003.7** | **282.6 / 3.6** | **1251.7 / 658.8** | **3018.3** | **1539.1** | **11159.5** |

#### Forward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 4431.2 | 34.9 | 112 | 9.5 | 798.5 | 596.8 / 172.7 | 165.3 / 0.4 | 526.9 / 97.8 | 1832.7 | 337.0 | 4266.7 |
| 4,096 | 4376.2 | 70.7 | 192 | 11.3 | 720.9 | 887.8 / 232.2 | 215.3 / 0.6 | 694.6 / 148.4 | 1177.4 | 497.4 | 4204.6 |
| 8,192 | 4795.9 | 129.0 | 318 | 10.1 | 926.1 | 682.2 / 604.4 | 95.8 / 2.6 | 499.4 / 412.8 | 1894.7 | 550.6 | 4659.0 |
| 16,384 | 6040.2 | 204.8 | 481 | 10.2 | 1751.1 | 954.1 / 864.3 | 141.4 / 3.6 | 565.6 / 728.9 | 1711.8 | 814.9 | 5949.2 |
| **32,768 ★** | **11455.2** | **216.0** | **494** | **11.2** | **3405.3** | **1788.5 / 922.2** | **282.4 / 3.6** | **1027.5 / 802.6** | **3248.2** | **1562.1** | **11325.3** |

#### Forward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2615.0 | 59.1 | 189 | 9.7 | 634.3 | 294.6 / 349.8 | 30.2 / 2.1 | 156.8 / 328.6 | 1111.7 | 273.5 | 2510.8 |
| 4,096 | 2692.3 | 114.9 | 312 | 11.4 | 644.0 | 397.5 / 518.7 | 43.2 / 2.9 | 181.2 / 568.8 | 1026.6 | 310.9 | 2614.7 |
| 8,192 | 3368.3 | 183.6 | 453 | 11.7 | 919.3 | 662.2 / 622.6 | 78.5 / 3.2 | 352.0 / 585.7 | 781.3 | 490.8 | 3295.8 |
| 16,384 | 6108.5 | 202.5 | 475 | 11.0 | 1739.6 | 1103.3 / 747.4 | 143.3 / 3.5 | 702.9 / 586.6 | 1538.5 | 800.8 | 6039.4 |
| **32,768 ★** | **11995.6** | **206.2** | **471** | **11.1** | **3401.4** | **2207.6 / 747.1** | **281.0 / 3.6** | **1493.6 / 552.1** | **3000.0** | **1532.5** | **11927.1** |

#### Backward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1909.6 | 161.9 | 9.3 | 746.3 | 258.2 / 798.4 | 32.0 / 3.9 | 170.8 / 603.4 | 566.6 | 126.4 | 1909.6 |
| 4,096 | 2385.6 | 259.3 | 9.0 | 750.9 | 477.9 / 862.8 | 61.2 / 4.1 | 310.6 / 663.8 | 629.2 | 146.9 | 2385.6 |
| 8,192 | 3384.9 | 365.4 | 9.1 | 817.0 | 906.6 / 909.6 | 113.3 / 4.4 | 529.6 / 778.5 | 744.3 | 265.1 | 3384.9 |
| 16,384 | 6617.5 | 373.8 | 8.6 | 1622.9 | 1849.4 / 891.8 | 221.7 / 4.5 | 1014.9 / 812.5 | 1440.6 | 459.4 | 6617.5 |
| **32,768 ★** | **13059.5** | **378.9** | **9.0** | **3147.9** | **3659.1 / 901.5** | **466.0 / 4.3** | **2013.7 / 819.0** | **2838.5** | **925.3** | **13059.5** |

#### Backward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 4124.0 | 75.0 | 31.4 | 1581.1 | 822.6 / 250.6 | 95.0 / 1.3 | 719.2 / 143.3 | 624.2 | 250.5 | 4124.0 |
| 4,096 | 4420.1 | 139.9 | 21.2 | 1053.7 | 1169.6 / 352.5 | 160.6 / 1.6 | 906.1 / 227.5 | 688.3 | 420.5 | 4420.1 |
| 8,192 | 5441.9 | 227.3 | 48.6 | 1269.0 | 1345.8 / 612.7 | 241.0 / 2.1 | 1146.5 / 359.6 | 971.4 | 419.6 | 5441.9 |
| 16,384 | 8089.9 | 305.8 | 25.3 | 2064.1 | 2013.8 / 819.0 | 289.0 / 3.5 | 1330.1 / 620.0 | 1751.7 | 615.9 | 8089.9 |
| **32,768 ★** | **14832.8** | **333.6** | **38.3** | **3413.5** | **4137.4 / 797.2** | **582.5 / 3.5** | **2358.6 / 699.3** | **3135.1** | **1167.4** | **14832.8** |

#### Backward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1753.1 | 176.4 | 8.9 | 614.2 | 266.8 / 772.7 | 31.7 / 4.0 | 167.2 / 616.4 | 573.9 | 90.4 | 1753.1 |
| 4,096 | 2233.5 | 276.9 | 9.2 | 627.8 | 461.0 / 894.5 | 60.8 / 4.1 | 288.7 / 714.2 | 637.1 | 149.0 | 2233.5 |
| 8,192 | 3298.5 | 375.0 | 9.2 | 841.7 | 849.4 / 970.8 | 112.9 / 4.5 | 486.4 / 847.8 | 741.9 | 256.9 | 3298.5 |
| 16,384 | 6476.6 | 382.0 | 8.9 | 1655.3 | 1754.4 / 940.1 | 221.0 / 4.6 | 916.5 / 899.8 | 1434.4 | 486.1 | 6476.6 |
| **32,768 ★** | **12934.5** | **382.5** | **9.0** | **3164.3** | **3675.7 / 897.4** | **483.3 / 4.2** | **1828.0 / 902.2** | **2820.9** | **953.3** | **12934.5** |

#### Step time — Triton vs HIPBLASLT vs CK

Forward + Backward wall time per BS. **Δ** columns compare against Triton at the same BS.

| BS | Triton fwd | Triton bwd | Triton step | HBLAS fwd | HBLAS bwd | HBLAS step | HBLAS Δ | CK fwd | CK bwd | CK step | CK Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,767 | 1,910 | **4,676** | 4,431 | 4,124 | 8,555 | +82.9% | 2,615 | 1,753 | 4,368 | -6.6% |
| 4,096 | 2,808 | 2,386 | **5,194** | 4,376 | 4,420 | 8,796 | +69.4% | 2,692 | 2,233 | 4,926 | -5.2% |
| 8,192 | 3,211 | 3,385 | **6,596** | 4,796 | 5,442 | 10,238 | +55.2% | 3,368 | 3,299 | 6,667 | +1.1% |
| 16,384 | 5,742 | 6,617 | **12,360** | 6,040 | 8,090 | 14,130 | +14.3% | 6,109 | 6,477 | 12,585 | +1.8% |
| 32,768 ★ | 11,228 | 13,060 | **24,287** | 11,455 | 14,833 | 26,288 | +8.2% | 11,996 | 12,934 | 24,930 | +2.6% |

### Qwen3-235B-A22B

`experts=128 top_k=8 hidden=4096 intermediate=1536 ep_size=8 local_experts=16`

#### Forward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2210.8 | 279.7 | 584 | 10.7 | 459.0 | 521.8 / 790.1 | 35.4 / 3.6 | 240.9 / 855.7 | 616.9 | 247.5 | 2132.2 |
| 4,096 | 3558.1 | 347.6 | 556 | 11.1 | 829.9 | 927.9 / 888.7 | 63.6 / 4.0 | 488.9 / 843.4 | 770.0 | 394.2 | 3485.6 |
| 8,192 | 6420.3 | 385.3 | 523 | 9.9 | 1559.6 | 1525.3 / 1081.2 | 118.2 / 4.3 | 1007.0 / 818.9 | 1497.7 | 634.8 | 6352.6 |
| **16,384 ★** | **12586.4** | **393.1** | **485** | **9.9** | **3012.7** | **3132.9 / 1052.9** | **242.7 / 4.1** | **2033.0 / 811.2** | **2909.9** | **1178.6** | **12519.8** |
| 32,768 | 25174.2 | 393.1 | 461 | 11.0 | 5925.8 | 7004.8 / 941.8 | 474.2 / 4.2 | 3730.7 / 884.1 | 5661.2 | 2297.8 | 25105.7 |

#### Forward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 4257.4 | 145.3 | 303 | 13.0 | 700.0 | 954.9 / 431.8 | 253.0 / 0.5 | 785.7 / 262.4 | 858.2 | 507.2 | 4072.0 |
| 4,096 | 4936.9 | 250.6 | 401 | 9.9 | 839.9 | 1209.8 / 681.6 | 63.9 / 3.9 | 628.3 / 656.2 | 1699.7 | 341.3 | 4793.0 |
| 8,192 | 6502.7 | 380.4 | 516 | 10.1 | 1561.3 | 1602.9 / 1028.9 | 119.4 / 4.2 | 847.4 / 973.2 | 1590.8 | 643.7 | 6375.4 |
| **16,384 ★** | **12886.7** | **383.9** | **474** | **10.2** | **3005.5** | **3086.2 / 1068.8** | **267.6 / 3.8** | **1694.4 / 973.3** | **3347.4** | **1295.5** | **12706.8** |
| 32,768 | 24046.5 | 411.5 | 483 | 11.0 | 5935.4 | 5586.0 / 1181.0 | 552.5 / 3.6 | 3107.6 / 1061.4 | 6265.1 | 2425.1 | 23882.7 |

#### Forward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2230.9 | 277.2 | 579 | 11.3 | 452.7 | 631.5 / 652.9 | 35.7 / 3.5 | 291.7 / 706.7 | 449.2 | 266.4 | 2138.5 |
| 4,096 | 3901.8 | 317.0 | 507 | 11.1 | 835.8 | 1139.9 / 723.4 | 62.8 / 4.0 | 596.5 / 691.2 | 813.3 | 373.7 | 3833.1 |
| 8,192 | 7182.0 | 344.5 | 467 | 10.0 | 1558.2 | 2062.7 / 799.6 | 117.2 / 4.3 | 1230.8 / 670.0 | 1512.2 | 622.8 | 7114.1 |
| **16,384 ★** | **14062.6** | **351.8** | **434** | **11.6** | **3003.9** | **4300.0 / 767.1** | **235.3 / 4.3** | **2376.7 / 693.9** | **2897.7** | **1166.0** | **13991.1** |
| 32,768 | 27701.9 | 357.2 | 419 | 11.0 | 5941.9 | 8578.1 / 769.1 | 473.3 / 4.3 | 4568.5 / 722.0 | 5789.5 | 2272.2 | 27634.7 |

#### Backward — Triton

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2721.4 | 454.5 | 8.8 | 584.3 | 925.2 / 891.3 | 52.1 / 4.8 | 522.5 / 789.2 | 507.9 | 120.7 | 2721.4 |
| 4,096 | 4545.9 | 544.2 | 9.1 | 825.9 | 1794.4 / 919.1 | 92.6 / 5.4 | 866.8 / 951.3 | 728.3 | 228.7 | 4545.9 |
| 8,192 | 8791.2 | 562.8 | 9.0 | 1540.7 | 3561.6 / 926.1 | 181.3 / 5.6 | 1666.6 / 989.6 | 1400.3 | 431.8 | 8791.2 |
| **16,384 ★** | **17070.4** | **579.7** | **8.6** | **2964.8** | **6778.6 / 973.2** | **378.4 / 5.3** | **3379.1 / 976.2** | **2734.9** | **826.0** | **17070.4** |
| 32,768 | 37407.4 | 529.1 | 8.8 | 5852.6 | 12849.1 / 1026.9 | 807.9 / 5.0 | 10817.6 / 609.8 | 5418.0 | 1653.5 | 37407.4 |

#### Backward — HIPBLASLT

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 4663.3 | 265.2 | 9.1 | 907.2 | 1430.1 / 576.6 | 124.5 / 2.0 | 998.1 / 413.1 | 815.1 | 379.3 | 4663.3 |
| 4,096 | 5917.9 | 418.0 | 31.5 | 995.0 | 1957.4 / 842.6 | 166.9 / 3.0 | 1430.1 / 576.6 | 979.7 | 357.3 | 5917.9 |
| 8,192 | 9946.5 | 497.4 | 44.9 | 1623.8 | 3220.3 / 1024.3 | 272.8 / 3.7 | 2537.8 / 649.9 | 1649.2 | 597.7 | 9946.5 |
| **16,384 ★** | **17708.8** | **558.8** | **22.2** | **3434.4** | **6266.6 / 1052.7** | **470.3 / 4.3** | **3447.5 / 956.8** | **3063.7** | **1004.1** | **17708.8** |
| 32,768 | 33594.3 | 589.1 | 32.1 | 6469.1 | 11861.0 / 1112.4 | 903.7 / 4.5 | 6671.8 / 988.8 | 5753.5 | 1903.1 | 33594.3 |

#### Backward — CK

| Batch Size | Time (us) | Compute (TFLOPS) | sort (us) | dispatch (us) | fc1 (us / TFLOPS) | swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2498.2 | 495.1 | 9.2 | 471.1 | 926.4 / 890.1 | 49.6 / 5.1 | 507.7 / 812.1 | 408.7 | 125.5 | 2498.2 |
| 4,096 | 4495.0 | 550.4 | 8.9 | 850.1 | 1756.3 / 939.1 | 89.8 / 5.6 | 843.8 / 977.3 | 729.2 | 216.9 | 4495.0 |
| 8,192 | 8802.4 | 562.1 | 9.0 | 1527.9 | 3652.8 / 903.0 | 184.9 / 5.4 | 1601.5 / 1029.8 | 1397.4 | 428.9 | 8802.4 |
| **16,384 ★** | **17145.1** | **577.2** | **9.0** | **3113.0** | **6717.0 / 982.1** | **387.0 / 5.2** | **3357.4 / 982.5** | **2735.0** | **826.7** | **17145.1** |
| 32,768 | 33485.4 | 591.0 | 8.7 | 6057.6 | 12666.1 / 1041.7 | 797.9 / 5.0 | 6874.9 / 959.6 | 5422.6 | 1657.8 | 33485.4 |

#### Step time — Triton vs HIPBLASLT vs CK

Forward + Backward wall time per BS. **Δ** columns compare against Triton at the same BS.

| BS | Triton fwd | Triton bwd | Triton step | HBLAS fwd | HBLAS bwd | HBLAS step | HBLAS Δ | CK fwd | CK bwd | CK step | CK Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,211 | 2,721 | **4,932** | 4,257 | 4,663 | 8,921 | +80.9% | 2,231 | 2,498 | 4,729 | -4.1% |
| 4,096 | 3,558 | 4,546 | **8,104** | 4,937 | 5,918 | 10,855 | +33.9% | 3,902 | 4,495 | 8,397 | +3.6% |
| 8,192 | 6,420 | 8,791 | **15,211** | 6,503 | 9,947 | 16,449 | +8.1% | 7,182 | 8,802 | 15,984 | +5.1% |
| 16,384 ★ | 12,586 | 17,070 | **29,657** | 12,887 | 17,709 | 30,596 | +3.2% | 14,063 | 17,145 | 31,208 | +5.2% |
| 32,768 | 25,174 | 37,407 | **62,582** | 24,047 | 33,594 | 57,641 | -7.9% | 27,702 | 33,485 | 61,187 | -2.2% |

## Production-point summary (spread routing)

Latency at each model's production token count per EP rank (`mbs × seq`,
forward only). All times in microseconds, single MoE layer, BF16
activations, BF16 weights, Triton grouped-gemm backend, sync-free
MoE stage 2, DeepEP intranode all-to-all.

| Model | Tokens/rank | Total | dispatch | fc1 (TFLOPS) | swiglu | fc2 (TFLOPS) | combine | misc | e2e (TFLOPS) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_30B_A3B    | 32,768 | 11,191 | 3,393 (30%) | 1,656 (996) |   282 | 1,228 (671) | 3,014 (27%) | 1,541 (14%) | 221 |
| deepseek_v2_lite | 49,152 | 16,274 | 4,807 (30%) | 3,578 (951) |   529 | 2,088 (815) | 3,563 (22%) | 1,634 (10%) | 313 |
| qwen3_235B_A22B  | 16,384 | 12,612 | 3,013 (24%) | 3,142 (1050) |   242 | 2,036 (810) | 2,944 (23%) | 1,161 (9%)  | 392 |
| deepseek_v3      |  8,192 | 12,427 | 2,563 (21%) | 3,917 (982) |   152 | 2,142 (898) | 2,556 (21%) | 1,023 (8%)  | 464 |

Key reads:

- **Dispatch ≈ combine, together 40–60% of the MoE layer at every
  production point.** They scale linearly with `num_tokens × topk` and
  are completely independent of `ffn`. The smaller-hidden +
  larger-token-count models (DSV2-Lite, Qwen3-30B-A3B) spend a much
  larger fraction of their MoE budget in comm than the GEMM-heavy DSV3.
- **fc1 is the GEMM that pays.** Once `BS ≥ 8k`, fc1 lands between
  **950–1080 TFLOPS** across all four configs — essentially at the
  CDNA4 BF16 matrix-engine ceiling (theoretical ~1.3 PFLOPS, sustained
  ceiling ~1.1 PFLOPS for grouped GEMM with reasonable group sizes).
  Even Qwen3-30B-A3B with `ffn=768` saturates fc1 at production BS
  (996 TFLOPS), because the (N, 2·ffn) output is wide enough to fill
  the MFMA tile when `N` is large.
- **fc2 lags by 10–30%.** The down-projection has half the FLOPs per
  output element and the GEMM shape `(N, F) × (F, H)` is K-skinny when
  `F` is small. Qwen3-30B-A3B sits at **671 TFLOPS** (`F=768` is too
  thin to fill the K dimension); DSV3 (`F=2048`) recovers to **898
  TFLOPS**; Qwen3-235B and DSV2-Lite are in between.
- **swiglu is consistently 3–5 TFLOPS** — i.e. bandwidth-bound, no
  surprise (it's an elementwise `silu(gate) * up * prob` over a
  `(N, F)` tensor). It costs **2–4% of the MoE layer** in absolute
  terms, but is the only stage where the matrix engines sit idle, so
  fusing it into either fc1's epilogue or fc2's prologue would be a
  clean ~3% win across the board.
- **misc (sort + dispatch_preprocess + post + combine_preprocess + post)
  scales sub-linearly** and stays around 10–14% — cheap to keep, but
  worth folding into the dispatch / combine kernels for the bandwidth-
  bound models if we want to push beyond the current ceiling.

## Reproducing

```bash
ssh mi355-gpu-26
podman exec -it xiaoming-dev bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/Primus
bash slab/notes/moe_perf/turbo/run_all_models.sh
# writes <model>-spread-breakdown.csv into slab/notes/moe_perf/turbo/
```

The driver pins `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON` (CK Tile has
HIP-invalid-config issues on very small expert groups) and passes
`--sync-free-stage 2` so DeepEP buffers are pre-allocated for the worst
case. Per-model flags (`hidden_size`, `moe_ffn_hidden_size`,
`num_experts`, `topk`, `deepep_num_cu`) live in the `MODELS` array of
the driver and were copied verbatim from
`primus/configs/models/megatron/<model>.yaml` plus the matching
`examples/megatron/configs/MI355X/<model>-BF16-pretrain.yaml`.

The bench script `bench_turbo_moe_e2e.py` only depends on
`primus_turbo.pytorch` and `torch.distributed`; it has no relative
imports, so it runs as-is from any cwd as long as the container exposes
`primus_turbo`. See `bench-script-notes.md` for the per-stage
table the script prints and the CLI knobs it accepts.

## Autotune experiment (kept disabled in the canonical sweep)

We also tried `PRIMUS_TURBO_AUTO_TUNE=1` (no `*_BACKEND` pin), expecting
`GlobalBackendManager` to pick the best grouped-GEMM backend per shape
and **at worst** fall back to Triton. In practice autotune is a **mixed
bag**: it wins big on grouped GEMM but globally regresses DeepEP
combine/dispatch, so end-to-end it only wins on DSV3 at the PROD point
and slightly regresses Qwen3-30B-A3B / DSV2-Lite.

End-to-end (per layer, one micro-batch, fwd + bwd):

| Model | PROD BS | Triton fwd (us) | Auto fwd (us) | Δfwd | Triton bwd (us) | Auto bwd (us) | Δbwd | Triton fwd+bwd | Auto fwd+bwd | Δstep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V2-Lite[¹] | 32,768 | 10,886 | 11,076 | +1.7% ✗ | 14,222 | 14,659 | +3.1% ✗ | 25,108 | 25,735 | **+2.5% ✗** |
| DeepSeek-V3 | 8,192 | 12,394 | 12,521 | +1.0% ✗ | 18,305 | 17,149 | -6.3% ✓ | 30,699 | 29,670 | **-3.4% ✓** |
| Qwen3-30B-A3B | 32,768 | 11,224 | 11,146 | -0.7% ✓ | 13,058 | 13,692 | +4.9% ✗ | 24,281 | 24,838 | **+2.3% ✗** |
| Qwen3-235B-A22B | 16,384 | 12,613 | 12,435 | -1.4% ✓ | 17,068 | 17,273 | +1.2% ✗ | 29,681 | 29,708 | **+0.1% ≈** |

[¹] DSV2-Lite production is BS=49,152, but the autotune sweep timed out
on that point (and on DSV3 BS=16,384) — autotune's per-shape warmup +
profile loop pushes both to >8 min/run. Comparing at the next-largest
shared BS, 32,768.

### Where autotune helps — grouped GEMM (PROD BS)

| Model | PROD BS | fc2 fwd TFLOPS (Tri→Auto) | fc1 bwd TFLOPS (Tri→Auto) | fc2 bwd TFLOPS (Tri→Auto) |
|---|---:|---|---|---|
| DeepSeek-V2-Lite | 32,768 | 782 → 989 (+26%) | 944 → 1069 (+13%) | 767 → 733 (-4%) |
| DeepSeek-V3 | 8,192 | 898 → 1129 (+26%) | 954 → 1169 (+22%) | 920 → 1064 (+16%) |
| Qwen3-30B-A3B | 32,768 | 659 → 812 (+23%) | 904 → 977 (+8%) | 815 → 816 (±0%) |
| Qwen3-235B-A22B | 16,384 | 816 → 1066 (+31%) | 978 → 1081 (+11%) | 973 → 965 (-1%) |

Forward `fc2` and backward `fc1` see consistent +10–30% TFLOPS wins —
those shapes have K-skinny `N×K` GEMMs where CK-Tile / aiter beat
Triton, and autotune picks the right backend automatically.

### Where autotune hurts — DeepEP A2A side effect (PROD BS)

| Model | PROD BS | combine fwd (us) | dispatch bwd (us) | combine bwd (us) |
|---|---:|---|---|---|
| DeepSeek-V2-Lite | 32,768 | 2387 → 2573 (+8%) | 2573 → 2728 (+6%) | 2590 → 2825 (+9%) |
| DeepSeek-V3 | 8,192 | 2525 → 2831 (+12%) | 2597 → 2954 (+14%) | 2410 → 2722 (+13%) |
| Qwen3-30B-A3B | 32,768 | 2984 → 3107 (+4%) | 3136 → 3427 (+9%) | 2830 → 3144 (+11%) |
| Qwen3-235B-A22B | 16,384 | 2916 → 3169 (+9%) | 2976 → 3257 (+9%) | 2737 → 3050 (+11%) |

Combine and bwd-dispatch regress **+5–14%** every time autotune is on.
We confirmed this is **not noise**: with autotune off and Triton pinned,
DSV3 BS=8192 combine fwd reproduces at 2,544 us ±0.6% across 3 runs;
with autotune on it reproduces at 2,887 us ±2.2% across 3 runs.

### Root cause

`PRIMUS_TURBO_AUTO_TUNE` is a **global** switch in
`GlobalBackendManager`. It enables `tune()` on every
`AutoKernelDispatcher` subclass, including `MoEDispatchKernelDispatcher`
and `MoECombineKernelDispatcher`. On this image both of those
dispatchers only have one viable backend (the bundled
`MoEDispatchTurboBackend` / `MoECombineTurboBackend`): the `DEEP_EP`
backend variants are rejected by `can_handle()` because they gate on
`import deep_ep` of the **standalone** package, but primus-turbo only
ships its bundled DeepEP at `primus_turbo.pytorch.deep_ep`. So
autotune has nothing to pick from for dispatch/combine — but it still
runs `tune()`'s warmup + profile loop against the (only) `TURBO`
backend, exercising the DeepEP A2A buffers in an extra warmup loop.
That measurably degrades the steady-state A2A latency by ~10%, and the
effect persists across batch sizes.

`bench_turbo_moe_e2e.py` partially mitigates this by monkey-patching
the two MoE dispatchers' `tune()` to a no-op when autotune is on, which
recovers ~2–3% of the regression but not all of it — the rest comes
from grouped-GEMM autotune's own warmup loops affecting GPU state.

### Verdict and recommendation

- **Keep the canonical sweep on `BACKEND=TRITON`** (current state of
  `run_all_models.sh`). That gives us a regression-free reference that
  matches what `examples/megatron/configs/MI355X/*-BF16-pretrain.yaml`
  actually run with today.
- **Use autotune selectively** if you only care about backward (e.g.
  DSV3 grad-accumulation-heavy training, where bwd dominates step time)
  — it saves ~6% bwd on DSV3.
- **Upstream fix worth filing** against primus-turbo: scope
  `PRIMUS_TURBO_AUTO_TUNE` to grouped-GEMM dispatchers only (or expose
  per-dispatcher opt-in), and either install the standalone `deep_ep`
  module or relax the `HAVE_DEEP_EP` guard in
  `MoEDispatchDeepEPBackend.can_handle()` so the DeepEP backend
  candidate is actually visible to autotune.

To rerun the autotune comparison:

```bash
# Edit run_all_models.sh to swap the two stanzas: comment out the
# Triton-pinned block and uncomment the AUTO_TUNE=1 block.
PRIMUS_TURBO_AUTO_TUNE=1 bash slab/notes/moe_perf/turbo/run_all_models.sh
mv *.csv archive_autotune_experiment/
```

## Backend comparison summary — Triton vs HIPBLASLT vs CK (no autotune)

Same setup as the per-model tables above: `BACKEND=TRITON|HIPBLASLT|CK bash run_all_models.sh`, autotune off, raw CSVs in `archive_backends/<backend>/`. The full per-BS breakdown for each backend is in the *Per-model breakdown* section.

### End-to-end step time at PROD batch size (us)

| Model | PROD BS | Triton fwd | HBLAS fwd | CK fwd | Triton bwd | HBLAS bwd | CK bwd | Triton step | HBLAS step | CK step | HBLAS Δstep | CK Δstep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V2-Lite | 49,152 | 16,270 | — | 17,880 | 21,255 | — | 21,872 | **37,525** | — | 39,752 | — | +5.9% |
| DeepSeek-V3 | 8,192 | 12,410 | 12,988 | 13,821 | 18,328 | 17,771 | 18,478 | **30,738** | 30,759 | 32,299 | +0.1% | +5.1% |
| Qwen3-30B-A3B | 32,768 | 11,228 | 11,455 | 11,996 | 13,060 | 14,833 | 12,934 | **24,287** | 26,288 | 24,930 | +8.2% | +2.6% |
| Qwen3-235B-A22B | 16,384 | 12,586 | 12,887 | 14,063 | 17,070 | 17,709 | 17,145 | **29,657** | 30,596 | 31,208 | +3.2% | +5.2% |

### Grouped-GEMM kernels (TFLOPS at PROD BS)

| Model | PROD BS | fc1_fwd Tri / HBLAS / CK | fc2_fwd Tri / HBLAS / CK | fc1_bwd Tri / HBLAS / CK | fc2_bwd Tri / HBLAS / CK |
|---|---:|---|---|---|---|
| DeepSeek-V3 | 8,192 | 981 / 919 / 781 | 907 / 1054 / 761 | 951 / 1176 / 952 | 916 / 875 / 908 |
| Qwen3-30B-A3B | 32,768 | 1004 / 922 / 747 | 659 / 803 / 552 | 901 / 797 / 897 | 819 / 699 / 902 |
| Qwen3-235B-A22B | 16,384 | 1053 / 1069 / 767 | 811 / 973 / 694 | 973 / 1053 / 982 | 976 / 957 / 982 |

### DeepEP combine/dispatch side-effect (us at PROD BS)

| Model | PROD BS | dispatch_fwd | combine_fwd | dispatch_bwd | combine_bwd |
|---|---:|---|---|---|---|
| DeepSeek-V3 | 8,192 | 2559 / 2563 (+0.1%) / 2560 (+0.0%) | 2526 / 2909 (+15.2%) / 2537 (+0.4%) | 2568 / 2807 (+9.3%) / 2704 (+5.3%) | 2405 / 2699 (+12.2%) / 2407 (+0.1%) |
| Qwen3-30B-A3B | 32,768 | 3414 / 3405 (-0.3%) / 3401 (-0.4%) | 3018 / 3248 (+7.6%) / 3000 (-0.6%) | 3148 / 3413 (+8.4%) / 3164 (+0.5%) | 2839 / 3135 (+10.4%) / 2821 (-0.6%) |
| Qwen3-235B-A22B | 16,384 | 3013 / 3006 (-0.2%) / 3004 (-0.3%) | 2910 / 3347 (+15.0%) / 2898 (-0.4%) | 2965 / 3434 (+15.8%) / 3113 (+5.0%) | 2735 / 3064 (+12.0%) / 2735 (+0.0%) |

### Verdict

- **At PROD batch size, Triton is the best choice on all 4 models.**
  HIPBLASLT ties on DSV3 (+0.1%) and loses on the other three (+3 to
  +8%); CK loses on all four (+2 to +6%).
- **HIPBLASLT shines at very large BS** (DSV3 BS=16,384: -5.8%,
  Q235B-A22B BS=32,768: -7.9%) — these are points where the GEMM is
  large enough that the heuristics actually pay off. Useful if a model
  ever runs above its PROD point.
- **Triton is by far the best at small BS** (DSV3 BS=1,024:
  HIPBLASLT is 2.1× slower; Qwen3-30B-A3B BS=2,048: HIPBLASLT is 1.8×
  slower). HIPBLASLT's per-shape heuristic warmup dominates at these
  sizes.
- **HIPBLASLT also robustness-fails** on DSV2-Lite BS=49,152 (first-call
  hangs >10 min). Triton has no such failure modes in the sweep.
- **CK is a no-go on this image** — no shape where it beats Triton,
  and the prebuilt kernels lose 20–30% TFLOPS on fc1/fc2 vs Triton.
  Likely needs a CK-Tile tuning run for these MoE expert sizes.

To rerun the backend sweep:

```bash
for B in TRITON HIPBLASLT CK; do
  mkdir -p slab/notes/moe_perf/turbo/archive_backends/${B,,}
  BACKEND=$B \
    OUT_DIR=slab/notes/moe_perf/turbo/archive_backends/${B,,} \
    bash slab/notes/moe_perf/turbo/run_all_models.sh
done
python3 slab/notes/moe_perf/turbo/build_backend_comparison.py > /tmp/backend_comparison.md
```
