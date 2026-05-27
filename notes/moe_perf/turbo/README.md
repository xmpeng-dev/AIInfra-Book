# Turbo MoE forward-pass breakdown — MI355X EP8

Forward-only latency breakdown of the Primus-turbo MoE pipeline
(dispatch + grouped fc1 + swiglu + grouped fc2 + combine), measured on a
single MI355X node with EP=8 (1 node × 8 GPUs, no TP). The numbers below
correspond to **one transformer layer's MoE submodule, one micro-batch,
forward only**, with the routing table that Megatron's
`moe_router_force_load_balancing` produces.

All data was collected by `bench_turbo_moe_e2e.py` inside the
`xiaoming-dev` container on `mi355-gpu-26`. The wrapper that swept all
four configs is `run_all_models.sh`.

## Layout

```
moe_perf/turbo/                   # this directory
├── README.md                     # this file
├── bench_turbo_moe_e2e.py        # standalone benchmark (only depends on primus_turbo + torch)
├── run_all_models.sh             # 4-model sweep driver
├── bench-script-notes.md         # per-stage table format and CLI knobs
└── *.csv                         # see Files table below
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
breakdown for each model is the corresponding `*-spread-breakdown.csv`.

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

Each table is the rank-0 view of one MoE layer's forward pass. `Time
(us)` is the end-to-end forward wall time (CUDA events). `Compute
(TFLOPS)` and `Global Memory (GB/s)` are derived from the per-rank
shape (`num_tokens × topk × 2 × hidden × ffn × 2` FLOPs for the two
grouped GEMMs, plus the dispatch / combine bytes). `all_kernels (us)`
is the sum of the per-stage CUDA-event medians; the residual vs `Time
(us)` is launch-queue overhead inside the loop. The ★ row marks each
config's production token count per rank (`mbs × seq_length`).

### DeepSeek-V2-Lite

`experts=64 top_k=6 hidden=2048 intermediate=1408 ep_size=8 local_experts=8`

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fused_moe (us) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  4,096 |  1766.0 | 240.8 | 424 |  9.3 |  482.6 |  594.5 |  408.4 |  202.8 |  1697.5 |
|  8,192 |  2981.1 | 285.3 | 456 |  9.7 |  887.4 | 1060.1 |  617.5 |  339.3 |  2913.9 |
| 16,384 |  5579.5 | 304.8 | 462 | 10.0 | 1674.8 | 2009.2 | 1231.1 |  591.8 |  5516.9 |
| 32,768 | 10938.2 | 311.0 | 459 | 10.2 | 3246.8 | 4143.5 | 2366.3 | 1107.0 | 10874.0 |
| **49,152 ★** | **16304.9** | **312.9** | **458** | **10.9** | **4822.3** | **6264.6** | **3524.4** | **1618.4** | **16240.5** |

### DeepSeek-V3

`experts=256 top_k=8 hidden=7168 intermediate=2048 ep_size=8 local_experts=32`

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fused_moe (us) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1,024 |  1947.3 | 370.5 | 1740 | 10.1 |  397.8 |   900.2 |  386.1 |  183.3 |  1877.6 |
|  2,048 |  3340.7 | 432.0 | 1185 | 10.4 |  718.8 |  1561.7 |  705.9 |  278.7 |  3275.5 |
|  4,096 |  6384.6 | 452.1 |  799 | 10.7 | 1336.2 |  3153.4 | 1308.1 |  511.6 |  6320.0 |
| **8,192 ★** | **12454.0** | **463.5** | **593** | **10.7** | **2558.9** | **6270.9** | **2529.6** | **1021.9** | **12392.0** |
| 16,384 | 24270.0 | 475.7 |  492 | 11.0 | 4988.2 | 12207.4 | 5040.1 | 1959.7 | 24206.4 |

### Qwen3-30B-A3B

`experts=128 top_k=8 hidden=2048 intermediate=768 ep_size=8 local_experts=16`

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fused_moe (us) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  2,048 |  1159.4 | 133.4 | 427 |  9.4 |  281.9 |  312.0 |  327.9 |  162.8 |  1094.0 |
|  4,096 |  1668.4 | 185.4 | 503 |  9.7 |  487.9 |  434.2 |  425.4 |  247.1 |  1604.3 |
|  8,192 |  3016.2 | 205.0 | 506 |  9.7 |  912.3 |  822.3 |  776.9 |  433.7 |  2954.9 |
| 16,384 |  5743.2 | 215.4 | 505 | 10.3 | 1741.8 | 1596.7 | 1533.9 |  797.3 |  5680.0 |
| **32,768 ★** | **11198.3** | **220.9** | **505** | **10.4** | **3412.3** | **3182.7** | **2994.3** | **1534.1** | **11133.8** |

### Qwen3-235B-A22B

`experts=128 top_k=8 hidden=4096 intermediate=1536 ep_size=8 local_experts=16`

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fused_moe (us) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  2,048 |  1884.4 | 328.2 | 686 |  9.1 |  448.2 |   706.8 |  463.1 |  193.3 |  1820.5 |
|  4,096 |  3445.5 | 359.0 | 575 |  9.7 |  826.1 |  1353.4 |  871.9 |  320.9 |  3382.0 |
|  8,192 |  6422.4 | 385.2 | 522 |  9.6 | 1554.7 |  2694.5 | 1480.1 |  619.7 |  6358.6 |
| **16,384 ★** | **12651.4** | **391.1** | **483** | **9.7** | **3005.7** | **5491.6** | **2913.4** | **1168.2** | **12588.6** |
| 32,768 | 25334.3 | 390.6 | 458 | 10.6 | 5932.2 | 11295.8 | 5772.4 | 2260.5 | 25271.6 |

### DeepSeek-V3 — cluster routing (legacy, small-BS sweep)

`experts=256 top_k=8 hidden=7168 intermediate=2048 ep_size=8 local_experts=32`

From `deepseek-v3-cluster-breakdown.csv`. Cluster routing concentrates
all `topk` experts of a token on one EP rank, so the dispatch all-to-all
collapses to a single peer and the TFLOPS column reads ~8× higher than
real production. Kept here as the only data we have for `num_tokens <
1024` (DeepEP's intranode buffer hits an edge case with spread routing
below ~256 tokens/rank).

| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | sort (us) | dispatch (us) | fused_moe (us) | combine (us) | misc (us) | all_kernels (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|     32 |  1243.4 |  18.1 | 2281 | 7.0 |  89.9 |   827.6 |   77.4 |  184.7 |  1186.6 |
|     64 |  1253.7 |  36.0 | 2277 | 7.3 |  96.4 |   830.0 |   83.4 |  179.4 |  1196.4 |
|    128 |  1326.3 |  68.0 | 2179 | 8.7 | 105.4 |   831.8 |  145.1 |  177.8 |  1268.8 |
|    256 |  1344.5 | 134.2 | 2202 | 8.7 | 108.0 |   824.2 |  166.7 |  179.4 |  1287.0 |
|    512 |  1391.6 | 259.3 | 2230 | 8.9 | 122.4 |   858.9 |  165.5 |  180.2 |  1335.8 |
|  1,024 |  1459.7 | 494.3 | 2322 | 8.8 | 156.8 |   882.7 |  164.7 |  191.2 |  1404.3 |
|  2,048 |  2282.7 | 632.2 | 1735 | 9.3 | 206.2 |  1540.3 |  248.4 |  223.0 |  2227.2 |
|  4,096 |  4331.6 | 666.3 | 1177 | 9.1 | 350.4 |  3087.7 |  448.9 |  379.4 |  4275.5 |
|  8,192 |  8179.8 | 705.7 |  902 | 9.4 | 565.5 |  6061.8 |  714.0 |  773.7 |  8124.5 |
| 16,384 | 15891.8 | 726.5 |  752 | 9.8 | 997.4 | 11845.9 | 1435.5 | 1547.5 | 15836.0 |

## Production-point summary (spread routing)

Latency at each model's production token count per EP rank (`mbs × seq`,
forward only). All times in microseconds, single MoE layer, BF16
activations, BF16 weights, Triton grouped-gemm backend, sync-free
MoE stage 2, DeepEP intranode all-to-all.

| Model | Tokens/rank | Total | dispatch | fc1 + swiglu + fc2 | combine | misc | Compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_30B_A3B    | 32,768 | 11,198 | 3,412 (30%) | 3,183 (28%) | 2,994 (27%) | 1,534 (14%) | 221 TFLOPS |
| deepseek_v2_lite | 49,152 | 16,305 | 4,822 (30%) | 6,265 (38%) | 3,524 (22%) | 1,618 (10%) | 313 TFLOPS |
| qwen3_235B_A22B  | 16,384 | 12,651 | 3,006 (24%) | 5,492 (43%) | 2,913 (23%) | 1,168 (9%)  | 391 TFLOPS |
| deepseek_v3      |  8,192 | 12,454 | 2,559 (21%) | 6,271 (50%) | 2,530 (20%) | 1,022 (8%)  | 464 TFLOPS |

Key reads:

- **Dispatch and combine together cost 40–60% of the MoE layer at
  every production point.** They scale linearly with `num_tokens × topk`
  and are completely independent of `ffn`. The
  smaller-hidden + larger-token-count models (DSV2-Lite, Qwen3-30B-A3B)
  spend a much larger fraction of their MoE budget in comm than the
  GEMM-heavy DSV3.
- **Dispatch ≈ combine.** Same A2A volume / same kernel cost, as
  expected from DeepEP intranode A2A symmetry.
- **fused_moe (grouped GEMM × 2 + SwiGLU) scales with `K × hidden ×
  ffn`.** DSV3's huge per-token compute (`K=8 × H=7168 × ffn=2048`)
  pushes it to 464 TFLOPS — the only config that's close to the BF16
  matrix-engine roofline. Qwen3-30B-A3B's tiny `ffn=768` keeps the
  grouped GEMM in the bandwidth regime (~220 TFLOPS).
- **misc (sort + dispatch_preprocess + post + combine_preprocess + post)
  scales sub-linearly** and stays around 10–15% — cheap to keep, but
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
