# Turbo MoE forward-pass breakdown — MI355X EP8

Forward-only latency breakdown of the Primus-turbo MoE pipeline
(dispatch + grouped fc1 + swiglu + grouped fc2 + combine), measured on a
single MI355X node with EP=8 (1 node × 8 GPUs, no TP). The numbers below
correspond to **one transformer layer's MoE submodule, one micro-batch,
forward only**, with the routing table that Megatron's
`moe_router_force_load_balancing` produces.

All data was collected by `turbo/bench_turbo_moe_e2e.py` inside the
`xiaoming-dev` container on `mi355-gpu-26`. The wrapper that swept all
four configs is `turbo/run_all_models.sh`.

## Layout

```
moe_perf/
├── README.md                 # this file
└── turbo/                    # everything related to the Primus-turbo MoE pipeline
    ├── bench_turbo_moe_e2e.py
    ├── run_all_models.sh
    ├── bench-script-notes.md
    └── *.csv                 # see table below
```

## Files (all under `turbo/`)

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

The bench script `turbo/bench_turbo_moe_e2e.py` only depends on
`primus_turbo.pytorch` and `torch.distributed`; it has no relative
imports, so it runs as-is from any cwd as long as the container exposes
`primus_turbo`. See `turbo/bench-script-notes.md` for the per-stage
table the script prints and the CLI knobs it accepts.
