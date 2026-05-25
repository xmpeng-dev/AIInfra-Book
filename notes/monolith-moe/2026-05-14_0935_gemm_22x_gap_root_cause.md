# GEMM gap vs standalone grouped GEMM: **corrected accounting** (was inflated ~8×)

**date**: 2026-05-14 09:35 (UTC+8), **revised same day** after scale check  
**node**: mi355-gpu-26 (xiaoming-dev container) — re-run commands below on that host  
**hardware**: 8× MI355X (gfx950), 256 CU/GPU  
**workload**: DSV3 MoE, `T_src=8192` per rank, `topk=8`, `E=256`, `ws=8` → `epg=32`  
**tag**: cycle 15 diagnosis — **baseline row-count fixed**; “22×” headline retracted  

## What was wrong in the first write

1. **Wrong M on the standalone reference**  
   The table compared super-kernel inner time to `bench_grouped_gemm_t8k` runs with **`B=32, M=256`**, labeled “exact dispatched shape”.  
   For this training geometry, tokens **received on one rank** for its 32 local experts sum to `T_src × topk = 8192 × 8 = 65536` rows (each row is one routed token–expert activation).  
   Averaged over **32** local experts → **65536 / 32 = 2048** rows per expert on that rank, **not 256**.  
   `256` is the **per-(src, e)** chunk (65536 / 8 srcs / 32 experts), i.e. one shard before concatenating all srcs for the same expert — it is **not** the M of a full per-expert GEMM that already sees all srcs.

   GEMM FLOPs scale ~linearly in `M`. Using **`M=256` instead of `M=2048` understates work by ~8×**, so standalone time and implied “gap vs super-kernel” were **systematically bogus** (same order as the “8× MFMA row waste” story — the error and the geometry factor **must not be double-counted**).

2. **“PyTorch” was never measured there**  
   Sub-ms / hundreds-of-TFLOPS numbers came from **in-repo HIP** `grouped_gemm` / `grouped_gemm_persistent` (`csrc/grouped_gemm.hip`), not from `bench_pytorch_rccl_dsv3.py` (per-expert `torch.matmul` + real dispatch).  
   Do **not** equate that microbench with Megatron-style PyTorch+RCCL GEMM throughput.

## Correct row-count math (don’t mix “per shard” with “per expert”)

- Per rank, routed rows into the MoE layer: **`T_src × topk`** (here **65536**).  
- Per rank, rows handled by **one local expert** on average: **`(T_src × topk) / epg` = 2048**.  
- **`256`** = **`(T_src × topk) / (ws × epg)`** = per **(src rank, local expert)** average — useful for **dispatch / IPC tile** sizing, wrong as **`M` for a fused per-expert GEMM** that already aggregates all srcs.

Reference HIP bench should use **`M=2048`** (and optional `M=256` only as a **shard** microbench, clearly labeled).

## What stays true (geometry / algorithm)

Super-kernel still schedules GEMM at **per-(src, e)** granularity with **`M_TILE=256`** while valid rows per shard can be **32**: that is still **~87.5% masked MFMA rows per tile** on that shard — a real **algorithm / launch-shape** issue.

What **does not** follow without a **corrected** standalone row: the old **“22× vs grouped GEMM”** table. After fixing **`M`**, expect the HIP standalone reference for FC1/FC2 to land in the **low–mid single-digit ms** ballpark (order-of-magnitude), and the **ratio** vs super-kernel inner to shrink to a **small number ×**, not **20×**. Re-measure on hardware.

## How to re-run (on ROCm box with `hipcc`)

From repo root:

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx950 -Icsrc \
  -o /tmp/bench_grouped_gemm_t8k benchmarks/bench_grouped_gemm_t8k.hip
/tmp/bench_grouped_gemm_t8k
```

`bench_grouped_gemm_t8k.hip` prints, in order, **`M=2048`** (correct per-expert scale), then **`M=256`** (per-(src,e) shard context). Compare super-kernel FC buckets against the **`M=2048`** lines, not the `M=256` lines.

PyTorch+RCCL end-to-end / GEMM slice (8 GPUs):

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/bench_pytorch_rccl_dsv3.py \
  --tokens 8192 --topk 8 --hidden 7168 --ffn 2048 --num-experts 256 --warmup 5 --iters 30
```

Use **`gemm`** breakdown from that script as the **software** baseline expectation — it will be **much slower** than HIP `grouped_gemm` on a synthetic full-matrix bench.

## Optimization direction (unchanged intent, re-scoped numbers)

- **Cycle 16 (per-expert batched GEMM)** is still the right structural fix: **concatenate shards so each expert GEMM sees `M≈2048`**, aligning MFMA tiles with real row counts.  
- **Do not** justify it with a **22×** vs `M=256` microbench; justify it with **(a)** corrected HIP reference at **`M=2048`** and **(b)** masked-tile / utilization evidence inside the super-kernel.  
- After re-bench, update FC1/FC2 inner ms and wall projections from measured deltas, not from the retracted table.

## Retracted table (kept only as a warning label)

| | super-kernel inner (cycle 14b) | old standalone (`M=256`, wrong scale) | status |
|---|---|---|---|
| FC1 | 17.27 ms | 0.769 ms | **invalid compare** |
| FC2 | 11.14 ms | 0.538 ms | **invalid compare** |

**Conclusion:** the **“22×”** headline was an **artifact of mismatched GEMM M**, not a validated PyTorch or even validated HIP apples-to-apples gap. Re-run with **`M=2048`**, then re-evaluate how much of the remaining gap is **tile geometry** vs **occupancy / fusion / comm WG**.
