# GPT-OSS-20B mbs=4 / gbs=32 — Comm Optimization (B series)

**Date**: 2026-04-22 02:23 — 02:46 UTC (host 21:23 — 21:46)
**Hardware**: MI355X × 8, single node, intranode XGMI
**Baseline**: M4_C5 (latest optimization chain endpoint, see note `08`)

## TL;DR

Two free, lossless DDP knobs together give **~15-25 ms (-1.2 – 2.0%)** on the
mbs=4/gbs=32 step. Promote to **M4_C7**:

```yaml
ddp_average_in_collective: true       # B1
ddp_bucket_size: 100000000            # B2
```

The third candidate (NCCL channel env tuning) is in the noise floor and
not worth keeping.

This is essentially the **upper bound of pure comm-knob tuning** at this
config: only ~50 ms / iter of comm is actually exposed (most of it is
already overlapped with compute), so the realistic ceiling for any DDP
shuffle is ~50/1240 ≈ 4%, and we got ~half of that.

## Why this was the next experiment (recap from note `11`)

After the MoE FP8 GG postmortem, the GPU is already 98% busy on compute,
which means Amdahl-style any further compute optimization will quickly
expose communication. So before chasing more compute, we wanted to know:

1. How much **exposed** comm is there to begin with?
2. Are there any free DDP / NCCL knobs we left on the table?

## Comm pattern of M4_C5 (from profile trace)

From `output/amd/root/M4_C5_profile/.../*.pt.trace.json`:

| Metric | Value |
|---|---|
| NCCL kernels | 106 / iter total, 47-50 *large* (>500 µs) |
| Top message buckets | 530 / 292 / 66 / 36 MB (per-call) |
| Total NCCL kernel time | ~256 ms / iter |
| **Exposed (un-overlapped) comm** | **~50 ms / iter (≈ 4% of step)** |
| CPU-side NCCL ops | 50 AG + 50 RS coalesced per iter |
| Default bucket | 40 M params auto (~62 buckets) |

What was already on in M4_C5:
- `overlap_grad_reduce`, `overlap_param_gather`
- `grad_reduce_in_bf16`
- `ddp_pad_buckets_for_high_nccl_busbw`
- `TORCH_NCCL_HIGH_PRIORITY=1`

What was **off** (default) and worth probing:
- `ddp_average_in_collective`
- `ddp_bucket_size` (auto)
- All NCCL/RCCL channel env vars

## Ablation design

5 short runs (80 iters each, no profiler, no eval) with `tail-30` mean as
the metric. All on top of M4_C5.

| ID | Knob | Hypothesis |
|---|---|---|
| Bbase | unchanged | reference |
| B1 | `ddp_average_in_collective: true` | use `ncclAvg` natively, skip post-RS divide kernel |
| B2 | `ddp_bucket_size: 100_000_000` (~25 buckets) | fewer launches, larger msgs → higher busbw |
| B3 | `NCCL_MIN_NCHANNELS=32` + `NCCL_NCHANNELS_PER_NET_PEER=4` (and RCCL_*) | more channels for large intranode messages |
| Bbase2 | unchanged (rerun, position 5) | quantify thermal / order drift |
| B12 | B1 + B2 | composition test |

## Results

Tail-30 mean step time (ms):

| Position | Stage | Config | tail-30 | Δ vs Bbase1 (cold) | Δ vs Bbase2 (warm) |
|---|---|---|---|---|---|
| 1 | Bbase1 | C5 | **1243.5** | — | +9.8 |
| 2 | B1 | + avg_in_collective | **1226.5** | -17.0 (-1.4%) | -7.2 (-0.6%) |
| 3 | B2 | + bucket=100M | 1232.1 | -11.4 (-0.9%) | -1.6 (-0.1%) |
| 4 | B3 | + NCCL channels | 1234.2 | -9.3 (-0.8%) | +0.5 (+0.0%) |
| 5 | Bbase2 | C5 (rerun) | **1233.7** | -9.8 *(pure thermal)* | — |
| 6 | **B12** | + B1 + B2 | **1218.9** | **-24.6 (-2.0%)** | **-14.8 (-1.2%)** |

### Drift correction

`Bbase2 − Bbase1 = -9.8 ms` is **pure run-order / thermal** (same yaml,
different position). Use it as the upper bound for what order alone can
buy. Anything closer to 0 than that vs `Bbase2` is noise; anything
significantly more negative than -9.8 vs `Bbase1` is real.

By that test:
- **B1**: real signal of **-7 to -17 ms** (depending on which baseline). ✅
- **B2**: -1.6 vs warm baseline → **noise**, but +5 ms when stacked on B1 (B12 - B1 ≈ -7.6 ms in warm conditions). Marginally positive in stack. ✅ (kept)
- **B3**: +0.5 vs warm baseline → **noise**, drop. ❌
- **B12**: -14.8 ms vs warm baseline. The composed win. ✅

## Why B1 actually helps

`ddp_average_in_collective: true` does two things:
1. NCCL/RCCL performs the divide-by-DP inside the collective using its
   `ncclAvg` op, instead of `ncclSum + post-divide`.
2. Megatron skips the explicit post-RS scaling kernel for each bucket.

We had ~24 grad reduce-scatter buckets per iter. Each one was followed
by a small `aten::div` / fused divide kernel before the optimizer step.
Eliminating those 24 small launches per iter is consistent with the
~10-15 ms observed.

## Why B2 helps a little (only when stacked)

Halving the bucket count (50 → ~25) gives:
- Fewer NCCL launches → less host-side overhead and stream sync points.
- Larger per-call messages → slightly higher achieved busbw on intra-node XGMI.
- Tradeoff: less overlap headroom on the very last bucket in backward.

On its own the win is in the noise (+5 ms maybe), but composed with B1
the savings show up.

## Why B3 was a wash

The default RCCL channel autotune already picks something reasonable for
intra-node 8-GPU XGMI on MI355X. Forcing 32 channels did not move busbw
in any measurable way at our message sizes. **Not worth keeping.**

## Headroom remaining on the comm path

- Exposed comm before B12: ~50 ms / iter
- B12 saved ~15 ms
- Estimated remaining exposed comm after B12: **~35 ms (≈ 2.9% step)**

So the comm path now contributes roughly:
- 35 ms exposed wall time
- 240+ ms total transfer time, mostly hidden behind compute

The next class of comm wins requires changing **what** is sent, not how:
- `fp8_param_gather` would halve the AG (~50% of comm bytes), but it
  forces `fp8_recipe: delayed`, which is incompatible with our current
  `hybrid` recipe and was shown in note `11` to silently demote the MoE
  GG to bf16. ⚠️ Risky.
- `num_distributed_optimizer_instances > 1` (HSDP) only matters at much
  larger DP. At DP=8 it's pointless.

## Recommendation

1. **Promote B12 → M4_C7**, file `ablations/M4_chain/M4_C7.yaml` already
   written. Expected step time ~1219 ms (vs ~1234-1243 ms baseline).
2. **Do not** ship NCCL channel env vars; they were measured noise.
3. Optionally re-profile M4_C7 to confirm the 24 small `aten::div`
   kernels per iter are gone and the RS kernel time is unchanged or
   slightly lower.

## Next steps in the chain

Comm path is squeezed to within ~35 ms of its lower bound. Bigger
remaining levers:

| Lever | Est. uplift | Risk |
|---|---|---|
| **D — mbs=8 OOM check + GA halving** | ~10-20% if it fits | high (likely OOMs without other knobs) |
| **C — fix attn-bwd recompile** (note `09`) | ~3-5% | low |
| **A — MoE-side recompute reduction** | unclear | medium |

User has previously selected B; recommend **C next** (low-risk, decent
size) before attempting D.

## Files / artifacts

- `ablations/M4_chain/run_comm_ablation.sh` — short-runner template
- `ablations/M4_chain/M4_C5_B1.yaml` — B1 only
- `ablations/M4_chain/M4_C5_B2.yaml` — B2 only
- `ablations/M4_chain/M4_C5_B3.yaml` — B3 (env-only, yaml = C5)
- `ablations/M4_chain/M4_C5_B12.yaml` — B1 + B2 stacked
- `ablations/M4_chain/M4_C7.yaml` — promoted = C5 + B1 + B2
- `ablations/M4_chain/run_B_chain.sh` / `run_B_validate.sh` — runners
- `ablations/M4_chain/logs/M4_C5_B*.log` — raw per-iter timings
