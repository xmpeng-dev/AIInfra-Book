# optimization/memory — Recipes for MEMORY_BOUND

## Purpose

Diagnose-recommended entry when `bottleneck = MEMORY_BOUND`. This is
typically the most common bottleneck on small clusters (intra-node
HBM is the constraint) and a common one on large clusters too once
`mbs` and `recompute` start interacting with sharded states.

## When invoked

`workflow/diagnose.md` sets:

```yaml
bottleneck: MEMORY_BOUND
recommended_skills:
  - skills/pilot/optimization/memory/SKILL.md
  - skills/pilot/optimization/memory/<sub>.md
env_suspect:
  - {flag: PYTORCH_HIP_ALLOC_CONF, hint: env/alloc.md#expandable-segments}   # often co-emitted
```

## Lever priority

| # | Axis | Type | Sub-file | Effect | Cost / risk |
|---|------|------|----------|--------|-------------|
| 1 | `runtime.recompute` none→selective | structural | `recompute.md` | -30..50% act mem | -3..8% tps |
| 2 | `runtime.recompute` selective→full | structural | `recompute.md` | -50..70% act mem | -10..18% tps |
| 3 | `runtime.recompute_pattern` (which layers) | structural | `recompute.md` | -10..25% act mem | -2..5% tps |
| 4 | `env.PYTORCH_HIP_ALLOC_CONF` | weakly_local / strongly_local | `env.md` → `../../env/alloc.md` | -10..20% reserved | tiny tps |
| 5 | `runtime.mbs ↓` | structural | `../compute/mbs.md` | -linear act mem | -linear tps |
| 6 | `parallelism.tp ↑` | structural | `../compute/parallel.md` | -linear param/grad mem | comm ↑, tps ↓ |
| 7 | `parallelism.ep ↑` (MoE only) | structural | `../moe/SKILL.md` | -large expert mem | comm ↑ |
| 8 | `offload` (optimizer states / activations) | structural | `offload.md` | -large param/optim mem | comm + host bw bound |

## Apply order rationale

- Try **non-recompute levers first** when headroom is small (≤ 5%
  over cap): allocator config (4) is cheap and reversible.
- If that fails, escalate to `recompute` (1 → 2). Biggest single
  lever but hurts tps.
- Reserve `mbs↓` (5), `tp↑` (6), `ep↑` (7) for when nothing else
  fits — they hurt throughput predictably and may interact with
  comm.
- `offload` (8) is a last resort: only useful if `param + optim mem`
  alone exceed `0.7 × hbm_capacity`.

## Fragmentation special case

If `mem_reserved_to_alloc_ratio > 1.4` AND `mem_alloc < 0.85 ×
hbm_capacity`:

- The model fits in *allocated* memory; the *reserved* spike is
  causing OOM.
- Lever 4 (`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`)
  usually resolves this without giving up tps.
- Skip levers 1-3 in this case; recompute reduces allocated mem,
  not fragmentation.

This is a **distinct sub-mode** of MEMORY_BOUND; `diagnose.md` already
emits the env_suspect, but Re-Plan must respect the priority shift.

## Cross-links

- "Fragmentation vs true OOM?" → `workflow/diagnose.md` § fragmentation rule
- Predicted memory → `execution-model/memory.md`
- Pre-submit OOM rejection → `constraints/oom.md`
- Env-side fragmentation tuning → `env/alloc.md`
- MoE-specific memory recipes → `optimization/moe/SKILL.md`

## Stubbed sub-files

| File | Status |
|------|--------|
| `recompute.md` | stub |
| `offload.md` | stub |
| `fragmentation.md` | stub |
| `env.md` | stub (one-liner per env axis with link) |

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Most common cause | activation memory + workspace at high mbs | activation + sharded optim states (ZeRO-style) edges |
| Lever 4 (allocator) impact | high (intra-node frag is dominant) | moderate (sharding usually compresses live mem) |
| Lever 6 (`tp ↑`) cost | low (intra-node xGMI fast) | higher (cross-node TP comm is expensive — usually avoid) |
| Lever 7 (`ep ↑`) (MoE) | bounded by `gpus_per_node` typically | wider freedom; co-tune with `RCCL_MSCCL_ENABLE` (env hint) |
| Lever 8 (`offload`) viability | rare (PCIe is the bottleneck) | rare (NVMe / host bw bound); only when nothing else fits |
