# optimization/pipeline — Recipes for PIPELINE_BOUND

## Purpose

Diagnose-recommended entry when `bottleneck = PIPELINE_BOUND`. This
classifier triggers only when `parallelism.pp > 1` and
`bubble_ratio > 0.15`; thus this folder is **dormant on `pp = 1`**
runs.

## When invoked

```yaml
bottleneck: PIPELINE_BOUND
recommended_skills:
  - skills/pilot/optimization/pipeline/SKILL.md
  - skills/pilot/optimization/pipeline/<sub>.md
```

## Lever priority

| # | Axis | Type | Sub-file | Effect | Risk |
|---|------|------|----------|--------|------|
| 1 | `parallelism.vpp ↑` (1→2→4 interleaving) | structural | `vpp.md` | -bubble (formula: `(pp-1)/(pp-1+M·V)`) | mem ↑ slightly |
| 2 | `runtime.mbs / num_microbatch` retune (M ↑) | structural | `microbatch.md` | -bubble linearly with M | mem ↑ |
| 3 | Stage balance (layer partition) | structural | `balance.md` | varies; up to -50% bubble if previously imbalanced | framework-dependent |
| 4 | `parallelism.pp ↓` (e.g. 4→2) | structural | `../compute/parallel.md` | -bubble dramatically | mem ↑ (param/grad shard fewer) |
| 5 | 1F1B / interleaved schedule (framework option) | structural | `vpp.md` | -bubble + better mem use | framework support |

## Apply order rationale

- **Lever 1 (vpp)** is the textbook fix; cheap to test, large bubble
  reduction. Memory cost is small (extra micro-batch buffers).
- **Lever 2 (more microbatches)** = increase `gbs / mbs` ratio if
  `gbs` is fixed; this is essentially `mbs ↓` + same `gbs` →
  `num_microbatch ↑`. Test cost: full Champion-Challenger run.
- **Lever 3 (stage balance)** is framework-specific; consult
  `balance.md` for the framework adapter that exposes the layer
  partition.
- **Lever 4 (`pp ↓`)** is the structural escape hatch when nothing
  else helps; large impact, but rebalances all of compute/mem/comm
  — usually triggers re-Diagnose next round.

## Cross-links

- "What's the bubble formula?" → `execution-model/pipeline.md`
- Bottleneck classification → `workflow/diagnose.md` § PIPELINE_BOUND
- Predictions of mem after `vpp ↑` → `execution-model/memory.md`
  (vpp adds buffer per stage)
- Re-Diagnose after `pp ↓` (lever 4) is common; let `settle.md` handle
  the bottleneck shift naturally

## Stubbed sub-files

| File | Status |
|------|--------|
| `vpp.md` | stub |
| `microbatch.md` | stub |
| `balance.md` | stub |

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Triggering frequency | rare (`pp = 1` typical on single node) | common when `pp > 1` (large models split across nodes) |
| Lever 1 (vpp) impact | rarely needed | large; usually a 5-15% TPS lift |
| Lever 4 (`pp ↓`) viability | almost always (collapses to pp=1) | constrained by per-GPU memory at low pp |
| Bubble vs comm trade | comm is intra-node so cheap; bubble dominates | both can dominate; tune jointly |

## Note on `pp = 1` runs

If `parallelism.pp == 1`, `PIPELINE_BOUND` should never trigger
(`bubble_ratio = 0` by construction). If it does, that's a Diagnose
bug — log to `knowledge/anti-patterns.md` and route via the next
most-likely bottleneck.
