# optimization/compute — Recipes for COMPUTE_BOUND

## Purpose

Diagnose-recommended entry when `bottleneck = COMPUTE_BOUND`. Lists
the levers in priority order; sub-files (stubbed) hold per-axis
recipes filled on demand.

## When invoked

`workflow/diagnose.md` sets:

```yaml
bottleneck: COMPUTE_BOUND
recommended_skills:
  - skills/pilot/optimization/compute/SKILL.md
  - skills/pilot/optimization/compute/<sub>.md      # see table
```

`workflow/replan.md` consumes this file's priority table to generate
candidates.

## Lever priority

| # | Axis | Type (axis_taxonomy) | Sub-file | Typical gain | Risk |
|---|------|---------------------|----------|--------------|------|
| 1 | `runtime.mbs ↑` | structural | `mbs.md` | 3-12% | OOM |
| 2 | `runtime.recompute` selective→none | structural | `../memory/recompute.md` | 5-15% | OOM |
| 3 | `parallelism.tp ↓` (e.g. 8→4) | structural | `parallel.md` | 4-10% | mem ↑, may OOM |
| 4 | `parallelism.vpp 1→2` (when `pp > 1`) | structural | `../pipeline/vpp.md` | 2-5% | bubble change |
| 5 | kernel-level hints (fused attn, hipBLASLt prefer) | weakly_local | `kernel.md` + `../../env/hsa.md` | 3-8% | numeric |
| 6 | `runtime.dtype` (bf16→fp8) | structural | `../../env/SKILL.md` (out of scope here) | 15-30% | numeric, big |
| 7 | env compute-side knobs (`OMP_NUM_THREADS`, NUMA pinning) | cluster_shared | `env.md` → `../../env/threading.md` | 1-4% | low |

## Cross-links

- "Why was this COMPUTE_BOUND?" → `workflow/diagnose.md`
- Predictions → `execution-model/compute.md`
- MoE compute issues → `optimization/moe/SKILL.md`
- Env flags for compute → `env.md` (this folder) → `env/threading.md`,
  `env/hsa.md`

## Stubbed sub-files

| File | Status |
|------|--------|
| `mbs.md` | stub |
| `parallel.md` | stub |
| `kernel.md` | stub |
| `env.md` | stub (one-liner per env axis with link) |

Stubs follow the `optimization/SKILL.md` recipe template; filled when
Re-Plan first emits a candidate on that axis. Until then, defaults from
`axis_taxonomy.md` registry are sufficient.

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Common COMPUTE_BOUND cause | low utilization due to small mbs / small kernels | data-load stall or under-saturated dp |
| Most-effective lever | `mbs ↑` then `recompute → none` (mem allowing) | `mbs ↑` then `parallelism.dp` rebalance |
| Lever 5 (kernel hints) impact | high (intra-node) | moderate (other bottlenecks usually first) |
| `tp ↓` viability | constrained by single-node HBM cap | wider headroom across nodes |
