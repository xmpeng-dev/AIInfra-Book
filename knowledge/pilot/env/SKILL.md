# env — Environment Variable Catalog (Top Entry)

## Purpose

Single source of truth for every environment variable Pilot may tune.
Each flag is **fully defined exactly once** in the appropriate
catalog sub-file under this directory. Other skills (e.g.
`optimization/<bottleneck>/env.md`) only list "which flags to try
under this bottleneck" and link back here.

This rule prevents knowledge from scattering: changing a flag's
default or known-caveat list is a single-file edit.

## Catalog files

| File | Owns flags in family |
|------|---------------------|
| `rccl.md` | `NCCL_*` and `RCCL_*` (incl. MSCCL) |
| `hsa.md` | `HSA_*`, `HIP_*`, `GPU_MAX_HW_QUEUES` |
| `alloc.md` | `PYTORCH_HIP_ALLOC_CONF`, `MALLOC_*` |
| `threading.md` | `OMP_*`, `MKL_*`, `numactl` policies |
| `presets.md` | per-cluster-type validated combinations (read-only quick-starts) |

## Flag entry format

Each flag in a catalog file follows:

```markdown
## <FLAG_NAME>

### Default
   Cluster-baseline default (the value PREFLIGHT promotes if no
   override). One line.

### Range / valid values
   Discrete or continuous range; with units.

### Type
   cluster_shared | weakly_local | strongly_local
   (matches axis_taxonomy.md)

### When to tune
   Symptoms that suggest this flag (cross-link diagnose.md / a
   bottleneck SKILL).

### Co-tune with
   Other flags / structural axes that interact.

### Known caveats / dangers
   Failure modes, version dependencies, platform restrictions.

### Reference
   Documentation links, internal tickets, prior session learnings
   (knowledge/cases.md case_id).
```

When the Agent needs to evaluate a flag, it reads exactly one entry.

## How catalog interacts with the workflow

```
PREFLIGHT (env_probe.run)
  ├─ reads env/<area>.md to know which flags to probe (defaults)
  └─ writes ClusterProfile.env_baseline.<area>

DIAGNOSE
  └─ if env_suspect appears, references env/<area>.md#<anchor> in
     DiagnosisReport

REPLAN
  └─ for each candidate axis, reads env/<area>.md to bound `range`

ENVSWEEP
  ├─ reads env/<area>.md range_for(bottleneck, scale) for combos
  ├─ reads constraints/env.md for incompatibility matrix
  └─ writes EnvSweepResult; merges winner per axis_type rules

LEARN
  └─ for cluster_shared promotions, writes back into
     ClusterProfile.env_baseline (with version bump);
     for weakly_local/strongly_local, writes a case in
     knowledge/cases.md (per model_family + scale)
```

## Promotion rules (where env values eventually live)

Mirrors `axis_taxonomy.md`:

| Type | Lives in | Promoted when |
|------|----------|---------------|
| `cluster_shared` | `ClusterProfile.env_baseline.<area>` | Winner survives 2 sessions on same cluster (LEARN's job; not auto). |
| `weakly_local` | `Plan.env.diff` | Cached in `knowledge/cases.md` per `(model_family, scale)`. Never globalized. |
| `strongly_local` | `Plan.env.diff` | Stays per-plan; `knowledge/cases.md` records derivation pattern only. |

## Sub-file status

| File | Status |
|------|--------|
| `rccl.md` | stub (entry list) — fill from MI300X/MI355X canonical defaults on first use |
| `hsa.md` | stub |
| `alloc.md` | stub |
| `threading.md` | stub |
| `presets.md` | stub (one entry per validated cluster) |

## Cross-links

- Axis types → `workflow/axis_taxonomy.md`
- Probing protocol → `profiling/env_probe.md`
- Compatibility matrix → `constraints/env.md`
- Inner sweep mechanics → `workflow/envsweep.md`
- Bottleneck → flag mapping → `optimization/<bottleneck>/env.md`
- Promotion mechanics → `workflow/learn.md`

## Scale-aware notes

| Family | Single node | Multi node |
|--------|-------------|-----------|
| `rccl.md` | small subset matters (intra-node only) | full file relevant; IB/RoCE-related flags critical |
| `hsa.md` | full subset matters | full subset matters |
| `alloc.md` | full subset matters | full subset matters |
| `threading.md` | full subset matters; NUMA pinning a common win | full subset matters |
| `presets.md` | per-cluster preset includes only intra-node defaults | preset includes IB topology defaults too |
