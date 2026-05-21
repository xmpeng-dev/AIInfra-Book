# axis_taxonomy — Axis Classification

## Purpose

Classify every tunable knob into one of four **types**. The type
governs:

1. Which **execution strategy** Re-Plan uses (Champion-Challenger /
   Per-Plan / Halving — see `execution_strategy.md`).
2. Whether a discovery is **propagated** into the cluster baseline
   (`cluster_shared`) or kept per-plan (`weakly_local`,
   `strongly_local`).
3. Where the value lives (`ClusterProfile.env_baseline` vs
   `Plan.env.diff`).
4. Whether the axis can be searched in the inner EnvSweep loop or
   must be done structurally.

## The four types

```
structural        Changes the model / parallelism / memory / numerical layout.
                  → recompute=full | mbs↑ | tp↑ | pp↑ | ep↑ | vpp↑ | seq_len↑
                  Lives in Plan.parallelism / Plan.runtime.
                  Must be tested by full submit.run; OOM risk; correctness risk.

cluster_shared    Affects all jobs on the cluster identically. Independent of
                  model shape. Worth promoting to ClusterProfile.env_baseline
                  once validated.
                  → NCCL_IB_HCA, NCCL_NET_GDR_LEVEL, NCCL_SOCKET_IFNAME,
                    HSA_FORCE_FINE_GRAIN_PCIE, GPU_MAX_HW_QUEUES,
                    OMP_NUM_THREADS, NUMA pinning policy
                  Lives in ClusterProfile.env_baseline.{rccl,hsa,threading}.

weakly_local      Optimal value depends moderately on model shape (mbs,
                  world_size, msg_size). Not safe to globalize but cheap to
                  re-tune per (model, scale).
                  → NCCL_BUFFSIZE, NCCL_MIN_NCHANNELS, NCCL_ALGO,
                    PYTORCH_HIP_ALLOC_CONF (most variants), MSCCL_ENABLE
                  Lives in Plan.env.diff. Cached per (model_family, scale).

strongly_local    Optimal value depends sharply on this exact plan's structure
                  (specific layer count, mbs, tp×pp). Re-tune every time
                  structure changes.
                  → bucket_size_mb, recompute_pattern.attn_only_n,
                    PYTORCH_HIP_ALLOC_CONF.max_split_size_mb (when sized to
                    actual activation distribution), expert dispatch chunk size
                  Lives in Plan.env.diff or Plan.comm. Never promoted.
```

## Taxonomy table (canonical)

For every axis Pilot may tune, this table assigns a type and points
to the catalog file with the full definition. Add new axes here, not
elsewhere.

```yaml
# Structural (Plan.parallelism / Plan.runtime / Plan.comm)
- {axis: parallelism.tp,             type: structural,     catalog: optimization/compute/parallel.md}
- {axis: parallelism.pp,             type: structural,     catalog: optimization/pipeline/SKILL.md}
- {axis: parallelism.dp,             type: structural,     catalog: optimization/compute/parallel.md}
- {axis: parallelism.ep,             type: structural,     catalog: optimization/moe/SKILL.md}
- {axis: parallelism.vpp,            type: structural,     catalog: optimization/pipeline/vpp.md}
- {axis: parallelism.cp,             type: structural,     catalog: optimization/compute/parallel.md}
- {axis: runtime.mbs,                type: structural,     catalog: optimization/compute/mbs.md}
- {axis: runtime.gbs,                type: structural,     catalog: optimization/compute/mbs.md}
- {axis: runtime.recompute,          type: structural,     catalog: optimization/memory/recompute.md}
- {axis: runtime.recompute_pattern,  type: structural,     catalog: optimization/memory/recompute.md}
- {axis: runtime.dtype,              type: structural,     catalog: optimization/compute/parallel.md}
- {axis: comm.bucket_size_mb,        type: strongly_local, catalog: optimization/comm/bucket.md}
- {axis: comm.overlap,               type: structural,     catalog: optimization/comm/overlap.md}

# RCCL / NCCL family (env)
- {axis: NCCL_IB_HCA,           type: cluster_shared,  catalog: env/rccl.md}
- {axis: NCCL_NET_GDR_LEVEL,    type: cluster_shared,  catalog: env/rccl.md}
- {axis: NCCL_SOCKET_IFNAME,    type: cluster_shared,  catalog: env/rccl.md}
- {axis: NCCL_IB_GID_INDEX,     type: cluster_shared,  catalog: env/rccl.md}
- {axis: NCCL_BUFFSIZE,         type: weakly_local,    catalog: env/rccl.md}
- {axis: NCCL_MIN_NCHANNELS,    type: weakly_local,    catalog: env/rccl.md}
- {axis: NCCL_ALGO,             type: weakly_local,    catalog: env/rccl.md}
- {axis: NCCL_PROTO,            type: weakly_local,    catalog: env/rccl.md}
- {axis: RCCL_MSCCL_ENABLE,     type: weakly_local,    catalog: env/rccl.md}

# HSA / HIP family (env)
- {axis: HSA_FORCE_FINE_GRAIN_PCIE,  type: cluster_shared, catalog: env/hsa.md}
- {axis: GPU_MAX_HW_QUEUES,          type: cluster_shared, catalog: env/hsa.md}
- {axis: HIP_FORCE_DEV_KERNARG,      type: weakly_local,   catalog: env/hsa.md}

# Allocator (env)
- {axis: PYTORCH_HIP_ALLOC_CONF,                 type: weakly_local,   catalog: env/alloc.md}
- {axis: PYTORCH_HIP_ALLOC_CONF.max_split_size_mb, type: strongly_local, catalog: env/alloc.md}

# Threading / NUMA (env)
- {axis: OMP_NUM_THREADS,            type: cluster_shared, catalog: env/threading.md}
- {axis: MKL_NUM_THREADS,            type: cluster_shared, catalog: env/threading.md}
- {axis: numactl.cpunodebind,        type: cluster_shared, catalog: env/threading.md}
```

When in doubt, **default to `strongly_local`**: it's the safest
classification (never auto-promoted, always per-plan).

## Strategy mapping

```yaml
suggested_strategy_by_dominant_type:
  structural:        Champion-Challenger | Per-Plan
  cluster_shared:    Champion-Challenger    # propagate winner to baseline
  weakly_local:      Champion-Challenger | Per-Plan Local Sweep (in EnvSweep)
  strongly_local:    Per-Plan Local Sweep   # cheap, runs in EnvSweep
mixed:               Successive Halving when budget tight
```

`replan.md` Step 6 reads `DiagnosisReport.candidate_axes[*].type` and
this mapping to set `selected_strategy`.

## Promotion rules (which discovery moves to baseline)

| Type | Promote to ClusterProfile.env_baseline? | When |
|------|-----------------------------------------|------|
| structural | N/A (lives in Plan, not env) | — |
| cluster_shared | YES, with version bump | After winner survives 2 sessions on same cluster (LEARN's job; not auto). |
| weakly_local | NO; may be cached in `knowledge/cases.md` | LEARN records as `(model_family, scale) → value` mapping. |
| strongly_local | NO | Stays in Plan.env.diff; LEARN records derivation pattern only, not value. |

## Validity / co-tuning hints

Some axes only matter together; record co-tuning hints next to the
catalog entry, not here. Examples (informational):

- `NCCL_BUFFSIZE` only useful when `msg_size_p95 ≥ BUFFSIZE`.
- `comm.bucket_size_mb` interacts with `NCCL_MIN_NCHANNELS`.
- `runtime.recompute_pattern` only when `runtime.recompute = selective`.
- MoE: `parallelism.ep` vs `RCCL_MSCCL_ENABLE` vs alltoall topology.

`replan.md` Step 5 (constraint check) calls `constraint.check_env(...)`
which encodes incompatible combinations from `constraints/env.md`.

## Adding a new axis

1. Append a row to the taxonomy table above.
2. If env, add a full entry to the matching `env/<area>.md` catalog
   (default value, range, known caveats, references).
3. If structural, add or update the corresponding
   `optimization/<bottleneck>/<axis>.md` recipe.
4. If `constraint.check_env` should reject some combination, add to
   `constraints/env.md`.

## Scale-aware notes

The taxonomy is scale-independent — the *types* don't change. What
changes is the **set of axes worth touching** at each scale:

- **Single node**: ignore all `NCCL_IB_*` and `NCCL_SOCKET_IFNAME`
  (no inter-node path). `OMP_NUM_THREADS` and
  `PYTORCH_HIP_ALLOC_CONF` are the most active env axes.
- **Multi node**: full RCCL family becomes relevant; `NCCL_BUFFSIZE`
  and `NCCL_MIN_NCHANNELS` move from "rare tuning" to "default
  EnvSweep candidates".
