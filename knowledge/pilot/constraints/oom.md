# constraints/oom — Memory Feasibility Rules

## Purpose

Reject plans that won't fit in HBM **before** wasting wall-time. Owns:

- `constraint.estimate_mem(plan)` semantics.
- The safety margins (e.g. 92% rule).
- Per-component memory model (consumed by `execution-model/memory.md`,
  re-stated here from a constraint POV).

## Tool contract

```python
constraint.estimate_mem(plan) -> {
    mem_gb: float,                   # peak per-GPU
    breakdown: {
        param: float,
        grad:  float,
        optim: float,
        act:   float,
        workspace: float,
        frag_padding: float,
    },
    confidence: float,               # 0..1
}
```

Used by:
- `projection.md` to filter initial plans.
- `replan.md` Step 3 to filter candidates.
- `smoke.md` / `baseline.md` / `execute.md` defensively before submit.

## Safety margins

```yaml
hbm_capacity_gb: cluster_profile.hbm_capacity_gb       # per-GPU HBM

reject_if:                                # pre-submit
  estimate.mem_gb > 0.92 × hbm_capacity_gb

warn_if:                                  # accept but flag
  0.85 × hbm_capacity_gb < estimate.mem_gb <= 0.92 × hbm_capacity_gb

low_confidence_extra_margin:              # if confidence < 0.7
  effective_cap = 0.88 × hbm_capacity_gb  # tighten the bar
```

The 92% rule reserves ~8% for:
- allocator fragmentation spikes not captured in `frag_padding`,
- comm buffers grown on first allreduce / alltoall,
- transient kernel workspace peaks above the modeled average.

`hbm_capacity_gb` is read per-GPU from `ClusterProfile`; the rule is
identical regardless of GPU model.

## Memory model (per GPU)

```
M_total =  M_param / shard_factor_param
        +  M_grad  / shard_factor_grad
        +  M_optim / shard_factor_optim
        +  M_act
        +  M_workspace
        +  M_frag_padding

shard_factor_param  = tp × ep_for_experts × pp_param_shard_count
shard_factor_grad   = same as param (typical)
shard_factor_optim  = same; with ZeRO it's also × dp (dp=1 here)

M_act = M_act_per_layer × layers_on_this_pp_stage × mbs_factor(recompute)
  mbs_factor:
    none      = mbs · seq_len · hidden · k_const
    selective = mbs · seq_len · hidden · k_const × 0.4   # checkpoint reduces
    full      = mbs · seq_len · hidden · k_const × 0.18  # reduces further

M_workspace      vendor-dependent (RCCL/NCCL buffers, kernel scratch);
                 calibrated per cluster at PROJECTION; typical 4-10 GB
M_frag_padding   frag_factor × (M_param + M_grad + M_optim + M_act)
                 frag_factor calibrated at PROJECTION (0.05-0.25 typical)
```

For MoE:

```
M_param_expert = num_experts × M_per_expert / ep
M_param_shared = M_dense_part / tp
M_param        = M_param_expert + M_param_shared
```

(All dimensional details and coefficients live in
`execution-model/memory.md`; this file uses them as-is.)

## Confidence calculation

```python
confidence = (
    0.5 * R²_of_M_act_fit                       # from PROJECTION
    + 0.3 * (1 - extrapolation_distance)        # mbs / recompute distance from profiled grid
    + 0.2 * (1 - frag_factor / 0.25)            # higher frag → lower trust
)
clip(confidence, 0.0, 1.0)
```

Used by `replan.md` priority and by safety margin tightening.

## Worked rejection example (illustrative)

```yaml
plan: {tp:8, pp:1, ep:8, mbs:4, recompute:selective, ...}
cluster: {hbm_capacity_gb: 256}
estimate:
  mem_gb: 248
  breakdown:
    param: 96
    grad:  96
    optim: 38
    act:   18
    workspace: 6
    frag_padding: 18
  confidence: 0.74

cap = 256 × 0.92 = 235.5
violation:
  rule: "estimate.mem_gb > 0.92 × hbm_capacity"
  observed: 248 > 235.5
  hint:
    - "biggest term is grad+param=192GB; tp↓ won't help (already 8)"
    - "ep retune: try ep=4 (mem ↑) only if you accept comm ↑"
    - "recompute=full saves more on act+frag than mbs↓"
    - "or mbs↓ to 3 for ~10GB act savings"
```

This output is consumed by `replan.md`; the `hint` field can shape the
next round's candidate axes.

## On failure (post-submit OOM detected via observe.snapshot)

```python
constraint.diagnose_failure(snapshot_with_oom) -> FailureReport(
  kind: OOM,
  evidence: ["snap.status=oom at step {s}, mem_peak={observed}GB"],
  suggested_transition: {to: REPLAN, hints: {force_recompute_full: True if recompute != full else None}},
  counts_against_budget: false,
)
```

ExecutionModel feedback (separate from FailureReport):

```python
execution_model.feedback(
  axis_key='M_act',
  observed=observed_act_gb,
  predicted=estimated_act_gb,
  bias=+0.10 if observed > predicted * 1.10 else None
)
```

Future estimates for similar plans get the bias applied.

## Anti-patterns to flag

Anti-patterns are derived from `knowledge/anti-patterns.md` + this
session's failures; the catalog grows over time, not enumerated here.
Examples (illustrative):

- `(MoE-class, tp=high, ep=high, mbs≥2, recompute=selective)` →
  OOM-prone. Auto-reject + record after first observed.
- `(Dense≥30B-class, recompute=none, mbs≥3)` → borderline; prefer
  warn over reject unless `confidence < 0.6`.

## Reference / tiny-run exclusion

Do NOT apply this file's logic to plans with `mode='reference'`
(single-GPU tiny correctness ref runs); they have a different
parameter shape and the cap doesn't apply.

## Cross-links

- "How do I free memory?" → `optimization/memory/SKILL.md`
- "Where do these formulas come from?" → `execution-model/memory.md`
- Env-side mem effects → `env/alloc.md`
- "Predicted mem was way off" → LEARN's drift handling →
  `workflow/learn.md`

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| HBM pool | one per GPU; intra-node parallelism only spills via xGMI | shardable across nodes via `dp` / ZeRO; more headroom available |
| `M_workspace` | RCCL intra-node small | RCCL/NCCL inter-node may add 1-2 GB extra |
| `frag_padding` typical | 0.05-0.20 | 0.05-0.25 (more variability across nodes) |
| Common rejection | high `mbs × recompute` combinations | wide `tp × pp` with insufficient sharding of optimizer states |
| Workspace warn-threshold | 84-90% (kernel scratch can spike on first call) | same; plus IB connection setup adds transient memory |
