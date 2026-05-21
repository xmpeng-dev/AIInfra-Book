# smoke — Tiny-Scale Boot Test

## Purpose

Cheap sanity check that the **first plan from PROJECTION** can:

- Boot the framework end-to-end (model build + dataloader + optimizer).
- Reach forward + backward + optimizer step without NaN.
- Stay within predicted memory ± 20%.
- Pass without spending baseline-level wall time.

SMOKE catches the majority of trivial bring-up bugs (wrong tokenizer
path, dtype mismatch, missing kernel, NCCL boot issue) before BASELINE
burns large amounts of GPU·h.

## Inputs

```yaml
candidate_plan:    state/initial_plans.yaml.plans[0]   # highest-tps plan
cluster_profile:   state/cluster_profile.yaml
target_vector:
```

## Skills consulted

- `constraints/oom.md` (pre-submit memory bound)
- `constraints/config.md` (parallelism legality)
- `profiling/preflight.md` (only if SMOKE fails — re-probe environment)

## Tools called

```python
constraint.check(plan, cluster_profile)
constraint.estimate_mem(plan)
submit.run(plan, scale={...steps:50..100, mode:'smoke'})
observe.snapshot(job_id)
constraint.diagnose_failure(snapshot|error)
```

## Smoke plan derivation

The plan submitted is **not** the candidate plan as-is. Derive a tiny
variant:

```python
smoke_plan = candidate_plan.copy()
smoke_plan.model.layers      = min(candidate_plan.model.layers, 4)
smoke_plan.runtime.mbs       = min(candidate_plan.runtime.mbs, 1)
smoke_plan.runtime.seq_len   = min(candidate_plan.runtime.seq_len, 2048)
smoke_plan.runtime.recompute = 'full'           # safest mem
smoke_plan.runtime.steps     = smoke_steps      # see Scale-aware notes

# Keep parallelism (tp/pp/dp/ep) IDENTICAL — that's what we want to validate.
```

Reason: confirm parallelism + framework wiring works, not full-shape
numerics. Numerics are CORRECTNESS's job.

## Procedure

1. `constraint.check(smoke_plan)` and `constraint.estimate_mem(smoke_plan)`.
   Reject before submit if either fails.
2. `submit.run(smoke_plan, scale={..., steps:smoke_steps, mode:'smoke'})`.
3. Poll until finished or wall timeout (default 10 min single-node, 20
   min multi-node).
4. `observe.snapshot(job_id)`.
5. Apply pass / fail rules below.

## Pass / fail rules

```python
pass = (
    snap.status == 'ok'
    and snap.completed_steps >= smoke_steps
    and not snap.metrics.has_nan
    and snap.metrics.mem_peak_gb <= 1.2 * predicted.mem_gb
    and snap.exit_code == 0
)
```

Soft warnings (do NOT fail SMOKE, but record them):

| Signal | Meaning | Action |
|--------|---------|--------|
| `nccl_init_time_s > 30` | RCCL slow init | Note in `state/smoke_result.yaml.warnings`; let EnvSweep pick up later |
| `tps < 0.5 × predicted.tps` | very off prediction | Note; PROJECTION calibration may be poor — flag for LEARN |
| `cpu_wait_time_pct > 10` | dataloader stall on tiny seq | Usually disappears at full seq_len; note only |

## State written

```yaml
state/smoke_result.yaml:
  smoke_plan_id: r0_smoke
  source_plan_id: r0_p_init_1
  status: pass | fail
  metrics:
    mem_peak_gb: 64
    tps: 9000              # tiny-scale, not comparable to baseline
    completed_steps: 50
    wall_time_s: 142
  warnings: [...]
  failure: null | {kind: ..., evidence: [...]}
```

## Exit conditions

- **pass** → BASELINE.
- **fail** → On_fail.

## On_fail

| Failure kind | Transition |
|--------------|------------|
| `OOM` even at 4 layers + mbs=1 + recompute=full | `to: ABORT` `kind: STRUCTURAL_INVALIDATION` (this plan can never run; PROJECTION must be re-derived with stricter memory bias) |
| `HANG` | `to: PREFLIGHT` (env_probe subset) |
| `INVALID_CONFIG` (parallelism rejected by framework but accepted by `constraint.check`) | log inconsistency to `knowledge/anti-patterns.md`; `to: PROJECTION` to pick the next plan |
| `NUMERICAL` (NaN in tiny run) | `to: ABORT` `kind: NUMERICAL`; do NOT auto-recover |
| `framework_import_error` / `tokenizer_not_found` etc. | `to: ABORT` `kind: UNKNOWN`, escalate; user-environment issue |

## Cost budget

Charged to `setup_cost`:

- Single-node: ~10 min wall, ≤ ~1.5 GPU·h.
- Multi-node: ~20 min wall, scales with node count.

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| `smoke_steps` default | 50 | 100 (also exercise inter-node RCCL warmup) |
| Wall timeout | 10 min | 20 min |
| What SMOKE catches | Framework wiring, OOM, numeric | + RCCL boot, IB topology, GID, cross-node NaN |
| Recommended `seq_len` | 2048 | 4096 (catches msg-size dependent bugs) |
| GPUs used | All on the node (mirror real topology) | All requested nodes (mirror real topology) |

In both cases, never run SMOKE on fewer GPUs than the candidate plan
will use — the goal is to validate the parallelism wiring, not save
cost.
