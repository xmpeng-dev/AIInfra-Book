# execute — Submission + Early Stop

## Purpose

Submit the selected candidates from `CandidatePool` and collect raw
results. This stage is the only one (besides BASELINE / SMOKE) that
calls `submit.run`.

`execute.md` does NOT decide what to run (that's `replan.md`) or what
to promote (that's `settle.md`). Its job: run safely, stop early when
clearly bad, return clean snapshots.

## Inputs

```yaml
candidate_pool:    state/candidate_pools/r{N}.yaml      # selected[]
strategy:          state/candidate_pools/r{N}.yaml.selection.strategy
plan_graph:        state/plan_graph.yaml
target_vector:
cluster_profile:
```

## Skills consulted

- `observe.md` (Snapshot schema collected per job)
- `correctness.md` (lite mode follows immediately after each job
  finishes; conceptually the next stage but called per plan)
- `constraints/oom.md`, `constraints/config.md` (final pre-submit
  defensive check)

## Tools called

```python
constraint.check(plan, cluster_profile)
constraint.estimate_mem(plan)
submit.run(plan, scale={...steps:S, mode:'challenger'}) -> job_id
submit.cancel(job_id) -> status
observe.snapshot(job_id) -> Snapshot
constraint.diagnose_failure(snapshot|error) -> FailureReport
```

## Per-strategy procedure

### Champion-Challenger (parallel or serial)

```python
S = challenger_steps                           # 200 default; see execution_strategy.md
champion_tps = plan_graph.champion.metrics.tps

if scheduling == "parallel":
    job_ids = [submit.run(p, scale={steps:S, mode:'challenger'}) for p in selected]
    # poll all; early-stop independently
else:  # serial
    snapshots = []
    for p in selected:
        jid = submit.run(p, scale={steps:S, mode:'challenger'})
        snapshots.append(poll_with_early_stop(jid, champion_tps))
```

### Per-Plan Local Sweep

```python
for cand in selected:
    sweep_id = env_probe.sweep(
        base_plan=cand,
        candidate_envs={cand.strongly_local_axis: cand.sweep_values},
        max_steps=30,
    )
    cand.env.diff[axis] = sweep_id.best_env_diff
    # then run cand at full step budget (Champion-Challenger style)
```

### Successive Halving

```python
rung_steps = [50, 150, S]                      # see execution_strategy.md
pool = selected
for steps in rung_steps:
    job_ids = [submit.run(p, scale={steps:steps, mode:'challenger'}) for p in pool]
    snaps   = [poll_with_early_stop(jid, champion_tps) for jid in job_ids]
    pool    = top_half_by_tps([s for s in snaps if s.status == 'ok'])
final = pool[0]
```

## Early-stop rules (`poll_with_early_stop`)

Apply in order, first match wins:

| Rule | Condition (after `min_steps` measured) | Action |
|------|---------------------------------------|--------|
| OOM imminent | `mem_peak > 0.97 × hbm_capacity` | `submit.cancel`; `kind: OOM` |
| HANG | no progress for `hang_timeout_s` | `submit.cancel`; `kind: HANG` |
| TPS below champion | `tps_ema_50 < champion.tps × 0.85` AND `steps ≥ 50` | `submit.cancel`; `early_stop_reason: tps_below_champion_85pct` |
| TPS far below prediction | `tps_ema_50 < predicted.tps × 0.70` AND `steps ≥ 50` | `submit.cancel`; `early_stop_reason: tps_below_prediction_70pct` |
| Loss drift early | lite-correctness shows `delta_pct > 5%` at step 100 | `submit.cancel`; `early_stop_reason: loss_drift_early` |
| Numerical (NaN/inf) | `has_nan` OR `has_inf` ever | `submit.cancel`; `kind: NUMERICAL` |
| OOM (actual) | OOM raised by framework | snapshot collected; `kind: OOM` |

`hang_timeout_s` defaults: 300 single-node, 600 multi-node.

`min_steps` = 20 (do not early-stop during framework warmup).

Early-stopped jobs still produce a Snapshot (with reduced `metrics`
confidence). They do **not** count against round budget when
`kind in {OOM, HANG, NUMERICAL}` (per state_machine.md). They DO count
when `early_stop_reason` is `tps_below_*` (we got real signal).

## Defensive pre-submit check

Even though `replan.md` already validated, redo:

```python
for cand in selected:
    assert constraint.check(cand, cluster).valid
    assert constraint.estimate_mem(cand).mem_gb <= 0.92 * cluster.hbm_capacity_gb
    assert constraint.check_env(cand.env.diff, baseline).valid
```

Inconsistencies here indicate a bug in `replan.md` or stale state →
log to `knowledge/anti-patterns.md` and skip the candidate.

## Submit conventions

`submit.run(plan, scale={...})` returns immediately with a `job_id`.
The Agent polls via `observe.snapshot(job_id)` (which works on
running jobs too, returning partial Snapshot) and decides on
early-stop.

`scale` carries:

```yaml
scale:
  nodes:    <int>            # from plan derivation × ClusterProfile
  gpus:     <int>
  steps:    <int>            # the step budget
  mode:     challenger | smoke | baseline | reference | envsweep
  timeout_s: <int>           # hard wall; 2 × expected step time × steps
```

## Outputs

```yaml
state/snapshots/r{N}_p{i}.yaml: <Snapshot>          # one per executed candidate
state/plan_graph.yaml.nodes.r{N}_p{i}:
  status: completed | dead     # per-job FailureReport
  snapshot_ref: state/snapshots/r{N}_p{i}.yaml
  early_stop_reason: ...
```

`status: dead` is set here when:

- `kind in {OOM, NUMERICAL, INVALID_CONFIG}` after submit, OR
- Successive Halving culled this candidate.

`status: completed` is set here when the job finished or was
early-stopped without a `kind`. `settle.md` later reclassifies as
`shelved` / `champion`.

## Exit conditions

- **success**: every selected candidate has either `completed` or
  `dead` status; at least one `completed` Snapshot is available.
  → CORRECTNESS-LITE.
- **soft_fail**: all `dead`. → SETTLE with `decision: continue,
  no_promotable`. Settle decides whether stagnation logic kicks in.
- **hard_fail**: `kind: HANG` or `CLUSTER` triggered on a job → On_fail.

## On_fail

Per-job failures route through `state_machine.md` on_fail table
(driven by `FailureReport.kind`). Round-level rules:

| Round-level condition | Effect |
|----------------------|--------|
| Every job failed with same `kind: OOM` | log `oom_outbreak`; SETTLE will request `force_envsweep=false` and replan with stricter mem bias |
| Every job failed with same `kind: HANG` | route via state machine to `PREFLIGHT (env_probe subset)` immediately; do not pretend round happened |
| Mixed failures with one `kind: NUMERICAL` | `to: ABORT` (numerical always escalates) |

## Cost bookkeeping

Each candidate's actual GPU·h is recorded in its Snapshot
(`wall_time_s × gpus`). Aggregated per round:

```yaml
state/round_costs.yaml.r{N}:
  planned_gpu_h: 1.4         # sum of est_cost_gpu_h
  actual_gpu_h:  1.6
  variance_pct:  +14
  early_stops:   1           # count
  oom:           0
  hang:          0
```

## Cross-links

- Strategy choice → `execution_strategy.md`
- Snapshot schema → `observe.md`
- Lite-correctness → `correctness.md`
- Failure routing → `state_machine.md`
- Per-job promotion / demotion → `settle.md`

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Default scheduling | serial | parallel when `top_k × plan.gpus_required ≤ cluster.gpus_total` |
| `hang_timeout_s` | 300 | 600 (NCCL/IB warmup is slower) |
| `challenger_steps` typical | 200 | 200-300 (cross-node tps stabilizes slower) |
| Polling cadence | every 5 s | every 10 s |
| Early-stop on `tps_below_champion_85pct` | aggressive (intra-node tps stable fast) | defer to ≥ 80 steps (cross-node noise high) |
