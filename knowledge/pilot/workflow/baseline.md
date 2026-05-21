# baseline — First Full-Scale Run

## Purpose

Establish the **reference Snapshot** that all tuning rounds compare
against. BASELINE is the first time the model runs at full shape and
real step count; its metrics seed the PlanGraph as `r0_p0` (initial
champion).

## Inputs

```yaml
candidate_plan:    state/initial_plans.yaml.plans[0]   # same plan SMOKE validated
cluster_profile:
target_vector:
smoke_result:      state/smoke_result.yaml             # must be 'pass'
```

## Skills consulted

- `constraints/oom.md`, `constraints/config.md` (final pre-submit check)
- `observe.md` (Snapshot schema this stage will produce)

## Tools called

```python
constraint.check(plan, cluster_profile)
constraint.estimate_mem(plan)
submit.run(plan, scale={...steps:S, mode:'baseline'})
observe.snapshot(job_id)
constraint.diagnose_failure(snapshot|error)
state.checkpoint(tuning_state)
```

## Step count `S`

```python
S = max(target_vector.budget.baseline_steps_min,            # default 200
        smoke_result.recommended_steps,                      # if SMOKE warned tps unstable
        steps_to_reach_warmup_done(plan))                    # framework warmup + LR ramp
```

Default `S = 500` (wider than challenger's 200; we want a trustworthy
reference).

## Procedure

1. Final `constraint.check` + `constraint.estimate_mem` — defensive
   verification (should never fail after SMOKE pass, but cheap).
2. `submit.run(plan, scale={..., steps:S, mode:'baseline'})`.
3. Poll until finished. Apply only **one** early-stop rule (very
   conservative — we want a clean reference):
   - OOM imminent (`mem_peak > 0.97 × hbm_capacity`) → cancel +
     `kind: OOM`.
4. `observe.snapshot(job_id)`.
5. Persist as `r0_p0` in PlanGraph; mark as champion.

## Pass / fail rules

```python
pass = (
    snap.status == 'ok'
    and snap.completed_steps >= max(200, S * 0.8)
    and not snap.metrics.has_nan
    and snap.metrics.tps > 0
)
```

BASELINE has no tps target to beat; whatever it returns *is* the bar.
CORRECTNESS (next stage) verifies the loss curve.

## State written

```yaml
state/snapshots/r0_p0.yaml: <full Snapshot>

state/plan_graph.yaml:
  champion: r0_p0
  champion_history:
    - {round: 0, plan_id: r0_p0, promoted_at: ts, by: baseline}
  nodes:
    r0_p0:
      parent: null
      derived_axis: null
      status: champion
      metrics: {tps, mem_gb, comm_ratio, bubble_ratio, ...}
      snapshot_ref: state/snapshots/r0_p0.yaml
      plan_ref:     state/initial_plans.yaml.plans[0]
  exhausted_neighborhoods: []
  metadata:
    rounds_since_promotion: 0
    rounds_since_explore:   0
    backtrack_count:        0

state/baseline_summary.yaml:
  plan_id: r0_p0
  steps_completed: 500
  wall_time_s: 1820
  tps: 16100
  mem_peak_gb: 178
  comm_ratio: 0.14
  bubble_ratio: 0.05
  vs_predicted:
    tps_ratio: 0.89                   # actual / predicted
    mem_ratio: 1.06
    drift: moderate                   # ok | moderate | high (>0.25 → flag for LEARN)
```

## Exit conditions

- **pass** → CORRECTNESS.
- **soft_fail** (`completed_steps < S` but ≥ 200, no OOM): treat as
  pass; record `early_truncation` in baseline_summary; CORRECTNESS may
  still proceed if it has enough loss samples.
- **hard_fail** → On_fail.

## On_fail

| Failure kind | Transition |
|--------------|------------|
| `OOM` (despite SMOKE passing) | mark plan `dead`; `to: PROJECTION` to pick next plan with stricter mem bias. **Update ExecutionModel** with the actual mem delta. |
| `HANG` after smoke passed | `to: PREFLIGHT` (env_probe subset); something drifted in env between SMOKE and BASELINE. |
| `NUMERICAL` (NaN at full shape but not at smoke) | `to: ABORT` `kind: NUMERICAL`; record full layer shape in escalation. Likely a real model bug. |
| `INVALID_CONFIG` | impossible after SMOKE pass — log to `knowledge/anti-patterns.md` and `to: ABORT` `kind: UNKNOWN`. |

## Drift annotations

If `vs_predicted.tps_ratio < 0.7` or `mem_ratio > 1.25`:

- Log a `prediction_drift` event in `state/execution_model.yaml.events`.
- LEARN reads these at session end to decide whether to recalibrate
  coefficients. Do NOT recalibrate mid-session.

## Cost budget

- Default 500 steps. Wall time scales with `step_time × S / dp`.
- Charged to `setup_cost` (not tuning rounds).

## Scale-aware notes

The procedure is identical regardless of cluster shape; only defaults
and thresholds shift:

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| `S` default | 500 | 500-1000 (more steps for stable AR / A2A measurement) |
| `mode` flag | `baseline` | `baseline` |
| Expected `comm_ratio` band | < 15% (intra-node xGMI is fast) | 15-30% (IB cross-node dominant) |
| Expected `bubble_ratio` | 0 if `pp=1` (typical) | 5-15% if `pp > 1`; 0 otherwise |
| Wall time | tens of minutes | minutes-to-hours; budget accordingly |

For multi-node: collect cross-node timing breakdowns
(`alltoall_share_pct`, `ar_share_pct`) carefully — Diagnose relies on
them in round 1.
