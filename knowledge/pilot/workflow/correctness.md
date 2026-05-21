# correctness — Numerical Validation

## Purpose

Verify that a Snapshot's loss curve matches a **reference**. Two modes:

- **Stage 5 CORRECTNESS** (full): runs after BASELINE. Establishes the
  reference-vs-current relationship; gates entry to OPTIMIZE_LOOP.
- **Stage 6.5 CORRECTNESS-LITE** (per round): cheap drift check on
  each challenger. Marks divergent candidates `dead` so they don't
  get promoted as fast-but-wrong.

Both modes use the same logic; they differ in the **reference source**
and **step horizon**.

## Inputs

```yaml
mode: full | lite

snapshot:                       # the run being validated
target_vector.constraints.loss_drift_max_pct:   default 1.0   # %
target_vector.constraints.loss_window_steps:    default 200

# Mode-specific:
reference (full):  user-supplied OR a tiny single-GPU run on same data
reference (lite):  champion's recorded loss curve (state/snapshots/<champion>.yaml)
```

## Skills consulted

- `observe.md` (loss curve must be present in Snapshot)

## Tools called

```python
observe.compare_loss(job_id, reference) -> {pass, drift, delta_pct, where_diverged}
constraint.diagnose_failure(snapshot) -> FailureReport   # if NaN / inf detected
```

## Reference acquisition (full mode)

In order of preference:

1. **User-supplied curve** (`target_vector.reference_curve_path`). Best
   case; just compare.
2. **Tiny single-GPU reference run** at this stage (one-time cost):
   ```python
   ref_plan = baseline_plan.copy()
   ref_plan.parallelism = {tp:1, pp:1, dp:1, ep:1}
   ref_plan.runtime.layers = min(4, baseline_plan.layers)
   ref_plan.runtime.steps  = 200
   submit.run(ref_plan, scale={nodes:1, gpus:1, mode:'reference'})
   ```
   Persist as `state/reference_curve.yaml`.
3. **Cross-config comparison** (no external reference): compare the
   first 200 steps of the parallelized baseline against itself with
   recompute=full (degenerate fallback; only flags catastrophic NaN).

## Compare procedure

```python
ref_loss = load(reference)             # list of (step, loss)
cur_loss = snap.loss.curve              # list of (step, loss)

# Align on common step grid, smooth with EMA(alpha=0.1)
delta = max_per_step(|EMA(cur) - EMA(ref)|) / EMA(ref) * 100

result.pass        = delta <= loss_drift_max_pct AND no NaN AND no inf
result.delta_pct   = delta
result.where_diverged = first_step where |Δ| > loss_drift_max_pct
```

## Pass / fail rules (full mode)

```python
pass = (
    not snap.metrics.has_nan
    and not snap.metrics.has_inf
    and result.delta_pct <= loss_drift_max_pct
    and snap.completed_steps >= loss_window_steps
)
```

## Pass / fail rules (lite mode, per challenger)

```python
# 'lite' compares the first 100-200 steps of challenger vs champion
pass = (
    not snap.metrics.has_nan
    and result.delta_pct <= 2 * loss_drift_max_pct   # looser bar mid-loop
    and snap.completed_steps >= 100
)

drift = result.delta_pct > loss_drift_max_pct       # warn-level, not fail
```

Lite mode is **cheap and noisy**: a "drift" alone doesn't kill a
plan; only a full fail does. Settle uses `drift` as a tie-breaker.

## State written

```yaml
state/correctness/<scope>.yaml:
  scope: full | r{N}_p{i}
  reference_source: user_supplied | tiny_ref | self_recompute
  reference_path: state/reference_curve.yaml
  result:
    pass: true | false
    delta_pct: 0.43
    where_diverged: null | <step>
    has_nan: false
    has_inf: false
  evaluated_at: <ts>

# In lite mode, also:
state/snapshots/r{N}_p{i}.yaml.correctness:
  pass: true | false
  drift: true | false
  delta_pct: 0.43
```

## Exit conditions

- **full pass** → OPTIMIZE_LOOP.
- **lite pass** → continue round; candidate is eligible for promotion.
- **lite drift (not fail)** → candidate stays alive; flag in
  Snapshot; Settle penalizes it in tie-breaks.
- **fail** → On_fail.

## On_fail

| Mode | Failure | Transition |
|------|---------|------------|
| full | `delta_pct > tolerance` | `to: ABORT` `kind: NUMERICAL`, escalate. The baseline plan is wrong; never auto-recover. |
| full | `has_nan` | `to: ABORT` `kind: NUMERICAL`, escalate. |
| full | reference unavailable AND fallback rejected | `to: ABORT` `kind: UNKNOWN`, escalate; ask user for reference. |
| lite | `delta_pct > 2× tolerance` | mark candidate `dead` in PlanGraph; round continues with remaining candidates. |
| lite | `has_nan` | mark candidate `dead`; record axis_change in `knowledge/anti-patterns.md`. |
| lite | all candidates fail correctness | `to: SETTLE` with `request_envsweep=false, decision: continue`; let stagnation handling kick in next round (likely escape via `recompute` / `mbs` axes). |

## Reference longevity

`state/reference_curve.yaml` is reused across all sessions on the
same `(model_spec, dataset, seed, dtype)` combination. Invalidate
when any of those change (PROJECTION's `STRUCTURAL_INVALIDATION`
transition wipes it).

## Cost budget

- Full mode (with tiny ref): ~10 min wall + ~0.2 GPU·h. Once per
  session (or zero if cached / user-supplied).
- Lite mode: zero extra — uses the challenger's loss curve already
  in its Snapshot.

## Scale-aware notes

The compare logic is identical regardless of cluster shape. Only the
reference-acquisition fallback differs:

- **Single node**: tiny reference uses 1 GPU on the same node; fast
  and isolates numerical effects from parallelism.
- **Multi node**: tiny reference still uses 1 GPU (avoid NCCL paths
  in the reference); allocate from any single node. Cross-node
  numerical drift is a real concern, so do not skip CORRECTNESS even
  if SMOKE passed.
- **MoE both scales**: include `expert_load_imbalance_pct` curve too;
  sudden imbalance changes can mask "loss looks the same" while
  gradients are not (a known anti-pattern, document in
  `knowledge/anti-patterns.md`).
- Skip lite mode in the **first** round after promotion if the new
  champion just passed full CORRECTNESS for the same axis_change
  family (avoid redundant cost).
