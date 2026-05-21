# constraints/validation — `diagnose_failure` Mapping

## Purpose

Defines `constraint.diagnose_failure(snapshot|error) -> FailureReport`:
turn a raw failure into a structured `FailureReport` that
`state_machine.md` can route. Every failure path in the workflow
funnels through this file.

`FailureReport` schema is canonical in `constraints/SKILL.md` §
FailureReport.

## Tool contract

```python
constraint.diagnose_failure(input) -> FailureReport
# input is one of:
#   - Snapshot (status != ok)        # normal path
#   - exception/error from submit.run / observe.snapshot
#   - timeout sentinel
```

## Mapping rules (in order)

Apply rules top-down; first match wins. Each rule produces one
`kind` plus an optional `suggested_transition.hints` block.

### 1. NUMERICAL (highest priority — never auto-recover)

```python
if snap.metrics.has_nan or snap.metrics.has_inf:
    return FailureReport(
        kind="NUMERICAL",
        evidence=["snap.metrics.has_nan/has_inf at step {first_nan_step}"],
        suggested_transition={to: "ABORT"},
        counts_against_budget=True,
        escalate=True,
    )

# Loss drift detected by lite-correctness:
if snap.correctness == "fail" and snap.correctness_kind == "loss_drift":
    return FailureReport(
        kind="NUMERICAL",
        evidence=["loss_delta_pct={delta} > tolerance"],
        suggested_transition={to: "ABORT"},
        counts_against_budget=True,
        escalate=True,
    )
```

### 2. OOM

```python
if snap.status == "oom" or snap.metrics.has_oom_event:
    return FailureReport(
        kind="OOM",
        evidence=["snap.status=oom, mem_peak={observed}GB at step {s}"],
        suggested_transition={
            to: "REPLAN",
            hints: {
                force_recompute_full: snap.plan.recompute != "full",
                prefer_axis: "runtime.recompute" or "runtime.mbs",
            },
        },
        counts_against_budget=False,
    )
```

### 3. HANG

```python
if snap.status == "hung" or wall_time_no_progress > hang_timeout_s:
    return FailureReport(
        kind="HANG",
        evidence=[f"no step progress for {dt}s; nccl_init_time={snap.nccl_init_time_s}s"],
        suggested_transition={
            to: "PREFLIGHT",
            hints: {env_probe_subset: candidate_env_keys_recently_changed},
        },
        counts_against_budget=False,
    )
```

### 4. CLUSTER

```python
if error_kind in {"node_down", "driver_error", "ecc_uncorrectable",
                  "rocm_runtime_error", "framework_init_fail_after_smoke"}:
    return FailureReport(
        kind="CLUSTER",
        evidence=[error_message_summary],
        suggested_transition={to: "PREFLIGHT", hints: {full: True}},
        counts_against_budget=False,
    )
```

### 5. INVALID_CONFIG

```python
# constraint.check rejected something at submit-time but pre-submit
# accepted it (rare; a real bug)
if error_kind == "framework_rejects_config":
    return FailureReport(
        kind="INVALID_CONFIG",
        evidence=[error_detail, "constraint.check did not catch this"],
        suggested_transition={to: "REPLAN", hints: {drop_axis: violated_axis}},
        counts_against_budget=False,
        escalate=False,
    )
    # also append to knowledge/anti-patterns.md (catalog gap)
```

### 6. STRUCTURAL_INVALIDATION

```python
if model_spec_changed_since_projection or dataset_shape_changed:
    return FailureReport(
        kind="STRUCTURAL_INVALIDATION",
        evidence=["model_spec_hash mismatch with execution_model.yaml"],
        suggested_transition={to: "PROJECTION"},
        counts_against_budget=True,
    )
```

### 7. BUDGET_EXCEEDED

```python
if budget_used.gpu_h >= target_vector.budget.total_gpu_h:
    return FailureReport(
        kind="BUDGET_EXCEEDED",
        evidence=[f"budget_used.gpu_h={used} >= limit={limit}"],
        suggested_transition={to: "REPORT"},
        counts_against_budget=False,
    )
```

### 8. UNKNOWN (fallback)

```python
return FailureReport(
    kind="UNKNOWN",
    evidence=[full_error_dump_summary],
    suggested_transition={to: "ABORT"},
    counts_against_budget=True,
    escalate=True,
)
```

## Confidence in classification

The mapping above is rule-based and deterministic; no probabilistic
output. If two rules apply (e.g. a job both OOMed AND had NaN),
**NUMERICAL wins** (highest priority).

## Reentry hints

`suggested_transition.hints` are passed forward; the receiving stage
respects or ignores them per its own logic. Examples:

| Hint | Consumed by | Behavior |
|------|-------------|---------|
| `force_recompute_full: true` | `replan.md` Step 3 | bias next candidate to `recompute=full` |
| `prefer_axis: <key>` | `replan.md` Step 4 | priority bonus for candidates touching this axis |
| `env_probe_subset: [keys]` | `preflight.md` | re-probe only these env keys (subset mode) |
| `drop_axis: <key>` | `replan.md` Step 5 | exclude this axis from current round |

## Anti-loop guards

Tracked in TuningState, not here, but `validation.md` is responsible
for **emitting** the events that the guard counts:

```yaml
state/tuning_state.yaml.failure_events:
  - {at: <ts>, kind: OOM, plan_id: r2_p3}
  - {at: <ts>, kind: OOM, plan_id: r2_p4}
  - {at: <ts>, kind: OOM, plan_id: r2_p5}    # 3 in a row → state_machine ABORTs
```

`state_machine.md` § "Reentry guard" reads this list to enforce
`max_in_session` and consecutive-same-kind ABORT.

## Audit trail

Every `FailureReport` is appended to `state/failure_log.yaml` with
its `failure_id` for postmortem and LEARN consumption (anti-patterns
catalog).

## Cross-links

- FailureReport schema → `constraints/SKILL.md`
- State-machine routing → `workflow/state_machine.md` § On-fail
- OOM-specific rules → `constraints/oom.md`
- Anti-pattern persistence → `workflow/learn.md`,
  `knowledge/anti-patterns.md`

## Scale-aware notes

The mapping is identical at all scales. Frequencies differ:

- **Single node**: `OOM` and `INVALID_CONFIG` dominate; `HANG` /
  `CLUSTER` rare.
- **Multi node**: `HANG` (NCCL/IB timeouts) and `CLUSTER` (node
  flakiness) common; `OOM` still common but reasons can be
  cross-node-driven (uneven sharding).
