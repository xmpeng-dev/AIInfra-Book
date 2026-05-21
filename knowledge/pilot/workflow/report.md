# report — Final Config + Decision Trace

## Purpose

Produce the human-and-machine-readable artifact that closes a session:

- The **final winning Plan** (champion at stop time).
- A **decision trace** linking every promotion / backtrack / dead
  entry to the rule that produced it.
- A **TargetVector satisfaction report** (which constraints were
  met, which were not).
- **Cost actuals vs budget**.
- The set of **inputs LEARN should consume** to update
  `knowledge/`.

REPORT is a **terminal stage** in the happy path; only LEARN runs
after it.

## Inputs

```yaml
plan_graph:        state/plan_graph.yaml
target_vector:
budget_used:
all_round_summaries: state/round_summary/r{0..N}.yaml
all_snapshots:     state/snapshots/r{*}.yaml
cluster_profile:
execution_model:
session_id:
```

## Skills consulted

None (REPORT is a writer, not a thinker).

## Tools called

```python
state.checkpoint(tuning_state)        # final session-state snapshot
# Optional, if integrated:
knowledge.write_report_artifact(report)
```

## Output schema

```yaml
# state/report.yaml
schema_version: 1.0
session_id: pilot_run_20260418_a3
generated_at: <ts>

# === Final config (the deliverable) ===
final:
  plan: <full Plan schema, see workflow/plan.md>
  metrics:                            # from final champion's snapshot
    tps:           18400
    mem_peak_gb:   168
    comm_ratio:    0.16
    bubble_ratio:  0.05
    correctness:   pass
  vs_baseline:
    plan_id:        r0_p0
    tps_ratio:      1.53              # final / baseline
    mem_ratio:      0.94
    comm_delta_pct: -8                # negative = lower is better
  resolved_env:                       # cluster_baseline ⊕ plan.env.diff
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
    NCCL_BUFFSIZE: 16777216
    ...

# === TargetVector satisfaction ===
target_status:
  primary:     {metric: tps, value: 18400, baseline: 12000}
  constraints:
    - {expr: mem_peak_gb <= 180,            met: true,  value: 168}
    - {expr: per_token_cost_usd <= 1.2e-7,  met: true,  value: 0.94e-7}
    - {expr: correctness.loss_delta_pct <= 1.0, met: true, value: 0.43}
  budget:
    gpu_h:       {used: 7.2, limit: 10}
    rounds:      {used: 4,   limit: 5}
    wallclock_h: {used: 12,  limit: 24}
  preferences_honored:
    - {pref: prefer_lower_pp,         honored: true,  detail: "final pp=2 (baseline pp=4)"}
    - {pref: prefer_known_env_presets, honored: false, detail: "no preset matched cluster"}
  overall_outcome: success | partial | failed

# === Decision trace (auditable) ===
decision_trace:
  - round: 0
    stage: BASELINE
    event: champion_set
    plan_id: r0_p0
    by: baseline
    rule_ref: workflow/baseline.md
    metrics: {tps: 12000}

  - round: 1
    stage: SETTLE
    event: champion_promoted
    plan_id: r1_p2
    by: tps
    rule_ref: workflow/settle.md#step-3
    prev: r0_p0
    gain_pct: 31.7
    derived_axis: {key: comm.overlap, from: false, to: true, type: structural}

  - round: 1
    stage: ENVSWEEP
    event: env_merged
    sweep_id: r1_envsweep_1
    selected_diff: {NCCL_BUFFSIZE: 16777216, NCCL_MIN_NCHANNELS: 16}
    rule_ref: workflow/envsweep.md#step-5

  - round: 2
    stage: SETTLE
    event: backtrack
    plan_id: r1_p3
    prev: r2_p4
    reason: subtree_dead_rate=0.6
    rule_ref: workflow/settle.md#step-6

  - round: 3
    stage: SETTLE
    event: stop
    reason: target_met_and_stagnant
    rule_ref: workflow/settle.md#step-8

# === PlanGraph summary ===
plan_graph_summary:
  total_plans_considered: 14
  total_plans_executed:    9
  champions_total:         3              # length of champion_history
  backtracks:              1
  envsweeps:               2
  dead_plans:              3
  reentries:                              # from state machine reentry_log
    - {from: DIAGNOSE, to: PREFLIGHT, reason: cluster_profile.age>7d, round: 1}
  exhausted_neighborhoods_count: 11

# === Cost actuals ===
cost_actuals:
  preflight_gpu_h:    0.4
  projection_gpu_h:   1.0
  smoke_gpu_h:        0.2
  baseline_gpu_h:     0.6
  correctness_gpu_h:  0.2
  optimize_loop_gpu_h: 4.5
  envsweep_gpu_h:     0.4
  total_gpu_h:        7.3
  vs_budget:          0.73

# === Drift / calibration notes ===
calibration_notes:
  prediction_drift_events:                # from state/execution_model.yaml.events
    - {plan: r0_p0,  metric: tps, predicted: 18000, actual: 16100, drift_pct: -10.6}
    - {plan: r2_p4,  metric: mem, predicted: 165,   actual: 192,   drift_pct: +16.4}
  recommend_recalibration: true           # if any drift > 25% or repeated dir bias

# === Pointers for LEARN ===
learn_inputs:
  champion_path:        state/plans/r3_p_champion.yaml
  champion_snapshot:    state/snapshots/r3_p_champion.yaml
  shelved_winners:      [r1_p3, r2_p_promising_but_drift]   # to record as alternatives
  dead_with_lessons:    [r2_p5]                              # OOM with axis_change → anti-pattern
  env_promoted_to_baseline:                                  # cluster_shared upgrades
    - {key: NCCL_NET_GDR_LEVEL, value: 4, from_session: this}
  knowledge_writes_planned:
    - patterns:        ["MoE intra-node ep≤8 + comm.overlap=true is +30% on this cluster"]
    - cases:           [model_family: gpt_oss_20B_dense, scale: 16-node, config: <ref>]
    - anti_patterns:   ["pp=4 with mbs=2 OOMs on 192GB HBM (act mem underestimate)"]
```

## Procedure

```python
# 1. Collect
champion = plan_graph.champion
champion_node = plan_graph.nodes[champion]
champion_snap = load(champion_node.snapshot_ref)
baseline_node = plan_graph.nodes["r0_p0"]
baseline_snap = load(baseline_node.snapshot_ref)

# 2. Build final block
final = {
    plan: load(champion_node.plan_ref),
    metrics: champion_snap.metrics,
    vs_baseline: compare(champion_snap, baseline_snap),
    resolved_env: cluster.env_baseline | champion_node.plan.env.diff,
}

# 3. Evaluate target_status
target_status = evaluate_target_vector(target_vector, champion_snap, budget_used)

# 4. Walk all round summaries to assemble decision_trace
decision_trace = aggregate_events(round_summaries, plan_graph)

# 5. Assemble plan_graph_summary, cost_actuals, calibration_notes
# ...

# 6. Identify LEARN inputs
learn_inputs = derive_learn_inputs(plan_graph, champion, drift_events)

# 7. Persist
state.checkpoint(report)
```

## `overall_outcome` mapping

```python
if all constraints met AND primary improved over baseline AND not budget exhausted before constraints met:
    outcome = "success"
elif primary improved over baseline AND some constraints met:
    outcome = "partial"
else:
    outcome = "failed"            # rare: usually we'd ABORT first
```

## Exit conditions

- **success** → LEARN.
- (Settle has already gated us; report itself doesn't fail.)

## On_fail

If REPORT raises (e.g. corrupt state preventing write):

| Condition | Action |
|-----------|--------|
| Cannot read final champion snapshot | `to: ABORT` `kind: UNKNOWN`, escalate. |
| `state.checkpoint(report)` fails | retry once; if still fails, `to: ABORT`. |

REPORT failures should never delete the working state — the user can
recover by re-running REPORT against the final TuningState.

## Output artifacts

```
state/report.yaml             # the document above
state/checkpoints/report/<ts>/  # full state snapshot for replay
```

A future feature may render `state/report.yaml` to Markdown for human
viewing — out of scope for this skill.

## Cross-links

- Stop reasons → `settle.md` Step 8
- LEARN consumes `learn_inputs` → `learn.md`
- Cost accounting → `execute.md` § Cost bookkeeping +
  `envsweep.md` § Cost budget

## Scale-aware notes

The schema is scale-independent. What changes is how loud the
`calibration_notes.prediction_drift_events` section tends to be:

- **Single node**: prediction drift is usually small (< 10%) once
  PROJECTION has run; recalibration rarely needed.
- **Multi node**: cross-node bandwidth assumptions can drift more
  (network contention, RCCL version skew); recommend recalibration
  more readily and surface drift events in `learn_inputs.patterns`.
