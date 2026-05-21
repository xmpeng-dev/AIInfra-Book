# settle — PlanGraph Promotion + Stop Conditions

## Purpose

Take the round's results (Snapshots from `Execute` + optional
`EnvSweepResult`) and decide:

- Which candidate (if any) becomes the **new champion**.
- Which become **shelved** vs **dead**.
- Whether the loop should **continue** or **stop**.
- What `policy_for_next_replan` should be (`exploit` / `explore` /
  `explore_exploit`) and whether to set `force_envsweep`.

Settle is the **only writer** of `champion`, `champion_history`,
`frontier`, `exhausted_neighborhoods`, and `metadata.*` fields.

## Inputs

```yaml
round_snapshots:    state/snapshots/r{N}_p{*}.yaml     # all completed/dead in this round
envsweep_result:    state/envsweep/r{N}.yaml | null
plan_graph:         state/plan_graph.yaml
target_vector:
budget_used:        state/tuning_state.yaml.budget_used
correctness_lite:   per-snapshot pass/drift
```

## Procedure

### Step 1 — Classify each round snapshot

```python
for snap in round_snapshots:
    node = plan_graph.nodes[snap.plan_id]

    if snap.failure_kind in {OOM, NUMERICAL, INVALID_CONFIG}:
        node.status = "dead"
        plan_graph.exhausted_neighborhoods.append(
            {around: node.parent, axis: node.derived_axis.key,
             value: node.derived_axis.to,
             source: "settle", reason: failure_to_reason(snap.failure_kind)})
        continue

    if snap.correctness == "fail":
        node.status = "dead"
        exhausted_neighborhoods.append({..., reason: "correctness_fail"})
        continue

    if snap.early_stop_reason in {tps_below_champion_85pct,
                                   tps_below_prediction_70pct}:
        node.status = "shelved"
        exhausted_neighborhoods.append({..., reason: "tps_no_gain"})
        continue

    # job completed cleanly
    node.status  = "completed"      # transient; promotion logic below sets shelved/champion
    node.metrics = snap.metrics
```

### Step 2 — Pick round winner

```python
clean = [n for n in round if n.status == "completed"]
winner = max(clean, key=lambda n: n.metrics.tps, default=None)
```

If `winner is None` → no promotion; jump to Step 5.

### Step 3 — Promotion margins

```python
EPS_PROMOTE = 0.02            # 2% TPS bar to upgrade champion
EPS_HARD    = 0.05            # 5% TPS to override correctness 'drift' tag

champion = plan_graph.champion

if winner.tps > champion.metrics.tps * (1 + EPS_PROMOTE):
    if winner.correctness in ("pass", null) or \
       (winner.correctness == "drift" and winner.tps > champion.metrics.tps * (1 + EPS_HARD)):
        promote(winner)               # → Step 4
    else:
        shelve(winner)                # drift but not enough gain to override
elif winner.tps > champion.metrics.tps:
    shelve(winner)                    # marginal gain; stays shelved
else:
    shelve(winner)                    # no gain
```

Mark all non-winner clean nodes as `shelved`.

### Step 4 — Apply promotion

```python
plan_graph.champion = winner.plan_id
plan_graph.champion_history.append({
    round: N, plan_id: winner.plan_id, promoted_at: ts, by: "tps",
    prev_champion: champion, gain_pct: gain * 100,
})
plan_graph.metadata.rounds_since_promotion = 0
# Old champion stays as 'shelved' (NOT dead) — useful for backtrack
plan_graph.nodes[champion].status = "shelved"
```

### Step 5 — EnvSweep merge bookkeeping

If `envsweep_result.status == "success"`:

```python
champion.plan.env.diff.update(envsweep_result.selected_diff)
plan_graph.nodes[champion].metrics_pending = True   # re-measure next round
```

If `envsweep_result.status == "no_winner"`:

```python
for combo in envsweep_result.candidates:
    exhausted_neighborhoods.append(
        {around: champion, axis: list(combo.env_diff.keys())[0],
         value: combo.env_diff_values, source: "envsweep",
         reason: "env_no_gain" if combo.delta_pct >= 0 else "env_regression"})
```

### Step 6 — Stagnation + backtrack rules

Update metadata, then apply rules in order:

```python
if no promotion this round:
    plan_graph.metadata.rounds_since_promotion += 1

# Rule A: stagnation → next round's policy switches to explore
if plan_graph.metadata.rounds_since_promotion >= 2:
    plan_graph.metadata.policy_for_next_replan = "explore"

# Rule B: forced explore round
if plan_graph.metadata.rounds_since_explore >= 3:
    plan_graph.metadata.policy_for_next_replan = "explore"
    plan_graph.metadata.rounds_since_explore = 0
elif plan_graph.metadata.policy_for_next_replan == "explore":
    plan_graph.metadata.rounds_since_explore += 1
else:
    plan_graph.metadata.rounds_since_explore += 1

# Rule C: subtree dead-out → backtrack
champion_dead_rate = subtree_dead_rate(champion, plan_graph)
if champion_dead_rate > 0.5 AND plan_graph.metadata.rounds_since_promotion >= 2:
    backtrack_to = top_k_excluding_champion(frontier, k=1)[0]
    promote_via_backtrack(backtrack_to)
    plan_graph.metadata.backtrack_count += 1
```

`promote_via_backtrack` is identical to Step 4 but `by: "backtrack"`
and previous champion still goes `shelved`.

### Step 7 — Update frontier

```python
plan_graph.frontier = (
    [plan_graph.champion]
    + top_k(shelved_nodes, key=lambda n: n.metrics.tps, k=2)
)
```

### Step 8 — Stop conditions

Apply in order:

```python
# (a) Budget exhausted
if budget_used.gpu_h >= target_vector.budget.total_gpu_h:
    return STOP, "budget_gpu_h_exceeded"
if budget_used.rounds >= target_vector.budget.max_rounds:
    return STOP, "max_rounds_reached"
if budget_used.wallclock_h >= target_vector.budget.wallclock_h:
    return STOP, "wallclock_exceeded"

# (b) Target met and primary stagnant
constraints_met = all(target_vector.constraints satisfied by champion)
primary_stagnant = (rounds_since_promotion >= 2)
if constraints_met and primary_stagnant:
    return STOP, "target_met_and_stagnant"

# (c) Hopelessness: no high-priority candidate left in frontier
no_promising_axis = max(predicted_priority(frontier)) < 1.0
if rounds_since_promotion >= 2 and no_promising_axis:
    return STOP, "frontier_dry"

# (d) Hard floor: never stop without at least 1 round of optimization
if N == 0:
    return CONTINUE
```

If none triggers → CONTINUE (next round's OBSERVE).

### Step 9 — Decide `force_envsweep` for next round

```python
plan_graph.metadata.force_envsweep_next = (
    rounds_since_promotion >= 1
    and not envsweep_just_ran_this_round
    and any_unswept_weakly_local_axis(diagnosis_report)
)
```

This is consumed by `envsweep.md` triggers in round N+1.

## State written

```yaml
state/plan_graph.yaml:                # in place mutation; full schema in plan_graph.md
  champion:               <new>
  champion_history:       [...]
  frontier:               [...]
  exhausted_neighborhoods: [...]
  nodes.r{N}_p{i}.status: champion|shelved|dead
  nodes.r{N}_p{i}.metrics: {...}
  nodes.r{N}_p{i}.correctness: pass|fail|drift
  metadata:
    rounds_since_promotion: <int>
    rounds_since_explore:   <int>
    backtrack_count:        <int>
    policy_for_next_replan: exploit|explore|explore_exploit
    force_envsweep_next:    bool

state/round_summary/r{N}.yaml:
  decision: continue | stop
  stop_reason: null | budget_* | target_met_and_stagnant | frontier_dry
  promoted: null | <plan_id>
  shelved:  [...]
  dead:     [...]
  envsweep: null | <sweep_id>
  next_policy: exploit|explore|explore_exploit
```

## Exit conditions

- **CONTINUE** → next round (OBSERVE).
- **STOP** → REPORT.

## On_fail

Settle's own logic should never fail. If state inconsistencies are
detected (e.g. `plan_graph.champion` not in `nodes`):

| Condition | Recovery |
|-----------|---------|
| Champion missing from nodes | `to: ABORT` `kind: UNKNOWN`, escalate; State Layer corrupt |
| All round snapshots missing | `to: ABORT` `kind: UNKNOWN` |
| Multiple winners with identical TPS within 0.5% | tie-break by `confidence` (highest), then `correctness` (pass > drift), then candidate id (lower) |

## Cross-links

- PlanGraph schema → `plan_graph.md`
- Re-Plan reads `policy_for_next_replan` and `force_envsweep_next`
- Stop conditions feed `report.md`
- Backtrack rationale → `pilot/README.md` §3.1 step 8

## Scale-aware notes

The promotion rules are scale-independent. Only thresholds may differ
in practice (configured via `target_vector.constraints`):

| Threshold | Single node typical | Multi node typical |
|-----------|---------------------|--------------------|
| `EPS_PROMOTE` | 0.02 (2%) | 0.02 (2%) |
| `EPS_HARD` (override drift) | 0.05 | 0.05 |
| `subtree_dead_rate` backtrack threshold | 0.5 | 0.5 |
| `rounds_since_explore` cap | 3 | 3 |
| `rounds_since_promotion` for stop | 2 | 2-3 (cross-node noise can hide a real win) |

For multi-node, prefer **2 consecutive low-gain rounds** rather than
1 before declaring stagnation, because cross-node TPS noise is
typically ±1.5%.
