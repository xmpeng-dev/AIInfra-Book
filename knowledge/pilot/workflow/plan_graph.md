# plan_graph — Search-Tree Data Model

## Purpose

`PlanGraph` is the persistent data structure that records:

- Every plan considered (champion, shelved, dead, pending).
- Parent-child derivation (which plan was derived from which by which
  axis change).
- Pruning memory (`exhausted_neighborhoods`).
- Champion history.

It is the **shared substrate** between `replan.md` (reads) and
`settle.md` (writes). This file defines the schema and access rules;
mutation logic is owned by `settle.md`.

Schema mirrors `pilot/README.md` §8.9.

## Schema

```yaml
# state/plan_graph.yaml
schema_version: 1.0
session_id: <str>

champion: <plan_id>                  # current best, references nodes[*]

champion_history:
  - {round: 0, plan_id: r0_p0, promoted_at: ts, by: baseline}
  - {round: 2, plan_id: r2_p1, promoted_at: ts, by: tps,
     prev_champion: r0_p0, gain_pct: 8.4}

frontier:                            # "expandable" nodes (eligible Re-Plan source)
  - r0_p0                            # current champion always in frontier
  - r1_p3                            # high-priority shelved
  # populated by settle.md after each round

nodes:
  r0_p0:
    parent:        null
    derived_axis:  null              # axis_change object that produced this node
    status:        champion          # champion | shelved | dead | pending | completed
    round:         0
    plan_ref:      state/plans/r0_p0.yaml
    snapshot_ref:  state/snapshots/r0_p0.yaml
    metrics:                         # mirrored from snapshot for fast access
      tps:          16100
      mem_gb:       178
      comm_ratio:   0.14
      bubble_ratio: 0.00
    correctness:   pass              # pass | fail | drift | null
    notes:         "baseline"

  r1_p1:
    parent:        r0_p0
    derived_axis:
      key:    runtime.mbs
      from:   2
      to:     3
      type:   structural             # see axis_taxonomy.md
    status:        shelved
    round:         1
    plan_ref:      state/plans/r1_p1.yaml
    snapshot_ref:  state/snapshots/r1_p1.yaml
    metrics:       {tps: 17200, mem_gb: 211, ...}
    correctness:   pass
    notes:         "above champion but within 3% margin; not promoted"

exhausted_neighborhoods:
  # Tuples of (around_node, axis, value) that should NOT be re-tried.
  - {around: r0_p0, axis: NCCL_BUFFSIZE, value: 4M,  source: settle, reason: tps_no_gain}
  - {around: r0_p0, axis: NCCL_BUFFSIZE, value: 16M, source: settle, reason: tps_regression}
  - {around: r1_p1, axis: runtime.mbs,   value: 4,   source: replan, reason: mem_oom_predicted}

metadata:
  rounds_since_promotion: 1
  rounds_since_explore:   2
  backtrack_count:        0
  rollback_pending:       false
  policy_for_next_replan: exploit | explore | explore_exploit
  total_plans_considered: 14
  total_plans_executed:   9
  total_plans_dead:       3
```

## Node statuses

| Status | Meaning | Re-Plan can derive from? |
|--------|---------|-------------------------|
| `champion` | Current best, satisfies promotion margin and constraints. | Yes (primary). |
| `shelved` | Completed cleanly, didn't beat champion (or only marginally). | Yes (secondary, in explore rounds). |
| `dead` | OOM / failed / correctness-fail / early-stopped well below champion. | **No.** |
| `pending` | Selected by Re-Plan, not yet executed. | No (must complete first). |
| `completed` | Executed but not yet classified by Settle. | Transient; never seen across rounds. |

## Derivation invariants

- Every non-root node has exactly one `parent`.
- `derived_axis` is non-null for non-root nodes (records the single
  axis that differs from `parent`).
- A node may differ from its parent by **only one axis** in the typical
  case. Multi-axis derivation is allowed but discouraged (Settle treats
  them as low-attribution: gains can't be cleanly assigned to one
  axis).

## `exhausted_neighborhoods` semantics

A tuple `(around, axis, value)` means: **don't try setting `axis=value`
when deriving from `around`** (or any of its descendants in the same
subtree, **unless** the derivation crosses a champion promotion that
changes the bottleneck).

Conservative interpretation: Re-Plan's `Step 3 · Filter` matches by
exact `(parent, axis, value)`. To be more aggressive, future versions
may match by neighborhood (e.g. `mbs ∈ ±1`) — out of scope here.

`reason` is one of:
- `tps_no_gain` — execution showed no improvement.
- `tps_regression` — execution showed regression.
- `mem_oom_actual` — execution OOMed.
- `mem_oom_predicted` — `constraint.estimate_mem` rejected pre-submit.
- `correctness_fail` — lite or full correctness failed.
- `anti_pattern_match` — `knowledge/anti-patterns.md` hit.

## Frontier maintenance

Frontier = candidates `replan.md` may use as derivation source.

```python
frontier = [champion]
frontier += top_k(shelved, key=lambda n: n.metrics.tps, k=2)
```

Updated by `settle.md` after each round. Size kept small (≤ 3-5) to
keep Re-Plan tractable.

## Champion lineage

`champion_history` is append-only. Useful for:

- Calculating `parent_stability_bonus` in `replan.md`.
- LEARN: identifying "structural attractors" (axes that keep producing
  champions across sessions).

## Backtrack mechanics

When `settle.md` triggers backtrack:

```yaml
champion: <new>                   # one of frontier (not previous champion)
champion_history.append:
  {round: N, plan_id: <new>, promoted_at: ts, by: backtrack,
   prev_champion: <old>, reason: subtree_exhausted}
metadata.backtrack_count += 1
```

The previous champion stays as `shelved` (not `dead`); its subtree's
`exhausted_neighborhoods` entries persist.

## STRUCTURAL_INVALIDATION reset

If state machine triggers re-PROJECTION:

```yaml
nodes: {}
champion: null
champion_history: []
exhausted_neighborhoods: []
metadata.total_plans_considered: 0
# session_id continues; new ExecutionModel + new initial plans seed it
```

## Read access patterns

| Caller | What it reads |
|--------|--------------|
| `replan.md` Step 1 | `champion`, `frontier`, `metadata.policy_for_next_replan` |
| `replan.md` Step 3 | `exhausted_neighborhoods` |
| `replan.md` Step 4 | `champion_history`, `nodes` |
| `settle.md` | `nodes`, `metadata.rounds_since_*`, `frontier` |
| LEARN | full graph |
| REPORT | full graph |

## Write access

ONLY `settle.md` writes nodes' `status`, `metrics`, `correctness`,
`exhausted_neighborhoods`, `champion`, `champion_history`, `frontier`,
`metadata`. `replan.md` may add `pending` nodes; nothing else may
mutate.

## Persistence

Written by `state.checkpoint(tuning_state)` at every stage exit. The
`tuning_state` superset includes a reference to `plan_graph.yaml` plus
the round-local pieces (CandidatePool, snapshots).

## Scale-aware notes

The schema is scale-independent. Typical sizes:

- **Single node**: 20-50 nodes total per session, frontier ≤ 3, ≤ 2
  backtracks. File typically < 200KB.
- **Multi node**: similar order of magnitude (the search loop has the
  same K and rounds), but each node's metrics include
  `inter_node_util_avg` and per-axis-type counts may skew toward
  comm-related axes.
- For both: `exhausted_neighborhoods` stays small (≤ 30 entries
  typical); full-table scan in Re-Plan filter is fine.
