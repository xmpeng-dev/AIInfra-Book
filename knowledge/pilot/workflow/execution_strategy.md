# execution_strategy — How to Spend the Round Budget

## Purpose

Re-Plan generates a CandidatePool of N plans (often N > round
budget). This file decides:

- **Which strategy** to use (Champion-Challenger / Per-Plan Local
  Sweep / Successive Halving).
- **How many plans (`top_k`)** to actually run this round.
- **How to schedule** the round's GPU·h across them (parallel /
  serial / staged).

Mirrors `pilot/README.md` §3.1 step ⑥ and §8.10
`selection.strategy`.

## Inputs

```yaml
candidate_axes:        # diagnosis_report.candidate_axes (already typed)
budget:                # target_vector.budget
plan_graph_metadata:   # plan_graph.metadata
cluster_profile:       # for parallelism feasibility & per-plan cost estimate
predicted_costs:       # est_cost_gpu_h per candidate (from replan Step 4)
```

## The three strategies

### Champion-Challenger (default)

Run the top-K candidates **in parallel** (or serial on small clusters)
for a fixed step budget; promote the best that passes correctness.

- **Best for**: `cluster_shared` and `weakly_local` axes; most
  structural axes when prediction confidence ≥ 0.7.
- **K**: 2-4 typical (3 recommended).
- **Step budget per plan**: `challenger_steps = max(150,
  baseline_steps_min × 0.4)` (default 200).
- **Early stop**: `tps < champion × 0.85` after 50 steps.
- **Cost**: O(K · per_plan_cost).

### Per-Plan Local Sweep

For each candidate, run a tiny local sweep over a `strongly_local`
axis (e.g. `bucket_size_mb`, `max_split_size_mb`) to find that
plan's *own* optimum before comparing across candidates.

- **Best for**: `strongly_local` axes that are cheap to sweep
  (env-only, ≤ 30 step micro-runs).
- **Procedure**:
  ```
  for cand in selected:
      best_local = sweep(cand, axis=strongly_local_axis,
                         values=[v1, v2, v3], steps=30)
      cand.env.diff[axis] = best_local
  then run cand at full step budget
  ```
- **Cost**: O(K · (sweep_size · 30 + per_plan_cost))
  — only worth it when sweep_size ≤ 3 AND per_plan_cost is large.

### Successive Halving

Allocate a small step budget to many candidates, kill the bottom
half, double the budget for survivors, repeat.

- **Best for**: budget-tight rounds with many low-confidence
  candidates (mixed type pool, no clear bottleneck).
- **Procedure**:
  ```
  rung_steps   = [50, 150, baseline_steps_min]
  pool         = selected      # e.g. K=8
  for steps in rung_steps:
      run pool for `steps` each
      pool = top_half_by_tps(pool)
  promote pool[0]
  ```
- **Cost**: O(K · 50 + K/2 · 150 + K/4 · 500)
  ≈ O(K · per_plan_cost · 0.4) for K ≥ 4 — cheaper than
  Champion-Challenger when K is large.

## Strategy selection rule

```python
def select_strategy(axes, budget, metadata):
    types = {a.type for a in axes}

    # Tight budget always biases to halving when K > 3
    if budget.gpu_h_remaining < 3 * avg_per_plan_cost and len(candidates) > 3:
        return "Successive Halving"

    # Pure strongly_local pool & prediction confident → per-plan sweep
    if types <= {"strongly_local"} and avg_confidence >= 0.7:
        return "Per-Plan Local Sweep"

    # cluster_shared / weakly_local mostly → CC (so winner can promote)
    if {"cluster_shared", "weakly_local"} & types:
        return "Champion-Challenger"

    # Structural pool with mixed confidence → CC default; if K large → halving
    if "structural" in types:
        return "Champion-Challenger" if len(candidates) <= 4 else "Successive Halving"

    return "Champion-Challenger"
```

`Diagnose` may suggest a strategy via `suggested_strategy`; this rule
**defaults to following Diagnose** unless budget forces a downgrade
to Successive Halving.

## `top_k` selection

```python
top_k = min(
    len(candidates),
    floor(budget.gpu_h_remaining / avg_per_plan_cost / safety=1.3),
    {"Champion-Challenger": 4,
     "Per-Plan Local Sweep": 3,
     "Successive Halving":  8}[strategy],
)
top_k = max(top_k, 1)        # always run at least one plan
```

Always include at least 1 explore candidate when
`policy in {"explore", "explore_exploit"}` (see `replan.md` Step 7).

## Scheduling: parallel vs serial

```python
parallel_capacity = floor(cluster.gpus_total / plan.gpus_required)
parallel_count    = min(top_k, parallel_capacity, max_parallel_jobs)

# defaults
max_parallel_jobs = 1 if cluster.nodes <= 1 else 4
```

If `parallel_count == top_k`: run all in parallel, single submission
batch. Otherwise: serial (or batched) — see `execute.md` for the
mechanics.

**Single-node tuning** typically runs serial; **multi-node** can run
parallel when `top_k × plan.gpus_required ≤ cluster.gpus_total` and
each plan fits on a node-aligned subset.

## Cost / gain bookkeeping

For each strategy, record at end of round:

```yaml
state/candidate_pools/r{N}.yaml.execution:
  strategy:        Champion-Challenger
  parallel_count:  3
  serial_count:    0
  total_cost_gpu_h: 1.4
  step_budget_per_plan: 200
  outcome_summary:
    promoted:   r3_p1  (or null)
    shelved:    [r3_p2, r3_p3]
    dead:       []
```

These feed LEARN's strategy-effectiveness analysis.

## On_fail (strategy-level)

| Condition | Recovery |
|-----------|---------|
| Successive Halving rung 1 has no survivors above champion × 0.5 | abort halving, mark all `dead`, request `force_envsweep=true` from Settle |
| Per-Plan Local Sweep's micro-run takes > 2× expected (e.g. 30→90 step due to slow startup) | switch this round to Champion-Challenger; record `strategy_downgrade` event |
| `parallel_count` > capacity (e.g. plan resized at submit) | fall back to serial; Settle still uses results normally |

## Cross-links

- Pool generation → `replan.md`
- Step-budget rules / early-stop → `execute.md`
- Promotion rules → `settle.md`
- Axis types → `axis_taxonomy.md`

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| `max_parallel_jobs` | 1 (whole node serves one plan at a time) | 4 (or `nodes / plan.nodes_required`) |
| Default strategy | Champion-Challenger (serial); Per-Plan Sweep when env-only round | Champion-Challenger (parallel) when capacity allows; Halving when budget tight |
| Halving rung sizes | `[30, 100, 200]` (cheaper steps, but fewer total plans) | `[50, 150, 500]` (more steps to stabilize cross-node tps) |
| `top_k` typical | 2-3 | 3-4 |
