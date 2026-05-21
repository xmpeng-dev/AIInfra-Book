# envsweep — Inner Env Sweep (Optional)

## Purpose

Lock the current champion's structure and **sweep environment axes**
on it. Because env doesn't change shape, the sweep is cheap (30-50
steps), low-risk (no OOM), and lets the outer loop focus on
structural moves.

EnvSweep is the "inner" of the **double-layer tuning loop**. It runs
**conditionally** between Stage 6.5 (CORRECTNESS-LITE) and 6.7
(SETTLE) — see `workflow/SKILL.md` §1.

Schema mirrors `pilot/README.md` §8.5.

## Trigger conditions

EnvSweep runs in round N when **any** of:

```yaml
trigger:
  diagnose.env_suspect_present:        true       # most common path
  settle.force_envsweep:               true       # set by stagnation handling
  rounds_since_last_envsweep:          >= 3       # forced refresh
  promotion_just_happened_with_env_axes_unswept:  true
```

It does **not** run when:

- `kind` of any current-round failure is `HANG` or `CLUSTER` (env
  baseline itself suspect — fix at PREFLIGHT first).
- Champion just changed parallelism radically (let structure settle
  before tuning env on it).

## Inputs

```yaml
champion_plan:                        # the locked structure
diagnosis_report.env_suspect:         # primary candidate axes
target_vector.budget:
cluster_profile:
plan_graph:
```

## Skills consulted

- `env/SKILL.md` (catalog navigation)
- `env/<area>.md` (definitive flag definitions)
- `optimization/<bottleneck>/env.md` (bottleneck-specific env hints)
- `axis_taxonomy.md` (only `cluster_shared`, `weakly_local`,
  `strongly_local` axes eligible)
- `constraints/env.md` (incompatibility matrix)

## Tools called

```python
constraint.check_env(env_diff, baseline) -> {valid, violations}
env_probe.sweep(base_plan, candidate_envs, max_steps) -> {best_env_diff, results}
state.checkpoint(tuning_state)
```

## Procedure

### Step 1 — Build candidate set

Sources (in order of precedence):

1. `diagnosis_report.env_suspect[*]` (highest priority).
2. `optimization/<bottleneck>/env.md` recommended flags for the
   current bottleneck.
3. Forced-refresh: rotate through `weakly_local` flags whose last
   sweep is older than `rounds_since_last_envsweep ≥ 3`.

For each candidate flag, choose values from the catalog's `range` plus
the current value (control). Cap candidates per round:

```yaml
caps:
  max_flags:        5
  max_combinations: 8
  max_steps_per_run: 50
```

### Step 2 — Filter & expand combinations

```python
flags = top_5(env_suspect_priority + bottleneck_recommended)
values = {f: catalog[f].range_for(bottleneck, scale) for f in flags}

# Generate combinations: usually one-flag-at-a-time
combos = [{f: v} for f in flags for v in values[f]]

# If diagnose suggests two flags interact, add the cross product
if env_suspect.has_interaction("NCCL_BUFFSIZE", "NCCL_MIN_NCHANNELS"):
    combos += cross_product([flags["NCCL_BUFFSIZE"], flags["NCCL_MIN_NCHANNELS"]])

combos = combos[:max_combinations]
combos = [c for c in combos if constraint.check_env(c, baseline).valid]
```

### Step 3 — Sweep

```python
sweep = env_probe.sweep(
    base_plan=champion_plan,
    candidate_envs=combos,
    max_steps=50,
)
# Internally: spawn each combo as a short job (parallel where capacity
# allows, serial otherwise); record TPS / mem / correctness drift.
```

### Step 4 — Pick winner

Winner = combo with `tps_delta_pct ≥ ε_envsweep_promote = 1.5%` over
locked baseline AND no correctness drift AND `mem_peak ≤ 1.05 ×
baseline.mem_peak`.

If no combo passes the threshold, **no winner** — record null and
proceed to SETTLE without merging.

### Step 5 — Merge winner into champion plan

```python
# strongly_local + weakly_local: stay in Plan.env.diff
champion_plan.env.diff.update(winner.env_diff)

# cluster_shared: promote to ClusterProfile.env_baseline (with bump)
if winner.axis_type == "cluster_shared":
    cluster_profile.env_baseline.update(winner.env_diff)
    cluster_profile.env_baseline.version = bump(version)
    # all open Plan.env.diff with same key drop the key (now default)
```

Champion's metrics are NOT immediately updated (the sweep used short
runs). Settle re-evaluates next round; if the merged env stays a win,
it sticks; otherwise rollback per `Step 6`.

### Step 6 — Soft rollback (optional safety)

If the next outer round's first observation shows
`tps < champion.tps × (1 - ε_rollback = 1.0%)`, automatically
**revert** the merged env keys (record the rollback in
`knowledge/anti-patterns.md` so we don't re-pick same combo).

## EnvSweepResult (output schema)

Mirrors `pilot/README.md` §8.5:

```yaml
sweep_id: r3_envsweep_1
parent_plan: r2_p4
trigger: COMM_BOUND | env_suspect | forced_refresh
candidates:
  - env_diff: {NCCL_BUFFSIZE: 8388608}
    tps: 17900
    delta_pct: +0.6
    mem_peak_gb: 158
    correctness_drift_pct: 0.1
  - env_diff: {NCCL_BUFFSIZE: 16777216, NCCL_MIN_NCHANNELS: 16}
    tps: 18650
    delta_pct: +4.8                # ← winner
    mem_peak_gb: 159
    correctness_drift_pct: 0.2
  - env_diff: {NCCL_BUFFSIZE: 33554432}
    tps: 17600
    delta_pct: -1.1
    mem_peak_gb: 158
    correctness_drift_pct: 0.0
selected_diff:
  NCCL_BUFFSIZE: 16777216
  NCCL_MIN_NCHANNELS: 16
selected_axis_type: weakly_local
merged_into: champion_plan          # or 'cluster_baseline' if cluster_shared
cost_gpu_h: 0.3
status: success | no_winner | partial_failure
```

## Exit conditions

- **success**: winner found and merged. → SETTLE.
- **no_winner**: no combo passed threshold. → SETTLE.
- **partial_failure**: some combos hit OOM/HANG (rare, env shouldn't
  cause OOM); record the failed combos in
  `exhausted_neighborhoods` with `reason: env_oom`. Still → SETTLE
  unless any failure was `kind: NUMERICAL` (then ABORT).

## On_fail

| Condition | Transition |
|-----------|------------|
| `env_probe.sweep` itself raises | `to: PREFLIGHT (env_probe subset)`; mark suspected env baseline drift |
| `kind: NUMERICAL` from any combo | `to: ABORT` `kind: NUMERICAL`, escalate |
| All combos `kind: HANG` | `to: PREFLIGHT`; same baseline drift suspicion |

## Cost budget

Per sweep (capped):

- ≤ 8 combos × ≤ 50 steps × short = **≤ 0.4 GPU·h** typical.
- Cheaper than one structural Champion-Challenger round.

EnvSweeps **count against `target_vector.budget.gpu_h`** but do **not**
count against `budget.max_rounds` (they're inner-loop).

## Cross-links

- Catalog of every flag → `env/<area>.md`
- Bottleneck → flag hints → `optimization/<bottleneck>/env.md`
- Compat matrix → `constraints/env.md`
- Promotion to baseline → `learn.md` (formalized at session end)

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Common candidates | `PYTORCH_HIP_ALLOC_CONF`, `HSA_*`, `OMP_NUM_THREADS` | `NCCL_BUFFSIZE`, `NCCL_MIN_NCHANNELS`, `NCCL_ALGO`, `MSCCL_ENABLE` |
| Trigger frequency | low (env matters less intra-node) | high (env is a major lever for inter-node comm) |
| Combo runtime | ~30 steps × 2-3 min | ~50 steps × 3-5 min |
| `max_combinations` typical | 4 | 8 |
| Promotion to `env_baseline` for `cluster_shared` | rare (mostly `weakly_local` here) | common (NCCL/IB stable across jobs) |
