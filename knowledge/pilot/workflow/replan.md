# replan — Candidate Generation + Priority

## Purpose

Read a `DiagnosisReport` + `PlanGraph` and produce a `CandidatePool`
of plans for `Execute` to run. Mirrors `pilot/README.md` §8.10.

Re-Plan is the one stage with most "creativity"; this file pins down
exactly **which steps** it runs and **how priority is computed**, so
the Agent's choices are reproducible.

## Inputs

```yaml
diagnosis_report:    state/diagnosis/r{N}.yaml
plan_graph:          state/plan_graph.yaml
target_vector:
cluster_profile:
execution_model:     state/execution_model.yaml
```

## Skills consulted

- `optimization/<bottleneck>/SKILL.md` (and any sub-files referenced
  in `diagnosis_report.recommended_skills`)
- `axis_taxonomy.md` (axis type → search behavior)
- `execution_strategy.md` (final candidate count + scheduling)
- `constraints/oom.md`, `constraints/config.md`, `constraints/env.md`
- `knowledge/cases.md`, `knowledge/anti-patterns.md` (priors / dedup)

## Tools called

```python
constraint.check(plan, cluster_profile)
constraint.estimate_mem(plan)
constraint.check_env(plan.env.diff, baseline)
```

`replan.md` does NOT call `submit.run` — that's `execute.md`.

## Procedure (7 steps)

### Step 1 — Pick derivation source(s)

```python
policy = plan_graph.metadata.policy_for_next_replan   # set by settle.md

if policy == "exploit":
    sources = [plan_graph.champion]
elif policy == "explore":
    sources = top_k(plan_graph.frontier - {champion}, key=tps, k=1)
else:  # "explore_exploit"
    sources = [plan_graph.champion, top_1(shelved frontier)]
```

`policy` decisions live in `settle.md` (`stagnation` /
`rounds_since_explore` rules); this file just consumes.

### Step 2 — Map bottleneck to skills + axes

```python
recommended_skills = diagnosis_report.recommended_skills
candidate_axes    = diagnosis_report.candidate_axes   # already taxonomy-tagged
```

If `recommended_skills` references files not yet read, read them now.
Each sub-skill describes (a) the axes it touches and (b) **expected
gain** ranges for each axis change family (used in Step 4).

### Step 3 — Generate candidate plans

For each `(source, axis)` pair, generate one candidate per `value` in
`candidate_axes[axis].candidates`:

```python
for src in sources:
    for axis_spec in candidate_axes:
        for value in axis_spec.candidates:
            cand = derive(src, axis_spec.axis, value)
            cand.generated_by = {
                bottleneck: diagnosis_report.bottleneck,
                strategy:   <which optimization/* skill produced this candidate>,
                axis_change: {key: axis_spec.axis, from: src.value(axis), to: value,
                              type: axis_spec.type},
            }
            cand.id = f"r{N}_p{i++}"
            candidates.append(cand)
```

`derive(src, axis, value)`: copy `src`'s plan; set `axis` to `value`;
recompute `comm.bucket_size_mb` if it depends on world_size; renew
`predicted` via `execution-model/SKILL.md`.

### Step 4 — Predict + score priority

For each candidate:

```python
predicted_tps        = execution_model.predict_tps(cand)        # see execution-model/
predicted_mem_gb     = execution_model.predict_mem(cand)
confidence           = execution_model.confidence(cand)         # 0..1

predicted_gain_pct   = (predicted_tps / champion.tps - 1) * 100
est_cost_gpu_h       = execution_model.predict_cost(cand)        # ~ time × gpus

novelty_bonus        = 1.20 if axis_change unseen around src else 1.00
parent_stability_bonus = 1.10 if src in champion_history (≥2 rounds) else 1.00

priority = (max(predicted_gain_pct, 0.1) * confidence
            / max(est_cost_gpu_h, 0.05)
            * novelty_bonus
            * parent_stability_bonus)
```

The `max(..., 0.1)` and `max(..., 0.05)` floors keep priority finite
when prediction shows zero gain or near-zero cost; they're tunable in
LEARN, not here.

### Step 5 — Filter

Reject any candidate where:

```python
not constraint.check(cand, cluster).valid                      # parallelism illegal
constraint.estimate_mem(cand).mem_gb > 0.92 * cluster.hbm_capacity_gb
not constraint.check_env(cand.env.diff, baseline).valid        # env incompatible
(cand.parent, cand.axis_change.key, cand.axis_change.to) in plan_graph.exhausted_neighborhoods
matches_anti_pattern(cand, knowledge/anti-patterns.md)
```

Each rejection is recorded with `reason` (mirrors §8.10
`selection.rejected[]`).

### Step 6 — Pick execution strategy

```python
selected_strategy = strategy_for(
    diagnosis_report.candidate_axes,
    target_vector.budget,
    plan_graph.metadata,
)  # see execution_strategy.md
```

Possible values: `Champion-Challenger` (default),
`Per-Plan Local Sweep`, `Successive Halving`.

### Step 7 — Select top-K and emit CandidatePool

```python
top_k = strategy.k_for(selected_strategy, budget_remaining)
selected = sort_by_priority_desc(candidates)[:top_k]
```

Always include at least 1 `tag: explore` candidate when
`policy in {"explore", "explore_exploit"}`; if none exists in the
top-K list, swap in the highest-priority explore candidate.

## CandidatePool (output schema)

Mirrors `pilot/README.md` §8.10:

```yaml
candidate_pool:
  generated_at: round_3
  derived_from:
    primary:   r2_p4
    secondary: [r1_p3]               # if explore round
  policy: explore_exploit

  candidates:
    - id: r3_p1
      parent: r2_p4
      axis_change: {key: runtime.mbs, from: 2, to: 3, type: structural}
      predicted_tps: 18800
      predicted_gain_pct: 6.8
      confidence: 0.82
      est_cost_gpu_h: 0.4
      novelty_bonus: 1.0
      stability_bonus: 1.10
      priority: 1.55
      tag: exploit
    - id: r3_p2
      ...
      tag: exploit
    - id: r3_p3
      parent: r1_p3
      axis_change: {key: parallelism.ep, from: 4, to: 8, type: structural}
      predicted_tps: 17900
      confidence: 0.55
      est_cost_gpu_h: 0.5
      novelty_bonus: 1.20
      priority: 0.78
      tag: explore

  selection:
    strategy: Champion-Challenger
    pick_top_k: 3
    selected: [r3_p1, r3_p2, r3_p3]
    rejected:
      - {id: r3_p_x, axis_change: {comm.bucket_size_mb: 32→64},
         reason: "exhausted_neighborhoods around r0_p0"}
      - {id: r3_p_y, axis_change: {parallelism.pp: 2→4},
         reason: "constraint.estimate_mem 210GB > 192GB cap"}

  priority_formula: |
    priority = max(predicted_gain_pct, 0.1)
             * confidence
             / max(est_cost_gpu_h, 0.05)
             * novelty_bonus
             * parent_stability_bonus
```

Persist as `state/candidate_pools/r{N}.yaml`. Each `selected` plan also
writes its full `Plan` schema to `state/plans/<plan_id>.yaml`.

## State written

```yaml
state/candidate_pools/r{N}.yaml: <CandidatePool>
state/plans/r{N}_p{i}.yaml:      <Plan>      # one per selected candidate
state/plan_graph.yaml.nodes:
  r{N}_p{i}: {parent, derived_axis, status: pending, plan_ref, ...}
```

## Exit conditions

- **success**: `selected.length ≥ 1`, all selected pass all
  constraints. → EXECUTE.
- **soft_fail**: `selected.length == 0` because all candidates were
  rejected. Set `replan.escalate = true`; `to: SETTLE` with
  `decision: continue` and `force_envsweep = true` (see
  `state_machine.md` reentry edges) — escape the local optimum via
  env axis instead of structural.
- **hard_fail**: predicted_tps for ALL candidates is below champion ×
  0.5 (model says nothing structural will help). → SETTLE with
  `decision: stop, reason: replan_dry`.

## On_fail

| Condition | Transition |
|-----------|------------|
| `execution-model` predict raises (e.g. coefficients missing for a new axis) | `to: PROJECTION` for re-fit; record axis as `coverage_gap` for LEARN |
| `constraint.check_env` raises (catalog gap) | `to: ABORT` `kind: UNKNOWN`, escalate; the env catalog needs fixing |

## Cross-links

- Source-pick policy → `settle.md`
- Bottleneck → axis families → `optimization/<bottleneck>/SKILL.md`
- Axis types → `axis_taxonomy.md`
- Strategy choice → `execution_strategy.md`
- Predictions → `execution-model/SKILL.md`
- Filters → `constraints/SKILL.md`
- Anti-pattern dedup → `knowledge/anti-patterns.md`

## Scale-aware notes

The 7-step procedure is scale-independent. Defaults differ:

- **Single node**: candidate pool size typically ≤ 5; structural
  axes dominate (mbs, recompute, tp). EnvSweep is rare in the first
  rounds.
- **Multi node**: candidate pool size 4-8; communication-related
  axes (`bucket_size_mb`, `comm.overlap`, `NCCL_BUFFSIZE`) appear
  often; `EnvSweep` is triggered more frequently due to env
  sensitivity.
- **Top-K** at both scales: default `k = 3`; reduced to `k = 2` when
  `target_vector.budget.gpu_h_remaining < 1.5 × est_cost_per_plan`.
