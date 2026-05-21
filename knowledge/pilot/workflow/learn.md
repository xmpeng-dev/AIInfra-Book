# learn — Knowledge Writeback

## Purpose

The closing stage of every successful session. Take the
`learn_inputs` from `report.md` and update `knowledge/pilot/knowledge/`
files so the next session starts smarter.

LEARN is the **only** stage allowed to modify Skills (specifically
under `knowledge/pilot/knowledge/`). All other stages treat Skills as read-only.

## Inputs

```yaml
report:                state/report.yaml             # learn_inputs section
plan_graph:            state/plan_graph.yaml
final_snapshot:        state/snapshots/<champion>.yaml
session_id:
cluster_profile:
execution_model:
```

## Skills consulted (read)

- `knowledge/SKILL.md` (write protocol; what counts as
  pattern / case / anti-pattern)
- `knowledge/cases.md`, `knowledge/patterns.md`,
  `knowledge/anti-patterns.md` (existing entries; for dedup)

## Tools called

```python
knowledge.write(report, kind) -> written_paths
state.checkpoint(tuning_state)
```

`knowledge.write` writes append-only YAML blocks under the appropriate
sub-file; never overwrites existing entries.

## What goes where

```
patterns.md         General rules of thumb that hold across sessions.
                    Format: "When <observation>, <action> is +X%."
                    Example: "MoE intra-node ep≤8 + comm.overlap=true is
                              +30% TPS on MI300X-class clusters."
                    Source: cross-session statistics + this session's
                            promotions that match prior champions.

cases.md            Concrete (model_family, scale, cluster) → champion
                    config records. Used by PROJECTION's knowledge_hint.
                    Format: full Plan + final metrics + brief notes.
                    Source: this session's final champion if outcome=='success'.

anti-patterns.md    Things to NOT try, with concrete reason.
                    Format: "(model_family, scale) + axis_change → <failure_kind>".
                    Source: this session's `dead` plans, EnvSweep
                            no_winner combos, drift events, soft-rollbacks.
```

## Procedure

### Step 1 — Compute writes from report

```python
plans_to_write = {}

# patterns: cross-session rules
for plan in champion_history:
    matched = match_against_existing_patterns(plan, knowledge.patterns)
    if matched:
        increment_pattern_support(matched, plan)
    else:
        candidate_pattern = abstract_pattern(plan)        # generalize axis_change
        if has_strong_evidence(candidate_pattern, this_session, prior_sessions):
            plans_to_write["patterns.md"].append(candidate_pattern)

# cases: this session's winner (if success)
if report.target_status.overall_outcome == "success":
    plans_to_write["cases.md"].append({
        case_id: <derived from session_id + model_family>,
        model_family: <from model_spec>,
        cluster: cluster_profile.version,
        target_summary: target_vector_summary,
        final_plan: report.final.plan,
        final_metrics: report.final.metrics,
        decision_trace_ref: state/report.yaml#decision_trace,
        learned_at: <ts>,
    })

# anti-patterns: dead plans + sweep losers + drift rollbacks
for dead_plan in plans_dead_with_lessons:
    plans_to_write["anti-patterns.md"].append({
        when: {model_family, scale_band, parent_axes_subset},
        action: dead_plan.derived_axis,
        outcome: dead_plan.failure_kind,
        evidence: [dead_plan.snapshot.error or .early_stop_reason],
        first_seen: session_id,
    })

# env_baseline upgrades (cluster_shared promotions)
for promo in report.learn_inputs.env_promoted_to_baseline:
    plans_to_write["cluster_profile_diff"] = promo
```

### Step 2 — Dedup against existing knowledge

Before writing, scan existing entries for:

- Exact duplicates → drop (writes are no-ops).
- Conflicting patterns (same `when`, different `action`) → mark
  the new entry with `conflict_with: <existing_id>` so a human can
  reconcile in a future review.
- Anti-pattern that contradicts a `cases.md` champion → flag for
  review, do NOT auto-resolve.

### Step 3 — Apply writes (transactional)

```python
for sub_file, entries in plans_to_write.items():
    if sub_file == "cluster_profile_diff":
        update_cluster_profile_baseline(entries)         # bumps version
    else:
        knowledge.write(entries, kind=sub_file_to_kind(sub_file))
```

If any write fails partway:

- Roll back successful writes within this LEARN call (`knowledge.write`
  must support a transaction id).
- Record the failure in `state/learn_failure.yaml`; do NOT escalate
  to ABORT — the session was successful, knowledge writeback failure
  is recoverable later.

### Step 4 — Recalibration recommendations (do NOT apply)

If `report.calibration_notes.recommend_recalibration == true`:

- Append a `recalibration_recommended` event to
  `state/execution_model.yaml.events`.
- Do NOT modify ExecutionModel coefficients automatically; the next
  session's PROJECTION can decide whether to re-fit (if cache-stale)
  or honor the recommendation explicitly.

### Step 5 — Final checkpoint

```python
state.checkpoint(tuning_state)        # marks LEARN done
session_status = "DONE"
```

## State written

```yaml
state/learn_summary.yaml:
  session_id: <id>
  written:
    patterns:        [<entry_id>, ...]
    cases:           [<case_id>]      # 0 or 1
    anti_patterns:   [<entry_id>, ...]
  cluster_profile_promoted:
    version_bumped_to: <new_version>
    keys: [...]
  recalibration_recommended: true | false
  failures: []                         # any failed write events
  deduped: <count>
  conflicts_flagged: <count>
```

### Skill-side appends (the actual knowledge mutation)

These go into `knowledge/pilot/knowledge/<file>.md` as appended YAML
blocks, NOT inline in this state file:

- `knowledge/patterns.md`     ← new patterns
- `knowledge/cases.md`        ← this session's champion (if success)
- `knowledge/anti-patterns.md`← new anti-patterns

The exact append format is defined in `knowledge/SKILL.md`.

## Exit conditions

- **success** → DONE.
- **no_writes** (nothing new to learn): still → DONE; record
  `written: {}`.
- **partial_failure** (some writes failed): record in
  `state/learn_failure.yaml`, still → DONE. Do NOT loop or escalate.

## On_fail

| Condition | Action |
|-----------|--------|
| `knowledge.write` raises | rollback within transaction; record failure; → DONE. |
| Cluster profile promotion conflicts with concurrent session | record `conflict_with`; do NOT promote; → DONE. |
| Catastrophic State corruption (cannot read report) | `to: ABORT` `kind: UNKNOWN`, escalate. |

## What LEARN must NOT do

- Modify any non-`knowledge/` skill.
- Change `state_machine.md` rules.
- Auto-update ExecutionModel coefficients (only flag for next
  PROJECTION).
- Promote `weakly_local` or `strongly_local` env values to
  `cluster.env_baseline`. Only `cluster_shared` promotions are valid.
- Delete prior knowledge entries. Append-only.

## Cross-links

- Inputs source → `report.md` § learn_inputs
- Knowledge file conventions → `knowledge/SKILL.md`
- Axis type rules for promotion → `axis_taxonomy.md`
- ExecutionModel events / drift handling → `execution-model/SKILL.md`

## Scale-aware notes

LEARN's writeback rules are scale-independent. What differs is
**which `knowledge/` files get most action**:

- **Single node**: most appends land in `cases.md` (per-model
  configs); `patterns.md` rarely gains new entries because
  intra-node behavior generalizes well.
- **Multi node**: `patterns.md` and `cluster_profile` env_baseline
  promotions are common (RCCL config tends to stick across
  sessions on the same physical cluster); `anti-patterns.md` also
  grows faster (more failure modes available).
