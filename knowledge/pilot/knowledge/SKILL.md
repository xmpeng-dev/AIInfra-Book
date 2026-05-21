# knowledge — Persisted Cross-Session Learnings (Top Entry)

## Purpose

Long-lived, append-mostly knowledge base that lets each new tuning
session start smarter. Written by `workflow/learn.md`; read by
`workflow/projection.md`, `workflow/replan.md`,
`workflow/diagnose.md`, and `constraints/*`.

This directory is the **only** Pilot state that survives across
sessions. `state/` is per-session (and resumable within session);
`knowledge/` is global.

## Files

| File | Owns | Read by |
|------|------|--------|
| `patterns.md` | reusable, high-confidence patterns ("if X observed → try Y") | `replan.md`, `diagnose.md` |
| `cases.md` | concrete past sessions: model + cluster + winning Plan + delta | `projection.md` (warm-start), `replan.md` (priors) |
| `anti-patterns.md` | known-bad combinations and confirmed dead branches | `constraints/oom.md`, `constraints/env.md`, `replan.md` (negative priors) |
| `cluster_baselines.md` | per-cluster `env_baseline` snapshots and recommended defaults | `preflight.md` (warm-start ClusterProfile) |

Sub-files are stubbed / grown by LEARN; not all have content from
day one.

## Entry format

Every entry is a YAML block prefixed with provenance metadata, so it
is unambiguously linkable from sessions:

```yaml
- id: <stable_uuid_or_slug>
  added_at: <iso8601>
  added_by: session_id=<...>, plan_id=<...>
  scope:
    model_arch: dense | moe | any
    model_size_band: <e.g. 7B-30B | 30B-100B | 100B+ | any>
    cluster_shape: single_node | multi_node | any
    interconnect_class: nvlink | xgmi | ib | rocev2 | any
    other: {...}
  evidence:
    - {snapshot_id: ..., metric: ..., value: ...}
  rule: |
    <prose statement of the pattern / case / anti-pattern>
  confidence: low | medium | high
  reproductions: <int>      # how many sessions have re-confirmed it
  last_seen: <iso8601>
```

Consumers use `scope` to filter relevance; `confidence` and
`reproductions` to weight priors.

## Read paths (where each consumer pulls)

### projection.md (warm-start)

```python
relevant_cases = knowledge.cases.filter(scope, top_k=5)
seed_plans = derive_initial_plans_from(model_spec, cluster_profile,
                                       priors=relevant_cases)
```

### replan.md (priority bonus)

```python
for cand in candidates:
    if matches_pattern(cand, knowledge.patterns):
        cand.priority *= pattern.priority_multiplier
    if matches_anti_pattern(cand, knowledge.anti_patterns):
        cand.dropped = true
        cand.reason = "matches anti-pattern <id>"
```

### diagnose.md (fingerprint matching)

```python
for past_case in knowledge.cases:
    sim = fingerprint_similarity(snap, past_case.snapshot_fingerprint)
    if sim > 0.85:
        diagnosis.context_hint = past_case.id
```

### constraints/*

```python
# oom.md: anti-patterns inform the rejection priors
# env.md:   anti-patterns inform the env mutex catalog
```

### preflight.md (cluster_baselines reuse)

```python
hit = knowledge.cluster_baselines.lookup(cluster_signature)
if hit and not args.force_full_probe:
    cluster_profile.env_baseline = hit.env_baseline
    cluster_profile.warm_started = True
```

## Write rules

`learn.md` is the **only** writer. Other stages may not modify
`knowledge/`; they may only emit `learn_inputs` to be picked up at
LEARN.

Updates are **append + dedupe**, never delete. Patterns can be
**superseded** by linking `superseded_by: <new_id>` rather than
removing the old entry. Anti-patterns are sticky: once added, they
require explicit, evidence-backed retraction (a session that
reproduces the supposedly bad combination and shows it works under
a clearly defined condition).

## Confidence promotion / demotion

```yaml
add_with_confidence: low
on_first_reproduction: low → medium
on_third_reproduction: medium → high
on_contradiction: log + open follow-up; do not auto-demote until
                  the contradicting evidence is itself confirmed
                  ≥ 2 times
```

These thresholds are guidance; per-rule overrides allowed in the
entry's metadata (`confidence_policy`).

## Privacy / leak control

Entries in `knowledge/` are repo-visible and assumed public to all
Agent users. Do not include:

- raw user data, secret tokens, customer-specific path patterns
- model weights paths or container registry credentials
- private cluster names — anonymize as `cluster_<hash>` if needed

## Cross-links

- Writer: `workflow/learn.md`
- Final-report inputs: `workflow/report.md` § learn_inputs
- Negative priors / catalog: `constraints/oom.md`, `constraints/env.md`
- Initial-plan generation: `workflow/projection.md`

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Cases growth rate | high (faster sessions) | lower (each session is expensive) |
| Anti-pattern usefulness | medium (limited fabric variety) | high (HCA/topology mistakes are costly) |
| Cluster baselines reuse | very high (boxes are interchangeable) | high but versioned per fabric topology |
| Patterns about env | medium | very high (most cross-session value is here) |
