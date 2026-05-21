# state_machine — Transitions, Reentry, On-Fail

This file is the **only authoritative source** for "after stage X, go
to stage Y" rules. No other skill file should contain transition logic;
if you need a rule, add it here. Mirrors `pilot/README.md` §3.2 and
§12.2.

## State set

```yaml
states:
  - PREFLIGHT
  - PROJECTION
  - SMOKE
  - BASELINE
  - CORRECTNESS
  - OPTIMIZE_LOOP:
      sub_states:
        - OBSERVE
        - DIAGNOSE
        - REPLAN
        - EXECUTE
        - CORRECTNESS_LITE
        - ENVSWEEP        # optional
        - SETTLE
  - REPORT
  - LEARN
  - DONE
  - ABORT                  # terminal, requires user attention
```

## Forward transitions (happy path)

```yaml
PREFLIGHT       → PROJECTION
PROJECTION      → SMOKE
SMOKE           → BASELINE
BASELINE        → CORRECTNESS
CORRECTNESS     → OPTIMIZE_LOOP.OBSERVE
OBSERVE         → DIAGNOSE
DIAGNOSE        → REPLAN
REPLAN          → EXECUTE
EXECUTE         → CORRECTNESS_LITE
CORRECTNESS_LITE→ ENVSWEEP        # if envsweep.md says triggered
                  | SETTLE        # otherwise
ENVSWEEP        → SETTLE
SETTLE          → OBSERVE         # if continue
                  | REPORT        # if stop
REPORT          → LEARN
LEARN           → DONE
```

## Reentry edges (rare but legal jumps backward)

A later stage may discover that an earlier stage's State output is
stale or invalid. Reentry MUST be requested via
`FailureReport.suggested_transition` (schema in `constraints/SKILL.md`,
mirroring `pilot/README.md` §8.8). Reentries are **not counted against
`budget.max_rounds`**.

| From | Condition (`reentry_when`) | To | Effect |
|------|---------------------------|----|----|
| any | `cluster_profile.age > 7d` (auto-checked at OBSERVE) | PREFLIGHT | re-probe; old `env_baseline` retired |
| any | `cluster_profile.driver_changed` | PREFLIGHT | full re-probe; force `version` bump |
| DIAGNOSE | `model_spec` changed since PROJECTION | PROJECTION | re-derive ExecutionModel; PlanGraph reset |
| REPLAN | All candidates filtered by exhausted_neighborhoods AND PlanGraph stagnates | OBSERVE (with `force_envsweep=true`) | exit local optimum via env axis instead of structural |
| any (EXECUTE) | `FailureReport.kind = HANG` | PREFLIGHT (env_probe only) | suspect env baseline drift |

## On-fail transitions

`FailureReport.kind` (from `constraint.diagnose_failure(...)`) drives
recovery. `counts_against_budget` controls whether the failed attempt
consumes a Settle round. Mirrors `pilot/README.md` §12.2.

| `failure_kind` | Default `to` | `counts_against_budget` | Notes |
|---------------|-------------|------------------------|-------|
| `OOM` | REPLAN | false | Mark plan `dead`. Penalize ExecutionModel mem prediction by recorded delta. |
| `HANG` | PREFLIGHT (env_probe subset) | false | Re-probe RCCL / xGMI / alloc; if clean, retry once. |
| `INVALID_CONFIG` | REPLAN | false | Drop plan before EXECUTE; `constraint.check` should have caught this — open a follow-up. |
| `NUMERICAL` | ABORT + escalate | true | Loss NaN / drift > tolerance. STOP and ask user. Never auto-recover. |
| `CLUSTER` | PREFLIGHT | false | Node down, driver error. Mark `cluster_profile.status = stale`. |
| `BUDGET_EXCEEDED` | REPORT | n/a | Early termination; emit current champion. |
| `STRUCTURAL_INVALIDATION` | PROJECTION | true | Model / dataset shape changed. Wipe ExecutionModel cache and PlanGraph. |
| `UNKNOWN` | ABORT + escalate | true | Do not guess. Hand to user with full FailureReport attached. |

## Reentry guard (anti-loop)

To prevent infinite re-probe / re-project loops:

```yaml
reentry_guard:
  PREFLIGHT:
    max_in_session: 3
    cooldown_min: 10            # don't re-probe more than once per 10 min
  PROJECTION:
    max_in_session: 2
  ABORT_after_n_consecutive_failures:
    same_kind: 3                # 3 OOMs in a row on different plans → ABORT
```

If a reentry guard trips, raise an `ABORT` with `failure_kind: UNKNOWN`
and `escalate=true`.

## Stagnation handling (escape local optimum)

This is **not a failure**, just a rule to prevent the search from
looping in a tight neighborhood. Owned by `settle.md` (decision) and
consumed here (transition).

| Condition | Action |
|-----------|--------|
| `rounds_since_promotion ≥ 2` | Set `replan.policy = explore` (Re-Plan picks from shelved). |
| `rounds_since_explore ≥ 3` | Force one explore round regardless. |
| `dead_rate_in_subtree[champion] > 0.5` after 2 rounds | Backtrack: champion ← frontier next-best. |

Stagnation rules count toward `budget.max_rounds`.

## Persistent identifiers

- `session_id`: assigned at PREFLIGHT, never changes (used for State
  paths).
- `cluster_profile.version`: changes only when PREFLIGHT writes a new
  one.
- `plan_id`: monotonic per session, format `r{round}_p{idx}`.
- `sweep_id`: monotonic per session, format `r{round}_envsweep_{idx}`.

## Mode hooks (out of scope for this skill, documented for future)

If a future skill needs a different mode (e.g. RL post-training with
two loops), it adds a new top-level state and edges here. This skill's
loop runs in mode `single_loop_train`.

## Where to look for actions, not transitions

| Question | File |
|----------|------|
| How to choose between candidates | `replan.md` and `execution_strategy.md` |
| How to compute confidence in a prediction | `execution-model/SKILL.md` |
| How to detect a bottleneck | `diagnose.md` |
| How to know we've converged | `settle.md` § Stop conditions |
| How `FailureReport` is constructed | `constraints/validation.md` |

## Scale-aware notes

Transition rules are identical for single-node and multi-node. What
differs is the *frequency* of certain reentries:

- **Single node** rarely triggers `CLUSTER` or `HANG`; expect most
  failures to be `OOM` or `INVALID_CONFIG`.
- **Multi node** triggers `HANG` and `CLUSTER` more often (NCCL/IB
  timeouts, node flakiness); `PREFLIGHT (env_probe subset)` is a
  common reentry target.
