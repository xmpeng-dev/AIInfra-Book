# constraints — Pre-Submit Validation + Failure Diagnosis

## Purpose

Hard rules that **filter plans before they cost wall-time** and
**classify failures into actionable categories** when something does
go wrong. Constraints are the "no-go" guardrails complementing
`optimization/*` (the "go" recipes).

## Tools owned

```python
constraint.check(plan, cluster_profile) -> {valid, violations}
constraint.estimate_mem(plan) -> {mem_gb, breakdown, confidence}
constraint.check_env(env_diff, env_baseline) -> {valid, violations}
constraint.diagnose_failure(snapshot|error) -> FailureReport
```

These are all **pure / fast** (no GPU jobs); always called before
`submit.run`.

## Sub-files

| File | Owns |
|------|------|
| `oom.md` | Memory feasibility rules (the most common rejection) |
| `config.md` | Parallelism / shape legality (e.g. `tp` divides heads, `world_size = tp · pp · dp`) |
| `validation.md` | `constraint.diagnose_failure` mapping (snapshot/error → FailureReport) |
| `env.md` | Env compatibility matrix (which env flags conflict) |

## Where each tool fits

```
PROJECTION
  └─ filter initial plans
     ├─ constraint.check          (structural)
     ├─ constraint.estimate_mem   (oom)
     └─ constraint.check_env      (env baseline + diff)

REPLAN Step 5 (filter)
  └─ same three, plus exhausted_neighborhoods check (replan owns that)

SMOKE / BASELINE / EXECUTE — pre-submit
  └─ same three (defensive; should never fail at this point if
     PROJECTION/REPLAN did their job)

EXECUTE / SMOKE / BASELINE — on failure
  └─ constraint.diagnose_failure → FailureReport → state_machine routing

ENVSWEEP — per candidate
  └─ constraint.check_env (compatibility)
```

## FailureReport schema (canonical)

Defined here, used by `state_machine.md` for routing. Mirrors
`pilot/README.md` §8.8.

```yaml
# constraint.diagnose_failure output
failure_id:           <hash>
diagnosed_at:         <ts>
job_id:               <ref>
plan_id:              <ref>
kind: OOM | HANG | INVALID_CONFIG | NUMERICAL | CLUSTER
      | BUDGET_EXCEEDED | STRUCTURAL_INVALIDATION | UNKNOWN
evidence:
  - "<what was observed>"
suggested_transition:
  to:    <state>                    # see state_machine.md table
  hints:                            # passed forward to the receiving stage
    force_recompute_full: bool
    force_envsweep:       bool
    prefer_axis:          <axis_key>
counts_against_budget: bool
escalate: bool                      # true → ABORT + user attention
```

The `suggested_transition` is **advisory**; `state_machine.md` is the
final authority. A constraint may suggest a transition, but the
state machine's `on_fail` table can override (e.g. anti-loop guards).

## Validation philosophy

- **Cheap to evaluate, conservative in rejection.** False positives
  (rejecting a plan that would have run) cost search-space coverage;
  false negatives (accepting a plan that OOMs) cost ~30 min wall + a
  dead PlanGraph node. Bias toward false positives.
- **Always cite the rule that fired.** A rejection without a citable
  rule is unactionable — Re-Plan can't generate a fix.
- **One rule, one rejection reason.** Multiple violations are listed
  separately, not coalesced. Re-Plan needs to know what to relax.
- **Constraints are versioned.** When `cluster_profile.version` bumps,
  pre-submit constraint outputs become stale; re-evaluate.

## Constraint vs Optimization vs Diagnose

| Question | Owner |
|----------|-------|
| "Will this plan even run?" | `constraints/` (oom, config, env) |
| "Why did this plan fail?" | `constraints/validation.md` (FailureReport) |
| "Why is this plan slow?" | `workflow/diagnose.md` (DiagnosisReport) |
| "How do I make it faster / fit?" | `optimization/<bottleneck>/` |

Don't put speed reasoning in constraints; don't put feasibility
checks in optimization recipes.

## Cross-links

- State-machine on_fail table → `workflow/state_machine.md` § On-fail
- Anti-pattern dedup hits → `knowledge/anti-patterns.md`
- ExecutionModel mem prediction → `execution-model/memory.md`

## Sub-file status

| File | Status |
|------|--------|
| `oom.md` | filled (entry already exists) |
| `config.md` | filled (entry skeleton) |
| `validation.md` | filled (entry skeleton) |
| `env.md` | filled (entry skeleton) |

## Scale-aware notes

| Family | Single node | Multi node |
|--------|-------------|-----------|
| `oom.md` | dominant; intra-node HBM is the constraint | still dominant; sharding offers more headroom |
| `config.md` | small set (no inter-node shapes) | adds `world_size` checks, `pp` × `tp` × `dp` divisibility, ZeRO/FSDP shape rules |
| `validation.md` | rare `CLUSTER` / `STRUCTURAL_INVALIDATION`; mostly `OOM` / `INVALID_CONFIG` | wider failure spectrum; `HANG`, `CLUSTER` regular |
| `env.md` | small (no NCCL_IB cross-checks) | larger (RCCL/IB compatibility matrix) |
