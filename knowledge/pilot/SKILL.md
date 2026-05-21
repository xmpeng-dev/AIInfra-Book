---
name: pilot
description: >-
  Auto-tune training jobs (Dense / MoE bring-up, scaling regression, parallel +
  comm joint tuning). Drives a state-machine workflow — Preflight → Projection
  → Smoke → Baseline → Correctness → OPTIMIZE_LOOP { Observe → Diagnose →
  Re-Plan → Execute → Correctness-lite → EnvSweep? → Settle } → Report → Learn
  — backed by a PlanGraph search tree, axis taxonomy, env tuning inner loop,
  and explicit on_fail / reentry rules. Read this skill when an Agent needs to
  optimize TPS / memory / cost of a training run on any cluster shape (single
  node or multi-node, same workflow). Sub-skills under workflow/, execution-
  model/, optimization/, env/, profiling/, constraints/, knowledge/.
---

# Pilot — Training Auto-Tuning Skill

> **Single source of truth: the design document at `pilot/README.md`** in
> the parent project. This skill operationalizes that document for an
> Agent. Whenever this skill and the design document disagree, the
> design document wins; raise a follow-up to fix the skill.

## When to apply

Use this skill when **any** of:

- The user wants to **auto-tune a training run** (TPS / memory / cost /
  bubble) on a known cluster.
- The user is doing **bring-up** of a new model or new cluster and wants
  to discover a viable initial configuration.
- The user is investigating a **scaling regression** (worse-than-linear
  with cluster size).
- The user wants **joint parallelism + communication tuning** instead of
  one-axis-at-a-time manual sweeps.

Do **not** use this skill for:

- Modifying kernels, model architecture, or communication library
  internals (see `cuda_*` / `amd-gemm-optimization` skills instead).
- Inference / serving optimization.
- Tasks that don't have a measurable training step (e.g. data prep
  pipelines).

## Workflow at a glance

```
PREFLIGHT → PROJECTION → SMOKE → BASELINE → CORRECTNESS
   → OPTIMIZE_LOOP { Observe → Diagnose → Re-Plan → Execute
                     → CORRECTNESS-LITE → EnvSweep? → Settle }*
   → REPORT → LEARN
```

The loop is **double-layer**: the outer loop varies *structural* axes
(parallelism, mbs, recompute); the optional inner `EnvSweep` varies
*environment* axes (RCCL, HSA, allocator) on a locked structure.

State is persisted via `state.checkpoint(...)` at every stage exit. Any
failure routes through `constraint.diagnose_failure(...)` → a
`FailureReport` → `state_machine.md` decides the next stage.

## How to use (Agent operational guide)

The Agent **always**:

1. Reads `workflow/SKILL.md` first to learn the stage order.
2. Reads `workflow/state_machine.md` for transition / on_fail / reentry
   rules. This is the **only authoritative source** for "after stage X,
   go to stage Y".
3. For each stage `S`, reads `workflow/<S>.md` to learn the contract:
   inputs, Skills consulted, Tools called, State writes, exit conditions.
4. Reads `optimization/<bottleneck>/SKILL.md` only when `Diagnose`
   produces that bottleneck.
5. Reads `execution-model/SKILL.md` once at PROJECTION; consults
   `compute.md` / `memory.md` / `communication.md` only on demand
   (prediction or OOM check).
6. Reads `env/SKILL.md` only when `EnvSweep` is triggered or when
   `env_suspect` appears in a `DiagnosisReport`.
7. Reads `constraints/oom.md` and `constraints/config.md` before any
   `submit.run(...)` to reject infeasible plans cheaply.
8. Writes State via Tool calls (`state.checkpoint`); never mutates YAML
   files directly.
9. **Stops and asks the user** if a `FailureReport` of kind
   `NUMERICAL` / `UNKNOWN` / `BUDGET_EXCEEDED` is raised, or if the
   PlanGraph stagnates before any constraint in `TargetVector` is met.

## Sub-skill index

| Path | Purpose | When read |
|------|---------|-----------|
| `workflow/SKILL.md` | Stage map + invocation order | Always first |
| `workflow/state_machine.md` | Transitions / reentry / on_fail | Always second |
| `workflow/preflight.md` | Cluster baseline + env probe | Stage 1 |
| `workflow/projection.md` | Build Execution Model + initial plans | Stage 2 |
| `workflow/smoke.md` | Tiny-scale boot test | Stage 3 |
| `workflow/baseline.md` | First full-scale run | Stage 4 |
| `workflow/correctness.md` | Loss-vs-reference gate | Stage 5 + per-round (lite) |
| `workflow/observe.md` | Snapshot schema + collection | Each loop iter |
| `workflow/diagnose.md` | Bottleneck classification | Each loop iter |
| `workflow/plan.md` | Plan schema (incl. env.diff) | Each loop iter |
| `workflow/plan_graph.md` | Search-tree data model | Re-Plan / Settle |
| `workflow/replan.md` | CandidatePool generation + priority | Each loop iter |
| `workflow/axis_taxonomy.md` | cluster_shared / weakly_local / strongly_local | Re-Plan |
| `workflow/execution_strategy.md` | Champion-Challenger / Per-Plan / Halving | Re-Plan |
| `workflow/execute.md` | Submission + early stop | Each loop iter |
| `workflow/envsweep.md` | Inner env sweep | Conditional |
| `workflow/settle.md` | PlanGraph maintenance + stop conditions | Each loop iter |
| `workflow/report.md` | Final config + decision trace | Stage 7 |
| `workflow/learn.md` | Knowledge writeback | Stage 8 |
| `execution-model/*` | Time / memory / comm formulas | PROJECTION + OOM checks |
| `optimization/<b>/*` | Bottleneck-specific recipes | After Diagnose |
| `env/*` | Env catalog (single source of truth per flag) | EnvSweep / env_suspect |
| `profiling/*` | Data collection protocols | PREFLIGHT + per-job |
| `constraints/*` | Hard validation + failure diagnosis | Before any submit + on failure |
| `knowledge/*` | Past best configs / patterns / anti-patterns | Re-Plan (consult), LEARN (write) |

## Tool contract

The Agent calls Tools (Python functions). Canonical list, mirroring
`pilot/README.md` §5:

```
preflight.run(cluster_id) -> ClusterProfile
env_probe.run(cluster_id, candidate_envs) -> EnvBaseline       # writes ClusterProfile.env_baseline
env_probe.sweep(base_plan, candidate_envs, max_steps) -> {best_env_diff, results}
profiler.run(model_spec, configs) -> ProfilingResult
submit.run(plan, scale={...}) -> job_id
submit.cancel(job_id) -> status
observe.snapshot(job_id) -> Snapshot
observe.compare_loss(job_id, reference_curve) -> {pass, drift, delta_pct}
constraint.check(plan, cluster) -> {valid, violations}
constraint.check_env(env_diff, baseline) -> {valid, violations}
constraint.estimate_mem(plan) -> {mem_gb, breakdown, confidence}
constraint.diagnose_failure(snapshot|error) -> FailureReport
state.checkpoint(tuning_state) -> path
state.resume(path) -> tuning_state
knowledge.write(report, kind) -> written_paths
```

Backend adapters (e.g. Primus, Megatron, TorchTitan) live under
`tools/submit_<backend>.py` and conform to the `submit.run` signature
above; the Agent does not need to know which backend is wired.

## Schemas

All YAML schemas referenced by sub-skills mirror `pilot/README.md` §8:

`ClusterProfile (§8.1)` · `Plan (§8.2)` · `Snapshot (§8.3)` ·
`DiagnosisReport (§8.4)` · `EnvSweepResult (§8.5)` ·
`TargetVector (§8.6)` · `TuningState (§8.7)` ·
`FailureReport (§8.8)` · `PlanGraph (§8.9)` · `CandidatePool (§8.10)`.

Sub-skills repeat the relevant fields they read or write; the canonical
definition is the design document.

## Conventions

- **Front-matter only on this file.** Sub-files are plain Markdown so
  they aren't registered as separate skills.
- **One canonical definition per concept.** Sub-files cross-reference;
  numeric thresholds live next to the rule that uses them.
- **State is the only shared memory.** Tools are stateless; all cross-
  stage information flows through State Layer YAML.
- **Skills are knowledge, not logic.** Every "if X then Y" rule (e.g.
  state_machine.md, diagnose.md thresholds) is documented in Markdown;
  the Agent is the executor, not the rule owner.
- **Scale-aware notes.** Where single-node and multi-node behavior
  diverge meaningfully, sub-files include a `Scale-aware notes` section
  with `single_node` / `multi_node` callouts. The workflow itself is
  identical; only thresholds, priorities, and default candidate sets
  may differ.
