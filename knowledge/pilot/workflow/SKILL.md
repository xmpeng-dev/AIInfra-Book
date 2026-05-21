# workflow — Stage Map + Invocation Contract

This file is the **first sub-skill** an Agent reads after the top
`pilot/SKILL.md`. It enumerates every stage of the Tuning Loop, their
order, their entry / exit contracts, and which sub-file owns each
stage's detailed rules.

It mirrors `pilot/README.md` §3 (system flow). When the design document
changes, this file follows.

## Three views of the workflow

Like `pilot/README.md` §3, there are three layered views of the same
process. Read them in order:

1. **§1 Stage table** below — the canonical list of stages, their
   owner files, and exit criteria.
2. **§2 Block diagram** — the stage-by-stage data flow.
3. **§3 Inner swimlane** — Agent / Skill / Tool roles inside the loop.

## §1 Stage table

| # | Stage | Owner file | Inputs | Outputs (State writes) | Exit conditions |
|---|-------|-----------|--------|----------------------|-----------------|
| 1 | `PREFLIGHT` | `preflight.md` | `cluster_id` | `state/cluster_profile.yaml` (incl. `env_baseline`) | `status: validated` or `tentative` |
| 2 | `PROJECTION` | `projection.md` | `model_spec`, `ClusterProfile` | initial `Plan`s, `ExecutionModel` cache | ≥ 1 plan with `predicted.tps > 0`, confidence ≥ 0.6 |
| 3 | `SMOKE` | `smoke.md` | first plan (tiny scale) | smoke `Snapshot` | `status: success`, no NaN, mem within prediction band |
| 4 | `BASELINE` | `baseline.md` | post-smoke plan (full scale) | baseline `Snapshot`, `plan_graph.r0_p0` | run completed, ≥ baseline_steps_min |
| 5 | `CORRECTNESS` | `correctness.md` | baseline `Snapshot` + reference | pass / fail | `loss_delta_pct ≤ TargetVector.constraints` |
| 6 | `OPTIMIZE_LOOP` | (sub-stages below) | TargetVector + PlanGraph | updated PlanGraph each round | Settle says STOP |
| 6.1 | `Observe` | `observe.md` | `job_id` | `Snapshot` | snapshot collected |
| 6.2 | `Diagnose` | `diagnose.md` | `Snapshot` + ExecutionModel | `DiagnosisReport` | bottleneck assigned, confidence ≥ 0.6 |
| 6.3 | `Re-Plan` | `replan.md` | `DiagnosisReport`, `PlanGraph` | `CandidatePool` | ≥ 1 candidate selected, all pass `constraint.check` |
| 6.4 | `Execute` | `execute.md` | selected candidates | per-plan `Snapshot`s | all jobs finished or early-stopped |
| 6.5 | `CORRECTNESS-LITE` | `correctness.md` (lite mode) | per-plan `Snapshot`s | per-plan pass / drift | drift jobs marked `dead` |
| 6.6 | `EnvSweep` (optional) | `envsweep.md` | champion + `env_suspect` (or forced) | `EnvSweepResult`, possible env_baseline merge | sweep complete or no candidate |
| 6.7 | `Settle` | `settle.md` | round results, PlanGraph | promoted champion / shelved / stop | continue / stop decision |
| 7 | `REPORT` | `report.md` | final PlanGraph | `state/report.yaml` | written |
| 8 | `LEARN` | `learn.md` | report + PlanGraph | appended `knowledge/{patterns,cases,anti-patterns}.md` | `knowledge.write` succeeded |

Stages 1-5 run **sequentially** with `state.checkpoint()` at each exit.
Stage 6 is the loop; sub-stages 6.1-6.7 are sequential within a round.
For all transition / failure / reentry rules, read `state_machine.md`
(this file does NOT define them).

## §2 Block diagram (mirrors README §3.1)

```
┌──────────────────────┐
│      User Input      │
│  - Model Spec        │
│  - Cluster Size      │
│  - TargetVector      │
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│ 1. PREFLIGHT++                                           │
│    GEMM/MFMA peak, IB/xGMI bw, AR/A2A baseline,          │
│    env probe (NCCL/HSA/alloc connectivity + micro-bench) │
│  → ClusterProfile (incl. env_baseline; reusable across   │
│    sessions until version/age expires)                   │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│ 2. PROJECTION / Modeling                                 │
│    Single-node profiling → fit T_comp, M_act             │
│  → ExecutionModel cache + initial Plans (with            │
│    scale-aware env diff)                                 │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│ 3. SMOKE      tiny scale × ~100 step                     │
│ 4. BASELINE   full scale, history[0]                     │
│ 5. CORRECTNESS  loss / grad vs reference                 │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│ 6. OPTIMIZE_LOOP (double-layer)                          │
│                                                          │
│  Outer (structural):                                     │
│    Execute → CORRECTNESS-LITE → Observe → Diagnose       │
│    → Re-Plan { source pick · skill mapping by bottleneck │
│                · CandidatePool · constraints check       │
│                · exhausted_neighborhoods dedup           │
│                · execution_strategy }                    │
│                                                          │
│  Inner (env, optional, after Settle):                    │
│    EnvSweep on locked structure, ≤ 5 flags, ≤ 8 combos   │
│    → merge best diff into baseline                       │
│                                                          │
│  Settle:                                                 │
│    PlanGraph maintenance (champion / shelved / dead /    │
│    frontier / exhausted_neighborhoods),                  │
│    backtrack / explore-round / stop conditions           │
└──────────┬───────────────────────────────────────────────┘
           ▼
   not converged ◄──► converged
       │                 │
   round++               ▼
                ┌────────────────────────┐
                │ 7. REPORT              │
                │   final config + trace │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │ 8. LEARN               │
                │   write knowledge/     │
                └────────────────────────┘

Reentry edges (do NOT count against round budget):
  Diagnose detects ClusterProfile stale  → PREFLIGHT
  Re-Plan detects structural invalidation → PROJECTION
  HANG (NCCL/IB timeout)                  → PREFLIGHT (env_probe)
  CLUSTER (node down / driver error)      → PREFLIGHT
  OOM / INVALID_CONFIG                    → Re-Plan (mark plan dead)
  NUMERICAL drift (CORRECTNESS-LITE)      → ABORT + escalate
```

## §3 Inner swimlane (mirrors README §3.3)

The Tuning Loop expands into Agent / Skill / Tool roles:

```
Agent (reasoning)         Skill (knowledge)             Tool (execution)
──────────────────        ─────────────────             ─────────────────
[Execute]
  └─ read execute.md ► early_stop, scheduling
  └─ call ─────────────────────────────────────────► submit.run(plan)

[CORRECTNESS-LITE]
  └─ read correctness.md ► loss_delta_pct
  └─ call ─────────────────────────────────────────► observe.compare_loss()

[Observe]
  └─ read observe.md ► snapshot schema
  └─ call ─────────────────────────────────────────► observe.snapshot(job_id)

[Diagnose]
  └─ read diagnose.md ► classification rules
  └─ Agent infers bottleneck

[Re-Plan]
  └─ read optimization/<bottleneck>/SKILL.md
  └─ read axis_taxonomy.md, execution_strategy.md
  └─ call ─────────────────────────────────────────► constraint.check(plan)
                                                   ► constraint.estimate_mem(plan)

[EnvSweep] (conditional)
  └─ read env/SKILL.md, env/<area>.md
  └─ read optimization/<bottleneck>/env.md
  └─ call ─────────────────────────────────────────► constraint.check_env()
                                                   ► env_probe.sweep()

[Settle]
  └─ read settle.md ► promotion + stop rules
  └─ Agent decides continue / stop
```

## Per-stage Skill / Tool contract (template)

Every stage file (e.g. `projection.md`) follows the same template:

```markdown
# <Stage>

## Inputs            (which State / TargetVector keys are read)
## Skills consulted  (which other Markdown files this stage reads)
## Tools called      (which Python functions, with arg schema)
## State written     (which YAML files / keys this stage writes)
## Exit conditions   (success / soft_fail / hard_fail criteria)
## On_fail           (which transition is requested via FailureReport;
                      details in state_machine.md)
## Notes             (optional)
## Scale-aware notes (optional; single_node / multi_node callouts)
```

Agents must respect this contract; they MAY NOT call tools or write
state keys outside what the stage file authorizes.

## Why a state machine, not a linear pipeline

Quoted from `pilot/README.md` §3.2:

> A state machine lets us add new stages, jumps, or operating modes
> (e.g. RL post-training with two loops) by adding nodes and edges,
> without rewriting the main swimlane. It also makes `reentry_when`
> (when to jump back) and `on_fail` (where to recover) explicit, so
> the Agent's decisions follow rules rather than ad-hoc judgment.

`state_machine.md` is the canonical source for these rules.

## Why outer / inner separation

> Env tuning doesn't change shape (no OOM risk) and a single trial is
> cheap (30-50 steps). It fits a "safe sweep after each outer baseline
> stabilizes". But the optimal env value depends on structure (mbs,
> world_size), so it can't be tuned once and forgotten — it must be
> embedded in each outer round.

`envsweep.md` defines the trigger conditions and sweep protocol.

## Scale-aware notes

The 9-stage workflow is identical regardless of cluster shape. Where
defaults differ:

- **Single node**: SMOKE typically uses 1 layer × mbs=1 × 50 steps;
  COMM_BOUND threshold is tighter (intra-node xGMI is cheap, so > 20%
  is suspicious); EnvSweep candidate set focuses on
  `PYTORCH_HIP_ALLOC_CONF` and HSA flags.
- **Multi node**: SMOKE typically uses ≥ 1 node × ≥ 100 steps to also
  validate cross-node RCCL paths; COMM_BOUND threshold is looser
  (~25%) because IB AR/A2A is expected to dominate; EnvSweep
  prioritizes RCCL flags (`NCCL_BUFFSIZE`, `NCCL_MIN_NCHANNELS`,
  `NCCL_IB_*`).

These are tuning of defaults, not workflow changes. Each stage file
records its own scale-aware notes in the dedicated section.
