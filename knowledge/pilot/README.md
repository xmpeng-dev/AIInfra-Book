# Pilot · Skill Pack

This directory operationalizes `pilot/README.md` (the Primus Pilot
design document) as an Agent skill. Read `SKILL.md` if you are an
Agent; read this `README.md` if you are a human contributor.

## Layout

```
skills/pilot/
├── SKILL.md                    # Cursor skill entry (front-matter)
├── README.md                   # this file
│
├── workflow/                   # Stages + state machine + plan/graph schemas
├── execution-model/            # T_comp / Mem / T_comm / Bubble formulas
├── optimization/               # Bottleneck-driven recipes
│   ├── compute/
│   ├── memory/
│   ├── comm/
│   ├── pipeline/
│   └── moe/
├── env/                        # Env catalog (rccl / hsa / alloc / threading / presets)
├── profiling/                  # Preflight + per-job + env probe protocols
├── constraints/                # Hard validation + FailureReport mapping
└── knowledge/                  # patterns / cases / anti-patterns (LEARN target)
```

## Scope

| Aspect | Setting |
|--------|---------|
| Workflow | Same state machine for single-node and multi-node |
| Models | Dense / MoE; bring-up, scaling regression, joint parallel + comm tuning |
| Hardware | Cluster-shape-agnostic; `ClusterProfile` abstracts vendor + topology |
| Frameworks | Backend-agnostic axes; adapters under `tools/submit_<backend>.py` |
| Out of scope | Kernel / model architecture / comm-library implementation; inference / serving |

## Reading order for humans

1. `SKILL.md` — what this skill is and when to apply it.
2. `workflow/SKILL.md` — the stage map + invocation contract.
3. `workflow/state_machine.md` — how stages transition / fail / re-enter.
4. `workflow/plan.md` + `workflow/plan_graph.md` — the data
   model the loop searches over.
5. `workflow/diagnose.md` — what counts as which bottleneck.
6. `workflow/replan.md` + `workflow/execution_strategy.md` — how
   candidates are generated and ranked.
7. `workflow/settle.md` — how the search converges.
8. `execution-model/SKILL.md` — formulas that drive prediction.
9. Other directories — read on demand from the stage that needs them.

## Reading order for an Agent (operational)

Always start at `SKILL.md`. The "How to use" section there enumerates
exactly which sub-skills to read at each stage. Do not pre-read
sub-skills unprompted.

## Conventions

- **Sub-files are plain Markdown.** Front-matter lives only in
  `SKILL.md` so Cursor doesn't register sub-files as separate skills.
- **One canonical definition per concept.** Cross-reference rather
  than copy. The state machine is the only place that defines
  transitions; bottleneck thresholds live in `diagnose.md`; etc.
- **YAML for schema and State; Python pseudocode for logic.** No
  real implementation code in skills.
- **Scale-aware, not scale-specific.** The workflow is uniform across
  node counts; sub-files use a `Scale-aware notes` section to call out
  where defaults or thresholds differ between single-node and
  multi-node.
- **Single source of truth for env flags.** Each flag is fully defined
  exactly once in `env/<area>.md`. `optimization/<bottleneck>/env.md`
  only lists "which flags to look at for this bottleneck" and links
  back to the catalog.

## Status

Initial scaffold (workflow + execution-model + optimization + env +
profiling + constraints + knowledge top-level entries). Sub-files
beyond top-level entries are filled out as the workflow demands them.
The design document `pilot/README.md` in the parent project drives all
content and is the reference whenever a skill is unclear.
