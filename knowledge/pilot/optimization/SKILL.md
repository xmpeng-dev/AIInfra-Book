# optimization — Bottleneck-Driven Recipes (Top Entry)

## Purpose

Top-level index for optimization recipes. `workflow/diagnose.md`
classifies the run into one of five **bottlenecks**; this directory
holds one sub-folder per bottleneck with the actual levers.

This entry file:

1. Lists the bottleneck → folder mapping.
2. Defines the **recipe template** every leaf file follows.
3. Documents the cross-cutting rule: every recipe lists axes
   tagged with `axis_taxonomy.md` types so `replan.md` can pick the
   right execution strategy.

## Bottleneck → folder

```
COMPUTE_BOUND     → optimization/compute/
MEMORY_BOUND      → optimization/memory/
COMM_BOUND        → optimization/comm/
PIPELINE_BOUND    → optimization/pipeline/
MOE_IMBALANCED    → optimization/moe/
```

Each folder has:

```
<bottleneck>/
├── SKILL.md       # entry: lever priority table + cross-links
├── <axis>.md      # one per major lever (e.g. mbs.md, recompute.md)
└── env.md         # which env flags help this bottleneck (links env/<area>.md)
```

`env.md` files are **lookup-only**: they list flags relevant to the
bottleneck and link the catalog (`env/<area>.md`). They never duplicate
the catalog definitions.

## Recipe template

Every leaf file (e.g. `mbs.md`, `recompute.md`) follows:

```markdown
# <axis>

## Precondition
   What must be true (memory headroom, parallelism shape, etc.) before
   trying this change.

## Action
   Concrete values to try. Prefer small steps (e.g. mbs +1, not ×2).

## Expected effect
   Quantitative: tps gain range, mem delta, comm delta — with the
   confidence band from execution-model/.

## Confidence signals
   What in the Snapshot predicts success / failure for this axis.

## Risk
   What can go wrong; which `kind` (OOM / NUMERICAL / etc.) is most
   likely; the suggested on_fail transition.

## Anti-patterns
   Cross-link to knowledge/anti-patterns.md entries that should
   override this recipe.
```

## Lever priority convention

Each `<bottleneck>/SKILL.md` lists a **priority table** with columns:

```
| Priority | Axis | Type (axis_taxonomy) | Sub-file | Typical gain | Risk |
```

Re-Plan walks this table top-down, generating one candidate per axis,
filtering by `exhausted_neighborhoods` and `constraints/*`.

Priorities are heuristics, not hard rules. If a higher-priority axis
hits an `exhausted_neighborhood` for the current parent, Re-Plan
naturally falls through to the next axis.

## Cross-cutting rules

These apply across all bottlenecks:

1. **Always include the axis type** from `axis_taxonomy.md` in the
   priority table. Re-Plan uses it for `execution_strategy.md` choice.
2. **Always cite `execution-model/<term>.md`** in `Expected effect`.
   Predictions must come from the model, not the recipe author's
   intuition.
3. **Anti-patterns override recipes.** Before Re-Plan emits a
   candidate from a recipe, it checks
   `knowledge/anti-patterns.md` for a hit on
   `(model_family, scale, axis_change)`.
4. **Env flags belong in `env/<area>.md`.** A recipe may reference
   `optimization/<bottleneck>/env.md` which in turn links the
   catalog; never define a flag inline.
5. **No transition logic in recipes.** All "if X then go to stage Y"
   rules live in `state_machine.md`; recipes only return candidate
   axes + values.

## Multi-bottleneck rounds

If `Diagnose` returns two bottlenecks with similar confidence (rare,
typically `MEMORY` + `COMM`), Re-Plan reads BOTH `<bottleneck>/SKILL.md`
files and merges their priority tables; ties broken by:

1. Higher predicted gain (from execution-model).
2. Lower est_cost.
3. `cluster_shared` > `weakly_local` > `strongly_local` > `structural`
   (cheaper axes first).

## Sub-folder status

| Folder | Status | Notes |
|--------|--------|-------|
| `compute/` | entry done; sub-files stubbed | mbs / parallel / kernel |
| `memory/`  | entry done; sub-files stubbed | recompute / offload / fragmentation |
| `comm/`    | entry done; sub-files stubbed | bucket / overlap / topology |
| `pipeline/`| entry done; sub-files stubbed | vpp / microbatch / balance |
| `moe/`     | entry done; sub-files stubbed | routing / dispatch / load_balance |

Sub-files are filled out as `replan.md` first emits a candidate on
that axis. Until then, `axis_taxonomy.md` provides defaults sufficient
for Re-Plan to function.

## Cross-links

- Bottleneck classification → `workflow/diagnose.md`
- Axis types → `workflow/axis_taxonomy.md`
- Predictions → `execution-model/SKILL.md`
- Pre-submit validation → `constraints/SKILL.md`
- Anti-pattern lookup → `knowledge/anti-patterns.md`

## Scale-aware notes

Bottleneck **distribution** changes with scale (see
`diagnose.md` § Scale-aware notes priors). The recipes themselves
are scale-agnostic — they describe the lever, not the cluster shape.
Each leaf file may include a `Scale-aware notes` section to call out
where defaults differ.
