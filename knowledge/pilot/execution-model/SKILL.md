# execution-model — Time / Memory / Comm Formulas

## Purpose

Provide closed-form (or near-closed-form) predictions of `T_step`,
`Mem_peak`, `T_comm`, and `T_bubble` for a candidate Plan, fitted
once at PROJECTION and consulted by `replan.md` Step 4 (priority) and
`constraints/oom.md` (pre-submit).

Mirrors `pilot/README.md` §6.

## Sub-files

| File | Owns |
|------|------|
| `compute.md` | `T_comp(layers, mbs, dtype)` and `eff_compute` calibration |
| `memory.md` | `Mem(plan)` decomposition: `M_param + M_grad + M_optim + M_act + M_buffer` |
| `communication.md` | `T_AR`, `T_A2A`, `T_AG` from `ClusterProfile.rccl_baseline` |
| `pipeline.md` | `T_bubble(pp, M)` and stage-balance rules |
| `partition.md` | layer partition strategy (mostly framework-driven; we predict the result) |
| `examples.md` | worked examples for Dense / MoE |

This entry file (`SKILL.md`) defines the **top-level decomposition**
and the **cache contract** with PROJECTION.

## Top-level decomposition

```
T_step = T_comp + T_comm_exposed + T_bubble

T_comm_exposed = T_comm - T_overlap
T_overlap      = min(T_comm_overlappable, T_comp_spare)

Mem_peak = M_param + M_grad + M_optim + M_act + M_buffer
```

Each sub-term is computed from per-layer formulas + parallelism
mapping. Sub-files own the details.

## Cache contract

PROJECTION fits coefficients once and writes:

```yaml
state/execution_model.yaml:
  fitted_at: <ts>
  cluster_profile_ref: <version>
  model_spec_hash: <sha>
  coefficients:
    T_comp:
      a: 4.7        # ms per layer per microbatch (dtype-specific)
      b: 0.8        # constant overhead per step
    M_act:
      full:      {c: 0.18, d: 0.05}   # GB per layer per microbatch + base
      selective: {c: 0.62, d: 0.05}
      none:      {c: 1.20, d: 0.05}
    eff_compute: 0.51                  # achieved / peak BF16 TFLOPS
    bw_eff_intra: 620                  # GB/s sustained
    bw_eff_inter: 320                  # multi-node only
  events:
    - {type: prediction_drift, plan: r0_p0, metric: tps, drift_pct: -10.6}
    - {type: recalibration_recommended, at: <ts>}
```

`replan.md` and `constraints/oom.md` read these coefficients via
`execution_model.predict_*(...)` helpers; sub-files document the exact
formulas.

## Confidence

Every prediction returns `(value, confidence)` where `confidence ∈ [0,
1]` reflects:

- `R²` of the fitted regression (low → low confidence).
- Distance from the profiled grid (extrapolating far → low).
- Whether `knowledge/cases.md` matches (`+0.1`).

Used in `replan.md` priority and as a `force_envsweep` heuristic in
`settle.md` (low confidence → favor cheap env exploration over
structural moves).

## When to recalibrate

PROJECTION re-fits when **any** of:

- `model_spec_hash` changes.
- `cluster_profile.version` changes.
- `state/execution_model.yaml.events` contains
  `recalibration_recommended` (set by LEARN at session end).
- Cache age > 7 days.

Mid-session recalibration is **forbidden** — it would invalidate all
in-flight predictions and PlanGraph priority math. Drift events are
just recorded for the next session.

## Sub-file status

Sub-files (`compute.md`, `memory.md`, etc.) are filled out as the
workflow demands them; this top-level entry alone is enough for the
Agent to call `execution_model.predict_*` via the tool layer. Each
sub-file documents:

- Its formula (with units).
- Which `coefficients[*]` it consumes.
- Which `cluster_profile.*` fields it consumes.
- Confidence rules specific to this term.
- Scale-aware notes (where intra- vs inter-node differs).

## Cross-links

- Calibration is invoked by → `workflow/projection.md`
- Predictions consumed by → `workflow/replan.md`,
  `workflow/constraints/oom.md`
- Drift events surfaced by → `workflow/baseline.md`,
  `workflow/report.md`
- Recalibration recommended by → `workflow/learn.md`

## Scale-aware notes

| Term | Single node | Multi node |
|------|-------------|-----------|
| `T_comp` | unchanged | unchanged |
| `T_comm` | dominated by xGMI / NVLink (intra-node only) | dominated by IB / RoCE between nodes; AR / A2A formulas use `bw_eff_inter` |
| `T_bubble` | typically 0 (`pp = 1`) | non-zero when `pp > 1`; PROJECTION must include `pipeline.md` |
| `M_act` | unchanged | unchanged per GPU; total = per-GPU × dp |
| Calibration cost | ~30 min single-node profiling | same single-node profiling; multi-node terms are extrapolated, not measured directly |
