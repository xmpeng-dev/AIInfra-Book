# projection — Build Execution Model + Initial Plans

## Purpose

Take `ClusterProfile` + `model_spec` and produce:

1. An **ExecutionModel** cache (per-layer compute / memory / comm
   formulas calibrated to this cluster).
2. A small set of **initial Plans** (typically 1-3) ranked by predicted
   `tps`, with a `confidence` field per plan.

PROJECTION runs once per session (or after `STRUCTURAL_INVALIDATION`).
Cost is dominated by single-node profiling and is largely
cluster-shape-independent.

## Inputs

```yaml
required:
  model_spec:        <Model spec from user>
  cluster_profile:   state/cluster_profile.yaml
optional:
  knowledge_hint:                       # if knowledge.cases.md has a match
    matched_case_id: <id>
    matched_axis_priors: {...}
  reuse_if_fresh: true                  # use cached ExecutionModel if recent
```

## Skills consulted

- `execution-model/SKILL.md` (always; entry point)
- `execution-model/compute.md` (for `T_comp(layer, mbs)` calibration)
- `execution-model/memory.md` (for `M_act(layer, mbs, recompute)`)
- `execution-model/communication.md` (for AR / A2A formulas)
- `execution-model/pipeline.md` (for `T_bubble`)
- `optimization/moe/SKILL.md` if `model_spec.arch == MoE`
- `constraints/oom.md` (filter infeasible initial plans)
- `knowledge/cases.md` (prefer historical winners)

## Tools called

```python
profiler.run(
  model_spec=...,
  configs=[
    # cheap micro-runs to fit ExecutionModel coefficients
    {layers: 4, mbs: 1, recompute: full},
    {layers: 4, mbs: 2, recompute: selective},
    {layers: 4, mbs: 4, recompute: none},
  ],
  scale={nodes: 1, gpus: <gpus_per_node>, steps: 50},
) -> ProfilingResult

constraint.check(plan, cluster_profile) -> {valid, violations}
constraint.estimate_mem(plan) -> {mem_gb, breakdown, confidence}
```

Profiling is always single-node (cheap); larger-scale predictions are
extrapolated via the formulas in `execution-model/`.

## Procedure

1. **Cache lookup**. If `state/execution_model.yaml` exists for this
   `(model_spec, cluster_profile.version)` and is < 7d old, load and
   skip step 2.
2. **Profile**. Call `profiler.run(...)` with the configs above. Record
   wall time per step, GPU memory peak, sustained interconnect BW.
3. **Fit coefficients**. Solve linear regression for:
   - `T_comp(layer, mbs) = a · mbs + b`
   - `M_act(layer, mbs, recompute) = c(recompute) · mbs + d(recompute)`
   - Use `ClusterProfile.peak_tflops_*` as anchor for `efficiency`.
   See `execution-model/compute.md` for the equations.
4. **Predict bubble & comm** from formulas (no profiling needed):
   - `T_bubble = (pp-1)/(pp-1+M) · T_comp` (M = num microbatch)
   - `T_AR(grad_size, dp_or_tp) = grad_size / bw_eff(size)`
   - `T_A2A(msg_size, ep, top_k) = msg_size / bw_a2a_eff(size)`
   - Use `ClusterProfile.rccl_baseline` as bandwidth anchors.
5. **Generate initial Plans**. Enumerate a tiny grid of `{tp, pp, dp,
   ep, mbs, recompute}` (typically ≤ 8 points after cluster-shape
   pruning) and score with `predicted.tps - λ · risk(mem, oom_margin)`.
6. **Filter**. Drop plans failing `constraint.check` or
   `constraint.estimate_mem > 0.92 × hbm_capacity`.
7. **Knowledge prior** (if hint present): boost the plan that's
   closest in axes to `matched_axis_priors`.
8. Output the **top 1-3 plans** with predicted metrics and a note on
   which are **exploit** vs **explore** (if more than 1).

## State written

```yaml
state/execution_model.yaml:
  fitted_at: <timestamp>
  cluster_profile_ref: <version>
  model_spec_hash: <sha>
  coefficients:
    T_comp: {a: 4.7, b: 0.8}                # ms per layer per microbatch
    M_act:
      full:      {c: 0.18, d: 0.05}         # GB per layer per microbatch
      selective: {c: 0.62, d: 0.05}
      none:      {c: 1.20, d: 0.05}
    eff_compute: 0.51                        # achieved / peak BF16 TFLOPS
    bw_eff_intra: 620                        # GB/s sustained, vs peak
    bw_eff_inter: 320                        # multi-node only
  notes: ["recompute=full clamps act mem at expected level",
          "MoE alltoall scales linearly within ep ≤ 8 intra-node"]

state/initial_plans.yaml:
  generated_at: <timestamp>
  plans:
    - id: r0_p_init_1
      parallelism: {tp: 4, pp: 1, dp: 4, ep: 8, vpp: 1}
      runtime: {mbs: 2, recompute: selective}
      env.diff: {NCCL_MIN_NCHANNELS: 8}
      predicted: {tps: 18000, mem_gb: 168, comm_ratio: 0.12,
                  bubble: 0.0, confidence: 0.78}
      tag: exploit
    - id: r0_p_init_2
      parallelism: {tp: 8, pp: 1, dp: 2, ep: 8, vpp: 1}
      runtime: {mbs: 1, recompute: full}
      predicted: {tps: 14500, mem_gb: 112, ..., confidence: 0.85}
      tag: explore         # safer mem, lower tps; useful if r0_p_init_1 OOMs
```

## Exit conditions

- **success**: at least one plan with `predicted.tps > 0` and
  `confidence ≥ 0.6`. Pick the highest-`tps` plan as **the SMOKE /
  BASELINE candidate**; keep the rest for Re-Plan to draw from later.
- **soft_fail**: best `confidence < 0.6` → log a warning, still
  proceed.
- **hard_fail**: no plan satisfies `constraint.check` →
  `FailureReport {kind: STRUCTURAL_INVALIDATION,
   hint: "model too large for this cluster shape"}`.

## On_fail

| Condition | FailureReport.suggested_transition |
|-----------|------------------------------------|
| `profiler.run` itself fails | `to: PREFLIGHT` (suspect environment) |
| All plans OOM-by-prediction even at recompute=full | `to: ABORT` with `kind: STRUCTURAL_INVALIDATION` |

## Cost budget

- Profiling: ~30-60 min single-node, ~1 GPU·h. One-time per
  `(model_spec, cluster_profile.version)`.
- Fitting + plan generation: pure CPU, < 1 min.
- Cached run: free.

## Notes on confidence

The `confidence` field combines:

- `R²` of fitted regression (low → low confidence in compute / mem
  prediction)
- Distance of the proposed `(mbs, recompute, tp, pp, ep)` from the
  profiled grid (extrapolating far → low confidence)
- Whether `knowledge_hint` matched (boosts confidence by +0.1 if so)

Used in `replan.md` priority and by Re-Plan strategy choice (low
conf → prefer exploit).

## Scale-aware notes

The procedure is identical regardless of cluster shape; defaults
differ:

- **Single node**: `pp ≤ 1` typical; `dp = 1`; bubble term is zero.
  Initial plan grid stays small (≤ 4 points). Profiling is on the
  same node it'll train on.
- **Multi node**: enumerate `pp ∈ {1, 2, 4}`, `dp` derived from
  `nodes × gpus_per_node / (tp × pp)`. Bubble term matters when
  `pp > 1`. Profiling is single-node but predictions account for
  inter-node bandwidth from `ClusterProfile.rccl_baseline`.
- **Cluster-shape-aware env diff**: initial plans may carry a
  `env.diff` tuned to scale (e.g. `NCCL_BUFFSIZE` larger at higher
  message sizes implied by larger DP). These are seeds; `EnvSweep`
  refines them.
