# diagnose — Bottleneck Classification

## Purpose

Read a `Snapshot` and produce a `DiagnosisReport` (mirrors
`pilot/README.md` §8.4) that:

- Names exactly one **primary bottleneck**.
- Lists evidence (which thresholds / observations triggered it).
- Lists `recommended_skills` (paths under `optimization/`).
- Lists `candidate_axes` (with axis taxonomy tags) for `replan.md`
  to consume.
- Optionally lists `env_suspect` (with `hint` paths) to trigger
  EnvSweep.
- Recommends an `execution_strategy` (Champion-Challenger / Per-Plan
  / Halving).

## Inputs

```yaml
snapshot:        # see workflow/observe.md for full schema
plan:            # the plan that produced this snapshot
execution_model: state/execution_model.yaml
plan_graph:      state/plan_graph.yaml
target_vector:
cluster_profile:
```

## Bottleneck taxonomy

```
MEMORY_BOUND      → optimization/memory/
COMPUTE_BOUND     → optimization/compute/
COMM_BOUND        → optimization/comm/
PIPELINE_BOUND    → optimization/pipeline/
MOE_IMBALANCED    → optimization/moe/      # MoE only
```

## Classification rules (in order)

Apply rules **top-down**, stop at first match. Default thresholds
mirror `pilot/README.md` §8.4 / §3 examples; see `Scale-aware notes`
for tightenings.

### 1. MEMORY_BOUND

```python
if snapshot.status == 'oom':
    bottleneck = MEMORY_BOUND
    evidence  = ["job OOMed at step {step}, mem_peak={actual_mem}GB"]
    confidence = 1.0

elif snapshot.mem_peak_gb / cluster.hbm_capacity_gb > 0.92:
    bottleneck = MEMORY_BOUND
    evidence  = ["mem_peak {gb}/{cap} = {pct}% > 92% safety margin"]
    confidence = 0.85

elif snapshot.mem_reserved_to_alloc_ratio > 1.4:
    bottleneck = MEMORY_BOUND
    evidence  = ["allocator fragmentation: reserved/alloc = {ratio}"]
    env_suspect = [{flag: PYTORCH_HIP_ALLOC_CONF,
                    hint: env/alloc.md#expandable-segments}]
    confidence = 0.7
```

### 2. MOE_IMBALANCED (MoE only)

```python
if model.arch == MoE and snapshot.expert_load_imbalance_pct > 25:
    bottleneck = MOE_IMBALANCED
    evidence = ["top expert receives {hi}× the bottom expert"]
    confidence = 0.9
```

### 3. PIPELINE_BOUND

```python
if plan.parallelism.pp > 1 and snapshot.bubble_ratio > 0.15:
    bottleneck = PIPELINE_BOUND
    evidence = ["bubble_ratio={br} > 0.15 with pp={pp}"]
    confidence = 0.85
```

(Skip this rule when `pp = 1`; it can never trigger.)

### 4. COMM_BOUND

```python
if snapshot.comm_ratio > comm_threshold:
    bottleneck = COMM_BOUND
    sub = "alltoall" if alltoall_share_pct > ar_share_pct else "allreduce"
    evidence = ["comm_ratio={comm_ratio} > {comm_threshold}",
                "{sub} dominates: {share}%"]
    confidence = 0.8

    # Likely env hint
    if snapshot.msg_size_p95_mb < 4 and (env.NCCL_BUFFSIZE_mb or 4) <= msg_p95:
        env_suspect.append({flag: NCCL_BUFFSIZE,
                            hint: env/rccl.md#buffsize})
```

`comm_threshold` defaults to `0.25` (multi-node), tightened to `0.20`
on single node — see `Scale-aware notes`.

### 5. COMPUTE_BOUND (default)

```python
if snapshot.gpu_util_avg < 0.65 or snapshot.tps < expected.tps * 0.8:
    bottleneck = COMPUTE_BOUND
    evidence = ["gpu_util={util} < 0.65",
                "tps={tps} vs expected {exp} (ratio={r})"]
    confidence = 0.7
else:
    bottleneck = COMPUTE_BOUND        # well-utilized, still room? marginal
    evidence = ["no other rule triggered, treating as COMPUTE_BOUND"]
    confidence = 0.5
```

## DiagnosisReport (output schema)

Mirrors `pilot/README.md` §8.4:

```yaml
snapshot_id: <ref>
plan_id:     <ref>
bottleneck:  COMPUTE_BOUND
confidence:  0.78
evidence:
  - "gpu_util_avg=0.62 < 0.65"
  - "tps=15400 vs predicted 18000 (ratio 0.86)"
  - "no MEM/COMM/PIPELINE rule triggered"

recommended_skills:
  - skills/pilot/optimization/compute/SKILL.md
  - skills/pilot/optimization/compute/mbs.md
  - skills/pilot/optimization/compute/parallel.md

candidate_axes:                    # Re-Plan consumes this
  - {axis: mbs,                  type: structural,     candidates: [2, 3, 4]}
  - {axis: tp,                   type: structural,     candidates: [2, 4, 8]}
  - {axis: recompute,            type: structural,     candidates: [selective, none]}
  - {axis: NCCL_BUFFSIZE,        type: strongly_local, candidates: [8M, 16M]}
  - {axis: PYTORCH_HIP_ALLOC_CONF, type: weakly_local, candidates: [seg, "seg,max_split_size_mb:512"]}

env_suspect:                       # if any → triggers EnvSweep
  - {flag: NCCL_BUFFSIZE, reason: "msg_size_p95=4MB but BUFFSIZE=4MB",
     hint: skills/pilot/env/rccl.md#buffsize}

suggested_strategy: Champion-Challenger
  rationale: |
    Mostly structural axes; weakly_local env axes can wait until Settle.
    Per-Plan would over-spend the budget for marginal NCCL_BUFFSIZE gains.
```

`type` here is the `axis_taxonomy.md` classification — `structural`
(parallelism / mbs / recompute), `cluster_shared`, `weakly_local`,
`strongly_local`.

## Confidence guidance

- `≥ 0.85` → take the bottleneck verdict at face value.
- `0.6-0.85` → proceed but mark `replan.policy = explore_exploit`
  (mix in 1 explore candidate from shelved).
- `< 0.6` → DO NOT trust the verdict alone. Either:
  - request EnvSweep (cheap, may resolve ambiguity), OR
  - request a profiling-trace stage (see `profiling/trace.md`).

## Soft signals (augment evidence, don't reclassify)

| Signal | Meaning |
|--------|---------|
| `intra_node_util_avg > 0.7` | Intra-node interconnect heavy → COMM hint even if comm_ratio low |
| `inter_node_util_avg > 0.7` (multi-node) | Cross-node IB heavy → COMM hint, biased to AR sub |
| `cpu_wait_time_pct > 5` | dataloader stall — out of scope, surface to user |
| `kernel_launch_overhead_pct > 8` | small kernels, recommend mbs↑ or fused ops |
| `nccl_init_time_s > 30` | env mismatch — almost always env_suspect |

## Reentry triggers (set in DiagnosisReport, consumed by state machine)

| Observation | Suggested transition |
|-------------|---------------------|
| `cluster_profile.age > 7d` | reentry to PREFLIGHT |
| `model_spec` differs from `execution_model.model_spec_hash` | reentry to PROJECTION |
| `nccl_init_time_s > 30` AND env_baseline unchanged | reentry to PREFLIGHT (env_probe subset) |

These are recorded in the DiagnosisReport but the actual transition
is handled by `state_machine.md`.

## Scale-aware notes

Same rule structure across scales; threshold defaults differ:

| Threshold | Single node | Multi node |
|-----------|-------------|-----------|
| `comm_threshold` (Rule 4) | 0.20 | 0.25 |
| `bubble_threshold` (Rule 3) | 0.15 (rare; pp=1 typical) | 0.15 |
| `mem_safety_margin` (Rule 1) | 0.92 | 0.92 |
| Bottleneck prior (informational) | MEM ~0.40, COMPUTE ~0.35, COMM ~0.15, MoE ~0.07, PIPELINE ~0.03 | COMM ~0.30, MEM ~0.25, COMPUTE ~0.20, PIPELINE ~0.15, MoE ~0.10 |

The prior is used only as a tie-breaker when two rules trigger with
similar evidence; it never overrides hard signals (OOM, NaN).
