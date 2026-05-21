# observe — Snapshot Schema + Collection Protocol

## Purpose

Defines the canonical `Snapshot` schema that every other stage produces
or consumes. Owns the contract between `submit.run` (job execution) and
the rest of the workflow.

This file is **schema + collection rules only**. Bottleneck logic
lives in `diagnose.md`; promotion logic in `settle.md`.

Schema mirrors `pilot/README.md` §8.3.

## When invoked

After every `submit.run` finishes (or is cancelled by early-stop):

```python
snap = observe.snapshot(job_id)
# state machine then routes per stage (Smoke / Baseline / Execute / Reference)
```

## Tool contract

```python
observe.snapshot(job_id) -> Snapshot
observe.compare_loss(job_id, reference) -> {pass, drift, delta_pct, where_diverged}
```

`observe.snapshot` MUST return all required fields below (use `null`
for unobtainable). Missing required fields cause Diagnose to abort;
`null` is fine.

## Snapshot schema (canonical)

```yaml
# Identity
job_id:            <str>
plan_id:           <str>           # set by submit.run from plan
session_id:        <str>
round:             <int>           # 0 for smoke/baseline, ≥1 for loop
mode:              smoke | baseline | challenger | reference | envsweep
submitted_at:      <ts>
finished_at:       <ts>

# Outcome
status:            ok | oom | failed | early_stopped | hung
exit_code:         <int>
early_stop_reason: null | tps_below_champion_85pct | tps_below_prediction_70pct
                   | oom_imminent | loss_drift_early | hang_5min
completed_steps:   <int>
wall_time_s:       <float>
confidence_in_metrics: high | medium | low      # low if completed_steps < 50

# Throughput / utilization
metrics:
  tps:                          <float>     # tokens/sec aggregated across DP
  step_time_ms:                 <float>
  tps_ema_50:                   <float>
  tps_ema_full:                 <float>
  gpu_util_avg:                 <float>     # 0..1
  gpu_util_p50:                 <float>
  gpu_util_p95:                 <float>
  intra_node_util_avg:          <float>     # xGMI / NVLink utilization 0..1
  inter_node_util_avg:          <float|null> # IB utilization; null if single node
  cpu_wait_time_pct:            <float>     # %
  kernel_launch_overhead_pct:   <float>

# Memory (per-GPU peak unless noted)
  mem_peak_gb:                  <float>
  mem_alloc_avg_gb:             <float>
  mem_reserved_avg_gb:          <float>
  mem_reserved_to_alloc_ratio:  <float>     # frag indicator
  has_oom_event:                <bool>

# Communication
  comm_ratio:                   <float>     # T_comm_exposed / T_step
  bubble_ratio:                 <float>     # 0 if pp=1
  overlap_ratio:                <float>     # 0..1, fraction of comm hidden in compute
  ar_share_pct:                 <float>     # of total comm time
  alltoall_share_pct:           <float>     # of total comm time
  msg_size_p50_mb:              <float>
  msg_size_p95_mb:              <float>
  nccl_init_time_s:             <float>

# Numerical
  has_nan:                      <bool>
  has_inf:                      <bool>

# MoE (null for Dense)
  expert_load_imbalance_pct:    <float|null>   # max/min - 1
  expert_drop_rate_pct:         <float|null>
  alltoall_skew_pct:            <float|null>

# Loss (full curve, downsampled to one point per N steps)
loss:
  curve:           [[step, loss], ...]    # N=10 default; never null on real run
  ema_50:          <float>
  ema_full:        <float>
  drift_vs_reference: null | <delta_pct>  # filled by compare_loss

# Resolved env (cluster_profile.env_baseline + plan.env.diff)
resolved_env:
  PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
  ...

# Pointers
artifacts:
  stdout: state/checkpoints/r{N}/<plan>_stdout.log
  stderr: state/checkpoints/r{N}/<plan>_stderr.log
  trace:  null | <path>                   # if profiling/trace.md was applied
```

## Collection rules

### Required, always present

`status`, `exit_code`, `completed_steps`, `wall_time_s`, `metrics.tps`,
`metrics.mem_peak_gb`, `metrics.has_nan`, `loss.curve` (≥ 1 point),
`resolved_env`.

If any required field can't be collected, `observe.snapshot` raises;
the caller treats it as `kind: UNKNOWN` and escalates.

### Conditionally required

| Field | Required when |
|-------|---------------|
| `metrics.bubble_ratio` | `plan.parallelism.pp > 1` |
| `metrics.alltoall_share_pct`, `expert_*` | `model.arch == MoE` |
| `metrics.inter_node_util_avg` | `cluster.nodes > 1` |
| `loss.drift_vs_reference` | reference exists; CORRECTNESS sets it |
| `early_stop_reason` | non-null only when `status=early_stopped` |

### Optional (nice-to-have)

`gpu_util_p50/p95`, `kernel_launch_overhead_pct`, `nccl_init_time_s`,
`msg_size_p50_mb`. Used as soft signals in `diagnose.md`.

## Sampling cadence

| Source | Sampled at |
|--------|------------|
| `tps`, `gpu_util`, `mem_alloc/reserved` | every step (after warmup), aggregated to ema_50 / ema_full |
| `intra_node_util`, `inter_node_util`, `comm_ratio` | every 10 steps (cheap to derive from RCCL stats) |
| `loss.curve` | every 10 steps (dense enough for compare_loss) |
| `expert_load_imbalance_pct` | every 50 steps (MoE only; expensive) |
| `nccl_init_time_s` | once at boot |

Warmup window = first 20 steps; excluded from EMA / averages.

## Snapshot for early-stopped jobs

If `status=early_stopped` AND `completed_steps < 50`:

- Set `confidence_in_metrics: low`.
- Still populate all required fields with what was observed.
- `tps_ema_full` may equal `tps_ema_50` (or be `null` if < 50 steps).
- `loss.curve` may be very short — Settle will skip lite-correctness
  for it.

Do NOT discard early-stopped Snapshots; they feed
`exhausted_neighborhoods` in PlanGraph and inform future predictions.

## State written

```yaml
state/snapshots/<plan_id>.yaml: <Snapshot above>
```

One file per `plan_id`. Smoke / baseline / reference get `r0_smoke`,
`r0_p0`, `r0_ref` respectively; challenger snapshots get `r{N}_p{i}`.

## Schema versioning

```yaml
schema_version: 1.0
```

Bump on backward-incompatible change. LEARN reads this to migrate old
case files.

## Scale-aware notes

| Field / behavior | Single node | Multi node |
|------------------|-------------|-----------|
| `inter_node_util_avg` | always `null` | always required |
| `comm_ratio` typical band | < 15% | 15-30% |
| `bubble_ratio` typical | 0 (pp=1) | 5-15% if pp > 1 |
| `nccl_init_time_s` | small (intra-node only) | larger; > 30s suggests env baseline drift |
| For MoE: `alltoall_share_pct` of `comm_ratio` | high (intra-node A2A is the dominant comm) | depends on `ep` placement (intra vs cross-node) |
