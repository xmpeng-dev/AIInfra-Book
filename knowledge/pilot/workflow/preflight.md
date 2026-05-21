# preflight — Cluster Baseline + Env Probe

## Purpose

First stage of the workflow. Produces a `ClusterProfile` that captures
the **hardware baseline** and **cluster-level env baseline**, both
versioned and reusable across sessions.

PREFLIGHT is the only stage that may write `cluster.env_baseline`. All
later stages read it as a fixed substrate; per-plan deviations live in
`Plan.env.diff` (see `plan.md`).

## Inputs

```yaml
required:
  cluster_id:    <str>            # logical cluster name
optional:
  candidate_envs:                  # if explicit baseline candidates supplied;
    rccl: {...}                    # otherwise env_probe.run uses internal defaults
    hsa:  {...}
    alloc: {...}
    threading: {...}
  reuse_if_age_le_days: 7          # skip re-probe if a fresh profile exists
```

## Skills consulted

- `profiling/SKILL.md` (entry to data collection protocols)
- `profiling/preflight.md` (detailed measurement protocol)
- `profiling/network.md` (multi-node only — IB / RCCL micro-bench)
- `env/SKILL.md` (env catalog; informs `env_probe.run` defaults)

## Tools called

```python
preflight.run(cluster_id) -> ClusterProfile        # hardware baseline
env_probe.run(cluster_id, candidate_envs) -> EnvBaseline   # writes ClusterProfile.env_baseline
state.checkpoint(tuning_state) -> path
```

`preflight.run` and `env_probe.run` may share underlying probes; the
split is logical (compute / memory / interconnect peaks vs env
connectivity + RCCL micro-bench).

## Procedure

1. **Reuse check**. If `state/cluster_profile.yaml` exists for this
   `cluster_id` and `age ≤ reuse_if_age_le_days` and no
   `driver_changed` flag, **skip steps 2-4**.
2. **Hardware baseline** (`preflight.run`):
   - GEMM / MFMA peak (BF16, FP8 if supported)
   - HBM bandwidth + capacity per GPU
   - Intra-node interconnect (xGMI / NVLink) bandwidth
   - Inter-node interconnect (IB / RoCE) bandwidth (multi-node only)
3. **Comm baseline**:
   - AllReduce micro-bench across representative message sizes (1MB,
     16MB, 256MB)
   - AllToAll micro-bench (multi-node + MoE-likely scenarios)
4. **Env baseline** (`env_probe.run`):
   - Connectivity check on candidate flags (must boot a 1-step job
     successfully)
   - Micro-bench for performance-sensitive flags (e.g. `NCCL_NET_GDR_LEVEL`)
   - Reject candidates that fail fast (≤ 30s); promote survivors to
     baseline
5. Persist `ClusterProfile` with bumped `version`.

## State written

Mirrors `pilot/README.md` §8.1:

```yaml
state/cluster_profile.yaml:
  cluster_id:     <str>
  collected_at:   <ts>
  version:        <e.g. "mi300x-16node-v3">
  status:         validated | tentative
  age_days:       <int>           # auto-computed on read
  nodes:          <int>
  gpus_per_node:  <int>

  compute:
    peak_tflops_bf16:    <float>
    peak_tflops_fp8:     <float|null>
    hbm_bandwidth_gbs:   <float>
    hbm_capacity_gb:     <float>

  interconnect:
    intra_node:
      type:       xgmi | nvlink
      bandwidth_gbs: <float>
    inter_node:                    # null if single node
      type:       ib | roce | null
      bandwidth_gbs: <float|null>

  rccl_baseline:
    allreduce: [{size_mb, bw_gbs}, ...]
    alltoall:  [{size_mb, bw_gbs}, ...]

  env_baseline:
    version:    <str>
    status:     validated | tentative
    rccl:       {NCCL_*: ...}
    hsa:        {HSA_*: ..., GPU_MAX_HW_QUEUES: ...}
    alloc:      {PYTORCH_HIP_ALLOC_CONF: ...}
    threading:  {OMP_NUM_THREADS: ..., MKL_*: ...}
```

## Exit conditions

- **success**: `status: validated` (all probes passed) → PROJECTION.
- **soft_fail (`tentative`)**: hardware probes ok but one or more env
  candidates failed. Continue with the surviving baseline; mark
  `status: tentative`; record failed flags in
  `state/preflight_warnings.yaml`. Still → PROJECTION.
- **hard_fail**: hardware probe itself failed (e.g. driver error, node
  down). On_fail.

## On_fail

| Condition | FailureReport.suggested_transition |
|-----------|------------------------------------|
| Hardware probe failed (`preflight.run` raises) | `to: ABORT` `kind: CLUSTER`, escalate. Operator must fix hardware before tuning. |
| Env probe found NO viable baseline (all candidates failed connectivity) | `to: ABORT` `kind: CLUSTER`, escalate. The env catalog or driver state needs human attention. |
| `reentry_guard.PREFLIGHT.max_in_session` exceeded | `to: ABORT` `kind: UNKNOWN`, escalate. |

## Reuse semantics

`ClusterProfile` is reusable across sessions when:

- `cluster_id` matches.
- `version` of the existing profile is current (no driver bump).
- `age ≤ reuse_if_age_le_days` (default 7).

When reused, PREFLIGHT logs `reused: true` in `TuningState.stage_history`
and proceeds straight to PROJECTION.

When re-entered mid-session (via `HANG`, `CLUSTER`, or expiration),
`env_probe.run` may run **subset** mode (only the suspected flags) to
keep cost bounded. Subset mode bumps `version` only if values change.

## Cost budget

Charged to `setup_cost`, not tuning rounds:

- Hardware baseline: ~10-30 min.
- Env probe (full): ~10-20 min (each candidate flag + 1-step boot).
- Subset re-probe: ~5 min.

A reused profile is essentially free.

## Scale-aware notes

- **Single node**: skip inter-node interconnect probe and IB-related
  RCCL micro-bench. Env probe focuses on `HSA_*`, `PYTORCH_HIP_ALLOC_CONF`,
  `OMP_NUM_THREADS`. Total cost ~5-10 min.
- **Multi node**: full probe including IB topology, RoCE/IB GID, NCCL
  cross-node AR/A2A baselines. Env probe candidate set includes
  `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL`, `NCCL_SOCKET_IFNAME`. A bad
  IB env can take 30+ s per fail; the `≤ 30s fail-fast` rule (see
  `pilot/README.md` §12.1) limits damage.

## Notes

- `env_baseline` is the **single source of truth** for cluster-level
  env defaults. `optimization/<bottleneck>/env.md` and
  `Plan.env.diff` are read against this baseline; never re-define
  baseline values elsewhere.
- A version bump on `env_baseline` invalidates all open `Plan.env.diff`
  references; in practice this only happens at session boundaries
  because PREFLIGHT runs first.
- For audit, every PREFLIGHT writes a checkpoint to
  `state/checkpoints/preflight/<ts>/` containing the raw probe outputs.
