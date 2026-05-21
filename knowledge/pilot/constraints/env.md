# constraints/env — Env Compatibility Matrix

## Purpose

Reject environment-variable combinations that are known to be
incompatible (silently break, hang, or corrupt). Owns
`constraint.check_env(env_diff, env_baseline)`.

This file is a **catalog of forbidden / risky combinations**. Per-
flag definitions live in `env/<area>.md`.

## Tool contract

```python
constraint.check_env(env_diff, env_baseline) -> {
    valid: bool,
    violations: [
        {rule: <str>, conflict: [<flag1>, <flag2>], hint: <str>},
        ...
    ],
    warnings: [
        {rule: <str>, conflict: [...], hint: <str>},
        ...
    ],
}
```

`env_diff ⊕ env_baseline` is the resolved env to be applied; rules
operate on the **resolved set**, not the diff alone (because a baseline
value can conflict with a diff change).

## Hard incompatibility rules (reject)

```yaml
- rule: mutex_msccl_vs_certain_gdr
  conflict: [RCCL_MSCCL_ENABLE=1, NCCL_NET_GDR_LEVEL ∈ {0, 1}]
  hint: "MSCCL requires GDR; NET_GDR_LEVEL must be ≥ 2"

- rule: alloc_invalid_combo
  conflict: [PYTORCH_HIP_ALLOC_CONF.expandable_segments=True,
             PYTORCH_HIP_ALLOC_CONF.max_split_size_mb < 16]
  hint: "expandable_segments needs at least 16MB max_split"

- rule: numa_pin_conflict
  conflict: [numactl.cpunodebind set, OMP_PROC_BIND=spread]
  hint: "explicit NUMA pin conflicts with OpenMP spread; pick one"

- rule: ib_hca_set_without_socket_ifname
  conflict: [NCCL_IB_HCA set, NCCL_SOCKET_IFNAME unset]
  hint: "explicit IB HCA needs explicit SOCKET_IFNAME for control plane"

- rule: gpu_max_hw_queues_too_low
  conflict: [GPU_MAX_HW_QUEUES < 2]
  hint: "value below 2 starves stream concurrency on most workloads"
```

## Soft warnings (accept but flag)

```yaml
- warning: nccl_buffsize_far_above_msg_p95
  conflict: [NCCL_BUFFSIZE > 4 × snapshot_msg_size_p95_mb]
  hint: "buffsize wastes mem; tune to ~msg_size_p95"

- warning: msccl_unsupported_topology
  conflict: [RCCL_MSCCL_ENABLE=1, cluster_profile.topology not in MSCCL_supported]
  hint: "MSCCL falls back silently; expect no gain"

- warning: omp_threads_high_with_dataloader_workers_high
  conflict: [OMP_NUM_THREADS × dataloader_workers > cpus_per_task]
  hint: "oversubscription likely; reduce one"
```

Warnings appear in `EnvSweepResult.warnings` and DiagnosisReport
context, not as rejections.

## Conditional rules (depend on cluster_profile)

```yaml
- rule: gdr_required_for_ib_perf
  expr: cluster_profile.interconnect.inter_node.type == ib AND NCCL_NET_GDR_LEVEL < 2
  hint: "IB cluster but GDR disabled; expect 30-50% comm regression"

- rule: ib_hca_must_match_topology
  expr: NCCL_IB_HCA set AND specified HCAs not in cluster_profile.interconnect.inter_node.hcas
  hint: "specified HCA not present on this cluster"
```

## Catalog mutability

Adding a new conflict requires:

1. Append a rule here.
2. Reference both flags in their respective `env/<area>.md` entries
   under "Co-tune with".
3. If the conflict was discovered mid-session via failure, also append
   to `knowledge/anti-patterns.md` with the failure_id for traceability.

## Output ordering

Order rules by:

1. Hard mutex (the most likely cause of hang/break).
2. Per-flag config inconsistency.
3. Cluster-profile-conditional rules.
4. Warnings last.

Re-Plan / EnvSweep relax the first violation when retrying.

## Cross-links

- Per-flag definitions → `env/<area>.md`
- Axis types → `axis_taxonomy.md`
- EnvSweep mechanics → `workflow/envsweep.md`
- Validation pipeline → `constraints/SKILL.md`

## Scale-aware notes

| Family | Single node | Multi node |
|--------|-------------|-----------|
| Hard rules active | small subset (no IB / SOCKET_IFNAME / MSCCL-on-fabric) | full set |
| Warning rules active | small subset | full set |
| Most common rejection | `expandable_segments + max_split_size_mb` mismatch | `MSCCL × GDR_LEVEL` mismatch, `IB_HCA × SOCKET_IFNAME` |
| Catalog growth rate | low | moderate (new RCCL versions add new flags / quirks) |
