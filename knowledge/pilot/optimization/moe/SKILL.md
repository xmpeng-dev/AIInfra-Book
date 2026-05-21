# optimization/moe — MoE-Specific Recipes

## Purpose

Two roles:

1. **Diagnose-recommended entry** when `bottleneck = MOE_IMBALANCED`.
2. **Cross-cut entry** consulted by `compute / memory / comm` SKILLs
   whenever `model.arch == MoE` (because MoE changes the typical
   levers and their costs).

## When invoked

```yaml
# Direct invocation:
bottleneck: MOE_IMBALANCED
recommended_skills:
  - skills/pilot/optimization/moe/SKILL.md
  - skills/pilot/optimization/moe/load_balance.md

# Cross-cut from other bottlenecks (always read if MoE):
recommended_skills:
  - skills/pilot/optimization/<bottleneck>/SKILL.md
  - skills/pilot/optimization/moe/SKILL.md           # consult § MoE adjustments
```

## Lever priority

| # | Axis | Type | Sub-file | Effect | Risk |
|---|------|------|----------|--------|------|
| 1 | `moe.capacity_factor ↑` (1.0→1.25) | structural | `load_balance.md` | ↓ drop rate, smoother | mem ↑ |
| 2 | router warmup / aux-loss tweak | structural | `load_balance.md` | balance ↑ | numerical (loss curve) |
| 3 | `parallelism.ep` retune | structural | (this file's "EP retune") | varies by shape | comm ↑ at small ep |
| 4 | `moe.expert_parallelism_mode` (inflight ↔ sequential) | structural | `dispatch.md` | overlap / mem trade | numerical |
| 5 | `moe.top_k ↓` (rarely 2→1) | structural | (this file) | ↓ alltoall vol | accuracy concern (escalate) |
| 6 | `env.RCCL_MSCCL_ENABLE` (alltoall path) | weakly_local | `env.md` → `../../env/rccl.md` | -5..20% A2A on supported topo | compatibility |

## EP retune (recurring theme)

Within `ep ∈ {1, 2, 4, 8, 16, ...}` (capped by `gpus_per_node` for
intra-node A2A; can extend cross-node with placement care):

```
ep=high
  + max alltoall volume
  + min expert-weight memory per GPU
  + best load distribution potential
  - sensitive to imbalance; one slow GPU stalls all
  - cross-node A2A becomes dominant when ep > gpus_per_node

ep=mid
  + halved alltoall message count (vs high)
  + 2 expert groups → some imbalance averaged
  - 2× expert weight per GPU (mem ↑)

ep=low
  + minimal alltoall (or none)
  + maximum mem per GPU (often won't fit on large MoE)
```

Decision rule (parametric in current scale):

```python
if expert_load_imbalance_pct > 25:                    # MOE_IMBALANCED triggered
    if mem_headroom_at_ep_minus_one_step > 10%:
        candidate += {parallelism.ep: ep / 2}         # reduce, average imbalance
    else:
        candidate += {moe.capacity_factor: 1.25}      # stay, soften via cap

elif comm_ratio > comm_threshold AND alltoall_share_pct > 60:
    candidate += {parallelism.ep: ep / 2}             # cut alltoall message count
```

See `optimization/SKILL.md` for cross-bottleneck conflict resolution
when MoE-imbalance + memory both trigger.

## MoE adjustments to other bottlenecks (cross-cut section)

When invoked as cross-cut from another bottleneck SKILL, modify those
levers as follows:

### Memory-bound on MoE

- `recompute=full` is more expensive per step on MoE (recomputes
  expert forward + dispatch). Bias toward
  `recompute_pattern = expert_only`.
- `ep ↑` is often a better memory lever than `tp ↑` (ep cuts expert
  weights; tp doesn't reduce them).

### Compute-bound on MoE

- `mbs ↑` interacts with `capacity_factor`: large mbs + cap=1.0 →
  drop spikes. Co-tune.
- `top_k = 2 → 1` halves expert FLOPs and alltoall volume but is a
  model-quality decision; **flag for user, do NOT auto-pick**.

### Comm-bound on MoE

- Already covered (lever 3 / 4 / 6).
- Avoid `NCCL_BUFFSIZE` increases here — MoE alltoall messages are
  highly variable in size; a too-large buffer wastes mem.
- For multi-node MoE, `ep` placement (intra- vs inter-node) is
  a major lever; see `comm/topology.md`.

## Cross-links

- Imbalance diagnosis → `workflow/diagnose.md` § MOE_IMBALANCED rule
- MoE alltoall modeling → `execution-model/communication.md` § EP A2A
- Initial MoE plan generation → `workflow/projection.md` § MoE
- MoE env interactions → `env/SKILL.md`, `env/rccl.md`

## Stubbed sub-files

| File | Status |
|------|--------|
| `routing.md` | stub |
| `dispatch.md` | stub |
| `load_balance.md` | stub |
| `env.md` | stub |

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Effective `ep` range | `1..gpus_per_node` (intra-node A2A) | `1..world_size` (cross-node A2A possible; placement matters) |
| A2A path | xGMI / NVLink only | xGMI within node + IB / RoCE between nodes |
| Capacity-factor sensitivity | moderate (low drop tail) | higher (cross-node dispatch latency makes drops costlier) |
| `expert_parallelism_mode = inflight` impact | overlap-positive on intra-node | even more positive cross-node (hides IB latency) |
| Common early-round bottleneck | imbalance for fresh runs; resolves with `capacity_factor` by round 2 | imbalance + cross-node A2A both contend; usually 2-3 rounds to settle |
| MSCCL / SHARP-style optimizations | rarely needed | high-value when topology supports |
