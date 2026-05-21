# optimization/comm — Recipes for COMM_BOUND

## Purpose

Diagnose-recommended entry when `bottleneck = COMM_BOUND`. Sub-rule
in `diagnose.md` already split the dominant comm into `alltoall` vs
`allreduce` (via `alltoall_share_pct` vs `ar_share_pct`); sub-files
diverge accordingly.

## When invoked

`workflow/diagnose.md` sets:

```yaml
bottleneck: COMM_BOUND
recommended_skills:
  - skills/pilot/optimization/comm/SKILL.md
  - skills/pilot/optimization/comm/<sub>.md
env_suspect:                   # often co-emitted
  - {flag: NCCL_BUFFSIZE,        hint: env/rccl.md#buffsize}
  - {flag: NCCL_MIN_NCHANNELS,   hint: env/rccl.md#min-nchannels}
```

## Lever priority

| # | Axis | Type | Sub-file | Effect | Risk |
|---|------|------|----------|--------|------|
| 1 | `comm.overlap` (compute-comm overlap) | structural | `overlap.md` | -10..30% exposed comm | numerical (rare) |
| 2 | `parallelism.tp ↓` (less AR traffic) | structural | `../compute/parallel.md` | -linear AR vol | mem ↑ |
| 3 | `parallelism.ep` retune (MoE alltoall) | structural | `../moe/SKILL.md` | varies | imbalance shift |
| 4 | `env.NCCL_BUFFSIZE ↑` (matches `msg_size_p95`) | weakly_local | `env.md` → `../../env/rccl.md` | -5..15% comm | mem ↑ |
| 5 | `comm.bucket_size_mb` retune | strongly_local | `bucket.md` | -3..10% comm | numerical |
| 6 | `env.NCCL_MIN_NCHANNELS ↑` | weakly_local | `env.md` → `../../env/rccl.md` | -2..8% comm | scheduling |
| 7 | `env.NCCL_ALGO` / `env.NCCL_PROTO` | weakly_local | `env.md` → `../../env/rccl.md` | -3..10% comm | varies by msg size |
| 8 | `env.RCCL_MSCCL_ENABLE` (AMD path) | weakly_local | `env.md` → `../../env/rccl.md` | -5..20% A2A on supported topo | compatibility |
| 9 | Topology placement (intra vs inter-node ranks) | structural | `topology.md` | varies | framework-dependent |

## Apply order rationale

- **Lever 1 (overlap) is always tried first**: overlap is a "free"
  win if the framework supports it; cheap to test via flag.
- For TP-AR-dominated cases: try lever 2 (`tp ↓`) but check OOM
  headroom — frees comm at the cost of more memory per GPU.
- For EP-A2A-dominated cases (MoE only): lever 3 (re-tune `ep`) —
  often `ep ↓` reduces A2A volume per step at the cost of more
  expert weights per GPU.
- Levers 4-8 are env / framework knobs; mostly handled via
  `EnvSweep` when they show up as `env_suspect`.
- Lever 9 (topology) is structural and framework-dependent;
  use only when the simpler levers are exhausted.

## TP-AR vs EP-A2A routing

```
if alltoall_share_pct > ar_share_pct:        # MoE alltoall heavy
    recommended_subs += ['alltoall.md', '../moe/dispatch.md']
else:                                        # AR (TP / DP) heavy
    recommended_subs += ['allreduce.md']
```

## Cross-links

- "What's the modeled comm cost?" → `execution-model/communication.md`
- Env-side comm flags → `env/rccl.md`
- MoE-specific comm levers → `optimization/moe/dispatch.md`
- Bottleneck classification → `workflow/diagnose.md`

## Stubbed sub-files

| File | Status |
|------|--------|
| `overlap.md` | stub |
| `bucket.md` | stub |
| `alltoall.md` | stub |
| `allreduce.md` | stub |
| `topology.md` | stub |
| `env.md` | stub (one-liner per env axis with link) |

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| Dominant comm path | xGMI / NVLink (intra-node only) | IB / RoCE between nodes; xGMI within |
| Lever 1 (overlap) impact | moderate (intra-node already fast) | high (hides cross-node latency) |
| Lever 2 (`tp ↓`) cost | low (mem ↑ usually OK) | low if `dp / pp` absorbs the increase |
| Lever 4 (`NCCL_BUFFSIZE`) impact | low (xGMI-bound) | high (matches IB MTU / msg_size) |
| Lever 8 (MSCCL) | low (intra-node A2A is cheap) | high on MoE / AMD topologies |
| Common bottleneck mode | rare; usually MoE A2A or wide-TP AR | very common; AR + A2A both contend |
