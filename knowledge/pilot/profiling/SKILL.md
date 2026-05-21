# profiling — Data Collection Protocols (Top Entry)

## Purpose

Defines **how** Pilot collects data at every stage that needs it.
This skill does not decide *what* to do with the data (that's
`diagnose.md` / `execution-model/`); it pins down measurement
protocols so results are reproducible.

## Sub-files

| File | Owns |
|------|------|
| `preflight.md` | Cluster baseline measurement (compute / mem / interconnect / RCCL micro-bench) |
| `gpu.md` | Per-GPU runtime metrics collection (utilization, mem, kernel timeline) |
| `network.md` | IB / RCCL / inter-node telemetry (multi-node only) |
| `trace.md` | Timeline / trace capture (rocprof / RCCL profiler), expensive — opt-in |
| `env_probe.md` | Safe env-probing protocol (connectivity → micro-bench → multi-node) |

## When invoked

```
PREFLIGHT
  ├─ profiling/preflight.md          (compute / mem / interconnect baselines)
  ├─ profiling/network.md            (multi-node only; IB / RoCE topology)
  └─ profiling/env_probe.md          (cluster-level env validation)

PROJECTION
  └─ profiling/preflight.md           (single-node profile to fit ExecutionModel)

OBSERVE
  └─ profiling/gpu.md                  (Snapshot collection)

DIAGNOSE / RE-PLAN (on demand)
  └─ profiling/trace.md                (when confidence < 0.6 and structural axes
                                        exhausted; opt-in due to cost)

ENVSWEEP
  └─ profiling/env_probe.md            (per-combo connectivity + tps measurement)
```

## Cost model

| Protocol | Per-call cost | Cadence |
|----------|---------------|--------|
| `preflight.md` (compute / mem) | ~10-30 min | Once per ClusterProfile version |
| `preflight.md` (RCCL micro-bench) | ~5-10 min | Same |
| `network.md` (IB topo) | ~5 min (multi-node only) | Same |
| `env_probe.md` (full) | ~10-20 min | Once per env_baseline version |
| `env_probe.md` (subset) | ~5 min | On HANG / drift suspicion |
| `gpu.md` (per-step metrics) | ~0% overhead | Always on |
| `trace.md` | 5-15% overhead | Opt-in only |

## Conventions

- **Tool layer owns invocation**: `preflight.run`, `env_probe.run`,
  `observe.snapshot`, etc. are the entry points; this skill describes
  the **expected output schema** and **pass/fail criteria** the tools
  must implement.
- **Schema lives in the consumer's skill**: e.g. Snapshot fields are
  in `workflow/observe.md`; ClusterProfile fields are in
  `workflow/preflight.md`. This skill points to them; doesn't
  duplicate.
- **Probes must fail fast**: any probe that runs > 30s without
  producing output is killed and reported as `kind: HANG` (env
  baseline suspect) or `kind: CLUSTER` (hardware issue).
- **Reuse aggressively**: PREFLIGHT outputs are reusable across
  sessions per `version` + `age` rule; do NOT re-probe unless
  necessary.

## Sub-file status

| File | Status |
|------|--------|
| `preflight.md` | stub (protocol skeleton) |
| `gpu.md` | stub |
| `network.md` | stub |
| `trace.md` | stub |
| `env_probe.md` | stub |

Stubs are filled when the workflow first needs the detailed protocol;
until then, the tool implementations and the consumer-side schema
docs (`workflow/preflight.md`, `workflow/observe.md`) are sufficient.

## Cross-links

- ClusterProfile schema → `workflow/preflight.md`
- Snapshot schema → `workflow/observe.md`
- Env baseline schema → `workflow/preflight.md` § env_baseline
- Trace consumer → `workflow/diagnose.md` (low-confidence path)

## Scale-aware notes

| Aspect | Single node | Multi node |
|--------|-------------|-----------|
| `preflight.md` content | compute / mem / xGMI only | + inter-node IB topology + cross-node AR/A2A |
| `network.md` | not used | central; IB topology + GID + HCA mapping |
| `env_probe.md` | small flag set | full RCCL flag set; longer wall time |
| `trace.md` overhead | small (intra-node) | larger (cross-node trace aggregation) |
