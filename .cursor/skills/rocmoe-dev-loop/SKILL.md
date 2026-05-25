---
name: rocmoe-dev-loop
description: >-
  RocMoE-v2 super-kernel project development loop. Use when working in
  `~/workspace/RocMoE/` or anytime the user says "继续 RocMoE 开发", "下一个
  milestone", "M1/M2/...", "跑 RocMoE bench", "写 RocMoE note", or asks to
  optimize the RocMoE-v2 fused MoE kernel. Encodes the four mandatory
  development rules: (1) every change must be measured against a fixed
  PyTorch+RCCL baseline at the same workload as RocMoE-bak's bench_e2e;
  (2) every phase must have a HIP test case that is rerun after edits;
  (3) every optimization round (success or failure) must produce a
  progress note whose filename carries an UP/DOWN/FLAT flag; (4) all
  compile + run happens inside the `xiaoming-dev` container on a
  squeue-allocated node, never on the login machine.
---

# RocMoE-v2 dev loop

Authoritative project context: `~/workspace/slab/notes/rocmoe/README.md` and
`~/workspace/slab/notes/rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md`
(architecture proposal).

## Project anchors

| anchor | path |
|---|---|
| Working tree (clean, where new code lives) | `~/workspace/RocMoE/` |
| Architecture-cleanest reference (Layout-P / pull / scoreboard) | `~/workspace/RocMoE-bak/` |
| Highest-perf reference (`mfma_tile.h` 99.3% MFMA, push path) | `~/workspace/MonolithEP/` |
| Production sibling (4.82 ms BF16, Primus integrated) | `~/workspace/MMOE/` (kernel) + `~/workspace/slab/notes/monolith-moe/` (notes) |
| Project notes (single source of truth for status) | `~/workspace/slab/notes/rocmoe/` (also reachable as `~/workspace/RocMoE/notes/` symlink) |
| This skill | `~/workspace/slab/.cursor/skills/rocmoe-dev-loop/SKILL.md` |

## Hardware default

8 × AMD Instinct MI355X (gfx950, CDNA4) on a single SLURM node, XGMI
all-to-all. ROCm 7.2 toolchain, hipcc, PyTorch 2.12+rocm7.1, Primus,
container `xiaoming-dev` (image `docker.io/rocm/primus:v26.2`).

## Mandatory rules (the loop)

### Rule 1 — Fixed PyTorch+RCCL baseline at the same workload

Every benchmark run must report two numbers side-by-side:

| line | meaning |
|---|---|
| `[bench_e2e]` | RocMoE-v2 super-kernel, 4-stage host-driven or super mode |
| `[baseline_pt_rccl]` | reference PyTorch + `torch.distributed.all_to_all_single` MoE forward, **same workload** |

**Workload contract** (must match exactly between super and baseline):

| param | DSv3 default | knob (cli flag) |
|---|---|---|
| ranks | 8 (single node) | `--ranks` |
| num_experts (global) | 256 (32 / GPU) | `--experts` |
| topk | 8 | `--topk` |
| H (hidden) | 7168 | `--H` |
| F (FFN intermediate) | 2048 | `--F` |
| dtype | bf16 (fp8 / mxfp8 later) | `--dtype` |
| T_src (per-GPU input tokens) | 2048 (the production sweet spot)<br/>also test 512 / 4096 / 8192 | `--T` |
| seed | 0xC0FFEE | `--seed` |

The PyTorch baseline, `~/workspace/RocMoE/baselines/pt_rccl_moe.py`,
implements the textbook EP MoE forward (router → permute →
`all_to_all_single` → grouped GEMM → SwiGLU → grouped GEMM →
`all_to_all_single` → un-permute), warms up 10 iters, times 50 iters
with `cuda.Event`. The HIP bench `benchmarks/bench_e2e.hip` accepts the
same workload args and prints in the same format. A wrapper
`benchmarks/run_bench_pair.sh` runs both and prints the speedup.

**Never** report a perf number without running the baseline at the same
workload **in the same job**. NIC weather, peer-access state, and HBM
ECC scrubs change between SLURM allocations — the only honest pair is
same-allocation, same-iter-count, same-workload.

### Rule 2 — Per-phase test case, rerun after every code edit

| phase | C++ test binary | semantic check |
|---|---|---|
| 0 ROUTE_META | `tests/test_route_meta` | per-(src, expert) header == reference compute |
| 1 DISPATCH_PULL | `tests/test_dispatch` | local `block_layout` token slots match a CPU reference re-permutation; `block_ready` bitmap correct |
| 2 FC1 | `tests/test_gemm_fc1` | bf16 `Y = X @ W1.T` vs naive bf16 matmul, max-abs ≤ 1e-2 |
| 2 SwiGLU | `tests/test_swiglu` | `silu(gate)*up` vs `torch.nn.functional.silu`, max-abs ≤ 1e-3 |
| 2 FC2 | `tests/test_gemm_fc2` | bf16 `Y = X @ W2.T`, max-abs ≤ 1e-2 |
| 3 FC2_PUSH | `tests/test_fc2_push` | per-(src, t) shard at peer combine_buf == local FC2 output |
| 4 COMBINE_PULL | `tests/test_combine` | final_out[t] == Σ_src topk_w[t,k] * combine_buf[src][t] |
| e2e | `tests/test_e2e` | full pipeline equals reference PyTorch implementation, max-abs ≤ 1e-2 |
| smoke | `tests/smoke_super_kernel` | super-kernel exits cleanly, no hang, no NaN |

CMakeLists registers each as a `ctest` target. Workflow:

```bash
cmake --build build -j
ctest --test-dir build -V
```

If `ctest` fails, **stop and write a note immediately** with `_CRASH` or
`_WRONG` flag (see Rule 3) — do not move on to the next milestone.

### Rule 3 — Every optimization round writes a note with a flag

Filename: `~/workspace/slab/notes/rocmoe/YYYY-MM-DD_HHMM_<flag>_<topic>.md`

Flags (one per filename, mandatory):

| flag | meaning |
|---|---|
| `UP`   | super wall improved by ≥ 1 % vs previous-best (vs baseline ratio improved) |
| `DOWN` | super wall regressed by ≥ 1 % |
| `FLAT` | within ±1 % noise; no measurable change |
| `CRASH` | ctest / build / runtime failure; revert then debug |
| `WRONG` | numerics regression (max-abs > tolerance) |
| `BASELINE` | initial number with no comparison yet |

Examples:

- `2026-05-21_1500_BASELINE_m0_mfma_tile_ported.md`
- `2026-05-22_0930_UP_m1_layout_p_dispatch_landed.md` (super 13.1→11.4 ms)
- `2026-05-22_1430_DOWN_m1b_block_size_64_revert.md` (11.4→13.6 ms)
- `2026-05-22_1830_FLAT_m1c_atomic_load_acquire.md`
- `2026-05-23_1010_CRASH_m2_state_machine_deadlock.md`

Note body follows `~/workspace/RocMoE/.cursor/rules/40-notes-style.mdc`
(TL;DR → Background → What I did → Result → Interpretation → Next).
The Result section **must** quote both super and baseline numbers from
the same SLURM allocation, with iteration count and noise bound.

After writing the note, append a row to
`~/workspace/slab/notes/rocmoe/README.md` progress timeline.

### Rule 4 — Build and run inside `xiaoming-dev`, on a squeue-allocated node

Login machine has no GPU; never run `cmake` / `ctest` / `bench` there.

Step-by-step at the start of every session, and any time the previous
allocation may have ended:

```bash
# 1. Find xiaoming's running allocation (or ask user to start one).
squeue -u xiaoming -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
# 2. Pick an R-state job, expand its node list.
scontrol show hostnames <NodeList>
# 3. Pick one node (8 × MI355X) for compile + run.
NODE=mi355-gpu-XX
# 4. SSH (BatchMode, no password prompts).
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no $NODE 'hostname'
# 5. Ensure xiaoming-dev container is up.
ssh -o BatchMode=yes ... $NODE 'podman ps --filter name=^xiaoming-dev$'
# 6. Wait for any active python/torchrun/cmake/ctest before launching.
ssh ... $NODE 'podman exec xiaoming-dev bash -lc "pgrep -af python|torchrun|cmake|ctest"'
# 7. Run the actual work inside the container, from the workspace path:
ssh ... $NODE 'podman exec xiaoming-dev bash -lc "cd \$HOME/workspace/RocMoE && cmake -S . -B build && cmake --build build -j && ctest --test-dir build -V"'
```

Detailed primitives are in
`~/workspace/RocMoE/.cursor/skills/slurm-xiaoming-dev-container/SKILL.md`
and `~/workspace/RocMoE/.cursor/skills/ssh-node-xiaoming-dev-container/SKILL.md`
— this skill defers to them for exact commands.

Helper script `scripts/dev_on_node.sh` wraps steps 4-7. Usage:

```bash
bash scripts/dev_on_node.sh build         # cmake configure + build
bash scripts/dev_on_node.sh test          # ctest --output-on-failure
bash scripts/dev_on_node.sh bench M0      # run M<n> bench preset
bash scripts/dev_on_node.sh bench M0 4096 # override T_src
bash scripts/dev_on_node.sh shell         # interactive shell in container
```

The script itself runs on the login machine; it issues SSH + podman exec
under the hood. Set `ROCMOE_NODE` to override auto-pick.

## Milestone schedule (from architecture design §5)

| # | name | acceptance | flag goal |
|---|---|---|---|
| **M0** | bootstrap repo + cherry-pick `mfma_tile.h` from MonolithEP, standalone GEMM bench | DSV3 grouped GEMM ≥ 950 TFLOPS / GPU | `BASELINE` |
| **M1** | Layout-P + 64-bit `block_ready` bitmap + receiver-pull dispatch | dispatch wall ≤ 1 ms / 8 GPU @ T_src=2048; e2e correctness PASS | `UP` |
| **M2** | persistent state machine + role work-steal | super-kernel 5 phase end-to-end PASS, 0 hang | `BASELINE` for state-machine variant |
| **M3** | FC1+SwiGLU+FC2 in-LDS fused (full DTOLDS) | super wall ≤ 9 ms @ T_src=2048 | `UP` |
| **M4** | wave specialization (LOADER/MFMA 2-2 split) | MFMA util ≥ 95 % (rocprof) | `UP` |
| **M5** | atomic-free combine pull (register reduce) | combine wall ≤ 0.5 ms | `UP` |
| **M6** | mxfp8 weights | super wall ≤ 5.5 ms @ T_src=2048 | `UP` |
| **M7** | per-tile-class K_TILE template | DSV3 grouped GEMM ≥ 1.05 TFLOPS / GPU | `UP` |
| **M8** | decomposed backward | bwd wall ≤ 1.2 × fwd wall | `UP` |

Each milestone is a sequence of micro-rounds; each round → one note.

## What this skill does NOT cover

- Distillation of new operator libraries → use `distill-operator-repo`.
- Paper reading → use `read-paper` or `paper-deep-analysis`.
- Cross-project weekly status → use `weekly` directory and corresponding skill.
- Generic SLURM cluster health checks → use `slurm-idle-node-check`.

## Quick start (resuming a session)

1. Read `~/workspace/slab/notes/rocmoe/README.md` to see current
   status / last completed milestone.
2. Read the most recent progress note in
   `~/workspace/slab/notes/rocmoe/` for the immediate next step.
3. Run `squeue -u xiaoming` and pick a node (Rule 4).
4. Run `bash scripts/dev_on_node.sh build && bash scripts/dev_on_node.sh test`
   to confirm a green baseline.
5. Open the next milestone work, edit, build, test, bench, write note.
