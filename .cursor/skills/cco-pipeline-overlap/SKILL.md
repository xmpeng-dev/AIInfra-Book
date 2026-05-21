---
name: cco-pipeline-overlap
description: >-
  Implementation techniques for fusing comm + compute into a single persistent
  GPU super-kernel with deterministic chunk-level overlap. Encodes three core
  principles (full MFMA utilization, conflict-free GEMM input via pre-phase
  shuffle, minimal barriers) and concrete patterns to realize each one on AMD
  CDNA3/CDNA4 — software pipelining, LDS swizzle / padding, async direct-to-LDS
  loads, wave specialization, release-acquire ready flags, persistent worker
  queues. Use when writing or reviewing the MonolithMoE forward / backward
  super-kernel, or any HIP kernel that has to overlap XGMI / IPC traffic with
  in-kernel MFMA on MI300X / MI355X.
---

# CCO Pipeline Overlap — Implementation Techniques

## When to Apply

- Writing or reviewing `monolith_moe/v1_bf16/` (forward fuse) and the
  symmetric `v1_bf16/...bwd...` (P3) — the persistent super-kernels that
  fold pack-scatter / permute / FC1 / SwiGLU / FC2 / un-permute / combine
  into one launch.
- Building any HIP kernel that needs to keep MFMA fed while a subset of WGs
  drives XGMI / HIP IPC traffic on the same CUs.
- Diagnosing a fused kernel that compiles + numerically passes but fails
  to beat its un-fused host-driven baseline.

Anchors: see `README.md` §"MFMA GEMM 内核" / §"计算-通信重叠" and
`notes/P2_v1_fwd_progress.md` Milestone 3 for the current numbers
(17.2 ms un-fused, ≤ 9 ms post-fuse target).

Library context (read first if you have not):
`knowledge/libraries/composable-kernel.md` — the CK tile abstraction
and pipeline scheduling primitives this skill borrows from heavily;
`knowledge/libraries/_patterns.md` for the broader pattern catalogue.

---

## Principle 1 — 每次 GEMM 算力打满，除非最后一次

**Goal**: every MFMA-capable cycle in the steady-state K-loop issues a real
MFMA against full-width A/B fragments. Only the trailing K-tile of the
trailing M/N tile may be predicate-masked.

### Technique 1.1 — Two-stage LDS double buffer with prefetch

The minimum viable software pipeline. Compute K-tile `kt` while loading
K-tile `kt+1` into the other LDS buffer.

```
// Prologue
buffer_load_dwordx4  v_a0, ...   ; tile 0 A → VGPR
buffer_load_dwordx4  v_b0, ...   ; tile 0 B → VGPR
s_waitcnt vmcnt(0)
ds_write_b128       lds_A[0], v_a0
ds_write_b128       lds_B[0], v_b0
__syncthreads();                  ; ONE sync, end of prologue

// Steady-state: K-tile kt
for (int kt = 0; kt < K_TILES - 1; ++kt) {
    int cur = kt & 1, nxt = (kt + 1) & 1;

    buffer_load_dwordx4 v_a_nxt, ... (tile kt+1)   ; HBM → VGPR (no wait yet)
    buffer_load_dwordx4 v_b_nxt, ...

    #pragma unroll
    for (int ks = 0; ks < K_STEPS_PER_TILE; ++ks) {
        ds_read_b128 a_frag, lds_A[cur][ks]
        ds_read_b128 b_frag, lds_B[cur][ks]
        s_waitcnt lgkmcnt(0)                       ; wait LDS read
        v_mfma_f32_32x32x16_bf16 acc, a_frag, b_frag, acc
    }

    s_waitcnt vmcnt(0)                             ; wait HBM load
    ds_write_b128 lds_A[nxt], v_a_nxt
    ds_write_b128 lds_B[nxt], v_b_nxt
    __syncthreads();                               ; ONE sync per tile boundary
}

// Epilogue: drain last tile (the only one allowed to be K-masked)
```

Key invariants:
- **One `__syncthreads` per K-tile boundary** (Principle 3).
- `buffer_load` for tile `kt+1` is issued **before** the inner MFMA loop, so
  HBM latency hides behind 4–8 MFMA-32 instructions.
- `ds_read` for K-step `ks` and MFMA for K-step `ks-1` are separated by
  exactly one `s_waitcnt lgkmcnt(0)` — never `s_waitcnt vmcnt(0)`, which
  would force draining HBM loads in the inner loop.

### Technique 1.2 — Three-stage pipeline when K is short

When K_TILES ≤ 4 (e.g. dW K-skinny path with K=2048, K_TILE=128 → 16 tiles,
or backward K=1024 → 8 tiles), two stages leave `s_waitcnt vmcnt(0)` exposed.
Add a third LDS buffer:

```
LDS[0]: tile kt-1   (being read for MFMA)
LDS[1]: tile kt     (loaded last cycle, settling)
LDS[2]: tile kt+1   (being written from VGPR)
                    + buffer_load issued for tile kt+2
```

Cost: 1.5× LDS budget (66 KB → 99 KB on gfx950 BF16 default tile). On
gfx950 with 160 KB LDS/CU you keep 1 WG/CU; on gfx942 (64 KB) you can't
afford it — fall back to two-stage with a smaller tile.

### Technique 1.3 — Wave specialization (producer-consumer)

When M is small (per-expert M < 4 × M_TILE) and you can't fill the K
pipeline, dedicate 1 wave per WG to driving `buffer_load` and the other 3
waves to MFMA + `ds_read`. The producer wave runs ahead by 2 K-tiles,
publishing into LDS via a tiny per-wave ready counter (still intra-WG, so
`s_waitcnt lgkmcnt` + a single `__syncthreads` per K-tile suffices).

This is the AMD analog of CUTLASS "warp specialization". On gfx950 it
typically wins when:
- per-expert M < 1024 (most routing-skewed experts), AND
- HBM bandwidth headroom > 30% (so the producer can sustain 2-tile lookahead).

For balanced DSv3 routing (per-expert M ≈ 2048), uniform waves + Technique
1.1 is enough.

### Technique 1.4 — Predicate-masked K-tail (the "除非最后一次" case)

When per-expert M after routing skew is not a multiple of M_TILE, only the
trailing M-tile is partial. Implement the mask as a **predicate, not a
branch**:

```
// In the inner K-loop, A-fragment load for the trailing tile:
const int row_in_tile = ...;
const bool valid = (m_base + row_in_tile) < m_per_expert;

// buffer_load with bounds check (hardware predicate, no branch divergence):
v_a_frag = valid ? buffer_load_dwordx4(A_ptr + offset)
                 : v_zero;
// MFMA still issues full-width — accumulator just gets +0 from masked rows.
```

The MFMA still executes on the full 32×32×16 fragment; you pay no extra
cycles. Branching out (`if (m_in_tile < M_TILE) { mfma } else { skip }`)
costs ~20 cycles of wave divergence per tile and is forbidden.

### Diagnostic for Principle 1

```bash
rocprof --pmc CU_BUSY,SQ_INSTS_VALU_MFMA,SQ_WAIT_INST_LGKM,SQ_WAIT_INST_VMEM \
        --kernel monolith_moe_fwd_bf16_persistent  ./bench
```

Steady-state target on gfx950: `MFMA / CU_BUSY ≥ 0.85`,
`WAIT_LGKM / total_cycles ≤ 0.05`, `WAIT_VMEM / total_cycles ≤ 0.02`.

---

## Principle 2 — GEMM input 易冲突 → pre-phase 处理成 conflict 较少的 layout

**Goal**: zero LDS bank conflicts in the steady-state `ds_read` of A/B
fragments. Conflicts kill the K-loop schedule from Principle 1 because every
serialized read inflates `lgkmcnt`.

### MI355X / MI300X bank model (the constraint)

- 32 banks × 4 B / bank / cycle → one `ds_read_b128` (16 B) consumes 4 banks
  per lane → a wave of 64 lanes consumes `64 × 4 = 256` bank slots per cycle
  but only `32` exist → 8-cycle natural serialization per `ds_read_b128`.
- Conflict happens when **two lanes in the same wave hit the same bank**
  with **different addresses** (same address is broadcast, free).
- `[M][K]` row-major with `K = 128` BF16 → stride 256 B = 64 banks → lane 0
  and lane 32 collide on bank 0. **Default layout is conflict-prone.**

### Technique 2.1 — Padding (cheapest, default)

Add a few elements between rows so the row stride isn't a multiple of 32
banks:

```c
constexpr int PAD = 4;  // 4 BF16 = 8 B = 2 banks
__shared__ uint16_t lds_A[2][M_TILE][K_TILE + PAD];
__shared__ uint16_t lds_B[2][N_TILE][K_TILE + PAD];
```

Row stride becomes `(K_TILE + 4) × 2 = 264 B = 66 banks → gcd(66, 32) = 2`,
so consecutive lanes spread across all 32 banks. Cost: ~1 KB extra LDS per
tile. **Always start here.**

When PAD = 4 is not enough (e.g. you want `ds_read_b64` for FP8 K_TILE=64),
try PAD = 8 or PAD = 16 — pick the smallest that gives `(stride_B mod 32) ≠ 0`.

### Technique 2.2 — XOR swizzle (transpose-free, no LDS overhead)

When you can't afford padding (LDS-tight gfx942 or 256×256 tile), permute
the column index by XORing in high bits of the row index:

```c
// On write:
int swz_col = col ^ ((row >> 3) & 0x7) << 4;  // swizzle within 128-byte block
ds_write_b128(&lds_A[row][swz_col], v_a);

// On read (same XOR — involution):
int swz_col = col ^ ((row >> 3) & 0x7) << 4;
ds_read_b128(a_frag, &lds_A[row][swz_col]);
```

Effect: the bank touched by lane `l` becomes a function of both row and
col, eliminating the modulo-32 collision. Zero LDS overhead, but the swizzle
constants must match exactly between writer and reader (one off-by-one and
you silently corrupt the GEMM).

Use only when padding doesn't fit. CUTLASS calls this "permuted layout";
the same XOR trick works on AMD because banks are address-modulo-32.

### Technique 2.3 — Producer-side pre-shuffle (the owner's "pre phase")

The pattern the principle is named after: do the layout transform in a
phase that's already paying HBM bandwidth, so the GEMM input is born
conflict-free.

In MonolithMoE forward, the existing 3-pass `permute` (count → prefix →
write) already touches every dispatched token. Modify the **write** pass:

| Before | After |
|---|---|
| `permuted_tokens[pos][:H]` row-major contiguous | `permuted_tokens[pos / 128][:128][:H]` blocked per M-tile, K dimension split into K_TILE chunks of 128 BF16 |
| Per-expert blocks may straddle 128-row boundaries | Each per-expert block padded up to next 128-row boundary (insert zero rows; mask via Technique 1.4) |
| `H` written as one 14336 B row | `H` written as `H / K_TILE = 56` chunks of 256 B each, contiguous |

Now FC1's `buffer_load_dwordx4` for A-tile reads a contiguous 256 B per
lane → 4 banks × 64 lanes = 256 bank slots, perfectly distributed.

You paid one extra HBM write pass during permute (already there) and saved
8–10% wall in the FC1 K-loop.

### Technique 2.4 — Direct-to-LDS load (skip VGPR, gfx942/950)

`buffer_load_lds_b32` writes HBM straight into LDS without ever landing in
VGPR. Saves 4 VGPR per fragment + removes the `ds_write` step (and thus the
`__syncthreads` after it):

```
buffer_load_lds_b32 lds_A[nxt][ks*256 + tid*4], A_ptr + offset, ...
// no ds_write; no s_waitcnt vmcnt at the boundary; LDS coherence
// handled by the next __syncthreads
```

Constraint: the LDS destination must be 32-bit aligned and the write pattern
must be known at compile time (no dynamic index). Combine with Technique
2.1 (PAD) to keep the destination conflict-free on the read side.

Net win: ~12 VGPR freed per WG (typically lifts gfx942 from 1 WG/CU to 2
WG/CU on the FC1 path).

### Technique 2.5 — Vector width matching

`ds_read_b128` (16 B) is the fastest LDS instruction on CDNA but costs 4
banks per lane. If your K_TILE × dtype gives a row stride < 64 B, you're
forcing under-utilized 4-bank reads. Fix by either:
- enlarging K_TILE to keep the row at ≥ 64 B (preferred), or
- switching to `ds_read_b64` and unrolling the K-step 2× to issue twice as
  many reads (last resort, hurts instruction-issue parallelism).

### Diagnostic for Principle 2

```bash
rocprof --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
        --kernel monolith_moe_fwd_bf16_persistent  ./bench
```

Steady-state target: `BANK_CONFLICT / INSTS_LDS ≤ 0.01` in chunks 1..N-1.
Any chunk-0 conflicts (cold LDS) are tolerable.

---

## Principle 3 — Barrier 不能太多

**Goal**: every synchronization in the kernel is the cheapest scope that's
correct. The hierarchy, in cycles:

| Mechanism | Scope | Cost (gfx950) | Use for |
|---|---|---:|---|
| (none, register dependency) | thread | 0 | within a wave |
| `s_waitcnt lgkmcnt(0)` | wave | 1–4 | LDS read-after-write within wave |
| `__syncwarp` / `s_barrier` (wave) | wave | ~4 | rare |
| `__syncthreads` | workgroup | 8–16 | LDS exchange across waves in WG |
| `__threadfence` + atomic flag | device (cross-WG, same GPU) | ~50–200 | producer/consumer WGs on same GPU |
| `__threadfence_system` + atomic_system flag | system (cross-GPU) | 500–2000 | IPC enter/exit barrier only |

Pick the lowest row that's correct. Every step up the table costs ~5–10×.

### Technique 3.1 — Release-acquire flag instead of full sync

For cross-WG-same-GPU coordination (compute WG waiting for comm WG to
finish writing a chunk), don't use `__syncthreads` — it can't cross WG
anyway. Use the release-acquire pair:

```c
// Producer WG (comm), end of phase:
__threadfence();   // device scope — cheap; orders LDS+global writes
if (threadIdx.x == 0)
    __hip_atomic_store(&phase_done[chunk][P1], 1u,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

// Consumer WG (compute), start of phase:
if (threadIdx.x == 0) {
    while (__hip_atomic_load(&phase_done[chunk][P1],
                             __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) == 0u) {
        // spin
    }
}
__syncthreads();   // broadcast "go" within consumer WG
```

`__ATOMIC_ACQUIRE` on the load creates the same happens-before that a full
fence would, but only on the depending load — no global ordering, no
draining the entire memory pipeline.

### Technique 3.2 — Cross-GPU: enter/exit only

System-scope fences are ~10× more expensive than device-scope. Budget:
exactly **two** `__threadfence_system` per super-kernel (the DeepEP-style
enter and exit barriers on `barrier_ptrs[8]`). Per-chunk cross-GPU
coordination uses **already-published** `dispatch_src_ready[8]` flags from
the IPC scatter — the producing GPU pays the system fence once when it
finishes writing its src chunk; the consuming GPU's spin-load is a normal
`atomic_load_acquire_system` (no fence on consume).

### Technique 3.3 — Single-thread fence-then-publish

A `__threadfence` issued by every thread in a WG costs the same as one
issued by a single thread (the fence is per-CU, not per-thread). Wrap the
publish in `if (threadIdx.x == 0)`:

```c
__threadfence();           // every thread issues — same cost, but...
if (threadIdx.x == 0)      // ...only one thread does the atomic publish.
    atomic_store_release(...);
```

The `__threadfence` itself is needed across all threads (it pairs with their
LDS/global writes), but the atomic store is once. This is what
`k1_publish_ready` / `k_publish_gather_ready` already do — preserve the
pattern in fused code.

### Technique 3.4 — Persistent worker queue (avoid per-chunk launch)

When you have N chunks but only M < N compute WGs, don't loop `for chunk in
0..N` in every WG (every WG repeats work). Use an atomic work-stealing
counter:

```c
__shared__ int s_chunk;
while (true) {
    if (threadIdx.x == 0)
        s_chunk = atomicAdd(&g_next_chunk, 1);
    __syncthreads();             // ONE sync to broadcast s_chunk
    if (s_chunk >= NUM_CHUNKS) break;
    process_chunk(s_chunk);
}
```

`g_next_chunk` is a single device-scope atomic, contention bounded by the
number of compute WGs — way cheaper than even one IPC barrier per chunk.

### Technique 3.5 — Replace `__syncthreads` with `s_waitcnt` when you can

If an exchange happens entirely within one wave (64 threads), you don't
need `__syncthreads`; `s_waitcnt lgkmcnt(0)` is enough. Common cases:

- Per-wave reductions (each wave reduces its 64 lanes via DPP / cross-lane
  ops, then one wave aggregates) — only the cross-wave aggregation needs
  `__syncthreads`.
- LDS-based register shuffles where the source and destination lanes are in
  the same wave.

The MFMA fragment loads in Principle 1 are wave-local — that's why the
inner K-loop has zero `__syncthreads`.

### Technique 3.6 — Pull-based comm flag fanout

When N consumer WGs all wait for the same flag, **don't** have the producer
broadcast (1 fence × N atomic stores = N system stores). Have the producer
write **once** to one flag and let consumers spin:

```
Producer WG (1 thread):
    __threadfence_system();
    atomic_store_release_system(&dispatch_src_ready[me], chunk_id + 1);

Consumer WGs (1 thread each):
    while (atomic_load_acquire_system(&dispatch_src_ready[src]) <= chunk_id) {}
```

Spin reads are cache-coherent within the GPU, so 100 spinning consumer WGs
= 100 L2 hits per flag check, not 100 IPC reads.

### Diagnostic for Principle 3

Static check (works without rocprof):

```bash
# Count syncs in the compute path (target: ≤ 8 per chunk per compute WG)
rg -c '__syncthreads|__threadfence' monolith_moe/v1_bf16/monolith_moe_fwd_bf16.hip

# Verify zero system fences inside the chunk loop
rg -n '__threadfence_system' monolith_moe/v1_bf16/monolith_moe_fwd_bf16.hip
# Expected: only inside enter_barrier / exit_barrier helpers
```

Dynamic (gpu-trace-analysis skill):

```bash
rocprofv3 --att --output-format pftrace ./bench
# In Perfetto: look at the comm WG and compute WG timelines.
# Compute WG idle (s_sleep / s_waitcnt) gaps > 2 µs between chunks =
# either a missed flag pair or a system-scope fence in the wrong place.
```

---

## Cross-Cutting Patterns

These are the supporting structures the three principles plug into. They
aren't optimizations themselves — they're the skeleton that lets the
optimizations stack.

### Pattern A — Persistent kernel + WG role partition

Launch grid sized to fill all CUs (`gridDim.x = N_CU × WGs_per_CU`); each
WG reads its `blockIdx.x` once, decides its role from a fixed split:

```
N_COMM = max(1, gridDim.x / 20);   // 5%
N_TAIL = max(1, gridDim.x / 20);   // 5%
role = (wg < N_COMM)            ? COMM
     : (wg >= grid - N_TAIL)    ? TAIL
                                : COMPUTE;
```

Comm WGs and compute WGs get scheduled to **different CUs** (HW round-robin
on `blockIdx.x`), so XGMI traffic and MFMA share the chip but not the same
SIMD units.

### Pattern B — Chunk pipeline with depth = number of sub-phases

For a 4-sub-phase forward (P1 scatter → P2a FC1 → P2b SwiGLU → P2c FC2 →
P3 gather), set `NUM_CHUNKS ≥ 4` so steady state has all 4 stages active
on different chunks simultaneously. For 8-GPU EP, NUM_CHUNKS = 4 (chunks
sized at 2 src GPUs each) is the natural choice — the comm WG fires one
XGMI burst per chunk that's large enough (~7 MB, validated by
`spike_xgmi_burst_sweep`) to hit 95% link efficiency.

### Pattern C — Phase-done flag matrix

```
__device__ uint32_t phase_done[NUM_CHUNKS][NUM_SUB_PHASES];
```

`NUM_CHUNKS × NUM_SUB_PHASES` is small (4 × 4 = 16 flags) → fits in L2
trivially → spin reads are free. Don't fan out to per-(chunk, src, expert)
— Layout C guarantees expert sub-blocks are contiguous inside each (src,
chunk) region; one flag per (chunk, sub-phase) is enough granularity.

### Pattern D — Async pipeline depth tuning

Two-stage (Technique 1.1) is the default. Bump to three-stage (Technique
1.2) only when:
1. K_TILES per GEMM ≤ 4 (short K), AND
2. LDS budget allows (gfx950: 99 KB ≤ 160 KB, fits 1 WG/CU).

In MonolithMoE forward: FC1 K=7168 / 128 = 56 K-tiles → two-stage. Backward
dW K=2048 / 128 = 16 K-tiles → still two-stage. Backward dW with M_TILE=256
sweep may push below the threshold → re-evaluate per shape.

---

## Quick Reference

| Symptom | Likely principle violated | First technique to try |
|---|---|---|
| MFMA busy% < 70% in steady state | 1 | Technique 1.1 (verify two-stage prefetch is actually issuing `buffer_load` before MFMA loop) |
| `WAIT_LGKM` > 10% of cycles | 2 | Technique 2.1 (add LDS PAD = 4) |
| `WAIT_VMEM` > 5% of cycles | 1 or 3 | Technique 1.2 (3-stage pipeline) or Technique 3.1 (replace fence with release-acquire) |
| Compute WG idle gaps > 2 µs between chunks | 3 | Technique 3.6 (pull-based flag) + Technique 3.2 (no system fence per chunk) |
| Per-chunk wall scales linearly with NUM_CHUNKS | not pipelined at all | Pattern B (verify role split puts comm + compute on different CUs) |
| Numerical drift after fuse | usually 2 (swizzle off-by-one) | Re-verify Technique 2.2 XOR constants on writer/reader |
