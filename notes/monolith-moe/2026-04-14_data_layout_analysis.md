# MoE IPC Buffer Layout Analysis for Comm-Compute Overlap

> Theoretical analysis of data flow and buffer layout options for the
> MonolithMoE persistent super-kernel.  Goal: find the IPC workspace layout
> that maximizes comm-compute overlap on an 8-GPU MI355X node.

---

## 1. Complete Data Flow Trace (DSV3 671B, 4096 tokens)

### 1.1 Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size (H) | 7168 | |
| ffn_hidden_size (F) | 2048 | per-expert, half of gated width |
| num_experts (E) | 256 | total across 8 GPUs |
| experts_per_gpu (E_local) | 32 | 256 / 8 |
| top_k (K) | 8 | |
| tokens_per_gpu (T) | 4096 | |
| EP size | 8 | one node, fully-connected XGMI |
| dtype | bf16 | 2 bytes per element |

### 1.2 Per-Phase Tensor Inventory

**Gate (standalone kernel, before super-kernel):**

| Tensor | Shape | Size | Location |
|--------|-------|------|----------|
| hidden_states | [4096, 7168] | 56.0 MB | Local HBM |
| gate_weight | [7168, 256] | 3.5 MB | Local HBM |
| gate_scores | [4096, 256] | 4.0 MB (fp32) | Local HBM |
| topk_expert_ids | [4096, 8] | 0.125 MB (int32) | Local HBM |
| topk_weights | [4096, 8] | 0.125 MB (fp32) | Local HBM |

**Phase 1 — Dispatch (per source→dest GPU pair, assuming uniform routing):**

Each GPU sends ~4096 token-expert pairs, distributed ~512 per destination GPU.

| Tensor | Shape | Size per dest | Total XGMI write |
|--------|-------|---------------|-------------------|
| dispatch_tokens | [512, 7168] | 7.0 MB | 7 × 7.0 = 49.0 MB |
| dispatch_expert_ids | [512] | 2 KB | negligible |
| dispatch_src_ids | [512] | 2 KB | negligible |
| dispatch_counts | [1] | 4 B | negligible |

Total XGMI dispatch traffic **per GPU**: ~49 MB outbound (7 remote peers).
System-wide: 8 × 49 = 392 MB.

**Phase 2 — Expert Compute (per GPU, local):**

Each GPU receives ~4096 token-expert pairs across 32 local experts.
Per expert (uniform): 128 tokens.

| Operation | A shape | B shape | Output | FLOPs |
|-----------|---------|---------|--------|-------|
| FC1 | [128, 7168] | [7168, 4096] | [128, 4096] | 7.52 GF |
| SwiGLU | [128, 4096] → [128, 2048] | — | [128, 2048] | — |
| FC2 | [128, 2048] | [2048, 7168] | [128, 7168] | 3.76 GF |

Per expert: 11.3 GFLOPs.  All 32 experts: **361 GFLOPs** per GPU.
Total across 8 GPUs: 2.89 TFLOPs (matches benchmark).

HBM traffic per expert (compute-only, assuming weights cached):

| Read/Write | Bytes |
|------------|-------|
| Read A (tokens) | 128 × 7168 × 2 = 1.75 MB |
| Read W1 | 7168 × 4096 × 2 = 56.0 MB |
| Write FC1 out | 128 × 4096 × 2 = 1.0 MB |
| Read FC1 (SwiGLU) | 1.0 MB |
| Write SwiGLU out | 128 × 2048 × 2 = 0.5 MB |
| Read SwiGLU out | 0.5 MB |
| Read W2 | 2048 × 7168 × 2 = 28.0 MB |
| Write FC2 out | 1.75 MB |
| **Total per expert** | **~90.5 MB** |
| **Total 32 experts** | **~2.9 GB** |

**Phase 3 — Combine (per expert-GPU→source-GPU pair):**

Mirror of dispatch.  Same volume: ~49 MB XGMI per GPU outbound.

### 1.3 Data Movement Diagram

```
GPU_0                                                    GPU_j (j=1..7)
┌──────────────────┐                                 ┌──────────────────┐
│ hidden[4096, H]  │                                 │                  │
│        ↓         │                                 │                  │
│ Gate → topk_ids  │                                 │                  │
│        ↓         │                                 │                  │
│ ┌──────────────┐ │   XGMI: 7 MB per dest GPU      │ ┌──────────────┐ │
│ │ Pack by dest │─┼──────────────────────────────→  │ │ dispatch_buf │ │
│ │ GPU + scatter│ │   token data + expert_ids       │ │ [src=0][..]  │ │
│ └──────────────┘ │                                 │ └──────┬───────┘ │
│                  │                                 │        ↓         │
│                  │                                 │ Expert GEMM      │
│                  │                                 │ FC1→SwiGLU→FC2   │
│                  │                                 │        ↓         │
│ ┌──────────────┐ │   XGMI: 7 MB per source GPU    │ ┌──────────────┐ │
│ │ combine_buf  │←┼──────────────────────────────── │ │ Pack results │ │
│ │ [src=j][..]  │ │   expert results + weights      │ │ + scatter    │ │
│ └──────┬───────┘ │                                 │ └──────────────┘ │
│        ↓         │                                 │                  │
│ Weighted sum     │                                 │                  │
│ → output[4096,H] │                                 │                  │
└──────────────────┘                                 └──────────────────┘
```

### 1.4 Benchmark Reference (8× MI355X)

| Component | Latency | Category |
|-----------|---------|----------|
| prep (gate GEMM) | 0.035 ms | Overhead |
| dispatch | 1.427 ms | A2A |
| sort+index | 0.116 ms | Overhead |
| fc1 | 2.727 ms | GEMM |
| act (SwiGLU) | 0.161 ms | Overhead |
| fc2 | 1.719 ms | GEMM |
| combine | 2.242 ms | A2A |
| topk_sum | 0.038 ms | Overhead |
| **TOTAL** | **8.466 ms** | |
| **A2A** | **3.669 ms (43.3%)** | |
| **GEMM** | **4.446 ms (52.5%)** | |
| **Overhead** | **0.350 ms (4.1%)** | |

**Key observation:** GEMM (4.4 ms) > A2A (3.7 ms), so **100% of A2A is
theoretically hideable** if the pipeline has sufficient granularity and
the layout avoids extra gather/sort overhead in the overlap path.

---

## 2. Overlap Pipeline Model

### 2.1 WG Role Partitioning

The persistent super-kernel partitions WGs into three roles:

```
Grid: N persistent WGs
  ┌──────────────────────────────────────────────┐
  │ WG [0 .. C-1]      → Comm: Pack + Scatter    │  Phase 1
  │ WG [C .. N-C-1]    → Compute: Expert GEMM    │  Phase 2
  │ WG [N-C .. N-1]    → Tail: Gather + Combine  │  Phase 3
  └──────────────────────────────────────────────┘
```

Comm WGs and Compute WGs run on **different CUs**, so XGMI writes and
MFMA instructions execute in true parallel (not time-sliced).

### 2.2 Current Pipeline (Layout A: Source-GPU Chunked)

```
Timeline on GPU_j (receiving side):

Phase 1 (other GPUs' comm WGs write to GPU_j):
  GPU_0 writes:  [=========]
  GPU_1 writes:      [=========]
  GPU_2 writes:          [=========]
  ...                                        (7 overlapping XGMI streams)

Phase 2 (GPU_j compute WGs):
  [wait] [chunk_0: scan+gather+GEMM] [chunk_1: scan+gather+GEMM] ... [chunk_7]
          ↑ signal dispatch_ready[0]   ↑ signal dispatch_ready[1]

Phase 3 (GPU_j tail comm WGs):
  [wait]                              [combine chunk_0] [combine chunk_1] ...
                                       ↑ signal combine_ready[0]
```

**Pipeline stages:** 8 (one per source GPU).

**Per-stage cost breakdown:**

| Sub-step | Time (estimated) | Bottleneck |
|----------|-----------------|------------|
| Wait for dispatch_ready[src] | variable | XGMI latency |
| Scan expert_ids to group tokens | ~0.1 ms | HBM bandwidth |
| Gather scattered tokens per expert | ~0.2 ms | Random HBM reads |
| FC1 GEMM (all experts, this chunk) | ~0.34 ms | MFMA compute |
| SwiGLU | ~0.02 ms | HBM bandwidth |
| FC2 GEMM (all experts, this chunk) | ~0.21 ms | MFMA compute |
| Write combine results to peer | — | XGMI (by tail WGs) |
| **Subtotal per stage** | **~0.87 ms** | |

**Pipeline fill:** First stage has no overlap (compute WGs idle while waiting
for the first chunk).  Fill penalty ≈ 1 / 8 = 12.5% of total.

**Pipeline drain:** Last stage's combine has no overlap with compute.
Drain penalty ≈ 1 / 8 = 12.5%.

**Gather overhead:** ~0.3 ms per stage × 8 stages = **~2.4 ms total wasted**
on scanning + gathering tokens that are not sorted by expert.

### 2.3 What Determines Overlap Quality

Three factors control how well A2A hides behind GEMM:

1. **Pipeline granularity (P):** More stages → smaller fill/drain penalty.
   Ideal: fill + drain < 5% of total time.
   - P=8: fill+drain = 25% → poor
   - P=32: fill+drain = 6.25% → acceptable
   - P=128+: fill+drain < 2% → excellent

2. **Gather overhead (G):** Any token reordering between dispatch arrival and
   GEMM input adds latency to the critical path.  If tokens arrive already
   sorted by expert, G=0.

3. **XGMI write pattern:** Contiguous writes achieve higher bandwidth than
   scattered writes.  The sender should pack tokens into large contiguous
   blocks before writing to peer HBM.

---

## 3. Layout Options

### Layout A: Source-GPU Chunked (Current)

```
IPC dispatch workspace on GPU_j:

  dispatch_tokens[src=0][ slot_0 | slot_1 | ... | slot_N ][H]
  dispatch_tokens[src=1][ slot_0 | slot_1 | ... | slot_N ][H]
  ...
  dispatch_tokens[src=7][ slot_0 | slot_1 | ... | slot_N ][H]

  dispatch_expert_ids[src][slot]  — local expert id for each token
  dispatch_ready[src]            — set by sender when entire chunk written
```

**Sender logic:** Pack all tokens destined for GPU_j into a contiguous block.
No sorting by expert — tokens appear in the order they occur in
`topk_expert_ids`.  One large contiguous XGMI write per (src, dest) pair.

**Receiver logic:** Wait for `dispatch_ready[src]`.  Then **scan all tokens**
to group them by expert (LDS histogram + prefix-sum or linear scan).
**Gather scattered tokens** into contiguous per-expert buffers before GEMM.

| Property | Value |
|----------|-------|
| Pipeline stages | 8 |
| Gather overhead | **HIGH** — O(N) scan + random gather per stage |
| XGMI write pattern | Excellent — one large contiguous write per pair |
| Ready flags | 8 per GPU |
| Sender complexity | Low |
| Receiver complexity | High (scan + gather) |
| Fill + drain | 25% |

**Verdict:** Simple sender, but the per-stage gather overhead (~0.3 ms × 8)
adds ~2.4 ms to the critical path, severely limiting overlap benefit.
The 8-stage pipeline also has high fill/drain overhead.

### Layout B: Expert-First (Global Sort)

```
IPC dispatch workspace on GPU_j:

  expert_region[e=0] [ token_0 | token_1 | ... ][H]
  expert_region[e=1] [ token_0 | token_1 | ... ][H]
  ...
  expert_region[e=31][ token_0 | token_1 | ... ][H]

  expert_counts[e]   — total tokens received for expert e (across all senders)
  expert_ready[e]    — set when ALL senders have finished writing expert e
  write_cursor[e]    — atomic counter for concurrent sender writes
```

**Sender logic:** Sort local tokens by (dest_gpu, expert_id).
For each (dest, expert) pair, atomically claim a write position in
`write_cursor[expert]` and write tokens contiguously starting there.
After all tokens for expert_e are written, the **last sender** signals
`expert_ready[e]`.

**Receiver logic:** Wait for `expert_ready[e]`.  Tokens are **already
contiguous per expert** — directly feed to GEMM, no gather needed.

| Property | Value |
|----------|-------|
| Pipeline stages | 32 (one per expert) |
| Gather overhead | **ZERO** — tokens arrive pre-sorted |
| XGMI write pattern | **POOR** — multiple senders write to same expert region, interleaved |
| Ready flags | 32 per GPU |
| Sender complexity | High (sort + cross-GPU atomic contention) |
| Receiver complexity | Low |
| Fill + drain | 6.25% |

**Write conflict analysis:**
For expert_e, up to 8 GPUs write concurrently.  Each uses `atomicAdd_system`
on `write_cursor[e]` to claim a slot.  At 8 contenders, this is ~200 ns per
atomic round-trip over XGMI.  For 128 tokens × 8 senders = 1024 atomics
per expert.  But the atomics are only for the cursor — the actual data
writes go to **different slot offsets**, so they don't conflict.

The real problem is that **different senders' writes to the same expert
region are interleaved in time**, creating a fragmented XGMI write pattern
that underutilizes link bandwidth.

**Verdict:** Excellent for the receiver (zero gather, fine pipeline), but
the sender-side cross-GPU atomic contention and fragmented XGMI writes
make this impractical.  The "last sender signals" protocol is also
difficult to implement correctly without a global barrier.

### Layout C: Source × Expert (Two-Level Sort)

```
IPC dispatch workspace on GPU_j:

  src_chunk[src=0]:
    expert_block[e=0][ token_0 | ... | token_N ][H]
    expert_block[e=1][ token_0 | ... | token_N ][H]
    ...
    expert_block[e=31][ ... ][H]

  src_chunk[src=1]:
    expert_block[e=0][ ... ][H]
    ...

  expert_counts[src][e]   — tokens from src for expert e
  chunk_ready[src]        — set by sender when its entire chunk written
```

Memory layout (linearized):
```
offset = src * max_recv_per_src * H
       + expert_prefix[src][e] * H
       + slot * H

where expert_prefix[src][e] = cumulative token count for experts 0..e-1
                               from source src
```

**Sender logic:** Sort local tokens by (dest_gpu, expert_id).
Write each expert's tokens as a contiguous sub-block within the
source chunk.  **No cross-GPU atomics** — each sender writes only to
its own region `src_chunk[rank]`.  One large contiguous XGMI write,
but now expert sub-blocks are contiguous within it.

**Receiver logic:** Wait for `chunk_ready[src]`.  Then iterate experts:
tokens for expert_e from source_src are at a known contiguous offset.
**No scatter/gather** — direct pointer into the dispatch buffer.

Compute WGs can process expert_e from source_0, then expert_e from source_1,
etc., accumulating GEMM tiles across sources for the same expert.

| Property | Value |
|----------|-------|
| Pipeline stages | 8 (per source), but no gather overhead |
| Gather overhead | **ZERO** — tokens pre-sorted by expert within each chunk |
| XGMI write pattern | **Excellent** — one large contiguous write per (src, dest) |
| Ready flags | 8 per GPU |
| Sender complexity | Medium (local sort by expert, no cross-GPU atomics) |
| Receiver complexity | Low (indexed access by [src][expert]) |
| Fill + drain | 25% (but could pipeline at sub-chunk level) |

**Optimization — sub-chunk signaling:** Instead of one `chunk_ready[src]`
per source, use `expert_ready[src][e]` (8 × 32 = 256 flags).  Each sender
signals per-expert completion.  Compute WGs can start expert_e as soon as
ANY source has expert_e ready, achieving up to **8 × 32 = 256 effective
pipeline stages** without changing the write pattern.

With sub-chunk signaling:

| Property | Value |
|----------|-------|
| Effective pipeline stages | up to 256 |
| Fill + drain | < 1% |
| Flag count | 256 per GPU (still < 1 KB) |

**Verdict:** Best balance of all factors.  The sender does only a local sort
(no cross-GPU coordination), XGMI writes are large and contiguous, the
receiver gets zero-gather pre-sorted data, and sub-chunk signaling
unlocks very fine-grained overlap.

### Layout D: Tile-Aligned Streaming

```
IPC dispatch workspace on GPU_j:

  tile_buf[tile_0][ row_0 | row_1 | ... | row_{M_TILE-1} ][H]
  tile_buf[tile_1][ ... ][H]
  ...
  tile_buf[tile_511][ ... ][H]

  tile_meta[tile_i] = { expert_id, num_valid_rows, src_gpu, src_token_ids[M_TILE] }
  tile_ready[tile_i]                — set when tile data is written
  tile_write_cursor                 — atomic counter for global tile allocation
```

**Sender logic:** Sort tokens by (dest_gpu, expert_id).
Group into M_TILE=64 sized tiles.  Atomically claim a tile slot via
`tile_write_cursor`.  Write tile data + metadata.  Signal `tile_ready`.

**Receiver logic:** Compute WGs poll `tile_ready` for the next tile.
Each tile is a complete GEMM input — directly feed M_TILE rows to
`mfma_gemm_tile()`, no gathering or reordering.

| Property | Value |
|----------|-------|
| Pipeline stages | ~512 (32768 tokens / 64) |
| Gather overhead | **ZERO** — tiles are GEMM-ready |
| XGMI write pattern | Medium — tiles are contiguous but small (64 × 7168 × 2 = 896 KB each) |
| Ready flags | ~512 per GPU |
| Sender complexity | High (tile packing + global atomic cursor + metadata) |
| Receiver complexity | Low (poll + direct GEMM) |
| Fill + drain | < 0.5% |
| Metadata overhead | ~512 × (4 + 4 + 4 + 256) bytes = ~137 KB |

**XGMI efficiency concern:** Each tile is 896 KB, written in one burst.
MI355X XGMI achieves peak bandwidth at ~2 MB+ transfer sizes.  896 KB
is marginal — expect ~70-80% of peak link BW per tile write.

**Tile fragmentation:** Under non-uniform routing, some experts may get
fewer than M_TILE tokens.  Partial tiles waste MFMA cycles (lanes masked
off).  With 128 tokens/expert and M_TILE=64, we get exactly 2 full tiles
per expert — no fragmentation under uniform routing.  But skewed routing
(common in practice) will produce many partial tiles.

**Atomic contention:** All 8 senders compete for `tile_write_cursor` on
the receiver's HBM via XGMI atomics.  At ~200 ns per XGMI atomic
round-trip, 512 tiles means ~100 μs of serialized atomic overhead.
Tolerable, but adds to dispatch latency.

**Verdict:** Finest pipeline granularity and zero gather overhead, but
the tile metadata overhead, XGMI atomic contention, partial tile waste,
and sub-optimal XGMI burst size create practical challenges.  Best
suited if M_TILE is large enough for efficient XGMI writes and routing
is roughly uniform.

### Layout E: Expert-Contiguous with Per-Source Zones

```
IPC dispatch workspace on GPU_j:

  expert_buf[e=0]:
    zone[src=0][ token_0 | ... ][H]
    zone[src=1][ token_0 | ... ][H]
    ...
    zone[src=7][ token_0 | ... ][H]

  expert_buf[e=1]:
    zone[src=0][ ... ][H]
    ...

  zone_count[e][src]    — tokens from src for expert e
  zone_ready[e][src]    — set by sender per (expert, src) pair
```

**Sender logic:** Sort tokens by (dest_gpu, expert_id).  For each
(dest, expert) pair, write tokens into `expert_buf[expert].zone[rank]`.
**No cross-GPU atomics** — each sender writes only to its own zone.

**Receiver logic:** For each expert_e, poll `zone_ready[e][src]` for each
source.  As soon as one source's zone is ready, start GEMM on those
tokens.  Accumulate across sources incrementally.

| Property | Value |
|----------|-------|
| Pipeline stages | 256 (32 experts × 8 sources) |
| Gather overhead | **ZERO** |
| XGMI write pattern | **POOR** — each (src, dest, expert) write is very small |
| Ready flags | 256 per GPU |
| Sender complexity | Medium (same sort as Layout C) |
| Receiver complexity | High (poll 256 flags, manage partial-expert GEMM) |
| Fill + drain | < 1% |

**XGMI efficiency analysis:**
Per (src, dest, expert) write size: 128/8 × 7168 × 2 = 16 tokens × 14 KB
= 224 KB.  This is **far below** XGMI peak efficiency threshold (~2 MB).
Expect only 30-40% of peak link bandwidth.  The fine granularity that
enables maximum overlap also creates the worst XGMI write pattern.

**Partial-expert GEMM complexity:** Starting GEMM with only 16 tokens
(one source's contribution to one expert) means M=16 in the MFMA tile.
With M_TILE=64 and MFMA_M=32, the 16-row tile wastes 75% of MFMA
throughput.  The receiver must either:
- Wait for multiple sources to accumulate enough tokens (defeating the purpose), or
- Use a smaller M_TILE (reducing MFMA efficiency), or
- Concatenate zones from multiple sources into one buffer (reintroducing gather)

**Verdict:** Maximum theoretical overlap, but the tiny XGMI writes and
poor MFMA utilization on partial tiles make this impractical.  The
overhead from 256 flags and partial GEMM management exceeds the benefit
of fine-grained overlap.

---

## 4. Quantitative Comparison

### 4.1 Summary Table

| Property | A (Current) | B (Expert-First) | C (Src×Expert) | D (Tile-Aligned) | E (Expert+Zone) |
|----------|:-----------:|:-----------------:|:---------------:|:-----------------:|:----------------:|
| Pipeline stages | 8 | 32 | 8 (→256 w/ sub-signal) | ~512 | 256 |
| Gather overhead | **2.4 ms** | 0 | 0 | 0 | 0 |
| XGMI write size | 7 MB | fragmented | 7 MB | 896 KB | 224 KB |
| XGMI efficiency | ~95% | ~40% | ~95% | ~75% | ~35% |
| Ready flags | 8 | 32 | 8 (or 256) | ~512 | 256 |
| Cross-GPU atomics | No | Yes | No | Yes | No |
| Sender sort needed | No | Yes (+ atomic) | Yes (local only) | Yes (+ atomic) | Yes (local only) |
| MFMA tile fill | N/A (gather) | 100% | 100% | 100% (uniform) | 25% (16 rows) |
| Code complexity | Low | High | **Medium** | High | Very high |

### 4.2 Estimated Overlap Efficiency

Using a simple pipeline model: overlap_ratio = 1 - (fill + drain + gather) / total_time

| Layout | Fill+Drain | Gather | Estimated Overlap |
|--------|-----------|--------|-------------------|
| A (Current) | 2 × 0.87 ms = 1.74 ms | 2.4 ms | ~0% (gather dominates) |
| B (Expert-First) | 2 × 0.14 ms = 0.28 ms | 0 | ~92% |
| C (Src×Expert, 8-stage) | 2 × 0.55 ms = 1.10 ms | 0 | ~70% |
| C (Src×Expert, 256-stage) | 2 × 0.017 ms = 0.03 ms | 0 | **~99%** |
| D (Tile-Aligned) | 2 × 0.009 ms = 0.02 ms | 0 | ~99% |
| E (Expert+Zone) | 2 × 0.017 ms = 0.03 ms | 0 | ~99% (but XGMI limited) |

### 4.3 Estimated End-to-End Latency (8-GPU DSV3)

| Layout | GEMM (unchanged) | Exposed A2A | Overhead | **Total** | **Speedup vs Separated** |
|--------|:-----------------:|:-----------:|:--------:|:---------:|:------------------------:|
| Separated (no overlap) | 4.45 ms | 3.67 ms | 0.35 ms | 8.47 ms | 1.00× |
| A (Current, 8-stage) | 4.45 ms | ~3.0 ms | 2.4 ms (gather) | ~9.85 ms | 0.86× (slower!) |
| B (Expert-First) | 4.45 ms | ~0.3 ms | 0.35 ms | ~5.10 ms | 1.66× |
| C (Src×Expert, 8-stage) | 4.45 ms | ~1.1 ms | 0.35 ms | ~5.90 ms | 1.44× |
| **C (Src×Expert, 256-stage)** | **4.45 ms** | **~0.03 ms** | **0.35 ms** | **~4.83 ms** | **1.75×** |
| D (Tile-Aligned) | 4.45 ms | ~0.02 ms | 0.49 ms (meta) | ~4.96 ms | 1.71× |
| E (Expert+Zone) | 4.45 ms | ~0.03 ms | 0.35 ms + MFMA waste | ~5.5 ms | 1.54× |

Layout A with gather overhead is actually **worse** than no overlap,
because the gather work sits on the critical path and doesn't overlap
with anything.

---

## 5. Combine Phase Layout (Mirrored Analysis)

The combine (Phase 3) is the reverse of dispatch:

```
Dispatch:  source GPU → (XGMI write) → dest GPU dispatch_buf → expert GEMM
Combine:   expert GEMM output → (XGMI write) → source GPU combine_buf → weighted sum
```

### 5.1 Combine Data Flow

After expert GEMM, each GPU has results for tokens that originated from
all 8 source GPUs.  These results must be sent back to the originating
GPU and weighted-summed into the output.

| Combine sub-step | What happens |
|------------------|--------------|
| Pack results by source GPU | Reorder FC2 outputs by original token source |
| XGMI write | Write results to source GPU's combine buffer |
| Signal combine_ready | Notify source that results are available |
| Weighted scatter-add | Source GPU reads results, multiplies by routing weight, atomicAdd to output |

### 5.2 Layout Impact on Combine

The dispatch layout choice directly constrains the combine layout:

**Layout A (Source-GPU Chunked):**
Results are naturally grouped by source GPU (since dispatch chunks are
per-source).  Compute WGs process all experts for one source chunk,
then write results back to that source.  Combine write is one large
contiguous block → good XGMI efficiency.

**Layout C (Source × Expert):**
Within each source chunk, results are sorted by expert.  The combine
write-back can preserve this order (source GPU doesn't care about
expert ordering in the combine buffer).  Same XGMI efficiency as
dispatch.

For combine, the **scatter-add** pattern is the bottleneck regardless of
layout: `output[token_id] += weight * result[h]` requires either:
- Atomic adds (high contention when multiple experts write to the same token)
- Accumulation in a temporary buffer, then a final reduction pass

### 5.3 Combine Pipeline Overlap

Combine overlap works the same way as dispatch overlap, but in reverse:

```
Compute WGs: [GEMM chunk_0] [GEMM chunk_1] [GEMM chunk_2] ...
Tail Comm:     [wait]  [combine_0]  [combine_1]  [combine_2] ...
```

The tail comm WGs start combining chunk_0 results while compute WGs
process chunk_1.  Layout C's sub-chunk signaling enables per-expert
combine overlap, same as dispatch.

### 5.4 Scatter-Add Optimization

For the weighted scatter-add, the key optimization is to **pre-sort
combine results by destination token ID** on the expert GPU before XGMI
write-back.  This enables the source GPU to do a contiguous weighted
accumulation instead of random scatter-add:

```
Without sort: output[random_id] += w * result[i]   → random writes, atomics needed
With sort:    output[sorted_id] += w * result[i]    → mostly sequential, fewer conflicts
```

This pre-sort is natural with Layout C: within each source chunk, if
tokens are sorted by expert, and within each expert, by original token
ID, the combine results will have some locality (tokens from the same
source tend to have nearby IDs).

---

## 6. Recommendation

### Primary: Layout C (Source × Expert) with Sub-Chunk Signaling

Layout C with per-expert ready flags (`expert_ready[src][e]`, 256 flags)
provides the optimal balance:

| Factor | Assessment |
|--------|------------|
| Overlap quality | ~99% of A2A hideable (256 effective pipeline stages) |
| XGMI efficiency | ~95% (large contiguous writes, same as Layout A) |
| Gather overhead | Zero (tokens pre-sorted by expert) |
| MFMA utilization | 100% (full tiles, 128 tokens per expert) |
| Sender complexity | Medium (local sort by expert, no cross-GPU atomics) |
| Receiver complexity | Low (indexed access, no reordering) |
| Code change from current | Moderate (add expert sort in pack phase, add sub-chunk flags) |

**Estimated end-to-end: ~4.83 ms** (vs 8.47 ms separated = **1.75× speedup**).

### Dispatch Buffer Layout (Concrete)

```c
// Per GPU, IPC-mapped:
struct MoeIpcWorkspaceV2 {
    // Dispatch: [src_gpu][expert][token_slot][H]
    bf16_t*  dispatch_tokens;     // linearized as below
    int*     expert_offsets;      // [NUM_GPUS][E_local + 1] prefix sums
    int*     expert_src_ids;      // [NUM_GPUS][max_tokens] original token IDs
    int*     expert_ready;        // [NUM_GPUS][E_local] per-(src, expert) ready flag

    // Combine: same structure reversed
    bf16_t*  combine_results;
    int*     combine_offsets;
    float*   combine_weights;
    int*     combine_dst_ids;
    int*     combine_ready;       // [NUM_GPUS][E_local]
};

// Accessing tokens for expert e from source src:
//   base = src * max_recv_per_src * H
//   offset = expert_offsets[src][e]
//   count = expert_offsets[src][e+1] - expert_offsets[src][e]
//   tokens = &dispatch_tokens[(base + offset) * H]
```

### Implementation Roadmap

1. **Pack phase change:** Sort tokens by (dest_gpu, expert_id) during
   packing, compute per-expert prefix sums, write `expert_offsets[src][e]`.

2. **Signaling change:** Replace `chunk_ready[src]` with
   `expert_ready[src][e]`.  Sender signals after each expert sub-block
   is written (within the source chunk).

3. **Compute phase change:** Remove the per-expert scan + gather loop.
   Instead, index directly into `dispatch_tokens` using
   `expert_offsets[src][e]`.  Process expert_e from all available sources
   as soon as `expert_ready[*][e]` becomes 1.

4. **Combine phase change:** Mirror of dispatch.  Compute WGs write
   results sorted by (dest_token_id) within each source chunk.
   Per-expert combine ready flags.

### Why Not Layout D (Tile-Aligned)?

Layout D offers marginally better overlap (~99% vs ~99%) but at
significantly higher cost:
- Cross-GPU atomic contention for tile cursor
- ~137 KB metadata overhead per GPU
- Sub-optimal XGMI write size (896 KB < 2 MB sweet spot)
- Partial tile waste under non-uniform routing

Layout C achieves the same effective overlap with sub-chunk signaling,
while keeping XGMI writes large and avoiding cross-GPU atomics.

### Why Not Layout B (Expert-First)?

Layout B has the best receiver experience (pre-sorted, fine pipeline)
but the sender-side cross-GPU atomic contention and the "last sender
signals" protocol introduce correctness complexity and XGMI write
fragmentation that negate the receiver benefits.  Layout C achieves
the same receiver-side layout within each source chunk without any
cross-GPU coordination.
