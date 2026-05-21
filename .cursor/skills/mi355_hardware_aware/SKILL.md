---
name: mi355-hardware-aware
description: >-
  Hardware-aware programming guide for AMD Instinct MI355X (gfx950, CDNA4) GPUs,
  written for kernel / operator developers. Covers chiplet (XCD) topology and NPS
  partitioning, memory hierarchy (LDS / vL1$ / sL1$ / L2 / Infinity Cache / HBM3e),
  CDNA4 MFMA family (incl. f8f6f4 and MX block-scaled variants), VGPR / AGPR
  unified register file, async direct-to-LDS, wave specialization, roofline
  thresholds per precision, Triton-on-AMD knobs, and HIP / intrinsic / ISA
  cheatsheets. Use when writing or tuning a kernel on MI355X, porting a kernel
  from CDNA3 (MI300X / MI325X) or from NVIDIA H100 / B200, or debugging a
  performance regression on gfx950.
---

# MI355X Hardware-Aware Programming

## When to Apply

Use this skill when:
- Writing or tuning a performance-critical kernel that targets MI355X (gfx950).
- Porting a kernel from CDNA3 (MI300X / MI325X, gfx942), CDNA2 (MI250X), or
  NVIDIA H100 / B200 to MI355X — the MFMA shapes, LDS budget, async-copy
  semantics, and FP4 / FP6 / FP8 instruction set all differ.
- Debugging "compiles + numerically correct but slower than expected" — most
  often an occupancy, LDS-conflict, or async-pipeline issue.
- Setting a roofline target before deciding whether a kernel is compute-bound,
  memory-bound, or comm-bound.

> All numbers below are AMD vendor specs (see `knowledge/hardware/gpu-comparison.md`
> and the CDNA4 ISA reference). Per `.cursor/rules/10-gpu-kernels.mdc`, do not
> paste new numbers here without a citable source.

## 1. Chip Specifications (MI355X / gfx950)

| Spec | Value | Notes |
|---|---|---|
| Architecture | CDNA4 (gfx950) | TSMC 3 nm compute die + 6 nm I/O die |
| Compute Units (CUs) | **256** | 8 XCDs × 32 CUs/XCD |
| Accelerator Complex Dies (XCDs) | 8 | Each XCD is its own scheduling domain |
| SIMD Units per CU | 4 | Each SIMD = 16 lanes, 64 lanes/CU |
| Wavefront size | **64 threads** | Fixed; equivalent to "warp = 64" |
| Max wavefronts per SIMD | 8 | Limit on in-flight waves per SIMD |
| Max wavefronts per CU | 32 | 4 SIMDs × 8 waves |
| Max workgroups per CU | 16 | Hard ceiling (resource-limited in practice) |
| Peak engine clock | **~2.4 GHz** | MI355X spec; MI300X is ~2.1 GHz |
| HBM | **288 GB HBM3e**, 8 stacks | vs. MI300X 192 GB HBM3, MI325X 256 GB HBM3e |
| HBM bandwidth | **8 TB/s** peak | per-OAM, single GPU |
| TBP (board power) | **1400 W** | MI300X / MI325X were 750 W — cooling design differs |
| Form factor | OAM (UBB 2.0) | 8-GPU node default |
| Infinity Fabric scale-out | 7 links/GPU | up to 153 GB/s per link |

### What changed vs. MI300X (CDNA3, gfx942)

| Item | MI300X (gfx942) | MI355X (gfx950) | Kernel impact |
|---|---|---|---|
| CU count | 304 | 256 | Re-tune grid sizing; fewer but faster CUs |
| Peak clock | ~2.1 GHz | ~2.4 GHz | ~14 % more issue-rate per CU |
| LDS / CU | 64 KB | **160 KB** | Larger tiles, deeper pipelines fit on chip |
| HBM | 192 GB / 5.3 TB/s | 288 GB / 8 TB/s | More room for un-sharded weights |
| FP8 matrix dense | 2.6 PFLOPS | **5.0 PFLOPS** (OCP-FP8) | MFMA K-dim doubled |
| FP4 / MX support | none | **MXFP4 / MXFP6 10.1 PFLOPS** | New `*_f8f6f4` + `*_scale_*` instructions |
| BF16 matrix dense | ~1.3 PFLOPS | **~2.5 PFLOPS** | bf16 K-dim doubled to 32 |
| Async direct-to-LDS | partial | full path | `global_load_lds` is first-class |
| TBP | 750 W | 1400 W | Power throttling much more visible |

## 2. Chiplet (XCD) Architecture & NPS Modes

MI355X is a **chiplet GPU**: one OAM contains **8 Accelerator Complex Dies
(XCDs)** sitting on top of an I/O die that hosts the HBM controllers,
Infinity Fabric, and the **Infinity Cache (LLC)**. Every XCD has its own
**front-end command processor + per-XCD L2**, so kernel grids and memory
locality both have an XCD-level structure that flat "256 CU" mental models
miss.

```
                        ┌───────────────────────────┐
                        │   8 × HBM3E stacks        │
                        │   288 GB, 8 TB/s          │
                        └────────────┬──────────────┘
                                     │
        ┌────────────────────────────┴──────────────────────────────┐
        │                Infinity Cache (LLC, 256 MB)               │
        │      shared by all XCDs; serves L2 misses                 │
        └─┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘
          │      │      │      │      │      │      │      │
        ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
        │XCD│  │XCD│  │XCD│  │XCD│  │XCD│  │XCD│  │XCD│  │XCD│
        │ 0 │  │ 1 │  │ 2 │  │ 3 │  │ 4 │  │ 5 │  │ 6 │  │ 7 │
        │L2 │  │L2 │  │L2 │  │L2 │  │L2 │  │L2 │  │L2 │  │L2 │
        │32 │  │32 │  │32 │  │32 │  │32 │  │32 │  │32 │  │32 │
        │CU │  │CU │  │CU │  │CU │  │CU │  │CU │  │CU │  │CU │
        └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘
```

### NPS modes (memory partitioning)

The HBM controllers can be exposed as 1 / 2 / 4 NUMA partitions ("NPS1",
"NPS2", "NPS4"). The CU side can independently be exposed as a single
device (SPX) or partitioned (CPX). **NPS / CPX is set at boot**, so a kernel
author cares mostly about which mode the cluster admin picked.

| Mode | Visible HBM/GPU | When AMD ships it | Kernel impact |
|---|---|---|---|
| **NPS1 + SPX** (default for training) | 288 GB single pool | Most training clusters | One device, all 8 XCDs visible. Watch L2 / Infinity Cache locality across XCDs. |
| NPS4 + CPX (8-partition) | 36 GB × 8 | Inference / multi-tenant | Each XCD looks like a separate small GPU; intra-XCD locality is tighter. |

### XCD-locality rules of thumb

- A workgroup is launched onto a **single CU on a single XCD**; it cannot
  span XCDs. Cross-XCD synchronization is **L2-miss** territory.
- The Infinity Cache (LLC, 256 MB) is the only on-chip layer shared
  across XCDs. Hot tiles you want every XCD to see (for example a
  reused weight in EP=1) must fit in 256 MB.
- For grid tile-mapping, prefer **swizzles that keep a tile of `C[m,n]`
  on the same XCD across iterations** so its `acc` register column hits
  the same per-XCD L2 on epilog stores.

## 3. Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│             HBM3e (Global, 8 stacks)                    │
│             288 GB @ 8 TB/s peak                        │
└─────────────────────────────────────────────────────────┘
                       ↓  ~600+ cycle latency
┌─────────────────────────────────────────────────────────┐
│             Infinity Cache (LLC, on I/O die)            │
│             256 MB, shared across XCDs                  │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│             L2 Cache (per XCD)                          │
│             ~4 MB / XCD  (× 8 XCDs)                     │
│             128-byte cache line                         │
└─────────────────────────────────────────────────────────┘
                       ↓  ~100–200 cycle latency
┌──────────────────────────────┬──────────────────────────┐
│   Vector L1 (vL1$, per CU)   │   Scalar L1 (sL1$,       │
│   32 KB, 128-byte line       │   shared per CU pair,    │
│                              │   ~16 KB)                │
└──────────────────────────────┴──────────────────────────┘
                       ↓  ~20 cycle latency
┌─────────────────────────────────────────────────────────┐
│             LDS (Local Data Share, per CU)              │
│             160 KB, 32 banks × 4 B, ~peak 1 line/cycle  │
└─────────────────────────────────────────────────────────┘
                       ↓  ~5–10 cycle latency
┌─────────────────────────────────────────────────────────┐
│             VGPR + AGPR file (per SIMD, unified)        │
│             64 KB total / SIMD (= 512 × 32-bit / lane)  │
└─────────────────────────────────────────────────────────┘
```

### Per-CU resources

| Resource | Amount | Notes |
|---|---|---|
| LDS | **160 KB / CU** | Up from 64 KB on CDNA3 |
| Register file | **512 × 32-bit per lane** (VGPR + AGPR unified) | Hard ceiling per thread |
| Scalar GPRs (SGPRs) | up to 102 per wave | Compiler-allocated, rarely the binding constraint |
| Vector L1 (vL1$) | 32 KB | 128-byte line, write-through |
| Wavefront slots | 32 / CU (8 / SIMD) | Hardware ceiling |
| Workgroup slots | up to 16 / CU | Practical ceiling |

### Cache-line and transaction sizes (for coalescing)

| Layer | Granularity | What you should align to |
|---|---|---|
| HBM channel | 64 B burst | 64 B per pseudo-channel |
| L2 / Infinity Cache | **128 B line** | All vectorized loads should be 128-B aligned |
| vL1$ | **128 B line** | dwordx4 (16 B) per lane × 8 lanes = one line |
| LDS | 4 B/bank, 32 banks = **128 B/cycle** peak | One line per cycle if no conflicts |
| VGPR write-back from MFMA | 4 B per lane | Per-instruction |

## 4. Compute Unit (CU) Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Compute Unit (CU)                       │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │  SIMD 0  │ │  SIMD 1  │ │  SIMD 2  │ │  SIMD 3  │         │
│  │ 16 lanes │ │ 16 lanes │ │ 16 lanes │ │ 16 lanes │         │
│  │ VGPR/    │ │ VGPR/    │ │ VGPR/    │ │ VGPR/    │         │
│  │ AGPR     │ │ AGPR     │ │ AGPR     │ │ AGPR     │         │
│  │ 64 KB    │ │ 64 KB    │ │ 64 KB    │ │ 64 KB    │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │   Matrix Cores (MFMA pipes) — shared by 4 SIMDs         │ │
│  │   v_mfma_*, v_mfma_scale_* (CDNA4 MX block-scaled)      │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ Scalar Unit +    │  │   LDS — 160 KB, 32 banks × 4 B   │  │
│  │ SGPRs / sL1$     │  │   Direct Global→LDS path         │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  vL1$ (vector L1, 32 KB, 128-byte line, write-through)  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Wavefront execution model

A wavefront (64 threads) runs in **lockstep** under a per-wave `EXEC` mask.
Each SIMD has 16 lanes, and steady-state issue is **1 vector instruction
per cycle per SIMD**, with multiple resident waves round-robined to hide
latency.

```
SIMD 0 cycle 0: wave A, instr i              (issue, 64-wide)
SIMD 0 cycle 1: wave B, instr j              (independent wave)
SIMD 0 cycle 2: wave A, instr i+1            (latency-hidden by wave B)
...
```

Implications for kernel design:
- **More resident waves = more latency hiding.** Keep VGPR + LDS budgets
  low enough that ≥ 2 waves/SIMD (better: 4) are resident.
- **Divergent branches** serialize; both sides issue with masked-out
  lanes. Use predication for short branches; use `__shfl_*` /
  `__ballot` for wave-scope reductions.
- **Memory-issue overlap** is what hides HBM latency: if all in-flight
  waves are stalled on the same `s_waitcnt vmcnt(0)`, you've under-
  pipelined the loads.

## 5. Register File: VGPR & AGPR (CDNA3+)

CDNA3 unified the register file: each SIMD has a single **64 KB pool**
that is split between **VGPRs** (general vector regs, used by ALU /
load-store / `tl.dot` inputs) and **AGPRs** (accumulator regs, the
destination of `v_mfma_*`). MI355X (CDNA4) keeps this layout.

| Knob | Per-lane limit | Per-SIMD pool |
|---|---|---|
| VGPR + AGPR (combined) | 512 × 32-bit | 64 KB |
| VGPR only | up to 256 if AGPR=256 | — |
| AGPR only | up to 256 if VGPR=256 | — |
| Mixed (compiler-chosen) | sum ≤ 512 / lane | — |

Why kernel writers should care:
- `v_mfma_*` writes destination into AGPR. Moving the result to VGPR
  for epilog (scale, activation, store) costs `v_accvgpr_read_b32`
  instructions — fast but not free.
- The compiler may **promote AGPR → VGPR** when there's spare budget.
  Use `-Rpass-analysis=kernel-resource-usage` to confirm.
- A hand-rolled MFMA in inline asm or via `__builtin_amdgcn_mfma_*`
  must ensure accumulators land in AGPR (compiler will do this if you
  use the intrinsics correctly).

### Occupancy table (combined VGPR + AGPR per lane)

| (VGPR + AGPR) per lane | Max waves / SIMD | Comment |
|---|---|---|
| ≤ 64 | 8 | 100 %; rare for real GEMM |
| ≤ 96 | 5 | Acceptable for tight kernels |
| ≤ 128 | 4 | Common comfortable point |
| ≤ 168 | 3 | Latency hiding starts to suffer |
| ≤ 256 | 2 | Minimum to keep MFMA fed |
| ≤ 512 | 1 | Avoid; usually a spilled kernel |

### Spill detection

```bash
hipcc -Rpass-analysis=kernel-resource-usage -save-temps your_kernel.cpp
# Look for a line like:
#   "VGPRs: 168, AGPRs: 64, SGPRs: 80, ScratchSize: 0"
# ScratchSize > 0 → register spill to scratch (slow). Reduce tile size
# or split the inner loop.
```

## 6. MFMA Instructions (CDNA4 / gfx950)

The MFMA family is the only path to peak matrix throughput on AMD compute
GPUs — `tl.dot` in Triton, `rocwmma::mma_sync`, and CK's "tile op" all
ultimately lower to one of these. The set on **gfx950 is a strict
super-set of gfx942**, with three new families:

1. **Doubled-K legacy types** (bf16/fp16/i8): K dimension is doubled, so
   the same M×N tile produces 2× the FLOPs per instruction.
2. **Unified low-precision `*_f8f6f4`**: a single instruction whose A/B
   operands can be FP8 / FP6 / FP4 in any combination, encoded per-operand.
3. **Block-scaled `*_scale_*`**: for **MXFP4 / MXFP6 / MXFP8** (OCP MX
   format) — the instruction takes the per-block scale operands directly,
   so the kernel does not have to dequant before MFMA.

### Headline MFMA shapes on gfx950

| Instruction | M×N×K | A / B type | Acc | Per-instr ops (dense) | Notes |
|---|---|---|---|---|---|
| `v_mfma_f32_16x16x4_f32` | 16×16×4 | fp32 | fp32 | 2·16·16·4 = 2048 | Rare (FP32 GEMM is bandwidth-bound anyway) |
| `v_mfma_f32_32x32x4_f32` | 32×32×4 | fp32 | fp32 | 8192 | |
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | fp16 | fp32 | 8192 | gfx942 baseline |
| `v_mfma_f32_16x16x32_f16`† | 16×16×**32** | fp16 | fp32 | 16384 | **CDNA4 K-doubled** |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | fp16 | fp32 | 16384 | gfx942 baseline |
| `v_mfma_f32_32x32x16_f16`† | 32×32×**16** | fp16 | fp32 | 32768 | **CDNA4 K-doubled** |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | bf16 | fp32 | 8192 | gfx942 baseline |
| `v_mfma_f32_16x16x32_bf16`† | 16×16×**32** | bf16 | fp32 | 16384 | **CDNA4 K-doubled** |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | bf16 | fp32 | 16384 | gfx942 baseline |
| `v_mfma_f32_32x32x16_bf16`† | 32×32×**16** | bf16 | fp32 | 32768 | **CDNA4 K-doubled** |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | i8 | i32 | 16384 | INT8 GEMM |
| `v_mfma_i32_32x32x16_i8` | 32×32×16 | i8 | i32 | 32768 | INT8 GEMM |
| **`v_mfma_f32_16x16x32_f8f6f4`** | 16×16×**32** | fp8/fp6/fp4 | fp32 | 16384 | **CDNA4 unified low-precision** |
| **`v_mfma_f32_32x32x16_f8f6f4`** | 32×32×**16** | fp8/fp6/fp4 | fp32 | 32768 | **CDNA4 unified low-precision** |
| **`v_mfma_scale_f32_16x16x32_f8f6f4`** | 16×16×32 | MX-fp8/fp6/fp4 | fp32 | 16384 | **MX block-scaled, no SW dequant** |
| **`v_mfma_scale_f32_32x32x16_f8f6f4`** | 32×32×16 | MX-fp8/fp6/fp4 | fp32 | 32768 | **MX block-scaled** |
| `v_smfmac_*` (sparse variants) | various | bf16 / fp16 / fp8 | fp32 | 2× of dense | **2:4 structured sparsity, 2× throughput** |

† gfx950-only; **do not** assume these compile on gfx942.

### Operand encoding for `*_f8f6f4`

The unified instruction takes a 3-bit format selector for **A** and **B**
independently:

| Encoding | Format | Bits / element |
|---|---|---|
| 0 | OCP E4M3 (fp8) | 8 |
| 1 | OCP E5M2 (fp8) | 8 |
| 2 | reserved (FP6 E2M3) | 6 |
| 3 | reserved (FP6 E3M2) | 6 |
| 4 | OCP E2M1 (fp4) | 4 |

This means **mixed-precision GEMM** (e.g. FP8 weights × FP6 activations,
or FP4 weights × FP8 activations) is a single MFMA — no software upcast.
Block-scaled variants (`v_mfma_scale_*`) additionally consume two
**block-scale VGPR operands** that hold the per-32-element exponent
scale required by the OCP MX format.

### Thread → output mapping (16×16 case)

For `v_mfma_f32_16x16x32_f8f6f4` (and the matching bf16/fp16 16×16×K
shapes), the **64-lane wavefront tiles a 16×16 output** as follows:

```
       cols 0..3  cols 4..7  cols 8..11 cols 12..15
row 0  T0  T1     T4  T5     T8  T9     T12 T13
row 1  T2  T3     T6  T7     T10 T11    T14 T15
row 2  T16 T17    T20 T21    T24 T25    T28 T29
row 3  T18 T19    T22 T23    T26 T27    T30 T31
row 4  T32 ...                                    
...
row 15 T62 T63 ...
```

Each lane holds **4 fp32 accumulator elements** (= 4 AGPR / lane). Per
lane, the A operand contributes the matching K-strip, B the matching
K-row. The exact layout is what you must match when writing into LDS so
that `ds_read_b128` lands data in the right lanes.

For the 32×32 shapes each lane holds **16 fp32 accumulators**; you get
fewer instructions per tile but use more AGPR per wave.

### Choosing 16×16 vs 32×32

| Aspect | 16×16 tile | 32×32 tile |
|---|---|---|
| Acc per lane | 4 | 16 |
| AGPR pressure | low | 4× higher |
| MFMA latency (cycles) | ~16 | ~32 |
| K per instr (CDNA4 bf16) | 32 | 16 |
| Best for | small N (≤ 128) tiles, MoE per-expert | large dense GEMM, attention |
| Wave-tile count for 256×128 output | 8×4 = 32 | 4×2 = 8 |

Rule of thumb: prefer **32×32×16_bf16** for dense GEMM (FC1/FC2 of
non-MoE), prefer **16×16×32_f8f6f4** for grouped / per-expert GEMM where
N often ≤ 128 and you want fewer wave-tiles per workgroup.

### MFMA throughput sanity-check

```python
# Per-CU MFMA throughput (one matrix-pipe issue per cycle on the cluster
# of 4 SIMDs). At ~2.4 GHz the chip-wide peak is:

cus = 256
clock_ghz = 2.4

# bf16, K-doubled instruction (16x16x32_bf16): 16384 ops / 16 cycles
bf16_tflops_dense = cus * (16384 / 16) * clock_ghz / 1000
# ≈ 2520 TFLOPS dense   (matches AMD's ~2.5 PFLOPS bf16 spec)

# fp8 dense via 16x16x32_f8f6f4: 16384 ops / 8 cycles
fp8_tflops_dense = cus * (16384 / 8) * clock_ghz / 1000
# ≈ 5040 TFLOPS dense   (matches AMD's 5 PFLOPS OCP-FP8 spec)

# 2:4 structured sparse fp8: 2x dense
fp8_tflops_sparse = 2 * fp8_tflops_dense
# ≈ 10080 TFLOPS        (matches AMD's 10.1 PFLOPS sparse spec)
```

If your achieved bf16 GEMM is ~1.3 PFLOPS, you are getting **MI300X-class
performance on MI355X** — most likely you're not using the K-doubled
instruction. Check that the compiler emits `*_16x16x32_bf16` /
`*_32x32x16_bf16`, not the gfx942 K=16/8 variants.

## 7. Per-Precision Peak Throughput (MI355X chip-level)

| Precision | MFMA family used | Dense (PFLOPS) | 2:4 sparse (PFLOPS) | Notes |
|---|---|---|---|---|
| FP64 (matrix) | `v_mfma_f64_*` | ~0.08 | — | Mostly HPC; not the AI hot path |
| FP32 (matrix) | `v_mfma_f32_*x*x4_f32` | ~0.16 | — | |
| BF16 / FP16 | `*_16x16x32_bf16` / `*_32x32x16_bf16` | **~2.5** | ~5.0 | |
| FP8 (OCP E4M3 / E5M2) | `*_f8f6f4` | **5.0** | **10.1** | Block-scaled (MX) variant has the same peak |
| FP6 (OCP E2M3 / E3M2) | `*_f8f6f4` | ~5.0 | ~10.1 | Same instruction, narrower operand |
| FP4 (OCP E2M1) | `*_f8f6f4` | **~10.1** | — (already 2× density) | The headline MXFP4 number |
| INT8 | `v_mfma_i32_*_i8` | ~5.0 (TOPS) | ~10 (TOPS) | INT8 GEMM / quantised inference |

Vector (non-matrix) FMA peak is much lower — for any workload that can
reach matrix peak, do not fall back to scalar `v_fma_f32`.

## 8. LDS Programming

### Banking model

LDS has **32 banks × 4 B**, indexed by `bank = (addr / 4) % 32`. The
hardware can serve **one 128 B request / cycle** when there is no
conflict; conflicts serialize to multiple cycles.

| Pattern | Behaviour |
|---|---|
| All 64 lanes → 64 different banks (i.e. lane × 4 B stride) | 1 cycle (peak) |
| All 64 lanes → same address | 1 cycle (broadcast) |
| All 64 lanes → same bank, different addresses | 32-way conflict, 32 cycles |
| Even lanes bank 0, odd lanes bank 16 | 1 cycle (no conflict) |

### LDS layout patterns for MFMA tiles

Naïve row-major LDS for an MFMA-A tile (e.g. 32×32 bf16) collides on bank
0 for every column. Two standard fixes:

```cpp
// Pattern A: padding (cheap, 4 % LDS waste)
constexpr int K_TILE = 32;
constexpr int K_PAD  = 4;                  // 4 bf16 = 8 B padding
__shared__ bf16 tile_A[M_TILE][K_TILE + K_PAD];

// Pattern B: XOR swizzle (zero waste, harder to reason about)
__device__ int swizzle(int row, int col) {
    // 8-row group XOR; matches `ds_read_b128` 4 dwords/lane
    return row * (K_TILE) + (col ^ ((row >> 0) & 7));
}
```

Pattern B is used by CK and Triton-AMD when LDS budget is tight; pattern
A is fine for hand-rolled HIP kernels with budget headroom.

### LDS occupancy table (MI355X, 160 KB / CU)

| LDS / WG | Max WGs / CU | Implied min WG size for 4 waves/SIMD |
|---|---|---|
| ≤ 40 KB | 4 | 256-thread WG, 16 waves resident |
| ≤ 53 KB | 3 | |
| ≤ 80 KB | 2 | Common comfortable point |
| ≤ 160 KB | 1 | One WG per CU; only OK if WG is huge (≥ 8 waves) |

### Direct global → LDS path (gfx950)

CDNA4 has a first-class **direct-to-LDS load**: `global_load_lds_b32` /
`global_load_lds_b128` move data from HBM straight into LDS without
going through VGPR. This frees the VGPR file for accumulators and is
the foundation of all "async copy" pipelines on gfx950.

```cpp
// Triton lowers `tl.load(...)` to this when the destination is shared
// memory and the kernel uses async pipelining.
//
// In hand-written HIP, use the LLVM intrinsic:
__builtin_amdgcn_global_load_lds(
    global_ptr,          // i32* global source
    lds_ptr,             // __local i32* destination
    /*size=*/ 16,        // bytes; 4 / 8 / 16 supported
    /*offset=*/ 0,
    /*aux=*/ 0);

// Wait until all in-flight LDS writes are visible:
__builtin_amdgcn_s_waitcnt(0x0F70);   // vmcnt(0) & lgkmcnt(0)
__builtin_amdgcn_s_barrier();
```

> Async direct-to-LDS does **not** consume VGPR for the in-flight tile.
> This is what makes deep K-loop pipelines (3–4 stages) feasible on
> CDNA4 without spilling.

## 9. Memory Access Patterns

> Before reading this section's intrinsic-level mechanics, take 5 minutes
> on `knowledge/kernels/memory-access-patterns.md` — that file frames
> *which pattern to pick* (cross-row contiguity / LDS staging / wave
> lockstep vs independent / push vs pull) before you decide *how to
> implement it* with the intrinsics below.

### Global memory coalescing

CDNA L1 / L2 cache lines are **128 B**. Coalescing rule of thumb:

```cpp
// Good: 64 lanes × 4 B = 256 B → 2 cache lines, fully coalesced
float v = ptr[lane_id];

// Better: 64 lanes × 16 B = 1024 B = 8 lines / wave
float4 v = reinterpret_cast<const float4*>(ptr)[lane_id];

// Bad: 64 lanes × 64 B stride = 64 separate 128 B fetches
float v = ptr[lane_id * 16];
```

### Three load instruction families (and when to use each)

AMD GPUs have three flavours of vector load. They differ in addressing
mode, bounds checking, and cacheability:

| Family | LLVM intrinsic | Address mode | OOB | Cache hint | Typical use |
|---|---|---|---|---|---|
| `flat_load_*` | `__builtin_amdgcn_flat_load_*` | 64-bit pointer | trap | default | Generic pointer; accepts host / device alike |
| `global_load_*` | `__builtin_amdgcn_global_load_*` | 64-bit ptr + 13-bit imm offset | trap | default | The default for kernel work in HIP / Triton |
| `buffer_load_*` | `__builtin_amdgcn_raw_buffer_load_*` | **128-bit V# descriptor** + 12-bit offset | **no-trap, returns 0** | per-instance (sc0/sc1, glc, slc) | Hand-tuned kernels; OOB silent return is the *killer feature* for boundary tiles |

`buffer_load` with a V# (buffer descriptor) is what CK and most
production GEMM kernels use, because:
- Out-of-bounds lanes return 0 silently → no per-lane mask in inner loop.
- The descriptor encodes stride and num-records → masking is free.
- It bypasses vL1$ when `glc=1, slc=1` are set (uncached straight-to-L2),
  useful for one-shot streaming loads.

### Direct-to-LDS variants

Each load family also has `*_lds` variants that store into LDS:

| Intrinsic | What it does |
|---|---|
| `__builtin_amdgcn_global_load_lds` | HBM → LDS, dword granularity, no VGPR |
| `__builtin_amdgcn_raw_buffer_load_lds_b32` | Buffer-descriptor variant of the above |

These are the gfx950 pipelining primitives.

### Async pipeline waits

```cpp
// Wait counters used by CDNA:
//   vmcnt   : in-flight vector memory ops (HBM loads)
//   lgkmcnt : in-flight LDS / scalar / GDS ops
//   expcnt  : exports (rarely used in compute)

__builtin_amdgcn_s_waitcnt(0xCF7F);    // raw mask; readable form below
// vmcnt(0)   - all HBM loads have committed to LDS
// lgkmcnt(0) - all LDS reads/writes have committed
```

Tooling: `rocprof --hsa-trace` exposes the wait-counter stalls; Triton's
`triton.compiler.compile(..., debug=True)` dumps the AMD-GCN ISA so you
can see exactly where waits are inserted.

## 10. Synchronization Primitives

| Scope | HIP API | Notes |
|---|---|---|
| Workgroup (CTA) barrier | `__syncthreads()` (= `__builtin_amdgcn_s_barrier`) | One barrier ≈ 5–10 cycles + any pending wait-counts |
| Workgroup memory fence | `__threadfence_block()` | LDS + WG-local stores |
| Device fence | `__threadfence()` | Visible across the whole GPU (XCD-aware) |
| System fence | `__threadfence_system()` | Visible to other GPUs / CPU; flushes Infinity Cache |
| Wave reduce / scan | `__reduce_add(value)`, `__reduce_min`, etc. | DPP-based, no LDS needed |
| Wave shuffle | `__shfl(value, lane)`, `__shfl_xor(value, mask)` | Lower to `ds_bpermute_b32` |
| Wave ballot | `__ballot(cond)` → `unsigned long long` | 64-bit, full wave |

> **AMD vs NVIDIA naming:** ROCm wave-ops do **not** take a mask
> argument (the wave is always 64 wide). Don't paste NVIDIA's
> `__shfl_*_sync(0xFFFFFFFF, ...)` style — it's a CUDA-only API.

```cpp
// Triton equivalents
// tl.debug_barrier()  → __syncthreads()
// (Triton infers wave shuffles from tl.arange + index math)
```

## 11. Roofline Cheatsheet (per-precision)

For a kernel with arithmetic intensity `AI = FLOPs / Bytes` (read + write
to HBM), the boundary between memory-bound and compute-bound on MI355X
is **`AI* = peak_FLOPS / peak_BW`**:

| Precision | Peak (TFLOPS) | Peak HBM BW (TB/s) | AI* (FLOPs/Byte) | Implied K to be compute-bound at MN×MN GEMM (square) |
|---|---|---|---|---|
| BF16 | 2520 | 8.0 | **315** | K ≳ 315 (with FP16/BF16 inputs, fp32 acc) |
| FP8 (dense) | 5040 | 8.0 | **630** | K ≳ 630 |
| FP8 (2:4 sparse) | 10080 | 8.0 | 1260 | K ≳ 1260 |
| FP4 (MXFP4 dense) | 10080 | 8.0 | 1260 | K ≳ 1260 |

> "K to be compute-bound" means: for a square `M = N` GEMM in that
> precision, the K-dimension must exceed roughly `AI*` for a
> well-tuned kernel to be compute-bound rather than HBM-bound.
> Below that, HBM bandwidth is the ceiling and tiling more
> aggressively in K does not help.

### Implications for MoE / attention

- **Per-expert GEMM in MoE** with M (= tokens routed to expert) ≈ 32–512
  is **memory-bound at FP8** for any reasonable K, because AI* for FP8
  is ~630. Optimization focus: avoid extra HBM round-trips, fuse
  prologue / epilogue.
- **Attention kernels** at long context are memory-bound on softmax
  (`Q@K^T` + softmax + `S@V`); FlashAttention-style fusion is the
  point because it removes the softmax round-trip.
- **Dense GEMM in transformer FC1/FC2** with M ≈ 4096+ comfortably
  reaches compute-bound regime even at FP8.

## 12. Triton-on-AMD: knobs you actually have to set

Triton's AMD backend exposes several knobs that map to ISA-level
choices. The autotuner does not always pick well, especially for
grouped GEMM and attention.

### Key autotune dims

| Knob | Typical search space | What it controls |
|---|---|---|
| `BLOCK_M` | 64 / 128 / 256 | Output M-tile (workgroup-level) |
| `BLOCK_N` | 64 / 128 / 256 | Output N-tile |
| `BLOCK_K` | 32 / 64 / 128 | K-step; bf16 doubled instr likes 64 |
| `num_warps` | 4 / 8 (= 256 / 512 thr / WG) | "Warp" = wavefront on AMD; 4 means 4 waves/WG |
| `num_stages` | 1 / 2 / 3 / 4 | Software-pipeline depth — needs LDS budget × num_stages |
| `waves_per_eu` | 1 / 2 (kernel-attr) | Compiler hint to keep VGPR low for higher occupancy |
| `matrix_instr_nonkdim` | 16 / 32 | Forces `*_16x16x*` vs `*_32x32x*` MFMA shape |
| `kpack` | 1 / 2 | Pack two K-vectors per LDS load (gfx950 friendly) |

### Sample autotune block

```python
import triton
from triton import Config

def get_amd_configs():
    return [
        Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,
                "matrix_instr_nonkdim": 16, "kpack": 2, "waves_per_eu": 2},
               num_stages=3, num_warps=4),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,
                "matrix_instr_nonkdim": 32, "kpack": 1, "waves_per_eu": 1},
               num_stages=2, num_warps=8),
        # ... 6–10 configs is usually plenty
    ]

@triton.autotune(configs=get_amd_configs(), key=["M", "N", "K"])
@triton.jit
def gemm_kernel(...):
    ...
```

### `tl.dot` precision tags on AMD

```python
# Use float8e4nv (= OCP E4M3) for MI355X. Do NOT use float8e4b8
# (= AMD FNUZ); it is gfx942-only and silently upcasts on gfx950.
acc = tl.dot(
    a.to(tl.float8e4nv),
    b.to(tl.float8e4nv),
    acc,
    out_dtype=tl.float32,   # required; fp32 acc is the MFMA shape
)
```

### Inspecting what Triton lowered to

```bash
# Dump GCN ISA for a Triton kernel
TRITON_AMDGCN_OUT=1 TRITON_DUMP=1 python my_kernel.py

# What to look for in the .amdgcn:
#   v_mfma_f32_16x16x32_bf16          <-- good, K-doubled CDNA4
#   v_mfma_f32_16x16x16_bf16          <-- BAD, falls back to gfx942 shape
#   global_load_lds_b128              <-- direct-to-LDS, async pipeline
#   ds_read_b128                      <-- LDS reads, expect 1/cycle/SIMD
#   s_waitcnt vmcnt(0)                <-- HBM wait barriers
```

## 13. HIP / Intrinsic / ISA Cheatsheet

| Concept | HIP / Triton API | LLVM intrinsic | gfx950 ISA |
|---|---|---|---|
| Vector load (4 dwords) | `*reinterpret_cast<float4*>(ptr)` | `__builtin_amdgcn_global_load_dwordx4` | `global_load_dwordx4` |
| Buffer load (no-trap) | (CK, hand-asm) | `__builtin_amdgcn_raw_buffer_load_b128` | `buffer_load_dwordx4` |
| Direct global → LDS | `tl.load → shared` (Triton) | `__builtin_amdgcn_global_load_lds` | `global_load_lds_b32` |
| LDS read | `__shared__` access, `tl.load(LDS_ptr)` | `__builtin_amdgcn_ds_read_b128` | `ds_read_b128` |
| LDS write | `__shared__` write | `__builtin_amdgcn_ds_write_b128` | `ds_write_b128` |
| MFMA bf16 32×32×16 | `tl.dot(...)` | `__builtin_amdgcn_mfma_f32_32x32x16_bf16` | `v_mfma_f32_32x32x16_bf16` |
| MFMA f8f6f4 16×16×32 | `tl.dot(fp8, fp8, ...)` | `__builtin_amdgcn_mfma_f32_16x16x32_f8f6f4` | `v_mfma_f32_16x16x32_f8f6f4` |
| MFMA scale (MXFP4) | (compiler-emitted) | `__builtin_amdgcn_mfma_scale_f32_16x16x32_f8f6f4` | `v_mfma_scale_f32_16x16x32_f8f6f4` |
| Sparse MFMA bf16 | (compiler-emitted) | `__builtin_amdgcn_smfmac_f32_16x16x32_bf16` | `v_smfmac_f32_16x16x32_bf16` |
| Wavefront barrier | `__syncthreads()` | `__builtin_amdgcn_s_barrier` | `s_barrier` |
| Memory wait | (compiler-emitted) | `__builtin_amdgcn_s_waitcnt(mask)` | `s_waitcnt vmcnt(...) lgkmcnt(...)` |
| Wave shuffle | `__shfl_xor` / `__shfl` | `__builtin_amdgcn_ds_bpermute` | `ds_bpermute_b32` |
| Wave ballot | `__ballot` | `__builtin_amdgcn_ballot` | `v_cmp_*` + `s_or_saveexec` |
| AGPR ↔ VGPR move | (compiler-emitted) | `__builtin_amdgcn_v_accvgpr_*` | `v_accvgpr_read/write_b32` |

### Reading the wait-counter mask

```cpp
// Encoded mask used by s_waitcnt:
//   bits  3:0   → vmcnt (vector memory ops)         range 0..63
//   bits  6:4   → expcnt (export, ignored on compute)
//   bits 11:8   → lgkmcnt (LDS / scalar / GDS)      range 0..15
__builtin_amdgcn_s_waitcnt(0x0F70);   // vmcnt(0) & lgkmcnt(0) — drain everything
```

Most kernel bugs at the ISA layer are mis-tuned wait masks: e.g. an
inner loop that issues `global_load_lds` then immediately barriers
without `vmcnt(0)` will see torn LDS reads.

## 14. XGMI / Infinity Fabric (Multi-GPU)

MI355X exposes **7 Infinity Fabric scale-out links per GPU** at up to
**153 GB/s per link** unidirectional (per AMD MI355X spec page). In the
8-GPU OAM platform the topology is **all-to-all**: every GPU has a
direct link to every other GPU on the node.

| Property | Value |
|---|---|
| Links per GPU | 7 |
| Bandwidth per link | 153 GB/s (unidir) |
| Aggregate per-GPU scale-out BW | ~1071 GB/s |
| Topology | 8-GPU all-to-all |
| Latency | lower than PCIe, similar to MI300X |

### Programming considerations

```python
# Check XGMI topology (HIP / PyTorch)
import torch
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            can_access = torch.cuda.can_device_access_peer(i, j)
            print(f"GPU {i} → GPU {j}: {'XGMI' if can_access else 'PCIe'}")
```

### IPC handles for multi-process intra-node

```cpp
// Producer process:
hipIpcMemHandle_t handle;
hipIpcGetMemHandle(&handle, device_ptr);
// Send `handle` to consumer over a Unix socket / shared file.

// Consumer process:
void* remote_ptr;
hipIpcOpenMemHandle(&remote_ptr, handle, hipIpcMemLazyEnablePeerAccess);
// remote_ptr now reads via XGMI (no host bounce buffer).
```

For overlap of XGMI traffic with in-kernel MFMA, see
`.cursor/skills/cco-pipeline-overlap/SKILL.md`.

## 15. Performance Tuning Checklist

### 15.1 Occupancy

```bash
# Per-kernel resource usage at compile time
hipcc -Rpass-analysis=kernel-resource-usage -save-temps your_kernel.cpp

# At runtime
rocprof --stats ./your_app
# Inspect ratio of resident waves to peak (32 / CU).
```

Targets:
- **≥ 2 waves / SIMD** (= 8 / CU) — minimum to hide single-instr latency.
- **≥ 4 waves / SIMD** (= 16 / CU) — comfortable for memory-heavy kernels.

### 15.2 Memory bandwidth

```python
bytes_moved = input_bytes + output_bytes  # round-trip to HBM
achieved_bw_tbps = bytes_moved / kernel_time / 1e12
efficiency = achieved_bw_tbps / 8.0 * 100   # % of MI355X peak
# Target: > 60 % for memory-bound kernels; > 80 % for pure copy
```

### 15.3 Compute utilisation

Use the precision-aware peaks from §7, **not** a fixed 1350 TFLOPS:

```python
peak_tflops_by_dtype = {
    "bf16": 2520, "fp16": 2520,
    "fp8":  5040,                  # OCP E4M3, dense
    "fp4": 10080,                  # MXFP4, dense
}
flops = 2 * M * N * K              # GEMM
achieved_tflops = flops / kernel_time / 1e12
efficiency = achieved_tflops / peak_tflops_by_dtype[dtype] * 100
# Target: > 70 % for compute-bound dense GEMM
```

### 15.4 Common bottlenecks

| Symptom | Likely root cause | First thing to try |
|---|---|---|
| Low occupancy (`< 2 waves / SIMD`) | VGPR + AGPR > 256/lane, or LDS > 80 KB / WG | Shrink M/N tile, or split inner loop |
| HBM efficiency < 50 % | Uncoalesced load, or no async direct-to-LDS | Switch to `global_load_lds_b128`, vectorize to 16 B/lane |
| MFMA util < 50 %, BW low too | Compiler emits gfx942-shape MFMA | Set `matrix_instr_nonkdim` (Triton) or use the K-doubled intrinsic |
| MFMA util < 50 %, BW high | LDS bank conflicts on read | Add padding or XOR swizzle; verify with `LDSBankConflict` counter |
| Runtime cliff at certain tiles | VGPR spill to scratch | Check `ScratchSize > 0` in compile output |
| Kernel launch overhead | Too many small launches | Persistent kernel with `gridDim = num_CUs` |
| Power-throttled clocks | Sustained ~1.4 kW; thermal | `rocm-smi -c` to see current clock; kernel design rarely fixes this |

### 15.5 Profiling tools

```bash
# rocprof v1: counters + trace
rocprof --stats ./your_app
rocprof -i metrics.txt --hip-trace --hsa-trace ./your_app

# Omniperf: aggregated roofline + per-section attribution
omniperf profile -n my_profile -- ./your_app
omniperf analyze -p my_profile/

# Key counters to put in metrics.txt
#   SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_MFMA
#   TCP_TCC_READ_REQ_sum  (vL1 → L2 traffic)
#   TCC_HIT_sum / TCC_MISS_sum  (L2 hit rate)
#   GRBM_GUI_ACTIVE, MFMAUtilization, VALUUtilization
#   LDSBankConflict, LDS_INDEX_HIT
```

| Counter | Healthy value | Meaning |
|---|---|---|
| `VALUUtilization` | > 80 % | Vector ALU busy |
| `MFMAUtilization` | > 70 % | Matrix pipe busy |
| `LDSBankConflict` | < 5 % | LDS access efficiency |
| `TCC_HIT_sum / total` | > 90 % | L2 hit rate (good data reuse) |
| `MemUnitBusy` | < 80 % | HBM not the bottleneck |
| `SALUUtilization` | typically low | Scalar unit; rarely a hotspot |

## 16. Common Pitfalls (CDNA3 → CDNA4 migration)

| Pitfall | Why it happens | Fix |
|---|---|---|
| BF16 GEMM stuck at 1.3 PFLOPS on MI355X | Compiler emits `*_16x16x16_bf16` (CDNA3 shape) | Force `matrix_instr_nonkdim` in Triton, or upgrade ROCm so the AMD backend selects K=32 by default |
| FP8 silently upcast to FP16 | Used `tl.float8e4b8` (FNUZ) instead of `tl.float8e4nv` (OCP E4M3) | See `knowledge/kernels/fp8-expert-gemm.md` §5.2 |
| MXFP4 kernel fails to compile | Toolchain too old; `*_scale_*` intrinsics need ROCm ≥ 7.0 | Verify ROCm version, then use `__builtin_amdgcn_mfma_scale_*` |
| Kernel slower at the same tile size as on MI300X | LDS budget is now 160 KB but you're still using 64 KB / WG with 1 stage | Add a stage to the pipeline; `num_stages=3` is now realistic |
| 2:4 sparse path doesn't speed up | Mask is per-32-element block, not per-element | Verify mask layout matches `v_smfmac_*` operand encoding |
| Random hangs on 8-GPU all-to-all | Mixing peer-IPC pointers with RCCL on the same stream | Serialize into one channel or split streams; see `cco-pipeline-overlap` skill |

## 17. References

- AMD CDNA4 ISA Reference (gfx950) — vendor PDF, the only authoritative
  source for instruction encodings, latencies, and operand layouts.
- AMD MI355X product page —
  <https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html>
- AMD MI355X platform page (8-GPU OAM) —
  <https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html>
- ROCm Composable Kernel (CK) — production-grade reference for MFMA
  pipelines on CDNA: `<rocm-libraries>/composable_kernel/`.
- Triton AMD backend docs — knobs documented above
  (`matrix_instr_nonkdim`, `kpack`, `waves_per_eu`).
- Sibling slab docs:
  - `knowledge/hardware/gpu-comparison.md` — MI355X / B200 / H200 specs.
  - `knowledge/kernels/fp8-expert-gemm.md` — concrete FP8 grouped-GEMM
    case study; FP8 dtype gotchas.
  - `.cursor/skills/amd-gemm-optimization/SKILL.md` — CUTLASS-style
    GEMM decomposition recipe.
  - `.cursor/skills/cco-pipeline-overlap/SKILL.md` — comm-compute
    super-kernel patterns (XGMI + MFMA in one launch).
