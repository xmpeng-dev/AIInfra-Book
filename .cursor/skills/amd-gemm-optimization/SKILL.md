---
name: amd-gemm-optimization
description: >-
  Develop high-performance GEMM kernels for AMD GPUs (MI300X/MI355X) using CUTLASS-style 
  hierarchical decomposition. Covers MFMA instructions, LDS optimization, tile sizing, 
  pipelining, and Triton/HIP C++ implementation. Use when optimizing matrix multiplication 
  for AMD CDNA architecture, porting CUDA kernels to ROCm, or writing custom GEMM kernels.
---

# AMD GEMM Optimization (CUTLASS-Style)

> **Before custom-writing or wrapping a GEMM path**, check the distilled
> library notes — `knowledge/libraries/composable-kernel.md` (CK is AMD's
> CUTLASS-equivalent and most "should I rebuild this?" decisions are
> already answered there), `knowledge/libraries/aiter.md` (production
> dispatcher used by vLLM/SGLang), and `knowledge/libraries/_patterns.md`
> (cross-library trade-off table). Then come back here for the kernel-
> craft details. Hardware-side micro-architecture details live in
> `.cursor/skills/mi355_hardware_aware/SKILL.md`.

## When to Apply

Use this skill when:
- Developing high-performance GEMM kernels for AMD MI300X/MI355X GPUs
- Porting NVIDIA CUTLASS kernels to AMD ROCm
- Optimizing matrix multiplication for CDNA architecture
- Writing Triton kernels targeting AMD MFMA instructions
- Implementing grouped/batched GEMM for MoE workloads

## AMD CDNA Architecture Overview

### MI355X Hardware Specs (gfx950)

| Resource | Value | Notes |
|----------|-------|-------|
| Compute Units (CUs) | 256 | Each CU has 4 SIMD units |
| Wavefront size | 64 threads | Equivalent to CUDA warp |
| LDS per workgroup | 160 KB max | Shared memory |
| HBM bandwidth | 8 TB/s peak | Theoretical |
| MFMA throughput | ~1300+ TFLOPS (BF16) | Matrix instructions |
| Register file | 256 VGPRs/thread | Vector registers |

### Key Differences from NVIDIA

| Concept | NVIDIA (CUDA) | AMD (ROCm) |
|---------|---------------|------------|
| Thread group | Warp (32 threads) | Wavefront (64 threads) |
| Tensor Core | mma.sync / wgmma | MFMA instructions |
| Shared memory | SMEM | LDS (Local Data Share) |
| Thread block | Block | Workgroup |
| Grid | Grid | NDRange |

## CUTLASS-Style Hierarchical Decomposition

### The GEMM Loop Nest (Adapted for AMD)

```
for (int wg_n = 0; wg_n < GemmN; wg_n += WgTileN) {     // Workgroup-level (CTA)
  for (int wg_m = 0; wg_m < GemmM; wg_m += WgTileM) {   
    
    for (int wg_k = 0; wg_k < GemmK; wg_k += WgTileK) { // Main loop (K-dimension)
      
      for (int wave_n ...) {                            // Wavefront-level
        for (int wave_m ...) {
          
          for (int mfma_k ...) {                        // MFMA instruction level
            for (int mfma_n ...) {
              for (int mfma_m ...) {
                
                mfma_instruction(d, a, b, c);           // AMD Matrix Fused Multiply-Add
                
              }
            }
          }
          
        }
      }
      
    }
  }
}
```

### Mapping to AMD Hardware

| CUTLASS Level | AMD Equivalent | Typical Size |
|---------------|----------------|--------------|
| Device GEMM | Kernel launch | Full problem |
| Threadblock tile | Workgroup tile | 128x128 to 256x256 |
| Warp tile | Wavefront tile | 32x32 to 64x64 |
| MMA instruction | MFMA | 16x16x16, 32x32x8, etc. |

## MFMA Instructions Reference

### Available MFMA Shapes (MI300X/MI355X)

| Instruction | M | N | K | Acc Type | Input Type |
|-------------|---|---|---|----------|------------|
| `mfma_f32_16x16x16_f16` | 16 | 16 | 16 | f32 | f16 |
| `mfma_f32_16x16x16_bf16` | 16 | 16 | 16 | f32 | bf16 |
| `mfma_f32_32x32x8_f16` | 32 | 32 | 8 | f32 | f16 |
| `mfma_f32_32x32x8_bf16` | 32 | 32 | 8 | f32 | bf16 |
| `mfma_f16_16x16x16_f16` | 16 | 16 | 16 | f16 | f16 |
| `mfma_f16_32x32x8_f16` | 32 | 32 | 8 | f16 | f16 |

### MFMA Thread-Data Layout

Each MFMA instruction involves all 64 threads in a wavefront:
- Threads are mapped to output elements in a specific pattern
- Input matrices A and B are distributed across threads
- Accumulator registers hold partial results

```python
# Triton MFMA usage pattern
@triton.jit
def mfma_gemm_kernel(...):
    # Load tiles to registers
    a = tl.load(a_ptr + offsets_a)
    b = tl.load(b_ptr + offsets_b)
    
    # Accumulate with MFMA
    acc = tl.dot(a, b, acc)  # Triton abstracts MFMA
```

## Tile Size Selection Guidelines

### Empirical Results from MI355X Benchmarks

| Tile Config | LDS Usage | Performance | Best For |
|-------------|-----------|-------------|----------|
| 128x128x32 | 36 KB | 247 TFLOPS | General GEMM |
| 128x128x64 | 68 KB | 186 TFLOPS | Small K |
| 64x128x32 | 27 KB | 199 TFLOPS | Low occupancy ok |
| 256x128x32 | 72 KB | ~300 TFLOPS* | Large M (CK-style) |

*With advanced optimizations (buffer_load, LDS banking)

### Selection Criteria

1. **LDS constraint**: Stay under 160KB, prefer <80KB for 2 WG/CU
2. **K_TILE**: Prefer 32 over 64 (better LDS savings → higher occupancy)
3. **M/N tiles**: 128x128 balances register pressure and MFMA utilization
4. **Accumulator count**: More accumulators per wave = better MFMA utilization

## LDS (Shared Memory) Optimization

### LDS Banking

AMD LDS has 32 banks with 4-byte stride. Conflict-free access patterns:

```python
# Good: Each thread accesses different bank
# Thread i accesses LDS[i * 4]  # 4-byte elements, sequential

# Bad: Bank conflict
# All threads access LDS[i * 32]  # Same bank!
```

### LDS Layout for MFMA

```python
# Swizzled layout to avoid bank conflicts
def lds_swizzle_offset(row, col, lds_stride):
    # XOR-based swizzling
    bank = col % 32
    swizzled_bank = bank ^ (row % 8)
    return row * lds_stride + swizzled_bank
```

### Double Buffering

```python
# Allocate 2x LDS for pipelining
LDS_A = [TILE_M, TILE_K, 2]  # 2 buffers
LDS_B = [TILE_N, TILE_K, 2]

# Main loop with double buffering
for k in range(0, K, TILE_K):
    buf_idx = k // TILE_K % 2
    next_buf = 1 - buf_idx
    
    # Async load next tile while computing current
    async_load(LDS_A[:, :, next_buf], global_A[..., k+TILE_K:])
    async_load(LDS_B[:, :, next_buf], global_B[..., k+TILE_K:])
    
    # Compute on current buffer
    compute_mfma(LDS_A[:, :, buf_idx], LDS_B[:, :, buf_idx], acc)
    
    barrier()  # Wait for async loads
```

## Global Memory Access Optimization

### Coalesced Access Pattern

```python
# Good: Coalesced load (threads access consecutive addresses)
# Thread i loads from ptr + i * sizeof(element)

# For row-major A[M, K]:
# Each wavefront loads a contiguous 64-element chunk

@triton.jit
def load_tile_coalesced(ptr, M, K, TILE_M, TILE_K):
    pid_m = tl.program_id(0)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_k = tl.arange(0, TILE_K)
    
    # Coalesced along K dimension
    a_ptrs = ptr + offs_m[:, None] * K + offs_k[None, :]
    return tl.load(a_ptrs)
```

### Buffer Load with Prefetch (HIP C++)

```cpp
// Explicit prefetch for better latency hiding
__device__ void prefetch_global(const void* ptr) {
    asm volatile("buffer_load_dwordx4 v[0:3], %0, 0 slc" :: "v"(ptr));
}

// In main loop:
prefetch_global(A_ptr + prefetch_distance);
// ... compute ...
actual_data = __builtin_amdgcn_global_load_dwordx4(A_ptr);
```

## Pipelining Strategy

### Software Pipelining Stages

```
Stage 0: Load A[k+0], B[k+0] → LDS buffer 0
Stage 1: Load A[k+1], B[k+1] → LDS buffer 1, Compute(buffer 0)
Stage 2: Load A[k+2], B[k+2] → LDS buffer 0, Compute(buffer 1)
...
```

### Triton Implementation

```python
@triton.jit
def pipelined_gemm_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr = 3,
):
    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Prologue: Fill pipeline
    for stage in range(NUM_STAGES - 1):
        a = tl.load(a_ptrs + stage * BLOCK_K)
        b = tl.load(b_ptrs + stage * BLOCK_K)
        # Store to LDS buffer[stage]
    
    # Main loop
    for k in range(0, K - (NUM_STAGES-1)*BLOCK_K, BLOCK_K):
        # Load next tile
        a_next = tl.load(a_ptrs + k + (NUM_STAGES-1)*BLOCK_K)
        b_next = tl.load(b_ptrs + k + (NUM_STAGES-1)*BLOCK_K)
        
        # Compute on current tile
        a_curr = # from LDS
        b_curr = # from LDS
        acc = tl.dot(a_curr, b_curr, acc)
        
        # Store next to LDS
        # ...
    
    # Epilogue: Drain pipeline
    for stage in range(NUM_STAGES - 1):
        acc = tl.dot(a_remaining[stage], b_remaining[stage], acc)
    
    # Store result
    tl.store(c_ptrs, acc)
```

## Wavefront Specialization (Advanced)

Similar to NVIDIA's warp specialization, dedicate wavefronts to different tasks:

```
Workgroup (256 threads = 4 wavefronts):
├── Wavefront 0-1: Producer (Global → LDS loads)
└── Wavefront 2-3: Consumer (LDS → MFMA compute)
```

### Implementation Pattern

```python
@triton.jit
def specialized_gemm(A, B, C, ...):
    wave_id = tl.program_id(0) % 4
    
    if wave_id < 2:
        # Producer wavefronts: Load data
        producer_loop(A, B, lds_a, lds_b, signals)
    else:
        # Consumer wavefronts: Compute
        consumer_loop(lds_a, lds_b, signals, acc)
```

## Performance Debugging Checklist

### 1. Occupancy Analysis

```bash
# Check occupancy with rocprof
rocprof --stats your_kernel

# Target: 2+ workgroups per CU
# LDS usage < 80KB allows 2 WG/CU
```

### 2. Memory Bandwidth

```python
# Calculate achieved bandwidth
bytes_read = (M * K + N * K) * sizeof(element)
bytes_written = M * N * sizeof(element)
total_bytes = bytes_read + bytes_written
achieved_bw = total_bytes / kernel_time_seconds

# MI355X peak: 8 TB/s
# Good utilization: >60% of peak
```

### 3. MFMA Utilization

```python
# Calculate TFLOPS
flops = 2 * M * N * K  # For GEMM
tflops = flops / kernel_time_seconds / 1e12

# Compare to peak (~1300 TFLOPS for BF16 on MI355X)
```

### 4. Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Low TFLOPS, high BW | Compute bound | Increase tile size |
| Low TFLOPS, low BW | Memory bound | Check coalescing, prefetch |
| Inconsistent perf | Bank conflicts | Add LDS swizzling |
| Kernel launch overhead | Small tiles | Use persistent kernels |

## Code Templates

### Triton GEMM Skeleton

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))
```

### HIP C++ MFMA Kernel Skeleton

```cpp
#include <hip/hip_runtime.h>

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void mfma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory
    __shared__ half smem_A[BLOCK_M][BLOCK_K];
    __shared__ half smem_B[BLOCK_K][BLOCK_N];
    
    // Accumulator in registers
    float acc[BLOCK_M/64][BLOCK_N/64][4];  // Per-thread portion
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < sizeof(acc)/sizeof(float); i++) {
        ((float*)acc)[i] = 0.0f;
    }
    
    // Main loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // Cooperative load to shared memory
        // ... (load A and B tiles) ...
        
        __syncthreads();
        
        // MFMA compute
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += 16) {
            // Use inline assembly for MFMA
            asm volatile(
                "v_mfma_f32_16x16x16_f16 %0, %1, %2, %0"
                : "+v"(acc[...])
                : "v"(a_frag), "v"(b_frag)
            );
        }
        
        __syncthreads();
    }
    
    // Store results
    // ...
}
```

## Reference Implementations

### Recommended Codebases

1. **Composable Kernel (CK)** - AMD's official high-performance library
   - Best-in-class GEMM performance
   - Complex but production-ready

2. **Triton (AMD fork)** - High-level kernel authoring
   - Easier development
   - Good for prototyping

3. **rocBLAS** - Vendor library
   - Baseline for comparison

### Performance Targets (MI355X)

| Workload | CK Perf | Triton Perf | Your Target |
|----------|---------|-------------|-------------|
| Large GEMM (4K×4K×4K) | ~1200 TFLOPS | ~900 TFLOPS | >800 TFLOPS |
| Grouped GEMM (MoE) | ~400 TFLOPS | ~250 TFLOPS | >300 TFLOPS |
| Small batched | ~600 TFLOPS | ~400 TFLOPS | >450 TFLOPS |

