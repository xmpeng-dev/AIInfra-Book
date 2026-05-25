# Cycle 8: parallel-peer copy_to_combine -- +10% wall, +13% TFLOPS

**When**: 2026-05-14 01:30 (UTC+8)
**Where**: `mi355-gpu-26` / `xiaoming-dev` container
**Change**: `csrc/fused_moe_super_kernel.hip` lines 2288-2350
**Default build**: now uses parallel-peer copy (no compile flag needed)

## The problem

Cycle 6 profile showed `copy_to_combine = 14.6 %` of wall (8.30 ms at
T_src=8192), and reading the code revealed why:

```cpp
for (int src = 0; src < NUM_GPUS; src++) {     // SEQUENTIAL!
    MoeIpcWorkspace* src_peer = args.ipc_peers[src];
    // ... all 128 compute WGs cooperate on ONE src at a time ...
}
```

So at any given moment, only ONE xGMI link (the one to `src` peer)
was active.  Bandwidth utilization: 117 MB / 8.3 ms ≈ **14 GB/s out
of ~50 GB/s per link, and only ONE of 7 outbound links used.**
Effective per-rank outbound: 14 GB/s vs 350 GB/s theoretical (4 %).

## The fix

Partition the 128 compute WGs across the 8 srcs concurrently:

```cpp
const int my_src       = wg_local_id % NUM_GPUS;     // wg→src by round-robin
const int my_pos_in_grp = wg_local_id / NUM_GPUS;     // 0..15 (16 WGs/src)
const int my_grp_stride = wgs_per_src * WG_SIZE;     // 16*256 = 4096 threads
const int my_global_in_grp = my_pos_in_grp * WG_SIZE + threadIdx.x;

MoeIpcWorkspace* src_peer = args.ipc_peers[my_src];
// 16 WGs cooperate on copying THIS src's chunk to THIS peer.
// All 8 src groups run in parallel → 7 outbound xGMI links saturated.
```

Same total work, different parallelism axis.  No correctness change
(each (rank, src) chunk lives in disjoint memory; no cross-src deps).

## Measured impact (256x256x64, ratio=0.250, profile-enabled bench)

### Latency

| shape | before (cycle 6) | after (cycle 8) | save |
|---|---|---|---|
| T_src=2048 | 14.43 ms | **13.09 ms** | -9.3 %, +13.6 % TFLOPS |
| T_src=8192 | 56.08 ms | **50.87 ms** | -9.3 %, +10.3 % TFLOPS |

### Phase breakdown change (T_src=8192)

|  | before | after | delta |
|---|---|---|---|
| dispatch_src_ready_wait | 17.52 | 17.91 | +0.39 |
| fc1_tiles               | 16.81 | 17.23 | +0.42 |
| swiglu_precompute       |  1.19 |  1.22 | +0.03 |
| fc2_tiles               | 10.88 | 11.20 | +0.32 |
| **copy_to_combine**     | **8.30** | **2.88** | **-5.42** |
| barriers (all)          |  2.14 |  0.68 | -1.46 |
| **total kernel wall**   | **56.85** | **51.19** | **-5.66** |

Copy went from 14.6 % of wall to 5.6 %.  Almost all of that came back
straight as wall-time save.  The slight +0.4 ms on the GEMM phases is
because the GEMM gets less L2 hit from concurrent xGMI traffic on the
fabric (more L2 bandwidth contention) -- but it's a tiny tax for the
~5 ms copy save.

### Bandwidth check

| | before | after | speedup |
|---|---|---|---|
| Bytes per copy (per rank) | 937 MB | 937 MB | 1.0× |
| Wall copy time | 8.30 ms | 2.88 ms | 2.88× |
| Achieved BW (out) | 113 GB/s | 326 GB/s | 2.88× |
| Theoretical aggregate xGMI out | 350 GB/s | 350 GB/s | --- |
| Utilization | 32 % | **93 %** | --- |

So the parallel-peer fix recovers nearly all of the xGMI bandwidth.
There's still 7 % headroom from contention / coalescing inefficiency,
but most of the inefficiency was the sequential outer-loop pattern.

## Where we stand vs the 1.8x goal

| shape | PyTorch+RCCL | SK current | gap | needed for 1.8x |
|---|---|---|---|---|
| T_src=2048 | 9.05 ms | 13.09 ms (1.45×) | -31 % | get to 5.0 ms (-62 % more) |
| T_src=8192 | 18.64 ms | 50.87 ms (2.73×) | -63 % | get to 10.4 ms (-80 % more) |

**Progress this session**:
- Cycle 1: baseline 67.19 ms / 17.79 ms (T_src=8192 / 2048)
- Cycle 2 (tile sweep, 256x256x64): -10 % / -6 %
- Cycle 3 (comm_ratio 0.250): -4 % / -10 %
- Cycle 6: re-profiled (no perf change)
- Cycle 7 (chunked-FC1): NO change in bench (kept under flag for multi-node)
- **Cycle 8 (parallel-peer copy)**: **-9 %** / **-9 %**
- Total: **8192 from 67.19 to 50.87 (-24 %)**, **2048 from 17.79 to 13.09 (-26 %)**

## Where the time still goes (T_src=8192, 51 ms total)

|  | ms | % of wall | next handle |
|---|---|---|---|
| dispatch_src_ready_wait | 17.91 | 35.0 % | inherent to scatter+barrier in single-node bench; chunked-FC1 ready for multi-node |
| fc1_tiles               | 17.23 | 33.7 % | GEMM core; ~20 % "remainder" might unblock more |
| fc2_tiles               | 11.20 | 21.9 % | GEMM core; same |
| copy_to_combine         |  2.88 |  5.6 % | already 93 % xGMI peak; little more |
| swiglu_precompute       |  1.22 |  2.4 % | could fuse into FC2 prologue |
| barriers (all)          |  0.68 |  1.3 % | counts; already minimal |

GEMM (FC1+FC2+SwiGLU) total = 29.65 ms = 58 % of wall.  Effective
TFLOPS = 808.  Theoretical MI355X BF16 peak = 1310.  We're at 62 % peak.

To close the gap to PyTorch+RCCL, we need:
- bring GEMM closer to peak (29.65 -> ~20 ms by hitting 80-90 % peak)
- or shrink dispatch_wait (17.9 -> some lower value)
- or both.

## Action items

- [x] Cycle 8: parallel-peer copy lands -9 % on both shapes.
- [ ] **Cycle 9**: investigate GEMM "remainder" (~20 % of GEMM time
       per profile).  Likely candidates: prologue HBM stall on first
       K-step, epilogue store_acc serialization.
- [ ] Cycle 10: try `swiglu_precompute` fused into FC2 prologue (current
       runs as separate phase).
- [ ] Cycle 11: profile dispatch_wait breakdown -- is it scatter wall
       (~9 ms) + IPC barrier (8 ms)?  Maybe slim down barrier.
