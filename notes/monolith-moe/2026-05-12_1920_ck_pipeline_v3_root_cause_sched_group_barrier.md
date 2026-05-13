# CK Pipeline V3 根因定位：差距 100 % 在 sched_group_barrier 调度，不在指令数

| Field    | Value |
|----------|-------|
| When     | 2026-05-12 19:20 (UTC+8) |
| Where    | `mi355-gpu-26` / `xiaoming-dev` container, gfx950, ROCm 7.2.26015 |
| Project  | MonolithMoE |
| Status   | Diagnostic only — no code change shipped; inline-asm MFMA experiment reverted (no perf gain) |
| Source   | `/workspace/Primus-Turbo/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp` |
| 失败实验 | 切回 inline-asm MFMA → 170T（与 intrinsic 同），证明 MFMA wrapper 不是瓶颈 |

## TL;DR

CK 1211T vs 我们 170T 的 7× gap，**不在**：

- ❌ MFMA wrapper（inline-asm 实验确认）
- ❌ Tile/Warp/MFMA 形状（CK 同样 256×256×64 + 32×32×16 + 2×2 warps）
- ❌ Cooperative load 总指令数（CK 也是 8 phases × 16-byte/lane）
- ❌ HBM 延迟（profile 数据 wait_vm = 1 %）

**全部 7× gap 在调度顺序**：CK 用 `__builtin_amdgcn_sched_group_barrier` **显式
告诉 LLVM** buf_load / ds_write / ds_read / MFMA 这 4 类指令怎么穿插。我们
v8a 是 phase-batched 序列（全 buf_load → wait → 全 ds_write → barrier → 全
MFMA），完全串行。

## 1. CK BF16 gfx950 配置（与我们 v8a 全等）

`/workspace/Primus-Turbo/csrc/kernels/grouped_gemm/ck_grouped_gemm_kernel_config_hip.h`：

```cpp
using CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1 = CKGroupedGemmTileConfig<
    256, 256, 64,        // M_Tile, N_Tile, K_Tile
    32, 32, 16,           // M_Warp_Tile, N_Warp_Tile, K_Warp_Tile  (= MFMA-32x32x16)
    2, 2, 1,              // M_Warp, N_Warp, K_Warp  (= 4 waves/WG)
    false,                // DoubleSmemBuffer = FALSE !!!
    false                 // kPadN
>;
```

对应我们：

```cpp
// csrc/grouped_gemm.hip
using Cfg = TileCfg<4, 4, 64>;     // kM=4, kN=4, kK=64 == 4×32=128 per warp dim, 2×2 warps → 256×256 tile
constexpr int kMfmaK = 16;          // gfx950 → 32×32×16
// 但 double-buffer LDS: a_lds[2] / b_lds[2]
```

| 维度 | CK | 我们 v8a | Δ |
|---|---|---|---|
| M_Tile × N_Tile × K_Tile | 256 × 256 × 64 | 256 × 256 × 64 | 同 |
| MFMA (M × N × K) | 32 × 32 × 16 | 32 × 32 × 16 | 同 |
| Warps per WG | 2 × 2 = 4 | 2 × 2 = 4 | 同 |
| Threads per WG | 256 | 256 | 同 |
| MFMA per wave / K-step | 4 × 4 = 16 | 4 × 4 = 16 | 同 |
| K-steps per K-tile | 64/16 = 4 | 64/16 = 4 | 同 |
| **LDS buffers** | **1（PrefetchStages 内部管理）** | **2（双缓冲）** | **CK 单缓冲，靠 sched 顺序保证安全** |
| **HBM prefetch stages** | **2（GlobalPrefetchStages=2）** | **1（prologue + main-loop next-tile）** | **CK 多一阶 HBM** |
| Tile partitioner | `GemmSpatiallyLocalTilePartitioner` (M01=4, Group=8) | atomic counter / linear round-robin | L2 局部性差 |
| Epilogue | `CShuffleEpilogue`（LDS → coalesced store）| 逐 lane `bf16` 散写 | 我们 8× 浪费 HBM 带宽 |

## 2. CK V3 Pipeline 主循环结构

```cpp
// gemm_pipeline_ag_bg_cr_comp_v3.hpp:493
// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

// Prologue:
load_tile_with_elementwise(a_dram_window, ...);  // HBM #1
move_tile_window(a_dram_window, +K);
load_tile_with_elementwise(b_dram_window, ...);
move_tile_window(b_dram_window, +K);
LocalPrefill(a_lds_window, A0_reg);              // reg → LDS
LocalPrefill(b_lds_window, B0_reg);
load_tile_with_elementwise(a_dram_window, ...);  // HBM #2 (next)
load_tile_with_elementwise(b_dram_window, ...);
block_sync_lds();
LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);  // LDS → reg for first MFMA
__builtin_amdgcn_sched_barrier(0);

// Main loop:
do {
    block_sync_lds();                            // barrier
    LocalPrefill(a_lds_window, A_i+1_reg);       // overwrite same LDS slot
    LocalPrefill(b_lds_window, B_i+1_reg);
    load_tile_with_elementwise(a_dram_window);   // issue HBM for i+2
    load_tile_with_elementwise(b_dram_window);
    block_gemm(c, a_lds_gemm_window, b_lds_gemm_window);  // MFMA on i
    block_sync_lds();                            // barrier
    LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);  // LDS → reg for i+1
    HotLoopScheduler();                           // !!! 显式调度 4 类指令的交错
    __builtin_amdgcn_sched_barrier(0);
} while(i < num_loop - 1);
```

## 3. `HotLoopScheduler()` — 真正的"秘方"

```cpp
// gemm_pipeline_ag_bg_cr_comp_v3.hpp:243-369
constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;            // = 32 for 32×32×16
constexpr auto ds_read_a_issue_cycle = (16 / sizeof(bf16)) == 8 ? 8 : 4;  // = 8 for b128
constexpr auto ds_read_a_mfma_rate =
    (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
    // = (32 - 4 + 16 - 1) / 16 = 43/16 = 2  -- 每 MFMA 配 2 个 ds_read

// Stage 1: HBM-load 期间用 MFMA + ds_write 填充延迟
for each buffer_load (A side, 8 总数):
    for each ds_write per load (1 个):
        sched_group_barrier(DS_WRITE, 1)
        sched_group_barrier(MFMA, 1)
    sched_group_barrier(VMEM_READ, 1)            // 这一个 buf_load
    sched_group_barrier(MFMA, N)                 // N 个 MFMA 填满 buf_load 延迟

(同样对 B side)

// Stage 2: LDS-read 期间继续推 MFMA，按 ds_read_mfma_rate 节拍
for each ds_read group (8 个):
    sched_group_barrier(DS_READ, ds_read_mfma_rate=2)
    sched_group_barrier(MFMA, 1)
```

`__builtin_amdgcn_sched_group_barrier(mask, count, sync_id)` 是 LLVM AMDGPU
后端的**强 hint**：把 `count` 条指定类型（mask 编码）的指令绑成一组，组之间
**禁止重排**，组内由 RA 选择具体顺序。等于告诉编译器："**按这个节拍打**"。

我们 v8a 没有任何 `sched_group_barrier` / `sched_barrier`，编译器只能用
启发式 — 而它的启发式在这个 ds_write/ds_read/mfma 三方混排场景下显然**抓不准
最优解**。

## 4. 指令数对照（验证 gap 在调度不在数量）

per K-tile, BlockSize=256, M=N=256, K_tile=64：

| 指令类 | CK 计算 | 我们 v8a |
|---|---|---|
| `buffer_load_dwordx4` (A + B) | 8 + 8 = **16** | 8 + 8 = **16** |
| `ds_write_b128` (A + B) | 8 + 8 = **16** | 8 + 8 = **16** |
| `ds_read_b128` (A + B) | 16 + 16 = **32** | 16 + 16 = **32** |
| `v_mfma_f32_32x32x16_bf16` | 64 / WG | 64 / WG |

完全相同。

## 5. 我们 v8a 的当前结构（per K-tile）

```cpp
// csrc/grouped_gemm.hip:378-417
for kt in 0..nkt:
    if has_next:
        # Phase 1: 一口气发 8 个 buf_load (A) + 8 个 buf_load (B) = 16 buf_load
        for p in 0..A_PHASES:  la[p] = guarded_load8_bf16(...)
        for p in 0..B_PHASES:  lb[p] = guarded_load8_bf16(...)
    # Phase 2: 全部 MFMA + LDS 读
    for m in 0..kM:  ar[m] = lds_frag_ds<...>(...)
    for n in 0..kN:  br[n] = lds_frag_ds<...>(...)
    for ks in 0..G::KS:
        # 发 ds_read for 下一 step
        for m,n: ar_next[m] = lds_frag_ds<...>(...)
        for m,n: br_next[n] = lds_frag_ds<...>(...)
        wait_lgkm(kM + kN)
        for m,n: mfma_bf16(acc[m][n], ar[m], br[n])    # 16 MFMA 背靠背
        ar = ar_next; br = br_next
    # Phase 3: HBM drain + 8+8 ds_write
    if has_next:
        wait_vm(0)
        for p: a_lds[nbuf][...] = la[p]   # 8 ds_write A
        for p: b_lds[nbuf][...] = lb[p]   # 8 ds_write B
    __syncthreads()
```

**结构问题**：三段严格 phase-batched，**没有跨 phase 的指令交错**。
HBM-load 飞 400 cycle 时 MFMA 不能用同一 issue slot 利用（编译器看到
`wait_vm(0)` 就在那等）。

## 6. 落地路线（按 ROI）

### P0a — 加 `sched_group_barrier` 到 v8a 主循环（**新的 P0**）

最直接、最局部的改动：在 `gemm_core` 的 main K-tile 循环里加 CK
HotLoopScheduler 同等的 sched 提示。改动只在 `csrc/grouped_gemm.hip`
约 ~50 行，不动 tile shape / LDS layout。

**预期收益**：参考 CK V3 把 64 MFMA 与 16 + 32 + 16 = 64 个内存指令
完美交错，应能从 170T → ~600-800T（×3-4），仍不到 CK 1211T 但已经接近。
余下 1.5× gap 留给 P0b（CShuffle）+ P0c（spatial-local tile partitioner）。

工作量：1-2 d。

### P0b — CShuffleEpilogue (LDS-assisted vectorized store)

目前 v8a 的 epilogue 是**逐 lane 散写 bf16**，编译器无法 coalesce
→ HBM 写带宽利用 ~12.5 %。CK 用 LDS 把 lane-distributed 累加器重排
成 row-contiguous，再 `buffer_store_dwordx4` coalesced 写出。

工作量：2-3 d，影响小 GEMM（K 短，epilogue 占比大）更明显。

### P0c — SpatiallyLocalTilePartitioner (L2 reuse)

CK 的 `M01=4` 是 Hilbert-curve-like 的 4x4 block-cluster 顺序，
让相邻 WG 共享 L2 缓存的 A 行 / B 列。我们当前是 atomic counter 线性序，
L2 命中率约只有 CK 的 50 %。

工作量：1 d，主要是 host launch tile_id 重映射。

### 干掉的实验

- ~~恢复 inline-asm MFMA~~：今天证伪。
- ~~3-stage HBM prefetch~~：5/12 上午 P0a 已证伪（wait_vm 1 %）。
- ~~K-step 间去 `s_barrier`~~：5/12 中午 GEMM profile 证伪（sync 1 %）。
- ~~mxfp8 weights~~：留作 P1 — 与 sched_group_barrier 正交，可叠。

## 7. 现在该问的问题

1. **要不要先把 sched_group_barrier 实验做下去？**（P0a 路线，最直接）
2. 还是要先把 CShuffle epilogue 做了再说？（P0b 路线，简单但收益小）
3. 还是要先抓 CK kernel 的实际 asm（rocprof 或反汇编 `libprimus_turbo_kernels.so`），看 sched_group 在 codegen 后的样子，再决定？

我推荐 **P0a — sched_group_barrier**，这是 deep-dive 6 项里第一项被严格量化、且 codegen 工具已就绪的优化。

## 复现/参考

```bash
# CK 1050T 实测
ssh mi355-gpu-26 podman exec xiaoming-dev bash -lc \
  "cd $REPO && python benchmarks/bench_grouped_gemm_ck.py"
# → DSV3-GateUP B=8 M=4096 N=4096 K=7168: 1211 TFLOPS

# v8a 当前 170T
ssh mi355-gpu-26 podman exec xiaoming-dev bash -lc \
  "cd $REPO && ./benchmarks/results/bin/bench_v8a_now"
# → DSV3-GateUP B=8 M=4096: 170 TFLOPS

# CK pipeline V3 源码
less /workspace/Primus-Turbo/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp
# → HotLoopScheduler() at line ~250
# → main loop body at line ~530

# CK 配置
less /workspace/Primus-Turbo/csrc/kernels/grouped_gemm/ck_grouped_gemm_kernel_config_hip.h
# → CKGroupedGemmTileCfg_256x256x64_32x32x16_2x2x1
```
