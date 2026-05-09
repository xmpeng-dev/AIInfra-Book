# CK Grouped GEMM 实现深度解析（vs MonolithMoE 1050T → 530T 差距）

**日期**: 2026-05-08
**作者上下文**: 在分析 `MMOE/README.md` 可行性时发现 — MonolithMoE 想达到的"≥600 端到端 TFLOPS / 4.8ms DSV3"完全依赖 GEMM 阶段追上 CK 的 ~1050T；当前 `csrc/grouped_gemm.hip` 只有 ~530T，2× gap 必须在 super-kernel 之外先解决。本文系统拆解 CK（Composable Kernel + ck_tile DSL）在 MI355X 上跑出 1050T 的实现思路，并把每条优化对应回 MMOE 的 v8a 实现。

## 背景 / 目标

### 当前差距（DSV3 grouped GEMM, gfx950, ROCm 7.2）

| 实现 | GateUP avg | Down avg | 相对 CK |
|---|---|---|---|
| **primus_turbo (CK)** | **~1050 TFLOPS** | **~960 TFLOPS** | 1.00× |
| `csrc/grouped_gemm.hip` v8a (256×256×64) | ~530T | ~520T | **0.50–0.54×** |
| MMOE in-kernel grouped GEMM (super-kernel 内嵌) | 530T 上限 | 520T 上限 | 0.50× |
| 3rd/gg | ~233–442T | ~246–442T | 0.22–0.46× |

`bench_grouped_gemm_ck.py` 在 `B=8, M=4096, N=4096, K=7168` 跑出 `1057T`，硬件理论峰值 ~1300T（BF16 MFMA）→ CK 已经到 **81% 峰值**，我们到 **40% 峰值**。

### 为什么这个 gap 是阻塞性的

README 第 313–324 行的目标是 `8 GPU DSV3 4096tok 端到端 4.797 ms`（≥600 TFLOPS effective）。
- A2A 总量 3.67 ms，要被 hide 掉
- GEMM 总量 4.45 ms，必须填满整个 hide 窗口
- 但当前 super-kernel 的 grouped GEMM 是 530T —— 在 DSV3 形状下 GEMM 自己就要 **~9 ms**，比 RCCL+PyTorch 还慢，"完全 hide A2A" 救不回来

**结论**：在动 super-kernel overlap 之前，必须先把 GEMM 路径推到 CK 量级（≥900T）。否则 fixed-cost barrier 干掉之后下一个瓶颈立刻变成 GEMM。

---

## 主要发现 / 结论

CK 跑赢 v8a 2× 的原因不是"某个特别牛的 trick"，而是 **6 项中等强度优化叠乘**：

| # | 优化点 | CK 做了 | v8a 状态 | 估计单项贡献 |
|---|---|---|---|---|
| 1 | **Tile 形状 + L2 调度** | Stream-K / Split-K + persistent grid，自动选 256×256 / 256×128 等 | 静态 256×256 round-robin | +15–25% |
| 2 | **WGAttrCtl = Raw_avv** | C 在 AGPR，A/B 在 VGPR，零 AGPR↔VGPR 搬运 | 已是 AGPR-C（同向） | 同 |
| 3 | **多级软件流水（NUM_STAGES=3）** | 3 级 prefetch，prologue 填 2 级，主循环每次 issue +1 | 1 级（prologue 1 tile） | +10–15% |
| 4 | **精确 s_waitcnt 阶梯** | `lgkmcnt(N-1) → lgkmcnt(N-2) → ...` 让每个 MFMA 只等最少需要的 ds_read | 已经有阶梯，但不够紧（编译器插入了多余的 lgkmcnt(0)）| +5–10% |
| 5 | **LDS swizzle + 操作数复用** | XOR-swizzle 完全消除 bank conflict；A 复用次数 = NIterPerWarp（沿 N 内层）| PAD=4 是 padding 法（多用 LDS），A 复用部分 | +5–10% |
| 6 | **C-shuffle epilogue（LDS-assisted vectorized store）** | 用 LDS 把 lane-distributed accumulator 重排成 row-contiguous，再 `buffer_store_dwordx4` | 逐元素 `bf16 store`（编译器无法合并） | +5–10% |

**乘起来 ≈ 1.5–2.0×**，正好对得上 1050/530 = 1.98。

---

## 详细分析

### 1. CK 的多层抽象（ck_tile DSL）

CK 对外是 C++ 模板库，但内部结构是一个 **Tile-based 的 DSL**，按硬件层次分四级：

```
Device GEMM (kernel launch)
  └─ Block-level Tile (workgroup 处理一块 [BlockM, BlockN])
       └─ Block GEMM Pipeline (主循环 + 流水线)
            └─ Warp GEMM (每个 wave 处理一块 [WaveM, WaveN])
                 └─ MFMA Instruction Wrapper (单条 v_mfma_*)
```

每一级都是一个独立的、可替换的 C++ 模板组件：

| 层级 | 关键组件 | 责任 |
|---|---|---|
| Device | `gemm_kernel<...>` | grid mapping, tile_id → (m_block, n_block, k_block) |
| Block | `BlockGemmPipelineAGmemBGmemCRegV1` | 主循环（K 维），prefetch 调度，LDS double/triple buffer |
| Warp | `WarpGemmMfmaBf16Bf16F32M32N32K16` (gfx950) / `K8` (gfx942) | wave→MFMA mapping，operand reuse |
| MFMA | inline asm `v_mfma_f32_*` | 单条指令 + AGPR 约束 |

**关键设计点**：四级**全部可以编译期 swap**。CK 通过模板特化为不同 (M, N, K) 形状选不同的 BlockGemmPipeline，比如：

| 形状特征 | CK 选择 | 理由 |
|---|---|---|
| K 大、M/N 中（DSV3 GateUP: 4096×7168）| Stream-K + 3-stage prefetch | K 维有足够 unrolled work 喂满 prefetch |
| K 小、M/N 大（FC2: 7168×2048）| Split-K + 2-stage | K 短，第三级 prefetch 没收益反而吃寄存器 |
| Variable batch（grouped GEMM）| Persistent kernel + atomic tile counter | kernel launch 和 expert 切换 overhead 摊到一次 launch |

**MMOE v8a 的对应**：
- 现在是**单一模板 256×256×64 + AGPR + buffer_load**，不区分 GateUP 和 FC2 的形状差异
- Persistent kernel 已经在 `fused_moe_super_kernel.hip` 里，但 **work distribution 是静态 round-robin**（README Phase 2 第 420 行 TODO："工作窃取（work-stealing）的 tile 计数器"）
- 没有 Stream-K，K 维分块策略硬编码

### 2. Stream-K / Split-K 调度（最被低估的一项）

#### 经典 Data-parallel 的问题

```
GemmM=128, GemmN=2048, BlockM=128, BlockN=128
→ M_blocks=1, N_blocks=16, total_tiles=16

256 个 CU，但只有 16 个 tile → 240 个 CU 闲置！
```

DSV3 expert 形状 `T_e≈16, N=2F=4096, K=H=7168` 在 256×256 tile 下：
- M_blocks = ceil(16/256) = 1
- N_blocks = 4096/256 = 16
- 只有 16 个 tile / expert × 32 expert = 512 tile，分给 256 CU = 2 tile/CU
- 每个 CU 单独跑一个 K=7168 的循环 → **K 维没有跨 CU 并行**

#### Stream-K 的解法

把 (M_block, N_block, K_block) 的总计算切片，按 **CU 数量** 而不是按 tile 数量分配：

```
total_iterations = M_blocks * N_blocks * K_blocks
work_per_cu = total_iterations / num_CUs

CU_i 负责:
  iter_start = i * work_per_cu
  iter_end   = (i+1) * work_per_cu
  
跨 (m, n) tile 的 partial accumulators 在 L2 上做 fixup（一次 atomic add）
```

**收益**：
- 即使 N_blocks 只有 16，也能让 256 CU 全部工作（每个 CU 做 K 的一段）
- L2 cache 命中率显著提升（相邻 CU 共享 K 维 tile）
- Tile 计数器是 atomic，自动 work-stealing — 慢的 CU 自然少分

**MMOE 现状**：
- `csrc/grouped_gemm.hip` 是 grid-stride 数据并行，DSV3 expert 形状下大量 CU 闲置
- README Phase 2 把 "work-stealing tile counter" 列为 TODO，但**没提 Stream-K**

#### Stream-K 在 MoE 上的特殊变体

CK 对 grouped GEMM 用的是 **Stream-K + per-expert head pointer**：

```cpp
struct GroupedGemmDesc {
    int expert_id;
    int M_offset;  // 累计 token 偏移（前 expert_id 个 expert 的 token 总数）
    int M;         // 当前 expert 的 token 数
};

// Persistent kernel 一次 launch，原子取下一个 (expert_id, m_block, n_block, k_segment)
__device__ int next_work_id = 0;
while (true) {
    int wid = atomicAdd(&next_work_id, 1);
    if (wid >= total_work) break;
    
    auto [eid, mb, nb, ks] = decode_work_id(wid, descs);
    do_streamk_segment(eid, mb, nb, ks);
}
```

这同时解决了三个问题：
1. **Variable expert size**（有的 expert T_e=128，有的 T_e=4）— 自动 load balance
2. **Kernel launch overhead** — 32 个 expert 用一次 launch 而不是 32 次
3. **Skewed routing** — 慢 expert 的 tile 自然被快 CU 偷走

### 3. 软件流水线深度（NUM_STAGES = 2 vs 3）

#### CK 的 3-stage prefetch

```
Prologue (fill pipeline):
  iter 0: load A[0], B[0] → LDS[0]
  iter 1: load A[1], B[1] → LDS[1]    // LDS[0] 还在 in-flight ds_read
  iter 2: load A[2], B[2] → LDS[2]    // 计算 LDS[0]
  
Main loop (per iter):
  load A[i+2], B[i+2]    // global → reg
  s_waitcnt vmcnt(2)     // 等 i-1 的 vm
  ds_write LDS[(i-1)%3]  // reg → LDS（commit i-1 的）
  
  s_waitcnt lgkmcnt(0)
  s_barrier
  
  compute on LDS[i%3]    // MFMA × K_STEPS
  
  s_barrier
```

**关键**：3 级让 buffer_load 的 ~400 cycle 延迟完全埋在 MFMA 计算下。AMD MI355X 的 buffer_load 延迟是 100–400 cycle，单次 MFMA-32×32×16 是 ~64 cycle，4 个 K-step × 4 个 MFMA = 16 MFMA × 64 = 1024 cycle 主循环，3-stage 完全够。

#### v8a 的 1-stage prefetch

```cpp
// csrc/grouped_gemm.hip 当前结构（简化）：
prefetch(A[0], B[0]);  // prologue 1 个

for (k_tile in K) {
    if (k_tile < K_max-1) prefetch(A[k+1], B[k+1]);  // 1 级
    wait_vm(0);
    ds_write LDS;
    
    compute MFMA on current LDS;
}
```

**问题**：`wait_vm(0)` 在 ds_write 前等所有 in-flight load → 同时只能有 1 个 K-tile 的 load 飞行。在 K=7168 / K_TILE=64 = 112 次主循环里，每次都"等满 → 写 LDS → 计算"，buffer_load 的延迟没法被压住。

**估计修复后增益**：从 530T → 600–650T（约 +15–20%）。

### 4. 精确 s_waitcnt 调度

#### CK 的阶梯式 waitcnt

```asm
ds_read v[10:11], ...    ; lgkmcnt = 4 → 3
ds_read v[12:13], ...    ; lgkmcnt = 3 → 2
ds_read v[14:15], ...    ; lgkmcnt = 2 → 1
ds_read v[16:17], ...    ; lgkmcnt = 1 → 0

s_waitcnt lgkmcnt(2)     ; 等到只剩 2 个未完成（v10, v12 ready）
v_mfma a[0:15], v10, v12, a[0:15]    ; 用 v10 v12

s_waitcnt lgkmcnt(1)     ; 等 v14
v_mfma a[16:31], v10, v14, a[16:31]   ; A 复用 v10
                                       ; 此时 v16 仍未 ready，但下一条不需要
s_waitcnt lgkmcnt(0)     ; 等 v16
v_mfma a[32:47], v16, v12, a[32:47]
v_mfma a[48:63], v16, v14, a[48:63]
```

#### v8a 的现状

`docs/ck_asm_analysis.md` 第 312–319 行展示了 v8a 已经做到了类似的阶梯：

```asm
s_waitcnt lgkmcnt(1)
v_mfma_f32_32x32x8bf16_1k a[0:15], ...
v_mfma_f32_32x32x8bf16_1k a[16:31], ...
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8bf16_1k a[32:47], ...
v_mfma_f32_32x32x8bf16_1k a[48:63], ...
```

**但**：v8a 在 K-step 之间会插入额外的 `s_waitcnt lgkmcnt(0) + s_barrier`（见 `docs/ck_asm_analysis.md` 第 362–368 行）。CK 的 K-step 之间**没有 barrier**，因为 LDS 是 double/triple buffer，写新 buffer 不影响读旧 buffer。

**修复方向**：
1. 用 `__attribute__((address_space(3)))` 强制 ds_read（v8a 已经做了）
2. 把 K-step 之间的 `s_barrier` 拿掉（依赖 LDS double-buffer，新数据写到另一半）
3. 让编译器拒绝插入 `lgkmcnt(0)`：用 `__builtin_amdgcn_s_waitcnt(value)` 显式控制

**估计增益**：+5–10%。

### 5. LDS Swizzle vs Padding

#### v8a 用的是 padding 法

```cpp
// csrc/grouped_gemm.hip
constexpr int PAD = 4;
constexpr int A_LDS_STRIDE = K_TILE + PAD;  // 64 + 4 = 68 bf16
constexpr int B_LDS_STRIDE = K_TILE + PAD;
```

**代价**：每行多 4 个 bf16 = 8 字节，A 是 256×68×2B = 34816 B（vs 32768 B 无 pad）。256×256 tile 双缓冲下：A+B = (34816 + 34816) × 2 = **136 KB LDS** → 1 WG/CU 上限。

#### CK 用的是 XOR swizzle

```
原 layout:    addr(row, col) = row × LDS_STRIDE + col
swizzled:     addr(row, col) = row × LDS_STRIDE + (col ^ ((row & 7) × 4))
```

XOR swizzle 把 bank 冲突散开**而不增加 LDS 占用**。同样 256×256 tile 双缓冲，CK 的 LDS：
- A: 256×64×2B × 2 = 65536 B
- B: 256×64×2B × 2 = 65536 B  
- 合计 = **128 KB**

省下来的 8 KB 让 CK 可以做：
- 更深 prefetch（3-stage 的第三个 LDS slot，~22 KB） → CK 实际是把 LDS 进一步拆成 3-stage + bf16x2 alignment
- 或允许 2 WG/CU 跑（虽然 256×256 几乎用不到）

**MMOE 修复方向**：把 PAD=4 改成 XOR-swizzle，腾出 LDS 给 3-stage prefetch，或减小 tile 到 192×192 留出空间给两路并发 WG。

### 6. C-shuffle Epilogue（被普遍忽视的一项）

#### MFMA 输出的 lane 排布

`v_mfma_f32_32x32x16_bf16` 输出每 lane 16 个 fp32，对应一个 32×32 tile 的 lane 分布是：

```
lane l 持有 acc[i], i ∈ [0,16):
  sub_block = i / 4         (0..3)
  row = sub_block × 8 + (l/32) × 4 + (i % 4)
  col = l % 32
```

→ 同一 row 的 32 个元素分散在 32 个 lane 里。直接逐元素 `*ptr = bf16(acc[i])` 写 HBM 时，每个 lane 单独发一条 store，**完全无法 coalesce**，HBM 带宽只能用到 ~12.5%（128B cacheline / 16B coalesced = 8× loss）。

#### CK 的 C-shuffle

```
1. 把 acc[16] 通过 LDS 重排成 row-contiguous 布局
   - 每个 wave 把自己的 32×32 输出 ds_write 到 LDS 的 [row, col] 位置
   - __syncthreads()
2. 重新 ds_read 一个连续的 [WG_M, vec_width] 段（vec_width=8 bf16 = 16B）
3. buffer_store_dwordx4 写 HBM（128 bit = 8 bf16 一发，coalesced）
```

收益：epilogue 时间从 ~5% kernel time 降到 ~1%。在 K 较短的 GEMM（FC2: K=2048）上 epilogue 占比更大，CK 在 FC2 上 ~960T（接近 GateUP 的 1050T），而 v8a 在 FC2 上 520T、GateUP 530T，**FC2 几乎没退化**说明 v8a 的 epilogue 已经不是主瓶颈** —— 这一项不是 v8a 的最大问题，但仍能再榨 +5%。

#### MMOE 现状

`csrc/grouped_gemm.hip` 后半段（第 ~400 行起）的 C 写回是直接逐 lane store。改成 LDS-assisted 需要在主循环之后单独腾出 ~16 KB LDS（可以和 GEMM 的 LDS 复用，因为 epilogue 时已经不需要 A/B 了）。

### 7. 架构感知 MFMA 选择（MI16x16 vs M32x32）

CK 在 gfx950 上**只用** `v_mfma_f32_32x32x16_bf16`（K=16），不用 16×16×32（虽然 hardware throughput 表面上 16×16×32 = 4× 32×32×16 = 1024 FLOPS / cycle，但 lane 占用模式更碎，导致 ds_read pattern 难做向量化）。

**v8a 实验过 MI16x16**：`docs/ck_asm_analysis.md` 第 526 行报告 v9a (MI16x16) = **467T**，比 v8 的 778T 还差。结论：在我们当前 LDS 布局下 16×16×32 没法加速，**v8 的 32×32×16 选择正确**，这一项不需要改。

### 8. 编译器细节（ROCm 版本敏感性）

`slab/notes/monolith-moe/2026-04-10-mfma-gemm-optimization.md` 第 130–137 行已经记录：

| ROCm 版本 | v4c FC1 |
|---|---|
| 7.1 | 367T |
| **7.2** | **393T** |

CK 在 ROCm 7.2 下 buffer_load 调度更好。**前提**：MMOE 必须**锁定 ROCm 7.2+**（README 第 1 行的 PERF_REPORT 已经是 7.1，但 BENCHMARK_RESULTS 是 7.2）。混着跑会有 ~7% 不可解释的抖动。

---

## CK vs v8a 逐项对照表

| 维度 | CK (1050T, 81% peak) | v8a (530T, 41% peak) | 是否 MMOE 阻塞？ |
|---|---|---|---|
| Tile 形状 | 自动选择（Stream-K 后形状无关）| 静态 256×256 | ⚠️ DSV3 形状下 v8a underfill |
| 工作分配 | Stream-K + atomic counter (work-stealing) | 静态 grid-stride round-robin | ✅ 阻塞，README 已记 TODO |
| Prefetch 深度 | 3-stage | 1-stage | ✅ 阻塞，最大单点收益 |
| Waitcnt 精度 | 阶梯 + 跨 K-step 无 barrier | 阶梯 + K-step 间 barrier | ⚠️ 中等收益 |
| LDS 布局 | XOR swizzle (无浪费) | PAD=4 padding (浪费 8KB) | ⚠️ 间接阻塞（限制 prefetch 深度）|
| Accumulator | AGPR (Raw_avv) | AGPR ✅ | ✅ 已对齐 |
| Operand reuse | A 沿 N 复用 (M→N→K 循环) | A 沿 N 复用 ✅ | ✅ 已对齐 |
| MFMA 选择 | 32×32×16 (gfx950) | 32×32×16 ✅ | ✅ 已对齐 |
| Epilogue | C-shuffle (LDS-assisted) | 逐 lane store | ⚠️ 中等收益，FC2 上影响小 |
| ROCm 版本 | 7.2+ | 7.2+ ✅ | ✅ 已对齐 |
| Persistent kernel | ✅ atomic tile counter | ✅（super-kernel 是 persistent，但 GEMM 部分 round-robin）| ⚠️ 部分 |
| Variable batch grouped | ✅ per-expert head pointer | ❌ 每 expert 独立 launch（在 `bench_grouped_gemm.hip`）| ⚠️ 在 super-kernel 内已经用 fused 路径绕过 |

**未对齐项的预期收益叠加**（按已实测/可估算的中点）：

```
1.20 (Stream-K)
× 1.15 (3-stage prefetch)
× 1.07 (LDS swizzle 解锁 prefetch)
× 1.07 (waitcnt 拿掉 K-step barrier)
× 1.05 (C-shuffle)
≈ 1.62×
```

530 × 1.62 ≈ **860T**，到 CK 的 82%。剩下 18% 的差距来自 CK 的工程细节（micro-tiling、register file bank 调度、HW prefetch hint），需要看 CK 真实源码才能复现 —— 但 **860T 已经足够支撑 MMOE 的 4.8ms 目标**（GEMM 时间从 4.45ms × 1050/860 = 5.4ms 降到接近 baseline）。

---

## 下一步 / 建议

### MMOE GEMM 优化优先级（按 ROI）

| 优先级 | 改动 | 文件 | 预期增益 | 工程量（人天）|
|---|---|---|---|---|
| **P0** | 3-stage prefetch（最大单点）| `csrc/grouped_gemm.hip`、`fused_moe_super_kernel.hip` | +12–18% (530 → 620T) | 3–5 |
| **P0** | Stream-K + persistent atomic tile counter | `fused_moe_super_kernel.hip` 的 expert_compute_phase | +15–20%（DSV3 形状），DSV3 sparse 形状下能到 +50% | 5–7 |
| **P1** | LDS XOR swizzle 替换 PAD=4 | `csrc/grouped_gemm.hip` Section 4 LDS layout | +5–10%（与 P0 联动）| 2–3 |
| **P1** | 拿掉 K-step 间的 `s_barrier`（依赖 double buffer 正确性）| `csrc/grouped_gemm.hip` 主循环 | +3–7% | 1–2 |
| **P2** | C-shuffle epilogue (LDS-assisted vectorized store) | `csrc/grouped_gemm.hip` 后半段 | +3–5%（GateUP 形状），FC2 形状几乎无改变 | 3 |
| **P2** | 自适应 M_TILE / 多版本派发（README Phase 2 TODO）| 编译期模板 + runtime dispatch | DSV3 sparse 形状 +20–40% | 4–5 |
| **P3** | 调研 CK 源码 micro-tile 细节（最后 18% gap）| 调研 + 实验 | +5–10% | 7+ |

**建议执行顺序**：P0 两项**必须先做**，否则 MonolithMoE 的 4.8ms 目标根本碰不到。P1/P2 可并行，P3 在 P0/P1 完成后再评估 ROI。

### 实测建议

每改一项立刻跑：

```bash
# 单独 GEMM 性能
./bench_grouped_gemm  # 输出 GateUP/Down TFLOPS 对比 CK

# 端到端 super-kernel 性能（必看）
benchmarks/run_super_kernel_sweep.sh
```

并把结果落到 `benchmarks/results/` 留档，对比当前的 `super_kernel_sweep.txt`。

### 关于"是否要重写到 ck_tile DSL"

**不建议**。理由：
1. ck_tile DSL 学习曲线陡（要懂 BlockTileWindow / TileDistribution / 多级 partition）
2. CK 在 grouped GEMM 上**不是**直接用通用 DSL 拼出来的，而是 `examples/ck_tile/19_grouped_conv` 等特殊路径，本身就有定制
3. v8a 已经做了 60% 工作（AGPR、buffer_load、ds_read 都有），**剩下的 40% 全部能在裸 HIP 里完成**，不需要引入 CK 模板

**唯一例外**：如果项目要扩到 FP8（README Phase 3），CK 的 `WarpGemmAttributeMfmaImplF8F8F32M32N32K32_*` 已经给了完整的 FP8 wrapper，那时候**值得**直接 link primus_turbo 而不是手写。

---

## 相关文件

| 类型 | 路径 | 说明 |
|---|---|---|
| 源码（v8a）| `csrc/grouped_gemm.hip` | 当前 530T 实现 |
| 源码（super-kernel 内嵌）| `csrc/fused_moe_super_kernel.hip` (lines 200–600) | super-kernel 的 GEMM 部分，与 v8a 同基因 |
| 实测数据 | `benchmarks/BENCHMARK_RESULTS.md` | CK 1050T vs v8a 530T 详表 |
| 已有 CK ASM 分析 | `docs/ck_asm_analysis.md` | 包含汇编片段，但侧重指令级，不覆盖 Stream-K / pipeline 设计 |
| 之前的 GEMM 优化日志 | `slab/notes/monolith-moe/2026-04-10-mfma-gemm-optimization.md` | v2 → v4c 演进，247T → 497T |
| MFMA 性能对比 | `docs/ck_asm_analysis.md` 附录 D（v4c 391T, v8a 778T, CK 1300T BMM）| 注意 778T 是单 GEMM 实验，1050T 是 grouped 实测 |

## 附录：关键术语速查

| 术语 | 含义 |
|---|---|
| **Stream-K** | 把 (M, N, K) 三维 tile 总量按 CU 数切片而不是按 (M, N) 切片，K 维做 partial sum + L2 fixup |
| **WGAttrCtl** | CK 的模板参数，控制 MFMA 的 A/B/C 操作数放在 VGPR 还是 AGPR |
| **C-shuffle** | epilogue 阶段把 lane-distributed 的 accumulator 通过 LDS 重排成 row-contiguous，再向量化 store |
| **NUM_STAGES** | 软件流水线深度，决定主循环里同时有多少个 K-tile 的 buffer_load 在飞 |
| **Persistent kernel** | grid 启动后所有 WG 都不退出，通过 atomic counter 自取下一个 work item，避免 launch overhead |
| **Variable batch grouped GEMM** | 每个 expert 的 M（token 数）不同，需要 per-expert M_offset 数组 |
