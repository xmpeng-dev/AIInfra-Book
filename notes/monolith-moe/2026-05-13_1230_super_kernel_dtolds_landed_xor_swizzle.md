# 2026-05-13 12:30  Phase 1 + Phase 2 — super-kernel 落地 DTOLDS + XOR swizzle，wall 6.59 → 6.27 ms (−5 %)

> 时间: 2026-05-13 12:30 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, mi355-gpu-26), XGMI 全互联 / 1 节点 8 GPU
> 容器: xiaoming-dev (podman, ROCm 7.2 toolchain)
> 软件: hipcc 6.x · PyTorch 2.12+rocm7.1 · build flag `-DMOE_K_TILE=128`
> 代码: `csrc/fused_moe_super_kernel.hip` · 关键改动 ~150 行（write/read 端 XOR swizzle + DTOLDS_X4 macro + 反向 HBM gather）
> 原始日志: `benchmarks/results/dsv3_sparse_8gpu_phase1_xor_2026-05-13.txt`、`benchmarks/results/dsv3_sparse_8gpu_phase2_dtolds_2026-05-13.txt`

## 1. 时间点 / 上下文

- 上一次相关进展：[`2026-05-12_2130_dtolds_landed_grouped_gemm_plus204pct_super_kernel_reverted.md`](./2026-05-12_2130_dtolds_landed_grouped_gemm_plus204pct_super_kernel_reverted.md) — DTOLDS 在 standalone Grouped GEMM 上 +204 %，但首次移植到 super-kernel 反退 12 %，root cause 锁定为 PAD=0 prereq 触发 ds_read 端 bank conflict（DSV3 SPARSE 走的 small-tile 路径 MFMA 密度低，hide 不住）。
- 同时另一条线把 PAD=8 + K_TILE=128 的最终 8-GPU 端到端实测拿到（`benchmarks/results/dsv3_sparse_8gpu_e2e_2026-05-13.txt`）：DSV3 SPARSE **7.95 → 6.59 ms / 363 → 438 T / 1.07× → 1.28×**。这是 Phase 1 / Phase 2 的 baseline。
- 设计决策：在 ROI 评估两条路线后用户选 **A. 设计 wave-block LDS layout + DTOLDS**（B 选项是 mxfp8 weights，简单但收益有限）。两阶段计划：
  - **Phase 1**：先用 PAD=0 + 显式 XOR swizzle 在 *legacy `ds_write_b128` 路径* 上跑通新物理布局（必须 conflict-free，否则之后 DTOLDS 写进去 ds_read 还是会塌）。
  - **Phase 2**：把 `ds_write_b128` 替换成 `buffer_load_dwordx4_lds` (DTOLDS)，HBM gather 用反 XOR 让 DTOLDS 的*线性写*命中相同的 swizzled 物理位置。

## 2. 问题

DTOLDS 在 standalone Grouped GEMM 上证明可行（GateUP 199 → 643 T / +222 %），但 super-kernel 移植反退。诊断锁定到：

- DTOLDS 要求 LDS 行连续布局（`PAD=0`），不能跨 row 留 stride 空隙。
- `PAD=0` 单独切换会让 32 lanes 的 `ds_read_b128` 全部落在同一 bank quartet → 32-way bank conflict → ds_read latency ~100+ cycles。
- DSV3 SPARSE 走 small-tile 路径只有 1 MFMA / K-step (~32 cycles)，完全 hide 不住。
- 验证：单独切 PAD=0、不动 DTOLDS，`mfma_inner` 桶从 588 → 2470 us，证伪「PAD=0 单独安全」。

目标：

- 现状（PAD=8 + K_TILE=128 baseline）：**6.59 ms / 438 T / 1.28× vs PyTorch+RCCL**。
- 目标：把 DTOLDS 真正用进 super-kernel 而不退回，并把 ASM-level `lds_write` 桶从 50 % 砍下来。
- 卡点：DTOLDS 的*线性写* vs LDS conflict-free 布局所需要的*非线性写* 在硬件层面互斥；要找到能同时满足两者的物理布局。

## 3. 做了什么

### Phase 1 · PAD=0 + XOR swizzle on legacy `ds_write_b128` 路径

| # | 动作 | 关键 diff / 文件 |
|---|---|---|
| 1 | 把 `constexpr int LDS_PAD = 8` 改成 `constexpr int LDS_PAD = 0`，多省 ~16 KB LDS | `csrc/fused_moe_super_kernel.hip` |
| 2 | 加 `constexpr bool LDS_XOR_SWIZZLE = (LDS_PAD == 0);` 编译期开关 | 同上 |
| 3 | 加 helper `__device__ __forceinline__ uint32_t lds_swizzle_k_block(uint32_t k_block, uint32_t row)`，定义 `k_block ^ (row & 7)` | 同上 |
| 4 | 修改 `lds_frag_ds<>` (读端)：对 `k_block` 调用 swizzle 后再算 LDS 字节地址 | 同上 |
| 5 | 修改 `mfma_gemm_tile_t / mfma_gemm_tile_swiglu_t` 的 `ao[p] / bo[p]`（写端）：把 `va / vb` 通过 swizzle 重排再写 LDS | 同上 |

**关键洞察**：stride 256 B（PAD=0 下 row 长度等于 `K_TILE * 2 B = 256 B`）正好是 2× bank period，单独看 32 lanes 的 `ds_read_b128` 一次取 32 × 16 = 512 B，落在 8 个 distinct bank quartet × 4 sub-cycles = **conflict-free**。XOR `(row & 7)` 在 8-row 周期内打散 row 偏移，避免相邻 row 的 same-k_block 落到同一 quartet。

`ao` / `bo` 写端用 *same XOR*、`lds_frag_ds` 读端也用 *same XOR*，逻辑上 `A[m][k]` 不变，只是物理位置被同步重排了。这是 Phase 1 「不变更 GEMM 语义」的必要条件。

### Phase 2 · DTOLDS 替换 `ds_write_b128`

| # | 动作 | 关键 diff / 文件 |
|---|---|---|
| 1 | 加 `constexpr bool LDS_USE_DTOLDS = LDS_XOR_SWIZZLE;` | `csrc/fused_moe_super_kernel.hip` |
| 2 | 加 `DTOLDS_X4` macro（参考 `csrc/grouped_gemm.hip`，调用 `__builtin_amdgcn_raw_buffer_load_lds_x4`） | 同上 |
| 3 | 重构 `mfma_gemm_tile_t` (FC1)：`ao[p] / bo[p]` 用 **自然 offset**；HBM gather 偏移 `a_boff[p] / b_boff[p]` 改成 **反 XOR**（`va_load = va ^ (ra & 7)`）；staging 用 `if constexpr (LDS_USE_DTOLDS) DTOLDS_X4(...);` 否则 fall back to legacy | 同上 |
| 4 | 重构 `mfma_gemm_tile_swiglu_t` (FC2)：B 走 DTOLDS（无 fusion），A 保留 legacy + swizzle（SwiGLU `silu(gate)*up` 需要 VGPR 中转） | 同上 |
| 5 | OOB protection：persistent kernel 不能让 stale data 留 LDS，超 `valid_m / valid_n` 的 row 用 `voff = 0x80000000` 让 HW clamp 为 0 | 同上 |
| 6 | **bug fix**：第一次实现 valid_n 检查写成 `(n_offset + rb_phase[p] < valid_n)`，把全局索引拿去和 tile-local count 比，导致 `n_offset > 0` 的 tile 整行被 mask，输出全 0。改成 `(rb_phase[p] < valid_n)` 后 8/8 ranks PASS | 同上 |

**关键洞察**：DTOLDS 的写策略是 `lane k` 把数据写到 `M0 + k * 16`（线性、wave-uniform base），不能像 `ds_write_b128` 那样按线程任意 LDS 地址写。但是只要让 lane k 从 HBM 取的是 *k_block (k ^ (row & 7))*，落到 LDS 上的物理位置自然就成了 Phase 1 那个 swizzled 布局。
等价于把 swizzle 从 LDS-side 移到 HBM-gather-side，DTOLDS 的硬件约束被满足，conflict-free 还保持。

### 编译与验证

```bash
# build
cd csrc && hipcc -std=c++17 -O3 --offload-arch=gfx950 \
  -DMOE_K_TILE=128 -c fused_moe_super_kernel.hip ...

# correctness
./tests/test_super_kernel_correctness    # 8/8 ranks, max_abs=0.1718, max_rel=0.0003
./tests/smoke_super_kernel               # PASS

# perf
./benchmarks/results/profile_kt128.sh    # MOE_PROFILE build
./benchmarks/results/e2e_dsv3_sparse_8gpu.sh   # 8-GPU release-build wall sweep
```

## 4. 效果

### DSV3 SPARSE 8-GPU end-to-end wall（release build）

| 指标 | PAD=8 baseline | Phase 1 | Phase 2 | Δ vs baseline |
|---|---|---|---|---|
| **wall (ms)** | 6.590 | 6.535 | **6.268** | **−4.9 % / −322 us** |
| effective TFLOPS | 438 | 442 | **460** | **+5.0 %** |
| vs PyTorch+RCCL (8.466 ms) | 1.28× | 1.30× | **1.35×** | — |
| best `comm_ratio` | 0.180 | 0.180 | 0.180 | unchanged |

### GEMM-internal bucket（per compute WG / iter，profile build）

| Bucket | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| `gemm_total` | 4050 us | **3748 us** | **−302 us / −7 %** |
| `hbm_issue` | 702 us | 1102 us | +400 us (DTOLDS DMA 比 buffer_load 短指令长) |
| `wait_vm` | 143 us | 134 us | −9 us |
| `lds_write` | 2054 us | **1419 us** | **−635 us / −31 %**  ← DTOLDS 把 FC1 端整段干掉 |
| `sync_per_ktile` | 40 us | 43 us | +3 us |
| `mfma_inner` | 631 us | 588 us | −43 us（XOR + DTOLDS 配合略减 ds_read 仲裁） |

净效益：`lds_write -635 us / hbm_issue +400 us`，重叠后 `gemm_total -302 us`，最后传到 wall 是 −267 us（与 e2e wall 减少 322 us 吻合）。

### TILE-FIT（次要 workload，ratio=0.200）

| | Phase 1 | Phase 2 |
|---|---|---|
| wall | 3.954 ms / 208 T | **3.864 ms / 213 T** (−2.3 %) |
| `gemm_total` | 1212 us | 1103 us |
| `lds_write` | 564 us | 377 us (−187 us) |

### 定性观察

- Phase 1 单独看 wall 持平（6.59 → 6.535），但读 / 写端布局都成 conflict-free，是 Phase 2 的必要前置；不要因「持平」放弃 Phase 1。
- DTOLDS 在 super-kernel 上能跑出净收益的关键不是 DTOLDS 本身，是「**HBM gather 反 XOR**」让线性写命中 swizzled 物理位置 —— 把 Phase 1 的写端 XOR 从 LDS-side 转移到 HBM-side。
- FC2 的 A 还在 legacy 路径上（SwiGLU 融合需要 VGPR 中转）→ `lds_write` 桶剩下的 1419 us 全部来自 FC2-A。这是下一步的明确突破口。
- Correctness：8/8 ranks PASS，`max_abs=0.1718, max_rel=0.0003`，与 PAD=8 baseline 完全一致（仅物理布局变化，逻辑等价）。
- ❌ TILE-FIT Phase 1 单独看是 −4 % 回退（3.79 → 3.954 ms），但 Phase 2 拿回来 +2 % → 净 −1.5 %，可接受。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | **Phase 2.1：把 SwiGLU 从 FC2 内部抽出来做 HBM-resident pre-compute** | FC2 A 可以全 DTOLDS，`lds_write` 直接归零，估 wall ~5 ms / −20 % | 多一次 HBM round-trip（每 token 多写 F bf16），需 cross-WG barrier；估收益 −1.4 ms 可以盖过 +135 us 代价 |
| P0 | tail-wave critical-path 修正（`gather_combine_phase ≈ 7.93 ms / iter`） | compute 缩到 ~5 ms 后 tail 可能反扑成 wall 上限 | 需要 batched atomic_add 或部分 combine 移到 scatter 侧 |
| P0 | mxfp8 weights for FC1 / FC2 | HBM weight 流量 ÷2，估 wall −2 ms | weight 量化 pipeline + GEMM tile 升级 `v_mfma_scale_f32_32x32x64_f8f6f4`；与 Phase 2 正交 |
| P1 | per-tile-class K_TILE template（default tile 用 M=N=256 K=64，small 保 K=128） | default tile 单独 −40 %，DSV3 SPARSE 不动 | LDS region union + 模板分发；1–2 d |
| P2 | C-shuffle epilogue | +3–5 % FC1，FC2 影响小 | 与 FP8 联动 |
| P3 | Work-stealing tile counter（atomic 替代 round-robin） | 吸收 T_e 不均，−0.2~0.5 ms | persistent kernel 内 atomic 自然支持 |

## 相关文件

- 源码：`csrc/fused_moe_super_kernel.hip`
- Phase 1 原始日志：`benchmarks/results/dsv3_sparse_8gpu_phase1_xor_2026-05-13.txt`
- Phase 2 原始日志：`benchmarks/results/dsv3_sparse_8gpu_phase2_dtolds_2026-05-13.txt`
- 8-GPU e2e 脚本：`benchmarks/results/e2e_dsv3_sparse_8gpu.sh`
- 上一篇相关 note：[`2026-05-12_2130_dtolds_landed_grouped_gemm_plus204pct_super_kernel_reverted.md`](./2026-05-12_2130_dtolds_landed_grouped_gemm_plus204pct_super_kernel_reverted.md)
- 正确性测试：`tests/test_super_kernel_correctness.hip`、`tests/smoke_super_kernel.hip`
