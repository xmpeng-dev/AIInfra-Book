# 2026-05-13 12:45  Phase 2.1 — SwiGLU pre-compute → FC2 全 DTOLDS，wall 6.27 → 4.82 ms (−23 %) / 1.755× vs RCCL

> 时间: 2026-05-13 12:45 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, mi355-gpu-26), XGMI 全互联 / 1 节点 8 GPU
> 容器: xiaoming-dev (podman, ROCm 7.2 toolchain)
> 软件: hipcc 6.x · PyTorch 2.12+rocm7.1 · build flag `-DMOE_K_TILE=128`
> 代码: `csrc/fused_moe_super_kernel.hip` · 新增 `swiglu_precompute_phase()` device function (~50 行) + `expert_compute_phase` 编排改写 + FC2 dispatch 切换 + 2 个新 profile bucket (18/19)
> Commit: d273f22（push 前）
> 原始日志: `benchmarks/results/dsv3_sparse_8gpu_phase2p1_swiglu_precompute_2026-05-13.txt`

## 1. 时间点 / 上下文

- 上一次相关进展：[`2026-05-13_1230_super_kernel_dtolds_landed_xor_swizzle.md`](./2026-05-13_1230_super_kernel_dtolds_landed_xor_swizzle.md) — Phase 1 + Phase 2 把 DTOLDS 落地，DSV3 SPARSE wall 6.59 → 6.27 ms / 1.35× vs RCCL，`lds_write` 桶从 2054 → 1419 us（FC1 端归零，**剩下 1419 us 全在 FC2 A**）。
- FC2 A 留在 legacy `ds_write_b128` 路径的原因：`mfma_gemm_tile_swiglu_t` 在 LDS 写入前需要 `silu(gate) * up` 这一步 VGPR 中转，DTOLDS 的 HBM→LDS 一条 DMA 跳过 VGPR，**与 in-kernel fusion 物理互斥**。
- 触发本次工作：Phase 2 完成后的 profile 显示 `fc2_swiglu_tiles = 2612 us`（FC2 整段），FC1 已经被 DTOLDS 砍到 1474 us，FC2 成了 **新的 2× 短板**。打掉 FC2 lds_write 是收益最高的下一刀。

## 2. 问题

DTOLDS 能让 FC1 享受 `lds_write ≈ 0`、HBM 和 MFMA 完全重叠的流水，但 FC2 因为 SwiGLU fusion 而留在 legacy 路径上，导致：

- 现状（Phase 2）：DSV3 SPARSE **6.268 ms / 460 T / 1.35× vs RCCL**；`gemm_total` 3748 us，其中 `lds_write` 1419 us（38 %）来自 FC2 A。
- 目标：让 FC2 A 也走 DTOLDS，估算 wall 能压到 ~5 ms / 1.65× vs RCCL。
- 卡点：必须把 SwiGLU 计算从 GEMM-tile 内部抽出来，否则 DTOLDS 无法绕开 VGPR 中转；同时不能引入比省下来还多的 HBM round-trip 或 barrier 代价。

## 3. 做了什么

### 设计：HBM-resident pre-compute phase + 原地覆盖 fc1_scratch

把 SwiGLU 从 FC2 的内层循环里抽出来，独立成一个 phase 跑在 FC1 → FC2 之间：

| 阶段 | 内容 |
|---|---|
| FC1 输出 | 每 token 写 `fc1_scratch[t][0..2F)`：左半 `gate`，右半 `up`，row stride = `2 * F` |
| **NEW: SwiGLU pre-compute** | 读 `gate`/`up`，把 `silu(gate) * up` 原地覆盖 `fc1_scratch[t][0..F)`，`up` 半区不动 |
| FC2 输入 | FC2 dispatch 改 `K_full = F, a_stride = 2*F`，A 自然只读到前 F bf16（=`silu(gate)*up`），全 DTOLDS |

`up` 半区故意不动 → row stride 保持 `2*F` 不变 → FC1 写入路径零修改、IPC scratch 大小不变。

### 代码改动（按文件）

**`csrc/fused_moe_super_kernel.hip`** —— 主要改动：

| # | 动作 | 关键 diff |
|---|---|---|
| 1 | 新增 `__device__ void swiglu_precompute_phase(...)` device function | section 8.5，~50 行 |
| 2 | 在 `expert_compute_phase()` 里在 FC1 → `compute_barrier_1` 之后插入对 `swiglu_precompute_phase` 的调用，加 `compute_phase_barrier`（共 4 个 barrier / iter，原来 3 个） | section 9 |
| 3 | FC2 dispatch 从 `mfma_gemm_tile_swiglu / mfma_gemm_tile_swiglu_small` 换成 `mfma_gemm_tile / mfma_gemm_tile_small`，参数 `a_stride = 2*F`、`K_full = F` | section 9 |
| 4 | 删掉 `mfma_gemm_tile_swiglu_t` 模板（不再使用） | section 7 |
| 5 | profile bucket layout：加 18 (`swiglu_precompute`) 和 19 (`compute_barrier_swiglu`)；旧 bucket 4 (`t_fc2`) 描述从 "fc2_swiglu_tiles" 改成 "fc2_tiles" | profile macro 区 |

`swiglu_precompute_phase` 核心代码（精简）：

```cpp
__device__ void swiglu_precompute_phase(
    bf16_t* __restrict__ fc1_scratch,
    GemmLdsLayout* lds, int total_pairs, int F,
    int wg_local_id, int num_compute_wgs)
{
    constexpr int VEC = 8;                         // bf16 per uint4
    const int chunks_per_token = F / VEC;
    const int my_global  = wg_local_id * WG_SIZE + (int)threadIdx.x;
    const int wg_stride  = num_compute_wgs * WG_SIZE;
    const int row_stride_b = 2 * F;                // bf16 per fc1_scratch row

    for (int p = 0; p < total_pairs; ++p) {
        int T_e = lds->pair_T_e[p];
        if (T_e == 0) continue;
        int addr_offset = lds->pair_addr_offset[p];

        int total_chunks = T_e * chunks_per_token;
        for (int i = my_global; i < total_chunks; i += wg_stride) {
            int t  = i / chunks_per_token;
            int fv = (i - t * chunks_per_token) * VEC;
            bf16_t* row_base = fc1_scratch + (size_t)(addr_offset + t) * row_stride_b;

            uint4 g = *reinterpret_cast<const uint4*>(row_base + fv);
            uint4 u = *reinterpret_cast<const uint4*>(row_base + F + fv);
            uint4 r;
            bf16_t* gp = reinterpret_cast<bf16_t*>(&g);
            bf16_t* up_p = reinterpret_cast<bf16_t*>(&u);
            bf16_t* rp = reinterpret_cast<bf16_t*>(&r);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                float gf   = to_f32(gp[e]);
                float uf   = to_f32(up_p[e]);
                float silu = gf / (1.0f + expf(-gf));
                rp[e]      = to_bf16(silu * uf);
            }
            *reinterpret_cast<uint4*>(row_base + fv) = r;
        }
    }
}
```

工作分配特点：

- **外层枚举 pair**：从 LDS 复用 FC1 prologue 已经建好的 `pair_T_e / pair_addr_offset` 描述符，零额外建表。
- **内层全员瓜分 chunk**：每 thread 处理 1 个 uint4 = 8 bf16；所有 compute WG × WG_SIZE = 256 × 256 = 64 K threads 同时上，对稀疏 (T_e ≤ 32) workload 也能拿到高并行度。
- **HBM-only**：全程不进 LDS，DRAM bandwidth-bound；DSV3 SPARSE 每 token 多写 F bf16 = 4 KB，整 rank 多 ~2 MB HBM 流量。

**`benchmarks/bench_super_kernel.hip`** —— 加 bucket 18 / 19 的打印输出，`fc2_swiglu_tiles` 重命名 `fc2_tiles`。

**`tests/test_super_kernel_correctness.hip`** —— 自动 rebuild 调用新 super-kernel；CPU 参考逻辑保持 `silu(gate)*up`，结果与 super-kernel pre-compute 路径完全一致。

### 编译与验证

```bash
./benchmarks/results/profile_kt128.sh           # 加 swiglu pre-compute 后 single-GPU profile
./benchmarks/results/e2e_dsv3_sparse_8gpu.sh    # 8-GPU release-build wall sweep
./tests/test_super_kernel_correctness            # 8/8 ranks deterministic
```

Bug 修复：第一版 `swiglu_inplace` 索引乘数算错 (`row * 2 * ffn_size + col` 应该是 `row * ffn_size + col`)，让 in-kernel 退路无法编译；改正后 production 路径只用新 pre-compute phase。

## 4. 效果

### DSV3 SPARSE 8-GPU end-to-end wall（release build, ratio sweep）

| comm_ratio | Phase 2 (ms) | Phase 2.1 (ms) | Δ |
|---|---|---|---|
| 0.150 | 6.773 | 5.816 | −0.96 ms |
| 0.180 | **6.268** | 5.069 | −1.20 ms |
| 0.200 | 6.541 | 5.251 | −1.29 ms |
| 0.225 | 7.194 | 5.774 | −1.42 ms |
| **0.250** | 6.369 | **4.825** ← BEST | **−1.54 ms** |
| 0.300 | 7.675 | 5.687 | −1.99 ms |

**最优运行点 `comm_ratio` 从 0.180 漂到 0.250**：compute 端 (FC1 + pre-compute + FC2) 缩短后，可以多让 64 → 96 个 WG 去跑 scatter / comm，net 收益更高。

| 指标 | PyTorch+RCCL | PAD=8 baseline | Phase 2 | **Phase 2.1** | 累计 Δ vs PAD=8 |
|---|---|---|---|---|---|
| **wall (ms)** | 8.466 | 6.590 | 6.268 | **4.825** | **−1.765 ms / −26.8 %** |
| effective TFLOPS | 341 | 438 | 460 | **598** | **+36 %** |
| vs PyTorch+RCCL | 1.00× | 1.28× | 1.35× | **1.755×** | — |

### Phase-level breakdown（per compute WG / iter，profile build, ratio=0.180）

| Bucket | Phase 2 (us) | Phase 2.1 (us) | Δ |
|---|---|---|---|
| `dispatch_src_ready_wait` | 1503.5 | 1503.5 | 0 |
| `fc1_tiles` | 1474.6 | 1522.8 | +48 |
| `compute_barrier_1` | 372.5 | 372.5 | 0 |
| **`swiglu_precompute`** (NEW) | — | **93.9** | +94 |
| **`compute_barrier_swiglu`** (NEW) | — | **40.7** | +41 |
| `fc2_tiles` (was `fc2_swiglu_tiles`) | **2612.2** | **939.9** | **−1672 us / −64 %** |
| `compute_barrier_2` | 61.9 | 61.9 | 0 |
| `copy_to_combine` | 492.4 | 492.4 | 0 |
| `compute_barrier_3` | 138.7 | 138.7 | 0 |

净算账：`fc2_tiles -1672 us`、`swiglu_precompute + barrier_swiglu +135 us` → **+1537 us 节省**，几乎全部落到 wall。

### GEMM-internal bucket（per compute WG / iter）

| Bucket | Phase 2 | **Phase 2.1** | Δ |
|---|---|---|---|
| `gemm_total` | 3748.4 us | **2045.7 us** | **−1703 us / −45 %** |
| `hbm_issue` | 1102.2 us | 822.5 us | −280 us（FC2 A 只读 F 不读 2F） |
| `wait_vm` | 134.0 us | 148.5 us | +15 us |
| `lds_write` | 1419.0 us | **0.0 us** | **−1419 us / FULLY ELIMINATED** ← |
| `sync_per_ktile` | 43.1 us | 40.5 us | −3 us |
| `mfma_inner` | 588.6 us | 685.2 us | +97 us（XOR + DTOLDS 后 ds_read 略多） |

### TILE-FIT（次要 workload）

| | Phase 2 | **Phase 2.1** | Δ |
|---|---|---|---|
| wall (ms) | 3.864 | **3.389** | −12.3 % |
| TFLOPS | 213 | **243** | +14 % |
| best `comm_ratio` | 0.200 | 0.250 | 同样漂到 0.250 |

### 正确性

| 维度 | 结果 |
|---|---|
| `tests/test_super_kernel_correctness`（8 ranks, BF16 vs CPU 参考, tol abs≤0.5 rel≤0.05） | rank 0..7 `max_abs=0.1718`, `max_rel=0.0003`, `bad=0`，**PASS** |
| `tests/smoke_super_kernel`（零权重 IPC pipeline） | PASS |
| 重复 5 次 deterministic | bit-exact 一致 |

### 定性观察

- ✅ FC2 `lds_write` 彻底归零，GEMM-internal 终于变成 **HBM + MFMA 共拉满**：`hbm_issue 40 %` ≈ `mfma_inner 33 %` 接近 1:1 → 进一步 GEMM 内部优化空间已经接近 0。
- ✅ pre-compute 走 HBM round-trip 比预期省更多：每 token 多写 F bf16 ≈ 2 MB / rank，实测 94 us（含 issue + commit + L2 吸收），远小于省下的 1672 us。
- ✅ wall 4.825 ms 与「A2A 100 % 隐藏」的理论 fused 下限（4.797 ms）相差 **28 us / 0.6 %** —— compute-comm overlap 这一轴已经几乎拉满。
- ⚠️ Compute 缩到 4.8 ms 后 `gather_combine_phase = 7.93 ms / iter` 已经接近 compute 的 2×，下次再压 compute 必须先把 tail 拉出 critical path。
- ⚠️ `dispatch_src_ready_wait` 1503 us 没动，~30 % wall 在纯等 IPC 落地，有 overlap 空间。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | tail-wave critical-path 修正：`gather_combine_phase ≈ 7.93 ms / iter`，atomic-combine 端 HBM latency 反扑 | 估 wall −0.3 ~ −0.8 ms | 需要 batched `atomicAdd<bf16>` 或把部分 combine 移到 scatter 侧 |
| **P0** | `dispatch_src_ready_wait` 1503 us 与 FC1 prologue 重叠 | 估 wall −0.3 ~ −1 ms | 把 pair descriptor 构建、weights L2 prefetch 移到 wait spin 内 |
| **P0** | mxfp8 weights for FC1 / FC2（与 Phase 2.1 正交） | HBM weight 流量 ÷2，估 wall −0.5 ~ −1 ms | weight 量化 pipeline + `v_mfma_scale_f32_32x32x64_f8f6f4` |
| P1 | per-tile-class K_TILE template（default tile 用 M=N=256 K=64，small 保 K=128） | default tile −40 % GEMM-internal，DSV3 SPARSE 不动 | LDS region union + 模板分发；1–2 d |
| P1 | weight pre-permutation to MFMA fragment layout | 去掉 ds_read 重排，−5~10 % FC1 | 离线 weight transform pipeline |
| P2 | C-shuffle epilogue | +3–5 % FC1 | 与 FP8 联动 |
| P2 | TILE-FIT scatter 总时间下降（pack+scatter 合并 / per-rank XGMI link 并行） | TILE-FIT −1 ms | A1/A2 已证伪「让 wait/L2 reuse 帮忙」的路径，必须真正缩短 scatter 物理时间 |
| P3 | Work-stealing tile counter（atomic 替代 round-robin） | 吸收 T_e 不均，−0.2 ~ 0.5 ms |  |

## 相关文件

- 源码：`csrc/fused_moe_super_kernel.hip`（`swiglu_precompute_phase`、`expert_compute_phase`、FC2 dispatch）
- bench harness 改动：`benchmarks/bench_super_kernel.hip`（bucket 18/19 printout、`fc2_tiles` 重命名）
- Phase 2.1 原始日志：`benchmarks/results/dsv3_sparse_8gpu_phase2p1_swiglu_precompute_2026-05-13.txt`
- Phase 2 原始日志（前置）：`benchmarks/results/dsv3_sparse_8gpu_phase2_dtolds_2026-05-13.txt`
- 8-GPU e2e 脚本：`benchmarks/results/e2e_dsv3_sparse_8gpu.sh`
- 上一篇 note：[`2026-05-13_1230_super_kernel_dtolds_landed_xor_swizzle.md`](./2026-05-13_1230_super_kernel_dtolds_landed_xor_swizzle.md)
- Paper-style canvas：`.cursor/projects/.../canvases/monolith-moe-phase2.1-swiglu-precompute.canvas.tsx`
- 正确性测试：`tests/test_super_kernel_correctness.hip`
