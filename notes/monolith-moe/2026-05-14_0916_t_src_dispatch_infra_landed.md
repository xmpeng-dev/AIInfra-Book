# T_src 分桶 dispatch 基础设施落地（Large/Generic 双 super-kernel）

> 时间: 2026-05-14 09:16 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8x AMD Instinct MI355X (gfx950) / XGMI 全互联 / 单节点 8 GPU (mi355-gpu-26)
> 容器: xiaoming-dev (ROCm 7.2.0, hipcc 22.0.0git roc-7.2.0)
> 软件: HIP 7.2.26015-fc0010cf6a
> 代码: workspace/MMOE @ working tree (uncommitted), `csrc/fused_moe_super_kernel.hip`

## 1. 时间点 / 上下文

- 上一篇：`2026-05-14_0230_session_progress_cycle8_landed_diagnostics.md`（cycle 8 落地后会话总结，T_src=8192 wall 50.87 ms / -9.3 % vs cycle 6）。
- 触发事件：用户明确指示——
  > 「先不要管 T_src=2048 ， 直接盯着 8192 的，实现上可以多一层 dispatch， 每个大小一个 单独的 super kernel」。
  从此 super-kernel 优化只面向真实生产工况（DSV3 seq=4096 × mb=2 → T_src=8192）。

## 2. 问题

把 super-kernel 重新组织成 **按 T_src 分桶分派的多个独立 kernel binary**，让每个 size 都能用自己最优的 tile / comm_ratio / wave layout / 控制流，互不妥协。

- 现状（cycle 8 后单 binary）：T_src=8192 wall = 50.87 ms / TFLOPS = 894（vs PyTorch+RCCL 18.64 ms，落后 2.7×）。
- 目标短期（M2）：T_src=8192 追平 PyTorch+RCCL 18.64 ms（即再砍掉 ~64% wall）。
- 卡点：通用 kernel 里有大量「为小 T_e / 边界 / 空 pair」准备的分支与小 tile 路径，在 T_src=8192 全部是死代码，但仍占 I-cache / 寄存器 / 调度成本；单一 binary 也意味着 tile 形状必须妥协。

## 3. 做了什么

| # | 动作 | 关键文件 / 符号 | 备注 |
|---|---|---|---|
| 1 | 引入 `enum class MoeKernelVariant : int { Generic, Large }` + 阈值常量 `MOE_LARGE_T_SRC_THRESHOLD = 4096` | `csrc/fused_moe_super_kernel.hip:307-330` | T_src ≥ 4096 走 Large（avg_T_e ≥ 128，已塞满 256x256 默认 tile） |
| 2 | `expert_compute_phase` 改成 `template <MoeKernelVariant V = Generic>` 函数 | `csrc/fused_moe_super_kernel.hip:1830` | 在内部用 `if constexpr (V == Large)` 做 size-specific 特化 |
| 3 | 把原 `extern "C" __global__ fused_moe_super_kernel` 拆成 `__device__ __forceinline__ fused_moe_super_kernel_body<V>` + 两个薄 wrapper | `csrc/fused_moe_super_kernel.hip:2572-2693` | 一份源 → 两个独立的 `__global__` binary（Generic / Large） |
| 4 | 新增 `extern "C" __global__ void fused_moe_super_kernel_large(...)` | 同上 | 与 Generic 共享所有 comm/sort/scatter/gather 路径，只有 expert_compute_phase 因 V 不同而 DCE |
| 5 | 改造 `moe_super_kernel_max_blocks_per_cu` 接收 `bool large_variant`，分别缓存两个变体的 occupancy | 同文件 `:2911-2924` | Generic 与 Large 各自独立 query `hipOccupancyMaxActiveBlocksPerMultiprocessor` |
| 6 | `launch_fused_moe_super_kernel` 按 `args.tokens_per_gpu ≥ 4096` 选择 wrapper，并打一行 `[moe-super-kernel] variant=...` 日志（rank 0 一次） | 同文件 `:2998-3070` | Bench / 训练侧无需任何修改 |
| 7 | 第一个真正的 Large 特化：在 `expert_compute_phase` 4 个 `MOE_HAS_SMALL_TILE` 站点外包 `if constexpr (V != Large) { ... continue; }`，让 Large 变体彻底 DCE 小 tile 路径（descriptor build × 2 + tile dispatch × 2，FC1 + FC2） | 同文件 `:2146-2229, 2291-2386` | 等价于「Large 一定走 default 256x256 tile + `mfma_gemm_tile`」 |

### 验证命令

```bash
ssh mi355-gpu-26
podman exec -it xiaoming-dev bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE

hipcc -std=c++17 -O3 --offload-arch=gfx950 -I csrc \
    -DMOE_M_TILE=256 -DMOE_N_TILE=256 -DMOE_K_TILE=64 \
    -o benchmarks/results/bin/bench_sk_k64 benchmarks/bench_super_kernel.hip

# Large 路径
benchmarks/results/bin/bench_sk_k64 --tokens 8192 --hidden 7168 --ffn 2048 \
    --epg 32 --topk 8 --num-cus 256 --wgs-per-cu 1 --comm-ratio 0.250 \
    --warmup 5 --iters 30
# stderr 头一行：[moe-super-kernel] variant=Large (tokens_per_gpu=8192, threshold=4096)

# Generic 路径
benchmarks/results/bin/bench_sk_k64 --tokens 2048 ... 同上
# stderr 头一行：[moe-super-kernel] variant=Generic (tokens_per_gpu=2048, threshold=4096)
```

## 4. 效果

### Wall 时间（vs cycle 8 baseline，相同设置 256x256x64 + ratio=0.250）

| T_src | 走的 variant | cycle 8 wall | 现在 wall | Δ |
|---|---|---|---|---|
| 8192 | Large | 50.87 ms | **50.29 ms** | -1.1 % (≈噪声) |
| 2048 | Generic | 13.09 ms | **13.13 ms** | +0.3 % (≈噪声) |

### 定性

- ✅ **dispatch 基础设施 0 回归**：Generic 变体 latency 与 cycle 8 baseline 完全一致（13.09 vs 13.13 ms，差异在跑间噪声范围内），验证 template 拆分没破坏 codegen。
- ✅ **Large 变体可独立演化**：未来所有 T_src=8192 专属优化只需在 `expert_compute_phase<Large>` 内部 `if constexpr` 落地，不会污染 Generic / 小 T_src 路径。
- ✅ **stderr 一行可观测**：每个 process 第一次 launch 打印 `variant=Large (tokens_per_gpu=8192, threshold=4096)`，方便训练侧确认走对了路径。
- ⚠️ **DCE small-tile 单独无收益**：第 7 步把 4 处 `MOE_HAS_SMALL_TILE` branch 在 Large 变体里彻底 DCE，但 wall 没动（50.32 → 50.29 ms）。说明这条路径在 production 路由分布下本就极少被采（avg_T_e=256 ≫ 32），编译器之前就已经预测得很好。**dispatch 基础设施本身是 0 收益但 0 代价，真正的 Large 特化要靠后续 cycle**。
- ❌ **没有立即拿到 wall 减少**：cycle 14 = 纯基础设施 + DCE 试水，没动到任何热路径里的 LDS / MFMA 调度。

## 5. 可持续方向

下一步在 Large 变体 (T_src=8192) 内部继续榨干，按 ROI 排序：

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| P0 | **跨 src 合 batch GEMM**：每个 expert 把 8 srcs (T_e≈256) 合成 M=2048 的单次 GEMM。w1_e 重用 8 倍 / 全部 prologue × 8 → 1。可将 FC1 17 ms → 估算 8 ms。 | -8~10 ms wall (-16~20 %) | 需要把 dispatch 后的 token 摆放从 `[src][T_e]` 改成 `[expert][T_e × 8]` 连续布局；或加一次 ~120 us 的 pre-pack。 |
| P0 | **FC1 后立即把 SwiGLU + scatter-output 合进 FC2 prologue**：彻底吃掉 swiglu_pc 1.2 ms + bar_pc 0.4 ms + 一次 HBM round-trip。 | -1.5 ms wall | epilogue layout 大改，必须配合 cycle 9 swiglu fuse 重启动。 |
| P1 | **Large 专属 closed-form pair 调度**：Large 下 num_n 是常量（FC1=16，FC2=28），且 99.9 % 情况 num_m=1，可把 `pair_id = gt / num_n` 直接算出来，砍掉 LDS 上的 `pair_tile_offset[]` 线性扫和 `pair_kind[]` / `pair_T_e[]` 读。 | -0.3~0.5 ms wall | 要保留对 T_e > M_TILE 的 fallback 分支；hot path 干净就行。 |
| P1 | **`mfma_gemm_tile_t` 增加 `kFullTile` 模板参数**：Large 调度时 m_actual==M_TILE && n_actual==N_TILE 走 `<true>` 实例化，DCE 掉 valid_m/valid_n 的 cmp/select。 | -0.2~0.4 ms wall | 增加 binary code 但只在 Large kernel 内。 |
| P1 | **重新 profile Large 变体（cycle 14b）**：现在已经走 Large + cycle 8 parallel-peer copy，要重新拿 dispatch_wait / fc1 / fc2 / mfma_inner / hbm_issue 比例，确认 cycle 6 数字还成立。 | 不动 latency，定方向 | 已在后台跑 `bench/cycle6_profile_new_optimum.sh`，结果出来后接到本 note 后续。 |
| P2 | **把 cycle 12 的真实训练 + Large 变体 wire 到 Primus 4-layer 验证**：单跑 bench 无法复现 dispatch_wait 的真实分布（单节点是 launch skew artifact），训练里跑一次才能给 cycle 11 / 11b 定性。 | 校准方向用 | 需要 reset Primus 训练 + 收 `MMOE_FWD_PROFILE` log。 |

> ROI 重排说明：用户要求只追 T_src=8192，所以以前面向 small-T_src 的 cycle（small-tile / comm_ratio 在 sparse 工况下的表现）从下一步表里全部移除；M1 = 追平 PyTorch+RCCL @ T_src=2048 已不再追求，但 Generic 变体仍然存在（小 batch 推理 / 4-layer 验证仍会路由到它），不会主动 regress。

## 6. 附：Cycle 14b — Large 变体的 profile 重测（dispatch infra 落地后）

跑 `bash benchmarks/cycle6_profile_new_optimum.sh` 拿 profile-enabled 数据（256x256x64 + ratio=0.250 + cycle 8 parallel-peer copy + cycle 14 dispatch infra），结果与 cycle 6 一致 —— **没有任何回归**：

| WG 角色 | bucket | T_src=2048 (Generic) | T_src=8192 (Large) |
|---|---|---|---|
| 总 wall | latency_ms | 13.36 ms | **51.24 ms** |
| compute | dispatch_src_ready_wait | 4.70 ms | **17.99 ms (35.1 %)** |
| compute | ├─ until_first_src | 4.21 ms | 16.98 ms (94 %) ← 单节点 clock-domain artifact |
| compute | └─ first→all (skew) | 0.49 ms | 1.01 ms (6 %) |
| compute | fc1_tiles | 4.41 ms | **17.27 ms (33.7 %)** |
| compute | swiglu_precompute | 0.32 ms | 1.22 ms |
| compute | fc2_tiles | 2.92 ms | **11.14 ms (21.7 %)** |
| compute | copy_to_combine | 0.75 ms | 2.88 ms (5.6 %) ✓ cycle 8 |
| compute | barriers ×3 | 0.21 ms | 0.66 ms |
| comm | sort + scatter | 2.43 ms | 9.15 ms |
| GEMM-internal | gemm_total | 6.24 ms | **24.49 ms** |
| GEMM-internal | ├─ hbm_issue | 2.21 ms (35 %) | 8.77 ms (35.8 %) |
| GEMM-internal | ├─ wait_vm | 0.14 ms (2 %) | 0.43 ms (1.8 %) |
| GEMM-internal | ├─ lds_write | **0.00 ms (0 %)** | **0.00 ms (0 %)** ✓ Phase 2.1 全 DTOLDS |
| GEMM-internal | ├─ sync_per_ktile | 0.07 ms (1 %) | 0.27 ms (1.1 %) |
| GEMM-internal | └─ mfma_inner | 2.56 ms (41 %) | **10.12 ms (41.3 %)** |
| GEMM-internal | prologue/epilogue (剩余) | 1.27 ms (20 %) | **4.89 ms (20 %)** |

### 解读

- **GEMM 才是真正的 wall 大头**：FC1 + FC2 = 28.4 ms (55 % wall)，单看 mfma_inner 10.12 ms ≈ 22 % theoretical peak (5.76 TFLOP / 1.25 PFLOPS = 4.6 ms)。GEMM 还有 ~2× 空间，但需要 software pipeline / LDS layout 这种深层改动。
- **dispatch_wait 17.99 ms 90 %+ 是 "until_first_src"**：cycle 11 已经定性为单节点 clock-domain measurement artifact，但**仍然必须用真实训练验证**——如果训练里也是 ~18 ms wait，就是真瓶颈，需要 IPC 同步重新设计；如果训练里掉到 ~1-2 ms，那 wall 自动从 51 → 33 ms。**这是接下来 ROI 最高的一步**。
- **swiglu_precompute 1.22 ms + barrier 1.04 ms ≈ 2.3 ms** 仍是 cycle 9 的目标，但比 GEMM 的 ~28 ms 小一个数量级。
- **copy_to_combine 2.88 ms** = cycle 8 落地后的现状，xGMI 利用率已 93 %，难再压。

### 修订的 ROI 表（基于 profile 数据）

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | **Cycle 15 — 真实 4-layer 训练 @ T_src=8192 验证 dispatch_wait** | 如果训练里 wait → 1-2 ms，自动省 -30 % wall（51 → 33 ms） | 需要把 4-layer yaml 的 seq×mb 从 2048 改成 8192；约 30 min 跑 + 看 profile log |
| P0 | **Cycle 16 — 跨 src 合 batch GEMM (M=2048 per expert)** | -3~5 ms wall（FC1 17 → 12-14 ms），主要省 per-tile prologue/epilogue ×8 | 改 dispatch buffer 物理布局或加 pre-pack；中等结构性改动 |
| P0 | **Cycle 17 — GEMM software pipelining 深挖（mfma_inner ↓ to 1.5× theoretical）** | -3~5 ms wall（mfma_inner 10 → 5-6 ms） | 需要 ds_read 重叠 mfma 的高级 pipeline；高风险，可能要 inline asm |
| P1 | Cycle 18 — `mfma_gemm_tile_t` 增 `kFullTile` 模板（Large 走 `<true>` DCE valid_m/valid_n cmp） | -0.2~0.4 ms wall | 改 binary code size，只对 Large 有效 |
| P1 | Cycle 19 — Large 专属 closed-form pair lookup（去掉 lds 上的 `pair_tile_offset` 线性扫） | -0.3~0.5 ms wall | 要保留 num_m>1 的 fallback |
| P2 | Cycle 20 — 拆分 `prologue/epilogue` 4.89 ms 子 bucket 找下一个 hot spot | informational | 加 4-6 个 PROFILE bucket |

→ **下一步立刻做 Cycle 15 (real training validation)**。它可能直接证明 wall 大头是 measurement artifact，从而决定 Cycle 16/17 的优先级。

## 相关文件

- 代码 patch：`csrc/fused_moe_super_kernel.hip`（dispatch 基础设施 + DCE small-tile）
- 上游 note：
  - `slab/notes/monolith-moe/2026-05-14_0230_session_progress_cycle8_landed_diagnostics.md`（cycle 8 总结）
  - `slab/notes/monolith-moe/2026-05-14_0130_cycle8_parallel_peer_copy_minus10pct.md`（cycle 8 详情）
- profile 原始日志：
  - 控制台 + `benchmarks/results/cycle6_profile_new_optimum_20260514_0116.txt`（节内 §6）
- 验证日志：本机控制台输出（见上 §3 验证命令）
