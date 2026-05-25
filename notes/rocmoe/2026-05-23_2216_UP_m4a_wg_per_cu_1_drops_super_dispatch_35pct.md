# M4-α UP — `__launch_bounds__(_, 1)` 物理隔离 dispatch / GEMM CU, super-kernel dispatch wall 全 T 段 -35~37%

> 时间: 2026-05-23 22:16 (UTC+8)
> 项目: rocmoe-v2
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 单节点 `mi355-gpu-7`
> 容器: `xiaoming-dev` / `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2, hipcc 22.0.0, PyTorch 2.12+rocm7.1
> 代码: 工作树 `~/workspace/RocMoE/`, HEAD `6fa180f` (M1c-D) + 19:00/20:45/21:15 stash + 本次新增 `ROCMOE_WG_PER_CU` build knob

## 1. 时间点 / 上下文

- 上一篇相关进展: [DSv3 workload DOWN](./2026-05-22_2115_DOWN_dispatch_fc1_dsv3_workload_blocks_at_t512.md) + 在线 summary "dispatch wave-store-drop 用 wave0_b128_copy workaround 临时绕开, 但 super-kernel scaffold tax 比预期严重 (dispatch standalone 16 ms vs super 71 ms @ DSv3 T=8192)"
- 触发事件: user 拍板走 **B 路径 — 用 wave specialization 思路把 dispatch 跟 GEMM 在 CU/SIMD 上隔离**, 减少二者抢 wave scheduler / L2 line / wave-launcher 的 scaffold tax

## 2. 问题

要解决: super-kernel 持久 grid 形态下 dispatch wall **比 standalone dispatch 慢 ~4×** (DSv3 T=8192 standalone 16 ms vs super F=0 71 ms), 这部分 "scaffold tax" 把 chunk-overlap 的物理收益吃掉一半以上.

- 现状 (M2-G 默认 `__launch_bounds__(kWGSize, 2)`, `kNTotalWGs=512`): HW 把 512 个 WG 按 2/CU 均匀打包, **dispatch CU (kNDispatchWGs 个) 上必然有一个 GEMM WG 同居**; 这个共居既是 wave-store-drop hazard 的怀疑成因之一, 也是 dispatch wave scheduler 被干扰的来源
- 目标: 在 super-kernel 内把 dispatch wall 压向 standalone dispatch wall, 让 chunk-overlap 真正吃满
- 卡点: 持久 grid 的 grid_sync 要求 kNTotalWGs ≤ kWGPerCU × N_CU 才能保证所有 WG 共驻; 之前 `__launch_bounds__(_, 1)` 配 kNTotalWGs=512 grid_sync 直接死锁

## 3. 做了什么

### 3.1 落 `ROCMOE_WG_PER_CU` build knob

| 文件 | 改动 |
|---|---|
| `csrc/include/rocmoe/types.h` | 新增 `ROCMOE_WG_PER_CU` (default 2 保旧行为, 1 走 M4-α); `kNGemmWGs = ROCMOE_WG_PER_CU * ROCMOE_N_CU - kNDispatchWGs - tail`; `static_assert(kNTotalWGs <= kWGPerCU * N_CU)` |
| `csrc/super_kernel.hip` | `__launch_bounds__(kWGSize, kWGPerCU)` (rocmoe_super_kernel + rocmoe_super_skeleton_kernel 都改); kNGemmWGs 自动从 440 → 184 (kSubWGs=8) |

WG_PER_CU=1 时 grid 总数 256, HW 按 1/CU 打包, **dispatch CU 上不再有 GEMM 同居**.

### 3.2 build + 测试矩阵

| build dir | flags | 含义 |
|---|---|---|
| `build_w1` | `-DROCMOE_DISPATCH_COPY_WAVES=1` | baseline (WG_PER_CU=2 默认 + wave0-only copy 修 wave-store-drop) |
| `build_m4a_w1` | `-DROCMOE_WG_PER_CU=1 -DROCMOE_DISPATCH_COPY_WAVES=1` | M4-α + wave0-only copy |
| `build_m4a` | `-DROCMOE_WG_PER_CU=1` (默认 WAVES=2) | M4-α 单独不带 wave0-only |

### 3.3 副带 finding: M4-α 自己不修 wave-store-drop

`build_m4a` (WG_PER_CU=1 + WAVES=2 contig) 在 `test_dispatch 8 32 4 128 256 32` 出现完整 wave-store-drop 签名 (`sort#0 token bytes differ`). 说明 **co-residency 不是 wave-store-drop 的根因**, 跟 H=4096 hazard 一样是 multi-wave concurrent store 自己的事; wave0-only workaround 依然必须开着.

### 3.4 bench 矩阵 (DSv3 prod routing, H=7168, balanced, 5 iters / 2 warmup, crit_path p50)

| T | standalone `bench_dispatch` (WAVES=1) | super F=0 WG_PER_CU=2 (build_w1) | super F=0 WG_PER_CU=1 (build_m4a_w1) | Δ super (ms) | Δ super (%) |
|---|---|---|---|---|---|
| 128  | 0.507 | 2.616 | **1.673** | -0.94 | -36% |
| 256  | 0.694 | 3.669 | **2.358** | -1.31 | -36% |
| 512  | 1.131 | 5.611 | **3.562** | -2.05 | -37% |
| 1024 | 2.263 | 9.929 | **6.250** | -3.68 | -37% |
| 2048 | 4.231 | 18.911 | **12.240** | -6.67 | -35% |

注: 上面 super F=0 数据是 dispatch 真做 + GEMM/FC2/Tail 角色走 `noop_busy_wait(smoke_iters)` stub; 因此 super wall = `dispatch + super_kernel scaffold` (B1 cross-rank barrier + grid_sync 在 256/512 WG 上的 atomic 竞争 + persistent-role 调度抢 wave scheduler).

### 3.5 FC1 add-on cost (M4-α + WAVES=1, build_m4a_w1, T 只到 256 因为后面有别的 bug)

| T | F=0 super wall (ms) | F=2048 super wall (ms) | FC1 add-on (ms) |
|---|---|---|---|
| 128 | 1.673 | 2.802 | +1.13 |
| 256 | 2.358 | 3.489 | +1.13 |

FC1 add-on 跟 T 几乎独立 (~1.13 ms), 跟 [20:45 FLAT note](./2026-05-22_2045_FLAT_super_kernel_disp_fc1_first_full_sweep.md) 那一档 FC1 add-on (~0.5 ms, 假 workload experts=32) 对比, DSv3 (experts=256, FC1 FLOPs ×2) 翻倍后还是被 chunk-overlap 大量藏在 dispatch 后面.

## 4. 效果

### 主指标

| 指标 | Before (WG_PER_CU=2) | After (WG_PER_CU=1, M4-α) | Δ |
|---|---|---|---|
| super dispatch wall T=128 (ms) | 2.616 | 1.673 | **-36%** |
| super dispatch wall T=2048 (ms) | 18.911 | 12.240 | **-35%** |
| super dispatch wall / standalone (T=128) | 5.16× | 3.30× | -36% |
| super dispatch wall / standalone (T=2048) | 4.47× | 2.89× | -35% |
| dispatch / GEMM 是否共 CU | 是 | 否 | physical isolation |
| kNGemmWGs (kSubWGs=8) | 440 | 184 | -58% |

### 定性

- ✅ M4-α 在 super-kernel scaffold tax 上稳定 -35%, 跨 T={128, 256, 512, 1024, 2048} 全段一致
- ✅ FC1 chunk-overlap 仍生效, FC1 add-on (~1.13 ms @ DSv3) 跟 standalone bench_gemm DSv3 FC1 roofline (3.25 ms / 1185 TFLOPS) 比, 说明 GEMM 利用率不退化 (kNGemmWGs 184 跟之前 standalone bench 一致)
- ⚠️ M4-α 自己不修 wave-store-drop hazard; wave0-only workaround (WAVES=1) 仍然是 dispatch correctness 必备
- ⚠️ FC1 在 T≥512 + DSv3 routing 下 fused super_kernel **hard crash (HSA illegal memory access)** — 两个 build 都死, 跟 M4-α 正交, 是独立 bug 需要单独 debug
- ❌ 没追完 standalone-vs-super 的剩余 2.89× gap: scaffold 残量主要在 (a) grid_sync 在 256 WG 上的 atomic 竞争 + (b) cross-rank `phase_barrier` ~60 µs × 1 + (c) persistent role 全部 WG 走完 grid_sync<2/3> 的 epilogue
- ⚠️ 节点 cold-start 表现 fragile — GPU 在 sleep state (sclk 94 MHz) 时第一两次 multi-GPU IPC kernel 容易 hard crash, 跑过几次 small bench 暖起来后稳定; 建议 bench 前先 burn-in

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | **M4-β: 真正的 intra-WG wave specialization (dispatch loader/copy split)** | dispatch 内 1-2 wave 跑 sender meta / publish + 2-3 wave 专做 multi-wave cooperative copy (受控并发 + 显式 fence 序列化) — 目标 把 wave0-only 4-5× per-row store 惩罚拿回 50% 左右, 同时绕开 wave-store-drop 而非用 wave0 兜底 | 中 (需先把 wave-store-drop 30 行 repro 拍 ROCm, 才能放心松 1-wave 限制); 跟 hazard 本身物理机理强耦合 |
| **P0** | M2-G β-fix continuation: 把 FC1 fused 在 T≥512 DSv3 下不再 hard crash 弄出来 | 解锁 DSv3 prod (T=2048 / T=8192) 的真实 fused wall 数据, 才能跟 [BF16 7 ms 目标](./README.md) 对齐 | 高 (是 21:15 DOWN 同一 bug 的延伸, T≥512 + FC1 同居才触发, F=0 dispatch-only T=2048 已经能跑) |
| P1 | grid_sync atomic backoff + 256-WG 专属代码路径 | super-kernel scaffold 还有 ~50% 残量 (super 2.358 ms vs standalone 0.694 ms @ T=256), 把 grid_sync 重写成 hierarchical (per-CU `__syncthreads` + 跨 CU `atomic_add_acquire` 阶段化) 或 producer-consumer ticket 模型可能再砍 30% | 中 (类似 cooperative_groups::grid_group 在 AMD 上的 fast-path 实现) |
| P1 | M2-G γ kSubWGs 在 WG_PER_CU=1 下重 sweep | M4-α 把 dispatch CU 数从 [64+co-resident] 改成 [纯 64], kSubWGs=4/16 的最优点跟之前不同 (kNDispatchWGs 改, kNGemmWGs 也改, FC1 跟 dispatch 的 CU 配比重新搜索) | 低 (用 [17:00 FLAT note](./2026-05-22_1700_FLAT_m1c_e_ksubwgs_knob_kept_default_8_post_overlap_remodel.md) 同一 sweep 脚本) |
| P2 | 物理 wave specialization (MFMA-LOADER) for FC1 GEMM body | M4 原设计 (architecture design §3.4): GEMM WG 内 2-2 split, MFMA wave 不接触 HBM | 中 (LDS port contention 需 bank-aware allocation, 跟 mfma_tile.h 模板耦合) |

## 6. 相关文件

- knob + launch 改动: `csrc/include/rocmoe/types.h` (kWGPerCU + kNGemmWGs derivation + static_assert), `csrc/super_kernel.hip` (`__launch_bounds__(kWGSize, kWGPerCU)` × 2)
- 上一篇 (block-on): [`2026-05-22_2115_DOWN_dispatch_fc1_dsv3_workload_blocks_at_t512.md`](./2026-05-22_2115_DOWN_dispatch_fc1_dsv3_workload_blocks_at_t512.md)
- 上一篇 (root hazard): [`2026-05-22_1900_DOWN_m2g_beta_dispatch_h7168_wave1plus_store_drop.md`](./2026-05-22_1900_DOWN_m2g_beta_dispatch_h7168_wave1plus_store_drop.md)
- bench 命令模板 (M4-α): `build_m4a_w1/bench_super_kernel --T <T> --H 7168 --F-moe 0 --iters 5 --warmup 2 --dist dsv3 --num-experts 256 --topk 8 --n-group 8 --group-topk 4 --max-recv-factor 4`
- baseline 命令模板: 同上, 把 `build_m4a_w1` 换成 `build_w1`
- standalone 命令模板: `build_w1/bench_dispatch 8 256 8 <T> 7168 32 1 5 dsv3 32 balanced 4`
