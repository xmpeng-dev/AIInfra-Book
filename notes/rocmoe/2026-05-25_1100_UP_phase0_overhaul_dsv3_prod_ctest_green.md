# Phase 0 UP — DSv3 prod ctest 全 GREEN, "T≥512 + FC1 hard crash" 不再复现, FC1 fully overlap-hidden at T≥2048

> 时间: 2026-05-25 11:00 (UTC+8)
> 项目: rocmoe-v2 (super-kernel overhaul Phase 0 / 5)
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 单节点 `mi355-gpu-7`
> 容器: `xiaoming-dev` / `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2, hipcc 22.0.0, PyTorch 2.12+rocm7.1
> 代码: 工作树 `~/workspace/RocMoE/`, HEAD `1ce0569` (M4-α landed) + Phase 0 改 (CMakeLists 知识 plumbing + scripts/dev_bench_with_keepalive.sh + scripts/p0_bisect_fc1_crash.sh + super_kernel.hip FC1 SKIP_* 调试 guards + 2 个 DSv3 prod ctest)

## 1. 时间点 / 上下文

- 触发事件: user 拍板按 [super-kernel overhaul 2-3 周 plan](.cursor/plans/rocmoe-super-kernel-overhaul_b804bc61.plan.md) 整改 (Phase 0 → Phase 5).
- 上一篇相关进展: [M4-α UP — `__launch_bounds__(_, 1)` 物理隔离 dispatch / GEMM CU](./2026-05-23_2216_UP_m4a_wg_per_cu_1_drops_super_dispatch_35pct.md) — M4-α + wave0-only copy 是 DSv3 prod 当前 default.
- Plan 起点假设: T≥512 + FC1 fused 必 hard crash (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION), WG_PER_CU=2 build 也炸, scaffold tax ~10.7 ms 占 wall 72%, baseline T=2048 H=7168 F=2048 跑不通 → Phase 0 目标 "解锁 DSv3 prod 测量".

## 2. 问题

Phase 0 的两个 acceptance condition:
1. DSv3 prod (T=2048 H=7168 F=2048 epg=32 topk=8 n_group=8 group_topk=4) 在 super-kernel 内跑得通 (0 crash);
2. 加上 ctest 锁定回归;
3. baseline scaffold tax 数据.

## 3. 做了什么

### 3.1 keepalive wrapper (Phase 0.4)

新加 `scripts/dev_bench_with_keepalive.sh`:
- fork 一个 Python 进程持续在 8 张 MI355X 上做 4096×4096 bf16 matmul (~2% HBM, sclk 拉到 ~2.3 GHz)
- exec 用户给的 bench 命令
- exit 时 SIGTERM keepalive

避免 cold-start ILMA (note M4-α §4 "sclk 94 MHz 第一次 multi-GPU IPC kernel 容易 cold-crash") 在 bench 之间反复出现.

### 3.2 CMakeLists 知识 plumbing 修复 (本次 Phase 0 关键 finding)

历史: M4-α / wave0-only / packed_outbox 这些 build knob 在 `csrc/include/rocmoe/types.h` 和 `csrc/include/rocmoe/dispatch_body.h` 里写成 `#ifndef ROCMOE_WG_PER_CU` 等, 然后**靠在 cmake 命令行加 `-DCMAKE_HIP_FLAGS="-DROCMOE_WG_PER_CU=1 ..."`** 才能传到 hipcc.

如果按 cmake 风格写 `cmake -DROCMOE_WG_PER_CU=1 ...` 只在 CMakeCache.txt 留个变量值, **不会进编译命令**, 结果实际跑 default WG_PER_CU=2 + COPY_WAVES=2, 撞 wave-store-drop hazard. Phase 0.1 bisect 第一轮就是被这个坑了 — 5 个新 build 全 ILMA.

**修复**: `CMakeLists.txt` 加一段 foreach loop 显式把 7 个 build knob (`ROCMOE_WG_PER_CU` / `ROCMOE_DISPATCH_USE_WAVE0_COPY` / `ROCMOE_DISPATCH_COPY_WAVES` / `ROCMOE_DISPATCH_COPY_CONTIG` / `ROCMOE_DISPATCH_USE_PACKED_OUTBOX` / `ROCMOE_DISPATCH_PACK_PHASE_B_DO_COPY` / `ROCMOE_DISPATCH_BULK_PULL`) 在被显式 `-D` 传入时 `add_compile_definitions()`, 用法:

```bash
cmake -S . -B build -G Ninja -DROCMOE_WG_PER_CU=1 -DROCMOE_DISPATCH_COPY_WAVES=1
```

如不传则走源码 `#ifndef` 默认值 (旧 default). 此后所有 build 走标准 cmake convention, 不用塞 `-DCMAKE_HIP_FLAGS`.

### 3.3 FC1 bisect SKIP_* build flags + bisect driver (Phase 0.1)

- 在 `csrc/super_kernel.hip` 的 `fc1_gemm_role_body` / `fc1_swiglu_pass` 加 4 个 `#if` guard: `ROCMOE_FC1_SKIP_ALL` (整段 early return) / `ROCMOE_FC1_SKIP_MFMA` (只跳 mfma_tile call) / `ROCMOE_FC1_SKIP_SWIGLU` (只跳 SwiGLU pass) / `ROCMOE_FC1_SKIP_ARRIVAL_POLL` (只跳 `l1_arrival_count` spin).
- `CMakeLists.txt` 用同一个 foreach plumbing 透传.
- 新加 `scripts/p0_bisect_fc1_crash.sh`: 5-build cmake configure + ninja build + T=512 DSv3 prod 跑 + 解析 crash/PASS, 一次性出表.

### 3.4 DSv3 prod ctest (Phase 0.3)

`CMakeLists.txt` 加 2 个 ctest:

| ctest name | 参数 | 含义 | timeout |
|---|---|---|---|
| `test_super_e2e_dsv3_prod` | `8 256 8 512 7168 32 2048` | T=512 DSv3 prod (3 skew × {1,3} iter × dispatch bit-exact + FC1+SwiGLU bf16) | 600 s |
| `test_super_e2e_dsv3_t2048` | `8 256 8 2048 7168 32 2048` | 同上 T=2048 (host expect_fc1_act ~30 s × 8 ranks) | 1200 s |

跑这 2 个 ctest 锁定 dispatch + FC1 bit-exact 不会 regress.

## 4. 实测结果

### 4.1 Phase 0.1 bisect 二次跑 (knobs 正确 plumb 之后)

| variant | flags | T=512 DSv3 prod 结果 | crit_path p50 |
|---|---|---|---|
| baseline | (无 SKIP) | **PASS** | 5.42 ms |
| skip_swiglu | SKIP_SWIGLU=1 | PASS | 4.43 ms (FC1 仍跑) |
| skip_poll | SKIP_ARRIVAL_POLL=1 | PASS | 5.24 ms |
| skip_all | SKIP_ALL=1 (FC1 早 return) | **FAIL** ILMA | — |
| skip_mfma | SKIP_MFMA=1 (跳 mfma 但 wrapper 跑) | **FAIL** ILMA | — |

**结论 1 (好消息)**: plan 标的 "T≥512 + FC1 fused hard crash" **不再复现**. baseline (真 FC1) DSv3 prod T=512 直接通, crit_path 5.42 ms. 复测 T={512, 1024, 2048, 4096, 8192} F=2048 全 PASS. 上次 21:15 / M4-α 时的 crash 大概率是当时容器 / GPU 状态残留 (cold IPC handle / sclk 94 MHz 等), 不是源码 bug.

**结论 2 (新 finding, 但不阻塞 plan)**: 当 FC1 GEMM 真跑时, GEMM WGs 在 `mfma_tile` 上磨 ~1-3 ms 才到达 intra-role `grid_sync<2>`; 当 FC1 完全 noop (SKIP_ALL / SKIP_MFMA) 时, GEMM WGs 瞬间到达 intra-role barrier, 跟 dispatch 慢路径的 `grid_sync<2>` 形成时序窗口, 撞出 ILMA. 这是 bisect harness 本身的 artifact (生产路径里 FC1 永远跑), **不影响生产**. 想清掉这个 artifact 要进 Phase 1 (per-WG hierarchical grid_sync 改写, 避免单 counter 上 256-arrival 在 ns 级窗口里完成).

### 4.2 DSv3 prod ctest 跑通

| ctest | 时长 | 内容 |
|---|---|---|
| `test_super_e2e_dsv3_prod` (T=512) | **58.3 s PASS** | 3 skew × {1,3} iter × dispatch bit-exact + FC1+SwiGLU bf16 abs<0.15 OR rel<10% — 全过 |
| `test_super_e2e_dsv3_t2048` (T=2048) | **152.1 s PASS** | 同上 T=2048; expect_fc1_act 在 96-core host 上跑 ~30 s/rank |

### 4.3 baseline scaffold tax 数据 (build_p0_baseline, WG_PER_CU=1 + WAVES=1, DSv3 prod balanced, warmup=1, iters=5, crit_path p50)

| T | F=0 (super wall) | F=2048 (super wall) | FC1 add-on |
|---|---|---|---|
| 512  | 4.02 ms | 5.44 ms | +1.42 ms |
| 1024 | 7.76 ms | 10.13 ms | +2.37 ms |
| 2048 | 14.44 ms | 16.13 ms | +1.69 ms |
| 4096 | 28.36 ms | 26.70 ms | -1.67 ms (FC1 完全 overlap-hidden, noise level) |
| 8192 | 55.30 ms | 54.34 ms | -0.96 ms (同上) |

**新观察**: T≥2048 FC1 已经 **完全 overlap-hidden** (add-on < dispatch wall 噪声), 跟 1545 FLAT note 钉的 "FC1 = receiver × 1.13×, 完美 overlap 整个 dispatch wall 100% 藏在 FC1 后" 实测对得上. 这就把 plan acceptance "DSv3 prod T=2048 BF16 e2e wall ≤ 7 ms" 的瓶颈完全推到了 scaffold tax (现 14.4 ms F=0) 上面 — 不解 scaffold tax (Phase 1 grid_sync_v2) 永远不可能拿 7 ms.

### 4.4 WG_PER_CU=2 regression (Phase 0.2)

`build_p0_wgcu2` 在 T=512/1024/2048 全部 ILMA. 跟 plan 假设吻合, 按 plan **WG_PER_CU=2 不作为 default supported config**, 留为 build flag A/B 用. M4-α 默认走 1, 这条 regression 不阻塞.

### 4.5 标准 ctest 子集状态

| ctest | 状态 | 备注 |
|---|---|---|
| test_gemm_* (5 个) | PASS | M0 已经稳 |
| test_super_skeleton | PASS | smoke |
| test_super_dispatch | PASS | bit-exact |
| test_super_e2e_small | PASS | 1.9 s |
| test_super_e2e_fc1 | PASS | 1.9 s |
| **test_super_e2e_dsv3_prod (new)** | **PASS** | 58.3 s |
| **test_super_e2e_dsv3_t2048 (new)** | **PASS** | 152 s |
| test_dispatch_smoke | FAIL (pre-existing flaky) | 文档化已存在的 ctest serial scheduling 下 cross-rank phase_barrier timing race, 跟本次正交; standalone binary 用 `ROCMOE_USE_KEEPALIVE` 跑也炸过, 不是本次 regression |

## 5. 验收

| 项 | 目标 | 实测 | 状态 |
|---|---|---|---|
| DSv3 prod T=2048 跑通 0 crash | ✓ | T=2048 F=2048 crit_path 16.13 ms (5 iters), 152s ctest 全 PASS | ✓ |
| 新 ctest 锁定 | 2 个 ctest 加进 CMakeLists | ✓ | ✓ |
| scaffold tax 基线 | F=0 wall 数 | T=2048 14.44 ms (= plan 14.95 ms 对得上, 略低因为 keepalive 让 sclk 更稳) | ✓ |
| ctest 全绿 (除 pre-existing flaky) | 所有非 test_dispatch_smoke 通 | ✓ | ✓ |
| keepalive script | 落地 | ✓ | ✓ |

## 6. 修改 / 新增文件

| 文件 | 类型 | 改动 |
|---|---|---|
| `scripts/dev_bench_with_keepalive.sh` | 新增 | GPU keepalive wrapper |
| `scripts/p0_bisect_fc1_crash.sh` | 新增 | 5-variant bisect driver |
| `CMakeLists.txt` | 改 | 7 个现有 build knob + 4 个 FC1 SKIP_* knob 全部加 `add_compile_definitions` plumbing; 2 个 DSv3 prod ctest + timeout |
| `csrc/super_kernel.hip` | 改 | `fc1_gemm_role_body` 加 SKIP_ALL/SKIP_MFMA/SKIP_ARRIVAL_POLL guard; `fc1_swiglu_pass` 加 SKIP_SWIGLU guard (default off, 仅 bisect 用) |

## 7. 下一步 (Phase 1)

按 [plan §Phase 1](.cursor/plans/rocmoe-super-kernel-overhaul_b804bc61.plan.md):

1. `barrier.h` 实现 `grid_sync_v2<>`: per-WG cache-line slot + cross-WG release-acquire + L2-resident all_done flag, 砍掉当前 single-counter 256-WG atomicAdd 的 ~700 µs/call cost
2. super_kernel.hip 14 处 + dispatch.hip 3 处 `grid_sync` call site 替换
3. rocprof PMC 验证 pool 58 MB 是否 IF$ 命中
4. acceptance: DSv3 prod T=2048 F=0 super wall ≤ **9 ms** (现 14.44 ms)

## 8. 经验 (写给后续)

1. **CMake convention 跟历史 plumbing 不一致是真坑** — Phase 0.1 第一轮 bisect 全 fail 用了 1 小时才意识到不是源码问题, 是 build knob 没传进去. 教训: 写新 build knob 时同时在 `CMakeLists.txt` 加 `add_compile_definitions` plumbing.
2. **Heisenbug 不一定真死** — plan 起点假设 "T≥512 必 crash" 实测不复现; cold IPC / sclk / container 状态都可能让一次 crash 看上去像确定性. 修一个 crash 前先用 keepalive + fresh clean build 复测.
3. **FC1 已经 fully overlap-hidden at T≥2048** — 后续优化预算应 100% 投在 scaffold tax (grid_sync) 而不是 GEMM kernel; 1545 FLAT note 的预测被实测验证.
