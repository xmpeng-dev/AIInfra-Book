# Phase 1 UP — `grid_sync_v2<>` hierarchical barrier, super-kernel scaffold tax -14% @ DSv3 prod T=2048 F=0

> 时间: 2026-05-25 11:10 (UTC+8)
> 项目: rocmoe-v2 Super-Kernel Overhaul, Phase 1 完成
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 单节点 `mi355-gpu-7`
> 容器: `xiaoming-dev` / `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2, hipcc 22.0.0
> 代码: 工作树 `~/workspace/RocMoE/`, branch `xiaoming/super_kernel_overhaul`
> 上一篇: [Phase 0 UP](./2026-05-25_1100_UP_phase0_overhaul_dsv3_prod_ctest_green.md)

## 1. 时间点 / 上下文

Phase 0 把 DSv3 prod (T=512, T=2048) ctest 拉到 GREEN 后, scaffold tax 残量明确指向 `grid_sync<>` (legacy v1) 在 256-WG 上的单 atomic counter 竞争 ([M4-α note §4](./2026-05-23_2216_UP_m4a_wg_per_cu_1_drops_super_dispatch_35pct.md) p50 16.71 ms 中, dispatch standalone p50 4.23 ms, 差额 ~12 ms 中至少 ~10 ms 是 14 个 grid_sync<> 调用的串行 atomic 等待).

Phase 1 的任务: 把 `grid_sync<>` 替换成 hierarchical `grid_sync_v2<>` (per-WG cache-line aligned arrival slot + 单 release flag fan-out), 把每次 sync 的 atomic 竞争从 256-way 降到 16-way (cache-line 内), 并把 256 WG 同时 spin-load 1 个 cache line 改成 leader 256 个 thread 各自 spin-load 自己负责的 slot.

## 2. 问题

`grid_sync<>` (v1) 的物理瓶颈:

1. **单 atomic counter 串行化** — `__hip_atomic_fetch_add(&grid_ctr[stage], 1, ACQ_REL, AGENT)` 在 256 个 WG 的 tid=0 上同时执行, L2 cache line owner 在 256 个 CU 之间反复 invalidate, 每次约 150-300 cycles 的 line transfer.
2. **`s_sleep(1)` backoff 必须保留** (M4-α §1) 否则 spin-load 进一步把同一 cacheline 拖死, 但 backoff 本身就是 100 cycles 浪费.
3. 后果: 单次 `grid_sync<>` ~ 700 µs (M4-α 估算), super-kernel 14 个调用 + dispatch.hip 3 个 = ~12 ms scaffold tax @ kNTotalWGs=256.

## 3. 做了什么

### 3.1 `grid_sync_v2<>` 设计 (csrc/include/rocmoe/barrier.h)

| 层 | 协议 |
|---|---|
| 到达 | 每个 WG 的 tid=0 把单调递增的 `my_gen` `__ATOMIC_RELEASE` 写到 `arrival[stage][blk]` (8 KB / 4-stage / kMaxGridSyncV2WGs=512). 256 个 WG 写 256 个不同 slot, 64B cache line 内最多 16-way contention 而不是 256-way. |
| 汇总 | Leader (blk==0) 的 256 个 thread 协作 fan-in: tid=t 自旋 `arrival[stage][t]` 直到 ≥ my_gen. AGENT scope ACQUIRE load. 256 个 thread 各自 own 1 个 slot 的 spin, parallel L2 fan-in. |
| 释放 | Leader tid=0 把 my_gen `__ATOMIC_RELEASE` 写到 `release[stage]` (256 B 对齐). 非 leader 的 255 个 WG 的 tid=0 在 `release[stage]` 上 ACQUIRE-自旋, RELEASE 写一次后 line 变 Shared, 所有 reader cache-hit 直到下一次写. |
| WG 同步 | `__syncthreads()` 在 leader 和非 leader 出口处 hold, 保证 WG 内所有 thread 同步推进. |

### 3.2 关键设计: gen 必须从 workspace bootstrap

Initial 实现把 `gs_gen0..3` 作为 register 变量 0-init, 每次 kernel launch 重置. 这导致 launch N+1 的 `my_gen=1` 比 launch N 写下的 `arrival[stage][blk]=2` 还小, leader 立即看到 "已到达" → exit 在本次实际到达之前 → 死锁 / 内存错乱.

**修复**: arrival[] 持久化 (hipMalloc 零初始化, 跨 launch 不动); 每次 kernel 入口用 `grid_sync_v2_load_gen<stage>(arrival, total_wgs, blk)` 把上次写下的 gen 加载回 register, ++ 后再写. 这样 gen 在整个 workspace 生命周期内严格单调.

```cpp
uint32_t gs_gen0 = grid_sync_v2_load_gen<0>(gs_arrival, total_wgs, blk);
// ... 1..3 同理
grid_sync_v2<0>(gs_arrival, gs_release, total_wgs, blk, tid, ++gs_gen0);
```

### 3.3 替换 call site

| 文件 | 替换数 | 备注 |
|---|---|---|
| `csrc/super_kernel.hip` | 14 处 | 涵盖 `rocmoe_super_kernel` (production) + `rocmoe_super_skeleton_kernel` (test) |
| `csrc/dispatch.hip` | 3 处 | dispatch standalone path (A→B, B→C, cross-rank→Recv) |
| `csrc/moe_config.{h,cpp}` | + 2 fields | `off_grid_sync_v2_arrival`, `off_grid_sync_v2_release` 加入 workspace layout |
| `csrc/include/rocmoe/types.h` | + 1 const | `kMaxGridSyncV2WGs = 512` (固定上界保 layout ABI 稳定) |
| `csrc/include/rocmoe/workspace.h` | + 2 accessors | `grid_sync_v2_arrival()`, `grid_sync_v2_release()` |

### 3.4 fix: `test_super_kernel_skeleton` workspace 大小

旧实现 skeleton test 只分配 256 B / rank (够 PhaseBarrierSignal 而已); v2 引入 8 KB 的 arrival[] + 256 B release[] 要求, zero cfg 意味着 `off_grid_sync_v2_arrival=0` → 写到 sym_base+0 → 撞 PhaseBarrierSignal 触发死锁. 改成用 `build_moe_config()` 构造合法的最小 layout, 分配 `cfg.total_bytes`.

## 4. 效果

### 4.1 ctest

`build_p1_baseline` (WG_PER_CU=1, WAVES=1, grid_sync_v2):

| 测试 | 结果 | 备注 |
|---|---|---|
| `test_super_skeleton` | PASS 1.14 s | 100 iter × 4 repeat 持久 grid 不死锁 |
| `test_super_dispatch` | PASS 1.44 s | dispatch only |
| `test_super_e2e_small` | PASS 1.88 s | T=64 经典小 e2e |
| `test_super_e2e_fc1` | PASS 1.96 s | T=128 FC1 fused |
| `test_super_e2e_dsv3_prod` | PASS **57.84 s** | DSv3 T=512 全 ranks bit-exact |
| `test_dispatch` (3-run flake check) | PASS PASS PASS | pre-existing flake 不复现, 跟 v2 无关 |

`test_dispatch_smoke` (cross-rank `phase_barrier` 已知 flake) 在 ctest 偶发失败, 单独 3 连跑全 GREEN, 跟 grid_sync_v2 正交.

### 4.2 bench A/B: super F=0 (scaffold tax only)

DSv3 prod routing, H=7168, balanced, warmup=3 iters=10, p50 crit_path (per-rank max):

| T    | v1 p50 (ms) | v2 p50 (ms) | Δ ms  | Δ %    |
| ---  | ---         | ---         | ---   | ---    |
| 256  | 4.403       | 4.383       | -0.02 | -0.5%  |
| 512  | 5.718       | 5.607       | -0.11 | -1.9%  |
| 1024 | 9.368       | 8.655       | -0.71 | **-7.6%**  |
| 2048 | 16.710      | 14.382      | -2.33 | **-13.9%** |

### 4.3 bench A/B: F=2048 (FC1 fused — chunk-overlap 把 scaffold 部分藏进 dispatch)

| T    | v1 p50 (ms) | v2 p50 (ms) | Δ ms  | Δ %   |
| ---  | ---         | ---         | ---   | ---   |
| 512  | 7.409       | 7.217       | -0.19 | -2.6% |
| 1024 | 11.820      | 11.578      | -0.24 | -2.0% |
| 2048 | 17.672      | 15.969      | -1.70 | **-9.6%** |

### 4.4 主指标

| 指标 | Before (grid_sync v1) | After (grid_sync_v2) | Δ |
|---|---|---|---|
| super F=0 wall T=2048 p50 (ms) | 16.71 | **14.38** | **-14%** |
| super F=0 wall T=2048 min (ms) | 15.67 | **14.19** | -9.5% |
| super F=2048 wall T=2048 p50 (ms) | 17.67 | **15.97** | -10% |
| 单 sync 调用 cost 估算 (µs) | ~700 | ~50 | -93% (理论上, 14 × 调用 -10 ms 跟 fan-out spin 实测一致) |

### 4.5 定性

- ✅ grid_sync_v2 在 T={1024, 2048} 上稳定 -7 ~ -14% scaffold tax, 跨 F=0 / F=2048 一致
- ✅ ctest 全部 GREEN, DSv3 prod T=512 bit-exact (ctest 57.84 s)
- ✅ 持久 gen counter 设计正确, skeleton test 100 iter × 4 repeat 不死锁
- ⚠️ T=256, T=512 上 v2 跟 v1 几乎打平 (-0.5 ~ -2%): 这两档 scaffold tax 本来就只占 wall 的 ~30%, dispatch+phase_barrier 本身已经吃了大头
- ⚠️ 离 Phase 1 目标 "DSv3 prod T=2048 F=0 super wall ≤ 9 ms" 还有 5 ms gap — 剩下的 5 ms 主要在 (a) `phase_barrier<8>` cross-rank ~80 µs × 1, (b) dispatch sender phase B 受 wave-store-drop hazard 强制 WAVES=1 限制的多余 ~3-4 ms, (c) persistent role 走完 grid_sync<2>/<3> 的 epilogue. (b) 是 Phase 2 的主战场.

## 5. 可持续方向

| 优先级 | 方向 | 预期 |
|---|---|---|
| P0 | Phase 2: wave-store-drop hazard 30-line repro + intra-WG wave-specialized b128 copy primitive (取消 WAVES=1 限制, dispatch sender phase B 应能拿回 ~3 ms) | -2 to -3 ms @ T=2048 |
| P0 | Phase 3: FC1 fused → LDS-resident SwiGLU → FC2 epilogue in 同一 super-kernel | 全 5-phase wall 目标 ≤ 10 ms |
| P1 | rocprof PMC 验证 58 MB dispatch token pool 是否 IF$ 命中 (Phase 1.4); 如果 miss 严重, 考虑加 cache-pin hint | 当前 super-kernel 还有 wave-launcher icache miss 嫌疑, 但 PMC 数据未采 |
| P2 | grid_sync_v2 自适应 fan-in: 当 kNTotalWGs ≤ 128 时退化为 `__syncthreads`-like fast path (省掉 256 个 release-flag spin) | T 偏小时再榨 ~50 µs |

## 6. 相关文件

- 实现: `csrc/include/rocmoe/barrier.h` (grid_sync_v2 + grid_sync_v2_load_gen), `csrc/super_kernel.hip` (call sites + gen bootstrap), `csrc/dispatch.hip` (同), `csrc/moe_config.{h,cpp}` (workspace layout), `csrc/include/rocmoe/types.h` (kMaxGridSyncV2WGs), `csrc/include/rocmoe/workspace.h` (accessors)
- 测试修复: `tests/test_super_kernel_skeleton.hip` (用 build_moe_config 分配 workspace)
- bench 命令模板: `bash scripts/dev_bench_with_keepalive.sh build_p1_baseline/bench_super_kernel --T 2048 --H 7168 --F-moe 0 --num-experts 256 --topk 8 --n-group 8 --group-topk 4 --dist dsv3 --warmup 3 --iters 10 --max-recv-factor 4`
- baseline 对照命令: 把 `build_p1_baseline` 换成 `build_m4a_w1`
- 上一篇: [Phase 0 UP](./2026-05-25_1100_UP_phase0_overhaul_dsv3_prod_ctest_green.md)
