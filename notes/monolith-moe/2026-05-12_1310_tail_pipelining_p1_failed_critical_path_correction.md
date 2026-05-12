# P1 Tail-Pipelining 失败 + Critical-Path 误判修正

> 时间: 2026-05-12 13:10 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, XGMI 全互联), node `mi355-gpu-26`
> 容器: xiaoming-dev (Podman)
> 软件: ROCm 7.2 / hipcc, gfx950 codegen
> 代码: `csrc/fused_moe_super_kernel.hip` @ `8d00b73` (worktree dirty, P1 changes 已 revert，仅保留 README/PERF 文档)

## 1. 时间点 / 上下文

- 上一次相关进展：P0「Adaptive small-tile + flat tile dispatch」把 DSV3 SPARSE
  从 14.04 → 8.80 ms（1.60×），TILE-FIT 5.17 → 5.07 ms。`compute_barrier_1`
  从 8.55 ms 砍到 0.29 ms，compute WG 利用率从 ~20% 升到 ~80%。
- 触发本次工作：P0 后第一份 profile 显示 **tail WG `gather_combine_phase`
  kernel_total = 7.10 ms / 8.80 ms wall = 79 %**，看起来是新瓶颈，于是尝试
  并行化 tail。

## 2. 问题

加速 `gather_combine_phase`，把 DSV3 SPARSE 从 8.80 ms 再压一截。

- 现状：DSV3 SPARSE 8.80 ms / 327.9 TFLOPS（P0 收尾）
- 目标：DSV3 SPARSE ≤ 7.8 ms（−1 ms 估算，假设 tail 真在临界路径）
- 卡点：tail phase 是 16 个 WG 顺序处理 (src, e) 对，每对内部按 `wait_flag →
  combine`；初看像 inherent serial。

## 3. 做了什么

### 尝试 A：per-pair round-robin（每个 WG 拿一段 pair 子集）

把 `gather_combine_phase` 改成：

```cpp
// before: for (src ∈ ranks) for (e ∈ experts) for (h ∈ H)
// after:
int total_pairs = NUM_GPUS * epg;
for (int pair_id = wg_local_id; pair_id < total_pairs; pair_id += num_tail_wgs) {
    int src = pair_id / epg;
    int e   = pair_id % epg;
    wait_flag(src, e);   // 每个 WG 自己等自己的 pair
    combine(src, e);     // 不再跨 WG 同步
}
```

去掉跨 WG fence；让 16 个 tail WG 完全独立推进各自 pair 集。

### 尝试 B：slot × h-chunk hybrid

退回到「per-pair 串行」，但把每对 `combine` 的 H 维拆给所有 16 个 tail WG，
意图缩短 per-pair 处理时间：

```cpp
for (pair) {
    if (wg_local_id == 0) wait_flag(pair);
    cross_wg_barrier();   // 所有 tail WG 等 wait_flag
    for (h ∈ wg_chunk_of_H) combine(...);
    cross_wg_barrier();
}
```

| # | 动作 | 文件 | 结果 |
|---|---|---|---|
| A1 | per-pair round-robin | `csrc/fused_moe_super_kernel.hip::gather_combine_phase` | 编译 + smoke OK |
| A2 | Bench DSV3 SPARSE + TILE-FIT | `bash` via `mi355-gpu-26` `xiaoming-dev` | **DSV3 8.80 → 9.98 ms（regress），TILE-FIT 5.07 → 8.66 ms** |
| A3 | `MOE_PROFILE=1` 取 ROC trace | `benchmarks/results/p1_tail_profile.txt` | 见下表 |
| B1 | 改成 slot × h-chunk hybrid | 同文件 | smoke OK |
| B2 | Bench 同上 | `benchmarks/results/p1_hybrid_bench.txt` | DSV3 8.85 ms / TILE-FIT 5.05 ms（neutral） |
| C  | **整体 revert tail 改动**，恢复 P0 原版 | 同文件 | DSV3 8.80 ms / TILE-FIT 5.07 ms 复现 |

### 关键 profile（per-pair round-robin DSV3 SPARSE，单位 ms）

| 阶段 | P0 原版 | A1（regression） |
|---|---:|---:|
| dispatch_src_ready_wait | 0.92 | 0.91 |
| fc1_tiles               | 2.95 | 2.93 |
| fc2_swiglu_tiles        | 3.16 | 3.11 |
| copy_to_combine         | 0.32 | 0.34 |
| **tail wait + combine** | **1.31** | **2.94** |
| **wall**                | **8.80** | **9.98** |

P1-A 的 tail 反而 **多花 1.6 ms**。`rocprof` 的 HBM atomic 计数：A1 比 P0
高 ~3 ×。`wait_flag` 用的是 `atomicAdd_system` polling，原本 16 个 WG 顺序
撞 atomic 单元，本来就排队；现在 16 个 WG 同时对不同地址 spin，HBM atomic
unit 直接被打爆，每次 `__atomic_load(__ATOMIC_ACQUIRE)` 的 latency 翻倍。

### 关键洞察：tail 根本不在 critical path

把 P0 profile 再读一遍：

```
[per-WG kernel_total, DSV3 SPARSE]
comm WG (pack+scatter):     2.13 ms
compute WG:                 9.04 ms   ← 最长
tail WG (gather+combine):   7.10 ms   ← 比 compute 短 1.94 ms
wall:                       8.80 ms   ≈ compute kernel_total
```

三类 WG 在不同 CU 上 **完全并行**，wall = `max(kernel_total)`。compute 的
9.04 ms 已经超过 tail 的 7.10 ms，说明 **wall 由 compute 决定**，tail 现在
就算压到 0 ms 也只能省 wall − compute_kernel_total 的差，根本不是瓶颈。

之前的 "79 % tail time" 是把 `tail kernel_total / wall` 误读成 "critical path
比例"，忽略了三类 WG 是并行的。

## 4. 效果

| 指标 | P0 原版 | P1-A | P1-B | 最终（revert 回 P0） |
|---|---:|---:|---:|---:|
| DSV3 SPARSE wall (ms) | 8.80 | 9.98 (**+13 %**) | 8.85 | 8.80 |
| TILE-FIT wall (ms)    | 5.07 | 8.66 (**+71 %**) | 5.05 | 5.07 |
| DSV3 SPARSE TFLOPS    | 327.9 | 276.0 | 326.0 | 327.9 |
| HBM atomic count      | 1× | ~3× | ~2× | 1× |

定性观察：

- ❌ tail 并行化两种方案均无收益或回退。
- ✅ **临界路径搞清楚了**：wall ≈ compute_kernel_total，compute 才是下一步主战场。
- ⚠️ tail WG 上 `atomicAdd_system` polling 的扇出能力被 HBM atomic 单元限制
  （上限大约 16 路 spinner）。后续若真要并行 tail，得换成 LDS-cache 的 flag
  或者 src/e 局部 reduce。
- ⚠️ revert 后再跑一次完整 P0 数字，DSV3 8.80 / TILE-FIT 5.07 完美复现，
  验证 `git checkout -- csrc/fused_moe_super_kernel.hip` 没漏改。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| P0 | 把 compute kernel_total 从 9.04 ms 压到 ≤ 7.0 ms（**真正的临界路径**） | DSV3 −1.8 ms | 已立 P2「Batched (e, src, mi, ni) compute」 |
| P1 | TILE-FIT `dispatch_src_ready_wait` 加速（scatter publish 早一点出 src=0） | TILE-FIT −0.5~1.5 ms | 改 comm WG schedule |
| P2 | 再回头看 tail：若 compute 真比 tail 还快，可重启 tail 并行化，**但要先把 atomic spin 换成 LDS 或拆 flag 地址** | DSV3 ≤ 0.5 ms | 当前非瓶颈 |

## 关键教训

1. **多 WG 并行 kernel 的 wall = max(per-WG kernel_total)，不是 sum**。
   "某 phase 占 wall 79 %" 这种说法要看是不是真在 critical path 上。
2. **atomic-based polling 的并发度有硬上限**（MI355X 上 spin source 大概 16
   个 / atomic word），不要无脑提高并行 WG 数量。
3. 失败实验本身有价值：明确了下一步该投在 compute、不是 tail，省了后面继续
   钻 tail 的人力。

## 相关文件

- 代码（已 revert）：`csrc/fused_moe_super_kernel.hip::gather_combine_phase`
- Bench 日志：
  - `benchmarks/results/p1_tail_bench.txt`（per-pair round-robin）
  - `benchmarks/results/p1_tail_profile.txt`（per-pair round-robin profile）
  - `benchmarks/results/p1_hybrid_bench.txt`（slot × h-chunk hybrid）
- 上一篇 note：[`2026-05-08_ck_implementation_deep_dive.md`](./2026-05-08_ck_implementation_deep_dive.md)
- 同日 P2 成功记录：[`2026-05-12_1335_batched_compute_p2_single_phase.md`](./2026-05-12_1335_batched_compute_p2_single_phase.md)
