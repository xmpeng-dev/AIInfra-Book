# P2 Batched (e, src, mi, ni) Compute + 单相 FC1/FC2 + Barrier 合并

> 时间: 2026-05-12 13:35 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, XGMI 全互联), node `mi355-gpu-26`
> 容器: xiaoming-dev (Podman)
> 软件: ROCm 7.2 / hipcc, gfx950 codegen
> 代码: `csrc/fused_moe_super_kernel.hip` + `tests/smoke_super_kernel.hip` @ worktree on `8d00b73` (uncommitted)

## 1. 时间点 / 上下文

- 上一篇 note：[`2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md`](./2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md)，
  确认 wall = max(per-WG kernel_total) = **compute_kernel_total (9.04 ms)**，
  tail 不在 critical path。
- 触发本次工作：P1 复盘后，compute 9.04 ms 中明显有冗余：
  - 8 个 src 的 FC1 串行，每个 src 一段 `wait_dispatch → fc1 → barrier →
    fc2 → barrier → copy → barrier`，**24 个 cross-WG barrier**，barrier 总
    开销 DSV3 1.34 ms / TILE-FIT 1.37 ms。
  - 同一个 expert 的 `w1_e` / `w2_e` 在 8 次 per-src 循环中被反复 HBM-read，
    L2/L3 没有跨 src 复用。

## 2. 问题

把 compute kernel_total 从 9.04 ms 压到 ≤ 7.5 ms（target wall ≤ 8.0 ms）。

- 现状：DSV3 SPARSE wall 8.80 ms / 327.9 TFLOPS，TILE-FIT 5.07 ms / 162.5 TFLOPS
- 目标：DSV3 SPARSE ≤ 8.0 ms（首次反超 PyTorch+RCCL 基线 8.466 ms）；TILE-FIT ≤ 4.5 ms
- 卡点：8 个 per-src 阶段串行 + 24 cross-WG barrier + weight HBM 重复读

## 3. 做了什么

### 设计：把 `expert_compute_phase` 从 `for src ∈ ranks` 拆掉

旧版（chunked，per-src 串行）：

```cpp
for (int src = 0; src < NUM_GPUS; ++src) {
    wait_dispatch_src_ready(src);
    fc1_phase(src, ...);          // small + flat tile dispatch
    compute_phase_barrier();      // ← FC1→FC2
    fc2_swiglu_phase(src, ...);
    compute_phase_barrier();      // ← FC2→copy
    copy_fc2_to_combine(src, ...);
    compute_phase_barrier();      // ← copy→signal
    signal_combine_ready(src);
}                                  // 8 × 3 = 24 个 cross-WG barrier
```

新版（batched，单相）：

```cpp
// 1) 一次性等齐 8 个 dispatch_src_ready，再 publish 全部 expert offsets
for (src) wait_dispatch_src_ready(src);
for (src) publish_combine_expert_offsets(src);

// 2) 建全局 (src, e) pair tile descriptor，按 (e, src, mi, ni) 排
//    pair_id = e * NUM_GPUS + src，prefix sum 出 tile 全局偏移
build_global_pair_descriptor(/*FC1 shape: M=T_e, N=2F, K=H*/);

// 3) FC1：所有 compute WG 一起跑 total_tiles 个 tile
for (tile_id = wg_global_id; tile_id < total_tiles; tile_id += num_compute_wgs) {
    auto (src, e, mi, ni) = decode(tile_id, pair_tile_offset[]);
    gemm_tile_dispatch(src, e, mi, ni);  // 小 expert 用 32×128 tile
}
compute_phase_barrier();          // ← FC1→FC2，唯一一次

// 4) FC2：同样的 (e, src, mi, ni) 排序，但 N=H
rebuild_global_pair_descriptor(/*FC2 shape*/);
for (tile_id ...) gemm_tile_dispatch(...);
compute_phase_barrier();          // ← FC2→copy，唯一一次

// 5) per-src 顺序 copy fc2_scratch → peer combine_results
for (src) copy_fc2_to_combine(src);
compute_phase_barrier();          // ← copy→signal，唯一一次

// 6) 一次 fence + 一次性 signal 8 × epg 个 combine_expert_ready
__threadfence_system();
if (wg_lane == 0) for (src) for (e) signal_combine_ready(src, e);
```

**总 cross-WG barrier 从 24 缩到 3。**

### Scratch 扩容

旧版 `scratch_fc1 = max_recv × 2F`、`scratch_fc2 = max_recv × H`（per-src 复用）。
batched 后 8 个 src 的 FC1/FC2 output 必须共存：

```cpp
// csrc/fused_moe_super_kernel.hip MoeScratchSizes
s.fc1_bytes = (size_t)NUM_GPUS * max_recv_tokens * 2 * ffn_hidden_size * sizeof(bf16_t);
s.fc2_bytes = (size_t)NUM_GPUS * max_recv_tokens * hidden_size       * sizeof(bf16_t);
```

DSV3 SPARSE：`scratch_fc1` 4 MB → 32 MB，`scratch_fc2` 1.4 MB → 11 MB。
TILE-FIT：fc1 0.5 MB → 4 MB，fc2 0.25 MB → 2 MB。Persistent buffer，无 alloc 开销。
`tests/smoke_super_kernel.hip` 里的 `alloc_on_rank` 同步放大 8×。

### Global pair descriptor (LDS)

新增到 `GemmLdsLayout`：

```cpp
static constexpr int MAX_PAIRS = NUM_GPUS * MAX_EXPERTS_PER_GPU; // 8 × 64 = 512
int pair_addr_offset[MAX_PAIRS];   // src * max_recv + e_start_in_src
int pair_T_e        [MAX_PAIRS];   // per-(src, e) token count
int pair_num_n      [MAX_PAIRS];   // num_n_tiles for current FC shape
int pair_kind       [MAX_PAIRS];   // 0 = small (32×128 1×4), 1 = default
int pair_tile_offset[MAX_PAIRS + 1]; // exclusive prefix sum
int pair_total_tiles;
```

LDS 用量 ~10 KB（512 × 5 × 4B），gfx950 单 WG 64 KB LDS 预算还够，
occupancy clamp 不受影响（实测仍为 1 WG/CU）。

### 工作步骤

| # | 动作 | 文件 / 命令 | 备注 |
|---|---|---|---|
| 1 | 扩 `MoeScratchSizes` × NUM_GPUS | `csrc/fused_moe_super_kernel.hip` L2077–2097 | |
| 2 | 同步 smoke test alloc | `tests/smoke_super_kernel.hip` L114–115 | |
| 3 | 新增 `GemmLdsLayout::pair_*` | 同 .hip L444–454 | |
| 4 | 重写 `expert_compute_phase` | 同 .hip L1397–1733 | 单相 batched dispatch |
| 5 | smoke + correctness | `tests/test_super_kernel_correctness` 50/50 deterministic | PASS |
| 6 | ratio sweep 0.10~0.30 | `benchmarks/results/p2_ratio_sweep.txt` | 选 ratio=0.18 (DSV3) / 0.15 (TILE-FIT) |
| 7 | 主 bench | `benchmarks/results/p2_l2reuse_bench.txt` | DSV3 7.95 / TILE-FIT 4.31 |
| 8 | `MOE_PROFILE=1` profile | （见下表） | 验证 barrier 收益 |

## 4. 效果

### 主指标

| 指标 | P0（chunked） | P2（batched） | Δ |
|---|---:|---:|---:|
| **DSV3 SPARSE wall (ms)**   | 8.80 | **7.95** | **−0.85（−9.7 %）** |
| DSV3 SPARSE 有效 TFLOPS     | 327.9 | **363.1** | +10.7 % |
| DSV3 SPARSE vs PyTorch+RCCL（8.466 ms / 340.9 TFLOPS） | 0.96× | **1.07×（首次反超）** | — |
| **TILE-FIT wall (ms)**      | 5.07 | **4.31** | **−0.76（−15 %）** |
| TILE-FIT 有效 TFLOPS        | 162.5 | **191.4** | +17.8 % |
| `test_super_kernel_correctness` | 50/50 PASS | 50/50 PASS | — |

### Profile 分解（`MOE_PROFILE=1`，单位 ms）

| 阶段 | DSV3 P0 | DSV3 P2 | TILE-FIT P0 | TILE-FIT P2 |
|---|---:|---:|---:|---:|
| dispatch_src_ready_wait | 0.92 | 1.14 | 1.85 | 2.17 |
| fc1_tiles               | 2.95 | 2.87 | 0.85 | 0.80 |
| fc2_swiglu_tiles        | 3.16 | 2.99 | 0.93 | 0.82 |
| compute_barrier_1       | 0.29 | 0.44 | 0.96 | 0.08 |
| compute_barrier_2       | 0.57 | 0.07 | 0.09 | 0.08 |
| compute_barrier_3       | 0.48 | 0.11 | 0.32 | 0.08 |
| barrier 合计            | **1.34** | **0.62** | **1.37** | **0.24** |
| copy_to_combine         | 0.32 | 0.43 | 0.18 | 0.28 |
| **总 wall**             | **8.80** | **7.95** | **5.07** | **4.31** |

### 定性观察

- ✅ DSV3 **首次反超 PyTorch+RCCL 基线**（1.07×），项目主战场目标过线。
- ✅ Barrier 合并实测收益：DSV3 −0.72 ms，TILE-FIT −1.13 ms。
- ⚠️ **L2/L3 weight 复用没像预想那样减少 HBM 流量**：FC1/FC2 时间基本不变
  （DSV3 fc1 2.95 → 2.87 ms，−2.7 %，可视为噪声）。原因推测：
  - DSV3 单 expert `w1_e` = `H × 2F × 2B = 7168 × 4096 × 2B ≈ 58 MB`，
    MI355X 每 XCD 上 L2 32 MB + AID L3 256 MB，但 154 个 compute WG 同时
    对 8 个 src 推进，cache line 被竞争性替换。
  - tile dispatch 是 `(e, src, mi, ni)`：理论上 mi/ni 在不同 src 间错位时，
    一个 expert 的 w1_e block 应该能在 src=0 加载、src=1~7 命中。但实际
    workload 中每个 expert 的 T_e 只有 ~16 token / src，single src 的 FC1
    可能只用 1 个 mi tile，导致 w1_e 各 K-step 还没被 src=1 复用就被新 src
    的 packing 数据冲掉。
  - 结论：**P2 的真实胜利是 barrier 合并，不是 cache 复用**。
- ⚠️ `dispatch_src_ready_wait` 变长（DSV3 +0.22 ms / TILE-FIT +0.32 ms）：
  batched 模式失去了「等到一个 src 就开始算 FC1，剩下 7 个 src 在 background
  scatter」的 overlap，整个 wait 阶段被串成一段。TILE-FIT 上 wait 现在占
  2.17 ms / 4.31 ms = **50 % wall**，已是新瓶颈。
- ❌ L2 reuse 假设证伪。下次想真复用 weight，需要改成"每 expert 把 8 个
  src 的 T_e 沿 M 拼起来一起跑"（即把 batched 维度从 (e, src, mi, ni) 改成
  (e, M_concat, ni)），这才是真正的 batched-by-M。本次没做，留作 P1。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| P0 | **TILE-FIT scatter publish 加速**（让 compute 早一点拿到 src=0 ready） | TILE-FIT −0.5~1.5 ms（目标 ≤ 3 ms） | comm WG schedule 改造，需要 per-(rank, e) bump 或全局 ready 合并 |
| P0 | **DSV3 FC HBM 流量结构性减少**（FP8 weights / mxfp8 / weight pre-permute） | DSV3 −2~4 ms（目标 ≤ 4 ms） | FP8 cast 路径 + numerical 验证；MI355X 原生 mxfp8 MFMA |
| P1 | **真正的 (e, M_concat, ni) batched-by-M**（把 8 个 src 的 T_e 拼成一个 M） | DSV3 −0.5~1.0 ms | 需要新的 `dispatch_tokens` permute（pre-scatter） |
| P2 | Work-stealing tile counter（atomic 替代静态 round-robin），吸收 T_e 不均 | −0.2~0.5 ms | 实现简单，但收益取决于 T_e 方差 |
| P3 | 重启 tail WG 并行化（compute 真正比 tail 快之后） | 0.3~0.7 ms | 必须先把 atomic spin 换成 LDS-cached flag，否则像 P1 一样回退 |

## 关键教训

1. **Barrier 合并是 cross-WG 同步频繁场景的高 ROI 项**：把 24 → 3 直接拿到
   ~1 ms wall 收益，且零正确性风险。
2. **L2 reuse 不能纸面推断**：缓存替换、tile 维度、WG 数量、tile schedule
   交织起来，实际命中率往往远低于「dataset / cache size」估算。要真复用
   weight，得直接改成 batched-by-M 形状，不要寄希望于"调 schedule 让 cache
   帮我们"。
3. **每减一个串行依赖，原本被它隐藏的另一个串行依赖会浮出来**：P2 干掉
   barrier 后，wait_dispatch 立刻顶上来。链式优化的常态。

## 相关文件

- 代码 diff（uncommitted）：
  - `csrc/fused_moe_super_kernel.hip`（`expert_compute_phase`、`GemmLdsLayout::pair_*`、`MoeScratchSizes`）
  - `tests/smoke_super_kernel.hip`（scratch alloc × NUM_GPUS）
- Bench 日志：
  - `benchmarks/results/p2_ratio_sweep.txt`（ratio 0.10~0.30 sweep）
  - `benchmarks/results/p2_l2reuse_bench.txt`（主结果）
- README 同步：`README.md` L368–449（含 P2 profile 表 + 新瓶颈说明）
- 详细 perf：`benchmarks/PERF_REPORT.md`
- 上一篇 note：[`2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md`](./2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md)
