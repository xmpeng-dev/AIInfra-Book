# 2026-05-22 10:35  Dispatch 5-phase profile：修正 09:50 note 的 skew 归因

**日期**: 2026-05-22 10:35 (Asia/Shanghai)

> 项目: rocmoe
> 类型: FLAT — 无性能变化，仅给 dispatch kernel 加 per-phase `clock64()` 探针 + 新 bench，把 5 个 phase 各自耗时量出来，**修正** [09:50 BASELINE note](./2026-05-22_0950_BASELINE_dispatch_skew_sensitivity_kills_pull_robust_hypothesis.md) 两个关键判断：
>   1. **标签错了**：09:50 note 通篇写的 "M1b standalone"，实际当前 `csrc/dispatch.hip` 跑的是 **M1c-A**（`ROCMOE_DISPATCH_USE_PACKED_OUTBOX=1` 默认开 + 3-phase sender + grid_sync<0/1/2> + cross-rank phase_barrier + packed_outbox 路径）。所有 09:50 的 "M1b" 都应读作 "M1c-A standalone (current default)"。
>   2. **机理错了**：09:50 note §3 把 skew tax 归因于 "pull 的 per-block `block_ready` polling 让 receiver 卡在最慢 sender 的最长 bucket"。事实上**当前代码里没有 `block_ready` polling**；真正的 skew 来源是 (a) PhaseB pack max-rank +41%、(b) syncB grid_sync<1> +597%、(c) Receiver per-rank spread +31%；**cross-rank phase_barrier 占 wall < 1%（39-64 μs），完全不是 skew 主因**。
>
> 触发: user "profile 一下这 5 个 phase，看看各自的耗时"
> 硬件: 8× MI355X (gfx950, mi355-gpu-7)，SLURM 13588
> 容器: `xiaoming-dev` (podman, ROCm 7.2)
> 数据:
>   - `bench_results/dispatch_phase_profile_20260522_1035.csv`（3 个 skew × T=8192 dsv3，per-phase ms）
>   - 同一节点 stdout：`bench_results/log_dispatch_phase_profile_20260522_1035/`
> 上游:
>   - [`2026-05-22_0950_BASELINE_dispatch_skew_sensitivity_kills_pull_robust_hypothesis`](./2026-05-22_0950_BASELINE_dispatch_skew_sensitivity_kills_pull_robust_hypothesis.md)
>     — 被本 note 修订

## 1. 背景 / 目标

09:50 BASELINE note 把 4 model × 3 skew 的 dispatch wall 跟 RCCL 对照后，得出 "pull-dispatch 在 hot_cov50 下 +31-50% / RCCL 只 +11-19%"，并把根因归到 receiver 端的 per-block `block_ready` polling。

问题：

1. 我们当前 `csrc/dispatch.hip` 走的路径其实是 M1c-A（3-phase sender pack 到 `packed_outbox` 然后 receiver 顺序读 slot），不是 M1b（receiver 直接 scatter 读 `peer.input_token_buf[src_t]`）。这两条路径下 receiver 都没有 per-block 的 `block_ready` polling —— 只有一次 cross-rank `phase_barrier` 等所有 sender publish 完 `send_done_flag`，然后 receiver 全 grid 并行读。
2. 既然没有 per-block polling，那 skew tax 到底花在哪？要确认就只能把 kernel 5 个 phase 各自的耗时量出来。

本 note 给 `rocmoe_dispatch_kernel` 加了零开销 per-WG `clock64()` 探针 + 一个独立 bench (`bench_dispatch_phases`)，把答案钉死。

## 2. 主要发现 / 结论

### 2.1 5 phase 耗时分解（T=8192, dsv3, M1c-A, 8× MI355X）

| Phase                                  | balanced | realistic_cov20 | hot_cov50 | Δhot/bal  | 占 hot wall |
|----------------------------------------|---------:|----------------:|----------:|----------:|------------:|
| PhaseA   sender meta（sub_wg==0）      |    0.136 |           0.136 |     0.137 |      +1%  |       1.3%  |
| syncA    grid_sync<0>                  |    0.138 |           0.138 |     0.139 |      +1%  |       1.3%  |
| **PhaseB  sender pack（8 sub-WGs）**   |    1.180 |           1.335 |     1.667 |   **+41%**|     15.3%   |
| **syncB   grid_sync<1>**               |    0.120 |           0.339 |     0.837 | **+597%** |       7.7%  |
| PhaseC   sender publish（sub_wg==0）   |    0.002 |           0.002 |     0.003 |         — |      <0.1%  |
| **cross-rank phase_barrier (blk==0)**  |  **0.039** |     **0.041** | **0.064** |    +64%   |     **0.6%**|
| syncC    grid_sync<2>                  |    0.043 |           0.046 |     0.068 |     +58%  |       0.6%  |
| **Receiver pull（all WGs）**           |    7.957 |           7.969 |     7.987 |      +0%  |    **73.3%**|
| **Σ phase walls**                      |    9.615 |          10.008 |    10.901 |     +13%  |             |
| hipEvent total（critical-path rank）   |    9.435 |          10.406 |    12.381 |     +31%  |             |

> 数值是 8 rank × 10 iter 的中位，每个 phase 的 wall 取 "per-WG max delta"（同一 CU 内的 clock 是单调的，per-WG delta 才有意义；不能跨 WG 比绝对时间戳，因为 `clock64()` 在 AMD CDNA 是 per-CU shader counter）。校准用 max-over-WGs of `(ts[8] - ts[0])` 对 hipEventElapsedTime，校出 2.400 GHz —— 跟 MI355X boost clock 完全对得上。

### 2.2 Receiver per-rank 离散（真正的 critical-path 来源）

| 指标                  | balanced     | realistic       | hot              |
|-----------------------|-------------:|----------------:|-----------------:|
| Receiver wall 范围（min-max per rank）| 7.886 – 8.033 ms | 7.369 – 8.872 ms | 6.256 – **10.502** ms |
| Receiver per-rank 离散度（max - min） |    0.15 ms       |    1.50 ms       |    **4.25 ms**   |

Receiver 阶段的 **per-rank mean 几乎不变**（~8 ms），但 **max rank 在 hot 下跳到 10.5 ms** —— 多出来的 2.5 ms 就是持有 hot expert 的那个 rank 比其他 rank 多收的 token 量带来的工作量差。Σ-phase-walls 表只看 mean-across-rank，所以 +13% 看上去温和；实际 hipEvent critical-path = max-rank = +31%。

### 2.3 PhaseB pack max-rank（sender 端的 skew 放大）

| Phase                       | balanced max-rank | realistic max-rank | hot max-rank |
|-----------------------------|------------------:|-------------------:|-------------:|
| PhaseB pack                 |        1.199 ms   |       1.372 ms     | **1.709 ms** |
| syncB grid_sync<1>          |        0.140 ms   |       0.371 ms     | **0.868 ms** |

Pack 阶段同一 partner WG（=同一 dst_rank）下，hot expert 所在的那个 (dst_rank, hot_e) bucket 会比 balanced 多收 ~6× token。8 个 sub-WG 协作摊薄掉绝大部分，但最慢那个 partner 仍然慢 ~40%。其他 partner 早早干完，全部卡在 grid_sync<1> 等它 —— 这就是 syncB +597% 相对涨幅的物理含义（绝对增量 +0.72 ms，是 sender 端 skew 损失的主要可见点）。

## 3. 详细分析

### 3.1 09:50 note 错在哪里

09:50 note §3 假设当前代码是这样的（原文伪代码）：

```
for each (e, src_rank, block_b) it owns:
    spin on atomic_load_acquire(peer.block_ready[blk]) until set
    cooperative_b128_copy(peer.input_token_buf[bucket], local.expert_token_pool[slot])
```

但 `csrc/include/rocmoe/dispatch_body.h` 里 receiver 实际是这样的：

```cpp
for (int e = tid; e < epg; e += kWGSize) {
    smem_counts[e] = atomic_load_agent_acquire(
        self.expert_recv_count_ptr(e, src_rank));   // ← AGENT-scope, NO per-block flag
}
...
for (int e = 0; e < epg; ++e) {
    int n_slots = smem_counts[e];
    for (int s = sub_wg; s < n_slots; s += kSubWGs) {
        bf16_t* src_row = peer.packed_outbox_row(my_rank, e, s);  // M1c-A
        cooperative_b128_copy(dst_row, src_row, H * sizeof(bf16_t));
        ...
    }
}
```

cross-rank `phase_barrier` 在 receiver 启动**之前**就已经把 "所有 peer 都 publish 完 `send_done_flag`" 这件事保证好了，所以 receiver 不需要任何 per-block 轮询。AGENT-scope acquire 那一下只是 fence 性质（保证 receiver 看到 peer 的 publish），不会 spin。

→ 真正的 skew 损失分解（按数值占比排序）：

1. **Receiver per-rank imbalance** （hot 下 4.25 ms 离散）：mean 不动，max 比 mean 多 +31%。物理含义是不同 rank 持有的 hot/cold expert 数量不同 → 收到的 token 数不同 → 拉取耗时不同。
2. **PhaseB pack max-rank** （hot 下 +41%）：sender 端最慢 partner 多 6× 行要 copy 到 packed_outbox。
3. **syncB grid_sync<1>** （hot 下 +597% 相对）：fast partner 等 slow partner，是 (2) 的 grid 内 spill-over。
4. **cross-rank phase_barrier** （hot 下 39 → 64 μs）：< 1% wall，**几乎可以忽略**，跟昨晚 note 的归因相反。
5. **PhaseA / PhaseC** 均 < 1% 且 skew-immune（counting sort 总功 = T·topk = const，publish 只写 epg+1 个 u32）。

### 3.2 工具：phase profile 的实现

加在 `csrc/include/rocmoe/dispatch.h` 的 `DispatchArgs`：

```cpp
constexpr int kDispatchPhaseSlots = 9;
struct DispatchArgs {
    ...
    uint64_t* phase_timestamps = nullptr;   // 默认 null，零开销
};
```

加在 `csrc/dispatch.hip` 的 `rocmoe_dispatch_kernel`：

```cpp
uint64_t* ts_base = args.phase_timestamps
    ? args.phase_timestamps + blk * kDispatchPhaseSlots : nullptr;
auto record_ts = [&] (int slot) {
    if (ts_base && tid == 0) ts_base[slot] = clock64();
};
record_ts(0);
if (sub_wg == 0) dispatch_sender_meta_phase(args, partner, &lds);
record_ts(1);
grid_sync<0>(...); record_ts(2);
dispatch_sender_pack_phase(args, partner, sub_wg);
record_ts(3);
grid_sync<1>(...); record_ts(4);
if (sub_wg == 0) dispatch_sender_publish_phase(args, partner);
record_ts(5);
if (blk == 0) phase_barrier<kMaxRanks>(...);
record_ts(6);
grid_sync<2>(...); record_ts(7);
dispatch_receiver_stage(args, partner, sub_wg, &lds);
record_ts(8);
```

`args.phase_timestamps == nullptr` 时整个 branch 是 uniform-WG 的常量比较，编译器会把它从循环里 hoist 掉 —— 生产路径（`bench_dispatch`、super-kernel）零成本，bit-exact `test_dispatch` 重跑全 PASS。

聚合关键点（实现在 `benchmarks/bench_dispatch_phases.hip`）：

- **`clock64()` 在 AMD CDNA 是 per-CU shader counter**，**不同 CU 之间有任意 offset**（实测 WG 0 起始 5.883e13、WG 63 起始 5.795e13，差 1e11 cycles）。所以**跨 WG 比较绝对时间戳是无效的**，只能用 per-WG delta。
- 每个 phase 的 "wall" 定义为 **max over WGs of (ts[p+1] − ts[p])**：因为每个 phase 之间都有 grid_sync / phase_barrier，最慢 WG 决定 phase wall。`Σ phase walls` 跟 hipEvent total 在 balanced 下偏差 0.18 ms（mean-across-rank 损失了一点），可接受。
- 校准 `cycles/ms` 用 per-iter max-over-WG of `(ts[8] − ts[0])` 对 hipEventElapsedTime 的中位 —— 实测 2.400 GHz，对得上 MI355X boost clock。
- `actv%` 列报告 "delta > 5% × phase_wall 的 WG 比例"：
  - PhaseA / PhaseC = 12.5% = 8/64（=sub_wg==0 of each partner）✓
  - phase_barrier = 1.6% = 1/64（=blk==0）✓
  - sync* / pack / receiver = ~100% ✓

新文件：

- `csrc/dispatch.hip` —— 加 `record_ts` 9 个 marker
- `csrc/include/rocmoe/dispatch.h` —— `DispatchArgs::phase_timestamps`
- `benchmarks/bench_dispatch_phases.hip` —— 独立 bench
- `CMakeLists.txt` —— 加 `bench_dispatch_phases` target

### 3.3 M2-G overlap 还能拿多少

receiver pull 占 dispatch wall 73-85%（balanced ~8.0 ms）。M2-G 的设计是让 GEMM 角色 WG 跟 dispatch 角色 WG 在 persistent grid 内并行，FC1 GEMM 在 dispatch 还没完成时就开始处理已 ready 的 pool block。

DSv3 prod (T=8192, EP=8, epg=32, H=7168, F=2048) 下 FC1 grouped GEMM 估算：

- 每 rank token 数 N_local = T·topk = 65536，对应 epg=32 个 expert（每 expert 平均 2048 token）
- FC1 = gate + up，每 token 2 × H × F muladd = 2 × 7168 × 2048 = 2.94e7 FLOPs
- Σ over rank ≈ 65536 × 2.94e7 = 1.93 TFLOPs / rank
- MI355X BF16 peak ~1.3 PFLOPs → **FC1 standalone ≈ 1.5 ms** （需 M0 GEMM bench 校准；roofline 估计）

→ **FC1 完美藏在 receiver pull 后面，最多省 ~1.5 ms / 10 ms = 15%**。这跟 09:50 note 暗示的 "M2-G 是大头" 差距较大 —— receiver 8 ms 里只有 1.5 ms 能被 FC1 吃掉，剩下 6.5 ms 还是裸 XGMI。

这并不否决 M2-G 路线（隐藏 15% + 把 dispatch tail 跟 FC1 头部串起来本来就是 super-kernel 的核心收益），但接下来需要：

1. **先用 standalone GEMM bench** 精确量出 FC1 在 DSv3 prod 形状下的真实 ms（M0 `bench_gemm` 已经能跑，不到 1 min），把 overlap 上限算准。
2. **如果 FC1 的确只有 1-2 ms**，M2-G 之后大概率还要走 **M3 (FC1+SwiGLU+FC2 in-LDS fused)** 或 **mxfp8 weights** 才能让 GEMM 阶段总耗时跟 dispatch 同量级（~8 ms），overlap 才能拉满。
3. M2-G note 验收门里那条 "hot_cov50 退化必须从 +31-50% 降到 ≤ +15%" 仍然成立（FC1 吃掉 PhaseB + syncB 的 skew 损失 ~0.85 ms 完全够），不必修改。

## 4. 下一步 / 建议

| 项 | 内容 |
|----|------|
| (a) | ✅ 修订 [09:50 BASELINE note](./2026-05-22_0950_BASELINE_dispatch_skew_sensitivity_kills_pull_robust_hypothesis.md)：顶部加 **已修订** 横幅，指向本 note；标签 "M1b standalone" 全部读作 "M1c-A standalone (current default)"；§3 receiver state machine 伪代码已被本 note §3.1 替换。 |
| (b) | 顺手补一组真 M1b 的 phase profile（rebuild `-DROCMOE_DISPATCH_USE_PACKED_OUTBOX=0`），看 Receiver / PhaseB 各自怎么变 —— 预期 M1b 的 PhaseB 几乎为 0（pack 是 no-op），Receiver 应该更短但 max-rank 离散度可能更大（scatter 读 `peer.input_token_buf` 比 contiguous 读 `peer.packed_outbox` 在 hot 下更不可预测）。**可选，不阻塞 M2-G。** |
| (c) | **优先**：跑 `build/bench_gemm` 量出 DSv3 prod (M ≈ 65536, N=2048, K=7168 grouped) FC1 真实 ms，校准 M2-G 的 overlap 上限。 |
| (d) | 启动 M2-G：把 GEMM body 装进 persistent super-kernel 的 GEMM 角色，验收门走原计划。phase profile 工具留着，M2-G 之后跑同一组 3-skew profile，对比 dispatch wall 在 super-kernel 里被压成多少。 |

## 5. 相关文件

- 工具
  - `csrc/dispatch.hip` —— per-WG `record_ts` 探针
  - `csrc/include/rocmoe/dispatch.h` —— `DispatchArgs::phase_timestamps`
  - `benchmarks/bench_dispatch_phases.hip` —— 独立 bench + 聚合 + 校准
  - `CMakeLists.txt` —— `bench_dispatch_phases` target
- 数据
  - `RocMoE/bench_results/dispatch_phase_profile_20260522_1035.csv`（3 skew × T=8192 dsv3）
  - `RocMoE/bench_results/log_dispatch_phase_profile_20260522_1035/*.log`（3 个原始 stdout）
- 同名 HTML 报告（中文，浏览器直接打开，跟本 md 同目录）
  - [`2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.html`](./2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.html)
- 上游 note
  - `2026-05-22_0950_BASELINE_dispatch_skew_sensitivity_kills_pull_robust_hypothesis.md`
  - `2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md`
  - `2026-05-21_2100_DOWN_m1c_a_revisit_dsv3_production_size.md`
