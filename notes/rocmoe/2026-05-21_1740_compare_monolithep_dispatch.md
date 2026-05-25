# 2026-05-21 17:40  RocMoE-v2 dispatch vs MonolithEP dispatch — apples-to-apples 对比

> 时间: 2026-05-21 17:40 (Asia/Shanghai)
> 项目: rocmoe
> 类型: 跨项目性能对比 (不是新实验, 是把已有数据拉到同一张图上读)
> 数据源:
>   - RocMoE-v2 M1b standalone: `notes/2026-05-21_1700_UP_m1b_dispatch_subwg_widening.md` (4 T sweep, kSubWGs=8, `bench_dispatch` per-rank `hipEvent`)
>   - RocMoE-v2 M2-D in-super-kernel: `notes/2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md` (4 T sweep, `bench_super_dispatch`)
>   - MonolithEP r5 phase-1 sweep: `~/workspace/MonolithEP/notes/r5_phase1_perf_analysis.md` (sub_wg/dest ∈ {2,4,8,16,32}, B_PER_RANK=8192)
>   - MonolithEP BF16 sprint closeout: `~/workspace/MonolithEP/notes/2026-05-21_1310_bf16_sprint_closeout_pivot_fp8.md` (super-kernel 15.10 ms 地板)
>   - MonolithEP deep-profile: `~/workspace/MonolithEP/notes/2026-05-21_1013_deep_profile_super_15_14ms_g0_concentrates_loss.md` (current build bench_dispatch 数)
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), 节点 mi355-gpu-7
> 容器: `xiaoming-dev` (podman, ROCm 7.2 / PyTorch 2.10)

## TL;DR

把今天能拿到的 dispatch wall 数据一次性归到同一张表, 修正 M1 BASELINE note 里 "比 MonolithEP 等价 stage 快 33 %" 这条**量纲混了的错误对比**。修正后的结论:

| 比较点 | MonolithEP (push, Layout-E) | RocMoE-v2 (pull, Layout-P) | 差距 |
|---|---|---|---|
| Standalone dispatch (per-token) | **0.525 µs/tok** (T=8192, 64 WG) | 0.752 µs/tok (T=8192 外推, 64 WG) | RocMoE-v2 **慢 ~43 %** |
| In-super-kernel dispatch (per-token) | 0.659 µs/tok (T=8192, BF16 closeout) | 0.815 µs/tok (T=2048, M2-D) | RocMoE-v2 慢 ~24 % (**不公平比较**) |
| 1.57 ms 物理 BW 地板 | 同 (= 822 MB / 525 GB/s) | 同 | 一致 |

**Standalone 上 push 结构性更快**, 因为 pull 多一个 `atomic_load_acquire(block_ready)` round-trip + inbound read 比 outbound write 少在飞 transactions。**但 standalone 不是 RocMoE-v2 押的赌注** —— 赌的是 super-kernel overlap (M2-G 之后才能验收) + Layout-P 砍掉 g=0 fan-in spin 1.34 ms (M3 之后才能验收) + pull-combine 砍掉 FC2 8-way outbound contention 0.7 ms (MonolithEP BF16 sprint closeout 已经证明在 BF16 下不可消除)。**今天写这份 note 是为了把"我们在 standalone 上结构性慢"作为已知事实记下来, 防止下一阶段误把 standalone 数当胜负判据。**

## 1. 平台对齐 / 数据来源

两边都跑在 8×MI355X / DSv3 H=7168 / BF16, 都用一样的 `bench_dispatch`-style per-rank `hipEvent` device wall 量法 (取跨 rank max = critical path)。两边的 sub-WG 调到了一样的 sweet spot:

| 项 | MonolithEP | RocMoE-v2 |
|---|---|---|
| Dispatch WG 总数 | 64 (= 8 dst × 8 sub_wg/dest, sweep 后定 sweet spot) | 64 (= 8 partner × 8 kSubWGs, M1b round 2 sweep 后定) |
| 物理 BW 理论地板 | 822 MB / 525 GB/s = **1.57 ms** | 同 (相同 token 量和 H 算出来一样) |
| MFMA / GEMM 路径 | 无 (dispatch is BW-bound) | 无 (dispatch is BW-bound) |

**唯一不一样的 metric 是 T_per_rank**: MonolithEP 把 `B_PER_RANK=8192` 写死成编译期常量, RocMoE-v2 把 T 当 sweep 参数。因为 dispatch 是 BW-bound, 公平 metric 是 **µs/token**, 跟 T 几乎独立 (RocMoE-v2 实测 T=512..4096 范围内 µs/tok 落在 0.75-0.84 µs/tok 区间内, 拉到 T=8192 线性外推就行)。

## 2. Standalone dispatch (per-token apples-to-apples)

| backend | 协议 | layout | dispatch WG | T_per_rank | dev wall | µs/token | XGMI link util |
|---|---|---|---:|---:|---:|---:|---:|
| MonolithEP r5 sweep (N=8 sweet spot) | **push** | Layout-E | 64 | 8192 | **4.30 ms** | **0.525** | 37 % |
| MonolithEP 2026-05-21 deep-profile (iso) | push | Layout-E | 64 | 8192 | 4.14 ms | 0.505 | 38 % |
| MonolithEP 2026-05-21 deep-profile (fused build) | push | Layout-E | 64 | 8192 | 6.96 ms | 0.850 | 23 % (生产 build 跟 phase-2 共享 launch_bounds) |
| RocMoE-v2 M1b standalone | **pull** | Layout-P | 64 (kSubWGs=8) | 512 | 0.432 ms | 0.844 | ~27 % |
| RocMoE-v2 M1b standalone | pull | Layout-P | 64 | 1024 | 0.798 ms | 0.779 | ~29 % |
| RocMoE-v2 M1b standalone | pull | Layout-P | 64 | 2048 | 1.568 ms | 0.766 | ~30 % |
| RocMoE-v2 M1b standalone | pull | Layout-P | 64 | 4096 | 3.082 ms | 0.753 | ~30 % |
| RocMoE-v2 M1b extrapolated linear | pull | Layout-P | 64 | 8192 | ~6.16 ms | ~0.752 | ~30 % |

**核心数字**: 同 T=8192, push 0.525 µs/tok vs pull 0.752 µs/tok = **push 快 ~30 % per-token (反过来说: pull 慢 ~43 %)。**

## 3. In-super-kernel dispatch (snapshot — not apples-to-apples)

| backend | 工况 | T_per_rank | dispatch 段 wall | µs/token | 备注 |
|---|---|---:|---:|---:|---|
| MonolithEP BF16 closeout (super) | 完整 5-phase super-kernel (dispatch + 4-group FC1/SwiGLU/FC2 + combine), `g=0` FC2 跟 phase 1 XGMI **共占链路** | 8192 | **5.4 ms** (phase 1 in super) | 0.659 | +1.1 ms vs iso 4.30 ms, contention 来自 phase-2 FC2 push 抢 XGMI |
| MonolithEP 2026-05-21 deep-profile (super, 早一点 build) | 同上, 早版 | 8192 | 7.71 ms (phase 1 in super, vs iso 4.14 ms) | 0.941 | super-vs-iso +3.57 ms |
| RocMoE-v2 M2-D BASELINE (super) | persistent grid, **只有 dispatch role 真 body**, GEMM / FC2_PUSH / TAIL_COMBINE 全是 noop stub | 2048 | **1.669 ms** | 0.815 | +100 µs 平坦 scaffold 税 (B1 phase_barrier<8> + 两个 grid_sync), **没有其它角色 contention** |
| RocMoE-v2 M2-D extrapolated linear | (同上) | 8192 | ~6.3 ms | ~0.77 | |

**这一行的数没法直接对比**, 因为:

- MonolithEP 5.4 ms = "**已经被 phase-2 contention 抬高的 phase 1**" 数 (iso 4.30 ms → super 5.4 ms = +1.1 ms FC2 push 抢 XGMI 的成本)。
- RocMoE-v2 1.67 ms = "**孤岛里跑的 dispatch**" 数 (其它角色 noop), 既没经历 FC1 GEMM contention 也没拿到 FC1 overlap 收益。

**真正的对比**等 M2-G 落地之后才有意义 — 那时 RocMoE-v2 的 pull-dispatch peer-read 流量会跟 FC1 MFMA burst 在 chunk 级 overlap, RocMoE-v2 才会同时:
1. **承担**跟 FC1 共享 CU / L2 的 contention (向上的力)
2. **获得**把 dispatch wall *藏到* FC1 MFMA 后面的 overlap (向下的力)

赌的就是第 2 项的减项 > 第 1 项的加项。

## 4. 为什么 standalone 上 pull 结构性慢

这是 RocMoE-v2 架构设计 (`notes/2026-05-21_1252_rocmoe_v2_architecture_design.md` §2.2) 已经写过的取舍, 现在拿数据钉一下:

| 维度 | push (MonolithEP) | pull (RocMoE-v2) |
|---|---|---|
| XGMI 方向 | 每个 sender 8 outbound writes | 每个 receiver 7 inbound reads |
| Per-row 完成判据 | issue 完就走 (fire-and-forget, credit 控反压) | issue → wait 回包 → 算这行完 (request-response) |
| 同步开销 | 仅 `__syncthreads()` (wave-level coalesce) | `__syncthreads()` + per-block `atomic_load_acquire(block_ready)` 等 sender 写完 metadata |
| In-flight transactions / CU | 高 (XGMI write queue 深, MonolithEP rocprof 实测 `WRREQ_LEVEL` 7.15e9/ms @ N=8) | 低 (XGMI inbound read 需 RTT, M1b round 4 间接证明 wave-lockstep 是 coalescing 前提) |
| 失败模式 | one-slow-receiver 阻 8 个 sender (8 写一个 peer 时争 inbound) | one-slow-sender 只阻自己那块 block_b 的 receive WG, 其它 block 不受影响 |

**结论**: standalone latency 这个维度 push 结构性占优, 30 % gap 不会通过继续 micro-tune 关掉 (M1b round 3/4 已经实证 — XGMI 带宽夹死 + wave-lockstep 是 coalescing 前提)。**架构上没指望在 standalone 上赢 MonolithEP**。

## 5. RocMoE-v2 的赌注押在哪三处 (尚未兑现)

| 赌注 | 期望收益 (architecture-design note §2 估算) | 何时验收 |
|---|---|---|
| Layout-P 64-bit per-block scoreboard 砍掉 g=0 fan-in spin | **-1.34 ms** (MonolithEP deep-profile §4 量到 g=0 自旋占 super-vs-iso 损失 84 %) | M2-G (GEMM 装进 super-kernel, 用 per-block ready bitmap 驱动 work-steal) |
| pull-dispatch peer-read 跟 FC1 MFMA chunk-级 overlap | **dispatch wall 部分藏到 FC1 后面**, 抵消 standalone 慢的 30 % gap | M2-G |
| pull-combine 绕过 FC2 8-way outbound scatter | **-0.7 ms** (MonolithEP BF16 sprint closeout 已经证明在 BF16 下这条不可消除, FP8 才能减半) | M3 (FC2 + combine 装进 super-kernel) |
| 合计 (设计估算) | -34 ms → -13.5 ms BF16 super wall (-60 %) | M3 end-to-end vs PT+RCCL |

**最值钱的赌注是第 1 个 (Layout-P g=0 spin)** —— MonolithEP 2026-05-21 BF16 sprint closeout 用 D / G / A 三个 iter 验证过, 在 Layout-E 的框架下这 1.34 ms 自旋拿不掉 (XGMI traffic conserved), 只能切 FP8 把 GEMM 砍一半才能间接绕过。RocMoE-v2 的 Layout-P 是从结构上消除这条 stall, 这是它存在的核心 justification。

## 6. 修正以前的错对比

之前 M1 BASELINE note (`notes/2026-05-21_1630_BASELINE_m1_dispatch_ported.md`) 里写:

> DSv3 (T=2048) host wall 2.03 ms (mean), 较 MonolithEP 等价 stage 快 ~33%

这条**不成立**。具体哪一步出问题:

1. 量纲混了: RocMoE-v2 用的是 T=2048 host wall, MonolithEP 用的是 T=8192 super-kernel 内 phase 1 wall (~3 ms 引用本来就模糊)。
2. 没换算到 µs/tok: 真换算后, RocMoE-v2 M1b 0.766 µs/tok @ T=2048 vs MonolithEP standalone 0.525 µs/tok @ T=8192, **是 RocMoE-v2 慢 46 %**, 不是快 33 %。
3. 没区分 standalone vs in-super-kernel: MonolithEP 的 ~3 ms 是 super-kernel 内的 phase 1 (被 phase 2 contention 抬过的数), 不是 standalone bench_dispatch。

**这条以前的对比作废**, 以本 note §2/§3 为准。M1 BASELINE note 不改 (按 README 约定 "老 note 不改"), 但本 note 在引用 §6 里写了 superseded。

## 7. 命名 / 量纲约定 (后续 RocMoE 项目内统一)

为了避免再出现量纲混的错对比, 这之后所有 RocMoE-v2 perf 报数:

| 量 | 报哪种 |
|---|---|
| 单点 dispatch wall | per-rank `hipEvent` device wall, 跨 rank max = critical path; 同时报 host wall 但只在 footnote |
| 跨工况 / 跨 backend 对比 | **µs/token** (= dev wall / T_per_rank) 优先, 因为 dispatch 是 BW-bound |
| Standalone vs in-super-kernel | 明确标注 "standalone" 或 "in-super-kernel + 还存在的真 body 列表" |
| 跟 MonolithEP 对 | 永远拉到同 T (用 µs/tok), 或者用 RocMoE-v2 T=8192 线性外推数 |
| 跟 PT+RCCL baseline 对 | 留到 M3 end-to-end 时再说, 单段 dispatch 跟 PT 的 `all_to_all_single` 不可对照 |

## 8. 相关文件

- 上一节点: [`2026-05-21_1700_UP_m1b_dispatch_subwg_widening.md`](./2026-05-21_1700_UP_m1b_dispatch_subwg_widening.md)
- 下一节点: M2-D 同时段已有 [`2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md`](./2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md) (in-super-kernel dispatch 落地)
- 架构设计: [`2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md)
- MonolithEP 原始 perf 来源:
  - `~/workspace/MonolithEP/notes/r5_phase1_perf_analysis.md` — sub_wg sweep + rocprofv3 量化
  - `~/workspace/MonolithEP/notes/2026-05-21_1013_deep_profile_super_15_14ms_g0_concentrates_loss.md` — current build standalone + super 数
  - `~/workspace/MonolithEP/notes/2026-05-21_1310_bf16_sprint_closeout_pivot_fp8.md` — BF16 15.10 ms 地板 + 为什么 g=0 / FC2 在 BF16 下无解
- 修正对象: `notes/2026-05-21_1630_BASELINE_m1_dispatch_ported.md` §"33% 快于 MonolithEP" (已 superseded)
