# 2026-05-21 17:00  M1b [UP] — Dispatch sub-WG 拉宽 (4 → 8) 关掉了 19 % 的差距

> 时间: 2026-05-21 16:30 → 17:00 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, mi355-gpu-7), SLURM job 27091.batch
> 容器: xiaoming-dev (podman, ROCm 7.2 / PyTorch 2.10)
> 上一节点: M1 BASELINE (receiver-pull dispatch 移植 + bit-exact, T=2048 host wall 2.03 ms) [2026-05-21_1630_BASELINE_m1_dispatch_ported.md]
> 代码: `csrc/include/rocmoe/dispatch.h` 单常量改动 (`kSubWGs 4→8`, `kDispatchWGs 32→64`) + `benchmarks/bench_dispatch.hip` 改用 per-rank `hipEvent` 报 device wall
> 本轮 flag: **`UP`** (round 2 是 headline; round 1 `FLAT` / round 3 `DOWN` 已回退 / round 4 `DOWN` 已回退)

## TL;DR

把 dispatch kernel 的 `kSubWGs` 从 4 拉到 8 (per-src 接收组从 4 个 sub-WG 变 8 个, dispatch 总 WG 数从 32 涨到 64), DSv3 production (8x MI355X, 32 experts, topk=8, T=2048, H=7168) **device wall 1.940 → 1.567 ms = -19 %**, 全 T sweep 均一 -18~-19 %, `test_dispatch` 4 个 workload 仍然 bit-exact。剩下两条路 (kSubWGs=16 / 去掉接收循环末尾的 `__syncthreads()`) 分别在 XGMI 带宽和 wave-lockstep 上撞墙,都已回退。结论: 单 kernel 形式的 dispatch 已经被 XGMI peer-read 带宽夹住,继续往 1 ms 验收门槛走必须靠 M2 持久 super-kernel 让 dispatch 和 FC1 的内存流量真正 overlap。

## 1. 进入 M1b 的输入

| metric | M1 BASELINE (host wall) | M1 BASELINE (device wall) |
|---|---|---|
| T=2048 latency | 2.030 ms | 1.940 ms |
| T=4096 latency | ~3.94 ms | 3.808 ms |
| topk skew | balanced (skew 0.977) | — |

M1 note 列出了三条疑似差距来源:
1. host 端 launch + stream sync 开销 (M1 怀疑 ~0.3 ms)
2. `kSubWGs=4` × `kMaxRanks=8` = 32 WG 在 MI355X 256 CU 上只填 12.5 %, CU 占用率太低
3. cooperative b128 copy (16 bytes/lane/次) 在 H=7168 行宽下可能没充分 pipeline

M1b 把这三条逐条验证。

## 2. 做了什么 / 四轮实验

### 2.1 Round 1 — `hipEvent` device-wall 计时 (`FLAT`)

把 `bench_dispatch.hip` 从单一 host wall 改成 per-rank `hipEvent` pair 包住每次 launch + sync, 取跨 rank max (= critical path) 作为 device wall, 同时也保留 host wall。

- T=2048: host wall 1.964 ms, **device wall 1.940 ms** → host 开销只占 0.025 ms (~1 %)。
- per-rank skew (T=2048) = 0.977, 几乎完美平衡, 没有 straggler rank, 也没有不均衡的 peer-read 树。
- **结论**: 假设 (1) 是错的, 2 ms 是 kernel 本身, 不是 host launch。`FLAT` (perf 不变, 但 device-wall 量法以后一直保留)。

### 2.2 Round 2 — `kSubWGs` 4 → 8 (**`UP`**, headline)

每个 src-rank 的 receiver 组从 4 个 sub-WG 变成 8 个, per-src 32 次 receiver pull 从 4 路并行变 8 路并行。dispatch 总 grid: `kMaxRanks × kSubWGs` = 8 × 8 = **64 WG / rank**。

代码改动只有 `csrc/include/rocmoe/dispatch.h` 一个常量。`test_dispatch` 4 个 workload 仍然 bit-exact。

| T    | M1 baseline (dev wall) | M1b round 2 (dev wall) | delta |
|------|------------------------|------------------------|-------|
| 512  | 0.528 ms               | **0.431 ms**           | -18 % |
| 1024 | 0.991 ms               | **0.798 ms**           | -19 % |
| 2048 | 1.940 ms               | **1.567 ms**           | -19 % |
| 4096 | 3.808 ms               | **3.082 ms**           | -19 % |

- **Flag = `UP`**, 整个 T sweep 均一 ~19 % 提升。
- 跟 T 的线性 scaling 仍然成立 (1.567 / 0.798 ≈ 1.96, 3.082 / 1.567 ≈ 1.97), 说明 kernel 依然是带宽 bound, 只是带宽利用率往上走了一截。

### 2.3 Round 3 — `kSubWGs` 再拉到 16 (`DOWN`, 已回退)

继续往上拉, per-src 16 个 sub-WG, 总 128 WG/rank (~50 % of 256 CU)。

- T=2048 device wall: 1.581 ms vs round 2 的 1.567 ms = **+0.9 %**。
- T=4096: 3.096 ms vs 3.082 ms = +0.5 %。
- T=512: 0.445 ms vs 0.431 ms = +3.3 %。
- **Flag = `DOWN`** (大 T 在 noise 里, 小 T 真退步)。已回退到 `kSubWGs=8`。
- **解释**: 新瓶颈是 peer XGMI read 带宽, 不再是 receiver 侧 CU 占用率。再加 pull-WG 只是增加 peer-read 争用, 链路在 8 sub-WG 已经被打满。

### 2.4 Round 4 — 去掉 receiver 内循环尾的 `__syncthreads()` (`DOWN`, 已回退)

假设: `tid==0` 写完 meta + topk_wts + 给 atomic counter +1 之后, 那个尾部 `__syncthreads()` 只是在保 thread 之间的公平; `tid != 0` 的 thread 理论上可以提前进 slot k+1 的 `cooperative_b128_copy`, 跟 shared memory 或 HBM 都无依赖。

- 正确性还在 (`test_dispatch` 仍 PASS), 但 **+15 % regression**: T=2048 device wall **1.567 → 1.803 ms**。
- **Flag = `DOWN`**。已恢复 sync。
- **解释**: 去掉 post-atomic sync 让 WG 内的 wave 漂移开 —— `tid==0` 在 L2 atomic 的 round-trip 上掉队, 其它 thread 提前进下一行 b128 load, 结果 XGMI peer read 被散到不同 cache line 上, 没法在一拍内合并成同一条 line。多吃的带宽比省掉的 sync 还贵。具体地证实了: **在这个 dispatch kernel 里, wave-lockstep 是 XGMI read coalescing 的必要条件**, 不只是正确性条件。

## 3. M1b 终态

| metric | M1 BASELINE (dev wall) | **M1b FINAL** | vs BASELINE | vs 1 ms 目标 |
|--------|------------------------|---------------|-------------|---------------|
| T=512  | 0.528 ms               | **0.431 ms**  | -18 %       | 已 < 1 ms     |
| T=1024 | 0.991 ms               | **0.798 ms**  | -19 %       | 已 < 1 ms     |
| T=2048 | 1.940 ms               | **1.567 ms**  | -19 %       | **1.57 × over** |
| T=4096 | 3.808 ms               | **3.082 ms**  | -19 %       | 3.08 × over (1 ms 目标仅约定 @ T=2048) |

- T ≤ 1024 已经在 1 ms 以内。
- T=2048 (DSv3 production 点) 落在 1.57 ms, 离 1 ms 还差 1.57 ×。Round 4 已经证明内循环不安全到能再削同步; round 3 已经证明 kernel 在 8 sub-WG 已经被 XGMI 带宽夹住。
- **单 kernel 形式下的 bandwidth-bound 工作已经触顶**。

## 4. 为什么这里停手, 不继续 micro-tune 单 kernel

把 receiver-pull 作为 **standalone launch** 已经被 XGMI peer read 带宽夹死。想再啃掉剩下的 ~37 %, dispatch 必须 **停止作为 standalone launch**:

- 装到持久 super-kernel 之后, dispatch 可以跟 FC1 共享同一个 persistent grid, peer-read 流量跟 FC1 的 MFMA burst 在 chunk 级 overlap (`cco-pipeline-overlap` skill 原则 3: minimal barriers)。
- 1 ms 验收门槛本来就是 **overlapped 工况**下的目标, 不是 standalone launch latency 的目标。
- 继续打磨 standalone latency 的风险: 优化在 pipeline 内可能根本不起作用, 等于白烧时间。

所以 M1b 收尾。Headline 一句话: **`kSubWGs=4 → 8` 是个零正确性风险的 +19 % 干净胜利**(bit-exact 全保住), 剩下的差距是 standalone launch 模型本身的属性, 等 M2 的 persistent super-kernel 把它溶解掉。

## 5. 下一步 (M2 起)

1. **M2** — 移植 persistent super-kernel (5-phase: dispatch → FC1+SwiGLU+FC2 → push → combine), 直接复用 `dispatch_body.h`, 保住 M1b 这 19 % receiver-pull 增益。验收: 0 hang / 0 NaN / 小 T 上对 PT+RCCL bit-exact。
2. **M2** — 在 DSv3 工况测 end-to-end forward latency, 这才是 1 ms dispatch 目标第一次有意义的地方。
3. **M3** — 只在 M2 表明 overlap 之后 dispatch 仍然 bound 时, 再回头重做 `cooperative_b128_copy` 为 2-row pipelined 版本 (每次 `b128_copy` 覆盖 2 行 token, 让 `n=896 > kWGSize=512`, unrolled prefetch loop 才会真的 fire; H=7168 + kWGSize=512 当前会退化成 un-pipelined tail loop)。

## 6. 本轮触碰的文件

- `csrc/include/rocmoe/dispatch.h` — `kSubWGs 4 → 8`, `kDispatchWGs 32 → 64`。
- `csrc/include/rocmoe/dispatch_body.h` — round 4 回退后, 在 per-row 尾 `__syncthreads()` 上补了一条理由注释 (wave-lockstep 是 XGMI coalescing 的前置条件)。
- `benchmarks/bench_dispatch.hip` — per-rank `hipEvent` pair, 报 device wall (跨 rank max = critical path)。

## 7. 相关文件

- 上一节点 (M1 BASELINE): [`2026-05-21_1630_BASELINE_m1_dispatch_ported.md`](./2026-05-21_1630_BASELINE_m1_dispatch_ported.md)
- 上上节点 (M0 BASELINE): [`2026-05-21_1330_BASELINE_m0_mfma_tile_ported.md`](./2026-05-21_1330_BASELINE_m0_mfma_tile_ported.md)
- 架构设计: [`2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md)
- skill: `~/workspace/slab/.cursor/skills/rocmoe-dev-loop/SKILL.md`, `~/workspace/slab/.cursor/skills/cco-pipeline-overlap/SKILL.md`
