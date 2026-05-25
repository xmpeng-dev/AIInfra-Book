# 2026-05-21 21:00  M1c-A [DOWN] — DSv3 真实 size 复测,L2 cliff 假设部分被推翻

> 时间: 2026-05-21 20:30 → 21:00 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, mi355-gpu-7), SLURM 同一 allocation
> 容器: xiaoming-dev (podman, ROCm 7.2 / PyTorch 2.10)
> 上一节点: [M1c-A toy-size DOWN](./2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md) — 在 toy size (ne=32, T=2048) 看到 +50% regression,把原因归到"peer L2 cliff"。
> 触发: user "用真实大小, 不要用 toy, toy 很容易忘记, 最终误导优化" — 强制把 bench default 切到 DSv3 production size (num_experts=256, EP=8, epg=32, T=4096/8192, n_group=8, group_topk=4)
> 本轮 flag: **`DOWN`** (M1c-A 仍然慢, 但只 +25-30% 而非 +50%; 上一节点的"L2 cliff"叙述要修正)

## TL;DR

把 bench default 从 toy (ne=32, T=2048) 切到 DSv3 production (ne=256, T=4096/8192) 之后,
M1c-A 仍然是一个 DOWN, 但 regression **从 +50% 缩到 +25-30%**:

| Config (DSv3 prod) | M1b legacy | M1c-A packed | 回归 |
|---|---|---|---|
| T=4096, dist=dsv3    | 3.632 ms | 4.725 ms | **+30 %** |
| T=4096, dist=uniform | 3.655 ms | 4.751 ms | **+30 %** |
| T=8192, dist=dsv3    | 7.491 ms | 9.410 ms | **+26 %** |
| T=8192, dist=uniform | 7.486 ms | 9.372 ms | **+25 %** |

工程含义:
- M1c-A 仍然是一个明显回归 — `revert` 决定不变,继续走 M1c-B 的 **receiver 端 src_t 排序** 路线
- 但 [上一篇 note](./2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md) 把全部 +50% 都归因到
 "peer L2 cliff (28 MB → 3.59 GB)" 是部分错误 — 在 production size 下 `input_token_buf` 已经 56 MB > L2 32 MB,
 M1b 本身就不再独享 L2,但 M1b 还是赢 25-30%。所以 L2 cliff 解释 toy 数据没问题,**但解释不了 production 数据**
- 实际 production size 下 M1c-A 的输坏点是 **每 token 强制 8× 数据复制 + 来回 HBM round-trip**, 跟 L2 capacity
 是不是 fit 关系不大

## 1. 真实 size 的数字 (DSv3 production)

DSv3 router 配置 (从 `Primus-dev/primus/configs/models/megatron/deepseek_v3.yaml` 拷过来的):

| 参数 | 值 |
|---|---|
| EP ranks | 8 (单节点 MI355X) |
| num_experts (全局) | 256 |
| epg (每卡 experts) | 32 |
| topk | 8 |
| n_group | 8 |
| group_topk | 4 |
| router | sigmoid + group-limited top-k |
| H (hidden) | 7168 |
| T (per-rank tokens) | 4096 / 8192 (BS=1 / 2 at seqlen=4096) |
| max_recv_per_e_per_src | mean_bucket × 4 = 128 × 4 = 512 (T=4096) / 256 × 4 = 1024 (T=8192) |

Workspace 总 size: 7.47 GB (T=4096) / 14.95 GB (T=8192) per rank。
packed_outbox 占大头 — 8 dst × 32 epg × max_recv × 7168 × bf16 = **1.79 GB** (T=4096) / **3.58 GB** (T=8192) per rank。

#### Routing skew 实测 (DSv3 vs uniform)

| 指标 | dsv3 (T=4096) | uniform (T=4096) | dsv3 (T=8192) | uniform (T=8192) |
|---|---|---|---|---|
| per-expert global count skew (max/mean) | 1.08x | 1.08x | 1.06x | 1.06x |
| per (src_rank, dst_local_e) bucket max | 165 | 161 | 310 | 307 |
| distinct dst ranks per token (mean) | **3.96** | 5.29 | **3.96** | 5.29 |
| distinct dst ranks per token (min, max) | 2, **4** | 2, 8 | 2, **4** | 2, 8 |

DSv3 group-limited top-k 把每 token 强制限制到 ≤ 4 个 dst rank (`group_topk=4`),
uniform random 没这个约束所以 distinct 平均 5.29 / 最大 8。**但聚合到 wall 上几乎没差**:

| Config | M1c-A dsv3 vs uniform | M1b dsv3 vs uniform |
|---|---|---|
| T=4096 | 4.725 / 4.751 = -0.5 % | 3.632 / 3.655 = -0.6 % |
| T=8192 | 9.410 / 9.372 = +0.4 % | 7.491 / 7.486 = +0.1 % |

→ Routing distribution 在 T ≥ 4096 batch 下被 LLN 平均掉了,**dispatch wall 跟 routing skew 几乎正交**。
这也是为什么 toy bench 用 uniform 仍然能拿到 production-质量的结论 — 至少在 dispatch 阶段。
(GEMM grouped-GEMM 阶段可能不同, M1c-B 之后再看。)

#### Per-rank load balance 也好得反常

| Config | per-rank dev wall skew (min / max) |
|---|---|
| T=4096, dsv3, M1c-A | 0.992 |
| T=4096, dsv3, M1b   | 0.989 |
| T=8192, dsv3, M1c-A | 0.992 |
| T=8192, dsv3, M1b   | 0.990 |

→ 0.99 = 几乎完美均衡。受益于:
1. DSv3 sigmoid router 本身就是 load-balance-friendly (sigmoid + group_topk 限制了 hot spot)
2. T=4096 / rank 是 4K token,过 router 之后每 dst 收到 ~ T × group_topk / num_ranks = 2048 events,LLN 把 per-rank stddev 压到 ~1%

(这跟 toy size T=2048 ne=32 的 skew = 0.987 一致 — 不是 size 决定的,是 batch size 决定的。)

## 2. L2 cliff 假设, 重新审视

[上一篇 note](./2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md) §1 第 3 段:

> 根本原因 — M1b 的 receiver scatter read 不是 bug, 是 **topk dedup 红利**: peer 的 input_token_buf
> 只有 T·H·2 = 28 MB 完整落在 MI355X 的 32 MB L2 里, topk=8 意味着同一行被 8 个 dst rank 各读一遍,
> 第 1 个 peer 读完后剩下 7 个全是 L2 命中, …

这个解释 **在 toy size 下成立, 在 production size 下成立不了**。

#### 实测对比 (input_token_buf 与 L2 容量)

| size 档 | input_token_buf (per rank) | L2 cap (MI355X) | input fits L2 ? |
|---|---|---|---|
| toy: T=2048, H=7168 | 28 MB | 32 MB | **yes** |
| **DSv3 prod: T=4096, H=7168** | **56 MB** | 32 MB | **no** |
| DSv3 prod: T=8192, H=7168 | 112 MB | 32 MB | no |

production 下 `input_token_buf` 已经 1.75× L2 容量。如果"L2 dedup 红利"是 M1b 唯一优势,M1c-A 在
production 应该几乎不输 (因为 M1b 也 L2 thrash 了)。但 M1c-A 还是 +25-30% 慢。

#### 真实损失从哪来 (production size)

M1c-A 的固有 overhead **跟 L2 fit 无关的部分**:

| 项 | 含义 | T=4096 字节量 |
|---|---|---|
| (a) Phase B 写 packed_outbox | 每 token 复制 topk 次到 dst 通道 | T · topk · H · 2 = 4096·8·14336 = **470 MB** local HBM 写 |
| (b) Receiver 拉 packed_outbox | XGMI 拉每个 dst 的 slot | num_ranks · 4096 · 14336 = **3.76 GB** total inbound (468 MB/rank inbound) |
| (a+b) 总 | 比 M1b 多一次 HBM roundtrip | ~ 940 MB extra traffic per rank |

M1b 同样配置: receiver 直接拉 peer.input_token_buf,**peer 那边只有 56 MB total**,topk=8 重读
跨多个 dst rank 拉同一行,实际 inbound traffic 是 num_ranks · T_per_src · group_topk · H · 2 =
8 · 4096 · 4 · 14336 = 1.84 GB total → 每 dst rank inbound ~ 230 MB。

→ M1c-A 比 M1b 多约 **2× peer-side inbound XGMI traffic** + **额外的本地 HBM 写**。 XGMI 是 critical path,
2× 流量在带宽 bound 区直接 → 2× wall。 实测只回归 +30% 而不是 +100%,大概率是因为:
1. M1c-A 的 inbound 100% 连续 → coalescing 比 M1b 的 strided 好,平均每字节用更少 cache line transactions
2. M1c-A 的 Phase B 本地 HBM 写跟 Receiver XGMI 读在不同 stream/CU 上有并行

但这两个 win 一起也没盖过 "多搬一遍数据" 的硬支出 → 净 +30%。

### 结论:**M1c-A 的根本病不是 L2 capacity miss, 是架构上多搬了一遍数据**。

修法仍然是 M1b 路线 (在 sender 端不动数据,只在 src_index_table 元数据上排序),让 receiver 的散列
读自带 src_t 单调性 → prefetcher 友好,但 peer 那边只搬一份 input_token_buf。

## 3. 工程 takeaway

| 教训 | 落地 |
|---|---|
| **toy bench 不能解释 production behavior** — toy 给的"L2 cliff"叙述在 prod 不对。 user 那句"toy 容易忘记最终误导优化"是字面正确。 | bench_dispatch / bench_super_dispatch 的 default 已经切到 DSv3 production (ne=256, T=4096, dist=dsv3),后续优化只看这个数字。 |
| **DSv3 router skew 在 T ≥ 4096 batch 下被 LLN 平均掉** — dispatch wall 跟 routing distribution 几乎正交。 | 之后写 router-stress test 应该用 small batch (T ≤ 512) 才有机会观察到 skew 的真实影响。 |
| **per-rank load balance ~0.99** — 单节点 EP dispatch 不会被 router 拖垮。 | 之后跟 baseline 对比时,不用担心 dispatch 端 imbalance;真正的 imbalance 风险在 expert-grouped GEMM。 |
| **clean cmake state matters** — bisecting 的时候 toggling `-D` flag 不重新 cmake / 不删 build 会留 stale .o,导致诡异的 multi-iter 崩溃假象。 | 写一个 `scripts/clean_rebuild.sh` 把 `rm -rf build && cmake && make` 包起来,toggle USE_PACKED_OUTBOX 时强制走它。 |
| **DispatchLds 的 epg 上限 32 是 DSv3 的精确临界值**, 没余量。 | 已经把 `DispatchLds::counts[kMaxRanks*4]` 改成 `counts[kMaxLocalExperts]` (=64, 2× headroom),后续 EP=4 (epg=64) 配置不会静默踩界。 |
| **packed_outbox 大缓冲触发的真问题是 multi-GB workspace 的稳定性**, 不是 L2 capacity。 | 接下来 M1c-B 的 receiver-side sort 不需要 packed_outbox,workspace 可以省回 1.79 GB / rank。 |

## 4. 下一步 (M1c-B 重排)

M1c 路线收紧成两步:

### Step M1c-B: receiver-side small-sort

- Receiver 在跑 cooperative_b128_copy 之前,把 `src_index_table[e][src_rank][0..n_slots-1]` 按 `src_t` 升序排
- LDS 排序 (n_slots ≤ 1024 in production),`block_m=32` 分块的话每块只需要 ~32 元素,bitonic sort 拿 4-5 个 cycle
- 不引入新 cross-rank 同步, peer 端 input_token_buf 仍然 28 MB-56 MB,L2 行为不变
- 期望收益: HBM burst aggregation,3-5 % wall 改善

### Step M1c-C: combined LDS staging

- 多个 sub_wg 共用 LDS,先把要拉的 src_t 行 prefetch 到 LDS, 再 cooperative_b128_copy 出去
- 减少 XGMI bound 的暴露面 (LDS 在 wave-level 持久化,XGMI 在 WG 边界外不可见)
- 期望收益: 把 inbound XGMI 从 critical path 上挪到 LDS 排队,~10 % wall 改善

### NOT M1c-A: sender-pack

- 已经 2 次验证 (toy +50%, production +25-30%),无救方案 — 不再尝试

## 5. 提交清单

代码侧 (没新增 file,纯调整 default & 1 个 helper):
1. `benchmarks/bench_dispatch.hip`, `benchmarks/bench_super_dispatch.hip`:
 default 切到 DSv3 prod (ne=256, T=4096, dist=dsv3);加 `max_recv_factor` CLI 参数和 routing skew 打印;集成 `bench_routing::generate_routing_table` 路由生成
2. `benchmarks/bench_routing.h` (新): host-side DSv3 router (sigmoid + group_limited_topk) + uniform 选项 + skew stats 工具
3. `csrc/include/rocmoe/types.h`: 新增 `kMaxLocalExperts = 64` 常量
4. `csrc/include/rocmoe/dispatch_body.h`: `DispatchLds::counts[]` / `base[]` 大小用 `kMaxLocalExperts` 代替 `kMaxRanks*4`,留 2× headroom
5. `scripts/bench_dsv3_sweep.sh`, `scripts/bench_dsv3_full.sh`: 4-point sweep 脚本 (T=4096/8192 × dist=dsv3/uniform)

测试:
- `ctest` 8/8 PASS (test_dispatch 等 correctness 测试不变,小 size 跑得快)
- Bench DSv3 prod size 单次 / 多 iter 都稳定,no HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION

数字对照 (this round):

| 配置 | M1b dev wall (ms) | M1c-A dev wall (ms) | regression |
|---|---|---|---|
| toy (ne=32, T=2048) | 1.568 | 2.354 | +50 % |
| **DSv3 prod (ne=256, T=4096, dsv3 router)** | **3.632** | **4.725** | **+30 %** |
| DSv3 prod (ne=256, T=4096, uniform) | 3.655 | 4.751 | +30 % |
| DSv3 prod (ne=256, T=8192, dsv3) | 7.491 | 9.410 | +26 % |
| DSv3 prod (ne=256, T=8192, uniform) | 7.486 | 9.372 | +25 % |

→ M1c-A 决定保持 revert。 M1c-B 开始动手。
