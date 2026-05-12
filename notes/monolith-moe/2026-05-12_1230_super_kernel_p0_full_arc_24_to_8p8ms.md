# Super-Kernel P0 全程：DSV3 SPARSE 24.0 → 8.80 ms（项目里最大的一跃）

> 时间: 2026-05-12 12:30 (Asia/Shanghai)（最后一步 `8d00b73` 落地于 2026-05-11 21:47 -0500 = 2026-05-12 10:47 SH，bench 跑完 ~12:40 SH）
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, XGMI 全互联), node `mi355-gpu-26`
> 容器: xm-mlperf
> 软件: ROCm 7.2 / hipcc, gfx950 codegen
> 代码: 跨 4 个 commit — `dd9dcfb` → `31fb54c` → `4c56ab0` → `8d00b73`

## 1. 时间点 / 上下文

- 上一篇相关 note：[`2026-05-08_ck_implementation_deep_dive.md`](./2026-05-08_ck_implementation_deep_dive.md)
  （把 CK 1050T vs MMOE 530T 的 2× gap 拆成 6 项叠乘优化，估算闭合路线）。
- 这篇 note 是 **回溯归档**（retrospective）：P1/P2 已经写过，但 P0 的 4 步
  优化散落在 2026-05-09 ~ 2026-05-12 之间 4 个 commit 里（部分 commit message
  是 `update`），需要补一篇完整记录。
- 触发：用户 2026-05-12 13:56 SH 提醒「24.0 → 14.04 → 8.80 那段是项目里最大
  的一跃，补一篇」。

## 2. 问题

把刚修完一致性 bug、回到 **24.0 ms** 的 super-kernel 推到 PyTorch+RCCL
基线（8.466 ms）量级。

- 现状（2026-05-09 起点）：DSV3 SPARSE 24.0 ms / 120 TFLOPS，比 PyTorch+RCCL
  慢 **2.83×**。"compute-comm overlap" 项目本身没意义。
- 目标：先 ≤ 14 ms（追平 1 GPU 单卡 6.26 ms × 2 的下界感觉），最终 ≤ 9 ms。
- 卡点：
  1. 一致性补丁（`__ATOMIC_ACQ_REL` + `__threadfence`）从 "broken-but-fast"
     18.4 ms 退回正确的 24.0 ms，**回归 5.6 ms 必须从别处赚回来**。
  2. `compute_phase_barrier` 每个 (src, e) 增 3 次，DSV3 epg=32 × NUM_GPUS=8
     × 3 = **~600 次跨 WG agent atomic / launch**，L2 严重打。
  3. sparse MFMA padding：T_e≈16，tile=64 → 25 % row util，tile=128 → 12.5 %。
  4. tile=128 + flat 不开时，`num_n_tiles=32 ≪ num_compute_wgs=128`，
     **96 个 WG 在 barrier_1 上 idle 8.55 ms**。

## 3. 做了什么

四步串起来 24.0 → 8.80 ms，每一步都先 `test_super_kernel_correctness` 50/50
PASS 才接着量 perf：

### Step 1 — `wgs_per_cu=1 ratio=0.25` sweep（24.0 → 15.21 ms）

`dd9dcfb` 2026-05-09 15:10 SH。原来的窄 sweep 默认 `WGS_PER_CU=2 ratio=0.20`
来自一致性补丁前 "broken-but-fast" 时代的参数。展开 sweep 到
`WGS_PER_CU ∈ {1, 2}`、`ratio ∈ {0.15, 0.20, 0.25, 0.30}`：

- compute WG 数从 **410 → 192**：每次 `compute_phase_barrier` 等的增量减半，
  L2 atomic 队列长度也减半。
- in-flight tile 并行只有原来一半，但 DSV3 SPARSE 在 T_e=16 下根本撑不满
  410 WG，所以损失可以忽略。

**Δ = −8.79 ms**（占整段最大单步收益，回收了一致性补丁的全部退化）。

### Step 2 — Per-(src, e) wait_flag → per-src counter（15.21 → 14.86 ms）

`31fb54c` 2026-05-09 15:39 SH。compute WG 不再 poll `epg=32` 个独立的
`(src, e)` flag，scatter 侧改成 dispatch 完最后一个 (src, e) 时 bump 一次
`dispatch_src_ready[NUM_GPUS]`（共 8 个 counter）。

- 192 个 compute WG 并行 poll 的 atomic 流量从 192 × 32 = **6144 路** 降到
  192 × 8 = **1536 路**。
- 编码上 `multi_wg_scatter_phase` 收尾处把 last-scatter detection 改成
  per-src 计数到 epg。

**Δ = −0.35 ms**。

### Step 3 — Per-src compute-barrier coalescing（14.86 → 14.04 ms）

`4c56ab0` 2026-05-09 16:32 SH。最大的结构性改动：把 `expert_compute_phase`
里 per-(src, e) 的 3-段 barrier（FC1 / SwiGLU / copy）合并成 per-src 的 3 段
（FC1 across all e / FC2+SwiGLU(fused) across all e / copy across all e）：

```cpp
// before: for (src) for (e) { fc1; bar; fc2; bar; copy; bar }   // 3 × 32 × 8 = 768 cycles
// after:  for (src) { fc1_all_e; bar; fc2_all_e; bar; copy_all_e; bar }   // 3 × 8 = 24 cycles
```

实现关键：`fc1_scratch` / `fc2_scratch` 改成 `(e_start + m_start)` 寻址
（旧寻址只能存一个 e 的 output，新寻址让所有 e 的 output 并存），不再需要
phase 内复用 scratch。Adaptive SwiGLU 的 standalone 分支也被砍了（原本想用
extra barrier 换 expf 隐藏，新布局下这个 barrier 是 epg-many 个，永远输）。

- 跨 WG agent atomic：**~600 / launch → 24 / launch**（25× 缩减）。
- `__threadfence_system()` 调用：**~256 → 8**。

**Δ = −0.82 ms**。注意此时 small-tile 模板已经在源代码里但**没接入运行路径**
（commit message 里写「small-tile 4× 缩 tile cycles 但也 4× 缩 MFMA pipeline，
DSV3 上 net wash」—— 这个判断在 Step 4 被修正）。

### Step 4 — Flat (e, mi, ni) tile dispatch + adaptive small-tile（14.04 → 8.80 ms）★

`8d00b73` 2026-05-11 21:47 -0500 = 2026-05-12 10:47 SH。**单步收益最大**。两件事一起做：

**(a) Flat tile dispatch**：原来 compute WG 在每个 (src) 阶段内还是按 e 嵌
循环（`for e in epg: for tile in e: ...`），单个 e 的 `num_n_tiles=2F/N_TILE=32`，
而 compute WG 数 = 128 → **96 个 WG 没活干，全在 barrier_1 上 spin 8.55 ms**。

新写法：在每个 (src) 阶段开头先用一个 compute WG 在 LDS 里建好一张
`(e, mi, ni, kind, tile_offset)` 的全局扁平表（按 expert prefix sum），
然后所有 128 个 compute WG **跨 expert 同时领 tile**：

```cpp
for (int t = wg_id; t < total_tiles_this_src; t += num_compute_wgs) {
    auto (e, mi, ni, kind) = decode(t, flat_descriptor);
    if (kind == SMALL) mfma_gemm_tile_t<32, 128, 1, 4>(...);
    else               mfma_gemm_tile_t<M_TILE, N_TILE, 4, 4>(...);
}
```

`compute_barrier_1` 从 **8.55 ms → 0.29 ms**（30× 缩减）。

**(b) Adaptive small-tile**：`MOE_SMALL_TILE_T_E_MAX` 默认 32，sweep 后选 64
最优。对 `T_e ≤ MOE_SMALL_TILE_T_E_MAX` 的 expert，runtime 把 tile 切到
`mfma_gemm_tile_t<32, 128, 1, 4>`（M_TILE_SMALL=32, 1×4 wave）：

- T_e=16 + tile=128 → 12.5 % row util  
- T_e=16 + tile=64  → 25 % row util  
- T_e=16 + small (32×128) → **50 % row util**（×4 vs tile=128, ×2 vs tile=64）

修正了 Step 3 commit message 的判断：之前以为 small-tile 缩 MFMA pipeline 会
亏掉 LDS-load 隐藏，但**配合 flat dispatch 后**，128 个 WG 同时跑 small-tile
（每 WG 拿 8 tiles/src），MFMA 单元利用率反而提高。

**Δ = −5.24 ms**（DSV3 SPARSE），同时 TILE-FIT 也跟着拿到 5.17 → 5.07 ms。

### 步骤总表

| # | 改动 | Commit | DSV3 (ms) | Δ |
|---|---|---|---:|---:|
| 0 | 一致性补丁基线（ACQ_REL + threadfence, wgs/CU=2 ratio=0.20） | `141867c` | 24.00 | — |
| 1 | `wgs_per_cu=1 ratio=0.25` sweep | `dd9dcfb` | 15.21 | **−8.79** |
| 2 | Per-src wait_flag counter | `31fb54c` | 14.86 | −0.35 |
| 3 | Per-src compute-barrier coalescing（跨 e 合并） | `4c56ab0` | 14.04 | −0.82 |
| 4 | **Flat (e, mi, ni) tile dispatch + adaptive small-tile** | `8d00b73` | **8.80** | **−5.24** |
| Σ | | | | **−15.20（1.57× vs PyTorch+RCCL 9.62×）** |

> 注：表头的 9.62× 是 24.0 / 2.5 ms（hardware-ideal），即 P0 收尾前我们距硬
> 件下限还有 9.62×；P0 收尾后只剩 3.5×。

## 4. 效果

### 主指标（DSV3 SPARSE）

| 指标 | Step 0（24.0 ms） | Step 3（14.04 ms） | Step 4 = P0 终态（8.80 ms） |
|---|---:|---:|---:|
| Wall (ms) | 24.00 | 14.04 | **8.80** |
| 有效 TFLOPS | 120 | 205.6 | **327.9** |
| vs PyTorch+RCCL（8.466 ms） | 0.35× | 0.60× | **0.96×（−0.33 ms）** |
| FC1+FC2 总时间 (ms) | — | 11.19 | **6.11（−45 %）** |
| `compute_barrier_1` (µs) | — | 8 551 (tile=128) | **291（30× 缩减）** |
| 跨 WG agent atomic / launch | ~1200 | 24 | 24 |
| `__threadfence_system()` / launch | ~256 | 8 | 8 |

### Profile 分解（µs / WG / iter，`MOE_PROFILE=1`，DSV3 SPARSE）

| Phase | tile=64 pre-P0 (14 279) | tile=128 pre-P0 (19 969) | **P0 终态 tile=128 + flat + small64 (9 041)** |
|---|---:|---:|---:|
| dispatch_src_ready_wait | 861 | 832 | 920 |
| fc1_tiles               | 5 418 | 2 777 | **2 945** |
| compute_barrier_1       | 258 | **8 551** | **291** |
| fc2_swiglu_tiles        | 5 776 | 2 941 | **3 161** |
| compute_barrier_2       | 945 | 3 846 | 565 |
| copy_to_combine         | 322 | 203 | 291 |
| compute_barrier_3       | 671 | 549 | 420 |
| tail WG gather_combine  | 9 970 | 13 455 | **7 098** |

关键观察：
- tile=128 pre-P0 的 19.97 ms 几乎全是 `barrier_1 = 8.55 ms`（96 WG idle）。
  Flat dispatch 直接把这块吃干净。
- tile=64 pre-P0 的 14.28 ms 主要是 fc1+fc2 = 11.2 ms（sparse MFMA padding）。
  small-tile 把 fc1+fc2 砍到 6.11 ms（−45 %）。
- P0 让 tile=128 跑赢 tile=64 —— 在此之前 tile=128 因为 idle 问题始终是最差
  配置，"sweep 里 tile=64 最优" 是被 idle 掩盖的假象。

### TILE-FIT（对照组）

T_e=128 已经撑满 tile=128，**P0 对它影响很小**（5.17 → 5.07 ms，−0.10 ms）。
确认收益来自 sparse 路径，不是普适加速。

### 定性观察

- ✅ **DSV3 SPARSE 9.62× hardware-gap 缩到 3.5×**，进入 PyTorch+RCCL ±5 %
  的窄区间，从此 P1/P2 才有意义。
- ✅ `compute_barrier_1` 8.55 → 0.29 ms 是项目里**单次最大幅度的指标改善**，
  把"96 WG idle"这种结构性浪费彻底削掉。
- ✅ Step 3 关于 "small-tile 不香" 的早期判断被 Step 4 推翻 —— 单独看 small-tile
  确实 net wash，但 flat dispatch 之后 small-tile 才能真正 amortize。优化
  组合非可加。
- ⚠️ tile=128 deadlock 历史问题（commit `3370847`）通过 launcher
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` clamp 修好，否则 P0 这套
  config 跑不起来。
- ⚠️ P0 终态显式把 **tail WG `gather_combine_phase = 7.10 ms / 9.04 ms wall
  = 79 %**" 标为新瓶颈 —— **这个判断在 P1 阶段（2026-05-12 13:10）被证伪**，
  详见后续 note：tail kernel_total < compute kernel_total，wall 由 compute
  限制，不是 tail。

## 5. 可持续方向

**P0 之后已经做完的**（不重列，留个指针）：

- P1 tail pipelining 尝试 + critical-path 修正 → [`2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md`](./2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md)
- P2 batched (e, src, mi, ni) compute → [`2026-05-12_1335_batched_compute_p2_single_phase.md`](./2026-05-12_1335_batched_compute_p2_single_phase.md)
  （DSV3 8.80 → 7.95 ms，TILE-FIT 5.07 → 4.31 ms）

**P0 当时没做、留给未来的余量**：

| 优先级 | 方向 | 当时预估 | 现状 |
|---|---|---|---|
| ~~P0（当时）~~ | tail WG `gather_combine_phase` 加速 | −1.5 ms | **已证伪（P1）**，tail 不在 critical path |
| 仍存活 | LDS XOR swizzle 替换 PAD=4 | 解锁 prefetch 深度 | 未做，与 CK deep dive 路线对齐 |
| 仍存活 | 3-stage prefetch + Stream-K（grouped GEMM 内核侧） | 530T → 860T | 未做，super-kernel 内嵌路径上限 |
| 仍存活 | C-shuffle epilogue | +3–5 % FC1 | 未做 |

## 关键教训

1. **不要孤立评估优化**。Step 3 的 commit message 说 small-tile 在 DSV3 上
   net wash，把它锁在 source 里没接入；Step 4 把 small-tile 接进 flat
   dispatch 之后立刻拿到 −5 ms。优化组合非可加，**单独跑 micro-bench 容易
   做出错误判决**。
2. **sweep 默认值会过期**。一致性补丁之前的 `WGS_PER_CU=2 ratio=0.20` 是
   不正确语义下的最优；正确语义下 `WGS_PER_CU=1 ratio=0.25` 才是新最优。
   每次大补丁后必须重新 sweep，不能 trust 旧默认。
3. **WG 数 ≪ tile 数 时 flat dispatch 是结构性必选**。`num_n_tiles=32` /
   `num_compute_wgs=128` → 75 % WG idle 的场景，光调 ratio / tile 是徒劳，
   必须改 dispatch schedule 让 WG 跨 expert 同时领 tile。
4. **把"占比"和"critical path"分清楚**。P0 收尾时把 tail 79 % 标成下一步
   P0，是把 `phase_time / wall` 当成 `critical_path_share`。多 WG 类型
   kernel 里 wall = max(per-WG kernel_total)，phase_time 之和 = sum > wall。
   这个错误直接造成 P1 第一轮失败，**3 小时返工，应被项目级 lesson 收编**。

## 相关文件

- 代码 commit（按时间）：
  - `dd9dcfb` — `Tune super-kernel sweep defaults to wgs_per_cu=1 ratio=0.25`
  - `31fb54c` — `Coalesce per-(src, e) wait_flag into per-src counter`
  - `4c56ab0` — `update`（per-src compute-barrier coalescing）
  - `8d00b73` — `update`（flat tile dispatch + adaptive small-tile）
- Bench 日志：
  - `benchmarks/results/super_kernel_sweep.txt`、`super_kernel_sweep.prev.txt`
  - `benchmarks/results/p0_small_tile_bench.txt`、`p0_flat_bench.txt`、`p0_ratio_sweep.txt`
  - `benchmarks/results/p0_flat_profile.txt`、`moe_profile_dsv3.txt`
- README / PERF / SCENARIOS：见 `README.md`、`benchmarks/PERF_REPORT.md`（"History of how we got here" + Detailed Sweep Table）、`benchmarks/SCENARIOS.md`
- 上一篇 note：[`2026-05-08_ck_implementation_deep_dive.md`](./2026-05-08_ck_implementation_deep_dive.md)
- 后续 note：[`2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md`](./2026-05-12_1310_tail_pipelining_p1_failed_critical_path_correction.md) → [`2026-05-12_1335_batched_compute_p2_single_phase.md`](./2026-05-12_1335_batched_compute_p2_single_phase.md)
