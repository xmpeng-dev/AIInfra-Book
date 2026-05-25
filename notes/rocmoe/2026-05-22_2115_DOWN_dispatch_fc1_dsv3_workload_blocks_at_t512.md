# M2-G α — DSv3 真实 workload bench, T≥512 撞 hard crash (DOWN)

> Tag: **DOWN** (用了 user 指的 DSv3 真 workload 后, 19:00 DOWN 的 dispatch wave-1+ store drop 在 DSv3 高并发 peer-pull 下从 silent data corruption **升级成 hard crash (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION)**, 比之前估的更严重)
> 时间: 2026-05-22 21:15 UTC+8
> 触发: user "用 dsv3 的 workload" — 之前 20:45 FLAT sweep 用的是 experts=32 topk=4, 不是 DSv3 (真 DSv3 = experts=256 topk=8 n_group=8 group_topk=4)

## When / What problem / What did

### 当时背景

- 20:45 FLAT 拿了第一份 dispatch+FC1 fused 完整 perf sweep, 但用的是 experts=32 topk=4 (小 num_experts 配置, 不是 DSv3)
- user 指出要换成 DSv3 workload, 用 `--dist dsv3 --num-experts 256 --topk 8 --n-group 8 --group-topk 4`

### 我做的

1. **bench_super_kernel.hip 加 reset_counters** (host-side 多 iter 必备):
   - `l1_arrival_count` / `expert_recv_count` / `send_done_flag` / `fc1_work_counter` 是 atomic-add 累加, M2-COMB B5 没装就得 host 手动 zero
   - mirror `test_super_kernel_e2e::reset_counters` 用同步 `hipMemset` + 跨 rank `hipDeviceSynchronize`
   - **不要 zero `barrier_signal`** — counter 编码 PhaseBarrier ping-pong protocol (phase, sign) mod 4, 中途清零会破坏 cross-rank sync state 导致 H≥7168 偶发 illegal mem access; 这个坑 ~30 分钟才 isolate 出
   - 改 2 处: 警告 comment + lambda + warmup 循环和 iter 循环都加 reset call

2. **跑 DSv3 sweep**, 用 `--dist dsv3 --num-experts 256 --topk 8 --n-group 8 --group-topk 4 --H 7168 --F-moe 2048` 配 5 warmup × 30 iters × balanced skew

3. **发现 T≥512 全部 hard crash** (memory aperture violation), 跟 reset 是否启用无关, 跟 max_recv_factor 加大到 32 也无关 — 跑 5 iters 也死, 跑 1 iter 也死

## 关键发现 — DSv3 + 当前 dispatch kernel 跑不下去

### Crash matrix (DSv3: experts=256, topk=8, n_group=8, group_topk=4, balanced)

| T | H | 状态 | 说明 |
|---|---|---|---|
| 128 | 256-7168 | **PASS** | 唯一能跑的 T 档, 偶发 H=256 F=0 crash (重跑 3 次 2/3 通) |
| **256** | **2048-7168** | **CRASH** | bench_dispatch 直接死, bench_super_kernel 30 iters 死 |
| **512** | **256-7168** | **CRASH** | iters=1 也死, max_recv_factor=32 也死, F=0 也死 |
| **1024+** | **任意 H** | **CRASH** | 全部死 |

跟 19:00 DOWN note 文档的 dispatch wave-1+ store drop bug (H≥4096) 是同一类, 但 DSv3 (topk=8 + epg=32 让 peer-pull 并发度从 4×8 升到 32×8 = 8× concurrent) 让漂掉的 store **从写到 wrong slot (silent data corruption) 升级成写到 unmapped 物理页 (hard crash, kernel abort)**.

### DSv3 在 T=128 (能跑) 的 perf 数据

DSv3, T=128, balanced, crit_path p50, 30 iters:

| H | disp-only (ms) | disp+FC1 (ms) | Δ FC1 (ms) | μs/tok (disp+FC1) |
|---|---|---|---|---|
| 256  | 0.941 | 1.083 | +0.142 | 8.46 |
| 2048 | 1.102 | 1.960 | +0.858 | 15.31 |
| 4096 | 1.282 | 2.444 | +1.162 | 19.09 |
| **7168** | **1.720** | **3.238** | **+1.518** | **25.29** |

CSV: `bench_results/super_kernel_disp_fc1_dsv3_t128_20260522.csv`

### 跟 20:45 假 workload (experts=32 topk=4) 对比, 同 T=128 H=7168

| 工况 | disp-only | disp+FC1 | Δ FC1 |
|---|---|---|---|
| 假 (E=32 K=4) T=128 H=7168 | 0.687 ms | 1.191 ms | +0.504 ms |
| **真 DSv3 (E=256 K=8) T=128 H=7168** | **1.720 ms** | **3.238 ms** | **+1.518 ms** |
| 倍数 | 2.5× | 2.7× | 3.0× |

DSv3 即使在 T=128 这个小工况, dispatch 也比假 workload 慢 2.5× (epg 4→32 让每个 dispatch WG 要 sender-pack 8× 多的 (le, slot) 组合, 受 src_index_table / packed_outbox 占的 L2 footprint 影响). FC1 add-on 慢 3× (FC1 FLOPs 量翻倍 + persistent role 内部更多 atomic claim 竞争).

## 关键判断 — DSv3 prod 数据要先修 dispatch bug 才能拿

外推 DSv3 T=8192 (mcore baseline prod), 按 T=128→T=8192 是 64× scaling, 即使大 T 时 fixed cost 摊薄, 整 super-kernel wall 预估在 **20-40 ms** 区间. 跟 Megatron MoELayer DSv3 T=8192 = 18.13 ms 是同量级 / 稍慢, **但这只能纸上算 — 实测拿不到, kernel 在 T=512 就死**.

## Achieved effect

- 把 reset_counters 接到 bench_super_kernel 是 structurally 必备的 (M2-COMB B5 没装就得有 host 兜底), 拿到了第一份 DSv3 真 workload 的 T=128 wall 数据
- 印证了 19:00 DOWN note 钉的 wave-1+ store drop **不只是 H≥4096 老 hazard**, 而是会随并发度升级到 hard crash, **是 M2-G β-fix 的真实 P0** (不修就拿不到任何 DSv3 prod 数据)
- 验证 PhaseBarrier `barrier_signal.counter` 不能 host 端 zero (踩过这个坑可以让后面 M2-COMB B5 设计避开)

## Next directions

按重要性:
1. **M2-G β-fix (P0)** — 抠 30 行 dispatch repro 隔离 H≥4096 + epg=32 + topk=8 下的 wave-store-drop, 拍 ROCm. 不修这个就拿不到任何 DSv3 prod perf 数据 (T=8192 / T=2048 全部不能跑)
2. **M2-G γ kSubWGs sweep** — 必须等 β-fix 完了才能在 prod shape 上做; T=128 sweep 也能做但样本太小不代表 prod 行为
3. **M4 wave specialization** (绕开方案) — 把 dispatch 限制到 2 SIMD, GEMM 占 2 SIMD, 跨 SIMD 无 wave scheduler 竞争; 也能根治 wave-store-drop 因为不再混 stream

## 工件 / artifacts

| 文件 | 位置 |
|---|---|
| DSv3 T=128 CSV | `bench_results/super_kernel_disp_fc1_dsv3_t128_20260522.csv` |
| Bench (reset_counters 接入) | `benchmarks/bench_super_kernel.hip` (lambda + warmup + iter 循环各加一次 reset, 不动 barrier_signal) |
| 这篇 note | `notes/2026-05-22_2115_DOWN_dispatch_fc1_dsv3_workload_blocks_at_t512.md` |

## 环境

| 项 | 值 |
|---|---|
| 节点 | mi355-gpu-7 (8x MI355X gfx950) |
| 容器 | xiaoming-dev / docker.io/rocm/primus:v26.2 |
| ROCm | 7.2 |
| commit | M2-G α + 19:00 DOWN 残留的 dispatch fence/sync 防御补丁 + 20:45 FLAT bench_super_kernel 接 w1_gate_up + 本 note 加 reset_counters |
| Routing | dsv3 dist / num_experts=256 / topk=8 / n_group=8 / group_topk=4 / balanced |
| Warmup / iters | 5 / 30 (crit_path = max over 8 ranks 的 hipEventElapsed, p50 over 30 iters) |
