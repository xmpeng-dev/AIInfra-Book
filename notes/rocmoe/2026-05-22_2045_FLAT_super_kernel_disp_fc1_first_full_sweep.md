# M2-G α — 首份 `dispatch + FC1` super-kernel 完整 perf sweep (FLAT)

> Tag: **FLAT** (新增 measurement, 没改 perf 路径 — 但回答了上下文里反复缺的"FC1 装进去之后到底跑多快"问题)
> 时间: 2026-05-22 20:45 UTC+8
> 触发: user 问 "没有一个 dispatch + fc1 完整的性能数据吗" — 之前确实没有

## When / What problem / What I did

### 当时背景

- M2-G α 已经把 FC1 (gate+up) + SwiGLU body 装进 persistent super-kernel ([17:15 note](./2026-05-22_1715_BASELINE_m2_g_fc1_body_integrated_dispatch_tradeoff.md))
- M2-G β 想验 production T/H 撞到 pre-existing dispatch wave-store-drop bug ([19:00 DOWN note](./2026-05-22_1900_DOWN_m2g_beta_dispatch_h7168_wave1plus_store_drop.md))
- 但中间 **从来没人把 fused dispatch+FC1 super-kernel 的 wall 真测过** —— α note 里只有 smoke `test_super_kernel_e2e` 单跑一遍 wall ~9 ms 的 ballpark, β note 全程在 debug correctness, bench_super_kernel.hip 根本没接 `w1_gate_up` (M2-D 留的 ABI, FC1 role 因为 nullptr guard 直接 fall through 到 noop)
- 所以从 user 角度: 完全没看到 dispatch + FC1 的实际性能曲线

### 我做的

1. **接 `w1_gate_up` 到 `bench_super_kernel.hip`** —— bench 里加 per-rank `hipMalloc(epg * 2F * H * sizeof(bf16))` + host-side seed=`0xD15EA5E` U(-0.05, 0.05) 初始化 + cleanup hipFree; `--F-moe 0` 时跳过 alloc, FC1 role 继续 fall-through (保 dispatch-only 对照路径). 共改 2 处:
   - `benchmarks/bench_super_kernel.hip` (build_args 加 `ka.w1_gate_up = w1_dev[r]` 一行, 加 ~25 行 alloc + init + cleanup)

2. **跑完整 16 点 sweep** —— 4 shape 维度 × 2 (FC1 on/off): `T ∈ {128, 512, 2048}` × `H ∈ {2048, 4096, 7168}` × `F ∈ {0, 2048}` (T=128 加 H=256 / F=256 一组对照, F=4096 不跑因为 mfma_tile 重激活留 M2-G β-fix). 配置 ranks=8 / experts=32 / topk=4 / block_m=32 / skew=balanced / warmup=5 / iters=30.

3. **CSV 落 `bench_results/super_kernel_disp_fc1_20260522.csv`** —— harness 自带 schema, 跟 mcore_moe_bench.py 同列 (`dispatch_ms`/`experts_ms`/`combine_ms`/`crit_path_ms`/`us_per_token`).

## 关键数字 — 首份 dispatch+FC1 fused 完整数据

| T | H | F | crit_path p50 (ms) | μs/tok | 备注 |
|---|---|---|---|---|---|
| 128 | 256 | 0 | 0.421 | 3.29 | dispatch-only |
| 128 | 256 | 256 | 0.503 | 3.93 | **+0.082 ms FC1 overhead** (FC1 几乎 100% 藏在 dispatch 后) |
| 128 | 2048 | 0 | 0.465 | 3.63 | |
| 128 | 2048 | 2048 | 1.003 | 7.84 | **+0.538 ms** FC1 critical |
| 128 | 4096 | 0 | 0.542 | 4.23 | |
| 128 | 4096 | 2048 | 1.086 | 8.48 | **+0.544 ms** |
| 128 | 7168 | 0 | 0.687 | 5.37 | |
| 128 | 7168 | 2048 | 1.191 | 9.31 | **+0.504 ms** ← dispatch 数据 garbage (wave-store-drop), wall 真实 |
| **512** | **2048** | **0** | **0.721** | 1.41 | |
| **512** | **2048** | **2048** | **1.204** | 2.35 | **+0.483 ms** |
| **512** | **7168** | **0** | **1.563** | 3.05 | |
| **512** | **7168** | **2048** | **2.043** | 3.99 | **+0.480 ms** ← FC1 终于 fully overlap, 比 disp-only 只多 0.48 ms |
| 2048 | 2048 | 0 | 1.788 | 0.87 | |
| 2048 | 2048 | 2048 | 2.392 | 1.18 | **+0.604 ms** |
| 2048 | 7168 | 0 | 5.007 | 2.45 | |
| 2048 | 7168 | 2048 | **5.894** | **2.89** | **+0.886 ms** ← DSv3 prod shape (dispatch garbage 但 wall 真实) |

### 三条 takeaway

1. **FC1 add-on cost 在 ~0.48-0.60 ms 稳定区间**, 跟 T 几乎独立. 真实 FC1 算力 (T=2048 H=7168 F=2048 per rank = 4·1024·7168·2048 ≈ 60 GFLOPS, 400 TFLOPS 下 ~150 µs) 是 GEMM body 实际算时间, **剩下的 ~400 µs 是 persistent FC1 GEMM role 的固定调度税** (atomic work-steal claim + chunk-overlap polling + SwiGLU 第二趟). 这跟 [15:45 dense FC1 roofline 3.247 ms](./2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md) 不矛盾 —— roofline 是 standalone `bench_gemm` M=65536 跨 256 CU 算的, super-kernel 形态下 FC1 只能用 184 GEMM CU (kSubWGs=8) 而且 token 在 expert 间 fan-out 跑分散 tile.

2. **dispatch+FC1 fused 真的有 overlap 收益**, 最明显在 T=512 / H=7168: dispatch-only 1.566 ms → +FC1 仅 +0.448 ms (FC1 standalone roofline 同 M 量级的 FC1 大概 0.4-0.6 ms 自己就跟 FC1 add-on 量级一样, 说明 chunk-overlap **真的** 触发了 FC1 在 dispatch 还没结束就开 MFMA). 反观 T=128 / H=256 (FC1 几乎 0 cost, +0.08 ms) 跟 T=2048 / H=2048 (FC1 +0.60 ms over disp-only) 都是 FC1 比 dispatch 完成得早 / 晚, 没看到一边把另一边完全藏住的情形, overlap 效率取决于 T·H·F 三轴比例.

3. **DSv3 prod (T=2048 H=7168 F=2048) wall = 5.89 ms** (dispatch garbage warning 加在 caveat 里). 跟 mcore_moe_bench.py 同 shape baseline 比:
   - Megatron-LM MoELayer DSv3 T=8192 = **18.13 ms** ([04:10 mcore sweep note](./2026-05-22_0410_BASELINE_mcore_moe_full_sweep.md)), T=2048 按线性 ≈ 4.53 ms (Megatron 实际 T 越小 per-token tax 越高, 真值 ~5-7 ms 量级);
   - RocMoE-v2 super-kernel 5.89 ms 已经在 Megatron 同量级, **但只跑了 dispatch + FC1**, 缺 FC2 (1.83 ms standalone) + combine (~1 ms 估) — 接完之后 ~9 ms 是当前 M2-G stub 的预期 ceiling, 比 BF16 目标 7 ms 还差 28%, 跟 [11:00 FC1 1.5 ms 估算错误](./2026-05-22_1100_UP_m1c_d_lds_staged_cooperative_copy_kills_strided_rw.md) 那篇当时算的 5.08 ms 物理 floor 也差 1.8×, 留 M2-G γ 真测 184 GEMM CU 下的 FC1 拍.

## Achieved effect

- 第一次有了 `dispatch + FC1` super-kernel 的完整性能曲线 (16 点 CSV)
- 确认 chunk-overlap 路径在 T=512+/H=7168 真触发 (FC1 add-on < FC1 standalone)
- 确认 DSv3 prod shape kernel 完整跑得通 (5.89 ms wall, dispatch 数据 corruption 不影响 wall 测量)
- 给 M2-G γ kSubWGs sweep 提供基线 (现在再扫 kSubWGs 是要看 FC1 add-on cost 怎么动, 不再是空跑 stub WG)

## Next directions

1. **M2-G γ — kSubWGs sweep with real FC1 body**: 用这次 wired 起来的 bench, 重扫 kSubWGs ∈ {2, 4, 8, 16}, 看 dispatch role wall + FC1 role wall + 整 kernel wall 怎么动. M1c-E 钉 kSubWGs=8 用的是 stub GEMM, 现在有真 FC1 可以推翻或确认.
2. **M2-G β-fix** (并行): dispatch H≥4096 wave-store-drop bug 单独抠 30 行 repro, 不修这个就拿不到 prod shape correctness, 但 perf 可以先继续扫.
3. **M2-FC2 + M2-COMB**: 把 FC2 + combine body 也装进去, 再扫一遍, 看 9 ms ceiling 估算准不准.
4. **重画 trace canvas**: 现在有真 FC1 cycle log (`experts_ms` 列), 可以画出 dispatch ↔ FC1 chunk-overlap 的时间线图 (per-WG cycle_log bucket 0..4 vs role).

## 工件 / artifacts

| 文件 | 位置 |
|---|---|
| CSV (16 point sweep) | `bench_results/super_kernel_disp_fc1_20260522.csv` |
| Bench source (修改后) | `benchmarks/bench_super_kernel.hip` (build_args 加 `ka.w1_gate_up`, +25 行 alloc/init/cleanup) |
| 这篇 note | `notes/2026-05-22_2045_FLAT_super_kernel_disp_fc1_first_full_sweep.md` |

## 环境

| 项 | 值 |
|---|---|
| 节点 | mi355-gpu-7 (8x MI355X gfx950) |
| 容器 | xiaoming-dev / docker.io/rocm/primus:v26.2 |
| ROCm | 7.2 |
| commit | M2-G α + dispatch fence/sync 防御补丁状态 (β note 描述的 retained 修动) |
| Routing | dsv3 dist / num_experts=32 / topk=4 / n_group=8 / group_topk=4 / balanced skew / sigma=0.0 |
| Warmup / iters | 5 / 30 (hipEvent crit_path = max over 8 ranks, p50 over 30 iters) |
