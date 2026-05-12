# A2 M-concat tile dispatch 失败 + 并行度 > 局部性总结

> 时间: 2026-05-12 15:30 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, XGMI 全互联), node `mi355-gpu-26`
> 容器: xiaoming-dev (Podman)
> 软件: ROCm 7.2 / hipcc, gfx950 codegen
> 代码: `csrc/fused_moe_super_kernel.hip` @ `741bed8` (P2 baseline，本 note 完成时 A2 已 revert)

## 1. 时间点 / 上下文

- 前两次失败：P1 (tail per-pair, 2026-05-12 13:10) 和 A1 (per-expert pipelined compute,
  14:55) 都因「以为是 critical path / 以为能换 wall time」而回退。
- A1 复盘明确：**dispatch_src_ready_wait** 是 spin-during-compute，不是 lead time，
  缩短它不能换 wall。
- 本次 A2 的 motivation：保留 P2 flat (e, src, mi, ni) → 「concat 8 路 source 的 M
  维 → 同一 expert 的 weight (kk, ni) tile 只 load 一次（而不是 8 次）」。
  设想 4× HBM weight 流量下降 → DSV3 SPARSE 7.95 → ~6.5 ms。

## 2. 问题

让 expert `e` 的 8 路 source token 在 dispatch 前 **物理 gather** 成 concat layout
`concat_input[e][concat_row, H]`，然后 FC1/FC2 沿 (e, mi, ni) 平铺，每个 weight
(kk, ni) stripe 在一个 expert 内只被读一次。

- 现状：DSV3 SPARSE 7.95 ms / 363 TFLOPS（P2），TILE-FIT 4.31 ms（P2）
- 目标：DSV3 ≤ 6.5 ms；TILE-FIT ≤ 3.8 ms
- 假设：FC1 / FC2 weight HBM 流量 ~16 GB → ~2 GB（per layer per rank），HBM-bound
  阶段约缩短 4×

## 3. 做了什么

### 设计

1. `GemmLdsLayout` 加 `concat_e_start[epg+1]` (33 ints) + `concat_src_base[epg][NUM_GPUS]`
   (256 ints) + `concat_total` —— ~1.2 KB LDS。
2. `MoeScratchSizes` 加 `concat_input_bytes = NUM_GPUS * max_recv * H * 2 B`，约 3.6 MB
   per rank（DSV3 SPARSE）。
3. `expert_compute_phase` 重写：
   - **[3a]** 在 LDS 算 `concat_e_start[e]` + `concat_src_base[e][s]`（per-WG 私有，
     不需 cross-WG barrier）。
   - **[3b]** Gather pre-pass：对 256 个 (e, s) pair round-robin，把
     `dispatch_tokens[s_slab + src_off + i]` 拷到 `concat_input[concat_src_base[e][s] + i]`。
     uint4 (8 bf16) 向量 stride。
   - **[3c]** 新增 1 个 cross-WG barrier（gather → FC1）。
   - **[3d]** FC1：pair 描述符按 `epg` 索引（不再是 `epg*NUM_GPUS=256`），每个 tile
     的 M 维 = `T_e_concat = Σ_s T_e_s`（DSV3 SPARSE avg=128 → 正好填满 M_TILE=128）。
   - FC2 同上（rebuild num_n for N=H）。
   - **[7]** Per-src copy 重写：对 256 个 (e, s) pair round-robin，从
     `fc2_scratch[concat_src_base[e][s] + i]` 拷到 `peer[s]->combine_results[my_rank*max_recv + src_off + i]`。
4. `tests/smoke_super_kernel.hip` / `correctness` / `bench` 加 `concat_input` 分配 +
   launcher 透传。

总 cross-WG barrier 数：4（gather→FC1→FC2→copy→signal），P2 是 3。

### Bench

```
# A2 ratio sweep 2026-05-12T07:20:34Z  (mi355-gpu-26)
# DSV3_SPARSE (tokens/gpu=512 H=7168 F=2048 epg=32 topk=8)
ratio=0.15  latency_ms=9.10
ratio=0.18  latency_ms=10.13
ratio=0.20  latency_ms=9.03   ← A2 best
ratio=0.25  latency_ms=10.23
ratio=0.30  latency_ms=10.39

# TILE_FIT (tokens/gpu=1024 H=4096 F=1024 epg=4 topk=4)
ratio=0.15  latency_ms=7.28
ratio=0.18  latency_ms=7.30
ratio=0.20  latency_ms=7.12
ratio=0.25  latency_ms=6.75   ← A2 best
ratio=0.30  latency_ms=6.99
```

50/50 correctness PASS（功能正确，性能崩溃）。

## 4. 结果

| 场景         | P2 best | A2 best | A2 vs P2 |
|--------------|---------|---------|----------|
| DSV3 SPARSE  | 7.95 ms | 9.03 ms | **+13.6 %**（回退） |
| TILE-FIT     | 4.31 ms | 6.75 ms | **+56.6 %**（回退） |

**结论：A2 失败。已 revert 到 P2 (`741bed8` = post-A1-revert baseline).** raw log:
`benchmarks/results/a2_concat_bench.txt`。

## 5. 复盘 / 经验

### (a) M-concat 让 tile 数砍 8×，**192 个 compute WG 直接闲一半**

| 维度         | P2 small-tile (M_SMALL=32, N=128) | A2 concat (M_TILE=128, N=128) |
|--------------|---------------------------------|-------------------------------|
| 每 (s, e) 的 tile 数 (DSV3 FC1) | 1 mi × 32 ni = 32             | n/a |
| 每 e 的 tile 数 (DSV3 FC1)       | 8 srcs × 32 = 256             | 1 mi × 32 ni = 32 |
| epg=32 全部 tile 数              | 8192                          | 1024 |
| 每 WG 处理 tile (compute WGs=192) | 42.6 tile/WG                  | 5.3 tile/WG |

WG count = 192，A2 在 expert 内部 32 tile（≪192），导致 **大量 WG 在一个 expert 内
拿不到 tile**。round-robin 跨 expert 也无法弥补：A2 总 tile=1024，192 WG 各拿 ~5 tile，
WG 利用率从 P2 的 **42 tile/WG 摊销 barrier 成本** 跌到 **5 tile/WG 几乎全是 barrier**。

TILE-FIT 更惨：epg=4，A2 总 tile = 4 × 32 = 128 < 192 = WG 数。直接 **2/3 WG 空转**。
反映在 wall：4.31 → 6.75 ms（+56 %）。

### (b) L2 reuse 假设根本不成立

P2 note (2026-05-12 13:35) 已经写过 **「L2 weight reuse 假设证伪，真实收益来自
barrier 合并」**。当时的证据：P2 已经按 `pair_id = e*NUM_GPUS + src` 顺序处理 tile，
即 「同一 expert 的 8 个 source 连续被处理」。从硬件 L2 视角看，**P2 已经在隐式
做 weight reuse**：第一个 source 加载 `w1_e` 流过 L2，剩下 7 个 source 在 L2 热的
时候直接命中，不需要再走 HBM。MI355X infinity cache 256 MB，单个 `w1_e` ≈ 56 MB，
8 个 source 的间隔时间足够在 L2 命中区间内。

A2 多做的「物理 gather」其实是把 「L2 已经在做的事」 显式化，**没省 HBM 流量，反而
增加了 3.6 MB 写 + 3.6 MB 读 + 1 个额外 barrier**。这是 100% 浪费。

### (c) P1 / A1 / A2 三连同根错误：**parallelism > locality on 192 WGs**

| Failure | Mistaken hypothesis | 实际 root cause |
|---------|---------------------|----------------|
| P1 (tail per-pair)         | tail 79 % wall = tail is critical path | wall = max(per-WG kernel_total); compute is critical |
| A1 (per-expert pipelined)  | dispatch_src_ready_wait 50 % wall → scatter is lead time | spin-during-compute; per-expert serial loses 192 → 30 WG-wide parallelism |
| A2 (M-concat tile)         | per-source 8× weight re-read = HBM waste | P2 already L2-reuses across srcs; A2 collapses tile count 8× |
| **共同根**                  | 「重构 compute layout 一定有结构性 gain」 | **MI355X 上 compute 是 parallelism-bound，任何把 tile 数砍下来的改动都直接拉低 wall** |

P2 是这个硬件 + 这个 workload 下的 **「最大 tile-level 并行度 + 隐式 L2 reuse」** 的局部
最优点。把 (e, src, mi, ni) 任何一层从外向内合并（per-expert / M-concat / per-pair）都
减 tile 数 → WG 利用率掉 → wall 涨。

### (d) 决策修正：不要再动 compute 排布

剩余 wall 必须从 **「物理上更少的 work」** 来：

1. **HBM weight 流量减半** → FP8 / mxfp8 weights。weight 物理变小，不靠 layout 重排，
   不影响 tile 数。MI355X 原生支持 mxfp8。**预期 DSV3 FC1+FC2 5.86 → ~3 ms（−2.8 ms）。**
2. **Scatter 总时间下降**：TILE-FIT scatter 占 ~50 % wall。pack + scatter 合并为 1 个
   kernel pass；per-rank XGMI link 并行；skip empty pair early。**预期 TILE-FIT −1 ms。**
3. **GEMM 内核单点上限提升**（3-stage prefetch / Stream-K / LDS XOR swizzle / C-shuffle
   epilogue）：超越 530T → 860T 区间。这是 DSV3 4.8 ms 目标的前置依赖。

下一轮 P0 应该是 **(1) FP8 weights** 或 **(2) scatter 总时间**。彻底放弃 compute layout
重构方向。

## 6. 下一步（修正后）

新 P0 候选（**不再触 compute 排布**）：

1. **FP8 weights for FC1 / FC2**: storage 直接 ÷2，HBM 流量 ÷2，DSV3 FC 5.86 → ~3 ms。
   实现成本：需引入 weight quant pre-pass + mxfp8 MFMA dispatch。RoI 最大。
2. **TILE-FIT scatter 总时间下降**: pack+scatter 合并 / per-link XGMI 并行 / skip empty
   pair。目标 TILE-FIT −1 ms。
3. **3-stage prefetch + Stream-K**: GEMM 内核绝对 TFLOPS。详见
   [ck_implementation_deep_dive](./2026-05-08_ck_implementation_deep_dive.md)。

## 7. 状态

- 代码：`csrc/fused_moe_super_kernel.hip` 未 commit，已 `git checkout` 还原到 `741bed8`
- 测试：`test_super_kernel_correctness` 还原后立即 PASS
- Bench：DSV3 SPARSE 7.95 ms / TILE-FIT 4.31 ms（P2 baseline 保持）
- Raw log：`benchmarks/results/a2_concat_bench.txt`，`benchmarks/run_a2_bench.sh`
