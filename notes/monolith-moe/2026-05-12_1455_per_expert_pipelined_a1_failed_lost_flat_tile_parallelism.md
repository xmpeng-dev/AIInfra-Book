# A1 Per-expert Pipelined Compute 失败 + flat-tile 并行度复盘

> 时间: 2026-05-12 14:55 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 8× AMD Instinct MI355X (gfx950, XGMI 全互联), node `mi355-gpu-26`
> 容器: xiaoming-dev (Podman)
> 软件: ROCm 7.2 / hipcc, gfx950 codegen
> 代码: `csrc/fused_moe_super_kernel.hip` @ `f50bb43` (P2 baseline，本 note 完成时 A1 已 revert)

## 1. 时间点 / 上下文

- 上一次进展：P2「Batched (e, src, mi, ni) compute + L2 reuse」把 DSV3 SPARSE 从
  8.80 → 7.95 ms（1.11×，首次反超 PyTorch baseline），TILE-FIT 5.07 → 4.31 ms
  （1.18×）。主要 gain 来自 barrier 数从 24 → 3，而不是 L2 reuse。
- 触发本次工作：P2 profile 显示 TILE-FIT 中 `dispatch_src_ready_wait` 占 wall ~50%；
  Scatter 仍未与 Compute 真正 overlap。直觉：「按 expert 触发 compute」能在更细
  粒度上把 scatter / compute 错开。

## 2. 问题

让 compute 在某个 expert `e` 的 8 路 token 全部到齐后就立刻启动该 expert 的 FC1/FC2，
不必等所有 expert 的所有 source 都 scatter 完。

- 现状：DSV3 SPARSE 7.95 ms / 363.1 TFLOPS（P2 收尾，ratio=0.18）
- 目标：DSV3 SPARSE ≤ 7.2 ms；TILE-FIT ≤ 3.8 ms（假设 scatter ↔ compute overlap
  能换回 ~10% wall）

## 3. 做了什么

### 设计

1. `MoeIpcWorkspace` 新增 `dispatch_expert_arrive[epg]`：每路 source scatter 完
   `(dest, e)` 这一对的所有 token 后，原子 `+1`。
2. `multi_wg_scatter_phase`：在每条 pair 末尾 `atomicAdd_sys(&peer->dispatch_expert_arrive[e], 1)`。
3. `expert_compute_phase` 重写为「按 expert 串行 pipeline」：
   - 外层 `for (e = 0; e < epg; ++e)`：
     - tid0 自旋等到 `self->dispatch_expert_arrive[e] == NUM_GPUS`；
     - 第一个 expert 时一次性 publish `combine_expert_offsets` 给所有 peer；
     - 构 8 条 (src) pair 描述符 → FC1 → barrier → FC2 + SwiGLU → barrier →
       per-source copy → barrier → signal `combine_expert_ready[rank*epg + e]`。
   - 每个 expert 内部 3 个 barrier，共 `3 × epg` 个 compute barrier。

实现细节同 prompt 中描述：保留 P2 的 cooperative tile load + small-tile dispatch，
仅修改 「外层调度顺序」 和 「触发条件」。

### Bench

```
# A1 ratio sweep 2026-05-12T06:51:42Z  (mi355-gpu-26)
# DSV3_SPARSE (tokens/gpu=512 H=7168 F=2048 epg=32 topk=8)
ratio=0.15  latency_ms=24.85
ratio=0.18  latency_ms=23.89
ratio=0.20  latency_ms=23.39
ratio=0.25  latency_ms=22.13
ratio=0.30  latency_ms=21.64   ← A1 best

# TILE_FIT (tokens/gpu=1024 H=4096 F=1024 epg=4 topk=4)
ratio=0.15  latency_ms=5.03
ratio=0.18  latency_ms=4.87
ratio=0.20  latency_ms=4.74
ratio=0.25  latency_ms=4.56    ← A1 best
ratio=0.30  latency_ms=5.63
```

50/50 correctness PASS（功能正确，性能崩溃）。

## 4. 结果

| 场景         | P2 best | A1 best | A1 vs P2 |
|--------------|---------|---------|----------|
| DSV3 SPARSE  | 7.95 ms | 21.64 ms| **+172%（回退 2.72×）** |
| TILE-FIT     | 4.31 ms | 4.56 ms | +5.8%（小回退）|

**结论：A1 完败。已 revert 到 P2 (`f50bb43`).** raw log:
`benchmarks/results/a1_pipelined_bench.txt`。

## 5. 复盘 / 经验

A1 设计有两个互相加成的错误，是 P1 同款误判的「升级版」：

### (a) 错误地放弃 P2 的 flat (e, src, mi, ni) 并行度

P2 的 hot loop 是「全部 (e, src, mi, ni) tile 摊给所有 compute WG 同步抢」。在
192 个 compute WG × 256 个 src_expert_pairs × small64 tile 切分下，**每个 WG 实际
在做 5–8 个 tile，整个 compute 阶段一次扫描就结束**。3 个 barrier，barrier 之间
没有 idle。

A1 把外层换成 `for e ∈ [0, 32)`，每个 expert 只剩 8 条 src tile（DSV3 SPARSE
每对 ~16 token / m_tile=64 → 1 个 mi × 16 个 ni = 16 tile/pair，总 128 tile/expert）。
192 个 WG 抢 128 个 tile → ~30 个 WG 在 FC1，其余 idle；FC2 同理。**并行度从
192-wide 摊到 ~30-wide，等同于 6× 浪费 compute CU**。

进一步：每个 expert 都要单独跑「FC1 → barrier → FC2 → barrier → copy → barrier
→ signal」，共 `3 × 32 = 96` 个 cross-WG barrier，比 P2 的 3 个多了 32×。每个
barrier 即便只多 50 µs，总开销也 ~5 ms。

两个因素相乘：~3× 计算时间 + 96 个额外 barrier ≈ 实际看到的 2.7× 回退。

### (b) 错误地以为 scatter 是临界路径

`dispatch_src_ready_wait` 占 50% wall **不意味着 scatter 是瓶颈**。在 P2 里，
所有 192 个 compute WG 同时在 `wait_flag → flat tile dispatch`，wait 是
**和 compute 并行的硬等待时间**，不是 compute 之前的 lead time。降低 wait 时间
等价于 「让 scatter 先 finish」，但 scatter 已经吃满 comm_ratio 的 38 个 comm WG，
继续给它更多 WG 反而抢走 compute。

正确的诊断：scatter ↔ compute 已经 overlap，wait 期间 compute WG 在 spin 是
**功率浪费**而非时间浪费。要继续优化，需要从 「让 wait 期间 compute WG 干活」
或 「scatter 总时间」 入手，而不是 「让 compute 按 expert 触发」。

### (c) 和 P1 的同源教训

| 维度       | P1 (tail per-pair)            | A1 (per-expert compute)       |
|------------|-------------------------------|-------------------------------|
| 错误诊断   | tail kernel_total 79% wall ⇒  | dispatch_src_ready_wait 50%   |
|            | 认为 tail 是 critical path    | ⇒ 认为 scatter 是 critical    |
| 实际原因   | wall = max(per-WG total)，    | spin 期间 compute WG 已 occupy|
|            | tail WG 数太少，但其他 WG idle| CU；问题是 「该干啥的人没干啥」|
| 修复方向   | 增加 tail WG / 内层 chunk     | 让 compute WG 在 spin 内做活  |
| 共同陷阱   | profile metric 是 「相对 wall 的比例」，不是 「critical path 因果」 |

## 6. 下一步（修正后）

正确的「按 ROI 排序」候选：

1. **A2 (M-concat tile dispatch, true L2 reuse)**: 仍保留 P2 的 flat tile，但把
   8 路 source 在 dispatch 前 「逻辑拼接」成 `M_concat = Σ_src T_e,src` 的大 M，
   让一个 tile 同时跑多 source，硬件层 weights cache hit 翻倍。需要 gather
   pre-pass 或 row-LUT。Risk: 引入新 kernel 阶段 / 增加 fc1/fc2 scratch。
2. **Comm-Compute Decoupling**: 让 compute WG 在 `wait_flag` spin 时主动做
   pre-fetch（`__builtin_prefetch` weights of next pair）或参与 scatter slot
   filling。需要 LDS 共用方案。
3. **降 scatter 总时间**: TILE-FIT 的 scatter 仍占 ~50% wall；可以从 「pack +
   scatter 合并到 1 个 kernel pass」、「per-rank XGMI link 并行」、「skip empty
   pair early」入手。

A2 是结构性 gain 最大的，作为下一个 P0。

## 7. 状态

- 代码：`csrc/fused_moe_super_kernel.hip` 已 revert 到 P2 (`f50bb43`)
- 测试：`test_super_kernel_correctness` 50/50 PASS（revert 后）
- Bench：DSV3 SPARSE 7.95 ms / TILE-FIT 4.31 ms（baseline 保持）
- Raw log：`benchmarks/results/a1_pipelined_bench.txt`，`benchmarks/run_a1_bench.sh`
