# 2026-05-21 18:30  M2 [BASELINE] — 持久 super-kernel 脚手架 (Phase 0) 落地

> 时间: 2026-05-21 17:00 → 18:30 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, mi355-gpu-7), SLURM job 27091.batch
> 容器: xiaoming-dev (podman, ROCm 7.2 / PyTorch 2.10)
> 上一节点: M1b UP (`kSubWGs 4→8`, DSv3 dispatch device wall 1.940 → 1.567 ms = -19 %) [2026-05-21_1700_UP_m1b_dispatch_subwg_widening.md]
> 代码: `csrc/include/rocmoe/super_kernel.h` (新, 53 LOC) + `csrc/super_kernel.hip` (新, 154 LOC) + `csrc/launcher.{h,cpp}` (取消 `launch_on_device` placeholder) + `tests/test_super_kernel_skeleton.hip` (新, 178 LOC) + `CMakeLists.txt`
> 本节点 flag: **`BASELINE`** (没有 perf 优化, 这一篇是后续 M2-D / M2-G / M2-FC2 / M2-COMB 各 sub-round 的对照基线)

## TL;DR

把 RocMoE-bak 的 Phase 0 super-kernel skeleton 移过来 —— 一个 persistent grid, 按 WG idx 把 256 个 WG 分成 4 个角色 (DISPATCH / GEMM / FC2_PUSH / TAIL_COMBINE), 每个角色现在还是 no-op stub, 但 5-phase 的 barrier 序列 (B1 / B4 / B5 跨 rank, B2 / B3 内部) 已经按设计文档 §5.1 接好。`test_super_kernel_skeleton` 8 ranks × 4 次背靠背 launch 全 PASS, 跨 rank PhaseBarrier ping-pong 没有残留, per-WG `clock64()` 在 5 个桶上单调。脚手架自身开销 ≈ **170 µs / launch** (每个 cross-rank `phase_barrier<8>` + 围着它的 `grid_sync<>` 约 55–60 µs), 占 M1b standalone dispatch wall (1.57 ms) 的 ~11 %, 给后面 4 个 sub-round 的真实角色 body 留出 >85 % 时间窗口。

## 1. 这个 milestone 为什么存在

M1 结尾, receiver-pull dispatch 作为 **standalone launch** 停在 1.57 ms / T=2048, 被 XGMI peer-read 带宽夹住。M1b 收尾时已经说过: "剩下的差距是 standalone launch 模型本身的属性, 等 M2 的 persistent super-kernel 把它溶解掉。"

溶解的办法是把 dispatch + FC1 + FC2 + push + combine 装进 **同一个 persistent grid**, 让:

- dispatch 的 peer-read 跟 FC1 的 GEMM burst 在 chunk 级 overlap (`cco-pipeline-overlap` skill 原则 3: minimal barriers);
- 5 个 cross-rank phase barrier 只守住绝对必要的全局同步点 (`DESIGN.md §5.1`);
- per-pool-block `l1_arrival_count` 提供 rank 内细粒度 scoreboard, 让单个 GEMM block 在自己的 pool block 写满那一刻立刻起 GEMM, 不等全局 dispatch 完成。

但 overlap 跑起来的前提是 **脚手架自身先正确**: persistent grid 要按 `kNTotalWGs = 256` 起来, 每个 block 正确分到 4 个角色之一, 3 个 cross-rank PhaseBarrier 按顺序触发, 不 hang。M2 BASELINE 就是把这一层证明出来。

## 2. 落地的代码

- `csrc/include/rocmoe/super_kernel.h` — 公共 ABI:
  - `SuperKernelArgs` POD (96 B, layout v0): `sym` (SymBuffer), `barrier_sig` (PhaseBarrierSignal, 指向 sym base), `num_ranks`, `smoke_iters`, `cycle_log`。
  - `void launch_super_kernel(const SuperKernelArgs&, hipStream_t)`。
- `csrc/super_kernel.hip` — Phase 0 skeleton:
  - `__global__ rocmoe_super_kernel(SuperKernelArgs)` —— 角色分配 + 5-phase barrier 骨架。
  - 4 个 `stub_*_body()` 被调函数, 每个是一个小 `noop_busy_wait(smoke_iters)`。
  - `log_cycle()` 辅助, 把 per-`(blk, bucket)` 的 `clock64()` 时间戳写入 host 可读的 buffer (每个 WG 8 个 bucket)。
- `csrc/launcher.{h,cpp}` — `launch_on_device(int device, args, stream)` 这个 convenience wrapper 现在解开注释 (M1 阶段是 placeholder)。
- `tests/test_super_kernel_skeleton.hip` — Phase 0 完整 smoke test: 分配 symmetric workspace + cycle_logs, 在 N 个 rank 上 launch, sync, 校验 per-WG `clock64()` 在 B1 < B4 < B5 桶之间单调, 重复 4 次以触发 PhaseBarrier ping-pong 的残留处理。
- `CMakeLists.txt` — `super_kernel.hip` 进 static lib, `test_super_kernel_skeleton` 成为第 7 个 ctest 入口。

## 3. Smoke test 在校验什么

在 8 个 MI355X 上, 单进程多设备:

1. 每个 rank 分配 256 B symmetric workspace, 全 zero (这刚好够装一个 `PhaseBarrierSignal` 在 offset 0 —— Phase 0 kernel 只会摸这一段)。
2. 每个 rank 分配一份 cycle_log: `kNTotalWGs × kCycleNumBuckets = 256 × 8 = 2048 long long` slot。
3. 跑 `repeat=4` 次:
   - 清空所有 cycle_log;
   - 在每个 device 上 launch `rocmoe_super_kernel`;
   - 8 条 stream sync;
   - 读回 rank-0 的 cycle_log, 校验 per-WG 在桶迁移 0→1, 1→4, 4→5, 5→6 上的单调性 (bucket 含义: `start` / `after_B1` / `after_B4` / `after_B5` / `end`)。
4. 把 3 个可见 stage 间隔的 per-WG 中位数 (per-CU `clock64()` 差值) 打印出来作诊断。

## 4. 实测结果

热身后 (iter ≥ 1, 过完 JIT + 首次 XGMI fabric warm-up), 单进程多设备, 8x MI355X:

| smoke_iters | total / WG          | StageA→B1          | StageB→B4          | B4→B5              |
|-------------|---------------------|--------------------|--------------------|--------------------|
| 0           | ~360k cyc (~170 µs) | ~115k cyc (~55 µs) | ~125k cyc (~60 µs) | ~115k cyc (~55 µs) |
| 100         | ~2.0M cyc (~1.0 ms) | ~926k cyc (~440 µs)| ~450k cyc (~215 µs)| ~700k cyc (~330 µs)|

(换算用 MI355X SCLK ≈ 2.1 GHz。AMD GPU 上 `clock64()` 是 per-CU 的, 所以 within-WG 差值是良定义的; cross-WG min/max 的 clock64 **没有意义** —— 早期诊断版本这么做了, 拿到无意义的 425e9-cycle "总和", 已经改成只做 per-WG。)

读数告诉我们:

- 每个 cross-rank `phase_barrier<8>` + 围着的 `grid_sync<>` ≈ **55 – 60 µs** 纯开销 (`smoke_iters=0` 状态)。
- 整个脚手架总成本 (3 个 cross-rank phase barrier + 6 个 grid_sync) ≈ **170 µs / super-kernel launch**。
- 跟 M1b standalone dispatch (1.57 ms) 比, **持久 grid 脚手架占一个 dispatch wall 的 ~11 %** —— 留出 >85 % 的时间给真实 role body 去做有用的 overlap。
- `iter=0` 有一次性 ~100 ms 热身 (首次 XGMI exchange + JIT)。之后所有 launch 稳定在上表的稳态。
- 4 次背靠背 launch 在 ordering 上都 PASS —— PhaseBarrier ping-pong (counter % 4 → phase, sign) 正确消除了 forward 之间的残留。证实 cross-rank sync 对 re-entry 鲁棒, 这是 super-kernel 必须的属性, 因为它每个 forward 被调用一次。

## 5. 为什么这一篇 flag 是 BASELINE 不是 UP

Phase 0 还没干 MoE 的事。角色 body 是 stub, 没有 GEMM, 没有 dispatch peer-pull, 没有 combine。所以现在没东西可以跟 PT+RCCL 或 M1 比。它确立的东西是:

- 一个稳定的 launch ABI (`SuperKernelArgs` / `launch_super_kernel` / `launch_on_device`), 接下来 4 个 M2 sub-round 都按这一份继承, 不再动;
- 一个 1 秒就能跑完的 smoke harness (`test_super_kernel_skeleton`), barrier ordering 一旦出 bug 立刻被抓到, 给后面 sub-round 留出快速安全网;
- 一个 quantitative 脚手架成本 (~170 µs), 后面看 end-to-end wall 推理时间去向时可以减掉这一段。

这就是 BASELINE 的全部。第一组真的值得 UP / DOWN 比较的数字会在 M2-D 出现 —— 那时 real receiver-pull dispatch body 进 persistent grid (替换 `stub_dispatch_body`), 跟 M1b 的 1.57 ms standalone-launch 数字打。

## 6. 本轮遵循的工程规则

按 `rocmoe-dev-loop` skill:

1. **在 cluster 节点上、container 内 build/run** —— 全程走 `bash scripts/dev_on_node.sh build / test / raw "..."`, 没有 host 侧编译, 没有跳出 container。
2. **Per-phase test case** —— `test_super_kernel_skeleton` 是 M2 自己的 phase test, 跟 M0 的 GEMM tests + M1 的 `test_dispatch` 一起组成 7 个 phase test, ctest 端到端 3.7 s。
3. **flag 之前必须先 test** —— 这一轮 3 次 build / 3 次 ctest 全绿, 才推进 commit; 没有 7 个 phase test 全绿就不推进。
4. **note 上 explicit flag** —— 即本篇。Flag = `BASELINE`。按 skill 约定, BASELINE 是唯一一个独立成立的 flag (不跟自己 UP/DOWN 比), 后续 M2 sub-round 跟它比, 也跟 M1b 比。

## 7. 下一步 (M2 sub-rounds, 严格顺序)

每一轮都按 M1 那一套走 build / ctest / note loop, 期待跟上一轮拿 UP 或 FLAT (DOWN 就回退):

1. **M2-D**: 把 `stub_dispatch_body` 换成 M1b 的 receiver-pull body, 直接复用 `csrc/include/rocmoe/dispatch_body.h`, 不 fork 一份实现。预期 vs M1b standalone 拿 `UP`: dispatch 和 FC1 之间的 cross-rank PhaseBarrier 把原本 host 侧那条单独 kernel launch 折成 persistent grid 内的一次 B1, 消掉 ~50 µs 的 launch + sync gap。
2. **M2-G**: 把 BF16 GEMM (M0 已 verbatim 移植的 `mfma_tile.h`) 接进 `stub_gemm_body`, 用 `l1_arrival_count` scoreboard 做 per-pool-block work-stealing。这是 chunk 级 dispatch / FC1 overlap 真正开始 fire 的点。
3. **M2-FC2**: FC2 epilogue + push (peer 写 combine slot 到 dst rank 的 TAIL_COMBINE 输入区)。当前先 inline 进 GEMM epilogue; 后续如果 profile 表明拆出来更快, 再分给 FC2_PUSH 角色。
4. **M2-COMB**: TAIL_COMBINE 拉 8 个 peer slot, 按 topk 权重 reduce, 产出最终 output token。
5. **M3**: 跟 PT+RCCL reference 做 end-to-end 正确性比对; **只有这一步过了, 1 ms dispatch 目标才在它真实的 (overlapped) 工况下变得 measurable**。

## 8. 本轮触碰的文件

- `csrc/include/rocmoe/super_kernel.h` (新, 53 LOC)。
- `csrc/super_kernel.hip` (新, 154 LOC)。
- `csrc/include/rocmoe/launcher.h` — 取消 `launch_on_device` decl 注释。
- `csrc/launcher.cpp` — 实现 `launch_on_device`, 删掉 M1 的 placeholder 块。
- `tests/test_super_kernel_skeleton.hip` (新, 178 LOC)。
- `CMakeLists.txt` — `super_kernel.hip` 进 lib + 加 `test_super_skeleton` ctest。

## 9. 复现

```bash
bash scripts/dev_on_node.sh build
bash scripts/dev_on_node.sh test
bash scripts/dev_on_node.sh raw "build/test_super_kernel_skeleton 8 100 4"
bash scripts/dev_on_node.sh raw "build/test_super_kernel_skeleton 8   0 4"
```

## 10. 相关文件

- 上一节点 (M1b UP): [`2026-05-21_1700_UP_m1b_dispatch_subwg_widening.md`](./2026-05-21_1700_UP_m1b_dispatch_subwg_widening.md)
- M1 BASELINE: [`2026-05-21_1630_BASELINE_m1_dispatch_ported.md`](./2026-05-21_1630_BASELINE_m1_dispatch_ported.md)
- M0 BASELINE: [`2026-05-21_1330_BASELINE_m0_mfma_tile_ported.md`](./2026-05-21_1330_BASELINE_m0_mfma_tile_ported.md)
- 架构设计 (5-phase + barrier 序列): [`2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md)
- skill: `~/workspace/slab/.cursor/skills/rocmoe-dev-loop/SKILL.md`, `~/workspace/slab/.cursor/skills/cco-pipeline-overlap/SKILL.md`
- super-kernel 源参考: `~/workspace/RocMoE-bak/csrc/super_kernel*.hip`, `~/workspace/MonolithEP/csrc/super_kernel.hip`
