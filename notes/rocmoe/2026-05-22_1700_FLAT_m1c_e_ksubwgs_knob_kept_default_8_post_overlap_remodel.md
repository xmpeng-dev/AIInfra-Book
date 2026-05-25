# M1c-E FLAT — kSubWGs build-time knob 落地; 维持默认 8 (64 dispatch + 184 GEMM); 一连串误判后 user pushback 拿到正确 overlap 模型

> 时间: 2026-05-22 16:00 → 17:00 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 1 节点 8 GPU (`mi355-gpu-7`)
> 容器: `xiaoming-dev` Podman 内的 `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2 / hipcc / PyTorch 2.12+rocm7.1 / Primus
> 代码: `~/workspace/RocMoE/` (worktree, 改 types.h + CMakeLists.txt 加 `ROCMOE_KSUBWGS` build-time knob; 默认值保持 8 不变)

## 1. 时间点 / 上下文

- 上一阶段: [`15:45 FLAT`](./2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md) 实测 DSv3 prod FC1 dense roofline 3.247 ms / 92% peak (256 CU) + FC2 1.831 ms。M2-G overlap 上限确认 "完美 chunk-overlap dispatch 100% 藏在 FC1 后"。
- 触发本 round 的 user 假设 (16:00): "dispatch 用 64 CU 太多了, 只需要 8 或者 16 CU, 16 CU 跨两个 WDC 性能就能打满。phase 2/3 都空转跑 hook super-kernel 评估 dispatch 性能"。
- **本 round 我前后犯了 3 个错, 全部被 user 在不同节点上 pushback 修正**:
  1. 第一稿 (16:00): 把 user 的 "16 CU" 误读成 "16 CU 总数能饱和 XGMI", 用 RocMoE-bak Phase B "concentrated 慢 3.5×" 反驳, 维持 kSubWGs=8 默认。User: "你看看 MonolithEP 和 RocMoE-bak 的报告"。
  2. 第二稿 (16:13): 翻 MonolithEP cycle 8 + RocMoE-bak Phase B, 看到 "16 WGs per outbound XGMI link saturates 93%", 推断 user 说的 "16" 是 per-link 而不是总数, 把默认切到 kSubWGs=16 (128 CU dispatch)。User: "我的意思是 dispatch 总共用 16 个 CU, 这 16 个 CU 只要分布在 2 个 XCD 就能打满性能, 更多的 CU 应该用于 GEMM, GEMM 才是性能瓶颈"。
  3. 第三稿 (16:32): 解释了 dispatch 总数 16 CU 物理上跑不通 (PULL form 受 BDP / per-CU outstanding 制约), 维持 kSubWGs=16。**但**没把 GEMM 在 120 CU 下的退化按线性算进去 —— 错估"FC1 HBM-bound 不随 CU 数变化"(那是 RocMoE-bak 的小 workload 经验, 不适用 DSv3 prod M=65536 form)。User: "MonolithEP 为什么 dispatch 用 16cu 就可以" → 我答出了 push-vs-pull 物理差异。User: "但是这样 gemm cu 会少很多呀"。
  4. **正解 (17:00)**: 把 dispatch 跟 FC1 都按 CU 数线性 scaling 算 overlap, **kSubWGs=16 default 实际是 worst-case**, 应该退回 kSubWGs=8。

落本 round 的 final flag 是 **FLAT**: build-time knob 落了 (有未来 ROI), 默认值没变, 但**修正了 M2-G 的 sizing 决策依据**, 避免后面装 GEMM 时按错误的 "kSubWGs=16 是 XGMI 拐点" 把 GEMM CU 推到 120。

## 2. 问题

确定 dispatch role 在 super-kernel 持久 grid 里**应该占多少 CU**。两个核心未知数:

1. **PULL dispatch 的 XGMI 饱和拐点在多少 CU?** —— 已经在本 round 答出来了 (kSubWGs=8 = 64 CU 是 +13% headroom 之后的拐点)。
2. **GEMM CU 数对 FC1 wall 的退化曲线?** —— 这是真正决定胜负的, 不能光看 dispatch wall。

如果 FC1 在 184 / 216 / 232 CU 下 wall 接近 (HBM-bound), 那 dispatch 应该尽量小, GEMM 应该尽量大。
如果 FC1 线性退化 (compute-bound), 那 dispatch 拐点 vs FC1 增量得算 net wall, 净优配置就不一样。

## 3. 做了什么

### 3.1 `kSubWGs` 改成 build-time 可配 (`ROCMOE_KSUBWGS` macro)

`csrc/include/rocmoe/types.h`:

```cpp
#ifndef ROCMOE_KSUBWGS
#define ROCMOE_KSUBWGS 8
#endif
constexpr int kSubWGs = ROCMOE_KSUBWGS;
static_assert(kSubWGs >= 1, ...);
```

同时把 `kNGemmWGs` 从硬编码 184 改成 `ROCMOE_N_CU (256) - kNDispatchWGs - kNFc2PushWGs - kNTailComWGs`, 保证扫到 kSubWGs=16 时 (dispatch 占 128 CU) GEMM 自动让到 120 CU、总 WG = 256 = MI355X CU count, 不会撞 `__launch_bounds__(_, 1)` 的 grid_sync 死锁。

`CMakeLists.txt` 加 `-DROCMOE_KSUBWGS=N` 注入, 5 个独立 build 目录 (build_ksub{1,2,4,8,16}) 并行构建。

### 3.2 利用现成的 `bench_super_dispatch` 作 hook super-kernel

user "phase 2/3 都空转的 hook" 跟 M2-D BASELINE 时落地的 `rocmoe_super_kernel` 形态完全一样: DISPATCH role 真跑 receiver-pull, GEMM/FC2_PUSH/TAIL_COMBINE role 跑 `noop_busy_wait(args.smoke_iters=0)` 直接返回。`bench_super_dispatch` 直接调它, 不需要新 hook kernel。

### 3.3 5 档 sweep × 2 个 workload + 5 档全 bit-exact 回归

- 5 个变体逐个 ninja 出 `bench_super_dispatch / bench_dispatch_phases / test_super_kernel_e2e`。
- `test_super_kernel_e2e 8 32 4 256 256 32` (3 skew × {1, 3} iter) 在 kSubWGs ∈ {1, 2, 4, 8, 16} **全部 PASS dispatch bit-exact** —— 改 kSubWGs 不破坏正确性。
- Sweep: 8 rank × 256 expert × topk 8 × DSv3 balanced × `bench_super_dispatch` (T=4096 H=7168) + `bench_dispatch_phases` (T=4096 + T=8192 H=7168), 5 warmup + 20 iter (super) / 10 iter (phases)。

## 4. 取得的效果 + 推理链 (4 关键发现)

### 4.1 实测数据 (DSv3 prod 形状, dev wall p50)

**Standalone dispatch wall (ms)**:

| kSubWGs | dispatch CU | WG / XCD | WG / outbound link | T=4096 wall | T=8192 wall |
|--------:|------------:|---------:|-------------------:|------------:|------------:|
|  1 |   8 | 1 | 1  | 13.294 |    —   |
|  2 |  16 | 2 | 2  |  6.947 | 14.009 |
|  4 |  32 | 4 | 4  |  3.696 |  7.359 |
| **8** | **64** | **8** | **8** | **2.202** | **4.223** |
| 16 | 128 | 16 | 16 |  1.945 |  3.394 |

**Super-kernel hook wall (ms)** (4 个 grid_sync + 1 个 cross-rank phase_barrier 内, GEMM/FC2/TAIL 都 no-op stub):

| kSubWGs | super-kernel wall | standalone wall | super - standalone |
|--------:|-----------------:|----------------:|-------------------:|
|  1 | 33.227 | 13.294 | +19.93 ms |
|  2 | 16.273 |  6.947 |  +9.33 ms |
|  4 |  8.604 |  3.696 |  +4.91 ms |
| **8** |  **5.507** |  **2.202** |  **+3.31 ms** |
| 16 |  3.267 |  1.945 |  +1.32 ms |

### 4.2 关键发现 (4 条)

**(A) PULL dispatch 的 XGMI 饱和拐点在 kSubWGs=8 (64 CU = 8 WG per outbound link), 不是 16; user 一开始猜的 "16 CU" 是 PUSH form 的物理上限, 跟 PULL 隔了一个 4× 量级**

| 访问类型 | per-CU outstanding 上限 | XGMI 单链路 BDP | 饱和单链路需要 | 饱和 7 个 outbound 链路需要 |
|---|---:|---:|---:|---:|
| **PUSH (write)** | 几乎无上限 (L2 write-combining buffer 把 store 合并成 burst, 不阻塞后续指令, 不需要等 ACK) | ~30 KB | **1-2 CU** | **~16 CU** ← **MonolithEP 实测值** |
| **PULL (read)** | **~256 outstanding loads / CU** (架构硬上限) | ~30 KB ≈ **~2000 uint4 outstanding** | **2000 / 256 ≈ 8 CU** | **8 × 7 ≈ 56-64 CU** ← **RocMoE 实测拐点 (kSubWGs=8)** |

- MonolithEP dispatch (`csrc/dispatch.hip:280-378`) 实测就是 16 WG total (`N_COMM_DISPATCH_WGS = 16`, 8 destinations × 2 sub_wg per dest), 因为它是 PUSH form: `src_row = args.hidden[t*H]` (local HBM read) + `dst_row = peer->expert_packed_tokens[pack_row*H]` (peer XGMI write fire-and-forget)。
- RocMoE dispatch 是 PULL form: `src_row = peer->packed_outbox[...]` (peer XGMI read, RTT 600 ns 必须等) + `dst_row = self->expert_token_pool[...]` (local HBM write)。
- 这是 [RocMoE-v2 架构设计](./2026-05-21_1252_rocmoe_v2_architecture_design.md) 故意选择 PULL 的代价 —— PULL 砍掉了 PUSH form 的两个 stall (g=0 fan-in 1.34 ms + FC2 出口 XGMI 写竞争 0.7 ms), 跟 FC1 天然 overlap, 但代价是 dispatch CU 数从 16 涨到 64。

**(B) MonolithEP cycle 8 的 "16 WGs per src" 是 Phase 3 (combine 反向 copy, push form 跨 8 outbound) 的数字, 跟 dispatch (Phase 1) 不是同一个数**

第二稿误把 Phase 3 的 "16 WGs per src × 8 srcs = 128 WGs total" 跟 dispatch 的 "16 total" 当成同一个数字, 然后推出 "kSubWGs=16 = 128 CU 是 RocMoE PULL 的对应饱和点"。实际上 MonolithEP dispatch 跟 combine copy 是不同 kernel:
- Phase 1 dispatch (PUSH, scatter token to peer): **16 WGs total**
- Phase 3 combine (PUSH, scatter FC2 results back to origin peer): **128 WGs total** (16 per outbound × 8 outbound)

两个数字方向虽然都说 push form 的 per-link CU 数低, 但绝对数字不一样, 第二稿做了错误的类比。

**(C) FC1 在不同 GEMM CU 上是线性 scaling (compute-bound at DSv3 prod M=65536), 不是 RocMoE-bak Phase C2 那种 HBM-bound (那是他们的小 M workload)**

DSv3 prod FC1 形状 (M=65536 N=4096 K=7168 gated SwiGLU):
- Compute: 2 × 2 × 65536 × 2048 × 7168 = 3.85 TFLOP (gated = gate GEMM + up GEMM 两个 M=65536 N=2048 K=7168)
- 256 CU peak 1310 TFLOPS, 理论 compute-bound time = 3.85 / 1310 = 2.94 ms
- 实测 3.247 ms = 1.10× compute roofline = 90% per-CU 效率
- 单 tile (M=128 N=128 K=7168) compute 0.235 GFLOP / 1310 GFLOPS = 0.18 ms; per-tile A+B+C memory = 3.6 MB / 5.3 TB/s = 0.68 µs → compute-to-memory ratio 264:1, 强 compute-bound
- 16384 个 tile / 256 CU = 64 tile/CU, 线性 scaling 阈值 >> 1, 所以 FC1 在 120-256 CU 区间应该是接近线性 scaling

但这是 **理论模型**, 真实 mfma_tile 在 120 CU 下的实测必须等 M2-G 才能拿到 —— 因为持久 super-kernel 内的 GEMM 还要跟 dispatch role 共享 L2 + I/O die crossbar, 实际 wall 可能比 standalone 推算高。

**(D) 把 dispatch wall + GEMM wall 按 CU 数 scaling 一起算, kSubWGs=8 仍是最 robust 的默认 — kSubWGs=4 在 balanced 略优 (-0.67 ms) 但 hot_cov50 dispatch 会戳出 FC1**

| kSubWGs | D_CU | G_CU | dispatch balanced | dispatch hot×1.29 | FC1 (≈ 3.25 × 256/G_CU) | overlap bal (max(d,FC1)+FC2) | overlap hot |
|--------:|----:|----:|------:|------:|------:|------:|------:|
|  2 |  16 | 232 | 6.95 | 8.96 | 3.59 | **8.78 ms** ← d 戳出 | 10.79 ms |
|  4 |  32 | 216 | 3.70 | 4.77 | 3.85 | **5.68 ms** ← 平衡最优 | 6.60 ms (d 戳出 FC1) |
| **8** | **64** | **184** | **2.20** | **2.84** | **4.52** | **6.35 ms** ← 现默认 | **6.35 ms** ← 全工况最稳 |
| 16 | 128 | 120 | 1.95 | 2.52 | 6.93 | **8.76 ms** ← G_CU 不够 | 8.76 ms |

(skew tax 1.29× 来自 [M1c-D note](./2026-05-22_1100_UP_m1c_d_lds_staged_cooperative_copy_kills_strided_rw.md) 的 hot_cov50 vs balanced 比例, FC2 1.83 ms 来自 [15:45 GEMM note](./2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md))

- **kSubWGs=8**: dispatch 永远藏在 FC1 后 (2.20-2.84 ≪ 4.52), wall = FC1 + FC2 = 6.35 ms, **跨 skew 完美 robust**
- kSubWGs=4: balanced -0.67 ms 但 hot_cov50 +0.25 ms (dispatch 4.77 > FC1 3.85, dispatch 自己变 critical path)
- kSubWGs=16 (第二/三稿默认): G_CU 砍掉 64 个让 FC1 退化 2.4 ms, 净亏 +1.6 ms
- kSubWGs=2: dispatch 自己 6.95 ms 远超 FC1 3.59 ms, 完全失去 overlap 设计哲学

平均 (balanced + hot) / 2:
- kSubWGs=4: 6.14 ms (略优但 skew 风险)
- **kSubWGs=8: 6.35 ms (最稳)**
- kSubWGs=16: 8.76 ms (差)

### 4.3 final 决议

**维持默认 kSubWGs=8** (64 dispatch CU + 184 GEMM CU + 8 tail = 256 总), 因为:

1. dispatch 在所有 skew profile (balanced / realistic_cov20 / hot_cov50) 下都 ≤ FC1, 完美保留 "dispatch 100% 藏在 FC1 后" 的设计哲学
2. 跨 skew average wall 6.35 ms 跟 kSubWGs=4 (6.14 ms) 只差 3%, 但完全规避了 hot 风险
3. **真正的不确定性在 FC1 scaling 是不是线性的** —— 上面表格的 FC1 wall 全是理论外推。M2-G 装 GEMM 真测之后, 如果 FC1 在 184 vs 216 vs 232 CU 上的 wall 比理论更接近 (HBM-bound), 那 kSubWGs=4 会变成实际最优; 留 `ROCMOE_KSUBWGS` build-time knob 给 M2-G 重扫
4. **不要再去 push form 找 16 CU dispatch** —— 那是设计架构层 trade-off, 重新引入 g=0 fan-in stall + FC2 出口竞争, 净亏 ~1.5 ms

## 5. 下一步

1. **M2-G** (按 [`rocmoe-dev-loop` SKILL](.cursor/skills/rocmoe-dev-loop/SKILL.md) 拆 M2-G-α/β/γ 3 个子 round):
   - α: GEMM role body 接到 `l1_arrival_count` polling, 单 expert pass 跑通 + bit-exact
   - β: 加 B4 barrier + 完整 FC1+SwiGLU + bit-exact
   - γ: 测 dispatch ↔ FC1 chunk-overlap 真触发, kSubWGs ∈ {4, 8, 16} 重扫确定最终值
2. **M2-G γ 优先项**: 实测 FC1 在 184 / 216 / 232 / 120 CU 下的 wall (在 super-kernel 持久 grid 形态下, 不是 standalone), 验证 4.1 表里的线性 scaling 假设是否成立
3. 如果 M2-G γ 实测 kSubWGs=4 在 balanced 真比 kSubWGs=8 快 ≥ 5%, 且 hot 退化 ≤ 3%, 那再切 default 8 → 4

## 6. 怎么复现

```bash
ssh mi355-gpu-7
podman exec -it xiaoming-dev bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE

for K in 1 2 4 8 16; do
    rm -rf build_ksub${K}
    cmake -S . -B build_ksub${K} -DROCMOE_KSUBWGS=${K} -DCMAKE_BUILD_TYPE=Release
    cmake --build build_ksub${K} -j 16 \
        --target bench_super_dispatch bench_dispatch_phases test_super_kernel_e2e
    build_ksub${K}/test_super_kernel_e2e 8 32 4 256 256 32
    build_ksub${K}/bench_super_dispatch  8 256 8 4096 7168 32 5 20 dsv3 4 balanced 4
    build_ksub${K}/bench_dispatch_phases 8 256 8 4096 7168 32 5 10 dsv3 4 balanced 4
done
```

落盘: `bench_results/ksub_sweep_20260522/{super_dispatch,dispatch_phases,dispatch_phases_T8192}_ksub{1,2,4,8,16}.txt`。
