# FLAT — DSv3 prod FC1/FC2 dense roofline 实测 3.25 / 1.83 ms, 修正前两篇 note 错估的 1.5 ms, M2-G overlap 上限从 "藏 50%" 翻成 "藏 100%"

> 时间: 2026-05-22 15:45 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 1 节点 8 GPU (`mi355-gpu-7`)
> 容器: `xiaoming-dev` Podman 内的 `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2 / hipcc / PyTorch 2.12+rocm7.1 / Primus
> 代码: `~/workspace/RocMoE/` @ `6fa180f` (`Bootstrap RocMoE-v2 + M1c-D LDS-staged dispatch`)

## 1. 时间点 / 上下文

- 上一次相关进展: [`2026-05-22 11:00 M1c-D UP — LDS-staged cooperative_b128_copy + sender 连续分片`](./2026-05-22_1100_UP_m1c_d_lds_staged_cooperative_copy_kills_strided_rw.md), Receiver 8.03 → 2.88 ms (-64%), 整 dispatch 9.41 → 4.22 ms (2.23×)。
- 触发本次工作的事件: 上一篇 note 里以及更早的 [`10:35 FLAT phase profile`](./2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md) 都用了一个 "FC1 在 DSv3 prod roofline ≈ 1.5 ms" 的估计 (从 M0 BASELINE 的 `8192x4096x7168 = 1290 TFLOPS = 0.377 ms` 拍脑袋 ×4 推出来), 用来算 M2-G overlap 上限。这个估计有两个问题: (a) M=8192 不是 dispatch 之后 FC1 的真实 M (后者是 epg × tokens_per_expert), (b) FC1 是 gated SwiGLU 实际 N 是 2*F 不是 F。今天的 P0 是**真跑一次 `bench_gemm`** 把 FC1 / FC2 真实 wall 量出来, 校 M2-G 走不走得通。

## 2. 问题

要回答的问题非常简单: **DSv3 prod (T=8192, H=7168, F=2048) 后端到端 dense FC1+FC2 在 MI355X 单 GPU 上一次 forward 要多少 ms?**

- **现状假设** (之前 note 用的): FC1 ≈ 1.5 ms, receiver ≈ 3 ms, FC1/receiver ≈ 0.5× → 完美 overlap 只能藏 50% receiver, M2-G 单独不够, 必须立刻接 M3 fused 或 mxfp8。
- **目标**: 用 `build/bench_gemm` 在 dispatch 之后真实 M (= `epg × tokens_per_expert` = 32 × 2048 = 65536 balanced) 上跑 dense GEMM, 看 FC1 (gated, N=2*F=4096) 跟 FC2 (N=H=7168, K=F=2048) 实际 wall 各是多少, 重新对 receiver 算 overlap 上限。
- **卡点**: `bench_gemm` 是 dense 单 GEMM, 不是 grouped (32 个 weight matrix); 跑出来的 wall 是 grouped GEMM 的**最优下界** (= 所有 expert 共享一个权重时的 wall), grouped 的真实 wall 会因 per-group tile 小 + 分发开销而更高。但 dense roofline 是 M2-G overlap 上限的关键 input —— 我们在 super-kernel 里跑的本来就是把 32 group 在一个 persistent grid 内 work-steal 起来, 比 Megatron grouped 的 per-launch 分发开销少, 应该接近 dense 而不是 grouped。

## 3. 做了什么

### 3.1 build `bench_gemm`

```bash
ssh mi355-gpu-7 "podman exec xiaoming-dev bash -lc 'cd /shared/.../RocMoE/build && \
    cmake --build . --target bench_gemm'"
```

### 3.2 跑 5 个形状 (DSv3 prod FC1/FC2 主点 + hot_cov50 上界 + M0 sanity)

```bash
./build/bench_gemm 65536 4096 7168 10 100   # FC1 gated (gate+up fused), balanced
./build/bench_gemm 65536 2048 7168 10 100   # FC1 half-gated (gate or up alone)
./build/bench_gemm 65536 7168 2048 10 100   # FC2 down
./build/bench_gemm 98304 4096 7168 10 100   # FC1 hot_cov50 worst-case M (1 expert 12198 tok × 8 src / 32 le 节点 max)
./build/bench_gemm  8192 4096 7168 10 100   # sanity, 对 M0 BASELINE 已知 0.377 ms / 1278 TFLOPS
```

原始输出落 `bench_results/bench_gemm_dsv3_fc_20260522.txt`。

## 4. 效果

### 4.1 dense GEMM wall (单 GPU MI355X, bf16, hipEvent 100-iter)

| stage | M | N | K | wall (ms) | TFLOPS | % MI355X bf16 peak (1290) |
|---|---|---|---|---|---|---|
| **FC1 gated (fused gate+up)** | 65536 | 4096 | 7168 | **3.247** | **1185** | **92%** |
| FC1 half-gated | 65536 | 2048 | 7168 | 1.773 | 1085 | 84% |
| **FC2** | 65536 | 7168 | 2048 | **1.831** | **1051** | **82%** |
| FC1 hot_cov50 worst (M=98304) | 98304 | 4096 | 7168 | 4.883 | 1182 | 92% |
| sanity (M0 已知) | 8192 | 4096 | 7168 | 0.377 | 1278 | 99% |

**FC1 + FC2 总** (M2-G+M3 fused 兜底 wall, balanced): **5.08 ms**。

### 4.2 修正 M2-G overlap 上限

| 量 | 11:00 M1c-D note 估计 | 本次实测 | Δ |
|---|---|---|---|
| FC1 wall (DSv3 prod, single-GPU dense) | ~1.5 ms (M=8192 sanity 拍脑袋 ×4) | **3.247 ms** | +117% |
| Receiver wall (M1c-D 后, balanced) | 3.0 ms (= 2.878 ms) | 2.878 ms | 一致 |
| **FC1 / Receiver** | 0.5× | **1.13×** | **win 翻倍** |
| 完美 overlap 时 dispatch wall 被藏比例 | 50% | **100% (FC1 ≥ receiver 完全 cover)** | — |
| M2-G+M3 fused 稳态 wall 估 (max(receiver, FC1+FC2)) | — | **~5.08 ms** (= FC1+FC2) | — |

**关键结论翻转**: FC1 (3.25 ms) 不是比 receiver (2.88 ms) 小一半, 而是**比 receiver 还大 13%**。这意味着:

- 完美 chunk-overlap 时, **M2-G 单独就能把整个 dispatch wall 藏在 FC1 后面**, 不需要等 M3 fused 才看到收益。
- M2-G+M3 fused 的稳态 wall ≈ FC1 + FC2 = **5.08 ms**, 而不是之前估计的 receiver-bound 3-4 ms。
- 但 5.08 ms 是 BF16 dense 物理 floor, M3 fused 之外要继续压只能靠 mxfp8 (M6) 把 GEMM 自己再砍一刀。

### 4.3 跟 Megatron `mcore_alltoall_gg` (DSv3 T=8192 balanced, 05-21 15:14 baseline) 对位

| stage | Megatron `mcore_alltoall_gg` (ms) | RocMoE-v2 路线图预测 | 差距 |
|---|---|---|---|
| route | 1.501 | (super-kernel B1 / Stage A, ~0.15 ms 从 PhaseA+syncA 推) | -10× |
| dispatch | 2.557 | M1c-D standalone 4.222 → **M2-G 后期望 0 (完全 hide 在 FC1 后)** | M2-G 前 1.65× / M2-G 后理论 0 |
| **experts (FC1+SwiGLU+FC2 grouped)** | **9.298** | **5.08 (dense roofline)** | **Megatron grouped 比 dense 慢 1.83×** (per-expert tile 小 + per-launch 分发) |
| combine | 3.492 | 留给 M5 atomic-free 寄存器归约 | 暂略 |
| postprocess | 1.727 | (super-kernel TAIL_COMBINE) | 暂略 |
| **forward total (crit_path_ms)** | **17.46** | **理论 ~6-7 ms** (M2-G+M3 hide dispatch + dense FC1+FC2 + combine 0.5 ms 估) | **2.5-2.9× Megatron** |

✅ 验证: M2-G 验收门槛 "hot_cov50 dispatch tax ≤ +15%" 完全可达, 而且 FC1 比 receiver 还大, 留给 dispatch 的 tail (hot 5.42 ms - balanced 4.22 ms = +1.2 ms skew tax) 都能藏在 FC1 一次循环里。

定性观察:

- ✅ FC1 实测 92% peak (1185 TFLOPS, 比之前最高的 8192x4096x7168 = 99% 略低, 因为 M=65536 时不再是 wave-tile 完美整除, 跟 RocMoE-bak super 推断一致), FC2 82% peak (N=7168 大行向上 tile 利用率稍差), 不需要额外 tile 调优。
- ✅ hot_cov50 worst-case M=98304 也能保 92% peak, 说明 GEMM body 不会因为 skew M 抖动掉率, 这一点跟 dispatch skew tax 是两个独立维度。
- ⚠️ **未做**: 没量 grouped GEMM 单独的 dense vs grouped overhead —— 必须等 M2-G GEMM body 装进 super-kernel 后做端到端测才能确认 work-stealing 调度能不能保住 dense roofline (而不是退化到 Megatron grouped 1.83× cost)。
- ❌ 没动 M2-G 代码, 只校了 overlap 上限的输入参数。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | **M2-G GEMM body 装进 persistent super-kernel** | 验证 chunk-level dispatch ↔ FC1 MFMA overlap 真触发, hot_cov50 dispatch tax 从 +28% 压到 ≤ +15% (=matches RCCL), 单 forward wall 从 17.46 ms (Megatron) 朝 ~6 ms (RocMoE 理论) 收敛 | 中, 已经有 M0 `mfma_tile.h` 99% MFMA 的 body + M2-D 的 work-stealing scaffold, 主要是 wire 进 GEMM 角色 + per-pool-block `l1_arrival_count` driver |
| **P1** | **治 H=7168 packed_outbox bit-exact** | 解锁 super-kernel 端到端正确性测试 (`test_super_kernel_e2e` 上 prod shape), M2-G 验收的必要前置 | 中, 跟 M1c-A 05-21 21:00 DOWN note 同源 (L2 capacity / TLB thrash) |
| **P1** | 量真实 grouped GEMM 在 work-steal 调度下的 wall, 跟 dense roofline 比 | 校 5.08 ms 兜底估计的现实性 | 0, 等 M2-G 装完直接测 |
| P2 | M6 mxfp8 weights for FC1/FC2 | 把 5.08 ms FC1+FC2 砍到 ~2.5 ms (FP8 2× BF16 peak), 整 forward 收到 ≤ 5 ms | 高, 需离线量化 + MFMA fragment f8f6f4 |

## 相关文件

- 代码: `benchmarks/bench_gemm.hip` (M0 落地, 本次未改)
- 上游 note (含错估的 1.5 ms): [`2026-05-22_1100_UP_m1c_d_...`](./2026-05-22_1100_UP_m1c_d_lds_staged_cooperative_copy_kills_strided_rw.md) · [`2026-05-22_1035_FLAT_dispatch_phase_profile_...`](./2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md)
- 原始 bench 输出: `bench_results/bench_gemm_dsv3_fc_20260522.txt`
- 对位 Megatron: `bench_results/mcore_baseline_20260521_1014.csv`
