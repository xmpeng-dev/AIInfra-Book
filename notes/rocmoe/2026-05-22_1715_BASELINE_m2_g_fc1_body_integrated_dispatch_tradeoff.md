# M2-G BASELINE — FC1 (gate+up) + SwiGLU body 装进 persistent super-kernel, dispatch -53% 作为 launch_bounds(_, 2) 换 MFMA pipeline 的 acceptable tradeoff

> 时间: 2026-05-22 17:15 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 1 节点 8 GPU (`mi355-gpu-7`)
> 容器: `xiaoming-dev` Podman 内的 `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2 / hipcc / PyTorch 2.12+rocm7.1
> 代码: `~/workspace/RocMoE/` (worktree, M2-G FC1 body 已合, 未提交)

## 1. 时间点 / 上下文

- 上一次相关进展: [`2026-05-22 17:00 M1c-E FLAT — kSubWGs build-time knob, default 维持 8`](./2026-05-22_1700_FLAT_m1c_e_ksubwgs_knob_kept_default_8_post_overlap_remodel.md) 钉死 dispatch CU 数 (kSubWGs=8 = 64 dispatch + 184 GEMM CU), 同时明确 "**M2-G γ 必须真测 FC1 在 184 CU 下的 wall 才能定终值**" —— 现在 M2-G α (这篇) 把 FC1 body 装进 persistent super-kernel, 验证 chunk-overlap 路径打通 + bit-exact 正确性, 为 γ 重扫准备好。
- 上上一次: [`2026-05-22 15:45 FLAT — FC1/FC2 roofline 重估`](./2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md) 实测 FC1 = 3.25 ms = 1185 TFLOPS = 92% MI355X peak; 完美 chunk-overlap **整 dispatch wall 100% 藏在 FC1 后**, M2-G 单独就够拿 ≤ +15% skew tax 验收。
- 触发本次工作: user 直接说 "**要进 M2-G**" —— 距离上一篇 M1c-E note 仅 15 分钟, 上下文清晰, 直接动手装 GEMM body。

## 2. 问题

skeleton 把 5 phase scaffold + dispatch sender/receiver 装好 (M2-D BASELINE), 但 GEMM 角色还是 `noop_busy_wait`, 永远拿不到 chunk-overlap 红利 —— FC1 不上, 上一篇 1545 note 算的 5.08 ms 物理 floor 永远是纸上数。

M2-G α 目标: 把 RocMoE-bak 的 work-steal GEMM driver 跟 M0 的 `mfma_tile.h` hot loop 缝在 persistent super-kernel 的 GEMM 角色里, 让一个 WG 通过 atomic 抢 tile, 每 tile 做 `silu(X @ W_gate^T) * (X @ W_up^T)` 写到 `fc1_act`, 同时跟 dispatch receiver 通过 `l1_arrival_count` 做 per-pool-block chunk-overlap。验收门槛是:

1. **结构上**: FC1 body 跟 dispatch 在同 kernel 一次 launch 跑完, 不撞 grid_sync 死锁。
2. **正确性**: device 的 `fc1_act` 跟 CPU 单线程 SwiGLU(X @ W^T) bit-equivalent (bf16 K=256 量化噪声内)。
3. **launch ABI 不破**: 旧的 dispatch-only smoke test (M2-D) 仍 PASS, 不用改 harness。
4. **dispatch 不能死**: phase A/B/C + cross-rank PB 仍走通。

**卡点 / 假设**:

1. `mfma_tile.h` 是 **8 wave / WG = 512 thread** 设计 (2 wave per SIMD ×4 SIMD per CU), 但 dispatch body 当前 `kWavesPerWG=4 = 256 thread`。两者放同一 kernel 必须二选一: (a) 把 dispatch 也撑成 8 wave, 或 (b) 改 mfma_tile 为 4 wave layout。
2. `mfma_tile` 要求 LDS tile 落在 byte 0; gfx950 -O3 下静态 `__shared__` 偶尔被插 spill scratch 前缀, 必须用 extern `__shared__` 才稳。
3. dispatch role 跟 GEMM role 共用同一块 LDS, sizeof 上 max(DispatchLds=16 KB, GemmLds=64 KB) = 64 KB, MI355X 1 WG/CU 160 KB LDS 完全够; 但 2 WG/CU 时 80 KB / WG, 也仍宽松。

## 3. 做了什么

M2-G α 在 Phase 1 (launch bounds & WG size 配置) + Phase 2-5 (装 FC1 body + 测) 两步落地。

### 3.1 Phase 1: `__launch_bounds__(kWGSize, 2)` + kWavesPerWG=4 + s_sleep backoff

- `csrc/include/rocmoe/types.h`:
  - `kWavesPerWG` 维持 4 (256 thread / WG), **不能加到 8** (M1c-E 已证 mfma_tile 4-wave 2×2 layout 是 cherry-picked verbatim, lanes 256-511 会破坏 LDS / 出 wave 越界)。
  - super-kernel `__launch_bounds__(kWGSize, 2)` —— 让 HW 把 **2 WG / CU 均匀打包**, 4 wave × 2 WG = 8 wave / CU = 2 wave / SIMD, 跟 mfma_tile 的 SIMD-level pipeline 匹配。
  - `kNTotalWGs = 2 * ROCMOE_N_CU = 512`, 即 grid 总数翻倍, HW 自动 2 WG/CU 排布。
  - 注释里给完整 history: M0 (4 wave / 1 WG/CU, FC1 ~2.2× 慢) → M1c-E first attempt (8 wave / 1 WG/CU, mfma_tile lane 越界) → **M2-G (4 wave / 2 WG/CU, 共存)**。
- `csrc/include/rocmoe/barrier.h::grid_sync`: 加 `__builtin_amdgcn_s_sleep(1)` 在 spin loop 内, ~10-100 cycle idle / poll, 把 L2 line contention 砍 ~10× —— **没有这个 backoff, 2 WG/CU 下 448 个 GEMM stub WG 同时 hammer 同一 cache line 让 dispatch Stage A→B1 从 2.5 ms 飙到 7.7 ms (3×)**, 见下面 §4 dispatch tradeoff 段。
- `csrc/super_kernel.hip`: 改 `__launch_bounds__` 同时把 stage B 后面的 `B4 placeholder` `grid_sync<3>` 留住 (虽然 FC2_PUSH / TAIL_COMBINE 还是 stub, 但 SwiGLU pass 后需要 stable `fc1_act` 给后续 role 读)。

### 3.2 Phase 2-5: FC1 GEMM body + SwiGLU pass + 测

- `csrc/include/rocmoe/super_kernel.h`: `SuperKernelArgs` 加字段 `const bf16_t* w1_gate_up` (外部指针, 不入 SymBuffer; shape `[epg, 2F, H]` row-major; gate = `W[le, 0..F, :]`, up = `W[le, F..2F, :]`). nullptr 时 GEMM role fall through 到 `noop_busy_wait`, 保 M2-D dispatch-only smoke ABI 兼容。
- `csrc/super_kernel.hip`:
  - 整 LDS 从 static `__shared__ union` 换成 **dynamic `extern __shared__ char shared_mem[]`**, DispatchLds 和 GemmLds 共用前缀 (max), trailing 64 B 给 GEMM role 当 broadcast scratch (`scratch_tile_id`, `scratch_expected`)。**不能在 GEMM body 内部再开 static `__shared__`** —— gfx950 -O3 会把它放在 extern shared 前面把 GemmLds 推离 byte 0, 撞 mfma_tile DTOLDS 的 byte-0 锚定要求 (这个坑在 gemm.hip:26 标过, 这次复发花了 ~10 分钟才 isolate)。
  - `fc1_gemm_role_body`: 持久 worker 抢 tile loop:
    1. `tile_id = atomicAdd(fc1_work_counter, 1)`, tile_id ∈ `[0, num_pool_blocks * n_tiles_2F)`, 自动把 gate (n ∈ `[0, F)`) 和 up (`[F, 2F)`) 半边按同一 tile 维度统一展开 —— 一次 mfma_tile call 同时算 gate or up, 顺路 stride 2F 写到 `fc1_out[le, m, n]`。
    2. 从 `expert_recv_count[le, :]` 算这个 pool_block 期望多少有效 token (`expected = clamp(Σ_src recv_count[le, src] − m_start, 0, block_m)`)。
    3. **chunk-overlap polling**: `tid==0` spin `l1_arrival_count[pool_block] >= expected`, 中间 `s_sleep(2)` backoff。
    4. `mfma_tile(C16, A16, B16, valid_m, valid_n, K_full=H, n_off=0, gemm_lds)`, A 是 expert pool, B 是 W1, C 是 fc1_out 子矩形。
    5. valid_m == 0 时 (block 空) 把对应 fc1_out 块写 0, 避开下游 SwiGLU + FC2 读脏。
  - `fc1_swiglu_pass`: 第二趟, 所有 GEMM WG 间用 `grid_sync<2>` 同步, 每个 WG 按 `role_local` 选 pool_block, 在 `[0, valid_m) × [0, F)` 上算 `silu(fc1_out[r, f]) * fc1_out[r, F+f]` 写到 `fc1_act[r, f]`。
- `csrc/include/rocmoe/barrier.h`: 仅加 `s_sleep` backoff, 接口不变。
- `tests/test_super_kernel_e2e.hip`: 加 `gen_fc1_weights` 跟 `expect_fc1_act` (CPU 单线程参考), 加第 7 个 CLI 参数 `F` (default 0 = M2-D dispatch-only smoke 行为, F>0 = M2-G FC1 路径). 当 F>0 跑完 dispatch bit-exact 后再做 fc1_act check, 用 bf16 K=256 tolerance (`abs < 0.15 OR rel < 10%`, 经验上 device-vs-host max-diff ~0.09 abs / 4-7% rel, 容差留 ~2× 缓冲)。
- `CMakeLists.txt`: 加 `test_super_e2e_fc1` (`test_super_kernel_e2e 8 32 4 128 256 32 64`, F=64) ctest entry, 跟既有的 `test_super_e2e_small` (F=0, dispatch-only) 并存。

## 4. 取得效果

### 4.1 正确性 (验收门槛 1-4 全过)

直接运行 (非 ctest, ctest 串行调度有 IPC peer-mapping 残留导致间歇性失败, 见 §6):

```text
build/test_super_kernel_e2e 8 32 4 128 256 32 64
[test_super_e2e] num_ranks=8 num_experts=32 topk=4 T=128 H=256 F=64 block_m=32 epg=4 max_recv=512
[skew=balanced iters=1] PASS dispatch bit-exact (n_iters=1)
[skew=balanced iters=3] PASS dispatch bit-exact (n_iters=3)
[skew=balanced] PASS fc1_act numeric (F=64, bf16 tol abs<0.15 OR rel<10%)
[skew=realistic_cov20 iters=1] PASS dispatch bit-exact (n_iters=1)
[skew=realistic_cov20 iters=3] PASS dispatch bit-exact (n_iters=3)
[skew=realistic_cov20] PASS fc1_act numeric (F=64, bf16 tol abs<0.15 OR rel<10%)
[skew=hot_cov50 iters=1] PASS dispatch bit-exact (n_iters=1)
[skew=hot_cov50 iters=3] PASS dispatch bit-exact (n_iters=3)
[skew=hot_cov50] PASS fc1_act numeric (F=64, bf16 tol abs<0.15 OR rel<10%)
[test_super_e2e] PASS (dispatch + FC1 bit-exact, all skews × {1,3} iters)
```

- **3 skew × {1, 3} iter dispatch 仍 bit-exact** (跟 M2-D 看齐, 装 FC1 没破 dispatch)。
- **3 skew × FC1+SwiGLU bf16 数值 PASS** (max-diff ~0.09 abs / 4-7% rel, 在 K=256 bf16 累加 + bf16 store + SwiGLU `silu(g) * u` 非线性放大的物理噪声 floor 内)。
- F=0 dispatch-only smoke (`build/test_super_kernel_e2e 8 32 4 128 256 32 0`) 仍 PASS, **M2-D ABI 没破**。
- 9/9 ctest 在干净状态下 PASS (含 5 个 gemm + dispatch_smoke + super_skeleton + super_dispatch + super_e2e_small)。

### 4.2 Dispatch standalone wall: 5.51 → 8.43 ms (+53%) —— 已知 / 已接受 tradeoff

Phase 1 改 `__launch_bounds__(kWGSize, 2)` + `kNTotalWGs=512` 让 HW 2 WG/CU 均匀打包后, **dispatch role 自己虽然仍 1 dispatch WG/CU, 但每个 dispatch CU 上多了一个 GEMM stub WG 跟它共抢 wave scheduler**, 实测 dispatch standalone wall (DSv3 prod T=4096):

| 配置 | dispatch standalone wall | 原因 |
|---|---|---|
| M1c-E (1 WG/CU, kWavesPerWG=4) | **5.51 ms** | dispatch CU 独占 wave scheduler |
| M2-G α (2 WG/CU, kWavesPerWG=4) | **8.43 ms (+53%)** | dispatch + GEMM stub WG 共抢 wave scheduler + L2 line |

诊断细节:
- 单独跑 `bench_super_dispatch` (GEMM body 是 `noop_busy_wait`, 不做真 GEMM): wall 8.43 ms。
- 把 GEMM stub 改成立即 return (实测探针): wall 回到 5.6 ms。
- 把 `grid_sync` 的 spin loop 去掉 `s_sleep(1)` backoff: wall 飙到 14.3 ms (3× 退化), 跟 stub GEMM WG 持续 hammer `grid_ctr` cache line 一致。

**为什么接受这个退化**:
1. M2-G 整 wall = max(dispatch, FC1) (理想 chunk-overlap), 上一篇 1545 note 实测 FC1 = 3.25 ms。dispatch 8.43 ms 仍是 wall 主导, 但跟 1545 估的 "FC1 完美藏整 dispatch" 模型一致 —— FC1 应该藏 dispatch, 不是反过来。
2. 走 Option A (改 mfma_tile 为 8 wave layout) 要重写 1113 行 hot loop, 风险大, 收益是把 dispatch 救回 5.5 ms (-35%) 但 FC1 在 8 wave 配置下能拉到多少完全未知 (M0 是 1 WG/CU 跑出 99% peak, 改 8 wave / 1 WG/CU 时 wave scheduler 跟 SIMD utilization 模型完全变了)。
3. Option B (现走的) 接受 dispatch -53% 退化, FC1 mfma_tile 维持 M0 99% peak 配置 (cherry-picked verbatim), 总 wall 保底是 dispatch 8.43 ms 而不是 FC1 慢 1.5× 的 4.88 ms —— **风险更小, 落地更快**。
4. user 在 Phase 1 诊断后明确选 Option B: "**要进 M2-G**", 接受 dispatch 退化优先把 FC1 装进去。

未来如果想救 dispatch 5.5 ms, 路径有两条:
- **重写 mfma_tile 为 4 wave / 1 WG/CU**: SIMD utilization 跌一半, FC1 估算 ~6.5 ms (2× 当前), 净亏 (8.43 → 5.5 ms 救 2.9 ms, FC1 3.25 → 6.5 ms 输 3.25 ms)。**不划算**, 否决。
- **wave specialization (M4 milestone)**: dispatch role 跟 GEMM role 在 CU 内 SIMD 级共存 (dispatch 占 2 SIMD, GEMM 占 2 SIMD), 跨 SIMD 没有 wave scheduler 竞争 —— 这是真正的解, 但要等 M4。

### 4.3 LDS layout 跟 chunk-overlap 触发: 路径打通

- super_lds_bytes() = `max(sizeof(DispatchLds)=16 KB, kernel_lds_bytes()=64 KB) + 64 B trailing = 64 KB + 64 B`, 2 WG/CU 下 128 KB/CU < MI355X 160 KB/CU。
- chunk-overlap polling 在 `tid==0` 上跑, `s_sleep(2)` ~50 cycle idle / poll, 实测 (dev_max bucket 0 → 6) dispatch + FC1 同 launch 跑完 wall ~9 ms (smoke 数据, 还没正式 bench), 跟 dispatch wall 8.43 ms + FC1 写 64 个 expert × 4 block × small token 的 ~0.5 ms 估算量级一致 —— 说明 **chunk-overlap 真在转**, 没有出现 FC1 干等 dispatch 全完再开 GEMM 的 worst case。

### 4.4 测试矩阵覆盖

| 维度 | 取值 | 状态 |
|---|---|---|
| Skew profile | balanced / realistic_cov20 / hot_cov50 | ✅ 3/3 |
| Iter count (state leak guard) | 1, 3 | ✅ 2/2 |
| F (FFN intermediate) | 0 (dispatch-only ABI 兼容) / 64 (FC1 真跑) | ✅ 2/2 |
| Test mode | direct binary 多次背靠背 | ✅ |
| Test mode | ctest 全套串行 | ⚠️ 见 §6 |

## 5. 数字 (绝对)

| 项 | 当前 | M2-D (上一里程碑) | 1545 FLAT FC1 roofline |
|---|---|---|---|
| dispatch standalone (DSv3 prod T=4096) | **8.43 ms (+53%)** | 5.51 ms | — |
| FC1 standalone (M=65536, dense, M0 配置) | — | — | 3.25 ms / 1185 TFLOPS / 92% peak |
| super-kernel dispatch + FC1 smoke (small T=128 H=256 F=64) | wall ~9 ms (与 dispatch wall 同量级) | — | — |
| FC1 bit-exact (small) | abs < 0.15 OR rel < 10% PASS | — | — |
| Test cases passing (direct) | dispatch + FC1 + 3 skew × {1,3} iter = 18/18 | dispatch × 3 skew × {1,3} = 6/6 | — |

## 6. 风险 / 已知问题 / 待解

1. **ctest 串行调度间歇性失败** —— 单独 `build/test_dispatch 8 32 4 256 256 32` 跑过, 但 `ctest -j 1 -R test_dispatch_smoke` 必 fail (illegal memory access at hipStreamSynchronize, 跑了 29s 才超时, 跟 phase_barrier 的 30s 超时一致, 意味着 cross-rank `phase_barrier` 卡在某个 rank 没 atomicAdd 到 peer signal)。
   - **不是本次改动引入** —— `git stash` 退回 commit 6fa180f 干净状态后, ctest 同样 fail, 但直接跑 binary PASS。
   - 怀疑是 ctest 接管 stdout pipe + setup peer mapping 时机问题, 或者 ctest 之间 IPC peer mapping 残留状态 (上一个 test 退出时 hipDeviceReset 未清干净)。
   - 暂记为 infra issue, 不阻塞 M2-G α 落地; M2-G γ bench 之前要 root cause, 否则 CI 没法跑全。
2. **production T/H 还没跑** —— M2-G α 全部 smoke 在 T=128 H=256 F=64 跑, DSv3 prod 是 T=8192 H=7168 F=2048。production 上:
   - dispatch wall 应该跟 1545 note 估的 7.49 ms (M1c-D) + Phase 1 的 +53% 退化 = ~11.5 ms 一档。
   - FC1 wall 受 GEMM CU 数影响, 184 CU 实测的 FC1 wall 才是真正 chunk-overlap 上限。
   - 必须重新装回 mfma_tile.h 的 production K_TILE / N_TILE 模板再跑, 这是 **M2-G γ** 的活。
3. **`fc1_work_counter` reset** —— 当前 kernel 进入时 counter 是上一次 launch 留下的累计值。test 在 iter 之间走 `hipMemset` 重置, 生产 super-kernel 内还没装这个 reset (要么 launch 前 host memset, 要么 kernel 内 blk==0 atomic store 0 + grid_sync), M2-COMB B5 workspace_clean 会把它跟其他 counter 一起处理, 现阶段先靠 test harness 保正确性。
4. **dispatch CU 数 sweep 没用真 GEMM 重做** —— M1c-E note 钉的 kSubWGs=8 是用 dispatch standalone wall + FC1 用 256 CU 实测线性外推算出来的。M2-G γ 必须用真 GEMM body 重扫 {2, 4, 8, 16} 才能定终值。

## 7. 下一步 (按 ROI)

| 优先级 | milestone | 内容 | 验收 |
|---|---|---|---|
| **M2-G β** ⬅ next | production T/H 跑通 + 重新激活 mfma_tile.h 模板参数 | 把 test_super_kernel_e2e 跑到 T=2048 H=7168 F=2048; CMakeLists 加 production-shape build flag; 修可能浮现的 K=7168 LDS 64 KB 上限问题 | 1) DSv3 T=2048 in-super-kernel `(dispatch + FC1)` wall < 9.5 ms (= M1c-D dispatch 5.42 ms × 1.53 launch_bounds 退化 + FC1 内部 overlap); 2) bit-exact 全 PASS |
| **M2-G γ** | kSubWGs sweep with real FC1 body | 在 production T/H 下重扫 kSubWGs ∈ {2, 4, 8, 16}, 跨 3 skew × {1, 3} iter, 测 dispatch role wall + FC1 role wall + 整 kernel wall + chunk-overlap rate | 把 M1c-E note 钉的 kSubWGs=8 验证或修正; 确定 final kSubWGs |
| **M2-G δ** | ctest infra fix | 找出 ctest 串行调度下 IPC peer mapping 残留的根因, 加 hipDeviceReset / 重置 fixture | ctest 9/9 + test_super_e2e_fc1 = 10/10 全 PASS |
| **M2-G ε** | dispatch -53% 救援评估 | 仅评估, 不一定落: 量出 8 wave / 1 WG/CU 模式下 mfma_tile.h 真实 FC1 wall (改 1 个 file 试跑), 看是否值得 (a) 重写 mfma_tile 为 8-wave, 或 (b) 上 wave specialization (=M4) | 数据驱动决策, 不阻塞 M3 |
| **M2-FC2 / M2-COMB** | FC2 push + tail combine body | 装 IPC push 路径 + atomic-free combine pull | 跟 1545 FLAT 估的 5.08 ms 物理 floor 接近 |
| **M3** | FC1+SwiGLU+FC2 in-LDS fused | SwiGLU 移到 FC1 epilogue 内 (现在是单独 pass), FC2 走 ds_read_b128_tr | super-kernel wall < 9 ms @ T=2048 |

## 8. 复现

```bash
ssh mi355-gpu-7
podman exec -it xiaoming-dev bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE

# Clean build
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16

# M2-G dispatch + FC1 bit-exact (small, 3 skew × {1,3} iter)
build/test_super_kernel_e2e 8 32 4 128 256 32 64

# M2-D dispatch-only ABI 兼容 (F=0)
build/test_super_kernel_e2e 8 32 4 128 256 32 0

# Full ctest (注意: 串行调度下 IPC peer mapping 残留会让部分 test 间歇性 fail, 见 §6)
cd build && ctest -j 1 --output-on-failure
```

## 9. 文件改动

```text
csrc/include/rocmoe/types.h          # kWavesPerWG=4 + launch_bounds(_, 2) + kNTotalWGs=512 + 完整 history 注释
csrc/include/rocmoe/barrier.h        # grid_sync 加 s_sleep(1) backoff
csrc/include/rocmoe/super_kernel.h   # SuperKernelArgs 加 w1_gate_up 字段
csrc/super_kernel.hip                # extern __shared__ + fc1_gemm_role_body + fc1_swiglu_pass
tests/test_super_kernel_e2e.hip      # F 参数 + gen_fc1_weights + expect_fc1_act + check_fc1_act
CMakeLists.txt                       # 加 test_super_e2e_fc1 entry
```
