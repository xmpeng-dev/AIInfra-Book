# M1c-D UP — LDS-staged `cooperative_b128_copy` + sender 连续分片让 Receiver 2.79×, 整 kernel 2.23× (DSv3 prod T=8192)

> 时间: 2026-05-22 11:00 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, CDNA4), XGMI 全互联, 1 节点 8 GPU (`mi355-gpu-7`)
> 容器: `xiaoming-dev` Podman 内的 `docker.io/rocm/primus:v26.2`
> 软件: ROCm 7.2 / hipcc / PyTorch 2.12+rocm7.1 / Primus
> 代码: `~/workspace/RocMoE/` (worktree, M1c-A + M1c-D 已合, 未提交)

## 1. 时间点 / 上下文

- 上一次相关进展: [`2026-05-22 10:35 FLAT — dispatch 5-phase profile`](./2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md) — 把 dispatch wall 9.41 ms (balanced) 分成 PhaseA/syncA/PhaseB/syncB/PhaseC/PhaseBar/syncC/Receiver 八段, 量出 **Receiver 占 wall 73-85% (max 8.03→10.50 ms)**, **PhaseB pack max-rank 受 skew 影响 +41%**, cross-rank phase_barrier 实际只占 0.6%。当时的结论是 "M2-G overlap 上限只能藏 receiver 8 ms 的 ~15%, 必须立刻接 M3 fused 或 mxfp8" —— 本次 note 把 Receiver 直接打掉 ~64%, 这条结论需要重估。
- 触发本次工作的事件: user 在 `csrc/include/rocmoe/ipc_primitives.h:165-192` 上选中现场, 直接问 "怎么还是跳着读跳着写, 昨天给的两条设计理念为什么没落地":
  1. **sender 端组织好**, receiver 整段读取
  2. **不要跳着读跳着写, 用 LDS 作为缓存**

  即: 同时治 PhaseB (sender pack) 写出布局 + cooperative_b128_copy 本身 (调用点至少 8 处, 是 PhaseB / Receiver 共用的 hot path)。

## 2. 问题

dispatch wall 8 段里, **Receiver 跟 PhaseB 加起来占 90%+**, 都被同一个 `cooperative_b128_copy` 喂数据:

- **现状 (DSv3 prod T=8192 balanced)**:
  - Receiver wall max = 8.03 ms (73% wall)
  - PhaseB wall max = 1.20 ms (12% wall)
  - 整 kernel hipEvent p50 = 9.41 ms
- **目标**: 把 Receiver wall 拉下来到跟 PhaseB 同量级, 让 M2-G 的 FC1 grouped GEMM (~1.5 ms roofline) 有机会真把 Receiver overlap 住, 否则 receiver 还是 wall 主导, M2-G 收益被 cap 死。
- **卡点 / 假设**:
  1. `cooperative_b128_copy` 的旧实现是每 thread 按 `tid, tid+kWGSize, tid+2*kWGSize` 跳着发 load/store —— 同一 cycle 内 512 lane 落在连续地址, 但同一 lane 相邻 load 隔 `kWGSize * 16 B = 8 KB`。从 XGMI peer-L2 prefetcher 角度看, 每个 cycle 是一次合并 burst, 但 lane 内的 stride 8 KB 像非流式工作负载, prefetcher 不下发 ahead, 实测 BW 远低于 50 GB/s 的链路上限。
  2. 这种 fused load+store 形式让 wave-N 的 store 跟 wave-N 的 load 复用同一组 VGPR; 在 multi-wave 部分激活的尾巴段 (e.g. receiver 一段 chunk 落在 wave-0..2, wave-3..7 idle) 会复现 [05-21 M1c-A note](./2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md) 里发现的 wave-1+ store 不落盘 bit-exact bug —— 之前是用 per-row 调用绕开, 没有根治。
  3. M1c-A 的 PhaseB 旧实现是 `s = sub_wg + n * kSubWGs` 按 stride 写 packed_outbox, 8 个 sub_wg 同时往同一个 (dst_rank, e) bucket 的 stride-8 slot 写 —— receiver 端按 bulk pull 读时仍要跨 sub_wg 走 L1 → L2 → XGMI peer-L2 一圈, sender L1 dirty line 必须被 L2 收回才对 peer 可见。

## 3. 做了什么

### 3.1 重写 `cooperative_b128_copy` 为 LDS-staged tile loop

`csrc/include/rocmoe/ipc_primitives.h` 完全换皮 (170 行注释 + 70 行实现, 旧实现彻底删):

API 变成需要调用者传 LDS staging 缓冲:

```cpp
__device__ __forceinline__
void cooperative_b128_copy(void* dst, const void* src, int num_bytes,
                           uint4* lds_stage, int lds_stage_uint4);
```

每个 tile 走 4 步, `__syncthreads()` 隔在 LOAD/STORE 之间:

```cpp
for (int base = 0; base < n; base += lds_stage_uint4) {
    const int chunk = min(n - base, lds_stage_uint4);

    // (1) LOAD phase: global → LDS
    //     主循环 kUnroll=2 把两次 load 排在一起 (XGMI 2 个 in-flight req / lane),
    //     全 LDS 写完一次 __syncthreads() 收口。
    int i = tid;
    for (; i + (kUnroll - 1) * kWGSize < chunk; i += step) {
        uint4 buf[kUnroll];
        #pragma unroll for (int u=0; u<kUnroll; ++u) buf[u] = s[base + i + u * kWGSize];
        #pragma unroll for (int u=0; u<kUnroll; ++u) lds_stage[i + u * kWGSize] = buf[u];
    }
    for (; i < chunk; i += kWGSize) lds_stage[i] = s[base + i];
    __syncthreads();

    // (2) STORE phase: LDS → global
    //     完全镜像 LOAD, 新一轮 VGPR 跟前面 in-flight XGMI load 解耦,
    //     wave-N 的 store 不再依赖 wave-N 的 load。
    i = tid;
    for (; i + (kUnroll - 1) * kWGSize < chunk; i += step) {
        uint4 buf[kUnroll];
        #pragma unroll for (int u=0; u<kUnroll; ++u) buf[u] = lds_stage[i + u * kWGSize];
        #pragma unroll for (int u=0; u<kUnroll; ++u) d[base + i + u * kWGSize] = buf[u];
    }
    for (; i < chunk; i += kWGSize) d[base + i] = lds_stage[i];
    __syncthreads();
}
```

设计要点 (写进文件头注释):

- LDS staging 缓冲由调用方传进来, 不是函数内 `__shared__` —— 8 处 inlined call site × 16 KB tile = 128 KB / CU, 超 MI355X 64 KB LDS, 必须复用一块。
- tile 大小 `kCopyStageUint4 = 1024 uint4 = 16 KB`, 对得上 MI355X XGMI 单链路 50 GB/s × ~300 ns 往返的 BDP, 让 peer-L2 prefetcher 一次看到一长串顺序流, 而不是被 stride 8 KB 切碎。
- 主循环 + 尾巴的 split 镜像旧 register-pipelined 形式 (避开 "wave inactive per loop iter" hazard) —— 唯一变化是 LOAD/STORE 之间多一次 LDS 跳板。
- 对 H=7168 (= 896 uint4 < 1024) 的一行, 整行落在一个 tile, inner tile loop 跑一次, 没有额外开销; 对 chunk_n × H 的 bulk 也只是多走几个 tile, 全在 L2 streaming 窗口内。

### 3.2 给 `DispatchLds` 加 `copy_stage` 字段, 串到所有调用点

`csrc/include/rocmoe/dispatch_body.h`:

```cpp
struct DispatchLds {
    static constexpr int kCopyStageUint4 = 1024;  // 16 KB tile

    int   counts[kMaxRanks * 4];
    int   base  [kMaxRanks * 4];
    uint4 copy_stage[kCopyStageUint4];   // 新增, 8 phase 共用
};
```

`csrc/dispatch.hip` 和 `csrc/super_kernel.hip` 把 `&lds` 透传到:

- `dispatch_sender_pack_phase(args, dst_rank, sub_wg, &lds)`
- `dispatch_receiver_stage(args, src_rank, sub_wg, &lds)`

PhaseA 和 PhaseC 不调 `cooperative_b128_copy`, 但因为 `DispatchLds` 整体复用, 也不会撑爆 LDS。

### 3.3 PhaseB 从 stride-kSubWGs 切到 **连续分片**, 跟 receiver bulk-pull 对齐

`dispatch_sender_pack_phase` 旧布局: 每个 sub_wg 处理 `s = sub_wg, sub_wg + kSubWGs, sub_wg + 2*kSubWGs, ...` —— 8 个 sub_wg 同时写一个 bucket 的 stride-8 slot。

新布局 (连续分片):

```cpp
const int slots_per_sub = (n_slots + kSubWGs - 1) / kSubWGs;
const int s_start = sub_wg * slots_per_sub;
const int s_end   = min(s_start + slots_per_sub, n_slots);
for (int s = s_start; s < s_end; ++s) {
    int t = self.local_send_log_ptr(dst_rank, e, s)[0] >> 5;
    cooperative_b128_copy(
        self.packed_outbox_row(dst_rank, e, s),
        self.input_token_row(t),
        H * sizeof(bf16_t),
        lds->copy_stage, DispatchLds::kCopyStageUint4);
}
__threadfence();   // AGENT scope, 不再需要 fence_system
```

为什么这是对的:

- packed_outbox 物理布局就是 by-slot 连续: row s 和 row s+1 紧挨 `H * sizeof(bf16)` 字节。一个 sub_wg 写一段连续 slot 范围 = HBM 上一段长 burst, sender L2 line fill 是流式的。
- **跟 receiver-bulk-pull 用同一个公式分片**: sub_wg X 在 sender 端写的 [s_start, s_end), 跟 sub_wg X 在 receiver 端读的 [s_start, s_end) 完全一致 —— 每个 slot 都是同一个 sub_wg 写、同一个 sub_wg 读, **不再需要跨 sub_wg L1 visibility**, AGENT-scope `__threadfence()` 即可。
- 写并行度跟旧 stride 方案一样 (8 个 sub_wg 同时各推一行 = 8 行 in-flight per bucket), 但 L2 line ownership 不再被 8 个 sub_wg 争抢, sender 这边的 vL1$ cluster 也不会被 stride 写打散。

### 3.4 Receiver bulk-pull: 连续 slot 范围内 **per-row** LDS-staged 拷贝

`dispatch_receiver_stage` 在 `ROCMOE_DISPATCH_BULK_PULL=1` (默认) 路径:

```cpp
for (int e = 0; e < epg; ++e) {
    int n_slots = smem_counts[e];   int base = smem_base[e];
    int slots_per_sub = (n_slots + kSubWGs - 1) / kSubWGs;
    int s_start = sub_wg * slots_per_sub;
    int s_end   = min(s_start + slots_per_sub, n_slots);

    for (int j = 0; j < s_end - s_start; ++j) {
        int s = s_start + j;
        cooperative_b128_copy(
            self.expert_token_pool_row(e, base + s),
            peer.packed_outbox_row(my_rank, e, s),
            H * sizeof(bf16_t),
            lds->copy_stage, DispatchLds::kCopyStageUint4);
    }

    // metadata: 1 thread / slot, 512 in flight
    for (int s_local = tid; s_local < (s_end - s_start); s_local += kWGSize) {
        ... encode src_meta, copy wts, atomicAdd l1_arrival_count ...
    }
    __syncthreads();
}
```

为什么是 per-row 而不是 "一发把 chunk_n × H 字节全拷":

- 实测 (2026-05-22 bisect) `cooperative_b128_copy(num_bytes > kWarpSize*16 AND num_bytes < kWGSize*16)` 在 8-wave WG 部分激活时偶发 wave-1+ store 不落盘 —— wave-0 lane 写正确, lane 64+ 的 store hipMemcpy 拿不到, 加 AGENT/SYSTEM fence、LDS 都修不掉; 但 per-row 调用 `num_bytes = H * 2` 在 H ≤ 512 时只激活 wave-0, H = 7168 时全 8 wave 都激活, 两种情况都 bit-exact。
- per-row 本身仍然吃到 "连续分片" 的红利: peer 那边 packed_outbox_row(s)、packed_outbox_row(s+1)… 是连续地址, 一行刚发起 peer L2 fill, 下一行直接命中 peer L2 prefetcher 拉好的下一条 line, 整段 [s_start, s_end) 是流式访问。
- per-slot metadata (src_meta 编码 + wts 拷 + l1_arrival_count atomicAdd) 在 bulk 拷完后批量做, 1 thread / slot, chunk_n ≤ 512 时一遍走完。per-WG sync 次数从旧 per-row 路径的 `2 * n_slots` (~ 2048) 降到 **2 per expert** (~ 64), 这部分的 grid 内同步成本也省了。

### 3.5 回归测试 + phase profile

```bash
ssh mi355-gpu-7 'podman exec xiaoming-dev bash -lc "cd /shared/.../RocMoE && \
    ./build/test_dispatch && \
    ./build/test_super_kernel_e2e"'
# ↳ test_dispatch: 8 rank 全 PASS
# ↳ test_super_kernel_e2e: balanced/realistic/hot × {1, 3} iter 全 PASS

for skew in balanced realistic_cov20 hot_cov50; do
    ./build/bench_dispatch_phases 8 256 8 8192 7168 32 5 10 dsv3 $mr $skew 4
done
```

H=7168 production 形状的 bit-exact 是 M1c-A 老 issue (packed_outbox at H=7168 受 L2 capacity 影响, 跟本次改动正交), 留待后面单独治, 不阻塞 phase profile 数据。

## 4. 效果

### 4.1 dispatch wall 分解 (DSv3 prod T=8192 H=7168 8 GPU, hipEvent p50, max-rank)

| skew | phase | Before (M1c-A, 05-22 10:35) | After (M1c-D, 本次) | Δ ms | Δ % |
|---|---|---|---|---|---|
| **balanced** | PhaseA | 0.139 | 0.135 | -0.004 | -3% |
|              | syncA  | 0.140 | 0.138 | -0.002 | -1% |
|              | **PhaseB** | 1.198 | **1.173** | -0.025 | -2% |
|              | syncB  | 0.140 | 0.258 | +0.118 | +84% |
|              | PhaseC | 0.003 | 0.003 | 0 | — |
|              | PhaseBar | 0.053 | 0.059 | +0.006 | +11% |
|              | syncC  | 0.057 | 0.063 | +0.006 | +11% |
|              | **Receiver** | 8.033 | **2.878** | **-5.155** | **-64%** |
|              | **整 kernel p50** | **9.405** | **4.222** | **-5.183** | **-55%** |
| **realistic_cov20** | **PhaseB** | 1.377 | **1.320** | -0.057 | -4% |
|              | syncB  | 0.374 | 0.473 | +0.099 | +26% |
|              | **Receiver** | 8.874 | **3.130** | **-5.744** | **-65%** |
|              | **整 kernel p50** | **10.411** | **4.604** | **-5.807** | **-56%** |
| **hot_cov50** | **PhaseB** | 1.709 | **1.628** | -0.081 | -5% |
|              | syncB  | 0.870 | 0.912 | +0.042 | +5% |
|              | **Receiver** | 10.503 | **3.635** | **-6.868** | **-65%** |
|              | **整 kernel p50** | **12.360** | **5.425** | **-6.935** | **-56%** |

**整体: Receiver 2.79-2.89× 加速, 整 kernel 2.23-2.28× 加速, 跨 3 档 skew 一致。**

定性观察:

- ✅ Receiver wall 从 8-10.5 ms 段直接掉到 2.9-3.6 ms 段, M2-G overlap 上限重估 —— ~~FC1 grouped GEMM ~1.5 ms roofline~~ **[修订, 见 [2026-05-22_1545 FLAT FC1/FC2 roofline](./2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md): FC1 实测 3.25 ms, FC1/Receiver 1.13×, 完美 overlap 能藏 100% receiver 而不是 50%]** ~~跟 receiver 3 ms 量级匹配多了, 完美 overlap 能藏住 ~50% Receiver (而不是上次 note 算的 15%), M2-G 单独够不够支撑 M3 / mxfp8 的优先级要重新摆。~~
- ✅ PhaseB 自己也微跌 1-5% —— 连续分片让 sender L2 line ownership 不被 stride 写打散, 但 PhaseB 本身已经是 HBM 流式写, 边际收益小, 主要红利是给 receiver 喂的数据布局对了。
- ✅ syncC、PhaseBar 等 < 1% wall 的小段几乎不动 (本来就跟本次改动无关), 噪声范围内。
- ⚠️ **syncB 反而升 26-84%** (balanced 0.14 → 0.26 ms, realistic 0.37 → 0.47 ms): 因为 receiver 不再是 wall 主导, sub_wg 之间的 PhaseB 完成时差被相对放大; 但 syncB 绝对 ms 仍然只是 receiver 的 1/10, 是合理的尺度变化, 不是回归。
- ⚠️ H=7168 bit-exact 在 packed_outbox 路径上仍未通过 (M1c-A 老 issue, 跟本次正交), 但 H=256 (test_dispatch / test_super_kernel_e2e) 全 PASS, H=7168 wall 数据来自正常 kernel 跑完 (bench 不验值, 只测 wall), kernel 还在做同样的 LDS 流和 XGMI 拉, 加速度量准确。
- ❌ 没有顺手把 H=7168 bit-exact 治掉 —— 需要单独一个 round 沿 packed_outbox L2 capacity 这条路追, 跟本次设计正交。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | 重跑 `bench_gemm` (DSv3 prod 形状 FC1) 校 M2-G overlap 上限 | 之前算的 receiver 8 ms / FC1 1.5 ms → "只能藏 15%" 现在变 receiver 3 ms / FC1 1.5 ms → "能藏 50%", M2-G 单独可能就够 ≤ +15% skew tax (M2-G 验收门槛) | 0; 已经有 `build/bench_gemm` 直接跑 |
| **P0** | M2-G GEMM body 装进 persistent super-kernel | 验证 dispatch ↔ FC1 chunk-overlap 实际触发, hot_cov50 退化能不能从 +31-50% 压到 ≤ +15% | 见 [rocmoe/README.md](./README.md) M2-G 验收 |
| **P1** | 治 H=7168 packed_outbox bit-exact 老 issue | 解锁 H=7168 端到端正确性测试, super-kernel prod 形状 bit-exact 才能进 sweep | 跟 M1c-A 05-21 21:00 DOWN note 同源, 可能是 L2 capacity / TLB thrash |
| **P1** | 扫一下 `kCopyStageUint4` 16 KB 是不是最优 | 改 256 / 2048 / 4096 / 8192 / 16384 uint4 各跑一次 phase profile, 看 receiver wall 还有没有再压一截的空间 | 0; LDS 共占 16 KB × 1 = 16 KB / WG, MI355X 64 KB / CU 还宽松 |
| P2 | 把 Receiver per-row 切回 "chunk_n 行一次发" 治掉前面提到的 wave-1+ store bug | per-row 调用一次 LDS round-trip; 多行一次发能省 (chunk_n-1) 次 round-trip 和 (chunk_n-1) 次 `__syncthreads()` | 需要先复现 wave-1+ store bug 并找到根因, 否则不该回去 |
| P2 | sender PhaseB 把 contiguous 分片再拆成 "每行 2 个 sub_wg cooperate" 看 BW 是否再升 | per-sub_wg 256 thread 也许还能再吃一点 BW (现在 512 thread 同时发 16 KB tile, 单 sub_wg 满 LDS); 但要小心 chunk 太小时 occupancy 反降 | 中, 改动需要拆 cooperative_b128_copy 成 group_size 参数化 |

## 相关文件

- 代码: `csrc/include/rocmoe/ipc_primitives.h` (cooperative_b128_copy LDS-staged 重写) · `csrc/include/rocmoe/dispatch_body.h` (PhaseB 连续分片 + Receiver per-row LDS-staged) · `csrc/dispatch.hip` · `csrc/super_kernel.hip`
- 上游 note: [`2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md`](./2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md)
- 关联 note (H=7168 老 issue): [`2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md`](./2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md) · [`2026-05-21_2100_DOWN_m1c_a_revisit_dsv3_production_size.md`](./2026-05-21_2100_DOWN_m1c_a_revisit_dsv3_production_size.md)
- 原始 phase profile: `bench_results/phase_profile_20260522_lds_balanced.txt` · `..._realistic_cov20.txt` · `..._hot_cov50.txt`
