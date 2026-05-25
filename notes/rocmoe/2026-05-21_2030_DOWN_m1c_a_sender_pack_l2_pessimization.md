# 2026-05-21 20:30  M1c-A [DOWN] — Sender pack 因为撞穿 peer L2 被回退

> 时间: 2026-05-21 19:30 → 20:30 (Asia/Shanghai)
> 项目: rocmoe
> 硬件: 8x AMD Instinct MI355X (gfx950, mi355-gpu-7), SLURM job 13489
> 容器: xiaoming-dev (podman, ROCm 7.2 / PyTorch 2.10)
> 上一节点: M2-D BASELINE (dispatch body 装进 persistent super-kernel, T=2048 in-super-kernel dev wall 1.669 ms) [2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md]
> 代码: 新增 `csrc/include/rocmoe/{moe_config.h, workspace.h, dispatch_body.h}` + `csrc/{moe_config.cpp, dispatch.hip, super_kernel.hip}` 共 ~250 行 (3-phase sender split + `packed_outbox` 对称缓冲), 跑通 bit-exact, **bench T=2048 dev wall 1.568 → 2.354 ms = +50 %**。第一轮 review 发现一个明确的 logic bug — sender 写 `peer.packed_outbox[my_rank=S][e][s]`, receiver 读 `peer.packed_outbox[my_rank=D][e][s]`, 两个槽位物理不同 — 改成 sender 写 `self.packed_outbox[dst_rank=D][e][s]` (PULL 语义) 后 ctest 8/8 PASS, 但 dev wall 还是 +50 %, **证实 L2 容量崖是真实的性能瓶颈, 不是 bug 表象**。代码保留在 tree 上供 review (用 `ROCMOE_DISPATCH_USE_PACKED_OUTBOX` 编译开关切换)。
> 本轮 flag: **`DOWN`** (代码保留, 默认不启用; 撞 L2 容量崖的判断已被对称性修复后的重测确认)
> 配套补的领域知识: [`slab/knowledge/kernels/memory-access-patterns.md`](../../slab/knowledge/kernels/memory-access-patterns.md) (5 问 review checklist, 这次实验正是它的第一条反例 — Q1 contiguous-read 让步给 Q3 register/cache staging)

## TL;DR

试了一遍 [跨项目 dispatch 对比 note](./2026-05-21_1740_compare_monolithep_dispatch.md) 里
对 MonolithEP push 慢 ~43 % per-token 的"接收端 scatter read 不连续"假设修复方案 ——
sender 端用 `packed_outbox[dst][le][slot][H]` 把每个目标 expert 的 token 预先紧密打包, receiver
改读 `peer.packed_outbox[my_rank][e][s]` 让 s 维度物理连续。**bit-exact 全 PASS, ctest 8/8,
但 T=2048 device wall 从 1.568 ms 退化到 2.354 ms (+50%), 全 T sweep 均一 退化, 无法翻身**。

定位过程用 2 个编译开关把 +786 µs regression 切成三块:

| 拆解项 | T=2048 delta | 占比 |
|---|---|---|
| M1c-A infra 本身 (4 个 grid_sync + local mirror writes) | +4 µs | 0.5 % (噪声) |
| Receiver 改读 `peer.packed_outbox` (L2 capacity miss) | **+546 µs** | **69 %** |
| Phase B 本地 HBM-to-HBM 打包 copy 本身 | +236 µs | 30 % |

根本原因 — M1b 的 receiver scatter read 不是 bug, 是 **topk dedup 红利**: peer 的 `input_token_buf` 只有
`T·H·2 = 28 MB` 完整落在 MI355X 的 32 MB L2 里, topk=8 意味着同一行被 8 个 dst rank 各读一遍,
第 1 个 peer 读完后剩下 7 个全是 L2 命中, 散列访问的实际代价被 L2 cache 抹掉了。把数据按 (dst, le)
预打包就 8× 复制了 token 内容, `packed_outbox` 一下变 `num_ranks · epg · max_recv · H · bf16 ≈ 3.59 GB`
(T=2048, max_recv=8192) — peer L2 装不下, TLB 也撑不住 (每 bucket 117 MB / 2 MB-page = 60 页 vs M1b 14 页),
原本"连续读优于跳读"的 Q1 假设输给了 Q3 cache/staging 容量约束。

修法: 把方向反过来 — **保住 input_token_buf 28 MB 的 L2 红利**, 改在 receiver 端做小排序让
`src_t` 在 (e, src_rank) bucket 内单调, peer 的 28 MB 还在 L2, 但 receiver 的 HBM burst 拿到了 prefetcher
扶持。M1c 重新规划 → 见 §5。

## 1. 进入 M1c-A 的输入

M1b 跑出来标准基线 (这次 DOWN 实验的 ground truth):

| T | M1b dev wall (= M1c-A revert 重测一遍, 同一个 SLURM allocation) |
|---|---|
| 512  | 0.435 ms |
| 1024 | 0.802 ms |
| 2048 | **1.568 ms** |
| 4096 | 3.090 ms |

Scaling 线性, 1.568 / 0.802 ≈ 1.95, 3.090 / 1.568 ≈ 1.97 → 带宽 bound 没变。

跨项目对比 note 里把 RocMoE-v2 pull 跟 MonolithEP push 拉到同一 T (T=8192):

| 路径 | 同 T=8192 per-token 时延 |
|---|---|
| MonolithEP push  | 0.525 µs / tok |
| RocMoE-v2 pull   | 0.752 µs / tok |

差距 +43 %, 根因当时定为"pull 多一个 inbound read + atomic 的 round-trip, scatter pattern 让 peer L2
prefetcher 无效"。看起来很合理 — receiver 现在读的是 `peer.input_token_row(src_t)`, `src_t` 是
`src_index_table[e][src_rank][s]` 解码出来的, 跟 `s` 的递增没有关系, peer 的 HBM 看到的是个非单调的索引
序列, 每行一个 fresh cache line miss, prefetcher 完全用不上。

按照 [memory-access-patterns 5 问 checklist](../../slab/knowledge/kernels/memory-access-patterns.md) 推, 该改两点:
1. **Q1 (cross-row 读连续)** — sender 端把要发的 token 提前打包成 `packed_outbox[dst][le][slot][H]`,
   receiver 改读 `peer.packed_outbox[my_rank][e][s]`, s 维度物理连续, L2 / HBM burst engine 立刻有用武之地。
2. **Q5 (push vs pull 方向)** — 仍然保持 pull (XGMI 出口竞争 / outbound write 树形 fan-in 还是 push 的痛点),
   只是让 pull 的目标地址走连续。

赌注: T=2048 dev wall 1.568 → ~1.0 ms (-300~-500 µs)。代价: 多一遍 sender-side 本地 HBM→HBM 的 pack
copy, 估算 ~30-125 µs (8 sub_wg × 256 events × ~480 ns)。

## 2. 做了什么

### 2.1 数据结构: 给 `MoEConfig` 加 3 个对称缓冲字段

```
off_packed_outbox      // [num_ranks][epg][max_recv_per_e][H] bf16, 对称
off_local_send_log     // [num_ranks][epg][max_recv_per_e]    i32, 本地 mirror
off_local_send_count   // [num_ranks][epg]                    u32, 本地 mirror
```

`packed_outbox` 是大头 — T=2048 H=7168 max_recv=8192 时 **3.59 GB / rank**。MI355X 单卡 288 GB HBM,
8 rank × ~14 GB workspace ≈ 112 GB, 装得下, 但已经是原 workspace 10 GB 的 1.4×。

`local_send_log` / `local_send_count` 是 sender 内部 mirror, 只在本卡可见, 1 MB + 128 B 量级, 微不足道。

### 2.2 Dispatch sender 拆 3 phase + 4 个 grid_sync

把原来一个 `dispatch_sender_stage` 拆成:

```
Phase A (sub_wg==0 of partner=dst_rank only):
    扫 (t, k) 事件, atomicAdd 拿 slot, 同时写:
        peer.src_index_table[le][my_rank][slot] = encoded(t, k)    (XGMI 4-byte, 跨卡 metadata)
        self.local_send_log[dst][le][slot]      = encoded(t, k)    (本地 mirror)
        peer.expert_recv_count[le][my_rank]     = smem_counts[e]   (XGMI, 跨卡 metadata)
        self.local_send_count[dst][e]           = smem_counts[e]   (本地 mirror)
    *不* publish send_done_flag — 等到 Phase B 完成后再 publish, 否则 peer 看到 flag 后会拉
    一个空的 packed_outbox。

grid_sync<0>  (intra-rank, ensure all sub_wgs of all partners see local_send_*)

Phase B (ALL 8 sub_wgs of partner=dst_rank participate):
    for e in [0, epg):
        n = self.local_send_count[dst_rank][e]
        for s = sub_wg; s < n; s += kSubWGs:
            encoded = self.local_send_log[dst_rank][e][s]
            t = encoded >> 5
            cooperative_b128_copy(self.packed_outbox_row(dst_rank, e, s),
                                  self.input_token_row(t),
                                  H * 2)
            __syncthreads()
    这里把 sender 工作量从 1 个 WG / partner 横铺到 8 个 sub_wg / partner, 跨 8 个 partner
    总共 64 WG 并行做 LOCAL HBM-to-HBM 拷贝, 估算 ~30 µs 单段。

grid_sync<1>  (intra-rank, ensure all sub_wgs done writing packed_outbox)

Publish (sub_wg==0 only):
    fence_system + atomic_store_system_release on peer.send_done_flag

grid_sync<2>  (intra-rank, ensure flag drain)

B1 (blk==0 only):
    phase_barrier<kMaxRanks>(...)    // cross-rank, peer 看到 flag = N 才能往下

grid_sync<3>  (intra-rank, ensure B1 broadcast)

Receiver (all sub_wgs):
    每行改读 peer.packed_outbox_row(my_rank, e, s)  ← Q1 contiguous read 落地
```

`super_kernel.hip` 同步改造: production kernel 走新的 4-grid_sync 流; skeleton kernel 保留
legacy `dispatch_sender_stage` (它只用来跑 num_actual_tokens=0 的 cross-rank barrier 烟测,
不需要 pack, 也不能依赖 packed_outbox 真填了数据)。

`PhaseBarrierSignal::grid_ctr[4]` 正好 4 个槽, 一次性吃干, 不需要槽位复用。

### 2.3 调试: 用编译开关把 regression 拆三块

`bit-exact 都 PASS` 之后立刻 bench, 结果直接 **+50% regression**, 远超估算的 +30~125 µs phase B
overhead。为了定位, 加两个开关:

```c
#ifdef ROCMOE_M1C_A_PHASE_B_NOOP
    return;          // 在 dispatch_pack_phase_b 开头, 不做 token copy
#endif
#ifdef ROCMOE_M1C_A_LEGACY_RECV
    bf16_t* src_row = peer.input_token_row(src_t);   // 回到 M1b 的 receiver 读法
#else
    bf16_t* src_row = peer.packed_outbox_row(my_rank, e, s);
#endif
```

四个 build 跑同一个 SLURM allocation, 同一份 input, 同一个 T-sweep:

| build | flags | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| M1b baseline (`revert`) | — | 0.435 | 0.802 | **1.568** | 3.090 |
| M1c-A infra + legacy recv (phase B off) | `LEGACY_RECV + PHASE_B_NOOP` | 0.438 | 0.810 | **1.571** | 3.087 |
| M1c-A infra + new recv (phase B off) | `PHASE_B_NOOP` | 0.473 | 0.886 | **2.117** | 4.191 |
| M1c-A 全套 | — | 0.537 | 1.001 | **2.354** | 4.715 |

(全部是 critical-path device wall p50, ms.)

T=2048 时拆分:

| 增量项 | µs | 解释 |
|---|---|---|
| 4 grid_sync + local mirror 写入 | +4 | M1c-A 基础设施开销, 噪声级 |
| Receiver 改读 packed_outbox 地址 | **+546** | 即使 packed_outbox 全是 0 (phase B noop), 读它也比读 input_token_buf 慢这么多 |
| Phase B 本地 HBM→HBM token copy | +236 | 跟估算 30-125 µs 量级一致 |
| **合计** | **+786** | M1c-A 全套 wall |

最贵的一项不是计算 (phase B copy 只 236 µs), 而是 **receiver 端读 packed_outbox 的地址**这个看起来零工作量的改动 — 它独占了 70 % 的 regression。

## 3. 为什么"连续读"反而更慢 — peer L2 容量崖

5 问 checklist 里 **Q1 (contiguous read) 和 Q3 (register/LDS/cache staging 容量) 同时打架时, 谁赢看绝对尺寸**。
M1b 那套被认为是 scatter 的 receiver 其实享受了一个隐藏红利:

- `peer.input_token_buf` 大小 = `T · H · 2 = 2048 · 7168 · 2 = 28 MB`
- MI355X 单卡 L2 cache = **32 MB** (每 XCD 各 24 MB, 双 NPS1 模式 ~32 MB / GPU 视为全局可见)
- **topk=8 = 同一个 token row 被 8 个 expert 消费, 这 8 个 expert 分布在 1~8 个 rank 上** (取决于 routing; 实测 8 个 global expert 在 num_experts=32/epg=4 下期望覆盖 ≈ 7.4 个不同 rank, 但 worst-case 也可能落在 2~3 个 rank 上)
- 不管这些 expert 落在几个 rank, **每个 (dst_rank, local_e) 消费配对都触发一次 `peer.input_token_row(src_t)` 读**, 都打到 sender 这同一个 14336 B 物理行
- 第一次读 fill sender L2, 之后所有读 (同一 dst rank 内的多个 local expert, 或者不同 dst rank, 不管时间错开还是几乎并行) 都是 sender L2 hit
- 整张 input_token_buf 28 MB < 32 MB L2, **全部驻留**, 散列 src_t 索引的代价被 sender L2 hit rate 抹平 —— **dedup 的关键是 "sender 端 28 MB working set 全驻留 L2", 跟"被几个 rank 读"无关**
- 对比 Megatron / MonolithEP 的 push (all_to_all) 路径: sender 端就把 topk 复制展开成 N×T×H 的 send buffer, 数据已经在 HBM 物理复制了, 拿不到这个 L2 dedup 红利。这恰恰是 RocMoE-v2 pull 相对 push 仅有的结构性优势, M1c-A 错就错在把它扔了

把这个 dedup 红利写明白后, M1c-A 的失败就顺理成章了:

- `packed_outbox` 的设计假设是 "每个 (dst, le, slot) 独占一行", 数据按 topk 复制了 8 份
- 实际占用 = `num_ranks · epg · max_recv · H · 2 = 8 · 4 · 8192 · 7168 · 2 = 3.59 GB`
- 远远超过 L2 (32 MB), **每行都是 fresh HBM miss**, 没有任何缓存命中
- TLB 也跟着炸 — 每 (e, src_rank) bucket 跨越 `max_recv · H · 2 = 117 MB`, 按 2 MB hugepage 算是 ~60 页, MI355X TLB ~32 项, thrash

实测两件事:

- `phase_b_noop` build (phase B 不写, packed_outbox 全是 0) 跟全开 build 在 receiver 段几乎一样慢 — 证实
  慢的不是 phase B 写入污染, 是 **读访问模式本身的 cache miss penalty**, 与数据内容无关。
- `LEGACY_RECV + PHASE_B_NOOP` build (基础设施全套, receiver 回到读 input_token_buf) 与 M1b baseline 几乎相等
  (1.571 vs 1.568, 4 µs 差异在 hipEvent 噪声内) — 证实 grid_sync × 4 / phase A 多一份 local mirror 写入 / 大 workspace
  分配带来的潜在 TLB 影响, 加起来 < 1 % 的开销, 不构成阻塞。

## 4. 跟"跨项目对比 note"的假设对齐

[2026-05-21 17:40 跨项目对比 note](./2026-05-21_1740_compare_monolithep_dispatch.md) §3.2 把
"RocMoE-v2 pull 比 MonolithEP push 慢 ~43 % per-token" 归到三个因素:

| # | 假设来源 | 这次 M1c-A 的验证 |
|---|---|---|
| (a) inbound read RTT + atomic round-trip | 对的 — pull 一定多一个跨卡 RTT, 这是结构性的, 不打算消除 |
| (b) receiver scatter read 让 peer L2 prefetcher 失效 | **错的** — peer L2 在 28 MB working set + topk dedup 下是 100 % 驻留的, 不是 prefetcher 的事, 是 dedup 红利 |
| (c) sender wider parallelism | 部分对 — MonolithEP 整 dispatch 用 256 WG, RocMoE 只用 64 WG, 但 M1b round 3 已经证明 wider parallelism (kSubWGs=16) 反而被 XGMI 撞墙, 也不是这里的瓶颈 |

所以那个对比 note 的"修复方案" (b 路) 这次直接证伪。M1c-A 的代码本身没 bug, 是赌注下错了方向。

## 5. 下一步重新规划

M1c-A 代码保留在 tree 上, **默认 `ROCMOE_DISPATCH_USE_PACKED_OUTBOX=1`**, ctest 8/8 PASS, 但 bench 数据明确显示 +50 %, 不在默认开启路径上演进。一旦未来 `T·H > L2` (例如 T≥8192 + H=7168, input_token_buf ≥ 112 MB), 这条代码路径会自动反转优势, 不要删。

接下来不再追"packed_outbox"这条路, **保留 input_token_buf 28 MB 全 L2 驻留这个红利**, 改在
receiver 端做小幅度排序让 `src_t` 在 bucket 内单调:

### M1c-B' (重新定义) — receiver 端按 src_t 排序后再做 cooperative copy

1. Phase A 不变 — sender 还是写 `peer.src_index_table[e][my_rank][slot] = encoded(t, k)`。
2. 增加 Phase A.5 — receiver 在拿到 expert_recv_count 之后, **在本地 src_index_table 内对每个 (e, src_rank)
   bucket 按 src_t 升序排一次**。Bucket size ≈ 512 (T=2048 平衡 topk=8 时), 4 expert × 8 src_rank = 32 bucket,
   每个 bucket 512 个 i32 元素的 stable sort 在 LDS 里 (counting sort 用 11-bit src_t 范围), 单 sub_wg 一两微秒。
3. Receiver 仍然读 `peer.input_token_row(src_t)`, **但 src_t 现在在 bucket 内单调**, peer 的 HBM burst engine
   能预取下一行, L2 仍然命中 (28 MB 还是放在 L2 里), 拿到 prefetcher 红利。

预期: T=2048 dev wall 1.568 → 1.2~1.3 ms (-15~-25 %)。需要先把
[`slab/knowledge/kernels/memory-access-patterns.md`](../../slab/knowledge/kernels/memory-access-patterns.md)
Q1+Q3 这一对修一下 — 它现在写得只强调 "contiguous 好, scatter 坏", 没强调 working set 跟 cache 容量的 cross-over。
我会在下一次 wire-knowledge 时补一条注释。

### 不再追的方向 (这次实验排除掉的)

- ❌ **任何会让 sender 端实例化 packed_outbox-style 全展开缓冲的设计**, 都会撞 L2 / TLB cliff, 除非 H 大到
  `T·H` 已经 >> L2 (e.g. T=8192 H=7168 时 input_token_buf 已经 112 MB, 才会反过来值得 pack — 但 RocMoE-v2 现在还没准备好 T=8192 的 SLURM allocation)。
- ❌ **任何"按 dst rank 顺序写出 outbox" 的变体** — 跟 packed_outbox 一样 8× 复制数据, 一样炸 L2。
- ⚠️ **wave-granular receiver pull (原 M1c-C 计划)** — 没有 packed_outbox 做铺底, wave 各自打开
  cooperative pipeline 也救不了 XGMI 带宽 bound (M1b round 3 已经撞过)。要做也只能等 M3 之后用
  Layout-P + persistent state machine 把 dispatch / FC1 流量在时间维度上 overlap。

## 6. 验收

| 项 | 状态 |
|---|---|
| M1c-A 代码 (3-phase sender + packed_outbox) 完整保留在 tree 上, 用 `ROCMOE_DISPATCH_USE_PACKED_OUTBOX` 编译开关切换 | ✅ |
| 第一轮 review 暴露的对称性 bug (sender 写 `peer.packed_outbox[my_rank=S]` vs receiver 读 `peer.packed_outbox[my_rank=D]`, 不同物理槽位) 已修复, 改为 PULL 语义: sender 写 `self.packed_outbox[dst_rank=D]`, receiver 读 `peer.packed_outbox[my_rank=D]` | ✅ |
| ctest 8/8 PASS (含 `test_dispatch_smoke` bit-exact 4 个 workload) | ✅ (修复 bug 后) |
| `bench_dispatch` T-sweep 在默认开启 M1c-A 时 +50 %, 在 `ROCMOE_DISPATCH_USE_PACKED_OUTBOX=0` 时回到 M1b baseline 同一数字 | ✅ |
| Progress note 写好, README 时间线更新 | ✅ |
| 新增的 `slab/knowledge/kernels/memory-access-patterns.md` 已 commit / push (df956f0 → d756e8a) | ✅ |
| L2 / TLB cliff 推论补回到 memory-access-patterns 文档 | TODO (下次 wire-knowledge 时一并改) |
| 措辞修正: "topk=8 = 8 个 expert 消费, 不是 8 个 rank 各读一份"; "L2 dedup 发生在 sender L2 上, 跟 dst rank 数无关" | ✅ (本 note §3) |

## 文件改动

M1c-A 完整 net diff (本地树跟 M2-D 落地状态相比):

- `csrc/include/rocmoe/moe_config.h`     — +2 字段 (off_packed_outbox / off_local_send_log / off_local_send_count) + 设计 note
- `csrc/moe_config.cpp`                  — +3 段 align256 分配
- `csrc/include/rocmoe/workspace.h`      — +6 个 accessor (packed_outbox_row / local_send_log_ptr / local_send_count_ptr)
- `csrc/include/rocmoe/dispatch_body.h`  — sender 拆为 3 个 `dispatch_sender_{meta,pack,publish}_phase`, 加 2 个编译开关 `ROCMOE_DISPATCH_USE_PACKED_OUTBOX` / `ROCMOE_DISPATCH_PACK_PHASE_B_DO_COPY`, receiver 端 `#if` 切换两种读地址
- `csrc/dispatch.hip`                    — 改为 4 个 grid_sync 串起 phase A→B→C→cross-rank→Recv
- `csrc/super_kernel.hip`                — production + skeleton 两个 kernel 都用 3-phase sender; skeleton 复用 grid_ctr 槽位 (B4/B5 落到 slot 0/1, 2/3)
- `notes/2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md` — 本 note (新增)
- `notes/README.md`                       — 进展时间线 + 状态 + 下一步 行 (本 note 之后会改)
