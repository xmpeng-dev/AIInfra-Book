# Memory Access Patterns — kernel review checklist

> **作用**: 写或 review 任何在 GPU 上搬数据的 kernel 时, 默认先用这 5 个问题过一遍, 再看实现细节。
> **覆盖范围**: HBM ↔ HBM / HBM ↔ XGMI peer / HBM ↔ LDS / LDS ↔ VGPR 全部场景, AMD CDNA3/CDNA4 + NVIDIA H100/B200 通用 (硬件名词差异在文中标出)。
> **配套**: `~/workspace/slab/.cursor/skills/cco-pipeline-overlap/SKILL.md` (super-kernel 内 overlap 实现) · `~/workspace/slab/.cursor/skills/mi355_hardware_aware/SKILL.md` (gfx950 LDS / async-direct-to-LDS / wave 细节) · `~/workspace/slab/knowledge/hardware/gpu-comparison.md` (带宽数字)。
> **从哪里抽象出来**: RocMoE-v2 M1b round 4 / M2-D 的 4 轮失败实验; MonolithEP `scatter_phase` 的 pre-pack 路径; CK / hipBLASLt 的 tile pipeline 模板。具体反例见各小节末尾的 "钉到代码" 段。

---

## 0. 五问 (TL;DR — 把这张表背下来)

| # | 问题 | 触发信号 (代码 smell) | 失败时的物理代价 | 主要修法 |
|---|---|---|---|---|
| **Q1** | Read pattern 跨 row 连续吗? | 内层索引 `src_row = base[idx[s]]`, `idx[]` 不单调 | peer / HBM L2 prefetcher 跨 row 失效, 每行从冷开始 | sender 端 pre-pack 成 dst-order, receiver 一次连续读 |
| **Q2** | Write pattern 跨 row 连续吗? | dst 行号是 routing/permutation 的结果, 不是 `0,1,2,...` | HBM write merge buffer 命中率掉, write amplification | 同样在生产端 pre-permute, 或读后用 LDS 暂存再批量写 |
| **Q3** | Register 是否**同时**背着 read 和 write? | `buf[u] = src[..]; dst[..] = buf[u]` 紧贴, 中间靠 VGPR 槽位过 | VMCNT 强耦合 read/write, 每行只能 1 行 in-flight | 改 `async direct-to-LDS` (gfx950 `buffer_load_lds`) 让 LGKMCNT 和 VMCNT 独立排队 |
| **Q4** | Wave 之间应该锁步还是独立? | 行末有 `__syncthreads()`, 但每行 256 thread 协同的合理性没说明 | 锁步: 4 wave 共担 1 row latency, in-flight 行数 = 1; 独立: 4 wave 同时跑 4 row | 写 / load coalesce 需要锁步; read RTT 隐藏 / 不同 row 之间需要独立, **不能两个都要** |
| **Q5** | 数据流方向跟硬件强 / 弱方向匹配吗? | 选 push 还是 pull / 用哪条链路 / 谁拿主动权, 看上去随意 | 押错方向 = 选了 fire-and-forget 反过来变 request-response, 单 kernel 慢 30 %+ | 写优先 outbound, 读优先 inbound 但要 prefetch hint; cross-tier 优先 LDS → HBM → XGMI, 不反向 |

剩下章节: 每个问题展开成 (定义 / 信号 / 修法 / 钉到代码) 四段。

---

## Q1 — Read pattern 跨 row 是否连续

### 定义

判 "连续": 沿 row index `s = 0, 1, 2, ...` 推进时, 物理地址 `&data[indirection(s)]` 是否单调增 + 步长 = sizeof(row)。**不是问 row 内部连续不连续** (row 内部基本都是连续的, 自然 coalesce), 是问 **行与行之间**。

### 触发信号

任何 indirection layer 在 row index 上, 例如:

```cpp
int t   = encoded_idx[s];                 // ← s 是新顺序, t 是原顺序, 一般跳
src_row = &base[t * H];                   // ← 跨 row L2 prefetcher 跨断了
```

或:

```cpp
for (s = 0; s < N; ++s) {
    dst_row = &pool[perm[s] * H];         // ← perm 是 routing 决定的, 跳
    cooperative_b128_copy(dst_row, ...);
}
```

### 物理代价

GPU L2 cache prefetcher (CDNA3/4 和 H100 都有) 在检测到连续 cache line 访问时会自动向前预取 4-16 line。如果 row-to-row 跳了, prefetcher **每行都从冷启动**, 命中率从 ~80% 掉到 ~10%, 等效 inbound 带宽下降 5-7×。

XGMI 链路上更严重: peer 的 L2 prefetcher 跨 row 完全失效, MI355X 的 153 GB/s 链路实际只能用到 30-40%。

### 修法

**核心**: 在**生产端**就把要发的数据 pack 成 dst-order, 让 consumer 一次性 burst 读。这是 MonolithEP push 比 RocMoE-v2 pull 快 30% 的真实原因。

模板:

```cpp
// 生产端 (sender / writer side):
//   sort 路由结果, 写一个本地 packed buffer
__shared__ int counts[N_DST];
for (s = tid; s < num_events; s += WG_SIZE) {
    int dst = routing(s);
    int slot = atomicAdd(&counts[dst], 1);
    packed[dst][slot] = original[s];
}
__syncthreads();
// 发布 counts 给 consumer

// 消费端 (receiver / reader side):
//   一次性读连续的 N_dst 行, prefetcher 全程命中
for (s = tid_in_wg; s < counts[my_src]; s += WG_SIZE) {
    cooperative_b128_copy(local[s], packed[my_src][s]);
}
```

成本: 生产端多一次 local HBM → local HBM 的 sort/pack。MI355X HBM3e 8 TB/s, 比 XGMI 525 GB/s 快 15×, 这笔账划算 (T=2048 × topk=8 × H=7168 × 2 B / 5 TB/s ≈ 0.05 ms, 可忽略)。

例外 (不需要 pre-pack):
- 数据本身就是 dst-order (例如 GEMM 输出已经按 expert 排好)。
- Row 数 < 32 (太少, prefetcher 启动开销大于收益)。
- 单 row 已经 > 64 KB (跨 row prefetch 失效也无所谓, 本来就 row-bound)。

### 钉到代码

| 仓库 | 文件:行 | 状态 | 备注 |
|---|---|---|---|
| `RocMoE/csrc/include/rocmoe/dispatch_body.h:147-154` | RocMoE-v2 pull receiver | **违规** | `peer.input_token_row(src_t)`, `src_t` 由 routing 决定, 跨 row 跳 |
| `MonolithEP/csrc/dispatch.hip:225-310` | MonolithEP push scatter_phase | **合规** | sender 先 sort 写 `pack_perm[base..base+cnt]`, 然后内层 loop 顺序读 `args.hidden[pack_perm[base+s]]` 仍是 indirection, 但 push 是 *写* peer, 写端 fire-and-forget 不靠 prefetcher (转化为 Q5 问题) |

> RocMoE-v2 的修复路径: 给 sender_stage 加一个本地 packed buffer, receiver 改读 `peer.packed[my_rank][...]`。设计草图见 `~/workspace/slab/notes/rocmoe/2026-05-21_1740_compare_monolithep_dispatch.md` §"sender pack" 段。

---

## Q2 — Write pattern 跨 row 是否连续

### 定义

跟 Q1 对称: 沿 row index 推进时, dst 物理地址是否单调连续。**问的是 row 间, 不是 row 内**。

### 触发信号

```cpp
dst_row = &output[pool_pos_table[s] * H];  // ← pool_pos_table 跳, dst 跨 row 不连续
```

或更隐蔽的:

```cpp
// Permute kernel: input 顺序读, output 按 expert id 散开
for (s = tid; s < N; s += WG_SIZE) {
    int e = expert_of(s);
    int slot = atomicAdd(&expert_pos[e], 1);
    output[e][slot] = input[s];           // ← output 写跨 expert 跳, 跨 row 散
}
```

### 物理代价

HBM write merge buffer (HBM3e 8 stack × 32 channel) 每 channel 有 ~4 个 16-line 写聚合 buffer。**连续写**会被合并成一次 burst (一次 256 byte burst 占 channel 几 ns); **散写**每行触发一次 channel 切换, 整体写带宽下降 2-3×。

NVIDIA H100/B200 上同样, GDS write coalescing 失效后 effective HBM3 bw 从 3 TB/s 掉到 ~1.5 TB/s。

### 修法

| 场景 | 解法 |
|---|---|
| Pure permute (input 顺序, output 散) | 用 LDS 做整 row 暂存: input → LDS → output, **写 HBM 时按 LDS 顺序连续写, 不是按 routing 顺序** |
| 跨 row 物理上必须散 (e.g. 不同 expert 的 token 散到不同 expert 块) | 接受写代价 OR 改成 producer-side pre-bucketed buffer (跟 Q1 对称) |
| Strided write (e.g. `output[s * stride]`, stride 不是 cache line 整数倍) | 调 layout 让 stride = N × 128 byte (CDNA cache line) |

LDS 暂存模板:

```cpp
// Stage 1: input → LDS (input 顺序读, 自然 coalesce)
__shared__ bf16_t lds_row[H_PAD];   // H_PAD 加几个元素破 LDS bank conflict
cooperative_b128_copy_to_lds(lds_row, &input[s * H], H * 2);

// Stage 2: LDS → HBM output (write 顺序是 LDS 内部的, 永远连续)
cooperative_b128_copy_from_lds(&output[pos * H], lds_row, H * 2);
```

注意 `cooperative_b128_copy_to_lds` 用 `ds_write_b128` (LGKMCNT 队列), `cooperative_b128_copy_from_lds` 是 `ds_read_b128` + `global_store_b128` (LGKMCNT + VMCNT 不同队列, 可流水)。

### 钉到代码

| 仓库 | 文件:行 | 状态 | 备注 |
|---|---|---|---|
| `RocMoE/csrc/include/rocmoe/dispatch_body.h:151-153` | RocMoE-v2 pull receiver dst | **合规** | `dst_row = self.expert_token_pool_row(e, pool_idx)`, `pool_idx = base + s`, 单调 |
| `MonolithEP/csrc/dispatch.hip:310-330` | MonolithEP push 写 peer | **合规** | `pack_row = le * MPE + packed_off_base + s_local`, 单调 |
| 假想的 naive permute kernel | — | **常见违规** | input 顺序读 + output 按 routing 散写, 没用 LDS 暂存 |

---

## Q3 — Register 是否同时背着 read 和 write

### 定义

经典 GPU dataflow 有三段队列:

| 队列 | 管什么 | 计数器 (CDNA) | 计数器 (NVIDIA Hopper+) |
|---|---|---|---|
| VMEM load/store | HBM / XGMI 读写 (经 VGPR) | `VMCNT` | LDGSTS counter |
| LDS load/store | LDS 读写 | `LGKMCNT` | LDGSTS_LDS |
| MFMA / WMMA | tensor core | `SC0/SC1` | MMA scoreboard |

如果数据流是 `HBM → VGPR → HBM`, **VGPR 同时被 read 和 write 占用**, 这两个动作都走 VMCNT。`vmcnt(0)` 必须等 read 全部回来才能开始 issue write, 而 write 又必须等 read 释放 VGPR —— 单 row 只能 1 个 in-flight, throughput 卡死。

### 触发信号

```cpp
for (i; ...; i += step) {
    uint4 buf[UNROLL];
    for (u) buf[u] = src[i + ...];   // VMEM read → VGPR
    for (u) dst[i + ...] = buf[u];   // VGPR → VMEM write
}
```

或更隐蔽的:

```cpp
// 用同一组 VGPR 读 A 块 + 写 B 块
for (k) {
    a_frag = load_global(A + k);
    process(a_frag);
    store_global(B + k, a_frag);
}
```

### 物理代价

`buffer_load → s_waitcnt vmcnt(0) → buffer_store` 的 RTT 在 MI355X 上 ~600 ns (XGMI inbound) / ~150 ns (本地 HBM)。Unroll=4 时每 thread 同时 in-flight 4 个 uint4 (= 64 byte), 总共 256 thread × 64 byte = 16 KB in-flight, **同一 row 14 KB 已经吃满了**。下一行必须等当前行 write 写完才能开始 read。

对比: async direct-to-LDS 模式下, LGKMCNT 和 VMCNT **独立排队**, 同一 WG 可以同时:
- 用 `buffer_load_lds_b32` 把 row k+1 灌进 LDS (走 VMCNT)
- 用 `ds_read_b128` + `global_store_b128` 把 row k 从 LDS 写到 HBM (走 LGKMCNT + VMCNT 不同 phase)

VGPR 退出关键路径, in-flight rows 可以 double 或 triple。

### 修法

CDNA4 (gfx950) async direct-to-LDS (引用 `~/workspace/slab/.cursor/skills/mi355_hardware_aware/SKILL.md` §"Async Direct-to-LDS"):

```cpp
// Phase A: HBM/XGMI → LDS (不经 VGPR)
__builtin_amdgcn_global_load_lds(src_ptr, lds_buf, num_bytes,
                                  /*offset=*/0,
                                  /*aux=*/0);
// LGKMCNT 异步累加, 不挂 VGPR

// Phase B: 等 LDS 写完, 然后 LDS → HBM
__builtin_amdgcn_s_waitcnt(0x0fff);  // 只等 LGKMCNT 那一边
// LDS → HBM 用普通 ds_read + global_store, VGPR 短暂吃 ds_read 结果就吐
```

CDNA3 (gfx942 MI300X) 也有 `buffer_load_lds_b32` 但 unroll 不够好; CDNA4 上才推荐用。

NVIDIA Hopper+ 等效物: `cp.async.bulk` (TMA, H100+) / `cp.async` (A100+)。CUDA pattern 见 `~/workspace/slab/knowledge/libraries/composable-kernel.md` §"Pipeline scheduling primitives"。

### 钉到代码

| 仓库 | 文件:行 | 状态 | 备注 |
|---|---|---|---|
| `RocMoE/csrc/include/rocmoe/ipc_primitives.h:166-192` | `cooperative_b128_copy` | **违规** | unroll 内 read 紧贴 write, VGPR 全程被占, in-flight 单行 |
| `MonolithEP/csrc/dispatch.hip:298-333` | 同模板 | **违规** | unroll=2-4 但仍是 VGPR-only buffer; 因为是 *push* 写 peer, write 本身 fire-and-forget, 损失部分被掩盖 |
| `RocMoE-bak/csrc/.../fc1_gemm.hip` (假设) | MFMA hot loop | **合规** | `mfma_tile.h` 已经用 LDS double-buffer + `s_waitcnt lgkmcnt(0)` 解耦 |

---

## Q4 — Wave 之间应该锁步还是独立

### 定义

一个 WG 通常含 4-16 个 wave (AMD wave=64 thread; NVIDIA warp=32 thread, 一个 256-thread CTA = 8 warp)。**这些 wave 应该一起做同一件事 (锁步), 还是各做各的 (独立)?** 不同选择给不同物理优势, **不能两个都要**。

### 锁步: 全 WG 协作一个 row

```cpp
cooperative_b128_copy(dst_row, src_row, H * 2);   // 256 thread 一起读 1 row
__syncthreads();
```

| 优 | 劣 |
|---|---|
| 256 thread × 16 byte = 4 KB / step, **L2 cache line 利用率最高** (32 line per step) | in-flight 行数 = 1, 单行 RTT 完全暴露 |
| Write merge buffer 整 row burst | wave 之间必须同步, 任何 wave 落后整 WG 等 |
| 适合 outbound write (fire-and-forget) + 需要每行高带宽 | 对 inbound read (RTT-bound) 是浪费 |

### 独立: 每个 wave 各做自己的 row

```cpp
int wave_id = tid / 64;
int lane    = tid & 63;
for (s = sub_wg * waves_per_wg + wave_id; s < N; s += sub_wgs * waves_per_wg) {
    cooperative_b128_copy_wave(dst[s], src[s], H * 2, lane);
    // 不需要 __syncthreads, 因为不跟其它 wave 共享 row
    if (lane == 0) atomic_add_agent(counter, 1u);
}
```

| 优 | 劣 |
|---|---|
| in-flight 行数 = waves_per_wg (4-8), **掩盖单行 RTT** | 每 wave 64 thread × 16 byte = 1 KB / step, 比锁步少 4× cache line | 
| Wave 之间天然不阻塞, sender slow 只阻 1 wave | 必须每 wave 独立 atomic publish, signal 频率高 |
| 适合 inbound read (RTT-bound) + 每行流量 < L2 line × wave_size | 对 outbound write 没收益 (write 本来就不卡 RTT) |

### 怎么选

| 信号 | 选 |
|---|---|
| HBM / XGMI 是 **write**, fire-and-forget, write queue 深 | **锁步** (MonolithEP push 选这条) |
| HBM / XGMI 是 **read**, RTT-bound, in-flight 受 vmcnt 限 | **独立** (RocMoE-v2 pull 应该但没用) |
| 内层是 MFMA, A/B fragment 从 LDS 来, 跨 wave 共享 LDS 块 | **锁步** (cco-pipeline-overlap Principle 1) |
| 每个 wave 处理完全独立的 tile (M / N split) | **独立** (wave specialization, cco-pipeline-overlap Technique 1.3) |
| 内层 `__syncthreads()` 是 MUST (LDS exchange across waves) | **锁步**, 没得选 |

### 钉到代码

| 仓库 | 文件:行 | 状态 | 备注 |
|---|---|---|---|
| `RocMoE/csrc/include/rocmoe/dispatch_body.h:142-180` | pull receiver 内层 | **选错** | inbound read 用了锁步, 每 WG 只有 1 row in-flight, M1b round 4 实验证 wave 漂移会破 coalesce, 但根本问题是这个设计不该锁步 |
| `MonolithEP/csrc/dispatch.hip:225-378` | push scatter | **合规** | outbound write 用锁步 (`__syncthreads` 行末), 4 wave 共担 1 row, write queue 深, link util 37 % |
| `mfma_tile.h` 内 MFMA hot loop | — | **合规** | wave 之间共享 LDS A/B tile, 必须锁步 |
| `cco-pipeline-overlap` Technique 1.3 | wave specialization | — | producer wave + MFMA wave 各做各的, 独立 |

---

## Q5 — 数据流方向跟硬件强 / 弱方向匹配吗

### 硬件分层带宽 (MI355X, 8-GPU 节点; H100 SXM5, 8-GPU 节点)

| 链路 | MI355X | H100 | 方向特性 |
|---|---|---|---|
| LDS | 256 GB/s / CU / dir × 256 CU ≈ 65 TB/s 总 | ~33 TB/s | 双向对称, fully pipelined |
| HBM | **8 TB/s** | 3.35 TB/s (HBM3) | read / write 基本对称, write merge buffer 让 burst write 略快 |
| L2 → HBM | 同上 | 同上 | — |
| Per-GPU XGMI (CDNA) / NVLink (NV) | **525 GB/s** (7 × 75 GB/s) | 900 GB/s (NVL4) | **outbound write 是 fire-and-forget**, **inbound read 必须 RTT** (CDNA inbound RTT ~600 ns, NV ~250 ns) |
| Network (Infinity Fabric off-node / IB) | ~50-100 GB/s | 同 | 跨节点, 不在本 doc 讨论范围 |

数字源: `~/workspace/slab/knowledge/hardware/gpu-comparison.md`。

### 关键非对称

1. **HBM ≫ XGMI ≫ inter-node** —— 跨层数据流应该 *把工作留在尽可能高层级*。例如 sender pack (Q1 解法) 是用 8 TB/s HBM 换 525 GB/s XGMI, 划算。
2. **XGMI outbound write fire-and-forget, inbound read RTT-bound** —— 这是 push/pull 选型的根本。Push 不付 RTT, pull 必须付。
3. **HBM 跟 LDS 之间, LDS → HBM 比 HBM → LDS 略快** (write merge buffer)。所以 staging 模式优先 *先把数据吸进 LDS, 再 burst 写出去*, 不要反向。

### 触发信号

任何选 push / pull / 谁拿主动权的决策, 但 PR 说明里没列硬件方向匹配。例:

| 看起来对称, 其实押错方向的例子 |
|---|
| "用 pull 因为它更优雅 / 跟数据流匹配" — 没说 RTT 代价 |
| "用 push 因为 sender 知道 routing" — 没说 outbound contention (8 sender × 1 receiver 时 8-way write 抢 link) |
| "用 IPC scatter" — IPC scatter 在 1 to N 时是 N outbound, 在 N to 1 时是 N inbound RTT, 量纲差 N× |

### 决策表

| 场景 | 推荐方向 | 反例 |
|---|---|---|
| **1 sender → N receiver** (例如 broadcast weight) | sender push (1 outbound × N) | receiver pull = N inbound RTT, 慢 5-10× |
| **N sender → 1 receiver** (例如 dispatch fan-in, all-to-all in) | receiver pull (1 inbound × N) **或** push + 抗 contention | sender push 出现 N-way outbound 抢 link, MonolithEP `g=0` 1.34 ms spin 就是这个 |
| **N sender → N receiver, balanced** (例如 all-to-all token shuffle) | 看哪边更适合 contention: write 抢 link 还是 read 抢 RTT | 默认 push, 因为 fire-and-forget; 但要做 contention 测试 |
| **HBM ↔ LDS** | 数据进 LDS 后再消费; LDS 出来时一次 burst 写 HBM | 在 LDS 内频繁 read-modify-write 写回 HBM, 浪费 LDS 带宽 |
| **LDS ↔ Register** | MFMA hot loop 内: `ds_read → MFMA → result in VGPR`, 不写回 LDS | 写回 LDS 再读, 加一次 32 cycle 同步 |

### 钉到代码

| 仓库 | 决策 | 状态 | 备注 |
|---|---|---|---|
| `RocMoE/csrc/include/rocmoe/dispatch_body.h` | N sender → 1 receiver 用 pull | **方向押对** | 避免 8-way outbound contention; 但 Q1 没修, RTT 没掩盖 |
| `MonolithEP/csrc/dispatch.hip` | N sender → 1 receiver 用 push | **方向押错 (在 standalone 上反而快)** | 收 g=0 ready_mask 1.34 ms 同步代价 (架构成本) 换 standalone latency 优势; BF16 sprint closeout 证 FC2 8-way contention 不可消除 |
| `~/workspace/MonolithEP/csrc/combine.hip` | 1 → N pull combine | **方向押对** | wave-broadcast 寄存器归约, 避免 N-way outbound write |
| `RocMoE-v2 plan` | combine 用 pull | **方向押对** | 跟 monolith 反向, 算是 RocMoE-v2 三大架构赌注之一 |

---

## 附录: 工程操作建议

### A. Review 模板 (复制到 PR 描述)

```markdown
## Memory access pattern review

- Q1 (cross-row read 连续?): YES / NO + 一行解释
- Q2 (cross-row write 连续?): YES / NO + 一行解释
- Q3 (Register 同时背 read+write?): NO / YES + 解释 (如 YES 必须列 LDS staging 计划)
- Q4 (Wave 锁步 vs 独立?): LOCKSTEP / WAVE-INDEPENDENT + 一行理由 (引用 Q5)
- Q5 (硬件方向): 选 push/pull/in-LDS 的硬件理由
```

### B. 量化触底 (用 rocprof / Nsight)

| 信号 | 工具 | 命令 | 期望 |
|---|---|---|---|
| L2 read prefetch hit rate (Q1) | rocprof | `--pmc TCC_HIT_sum, TCC_MISS_sum` | `HIT / (HIT + MISS) > 0.7` |
| HBM write merge effectiveness (Q2) | rocprof | `--pmc TCC_EA_WRREQ_64B` | 每行连续时 64B request 数 ≈ row_bytes / 64 |
| VMCNT stall (Q3) | rocprof | `--pmc SQ_WAIT_INST_VMEM` | `< 5%` |
| Wave occupancy (Q4) | rocprof | `--pmc SQ_WAVES, SQ_BUSY_CYCLES` | 看是否所有 wave 都 active |
| XGMI link util (Q5) | rocprof | `--pmc TCC_EA0_WRREQ_GMI_*` | per-direction 算 bytes/ms vs 525 GB/s |

NVIDIA 等效: Nsight Compute 的 `lts__t_sector_op_*` (Q1/Q2), `smsp__inst_executed_op_*` (Q3), `sm__warps_active` (Q4), `nvlink__throughput` (Q5)。

### C. 反模式速查 (M1b round 4 的教训)

| 看起来像优化的反模式 | 实际后果 | 正确做法 |
|---|---|---|
| 删掉行末 `__syncthreads()` 让 wave 跑得快 | wave 漂移破 cache line coalesce, 反而慢 | 不动 sync, 改换 Q4 的 wave-independent 设计 |
| 把 dispatch WG 数从 64 涨到 128 求更多并行 | 撞 XGMI 链路饱和, contention 反增 | 不动 WG 数, 改换 Q3 的 LDS staging 或 Q1 的 sender pack |
| 加大 unroll 让每 thread 同时 in-flight 更多 uint4 | VGPR 爆掉, occupancy 掉 | 改 async direct-to-LDS, VGPR 退出 |
| "我用了 atomic 所以没数据 race" | atomic 不保证 ordering vs 其它 store; 必须配 `__threadfence` + correct memory scope | 看 `cco-pipeline-overlap` Principle 3 |
