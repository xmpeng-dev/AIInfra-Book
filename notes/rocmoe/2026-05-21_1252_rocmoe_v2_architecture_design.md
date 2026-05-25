# 2026-05-21 12:52  RocMoE-v2 架构设计：Layout-P + receiver-pull + MonolithEP hot loop

> 时间: 2026-05-21 12:52 (Asia/Shanghai)
> 项目: rocmoe (workspace `~/workspace/RocMoE/`, 全新 clean 仓库)
> 硬件目标: 8x AMD Instinct MI355X (gfx950, CDNA4) / XGMI 全互联 / 单节点
> 软件: ROCm 7.2 toolchain · hipcc · PyTorch 2.12+rocm7.1 · Primus
> 模型: DeepSeek-V3 671B, 256E EP8 -> 32 local, top_k=8, H=7168, F=2048
> 参考实现:
>   - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/MonolithEP` (性能最高, 34.07 ms / iter @ 3-phase super)
>   - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE-bak` (架构最干净, super 2.21 ms vs iso 4.26 ms = 1.93x)
> 知识来源: `~/workspace/slab/.cursor/skills/cco-pipeline-overlap/SKILL.md` · `mi355_hardware_aware/SKILL.md` · `amd-gemm-optimization/SKILL.md` · `~/workspace/slab/knowledge/moe/dataflow.md` · `knowledge/libraries/_patterns.md` · `knowledge/libraries/composable-kernel.md`

## TL;DR

把 RocMoE-bak 的 **Layout-P + receiver-pull + scoreboard 同步** 接进 MonolithEP 的 **`mfma_tile.h` hot loop + 寄存器归约 combine + persistent role-split 调度**, 形成第三代设计 **RocMoE-v2**。三个核心改动 (按 ROI 排序):

1. **Layout-E -> Layout-P**: 把 `[expert][src][m]` 改成 `[expert][block_b][slot]` (block_b = 32-token 物理块)。彻底消灭 MonolithEP `g=0` 在 ready_mask 上的全局 spin (同步 ~1.34 ms wall)。
2. **sender push -> receiver pull**: dispatch / FC2 出口都改成 receiver 端按需拉, 把 8 条 outbound XGMI 写竞争换成 7 条 inbound 拉; 每个 receive WG 只读自己负责的 block, 不再因为对端慢而全 grid 等。
3. **cross-WG `__syncthreads_system` -> per-pool-block scoreboard**: 把 MonolithEP 的 12 个 cross-WG barrier (per-iter `compute_barrier_*`) 全砍掉, 换成 **64-bit per-block ready bitmap + `atomic_load_acquire`**, 每个 compute WG 只在自己当前 tile 依赖的 block 上 spin, 与其它 WG 物理解耦。

预期: BF16 wall **34 ms -> ~13.5 ms (-60%)**, FP8 (mxfp8 weights + DTOLDS) **<= 8 ms**, 接近 mfma_tile.h 在孤立 GEMM 下测出的 99.3% MFMA roofline。

## 1. Background: 为什么需要第三代

`monolith-moe/` 项目 (4.82 ms / 8x MI355X / 512 t/g) 已经把 single-shape micro-bench 压到极限, 但在训练真实工况 (T_src >= 2048) 反输 PyTorch+RCCL —— 见 [`../monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md`](../monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md)。根因不在 GEMM, 而在两条路径上的**架构性 stall**:

| stall 类型 | 物理位置 | 现象 |
|---|---|---|
| g=0 spin | dispatch -> FC1 边界 | compute WG 在 `ready_mask[expert] == 0xFF` 上全局自旋, 直到 8 个 sender 都把这个 expert 的最后一个 token 写完才能起步; per-(src, expert) 的 fan-in 决定了它必然是临界路径 |
| FC2 scatter contention | FC2 -> combine 边界 | 256 个 compute WG 并发 push 到 8 个 peer 的 combine_buf, 8 条 outbound XGMI 链路被同一时刻 256-way 写抢占 |

MonolithEP 把 GEMM hot loop 调到了 99.3% MFMA, 但 super-kernel 整体只到 ~85% 物理上限, 剩下 15% 全在 stall。RocMoE-bak 在 **架构层** (而不是 kernel 层) 解决了这两个问题, 但 RocMoE-bak 的 GEMM tile 不如 MonolithEP 调得好 —— 把两边的优势合起来就是 RocMoE-v2。

## 2. 两套架构对比 (按层级)

### 2.1 数据布局 (workspace 层)

| 层 | MonolithEP (Layout-E) | RocMoE-bak (Layout-P) |
|---|---|---|
| 单位 | per-(src, expert) 切片, 形状 `[E][S][M_e_max][H]`, M_e_max ~ 96 padded | per-expert block, 形状 `[E][B_e][32][H]`, B_e = ceil(T_e / 32) |
| ready 信号粒度 | per-(src, expert), 共 E*S = 256 个 mask bit (8-bit, src ready) | per-(expert, block_b), 共 sum(B_e) ~ 1.5x T 个 64-bit slot |
| compute WG 取数 | 必须等 expert e 的全部 8 个 src 写完, 因为 GEMM 跨 src 拼成连续 M | 只等当前 tile 落在的那个 block_b 写完, block 之间无依赖 |
| HBM 占用 | 8 * E * M_e_max * H = ~2.5x 真实 token 量 (硬等 fan-in 决定) | 1.5x 真实 token 量 (block 紧凑, padding 只在最后一个 block) |
| 写入路径 | sender 选好 (src, expert), 直写 8 个 peer 对应槽位 | sender 写 routing meta (small), receiver 按 meta pull token (big) |
| 关键代价 | g=0 fan-in spin ~1.34 ms; 内存膨胀 1.7x | 双向元数据交换需 1 个 RTT (small fixed cost ~120 us) |

**结论**: Layout-P 把 **同步复杂度** 从 `O(E*S)` 降到 `O(B_e)`, 同时把同步**实际**位置从 dispatch 出口移到 GEMM tile loop 内部 —— 这才是真正的 fine-grained CCO。

### 2.2 通信模型 (XGMI 层)

| 维度 | MonolithEP (push) | RocMoE-bak (pull) |
|---|---|---|
| 谁主导写 | sender, 直写远端 IPC buffer | receiver, 主动从远端 IPC buffer 拉 |
| 一瞬间的 link 利用 | 8 个 sender 同时往 1 个 receiver 写, **inbound** 拥塞 | 1 个 receiver 同时从 7 个 sender 拉, **inbound** 串行有序 (HW round-robin) |
| 失败模式 | one-slow-sender 阻所有 receive WG (因为 ready_mask 是全 8 bit) | one-slow-sender 只阻自己那块 block_b 的 receive WG, 其它 block 不受影响 |
| 与 compute overlap | sender 写 IPC -> compute 等 ready -> compute GEMM, 三段强串行 | receiver 拉 -> tile-by-tile 解锁 compute, 拉与 GEMM 同卡内 overlap |
| 物理瓶颈 (8 GPU XGMI) | 8 outbound link 同时打满 (write-side congestion) | 7 inbound link 自然 fan-in (read-side OK, MI355X HBM3e 8 TB/s 完全吃得下) |

**MonolithEP push 的优势**: 一次 commit, 不需要协议握手, 在 short-tail 情况下延迟最低; FC2 scatter 也合在同一段。
**RocMoE-bak pull 的优势**: 受 sender 抖动影响小, compute 立刻起步, 整体 critical path 由 max(per-block latency) 决定而不是 max(per-src latency)。

### 2.3 同步原语 (kernel 内层)

| 原语 | MonolithEP | RocMoE-bak | RocMoE-v2 |
|---|---|---|---|
| FC1 / SwiGLU / FC2 间 | 12 个 `compute_phase_barrier` (per-expert, 4/iter * 32 expert) | 0 cross-WG barrier; 全用 per-pool-block atomic_load_acquire | 0 cross-WG barrier (沿用 RocMoE-bak) |
| ready 信号 | atomicAdd_system on `ready_mask[expert]` (8-bit, sender 多次 +1) | atomicOr 64-bit bitmap on `block_ready[block_b]` (sender 一次写) | 64-bit bitmap, **每 block 一次 atomic_store_release** (减半 atomic 流量) |
| spin 协议 | DeepEP 风格 `while (atomicAdd_system(&mask, 0) != 0xFF)`, 每次 spin 都打 atomic | `while (!(atomic_load_acquire(&block_ready[b]) & mask))`, **不打 atomic** (用 acquire load) | 同 RocMoE-bak |
| sender 数量门限 | 必须等满 8 src | 任意 src 完成自己的 token 就 release 对应 block bit | 同 RocMoE-bak; 最坏退化 = 8 bit (与 MonolithEP 等价) |
| post-FC2 combine 同步 | tail_combine 阶段 atomic_add 到 final_out (atomic 串行) | combine pull, 寄存器归约 (atomic-free) | 沿用 MonolithEP 的 ContigCombine 寄存器归约 |

`atomic_load_acquire` 在 gfx950 上展开成 `s_load_dword + s_waitcnt lgkmcnt(0)` + L2 invalidate, 比 `atomicAdd(0)` 便宜约 4x —— 这是 mi355_hardware_aware 里 cache coherence 一节专门点过的 cheapest 同步原语 (见 `~/workspace/slab/.cursor/skills/mi355_hardware_aware/SKILL.md` §6.3)。

### 2.4 GEMM 实现 (compute hot loop)

| 维度 | MonolithEP `mfma_tile.h` (1113 行) | RocMoE-bak `gemm.hip` |
|---|---|---|
| MFMA shape | `v_mfma_f32_32x32x16_bf16` 主, 小 tile 用 16x16x16 | 同 |
| 块切分 | M=128 N=128 K=64, 4 wave/block, double buffered LDS | M=128 N=128 K=64, 同 |
| LDS 写法 | DTOLDS (`buffer_load_dwordx4_lds`) 全程, HBM->LDS 一条 DMA | 普通 `ds_write_b128`, 经过 VGPR 中转 |
| LDS 读写消歧 | XOR swizzle (3-bit, `va ^ (ra & 7)`) 写读端对齐, conflict-free | PAD=8 (row stride 68->72 dword), 浪费一点 LDS |
| K-tail 处理 | predicate-mask, K_TILE 不需对齐 | 要求 K 整除 K_TILE |
| 实测 MFMA 利用率 | 99.3% (rocprof, K=7168 case) | ~85% (P1 phase) |
| VGPR / AGPR 占用 | 256 V + 0 A + 240 B/lane scratch -> 1 wave/SIMD | ~200 V, 1.5 wave/SIMD |
| Occupancy 限制 | LDS-bound, 1 wave/SIMD | VGPR-bound 但有 2 wave headroom |
| 与 super-kernel 集成 | 已包成模板, role-split 可直接调用 | 接口干净但精度调到 P1 阶段 |

**结论**: `mfma_tile.h` 是 MonolithEP 整个项目 6 个月调优的核心产物, 设计阶段就该原样移植过来, 不要重写。RocMoE-v2 的 `csrc/include/mfma_tile.h` 直接 cherry-pick MonolithEP 的对应文件。

### 2.5 调度模型 (persistent kernel 层)

| 维度 | MonolithEP | RocMoE-bak |
|---|---|---|
| 调度方式 | 手 hard-code 角色: WG[0..15]=COMM, WG[16..N+15]=COMPUTE, WG[N+16..]=TAIL | persistent state machine, 每个 WG 进 while(state != DONE) 自循环, role 由 work queue 决定 |
| 工作分配 | 静态 N_COMPUTE_WGS sweep, sweet spot N=192 (硬编码) | 动态 atomic counter steal, T_e 不均时自适应 |
| 添加 phase | 改 grid 配置 + 写新的 if-else | 在 state machine 加一个 case, 不影响其它 phase |
| LDS 共享 | union 多 phase, 但每个 WG 一次只跑一个 phase, 切换得 barrier | role-split 时 LDS 也分 region, 物理隔离 |
| 编译器视角 | role 是常量, 编译期 dead-code-elim 干净, 寄存器最少 | state 是变量, 编译器需要保守处理, 多用 ~10-20 V |

MonolithEP 的静态 role 在 BF16 case (compute 占主导) 占优, 但加 FP8/decomposed bwd 这些新 phase 时, 改 grid 调度是噩梦; RocMoE-bak 的 state machine 加新 phase 几乎零成本, 代价是 ~10 V 寄存器冗余。**对 RocMoE-v2 这个长期项目, state machine 更值得投资。**

### 2.6 Combine 实现 (FC2 出口 -> final_out)

两边路径出奇地一致, 都用 **ContigCombine** (atomic-free, register reduce):

```text
combine 阶段: 每个 WG 负责输出 final_out[token_t] 的一个 row tile
              for src in 0..7:
                read combine_buf[src][t] -> regs (peer DTOLDS or LDS)
                accumulate into reg_acc                                  (寄存器加)
              write reg_acc -> final_out[t]                              (一次性 HBM 写)
```

差别: MonolithEP 的 combine 拿 8 个 src 的数据是从 sender push 进来的 LDS-resident scratch 读, 而 RocMoE-bak 是 receiver 主动 pull peer 的 IPC buffer。物理性能两者都是 1.7 ms / 8x MI355X 量级, 但 pull 版本对 sender 抖动免疫。

## 3. RocMoE-v2 设计

### 3.1 高层结构 (super-kernel 内的 5 个 phase)

```text
phase 0  ROUTE_META       (8 WG)         topk -> per-(src, expert) header, write to peers' meta_buf
phase 1  DISPATCH_PULL    (32 WG)        receiver pull, write block_b -> block_ready bitmap
phase 2  FC1 + SwiGLU     (192 WG)       wave-specialized: 2 wave prefetch / 4 wave MFMA, in-tile fused
phase 3  FC2_PUSH         (32 WG)        per-block result push to peer combine_buf, fire combine_ready
phase 4  COMBINE_PULL     (192 WG)       atomic-free combine, register reduce, write final_out
```

5 个 phase **无 cross-WG barrier**, 用 4 类信号通讯:

| 信号 | 类型 | 触发方 | 等待方 |
|---|---|---|---|
| `meta_ready[src][exp]` | 1 bit | route WG | dispatch WG |
| `block_ready[exp][block_b]` | 64 bit (mask of 8 src bits + tail flag) | dispatch WG | FC1 / FC2 WG |
| `fc2_ready[exp][block_b]` | 1 bit | FC2 WG | FC2_push WG |
| `combine_ready[t][src]` | 1 bit | FC2_push WG | combine WG |

**所有信号都是 64-bit 字, 用 `__atomic_store_n(p, v, __ATOMIC_RELEASE)` 写, `__atomic_load_n(p, __ATOMIC_ACQUIRE)` 读**。在 MI355X 上对应 `flat_store_dword + buffer_wbinvl1_vol` (release) 和 `s_load_dword + s_waitcnt lgkmcnt(0)` (acquire), 单端 ~30 cycle (不打 atomic ALU)。

### 3.2 Layout-P 工作空间布局

```text
ipc_meta_buf      [E][S]                64 B header per (expert, src)        ~64 KB total
ipc_token_buf     [E][B_e_max][32][H]   bf16 token slots                      ~1.5 GB / GPU
block_ready       [E][B_e_max]          uint64 mask (per block scoreboard)    ~64 KB
fc2_intermediate  [E][B_e_max][32][F]   bf16 FC2 output                       ~1 GB / GPU
combine_buf       [T][8][H]             bf16 per-(token, src) shard           varies
combine_ready     [T][8]                uint8 ready bit                       8x token count
final_out         [T][H]                bf16, kernel write-once               varies
```

**B_e_max** 取 `ceil(T_max / 32) + slack`; `T_max = T_src * top_k * num_experts_per_group / 8` (每个 src 实际写到本 GPU 的 token 数上界)。在 DSV3 / T_src=2048 / top_k=8 / EP=8 情况下 B_e_max ~ 64, 32-token block 是 MI355X 一个 wave 直接吃完的最佳粒度 (2 cyc / row * 32 = 64 cyc, 配 LDS double buffer)。

**关键不变量** (写注释里):
- `block_ready[e][b]` 的 bit i = 1 当且仅当 src=i 已经把它分到本 GPU 的、属于 expert e 的、第 b 个 block 的所有 token 写完。
- `block_ready[e][b]` 的 bit 63 = "tail flag", 由本 GPU 的 dispatch WG 在所有 8 个 src 都 OR 进来后置 1, FC1 spin 等这个 bit。
- 这样保证 FC1 看到的 block 总是 8-src-complete, 即使中间某 src 的某个 block 实际是空的 (写 1 个 dummy + 留 padding)。

### 3.3 接 phase 之间的协议

#### dispatch -> FC1
```text
FC1 WG block-tile 循环 (M_TILE = 128, K_TILE = 64):
  for e in expert_chunk_for_this_wg:
    for block_b in 0 .. B_e[e]:
      while ((__atomic_load_n(&block_ready[e][block_b], __ATOMIC_ACQUIRE) >> 63) & 1 == 0) {
        // sleep 64 cycle (s_sleep 1) 减少 spin 功耗
      }
      // 现在保证 block_b 的 32 个 token 在 ipc_token_buf[e][block_b][..][..] 已经全都写好
      mfma_tile_t::run(ipc_token_buf[e][block_b], W_fc1[e], lds_buf, fc1_out_block[e][block_b]);
```
关键点: spin 是 **per-(e, block_b)** 而不是 per-expert; FC1 WG 只阻塞自己当前要消费的那个 block, 其它 block 不受影响。

#### FC1 -> SwiGLU (in-tile fusion, no barrier)
SwiGLU 在 FC1 GEMM 的 epilogue 里直接 fuse: `silu(gate) * up` 在 LDS 里完成, 写到 `fc1_out_block` 时已经是 `[F]` 而不是 `[2F]`。**不需要** monolith-moe 的 swiglu_precompute_phase, 因为 fc1_out 不进 IPC buffer (它本地, 全 LDS-resident)。

#### FC1 -> FC2 (in-WG, no IPC)
fc1_out_block 留在本 WG 的 LDS, 直接喂给 FC2 GEMM 的 A 矩阵; FC2 不再走 DTOLDS-from-HBM 路径, 而是 **DTOLDS-from-LDS** (gfx950 支持 `ds_read_b128_tr` 配合 MFMA fragment layout)。这会消掉一次 HBM round-trip。

#### FC2 -> combine (sender push -> receiver pull, 类似 dispatch)
FC2 结果写到本地 `fc2_intermediate[e][block_b]`, 然后 **同一个 WG** 在 epilogue 里调用 `__atomic_store_n(&fc2_ready[e][block_b], 1, __ATOMIC_RELEASE)`。push WG (32 个) 持续 spin `fc2_ready` 数组, 把 ready 的 block 写到 8 个 peer 的 combine_buf, 设置 `combine_ready`。

combine WG 看到自己负责的 token t 的 `combine_ready[t][8]` 全部 1 后, 走寄存器归约 (沿用 MonolithEP `combine.hip` 的 8x32 wave-broadcast 思路)。

### 3.4 wave-specialization (FC1/FC2 的 192 WG 内部)

每个 GEMM WG 是 4 wave (256 thread), 在 MI355X 上做 **2-2 split**:

| Wave 角色 | 数量 | 工作 |
|---|---|---|
| LOADER | 2 | 起 `buffer_load_dwordx4_lds` 异步 DMA, 维护 LDS double buffer; 也负责 spin block_ready |
| MFMA | 2 | 用 `v_mfma_f32_32x32x16_bf16` 计算; 不接触 HBM, 不 spin |

LOADER 提前 1 个 K-tile prefetch, MFMA wave 在 K-tile k 时算 k-1 的乘加; 通过 LDS 上的 ping-pong 缓冲交接, 用 1 个 wave-level `s_barrier` (而不是 workgroup) 同步。这是 cco-pipeline-overlap skill §3.4 的标准 pattern, MonolithEP 在 small tile 路径里已经验过 (单 wave 时 MFMA 利用率 99.3%, 4-wave 一起跑会被 LDS write port 限制到 ~85%)。

### 3.5 occupancy 预算

按 MI355X gfx950 (CDNA4) 的物理 budget:

| 资源 | 单 SIMD budget | 单 WG 占用 (4 wave) | wave/SIMD |
|---|---|---|---|
| VGPR | 512 | 256 V (mfma_tile.h 实测) | 2 |
| AGPR | 0 (CDNA4 unified) | -- | -- |
| LDS | 64 KB / WG (1 wave) 或 128 KB / 2 WG | ~50 KB (double buf 128x64 BF16 + 128x64 BF16 + epilogue scratch) | 1 wave/SIMD 上限 |
| Scratch | inf | 240 B/lane (可接受) | -- |

实际单 SIMD = 1 wave 是 LDS-bound 决定; 这是 MonolithEP 实测的最优 occupancy。**RocMoE-v2 不要试图压 LDS 到换 2 wave/SIMD**, MonolithEP 已经在 P0a 3-stage prefetch 上证伪过 (见 `../monolith-moe/2026-05-12_1635_3stage_prefetch_p0a_failed_not_hbm_latency_bound.md`)。

### 3.6 与 RocMoE-bak 的具体偏离

RocMoE-bak 已经把架构跑通到 P3 (combine_e2e), 但有几个地方设计偏保守, RocMoE-v2 要修:

| RocMoE-bak 现状 | RocMoE-v2 改动 | 理由 |
|---|---|---|
| 32-token block 写到 IPC token_buf 用 b128 + LDS 中转 | 全程 DTOLDS (`buffer_load_dwordx4_lds`) | MI355X DTOLDS 对 32-thread tile 是 free; LDS 中转浪费 ~30% 写带宽 |
| FC2 也输出 `[2F]` 让 combine 做后处理 | FC2 已经 SwiGLU-fused 输出 `[F]` (in-LDS), combine 单纯加权和 | 省 0.4 ms wall, FC2 HBM 写流量 -50% |
| Persistent state machine 用 32-bit state | 改 8-bit state + 24-bit ticket counter | 单 atomic counter 拿 work item, 把 state 转换从 atomicCAS 简化到 atomicAdd |
| 4 个 phase (route, dispatch, fc, push, combine) 共享一个 grid (state machine 切) | 5 phase 但 phase 2 (FC1+SwiGLU+FC2) 用 wave specialization 而不是 state 切 | FC1->FC2 数据全 LDS-resident, 不需要 phase 边界 |
| GEMM tile 自己写的 (P1 阶段精度) | 直接 cherry-pick MonolithEP `mfma_tile.h` 1113 行 | 不浪费 6 个月 |

### 3.7 与 MonolithEP 的具体偏离

MonolithEP 的硬伤是 g=0 spin 和 push contention, RocMoE-v2 用 RocMoE-bak 的架构修掉; 同时:

| MonolithEP 现状 | RocMoE-v2 改动 | 理由 |
|---|---|---|
| `compute_phase_barrier` * 4 / iter * 32 expert = 128 cross-WG barrier | 0 cross-WG barrier | scoreboard 取代 |
| 静态 hardcoded N_COMPUTE_WGS=192 | persistent state machine + work steal | 训练真实 T_e 不均时不再被最慢的 expert 拖死 |
| swiglu_precompute_phase 单独一个 device function (因为 FC2 走 IPC scratch) | SwiGLU 在 FC1 epilogue 内 fuse (因为 FC2 不再走 IPC) | 省 1 个 phase + 省 1 ms wall |
| ready_mask 用 atomicAdd_system | block_ready bitmap 用 atomic_store_release | 省 atomic ALU, 同步原语成本 -4x |

## 4. 物理性能预算

按 mi355_hardware_aware §7.2 roofline:

| 阶段 | 数据量 | 物理底 (8 MI355X, BF16) | 预算 |
|---|---|---|---|
| ROUTE_META | 256 token * 256 expert * 64 B = 4 MB | ~0.05 ms | 0.05 |
| DISPATCH_PULL | T*top_k*H*BF16 / 7 inbound = 2048*8*7168*2 / 7 / 800 GB/s = 0.04 ms 物理 | + 协议开销 -> 0.6 ms | 0.6 |
| FC1+SwiGLU+FC2 | 2 GEMM * (M*N*K) * 2 FLOP = 2 * 2048*32 * 2*2048*7168 = 1.5 TFLOP / iter | / (8 GPU * 1.3 PFLOPS BF16) = 0.14 ms 物理 | 1.5 ms (约 90% MFMA 利用) |
| FC2_PUSH | T*top_k*H*BF16 = 0.5 GB out / 7 outbound = 0.7 ms 物理 | + 协议 -> 0.9 ms | 0.9 |
| COMBINE_PULL | 7 * T*H*BF16 inbound + T*H out = 7 * 0.5GB / 800 GB/s = 4.4 ms 物理 | 但 reg-reduce 与 pull 完全 overlap -> 1.5 ms 实际 | 1.5 ms |
| 合计 (5 phase, 完美 overlap) | -- | max(p1+p2+p3, p4+p5) = max(2.0, 2.4) = 2.4 ms | -- |
| 合计 (顺序无 overlap) | -- | sum = 4.55 ms | -- |
| 实际目标 (含 stall + barrier 残余) | -- | -- | **3.5 - 4.0 ms BF16 / 8 GPU @ T_src=2048** |

把 MonolithEP super-mode 4.82 ms / monolith-moe 工况映射到 RocMoE-v2 等效工况, 砍 stall 后预期 ~3.7 ms; 训练真实 T_src=2048 工况 (MonolithEP 13.09 ms, PyTorch+RCCL 9.05 ms) 应能压到 ~7 ms, 反超 PyTorch+RCCL 1.3x。

FP8 / mxfp8 weights 路径: HBM weight 流量 -50%, FC1+FC2 段 1.5 ms -> 0.9 ms, 合计 ~ 2.7 ms BF16 / 8 GPU, 训练工况 ~5 ms。

## 5. 落地路线 (按 ROI 排序的 8 个 milestone)

| # | 名称 | 引入 | 验收 | 预期 ROI | 风险 |
|---|---|---|---|---|---|
| M0 | 仓库 bootstrap + 移植 mfma_tile.h | RocMoE-bak 骨架 + cherry-pick MonolithEP `mfma_tile.h`, `lds_layout.h` | standalone GEMM bench >= 950 TFLOPS @ DSV3 grouped | 0 (基础设施) | 低 |
| M1 | Layout-P + 64-bit block_ready bitmap | 重写 `dispatch.hip` (receiver-pull body) | dispatch 单段 wall <= 1 ms / 8 GPU @ T_src=2048; correctness 5 case | -1.34 ms vs MonolithEP push | 中 (协议正确性是难点) |
| M2 | persistent state machine + 角色 work-steal | 重写 `super_kernel.hip` 调度层 | super-mode 5 phase 各跑通; 0 hang | 0 (中长期投资) | 中 |
| M3 | FC1+SwiGLU+FC2 full DTOLDS, in-LDS fused | 直接复用 MonolithEP swiglu pre-compute idea, 但放 FC1 epilogue 内 | super-kernel BF16 wall <= 9 ms @ T_src=2048 | -3 ms vs MonolithEP | 低 (idea 已验过) |
| M4 | wave specialization (LOADER vs MFMA) for FC1/FC2 | 改 mfma_tile.h 模板, 加 `WAVE_ROLE` 编译参数 | MFMA util >= 95% (rocprof) | -1 ms vs M3 | 中 (LDS port contention) |
| M5 | atomic-free combine pull (reg-reduce) | 移植 MonolithEP `combine.hip` 寄存器归约 | combine 单段 wall <= 0.5 ms | -0.4 ms | 低 |
| M6 | mxfp8 weights for FC1 / FC2 | weight 离线量化 + MFMA fragment 改成 f8f6f4 | super-kernel wall <= 5.5 ms @ T_src=2048 | -2 ms | 中 (精度验证) |
| M7 | per-tile-class K_TILE template (default M=N=256 K=64) | mfma_tile.h 加 K_TILE 模板 | DSV3 grouped GEMM >= 1.05 TFLOPS / GPU | -0.5 ms (decomposed bwd 也用) | 低 |
| M8 | decomposed backward as super-kernel | 复用 super-kernel scheduler, 反向加 4 phase | bwd wall <= 1.2x fwd wall | training step -50% | 高 (新设计) |

每个 milestone 落地后写一篇 progress note 到 `slab/notes/rocmoe/` 并回写本目录 README。

## 6. 风险与缓解

| 风险 | 触发场景 | 缓解 |
|---|---|---|
| receiver-pull 协议在多 phase 共存时死锁 | dispatch 与 combine 同时拉 peer, peer 资源不足 | M1 验证: 加 token_buf / combine_buf 物理隔离; pull WG 数硬限 32 个; 加 watchdog timer |
| state machine 增加 ~10 V 寄存器, 让 mfma_tile.h 1 wave/SIMD 退化 | M2 + M0 一起编译 | 把 state machine 写在外层 wrapper kernel, mfma_tile.h 还是 standalone (经 inline ABI 调用); 实测 register usage 不退化 |
| Layout-P 在小 T_src (<512) 因为 padding 比例高反而慢 | inference 单 batch | 加 Generic / Large variant 编译开关 (沿用 monolith-moe 的 `MoeKernelVariant`); T_src < 1024 时 fallback 到 Layout-E |
| wave-specialization 让 LOADER 和 MFMA 抢 LDS port | M4 | 用 LDS bank-aware allocation: LOADER 写 bank 0-15, MFMA 读 bank 16-31, 物理隔离; ds_read_b128_tr 是单 port 操作 |
| FP8 量化精度损失训练发散 | M6 | 先验证推理 PASS, 再做 bf16 fwd + fp8 weight 的 mixed precision; per-block scale (mxfp8 OCP spec) |
| decomposed bwd 与 fwd 共享 LDS 设计撞 | M8 | bwd 单独的 super-kernel 二进制, fwd binary 不动 |

## 7. 与 monolith-moe 的关系

monolith-moe 项目继续维护现有 4.82 ms / iter 的版本, 用作:

1. **training 上线版本**: Primus + DSV3 e2e 已经跑通, 稳定可用; loss parity 已验。
2. **roofline 对照**: RocMoE-v2 在 BF16 / 同等工况下必须打过 monolith-moe, 否则架构改动失败。
3. **GEMM 资产源**: `mfma_tile.h` 直接源自 monolith-moe, 任何 GEMM 调优 (M7 K_TILE template 之类) 应当在 monolith-moe 先 PASS, 再 cherry-pick 到 RocMoE-v2。

## 8. 相关文件

- 参考实现:
  - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/MonolithEP/csrc/include/monolith/mfma_tile.h` (1113 行, 移植目标)
  - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/MonolithEP/csrc/{dispatch,gemm,combine}.hip` (push 路径参考)
  - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE-bak/docs/DESIGN.md` (Layout-P 原始设计)
  - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE-bak/csrc/super_kernel.hip` (state machine 骨架)
  - `/shared/amdgpu/home/xiaoming_peng_qle/workspace/RocMoE-bak/include/rocmoe/dispatch_body.h` (receiver-pull body)
- 知识来源:
  - `~/workspace/slab/.cursor/skills/cco-pipeline-overlap/SKILL.md`
  - `~/workspace/slab/.cursor/skills/mi355_hardware_aware/SKILL.md`
  - `~/workspace/slab/.cursor/skills/amd-gemm-optimization/SKILL.md`
  - `~/workspace/slab/knowledge/moe/dataflow.md`
  - `~/workspace/slab/knowledge/libraries/_patterns.md`
  - `~/workspace/slab/knowledge/libraries/composable-kernel.md`
- 历史相关 note:
  - [`../monolith-moe/README.md`](../monolith-moe/README.md) (前代项目 4.82 ms 进展线)
  - [`../monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md`](../monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md) (训练工况反输的根因)
  - [`../monolith-moe/2026-05-13_1245_swiglu_precompute_fc2_full_dtolds.md`](../monolith-moe/2026-05-13_1245_swiglu_precompute_fc2_full_dtolds.md) (DTOLDS 全路径前置实验)

## 9. 下一步 (immediate)

1. M0 仓库 bootstrap: 在 `~/workspace/RocMoE/` 下建 `csrc/include/`, cherry-pick MonolithEP 的 `mfma_tile.h`, `lds_layout.h`, `v1_common.h`; 起 standalone bench 跑通 GEMM。
2. M1 dispatch 重写: 直接搬 RocMoE-bak `dispatch_body.h` 的 receiver-pull, 加 64-bit `block_ready` bitmap, 5 case correctness。
3. 写 M0 progress note, 命名 `2026-05-21_HHMM_m0_mfma_tile_ported.md`, 回写本目录 README。


