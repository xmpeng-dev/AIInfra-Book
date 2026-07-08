# 参考设计研读 — DeepGEMM Mega MoE 的 dispatch → FC1 重叠机制 (SM100)

> **When**: 2026-06-25 10:25 UTC+8
> **Where**: slab 知识研读 (无 GPU 运行, 纯代码阅读)
> **Context**: rocmoe-v2 在 MI355X 上做的正是"dispatch→FC1 chunk-overlap 持久 super-kernel", DeepGEMM Mega MoE 是 NVIDIA SM100 上的同构参考实现, 研读它的握手协议为 rocmoe 的 `l1_arrival_count` chunk-overlap 设计提供对照

## TL;DR

DeepGEMM 的 `fp8_fp4_mega_moe` 把 EP dispatch → FC1 → SwiGLU → FC2 → EP combine 折叠进**单个 persistent kernel**, 用 **warp specialization**(dispatch / TMA-load / MMA / epilogue 四类 warp 同核常驻)+ **一组带 acquire/release 的环形计数器**(`l1_full_count` / `l1_empty_count`)以 **BLOCK_M 个 token 为粒度**在 dispatch 和 FC1 之间握手。NVLink 跨卡搬运全部交给 **TMA(DMA)引擎**(经 symmetric memory 重映射), 因此 NVLink 还在拉后面专家的 token 时, tensor core 已经在算前面已到齐的块——通信带宽与算力同时打满。这跟 rocmoe 的 receiver-pull + `l1_arrival_count` 设计是同一思路, 关键差异在 rocmoe 用 atomic counter + `s_sleep` polling, DeepGEMM 用 `ld_acq` 自旋 + `red_add_rel` 信号。

## Background

rocmoe-v2 的核心赌注之一就是 "pull-dispatch overlap FC1 MFMA"(见 [架构设计 note](./2026-05-21_1252_rocmoe_v2_architecture_design.md))。M2-G α 落地了 `fc1_gemm_role_body` 的 `l1_arrival_count` chunk-overlap polling([m2_g α BASELINE](./2026-05-22_1715_BASELINE_m2_g_fc1_body_integrated_dispatch_tradeoff.md)), 并在 [20:45 FLAT](./2026-05-22_2045_FLAT_super_kernel_disp_fc1_first_full_sweep.md) 实测到 chunk-overlap 真触发(T=512 H=7168 disp-only 1.566 ms / +FC1 仅 +0.448 ms)。DeepGEMM Mega MoE(2026.04.16 发布, PR #304/#316)是目前唯一公开的、把同一套思路做到生产级的 NVIDIA 实现, 值得逐行对照。

研读对象(`github.com/deepseek-ai/DeepGEMM`, main 分支):

| 文件 | 作用 |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | 1392 行主 kernel, 四类 warp 角色全在此 |
| `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` | `MegaMoEScheduler`: 按 wave 持久调度 (expert, m_block, n_block), `BlockPhase` 在 Linear1/Linear2 切换 |
| `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` | `Workspace`: 4 个环形计数器 + 各 buffer 偏移布局 |
| `deep_gemm/include/deep_gemm/comm/barrier.cuh` | `grid_sync` / `nvlink_barrier`(跨 rank 用 symmetric memory 信号) |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | block/wave/stage/`num_bytes_per_pull` 的启发式选择 |

## 主要发现 / 结论

### 1. 线程角色划分(warp specialization)

grid = `num_sms` 个 block(每 2 个 CTA 组 cluster 做 2-SM UMMA), block 内按 `warp_idx` 分角色, 各自 `reg_dealloc/alloc` 重分配寄存器:

| warp_idx | 角色 | 寄存器/线程 | 职责 |
|---|---|---|---|
| `0 .. kNumDispatchWarps-1`(4 warp) | **Dispatch** | 48 / 96 | 计数 + NVLink TMA 拉 token → 写 L1 ring |
| `kNumDispatchWarps`(4) | TMA-load A | 40 / 88 | L1 ring token(act)+SFA → smem |
| `+1`(5) | TMA-load B | 40 / 88 | 专家权重(FP4)+SFB → smem |
| `+2`(6) | **MMA issue** | 40 / 88 | leader CTA 发 `tcgen05` UMMA |
| `+3`(7) | 空转 | 40 / 88 | 仅让出寄存器给 epilogue |
| `>= kNumDispatchWarps+4`(4 或 8 warp) | **Epilogue / Combine** | 208 / 160 | TMEM→SwiGLU→FP8 量化→写 L2 ring; 末尾做 combine |

关键: 四类 warp **同核并发常驻**, 通信段和计算段之间**没有 kernel 边界、没有 device-wide barrier**。这是 overlap 的物理前提(对照 rocmoe 早期 standalone dispatch kernel → GEMM kernel 必须串行)。

### 1.5 block(CTA)无角色划分 + token/tile 归属判断

**block 本身没有角色划分**: grid = `num_sms` 个 block, 每个 block 跑**完全相同**的 kernel body, 角色专精只发生在 **warp 层**(每个 block 内部都同时有 dispatch / TMA-load / MMA / epilogue warp)。**没有** "这些 block 专做 dispatch、那些 block 专做 GEMM" 的 block 级划分——这是与 rocmoe 最核心的架构差异(rocmoe 是 `kNDispatchWGs` dispatch WG vs `kNGemmWGs` GEMM WG 的 block 级划分 + M4-α CU 物理隔离)。

两处"看起来像 block 角色"但其实不是动态角色的差异:

| 差异 | 性质 | 说明 |
|---|---|---|
| cluster leader / non-leader CTA(`is_leader_cta = block_rank_in_cluster()==0`) | 2-SM UMMA 硬件要求的固定两半分工 | leader 发 `tcgen05` UMMA + `arrive_and_expect_tx` + 分配 TMEM; non-leader 只 `arrive(0u)`, 持 A 的另一半(`LOAD_BLOCK_M = BLOCK_M/2` multicast, act 偏移 `+valid_m/2`) |
| `sm_idx` 条件化 bookkeeping | 元数据工作分摊, 非角色 | 仅 `sm_idx==0` 跨 rank 写 expert recv count + 参与 NVLink barrier 信号收发 + 清 `expert_send_count`; 其余 SM 按 `for(i=sm_idx-1; ; i+=kNumSMs-1)` 清 L1/L2 ring 计数器 |

**每个 block 处理哪些 token / tile**——全程是 **`blockIdx.x`(+ warp 索引)做静态 strided 划分**, 无原子 work-steal(对照 rocmoe M2-G FC1 的 `fc1_work_counter` atomic work-steal):

| 阶段 | 划分维度 | "token" 含义 | 判断公式 |
|---|---|---|---|
| dispatch 计数/写 src 索引 | 全局 warp round-robin | 本 rank 输入 token × topk(lane 铺 topk slot) | 起 `(sm·W+warp)·kNumTokensPerWarp`, 步 `kNumSMs·W·kNumTokensPerWarp`(`W=kNumDispatchWarps`, `kNumTokensPerWarp=32/kNumTopk`) |
| dispatch pull | 全局 warp round-robin | local expert 收到的 pool token(各 expert recv token 拼成线性序列) | 起 `sm·W+warp`, 步 `kNumGlobalWarps`; 线性 `token_idx` → (expert,rank,slot) 靠 `scheduler.get_num_tokens` 每 expert recv 计数 + round-robin min-peeling |
| FC1/FC2 | block-strided | GEMM tile (expert, m_block, n_block) | `block_idx = blockIdx.x`, 每次 `block_idx += kNumSMs`; 用每 expert `num_m_blocks = ceil(recv_tokens/BLOCK_M)` 把线性 idx 切回 (expert, m_block, n_block); cluster 2 CTA 落同 m_block、相邻 n_block |
| combine | 全局 warp round-robin | 本 rank 输出 token(1 warp / token) | 起 `sm·kNumEpilogueWarps+epi_warp`, 步 `kNumSMs·kNumEpilogueWarps` |

**关键对齐**: FC1 读的 ring 槽 `pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx`, 与 dispatch 的 `expert_pool_block_offset` 是同一个量 → 某 block 算的正是 dispatch 填进同一 ring 槽的那批 token。**所有归属判断的共同前提**是每 expert recv 计数提前 ready(`fetch_expert_recv_count` 跨 rank 自旋等 `kNumSMs*kNumRanks` 都写完), 负载均衡靠 wave 划分而非动态偷取——这是静态调度省掉原子竞争的代价。

### 2. dispatch ↔ FC1 的桥梁: L1 ring buffer + 4 个环形计数器

`Workspace`(`layout/mega_moe.cuh`)按 `ring_block_idx` 维护两对计数器:

| 计数器 | 写者 | 读者 | 语义 |
|---|---|---|---|
| `l1_full_count[rb]` | dispatch (`red_add_rel`) | FC1 act-load warp (`ld_acq` 自旋) | 该 ring block 已到多少 token |
| `l1_empty_count[rb]` | FC1 epilogue (`red_add`) | dispatch (`ld_acq` 自旋) | 该 ring block 已被消费、可复用 |
| `l2_full_count[rb]` | FC1 epilogue (`red_add_rel`) | FC2 act-load warp | FC1 结果(=FC2 输入)就绪块数 |
| `l2_empty_count[rb]` | FC2 epilogue (`red_add`) | FC1 epilogue 自旋 | L2 ring 槽位可复用 |

token 数据放在 `l1_token_buffer`(`kNumRingTokens` 槽位的**环形**缓冲), dispatch 把远端 token 拉进 ring, FC1 从同一 ring 读出来算——两者以 **BLOCK_M 个 token 为粒度**握手, 并通过 `l1_empty_count` 形成双向背压。

> 注意: dispatch pull 与 FC1 load 的交接走的是**全局 HBM 的 L1 ring**, 不是 SM 内 smem。dispatch 拉取是 "远端 global → smem 暂存(`dispatch_send_buffer`, 临时跳板)→ 本地全局 ring"; FC1 load-A 再从全局 ring TMA 进自己的 `smem_a`。生产者 SM 与消费者 SM **一般不同**(dispatch 按全局 warp round-robin 拉, FC1 按 `blockIdx.x` strided 算, 两套映射对不齐), 所以握手必须放全局内存 + 全局原子计数器, 不能靠 smem / `__syncthreads`(后者只在单 WG 内有效, 够不着跨 SM)。

### 2.5 load-A / load-B / MMA 如何对齐到同一 tile + 启动计算

三个 GEMM warp(load-A=warp4, load-B=warp5, MMA=warp6)**没有共享 loop 计数器**, 而是各自独立跑同一个确定性 `scheduler.for_each_block`, 用一组 per-stage mbarrier 在同一块 smem 上对齐。绑定它们的是两对 barrier(warp 2 初始化):

| barrier | init 期望 | 写者 | 读者 | 语义 |
|---|---|---|---|---|
| `full_barriers[stage]` | `2*2`=4 arrive + transaction 字节 | load-A + load-B(各 2 CTA)| MMA | 该 stage 的 A/B tile 都到齐 |
| `empty_barriers[stage]` | `1` arrive | MMA(`umma_arrive`)| load-A + load-B | 该 stage smem 槽已被消费、可重填 |
| `tmem_full_barriers[accum]` | `1` | MMA | epilogue | 累加器就绪 |
| `tmem_empty_barriers[accum]` | `2*kNumEpilogueThreads` | epilogue | MMA | 累加器槽可复用 |

**一个 tile(一个 K-block stage)的握手**:

```
load-A (warp4)              load-B (warp5)              MMA (warp6, 仅 leader)
empty[s].wait(ph^1) ◄─┐     empty[s].wait(ph^1) ◄─┐    (上轮) umma_arrive(empty[s]) ─┘ 释放槽
  │ TMA A+SFA→smem_a[s]      │ TMA B+SFB→smem_b[s]
  ▼                          ▼
full[s].arrive_and_         full[s].arrive_and_
  expect_tx(bytesA)           expect_tx(bytesB)
  (+非leader arrive(0u))      (+非leader arrive(0u))
       └──────────┬───────────────┘
                  ▼ (4 arrive 齐 + TMA 字节全落地 → flip phase)
                          full[s].wait(ph) ──► 解锁, 发 UMMA: smem_a[s]×smem_b[s]→TMEM
                                 │
                                 ▼ umma_arrive(empty[s])  ← UMMA 完成自动释放槽
                                   if 末 k_block: umma_arrive(tmem_full[accum]) → epilogue
```

要点:
- **MMA 不靠谁"通知开始"**, 它卡在 `full_barriers[stage_idx].wait(phase)`; 该 barrier 只有当 load-A 和 load-B **都 arrive(共 4 次)且两边 TMA 的 transaction 字节(`bytesA+bytesB`)全部落地**才翻 phase——**full_barrier 翻转本身就是发令枪**。`arrive_and_expect_tx` 把 "arrive 计数" 与 "字节到齐" 合一, 保证 MMA 读 smem 时数据完整。
- **UMMA 自己发 release**: `cutlass::arch::umma_arrive_multicast_2x1SM` 让 `tcgen05` 指令完成后**异步** arrive `empty_barriers[s]`, 把槽还给生产者(`tcgen05.commit` 隐式 fence)。
- **三者怎么"知道在同一 tile"**: scheduler 确定性(只依赖 `blockIdx.x` + recv 计数)→ 三个 warp 产出的 (expert,m,n) 块序列逐项相同; 各自用同一 `advance_pipeline` 推进 `stage_idx`/`phase`(绕回第 0 stage 时翻 phase)→ 步调一致; `full/empty_barriers[stage_idx]` 把它们钉在同一块物理 `smem_a/b[stage_idx]`。**确定性迭代 + per-stage mbarrier = 无需中央协调即对齐**。
- **接力 epilogue**: MMA 在块末 `umma_arrive(tmem_full[accum])` → epilogue `wait` 解锁读 TMEM 做 SwiGLU/量化 → `tmem_empty[accum].arrive` 还槽; `kNumEpilogueStages=2` 让 MMA 算下一块时 epilogue 还能处理上一块。

### 2.6 load-A / load-B / MMA 在整个 K 循环里如何并发跑(稳态流水)

§2.5 是"一个 tile 的握手", 这里是"整段怎么重叠跑"——核心是 `kNumStages` 级 smem 多缓冲让 **load 跑在 MMA 前面**:

- **load warp 发 TMA 后不等数据落地**(落地由 mbarrier transaction count 追踪), 立刻 `advance_pipeline` 去发下一 stage 的 TMA → **一个 load warp 能让多个 stage 的 TMA 同时在途**。它只在 `empty_barriers[s].wait` 被挡——追上 MMA、要复用 MMA 没消费完的槽时。
- **MMA warp** 在 `full_barriers[s].wait` 等当前 stage A/B 齐, 算完 `umma_arrive(empty[s])` 还槽再 `advance_pipeline`。
- 两者通过 `kNumStages` 个槽 + full/empty 解耦 → **load 领先 MMA 最多 `kNumStages-1` 个 K-block**。

稳态时间线(kNumStages=4 为例):

```
槽位 ring: s0 s1 s2 s3 → s0 s1 ...   (advance_pipeline 轮转, 绕回 s0 翻 phase)

k_block:     0    1    2    3    4    5   ...
load-A TMA: [A0][A1][A2][A3][A4][A5] ...        ← 填 smem_a[s], s=k%4
load-B TMA: [B0][B1][B2][B3][B4][B5] ...        ← 填 smem_b[s]
              └ full[0]齐 ┐
MMA:          (等full[0]) [M0][M1][M2][M3][M4] ...
                            └umma→empty[0]↑ → load 才能把 s0 复用成 A4/B4

  任一时刻: 前几 stage 的 TMA 在途(吃 HBM/XGMI 带宽) ‖ MMA 啃更早 stage(吃 tensor core)
```

- **启动**: `empty_barriers` 初值让前 `kNumStages` 槽一上来即"空闲", loaders 连发 `kNumStages` 个 TMA 灌满流水, MMA 等 `full[0]` 齐就开算。
- **稳态**: load 始终领先 MMA 约 `kNumStages` 个 block; loaders 打满取数带宽、MMA 打满 tensor core, **时间重叠**; 只要 load 带宽 ≥ MMA 消费速率, MMA 不饿 → GEMM compute-bound。
- **节流**: loaders 唯一停点是 `empty[s].wait`(领先太多要覆盖未消费的槽)。`kNumStages` 深度 = 能掩盖多少 HBM/XGMI 取数延迟的缓冲深度(越深越抗抖动, 但吃越多 smem)。
- **load-A 与 load-B 彼此不直接同步**: 各自独立 `empty.wait→发 TMA→full.arrive`, A/B tile 可任意先后落地; MMA 的 `full[s].wait` 等**较慢的那个**也齐(4 arrive + `bytesA+bytesB` 字节全到)才解锁——A、B 是两条并行供给流, 在每个 stage 的 full_barrier 处汇合。

### 3. 为什么能高效 overlap(核心五点)

1. **同核 warp specialization, 无段间全局同步**——四类 warp 并发常驻每个 SM, 没有 kernel relaunch / device sync 把通信和计算隔开。
2. **跨卡传输交给 TMA/NVLink, 不占算力**——dispatch 拉取是 `tma_load_1d`(远端 symmetric memory)+ `tma_store_1d`(本地 ring), `sym_buffer.map(local_ptr, peer_rank)` 把本地指针重映射到对端 GPU 地址走 NVLink; DMA 引擎搬运时 tensor core 仍在算已到齐的块。
3. **BLOCK_M 粒度细粒度握手**——FC1 act-load warp 只等"自己这一块"(`while (ld_acq(l1_full_count_ptr(rb)) != num_expected_tokens)`), 不等整个 dispatch 完成; dispatch 用 `red_add_rel` 逐块发信号, acquire/release 保证可见性, 形成软件流水线。
4. **有界 ring + 双向背压**——FC1 消费完 L1 槽后 `l1_empty_count++` 释放给 dispatch 复用; heuristics(`get_num_experts_per_wave_for_mega_moe`)按 ring 容量 + 填满 SM 所需块数选 `kNumExpertsPerWave`, 把流水线深度调到刚好掩盖 NVLink 延迟。
5. **链式重叠延伸到 FC2/combine**——FC1 epilogue 直接把结果以 FP8 写进 L2 ring 并 `l2_full_count++`, 所以 dispatch→FC1→FC2→combine 是一条连续流水线; 同一时刻不同 SM 分别在拉 token / 算 FC1 / 算 FC2 / 写回 combine。

## 详细分析

### dispatch 流程(warp 0..3)

1. **本地计数**: `read_topk_idx` 遍历 topk_idx, `atomicAdd_block` 到 smem `expert_token_count[expert]`。
2. **全局抢槽位**: 对每个 expert 用一次 `atomic_add` 累加到 `expert_send_count`(高 32 位计 rank 数, 低 32 位计 token 数), 拿回全局偏移。
3. **写"发给你哪些 token"到对端**: 算出目标 (rank, 本地 expert, slot), `*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx`(NVLink 写)。
4. **grid sync + nvlink_barrier**: 各 rank 对齐 `expert_recv_count`。
5. **拉数据 (pull loop, overlap 主体)**: 每个全局 warp 领 token →(min-peeling round-robin 算源 rank/slot)→ 算 `pool_block_idx → ring_block_idx` →(背压: `while (ld_acq(empty_ptr) < target)`)→ `tma_load_1d`(远端→smem 暂存)+ `tma_store_1d`(smem→本地 ring)+ 搬 SF/weight/写 `TokenSrcMetadata` → `red_add_rel(l1_full_count_ptr(rb), ...)` 发信号。

### FC1 流程(warp 4/5/6 + epilogue)

由 `MegaMoEScheduler::for_each_block` 驱动, 按 wave 持久调度 (expert, m_block, n_block):

1. **act-load warp(4)** 对每个 Linear1 block 先自旋等这一块到齐(`l1_full_count`), 再 TMA copy token+SFA → `smem_a`, 经 `full_barriers[stage]` 通知 MMA。
2. **weight-load warp(5)** 同时把 FP4 权重 + SFB 搬进 `smem_b`。
3. **MMA warp(6, 仅 leader CTA)** 发 `SM100_MMA_MXF8F6F4_2x1SM_SS`(2-SM block-scaled UMMA, swap A/B, K-major), SFA/SFB 经 UTCCP 进 TMEM, 结果累加在 TMEM。
4. **epilogue warpgroup** 从 TMEM 读结果 → 就地 SwiGLU(`silu(gate)*up*weight`)→ amax → 量化 FP8 E4M3 → STSM 进 smem → TMA store 进 `l2_acts`(L2 ring), 同时写 UE8M0 SF; 最后 `red_add_rel(l2_full_count)` + `red_add(l1_empty_count)` 释放 L1 槽。

### 流水线时序图(稳态, 同一 SM 内 warp 角色随时间)

```
时间 →  t0      t1      t2      t3      t4      t5      t6      t7
        |-------|-------|-------|-------|-------|-------|-------|-------|

DISP    [count ][nvlink ][pull blk0][pull blk1][pull blk2][pull blk3][pull blk4][clean ]
warp0-3   元数据  barrier  TMA载入    TMA载入    TMA载入    TMA载入    TMA载入   (与combine
                          ↓l1_full↑0  ↓l1_full↑1 ↓l1_full↑2 ↓l1_full↑3  ↓l1_full↑4  错峰)
                              │           │          │          │          │
                              │ (等blk0满) │          │          │          │
                              ▼           ▼          ▼          ▼          ▼
LOAD-A          (idle/spin)  [ld blk0 ][ld blk1 ][ld blk2 ][ld blk3 ][ld blk4 ]
warp4                         act+SFA    act+SFA    act+SFA    act+SFA    act+SFA
                                 │          │          │          │
                                 ▼          ▼          ▼          ▼
MMA                           [mma b0 ][mma b1 ][mma b2 ][mma b3 ][mma b4 ]   ← tensor core
warp6                          UMMA→TMEM  UMMA      UMMA      UMMA      UMMA       与上方 DISP
                                 │          │          │          │              的 TMA 同时跑
                                 ▼          ▼          ▼          ▼
EPILOG                        [swiglu b0][swiglu b1][swiglu b2][swiglu b3] ...  → 写 L2 ring
warp8+                         FP8量化     ↑l1_empty↑0(释放槽给DISP复用)             → l2_full↑
                                          └──────────背压回环──────────┘            → FC2 起步

         └── NVLink 带宽 (DISP 的 TMA) 与 tensor core (MMA) 在 t2..t6 完全重叠 ──┘
```

要点: dispatch 在 t2 拉完 blk0 并 `l1_full↑` 后, FC1 的 LOAD-A 立刻在同一 t2..t3 区间起步算 blk0, **不等 dispatch 把 blk1..blk4 拉完**。于是 DISP 行(NVLink/TMA)和 MMA 行(tensor core)在 t2 之后逐块错位重叠; `l1_empty` 回环让 ring 槽位有界复用。

### 与 rocmoe 当前实现的对照

| 维度 | DeepGEMM Mega MoE (SM100) | rocmoe-v2 (MI355X / CDNA4) |
|---|---|---|
| 握手计数器 | `l1_full_count` / `l1_empty_count` (per ring block) | `l1_arrival_count[pool_block]` (M2-G α) |
| 信号语义 | `red_add_rel` + `ld_acq` 自旋 | atomicAdd + `s_sleep(2)` backoff polling |
| 跨卡搬运 | TMA `load_1d`/`store_1d` + symmetric memory map | `cooperative_b128_copy` LDS-staged + peer XGMI read (pull) |
| 握手粒度 | BLOCK_M token 块 | pool_block (同 BLOCK_M 概念) |
| 角色隔离 | warp specialization (软件, 同 SM) | M4-α `__launch_bounds__(_,1)` CU 物理隔离 / M4 wave specialization (规划中) |
| dispatch 方向 | receiver-pull (TMA load remote) | receiver-pull (M1: pull 比 push 慢 ~43%/tok, 见 [compare note](./2026-05-21_1740_compare_monolithep_dispatch.md)) |
| 量化 | FP8 act × FP4 weight (MXFP8FP4) | BF16 (M2-G), mxfp8 weights 规划在 M6 |

**可借鉴点**:
1. DeepGEMM 把跨卡拉取完全压在 **TMA 异步引擎**上, SM 的 ALU/tensor core 完全不参与 byte 搬运; rocmoe 的 `cooperative_b128_copy` 是 thread 显式搬运, 仍占 wave scheduler——这正是 rocmoe M2-G α dispatch +53% 退化的来源, DeepGEMM 的 TMA 路径天然规避了这个竞争。CDNA4 的 async direct-to-LDS(`buffer_load ... lds`)是 rocmoe 对应的可用原语。
2. DeepGEMM 的 ring 槽位 + `l1_empty` 双向背压, 把显存占用限制在 `kNumRingTokens`; rocmoe 的 `expert_token_pool` 目前是全量 pool, 可考虑引入有界 ring 降低 L2/HBM footprint(对照 [M1c-A L2 cliff](./2026-05-21_2030_DOWN_m1c_a_sender_pack_l2_pessimization.md))。
3. wave 调度 `get_num_experts_per_wave_for_mega_moe` 用 "ring 容量 vs 填满 SM 所需块数 + imbalance factor 2" 选波宽, 跟 rocmoe 的 kSubWGs sizing 模型([M1c-E](./2026-05-22_1700_FLAT_m1c_e_ksubwgs_knob_kept_default_8_post_overlap_remodel.md))是同类决策, 可对照其公式。

## ROCm / MI355X 移植设计建议

**核心命题**: DeepGEMM 的高效 overlap 有一半是 TMA 白送的——TMA 是独立异步 DMA 引擎, 跨卡搬运完全不占 SM 的 VALU/tensor core, 且用 mbarrier 自动做 transaction 计数。CDNA4 **没有 remote-TMA**, 所以 ROCm 设计的核心命题是: **在"搬运必须由线程发起"的前提下, 把"通信不抢算力"这个性质找回来**。

### 原语映射 DeepGEMM → CDNA4

| DeepGEMM (SM100) | CDNA4 (gfx950) 等价 | 关键差异 |
|---|---|---|
| TMA `load_1d`/`store_1d`(远端) | `global_load_lds_b128` / `buffer_load_lds`(async direct-to-LDS) | **不占 VGPR**, 最接近 TMA; 但要线程发起 + `s_waitcnt vmcnt` 收尾 |
| mbarrier `arrive_and_expect_tx` | HBM atomic counter + `s_waitcnt` | 无 transaction-count barrier |
| symmetric memory `sym_buffer.map` | `hipIpcOpenMemHandle` + peer 指针 | 等价, XGMI all-to-all |
| TMEM 累加 + 2 epilogue stage | AGPR 累加(512×32b/lane 与 VGPR 共享) | **无 TMEM**, 累加器吃 AGPR, 流水级数受限 |
| 2-CTA cluster UMMA | 无 cluster | **不要模拟**, CDNA 无分布式 shared memory |
| `v_mfma_scale_*_f8f6f4`(MXFP8FP4) | `__builtin_amdgcn_mfma_scale_f32_16x16x32_f8f6f4` | **原生支持**, FP8×FP4 不用软件 dequant |
| warp specialization(同 SM) | wave specialization(同 CU 分 SIMD) | rocmoe M4 已规划方向 |

### 排序后的建议

| 优先级 | 建议 | 为什么 | 对应 milestone |
|---|---|---|---|
| **P0** | 把 `cooperative_b128_copy` 换成 `global_load_lds_b128`(HBM/XGMI → LDS 不过 VGPR) | M2-G α dispatch +53% 退化根因是 thread 显式 `LOAD→VGPR→LDS→STORE` 占满 VGPR + VALU 发射槽抢 MFMA; async-to-LDS 数据在途**不占 VGPR**, 拉取 wave 大部分时间 stall 在 XGMI 延迟上几乎不耗 VALU → 与 MFMA wave 廉价共驻, 复现 DeepGEMM "每 CU 完整流水线"。**需先验证 peer-mapped 指针对 `global_load_lds` 可用** | M2-G ε / M4 |
| **P0** | wave specialization(M4)取代 CU 物理隔离(M4-α) | M4-α `__launch_bounds__(_,1)` 是钝刀, dispatch CU 等 XGMI 延迟时不贡献 MFMA 浪费 1/4 CU; 应**在 CU 内分 SIMD**(如 SIMD0 跑 pull-loader, SIMD1-3 跑 MFMA), loader wave 不占发射槽 → 同打满 XGMI 带宽 + MFMA。配 `global_load_lds` 后 co-residency 无 M2-G α 竞争惩罚 | M4 |
| **P1** | 给 `l1_arrival_count` 加 `l1_empty` 回环做有界 ring | rocmoe 现在全量 `expert_token_pool`(M1c-A 在 T=2048 撞穿 L2 32MB); DeepGEMM `l1_full`/`l1_empty` 把 buffer 限制在 `kNumRingTokens`, 消费完即释放槽 → input footprint 重回 L2/Infinity Cache 驻留。用现成 `grid_sync_v2` release/acquire 基建 | M1c 续 / M3 |
| **P1** | 深化 async 队列掩盖 XGMI pull RTT | M1 实测 pull 比 push 慢 ~43%/tok(XGMI RTT); DeepGEMM 用 TMA 深队列藏 RTT。AMD 上**保持多条 `global_load_lds` 同时在途**(非 M1c-D 单 16KB staged tile 串行), 深度按 XGMI BDP 拍; 先 rocprof 确认是 latency-bound 还是 bandwidth-bound | M2-G γ |
| **P2** | SwiGLU 融进 FC1 的 MFMA epilogue(读 AGPR), 取消独立 `fc1_swiglu_pass` + intra-role grid_sync | 无 TMEM, MFMA 结果落 AGPR; `STORE_BLOCK_M` 保持小控 AGPR 压力留 ≥2 wave/SIMD; epilogue 内 SwiGLU + FP8 量化一次写进 L2 ring(= DeepGEMM 做法) | M3 |
| **P2** | XCD locality(DeepGEMM 不需要、AMD 必须管) | 8 XCD 各自 ~4MB L2 + front-end, 只 256MB Infinity Cache 跨 XCD 共享; 让某 expert 的 ring buffer 固定在计算它的 XCD 近端 HBM, swizzle 使一个 expert 的 m_block 落同一 XCD → FC1 epilogue 写 L2 ring + 下游读命中同一 per-XCD L2 | M3+ |

### 不要照搬

| 别抄 | 原因 | AMD 该用什么 |
|---|---|---|
| 2-CTA cluster / distributed smem | CDNA 无 cluster | 单 WG 算一个 tile |
| `arrive_and_expect_tx` transaction barrier | 无对应硬件 | `s_waitcnt vmcnt` + HBM atomic counter |
| TMA descriptor prefetch / tensor map | AMD 用 V# buffer descriptor, 无全局描述符 | 现场构造 buffer descriptor |
| 纯静态 strided 调度的"无脑均衡" | MoE 负载不均 | 静态 strided(省原子竞争)+ 仅 tail 用 work-steal; wave-sizing 抄 `get_num_experts_per_wave`(含 `kImbalanceFactor=2` 过订阅) |

**一句话**: ROCm 上重建这套 overlap, **关键不是抄 ring buffer 协议(rocmoe 已有), 而是把"搬运"从占算力的 thread copy 换成不占 VGPR 的 `global_load_lds` async 引擎 + CU 内 wave specialization**——这两步合起来才是 CDNA4 上 TMA 的功能等价替身, 也是把 M2-G α 那个 +53% 退化真正消掉的正解(而非 M4-α 钝刀隔离)。

## 下一步 / 建议

- rocmoe M2-G ε(dispatch -53% 救援)评估时, 参考 DeepGEMM 把 byte 搬运交给 async-copy 引擎而非 thread 显式 copy 的思路——CDNA4 `global_load_lds` / async direct-to-LDS 是对应原语(见 `.cursor/skills/mi355_hardware_aware/SKILL.md`)。
- 考虑给 rocmoe 的 `l1_arrival_count` 加 `l1_empty` 回环计数器, 实现有界 ring 复用而非全量 pool。
- 本研读为代码阅读, 无 perf 实测; 若要量化 DeepGEMM 在 MI355X 等价工况的差距, 需 H100/H800 上跑 `tests/test_mega_moe.py` 取基线。

## 相关文件

- 源码(已 clone 到 `/tmp/DeepGEMM`, 非持久): `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` 等
- 上游: `https://github.com/deepseek-ai/DeepGEMM` (PR #304 Mega MoE, #316 benchmarks)
- rocmoe 架构设计: [`2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md)
- rocmoe FC1 chunk-overlap 落地: [`2026-05-22_1715_BASELINE_m2_g_fc1_body_integrated_dispatch_tradeoff.md`](./2026-05-22_1715_BASELINE_m2_g_fc1_body_integrated_dispatch_tradeoff.md)
