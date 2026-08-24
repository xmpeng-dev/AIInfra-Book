# 立项：一个 HK 形状的 repo —— AMD 上融合 MoE 训练 kernel 的 tile 原语库

> **When**: 2026-08-12 15:30 UTC+8（**15:52 大幅修订，见 §0**）
> **Where**: 登录机，纯设计文档（未跑 GPU、未写代码）
> **Context**: 想自己维护一个类似 HipKittens 的 repo，重点是 mega kernel。本篇把它变成可执行的立项
> **命名**: `peer-tiles` 是占位名，见 §8 待决问题

---

## 0. 修订（15:52）：论点收紧为「stage 级成对融合」，且 gen-3 可能已经赢了

### 0.1 一个未被记录的结果

口径已核对可比：MegaMoE 的 `--num-tokens` 传给 `num_max_tokens_per_rank`（`bench_mega_moe_fp8.py:221`），与 monolith-moe 的 `T_src` 同为 per-rank；配置同为 DSV3 / EP8 / H=7168 / I=2048 / E=256 / topk=8。

| T=8192 per-rank，同配置 | forward | vs PyTorch+RCCL |
|---|---|---|
| PyTorch+RCCL bf16（2026-05-13） | 18.64 ms | 1.00× |
| **gen-1** monolith-moe super-kernel bf16（2026-05-13） | **64.47 ms** | **0.29×** |
| **gen-3** MegaMoeFlydsl bf16（19.97 fwd+bwd − 12.169 bwd） | **~7.80 ms** | **2.4×** |
| **gen-3** MegaMoeFlydsl mxfp8（14.117 − 8.734） | **~5.38 ms** | **3.5×** |

**同工况同口径，gen-1 → gen-3 是 64.47 → 7.80 ms（8×）。** 而 `career-strategy/2026-08-06` 至今写着"三代 super-kernel 没长成东西，512 t/g 赢、2048 t/g 就输"，引的证据只有 gen-1 那篇。**该结论对 gen-1 成立，对 gen-3 很可能已失效，只是这两个数从未被并排看过。**

⚠ **四个口径风险，必须先验证**：不同机器（mi355-gpu-26 vs n01-21 / n05-29）；不同日期（05 月 vs 07 月，sclk ±30%）；PyTorch baseline 是 5 月的（RCCL 可能已改进）；forward 是减法得来而非直接测。**→ 见 M0。**

### 0.2 范式的答案是经验性的：gen-1 与 gen-3 的结构差

**gen-1 是真 mega kernel**：pack-scatter / permute / FC1 / SwiGLU / FC2 / un-permute / combine 全折进一次 launch —— 4 个 role、5 个 sub-phase、chunk 流水深度要调、grid-wide barrier。

**gen-3 不是一个大 kernel，是若干「一次通信 × 一个 GEMM」的融合 stage**：反向四个 —— `dispatch_fc2_dgrad` / `fc2_wgrad` / `fc1_wgrad` / `fc1_dgrad_combine`，每个只融两条腿。

用一个算式解释为什么两条腿赢。融合收益 ≤ 被藏起的通信时间（**有界**）；融合代价 = FLOPs × (1/η_fused − 1/η_library)（**随 T 线性增长**）：

| | η（有效算力） | 代价 | 收益上界 | 结果 |
|---|---|---|---|---|
| gen-1 @ T=8192 | 5.77e12 FLOP / 64.47 ms = **90 TFLOPS/GPU**；库路径 GEMM 单看 **521 TFLOPS** | 5.77e12 ×(1/90−1/521) ≈ **53 ms** | dispatch 3.78 + combine 3.61 = **7.39 ms** | 输 7 倍，且 T 越大越输 |
| gen-3 | wgrad GEMM **2007–2199 TFLOPS**（自测上限的 85%）；`fc1_dgrad_combine` GEMM 2.200 ‖ PUSH 2.029 **两腿等长**，overlap 效率 87% | **≈ 0** | 同量级 | 收益全部落袋 |

### 0.3 修订后的论点

> **AMD 上的 mega 范式是「stage 级成对融合」：一次通信配一个保持库级 tile 几何的 GEMM，两条腿等长。整层单核融合会把 GEMM 的算术强度切碎，而该损失随 token 数线性增长，必然在生产规模上翻盘。**

这条命题同时解释：gen-1 为何输、gen-3 为何赢、crossover 为何在 512→2048 之间（小 T 时库 GEMM 自身效率也低，切碎不吃亏）。它与 HK §7.1 的寄存器经济学同源——**role 越少，寄存器与 CU 的竞争越小，每条腿才能跑到库级效率**（gen-1 四 role，gen-3 两腿）。

### 0.4 HK 的角色随之改变

**不是"HK 有高性能 GEMM 所以适合 mega"**——GEMM 已不是瓶颈：wgrad 2199 TFLOPS = 自测上限 85%，`mfma_tile.h` 规则形状 1290 TFLOPS。更快的 GEMM 换不到东西。

HK 对症的是**让 GEMM 与别的 role 共驻而不掉效率**，正是范式命题的核心约束：

| HK 能力 | 对应的 mega 约束 | 证据 |
|---|---|---|
| **register pinning** | mega 里寄存器被 GEMM 累加器 / comm staging / flag-epoch 三方分 | `WG_PER_CU=1` 拿 −35%（T=2048 18.911→12.240）；dW1 wgrad 卡在 VGPR 248/512 = 2 waves/SIMD（3 waves 需 ≤170）。**同一约束三次现身** |
| 8-wave ping-pong / 4-wave interleave | role 轮换取代固定分区，是"两腿等长"的实现手段 | rocmoe M4-α（CU 隔离）vs M4-β（CU 内分 SIMD）悬而未决 |
| Table 5 LDS phase/bank | mega stage 里 GEMM tile 与 comm staging 抢同一块 LDS | 历史上 PAD/XOR swizzle 全靠试 |
| 单 kernel 混用多 MFMA 形状 + 同 tile 行/列双读 | mega stage 天然含多个形状不同的 GEMM（FC1 K=7168 / FC2 / wgrad variable-K） | HK attention backward 已示范 |

**一句话：HK 提供"让 GEMM 与别的 role 共存而不掉效率"的纪律，不是一个更快的 GEMM。**

### 0.5 §1 的"第四条原语"降级

「所有权与可见性」仍然是**实现这个范式所需的工具**（stage 之间、腿与腿之间的 publish/acquire 就是它），但**不再是论文的主论点**。主论点是 §0.3 的范式命题；第四条原语是让这个范式可以被简洁地写出来的手段。§3 的 API 草案不变。

### 0.6 M0 被替换

原 M0（论点定稿，零代码）替换为：**同日同机同 routing 的 gen-1 / gen-3 / PyTorch+RCCL 三方 A/B，T ∈ {512, 2048, 8192}，直接测 forward（不用减法）。** 两个 bench 都现成，成本低，且它决定项目叙事是"探索融合能不能行"还是"已做出第一个在生产规模打败库路径的 AMD mega kernel"。后者强得多。

---

## TL;DR（原文，论点部分已被 §0 收紧）

**论点（一句话）**：tile 抽象缺了第四条原语——**所有权与可见性**（谁写、何时对谁可见、代价多少）；把它补上之后，跨 GPU 的融合 kernel 才能像写单卡 GEMM 一样写，而这条原语在 AMD 上比在 NVIDIA 上更自然。

**为什么这是 HK 形状**：HK 检验并重做了 ThunderKittens 的三条原语（tiles / overlapping / grid scheduling），kernel 是论证的证据而非产品。本项目加第四条，同样用 kernel 当证据。

**范围**：MoE 层 fwd+bwd 的融合 super-kernel，AMD CDNA3/4。**不做**通用 GEMM、不做 attention、不做新 DSL、不做推理。

**第一个可交付（M1）**：把三代 super-kernel 里手搓的原语抽成 HIP C++ header，用库重写出**同样**的数字——证明抽象税为零。这一步不需要任何新性能。

**最诚实的一条**：论点必须把 regime 边界写进去。gen-1 在 512 t/g 赢 1.46×、2048 t/g 0.53×、8192 t/g 0.29×。这张表是本项目最值钱的资产之一，因为没人发表它——**而 §0.1 显示 gen-3 可能已经把它翻过来了，那会是更强的资产。**

---

## 1. 论点

### 1.1 HK 做了什么（对照物）

HK 把 TK 这类 DSL 的贡献归纳成三条，逐条在 AMD 上检验（`papers/hipkittens.md` §2）：

| # | 原语 | HK 的结论 |
|---|---|---|
| 1 | **Tiles** — 带优化访存模式的数据类型 + PyTorch 风格批量算子 | **通用**，原样搬过来就能用 |
| 2 | **Overlapping** — 把 worker 调度到不同硬件单元 | **必须重做**（wave specialization 在 AMD 上是负优化 → 8-wave ping-pong / 4-wave interleave） |
| 3 | **Grid scheduling** — 按顺序分配 block 最大化 cache 复用 | **必须重做**（chiplet 感知，+19%） |

这三条有一个**共同的隐含假设**：一个 tile 有单一所有者，它所在的内存层级是"单 GPU 内私有或共享"，同步是 `__syncthreads` 形状的。

### 1.2 第四条原语

> **Ownership & Visibility** — 一个 tile 可以由一个 agent 生产、另一个 agent 消费，两者可能在不同设备上。抽象必须携带：(a) 它住在哪一层**（包括 peer HBM）**、(b) 现在谁拥有它、(c) 什么 publish/acquire 对使它对谁可见、(d) 这个可见性花多少钱。

今天这四件事**全部是每个 kernel 手搓的**：DeepEP 手搓 `atomicAdd_system` + `ld_volatile_global`，DeepGEMM 手搓 mbarrier + TMA + `l1_full_count/l1_empty_count` 环形计数器，三代 super-kernel 手搓 `phase_done[chunk][phase]` + `grid_sync_v2` + `dispatch_src_ready[8]`。**没有人把它做成 tile 原语。**

**两个候选表述**（见 §8 待决）：

| | 表述 | 优点 | 缺点 |
|---|---|---|---|
| A | **跨设备 tile**：内存层级加一级 peer HBM | 具体、好懂、好画图 | 只覆盖跨卡；解释不了同卡跨 WG 的 producer/consumer |
| **B** | **所有权与可见性**：tile 的第一等属性 | **统一解释** push-vs-pull（`memory-access-patterns.md` Q5）、flag matrix（Pattern C）、barrier 代价表（Principle 3）、同卡跨 WG release-acquire（Technique 3.1）；peer HBM 只是它变得不可回避的那一层 | 更抽象，需要好的 API 才立得住 |

倾向 **B**，把 A 当作 B 的特例来讲。

### 1.3 为什么这条原语应该先在 AMD 上出现

这是本项目的护城河，也是论文的 hook——**不是"AMD 追上了"，是"这个抽象属于 AMD"**：

| 事实 | 来源 |
|---|---|
| 8 卡 XGMI 全互联 + HIP IPC → **peer HBM 可 device-side 直接寻址** | `notes/career-strategy/2026-08-06...` §为什么 AMD 结构性有利（HIP IPC 工作已验证） |
| NVIDIA 侧要靠 TMA + symmetric memory map 绕，且 grid 无角色划分 | `notes/rocmoe/2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md` |
| **RCCL 在语义上不可能 overlap**——collective 是原子的、全 rank 参与的操作，这是架构限制不是实现 bug | `notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md` |
| **AMD CDNA 上的 in-kernel XGMI overlap 是全空白** | `notes/monolith-moe/2026-07-07_latest-moe-systems-papers-scan.md`：「全部为 NVIDIA/通用 GPU 语境，没有一篇针对 AMD CDNA」 |
| HK 自己**完全没有多卡内容** | `papers/hipkittens.md` §9 批判 3 |

第三条尤其重要：它把"为什么必须进 kernel"变成了一个**架构论证**而不是 benchmark 结论。这是论文 Motivation 该长的样子。

### 1.4 与在位者的差异化

| 项目 | 它拥有什么 | 它没有什么 |
|---|---|---|
| **DeepEP** | EP 通信库，hook 给上层做 overlap | 不在 kernel 内融计算；NVIDIA-first |
| **DeepGEMM** | fp8 细粒度 scaling GEMM，极简 | 单卡；GEMM only |
| **triton-distributed / Comet** | tile 级 overlap 尝试 | Triton 抽象层；AMD 路径受阻（rocSHMEM bitcode，见 `2026-04-13_moe_comm_overlap_analysis.md`） |
| **MegaScale-MoE** | 生产级 **op 级** comm-compute overlap，1.88× | 停在 op 级；**这是 op 级的天花板参照** |
| **HipKittens** | tile 原语 + AMD 调度/布局重做 | 无 MoE、无多卡、全规则形状 |
| **AITER / FlyDSL** | 出货口 / 写 kernel 的语言 | 不是论点；AITER 训练侧峰值投入只有 attention backward 一个算子 |
| **本项目** | **in-kernel chunk 级融合的原语词汇表 + regime 判据，AMD/XGMI** | 不覆盖推理、不覆盖 attention、不做通用 GEMM |

---

## 2. 范围

### 2.1 做

- MoE 层 forward + backward 的融合 super-kernel 所需的 tile 原语
- 三个层级的原语族（§3）：放置与流水 / 所有权与可见性 / 角色与骨架
- 每条原语自带 **rocprof 判据**（把 `cco-pipeline-overlap` 的三个 Diagnostic 段变成库的 verification mode）
- regime 判据：什么时候融合赢、什么时候该退回库路径

### 2.2 不做（明确写进 README，防止范围膨胀）

| 不做 | 理由 |
|---|---|
| 通用 GEMM | hipBLASLt / CK / FlyDSL / HK 四个在位者；`mfma_tile.h` 已证明规则形状能到顶，无论点空间 |
| attention | dense 被 aiter ASM 吃满；sparse 有 11 个后端赛马（见 `notes/hk-attn-bwd/2026-08-12_1354...`） |
| 新 DSL | FlyDSL 官方 781 commits / ~60 作者 / 已是 aiter 的 pinned 依赖；HK 也在 |
| 推理 / decode / paged | AITER 的地盘，且与本论点无关 |
| 覆盖面 | DeepGEMM 靠简洁赢。**核心 < 5000 行** |

### 2.3 载体：FlyDSL（**2026-08-12 16:20 修正，原写 HIP C++ header 是错的**）

**原判断错在事实层**：我以 `mfma_tile.h`（HIP C++，8192×4096×7168 上 1290 TFLOPS / 99.3% MFMA issue）为依据推荐 HIP C++ header。但**跑出 2.4× 的 gen-3 不是 HIP，是 100% FlyDSL**：

| gen-3 的实际构成 | 路径 | 行数 |
|---|---|---|
| GEMM tile（**库级效率的来源**） | `primus_turbo/flydsl/utils/gemm_helper.py` | **1496** |
| mxfp8 GEMM tile | `flydsl/mega/fp8/gemm_mxfp8_tile.py` | 363 |
| peer HBM 寻址（= F1 的 `peer` tier） | `flydsl/mega/symm_buffer.py` + `fp8/symm_buffer.py` | 422 + 728 |
| **所有权与可见性原语（= F2）** | `flydsl/mega/prims.py` + `fp8/prims.py` | 256 + 84 |
| barrier | `flydsl/mega/barrier.py` + `fp8/barrier.py` | 116 + 105 |
| EP intranode 通信 | `flydsl/mega/ep_intranode.py` | 309 |
| autotune | `flydsl/mega/tune_utils.py` | 136 |
| 四个 fused stage（客户代码） | `dispatch_grouped_gemm_*` / `grouped_gemm_combine_*` / `swiglu_*` / `dispatch_prologue*`，bf16 + fp8 两套 | ~2100 + ~2900 |
| **合计** | `primus_turbo/flydsl/mega/` + `utils/gemm_helper.py` | **~10,000** |

`csrc/` 下**没有** mega / MoE 的 HIP kernel（只有 `moe_permute`）。**峰值 GEMM tile 是 FlyDSL 发射的 MFMA（经 rocdl），把它改写成 HIP C++ 是纯风险无收益。**

**FlyDSL 可以被独立 repo 依赖**（这是原判断的另一个错处）：

- 仓库是 `github.com/ROCm/FlyDSL`（ROCm 官方组织）
- **公开可 pip 装**：当前 `0.3.1`，已发布 17 个版本（0.1.1 → 0.3.1）
- aiter 自己就 pin 了 `flydsl==0.1.7` 并在 build 时 AOT 预编译其 MOE/GEMM kernel

所以修正后的载体决策：**FlyDSL，并明确"骑在官方 DSL 上而不是与它竞争抽象层"**。

⚠ **要承担的代价（用自己的证据）**：绑 FlyDSL 版本 = 吃它的版本churn。`notes/MegaMoeFlydsl/2026-08-01_1440_mega_fp8_bwd_only_regression_flydsl024_t8192_mi355x.md` 就是一次 flydsl 0.2.4 引起的回归；aiter pin 在 0.1.7 而上游已 0.3.1，版本 skew 是真实的。对冲照抄 aiter：**pin + CI 双跑**。

**放弃 HK 的 C++ header 形式所失去的**：可读性 / 零依赖 / 直接可引。这是真损失，但换来的是不重写已经跑到 2199 TFLOPS 的东西——这笔交易值得做。HK 的贡献因此收敛为**纪律与技法**（§0.4），不是形式。

---

## 3. 原语 API 草案

> **2026-08-12 16:20 修正**：本节原写作"从 494 行 skill 提炼出的草案"。更准确的说法是——**库的内容一半是散文（skill），另一半已经是代码，而且是跑出 2.4× 的那一半。** 见 §2.3 的清单。所以本节的性质从「设计一个新 API」变成「给一个已存在的 API 加纪律」。
>
> 对照表（charter 的族 → 已存在的实现）：
>
> | 族 | 已存在 | 缺什么 |
> |---|---|---|
> | F1 GEMM tile / 流水 | `utils/gemm_helper.py` (1496) · `fp8/gemm_mxfp8_tile.py` (363) | swizzle/pad 常量仍是手写，writer/reader 不共享类型 |
> | F1 `peer` tier | `mega/symm_buffer.py` + `fp8/symm_buffer.py` (1150) | 没被表述成 tile 的一个 tier |
> | **F2 所有权与可见性** | **`mega/prims.py` + `fp8/prims.py` (340)**：`memory_fence(order, scope)` · `ld(order, scope, space)` · `st(...)` · `atomic_add(...)` · `spin_timed_out` · `copy_warp` | **scope/order 是运行时参数而非编译期** → 编译器无法拒绝"chunk 循环里用 system scope" |
> | F2 barrier | `mega/barrier.py` + `fp8/barrier.py` (221) | "全 kernel 只允许两个 system fence"无强制 |
> | F3 骨架 | 四个 fused stage kernel 本身 | 骨架是复制粘贴的，不是可复用类型 |
>
> **`prims.py` 已经把 scope/order 参数化了**，这正是 F2 草案的核心语义——它已经实现，只是没被纪律化。下面的 API 草案应读作"这些语义该长成什么形状"，不是"从零设计"。

以下草案的语义来源：`.cursor/skills/cco-pipeline-overlap/SKILL.md`（494 行，3 原则 × 4–6 技法 + 4 pattern）。

### F1 — 放置与流水（来自 Principle 1 + 2）

```cpp
// Tier 是 tile 的一等参数，peer 是新增的那一层
enum class tier { reg, lds, hbm, peer };

template <class T, int M, int N, class Layout, tier Tier>
struct tile;

// swizzle / pad 是 Layout 的参数，不是手写的 XOR 常量
using lds_a = layout::padded<K_TILE, /*PAD=*/4>;        // Technique 2.1
using lds_b = layout::xor_swizzled<128 /*byte block*/>;  // Technique 2.2

// 流水深度是参数，不是复制粘贴的 buffer 索引
template <int Depth> struct staged;                      // Technique 1.1 / 1.2, Pattern D

// 直接进 LDS，不过 VGPR
void load_async(tile<T,M,N,lds_a,tier::lds>& dst,
                tile<T,M,N,L,tier::hbm> const& src);      // Technique 2.4
```

**关键设计取舍**：swizzle 做成 `Layout` 参数而不是让用户手写 XOR，是为了消灭 skill 的 Quick Reference 里那条失败模式——「Numerical drift after fuse / usually 2 (swizzle off-by-one) / Re-verify XOR constants on writer/reader」。**writer 和 reader 共享同一个 Layout 类型，off-by-one 在类型系统里就不可能发生。** 这是库相对散文的第一个真实增量。

### F2 — 所有权与可见性（来自 Principle 3，本项目的核心）

```cpp
// scope 直接编码 Principle 3 的代价表
enum class scope { wave, wg, agent, system };   // 1-4 / 8-16 / 50-200 / 500-2000 cycles

// publish/acquire 成对出现；single-thread publish 折进实现，用户无法写错
template <scope S> void publish(tile_ref t, ready_slot& slot);   // Technique 3.1 + 3.3
template <scope S> void acquire(ready_slot const& slot, uint32_t epoch);

// flag 矩阵是类型，不是裸数组
template <int Chunks, int Phases> struct ready_matrix;           // Pattern C

// 整个 kernel 只允许两个 system fence，用编译期计数强制
struct system_barrier { /* enter / exit only */ };               // Technique 3.2

// publish 的默认语义是 pull-based fanout（生产者写一次，N 个消费者自旋）
// 而不是生产者广播 N 次                                          // Technique 3.6
```

**这一族是论点的载体。** 三件事让它成为原语而非工具函数：

1. **代价进类型**：`scope` 是模板参数，库可以在编译期拒绝"在 chunk 循环里用 system scope"这类错误——正好是 skill 的 Diagnostic 3 用 `rg` 静态检查的那条规则。
2. **epoch 语义内建**：`grid_sync_v2` 那次踩的坑（"gen counter 必须从 workspace bootstrap，复位会让 leader 看到 stale gen 立即 exit → deadlock"，`notes/rocmoe/2026-05-25_1110...`）是所有权语义没被抽象的直接后果。库应该让 epoch 无法被写错。
3. **publish 的方向性**：`memory-access-patterns.md` Q5（push vs pull / 谁拿主动权）和 Q1（sender 端 pre-pack）在库里应该表现为 publish 的两种实现，并由 tile 的 tier + 形状**推荐**方向，而不是让每个 kernel 重新猜。MonolithEP push 比 RocMoE-v2 pull 快 30% 的原因已经查清（peer L2 prefetcher 跨 row 失效），这个知识应该固化在库里。

### F3 — 角色与骨架（来自 Pattern A/B/D + Technique 1.3/3.4）

```cpp
template <int Comm, int Compute, int Tail> struct roles;         // Pattern A
template <int Chunks, int Phases>          struct chunk_pipeline; // Pattern B
struct work_queue;                                                // Technique 3.4
```

⚠ **这里有一个必须由本项目解决的矛盾**：skill 的 Technique 1.3 推荐 wave specialization（条件：per-expert M < 1024 且 HBM 余量 > 30%），而 HK 论文 §7.1 证明 **AMD 上 wave specialization 是负优化**（寄存器静态均分 → producer 白占寄存器 → 输出 tile 变小 → 算术强度掉；0 producer + 256×256 = 1610 vs 4 producer + 128×256 = 893 TFLOPS）。

两者不矛盾的可能解释：HK 测的是 GEMM，producer 干纯预取（**可以**被计算 wave 顺带做掉，所以 0 producer 赢）；MoE 的 comm role 干跨卡 IPC，**不能**被顺带做掉。而 `buffer_load_lds` 不占 VGPR 这一点又可能让 HK 的反对意见失效。

→ **库不能把 wave specialization 当默认值，必须把它做成一个由判据选择的选项。** 这正是"库携带 regime 边界"的第一个具体实例，也是 rocmoe M4 一直悬着没定的那个决定（M4-α CU 物理隔离 vs M4-β CU 内分 SIMD）。

### F4 — 验证模式（把 skill 的 Diagnostic 变成库的一部分）

三条判据直接来自 skill，应该做成库的 `verify` 编译开关 + 一个 rocprof wrapper：

| 原则 | 判据 | 阈值 |
|---|---|---|
| 1 | `MFMA / CU_BUSY` | ≥ 0.85 |
| 1 | `WAIT_LGKM / cycles` · `WAIT_VMEM / cycles` | ≤ 0.05 · ≤ 0.02 |
| 2 | `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` | ≤ 0.01（chunk 0 除外） |
| 3 | 每 chunk 每 compute WG 的 sync 数 | ≤ 8 |
| 3 | chunk 循环内的 `__threadfence_system` 数 | **0**（只允许在 enter/exit） |

**没有这一族，库就只是一堆 header。** 有了它，"用库写的 kernel 自动满足三条原则"才是可检验的主张。

---

## 4. 证据 → 论文章节编排

这是本项目相对从零开始的最大优势：**实验章节的数据大部分已经在硬盘上**。

| 论文章节 | 用什么 | 来源 |
|---|---|---|
| **Motivation** | RCCL 语义上不可 overlap（架构论证，非 benchmark）；op 级 overlap 的天花板是 MegaScale-MoE 1.88× | `2026-04-14_rccl_overlap_analysis.md`；`2026-07-07` paper scan |
| **Design** | 四条原语，F1–F3 | 本文 §3 |
| **Eval 1：融合有效** | MonolithEP 4.82 ms / 598 TFLOPS / **1.76×** vs PyTorch+RCCL 8.466 ms @512 t/g；RocMoE dispatch **2.23×**；MegaMoE FP8 e2e **1.36×→1.41×** | 三个项目 README |
| **Eval 2：regime 反转（最值钱）** | 512 t/g **1.46×** → 2048 t/g 16.98 vs 9.05 = **0.53×** → 8192 t/g 64.5 vs 18.6 = **0.29×** | `2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md` |
| **Eval 3：判据** | 从 Eval 2 反推 crossover 模型（§5） | 待测 |
| **Ablation：逐技法增量** | LDS PAD=8 grouped GEMM **+25%**；DirectToLDS **+204%**；`grid_sync_v2` super wall **−14%**；`WG_PER_CU=1` T=2048 18.911→12.240 ms **−35%**；dual-quant 融合 **−9.4%**；fc1_dgrad GEMM‖PUSH overlap 省 **1.71 ms / 40%**（效率 87%） | 各 note |
| **负结果（诚实性资产）** | VMEM prefetch 重排 +4% 回滚；barrier 减 sync wall 不变（straggler 而非同步开销）；树规约寄存器压力反噬；BT 128/512 双向回退 | `mxfp8_moe_bwd_perf_summary.md`、`agent/historical_experience/` |

**Eval 2 是这篇论文与所有竞品的区别。** 别人只发赢的 regime。三代失败的量化记录从负债变成了唯一无法被复制的证据。

---

## 5. Regime 判据要测什么

**待验证的假设**：融合的收益 = 被藏起来的通信时间；融合的成本 = GEMM 效率损失 + scaffold 税。T 变大时 per-expert M 变大 → 未融合的库 GEMM 效率变得很好（PyTorch 在大 M 下 strongly super-linear），而融合 kernel 的 GEMM 效率损失 × 大 FLOPs 超过了藏起来的通信 → 反转。

要扫的轴（先一维一维扫，别做全组合）：

| 轴 | 范围 | 为什么 |
|---|---|---|
| **T（tokens per GPU）** | 512 / 1024 / 2048 / 4096 / 8192 | 已知反转发生在 512→2048 之间，**这是主轴** |
| per-expert M（routing skew） | balanced / 真实 skew | 决定 GEMM 效率与 Technique 1.3 的适用性 |
| H / F | DSV3 7168 / 2048 及 ±2× | 决定算术强度 |
| chunk 数 | 2 / 4 / 8 | XGMI burst 大小 vs 流水深度（已知 ~7 MB 达 95% 链路效率） |
| 精度 | bf16 / mxfp8 | fp8 把 GEMM 时间砍半 → 通信占比上升 → 应当**扩大**融合的赢面。**这是一个可预测的、可证伪的推论** |

最后一行是关键：如果判据模型对，那么**精度越低，融合越该赢**——因为通信量不变而计算变快。这给了一个独立的验证点。

⚠ **前置项**：sclk ±30%，`setperflevel high` 不支持。必须先做 `notes/career-strategy/2026-08-12_1505...` 的 S0（噪声地板），否则 crossover 点定不准。

---

## 6. 里程碑

| M | 内容 | 验收 | 为什么这个顺序 |
|---|---|---|---|
| **M0**（**已按 §0.6 替换**） | **三方 A/B**：gen-1 / gen-3 / PyTorch+RCCL，同日同机同 routing，T ∈ {512, 2048, 8192}，**直接测 forward**（不用 fwd+bwd 减法）。顺带刷新 PyTorch+RCCL baseline（5 月的可能已过期） | 确认或推翻 §0.1 的"gen-3 已赢 2.4×"。**这一条决定整个项目的叙事** | 两个 bench 都现成，成本最低，且结论决定后面所有事怎么讲 |
| **M0'** | 论点 + API 草案定稿，**零代码** | 一页别人能反驳的东西；§8 三个待决问题有答案 | 论点错了写代码是浪费 |
| **M1**（**16:20 改写**） | ~~抽取手搓原语成 header，证明抽象税为零~~ —— **作废**：原语已经是库（`prims.py`/`symm_buffer.py`/`gemm_helper.py`），且已跑出 2199 TFLOPS，抽象税已被 gen-3 自己证明为零。**改为「毕业」**：把 `primus_turbo/flydsl/mega/` 抽成一个独立 public repo，只依赖 `pip install flydsl`，在 Turbo 之外复现 M0 的数字 | 独立 repo 从零环境 `pip install` 后跑出与 M0 同一数字（噪声内） | **这才是"换工作带得走"的真实测试**——gen-3 现在住在 AMD 产品仓的子目录里，抽出来 + 论文才是可携带的资产（HK 的 kernel 进了 AITER，但 repo 与论文永远是 Stanford 的） |
| **M2** | regime 扫描 + 判据模型（§5），含 fp8 那个可证伪推论 | 能预测 crossover 点，误差在 S0 的可判阈值内 | 判据是论点的另一半 |
| **M3** | 用库写一个**原来写不出来的** kernel：量化融进 GEMM 的 prologue/epilogue，删掉跨 op 格式往返 | 反向净 FP8 收益 1.29× → ≥1.6× | 见 `2026-08-12_1505_the-quant-gemm-seam.md`——那篇的 S2/S3 就是本库的第一个新客户 |
| **M4** | 论文 | MLSys / ASPLOS 投稿 | — |

M1 之前不写任何新 kernel。M3 与 quant-gemm-seam 那篇是同一件事的两面：**那篇是"要做什么"，本篇是"用什么写"。**

## 7. 风险

| 风险 | 说明 | 对冲 |
|---|---|---|
| **第四代问题** | 三代 super-kernel 已经没长成东西，凭什么第四次成 | **artifact 类型不同**（库+判据 vs 点解 kernel），且论点把输的 regime 写进去。若融合在生产 regime 全输，则论文变成"in-kernel 融合的边界在哪"——**仍然成立，只是结论换向** |
| **抽象税** | 库化后比手写慢 → 论点死 | M1 就是这条的门槛，且放在最前 |
| 单人维护 kernel 库 | 极其消耗 | 范围锁死（§2.2），核心 < 5000 行 |
| AMD 官方自己做 | FlyDSL 团队有能力 | 空白已存在一年以上（paper scan 为证）；且本论点需要多卡 + 训练 + 生产 workload 三者齐备，Turbo/FlyDSL 团队目前是单机单卡视角 |
| **测量噪声** | sclk ±30%，<10% 不可判 | S0 前置；M2 的 crossover 是大效应（0.53× vs 1.46×），不怕噪声 |
| wave specialization 矛盾未解 | F3 的默认值定不下来 | 这是 M0 的产出物之一，见 §8 |

## 8. 待决问题（M0 的输入）

1. **论点用 A（跨设备 tile）还是 B（所有权与可见性）？** 倾向 B——它能统一解释 push/pull、flag matrix、barrier 代价表、以及同卡跨 WG 的 producer/consumer，而 A 只覆盖跨卡。但 B 需要 API 撑得住抽象，否则会显得空。**建议：论文标题用 B，图和第一节用 A 来落地。**
2. **wave specialization 到底该不该是默认？** skill Technique 1.3 说在窄条件下该用，HK §7.1 说 AMD 上是负优化。假设是"comm role 不能被计算 wave 吸收 + `buffer_load_lds` 不占 VGPR"两点让 HK 的结论在此不适用。**这需要一个实测**（rocmoe M4-α vs M4-β 的那个 A/B），成本不高，且它同时是 F3 的默认值和论文的一个 finding。
3. **repo 名字。** `peer-tiles` 强调 A；若走 B，更贴的是强调所有权的名字。不要用 `*-kittens`——蹭 lineage 会削弱"第四条原语"这个独立贡献。

## 相关文件

**本项目的原料（都已存在）**
- `.cursor/skills/cco-pipeline-overlap/SKILL.md` — **494 行，就是库的内容**，3 原则 × 4–6 技法 + 4 pattern + 3 Diagnostic
- `knowledge/kernels/memory-access-patterns.md` — 五问（Q3 register vs LDS、Q4 lockstep vs independent、Q5 push vs pull），是 F2 方向性语义的依据
- `notes/monolith-moe/` 42 篇 · `notes/rocmoe/` 28 篇 · `notes/MegaMoeFlydsl/` 9 篇 — Eval 1/2 + Ablation + 负结果
- `notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md` — Motivation 的架构论证
- `notes/monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md` — **Eval 2**
- `notes/monolith-moe/2026-07-07_latest-moe-systems-papers-scan.md` — 空白区论证 + op 级天花板
- `notes/rocmoe/2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md` — NVIDIA 侧对照 + 原语映射表

**对照物**
- `papers/hipkittens.md` §2（三条原语）· §7.1（wave specialization 为何亏）· §9 批判 3（无 MoE 无多卡）
- `knowledge/libraries/{aiter,composable-kernel,_patterns}.md`

**相邻计划**
- `notes/career-strategy/2026-08-12_1505_the-quant-gemm-seam.md` — M3 的内容来源；S0（噪声地板）是本项目 M2 的前置
- `notes/career-strategy/2026-08-06_primus-positioning-boundary-dissolution.md` — execution model 母论点；本项目是它的库形态
