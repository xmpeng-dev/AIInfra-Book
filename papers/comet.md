# Comet: Fine-grained Computation-Communication Overlapping for Mixture-of-Experts

> **arXiv:** [2502.19811](https://arxiv.org/abs/2502.19811) (v3, 2025-03-04) · **PDF:** https://arxiv.org/pdf/2502.19811
> **发表:** MLSys 2025 (评审 5/5/5/4)
> **机构:** ByteDance Seed + 上海交通大学
> **代码:** https://github.com/bytedance/flux (Flux/COMET)
> **领域:** MoE 分布式训练/推理 · 计算-通信细粒度 overlap · 单 kernel 融合
> **核心贡献:** 用 **shared-tensor 依赖分解 + 自适应 thread-block 分配**,在**单个融合 kernel 内**把 All-to-All 通信与 GroupGEMM 计算做 tile 级 overlap;隐藏 86.5% 通信延迟,单 MoE 层 **1.96×**、端到端 **1.71×**,已在万卡生产集群部署。

> **重读说明 (2026-06-02):** 本笔记据 arXiv v3 全文重写,修正旧版若干硬伤——旧版把方法写成 "warp 专用化 + RDMA + 反向 overlap + 内存 35% 节省",并拿 FlowMoE(2510.00207,晚于本文 8 个月)做对照、给出 "2.3×" 等数字,均非原文。COMET 实为 **thread-block(线程块)专用化**、用 **NVSHMEM**(非裸 RDMA)、**只优化前向**、内存开销仅 2MN(可忽略),真实加速 **1.96× 单层 / 1.71× 端到端**。本文与正在做的 **RocMoE-v2 super-kernel** 高度同源,新增第五节专门对照。

---

## 一、问题分析

### 1.1 研究背景

- MoE 把专家分布到多 GPU(expert parallelism),每个 MoE 层都要做 token 的 **dispatch(All-to-All)+ expert GEMM + combine(All-to-All)**。
- 在 8×H800 + Megatron-LM 上实测,**MoE 层的设备间通信平均占整模型执行时间的 47%**(Mixtral / Qwen2-MoE / Phi3.5-MoE,seq 4096/8192)。通信是头号瓶颈。
- 既有 overlap 方案(FasterMoE、Tutel、PipeMoE、ScheMoE)都走 **粗粒度 chunk 流水线**:把输入切成几个大 chunk,让 chunk 间的 comm 与 comp 重叠。

```
MoE 层执行(单 GPU 视角,时间向右 ──→):

完全串行(无 overlap):
  [recv dispatch] → [expert GEMM] → [send combine]
   ███ 通信等待 ███               ███ 通信等待 ███
   GPU 空转                       GPU 空转
   通信 ≈ 47% 时间全暴露在关键路径上
```

### 1.2 既有方案的两个根本低效(本文动机)

```
粗粒度 chunk 流水线(前作,把输入切成 2 个 chunk):

  stream0 (comp):            [GEMM chunk1][GEMM chunk2]
  stream1 (comm): [recv c1]            [recv c2]      [send c1][send c2]
                   ███ 首段不可覆盖              ███ 尾段不可覆盖
                   ↑                            ↑
              收第一个 chunk 时 GPU 空转    发最后一个 chunk 时 GPU 空转

  且 chunk 越小 → 每 expert token 越少 → tile 利用率掉 → t1+t2 > t
```

| 低效 | 机理 |
|---|---|
| **粗粒度切分伤计算效率** | chunk 越切越小,每个 expert 拿到的 token 数变小,GroupGEMM tile 利用率下降,切完后 `t1+t2 > t`;且首尾 comm 段(收第一个 chunk、发最后一个 chunk)天然无法被计算覆盖 → GPU 空转。 |
| **kernel 级调度不可控** | 几乎所有前作把 comm 和 comp 封成**独立 kernel 放不同 stream**,硬件资源分配由 runtime 决定、非确定,comp(chunk1)与 recv(chunk2)经常错位对不齐;MoE 动态路由又让每个 expert 的 token 数(M)运行时才知道,负载多变。 |

### 1.3 两个核心洞察

1. **解决 comm 与 comp 的"粒度失配"是高效 overlap 的关键**:token 是通信的最小单位,但 GEMM 以 128×128 tile 为单位 —— 一个 tile 可能需要分散在多 GPU 上的 128 个 token,数据依赖复杂且不规则。要做 fine-grained overlap,必须让"每个计算 tile 只读/写它真正需要的数据"(经 Unified Virtual Address 直接访存),并重排数据与计算顺序去藏延迟。
2. **资源分配要在 kernel 内运行时自适应**:comm/comp 的负载随 input shape、并行策略(TP/EP)、硬件(NVLink vs PCIe)而变,把工作封成独立 kernel 就失去了对资源比例的精确控制。

---

## 二、COMET 核心设计

COMET 把一个 MoE 层看成两条 **生产者-消费者流水线**:
- **layer0 = comm→comp 流水线**:dispatch(All-to-All / AllGather)→ FC1 GroupGEMM;
- **layer1 = comp→comm 流水线**:FC2 GroupGEMM → top-k reduce + combine(All-to-All / ReduceScatter)。

两条流水线由一块 **shared tensor**(生产者的输出 buffer = 消费者的输入 buffer)连接,全局形状均为 `(M×topk, N)`。

```
                    ┌── shared tensor ──┐          ┌── shared tensor ──┐
  layer0:  dispatch ─►│ (M×topk, N)       │─► FC1   │ (M×topk, N)       │
           (comm)     └───────────────────┘  GEMM   └───────────────────┘
                       生产者输出=消费者输入        (producer)   (consumer)
                                                        │
  layer1:  FC2 GEMM ─►┌── shared tensor ──┐─► topk-reduce + combine
           (producer) │ (M×topk, N)       │   (consumer, comm)
                      └───────────────────┘

  COMET 优化路径:  ①decompose(沿独立维切) → ②reschedule(重排 tile 顺序)
                    → 再用自适应 thread-block 分配生成融合 kernel
```

两个核心机制:

### 2.1 机制一:Shared-tensor 依赖分解(dependency resolving)

**目标**:把"消费者必须等生产者全部完成"的粗粒度依赖,打散成 tile 级细粒度依赖。

**① 沿哪个维度分解(decompose)** —— 只在消费者算子上"数据独立"的维度切:

| 流水线 | 消费者 | shared tensor 角色 | 可分解维度 | 原因 |
|---|---|---|---|---|
| layer0 (comm→comp) | GroupGEMM | GEMM 的输入矩阵 | **M(token)维** | token 之间相互独立;沿 embedding(归约)维切不可行 |
| layer1 (comp→comm) | top-k reduce + combine | reduce 的输入 | **N 维** | top-k reduce 沿 M 维归约 → M 维 token 强耦合,只能切 N |

**② 怎么重排(reschedule)** —— 切完后子张量要重组回 tile 粒度,遵循两条原则:(a) 对齐原始计算 tile 粒度保利用率;(b) **优先调度消费者能立刻用上的那部分**,让消费者尽早起步。

```
layer0 重排:按 source rank 排序,本地 token 的 tile 先算
                                    GroupGEMM 计算序列 ──→
  原始 shared tensor      重排后        tile#0    tile#1    tile#2
  (token 乱序混排)   ─►  本地(rank0) ─►  [本地]    [远端]    [远端]
                        远端(rank1)       ↑         ↑
                                      立即可算   此时还在
                                                 XGMI/NVLink 传输
                        → 本地 tile 的计算 覆盖 远端 token 的通信

layer1 重排:GroupGEMM 按列(N)执行,而非逐 expert 顺序
        N ──→                          前 TN 列一算完
  ┌──┬──┬──┬──┐  Expert#0             reduce+combine 立刻起步
  │①│②│③│④│   ↓ 执行顺序 ① → ② → ③ → ④  (不必等所有 expert)
  ├──┼──┼──┼──┤  Expert#1
  │  │  │  │  │  top-3 reduce 沿 M 维(列内)归约
  └──┴──┴──┴──┘
   TN
```

- **layer0**:把 token **按 source rank 排序**,GroupGEMM 的 tile 计算序列从**本地 token 的 tile 开始算**,远端 token 同时还在传输 —— 计算覆盖通信。
- **layer1**:GroupGEMM **按列(N 维)执行**而不是逐 expert 顺序执行 —— 只要前 `TN` 列算完,reduce + combine 就能立即开始,而不必等所有 expert 算完。

这一步纯粹是**数据布局 + 计算顺序的重排**,不改 GEMM kernel 本身。

### 2.2 机制二:自适应 workload 分配(adaptive workload assignment)

**① Thread-block 专用化(horizontal fusion,水平融合)**

```
垂直融合(vertical,前作)         水平融合(horizontal,COMET)
─────────────────────────         ─────────────────────────────
每个 SM 上的同构 block:           SM 按角色物理隔离:
 ┌─────────────────┐               ┌──────────┐   ┌──────────┐
 │ GEMM warp       │               │ SM 0..np │   │ SM .. nc │
 │ + comm warp(塞) │ ← 互相干扰     │ comp     │   │ comm     │
 │   I/O 卡住 TMA  │               │ (GEMM)   │──►│ (NVSHMEM │
 └─────────────────┘               │ CUTLASS  │   │  reduce/ │
   overlap 不规则、延迟不定          │ 原样不改  │   │  收发)   │
                                    └──────────┘   └──────────┘
                                    comp 块用与融合前完全相同的 kernel
                                    np/nc 划分点按 shape 自适应
```

- 反面:**vertical fusion(垂直融合)**——把 comm I/O 塞进 GEMM 的 prologue/epilogue,所有 thread block 同构。问题:overlap 不规则、延迟不确定;且 token 级 fine-grained I/O 会拖垮 GEMM kernel 效率,在 Hopper 上尤其严重(长延迟远端 I/O 会卡住 TMA 异步流水)。
- COMET 做法:**comm thread block 与 comp thread block 物理隔离**,分占不同 SM。
  - **comp(GEMM)thread block 用与融合前完全相同的 CUTLASS 实现**(producer warp 用 TMA 异步搬数到 shared memory,consumer warp 发 tensor-core MMA),计算效率不被通信干扰。
  - **comm thread block** 从 global memory 读 consumer warp 产出的结果,做 top-k reduce 后写本地或经 NVSHMEM 发往远端。
  - 这套模型可移植到 Ampere/Volta,只需替换 compute block 的实现。
  - **硬件约束**:不把 comm warp 和 comp warp 塞进同一个 block —— warp 内线程数限制会让 comm 吃不满带宽,且 comm warp 会干扰同 block 的 comp warp。

**② 自适应 thread-block 数划分**

设融合 kernel 共 `n` 个 thread block,其中 `np` 个做生产者、`nc` 个做消费者(通信)。最优 `np/nc` 划分点取决于 input shape 与并行策略:

- 实测(Hopper 132 SM,layer1):`TP=8` 时 M 从 4096→16384,最优 `nc` 从 18→26;`TP` 从 8→4(M=16384),最优 `nc` 从 26→46。
- COMET 预编译**多个不同划分点的 kernel**,部署前 profile 出每种配置的最优值存成 metadata,运行时按 metadata 选 kernel。

### 2.3 实现要点

- **规模**:约 12k 行 C++/CUDA + 2k 行 Python,集成进 Megatron-LM,提供 Python API。
- **GEMM**:基于 **CUTLASS** 模板;一个关键优化 —— layer0 的输入行索引每次 K 迭代都要从 global memory 取,COMET 把行索引**缓存进寄存器**,大幅降访存。
- **通信库**:用 **NVSHMEM**(非 NCCL),因为它提供细粒度、GPU-initiated 的跨 GPU 全局地址空间访问,适配 kernel 内的 token 级 I/O。
- **内存开销**:NVSHMEM 通信 buffer = `2MN`(BF16),全模型共享一块、跨层跨专家复用,实测 32–64 MB,可忽略。

---

## 三、实验效果

### 3.1 实验设置

| 项 | 内容 |
|---|---|
| 硬件 | 8×NVIDIA H800(80GB,NVLink);另在 8×L20(46GB,PCIe,~25 GB/s)上验证带宽受限场景 |
| 软件 | CUDA 12.3 · NVSHMEM 2.11 · PyTorch 2.4.0 · Megatron-LM |
| 模型 | Mixtral-8x7B(E=8,topk=2,N=4096,K=14336)· Qwen2-MoE-2.7B(E=64,topk=4,N=2048,K=1408)· Phi-3.5-MoE(E=16,topk=2,N=4096,K=6400) |
| Baselines | Megatron-Cutlass · Megatron-TE · FasterMoE · Tutel |

### 3.2 主要结果

| 指标 | 结果 |
|---|---|
| **端到端延迟下降** | vs Megatron-Cutlass **34.1%** · Megatron-TE **42.6%** · FasterMoE **44.4%** · Tutel **31.8%** → 平均 **1.71×** |
| **单 MoE 层加速** | 平均 **1.96×**;随 token 数变化稳定在 **1.28×–2.37×**(M 小时优势最明显,省 host 调度开销) |
| **通信隐藏率** | COMET **86.5%** vs FasterMoE 29.2% vs Tutel 68.6%(且 expert 计算效率不受影响) |
| **不同 E / topk** | 1.16×–1.83× |
| **负载不均(std)** | 生产环境平均 std=0.032;std 增大时所有系统都变慢,COMET 始终领先 |
| **跨集群(L20 PCIe 带宽受限)** | 1.19×–1.46× |
| **生产部署** | 万卡级集群,节省数百万 GPU 小时 |

### 3.3 与 baseline 的定性差异

- Megatron-Cutlass ≈ Megatron-TE(只差 GEMM 实现,都不 overlap);Tutel 靠优化 all-to-all + 调度部分重叠,但 E 大(Qwen2)时调度开销吃掉优势;FasterMoE 只支持 EP,且 expert 小时 kernel 调用开销主导。
- COMET 优势来源:**充分 overlap + 融合 kernel 内调度大幅减少 CPU 侧开销**,且对 TP 增长不敏感(TP 拆专家会产生碎片小 GEMM,baseline 退化而 COMET 因 shared-tensor 重排保住了计算效率)。

---

## 四、业界类似方案与定位

| 方案 | overlap 粒度 | 实现层次 | 关键局限 |
|---|---|---|---|
| FasterMoE | 流水度=2 | kernel 级调度 | 粗粒度,隐藏率仅 29% |
| Tutel | 启发式/手设流水度 | kernel 级调度 | 搜索空间有限,E 大时调度开销大 |
| PipeMoE / ScheMoE | chunk 调度 | kernel 级调度 | 不解决细粒度数据依赖 |
| Flux(同团队前作) | tile 级 | kernel 融合 | 面向 dense TP overlap,非 MoE 动态路由 |
| **COMET(本文)** | **token↔tile,shared-tensor 分解** | **单融合 kernel + thread-block 专用化** | 仅前向;依赖 NVSHMEM + 离线 profile |

**本文定位**:把 dense 模型里"分解依赖做 overlap"的思路(Wang et al. ASPLOS'22、Centauri、Flux)迁移并适配到 **MoE 的动态、不规则数据依赖**上,核心创新是 **shared-tensor 维度分解 + tile 重排 + thread-block 级资源自适应**。

---

## 五、与 RocMoE-v2 super-kernel 的对照(重点)

COMET 与正在做的 [RocMoE-v2](../notes/rocmoe/README.md)(8×MI355X 上的 MoE 融合 super-kernel)是**同一类工作的两个独立实现**——都在"单 kernel 内做 MoE 计算-通信细粒度 overlap"。把两者拉到同一张表,RocMoE 踩过的坑和 COMET 的设计高度互证。

### 5.1 逐维度对照

| 维度 | COMET | RocMoE-v2 |
|---|---|---|
| 硬件 / 互联 | NVIDIA H800 / L20,NVLink / PCIe | AMD MI355X(gfx950 CDNA4),XGMI 全互联 |
| 通信库 | **NVSHMEM**(成熟库,GPU-initiated) | **手写 HIP IPC peer-access + scoreboard**(无现成 NVSHMEM 等价) |
| 融合范围 | 一层拆两个融合 kernel(comm→comp / comp→comm) | **整个前向(dispatch→FC1→SwiGLU→FC2→combine)塞进一个 persistent super-kernel**,更激进 |
| 细粒度依赖机制 | shared-tensor 沿 M(layer0)/ N(layer1)分解 | **Layout-P**(`[expert][block_b][slot]`)+ 64-bit `block_ready` 位图 scoreboard |
| 重排策略 | 按 source rank 排序,**本地 token tile 先算** | **receiver-pull**:compute WG 看到自己那块 `block_ready` 即起步,不等最慢 sender |
| comm/comp 隔离 | **thread-block 专用化**(comm block 与 GEMM block 占不同 SM) | **`__launch_bounds__(_,1)` + `kNGemmWGs`/`kNDispatchWGs` 物理隔离 CU**(M4-α) |
| 资源比例划分 | 自适应 `np/nc`,**离线 profile + 运行时选 kernel** | `kSubWGs` build-time knob(默认 64 dispatch + 184 GEMM CU,M1c-E) |
| GEMM | CUTLASS grouped GEMM | `mfma_tile.h`(99.3% MFMA,从 MonolithEP cherry-pick) |
| 范围 | **仅前向**;生产部署 | 前向(开发中),反向作为 M8 规划 |
| 结果 | 1.96× 单层 / 1.71× e2e,隐藏 86.5% 通信 | 目标 BF16 ≤7ms(反超 PyTorch+RCCL 1.3×);T≥2048 时 FC1 已 100% overlap-hidden |

### 5.2 三个最强的"同源"点

1. **物理隔离 comm 与 comp 是收敛结论**。COMET 明确反对 vertical fusion(把 comm 塞进 GEMM block),理由是"comm warp 会干扰同 block 的 comp warp + 吃不满带宽"。RocMoE 在 **M2-G α 撞到了一模一样的现象**:把 dispatch CU 和 GEMM stub WG 用 `launch_bounds(_,2)` 共驻一个 CU 时,**dispatch wall 从 5.51→8.43 ms(+53%)**,根因正是"两者共抢 wave scheduler / L2 line"。随后 **M4-α 用 `__launch_bounds__(_,1)` 把两类 WG 强制分到不同 CU,dispatch wall 全 T 段降 35–37%** —— 这就是 COMET "thread-block 专用化" 在 AMD CDNA4 上的等价物。**两边各自独立得出"通信和计算必须占不同的物理执行单元"。**

```
   COMET (Hopper SM)                RocMoE-v2 (CDNA4 CU)
   ─────────────────                ────────────────────
   comp SM  │  comm SM              GEMM CU  │  dispatch CU
   (CUTLASS)│ (NVSHMEM)             (mfma_tile)│ (IPC pull)
            │                                │
   thread-block 专用化               __launch_bounds__(_,1)
   (np/nc 自适应)                    (kSubWGs: 184G + 64D)
            ▲                                ▲
            └──── 同一收敛结论:comm/comp 占不同物理执行单元 ────┘
            反例代价:塞一起 → COMET TMA 卡顿 / RocMoE dispatch +53%
```

2. **重排计算顺序,让消费者从"本地/已就绪"数据先算**。COMET 按 source rank 排序、本地 token tile 先算;RocMoE 用 receiver-pull + per-block scoreboard,compute WG 只在自己依赖的 block 就绪时起步。两者都把"等全部 token 到齐"的粗依赖,降成"等这个 tile/block 的 token 到齐"。

3. **comm/comp 资源比例需要按 shape 调**。COMET 的 `nc` 随 M 和 TP 漂(18→46),用离线 profile;RocMoE 的 `kSubWGs` 经过五档 sweep + 三次返工才钉在 8,且明确"真值要 M2-G γ 在真 FC1 body 下实测"。**都承认这个划分点是 shape-dependent、不能拍死。**

### 5.3 RocMoE 可以从 COMET 借鉴的点

| COMET 的做法 | 对 RocMoE 的启示 |
|---|---|
| **离线 profile 多个划分点 → 运行时按 metadata 选 kernel** | RocMoE 的 `kSubWGs` 是 build-time 单值;可考虑预编译多档、按 (T, skew) 运行时选,尤其 hot_cov50 下 dispatch 自己会变 critical path |
| **layer1 GroupGEMM 按列(N)执行,让 combine 尽早起** | RocMoE combine 仍走 pull;可评估 FC2 是否也能"按 N 列产出即推 combine",进一步藏 combine 段 |
| **行索引缓存进寄存器降 K 迭代访存** | mfma_tile.h 的 grouped GEMM 取行索引路径可对照,看是否有同类省访存空间 |
| **thread-block 专用化对架构可移植**(换 compute block 即可) | RocMoE 的 role 划分若抽象成"可替换 compute role",未来换 GEMM tile 实现(mxfp8 / f8f6f4)代价更小 |

### 5.4 RocMoE 相对 COMET 更激进 / 更难的地方

- COMET 一次只融合**半层**(comm→comp 或 comp→comm),靠 NVSHMEM 兜底跨 GPU;RocMoE 把**整个前向五个 phase 融成一个 persistent super-kernel**,且无 NVSHMEM,要手写 IPC + scoreboard + cross-rank phase barrier —— 调度税(grid_sync)成了 RocMoE 的主要开销(Phase 1 `grid_sync_v2` 单项就 -14%),这是 COMET 不会遇到的。
- COMET 在 NVLink(带宽充裕)上 1.96×;RocMoE 在 XGMI 上还要先**反超 PyTorch+RCCL**,且 receiver-pull 比 push 每 token 慢 ~43%(架构性),靠 overlap 把它藏掉。
- COMET 只做前向且已生产部署;RocMoE 是研究线,反向(M8)、mxfp8(M6)都还在路线图上。

---

## 六、局限与复现

### 6.1 论文局限

- **只优化前向**,反向传播的 comm-comp overlap 未涉及。
- 依赖 **NVSHMEM**(NVIDIA 专有),非 NVIDIA 平台需要等价的细粒度 GPU-initiated 通信原语。
- 自适应划分靠**离线 profile + 预编译多 kernel**,新 shape/硬件需重新 profile。
- 评测限 8 GPU 单节点(NVLink / PCIe),跨节点 RDMA 大规模 overlap 行为未在论文展开(但已生产部署万卡)。

### 6.2 复现清单

| 项 | 状态 |
|---|---|
| 代码开源 | ✅ https://github.com/bytedance/flux (Flux/COMET) |
| 硬件 | 8×H800(NVLink)/ 8×L20(PCIe);CUDA 12.3 + NVSHMEM 2.11 |
| 模型配置 | Mixtral-8x7B / Qwen2-MoE-2.7B / Phi-3.5-MoE,见表 2 |
| 关键依赖 | CUTLASS(GEMM)+ NVSHMEM(comm)+ Megatron-LM(集成) |

---

## 七、术语表

| 英文 | 中文 | 说明 |
|---|---|---|
| Shared tensor | 共享张量 | 生产者输出 = 消费者输入的桥接 buffer,形状 `(M×topk, N)` |
| Dependency resolving | 依赖分解 | 沿独立维度切 shared tensor + 重排 tile 顺序 |
| Thread-block specialization | 线程块专用化 | comm block 与 comp block 物理隔离,占不同 SM(= 水平融合) |
| Vertical fusion | 垂直融合 | 把 comm I/O 塞进 GEMM 的 prologue/epilogue(本文反对) |
| `np / nc` | 生产者/消费者块数 | 融合 kernel 内做计算 vs 通信的 thread block 划分点 |
| NVSHMEM | — | NVIDIA 细粒度、GPU-initiated 跨 GPU 通信库 |

---

## 延伸阅读

- 🔧 **Flux / COMET 代码** — https://github.com/bytedance/flux
- 📄 **Flux**(同团队 dense overlap 前作) — arXiv:2406.06858
- 📄 **Wang et al. "Overlap communication with dependent computation via decomposition"** — ASPLOS'22(分解依赖做 overlap 的 dense 源头)
- 📄 **Centauri** — ASPLOS'24(通信分区调度)
- 📄 **Tutel** — MLSys'23(自适应 MoE + 2D 层次 all-to-all)
- 📓 **RocMoE-v2 架构与进展** — [`../notes/rocmoe/README.md`](../notes/rocmoe/README.md)、[`../notes/rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md`](../notes/rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md)
- 🛠 **CCO 实现技巧** — [`../.cursor/skills/cco-pipeline-overlap/SKILL.md`](../.cursor/skills/cco-pipeline-overlap/SKILL.md)

---

*笔记据 arXiv:2502.19811 v3 全文重写于 2026-06-02(原 2026-03-07 版含多处与原文不符的内容,已修正)。HTML 阅读版见 [`comet.html`](./comet.html)。*
