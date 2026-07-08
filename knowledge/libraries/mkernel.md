# mKernel (uccl-project)

> **Repo:** `uccl-project/mKernel` &nbsp; **Local path:** 未克隆 — GitHub remote
> **Snapshot:** `7341f11d0` &nbsp; `2026-06-08` &nbsp; on branch `main`
> **Size:** ~14 MB &nbsp; **License:** MIT
> **Distilled:** 2026-06-09
> 来源:README + `include/` 公开头文件 doxygen 注释(未读 `.cu` 实现)。
> Blog: <https://uccl-project.github.io/>。

## TL;DR

mKernel 是 UCCL 项目下的一组 **"把整条 多-GPU/多-node 集合通信 + 计算
塞进一个 persistent CUDA kernel"** 的 fused 算子(AG+GEMM、GEMM+AR、
GEMM+RS、MoE dispatch+grouped-GEMM、MoE dispatch+FFN+combine、Ring
Attention)。它的核心主张是**用 CTA 角色专精(SM specialization)+
GPU-driven 网络(直接写 libibverbs,不经 NCCL/NVSHMEM)+ tile 粒度
producer→consumer overlap**,让矩阵乘在集合通信还没结束时就开始消费已
到达的 tile。最值得我们借鉴的,是它**一整套无锁、单生产者-单消费者、
WRITE_WITH_IMM + 单调 arrival flag 的跨节点 dataflow 契约**——这正是
RocMoE / MonolithMoE super-kernel 想在 AMD 上做的 inter-node 版本。

注意:**纯 NVIDIA Hopper(`sm_90a`)实现**,大量依赖 TMA、warpgroup
MMA(wgmma)、NVLink multimem/VMM multicast、CUDA IPC。对我们而言它是
**设计参考**而非可直接落地的库;借鉴的是模式,不是代码。

## 1. 库定位 (positioning)

- **一句话:** 一个研究性质的 **multi-node megakernel 算子集**,把
  "collective + GEMM/Attention" 融成单次 launch,自带从零搭的
  GPU-driven RDMA 通信栈。
- **它是什么:**
  - 6 个 fused 算子(见下表),每个 = 一个 persistent kernel,一次
    `fused()` launch 覆盖 intra-node + inter-node 全流程。
  - 一套**自研通信层**(`include/comm/`):CUDA IPC(intra-node)、
    NVLink multimem/VMM multicast、以及基于 **libibverbs** 的 CPU
    proxy + D2H FIFO 跨节点 RDMA(inter-node),完全绕开 NCCL /
    NVSHMEM。
  - 一套 **`dist::` 分布式 tensor 抽象**(`include/dist/`),把
    ThunderKittens 的 tile 元数据(`kittens::st/sv`)结构化复用进
    多-GPU descriptor,但不继承 `kittens::gl`。
  - 可插拔 inter-node 后端:**CX7**(ConnectX-7 / IB / RoCE,libibverbs
    RC)与 **EFA**(AWS,libibverbs + efadv SRD),共享同一份 host API
    与同一份 on-GPU kernel,只换 proxy/session。
- **它不是什么 / 不做什么:**
  - **不是通用通信库**(不是 NCCL/RCCL 替代品):只为这 6 个 fused
    pattern 服务,collective 与具体 kernel 绑死。
  - **不做 AMD / 跨厂商**:`sm_90a` 写死,roadmap 才把 Blackwell /
    异构加速器列为 🚧。
  - **不是 autotuning 框架**:tile 形状、SM split 多为编译期常量
    (`SM_COUNT=132`、`CHUNK_BYTES=16 MiB` 等硬编)。
  - **compute 不自研**:MMA/compute 代码改编自 **ThunderKittens**
    (HazyResearch),mKernel 的原创性集中在 `comm/` + `dist/` +
    `operators/` 的融合编排。
- **谁在用:** UCCL 生态自用 + 研究展示(2-node H200 benchmark)。属于
  active-development 的参考实现,不是 production 依赖。

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| `include/operators/<op>/` | 6 个 fused 算子的 host setup + device prelude(config / CTA-role 布局 / globals / entrypoint),kernel body 在 `src/<op>.cu` | 每个算子一个 `*.cuh`(角色共享 helper)+ 一个 `session.cuh`(pybind glue) |
| `include/comm/` | 通信原语总集:`ipc.cuh`(intra-node IPC)、`multimem.cuh`/`vmm.cuh`(NVLink multicast)、`atomic_u32`/`global_u64`(无锁标志) | `comm.cuh` 是 umbrella include |
| `include/comm/internode/` | 跨节点 RDMA 全栈:CPU `proxy`(tight-poll → 链式 RDMA write)、`d2h_fifo`(GPU→CPU 命令队列)、`rdma_transport`/`session`(CX7)+ `*_efa`(EFA)、`session_select.h` 选后端 | "GPU-driven networking, built from scratch" 都在这里 |
| `include/dist/` | 分布式 tensor 抽象:`local_tensor`(单卡)、`distributed_buffer`(intra+inter)、`parallel_buffer`、`tma`/`tma_encode`(直调 `cuTensorMapEncodeTiled`) | 设计契约写在 `distributed_buffer.cuh` 文件头 |
| `include/common/` + `include/memory/` | ThunderKittens 移植:tile/vec 类型(`tk_types_*`)、MMA warpgroup、shared↔register、TMA 搬运 op | compute 层,改编自 TK |
| `bench/` + `python/` | 每个算子的 baseline(NCCL / torch)对比脚本 + 2-node run 脚本;`load_module.py` 动态加载编出来的 `.so` | benchmark 假设 torchrun + 免密 SSH + 可路由 data-plane IP |

## 3. 核心设计理念 (core design ideas)

### 3.1 一个 persistent kernel 同时吃 intra-node + inter-node

README Highlights 原文:"**Multi-GPU + multi-node, in one kernel.**
Handling both intra-node and inter-node GPU-driven communication inside
the same kernel."。传统做法是 collective(NCCL)与 GEMM 分两次 launch、
靠 stream/event 在 kernel 边界 overlap;mKernel 把整条
dispatch→GEMM→combine 写进**单次 `fused()`**,kernel 内部自己跨节点握手
(`dispatch_gemm_glu_combine` 注释:"an in-kernel cross-node handshake
makes the combine self-contained (done before exit)")。换来的是消除
launch 间隙、消除中间 buffer 的 HBM round-trip、让 overlap 粒度细到
tile 而不是 kernel。

### 3.2 CTA 角色专精(SM specialization)= kernel 内的 producer/consumer

README:"**Persistent kernel with SM specialization.** CTAs are assigned
roles, such as compute / intra-comm / inter-send / inter-reduce."。
`gemm_ar.cuh` 文件头把 4 个角色写死:**compute / intra-AR /
inter-send / inter-reduce-and-publish**;`dispatch_gemm_glu_combine` 是
**send / copy / dispatch / compute**。一次 launch 里不同 CTA 走完全不同
的代码路径,通过共享的 ready 标志做 kernel 内 producer→consumer。这把
"通信 CTA 在等网络 RTT" 和 "计算 CTA 在跑 MMA" 在**同一个 kernel、同一
批 SM**上真正并行,而不是靠两个 stream 抢占。

### 3.3 GPU-driven networking,直接写 libibverbs(绕开 NCCL/NVSHMEM)

README:"**GPU-driven networking, built from scratch.** Directly
implement communication over Libibverbs ... for maximal performance."。
机制(`d2h_fifo.cuh` + `proxy.h`):GPU 的 comm CTA 用 `push()` 把定长
`TransferCmd` 写进 host-pinned triggers,**`atomicAdd` 抢 head 槽位(纯
device 内存,不付 PCIe round-trip)**;CPU proxy 线程 tight-poll FIFO,
读到命令就 post 链式 RDMA write(tile 数据 + arrival flag),poll send
CQ,推进 tail 做 backpressure。FIFO 是 UCCL/mscclpp 的
`fifo_device.hpp` 模式。意义:把"谁发起通信"从 host 挪到 GPU,host 只
当一个无脑搬运 WQE 的 proxy,通信能被 GPU 的数据流精确触发到 tile 级。

### 3.4 无锁 dataflow 契约:1 channel = 1 QP = 1 comm CTA

`distributed_buffer.cuh` 文件头把契约写得极清楚,值得整段记住:

> - One channel = one QP = one comm CTA. No CAS in fast path.
> - Inter-node sends are RDMA WRITE_WITH_IMM, imm = tile_id.
> - DMA-BUF zero-copy by default.
> - Send ring per channel is single-producer / single-consumer ... No CAS
>   on producer or consumer index — each side owns its own counter.
> - 128B-aligned arrival flags — no false sharing.
> - Monotonic flag semantics — no inter-iter reset.

这是整个库性能的根:**把并发收敛成一堆 SPSC 关系,从而 fast path 上一
个 CAS 都没有**。tile 到达用 `WRITE_WITH_IMM` 的 immediate 当 tile_id
直接当信号,arrival flag 128B 对齐避免 false sharing、单调递增避免每轮
reset。这套契约比"它有多少 dtype"重要得多。

### 3.5 tile/chunk 粒度的 producer→consumer overlap

README:"**Fine-grained intra-kernel overlapping.** Compute and
communication overlap at tile/chunk granularity."。具体到每个算子:
AG+GEMM "matmul starts before the collective finishes";GEMM+AR "output
tiles are pushed into the reduction tree the instant they're produced,
hiding the AllReduce inside the GEMM tail";MoE dispatch "tokens are
matmul'd as soon as they land, no staging buffer round-trip"。即
overlap 的单位是 **tile / 16 MiB chunk**(`CHUNK_BYTES`),不是整个
tensor——这是它能把通信藏进计算的前提。

### 3.6 epilogue 融合消除中间 HBM round-trip

`dispatch_gemm_glu_combine.cuh`:"gemm1+SwiGLU (fused): ... with SwiGLU
applied in the gemm1 epilogue (**no h1[M,2I] HBM round-trip**)"。MoE FFN
的 `gemm1 → SwiGLU → gemm2` 里,中间激活 `h1` 不落 HBM,直接在 gemm1
epilogue 算完 SwiGLU 喂给 gemm2。配合 3.1 的单 kernel,把"通信省 + 中
间激活省"两类 HBM 流量一起砍掉。

### 3.7 可插拔 inter-node 后端,kernel 不变

README Backends 表 + `session_select.h`:CX7(`-DINTERNODE_BACKEND_IBVERBS`,
libibverbs RC)与 EFA(`-DINTERNODE_BACKEND_EFA`,libibverbs + efadv
SRD)"share the same host-side API and the same on-GPU kernel; only the
proxy / session implementation differs"。把"网络传输差异"隔离在 host
proxy/session 一层,**on-GPU 的 dataflow 与触发逻辑对后端完全无感**。

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| **CTA 角色专精的固定角色表**(compute / intra-comm / inter-send / inter-reduce 一次 launch 内并存) | 通信 CTA 等 RTT 时计算 CTA 不能闲着,需要在同一 kernel 内做 producer/consumer 而非两 stream 抢占 | RocMoE / MonolithMoE super-kernel 的 wave/CU specialization——把 mKernel 这套**显式角色taxonomy**直接搬过来当模板(见 `.cursor/skills/cco-pipeline-overlap/SKILL.md` 的 wave specialization 原则),inter-node 阶段尤其缺一个清晰的 send/reduce 角色划分 | AMD wave=64、无 wgmma;角色间 ready 标志要走 release-acquire,别照抄 NVIDIA 的 fence 语义 |
| **无 CAS 的 SPSC send-ring + 128B 单调 arrival flag**(每端各持自己的 counter,flag 单调不 reset) | 多 CTA / 多 channel 并发写跨设备 buffer 时,CAS 与 false sharing 是 fast-path 杀手 | MonolithMoE 的 XGMI/IPC tile staging:把每条 (producer CTA → consumer CTA) 关系收敛成 SPSC、flag 128B(AMD cacheline 64B,取 128B 仍安全)对齐、单调递增——正对 `knowledge/kernels/memory-access-patterns.md` Q3/Q5 与 cco-overlap 的 release-acquire ready flag | AMD L2/缓存行为与 NVIDIA 不同,单调 flag 的可见性要用 `__hip_atomic_*` system scope 验证;别假设 PCIe-free 的 head 抢占在 AMD 上等价 |
| **GPU-driven RDMA via CPU proxy + D2H FIFO**(GPU `atomicAdd` 抢槽位推 TransferCmd,host proxy 无脑 post WQE) | inter-node MoE dispatch/combine 用 RCCL all2all 时,host 发起 + kernel 边界同步成为瓶颈,无法 tile 级触发 | AMD 集群同样是 libibverbs(IB/RoCE)NIC:这套 proxy+FIFO 模式**与厂商无关**,可作为 RocMoE 跨节点 dispatch 的通信底座,绕开 RCCL 的 launch/同步开销 | 需要 GPU HBM 的 RDMA 注册(AMD 走 DMA-BUF / peermem,与 ROCm + NIC 驱动强耦合);proxy 线程 CPU 亲和与 tight-poll 烧核需 SLURM 端配合 |
| **WRITE_WITH_IMM,imm = tile_id 当到达信号** | tile 到达不需要额外一次 flag write/poll,immediate 字段天然携带"哪个 tile 到了" | RocMoE inter-node combine 的到达通知;能省掉一次独立的 flag RDMA,直接在 CQ 上读 imm 分发 | AMD 侧 RoCE/IB verbs 同样支持 IMM,但 efadv(EFA)语义不同;需确认目标 NIC 的 IMM/CQ 路径 |
| **epilogue 内做 activation,中间激活不落 HBM**(SwiGLU 融进 gemm1 epilogue,省 `h1[M,2I]` round-trip) | MoE FFN 的中间激活是除通信外第二大 HBM 流量源 | RocMoE 融合 MoE FFN super-kernel:gemm1→SwiGLU→gemm2 在寄存器/LDS 内接力,对照我们 `knowledge/kernels/fp8-expert-gemm.md` 的 grouped GEMM 结论,epilogue 融合是下一步省带宽的明确方向 | gfx942/gfx950 无 TMA,中间 tile 的 shared→register 接力要手排 LDS swizzle;FP8 量化点位置会改变 epilogue 融合可行性 |
| **传输后端可插拔、on-GPU kernel 不变**(host proxy/session 隔离 CX7 vs EFA) | 同一份 fused kernel 要在不同 NIC/网络上跑,不想为每种网络改 device 代码 | 我们若做 AMD 跨节点 super-kernel,可同样把 RoCE-v2 vs IB vs 不同 NIC 隔离在 host session 层,device 侧只认 "push 命令 + 等 flag" 抽象 | 抽象边界要划在"命令 ABI"上;若 device 侧偷偷依赖某后端的 inline/SGE 上限就漏抽象了(mKernel 用 `max_send_sge` 显式参数化,可借鉴) |
| **`dist::` 层结构化复用 compute 库 tile 元数据,但不继承其 GL**(复用 `kittens::st/sv` 当 POD metadata,自己建 descriptor + 直调 `cuTensorMapEncodeTiled`) | 想借 compute 库(TK / CK / ck_tile)的 tile 抽象,又不想被它的 global-layout 类型绑架到单卡视角 | 我们在 super-kernel 里复用 CK/ck_tile 的 warp-tile 类型时,可学这招:只取 tile 形状/swizzle 这类 POD 元数据,分布式 descriptor 自己持有,避免把单卡 layout 概念渗进多卡代码 | CK 的 tile 类型耦合度比 TK 高,"只取 POD" 未必干净;需评估 ck_tile 的哪些类型是纯 metadata |

## 5. 与生态的关系 (ecosystem position)

```
mKernel (operators: fused collective+compute, persistent megakernel)
   ├── compute   → ThunderKittens (改编;tile/MMA/TMA, sm_90a wgmma)
   ├── intra-node→ CUDA IPC + NVLink multimem / VMM multicast
   └── inter-node→ libibverbs (CX7 RC | EFA SRD) + CPU proxy + D2H FIFO
                   (绕开 NCCL / NVSHMEM)
```

mKernel 坐在"**compute-comm overlap megakernel**"这一格,横向对标:
**NVSHMEM-based 通信**(它显式不用)、**DeepEP**(MoE dispatch/combine
专用,但仍是独立 kernel + host 编排)、**Flux / FlashOverlap / CoCoNet**
(comm-compute overlap,多为 tile 级但通常 collective 仍走 NCCL)。
mKernel 的差异点是**把通信栈也自研进 kernel**,而不只是 overlap 既有
NCCL。对我们而言,它的同类位置在 AMD 侧由 RocMoE / MonolithMoE
super-kernel 占据:都是"persistent + 角色专精 + tile 级 overlap",但我
们底座是 **XGMI/IPC + RCCL 或自研 RDMA**,compute 用 **CK/ck_tile/MFMA**
而非 TK/wgmma。本库与 `knowledge/libraries/` 里已蒸馏的 CK/AITER/turbo
**不重叠**:那几个是 AMD 单卡 kernel 库,mKernel 是 NVIDIA 的多卡融合
编排参考。配套阅读:`knowledge/kernels/cco-overlap.md`、
`knowledge/kernels/memory-access-patterns.md`、
`.cursor/skills/cco-pipeline-overlap/SKILL.md`、
`.cursor/skills/rocmoe-dev-loop/SKILL.md`。

## 6. 进一步阅读 / TODO

入口文件(≤ 5,各一句"为什么读"):

- `README.md` —— 4 条 Highlights 把全部设计取向讲清,先读这个。
- `include/dist/distributed_buffer.cuh`(文件头) —— 那段 6 条 design
  contract 是整库无锁 dataflow 的精华,想抄 SPSC + flag 语义必读。
- `include/comm/internode/d2h_fifo.cuh` + `proxy.h`(文件头) ——
  GPU-driven networking 的 GPU 推命令 / CPU proxy 两端协议。
- `include/operators/gemm_ar/gemm_ar.cuh`(文件头) —— 4 个 CTA 角色
  的最干净样本(compute / intra-AR / inter-send / inter-reduce)。
- `include/operators/dispatch_gemm_glu_combine/dispatch_gemm_glu_combine.cuh`
  (文件头) —— 完整 MoE 层一个 kernel,epilogue 融合 + in-kernel
  cross-node handshake 的总装样本。

待验证 / 待沉淀(**不在本文,做了归别处**):

- mKernel 的 SPSC + 单调 flag 契约在 **AMD gfx942/gfx950** 上的等价实
  现(`__hip_atomic_*` scope、cacheline、XGMI 可见性),实测后归
  `notes/RocMoE/` + 必要时补进 `knowledge/kernels/cco-overlap.md`。
- proxy + D2H FIFO 模式在 AMD + ROCm + DMA-BUF/peermem 上的可行性
  (GPU HBM RDMA 注册路径),归 `notes/` 实测。
- TMA(`cuTensorMapEncodeTiled`)对应到 gfx950 的 async direct-to-LDS
  (`buffer_load_lds`)缺口有多大,归 `.cursor/skills/mi355_hardware_aware/`
  范畴评估。
- 与 DeepEP / Flux / FlashOverlap 的逐项对比(是否自研通信、overlap
  粒度、是否单 kernel),若要做,走 `read-paper` / 单独 `notes/`,不塞
  进本库蒸馏。
```

