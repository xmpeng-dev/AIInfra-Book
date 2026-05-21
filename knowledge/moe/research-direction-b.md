# 方向 B 深度分析：MoE-Native IR + 跨层通信调度编译器

> **定位：** 对 README_research.md 方向 B 的完整展开  
> **关联：** rocflow/ir_design.md（工程实现）· README_research.md（研究背景）  
> **更新：** 2026-03-09

---

## 目录

1. [问题背景：为什么现有方案不够](#1-问题背景)
2. [核心洞察：通信调度的三个层次空白](#2-核心洞察)
3. [IR 设计：RFGraph 的形式化定义](#3-ir-设计)
4. [四个核心编译 Pass](#4-四个核心编译-pass)
5. [与 torch.compile 的协作模型](#5-与-torchcompile-的协作模型)
6. [动态路由的静态化策略](#6-动态路由的静态化策略)
7. [论文贡献点梳理](#7-论文贡献点梳理)
8. [实验设计方案](#8-实验设计方案)
9. [实现路线图](#9-实现路线图)
10. [与相关工作的差异定位](#10-与相关工作的差异定位)

---

## 1. 问题背景

### 1.1 MoE 训练的调度本质

一个完整的 MoE 训练 Step，从调度视角来看，是一组有依赖关系的异构任务的执行问题：

```
任务类型：
  ComputeTask  → GPU GEMM、激活函数、LayerNorm
  CommTask     → All-to-All、AllReduce、AllGather、ReduceScatter
  MemTask      → Tensor 分配、释放、拷贝

依赖关系：
  数据依赖（DataDep）：Task B 需要 Task A 的输出数据
  顺序依赖（OrderDep）：Task B 必须在 Task A 之后执行（同一 Stream 内）
  事件依赖（EventDep）：Task B 等待 Task A 的完成事件（跨 Stream）

调度目标：
  最小化总执行时间
  = 找到一个任务排列，使所有任务的关键路径（Critical Path）最短
  同时满足：内存约束、带宽约束、硬件路径约束
```

**这是一个经典的调度优化问题（Schedule Optimization）。**

但当前所有 MoE 训练框架都没有把它当做调度问题来解——他们用 Python 函数调用的顺序作为「调度方案」，相当于用手动排序替代了最优调度。

### 1.2 当前方案的本质缺陷

```
框架对比（调度能力）：

                  全局视野    编译期分析   通信一等公民   硬件路径感知
nn.Module eager     ✗           ✗              ✗              ✗
torch.compile       部分         ✓              ✗（graph break）✗
FlowMoE             ✓           ✗（运行时）     ✓              ✗
Megatron PP         部分         部分            ✗              ✗
RFGraph（目标）      ✓           ✓              ✓              ✓

→ 没有任何现有系统同时满足这四个条件
```

### 1.3 代价的量化

```
当前框架由于调度次优付出的代价（估算）：

① 通信阻塞时间（A2A 完全串行）：
   A2A 延迟 ≈ 15ms per layer，64 层，Forward+Backward
   → 额外等待时间 ≈ 15ms × 64 × 2 = 1920ms per step
   → 占总 step 时间 50~60%

② AllReduce 串行等待（梯度同步在反向完成后才开始）：
   AllReduce 延迟 ≈ 8ms per layer
   → 额外等待 ≈ 8ms × 64 = 512ms per step
   → 若与反向传播 overlap，可节省 400ms+

③ Python scheduling overhead（FlowMoE 运行时 DAG）：
   每 step 重新分析 DAG ≈ 1~5ms
   → 可通过编译期静态分析消除

总可节省时间：~2400ms out of ~4000ms step time = ~60% 理论上限
```

---

## 1.5 为什么叫 "MoE-Native"

RFGraph 不是一个通用调度 IR 碰巧用在 MoE 上，而是从 MoE 的计算-通信模式出发**反向设计**的。它的节点类型、编译 Pass、内存复用策略都编码了 MoE 特有的结构不变量——这些优化对 Dense 模型无意义，但对 MoE 是关键性能瓶颈。

```
"MoE-Native" 具体体现在五个层面：

① 节点类型系统是 MoE 语义的
   ComputeNode.op_type ∈ {EXPERT_GEMM, ATTN_GEMM, GATE, ...}
   CommNode.comm_op    ∈ {A2A_DISPATCH, A2A_GATHER, RELAYOUT, ...}
   → A2A_DISPATCH 和 RELAYOUT（FSEP 重布局）只在 MoE 中存在
   → IR 知道这些操作的语义，才能做 MoE 特有的优化

② Comm Hoisting（Pass 1）利用了 MoE 的结构不变量
   Gate 和 Attention 共享输入（x_norm），但互相不依赖
   → Gate 通常先完成，A2A 只依赖 Gate，不依赖 Attention
   → A2A 可以在 Attention 还在跑的时候就发出去
   → 这个优化只对 MoE 有意义，Dense 模型没有 Gate 也没有 A2A

③ 硬件路径分配（Pass 3）利用了 MoE 的通信模式
   MoE 天然产生两类通信：节点内（TP AllReduce）+ 跨节点（A2A）
   → 正好对应 XGMI 和 RDMA 两条独立物理路径
   → 这种"双类通信并发"的机会来源于 MoE 的 EP+TP 架构特性

④ 内存 Pass（Pass 4）有 MoE 特有的 Buffer 复用
   A2A Dispatch 输出 → Expert GEMM → A2A Gather
   Forward 中 A2A_D 的输出 buffer 在 Expert GEMM 后可复用为 A2A_G 的输出 buffer
   → ExpertSlotTensor 复用模式只在 MoE 的 A2A→Expert→A2A 流水线中存在

⑤ 动态路由的静态化（Section 6）完全是 MoE 问题
   每个 Expert 收到的 token 数不固定 → 动态 shape → compile 困难
   → Padded Static Dispatch 专为 MoE Top-K 路由设计
   → Dense 模型 shape 完全静态，不需要这个设计
```

```
对比总结：

设计维度           通用 IR（torch.compile / XLA）     MoE-Native IR（RFGraph）
──────────────────────────────────────────────────────────────────────────────
All-to-All         graph break，看不见                CommNode(A2A_DISPATCH) 一等公民
Expert 路由        动态 shape → recompile 或 break    Padded Static Dispatch 专门处理
Expert GEMM        当普通 GEMM 处理                   ComputeNode(EXPERT_GEMM) 独立语义
内存复用           通用 liveness analysis              ExpertSlotTensor：MoE A2A 特有复用
跨层 overlap       不感知 MoE 层结构                  知道 Attn→Gate→A2A→Expert 固定流水线
硬件路径           不区分通信类型                      XGMI(节点内) vs RDMA(跨节点) 分流
```

---

## 2. 核心洞察

### 2.1 三个层次的调度空白

```
调度空白全景图：

粒度        现有最优方案        覆盖范围          空白
──────────────────────────────────────────────────────────
跨 Step      无                 无               ✗ 全部空白
跨 Block     FlowMoE           chunk 级 overlap  ✗ 无编译期版本
Block 内     Comet             tile 级 overlap   ✗ 仅 CUDA，无 HIP
                                                 ✗ 无跨通信 fusion
Kernel 内    hipBLASLt 自动调优 单个 GEMM        ✗ 不感知通信
```

**关键洞察一：跨 Block 的调度可以在编译期完成**

```
FlowMoE 的运行时 DAG 分析每步都要重跑：
  分析 Block_i 和 Block_{i+1} 的依赖 → O(N_layers²) 分析
  → 每 step 1~5ms overhead

但 MoE 模型的结构是静态的（同构 Transformer Block 重复堆叠）：
  Block_i 和 Block_{i+1} 的依赖关系在编译期就确定了！
  → 只需 trace 一次，生成静态 Scheduled Execution Plan
  → 运行时 overhead 降为 0
```

**关键洞察二：通信操作的「最早可发起时间」可以静态分析**

```
对于 Block_i 的 A2A Dispatch：
  输入依赖：Gate 计算的输出（routing decisions）
  Gate 何时完成？→ 静态可知（Gate 是一个固定大小的 GEMM，时间可估算）

  → A2A Dispatch 的最早发起时间 = Gate GEMM 完成时间
  → 不需要等 Attention GEMM 完成（没有数据依赖！）

  传统方案：Attention → Gate → A2A（串行）
  最优方案：Attention 执行期间，Gate 完成就立刻发 A2A
            → A2A 的延迟被 Attention 的剩余时间覆盖
```

**关键洞察三：反向传播的 dW 和 dX 是可以并行的独立计算**

```
Expert FFN 反向：
  dX = dOutput @ W_down.T            # 传给上一层的梯度（关键路径）
  dW = partial_act.T @ dOutput       # 本层参数梯度（非关键路径）

  dX 和 dW 只有 dOutput 这一个共同输入，两者互相独立
  → 可以在两个 Stream 上同时计算

  dX 需要传给上一层 Block（通过 A2A_bwd）
  dW 需要 AllReduce（跨 DP 组同步）

  → dW 的 AllReduce 与 Block_{i-1} 的全部反向传播都可以 overlap！
  → 所有梯度通信被完全隐藏（这是目前所有论文都没做到的）
```

### 2.2 IR 设计的核心需求矩阵

```
需求                                  为什么重要
──────────────────────────────────────────────────────────────────
通信节点是一等公民（非 graph break）   支持跨通信的全局优化
跨 Module 图展平                      发现跨 Block 的 overlap 机会
硬件路径标注（XGMI vs RDMA）          双通道真并发（AMD 优势）
Tensor lifetime 精确追踪              激活内存精确复用
编译期静态化                          消除运行时 scheduling overhead
与 torch.compile 协作                 不重复造轮子（Kernel codegen）
支持动态 shape（MoE 路由）            实际可用性
```

---

## 3. IR 设计

### 3.1 RFGraph 的图结构定义

```
RFGraph = (V, E_data, E_overlap, E_stream)

节点集 V：
  V = ComputeNode ∪ CommNode ∪ MemNode ∪ SyncNode

边集 E：
  E_data    ：数据依赖边，A→B 表示 B 需要 A 的输出 Tensor
  E_overlap ：overlap 约束边，A‖B 表示 A 和 B 可以并发执行
  E_stream  ：stream 内顺序边，A→B 表示 A、B 在同一 stream 上顺序执行

关键性质：
  E_data 决定正确性（必须满足）
  E_overlap 决定性能（尽量利用）
  E_stream 决定硬件执行顺序（由调度器填充）
```

### 3.2 节点的关键属性

```
ComputeNode：
  op_type      ∈ {EXPERT_GEMM, ATTN_GEMM, GATE, LAYERNORM, ACTIVATION, ...}
  tile_size    : int              # Tile 粒度（for Tile-level overlap）
  hw_unit      : HWUnit           # CU 分配（for Wavefront Specialization）
  compiled_fn  : Optional[...]    # torch.compile 生成的 Kernel（运行时填充）

CommNode：
  comm_op      ∈ {A2A_DISPATCH, A2A_GATHER, ALL_REDUCE, ALL_GATHER,
                  REDUCE_SCATTER, RING_ATTN, RELAYOUT}
  hw_path      ∈ {XGMI, RDMA, LOCAL}    # 硬件路径标注（AMD 特有）
  process_group: ProcessGroup
  chunk_id     : Optional[int]           # 若做 chunk 切分，所属 chunk 编号
  is_async     : bool = True

MemNode：
  mem_op       ∈ {ALLOC, FREE, COPY, SWAP_OUT, SWAP_IN}
  tensor_ref   : RFTensor
  lifetime     : Tuple[int, int]         # [first_use_node_id, last_use_node_id]

SyncNode：
  wait_for     : List[CommNode]          # 等待哪些通信完成
  on_stream    : int                     # 在哪个 stream 上插入 wait
```

### 3.3 一个 MoE Block 的 RFGraph 示例

```
Block_i 的 RFGraph（Forward）：

节点：
  n1: ComputeNode(LAYERNORM,   in=[x_i],           out=[x_norm])
  n2: ComputeNode(ATTN_GEMM,   in=[x_norm],         out=[attn_out])
  n3: SyncNode(               wait=[prev_a2a_g])   # 等上一层 A2A_Gather 完成
  n4: ComputeNode(GATE,        in=[x_norm],         out=[routing])
  n5: CommNode(A2A_DISPATCH,   in=[x_norm, routing],out=[dispatched],  hw=RDMA)
  n6: ComputeNode(EXPERT_GEMM, in=[dispatched],     out=[expert_out])
  n7: CommNode(A2A_GATHER,     in=[expert_out],     out=[gathered],    hw=RDMA)
  n8: CommNode(TP_ALL_REDUCE,  in=[attn_out],       out=[attn_sync],   hw=XGMI)
  n9: ComputeNode(ELEMENTWISE, in=[attn_sync, gathered, x_i], out=[x_out])

数据依赖边（E_data）：
  n1→n2, n1→n4, n4→n5, n5→n6, n6→n7, n2→n8, n7→n9, n8→n9

Overlap 约束边（E_overlap）：
  n2 ‖ n4     # Attn GEMM 和 Gate 无依赖，可并发（n4 依赖 n1，n2 也依赖 n1）
  n2 ‖ n5     # Gate 完成后立刻发 A2A，Attn GEMM 还在执行
  n6 ‖ n8     # Expert GEMM 和 TP AllReduce 走不同路径，可并发
  n7 ‖ n8     # A2A Gather 和 TP AllReduce 也可并发（不同硬件路径）

硬件路径分配：
  n5(A2A_DISPATCH) → RDMA stream_0
  n7(A2A_GATHER)   → RDMA stream_1（与 n5 不同 stream，可并发）
  n8(TP_ALL_REDUCE)→ XGMI stream_2（完全独立路径）

可视化（时间轴）：
  XGMI stream:  ────────────[TP_AllReduce_i]──────────────────────────
  RDMA stream0: ─────────────────[A2A_D_i]───────────────────────────
  RDMA stream1: ────────────────────────────────[A2A_G_i]─────────────
  Compute:      [Norm][Attn_GEMM][Gate][Expert_GEMM][Merge]
                       ↕ overlap  ↕ overlap    ↕ overlap
```

---

## 4. 四个核心编译 Pass

### 4.1 Pass 1：Comm Hoisting（通信提前）

**目标：找到每个 CommNode 的最早可发起时刻，并将其「前移」**

```
算法：Earliest Start Time Analysis

输入：RFGraph G
输出：每个 CommNode 的 earliest_start_time[n]

步骤：
  1. 拓扑排序 G，得到节点序列 [n_1, n_2, ..., n_k]
  2. 对每个节点 n，计算：
       earliest_start_time[n] = max(finish_time[pred] for pred in n.deps)
     其中：
       finish_time[ComputeNode] = earliest_start_time + estimated_compute_time
       finish_time[CommNode]    = earliest_start_time + estimated_comm_time
  3. 对每个 CommNode，找到其最早可发起的位置
     如果当前位置比最早位置「晚」→ 前移（Hoist）

示例（Block_i 的 A2A Dispatch）：
  deps = {Gate_i}
  Gate_i 的完成时间 t_gate

  传统位置：在 Attn_GEMM 完成后（t_attn > t_gate）才发 A2A
  Hoisted 位置：Gate_i 完成时（t_gate）立刻发 A2A

  节省时间 = t_attn - t_gate（Attn GEMM 的剩余时间）
  → A2A 延迟被 Attn GEMM 覆盖
```

```
Comm Hoisting 的效果图：

Before Hoisting：
  时间轴: ─────────────────────────────────────────────────→
  Compute: [Norm][──Attn GEMM──][Gate][──Expert GEMM──]
  Comm:                              [────A2A Disp────][──A2A Gather──]

After Hoisting：
  时间轴: ─────────────────────────────────────────────────→
  Compute: [Norm][──Attn GEMM──][Gate][──Expert GEMM──]
  Comm:                   [────A2A Disp────][──A2A Gather──]
                          ↑ Gate 完成后立刻发，不等 Attn
  节省：A2A Dispatch 被 Attn 剩余时间覆盖（约 3~8ms）
```

### 4.2 Pass 2：Overlap Maximization（重叠最大化）

**目标：找到所有可并发的 (CommNode, ComputeNode) 对，分配到不同 Stream**

```
算法：Conflict-Free Concurrency Analysis

定义：两个节点 A 和 B 可以并发，当且仅当：
  ① A 和 B 之间没有数据依赖（不在同一条依赖链上）
  ② A 和 B 使用不同的硬件资源（不同 Stream）
  ③ A 和 B 不竞争同一块内存（无写写冲突）

算法步骤：
  1. 构建节点的「冲突图」C：若 A、B 有依赖或资源竞争，则 C 中连边
  2. 在冲突图的补图中，找最大独立集（≈ 最多可并发节点集合）
  3. 对可并发节点对，分配到不同 Stream
  4. 插入最少必要的同步点（SyncNode）

Stream 分配策略（AMD 双通道优先）：
  stream_0: Compute（GEMM、激活函数等）
  stream_1: XGMI 通信（TP AllReduce、节点内 ReduceScatter）
  stream_2: RDMA 通信（A2A Dispatch、A2A Gather、跨节点 AllReduce）
  stream_3: Memory 操作（FSEP Re-layout 的异步参数搬迁）

  AMD 优势：XGMI 和 RDMA 是完全独立的物理路径
  → stream_1 和 stream_2 真正并发，无带宽竞争
  → NVIDIA NVLink + IB 共享交换机端口，有竞争
```

```
Pass 2 的输出（以 Block_i 为例）：

并发组分配：
  Stream_0（Compute）: [Norm_i] → [Attn_GEMM_i] → [Gate_i] → [Expert_GEMM_i]
  Stream_1（XGMI）:               [TP_AllReduce_i]
  Stream_2（RDMA）:                          [A2A_D_i]     [A2A_G_i]

同步点：
  SyncNode_1: Stream_0 wait Stream_2(A2A_D_i 完成) before Expert_GEMM_i
  SyncNode_2: Stream_0 wait Stream_1(TP_AR_i 完成) before Merge_i
  SyncNode_3: Stream_0 wait Stream_2(A2A_G_i 完成) before Merge_i
```

### 4.3 Pass 3：Hardware Path Assignment（硬件路径分配）

**目标：为每个 CommNode 分配最优的硬件传输路径**

```
决策规则（基于通信参与的进程组拓扑）：

输入：CommNode n，其 process_group 包含 {GPU_0, GPU_1, ..., GPU_k}

规则：
  IF 所有 GPU 在同一物理节点（same-node）:
    n.hw_path = XGMI        # AMD Infinity Fabric，近内存带宽
    n.stream  = XGMI_stream

  ELIF 所有 GPU 在同一 Rack（同交换机）：
    n.hw_path = RDMA_LOCAL  # InfiniBand / RoCE，较低延迟
    n.stream  = RDMA_stream

  ELSE：
    n.hw_path = RDMA_REMOTE # 跨 Rack，仅在必要时
    n.stream  = RDMA_stream

特殊情况：
  TP AllReduce → 通常在同节点（TP=8 正好一个 MI300X 节点）→ XGMI
  A2A Dispatch/Gather → 跨节点（EP=64 跨多节点）→ RDMA
  FSEP ReduceScatter → 同节点（Expert 分片在同节点内）→ XGMI ← AMD 独有加速！
  DP AllReduce（梯度）→ 跨节点 → RDMA
```

```
路径分配效果（8-GPU 节点，EP=8）：

通信操作                   路径      带宽        延迟
──────────────────────────────────────────────────────
TP AllReduce（8 GPU）       XGMI     896 GB/s    ~3μs
FSEP ReduceScatter（8 GPU） XGMI     896 GB/s    ~3μs   ← 关键！
A2A Dispatch（跨节点）      RDMA     400 Gbps    ~15μs
A2A Gather（跨节点）        RDMA     400 Gbps    ~15μs
DP AllReduce（梯度）        RDMA     400 Gbps    ~15μs

由于 XGMI 和 RDMA 完全独立：
  XGMI 操作和 RDMA 操作可以真正并发，不竞争带宽
  → TP AllReduce + A2A Dispatch 同时进行
  → FSEP ReduceScatter + A2A Gather 同时进行
```

### 4.4 Pass 4：Memory Lifetime Minimization（内存生命周期最小化）

**目标：精确追踪每个 Tensor 的使用范围，在最后一次使用后立即释放**

```
算法：Tensor Liveness Analysis（借鉴编译器的活跃变量分析）

对每个 RFTensor t，计算：
  first_use[t] = min(node_id for node in G if t in node.tensor_in)
  last_use[t]  = max(node_id for node in G if t in node.tensor_in)
  lifetime[t]  = [first_use[t], last_use[t]]

内存优化策略：
  ① 立即释放：在 last_use[t] 对应的节点执行完后，立即 FREE(t)
  ② 内存复用：两个 lifetime 不重叠的 Tensor 可以共享同一块内存
  ③ 激活 Checkpoint：对 lifetime 长的大 Tensor，考虑 Recompute

MoE 特有的优化（ExpertSlotTensor）：
  Expert GEMM 的输入（dispatched tokens）和输出（expert outputs）
  在反向传播前后的角色互换 → 可以共享同一块 Buffer（省 50% 激活内存）

  Forward:  Buffer_A ← A2A Dispatch 的输出（输入 Expert GEMM）
  Expert GEMM 后：Buffer_A 不再需要（已经用完）
  → Buffer_A 被复用为 A2A Gather 的输出缓冲

  Backward：Buffer_B ← A2A Gather_bwd 的输出（梯度）
  → Buffer_B 复用 Forward 中某个已释放的 Tensor 的内存
```

---

## 5. 与 torch.compile 的协作模型

### 5.1 分工设计

```
两者的核心分工：

RFGraph（调度层）负责：
  ① 操作调度顺序（什么时候执行哪个操作）
  ② 通信操作的 Stream 分配和 hw_path 选择
  ③ 内存生命周期管理
  ④ 跨 Module 的全局 overlap 分析

torch.compile（Kernel 生成层）负责：
  ① 单个 ComputeNode 的 Kernel 代码生成（Inductor → HIP）
  ② 相邻计算操作的算子融合（LayerNorm + Attention 等）
  ③ hipBLASLt autotuning（Expert GEMM 的矩阵乘优化）
  ④ 内存访问模式优化（coalescing、cache 利用）

两者不重叠：
  torch.compile 不能跨 A2A 做 fusion → RFGraph 填补这个空白
  RFGraph 不做 Kernel codegen → torch.compile 负责这部分
```

### 5.2 集成接口

```python
# RFGraph 与 torch.compile 的集成方式

class ComputeNode(RFNode):
    def __init__(self, ...):
        self.eager_fn = None        # 原始 PyTorch 函数
        self.compiled_fn = None     # torch.compile 编译后的版本

    def compile(self, mode="max-autotune"):
        """
        对本 ComputeNode 调用 torch.compile
        RFGraph 提供形状信息，compile 生成最优 Kernel
        """
        # 提取本节点对应的 subgraph
        subgraph = self._extract_subgraph()

        # 用 torch.compile 编译（inductor 后端 → HIP codegen）
        self.compiled_fn = torch.compile(
            subgraph,
            backend="inductor",
            options={
                "rocm": True,           # 启用 ROCm 后端
                "max_autotune": True,   # hipBLASLt autotuning
            }
        )

class RFGraph:
    def lower(self):
        """
        将 RFGraph 降低（Lower）为可执行的 Scheduled Plan：
          1. 对所有 ComputeNode 调用 torch.compile
          2. 对所有 CommNode 绑定 RCCL 调用 + stream
          3. 生成 stream.wait_event 同步点
          4. 返回一个 Python callable（接近 CUDA Graph 的 overhead）
        """
        for node in self.compute_nodes:
            node.compile()

        return ScheduledExecutionPlan(self)
```

### 5.3 通信操作的 Graph Break 解决方案

```
核心问题：torch.compile 在遇到 dist.all_to_all 时产生 graph break
  → MoE 层被切成多段，无法做跨 A2A 的算子融合

解决方案：Custom Lowering Rule + CUDA Graph Placeholder

步骤：
  ① 在 RFGraph 中，CommNode 持有一个「占位算子」(CommPlaceholder)
     CommPlaceholder 是一个 torch.autograd.Function，语义上是 identity

  ② torch.compile 可以正常 trace 含有 CommPlaceholder 的 graph
     （因为 CommPlaceholder 在 trace 阶段不产生 graph break）

  ③ 在 CUDA Graph capture 阶段，CommPlaceholder 被替换为实际的 RCCL 调用
     CUDA Graph 可以包含 RCCL（PyTorch 2.x 已支持）

  ④ 最终生成的 CUDA Graph 包含：
     [Kernel_before_A2A] → [RCCL_A2A] → [Kernel_after_A2A]
     整个 MoE 层是一个完整的 CUDA Graph，无 Python overhead

效果：
  vs eager MoE：消除所有 Python dispatch overhead（每 step ~5ms）
  vs torch.compile（有 graph break）：消除 graph break 的额外启动开销
  总提升：+15~25% 吞吐（overhead 消除）+ 更好的 Kernel fusion 机会
```

---

## 6. 动态路由的静态化策略

### 6.1 问题本质

```
MoE 路由的动态性来源：
  top_k_indices = argtopk(gate_logits)   # 每次 batch 结果不同
  → 每个 Expert 收到的 token 数 T_e 是动态的
  → torch.compile 遇到动态 shape → 要么 recompile，要么 graph break

传统解法：
  capacity_factor = 1.25（允许每个 Expert 收到最多 1.25x 平均 token）
  超过上限的 token 被 drop（损失精度）
  → 换来固定 shape，可以 compile

RFGraph 的目标：既保持固定 shape（可 compile），又不 drop token
```

### 6.2 Padded Static Dispatch 方案

```
设计：

  max_tokens_per_expert = capacity（固定值，如 T × K / N_experts × 1.2）

  dispatched_padded[N_experts, capacity, H]   ← 固定形状，可以 compile
  valid_mask[N_experts, capacity]             ← 标记哪些是真实 token

  填充规则：
    真实 token 填入前 T_e 个位置
    剩余位置用 0 填充（或最后一个真实 token 的副本）

  Expert GEMM：
    output_padded = expert_gemm(dispatched_padded)   # 固定 shape
    output_valid  = output_padded * valid_mask        # 过滤无效输出

  内存代价：
    padding 浪费 ≈ (capacity - avg_T_e) / capacity × 100%
    典型值：capacity = 1.2 × avg_T_e → 浪费约 16%
    但换来：完整的 compile 优化 + 无 token drop

符号化形状追踪（备选方案）：
  torch.compile 的 dynamic=True 模式支持 symbolic shape
  RFGraph 可以标注：
    dispatched.shape = [N_experts, symbolic_T_e, H]
    symbolic_T_e 在 [1, capacity] 范围内
  → compile 生成「形状参数化」的 Kernel，无需 recompile
  → 但生成的 Kernel 不如固定 shape 优化程度高
  → 推荐：训练早期用 symbolic（灵活），稳定后切 padded static（性能）
```

---

## 7. 论文贡献点梳理

### 7.1 理论贡献

```
贡献 T1：MoE 通信调度问题的形式化
  将 MoE 训练的调度问题形式化为：
    有向无环图（RFGraph）上的加权关键路径最小化问题
  证明：最优调度的理论下界（通信和计算完全重叠时的理想吞吐）
  分析：现有框架与理论下界的 gap（量化了优化空间）

贡献 T2：通信提前（Comm Hoisting）的正确性证明
  定理：在满足数据依赖约束的前提下，
        将 CommNode 尽量前移不影响计算结果的正确性
  证明：基于数据流分析（Data Flow Analysis）中的 Liveness Analysis

贡献 T3：AMD 双通道并发的带宽分析
  证明：在独立 XGMI 和 RDMA 路径上，
        两类通信操作的总有效带宽 = XGMI_BW + RDMA_BW（超线性叠加）
  对比：NVIDIA 共享交换机端口场景下，两类通信有带宽竞争
```

### 7.2 系统贡献

```
贡献 S1：RFGraph IR 的设计与实现
  - 节点类型系统（ComputeNode / CommNode / MemNode / SyncNode）
  - OverlapEdge 语义（超越传统数据依赖边）
  - 跨 Module 展平的 Tracing 机制（基于 torch.fx 扩展）

贡献 S2：四个编译 Pass 的实现
  - Comm Hoisting：O(V+E) 的拓扑排序算法
  - Overlap Maximization：冲突图 + 最大独立集近似算法
  - Hardware Path Assignment：基于进程组拓扑的决策树
  - Memory Lifetime Minimization：活跃变量分析 + Buffer 复用

贡献 S3：与 torch.compile 的深度集成
  - CommPlaceholder 机制（打破 graph break）
  - CUDA Graph 包含 RCCL 调用
  - 动态路由的 Padded Static Dispatch 方案

贡献 S4：在 AMD MI300X 上的首次实现
  - XGMI 路径的 RCCL 调用绑定
  - HIP Stream 管理
  - AMD MFMA 指令优化的 Expert GEMM
```

### 7.3 实验贡献

```
贡献 E1：消融实验（每个 Pass 的独立贡献）
  关掉 Pass 1（不做 Hoisting）→ 测 A2A 暴露延迟
  关掉 Pass 2（不做 Overlap）→ 测通信串行代价
  关掉 Pass 3（不做路径分配）→ 测带宽竞争损失
  关掉 Pass 4（不做内存优化）→ 测内存峰值

贡献 E2：与现有最优系统的端到端对比
  基线：Megatron-LM on H100 / FlowMoE on H100
  本系统：RFGraph on MI300X
  指标：MFU / 通信-计算重叠率 / 端到端吞吐 / 内存峰值

贡献 E3：规模扩展性（Scalability）
  64 GPU → 256 GPU → 1024 GPU
  关键问题：随规模增大，调度 overhead 是否可控？
  结论（预期）：静态编译使 overhead 为 O(1)（与规模无关）
```

---

## 8. 实验设计方案

### 8.1 实验配置

```
硬件环境：
  系统 A：8 × AMD MI300X（1 节点，验证 XGMI 路径）
  系统 B：64 × AMD MI300X（8 节点，验证 RDMA + XGMI 双路径）
  对比系统：8 × NVIDIA H100（1 节点，作为基线）

软件环境：
  ROCm 6.x / RCCL / hipBLASLt
  PyTorch 2.x（支持 CUDA Graph + RCCL）
  torch.compile with Inductor-ROCm backend

模型配置：
  Mixtral 8x7B：标准 MoE benchmark，有公开基线数据
  Mixtral 8x22B：更大规模，负载不均衡更明显
  DSv3-scale（1/4）：256 expert、Top-4 路由，最接近生产场景
```

### 8.2 核心 Benchmark

```
Benchmark 1：通信-计算重叠率
  测量方式：GPU profiling（ROCm Omniperf）
  测量对象：A2A Dispatch / Gather 与 Expert GEMM 的并发时间比例
  对比：eager / torch.compile / FlowMoE / RFGraph
  期望：RFGraph 达到 85~92%（当前最优 FlowMoE 约 68%）

Benchmark 2：端到端训练吞吐（MFU）
  测量方式：tokens/sec → MFU = actual_FLOP / peak_FLOP
  测量对象：完整训练 100 步，排除前 10 步 warmup
  对比：Megatron-LM on H100 vs RFGraph on MI300X
  期望：MI300X 的硬件算力优势（192GB HBM3e）通过更好的调度得以发挥

Benchmark 3：调度 Overhead
  测量方式：每 Step 的 Python 调度时间（cProfile）
  对比：FlowMoE（运行时 DAG）vs RFGraph（编译期静态）
  期望：RFGraph 的运行时 overhead < 0.1ms（vs FlowMoE 的 1~5ms）

Benchmark 4：内存峰值
  测量方式：torch.cuda.max_memory_allocated()
  对比：eager / MemFine / RFGraph（含 Pass 4）
  期望：RFGraph 内存峰值 ≈ MemFine（-40~50% vs eager）
```

### 8.3 消融实验矩阵

```
配置名              Pass1  Pass2  Pass3  Pass4   预期 MFU
──────────────────────────────────────────────────────
Baseline（eager）    ✗      ✗      ✗      ✗       ~35%
+ Hoisting          ✓      ✗      ✗      ✗       ~40%
+ Overlap           ✓      ✓      ✗      ✗       ~45%
+ HW Path           ✓      ✓      ✓      ✗       ~50%
+ Mem Lifetime      ✓      ✓      ✓      ✓       ~52%
Full RFGraph        ✓      ✓      ✓      ✓       ~52%（+ Kernel fusion ~55%）
```

---

## 9. 实现路线图

### 9.1 阶段划分

```
Phase 0（1 个月）：原型验证
  目标：证明 Comm Hoisting 的可行性，不需要完整 IR
  工作：
    ① 在 Megatron-LM 的 MoE 层手动插入 A2A 提前调用
    ② 测量 overlap 率的提升（vs 原版）
    ③ 验证正确性（loss 曲线不变）
  交付：一篇技术博客 / workshop 短文

Phase 1（2~3 个月）：RFGraph 核心数据结构
  目标：实现 RFGraph 的图表示，支持基本的 Trace 和分析
  工作：
    ① 实现 RFNode 类体系（ComputeNode / CommNode / MemNode / SyncNode）
    ② 实现基于 torch.fx 的 RFTracer（识别通信操作并标注）
    ③ 实现 Pass 1（Comm Hoisting）
    ④ 单节点（8 GPU）功能验证
  交付：可运行的原型，Pass 1 的性能数据

Phase 2（3~4 个月）：完整 Pass + AMD 集成
  目标：实现 Pass 2~4，完成 AMD XGMI/RDMA 双路径支持
  工作：
    ① 实现 Pass 2（Overlap Maximization）
    ② 实现 Pass 3（Hardware Path Assignment，XGMI/RDMA 分流）
    ③ 实现 Pass 4（Memory Lifetime Minimization）
    ④ 多节点（64 GPU）扩展性测试
  交付：64 GPU 规模的完整性能数据

Phase 3（2~3 个月）：torch.compile 深度集成
  目标：CommPlaceholder + CUDA Graph，消除 graph break
  工作：
    ① 实现 CommPlaceholder 机制
    ② 验证 CUDA Graph 包含 RCCL 调用（PyTorch 2.x）
    ③ 实现 Padded Static Dispatch
    ④ 与 hipBLASLt autotuning 联调
  交付：完整的 compile 集成，论文初稿

Phase 4（1~2 个月）：实验与写作
  目标：完成所有 Benchmark，撰写论文
  工作：
    ① 跑完全部消融实验和对比实验
    ② 性能数据可视化
    ③ 论文撰写（目标会议：EuroSys / MLSys）
  交付：完整论文投稿
```

### 9.2 关键技术风险

```
风险 R1：CUDA Graph + RCCL 的稳定性
  问题：PyTorch 的 CUDA Graph with RCCL 在实践中有 bug
  缓解：先用 eager + stream 管理实现 overlap（Phase 1~2），
        CUDA Graph 作为 Phase 3 的增量优化

风险 R2：AMD HIP Stream 与 XGMI 的行为差异
  问题：AMD 的 XGMI stream 行为与 NVIDIA NVLink 有细微差异
  缓解：Phase 1 仅在单节点测试，充分理解 XGMI 行为后再扩展

风险 R3：torch.fx Trace 对动态路由的覆盖率
  问题：MoE 的 Top-K 索引操作可能导致 trace 失败
  缓解：先实现 Padded Static Dispatch（Phase 1），
        symbolic shape 作为 Phase 3 的改进

风险 R4：过度优化导致的正确性问题
  问题：Comm Hoisting 移动通信时序，可能在 edge case 下出错
  缓解：每个 Phase 都做完整的收敛性测试（loss curve 对比）
```

---

## 10. 与相关工作的差异定位

### 10.1 与 FlowMoE 的差异

```
FlowMoE（NeurIPS '25）：
  方式：运行时 DAG 调度（每 step 重新分析）
  粒度：Tensor Chunk 级别
  通信感知：是（能看到通信操作）
  编译期：否（运行时调度，有 Python overhead）
  硬件路径感知：否（不区分 XGMI 和 RDMA）
  AMD 支持：否（未在 AMD 上实现和测试）

RFGraph：
  方式：编译期静态分析（trace 一次，执行无 overhead）
  粒度：操作级（CommNode 单独调度）
  通信感知：是（CommNode 是一等公民）
  编译期：是（零运行时 overhead）
  硬件路径感知：是（XGMI vs RDMA 分流）
  AMD 支持：是（首要目标平台）

差异总结：FlowMoE 是「灵活的运行时调度器」，RFGraph 是「静态编译调度 IR」
  两者可以互补：RFGraph 借鉴 FlowMoE 的 DAG 思路，但在编译期完成
```

### 10.2 与 torch.compile 的差异

```
torch.compile：
  解决：单个 Module 内的算子融合和 Kernel 优化
  不解决：跨 Module 的 overlap / 通信操作的调度 / 硬件路径感知

RFGraph：
  解决：跨 Module 的全局调度 / 通信 overlap / 硬件路径感知
  不解决（委托给 torch.compile）：单个操作的 Kernel 优化

关系：RFGraph 和 torch.compile 是互补的，不是竞争的
      RFGraph 是「调度编译器」，torch.compile 是「Kernel 编译器」
      两者组合 = 完整的 MoE 训练编译栈
```

### 10.3 论文定位一句话

> **RFGraph 是第一个将通信操作作为一等公民、在编译期完成跨 Module 全局调度分析、并支持 AMD 硬件拓扑感知的 MoE 训练 IR，在 AMD MI300X 上实现了 X% 的通信-计算重叠率和 Y% 的端到端训练吞吐提升。**

---

*方向 B 深度分析整理于 2026-03-09 | ROCflow 框架研究讨论 | AIInfra-Book*
