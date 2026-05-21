# MoE 系统研究方向分析：论文空白与创新机会

> **定位：** 基于 100+ 篇 MoE 论文的系统性梳理，识别当前研究空白和高价值创新方向  
> **视角：** AI Infra 工程师 + AMD 硬件背景  
> **更新：** 2026-03-09

---

## 第一部分：现有研究的覆盖地图与空白分析

### 1.1 已被充分研究的方向（避免重复）

```
研究饱和度热力图（越深 = 越拥挤）：

方向                              论文数量    代表性工作              饱和度
──────────────────────────────────────────────────────────────────────
基础负载均衡 Loss                   高         Auxiliary Loss 标准化    ████████ 饱和
基础 Expert Parallel               高         DeepEP, FSMoE            ████████ 饱和
Expert 量化压缩                    中高        MoEQuant, PuzzleMoE      ██████   较饱和
推理 Expert 卸载/预取               高         KTransformers, PROBE     ███████  较饱和
Chunk 级通信-计算 Overlap           中          FlowMoE                  █████    中等
Tile 级 Kernel Overlap             低          Comet（仅 CUDA）         ███      有空间
FSEP 全分片专家并行                 低          LAER-MoE                 ██       早期
跨硬件平台（AMD）的 MoE 优化        极低        几乎空白                 █        空白
MoE 与 torch.compile 深度集成      极低        几乎空白                 █        空白
统一通信调度 IR                    极低        几乎空白                 █        空白
```

### 1.2 现有研究的三个系统性盲点

```
盲点 ①：各优化孤立，缺乏统一视角
  ──────────────────────────────────────────────────────────────────
  Comet 解决通信 Overlap → 但不感知负载不均（LAER-MoE 的问题）
  LAER-MoE 解决负载均衡 → 但没有 Tile 级 Overlap（Comet 的能力）
  FlowMoE 跨层调度      → 但不感知 AMD 硬件拓扑
  
  没有一篇论文把这三层叠加在一起做系统评估
  → 机会：端到端集成优化 + 叠加效果的实证研究

盲点 ②：所有工作都在 NVIDIA GPU 上
  ──────────────────────────────────────────────────────────────────
  Comet 的 Warp Specialization → CUDA 专属，未移植 HIP
  Megatron Parallel Folding    → H100 调优，未在 MI300X 验证
  DeepEP                       → 有 ROCm 版但不完整
  
  AMD MI300X 有独特优势（192GB HBM3e, XGMI 896GB/s, 双通道）
  → 机会：在 AMD 硬件上复现 + 利用 AMD 特性超越 NVIDIA 结果

盲点 ③：通信感知的路由设计
  ──────────────────────────────────────────────────────────────────
  所有现有路由算法（Top-K softmax）只考虑 token-expert 亲和度
  完全不考虑：目标 Expert 在哪、通信代价是多少
  MegaScale-MoE 做了拓扑感知的 Expert 放置，但没有改路由算法本身
  
  → 机会：通信代价入路由打分函数，同时影响模型质量和系统效率
```

---

## 第二部分：高价值创新方向（详细分析）

---

### 方向 A：Comm-Aware Token Routing（通信感知路由）

**新颖性：** ⭐⭐⭐⭐⭐  **工程难度：** ⭐⭐⭐  **论文发表潜力：** 顶会（OSDI / EuroSys / MLSys）

#### A.1 核心问题

```
现有 Top-K 路由：
  score(token_i, expert_j) = softmax(token_i @ W_gate)[j]
  → 只看 token-expert 语义亲和度
  → Expert_3（本地 GPU）和 Expert_200（3 跳之外）得分相同时，随机选择

通信代价差异（MI300X 8-GPU 节点）：
  本地 GPU Expert:   0     延迟
  XGMI 同节点 Expert: ~3μs  延迟（896 GB/s）
  RDMA 跨节点 Expert: ~15μs 延迟（400 Gbps）
  
  同等语义得分下，选择本地 Expert 可节省 5~15μs per token per layer
  × 64 MoE 层 × 百亿 token = 训练总通信量可减少 20~40%
```

#### A.2 创新设计

```
Comm-Aware Routing 公式：
  score(token_i, expert_j) = affinity(token_i, expert_j)
                            - λ(t) × comm_cost(i, j)

  comm_cost(i,j)：
    0      if expert_j is on same GPU
    α      if expert_j is on same node (XGMI)
    β      if expert_j is on remote node (RDMA)    β >> α

  λ(t)：动态权重，训练初期 λ=0（保证收敛），后期逐渐增大
         类似学习率调度，叫做 Communication Annealing

研究挑战（这些是论文的贡献点）：
  ① λ 如何调度？手动 vs 自适应（基于路由熵/负载统计）
  ② 对模型质量的影响？需要设计 Expert 局部性 vs 收敛性的权衡分析
  ③ 与负载均衡 Loss 如何协同？（两者都在影响路由决策）
  ④ 在不同拓扑（NVLink / XGMI / IB）下的效果差异

可量化的研究贡献：
  跨节点 A2A 流量减少 X%（可测量）
  训练吞吐提升 Y%（端到端，最有说服力）
  模型质量损失 < Z%（perplexity / downstream tasks）
```

#### A.3 与现有工作的差异

```
MegaScale-MoE：改变 Expert 的物理放置（系统层）
Comm-Aware Routing：改变路由算法本身（算法层）

两者可以叠加，且 Comm-Aware Routing 更通用：
  不依赖特定的 Expert 放置策略
  在任何 EP 配置下都有效
  天然与 FSEP Re-layout 协同（Re-layout 后路由代价表自动更新）
```

---

### 方向 B：MoE-Native IR + 跨层通信调度编译器

**新颖性：** ⭐⭐⭐⭐⭐  **工程难度：** ⭐⭐⭐⭐⭐  **论文发表潜力：** 顶会（PLDI / OSDI / MLSys）

#### B.1 核心问题

```
现有框架的通信调度方式：

  torch.compile：
    图范围 = 一个 nn.Module 的 forward()
    通信操作 = Graph break（图的边界，无法跨越）
    → 无法跨 Module 做通信-计算 overlap
    → MoE A2A 永远是阻塞点

  FlowMoE：
    运行时 DAG 调度
    每次 Step 都重新分析依赖（有 Python overhead）
    → 适合灵活调度，但无法提前编译优化

  没有一个系统同时做到：
    ① 跨 Module 边界的全局视野
    ② 编译期静态分析（零运行时 overhead）
    ③ 通信操作是 first-class citizen（不是 graph break）
    ④ 硬件路径感知（XGMI vs RDMA 分流）
```

#### B.2 创新设计

```
MoE IR（RFGraph）的核心创新：

传统编译器 IR：
  计算图 = 算子节点 + 数据边
  通信操作 = 特殊算子（但本质上还是一个节点）

RFGraph 的扩展：
  ① CommNode 有 hw_path 属性（XGMI / RDMA / LOCAL）
  ② CommNode 和 ComputeNode 之间有 OverlapEdge（不是数据依赖，是调度约束）
  ③ 图的范围跨越多个 nn.Module（Block_i 和 Block_{i+1} 在同一图中）

关键编译 Pass（这些是论文贡献）：
  Pass 1: Comm Hoisting（通信提前）
    分析：某个 A2A 的数据依赖最早何时满足？
    优化：尽可能早地发起通信（不等到计算完全结束）

  Pass 2: Overlap Maximization
    找到所有可以并发的 (CommNode, ComputeNode) 对
    分配到不同 Stream，插入最少必要的同步点

  Pass 3: Hardware Path Assignment
    节点内通信 → XGMI stream
    节点间通信 → RDMA stream
    两类通信并发，互不阻塞

  Pass 4: Memory Lifetime Minimization
    追踪每个 Tensor 的最后消费者
    在最后消费者执行后立即释放内存（而非等到整层结束）

论文实验设计：
  对比：eager mode / torch.compile / FlowMoE(runtime) / RFGraph(compile)
  指标：通信-计算重叠率 / 端到端吞吐 / Python overhead / 内存峰值
  模型：Mixtral 8x7B / DeepSeek-V3 scale
```

---

### 方向 C：FSEP 在 AMD 硬件上的原生实现与超越

**新颖性：** ⭐⭐⭐⭐  **工程难度：** ⭐⭐⭐⭐  **论文发表潜力：** 顶会（EuroSys / ASPLOS / ATC）

#### C.1 核心问题

```
LAER-MoE（ASPLOS '26）的 FSEP 是在 NVIDIA GPU 上实现的：
  ReduceScatter：基于 NCCL
  Expert GEMM 分片：基于 cuBLAS
  Re-layout：基于 NCCL All-to-All

AMD MI300X 有独特特性，可以做到更好：

特性 ①：统一内存（192GB HBM3e）
  FSEP 的 ReduceScatter 需要各卡读取 partial_out_i
  MI300X 的 XGMI 允许直接访问其他 GPU 的 HBM（无需显式发送）
  → ReduceScatter 可以变成「直接读取 + 本地加法」
  → 消除了显式通信的 latency（变成内存访问延迟）

特性 ②：XGMI 双向 896 GB/s（8 GPU 节点）
  LAER-MoE 的 ReduceScatter 通信量 = T × H
  在 XGMI 上：延迟 = T×H×2bytes / 896GB/s ≈ 0.1ms（几乎可忽略）
  → FSEP 的通信额外开销在 MI300X 上接近于 0

特性 ③：LDS（Local Data Share）64KB/CU（比 NVIDIA 大）
  Comet 的 Tile-level Overlap 需要 LDS 作为 GEMM→RDMA 的中间缓冲
  更大的 LDS → 更大的 Tile → 更少的同步频率 → 更高的 overlap 率
```

#### C.2 创新设计

```
AMD-FSEP（超越原版 LAER-MoE 的设计）：

优化 ①：XGMI 直接访问替代 ReduceScatter
  传统 ReduceScatter：
    GPU_i 发送 partial_out_i → 所有 GPU 接收 → 求和
    通信延迟 = T × H / XGMI_bandwidth

  AMD-FSEP 的 In-Place Reduction：
    GPU_i 直接通过 XGMI 读取 GPU_j 的 partial_out_j
    在本地 HIP Kernel 内完成加法
    延迟 = max(读取延迟, 计算延迟)（可以 overlap）
    → 比显式 ReduceScatter 快 2~3x

优化 ②：HIP Wavefront 专用化（类 Comet，但在 AMD 上首发）
  Wavefront Group 0,1,2 → Expert GEMM（MFMA 指令）
  Wavefront Group 3     → XGMI 直接读取 + 部分和累加

  AMD MFMA（Matrix Fused Multiply-Add）指令：
    比 NVIDIA wmma 更灵活的矩阵计算指令集
    支持 FP8 / BF16 / FP16 多精度
    → 为 Expert GEMM 提供专门优化路径

优化 ③：Re-layout 与 HBM 带宽的协同
  Re-layout（参数搬迁）在 MI300X 上通过 XGMI 直接完成
  利用反向传播时的计算-搬迁重叠窗口
  峰值内存增加 < 3%（原版 5~10%）

预期论文贡献：
  系统贡献：首个在 AMD GPU 上实现 FSEP 的框架
  性能贡献：在 MI300X 上超越 LAER-MoE 在 H100 上的结果（借助硬件优势）
  分析贡献：AMD vs NVIDIA 在 FSEP 这类 communication-heavy workload 上的详细对比
```

---

### 方向 D：MoE 训练的 Backward Pass 通信调度自由度

**新颖性：** ⭐⭐⭐⭐⭐  **工程难度：** ⭐⭐⭐  **论文发表潜力：** MLSys / EuroSys

#### D.1 一个被所有论文忽视的问题

```
所有现有 MoE 通信优化论文（Comet / FlowMoE / LAER-MoE）
都专注于前向传播（Forward）的通信优化

但反向传播（Backward）有一个独特性质：

前向传播：
  Block_0 → Block_1 → ... → Block_N
  顺序固定，依赖严格

反向传播：
  Block_N → Block_{N-1} → ... → Block_0
  
  关键：Block_i 的 A2A_bwd 和 Block_{i-1} 的 Expert_bwd GEMM
        在数据依赖上是 独立的！

  Block_i 反向：
    dX_i = f(dY_i)         → 传给 Block_{i-1}
    dW_i = g(X_i, dY_i)    → Expert 权重梯度（与 dX_i 独立计算）
    A2A_bwd_i = h(dX_i)    → 梯度通信

  Block_{i-1} 反向：
    需要：A2A_bwd_i 完成（dX_i 传回）才能开始
    不需要：等待 dW_i 的 AllReduce（梯度同步可以异步）

  → dW_i 的 AllReduce 可以与 Block_{i-1} 的 Expert_bwd GEMM 完全重叠！
  → 而且 Block_{i-1} 的 A2A_bwd 也可以与 dW_i 的 AllReduce 重叠！
```

#### D.2 创新设计

```
Backward Comm Scheduling（反向传播通信调度最优化）：

当前最优时间线（即使是 FlowMoE）：
  Block_N bwd: [Expert GEMM] → [A2A_bwd] → [dW AllReduce]
               ↑串行↑
  Block_{N-1} bwd:                         [Expert GEMM] → ...

最优时间线（本方向的贡献）：
  Block_N bwd:    [Expert GEMM dX] [Expert GEMM dW]
                       │                  │
                  [A2A_bwd_N]        [AllReduce dW_N]
                       │                  │
                  ↓ dX_N 传回        ← 并发 →        ← 并发 →
  Block_{N-1} bwd: [Expert GEMM dX]  [Expert GEMM dW]
                        │                  │
                   [A2A_bwd_{N-1}]    [AllReduce dW_{N-1}]

  关键insight：
    dX（传给上一层的梯度）和 dW（本层的参数梯度）可以并行计算
    dW 的 AllReduce 可以与下一层（Block_{i-1}）的所有计算重叠
    → 所有 AllReduce 被完全隐藏在计算之后

可量化贡献：
  梯度同步时间（AllReduce）完全消除（被计算覆盖）
  vs FlowMoE：额外 +5~15% 吞吐提升
  实现复杂度：不需要新 Kernel，只需要调度框架的改进
```

---

### 方向 E：MoE 的 torch.compile MoE-specific Pass

**新颖性：** ⭐⭐⭐⭐  **工程难度：** ⭐⭐⭐⭐  **论文发表潜力：** PLDI / CGO / MLSys

#### E.1 核心问题

```
torch.compile 对 MoE 支持极差的根本原因：

问题 ①：动态控制流（Graph Break）
  Top-K 路由后，每个 Expert 收到的 token 数是动态的
  → torch.compile 遇到动态 shape 就产生 graph break
  → MoE 层被分成 N 个小 graph，每个 Expert 单独编译
  → 无法做 Gate+Dispatch+GEMM 的整体融合

问题 ②：稀疏索引操作
  routing_index = argtopk(gate_logits)   # 动态索引
  dispatched = input[routing_index]      # 稀疏 gather
  → inductor 不擅长优化不规则内存访问模式

问题 ③：All-to-All 是通信 primitive，不是可 fuse 的算子
  torch.compile 的 graph 在 dist.all_to_all 处强制结束
  → 通信前后分别编译，无法做跨通信的 kernel fusion
```

#### E.2 创新设计

```
MoE-Compile Pass 的三个核心创新：

创新 ①：Static-Shape Expert Dispatch（静态化动态 shape）
  问题：每个 Expert 的 token 数动态变化 → 无法静态编译

  解法：Padded Static Dispatch
    capacity = max_tokens_per_expert（固定容量）
    用 mask 标记有效 token：
      dispatched_padded[T_expert, H]  ← 固定形状
      valid_mask[T_expert]            ← 标记实际 token

    → torch.compile 看到的是固定 shape，可以正常 trace
    → Expert GEMM 在 padded tensor 上执行
    → 结果 × valid_mask 过滤无效输出
    → padding 浪费约 5~10%，换来 compile 的完整优化能力

创新 ②：Gate-Dispatch 联合 Kernel
  当前：Gate（softmax + topk）和 Dispatch（gather）是两个独立操作
  
  融合 Kernel：
    input: X[T, H]
    output: dispatched[N_experts, capacity, H]（直接写入目标位置）
    
    一个 HIP/CUDA Kernel 完成：
      计算 gate logits
      softmax + topk（得到路由决策）
      直接把 token scatter 写入对应 Expert 的 slot
    
    → 消除中间 routing_buffer，内存节省 ~30%
    → 内存访问次数从 3 次（read→gate→dispatch）降为 1.5 次

创新 ③：A2A 前后的 Graph 融合（打破 Graph Break）
  torch.compile 的 inductor 后端可以接受 custom lowering

  设计 MoE-specific lowering rule：
    [Gate+Dispatch Kernel] → [A2A 占位符] → [Expert GEMM] → [A2A 占位符] → [Gather Kernel]
                              ↑ 不 break graph                ↑
                              用 custom cuda graph 节点表示

    在 CUDA Graph capture 阶段，A2A 占位符展开为实际 NCCL/RCCL 调用
    → 整个 MoE 层（含通信）作为一个 CUDA Graph 执行
    → 消除 Python overhead，接近理论峰值吞吐

预期贡献：
  vs eager MoE：+40~60% 吞吐（compile 带来的 kernel fusion 收益）
  vs 当前 torch.compile MoE：+20~30%（专用 pass vs 通用 compile）
  内存节省：~30%（Gate-Dispatch 融合消除中间 Buffer）
```

---

### 方向 F：MoE 训练的 Online 性能自适应系统

**新颖性：** ⭐⭐⭐⭐  **工程难度：** ⭐⭐⭐  **论文发表潜力：** EuroSys / SOSP / ATC

#### F.1 核心问题

```
所有现有论文的优化都是「静态的」或「规则触发的」：

  LAER-MoE Re-layout：每 K 步检测，规则：imbalance > threshold → 重排
  Comm-Aware Routing：λ 静态或按预设 schedule 变化
  MoE Parallel Folding：并行配置在训练前手动设置，固定不变

但 MoE 训练中有大量「动态特性」：
  ① Expert 负载分布随训练进展变化（早期均衡，后期出现专业化）
  ② 网络拥塞随时间变化（批量作业场景下，带宽被其他任务竞争）
  ③ GPU 故障导致集群拓扑动态变化
  ④ 不同 batch 的 token 分布不同（代码 vs 数学 vs 自然语言）

一个真正高效的系统应该对这些动态变化实时响应
```

#### F.2 创新设计

```
MoE Online Adaptive Scheduler（在线自适应调度器）：

感知层（Perception）：
  每 Step 收集：
  ┌─────────────────────────────────────────────────┐
  │  expert_load[N_experts]      ← Gate 路由统计    │
  │  comm_latency[N_gpus, N_gpus]← 实测通信延迟     │
  │  compute_time[N_gpus]        ← 各卡计算耗时     │
  │  memory_usage[N_gpus]        ← 实时显存状态     │
  └─────────────────────────────────────────────────┘

决策层（Decision）：
  每 K Step 做一次综合决策：

  问题建模（多目标优化）：
    minimize: max_gpu(compute_time + comm_latency)
    subject to: memory_constraint, bandwidth_constraint

  决策变量：
    ① Expert 分片度 S_e（哪些 Expert 需要 FSEP + Re-layout）
    ② 路由 λ 值（Comm-Aware Routing 的通信惩罚强度）
    ③ Chunk 大小 C（FlowMoE 风格的分块粒度）
    ④ Stream 分配（哪些通信走 XGMI，哪些走 RDMA）

  求解器：
    不用全局最优（太慢），用「贪心 + 局部搜索」
    利用历史数据做预测（滑动窗口平均）
    决策时间 < 1ms（必须在 Step 间隙内完成）

执行层（Execution）：
  在下一个 Step 开始前异步应用新配置
  热变更：不重启、不重新 compile，动态修改调度参数

论文贡献层次：
  ① 在线自适应调度的问题形式化（新颖）
  ② 轻量级在线求解器设计（贡献）
  ③ 在不同 workload 类型（均匀 vs 不均匀 batch）下的自适应效果（实验）
  ④ 与静态配置的对比：自适应系统在动态负载下的优势（可量化）
```

---

## 第三部分：各方向的综合评分与优先级

```
研究方向评分矩阵：

方向                          新颖性  工程可行性  发表价值  AMD差异化  综合优先
────────────────────────────────────────────────────────────────────────────
A. Comm-Aware Routing          ★★★★★    ★★★★      ★★★★     ★★★★     🔥🔥🔥🔥 高
B. MoE-Native IR + Compiler    ★★★★★    ★★★       ★★★★★    ★★★★★    🔥🔥🔥   中高（长期）
C. AMD-FSEP 原生实现            ★★★★     ★★★★      ★★★★     ★★★★★    🔥🔥🔥🔥 高
D. Backward Comm Scheduling    ★★★★★    ★★★★      ★★★★     ★★★      🔥🔥🔥   中高
E. torch.compile MoE Pass      ★★★★     ★★★       ★★★★     ★★★★     🔥🔥🔥   中高
F. Online Adaptive Scheduler   ★★★★     ★★★★      ★★★      ★★★      🔥🔥     中

说明：★ = 1分，最高 ★★★★★ = 5分
AMD差异化：越高 = 越能体现 AMD 硬件优势，越难被 NVIDIA 直接复制
```

---

## 第四部分：推荐论文路线图

### 路线 1：短期高回报（6~9 个月，适合 workshop/短文或工程博客）

```
Step 1（1~2 个月）：
  实现 Comm-Aware Routing 的基础版本（静态 λ）
  在 ROCflow 或 Megatron 上插入通信代价惩罚项
  测量：跨节点流量减少量 vs 模型质量影响

Step 2（2~3 个月）：
  实现 AMD-FSEP 的基础版本（固定分片度，无 Re-layout）
  基于 RCCL ReduceScatter + hipBLASLt Group GEMM
  对比 LAER-MoE 在 H100 上的结果

Step 3（2~3 个月）：
  叠加 Comm-Aware Routing + AMD-FSEP
  端到端对比 Megatron-LM on MI300X
  输出：系统报告 / workshop 论文 / 技术博客
```

### 路线 2：中期顶会论文（12~18 个月，目标 EuroSys/MLSys/OSDI）

```
核心论文方向：「面向 AMD GPU 的 MoE 训练通信优化系统」

贡献 ①：Comm-Aware Routing（算法层）
  新的路由公式 + λ 自适应调度策略
  理论分析：通信代价 vs 路由质量的 Pareto 分析

贡献 ②：AMD-FSEP（系统层）
  利用 XGMI 直接访问替代 ReduceScatter
  HIP Wavefront 专用化的 Expert GEMM-RDMA 联合 Kernel

贡献 ③：Backward Comm Scheduling（调度层）
  反向传播的 dX/dW 分离计算 + AllReduce 全隐藏

实验设计：
  基线：Megatron-LM + NCCL on NVIDIA H100
  对比：ROCflow on AMD MI300X
  模型：Mixtral 8x7B / 8x22B / DeepSeek-V3 scale (1/4)
  指标：MFU / 通信-计算重叠率 / 跨节点流量 / 端到端吞吐
```

### 路线 3：长期系统论文（18~24 个月，目标 SOSP/OSDI）

```
核心论文方向：「MoE 训练的统一通信调度 IR」（方向 B）

这是技术难度最高、但影响力最大的方向

贡献 ①：RFGraph IR 的形式化定义
  通信操作作为 first-class 节点
  OverlapEdge 的语义和正确性证明

贡献 ②：三个核心编译 Pass
  Comm Hoisting / Overlap Maximization / HW Path Assignment

贡献 ③：与 torch.compile 的集成
  MoE-specific lowering rules
  动态路由的 symbolic shape 处理

贡献 ④：大规模实验验证
  1024 GPU 以上的 MoE 训练
  与 FlowMoE / Megatron / torchtitan 的全面对比
```

---

## 第五部分：当前研究格局的一句话总结

```
现有论文解决的问题：
  Comet      → 通信与计算如何重叠（HOW to overlap）
  FlowMoE    → 不同操作如何调度（HOW to schedule）
  LAER-MoE   → 负载如何均衡（HOW to balance）
  MoEBlaze   → 内存如何节省（HOW to save memory）

没有人回答的问题：
  ① WHEN 应该改变路由策略（自适应时机）
  ② WHY 当前硬件特性没有被充分利用（AMD 视角）
  ③ HOW MUCH 这些优化叠加的总收益（系统性评估缺失）
  ④ WHERE 在编译期 vs 运行期做这些决策更好（IR 设计问题）

这四个「没有人回答」就是研究机会所在。
```

---

*研究方向分析整理于 2026-03-09 | 基于 100+ 篇 MoE 论文系统梳理 | AIInfra-Book*
