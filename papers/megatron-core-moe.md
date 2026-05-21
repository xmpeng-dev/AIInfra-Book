 Scalable Training of Mixture-of-Experts Models with Megatron Core

> **arXiv:** [2603.07685](https://arxiv.org/abs/2603.07685) | **PDF:** https://arxiv.org/pdf/2603.07685  
> **发表时间:** 2026年3月  
> **机构:** NVIDIA  
> **框架:** Megatron-Core (v0.16)  
> **核心贡献:** MoE 训练完整系统栈，GB300 上 DeepSeek-V3 达 **1,233 TFLOPS/GPU**，支持万亿参数规模

---

## 1. 核心问题：MoE 训练的三面墙

### 1.1 根本矛盾：参数-计算不匹配

```
Dense 模型的平衡：
  参数量 = N_total
  每 token 计算 ≈ 6 × N_total
  → 参数和计算同步增长，GPU 始终有足够计算掩盖通信

MoE 模型的不匹配：
  参数量 = N_total（∝ E，专家数量）
  每 token 计算 ≈ 6 × N_active（∝ K，激活专家数）
  → K << E，参数和计算增长完全脱耦

  DeepSeek-V3 具体数据：
    总参数：685B（671B 主模型 + 14B MTP）
    激活参数：37B / token
    差距：18×

  后果：
    1. 内存需求 ∝ N_total（远大于同等计算量的 Dense 模型）
    2. 需要更多 GPU 分摊内存 → 通信量增加
    3. 但每 token 计算 ∝ N_active 不增长 → 无法掩盖通信
```

### 1.2 三面墙的定义

```
┌─────────────────────────────────────────────────────────────┐
│                    三面墙（Three Walls）                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  内存墙（Memory Wall）                                        │
│    所有 E 个专家的参数/梯度/优化器状态常驻内存                  │
│    但只有 K 个激活 → 内存压力远超同计算量 Dense 模型             │
│    动态路由造成不可预测的内存峰值                                │
│                                                              │
│  通信墙（Communication Wall）                                  │
│    EP 需要 All-to-All 通信                                    │
│    每 GPU 发送量 ≈ T × K × h × (EP-1)/EP                     │
│    dispatch + combine 翻倍                                    │
│    大 EP 跨节点 → 带宽降一个数量级                              │
│    未优化时 All-to-All 可占训练时间的 60%                       │
│                                                              │
│  计算效率墙（Compute Efficiency Wall）                          │
│    小 GEMM：细粒度 Expert 产生大量小矩阵乘（利用率低）          │
│    路由开销：路由+排列 ≈ 9% 层执行时间                          │
│    负载不均：动态路由导致部分专家过载、部分空闲                   │
│    Host 开销：MoE 启动更多内核 → CPU 跟不上 GPU                 │
│                                                              │
│  关键：三面墙紧密耦合，优化一面常常加重另一面                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 DeepSeek-V3 内存剖析（PP4×VPP4×EP64, 256 GPUs）

| 组件 | 每 GPU 内存 | 优化手段 |
|------|-----------|---------|
| 权重 & 梯度 | 36.4 GB | PP, EP, TP 分片 |
| 主权重 & 优化器状态 | 32.1 GB | 分布式优化器, BF16 moments |
| **激活** | **131.0 GB** | 低精度, 重算, Offload |
| **总计** | **199.5 GB** | 远超 H100 的 80GB |

> **激活内存是最大消费者**，超过权重和优化器状态的总和，是优化的首要目标。

---

## 2. Megatron-Core MoE 架构

### 2.1 MoE 层四阶段前向通路

```
Input Tokens [B×S, H]
     │
     ▼
┌─── Stage 1: Route ──────────────────────────────────────┐
│  TopKRouter:                                              │
│    Gating Linear → Score Function (Softmax/Sigmoid)       │
│    → Top-k Selection → Load Balancing                     │
│  输出：probs (路由权重) + routing_map (布尔掩码)           │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌─── Stage 2: Dispatch ──────────────────────────────────┐
│  Token Dispatcher:                                        │
│    Permute (按 Expert 分组) → All-to-All (发送到目标 GPU)  │
│  三种后端：AllGather / All-to-All / Flex (DeepEP/HybridEP)│
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌─── Stage 3: Expert Computation ────────────────────────┐
│  Grouped GEMM (TEGroupedMLP):                             │
│    所有本地 Expert 单次调用 → 最大化 GPU 利用率             │
│  可选：Shared Expert 并行计算                              │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌─── Stage 4: Combine ──────────────────────────────────┐
│  All-to-All 返回 → Unpermute → 加权合并                   │
│  + Shared Expert 输出相加                                  │
└────────────────────────────────────────────────────────────┘
     │
     ▼
Output Tokens [B×S, H]
```

### 2.2 进程组管理

```
ProcessGroupCollection
├── Attention Layer Groups: tp, cp, dp, pp
└── Expert Layer Groups:    ep, expt_tp, expt_dp, pp

组件到进程组映射：
  Router        → tp, cp, tp_cp     （权重在 EP ranks 复制）
  Dispatcher    → ep, tp_ep         （跨 EP All-to-All）
  Experts       → ep, expt_tp, expt_dp  （EP 分片，EDP 梯度归约）
  Shared Expert → tp                （同 Dense MLP）

这种分离使 Parallel Folding 成为可能：
  Attention 和 MoE 层使用不同的 TP/DP 配置
```

---

## 3. Parallel Folding：解决 Dense-Sparse 不匹配

### 3.1 Dense-Sparse 不匹配问题

```
同一 Transformer Block 中的两种矛盾需求：

             Attention (Dense)          MoE (Sparse)
计算特征    每个 token 参与所有计算      每个 token 路由到 K/E 个专家
TP 需求     大 QKV 矩阵 → 高 TP 有利    小 Expert → 高 TP 有害
CP 需求     长序列 → 高 CP 有利          无序列依赖 → CP 无意义
EP 需求     无（没有 Expert）             必需（分发 Expert）

传统框架：EP ⊆ DP，强制 attention 和 MoE 共用一套并行配置
  → World Size = TP × CP × PP × DP，其中 EP ⊆ DP

三个致命问题：
  1. GPU 需求乘法膨胀：EP=8 → DP≥8，加上 CP=8 → 至少 64 GPUs
  2. 被迫次优：高 TP 利好 attention 但碎片化 expert，低 TP 反之
  3. 跨节点通信：EP 被迫走低带宽路径
```

### 3.2 Parallel Folding 方案

```
核心思想：不强制 Attention 和 MoE 共用并行配置

Attention 层: TP × CP × DP × PP   （优化序列级密集计算）
MoE 层:      ETP × EP × EDP × PP  （优化专家分发）

唯一约束：PP 必须一致（保证梯度正确流动）

五维并行栈：
  TP    (Tensor)      → Attention:  分片大 QKV 矩阵
  CP    (Context)     → Attention:  分布长序列
  DP    (Data)        → Attention:  处理不同批次
  PP    (Pipeline)    → Both:       按层分割（必须一致）
  EP    (Expert)      → MoE:        跨 GPU 分布专家
  ETP   (Expert TP)   → MoE:        专家内分片（很少用）
  EDP   (Expert Data) → MoE:        复制专家增加吞吐

关键优势：
  ┌────────────────────────────────────────────────┐
  │  1. 打破 EP ≤ DP 约束                           │
  │     传统：Attention TP=4, CP=2, DP=8 → EP≤8     │
  │     Folding：MoE ETP=1, EP=64, EDP=1  → 8× EP   │
  │                                                   │
  │  2. 降低最低 GPU 要求                             │
  │     传统 CP=8, EP=8 → 至少 64 GPUs               │
  │     Folding CP/EP 共享 → 仅需 8 GPUs             │
  │                                                   │
  │  3. 独立优化各层                                  │
  │     Attention 用高 TP (大矩阵)                    │
  │     MoE 用 ETP=1 (完整 Expert GEMM)              │
  │                                                   │
  │  4. 高带宽通信留在 NVLink 域                      │
  │     CP 和 EP 的 All-to-All 都在 NVLink 组内      │
  └────────────────────────────────────────────────┘
```

---

## 4. 突破内存墙

### 4.1 Memory-Efficient Permutation（零开销）

```
标准公式（路由权重在 Expert 计算后应用）：
  y = Σ p_i · W2_i · φ(W1_i · x)
  
  反向传播需保存每个 Expert 输出 E_i(x) 来计算 ∂L/∂p_i
  → 额外内存开销

Memory-Efficient Permutation（路由权重在第二层线性层前应用）：
  y = Σ W2_i · (p_i · φ(W1_i · x))
  
  数学等价（无 bias 时线性映射可交换标量乘法）
  ∂L/∂p_i 仅依赖 φ(z_i)，可从 z_i 即时重算
  z_i 本来就要为 SwiGLU 反向保存 → 无额外 buffer

效果：DeepSeek-V3 每 GPU 节省 ~26.3 GB，零计算开销
```

### 4.2 细粒度重算

```
传统全层重算：+33% 计算开销，MoE 更贵（重新触发 All-to-All）

Megatron-Core 细粒度重算策略：
  ┌──────────────────────────┬──────────────┬──────────────┐
  │ 重算目标                  │ 节省内存/GPU  │ 计算开销     │
  ├──────────────────────────┼──────────────┼──────────────┤
  │ MLA Up-Projection        │ 30.4 GB      │ < 5%         │
  │ Activation (SwiGLU)      │ 3.8 GB       │ < 5%         │
  │ LayerNorm                │ 8.2 GB       │ < 5%         │
  ├──────────────────────────┼──────────────┼──────────────┤
  │ 总计                      │ 42.4 GB      │ < 5%         │
  └──────────────────────────┴──────────────┴──────────────┘

两项技术组合：
  1. 粒度化重算：仅选择内存大/计算小的操作重算
  2. 输出丢弃重算：checkpoint 模块的输出立即释放，反向时重算恢复
```

### 4.3 细粒度激活 Offloading

```
动机：
  DeepSeek-V3 激活 37B / token（18× ratio）
  Kimi-K2 达到 1T 总参数，32B 激活（31× ratio）
  激活内存不随 EP/PP 减少（它们减少的是参数内存）

核心思路：GPU Copy Engine 和 Compute Engine 独立运行
  → D2H copy 与后续模块计算并行 → 零成本

前向：模块计算后立即 offload 输入激活到 CPU（专用 D2H stream）
反向：Layer-Staggered Reload—从下一层 reload 同模块激活
     （当前层计算梯度的同时，reload 下一层要用的激活）

关键特性：
  • 模块级粒度：--offload-modules 指定哪些模块 offload
  • 混合策略：轻量模块(LayerNorm)用重算，昂贵模块(attention/expert)用 offload
  • 与 CUDA Graphs 兼容：使用 external events 而非 stream 同步

效果：
  DeepSeek-V3: 169GB → 151GB（-10.7%内存），仅 -1.6% 吞吐
  Qwen3-235B:  TP2→TP1 + EP16→EP64，+15.0% 吞吐（offload 换来更优并行配置）
```

### 4.4 FSDP + EP

```
Megatron-FSDP 双 DeviceMesh 架构：
  Primary DeviceMesh → Dense 模块: DP-Shard, DP-Outer, TP, CP
  Expert DeviceMesh  → EP 模块: FSDP scoped to EDP dimension

  Attention/Norm → 主 mesh
  Expert FFN     → Expert mesh

  AllGather/ReduceScatter 限制在 EDP 小组内
  → 不跨全 DP ranks

零拷贝通信：
  1. 非均匀分片：模块内参数扁平化拼接 → 分片边界对齐通信 buffer
     → 集合操作直接读取存储，无冗余拷贝（Llama3 405B 通信开销 -10%）
  2. 持久双 buffer + NCCL User Buffer Registration
     → SM 占用从 8-32 SM 降至 1-4 SM
```

### 4.5 内存优化汇总

| 技术 | 目标 | 代价 |
|------|------|------|
| Reduced-Precision Training | 激活 | 精度 + CPU 开销 |
| Memory-Efficient Permutation | 激活 | **无** |
| Fine-grained Recomputation | 激活 | 计算开销 |
| Fine-grained Offloading | 激活 | CPU/PCIe 开销 |
| Precision-aware Optimizer | 优化器状态 | 精度 |
| FSDP (with EP) | 参数+优化器 | 通信开销 |

---

## 5. 突破通信墙

### 5.1 DeepEP 和 HybridEP

```
传统 All-to-All 问题：
  需要先 permute 步骤 → token 复制 top-k 次 → 冗余通信
  host 端预处理引入额外 CPU 开销

Token-based Dispatch（DeepEP / HybridEP）：
  消除 permute 步骤，避免发送冗余 token
  利用 TMA / IBGDA 硬件原语

HybridEP Dispatch 设计：
  读数据 → 共享内存 → FIFO 队列 → 写入目标
  跨节点：先 RDMA 交换同 local-index GPU 间数据
         再节点内转发 → 跨节点/节点内传输重叠

HybridEP Combine 设计：
  将 reduction 融合进通信内核
  读 FIFO → 归约 → 直接写目标地址
  （标准 All-to-All 需要通信后单独 unpermute）

性能（GB200, h=7168, seq=4096, 256 experts）：
  ┌───────┬──────────────┬──────────────┐
  │ EP=64 │ HybridEP(µs) │ All-to-All(µs)│
  ├───────┼──────────────┼──────────────┤
  │ disp  │ 675          │ 930          │
  │ comb  │ 744          │ 827          │
  └───────┴──────────────┴──────────────┘
```

### 5.2 EP 通信重叠

```
问题：优化后的 dispatcher 仍在关键路径上
  DeepSeek-V3 EP64 → All-to-All 仍占 30-40% 迭代时间

方案：1F1B FWD-BWD 合并 + 流分离

  Merged FWD-BWD（DualPipe 等价）：
    micro-batch 1 的 FWD 与 micro-batch 0 的 BWD 合并
    → 无额外内存开销（激活被复用）

  双 CUDA Stream：
    Compute Stream → attention / expert MLP 计算
    Comm Stream    → All-to-All dispatch / combine
    → 两个 stream 交替执行，All-to-All 与计算并行

  W/D 分割：
    BWD MLP 拆分为：
      W/mlp（权重梯度，独立于 dispatch）
      D/mlp（数据梯度，馈入 dispatch）
    → W/mlp 可与 F/mlp 重叠来掩盖 B/dispatch

效果：
  EP 通信开销从 30-40%（优化 dispatcher 后）→ < 5% 迭代时间
  93% 的 Expert 通信延迟被成功掩盖
```

---

## 6. 突破计算效率墙

### 6.1 Grouped GEMM 和内核融合

```
Grouped GEMM 实现：
  i.  Multi-stream cuBLASLt GEMMs（多流重叠）
  ii. CUTLASS Grouped GEMM（单内核融合，大 GEMM 数更优）
  iii.cuBLASLt Grouped GEMM（设备端形状，CUDA Graph 友好）★ 开发中
  iv. cuteDSL Grouped GEMM（融合激活/量化/swizzling）★ 开发中

三级内核融合：
  1. Permutation Fusion：预处理 → permute → unpermute 融合
  2. Router Fusion：score 计算 + top-k + softmax/sigmoid 融合
  3. Aux-Loss Fusion：辅助损失计算融合为单内核
```

### 6.2 CUDA Graphs

```
CPU 开销来源：
  Python 执行 + Framework 开销 + 内核启动开销
  趋势加剧：更快 GPU / MoE 复杂度 / 低精度训练量化内核

Full CUDA Graphs（整个前向-反向）：
  仅支持 drop-and-pad MoE（静态形状）

Partial CUDA Graphs（分层捕获）：
  可 Graph 的：Attention / Router / EP 预处理 / Shared Expert / Dense MLP
  不可 Graph 的：Token Dispatch / Expert GEMM / Token Combine

  效果：DeepSeek-V3 GB200 训练 +10% 端到端加速，额外 ~7GB 内存

CUDA Graph 内存优化：
  1. 减少 Graph 数量：无 PP 时 microbatch 共享 Graph（L×2 vs L×M×2）
  2. Pool sharing：所有 Graph 按执行顺序共享一个 pool
  3. Buffer reuse：静态 buffer 按 PP 执行顺序复用
```

### 6.3 Full CUDA Graphs for Dropless MoE（三项创新）

```
挑战 1：内核启动不知道实际问题规模
  → Device-Initiated GPU Kernels
    GPU 直接从设备内存读取形状信息
    静态启动配置 + 运行时跳过多余计算
    实现：cuBLASLt Grouped GEMM（CUDA 13.1+）
         cuteDSL Grouped GEMM（融合 SwiGLU + 量化）
         Sync-Free HybridEP Dispatch（预分配上界 buffer）

挑战 2：内存分配不知道实际大小
  → ECHO（Elastic Cloning for Hot Experts）
    1. Planner 识别热门 Expert → 生成 hot expert map + routing map
    2. Expert Dispatch 通过 HybridEP 克隆权重到空闲 rank
    3. Token Dispatch 将溢出 token 路由到克隆
    4. 反向时梯度从克隆 reduce 回 home expert
    
    效果：减少负载方差 → worst-case buffer 更接近实际用量
         改善 GPU 利用率 → 消除拖尾者问题

  → Paged Stashing
    思路：worst-case buffer 只需 1 个（跨层共享 tmp buffer）
         每层仅存储实际 token 到 paged stashing buffer
    
    内存：O(layers × worst_case) → O(worst_case + actual_total)
    
    实现：固定页大小（默认 64 token/页）
         free list 用循环 buffer 管理
         device-initiated stash/reload 内核
         Pack/Unpack stream 与计算重叠

组合效果：完全消除 per-iteration host-device 同步
         Dropless MoE 实现 Full CUDA Graphs
```

---

## 7. FP8/FP4 低精度训练

### 7.1 为什么低精度对 MoE 特别重要

```
低精度训练是唯一同时影响三面墙的跨领域优化：

  内存墙：FP8 激活比 BF16 减少 50%，FP4 减少 75%
           DeepSeek-V3 FP8 节省 ~16 GB 激活内存/GPU
  通信墙：参数 AllGather 体积减半（1 字节 vs 2 字节）
  计算墙：FP8/FP4 Tensor Core 吞吐量高于 BF16

MoE 放大了收益也放大了风险：
  放大收益：数百个专家 → 激活内存与专家数线性增长 → 低精度节省更大
  放大风险：Router 依赖精确 score → 量化噪声可能导致专家选择不稳定
           → 训练不稳定 / 模型质量下降 / 专家坍塌

策略：选择性精度（Selective Precision）
  1. 保护 Router：Router 保持 FP32（确保稳定的专家选择）
  2. 保护关键组件：Embedding、Output Layer、主梯度、主权重、优化器状态保持原精度
  3. 量化大头计算：Expert GEMM（计算量占比最大）使用低精度 + 精心设计的量化方案
```

### 7.2 低精度对三面墙的影响汇总

| 墙 | 收益 | 细节 |
|----|------|------|
| 内存墙 | FP8 激活 -50% / FP4 激活 -75% | 线性层输入从 BF16→FP8/FP4 |
| 内存墙 | 消除 BF16 权重副本 | FP8/FP4 Primary Weights |
| 内存墙 | 优化器状态 BF16 moments -50% | 正交技术 |
| 通信墙 | 参数 AllGather -50% | FP8/FP4 Primary Weights |
| 计算墙 | 更快 Tensor Core GEMM | FP8(Hopper+) / FP4(Blackwell+) |
| 计算墙 | 量化内核开销（副作用） | 通过融合/CUDA Graphs 缓解 |

### 7.3 四种低精度训练方案

```
一个低精度训练方案 (Recipe) 包含：
  • 数据格式：E4M3 / E5M2 / Hybrid / E2M1(FP4)
  • 缩放粒度：Per-Tensor / Per-Block / Per-1×32
  • 量化范围：通常所有线性层量化，Embedding/LM Head/SDPA 保持高精度
  • 附加机制：随机舍入、RHT 等

┌─────────────────┬────────────────┬──────────────┬──────────────────┐
│ 方案             │ 缩放粒度        │ 硬件平台      │ 推荐程度         │
├─────────────────┼────────────────┼──────────────┼──────────────────┤
│ Per-Tensor FP8  │ 1 scale/tensor │ Hopper+      │ 入门（简单）      │
│ Blockwise FP8   │ 128×128 block  │ Hopper       │ ★ Hopper 推荐    │
│ MXFP8           │ 1×32 elements  │ Blackwell    │ ★ Blackwell 推荐 │
│ NVFP4           │ 双级微缩放      │ Blackwell    │ 最激进（4-bit）   │
└─────────────────┴────────────────┴──────────────┴──────────────────┘

Per-Tensor FP8：
  格式：Hybrid（E4M3 输入/权重，E5M2 梯度）
  两种变体：
    延迟缩放（Delayed）：用历史窗口 amax → 性能好但精度差 → 不推荐
    当前缩放（Current/Live）：JIT 计算 amax → 精度更好 → 推荐
  Hopper 限制：仅支持 TN 布局 FP8 GEMM → 需存储转置 FP8 激活
  Blackwell 改进：支持所有 FP8 GEMM 布局 → 无需转置版本 → 进一步节省内存

Blockwise FP8（Hopper 推荐）：
  格式：E4M3 for all（激活/梯度 1×128 tiles，权重 128×128 blocks）
  已在 DeepSeek-V3、Minimax-M2、Ant Ling-2.0 等大规模生产验证
  TransformerEngine 提供高度优化的量化/GEMM 内核

MXFP8（Blackwell 推荐）：
  格式：1×32 粒度 + E8M0 缩放因子
  Blackwell 第五代 Tensor Core 原生支持 → 更高精度 + 更好性能
  进行中优化：分组量化、激活+量化融合到 GEMM、全流程 CUDA Graph 化

NVFP4（最激进）：
  格式：E2M1 + 双级微缩放（Per-Tensor FP32 + Per-Block E4M3）
  Block 大小：16 个连续元素
  三项稳定性技术：
    1. RHT（Random Hadamard Transform）：减少权重梯度中的异常值影响
    2. 2D 缩放：16×16 权重块缩放，减少前反向量化失配
    3. 随机舍入：梯度 FP4 转换时减少舍入偏置
```

### 7.4 FP8/FP4 Primary Weights（消除冗余存储）

```
传统方案：三层参数层级
  FP32 主权重 → BF16 模型权重 → FP8/FP4 计算权重
  → BF16 副本造成冗余内存开销

原生 FP8/FP4 方案：跳过 BF16 中间层
  FP32 主权重 → 直接量化 → FP8/FP4 计算权重
  → 节省内存 + 加速参数 AllGather

实现步骤（Per-Tensor / Current Scaling）：
  Step 1：从本地主权重获取 local abs-max（无分片则设 0）
  Step 2：AllReduce 获取 global abs-max
  Step 3：用 global abs-max 和主权重执行部分量化

Blockwise 方案特殊处理：
  abs-max 在 2D 块上计算 → 需专用内核感知权重 2D 布局
  + 主权重与原始权重的对应关系
```

### 7.5 MoE 特有的低精度挑战

```
挑战 1：动态形状对齐（Padding / Unpadding）
  FP8/FP4 GEMM 要求维度对齐到特定倍数：
    Per-Tensor / Blockwise → 16
    MXFP8 / NVFP4 → 32
    NVFP4 分组量化 → 128（线程块不能跨专家边界）
  
  解决方案：
    • Routing Map Padding：填充路由表而非 token → 少量额外 token
    • Padding 融合进 Permutation：避免一轮全局内存读写

挑战 2：分组量化（Grouped Quantization）
  朴素方案：逐专家调用量化内核 → 大量小内核 → CPU 开销大
  优化方案：分组量化内核融合所有专家输入到单内核
    → 减少 CPU 开销 + 提高 GPU 利用率 + CUDA Graph 兼容

挑战 3：NVFP4 量化融合
  NVFP4 量化不只是"缩放+类型转换"：
    需集成 RHT + 2D 缩放 + 随机舍入

  RHT 融合（最关键）：
    分离实现需额外 BF16 全局内存读写 → 带宽成本高
    融合后单内核完成 Hadamard + FP4 量化

  前向时生成两份 FP4 输出：
    1. 标准 FP4 → 当前前向 GEMM
    2. 转置 + RHT + FP4 → 存储供反向 Wgrad 使用
    → 原始高精度输入可立即释放

  Per-Expert 第二级缩放：
    MoE 中 per-tensor → per-expert（保持数值稳定性）
    利用 128 对齐的 tokens-per-expert 保证
    实现 CUDA Graph 安全的分组 Hadamard-amax 内核
```

---

## 8. 长上下文 MoE 训练

### 8.1 计算特征变化

```
关键洞察：MoE 和 Attention 的计算复杂度不同

  MLP 组件（含 MoE）：O(s) 线性 → 与序列长度线性增长
  Attention SDPA：    O(s²) 二次 → 与序列长度二次增长

计算占比随序列长度的变化：
  4K tokens：  MoE 占 ~59.4%（MoE 主导）
  64K tokens： SDPA 占 ~69.7%（Attention 主导）

长上下文训练重塑了三面墙的相对重要性：
  → 第 4 节的 MoE 优化仍有价值，但不再针对主瓶颈
  → 优化焦点转向内存和通信
  → SDPA 本身已高度优化（FlashAttention, cuDNN），不成为性能瓶颈

SDPA cuDNN 性能（DeepSeek-V3）：
  ┌──────────┬───────────┬───────────┬────────────┐
  │ 平台      │ SeqLen    │ FWD TFLOPS │ BWD TFLOPS │
  ├──────────┼───────────┼───────────┼────────────┤
  │ Hopper   │ 4,096     │ 553       │ 422        │
  │ Hopper   │ 16,384    │ 638       │ 523        │
  │ Blackwell│ 4,096     │ 1,324     │ 1,083      │
  │ Blackwell│ 16,384    │ 1,698     │ 1,298      │
  └──────────┴───────────┴───────────┴────────────┘
```

### 8.2 激活内存管理

```
核心原则：保持 sub-sequence 长度常数（通常 4096 或 8192）

1. CP + TP（最主要手段）
   CP × TP 随序列长度等比扩展 → 每设备内存接近基线
   → 工作负载接近短上下文训练，大部分短上下文优化仍适用

2. 优化器 CPU Offloading
   优化器状态可达数十 GB → offload 到 CPU 几乎完全回收
   DeepSeek-V3 on 256 H100（~50% MFU, ≥16K seq）：worst-case 开销 ~2%
   长上下文训练通常值得启用

3. 选择性重算
   关键：模块级选择性（而非全层重算）
   长上下文下 SDPA 计算主导 → 重算 SDPA 代价太高
     DeepSeek-V3 64K seq：
       SDPA 贡献 72% 总计算
       重算 SDPA → +18% 计算开销，-16% 性能，仅省 9 GB
       重算非 SDPA 模块 → 省 89.8 GB，性能影响更低
   → 建议禁用 core attention recomputation，优先重算其他模块
```

### 8.3 CP vs TP 选择

```
两者都减少激活内存，但通信模式不同：

              CP (P2P)           CP (A2A)           TP
权重        复制                复制                分片（减少参数内存）
SDPA        序列分片+P2P环      头分片              头分片
通信模式    Ring P2P 重叠计算   All-to-All          线性层集合通信
特点        通信自然重叠        避免多步环交换       提高 SDPA 效率

实践指导：
  节点内：TP 优先（通信快 + 内存收益强）
         可组合 all-to-all CP 提高 SDPA 效率
  跨节点：P2P CP 优先（TP 通信开销增大，P2P 重叠仍有效）

Megatron-Core 支持层次化 CP（hierarchical CP）：
  节点内 all-to-all CP + TP → 提高 SDPA 性能 + 减少参数内存
  跨节点 P2P CP → 保持通信-计算重叠
```

### 8.4 变长序列支持

```
Packed Sequences（THD 格式）：
  传统 SBHD 格式 → 需 padding 到最大长度 → 浪费 40-60% 计算
  THD 格式 [total_tokens, num_heads, head_dim] → 多序列拼接无 padding
  通过 cumulative sequence length 跟踪序列边界

效果：RL 训练中减少 40-60% 内存，吞吐提升 1.5-2×

Dynamic Context Parallelism（Dynamic-CP）：
  问题 1：DP 不平衡 — 即使 packed token 总数相同，注意力工作量因
          O(s²) 复杂度差异而不同
  问题 2：CP 低效 — 静态 CP 按最长序列设置 → 短序列被迫使用不必要的大 CP

  解决方案：按 micro-batch 自适应选择 CP size
    • 长序列 → 大 CP（保证内存安全）
    • 短序列 → 小 CP（最大化计算强度，减少通信）
    • 预构建多个 CP 组 → 运行时选择 → 无动态创建通信组开销
    • 求解器异步运行 → 与训练迭代重叠
    • 蛇形排序微批次：按注意力成本先升后降 → 减少同步气泡

  效果：高度不平衡序列长度分布下 35-60% 端到端性能提升

长上下文效果汇总：
  DeepSeek-V3 (256 H100, 256K seq)：达短上下文 MFU 的 88%
  Qwen3-235B (256 H100, 256K seq)：达短上下文 MFU 的 129%
    （超 100% 因为长序列下 SDPA kernel 效率更高）
```

---

## 9. 生产特性

### 9.1 负载均衡与 Token Dropping

```
三种负载均衡策略：
  1. 辅助损失（Auxiliary Loss）：可微惩罚项，阻止所有 token 路由到少数专家
  2. 专家选择路由（Expert Choice）：最优传输问题形式化
  3. 无辅助损失均衡（Aux-Loss-Free）：可学习专家偏置项，基于历史负载动态调整

两种 Dispatch 策略：
  Dropless（默认）：所有路由 token 均处理 → 最大模型表达力，但工作负载可变
  Droppable：专家容量上限 → 超限 token 走残差连接 → 可预测内存
    + Pad-to-Max：固定形状 → 启用 Full CUDA Graphs
```

### 9.2 共享专家 & Latent MoE

```
共享专家（Shared Expert）：
  所有 token 均通过 → 提供一致的基线能力
  可启用重叠（--moe-shared-expert-overlap）→ 与 dispatch/compute/combine 并行

Latent MoE：
  核心思想：在 dispatch 前压缩，在 combine 后解压
    W↓ ∈ R^{ℓ×d} 降维 → All-to-All 通信量降 d/ℓ 倍
    Expert 在压缩空间 ℓ 操作 → 权重大小也降 d/ℓ 倍
    W↑ ∈ R^{d×ℓ} 升维
    Router 仍在全维度 d 操作（保证路由质量）

  两种模式：
    ℓ-MoEeff：E 放大 α 倍，K 不变 → 降推理成本
    ℓ-MoEacc（推荐）：E 和 K 均放大 α 倍 → 组合空间指数扩大 → 同成本更高精度

  已被 NVIDIA Nemotron-3 Super/Ultra 采用
```

### 9.3 其他生产特性

```
分布式检查点：
  并行无关保存：ShardedTensor 描述符编码全局形状/偏移/分片模式
  全并行保存：每 rank 独立写本地分片 → 无协调瓶颈
  任意并行重新配置：TP=2,EP=4 保存 → TP=4,EP=8 加载 → 无需离线转换
  后端：Zarr（默认）/ PyTorch Distributed

弹性非对称 VPP：
  不要求均匀层分配 → 不同 virtual stage 可有不同数量/类型的层
  DeepSeek-V3 示例 (PP=16, VPP=2)：
    Rank 0:  embedding + 3× dense decoder | 2× decoder
    Rank 1-13: 2× decoder | 2× decoder
    Rank 14: 2× decoder | MTP
    Rank 15: 2× decoder | loss

Upcycling：
  Dense → MoE 转换，无需从零训练
  MLP 权重在中间维度切片+复制 → 初始化后 MoE 输出与 Dense 相同

Multi-Token Prediction (MTP)：
  每个位置预测多个连续 token → 密集化监督信号
  通过隐状态转换保持因果依赖 → 加速收敛
  推理时回退到单 token 预测

Muon 优化器：
  矩阵感知优化 → 正交化整个权重矩阵
  相比 AdamW 显著减少训练步数
  MuonClip 处理 query-key 点积无界增长（防止注意力爆炸）
```

---

## 10. 性能评估

### 10.1 实验设置与关键结果

```
基准模型：DeepSeek-V3-685B + Qwen3-235B（细粒度 MoE，压力测试三面墙）
硬件：GB300 / GB200 / H100
软件：Megatron-Core v0.16 + TransformerEngine（最新）

完整优化栈：
  • FP8：MXFP8 (Blackwell) / Blockwise FP8 (Hopper)
  • 内存：选择性重算 + 激活/优化器 offload
  • 通信：HybridEP (GB200 NVL) / DeepEP (H100) + EP 重叠
  • 计算：Grouped GEMM + 内核融合 + CUDA Graphs

主要结果（SeqLen=4096, Force-Balanced Routing）：
  ┌──────────────┬────────┬──────┬────────┬──────────┬──────────────┐
  │ 模型          │ 系统    │ GPU  │ 精度    │ TFLOPS   │ tokens/s/GPU │
  ├──────────────┼────────┼──────┼────────┼──────────┼──────────────┤
  │ DeepSeek-V3  │ GB300  │ 256  │ MXFP8  │ 1,233    │ 4,730        │
  │ DeepSeek-V3  │ GB200  │ 256  │ MXFP8  │ 1,048    │ 4,020        │
  │ DeepSeek-V3  │ GB200  │ 256  │ BF16   │ 857      │ 3,298        │
  │ DeepSeek-V3  │ H100   │ 1024 │ FP8-BLK│ 368      │ 1,412        │
  ├──────────────┼────────┼──────┼────────┼──────────┼──────────────┤
  │ Qwen3-235B   │ GB300  │ 256  │ MXFP8  │ 974      │ 6,583        │
  │ Qwen3-235B   │ GB200  │ 256  │ MXFP8  │ 919      │ 6,212        │
  │ Qwen3-235B   │ GB200  │ 256  │ BF16   │ 750      │ 5,100        │
  │ Qwen3-235B   │ H100   │ 256  │ BF16   │ 320      │ 2,132        │
  ├──────────────┼────────┼──────┼────────┼──────────┼──────────────┤
  │ Qwen3-235B   │ GB300  │ 128  │ MXFP8  │ 1,150    │ 1,556        │
  │ (长上下文)    │        │      │        │          │ (SeqLen=131K)│
  └──────────────┴────────┴──────┴────────┴──────────┴──────────────┘

关键发现：
  • GB200/GB300 token 吞吐量约为 H100 的 3×
  • FP8 显著提升性能（GB200 MXFP8 vs BF16: +22% for DSV3, +23% for Qwen3）
  • 长上下文 131K 仍达 1,150 TFLOPS/GPU
```

---

## 11. 性能最佳实践

### 11.1 系统化优化工作流（三阶段）

```
Phase 1：建立内存可行的并行配置
  ┌──────────┬──────────────┬──────────────┬────────────────┬──────────┐
  │ 策略      │ 峰值激活      │ 权重内存      │ 优化器状态      │ 通信开销  │
  ├──────────┼──────────────┼──────────────┼────────────────┼──────────┤
  │ TP       │ 1/d (with SP)│ 1/d          │ 1/d            │ 高       │
  │ EP       │ ~1 (负载相关) │ 1/d (MoE)    │ 1/d            │ 中       │
  │ PP       │ 1 (>1 w/VPP) │ 1/d          │ 1/d            │ 中       │
  │ CP       │ 1/d          │ 1            │ 1/d†           │ 中       │
  │ DP       │ 1            │ 1            │ 1/d†           │ 低       │
  └──────────┴──────────────┴──────────────┴────────────────┴──────────┘
  †需分布式优化器

  快速验证：--fake-init-process-group 在单 GPU 模拟分布式训练

Phase 2：选择最优并行策略
  指导原则：
    1. 最小化模型并行，最大化数据并行
    2. 保持 EP×TP 在 NVLink 域内
    3. 多节点扩展用 PP（而非跨节点 TP/EP）
    4. Expert 层优先 EP 而非 TP（更好 GEMM 效率 + 更低通信）
    5. 长序列 ≥8K 启用 CP；<4K 通常不值得

Phase 3：Profile 驱动的瓶颈优化
  
  内存瓶颈 → FP8 / 选择性重算 / Offloading / 精度优化器
  通信瓶颈 → DP 梯度重叠 / TP 通信重叠 / EP dispatcher / EP A2A 隐藏
  CPU 开销瓶颈 → 禁用 GC / 减少内核启动 / CUDA Graphs
  计算瓶颈 → Grouped GEMM / 内核融合 / FP8

  关键：三阶段是迭代的
    内存优化 → 启用更小并行度 → 回到 Phase 1
    Phase 3 优化有自身内存开销（EP 重叠需 buffer, CUDA Graph 需额外内存）
    → 可能需要回退早期决策
```

### 11.2 案例研究：DeepSeek-V3 on GB200 vs H100

```
最终配置对比：
  ┌──────────────────────┬──────────────────┬──────────────────┐
  │ 配置                  │ GB200 (256 GPU)  │ H100 (1024 GPU)  │
  ├──────────────────────┼──────────────────┼──────────────────┤
  │ TP / PP / EP         │ 1 / 4 / 64       │ 2 / 8 / 64       │
  │ VPP                  │ 4                │ 4                │
  │ GBS / MBS / SeqLen   │ 8192 / 1 / 4096  │ 8192 / 1 / 4096  │
  │ 精度                  │ MXFP8            │ FP8-Blockwise    │
  │ Dispatcher           │ HybridEP         │ DeepEP           │
  │ 重算                  │ mlp              │ mlp, mla_up_proj,│
  │                      │                  │ moe_act, layernorm│
  │ CUDA Graphs          │ ✓                │ —                │
  │ EP A2A Overlap       │ —                │ ✓                │
  │ TFLOPS/GPU           │ 1,048            │ 368              │
  └──────────────────────┴──────────────────┴──────────────────┘

为什么同一模型需要不同策略：

  GB200 (NVL72, 192GB/GPU)：
    大内存 → TP1/PP4（管线更短，气泡更少）
    EP64 全在 NVLink 域内 → 不需 EP 通信重叠
    FP8 加速暴露 CPU 瓶颈 → CUDA Graphs + 内核融合
    C2C 高带宽 → 优化器 offload 有效
    重算仅需 mlp

  H100 (NVL8, 80GB/GPU)：
    小内存 → TP2/PP8（更深管线）
    EP64 跨 8 个节点 → 必须 DeepEP + EP 通信重叠
    FP8 省下的内存恰好用于 EP 重叠的额外 buffer
    重算需更激进（mlp + mla_up_proj + moe_act + layernorm）

四条通用教训：
  1. 平台特性决定策略（内存大小、NVLink 拓扑、C2C 带宽）
  2. Parallel Folding 释放灵活性（独立优化 attention TP 和 expert EP）
  3. FP8 改变瓶颈（加速 GEMM + 减少内存 → 但放大 CPU 开销）
  4. 迭代优化（内存优化 → 启用通信重叠 → 暴露计算效率瓶颈）
```

---

## 12. MoE 在强化学习中的应用

```
RL Post-Training 的特殊挑战：

  1. 变长序列：最大 128K-1M，均值仅 1/2~1/4 最大值 → 长尾分布
  2. 内存 Offloading：训练/推理引擎共享 GPU → 需快速完全释放/恢复内存
  3. 在线权重导出：训练后快速导出到推理引擎可加载的格式
  4. 训练稳定性：推理和训练引擎使用不同优化内核 → 同参数不同 token 概率
     MoE 放大问题：同序列 token 可能路由到不同专家 → 偏差更大

Megatron-Bridge：
  HF ↔ Megatron 检查点快速双向转换
  被 veRL、Slime、NeMo RL 等 RL 框架使用

关键优化：
  • Packed Sequences + 打包感知动态批大小
  • 蛇形排序微批次（按注意力成本先升后降 → 减少同步气泡）
  • Dynamic-CP：按微批次自适应 CP 大小
  • CPU 优化器 Offloading：前反向时卸载，仅更新时加载回
  • FP16 训练路径：某些 RL 超参下 FP16 比 BF16 更稳定
  • Router Replay：推理时记录路由决策 → 训练时强制相同路由
    → 解耦路由变异和权重更新 → 更稳定优化轨迹
```

---

## 13. 总结

```
核心贡献：
  ┌────────────────────────────────────────────────────────────────┐
  │  Megatron-Core MoE = 完整系统栈（开源）                         │
  │                                                                │
  │  从两个根本挑战出发：                                            │
  │    1. 参数-计算不匹配 → 三面墙                                  │
  │    2. Dense-Sparse 不匹配 → 需要解耦并行                        │
  │                                                                │
  │  系统性解决方案：                                                │
  │    并行化 → Parallel Folding + 多维并行                         │
  │    内存  → 重算 + 高效排列 + Offload + 精度优化器 + FSDP        │
  │    通信  → DeepEP/HybridEP + 通信重叠                          │
  │    计算  → Grouped GEMM + 融合 + CUDA Graphs + Sync-Free       │
  │    精度  → FP8/FP4 选择性量化（跨三面墙收益）                    │
  │    长上下文 → CP/TP + 选择性重算 + Dynamic-CP                   │
  │    生产  → 负载均衡 + 检查点 + VPP + Upcycling + MTP + Muon    │
  │    RL   → Megatron-Bridge + Packed Seq + Router Replay         │
  │                                                                │
  │  标杆性能（Megatron-Core v0.16）：                               │
  │    DeepSeek-V3: 1,233 TFLOPS/GPU (GB300)                       │
  │                 1,048 TFLOPS/GPU (GB200)                       │
  │                   368 TFLOPS/GPU (H100)                        │
  │    Qwen3-235B:   974 TFLOPS/GPU (GB300)                        │
  │                   919 TFLOPS/GPU (GB200)                       │
  │                   320 TFLOPS/GPU (H100)                        │
  │    GB200/300 比 H100 吞吐量 ~3×                                │
  │                                                                │
  │  支持从研究原型到万亿参数生产模型的全链路                          │
  └────────────────────────────────────────────────────────────────┘
```