# MoE Parallel Folding: Heterogeneous Parallelism for Large-Scale MoE Training

> **arXiv:** [2504.14960](https://arxiv.org/abs/2504.14960) | **PDF:** https://arxiv.org/pdf/2504.14960  
> **发表时间:** 2025年4月  
> **机构:** NVIDIA  
> **框架:** Megatron-Core  
> **核心贡献:** H100 上 Mixtral 8x22B 达 **49.3% MFU**，支持 **1024 GPUs + 128K token 序列**

---

## 1. 核心问题：MoE 模型的并行策略困境

### 1.1 传统并行维度的局限

大规模 LLM 训练有四种基本并行策略，但 MoE 模型让这些策略变得更复杂：

```
传统 Dense LLM 并行：
  DP：数据并行（复制模型，分割数据）
  TP：张量并行（分割矩阵运算）
  PP：流水线并行（分割模型层）
  CP：序列/上下文并行（分割长序列）
  
  ↓ 对 Dense 模型，这 4 种并行的配置方式相对简单 ↓

MoE 模型新增的挑战：
  EP：专家并行（Expert Parallel）
  
  但 EP 和其他并行维度的交互非常复杂：
  - EP 和 TP 可以吗？→ Expert 参数的分割方式与 TP 不同
  - EP 和 PP 可以吗？→ Expert 在哪个 PP 阶段？多层间如何调度？
  - EP 和 CP 可以吗？→ 长序列 token 被 CP 分割，但 MoE 路由需要全局视角
```

### 1.2 已有方案的不足

| 方案 | 支持的并行维度 | 局限 |
|------|-------------|------|
| **DeepSpeed MoE** | DP + EP | 无 TP/PP，难以扩展到超大模型 |
| **Megatron-LM（早期）** | TP + PP + DP | EP 支持有限，无 CP，无灵活 MoE 调度 |
| **Tutel** | DP + EP | 无 TP/PP |
| **MoE Parallel Folding** | **TP + EP + CP + DP + PP** | ✅ 五维完整支持 |

### 1.3 MoE 特有的并行挑战

```
DeepSeek-V3 / Mixtral 类模型的结构：

每个 Transformer Block：
  ┌─── Attention Layer (MHA 或 GQA) ───┐
  └─── MoE FFN Layer ──────────────────┘

关键点：
  Attention 层和 MoE FFN 层有不同的最优并行策略！
  
  Attention 层：
    - 参数相对小
    - 计算密集，适合 TP（矩阵分割）
    - 序列维度可以 CP 分割
  
  MoE FFN 层：
    - Expert 数量多，适合 EP（专家分配到不同 GPU）
    - EP 与 TP 的交互需要特殊处理
    - 动态路由需要 All-to-All 通信

传统方法的矛盾：
  对整个 Block 用同一并行配置
  → Attention 最优配置 ≠ MoE 最优配置
  → 必有一个层的效率受损
```

---

## 2. MoE Parallel Folding 核心设计

### 2.1 核心思想：解耦 Attention 和 MoE 的并行策略

```
MoE Parallel Folding 的关键创新：

同一个 Transformer Block 内，Attention 和 MoE FFN 使用不同的并行配置！

例子（8 GPUs）：

Attention 层配置：TP=4, DP=2, EP=1, CP=1
  GPU 0,1,2,3 → TP 组 1（处理 batch 的一半）
  GPU 4,5,6,7 → TP 组 2（处理 batch 的另一半）

MoE FFN 层配置：TP=1, DP=1, EP=8, CP=1
  GPU 0 → Expert 0,1
  GPU 1 → Expert 2,3
  ...
  GPU 7 → Expert 14,15

两个层之间：通过 All-to-All 通信完成布局重映射（"Parallel Folding"）
```

### 2.2 五维并行的具体实现

#### 2.2.1 张量并行（TP）

```
在 Attention 层：
  Q, K, V 矩阵按 head 维度分割到 TP 组内
  每个 GPU 计算部分 heads，最后 All-Reduce 合并
  
  Attention: [batch, seq, d_model] → 分割为 [batch, seq, d_model/TP]
             TP 并行度 = 4 → 每 GPU 处理 d_model/4 的宽度

在 MoE FFN 层（Expert 内部）：
  Expert 本身可以 TP 并行（Expert FFN 足够大时有意义）
  Expert FFN: [d_model → d_expert] 按 d_expert 维度分割
```

#### 2.2.2 专家并行（EP）

```
Expert 分配到不同 GPU 组：
  设 EP=8：共 8 组，每组 1 个 GPU
  Expert 均匀分配：每 GPU 负责 (N_expert / EP) 个 Expert
  
  通信：All-to-All（每个 GPU 将 token 发到包含目标 Expert 的 GPU）
  
  EP 与 TP 的组合（TP_EP）：
    Expert 参数同时做 TP 分割（内部并行）和 EP 分割（跨 GPU 分配）
    允许超大 Expert 的训练（TP_EP = TP_per_expert × EP_groups）
```

#### 2.2.3 上下文并行（CP）

```
针对长序列（128K tokens）：
  CP=N → 将序列长度 L 分割为 N 段，每段 L/N tokens
  每个 CP 组内的 GPU 负责不同位置的 tokens
  
  Attention 中的 CP（Ring Attention）：
    每个 GPU 的 Q 与所有 GPU 的 K,V 做 Attention
    用 Ring-style All-to-All 传递 K,V
    
  MoE 中的 CP：
    各 CP 段的 tokens 独立路由到 Expert
    Expert 端按正常方式处理（无需感知 CP）
    
  关键：CP 与 EP 可以同时启用！
```

#### 2.2.4 流水线并行（PP）

```
将模型按层分割到多个 PP Stage：
  
  PP Stage 0：Layer 1-8   （GPU 0-3）
  PP Stage 1：Layer 9-16  （GPU 4-7）
  PP Stage 2：Layer 17-24 （GPU 8-11）
  PP Stage 3：Layer 25-32 （GPU 12-15）
  
  每个 Stage 内可以有独立的 TP/EP/DP/CP 配置！
  
  关键创新：
    同一 PP Stage 内，不同 Block 类型（Dense Block / MoE Block）
    可以有不同的 TP/EP 配置
    这就是 "Parallel Folding" 的精髓！
```

#### 2.2.5 数据并行（DP）

```
DP 作为最外层并行维度：
  DP=N → 复制 N 份整个模型（含所有其他并行维度）
  每份处理不同的数据批次
  梯度通过 All-Reduce 同步
  
  在 MoE 中，EP 通常与 DP 合并考虑：
    有效 DP = total_GPUs / (TP × PP × EP × CP)
    EP 组内的 GPU 实际上也做了某种形式的 DP
```

### 2.3 Parallel Folding 的通信开销分析

```
Attention → MoE FFN 的布局转换（Fold 操作）：

假设：
  Attention：TP=4, DP=2
  MoE FFN：EP=8, TP=1
  
  Token 的"物理位置"需要重新排列：
    Attention 后：token 按 DP 组分布（每 GPU 处理不同 token）
    MoE 前：token 需要路由到含目标 Expert 的 GPU
    
  Fold 操作 = All-to-All 通信（将 token 重分配）
  
  开销分析：
    每次 Fold 的通信量 ≈ batch_tokens × d_model × sizeof(dtype)
    但这与 MoE 本来就需要的 All-to-All（Expert Dispatch）合并！
    → 额外开销接近于零！
```

---

## 3. Token 调度机制

### 3.1 灵活 Token-Level 调度

```
传统 MoE 路由：Token → 固定 Expert（按 expert_id 分配到 GPU）

MoE Parallel Folding 的增强：
  支持更灵活的 Token 调度策略
  
  1. Top-K 路由（传统）：每 token 路由到 K 个 expert
  2. 负载均衡辅助损失（传统）：惩罚路由不均
  3. Expert Choice 路由（改进）：每个 Expert 选择固定数量的 token
  4. Shared Expert（DeepSeek-V3 风格）：
     部分 Expert 是"共享的"，所有 token 都会经过
     其余 Expert 是"路由的"，按 Top-K 选择
```

### 3.2 DeepSeek-V3 风格的路由兼容

```python
# MoE Parallel Folding 支持的路由模式（伪代码）
class FlexibleMoERouter:
    def __init__(self, n_routed_experts, n_shared_experts, top_k):
        self.n_routed = n_routed_experts
        self.n_shared = n_shared_experts
        self.top_k = top_k
        
    def forward(self, hidden_states):
        # Shared experts：所有 token 都经过（串联计算）
        shared_out = sum(
            expert(hidden_states) 
            for expert in self.shared_experts
        )
        
        # Routed experts：Top-K 选择
        router_logits = self.gate(hidden_states)
        topk_indices = router_logits.topk(self.top_k).indices
        routed_out = self.dispatch_and_compute(hidden_states, topk_indices)
        
        return shared_out + routed_out
```

---

## 4. 系统实现（基于 Megatron-Core）

### 4.1 Megatron-Core 的优势

```
Megatron-Core 为 MoE Parallel Folding 提供的基础设施：

1. 并行组管理：
   ProcessGroup(TP_ranks=[0,1,2,3])    # TP 通信组
   ProcessGroup(EP_ranks=[0,4,8,12])   # EP 通信组
   ProcessGroup(DP_ranks=[0,1,...])    # DP 通信组
   
2. 分布式 Attention：
   - Ring Attention（CP 支持）
   - Flash Attention 集成
   
3. Expert Dispatch 框架：
   - All-to-All 实现
   - 负载均衡 token 调度
   
4. 流水线调度：
   - 1F1B 调度
   - Interleaved 1F1B（减少 PP bubble）
```

### 4.2 关键实现细节

```python
# Megatron-Core MoE 层的并行配置（概念代码）
class MegatronMoELayer(MegatronModule):
    def __init__(self, config):
        super().__init__()
        
        # 独立配置 MoE 层的并行策略
        self.expert_parallel_group = get_expert_model_parallel_group()
        self.tensor_parallel_group = get_tensor_model_parallel_group()
        
        # Expert 数量 = 路由专家 + 共享专家
        n_local_experts = config.num_experts // expert_parallel_world_size()
        
        # 每个 GPU 的专家列表
        self.local_experts = nn.ModuleList([
            ExpertMLP(config) for _ in range(n_local_experts)
        ])
        
        # 路由器（包含 Top-K 和 expert choice 模式）
        self.router = MoERouter(config)
    
    def forward(self, hidden_states):
        # 1. 计算路由权重
        dispatch_weights, expert_indices = self.router(hidden_states)
        
        # 2. Fold：重排布局（从 Attention 的 TP 布局到 EP 布局）
        hidden_states = parallel_fold(hidden_states, 
                                       from_tp=self.attn_tp_size,
                                       to_ep=self.expert_ep_size)
        
        # 3. Expert Dispatch（All-to-All）
        dispatched = all_to_all_dispatch(hidden_states, expert_indices,
                                          expert_parallel_group=self.expert_parallel_group)
        
        # 4. 本地 Expert 计算
        expert_outputs = [
            self.local_experts[i](dispatched[i])
            for i in range(len(self.local_experts))
        ]
        
        # 5. Expert Gather（All-to-All 反向）
        gathered = all_to_all_gather(expert_outputs, 
                                      expert_parallel_group=self.expert_parallel_group)
        
        # 6. Unfold：恢复到 Attention 所需的布局
        output = parallel_unfold(gathered,
                                  from_ep=self.expert_ep_size,
                                  to_tp=self.attn_tp_size)
        
        return output
```

---

## 5. 性能实验结果

### 5.1 核心指标

| 指标 | 数据 | 说明 |
|------|------|------|
| **MFU（模型浮点利用率）** | **49.3%** | H100 上 Mixtral 8x22B |
| **GPU 规模** | **1024 GPUs** | 已验证的最大规模 |
| **序列长度** | **128K tokens** | 支持长上下文训练 |
| **对标系统** | DeepSpeed/Megatron | 同等模型规模下更高 MFU |

### 5.2 49.3% MFU 的含义

```
理解 MFU（Model FLOP Utilization）：
  
  理论峰值 FLOPS：
    H100 SXM5 BF16 Tensor Core = 1,979 TFLOPS
    1024 个 H100 = 2,026,496 TFLOPS
  
  实际 FLOPS（模型计算量）：
    Mixtral 8x22B，批量足够时：
    实际计算量 / 理论峰值 = MFU = 49.3%
  
  49.3% 代表什么：
    ✅ 优秀水平（Dense 模型通常 30~50%，MoE 模型通常 20~40%）
    ✅ MoE 模型达到 Dense 模型的 MFU 水平
    ✅ 说明并行策略几乎消除了通信瓶颈
  
  对比：
    早期 MoE 训练 MFU 常常 < 30%
    MoE Parallel Folding 将 MFU 提升到接近 Dense LLM 水平
```

### 5.3 并行配置的消融分析

```
不同并行配置的 MFU 对比（推测）：

配置 A：TP=4, EP=4, PP=4, DP=16
  → MFU ≈ 35%（EP 和 TP 冲突，通信开销大）

配置 B：TP=1, EP=16, PP=4, DP=16
  → MFU ≈ 40%（EP 充分，但缺少 Attention 的 TP 优化）

配置 C（Parallel Folding）：Attention用TP=4，MoE用EP=16，PP=4
  → MFU ≈ 49.3%（Attention 和 MoE 各用最优并行策略）

提升来源：
  Attention 用 TP → 减少 Attention 的内存需求，支持更大 batch
  MoE 用 EP → 减少 All-to-All 通信量
  Fold/Unfold → 两者间的通信与 All-to-All 合并，额外开销 < 1%
```

---

## 6. 与 FSDP2 / PyTorch 的关系

### 6.1 Megatron-Core vs FSDP2

```
两种分布式训练范式的对比：

Megatron-Core（MoE Parallel Folding 的基础）：
  优点：极度优化的通信原语，TP/PP/EP/CP 深度集成
  适合：超大规模训练（100B+），需要最高效率
  挑战：代码复杂，与 PyTorch 生态集成成本高

PyTorch FSDP2：
  优点：简单易用，与 PyTorch 生态无缝集成
  适合：中等规模（10B~100B），快速迭代
  挑战：对 MoE 的支持不如 Megatron-Core 完善

MoE Parallel Folding 对 FSDP2 用户的借鉴：
  → 解耦 Attention 和 MoE 层并行策略的思想可以在 FSDP2 中实现
  → 使用 FSDP2 的 Expert 分组 + 手动管理 All-to-All
```

### 6.2 在 PyTorch 生态中的实现思路

```python
# FSDP2 + MoE 的简化实现（概念）

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

class FSDPMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Attention 层：用 FSDP（类似 DP + ZeRO-3）
        self.attention = FSDP(AttentionLayer(config),
                               sharding_strategy=ShardingStrategy.FULL_SHARD)
        
        # MoE 层：用 Expert Parallel（不用 FSDP，用手动 EP）
        # 每个 GPU 只存本地 Expert
        local_expert_ids = range(
            local_rank * experts_per_gpu,
            (local_rank + 1) * experts_per_gpu
        )
        self.local_experts = nn.ModuleList([
            ExpertMLP(config) for _ in local_expert_ids
        ])
        
        # Fold/Unfold 操作（简化版）
        self.fold = ParallelFold(
            from_world_size=dist.get_world_size(),
            to_ep_size=config.ep_size
        )
    
    def forward(self, x):
        # Attention（FSDP 管理）
        attn_out = self.attention(x)
        
        # Fold to EP layout
        ep_layout = self.fold(attn_out)
        
        # Expert Dispatch + Compute + Gather
        routed = all_to_all_dispatch(ep_layout, ...)
        expert_out = [self.local_experts[i](routed[i]) for i in ...]
        gathered = all_to_all_gather(expert_out, ...)
        
        # Unfold back
        return self.fold.inverse(gathered)
```

---

## 7. 对大规模 MoE 训练的指导意义

### 7.1 并行策略选择指南

```
基于 MoE Parallel Folding 的经验，推荐的并行配置策略：

集群规模：N GPUs，模型：MoE with E Experts

Step 1: 确定 PP（流水线并行）
  PP = max(1, N / 64)    （每个 PP stage 约 64 GPUs）
  
Step 2: 确定 EP（专家并行）
  EP = min(E, N/PP)      （Expert 数量限制，每 stage 内 GPU 数限制）
  
Step 3: 确定 TP（张量并行）
  TP_attention = 4 或 8  （Attention 层固定使用 TP）
  TP_expert    = 1 或 2  （Expert 层通常不需要 TP）
  
Step 4: 确定 CP（上下文并行）
  CP = max(1, seq_len / 8192)  （序列超过 8K 时启用）
  
Step 5: DP = N / (PP × EP × TP_effective × CP)
```

### 7.2 通信量分析

```
各并行维度的通信类型和量：

TP：All-Reduce（每层 2 次）
  量：2 × batch × seq_len × d_model × (TP-1)/TP

EP：All-to-All（每个 MoE 层 2 次）
  量：2 × batch × seq_len × d_model / EP

CP：Ring All-to-All（Attention 内，每层 1 次）
  量：batch × seq_len × 2*d_kv × (CP-1)/CP

PP：P2P 通信（每个 stage 边界）
  量：batch × micro_seq_len × d_model

Fold/Unfold：等效于 All-to-All
  量：与 EP 的 All-to-All 合并，几乎无额外开销

最优配置的通信量：≈ O(batch × d_model)，不随规模 N 线性增长
```

---

## 8. 横向对比总结

| 论文 | 核心问题 | 主要手段 | 最大收益 | 实现难度 | 规模要求 |
|------|---------|---------|---------|---------|---------|
| **MoE Parallel Folding** (本文) | 多维并行冲突 | 解耦 Attention/MoE 并行 + 5D | **49.3% MFU** | ⭐⭐⭐⭐⭐ | 1024+ GPUs |
| **LAER-MoE** [2602.11686] | 负载不均衡 | FSEP 动态重排 | **1.69x 吞吐** | ⭐⭐⭐⭐ | 任意规模 |
| **FlowMoE** [2510.00207] | 调度串行化 | Chunk 流水线调度 | **57%** 时间减少 | ⭐⭐⭐ | 任意规模 |
| **SwiftMoE** [2504.19925] | 参数迁移开销 | 参数解耦 | **+30.5% 收敛** | ⭐⭐⭐ | 中等规模 |
| **MoEBlaze** [2601.05296] | 内存墙 | 数据结构+Kernel | **4x 局部** | ⭐⭐ | 任意规模 |
| **MemFine** [2511.21431] | 激活内存 | Chunk 调度重算 | **48% 内存减少** | ⭐⭐ | 任意规模 |
| **OmniMoE** [2602.05711] | 路由效率 | 原子专家+笛卡尔积 | **10.9x 推理** | ⭐⭐⭐ | 推理为主 |

---

## 9. 阅读建议

| 章节（推测） | 核心内容 | 阅读价值 |
|-----------|---------|---------|
| **Introduction** | MoE 并行策略冲突的形式化描述 | ⭐⭐⭐⭐⭐ |
| **Section 2: Background** | 5 种并行维度的回顾和交互分析 | ⭐⭐⭐⭐⭐ |
| **Section 3: Parallel Folding** | 解耦并行配置 + Fold/Unfold 操作 | ⭐⭐⭐⭐⭐ |
| **Section 4: Token Scheduling** | 灵活路由实现（Expert Choice + Shared） | ⭐⭐⭐⭐ |
| **Section 5: Implementation** | Megatron-Core 集成细节 | ⭐⭐⭐⭐⭐ |
| **Section 6: Evaluation** | MFU 实测 + 消融分析 | ⭐⭐⭐⭐⭐ |

---

## 延伸阅读

- 📄 **Megatron-LM** - 基础框架 → https://arxiv.org/abs/1909.08053
- 📄 **Ring Attention** - CP 的实现基础 → https://arxiv.org/abs/2310.01889
- 📄 **Mixtral 8x22B** - 实验用模型 → https://arxiv.org/abs/2401.04088
- 📄 **DeepSeek-V3** - 共享 Expert 的来源 → https://arxiv.org/abs/2412.19437
- 🔧 **Megatron-Core** - 源代码 → https://github.com/NVIDIA/Megatron-LM
- 📄 **LAER-MoE** - 互补的负载均衡优化 → https://arxiv.org/abs/2602.11686

---

*笔记整理于 2026-03-07，基于 arXiv 摘要及相关资料。完整 PDF：https://arxiv.org/pdf/2504.14960*
