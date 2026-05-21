# veScale-FSDP: Flexible and High-Performance FSDP at Scale

> **arXiv:** [2602.22437](https://arxiv.org/abs/2602.22437) | **PDF:** https://arxiv.org/pdf/2602.22437  
> **发表时间:** 2026年2月  
> **机构:** ByteDance（veScale 团队）  
> **作者:** Zezhou Wang, Youjie Li, Zhiqi Lin, Jiacheng Yang, Cong Xie, Guanyu Feng, Zheng Zhong, Ziyue Huang, Hongyu Zhu, Zhi Zhang, Yanghua Peng, Xin Liu  
> **核心贡献:** 重新设计 FSDP 系统，支持灵活分片格式 RaggedShard，**吞吐提升 5~66%，内存减少 16~30%，扩展至数万 GPU**

---

## 1. 为什么需要重新设计 FSDP？

### 1.1 FSDP 的设计假设正在失效

```
传统 FSDP（ZeRO）的核心假设：
  参数张量可以按「元素」或「行」均匀切分
  → 每个 GPU 持有相同大小的参数 shard
  → All-Gather / Reduce-Scatter 操作规整对称

这个假设在以下新场景中被打破：

场景 1：Block-wise 量化训练（如 Gemini、Kimi K2 使用）
  ┌────────────────────────────────────────┐
  │  Weight Matrix W [4096 × 4096]         │
  │                                        │
  │  block (0,0) │ block (0,1) │ ...       │
  │  128×128     │ 128×128     │           │
  │  ────────────┼─────────────┼───        │
  │  block (1,0) │ block (1,1) │ ...       │
  └────────────────────────────────────────┘
  
  Block-wise 量化要求：每个 block 的参数必须在同一个 GPU 上！
  （因为量化 scale factor 是 per-block 的）
  
  传统 FSDP 按元素/行切分 → block 被切碎 → 量化失效 ❌

场景 2：非元素级优化器（Shampoo、Muon、SOAP 等）
  传统 Adam：每个参数独立更新 → 按元素切分完全没问题
  
  Shampoo：维护 Kronecker 因子矩阵（L ∈ R^{d×d}，R ∈ R^{d×d}）
  → 更新时需要整个 Kronecker 因子
  → 如果参数被按行切分，Kronecker 因子也被切碎
  → 矩阵运算变成多次通信后才能完成 ❌
  
  Muon（最近被 Kimi K2 使用）：
  → 类似问题，需要整个子矩阵做 Newton-Schulz 迭代
  → 传统 FSDP 切分格式不兼容 ❌
```

### 1.2 传统 FSDP 的性能限制

```
现有 FSDP 实现（PyTorch FSDP2、DeepSpeed ZeRO）在大规模下的问题：

问题 1：通信效率
  All-Gather / Reduce-Scatter 的碎片化：
    参数更新后立即 All-Gather → 可能造成通信 burst
    没有系统性的通信调度优化
    
问题 2：内存效率
  shard 的粒度固定（整层或整个模块）
  无法根据实际内存压力动态调整
  
问题 3：扩展性天花板
  在数万 GPU 规模下，现有实现的通信 overhead 显著增大
  实测：PyTorch FSDP2 在 8K+ GPU 时效率明显下降
```

---

## 2. veScale-FSDP 的核心创新

### 2.1 RaggedShard：灵活分片格式

```
核心思想：shard 不再要求"等大"，支持任意形状的分片

传统 FSDP（固定格式）：
  GPU 0: W[0:512, :]     ← 固定按行均分
  GPU 1: W[512:1024, :]
  GPU 2: W[1024:1536, :]
  GPU 3: W[1536:2048, :]

RaggedShard（灵活格式）：
  GPU 0: W[block(0,0), block(0,1), block(1,0)]   ← 按 block 划分
  GPU 1: W[block(0,2), block(0,3), block(1,1)]
  GPU 2: W[block(1,2), block(1,3), block(2,0)]
  GPU 3: W[block(2,1), block(2,2), block(2,3)]
  
  每个 GPU 持有的 shard 大小可以不同（ragged = 参差不齐）
  但保证：每个 block 的所有参数在同一个 GPU → 量化兼容 ✅

RaggedShard 的数据结构：
  shard_info = {
      'shapes': [(block_h, block_w), ...],    # 每个 block 的形状
      'offsets': [(row, col), ...],           # block 在原始矩阵中的位置
      'device': gpu_id,                       # 分配到哪个 GPU
  }
```

### 2.2 Structure-Aware Planning Algorithm（结构感知规划算法）

```
问题：给定模型参数和优化器需求，如何自动决定最优的 RaggedShard 分配？

这是一个优化问题：
  minimize:   max_gpu( memory_usage(gpu_i) )      ← 峰值内存均衡
  subject to:
    每个 block 完整分配到一个 GPU              ← 量化约束
    Kronecker 因子的 block 与参数 block 同 GPU  ← 优化器约束
    总通信量 ≤ C_max                            ← 通信约束

规划算法的步骤：

Step 1：分析模型结构
  → 识别 block-wise 量化的 block 边界
  → 识别 Shampoo/Muon 的 Kronecker 因子依赖

Step 2：构建依赖图
  节点：参数 block
  边：必须在同一 GPU 的约束（co-location constraint）
  
Step 3：图分区（类 METIS 算法）
  目标：最小化跨 GPU 边（通信）
  约束：每个连通分量（co-location group）整体分配
  
Step 4：负载均衡调整
  如果某些 GPU 内存超过阈值，局部调整分配方案
```

### 2.3 通信优化

```
veScale-FSDP 的通信改进：

改进 1：Lazy All-Gather（懒惰 All-Gather）
  传统：层结束 → 立即 All-Gather 参数
  veScale-FSDP：延迟 All-Gather，等到真正需要时才触发
  → 与前向计算重叠，减少等待
  
改进 2：Hierarchical Reduce-Scatter（分层 Reduce-Scatter）
  传统：所有 GPU 参与一次大 Reduce-Scatter
  veScale-FSDP：
    节点内 → NVLink Reduce-Scatter（高带宽）
    节点间 → IB Reduce-Scatter（较低带宽）
  → 减少跨节点通信量 ~40%

改进 3：通信-计算 Overlap（类 Comet 但更轻量）
  Block-level 通信调度：
    Block A 的 Reduce-Scatter 与 Block B 的反向传播重叠
    不需要 Tile 级的 Warp 专用化（更容易实现）
```

### 2.4 内存管理优化

```python
# veScale-FSDP 的内存管理（伪代码）

class veScaleFSDP:
    def __init__(self, model, sharding_plan):
        # 根据 RaggedShard 计划初始化
        self.sharding_plan = sharding_plan
        self.param_shards = self._init_shards(model, sharding_plan)
        
        # 内存池：预分配 All-Gather buffer
        self.ag_buffer_pool = BufferPool(
            sizes=[shard.full_size for shard in self.param_shards],
            policy='lru'  # 最近最少使用
        )
    
    def forward(self, x):
        for layer in self.model.layers:
            # All-Gather：从 shard 恢复完整参数
            full_params = self.all_gather(layer)
            
            # 前向计算
            out = layer(x, full_params)
            
            # 提前释放（而非等到反向传播结束）
            self.release_params(full_params)
            
            x = out
        return x
    
    def _init_shards(self, model, plan):
        """根据 RaggedShard 计划切分参数"""
        shards = {}
        for param_name, shard_spec in plan.items():
            param = model.get_parameter(param_name)
            # 按 block 切分（而非按行切分）
            shard = extract_blocks(param, shard_spec['block_indices'])
            shards[param_name] = shard.to(shard_spec['device'])
        return shards
```

---

## 3. 支持新型优化器：Shampoo 和 Muon

### 3.1 Shampoo 优化器与 FSDP 的兼容

```
Shampoo 更新规则：
  对参数 W ∈ R^{m×n}：
    L = W @ W^T + ε·I     ← 左 Kronecker 因子，R^{m×m}
    R = W^T @ W + ε·I     ← 右 Kronecker 因子，R^{n×n}
    
    更新：ΔW = L^{-1/4} @ gradient @ R^{-1/4}

问题：计算 L^{-1/4} 需要整个 L 矩阵
  传统 FSDP：W 被切分 → L 也被切分 → 无法单独在一个 GPU 上计算 L^{-1/4}

veScale-FSDP 的解决：
  RaggedShard 确保：对于每个"Shampoo block"（子矩阵），
  其对应的 L_block 和 R_block 也在同一个 GPU
  
  → 每个 GPU 独立计算自己负责的 block 的 L^{-1/4}
  → 不需要 All-Gather Kronecker 因子
  → 通信量不变，但 Shampoo 可以正确工作 ✅
```

### 3.2 Muon 优化器支持

```
Muon 的核心操作：Newton-Schulz 迭代
  目标：计算 W / ||W||_F * nsharpener(W / ||W||_F)
  
  nsharpener 是矩阵函数，需要整个子矩阵参与
  
传统 FSDP 问题：
  W 被按行切分 → 每个 GPU 只有 W 的一部分
  → Newton-Schulz 需要通信才能完成 ❌

veScale-FSDP 解决：
  按 Muon 的操作粒度（"Muon block"）切分
  每个 GPU 持有完整的 Muon block
  → Newton-Schulz 在本地完成，无额外通信 ✅
  → Kimi K2 等使用 Muon 的模型可以高效训练
```

---

## 4. 性能实验结果

### 4.1 核心性能指标

| 对比维度 | veScale-FSDP vs PyTorch FSDP2 | veScale-FSDP vs DeepSpeed ZeRO3 |
|---------|------------------------------|--------------------------------|
| **吞吐量** | **+5~66%** | **+15~50%** |
| **内存使用** | **-16~30%** | **-10~25%** |
| **扩展性** | 数万 GPU，接近线性 | — |

> **注：** 5% 是小模型场景的下限，66% 是大规模 + 非元素级优化器场景的上限

### 4.2 不同配置下的效果

```
场景 A：标准训练（Adam 优化器，无量化）
  吞吐提升：+5~15%
  原因：通信优化 + 内存管理更高效
  
场景 B：Block-wise 量化训练（INT8/INT4）
  吞吐提升：+30~50%（因为传统 FSDP 根本无法正确运行！）
  原因：RaggedShard 使量化真正可行，同时避免冗余通信

场景 C：Shampoo/Muon 优化器 + 大模型
  吞吐提升：+40~66%
  原因：传统 FSDP 需要额外 All-Gather Kronecker 因子，veScale-FSDP 不需要

场景 D：超大规模（8K+ GPU）
  线性扩展：效率损失 < 5%（传统 FSDP2 损失约 15~20%）
```

### 4.3 内存节省分解

```
16~30% 内存节省来源：

1. RaggedShard 减少 padding：     +8%
   （传统均匀切分需要 pad 到相同大小，RaggedShard 不需要）

2. Kronecker 因子不再需要全量：   +10%
   （Shampoo 的 L/R 矩阵不用全 All-Gather）

3. Lazy All-Gather 减少峰值：     +5%
   （不再同时在内存中持有多个层的完整参数）

4. Buffer Pool 复用：              +5%
   （All-Gather buffer 在层间复用，而非每层单独分配）
```

---

## 5. 与现有 FSDP 实现的深度对比

### 5.1 功能对比

| 功能 | PyTorch FSDP2 | DeepSpeed ZeRO3 | **veScale-FSDP** |
|------|-------------|-----------------|-----------------|
| **标准 Adam** | ✅ | ✅ | ✅ |
| **Block-wise 量化训练** | ❌ | ❌ | **✅** |
| **Shampoo 优化器** | ⚠️ 低效 | ⚠️ 低效 | **✅ 原生支持** |
| **Muon 优化器** | ⚠️ 低效 | ⚠️ 低效 | **✅ 原生支持** |
| **万卡扩展性** | ⚠️ 下降 | ⚠️ 下降 | **✅ 近线性** |
| **自定义分片格式** | ❌ | ❌ | **✅ RaggedShard** |

### 5.2 设计理念对比

```
PyTorch FSDP2：
  设计目标：易用性 + 与 PyTorch 生态兼容
  分片方式：按元素/行（element-wise / row-wise），固定
  局限：无法支持 block-structured 计算

DeepSpeed ZeRO3：
  设计目标：极致内存效率
  分片方式：全参数分片，跨 GPU
  局限：优化器状态分片策略固定，不支持 block 约束

veScale-FSDP：
  设计目标：灵活性 + 高性能 + 新型优化器支持
  分片方式：RaggedShard（结构感知，灵活块划分）
  优势：同时满足量化、优化器、通信三方面需求
```

---

## 6. 对工程实践的启示

### 6.1 什么时候应该用 veScale-FSDP？

```
强烈推荐使用 veScale-FSDP 的场景：
  ✅ 使用 Shampoo、Muon、SOAP 等二阶优化器
  ✅ 使用 block-wise 量化（INT8/INT4 量化感知训练）
  ✅ 超大规模训练（> 4096 GPUs）
  ✅ 训练 Gemini、Kimi K2 类架构的模型

PyTorch FSDP2 仍然足够的场景：
  ✅ 标准 AdamW 训练
  ✅ 1024 GPU 以内的中等规模
  ✅ 追求与 PyTorch 生态的简单集成
```

### 6.2 集成示例

```python
# veScale-FSDP 使用方式（与 PyTorch FSDP2 API 高度兼容）

from vescale_fsdp import FSDP, ShardingPlan, RaggedShardConfig

# 1. 定义 sharding 策略（自动规划或手动指定）
sharding_config = RaggedShardConfig(
    block_size=128,              # 量化 block 大小
    optimizer_type='shampoo',    # 告知优化器类型，自动处理 Kronecker 因子
    balance_factor=0.1,          # 负载均衡容忍度（0.1 = 允许 10% 不均衡）
)

# 2. 自动规划（推荐）
plan = ShardingPlan.auto(model, sharding_config)

# 3. 包装模型（与 FSDP2 API 相同）
model = FSDP(model, sharding_plan=plan)

# 4. 配合 Shampoo 优化器
optimizer = DistributedShampoo(
    model.parameters(),
    lr=1e-3,
    # veScale-FSDP 会自动处理分布式 Kronecker 因子
)

# 训练循环与标准 PyTorch 完全相同
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 6.3 迁移成本

```
从 PyTorch FSDP2 迁移到 veScale-FSDP：

标准 Adam 场景：
  改动：仅替换 import 和初始化代码
  工作量：1~2 天
  收益：5~15% 吞吐提升

Shampoo/Muon 优化器场景：
  改动：替换 import + 移除之前的 workaround 代码
  工作量：3~5 天
  收益：40~66% 吞吐提升（相当于直接解锁了一个方向）

Block-wise 量化训练场景：
  改动：替换 import + 适配量化 block 大小配置
  工作量：1~2 周
  收益：让量化感知训练真正 work
```

---

## 7. 与其他论文的关系

### 7.1 在分布式训练优化栈中的位置

```
┌────────────────────────────────────────────────────────┐
│ 应用层：模型架构 + 优化器（Shampoo, Muon, Gemini 等）  │
├────────────────────────────────────────────────────────┤
│ 分片层：veScale-FSDP (RaggedShard) ← 这里              │
│         PyTorch FSDP2 / DeepSpeed ZeRO3               │
├────────────────────────────────────────────────────────┤
│ 并行策略层：MoE Parallel Folding（5D并行）              │
├────────────────────────────────────────────────────────┤
│ MoE 优化层：LAER-MoE, MoEBlaze, Comet                  │
├────────────────────────────────────────────────────────┤
│ 通信底层：NCCL, DeepEP, RDMA                           │
└────────────────────────────────────────────────────────┘
```

### 7.2 与 MoE 相关论文的协作

| 论文 | 与 veScale-FSDP 的关系 |
|------|----------------------|
| **LAER-MoE** | LAER-MoE 处理 Expert 分片，veScale-FSDP 处理 Dense 参数分片，可叠加 |
| **Comet** | Comet 做 Tile 级 Overlap，veScale-FSDP 做分层 Reduce-Scatter，互补 |
| **MoE Parallel Folding** | Parallel Folding 的 5D 并行建立在正确的 sharding 之上，veScale-FSDP 可作为底层 |
| **MegaScale-MoE** | MegaScale-MoE 是系统工程，veScale-FSDP 是框架层，风格不同但互补 |

---

## 8. 核心论文贡献总结

```
veScale-FSDP 的三大核心贡献：

1. RaggedShard —— 解决「灵活性」问题
   传统：参数必须等大切分
   现在：按任意 block 结构切分
   
2. Structure-Aware Planning —— 解决「自动化」问题
   传统：用户手动指定分片方式
   现在：自动分析优化器和量化需求，生成最优计划

3. 通信+内存优化 —— 解决「性能」问题
   分层 Reduce-Scatter + Lazy All-Gather
   5~66% 吞吐提升，16~30% 内存节省
```

---

## 9. 阅读建议

| 章节 | 核心内容 | 阅读价值 |
|------|---------|---------|
| **§1 Introduction** | Block-wise 量化和非元素级优化器的问题形式化 | ⭐⭐⭐⭐⭐ |
| **§3 RaggedShard** | 灵活分片格式的设计与实现 | ⭐⭐⭐⭐⭐ |
| **§4 Planning Algorithm** | 结构感知规划算法 | ⭐⭐⭐⭐⭐ |
| **§5 Communication** | 分层通信优化 | ⭐⭐⭐⭐ |
| **§6 Evaluation** | 对比实验：支持 Shampoo/Muon 的性能数据 | ⭐⭐⭐⭐⭐ |

---

## 延伸阅读

- 🔧 **veScale GitHub** → https://github.com/volcengine/veScale
- 📄 **PyTorch FSDP2** → https://pytorch.org/docs/stable/fsdp.html
- 📄 **Shampoo 优化器** → https://arxiv.org/abs/1802.09568
- 📄 **Muon 优化器** → 搜索 "Muon optimizer Kimi K2"
- 📄 **ZeRO: Memory Optimizations** → https://arxiv.org/abs/1910.02054
- 📄 **Comet** (计算通信 Overlap) → MLSys'25

---

*笔记整理于 2026-03-07 | arXiv:2602.22437 | ByteDance veScale 团队*
