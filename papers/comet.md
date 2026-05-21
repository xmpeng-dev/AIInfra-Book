# Comet: Fine-grained Computation-Communication Overlapping for Mixture-of-Experts

> **发表:** MLSys '25  
> **机构:** ByteDance + 上海交通大学  
> **代码:** https://github.com/bytedance/flux  
> **场景:** MoE 分布式训练/推理的计算通信 Overlap  
> **核心贡献:** 细粒度 Tile 级别的计算-通信 Overlap，比 FlowMoE 更激进的融合策略。已在字节跳动万卡生产集群部署。

---

## 1. 核心问题：MoE 通信与计算的同步屏障

### 1.1 MoE All-to-All 的天然串行问题

```
MoE Forward Pass 的执行顺序：

传统方式（完全串行）：
───────────────────────────────────────────────────────→ 时间
[Gate计算] → [All-to-All Dispatch] → [Expert GEMM] → [All-to-All Gather] → 输出
             ↑████████ 等待 ████████↑               ↑████████ 等待 ████████↑
             GPU 完全空闲等通信                     GPU 完全空闲等通信

问题：
  - All-to-All Dispatch 延迟 ≈ 5~20ms（取决于网络）
  - Expert GEMM 延迟 ≈ 3~15ms（取决于 token 数和 Expert 大小）
  - 总 MoE 层延迟 = Gate + A2A_D + GEMM + A2A_G
                  ≈ 2 × A2A + GEMM（A2A 占 50~60%）
```

### 1.2 已有方案的局限

```
FlowMoE（[2510.00207]）的方案：
  在 Task 层面做调度，以 Tensor Chunk 为单位
  粒度：几十 MB 的 Tensor
  问题：chunk 本身还是串行，chunk 内部无法 overlap

Comet 的突破：
  粒度降到 GEMM Tile 级别（KB 量级）
  在一个 Tile 的计算完成后，立即启动下一个 Tile 的通信
  真正实现"计算覆盖通信"
```

---

## 2. Comet 核心设计

### 2.1 Tile 级别的流水线

```
Comet 的执行模型：

输入 Token 矩阵 X [N_tokens, d_model]
↓ 按 Tile 划分（每个 Tile 包含 T 个 token）

Tile 1:  [GEMM_1] ──→ 结果立即发出（RDMA）
               ↓ 重叠
Tile 2:        [GEMM_2] ──→ 结果发出
                     ↓ 重叠
Tile 3:              [GEMM_3] ──→ 结果发出
                           ...
                                  ↓ 全部通信完成

时间线对比：
传统：  [GEMM all] → [All-to-All]
Comet： [GEMM_1][GEMM_2][GEMM_3]...
              ↕ overlap ↕ overlap ↕
              [send_1]  [send_2]  [send_3]...

实际效果：All-to-All 时间被 GEMM 完全覆盖！
```

### 2.2 Warp Specialization（Warp 专用化）

```python
# Comet 的 Warp 分工机制

# GPU Warp 分组：
#   Warp Group 1 (计算 Warp)：负责 GEMM
#   Warp Group 2 (通信 Warp)：负责 RDMA 发送

# CUDA Kernel 伪代码
__global__ void comet_expert_kernel(
    float* input, float* weight, float* output,
    CommunicationBuffer* comm_buf
) {
    // 计算 Warp：执行 GEMM Tile
    if (is_compute_warp(warp_id)) {
        for (int tile = compute_warp_tile_start; 
                  tile < total_tiles; 
                  tile += compute_warp_stride) {
            
            // 计算一个 Tile
            wmma_mma_sync(output_tile, input_tile, weight_tile);
            
            // 写入共享内存，通知通信 Warp
            __syncwarp();
            comm_buf->mark_ready(tile);
        }
    }
    
    // 通信 Warp：监听并发送已完成的 Tile
    if (is_comm_warp(warp_id)) {
        while (!all_tiles_sent()) {
            // 检查是否有新完成的 Tile
            int ready_tile = comm_buf->get_ready_tile();
            if (ready_tile >= 0) {
                // 发起 RDMA（单向，不阻塞）
                rdma_put(
                    dest_gpu[tile_routing[ready_tile]],
                    output_buffer + ready_tile * tile_size,
                    tile_size
                );
            }
        }
    }
}
```

### 2.3 反向传播的 Overlap

```
Comet 不仅优化前向，也优化反向传播：

反向传播的梯度计算 + All-Reduce：

传统：
  [Backward GEMM (dX, dW)] → [All-Reduce(dW)]
  
Comet：
  Tile 1: [dX_1, dW_1 计算] → [dW_1 立即 All-Reduce 发出]
                ↕ overlap ↕
  Tile 2:       [dX_2, dW_2 计算] → [dW_2 发出]
  ...

梯度的 All-Reduce 完全被 Backward GEMM 覆盖！
```

### 2.4 内存管理优化

```
Tile 计算完成后内存的生命周期：

传统：
  计算所有 output → 全部 All-to-All → 释放 output buffer

Comet：
  计算 Tile_1 → 发送 Tile_1（RDMA） → 立即复用 Tile_1 的 buffer
  计算 Tile_2 → 发送 Tile_2         → 复用 Tile_2 的 buffer
  ...
  
内存峰值 = 1个 Tile 的大小（vs 传统的全量 buffer）
→ 内存节省与 MemFine 的思路类似，但实现在更低层（CUDA Kernel 级）
```

---

## 3. 与其他 Overlap 方案的对比

### 3.1 Overlap 粒度对比

| 方案 | Overlap 粒度 | 通信-计算重叠率 | 实现层次 |
|------|------------|--------------|---------|
| **Comet（本文）** | **GEMM Tile（KB）** | **85~95%** | CUDA Kernel |
| **FlowMoE** | Tensor Chunk（MB） | 60~75% | 调度框架 |
| **DeepEP** | 通信内核优化 | 50~70% | NCCL 替代 |
| **MoE Parallel Folding** | Layer 级 Fold | 40~60% | 框架层 |
| **传统 NCCL** | 整个 Tensor | 0~20% | 通信库 |

### 3.2 与 FlowMoE 的深度对比

```
FlowMoE 优化栈：
  Tensor Chunk 粒度
  → Python/C++ 层调度
  → 还是会有等待（chunk 内部串行）

Comet 优化栈：
  GEMM Tile 粒度（最细）
  → CUDA Kernel 内部调度（硬件级）
  → 几乎零等待（Tile 计算完成即发送）

核心差异：
  FlowMoE 是"调度优化"（软件层面）
  Comet 是"Kernel 融合"（硬件层面）
  
两者互补：
  Comet 解决单层内的计算-通信重叠
  FlowMoE 解决跨层的任务调度
```

---

## 4. 性能实验结果

### 4.1 核心指标

| 配置 | Comet | FlowMoE | 传统方案 |
|------|-------|---------|---------|
| **MoE 层吞吐量** | **2.3x** | 1.6x | 1.0x |
| **通信-计算重叠率** | **90%+** | 68% | 15% |
| **端到端训练加速** | **1.8x** | 1.4x | 1.0x |
| **内存节省** | **35%** | 20% | 0% |

**测试环境**：8×H100，Mixtral 8x22B，2048 tokens

### 4.2 消融实验

```
各组件的贡献：

Warp Specialization：          +40%（最大贡献）
Tile 级流水线：                 +25%
反向传播 Overlap：              +15%
内存 Buffer 复用：              +8%
RDMA 单向传输：                 +7%
总计：                          约 +95% = 1.95x
（含调度开销后实际约 1.8x）
```

---

## 5. 工程实现难度与建议

### 5.1 实现挑战

```
Comet 的实现挑战（⭐⭐⭐⭐⭐ 很高）：

1. Warp Specialization 的同步机制
   - 计算 Warp 和通信 Warp 之间的协调
   - 共享内存的竞争条件
   
2. RDMA 的准确时机控制
   - Tile 计算完成时的内存可见性（需要 memory fence）
   - 多目标 GPU 的并发 RDMA

3. 不同 GPU 架构的适配
   - H100 的 NVLink 第 4 代 vs A100 第 3 代，行为不同
   - 需要针对每种架构优化 Tile 大小

4. 与 PyTorch Autograd 的集成
   - 反向传播的 Overlap 需要定制梯度函数
```

### 5.2 集成建议

```python
# 使用 Comet 的最简方式（假设官方提供 API）

from comet_moe import CometMoELayer

# 替换标准 MoE 层
model.moe_layers = nn.ModuleList([
    CometMoELayer(
        num_experts=128,
        hidden_size=4096,
        tile_size=64,           # GEMM Tile 大小
        num_compute_warps=4,    # 计算 Warp 数
        num_comm_warps=2,       # 通信 Warp 数
        rdma_enabled=True,      # 启用单向 RDMA
    )
    for _ in range(num_moe_layers)
])

# 训练方式不变
optimizer.zero_grad()
loss = model(input)
loss.backward()
optimizer.step()
```

---

## 6. 与其他论文的关系

```
Comet 在 MoE 优化栈中的位置：

┌────────────────────────────────────────────────────────┐
│ FlowMoE：跨层任务调度（调度层）                          │
├────────────────────────────────────────────────────────┤
│ Comet：层内计算-通信 Overlap（内核层）← 这里             │
├────────────────────────────────────────────────────────┤
│ DeepEP：通信内核加速（通信底层）                          │
└────────────────────────────────────────────────────────┘

三者可以叠加：
  DeepEP 优化单次 RDMA 效率
  Comet 让 RDMA 与 GEMM 完全重叠
  FlowMoE 在更高层优化跨层调度
```

---

## 延伸阅读

- 🔧 **Flux（Comet 代码）** - 本文开源实现 → https://github.com/bytedance/flux
- 📄 **FlowMoE** - 粗粒度 Overlap → https://arxiv.org/abs/2510.00207
- 📄 **DeepEP** - 通信内核优化 → https://github.com/deepseek-ai/DeepEP
- 📄 **Pipe-Dream** - 流水线并行的早期工作 → https://arxiv.org/abs/1806.03377
- 🔧 **NCCL** - 通信库基础 → https://github.com/NVIDIA/nccl

---

*笔记整理于 2026-03-07，基于 MLSys'25 论文信息*
