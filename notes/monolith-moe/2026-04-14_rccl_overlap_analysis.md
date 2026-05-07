# RCCL 无法 Comm+Compute Overlap 的根本原因分析

## 1. 问题回顾

根据昨天的测试结果，使用 multi-stream 方式尝试 overlap RCCL 通信和 GEMM 计算时，观察到以下现象：

| 模式 | 时间 | 说明 |
|------|------|------|
| Compute only | 0.44ms | - |
| Comm only | 0.39ms | - |
| Sequential | 0.83ms | 符合预期 |
| **Ideal overlap** | 0.44ms | max(compute, comm) |
| One-stream | 3.1ms | 比 sequential 慢 4x！|
| Two-stream + sync | 0.91ms | +10% 开销 |
| Independent streams | 0.90ms | +8% 开销 |
| **Best achieved** | 0.85ms | -3% vs sequential |

**核心问题：** 即使使用独立 stream，也无法实现真正的 overlap，最好情况只比串行快 3%。

---

## 2. NCCL/RCCL 无法并行的根本原因

### 2.1 设计层面的限制

NCCL/RCCL 的设计目标是 **集合通信** (Collective Communication)，而非与计算 overlap。其内部有多个机制导致无法并行：

#### 2.1.1 阻塞式 Kernel 设计

NCCL kernel 本质是 **同步阻塞** 的：

```
NCCL Kernel 执行流程:
1. Launch ncclSend/Recv kernel
2. Kernel 内部: 等待对端 GPU 也启动相应 kernel
3. 数据传输
4. Kernel 完成

问题: 步骤 2 是 blocking wait，占用 GPU 资源
```

来源：[NCCL Issue #146](https://github.com/NVIDIA/nccl/issues/146)
> "NCCL operations are blocking kernels that wait for other GPUs to launch."

#### 2.1.2 隐式同步机制

NCCL 在以下情况会触发隐式同步：
- `ncclCommInitRank()`: 全局 barrier
- `ncclGroupEnd()`: 等待所有操作完成
- 内部 ring/tree 算法需要多轮同步

```
问题示例:
Stream A: [NCCL AllGather] ←等待 Stream B 上的 NCCL 操作
Stream B: [GEMM] [NCCL Send] ←被 Stream A 阻塞

结果: 即使在不同 stream，NCCL 操作也会相互等待
```

#### 2.1.3 CUDA Driver 序列化

来源：[NCCL Issue #217](https://github.com/NVIDIA/nccl/issues/217)
> "CUDA does not guarantee multi-stream concurrent execution."

CUDA/HIP driver 在某些情况下会序列化 kernel：
- 同一设备的 kernel launch 需要获取 mutex
- cudaMalloc/hipMalloc 会触发同步
- 部分驱动版本对多 stream 支持不完善

#### 2.1.4 资源竞争

来源：[NCCL Issue #1433](https://github.com/NVIDIA/nccl/issues/1433)
> "You may not compete for SMs but still compete for memory bandwidth."

即使 NCCL kernel 只用 1 个 SM (`<<<1,1,1>>>`):
- **HBM 带宽竞争**: NCCL 传输消耗大量内存带宽
- **L2 Cache 污染**: NCCL 数据冲刷 compute kernel 的 cache
- **NVLink 竞争**: 节点内通信占用 xGMI/NVLink 带宽

### 2.2 实验验证

NVIDIA 工程师在 [Issue #1931](https://github.com/NVIDIA/nccl/issues/1931) 中确认：

```
测试: 4× A100-SXM4-80GB, NCCL 2.28

现象:
- 即使 compute 和 comm 完全独立 (无数据依赖)
- 即使使用不同 CUDA stream
- NCCL AllGather 仍然等待 GEMM 完成后才执行

结论: "NCCL always waits for cublas kernel to finish"
```

这个问题自 2019 年就存在 ([Issue #205](https://github.com/NVIDIA/nccl/issues/205))，是 **设计层面** 的限制，不是 bug。

---

## 3. 为什么 Multi-Stream 方案无法工作

### 3.1 Stream 只是逻辑概念

```
期望:
Stream 0: [===== GEMM =====]
Stream 1: [===== NCCL =====]
Timeline: |<--- overlap --->|

实际:
Stream 0: [===== GEMM =====]
Stream 1:                    [===== NCCL =====]  ← 被序列化
Timeline: |<--- sequential --->|
```

Stream 只提供 **逻辑顺序** 保证，不保证 **物理并行**。GPU scheduler 可能因为以下原因序列化执行：
- 资源不足 (SM、memory)
- Driver 锁
- 硬件队列限制

### 3.2 GPU Hardware Queue 限制

AMD MI300X 默认只有 4 个 hardware queue：

```bash
# 默认设置
GPU_MAX_HW_QUEUES=4

# 实测效果
GPU_MAX_HW_QUEUES=4:  overlap = 0%
GPU_MAX_HW_QUEUES=8:  overlap ≈ 0% (无明显改善)
GPU_MAX_HW_QUEUES=16: overlap ≈ 0%
```

增加 hardware queue 对 RCCL overlap 无帮助，因为瓶颈不在这里。

### 3.3 NCCL Group Semantics

```python
# 常见误解: 认为这样可以 overlap
with torch.cuda.stream(comm_stream):
    dist.all_gather(...)

with torch.cuda.stream(comp_stream):
    gemm(...)

# 实际: ncclGroupEnd() 会等待所有 rank 完成
# 导致隐式同步
```

---

## 4. NCCL 2.28 的解决方案 (NVIDIA 新进展)

NVIDIA 在 2025年11月发布的 **NCCL 2.28** 引入了三个关键特性来解决这个问题：

### 4.1 Device API (GPU-Initiated Networking)

**核心思想**: 在 GPU kernel 内部直接发起网络操作，避免 host-side 同步。

```cpp
// 传统方式 (Host-initiated)
ncclAllGather(send, recv, count, ...);  // Host 调用，需要同步

// NCCL 2.28 Device API (Kernel-initiated)
__global__ void fused_kernel() {
    // 计算
    compute_tile();
    
    // 在 kernel 内直接发起通信
    nccl_device_put(data, dest_pe);  // 无需返回 host
    
    // 继续计算
    compute_next_tile();
}
```

**三种模式**:

| 模式 | 机制 | 适用场景 |
|------|------|----------|
| **LSA** (Load/Store Accessible) | CUDA P2P / NVLink | 节点内 |
| **Multimem** | NVLink SHARP multicast | 节点内广播 |
| **GIN** (GPU-Initiated Networking) | GPU 直接控制 NIC | 跨节点 |

### 4.2 Copy Engine (CE) Collectives

**核心思想**: 使用专用 Copy Engine 做通信，完全不占用 SM。

```
传统 NCCL:
SM-based: NCCL kernel 占用 SM → 与 compute 竞争

NCCL 2.28 CE:
CE-based: 通信由 Copy Engine 执行 → SM 100% 用于 compute
```

**优势**:
- Zero-SM operation
- 更大的 NVLink 事务宽度
- 真正的 compute+comm overlap

**限制**:
- 仅支持 AlltoAll, AllGather
- 需要 NVLink (节点内)
- 目前仅 NVIDIA GPU 支持

### 4.3 对 AMD/RCCL 的启示

RCCL 目前 **没有** Device API 和 CE Collectives：
- RCCL 2.27 仍是传统 host-initiated 设计
- AMD 没有等效的 Copy Engine offload
- 需要等待 AMD 跟进实现

---

## 5. ROCShmem 方案分析

### 5.1 ROCShmem 是什么

ROCShmem (现已迁移到 `rocm-systems` repo) 是 AMD 的 GPU-centric networking library：

| 特性 | 说明 |
|------|------|
| 编程模型 | OpenSHMEM PGAS (Partitioned Global Address Space) |
| 通信方式 | One-sided put/get |
| 后端 | IPC, Reverse Offload, **GPU Direct Async (GDA)** |
| NIC 支持 | Mellanox ConnectX-7, Broadcom Thor2, AMD Pensando |

**关键特性**: GDA 后端可以从 **GPU kernel 内部** 直接发起 RDMA 操作。

### 5.2 ROCShmem 的优势

```cpp
// ROCShmem: GPU kernel 内直接通信
__global__ void my_kernel() {
    // 从 kernel 内部直接发起 RDMA put
    shmem_putmem_nbi(dest, src, size, dest_pe);
    
    // 继续计算，不需要返回 host
    compute();
    
    // 同步
    shmem_fence();
}
```

**对比 RCCL**:

| 特性 | RCCL | ROCShmem |
|------|------|----------|
| 发起位置 | Host | GPU kernel 内 |
| 同步模型 | Collective (需要所有 rank) | One-sided (单边) |
| Blocking | 隐式 blocking | 非阻塞 put/get |
| 与 compute overlap | ❌ 困难 | ✅ 天然支持 |

### 5.3 DeepEP 如何使用 ROCShmem

DeepEP (AMD 版本的 DeepSeek EP 实现) 使用 ROCShmem 实现 MoE All-to-All：

```
架构: Two-Tier Communication
├── 节点内 (Intranode)
│   └── NVLink/xGMI + IPC (直接内存访问)
└── 跨节点 (Internode)
    └── ROCShmem + RDMA + GPU Direct Async
```

**关键技术**:

1. **Symmetric Memory**: 所有 GPU 分配相同布局的 buffer
2. **One-sided Put**: GPU 直接写入远程 GPU 内存
3. **Atomic Tail Update**: 使用 `shmem_atomic_add` 通知接收方
4. **Producer-Consumer Pattern**: 无锁环形 buffer

```cpp
// DeepEP 风格的 dispatch
__global__ void dispatch_kernel(...) {
    // 1. 获取远程 buffer slot
    int slot = shmem_atomic_fetch_add(&tail, 1, dest_pe);
    
    // 2. 准备数据
    pack_token_data(local_buf, token_idx);
    
    // 3. 直接 RDMA put 到远程 GPU
    shmem_putmem_nbi_wg(
        remote_buf + slot * stride,
        local_buf,
        size,
        dest_pe
    );
    
    // 4. 更新 tail 通知接收方
    shmem_fence();
    shmem_atomic_add(&tail_signal, 1, dest_pe);
}
```

### 5.4 ROCShmem 能否解决 Overlap 问题？

**理论上可以**，但有以下挑战：

| 挑战 | 说明 |
|------|------|
| **编程复杂度** | 需要重写所有通信逻辑 (不是简单替换 API) |
| **节点内效率** | 单节点 xGMI 已经很快 (0.35ms)，ROCShmem 可能更慢 |
| **内存管理** | 需要 symmetric buffer allocation |
| **同步复杂** | 需要手动管理 fence/barrier |
| **生态系统** | 缺乏与 PyTorch 的集成 |

**适用场景**:

| 场景 | 推荐 |
|------|------|
| 单节点 8-GPU | ❌ 用 RCCL + CK 更简单高效 |
| 跨节点 Comm > 1ms | ✅ ROCShmem + Persistent Kernel |
| 需要 Comm+Compute Fusion | ✅ ROCShmem (DeepEP 方案) |

---

## 6. 可行的解决方案对比

### 6.1 方案一: 等待 RCCL 支持 Device API (被动)

- **可行性**: 取决于 AMD
- **时间**: 未知 (可能 2026+)
- **工作量**: 零
- **推荐度**: ⭐⭐

### 6.2 方案二: ROCShmem + Persistent Kernel (主动)

```
架构:
┌─────────────────────────────────────────────┐
│              Persistent Kernel              │
├─────────────┬──────────────┬────────────────┤
│  CU Group A │  CU Group B  │   CU Group C   │
│   (Compute) │   (Compute)  │    (Comm)      │
│   GEMM Tile │   GEMM Tile  │  ROCShmem Put  │
└─────────────┴──────────────┴────────────────┘
         ↓            ↓              ↑
    [Tile 0 Ready] [Tile 1 Ready]  [Send Tile 0]
```

**工作流程**:
1. 划分 CU: 90% compute, 10% comm
2. Compute CU 生产数据到 shared buffer
3. Comm CU 使用 ROCShmem 发送完成的 tile
4. 无需返回 host，完全 GPU 内部协调

**复杂度**: 高 (需要实现完整调度器)
**收益**: 跨节点 40-50% 加速
**推荐度**: ⭐⭐⭐⭐ (跨节点场景)

### 6.3 方案三: 跳过 Overlap，优化 Tile GEMM (务实)

```
目标: 让 Tile GEMM 性能接近 CK

当前: 524 TFLOPS (CK 的 52%)
目标: 900+ TFLOPS (CK 的 90%)

如果达成:
  单节点: T = 0.53ms < CK+Comm = 0.83ms → 有收益
  跨节点: T = 0.53ms << CK+Comm = 1.48ms → 大收益
```

**优化方向**:
1. 向量化 store (C-shuffle epilogue)
2. 更大 tile (256×256×64)
3. Software prefetch
4. 减少 register spilling

**复杂度**: 中
**收益**: 通用
**推荐度**: ⭐⭐⭐⭐⭐

### 6.4 方案四: MSCCL++ (微软替代方案)

MSCCL++ 是微软开发的高性能通信库，声称比 NCCL 快 3.8x：

| 特性 | NCCL | MSCCL++ |
|------|------|---------|
| 设计 | 通用集合通信 | 专用优化 |
| Overlap 支持 | 有限 | 更好 |
| 编程模型 | Host-initiated | 可定制 |

**限制**:
- 主要针对 NVIDIA GPU
- AMD 支持不完善
- 文档较少

**推荐度**: ⭐⭐ (观望)

---

## 7. 结论与建议

### 7.1 为什么 RCCL Multi-Stream 不工作

```
根本原因:
1. RCCL/NCCL 设计为 Collective 语义 (需要所有 rank 同步)
2. Host-initiated 模式导致隐式同步
3. 内存带宽和 L2 cache 竞争
4. CUDA/HIP driver 序列化
```

### 7.2 ROCShmem 能否解决

```
✅ 理论可行: ROCShmem 支持 GPU-initiated RDMA
❌ 工程复杂: 需要完全重写通信逻辑
❌ 单节点不划算: xGMI 已经足够快
✅ 跨节点有价值: 当 Comm > 1ms 时收益明显
```

### 7.3 推荐路径

| 优先级 | 方案 | 场景 | 工作量 |
|--------|------|------|--------|
| 1 | 优化 Tile GEMM 到 900T | 通用 | 中 |
| 2 | 单节点直接用 CK (不 overlap) | 8-GPU NVLink | 零 |
| 3 | 跨节点用 ROCShmem + Persistent Kernel | 多节点 | 高 |
| 4 | 等待 RCCL Device API | 未来 | 零 |

### 7.4 如果必须实现 Overlap

```
推荐: ROCShmem + Persistent Kernel (DeepEP 方案)

步骤:
1. 安装 ROCShmem with GDA support
2. 实现 Symmetric Buffer 分配
3. 实现 Persistent Kernel 调度器
4. 划分 CU 给 compute 和 comm
5. 实现 Producer-Consumer 同步

预期收益:
  - 跨节点 40-50% 加速
  - 需要 2-4 周开发时间
```

---

## 附录 A: 相关资源

- [NCCL Issue #1433: Memory Bandwidth Competition](https://github.com/NVIDIA/nccl/issues/1433)
- [NCCL Issue #1931: Multi-Stream Parallelization](https://github.com/nvidia/nccl/issues/1931)
- [NCCL 2.28 Device API](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
- [ROCm/rocSHMEM](https://github.com/ROCm/rocSHMEM) → 现已迁移到 rocm-systems
- [ROCm/DeepEP](https://github.com/ROCm/DeepEP)
- [MSCCL++](https://arxiv.org/html/2504.09014v1)

## 附录 B: ROCShmem 安装

```bash
# 依赖
# - Open MPI
# - UCX  
# - ROCm 6.3.4+

# 克隆
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems/rocSHMEM

# 配置 (启用 GDA for Mellanox NIC)
cmake -B build \
  -DCMAKE_INSTALL_PREFIX=/opt/rocshmem \
  -DENABLE_GDA=ON \
  -DGDA_PROVIDER=MLX5 \
  ..

# 编译
cmake --build build -j
cmake --install build
```

## 附录 C: DeepEP 关键代码路径

```
csrc/kernels/internode.cu:
├── dispatch_kernel()      # Token 分发
│   ├── shmem_putmem_nbi() # 非阻塞 RDMA put
│   ├── shmem_fence()      # 内存屏障
│   └── shmem_atomic_add() # 通知接收方
├── combine_kernel()       # 结果聚合
└── barrier()              # 全局同步
```
