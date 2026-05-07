# Tile-Level Comm+Compute Overlap 可行性分析

## 1. 背景与目标

在 MoE (Mixture of Experts) 推理/训练中，All-to-All 通信和专家计算是两个主要开销。本文档分析将 compute 分成 N 个 tile 与 comm 进行 overlap 的可行性。

### 1.1 核心思想

```
原方案:    T_original = A (comm) + B (compute)
Overlap:   T_overlap  = max(A, N × C) + sync_overhead

其中:
  A = All-to-All 通信时间
  B = 完整 GEMM 计算时间 (baseline)
  C = 单个 tile 的计算时间
  N = tile 数量
```

### 1.2 收益条件 (用 CK 做 baseline)

```
Overlap 有收益条件:
  max(A, N×C) < T_ck + A

其中:
  T_ck = CK GEMM 时间 (0.35ms)
  A    = 通信时间
  N×C  = 我们的 tile GEMM 总时间
```

**临界点分析** (CK=0.48ms):
```
当 N×C > A 时 (compute 主导):
  N×C < T_ck + A
  => A > N×C - T_ck

例如 N=1: A > 0.917 - 0.48 = 0.44ms 才有收益
     N=4: A > 1.04 - 0.48 = 0.56ms 才有收益
```

---

## 2. DeepSeek-V3 配置分析

### 2.1 模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 7168 | 隐藏层维度 |
| moe_intermediate_size | 2048 | 专家中间层维度 |
| n_routed_experts | 256 | 路由专家数量 |
| num_experts_per_tok | 8 | 每 token 激活专家数 |
| GateUP | K=7168, N=4096 | FC1: 2×intermediate |
| Down | K=2048, N=7168 | FC2 |

### 2.2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| mbs (micro_batch_size) | 2 | 微批次大小 |
| gbs (global_batch_size) | 4096 | 全局批次大小 |
| seq_len | 4096 | 序列长度 |
| EP (expert_parallel) | 8 | 专家并行度 |
| DP | gbs / mbs = 2048 | 数据并行度 |

### 2.3 计算量分析

```python
# 每个 micro batch 的 token 数
tokens_per_mbs = mbs × seq_len = 2 × 4096 = 8192 tokens

# 每个 EP rank 的本地专家数
local_experts = 256 / 8 = 32 experts

# 每个 token 激活 8 个专家，假设均匀分布到 8 个 EP rank
# 平均每个 EP rank 处理: 8192 × (8/8) = 8192 tokens (for activated experts)
# 但这些 token 分布到 32 个本地专家，每个专家平均处理:
tokens_per_expert = 8192 / 32 = 256 tokens/expert
```

---

## 3. 通信量分析

### 3.1 All-to-All Dispatch

```python
# 每个 token 需要发送 hidden_state 到其激活的专家所在 rank
# 数据量 = tokens × activated_experts × hidden_size × dtype_bytes
dispatch_data = 8192 × 8 × 7168 × 2  # bf16
             = 939,524,096 bytes ≈ 896 MB (total across all ranks)

# 每个 rank 发送/接收量 (假设均匀分布)
dispatch_per_rank = 896 / 8 ≈ 112 MB
```

### 3.2 All-to-All Combine

```python
# 专家输出需要聚合回原 rank
combine_data = 8192 × 8 × 7168 × 2 = 896 MB (total)
combine_per_rank ≈ 112 MB
```

### 3.3 通信时间 (MI355X 8-GPU 单节点实测)

**测试配置**: 8× MI355X, RCCL All-to-All

| N_tiles | M/tile | Data/rank (MB) | 时间 (ms) | 带宽 (GB/s) |
|---------|--------|----------------|-----------|-------------|
| 1 | 8192 | 117.4 | **0.353** | 333 |
| 2 | 4096 | 58.7 | **0.193** | 305 |
| 4 | 2048 | 29.4 | **0.104** | 281 |
| 8 | 1024 | 14.7 | **0.064** | 230 |
| 16 | 512 | 7.3 | **0.052** | 142 |

**关键发现**:
- 单节点 All-to-All 非常快 (< 0.4ms)
- 小数据量带宽效率下降 (333 → 142 GB/s)
- **NVLink 场景: comm 不是瓶颈！**

**跨节点估算** (IB HDR):
- 单节点: ~0.35ms
- 跨节点 (2 nodes): ~0.8-1.0ms
- 跨节点 (8 nodes): ~1.5-2.0ms

---

## 4. 计算时间分析 (MI355X 实测)

### 4.1 Baseline 对比

配置: M=8192, K=7168, N=4096 (FC1 GateUP)

| 实现 | 时间 | 吞吐 | 效率 | 说明 |
|------|------|------|------|------|
| **CK (primus_turbo)** | **0.48ms** | **~1000 TFLOPS** | **77%** | 目标基准 |
| 我们的手写 GEMM | 0.917ms | 524 TFLOPS | 40% | Tile 实现 |

**关键发现**: CK 性能是我们的 **1.9x**

### 4.2 Tile 切分开销 (实测)

将 M=8192 切分成 N 个 tile:

| N_tiles | M/tile | 每tile时间(C) | 总时间(N×C) | Overhead | 效率 |
|---------|--------|---------------|-------------|----------|------|
| 1       | 8192   | 0.917ms       | 0.917ms     | 1.00x    | 40.3% |
| 2       | 4096   | 0.477ms       | 0.955ms     | 1.04x    | 38.8% |
| 4       | 2048   | 0.259ms       | 1.04ms      | 1.13x    | 35.7% |
| 8       | 1024   | 0.185ms       | 1.48ms      | 1.61x    | 25.0% |
| 16      | 512    | 0.185ms       | 2.95ms      | 3.22x    | 12.5% |
| 32      | 256    | 0.185ms       | 5.91ms      | 6.45x    | 6.2%  |

**关键发现**:
1. M/tile ≤ 1024 时，kernel launch overhead 主导 (~0.185ms/tile)
2. **N=4 是最佳平衡点**: 13% overhead，保持合理效率

---

## 5. Overlap 收益分析

### 5.1 理论模型

```
T_overlap = max(A, N×C) + sync_overhead

假设:
- sync_overhead ≈ 0.05ms (per tile synchronization)
- 实际 overlap 效率 η ≈ 80-90%
```

### 5.2 Overlap 收益矩阵 (vs CK + Comm, CK=0.48ms)

**计算方法**:
```
CK+Comm = 0.48ms + A
N=1 (串行) = A + 0.917ms (无 overlap，不推荐)
N≥2 (overlap) = max(A, N×C)
Speedup = (CK+Comm) / T - 1
```

| Comm(A) | CK+Comm | N=1 (串行) | N=2 (overlap) | N=4 (overlap) | N=8 (overlap) |
|---------|---------|------------|---------------|---------------|---------------|
| 0.3ms   | 0.78ms  | -36% ❌    | -18% ❌       | -25% ❌       | -47% ❌       |
| 0.5ms   | 0.98ms  | -31% ❌    | +3% ⚠️       | -6% ❌        | -34% ❌       |
| 0.8ms   | 1.28ms  | -26% ❌    | +35% ✅       | +23% ✅       | -14% ❌       |
| 1.0ms   | 1.48ms  | -23% ❌    | +48% ✅       | +42% ✅       | 0% ⚠️         |
| 1.5ms   | 1.98ms  | -18% ❌    | +52% ✅       | +48% ✅       | +25% ✅       |
| 2.0ms   | 2.48ms  | -15% ❌    | +56% ✅       | +52% ✅       | +40% ✅       |

**关键发现**: 
- **N=1 串行永远不如 CK**（因为我们的 GEMM 比 CK 慢）
- **N=2 在 Comm ≥ 0.5ms 时开始有收益**
- **N=4 在 Comm ≥ 0.8ms 时有收益**
- N=8 tile 开销太大，仅 Comm > 1.5ms 时可用

### 5.3 NVLink 场景 (A=0.353ms 实测) - CK=0.48ms

**注意**: N=1 表示不分 tile，是串行执行 (Comm + Compute)，不是 overlap！

```
Baseline (CK) = Comm + CK = 0.353 + 0.48 = 0.833ms
N=1 (串行):    = Comm + Our = 0.353 + 0.917 = 1.27ms (无 overlap)
N≥2 (overlap): = max(Comm, N×C)
```

| N_tiles | 计算方式 | T_total | vs CK+Comm | 结论 |
|---------|----------|---------|------------|------|
| 1 (串行) | 0.353+0.917 | 1.27ms | **-34%** ❌ | 比 CK 慢很多 |
| 2 | max(0.353, 0.95) | 0.95ms | **-13%** ❌ | CK 更快 |
| 4 | max(0.353, 1.04) | 1.04ms | **-20%** ❌ | CK 更快 |
| 8 | max(0.353, 1.48) | 1.48ms | **-44%** ❌ | CK 更快 |

**结论**: 单节点 NVLink，**任何配置都比 CK 慢，不要用 overlap**

### 5.4 IB HDR 场景 (A=1.0ms 跨节点) - CK=0.48ms

```
Baseline (CK) = Comm + CK = 1.0 + 0.48 = 1.48ms
N=1 (串行):    = Comm + Our = 1.0 + 0.917 = 1.917ms (无 overlap)
N≥2 (overlap): = max(Comm, N×C)
```

| N_tiles | 计算方式 | T_total | vs CK+Comm | 结论 |
|---------|----------|---------|------------|------|
| 1 (串行) | 1.0+0.917 | 1.92ms | **-23%** ❌ | 比 CK 慢 |
| 2 | max(1.0, 0.95) | 1.00ms | **+48%** ✅ | Overlap 有效 |
| 4 | max(1.0, 1.04) | 1.04ms | **+42%** ✅ | Overlap 有效 |
| 8 | max(1.0, 1.48) | 1.48ms | **0%** ⚠️  | 刚好持平 |

**结论**: 跨节点 IB，**N=2-4 有 40-50% 收益**（必须分 tile 才有 overlap）

---

## 6. 推荐配置 (修正后)

### 6.1 结论: 网络速度决定策略

| 网络环境 | Comm | 推荐方案 | 理由 |
|----------|------|----------|------|
| NVLink 单节点 | 0.35ms | **直接用 CK** | Overlap 反而慢 |
| IB 200G (0.5-0.8ms) | 0.5-0.8ms | **Overlap N=2** ✅ | 收益 3-35% |
| IB 100G (0.8-1.5ms) | 1.0ms | **Overlap N=2-4** ✅ | 收益 42-52% |
| Ethernet (> 1.5ms) | 2.0ms | **Overlap N=2-4** ✅ | 收益 52-56% |

**注意**: N=1 (不分 tile) 永远不如 CK，必须 N≥2 才能 overlap！

### 6.2 GEMM 性能目标 (单节点 NVLink)

**已知条件**:
- CK: 0.48ms @ 1000T
- Comm (N=1): 0.353ms
- CK + Comm = 0.833ms

**性能目标表**:

| 目标 | 所需时间 | 所需 TFLOPS | vs 当前 524T |
|------|----------|-------------|--------------|
| **Break-even** | < 0.833ms | **578T** | +10% |
| **10% gain** | < 0.757ms | **635T** | +21% |
| **20% gain** | < 0.694ms | **693T** | +32% |
| **Ideal (完美 overlap)** | < 0.353ms | **1363T** | +160% |

**当前状态 (524T)**:
```
Tile time = 0.918ms > CK+Comm = 0.833ms
Overlap = 0.918ms
Speedup = 0.833 / 0.918 - 1 = -9%  ← 需要再提升 10% 性能才能持平
```

### 6.3 适用场景实现

```
适用: 跨节点 (Comm ≥ 0.5ms)
推荐: N = 2 tiles (最佳平衡)

示例 (IB 100G, Comm=1.0ms, N=2):
  Tile 0: [=COMM_0=][=COMP_0 (0.48ms)=]
  Tile 1:          [=COMM_1=][=COMP_1 (0.48ms)=]
  
  实际: T = max(Comm, 2×C) = max(1.0, 0.95) = 1.0ms
  对比: CK + Comm = 0.48 + 1.0 = 1.48ms
  收益: 1.48 / 1.0 - 1 = 48%

错误示例 (N=1, 串行):
  [===COMM (1.0ms)===][===COMP (0.92ms)===]
  
  实际: T = 1.0 + 0.92 = 1.92ms  ← 比 CK+Comm 还慢！
```

---

## 7. 验证实验设计

### 7.1 实验 1: Tile 切分开销测量

```bash
# 运行 tile granularity benchmark
./bench_tile_granularity

# 测量不同 N 下的:
# - 单 tile 计算时间 C
# - 总计算时间 N×C
# - Overhead factor
```

### 7.2 实验 2: 模拟 Overlap

```python
# 模拟不同 comm 时间下的 overlap 收益
for A in [0.3, 0.5, 1.0, 1.5, 2.0]:  # comm time (ms)
    for N in [1, 2, 4, 8, 16]:
        C = measure_tile_time(M=8192//N)
        T_original = A + B
        T_overlap = max(A, N * C)
        speedup = T_original / T_overlap
        print(f"A={A}, N={N}, speedup={speedup:.2f}x")
```

### 7.3 实验 3: 端到端验证

1. 实现 N=4 tile 的 persistent kernel
2. 使用 RCCL/NCCL 模拟 All-to-All
3. 测量实际 overlap 效率

---

## 8. 结论 (基于完整测试 - 2026-04-09 更新)

### 8.1 实测数据汇总

```
计算性能:
  CK Baseline:     0.48ms @ ~1000 TFLOPS (目标基准)
  Our Tile GEMM:   0.917ms @ 524 TFLOPS  (CK 的 52%)

通信性能 (8-GPU 单节点 All-to-All):
  Full (N=1):   0.353ms @ 333 GB/s
  N=2:          0.193ms @ 305 GB/s  
  N=4:          0.104ms @ 281 GB/s
  N=8:          0.064ms @ 230 GB/s

关键比值:
  CK time / Comm(N=1) = 0.48 / 0.35 = 1.4x  ← 计算略慢于通信
  Our GEMM / CK = 0.917 / 0.48 = 1.9x  ← 我们慢 1.9 倍
```

### 8.2 可行性判断 (CK=0.48ms, 实测 All-to-All)

**注意**: N=1 是串行 (Comm+Compute)，N≥2 才是 overlap (max(Comm, N×C))

**单节点 (8-GPU NVLink)** - 实测数据:

| N | Comm | T_total | CK+Comm | Speedup | 可行? |
|---|------|---------|---------|---------|-------|
| 1 (串行) | 0.353ms | **1.27ms** | 0.833ms | **-34%** | ❌ |
| 2 | 0.193ms | 0.954ms | 0.673ms | **-29%** | ❌ |
| 4 | 0.104ms | 1.036ms | 0.584ms | **-44%** | ❌ |
| 8 | 0.064ms | 1.480ms | 0.544ms | **-63%** | ❌ |

**跨节点 (IB HDR 估算)**:

| 网络 | Comm | N=2 Overlap | CK+Comm | Speedup | 可行? |
|------|------|-------------|---------|---------|-------|
| 2 nodes | 0.8ms | 0.95ms | 1.28ms | **+35%** | ✅ |
| 4 nodes | 1.2ms | 1.20ms | 1.68ms | **+40%** | ✅ |
| 8 nodes | 1.8ms | 1.80ms | 2.28ms | **+27%** | ✅ |

### 8.3 关键结论

```
❌ 用手写 GEMM (524T) 做 tile overlap 方案大多数场景不可行！

原因:
1. CK 太快: 0.350ms vs 我们的 0.917ms (2.6x 差距)
2. Tile comm 也有开销 (N=4 时 1.22x)
3. 只有慢速网络 + 小 N (1-2) 才有收益

可行条件:
- 网络 ≥ IB HDR 200G (comm ≥ 0.8ms)
- N = 1-2 (避免 tile 开销累积)
- 预期收益: 15-25%
```

### 8.4 替代方案

**方案 A: 使用 CK kernel + Stream Overlap (推荐)**
```python
# 简单且高效
stream_comm = hip.Stream()
stream_comp = hip.Stream()

for tile in tiles:
    all_to_all_async(tile.input, stream_comm)
    stream_comp.wait(stream_comm)
    ck_grouped_gemm(tile.input, tile.output, stream_comp)
```
- 优点: 使用 CK 的 1375T 性能
- 优点: 无需实现复杂的 persistent kernel
- 缺点: Stream 同步开销 (~0.01ms/tile)

**方案 B: 优化手写 GEMM 到 CK 水平**
```
目标: 0.35ms (需要 2.6x 性能提升)
需要:
- 向量化 store (C-shuffle epilogue)
- 更大 tile (256×256×64)
- 更精细 pipeline (software prefetch)
- 寄存器优化 (reduce spilling)
```

**方案 C: 仅慢速网络使用当前方案**
```
适用: IB HDR 100G/200G (comm > 0.8ms)
配置: N = 1-2
收益: 15-25%
```

### 8.5 建议

| 优先级 | 方案 | 适用场景 | 工作量 |
|--------|------|----------|--------|
| 1 | CK + Stream overlap | 通用 | 低 (1-2天) |
| 2 | 优化手写 GEMM | 需要 kernel 融合 | 高 (1-2周) |
| 3 | 当前方案 | 慢速网络 | 低 (已实现) |

### 8.6 CK Tile Overlap 可行性 (实测更新)

**CK 在不同 tile 大小下的性能**:

| N | M/tile | CK/tile | N×CK | 效率 |
|---|--------|---------|------|------|
| 1 | 8192 | 0.352ms | 0.352ms | 100% (1369T) |
| 2 | 4096 | 0.183ms | 0.367ms | **96%** (1313T) |
| 4 | 2048 | 0.135ms | 0.540ms | 65% (890T) |
| 8 | 1024 | 0.086ms | 0.689ms | 51% (698T) |

**单节点 Overlap 分析** (Baseline = CK + Comm = 0.705ms):

| N | N×CK | Comm | Overlap | vs Baseline | 结论 |
|---|------|------|---------|-------------|------|
| 2 | 0.367ms | 0.193ms | 0.367ms | **+92%** ✅ | **强烈推荐** |
| 4 | 0.540ms | 0.104ms | 0.540ms | **+30%** ✅ | 有收益 |
| 8 | 0.689ms | 0.064ms | 0.689ms | +2% ⚠️ | 勉强持平 |

**理论收益**:
```
用 CK 做 tile overlap，N=2 理论有 92% 收益
```

**实测结果 (Stream Overlap)**:
```
❌ NCCL All-to-All 不支持真正的跨 stream async!

测试结果:
- Sequential: 0.82ms (正常)
- Parallel Streams: 11ms !!! (NCCL 有同步开销)
- D2D Memcpy + Compute: 0.49ms ✓ (有 overlap)

原因:
- NCCL 内部有同步机制，不能和 compute 真正 overlap
- Stream overlap 方案在单节点 NCCL 场景不可行
```

**NCCL/RCCL Stream Overlap 深入分析**:

1. **测试环境**:
   - 8× MI355X, RCCL 2.27.7
   - GPU_MAX_HW_QUEUES=8
   - TORCH_NCCL_USE_COMM_NONBLOCKING=1

2. **测试结果**:
   | 模式 | 时间 | 说明 |
   |------|------|------|
   | Compute only | 0.44ms | - |
   | Comm only | 0.39ms | - |
   | Sequential sum | 0.83ms | - |
   | **Ideal overlap** | 0.44ms | max(compute, comm) |
   | One-stream | 3.1ms | 比 sequential 慢 4x！ |
   | Two-stream + sync | 0.91ms | +10% 开销 |
   | Independent streams | 0.90ms | +8% 开销 |
   | **Best achieved** | 0.85ms | -3% vs sequential |

3. **根本原因 (来自 NCCL GitHub Issues)**:
   - NCCL 操作是阻塞 kernel，等待其他 GPU 启动 [#146]
   - 两个 NCCL 操作在不同 stream 可能死锁 [#217]
   - One-stream 模式在某些情况反而最快 [#205]
   - CUDA 不保证多 stream 并发执行 [#217]
   - NCCL kernel 和 compute 竞争 SM 资源

4. **环境变量测试**:
   - GPU_MAX_HW_QUEUES: 无明显效果
   - TORCH_NCCL_BLOCKING_WAIT=0: 略有改善
   - TORCH_NCCL_USE_COMM_NONBLOCKING=1: 略有改善

**结论**:
```
❌ Stream-level overlap 在当前 RCCL 实现下不可行
   - Best: 0.85ms, Sequential: 0.83ms (无收益)
   - NCCL 内部有同步机制
   - SM 资源竞争导致性能下降
```

**可行方案**:
```
1. Persistent Kernel (推荐):
   - 在 kernel 内部划分 CU: 部分做 compute，部分做 comm
   - 绕过 NCCL，使用 GPU Direct RDMA / HIP IPC
   - DeepEP 论文方案

2. 跨节点场景:
   - Comm 延迟更高 (1-2ms)
   - 即使串行执行，tile 分割也能减少延迟
   - 但需要更多内存和复杂调度

3. DeepEP 方案:
   - 使用专用 GPU 做通信
   - 计算 GPU 和通信 GPU 分离
```

---

## 附录 A: 计算公式

```python
# GEMM FLOPS
flops = 2 * M * K * N

# 时间估算
time_ms = flops / (tflops * 1e12) * 1e3

# Overlap 收益
speedup = (A + B) / max(A, N * C)

# 有效带宽
bw_gbps = data_bytes / (time_ms * 1e-3) / 1e9
```

## 附录 B: 测试命令

```bash
# 编译 tile granularity benchmark
cd /home/xiaompen/AIInfra-Book/3rd/MMOE
hipcc -std=c++17 -O3 --offload-arch=gfx950 \
    benchmarks/bench_tile_granularity.hip -o benchmarks/bench_tile

# 运行测试
./benchmarks/bench_tile

# 编译 overlap 验证
hipcc -std=c++17 -O3 --offload-arch=gfx950 \
    benchmarks/bench_tile_overlap.hip -o benchmarks/bench_overlap
```
