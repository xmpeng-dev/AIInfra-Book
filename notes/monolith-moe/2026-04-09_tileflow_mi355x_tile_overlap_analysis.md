# TileFlow Tile-Level Overlap 技术分析 — MI355X (gfx950)

> 日期: 2026-04-09
> 硬件: 8x AMD Instinct MI355X (gfx950), XGMI 全互联
> 容器: tasimage/primus:pr-563-ainic
> 软件: PyTorch 2.12+rocm7.1, Triton (ByteDance fork), triton-distributed 3.4.0

---

## 1. 背景

TileFlow 移植了 [Comet (MLSys '25)](https://arxiv.org/abs/2502.19811) 的 tile-level compute-communication overlap 算法，从 NVIDIA (NVSHMEM + NVLink) 迁移到 AMD (rocSHMEM + XGMI)。

Comet 的核心思想：在单个 cooperative kernel 内，compute wavefront 完成一个 GEMM tile 后立即通过 SHMEM signal 通知 comm wavefront 发送结果，实现 tile 粒度（~us）的通信隐藏。这比传统的 stream-based overlap（chunk 粒度，~100us）精细一个数量级。

## 2. 代码结构

```
tileflow/triton_kernels/
├── moe_overlap.py          # 统一 API 层（rocSHMEM dispatch + stream fallback）
├── moe_reduce_rs.py        # Comet GEMM+ReduceScatter fused kernel
├── moe_reduce_ar.py        # Comet GEMM+AllReduce fused kernel
├── allgather_group_gemm.py # Comet AG+GEMM fused kernel
├── ipc_primitives.py       # HIP IPC 跨 GPU 通信原语（实验性）
├── rocshmem_device.py      # rocSHMEM 设备端 API 封装
├── rocshmem_utils.py       # rocSHMEM 主机端工具
└── common_ops.py           # grid barrier, 跨 GPU barrier
```

## 3. 三个 overlap 路径的验证状态

### 3.1 Stream-based overlap（torch.distributed）

**原理**: 用两个 CUDA stream，compute stream 执行 GEMM chunks，reduce stream 异步执行 ReduceScatter/AllReduce。

**状态**: ✅ 全部通过

| 测试 | 结果 |
|------|------|
| 单 GPU 基础 (barrier/schedule/engine/MoE) | 14/14 PASS |
| Triton Kernel 单 GPU (moe_utils/grouped GEMM) | 6/6 PASS |
| 多 GPU 分布式 (EP MoE, reduce_ar/rs, overlap) | 10/10 PASS |

**性能数据** (8x MI355X):

| Config | Pattern | baseline | nc=2 | nc=4 | speedup |
|--------|---------|----------|------|------|---------|
| Small (512tok, 8E, 4K×4K) | RS | 0.436 ms | 1.078 ms | 1.945 ms | 0.40x |
| Medium (2048tok, 8E, 4K×14K) | RS | 1.326 ms | 1.459 ms | 1.882 ms | 0.91x |
| DS-V3 (4096tok, 64E, 7K×2K) | AR | 1.017 ms | 1.013 ms | 1.118 ms | 1.00x |
| Llama-70B (4096tok, 8E, 4K×14K) | RS | 2.196 ms | 2.361 ms | 2.663 ms | 0.93x |

**结论**: Stream overlap 在 MI355X 上无法加速。XGMI AllReduce/ReduceScatter 延迟极低（~50-200us），stream event sync overhead（~10-20us）相对于 chunk GEMM 时间不可忽略。

### 3.2 rocSHMEM tile-level overlap（Comet 原始方案）

**原理**: 单个 cooperative kernel 内 compute/comm wavefront 通过 rocSHMEM `signal_wait_until` / `symm_at` 做 tile 级同步。

**验证步骤**:

| 步骤 | 状态 | 说明 |
|------|------|------|
| IPC-only rocSHMEM 编译 (`USE_GDA=OFF, USE_IPC=ON, gfx950`) | ✅ | `/opt/rocshmem_ipc/` |
| pyrocshmem 编译 (链接 IPC rocSHMEM) | ✅ | `.so` 安装成功 |
| device bitcode 编译 (`librocshmem_device.bc`, gfx950) | ✅ | 1.6MB |
| rocSHMEM runtime init (`init_rocshmem_by_torch_process_group`) | ✅ | 8 GPU 成功 |
| `_shmem_available()` = True | ✅ | 上下文创建成功 |
| RS/AR context 创建 (symmetric tensor 分配) | ✅ | 正常 |
| GEMM kernel 编译 + launch | ✅ | 正常 |
| `barrier_all_intra_node_atomic_cas_block` JIT 编译 | ✅ | 修复了 ImportFrom bug |
| **reduce_topk_reduce_scatter kernel launch** | ❌ | `hipErrorNoBinaryForGpu` |
| **reduce_topk_allreduce kernel launch** | ❌ | `hipErrorNoBinaryForGpu` |

**阻塞原因**: Triton 编译 fused reduce kernel 时链接 `librocshmem_device.bc`，生成的 `.hsaco` 虽然标记为 gfx950，但 HIP runtime 加载时报 `hipErrorNoBinaryForGpu`。怀疑 rocSHMEM device bitcode 中的某些指令/ABI 与 gfx950 不兼容。

### 3.3 HIP IPC + Triton system atomics（DeepEP 启发的方案）

**原理**: 绕过 rocSHMEM runtime，直接用 `hipIpcGetMemHandle`/`hipIpcOpenMemHandle` 共享 GPU 内存，用 Triton 的 `tl.atomic_add(scope="sys")` 做跨 GPU 原子操作。

**验证步骤**:

| 步骤 | 状态 | 说明 |
|------|------|------|
| `hipExtMallocWithFlags(hipDeviceMallocUncached)` | ✅ | ctypes 调用成功 |
| `hipIpcGetMemHandle` / `hipIpcOpenMemHandle` | ✅ | 8 GPU handle 交换成功 |
| IPC pointer table 创建 | ✅ | `int64[8]` 指针表可用 |
| Triton `tl.atomic_xchg(ipc_ptr, scope="sys")` 跨 GPU 写 | ✅ | 8 GPU 互相写入+读取 PASS |
| Triton `tl.atomic_add(ipc_ptr, scope="sys")` 跨 GPU 原子加 | ✅ | 单次操作正确 |
| **IPC barrier (多轮 atomic add/sub spin-wait)** | ❌ | **Hang** — kernel 不返回 |

**阻塞原因**: Triton 编译器生成的 system-scope atomic 指令在 IPC 映射内存上做多轮 spin-wait 时 hang。单次原子操作正确，但持续 spin-wait 模式不工作。怀疑 Triton 在 gfx950 上的 system-scope atomic 代码生成不完整（可能缺少正确的 cache flush / memory ordering 指令）。

### 3.4 DeepEP 的做法（对照参考）

DeepEP 在同一台机器上正常工作，其通信 kernel 使用 **HIP C++**（不是 Triton）：

```cpp
// DeepEP barrier — HIP C++ (hipcc 编译)
atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
while (true) {
    auto value = ld_volatile_global(barrier_signal_ptrs[rank] + thread_id);
    if (__all_sync(kFullWarpMask, value <= 0)) break;
}
```

关键区别: `atomicAdd_system` / `ld_volatile_global` 由 `hipcc` 编译为 gfx950 原生指令，保证了正确的 system-scope 语义。Triton JIT 编译器在此方面有缺陷。

## 4. 总结与建议

### 4.1 当前可用方案

| 方案 | 性能 | 复杂度 | 推荐场景 |
|------|------|--------|----------|
| Stream overlap (torch.distributed) | ~0.9x baseline | 低 | 功能验证、小规模 |
| triton_dist run_moe_reduce_rs_overlap | 与 stream 相同 | 中 | 已实现的 chunk 级 overlap |

### 4.2 实现 tile-level overlap 的路径

| 路径 | 可行性 | 工作量 | 说明 |
|------|--------|--------|------|
| 在 MI300X (gfx942) 上验证 rocSHMEM path | 高 | 小 | gfx942 是 rocSHMEM 原生支持的目标 |
| 用 HIP C++ extension 实现通信 kernel | 高 | 中 | 参考 DeepEP，GEMM 保留 Triton |
| 等 Triton gfx950 system-scope atomic 修复 | 中 | 等待 | AMD Triton 团队 / triton-distributed |
| 用 PyTorch custom op 包装 HIP C++ barrier | 高 | 中 | 混合 Triton + HIP C++ |

### 4.3 推荐下一步

**短期 (EP=8 单机)**:
1. 将 reduce-scatter/allreduce 通信 kernel 用 HIP C++ PyTorch extension 实现（参考 DeepEP 的 `barrier_block` + `atomicAdd_system`）
2. GEMM 部分继续用 Triton grouped GEMM kernel（已验证）
3. 通过 `moe_overlap.py` 的 dispatch 层串联两者

**中期**:
4. 在 MI300X 集群上验证完整的 rocSHMEM tile-level 路径
5. 向 AMD Triton 团队反馈 gfx950 system-scope atomic + IPC 的问题

---

## 附录: 单 GPU Kernel 性能基准

| Kernel | Config | Latency | TFLOPS / BW |
|--------|--------|---------|-------------|
| torch.matmul (bf16) | 8K×4K×14K | 0.760 ms | 1266 TFLOPS |
| Triton Grouped GEMM (bf16) | 4096tok, 8E, 4K×14K | 1.549 ms | 621 TFLOPS |
| TopK Reduce | 8192tok, top2, N=16K | 0.155 ms | 5212 GB/s |
| Index Compute | 4096tok, 8E | 0.087 ms | — |

## 附录: 环境信息

```
GPU:    AMD Instinct MI355X (gfx950:sramecc+:xnack-)
ROCm:   7.8.0
Torch:  2.12.0.dev20260408+rocm7.1
Triton: ByteDance fork (cea556df, triton_dist 3.4.0)
MPI:    OpenMPI 5.0.8a1
```
