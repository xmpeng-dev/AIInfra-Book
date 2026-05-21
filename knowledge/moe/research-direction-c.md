# 方向 C 深度分析：FSEP 在 AMD 硬件上的原生实现与超越

> **定位：** 对 README_research.md 方向 C 的完整展开  
> **关联：** [`papers/laer-moe-fsep.md`](../../papers/laer-moe-fsep.md) · rocflow/README.md  
> **硬件目标：** AMD MI300X / MI325X  
> **更新：** 2026-03-09

---

## 目录

1. [背景：LAER-MoE 的 FSEP 在 NVIDIA 上的局限](#1-背景)
2. [AMD MI300X 的硬件特性深度解析](#2-amd-mi300x-硬件特性)
3. [核心创新一：XGMI In-Place Reduction 替代 ReduceScatter](#3-xgmi-in-place-reduction)
4. [核心创新二：HIP Wavefront 专用化 Expert Kernel](#4-hip-wavefront-专用化)
5. [核心创新三：双通道并发调度](#5-双通道并发调度)
6. [核心创新四：Re-layout 的 AMD 原生加速](#6-re-layout-amd-原生加速)
7. [AMD-FSEP 完整计算流程](#7-完整计算流程)
8. [性能模型与理论分析](#8-性能模型)
9. [论文贡献点梳理](#9-论文贡献点)
10. [实验设计方案](#10-实验设计)
11. [实现路线图](#11-实现路线图)
12. [与 LAER-MoE 及相关工作的差异定位](#12-差异定位)

---

## 1. 背景

### 1.1 LAER-MoE FSEP 的核心假设

LAER-MoE（ASPLOS '26）的 FSEP 在设计时基于以下假设：

```
LAER-MoE 的通信假设：

  ReduceScatter：基于 NCCL，走 NVLink（节点内）或 IB（跨节点）
  All-to-All：  基于 NCCL，所有通信走同一套通信库
  Re-layout：   基于 NCCL All-to-All，参数搬迁通过 NCCL 完成

  隐含假设：通信库是单一路径，不感知底层硬件的异构性
  → 节点内和节点间通信被 NCCL 统一处理
  → 即使 NVLink 和 IB 是不同物理介质，软件层看不到区别
```

### 1.2 这个假设在 AMD 上不成立

AMD MI300X 集群的通信架构是双层异构的：

```
AMD 集群的通信层次：

  ┌─── Node（8 × MI300X）──────────────────────────────┐
  │                                                    │
  │  GPU0 ←──── XGMI（Infinity Fabric）─────→ GPU7   │
  │             896 GB/s 全双工                        │
  │             延迟 ~3μs（近内存访问速度）             │
  │             每块 GPU 可直接读写其他 GPU 的 HBM      │
  │             → Peer-to-Peer（P2P）原生支持           │
  └───────────────────────────┬────────────────────────┘
                               │
                      RDMA over RoCE / IB
                      400 Gbps（跨节点）
                      延迟 ~15μs

关键差异：
  XGMI 提供的不仅是高带宽，还有 P2P 内存访问语义
  → GPU_i 可以直接 load/store GPU_j 的 HBM 地址
  → 不需要显式 send/recv，是内存操作而非通信操作

这在 NVIDIA NVLink 上也存在，但 AMD 的 XGMI 集成更深：
  MI300X 的 XGMI 连接了芯粒（chiplet）间的统一内存空间
  192GB HBM3e 对节点内所有 GPU 呈现为一个逻辑统一内存池
  → "跨 GPU 读写"的代价接近"本卡大容量 L3 Cache 访问"
```

### 1.3 机会：用 AMD 硬件特性重新设计 FSEP

```
LAER-MoE FSEP 的通信瓶颈（NVIDIA 上）：

  Step 4 - ReduceScatter：
    GPU_i 的 partial_out_i → 通过 NCCL → 聚合到目标 GPU
    通信量 = T_E × H per GPU
    延迟 = T_E × H × 2B / NVLink_BW ≈ 0.5~2ms

  Step 2 - A2A Dispatch（含 FSEP 的 token 广播）：
    所有 GPU 接收发往其 Expert 分片的 token
    通信量 = T × H（可能比传统 EP 高）
    延迟 ≈ 5~15ms（跨节点部分）

  两者都用 NCCL → 带宽竞争，无法真正并发

AMD 的机会：
  ReduceScatter → 改用 XGMI P2P Direct Access（消除 NCCL 开销）
  A2A Dispatch  → 走 RDMA over RoCE（独立路径，不与 RS 竞争）
  → 两类通信真正并发，总有效带宽 = XGMI_BW + RDMA_BW
```

---

## 2. AMD MI300X 硬件特性深度解析

### 2.1 XGMI（Infinity Fabric）的关键参数

```
MI300X XGMI 规格：
  拓扑：8 GPU 全互联（All-to-All，非 Ring）
  带宽：每条链路 ~112 GB/s（双向），8 GPU 总计 ~896 GB/s 聚合
  延迟：~2~4μs（节点内 GPU 间）
  协议：支持 HSA（Heterogeneous System Architecture）P2P

与 NVIDIA NVLink 4.0 对比：
  NVLink 4.0（H100 SXM）：900 GB/s，Ring + Switch 拓扑
  XGMI（MI300X）：      896 GB/s，All-to-All 全互联拓扑

  关键差异：
  NVLink 走 NVSwitch 中转，P2P 有额外跳转
  XGMI 是直接 Die-to-Die 连接，无中转节点
  → XGMI 的 P2P 延迟更低，且支持更灵活的访问模式
```

### 2.2 P2P Direct Access 的工作原理

```
XGMI P2P Direct Access：

  传统 ReduceScatter（通过 RCCL）：
    GPU_0: partial_out_0[T_E, H]
      → RCCL 打包 → 发送到网络 → GPU_1 接收 → RCCL 解包 → 加法
    延迟组成：打包 + 网络传输 + 解包 + 内存写入

  AMD P2P Direct Access：
    GPU_1 的 HIP Kernel 直接执行：
      float* remote_ptr = (float*)get_peer_mem_ptr(gpu_0, partial_out_0)
      local_sum[idx] += remote_ptr[idx]   // 直接读取 GPU_0 的 HBM
    延迟组成：只有内存访问延迟（~3μs），无通信协议开销

  带宽利用：
    8 GPU 同时互读 → 聚合读带宽 = 8 × 单链路带宽 ≈ 896 GB/s
    VS RCCL ReduceScatter：受协议开销限制，实际带宽利用率 ~60~70%
    → P2P 方案带宽利用率可达 ~90%
```

### 2.3 HBM3e 的内存带宽优势

```
MI300X HBM3e 规格：
  总容量：192 GB（8 HBM3e 堆叠）
  内存带宽：5.3 TB/s（理论峰值）
  实际训练中：~4.0~4.5 TB/s（80%+ 利用率可达）

对 FSEP 的意义：

  Expert GEMM 是内存密集型操作（对于长 token 序列）：
    Arithmetic Intensity = 2 × T × H × F / (T×H + H×F + F×H) × sizeof(BF16)
    当 T 较小（稀疏路由）时，AI < roofline 拐点 → 内存带宽限制

  MI300X HBM3e 5.3 TB/s vs H100 SXM 3.35 TB/s：
    → 内存带宽提升 ~58%
    → 对于内存带宽限制的 Expert GEMM，吞吐直接提升 ~58%

  Expert 参数分片存储的好处（在高带宽 HBM 上）：
    分片参数 = 1/S × 完整参数大小
    → 每次 GEMM 需要从 HBM 加载的参数量减少 S 倍
    → 若 Expert GEMM 是内存带宽限制，则吞吐提升 S 倍
    → 结合 5.3 TB/s HBM，AMD-FSEP 的 Expert 计算效率远超 NVIDIA
```

### 2.4 MFMA 指令集的计算优势

```
AMD MFMA（Matrix Fused Multiply-Add）指令：

  vs NVIDIA wmma（Warp Matrix Multiply Accumulate）：
    wmma：以 Warp（32 线程）为单位操作固定大小矩阵块
    MFMA：以单指令完成更大矩阵块的 FMA，更高的指令级并行

  MI300X MFMA 性能参数（BF16）：
    峰值：1307.4 TFLOPS（BF16）
    H100 SXM：989 TFLOPS（BF16 with sparsity: 1979）
    → 密集计算下 MI300X MFMA 峰值约为 H100 的 1.32x

  FP8 支持（训练加速）：
    MI300X MFMA FP8：2614.9 TFLOPS
    H100 FP8：3958 TFLOPS（含稀疏）/ 1979 TFLOPS（密集）
    → FP8 场景 H100 领先，但 BF16 场景 MI300X 更优
    → MoE 训练通常用 BF16，MI300X 有优势

  对 Expert GEMM 的意义：
    每个 Expert 的 GEMM：[T_E, H] × [H, F/S] → [T_E, F/S]
    较小的分片矩阵 [H, F/S] 更适合 MFMA 的矩阵块大小
    → 分片 GEMM 的 MFMA 利用率高于完整 GEMM
```

---

## 3. 核心创新一：XGMI In-Place Reduction

### 3.1 标准 ReduceScatter 的问题

```
标准 FSEP 的 ReduceScatter（Step 4）：

  目的：将 S 块 GPU 各自计算的 partial_out_i[T_E, H] 聚合
  方式：RCCL ReduceScatter
  
  执行流：
    GPU_i 计算完 partial_out_i ← GEMM 结束
    等待所有 GPU 计算完成（同步点）
    RCCL 协调：发送 partial_out_i，接收其他 GPU 的 partial
    本地求和
    结果：每 GPU 持有 T_E/S 个 token 的完整 Expert 输出

  问题：
    ① RCCL ReduceScatter 有协议 overhead（~10~20% 带宽损失）
    ② 等待所有 GPU 计算完成才能发起通信（同步屏障）
    ③ RCCL 通信占用 XGMI 带宽，与 A2A（若并发）竞争
```

### 3.2 XGMI In-Place Reduction 设计

```
AMD-FSEP 的 In-Place Reduction：

核心思想：
  不通过 RCCL 发送数据，而是让 GPU_i 直接"读取" GPU_j 的 partial_out_j
  在本地 HIP Kernel 内完成求和

实现机制：
  Step 1：Expert GEMM 完成后，GPU_i 的 partial_out_i 在 HBM 中
  Step 2：GPU_i 注册 partial_out_i 的内存为 XGMI 可访问区域
           hipDeviceEnablePeerAccess(peer_gpu, 0)
           hipMemAdvise(partial_out_i, PEER_ACCESS_ENABLED)
  Step 3：GPU_i 启动 In-Place Reduction Kernel：

    __global__ void inplace_reduce_kernel(
        float* local_partial,      // 本卡的 partial_out_i
        float** peer_partials,     // 其他卡的 partial 指针（通过 XGMI 可访问）
        float* output,             // 本卡负责的 T_E/S 行的完整输出
        int T_chunk, int H,
        int my_rank, int world_size
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= T_chunk) return;

        // 本卡负责 [my_rank * T_chunk, (my_rank+1) * T_chunk) 行
        int global_row = my_rank * T_chunk + row;

        float sum = local_partial[global_row * H + threadIdx.y];
        for (int peer = 0; peer < world_size; peer++) {
            if (peer != my_rank) {
                // 直接读取 peer GPU 的 HBM（通过 XGMI，无 RCCL）
                sum += peer_partials[peer][global_row * H + threadIdx.y];
            }
        }
        output[row * H + threadIdx.y] = sum;
    }

  Step 4：输出 output[T_E/S, H] 即为本卡负责部分的完整 Expert 结果
```

### 3.3 性能对比分析

```
通信延迟对比（T_E = 400 tokens，H = 4096，S = 8，BF16）：

数据量：T_E × H × 2B = 400 × 4096 × 2 = 3.2 MB per GPU

方案 A：标准 RCCL ReduceScatter
  XGMI 有效带宽（含协议开销）：~600 GB/s
  时间 ≈ 3.2 MB × 8 / 600 GB/s ≈ 0.043 ms
  + 协议 overhead（同步、打包）：~0.02 ms
  总计：~0.06 ms

方案 B：XGMI In-Place Reduction
  每 GPU 读取 7 个对端的 3.2 MB：7 × 3.2 = 22.4 MB
  XGMI P2P 读带宽：~112 GB/s（单链路）× 7 个对端 ≈ 784 GB/s（理想）
  实际（考虑争用）：~500 GB/s
  时间 ≈ 22.4 MB / 500 GB/s ≈ 0.045 ms

  但关键区别：
  ① In-Place Reduction 与 GEMM 可以 overlap（流水线执行）
     GEMM 计算前 S-1 个 Tile 时，Reduction 处理已完成的 Tile
  ② 无同步屏障：不需要等所有 GPU 的 GEMM 完成再发起 RS
  ③ RDMA 通道完全空闲：A2A 可以与 In-Place Reduction 完全并发

有效延迟（考虑 overlap）：
  方案 A：0.06 ms（串行，且占用 XGMI 带宽）
  方案 B：≈ 0 ms（隐藏在 GEMM 计算中）+ A2A 并发不受影响

结论：In-Place Reduction 在延迟上接近零，带宽利用上更优
```

---

## 4. 核心创新二：HIP Wavefront 专用化

### 4.1 Comet 的 CUDA Warp Specialization 回顾

```
Comet（MLSys '25）在 CUDA 上实现了：
  Warp Group 0,1,2 → Expert GEMM（wmma 指令）
  Warp Group 3     → RDMA tile send（通信 warp）

  在一个 CUDA Kernel 内，计算和通信并发
  → 85~95% 的计算-通信重叠率

限制：
  ① CUDA 专属，未移植到 HIP
  ② 使用 NVIDIA 特有的 LDGSTS 指令（异步内存拷贝）
  ③ 没有利用 AMD 的 XGMI P2P 能力
```

### 4.2 AMD-FSEP 的 HIP Wavefront 专用化设计

```
AMD CU（Compute Unit）的线程组织：
  每个 CU：4 个 SIMD32（每个 SIMD32 = 1 个 Wavefront = 64 线程）
  一个 Workgroup 可以包含多个 Wavefront

  MI300X 的 CU 数量：304 CU（远多于 H100 的 132 SM）
  → 更多的并发 Workgroup → 更大的 Wavefront 专用化空间

AMD-FSEP Expert Kernel 的 Wavefront 分工：

  Wavefront Group A（计算，3/4 CU）：
    → 执行 Expert GEMM（MFMA 指令）
    → 计算 partial_out_i 的每个 Tile

  Wavefront Group B（通信+归约，1/4 CU）：
    → 监听 Group A 完成的 Tile
    → 通过 XGMI P2P 读取其他 GPU 的对应 Tile
    → 在本地完成部分和累加（In-Place Reduction）
    → 写入最终输出 Buffer

  LDS（Local Data Share，64KB/CU on MI300X）的作用：
    Group A 将计算完的 Tile 写入 LDS
    Group B 从 LDS 读取（高速），同时向 XGMI 发起 P2P 读请求
    → LDS 是两个 Wavefront 组之间的高速缓冲

伪代码（HIP）：

__global__ void amd_fsep_expert_kernel(
    float* X,              // 输入 tokens [T_E, H]
    float* W_up, W_down,   // Expert 参数分片 [H, F/S], [F/S, H]
    float** peer_partials, // 其他 GPU 的 partial 指针（XGMI 可访问）
    float* output,         // 输出 [T_E/S, H]
    int tile_size, int T_E, int H, int F_shard
) {
    __shared__ float lds_tile[TILE_SIZE * H_TILE];  // 64KB LDS

    int wf_id = __builtin_amdgcn_workitem_id_x() / 64;
    bool is_compute_wf = (wf_id < NUM_COMPUTE_WF);
    bool is_reduce_wf  = (wf_id >= NUM_COMPUTE_WF);

    if (is_compute_wf) {
        // Wavefront Group A：计算 GEMM Tile
        for (int tile = compute_wf_tile_start; tile < total_tiles; tile++) {
            // 用 MFMA 指令计算矩阵乘
            __builtin_amdgcn_mfma_f32_32x32x8bf16(
                acc, X_tile, W_up_tile, acc, 0, 0, 0
            );
            // 激活函数（SiLU）
            act_tile = silu(acc) * W_gate_result;
            // Down proj
            __builtin_amdgcn_mfma_f32_32x32x8bf16(
                out_acc, act_tile, W_down_tile, out_acc, 0, 0, 0
            );
            // 写入 LDS，通知 Reduce Wavefront
            lds_tile[tile % LDS_SLOTS] = out_acc;
            __s_sendmsg(MSG_TILE_READY | tile);
        }
    }

    if (is_reduce_wf) {
        // Wavefront Group B：从 LDS 读取 + XGMI P2P 归约
        while (!all_tiles_reduced()) {
            int ready_tile = wait_for_tile_ready();
            float local_val = lds_tile[ready_tile % LDS_SLOTS];

            // 通过 XGMI P2P 读取其他 GPU 的 partial（直接内存访问）
            float sum = local_val;
            for (int peer = 0; peer < world_size - 1; peer++) {
                sum += peer_partials[peer][ready_tile * H + col_idx];
                // ^ 这是直接内存读取，不经过 RCCL，延迟 ~3μs
            }
            output[my_row * H + col_idx] = sum;
        }
    }
}
```

### 4.3 与 Comet CUDA 版的关键差异

```
差异维度              Comet（CUDA）               AMD-FSEP（HIP）
──────────────────────────────────────────────────────────────────────
通信 Warp 的工作     RDMA send（发送 tile）       XGMI P2P read（读取 peer tile）
延迟来源             网络传输延迟（5~20ms）        内存访问延迟（~3μs）
带宽来源             RDMA（200 Gbps）             XGMI（896 GB/s 聚合）
通信方向             主动发送（push）              被动读取（pull）
与计算的重叠方式     发 tile → 发起 RDMA send     GEMM tile → P2P read + 累加
LDS 大小             48 KB/SM（CUDA L1）          64 KB/CU（更大，更大 Tile）
架构指令             wmma + LDGSTS               MFMA + P2P load intrinsic

核心优势：
  AMD 版的"通信 Wavefront"做的是内存读取而非网络通信
  → 延迟从 5~20ms 降到 ~3μs（100x 改善）
  → 可以在 GEMM 计算的极短间隙内完成
  → 实际重叠率可超过 Comet 的 90%，接近 99%
```

---

## 5. 核心创新三：双通道并发调度

### 5.1 通信路径的完整分析

```
AMD-FSEP 的完整通信清单（一个 MoE Block，EP=8 节点内）：

Operation              通信量           路径         延迟（估算）
─────────────────────────────────────────────────────────────────────
A2A Dispatch（节点内）  T × H            XGMI         ~5ms（受 token 数影响）
A2A Dispatch（跨节点）  T_remote × H    RDMA         ~15ms
FSEP In-Place RS       T_E × H × S     XGMI P2P     ~0ms（隐藏在 GEMM 中）
A2A Gather（节点内）    T × H            XGMI         ~5ms
A2A Gather（跨节点）    T_remote × H    RDMA         ~15ms
TP AllReduce           token × H        XGMI         ~2ms
FSEP Re-layout（异步） param_shard      XGMI         后台，不在关键路径

XGMI 通道总负载 ≈ A2A(节点内) + TP AR + RS（被 GEMM 覆盖）
RDMA 通道总负载 ≈ A2A(跨节点)

由于两条路径物理隔离：
  XGMI 满负载 + RDMA 满负载 → 总带宽利用 = XGMI_BW + RDMA_BW
  （NVIDIA 方案：共享 IB，总带宽 ≤ IB_BW，相互竞争）
```

### 5.2 双通道并发的调度设计

```
AMD-FSEP 双通道调度时间线：

时间轴 ────────────────────────────────────────────────────→

XGMI Stream（节点内高带宽）：
  [A2A_Dispatch_intra] ─── [TP_AllReduce] ─────────────────

RDMA Stream（跨节点）：
  ──────────── [A2A_Dispatch_inter] ─── [A2A_Gather_inter] ─

Compute Stream：
  [Gate] ─── [Expert_GEMM + In-Place_RS（合并 Kernel）] ─── [Merge]

同步点：
  SyncA：Compute wait XGMI（A2A_Dispatch_intra 完成后，Expert GEMM 才能启动）
  SyncB：Compute wait RDMA（A2A_Dispatch_inter 完成后，跨节点 token 才就绪）
  SyncC：Merge wait Compute + XGMI（TP AR 完成 + Expert 输出就绪）

关键：XGMI Stream 和 RDMA Stream 完全并发，互不阻塞
  → 节点内 A2A 与跨节点 A2A 同时进行
  → TP AllReduce（XGMI）与 A2A Gather（RDMA）同时进行
  → 每条物理链路都在满负载工作
```

### 5.3 带宽利用率的理论分析

```
设：
  B_XGMI = 896 GB/s（节点内聚合带宽）
  B_RDMA = 400 Gbps ≈ 50 GB/s（跨节点带宽）
  T = 2048 tokens，H = 4096，EP = 8（8 节点），K = 4（Top-4 路由）

每 step 的通信量：
  A2A Dispatch 总量：T × H × K × 2B = 2048 × 4096 × 4 × 2 ≈ 64 MB
  A2A Gather 总量：  同上 ≈ 64 MB
  TP AllReduce：     T × H × 2B ≈ 16 MB（节点内）

NVIDIA 方案（共享 IB 200 Gbps）：
  时间 ≈ (64 + 64 + 16) MB / 25 GB/s ≈ 5.8 ms（串行最优）

AMD 双通道方案：
  XGMI 负载（节点内 A2A + TP AR）≈ 0.3 × 64 + 16 = 35.2 MB
  RDMA 负载（跨节点 A2A）≈ 0.7 × 64 = 44.8 MB

  时间 ≈ max(35.2/896, 44.8/50) ms ≈ max(0.04, 0.9) ms ≈ 0.9 ms

理论加速比：5.8 ms / 0.9 ms ≈ 6.4x（通信时间）

实际加速比（考虑各种 overhead）：预期 2~4x 端到端
```

---

## 6. 核心创新四：Re-layout 的 AMD 原生加速

### 6.1 LAER-MoE Re-layout 的代价

```
LAER-MoE 的 Expert Re-layout 过程（NVIDIA 版）：

  触发条件：Load Planner 检测到不均衡 → 决定将 E2 的分片度从 S=2 扩展到 S=4
  
  执行过程：
    ① 确定需要搬迁的 Expert 分片（哪些 GPU 要新增 E2 的分片）
    ② 通过 NCCL All-to-All 发起参数搬迁
       搬迁量：Expert 参数大小 × 搬迁分片数
               = H × F / S × sizeof(BF16) per shard
               = 4096 × 14336 / 2 × 2 ≈ 59 MB per shard
    ③ 搬迁发生在反向传播期间（异步）
    ④ 反向完成后 double buffer 内存释放

  时间开销（NVIDIA IB 200 Gbps）：
    59 MB / 25 GB/s ≈ 2.4 ms（若在反向传播的 12ms 内完成，不在关键路径）
    但若频繁触发（每 K=50 步一次），平均每步额外 2.4/50 = 0.05 ms
```

### 6.2 AMD 的 XGMI 直接 Memcpy 加速

```
AMD-FSEP 的 Re-layout 优化：

  利用 XGMI P2P 做 Expert 参数搬迁（而非 RCCL All-to-All）：

  传统方式（RCCL）：
    GPU_src 准备数据 → RCCL 打包 → 网络传输 → GPU_dst 解包 → 写入 HBM
    延迟 ≈ 59 MB / 25 GB/s ≈ 2.4 ms

  AMD XGMI Memcpy：
    hipMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, 59MB)
    → 直接通过 XGMI 做 HBM-to-HBM 拷贝
    → 带宽利用：~112 GB/s（单链路，点对点）
    延迟 ≈ 59 MB / 112 GB/s ≈ 0.53 ms

  加速比：2.4 ms → 0.53 ms = 4.5x

  更进一步：多个 Expert 分片的 Re-layout 可以并行进行
    E2 的分片搬到 GPU4：hipMemcpyPeer（XGMI 链路 4）
    E2 的分片搬到 GPU5：hipMemcpyPeer（XGMI 链路 5）
    → 两个搬迁用不同 XGMI 链路，完全并发
    → 多分片 Re-layout 时间不增加（只取决于单次搬迁时间）

  AMD Re-layout 时间：~0.53 ms（远小于反向传播 ~8ms 的窗口）
  内存峰值增加：< 3%（vs LAER-MoE 的 5~10%）
```

---

## 7. AMD-FSEP 完整计算流程

### 7.1 前向传播（Forward）

```
AMD-FSEP Forward Pass（8 GPU，EP=8，节点内）：

参数布局：
  每块 GPU 持有所有 Expert 的 1/S 分片
  Expert E_j 的参数：W_up_sj[H, F/S], W_down_sj[F/S, H] 在 GPU_j mod S

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1：Gate 计算（本地，无通信）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  routing = TopK(softmax(X @ W_gate), k=K)
  同时：启动 Load Planner 的负载统计更新（异步后台）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2：双通道 A2A Dispatch（XGMI + RDMA 并发）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  XGMI Stream：发送给同节点 Expert 分片的 token
  RDMA Stream：发送给跨节点 Expert 分片的 token（若有 EP > 8）

  等待：两个 Stream 都完成（SyncNode）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 3：AMD-FSEP Expert Kernel（分片 GEMM + In-Place RS 融合）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  每块 GPU 启动 amd_fsep_expert_kernel：

  Wavefront Group A（3/4 CU）：
    对收到的 token 执行分片 GEMM（MFMA 指令）
    Up-proj → SiLU → Down-proj
    结果写入 LDS

  Wavefront Group B（1/4 CU）：
    从 LDS 读取已完成的 GEMM Tile
    通过 XGMI P2P 读取其他 GPU 的对应 Tile
    累加，写入输出 Buffer（output[T_E/S, H]）

  → 两组 Wavefront 并发执行，In-Place Reduction 被完全隐藏在 GEMM 中

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 4：双通道 A2A Gather（XGMI + RDMA 并发）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Expert 输出送回 token 原始所在 GPU
  同时：TP AllReduce（Attention 输出）也在 XGMI 上进行（独立链路）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 5：加权合并 + Residual Add
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  output = Σ_k (routing_weight_k × expert_output_k)
  + x_residual → 下一 Block

后台（异步）：
  Load Planner 分析负载统计，若触发 Re-layout：
    hipMemcpyPeer 完成参数搬迁（XGMI 直接 Memcpy，~0.5ms）
    在反向传播期间完成，不影响前向
```

### 7.2 与 LAER-MoE 的逐步对比

```
Step       LAER-MoE（NVIDIA）              AMD-FSEP（MI300X）
──────────────────────────────────────────────────────────────
A2A Disp   NCCL All-to-All                双通道：XGMI + RDMA 并发
GEMM       cuBLAS wmma                    MFMA 指令，高 BF16 峰值
RS         NCCL ReduceScatter（5%延迟）    XGMI P2P 直接访问（~0ms）
GEMM+RS    串行执行                       WaveFront 专用化，完全融合
Re-layout  NCCL A2A（~2.4ms）             XGMI Memcpy（~0.53ms）
总通信      约 3 × T × H（串行）           约 2 × T × H（并发，RS 隐藏）
```

---

## 8. 性能模型

### 8.1 理论 MFU 分析

```
设：Expert GEMM 的理论 FLOP = 2 × T_E × H × F × N_layers

理论 MFU = Actual_FLOP / (Peak_FLOP × step_time)

LAER-MoE 在 H100 上（推测）：
  Peak FLOP（BF16）：989 TFLOPS
  Step time ≈ 计算时间 + 通信时间（约 50:50）
  MFU ≈ 42%（LAER-MoE 论文数据）

AMD-FSEP 在 MI300X 上（预期）：
  Peak FLOP（BF16）：1307 TFLOPS（H100 的 1.32x）
  Step time：
    计算时间：1307/989 ≈ 0.76x（MI300X 计算更快）
    通信时间：
      NVIDIA：通信约占 50% = 0.5 × T_nvidia
      AMD：
        RS 被隐藏（≈0），A2A 并发（XGMI+RDMA），通信约占 20%
        通信时间 ≈ 0.2 × T_amd
    总 step 时间：
      T_amd ≈ 0.76 × T_nvidia_compute + 0.2 × T_amd
      解：T_amd ≈ 0.76 / 0.8 × T_nvidia_compute ≈ 0.95 × T_nvidia_compute
      VS T_nvidia = T_nvidia_compute + 0.5 × T_nvidia_compute = 1.5 × T_nvidia_compute
      T_amd ≈ 0.63 × T_nvidia

  MFU_amd = (1307 / 989) × (T_nvidia / T_amd) × MFU_nvidia
           ≈ 1.32 × 1.59 × 42% ≈ 88%（理想上界）

实际预期（考虑各种 overhead）：55~65% MFU

对比：
  LAER-MoE（NVIDIA H100）：~42% MFU
  AMD-FSEP（MI300X）：     ~55~65% MFU（预期）
  提升幅度：约 30~55%
```

### 8.2 通信-计算重叠率分析

```
重叠率 = 1 - (暴露通信时间 / 总通信时间)

传统 EP（无 overlap）：   重叠率 ≈ 0%
LAER-MoE（NVIDIA）：      重叠率 ≈ 30~40%（细粒度 A2A，但 RS 串行）
Comet（CUDA）：           重叠率 ≈ 85~95%（Tile 级，RDMA 方向）
AMD-FSEP（目标）：        重叠率 ≈ 95~99%

来源分析：
  RS 被完全隐藏（In-Place Reduction，~99%）：XGMI P2P 速度极快
  A2A 与 TP AR 并发（双通道，~100%）：物理路径独立
  A2A 内部 Tile-level overlap（Wavefront 专用化，~90%）：类 Comet
```

---

## 9. 论文贡献点

### 9.1 系统贡献

```
S1：AMD-FSEP 系统的完整设计与实现
  ① XGMI In-Place Reduction（零延迟 ReduceScatter）
  ② HIP Wavefront 专用化 Expert Kernel（MFMA + P2P）
  ③ 双通道并发调度（XGMI Stream + RDMA Stream）
  ④ XGMI Memcpy 加速的 Re-layout
  ⑤ 首个在 AMD GPU 上完整实现 FSEP 的系统

S2：AMD vs NVIDIA 的深度性能对比分析
  在 FSEP 这类 communication-heavy workload 上：
  ① 通信延迟对比（NCCL RS vs XGMI P2P）
  ② 带宽利用对比（单通道 vs 双通道并发）
  ③ 端到端 MFU 对比（含硬件差异的归因分析）
  论文价值：为社区提供"AMD 在 MoE 训练上真正有多强"的权威数据

S3：硬件感知的通信调度框架
  设计一个通用的"硬件拓扑感知调度接口"：
    输入：通信操作类型 + 进程组拓扑
    输出：最优硬件路径（XGMI / RDMA / P2P）+ Stream 分配
  可被其他 MoE 系统复用（通用性贡献）
```

### 9.2 理论贡献

```
T1：双通道 MoE 通信模型
  建立 AMD 双通道架构下的通信代价模型：
    C_total = max(C_XGMI, C_RDMA) + C_compute（理想并发）
  vs 单通道模型：
    C_total = C_comm_all + C_compute（串行）
  证明：在负载均衡时，双通道模型使通信成为非关键路径

T2：XGMI P2P Reduction 的正确性与性能分析
  证明：In-Place P2P Reduction 在异步场景下的正确性条件
  分析：P2P 带宽与 RCCL ReduceScatter 的带宽效率对比
        （考虑 memory contention 和 cache effects）

T3：Wavefront 专用化的效率边界
  分析：Wavefront Group A（GEMM）和 Group B（P2P Reduction）的
        最优比例（如 3:1 vs 2:1 vs 1:1）
  理论：当 P2P 延迟 << GEMM Tile 时间时，Group B 可以极小化
```

### 9.3 实验贡献

```
E1：XGMI P2P vs RCCL ReduceScatter 微基准
  在 8×MI300X 节点上，对不同数据大小的 RS 延迟和带宽测试
  → 量化 P2P 优势的 Pareto 边界（数据量多大时 P2P 更优）

E2：双通道并发的有效带宽测试
  同时发起 XGMI 操作和 RDMA 操作
  → 证明两者真正独立（无带宽竞争）
  → 对比 NVIDIA NVSwitch 场景的竞争情况

E3：端到端 MoE 训练吞吐
  模型：Mixtral 8x7B / DeepSeek-V3 scale（1/4）
  对比：LAER-MoE（H100）/ Megatron EP（MI300X）/ AMD-FSEP（MI300X）
  → 核心数据：AMD-FSEP 相比 LAER-MoE 的绝对提升

E4：扩展性测试（多节点）
  8 → 64 → 256 GPU
  → 验证 RDMA 跨节点场景下双通道调度的扩展性
```

---

## 10. 实验设计方案

### 10.1 硬件配置

```
主要测试平台：
  节点 A：8 × AMD MI300X（同节点，XGMI 全互联）
  集群 B：4 节点 × 8 × MI300X = 32 GPU（RDMA over RoCE 跨节点）
  对比平台：8 × NVIDIA H100 SXM（NVLink，相同网络配置）

软件环境：
  ROCm 6.x，RCCL 2.x，hipBLASLt
  PyTorch 2.x（ROCm 后端）
  自研 AMD-FSEP 实现（基于 ROCflow）
```

### 10.2 消融实验矩阵

```
配置                            启用特性                预期 MFU
──────────────────────────────────────────────────────────────────
Baseline：Megatron EP            基础 EP                 ~30%
+ FSEP（RCCL RS）                FSEP + RCCL             ~38%
+ XGMI P2P RS                   FSEP + P2P              ~42%
+ Wavefront 专用化               FSEP + P2P + WF Spec    ~48%
+ 双通道调度                      全部 + 双通道            ~54%
+ Re-layout（XGMI Memcpy）       完整 AMD-FSEP           ~57%
Full AMD-FSEP                    全部 + hipBLASLt 调优   ~60%

对比基线：LAER-MoE on H100 ≈ 42% MFU
目标：AMD-FSEP on MI300X ≈ 55~60% MFU（提升约 30~43%）
```

### 10.3 关键 Benchmark

```
Benchmark 1：XGMI P2P vs RCCL 微基准
  数据大小：1 MB → 1 GB（以 2x 递增）
  指标：延迟（μs）、带宽（GB/s）
  预期结论：< 100MB 时 P2P 大幅优于 RCCL（延迟 10x）

Benchmark 2：双通道并发有效带宽
  同时发起：XGMI AllReduce（64 MB）+ RDMA AllToAll（64 MB）
  指标：实际观测带宽 vs 理论峰值
  预期结论：两者带宽之和接近两条路径各自峰值之和

Benchmark 3：Expert Kernel 计算-P2P 重叠率
  对不同 Tile 大小（16, 32, 64, 128 token per tile）
  指标：Omniperf 测量的 XGMI P2P 时间 vs GEMM 时间的重叠比例
  预期结论：Tile 越大，重叠率越高（接近 99%）

Benchmark 4：端到端训练吞吐（核心 Benchmark）
  Mixtral 8x7B，64 GPU（8 节点）
  指标：tokens/sec，MFU
  对比：3 个基线 + AMD-FSEP + 消融版本
```

---

## 11. 实现路线图

### 11.1 阶段划分

```
Phase 0（2~4 周）：XGMI P2P 特性探索
  目标：确认 MI300X P2P 的实际性能参数
  工作：
    ① 编写 P2P 微基准（不同数据大小，不同 GPU 对）
    ② 测量 hipMemcpyPeer vs RCCL AllReduce 的性能交叉点
    ③ 测试 P2P 在 HIP Kernel 内直接读取的可行性
  交付：性能数据报告，验证 P2P 优势的数据支撑

Phase 1（1~2 个月）：XGMI In-Place Reduction
  目标：实现并验证 P2P Reduction 替代 RCCL ReduceScatter
  工作：
    ① 实现 inplace_reduce_kernel（基础版，无 Wavefront 专用化）
    ② 在 8×MI300X 节点上验证正确性和性能
    ③ 对比 RCCL RS 的性能（消融实验 Benchmark 1、2）
  交付：P2P Reduction 的实现 + 性能对比数据

Phase 2（2~3 个月）：HIP Wavefront 专用化 Expert Kernel
  目标：实现 GEMM + P2P Reduction 融合 Kernel
  工作：
    ① 实现 amd_fsep_expert_kernel（Wavefront 专用化）
    ② 调优 MFMA 指令的 Tile 大小（LDS 利用率）
    ③ 测量重叠率（Omniperf profiling）
    ④ 实现双通道调度（XGMI Stream + RDMA Stream）
  交付：融合 Kernel 实现 + 重叠率数据（Benchmark 3）

Phase 3（2~3 个月）：完整 AMD-FSEP 系统
  目标：集成所有优化，完成端到端 MoE 训练测试
  工作：
    ① 集成 Load-Adaptive Planner（XGMI Memcpy Re-layout）
    ② 多节点扩展（RDMA + XGMI 双通道）
    ③ 与 hipBLASLt autotuning 联调
    ④ 完整消融实验和端到端 Benchmark
  交付：完整系统 + 所有 Benchmark 数据

Phase 4（1~2 个月）：论文写作
  目标：整理数据，撰写论文
  目标会议：EuroSys '27 / ASPLOS '27 / ATC '26
```

### 11.2 关键技术风险

```
风险 R1：XGMI P2P 在 Kernel 内直接读取的稳定性
  问题：HIP Kernel 内的 P2P 读取在某些场景下可能触发硬件错误
  缓解：Phase 0 充分测试，发现问题时退回 hipMemcpyPeer（异步拷贝）

风险 R2：Wavefront 专用化的利用率平衡
  问题：若 P2P 读取速度远快于 GEMM，Group B 的 Wavefront 利用率极低
  缓解：动态调整 Group A/B 的 Wavefront 比例，或让 Group B 做更多工作
        （如负责 Re-layout 的轻量计算）

风险 R3：多节点扩展时 RDMA 成为瓶颈
  问题：当 EP 规模增大（>8 GPU），RDMA 的 A2A 延迟增加，双通道优势减弱
  缓解：引入 Comm-Aware Routing（方向 A），减少跨节点 A2A 流量

风险 R4：与 ROCm 版本的兼容性
  问题：不同 ROCm 版本的 P2P 行为可能不一致
  缓解：明确标注测试使用的 ROCm 版本，提供版本检测逻辑
```

---

## 12. 差异定位

### 12.1 与 LAER-MoE 的差异

```
维度                LAER-MoE（NVIDIA）              AMD-FSEP（本工作）
──────────────────────────────────────────────────────────────────────
硬件平台            H100 SXM                        MI300X（首次）
ReduceScatter       NCCL（协议 overhead）            XGMI P2P（近零延迟）
GEMM+RS 关系        串行（RS 在 GEMM 后）            融合（Wavefront 专用化）
通信路径            单通道（NCCL 统一）              双通道（XGMI + RDMA）
Re-layout 速度      NCCL A2A（~2.4ms）              XGMI Memcpy（~0.5ms）
峰值计算力          989 TFLOPS（BF16）              1307 TFLOPS（BF16 +32%）
预期 MFU            ~42%                            ~55~60%（预期）

核心叙事：
  LAER-MoE 解决了 FSEP 的算法问题（负载均衡）
  AMD-FSEP 解决了 FSEP 在 AMD 硬件上的实现问题（充分利用 AMD 特性）
  两者不是竞争，AMD-FSEP 是 LAER-MoE 的"AMD 原生高性能实现"
```

### 12.2 与 Comet 的差异

```
Comet（MLSys '25）：
  硬件：NVIDIA H100
  通信方向：计算 → RDMA 发送（push 模型）
  延迟来源：RDMA 网络（5~20ms）
  重叠率：85~95%

AMD-FSEP Wavefront 专用化：
  硬件：AMD MI300X（首次 HIP 实现）
  通信方向：P2P 读取对端数据（pull 模型）
  延迟来源：XGMI 内存访问（~3μs，比 RDMA 快 100x）
  重叠率：95~99%（P2P 延迟极短，几乎完全隐藏）

核心差异：
  Comet 是"发出去等确认"（push），受网络延迟限制
  AMD-FSEP 是"直接读邻居"（pull），受内存延迟限制
  → AMD P2P 的延迟是 RDMA 的 1/100，重叠率更高
```

### 12.3 论文一句话定位

> **AMD-FSEP 是首个利用 AMD MI300X 的 XGMI P2P 直接内存访问和双通道通信架构，将 FSEP 的 ReduceScatter 通信延迟降至接近零、并实现真正双路径并发的 MoE 训练系统，在 AMD MI300X 上实现了超越 LAER-MoE 在 NVIDIA H100 上的训练效率。**

---

*方向 C 深度分析整理于 2026-03-09 | ROCflow 框架研究讨论 | AIInfra-Book*
