# MonolithMoE super-kernel vs PyTorch+RCCL baseline（mi355-gpu-26，xiaoming-dev）

**时间（节点日志）**: PyTorch 打印为 `W0514 03:39:xx` 形式 — 这是 **运行节点默认时区的本地时间**，**不是**我在笔记里曾误写的 UTC。若未核对 `timedatectl` / `TZ`，不应标成 UTC；若节点为 **东八区**，同一时刻的 UTC 会比本地 **早约 8 小时**（日历日也可能落在 UTC 的「前一天」），看起来就像「UTC 更早了」。本次仅以日志里的 **本地日期+时间** 作为记录锚点：**2026-05-14 ~03:39（节点本地）**。  
**节点**: `mi355-gpu-26`  
**环境**: `podman exec xiaoming-dev`，`HIP_VISIBLE_DEVICES=0..7`，ROCm **7.2** / HIP **7.2**，PyTorch **2.10.0a0+git449b176**（`/opt/venv`）  
**代码**: `/shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE`（容器内共享路径）  
**Super-kernel 构建**:

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx950 -I csrc -DMOE_K_TILE=128 \
  -o /tmp/bench_sk_cmp benchmarks/bench_super_kernel.hip
```

**DSV3 维度**: `H=7168`, `F=2048`, `epg=32`, `topk=8`, `E=256`, **8 GPU**，`num_cus=256`, `wgs_per_cu=1`, `comm_ratio=0.25`；PyTorch 脚本：`torchrun --standalone --nproc_per_node=8 benchmarks/bench_pytorch_rccl_dsv3.py`，`warmup=5`, `iters=15`。

---

## 1. 原始输出摘要

### 1.1 Super-kernel（`bench_super_kernel`）

| T_src | variant   | latency_ms / iter | bench 打印 `effective_tflops` |
|------|-----------|-------------------|-------------------------------|
| 8192 | Large     | **59.539**        | **775.6**                     |
| 512  | Generic   | **4.512**         | **639.7**                     |

### 1.2 PyTorch + RCCL（`bench_pytorch_rccl_dsv3.py`）

| T_src | median wall（all-rank **MAX**）/ ms | 打印 TFLOP/s/**GPU** | RCCL `comm_ms` | 本地 `comp_ms` |
|------|--------------------------------------|----------------------|----------------|----------------|
| 8192 | **18.802**                           | **307.0**            | 5.60           | 13.07          |
| 512  | **7.123**                            | **50.6**             | 1.29           | 5.67           |

（legacy buckets rank0：8192 → dispatch 3.91 / gemm 10.97 / combine 3.78 ms；512 → 0.77 / 5.30 / 0.93 ms。）

---

## 2. 口径对齐（必读）

- **`bench_super_kernel` 的 `effective_tflops`**（见 `benchmarks/bench_super_kernel.hip`）为  
  `flops_per_rank * NUM_GPUS / wall`，即 **8 卡算量之和 / 单 iter 墙钟**，数值上约等于「**集群总吞吐 / 1**」，**不是** PyTorch 脚本里那种「单卡有效 TFLOP/s」。
- **`bench_pytorch_rccl_dsv3.py`** 的 TFLOP/s 为 **单 rank FLOPs / 全 rank max wall**，标签里写了 **/GPU**。

**换算成「每 GPU 有效 TFLOP/s」（与 PyTorch 同一口径）**：

| T_src | SK wall_ms | SK FLOPs/rank（与 PyTorch 同公式） | SK TFLOP/s/**GPU** ≈ |
|------|------------|-------------------------------------|----------------------|
| 8192 | 59.539     | `2·(T·K)·(2·H·F + F·H)`             | **≈ 96.9**           |
| 512  | 4.512      | 同上                                 | **≈ 79.9**           |

（PyTorch：8192 → 307 T/GPU；512 → 50.6 T/GPU。）

---

## 3. 墙钟对比（同一负载形状下）

| 配置 | Super-kernel latency_ms | PyTorch max-rank wall_ms | 谁更快 |
|------|---------------------------|---------------------------|--------|
| **T_src=8192**（mbs2×seq4096×topk8） | **59.54** | **18.80** | **PyTorch+RCCL**（约 **3.2×**） |
| **T_src=512**（README 稀疏档 avg_Te≈16） | **4.51** | **7.12** | **Super-kernel**（约 **1.58×**） |

**负载差异说明**（影响绝对 ms，不影响上表数量级结论）：

- HIP bench 使用 `tests/smoke_super_kernel.hip` 里**固定的** `(rank,t,k)` 路由模式。  
- PyTorch bench 使用 **randperm 均衡随机** expert。  
- 两者 **每 rank 的 GEMM 体积**在「均衡、满 topk」假设下一致；但 **scatter/combine 与 tail atomic** 行为仍可因路由分布不同而有偏差。

---

## 4. 结论与后续

1. **稀疏短序列（T_src=512，DSV3 README 主战场）**：当前 super-kernel 在该节点上 **墙钟仍明显优于** 纯 PyTorch matmul + RCCL all_to_all 的 baseline。  
2. **满 microbatch（T_src=8192）**：`Large` 变体在本节点、当前实现下 **墙钟慢于** PyTorch+RCCL；README 中「理想融合」对比的是 **RCCL A2A 与 GEMM 串行** 的分解时间，而本 PyTorch 脚本里 **GEMM 走高度优化的库路径**，且 **super-kernel 的 tail `bf16` atomic combine + 常驻核调度** 在大 token 下成本高。  
3. **工程后续**（与仓库内已有笔记一致）：大 T_src 方向包括 **Layout E / packed expert**、**tail-wave 与 scatter–compute overlap**、**降低 combine 原子争用** 等；本次对比数据可作为后续优化的基线锚点。

---

## 5. 复现命令（节选）

```bash
ssh mi355-gpu-26 "podman exec xiaoming-dev bash -lc '
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /tmp/bench_sk_cmp \
  --tokens 8192 --hidden 7168 --ffn 2048 --epg 32 --topk 8 \
  --num-cus 256 --wgs-per-cu 1 --comm-ratio 0.25 --warmup 5 --iters 15
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /opt/venv/bin/torchrun --standalone --nproc_per_node=8 \
  benchmarks/bench_pytorch_rccl_dsv3.py \
  --tokens 8192 --topk 8 --hidden 7168 --ffn 2048 --num-experts 256 --warmup 5 --iters 15
'"
```

（512 档将 `--tokens` 改为 `512` 即可。）
