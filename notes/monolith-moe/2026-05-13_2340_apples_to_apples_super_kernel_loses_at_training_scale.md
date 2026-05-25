## 时间 / 环境

- **时间**: 2026-05-13 23:40 +0800
- **机器**: `mi355-gpu-26` (8× MI355X / gfx950 / XGMI 全互联)
- **容器**: `xiaoming-dev` (podman)
- **配置**: top_k=8, H=7168, F=2048, E=256, EP=8（DSV3 hyperparams，runtime log 确认 `moe_router_topk=8`）
- **bench 工具**:
  - super-kernel: `benchmarks/bench_super_kernel.hip` 重编（fresh build；旧 binary 5/12 的 struct 不含 `fc1_gate_up_save` 字段，size 不兼容）
  - PyTorch+RCCL: 新写 `benchmarks/bench_pytorch_rccl_dsv3.py`，bf16，per-expert `torch.matmul` + `dist.all_to_all_single`，balanced 随机路由

## 什么问题

之前所有 super-kernel perf 数字（README headline `4.82 ms / 598 TFLOP/s`）都跑在 `T_src=512 / top_k=8` 上，跟训练（`T_src=2048` 4-layer 验证）/ 生产（`T_src=8192` DSV3）的 token 数差 **4× / 16×**。用户先后两次指出过这个对齐问题，被错过。这次按训练实际配置重新跑一遍 super-kernel 和 PyTorch+RCCL baseline，确认到底哪个数字才是真实的。

## 做了什么

1. **rebuild super-kernel bench**（kernel struct 改了，必须重编）：
   ```
   hipcc -std=c++17 -O3 --offload-arch=gfx950 -I csrc -DMOE_K_TILE=128 \
       -o bench_sk_apples benchmarks/bench_super_kernel.hip
   ```
2. **跑 super-kernel** @ T_src ∈ {512, 2048, 8192}，每档 sweep `comm_ratio` ∈ {0.18, 0.225, 0.25, 0.30}，warmup=5 / iters=30
3. **跑 PyTorch+RCCL** @ T_src ∈ {512, 2048, 8192}，同 H/F/E/top_k，warmup=5 / iters=20

原始 log：
- super-kernel: `benchmarks/results/dsv3_apples_to_apples_20260513_1537.txt`
- PyTorch+RCCL: `/tmp/rccl_bench_logs/*/attempt_0/0/stdout.log`（rank-0 print）

## 取得了什么效果

### 1) super-kernel 单跑（标准 bench 工况，无训练干扰）

| T_src | best comm_ratio | wall | aggregate TFLOP/s |
|---|---|---|---|
| 512 | 0.250 | **4.795 ms** | **601.9** |
| 2048 | 0.180 | **16.98 ms** | **680.0** |
| 8192 | 0.180 | **64.47 ms** | **716.3** |

- 4× tokens, 3.54× wall —— super-kernel 在大 M 下 **slightly super-linear**（每 4× tokens 吞吐还涨 ~5-13%）
- 最优 comm_ratio 从 0.25 → 0.18：M 大后 compute 占比涨，dispatch CU 需求降
- **这跟我之前推断的 sub-linear scale 是反的**——kernel 自己其实能在大 M 下发挥得更好

### 2) PyTorch+RCCL baseline（per-expert `torch.matmul` + RCCL all-to-all）

| T_src | dispatch | per-expert GEMM | combine | total | per-GPU TFLOP/s |
|---|---|---|---|---|---|
| 512 | 0.72 | 4.96 | 1.25 | **7.00 ms** | 51.5 |
| 2048 | 1.30 | 6.42 | 1.21 | **9.05 ms** | 159.5 |
| 8192 | 3.78 | 11.07 | 3.61 | **18.64 ms** | 309.7 |

- PyTorch 这条路径用的就是 hipBLASLt grouped GEMM（per-expert `@`），它在大 M 下越跑越快
- 4× tokens 只用 ~1.3-2× wall——**PyTorch+RCCL 在大 M 下 strongly super-linear**

### 3) **Head-to-head：哪个赢？**

| T_src | super-kernel | PyTorch+RCCL | super-kernel speedup |
|---|---|---|---|
| 512 | 4.80 ms | 7.00 ms | **1.46× ✓** |
| 2048 | 16.98 ms | **9.05 ms** | **0.53× ❌（super-kernel 慢 1.88×）** |
| 8192 | 64.47 ms | **18.64 ms** | **0.29× ❌（super-kernel 慢 3.46×）** |

**只有在历史 bench 工况（512 t/g）super-kernel 才赢；训练实际工况（2048 / 8192）全面落后**。

### 4) 训练里的 super-kernel 比单跑 bench 还慢 56%

| 工况 | wall | 备注 |
|---|---|---|
| Standalone bench @ T_src=2048 | 16.98 ms | controlled env |
| 训练里实测 @ T_src=2048（fwd-prof） | **26.51 ms** | DSV3 4-layer Primus 训练 |
| Δ | **+9.53 ms (+56%)** | 训练环境开销 |

训练环境额外慢的来源（待 isolate）：
- HBM 上同时驻留 optimizer state / activations / non-MoE 层 weight，cache 不友好
- 多个 CUDA stream 抢资源（Megatron 的 grad-reduce / param-gather overlap stream）
- Workspace 跟 model state 在同一 allocator pool，page table 抖动
- 上游 attention/dense 收尾不彻底，super_kernel start 时 wait HBM idle

## 教训 / 修正

1. **README headline 必须改**：「**4.82 ms / 1.76× speedup vs PyTorch+RCCL**」只在 `T_src=512` 成立，**训练工况 (2048+) super-kernel 比 PyTorch+RCCL 慢 1.88-3.46×**
2. **bench 必须按训练工况重新设计**：把 `--tokens 2048 / 8192 --topk 8` 当成 first-class bench，sweep comm_ratio + K_TILE + MFMA tile 全跑一遍
3. **之前所有 P0/P1/P2 优化都是在 T_src=512 上调的**——512 t/g 的最优解（tile=128 / K_TILE=128 / comm_ratio=0.25 / 16×16 MFMA）在 8192 t/g **几乎肯定不是最优**，需要重做整个 tile + ratio sweep
4. **训练环境额外 +56% 开销也是 first-class 问题**：即便把单跑 bench 调到 PyTorch 水平，training 里 super-kernel 还会再损失一半

## 下一步（按 ROI 重排）

| 优先级 | 项 | 预期 |
|---|---|---|
| **P0** | 把 README headline 改成 `4.82 ms @ T_src=512`，并明确标注训练工况下的 actual 数字 | 必做，统一口径 |
| **P0** | 在 T_src=2048 / 8192 重做 tile + comm_ratio sweep；尝试 32×32 MFMA + K_TILE=256；目标至少追平 PyTorch+RCCL | 16.98 → 9 ms / 64.5 → 19 ms |
| **P1** | 隔离训练 +56% overhead：用 `torch.profiler` 抓 super_kernel call 内部，看是 launch wait 还是 HBM 抢占 | -9 ms / call |
| P2 | 重新评估：如果 PyTorch+RCCL 在生产档已经 18.64 ms，super-kernel 的设计前提（fuse 三 phase 进一个 kernel 才能 hide A2A）在这个 token 量级是不是还成立？大 M 下 PyTorch 已经超过 RCCL 通信链路的 effective overlap 极限 | 决策点 |
| P2 | 跟 decomposed bwd 性能优化（todo `decomposed_bwd_perf_opt`）合并考虑——bwd 慢的可能也是同样的 tile mismatch | 联合优化 |

## 数据原文

```
super-kernel (best comm_ratio per row):
  T_src=512  ratio=0.250  latency=4.795 ms  tflops_agg=601.9
  T_src=2048 ratio=0.180  latency=16.98 ms  tflops_agg=680.0
  T_src=8192 ratio=0.180  latency=64.47 ms  tflops_agg=716.3

PyTorch+RCCL (per-call median wall across 20 iters, all-rank max):
  T_src=512  wall=7.00 ms  dispatch=0.72  gemm=4.96  combine=1.25
  T_src=2048 wall=9.05 ms  dispatch=1.30  gemm=6.42  combine=1.21
  T_src=8192 wall=18.64 ms dispatch=3.78  gemm=11.07 combine=3.61
```

bench 脚本：
- `benchmarks/bench_apples_to_apples_sk.sh`
- `benchmarks/bench_pytorch_rccl_dsv3.py`
