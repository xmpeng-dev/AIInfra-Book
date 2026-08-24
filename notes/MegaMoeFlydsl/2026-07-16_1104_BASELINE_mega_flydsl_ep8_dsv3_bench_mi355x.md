# Mega MoE (FlyDSL) 两融合算子 EP8 基准 — DeepSeek-V3 @ MI355X

> **When**: 2026-07-16 UTC+8
> **Where**: `smci355-ccs-aus-n03-33`（MI355X / gfx950 ×8），容器 `xiaoming-dev`（`rocm/primus:v26.3`）
> **What**: 安装仓库后跑 `benchmark/ops/training/bench_mega_moe.py` 两模式（`dispatch_grouped_gemm` / `grouped_gemm_combine`），对 `primus_turbo/flydsl/mega` 两个通信-计算融合算子做 EP8 正确性 + 性能基准
> **Repo**: `/perf_apps/xiaoming/MegaMoE` @ `573bb39`（feat(moe): fused Mega Kernel MoE on FlyDSL, #412），editable 安装，复用与镜像同 torch 的预编译 `.so`（未现编 csrc）
> **软件**: torch `2.10.0+git94c6e04`，HIP `7.2.53211`（ROCm 7.2.1），**flydsl `0.2.4`**，triton `3.6.0`
>
> **今天（2026-07-17）的从源码 build + run 结果见 `2026-07-17_1411_..._n0125_srcbuild_mi355x.md`。**

## TL;DR

- **正确性全绿**：5 个阶段（dispatch fwd/bwd-dgrad/bwd-wgrad + combine fwd/bwd-dgrad）全部 `cos≈1.00000 / PASS`（vs Primus-Turbo DeepEP 参考）。
- **GEMM 达峰**：bf16 dense/grouped GEMM ~**1230–1283 TFLOPS**，约为 MI355X bf16 稠密峰值（2.5 PFLOPS）的 **~50%**，是正常的实用 bf16 GEMM 水平。
- **重叠设计有效**：overlap roofline `max(gemm,comm)/fused` = **87.6%–92.1%**；`grouped/dense` 效率 91.9%–100%；speedup vs serial **1.46×–1.61×**（GEMM 快 → 藏通信收益大）。
- **通信在位**：XGMI dispatch ~368–381 GB/s、combine ~357 GB/s。

## 环境 / 复现

`/perf_apps/xiaoming/MegaMoE` 是共享盘上的 `#412` checkout（含 mega），以同路径挂进容器。安装方式（本环境无法现编 CK，故复用预编译产物）：

1. 镜像 torch 构建串与仓库预编译 `.so` 一致（`2.10.0+git94c6e04`），直接复用仓库里已就位的 `pytorch/_C*.so` + `lib/libprimus_turbo_kernels.so`（`import primus_turbo.pytorch` ABI OK）。
2. 卸载镜像自带 primus_turbo，`PRIMUS_TURBO_SKIP_EXT_BUILD=1 pip install -e . --no-build-isolation --no-deps` 装成 editable（跳过扩展编译；`--no-deps` 保护 flydsl 0.2.4 / triton 3.6.0）。Python 从仓库解析，无需 PYTHONPATH。
3. autotuner shim（`tune_utils.py`）：flydsl 0.2.4 有 `_run_config`，走原生路径（本 shim 只在缓存命中重放时生效，不影响调优质量）。跑前清 `~/.flydsl` 强制在 0.2.4 下从零重调 + 重编译。

复现命令（容器内）：

```bash
cd /perf_apps/xiaoming/MegaMoE/benchmark/ops/training
PYTORCH_ROCM_ARCH=gfx950 python bench_mega_moe.py --mode dispatch_grouped_gemm --models DeepSeek-V3 --num-processes 8
PYTORCH_ROCM_ARCH=gfx950 python bench_mega_moe.py --mode grouped_gemm_combine  --models DeepSeek-V3 --num-processes 8
```

配置：DeepSeek-V3，EP8，T(tokens/rank)=8192，H=7168，I=2048，E=256，topk=8，bf16。warmup=20 / iters=30。

## 结果

指标释义：`dense` = 同 M×N×K 单权重稠密 GEMM 参考（固定 tile，GROUP_M=4）；`gemm_only` = grouped GEMM（无通信）；`grouped/dense` = 分组相对稠密效率；`comm` = 纯 dispatch/combine（XGMI）；`fused` = 融合算子；`roofline` = `max(gemm,comm)/fused`（重叠效率）；`speedup` = `(gemm+comm)/fused`。取跨 8 rank 的最慢值。

### dispatch_grouped_gemm

| stage | dense (ms / TFLOPS) | gemm_only (ms / TFLOPS) | grouped/dense | comm (ms / GB/s) | fused (ms / TFLOPS) | roofline | speedup | acc |
|---|---|---|---|---|---|---|---|---|
| forward (nt)        | 3.248 / 1263.4 | 3.247 / 1264.0 | 100.0% | 2.238 / 367.6 | 3.600 / 1140.0 | 90.2% | 1.52× | 1.00000 PASS |
| backward dgrad (nn) | 1.627 / 1261.0 | 1.643 / 1249.1 | 99.1% | 2.195 / 374.8 | 2.384 / 860.7 | 92.1% | 1.61× | 1.00000 PASS |
| backward wgrad dW1 (tn) | 3.288 / 1248.3 | 3.287 / 1248.7 | 100.0% | 2.160 / 380.9 | 3.709 / 1106.5 | 88.6% | 1.47× | 1.00000 PASS |

### grouped_gemm_combine

| stage | dense (ms / TFLOPS) | gemm_only (ms / TFLOPS) | grouped/dense | comm (ms / GB/s) | fused (ms / TFLOPS) | roofline | speedup | acc |
|---|---|---|---|---|---|---|---|---|
| forward (nt)        | 1.671 / 1228.1 | 1.812 / 1132.6 | 92.2% | 2.299 / 357.8 | 2.564 / 800.4 | 89.7% | 1.60× | 0.99999 PASS |
| backward dgrad (nn) | 3.199 / 1282.8 | 3.482 / 1178.4 | 91.9% | 2.301 / 357.5 | 3.973 / 1032.8 | 87.6% | 1.46× | 1.00000 PASS |

## 结论 / 后续

1. **mega FlyDSL 两融合算子在 MI355X/gfx950 上功能正确、重叠机制有效**：5/5 阶段 cos 全绿，overlap 88–92%，grouped/dense 92–100%，融合前向 ~1.14 PFLOPS。
2. **GEMM 到 ~50% 峰值，仍有空间**：可用 rocprofv3 profile ~1.26 PF 的 grouped GEMM（occupancy / MFMA 发射 / 访存哪类瓶颈）；combine 侧 `grouped/dense` ~92% 略低于 dispatch 侧（99–100%），可查 combine 的 tile/epilogue。

## 附录：原始产物

- 日志：`logs/v263_dispatch_grouped_gemm.log`、`logs/v263_grouped_gemm_combine.log`
- CSV：`logs/dispatch_grouped_gemm_20260716_v263_MI355X.csv`、`logs/grouped_gemm_combine_20260716_v263_MI355X.csv`
- 脚本：`_repro/{discover_v263.sh,install_v263.sh,run_both_v263.sh}`
- 收尾日志里的 `TCPStore recvValue failed ... remote server shutdown` 是 rank0 先退、其余 rank 拆 PG 的正常 teardown 噪声，非崩溃（结果已 `Results saved`）。
