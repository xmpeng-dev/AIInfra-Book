# Mega MoE (FlyDSL) 两融合算子 EP8 — 从源码 build + run @ MI355X（n01-25）

> **When**: 2026-07-17 UTC+8
> **Where**: `smci355-ccs-aus-n01-25`（MI355X / gfx950 ×8），容器 `xiaoming-dev`（`rocm/primus:v26.3`）
> **What**: 在本机**从源码编译**仓库（含 CK/HIP `csrc`）后跑 `benchmark/ops/training/bench_mega_moe.py` 两模式，对 `primus_turbo/flydsl/mega` 两融合算子做 EP8 正确性 + 性能基准
> **Repo**: `/perf_apps/xiaoming/MegaMoE` @ `573bb39`（feat(moe): fused Mega Kernel MoE on FlyDSL, #412）
> **软件**: torch `2.10.0+git94c6e04`，HIP `7.2.53211`（ROCm 7.2.1），**flydsl `0.2.4`**（从镜像自带的 `0.1.1.dev409` egg 升级），triton `3.6.0`
>
> 昨天（2026-07-16, n03-33，editable + 复用预编译 `.so`）的基准见 `2026-07-16_1104_BASELINE_...md`；两者结论一致。

## TL;DR

- **从源码 build 成功**：`git submodule update` 拉 CK/hipify_torch → 编 `csrc`（CK/HIP + rocSHMEM）生成新的 `_C*.so` + `libprimus_turbo_kernels.so`，`BUILD exit=0`。
- **正确性全绿**：5/5 阶段 `cos≈1.00000 / PASS`（vs Primus-Turbo DeepEP 参考）。
- **GEMM 到位**：bf16 dense/grouped GEMM ~**980–1303 TFLOPS**（≈ MI355X bf16 稠密峰值 2.5 PFLOPS 的 ~40–52%）；融合前向 ~0.95–1.10 PFLOPS。
- **重叠有效**：overlap roofline **86.5%–88.2%**，`grouped/dense` 90–96%（wgrad tn 达 129%，见下注），speedup vs serial **1.37×–1.75×**。
- **通信在位**：XGMI dispatch ~391–402 GB/s、combine ~381–385 GB/s。

## 从源码 build 步骤（容器内）

本机镜像**自带 flydsl 是 `0.1.1.dev409`（egg）**，缺 `#412` 需要的 `flydsl._mlir.dialects.fly_rocdl.TargetAddressSpace`，故须先升级 flydsl。

```bash
# 1) git 归属 + 子模块
git config --global --add safe.directory /perf_apps/xiaoming/MegaMoE
git submodule update --init --recursive        # composable_kernel + hipify_torch

# 2) 从源码编译安装（编 CK/HIP csrc，含 rocSHMEM）
MAX_JOBS=128 PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 \
  pip install -e . --no-build-isolation --no-deps      # 生成新的 _C*.so + libprimus_turbo_kernels.so

# 3) 升级 flydsl 到仓库要求的版本（卸掉旧 egg 再装）
pip uninstall -y flydsl && pip install flydsl==0.2.4

# 4) 清缓存强制在 0.2.4 下从零重调 + 重编译
rm -rf ~/.flydsl
```

依赖告警（均无害）：`primus-turbo requires triton==3.7.0`（镜像是 3.6.0，实测可跑）、`amd-aiter requires flydsl==0.1.1.dev409`（我们不用 aiter 路径）。

复现命令：

```bash
cd /perf_apps/xiaoming/MegaMoE/benchmark/ops/training
PYTORCH_ROCM_ARCH=gfx950 python bench_mega_moe.py --mode dispatch_grouped_gemm --models DeepSeek-V3 --num-processes 8
PYTORCH_ROCM_ARCH=gfx950 python bench_mega_moe.py --mode grouped_gemm_combine  --models DeepSeek-V3 --num-processes 8
```

配置：DeepSeek-V3，EP8，T=8192，H=7168，I=2048，E=256，topk=8，bf16，iters=30。

## 结果

指标释义：`dense` = 同 M×N×K 单权重稠密 GEMM 参考（固定 tile，GROUP_M=4）；`gemm_only` = grouped GEMM（无通信）；`grouped/dense` = 分组相对稠密效率；`comm` = 纯 dispatch/combine（XGMI）；`fused` = 融合算子；`roofline` = `max(gemm,comm)/fused`；`speedup` = `(gemm+comm)/fused`。取跨 8 rank 最慢值。

### dispatch_grouped_gemm

| stage | dense (ms / TFLOPS) | gemm_only (ms / TFLOPS) | grouped/dense | comm (ms / GB/s) | fused (ms / TFLOPS) | roofline | speedup | acc |
|---|---|---|---|---|---|---|---|---|
| forward (nt)        | 3.646 / 1125.4 | 3.805 / 1078.7 | 95.8% | 2.103 / 391.3 | 4.313 / 951.4 | 88.2% | 1.37× | 1.00000 PASS |
| backward dgrad (nn) | 1.873 / 1095.8 | 2.076 / 988.3 | 90.2% | 2.045 / 402.4 | 2.358 / 870.1 | 88.0% | 1.75× | 1.00000 PASS |
| backward wgrad dW1 (tn) | 4.199 / 977.4 | 3.247 / 1263.9 | 129.3% | 2.053 / 400.7 | 3.746 / 1095.4 | 86.7% | 1.41× | 1.00000 PASS |

### grouped_gemm_combine

| stage | dense (ms / TFLOPS) | gemm_only (ms / TFLOPS) | grouped/dense | comm (ms / GB/s) | fused (ms / TFLOPS) | roofline | speedup | acc |
|---|---|---|---|---|---|---|---|---|
| forward (nt)        | 1.621 / 1265.9 | 1.750 / 1172.6 | 92.6% | 2.159 / 381.1 | 2.447 / 838.6 | 88.2% | 1.60× | 0.99997 PASS |
| backward dgrad (nn) | 3.148 / 1303.5 | 3.452 / 1189.0 | 91.2% | 2.136 / 385.1 | 3.989 / 1028.9 | 86.5% | 1.40× | 1.00000 PASS |

> **注 (wgrad tn `grouped/dense`=129.3%)**：variable-K tn 阶段里 grouped GEMM 比“dense 参考”还快，是因为 dense 参考是**固定 tile**（256×256, GROUP_M=4），对该形状并非最优；grouped 侧 autotune 选到了更好的 tile。故 >100% 属正常，不是异常。

## 结论 / 后续

1. **从源码 build 的 #412 在 MI355X/gfx950 功能正确、重叠有效**：5/5 cos 全绿，overlap 86–88%，融合前向 ~0.95–1.10 PFLOPS。与昨天 n03-33（复用预编译 `.so`）结论一致，说明结果与安装方式无关。
2. **绝对值随 fresh autotune 有 ~10% 抖动**（如 dispatch fwd dense 1125 vs 昨天 1263），因为每次清缓存重调选到的 tile 不完全相同；量级一致。
3. **GEMM ~40–52% 峰值，仍有空间**：下一步可 rocprofv3 profile grouped GEMM（occupancy / MFMA 发射 / 访存瓶颈）；dispatch wgrad 的固定 dense 参考 tile 偏弱（被 grouped 反超），dense roofline 本身也可加 autotune 使参考更贴近峰值。

## 附录：原始产物

- 日志：`logs/n0125_src_dispatch_grouped_gemm.log`、`logs/n0125_src_grouped_gemm_combine.log`、`logs/build.log`
- CSV：`logs/dispatch_grouped_gemm_20260717_n0125src_MI355X.csv`、`logs/grouped_gemm_combine_20260717_n0125src_MI355X.csv`
- 脚本：`_repro/{build_discover.sh,build.sh,fix_flydsl.sh,run_both_n0125.sh}`
- 收尾日志里的 `TCPStore recvValue failed ... remote server shutdown` 是 rank0 先退、其余 rank 拆 PG 的正常 teardown 噪声，非崩溃（结果已 `Results saved`）。
