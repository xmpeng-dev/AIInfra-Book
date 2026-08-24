# Mega MoE MXFP8 — CU 配比重扫 + wgrad variable-K GEMM campaign（T=8192）@ MI355X

> **用途**: 通信机制换成 epoch 自复位后（见 `2026-07-23_1452_mega_fp8_epoch_comm_stage_perf_t8192_mi355x.md`），
> 重扫各 stage 的 **CU 角色配比**（combine/reduce、dispatch/preshuffle）找新最优；外加 fp8 **变长-K wgrad GEMM** 的一轮
> kernel-optimize campaign（瓶颈分析 + autotune 扩候选）。
> **Where**: `smci355-ccs-aus-n01-21`（MI355X / gfx950 ×8），容器 `xiaoming-dev`，job 22522。
> **配置**: DeepSeek-V3，EP8，**T=8192**，H=7168，I=2048，E=256，topk=8，BM=BN=256，routing=load_balanced。
> **精度**: SNR(dB)=`10·log10(‖ref‖²/‖ref−out‖²)`，跨 8 rank 取最慢。

---

## Part A — CU 角色配比重扫（epoch 通信下最优点变了）

CU-split 默认值当初是在旧的 host-barrier 通信下调的；换成 epoch 自复位后重扫。旋钮:
combine kernel = `num_combine_cu / num_reduce_cu`（grid = combine + reduce + GEMM tiles，reduce_cu=0 → tail-reduce）;
dispatch kernel = `num_dispatch_cu / num_preshuffle_cu`。

### 逐 stage 扫描结果（isolated stage bench, T=8192）

| stage | 旧默认 | 扫描结果（best 粗体） | 新最优 | isolated Δ |
|---|---|---|---|---|
| **L2 fwd combine** (combine/reduce) | 48/0 → 956 TF | 16→710, 32→**1000**, 48→956, 64→934, 80→882; reduce_cu>0 一律更差 | **32/0** | +4.6% |
| **STEP3 bwd combine** | 16/0 → 1302 TF | 8→789, 16→1302, 24→**1370**, 32→1357, 48→1298; 16:64→1053, 24:128 崩 | **24/0** | +5.2% |
| **L1 fwd dispatch** (dispatch/preshuffle) | 16/16 → 1096 TF | 16/16→1096, **24/8→1530**, 32/8→1401, 12/20→1404, 20/8→1279, 28/8→1159 | 24/8（isolated）| +40% **(假象, 未采纳)** |
| **bwd L2-dgrad dispatch** | 24/8 | 24/8→**1222**, 24/16→1221, 16/16→1149, 32/8→1117, 20/8→917, 28/8→846 | 24/8（不变）| — |

规律: 两个 combine stage 都收敛到 ~24–32，且 **tail-reduce（reduce_cu=0）完胜 dedicated-reduce**（reduce_cu>0 全更差，128 还崩）。

### ⚠️ 关键陷阱: isolated stage bench 不一定忠实 → 必须用 e2e 复核（Rule 11）

L1 dispatch 的 “24/8 +40%” **是 bench 假象,没采纳**:
- isolated `l1` 的 `_fp8()` **每 iter 重跑 `dispatch_prologue`**,量的是 prologue↔dispatch 的 back-to-back **重叠竞争**,不是每-forward 的真实 dispatch 成本。
- 用 op 路径（`--stage fwd`）复核: 16/16 → 4.85 ms、24/8 → 4.92 ms —— **前向对该 split 不敏感**,+40% 不 transfer。
- 反例: L2 / STEP3 的 isolated bench **只计 combine kernel**（与生产同一调用）,是忠实的,收益如实 transfer。

**结论**: 只落地 **L2 combine 48→32、STEP3 combine 16→24**;L1 保持 16/16,L2-dgrad 保持 24/8。

### e2e 验证（fwd+bwd 完整 autograd，T=8192, EP8）

| 配置 | fp8 fwd+bwd | vs bf16 | 梯度 SNR | 状态 |
|---|---|---|---|---|
| 重扫前 | 14.94 ms | 1.34× | dx/dtw/dW1/dW2 ≥15 dB | PASS |
| **重扫后 (L2=32, STEP3=24)** | **14.71 ms** | **1.36×** | dx20.1/dtw22.5/dW1 19.4/dW2 19.7 | PASS |

净 ~0.23 ms（~1.5%，落在 e2e 噪声边缘，但方向一致且 SNR 不变；isolated combine +4.6%/+5.2% 是忠实来源）。

### 留下的基础设施
- `bench_mega_moe_fp8.py`: `--dispatch-cu / --preshuffle-cu / --combine-cu / --reduce-cu`（-1=stage 默认）。
- `_dispatch_l2_dgrad_mxfp8_flydsl_kernel`: 可选 `num_dispatch_cu / num_preshuffle_cu` kwargs。
- sweep 脚本: `agent/workspace/gemm_mxfp8_vark_wgrad_flydsl_gfx950_20260724/sweep_cu.sh`
  （`STAGE=.. KIND=combine|dispatch CONFIGS="a:b .."`，每档换 `MASTER_PORT` + 清残留 + `trap` 中断自清）。

---

## Part B — wgrad variable-K GEMM kernel-optimize campaign

**动机**: 真实 DSv3（E=256 → 每 rank 32 个瘦组，K_g~2k）下 mxfp8 变长-K wgrad GEMM 只有 ~1900–2000 TF，
而玩具 E=32（4 个肥组，K_g~16.5k）能到 ~2300–2371 TF。**同一份 kernel 代码**（已验证 MegaMoE @ E=32 == 上游）。
差距是“组多、每组 K 小”regime 的固有代价:8× 的 `[OUT_M,OUT_N]` 输出 tile + 8× 更低的 per-组 K 摊薄。

kernel: `primus_turbo/flydsl/grouped_gemm/mxfp8_grouped_kernel.py::_build_grouped_mxfp8_wgrad_kernel`
（config `(bm,bn,gm,xcd,gn)`，默认 `(256,256,4,8,0)`）。LOCAL kernel（无跨 rank 通信）→ 单卡可复现/profile。

### Baseline + profile（单卡，standalone `quick_test_bench.py`）
- 基线 geomean **2027 TF**（dw2_uni 1976 / dw1_uni 2059 / dw2_skew 2002 / dw1_skew 2072），SNR 22.6 PASS，skew≈uniform（已 skew-robust）。
- 分解: wgrad GEMM **948us (98%)** + preshuffle 21us (2%)。
- rocprofv3（dw2, autotune off）: **MfmaUtil 47%**、**MeanOccupancyPerCU 7.15（~25%）**、VGPR 128、**LDS 128KB/block → 每 CU 仅 1 block**、HBM 解析 ≈1.6 TB/s（**~20%** of 8 TB/s）。
- **瓶颈: stall/占用受限** —— 128KB LDS（4-buffer K-split 双缓冲）把驻留卡在 1 block/CU → 占用 ~25% → MFMA 空转 53%。

### 迭代（linear，campaign 在 `agent/workspace/gemm_mxfp8_vark_wgrad_flydsl_gfx950_20260724/`）
| round | 改动 | geomean | 决策 |
|---|---|---|---|
| 1 | baseline | 2027 TF | — |
| 2 | autotune 候选 4→9（更多 gm/xcd + bm=128） | **2052 TF (+1.2%)** | ✅ ACCEPT（更好 L2 swizzle；bm=128 未被采纳）|
| 3 | pipeline chunk 8→16 | 1994 (−2.8%) | ❌ ROLLBACK（skew 大组深展开→VGPR 压力→占用更低）|
| 4 | pipeline chunk 8→4 | 2020 (−1.6%) | ❌ ROLLBACK（更多 chunk 边界开销；占用受 LDS 非 VGPR 限）|

- chunk=8 已最优；反证 **占用限制来自 LDS 不是寄存器** → 任何调度旋钮（chunk/waves_per_eu）都抬不动占用。
- 配置 + 流水深度空间**已榨干**。唯一剩余杠杆: 把 4-buffer 双缓冲 LDS 砍到 ≤64KB → 每 CU 2 block（占用翻倍）—— 需重写手工流水的 `_wgrad_ssa_chunk`/`_wgrad_mx_body_4buf`，**高风险 + 收益不确定**（MFMA GEMM 上双缓冲预取常比占用更值钱，单缓冲可能反而更慢），**暂缓**。

### 工具备忘
- rocprof-compute 在容器里缺 python 依赖（plotly/dash/textual），不可用 → 用 `rocprofv3 --pmc`。
- rocprofv3 派生指标若展开到多 TCC channel（FetchSize/WriteSize）→ 多 pass/超时 → HBM 用解析估算；MfmaUtil+占用 单 pass 即可（FLYDSL 盘缓存热后 ~5s）。
- profile 时 `PT_MXGG_AUTOTUNE=0` 钉住单 config，否则首调 autotune 扫描污染 trace。

---

## Commits（feat/xiaompen/mega_moe_flydsl_mxfp8）
- `aecdb724` perf(moe): widen mxfp8 variable-K wgrad autotune candidates (+1.2% at DSv3 G=32)
- `b41054f6` perf(moe): retune fp8 combine CU split for the epoch self-reset comm（L2 48→32, STEP3 16→24, +CU-sweep infra）

## 复现
```bash
cd /perf_apps/xiaoming/MegaMoE; export PYTHONPATH=$PWD
# CU 扫描（分布式，8 卡）:
STAGE=l2 KIND=combine CONFIGS="48:0 32:0 64:0" T=8192 \
  bash agent/workspace/gemm_mxfp8_vark_wgrad_flydsl_gfx950_20260724/sweep_cu.sh
# wgrad GEMM 单卡 baseline/bench:
HIP_VISIBLE_DEVICES=0 python agent/workspace/gemm_mxfp8_vark_wgrad_flydsl_gfx950_20260724/quick_test_bench.py
# e2e 复核:
MASTER_PORT=$((20000+RANDOM%20000)) python benchmark/ops/bench_mega_moe_fused_fp8_bwd.py --num-processes 8 --num-tokens 8192
```
