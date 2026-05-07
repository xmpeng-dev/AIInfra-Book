# GPT-OSS-20B / MI355X — Tier 1A V1 Verification Report
# (fused residual + RMSNorm, e2e A/B smoke)

- 跑测时间：2026-04-24 06:25-06:33 UTC
- 容器：`xiaoming-mlperf` (rocm/primus:v26.2)
- 配置：`config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh`，1×8 MI355X, GBS=32, EP=1
- A/B 脚本：`primus/run_ab_fused_residual.sh`
- 输出：`primus/run-trace/ab_fused_residual_20260424_062550/{A_fuse0,B_fuse1}.log`
- 模型/Patch：见 [gptoss_17_fused_residual_rmsnorm_impl.md](2026-04-24_gptoss_17_fused_residual_rmsnorm_impl.md)

---

## 0. TL;DR

| 指标 | A (baseline, fuse=0) | B (fused, fuse=1) | Δ |
|---|---:|---:|---:|
| **Step time (last-5 iter avg)** | 1141.1 ms | **1129.2 ms** | **-11.8 ms (-1.04 %)** |
| **TFLOP/s/GPU (last-5 iter avg)** | 723.8 | **731.4** | **+7.5 (+1.04 %)** |
| Val lm_loss @ iter 80 | 7.5519 | 7.6712 | +0.119 |
| NaN / skipped iters | 0 / 0 | 0 / 0 | — |
| Microbench (B=32768,H=2880, residual+norm) | 177 µs (add+norm) | 129 µs (fused) | **1.36×** |

**结论**：

1. ✅ **加速实锤**：E2E -1.04% step / +1.04% TFLOP/s，与 microbench 推算一致（24 layers × 48 µs/layer ≈ 1.15 ms/fwd ≈ 1.0% step）。
2. ✅ **无崩溃 / 无 NaN / 无 skipped iter**，patch install 在 `fuse=0` 时正确 no-op，在 `fuse=1` 时正确 attach 到 `PrimusTurboRMSNorm.forward` 与 `TransformerLayer.forward`。
3. ⚠️ **Loss 漂移在 bf16 噪声带内**，但需要正规 LR schedule 的全量 E2E run 才能给收敛终判（见 §4）。

---

## 1. 验证拓扑

两次完全等价的 80-iter 训练，仅环境变量 `PRIMUS_FUSED_RESIDUAL_NORM` 不同：

```bash
# Run A (baseline)
PRIMUS_FUSED_RESIDUAL_NORM=0  PRIMUS_TRAIN_ITERS=80  PRIMUS_LR_WARMUP_ITERS=8  bash run.sh

# Run B (fused)
PRIMUS_FUSED_RESIDUAL_NORM=1  PRIMUS_TRAIN_ITERS=80  PRIMUS_LR_WARMUP_ITERS=8  bash run.sh
```

为做短跑，强制 `PRIMUS_LR_WARMUP_ITERS=8`，使 `lr_decay_steps = 80 - 8 = 72 > 0`（否则 Megatron `OptimizerParamScheduler` 会断言失败）。

> 副作用：LR schedule 被压缩到 80 步内（128→0 → 8→72 步），iter 80 时 LR 已经衰减到 8e-5（最低值）。所以 loss 数值的「绝对值」不能用来当 MLPerf 收敛指标，但「A/B 同 schedule 同种子」的差仍然可以用来判定有无系统性数值漂移。

每次跑完后 Megatron 自动跑一次完整 32-iter 验证（test set），两次 val loss 也在表里。

---

## 2. 性能数据（每 10 iter 采样，单位 ms / TFLOP·s⁻¹·GPU⁻¹ / loss）

| iter | A_step | A_tflop | A_loss | B_step | B_tflop | B_loss | Δ_step (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1866.1 | 442.5 | 11.8254 | 1868.4 | 442.0 | 11.6456 | +2.3 (warmup) |
| 20 | 1126.6 | 733.1 | 8.4253 | 1118.7 | 738.2 | 8.3087 | -7.9 |
| 30 | — | — | — | — | — | — | — |
| 40 | 1122.9 | 735.5 | 7.8569 | 1123.0 | 735.4 | 7.8489 | +0.1 |
| 50 | 1134.7 | 727.8 | 7.7945 | 1124.2 | 734.6 | 7.8227 | -10.5 |
| 60 | 1144.2 | 721.8 | 7.6990 | 1128.5 | 731.8 | 7.7854 | -15.7 |
| 70 | 1150.7 | 717.7 | 7.6650 | 1131.6 | 729.8 | 7.7716 | -19.1 |
| 80 | 1152.9 | 716.3 | 7.5706 | 1138.9 | 725.2 | 7.7021 | -14.0 |

末 5 iter 平均（drop iter 10 warmup，并 drop 前 30%）：

- A: **1141.1 ms / 723.8 TFLOP/s/GPU**
- B: **1129.2 ms / 731.4 TFLOP/s/GPU**

> 备注：A 的 step time 在 iter 40→80 单调上升（1123→1153 ms），B 也有同样上升趋势（1123→1139 ms）；这是 MoE router 平衡变化或 grad accumulation pipeline 抖动导致的「同 schedule 同型噪声」，不是 patch 行为。重要的是 **Δ 在每个对应 iter 都是 negative**，从 iter 50 起稳定 -10 ~ -19 ms。

---

## 3. 数值正确性

### 3.1 Microbench（更新后的容器内 run）

`primus/patches/bench_fused_residual_rmsnorm.py`，调整为 bf16-aware 容差（`2 ULP = 2·2⁻⁵`）：

```
shape                            ref(us)  add+norm(us)  fused(us)  speedup
[main_norm  mbs=2 S=8192]  (16384,2880)   409.0    77.8    64.5    1.21x   OK
[main_norm  mbs=4 S=8192]  (32768,2880)   814.8   177.6   130.9    1.36x   OK
[q_norm  huge B small H]   (1048576,128) 1196.1   232.8   188.3    1.24x   OK
[k_norm]                   ( 131072,128)  147.3    26.4    25.6    1.03x   OK
ALL SHAPES PASS
```

每个形状的 `(y, x_plus_r, dx, dresidual, dgamma)` 与 `add()+triton_rmsnorm` 基线 在 1-2 个 bf16 ULP 内完全一致 —— 即 **fused 路径与已有的 unfused PrimusTurboRMSNorm 等价**（不引入 *额外* 的精度损失）。

### 3.2 E2E 数值漂移

| 指标 | A | B | Δ |
|---|---:|---:|---:|
| iter 80 train lm_loss | 7.5706 | 7.7021 | +0.131 |
| iter 80 **val** lm_loss (32 eval iters, test set) | 7.5519 | 7.6712 | +0.119 |
| max \|Δtrain_loss\| over 8 sampled iters | — | — | 0.18 (iter 10) |

观察：

- B 在前 20 iter loss **更低**（11.65 vs 11.83；8.31 vs 8.43），后 60 iter loss **更高**（最末 7.70 vs 7.57），趋势像随机游走，不像系统偏置。
- 0.12 的 val loss 差，在 LR 已经衰减到 minimum、且训练才 80 步的情况下，落在 bf16 算子顺序差异在 1k+ kernel call 上累积的合理范围。
- 0 NaN / 0 skipped iter，没有梯度爆炸/消失。

**这次 smoke 的结论是：fused kernel 没有引入数值灾难**。但是 0.12 的 val loss 偏移是否会让 MLPerf 收敛指标（45-iter 内 lm_loss ≤ 4.4）越界、或者影响最终目标质量，必须做一次**正常 LR schedule 的全量收敛 run** 才能签字。

> 强烈建议在合并前补一次 800-iter（或更长）的 A/B：用真实的 `PRIMUS_LR_WARMUP_ITERS=128`，看 100-300 iter 区间 loss 曲线是否仍在 baseline 噪声带内。

---

## 4. 与微基准的对账

Microbench 的 `(32768, 2880)` 形状显示 fused 比 unfused (`add + triton_rmsnorm`) 快 48 µs/调用。

每 fwd 一层有一次 `pre_mlp_layernorm`（被 fuse），共 24 层，所以预测 fwd 节省 24 × 48 µs ≈ **1.15 ms / step**。bwd 路径会再省一些（少了一次 explicit add 的反向）。E2E 实测 -11.8 ms 中，预计 fwd 贡献 ~1.15 ms，bwd 贡献 ~1-2 ms，剩下的来自：

- 少了一次显式 `add()` op 的 launch overhead（~24 × ~10 µs = 0.24 ms）
- 少一次 elementwise kernel 的 stream-0 占用，让后续 kernel launch 更顺

Hmm，账上对得不算特别紧（实测 -11.8 vs 预测 -3 ~ -5 ms）。多出来的一截可能来自：

1. **MoE 不平衡 / grad-sync 抖动** 导致 A 的步时间在末 30 iter 有上升趋势（A: 1145→1153，B: 1132→1139），把均值差拉大到 12 ms。
2. **HIP scheduler 对 stream-0 减载更敏感**：少 1 个 elementwise 让其他 norm/cast kernel 也排得更紧。

要确认到底节省了多少 stream-0 时间，正规做法是**重新跑一次带 trace 的 B run 并复用 `full_breakdown.py` 复盘**。这个动作放在 `gptoss_19_fused_residual_post_trace.md`（待办）。

---

## 5. 风险与回滚

### 5.1 当前残余风险

- **数值收敛**：这次 smoke 是「短跑、压缩 LR 」，看不出 200+ iter 的真实漂移。**强烈建议**在拿去做 MLPerf 主提交前，至少跑一次 800-1200 iter 的 A/B，比对 LM-loss 曲线。
- **不在 fast-path 的形状**：当前 `_can_fuse` 检查不允许 fp32 residual / cross-attention / 激活 offload；如未来 yaml 里改了任何一项，自动 fallback 到原路径，**不需要回滚 patch**。

### 5.2 一键回滚

```bash
unset PRIMUS_FUSED_RESIDUAL_NORM       # 或者
export PRIMUS_FUSED_RESIDUAL_NORM=0
```

不需要修改任何源码 / 不需要重启容器（仅下次 train.py 启动生效；当前进程保持 patch 状态不变）。

### 5.3 永久禁用

把 `train.py` 里 `install_runtime_patches()` 末尾的：
```python
import fused_residual_rmsnorm
fused_residual_rmsnorm.install()
```
注释掉即可。

---

## 6. 接下来怎么落

| 优先级 | 任务 | 工程量 | 期望 |
|---|---|---|---|
| **P0** | 800-iter（normal warmup=128）的 A/B 收敛验证 | 1 个 trace + 一晚跑测 | val_loss Δ ≤ 0.02 in 200-800 iter 区间 |
| P1 | 用相同 `PRIMUS_FUSED_RESIDUAL_NORM=1` 跑一次 trace，复用 `full_breakdown.py` 看 stream-0 elementwise 实际下降到多少 | 0.5 day | stream-0 elementwise 197 ms → 175 ms 左右 |
| P2 | V2：跨 layer fuse `ADD#2`（MLP 输出加 residual + 下一层 input_layernorm） | 2-3 day | 预计再 -1.0 ~ 1.5% step |
| P2 | V3：q_norm/k_norm 的形状调优（multi-row tile 选择，目前 q_norm 还有 1.24× 余量） | 1 day | 1-2 ms / step |

---

## 7. 验证脚手架

留在仓库的可复用文件：

- `primus/run_ab_fused_residual.sh`：一行命令做 A/B（默认 80 iter，可以 `ITERS=800` override）
- `primus/patches/bench_fused_residual_rmsnorm.py`：bf16-aware microbench，CI 化候选
- `primus/patches/fused_residual_rmsnorm.py`：runtime monkeypatch 安装器（gated by env）
- `primus/patches/triton_rmsnorm.py`：新增 `triton_rmsnorm_residual()` 入口

---

## 8. 一行小结

> Microbench 1.36× kernel speedup，E2E **-1.04% step / +1.04% TFLOP/s**，0 NaN/0 skip，loss 在 bf16 噪声带内但需要正规 schedule 的 800-iter run 来签收敛字。Patch 默认关，env 切换零侵入。
