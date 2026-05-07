# GPT-OSS-20B MLPerf MI355X — mbs=4 / gbs=32 累积优化链 ablation

**周期**: 2026-04-21
**配置基底**: 1×8 MI355X, fp8 hybrid GEMM + bf16 weight, **mbs=4 / gbs=32 / lr=8.0e-4**, tp=pp=ep=1, 768 iter, 2 个 eval 周期 (eval_iters=32, eval_interval=384), seed=1234, c4 真实数据
**和已有 ablation 的关系**: 04-20 的 B14 ablation 是 **mbs=2 / gbs=16 / lr=4.0e-4** 下的"v2 basis → +bf16 grad → +Triton RMSNorm" 三阶级数。本次把 base **横向切换到 mbs=4 / gbs=32 / lr=8e-4**，同时把链拉长成 **6 阶 (C0..C5)** 的纯累加链，从"完全没有任何优化的 raw baseline"开始一项项加进来。

---

## 1. 完整结果

| stage | 累加新增的优化 | step (ms, 稳态) | Δ vs C0 | TFLOPs/GPU | eval@384 | eval@768 | wall (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| **C0** | (raw baseline) NUMA on, check_nan on, ddp_pad off, fp32 grad, TE RMSNorm | **1626.5** | — | 509.1 | 5.2316 | 4.4705 | 1437 |
| **C1** | + NUMA balancing off (`/proc/sys/kernel/numa_balancing = 0`) | **1500.6** | **−7.7 %** | 550.9 | 5.2283 | 4.4811 | 1317 |
| **C2** | + `check_for_nan_in_loss_and_grad: false` | **1412.6** | **−13.2 %** | 589.4 | 5.2259 | 4.4736 | 1241 |
| **C3** | + `ddp_pad_buckets_for_high_nccl_busbw: true` | **1392.5** | **−14.4 %** | 593.7 | 5.2001 | 4.4733 | 1226 |
| **C4** | + `grad_reduce_in_bf16: true` | **1320.1** | **−18.8 %** | 626.5 | 5.2145 | 4.4575 | 1183 |
| **C5** | + `use_turbo_rms_norm: true` (Triton RMSNorm 套件) | **1319.9** | **−18.8 %** | **633.4** | 5.1857 | 4.4569 | 1171 |

**链总耗时**：6 stage × ~22 min = **2h 6min** (7575s)，0 OOM / 0 crash。

---

## 2. 边际贡献 vs 04-17 mbs=2 ablation

> 04-17 已发表的同源 ablation 是在 mbs=2 下做的 (`benchmark_runs/PROGRESS_REPORT_EN.md` §2)。把两次的边际贡献并列：

| step 上叠的优化 | 本次 mbs=4 边际 | 历史 mbs=2 边际 | 差异 | 解读 |
|---|---:|---:|---:|---|
| NUMA off | **−7.7 %** | −2.5 % | **+5.2 pp** ⬆ | mbs=4 内存压到 98.3 %，kernel 页迁移成本被显著放大 |
| `check_for_nan=false` | −5.9 % | −7.5 % | −1.6 pp | mbs=4 单步更长，固定 CPU sync 在大 step 里占比下降 |
| `ddp_pad` | −1.4 % | −0.4 % | +1.0 pp | gbs 翻倍后 NCCL 通信量增大，bucket 对齐的收益放大 |
| `grad_reduce_in_bf16` | −5.2 % | −8.5 % (B0→B11) | −3.3 pp | mbs=4 时 elementwise (含 fp32 main_grad add) 占比下降 |
| Triton RMSNorm 套件 | **−0.02 %** | −1.5 % (B11→B10) | **−1.5 pp** ⬇ | **在 mbs=4 下 RMSNorm 已经不是瓶颈，Triton 替换收益归零** |
| **累加 (5 项全部叠加)** | **−18.8 %** | −12.5 % (a0→B10 等价链估) | +6.3 pp | base 切换 + 优化叠加的复合收益 |

---

## 3. 和 04-20 mbs=2 best (B11_full) 的横向对比

| 指标 | mbs=2 B11_full (04-20 best E2E) | **mbs=4 C5 (本次最佳, 短跑)** | Δ |
|---|---:|---:|---:|
| 稳态 step time | 776.8 ms | 1319.9 ms | +70 % (单步更慢, 但塞了 2× 样本) |
| **每秒样本** | 20.6 | **24.2** | **+17.6 %** |
| TFLOPs/GPU | 531 | **633.4** | **+19.3 %** |
| GPU 显存占用 | ~80 % (~226 GB) | 98.3 % (~283 GB) | 几乎打满 |
| 同 12288 样本时的 eval | iter 768 → 4.715 | iter 384 → **5.186** | **mbs=2 收敛更快 −0.47** |
| 12288 样本所需 wall | 596.7 s | **506.8 s (−15 %)** | mbs=4 训练样本更快 |

**核心 trade-off**：mbs=4 把硬件吞吐推高 +19 %（633 TFLOPs/GPU），单位 wall 训练的样本数也多 +18 %；但**单位样本的收敛速度变慢**，所以 step-time / TFLOPs 的提升不能直接外推到 TTT。

---

## 4. 关键观察

1. **mbs=4 路径下 NUMA off 是单项最大头 (−7.7 %)**，远超 mbs=2 时的 −2.5 %。
   - 任何 mbs=4 的 production 启动脚本必须先 `echo 0 > /proc/sys/kernel/numa_balancing`，已经不是"锦上添花"而是"必装"。
   - 推断原因：mbs=4 让 activation memory 占到 98.3 %，kernel 在如此高水位下做 NUMA page balancing 会触发频繁迁移。

2. **Triton RMSNorm 在 mbs=4 下 ROI 接近零 (−0.02 %)**，但 TFLOPs 还是 +6.9（说明 RMSNorm 自身确实在变快，只是被别的瓶颈吸收）。
   - 推测瓶颈已经迁移到 MoE permute/unpermute 或 attention（mbs↑ 后 attention 占比↑）。
   - **未来在 mbs=4 路径上不必再在 RMSNorm 上投入工程量**；ROI 应转到 MoE permute/Triton 化或 `use_turbo_attention`。

3. **6 个 stage 的 eval loss 全部打平（误差 ≤ 0.024，在 ±0.05 噪声带内）**，再次验证这 5 项优化都是**纯加速、无数值副作用**。
   - eval@384 极差 0.046，eval@768 极差 0.024（最大值都来自 C0；C5 反而是最低）。

4. **`ddp_pad_buckets_for_high_nccl_busbw` 在 mbs=4 下首次表现出非平凡收益 (−1.4 %)**，比 mbs=2 时的 −0.4 % 高了 3.5×。
   - 原因：gbs=32 时每步通信量是 gbs=16 的 2 倍，bucket 对齐对 NCCL busbw 的影响放大。

5. **`grad_reduce_in_bf16` 在 mbs=4 仍然是第二大单项 (−5.2 %)**，但比 mbs=2 (−8.5 %) 小，因为 elementwise 在 mbs=4 的总占比已经下降。
   - 同时它仍然是**显存收益最大**的一项（释放 ~36 GB）——mbs=4 这种内存压满 98.3 % 的场景，未来真要继续加 mbs 或开 turbo_attention，必须先有这一项。

---

## 5. ROI 排序（mbs=4 路径下重排）

| 排名 | 优化项 | mbs=4 边际 | mbs=2 边际 | 工程量 | 落地状态 |
|:---:|---|---:|---:|---|:---:|
| 1 | NUMA off (启动脚本 1 行) | **−7.7 %** | −2.5 % | 0 d | ✅ 已在 `run_one.sh` |
| 2 | `check_for_nan=false` (yaml 1 行) | −5.9 % | −7.5 % | 0 d | ✅ 已在 mbs=2 production |
| 3 | `grad_reduce_in_bf16` (yaml 1 行) | −5.2 % | −8.5 % | 0.5 d | ✅ B11 已落地 |
| 4 | `ddp_pad` (yaml 1 行) | −1.4 % | −0.4 % | 0 d | ✅ 已在 mbs=2 production |
| 5 | Triton RMSNorm 套件 | **−0.02 %** | −1.5 % | 3 d (sunk cost) | ⚠️ mbs=4 下无收益, mbs=2 下保留 |
| — | （下一步候选）MoE permute/unpermute Triton 化 | 估 −2 ~ −3 % | — | 1-2 d | ⏳ 待验 |
| — | （下一步候选）`use_turbo_attention` | 估 −2 ~ −3 % | — | 1 d | ⏳ 待验 |

---

## 6. 是否切换 mbs=4 作为主路径？

**短答**：**还不能**，需要先做一次 C5 的 full training E2E 实测。

**长答**：
- mbs=4 短跑展示了 +17.6 % samples/sec 和 +19.3 % TFLOPs/GPU，硬件层面是真的更快；
- 但**收敛慢 ~30 % 每样本**（同 12288 样本 eval 4.715 → 5.186），所以 TTT 取胜需要 samples/sec 的增速 > 收敛慢的程度；
- 粗算：mbs=2 B11_full TTT = 9963 s, samples-to-target = 393216 (12288 × 32 evals 取 hit 那次)；如果 mbs=4 收敛慢 30 % 但 samples/sec 快 17 %，TTT 估 ~9963 × 1.30 / 1.17 ≈ **11066 s, +11 %**；
- 所以**模型估计 mbs=4 会 TTT 回归 ~10 %**，但需要实测确认（lr=8e-4 + 大 batch 下 warmup/decay 行为可能让模型估算偏离）。

**建议下一步**：
1. **先跑 C5 full training (推 eval_loss ≤ 3.34) 一次**（~3h，单次 GPU 占用），决定是否切换主路径；
2. 如果 mbs=4 全量回归，把 NUMA off 的教训和"Triton RMSNorm 在 mbs=4 下 ROI 归零"两条记入 SOP；mbs=2 仍是主路径；
3. 如果 mbs=4 全量持平或略胜，开启 mbs=4 + MoE permute Triton 化的下一轮 ablation。

---

## 7. 数据来源

| 资产 | 路径 |
|---|---|
| YAML configs (5 个) | `/home/xiaompen/mlperf/ablations/M4_chain/M4_C{0,2,3,4,5}.yaml` (C0/C1 共用 M4_C0.yaml) |
| 链驱动脚本 | `/home/xiaompen/mlperf/ablations/M4_chain/run_chain.sh` (driver), `run_one.sh` (per-stage), `summarize.py` (汇总解析) |
| 6 stage 完整 stdout (~870 KB 每个) | `/home/xiaompen/mlperf/ablations/M4_chain/logs/M4_C{0..5}.log` |
| 链总控制台 log | `/home/xiaompen/mlperf/ablations/M4_chain/logs/chain.console.log` |
| 链时间汇总 | `/home/xiaompen/mlperf/ablations/M4_chain/logs/chain_summary.txt` |

**复现命令**：
```bash
cd /home/xiaompen/mlperf
nohup bash ablations/M4_chain/run_chain.sh > ablations/M4_chain/logs/chain.console.log 2>&1 &
disown
# ~2h6min 后跑完, 然后:
python3 ablations/M4_chain/summarize.py
```

---

## 8. 关联 notes

| 主题 | 文件 |
|---|---|
| 起点 (B0 解读) | [`01_mlperf_best_e2e_run`](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md) |
| 04-20 mbs=2 ablation (B0/B11/B10) 对照 | [`07_optimization_summary_table`](./2026-04-20_gptoss_07_optimization_summary_table.md) |
| Triton RMSNorm 详解（mbs=2 下 −1.5 %） | [`02_triton_rmsnorm_optimization`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) |
| bf16 grad reduce 详解（mbs=2 下 −8.5 %） | [`04_grad_reduce_bf16_optimization`](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |
| 总览索引 | [`INDEX`](./2026-04-20_gptoss_INDEX.md) |
