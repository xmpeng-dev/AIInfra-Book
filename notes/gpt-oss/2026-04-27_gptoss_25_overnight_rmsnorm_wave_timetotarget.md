# GPT-OSS-20B / MI355X — 今日 RMSNorm 优化 wave 的过夜 MLPerf time-to-quality A/B

> ⚠️ **PARTIALLY SUPERSEDED — 见 [GPT-OSS MLPerf legal baseline 复测](2026-04-28_gptoss_27_mlperf_legal_baseline)**
>
> 本 note 跑的是 **engineering schedule**（`max_steps=12000`, `lr_decay=11872`），
> **不是 MLPerf v6.0 closed division 合法配置**（规则要求 `max_steps=1200000`,
> `lr_decay = 1_200_000 − warmup`）。因此：
> - **§1 / §2.2 / §4 报的 −7.71 % wall TTT 和 −5.88 % iter@target 是工程 A/B 数字，不能进 submission。**
>   note 27 在合法 schedule 下重测，wave 真实 wall 增益是 **−2.15 %**，iter@target Δ=0。
> - **§2.1 的 per-iter −2.0 % / TFLOP/s/GPU +2.15 % 仍然有效**（per-iter 跟 schedule 无关）。
> - 收敛"红利"主要来自 12k cosine 在 iter ~6k 时已退到 ~4.3e-4 时 wave 跟较低 LR 的耦合，
>   合法 schedule 下 LR 全程 ≈ 8e-4 不动，这部分 wave 增益消失。

**日期:** 2026-04-25 夜跑 → 2026-04-27 早晨汇总
**硬件:** 8 × MI355X 单机, TP1 PP1 EP1, GBS 32 / MBS 4
**目标:** MLPerf closed, eval lm_loss ≤ 3.34, BF16 ↔ FP8 hybrid recipe
**⚠️ Schedule:** `max_steps=12000`, `lr_decay=11872`（**engineering A/B preset，非 MLPerf-legal**）
**两组 run（同 SEED=1234，顺序执行，相同 dataset cache）:**

| 组 | `PRIMUS_FUSED_RESIDUAL_NORM` | `PRIMUS_MOE_SWIGLU_NOCAT` | 描述 |
|---|---|---|---|
| **BASE** | 0 (OFF) | 0 (OFF) | 没做今日的 RMSNorm wave |
| **OPT**  | 1 (ON)  | 1 (ON)  | V1 fused residual+RMSNorm + MoE SwiGLU no-cat |

V2 (`PRIMUS_FUSED_RESIDUAL_NORM_V2`) 都是 0 — note 24 已确认 neutral。

## 1. TL;DR — MLPerf 官方 TTT（submission RESULT 口径）

| metric | BASE | OPT | Δ |
|---|---|---|---|
| **RESULT TTT (wall, MLPerf submission 口径)** | **8364 s (139.40 min)** | **7719 s (128.65 min)** | **−645 s / −10.75 min / −7.71 %** |
| `samples_count` @ target | 208,896 | 196,608 | −12,288 (−5.88 %) |
| `iter` @ target         | 6,528   | 6,144   | −384 (−5.88 %) |
| `eval_accuracy` @ run_stop | **3.331524** | **3.332806** | 都 ≤ 3.34 ✓ |
| `run_stop.status` | `"success"` | `"success"` | ✓ |
| NaN / skipped iter | 0 / 0 | 0 / 0 | 稳定 |

RESULT 行来源（log 尾行，MLPerf submission 提交格式）:
```
BASE: RESULT,GPT_OSS_20B,,8364,AMD,2026-04-25 02:58:41 PM
OPT : RESULT,GPT_OSS_20B,,7719,AMD,2026-04-25 05:18:21 PM
```

三种 TTT 口径对照（仅供 cross-check）:

| 口径 | BASE | OPT | 定义 |
|---|---|---|---|
| **RESULT 行（官方 TTT）** | **8364 s** | **7719 s** | 整个 phase wall：init + dataset load + run_start→run_stop + teardown |
| MLLOG `run_duration` | 8281.91 s | 7642.54 s | 仅 `run_start → run_stop`（核心训练段，不含 init / teardown） |
| runner.log bash wall | 8364 s | 7719 s | docker exec 测的 wall，与 RESULT 行完全一致 |

**用 MLPerf submission 的官方 TTT（RESULT 行），今日 RMSNorm wave 让达到 quality target 的 wall time 缩短 10.75 分钟（−7.71 %）**。

## 2. 为什么是 7.71 %（不是 2 % 的 per-iter 加速）

提升拆成两层：

1. **每 iter 更快（−2.0 %）** — Triton fused-residual+norm 和 SwiGLU no-cat kernel 减少 stream 0 上的 elementwise tax。稳态 ms/iter 分布（去掉 eval 前后 ±4 iter、n=579 common steady-state iters，见 §3）:

   | | BASE | OPT | Δ |
   |---|---|---|---|
   | median ms/iter | 1202.10 | 1176.70 | **−25.40 ms (−2.11 %)** |
   | trim10 ms/iter | 1204.86 | 1180.72 | **−24.14 ms (−2.00 %)** |
   | median TFLOP/s/GPU | 687.0 | 701.8 | **+14.80 (+2.15 %)** |
   | iters where OPT faster than BASE | — | 517 / 579 (**89.3 %**) | 非常一致 |

2. **收敛略微更快（−5.88 %）** — 达到 `eval_accuracy ≤ 3.34` 少跑 384 个 iter。eval 曲线见 §3；从 iter ~1500 起 OPT 的 eval lm_loss 稳定比 BASE 低 0.02~0.04。

合并：`(1 − 0.0211) × (1 − 0.0588) = 0.9216`，预期 −7.84 %；实测 −7.71 %。符合得很好。

## 3. Eval 曲线（每 384 iter 一次 eval，17 个 eval 点 × 2 runs）

```
  iter     BASE_lm    BASE_ppl      OPT_lm     OPT_ppl       Δ_lm    Δ_ppl_%
  --------------------------------------------------------------------------
   384     5.28983     198.309     5.28742     197.833   -0.00240     -0.24%   (warmup 末期，一致)
   768     4.54163      93.843     4.54585      94.240   +0.00422     +0.42%   (微偏离，噪声)
  1152     4.24488      69.748     4.23192      68.849   -0.01296     -1.29%   (OPT 开始领先)
  1536     4.02019      55.712     4.01478      55.411   -0.00541     -0.54%
  1920     3.88212      48.527     3.85896      47.416   -0.02317     -2.29%
  2304     3.78344      43.967     3.75813      42.868   -0.02531     -2.50%
  2688     3.72885      41.631     3.68577      39.876   -0.04309     -4.22%   (最大 gap)
  3072     3.65196      38.550     3.61802      37.264   -0.03394     -3.34%
  3456     3.59643      36.468     3.56720      35.417   -0.02923     -2.88%
  3840     3.55185      34.878     3.52256      33.871   -0.02928     -2.89%
  4224     3.51045      33.463     3.48253      32.542   -0.02791     -2.75%
  4608     3.47921      32.434     3.45033      31.511   -0.02888     -2.85%
  4992     3.44037      31.199     3.41433      30.397   -0.02604     -2.57%
  5376     3.41096      30.294     3.38537      29.529   -0.02559     -2.53%
  5760     3.38535      29.528     3.36089      28.815   -0.02446     -2.42%
  6144     3.35573      28.666     3.33281      28.017   -0.02292     -2.27%  (OPT 过线，run_stop)
  6528     3.33152      27.981          --          --         --         --  (BASE 过线，run_stop)
```

- 早期（iter ≤ 768）两条曲线重合（bf16 噪声同阶）。
- iter ≥ 1152 起 OPT 稳定低 0.013~0.043 lm_loss。
- OPT 在 iter 6144 先穿过 3.34，BASE 在 iter 6528 才穿过 → 差 384 iter（整 1 个 eval-interval）。

## 4. 收敛为什么能"顺手"变快？

同 SEED、同 GBS、同 dataset shuffle。理论上 BASE/OPT 应该得到相同 loss curve。
实际差 ~0.025 lm_loss，且方向**系统性的 OPT 更低**，不是随机抖动。最可能的原因：

1. **fused Triton kernel 的数值精度略好**。原始 Megatron 路径做
   `x_plus_r = add_bf16(x, r)` 先把 x+r 实体化成 bf16，再喂给 RMSNorm
   （RMSNorm 内部有 fp32 累加）。Triton `triton_rmsnorm_residual` 直接
   读 bf16 的 x 和 r，**在 fp32 累加器里合并**，**x+r 从头到尾没有落成 bf16**，
   只在归一化之后才 cast 回 bf16。等价于每层的 residual chain 多了一次
   fp32 精度。24 层积累下来 0.02~0.04 lm_loss 的系统性优势是合理的。
2. 次要：SwiGLU no-cat Triton kernel 也做了类似的 bf16→fp32→bf16 简化。

这解释了：(a) 早期（warmup）差距几乎为零；(b) 中段（iter 1500~3000）差距快速扩大到峰值
0.043；(c) 后段（iter 3000+）差距稳定在 0.025 左右（两条曲线都渐近
on-policy 梯度累加的同一个 fixed point，只是 OPT 到达更快）。

值得 flag：**这是"免费"的 0.5~1 B-step 级别的等效 quality gain**。
在 MLPerf closed 上 time-to-target 是官方 metric，所以这份收敛收益和
per-iter 加速都计入提交成绩。

## 5. Stability

| | BASE | OPT |
|---|---|---|
| NaN iter | 0 | 0 |
| skipped iter | 0 | 0 |
| run_stop.status | `"success"` | `"success"` |
| loss scale | 1.0 全程（bf16 / fp8 hybrid 不用动态 loss scale） | 同 |
| grad_norm 分布 | healthy, no spikes | healthy, no spikes |

两个 run 都从头到尾零 NaN 零跳步，eval loss 单调下降到目标。

## 6. Run artifacts

```
small_llm_moe_pretraining/primus/ab_runs/20260425_overnight_12k/
├── runner.log                              ← phase timing
├── runner.stdout.log                       ← docker nohup
├── run_overnight.sh                        ← 顺序 runner（仍可复用）
├── summary.py                              ← 本报告的数据源
├── snapshot.py                             ← 中途 peek 脚本
├── base/
│   ├── base.log      (638 KB)              ← BASE, 6528 iter
│   └── base.run.log  (638 KB)
└── opt/
    ├── opt.log       (616 KB)              ← OPT, 6144 iter
    └── opt.run.log   (616 KB)
```

## 7. 结论 & 下一步

- **今日 RMSNorm wave（V1 fused residual+norm + SwiGLU no-cat）保持默认 ON**
  —— 它带来 **−10.75 min / −7.71 % wall time-to-target**（MLPerf RESULT 口径），
  完全符合 MLPerf closed submission 的 "time-to-quality" 定义，且零精度风险。
- **V2 保持默认 OFF**（note 24 已验证 neutral）。
- **后续优化目标** —— per-iter 层面剩下的大头仍是 attention bwd（~126 ms /
  step, `aiter::fmha_bwd_hd64_bf16_causal_a16_rtne_recompile`）。Tier-2 的
  `use_turbo_attention`、FP8 attention bwd 等已列在 note 22；需要单独的
  trace 会话去评估，不再上 A/B 糊一下了。
