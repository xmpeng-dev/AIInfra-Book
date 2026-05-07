# GPT-OSS-20B mbs=4 / gbs=32 — Grad Sync overlap (Phase B / HSDP) 负结果

**日期**: 2026-04-24 02:36 — 02:39 UTC（本机 2026-04-23 21:36 — 21:39）
**硬件**: MI355X × 8，单节点，节点内 XGMI
**基线**: `run_base.log`（mbs=4 / gbs=32 / FP8 hybrid 全量收敛 run）
**关联**: notes `08`（mbs=4 优化链）、`12`（B 系列 comm tuning）、`06`（B11 全量收敛）

## TL;DR

接 note `12` 的 comm overlap 思考线，针对 `run_base` 中残留的 Grad Sync 时间，
试了 HSDP（`num_distributed_optimizer_instances=2`）。

**结果：负优化。** HSDP-2 在 DP=8 下 **+23.8% 慢**（1212 → 1500 ms/iter），
TFLOPS 掉 ~12%，显存多吃 ~10 GB。
完全印证 note `12` 末尾的判断："HSDP only matters at much larger DP. At DP=8 it's pointless."

**附带发现：当前 `run_base.log` 把 note 12 的 B1+B2（M4_C7）丢了。**
重新补回 B1+B2 应能先收回 ~1.2-2.0% 的差距，再去做任何新对比。

**动作**：HSDP 路径关闭；先把 M4_C7 落回 baseline，再继续。

## 这次实验为什么会上桌

确认 mbs=4 收敛性（`run_base.log`，245,760 samples 命中 loss 3.34，稳态 ~1212 ms/iter）后，
做了一个分析式的 iteration breakdown，估出 ~230 ms/iter 还挂在 Grad Sync 上。

**这个估计偏高了**：note `12` 用 trace 实测过，B12 之后的 exposed comm 只有 ~50 ms/iter，
B12 之前也只有 ~65 ms。我把"总传输时间 ≈ 240 ms"误算成了"暴露的 wall time"。

候选 plan 是：

| Phase | 旋钮 | 期望 | 状态 |
|---|---|---|---|
| A | `ddp_bucket_size` 5 点扫 {20/40/80/160/320 MB} | +1-3% | 不跑了，note `12` B2 已扫过，100M 最佳 |
| **B** | `num_distributed_optimizer_instances=2` (HSDP) | est +2-5% | **本笔记：-23.8%** |
| C | `patch_moe_overlap=true` | +1-3% | 待 smoke |

## Phase B — HSDP-2 实测

**Run log**: `small_llm_moe_pretraining/primus/run_phaseB_smoke.log`
**yaml diff vs run_base**: 仅 `num_distributed_optimizer_instances: 2`，其它不动

### 稳态指标

| 指标 | baseline (run_base) | HSDP-2 | Δ |
|---|---:|---:|---:|
| ms / iter（稳态） | **1212** | **1500** | **+288 ms (+23.8%)** |
| TFLOPS / GPU | ~625 | ~550 | -12% |
| Tokens / s / GPU | ~25,100 | ~21,850 | -13% |
| HBM 已用 | ~256 GB | **~267 GB** | +10 GB (+4%) |
| HBM 占比 | 89% | 92.7% | — |

iteration 时间线（HSDP-2）：
```
iter 10: 1526.7 ms  (cold)
iter 30: 1498.3 ms
iter 50: 1497.9 ms
iter 60: 1500.0 ms  ← 稳态
```

iter 60 被 SIGTERM 结束（smoke 目标）。

## HSDP-2 为什么在 DP=8 反而变慢

`num_distributed_optimizer_instances=2` 把 DP=8 group 重排成：
```
4 (sharded)  ×  2 (replicated)
```

- **Param all-gather** 改在 DP=4 内部做 → 单次消息变小
- **Grad reduce-scatter** 也在 DP=4 内部 → 单次消息变小
- **但** optimizer state 在两份 replica 之间复制了一份，
  每个 step 必须额外做一次**跨 replica 的 optimizer state all-reduce**

DP=64+ 时 AG 省下来的字节数大到能压住额外 AR；DP=8 时：
- 原来的 RS+AG exposed ~50-65 ms/iter（note `12` 实测）
- 新增的跨 replica AR 在 ~70 GB optimizer state 上要花数十 ms
- 显存代价：optimizer state 复制 2× → 实测 +10 GB

DP=8 这个规模上算不平账。

跟 note `12` 的预测以及 Megatron HSDP 的设计意图（64+ GPU regime）都对得上。

## Regression flag — run_base 漏掉了 B1/B2

排查 baseline 的时候发现：

```
ddp_average_in_collective ....................... False    # note 12 B1: 应为 True
ddp_bucket_size ................................. None     # note 12 B2: 应为 100_000_000
```

note `12`（M4_C7）实测过这两个组合 **-1.2% 到 -2.0%**，并已建议落地。
但 `run_base.log` 里完全没带。

**TODO**：跑任何新的短 A/B 之前，先把 B1+B2 加回当前 baseline yaml。
否则后续每个所谓"新增收益"里都混着一部分追 M4_C7 的缺口。

## Grad Sync 真正还剩多少

回看 note `12 §"Headroom remaining on the comm path"`：
- B12 之后 exposed comm ≈ **35 ms/iter**（≈ 2.9% step）
- 总 NCCL 传输 ~240 ms，但绝大部分被 compute 盖住

也就是说，我之前那个"~231 ms 归因 Grad Sync"是**总传输时间**，不是**暴露时间**。
纯 DDP 调度还能拿的 Amdahl 上限大概是 **3% step**，不是 19%。

含义：comm/sync 路径上现实可拿的剩余收益就只有：
- 重新落 B1+B2（~1.5%）
- 也许 `patch_moe_overlap`（待定，EP=1 下可能 no-op）
- 其它（HSDP、NCCL channel 调参等）都在噪声地板里

之后**再要拿大头收益必须打 compute**（GEMM、attention、Triton fusion），不是 comm。

## 决策

| 旋钮 | 状态 | 原因 |
|---|---|---|
| `num_distributed_optimizer_instances=2` (HSDP-2) | ❌ 关闭 | DP=8 下 -23.8%；印证 note 12 预测 |
| `ddp_average_in_collective=true` (B1) | 🔁 **重新落** | regress 出 run_base 了；单独 -1.4% |
| `ddp_bucket_size=100_000_000` (B2) | 🔁 **重新落** | regress 出 run_base 了；叠加再 -0.5-1% |
| `patch_moe_overlap=true` | ⏳ 待 smoke | 低成本试一下 |
| `ddp_bucket_size` 5 点扫 | ⛔ 跳过 | note 12 已扫过，100M 最佳 |
| 更多 DDP/NCCL 调参 | ⛔ 跳过 | 噪声地板 (~3% 总 headroom) |

## 下一步

1. **把 M4_C7 重新落回 baseline**（B1+B2 加到当前 run_base 配置上），
   跑 80 iter 短 A/B 看新稳态。期望 ~1190-1200 ms（vs 当前 ~1212）。
2. **smoke `patch_moe_overlap=true`**（15 min，单 A/B）。决定要不要保留。
3. **回到 compute 侧**：
   - GEMM (~510 ms/iter) — 剩下最大头
   - MoE Triton kernel (~95 ms) — 看有没有 fusion 空间
   - attention 路径 note `13` 已经处理过

## 文件

- 工作笔记：`small_llm_moe_pretraining/primus/NOTES.md`
- HSDP-2 原始 log：`small_llm_moe_pretraining/primus/run_phaseB_smoke.log`
- baseline（真 source-of-truth）：`small_llm_moe_pretraining/primus/run_base.log`
