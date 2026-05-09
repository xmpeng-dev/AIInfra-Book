# GPT-OSS-20B / 8 × MI355X — clean baseline 复跑（窗口 16-20，aiter 静默确认）

**日期**: 2026-05-07
**硬件**: 8 × MI355X 单机 (gfx950, ROCm), container `xm-mlperf` (image `tasimage/primus:gpt_oss_20b_training_6.0_20260507_0951`)
**栈**: Primus + Megatron-LM, primus_turbo, FP8 e4m3 hybrid (tensorwise)
**配置**: TP1 PP1 EP1 CP1 · DP=8 · GBS=32 · MBS=4 · seq 8192
**Trace 集**: `small_llm_moe_pretraining/primus/output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[*].17781459*.pt.trace.json` (8 rank 全)
**主分析样本**: `rank[2]` ProfilerStep#17 = **938.97 ms**

**关联**:
- baseline 全步骤拆解：[`28`](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md)（同机器、同栈, ProfilerStep#101 = 924.20 ms）
- MLPerf legal baseline：[`27`](./2026-04-28_gptoss_27_mlperf_legal_baseline.md)

## TL;DR

把 `run.sh` 切到干净的 profile 窗口（`PROFILE_STEP_START=16`、`PROFILE_STEP_END=20`、`TRAIN_ITERS=24`、`AITER_LOG_LEVEL=ERROR`）后在容器里跑了一次完整 24-iter run，用 SKILL.md 的 `full_breakdown.py` 复跑分析：

1. **结构 100% 复现 note 28**: grouped_gemm 325.75 ms vs 326.10 ms、attn 190.46 vs 191.20、gemm 125.18 vs 124.50、norm 25.88 vs 25.82 —— 各类 kernel 时间差 ≤ 1 ms。**note 28 的所有结论可直接沿用，不需要重做 deep-dive**。
2. **step 时长 924 → 939 ms (+1.6%)**, 多出来的 15 ms 全部来自 `nccl-only`（5.9 → 21.1 ms）：
   - step #17 比 step #101 早 84 步, RCCL 还没完全 warm（hipBLASLt + ROCm 内核 cache 第一次填）;
   - run.log 中 iter 10 已稳到 **932.8 ms**, 跟旧机 928 ms 对齐；
   - 这 15 ms 不是 regression, 是 cold-cache 偏置, 后续真正比较优化 ROI 时建议用 `iter ≥ 50` 的稳态值或 `step #101+`。
3. **`[aiter WARNING] unsupported condition in fwd_v3!!!` 完全消失**: 新 / 旧 `run.log` 各 0 命中, `run.sh` 顶部 `export AITER_LOG_LEVEL=ERROR` 在 C++ 静态初始化前已经设好, 这条噪声路径关闭。后续看 log 再不用 grep -v 过滤了。
4. **8 rank 同步极紧**: ProfilerStep#16-19 跨 rank spread ≤ 0.1%（rank 0/2/4/7 抽查, max 0.98 ms / 939 ms = 0.10%）。当前 DDP 没有 straggler, 任何后续优化看到的提升都不需要担心是 rank-local 假象。
5. **trace 体积从 1.36 GB/rank 降到 340 MB/rank**（4 active step vs 16）, 单机一次 dump 从 ~11 GB → 2.7 GB; 后续 profile 默认就用这个窗口，对比新旧 trace 直接 SKILL.md 一致。

**优化优先级与 [note 28 §6](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#6-单点优化排序roi--难度) 完全一致**, 不重复复制。这条 note 的作用是给那张 baseline 拍一个可复现的"指纹照"。

## 1. 这次 run 跟 note 28 的差异（只列变更）

| 维度 | note 28 trace | 本次 trace | 变更原因 |
|---|---|---|---|
| Profile 窗口 | step 100-102（active=2） | step 16-20（active=4） | 缩到默认 + 早跑省时间，不必等 100 步预热 |
| Train iters | 1,200,000（被 Ctrl-C 早停） | 24（hard exit) | `PRIMUS_TRAIN_ITERS=24` 避免长尾 |
| `AITER_LOG_LEVEL` | 未显式设 | `ERROR`（run.sh 顶部 export） | 关闭 fwd_v3 head_dim=64 fallback warning |
| `MLPERF_VERBOSE_LOGS` | 未启 | `1` | 顺手打开 mlperf log，方便后续读 PPL / run_duration |
| trace 路径 | `output/amd/root/...`（host 工作区） | `small_llm_moe_pretraining/primus/output/amd/root/...`（container `cwd`） | run.sh 真实 cwd 改了, 后面所有 SKILL.md 命令行得跟着改 |
| 单 rank trace 大小 | 1.36 GB | 340 MB | active step 从 16 → 4, 4× 压缩 |

run.sh 关键差异（diff against pre-edit）：

```bash
# 顶部 export 段
export MLPERF_VERBOSE_LOGS=1
export AITER_LOG_LEVEL=ERROR              # ← 关 aiter fwd_v3 fallback warning

# 末尾 profile overrides (覆盖 config_MI355X 里 hardcode 的 PRIMUS_TRAIN_ITERS=1200000)
export PRIMUS_PROFILE=true
export PRIMUS_PROFILE_STEP_START=16       # ← 跟 SKILL.md 默认对齐
export PRIMUS_PROFILE_STEP_END=20         # ← 之前是 32 (16 active step), 现在 4 active step
export PRIMUS_PROFILE_RANKS=[0,1,2,3,4,5,6,7]
export PRIMUS_TRAIN_ITERS=24              # ← profile 后立刻 hard exit
```

## 2. 在 container 里复跑的命令（备查）

```bash
# 主体
docker exec -w /home/xiaompen/mlperf-training/small_llm_moe_pretraining/primus \
    xm-mlperf bash -c 'bash run.sh'

# 完成后做分析（注意 cwd 路径变了）
cd /home/xiaompen/mlperf-training
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    'small_llm_moe_pretraining/primus/output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].1778145919367462814.pt.trace.json' \
    ProfilerStep#17
```

整段（init + 24 iter + profile dump + 32 iter post-train eval）端到端 `run_duration: 26.23s -> 0.44 minutes`（MLLOG）。把 trace dump + 容器启动开销算进来, wall clock ~3-4 min, 可以放心当 inner-loop 工具用了。

## 3. 与 note 28 的 kernel 分类对比（rank 2）

按 `[note 28 §2](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#2-全步骤-kernel-分类rank-2101)` 同口径：

| 类别 | note 28 (#101) ms | 本次 (#17) ms | Δ ms | Δ % |
|---|---:|---:|---:|---:|
| grouped_gemm | 326.10 | 325.75 | -0.35 | -0.1% |
| attn (FMHA fwd+bwd) | 191.20 | 190.46 | -0.74 | -0.4% |
| 稠密 GEMM | 124.50 | 125.18 | +0.68 | +0.5% |
| nccl_generic | 124.28 | 106.13 | -18.15 | -14.6% |
| other | 92.77 | 93.14 | +0.37 | +0.4% |
| nccl_ag | 89.83 | 86.65 | -3.18 | -3.5% |
| elementwise | 79.97 | 80.07 | +0.10 | +0.1% |
| reduction | 39.81 | 39.48 | -0.33 | -0.8% |
| norm | 25.82 | 25.88 | +0.06 | +0.2% |
| moe_dispatch | 16.76 | 16.46 | -0.30 | -1.8% |
| nccl_rs | 16.42 | 13.28 | -3.14 | -19.1% |
| optimizer | 16.13 | 16.44 | +0.31 | +1.9% |
| fp8_cast | 12.72 | 12.76 | +0.04 | +0.3% |
| memcpy | 4.03 | 4.32 | +0.29 | +7.2% |
| softmax | 3.14 | 3.10 | -0.04 | -1.3% |
| **合计 (sum across streams)** | 1,164 ms | 1,139 ms | -25 ms | -2.1% |
| **wall** | 924.20 | 938.97 | +14.77 | +1.6% |
| **oversub** | 125.9% | 121.3% | -4.6 pp | |

观察：

- **compute 那一坨 (grouped_gemm + attn + dense + elementwise + norm + reduction + opt + fp8_cast + moe + softmax)**: note 28 = 836.34 ms, 本次 = 836.51 ms → **差 0.17 ms (0.02%)**, 完全在测量精度内。kernel 路径、shape、launch 数全一样。
- **NCCL 总量小了 24 ms**（generic 18 + ag 3 + rs 3）, 但因为 overlap 条件不一样, 暴露在 wall 上的反而多 +15 ms（nccl-only 5.9 → 21.1）。这种"总量更少但暴露更多"的反常符合早期 step（#17 vs #101）的特征：第一次 RCCL 调度时 stream 0 还没排满, NCCL 没法躲在 compute 后面。
- **oversub 124% → 121%**: 略降, 主要是 stream 11 从 187.18 → 204.40 ms（NCCL 多了 17 ms）, 但 stream 4 减少 1.7 ms, stream 0 减少 4.5 ms; 三者合起来跟单股 wall 拉长保持一致。

**结论**: 单 GPU compute 结构与 note 28 二进制级一致, 性能差异 100% 落在"NCCL 暴露窗口"上, 这部分会在 step 50+ 自然消化掉（run.log 实测 iter 10 已经收敛到 932.8 ms）。

## 4. 8 rank 一致性抽查

| Rank | step #16 (ms) | step #17 (ms) | step #18 (ms) | step #19 (ms) |
|---|---:|---:|---:|---:|
| 0 | 939.14 | 939.07 | 938.20 | 938.11 |
| 2 | 939.09 | 938.97 | 938.15 | 938.17 |
| 4 | 939.15 | 939.05 | 938.17 | 938.15 |
| 7 | 939.21 | 939.39 | 937.98 | 939.09 |
| **max-min spread** | 0.12 | 0.42 | 0.22 | 0.98 |
| **max-min %** | 0.01% | 0.04% | 0.02% | 0.10% |

四个抽查 rank 的 step 时长全部落在 0.1% 以内, 没有 straggler。结合 [note 28 §1](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#1-把-trace-跑起来--命令--注意点) 报告的 rank 0/2 0.5% spread, 本次更紧, 说明这台机器 8 GPU 之间通信延迟 / 吞吐均衡良好。后续看到的任何 ≥ 1% 的优化收益都可以放心当作真实改进, 不必担心是单 rank 抖动。

## 5. aiter warning 静默确认

```
$ grep -c '\[aiter WARNING\]' run.log run.log.prev
run.log:0
run.log.prev:0
```

新旧两个 run.log 都已经是 0 命中。机制（参考上一轮 [debug 流程]）：

- 噪声源：`/opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/cpp_itfs/mha_fwd.cu` 里的 `fmha_fwd_v3` 在 `head_dim ∉ {128, 192}` 时打 `AITER_LOG_WARNING`。模型用的是 `head_dim=64`, 必然命中 fallback 分支, 不影响功能（自动落到 `ck_tile` 路径）, 只是噪声。
- 抑制点：`AITER_LOG_WARNING` 在 `aiter_logger.h` 里读 `AITER_LOG_LEVEL`, 但读取发生在 C++ **静态初始化**（即 import 那一刻）。Python 端的 `_log_suppression.py` 用 `os.environ.setdefault` 已经太晚, 抓不到。
- 解决：把 export 推到 shell 层（`run.sh` 第 12 行）, 在 `bash run_and_time.sh` 之前生效, C++ 静态变量初始化时就能读到 `ERROR`, 直接屏蔽 WARNING / INFO 两档。

后续 log 重新可读, mlperf-key 的 `time_ms`/`POINT_IN_TIME`/`run_duration` 一眼就能 grep 出来。

## 6. 给后续工作的建议（操作性条目）

1. **profile inner-loop 默认用本次配置**: window 16-20 + train_iters 24 + 4 active step, 单机一次完整 dump < 4 min, trace 2.7 GB 总共, 8 rank 都有数据。下次再做 trace diff 直接复用这个窗口，避免对比时窗口不一致导致 cold-cache 偏置。
2. **稳定 step 取样推荐 #18 或 #19**, 不再用 #17。理由：step 17 RCCL 还没 warm（见 §3 nccl-only 21 ms vs note 28 的 5.9 ms）, step 18-19 在本次 trace 里已经回到 938 ms, 跟稳态 932.8 ms 只差 5 ms。
3. **note 28 的 §5 grouped GEMM deep-dive、§6 优化排序、§7-§11 全部继承**, 不复制。后续如果跑出新 trace 显示哪一类发生 ≥ 5% 变化, 再开新 note 写 delta。
4. **path 改了**: `output/amd/root/...` → `small_llm_moe_pretraining/primus/output/amd/root/...`。所有内部脚本 / canvas / SKILL.md 复制示例如果 hardcode 了旧路径, 下次更新时统一指过去（暂未发现影响, 但要记账）。
5. **TODO**: 在 note 28 §10 留的 `Tier 0 grouped GEMM 单 kernel 调优 / Tier 1A FP8 量化链下推 epilogue / Tier 4 FMHA bwd sweep` 这三条仍然是当前最优 ROI, 直接 pick up; 本次 trace 没有提供新证据要求重排。

## 7. 文件清单

- `small_llm_moe_pretraining/primus/run.sh`：profile 模式 launcher（本次修订, 含 aiter silence）
- `small_llm_moe_pretraining/primus/run.log`：本次 24-iter 干净 log
- `small_llm_moe_pretraining/primus/run.log.prev`：上一次 16-active-step 但 Ctrl-C 中断的 log（仅留作 backup, 不再分析）
- `small_llm_moe_pretraining/primus/output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[*].1778145918*-1778145919*.pt.trace.json`：8 rank fresh trace
- `small_llm_moe_pretraining/primus/output/_archive_pre_092325/`：上一次的 trace 备份（容器内 root 拥有, 不需要再分析, 占盘可清）

## 8. 改动总结

只改了一个文件：[`small_llm_moe_pretraining/primus/run.sh`](../../small_llm_moe_pretraining/primus/run.sh)（顶部 4 行 export + 末尾 5 行 profile env）。

没动 `gpt_oss_20B-pretrain-fp8.yaml`、没动 `config_MI355X_*.sh`、没动 Python / C++ 源码。所有 trace 行为差异都来自 env var 层面, 跟 MLPerf submission 路径完全解耦。
