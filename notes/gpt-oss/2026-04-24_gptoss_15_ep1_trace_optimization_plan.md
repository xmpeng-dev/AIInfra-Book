# GPT-OSS-20B mbs=4 / gbs=32 — EP=1 单步 trace 重新校准 + 优化栈重排

**日期**: 2026-04-24
**硬件**: MI355X × 8，单节点，TP1 PP1 EP1，DP=8，GBS=32 / MBS=4，FP8 hybrid
**Trace**: `run-trace/torch_trace_base_20260423_223318/torchprof/primus-megatron-exp[gpt_oss_20b]-rank[2].1777002476000591444.pt.trace.json`
**ProfilerStep**: #17（rank 2），步长 1129.33 ms / 32 sample = 35.3 ms/sample
**渲染**: canvas `mi355x-ep1-gbs32-trace.canvas.tsx`
**关联**: notes `12`（comm overlap B 系列）、`14`（HSDP 负结果 + Grad Sync 重估）

## TL;DR

把 note `14` 里"Grad Sync 真实暴露 ≈ 35 ms"的判断用新 trace 复核了一遍，并且第一次拿到了
**全步骤 + 所有 stream 的 kernel 级分类**。结论：

1. **Comm 这条路已经走完了。** 真实暴露的 RCCL ≈ 46 ms (4.1% step)，
   80.7% 的 RCCL 流量被 compute 盖住。继续打 DDP 旋钮上限就是 ~3%。
2. **新的最大可削目标 = stream 0 上的 elementwise / cast / norm tax**
   = 160 + 53 + 39 = **252 ms (22% step)**，且**没有副流可以救**——每一毫秒都进 wall。
3. Compute stream busy **95.2%**，4 条 grouped-GEMM 副流 152% 过订阅，
   GEMM 总量 941 ms 里很大一块被并行掉了。GEMM 这条路已经被压得很满。
4. **Optimizer tail ≈ 40 ms** 单流串行，所有副流空闲——这是第二块可拿的肉。

按 ROI 重排的优化栈：**Elementwise fusion (Tier 1, ~7-9%) > Optimizer graph capture (Tier 2, ~2%) > 剩余 comm 旋钮 (Tier 3, ~1%) > attention bwd (Tier 4, 待 Tier 1 完成后再看)**。

## Trace 给出的硬约束

| 指标 | 数值 | 含义 |
|---|---:|---|
| step wall | 1129 ms / 32 sample | 35.3 ms/sample |
| stream 0 (compute) busy | **95.2%** | 计算流已基本饱和，wall 几乎等于 stream 0 时长 |
| RCCL 隐藏率 | **80.7%** | 真实暴露的 comm ≈ 240 × (1−0.807) ≈ **46 ms (4.1%)** |
| 多流过订阅 | 152% | 4 条 grouped-GEMM 副流（13–16）+ 主流并行 |
| 活跃 stream 数 | 8 | 1 compute + 4 grouped-GEMM + 1 RCCL + 1 memcpy + 1 idle |

**重要**：trace 里 "NCCL all-to-all (MoE-style) 130 ms" 这一行**不是 a2a**，是 `ncclDevKernel_Generic_1`
里没能被 `c10d::*` cpu_op 覆盖到的 DDP bucket sub-collective。EP=1 路径上没有真 a2a。
methodology callout 里写得很清楚：54/130 ms 已经 relabel 到 AG，剩下的是 short DDP bucket
collective，落在一个 c10d span 内部。

## 全步骤 kernel 分类（rank 2，#17）

| 类别 | Busy (ms) | % step wall | 备注 |
|---|---:|---:|---|
| GEMM (dense + grouped MoE) | 941.07 | 83.3% | **被 152% 过订阅压平了**，stream 0 实际占用远小于 941 |
| Attention (FMHA fwd/bwd) | 203.21 | 18.0% | bwd 是后半段长杆 |
| Elementwise / cast / activation | 160.28 | 14.2% | **stream 0 独占** |
| NCCL all-to-all (实为 DDP generic) | 129.50 | 11.5% | 90% 隐藏在 compute 后 |
| NCCL AllGather (DDP) | 98.36 | 8.7% | 80%+ 隐藏 |
| Other (autograd / misc) | 53.10 | 4.7% | **stream 0 独占** |
| RMSNorm / LayerNorm | 39.22 | 3.5% | **stream 0 独占** |
| MoE dispatch / permute / topk | 28.08 | 2.5% | 本地 permute（EP=1 没有 a2a） |
| Optimizer / param update | 16.63 | 1.5% | tail 串行 |
| Reduction / MemCopy / RS / AR | 41.75 | 3.7% | 大头被盖住 |

**stream 0 独占且无法分流的 wall = 160 + 53 + 39 + 16 (opt) ≈ 268 ms (24%)。**
这是真正可削的肉所在。

## Stream 占用一览

| Stream | 角色 | Busy (ms) | % step |
|---|---|---:|---:|
| 0 | compute（attn · 主 GEMM · norm · MoE permute · opt） | 605 | 53.5% |
| 11 | RCCL DDP（RS+AG+AR 专用） | 240 | 21.2% |
| 14 | grouped GEMM lane | 222 | 19.7% |
| 13 | grouped GEMM lane | 205 | 18.1% |
| 16 | grouped GEMM lane | 202 | 17.9% |
| 15 | grouped GEMM lane | 185 | 16.3% |
| 4 | MemCopy / D2D | 56 | 4.9% |
| 12 | (idle) | 0 | 0% |

stream 0 只有 605 ms / 1129 ms = 53.5% busy，**但 95.2% 的 step 时间 stream 0 都有 kernel 在跑**——
说明 stream 0 上 kernel 平均利用率不高（很多 elementwise 跑得很短但占满 launch 间隔），
和"compute kernel 时间 / wall = 53.5%"是两个东西。这个 gap 印证了 elementwise tax 的影响。

## Phase 切分

- **Forward**: 0 – 310 ms (≈ 27% step)
- **Backward**: 310 – 1090 ms (≈ 69% step)
- **Optimizer tail**: 1090 – 1130 ms (≈ 3.5% step)，单 stream 0，无并行

backward 占 69% 是预期的（dense GEMM bwd 要算 dgrad+wgrad）。
optimizer tail 是当前**唯一一段 stream 0 在干活、所有副流空闲**的窗口。

## 重排后的优化栈

### Tier 1 — Elementwise / Cast / Norm 融合 — **目标 −7~9% step**

依据：stream 0 独占的 elementwise(160) + other(53) + norm(39) = 252 ms = 22% step；
没有副流可以救，每 ms 都进 wall。

具体动作：

1. **采样 top-25 GPU kernel 名单**，挑出 elementwise/cast 的具体函数：

   ```bash
   cd /home/xiaompen/mlperf-training/b200 && \
   python full_breakdown.py \
     /home/xiaompen/mlperf-training/run-trace/torch_trace_base_20260423_223318/torchprof/\
   primus-megatron-exp\[gpt_oss_20b\]-rank\[2\].1777002476000591444.pt.trace.json \
     ProfilerStep#17 | sed -n '/Top-25/,/Per-stream/p'
   ```

   预期会看到 `elementwise_kernel<...>`, `vectorized_*`, `bf16 cast`, `silu`, `cat` 这些。

2. **TE epilogue 融合检查**：当前 `te.Linear` / `te.LayerNormLinear` 是否启用了
   `fuse_wgrad_accumulation`、`return_layernorm_output`、`return_bias`，
   把 `bf16 cast → silu → matmul` 收进 TE epilogue。这块在 `run_base.log` 启动 dump
   里能看到当前实际值。

3. **Primus Triton patches 覆盖面**：`PRIMUS_TRITON_GG_*` 已经开了 grouped GEMM，
   但 `permute → cast → grouped_gemm` 这个 chain 是否都进了同一个 Triton kernel？
   如果还在中间走 cast，可以再写一个把 permute 后的 cast 合进去（顺带吃掉部分 28 ms 的 moe_dispatch）。

4. **RMSNorm fused residual**：39 ms norm + 部分 elementwise 是 norm 前后的 add/cast。
   `primus.rms_norm` 是否带 `residual` 参数？没开就开。

预期：252 ms → 150–170 ms，省 80–100 ms ≈ **7–9% step**。

### Tier 2 — Optimizer tail 隐藏 — **目标 −2~3% step**

依据：最后 ~40 ms 是 `multi_tensor_adam` + autograd cleanup，所有副流空闲，纯白白浪费的窗口。

按代价由低到高：

1. **HIP Graph 包 optimizer step**：optimizer 是固定形状，capture 一次后每步 launch 开销几乎归零。
   Megatron 已支持 `--use-cuda-graph`，需要确认 EP=1 + DistOpt 路径不冲突。
2. **Async RS-after-step**：`overlap_param_gather_with_optimizer_step` 需要 PP>1 用不了。
   但可以自定义：把 DistOpt 的 final all-gather 放到下一步 forward 的第一个 layer 之前，
   借第一个 layer GEMM 隐藏。改 Megatron `distrib_optimizer.py` 几十行。

预期：40 ms tail → 15 ms ≈ **2%**。

### Tier 3 — 剩余 comm 旋钮 — **目标 −1% 以内**

note `14` 已经下调过：

| 旋钮 | 状态 | 备注 |
|---|---|---|
| 重新落 B1+B2 (`ddp_average_in_collective=true`, `ddp_bucket_size=100_000_000`) | 🔁 必做 | regress 修复，~1.5% |
| `patch_moe_overlap=true` smoke | ⏳ 试一下 | EP=1 下可能 no-op，15 min 成本 |
| `ddp_bucket_size` 5 点扫 | ⛔ 跳过 | note 12 B2 已扫过，100M 最优 |
| HSDP-2 | ❌ 关闭 | note 14 −23.8% |
| NCCL channel 调参 | ⛔ 跳过 | 噪声地板 |

### Tier 4 — Attention bwd 长杆 — **Tier 1 完成后再看**

依据：FMHA bwd 在 backward 后半段 burst 到 7–11 ms / 14 ms bin，attention 总计 203 ms (18%)，
但当前已经在 ck_v3 上。

- 拉 `aiter::fmha_bwd` 各变体的 hipBLASLt tuning cache 看是否命中最优 tile。
- 检查 sliding-window 那一半层是否真走了短 attention path（trace 里 burst 长度看起来均匀，怀疑没走）。
- 只有 Tier 1 完成后 attention 才会成为占比最高项，**现在不动**。

## 执行顺序

```
Week 1
  [√] 提取 elementwise top-25 kernel 名单            (1 h)
  [√] 重新落 B1+B2 + 80 iter A/B（修 regress）       (1 h)
  [√] patch_moe_overlap smoke                         (15 min)
  [ ] 设计 1-2 个 fused epilogue Triton kernel        (2-3 d)
  [ ] 接入 + 80-iter smoke 验证                       (1 d)

Week 2
  [ ] 评估 fused epilogue 收益，决定是否扩展
  [ ] HIP Graph 包 optimizer step                     (1 d)

Week 3 (条件触发)
  [ ] 若 Tier 1+2 累计 ≥ 8%，跑收敛 run               (~10 h)
  [ ] 若 < 5%，转 Tier 4 attention 优化
```

## 现实的 Amdahl 上限

| 来源 | 可拿上限 | 备注 |
|---|---:|---|
| Tier 1 elementwise fusion | ~9% | 直接砍 stream 0 wall |
| Tier 2 optimizer tail | ~3% | HIP graph + tail 隐藏 |
| Tier 3 剩余 comm | ~1.5% | 主要是 B1+B2 regress 修复 |
| Tier 4 attention | ?% | 等 Tier 1 完成再评 |
| **保守可达** | **~12-13% step** | ≈ 1129 → 980-990 ms |

把目标 step 锚到 **~990 ms / 32 sample = 31 ms/sample** 当作下一个里程碑。
对应 E2E 在 7680 iter（baseline 收敛点）大约 ~7600 s ≈ 2h7min（vs 当前 9777 s ≈ 2h43min，−22%）——
但要注意 7680 iter 是 baseline 的收敛点，单 iter 加速不一定保 sample-to-target 线性提升，
需要全量 E2E 验证。

## 待同步到 `small_llm_moe_pretraining/primus/NOTES.md` 的修订

1. **§2 iteration breakdown**：把 "Grad Sync residual ~231 ms" 改成
   "Grad Sync 真实暴露 ~46 ms（trace 校准，注 15）"，并把 §3 表 Phase A 的期望 Δ 下调到 +0.5–1.5%。
2. **新增 §7：Elementwise / Cast tax = 252 ms**，作为新的最大可削目标，附 trace canvas 引用。
3. **§6 Knobs decisively closed** 显化 "PP-only / EP-only 优化全部不适用" 这一行（已暗含但单列更清晰）。

这三条等下一并落到 NOTES.md，本笔记是 source-of-truth。

## 文件

- Trace canvas（含 per-stream timeline 图）：
  `~/.cursor/projects/.../canvases/mi355x-ep1-gbs32-trace.canvas.tsx`
- 原始 trace：
  `run-trace/torch_trace_base_20260423_223318/torchprof/primus-megatron-exp[gpt_oss_20b]-rank[2].1777002476000591444.pt.trace.json`
- 解析脚本：`b200/full_breakdown.py`
- 工作笔记：`small_llm_moe_pretraining/primus/NOTES.md`
- 关联：notes `12`（comm B 系列）、`14`（HSDP 负结果）
