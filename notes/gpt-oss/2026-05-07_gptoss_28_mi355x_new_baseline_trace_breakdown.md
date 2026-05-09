# GPT-OSS-20B / 8 × MI355X — 新机器 baseline trace 全步骤分解

**日期**: 2026-05-07
**硬件**: 8 × MI355X 单机 (gfx950, ROCm), host `mi355-gpu-26`, ip 10.2.144.19
**栈**: Primus + Megatron-LM, primus_turbo, FP8 e4m3 hybrid (tensorwise)
**配置**: TP1 PP1 EP1 CP1 · DP=8 · GBS=32 · MBS=4 · seq 8192
**Trace**: `output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].1778136779066144540.pt.trace.json`
**ProfilerStep**: `#101` (rank 2)，步长 **924.20 ms** / 32 sample = 28.88 ms/sample
**渲染**: canvas
- `gpt-oss-20b-mi355x-baseline.canvas.tsx` (英文)
- `gpt-oss-20b-mi355x-baseline-zh.canvas.tsx` (中文)

**关联**: notes [`15`](./2026-04-24_gptoss_15_ep1_trace_optimization_plan.md)（旧 cluster EP=1 trace plan）、[`27`](./2026-04-28_gptoss_27_mlperf_legal_baseline.md)（MLPerf legal baseline 9234 s）

## TL;DR

把 [`gpu-trace-analysis` skill](../../.cursor/skills/gpu-trace-analysis/SKILL.md) `full_breakdown.py` 在新机器 (mi355-gpu-26) 的 fresh baseline trace 上跑一遍，得到当前真实结构：

1. **新 cluster 的稳态步长 = 924 ms / 32 sample = 28.9 ms/sample**，比 4-28 那台旧机的 1,196 ms 快 **−22.7%**。差距大概率是机器代际 + ROCm 栈 + 已落地 turbo patch 的合并效果，**不是单一旋钮的功劳**；要不要拿来当新 reference 还得跑一次 E2E 验证 schedule 合规（见 §6）。
2. **彻底 compute-bound**: GEMM (47.0%) + 注意力 (20.3%) = 67.3% wall。RCCL 真正暴露的只有 **25.3 ms (2.7%)**，GPU 只闲置 3.7 ms (0.4%)。换句话说 DDP 这条路在这台机器上**完全走完了**，比旧 trace 的 4.1% 暴露还紧。
3. **EP=1 仍然是结构性瓶颈**: 32 个专家驻留每个 rank, MoE grouped GEMM 308 ms (33% step) 串行压在 stream 0。stream oversub 只有 124%（旧 trace 里 4 条 grouped-GEMM 副流时是 152%）—— **新机器多流并行度反而下降**，需要核实 yaml 是不是退化到没有副流 expert lane。
4. **新的最大单点 = `aiter::fmha_bwd_hd64_bf16_causal` = 113.8 ms (12.3% step)**。GEMM 之外它就是最长的那根杆子，比所有 elementwise 加起来还重。Tier 4（attention bwd）在这次 trace 上提前升到 ★★。
5. **31–38% 是一段 lm_head + 交叉熵的纯稠密 GEMM 块**，stream 11 完全空闲，没有任何 collective 叠加 —— 这是新发现的可塞 bubble 的窗口。

按 ROI 重排：**Grouped GEMM 单 kernel 调优 (新 Tier 0, §5) ≈ FMHA bwd 调优 (Tier 4) ≈ FP8 量化链下推 epilogue (Tier 1A, 新主目标) > lm_head 段塞 RS (Tier 2′) > SwiGLU bwd 去 cat (Tier 1B) > 残余 comm (Tier 3, ⛔)**。

> Tier 1 fused-residual-rmsnorm V1/V2 / Triton RMSNorm / RoPE / TE cast_transpose **都已经在这次 trace 里直接看到**（详见 §6 Tier 1）, 那条线已基本榨干; 真正剩的是 **FP8 量化链 71.6 ms 没下推到 hipBLASLt epilogue**, 这是 elementwise 这边新的最大可拿肉。

**§5 单独把 grouped GEMM 拆出来分析了 shape / per-launch / TFLOP/s**: 有效算力 1524 TFLOP/s = MI355X FP8 peak (5033) 的 **30.3%**, 头顶有 1.5–2× 空间; 其中 dgrad (`variable_k`) 路径只到 26% peak, 是最弱的一段。详见 §5。

## 1. 把 trace 跑起来 — 命令 + 注意点

```bash
cd /home/xiaompen/mlperf-training
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    'output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].1778136779066144540.pt.trace.json' \
    ProfilerStep#101
```

几个 trace 自身的坑，下次复测时得避开：

| 现象 | 解释 | 处理 |
|---|---|---|
| `ProfilerStep#17` 不存在 | 这次 `PRIMUS_PROFILE_STEP_START` 设到了 100；trace 里只有 `#100/#101/#102` | 用 `#101` 当 mid-step（前后两个步长一致 ±0.5%） |
| 同一 `#101` 在 trace 里出现两次（924 ms 和 270 ms） | 一条是 CPU annotation 视角，一条是 GPU 视角；ijson 第一个命中是 CPU 那条 | 解析器自动取第一条；两个数字差 654 ms = grad_reduce + opt 的 CPU side wait |
| 8 个 RCCL kernel 走 `ncclDevKernel_Generic_*` 路径 | RCCL 把多次 AG/RS bucket 打成一个 persistent kernel | `SPLIT_NCCL_BY_CPU=1` 用 `c10d::*` cpu_op 时间戳重打 8 个标签；剩下 124 ms 仍 generic |
| stream 12 报告 24 个 kernel 但只 0.10 ms busy | 全是 0-byte memcpy probe，可忽略 | 视作未用 |

ranks 间一致性：rank 2 vs rank 0 的 step 时长差 0.5%（923.73 vs 924.20），在 ±3% 容许内。

## 2. 全步骤 kernel 分类（rank 2，#101）

按 stream 求和后的 1144 ms vs wall 924 ms → oversubscription = **123.8%**（旧 trace 是 152%）。

| 类别 | Busy (ms) | % step wall | 备注 |
|---|---:|---:|---|
| Grouped GEMM (MoE 专家) | 308.28 | 33.4% | 全在 stream 0；EP=1 没拆开 |
| Attention (FMHA fwd + bwd) | 187.77 | 20.3% | bwd 占 134 ms，fwd 占 53 ms |
| 稠密 GEMM (qkv / out / proj / lm_head) | 126.28 | 13.7% | hipBLASLt FP8 + bf16 各占一半 |
| RCCL Generic (打包集合通信) | 124.28 | 13.4% | **绝大部分被 hide**，不是真实暴露 |
| 其它小 kernel | 92.77 | 10.0% | 待 §3 拆 |
| RCCL AllGather (param gather) | 89.83 | 9.7% | |
| Elementwise / 激活 | 79.97 | 8.7% | swiglu_with_mask + vectorized_elementwise |
| Reduction | 39.81 | 4.3% | 大部分是 sum / amax |
| RMSNorm | 25.82 | 2.8% | turbo `_rmsnorm_*_kernel` |
| MoE 路由 / permute / topk | 16.76 | 1.8% | local，无 a2a |
| RCCL AllReduce | 16.42 | 1.8% | 单个，最末段 |
| Adam 优化器 (multi_tensor) | 16.13 | 1.7% | tail，全程 stream 0 |
| FP8 cast / amax | 12.72 | 1.4% | turbo unary cast |
| MemCopy / D2D | 4.03 | 0.4% | |
| Softmax | 3.14 | 0.3% | |

**stream 0 独占且没有副流可救的部分（每 ms 进 wall）**:
elementwise(79.97) + other(92.77) + reduction(39.81) + norm(25.82) + opt(16.13) + fp8_cast(12.72) + moe(16.76) + softmax(3.14) ≈ **287 ms (31% step)**。

跟旧 trace 的 252 ms (22%) 相比**绝对值反而上升**，但因为 step 整体压短了，相对占比也涨。这意味着 Tier 1 elementwise tax 这条路依然有肉，但优先级被 attention bwd 抢走了（见 §4）。

## 3. Stream 占用一览

| Stream | 角色 | Busy (ms) | % step | n kernels |
|---|---|---:|---:|---:|
| 0 | 计算（GEMM · attn · norm · MoE · opt） | 893.92 | **96.7%** | 4,421 |
| 11 | RCCL DDP（Generic / AG / AR） | 227.78 | 24.6% | 16 |
| 4 | elementwise 副流 | 22.23 | 2.4% | 151 |
| 12 | memcpy（基本闲置） | 0.10 | 0.0% | 24 |

stream 0 96.7% busy + 124% oversub → wall 几乎等于 stream 0 时长，跟旧 trace 的 95.2% busy + 152% oversub 是**完全不同的形态**。差异：

| 维度 | 旧 trace (4-23, 1129 ms/step) | 本次 trace (5-07, 924 ms/step) |
|---|---|---|
| 活跃 stream 数 | 8（1+4+1+1+1） | **4（1+1+1+1）** |
| Grouped-GEMM 副流 | 4 条（13–16），各 ≈200 ms | **0 条** |
| Stream oversub | 152% | 124% |
| RCCL 隐藏率 | 80.7% | **89.1%** |
| Compute kernel 占用率 | 95.2% | **96.9%** |

→ **Grouped GEMM 在新 trace 里没有用副流并行**，跟旧 trace 的最大形态差异。可能原因：
1. yaml 里 `moe_use_legacy_grouped_gemm: true` + `use_turbo_grouped_mlp: true` 在新栈下走的是 persistent single-stream kernel（`_grouped_fp8_persistent_gemm_kernel`，188 ms 单 kernel），这个 kernel 已经把所有专家串行喂进同一个 launch；
2. 或者 EP=1 + 新 turbo 路径下 expert lane 多流策略已被关掉；
3. 也可能是 [`gptoss_20`](./2026-04-25_gptoss_20_use_turbo_grouped_mlp_envvar_truthy_fix.md) 那个 `use_turbo_grouped_mlp` truthy 修复后行为变了。

**TODO**: 启动 log 里 dump `use_turbo_grouped_mlp` / `moe_grouped_gemm` 的实际生效值，确认是上面哪种路径，再判断要不要把多流形态找回来（旧 trace 的 4 条副流让 GEMM 总量 941 ms 几乎完全藏在 wall 之外）。

## 4. Phase 切分 + 多流时间线

按 80 个 11.55 ms bin 的 kernel mix 看 step 大致结构：

| 段 | 区间 (% step) | 长度 | 主要内容 |
|---|---|---:|---|
| Embed warm-up | 0 – 2% | ~18 ms | embedding lookup + norm + 第一次 AG |
| **Forward 24 层** | 2 – 31% | ~268 ms | 每层 ~11.5 ms：qkv → fmha fwd → out → norm → moe → grouped MLP |
| **稠密尾块（lm_head + ce loss）** | 31 – 38% | ~65 ms | 全部 GEMM，**stream 11 完全空闲** |
| **Backward 24 层** | 38 – 93% | ~510 ms | fmha_bwd (134 ms) + grouped MLP bwd (208 ms) 交错；RCCL RS bucket bursts |
| Grad finalize | 93 – 95% | ~18 ms | fp8 cast + reduction + 最后一波 AR |
| Adam tail | 95 – 97% | ~18 ms | `multi_tensor_adam` 单 stream |
| Idle | 97 – 100% | ~28 ms | step boundary，等下次 forward |

跟旧 trace 比，关键变化是 **31–38% 出现了一整段稠密 GEMM-only 窗口**，stream 11 在这 65 ms 内完全没有 RCCL 流量。这是个净收益候选窗口（详见 §5）。

## 5. Grouped GEMM 单独剖析（shape · per-launch · TFLOP/s）

整个 step 里 GEMM 占 47%, 其中 grouped GEMM 一项就占 33%, 是单点最大的开销类别。
这一节单独把 grouped GEMM 拆开 — shape、kernel 路径、每次 launch 时长、有效算力。

### 5.1 Kernel 路径 & launch 数

| kernel | n launches | total ms | avg ms | min / max ms | 角色 |
|---|---:|---:|---:|---:|---|
| `_grouped_fp8_persistent_gemm_kernel` | 96 | 188.77 | 1.97 | 1.16 / 4.75 | fwd + wgrad（两个都是 K 固定 = tokens 数） |
| `_grouped_variable_k_gemm_kernel` | 48 | 119.51 | 2.49 | 1.57 / 3.35 | dgrad（K = tokens 在 expert 间不等长，要 variable-K） |

每层 6 次 launch（2 fwd + 2 wgrad + 2 dgrad）× 24 层 = 144 次, 与统计一致。

CPU 侧的 autograd 包装：

| op | n | total ms | 备注 |
|---|---:|---:|---|
| `GroupedGemmFP8TensorFunc` (fwd 入口) | 48 | 13.02 (CPU) | 24 层 × 2 (W1, W2) |
| `GroupedGemmFP8TensorFuncBackward` | 48 | 16.21 (CPU) | 24 层 × 2，每次 spawn 一个 dgrad + 一个 wgrad |
| `primus_turbo::grouped_gemm_fp8_impl` (实际 fp8 入口) | 96 | 15.54 (CPU) | 48 fwd + 48 wgrad，把 BF16 权重 cast→FP8 后 launch persistent kernel |

### 5.2 真实 shape（从 trace `Input Dims` 拿到，不是从 yaml 推出来）

虽然 yaml 里 `record_shapes=false`, `primus_turbo` 的 autograd 包装层把 shape 写成了 `Concrete Inputs`/`Input Dims`, 所以 trace 里直接能读出来：

```
GroupedGemmFP8TensorFunc (W1, gate_up 合并):
  Input A (activations): [131072, 2880]        BF16
  Input B (weight):      [32,    2880, 5760]   BF16   ← E×K×N (gate+up cat 在 N 维)

GroupedGemmFP8TensorFunc (W2, down):
  Input A (activations): [131072, 2880]        BF16
  Input B (weight):      [32,    2880, 2880]   BF16   ← E×K×N

primus_turbo::grouped_gemm_fp8_impl:
  Input A: FP8 e4m3   [131072, 2880]
  Input B: FP8 e4m3   [32, 2880, 2880] 或 [32, 2880, 5760]   (impl 不区分 W1/W2，按调用区分)
  expert offsets:     [32]   long
  expert offsets+1:   [33]   long
```

几个推出来的关键数：

| 量 | 值 | 来历 |
|---|---:|---|
| MBS × seq | 32,768 tokens / GPU / microbatch | 4 × 8192 |
| topk dispatch 后 | **131,072 expert assignments** | 32,768 × 4 (top-4) |
| 32 个专家平均 | **4,096 tokens / expert** | 131,072 / 32 (假设负载均衡) |
| W1 (gate_up 合并) | E=32, K=2880, N=**5760** | 5760 = 2 × ffn_hidden = 2 × hidden(SwiGLU 双门) |
| W2 (down) | E=32, K=2880, N=2880 | K=ffn, N=hidden, 平方 |

注意 `moe_router_load_balancing_type: none` + `moe_pad_expert_input_to_capacity: false` 意味着 router 不强制均衡, 也不做 capacity pad —— 实际 per-expert tokens 会浮动, 这是后面 per-launch 时间方差的来源之一。

### 5.3 每次 launch 的 FLOPs

每次 grouped GEMM 处理 **全部 131,072 个 token-expert 分发**, 内部按 expert 分组, 一次 kernel 启动跑完。所以 per-launch FLOPs = `2 × M × K × N` (M = 131,072, 不是 4,096):

```
W1 per-launch:  2 × 131,072 × 2880 × 5760 = 4.349 TFLOP
W2 per-launch:  2 × 131,072 × 2880 × 2880 = 2.174 TFLOP
```

24 层 × 单 step 总 FLOPs:

| 阶段 | kernel | per-launch | n launches | 总 FLOPs |
|---|---|---:|---:|---:|
| fwd W1 (gate_up) | persistent | 4.349 TFLOP | 24 | 104.4 TFLOP |
| fwd W2 (down) | persistent | 2.174 TFLOP | 24 | 52.2 TFLOP |
| wgrad W1 | persistent | 4.349 TFLOP | 24 | 104.4 TFLOP |
| wgrad W2 | persistent | 2.174 TFLOP | 24 | 52.2 TFLOP |
| dgrad W1 | variable_k | 4.349 TFLOP | 24 | 104.4 TFLOP |
| dgrad W2 | variable_k | 2.174 TFLOP | 24 | 52.2 TFLOP |
| **合计** | | | **144** | **469.7 TFLOP** |

(GEMM 的 dgrad / wgrad 与 fwd 同 FLOPs, 都是 `2 × M × K × N`。)

### 5.4 实测 TFLOP/s（每个 shape × 每条 kernel 路径）

把 96 次 persistent kernel 按 W shape 分桶（用 impl CPU op 的 `Input Dims` 一一对应 GPU kernel 的执行序）, 再算 TFLOP/s:

| shape × kernel | n | time (ms) | TFLOP | **TFLOP/s** | % FP8 peak (5033) | % BF16 peak (2500) |
|---|---:|---:|---:|---:|---:|---:|
| W1 [32,2880,5760] · persistent (fwd+wgrad) | 48 | 121.61 | 208.7 | **1716** | **34.1%** | 68.6% |
| W2 [32,2880,2880] · persistent (fwd+wgrad) | 48 | 67.15 | 104.4 | **1554** | **30.9%** | 62.1% |
| W1+W2 · variable_k (dgrad) | 48 | 119.51 | 156.6 | **1310** | **26.0%** | 52.4% |
| **aggregate (全部 grouped GEMM)** | **144** | **308.28** | **469.7** | **1524** | **30.3%** | 61.0% |

> peak 数字: MI355X 单卡 FP8 dense matrix = **5,033 TFLOP/s**, BF16 dense matrix = **2,500 TFLOP/s** ([AMD MI355X spec](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html))。Persistent kernel 输入是 FP8 e4m3, 但输出累加和部分中间张量是 BF16/FP32, 所以实际效率介于这两个 peak 之间;  variable_k 看起来更接近 BF16 peak (52%), 怀疑这条路径不是纯 FP8 计算。

### 5.5 Per-launch 时间分布 → 看负载均衡 & 形状效率

每次 launch 处理同样的 131,072 token-expert pair, 时间却差好几倍, 说明：

| kernel × shape | min ms | max ms | max/min | 解释 |
|---|---:|---:|---:|---|
| persistent W1 [32,2880,5760] | 1.84 | 4.75 | **2.6×** | gate_up 合并矩阵, FLOPs 翻倍, 数据量也翻倍; 极端 max 怀疑命中了某次 expert load 严重失衡 (`moe_router_load_balancing_type=none`) |
| persistent W2 [32,2880,2880] | 1.16 | 2.46 | 2.1× | 形状对称小一倍, 时间也大致一半 (avg 1.40 ms vs W1 2.53 ms) |
| variable_k (dgrad) | 1.57 | 3.35 | 2.1× | 比 persistent 略稳, 但有 K 不等长开销 |

观察：

1. **W1 / W2 时间几乎按 N 维 (5760 vs 2880) 线性 scale**, 说明 kernel 已经到 compute-bound 区间, 不是被 memory 限速 (Roofline 检查见 5.6)。
2. **W1 max 4.75 ms ÷ avg 2.53 ms ≈ 1.9×**, 而 expected 全均衡情况下 max ≈ avg, 说明**负载不均衡导致 ~25% 的等待**。如果换成 `moe_router_load_balancing_type=aux_loss` 或 `seq_aux_loss`, expected W1 时间能降到 ~3.6 ms (avg 跟 min 拉近), aggregate 收益 ~30 ms / step ≈ −3.2% step。
3. **variable_k 比 persistent 慢 30%+** (1310 vs 1716 TFLOP/s on the same logical FLOPs)。GEMM 在 K 维 variable 时不能用 hipBLASLt 最优 tile, 是 hipBLASLt 在这条路径上的已知短板。改用 `_grouped_constant_k_gemm` (要把 dgrad 重新 reshape 成 fixed-K) 或者切回 BF16 dgrad 路径或许更快。

### 5.6 Roofline sanity check

W1 fwd 单次 launch:

```
A 读: 131072 × 2880 × 1 byte (FP8)  =   377 MB
B 读:    32 × 2880 × 5760 × 1 byte  =   531 MB
C 写: 131072 × 5760 × 2 byte (BF16) = 1,510 MB
Σ memory  ≈ 2.42 GB
FLOPs     ≈ 4.35 TFLOP

Arithmetic intensity = 4350 GFLOP / 2.42 GB = 1798 FLOP/byte
MI355X roofline 拐点 (FP8) = 5033 TFLOP/s ÷ 8 TB/s = 629 FLOP/byte
1798 ≫ 629  →  完全在 compute roof 区间, 不是内存带宽问题。
```

W2 fwd 同样 compute-bound (AI ≈ 1556 FLOP/byte)。所以 grouped GEMM 这边**唯一拿不到 peak 的原因是 hipBLASLt tile 选择 + expert 负载不均**, 不是 HBM 带宽不够。

### 5.7 Take-aways（grouped GEMM 视角）

1. **当前 ~30% peak utilization (FP8) 还有 1.5 – 2× 头顶空间**。一个调好的 fp8 grouped GEMM 在 MI355X 上应该能到 60–70% peak (≈ 3000–3500 TFLOP/s)。
2. **W2 (down proj) 比 W1 效率更低** (1554 vs 1716 TFLOP/s)。down 的 N=2880 比较小, 一些 hipBLASLt tile 库可能没覆盖好这个 shape。
3. **dgrad (variable_k) 是最弱的一段 (1310 TFLOP/s, 26% peak)**, 占 backward 119.5 ms (12.9% step)。如果换成 fixed-K 路径或者 ROCm 7.x 的新 grouped dgrad kernel, 单这条路径就能省 ~50 ms。
4. **expert 负载不均衡**贡献了 W1 ~25% 的等待时间, 可以靠开 `moe_router_load_balancing_type=aux_loss` 或者 `moe_router_force_load_balancing=true` (yaml 里目前注释掉了) 来减小, 收益 ~30 ms/step。
5. **EP=1 的根本约束**：所有 32 个专家驻留每个 rank, 单 step 走 6 × 24 = 144 次 grouped GEMM, 总 470 TFLOP 全部压在单卡 stream 0 上。如果换成 EP=8（每 rank 4 专家），单 rank 只跑 4/32 = 1/8 的 expert weight, 但 token 经 a2a dispatch 后 batch 变大 8×, 算力需求理论持平但**通信变成新瓶颈** —— 这是 4-23 [note 14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md) 既定结论的一部分。但 grouped GEMM 单 launch 的工作量从 4.35 TFLOP 上升到 ~35 TFLOP, hipBLASLt tile 利用率会显著提升, 说不定比 30% 翻倍。值得用 EP=8 跑一次 80-iter A/B 验证。

→ grouped GEMM 这条优化线在 ROI 表上的位置：**升到 ★★★, 与 FMHA bwd 并列**。具体动作清单见 §6 Tier 0 (新增)。

## 6. 单点优化排序（按 ROI）

### Tier 0 — Grouped GEMM 单 kernel 调优 — **目标 −5 ~ 8% step (新增, ★★★)**

依据：§5 算出当前 grouped GEMM 综合 1524 TFLOP/s = 30% FP8 peak, 头顶有 1.5–2× 空间。三条具体路径:

1. **dgrad 走 fixed-K 路径**: 当前 `_grouped_variable_k_gemm_kernel` 在 hipBLASLt 上只跑到 26% peak。试着在 backward 里把 token-batched gradients 重新 reshape 成 fixed-K layout, 调 `_grouped_constant_k_gemm` (如果存在) 或者退到 `_grouped_fp8_persistent_gemm` 同款 tile。**预期收益: 119.5 ms → 80–90 ms ≈ −3% step**。
2. **W2 (down) shape 调优**: `[32, 2880, 2880]` 在 hipBLASLt tile 库里命中率比 W1 低 (1554 vs 1716 TFLOP/s)。先 dump `HIPBLASLT_LOG_LEVEL=4` 看实际 tile, 再用 `hipblaslt-bench` 单独 sweep 这 shape。**预期: 67 ms → 55 ms ≈ −1.3% step**。
3. **expert 负载均衡**: 当前 `moe_router_load_balancing_type: none`, W1 max/avg 比例 1.9× 暗示 ~25% 等待时间是 stragglers 造成的。开 `moe_router_force_load_balancing=true` (yaml 已经预留 ENV, 注释掉了) 做 80-iter A/B, 看是否影响收敛。**预期: aggregate 308 ms → 280 ms ≈ −3% step**。

(注: 1+2+3 不能完全相加, 1 和 2 都是 hipBLASLt 端, 3 是 router 端正交收益。综合 −5~8% 是估上界。)

### Tier 4 — FMHA bwd 调优 — **目标 −4 ~ 6% step (从 ★ 升到 ★★★)**

依据：`aiter::fmha_bwd_hd64_bf16_causal_a16_rtz_recompile` 单 kernel **113.82 ms (12.3% step)**，比 elementwise 总和（80 ms）还高 42%。这次 trace 让它从"等 Tier 1 完成再看"直接顶到第一档可拿肉。

具体动作：

1. **核实 sliding-window 那一半层是否真走短 attention**：yaml `window_size=[128, 0]` + `window_attn_skip_freq=[1,0,1,0,…]`（24 层中 12 层走 sliding）。trace 里 fmha_bwd 几乎所有 burst 长度均匀，怀疑短 path 没生效。
2. **`use_turbo_attention=false` 是否需要打开**：当前用的是 aiter ck_v3 path，turbo 那条还没 enable。开关切到 turbo + 80-iter A/B。
3. **hipBLASLt tuning cache**：`recompile` 这个后缀强烈暗示当前命中的不是缓存中最优 tile。重跑一次 tuning sweep。
4. **`hd64` 前缀**：head_dim=64 是否最优 layout，可以试 hd128 或 grouped head 变体。

预期 113.8 ms → 60–80 ms，省 35–55 ms ≈ **−4 ~ 6% step**。

### Tier 2′ — 把 lm_head + ce loss 段塞进 collective — **目标 −2 ~ 3% step**

依据：31 – 38% 那 65 ms 是稠密 GEMM-only，stream 11 完全空闲，是当前唯一一段 compute 在跑、所有副流空闲的窗口（旧 trace 里这个空窗在 optimizer tail，新 trace 在 lm_head 段）。

具体动作：

1. 把 backward 阶段第一波 bucket 的 RS 提前发，让它在 lm_head GEMM 期间发起，借这 65 ms 隐藏。Megatron `distrib_optimizer.py` 大概几十行改动。
2. 或者反过来 — 把 final layer 的 weight gather (forward 末尾) 推迟到 lm_head 期间执行。
3. 旧 Tier 2 (HIP graph 包 optimizer) 在新 trace 里 tail 只剩 ~18 ms，绝对收益变小，**降档为 ★**。

预期 65 ms → ≤ 30 ms 暴露 ≈ **−2 ~ 3% step**。

### Tier 1 — Elementwise / norm 融合 — **目标 −2 ~ 4% step（已收割大半，剩余主要是 FP8 量化链）**

**先盘点已经落地的部分**（trace 直接验证）：

| 已落地 fusion | trace 证据 | 来源 |
|---|---|---|
| Fused residual + RMSNorm (V1) | `_rmsnorm_bwd_residual_kernel` 9.24 ms × 24 | [note 17/18](./2026-04-24_gptoss_17_fused_residual_rmsnorm_impl.md) |
| 跨层 ADD#2 + 下层 input_layernorm (V2) | `_rmsnorm_fwd_residual_kernel` 2.81 ms × 24 | [note 19](./2026-04-24_gptoss_19_fused_residual_rmsnorm_v2_impl_verify.md) |
| Triton RMSNorm multi-row | `_rmsnorm_*_kernel_multi_row` 8.05 ms | [note 02](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) |
| RoPE fwd/bwd 融合 | `transformer_engine::fused_rope_*` 18.23 ms | TE 自带 |
| TE cast + transpose 融合 | `cast_transpose_optimized_kernel` 10.53 ms × 144 | TE 自带 |

→ norm 类别现在只剩 25.82 ms = **2.79% step**, 跟 [note 02 优化前的 31%](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) 比已经少一个数量级, 这一档基本榨干。

**剩下的 elementwise tax 按机会大小排：**

| 块 | 总 ms | n calls | kernel | 状态 |
|---|---:|---:|---|---|
| **FP8 量化链** | **71.6** | 600+ | `primus_turbo::unary_kernel` (bf16→fp8 cast) 31.3 + `reduce_row AbsMaxOp` 25.5 + TE `amax_kernel` 8.0 + `cast_transpose_optimized` 10.5 + 杂项 ≈ 6 | **没融合, Tier 1 真正的肉** |
| SwiGLU fwd+bwd | 26.96 | 48 | `swiglu_with_mask_{fwd,bwd}_kernel` | [note 23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md) 验证过, 没 land 生产 yaml |
| Vectorized elementwise (residual add 残余 / copy / fill) | 49.31 | 736 | `vectorized_elementwise_kernel<*>` 各种 functor | 大部分是 autograd / Megatron infra 层面的零碎 kernel, 单点小, 难 fuse |
| Reduction / arith / softmax | 17 | — | — | 已经够小, 不动 |

**两条还能拿的肉**：

1. **FP8 量化链下推到 GEMM epilogue (★★, 主目标)** — 这条 chain 每次 GEMM 都跑一遍 `bf16 input → reduce_row(absmax) → compute_scale → unary cast(bf16→fp8)`, 144 次 cast + 144 次 amax = 600+ kernel launches 占了 7.7% step。 hipBLASLt 在 ROCm 7 后期版本支持 epilogue fuse `OUT_OF_EPILOGUE = ACT * SCALE → FP8`, 把 amax 和 scale 收到上一个 GEMM 出口, 把 cast 收到下一个 GEMM 入口, 理论可以把这 71 ms 砍到 < 20 ms。**预期 −5%**。
   - 行动：dump 当前 hipBLASLt epilogue 选项, 对照 [TE FP8 recipe](https://docs.nvidia.com/deeplearning/transformer-engine/) 看 primus_turbo 这条 wrapper 是否需要重写。
2. **Tier 1B SwiGLU 去 cat 落地 (★, 次目标)** — [note 23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md) 验证完没接进 yaml, swiglu bwd 16.37 ms 是大头, 单点 ~−1.7%。
   - 行动：把 [note 23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md) 的 patch 接到 `use_turbo_fused_act_with_probs` 路径下, 80-iter A/B。

**跳过**：vectorized elementwise 这 49 ms 是 PyTorch + Megatron + autograd cleanup 的固定 overhead, 单点都 < 7 ms 且 736 个 launch 散布全 step, 没有低成本统一融合方案, **不优先**。

总预期 Tier 1 残值：71 + 27 = 98 ms 还可削, 实际可拿 **40 – 50 ms ≈ −4 ~ 5% step**（vs 之前误估的 3–5%, 量级一致但来源完全不一样）。

### Tier 3 — 残余 comm 旋钮 — **⛔ 不再追**

RCCL 真实暴露 25.3 ms (2.7%)，比旧 trace 的 46 ms (4.1%) 又少了一半。Amdahl 上限 < 1%，按 [note 14](./2026-04-23_gptoss_14_grad_sync_overlap_hsdp_negative.md) 既定结论关闭这条线。

### Tier X — Grouped GEMM 多流形态恢复 — **★★，需先确认现状**

旧 trace 4 条 grouped-GEMM 副流让 GEMM 总量 941 ms 几乎完全藏在 wall 之外；本次只剩单流 308 ms。如果是 yaml/env 退化导致并行关掉，把它打开**理论可省到 100 ms 级别**（见 §3 TODO）。

## 7. 这条 trace 是不是 MLPerf-legal？

从启动 log 抓出来的关键行：

```
train_iters ........... 1200000  ✓
lr_decay_iters ........ 1199872  ✓
lr_warmup_iters ....... 128
lr .................... 0.0008
adam_beta1/beta2 ...... 0.9 / 0.95
weight_decay .......... 0.1
seq_length ............ 8192
```

schedule 全部对齐 v6.0 表（[note 27](./2026-04-28_gptoss_27_mlperf_legal_baseline.md) §4），**这次 run 是合法 schedule**。但是：

- 这次 profile run 只跑到 ~120 iter 就停（`exit_interval` 不在 args dump 里看到），**不是 E2E**；
- 924 ms vs note 27 的 9,234 s / 7,296 iter = **1,265 ms/iter** 差太多 —— 不能直接外推 E2E 9,234 × 924/1,196 = 7,135 s，因为：
  1. note 27 那台机器跟这台不是同一机；
  2. 924 ms 是稳态 mid-run，1,265 ms 含 eval + ckpt + warmup 摊销；
  3. 旧 cluster 4-28 baseline 的稳态 ms/iter 是 **1,196 ms** ([note 27 §1](./2026-04-28_gptoss_27_mlperf_legal_baseline.md) 表)，这个跟 924 ms 才是可比量。

→ **新机器稳态比旧机快 −22.7% / step**。在确认 schedule 合规 + 跑过一次 E2E 之前，**这台机器的数字不能拿来当新 reference**；下一步要做的 SOP：

1. 在 mi355-gpu-26 上启动一次 stock-legal 完整 run（按 [note 27 §5](./2026-04-28_gptoss_27_mlperf_legal_baseline.md) SOP），看 E2E 是否真正落到 7.1k–7.3k s 区间；
2. 若是，更新 README.md 的"当前最佳 E2E"行；
3. 把 §6 的 Tier 0 / Tier 4 / Tier 2′ 改动叠在新机器上做 80-iter A/B。

## 8. Top-25 kernel 名单（rank 2，#101）

| ms | % step | kernel |
|---:|---:|---|
| 230.53 | 24.9% | `ncclDevKernel_Generic_1` (RCCL 打包集合通信，89% hide) |
| 188.77 | 20.4% | `_grouped_fp8_persistent_gemm_kernel` (MoE 前向 GEMM) |
| 119.51 | 12.9% | `_grouped_variable_k_gemm_kernel` (MoE 反向 GEMM) |
| 113.82 | 12.3% | `aiter::fmha_bwd_hd64_bf16_causal_a16_rtz_recompile` |
| 64.09 | 6.9% | `Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_…` (hipBLASLt FP8 dense) |
| 45.91 | 5.0% | `ck_tile FmhaFwdKernel` (注意力前向) |
| 31.27 | 3.4% | `primus_turbo unary cast bf16→fp8_e4m3` |
| 25.52 | 2.8% | `primus_turbo reduce_row AbsMaxOp` (FP8 amax 统计) |
| 21.78 | 2.4% | `ck_tile FmhaBwdKernel` (注意力反向，小变体) |
| 20.26 | 2.2% | `vectorized_elementwise_kernel<8, AUnaryFunctor<bf16, …>>` |
| 18.42 | 2.0% | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_…` (bf16 稠密) |
| 17.70 | 1.9% | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_…` (bf16 稠密) |
| 16.37 | 1.8% | `swiglu_with_mask_bwd_kernel` |
| 16.24 | 1.8% | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_…` (bf16 稠密) |
| 16.13 | 1.7% | `multi_tensor_apply_kernel<…AdamFunctor<float, float, …>>` |
| 10.59 | 1.1% | `swiglu_with_mask_fwd_kernel` |
| 10.53 | 1.1% | `cast_transpose_optimized_kernel<bf16, fp8_e4m3>` |
| 9.58 | 1.0% | `transformer_engine::fused_rope_forward_kernel<bf16>` |
| 9.24 | 1.0% | `_rmsnorm_bwd_residual_kernel` |
| 8.65 | 0.9% | `transformer_engine::fused_rope_backward_kernel<bf16>` |
| 8.57 | 0.9% | `_unpermute_kernel` |
| 8.08 | 0.9% | `_permute_kernel` |
| 8.01 | 0.9% | `transformer_engine amax_kernel<16, true, bf16>` |
| 7.49 | 0.8% | `at::native::reduce_kernel<128, 4, sum>` |
| 6.83 | 0.7% | `vectorized_elementwise_kernel<8, CUDAFunctor_add<bf16>>` |

观察:

- **前 4 个 kernel = 64.5% step**，单点优化空间集中；
- `recompile` 后缀的 fmha_bwd 强烈暗示 hipBLASLt cache 没命中最优 tile；
- swiglu bwd / fwd 加起来 27 ms (Tier 1B 的目标)；
- fused_rope fwd/bwd 18 ms 已经是融合后的状态，没必要再动。

## 9. 文件

```
/home/xiaompen/mlperf-training/
├── output/amd/root/gpt_oss_20b/
│   ├── tensorboard/
│   │   └── primus-megatron-exp[gpt_oss_20b]-rank[*].1778136*.pt.trace.json   ← 8 ranks × ~250 MB
│   └── logs/pre_trainer/
│       ├── rank-0/{info,debug}.log
│       └── rank-7/{info,debug}.log
└── .cursor/skills/gpu-trace-analysis/
    ├── SKILL.md
    ├── scripts/full_breakdown.py                  ← 解析器，AMD + NVIDIA 通吃
    └── templates/{single-gpu,comparison}.canvas.tsx

~/.cursor/projects/home-xiaompen-mlperf-training/canvases/
├── gpt-oss-20b-mi355x-baseline.canvas.tsx        ← 英文
└── gpt-oss-20b-mi355x-baseline-zh.canvas.tsx     ← 中文（开侧栏推荐这份）
```

## 10. 下一步（按 ROI）

| 优先级 | 动作 | 目标 | 备注 |
|---|---|---|---|
| ★★★ | 在新机器跑一次完整 stock-legal E2E，确定真实 RESULT 秒数 | 验 §7 | ~7-8 h |
| ★★★ | 核实 grouped GEMM 多流是否退化（dump `use_turbo_grouped_mlp` / `moe_grouped_gemm` 实际 truthy 值） | 见 §3 TODO | 30 min |
| ★★★ | Grouped GEMM dgrad fixed-K 路径 + W2 hipBLASLt tile sweep + router 强制均衡 A/B | −5~8% step | 见 §6 Tier 0, §5.7, 2-3 d |
| ★★★ | FMHA bwd 调优 sweep：sliding-window 路径核实 + hipBLASLt tuning cache 重跑 + `use_turbo_attention=true` A/B | −4~6% step | 见 §6 Tier 4，2-3 d |
| ★★ | lm_head + ce loss 段（31–38%）塞 RS，把第一波 bucket gradient reduce 提前发起 | −2~3% step | 见 §6 Tier 2′ |
| ★★ | FP8 量化链 (cast + amax + scale) 下推到 hipBLASLt epilogue | −5% step | 见 §6 Tier 1, 主目标 71.6 ms |
| ★ | SwiGLU bwd 去 cat ([note 23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md)) 接到 yaml 落地 | −1.7% step | patch 早写好了, 没 land |
| ★ | HIP graph 包 optimizer tail（Tier 2 旧版） | < 2% | tail 只剩 18 ms，性价比下降 |
| ⛔ | 继续打 DDP / NCCL 旋钮 | < 1% | 暴露已只剩 25 ms |

## 11. 现实的 Amdahl 上限（新 baseline）

| 来源 | 可拿上限 | 备注 |
|---|---:|---|
| Tier 0 grouped GEMM 单 kernel 调优 | ~6% | dgrad fixed-K + W2 tile sweep + router 均衡 (§5.7) |
| Tier 4 FMHA bwd | ~5% | 单点 113 ms 是最大杆 |
| Tier 1A FP8 量化链下推 epilogue | ~5% | 71.6 ms cast+amax+scale 链, V1/V2 已收, 这是新肉 |
| Tier 2′ lm_head 段塞 RS | ~2.5% | 65 ms 空窗 |
| Tier 1B SwiGLU bwd 去 cat | ~1.7% | [note 23](./2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md) patch 没 land |
| Tier X grouped GEMM 多流恢复 | ~5–8% | **若退化属实，且与 Tier 0 部分覆盖** |
| **保守可达 (Tier 0+4+2′+1)** | **~15–17%** | 924 → 770-790 ms / step |
| **乐观可达 (含 Tier X, 多流 ∩ Tier 0 仅 +50%)** | **~20–24%** | 924 → 700-740 ms / step |

把目标 step 锚到 **≈ 770 ms / 32 sample = 24 ms/sample** 当作下个里程碑（vs 当前 28.9 ms/sample）。E2E 验证后再回写 README.md 的"稳态 step / TFLOP"行。
