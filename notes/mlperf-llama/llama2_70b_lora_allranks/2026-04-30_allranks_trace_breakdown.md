# Llama-2-70B LoRA SFT — 8 卡 profiler trace 分析（2026-04-30）

配套 canvas：
`~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/llama2-70b-lora-mi355x-allranks-trace.canvas.tsx`

## 1. Run 身份

| 字段 | 值 |
| --- | --- |
| 硬件 | 8 × MI355X（288 GiB HBM3e，totalGlobalMem = 309.22 GB） |
| 模型 | Llama-2-70B · LoRA SFT · bf16 + fp8 hybrid |
| 并行 | TP1 PP1 CP1 EP1 — 纯 DP=8（DDP + DistributedOptimizer + overlap_grad / overlap_param） |
| Batch | GBS=8 · MBS=1 · seq=8192（packed） |
| 可训参数 | 44.5M / 69.0B（0.06%） |
| Profile 窗口 | step 20–23 active，全部 8 个 rank |
| 采集设置 | `profile_ranks=[0,1,2,3,4,5,6,7]`、`record_shapes=True`、`nvtx_ranges=True` |
| Trace 文件 | `/results/torch_profiler_traces/allranks_rank{0..7}.pt.trace.json`（每个 ~605 MB） |
| Run log | `/results/profile_run_allranks.log` |

## 2. 头条数字（中位数 rank，ProfilerStep#21）

- step time = **1528.5 ms**
- TFLOP/s/GPU = **2390**
- samples/sec = **5.23**（8 GPU 合计）
- compute spread（rank 间最大−最小）= **72 ms**
- RCCL spread（rank 间最大−最小）= **76 ms**

## 3. 每 rank 分解 — 这次最关键的新发现

```
rank | step ms | compute ms | rccl ms | rccl % | compute+rccl
-----+---------+------------+---------+--------+--------------
  0  | 1528.14 |   1370.18  |  61.23  |  4.0   |    1431.41
  1  | 1528.49 |   1353.12  |  77.72  |  5.1   |    1430.84
  2  | 1528.48 |   1411.41  |  15.87  |  1.0   |    1427.28
  3  | 1528.49 |   1363.10  |  69.02  |  4.5   |    1432.12
  4  | 1528.94 |   1425.23  |   1.96  |  0.1   |    1427.19  ← straggler
  5  | 1528.48 |   1398.23  |  26.62  |  1.7   |    1424.85
  6  | 1528.49 |   1387.84  |  44.89  |  2.9   |    1432.73
  7  | 1528.77 |   1363.70  |  62.46  |  4.1   |    1426.16
```

**守恒律**：每个 rank 上 `compute_ms + rccl_ms ≈ 1430 ms`（spread 仅 7.9 ms）。
也就是说每个 rank 在 RCCL 流上花的时间**几乎全部都是在等 straggler**，
真正的 AllReduce 工作量 ≤ rank 4 的 1.96 ms。

这**直接推翻了之前单 rank trace 上得出的 "RCCL 是最大暴露 bottleneck" 的结论**。
单 rank trace 上看到 60 ms RCCL 尾部，是因为 rank 0 既不是计算最重的 rank、
也不是计算最轻的 rank。真正的瓶颈格局是：

1. **计算不平衡**（72 ms spread = step 的 4.7%）— 主要瓶颈
2. **Step 头部 idle**（每 step 95 ms — 与之前单 rank trace 一致）— 次要
3. **真正的 RCCL 带宽**（~2 ms / step）— 不是问题

## 4. 同样 batch shape 下，每 rank compute 为什么差 ~70 ms

GBS=8、MBS=1、seq=8192 packed → 每个 rank 看到的 kernel 序列与 shape 完全一致。
72 ms 的差距来源只能是下面一种或几种叠加：

- **NUMA / PCIe 拓扑**：不同 NUMA 节点上的 GPU 在 host→device staging 时
  延迟略有差异，会拖慢少量 CPU-bound op。
- **Inductor 编译不确定性**：每个 rank 的 compile worker 即便从同一份 FX
  graph 出发，也可能选出略有不同的 fused kernel。FX graph cache 我们是
  共享的，但 autograd cache 是 per-process。
- **温度 / 频率**：MI355X 每张卡的功耗各自报告，热的卡会掉 5–10% 时钟直到
  冷下来；rank↔卡 的映射在整个 3-step profile 窗口内不变，所以这种差异
  会表现为系统性偏差。
- **DataLoader 数据内容差异**：即便 `num_workers=0` + packed 序列，每个
  batch 内 token 分布的微小差异都可能让 FP8 GEMM 内部的 tile size 略变。
  在 808 ms 的 FP8 GEMM 总量上，5% 的 shape 驱动 kernel 速度差就是 40 ms。

下一步可以做的诊断：
- profile 期间用 `rocm-smi --showperflevel --showtemp -i N` 持续采样，按
  rank 把每张卡的频率/温度叠加到 step time 上比较。
- diff 8 个 `breakdown_rank{0..7}_step21.txt` 的 "Top-25 GPU kernel names"
  —— 如果 rank 4 上的 kernel 与别人完全相同但每个都慢 5%，就是热/频率
  问题；如果 kernel 集合不一样就是 inductor 的锅。

## 5. Pipeline（rank 4，straggler）

| 阶段 | bins | 时间窗 | 内容 |
| --- | --- | --- | --- |
| 头部 idle | 0–4 | 0–95 ms | data-loader + dispatcher 间隙 |
| forward | 5–36 | 95–707 ms | FP8 GEMM（~13 ms/bin）+ fmha fwd（~3 ms/bin）+ elementwise |
| fwd↔bwd | 37 | 707–726 ms | gemm 突增 3.45 ms，attn 跌到 1.19 ms |
| backward | 38–78 | 726–1510 ms | fmha bwd ~5.9 ms/bin，FP8 GEMM bwd ~0.5–0.6 ms/bin tile |
| 尾部 | 79 | 1510–1529 ms | 仅 ~2 ms RCCL — straggler 一进 collective 就立刻退出 |

如果同一份 step 看 rank 1，stream 33 会在 step 末段最后 100 ms 里被占用 77.7 ms
（在 AllReduce 里干等 rank 4 的梯度送到）。

## 6. GPU 工作分解（rank 4，ProfilerStep#21）

```
category                                     ms      % of step
FP8 GEMM (Custom_Cijk_*_F8*, hipBLASLt)    808.4   52.9
FlashAttention (aiter::fmha + ck_fused)    338.2   22.1
Elementwise / cast / dropout               123.9    8.1
TE SwiGLU / dgated_act / unary              67.0    4.4
bf16 GEMM (Cijk_Ailk_Bljk_BBS / BSS)        40.4    2.6
Fused QKV-RoPE                              23.2    1.5
RMSNorm (triton fwd + bwd)                  20.0    1.3
FP8 cast / transpose                        12.7    0.8
Reduction kernels                            8.5    0.6
MemCopy / D2D                                4.9    0.3
RCCL（仅 rank 4）                            1.94   0.1
Optimizer / softmax                          0.2    0.0
                                          ------   ----
total accounted                           1449.3   94.8
```

分析器原始输出里 `other` = 893.68 ms，其中 808.40 ms 实际是 FP8 GEMM
（`Custom_Cijk_*_F8*` 系列），剩下的 ~85 ms 是 TE 自己的若干 kernel
（silu / dgated_act / qkv_rope / cast_transpose / dropout），已经按类拆出。

## 7. Top-10 单个 kernel（rank 4）

```
   ms     %     name                                                 bucket
 363.6   23.8  Custom_Cijk_Alik_Bljk_F8B8BS_*shortname1               GEMM fwd
 294.5   19.3  Custom_Cijk_Alik_Bljk_F8BS_*shortname1                 GEMM bwd
 222.7   14.6  aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv          Attention bwd
 109.7    7.2  Custom_Cijk_Alik_Bljk_F8BS_*shortname0                 GEMM
  95.1    6.2  aiter::fmha_fwd_hd128_bf16_causal                      Attention fwd
  42.3    2.8  vectorized_elementwise_kernel<CUDAFunctor_add bf16>    Elementwise
  40.7    2.7  Custom_Cijk_Alik_Bljk_F8B8BS_*shortname0               GEMM
  34.5    2.3  transformer_engine::gated_act_kernel<silu>             Activation
  32.4    2.1  transformer_engine::dgated_act_kernel<silu>            Activation
  16.5    1.1  transformer_engine::unary_kernel<identity bf16>        Cast
```

## 8. 每 stream busy（rank 4）

```
stream  role                                      busy_ms   share
   0    Compute (GEMM/FMHA/norm/elem/activation)  1430.17   93.5%
  33    RCCL DDP grad-sync (Generic + AllGather)     1.92    0.1%
```

Stream 0 oversubscription = 93.7%。idle = 93.25 ms / step（6.1%）；
其中绝大部分都集中在 step 头部（bins 0–4）。

## 9. VRAM（rank 0，MI355X 309.22 GB cap）

```
mem-allocated-gigabytes      :  126.44
mem-max-allocated (Pmax)     :  285.84    ← peak working set
mem-max-reserved  (Rmax)     :  295.52    ← rocm-smi 上看到的就是这个
mem-alloc-retires            :       0
totalGlobalMem (cap)         :  309.22 GB
```

Reserved% = **95.6%**（TIGHT，超过 95% 阈值）
Allocated% = 92.4%
Frag% = 3.1%（很好）
Headroom = 13.7 GB

### Bucket 分解（Mode A — LoRA SFT）

```
bucket                                        GB      % of Pmax  备注
权重 (fp8 主权重 + bf16 LoRA)                69.0       24.1     70B@1B fp8 + 44.5M@2B bf16
FP8 transpose cache                         69.0       24.1     keep_fp8_transpose_cache=True 复制一份
Activations（TE selective recompute）       120.0       42.0     bs1 × 8192 × 8192 × 80 × ~2.5B
TE FP8 / amax / cuBLAS workspace            25.0        8.7     delayed-scaling 历史 + comm bufs
LoRA grads + Adam（DistOpt 分片 /8）          0.1        0.0     44.5M × 10B / 8 = 56 MB
Allocator slack（Rmax − Pmax）                9.7        3.4     fragmentation
                                           -----      -----
sum                                        292.8       102.4   ≈ Rmax 295.5（差 2.7 GB）
```

## 10. 三个发现 → 下一步实验

### 发现 1 — rank-0 trace 上的 "RCCL 瓶颈" 是假象

**之前的认知（基于 rank-0 trace）**：每个 step 尾部有 ~60 ms 暴露的 RCCL
allreduce。`bucket_size=2_000_000` 实验就是冲着这个来的（也已证实没用）。

**8 rank trace 显示**：真正的 AllReduce kernel 只跑 ~2 ms（rank 4 几乎看到
完整的 collective 时间，不需要等）。其它 rank 上 60+ ms 是**纯粹的 stream 33
上 GPU idle**，在 collective 里干等 straggler 算完 backward。把 bucket size
调小或者换 collective fusion 策略都救不回这部分时间 —— collective 本身
已经短到不能再短。

**下一步实验**：不要再动 DDP；换方向去解决 compute 不平衡。

### 发现 2 — Compute 不平衡 72 ms / step 的 4.7%

Rank 4（compute 1425 ms）始终最重，rank 1（1353 ms）始终最轻。Batch shape
和 kernel 序列完全一致，只要 808 ms 的 FP8 GEMM 块上每个 kernel 慢 5%，
就足够解释 40 ms 的差距。

**下一步实验**：
1. profile 期间持续采 `rocm-smi --showperflevel --showtemp`，把每张卡的
   实际频率/温度跟 compute_ms 关联看。
2. diff 8 份 `breakdown_rank{0..7}_step21.txt` 的 Top-25 列表 —— kernel
   集合一样但每个慢 5% → 是硬件（NUMA/热）；kernel 集合不一样 → 是
   inductor 编译不确定性。
3. 尝试 `HSA_FORCE_FINE_GRAIN_PCIE=1` 再 profile，可以判断是不是 host 端
   staging 导致的。

### 发现 3 — 头部 idle 仍然 ~95 ms / step

跟之前单 rank trace 的结论一致。考虑到 straggler rank 上 compute 已经吃掉
93.5% 的 step，把这 95 ms 干掉就把稳态 step 从 1530 ms 拉到 ~1435 ms ——
**每个 rank** 都得到 6.2% 的 step time 收益。两个便宜可做的 fix 之前
已经识别过：

- 把 8192² causal mask cache 一次复用，干掉 `aten::tril` /
  `aten::ones` / `aten::fill_` 那串 CPU op。
- 把 `aten::item()` 上的 D2H scalar 读取 gate 在
  `step % log_interval == 0` —— 也已经在 cache-hit canvas 里写过。

### Callout — VRAM headroom

Reserved 295.5 / 309.22 = 95.6%。当前 shape 稳定（0 retires，frag 3.1%）。
拿到 headroom 最便宜的杠杆是 `keep_fp8_transpose_cache=False`：可以释放
~70 GB，代价是每个 backward layer 多一次 bf16↔fp8 transpose。我们还没在
这个 recipe 上量化过 throughput 影响 —— 如果 eval / 更长 seq / 更大 MBS
要排上日程，值得专门跑一次 profile 验证。

## 11. 文件清单

| 文件 | 作用 |
| --- | --- |
| `/results/torch_profiler_traces/allranks_rank{0..7}.pt.trace.json` | 每 rank 的 Kineto trace（每个 605 MB） |
| `/results/profile_run_allranks.log` | run log，含 TFLOPS + mem-max-* |
| `notes/llama2_70b_lora_allranks/breakdown_rank4_step21.txt` | full_breakdown.py 在 rank 4 上的输出 |
| `scan_per_rank.py` | 快速汇总每 rank step / compute / RCCL 的脚本 |
| `~/.cursor/projects/home-xiaompen-mlperf-training-primus/canvases/llama2-70b-lora-mi355x-allranks-trace.canvas.tsx` | 本次分析对应的 canvas |
