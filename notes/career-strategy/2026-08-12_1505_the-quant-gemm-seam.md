# 靶子：量化↔GEMM 的格式缝——用自己的数据选出来的下一个算子

> **When**: 2026-08-12 15:05 UTC+8
> **Where**: 登录机，基于 `notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md` 的既有实测做静态分析（本次未跑 GPU）
> **Context**: 取代同日 [`2026-08-12_1428_own-the-judgment-layer.md`](./2026-08-12_1428_own-the-judgment-layer.md) 的主张。起因是对"判定层就是胶水层、换工作不加分，不如写更优的算子"的反驳——该反驳成立，但"写更优的算子"缺一个靶子，本篇用自己的数据把靶子定出来

## TL;DR

1. **判定层主张作废。** 它撞在 8-06 自己的论断上（"更聪明的 wrapper 还是 wrapper"），且绑死 Primus-Turbo → 离开 AMD 资产归零。
2. **但"写更快的 GEMM"在当前靶子上也已枯竭。** 用自己的 dense GEMM 当尺子：`mfma_tile.h` bf16 上限 1290 TF → MXFP8 上限 ~2580 TF；dW1 variable-K wgrad 实测 **2199 TF = 85%**，bf16 腿 1092/1290 **也是 85%**。之后 autotune 之外的四次尝试全部无效或高风险。剩余 <10%，而 sclk ±30% 使 <10% 不可验证。
3. **钱在两处，都不是"更快的 GEMM"**：①量化↔GEMM 的格式缝——FP8 GEMM 本体 1.84–1.96×，净收益被 colwise requant 稀释到 **1.15–1.29×**；②通信与 GEMM 等长——`fc1_dgrad_combine` 里 GEMM 2.200 ms vs XGMI PUSH 2.029 ms。
4. **新主张：删掉格式缝。** 这是写算子，且是最难的一类——它要求同时改 kernel 与改跨 op 契约，所以至今无人做。而"同时有权改 kernel 和改调用方"正是本人独有的站位。

## 背景：两个都被否决的框架

| 框架 | 否决理由 |
|---|---|
| **占判定层**（08-12 14:28） | 是更聪明的 wrapper；绑 Primus-Turbo，离职归零；与 8-06「集成层是低信用位置」自相矛盾 |
| **写更优的算子**（朴素版） | 缺靶子。同一个错误刚在 `notes/hk-attn-bwd/` 上犯过一次——以"能不能移植一个 kernel"为立项单位，找了三轮才确认无落点 |

真正的问题不是"胶水 vs 算子"，是**写哪个算子**。下面用实测把它定出来。

## 证据一：wgrad GEMM 已到 85%，继续抠不划算

不采用 AMD 标称 2.5 / 5 PFLOPS 口径（与自测的"99% peak"说法不一致，且口径争议无助于决策）。**改用自己的 dense GEMM 当尺子**：`mfma_tile.h` 在 8192×4096×7168 规则形状上 1290 TF（`notes/rocmoe/` M0），即本机 bf16 可达上限；MXFP8 矩阵峰值为 BF16 的 2×，故 MXFP8 可达上限 ≈ 2580 TF。

| kernel | 实测 | 占自测上限 | 来源 |
|---|---|---|---|
| dW1 variable-K wgrad, mxfp8（autotune 后 cfg 256,256,2,1,0） | 1.859 ms / **2199 TF** | **85%** | `mxfp8_moe_bwd_perf_summary.md` §dW1 autotune |
| dW1 variable-K wgrad, bf16 参考 | 3.743 ms / **1092 TF** | **85%** | 同上 §逐 stage |
| dW2 wgrad, mxfp8 | 1.004 ms / **2036 TF** | 79% | 同上 §dW1 autotune |

**变长 K 的 ragged wgrad 在两种精度上都达到自测规则形状上限的 85%**——这个自洽性是强信号：GEMM 本体接近做完。

autotune 之后的四次尝试（同一份 note 记录）：

| 尝试 | 结果 |
|---|---|
| VMEM prefetch 重排（提前 issue `a_g2s`/`ScaleS2R` + prologue 补 `a_next1`） | isolated GEMM 1.86→**1.93 ms (+4%)**，e2e flat，**已回滚**。原因："pipeline 已是 distance-2 双缓冲，compiler 调度已饱和" |
| `asm_mma=True` inline MFMA | FlyDSL 编译 **"not a valid operand"**；mxfp8 scaled wgrad 与 tensorwise wgrad 路径不兼容，未落地 |
| K-chunk barrier 减 sync（ATT 显示 `:1327` 占 29% stall） | 判定高风险；同族结论已在 `fc1_dgrad` 上实测——**减 sync 后 wall 不变甚至回退**，因为 barrier stall 是 epilogue 跨 wave straggler 的反映，不是同步指令开销 |
| VGPR 压到 3 waves / LDS 4→2 buffer | 历史结论：暂缓（高风险、收益不确定） |

⚠ **ATT 的 81% stall 不可读作"有 81% 可捞"**：一个 MFMA issue 接近饱和的 kernel 在 ATT 里同样显示 barrier / 等数据周期。`fc1_dgrad` 的 barrier 实验已经独立证明了这一点。

⚠ **测量约束**：本机 `rocm-smi --setperflevel high` **Not supported**，sclk 在 level 0/1 跳，绝对值摆动 ±30%；已被迫扫 8 卡取 spread 最紧的 GPU4（±0.16%）钉住。**这意味着 <10% 的收益在这台机器上无法验证**——直接否掉了"继续在 GEMM 上抠"这条路，与其技术上是否还有余量无关。

## 证据二：钱在格式缝

同一份 note 的逐 stage 表（T=8192，EP8，DSV3，load_balanced）：

| stage | GEMM 本体 vs bf16 | 净（FULL）vs bf16 | 被吃掉的部分 |
|---|---|---|---|
| `fc2_wgrad` (dW2) | **1.96×** | **1.15×** | GEMM 1.080 + requant 0.365 + quant 0.144 + meta 0.122 |
| `fc1_wgrad` (dW1) | **1.84×** | **1.29×** | GEMM 2.038 + requant 0.365 + quant 0.278 + meta 0.123 |

**FP8 的算力优势拿到了将近 2×，然后一半被 colwise requant/quant 吃掉。** 两个 wgrad 共约 1.15 ms 量化开销（dW2 占 FULL 的 34%、dW1 占 26%），且这些是**访存受限的转置量化 kernel**（`colwise_(re)quant_mxfp8_grouped_flydsl`），实测只跑到 **~35% HBM 带宽**。

已经做完的（单 kernel 内能做的都做了）：

- `rowcol_dual_quant_mxfp8_grouped_flydsl` 融合双输出一次读：0.3339 → **0.3024 ms**（−9.4%，byte-exact）。主力是 pow2 指数倒数替 fdiv（−7.8%）
- pool_x colwise 预量化挪到 forward（backward 省 ~0.36 ms）
- grouped `meta` 提到一次算好，在 fused-quant / dW1 / dW2 间共享（省 2 次 meta D2H ≈ 0.24 ms）
- 净结果：e2e fwd+bwd 14.71 → **14.117 ms**

然后撞墙，且撞墙原因被自己写了下来：

> 跌破 0.25ms 需改输出 layout，会**破坏下游 dW1 GEMM / STEP3 combine 契约**，超出本 kernel 范围

**这一句是本篇的立论点。** 剩下的量化开销不是某个 kernel 写得不够好——kernel 本身已 VALU-bound、bank-conflict-free、occupancy 非瓶颈。它是**格式契约横跨了三个 op**（量化 kernel → dW1 GEMM → STEP3 combine），所以没有任何单个 kernel 的作者有权删掉它。

## 证据三：通信与 GEMM 等长（第二处钱，本篇不作为主靶）

`fc1_dgrad_combine`（3.03 ms，反向单项最大，仅 1.26× vs bf16）的三腿拆解：

| 模式 | ms |
|---|---|
| GEMM_ONLY | 2.200 |
| PUSH_ONLY（XGMI combine） | **2.029** |
| NO_REDUCE（GEMM‖PUSH 实测 overlap） | 2.522 |
| serial 上界 | 4.229 |
| 理想 max(G,P) | 2.200 |

overlap 已省 1.71 ms（serial 的 40%），效率 ~87%，剩 **0.32 ms** gap。**GEMM 侧再快也压不动这个 stage，因为被 XGMI 那条腿钉住。** 这是 8-06 四条缝里的 collective↔compute 缝，已有三代 super-kernel 的积累，但本篇不作为主靶——因为格式缝的收益更大（~0.5–0.9 ms vs 0.32 ms）、风险更低、且不依赖多卡环境。

## 新主张

> **写一个让量化往返消失的算子**：GEMM 直接吃上游的 layout / 直接吐下游要的 scale 格式，把中间那趟转置量化整个删掉。

这是写算子，且是最难的一类：**它要求同时改 kernel 与改跨 op 契约。** 这也解释了为什么至今无人做——需要同时对 kernel 和调用方有权，而这个组合权限只有一人有（Primus-LM #1 26.8% / Primus-Turbo #2 21.8%，见 14:28 那篇 §核查一）。

**与 8-06 的关系**：这才是"消解边界"的正确落法。8-06 原文说的是"一族带明确 overlap 契约的可组合原语"，14:28 那篇把它读窄成了 dispatcher，是误读。删掉缝隙靠融合 kernel，不靠加一层派发。

## 为什么这个换工作加分

一句话可交付：

> 8×MI355X 上 DeepSeek-V3 MoE 的 MXFP8 反向，FP8 净收益从 1.29× 提到 X×——办法是把 per-group scaling 的量化往返融进 GEMM 的 prologue/epilogue，删掉跨 op 的格式契约。

可迁移的资产是**低精度训练 kernel 设计中"scaling / layout 契约跨越 op 边界"这一类问题**。不是 AMD 特有：Blackwell 的 MX 格式有同构问题，DeepSeek 的 DeepGEMM 存在的理由正是细粒度 FP8 scaling 的开销（已在 `notes/rocmoe/2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md` 逐行读过 `sm100_fp8_fp4_mega_moe.cuh`）。2026 年训练正在整体搬向 MX 格式，这个技能在变贵。

三个候选的对照：

| 候选 | 剩余空间 | 可迁移性 | 拥挤度 |
|---|---|---|---|
| 更快的 wgrad GEMM | **85% 已达，剩 <10% 且本机测不出** | 高 | CK / FlyDSL / Gluon 都在 |
| **量化↔GEMM 格式融合** | **~0.5–0.9 ms；1.29× → 目标 1.6–1.8×** | **高（MX 格式是行业方向）** | **无人做（需跨 op 权限）** |
| 判定层 | 大 | **低（绑 Turbo，离职归零）** | 无人做 |

## HipKittens 在这里的位置

降级为**备用手段，不是立项理由**。

理由：HK 相对 FlyDSL 唯一独有的能力是 register pinning（见 `notes/hk-attn-bwd/2026-08-12_1354_turbo_attention_ground_truth.md` §更正二），而它对症的瓶颈是寄存器压力。dW1 wgrad 的 ATT 确实显示 **VGPR 248/512 → 2 waves/SIMD（VGPR-bound，非 LDS-bound；3 waves 需 combined VGPR ≤ 170）**——对症。但既然 GEMM 本体只剩 <10%，现在用它是"没有病的药"。

**取出来用的时机**：融合 kernel 因为要在 GEMM 的 epilogue 里多做量化数学（amax 规约、cvt、scale 打包）而撞上寄存器墙时。那时它是解药。这个时机很可能会到——epilogue 加量化必然抬 VGPR，而 kernel 已经在 2 waves/SIMD 的边缘。

## 分期

| 阶段 | 内容 | 验收门槛 |
|---|---|---|
| **S0** | **可判性**：在钉死的 GPU4 上把目标 stage 重复 30+ 次，出 run-to-run 分布，定"最小可判差异"。产出物是一个数字 + 环境指纹入 CSV | 能声明"本平台 X% 以下不可判"。**这是所有后续 claim 的许可证**，不是独立主张 |
| **S1** | **契约测绘**：把量化 kernel → dW1/dW2 GEMM → STEP3 combine 之间的 layout / scale 格式契约画清楚，标出哪一段是"为了迁就下游"而存在的 | 一张契约图 + 每段的字节流量与耗时归属 |
| **S2** | **单点验证**：选开销最大的一段（dW2 requant 0.365 ms 或 dW1 requant 0.365 ms），做一个"GEMM prologue 直接吃未量化输入"或"epilogue 直接吐目标格式"的原型 | byte-exact 或梯度 SNR ≥ 15 dB；净收益 > S0 定的可判阈值 |
| **S3** | **推广 + 契约固化**：把成功的形态推到另外两段，把新契约写进 Turbo 的算子接口（这一步需要框架侧权限，是站位的兑现） | 反向净 FP8 收益 1.29× → 目标 ≥ 1.6×；e2e fwd+bwd 从 14.117 ms 下探 |
| **S4** | 若 epilogue 撞寄存器墙 → 取 HK register pinning | VGPR ≤ 170 / 3 waves per SIMD |

S0 是天级；S1 是纯阅读 + 归属分析，不跑 kernel；S2 是第一个真实 claim。

## 可证伪条件

- **S2 发现量化无法融进 GEMM**：例如 colwise scale 需要全列 amax，而 GEMM 的 tile 切分让单个 WG 看不到整列 → 融合在数学上不成立。这是最主要的技术风险，S1 的契约测绘就是为了尽早暴露它。若成立，退路是"改输出 layout 让下游不需要转置"，那仍然是跨 op 契约工作，但收益要重估。
- **精度不达标**：融合后为省一趟往返而降低 scale 粒度 → 梯度 SNR 跌破 15 dB。现有基线 dx≈20–22 / dW1≈19.4 / dW2≈19.7 dB，余量不大。
- **S0 结论是"<15% 不可判"**：则 0.5–0.9 ms（占反向 8.734 ms 的 6–10%）也落在噪声里，整条线的可验证性成疑。**这是最致命的风险**，也是 S0 必须最先做的原因。
- **上游改动被拒**：新契约要改 Turbo 的算子接口，若被 owner 拒绝则收益无法落进产品，只能作为论文/报告。

## 相关文件

- [`2026-08-12_1428_own-the-judgment-layer.md`](./2026-08-12_1428_own-the-judgment-layer.md) — 被本篇取代的主张；其 §核查一（站位数据）、§核查二（AITER ASM 层唯一 backward 是 `fmha_v3_bwd`）、§核查三（Turbo 现状）仍成立
- [`2026-08-06_primus-positioning-boundary-dissolution.md`](./2026-08-06_primus-positioning-boundary-dissolution.md) — 消解边界的母论点；本篇是它的正确落法
- `notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md` — **本篇全部数字的来源**：逐 stage 表、dW1 autotune + ATT、dual-quant loop、fc1_dgrad_combine 三腿拆解
- `notes/MegaMoeFlydsl/2026-07-24_1432_mega_fp8_fc1_dgrad_wgrad_combine_overlap_plan.md` — dgrad/wgrad/combine overlap 既有规划
- `notes/rocmoe/2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md` — DeepGEMM 的 fp8 细粒度 scaling 处理，NVIDIA 侧同构问题的参照
- [`notes/hk-attn-bwd/2026-08-12_1354_turbo_attention_ground_truth.md`](../hk-attn-bwd/2026-08-12_1354_turbo_attention_ground_truth.md) — HK 唯一独有能力是 register pinning 的论证
- `knowledge/kernels/fp8-expert-gemm.md` · `knowledge/kernels/memory-access-patterns.md` — 相关既有知识
