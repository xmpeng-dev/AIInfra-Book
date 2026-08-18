# Pretraining Large Language Models with MXFP4 on Native FP4 Hardware

> [arXiv 2605.09825](https://arxiv.org/abs/2605.09825)（**v4, 2026-08-12**；v1 2026-05-11，cs.LG/cs.AI）· 未见代码公开
> 单位：**Penn State + AMD**。一作 Musa Cim（`mtc5693@psu.edu`）与导师 Mahmut Taylan Kandemir（`mtk2@psu.edu`）在 PSU；另 5 位在 AMD（Sarthak Arora, Poovaiah Palangappa, Miro Hodak, Ravi Dwivedula, Meena Arunachalam，均 `@amd.com`）
> 硬件：**AMD Instinct MI355X 原生 FP4 tensor 支持**（作者称是 MI355X 级硬件上首次完整 MXFP4 预训练演示）· 软件：**ROCm Transformer Engine**，替换 transformer linear 层的 GEMM kernel
> 口径：Llama 3.1-8B，**MLPerf 的 C4 预训练设置**，收敛判据 = validation perplexity ≤ 3.3，指标 = 到点所需 token 数相对 FP8 基线的增量
> **范围警告：dense 模型，无 MoE / 无 grouped GEMM 实验。**

## TL;DR

**一句话：FP4 训练里真正脆的那一段是 Wgrad，而救它的不是"加随机性"，是"加确定性旋转"。**

作者在 MI355X 上按 Fprop → +Dgrad → +Wgrad 逐段打开 MXFP4，其他因素全部固定，用"到 ppl 3.3 需要多少 token"当唯一标尺。结果非常干净：

| 打开范围 | token 开销 vs FP8 |
|---|---|
| Fprop | 8–9% |
| Fprop + Dgrad | 10–11% |
| **Fprop + Dgrad + Wgrad** | **26–27%** |

然后测三种稳定化手段，**同一把梯子上做对照**：随机舍入（SR）和**随机** Hadamard 在全流水下**直接不收敛**；**确定性** Hadamard 把开销打回 **8–9%**，等于把 Wgrad 的代价完全消掉。

**最该记住的三件事**：

1. **Wgrad 是唯一的悬崖**：8–9% → 10–11% 是温和累加，一开 Wgrad 直接跳到 26–27%。所以 FP4 推进顺序应该是 Fprop → Dgrad 放开、Wgrad 单独治。
2. **FP4 的收益是被稳定性闸住的**：MXFP4 + H16 单步吞吐 **+20%**，扣掉 8–9% token 开销后**端到端只剩 +9–10%**；不治 Wgrad 的话 26–27% 的开销把 20% 吃干。**报 FP4 收益必须报到端到端这一层**，否则就是虚的。
3. **随机化在这里是负优化，且旋转尺寸这个变量已被排除**：确定性 H16 与 H32 **都能救**（都是 8–9%），随机 H16 不行。所以变量是随机化本身，不是矩阵尺寸 —— 这一条直接顶在 NVIDIA TE 的默认配置上（见 §6.2）。

## 1. Problem

FP4 训练的公认难点是激活/梯度 outlier 放大量化误差。但作者问的是一个更具体、也更工程的问题：

> 前向激活与激活梯度都稳的时候，**全流水 FP4 为什么还是会发散**？

以及配套的落地问题：能不能把 transformer linear 层的 FP8 GEMM 直接换成 MXFP4 GEMM 而不掉训练质量。

**方法论上的取舍**：此前 FP4 工作大量依赖**软件模拟**（论文点了 [3;4;8]），既有额外开销、又会掩盖真实的性能特征与稳定性行为。本文全部跑在 MI355X 的原生 FP4 tensor 支持上，所以"稳定性"和"吞吐"两件事可以在同一套硬件语义下一起测 —— 这是它相对同类工作最硬的差异点。

**MXFP4 的量化形式**（Appendix A.1）：按 32 元素为一块共享指数，`x_MX = Q_FP4(x_B / 2^E_shared) × 2^E_shared`，其中 `E_shared` 取块内最大指数，`Q_FP4` 是就近舍入到可表示的 4-bit 浮点值。

## 2. Contribution

- 一把**受控的逐段梯子**（Fprop / +Dgrad / +Wgrad，其余固定），把"FP4 会发散"定位到具体的 GEMM 通路。
- 四个假设 H1–H4 的验证：H1 Wgrad 主导退化；H2 随机方法只是加噪声、不治 outlier 驱动的 micro-scaling 误差；H3 确定性 Hadamard 降低 outlier 进入 MXFP4 块的影响从而稳住全流水；H4 多打开 GEMM 通路提升单步吞吐，且在 token 开销小时能转化为端到端加速。
- 在原生 FP4 硬件上给出端到端效率账（单步 / token / 端到端三层分开报）。

## 3. Method

### 3.1 Hadamard 旋转注入在 GEMM 之前，数学上完全抵消

`H` 是 ±1 的正交矩阵，`HH^T = I`。旋转可以理解为在特征空间里**把集中的能量摊到更多维度上**，从而改变进入量化器的分布，而不改变实际计算。三条通路的抵消证明在 Appendix C：

| 通路 | 原式 | 旋转后 | 结果 |
|---|---|---|---|
| Fprop | `Y = XWᵀ` | `X̃ = XH`, `W̃ = WH` | `X̃W̃ᵀ = X(HHᵀ)Wᵀ = XWᵀ` |
| Dgrad | `∇X = ∇Y·W` | `G̃ = (∇Y)H`, `W̃_rot = WᵀH` | `= (∇Y)W` |
| Wgrad | `∇W = (∇Y)ᵀX` | `G̃_T = (∇Y)ᵀH`, `X̃_T = XᵀH` | `= (∇Y)ᵀX` |

即**旋转是免费的语义**（只改数值分布，不改算术），代价只在 kernel 吞吐上。

### 3.2 确定性 vs 随机

作者明确写（Appendix A.2）：随机 Hadamard（随机符号翻转）虽被广泛使用，但**在这个设定下随机符号会损害收敛稳定性**，因此改用确定性变换。实现上用 tiled layout，默认选 **H16** —— 理由是目标架构上 kernel 吞吐更高，且经验上足以维持稳定收敛。

### 3.3 量化布局（Appendix D，对我们写 kernel 有用）

| 策略 | 形态 | 评价 |
|---|---|---|
| **2D block** | 32×32 区域，per-block E8M0 scale | 计算高效、**transpose 友好**、可硬件向量化，精度也好 |
| **1D row-wise** | 1×32，per-token E8M0 | 契合 transformer 的 token-wise 处理，**适合激活** |
| **1D column-wise** | 32×1，per-C_out | 权重可拿到细粒度 per-channel scale，但**转置 + 访存模式差，代价高** |

## 4. Experiments

### 4.1 逐段梯子 + 稳定化手段（Table 1）

token 开销 = 相对 FP8 基线、到 val ppl 3.3 所需 token 的相对增量。"Does Not Converge" = 发散或延长训练也到不了 3.3。

| 稳定化 | Hadamard | 打开 MXFP4 的通路 | token 开销 |
|---|---|---|---|
| FP8 基线 | – | 无（全 FP8） | 0% |
| 无 | – | Fprop | 8–9% |
| 无 | – | Fprop + Dgrad | 10–11% |
| 无 | – | **Fprop + Dgrad + Wgrad** | **26–27%** |
| 随机舍入 | – | Fprop / +Dgrad | 8–9% / 10–11% |
| 随机舍入 | – | **全流水** | **Does Not Converge** |
| 随机 Hadamard | H16 | Fprop / +Dgrad | 8–9% / 10–11% |
| 随机 Hadamard | H16 | **全流水** | **Does Not Converge** |
| **确定性 Hadamard** | **H32** | 全流水 | **8–9%** |
| **确定性 Hadamard** | **H16** | 全流水 | **8–9%** |

kernel 吞吐：**H16 比 H32 快 8%**（1.08× vs 1.00×）——所以尺寸小的那个既更快又够用。

> **这张表的对照设计是干净的**：随机 H16 与确定性 H16 同尺寸，一个不收敛一个 8–9%，所以"随机化本身有害"这个结论不被旋转尺寸混淆。我一开始怀疑的正是这个混淆（随机用 H16、确定性常被引成 H32，而 MXFP4 块正好是 32），论文自己把两个尺寸都跑了，坑是堵住的。

### 4.2 端到端效率（Table 2，MXFP4 + H16 vs FP8）

| 指标 | 结果 |
|---|---|
| 单步吞吐 | **+20%** |
| 到点 token 开销 | +8–9% |
| **端到端加速** | **+9–10%** |

同一套 MLPerf C4 设置、同 node/batch/训练配置下归一化。收益来源除了算力，还有**激活/权重变小带来的带宽压力下降**（作者点明大 batch 训练里带宽是关键瓶颈）。

### 4.3 收敛曲线（Figure 1/2）

MXFP4 + 确定性 H16 的 val ppl 曲线**紧贴 FP8**；无稳定化的全流水 MXFP4 收敛更慢且更不稳。

## 5. Limitations

**作者自陈（Key Understanding #3，很重要）**：

> MXFP4 的训练行为**高度依赖设定**；对全量预训练（MLPerf C4 + Llama 3.1-8B）有效的稳定化，**未必能迁移**到其他模型或微调方法。FP4 配方在没有更多证据前应当被视为**不通用**。

**我补的方法学风险**：

- **单模型单数据集单目标。** 只有 Llama 3.1-8B / C4 / ppl 3.3 一个点。8B 是相对小的规模，outlier 结构随规模变化是已知现象，**没有规模扫描**。
- **dense-only，没有 MoE。** 全文是 transformer linear 层，**没碰 grouped GEMM / expert GEMM**。对我们这条线是最大的缺口（见 §6.3）。
- **机理是"suggest"而非证明。** "structured micro-scaling errors along sensitive gradient paths" 是对现象的解释，论文没有给出 outlier 分布或误差谱的直接测量来支撑；H2/H3 是通过干预结果反推的。
- **与 MLPerf 博客不是独立信源。** 7 位作者里 5 位是 AMD，正文明确 follow MLPerf 的 C4 设置、用同一个 ppl 3.3 目标。它和 AMD MLPerf v6.0 官方博客本质是**同一支队伍、同一条流水线报了两次** —— 一致性有价值，但不能算交叉验证。
- **没有代码。** 复现要自己在 ROCm TE 上搭逐段开关 + Hadamard 注入。

## 6. Our take

### 6.1 可以直接用的三条

1. **推进顺序定了**：Fprop 和 Dgrad 可以激进（8–11% 开销，无需稳定化），**Wgrad 单独处理**。这条给 Primus 的 FP4 计划一个明确的分期，不必一上来就全流水。
2. **"稳定性闸"这个账法**要抄进我们自己的低精度汇报口径：单步吞吐 +20% → 端到端 +9–10%。我们在 MoE 2.0 blog 里报 FP8 grouped GEMM 的 1.2–1.6× 时也是同一个陷阱——**kernel 级加速 ≠ 端到端**，那篇已经用"system-level break-even"处理了，这里的三层分解（单步 / token / 端到端）是更清晰的模板。
3. **H16 够用且更快**（+8% kernel 吞吐），不必为了对齐 MXFP4 的 block 32 去用 H32。省一次我们自己的试错。

### 6.2 它顶在 NVIDIA 的默认配置上 —— 这是我们绕不开的第一个决策点

NVIDIA TE 的 `NVFP4BlockScaling` **默认开 stochastic rounding + Random Hadamard Transform**（见 [`./megatron-core-moe.md`](./megatron-core-moe.md)，NVIDIA 甚至把"转置 + RHT + FP4"的结果存下来给反向 Wgrad 用）。本文的结论正好相反：SR 和 RHT 在 Wgrad 量化后**都救不回来**。

**已知能排除的**：旋转尺寸（确定性 H16/H32 都行，随机 H16 不行）。
**剩下的候选解释**：量化格式本身 —— NVFP4 是 E4M3 block scale / block 16，MXFP4 是 **E8M0**（幂二 scale）/ block 32。E8M0 只能表达 2 的幂，块内最大指数一变，整块的量化格点跟着跳，这与"structured micro-scaling error"的说法是一致的。

**这仍是开放问题**，详见 [`../knowledge/systems/industry-training-optimization-2026.md`](../knowledge/systems/industry-training-optimization-2026.md) §6.3。要坐实需要一次**非 AMD 的独立复现**，或者在同一硬件上做 MXFP4 vs NVFP4 的对照。

### 6.3 对我们 MoE 线的直接缺口

**这篇是 dense 的，我们整条线是 MoE。** 三个必须自己回答的问题：

1. **Wgrad 敏感性会不会在 expert GEMM 上更糟？** MoE 的 wgrad 是**变长 K** 的 grouped GEMM，每个专家看到的 token 数不同且随路由波动。块内最大指数（E8M0 的 `E_shared`）在小 M 的专家上样本更少、更容易被单个 outlier 拖走 —— 先验是**比 dense 更脆**，但没人测过。
2. **Hadamard 旋转能不能穿过 permute / dispatch？** 我们的 FP8 路径已经要求 dispatch / permutation / expert GEMM 三者对 FP8 布局达成一致（见 `Primus/docs/tech_blogs/moe_package_2.0/moe_package.md` 低精度一节）。再插一层旋转，得确认旋转与 token permute 可交换、且 `HHᵀ=I` 的抵消不被 permute 破坏。
3. **旋转开销在 MoE 里摊得开吗？** dense 上 H16 只让 kernel 慢 0%（H16 是基准）、且换来 18 个百分点的 token 开销削减。MoE 的专家 GEMM 更小、更访存受限，旋转的相对开销会更大。
4. 行动项：**复现 Wgrad 结论 + 量 MI355X 上 Hadamard 旋转的开销**，这已经写进 [`../knowledge/systems/industry-training-optimization-2026.md`](../knowledge/systems/industry-training-optimization-2026.md) §3.1 的共同点里。

### 6.4 相关笔记

- [`./megatron-core-moe.md`](./megatron-core-moe.md) —— NVIDIA 侧的 FP4 做法（RHT + 转置存给 Wgrad），与本文结论直接冲突的那一边。
- [`../knowledge/systems/industry-training-optimization-2026.md`](../knowledge/systems/industry-training-optimization-2026.md) §2.3 / §3.1 / §6.3 —— 本文在大厂动向里的位置，以及 AMD/NVIDIA 随机性分歧的完整记录。
