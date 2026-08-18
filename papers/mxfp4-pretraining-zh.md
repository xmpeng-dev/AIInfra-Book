# 在原生 FP4 硬件上用 MXFP4 预训练大语言模型

> 全文中译。原文 [arXiv 2605.09825](https://arxiv.org/abs/2605.09825)（**v4，2026-08-12**；v1 2026-05-11，cs.LG / cs.AI）
> 原文许可：**CC BY 4.0**（[署名 4.0 国际](https://creativecommons.org/licenses/by/4.0/)），允许在署名前提下翻译与再分发。
> 技术解读、方法学质疑与对我们自己工作的延伸讨论见 [`mxfp4-pretraining.md`](./mxfp4-pretraining.md)。
>
> **翻译体例**：正文、表格、图注、公式逐段全译；专有名词（Fprop / Dgrad / Wgrad、micro-scaling、stochastic rounding、Hadamard、block、scale、token overhead 等）保留英文并在首次出现处给出中文；公式、格式名、参考文献保持原样；译者补注以「译注」标出。原文未公开代码。

**作者**：Musa Cim†、Sarthak Arora△、Poovaiah Palangappa△、Miro Hodak△、Ravi Dwivedula△、Meena Arunachalam△、Mahmut Taylan Kandemir†

† 宾州州立大学（The Pennsylvania State University）　△ Advanced Micro Devices, Inc.

`{mtc5693, mtk2}@psu.edu`　`{Sarthak.Arora, Poovaiah.Palangappa, Miro.Hodak, Ravi.Dwivedula, Meena.Arunachalam}@amd.com`

---

## 摘要

为什么大语言模型的全流水 FP4 训练常常发散——即便前向激活与激活梯度都保持稳定？我们通过一项针对 transformer 训练中 MXFP4 量化的**受控研究**来回答这个问题：在固定其他所有因素的前提下，逐步在前向传播（Fprop）、激活梯度（Dgrad）与权重梯度（Wgrad）上打开 FP4。在 C4 数据集上对 Llama 3.1–8B 做完整预训练时，我们观察到**量化 Wgrad 是收敛退化的首要驱动因素**，而仅在 Fprop 与 Dgrad 上使用 FP4 只带来温和的额外 token 需求。为解释这一行为，我们在受控实验设定下同时评估了结构化干预与随机化干预。我们发现，一旦 Wgrad 被量化，stochastic rounding（随机舍入）与 randomized Hadamard rotation（随机 Hadamard 旋转）**都无法**稳定训练，而 deterministic Hadamard rotation（确定性 Hadamard 旋转）能够**一致地**恢复稳定的优化过程。这些结果表明，FP4 训练的不稳定性源于敏感梯度路径上**结构化的 micro-scaling 误差**，而非随机性不足。我们的实验运行在具备**原生 MXFP4 支持**的 AMD Instinct MI355X GPU 上，从而无需依赖软件模拟即可对这些效应做受控考察。

---

## 1 引言

大语言模型（LLM）的快速增长——包括 Llama 3 [5]——加剧了对更高效、更贴合硬件的训练算法的需求。标准的半精度格式对显存带宽与计算资源提出了相当高的要求，而 FP8 近来已成为一种实用的低精度替代方案。然而，用 4-bit 格式训练（例如 microscaling 系列格式、NVFP4、MXFP4）仍是一个显著的挑战。4-bit 训练的一个关键困难在于：**激活与梯度中的 outlier（离群值）会放大量化误差，并可能破坏优化的稳定性**。当激活 outlier 被粗粒度地量化时，它们可能主导整个量化范围，严重降低低位宽浮点量化的质量——这一现象在早期 FP4 研究中已被观察到 [6]。文献 [10] 探索了通过**模块级混合精度**与**分阶段调度**来缓解 FP4 量化噪声的预训练配方；文献 [1] 则考察了用 NVFP4 这类厂商特定格式做预训练。

在本工作中，我们要问的是：**能否把 transformer 线性层中的 FP8 GEMM kernel 替换为 MXFP4 GEMM kernel，同时保持训练质量**；并且我们用受控实验来识别——哪些训练分量（Fprop、Dgrad、Wgrad）最强地决定了稳定性，以及哪些稳定化手段真正有效。为回答这个问题，我们在固定其他所有因素的前提下，逐步在 Fprop、Dgrad、Wgrad 上打开 MXFP4，并用这条受控的「阶梯」去**验证或伪证**候选的稳定化手段。

我们的首要动机是效率：有了原生 FP4 tensor core 支持，MXFP4 GEMM 相比 FP8 能提供更高吞吐、并降低显存带宽压力。然而，用 MXFP4 替换 FP8 可能会**增加到收敛所需的 token 数、甚至破坏训练稳定性**。除效率之外，理解低精度训练**在哪里失效、以及为什么失效**，还能为大型 transformer 在极端量化下的优化动力学提供可操作的洞见。这里我们用受控的、逐段打开的 MXFP4 来定位 FP4 训练的断点，并把 Hadamard 旋转当作一种**结构化探针**，用以解释是哪一类误差源在驱动不稳定性。

我们报告以下发现：

- **Where（在哪里）**：逐步打开 MXFP4 揭示出，收敛不稳定性**由 Wgrad 量化主导**，而仅在 Fprop 与 Dgrad 上使用 MXFP4 相对而言仍然稳定。
- **Why（为什么）**：当 Wgrad 被量化时，**随机化干预无法**稳定训练——这表明额外加入的随机性是在**放大**有效量化误差，而不是在缓解 outlier 效应。
- **How（怎么办）**：在随机化变体失效的同一受控设定下，**确定性 Hadamard 旋转能够恢复全流水 MXFP4 的稳定优化**，因而它可以作为一种实用探针，用于隔离出 FP4 训练中真正起稳定作用的结构。

> 译注：原文在此处还列出了附录 D「1D 与 2D 量化策略」作为附加内容，见本文 [附录 D](#附录-d1d-与-2d-量化策略)。

---

## 2 实验设置

### 2.1 原生 FP4 硬件支持

此前关于 FP4 的工作 [3, 4, 8] 往往依赖**软件模拟**，这既引入了额外开销，也会同时掩盖性能特征与训练稳定性两方面的真实表现。我们的实验则使用 **AMD Instinct MI355X GPU 上的原生 FP4 tensor 支持**，从而在不做模拟的情况下完成 MXFP4 计算。据我们所知，这是**在 MI355X 级硬件上首次完整演示 MXFP4 预训练** [2]。

### 2.2 模型与任务

我们遵循 **MLPerf 的 LLM 预训练设置**，在 C4 数据集上训练 **Llama 3.1–8B**。我们把收敛定义为**达到 validation perplexity 3.3**。对每种方法，我们报告首次达到 perplexity ≤ 3.3 所需的训练 token 数；并以相对 FP8 基线在该阈值处的相对增量来报告 **token overhead（token 开销）**。

---

## 3 评测结果

### 3.1 假设

我们的实验检验关于全流水 MXFP4 预训练稳定性的四个假设：

- **H1**：逐步量化会揭示出，收敛退化**由量化 Wgrad 主导**，相对于 Fprop 或 Dgrad 而言。
- **H2**：随机化方法（stochastic rounding 或 randomized Hadamard）**收益有限**，因为它们只是加入噪声，并未触及 MXFP4 micro-scaling 中由 outlier 驱动的根本误差。
- **H3**：确定性 Hadamard 旋转与「降低 outlier 进入 MXFP4 block 的影响」这一机制相符，从而能稳定全流水的优化过程。
- **H4**：在更多 GEMM 通路上启用 MXFP4（即逐段打开 MXFP4）会**提升训练步吞吐**；并且当 token overhead 保持较小时，这会在 MLPerf 目标处转化为**端到端加速**。

### 3.2 逐段打开 MXFP4

为了隔离出 MXFP4 训练中究竟哪些分量在驱动收敛退化，我们逐步在训练流水中打开 MXFP4，并测量到达 validation perplexity 3.3 所需的 token overhead（表 1）。

相对 FP8：**只对 Fprop 用 MXFP4 带来 8–9% 的 token overhead**；扩展到 **Fprop + Dgrad 后为相近的 10–11%**。相比之下，**对 Fprop + Dgrad + Wgrad 都启用 MXFP4 会把 overhead 推高到 26–27%**，这表明 **Wgrad 量化是收敛退化的主要贡献者**。

随后我们在同一条逐段打开的阶梯上评估各种稳定化策略。**Stochastic rounding** 在 Fprop 与 Fprop+Dgrad 上与未稳定化的 overhead 持平，但在**全流水（Fprop+Dgrad+Wgrad）下不收敛**。**Randomized Hadamard** 同样在部分阶段上与未稳定化持平，但在**全流水下也不收敛**。与此对照，对全流水 MXFP4 施加**确定性 Hadamard 变换能够恢复稳定训练，并把 token overhead 降回 8–9%**，同时维持相同的目标 perplexity；此外我们还发现，在我们的 kernel 中 **$H_{16}$（16×16 Hadamard 矩阵）比 $H_{32}$（32×32）快 8%**（1.08× vs 1.00×）。

**表 1：逐段打开 MXFP4 与各稳定化策略。** token overhead 相对 FP8 基线（validation perplexity 3.3）。「不收敛」表示该次运行发散，或即使延长训练也无法达到 perplexity 3.3 的目标。

| 稳定化策略 | Hadamard | 使用 MXFP4 的 GEMM（其余为 FP8） | token overhead |
|---|---|---|---|
| FP8 基线 | – | 无（全部 GEMM 为 FP8） | 0% |
| 无 | – | Fprop | 8–9% |
| 无 | – | Fprop + Dgrad | 10–11% |
| 无 | – | Fprop + Dgrad + Wgrad | **26–27%** |
| Stochastic Rounding | – | Fprop | 8–9% |
| Stochastic Rounding | – | Fprop + Dgrad | 10–11% |
| Stochastic Rounding | – | Fprop + Dgrad + Wgrad | **不收敛** |
| Randomized Hadamard | $H_{16}$ | Fprop | 8–9% |
| Randomized Hadamard | $H_{16}$ | Fprop + Dgrad | 10–11% |
| Randomized Hadamard | $H_{16}$ | Fprop + Dgrad + Wgrad | **不收敛** |
| Deterministic Hadamard | $H_{32}$ | Fprop | 8–9% |
| Deterministic Hadamard | $H_{32}$ | Fprop + Dgrad | 10–11% |
| Deterministic Hadamard | $H_{32}$ | Fprop + Dgrad + Wgrad | **8–9%** |
| Deterministic Hadamard | $H_{16}$ | Fprop | 8–9% |
| Deterministic Hadamard | $H_{16}$ | Fprop + Dgrad | 10–11% |
| Deterministic Hadamard | $H_{16}$ | Fprop + Dgrad + Wgrad | **8–9%** |

kernel 吞吐：$H_{16}$ 比 $H_{32}$ 快 **8%**（1.08× vs 1.00×）。

> **图 1**：Llama 3.1–8B 在 MLPerf C4 数据集预训练设置下，validation perplexity 随训练 token 数的变化。我们对比 FP8、全流水 MXFP4（Fprop + Dgrad + Wgrad，无稳定化）、以及全流水 MXFP4 + 确定性 Hadamard（$H_{16}$）。MXFP4 + 确定性 Hadamard 紧贴 FP8，而无稳定化的全流水 MXFP4 收敛更慢且更不稳定。

> **图 2**：训练后期的放大视图（来自图 2）。MLPerf 目标为 perplexity 3.3。相比未稳定化的 MXFP4 运行，确定性 Hadamard（$H_{16}$）与 FP8 基线保持紧密贴合。
>
> 译注：原文此处写作「from Figure 2」，按上下文应为「来自图 1」。

### 3.3 收敛性分析

我们现在把表 1 中 token overhead 结果背后的训练轨迹可视化出来。具体而言，我们把 validation perplexity 画成训练 token 数的函数，并评估每个配置是否**以稳定的方式**达到 MLPerf 目标（perplexity 3.3）。与逐段打开的结果一致：朴素的 MXFP4 在**一旦量化 Wgrad 之后**退化最严重，而确定性 Hadamard 能稳定全流水 MXFP4，并给出紧贴 FP8 基线的轨迹。

> **关键结论 #1**：逐步打开 MXFP4 表明，收敛退化**由量化 Wgrad 主导**。确定性 Hadamard 使稳定的全流水 MXFP4 训练成为可能；而当 Wgrad 被量化时，随机化技术呈现出**更高的**量化误差。

此外，我们还考察了：恢复收敛稳定性之后，FP4 最初的效率动机是否真正得以实现。对于稳定的配置（在 Fprop、Dgrad、Wgrad 上都施加确定性 Hadamard 的 MXFP4），在同样的 MLPerf C4 数据集设置下、在具备原生 FP4 支持的 MI355X GPU 上，我们观察到**比 FP8 基线更高的训练步吞吐**（表 2）。重要的是，这一吞吐优势**只有在 Wgrad 量化带来的主导不稳定性被控制住之后**，才会转化为端到端加速；否则，增加的到收敛 token 数或直接的发散会抵消掉原始的算力收益。这一观察说明：**FP4 的实际加速是「受稳定性闸控的」（stability-gated），而不是自动到手的。**

**表 2：端到端训练效率**（MXFP4 + $H_{16}$ vs. FP8 基线）。

| 指标 | MXFP4 + $H_{16}$ vs. FP8 |
|---|---|
| 训练步吞吐 | **+20%** |
| 到收敛的 token overhead | +8–9% |
| **端到端加速** | **+9–10%** |

> **关键结论 #2**：FP4 的实际加速受稳定性闸控——**不稳定 Wgrad 的话，增加的到目标 token 数会抵消训练步吞吐的收益**，从而掩盖 FP4 底层的硬件效率。

FP8 与 MXFP4 + $H_{16}$ 两组运行都采用相同的 MLPerf C4 数据集预训练设置；数值以在**相同 node、batch 与训练配置**下测得的 FP8 端到端 tokens/s 归一化。激活与权重体积的减小降低了带宽压力（这是大 batch 训练中的关键瓶颈），并且我们的方法还额外稳定了全流水训练（表 1）。

---

## 4 讨论与未来工作

我们的结果表明，**Wgrad 是全流水 MXFP4 训练中最敏感的分量**；并且稳定 FP4 预训练需要的是**控制 micro-scaling 误差**，而不是注入额外的随机性。同时，这套配方并不是普适的：**FP4 的训练行为会随模型、数据集与适配方法而变化。**

> **关键结论 #3**：MXFP4 的训练行为可能**高度依赖设定**：对完整预训练（MLPerf C4 数据集、Llama 3.1–8B）有效的稳定化手段，**未必能泛化**到其他模型或微调方法；因此在缺乏进一步证据之前，FP4 配方应当被视为**不通用的**。

---

## 5 结论

我们给出了一项受控研究：把 transformer 线性层中的 FP8 GEMM kernel 替换为 MXFP4，并表明收敛退化**由量化 Wgrad 主导**（相对于 Fprop 或 Dgrad）。对于 MLPerf C4 数据集设置下的 Llama 3.1–8B，**确定性 Hadamard 旋转是所测手段中唯一**能在恢复全流水 MXFP4 稳定训练的同时改善端到端效率的干预。总体而言，我们的结果给出了一条 FP4 LLM 训练的实用原则：**稳定性取决于控制最敏感梯度路径上的 micro-scaling 误差，而不取决于加入随机性。**

---

## 参考文献

> 按原文保持原样（未译）。

1. F. Abecassis, A. Agrusa, D. Ahn, J. Alben, S. Alborghetti, M. Andersch, S. Arayandi, A. Bjorlin, A. Blakeman, E. Briones, et al. *Pretraining large language models with nvfp4.* arXiv preprint arXiv:2509.25149. — 引用于 §1
2. Advanced Micro Devices, Inc. *AMD Instinct™ MI355X GPUs.* <https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html> — 引用于 §2.1
3. B. Chmiel, M. Fishman, R. Banner, and D. Soudry. *FP4 all the way: fully quantized training of LLMs.* arXiv preprint arXiv:2505.19115. — 引用于 §2.1
4. M. Cim, B. Topcu, and M. T. Kandemir. *Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4.* arXiv preprint arXiv:2603.08747. — 引用于 §2.1
5. A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. *The Llama 3 herd of models.* arXiv preprint arXiv:2407.21783. — 引用于 §1
6. S. Liu, Z. Liu, X. Huang, P. Dong, and K. Cheng. *LLM-FP4: 4-bit floating-point quantized transformers.* EMNLP 2023, pp. 592–605. — 引用于 §1
7. B. D. Rouhani, R. Zhao, A. More, M. Hall, A. Khodamoradi, S. Deng, D. Choudhary, M. Cornea, E. Dellinger, K. Denolf, et al. *Microscaling data formats for deep learning.* arXiv preprint arXiv:2310.10537. — 引用于附录 A.1
8. R. Wang, Y. Gong, X. Liu, G. Zhao, Z. Yang, B. Guo, Z. Zha, and P. Cheng. *Optimizing large language model training using FP4 quantization.* arXiv preprint arXiv:2501.17116. — 引用于 §2.1
9. X. Wei, Y. Zhang, X. Zhang, R. Gong, S. Zhang, Q. Zhang, F. Yu, and X. Liu. *Outlier suppression: pushing the limit of low-bit transformer language models.* NeurIPS 35, pp. 17402–17414, 2022. — 引用于附录 A.2
10. J. Zhou, D. Tang, R. Fu, B. Hu, H. Xu, Y. Wang, Z. Pei, Z. Su, L. Liu, X. Zhang, et al. *Towards efficient pre-training: exploring FP4 precision in large language models.* arXiv preprint arXiv:2502.11458. — 引用于 §1

---

## 附录 A　方法

### 附录 A.1　带 micro-scaling 的 MXFP4

标准的整数量化通常对**整个 tensor** 施加单一 scale factor。与之相对，MXFP4 使用 **micro-scaling**：为**小块（block）**定义一个共享指数（例如每 32 个元素一块）。设向量 $x$ 被划分为若干 block，对某个 block $x_B$，其重建值为：

$$x_{MX} = Q_{FP4}\!\left(\frac{x_B}{2^{E_{\mathrm{shared}}}}\right) \times 2^{E_{\mathrm{shared}}} \tag{1}$$

其中 $E_{\mathrm{shared}}$ 是**该 block 内的最大指数** [7]，$Q_{FP4}$ 表示就近舍入到可表示的 4-bit 浮点值。

### 附录 A.2　Hadamard 变换架构

我们使用**确定性** Hadamard 变换来改善收敛稳定性、降低 outlier 的影响 [9]。尽管 randomized Hadamard 旋转（例如随机符号翻转）已被广泛使用，但我们的实验发现，**在本设定下随机符号会对收敛稳定性产生不利影响**。因此我们采用确定性变换。

Hadamard 矩阵 $H$ 是元素为 ±1 的正交矩阵。施加 $H$ 可以理解为特征空间中的一次**旋转**，它把集中的能量**分散到更多维度上**。在我们的实现中，我们使用 tiled layout，并选择 Hadamard 变换尺寸为 16（$H_{16}$）。我们发现 $H_{16}$ 在目标架构上能提供更高的 kernel 吞吐，且经验上足以维持稳定收敛。

我们的流水在 **GEMM 之前**注入 Hadamard 旋转（见图 3）。由于该变换是**线性且正交**的，它只改变进入量化器的数据分布，而**不改变底层计算**。具体做法是：用 MXFP4 GEMM kernel 替换 transformer 线性层中所用的 FP8 GEMM kernel（针对选定的通路：Fprop、Dgrad 与/或 Wgrad），实现基于 **AMD 的 ROCm Transformer Engine**；这些 kernel 通过我们训练栈中标准的 transformer 线性模块被调用。

---

## 附录 B　架构示意图

> **图 3**：Hadamard 变换后的 MXFP4 架构（前向与反向）。进入 GEMM kernel 的输入被 $H$ 旋转，并在矩阵乘法过程中通过 $HH^{T} = I$ 抵消掉该旋转。

---

## 附录 C　抵消性的数学证明

为确保 Hadamard 变换不改变线性层的算术，我们利用正交性：

$$HH^{T} = H^{T}H = I \tag{2}$$

其中 $I$ 是单位矩阵。

### 附录 C.1　前向传播（FPROP）

在标准线性层中 $Y = XW^{T}$。我们对两个输入都施加变换：$\tilde{X} = XH$，$\tilde{W} = WH$。于是：

$$Y_{\mathrm{out}} = \tilde{X}(\tilde{W})^{T} = (XH)(WH)^{T} = XH(H^{T}W^{T}) = X(HH^{T})W^{T} = XW^{T} \tag{3}$$

因此 Hadamard 因子相互抵消，在**精确保持输出**的同时改善了量化鲁棒性。

### 附录 C.2　反向传播（DGRAD）

对输入梯度，标准运算是 $\nabla X = \nabla Y\,W$。取 $\tilde{G} = (\nabla Y)H$，$\tilde{W}_{\mathrm{rot}} = W^{T}H$：

$$\nabla X = \tilde{G}(\tilde{W}_{\mathrm{rot}})^{T} = ((\nabla Y)H)(W^{T}H)^{T} = (\nabla Y)HH^{T}(W^{T})^{T} = (\nabla Y)W \tag{4}$$

### 附录 C.3　权重梯度（WGRAD）

对权重梯度，$\nabla W = (\nabla Y)^{T}X$。在我们的流水中取 $\tilde{G}_{T} = (\nabla Y)^{T}H$，$\tilde{X}_{T} = X^{T}H$：

$$\nabla W = \tilde{G}_{T}(\tilde{X}_{T})^{T} = ((\nabla Y)^{T}H)(X^{T}H)^{T} = (\nabla Y)^{T}H(H^{T}X) = (\nabla Y)^{T}(HH^{T})X = (\nabla Y)^{T}X \tag{5}$$

---

## 附录 D　1D 与 2D 量化策略

我们对激活与梯度探索了 1D 与 2D 的 block 量化策略：

- **2D block 量化（32×32）**：把 tensor 划分为 32×32 的区域，为每个 $C_{\mathrm{in}} \times C_{\mathrm{out}}$ block 使用一个 per-block 的 E8M0 scale。这种方式**计算高效、对转置友好**，并且能启用硬件向量化，同时保持良好精度。
- **1D 行向量化（1×32）**：与 transformer 中的 token-wise 处理对齐。在 32 个连续元素上计算一个 per-token 的 E8M0 scale，保留了 token 级信息，**对激活效果良好**。
- **1D 列向量化（32×1）**：通过在 32 个元素上使用 per-$C_{\mathrm{out}}$ 的 scale，为权重提供细粒度的 per-channel scale；但**由于需要转置且访存模式不利，代价较高**。
