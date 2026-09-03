# AGoQ：面向 LLM 分布式训练显存效率的激活与梯度量化
# AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs

> **arXiv:** [2605.00539](https://arxiv.org/abs/2605.00539) | **HTML:** [全文](https://arxiv.org/html/2605.00539)
> **发表信息:** Preprint，arXiv 2026-05-01 提交，论文内页标注 "Preprint. May 12, 2026"（0 引用）
> **机构:** 哈尔滨工业大学（深圳）计算机科学与技术学院 · 华为技术有限公司；通讯作者 Shaohuai Shi
> **代码:** 未提供
> **领域:** 低精度训练 · 激活量化 · 梯度量化 · 集合通信 · Megatron-LM
> **核心贡献:** 两件事拼在一起。(1) **LAAQ**：按层类型和 PP stage 分配位宽，把激活压到近 4 bit——激活显存 28U → **7.75U**（3.6×），代价是 attention 模块明确不量化。(2) **QuanGrad**：FP8 存主梯度 + 一个"精度保持的 8-bit AllReduce"，把 AllReduce 重构成 **All-to-All → 本地反量化到 FP32 → 本地归约 → 再量化 → All-Gather**，理由是 **Reduce-Scatter 要求在通信过程中做加法，FP8 下极易溢出**。LLaMA 8B–32B、两个集群最多 64 GPU：显存最多降 **52%**，训练速度最多 **1.34×**。

> **读法建议**：本文对我们的价值几乎全部集中在 §5 那一段 AllReduce 重构，以及由它引出的 [§五「对 MI455X / Helios 的参考点」](#五对-mi455x--helios-的参考点)。如果只有五分钟，直接看 §五 的第 1、2、3 条。**结论先说：那段重构不是本文的原创（ZeRO++ 的 qgZ 在 2023 年就发表了同一个东西），而且 NVIDIA 已经在 Megatron-Core GTP 里以 `--gtp-remat-reduce-scatter-with-fp32-accumulation` 落地了完全同构的结构——只是线上仍走 BF16。所以"上游为什么把 wgrad RS 留在 BF16"的答案不是本文说的溢出，我们要抢的缺口也不是这个结构，而是格式。**

---

## 一、问题分析 (Problem Analysis)

### 1.1 研究背景 (Research Background)

**领域现状**：

- 训练 LLM 的显存由四块构成：模型参数、梯度、优化器状态、临时激活。**激活通常占最大比例**，且随序列长度和 batch size 线性增长——这两个恰好是最常被拉高的超参。
- 降激活显存有两条路：**重计算 / offload**（系统级，无精度副作用，但要付重算或搬运开销）和**激活量化**（用 INT8/FP8 存激活，backward 前反量化回 BF16/FP16）。
- 激活量化的现状停在 8 bit：Jetfire（INT8 数据流 + per-block）、COAT（FP8 + 动态量化 + 混粒度）。**两者都不适用于 4 bit。**
- 梯度侧的大量工作（1-bit Adam、gzccl、ZeRO++、Birder 等）压的是**通信量**，梯度本体仍存 FP32——**所以梯度占的显存和参数一样大，一分没省**。
- 唯一认真尝试 FP8 存梯度的是 Microsoft 的 FP8-LM：给每个梯度张量配 scaling factor，先 all-gather 各 rank 的 scale 取全局最小值 `s'_g`，让所有输入共享同一个 scalar，从而可以直接调标准 NCCL AllReduce。FP8-LM 自己在论文里承认，**NCCL 没有能力在 sub-tensor 粒度上处理 per-tensor scaling factor**，所以只能退到全局共享标量这一步。

**核心挑战**：

1. **朴素 4-bit 激活直接不收敛。** 本文消融明确记录：uniform 4-bit 应用到所有激活时**训练失败**。
2. **FP8 梯度在两处会溢出。** 一处是本地多 micro-batch 累加（`main_grad += local_grad` 直接在 FP8 里做会溢出）；一处是 AllReduce 内部——**Reduce-Scatter 的加法发生在"通信过程中"**，环上每一跳都要把收到的 partial sum 加进本地值再转发，这个加法在 FP8 里极易溢出。
3. **FP8-LM 的全局共享 scalar 换来了兼容性，代价是精度。** 本文 Fig. 8 的 loss 曲线显示 FP8-AllReduce 的 loss 显著高于 BF16 基线。

**研究动机**：把激活推到近 4 bit、梯度推到 8 bit，且**同时省显存和省通信**，还要与 8-bit Adam（Dettmers）正交叠加，不牺牲收敛。

### 1.2 问题定义 (Problem Definition)

- **输入**：一个标准 LLaMA 架构 transformer，跑在 Megatron-LM 的 TP × PP × DP 三维并行上，interleaved 1F1B 流水。
- **输出**：一套量化方案——每个 activation tensor 一个位宽、每个梯度 buffer 一个存储格式、一套 DP 梯度归约的通信算法。
- **约束**：(a) 收敛曲线不劣于 BF16 基线；(b) 下游 zero-shot 精度不掉；(c) 峰值显存不因补偿策略而上升；(d) 量化/反量化开销必须被融进 GEMM，不能新增可见延迟。
- **评价指标**：峰值显存（MB/GB）、迭代时间（ms）、吞吐（samples/s）、pretrain loss 曲线、六个 zero-shot 数据集精度、per-layer 梯度误差（MAE + normalized L2）、梯度范数轨迹。

**问题形式化**：AllReduce 定义为

$$X = \text{AllReduce}(X_1,\dots,X_P) = \sum_{i=1}^{P} X_i$$

本文的关键论断是：当 $X_i$ 以 FP8 存储时，把这个求和**分解成 Reduce-Scatter + All-Gather 会让部分和在低精度里逐跳累积**，因此必须改变分解方式。

### 1.3 解决方案 (Solution)

**核心思路**：低精度**传输**可以，低精度**累加**不行。因此 (a) 激活侧按"这一层的量化误差会被 Jacobian 放大多少"来分配位宽；(b) 梯度侧把所有加法都赶到本地、在宽类型里做，线上只承担搬运。

**方法概述**：

1. **误差分析定层级**（§4.1 + 附录 8.1）：对 RMSNorm、SiLU&Multiply、RMSNorm+GEMM、Attention 分别推导量化扰动 $\delta$ 传到梯度上的一阶界，比较"只存量化输入、中间量重算"（Case 1）与"输入和中间量都存量化"（Case 2）。
2. **LAAQ 分配位宽**：可安全量化的层压到 4-bit block-wise FP4（blocksize 128），attention 模块**完全不量化**，中间量一律重算不存。
3. **DBCA-PP 动态位宽补偿**（§4.2）：利用 interleaved 1F1B 各 stage 激活份数不等造成的显存空洞，给存得少的 stage 分配更高位宽。
4. **量化/反量化融进 GEMM**（§4.3）：forward 把 quantize 融进后继 GEMM，backward 把 dequantize 融进算 activation gradient 的 GEMM。
5. **QuanGrad 的两级防溢出**（§5）：本地累加时 FP8 主梯度先反量化到 FP16/BF16 做高精度加法再量化回去；跨 DP 时把 AllReduce 拆成 A2A + 本地 FP32 归约 + AG。

**技术细节**：

*LAAQ（Layer-Aware Activation Quantization）*：

- **作用**：决定每个 activation tensor 存几位、以及哪些中间量不存。
- **实现**：先把模块分成两类——"只需存输入"（MLP 里的 GEMM）与"还需存额外中间量"（RMSNorm 存 $r$、SiLU&Multiply 存 sigmoid、attention 存 $P$）。以 RMSNorm 为例，$Y=\gamma X/r$，$r=\sqrt{\|X\|_2^2/d+\epsilon}$，Jacobian $J=\mathrm{diag}(\gamma)/r-\mathrm{diag}(\gamma)XX^T/(dr^3)$，所以梯度需要 $X$ 和 $r$ 两者。把量化误差建模成乘性扰动 $X'=X\odot(1+\delta_X)$，一阶展开后 Case 1 得 $\|\Delta J\|_2 \lesssim 3\|\gamma\|_\infty\|\delta_X\|_\infty\cdot(\cdot)$，Case 2 得 $\|\Delta J\|_2 \lesssim 6\|\gamma\|_\infty\varepsilon_q/r$——**同阶，只差常数，而且重算那一侧的常数更紧**。SiLU&Multiply 的结论更强：在 $|x|,|y|<1$ 的常见区间里，Case 1 的界**严格小于** Case 2。
- **创新**：把"存不存中间量"从工程 trade-off 变成一个有一阶误差界支撑的判定，并给出 attention 必须排除的定量理由——attention 的梯度误差比 RMSNorm 大 $\Theta(L\sqrt{d_k})$（对 $\partial A/\partial V$）到 $\Theta(L^{3/2}\sqrt{d_k})$（对 $\partial A/\partial Q$、$\partial A/\partial K$）倍，随序列长度 $L$ 放大。这个理论预测被 Table 13 实测证实：Attention Q/K 的 normalized L2 误差 0.14/0.15，是全部层里最大的。

*DBCA-PP（Dynamic Bit-width Compensation for Activation with PP）*：

- **作用**：把 PP 造成的显存不均衡换成精度。
- **实现**：4 stage / 8 micro-batch 时，Device 1 峰值持有 11 份 mini-batch 激活，Device 2/3/4 只有 9/7/5 份——Device 1 的激活显存是 Device 4 的 **2.2×**。于是让持有份数少的 stage 用更高位宽：$B_i = 4\cdot N_1/N_i$（下限 4）。实测配出的是 **4、5、6、8 bit** 四档，峰值显存与全 4-bit 一致。
- **创新**：低 stage 数下算出的位宽配置可以**直接搬到高 stage 数**而不抬高峰值显存。

*QuanGrad（精度保持的梯度量化与通信）*——**本文对我们唯一重要的部分，完整展开**：

- **作用**：让梯度以 8 bit 存、以 8 bit 上线，同时不让任何一次加法发生在 8 bit 里。
- **实现分两条路径**：

  **路径 A：本地梯度累加（Fig. 2 右半）。** 一个 global batch 被切成多个 mini-batch，各自的梯度要先在本地累加再跨 DP 通信。如果主梯度存 FP8，`Q-Main Gradient += Local Gradient` **直接在 FP8 里累加会立刻溢出**——micro-batch 数就是累加次数，几十次累加后 FP8 E4M3 的 3 位尾数完全不够。所以每个 micro-batch 走：**反量化 Q-Main Gradient 到 FP16/BF16 → 与本 micro-batch 的 local gradient 在高精度里相加 → 把和量化回 FP8 → 写回 Q-Main Gradient**。也就是说 FP8 只是**存储格式**，累加器永远是 16 bit。代价是每个 micro-batch 一次 dequant + 一次 quant 的全梯度尺寸访存。

  **路径 B：跨 DP 归约（Fig. 5 + Fig. 6）。** 常规做法是 AllReduce = Reduce-Scatter + All-Gather。**Reduce-Scatter 的问题是它要求在通信过程中做加法**：环形 RS 的每一跳，接收方都要把收到的 partial sum 加到自己的对应分片上再转发下一跳，$W$ 个 rank 要走 $W-1$ 跳，就有 $W-1$ 次低精度加法（如果线上格式是 FP8，还要 $W-1$ 次 quant/dequant 往返）。本文的替换方案是：
  1. 本地把 FP8 梯度按 rank 切成 $W$ 段（Fig. 5 左：GPU1 持有 A1..A4，GPU2 持有 B1..B4，…）；
  2. **All-to-All**：把第 $j$ 段发给 rank $j$，于是 rank $j$ 收齐 A$j$、B$j$、C$j$、D$j$——**纯搬运，不做任何加法**；
  3. 每个 rank 把收到的 $W$ 个 FP8 片段**反量化到 FP32**；
  4. **本地 reduce**：在 FP32 里把 $W$ 个片段加成一个（Fig. 5 中间：A1+B1+C1+D1）；
  5. **重新量化**成 8 bit；
  6. **All-Gather** 把各 rank 的归约结果广播回全体，完成 AllReduce 语义（Fig. 5 右：每个 GPU 都拿到全部四段的和）。

  Fig. 6 是同一件事的数据流视图，标注了位宽变化：`32bit --Quantize--> 8bit --All-to-All--> 8bit --Dequantize--> 32bit --(+)--> 32bit --Quantize--> 8bit --All-Gather-->`。**关键性质：整条路径上量化/反量化各恰好发生一次，与 $W$ 无关；所有加法都在本地 FP32 里。**

- **创新**（本文自己的定位）：把"低精度归约"从"让集合通信库支持低精度加法"这个死胡同里搬出来，改成"重排通信结构使得加法根本不发生在线上"。**§三 会说明这个创新点在 2023 年已被 ZeRO++ 的 qgZ 发表过，且 qgZ 还多做了一层本文没做的层级化优化。**

**算法/架构描述**：整个系统是 Megatron-LM 上的一个改造（Fig. 2）。Forward 先算出全精度激活，立刻量化到约 4 bit 存下；backward 前反量化回 BF16/FP16 再算梯度。梯度侧每 mini-batch 算出 local gradient，走路径 A 累进 FP8 主梯度；一个 global batch 的所有 micro-batch 累完后，走路径 B 做跨 DP 的 FP8 AllReduce。优化器是 8-bit Adam。三个模块（A / G / O）在实现上互相独立。

---

## 二、实验效果 (Experimental Results)

### 2.1 实验设置 (Experimental Setup)

| Item | Details |
|------|---------|
| 主集群 | **64 GPU**：8 节点 × 8 **NVIDIA A6000**，200 Gb/s InfiniBand |
| 副集群 | **16 GPU**：2 节点 × 8 **NVIDIA Pro 6000**（Blackwell，为支持 COAT 需要的 FP8 格式） |
| 另有 | 华为 Ascend 910 NPU（附录 8.2） |
| 软件 | Ubuntu 20.04 · CUDA 12.1 · PyTorch 2.1.2 · **NCCL 2.18.5** |
| 基线 | Megatron-LM（w/o ZeRO）、Megatron-LM + ZeRO-1、DeepSpeed（ZeRO-1/2/3）、COAT |
| 模型 | 收敛验证 LLaMA2-7B、LLaMA3-8B、LLaMA3.2-1B；计时对比 LLaMA3-8B、LLaMA2-13B、CodeLLaMA-34B；对比 COAT 用 OLMo-1B |
| 数据 | OpenWebText，pretrain **2B token**（LLaMA2-7B）/ **10B token**（LLaMA3.2-1B）/ 100k iter × 32k token = 3.2B token（LLaMA3-8B 梯度范数实验） |
| 下游评测 | ARC-Challenge、ARC-Easy、HellaSwag、PIQA、SciQ、WinoGrande |

### 2.2 主要结果 (Main Results)

**激活显存的理论账（Table 1，$1U = BS \times L \times H \times 2$ bytes）：**

| 方法 | QKV | Attention | Linear | RMSNorm | FFN1 | Act Func (SiLU&Mul) | FFN2 | **Total** |
|---|---|---|---|---|---|---|---|---|
| Megatron-LM (BF16) | 1U | 5U | 1U | 4U | 1U | 12U | 4U | **28U** |
| COAT | 1U | 5U | 1U | 1U | 0.5U | 6U | 2U | **16.5U** |
| **AGoQ** | **0** | 5U | 0.25U | **0.5U** | **0** | **2U** | **0** | **7.75U** |

RMSNorm 那 4U 是因为 Megatron 的 BF16 训练里 RMSNorm 仍用 FP32。AGoQ 把它压到 0.5U，SiLU&Multiply 从 12U 压到 2U，QKV / FFN1 / FFN2 的输入直接不存（重算）。**attention 的 5U 一动不动**——这是 §4.1 误差分析的直接结果。28U → 7.75U ≈ **3.6×**。

**端到端时间（Table 2，LLaMA2-13B，64 GPU，PP=4 TP=8，mbs=1 gbs=16，R = 需重计算的 transformer 层数）：**

| Seq Len | Megatron R / ms | ZeRO-1 R / ms | AGoQ R / ms | vs Megatron |
|---|---|---|---|---|
| 32K | 3 / 37 635 | 2 / 37 038 | **0 / 36 568** | 1.03× |
| 40K | 4 / 51 922 | 4 / 51 418 | **0 / 45 590** | 1.14× |
| 48K | 6 / 67 932 | 6 / 67 928 | **0 / 57 047** | 1.19× |
| 56K | 8 / 88 200 | 8 / 86 601 | **0 / 69 544** | 1.27× |
| 64K | 8 / 104 444 | 8 / 103 706 | **0 / 82 519** | 1.27× |
| 72K | 10 / 128 085 | 10 / 128 152 | **0 / 97 615** | 1.31× |
| 80K | 10 / 149 667 | 10 / 149 288 | **0 / 111 422** | **1.34×** |

**这张表的 R 列比时间列重要。** 加速比与 R 的差值几乎完美同步：R 差 3 时加速 1.03×，R 差 10 时加速 1.34×。**摘要里那个 1.34× 的来源是"激活省下来的显存让 10 层重计算被取消"，与梯度量化基本无关**——而且这个配置下 DP = 64/(4×8) = **2**，梯度 AllReduce 只有两个参与方，路径 B 那套重构在 $W=2$ 时几乎无事可做。

**通信延迟分解（Table 4，TP=8 DP=8，格式为 200 Gbps / 10 Gbps，单位 ms）：**

| Message | All-Reduce | All-to-All | Quant/Dequant | All-Gather | **AGoQ 合计** |
|---|---|---|---|---|---|
| 2³⁰ (1 GB) | 4292.77 / 50365.45 | 599.28 / 7297.73 | 31.03 / 31.07 | 556.47 / **226.58** | 1186.78 / 7555.38 |
| 2²⁵ (32 MB) | 131.23 / 1603.78 | 18.81 / 233.34 | 0.99 / 1.10 | 19.37 / 197.92 | **39.17 / 432.36** |
| 2²⁰ (1 MB) | 4.13 / 51.07 | 0.83 / 5.52 | 0.07 / 0.07 | 0.55 / 7.87 | 1.45 / 13.46 |
| 2¹⁵ (32 KB) | 0.83 / 3.93 | 0.35 / 0.38 | 0.05 / 0.05 | 0.41 / 0.83 | 0.81 / 1.26 |

论文据此宣称 32 MB 下 **3.4×**（39.17 vs 131.23）、10 Gbps 下 **3.7×**。⚠ **这张表有两处必须点出的问题**，见 §2.4。

**显存分解（Table 8，LLaMA2-13B，TP=8 PP=1 seq 12K，单位 GB）：**

| | AGoQ (A+O+G) | O+G | O | Megatron-LM |
|---|---|---|---|---|
| GPU | **22.3** | 35.3 | 37.7 | 46.1 |
| NPU | **29.7** | 40.5 | 45.3 | 55.2 |

逐项归因（GPU 侧）：**8-bit Adam 贡献 8.4 GB（18%）、激活量化贡献 13.0 GB（28%）、梯度量化只贡献 2.4 GB（5.2%）**，合计 23.8 / 46.1 = **51.6%**（论文写 53%，取整口径不同）。NPU 侧 O = 9.9、A = 10.8、G = 4.8，合计 46%。

**长序列显存（Table 14，LLaMA3-8B）：** 32k / 36 层：48 606 → **16 594 MB（−65.9%）**；64k / 16 层：46 681 → **19 267 MB（−58.7%）**。

**vs COAT（Table 3，OLMo-1B，2 节点 × 8 Pro6000，gbs=64）：**

| Seq Len | 方法 | Time (ms) | Memory (MB) |
|---|---|---|---|
| 24k | COAT | 6291 | 94 100 |
| 24k | **AGoQ** | **6161** | **66 852**（−29%） |
| 32k | COAT（一半层重计算） | 8861 | 95 664 |
| 32k | **AGoQ** | **8076**（1.10×） | **86 012** |

论文写"reduces memory by 31% over COAT"，实算 $1-66852/94100 = 28.96\%$，应为 **29%**。

**逐层 wall-clock 分解（Table 5 / Table 6，单个 transformer decoder layer，ms）：**

| | ln | ag/rs | Attn | rs/ag | ln | ag/rs | FFN | rs/ag | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline Fwd | 1.3 | 17.02 | 19.34 | 21.79 | 1.3 | 17 | 13.3 | 23.01 | 114.06 |
| **AGoQ Fwd** | 1.3 | 16.82 | 20.62 | 21.65 | 1.3 | 17 | **15.64** | 21.78 | 116.11 |
| Baseline Bwd | 2.85 | 23.21 | 45.06 | 35.4 | 3.2 | 22.62 | 28.6 | 34 | 194.94 |
| **AGoQ Bwd** | 2.9 | 22.82 | **47.86** | 34.56 | 3.4 | 22.58 | **32.54** | 33.54 | 200.20 |

论文用这张表论证"融合后的量化/反量化开销可忽略"——FFN forward 13.3 → 15.64、FFN backward 28.6 → 32.54、Attn forward 19.3 → 20.6、Attn backward 45.1 → 47.8，总计 forward +1.8%、backward +2.7%。

**但这张表还暴露了论文没说的东西**：TP/SP 的 `ag/rs` + `rs/ag` 四项在 baseline forward 里合计 78.82 ms，占 114.06 的 **69%**；backward 里 115.23 ms，占 194.94 的 **59%**。**AGoQ 对这部分一个字节都没动**（16.82 vs 17.02 是噪声）。也就是说：论文优化的是 DP 梯度 AllReduce，而它自己的分解显示真正的通信大头是 **TP+SP 的激活 ReduceScatter/AllGather，且仍然是 BF16**。这一条对我们的项目极其关键，见 §五。

**收敛性证据（三条，强度递增）：**

1. **loss 曲线（Fig. 8）**：LLaMA2-7B（gbs=512）和 LLaMA3-8B（gbs=4）上 AGoQ 紧跟 BF16 基线，而 FP8-AllReduce（Peng et al. 2023b）的 loss 显著更高。
2. **zero-shot 精度（Table 10）**：

| Dataset | LLaMA2-7B Base | AGoQ | Δ | LLaMA3.2-1B Base | AGoQ | Δ |
|---|---|---|---|---|---|---|
| arc_c | 0.1988 | 0.1834 | **−0.0154** | 0.1877 | 0.2099 | +0.0222 |
| arc_e | 0.4179 | 0.4158 | −0.0021 | 0.4571 | 0.4714 | +0.0143 |
| hellaswag | 0.2886 | 0.2897 | +0.0011 | 0.3276 | 0.3298 | +0.0022 |
| piqa | 0.5990 | 0.6039 | +0.0049 | 0.6219 | 0.6284 | +0.0065 |
| sciq | 0.7260 | 0.7280 | +0.0020 | 0.7180 | 0.7100 | −0.0080 |
| winogrande | 0.4830 | 0.5036 | **+0.0206** | 0.5193 | 0.5185 | −0.0008 |

3. **梯度范数轨迹（Table 15，LLaMA3-8B，100k iter × 32k token）**——这是最有说服力的一条：

| Iter | AGoQ | Baseline | | Iter | AGoQ | Baseline |
|---|---|---|---|---|---|---|
| 0–10k | 7.28 | 7.10 | | 50–60k | 3.32 | 3.10 |
| 10–20k | 4.33 | 4.38 | | 60–70k | 3.17 | 3.15 |
| 20–30k | 3.79 | 3.77 | | 70–80k | 3.32 | 3.29 |
| 30–40k | 3.64 | 3.36 | | 80–90k | 3.20 | 3.37 |
| 40–50k | 3.38 | 3.20 | | 90–100k | **3.08** | 3.21 |

**逐层梯度误差（Table 13，10B token 训练后采样）：**

| Layer | MAE | Normalized L2 |
|---|---|---|
| LayerNorm | 2.8×10⁻¹⁰ | **0.003** |
| GEMM (Weight) | 5.3×10⁻⁷ | 0.026 |
| SiLU | 2.6×10⁻⁹ | 0.051 |
| **Attention Q** | 2.5×10⁻⁹ | **0.14** |
| **Attention K** | 1.7×10⁻⁹ | **0.15** |
| Attention V | 2.0×10⁻⁹ | 0.059 |

这张表是全文最扎实的证据：它独立验证了 §4.1 那套一阶误差分析的预测排序，Attention Q/K 确实高出一个数量级，正是被排除量化的两个。

**大规模扫描**：序列 16K–80K、PP=1 到节点数、TP∈{4,8}、GPU∈{8,16,32,64}，三个模型，**Megatron / ZeRO-1 / AGoQ 各 124 组共 372 次实验**。69 次 OOM（AGoQ 16、Megatron 33、ZeRO-1 20），303 次成功。平均 **1.23× vs Megatron、1.19× vs ZeRO-1**。

**Kernel 融合（Table 11）**：四个 LLaMA3-8B 的实际 GEMM 形状，融合 vs 顺序执行加速 1.03×–1.11×，均值 **1.07×**，端到端 **1.05×**。

### 2.3 消融实验 (Ablation Study)

| 配置 | 结果 | 备注 |
|---|---|---|
| **uniform 4-bit（全层一律 4 bit）** | **训练失败，不收敛** | 这是 LAAQ 存在的唯一必要性证明，也是全文最有力的消融 |
| **w/o DBC** | arc_c 17.66 / arc_e 41.88 / hellas 28.53 / piqa 60.12 / sciq 71.20 / winog 50.83 | |
| **w/ DBC** | arc_c 18.34 / arc_e 41.58 / hellas 28.97 / piqa 60.39 / sciq **72.80** / winog 50.36 | 六项里四升两降 |
| **O**（仅 8-bit optimizer） | 46.1 → 37.7 GB | 省 8.4 GB |
| **O+G**（+梯度量化） | → 35.3 GB | 梯度量化仅省 **2.4 GB** |
| **A+O+G**（完整 AGoQ） | → 22.3 GB | 激活量化省 13.0 GB |
| **ZeRO-1 → 2 → 3** | 速度单调下降 | 论文据此只与 ZeRO-1 深入对比 |

**消融结论**：

- **最重要的组件是 LAAQ 的"不要一律 4-bit"这个判定**——uniform 4-bit 直接崩，这是唯一的定性（而非定量）消融结果。
- **省显存最多的是激活量化（13.0 GB），其次是别人的 8-bit Adam（8.4 GB），梯度量化最少（2.4 GB）。** 梯度量化的价值在通信不在显存，而通信侧的独立消融**缺失**——Table 4 只是 microbenchmark，没有"只开 G、看端到端"这一格。
- **DBC 的精度收益不可信**：六项里四升两降，最大变动 sciq +1.6 个点，而这些模型在所有六个数据集上都处于随机猜测水平（见 §2.4）。

### 2.4 数据自洽性核查（本笔记补充）

论文有四处数据问题，两处是致命的：

**(1) Table 7 与其正文互相矛盾——最严重的一处。** 正文说"Our method consistently outperforms both baselines across all sequence lengths, achieving speedups of up to 1.33× over Megatron-LM and 1.16× over ZeRO-1 at 2k"，而表本身是：

| Seq | Megatron-LM | ZeRO-1 | AGoQ (Ours) |
|---|---|---|---|
| 2k | 2862.22 | 2498.13 | **2148.82** |
| 4k | 4626.94 | 4261.65 | **3968.20** |
| 8k | 8188.41 | 7848.12 | **7755.74** |

表头写的是 **Throughput (samples/sec)**，而 AGoQ 在**每一行都是最低的**。反推可知作者算的是 $2862.22/2148.82 = 1.33$、$2498.13/2148.82 = 1.163$——**即用"基线 ÷ AGoQ"当成了 AGoQ 的加速比**。如果表头正确（samples/sec），那么论文自己的数据说明在 2k–8k 短序列下 AGoQ **慢 1.06×–1.33×**；如果数字实际是迭代时间被误标成吞吐，那结论反过来但表头是错的。两种情况下这一节都不能作为证据使用。

**(2) Table 4 的 1 GB 行物理上不可能。** All-Gather 在 10 Gbps 下是 **226.58 ms**，比 200 Gbps 下的 **556.47 ms** 更快。带宽降 20 倍而延迟降 2.5 倍，没有任何解释能成立。（该数字与该行合计 7555.38 内部自洽，所以不是加总错误，是原始数据错。）

**(3) Table 4 的 3.4× 无法支持"重构比原生低精度 RS 更快"这个论点。** 200 Gbps / 32 MB 下 AllReduce 131.23 ms vs AGoQ 39.17 ms，比值 3.35×。而如果基线 AllReduce 走 FP32（4 B/elem）、AGoQ 走 FP8（1 B/elem），仅**格式**就给出 4× 的字节比（AllReduce = $2\times4\times N$ vs A2A+AG = $1\times N + 1\times N$）。**实测 3.35× ≈ 4× 减去量化开销，也就是说这 3.4× 全部来自位宽，一点也不来自结构重排。** 论文从未测量真正的对照组——一个原生 FP8 的 RS+AG。§三 会说明这个对照组的答案已经由别人给出了：**字节完全相同**。

**(4) Eq. 21 与其正文不符。** 论文写 $N_i = n + 2i - 1$，$n=4$ 时得 $N_1=5, N_2=7, N_3=9, N_4=11$（递增），而正文说的是"Device 1 持有 11 份，Device 2/3/4 持有 9/7/5 份"（递减）。要得到 Table 里实际用的 4/5/6/8 位宽，必须用 $N_1=11$ 代入 $B_i=4N_1/N_i$，即需要 $N_i = 3n-2i+1$。**印刷的公式与实际使用的配置不一致。**

另外，"平均 1.23× / 1.19×"这个数字是在一个**被删失的样本**上算的：Megatron OOM 33 次、AGoQ 只 16 次，两者的均值来自不同的配置子集，而被排除的正是显存最紧、AGoQ 优势最大（或最小）的那些点。

---

## 三、业界类似方案 (Industry Similar Solutions)

### 3.1 方案对比表 (Solution Comparison Table)

| Solution | Year | 核心思路 | 优点 | 缺点 | 关键数据 |
|---|---|---|---|---|---|
| **8-bit Optimizers**（Dettmers, ICLR'22） | 2022 | block-wise 量化优化器状态 | 正交、无副作用、已成事实标准 | 只管优化器状态 | AGoQ 里贡献 8.4 GB / 18% |
| **FP8-LM**（Microsoft, 2310.18313） | 2023 | 全局共享最小 scaling factor，使标准 NCCL AllReduce 可直接用于 FP8 梯度 | 零集合通信改动 | **加法仍在 FP8 里发生**；论文自承 NCCL 无法在 sub-tensor 粒度处理 per-tensor scale | AGoQ Fig. 8 显示其 loss 显著更高 |
| **ZeRO++ qgZ**（Microsoft, 2306.10209, ICLR'24） | 2023 | **彻底放弃 ring RS，改用 1-hop All-to-All + 反量化 + 高精度归约**；再加 **2-hop 层级化**（先节点内 A2A+reduce，再节点间 A2A）+ tensor slice reordering | **顺序 quant/dequant kernel 数从 $n$ 降到 1**；层级化把跨节点流量从 $M\cdot N/Z$ 降到 $M/Z$；kernel 融合 | INT4 精度激进 | 梯度 RS 通信量 **4×**（$M\to0.25M$），总量 $3M\to0.75M$；最多 384 GPU |
| **Jetfire**（ICML'24） | 2024 | INT8 数据流 + per-block 量化 | 端到端 INT8 | 不适用 4-bit | — |
| **COAT**（ICLR'25） | 2025 | FP8 激活 + FP8 优化器状态，动态范围扩展 + 混粒度量化 | 激活+优化器一起压 | 8-bit 天花板 | AGoQ 在 24k 上省它 29% 显存 |
| **Megatron-Core GTP**（NVIDIA, 2026） | 2026 | 权重以原生 MXFP8/NVFP4 分片并 packed all-gather；`--gtp-remat-reduce-scatter-with-fp32-accumulation` **把 ring RS 换成 A2A + 一次本地 FP32 求和** | 生产级；GB200 NVL72 上 128→3072 GPU 保持 ≥93% 扩展效率 | **线上仍是 BF16**（RS 恒定 2.0 B/elem） | NVFP4 下 wgrad RS 占单权重通信预算 **64%**（BF16 RS）到 **78%**（FP32 RS） |
| **NCCL 2.27+ 对称内存 / NVLS** | 2026 | 注册对称内存池，RS 走单个 symmetric device kernel，交换机内归约；`multimem.ld_reduce` 对 BF16 输入用 `.acc::f32` | **在网络里就以 FP32 累加**，SM 占用更低、延迟更低 | 依赖 NVLink Switch 硬件 | GTP 文档明确：**注册了对称池的 group 优先走它，A2A+FP32 路径被绕过** |
| **AGoQ**（本文） | 2026 | LAAQ 近 4-bit 激活 + FP8 主梯度（本地高精度累加）+ A2A→FP32 reduce→AG 的 FP8 AllReduce | 激活侧的一阶误差分析扎实；A+O+G 三模块正交 | **梯度侧的结构重构与 qgZ 重复**；DP ≤ 8 且主实验 DP=2；无原生低精度 RS 对照 | 显存 −52%、速度 1.34×（后者主要来自取消重计算） |

### 3.2 技术路线对比 (Technical Approach Comparison)

**路线 A：让集合通信库支持低精度归约。**
- 代表：FP8-LM（全局共享 scalar 绕开库的限制）、NCCL 的 `ncclFloat8e4m3`/`e5m2`、NVLS 的 `multimem.ld_reduce .acc::f32`、NCCL issue #2199（MXFP8/MXFP4/NVFP4 AllReduce 的 RFE，**仍未实现**）。
- 核心思路：把精度问题下推到通信库或交换机硬件里解决。
- 优劣：一旦硬件支持（NVLS 的交换机内 FP32 累加），这是最优解——零额外显存、零额外 kernel、SM 占用最低。但 (a) **块缩放格式（MX/NVFP4）至今没有任何集合通信库支持**；(b) 依赖专用交换机。

**路线 B：重排通信结构，让加法只发生在本地宽类型里。**
- 代表：**ZeRO++ qgZ（2023，首创）**、Megatron-Core GTP 的 `reduce_scatter_with_fp32_accumulation`（2026，BF16 线上）、**AGoQ（2026，FP8 线上）**。
- 核心思路：A2A 只搬运不计算，把 $W-1$ 次线上低精度加法压缩成 1 次本地 FP32 归约。
- 优劣：纯框架层工作，不需要通信库或硬件配合，**这是它唯一但决定性的优势**。代价是一个未分片梯度尺寸的临时 buffer、一次全量本地归约、以及 A2A 需要全 bisection 带宽才不吃亏。

**路线 C：从源头减少要归约的东西。**
- 代表：梯度稀疏化、1-bit Adam、Birder、低秩梯度。
- 与 A/B 正交，本文未涉及。

### 3.3 本文定位 (This Paper's Position)

- **相对路线 A 的改进**：不需要库或硬件支持任何低精度归约语义，纯框架侧可落地；且明确指出了 FP8-LM 那条"全局共享 scalar + 标准 AllReduce"路线的精度代价（Fig. 8）。
- **相对路线 B 的改进**：**基本没有。** 这是本文最大的问题。AGoQ 的 Fig. 5/Fig. 6 与 ZeRO++ qgZ 的 Figure 5 右半在结构上是同一个算法：量化 → A2A → 反量化 → 高精度归约。qgZ 的原话是"we completely abandon existing ring-based reduce-scatter approach and incorporate 1-hop all-to-all collective… we reduce the number of sequential quantization+dequantization kernel from the number of GPUs to 1"。差别只有三处，且**两处对 AGoQ 不利**：
  1. AGoQ 用 FP8，qgZ 用 INT4（AGoQ 略保守，可算小优势）；
  2. **qgZ 有 2-hop 层级化 A2A，AGoQ 只有 flat 1-hop。** qgZ 明确指出 1-hop A2A 在跨节点时通信量会 blow up（$M\cdot N/Z$），层级化后降到 $M/Z$。AGoQ 完全没有这一层，在多节点上会直接吃到 qgZ 已经解决过的问题。
  3. **qgZ 还有 tensor slice reordering** 保证 A2A 后分片落位正确，以及 (reorder+quantize)、(dequant+reduce+quantize) 的 kernel 融合。AGoQ 对落位和融合都未讨论。

  更要紧的是：**AGoQ 在引言里引用了 ZeRO++（Wang et al., 2024），并把 DeepSpeed 列为基线，但从未提及 qgZ、也从未打开 `zero_quantized_gradients=true` 做对比。** 它对 DeepSpeed 的比较只覆盖 ZeRO-1/2/3 的吞吐。这在一篇 2026 年的论文里是实质性的相关工作缺失。
- **独特贡献（去掉重复部分后剩下的）**：
  1. **激活侧的一阶误差分析**（§4.1 + 附录 8.1），特别是"attention 的梯度误差比 RMSNorm 大 $\Theta(L\sqrt{d_k})$ 到 $\Theta(L^{3/2}\sqrt{d_k})$"这个随序列长度放大的定量结论，加上 Table 13 的实测证实。**这部分是原创且扎实的，是本文真正的价值所在。**
  2. **"Case 1 重算中间量的误差界比 Case 2 缓存中间量更紧"**——一个反直觉且有用的结论：重算不仅省显存，误差还更小。
  3. **DBCA-PP**：用 PP 的显存不均衡换精度，且低 stage 数的配置可上移。这个 idea 本身新颖（尽管精度证据薄弱）。
  4. **本地梯度累加路径**（FP8 存、FP16/BF16 累加）：虽然思路直白，但把它与跨 DP 那条路径**并列**为"两处溢出"、并明确指出 micro-batch 数就是累加次数，这个表述是清晰的贡献。

### 3.4 推荐进一步阅读 (Recommended Further Reading)

| 论文 | 为什么值得读 |
|---|---|
| **ZeRO++**（[2306.10209](https://arxiv.org/abs/2306.10209)，ICLR'24） | **读这篇而不是 AGoQ。** qgZ 是 A2A-替代-RS 的原始出处，且多了 2-hop 层级化和 slice reordering。层级化那一节直接决定我们跨机架时该怎么做 |
| **Megatron-Core GTP 文档**（[docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/nightly/api-guide/core/generalized_tensor_parallel.html)） | 上游对同一问题的**生产级答案**：§2.6 的 A2A+FP32 累加、§2.7 的 NCCL 对称内存、以及那张 per-element 通信预算表。我们要抢的缺口的精确坐标在这里 |
| **FP8-LM**（[2310.18313](https://arxiv.org/abs/2310.18313)） | 反例的完整论证：为什么"全局共享 scalar + 标准 AllReduce"这条最省事的路会掉精度。含 underflow/overflow/SNR 的定量对比 |
| **COAT**（ICLR'25，[OpenReview](https://openreview.net/forum?id=XfKSDgqIRj)） | AGoQ 激活侧的直接前身与主要基线，动态范围扩展 + 混粒度量化 |
| **NCCL issue [#2199](https://github.com/nvidia/nccl/issues/2199)** | MXFP8/MXFP4/NVFP4 AllReduce 的 RFE。提问者把"block-scaled datatype semantics, cross-rank scaling/alignment, reduction behavior"列为实现难点——**这就是 §五 第 3 条的问题清单，由 NVIDIA 的客户自己写下来的** |
| **8-bit Optimizers**（Dettmers, ICLR'22） | AGoQ 里省显存第二多的那 8.4 GB 的出处，也是 block-wise 量化的原始形式 |

---

## 四、全文翻译 (Full Translation)

> 以下为论文全文中译，保持原有结构与段落划分。技术术语首次出现时标注英文原文。
> **说明**：本翻译基于已下载的 arXiv HTML 全文。原文的数学推导部分在 HTML 抽取过程中公式与表格严重错位（§4.1、附录 8.1 尤甚），这些段落按可辨识的语义忠实翻译，公式以 LaTeX 重排，无法确定的片段已明确标注。

### 摘要 (Abstract)

量化是降低训练大语言模型（large language models, LLMs）GPU 显存需求的关键手段。然而，现有方法对 **4-bit 激活**和 **8-bit 梯度**均不奏效——它们很容易导致收敛变慢或精度损失。为解决这一问题，我们提出 AGoQ，包含两项新技术：1) 一个**层感知激活量化**（layer-aware activation quantization）算法，根据激活所属的层类型和流水线阶段（pipeline stage）为其分配合适的位宽，从而实现接近 4-bit 的激活存储；2) 一个**梯度量化**算法，通过采用 8-bit 梯度存储与**精度保持的 8-bit All-Reduce 通信**，同时降低显存占用并缩短通信时间。我们在两个 GPU 集群（最多 64 GPU）上使用不同规模的 LLM 进行了大量实验，结果表明：在 8B 到 32B 的 LLaMA 模型上，相对于最先进的训练系统 Megatron-LM（带或不带 ZeRO）、COAT 和 DeepSpeed，AGoQ 将显存降低最多 **52%**，训练速度提升最多 **1.34×**，同时在预训练上取得收敛的 loss、在 LLaMA 架构的下游任务上取得可比的精度。

### 1. 引言 (Introduction)

分布式训练已成为在多 GPU/TPU 集群上加速深度神经网络（DNN）训练的事实标准做法。其中，**数据并行**（data parallelism, DP）通过把训练数据分发到不同的 worker（或 GPU）上协同训练一个模型而被广泛使用。然而，随着大语言模型带来的模型规模显著增长，训练 LLM 的显存需求成为一个重大压力。因此，训练 LLM 通常需要使用模型并行，包括**张量并行**（tensor parallelism, TP）和**流水线并行**（pipeline parallelism, PP），它们把模型参数切分到不同设备上，使每个 GPU 都有足够显存存放所需数据。DP、TP、PP 已成为流行 LLM 训练框架 Megatron-LM 的默认特性。另一个流行的显存高效训练系统 DeepSpeed 利用**零冗余优化器**（zero redundancy optimizer, ZeRO）系列（ZeRO-1/2/3）来节省 LLM 训练的显存。PyTorch 生态中新引入的**完全分片数据并行**（fully sharded data parallelism, FSDP）与 ZeRO-3 思路相似。

训练 LLM 的设备显存占用主要由模型参数、梯度、优化器状态和临时激活构成。其中，**激活通常占据最大的显存比例**，并随序列长度与 batch size 的增长而线性增长——这两者都是常见的超参数。已有大量研究致力于通过减少激活占用来降低显存，包括**激活重计算或 offload** 以及**激活量化**。激活重计算（或 offload）是一种系统级优化技术，因此对模型精度没有副作用，但它引入了在反向传播时重算（或上传）激活的额外开销。激活量化则使用低精度格式（如 INT8、FP8）存储激活值，并在反向传播时反量化回 BF16/FP16。然而，量化与反量化过程（即使只用 8-bit 的 INT8 或 FP8）相比纯 BF16/FP16 仍会造成精度损失。Jetfire 和 COAT 尝试通过动态量化和分块量化来解决 8-bit 激活量化的精度损失问题，但它们仍**不适用于更低位宽的格式（如 4-bit）**。

此外，在梯度显存方面，尽管有大量工作试图压缩梯度以降低通信开销，但**梯度仍以高精度（FP32）存储、仅在通信时量化到低精度**，这意味着梯度依然占据与模型参数同等大小的显存。Microsoft 的一项值得注意的研究尝试用 FP8 格式存梯度，通过 scaling factor 来保持模型收敛；然而，由于 FP8 下**梯度累加**的精度损失，它在训练 LLM 时仍然容易导致收敛变慢。

为此，本工作旨在把激活量化与梯度量化再推进一步，使它们在 LLM 训练中真正可用。具体地，我们提出 AGoQ，其**激活量化**可使用近似 4-bit 精度，**梯度量化**使用 8-bit 实现显存高效的存储与通信高效的集合通信，且与优化器状态量化兼容而不牺牲模型收敛。为保持模型精度并提升系统吞吐，AGoQ 配备了若干新技术：1) 一个**层感知激活量化（LAAQ）**算法（§4），依据层类型与 PP 阶段为不同层的激活分配合适的位数，使激活的每个元素平均接近 4 bit；2) 一个名为 **QuanGrad** 的精度保持的量化梯度存储与通信算法（§5），用 8-bit 表示（FP8）存储梯度以供本地累加从而节省显存，并用于 All-Reduce 通信从而减少通信时间。

我们在 64-GPU 集群上使用不同规模（8B 到 34B）的 LLM 进行了大量实验，结果表明 AGoQ 相比包括 Megatron-LM（带或不带 ZeRO）、DeepSpeed 和 COAT 在内的最先进训练系统，将显存降低最多 52%、训练速度提升 1.34×，且不牺牲训练 loss 或精度。

### 2. 预备知识 (Preliminaries)

#### 2.1 Transformer 层

目前采用 transformer 架构的 LLM 最为流行，一个 transformer 通常由多个堆叠的 transformer 层组成。一个 transformer 层包含两个主要子组件：**自注意力机制**和通常带两个前馈网络（FFN）的**多层感知机**（MLP）。为便于理解 §4.1 的误差分析，此处列出它们的公式。

**Attention.** 注意力层由若干线性层构成，把输入投影为 query（Q）、key（K）、value（V）。缩放点积注意力计算为：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V \tag{1}$$

其中 $d$ 是 key 向量的维度。

**MLP.** Transformer 层中的 MLP 块由两次线性变换加中间的非线性激活函数构成。通常 MLP 先用第一个线性层把特征维度从 $M$ 扩展到 $4M$（LLaMA 模型上是 $8M$），再投影回原维度：

$$\text{MLP}(X) = W_2 \times \text{actfunc}(W_1 \times X) \tag{2}$$

其中 $W_1\in\mathbb{R}^{M\times 4M}$、$W_2\in\mathbb{R}^{4M\times M}$ 是权重矩阵，actfunc 是类似 SiLU 的激活函数。

**SiLU.** Sigmoid 线性单元（SiLU）是一个平滑的非单调激活函数：

$$\text{SiLU}(X) = X \odot \sigma(X) \tag{3}$$

其中 $\odot$ 表示 Hadamard 积，$\sigma(X)$ 是 sigmoid 函数。

**LayerNorm.** 层归一化（LayerNorm）对每个 token 独立地在特征维上归一化输入。RMSNorm 是最著名的 LayerNorm 变体之一：

$$\text{RMSNorm}(X) = \gamma \frac{X}{\sqrt{\frac{1}{d}\|X\|_2^2 + \epsilon}} \tag{4}$$

其中 $\|X\|_2^2 = \sum X_i^2$，$\gamma$ 是可训练参数，$d$ 是 $X$ 的元素个数，$\epsilon$ 是保证数值稳定的小常数。

对每个隐藏层，其输入 $X$ 必须在前向过程中保存，并在反向传播时复用以计算关于激活（对可训练层还有关于权重）的梯度。这一需求导致了可观的显存占用。**由于不同层执行不同类型的计算，我们观察到把 $X$ 压缩成低位格式会在最终的梯度计算中引入差异很大、且可能很大的误差**（细节见 §4.1）。

#### 2.2 并行范式

**数据并行（DP）** 把一个 mini-batch 的样本分发到多个 worker。在反向传播中，同一 DP 组内各 worker 的梯度通过一次 All-Reduce 操作聚合，使它们能用相同的梯度更新模型参数。All-Reduce 操作使用一个归约算子（训练中通常是 sum 或 mean）把分布在所有 worker（设 $P$ 个）上的梯度（设 worker $i$ 上为 $X_i$）累加起来，形式化表示为：

$$X = \text{AllReduce}(X_1, X_2, \dots, X_P) = \sum_{i=1}^{P} X_i \tag{5}$$

梯度与模型权重具有相同的维度，这意味着需要额外的显存来存储它们以供通信和模型更新。**用 8-bit 压缩梯度会因 AllReduce 求和的数据溢出而容易造成精度损失。**

**流水线并行（PP）** 是分布式训练中常用的模型切分策略。在 PP 中，模型的各层被分布到多个设备上。对由重复 transformer 块构成的模型，这通常意味着给每个设备分配相同数量的连续 transformer 层。为在一个 batch 内利用并行性，每个 batch 被进一步切分成更小的 mini-batch，这些 mini-batch 的执行随后在设备间流水化，使不同设备上不同 transformer 层的处理相互重叠。

为减少气泡，被称为 **Interleaved 1F1B** 的 PP 变体（如 Fig. 1 所示）在 Megatron-LM 中被广泛使用。在该方案中，一个 mini-batch 走完整个设备序列（从第一个到最后一个设备）之后，会被送回第一个设备再走一遍。这要求把模型切成更细粒度的段，并按 mini-batch 遍历顺序把这些段均匀放置到各设备上。

此外，Fig. 1 中每次前向都会在 GPU 显存里存下一部分激活以供后续反向使用，而每次反向在计算后释放其中一部分。**当采用 PP 时，各设备上同时存储的激活量是不同的**，这一点将在 §4.2 讨论。

*Fig. 1：四个 stage、每个 mini-batch 切成八个 micro-batch 的 Interleaved 1F1B PP 示例。*

### 3. AGoQ：系统概览 (System Overview)

为显著降低 LLM 训练的 GPU 显存占用，我们设计 AGoQ 把激活压缩到接近 4 bit、把梯度压缩到 8 bit，并且与 Megatron-LM 之上的 8-bit Adam 优化器兼容。如 Fig. 2 所示，AGoQ 引入两个新组件：**近 4-bit 激活量化**与 **8-bit 梯度量化**。

首先，对激活量化而言，前向过程先生成全精度激活，随后把它们量化并以约 4-bit 精度存储。在反向过程中，量化后的激活在计算梯度之前被反量化回 BF16/FP16。原理上，4-bit 激活只需 FP16/BF16 四分之一的显存。然而，**朴素地把所有层的激活都量化到 4 bit 会导致显著的精度退化**。为了既拿到 4-bit 的显存收益又维持模型性能，我们引入层感知激活量化（§4）。

其次，对梯度量化而言，在每个 GPU 上，local gradient 先由每个 mini-batch 的前向与反向计算得到，然后通过**本地梯度累加**与 main gradient 相加。这一过程包括：**把量化后的 main gradient（Q-Main Gradient）反量化，在高精度下把它与 local gradient 相加，再把和量化回 8-bit 格式后拷回 Q-Main Gradient。** 本地累加完成后，各 GPU 执行一次跨 GPU 的**精度保持的 FP8 All-Reduce**，如 §5 所述。

如 Fig. 3 所示（OLMo-1B 模型），我们比较了 BF16 基线、Transformer Engine（TE）、FP8-LM、COAT 和我们的方法在不同组件上的显存占用。AGoQ 在激活（§4）和梯度（§5）两侧都实现了进一步压缩：**相比 COAT，我们把激活显存额外降低 30%，梯度显存降低 75%。**

*Fig. 2：激活与梯度量化与 Megatron-LM 的集成。*
*Fig. 3：OLMo-1B 模型上的训练显存占用。*

### 4. 层感知激活量化 (Layer-Aware Activation Quantization)

为在最大化整体显存节省的同时最小化精度退化，我们首先通过理论分析识别**哪些层的激活适合 4-bit 压缩**，因为不同层类型（如 Attention、FFN、LayerNorm）表现出不同的计算模式（§2.1）。其次，PP 训练范式导致各 PP 阶段显存占用不均衡，这可以被利用来设计一个**动态量化补偿策略**，从而利用未被充分使用的显存资源。

#### 4.1 激活量化的误差分析

为最小化精度损失，我们做一次数值分析来确定不同类型的层应该量化哪些激活。

我们首先按"计算过程中是否需要保存输入之外的额外激活"把不同模块分成两类。MLP 中的**矩阵乘（GEMM）模块只需保存输入激活**。需要保存额外激活的模块包括 **RMSNorm、SiLU & Multiply 和 attention 模块**。为说明两类的区别，以 RMSNorm（Eq. 4）为例。令

$$r = \sqrt{\frac{1}{d}\|X\|_2^2 + \epsilon} \tag{6}$$

则可写作 $Y = \text{RMSNorm}(X) = \gamma X / r$。梯度矩阵表示为：

$$J = \frac{\text{diag}(\gamma)}{r} - \frac{1}{d}\frac{\text{diag}(\gamma)XX^T}{r^3} \tag{7}$$

因此，为计算梯度，我们需要同时存储 $X$ 和 $r$。这里 $r$ 就是也应被缓存的额外激活。当使用重计算技术时，我们不存 $r$；相反，在反向过程中先由 $X$ 重算出 $r$，再做梯度计算。

对需要额外激活的模块，我们主要分析两种情形下的梯度误差：

- **Case 1（重算中间量）**：只存储量化后的输入激活，原本需要的额外激活在梯度计算时由量化后的输入激活重算得到。
- **Case 2（缓存中间量）**：同时存储量化后的输入激活和量化后的额外激活。

在分析与 RMSNorm 或 SiLU 等操作相邻的 GEMM 计算时，我们具体比较两种梯度计算策略：一种只存储前置操作（如 RMSNorm/SiLU）的量化输入，在反向传播时由这些量化值重算 GEMM 的输入（同样称为 Case 1）；另一种直接以量化形式存储 GEMM 输入本身，从而避免重算（Case 2）。

以下推导中我们用到三个标准范数不等式。对向量 $X, Y \in \mathbb{R}^d$，其逐元素积的 $\ell_2$ 范数满足

$$\|X \odot Y\|_2 \le \|X\|_2 \|Y\|_\infty \tag{8}$$

对矩阵 $A\in\mathbb{R}^{m\times k}$、$B\in\mathbb{R}^{k\times n}$，谱范数是次可乘的：

$$\|AB\|_2 \le \|A\|_2\|B\|_2 \tag{9}$$
$$\|AB\|_2 \le \|A\|_2\|B\|_\infty \tag{10}$$

**4.1.1 RMSNORM**

RMSNorm 及其梯度由 Eq. 4 至 Eq. 7 定义。

*Case 1（重算中间量）*：只存储量化后的输入 $x$。我们假设量化引入的误差可建模为乘性扰动：

$$X' = X\odot(1+\delta_X),\quad r' = r(X'),\quad J' = J(X') \tag{11}$$

令 $\Delta J = J' - J$。做一阶展开：

$$r' = r + \Delta r,\quad \Delta r \approx \frac{1}{2r}\cdot\frac{2}{d}\sum_i X_i^2 \delta_{X_i} \tag{12}$$

我们得到：

$$\Delta\frac{1}{r} \approx -\frac{\Delta r}{r^2},\quad \Delta\frac{1}{r^3} \approx -\frac{3\Delta r}{r^4} \tag{13}$$

记 $D_\gamma = \text{diag}(\gamma)$ 并代入 $\Delta J$，用 Eq. 8 至 Eq. 10 可得：

$$\|\Delta J\|_2 \lesssim \|\gamma\|_\infty\left(\frac{|\Delta r|}{r^2} + \frac{3|\Delta r|}{dr^4}\|X\|_2^2 + \frac{2}{dr^3}\|X\|_2^2\|\delta_X\|_\infty\right) \tag{14}$$

其中 $\|\gamma\|_\infty = \max|\gamma_i|$。进一步代入 Eq. 12，得到

$$\|\Delta J\|_2 \lesssim 3\|\gamma\|_\infty\|\delta_X\|_\infty\left(\frac{\|X\|_2^2}{dr^3} + \frac{\|X\|_2^4}{d^2r^5}\right) \tag{15}$$

通常 $\epsilon$ 是常数且 $r\approx\sqrt{\|X\|_2^2/d}$，因此主导阶为 $O(\|\gamma\|_\infty\|\delta_X\|_\infty/r)$。

*Case 2（缓存中间量）*：同时存储量化后的输入 $X$ 与量化后的额外激活 $r$。此时误差表示为：$X' = X\odot(1+\delta_X)$、$r' = r(1+\delta_r)$ 被量化，其中 $|\delta_r|, \|\delta_X\|_\infty \le \varepsilon_q$。扰动后的 Jacobian 为

$$J' = \frac{D_\gamma}{r'} - \frac{1}{d}\frac{D_\gamma X'(X')^T}{(r')^3} \tag{16}$$

展开到一阶：

$$\frac{1}{r'}\approx\frac{1}{r}(1-\delta_r),\quad \frac{1}{(r')^3}\approx\frac{1}{r^3}(1-3\delta_r)$$
$$X'(X')^T \approx XX^T + X(X\odot\delta_X)^T + (X\odot\delta_X)X^T \tag{17}$$

因此

$$J' \approx \frac{D_\gamma}{r} - \frac{\delta_r}{r}D_\gamma + \frac{3\delta_r}{dr^3}D_\gamma XX^T - \frac{1}{dr^3}D_\gamma\left[XX^T + X(X\odot\delta_X)^T + (X\odot\delta_X)X^T\right] \tag{18}$$

减去精确的 $J = D_\gamma/r - D_\gamma XX^T/(dr^3)$，得到一阶扰动：

$$\Delta J \approx -\frac{\delta_r}{r}D_\gamma + \frac{3\delta_r}{dr^3}D_\gamma XX^T - \frac{1}{dr^3}D_\gamma\left[X(X\odot\delta_X)^T + (X\odot\delta_X)X^T\right]$$

用 Eq. 8 至 Eq. 10 取 2-范数界：

$$\|\Delta J\|_2 \lesssim \frac{6\|\gamma\|_\infty \varepsilon_q}{r} \tag{20}$$

**根据 Eq. 15 和 Eq. 20，我们得出结论：只存储量化后的输入激活并在梯度计算时重算中间量，与缓存中间量具有相同的渐近误差阶，只差常数因子——而且值得注意的是，重算方案的常数因子看起来更紧。** 鉴于重算同时还降低了显存占用，**对 RMSNorm 我们把输入激活存成 4-bit，并在梯度计算时重算其中间量。**

其他操作（如 SiLU & Multiply、RMSNorm+GEMM、Attention）的详细分析与 RMSNorm 类似，故放在附录 8.1，此处直接给出结论。

我们对每个 transformer 层的 **Q、K、V**、**RMSNorm 与 SiLU & Multiply 的中间激活**、**MLP 中两个 FFN 的输入激活**应用 Case 1 重算策略，从而消除这些值的存储。基于附录 8.1 的分析，**量化 attention 模块激活所导致的梯度误差显著大于其他模块，因此我们不量化 attention 模块的激活。** 其他激活通过 blocksize = 128 的 **block-wise FP4 量化**压到 4 bit。

如 Table 1 所示，我们的方法把 RMSNorm 显存从 4U 降到 0.5U（4-bit），把 SiLU & Multiply 激活从 12U 降到 2U。Attention 保持在 5U（与 COAT 所报一致）。总体上，激活显存从 Megatron-LM 的 **28U** 降到我们方法的 **7.75U**——约**三倍**的削减。

> 脚注 1：注意在 BF16 训练的 Megatron-LM 中，RMSNorm 仍使用 FP32 以获得更好的收敛性。

#### 4.2 动态位宽补偿

Interleaved 1F1B PP 导致各设备显存占用不均衡。如 Fig. 1 所示（四个 PP 阶段、8 个 mini-batch），不同设备存储的激活批次数不同——例如 **Device 1 峰值持有 11 份 mini-batch 激活，而 Device 2、3、4 分别只存 9、7、5 份**。这导致 GPU 显存显著未被充分利用，Device 1 占用的激活显存是 Device 4 的 **2.2×**。

为利用这些未被使用的显存，我们提出**面向流水线并行的激活动态位宽补偿**（Dynamic Bit-width Compensation for Activation with Pipeline Parallelism, **DBCA-PP**）。**存储较少激活批次的设备被分配更高的量化位宽**，从而在不增加峰值显存的前提下补偿量化带来的精度损失。该策略充分利用了流水线上原本被浪费的显存，同时维持接近 4-bit 的激活存储。

形式化地，在 Interleaved 1F1B PP 中，阶段 $i$（共 $n$ 个阶段）存储的激活 mini-batch 数 $N_i$ 可表示为：

$$N_i = n + 2\cdot i - 1,\quad 1\le i\le n \tag{21}$$

> **译注**：此式与上文"Device 1 持有 11、Device 2/3/4 持有 9/7/5"的描述方向相反（Eq. 21 随 $i$ 递增）。要复现论文实际使用的 4/5/6/8 位宽配置，需取 $N_1 = 11$，即 $N_i = 3n - 2i + 1$。原文此处存在印刷不一致。

基于各阶段的显存可用量，阶段 $i$ 的量化位宽 $B_i$（最小为 4）可设为与激活 mini-batch 数 $N_i$ 成反比：

$$B_i = 4\cdot\frac{N_1}{N_i},\quad 1\le i\le n \tag{22}$$

值得一提的是，**为较少阶段数生成的位宽配置也可以直接应用到阶段数更多的设置上**。换言之，在更高阶段数的设置中直接复用来自较低阶段数配置的位宽分配方案，不会增加更高阶段数设置的峰值 GPU 显存占用。

#### 4.3 量化/反量化与 GEMM 的 Kernel 融合

激活的量化与反量化需要额外的计算开销。为解决这一问题，我们把这些操作与邻近的 GEMM 计算**融合进单个 GPU kernel**。这样做的动机是：量化与反量化主要是逐元素操作，因此只使用 CUDA core；而 GEMM 在现代 GPU 上利用 Tensor Core。

为达成这一目标，我们如 Fig. 4 所示精心调度 LLM 训练中激活量化、反量化与 GEMM 操作的执行。**在前向过程中，我们把量化过程与其后继的 GEMM 操作融合。在反向过程中，我们把反量化与负责计算激活梯度的 GEMM 操作融合。** 这一做法几乎可以消除激活量化的计算开销，从而提升执行效率。

*Fig. 4：用于量化/反量化与 GEMM kernel 融合的 attention 与 MLP 前向反向过程。*

### 5. 精度保持的梯度量化 (Precision-Preserved Gradient Quantization)

为同时最小化存储梯度带来的显存占用和梯度 All-Reduce 期间的通信开销，我们引入一种 **8-bit 分块（block-wise）梯度量化**技术。该方法在 All-Reduce 操作全程保持精度，并有效缓解累加过程中发现的**两个不同的溢出问题**。

**第一，** 在 LLM 训练中，一个 global batch 通常被切分成多个 mini-batch，它们的梯度在与其他 DP worker 通信之前先在本地累加。**当我们用 FP8 存储 main gradient 时，直接累加来自不同 mini-batch 的梯度会很容易造成溢出。** 因此，在本地梯度累加中，我们把 FP8 main gradient **反量化到 FP16/BF16 以进行不同 mini-batch 梯度的高精度加法**，最终结果再量化成 FP8，如 Fig. 2 所示。

**第二，** 梯度需要通过一次 All-Reduce 操作在各 DP worker 之间聚合，而 All-Reduce 可以被拆分为 Reduce-Scatter 与 All-Gather。**然而，Reduce-Scatter 要求在通信过程中执行加法，这在 FP8 下极易造成溢出。** 因此，我们把 All-Reduce 操作**拆成一个 All-to-All 操作加本地 reduce，再接一个 All-Gather**，如 Fig. 5 所示。

如 Fig. 6 所示，FP8 梯度通过一次 **All-to-All** 通信把压缩后的数据发送到所有设备。每个设备随后把从不同设备收到的数据**反量化到 FP32**，执行**本地 reduce** 操作，然后把结果**重新量化**以供其后的 **All-Gather**。之后，我们对求和后的数据执行 All-Gather，从而完成整个 All-Reduce 操作。

*Fig. 5：我们通过组合 All-to-All 与 All-Gather 来执行 All-Reduce 的过程示意。*
*Fig. 6：我们把来自不同 GPU 的梯度合并的过程示意。*

### 6. 实验评估 (Evaluation)

#### 6.1 实验设置

**测试平台。** 实验主要在一个 **64-GPU 集群**上进行，该集群由 8 个节点通过 200 Gb/s InfiniBand 连接，每节点配备 **8 块 NVIDIA A6000 GPU**。在与 COAT 的对比实验中，我们使用两个节点、每节点 **8 块 NVIDIA Pro 6000 GPU**，以支持 COAT 所需的 FP8 格式。软件环境为 Ubuntu 20.04、CUDA 12.1、PyTorch 2.1.2、NCCL 2.18.5。我们的系统也支持华为 Ascend 910 NPU（更多结果见附录 8.2）。

**基线。** 我们在 Megatron-LM 之上实现 AGoQ，并与三个代表性基线比较：Megatron-LM（不带和带 ZeRO）、DeepSpeed、COAT。

**模型。** 由于训练成本极高，我们主要在 LLaMA2-7B 上做预训练实验以验证收敛性，并在更大的模型（LLaMA3-8B、LLaMA2-13B、CodeLLaMA-34B）上做训练时间对比。与 COAT 比较时，我们采用原 COAT 论文提供的 OLMo-1B 模型。

#### 6.2 端到端训练时间对比

**LLaMA2-13B.** 为评估方法的有效性，我们把 AGoQ 与不带 ZeRO-1 和带 ZeRO-1 的 Megatron-LM 比较，在 LLaMA2-13B 上以 32K 到 80K 的序列长度基准测试训练速度。在 GPU 显存受限的情况下，我们采用**选择性激活重计算**：不在前向后缓存全部中间激活，而是在反向传播时通过一次额外前向重算它们。使用重计算的 transformer 层数根据实时显存占用自适应调整。

Table 2 报告了 64 GPU、mini-batch size 1、global batch size 16、PP=4、TP=8 的结果。总体上，AGoQ 相对 Megatron-LM 取得 **1.22×** 的平均加速、相对其 ZeRO-1 变体取得 **1.21×**（我们这里关注 ZeRO-1，因为 ZeRO-2 与 ZeRO-3 会引入额外通信开销，通常降低吞吐，如附录 §8.2 所示）。加速比随序列长度增长，印证了我们方法的收益；例如在 80K token 下，AGoQ 比 Megatron-LM 和 ZeRO-1 都快约 **1.34×**。

**不同配置下的性能。** 为进一步评估，我们拓宽实验设置：序列长度从 16K 变到 32K 或从 32K 变到 80K，PP 从 1 到节点数，TP 取 4、8，GPU 数取 8、16、32、64。我们测试三个模型：LLaMA3-8B、LLaMA2-13B、CodeLLaMA-34B。详细配置汇总在附录 8.2，其中我们为 Megatron-LM、ZeRO-1、AGoQ 各运行 **124 组实验，总计 372 次**。其中 69 次因显存不足（OOM）失败（AGoQ、Megatron-LM、ZeRO-1 分别有 16、33、20 例），303 次成功完成。总体结果显示 AGoQ 相对 Megatron-LM 平均加速 **1.23×**、相对 ZeRO-1 **1.19×**。

我们接着分别考察 GPU 数、序列长度、模型规模和 PP 的影响。结果见 Fig. 7，表明 AGoQ 在不同 GPU 数和 PP 度下都持续显著优于 ZeRO-1 和 Megatron-LM。

**与 COAT 的对比。** 我们进一步用两个各 8 块 Pro6000 GPU 的节点（以支持 COAT 的 FP8 格式）把 AGoQ 与 COAT 比较。global batch size 为 64、序列长度 24K 和 32K，评测 OLMo-1B 模型。对 32K 序列，COAT 遇到 OOM，需要对一半的 transformer 层做重计算。Table 3 的结果显示：24K 下 AGoQ 相对 COAT 降低 31% 显存且训练速度相当；32K 下（COAT 已启用重计算）AGoQ 取得 **1.1×** 的端到端加速。由于硬件限制，我们只在 16 块 Blackwell GPU 上做了实验。预期在更大集群上相对 COAT 的加速会更高，因为 AGoQ 允许 8-bit 通信从而显著降低 DP 通信开销。

**商用带宽（如 100 Gbps）下的通信节省。** 我们在两种代表性带宽条件下评估通信效率：200 Gbps（我们的主测试平台）和 10 Gbps（模拟商用约束）。Table 4 报告了 TP=8、DP=8 下两种配置（200 Gbps / 10 Gbps）的延迟分解。在 200 Gbps、32MB 时，我们的分解方案取得 **3.4×** 加速（39.17 ms vs 131.23 ms）。在 10 Gbps 下，加速提升到 **3.7×**（432.36 ms vs 1603.78 ms），证实我们的方法在不同带宽设置下都能带来实质性的通信节省。

**Wall-Clock 时间分解。** 我们对单个 Transformer decoder 层做了详细计时分解（ms）。由于 Megatron 的序列并行（SP），反向过程中 All-Gather 的开销加倍。量化/反量化被融进 GEMM kernel。Table 5 和 Table 6 分别报告基线与 AGoQ 的分解。结果显示 AGoQ 在计算受限的操作上引入极小开销，同时有效节省显存。**融合后的量化/反量化增加的延迟可忽略**，体现在 FFN forward（13.3 → 15.64 ms）、FFN backward（28.6 → 32.54 ms）、Attn forward（19.3 → 20.6 ms）、Attn backward（45.1 → 47.8 ms）的小幅增长上。

**扩展的吞吐分析（2k/4k/8k）。** 我们把吞吐分析扩展到 2k、4k、8k 序列长度。当显存充足、不需要重计算时，我们建议**只启用梯度压缩**，因为它能有效降低通信开销。Table 7 展示了与 Megatron-LM 和 ZeRO-1 基线的吞吐（samples/sec）对比。我们的方法在所有序列长度上都持续优于两个基线，在 2k 序列长度下相对 Megatron-LM 取得最高 **1.33×**、相对 ZeRO-1 **1.16×** 的加速。

> **译注**：如 §2.4 所述，Table 7 的数值与此段文字矛盾——按表头（samples/sec）读，AGoQ 在三行中均为最低值。

#### 6.3 收敛 Loss

我们通过在 OpenWebText 的 **2B token** 上预训练 LLaMA2-7B 与 LLaMA3-8B 来评估 AGoQ 的收敛性。我们用两个模型上的不同 global batch size 来测试方法的鲁棒性：LLaMA2-7B 用 512、LLaMA3-8B 用 4。使用 **4 个流水线阶段**的 interleaved 1F1B 调度，我们按 Eq. 22 应用 DBCA-PP，各阶段激活位宽为 **4、5、6、8**，与统一 4-bit 压缩的峰值显存占用相当。

在训练 LLaMA2-7B 时，我们额外测试了 **FP8-AllReduce**（Microsoft 的 FP8 AllReduce 方法）的训练曲线。如 Fig. 8 所示，**我们的方法紧密跟随基线 loss，而 FP8-AllReduce 显示出显著更高的 loss。**

**不同优化的对比。** 我们还验证了激活量化、梯度量化、优化器量化三个模块各自的贡献，同时考察 DeepSpeed、ZeRO-1、ZeRO-2、ZeRO-3 之间的差异。在 PP=1、序列长度 48K 的配置下，我们在 LLaMA2-13B 上做了三种设置的实验："A+O+G"（即 AGoQ）、"A+O"（只应用激活与优化器量化）、"O"（只应用 8-bit 优化器量化），结果见 Fig. 9，表明**随着更多量化模块被纳入，迭代时间逐步下降**。它也显示训练速度从 ZeRO-1 到 ZeRO-2 到 ZeRO-3 递减，而我们的 AGoQ 显著优于 Megatron-LM、DeepSpeed 和 ZeRO 系列。

附录 8.2 中我们还给出了若干消融研究。

### 7. 结论 (Conclusion)

在本工作中，我们通过一种整体性的量化方法解决了 LLM 训练中 GPU 显存消耗这一关键挑战。具体地，我们提出 AGoQ，它整合了：1) 一个**层感知激活量化策略**，根据层类型和流水线并行阶段为激活存储分配合适的位数；2) 一个**梯度量化算法**，通过使用低位梯度存储与**精度保持的低位数据 All-Reduce 通信**来节省显存并减少通信时间。在两个 GPU 集群（最多 64 GPU）上的大量实验证明，AGoQ 相比全精度训练降低显存占用 **52%**，并把端到端训练吞吐相对最先进系统（包括 Megatron-LM、DeepSpeed、ZeRO 和 COAT）提升最多 **1.34×**，同时在 LLaMA 架构的下游任务上保持有竞争力的精度。

### 8. 附录 (Appendix)

#### 8.1 激活量化的误差分析（续）

我们对 §4.1 提到的 SiLU & Multiply、RMSNorm+GEMM、Attention 做误差分析。

**8.1.1 SiLU & MULTIPLY.** SiLU & Multiply 是一个接受两个输入 $X$ 和 $Y$ 的逐元素操作。对来自这两个输入的一对对应元素 $x$、$y$，该操作定义为 $z = xy\sigma(y)$，其中 $\sigma$ 是 sigmoid 函数。其导数为

$$\frac{\partial z}{\partial x} = y\sigma(y),\quad \frac{\partial z}{\partial y} = x\sigma(y) + xy\sigma(y)(1-\sigma(y))$$

扰动为 $x' = x(1+\delta_x)$、$y' = y(1+\delta_y)$。

*Case 1（重算中间量）*：渐近地，$|\Delta(\partial z/\partial x)| \le O(|y|^2|\delta_y|)$；$|\Delta(\partial z/\partial y)| = O(|x||\delta_x| + |x||y||\delta_y|)$。

*Case 2（缓存中间量）*：令 $\sigma' \approx \sigma(y)(1+\delta_s)$。渐近地，$|\Delta(\partial z/\partial x)| \le O(|y|(|\delta_y|+|\delta_s|))$；$|\Delta(\partial z/\partial y)| = O(|x|(|\delta_x|+|\delta_y|+|y||\delta_s|))$。

比较 SiLU & Multiply 的两种情形：在输入 $x$、$y$ 大多小于 1 的常见场景下，**Case 1（重算中间量）给出严格更小的误差上界**。对 $\partial z/\partial x$，Case 1 给出 $O(|y|^2|\delta_y|)$ 而 Case 2 给出 $O(|y|(|\delta_y|+|\delta_s|))$；因为 $|y|\le1$ 意味着 $|y|^2|\delta_y| \le |y||\delta_y| \le |y|(|\delta_y|+|\delta_s|)$，重算的界总是更紧。对 $\partial z/\partial y$，Case 1 的渐近界 $O(|x||\delta_x|+|x||y||\delta_y|)$ 也低于 Case 2 的 $O(|x|(|\delta_x|+|\delta_y|+|y||\delta_s|))$，因为后者含一个与 $|\delta_s|$ 成正比的额外项——而当 sigmoid 由扰动输入精确重算时该项不存在。因此，只要缓存误差 $\delta_s$ 不可忽略、且典型输入幅度满足 $|x|,|y|<1$，**在线重算中间量可证明地产生更小的最坏情况梯度误差**。所以对 SiLU 计算，我们只存量化后的输入激活，并在反向过程中重算必要的中间量。

**8.1.2 RMSNORM + GEMM.** 考虑 $Y = WU$，其中 $U = \text{RMSNorm}(X)$。关于 $W$ 的梯度为 $\partial Y/\partial W = U^T$，$U = \gamma X/r$。

*Case 1*：输入扰动 $X' = X\odot(1+\delta_X)$、$r' = r(X')$、$U' = \gamma X'/r'$。一阶展开得

$$\|U'-U\|_2 \le \|\gamma\|_\infty\left(\frac{\|X\|_2}{r^2}+\frac{\|X\|_2^3}{dr^4}\right)\|\delta_X\|_\infty$$

渐近地为 $O(d\|\gamma\|_\infty\|\delta_X\|_\infty/\|X\|_2)$。

*Case 2*：$U_c = U\odot(1+\delta_U)$，梯度误差界 $\|\Delta U\|_2 = \|U\odot\delta_U\|_2 \le \|U\|_2\|\delta_U\|_\infty$，渐近地为 $O(\|U\|_2\|\delta_U\|_\infty)$。

两种情形给出相同的渐近误差阶，只差常数因子。鉴于重算同时降低显存占用，**对 GEMM 计算我们只保留 RMSNorm 的量化输入激活**。类似地我们发现，对 SiLU 之后的 GEMM，只存 SiLU 的输入激活并在梯度计算时重算 GEMM 的输入，也会降低梯度误差的上界。

**8.1.3 ATTENTION.** 单头注意力为 $A = PV = \text{softmax}(S)V$，$S = QK^T/\sqrt{d}$。其梯度为

$$\frac{\partial A}{\partial V} = P^\top,\quad \frac{\partial A}{\partial P} = V^\top,\quad \frac{\partial A}{\partial S} = P\left(\frac{\partial A}{\partial P} - z\mathbf{1}^\top\right),\quad z = \text{rowsum}\!\left(P\frac{\partial A}{\partial P}\right)$$
$$\frac{\partial A}{\partial Q} = \frac{1}{\sqrt{d}}\frac{\partial A}{\partial S}K,\quad \frac{\partial A}{\partial K} = \frac{1}{\sqrt{d}}\left(\frac{\partial A}{\partial S}\right)^\top Q$$

扰动为 $Q' = Q\odot(1+\delta_Q)$、$K' = K\odot(1+\delta_K)$、$V' = V\odot(1+\delta_V)$。

*Case 1（重算中间量）* 与 *Case 2（缓存中间量）* 的各项界见原文；两种情形下量级均随 $\|Q\|_2\|K\|_2\|V\|_2$ 与 $1/d$、$1/\sqrt{d}$ 的组合缩放。

基于"重算中间量"（Case 1）情形下导出的误差界，我们比较 RMSNorm 与 attention 操作的梯度扰动。根据 Eq. 15 及后续分析，在标准 Transformer 假设下（$\|X\|_2^2 = \Theta(d)$、$r = \Theta(1)$、$\|\gamma\|_\infty = \Theta(1)$），RMSNorm 的梯度误差满足 $\|\Delta J\|_2 = O(\eta)$，其中 $\eta = \|\delta_X\|_\infty$ 是归一化的输入扰动水平。相反，**attention 的梯度误差随序列长度 $L$ 和 per-head 维度 $d_k$ 的缩放严重得多**：对 $\partial A/\partial V$ 我们得到 $\|\Delta(\partial A/\partial V)\|_2 = O(\eta L\sqrt{d_k})$，而对 $\partial A/\partial K$ 和 $\partial A/\partial Q$ 界增长到 $O(\eta L^{3/2}\sqrt{d_k})$。这些量分别比 RMSNorm 的误差大 $\Theta(L\sqrt{d_k})$ 与 $\Theta(L^{3/2}\sqrt{d_k})$ 倍。**这样的乘性差距解释了为什么把激活量化应用到 Q、K、V 投影会导致梯度误差被严重放大、训练不稳定，而 RMSNorm 可以安全量化。** 因此，对 attention 操作我们选择不施加激活量化。

#### 8.2 消融研究

**不同模型上的性能。** 为评估我们方法在不同模型上的提升，我们按 Table 9 的设置组织实验，在 LLaMA3-8B、LLaMA2-13B、CodeLLaMA-34B 上比较 ZeRO-1 与 AGoQ 相对 Megatron-LM 的加速（Fig. 10）。结果表明我们的方法在所有模型上都持续带来实质性收益。

**Zero-shot 精度。** 我们进一步评估在 2B token 上训练的 LLaMA2-7B 与在 10B token 上训练的 LLaMA3.2-1B 的 zero-shot 精度，使用 ARC-Challenge、ARC-Easy、HellaSwag、PIQA、SciQ、WinoGrande。Table 10 显示六个数据集上的平均精度保持在高位、无退化。

**显存占用削减。** 我们在 Table 8 中比较激活量化、梯度量化、优化器量化对显存削减的效果（配置见 Table 9：TP=8、PP=1、序列长度 12K 的 LLaMA2-13B，GPU 与 NPU 两侧）。可以看出每个量化模块都对降低显存有显著贡献。具体地，只应用 8-bit 优化器量化（"O"）在 GPU 上节省约 **8 GB**、在 NPU 上约 **10 GB**；而我们的激活与梯度量化在 GPU 上进一步节省 **13 GB** 与 **2.4 GB**，在 NPU 上为 **10.8 GB** 与 **4.8 GB**。值得注意的是，我们的 AGoQ（"A+O+G"）在 GPU 上相对 Megatron-LM 实现了峰值显存占用 **53%** 的削减，在 NPU 上为 **46%**。

**Kernel 融合的改进。** 我们测试 §4.3 中 GEMM 计算与量化/反量化操作 kernel 融合带来的加速。如 Table 11 所示，我们以 LLaMA3-8B 中若干主要 GEMM 形状为例（$C = A\times B$，$A\in\mathbb{R}^{m\times k}$、$B\in\mathbb{R}^{k\times n}$）。结果表明我们的 kernel 融合方案相对 GEMM 与量化/反量化的顺序版本取得平均 **1.07×** 加速，进而在 PP=1、TP=8、序列长度 16K 的 LLaMA3-8B 上带来 **1.05×** 的端到端加速。

**DBC（Dynamic Block Compression）的消融。** 我们做了带与不带 DBC 的训练对比。Table 12 显示 DBC 在多数任务上持续提升精度。值得注意的是，在低流水线并行度（PP）下配置的 DBC 可以应用到高 PP 而不增加峰值显存，凸显了其灵活性。

**层感知位宽的消融。** 我们把层感知混合精度量化与朴素地对所有激活施加统一 4-bit 量化做了比较。**统一 4-bit 基线未能收敛**，凸显了自适应精度的重要性。

**逐层梯度误差。** 我们测量了 AGoQ 对每层输入激活的梯度引入的误差（相对全精度梯度的平均绝对误差与归一化 L2 距离），采样自一个在 10B token 上训练过的模型。对 GEMM，我们额外比较了权重梯度的误差。Table 13 报告结果。**Attention 的归一化 L2 误差（0.14–0.15）是所有层中最大的，这与我们的理论分析一致，也解释了为什么我们把 Attention 排除在量化之外。** 对其他层，归一化误差最小为 0.003（LayerNorm）、最大为 0.051（SiLU），二者对收敛的影响都可忽略。

**32k/64k 序列长度下的显存削减。** 我们在显存约束下对 Llama3-8B 做了扩展序列长度的显存实验。32k 序列配 36 层；64k 因显存限制配 16 层。如 Table 14 所示，AGoQ 实现了实质性的显存节省，32k 下最多降低 **66%**、64k 下 **59%**，证明其在显存受限的长序列场景下的有效性。

**梯度范数监控。** 我们在 Llama3-8B 上做了一次每迭代 token 预算 32k、跨 **100,000 次迭代**的训练。Table 15 报告每 10k 迭代区间的平均梯度范数。**AGoQ 在整个训练过程中保持梯度范数与基线紧密对齐，收敛稳定。**

### 参考文献 (References)

参考文献从略。关键条目：Dettmers et al. (ICLR'22) 8-bit optimizers；Xi et al. (ICML'24) Jetfire；Xi et al. (ICLR'25) COAT；Peng et al. (2023b/c) FP8-LM；Wang et al. (ICLR'24) ZeRO++；Rajbhandari et al. (SC'20) ZeRO；Narayanan et al. (SC'21) Megatron-LM；Zhao et al. (VLDB'23) PyTorch FSDP；Li et al. (NeurIPS'23) 4-bit optimizer states。

---

## 五、对 MI455X / Helios 的参考点

**总论**：这篇论文对我们的价值不在它的方案，而在它把一个我们正要动手的问题的**约束条件写清楚了**——而且顺带暴露了我们对上游动机的判断需要修正。核心修正是：**"上游把 wgrad reduce-scatter 留在 BF16"这件事，不是因为他们没解决通信中做加法的溢出问题。他们解决了，用的正是本文这套 A2A + 本地 FP32 归约的结构，而且已经在 Megatron-Core GTP 里带 CLI flag 出货了。他们停在 BF16 的原因在别处，而那个别处正好是我们能拿的地方。**

### 5.1 AGoQ 的溢出诊断能否解释 GTP 把 wgrad RS 留在 BF16？——不能，而且这不是主因

**先确认我们那组数字是对的。** Megatron-Core GTP 文档的单权重单 micro-batch 通信预算表原文：

| Format | Block | Data B/elem | Scale_inv B/elem | Fwd AG | Bwd AG | **Wgrad RS (bf16)** | Total | vs BF16 |
|---|---|---|---|---|---|---|---|---|
| BF16 | — | 2.0000 | — | 2.0000 | 2.0000 | **2.0000** | 6.0000 | 1.00× |
| MXFP8 | 32 | 1.0000 | 1/32 = 0.0313 | 1.0313 | 1.0313 | **2.0000** | 4.0626 | 0.68× |
| NVFP4 | 16 | 0.5000 | 1/16 = 0.0625 | 0.5625 | 0.5625 | **2.0000** | 3.1250 | 0.52× |

文档原话："gradient is reduce-scattered in bf16 regardless of weight precision"，以及"the wgrad RS becomes the dominant comm path in NVFP4 (~64% of the budget at bf16 RS, ~78% at fp32 RS)"。**我们记的 64%/78% 精确无误**（$2.0/3.125 = 64.0\%$；$4.0/5.125 = 78.0\%$）。

**但溢出不是他们的理由，因为他们已经把这个问题解决了两遍。** GTP 文档 §2.6 有一个 flag：

```
--gtp-remat-reduce-scatter-with-fp32-accumulation      # default: off
```

文档解释："A ring reduce-scatter rounds the partial sum at every one of its `N-1` hops, so BF16 gradient error compounds with the axis size (≈√N for gradient-like data, worse when contributions share a sign). **This flag replaces it with an all-to-all plus one local FP32 sum, eliminating that accumulation error for the same bytes on the wire.**"

**这就是 AGoQ 的 Fig. 5，一字不差地在上游生产代码里**（`megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py`），还有一个 DP 轴的孪生 flag `--ddp-reduce-scatter-with-fp32-accumulation`。启用条件是"wgrads 是 BF16 且 gtp_remat 轴 ≥ 4"，轴 ≤ 2 自动绕过。而且 §2.7 还有第二条路：`--gtp-remat-nccl-ub` 用 NCCL 对称内存注册，让 RS 走单个 symmetric device kernel，在 NVLink 域内走 **NVLS multimem**——文档明说 "NVLS symmetric reduce-scatters accumulate in fp32 in-switch (NCCL's `multimem.ld_reduce` uses `.acc::f32` for bf16)"，并且**注册了对称池的 group 优先走这条、A2A+FP32 那条被绕过**。

所以 NVIDIA 的偏好顺序是：**交换机内 FP32 归约 > A2A + 本地 FP32 求和 > 朴素 ring RS**。"通信中做加法会掉精度"这个问题在他们那里是**已解决状态**，在 BF16 线宽上解决的。AGoQ 的诊断解释的是"为什么不能直接 `ncclReduceScatter(fp8)`"，它**不解释"为什么线上还是 BF16"**。

**真正的四个原因，按约束强度排序：**

1. **通信库里没有块缩放数据类型。** NCCL 支持 `ncclFloat8e4m3`/`e5m2`，但 **MXFP8 / MXFP4 / NVFP4 至今没有任何集合通信库支持**——这是 NCCL 上一个**仍未实现**的 RFE（issue #2199）。提问者自己把难点列为 "block-scaled datatype semantics, cross-rank scaling/alignment, reduction behavior, and numerical stability"。所以即便你想让 RS 直接吃 GEMM 已经产出的格式，那个原语不存在。这是最硬的约束，且它是**库层**约束，不是数值约束。
2. **在 NVL72 上他们没有动机。** NVLS 已经给了他们 FP32 精确的 RS，代价只是 BF16 的线宽；而 AG 侧的 packed MXFP8/NVFP4 已经把总预算从 6.0 砍到 4.06/3.13 B/elem。再往下要动 RS 格式，收益要跟"改一个已经工作的 in-switch 快路径"的风险比。**他们的硬件替他们买断了精度问题，所以他们只剩下字节问题，而字节问题的优先级低于把 AG 侧做稳。**
3. **per-shard 的 scale/amax 语义。** RS 的输出是一个**分片**。如果线上格式带 per-block scale，接收方拿到的是 $W$ 份各自带 scale 向量的碎片，归约结果的 scale 必须在 FP32 求和之后重新计算——这在 A2A 结构里是一次，在 ring 里是 $W-1$ 次，且每一次都改变载荷的含义。更要命的是**分片边界必须对齐 block 边界**，否则一个 block 被劈到两个 rank，两边都无法反量化。**GTP 里那个 `pad_for_alignment` 参数（MXFP8 = 32、NVFP4 = 16、BF16 = 1）就是这条约束的物化。**
4. **真正的数值障碍是尾数宽度，不是动态范围。** AGoQ 说"溢出"，这个说法不够准确，而且不准确的方向会误导设计。block scale 锁定的是**指数**，所以动态范围不是瓶颈；瓶颈是把 $W$ 个数加起来需要约 $\log_2 W$ 位额外尾数。$W=72$ 时是 **6.2 位**，而 FP8 E4M3 只有 3 位尾数、FP4 只有 1 位。**也就是说，即使 block scale 选得完美，每一次部分和都会丢掉低位。** 正确的表述是"**累加器必须永远比线上格式宽**"——这是个结构性论断，它立刻推出 AGoQ 的答案（本地 FP32 累加）是唯一答案，而"溢出"这个词让人误以为换个 scale 就能绕过去。

**结论**：AGoQ 的诊断方向正确、表述不准、且不完整。**上游留在 BF16 是因为库层缺格式 + 硬件已代偿精度，不是因为没想通结构。** 对我们的直接推论是：**我们要抢的缺口不是那个结构重排（公开、有文档、有代码，抄就是了），而是格式。** 而格式恰好落在我们控制的两样东西上：量化 epilogue（Primus-Turbo）和 grouped send/recv + 本地归约（Primus）。**更关键的一点：AMD 侧没有 NVLS/SHARP 等价物，所以 A2A 重构在 AMD 上不是可选优化，而是拿到 FP32 精确归约的唯一途径——这让它在 Helios 上比在 GB200 上更有价值，而不是更没有。**

### 5.2 A2A → 本地 FP32 归约 → AG 适合 72 GPU 单跳域吗？——适合，而且这是它第一次真正适合

**先把字节算清楚。** 设权重 $N$ 个元素、轴宽 $W$、线上 $b$ 字节/元素。

- **ring RS**：每 rank 每跳发 $N/W$，共 $W-1$ 跳 → 每 rank 出口 $b\cdot N\cdot\frac{W-1}{W}$
- **1-hop A2A**：每 rank 给其他 $W-1$ 个各发 $N/W$ → 每 rank 出口 $b\cdot N\cdot\frac{W-1}{W}$

**完全相同。** GTP 文档那句 "for the same bytes on the wire" 是精确的，不是近似。AllReduce 版本同理：RS+AG 与 A2A+AG 都是 $2b N\frac{W-1}{W}$。**所以结构重排本身在字节上既不赚也不亏——它的全部价值是"让线上格式可以变窄"，以及把 $W-1$ 次量化往返压成 1 次。** 这也直接说明 AGoQ Table 4 那个 3.4× 不是结构的功劳（见 §2.4 第 3 条）。

**$W=72$ 的具体账。** $\frac{W-1}{W} = \frac{71}{72} = 0.9861$。取 Helios 每 GPU scale-up 3.6 TB/s（双向，单向按 1.8 TB/s 算；$72\times3.6 = 259$ TB/s ≈ 公布的 260 TB/s 机架值，口径自洽）。取一个 LLaMA-3 级 FFN 权重 $N = 4096\times14336 = 58.72$ M 元素：

| 线上格式 | RS 字节/元素 | 每 rank 出口 | @1.8 TB/s |
|---|---|---|---|
| BF16 ring RS（今天） | 1.972 | 115.8 MB | **64.3 µs** |
| BF16 A2A（GTP §2.6） | 1.972 | 115.8 MB | 64.3 µs（**字节持平，只赚精度**） |
| **MXFP8 A2A** | 1.017 | 59.7 MB | **33.2 µs** |
| **NVFP4 A2A** | 0.555 | 32.6 MB | **18.1 µs** |

再加本地 FP32 归约的 HBM 代价：读 $W$ 个碎片共 $b\cdot N \approx 60$ MB，写 $N/W\times4 = 3.3$ MB，合计约 **63 MB** HBM 流量。这个 kernel 是纯逐元素归约、**没有转置**，所以不应该退化到我们在 `colwise_requant_mxfp8_grouped_flydsl` 上实测的那 ~35% 带宽效率；按乐观 70% 算约 **9 µs**，按悲观的 35% 算约 **18 µs**。

**结论：**

| 方案 | 总耗时 | vs BF16 ring RS |
|---|---|---|
| BF16 ring RS | 64.3 µs | 1.00× |
| MXFP8 A2A + 本地归约（不重叠） | 42–51 µs | **1.26–1.53×** |
| MXFP8 A2A + 本地归约（归约被重叠） | 33.2 µs | **1.94×** |
| NVFP4 A2A + 本地归约（不重叠） | 27–36 µs | **1.8–2.4×** |

**赢，但赢多少完全由那个本地 FP32 归约决定**——它的 HBM 流量和 A2A 的线上流量是同一个量级。这就是为什么 §5.4 的 RCCL 重叠能力是这条线的成败关键，而不是附带问题。

**为什么 72 单跳域是这个方案的转折点。** 这里有一条 ZeRO++ 已经踩过的坑：**flat 1-hop A2A 在跨节点时通信量会 blow up。** qgZ 的原文给了公式——每节点 $N$ 个 GPU、模型 $M$、量化比 $Z$，单跳 A2A 产生 $M\cdot N/Z$ 的跨节点流量，而层级化（先节点内 A2A + reduce，再节点间 A2A）把它降到 $M/Z$，**$N$ 倍差距**。ring RS 则可以按 rank 序排环，让绝大多数跳落在节点内，天然是层级化的。**所以在 8 GPU 域的机器上，一个 $W=72$ 的轴要跨 9 个节点，flat A2A 会输给拓扑感知的 ring RS，你必须补上 ZeRO++ 那层 2-hop 才能打平。**

**Helios 把这个坑填了：$W=72$ 恰好整个装进单跳非阻塞 scale-up 域，没有节点边界，flat 1-hop A2A 不需要任何层级化修补，且字节与 ring RS 精确相同、跳数从 71 降到 1。** 而 ring RS 那 71 跳既是数值上的 $\sqrt{71}\approx8.4$ 倍误差累积来源，也是延迟来源。**A2A 在单跳全 bisection 下同时是字节最优、跳数最优、数值最优的；它此前唯一的缺点（需要全对全带宽）被 Helios 的拓扑消掉了。** 另外 432 GB HBM4 让那个"未分片 wgrad 尺寸的临时 buffer"（GTP 文档列的成本项，MXFP8 下约 60 MB/权重）彻底不成问题——在 192 GB 的 MI300X 上这还需要算一算。

**边界条件（必须写进设计约束）**：把这套东西用在**跨机架**的轴上会立刻失效。in-rack 与 cross-rack 是 3.6 TB/s vs 597 GB/s ≈ **6:1**，flat A2A 跨机架的每 rank 出口是 $\sim bN$，比机架内贵 6 倍还不算 blow-up。**规则：低精度 A2A-RS 只用在 ≤72 的域内轴上**（这正是 NVIDIA 对 GTP 做的事——文档里 GB200 NVL72 的例子用 GTP64 ≤ NVL72，并把 DP 归约和 EGTP2 专家权重收发显式留给 IB）。**要跨机架就必须上 ZeRO++ 的 2-hop：机架内 A2A + FP32 归约（72 倍缩减），再对已归约的 FP32/BF16 分片做机架间 ring RS——跨机架流量降 72×。**

### 5.3 对 MX 格式意味着什么：两个 MX 张量根本不能不反量化就相加

这是三条里最技术、也最能定死设计的一条。

**MXFP8 的 block 是 32 个元素共享一个 E8M0（纯 2 的幂指数）scale；NVFP4 是 16 个元素共享一个 E4M3 scale，外加一个全张量 FP32 scale。** 现在问：能不能把两个 MX 量化过的张量直接相加？

**不能，有三层原因，每层都独立成立：**

1. **和的 block scale 是数据相关的。** A 的第 $i$ 块有 scale $s_{A,i}$，B 的第 $i$ 块有 $s_{B,i}$。和的第 $i$ 块要用什么 scale？必须由和的 amax 决定，而那要先算出和。**所以"MX 加法"的输出格式在算之前是未知的**——这不是精度问题，是**类型系统问题**：MX 不是一个在加法下封闭的数值类型。
2. **即使 scale 相同，尾数也放不下。** MXFP8 的 scale 是纯指数，两块 scale 相同时理论上可以按指数对齐直接加尾数。但把 $W$ 个数相加需要约 $\log_2 W$ 位额外尾数：$W=72$ 时 **6.2 位**，FP8 E4M3 有 3 位、FP4 有 1 位。**block scale 管住了动态范围，管不住尾数宽度。** 这是 §5.1 第 4 点的具体化，也是"溢出"这个说法为什么不够——真正丢的是低位，不是高位。
3. **ring 结构下误差会双重放大。** ring RS 要 $W-1$ 次 requantize，误差按 $\sqrt{W-1}$ 累积（GTP 文档对 BF16 就是这么算的）。$W=72$ 时 $\sqrt{71}\approx8.4$；而 MXFP8 的单次相对误差约 $2^{-4}$ 量级，BF16 约 $2^{-8}$，差 **16×**。两者相乘 → **原生 MXFP8 ring RS 的误差比 BF16 ring RS 差约 135×**。**这就是为什么没有人做原生 MX reduce-scatter，而且这个理由比 AGoQ 的"溢出"严格得多、也更有说服力。**

**四条设计推论：**

- **A2A + 本地 FP32 归约对 MX 不是优化，是唯一正确的结构。** 它把量化往返从 $W-1$ 次压到恰好 1 次：发送侧量化一次，接收侧反量化一次 + FP32 归约一次，需要的话再量化一次。**误差与轴宽解耦**——这正是让 $W=72$ 变得可行、而 ring 在同样轴宽下灾难性的原因。（ZeRO++ 早就把这句话写成了"we reduce the number of sequential quantization+dequantization kernel from the number of GPUs to 1"。）
- **分片边界必须对齐 block 边界。** 一个 32 元素的 MXFP8 块被劈到两个 rank，两边都拿不到完整块，谁都无法反量化。**必须 pad 到 32（MXFP8）/ 16（NVFP4）的倍数**，就是 GTP 的 `pad_for_alignment`。这条是可验证的硬约束，不是建议。
- **scale 元数据必须与数据同行、并被正确重分块。** MXFP8 的实际线上开销是 1 + 1/32 = **1.03125 B/elem**，不是 1.0；NVFP4 是 0.5 + 1/16 = **0.5625**。我们的带宽账必须用这些数，否则会高估 3%（MXFP8）到 12.5%（NVFP4）。
- **在 ZeRO-3 / GTP 语义下，AG 那条腿是多余的。** AGoQ 做的是 AllReduce（ZeRO-1 语义：每个 rank 都要完整梯度），所以有 A2A + AG 两条腿。**GTP / ZeRO-3 只需要 reduce-scatter——每个 rank 只要自己那片。于是整套东西塌缩成"A2A + 本地 FP32 归约"，AG 腿消失，那次 re-quantize 也消失。** 线上字节直接减半，且量化往返从 2 次降到 1 次。**这是本笔记对读者最有操作价值的一句话：不要照抄 AGoQ 的 Fig. 5，要抄它的左半边。**

### 5.4 Primus / Primus-Turbo 要建什么，RCCL 是不是拦路虎

**要写的 kernel（两个）：**

1. **wgrad GEMM epilogue 直接吐 MXFP8/NVFP4 + swizzled scale。** 这正是 roadmap 上"拆掉 quant↔GEMM 格式契约"的 S2/S3，见 §5.5。AGoQ 自己测到融合 quant/dequant 与 GEMM 有 1.03×–1.11×（均值 1.07×）的收益，与我们在 `rowcol_dual_quant` 上测到的 −9.4% 同一量级。
2. **`dequant_reduce` kernel**：读 $W$ 个 block-scaled 碎片（各 $N/W$ 元素 + scale 向量），在 FP32 里归约成一个分片，输出 FP32 或 BF16 `main_grad`。访存量约 $b\cdot N + 4N/W$，纯逐元素、**无转置**——这一点很重要，因为我们已知转置量化 kernel 只跑到约 35% HBM 带宽，而无转置的归约应该能上到 70%+。这是整条线的性能关键路径（§5.2 里 9 µs vs 18 µs 的差别就是它）。

**RCCL 原语：够用，但只够用。** `ncclAllToAll` 在 NCCL/RCCL 里本来就不是独立原语——GTP 自己也是用 `ncclGroupStart/End` 里的 grouped `ncclSend`/`ncclRecv` + `_coalescing_manager` 实现的。**所以原语层面没有缺口，我们不需要 RCCL 加任何新 API 就能把结构搭起来。**

**RCCL 是拦路虎，但拦的是重叠，不是字节。** 我们自己三种方式验证过 **RCCL 架构上无法 overlap**（collective 语义 + host 发起）：compute-only 0.44 ms、comm-only 0.39 ms、sequential 0.83 ms、理想 overlap 0.44 ms，而**实测最好 0.85 ms，只比串行快 3%**。而 GTP 那条路径依赖三件我们都没有的东西：`async_reduction=True`、`--high-priority-stream-groups` 给通信流 SM 优先级、以及跨 CUDA-graph 边界的 RS 重叠。

**这件事的严重性要说准确**：

- **字节收益是安全的。** A2A 与 ring RS 字节相同，把线上从 BF16 换成 MXFP8 就是实打实少一半流量，这部分不依赖任何重叠能力。
- **延迟收益一半悬空。** 如果本地 FP32 归约完全暴露，§5.2 的账是 1.26–1.53×（MXFP8）；如果能藏进下一层的 GEMM，是 1.94×。**中间那 0.4–0.7× 全押在 RCCL 上。**
- **有一个更难看的风险**：A2A + 本地归约在**调用次数**上比单次 `ncclReduceScatter` 多，而 RCCL 每次 collective 都要付 host 发起的固定开销。在小消息（AGoQ Table 4 的 32KB 行：A2A 0.35 + AG 0.41 = 0.76 ms vs AllReduce 0.83 ms，几乎没差）上这会把收益吃光。**所以必须按消息尺寸做 predicate，小权重走原生 RS，大权重走低精度 A2A**——这跟我们在 hipBLASLt vs 融合 kernel 上得到的结论同构（"融合路径与库路径有真正不同的最优区间，必须做 predicate 而非全局开关"）。
- 这条线因此**与 roadmap C3-4（推 RCCL 向 NCCL 2.28 对齐：Copy Engine collectives + device API）强耦合**。但它不阻塞：C3-4 决定我们拿到 1.5× 还是 1.9×，不决定我们能不能拿到 1.5×。

**另外一个 AGoQ 自己没看到、但对我们价值可能更大的机会。** AGoQ 的 Table 5 显示 TP/SP 的 `ag/rs` + `rs/ag` 占 baseline forward 逐层墙钟的 **69%**、backward 的 **59%**，而 AGoQ 对它**一个字节都没动**。SP 在 attention/MLP 输出投影之后的那个 activation reduce-scatter 是一个**货真价实的、在通信中做加法的 reduce-scatter**，TP=8。**同一套 A2A + 本地 FP32 归约的论证一字不改地适用，而目标比 AGoQ 优化的 DP 梯度归约大 3–6 倍。** 这条应该单独立项评估，优先级可能高于 wgrad RS。

### 5.5 与"拆掉 quant↔GEMM 格式契约"的交互：能砍掉 AGoQ 多少机器

如果 wgrad GEMM 的 epilogue 能**直接吐出下游要的低精度格式**（roadmap S2/S3 的目标），AGoQ 那套梯度机器会被砍掉相当大一块：

| AGoQ 的部件 | epilogue 直吐格式之后 |
|---|---|
| 发送前的 quantize kernel | **消失**（融进 GEMM epilogue） |
| 本地累加的 dequant → FP16 加 → requant 往返 | **大部分消失**：若 epilogue 能直接累进一个 FP32/BF16 的 `main_grad`（Megatron 本来就这么干），FP8 主梯度这个设计前提本身就没必要——它存在的唯一理由是省那 2.4 GB，而 2.4 GB 只占 AGoQ 总节省的 5.2% |
| A2A 之后的 dequant + FP32 归约 | **必须保留。** 它消费 $W$ 份远端碎片，GEMM 看不到它们，无法融合。这是不可约的核心 |
| AG 之前的 re-quantize | **消失**（ZeRO-3/GTP 语义下 AG 腿本身就不存在，见 §5.3 第 4 条） |
| block scale 对齐 / padding 契约 | **必须保留**，而且要提升为跨 op 的一等约束（`pad_for_alignment` 语义） |
| AG 腿 | **消失** |

**粗算下来，AGoQ 梯度侧的机器有 60–70% 变得不必要，剩下的是两件东西：一个 `dequant_reduce` kernel，和一个 padding 契约。**

这个结论对"格式缝"那篇的主张是**加强而不是削弱**：它说明 epilogue 直吐格式这件事的收益不止在反向 GEMM 的 1.29× → 1.6–1.8×，它还**顺手把低精度集合通信的一大半实现成本消掉了**。反过来说，如果不先拆格式缝，做低精度 wgrad RS 就必须自己带一个独立的 quantize kernel，而我们已知这类 kernel 只跑到 35% HBM 带宽、在 dW1/dW2 上合计吃掉约 1.15 ms。**两件事应该当成一件事来做，先后顺序是：先拆缝，后通信。**

同时要记住那条已知风险：**epilogue 加量化数学必然抬 VGPR，而 dW1 wgrad 已经在 248/512 VGPR、2 waves/SIMD 的边缘。** 如果撞墙，HipKittens 的 register pinning 是解药——这正是它该被取出来用的时机。

### 5.6 先做什么：分期、工作量、量什么

| 阶段 | 内容 | 工作量 | 量什么 / 验收 |
|---|---|---|---|
| **S0 可判性** | 在钉死的 GPU 上把目标 collective 重复 30+ 次，定出"最小可判差异"。**这一步不能跳**——本机 `--setperflevel high` Not supported、sclk 摆动 ±30%，我们已经因此否掉过"继续抠 GEMM"那条线 | 1–2 天 | 一个百分比数字 + 环境指纹入 CSV。**这是后续所有 claim 的许可证** |
| **S1 缺口测绘** | 量出 Primus 当前（Megatron DDP + distributed optimizer 路径）逐权重的 AG:RS 字节比与耗时比，对齐 GTP 那张 B/elem 表。**同时把 TP/SP 的 activation RS 一起量**（§5.4 末） | 3–5 天，纯 profiling | 一张我们自己的 per-element 通信预算表。**若 RS 占比远低于 GTP 的 64%，说明我们的瓶颈不在这里，整条线降级** |
| **S2 结构（BF16，零精度风险）** | 移植 `reduce_scatter_with_fp32_accumulation` 到 RCCL：grouped send/recv + 本地 FP32 求和。**线上仍是 BF16。** 这一步零精度风险、纯收益（误差不再随轴宽增长），且是后面所有事的地基 | ~1 周 | (a) 墙钟 vs `ncclReduceScatter`，扫 $W\in\{8,16,32,64\}$；(b) 梯度误差 vs FP32 参考随 $W$ 的曲线——**应该变平**；(c) 找出 predicate 的消息尺寸阈值 |
| **S3 格式（MXFP8）** | 线上换 MXFP8，`pad_for_alignment=32`。量化优先走 GEMM epilogue（S2 of 格式缝那篇），退路是独立 kernel | 2–3 周 | (a) 字节：目标 RS 从 2.0 → 1.031 B/elem；(b) 墙钟：目标 1.26× 以上（§5.2 的悲观端）；(c) **梯度 SNR ≥ 15 dB**——现有基线 dW1 19.4 / dW2 19.7 dB，**只有约 4 dB 余量，这是最可能翻车的地方** |
| **S4 跨域** | 若轴宽要超 72，上 ZeRO++ 的 2-hop：机架内 A2A + FP32 归约，机架间对已归约分片做 ring RS | 1–2 周 | 跨机架流量降 72×；端到端在 >72 卡上不回退 |
| **S5 扩展到 TP/SP** | 把同一套东西用到 SP 的 activation reduce-scatter（§5.4 末，目标大 3–6×） | 待 S1 数据决定 | — |

**杀死条件（写在前面，避免沉没成本）：**

- **S1 显示 wgrad RS 在我们的配置下不是主要通信项** → 这条线的前提不成立，转去做 TP/SP RS 或别的。
- **S2 显示 A2A + 本地归约在 $W=8$ 就慢于 RCCL 原生 RS，且本地归约无法重叠** → 收益被 RCCL 的 host 发起开销吃掉，**挂账等 C3-4**，不要硬推。
- **S3 的梯度 SNR 跌破 15 dB** → 4 dB 余量不够，需要退到更细的 scale 粒度或退回 FP8 per-tensor，收益要重估。
- **S0 结论是"<15% 不可判"** → 整条线在这台机器上无法验证，必须先解决测量环境。

**最后一句战略判断。** 这篇论文最大的用处是**否掉了一个我们本来会犯的错**：以为"通信中加法溢出"是上游没解决的问题、以为 A2A 重构是我们的差异化。**不是。结构是公开的、有文档的、上游已出货的；ZeRO++ 三年前就发表了；差异化只在格式，而格式的钥匙在 epilogue 里。** 好消息是那把钥匙我们本来就打算去拿（"拆掉 quant↔GEMM 格式契约"），而且 **AMD 没有 NVLS 这件事，恰恰让框架层的 A2A 重构在 Helios 上成为必需品而非可选项——这是一个我们被迫做、因此也理应拥有的位置。**

---

## 附录 (Appendix)

### A. 术语表 (Glossary)

| English Term | 中文 | 说明 |
|---|---|---|
| LAAQ (Layer-Aware Activation Quantization) | 层感知激活量化 | 按层类型 + PP stage 分配位宽；attention 排除在外 |
| QuanGrad | 量化梯度 | 本文梯度侧方案：FP8 存储 + A2A/FP32-reduce/AG 的 AllReduce |
| DBCA-PP | 面向流水线并行的激活动态位宽补偿 | 用 PP 的显存不均衡换精度，配出 4/5/6/8 bit |
| Case 1 / Case 2 | 重算中间量 / 缓存中间量 | 本文误差分析的两个对照策略；结论是 Case 1 的误差界更紧 |
| $U$ | 激活显存单位 | $1U = \text{BS}\times\text{SeqLen}\times\text{Hidden}\times 2$ bytes |
| qgZ | ZeRO++ 的量化梯度通信 | **A2A 替代 ring RS 的原始出处（2023）**，含 2-hop 层级化与 slice reordering |
| GTP (Generalized Tensor Parallelism) | 广义张量并行 | Megatron-Core 的 `TP × GTP_remat` 分解，即权重上的 ZeRO-3-on-top-of-TP |
| `pad_for_alignment` | 块对齐补齐 | GTP 参数；NVFP4 = 16、MXFP8 = 32、BF16 = 1。**块缩放格式做 AG/RS 的硬约束** |
| NVLS / `multimem.ld_reduce` | NVLink Switch 归约 | 交换机内归约，对 BF16 输入用 `.acc::f32`，即**硬件替框架解决了低精度累加问题** |
| Block scale | 块缩放 | MXFP8：32 元素共享 E8M0（1/32 B/elem）；NVFP4：16 元素共享 E4M3（1/16 B/elem） |
| wgrad RS | 权重梯度 reduce-scatter | GTP 中恒为 BF16 的 2.0 B/elem；NVFP4 下占单权重通信预算 64%（bf16 RS）/ 78%（fp32 RS） |

### B. 复现检查清单 (Reproducibility Checklist)

- [ ] **代码开源**：**否**。未给出仓库链接，也未说明是否计划开源
- [ ] **数据可得**：**部分**。OpenWebText 公开；但预训练只跑 2B / 10B token，不足以判断长程收敛
- [ ] **超参完整**：**部分**。给出了 TP/PP/GPU 数/序列长度/global batch size/DBCA-PP 位宽（4/5/6/8）/FP4 blocksize=128，但缺学习率、warmup、优化器超参、梯度裁剪阈值
- [ ] **随机种子**：**否**。全文未提及
- [ ] **硬件要求**：8 节点 × 8 A6000 + 200 Gb/s IB（主）；2 节点 × 8 Pro 6000（COAT 对比）；Ascend 910 NPU（附录）。软件 CUDA 12.1 / PyTorch 2.1.2 / **NCCL 2.18.5**（较旧，无对称内存支持，所以本文的 AllReduce 基线不含 NVLS 快路径）
- [ ] **数据自洽**：**存在四处问题**，其中 Table 7（吞吐与正文矛盾）和 Table 4 的 1 GB 行（10 Gbps 的 AG 比 200 Gbps 更快）是硬错误。详见 §2.4
- [ ] **关键对照组缺失**：无原生低精度 RS+AG 的对照，因此"重构比原生低精度 RS 更优"这一隐含主张未被验证（且据 GTP 文档，字节上二者相同）
- [ ] **相关工作缺失**：引用了 ZeRO++ 但从未讨论或对比 qgZ，而 qgZ 与本文梯度侧的核心机制同构
- [ ] **作者可信度**：通讯作者 Shaohuai Shi 是分布式训练/梯度压缩方向的活跃研究者（其 Shi et al., 2021 被本文自引）；Mengyang Zhang h-index 22。抓取到的元数据里若干作者 h-index 为 0，属抓取不完整，不应作为负面信号

### C. 一句话取舍

**激活侧（LAAQ + 误差分析 + Table 13）值得读，是本文的真贡献；梯度侧的结构不用抄——去读 ZeRO++ qgZ 和 GTP 文档 §2.6/§2.7，那里有更完整的版本和生产级实现。** 我们要拿的是格式，不是结构。

## 参考

- 战略母论点（格式缝）：[`../notes/career-strategy/2026-08-12_1505_the-quant-gemm-seam.md`](../notes/career-strategy/2026-08-12_1505_the-quant-gemm-seam.md)
- Roadmap C3-4（RCCL 对齐 NCCL 2.28）/ C3-5（原生低精度参数存储与通信）：[`../notes/primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md`](../notes/primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md)
- RCCL 无法重叠的三方验证：[`../notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md`](../notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md)
- 业界低精度训练 landscape（含 GTP / NCCL 2.28 / MLPerf v6.0 MXFP4 条目）：[`../knowledge/systems/industry-training-optimization-2026.md`](../knowledge/systems/industry-training-optimization-2026.md)
- MXFP4 预训练的精度风险（Wgrad +26–27% token 成本）：[`./mxfp4-pretraining-zh.md`](./mxfp4-pretraining-zh.md)
