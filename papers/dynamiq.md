# DynamiQ：用压缩的多跳 All-Reduce 加速梯度同步
# DynamiQ: Accelerating Gradient Synchronization using Compressed Multi-hop All-reduce

> **arXiv/DOI:** [2602.08923](https://arxiv.org/abs/2602.08923) · DOI [10.1145/3789240.3829148](https://doi.org/10.1145/3789240.3829148) | **HTML:** [v1](https://arxiv.org/abs/2602.08923v1) · [v3](https://arxiv.org/abs/2602.08923v3)
> **发表信息:** **ACM SIGCOMM 2026**（2026-08-17～21，Denver, CO；ISBN 979-8-4007-2467-1/26/08）。v1 是 2026-02-09 的 arXiv [cs.LG] preprint，v3 是 SIGCOMM camera-ready
> **机构:** Wenchen Han（UCL）、Shay Vargaftik（VMware Research by Broadcom；v1 署名为 Broadcom）、Michael Mitzenmacher（Harvard）、Ran Ben Basat（UCL + Broadcom）
> **代码:** <https://github.com/CharlesHan24/DynamiQ>（**v3 才给出**；v1 写的是"发表后开源"）
> **领域:** 梯度压缩 · 多跳 all-reduce · 随机量化 · 分布式均值估计 · PyTorch DDP
> **核心贡献:** 现有梯度量化方案都是为**单跳 parameter-server** 设计的；多跳 all-reduce 里同一个坐标会沿聚合拓扑被**反复部分求和再重压缩**，误差逐跳累积。DynamiQ 在带宽约束下直接最小化**部分和**的压缩误差：先用一次轻量 all-reduce 收集 super-group 统计量，据此给不同 super-group 分配 2/4/8/16 不同位宽（**依据是坐标在聚合后梯度里的量级，不是本地量级**），再用**融合的 decompress–accumulate–recompress kernel** 在每一跳重建量化范围。8 GPU / 4 worker 上 TTA 比 OmniReduce / THC / MXFP4-6-8 中的最优者快至多 **34.2%**、比 BF16 快 **40.8%**，且是唯一稳定达到 BF16 **99.9%** 精度的方案（均为 v1/v3 一致）。
> **阅读指引:** 与 MI455X / Helios 直接相关的判断在 **§五**（含机架内单跳 vs 跨机架多跳的定量分界）。

---

## 一、问题分析

### 1.1 研究背景

**领域现状。** LLM 的 DDP 梯度同步普遍走**多跳 all-reduce**（ring、butterfly / recursive doubling），而不是单跳的 parameter server。模型和 worker 数一起长，梯度聚合越来越成为瓶颈；多租户集群里多个作业抢网络又把瓶颈放大一层。梯度压缩因此是很自然的方向，但论文点出一个被普遍忽略的结构性错配：

> 绝大多数 SOTA 梯度压缩方案（THC、OmniReduce、Espresso、CUPCAKE、FP8-LM 等）**默认单跳 PS 架构**——worker 压缩、server 解压后用高精度求和，聚合本身**不占带宽**，所以"高精度累加"是免费的。

多跳里这个免费午餐没有了。中间节点拿到部分和之后只有两条路：

1. **重压缩**部分和继续往下传 → 精度逐跳退化，最终损害模型质量；
2. **加位宽**给部分和留表示余量 → 端到端加速被吃掉。

**核心挑战。** 部分和的量级会沿聚合路径增长（ring 上第 i 跳的部分和是 i 个 worker 的和），而固定格式的动态范围是静态的。THC 因此要求 `b ≥ ⌈log(15n+1)⌉`，n>8 时从 8 bit 涨到 12 bit，且这个策略只撑到 n=64；MXFP8-e4m3 靠 FP8-LM 的 μ 自动缩放循环（溢出率超阈值就 `μ←2μ`，长期不溢出就 `μ←γμ`）压制溢出/下溢，**但消不掉**。

**研究动机。** 作者自己的 HotNets'24（Han et al., *Beyond throughput and compression ratios*）先把这个问题提出来，DynamiQ 是给答案的那篇。

### 1.2 问题定义

- **输入**：n 个 worker 各持一份 BF16 稠密梯度 `X_i ∈ R^d`。
- **输出**：所有 worker 拿到一份对 `Σ_i X_i` 的无偏估计。
- **约束**：平均每坐标的通信预算 `b̄` 位。v3 明确写出（v1 无此定义）：对任意吞吐最优原语，网络上每 worker 每坐标要发 `2(n-1)/n · b̄` 位，**这个预算包含全部元数据**（scale、super-group 统计量），且 `b̄` 必须至少够放元数据 + 每坐标 1 bit。
- **度量**：time-to-accuracy（TTA）、最终精度、吞吐（rounds/s）、以及 vNMSE `E[‖X-X̂‖²]/‖X‖²`。

**问题形式化（reduce-scatter 部分，本文对读者最重要的一段）。**

n 个 worker 各把梯度切成 n 块，`C_{i,j}` 表示第 j 个 worker 的第 i 块。对每个块索引 `i`，集合 `{C_{i,j} | j ∈ [0,n)}` 被聚合，各个 `i` **并行**进行。论文的关键抽象是：

> 对每个 `i`，reduce-scatter 的聚合拓扑是一棵 **in-arborescence（内向树）**——所有边都指向唯一的 sink 的有根树。

- **Ring**：单块的聚合拓扑退化成一条**路径**（path），深度 n−1。
- **Butterfly / recursive doubling**：拓扑是深度 log n 的树（原文 Figure 13）。worker 3 持有 worker 0～3 的部分和（子树大小 4）发给 worker 7，后者持有 4～7 的部分和。
- 脚注补充：不同 `i` 并行聚合，所以在 ring 上每个节点同时是发送方和接收方，**整体通信图是一个环**——但单块的聚合图是路径，不是环。这个区分是整篇文章的立论基础。

节点按角色分工（v3 把这段写得比 v1 明确得多）：

| 角色 | 定义 | 动作 |
|---|---|---|
| **Leaf** | 对该块索引不接收任何外部消息 | 压缩本地块并发出 |
| **Internal** | 有若干父节点 | 对除最后一个父节点外的输入调 `decompress-accumulate`；收到最后一个父节点的部分和时调 `decompress-accumulate-recompress`，把和压好发给下一跳 |
| **Sink** | 该块的终点 | 解压并加上自己的本地块，reduce-scatter 结束 |

**这个形式化的payoff在 Appendix B**：每一跳的压缩误差正比于所传部分和的量级，而部分和量级正比于**该节点的子树大小**。于是

$$\text{ring: } \text{MSE} \le \sum_i \epsilon S i^2 M^2 = O(\epsilon S M^2 n^3), \qquad \text{butterfly: } \text{MSE} \le \sum_{l \le \log n} \epsilon S (M2^l)^2 \cdot (n/2^l) = O(\epsilon S M^2 n^2)$$

**butterfly 的上界比 ring 小一个 n 因子**。也就是说决定误差的不是"跳数"这一个标量，而是**内向树的子树大小分布**。这一点在 §五 会直接用上。

### 1.3 解决方案

**核心思路。** 理想情况是逐坐标按量级分配位宽，但这不可行（论文自己列了四条）：逐坐标传位宽的元数据开销爆炸；任意位宽破坏字节对齐、融合 kernel 写不了；非周期位宽破坏访存合并、偏移元数据又是一笔开销；**位宽沿聚合路径变化需要重打包，在融合 kernel 里做不了**。

DynamiQ 的折中是**两级分组 + 一次性全局定档**：连续 16 个坐标组成 **group**（共享一个 scale），连续 16 个 group 组成 **super-group**（256 坐标，共享位宽和统计量）。位宽在主 all-reduce 开始前一次定死，**全程不变**，从而彻底回避重打包。

**方法概述（六步，对应原文 Figure 2）。**

1. **(a) 本地统计**：每个 worker 对每个 super-group j 算均值 `μ_{i,j}` 与平方 ℓ2 范数 `F_{i,j} = Σ x²`。
2. **(b) 轻量 all-reduce**：只同步这两个标量，得到全局 `μ_j = (1/n)Σ_i μ_{i,j}` 和 `F_j = Σ_i F_{i,j}`。通信量通常 **< 原梯度的 1%**（v3 §3.5 精确到：`32/C_s` 位每坐标，`C_s=256` 时 = 0.125 位）。
3. **(c) 归一化 + 重排**：每个 super-group 减去全局均值变成零均值；按 `F_j` 定位宽；**把相同位宽的 super-group 排到一起**，使主 all-reduce 里 kernel 面对的是位宽一致的连续流。重排是本地 permutation，**零通信**。
4. **(d)(e) 主 all-reduce**：ring 或 butterfly，中间节点调融合 kernel。
5. **(f) 还原**：逆重排 + 加回均值。

**技术细节。**

*变量位宽分配（§3.2，消融里贡献最大的一项）*

- 允许位宽限制为 2 的幂 `W ⊆ {1,2,4,8,16}`（原型用 `{2,4,8}`），16 位等于不压缩，从而**统一了"outlier/top-k 精确编码 + 其余量化"这一类方案**。
- 用阈值 `T_{a,b}` 划分：`F_j ∈ [T_{a,b}, T_{b,c})` 的 super-group 用 b 位。
- 阈值关系由"每多一位最坏 MSE 降 4×"推出的**等边际收益**条件确定：每比特收益 `T_{a,b}(4^{b-a}-1)/(4^b(b-a))` 对所有阈值相等，得
  $$T_{1,2} = \tfrac{5}{32}T_{2,4}, \quad T_{2,4} = \tfrac{17}{512}T_{4,8}, \quad T_{4,8} = \tfrac{257}{2^{17}}T_{8,16}$$
  只剩一个自由度，二分搜索它以满足 `b̄`。
- Appendix A 给了避免排序的闭式近似：`q_j = 2^{clamp([1,3], ⌊log₂(4/log₂(512/17) · log₂F_j + u)⌋)}`，跨轮维护 `u` 并二分。

*非均匀量化（§3.3）*

沿用 Einziger et al. 的 Ice Buckets 取值集：`f(ε,r) = ((1+2ε²)^r - 1)/((1+2ε²)^{2^{b-1}-1} - 1)`，`Q = {f(ε,r)}`。ε 越大越向 0 附近密集，形似浮点。符号位单独编码。

*层次化量化（hierarchical quantization，§3.3）*

group 的 scale 若用 BF16 直传，group=16 时元数据开销太大。改为：**super-group 的 `sf_G = max|G|` 用半精度**，**每个 group 的 scale 用 UINT8 随机量化**到 `sf_G = r_G · sf_𝒢 / 255`。因为两级随机化独立，`E[x̂] = E[x̂'] · E[sf_G] = x`，**逐项无偏性保住**。

*相关舍入（correlated rounding，§3.3）*

不独立抽 `u_i`，而是 `u_i = (π_i + γ_i)/n`，其中 π 是 0..n−1 的随机置换、由共享 PRNG 种子隐式约定（**不通信**），`γ_i ~ U[0,1]`。于是每个区间 `[k/n,(k+1)/n)` 恰好落一个 worker 的 `u_i`，一个 worker 大概率上舍入时必有另一个大概率下舍入，误差相消。

*融合 kernel（§4）——本文的系统内核*

四个 CUDA kernel：

| kernel | 用在哪 |
|---|---|
| `DynamiQ_compress(t)` | leaf 节点压缩本地块 |
| `DynamiQ_decompress(ct)` | all-gather 阶段解压 |
| `DynamiQ_decompress_accumulate(ct, t)` | internal 节点收到"非最后一个"父节点时 |
| `DynamiQ_decompress_accumulate_recompress(ct, t)` | internal 节点收到最后一个父节点时，一并压好待发 |

**为什么瓶颈是显存带宽而不是线速**：GPU 上逐元素操作是 memory-bound，压缩开销由 global memory 事务数决定而非 FLOPs。不融合就要为每个坐标付多轮 HBM 往返。融合把中间结果留在寄存器里。Table 2 把这件事量化了（单位：字节/坐标的额外 DRAM 事务，`AR=(n-1)/n`）：

| 方案 | 额外 DRAM 事务 |
|---|---|
| BF16 | `4 + 4·AR` |
| **DynamiQ** | **`22 + 11.875·AR`** |
| MXFP8 | `18 + 13·AR` |
| THC | `74 + 2·AR` |

DynamiQ 与 MXFP8 基本持平；THC 因为随机 Hadamard 需要 `O(log d)` 轮 global memory 访问，常数项高 4 倍，**在 LLaMA 1B 上吃掉至多 42% 的单轮训练时间**。

另外三条工程细节：super-group 重排让 kernel 拿到位宽一致的连续流，访存合并和寻址都变简单；group/super-group 尺寸取 2 的幂；per-group max 用经典 CUDA 并行归约。

*RDMA engine（v3 新增）*

v1 只说"DDP comm hook 建在 NCCL P2P 上，支持 NCCL 原生没有的 butterfly"。**v3 补出了跨节点路径**：节点内用 NCCL，节点间用自建 RDMA engine；**分块流水**，一个梯度块一做完 `decompress_accumulate_recompress` 就开始 RDMA 传输，后续块继续处理，**块大小 8 MB**。

**溢出/数值范围处理（这条要单独讲清楚）。**

DynamiQ 的答案是**结构性的，不是靠调参**：每一跳都完整解压到寄存器精度、在寄存器里加、再**重新求一次 `sf_G = max|G|`**。也就是说量化范围在每一跳都按当前部分和重新标定，**根本不存在一个会溢出的固定累加器**。v3 §6.2 明确把这条列为大规模下赢 MXFP8 的两个原因之一：

> "its decompress–accumulate–recompress procedure updates the group scales as the partial sums evolve, allowing the quantization range to track the data well throughout the all-reduce. In contrast, MXFP8-e4m3 reserves four of its eight per-value bits for the exponent to represent a wide dynamic range uniformly."

对照组：THC 要靠 `b ≥ ⌈log(15n+1)⌉` 预留余量（n>8 时 12 位，只撑到 n=64）；MXFP8 要靠 FP8-LM 的 μ 自动缩放，且论文明说 microscaling **"do not eliminate overflow/underflow entirely"**。另外零均值化（减去全局 `μ_j`）也顺带缩小了要表示的动态范围。

v3 §8 还补了一条对训练稳定性有用的说法：super-group 位宽分配**天然支持把 outlier 保在训练精度**（把最高范数的 super-group 分到 16 位），从而避免"极值被塞进不安全的低精度表示 → NaN → 发散"。

---

## 二、实验效果

### 2.1 实验设置

| 项 | 内容（v3 口径；差异处标注） |
|---|---|
| 硬件 | 4 台 CentOS 服务器 × 2 × NVIDIA **RTX A6000**（48 GB GDDR6，**768 GB/s** 显存带宽），机内 NVLink NV4；每台 2 × 100 Gb/s Mellanox ConnectX-6，**实验只用一张**；GPU 经 PCIe 4.0 x16 接主机（约 32 GB/s/方向）；2 × AMD EPYC 7313（16 核）+ 512 GB 主存。**合计 8 GPU / 4 worker** |
| | **版本差异**：v1 写的是 "RTX A6000 **ada**"，且没有显存带宽、PCIe、双 NIC 的描述。v3 去掉了 ada 并给出 768 GB/s——这与 Ampere 代 RTX A6000 吻合（RTX 6000 Ada 是 960 GB/s），可以认为 v3 是**更正**。 |
| 负载 | 四个 **全参数 SFT**（v3 明确加了 "full-parameter supervised fine-tuning ... that produce dense gradients"）：BERT-large MLM @ Wikitext-103（21 epoch）；Gemma 1B @ UltraChat（3 epoch）；LLaMA 1B @ UltraChat；LLaMA 1B @ MMLU。扩展性另用 TinyBERT @ GLUE（8–64 worker） |
| 基线 | BF16 · OmniReduce（`b=8`，chunked top-k，作者自建 union-of-topk 启发式适配多跳）· THC（本地 `q=4` bit，聚合 `b=8`；n>8 时 12 bit）· MXFP8/6/4（E4M3 / E3M2 / E2M1，chunk 32，per-chunk BF16 scale，求和与溢出处理照 FP8-LM）。**MXFP4/6 在 A6000 上无原生支持，只报"最好情况下界"**（精度走软件、时间走等量流量不做计算） |
| DynamiQ 配置 | group `s=16`，super-group `S=256`，`W={2,4,8}`，默认 `b̄=5` 位/坐标；group scale = UINT8，super-group scale = BF16 |
| 度量 | TTA、最终精度、吞吐（rounds/s）、vNMSE |

### 2.2 主要结果

**TTA（ring，8 GPU / 4 worker；v1/v3 数字完全一致）**

- Gemma 1B：到目标 perplexity 比 MXFP8 / MXFP6 分别快 **18% / 28%**；THC、OmniReduce、MXFP4 要么比 BF16 还慢，要么根本到不了目标。
- LLaMA 1B MMLU：到 **72.38%**（BF16 最终 73.04% 的 99%）比 MXFP8 快 **34.5%**、比 BF16 快 **40.8%**。
- 所有场景最终精度与 BF16 差距 **≤ 0.1%**，其他方案退化至多 **2.5%**。
- 摘要口径的 "up to **34.2%** over the best among ..." 与 §5.1 的 34.5% 是**两个不同测量**（前者来自 Figure 4 的柱状图、对"各基线中最优者"；后者是对 MXFP8 在特定目标下的），不是笔误，但引用时要区分。

**压缩误差 vNMSE（Table 3，全程平均；v1/v3 一致）**

| 方案 | BERT-large MLM | LLaMA 1B Chat | Gemma 1B Chat | LLaMA 1B MMLU |
|---|---|---|---|---|
| **DynamiQ (5b)** | **0.00217** | **0.00149** | **0.00122** | **0.00096** |
| MXFP8 (8.5b) | 0.00591 | 0.00320 | 0.00308 | 0.00299 |
| MXFP6 | 0.02332 | 0.01350 | 0.01458 | 0.01298 |
| MXFP4 | 0.12080 | 0.11059 | 0.11583 | 0.09039 |
| OmniReduce | 0.15499 | 0.08044 | 0.04676 | 0.04530 |
| THC | 0.00897 | 0.11978 | 0.15168 | 0.19599 |

**用 5 位打赢 MXFP8 的 8.5 位，vNMSE 低 2.5–3×。** 注意 THC 那一行：在三个 LLaMA/Gemma 负载上比 MXFP4 还差，这对一篇 NSDI'24 的方法很反常，**大概率是"被搬到多跳"这个适配本身的产物**，见 §2.4 的批评。

**位宽消融（Table 4，LLaMA 1B MMLU / Gemma 1B Chat）**

| 方案 | MMLU vNMSE | MMLU 吞吐 | Gemma vNMSE | Gemma 吞吐 |
|---|---|---|---|---|
| DynamiQ 3b | 0.01603 | 3.051 | 0.02334 | 1.440 |
| DynamiQ 4b | 0.00589 | 2.842 | 0.00831 | 1.397 |
| **DynamiQ 5b** | **0.00096** | **2.604** | **0.00122** | **1.353** |
| DynamiQ 6b | 0.00059 | 2.390 | 0.00053 | 1.306 |
| MXFP8 | 0.00299 | 2.123 | 0.00308 | 1.246 |

5b 相对 MXFP8 吞吐 **1.23× / 1.09×**，vNMSE 同时低 3.1× / 2.5×。6b 精度继续降但吞吐掉得更多，所以选 5b。

**共享网络（额外起 3 个 DDP 进程持续做 ring all-reduce 抢带宽）**：Gemma 对 MXFP8 的优势从 16% → **21.5%**；LLaMA MMLU 从 34.5% → **40.2%**。有意思的观察是暴露通信时间不到隔离态的 4 倍——多个作业收敛到只在部分重叠的时间窗内发送。

**Butterfly（Table 5，LLaMA 1B MMLU）**

| 方案 | 最终精度 (%) | vNMSE |
|---|---|---|
| BF16 | 73.04 | 0 |
| **DynamiQ** | **73.04** | **0.00067** |
| MXFP8 | 72.86 | 0.00203 |
| MXFP6 | 72.46 | 0.02008 |
| MXFP4 | 71.59 | 0.17058 |

到 99% 目标比 MXFP8 快 **12.0%**，把目标提到 99.5% 时优势扩大到 **37.8%**。butterfly 下 DynamiQ 的 vNMSE 从 ring 的 0.00096 降到 0.00067，与 Appendix B 的 `n³ → n²` 一致。

**扩展性**：LLaMA MMLU 2–8 worker、TinyBERT GLUE 8–64 worker，DynamiQ 全程 vNMSE 最低。THC 与 OR 的 vNMSE 增长看起来更慢是假象——THC 是因为 n>8 时把位宽从 8 提到 12（且这招只撑到 n=64），OR 是因为它固定丢掉范数最低的 50% 梯度，误差被 sparsification 策略而非 n 决定。

**大规模模拟（v3 §6.2 + Table 7，全新，v1 完全没有）**

DP=8192、每 worker 32768 个 super-group（每个 256 坐标）；合成数据用拟合 Figure 1(c) 经验分布的 LogNormal 混合生成 super-group 量级，组内坐标 i.i.d. `N(0, M_𝒢²/256)`。

| `b̄` | 5 | 6 | 8.5 |
|---|---|---|---|
| DynamiQ-ring | 4.76 | 0.751 | **0.0105** |
| MXFP8-ring | – | – | 6.11 |
| DynamiQ-butterfly | 0.0336 | **3.23×10⁻³** | **2.75×10⁻⁵** |
| MXFP8-butterfly | – | – | 0.0353 |

同为 8.5 位时 ring 上低 **约 582×**、butterfly 上低 **1283×**；DynamiQ 5 位的 ring 误差（4.76）大致追平 MXFP8 8.5 位（6.11）。论文自己的建议很诚实：**这个 DP 维度下 ring 要用 `b̄=8.5`，butterfly 用 `b̄=6` 就够**——也就是 v3 摘要/引言里"DP=8192 时 6 位打赢 MXFP8 的 8.5 位"这句话，**成立的前提是 butterfly**。

### 2.3 消融实验

Table 6（group 32；启用层次化量化时降到 16）：

| 配置 | LLaMA 1B Chat | LLaMA 1B MMLU |
|---|---|---|
| 均匀量化 | 0.1278 | 0.1207 |
| 非均匀量化 | 0.0707 | 0.0664 |
| **+ 变量位宽分配** | **0.0198** | **0.0130** |
| + 层次化量化 | 0.0138 | 0.0092 |
| + 相关舍入 | **0.0091** | **0.0059** |

累计降低 **14× / 22×**。分项贡献：

- **变量位宽分配是绝对主力，单项 3.5–5.1×**，超过其余三项之和；
- 非均匀量化 ~45%，层次化量化 ~30%，相关舍入 ~35%。

**消融结论对我们最有价值的一条**：贡献最大的那一项——按聚合后范数给 super-group 分配 2/4/8/16 位——**与"多跳"这个前提无关**。它纯粹是在利用梯度的空间局部性和偏斜：LLaMA 约 20%、Gemma 约 30% 的 super-group，其范数比中位数低若干个数量级，给它们 8 位就是纯浪费。这条在单跳拓扑上照样成立，§五会用到。

### 2.4 需要打问号的地方

**论文自己承认的（v3 §8，v1 没有这一节）**：testbed 只做 fine-tuning，受硬件所限；DP 维度只到 64；百万卡级需要加位宽或分层聚合；生产预训练对低精度压缩天然保守，压缩误差可能表现为隐性不稳定。

**我认为要额外标出的**：

1. **testbed 的带宽结构极度有利于压缩。** A6000 显存 768 GB/s，线速 100 Gb/s = 12.5 GB/s，**比值 61:1**。压缩多付的显存流量在这个比值下几乎免费。现代节点的这个比值小一个数量级以上（见 §5.2 的定量表），**论文的加速比不可直接外推**。
2. **DP=8192 那组是纯模拟 + 合成数据。** 分布来自对 LLaMA 1B MMLU **单个梯度快照**拟合的 LogNormal 混合，且各 worker 的坐标是**独立**采样的。真实 DDP 里各 worker 看同一个模型、不同数据，梯度之间**高度相关**——这恰恰会影响相关舍入的收益和部分和的增长速率。这组数字应当当作"该合成分布下的上界"。
3. **基线是作者自己适配到多跳的。** THC 和 OmniReduce 都被搬出了设计点，论文很坦白，但 THC 在 LLaMA/Gemma 上 vNMSE 比 MXFP4 还差（0.12–0.196）这个结果很可能主要反映适配质量，而不是 THC 本身。相对优势因此被放大。
4. **每步多出两次全张量遍历和一次额外全局同步。** 归一化+重排、逆重排+反归一化各是一次完整 permutation；轻量 all-reduce 是主 reduce 开始前的一个**串行化同步点**，n 变大、消息变小时这个 latency 会更难忽略。Table 2 的常数项 22 vs BF16 的 4 就是这些遍历的代价。
5. **无收敛性证明、无 error feedback。** 无偏性保住了，但方差随 n 增长；论文只说"经验上位宽增长比 log 慢"，明确留作 future work。
6. **MoE / FSDP 只在 v3 §8 里"讨论"，零实验。** MoE 那条其实是个更大的机会（未激活专家的 super-group 可以给 0 位），但完全未验证——而 MoE 梯度的分布结构与稠密 SFT 梯度差别很大。
7. **确定性未讨论。** 随机舍入 + 共享 PRNG，不是 bit-wise 可复现的常规意义。训练场景这是硬需求。
8. **集成点是 PyTorch DDP comm hook**，也就是完整梯度的 all-reduce。Megatron 分布式优化器 / FSDP 的 wgrad reduce-scatter 是另一个集成点（v3 §8 说"可以只在 RS 末尾解压"，但没做）。

---

## 三、业界类似方案

### 3.1 方案对比表

| 方案 | 年份 | 核心思路 | 优点 | 缺点 | 关键数据 |
|---|---|---|---|---|---|
| QSGD | 2017 | 均匀随机量化 + 编码 | 无偏、有收敛保证 | 为 PS 设计；均匀取值集在偏斜分布上很差 | — |
| TernGrad | 2017 | 三值梯度 | 极致压缩 | 多跳下溢出 | — |
| PowerSGD | 2019 | 低秩分解 + error feedback | 压缩比极高 | 有偏；对 LLM 稠密梯度效果不稳 | — |
| OmniReduce | SIGCOMM'21 | 块级 top-k 稀疏 + in-network | 稀疏梯度上很强 | **LLM 稠密梯度基本没有稀疏性** | 本文实测 vNMSE 0.045–0.155 |
| THC | NSDI'24 | **同态压缩**：压缩域可直接相加；随机 Hadamard 做均匀化 | 中间节点零计算，可放进交换机 | 需要 `b ≥ ⌈log(15n+1)⌉` 的余量位；Hadamard 需 `O(log d)` 轮 HBM | 本文实测 Hadamard 吃掉至多 **42%** 单轮时间 |
| MXFP4/6/8 (OCP MX) | 2023 | 微缩放浮点，chunk 32 + 共享 scale | 硬件原生、生态标准 | 求和语义不在规范内；固定指数位为最坏动态范围买单；溢出/下溢消不掉 | MXFP8 = 8.5 位/坐标，vNMSE 0.003–0.006 |
| FP8-LM | 2023 | FP8 梯度 + μ 自动缩放 | 工程上简单 | 靠反馈环压制溢出，仍会退化 | 本文用作 MXFPX 的求和实现 |
| **AGoQ** | 2026 | **结构性回避**：AllReduce 拆成 A2A → 本地 FP32 归约 → 重量化 → AllGather | **通信期间完全不做低精度加法**，仅 1 次重量化 | 需要全 bisection；固定 8 位块量化，不分配位宽 | 显存 −52%，≤64 GPU 上比 Megatron-LM 快至多 **1.34×** |
| **DynamiQ** | **SIGCOMM'26** | 保留多跳，每跳 decompress-accumulate-recompress + **按聚合后范数变量分配位宽** | 溢出结构性不可能；5 位打赢 8.5 位的 MXFP8 | 每跳都要算；额外 HBM 流量高；额外一次全局同步 | vNMSE 0.001–0.002；TTA +40.8% vs BF16 |

### 3.2 技术路线对比

**路线 A：在压缩域里直接求和（同态 / in-network）**
- 代表：THC、SwitchML / ATP 一类 in-network aggregation、Terngrad。
- 思路：让压缩表示对加法封闭，中间节点（甚至交换机）不解压就能累加。
- 优劣：中间跳零计算、可下沉到网络设备；**但必须为路径上最大的那个部分和预留动态范围**，余量位随 `log n` 增长，整条路径为最坏情况买单。THC 的 `b ≥ ⌈log(15n+1)⌉` 就是这个税，且只撑到 n=64。

**路线 B：每跳解压、高精度相加、再压缩**
- 代表：DynamiQ、FP8-LM 风格的 MXFPX 适配。
- 思路：范围跟着部分和走，溢出结构性不存在，每跳误差可独立最小化。
- 优劣：**代价是每跳的计算和显存流量**——所以必须配融合 kernel，否则 memory-bound 直接吃掉收益；而且误差沿路径复合累积，这正是 DynamiQ 花全部力气去压的那部分。

**路线 C：让多跳消失**
- 代表：AGoQ；以及任何在非阻塞 fabric 上直接做 direct/one-shot reduce-scatter 的实现。
- 思路：`A2A → 本地 FP32 归约 → 重量化 → AllGather`，全程只有 1 次低精度加法边界，且加法在本地 FP32 做。
- 优劣：误差直接落到单次分布式均值估计的理论下界；**但要求 fabric 支持高效 all-to-all**（full bisection）。在 ring / 只有邻接链路的拓扑上不可行。

### 3.3 本文定位

- **相对路线 A**：DynamiQ 认为"压缩域求和"这条路在多跳上从根本上受限于动态范围，用逐跳重标定换掉了余量位。代价是中间节点要算，收益是 5 位干掉 8.5 位。
- **相对路线 B 内部的既有做法**：既有做法（MXFPX 适配）在每跳重压缩时用的是**同一个静态格式**；DynamiQ 的增量是**按坐标在聚合后梯度里的量级分配位宽**，并且靠"一次定档 + 重排"让这个分配在整条路径上一致、无需重打包。
- **独特贡献**：(a) reduce-scatter 的 in-arborescence 形式化 + 子树大小驱动的误差界（解释了为什么 butterfly 比 ring 好一个 n 因子）；(b) 融合 decompress-accumulate-recompress kernel，把路线 B 的计算代价压到与 MXFP8 同量级的 HBM 事务数；(c) 层次化 scale + 相关舍入这套 DME 技术栈搬进多跳。
- **没有回答的**：路线 C。论文完全没有把 AGoQ 式的"拆成 A2A + 本地归约"当作对照，而这在高 bisection 的 scale-up 域上是最直接的竞品。

### 3.4 推荐进一步阅读

| 论文 | 为什么值得读 |
|---|---|
| [THC](https://www.usenix.org/conference/nsdi24/presentation/li-minghao)（NSDI'24，同作者群） | 路线 A 的代表，本文最主要的对照。它的余量位公式 `b ≥ ⌈log(15n+1)⌉` 是理解"多跳为什么难"的最短路径 |
| [AGoQ](https://arxiv.org/abs/2605.00539)（2605.00539） | 路线 C。本文完全没对照的结构性替代方案，而它恰恰更适合 scale-up 域，见 §五 |
| Han et al., *Beyond throughput and compression ratios*（HotNets'24） | 本文的问题陈述来源，同一批作者。想理解动机读这篇比读 intro 快 |
| Suresh et al., *Correlated quantization for DME*（ICML'22） | 相关舍入的原始机制与方差分析 |
| [EDEN](https://proceedings.mlr.press/v162/vargaftik22a.html) / [DRIVE](https://arxiv.org/abs/2105.08339) | 同一批作者的 DME 线，DynamiQ 的量化技术栈都从这里来 |
| [MXFP4 原生预训练](./mxfp4-pretraining.md)（Penn State + AMD） | **必读，与本文直接冲突的证据**：MI355X 上 wgrad 量化才是收敛退化主因，且**随机 Hadamard / 随机舍入在全流水下直接不收敛**，确定性 Hadamard 才救得回来 |
| [Swing](https://www.usenix.org/conference/nsdi24/presentation/de-sensi)（NSDI'24） | ring 之外的带宽最优拓扑；本文只比了 ring 和 butterfly |
| Patarasuk & Yuan（2009） | 带宽最优 all-reduce 的经典分析，本文 §5.3 引它解释 butterfly 为什么误差也更小 |

---

## 四、全文翻译

> 以下为论文正文的中文翻译，保持原文结构与段落划分。**除特别标注外均据 v3（SIGCOMM camera-ready）；v1 与 v3 有实质差异处以脚注形式标出。** 技术术语首次出现时标注英文原文。图片内容与图注中的数据点不重复罗列（已在 §二整理）。

### 摘要

多跳 all-reduce（multi-hop all-reduce）是大模型训练事实上的骨干。随着训练规模增长，网络常常成为瓶颈，这促使人们去削减传输数据量。相应地，近期的系统已经用梯度量化（gradient quantization）显著加速了训练过程。然而，这些系统并未针对**多跳聚合**做优化——在多跳聚合中，条目会沿其聚合拓扑被多次部分求和。

我们提出 DynamiQ，一个弥合"量化最佳实践"与"多跳聚合"之间鸿沟的量化框架。DynamiQ 引入了更好地表示部分和（partial sums）的新技术，并与一个 decompress–accumulate–recompress 融合内核协同设计以实现快速执行。

我们扩展 PyTorch DDP 使其支持在 NCCL P2P 之上运行 DynamiQ。在不同的 LLM、任务与规模上，我们展示了相对 Omni-Reduce、THC 等 SOTA 方法以及 MXFP4、MXFP6、MXFP8 等新兴标准中的**最优者**，稳定取得至多 **34.2%** 的改进。更进一步，DynamiQ 是所评估方法中**唯一**稳定达到接近基线精度（例如 BF16 基线的 99.9%）的方法，且是在显著加速训练的同时做到的。

**关键词**：分布式训练、all-reduce、梯度压缩、量化、算法、数据并行、大语言模型。

### 1. 引言

分布式数据并行（DDP）是 LLM 训练与微调的标准范式。在该范式下，模型在各 worker 上复制，每个 worker 处理数据的不同部分以计算本地梯度。随后这些梯度通过网络同步（聚合）以获得全局更新。LLM 训练中的梯度聚合通常依赖多跳 all-reduce 方案，例如 ring 与 butterfly。随着模型尺寸与 worker 数量增长，梯度聚合日益成为瓶颈。近来在同一集群内运行多个作业、作业间竞争网络资源的实践，进一步加剧了这一瓶颈。

因此，旨在减少通信梯度数据量的**梯度压缩**是加速梯度聚合的自然且有前景的路径。尽管已有大量前期工作，我们观察到 SOTA 方案通常考虑的是（单跳的）parameter-server 架构——在那里，聚合（在解压之后）可以用更高精度进行，且**不带来任何带宽代价**。特别地，它们并未针对多跳 all-reduce 做优化：在多跳中，梯度沿聚合拓扑被部分求和。在这种方案里，中间节点面临两难——要么重压缩部分和，从而降低精度并最终损害模型性能；要么增加表示所用的位数，导致端到端加速有限。正如我们在第 5 节所示，这一局限既适用于已有的量化与稀疏化方案（例如 THC 与 OmniReduce），也适用于近期的微缩放浮点（microscaling FP）格式。

> **[v3 新增]** 本文聚焦数据并行，提出 DynamiQ——一个为多跳 all-reduce 量身定制、且**同时适用于预训练与微调**的压缩框架。（v1 此处只说"我们提出 DynamiQ，一个为多跳 all-reduce 量身定制的压缩框架"，没有关于预训练适用性的声明。）

DynamiQ 在带宽约束下最小化部分和的压缩误差，利用一个融合的 decompress–accumulate–recompress 内核来最小化显存带宽开销并促成压缩与通信的重叠。DynamiQ 优越的精度-带宽权衡，其关键在于它的**两阶段方法**：根据不同坐标在**聚合后梯度**中的量级，用不同的位数去量化它们。

理想情况下，我们希望按每个坐标的量级来决定其位数。然而这带来若干挑战：

- 传输每个条目的量化位宽会带来难以承受的开销。
- 任意的量化位宽会破坏字节对齐，使高效的融合内核无法实现。
- 非周期的位宽损害访存合并（memory coalescing）。此外，偏移元数据也显著增加显存与带宽开销。
- 沿聚合路径变化的位分配需要重打包（repacking），会拖累性能，因为这在融合内核里无法高效完成。

作为替代，我们的框架如下工作。DynamiQ 把连续条目合并成小的 **group**（例如 16 个条目），再进一步组成 **super-group**（例如 16 个 group）。group 与 super-group 共享元数据：group 有一个共享的 scale 参数，而同一个 super-group 内的所有条目使用相同的位宽。这一做法在位分配灵活性（通过改变 group 大小获得）与元数据开销之间提供了良好平衡。DynamiQ 执行一次**初始的轻量 all-reduce** 来收集关于 super-group 的必要统计量。这使所有 worker 就位分配达成一致，而该分配在整个聚合过程中保持固定。进一步地，所有 worker 随后按位分配对 super-group **重排序**，从而使主 all-reduce 中的融合内核调用作用在连续数据上。

为进一步优化带宽-精度权衡，DynamiQ 使用了若干高级量化技术，包括：

- **非均匀量化**——DynamiQ 对数据做归一化，并使用一组预先确定的非均匀量化取值，以优化逐项的乘性误差。直观上，这是通过使用更多靠近零的量化值、更少较大的量化值来实现的，类似于浮点格式。具体地，我们采用 Einziger et al. 提出的量化取值选择。
- **跨 worker 的负相关**——DynamiQ 使用相关舍入（correlated rounding）使误差更可能互相抵消。直观上，DynamiQ 用共享随机性来提高"一个 worker 向上舍入时另一个向下舍入"的概率，从而降低聚合误差。

我们通过一个运行在 NCCL P2P 之上的通信钩子（communication hook）把 DynamiQ 集成进 PyTorch DDP。我们在不同的 LLM 微调负载（BERT-large MLM、LLaMA-1B chat 与 MMLU、Gemma-1B Chat）与 all-reduce 拓扑（ring 与 butterfly）上评估 DynamiQ。DynamiQ 相对 OmniReduce、THC 与现代 FP 格式（MXFP4/6/8）中的最优者，把 time-to-accuracy 改进至多 **34.2%**。在若干设定下，DynamiQ 是所有负载中唯一达到接近基线精度（即相对 BF16 达 99.9% 最终精度）的方法，同时相比 BF16 加速训练 **40.8%**。

> **[v3 新增]** 在模拟设定中，我们展示 DynamiQ 能很好地扩展到大 DP 维度。例如在 DP 维度为 8192 时，DynamiQ 用 `b̄=6` 位/坐标的精度大幅优于 MXFP8（`b̄=8.5` 位/坐标）。我们的代码已开源。（v1 此处为"我们计划在工作发表后开源代码"。）

### 2. 背景

本节提供关于量化的必要背景。量化是把连续或高精度的取值集合映射到一个更小的离散集合的过程，本质上是减少表示一个数所用的位数。在基于分布式梯度的训练框架（例如用于 LLM 的分布式 SGD、ADAM 或 AdamW）中，每一轮训练都必须聚合来自不同 worker 的梯度以计算全局梯度。因此在 worker 侧施加梯度量化可以减少通信。然而挑战在于让量化既准确又快速，从而更快达到期望的模型精度——这个度量被称为 time-to-accuracy。

#### 2.1 无偏量化

梯度量化的一个重要性质是**无偏**。即给定梯度 `X ∈ R^d` 及其量化估计 `X̂`，我们希望 `E[X̂] = X`。

直观上，无偏性在做平均时是可取的：当一些值向上舍入、另一些向下舍入时，误差在期望意义上相互抵消。关键在于，在温和条件下，无偏量化能保证训练收敛。

无偏量化的基础方法是**随机量化（SQ）**。在 SQ 中，给定标量 `x ∈ R` 与两个量化取值 `x↓, x↑`（`x ∈ [x↓, x↑]`），我们通过下式获得 `E[x̂] = x`：

$$\widehat{x}=\begin{cases}x_{\uparrow} & \text{以概率 } \dfrac{x-x_{\downarrow}}{x_{\uparrow}-x_{\downarrow}}\\ x_{\downarrow} & \text{否则}\end{cases}$$

更一般地，当用量化取值集 `Q` 量化向量 `X ∈ R^d` 时，对每个 `x ∈ X`，记 `x↓ = max{q ∈ Q | q ≤ x}`、`x↑ = min{q ∈ Q | q ≥ x}` 为其两个最近取值，并施加上述 SQ。于是每个量化值 `x̂ ∈ Q` 可以用其量化取值索引以 `log₂|Q|` 位表示（例如每坐标 4 位时可用 `|Q| = 16`）。

#### 2.2 分组量化

为了反量化一个量化后的向量，接收方必须知道集合 `Q`。虽然许多先前工作为整次 all-reduce 调用选择单个 `Q`，我们选择为每个 **group**（例如 16 个条目的连续序列）选一个集合，因为这允许针对具体条目优化 `Q`。

逐 group 选择 `Q` 更准确有两个主要原因。第一，梯度往往表现出**空间局部性**——相邻条目倾向于有相似的量级。因此分组把 `Q` 裁剪到一个小的取值范围。第二，梯度分布本身是**偏斜的**，少数坐标（离群值）可能比其他坐标大若干数量级。这些离群值对集合 `Q` 及量化有效性有不成比例的影响。使用小 group 的逐 group `Q` 降低了这些离群值的整体影响。

直观上，由于上述原因，group 越小量化越准确，因为我们为每个具体 group 裁剪了 `Q`（及其大小）。但由于每个 group 都有开销（即其 `Q` 的编码），过多的小 group 会抬高所需带宽，反而背离压缩的目的。为降低编码开销，可以用 **super-group**（例如每 16 个连续 group）来共享部分元数据。

我们做了一个实验来例示空间局部性与偏斜性。为此，我们把原始梯度中 group 与 super-group 范数的分布，与随机打乱条目之后的这些分布做对比。直观上，如果不存在空间局部性，两个分布应当相似。我们分析对 LLaMA 1B 做 MMLU 微调、以及 Gemma 1B 做 Ultrachat 微调的第一个梯度。结果在 group 大小 16 与 super-group 大小 256 两个尺度上给出。**这种空间局部性造成显著比例的 super-group（例如 LLaMA 约 20%、Gemma 约 30%）其范数比中位数低若干个数量级，凸显了变量位宽分配的机会。**

#### 2.3 非均匀量化

给定输入 `X ∈ R^d`，`Q` 的常见选择是在范围 `[min X, max X]` 内均匀放置 `|Q|` 个量化值（例如 QSGD 与 Uniform-THC）。为在给定 `|Q|` 下优化精度，我们可以**非均匀**地放置它们。

例如考虑 `X = (-1, 1/2, 1)`；若要 `|Q| = 3`，均匀 SQ 会用 `Q = {-1, 0, 1}`，得到 MSE `Σ E[(x̂-x)²] = 1/4`。相反，取非均匀的 `Q = {-1, 1/2, 1}` 得到 MSE 为 0。更一般地，非均匀 SQ 的 MSE 可以渐近更低：例如对 `X = (-1, 1/2, ..., 1/2, 1)`（`1/2` 重复 `d-2` 次），均匀 SQ 的 MSE 是 `Ω(d)`，而非均匀 SQ 是精确的。

#### 2.4 负相关

直观上，我们可以把无偏量化的想法再推进一步，显式地"鼓励"误差相消。这通过在不同 worker 之间使用**共享随机性**来实现——让它们共享一个伪随机数生成器种子即可。使用共享随机性在量化工作中是常见实践。

例如考虑简单情形：两个 worker 各持 `x₁, x₂ ∈ [0,1]`，各需量化成 1 位，目标是估计 `x₁ + x₂`。标准做法是使用独立随机性——worker 各自生成 `u₁, u₂ ~ U[0,1]` 并按 `u_i ≤ x_i` 与否量化为 1 或 0。

为利用负相关，我们用共享随机性令 `u₂ = 1 - u₁`：两个 worker 生成同一个 `u₁ ~ U[0,1]` 但以不同方式使用，以提高它们朝相反方向舍入的机会。例如若 `x₁ = x₂ = 1/2`，独立随机性下 `Var[x̂₁+x̂₂] = 1/2`，而负相关下方差为 **0**。更一般地，对任意输入 `x₁, x₂`，负相关方法的方差至多为 `1/4`，即最坏情况方差改善 2 倍。

### 3. DynamiQ 框架

在第一阶段，每个 worker 把梯度切成 `S` 个条目的 super-group，并逐 super-group 计算元数据（图 2a）。接着，这些本地元数据通过一次初始 all-reduce 调用被聚合（图 2b）——这次调用是轻量的，因为它只包含 super-group 的均值与 ℓ2 范数之和，所以数据量通常**不到原梯度的 1%**。然后我们做逐 super-group 的归一化，并为不同 super-group 分配变量位宽。我们对梯度重排序，使具有相同位宽的 super-group 连续出现（图 2c），随后执行主 all-reduce（图 2d、2e）。最后我们后处理聚合数据：把 super-group 重排回原位置并加回其均值，得到同步后的梯度（图 2f）。我们同时提出并采用了若干降低压缩误差的技术，并用融合内核实现算法以最小化计算开销。

#### 3.1 获取 super-group 统计量

设有 `n` 个 worker。令 `X_{i,j}` 为 worker `i` 的第 `j` 个 super-group。对每对 `(i,j)`，worker `i` 先计算均值 `μ_{i,j} = Σ_{x∈X_{i,j}} x / |X_{i,j}|` 与平方 ℓ2 范数 `F_{i,j} = Σ_{x∈X_{i,j}} x²`。DynamiQ 随后用一次初始 all-reduce 聚合这些值。该阶段结束时，对每个 super-group `j`，所有 worker 都拥有全局均值 `μ_j` 与平方范数之和 `F_j`：

$$\mu_j = \frac{1}{n}\sum_{i=1}^{n}\mu_{i,j}, \qquad F_j = \sum_{i=1}^{n}F_{i,j}$$

一旦获得这些值，每个 worker 通过从 super-group `j` 的每个条目中减去 `μ_j` 来归一化数据，使其零均值；然后用 `F_j` 决定位宽并重排数据。

#### 3.2 确定 super-group 位宽

由于梯度分布的偏斜性，给范数更大的 super-group 分配更多位可以大幅降低量化误差。我们用变量位宽分配来最小化量化误差并满足任意给定的带宽约束。

> **[v3 新增]** 具体地，DynamiQ 接受一个参数 `b̄` 表示**平均每坐标位预算**。也就是说，对任意吞吐最优原语（例如 Ring、Butterfly），网络上每 worker 每坐标要发送的总位数为 `2(n-1)/n · b̄`。这既包含量化数据也包含任何附加元数据（例如 scale）。（注意 `b̄` 必须大到足以容纳元数据以及每条目 1 位。）

为允许高效的位打包，我们把可能的量化位宽限制为 2 的幂，即 1、2、4、8、16。这还有一个好处是简化了寻找高性能的变量量化。由于 16 位对应未压缩值，这**推广了**已有的"部分条目（如 top-k / 离群值）精确编码、其余量化到更少位数"的做法。

现在描述我们对不同 super-group 做带宽划分的快速启发式方法。对给定的允许位宽集合（例如 `W = {1,2,4,8,16}`），我们用阈值 `T_{a,b}`（`a,b` 在 `W` 中相邻）来表示具有相同分配的 `F_j` 值的边界。直观上，量化一个集合的 MSE 正比于其平方范数，因此 `F_j` 值可作为第 `j` 个 super-group 期望误差的代理。

令 `T_{0,1} = 0`、`T_{16,32} = ∞`。则所有 `F_j ∈ [T_{a,b}, T_{b,c})` 的 super-group 内的条目被量化到 `b` 位。接下来推导阈值之间的关系。假设我们从一组给定阈值出发，想以最能降低 MSE 的方式增加带宽——这可以通过降低某个选定阈值 `T_{a,b}` 来实现，使某些 super-group 的量化位宽从 `a` 提升到 `b`。

我们方法背后的直觉基于一个简单的最坏情况分析。考虑 `Q` 只含两个量化值 `0, 1`、条目 `x ∈ [0,1]` 的例子。`x̂` 的最坏方差出现在 `x = 1/2`，为 `Var[x̂] = 1/4`。现在假设我们把量化位宽增加 1 位（即把 `Q` 的规模翻倍）。这时我们可以在 `Q` 中每两个相邻值之间放一个量化值，包括在 0 和 1 之间。若这个额外的量化值位于 `1/2`，则最坏情况变成 `x = 1/4`，`Var[x̂] = 1/16`，即降低 **4 倍**。这个降低可推广到任意两个相邻量化值，即每增加一位，整个向量的最坏 MSE 可降低 4 倍。

假设我们把 `T_{a,b}` 降低到恰好使单个 super-group `j` 的条目从 `a` 位改为 `b` 位编码。上述直觉表明每增加一位 MSE 大约降低 4 倍。若该 super-group 此前的 MSE 正比于 `T_{a,b} · 4^{-a}`，则其 MSE 大约按 `T_{a,b}(4^{-a} - 4^{-b})` 的比例下降，同时该 super-group 内每条目带宽增加 `b-a` 位。因此我们把这一动作的**每比特收益**估计为 `T_{a,b}(4^{b-a}-1)/(4^b(b-a))`。为优化阈值选择，我们要求提升所有阈值的每比特收益大致相同（例如 `T_{1,2}·3/16 = T_{2,4}·15/256`），得到：

$$T_{1,2} = \tfrac{5}{32}T_{2,4}, \quad T_{2,4} = \tfrac{17}{512}T_{4,8}, \quad T_{4,8} = \tfrac{257}{2^{17}}T_{8,16}$$

注意这给了我们 `|W|-1` 个约束，只留下一个自由度（例如选定 `T_{1,2}` 后其余都被确定）。据此我们搜索 `T_{1,2}` 的取值并由上式确定其余阈值，使期望的带宽约束 `b̄` 被满足。

由于我们希望最小化寻找上述阈值的计算开销，对于至多使用三种允许位宽的实际情形（例如我们的实现用 `W = {2,4,8}`），我们在附录 A 给出一个基于二分搜索的快速解法。

#### 3.3 DynamiQ 的量化

该算法既用于压缩聚合拓扑上第一个块的数据，也用于对部分和做解压与重压缩。

**非均匀量化。** DynamiQ 使用分组量化，给定的 group `G` 用 `b` 位量化。每个 group 关联一个元数据，即缩放参数 `sf`。以 `b` 位每条目，我们可以用一个符号位加上 `{0,...,2^{b-1}-1}` 中的一个表示来编码每个条目。此后我们假设所有量化值 `q ∈ Q` 非负，因为符号位单独编码。我们非均匀地选择 `Q` 的取值（类似 Einziger et al.）。具体地，记

$$f(\epsilon, r) = \frac{(1+2\epsilon^2)^r - 1}{(1+2\epsilon^2)^{2^{b-1}-1} - 1}, \qquad Q = \{f(\epsilon,r) \mid r \in \{0,...,2^{b-1}-1\}\}$$

结果是某个 `Q ⊂ [0,1]`，其中 `ε` 影响量化值的非均匀程度。直观上，`ε ≈ 0` 时 `Q` 在 `[0,1]` 上大致均匀划分；更大的 `ε` 产生更多接近零的量化值、更少的大值。

注意由于 `G` 中的条目可以是任意 BF16 值，我们需要先把它们归一化到 `[0,1]` 才能随机量化到 `Q`；这通过把每个条目 `x` 除以 `max|G| ≜ max{|x| : x ∈ G}` 并单独编码 `x` 的符号来实现。

**层次化量化。** 发送方用一个缩放因子 `sf` 和逐条目的二元组 `(r, ς)` 来表示 group，其中 `r` 是表示、`ς` 是符号位。接收方相应地把该条目估计为 `ς · f(ε,r) · sf`。

自然的选择是把 group `G` 的缩放因子设为 `sf_G = max{|x| : x ∈ G}`。然而这需要以高精度（例如 16 位）传输 `sf_G`，当 group 大小 `s` 很小时会带来显著的带宽开销。

作为替代，DynamiQ 通过在 super-group 内量化缩放因子来优化精度-带宽权衡，这一方法称为层次化量化。即令 `𝒢` 为一个 super-group、`sf_𝒢 = max|𝒢|` 为 `𝒢` 中条目绝对值的最大者。我们把 `sf_𝒢` 以半精度为整个 super-group 编码，并用均匀随机量化去量化每个 group 各自的缩放因子，使 `E[sf_G] = max|G|`。例如我们可以用 UINT8 表示 `r_G ∈ {0,...,255}` 来表示 `sf_G`，解码为 `sf_G = r_G · sf_𝒢 / 255`。

我们的层次化量化的一个关键性质是**逐条目估计仍然无偏**。考虑某个条目 `x ∈ G`。它先被归一化为 `x' = x/max|G|`，随后被随机量化为 `x̂' ∈ Q`。如上所述，group 本身的 scale 也被随机量化。因此 `x` 的估计值为 `x̂ = x̂' · sf_G`，由于两步量化所用随机性独立：

$$E[\widehat{x}] = E[\widehat{x'}] \cdot E[sf_G] = (x/\max|G|) \cdot \max|G| = x$$

也就是说，即便量化 `sf_G` 会同时缩放 `G` 中共享该缩放因子的所有条目，逐条目的无偏性依然保持。

**相关舍入。** 直观上，我们希望提高"某个 worker 把一个特定部分和向上量化时，另一个向下量化"的可能性，使总体结果更接近真实和。

形式化地，令 `p_i` 为 worker `i` 对某个重压缩部分和向上舍入的概率。随机舍入通常通过抽取 `u_i ~ U[0,1]`、`u_i < p_i` 时向上舍入来实现。使用 Suresh et al. 的相关采样方法，我们不独立抽取 `u_i`，而是令

$$u_i = \frac{\pi_i + \gamma_i}{n}$$

其中 `π = {π_i}` 是 `{0,...,n-1}` 的一个随机置换，`γ_i ~ U[0,1]`。重要的是，`π` 由所有 worker 用同一个伪随机数生成器独立生成，因此是**隐式约定的、不需要通信**。

每个 `u_i` 仍服从均匀分布，但现在跨 worker 相关：在每个区间 `[0,1/n), ..., [(n-1)/n, 1)` 内**恰好落入一个 worker 的 `u_i`**。直观上，若某个 worker 的 `u_i` 落在 `[0,1/n)`（从而它大概率向上舍入），就必然存在另一个 worker `i'` 使 `u_{i'} ∈ [(n-1)/n, 1)`，从而 `i'` 大概率向下舍入，抵消误差。

#### 3.4 主 all-reduce

DynamiQ 的主 all-reduce 阶段遵循 ring 与 butterfly 等标准通信模式，但传输是压缩的。

**Reduce-scatter。** `n` 个 worker 把各自的梯度切成 `n` 块，`C_{i,j}` 为第 `j` 个 worker 的第 `i` 块。对每个索引 `i ∈ {0,...,n-1}`，块集合 `{C_{i,j} | j}` 被聚合，与其他 `i` 的块集合**并行**进行。对每个这样的 `i`，reduce-scatter 拓扑是一棵 **in-arborescence**，即所有边都指向单个 sink 的树。例如在 ring all-reduce 上，单个块的聚合拓扑就是一条**路径**；对 butterfly 我们在附录 B 的图 13 中可视化其拓扑。<sup>脚注：注意不同的 `i` 是并行聚合的；例如在 ring all-reduce 中，每个节点同时作为发送方与接收方，总体通信模式构成一个环。</sup>

聚合按如下方式工作：**leaf 节点**（对特定块索引 `i` 不接收任何外部消息）把其**压缩后的**本地块传给拓扑中的下一个节点。**internal 节点**作为中介，解压并聚合部分和（包括其本地块）、重压缩、发给下一个节点。而该块的 **sink 节点**通过解压并加上其本地块来结束其 reduce-scatter 阶段。

> **[v1 差异]** v1 此段远为含糊："leaf 节点**直接传输其本地块**……internal 节点作为中介聚合部分和，而 sink 节点结束其 reduce-scatter 阶段。"没有明说每一跳都发生解压与重压缩。v3 的措辞是理解整个溢出论证的关键。

**All-gather。** reduce-scatter 阶段完成后，all-gather 阶段开始，此时各 sink 把聚合和广播给所有其他 worker。

**融合内核。** DynamiQ 采用四种不同类型的融合内核，由累加状态与节点类型决定。第一种内核用于在 leaf 节点压缩块条目。internal 节点在收到除最后一个父节点之外的所有父节点的部分和时使用 **decompress-accumulate** 内核。当收到最后一个父节点的部分和时，它们施加 **decompress-accumulate-recompress** 内核，从而同时把和准备好用于下一次传输。

聚合后的压缩块和随后进入 all-gather 阶段被广播。每当一个节点收到压缩和，它就调用 decompress 内核。操作以最终的重建步骤结束：条目被恢复到原始顺序，并通过加回在初始 all-reduce 步骤中减去的均值来反归一化，从而恢复最终输出。

#### 3.5 通信与运行时开销 **[v3 全新章节]**

**通信开销。** DynamiQ 在两个 all-reduce 阶段产生通信。第一，在轻量 all-reduce 阶段，令 `C` 与 `C_s` 为 group 与 super-group 大小。对每个 super-group，DynamiQ 通过归约恰好计算两个 BF16 数 `μ_𝒢` 与 `F_𝒢`。因此该阶段每坐标通信 `32/C_s` 位。第二，在主 all-reduce 阶段，DynamiQ 从总通信预算中**扣除**轻量阶段所用带宽，因此主阶段每坐标使用 `b̄ - 32/C_s` 位。

举例：`b̄ = 5`、`C_s = 256` 时，第一阶段用 `32/256 = 0.125` 位，留给第二阶段 `4.875` 位/坐标（同时容纳量化表示以及 group 与 super-group 的 scale）。对大小 `C = 16` 的 group，group 的 scale 占 `8/16 = 0.5` 位/坐标，super-group 的 scale 再占 `16/256 = 1/16` 位/坐标。这就给每个量化坐标的表示平均留下 **4.3125 位**。这个位预算按 §3.2 在使用 2、4、8 位宽的 super-group 之间划分，使得平均值等于 4.3125。**重排阶段不产生任何通信**，因为它只是把 super-group 按位宽单调递增顺序本地置换进一个预分配张量。

**计算开销。** DynamiQ 的所有操作对梯度尺寸都是线性的，并且依赖对 HBM 的顺序访存。轻量阶段用**单次内核启动**、每条目一次访存计算所有 super-group 的 `μ_𝒢` 与 `F_𝒢`。重排阶段每条目一次读一次写，只有维护 super-group 索引的少量额外开销。主 decompress–accumulate 与 decompress–accumulate–recompress 操作各自实现为**单个融合内核**：每个值从其量化 bin 与 scale 就地重建、累加、再量化，**不物化任何中间张量**。相关舍入只增加本地算术，不需要额外访存。

### 4. 实现

我们在 PyTorch DDP 之上、以 NCCL P2P 实现 DynamiQ 原型。原型围绕四个 CUDA 内核：

1. `DynamiQ_compress(t)` —— 在聚合拓扑的 leaf 节点压缩梯度块 `t`。
2. `DynamiQ_decompress(ct)` —— 在 all-reduce 的 all-gather 阶段解压压缩梯度块 `ct`。
3. `DynamiQ_decompress_accumulate_recompress(ct, t)` —— 在非 leaf 节点融合"解压 `ct`、与 `t` 累加、重压缩和"三步。
4. `DynamiQ_decompress_accumulate(ct, t)` —— 在中间跳执行 `ct` 的解压并与 `t` 累加（不重压缩）。

**高效的融合内核 CUDA 实现。** GPU 对逐元素操作通常是 memory-bound 的，这意味着压缩开销主要由全局内存事务决定。因此我们的设计通过融合内核最小化这一开销。

以 `DynamiQ_decompress_accumulate_recompress` 为例，它融合这三个操作。中间结果存放在**寄存器**中，避免全局内存访问。如表 2 所示，这显著降低了内存流量，使 DynamiQ 那套"复杂"的逻辑在计算上变得轻量。

super-group 重排使 GPU 内核接收到的是若干条位宽一致的连续条目流，从而实现高效的内存寻址与合并。为共享用于缩放的逐 group 梯度最大值，我们使用 CUDA 中经典的并行最大值归约算法。我们对 group 与 super-group 大小取 2 的幂，以获得更有效的线程访存与执行。

**DDP 通信钩子与 RDMA engine。** 我们用 P2P 原语构建通信钩子：**节点内用 NCCL，节点间用 RDMA**。这使得 NCCL 集合通信原生不支持的 all-reduce 拓扑（包括 butterfly）成为可能。此外，我们的 RDMA engine 用**分块流水**把压缩与通信重叠：一个梯度块一旦完成 `decompress_accumulate_recompress`，它的 RDMA 传输就开始，同时后续块继续被处理（实验中块大小为 **8 MB**）。

> **[v1 差异]** v1 只写"在 PyTorch DDP 之上以 NCCL 作为集合通信后端"，通信钩子一段也只说建在 NCCL P2P 原语上、支持 butterfly、并流水化计算与通信，举的例子是 all-gather 阶段"解压 `i` 的同时把 `j` 转发给下一跳"。**跨节点 RDMA engine 与 8 MB 分块流水是 v3 才写出来的。**

### 5. 评估

本节给出 DynamiQ 在四个 LLM 全参数有监督微调负载（产生**稠密梯度**）上的端到端 testbed 评估，并与 SOTA 梯度压缩方案对比。我们先用 ring all-reduce 拓扑，分别在隔离环境（§5.1）与共享网络（§5.2）下评估性能。随后为展示 DynamiQ 对不同拓扑的适用性，我们考虑 butterfly all-reduce（§5.3）。

*（testbed、负载、参数、基线、DynamiQ 配置、度量的具体内容见 §二·2.1；此处不重复。）*

#### 5.1 Ring all-reduce

DynamiQ 相对 BF16 提供显著加速，且 TTA 曲线明显优于所有其他被测压缩方案。重要的是，**DynamiQ 在所有负载上都优于 MXFP8，尽管它使用更低的位预算**——这归因于它既能保持低压缩误差、又在其"复杂"的两阶段工作流下只产生很小的计算开销。

**Time to accuracy。** 例如在 Gemma 1B 上，DynamiQ 达到目标 perplexity 比 MXFP8 与 MXFP6 分别快 **18%–28%**，而 THC、OmniReduce 与 MXFP4 要么收敛比 BF16 基线还慢，要么因为压缩误差过大根本达不到目标精度。类似地，对 LLaMA 微调，DynamiQ 达到 **72.38%**（BF16 精度的 99%）比 MXFP8 快约 **34.5%**、比 BF16 快 **40.8%**。关键在于，DynamiQ 在所有场景下把最终精度维持在未压缩 BF16 的 **0.1%** 以内，而其他压缩方案表现出至多 **2.5%** 的退化。

**吞吐。** DynamiQ 的 TTA 收益部分归功于其改进的训练吞吐。DynamiQ 的压缩开销很小，因为 GPU 上的梯度压缩通常是 memory-bound 而非 compute-bound。相应地，通过利用融合内核，DynamiQ 确保了合并的访存模式——每个梯度坐标只被访问一次，从而在内存事务量上与 MXFP8 持平（见表 2）。相反，THC 使用的随机 Hadamard 变换需要 `O(log d)` 次额外的 GPU 全局内存访问，制造出一个瓶颈，在 LLaMA 1B 负载上**消耗至多 42% 的单轮训练时间**。

**压缩误差。** DynamiQ 对未压缩梯度表现出更好的保真度，vNMSE 比 MXFP8 低 **2.5–3×**，比 MXFP4、THC 与 OmniReduce 低若干数量级。这进一步澄清了此前观察到的性能权衡：MXFP4 虽然吞吐更高，但其过大的误差拖慢收敛并降低最终精度。类似地，OmniReduce 在这些基准上表现不佳，是因为它依赖梯度的稀疏性与偏斜性（即大量接近零的条目），而这一属性在稠密的 LLM 梯度中基本不存在。

**位预算消融。** `b = 5` 确实达到了这一场景下的最佳权衡。低于该阈值会增加压缩误差并降低最终精度；提高 `b` 不带来精度收益，反而因通信量增加而降低吞吐。

#### 5.2 共享网络下的 ring all-reduce

在很多情形下训练作业并非隔离运行，而必须与其他作业或租户共享网络。相应地，本实验中我们额外启动三个持续执行 ring all-reduce 的 DDP 进程与训练作业争抢带宽。

结果如预期：压缩方法（尤其是 DynamiQ）相对 BF16 基线的 TTA 优势在带宽争用下**增大**。例如在 Gemma 1B + Chat 上，DynamiQ 相对 MXFP8 的优势从隔离时的 16% 增至共享网络下的 **21.5%**；在 LLaMA 1B + MMLU 上，优势从 34.5% 增至 **40.2%**。有意思的是，暴露的通信时间**短于**隔离时的 4 倍，因为不同作业收敛到只在部分重叠的时间窗内传输。

#### 5.3 Butterfly all-reduce

我们继续用 butterfly all-reduce 做实验，它把跳数降到关于 worker 数的对数，从而降低延迟。有意思的是，它同时也降低量化误差——因为需要的重量化次数更少，且聚合路径上被求和的部分和往往量级更接近。

在 LLaMA 1B MMLU 上，DynamiQ 取得比 MXFP4/6/8 更好的 TTA，尤其是更高的最终精度。具体地，DynamiQ 达到 **72.38%**（BF16 最终精度的 99%）比 MXFP8 快 **12.0%**；当目标提高到 BF16 最终精度的 99.5% 时，这个优势进一步扩大到 **37.8%**。此外，微缩放基线表现出可测量的最终精度退化，而 DynamiQ 的最终精度与 BF16 相当。这可由 DynamiQ 更低的 vNMSE 解释。

最后，我们认为随着 worker 数增加这一趋势会持续，并在附录 B 给出理论直觉支持。

### 6. 模拟研究

#### 6.1 扩展性分析

我们把 worker 数 `n` 从 2 变到 64，跨两个负载评估扩展性：LLaMA 1B MMLU（2–8 worker）与更小的 TinyBERT on GLUE（8–64 worker）。所有实验使用 ring all-reduce 并以 BF16 基线为参照，度量量化误差（vNMSE）与最终精度 / 交叉熵损失。对 THC，我们采用其作者的建议，在 `n > 8` 时分配 12 位以防止聚合期间的梯度溢出。

**LLaMA 1B。** 随着 worker 数增加，所有方法的 vNMSE 与精度退化都自然增加。然而 DynamiQ 表现出比基线更好的扩展性质，即使在 8 worker 时也接近 BF16 的精度。

**TinyBERT。** 把分析扩展到更大集群，DynamiQ 在至多 64 worker 时始终取得所有压缩方案中最低的 vNMSE，相应地最终精度也最接近 BF16 基线。我们注意到小模型固有的训练方差会带来轻微波动（例如 DynamiQ 在 `n=8` 时略优于 BF16，或 MXFP8 在 `n=16,32` 时交叉熵略低），但总体趋势确认 DynamiQ 比其他压缩方法更可扩展、更稳定。

最后我们观察到 THC 与 OR 的 vNMSE 随 `n` 增长更慢。对 THC，这源于 `n>8` 时把分配从 8 位提到 12 位以防溢出（满足 `b ≥ ⌈log(15n+1)⌉`）；然而作为一种策略，这只在 `n ≤ 64` 内有效。对 OmniReduce（`b=8`），该规模下的误差轮廓由其稀疏化策略决定——它一贯丢弃范数最低的 50% 梯度。

#### 6.2 大规模模拟 **[v3 全新章节]**

为在大规模下评估 DynamiQ，我们模拟 DP 维度 `n = 8192` 的压缩 all-reduce。为建模第 1、2.2 节观察到的空间局部性与尺度偏斜，我们给每个 super-group 赋一个量级参数，决定该 super-group 内所有坐标的方差。

**合成数据。** 我们把 super-group `𝒢` 的量级 `M_𝒢` 建模为两个 LogNormal 分布的混合；这很好地刻画了图 1(c) 的经验分布（详见附录 E）。对每个 `𝒢`（大小 256），我们为每个 worker i.i.d. 地从 `N(0, M_𝒢²/256)` 采样其每个坐标。因此，同一 super-group 内的坐标共享一个公共尺度、保持空间局部性，而跨 super-group 的尺度跨越若干数量级。

**设置。** 我们在 ring 与 butterfly 两种 all-reduce 下评估 `b̄ ∈ {5, 6, 8.5}`。我们与 MXFP8-e4m3 对比（省略其他基线，因为它们在此设定下会溢出、产生大到无法接受的 vNMSE）。由于 MXFP8-e4m3 的通信代价固定为 `b̄=8.5`，我们只在该预算下报告它。对 DynamiQ，我们允许位宽取自 `W = {2,4,8,16}`。

**结果。** DynamiQ 取得远低于 MXFP8 的 vNMSE。ring + `b̄=8.5` 时 DynamiQ 的 vNMSE 为 0.0105，比 MXFP8 低约 **582 倍**。另一方面，DynamiQ 5 位的 vNMSE 大致与消耗 8.5 位的 MXFP8 相当。类似地，butterfly + `b̄=8.5` 时 DynamiQ 的 vNMSE 为 `2.75×10⁻⁵`，比 MXFP8 低 **1283 倍**。我们注意到在这个大 DP 维度下，MXFP8-ring 以及 `b̄ ∈ {5,6}` 的 DynamiQ-ring 都会产生大到无法接受的 vNMSE。因此对该 DP 维度下的 DynamiQ-ring，我们**建议使用 `b̄=8.5`**；而对 DynamiQ-butterfly，`b̄=6` 就足够。

我们把这些增益部分归因于 DynamiQ 的两个特性。第一，它的位分配方案能给稀有的高范数 super-group 分配至多 16 位，而不必让所有坐标承担这一代价。第二，它的 decompress–accumulate–recompress 过程**随着部分和的演化更新 group 的 scale**，使量化范围在整个 all-reduce 过程中都能很好地跟踪数据。相反，MXFP8-e4m3 把八位中的四位留给指数，以统一地表示宽动态范围。

#### 6.3 参数研究

我们隔离 DynamiQ 各优化组件——变量位宽分配、非均匀量化、层次化量化、相关舍入——对压缩误差的影响。group 大小设为 32，在使用层次化量化（INT8 缩放参数）时降为 16。

这些技术的累计应用把 vNMSE 降低了 **14 倍**（LLaMA 1B Chat）与 **22 倍**（MMLU）。**变量位宽分配是主要驱动力，把量化精度改进 3.5–5.1 倍。** 互补技术提供显著的加性收益：非均匀量化降低 vNMSE 约 **45%**，层次化量化约 **30%**，相关舍入约 **35%**。这种数量级的误差降低对于维持与未压缩基线相当的模型精度是必要的。关键在于，这些增强只引入很小的计算开销。

### 7. 相关工作

**梯度压缩与向 all-reduce 的转变。** 梯度压缩是通过缓解通信瓶颈来加速 DDP 训练的成熟策略。虽然已提出许多此类方法，它们都是为 parameter server 架构设计的。事实上，为扩展 LLM 训练而向多跳 all-reduce 的近期范式转变，揭示了这些方法的显著局限。例如，OmniReduce 这类基于稀疏的方法难以在去中心化拓扑上高效合并本地 TopK 块。类似地，QSGD、THC、Terngrad 等量化方案在聚合部分和时容易发生**梯度溢出**——这是多跳拓扑中的根本问题，且随系统规模变大而恶化。虽然基于微缩放的方法（如 MXFP4）缓解了这一点，但它们并**未完全消除**溢出/下溢。相比之下，DynamiQ 显式地为多跳 all-reduce 而架构，采用**逐跳的解压/重压缩来严格防止溢出**，并利用变量位宽分配来保证鲁棒性。

**压缩误差与扩展性。** 虽然 LLM 对压缩噪声有一定容忍度，过大的误差会显著降低收敛稳定性与最终精度。已有方案往往优先考虑推理硬件兼容性或稀疏性，而不是最小化误差（vNMSE）。例如微缩放技术为 GPU 吞吐优化但缺少高级误差降低机制，而 OmniReduce 依赖在稠密 LLM 更新中基本不存在的梯度稀疏性。此外，随 worker 数 `n` 增加而保持有界误差是一个重大挑战：误差逐跳累积，通常需要位宽按聚合路径长度的对数增长（如 THC）以防溢出。**我们经验上观察到 DynamiQ 的这一增长更慢，但把进一步研究留作 future work。**

**硬件感知实现。** GPU 上的梯度压缩主要是 memory-bound 而非 compute-bound；性能由 HBM 带宽而非浮点吞吐决定。因此高效实现必须最小化 HBM 事务，理想情况是通过内核融合确保顺序的单遍访问。不尊重这一约束的方法会产生显著开销——一个显著例子是 THC，其 Hadamard 变换需要 `O(log d)` 遍内存，制造出一个瓶颈。DynamiQ 通过融合内核把中间结果留在寄存器或共享内存中避免了这类开销，维持与标准未压缩更新相当的访存模式。

**混合精度训练。** 使用更低精度算术是加速训练的新兴技术，动力来自为低精度操作提供更高吞吐的新硬件能力。当前最佳实践是把某些字段（例如离群值或累加器）保持在更高精度，其余用低精度。近来研究者提出在整个训练过程使用低精度，这通常带来在某些场景下可接受的精度退化。

**分片模型。** 当模型大到单 GPU 装不下时，实践者会把它们分片到多个 worker 上。这种情况下人们可能不需要 all-reduce 操作，而**只需要 reduce-scatter 阶段**，因为梯度与权重都跨 GPU 切分。**DynamiQ 可以无缝集成到这一做法中——在 reduce-scatter 阶段结束时解压即可。**

### 8. 讨论与局限 **[v3 全新章节]**

**对大规模预训练的启示。** 我们的 testbed 评估因硬件资源有限而聚焦微调，但它已覆盖至多 **64** 的 DP 维度。在真实预训练集群中，这可以对应大得多的总 GPU 数，因为 3D 并行使 DP 维度显著小于总 GPU 数；例如即便是至多 10 万 GPU 的集群，其 DP 维度也可能是可比的。因此我们的 testbed 结果与现实的大规模预训练部署**直接相关**。对更大部署（例如未来的百万 GPU 集群），扩展 DynamiQ 主要需要提高位预算或使用**分层聚合**。我们的模拟显示 DynamiQ 在 DP 维度 8192、8.5 位预算下仍然准确，表明适度更大的预算就能支撑大得多的 DP 组。另一种做法是分层 butterfly 或超立方体式聚合——这在该规模下本来就是控制尾延迟所必需的——也可以降低有效聚合深度。

**保持模型质量与数值稳定性。** 对大规模预训练任务，实践者对低精度压缩方案往往持保守态度，因为压缩误差可能表现为损害最终模型质量或收敛的隐性不稳定。例如偶尔把极值量化进不安全的低精度表示会导致 NaN、造成模型发散，或以其他方式显著降低精度。**DynamiQ 的 super-group 位宽分配方案天然支持把这类离群值保持在训练精度（例如 BF16）。**

**与 FSDP 和 MoE 梯度的兼容性。** DynamiQ 与 FSDP 式同步兼容，因为 FSDP 把同步分解为 reduce-scatter 与 all-gather 阶段，DynamiQ 的多跳通信原语可以按 §3.4 施加其上。对 MoE 梯度（专家可能激活或不激活），**DynamiQ 的位分配可以给未激活的专家 super-group 分配 0 位，同时给激活的分配更多位。**

**对高端训练硬件的启示。** 我们的硬件评估使用显存带宽 768 GB/s、节点间带宽 100 Gb/s 的 A6000 GPU。现代集群（例如 GB200 级 GPU）提供更快的 scale-out 网络（如 400 Gb/s 或 800 Gb/s 链路），但同时也有 **16× 更高的 FP16/BF16 FLOPS** 与 **10.4× 更高的显存带宽**。因此通信仍然是瓶颈，尤其是在大数据并行维度或共享集群争用下。

### 9. 结论

本文提出了 DynamiQ——一个为多跳 all-reduce 优化的实用梯度压缩框架，它可以适配不同带宽约束，并在通信开销与精度之间给出有吸引力的权衡。与为 parameter-server 架构设计、部署到多跳 all-reduce 时会产生精度退化的现有梯度压缩系统不同，DynamiQ 沿聚合路径保持低压缩误差，从而在不损害模型精度的前提下加速训练。

我们实现了 DynamiQ 并在多样的 LLM 训练负载上用 ring 与 butterfly all-reduce 评估其性能。结果显示 DynamiQ 相对备选方案稳定取得显著更好的 time-to-accuracy。值得注意的是，DynamiQ **只用 5 位每坐标**就达到 BF16 基线精度的 99.9%，优于 SOTA 的 MXFP8。它是唯一稳定保持这一保真度同时提供显著加速的被测方法——这一结果由其快速的、协同设计的融合 CUDA 内核驱动。

> **[v1 差异]** v1 结论末尾为"我们计划在发表后开源实现。本工作不引发任何伦理问题。"v3 换成了致谢（Michael Mitzenmacher 部分受 NSF CNS-2107078 与 NSF DMS-2023528 资助；Ran Ben Basat 部分受 Google Research Scholar Award 资助），并感谢匿名审稿人与 shepherd。

### 原文附录 A：变量位宽分配的更快解法

我们提出一个跨训练轮动态维护并调整 `T_{a,b}` 近似值的更快解法，假设至多三种可能位宽。我们针对 DynamiQ 原型使用的设定（允许位宽 `W = {2,4,8}`）来说明。为加速计算，我们**避免对 `F_j` 数组排序**（§3.2 描述的算法需要排序），改为用下式计算每个 super-group `j` 被量化的位数 `q_j`：

$$q_j = 2^{\text{clamp}\left([1,3],\ \left\lfloor \log_2\left(\tfrac{4}{\log_2(512/17)}\log_2 F_j + u\right)\right\rfloor\right)}$$

回忆阈值需要满足 `T_{2,4} = (17/512)·T_{4,8}` 与带宽约束 `S·Σ_j q_j ≤ d·b̄`（此处 `b̄` 表示总位宽预算减去用于传输元数据等的逐条目带宽）。

令 `z_j = (4/log₂(512/17))·log₂F_j + u`。观察到 `z_j < 4` 时 `q_j = 2`，`z_j ∈ [4,8)` 时 `q_j = 4`，`z_j > 8` 时 `q_j = 8`。由于 `T_{a,b}` 被定义为"`F_j ≥ T_{a,b}` 的 super-group 至少分到 `a` 位、`F_j ≤ T_{a,b}` 的至多分到 `b` 位"的阈值，我们有 `T_{2,4} = F_j ⟹ z_j = 4` 与 `T_{4,8} = F_j ⟹ z_j = 8`，得到三个方程：

- `4 = (4/log₂(512/17))·log₂T_{2,4} + u`
- `8 = (4/log₂(512/17))·log₂T_{4,8} + u`
- `S·Σ_j q_j ≤ d·b̄`

于是我们的目标是通过**二分搜索**调整 `u` 以满足位宽约束：若当轮计算出 `Σ_j q_j > d·b̄` 就减小 `u`，反之增大。`u` 确定后即可确定 `q_j`。

### 原文附录 B：不同 all-reduce 拓扑下的分析

与许多为 parameter-server 聚合设计的先前压缩方案不同，DynamiQ 天然支持不同的多跳 all-reduce 拓扑，包括 ring all-reduce 与 butterfly all-reduce（也称 recursive doubling）。我们指出，相比 ring，butterfly 在大规模 DDP 训练系统中通常取得更低的尾延迟。

我们进一步观察到把 DynamiQ 部署到 butterfly 也改善了压缩误差随 `n` 增大的扩展性。直觉是：**每一跳的压缩误差正比于被传输部分和的取值，而后者又正比于对应子树的大小**（若不同 worker 上的梯度服从相同分布）。图 13 说明了这一点：worker 3 的子树大小为 4，它压缩 worker 0～3 梯度的部分和并传给 worker 7，后者持有 worker 4～7 的部分和。

我们现在启发式地分析 ring 与 butterfly 的压缩误差。分析中我们使用各 worker 期望 MSE 之和。假设 worker `i` 第 `j` 个 super-group 中索引 `k` 处的梯度数据 `X_{i,j}[k]` 被 `M = max_{i,k}|X_{i,j}[k]|` 界定。可以推导出在 worker `i` 压缩部分和梯度 `s_{i,j}` 的 MSE 被 `MSE ≤ εS·max_k|s_{i,j}[k]|²` 界定。我们注意到 `max|s_{i,j}[k]| ≤ M·|subtree(i)|`，其中 `|subtree(i)|` 是以 worker `i` 为根的子树大小。于是对 ring all-reduce，期望的最坏 MSE 被界定为

$$\text{MSE} \le \sum_i \epsilon S i^2 M^2 = O(\epsilon S M^2 n^3)$$

而 butterfly 的为

$$\text{MSE} \le \sum_{l \le \log n} \epsilon S (M 2^l)^2 \cdot (n/2^l) = O(\epsilon S M^2 n^2)$$

也就是说，**我们对 butterfly 的 MSE 上界比 ring 小一个 `n` 因子**。

### 原文附录 C：附加实验设置（要点译述）

**OmniReduce 适配到 ring all-reduce。** OR 采用分块 top-k 压缩，每个 worker 选择并聚合其本地 top-k 梯度块。单跳 PS 下这很容易；但多跳中各 worker 的本地 top-k 块索引不同，中间跳聚合出的块数可能超过 `k`，带来通信开销上升。我们的适配是计算"至少出现在一个 worker 的本地 top-k 中"的索引并集，称为全局 top-`K` 块，其中 `K/n_chunks = b/16`（即等于期望压缩比）。给定固定 `K`，直接确定所需的本地 `k` 很困难（它随梯度分布动态变化），因此我们用启发式近似：第 `t` 轮给定 `k_t`，计算并集得到的实际全局块数 `K'_t`，用比值 `K/K'_t` 调整 `k_{t+1}`，按动量规则 `k_{t+1} = γk_t + (1-γ)(K/K'_t)k_t`（实验中 `γ = 0.8`）。

**微缩放浮点压缩（MXFPX）适配到 all-reduce。** 由于 MX 规范未定义 all-reduce 所需的求和算术，我们按 FP8-LM 的实现来适配。算法维护一个初始化为 `n` 的参数 `μ` 控制量化 scale。每轮先在每个 worker `i`、每个块 `j` 上算梯度块绝对值最大 `m_{i,j}`，all-reduce 得到全局最大 `gm_j = max_i m_{i,j}`。块的全局 scale 定为 `s_j = μ·gm_j`，原梯度量化为 `g'_{i,j} = (g_{i,j}/s_j)·FPX_MAX`。`μ` 的选择很关键：更小的 `μ` 导致更多溢出，更大的导致下溢。我们采用 FP8-LM 的自动缩放：若溢出率超过阈值 `ε`，下一步 `μ ← 2μ`；若溢出率持续小于 `γ`，则 `μ ← γμ`（`0 < γ < 1` 且接近 1）。

### 原文附录 D：附加评估结果（要点译述）

**完整 TTA 曲线。** 正文给的是放大版，附录给全尺度版本，展示各方法如何从初始低精度逐步逼近 BF16 基线；全尺度版本确认放大图中显示的收敛精度在延长时间后保持稳定、不再改善。

**带宽随时间的使用。** LLaMA 1B MMLU 的带宽曲线呈周期性，每个周期对应一轮训练。**BF16、DynamiQ、MXFP8 的计算区间保持一致，而通信区间显著缩短**——这清楚表明 DynamiQ 通过最小化通信开销降低了每轮训练时间。

**压缩误差随训练步的演化。** DynamiQ 与多数基线的 vNMSE 在训练推进中保持相对稳定，即便梯度分布随模型收敛而演化。值得注意的是 OmniReduce 在训练早期 vNMSE 上升，表明梯度稀疏性随训练推进而下降，使 OR 的固定比例稀疏化越来越无效。

### 原文附录 E：为大规模模拟合成数据（v3 全新，要点译述）

我们通过对图 1(c) 中 super-group ℓ2 范数的经验分布拟合 LogNormal 混合来生成每个 super-group 的量级 `M_𝒢`。对 `K` 个分量的混合，`φ(x;θ) = Σ_k π_k LogNormal(x | μ_k, σ_k)`，共 `3K-1` 个自由参数。由于经验量级跨越若干数量级，我们在**对数尺度**上最小化经验 CDF 与拟合 CDF 之间的平方差来拟合 `θ`。拟合结果：`K=1` 时 `π₁=1, μ₁=-5.11, σ₁=3.14`；`K=2` 时 `π₁=0.79, π₂=0.21, μ₁=-3.99, μ₂=-16.41, σ₁=1.22, σ₂=4.63`。

### 参考文献

参考文献从略。与本文关系最紧密的几条已在 §3.4「推荐进一步阅读」中列出。

---

## 五、对 MI455X / Helios 的参考点

**背景（我们的 open work item）**：NVIDIA 的 GTP（Generalized Tensor Parallelism）在 Megatron-LM 里把权重 all-gather 做到了原生 MXFP8 / NVFP4，但把**权重梯度的 reduce-scatter 留在 BF16**。算一下就知道为什么这是个洞：分布式优化器下，每个权重元素每步的通信约为 `AR·bytes_param`（param AG）+ `AR·bytes_grad`（wgrad RS）。

| 配置 | param AG | wgrad RS | wgrad 占比 |
|---|---|---|---|
| MXFP8 param + BF16 wgrad | 1 B | 2 B | **67%** |
| NVFP4 param + BF16 wgrad | 0.5 B | 2 B | **80%** |

这正好框住"64–78%"这个观察。而如果把 wgrad RS 压到 5 位（0.625 B）：NVFP4 配置下每权重总通信从 2.5 B 降到 **1.125 B（2.2×）**，MXFP8 配置下从 3 B 降到 **1.625 B（1.85×）**。即便保守到 8 位（1 B），NVFP4 配置也有 **1.67×**。**低精度 wgrad RS 是当前 per-weight 通信预算里最大的一块未开采储量**，这个立项理由本身是硬的。

目标硬件：MI455X / Helios（CDNA5，432 GB HBM4/GPU，per-GPU scale-up 3.6 TB/s over UALink-over-Ethernet，**72 GPU 单跳非阻塞 scale-up 域**，机架 scale-up 聚合 260 TB/s，per-GPU scale-out 约 597 GB/s，即约 **6:1** 的机架内/跨机架带宽落差）。

### 5.1 最要紧的问题：单跳域里，"部分和感知的量化"还买得到东西吗？

**先说结论**：DynamiQ 的**核心贡献（部分和感知）在机架内基本失效，在跨机架路径上才成立**；但它**消融里贡献最大的那一项（变量位宽分配）与跳数无关，两边都成立**。也就是说：这篇论文对我们最有价值的部分，恰恰不是它的标题。

**(a) "单跳非阻塞"描述的是 fabric，不是算法——但它确实允许算法退化成单跳。**

72 GPU 全非阻塞意味着任意一对 GPU 可以一次交换机穿越打满速率。在这样的 fabric 上，带宽最优的 reduce-scatter 不是 ring，而是 **direct / one-shot RS**：每个 rank 把自己第 j 块的分片**直接**发给块 j 的 owner，owner 在本地把 72 份加起来。用 DynamiQ 自己的语言说，这是**深度为 1、扇入为 72 的 in-arborescence——一颗星**，所有非 sink 节点都是 leaf。

这一步很关键：星形 RS 的**线上数据量与 ring 完全相同**（每 GPU 发/收 `(n-1)/n · d`），但它一跳完成、且**中间没有任何重压缩**。sink 在本地用它想要的任何精度累加，免费。所以在 Helios 的 72 卡域里，压根不该跑 ring。

代入 Appendix B 的界：所有子树大小 = 1，MSE 退化为 `Σ_i εSM² = O(εSM²·n)`——比 ring 的 `n³` 少两个 n，比 butterfly 的 `n²` 少一个 n。**fabric 把误差问题直接解决了，而且是免费解决的。** DynamiQ 那套 `decompress_accumulate_recompress` 在这里一次都不会被调用。

反过来说，这也正是 AGoQ 的结构，也正是 DynamiQ 自己在 §7「Sharded models」里顺口提到但没做的那条路（"只需要在 reduce-scatter 末尾解压即可"）。

唯一的注意点是消息尺寸：72 路切分后每条消息是 `d/72`。Megatron 的 grad bucket 一般是几十到几百 MB，除以 72 还有 MB 量级，在 3.6 TB/s 上不会掉进 latency-bound。要是 bucket 被切太碎才需要退回分层算法（从而重新引入跳）。

**(b) 但有三样东西即使在深度 1 也活下来，而且都不是"多跳"贡献。**

1. **变量位宽分配（消融里 3.5–5.1×，单项超过其余三项之和）与跳数完全无关。** 它利用的是梯度的空间局部性和偏斜：LLaMA 约 20%、Gemma 约 30% 的 super-group 范数比中位数低数量级，给它们 8 位是纯浪费；MXFP8 的 8.5 位是**处处等价**的，这就是它输给 5 位 DynamiQ 的全部原因。深度 1 下照样可以跑轻量 all-reduce 拿 `F_j`、分配 2/4/8/16 位、重排。**这是要从这篇论文里拿走的第一件东西。**
2. **层次化 scale**（per-group UINT8 挂在 per-super-group BF16 下，约 30%）是纯格式设计。group=16 + UINT8 scale 的元数据开销是 0.5 位/坐标，与 MXFP8 的 chunk-32 + BF16 scale **完全相同**，但粒度细一倍。这是一个免费的格式改进。
3. **相关舍入（约 35%）在深度 1 反而更强，不是更弱。** Suresh et al. 的机制本来就是为"n 个 worker 对同一坐标各自舍入"的经典 DME 场景设计的；72 个 leaf 同时喂一个 sink 正是这个场景的教科书形态。多跳反而把它搞复杂了（各跳重压缩的部分和不是 n 个对称贡献）。

**(c) 多跳在哪里？在跨机架，也就是 6:1 里窄的那一边。**

跨机架 DP 归约才有真正的多跳结构：分层方案是「机架内 RS → 跨机架 RS（在机架代表之间）→ 跨机架 AG → 机架内 AG」。跨机架那一段操作的是**已经在机架内被 72 卡求和过的数据**——这正是 DynamiQ 意义上的"部分和"，也正是格式需要跟踪一个增长中的量级的地方。**所以 DynamiQ 的贡献确实落在跨机架这一段，而且这一段还恰好是带宽真正疼的一段。**

但有个反直觉的推论必须说出来：**6:1 的层次结构让多跳问题变小了，不是变大了。** 因为机架内一跳吃掉 72 卡，要覆盖 DP 规模 N 所需的跨机架深度是 `log₂(N/72)`（butterfly），不是 `log₂N`。DP=8192 只需 8192/72 ≈ 114 个机架，butterfly 深度 7。而论文 Table 7 那组吓人的数字（582×、1283×）是在**扁平** n=8192 下取的——ring 深度 8191、butterfly 深度 13。**Helios 的层级结构已经免费砍掉了其中 6–7 层。** 所以 DynamiQ 相对"MXFP8 + 逐跳重标定"的边际价值，在 Helios 上会比论文的 headline 小得多。

**(d) 层级中的定位（一句话版本）：**

> **机架内（72 卡，单跳）：不需要 DynamiQ 的部分和机制。用 direct RS——量化一次、发一次、在 owner 上 FP32 本地归约。保留变量位宽分配、层次化 scale、相关舍入。**
> **跨机架（多跳，带宽窄 6 倍）：DynamiQ 的贡献字面成立，但有效深度只有 log(机架数)，边际收益远小于扁平 DP=8192 的宣传值。**
> **净结论：fabric 吃掉了这篇论文的大部分问题。活下来的是一个带宽分配结果，不是一个多跳结果。**

### 5.2 融合 kernel 在 ROCm / CDNA 上要花什么代价——以及一个决定成败的比值

**先给那个比值，因为它比任何定性讨论都有用。**

线上流量（每坐标每 worker，带宽最优 all-reduce）`W = 2·AR·(b̄/8)` 字节：BF16 是 `4·AR`，DynamiQ 5 位是 `1.25·AR`，节省 `ΔW = 2.75·AR` 字节。
额外 HBM 流量（论文 Table 2 之差）`ΔH = (22 + 11.875·AR) - (4 + 4·AR) = 18 + 7.875·AR` 字节。

盈亏平衡要求 `B_HBM / B_wire > ΔH/ΔW`。`AR = 71/72` 时这个门槛是 **约 9.5**。代入各环境：

| 环境 | HBM 带宽 | 有效线速 | 比值 | vs 门槛 ~9.5 |
|---|---|---|---|---|
| **论文 testbed**（A6000 + 100 GbE） | 768 GB/s | 12.5 GB/s | **61** | 大幅盈利（这就是他们数字那么好看的原因） |
| MI355X 单节点 XGMI | ~8 TB/s | ~1 TB/s 量级 | **~8** | **略亏** |
| **MI455X 机架内 scale-up** | ~16 TB/s（repo projection 假设） | 3.6 TB/s | **4.4** | **明确亏** |
| **MI455X 跨机架 scale-out** | ~16 TB/s | 0.597 TB/s | **27** | **明确盈利，约 2.8× 余量** |

> 口径说明：3.6 TB/s 与 597 GB/s 若按双向理解则各减半，比值分别变成 8.9 与 54——**结论方向不变**（机架内仍在门槛下、跨机架仍在门槛上 3–6 倍）。HBM 数字取自本 repo 的 projection 配置（`MI455X: 16000 GB/s`），不是官方规格。Table 2 是作者对其 CUDA 实现的估计，一个更省的 ROCm 实现可以把 `ΔH` 压下来，从而降低门槛。

**这张表从一个完全独立的方向印证了 §5.1**：机架内不该跑 DynamiQ，跨机架该跑。同时它也解释了为什么论文的加速比不能外推——它的 HBM/线速比值是 61，Helios 机架内是 4.4，**差 14 倍**。

**实现要点（CDNA 特有的好消息与坏消息）：**

- **好消息 1：group = 16 恰好是 wave64 的四分之一。** `sf_G = max|G|` 因此是纯 DPP row-shuffle 归约（`v_max_f32` + `row_shr`），**完全不需要 LDS**，一个 wave 同时处理 4 个 group。super-group = 256 就是 4 个 wave。这个尺寸对 CDNA 比对 CUDA 的 warp32 还合适。
- **好消息 2：整个 kernel 没有 MFMA，纯 memory-bound 流式 + 小归约。** 用 `dwordx4`（128 位）global load、寄存器里做完全部工作，是 CDNA 上最容易打满带宽的一类 kernel。
- **坏消息：`f(ε,r)` 千万不要在 kernel 里算。** `((1+2ε²)^r - 1)/((1+2ε²)^{2^{b-1}-1} - 1)` 逐坐标求幂会把 memory-bound 变 compute-bound。`b ≤ 8` 时最多 128 个不同取值，FP32 LUT 只有 512 B——放 LDS；`b ≤ 4` 时只有 8 个值，可以用 `v_perm_b32` 做寄存器内查表。**这是整个移植里最重要的一条实现笔记。**
- **2/4 位打包**靠 `v_bfe_u32` / `v_bfi_b32` / `v_perm_b32`，能做但逐坐标指令数不低。论文的论点是"反正 memory-bound 所以免费"——在 MI355X 8 TB/s × 256 CU 上大体成立，但需要实测验证，不能假设。
- **随机数**：Philox-4x32 在寄存器里跑。相关舍入需要一个跨 worker 共享的置换 π——每个坐标一次共享种子推导，各 worker 只取自己的 `π_i`，很便宜。
- **语言选择**：Triton on ROCm 做原型和 `compress`/`decompress` 是合适的（`tl.max`、`tl.rand` 都有），但 sub-byte 打包和 DPP 级控制在 Triton 里很别扭。**建议 Triton 出参考实现和数值验证，DAR kernel 若实测指令数是瓶颈再落到 HIP。** FlyDSL 不合适——它瞄的是 GEMM 形状的工作，这是流式逐元素 kernel。

**放哪儿：Primus-Turbo，但要切干净。**

- **Primus-Turbo 拿 kernel**：`supergroup_stats`、`quantize_supergroup`、`dequant_reduce_fp32`、`requantize`、`pack/unpack`、`supergroup_reorder/unreorder`。理由是它已经承载 FP8/MXFP8 量化 kernel 和 FlyDSL 后端，是"AMD 原生数值 kernel"的自然归属；而且**这些 kernel 可以脱离集群单测**（对着 torch FP64 参考验 vNMSE），迭代快。
- **Primus 拿编排**：分布式优化器的 grad-RS hook、拓扑计划、bucket 调度。这部分需要集群才能测。
- 这个切分很重要，因为它让 §5.5 的 Phase 0 可以在一张卡上跑完。

### 5.3 和 RCCL 能不能配？——这里有个反直觉的优势

repo 的 roadmap（C3-4）里已经写死了一条判断：**RCCL 架构上无法 overlap（collective 语义 + host 发起），三种方式验证过**；NVIDIA 在 NCCL 2.28 用 device API + copy-engine collectives 解决了，ROCm 侧公开进展只到"4-NIC 拓扑感知 + backport 算法"。

**DynamiQ 的主 all-reduce 根本不是一个 collective。** 它是自己用 NCCL P2P（`ncclSend`/`ncclRecv`）搭出来的 reduce-scatter，v3 里跨节点还是自建 RDMA engine。这件事在 AMD 上是**加分项**：

1. **P2P send/recv 恰好是 RCCL 里映射最干净的那部分。** `ncclSend`/`ncclRecv` 有语义一致的 RCCL 对应物，grouped P2P（`ncclGroupStart/End`）也可用。你不需要 device-initiated communication，不需要 copy-engine collectives，也不需要 RCCL 的算法选择机制——**你是在用 P2P 自己搭 reduce-scatter，正好绕开了 RCCL 最弱的那层抽象。**
2. **DynamiQ 需要的 overlap 是粗粒度、host 可调度的**：8 MB 块，`DAR(块 k)` 与 `传输(块 k-1)` 并行。这是 stream 级 overlap + 几个 event，**RCCL 今天就能做**。
   > 讲白了：**让 DynamiQ 在 NVIDIA 最新栈上显得别扭的那个设计（绕过 collective），恰恰让它在 AMD 当前栈上容易落地。** 这条值得在立项文档里明说。
3. **代价是什么**：放弃 RCCL 的拓扑感知算法选择、调优表、rail/NIC 感知路由、以及 4-NIC-per-GPU 布局的处理。在扁平非阻塞 scale-up 域里这个代价很小（direct RS 的路由是平凡的）；**跨机架就是真代价了**，那里 RCCL 的 rail 感知不是白给的。
4. **真正的风险点**：72 rank 手搓 P2P RS 意味着每 rank 每步 71 对并发 send/recv。RCCL 的 P2P channel/connection 建立开销、channel 数上限、proxy 线程代价在这个并发度下是未知数。**这是 Phase 1 要第一个量的东西。**
5. **一个保留 RCCL 的替代形态**：深度 1 的 RS 可以表达成 `all-to-all 压缩载荷` + `本地融合 dequant-reduce kernel` + `all-gather 压缩结果`。这只用到 RCCL 已经做得不错的 collective（`ncclAllToAll` / grouped P2P、`ncclAllGather`），把上面第 4 条的风险基本消掉。**而这正是 AGoQ 的形状**——这是支持走 AGoQ 形状的一条独立论据。

### 5.4 vs AGoQ：哪个更适合大 scale-up 域上的 wgrad RS

| 维度 | DynamiQ | AGoQ |
|---|---|---|
| 结构 | 保留多跳 RS，每跳 decompress-accumulate-recompress | 消除多跳归约：A2A → 本地 FP32 归约 → 重量化 → AllGather |
| 加法发生在哪 | 每个中间跳，寄存器精度，然后重量化 | 只有一次，本地，FP32 |
| 每坐标重量化次数 | = 内向树深度（ring `n-1` / butterfly `log n`） | **恰好 1** |
| 线上量 | `2·AR·b̄` 位/坐标 | A2A `AR·b` + AG `AR·b` = **同样 `2·AR·b`** |
| fabric 要求 | ring / 只有邻接链路也能跑 | **需要 full bisection** |
| 位分配 | **变量 2/4/8/16，按聚合后范数** | 固定 8 位块量化 |
| 报告数字 | 5 位下 vNMSE 0.001–0.002；TTA +40.8% vs BF16（8 GPU / 4 worker） | 显存 −52%；≤64 GPU 上比 Megatron-LM 至多 **1.34×**（LLaMA2-13B，80K 序列） |

**判断：在 72 卡非阻塞 scale-up 域上做 wgrad RS，应该走 AGoQ 的结构 + DynamiQ 的位分配。** 四条理由：

1. 非阻塞 72 卡域正是 AGoQ 的 A2A 所假设、而 DynamiQ 的 ring 并不需要的 fabric。A2A → 本地归约就是深度 1，**根本没有部分和问题要解**。
2. 每坐标恰好一次重量化 = 单次分布式均值估计的误差**理论下界**。DynamiQ 的全套机制是在从上方逼近这个下界；而在这个 fabric 上你可以直接站在下界上。
3. 它只用 RCCL 已经实现得不错的 collective，§5.3 的第 4 条风险基本消失。
4. **但 AGoQ 的固定 8 位块量化把最大的那块收益扔了**——DynamiQ 消融说变量位宽分配单项就是 14–22× 里的 3.5–5.1×。既然 20–30% 的 super-group 范数低数量级，统一 8 位就是在为没有信号的坐标付全价。

**具体的组合形态：**

```
1. 轻量 all-reduce 收 (μ_j, F_j)        [DynamiQ §3.1]
2. 按 F_j 给 super-group 分 2/4/8/16 位  [DynamiQ §3.2 — 最大收益项]
3. 零均值化 + 按位宽重排                 [DynamiQ §3.3, 本地零通信]
4. 压缩后 A2A（每 rank 把第 j 块发给 owner j）  [AGoQ 结构]
5. owner 本地融合 dequant → FP32 累加 72 份    ← 没有 DAR kernel
6. 重量化一次 → AllGather               [AGoQ 结构]
7. 逆重排 + 加回均值
```

`decompress_accumulate_recompress` 一次都不调用。层次化 scale 和相关舍入全部保留（后者在深度 1 下**更强**）。

**一个论文没做、但应该做的改进：用上一步的 `F_j`。** 第 1 步的轻量 all-reduce 虽然只有 0.125 位/坐标，但它是主 RS 之前的一个**串行化同步点**。而论文自己的 Figure 18 显示 vNMSE 跨训练步基本平坦，说明梯度的 super-group 范数分布高度自相关。**直接用上一步的 `F_j` 定这一步的位宽，可以把这个 collective 从关键路径上完全拿掉，代价只是一步的陈旧度。** 论文的 Appendix A 已经在跨轮维护阈值 `u` 了，等于走了一半——把 `F_j` 本身也跨轮复用是免费的下一步，而且如果这条线要出成果，这是个干净的差异化点。

### 5.5 先做什么：分阶段、工作量、量什么

**Phase 0（1–2 周，1 人，单卡，不需要集群）——kernel + 数值**

- 实现 `supergroup_stats`、`quantize_supergroup`（LUT 化的非均匀 Q、UINT8 group scale 挂 BF16 super-group scale、相关随机舍入）、`dequant_reduce_fp32`、`requantize`、`supergroup_reorder/unreorder`。先 Triton。
- **验收门槛：复现论文 Table 6 的消融阶梯**（LLaMA 1B Chat：均匀 0.1278 → 非均匀 0.0707 → +变量位宽 0.0198 → +层次化 0.0138 → +相关舍入 0.0091）。复现不出来就是有 bug，别往下走。另外验无偏性（10⁴ 次抽样下 `E[x̂] ≈ x`）。
- 用 `rocprof` / `omniperf` 量 DAR kernel 的实际 HBM 带宽利用率，**目标 ≥70% 峰值**；对照 Table 2 的事务数核账。
- **交付物：在从 Primus 真实 run 里 dump 出来的 DSv3 / Llama 梯度快照上，画 vNMSE-vs-位宽曲线，叠 MXFP8。这是整件事里决策价值最高的一张图。**

**Phase 1（2–3 周）——单节点 8×MI355X，AGoQ 形状的 RS**

- 接成 Megatron 分布式优化器的 grad-RS hook（**不是 DDP comm hook**——Primus 走的是 Megatron 分布式优化器）。压缩 A2A 用 grouped `ncclSend/Recv` on RCCL，本地 FP32 dequant-reduce，重量化一次，`ncclAllGather`。
- 量：暴露的 DP 通信时间、step 时间、HBM 流量增量、以及实测的 `B_HBM/B_wire` 对照 9.5 门槛。同时量 §5.3 第 4 条的 RCCL P2P channel 开销。
- **提前把预期讲明白：8×MI355X 的 XGMI 比值约 8，在门槛线下方，所以单节点大概率是打平或小亏。这一轮是测正确性和 kernel 效率的，不是拿加速比的。** 不先说清楚，项目会被第一个难看的数字杀死。
- **加速比必须跨节点量**：2–4 节点走 scale-out NIC，比值 20–30，那里才有数。

**Phase 2（3–4 周）——收敛**

- 论文只验了 fine-tuning。我们要问的是预训练。跑一个 DSv3-class 或 Llama-1B-class 的小规模预训练（几 B token），5 位 / 6 位 wgrad RS vs BF16，跟 loss 曲线偏离和梯度范数统计。
- 盯 v3 §8 点出的那个风险：极值被量化进不安全表示 → NaN。**确保 `W` 里包含 16，且阈值搜索不会把 16 位那一档饿死。**
- **必须和 repo 里已有的那条 AMD 证据交叉验证**：[MXFP4 原生预训练](./mxfp4-pretraining.md)（Penn State + AMD）在 MI355X 上测出 wgrad 量化是收敛退化主因（到 ppl 3.3 的 token 开销 Fprop 8–9% → +Dgrad 10–11% → **+Wgrad 26–27%**），而且**随机舍入与随机 Hadamard 在全流水下直接不收敛，确定性 Hadamard 才救得回来**。DynamiQ 的相关随机舍入不等同于随机 Hadamard，但这条 AMD 自家结论明确说 wgrad 路径对注入的随机性异常敏感。**所以要跑一组三方 A/B：相关随机舍入 / 独立随机舍入 / 确定性最近邻，三条 loss 曲线一起报。**（确定性最近邻是有偏的、破坏收敛保证，所以它是对照而不是候选——但如果它明显更稳，那本身就是重要发现。）

**记分卡（六项）**

1. 真实梯度上的 vNMSE-vs-位宽曲线 vs MXFP8 —— 决定这个格式值不值得做
2. DAR / dequant-reduce kernel 的实际 HBM 带宽利用率（% 峰值）—— 决定融合 kernel 的论点在 CDNA 上成不成立
3. 层级各级实测的 `B_HBM/B_wire` vs 约 9.5 门槛 —— 决定在拓扑的**哪一级**打开它
4. 2–4 节点上暴露 DP 通信时间的下降 —— 真正的加速比
5. 几 B token 上相对 BF16 的 loss 曲线偏离 —— 出货门槛
6. 轻量 all-reduce 的每步开销，以及改用上一步 `F_j` 后的差值 —— 决定那个额外同步点要不要处理

**总量：一个工程师约 6–9 周到一个带真实数字的 go/no-go。**

**明确的 no-go 判据**：如果 Phase 0 显示在真实 DSv3 梯度上，vNMSE-vs-位宽曲线在 ≤6 位处打不过 MXFP8，就停。整个立论建立在"生产 MoE 梯度的 super-group 范数分布，偏斜程度与 LLaMA-1B 微调梯度相当"这个假设上——而 MoE 梯度在结构上就不一样（专家权重只从被路由到的 token 拿梯度）。论文自己 §8 说未激活专家可以给 0 位，暗示这可能是**更大**的机会；但那是纯推测，一行实验都没有。**这条假设值得花两周去证伪，因为它便宜，而且它决定了后面 7 周做不做。**

---

## 附录

### A. 术语表

| English Term | 中文 | 说明 |
|---|---|---|
| Multi-hop all-reduce | 多跳 all-reduce | 梯度沿聚合拓扑被多次部分求和的 all-reduce，与单跳 parameter-server 相对 |
| In-arborescence | 内向树 | 所有边指向唯一 sink 的有根树。本文对单个 chunk 的 reduce-scatter 拓扑的形式化；ring 退化为路径，butterfly 为深度 log n 的树 |
| Partial sum | 部分和 | 聚合路径中间节点上的累加结果，其量级正比于该节点的子树大小 |
| Group / Super-group | 组 / 超组 | 16 个连续坐标为一 group（共享 scale），16 个 group 为一 super-group（共享位宽与统计量，256 坐标） |
| Stochastic quantization (SQ) | 随机量化 | 以与距离成反比的概率向上/向下舍入，保证 `E[x̂] = x` |
| Hierarchical quantization | 层次化量化 | per-group scale 用 UINT8 随机量化，挂在 per-super-group 的 BF16 scale 之下；两级随机性独立故逐项仍无偏 |
| Correlated rounding | 相关舍入 | `u_i = (π_i + γ_i)/n`，π 为共享 PRNG 生成的置换，使各 worker 舍入方向负相关 |
| vNMSE | 向量归一化均方误差 | `E[‖X-X̂‖²]/‖X‖²`，本文的主要压缩质量度量 |
| TTA (time-to-accuracy) | 到点精度时间 | 达到某个精度目标所需的墙钟时间 |
| `b̄` | 平均位预算 | 每坐标平均位数，**含全部元数据**；网络上每 worker 每坐标发 `2(n-1)/n·b̄` 位 |
| `AR = (n-1)/n` | — | reduce-scatter 与 all-gather 各阶段每 worker 传输的数据比例 |
| DAR | decompress-accumulate-recompress | 中间节点的核心融合 kernel |
| Butterfly / recursive doubling | 蝶形 / 递归倍增 | 跳数为 `log n` 的 all-reduce 拓扑；MSE 上界比 ring 小一个 `n` 因子 |

### B. 复现检查清单

- [x] **代码开源**：是，<https://github.com/CharlesHan24/DynamiQ>（**v3 才有**；v1 是"计划开源"）
- [x] **数据可得**：是。Wikitext-103、UltraChat、MMLU、GLUE 全部公开
- [x] **超参完整**：是。Table 1 给了 tokens/batch、batch size、初始 LR、LinearLR 终止因子与 epoch 数；DynamiQ 配置（`s=16`、`S=256`、`W={2,4,8}`、`b̄=5`、UINT8/BF16 scale）也齐全
- [ ] **随机种子**：**未给**。相关舍入依赖跨 worker 共享 PRNG 种子，论文未说明种子取值或结果的方差；小模型上作者自己承认有训练方差波动
- [ ] **数值确定性**：**未讨论**。随机舍入使得逐 bit 复现不成立
- [x] **硬件要求**：4 × 2 RTX A6000（48 GB）+ 100 Gb/s CX-6 + NVLink NV4。规模很小，学术组可复现
- [ ] **大规模结果**：**不可复现**。DP=8192 那组是合成数据模拟（Appendix E 给了 LogNormal 混合的拟合参数，所以模拟本身可复现，但它不等于真实梯度）
- [ ] **基线适配的可比性**：THC 与 OmniReduce 由本文作者适配到多跳，适配代码是否在 artifact 中未说明；THC 的异常差表现需要独立验证
- [x] **`ε` 与阈值**：非均匀量化的 `ε` 具体取值正文未给（只说"更大的 ε 更向零密集"），需查代码；阈值关系与 Appendix A 的闭式解完整

## 参考

- 论文：<https://arxiv.org/abs/2602.08923>（v1 2026-02-09；v3 = SIGCOMM'26 camera-ready）
- 代码：<https://github.com/CharlesHan24/DynamiQ>
- 与本文冲突的 AMD 侧证据：[`./mxfp4-pretraining.md`](./mxfp4-pretraining.md)（wgrad 量化是收敛主因、随机化在全流水下不收敛）
- RCCL / NCCL 2.28 差距的现状：[`../knowledge/systems/industry-training-optimization-2026.md`](../knowledge/systems/industry-training-optimization-2026.md)（§3.3、C3-4/C3-5）
- Helios / RSN 上的同类工作：[`./ultraep/README.md`](./ultraep/README.md)（点名 AMD Helios 的 rack-scale MoE 负载均衡）
