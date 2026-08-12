# 会议扫描 2026-08：OSDI/EuroSys/NSDI/ASPLOS/MLSys/FAST/PPoPP/CGO/HPCA '26 + SOSP/SC '25

> 依据 [`paper-venues-checklist.md`](./paper-venues-checklist.md) 的 Tier 0/1 名单做的第一轮全量会议扫描，补 [`arxiv-digest-2026-08.md`](./arxiv-digest-2026-08.md) 只扫 arXiv 的缺口。
> **语料**：11 个会议共 **1,249 篇**（OSDI'26 137 · NSDI'26 151 · EuroSys'26 139 · ASPLOS'26 160 · MLSys'26 136 · FAST'26 46 · PPoPP'26 52 · CGO'26 57 · HPCA'26 123 · SOSP'25 67 · SC'25 181）
> **漏斗**：1,249 → 关键词打分 ≥10 得 126 → 去掉 slab 已跟踪的 19 篇 → **107 篇新增**，其中本文列出最值得读的约 40 篇。
> **口径限制**：除 OSDI/NSDI/ASPLOS 外只拿到标题（DBLP 不含摘要与单位），相关性判断基于标题；**下方标注单位的都已逐条核实**，未标注的没核实。

## 这轮扫描验证了什么

**Tier 0 的判断是对的。** OSDI '26 一届里就有一批中国大厂的生产级论文，而这些在之前 2,498 篇的 arXiv 全量扫描里一篇都没出现：字节跳动的两篇 SDC 论文和数据流水线论文、阿里的两篇 RL 训练系统、Infinigence 的 DynaRL、腾讯在 NSDI 的 SwiftEP。**[Tessera](../../papers/tessera.md) 不是孤例，是一整类。**

**同时暴露了我自己打分方法的问题。** 几篇价值最高的论文因为标题短、关键词密度低而被打分排到阈值以下，是靠事后针对性 grep（AMD / megakernel / DSL）才捞回来的：HipKittens、MPK、Event Tensor、Wave 全部如此。这恰好复现了 Tessera §5 记的那条教训——**当下游是排序问题时，尾部误差比平均误差更致命**。后续扫描应当把「关键词打分」和「按主题定向 grep」两条路都跑一遍，不能只靠前者。

## 一、最该先读的（按对我们的相关度排序）

### 1. HipKittens: Fast and Furious AMD Kernels（MLSys '26）

**Stanford（William Hu, Drew Wadsworth, Sean Siddens, Stanley Winata, Daniel Y. Fu, Christopher Ré, Simran Arora）+ AMD（Ryan Swann, Muhammad Osama）**

**已精读** → [`papers/hipkittens.md`](../../papers/hipkittens.md)。本轮**对我们最相关的一篇**。ThunderKittens 原班人马 + AMD 官方人员做的 AMD 版本。ThunderKittens 是 NVIDIA 侧「tile 抽象 + 少量 C++ 模板换取接近手写性能」这条路线的代表，把它搬到 CDNA 上意味着直接回答了我们一直在问的问题：**AMD 上手写 HIP 与 tile 级抽象之间的性能差距到底有多大、抽象要付出什么代价**。与 FlyDSL 的路线选择直接相关，也是 ROCmoe 判断「哪些该手写 HIP」的现成对照。

### 2. MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs（OSDI '26）

**CMU（Xinhao Cheng, Zhihao Zhang, Ruihang Lai, Hongyi Jin, Bohan Hou, Mengdi Wu 等，Zhihao Jia 组）+ NVIDIA（Zihao Ye, Yingyi Huang）+ 清华**

把张量程序自动 mega-kernel 化的编译器与运行时。与我们已精读的 [AutoMegaKernel](../../papers/automegakernel.md) 是同一问题域，但出自 OSDI 且带 runtime。**MonolithEP 的「手写 megakernel」路线必须对照的自动化基线**——AutoMegaKernel 的负面结论（瓶颈在每 tile 跨 SM 同步、且在带宽最高的训练级芯片上最严重）在 MPK 里是否复现，是核心问题。

### 3. Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel（MLSys '26）

**CMU（Hongyi Jin, Bohan Hou, Ruihang Lai, Zhihao Jia, Tianqi Chen, Todd Mowry）+ NVIDIA（Vinod Grover）+ Xupeng Miao**

与 MPK 同一批人的姊妹工作，focus 在**动态** megakernel 的编译抽象。MoE 的路由动态性正是"动态"的典型来源，所以这篇很可能直接触及 MonolithEP 面对的调度问题。建议与 MPK 一起读。

### 4. Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap（OSDI '26）

**UCSD（Xinwei Qiang, Yue Guan, Zhengding Hu, Yufei Ding）+ George Mason/OpenAI（Keren Zhou）+ Meta（Yufei Ding, Adnan Aziz）**

自动化的 chunk 粒度计算-通信重叠。与我们已跟踪的 [AutoOverlap](../../papers/autooverlap/README.md) 是同一赛道的竞争工作，但发在 OSDI 且有 Meta/OpenAI 背景。注意它走的是 **chunk-centric**，而 Tessera §5 明确否掉了 Comet 式沿序列维切分——**Syncopate 的 chunk 切法会不会撞上同样的 GEMM 碎片化问题，是读它时最该带着的问题**。

### 5. Wave: A Symbolic Python DSL And Compiler for High-Performance Machine Learning（MLSys '26）

**Harsh Menon, Oleksandr Zinenko, Ivan Butygin, Martin Paul Lücke, Stanley Winata 等**（MLIR / IREE 核心圈）

符号化 Python DSL + 编译器。作者阵容是 MLIR/IREE 体系的人，**与 FlyDSL 的定位高度重叠**，是路线选型必须对照的一篇。与我们已蒸馏的 [avelang](../libraries/avelang.md) 放在一起看：avelang 走"零抽象、waitcnt 都是一等语法"，Wave 走符号化抽象，两者是同一问题的两个极端。

### 6. Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation（ASPLOS '26）

面向**低精度**的 tile 级 GPGPU 编程语言。低精度 kernel 的表达能力是 FP4/FP8 训练落地的关键卡点（尤其是 block-scaling 的布局），这篇是少见的直接针对该问题的语言设计。

## 二、按主题分组的其余发现

### SDC / 静默数据损坏 —— 一个我们完全没跟踪的生产主题

OSDI '26 一届出现**两篇**，都来自字节跳动，说明这在万卡生产里已是一等问题：

- **Safeguarding LLM Training at Scale: Online SDC Detection and Insights from 35 Million GPU Hours** — 清华（Kinman Lei, Yuyang Jin, Jidong Zhai）+ **字节跳动**（Liyan Zheng, Gaohong Liu, Zuquan Song, Wencong Xiao, Haibin Lin, Xin Liu 等）。**3,500 万 GPU 小时**的在线检测经验
- **SDCs in the Wild: Characterizing and Diagnosing SDC-Defective GPUs in Production LLM Training**（Operational Systems）— 上交（Wenxin Zheng, Xingda Wei, Haibo Chen 等）+ **字节跳动 Seed**

> 这两篇合起来是目前公开材料里关于生产集群 SDC 最硬的数据。我们的 [1024-GPU 稳定性笔记](./training-1024g-stability-interview-notes.md) 里完全没有 SDC 这一层，是明确缺口。

### RL 后训练基础设施 —— 本轮论文数量最爆炸的方向

- **RollArt: Disaggregated Multi-Task Agentic RL Training at Scale**（OSDI '26）— 港科大 + **阿里**（含 Tongyi Lab）
- **Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training**（OSDI '26）— 港科大 + **阿里**
- **RollPacker: Taming Long-Tail Rollouts for RL Post-Training with Tail Batching**（NSDI '26）— 港科大 + **阿里**
- **DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training**（OSDI '26）— 北大 + **无问芯穹 Infinigence** + 中科院计算所 + 清华 + 北航
- **RobustRL: Role-Based Fault Tolerance System for RL Post-Training**（OSDI '26）— 浙大
- **Laminar: A Scalable Asynchronous RL Post-Training Framework**（EuroSys '26）
- **Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter**（ASPLOS '26）
- **Beat the long tail: Distribution-Aware Speculative Decoding for RL Training**（MLSys '26）

> **港科大 Wei Gao / Tianyuan Wu / Lunxi Cao + 阿里 Shaopan Xiong / Siran Yang / Jiamang Wang 这个作者群一届内出了三篇**（RollArt / Weave / RollPacker），是当前 RL 训练系统最活跃的团队，值得整体跟踪。另外「长尾 rollout」在四篇里被独立identified为核心瓶颈，是个已被反复验证的问题。

### 集合通信

- **ForestColl: Throughput-Optimal Collective Communications on Heterogeneous Network Fabrics**（NSDI '26）— 华盛顿大学 + **微软** + 清华
- **HeteCCL: Synthesizing Near-Optimal Collective Communication Schedules for Heterogeneous GPU Clusters**（NSDI '26）— 东北大学 + **阿里云**（Jiamin Cao, Ennan Zhai）
- **FAST: An Efficient Scheduler for All-to-All GPU Communication**（NSDI '26）— CMU + **MangoBoost** + 华盛顿大学。**MangoBoost 是 AMD 生态的加速卡厂商，这篇的 A2A 调度可能带 AMD 侧数据**，值得优先看
- **Multipath Collective Communication Beyond Scale-up Networks in GPU Clouds**（EuroSys '26）
- **COCCL: A Collective Communication Library Supporting Easy Integration and Configuration of Customized Compression**（PPoPP '26）
- **Compression-Aware Gradient Splitting for Collective Communications in Distributed Training**（HPCA '26）
- **Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training**（SOSP '25）

### MoE 训练与服务

- **Sparse Checkpointing for Fast and Reliable MoE Training**（NSDI '26）— **Stanford + NVIDIA**（Swapnil Gandhi, Christos Kozyrakis）
- **SYMI: Efficient Mixture-of-Experts Training via Model and Optimizer State Decoupling**（NSDI '26）— **Stanford + NVIDIA + OpenAI**（Thomas Norrie）
- **SwiftEP: Accelerating MoE Inference with Buffer Fusion and TMA Offloading**（NSDI '26）— **腾讯**（大部分作者）+ 南京大学
- **MoEntwine: Unleashing the Potential of Wafer-Scale Chips for Large-Scale Expert Parallel Inference**（HPCA '26）
- **CRAFT: Fine-Grained Cost-Aware Expert Replication For Efficient MoE Serving**（MLSys '26）
- **Demystifying the Mixture of Experts Serving Tax**（MLSys '26）
- **Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading**（EuroSys '26）
- **EARTH / MoE-APEX**（ASPLOS '26，MoE 加速器与自适应精度专家 offload）

### 流水线并行

- **HelixPipe: Efficient Distributed Training of Long Sequence Transformers with Attention Parallel Pipeline Parallelism**（PPoPP '26）
- **SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training**（SC '25）
- **Revisiting Pipeline Parallelism for LLM Serving**（OSDI '26）— 高丽大学
- **gLLM: Global Balanced Pipeline Parallelism with Token Throttling**（SC '25）

### 低精度与 GEMM kernel

- **M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization**（ASPLOS '26）— 与我们记录的 FP4/MXFP 争议直接相关
- **LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel**（SC '25）
- **HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality**（SC '25）— tile 调度，与 amd-gemm-optimization 技能相关
- **GyRot: Hidden Synergy Between Rotation and Fine-Grained Group Quantization**（HPCA '26）— **旋转（Hadamard）+ 分组量化**，正是 AMD/NVIDIA 在 FP4 上分歧的那个技术点
- **Progressive Low-Precision Approximation of Tensor Operators on GPUs**（CGO '26）

### 编译器与工具链

- **Proton: Towards Multi-level, Adaptive Profiling for Triton**（CGO '26）
- **Triton-Sanitizer: A Fast and Device-Agnostic Memory Sanitizer for Triton**（ASPLOS '26）— device-agnostic，可能覆盖 AMD
- **QIGen: A Kernel Generator for Inference on Nonuniformly Quantized LLMs**（CGO '26）
- **A Reinforcement Learning Environment for Automatic Code Optimization in the MLIR Compiler**（CGO '26）
- **FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection**（HPCA '26）

### AMD 硬件

- **Characterizing Performance, Power, and Energy of AMD CDNA3 GPU Family**（SC '25）— MI300 系列的公开表征数据，可用于校准我们的 roofline 假设

### 训练可靠性与调度（次优先）

Checkmate（NSDI，零开销 checkpoint）· AdaCheck（FAST '26）· Elastor（PPoPP '26）· FLARE（NSDI '26，阿里/上交，千卡级训练发散诊断）· EROICA（NSDI '26，**阿里**在线性能排障）· eGPU（HPCA '26，**万卡级弹性共享，生产规模**）· Zeppelin / Arena / HetAuto（EuroSys '26 并行与调度）· MegaScale-Data / MegaScale-Omni（EuroSys '26，**字节** MegaScale 系列新成员）

## 三、建议的下一步

1. **优先精读 HipKittens**——AMD + Stanford，直接决定 FlyDSL 与手写 HIP 的取舍论据
2. **MPK + Event Tensor 一起读**——对照 AutoMegaKernel 的负面结论，判断 MonolithEP 手写路线的护城河还剩多少
3. **Syncopate**——与 AutoOverlap 对照，重点看它的 chunk 切分是否撞上 Tessera 指出的 GEMM 碎片化问题
4. **两篇 SDC 论文**——补 [1024-GPU 稳定性笔记](./training-1024g-stability-interview-notes.md) 里完全缺失的一层
5. **把 RL 后训练那 8 篇作为一个专题**统一扫一遍，港科大+阿里那个作者群是重点

## 四、复现方式

```bash
# USENIX 系（开放获取，含作者单位）
curl -sL -A "Mozilla/5.0" -o osdi26.html https://www.usenix.org/conference/osdi26/technical-sessions
# 标题在 <a href="/conference/<venue>/presentation/<slug>">TITLE</a>，紧随其后是作者+单位

# ACM/IEEE 系走 DBLP（只有标题+作者，无单位无摘要）
curl -sL https://dblp.org/db/conf/eurosys/eurosys2026.html   # <span class="title" itemprop="name">
# 注意：DBLP 对 OSDI'26 / ASPLOS'26 / MLSys'26 / ISCA'26 尚未索引，需走会议官网

# MLSys 官方 proceedings
https://proceedings.mlsys.org/paper_files/paper/2026
# ASPLOS 官方 program（标题行 + 作者行带单位，best-effort 解析）
https://www.asplos-conference.org/asplos2026/program/
```

**已知局限**：ISCA '26 未纳入（DBLP 未索引且官网抓取失败）；SIGCOMM '26（8 月）、SOSP '26 / SC '26 / MICRO '26（10–11 月）尚未发布，需下轮补扫。
