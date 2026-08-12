# HipKittens：快速而狂暴的 AMD 内核

> 全文中译。原文 [arXiv 2511.08083](https://arxiv.org/abs/2511.08083)（2025-11-11）· [MLSys '26](https://proceedings.mlsys.org/paper_files/paper/2026/file/bc75fa9843a7905bbed9d83895a88f7f-Paper-Conference.pdf) · [代码](https://github.com/HazyResearch/HipKittens)
> 原文许可：**CC0 1.0 Universal（公共领域奉献）**，作者已放弃著作权，翻译与再分发不受限制。
> 技术解读与延伸讨论见 [`hipkittens.md`](./hipkittens.md)。
>
> **翻译体例**：正文、表格、图注、脚注逐段全译；专有名词（wave、tile、swizzle、bank conflict 等）保留英文并在首次出现处给出中文；指令名、代码、参考文献保持原样；附录 E 的长代码清单只译说明文字，代码本身见原文与仓库。

**作者**：William Hu△、Drew Wadsworth△、Sean Siddens†、Stanley Winata†、Daniel Y. Fu‡、Ryan Swann†、Muhammad Osama†、Christopher Ré△、Simran Arora△

△ 斯坦福大学　† Advanced Micro Devices, Inc.　‡ 加州大学圣地亚哥分校

`{willhu, simarora}@stanford.edu`　2025 年 11 月 11 日

---

## 摘要

AMD GPU 提供了当前最先进的算力与显存带宽；然而，能跑到峰值性能的 AMD 内核是用裸汇编写的。为了解决把 AI 算法映射到硬件上的困难，近期工作提出了 ThunderKittens（TK）这类嵌入 C++、受 PyTorch 启发的领域专用语言（DSL），以简化 NVIDIA 硬件上的高性能 AI 内核开发。我们探究这样一批原语——用于显式的、基于 tile 的编程，配以优化过的访存和跨 worker 的细粒度异步执行——在多大程度上是 NVIDIA 特有的、又在多大程度上是通用的。我们给出了第一份关于「哪些编程原语能带来高性能 AMD AI 内核」的详细研究，并把这些洞见封装进 HipKittens（HK）编程框架。我们发现，先前 DSL 所用的基于 tile 的抽象可以推广到 AMD GPU，但我们需要重新思考在 AMD 上实例化这些抽象的算法。我们在 CDNA3 与 CDNA4 两个 AMD 平台上验证了 HK 的原语。在评测中，HK 内核在 GEMM 与 attention 上可与 AMD 手工优化的汇编内核相竞争，并一致地优于编译器基线。此外，汇编难以扩展到 AI 工作负载的广度；反映这一点，在某些设定下 HK 比所有可得的内核基线快 1.2–2.4×（例如 d = 64 的 attention、GQA 反向、访存瓶颈型内核）。这些发现有助于为「一个跨 GPU 厂商通用的、基于 tile 的高性能 AI 内核软件层」铺路。HipKittens 已开源：<https://github.com/HazyResearch/HipKittens>。

---

## 1 引言

尽管 AI 过去在很大程度上只用了单一硬件厂商 [2, 16, 26]，AMD 的 GPU 硬件如今已提供业界领先的峰值算力与显存带宽（表 2）。然而，成熟软件支持的缺失造就了一场硬件彩票（"CUDA 护城河"）[29, 30]。峰值性能的 AMD 内核由少数专家用裸汇编写成（即 AITER 库 [3]），这种方式很难快速铺开到 AI 工作负载的广度。举例来说，在 AMD MI355X GPU 上，AITER 和 PyTorch 的 Llama GQA 反向分别只达到 SoTA 性能的 30% 和 24%（第 4 节）。

几年前，开发 NVIDIA 内核同样需要极其艰苦的努力。例如，用底层的 CUDA / CUTLASS，从 H100 GPU 发布到峰值性能的开源 attention 内核发布，中间隔了两年 [31]。Triton [34] 这类编译器更易用，但牺牲了性能，并且难以快速支持新的硬件特性 [33, 35]。AI 设计的内核已显露早期的希望 [9, 17]，但当前模型同样难以驾驭新硬件特性 [27]，且易受奖励攻击（reward hacking）影响 [9]。近来，像 ThunderKittens（TK）这样轻量的、嵌入 C++ 的 DSL（以及 CuTe DSL [24]、Gluon [38] 等后继者）考虑用一小组"有主见的"（opinionated）原语来编码内核设计、把完全的控制权交还给开发者，从而简化开发：

**1. Tile。** 基本数据类型是带有优化访存模式的 tile。TK 在 tile 之上暴露了一批轻量的、受 PyTorch 启发的批量计算算子（`mma`、`exp` 等），内部包装 PTX。Tile 帮助开发者在 GPU 存储层级的每一层上显式管理数据。

> **图 1**：我们研究现有的基于 tile 的编程原语是否足以支撑 AMD 内核，还是需要全新的原语。我们的研究得到了 HipKittens：一组精简且有主见的原语，用于写出快速而狂暴的 AMD 内核。HK 引入了一种通用的 8-wave ping-pong 调度来重叠计算与访存、由程序员控制的寄存器分配，以及高效的共享内存与 chiplet 感知的 swizzle 算法，从而支撑起一整套高性能的 AMD AI 内核。

**2. 重叠（Overlapping）。** 少数几种基本的内核模式，能帮助开发者达到高占用率（occupancy），或者说把 worker（AMD 上是 wave，NVIDIA 上是 warp）调度到不同的硬件执行单元上。现代 NVIDIA 内核已经收敛到 wave specialization（生产者–消费者）这一类调度模式 [31, 32, 33, 36, 37]。

**3. Grid 调度。** 通过以恰当的顺序把工作分配给 thread block，开发者可以最大化对不可编程 cache 的复用。

我们的工作要问的是：简化 AMD 内核开发是否需要全新的编程原语，还是现有原语已经够用？理想情况下，我们希望有一个简单的框架，能帮开发者写出一大类高性能内核。我们的探索得到了 HipKittens（HK）：一组面向 AMD 的、嵌入 C++ 的精简编程原语。

**1. 为可编程 GPU 存储优化访存模式。** 细致的寄存器内存管理对峰值性能内核至关重要。HK 保留了先前 DSL 的 tile 数据结构来帮助开发者管理内存 [33]。但把 tile 针对 AMD 做优化会带来新挑战。Triton、HIPCC 这类编译器经常干扰内核开发者精细安排寄存器分配与生命周期的能力（3.2 节）。例如，HIPCC 不允许 HIP 开发者把某类寄存器（AGPR）用作矩阵指令的输入操作数[^1]。因此我们引入了一种让开发者彻底绕开编译器、显式钉住（pin）每个 tile 所属寄存器的机制。至于访存模式：NVIDIA 不同的矩阵指令形状全都由同一套底层 core matrix 结构搭出来，这使得像 TK 和 Linear Layouts [38] 那样对所有形状使用单一 tile swizzle 策略变得容易。而 AMD 的矩阵指令缺乏这种可组合结构，导致 tile 布局数量爆炸。更进一步，AMD 上共享内存的 bank 结构、以及一个 wave 内线程的执行顺序，会随访存指令的不同而不同（3.2 节）。HK 在创建 tile 时替开发者处理掉这些复杂性。

[^1]: AMD CDNA 的每个 SIMD 有 512 个寄存器，会被均分给同驻于该 SIMD 上的各个 wave。对于每个 SIMD 只有一个 wave 的内核，硬件把寄存器切成 256 个向量通用寄存器（VGPR）和 256 个累加器寄存器（AGPR）。
> 译注：原文此处为 "For kernels with a single SIMD per wave"，按上下文应为「每个 SIMD 一个 wave」。

**2. 重叠计算与访存的调度。** 理想情况下，我们希望有一批简单、可复用、能跨 AI 工作负载泛化的调度模式来安排内核内部的计算与访存。Wave specialization 模式主导了 NVIDIA 的内核与 DSL：生产者 wave 负责访存操作，消费者 wave 在大 tile 上执行批量计算。但我们发现，由于架构差异，这个模式在 AMD CDNA3 与 CDNA4 GPU 上表现不佳——**AMD 的静态寄存器分配意味着生产者 wave 占用了寄存器却不贡献计算**，这限制了每个 thread block 能计算的输出 tile 尺寸，进而限制了内核的算术强度。在 MI355X 上，wave specialization 只达到 BF16 GEMM 峰值性能的 80%（表 2）。

### 硬件概览

| 规格 | NVIDIA B200 SXM5 | AMD MI355X OAM |
|---|---|---|
| BF16 matrix / tensor | 2.2 PFLOPs | 2.5 PFLOPs |
| MXFP8 matrix / tensor | 4.5 PFLOPs | 5.0 PFLOPs |
| MXFP6 matrix / tensor | 4.5 PFLOPs | 10.1 PFLOPs |
| MXFP4 matrix / tensor | 9.0 PFLOPs | 10.1 PFLOPs |
| 显存容量 | 180 GB | 288 GB |
| 显存带宽 | 8.0 TB/s | 8.0 TB/s |

> **图 2：硬件概览。**（左）最新一代 GPU 平台的峰值访存与计算速度 [7, 23]。（右）AMD GPU 软硬件层级示意图。

我们识别出两种能一致跑到 AMD 峰值性能的替代调度模式：**8-wave ping-pong** 与 **4-wave interleave**，二者在可编程性与性能之间做权衡（表 3）。8-wave 模式在每个 SIMD 上放两个 wave，它们在计算与访存角色之间交替，每个都执行大块的批量 tile 操作（图 1）；4-wave 模式在每个 SIMD 上放一个 wave，并在小 tile 尺寸上精细交织指令。值得注意的是，简单的 8-wave 模式就足以在 BF16 GEMM、FP8 GEMM 和 attention 前向上追平 AMD 手工优化的汇编内核，并在 GQA 非因果反向上以 1.8× 超过基线。

**3. 为不可编程 GPU 存储优化访存模式。** Chiplet 架构正成为 GPU 扩展的主导路径——NVIDIA Blackwell 用了 2 颗芯片，AMD MI355X 用了 8 颗——但现有框架忽视了它们的层级化 cache 结构，把性能白白留在桌上。每颗 AMD CDNA4 chiplet 含 32 个处理器并带有私有 L2 cache，而所有 chiplet 共享一个位于 L2 与全局内存之间的末级缓存（LLC）。这两级层级化 cache 对「工作如何在 thread block 间并行化」有着彼此正交的偏好。表 4 展示了这样一个实例：对 BF16 GEMM，朴素的 row-major 顺序分配工作给 thread block 只得到 36% 的 L2 命中率。我们进一步说明，只针对 L2 复用做优化（例如把命中率提到 79%）会劣化 LLC 表现和总带宽。HK 引入了一个在调度 thread block 时同时建模两级 cache 的算法，相比朴素 row-major 基线提升 19% 性能（表 4）。

**评测。** 我们在 AMD CDNA3 MI325X 与 CDNA4 MI355X GPU 上验证 HipKittens。在 AI 中使用最广、被优化得最充分的工作负载上（BF16 / FP8 GEMM、GQA/MHA attention 前向与反向、RoPE、LayerNorm），HK 与所有 AMD 基线相竞争或胜出。HK 平均优于所有可得的 AMD 基线，包括 AMD 用裸汇编手工优化的内核。然而汇编并不是可扩展的内核开发方式，会让许多重要的 AI 工作负载得不到支持——在这类场景下（例如某些 attention 形状、GQA 反向、访存瓶颈型内核），HK 比可得的 AMD 基线快 1.2–10×。此外，HK 一致地优于编译器路线（例如比 Triton BF16 GEMM 快最多 3×、比 Mojo MHA 前向快 2×）。

我们的贡献是：（1）写高性能 AMD 内核的若干原则；（2）HK，一组献给 AI 社区的、有主见的 C++ 编程原语；（3）一套高性能的 AMD 内核。我们进一步表明，TK DSL 中提出的 tile 原语可以迁移到 AMD，这为「跨 AI 加速器存在一个统一且高性能的编程模型」提供了证据。把内核支持扩展到多个硅平台，是解锁"实现 AI 全部潜能所需算力"的关键 [25]。我们希望这项工作有助于打开 AI 的硬件版图。

---

## 2 背景

本节在 2.1 节给出 AMD GPU 硬件背景，在 2.2 节讨论相关工作。附录 B 提供了对相关工作的扩展讨论。

### 2.1 GPU 基础

GPU 内核是一些小程序：装载数据、对其做运算、把结果写回内存。本文总体采用 AMD 术语，并在附录 A 给出 AMD 与 NVIDIA 术语的对照。

**1. 计算层级。** 内核由分布在数百个处理器（称为"计算单元"，CU）上的数万个线程执行。CU 把自己的硬件资源组织为 4 个 SIMD（单指令多数据）单元。线程按层级排列：线程是最小执行单位；"wave"是 64 个线程的组，在单个 SIMD 上以 lockstep 方式执行；"thread block"是若干 wave 的组，被联合调度到 CU 上。AMD MI355X GPU 含 256 个 CU，以 chiplet 布局组织成 8 个加速器复合芯片（XCD），每个 32 个 CU。

**2. 存储层级。** 内存按层级组织：少量快速访问的存储 + 大量慢速访问的存储。单个 SIMD 含 512 个 32-bit 向量寄存器（每个 CU 合计 512 KB）。每个 CU 有 L1 cache 和共享内存，后者可被同一 thread block 内的多个 wave 访问。每个 XCD 共享一块 4 MB 的不可编程 L2 cache。所有 CU 共享一块大而慢的全局内存（HBM），并有一个末级缓存（LLC）位于 L2 与 HBM 之间。

**3. 占用率（Occupancy）。** 线程在物理执行单元（ALU、FMA、matrix core）上执行指令，这些单元针对不同类型的计算做了专门化。这些单元执行的指令各有固定的发射延迟和有限的带宽。不同 wave 可以同时占用不同单元，以避免打满任何单一单元。每个单元对内存布局（即逻辑数据元素到物理线程归属的映射）施加不同约束 [6]。

**软件概览。** 开发者可以在软件栈的不同层级写内核。裸汇编提供对寄存器使用、指令选择与排序的最大控制。CUDA / HIP C++ 经（NVCC、HIPCC）编译为汇编，编译器可能引入自己的指令重排和寄存器生命周期跟踪。LLVM 接受编译器提示（hints），让开发者引导编译器行为。有些编译器在 C++ 之上暴露高层接口（如 Python [28]、Triton [34]）。

### 2.2 相关工作

目前，峰值性能的 AMD 内核在裸汇编里精细交织计算与访存的指令发射（见 AITER 与 Composable Kernel 库）[3, 4]。相对地，为了简化并加速内核开发流程，AI 社区近来采用了在优化过的 tile 原语之上做批量编程算子的做法，如 ThunderKittens [33][^2] 及其后继者（如 CuTe DSL [24][^3]、Gluon [38][^4]）所提出。然而这些现有的基于 C++ 的 DSL 只跑在 NVIDIA GPU 上，包装的是 PTX 与 CUDA。Triton、TileLang、Mojo 这类编译器库建立在 LLVM/MLIR [18, 19, 20] 之上，可以为 AMD GPU 编译。但这些工作既没有为 AMD 提供可复用的原则或原语，也没有放出成体系的高性能 AMD 内核套件。例如，Mojo 的 MHA 内核受累于昂贵的 bank conflict，在 MI355X 上只达到峰值内核性能的 50%[^5]。HipKittens 提供了第一套系统性的 AMD AI 内核原语，朝着打开硬件版图迈进。

[^2]: <https://github.com/HazyResearch/ThunderKittens>（2024 年 5 月）
[^3]: <https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html>（2025 年 9 月）
[^4]: <https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon>（2025 年 6 月）
[^5]: 测量方式：`rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS --output-format csv --output-file profiles_3 -d out -- mojo bench mha.mojo`，代码取自 <https://github.com/modular/modular/tree/main/max/kernels/benchmarks/gpu>，在 2025 年 11 月 6 日的 Modular nightly 构建、MI355X GPU 上测得。

---

## 3 HipKittens

本节描述 HipKittens（HK）——一个在 AMD GPU 上编写 AI 内核的框架。HK 建立在 ThunderKittens 框架 [33] 之上，后者用嵌入 C++ 的、基于 tile 的编程原语来简化高性能且灵活的 AI 内核开发（见 3.1 节）。我们在 3.2 节描述 HK 优化可编程 GPU 存储访存模式的原则，在 3.3 节描述最大化占用率的原则，在 3.4 节描述优化不可编程 cache 访存模式的原则。

### 3.1 Tile 编程接口

与现有内核框架一样，HK 采用 tile 作为基本数据结构，并在 tile 之上提供一套优化过的算子。tile 的设计与算子集深受 PyTorch 与 NumPy [14, 28] 启发，因为它们为 AI 社区所熟悉。

| 方法 | 序列长度 | TFLOPS |
|---|---|---|
| HK | 4096 | 855 |
| HK + 钉住寄存器 | 4096 | 1024 |
| AMD 汇编（AITER） | 4096 | 1018 |
| HK | 8192 | 909 |
| HK + 钉住寄存器 | 8192 | 1091 |
| AMD 汇编（AITER） | 8192 | 1169 |

> **表 1：显式寄存器调度带来更强的开发者控制力。** 一个用 HIP 实现的 4-wave MHA 非因果反向内核，由于编译器限制而不及 AMD 的裸汇编内核（AITER）。我们通过绕开编译器、把寄存器 tile 钉到显式寄存器上，追平了 AITER。使用 batch size 16、heads 16、head dim 128。

- **存储。** 开发者可以在寄存器或共享内存中初始化 tile。一个 tile 由 dtype（FP32、BF16、FP16、FP8、FP6）、行数、列数和布局（行主序或列主序）参数化。tile 的行数与列数被限制为 matrix core 形状的倍数。HK 提供算子，在 GPU 存储层级的不同层之间装载与存储 tile。
- **计算。** HK 在 tile 之上提供一套批量计算算子，受 PyTorch 算子集启发（如 `mma`、`exp`、`add`）。这些函数是轻量的、不引入额外开销，因为它们直接包装裸 AMD CDNA 汇编 / HIP（TK 则是 NVIDIA PTX / CUDA）。

有了这些熟悉的编程原语，HK 会自动为 tile 优化访存模式。AMD GPU 上的内存管理在层级的每一层都提出了关键挑战，接下来讨论。

### 3.2 优化可编程存储的访问

下面讨论 HipKittens tile 的具体细节。

#### 3.2.1 开发者控制的寄存器调度

细致的寄存器管理对高性能至关重要。然而编译器要么阻止（如 Triton）、要么妨碍（如 HIPCC 编译器）开发者对寄存器分配的最大化控制。举例来说，在每个 SIMD 只有一个 wave 的内核里，AMD 硬件把该 SIMD 的 512 个寄存器切成 256 个向量通用寄存器（VGPR）与 256 个累加器寄存器（AGPR）。然而，尽管硬件确实支持把 AGPR 作为 matrix core 指令的输入，HIPCC 却不支持。对于同时涉及矩阵与向量运算的工作负载（如 attention 反向），经 HIPCC 编译的内核需要生成冗余的 `v_accvgpr_read` 指令，在发射矩阵指令之前把数据从 AGPR 搬到 VGPR。

**显式寄存器调度。** 这些编译器约束促成了 HK 中的一个特性：让开发者完全控制寄存器调度。开发者钉住每个 tile 所属的寄存器，而不是交给 HIPCC 管理。通过绕开编译器，开发者可以把 AGPR 用作矩阵指令的输入，由此得到我们达到 SoTA 水平的反向 attention 内核（表 1）。使用钉住寄存器 tile 的编程接口与使用标准的、编译器管理的寄存器 tile **完全一致**。我们保留两种选项，好让开发者自行选择想要的控制粒度。

#### 3.2.2 面向异构 matrix core 形状的 tile

AI 内核会根据工作负载性质使用不同的 matrix core 指令形状（M×N×K），以便细致地管理寄存器压力。然而在 AMD GPU 上同时使用多种形状是有挑战的。

**矩阵布局复杂性。** 回忆一下，GPU 矩阵指令规定了每个数据元素由哪个线程持有在其寄存器中。此外，如果一个 wave 内的多个线程试图同时访问同一个 bank，共享内存访问就会产生 bank conflict。Wave（以及 NVIDIA 的 warp）分**相位（phase）**执行共享内存访问；也就是说，一个 wave 中的一部分线程并发地访问共享内存 [13]。AMD 矩阵布局相对 NVIDIA 布局的复杂性，影响了 GPU 存储层级每一层上的访存模式。

第一，NVIDIA 的矩阵指令使用规则的模式（图 3a）：所有形状都由一个底层的 16 × 16 core matrix 构件组成，按整体矩阵指令形状的需要重复"盖章"若干次。因此，TK [33]、Linear Layouts [38] 这类先前框架可以使用一个能跨矩阵形状泛化的统一 swizzle 策略。而 AMD 的每条矩阵指令都用一个完全不同的布局，没有类似的底层结构。第二，NVIDIA 指令按顺序把线程分配到各相位（例如线程 0–7 在第一相位，8–15 在第二相位）；而在 AMD 上，相位是非顺序的，且随访存指令不同而不同[^6]。

[^6]: 例如，一个 wave 中的线程分 4 个相位执行 `ds_read_b128` 指令、从 64 个各 4 字节宽的共享内存 bank 装载数据；而 `ds_read_b96` 分 8 个相位执行、从 32 个 bank 装载。这些相位在 CDNA ISA 中**没有文档**，因此我们做了一个求解器来确定它们，并把相位记录在表 5 中。

> **图 3：NVIDIA 与 AMD GPU 上的矩阵布局。** 每个矩阵中的阴影格子代表线程 0 所拥有的元素。(a) NVIDIA core matrices；(b) AMD 16×16×32 MFMA 的 A/B 矩阵；(c) AMD 16×16×32 的 C/D 矩阵。

> **图 4：一个 16×32 BF16 tile 的 swizzle 模式。** AMD CDNA4 GPU 上共享内存的 bank 行为随指令不同而不同。`ds_read_b128` 通过 64 个各 32-bit 宽的 bank 访问共享内存，对应图中各个格子与数字。阴影格子代表：对一个 16×32 行布局寄存器 tile，`ds_read_b128` 指令第一相位所访问的 bank。左边是未 swizzle 的布局，遭受 2 路 bank conflict；右边是 swizzle 后的布局，无 bank conflict。这里施加的 swizzle 是：从第 8 行开始，把前 8 列与后 8 列对调。这一 swizzle 策略同时使得用 `ds_read_b64_tr_b16` 做列主序读取时也无 bank conflict。细节见 D.1。

**优化的 tile 存储。** 下面讨论 HK 如何把这些复杂性从内核开发者面前抽象掉：

1. **寄存器。** 默认情况下，HK 中的寄存器 tile 使用**最小**的 MFMA 指令，因为这能提供最大的调度控制力（见 3.3 节强调）。不过，对于要用其他尺寸的边缘情形内核，HK 允许开发者按 MFMA 指令形状来参数化所需的寄存器 tile。
2. **共享内存。** 在 AMD GPU 上，不可能对所有布局使用单一 swizzle 模式（附录 D.1 给出了一个简单反例）。虽然我们可以为每一种矩阵布局实现独有的 swizzle 模式，但这会增加代码复杂度。我们转而识别出那些**常常共同出现**的布局，并支持对这些情形无 bank conflict 的 swizzle 模式。图 4 展示了这样一个 swizzle：它对 16 × 32 的行布局与列布局装载都无 bank conflict。
3. **全局内存。** AMD GPU 支持 HBM 到共享内存的直接异步装载。与 TMA 类似，这些装载绕过寄存器文件。该指令接受每线程的 HBM 地址作为输入，每个线程将从这些地址读取数据。TK 这类 DSL 是直接对共享内存地址做 swizzle，而在 AMD 上，对共享内存的 swizzle 反而是通过**对 HBM 地址做 swizzle** 来完成的。

### 3.3 重叠计算与访存的利用率

我们研究了 AMD AI 内核内部指令调度的原则，并识别出两种能在多样工作负载上带来峰值利用率的高性能模式。

**当前做法及其局限。** 最先进的 AI 内核与 DSL 已经收敛到 wave specialization——一种由专门的生产者 wave 处理数据搬运、消费者 wave 处理计算的模式。这一做法主导了 NVIDIA 侧的实现，包括 FlashAttention-3 [31]、面向 MoE 的 COMET [37]、以及 GEMM [10]，还有 TK [33]、TileLang [36] 这类内核 DSL。在这一范式下，wave 长时间占据特定硬件单元，因而可以在大 tile 原语之上发射批量操作。这种基于 tile 的编程让代码体积紧凑、可读性好。

然而，由于根本性的架构差异，wave specialization 难以推广到现代 AMD 设备。相反，最先进的 AMD 内核（AITER [3]、CK [4]）诉诸裸汇编来精细交织指令发射——这一路线与基于 tile 的编程正交。虽然看上去 AMD 似乎需要为每种 AI 工作负载定制调度，但我们识别出了一些简单的通用原则，能在多样应用上取得高性能。

#### 3.3.1 Wave specialization 在 AMD 上表现不佳

NVIDIA 内核实现 wave specialization，依靠的是专用访存硬件（`tma`）、可直接从共享内存或 tensor memory 接受操作数的异步矩阵乘（`wgmma`、`tcgen05`）、由每处理器大容量共享内存所支撑的深流水（B200 的每处理器 SRAM 比 AMD MI355X 大 40%）、寄存器再分配（TMA 的寄存器高效性让生产者把寄存器让给消费者），以及硬件同步原语（`mbarrier`）。AMD 缺少这些架构特性，改变了内核的设计空间。

为评估这些差异如何影响性能，我们变动了同步机制、流水深度与生产者–消费者比例（表 2）。实验揭示了两条原则：我们需要**最大化每个 thread block 计算的输出 tile 尺寸**以提高算术强度（每搬运一字节所做的运算数），并且需要**最大化流水深度**以隐藏访存装载的延迟。

| # P / # C | MFMA 形状 | 输出 tile | TFLOPS |
|---|---|---|---|
| HK 4 / 8 | 16 × 16 × 32 | 128 × 256 | 893 |
| HK 4 / 12 | 16 × 16 × 32 | 192 × 256 | 1278 |
| HK 0 / 8 | 16 × 16 × 32 | 192 × 256 | 1281 |
| HK 0 / 8 | 16 × 16 × 32 | 256 × 256 | **1610** |
| TK | 256 × 256 × 16 | 256 × 256 | 1538 |
| CUTLASS | 256 × 256 × 16 | 256 × 256 | 1570 |

> **表 2：生产者–消费者对比。** 我们报告一系列生产者–消费者 BF16 GEMM 内核的结果，形状 M = N = K = 8192。P 与 C 分别表示生产者与消费者数量。我们标注了底层矩阵指令尺寸、每个 thread block 计算的输出 tile 尺寸，以及测得的 TFLOPS（500 次预热迭代、100 次测量迭代，输入取自 N(0, 1)）。AMD 内核跑在 MI355X 上，NVIDIA 内核（TK、CUTLASS）跑在 B200 上。

峰值性能的开源 TK 与 CUTLASS profiler 选出的 GEMM，在 B200 上使用 wave specialization 与 256 × 256 的输出 tile 尺寸[^7]。

[^7]: 该 profiler 会扫描并调优整套 CUTLASS GEMM，为给定形状与 dtype 选出最好的那个。

我们最好的 AMD GEMM，只有在**完全不用 wave specialization**（即零生产者）、每个 thread block 计算 256 × 256 输出 tile 时，才达到可比的性能；随着生产者数量增加，性能下降（表 2）。原因是 AMD 硬件把寄存器**静态**地均分给所有 wave [5]，这意味着生产者占用了寄存器却不贡献输出计算。这就限制了使用 wave specialization 时可用的输出 tile 尺寸。

| 内核 | 模式 | 代码行数 | TFLOPS |
|---|---|---|---|
| FP8 GEMM | 8-wave | 48 | 3222 |
| FP8 GEMM | 4-wave | 183 | 3327 |
| MHA 反向 | 8-wave | 331 | 894 |
| MHA 反向 | 4-wave | 989 | 1091 |

> **表 3：AMD 的调度模式。** 我们识别出两种主要范式——8-wave 与 4-wave——它们能跨工作负载泛化。两种模式都能利用 HK 的 tile 原语。我们报告热循环的代码体积与 TFLOPS，展示这两种模式如何在可编程性与性能之间权衡。

**权衡。** NVIDIA 更大的共享内存使其能在使用大矩阵指令形状（如 256 × 256 × 16）的同时使用深流水。但 AMD 更小的 tensor core 形状（如 16 × 16 × 32）提供了另一条建立深流水的路径：使用更细粒度的装载与计算阶段。

NVIDIA 的矩阵乘指令可以从共享内存或 tensor memory 接受操作数，有助于缓解寄存器压力；AMD 没有这一特性却能追平性能，可能令人意外。不过，AMD 设备有 2× 大的寄存器文件来补偿。

我们还验证了：用共享内存原子操作代替 `mbarrier` 只带来可忽略的开销；我们发现使用原子操作的 192 × 256 生产者–消费者内核，与我们非 wave-specialized 的内核表现相近，这强调了**输出 tile 形状才是影响性能的主导因素**（表 2）。

#### 3.3.2 面向 AMD AI 内核的高性能调度模式

AMD GPU 每个 CU 有四个 SIMD 单元，调度在同一 SIMD 上的 wave 可以重叠计算与访存指令。我们识别出两种以不同方式利用这一并行性、并能在各类 AI 工作负载上一致达到峰值性能的调度模式：

**1. 8-wave ping-pong（负载均衡的工作负载）。** 该模式每个 thread block 用八个 wave——每个 SIMD 常驻两个。这些 wave 被分为两组、每组四个，每组在每个 SIMD 上占一个 wave。在每个 SIMD 内部，两个 wave 交替各自的工作类型：一个只发射计算指令，另一个只发射访存指令，然后二者互换角色，在计算与访存之间来回翻转，如图 1 所示。一个**条件屏障**（conditional barrier）控制这种交替。当计算与访存的时长大致均衡时，该模式表现出色：某个 SIMD 的计算 wave 执行矩阵乘加（MFMA）指令，而与之配对的访存 wave 预取下一批数据，从而有效隐藏访存。

**2. 4-wave interleave（负载不均衡的工作负载）。** 该模式在处理器的四个 SIMD 上各放**恰好一个** wave。每个 wave 以精心错开的次序同时发射计算与访存指令，以最大化硬件单元的占用率。当工作负载不均衡时（偏计算或偏访存），这种细粒度模式能更好地同时打满 MFMA 与 LDS 流水线。每 SIMD 一个 wave 可以动态调整其指令组合。

这些调度在可编程性与性能之间权衡。HK 让开发者用基于 tile 的原语实现其中任一模式，只是 tile 粒度不同。8-wave ping-pong 允许使用与 wave specialization 中相似的大 tile 原语。另一方面，4-wave interleave 要求开发者用小的 base tile 原语编程，由于指令发射粒度更细，代码体积会变大。表 3 刻画了这一权衡。令人惊讶的是，我们发现 8-wave 就足以在 BF16 GEMM、FP8 GEMM 与 attention 前向上追平或超过 AMD 的裸汇编内核。在 GQA 非因果 attention 反向上，我们的 8-wave 内核比基线（PyTorch SDPA、CK、AITER）快 1.8×，而 4-wave 内核带来了更大的 2.3× 加速。

### 3.4 优化不可编程 GPU 存储的访问模式

现代 GPU——AMD 与 NVIDIA 都是——正在从单片（monolithic）走向 chiplet 架构（例如 Blackwell 由两颗芯片组成）。这带来了**解耦的 cache 层级**：不同的处理器簇挂在 GPU cache 的不同切片上（见图 2）。这里我们探讨解耦 cache 调度的原则，并介绍 HK 的 cache 复用算法。

| Block 顺序 | L2 % | LLC % | 显存带宽 | TFLOPS |
|---|---|---|---|---|
| **矩阵乘（M=N=K=9216，MT 192×256×64）** | | | | |
| Row-major | 55% | 95% | 15.1 TB/s | 1113 |
| XCD (W7/C216) | 79% | 24% | 14.9 TB/s | 991 |
| XCD (W5/C25) | 75% | 93% | 18.3 TB/s | **1145** |
| **矩阵乘（M=N=K=14592，MT 192×256×64）** | | | | |
| Row-major | 36% | 76% | 10.7 TB/s | 900 |
| XCD (W8/C542) | 79% | 7% | 13.9 TB/s | 980 |
| XCD (W8/C64) | 78% | 55% | 16.6 TB/s | **1068** |

> **表 4：为 cache 复用做 chiplet swizzle。** 对一个 M = N = K = 9216 BF16 GEMM 的输出矩阵，可视化三种不同的 grid 调度。颜色代表在整个 GPU（256 个 CU）上被调度的第一批 thread block 的 XCD 归属。调度 5a（表格第 1 行）按 block ID 分配 block 到 grid。调度 5b（第 2 行）与 5c（第 3 行）以不同的窗口与 chunk 尺寸参数应用算法 1。表 4 展示了这些调度如何在 L2 与 LLC 复用之间权衡以换取性能。图 18a 提供了 14592 形状的对应可视化。

**代价模型。** AMD 设备使用两类 cache——L2 与 LLC——其中 cache miss 的最坏情形惩罚为：L2 300 ns、LLC 500 ns。AMD 设备把 32 个（CDNA4）或 38 个（CDNA3）计算单元分配到一个簇（加速器复合芯片，XCD），每 GPU 含 8 个簇。硬件调度器以 round-robin 顺序把 thread block 分配给各 XCD。grid 调度（即分配给 thread block 的工作顺序）影响 cache 复用与实际达到的带宽：

$$\text{带宽} = \text{LLC 带宽} \times \text{LLC 命中率} + \text{L2 带宽} \times \text{L2 命中率} \tag{1}$$

在一个 GEMM 内核（D = AB + C）中，每个 thread block 计算输出矩阵 D 的一个不同 tile。当 thread block 按朴素的 row-major 顺序调度时，cache 复用是次优的（≈ 55%），因为共享同一个 L2 cache 的 block 常常装载 A 与 B 中**不同、不重叠**的 tile。于是它们的访存无法利用空间局部性，导致冗余的数据搬运。这一行为如图 5a 与表 4（第 1 行）所示。为缓解这一点，我们用两条关键原则改进 cache 复用：

1. **L2 复用。** 映射到同一 XCD（因而共享一个 L2 cache）的 thread block，应当覆盖输出矩阵的一个**矩形区域**——一个"L2 tile"。这种布局确保相继的 block 复用 A 的相同行与 B 的相同列。然而，只针对 L2 局部性做优化，会导致每个 XCD 取用 A 与 B 中互不相交的部分，在下一级 cache 上造成冗余装载。
2. **LLC 复用。** 为了进一步改善末级缓存（LLC）上的复用，我们必须协调**跨 XCD** 的访问。理想情况下，所有 XCD 的合并访问足迹——"LLC tile"——应当在 A 和 B 上都有重叠。换句话说，多个 XCD 应当在输入矩阵的邻近或相同区域上工作，以便共享数据能常驻 LLC。

通过联合优化这两条原则，我们可以同时提高 L2 与 LLC 命中率，带来更高的有效带宽（图 5c、表 4 第 3 行）。例如，表 4 显示一个 L2/LLC 感知的调度比默认 grid 顺序高出多达 15% 的性能。当输出矩阵的宽度（以 tile 计）与 XCD 数量**互质**时，收益尤其显著——例如 AMD MI355X 上 57 个 tile 对 8 个 XCD——因为默认调度会造成最坏情形的复用模式（表 4）。

#### 算法 1：面向 GEMM 的 cache 复用 XCD swizzle

> 译注：PDF 文本层中该算法的伪代码被表格化排版打散，逐行公式无法完整还原。下面保留可恢复的输入/输出签名与每一步的注释语义；精确表达式请参见原文图版或[开源实现](https://github.com/HazyResearch/HipKittens)。

**输入**：grid block 索引 (b.x, b.y, b.z)；grid 维度 (g.x, g.y, g.z)；XCD 数量 nXCD；问题规模 M、N；block 尺寸 BLOCK_M、BLOCK_N；窗口高度 W；chunk 尺寸 C
**输出**：重映射后的 tile 索引 (b.x′, b.y′, b.z)

1. `blocks_per_batch ← g.x × g.y` ▷ 每个 batch（单个 b.z 切片）的 block 数
2. `xy ← b.x + g.x × b.y` ▷ 在 batch 内把 (b.x, b.y) 展平
3. `blocks_per_cycle ← nXCD × C`
4. `limit ← ⌊blocks_per_batch / blocks_per_cycle⌋ × blocks_per_cycle` ▷ 最大的 (nXCD × C) 对齐前缀
5. **if** `xy > limit` **then**
6. 　 保持 `xy` 不变 ▷ 尾部区域：顺序不变
7. **else**
8. 　 `xcd ← (xy mod blocks_per_cycle) / C` ▷ 该 block 属于哪个 XCD（round-robin）
9. 　 `local_idx ← ...` ▷ 按 XCD 解交织后的局部索引
10. 　 `xy ← xcd × ... + C × ... + ...` ▷ 使连续的 C 个 ID 落在同一 XCD 上
11. `num_rows ← ⌈M / BLOCK_M⌉` ▷ 沿 M 的 tile 行数
12. `num_cols ← ⌈N / BLOCK_N⌉` ▷ 沿 N 的 tile 列数
13. `tiles_per_group ← W × num_cols` ▷ 一个窗口（高度 W）横跨所有列
14. `group_id ← xy / tiles_per_group` ▷ 属于哪一个行窗口
15. `first_row ← group_id × W`
16. `win_h ← min(num_rows − first_row, W)` ▷ 尾部安全的窗口高度
17. `ℓ ← xy mod tiles_per_group` ▷ 窗口内的局部索引
18. `row ← first_row + (ℓ mod win_h)` ▷ 快索引：在列内向下走
19. `col ← ℓ / win_h` ▷ 慢索引：走完 win_h 行后移到下一列
20. **return** (row, col, b.z) ▷ 逻辑 tile 坐标（+ batch）

**HipKittens chiplet swizzle 算法。** 为了让 cache 感知的调度对开发者可用，HipKittens 提供了一个简单且可调的策略，在广泛的 GEMM 问题规模上最大化 cache 复用。算法 1 分两步实现该策略：

1. **XCD 分组。** 把 2D grid 展平成线性序列，并重映射 block ID，使得连续 C 个 ID 常驻同一 XCD。这减少了跨 chiplet 的流量。
2. **层级化窗口遍历。** 我们不是逐行处理 grid，而是以高度为 W 的**竖直窗口**来处理。这相当于把输入的 block ID 空间"折叠"成矩形 tile，从而优化 L2 cache 复用。

两个参数 W 与 C 控制 L2 与 LLC 复用之间的权衡。由于 L2 带宽大约是 LLC 带宽的 3×，应当选择 W 来最大化 L2 命中率。在 AMD MI355X 上，每个 XCD 含 32 个 CU，经验结果表明形状为 8 × 4 或 4 × 8 的 L2 tile 达到最佳硬件利用率。进一步调优 chunk 尺寸 C 可以通过协调跨 XCD 的访问模式（使它们作用于输入矩阵的相近行）来改善 LLC 效率。

---

## 4 实验

本节验证 HK 能够用简单且可复用的、基于 tile 的原语，在广泛的 AI 操作上实现峰值性能内核。

> **图 6：GEMM。** 我们把 HK 的 BF16 与 FP8 GEMM 与最强的可得基线比较。
>
> **图 7：Attention 前向。** 我们把 HipKittens 的 GQA 与 MHA（图 16）与最强的可得基线比较。使用 batch 16、query heads 64、key value heads 8、head dim 64 与 128。
>
> **图 8：Attention 反向。** 我们把 HipKittens 的 GQA 与 MHA（图 15）与最强的可得基线比较。使用 batch 16、query heads 64、key value heads 8、head dim 128。
>
> **图 9：访存瓶颈型。** 我们把 HipKittens 的融合 dropout-residual-layernorm 与 rotary 内核与最强的可得基线比较，batch 16、heads 16、head dim 128。

**基线。** 我们与 PyTorch（compiled 与 SDPA）、AITER [3]、Composable Kernel [4]、ROCm 库 Triton [8]、HipBLASLt [8] 中表现最好的基线内核比较。我们在 MI325（CDNA3）与 MI355（CDNA4）上都做评测。我们通过 Python 绑定在 Python 脚本中基准测试 HK 内核（FP8 除外，因为 AMD 的 PyTorch 支持仍是实验性的）。对每个内核，我们使用 500 次预热运行，并在标准正态分布随机生成的输入张量上，报告 100 次运行的平均 TFLOPs/s 性能。所有内核都在 AMD 最近发布的 beta Docker（ROCm 7.0，`rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta`）中做基准测试。

HK 提供了一套全面的、峰值性能的 AMD AI 内核，全部用可复用的、基于 tile 的抽象写成。我们还在附录 E 提供了代码清单，在附录 C 提供了额外结果：

**1. BF16 与 FP8 GEMM。** HK 与用汇编写的 AMD 基线内核（AITER、HipBLASLt / PyTorch）相竞争。HK 比 Triton 编译器快 1.3–3.0×。此外，我们是用**单一的 8-wave 内核调度**取得这些结果的，它在所评测的各种问题形状上都能泛化。

**2. Attention 前向。** 我们在因果与非因果设定下、head dimension 为 64 与 128 时，评测多头注意力（MHA）与分组查询注意力（GQA）内核。HK 平均优于所有可得的 AMD 基线，包括由 AMD 工程师用手工优化的裸汇编写成的 AITER 内核。在图 7 中，HK 比 AITER 快 1.0–2.1×，比 PyTorch（SDPA）快 1.3–4.5×，比 CK 快 1.0–1.4×，比 Triton 内核快 1.2–4.5×。

HK 的 attention 前向内核使用 8-wave ping-pong。在计算簇内部，每个 wavefront 把在线 softmax 的向量操作（max / subtract / exp2 / accumulate）与 MFMA 指令交织起来。尽管 MI355X 与 NVIDIA B200 之间存在实质性的调度与硬件差异，该内核在可比设定下与 FlashAttention-3 相竞争 [31]。

**3. Attention 反向。** 我们的 GQA 因果与非因果反向 attention 内核在各设定下比基线快 1.8–2.5×（图 8）。我们的 MHA 内核与最强的可得基线（用汇编写成）相竞争（图 15）。Attention 反向是出了名的寄存器密集型工作负载。我们高效的 HK 内核使用了**多种 MFMA 指令形状**（16 × 16 × 32 与 32 × 32 × 16）、**不同的共享内存访问模式**（例如从同一个共享 tile 同时做行布局与列布局的寄存器装载），以及**显式寄存器钉住**。

**4. 访存瓶颈型结果。** 我们在图 9 中考虑一个融合的 dropout-residual-layernorm 内核（来自 prenorm Transformer 架构）与一个旋转位置编码内核。HK 在各设定下比 AITER 与 PyTorch compiled 内核快 1.1–2.2×。AMD 库表现的不一致性、以及汇编工程化内核难以扩展这一点（例如 head dim 64 的 attention 与 GQA 非因果反向进一步印证），反映出拥有一组简单的内核编程抽象来加速 AMD 内核开发的价值。

最后，为验证内核的稳定性，我们用自己的内核在 Slim Pajama 语料上预训练了 Llama 1B [2] 与 BERT 110M [12]，在训练 10B token 后，困惑度与使用 PyTorch 和 AITER 训练的模型相当。

---

## 5 讨论与结论

理想情况下，AI 系统能够利用现代硬件的全部多样性。AMD CDNA4 GPU 提供了业界领先的算力与显存带宽，但"CUDA 护城河"限制了其被采用。虽然 Triton 这类先前系统以多硅平台可移植性为目标，我们的研究表明这些编译器（有时甚至包括 C++ 编译器）常常无法带来 AMD 上的峰值性能。

本工作提供了第一份关于「哪些原则能带来高性能 AMD AI 内核」的系统性分析，并引入了 HipKittens——一组捕捉这些原则的、嵌入 C++ 的精简编程原语。尽管**抽象与前端接口**——tile 以及 tile 之上受 PyTorch 启发的批量操作——在 NVIDIA 与 AMD 之间保持不变，但**这些抽象的实例化**——就调度、数据搬运与 cache 优化而言——由于根本性的硬件差异而不同。我们通过实现一套有代表性的 AI 工作负载来评估 HK 中提出的想法，发现我们能在它们之上都取得峰值性能。通过把 AMD 内核的原则编纂进可组合的、开放的抽象，这些发现让社区更接近那个长期以来的愿景：**一个在多样硬件平台上都表现良好的通用软件栈**。

---

## 6 致谢

我们感谢 Hazy Research Lab 与 Stanford AI Lab 对本工作提出的反馈。我们诚挚感谢以下机构的支持：NIH（No. U54EB020405，Mobilize）；NSF（Nos. CCF2247015（Hardware-Aware）、CCF1763315（Beyond Sparsity）、CCF1563078（Volume to Velocity）、1937301（RTML））；US DEVCOM ARL（Nos. W911NF-23-2-0184（Long-context）、W911NF-21-2-0251（Interactive Human-AI Teaming））；ONR（No. N000142312633（Deep Signal Processing））；Stanford HAI（No. 247183）；NXP、Xilinx、LETI-CEA、Intel、IBM、Microsoft、NEC、Toshiba、TSMC、ARM、Hitachi、BASF、Accenture、Ericsson、Qualcomm、Analog Devices、Google Cloud、Salesforce、Total、HAI-GCP Cloud Credits for Research 项目、Stanford Data Science Initiative（SDSI），以及 Stanford DAWN 项目成员：Meta、Google 与 VMWare。美国政府有权为政府目的复制与分发重印本，无论其上有何版权标注。本材料中表达的任何观点、发现、结论或建议均为作者本人的，不必然反映 NIH、ONR 或美国政府的观点、政策或背书（无论明示或暗示）。

## 7 贡献

WH、DW 与 SA 设计并实现了 HipKittens、内核、微实验与基线。SS 设计了 cache 复用策略。WH 与 SA 撰写了论文。SW、DF、RS、MO、CR 提供了指导，SA 监督了本项目。

---

## 参考文献

> 按学术惯例保留原文。

[1] Helion, 2025. <https://github.com/pytorch/helion>
[2] AI@Meta. Llama 3 model card. 2024.
[3] AMD. AITER, 2025. <https://github.com/ROCm/aiter>
[4] AMD. Composable Kernel, 2025. <https://github.com/ROCm/composable_kernel>
[5] AMD. ROCm hardware features, 2025.
[6] AMD. AMD Matrix Instruction Calculator, 2025. <https://github.com/ROCm/amd_matrix_instruction_calculator>
[7] AMD. AMD Instinct™ MI355X GPUs, 2025.
[8] AMD. ROCm libraries, 2025. <https://github.com/ROCm/rocm-libraries>
[9] Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti. Kevin: Multi-turn RL for generating CUDA kernels, 2025. arXiv:2507.11948
[10] Ganesh Bikshandi, Jay Shah. Developing CUDA kernels for accelerated matrix multiplication on NVIDIA Hopper architecture using the CUTLASS library, 2023.
[11] Tianqi Chen et al. TVM: An automated end-to-end optimizing compiler for deep learning. OSDI 18, pp. 578–594, 2018.
[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. BERT. NAACL-HLT 2019.
[13] NVIDIA Developer Forum. How to understand the bank conflict of shared memory, 2023.
[14] Charles R. Harris et al. Array programming with NumPy. Nature, 2020.
[15] Young Jin Kim, Rawn Henry, Raffy Fahim, Hany Hassan Awadalla. Who says elephants can't run: Bringing large scale MoE models into cloud scale production. SustaiNLP 2022.
[16] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. NIPS 2012.
[17] Robert Tjarko Lange et al. Towards robust agentic CUDA kernel benchmarking, verification, and optimization, 2025. arXiv:2509.14279
[18] Chris Lattner, Vikram Adve. LLVM: A compilation framework for lifelong program analysis & transformation, 2004.
[19] Chris Lattner et al. MLIR: A compiler infrastructure for the end of Moore's law, 2020. arXiv:2002.11054
[20] LLVM Compiler Infrastructure. User Guide for AMDGPU Backend, 2025.
[21] NVIDIA. CUDA templates for linear algebra subroutines (CUTLASS), 2017.
[22] NVIDIA. NVIDIA CuTe, 2024.
[23] NVIDIA. NVIDIA Blackwell architecture technical brief, 2025.
[24] NVIDIA. nvidia-cutlass-dsl, 2025.
[25] OpenAI. AMD and OpenAI announce strategic partnership to deploy 6 gigawatts of AMD GPUs, 2025.
[26] OpenAI et al. gpt-oss-120b & gpt-oss-20b model card, 2025. arXiv:2508.10925
[27] Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini. KernelBench: Can LLMs write efficient GPU kernels? ICML 2025. arXiv:2502.10517
[28] Adam Paszke et al. PyTorch: An imperative style, high-performance deep learning library, 2019. arXiv:1912.01703
[29] Sara Hooker. The Hardware Lottery. CACM, 2021.
[30] SemiAnalysis. MI300X vs H100 vs H200 Benchmark Part 1: Training – CUDA Moat Still Alive, 2024.
[31] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. FlashAttention-3, 2024. arXiv:2407.08608
[32] Benjamin Spector, Jordan Juravsky, Stuart Sul, Owen Dugan, Dylan Lim, Dan Fu, Simran Arora, Christopher Ré. Look Ma, no bubbles! Designing a low-latency megakernel for Llama-1B, 2025.
[33] Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, Christopher Ré. ThunderKittens: Simple, fast, and adorable AI kernels. ICLR 2024.
[34] Philippe Tillet, H. T. Kung, David Cox. Triton: An intermediate language and compiler for tiled neural network computations. MAPL 2019.
[35] Triton. Gluon?, 2025. <https://github.com/triton-lang/triton/issues/7392>
[36] Lei Wang et al. TileLang: A composable tiled programming model for AI systems, 2025. arXiv:2504.17577
[37] Shulai Zhang et al. COMET: Fine-grained computation-communication overlapping for Mixture-of-Experts. MLSys 2025.
[38] Keren Zhou et al. Linear layouts: Robust code generation of efficient tensor computation using 𝔽₂. ASPLOS '26, 2025.

---

# 附录

附录组织如下：

1. 附录 B 包含对相关工作的扩展讨论。
2. 附录 C 提供扩展结果与消融实验。
3. 附录 D 讨论库的实现细节。
4. 附录 E 提供用 HK 写的示例内核清单。
5. 附录 F 提供 AMD 上 FP6 内核的案例研究。

## A 术语

尽管 NVIDIA 与 AMD GPU 大体上都遵循相似的 SIMT（单指令多线程）执行模型，它们的软硬件栈使用了不同的术语。下表总结了关键对应关系。

| 概念 | NVIDIA 术语 | AMD 术语 |
|---|---|---|
| 执行单元 | Warp（32 线程） | Wave（64 线程） |
| 线程块 | Thread Block（CTA） | Workgroup |
| 处理器 | Streaming Multiprocessor | Compute Unit |
| 共享内存 | Shared Memory（SMEM） | Local Data Share（LDS） |
| 寄存器 | Registers | 累加器 / 向量寄存器（AGPR / VGPR） |
| 全局内存 | HBM | HBM |
| Cache | L2（GPU 全局） | L2 Cache（chiplet 范围）+ LLC（GPU 全局） |
| 矩阵计算 | Tensor core | Matrix core |
| 矩阵指令 | WGMMA / WMMA / TCGEN05 | MFMA |
| 异步访存操作 | TMA | Buffer load to LDS |
| 编译器 / 工具链 | CUDA、NVCC | HIP、HIPCC |

## B 扩展相关工作

### B.1 面向 AI 内核的库与框架

**高性能 AMD 内核库。** AMD 提供了 AITER [3]，一个高性能内核库，但其最快的实现是用裸汇编写的。这些内核虽然有效，却没有暴露可复用的抽象，且难以跨工作负载扩展。CUDA 与 HIP 在单个线程的层级上暴露内核，而 ML 工作负载是由更大的、可复用的计算模式构成的，这些模式能从更粗的抽象中获益。已有若干库与编译器试图弥合这一鸿沟。Composable Kernel（与 CUTLASS 类似）使用深度嵌套的 C++ 模板，造成了使其难以使用与扩展的复杂度 [4, 21]。

**编译器。** TVM、Triton [1, 11, 34] 这类基于编译器的系统，以更高层的类 Python DSL 面向更广泛的 ML 受众。虽然对研究者更友好，但这些框架都牺牲了对寄存器与同步的细粒度控制，且在支持新硬件特性上一直较慢（见 B.2 节）。结果是，在 AMD 上开发者常常在 Triton 内核里诉诸内联汇编来找回性能 [15]。即便在 NVIDIA 上——那里对编译器系统的投入相对更多——基于 C++ 的 DSL 也能提供高达 10× 的性能提升 [33]。

还有一些更晚近的编译器：

1. **TileLang 与 Mojo** 可以经 LLVM IR 编译到 AMD，但它们缺少针对 AMD 架构约束的抽象（例如寄存器压力下的灵活 tile 尺寸设定、thread block 调度、cache 感知的 grid 排序）。它们在 AMD 上的评测很有限：TileLang 只报告了在 AMD MI300X GPU 上一个 257 TFLOPs 的 attention 内核。TileLang 还依赖对 CUTLASS / CK 的后端调用，而 Mojo 依赖编译器提示而非可复用的抽象。Mojo 面向 MI300X 的 attention 内核也表现出 bank conflict。至今两个框架都没有系统性地支持 AMD。
2. **Linear Layouts** 把 MMA / WGMMA / MFMA 布局形式化为线性映射，并在 Triton 后端实现了它们之间自动、优化的转换（通过 warp shuffle 与 swizzle 过的共享内存），有实测加速。然而，它没有为 thread block 或 grid 级调度定义抽象，其评测也没有展示混用 tensor core 形状的内核，也没有讨论访存指令的不同相位排序（如 3.2 节所述）。

**面向峰值性能的编程框架。** 近来涌现了一批为 AI 内核提出嵌入 C++ 的、基于 tile 的原语的 DSL，包括 ThunderKittens（TK）[33] 及其后继者（TileLang [36]、CuTe [22]、Linear Layouts [38]）。这些做法表明，小而有主见的抽象集合可以同时带来简洁性与高性能。然而，它们都没有提供一套全面的 AMD 内核抽象：ThunderKittens 与 CuTe 只支持并只在 NVIDIA 硬件上验证，其内核模板（如生产者–消费者调度）与 NVIDIA 特有的特性绑定。相比之下，我们的工作识别出了一组精简且有原则的抽象，足以在 warp、block 与 grid 三个层级上支撑高性能的 AMD 内核。我们通过在多样工作负载上实现端到端内核——包括需要混用 tensor core 形状的 attention 反向——来证明其充分性。

### B.2 AMD 软件生态是脆弱的

许多 AMD 库是 NVIDIA 库的 fork。考虑到硬件差异，这有导致次优性能的风险。我们还发现，尽管对 AMD 软件已有投入，当前生态仍是脆弱的，这正是 HK 的动机：

- **PyTorch 内核**：内置的 scaled dot product attention（SDPA）后端，在 AMD MI355X GPU 上跑 Llama GQA 反向只达到 259 TFLOPS（截至 2025 年 10 月、ROCm 7.0.0）。Attention 是现代 AI 工作负载中的主力操作，这一性能差距凸显了后端成熟度的不足。
- **汇编内核**：AITER [3] 包含高性能内核，但其最快的实现直接用裸汇编写成。这种方式难以扩展到 AI 工作负载的广度；我们可以从 AMD MI355X 上 GQA 反向缺乏成熟内核支持看出这一点——在序列长度 8192 时，AITER 在因果与非因果设定下分别只达到 272 / 384 TFLOPS。
- **Triton 内核**：AMD 上的 Triton 编译器在寄存器生命周期跟踪、以及把访存下降到最高性能的 intrinsic 上都很吃力。例如，它可能无法回收寄存器，或无法下降向量化装载。Torch 编译的内核在访存瓶颈型工作负载上可以给出有竞争力的性能（图 9），但这些优化是黑箱的，可能错过最优 intrinsic。例如，在类 Llama 维度上编译出的 LayerNorm 内核，其 L2 命中率比我们的 HK 内核低 23%。新 CDNA / PTX 特性到集成进编译器之间的时间也很慢；例如截至 2025 年 9 月，buffer load 在 AMD 上仍不是 Triton 装载/存储的默认选项。这样的生态正是 HK 的动机——HK 旨在帮助简化并加速高性能 AMD 内核开发。

## C 扩展分析

本节提供我们实验设置的细节与补充结果。

**设置细节。** AMD 在 <https://hub.docker.com/u/rocm> 提供了多个 docker 容器。我们使用 AMD 近期提供的 Docker 容器来基准测试内核：在 MI350X / MI355X 上使用 `docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta`，在 MI300X / MI325X 上使用 `docker.io/rocm/pytorch`。启动容器的示例命令如下：

```bash
podman run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  -v $(pwd):/workdir/ \
  -e USE_FASTSAFETENSOR=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
  bash
```

**基线。** 对 AITER 基线，我们使用图 10 的方式。如果 Docker 里没有自带 AITER，我们从源码安装。对 Composable Kernel 基线，我们使用图 12 中指明的安装流程与内核。对 PyTorch 基线，我们使用图 11。对 HipBLASLt 基线，我们在 AMD 提供的 Docker 内使用图 13 的命令。

```python
# Attention
out_aiter, softmax_lse = aiter.flash_attn_func(
    Q_aiter, K_aiter, V_aiter, causal=causal,
    return_lse=True, deterministic=False)
out_aiter.backward(dO_aiter)

# GEMM
from aiter.tuned_gemm import tgemm
C_aiter = tgemm.mm(A, B, None, None, None)
```
> **图 10：AITER 基准测试。**

```python
out_pt = torch.nn.functional.scaled_dot_product_attention(
    q_pt, k_pt, v_pt, attn_mask=None, dropout_p=0.0, is_causal=causal)
out_pt.backward(dO_bhnd)
```
> **图 11：PyTorch 基准测试。**

```bash
# Build
git clone https://github.com/rocm/composable_kernel
cd composable_kernel
mkdir build && cd build
../script/cmake-ck-dev.sh .. gfx950 -G Ninja
ninja tile_example_gemm_basic
ninja tile_example_fmha_fwd
ninja tile_example_fmha_bwd

# Run
./bin/tile_example_gemm_basic -prec=fp8 -m=1024 -n=1024 -k=1024 -warmup=500 -repeat=100 -v=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
```
> **图 12：CK 基准测试。** 对 GEMM 的每个维度，我们报告 `tile_example_streamk_gemm_basic`、`tile_example_gemm_basic`、`tile_example_gemm_universal` 三者中的最好性能。

```bash
# bf16
hipblaslt-bench --batch_count 1 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
  --rotating 512 --iters 100 --cold_iters 500 -m 1024 -n 1024 -k 1024

# fp8
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 \
  --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 \
  --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r \
  --bias_type f32_r --compute_type f32_r --rotating 4 --iters 100 --cold_iters 500 \
  -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --ldd 8192 --initialization norm_dist

# fp6（需从源码安装后）
./clients/hipblaslt-bench --api_method c -m 1024 -n 1024 -k 1024 --alpha 1 --beta 0 \
  --transA T --transB N --batch_count 1 --scaleA 3 --scaleB 3 \
  --a_type f6_r --b_type f6_r --c_type f16_r --d_type f16_r \
  --compute_type f32_r --rotating 0 --cold_iters 500 --iters 100
```
> **图 13：HipBLASLt 基准测试。**

### C.1 HipKittens 内核

我们收录第 4 节余下的内核图表。图 14 收录 MI325X 与 MI350X GPU 上的 BF16 GEMM 结果。图 16 收录 MI355X GPU 上 MHA 前向的结果，图 15 收录反向的结果。

> **图 14：BF16 GEMM。** 我们在 MI325X 与 MI350X 上把 HipKittens 与最强的可得基线比较。这些内核使用 M = N = K 的维度。
>
> **图 15：Attention 反向。** MI355X 上因果与非因果 attention 的结果。使用 batch size 16、heads 16、head dim 128。
>
> **图 16：Attention 前向。** MI355X 上因果与非因果 attention 的结果。使用 batch size 16、heads 16、head dim 128。
>
> **图 17：Attention 前向。** MI355X 上因果与非因果 attention 的结果。使用 batch size 16、heads 16、head dim 64。

### C.2 Grid 调度

在 3.4 节，我们用表 4 讨论了优化 L2 与 LLC 复用的 chiplet swizzle 策略。图 18 提供了 14592 维度 GEMM 的对应 grid 顺序可视化。

> **图 18：** 对一个 M=N=K=14592 BF16 GEMM 输出 D 矩阵的三种不同 grid 调度的可视化。颜色代表 XCD 归属。高亮的是在设备（256 个 CU）上被调度的第一个时间步的 thread block。调度 18a 按 block ID 分配 block 到 grid。调度 18b 与 18c 应用算法 1，使用不同的 chunk 尺寸和相同的窗口尺寸。表 4 展示了每种调度的性能。这个 GEMM 设定对这些优化格外敏感，因为默认调度导致最坏情形的 L2 复用，而巨大的内存足迹又让 LLC 复用变得更加重要。

### C.3 ThunderKittens 性能

我们在取自 N(0, 1) 的输入上基准测试 TK [33] 与 CuBLASLt 内核，使用 500 次预热迭代与 100 次测量迭代，与我们在 AMD GPU 上的协议保持一致。结果如图 19 所示[^8]。我们纳入这一部分，是为了凸显 TK 的理念（本工作正是对其的延伸）如何带来高性能的 NVIDIA 内核。

[^8]: 我们在 NVIDIA H100 与 NVIDIA B200 GPU 上都做了评测。我们总体观察到 NVIDIA 内核性能随迭代次数增加而下降，并注意到 Spector 等人 [33] 报告使用了更少的迭代次数。

> **图 19：ThunderKittens 性能。** 我们把 TK 与 NVIDIA CuBLASLt 的 BF16 GEMM 性能比较，发现 TK 内核提供了有竞争力的性能，尽管这些 TK 内核是大约 8–12 个月前发布的。

## D 库实现细节

### D.1 共享内存与寄存器内存

**寄存器布局。** 一个寄存器 tile 的布局，决定了线程是以行主序还是列主序持有元素[^9]。由于线程在 MFMA 指令的归约维度上持有连续元素，寄存器布局决定了我们是在内存的行上归约还是列上归约。当从共享内存装载数据到寄存器时，对 BF16 我们通常使用两类不同的指令：

[^9]: <https://github.com/ROCm/amd_matrix_instruction_calculator>。这是了解寄存器 tile 布局与形状的有用资源。

- **行布局（`ds_read_b128`）**：`ds_read*` 指令接受 3 个参数：1）作为数据目的地的向量寄存器；2）一个共享内存地址；3）相对该共享内存地址的常量偏移。`ds_read_b128` 指令分 **4 个相位**执行，每个相位由 64 个线程中的一个子集执行（表 5）。因此，只要确保同属一个相位的任意两个线程不访问同一个共享内存 bank，就能消除该指令的 bank conflict。每个相位有 16 个线程参与，每个线程访问 128 bit（即 4 个 bank），这样全部 64 个 bank 都会被读到，我们就最大化了共享内存吞吐。
- **列布局（`ds_read_b64_tr_b16`）**：通常，以列主序格式读取数据需要为访问的每一行分别发射多次装载。使用 `ds_read_b64_tr_b16` 指令，我们能让线程以更大的粒度访问共享内存，从而高效得多地完成这些列主序装载。以图 20 中的 16×32 寄存器 tile 为例。在一个 16×32 列布局寄存器 tile 中，每个线程在归约维度上持有 8 个连续元素（即 stride 8），线程 0 持有第 0–7 行的首个元素（两张表中也做了阴影标记）。`ds_read_b64_tr_b16` 指令完成这一装载的方式是：让不同线程去读那些将被放进**另一个线程**寄存器 lane 中的数据。例如，线程 4 在技术上读的是第二行的首个元素，但它不把该值放进自己的寄存器 lane，而是放进线程 0 的寄存器。该指令分**两个相位**执行：前 32 个线程在第一相位读，其余的在第二相位读。如果这个 SMEM tile 只需要支持来自列主序 16×32 寄存器 tile 的读取，那么不做 swizzle 的模式就足以消除 bank conflict。然而，如图 4 所示，要支持来自**行主序** 16×32 寄存器 tile 的读取，就必须做 swizzle。

> **图 20：** 读取一个 16×32 列布局寄存器 tile 时 `ds_read_b64_tr_b16` 的共享内存访问模式。每个格子代表一个 16 bit 值，数字代表线程。不同颜色代表不同的共享内存 bank（注意一个 bank 跨两个格子）。

**共享内存与寄存器 tile 形状。** 上一小节聚焦于从 16×32 共享内存 tile 形状装载到 16×32 寄存器 tile，但不同工作负载可能需要其他共享内存与寄存器 tile 形状，映射到不同的 MFMA 指令。只要两者之一是另一者的倍数，HK 就支持共享内存与寄存器 tile 形状之间的装载与存储。例如，从 16×32 共享内存 tile 装载到 32×16 寄存器 tile 是**不支持**的，但从 16×16 共享内存 tile 装载到 32×16 寄存器 tile 是**允许**的。每种共享内存 tile 形状还配有一个默认 swizzle 模式，作为消除常见访问模式下 bank conflict 的尽力而为的尝试。

**单一 swizzle 是不可能的。** 为说明为何在 AMD GPU 上单一 swizzle 模式不足以覆盖不同的寄存器 tile 形状与布局，考虑 attention 反向中出现的以下两种访问模式：

1. 一个行布局 16×16 BF16 tile 被写入共享内存。对这一 tile 配置，每个线程持有 4 个连续的 BF16 值——内存中 64 bit——发射该写入的最优指令是 `ds_write_b64`。要避免此访问的 bank conflict，需要一个尊重表 5 所列相位排序与 bank 行为的 swizzle 模式。在这种情况下，一个满足这些约束的 swizzle 是 `offset ^= ((offset % 512) >> 7) << 3`，即用 XOR swizzle 把 64-bit 的内存块在内存中挪位。
2. 一个行布局 16×32 BF16 tile 被从共享内存读出。对这一 tile，每个线程持有 8 个连续的 BF16 值——内存中 128 bit——发射该读取的最优指令是 `ds_read_b128`。

无论 `ds_read_b128` 需要何种 swizzle 模式，这两条指令的粒度彼此冲突。`ds_read_b128` 要求共享内存中至少 128 bit 是连续的，而 `ds_write_b64` 的 swizzle 模式把内存打散成 64-bit 的块。结果就是，两者必须使用不同的 swizzle 模式。

### D.2 相位与 Bank

由于每条指令的相位与 bank 行为都没有良好的文档，我们为二者各写了一个简单的求解器。**相位求解器**遍历一个 wave 内的每一对线程，让它们对同一个 bank 执行共享内存指令；如果发生 bank conflict，则这两个线程属于同一相位。**bank 求解器**取同属一个相位的两个线程，把其中一个固定为访问 bank 0，用另一个去访问其他 bank；从 bank 0 到首次出现 bank conflict 的那个 bank 之间的 bank 数，就代表该共享内存指令可访问的 bank 数量。

| 指令 | Bank 数 | 相位 | 活跃线程 |
|---|---|---|---|
| `ds_read_b128` | 64 | 0 | 0-3, 12-15, 20-27 |
| | | 1 | 4-11, 16-19, 28-31 |
| | | 2 | 32-35, 44-47, 52-59 |
| | | 3 | 36-43, 48-51, 60-63 |
| `ds_read_b96` | 32 | 0 | 0-3, 20-23 |
| | | 1 | 4-7, 16-19 |
| | | 2 | 8-11, 28-31 |
| | | 3 | 12-15, 24-27 |
| | | 4 | 32-35, 52-55 |
| | | 5 | 36-39, 48-51 |
| | | 6 | 40-43, 60-63 |
| | | 7 | 44-47, 56-59 |
| `ds_write_b64` | 32 | 0 | 0-15 |
| | | 1 | 16-31 |
| | | 2 | 32-47 |
| | | 3 | 48-63 |
| `ds_read_b64` | 64 | 0 | 0-31 |
| | | 1 | 32-63 |

> **表 5：相位–bank 表。** 每条共享内存指令可用的 bank 数量，以及每条指令所需的相位数（及每相位参与的线程）。

### D.3 钉住的寄存器 tile

HK 通过**寄存器区间**（register range）的概念，让开发者控制分配给不同寄存器 tile 的寄存器。例如：

```cpp
using Q_ranges = split_many_t<type_list<range<24, 39>>, 4>;
```

这定义了一个寄存器区间列表，其中每个区间恰好包含 4 个寄存器。这里的寄存器区间是 `v[24:27]`、`v[28:31]`、`v[32:35]` 与 `v[36:39]`。每个寄存器区间对应持有一个寄存器 tile 中单个 base tile 所需的寄存器；我们在定义寄存器 tile 时指定一个寄存器区间列表，像这样：

```cpp
rt<bf16, 16, 128, row_l, rt_16x32_s, Q_ranges> Q_i;
```

开发者可以调用 HK 中相同的函数，但现在这些函数会作用在指定的寄存器上。如 3.2 节所述，这让我们在编写 attention 反向内核时，能够把 AGPR 钉为 MFMA 指令的 A 或 B 矩阵输入。

### D.4 编译器提示

LLVM 编译器接受开发者提供的提示，以引导 AMD GPU 上的指令调度[^10]。我们在内核中使用了其中一些提示，以增强我们在 HIP 层面所施加的调度。我们发现有三组 intrinsic 有用。

[^10]: <https://llvm.org/docs/AMDGPUUsage.html>

1. `llvm.amdgcn.sched.barrier` intrinsic 接受一个掩码，告诉编译器在编译出的调度中哪些类型的指令可以越过该 intrinsic。文档中描述了针对所有指令、VALU（向量 ALU）指令、SALU（标量 ALU）指令、VMEM（全局内存）指令、MFMA（矩阵）指令等等的掩码。这个 intrinsic 用来在我们的指令簇之间建立**硬边界**。例如，见附录 E 内核清单中的 `__builtin_amdgcn_sched_barrier(0)`。
2. `llvm.amdgcn.sched.group.barrier` intrinsic 用于建立**调度流水线**。开发者考虑一组指令，并向编译器精确指定如何排序它们。调用该 builtin 时接受：一个指定指令类型的掩码、一个表示该调用适用于多少条此类指令的 size，以及一个作为标识符的 sync id。这个 builtin 创建了一个由若干"指令组"构成的"超组"。sync id 标识该超组；具有相同 sync id 的指令组之间会强制顺序，且指令组只相对于具有相同 sync id 的其他组被调度。掩码是位掩码，常用的有：

```c
#define MFMA_MASK 0x08
#define VMEM_MASK 0x20
#define DS_MASK   0x100
```

该 intrinsic 的每次调用都会**向后回看**，找到最近的、尚未属于此前 `__builtin_amdgcn_sched_group_barrier` 所创建的组的、对应类型的指令。例如：

```c
__builtin_amdgcn_sched_group_barrier(VMEM_MASK, 4, 0);
__builtin_amdgcn_sched_group_barrier(MFMA_MASK, 4, 0);
__builtin_amdgcn_sched_group_barrier(DS_MASK,   8, 0);
__builtin_amdgcn_sched_group_barrier(MFMA_MASK, 4, 0);
```

这会找到最后 4 条全局内存（VMEM）装载并把它们排在最前，然后找到最后 4 条矩阵（MFMA）指令排在全局内存装载之后，再找到最后 8 条共享内存到寄存器（DS READ）的装载排在那 4 条 MFMA 之后，最后找到位于前述最后 4 条之前的另外 4 条 MFMA 排在最末。

3. `__builtin_amdgcn_s_setprio` intrinsic 让我们为某个 wave 指定相对于其他争抢硬件资源的 wave 的优先级（0–3）。我们在 8-wave ping-pong 调度的计算簇周围使用它，如我们的 GEMM 与 attention 前向内核所示。

用这些提示做调度的局限在于：任何包在 `asm volatile` 里的代码对编译器都是黑箱，而且对某些指令（例如把 BF16 转 FP32 的 `v_cvt_pk_bf16_f32`）缺少 LLVM builtin。值得一提的是，当前（截至 2025 年 10 月）Modular AI 的 GEMM 内核依赖编译器提示（`sched_group_barrier`）。这条路线可行，因为 Modular 也在替换编译器本身；但它要求开发者去思考**每一条**指令的发射，而不是提供一个在批量 tile 原语层面思考的选项。我们的观点是：在**簇的作用域**上使用调度提示、并用 tile 原语来构成顶层的内核调度（如我们的 attention 前向内核那样），可能有助于简化可编程性并保持性能。

### D.5 同步

**装载。** 与异步的 tensor memory acceleration（TMA）类似，AMD CDNA3 与 CDNA4 GPU 有从全局内存直达 LDS（共享内存）的装载指令，称为 `buffer_load_dword`。这些指令可以装载一个 dword（4 字节，一个 bank）、三个（`dwordx3`，12 字节）或四个（`dwordx4`，16 字节）。这些指令跳过寄存器文件，并接受常量偏移，这也有助于缓解地址计算开销。一旦装载被发射，`vmcnt(x)` 指令表示等到只剩 x 条全局内存装载指令在飞行中，而 `vmcnt(0)` 表示等待所有未完成的装载。理想情况下，我们可以把装载发射与这些等待之间的距离拉开（如我们的 GEMM 与 attention 内核所示，见附录 E）。类似地，还有从共享内存到寄存器的异步装载 `ds_read_b32`（或 8 字节的 `b64`、12 字节的 `b96`、16 字节的 `b128`）。`lgkmcnt(x)` 指令表示等到只剩 x 条共享到寄存器的指令在飞行中，`lgkmcnt(0)` 表示等待所有未完成的共享到寄存器装载。

**执行。** `__builtin_amdgcn_s_barrier()` 在功能上等同于 `syncthreads`。注意 AMD 是 SIMD 模型而 NVIDIA 遵循 SIMT，所以在 AMD 上我们不需要在 warp 内同步线程。因此，AMD 上没有 `syncwarp` 的等价物。

## E HipKittens 内核清单

> 本节展示 HipKittens 的内核示例并讨论我们内核实现的算法细节。完整代码清单（图 21–23，约 390 行）见[原文 PDF](https://arxiv.org/abs/2511.08083) 与[开源仓库](https://github.com/HazyResearch/HipKittens)；此处翻译各清单的说明文字。

### E.1 矩阵乘

BF16 GEMM 内核（E.3 节）把问题分解为每个 thread block 计算一个 256 × 256 的输出 tile（以 `BLOCK_SIZE` 表示）。在**序幕**（prologue）中，内核把 A 与 B 输入矩阵从全局内存预装载到共享内存。内核插入一个**条件屏障**，让一半的 wave（每个 SIMD 一个）停下来，而另一半开始执行额外的装载。当这个**领跑 wavegroup** 完成其额外装载后，它通过 `s_barrier` 调用解除**跟随 wavegroup** 的阻塞。此后，两个 wavegroup 在热循环中所示的计算簇与访存簇之间交替，每个簇的末尾总是由一个 `s_barrier` 标定。这就是 3.3 节引入的 8-wave ping-pong 内核调度。

对内核的 MI325X 版本，我们保持相同的 8-wave 结构，但该硬件只有 65 KB 共享内存，因此我们无法在共享内存中做双缓冲。取而代之，我们**用寄存器文件做双缓冲**：不使用 HBM 到 LDS 的直接 buffer load，而是从 HBM 装载到一个寄存器缓冲区，与此同时各 wave 在此前已存下的寄存器 tile 上计算 MFMA。计算完成后，寄存器缓冲区中的数据用 `ds_write` 存到共享内存。

> **图 21：** HK BF16 GEMM，在 CDNA4 上与 AITER 有竞争力。

### E.2 融合 Dropout + Residual + Layernorm

一个非常简单的 HipKittens 内核，每个 thread block 沿序列维度处理一批向量。这份内核清单展示了 HK 的算子与向量，它们与 PyTorch 中的相似（例如 `sum`、`add`、`mul`、`div`）。

> **图 22：** 融合 Dropout + Residual + Layernorm 内核，性能优于 `torch.compile`。

### E.3 Attention

HipKittens 的 attention 内核使用 8-wave ping-pong 调度。每个 wave 为单个 head 与 batch 计算输出的一个 32 × 128 tile。在序幕中，全部八个 wave 首先协作装载 key 与 value 的 tile，以及各自私有的 query tile。线程执行初始的 query-key 矩阵乘与 softmax 的前半部分。然后内核用一个条件屏障让一半的 wave（每个 SIMD 一个）停下。领跑 wavegroup 先行推进，装载下一批 key 与 value 的 tile，完成后解锁跟随 wavegroup。在热循环中，两个 wavegroup 在**计算簇**（各自涉及矩阵乘与向量操作）与**访存簇**（涉及全局到共享内存的搬运）之间交替。

在计算簇内部，编译器会交织向量操作与矩阵操作。我们也可以使用 `sched_barrier` 提示，引导 LLVM 编译器按自定义模式交织。

> **图 23：** HipKittens 非因果 attention 前向内核，与 AMD AITER 库提供的汇编内核相竞争。

## F 案例研究：FP6 GEMM 的初步发现

我们讨论在 AMD MI350X 与 MI355X GPU 上对 FP6 硬件行为的初步经验观察。目标是刻画 FP6 的数据搬运与 matrix core 操作在实践中如何表现。撰写本文时 AMD 自家的 CK 库基线尚未优化，我们的结果应被视为初步的。FP6 令人兴奋，是因为 AMD matrix core 在 FP6 上达到 NVIDIA 设备两倍的峰值 FLOPS。然而在实践中，我们在把 FP6 值装入/写出内存时遇到了挑战，这影响了我们达到高利用率的能力。

**内存装载。** 我们的 FP6 GEMM 内核在每个访存簇中用 4 个 wave 协作地从全局内存装载一个 128 × 128 的 tile。以 FP6 数据类型，该 tile 是 128 × 128 × 6/8 = 12,288 字节，即每线程 48 字节。我们考虑以下 CDNA4 指令作为从全局内存装载的选项：

- **`buffer_load_dwordx4`** 使每线程发射的指令数最少（每 tile 3 条）。然而，朴素地用这条指令装载到共享内存、再用 `ds_read_b128` 后接 `ds_read_b64` 从共享内存把每个线程拥有的 24 个连续字节读进寄存器，会造成共享内存对齐问题，因为 `ds_read_b128` 必须 16 字节对齐才有最大性能。例如，tile 每行的第二个线程会在行内偏移 24 字节处执行 `ds_read_b128`。在这种配置下我们的内核变成共享内存瓶颈。一个解法是让每隔一个的线程（`laneid % 2 == 1`）**对调**这两条共享内存装载指令，使 `ds_read_b64` 装载该线程数据的前 8 字节、`ds_read_b128` 装载接下来的 16 字节。我们发现，装载到一个**依赖于线程 ID 的寄存器目的地**需要动用缓慢的 scratch 存储。于是，对那些对调了 `ds_read_*` 指令的线程，我们改为无论哪个线程都继续把 `ds_read_b128` 读进前四个寄存器、把 `ds_read_b64` 读进后两个寄存器。这样一来就需要一次 **shuffle**，即根据线程 ID 有条件地交换寄存器中的这两个区域。这会"打断 wave"，除了搬运内存所需的 VALU 指令外还产生两条跳转指令。我们发现 shuffle 带来的跳转 + VALU 占了内核热循环 **49% 的周期**，导致内核只达到 2430 TFLOPs。

  另一种做法是用两条 `ds_read_b96`。这会造成 `buffer_load_dwordx4` 从全局到共享读取的 16 字节块，与 `ds_read_b96` 读取的 12 字节块之间的**错位**。结果我们无法对共享内存中的数据做 swizzle，导致 4 路 bank conflict。鉴于 `buffer_load_dwordx4` 的这些问题，我们转而考察从全局到共享装载 FP6 值的其他选项。

- **`buffer_load_dwordx3`** 对 FP6 很有吸引力，因为它让一个 warp 恰好装载一个 8 × 128 的 tile，并且支持 swizzle。遗憾的是，对 FP6 而言，这条指令在指令发射数上并不优于 FP8：每个线程发射 4 条指令来装载 128 × 128 tile，与我们 FP8 GEMM 内核所用的条数相同。该指令还以 16 字节的 stride 装载每线程的 12 字节数据，**浪费了 25%** 的结果共享内存 tile，并使 `ds_read_b96` 可用的 32 个 bank 中有 8 个闲置（表 5）。尽管有这些缺点，我们发现在我们的 FP6 GEMM 用例中，这很可能是最有说服力的全局到共享装载指令。对用 `buffer_load_dwordx3` 装到共享内存的数据来说，`ds_read_b96` 是自然的 LDS 到寄存器装载指令。我们发现 `ds_read_b96` 在 16 字节对齐的共享内存地址上工作良好。

- **`buffer_load_dwordx1`** 避免了共享内存浪费、对齐问题与 swizzle 限制，但内核会被发射的指令数量卡住瓶颈，导致比用 `buffer_load_dwordx3` 构建的内核更慢。

**Matrix core 操作。** 在 BF16 与 FP8 GEMM 中，我们发现最快的 MFMA 指令是给定 dtype 下可用的**较小**那条（图 3），在本例中即 `v_mfma_f32_16x16x128_f8f6f4`。在这条指令中，每个线程拥有每个 FP6 操作数矩阵的 32 个连续元素，即 24 个连续字节。在我们的内核中，每个线程发射两条 `ds_read_b96` 指令：一条读前 12 字节，一条读后 12 字节。`ds_read_b96` 把寄存器文件中的目的基址约束为 16 字节对齐的地址，因此为了让这 24 字节数据连续，我们必须发射三条 `v_mov_b32_e32` 指令，把第二条 `v_mov_b32_e32` 写入的三个寄存器各下移一个寄存器。注意这次 shuffle 不像用 `buffer_load_dwordx4` 配 `ds_read_b128` 与 `ds_read_b64` 时那次 shuffle 那么昂贵，因为搬运的数据更少、且不会因此打断 wave。

我们注意到 HIPCC 对这一 shuffle 需求处理得不好：在 16384×16384×16384 规模下，该内核会向 scratch 内存**溢出 54 个寄存器**，导致内核又慢又不正确。为补救这一点，我们在显式调度寄存器的前提下重写了 FP6 GEMM，从而能明智地设定这次 shuffle 所用的寄存器，**完全消除了寄存器溢出**。这条路并非没有隐患。我们需要顾及 `v_mov_b32_e32` 的指令延迟，因此手动确保在 `v_mov_b32_e32` 指令与依赖它的 MFMA 指令之间至少隔 8 个周期的指令。某些情况下这涉及手动插入 `v_nop` 指令。

> **图 24：FP6 GEMM。** 我们在方阵形状 8192 与 16384 上，比较 AMD CK 库、B200 上 NVIDIA CUTLASS 与 HipKittens 的 FP6 GEMM 性能。使用 500 次预热迭代与 100 次测量迭代。

**结果。** FP6 matrix core 速度是 MI350X 与 MI355X GPU 的一项亮眼特性。我们的 FP6 GEMM 内核优于 AMD 的 CK 实现，并取得了与我们自己的 FP8 GEMM 相当的性能。这些结果反映的是我们对 FP6 硬件行为的初步观察；我们预期在未来工作中会有更多改进。
