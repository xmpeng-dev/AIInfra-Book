# HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs

> [arXiv 2605.23764](https://arxiv.org/abs/2605.23764) (v2, 2026-06-01, cs.DC) · 中科大 + 华为 + 北大 · CC BY-NC-ND 4.0
> 代码：<https://gitcode.com/mindspore/hyper-parallel/tree/master/hyper_parallel/core/multicore>
> 硬件：Ascend A3（每 AI Core = 1 AIC + 2 AIV；A3 单卡 25 AIC + 50 AIV + **192 MB L2**）
> 软件栈：MindSpore + MindFormers · 模型：DeepSeek-V3 式 MoE（hidden 7168 / inter 2048 / top-k=8 / bf16）
> 规模：profiling 用 256 卡（dp32 pp8 tp2 ep32），评测用 64 卡（DP32/TP2/EP{4,8,16}）

## TL;DR

**这是 ROCmoe 论点在昇腾上的完整实现，而且带着我们缺的那组数字。**

它把 MoE-FFN 从「算子逐 kernel 串行」改造成**编译期静态调度的 tile 级异构 taskflow**，三件事：AIV 驱动的单边通信（消掉 host 侧集合同步）、依赖保持的 tile 任务生成、事件驱动的静态调度；运行时在**一次 kernel launch** 内并发驱动 AIC/AIV worker。

Dispatch-to-Combine 模块延迟 **1.49–1.58×**（EP4/8/16，均衡路由），端到端训练步 **1.08–1.09×**（自然路由）。

**最该记住的两个数**：
1. **静态 vs 动态调度的每任务派发成本 = 0.1 µs vs 2.36 µs（23×）** ——这是 ROCmoe P4「编译期静态调度」目前唯一缺的量化论据，现在有了。
2. **模块级 1.5× → 端到端 1.08×** ——提醒我们自己报 MonolithEP 的「≥1.5× vs Megatron」时必须讲清测的是哪个边界。

**一个必须诚实面对的架构差异**：昇腾的 AIC/AIV 是**物理解耦、各有独立硬件队列、有跨队列事件原语**的。CDNA 没有这个——MFMA 和 vector ALU 在同一个 CU 里共享 wave slot。我们拿不到 AIC/AIV 式的异构重叠，只能用跨 CU 的 WG 角色分区去近似，粒度更粗、且要付出 CU 数量的代价（详见 §5.2）。

## 1. Problem

### 1.1 昇腾的异构执行模型（我们没有的那部分）

| 特性 | Ascend A3 | CDNA3/4 对照 |
|---|---|---|
| 计算单元 | 每 AI Core = 1 AIC（矩阵）+ 2 AIV（向量/搬运/通信），**物理独立** | CU 内 MFMA 与 VALU 共享 wave slot，不解耦 |
| 指令队列 | **CTQ**（Cube Task Queue）+ **VTQ**（Vector Task Queue），各自独立取任务 | 无对应物 |
| 跨队列同步 | `CrossCoreSetFlag` / `CrossCoreWaitFlag` 显式事件原语 | 只能用 device-scope atomic + 轮询 |
| 单卡规模 | 25 AIC + 50 AIV | 256 CU（MI355X） |
| 共享缓存 | **192 MB L2**，AIC/AIV 共享，读带宽 >4× HBM | 每 XCD 4 MB L2 + 256 MB Infinity Cache |

这套「显式队列 + 显式事件」的编程接口，让开发者能在 tile 粒度上直接指定哪类单元处理哪段计算——**这是他们能做静态异构调度的硬件前提**。

### 1.2 profiling 出来的四个数（256 卡 DSv3-671B）

| 观察 | 数值 |
|---|---|
| FFN 里的 GroupedGEMM 占 step 时间 | **25%**（最大单项） |
| AIC 的 MAC 利用率 | 仅 **~67%** |
| 关键路径上的 vector 算子占端到端 | **~18%** |
| EP 通信在关键路径上的开销 | **17%**，其中只有 **61%** 被 Cube 计算盖住，**39% 裸露** |
| DP / TP / CP 通信 | 已可忽略 |

根因是**调度模式与硬件架构不匹配**：Megatron 式和 MindSpore baseline 都能做粗粒度的计算-通信重叠，但仍然把 MoE 算子当整卡 kernel 发射，**没有把 AIC/AIV 队列暴露成独立的 tile 级调度目标**。结果是 AIC 主导的 kernel 必须跑完，AIV 主导的 kernel 才能开始，两类资源交替空转。

### 1.3 四个挑战（作者自己列的，每个都对应 ROCmoe 的一条原则）

1. **集合通信的全局屏障**挡住细粒度重叠 → 通信必须变成 AIV 可直接管理的设备侧任务，不要 host 编排。
2. **tile 分解要同时满足算子内与跨算子约束**：GMM 内部的访存重排与 L2 复用优化**要求每个 tile 覆盖完整的专家宽度**，切太细会破坏这些优化；跨算子上 AllToAll tile 与 SwiGLU tile 必须和 GMM 的输入/输出行分块对齐；完成信号必须精确反映目的侧 buffer 的可见性。
3. **调度要同时满足正确性、低开销、高质量**——在线动态调度能适应 token 分布变化，但依赖检查、任务选择、队列管理全在关键路径上。
4. **可编程性与可移植性**：手写全融合 kernel 会把调度策略、通信协议、算子实现耦死，模型/并行配置/数据类型/tiling 一变就得重写；而生产级 GMM、SwiGLU、通信算子里已经有大量硬件相关优化，从头重写既贵又大概率退步。

> 第 4 点正是 ROCmoe P6「核里 native，边界上标准」要防的「死于 NIH」。他们的答案是**低侵入的调度框架**：保留现有高性能算子，把依赖、tiling 策略、调度约束**外部声明**，框架自动生成任务描述、队列顺序和同步事件。

## 2. Method

### 2.1 AIV 驱动的单边通信

昇腾提供 SHMEM 式 RMA：一个 rank 上的 AIV worker 可以通过设备侧地址翻译直接访问远端 rank 暴露的通信 buffer，并更新目的侧的同步信号。

封装成一个原语 **`put_mem_signal(src_buf, dst_rank, dst_buf, range, dst_event_counter)`**：发送侧 AIV worker 把数据块写进远端 buffer，**在数据对接收方全局可见后**更新目的侧事件计数器。

> 和常规通信原语返回 tensor 不同，它在目的 rank 上暴露**两个可观测副作用**：目标 buffer 可读 + 事件计数器更新。所以它同时是数据传输和完成通知。
>
> 这和 NVSHMEM 的 `putmem_signal_nbi` 语义一致——也就是说 [Perseus](./perseus.md) 揭示的那个 fence 病理，在昇腾这条路径上同样有可能存在，只是本文没有跨节点评测所以没暴露。

AllToAll 被拆成一堆 `put_mem_signal` 任务，每个 tile 独立更新自己的完成计数器。Dispatch 方向：某个本地专家所需的输入 tile 全部到齐且信号可读，对应的 GMM 任务就能立刻开始，**不等无关专家/rank 的 Dispatch 完成**。Combine 是反方向。

### 2.2 从算子图生成 tile 任务

应用层用 **ODG（Operator Dependency Graph）** 描述一段 MoE-FFN：算子、张量连接、输入输出元数据、以及每个算子可以合法切分的维度。**这一层只定义数据怎么流、算子怎么可以拆，不决定执行顺序。**

前向 ODG 5 个节点：Dispatch → GMM₁（gate+up）→ SwiGLU → GMM₂（down）→ Combine。
反向 ODG 7 个节点：backward Dispatch → {GMM_act_grad, GMM_w2_grad} → SwiGLU_grad → {GMM_gate_grad, GMM_w1_grad} → backward Combine。

> **反向图里有两组天然独立的 GMM 分支**（act_grad ∥ w2_grad，gate_grad ∥ w1_grad），这是 §2.4 那个 L2 交错优化的着力点，也是 MegaMoE 完全没覆盖的部分。

### 2.3 静态调度与事件驱动同步

调度器离线决定两件事：每个任务放进 CTQ 还是 VTQ；任务之间的同步关系。**队列顺序只定义同资源类型内的执行序，跨资源依赖一律用事件显式表达**，所以 CTQ 和 VTQ 可以独立推进。

每个任务关联两个事件：**dependent event**（执行前的前置条件）和 **trigger event**（完成后发出的信号），每个事件配一个静态生成的**阈值**（要被触发多少次下游才可执行）。worker 执行前轮询依赖计数器直到达阈值，完成后更新触发计数器。计数器放在设备可见内存里，AIC/AIV 都能访问。

这个**基于阈值的机制天然表达多对多 tile 依赖**：多个上游可以累加同一个计数器（比如一个 GMM 任务等若干个 Dispatch tile 全部到齐），多个下游也可以等同一个事件。

正确性来自两点：任务全部由 ODG + SplitSpec 生成，输入区域和上游生产者确定；事件只在上游输出完全可读后触发。ODG 是 DAG，切分只沿拓扑序传播，静态生成的事件关系不会引入环。

整个执行计划序列化成 **SSC（Static Schedule Configuration）**——编译与运行时之间的核心抽象。

### 2.4 统一运行时与执行序优化

**统一运行时**：MindSpore 发起**一个 AscendC 算子**，传入 SSC、模型张量、通信 buffer、事件计数器。AIC worker 消费 CTQ、AIV worker 消费 VTQ，协议都是「取 TD → 等依赖 → 执行 handler → 触发后继」。运行时**不构建依赖图、不分配 TD、不在关键路径上重排队列**。

Task handler 只是包装现有优化过的 GMM / SwiGLU / 通信实现——**调度策略与算子实现干净解耦**。

**执行序优化**（都只重排相互独立的任务，不改 ODG 边、不改任务输入输出区域、不改事件语义）：

1. **RATR（Rank-Aware Task Reordering）**：朴素顺序下所有 rank 用相同的目的 rank 序列发通信，导致某些时间窗内流量集中到同一批目的 rank 或链路上。RATR **按源 rank ID 轮转通信任务顺序**，不同源 rank 从不同目的 rank 开始，环状遍历剩下的。传输的数据集合和依赖完全不变，只改独立通信任务的相对顺序。
2. **Cache-guided GMM interleaving**（反向）：backward Dispatch 之后，act_grad GMM 和 w2_grad GMM 都消费同一份 dispatched 专家激活但彼此独立。如果一条分支整个跑完再跑另一条，共享激活可能已经被挤出 L2。于是**按专家局部性交错这两条分支的 GMM 任务**，把同一专家的 GMM 尽量在时间上靠拢，缩短复用间隔。

## 3. Experiments

### 3.1 Dispatch-to-Combine 模块延迟（均衡路由，64×A3）

| EP | 方向 | Baseline (ms) | HyperParallel (ms) | 加速 |
|---|---|---|---|---|
| EP4 | Forward | 16.3 | 10.2 | **1.60×** |
| | Backward | 27.9 | 19.4 | 1.44× |
| | **Total** | 44.2 | 29.6 | **1.49×** |
| EP8 | Forward | 17.3 | 10.3 | **1.68×** |
| | Backward | 29.8 | 19.6 | 1.52× |
| | **Total** | 47.1 | 29.9 | **1.58×** |
| EP16 | Forward | 18.4 | 11.2 | 1.64× |
| | Backward | 30.5 | 19.9 | 1.53× |
| | **Total** | 48.9 | 31.1 | 1.57× |

**最有说服力的不是加速比，是扩展趋势**：baseline 随 EP 从 44.2 涨到 48.9 ms（集合通信路径的代价在长），HyperParallel-MoE 稳在 **29.6–31.1 ms 的窄区间**——通信、向量、GMM 都变成统一 runtime 里的可调度任务之后，EP 扩大不再线性传导到模块延迟。

前向收益（1.60–1.68×）稳定高于反向（1.44–1.53×），合理——反向的 GMM 密度更高，可压缩的空隙更少。

### 3.2 端到端训练步（自然路由）

**1.08–1.09×**。作者自己解释得很坦白：MoE-FFN 只是训练关键路径的一部分，未改动的模型计算和框架开销仍占大头；而且 baseline 里保留了 MindSpore 的 DVM 级自动融合和图级执行规划，会部分掩盖模块级收益。

### 3.3 微基准（SwiGLU + Add，A3）

**tile 交错与 L2 复用**（M=32K）：

| 指标 | 串行 | tile 交错 |
|---|---|---|
| 延迟 | 723.29 µs | **588.38 µs（1.23×）** |
| L2 命中率 | 5.20% | **25.44%（4.9×）** |

M=8K 时交错反而略慢——tile 同步和队列消费的额外开销还没被局部性收益补偿。**这个拐点的存在很重要**：交错不是免费的。

**静态 vs 动态调度**（这是全文对我们最有价值的一组数）：

| 指标 | 动态 | 静态 |
|---|---|---|
| 每 AIV 任务派发成本 | **2.36 µs** | **0.1 µs** |
| M=2K 总延迟 | 413.00 µs | **54.00 µs（7.65×）** |
| M=32K 总延迟 | 862.80 µs | **588.38 µs（1.47×）** |

注意这个对比是**在已经 taskize 之后**做的（两边同样的事件依赖），所以干净地隔离出了「顺序该在运行时定还是编译期定」这一个变量。

## 4. Limitations

**作者声明的：**

- 静态调度需要稳定的 shape 或 shape bucket；对高度不可预测的 shape / 依赖模式，动态调度仍然有用。
- 端到端只有 1.08–1.09×，作者明确说这应理解为「模块收益在完整训练流程里被稀释后的结果」。

**我认为需要打问号的：**

- **主结果没有做组件级归因**。论文原文：「These numbers evaluate the full optimized configuration...; component-level attribution is outside this main-results comparison.」也就是说 1.49–1.58× 里，单边通信、RATR、反向 GMM 交错、统一 launch 各占多少，**完全不知道**。对想复刻的人这是最要命的缺失。
- **模块级用均衡路由，端到端才用自然路由**。均衡路由下每个专家收到相同 token 数，确实隔离了 router 倾斜，但也意味着 1.5× 这个数字是在**最有利的负载分布**下取得的。[X-Stage](./x-stage.md) 的经验恰恰相反——倾斜路由下调度优化的收益更大——所以两者对不上，值得深究。
- **静态 SSC 与动态路由的张力没有正面回答**。每步只提供「张量指针、路由导出的偏移、新的事件计数器状态」，但专家的实际 token 数每步都在变。shape bucket 未命中时会怎样？重新编译的成本是多少？论文没说。
- profiling 在 256 卡上做，评测只在 64 卡上做；EP16 已是评测上限，而 profiling 的配置是 ep=32。
- 没有讨论数值确定性 / bit-wise 可复现（对比 [UniEP](./uniep/README.md) 明确保 bit-wise）。训练场景里这是硬需求。

## 5. Our take

### 5.1 这是 ROCmoe 立项文档最强的外部旁证

`rocmoe_DESIGN.md` 的六条原则，这篇几乎一一对应上了：

| ROCmoe 原则 | HyperParallel-MoE 的对应物 |
|---|---|
| **P1** stream-free by default | AIV 驱动的设备侧单边通信，完全不走 host 集合通信 |
| **P2** megakernel 是一等构造 | 整个 MoE-FFN 前向或反向 taskflow 在**一次 AscendC kernel launch** 内执行 |
| **P3** 通信是可调度的一等流水阶段 | 原文：「communication itself should be elevated to a first-class schedulable task」 |
| **P4** 静态调度 + 显式资源预算 | SSC 离线编译，运行时只剩 fetch / wait / trigger |
| **P6** 核里 native，边界上标准 | handler 包装现有优化算子，低侵入接进 MindSpore/MindFormers |

**意义在于**：我们文档里说「AMD 需要 native 执行内核而不是移植」，容易被质疑成「给自己找理由」。这篇证明了**另一个非 CUDA 平台的团队，在完全独立的硬件上，得出了同样的结论并做出了同样的架构选择**。这条应该直接写进立项文档的「外部佐证」一节。

而且 P4 一直缺一个量化论据——现在有了：**每任务 0.1 µs vs 2.36 µs，23× 的差距**。

### 5.2 必须诚实承认的架构劣势

他们能做 tile 级异构重叠，是因为 AIC/AIV **物理解耦 + 独立硬件队列 + 跨队列事件原语**。CDNA 上：

- MFMA 和 VALU 在**同一个 CU** 内共享 wave slot，没有独立队列；
- 我们的「异构重叠」只能靠**跨 CU 的 WG 角色分区**来近似（MonolithEP 的 224:16:16），粒度粗得多，而且**每分出去一个角色就少一份 MFMA 算力**——昇腾分给 AIV 的活不占 AIC 的份额，我们分给 COMM 的 CU 就是实打实从 COMPUTE 里扣的。

这是一个真实的劣势，不该在文档里回避。**我们的补偿优势是规模**：256 CU vs 25 AI Core，160 KB LDS/CU，角色分区的自由度大得多。合理的表述是：*昇腾用硬件异构换细粒度，CDNA 用 CU 规模换角色多样性*。

### 5.3 可以立刻抄的两条

**(1) RATR —— 几乎零成本，最该先试。**

MonolithEP 的 COMM_DISPATCH 大概率是在每个 rank 上**用相同顺序遍历目的 rank**，那就和论文说的朴素顺序完全一样，会在时间窗内把 XGMI 流量堆到同一批目的卡上。改成按 rank ID 轮转（`dst = (my_rank + i) % world_size`）是**一行改动**，不改数据、不改依赖。8 卡节点内 XGMI 是全连接，效果可能不如他们的跨卡拓扑明显，但值得先量一把链路占用的时间分布。

**(2) 反向的 cache-guided GMM 交错 —— MegaMonolith 走向训练时的现成结构。**

论文明确指出 **MegaMoE 主要服务推理、不覆盖 MoE 反向算子**，而 HyperParallel-MoE 前向反向都覆盖。ROCmoe 的滩头是**训练**，所以：

> **对 MegaMonolith 来说，HyperParallel-MoE 比 MegaMoE 是更贴切的参考对象。**

他们的反向 ODG（7 节点、两组独立 GMM 分支）和按专家局部性交错的做法可以直接搬。注意在 CDNA 上「靠拢以复用 L2」的目标缓存不一样——他们是 192 MB 统一 L2 且读带宽 >4× HBM，我们每 XCD 只有 4 MB L2，真正对应的是 256 MB Infinity Cache，**驻留特性和带宽都不同，必须自己测复用窗口**。

### 5.4 一个和 X-Stage、Fleet 连起来的判断

三篇放在一起看，**tile 交错这件事有一个共同的拐点结构**：

| 论文 | 交错带来的收益 | 交错带来的代价 | 拐点 |
|---|---|---|---|
| HyperParallel-MoE | L2 复用（5.2%→25.4% 命中率） | tile 同步 + 队列消费开销 | M=8K 以下变慢 |
| [X-Stage](./x-stage.md) | 通信突发排空窗口 | 跨专家交错**损失** L2 局部性 | 部分配置退到 0.94× |
| [Fleet](./fleet.md) | XCD 内绑定带来的 L2 复用 | 放弃跨 XCD 的负载灵活性 | — |

有意思的是 HyperParallel-MoE 和 X-Stage 对 L2 的判断**方向相反**：前者说交错**提升** L2 复用（因为交错的是生产者-消费者对，缩短复用间隔），后者说交错**降低** L2 复用（因为交错的是不同专家的权重，扩大了工作集）。

**结论不是矛盾，是粒度问题**：*沿数据流交错（同一份数据的生产者与消费者靠拢）提升局部性；跨数据流交错（不同专家的权重轮转）破坏局部性。* 这条区分在我们设计 MegaMonolith 的调度器时是直接可用的判据，而且三篇里没有一篇明确写出来——值得作为我们自己的贡献点。

### 5.5 对我们汇报口径的提醒

模块级 **1.5×** → 端到端 **1.08×**。MonolithEP 现在的目标是「DSv3 forward MoE 层从 Megatron 的 18.41 ms/rank 压到 ≤12 ms」，这是**层级**数字。一旦要对外讲，必须同时给出它在完整训练步里的占比和稀释后的端到端预期，否则会被同样的方式质疑。建议在 MonolithEP README 里补一行端到端外推。

## 6. 延伸阅读

1. **本文代码**（gitcode 上的 `hyper_parallel/core/multicore`）——SSC 的数据结构和 TD 的字段定义值得直接读，那是我们设计 ROCmoe 调度 IR 的现成参照。
2. **[AutoMegaKernel](./automegakernel.md)**——同样是「把调度决策提到编译期」，但走的是 schedule-IR + 静态安全校验的路子，和 SSC 互补。
3. **[UniEP](./uniep/README.md)**——本文 §7 明确对标的 GPU 侧同类工作。
4. **DeepSeek-V4 技术报告 / MegaMoE**——本文对它的定位判断（推理为主、无反向）值得我们独立验证一遍，因为 MegaMonolith 的整个立项都建立在「移植 MegaMoE 架构」上。

## 参考

- 论文：<https://arxiv.org/abs/2605.23764>
- 本次检索的完整清单：[`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md)
