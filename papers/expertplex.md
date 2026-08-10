# ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels

> [arXiv 2607.18002](https://arxiv.org/abs/2607.18002) (v2, 2026-07-21, cs.DC) · 北大（Bingyang Wu, Chao Jin, Zili Zhang, Yinmin Zhong, Xin Jin 等） · CC BY-SA 4.0
> 实现：改 DeepGEMM + DeepEP v1 + SGLang
> 硬件：H800 ×8/节点（NVLink），多节点最多 3 机，每节点 8×200 Gbps IB
> 模型：MiniMax-M2.7（230 GB FP8，256 routed experts，top-8）· GLM-5.1-FP8（756 GB FP8）

## TL;DR

**这是五篇里唯一站在 ROCmoe 对立面的一篇**，但读完之后会发现它其实是在给我们的 P4 划定适用边界，而不是推翻它。

论文的核心主张：**Green Context 式的静态 SM 空间划分跟不上 MoE 的动态性**。资源在一个 kernel 期间被定死，重配置要 CPU 介入所以只能在 prefill 层边界做；而每层每 rank 的激活专家数在变、attention 与 MoE 需求不同、MoE 内部在 dispatch/计算/combine 之间切换瓶颈。结果是要么 head-of-line blocking，要么预留资源空转。

它的答案是 **APK（Adaptive Persistent Kernel）**：常驻 kernel 在 **tile 边界**调度，给 decode 提供与序列长度无关的抢占上界，把空闲 CTA cluster 还给 prefill，全程不需要 CPU 介入或重新 launch，且**保持 CUDA Graph 兼容**。

最有说服力的一组对照（decode 延迟保护目标下）：

| 机制 | decode 延迟代价 | prefill 变慢 |
|---|---|---|
| 优先级 CUDA stream | **+13.79×** | — |
| MPS | 接近独占 | **3.33×** |
| Green Context | 接近独占 | **4.07×** |
| **ExpertPlex APK** | **+8%** | **1.12×** |

**但读法很重要**：它反的是「一个 kernel 期间固定的空间划分」，反的场景是「两条**独立请求流**共享 GPU、且算子时长差 84–101×」。它**没有**反「单一已知依赖图内部的编译期调度」——事实上 ExpertPlex 自己也把两个 grouped GEMM + 激活 + 前后处理 + 调度器融进了一个常驻 kernel。**分歧只在「谁来决定调度」，不在执行模型。**

## 1. Problem

### 1.1 两条现有路线各自的病

**instance-level PD 分离（PDD）**：每个阶段一份完整模型副本，分配粒度随模型增长。DeepSeek-V3 的部署报告里一个单元用 32 GPU 做 prefill、320 GPU 做 decode。

- 小集群配不出这个比例 → 资源浪费；
- 大集群配得出，但**故障爆炸半径大**（分层通信里一个 rank 挂掉能拖停整个单元），且只能按大部署单元的粒度伸缩；
- 重复的专家权重挤占 KV cache。

**PD 共置（Green Context / MPS / MIG）**：去重了权重，但**跟不上动态性**，三个维度：

1. EP 下每 GPU 上激活的专家数和 token 数**逐层变化**；
2. 同一层内 attention 与 MoE 的资源需求不同；
3. 每个 MoE 模块在 dispatch / 专家计算 / combine 之间切换，瓶颈在通信与计算之间来回移动。

固定划分跟不上这些变化，改分配又要 CPU 协调 + 等 kernel 完成，所以现有系统只能在 **prefill 层边界**重划分。两种失效模式：prefill 占太多 SM 时 decode 卡在不可抢占的 prefill kernel 后面；给 decode 预留的资源在它没活时空转。

**这个数字很关键**：EP4 下 MiniMax-M2.7，decode 的 grouped GEMM（8 个激活专家）耗时 **17.7–34.7 µs**，而 16K token 的 prefill GEMM 耗时 **1.8–2.9 ms**，差 **84–101×**。每层都有 MoE 操作，所以阻塞会逐层累积。

另外，划分每一张 GPU 还会让每个阶段的本地资源变少 → 需要更大的并行度 → 更多通信 → 网络干扰更严重。而且常规的双边通信在两阶段独立调度时会**死锁**（各自等着被对方占用的 rank 上的接收侧工作）。

### 1.2 五个属性的对照表

| | API 拦截 | CUDA stream | MPS | Green Context | MIG | ExpertPlex |
|---|---|---|---|---|---|---|
| CUDA Graph 兼容 | | ✓ | ✓ | ✓ | ✓ | ✓ |
| 时间复用 | ✓ | ✓ | | | | ✓ |
| 空间复用 | | | ✓ | ✓ | 受限 | ✓ |
| **有界快速抢占** | | | | | | **✓** |
| **有界快速回收** | | | | | | **✓** |

最后两行是全表的重点，也是 APK 的存在理由。

## 2. Method

### 2.1 混合架构：共享专家，分离 attention

MoE 权重占参数量 >95%（DSv4-Pro 95%、GLM-5.1-FP8 96%、MiniMax-M2.7 98%）。

- **共享 MoE 服务器**：消掉跨阶段的权重重复，并复用动态稀疏的专家计算——一个阶段的计算可以填另一个阶段 attention–expert 流水里的气泡。
- **按阶段分离 attention**：attention 权重 <5%，所以可以给每个阶段**整张 GPU**而不是 GPU 内分区。这样保住了每个阶段在一张卡内的本地算力，避免了「分区 → 需要更大并行度 → 更多通信 → 跨阶段网络干扰」这条链。

结果是用更少的 GPU、更少的流量匹配阶段需求，同时改善伸缩粒度和故障隔离。

### 2.2 APK：tile 级调度

**为什么选 tile 作为调度单位**：它是这些算子里最小的可独立完成单元，尺寸被寄存器、shared memory、TMA buffer、Tensor Core 形状限死，所以**输入变长只会增加 tile 数，不会拉长单个 tile**。SM90 上他们的 DeepGEMM 配置用 ≤128×192 输出块、128 宽归约的 CTA tile；实测 MoE 侧各操作的 tile 边界间隔 **2.2–25.3 µs**（GEMM 部分 <10.7 µs），与操作总长无关。

**tile 边界切换是零状态的**：此时没有活的累加器、TMA 事务、shared memory buffer 或通信状态，所以切换**不需要 checkpoint、restore 或重算**。更细的抢占会保留流水状态；kernel 或层级调度则保留长阻塞。APK 保持每个算子原生的 CTA/cluster 形状、TMA multicast、warp specialization，**只改「下一个 tile 是谁的」**。

调度以 **CTA cluster** 为粒度，因为被 TMA multicast 连起来的 CTA 必须执行同一个 tile。不同 cluster 跑不同阶段 = 空间复用；每个 cluster 在 tile 边界换阶段 = 时间复用与快速回收。

### 2.3 有界抢占协议（本文最值得抄的工程件）

**为什么不能各 CTA 自己在 tile 边界查一下**：高性能算子跨 tile 流水化了 warp 和 CTA——CTA 内 TMA warp 在载 tile `k+1` 而 math warp 在消费 tile `k`；如果一个 warp 换了阶段，另一个会永远等旧阶段的数据或 mbarrier 事件。跨 cluster 同理，一个独立切换的 CTA 会把同伴卡在 TMA multicast 或 cluster barrier 上。而每个 tile 后 stall 整个 cluster 又会把微秒级流水串行化。

**APK 的做法是让一个协同决策沿内存层级传播**：

```
attention server 写 system-scope word P（紧急 decode 到达）
  → CTA 0 在自己当前操作的 tile 边界检查 P，
    为每个 cluster i 写一个 device-scope word p_i     ← 避免反复的 system-scope 访问和 grid-wide barrier
      → cluster leader 在下一个 tile 边界通过 DSMEM 广播决策
        → CTA 内第一个流水 warp 读到决策，warp 内广播，
          用 mbarrier handoff 通知后续 warp（在它们再取 tile 之前）
```

决策在一个检查 epoch 内固定，所以观察到之后没有任何 CTA 能再取旧阶段的 tile。**抢占上界 = 一个 tile 执行时间 + 一次本地 cluster 检查 epoch，与被打断操作的总长无关。**

参考对比：REEF 报告的最好延迟是 35 µs 且需要重算被抢占的 kernel。

### 2.4 attention 发起的单边 MoE 通信

attention 服务器**推** dispatch 数据进 MoE 侧的最终 buffer，**拉** combine 结果出来。直接访问最终 buffer 消掉了 MoE 侧的协调与轮询，避免了跨阶段死锁，并让一个阶段的通信与另一个阶段的计算重叠。

实现上替换了 DeepEP 的 receiver-driven all-to-all，改成 attention-initiated M-to-N；IBGDA 路径上定制了 `nvshmemi_ibgda_get_nbi_warp` 支持在特定 RDMA 连接上做细粒度 pull。

流量隔离：prefill 的 scale-out 流量尽量在 attention 服务器之间走，decode 直接在 attention 与 MoE 服务器之间通信。

### 2.5 tile 感知的延迟模型（一个小而有用的建模结论）

拟合形式 `t̂(x,s) = α + βx + γxs + δxs²`。attention 的 `x` 是本地 batch、`s` 是序列长度。

**关键观察：对 MoE 计算，原始的 routed-token 数不够用，因为延迟跟随的是执行的 tile 数。** 若专家 `e` 收到 `m_e` 行 token、kernel tile 高为 `M_t`：

```
x_moe = Σ_{e | m_e > 0} ⌈ m_e / M_t ⌉
```

**同样 token 数的 batch，激活专家数不同时延迟不同**——因为 Tensor Core 和 TMA 流水按对齐 tile 工作，**每个活跃专家即使只收到几个 token 也要触发至少一个 tile 的元数据、权重搬运和计算**。作者强调这是分块 MoE kernel 的本质，不是某个模型或 GPU 的特性。

### 2.6 在线 SM 再分配

离线求解器针对平均负载给出 decode 的 cluster 份额 `q`，APK 把它当**争用策略**而不是静态划分：只有一个阶段就绪时用全部 cluster；争用时按当前 decode footprint 与离线期望的比值缩放：

```
q' = min(Q_max, ⌈ q · x_moe / x*_moe ⌉_c)
```

`⌈·⌉_c` 向上取到 CTA cluster 的整数倍，`Q_max` 保护 prefill 进度，剩下的全给 prefill。

## 3. Experiments

指标是 **P90 goodput**：至少 90% 请求同时满足 TTFT 和 TPOT SLO 的最高到达率。

### 3.1 端到端

**MiniMax-M2.7 / ShareGPT，11.3 req/s/node：**

| 对比基线 | 提升 |
|---|---|
| SGLang-ChunkedPrefill | **5.65×** |
| SGLang-Colocated | 2.72× |
| SGLang-PDD | **2.01×** |
| SGLang-PDMux（Green Context） | 1.41× |

**MiniMax-M2.7 / LooGLE（长请求）**：vs Colocated **4.12×**，vs PDMux 1.28×。

**GLM-5.1-FP8（多节点）**：vs ChunkedPrefill 3.3×(ShareGPT) / 5.0×(LooGLE)；vs Colocated 1.5× / 2.5×。ShareGPT 上与 PDMux 基本打平（约 1.5 req/s/node，PDMux 的 TP attention 给短请求更多并行度所以 TTFT 占优），**LooGLE 上 1.66×**——长请求暴露了网络干扰和 Green Context 在保 TPOT 前提下压不满 GPU 的问题。

### 3.2 APK 机制本身（单卡，GLM-5.1-FP8 形状）

prefill GEMM 启动后 10 µs 启动 decode GEMM（decode 128 token，prefill 8192 token，均激活 8 专家）。见 §TL;DR 的表。结论：**APK 是唯一同时进入 decode 低延迟区且保住 prefill 性能的机制**。原因是 decode GEMM 短而间歇——APK 平时把所有空闲 cluster 给 prefill，decode 到了才抢占，decode 做完立刻还回去。

### 3.3 开销

| 项目 | 开销 |
|---|---|
| tile 级调度（contiguous 布局，prefill） | **<12%** |
| tile 级调度（masked 布局，decode） | **<20 µs 绝对值**；多激活专家时相对 <10% |
| attention 发起的通信（normal 模式） | 与 DeepEP v1 相差约 **5%** 以内 |
| 同上（low-latency 模式） | 差异约 **45 µs** 以内 |

调度开销低的原因很关键：它**每个 tile group 跑一次，不改 TMA 和 Tensor Core 流水**。

## 4. Limitations

**结构性的：**

- 纯**推理服务**场景，完全不涉及训练。APK 的整个动机（两条时长差 84–101× 的独立请求流）在训练里不存在。
- 深度依赖 Hopper/Blackwell 特性：**CTA cluster、DSMEM、TMA multicast、mbarrier**。抢占协议的每一层传播都绑在这些原语上。
- tile 尺寸是按算子性能调的，不是按抢占调的；25.3 µs 的边界间隔是**结果**而非保证。换个 kernel 配置这个上界就变了。

**评测上需要打问号的：**

- **PDMux 这个基线可能被削弱了**。它基于 MuxWise（面向稠密模型），被作者改造来支持 MoE，且「实现只兼容张量并行 attention」所以被迫用 TP。拿一个为别的场景设计、又被外人改过的系统当主要对手，说服力打折。
- GLM-5.1-FP8 在 24 GPU 上 PDD 直接 OOM，所以**该模型没有 PDD 数字**；而多个基线表达不了 ExpertPlex 的 24 GPU 细粒度布局，于是跑在 16 GPU 上按「每节点请求率」比较。这个归一化对 ExpertPlex 有利。
- 为了对 PDD 公平，采样序列长度被截到 PDD 的 KV cache 容量上限——合理，但也限制了负载真实性。
- 离线放置优化器针对期望负载求解，在线适配只是一条比例规则（式 4），没有评估负载分布漂移时的鲁棒性。

## 5. Our take

### 5.1 它到底反了 ROCmoe 的什么

先把它反的东西说准：

| 它反对的 | 它没有反对的 |
|---|---|
| 一个 kernel 期间**固定**的 SM 空间划分（Green Context / MPS / MIG） | 编译期决定单一依赖图内部的执行顺序 |
| 要 CPU 介入才能重配置、只能在层边界改 | 常驻 megakernel 执行模型 |
| 两条**独立请求流**共享 GPU 时的静态比例 | 设备侧发起的单边通信 |

**判别式很清晰**：*工作是来自一张已知的依赖图，还是来自两条独立的请求流？*

- 训练稳态 = 前者。算子结构、并行配置、主导 shape 都可编译期确定（这也正是 [HyperParallel-MoE](./hyperparallel-moe.md) 用 SSC 的理由，而且它测出静态派发 0.1 µs vs 动态 2.36 µs）。
- 共置服务 = 后者。两条流的到达时间彼此不可知，算子时长差两个数量级。

所以 **ROCmoe P4「编译期静态调度」在训练滩头上是站得住的**，但文档里必须加一句作用域声明：

> P4 的前提是单一训练 taskflow。若 ROCmoe 未来要覆盖共置推理服务，P4 必须放松成 APK 式的「tile 边界争用策略」——静态求出的比例只作为争用时的目标值，非争用时全部让给就绪方。

顺带一提，ExpertPlex 的在线规则（式 4）其实就是这个折中：**离线算比例、在线按 tile 粒度重解释**。这条完全可以作为 ROCmoe 的 P4 演进路径写进文档，比「静态 vs 动态二选一」成熟得多。

### 5.2 更重要的发现：P1/P2/P3 是无争议的

ExpertPlex 自己**也**把两个 grouped GEMM + 激活 + MoE 前后处理 + 调度器融进了一个常驻 kernel，**也**用设备侧单边通信，**也**把通信当一等调度对象。

> 五篇论文横跨 NVIDIA、昇腾、静态派和自适应派，**没有一篇在质疑「常驻 megakernel + 核内调度 + 设备侧单边通信」这个执行模型本身**。争的只有「谁来决定调度」。

这对立项文档是个很强的论据：我们押的是被普遍接受的那部分，争议只落在一条可以带作用域声明的原则上。

### 5.3 最该抄的：tile 边界抢占传播协议

即使我们只做训练，这个协议也有用——**任何想在运行中改变 WG 角色配比的常驻 megakernel 都会撞上同样的死锁问题**。MonolithEP 现在 `224:16:16` 是编译期定死的；一旦想做动态再平衡（比如某层专家倾斜严重时临时把 COMM_DISPATCH WG 转成 COMPUTE），就必须解决「TMA warp 在预取下一 tile、math warp 在算当前 tile，单独切换会死锁」这个问题。

**CDNA 上没有 DSMEM、没有 CTA cluster、没有 mbarrier**，所以传播链要重新设计。一个可能的对应：

| ExpertPlex（SM90） | CDNA 候选对应物 |
|---|---|
| system-scope word `P` | system-scope atomic in fine-grained host-visible memory |
| 每 cluster 一个 device-scope word `p_i` | 每 XCD 一个 device-scope atomic（放 GDS 或 L2-resident） |
| cluster leader 经 DSMEM 广播 | WG 内 LDS 广播（我们没有跨 CU 的 DSMEM，粒度只能到 WG） |
| warp 内广播 + mbarrier handoff | `s_barrier` + `readfirstlane` |

粒度会比他们粗一级（我们的「cluster」只能是单个 WG），但**零状态 tile 边界切换**这个核心性质是可以保住的。值得单独做一个设计 spike。

### 5.4 立刻可用：`x_moe` 作为 roofline 的输入量

```
x_moe = Σ_{e | m_e > 0} ⌈ m_e / M_t ⌉
```

这条应该直接进 ROCmoe P5 的判据工具箱。它解释了一件我们在 MMOE profiling 里应该见过但可能没归因对的现象：**用 token 数衡量的「负载均衡」是误导性的**——一个只收到 3 个 token 的活跃专家，仍然要付一个完整 tile 的元数据、权重搬运和 MFMA 代价。

推论有两条：
1. 评估路由均衡时应该报 `x_moe` 而不是 token 方差；
2. 在 MI355X 上 `M_t` 通常比 SM90 大（MFMA tile 更宽），所以**同样的路由倾斜在 CDNA 上的 tile 浪费更严重**。这可能是一个我们独有的、值得量化的 AMD 侧问题。

### 5.5 AMD 侧的推论：Green Context 的批评在我们这里更成立

ROCm 上对应 Green Context 的机制是 **CU masking（队列级 CU mask）**，而且比 Green Context 更粗——mask 在**队列创建时**设定，改一次要重建队列。论文对 Green Context 的三条批评（kernel 期间固定、重配置要 CPU、跟不上逐层专家负载）在 AMD 上**只会更严重**。

这反过来强化了 ROCmoe 的立论：既然 AMD 的硬件分区手段比 NVIDIA 还弱，**把调度放进 kernel 内部就不是选项之一，而是唯一选项**。这条应该写进 `rocmoe_DESIGN.md` 的动机部分，和现有的「HSA queue → ACE 映射导致多流重叠不可靠」那段并列。

### 5.6 一个可能有用的通信设计点：pull 式 combine

三篇论文在同一设计空间里给出了三个不同的点：

| 方案 | 方向 | 接收侧负担 |
|---|---|---|
| [Perseus](./perseus.md) / [HyperParallel-MoE](./hyperparallel-moe.md) | 发送方 push + signal | 接收方轮询 flag |
| ExpertPlex dispatch | attention 侧 push 进最终 buffer | 无协调 |
| **ExpertPlex combine** | **消费方 pull** | **MoE 侧只发布结果 + ready 标志** |

pull 式 combine 对我们有个具体好处：**MoE 侧不需要知道目的地**，也就不需要常驻的 TAIL_COMBINE WG 去做散射——那 16 个 WG 可能可以还给 COMPUTE。当然这要求消费侧有能力发起 pull，在纯节点内 IPC 场景下是可行的。值得作为 MonolithEP 的一个备选 combine 路径评估。

## 6. 延伸阅读

1. **DeepEP v1 源码**——本文替换的那条 receiver-driven all-to-all 路径，以及 `nvshmemi_ibgda_get_nbi_warp` 的原始实现。
2. **REEF / LithOS / GPreempt / PipeSwitch**——GPU 抢占的前作，本文 §7.6 用作参照点（REEF 最好 35 µs 且需重算）。理解「为什么 tile 边界抢占是质变」需要它们做背景。
3. **[MegaScale-Infer](./megascale-infer.md) / [DisagMoE](./disagmoe/README.md)**——解耦式 MoE 服务的另一条路线，和本文的「共享专家 + 分离 attention」构成对照。

## 参考

- 论文：<https://arxiv.org/abs/2607.18002>
- 本次检索的完整清单：[`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md)
