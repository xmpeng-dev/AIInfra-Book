# Tessera: A Holistic Pipeline Parallelism Framework for Trillion-Parameter Heterogeneous MoE Training

> [OSDI '26](https://www.usenix.org/conference/osdi26/presentation/hu-weifang)（Operational Systems track，2026-07-13～15，Seattle）· [PDF](https://www.usenix.org/system/files/osdi26-hu-weifang.pdf)
> Weifang Hu（HUST，实习于阿里云）、Langshi Chen、Man Yuan（三人并列一作）+ Youyang Yao / Xiulong Yuan / Li Tian / Yong Li / Wei Lin / Zhengping Qian / Jingren Zhou（阿里云）、Xuanhua Shi（HUST）
> 硬件：阿里生产集群，NVIDIA Hopper，8 GPU/节点，RoCE 网络；生产规模 4,096–12,288 GPU，受控实验 128 / 256 GPU
> 模型：Qwen3-235B、Qwen3-Next-80B（公开）+ Qwen3-L/XL、Qwen3-Next-M/L/XL（未公开）；对比实验另含 DeepSeek-V3、Nemotron-3 Super
> 实现：Megatron-LM 的**插件**，11,000 行 Python + 2,000 行 C++。**未开源**。2025-04 起持续生产使用

## TL;DR

**核心命题：流水线切分和 overlap 调度是循环依赖的，而所有现有系统都把它们当成瀑布流做。** 切分决定了哪些 chunk 会在同一个 rank 上并发（形成 overlap pair），而 overlap 效率又反过来决定哪个切分最优——所以「按串行代价切平 → 套一个固定 overlap 模板」这条流水线在异构 MoE 上直接失效。

**触发点是 Qwen3-Next 的结构异构**：每 4 层按 3:1 排布（3 个 Gated DeltaNet 线性注意力 + 1 个 full softmax attention），每层后跟一个稀疏 MoE（TopK=10/512）。相邻 attention 算子在生产序列长度下有**最高 10× 的计算时间不对称**，即便最小的 2-stage 流水线也能产生**多达 8 种不同的 overlap pair 类型**，一个模板服务不了。实测不同 pair 的 overlap 收益差 **3×**（C-C pair 降 41.6% 延迟，D-D pair 只降 14.0%）。

**Tessera 的解法是「先 profile 再切分」**：枚举有界的候选切分 → 为每个候选 pair 合成细粒度 overlap 调度并**在真实硬件上实测** post-overlap cost → 用 MILP 按实测边代价选切分。外加一个运行时的 Dynamic Bubble Optimizer（DBO），用路由元数据预测气泡、往里塞 Wgrad 这类可移动任务。

**数字**：生产集群上 vs 阿里自己调优过的内部 Megatron fork，5 个 workload、4,096–12,288 GPU 全面 **+20%～33%**，万亿模型 Qwen3-XL 达 **39.0% MFU**。受控 256 GPU vs 公开 recipe 的 Megatron-Core MoE：Qwen3-235B **1.24×**，DeepSeek-V3 / Nemotron-3 Super **基本持平**（后两个数字比摘要诚实得多，见 §6）。

**对我们最有价值的三条，全在 §5「工程经验」里，而不在方法里**：

1. **他们评估过 Comet 式的 intra-microbatch 融合，然后在生产里否掉了**——理由是切序列维会碎掉 GEMM、降低算术强度，而激进融合会把 MoE kernel 和特定通信后端**耦死**，导致后端无法独立升级。这是对 MonolithEP / UniEP 这条融合路线的一次来自生产的正面反驳（§7.1 详辨）。
2. **EP 通信 kernel 稳定占用约 20 个 SM**，与 Attention / MoE-MLP kernel 并发时造成**稳定 10–20% 的减速**。这把「通信抢 CU」从定性变成了可用的量化常数。
3. **解析代价模型平均低估 5%，但尾部误差达 15%，而尾部误差会翻转 MILP 的相对比较**——所以他们放弃了便宜的 primitive 级 profiling，默认用昂贵的 chunk-pair 实测。任何想做自动并行的人都该记住这条。

---

## 1. Problem：三个复合失效

论文把现有系统的做法概括为瀑布流：**先按串行层代价切分 → 再套固定 overlap 模板 → 最后执行静态计划**。异构 MoE 把这三步各破坏一次。

### 1.1 背景：overlap pair 与 post-overlap cost

先定义执行模型。PP 把模型按层切成 **chunk**，chunk 分配到 virtual stage。interleaved 1F1B 给每个物理 rank 分配多个 virtual stage，于是同一个 rank 上相邻的操作可能来自**不同 chunk、不同 microbatch**——一个 chunk 的通信就能和另一个 chunk 的计算并发。论文管这个叫 **inter-microbatch overlap**，管共享并发窗口的两个 chunk 操作叫 **overlap pair**。

关键细节：**pair 是有方向的**。edge 记录两个端点各自的 pass 方向，所以 (B 的 backward, D 的 forward) 和 (B 的 forward, D 的 backward) 是两条不同的边。self-loop 表示同一个 virtual stage 不同 microbatch 之间的 overlap 机会。

一个 pair 的性能指标不是构成操作的串行和，而是 **post-overlap cost**——共同调度后的 makespan。它落在 `max(T_c, T_d)`（完美重叠）和 `T_c + T_d`（零重叠）之间。

**uniformity assumption**：传统 PP overlap 策略隐含假设模型由算术强度和通信模式相似的「Transformer 砖块」堆成，于是所有 pair 结构相同，一个手工模板 + 按层数配平自然同时得到有效 overlap 和负载均衡。异构架构打破了这条。

### 1.2 失效一：overlap 空间组合爆炸（§2.3.1）

Qwen3-Next 的 3:1 GDN/attention 混排，让 chunk 天然不对称。即使最小的 2-stage 切分，也能产生**多达 8 种不同 pair 类型**，每种需要结构上不同的细粒度交错：计算密集的 chunk 能吃掉通信密集 chunk 的 A2A，而两个通信密集的 chunk 凑一起就会留下大段暴露延迟。现有系统靠刚性模板（如「永远把 forward combine 对齐到 backward attention」）——最优交错其实取决于构成 chunk 的具体硬件资源占用。手工枚举不可行。

### 1.3 失效二：切分目标本身是误导的（§2.3.2）

标准切分器把层的串行执行时间求和后配平，这隐含假设 **overlap 效率（被隐藏的通信比例）在全模型是常数**。

实测（Qwen3-Next-80B 8 层、序列 256K、128 GPU）：

| overlap pair | 重叠带来的延迟下降 |
|---|---|
| C-C | **41.6%** |
| D-D | **14.0%**（两侧都是计算主导，没有 A2A 可藏） |

**串行配平的切分，在并行执行下会严重失衡**，其瓶颈 post-overlap cost 比 overlap-aware 切分高 **1.14×**。

推论很反直觉：**最优切分器应该故意制造串行意义上「不均衡」的分配**，只要这些配对具有互补的计算/通信剖面。而且依赖是双向的——切分质量无法在不先调度并实测每个候选 pair 的情况下评估。**这就是循环。**

论文补了一句很实在的话：在他们的经验里，领域专家为每个新模型架构手工调这个耦合要**花几周**。

### 1.4 失效三：运行时随机性（§2.3.3）

即便静态计划最优，路由随机性仍留下残余气泡。Qwen3-Next 的**极稀疏路由放大了这一点**：每 token 只激活 512 个专家里的 10 个，每个专家期望分到的 token 份额更小，导致跨迭代的 per-expert / per-rank token 数**相对波动更大**。

**使运行时优化可行的关键观察**：不是所有 rank 本地的工作都在 microbatch 关键路径上。

- **backbone task**：定义 stage 延迟的关键路径任务
- **movable task**：时序灵活的任务（权重梯度计算 Wgrad、梯度归约等）

延迟 movable task 并注入到路由引发的空隙里，就能回收静态规划留下的吞吐。

---

## 2. Method：静态规划三步走

Tessera **固定**高层流水调度模板（如 interleaved 1F1B），在模板内部协同优化切分与 overlap 调度。

### Step 1：overlap 图与候选空间

调度模板确定了一组 virtual stage 以及哪些 stage 可以在同一 rank 上并发——这些关系定义了 **overlap 图 G=(S,E)**。节点是 virtual stage，边是模板在某个 rank 上创造的每一个 overlap 机会（记录两端的 pass 方向，端点可以相同）。

**图的拓扑由模板固定，待定的只是每个节点承载哪些层。** Tessera 从串行代价配平的基线切分出发，通过**在基线边界附近扰动**为每个 stage 生成候选集 `C_s`（每个候选是一段连续层区间）。候选被限制在基线的**有界邻域**内，以保证 overlap 调度和 profiling 可承受。

### Step 2：overlap 调度合成 + 真机 profiling

对每条边上的每个候选对，层区间 / pass 方向 / 设备拓扑就全确定了，构成一个具体的 overlap pair。Tessera 把它分解成两个 **task DAG**（每个 chunk 一个），task 是原子调度单元（Dispatch / MLP / Combine / Attn / MLP-W 等），带有时长、资源类型（Comp / Comm）和依赖约束。

然后用**事件驱动的 list scheduler**（Algorithm 1）在两种资源上最小化 makespan，SELECT 用两条启发式：

- **backbone-first alignment**：backbone task 优先，gap-fit 规则挑时长最接近互补资源剩余窗口的任务
- **conditional deferral**：movable task 只在能塞进当前空隙且不延长 pair makespan 时就地放置；否则 defer。backbone 放置完成后，被 defer 的任务用 best-fit 回填到其生命期内的残余槽位，实在没槽位的挂到尾部

**然后在 reference device group 上真机执行合成的调度，实测 post-overlap cost。** 这一步是全文最重的工程判断：

> Physical profiling is necessary because real hardware interference causes the true overlapped cost to deviate from analytical estimates.

reference device group 是一组专用设备，配置与目标 rank 相同的 TP/EP 拓扑。为控制开销：每个不同的 overlap pair 只 profile 一次；当多个候选对产生相同的 chunk 规格 / pass 方向 / 设备拓扑时共享 profile；结果按 device-mesh class + chunk 规格缓存，跨 rank 和 replica 复用。

### Step 3：overlap-aware 切分选择（MILP）

所有 `T_{e,c,d}` profile 完之后，切分选择退化成图标号问题。目标函数用的是**代理目标**：**被选中的边中最大的 post-overlap cost**（`min T_dom, s.t. T_dom ≥ Σ T_{e,c,d}·z_{e,c,d} ∀e`）。论文明确说这是 surrogate——瞄准的是跨 microbatch 反复出现的稳态 overlap 机会，因为一条高代价的边会**反复**暴露成 straggler。

约束：每个节点恰选一个候选；边变量与两端选择一致；完整求解器另加连续不重叠层区间、设备拓扑兼容、per-rank 显存容量。

**复杂度**：把搜索解耦成 stage 级变量 + 相邻 stage 一致性约束，二元变量数从 rank 级枚举的 `O(N·K^vp)` 降到 `O(N·vp·K²)`。

**profile-guided pruning（安全剪枝）**：基线切分的瓶颈代价 `T_base` 是 `T_dom` 的上界，任何 profile 代价超过 `T_base` 的候选对直接丢弃（选它不可能优于基线）；某个 stage 候选若丢光所有关联边上的合法配对，则从搜索空间移除。实测把问题规模缩小 **3–5×**，且保留所有 `T_dom ≤ T_base` 的解。

### 2.4 Dynamic Bubble Optimizer

**气泡预测**：EP 组内各 rank 通过一个**搭在现有 MoE dispatch 路径上的轻量 collective** 交换 per-expert token 计数。因为只在 EP 组内、只传标量而非张量，**不引入集群级同步**。与静态 profile 的偏差就预测出气泡的位置和大小。

**槽位预标注**：静态规划期就在 overlap 调度中 **MoE 计算任务之后的结构边界**预先标注目标槽位。理由讲得很清楚——overlap pair 的两个 chunk 可能来自不同 microbatch、token 分布独立，二者延迟可能朝**相反方向**漂移，所以这些边界上的空窗跨迭代波动最大。

**异步定尺**：token 分布在 forward dispatch 时就已解析，比目标槽位早若干步可用，所以尺寸计算本身能和有用计算重叠。

**movable task pool**：按 deadline 紧迫度排的优先队列。Wgrad 在其前置 backward 计算完成后立即入队，硬 deadline 是迭代结束。论文观察到**这个池在生产中通常很充裕**——负载不均事件下 Wgrad 天然会在关键路径外堆积。池受**每 GPU 可配置容量 + 可用显存余量**双重限制；触顶后新就绪的 movable task 留在原位不 defer。

**保证执行**：某 movable task 的 deadline slack 逼近零而仍未找到合适槽位时，运行时抢先把它注入主流，**正确性优先于优化**。

**填充策略**：贪心，打分偏好「时长贴合 + slack 小」；找不到就留空。开销 **< 10 µs**——因为搜索空间受限、`b_β` 异步预算、且只在**稀疏预标注**的槽位前触发而非每个 task 后都跑。

### 2.5 实现：plan-agnostic 执行引擎

C++ 实现的执行引擎把「调度定义」和「执行」解耦——把静态计划当作**规格说明**而非硬编码控制流来解释，靠**无锁状态机**协调。主 Forward / Backward 的 Torch 线程各自驱动任务，在指定的 yield point 暂停以与引擎全局状态同步；另有一个后台线程按计划处理注册的 movable task。

**侵入性极低的 API 设计**（这点对 Primus 有直接借鉴价值）：

```python
tessera.scheduler.advance("F_DISP")      # 标记 host 侧探针点，引擎可在此阻塞/放行当前 fwd/bwd 线程
dispatched_tokens = dispatch(hidden_states)
tessera.scheduler.advance("F_MLP")
expert_outputs = compute_experts(dispatched_tokens, ...)

tessera.scheduler.register_task(          # 暴露 backbone 之外的可移动任务
    task_name="B_WGRAD", func=compute_weight_grad, args=(...))
```

`advance(TaskName)` 只是在**已有的** forward/backward 代码里插探针，**把连续执行切成命名 task，而不把张量计算搬出宿主框架**——不需要开发者把 MoE 模型定义重构成一堆孤立的 task 函数。另外支持 YAML 指定自定义交错策略，给专家留了口子。

---

## 3. 生产结果

### 3.1 五个 workload（vs 阿里内部优化过的 Megatron fork）

| Workload | TopK/#Experts | 规模 | 集群 | Baseline MFU | Tessera MFU | Speedup |
|---|---|---|---|---|---|---|
| Qwen3-L | 8/128 | Large | 8,192 | 29.7% | 36.3% | +22.0% |
| **Qwen3-XL** | 8/128 | **Trillion** | 8,192 | 32.0% | **39.0%** | +21.8% |
| Qwen3-Next-M | 10/512 | Medium | 4,096 | 16.7% | 20.0% | +20.0% |
| Qwen3-Next-L | 10/512 | Trillion | 8,192 | 19.6% | 24.0% | +22.5% |
| Qwen3-Next-XL | 10/512 | Trillion | **12,288** | 15.9% | 21.1% | **+32.8%** |

注意 Qwen3-Next 系列**基线 MFU 只有 15.9–19.6%**——论文自己解释是新部署模型缺少成熟全栈优化（kernel 融合、通信合并），叠加极稀疏下的低计算强度和高集成开销。所以那个最亮眼的 +32.8% 是在一个相当低的基数上取得的。

**规划开销不随集群规模增长**：Tessera 为一个模型并行设备 mesh 生成一份执行计划，跨 DP rank 复制，扩容只增加计划实例数而非规划问题规模；DBO 也在每个 EP 组内独立运行，无跨 replica 协调。

### 3.2 分阶段上线（2025-04-05 ~ 04-19，8,192 GPU Qwen3-XL）

这张生产 trace 是全文最有说服力的消融：

- **04-09 静态规划器上线**：立刻 **+~13%** 吞吐。分解为负载均衡贡献 **~9%**（overlap-aware 切分配平了 post-overlap stage 代价）+ 延迟隐藏（掩盖 A2A）
- **04-16 DBO 启用**：MFU 抬到 **39.0%**，且 trace 明显变平——**动态注入降低了生产吞吐的波动性**

### 3.3 迭代时间分解：什么时候有效，什么时候没效

两个 Qwen3-Next 生产 workload 的对比揭示了 Tessera 的适用边界：

| | Qwen3-Next-L（8,192 GPU） | Qwen3-Next-M（5,120 GPU） |
|---|---|---|
| 稳态暴露 EP 通信 | **4.7%** 迭代时间 | **18.2%** |
| EP 通信被隐藏比例（含 warmup/cooldown） | **73%** | **仅 26%** |
| 总暴露 EP | 8.3% | **38.9%** |

两例中 overlap-aware 切分都把稳态 PP 气泡压到约 **3%**，说明配平是有效的。差异的根因是**计算强度**：Qwen3-Next-M 每专家参数更少，GEMM 时长相对 A2A 传输更短，每个 overlap pair 没有足够计算来藏住 A2A。

> **结论直白：Tessera 的 overlap 收益随 workload 的计算/通信比缩放。** 通信受限到一定程度，调度就救不回来了——这时需要的是通信本身变快或计算强度变高。

**离理想还差多少**（即使在有利的 Qwen3-Next-L 上）：仍有 27% 的 EP 通信暴露（819 ms），其中 **超 40% 来自 warmup/cooldown**；PP 气泡另占 8.5%（835 ms）= 残余 stage 不均 366 ms + interleaved 1F1B 固有的 warmup/cooldown 结构代价 469 ms。合计 **17% 迭代时间**是他们正在攻的下一个目标。

### 3.4 Bitwise 等价

> Tessera modifies only **when and where** tasks execute on the device timeline; the computation graph, operator semantics, and **reduction order are unchanged**. Gradient equivalence therefore holds **by construction**.

万亿规模 Qwen3-Next workload 上，确定性模式下开/关 Tessera 两次运行产生 **bit-identical 的 loss 轨迹**。已持续数月生产预训练无质量回归。

**这是一个值得抄的设计约束**：调度器只动时序、不动归约序，正确性就是构造性的，不需要靠实验证明。对比之下，任何改变 tile 归约顺序的融合 kernel 都得自己背 bitwise 的包袱。

---

## 4. 受控对比与消融

### 4.1 vs Megatron-Core MoE v0.16.1（256 GPU，公开 recipe）

| 模型 | Internal Megatron | Megatron-Core MoE | Tessera |
|---|---|---|---|
| Qwen3-235B | 31.6% | 32.4% | **40.1%**（vs MCore **1.24×**） |
| DeepSeek-V3 | 27.2% | 33.4% | 33.7%（**基本持平**） |
| Nemotron-3 Super | 24.6% | 26.3% | 27.7%（**基本持平**） |

**这张图比摘要诚实。** 摘要的「1.24× higher MFU than Megatron-Core」只在 Qwen3-235B 上成立；在非 Qwen 模型上 Tessera 只是「comparable」。论文自己也这么写（"achieves comparable MFU on non-Qwen models"）。

归因说得比较细：Qwen3-235B 上 MCore 的 A2A overlap 已经消掉大部分暴露 EP 通信但**没动 PP wait**；Tessera 相对内部基线砍掉 ~75% 暴露 EP 和 ~90% PP wait。DeepSeek-V3 上两者都藏了 >70% EP 通信、PP 也均衡，Tessera 只靠 DBO 把延迟的 Wgrad 和 PP send/recv 重叠，多赢一点点。Nemotron-3 Super 上 MCore 的 A2A-overlap 依赖手工的 Transformer 层分解且**最多支持一个 MTP 层**，超出这个模式的 EP 通信没被重叠；Tessera 减少 40% 暴露 EP 和 30% 流水气泡。

### 4.2 切分与 overlap 必须协同（§6.4）

Qwen3-Next-80B / 128 GPU + Qwen3-Next-M / 256 GPU，interleaved 1F1B、PP4-interleave2、EP32、序列 64K（开 CP）。用 **mock router 均匀分发 token 并关掉 DBO**，隔离静态计划效率。

结果很干净：**关掉 overlap 时，串行配平的 Partition A 端到端吞吐略优于 Tessera 的 Partition B；打开 overlap 后排序翻转，B 显著胜出。** 这是对「切分必须与 overlap 协同优化」最直接的证据。

### 4.3 启发式 vs ILP（§6.5）

Qwen3-Next-M @ 4K 序列，EP32/EP8 分别产生 690 / 536 个调度实例。四种策略：FIFO / +ALIGN / +DEFER（他们的）/ ILP（CBC，每实例 300 s 预算）。

- EP32 的 overlap 收益空间大于 EP8（EP 通信占 pair 串行代价份额更大）
- **+ALIGN 单独就吃掉了相对 FIFO 的大部分收益**；+DEFER 平均增益不大，但**部分实例超过 10%**
- 两条启发式合起来距 ILP：解析 makespan 上 EP32/EP8 差 **1.19% / 0.12%**，实测 post-overlap cost 上差 **1.07% / 0.76%**
- **求解时间**：CBC 跑完所有实例要**数小时到数天**，且长尾——很多实例耗尽 300 s 预算仍未证明最优。Tessera 的事件驱动调度器 **< 1 分钟**

**取舍结论：用 1% 的质量换 100× 以上的规划时间，生产默认取启发式。**

### 4.4 DBO 消融（Qwen3-Next-M，256 GPU，池上限 8/GPU）

| 指标 | Baseline 切分 | Overlap-Aware 切分 |
|---|---|---|
| 迭代时间（静态） | 8.28 s | 6.98 s |
| 迭代时间（+DBO） | **7.83 s（-5.4%）** | **6.67 s（-4.4%）** |
| 迭代时间（always-keep） | 8.34 s | 7.06 s |
| 峰值显存（静态） | 69.3% | 67.4% |
| 峰值显存（+DBO） | 72.9% | 70.0% |

`always-keep` 跑完整的监控 / 定尺 / 决策路径但**不真的延迟任务**，用来隔离 DBO 的稳态开销——只加约 **1%**，说明监控和注入路径本身很轻。

**DBO 在 baseline 切分上收益更大（5.4% vs 4.4%）是预期内的**：串行配平留下更大的 PP wait 供 DBO 填；overlap-aware 切分在静态期已经把这块吃掉了。**两个机制部分重叠，不是简单叠加。**

**显存代价 +2～4 个百分点**——延迟的 movable task 延长了其输入激活和梯度的生命期。

生产验证：8,192 GPU Qwen3-XL 上 DBO 激活后 MFU 达 39.0%；另一个 6,144 GPU 的 Qwen3-L 上 DBO 单独把 PP 气泡减少 **641 ms**，吞吐 +3.4%。

### 4.5 规划开销（§6.7）

8,192 GPU 目标配置，profiling 在 64 GPU 的 reference group 上跑。

- **重复的模型结构把不同 overlap pair 数量限死**：PP8-C4 下 237–327 个，PP8-C2 下 575–637 个（序列 4K–128K 全范围）
- profiling 时间随序列长度增长，峰值约 **3,050 s**（万亿模型、PP8-C2、128K 序列），**仍不到一小时**
- MILP：变量数 9,358–35,904，剪枝后 1,782–12,207，**求解时间 0.34–4.22 s**（64 核服务器）

---

## 5. 工程经验（§5）——本文最有价值的部分

这一节是 Operational Systems track 的精华，含金量高于方法本身。

### 5.1 intra-microbatch vs inter-microbatch overlap：他们否掉了融合路线

> While intra-microbatch overlap offers theoretical latency hiding, we found it **operationally fragile at scale**.

两条理由：

1. **算术强度**：沿序列维切分会碎化 GEMM，把算术强度压到「计算吞吐主导」的阈值之下
2. **模块化**：激进的 kernel 融合（点名 Comet）能挽回效率，但在他们的生产栈里，**MoE kernel 与特定通信后端的耦合会阻碍后端独立升级**

Tessera 的 inter-microbatch 方案则保持完整 GEMM 计算强度 + 后端模块化。

### 5.2 理论 overlap vs 硅片现实

- 跨 Qwen3-Next 预训练 workload，理论代价**多数情况下低估**实测重叠执行，**平均偏差约 5%**
- **主因之一：SM 竞争。EP 通信 kernel 为 A2A dispatch/combine 保留约 20 个 SM，与 Attention 或 MoE-MLP kernel 重叠时造成稳定的 10–20% 减速**
- 另一因：task 是**粗粒度调度单元而非单个 kernel**（通信主导的 task 里可能夹着小计算 kernel），残余干扰甚至串行化会阻止理想调度达成完美重叠

### 5.3 profiling 粒度：为什么不用便宜的方案

他们试过 **primitive 级 profiling**（profile 任务原语两两之间可复用的 overlap 模式，再组合估计整个 chunk-pair 时间线）。相比纯理论估计，它通过捕获 SM 竞争这类稳定的成对效应降低了平均误差，**但残余误差在不同 overlap pair 之间不均匀，尾部达 15%**。

> This tail behavior is **problematic for partition selection because the MILP relies on relative cost comparisons**; a few misestimated overlap pairs can change the selected bottleneck edge or partition boundary.

残余 gap 来自跨原语效应：连续 kernel 之间的接力重叠、cache 和通信库里的累积执行状态、host 侧 launch 开销。

**所以默认是昂贵的 chunk-pair 实测，primitive 级只作为低成本回退。** 这条对任何做自动并行/自动调优的人都是硬教训：**平均误差不是关键指标，尾部误差才是，因为下游是个比较排序问题。**

### 5.4 基础设施抖动

超过 10,000 GPU 后，网络拥塞和交换机竞争造成流水通信延迟的跨迭代、跨 rank 随机波动。他们**探索**（未系统验证）扩展 DBO 吸收抖动：利用抖动的短期时间相关性，用**最近 10 次迭代的移动平均**预测气泡大小，per-rank 注入本地池中的 movable task。架构上可行，因为基础设施气泡和路由气泡走同一条预测路径。**受控抖动条件下的系统验证是 future work。**

### 5.5 显存压力权衡

延迟 Wgrad 会延长其关联激活和梯度张量的生命期。**生产中激进的气泡填充偶尔触发 OOM。** 对策是有界池，按观测到的显存余量配置 per-GPU 上限。

值得注意的扩展想法：**显存吃紧时，Tessera 可以把内存管理操作（激活 offload、重计算）本身表示为 movable task**，调度进合适的气泡。这是个漂亮的统一——把内存优化和调度优化放进同一个框架。

### 5.6 用 padding 解耦系统约束

标准 interleaved 1F1B 理论上要求微批数 M 能被流水并行度 K 整除。但生产中 **M 由收敛所需的全局 batch size 决定，K 由显存限制决定**，强行 `M%K==0` 往往要做损害模型质量的 batch size 调整。

Tessera 注入 **shadow action**：跳过计算但执行必要的流水通信（发送伪张量）的空操作，满足结构依赖而不改变 batch 的数学等价性。shadow action 被排除在 loss 累积、梯度缩放、优化器计账之外。**开销约 0.5%。**

---

## 6. 批判

**论文自陈的局限**：收益随计算/通信比缩放（Qwen3-Next-M 只藏了 26% EP 通信）；基础设施抖动扩展未验证；显存压力；残余 17% 迭代时间（暴露 EP + PP 气泡）。

**未言明的方法论风险**：

1. **主要数字的基线是他们自己的内部 fork，外部不可复现。** 「+20%～33%」的分母是一个移动靶。真正可比的是 §6.3 的 256 GPU 实验，而那里在非 Qwen 模型上只是持平。摘要的措辞（"up to 1.24×"）技术上准确但选择性强。

2. **候选空间是「基线附近的有界邻域」，没有全局最优保证。** 整个方法的质量上限被初始串行配平基线的质量绑定。剪枝用 `T_base` 做上界这一点也印证了这个结构——它保证不劣于基线，但没说离最优有多远。这与 §6.5 里和 ILP 的对比不是一回事：那个对比只覆盖**给定 pair 内部的调度**，不覆盖**切分搜索空间**。

3. **MILP 目标是代理目标**（最大选中边的 post-overlap cost），不是真实迭代时间。论文承认它瞄准稳态。而 §6.2 的分解显示 warmup/cooldown 贡献了超过 40% 的暴露 EP 通信和 469 ms 的 PP 气泡——**恰恰是代理目标不覆盖的部分**。这大概是那个残余 17% 的结构性来源。

4. **需要一组配置匹配 TP/EP 拓扑的专用 reference device group 做真机 profiling。** 对阿里这是可承受的（64 GPU、<1 小时），对没有闲置集群的团队这是硬门槛。这也意味着换硬件、换互联、换通信库版本都可能要重跑。

5. **DBO 的收益依赖 Wgrad 池充裕。** 论文说生产中「typically well-provisioned」，但如果 Wgrad 已被融进 backward kernel（很多高性能实现会这么做），可移动池就会枯竭。这是个未讨论的前提条件。

6. **未开源**，且深度绑定内部 Megatron fork。

---

## 7. 对我们的意义

### 7.1 与 MonolithEP / UniEP 融合路线的正面对撞

这是必须直面的一条。**Tessera 在 10,000+ GPU 生产环境里评估过 Comet 式融合，然后否掉了。**

但要把两条理由分开看，它们的强度完全不同：

- **「切序列维碎化 GEMM、降低算术强度」——这条只打 intra-microbatch 的序列切分做法，不打 MonolithEP。** MonolithEP 的融合是 Dispatch+GroupGEMM / GroupGEMM+Combine 沿 token/expert 维度的 tile 级流水，不靠切序列维制造重叠机会。UniEP 的论文数字（vs COMET 1.03–1.38×）也说明融合路线自己也在解决这个问题。
- **「融合把 MoE kernel 和特定通信后端耦死，阻碍后端独立升级」——这条打得很准，而且是个运维论点，不是性能论点。** 它没法用 benchmark 反驳。对 ROCmoe 尤其相关：如果 MonolithEP 把 RCCL / IPC 路径写进 kernel，那么 RCCL 版本升级、换用 device API、或者迁到不同互联（XGMI vs PCIe vs 未来的 rack-scale）都可能要改 kernel。**这是我们应该主动设计防线的地方**——比如把通信原语抽象成 kernel 内可替换的 device-side 接口层，而不是直接内联特定后端的调用序列。

**两条路线的作用层次其实不同，不是二选一**：Tessera 在 PP/调度层做 inter-microbatch，MonolithEP 在 kernel 层做 intra-layer。理论上 Tessera 完全可以架在融合 kernel 之上——它的 task 抽象只要求 task 有时长和资源类型。真正的冲突只在「后端耦合」这个运维维度上。

### 7.2 20 SM 那个数字

> EP communication kernels reserve ~20 SMs for All-to-All dispatch and combine, causing a **stable 10–20% slowdown** when they overlap with Attention or MoE-MLP kernels.

这把「通信抢计算单元」从定性直觉变成了可用的量化常数，而且明确了它是**稳定的**（不是偶发抖动）。对我们的直接推论：

- 任何「通信与计算重叠」的收益估算，都必须先扣掉这 10–20%。理想模型算出的 overlap 收益如果小于 20%，实际很可能是**负的**。
- 这恰恰是 megakernel / 持久 kernel 路线的**优势论据**：融合 kernel 里 WG 角色分区是显式的，通信 WG 占多少 CU 是设计者决定的、可调的；而 Hopper 上 EP 通信 kernel 抢 20 个 SM 是不可控的外部事实。**Tessera 只能测量并接受这个惩罚，MonolithEP 可以调它。** 这是我们对 §7.1 那个反驳的最强反击。
- AMD 侧需要测出对应的常数：RCCL 的 A2A kernel 在 MI300/MI355 上占多少 CU、与 MFMA kernel 并发时的减速幅度。**这是一个明确的、低成本的可做实验，值得优先做。**

### 7.3 「只动时序不动归约序 ⇒ bitwise 等价 by construction」

值得作为设计原则写进 ROCmoe：**把「改变执行时序的优化」和「改变数值行为的优化」严格分层。** 前者可以激进（调度、重叠、气泡填充），因为正确性是构造性的；后者（融合 kernel 里的 tile 归约顺序、split-K、原子累加）必须单独背 bitwise 的验证负担。

UniEP 用「确定性映射保 bitwise」也是同一思路——但 Tessera 展示的是更强的版本：**如果优化根本不碰计算图和归约序，连验证都是可选的**（他们仍然验了，得到 bit-identical loss 轨迹）。

### 7.4 profiling 方法论：尾部误差杀死排序

**「解析模型平均误差 5%，primitive 级 profiling 尾部误差 15%，而 15% 的尾部会翻转 MILP 的相对比较」**——这是对任何自动调优/自动并行工作的硬约束。

对 FlyDSL / Primus 的 autotuner 直接适用：如果下游是**选择**（选 tile 配置、选切分、选调度），那么代价模型的**尾部准确性**比平均准确性重要得多。宁可少枚举一些候选然后真机实测，也不要靠一个平均很准但偶尔错 15% 的模型去排序大量候选。

### 7.5 `advance()` 探针 API + plan-agnostic FSM 引擎

对 Primus 的流水线运行时有直接借鉴价值。**在不重构模型定义的前提下把连续执行切成命名 task**，靠探针点 + 无锁状态机让外部引擎控制线程放行——这比要求用户把模型改写成一堆孤立 task 函数的方案友好得多，也是他们能作为 Megatron **插件**而非 fork 存在的原因。

配套的 `register_task()` 把 backbone 外任务显式登记给后台线程，也是干净的分工。

### 7.6 可直接借用的两个小机制

- **shadow action / 微批 padding**：用跳过计算但执行流水通信的空操作，解耦 `M % K == 0` 这个把系统约束传染给超参的限制，开销约 0.5%。这个问题在任何跑 interleaved 1F1B 的地方都存在。
- **把 offload / 重计算表示为 movable task**：显存管理和调度统一在一个框架里，气泡既可以填 Wgrad 也可以填 offload。比把内存优化做成独立 pass 更有表达力。

### 7.7 需要警惕的一条

Qwen3-Next-M 的对照（只藏住 26% EP 通信，总暴露 EP 38.9%）说明：**当每专家参数量小到 GEMM 时长压不住 A2A 时，调度层已经无能为力。** 极稀疏 MoE（TopK=10/512 这个方向）正在把工作负载往这个区域推。

这反而是 kernel 层融合和通信本身优化的机会窗口——**调度救不了的场景，正是 MonolithEP 和更快的 A2A 该出场的地方。** 值得把这条记成 ROCmoe 的定位论据：不要在 Tessera 已经赢的区域（计算强度高、调度能藏住通信）竞争，而要瞄准它明确报告失效的区域。

---

## 8. Glossary

- **Overlap pair** → 在同一 rank 上共享并发执行窗口的两个 chunk 操作，**带 pass 方向**（B 的 backward + D 的 forward ≠ B 的 forward + D 的 backward）
- **Post-overlap cost** → 一个 overlap pair 共同调度后的 makespan，介于 `max(T_c,T_d)` 与 `T_c+T_d` 之间；Tessera 的一切都围绕**实测**这个量
- **Backbone task** → 定义 stage 延迟的关键路径任务
- **Movable task** → 时序灵活、可延迟填气泡的任务（Wgrad、梯度归约、offload、重计算）
- **Reference device group** → 与目标 rank 同 TP/EP 拓扑的专用设备组，用于真机 profiling
- **Shadow action** → 跳过计算但执行流水通信的空操作，用于微批 padding
- **GDN（Gated DeltaNet）** → Qwen3-Next 使用的线性注意力变体，与 full softmax attention 按 3:1 混排

## 9. 未决问题

1. **Tessera 能不能架在融合 kernel 之上？** 它的 task 抽象只要求时长 + 资源类型，理论上兼容。但融合 kernel 内部的通信不再是可独立调度的 Comm 资源，overlap 调度器还有什么可调的？值得想清楚这两层的接口。
2. **AMD 侧的「20 SM」常数是多少？** RCCL A2A kernel 占用多少 CU、与 MFMA kernel 并发的减速幅度——低成本高价值实验。
3. **代理目标漏掉的 warmup/cooldown 怎么办？** 论文报告它贡献超 40% 的暴露 EP 和 469 ms PP 气泡，但 MILP 目标不覆盖它。是需要换目标函数，还是需要一个专门的 warmup/cooldown 优化？
4. **Wgrad 已融进 backward kernel 时 DBO 还剩什么？** movable 池的充裕性是 DBO 的隐含前提，这在高度优化的 kernel 栈上不一定成立。
5. 无代码。文中 Algorithm 1/2 是伪码级别，`SELECT` 的 gap-fit 打分函数和 DBO 的 `COMPUTESCORE` 都未给出具体形式。
