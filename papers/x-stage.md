# X-Stage: An Overlooked Pipeline Stage for Communication–Computation Overlap in DiT Inference

> [arXiv 2607.23264](https://arxiv.org/abs/2607.23264) (v1, 2026-07-25, cs.DC) · KlingAI Research + 清华 + NVIDIA · CC BY 4.0
> 硬件：单节点 8 GPU，"recent NVIDIA architecture"，**148 SM/GPU**、NVLink/NVSwitch 全连接、EP=8（148 SM 基本坐实是 B200）
> 软件：CUDA 13.1 · PyTorch 2.9 · **DeepGEMM @7f2a703** · FlashAttention @77aacb6
> 载体：DeepGEMM 的 **MegaMoE** persistent kernel + Ulysses 序列并行 attention

## TL;DR

**这篇是五篇里对 MegaMonolith 最直接可操作的一篇**——因为它的实验载体就是 MegaMoE，也就是 MegaMonolith 要移植到 CDNA 的那个 kernel。

核心发现：现有 fine-grained 融合 kernel 的抽象只描述「什么时候发通信」和「什么时候数据可消费」，**漏掉了「发出去之后、远端可见之前」这段软件可见的流水阶段**，作者命名为 **X-Stage**。它既是重叠机会（发完就能回去算），也是背压风险（持续注入会耗尽有限的 outstanding 容量，反压回 Tensor Core）。

用一个三参数的 **Burst–Gap 模型** `M_X = (T_iss⁰, R, Q)` 就能预测发射开销、恢复所需的 gap、以及背压拐点，**不需要按应用重新拟合**。据此重排 MegaMoE 的调度（跨 wave 交错 Linear-1 与 Linear-2），在 84 个配置上拿到几何均值 **1.18×**、最大 **1.62×**——**不改依赖、不改通信量，只改调度顺序**。

**对我们最扎心的一条**：MonolithEP 立论里那句「COMM_DISPATCH（100% XGMI 写）与 COMPUTE（HBM 读 + MFMA）几乎不抢资源 → free overlap 成立」，讲的是**资源类型不冲突**。这篇说，真正的约束不是资源类型，而是**写路径本身有限的排空速率 R 和 outstanding 容量 Q**。free overlap 只在 `V/(T_iss⁰+G) ≤ R` 时成立，超过就退化。

## 1. Problem

### 1.1 从一个 1.5× vs 1.56× 的对不上开始

MegaMoE 把 Dispatch → Linear-1 → 激活 → Linear-2 → Combine 融进一个 persistent kernel，用 warp specialization 分工，把本地专家按 **expert wave** 编组：一般是先跑完一个 wave 的 Linear-1，再跑它的激活和 Linear-2，然后进下一个 wave。Linear-2 的 epilogue 读累加器，做转换和地址计算，然后用 remote store 写进对称 Combine buffer。

作者用「completion-coupled 解释」（假设发起方在 store 变成远端可见之前不能继续本地工作）套 MegaMoE 公开的分阶段时间：

```
T_wave^cc ≈ max(T_Lin1, T_Act) + max(T_Lin2, T_Combine^cc)
T_Combine^cc = T_local_epilogue + T_RS_issue + T_post-issue_completion
```

算出来加速上限约 **1.5×**，但实现报告的是 **1.56×**。**模型解释不了实测**——这说明发起方在 store 被接受之后确实能继续干活，而请求还在往远端推进。

> 差距不大，作者自己也说这不构成机制证明，但它是一个很漂亮的切入点：从一个 0.06× 的模型误差挖出一整个被忽略的流水阶段。

### 1.2 为什么 DiT-MoE 特别容易撞上

长序列 + 细粒度专家 + 输入相关路由三件事叠加：

- 长序列放大发往热专家的 Combine 流量；
- 细粒度专家**缩短 Linear-2 的 MMA mainloop**，而这个 mainloop 正是两次 epilogue 突发之间的天然 gap；
- EPLB 能缓解放置不均，但消不掉每输入的路由倾斜。

结果是 expert-wave 调度下连续多个 Linear-2 epilogue 把 Combine 的 remote store 挤成长突发，gap 却被压短。

## 2. Method

### 2.1 X-Stage 的定义与两级解耦

作者把融合 GEMM 的常规描述 Load–Compute–Epilogue 拆出**两级解耦**：

| 级别 | 解耦对象 | 载体 | 容量 |
|---|---|---|---|
| 一级 | compute ↔ issue | 累加器 / staging buffer + warp specialization | 有限槽位 |
| 二级 | **issue ↔ completion** | **X-Stage** | 有限 outstanding 容量 Q |

关键点：二级解耦**是有限的，不是无界异步**。持续注入超过排空速率会累积请求 → 拉长后续发射 → 通过本地 staging 反压回 Tensor Core 生产者。加软件缓冲只能推迟传播，消不掉速率和容量约束。

### 2.2 Burst–Gap 模型（全文最有价值的部分）

三个可独立测量的量：

| 符号 | 含义 | 实测值 |
|---|---|---|
| `T_iss⁰(K,V)` | 无背压时的突发发射时间 | K=148、B=32 KB 时 ≈ **0.76 µs** |
| `R` | 有效聚合排空速率 | ≈ **717 GB/s** |
| `Q(K)` | 有效 outstanding 容量 | ≈ **4.25 MiB**（K=148） |

（`K` = 并发 remote-store 生产者数，`B` = 每生产者每突发字节数，`V = K·B`，`G` = 突发之间不发新 store 的有用时间。）

**稳态周期遵循 max 律，而不是 completion-coupled 的加法：**

```
T_period = max(T_iss⁰ + G, V/R)          ← 实测符合
T_period^cc = G + V/R                    ← 被证伪
```

由此推出全套关系：

```
T_iss(G) = max(T_iss⁰, V/R − G)                    发射时间
ΔT_iss   = [V/R − G − T_iss⁰]₊                     背压开销
G*       = [V/R − T_iss⁰]₊                         恢复所需最小 gap
q_peak⁰  = [V − R·T_iss⁰]₊                         无容量限制时的峰值 outstanding
T_iss^iso = max(T_iss⁰, (V − Q)/R)                 孤立突发的发射时间
```

标定方法很干净：
- `R` 用**零 gap 周期突发**逼到排空受限稳态后拟合，且不同 `K/B` 分解同一 `V` 得到相近周期 → 说明控制这个区间的是下游排空而非单生产者的发射吞吐；
- `T_iss⁰` 取大 gap 下的平台值；
- `Q` 用**孤立突发**扫 V 找拐点（K=148 时拐点在约 33 KB/生产者，聚合约 4.77 MiB），再用 `q_peak⁰` 反推。

标定一次后，后续 gap 和容量实验**不再重新拟合**。还设了 local-memory 对照组（同样的生产者数、store 宽度、地址生成、循环结构，只是写本地），确认 gap 敏感性来自远端路径而非指令开销——这个对照设计值得抄。

### 2.3 设计判据与两种动作

不产生发射侧背压的充要条件（式 10）：

```
V / (T_iss⁰ + G) ≤ R      长期注入速率界
[V − R·T_iss⁰]₊ ≤ Q       单次突发容量界
```

三步法：标定 `M_X` → 审计目标 kernel 的 `V` 与天然 `G` → 二选一：

- **判据不满足 → 重塑注入**：把有用计算重新分布到突发之间（MegaMoE 走这条）。
- **判据已满足 → 搭便车**：让拥有输出的角色发完短突发直接回去算，**不预留专职通信角色**（FlashAttention–A2A 走这条）。作者明确指出：*专职通信角色在发射后干等，并不能提高下游排空速率*。

### 2.4 MegaMoE：跨 wave 交错

不同 wave 的 Linear-1 之间没有神经网络层面的依赖，只要 Dispatch 数据和目标 buffer 就绪，后面 wave 的 Linear-1 原则上可以在前一个 wave 的 Combine remote store 完成前就开始。

交错调度器（Algorithm 1）维护 L1/L2 两个游标，用一个最小行超前量 `D` 约束 L1 领先 L2 多少行，在 L2 就绪且满足超前约束时发 L2，否则发 L1。**通信量、依赖、warp specialization、对称 buffer、epilogue、同步路径全部不变，只改就绪 tile 的发射顺序。**

**一条重要的理论结论**：在排空受限区间，`T_iss + G = T_period = V/R` 是固定的。

> 对固定的通信量和排空速率，**调度改变不了这个周期**；它只能决定周期里有多少是有用计算、多少是发射侧空等。

这句话把「调优空间在哪」界定得非常清楚。

## 3. Experiments

### 3.1 MegaMoE kernel 级

84 个配置 = 7 个模型形状 × {W4A8, W8A8} × {均衡, 倾斜路由} × 多个序列长度/专家配置。两个实现共享全部计算 kernel，只差调度顺序。

| 指标 | 值 |
|---|---|
| 几何均值加速 | **1.18×** |
| 中位数 | 1.17× |
| 最大 | **1.62×** |
| 最差 | **0.94×**（回退） |

- **倾斜路由收益更大**：热专家上连续的 Linear-2 epilogue 串更长，交错进来的 Linear-1 提供的排空窗口更值钱。
- **回退案例值得警惕**：部分「均衡路由 + 大专家权重工作集」的配置掉到 0.94×，profiling 显示**跨专家交错降低了专家权重的 L2 局部性**，抵消了通信侧收益。作者的结论是 X-Stage 感知调度必须**同时平衡排空与数据局部性**。

### 3.2 机制验证（三层递进，方法论很扎实）

1. **Tensor Core 时间线**：wave 调度呈现相位化执行——连续 Linear-2 epilogue 注入后 Tensor Core 活跃度下降或拖长尾；交错后低利用率区间变短。（建立相关性）
2. **每 tile remote-store span**：epilogue 里用 `clock64()` 量发射侧跨度 `T_RS`。倾斜 W8A8 下 7 个形状的 wave 中位数 **7.9–9.9 µs 且有明显长尾**，交错后全部收窄到 **3.6–3.9 µs**；其中 3 个形状另做了「改写本地 HBM」的对照，交错后的远端 store 中位数与本地 store 对照**相差 0.5 µs 以内**——即已经落到无背压地板。
3. **映射到计算停顿**：MMA warpgroup 独立记录 tile 入口等 staging 槽位（`tmem_empty`）的时间。

| 模型 | Span wave/交错 (µs) | T_mma L2/L1 (µs) | Δt wave 预测/实测 (µs) |
|---|---|---|---|
| DiT-MoE | 9.93 / 3.77 | 5.30 / 9.77 | 4.63 / 4.18 |
| Qwen3.5 | 9.17 / 3.82 | 4.24 / 14.40 | 4.94 / 4.31 |
| Hy3 | 9.68 / 3.59 | 6.09 / 10.79 | 3.60 / 3.08 |
| MiMo-V2.5 | 9.22 / 3.94 | 7.77 / 7.57 | 1.45 / 0.71 |
| GLM-5.2 | 9.53 / 3.63 | 7.99 / 18.77 | 1.54 / 0.92 |
| DSv4-Flash | 9.48 / 3.83 | 8.01 / 7.40 | 1.47 / 0.65 |
| **DSv4-Pro** | 7.88 / 3.55 | **9.54** / 11.57 | **0 / 0.06** |

模型一致**高估** 0.45–0.82 µs，但保持了排序和量级。**DSv4-Pro 是一个漂亮的负对照**：它的 Linear-2 mainloop 9.54 µs 本来就盖得住 7.88 µs 的 wave span，所以预测和实测都是几乎零停顿，kernel 级收益也相应很小——这条反向验证了因果链。

### 3.3 FlashAttention–A2A（Ulysses 序列并行）

把 attention 后的 All-to-All 在 tile 粒度融进 FlashAttention：拥有输出 tile 的角色发对应的 remote store 然后立刻做下一个 tile，**下一个 Q-loop 就是排空窗口**。不占专职通信 warp，也不占专职 SM。

FA3 + A2A（发射侧可见时间，µs）：

| M | 8,192 | 16,384 | 32,768 | 49,152 | 65,536 |
|---|---|---|---|---|---|
| FA3 only | 195.2 | 785.5 | 3,236.7 | 7,275.6 | 13,249.4 |
| Serial | 295.6 | 968.3 | 3,579.6 | 7,767.9 | 13,936.6 |
| Fused | 207.0 | 795.5 | 3,241.5 | 7,285.7 | **13,218.3** |
| 隐藏率 | 88% | 95% | 99% | 98% | ~100% |
| 加速 | **1.428×** | 1.217× | 1.104× | 1.066× | 1.054× |

FA4 上最大 1.42×。序列越长、Q-loop 提供的 gap 越大，融合后的稳态时间**越逼近 FA-only**——即通信被完全吃掉。

注意作者对指标的谨慎定义：`E_res = T_fused − T_FA` 只表示发射侧残余开销接近零，**不断言远端可见完成**；隐藏率的分母用的是串行组合的实测增量而非独立计时的 A2A kernel（因为 cache 状态、同步、launch 开销在不同组合下不同）。这种自我约束在系统论文里不常见，加分。

## 4. Limitations

**作者声明的：**

- 微基准只研究**单边 remote store**；load、atomic、有不同 progress engine 的集合通信、**跨节点网络**可能表现出完全不同的约束。
- 模型只预测**发射侧背压**，不涵盖同步、接收侧拥塞、launch/调度开销。
- 参数需按平台重标定：拓扑、路由、store 宽度、生产者数、内存放置、GPU 代际变化都可能让当前的流体近似失效。
- 排空视界 `V/R` **不替代内存序语义**，X-Stage 感知的 kernel 仍要保留 fence、signal、buffer 生命周期规则和消费侧就绪检查。

**我认为需要打问号的：**

- 只在**一台 8 卡机、一个架构**上标定，`R = 717 GB/s`、`Q = 4.25 MiB` 这两个数字的可迁移性完全未知。论文自己也承认，但整篇的说服力其实很依赖这两个数在别处也稳定。
- 每 tile 停顿模型系统性高估 0.45–0.82 µs，作者归因于「单窗口模型」的近似误差，但没有给修正。
- 84 个配置里回退到 0.94× 的那些，论文只做了定性归因（L2 局部性），**没有给出「什么时候该用 wave、什么时候该用交错」的判别式**——而这恰恰是要落地必须回答的。
- 全部是**推理**场景（DiT / MoE 前向），训练的反向传播里 Combine 的对偶操作有不同的依赖结构，没有讨论。

## 5. Our take

### 5.1 直接冲击 MonolithEP 的「free overlap」立论

`rocmoe_DESIGN.md` P1 写的是：

> overlap 不来自"起几条 stream"，而来自在一个执行单元内把不同资源（MFMA / HBM / XGMI）分配给不同 WG 角色。参考 MonolithEP 的资源 contention 矩阵：COMM_DISPATCH（100% XGMI 写）与 COMPUTE（HBM 读 + MFMA）几乎不抢资源 → free overlap 成立。

X-Stage 说的是：**资源类型不冲突 ≠ 可以无限重叠**。写路径有自己的排空速率 `R` 和 outstanding 容量 `Q`，一旦 `V/(T_iss⁰+G) > R`，背压会从发射侧一路传回 MFMA 生产者——**这时候 COMM 和 COMPUTE 就冲突了，只不过冲突点不在「资源类型」而在「流控信用」**。

这不是推翻我们的设计，而是给它加了一个**定量的成立条件**。应该把 P1 从定性论断改写成带判据的形式。

### 5.2 必须做的实验：在 MI355X 上标定我们自己的 `M_X`

论文的微基准设计是可以直接照搬的（周期突发 + 孤立突发 + 本地 store 对照），我们要测的是 **XGMI remote store / HIP IPC peer write** 的三个参数：

| 参数 | 论文（NVLink，B200 8 卡） | 我们要测（XGMI，MI355X 8 卡） |
|---|---|---|
| `R` | 717 GB/s | ? — 注意 `rocmoe_DESIGN` 记的聚合 XGMI 是 ~525 GB/s，**如果 R 更低，我们比他们更早撞界** |
| `T_iss⁰` | 0.76 µs @ K=148, B=32 KB | ? |
| `Q(K)` | 4.25 MiB @ K=148 | ? |

**生产者配置的差异是个大问题**：论文用 `K=148`，正好是每 SM 一个生产者。MonolithEP 是 `COMPUTE:COMM_DISPATCH:TAIL_COMBINE = 224:16:16`，也就是 **K=16，每个生产者的 B 大得多**。论文明确说 `Q` 是 `K` 的函数，所以我们不能直接用 4.25 MiB，**必须自己扫出 Q(K) 曲线**。而且 K=16 意味着单生产者的发射吞吐更可能成为瓶颈（论文里 K 大时是下游排空主导），我们可能落在他们没覆盖的区间。

### 5.3 最可能白捡的 1.18×：检查我们的 chunk 流水是否已经交错

论文的收益来源是「把后面 wave 的 Linear-1 塞到前面 wave 的 Linear-2 突发之间」。对照到 MonolithEP 的 chunk pipeline，等价问题是：

> **chunk `i+1` 的 FC1 有没有被排到 chunk `i` 的 combine 写之间？还是我们也是「一个 chunk 走完再下一个」？**

如果是后者，那这 1.18× 几何均值（倾斜路由下更高）就是白放着的。这是**下一步最高性价比的实验**，因为它不改依赖、不改通信量、不改 WG 角色比例，只改就绪任务的发射顺序。

### 5.4 一个真实的矛盾：X-Stage 交错 vs Fleet 的 XCD 局部性

这篇发现跨专家交错会**降低专家权重的 L2 局部性**，导致 0.94× 回退。而 [Fleet](./fleet.md) 在 MI350 上的结论恰恰相反——它的收益就来自**把同一 XCD 上的 worker 绑到同一个 GEMM/expert，让权重留在 4 MB L2 里复用**（MoE 上低 batch 相对 vLLM 达 3.16×）。

**在 CDNA 上这两个方向会正面冲突**，而且 AMD 的 L2 是**每 XCD 4 MB、8 个 XCD 分立**，比 NVIDIA 的统一 L2 更碎，交错的局部性代价只会更大。这意味着：

- 不能无脑照抄 MegaMoE 的交错调度；
- 正确的做法可能是**在 XCD 内交错**（保住 L2 局部性）而不是跨 XCD 交错——即把 X-Stage 的 gap 供给约束和 Fleet 的 chiplet 绑定约束一起解；
- 这是一个我们独有的、论文里没有的设计点，**值得写进 ROCmoe 设计文档作为差异化论据**。

### 5.5 对 ROCmoe 设计文档的具体修改建议

- **P1 改写**：从「资源类型不抢 → free overlap」改成「free overlap 在 `V/(T_iss⁰+G) ≤ R` 且 `[V − R·T_iss⁰]₊ ≤ Q` 时成立；`(T_iss⁰, R, Q)` 由平台标定」。
- **P4（静态调度 + 显式资源预算）补一条**：预算不只是 CU 数和 fence 次数，还要包含**突发体积 `V` 与 gap `G`**。WG 角色比例决定 `K`，进而决定 `Q(K)`——这三者不能分开定。
- **P5（roofline 驱动）扩展**：现在有两个互补的通信成本模型可以进工具箱——[Perseus](./perseus.md) 的 α-β 模型管**跨节点固定开销**，X-Stage 的 Burst–Gap 模型管**节点内发射侧背压**。合起来才是完整的两级通信记账。

### 5.6 与已读论文的关系

- **[Perseus](./perseus.md)**：互补的另一半。Perseus 管跨节点 proxy fence，X-Stage 管节点内写路径背压。两篇都指向同一个元结论——**megakernel 的瓶颈已经从「算得快不快」转移到「通信路径的流控」**。
- **[Fleet](./fleet.md)**：见 §5.4，在 CDNA 上与本文构成设计张力。
- **[Comet](./comet.md) / [UniEP](./uniep/README.md)**：都是 thread-block 专用化 + 单 kernel 融合，属于本文 §7.2 归类的「描述了哪些 tile 可以并发、谁发起、何时可消费」那一类——正是本文说「不足以预测发射方行为」的对象。
- **MegaMOE（DeepGEMM）**：`~/workspace/MegaMOE` 就是本文改的那份代码（commit 7f2a703 附近）。可以直接在本地对照 expert-wave 调度器的实现，看交错版本要改哪里。

## 6. 延伸阅读

1. **DeepGEMM MegaMoE 源码**（本地 `~/workspace/MegaMOE`）——本文的 baseline 与改动点，配合读效率最高。
2. **ParallelKittens / TileLink / FLUX**——本文 §7.2 列的细粒度融合前作，界定了「X-Stage 之前大家在建模什么」。
3. **Ulysses 序列并行**（Jacobs et al., 2023）——FA–A2A 那半部分的背景。

## 参考

- 论文：<https://arxiv.org/abs/2607.23264>
- 本次检索的完整清单：[`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md)
