# Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects

> [arXiv 2511.02132](https://arxiv.org/abs/2511.02132)（2025-11-04）· ICML 投稿格式（双盲，正文署 Anonymous）
> 通讯作者脚注已暴露身份：**Mansi Choudhary（Duke ECE，实习于 AMD）· Karthik Sangaiah（AMD）**；单位为 Duke ECE / AMD / Duke CS
> 硬件：**MI300X**（仅此一款）· 实现：**Triton** · 剖析：ROCProfiler v3（聚合各 XCD 的 L2 命中率）
> 无开源代码链接。

> 本篇是精读 + 延伸讨论。原文为 arXiv 标准许可（非 CC0），故不做全文翻译，只摘引关键数据。
> 关联：[`hipkittens.md`](./hipkittens.md)（§5 是 GEMM 版的 chiplet swizzle）· [`fleet.md`](./fleet.md)（MI350 上的 chiplet megakernel + 协作 L2 tiling）

## TL;DR

**一句话：把 FlashAttention2 的 workgroup 网格重排,让同一个 attention head 的所有 Q-block 落在同一个 XCD 上,靠 per-XCD 私有 L2 复用 K/V,MI300X 上最高快 50%。**

核心洞察是一个很朴素的观察:FA2 里同一个 head 的所有 Q-block 都要读**完整的 K 和 V**。而 MI300X 的 L2 是**每 XCD 私有 4MB**(全卡 32MB),硬件默认又是 **chunk size = 1 的轮询派发**——相邻 workgroup 被丢到不同 XCD。结果是同一份 K/V 被 8 个 XCD 各自从 HBM 拉一遍,每个 XCD 的 L2 里塞的都是别人也需要的数据的副本,等效缓存容量被切成八份。

论文把"共享同一份 K/V 的那组 workgroup"命名为 **ACC(Attention Compute Cluster)**——MHA 下一个 head 一个 ACC,GQA 下一个 KV 组一个 ACC——然后主张:**让一个 XCD 一次只服务一个 ACC**。

**最刺眼的数字不是 50%,是 L2 命中率。** 在 H_Q=128 / N_CTX=128K 的极端配置下:

| 映射策略 | L2 命中率 | 谁在用 |
|---|---|---|
| **Swizzled Head-first(本文)** | **90–96%** | — |
| Naive Head-first | 40–60% | Triton 默认 FA kernel |
| Naive Block-first | **≈1%** | 基线 |
| Swizzled Block-first | **≈1%** | **AMD AITER** |

**AITER 当前部署的方案在这个配置下几乎全 miss。** 这是全文最有价值的一条:一个在中等规模下工作良好的 swizzle,在头数多 + 序列长的场景会彻底崩掉,而且崩得没有预警。

**代价是几乎为零**——就是 workgroup ID 的一次双射重映射,十来行代码,输出逐位一致。

**但要注意三条边界**(论文自己没充分讨论,见 §6):**causal 下会亏**、**head 数必须整除 XCD 数**、**backward 只有 1.10×**。

---

## 1. 问题:chiplet 把"缓存"这个概念变了

论文开篇铺的是一条架构演进线:单 die 统一 L2(A100/H100/MI200)→ 双 die(Blackwell/Rubin)→ 四 die 及以上(Rubin Ultra、MI300 系列)。

这里有个论文明确点出、但很多人没意识到的差别:

> NVIDIA 的 Blackwell 在两个 die 之间**维持完整的缓存一致性,把 NUMA 效应在硬件层面抽象掉了**;而 AMD 的 MI300X **把 NUMA 特性显式暴露给软件**。

也就是说,这类优化在 NVIDIA 上你想做也做不了(硬件替你做了,好坏都是它),在 AMD 上你不做就是亏。**这是 AMD 侧特有的一块免费性能**,和 HipKittens §5 的判断是同一个方向。

MI300X 的相关规格:

| 项 | 值 |
|---|---|
| XCD 数 | **8** |
| 每 XCD 的 CU | 38(全卡 304) |
| L1 / CU | 16 KB |
| **L2 / XCD** | **4 MB(全卡 32 MB,私有不共享)** |
| HBM3 | 192 GB / 5.3 TB/s |

关键是 L2 **私有**。跨 XCD 的数据共享只能落到 GPU 级的 LLC(Infinity Cache)或 HBM,延迟和带宽都是另一个量级。

### 硬件默认调度为什么反而有害

现代多 die GPU 用**分块轮询**把 workgroup 派到各个 die,而**当前硬件的块大小是 1**。这个选择本身是合理的——保证负载均衡、把各 HBM stack 的聚合带宽吃满。但副作用是:逻辑上相邻、共享数据的 workgroup 被系统性地拆到不同 die。

论文提醒了一句很实际的话:**这个策略实现在驱动里,会随 GPU 代际变化,程序员不能依赖固定行为**,所以修正逻辑必须写在 kernel 里、运行时自适应。这是对"要不要把 XCD 数写死在 kernel 里"的一个直接回答——写死是当前的现实做法,但要意识到它是在对抗一个可能变化的驱动行为。

前置工作:GEMM 上做同类空间感知映射,L2 命中率从 **43% → 92%**(AMD Tensile)。本文相当于把这个思路搬到 attention。

---

## 2. ACC:本文真正的概念贡献

抛开具体的 swizzle 公式,这篇论文留下来的东西是 **ACC(Attention Compute Cluster)** 这个抽象:

> **在 forward 或 backward 中共享同一组输入张量的那批 workgroup,构成一个 ACC。**

- **MHA**:每个 head 有独立的 K/V → **一个 head = 一个 ACC**
- **GQA**:多个 query head 共享一组 K/V → **一个 KV 组 = 一个 ACC**(ACC 更大)

优化目标随之变得可陈述:**把一个 ACC 整体放进一个 XCD,并且让这个 XCD 同一时刻只服务一个 ACC。**

两个收益机制被分开说清楚了:
1. ACC 的共享数据只进**一个** XCD 的 L2,被组内多个 workgroup 复用——避免缓存碎片化;
2. 每份共享张量**每卡只从 HBM 取一次**,而非被多个 XCD 各取一遍——直接砍 HBM 流量。

backward 也成立:计算 dQ/dK/dV 的 workgroup 同样共享该 head 的 Q、K、V、dO。

> **延伸**:ACC 这个抽象的价值超出 attention。任何"一组 CTA 共享一份只读大张量"的算子都适用——MoE 的 grouped GEMM 里,同一个 expert 的所有 token block 共享该 expert 的权重,这就是一个天然的 ACC。ROCmoe 里值得直接套用(见 §9)。

---

## 3. 四种映射策略

论文的分类框架是两个正交维度:**迭代顺序**(block-first vs head-first)× **是否 swizzle**。这个 2×2 拆得很干净,是本文组织实验的骨架。

| 策略 | 迭代顺序 | XCD 分配 | ACC 是否被切开 | 谁在用 |
|---|---|---|---|---|
| Naive Block-first | 块外层、头内层 | 轮询 | **切开** | 基线 |
| Swizzled Block-first | 块外层 + GQA 感知 swizzle | 组内聚拢 | 仅当 **GQA 组数 == XCD 数** 时不切 | **AMD AITER** |
| Naive Head-first | 头外层、块内层 | 轮询 | **切开**(每个 head 被条纹化到所有 XCD) | **Triton 默认 FA kernel** |
| **Swizzled Head-first** | 头外层 + 空间 swizzle | 整个 head 锁一个 XCD | **不切** | 本文 |

几点值得注意:

**Swizzled Block-first 的成立条件极其脆弱。** 它只在 GQA 组数恰好等于 XCD 数时保住局部性。MI300X 是 8 个 XCD,而 Llama-3 全系列恰好是 8 个 KV head——**AITER 的方案在 Llama 上好用是个巧合**。换成 MHA(比如 DeepSeek-V3 prefill 的 128 个 head),同一个 XCD 会被迫同时服务多个 ACC,L2 立刻被切碎。

**Naive Head-first 是个有意思的中间态。** 它虽然把每个 head 条纹化到了所有 XCD,但由于迭代顺序是头优先,**同一时刻全卡在处理的都是同一个 head 的数据**——每个 XCD 的 L2 里存的是同一份 K/V 的副本。副本浪费容量,但至少是命中的。所以它在多数配置下能逼近最优,只在序列极长、单份 K/V 塞不下 4MB 时才崩(降到 ~90%,L2 命中 40–60%)。这个解释是论文自己给的,也和数据吻合。

---

## 4. Swizzled Head-first 的实现

grid 仍然是一维的 `batch × num_q_heads × ceil(seqlen_q / BLOCK_M)`,在 kernel 里把线性 workgroup ID 拆成 (batch, head, block) 三个偏移。核心是让 **head 成为慢轴**:

```python
wid = tl.program_id(0)
wid_per_batch  = wid // BATCH
heads_per_xcd  = NUM_Q_HEADS // NUM_XCD
blocks_per_head = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
chunk_size     = NUM_XCD * blocks_per_head

head_offset  = ((wid_per_batch % NUM_XCD) * heads_per_xcd
                + wid_per_batch // (NUM_XCD * blocks_per_head))
block_offset = (wid_per_batch % chunk_size) // NUM_XCD
batch_offset = (wid // (blocks_per_head * NUM_Q_HEADS)) % BATCH
```

读法:`wid % NUM_XCD` 是硬件轮询决定的**目标 XCD**,乘上 `heads_per_xcd` 就把这个 XCD 分到的 head 段选出来;`wid // (NUM_XCD * blocks_per_head)` 在段内选具体 head。于是"硬件轮询"这个原本破坏局部性的行为,被反过来当成了 XCD 的选择器。

变换是**双射**的,所以输出逐位一致,不引入任何数值变化。这是它能被无风险接受的关键——不是近似优化。

论文对此的评价是"改动量极小却非常有效",这点确实成立:十来行整数运算,不碰计算主体。

---

## 5. 实验结果

**所有性能数字都归一化到 Swizzled Head-first,论文全程没有报告任何绝对 TFLOPS。** 这是最大的方法论问题,见 §6。

### 5.1 MHA 敏感性(主实验)

配置:H_Q = H_K ∈ {8,16,32,64,128},N_CTX ∈ {8K,32K,128K},batch ∈ {1,2,4,8},D_HEAD=128,BLOCK_M×BLOCK_N = 128×64。

- **头数少时四种策略几乎无差别**——优化只在规模上来后才有意义。
- **H_Q ≥ 64 且 N_CTX ≥ 32K**:block-first 系列只有 **64–70%** 的效率。
- **H_Q=128 / N_CTX=128K**:本文比 block-first **快最多 50%**。
- Naive Head-first 多数情况下贴近最优,极长序列时掉到 **~90%**。

L2 命中率解释了一切(见 TL;DR 的表)。头数少、序列短时四者都在 ~90%;规模一上来就急剧分化,block-first 直接崩到 **≈1%**。论文的措辞是"catastrophic cache degradation",不算夸张。

### 5.2 GQA(Llama-3 全系)

固定 8 个 KV head,H_Q ∈ {32,64,128},对应 Llama-3 的 8B / 70B / 405B。

- **两种 swizzled 方案表现相当**——因为 8 个 KV head 正好等于 8 个 XCD,Swizzled Block-first 恰好不切 ACC。
- Naive Block-first 在高 H_Q / 长序列下显著退化。
- Naive Head-first 在 **90–95%** 波动。

**结论要读反过来:GQA 上本文没有优势,因为竞争对手撞上了一个巧合。** 真正的价值在 MHA。

### 5.3 DeepSeek-V3 prefill 案例(MHA,128 head)

H_Q = H_K = 128,D_HEAD = **56**。这是 head 数(128)远超 XCD 数(8)的典型场景。

- 128K token 时,Naive Block-first 掉到 **<65%**;
- Swizzled Block-first(即 AITER 方案)在 128K / batch=8 时掉到 **76%**;
- D_HEAD=56 导致算术强度偏低,四种方案的绝对性能都不高。

### 5.4 Backward(AITER 的 FA2 bwd kernel)

H_Q=128,ctx ∈ {8K,32K,128K},batch ∈ {1,2}。相对 Naive Block-first 的加速:

| 策略 | 128K 时 |
|---|---|
| **Swizzled Head-first** | **1.10×** |
| Swizzled Block-first | 0.94× |
| Naive Block-first / Naive Head-first | 0.91× |

**backward 的收益(1.10×)远小于 forward(最高 1.5×)。** 论文的解释是 backward 有更多标量运算和额外复杂度,"怀疑本优化引入了新的瓶颈",明确留给未来工作。这是诚实的表述,但也说明**该技术主要是推理 prefill 的优化,对训练的价值有限**。

---

## 6. 局限与质疑

论文自己承认的只有 backward 收益有限一条。以下是我的补充:

**① 全程没有绝对性能数字,无法判断基线是否有竞争力。**
forward 全部是作者自己写的 Triton 实现,四种策略互相比。如果这套 Triton FA2 的绝对性能本身就落后于 CK / AITER 的手工 kernel,那"比自己的 block-first 版本快 50%"的说服力要打折。对照 HipKittens 论文里的数字——Triton 在 AMD 上有寄存器生命期跟踪问题、访存降不到最优 intrinsic——这个担心是有依据的。**唯一用了外部 kernel 的是 backward(AITER),而那恰好是收益最小的一组(1.10×)**,这个巧合值得警惕。

**② causal 完全没有讨论,而这是个真问题。**
论文所有实验都是非 causal 的,正文里连 causal 二字都没出现。但 causal mask 下第 i 个 Q-block 的工作量正比于 i,把 q_block 变成快轴会**把不等量的工作聚成堆**,直接引入负载不均。AMD 自己在 FlyDSL 里的实现证实了这一点(见 §7)——**causal 下实测亏 7%**,所以直接禁用。论文漏掉的这条边界,在落地时是硬约束。

**③ swizzle 公式要求 head 数整除 XCD 数。**
`heads_per_xcd = NUM_Q_HEADS // NUM_XCD` 是整除,不整除时映射不再均衡。常见模型(32/64/128 head vs 8 XCD)都满足,但不是通例。

**④ 只测了 MI300X(CDNA3)。**
MI350/MI355(CDNA4)同样是 8 个 XCD 但每 XCD 32 个 CU、缓存层次也有变化,论文没有验证。结论能不能平移到 CDNA4 是开放的。

**⑤ 只覆盖 prefill / 训练形状,decode 未涉及。**
decode 阶段 Q 只有一行,ACC 的结构完全不同(此时是 KV cache 的复用),本文框架不直接适用。这块是 [`fleet.md`](./fleet.md) 在做的事。

**⑥ 对硬件调度块大小 = 1 的依赖。**
论文自己提醒了这点,但整套 swizzle 恰恰是建立在"轮询且块大小为 1"之上的——`wid % NUM_XCD` 才等于 XCD 编号。驱动一改,公式即失效且**不会报错,只会静默变慢**。

---

## 7. 落地情况:AMD 已经把它实现进 FlyDSL

这篇论文不是纸面工作。AMD 官方的 MLIR kernel DSL [FlyDSL](https://github.com/ROCm/FlyDSL) 已经在 gfx950 的 flash attention 里实现了它,并在注释中直接引用了本文:

`FlyDSL/kernels/attention/flash_attn_utils.py:915-935`：

```python
def _init_dualwave_thread_mapping(ctx):
    # Swizzled Head-first Mapping (arXiv:2511.02132): the grid is head-fast, so one
    # head's q-blocks scatter across all XCDs and each re-streams its K/V. Re-derive
    # (head, q_block) with head as the slow axis to keep them on one XCD. Bijective,
    # so output is bit-identical; split-K's third grid axis would not survive it.
    # Non-causal only: under a causal mask q-block i does work proportional to i, so
    # making q_block the fast axis clusters unequal work and costs 7% (measured).
    if const_expr(
        traits.XCD_SWIZZLE and not traits.SPLITK and not traits.CAUSAL
        and traits.NUM_HEADS_Q % NUM_XCD_GFX950 == 0
    ):
        num_q_blocks = fx.Index(gpu.grid_dim.y)
        linear_wg = fx.Index(gpu.block_idx.x) + fx.Index(gpu.block_idx.y) * fx.Index(traits.NUM_HEADS_Q)
        ctx.h_idx = linear_wg // num_q_blocks
        ctx.q_block_idx = linear_wg % num_q_blocks
```

**这段注释的信息量比论文本身还大**,它记录了三条论文没写的工程约束:

1. **causal 下亏 7%(实测)** —— 论文完全没提的负面结果,AMD 实测出来了,所以直接在编译期禁用。
2. **split-K 不兼容** —— 第三个 grid 轴过不了这个双射变换。
3. **门槛条件**:`NUM_XCD_GFX950 = 8`、`MIN_Q_BLOCKS_XCD_SWIZZLE = 64`(`flash_attn_utils.py:35-36`)——**Q-block 数少于 64 时不启用**,因为块太少时根本填不满 8 个 XCD,swizzle 没有意义反而增加索引开销。

另外注意 FlyDSL 的实现**比论文的公式简单得多**:它不算 `heads_per_xcd` / `chunk_size`,而是直接把二维 grid 线性化后转置(`linear_wg // num_q_blocks` 取 head、`% num_q_blocks` 取 block)。等价效果,但可读性和指令数都更好。**要抄的话抄 FlyDSL 这版,不要抄论文的 Figure 11。**

`XCD_SWIZZLE` 在 FlyDSL 里是一个可调 trait,测试中暴露成 CLI flag——说明 AMD 自己也把它当成"默认关、按场景开"的优化,而非无条件启用。

---

## 8. 与 HipKittens / Fleet 的关系

三篇是同一主题的不同切面,建议放在一起读:

| | 对象 | 手段 | 收益 |
|---|---|---|---|
| **本文** | attention(prefill/train) | grid 重排,head 锁 XCD | 最高 **1.5×**(fwd)/ 1.10×(bwd) |
| [`hipkittens.md`](./hipkittens.md) §5 | GEMM | grid 重排,输出 tile 分块锁 XCD | **+19%** |
| [`fleet.md`](./fleet.md) | decode megakernel | chiplet-task 抽象 + 协作 L2 tiling | **1.3–1.5×** vs vLLM eager |

**共同结论:MI300/MI350 的 XCD 私有 L2 必须显式建模,否则默认调度会系统性地浪费缓存。** 三篇独立地在 GEMM、attention、decode 三个场景验证了同一件事,这个结论现在应该当成 AMD 上的常识而非技巧。

HipKittens 里还有一个本文没有的细节:默认 row-major 在"输出 tile 宽度与 XCD 数互质"时会踩到最坏情况。本文的 head-first 映射本质上是同一个陷阱在 attention 里的形态——**`wid % NUM_XCD` 这个模运算是所有 chiplet swizzle 的公共骨架**,HipKittens 的 `chiplet_transform_chunked` 和本文的 Figure 11 是它的两个特例。

---

## 9. 对我们的影响

**① ROCmoe:ACC 抽象可以直接搬到 grouped GEMM。**
MoE 的 expert GEMM 里,同一个 expert 的所有 token block 共享该 expert 的权重矩阵——这就是一个标准 ACC。当前如果用默认 grid,同一个 expert 的权重会被多个 XCD 各拉一遍。**把 expert 作为慢轴、锁到单个 XCD**,是和本文完全同构的优化,而且 MoE 的权重比 attention 的 K/V 更大、复用价值更高。需要注意的是 expert 负载不均(和 causal 的问题同构),要先确认不会因为聚拢而放大不均衡。

**② MonolithEP:持久化 kernel 里这个优化要重新表达。**
本文的手段是重排 grid,但持久化 megakernel 没有 grid 可重排——workgroup 常驻,工作靠队列分发。等价做法是**在工作分发时按 XCD 亲和性分配任务**,即让同一个 XCD 上的常驻 WG 优先领取共享同一份数据的任务。这比重排 grid 复杂,但收益机制相同。FlyDSL 的 MegaMoE 用原子 ticket 分角色,可以参考它怎么把 XCD 感知揉进队列。

**③ 如果我们在跑 AITER 的 attention,值得实测一下。**
论文的数据指向 AITER 当前的 Swizzled Block-first 在 **MHA + 多头 + 长序列**下会崩。我们如果有这类形状(DeepSeek-V3 系列的 prefill 就是),这是个低成本的检查:用 rocprofv3 看 L2 命中率,如果远低于 80% 就说明踩到了。

**④ 但不要高估。** 收益高度依赖形状:头数少或序列短时四种策略无差别;GQA 上 AITER 已经够好;backward 只有 1.10%–10%;causal 下是负优化。**这是一个针对特定形状的补丁,不是普适加速。**

---

## 10. 复现检查清单

- [ ] 代码开源:**否**,论文未给链接
- [ ] 数据:N/A(kernel 基准,无数据集)
- [ ] 超参完整:**是**(Table 2/3 给了完整 sweep 配置)
- [ ] 绝对性能数字:**否**,全部归一化,无法与外部 kernel 对比
- [ ] 硬件:MI300X 单卡;未测 CDNA4
- [ ] 可替代复现路径:**FlyDSL 的实现是公开的**(`kernels/attention/flash_attn_utils.py`),且带 `xcd_swizzle` 开关,可直接 A/B

## 待跟进

- [ ] 在 MI355X(CDNA4)上验证结论是否平移——论文只测了 MI300X
- [ ] 用 FlyDSL 的 `XCD_SWIZZLE` flag 做一组 A/B,拿到我们关心形状下的真实收益
- [ ] 把 ACC 思路套到 ROCmoe 的 grouped GEMM,评估 expert 锁 XCD 的可行性与负载不均风险
