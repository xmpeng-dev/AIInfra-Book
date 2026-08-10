# MatrixFSDP：ZeRO-3 参数分片下的零通信矩阵优化器
# MatrixFSDP: Communication-Free Matrix Optimizers under ZeRO-3 Parameter Sharding

> **arXiv:** [2607.05895](https://arxiv.org/abs/2607.05895) | **HTML:** [全文](https://arxiv.org/html/2607.05895)
> **发表信息:** Preprint, 2026-07-07（0 引用）
> **机构:** University of Pittsburgh · Google · 清华大学（署名见 §5 可信度备注）
> **代码:** 未提供
> **领域:** 分布式训练 · 矩阵优化器 · FSDP/ZeRO-3 · Muon/Shampoo/SOAP
> **核心贡献:** 不改优化器、改 **ZeRO-3 分片放在哪** —— 每个 2D 权重让一个 DP rank 持有整块、其余持空分片，普通 backward 归约天然把完整矩阵梯度落在 owner 上，于是 Newton-Schulz 本地跑、**optimizer step 零矩阵集合通信**。64×A100：optimizer step 相对 stock FSDP2-Muon 快 **4.2×（1 节点）→ 54.6×（8 节点）**，端到端 **1.37× → 2.15×**，同时保住 ZeRO-3 级别显存。

---

## 一、问题分析

### 1.1 背景

Muon / Shampoo / SOAP 这类矩阵优化器靠**整块 2D 矩阵**做正交化或预条件，token 效率优于逐坐标的 AdamW。但 FSDP2 / ZeRO-3 把参数、梯度、优化器状态都切成 per-rank 分片——**这个契约是为逐坐标 AdamW 量身定做的**：梯度分片一归约完，优化器就能本地更新，不需要再来一次模型尺寸的集合通信。

Muon 打破了这个契约。Newton-Schulz 里的 `XXᵀ` 把矩阵的所有行耦合在一起，**一个 row-band 不是同一个更新规则的合法输入**。

### 1.2 现有两个端点都不完整

| 方案 | 做法 | 代价 |
|---|---|---|
| **FSDP2-Muon**（保 ZeRO-3） | 每个 optimizer step 重建完整矩阵 | 每 rank 每步多搬约 `2Ns(W-1)/W` 字节，而且**发生在 backward 之后** |
| **ZeRO-1 owner placement**（Moonshot 分布式 Muon、Distributed Shampoo、Canzona 的 DP 路径） | 整矩阵指定 owner，矩阵运算本地做 | 每个 rank 都常驻完整参数，**放弃了 ZeRO-3 的显存节省** |

论文点出的关键在第一行的后半句：这部分流量**不像 FSDP 的 fwd/bwd all-gather 那样能被层计算盖住，它是纯粹的优化器关键路径**。对 transformer 这种 2D 权重占参数量绝大部分的结构，这个代价大、每步都付、且随模型规模与节点数增长。

### 1.3 解决方案

**核心思路**：不改被计算的优化器，改 ZeRO-3 分片**住在哪**。

对每个 DP 分片的 2D 权重 `W_i`，选一个 DP rank `o_i` 作为 owner。reshard 之后 owner 存**整个矩阵**，其余 rank 对该参数存**空分片**。同一个 FSDP unit 里的非 2D 张量打包成一个 tail role，整体交给 tail owner，仍走 AdamW。

跨整个 DP 组仍然只有一份常驻副本——**与普通 ZeRO-3 的差别只是这份副本按张量角色分配，而不是等分切给所有 rank**。

于是：backward 照常产生 per-rank 梯度，ZeRO-3 照常把它们归约进当前的本地分片形状；而因为 2D 矩阵的本地分片在 owner 上**就是整个矩阵**，归约后的 Muon 输入自然落在 owner 手里。owner 跑 NS、更新自己的参数和 Muon 状态；持空分片的 rank 对这个矩阵不做任何优化器工作。**optimizer step 里没有任何矩阵 all-gather、broadcast 或 redistribute。**

被 TP 切碎的矩阵不在此列，由外围 TP runtime 处理。

### 1.4 四个工程组件

代价是常驻分片不再等大、也不再每个 rank 都有。如果用普通等大集合通信硬扛，要么把空 rank 补齐到最大 owner 分片，要么在根本不做优化的 rank 上重建整个矩阵。所以论文把 ownership 做成一等公民：

**(a) MatrixShard 元数据。** 不用 `Shard(0)` 的等分区间，改成按 rank 索引记录每个扁平化参数区间的 placement：2D 矩阵恰好一个完整 segment 在 owner 上，其余为空；非 2D tail 打包成一个整段放 tail owner。同一份元数据同时驱动优化器路由、buffer pinning、DCP 存取和跨 world-size 重分片。

**(b) 均衡感知的 owner planner。** 全局规划、块内执行。三种策略：`ROLE-GREEDY`（每块内把最大 role 分给当前最轻的 rank，负载往后累计，显存保守的默认值）、`SCOPE-GREEDY`（把所有块的 role 汇总统一分配，全局均衡更好）、`COST-AWARE`（只在均衡收益不超 workspace 上限时采用 scope 方案）。最后校验每个参数只有一段连续本地 segment，保证热路径仍是标准 FSDP2 块调度。

**(c) Owner-segment P2P 集合通信。** 这是全文的工程核心，带宽模型也在这里。设 `W` 为 shard span，`P_B` 为 FSDP 块 B 的参数字节，`O_{B,r}` 为该块中 rank r 拥有的字节：

$$\gamma_B = \frac{\max_r O_{B,r}}{P_B / W}, \quad C^{\text{FSDP}}_B \approx P_B\frac{W-1}{W}, \quad C^{\text{owner}}_B = (W-1)\max_r O_{B,r} = \gamma_B \cdot P_B\frac{W-1}{W}$$

也就是 **owner 路径的带宽成本 = γ_B × 普通 FSDP 成本**。一个块里如果有一个巨型 owner，`γ_B ≈ W`，带宽会差 W 倍；planner 把矩阵和 tail owner 在块内摊开后 `γ_B ≈ 1`，带宽主项与普通 FSDP 持平，而 optimizer-step 的重建被消掉了。实现上 materialization 是 owner fanout（只发非空 segment 给当前需要完整参数的 rank），梯度归约则是每个 rank 贡献本地完整梯度缓冲、结果只写进非空 owner segment。快路径用确定性 native send/recv 调度，通信开始前做跨 rank 校验，把不一致变成显式报错而不是 NCCL 静默超时。

**(d) Owner-buffer pinning。** 一个 FSDP2 生命周期里的存储级隐患：autograd 会保存指向瞬态完整参数缓冲的 view，post-forward reshard 会原地缩小这个缓冲但不能把存储还给共享池；backward 前该 unit 又要 resize 并重新填充同一块存储，保存的 view 就在原地复活。实现上给待 backward 的收缩缓冲加了 pinned 状态，由中央 guard 在该 unit 完成 backward 且优化器就绪的本地分片恢复后才释放。

---

## 二、实验效果

### 2.1 设置

| 项 | 内容 |
|---|---|
| 硬件 | 多节点集群，最多 64×A100（80 GB），InfiniBand |
| 模型 | 合成 transformer：16L/3.2B、32L/6.4B、64L/12.9B、128L/25.8B；模型尺寸扫描 1.7B–32B（hidden 4096） |
| 基线 | **stock FSDP2-Muon**（在 sharded DTensor 上跑正确的分布式 NS，最强正确基线）、gather-once FSDP2-Muon、ZeRO-1 owner placement、DDP Muon（正确性 oracle） |
| 收敛校验 | 12 层 decoder-only LM，WikiText，10,000 bf16 步（5.24B token） |

值得注意：**gather-once FSDP2-Muon 反而比 stock 慢**——在每个 rank 上都物化完整矩阵再冗余跑 NS，比 PyTorch 的分片 NS 更贵。所以论文拿更快的 stock 当基线。

### 2.2 主要结果

**弱扩展（Table 3，深度随 shard span 增长）：**

| 节点(span) | 模型 | stock opt | MatrixFSDP opt | opt 加速 | stock 总 | MatrixFSDP 总 | **E2E** |
|---|---|---|---|---|---|---|---|
| 1 (8) | 16L / 3.2B | 367 ms | 87 ms | 4.2× | 991 ms | 725 ms | **1.37×** |
| 2 (16) | 32L / 6.4B | 1285 ms | 89 ms | 14.5× | 2612 ms | 1489 ms | **1.75×** |
| 4 (32) | 64L / 12.9B | 3079 ms | 91 ms | 34.0× | 5788 ms | 2689 ms | **2.15×** |
| 8 (64) | 128L / 25.8B | 5064 ms | 93 ms | 54.6× | 10004 ms | 4989 ms | **2.01×** |

这张表最重要的一列是 **MatrixFSDP opt 那列几乎是平的（87 → 93 ms）**，而基线从 367 涨到 5064 ms —— 因为基线的每步矩阵通信要跨节点间网络，而 MatrixFSDP 的 step 完全本地。**加速是个扩展性性质，不是固定常数。**

**模型尺寸扫描（Table 4，固定 span 64）：**

| 模型 | stock opt | MatrixFSDP opt | opt 加速 | E2E | MatrixFSDP 显存 | gather-once 显存 |
|---|---|---|---|---|---|---|
| 1.7B | 1016 ms | 6.4 ms | 159× | 3.3× | 1.4 GB | 4.4 GB |
| 4B | 1235 ms | 14.8 ms | 83× | 2.4× | 2.6 GB | 9.7 GB |
| 8B | 1583 ms | 34 ms | 47× | 2.0× | 4.3 GB | 18.4 GB |
| 14B | 2286 ms | 73 ms | 31× | 1.85× | 6.3 GB | 30.7 GB |
| 32B | 6219 ms | 160 ms | 39× | 2.2× | 10.0 GB | 61.6 GB |

**单节点分阶段（Table 1，16L/3.2B，8×A100）**：opt 87 vs 367 ms；forward 175 vs 163 ms（慢 1.07×）；backward 462 vs 460 ms（持平）。也就是 owner fanout 让 forward 付了 7% 的税，换掉了整个 optimizer 重建。

**显存**：4.3B 模型 64 GPU 上，ZeRO-1 owner placement 18.5 GB/rank、stock FSDP2-Muon 2.1 GB、MatrixFSDP 2.7 GB，optimizer step 分别 677 / 651 / 19 ms。到 12.9B 时对 ZeRO-1 owner 的显存差距拉到 15×（54.5 vs 3.6 GB/rank）；ZeRO-1 owner 在 ≥14B 越过 A100 80 GB 上限，MatrixFSDP 在 14B 用 6.3 GB、32B 用 10 GB。相对 stock FSDP2-Muon 只多付 owner 整矩阵常驻的代价（峰值分配约 +20% 以内）。

**正确性**：确定性 fp32 harness（关 TF32、固定 batch、rank 各异输入）逐步比对 loss、最终 logits、以及每个参数重建出的完整梯度；10,000 步 bf16 WikiText 实数据训练下与 DDP Muon 参考的最大打印 `|Δloss| = 0`（论文明确说是打印精度一致，不是 bit-wise 相同）。

**优化器泛化**：保持同一套 placement / collectives / routing / checkpointing，把 owner-Muon 换成 Shampoo（完整 L/R 预条件 + 四次方根逆）和 SOAP（Shampoo 特征基下的 Adam），64-GPU fp32 loss 轨迹相对各自的 gathered 参考分别匹配到 ≤4e-5 和 ≤1.3e-4。零 optimizer-step 通信这个性质是**结构性继承**的。

### 2.3 消融：真正干活的是 P2P collectives，不是 placement

Table 5（64L / 12.9B）：

| 节点(span) | role | scope | cost | **all-gather** | 退化 | balance (r/s/c) |
|---|---|---|---|---|---|---|
| 1 (8) | 2833 ms | 2842 | 2830 | 3114 ms | 1.10× | 1.07/1.02/1.07 |
| 2 (16) | 2736 | 2809 | 2729 | 5817 | 2.13× | 1.05/1.02/1.05 |
| 4 (32) | 2565 | 2572 | 2560 | 9786 | 3.82× | 1.12/1.03/1.11 |
| 8 (64) | 2511 | 2486 | 2508 | 18079 | **7.20×** | 1.27/1.04/1.27 |

**三种 owner 策略之间总步时相差不超过 3%**（`scope_greedy` 均衡最平，`role_greedy` 是显存保守的默认）。但把 owner-segment 自定义集合通信换成矩阵 all-gather，8 节点下端到端从 2.51 s 涨到 18.08 s，forward 慢 9.4×、backward 慢 6.6×。

**结论很硬：owner placement 本身不够，自定义 P2P owner collectives 才是让这个布局能塞进 FSDP2 正常 materialize/reshard 循环的机制。** 用 §1.4(c) 的语言说，自定义路径搬的是均衡后的 owner segment，all-gather 变体则退回到在 materialization 路径上搬完整矩阵。

**负载均衡**：全扫描下 per-rank 优化器计算不均衡（max/avg）1.26–1.39，常驻参数显存不均衡 1.18–1.21。论文自己拿去对标 Canzona 的 α-balanced 静态划分（报 1.43×，从 3.24× 改进而来），并强调 MatrixFSDP 还额外要保持执行块内局部以维持 FSDP2 的 prefetch 与 overlap。

---

## 三、与 DMuon 的路线对照

这是本文最值得对我们说的部分。**DMuon 和 MatrixFSDP 解决的是同一个问题，但动的层次不同。**

| 维度 | [DMuon](./dmuon.md) | MatrixFSDP |
|---|---|---|
| 改什么 | **runtime**：在 FSDP2 之上外挂一个 owner 通信运行时 | **分片放置**：改 ZeRO-3 的 shard 住在哪 |
| owner 怎么拿到完整矩阵 | 主动 gradient routing 归约到 owner | **普通 backward 归约天然落在 owner 上**（owner 的本地分片就是整矩阵） |
| 对通信的处理 | **藏**：forward 提前 materialize 发布、异步 publish 与下一步计算重叠、XOR owner layout 分散争用 | **消**：optimizer step 根本没有集合通信；fwd/bwd 走 FSDP2 原有 overlap |
| 集成方式 | **drop-in，用户 API 3 行**，不改框架源码 | 要改 FSDP2 内部四处：placement 元数据、owner collectives、buffer pinning、checkpoint 重分片 |
| NS kernel | Gram 空间 SYRK 对称 kernel（贡献 48% 优化器加速）+ shape batching + DSL autotune | 未做 kernel 优化，用现成的 |
| 负载均衡 | 实测代价 **MILP** 最小化 makespan（搜索空间过大时 greedy fallback） | greedy planner（role / scope / cost 三档） |
| TP 支持 | DP owner 内再指定 TP owner，gather slices → full NS → scatter | **明确不支持**，被 TP 切碎的矩阵跳过，交给外围 TP runtime |
| optimizer step 加速 | 6.85–163× vs Muon-AG | 4.2–54.6×（弱扩展）/ 31–159×（尺寸扫描）vs stock FSDP2-Muon |
| 端到端 | 1.48–3.01× vs Muon-AG；vs AdamW 仅 **+2%** | 1.37–2.15×（弱扩展）/ 1.85–3.3×（尺寸扫描） |

**一句话概括差别：DMuon 是「把通信藏起来」，MatrixFSDP 是「让通信不存在」。**

但这里有个必须点破的地方：**MatrixFSDP 并没有真的消灭通信，它是把通信从「不可重叠的 optimizer 关键路径」搬到了「本来就有 overlap 机制的 fwd/bwd materialization 路径」上。** 消融表里 all-gather 变体在 8 节点退化 7.2× 就是证据——那些字节还在，只是走对路径就能被计算盖住，走错路径就原形毕露。

这一点恰好回答了「optimizer overlap 该怎么做」：**最有效的 overlap 往往不是给关键路径上的通信找地方藏，而是改变数据布局，让它落到本来就能藏的位置去。** 这个思路可以脱离 Muon 复用。

两者互补性也清楚：MatrixFSDP 的放置方案 + DMuon 的 Gram NS kernel 和 MILP 均衡是可以拼的。MatrixFSDP 自己承认 NS kernel 没优化，而 DMuon 说 Gram SYRK 贡献了 48% 的优化器加速。

---

## 四、Limitations

论文自己在结论里给的边界（这几条很诚实，值得原样记下）：

- **单节点收益小。** 4.2× 是最小的那一端。
- **高梯度累积会摊薄收益** —— 一次 optimizer step 被摊到很多次 fwd/bwd 上。这条对我们特别相关：大规模 MoE 训练普遍开 GA。
- **固定模型的强扩展下会失效。** `W` 增长远超一个块里够大的 owner role 数量时，`γ_B` 变大、owner fanout 成为下一个瓶颈。论文提议的扩展方向是分层 owner fanout。

我另外补几条：

- **TP 明确不支持。** 对 Megatron 式 3D 并行栈这是硬伤，被 TP 切碎的矩阵完全不在方案覆盖内。Canzona 那条 TP 路径（域内异步重建 fragment）论文说是"future/orthogonal"。
- **完全没谈 MoE / EP。** DSv3 这类模型 expert 权重占参数量的绝大部分，而它们是按 EP 而不是 DP 切的。owner placement 在 EP 维度上怎么工作，论文一个字没提。
- **只在合成 transformer 上评测。** 收敛校验是 12 层 decoder LM 跑 WikiText，不是生产模型。
- **没有开源代码。**

### 可信度备注

这篇要打个问号：三位作者 h-index 均为 0、0 引用，署名机构写的是 University of Pittsburgh / Google / 清华，但通讯邮箱是 gmail 和 163。2026-07-07 的 preprint。**方法本身自洽、消融做得扎实、数据自恰**，但在没有代码、没有同行评议、没有生产模型验证的情况下，那些两位数的加速比应当当作"该设定下的上界"而不是可迁移的预期。真要用，先自己复现单节点那个 4.2×。

---

## 五、Our take

**这条路线对我们值得跟，但要跟的是思路不是实现。**

**为什么值得跟。** 它给了一个我们 roadmap 上正需要的东西的参考答案 —— L102「Layer-wise distributed Muon optimizer 集成」、L250「Muon + layer-wise distributed optimizer for AMD」。MatrixFSDP 证明了在**不改 Muon 语义、不放弃 ZeRO-3 显存**的前提下，optimizer step 的矩阵通信是可以整个拿掉的，而且给出了工程上必须配套的四件套。它同时是 Shampoo / SOAP 的通用答案（结构性继承），这个泛化性比 Muon 专用方案值钱。

**为什么不能直接抄。** 三条拦路虎：TP 不支持、MoE/EP 完全没谈、高梯度累积摊薄收益。我们的场景（DSv3 类 MoE + Megatron 3D 并行 + 大 GA）三条全中。所以直接移植大概率拿不到论文的数字。

**真正可以带走的两条。**

1. **「改布局让通信落到可重叠的位置」这个思路本身**，比 owner placement 这个具体方案更通用。回头看 Megatron 那个已知缺口（Muon 的参数 all-gather 在 optimizer step 后完全暴露，而元素级优化器早有 `--overlap-param-gather`），本质是同一类问题：通信没错，位置错了。
2. **γ_B 带宽模型**是个好工具。它把「owner 放置会不会拖慢 materialization」变成一个可以在 planner 里算的量。做 AMD 侧的 layer-wise Muon 时可以直接借来判断某个 owner 分配是不是带宽安全。

**建议的下一步**：先不动手实现，用 γ_B 模型套一下我们真实的 DSv3 配置（EP + TP + GA），算出理论上 owner placement 能省多少、materialization 会多付多少。如果算出来 GA 一开收益就掉到 10% 以内，这条线就该降优先级，转去做 Megatron 那个 param-gather 缺口——那个不需要改分片布局，收益更确定。

---

## 六、延伸阅读

| 论文 | 为什么值得读 |
|---|---|
| [Canzona](https://arxiv.org/abs/2602.06079)（阿里，2602.06079） | 本文的最近邻：同样解耦逻辑 owner 与物理放置，但 DP 路径走 ZeRO-1；**有 TP 路径**（域内异步重建 fragment），正是本文缺的那块 |
| [Dion](https://arxiv.org/abs/2504.05295)（Microsoft） | 反向路线：改算法而不是改布局。amortized power iteration + 低秩 + error feedback，任何单卡都不需重建整矩阵。代价是**数值上不等价于精确 NS** |
| Gram Newton-Schulz（[Tri Dao blog](https://tridao.me/blog/2026/gram-newton-schulz/)） | kernel 层的正交改进，与本文可叠加；Kimi K2 上优化器时间最多降 50% |
| [FORGE](https://arxiv.org/abs/2606.22932) | 另一个层次：把优化器融进 backward GEMM 的寄存器 epilogue。明说 Muon/Scion「仿射进入 state」故部分可融但未展开，是个空档 |

## 参考

- DMuon 深读笔记：[`dmuon.md`](./dmuon.md)
- 训练优化 landscape：[`../knowledge/systems/training-optimization-landscape-2026.md`](../knowledge/systems/training-optimization-landscape-2026.md)
- H2 2026 roadmap（L102 / L250 layer-wise distributed Muon）：[`../notes/primus-moe/2026-05-29_roadmap_h2_2026.md`](../notes/primus-moe/2026-05-29_roadmap_h2_2026.md)
