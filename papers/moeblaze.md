# MoEBlaze:打破 MoE 训练的内存墙
# MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

> **arXiv:** [2601.05296](https://arxiv.org/abs/2601.05296) (v1, 2026-01-08) · **PDF:** https://arxiv.org/pdf/2601.05296
> **发表:** MLSys 2026（Oral）· **机构:** Meta Platforms + Thinking Machines Lab · **部署:** Meta 推荐系统生产环境
> **代码:** 未开源（截至 2026-07）
> **领域:** MoE 训练 · GPU kernel co-design · 激活内存 · 算子融合
> **核心贡献:** 用「元数据索引 + on-the-fly gather/scatter」彻底消除 per-expert routed-token 缓冲区，再叠 SwiGLU 融合 kernel + SiLU backward 重算，单 H100 上单 MoE 层 vs Megablocks **激活内存 ↓4× / 端到端 ↑6.2×**（均为单卡、单层数字）。

> **重读修正（2026-07-07）：** 旧版笔记基于摘要推断，把 baseline 写成「PyTorch / DeepSpeed / Megatron-Core」并对「4× 是哪一级加速」存疑。据 arXiv v1 全文：**唯一 baseline 是 Megablocks**；实验是**单张 H100、单个 MoE 层、fwd+bwd（不含 optimizer）**；SiLU 内存最高 3.6×、加速 1.4–3.7×，SwiGLU 内存 ~4×、加速 2–6.2×。本页据原文重写。

---

## 一、问题分析

### 1.1 研究背景

- **内存墙（memory wall）**（Wulf & McKee 1995）：几十年来算力增速远快于内存带宽/延迟；即使算术单元充足，端到端吞吐往往被参数/激活的读写与交换速度卡住。
- **MoE 把这堵墙放大**：稀疏激活（每 token 只激活 top-k 专家）虽降 FLOPs，却降低了计算密度，并在分布式 LLM 训练里显著抬高内存压力。序列越长、batch 越大，性能越受内存/通信子系统而非 FLOPs 约束。
- **过往做法的代价**：早期用 token dropping / padding（Switch/GShard）限制缓冲区大小 → 伤模型质量；较新系统（Megablocks、DeepSpeed-MoE）聚焦稀疏计算的 compute/comm 复杂度，但 **token dispatch 的辅助缓冲 + 中间结果物化仍占内存大头**。

### 1.2 问题定义

记号：输入激活 $\mathbf{x}\in\mathbb{R}^{L\times d}$，$L$ = 一步内路由的 token 数（batch × seq），$K$ = 每 token 选中的专家数，$E$ = 专家总数，$d$ = 模型维度，$h$ = FFN 中间维度。

论文明确锁定**两处激活内存瓶颈**，并用 DeepSeek 量级配置给出量化：

| 瓶颈 | 复杂度 | DeepSeek 量级实例 |
|---|---|---|
| **① Token routing buffer** | $O(L\times K\times d)$ | $L\approx2\text{M}, K=4, d=6144$，bf16 → **≈ 94 GB / 层**（单个路由缓冲区就近百 GB） |
| **② FFN 中间激活**（SiLU/SwiGLU） | 前向 $O(L\times h)$，反向更高 | $L\approx2\text{M}, h=24576$，bf16 → **≈ 98 GB / FFN 层** |

这两块直接超出单卡 HBM，限制可用 batch/seq，且带来大量冗余数据搬运。

### 1.3 解决方案

**核心思路：** 不再把路由后的 token 物化成 per-expert 连续缓冲区，而是在 dispatch 阶段只生成**轻量索引结构**，在 expert 计算时**从原始未重排的激活张量按索引 on-the-fly gather**，输出时**按 token→expert 反向索引 on-the-fly reduce** 直接写回 $(L,d)$。再对 SwiGLU 这类复杂激活做 **kernel 融合 + 智能激活检查点（backward 重算 SiLU）**。两招都**不牺牲精度、不做 token dropping/padding**，且同时提升算力效率。

**方法概述：**
1. **元数据驱动的 dispatch**：用 4 个索引张量记录路由，不分配 routed-token 缓冲区。
2. **前向**：expert MLP 从原张量 gather；只缓存两层 back-to-back MLP 之间的中间结果；输出聚合与第二层 MLP 融合。
3. **反向**：用反向映射索引做 scatter，避免把 $(L,d)$ 梯度「扩展」成 $(L\times K,d)$ routed 梯度。
4. **高并行 dispatch 构建**：3 步、atomic-free、避免 GPU 上多趟 radix sort。
5. **SwiGLU 融合 kernel + 激活检查点**：两个第一层投影 + 激活 epilogue 融进一个 kernel，backward 重算 SiLU。

**技术细节：**

*4 个索引数据结构*（Figure 2 例子：$L{=}6,E{=}4,k{=}2$）：

| 结构 | 形状 | 作用 |
|---|---|---|
| `expert_token_indices` | $L\times k$ | 按专家拼接的 token-id 列表，expert 取自己的输入 token |
| `expert_token_offsets` | $E+1$ | 每专家 token 数的 exclusive prefix sum（分段边界） |
| `token_expert_indices` | $L\times k$ | 上者的逆映射，按 token 序存路由的 expert-id |
| `token_index_map` | $L\times k$ | 每 token 的 k 个专家输出在中间缓冲里的位置，供最终合并 gather |

*atomic-free 3 步 dispatch 构建*（替代多趟 radix sort）：
- **Step 1 建稠密位图** `dense_token_map`（$L\times E$）：每 token 对其 top-k 专家列写入 token-id，其余置空。每 warp 处理一批 token 行，专家 id 唯一 → warp 内无写冲突。
- **Step 2 算专家长度**：CTA 网格按列（专家）铺，每 CTA 数一列非零项 + warp 级归约得 `expert_lengths`，再 prefix sum 得 `expert_offsets`。
- **Step 3 路由索引到位**：两阶段 location map（tile 级 shared-memory exclusive scan + 加上全局 `expert_offsets`）算出每个非零项的最终位置，再一个并行 kernel 直接写入 `expert_token_indices`，**全程无 atomic**。

*SwiGLU 融合 kernel + 激活检查点*（Algorithm 1）：SwiGLU$=\text{SiLU}(xW_1)\odot(xW_2)$。观察到激活函数计算是 **memory-bandwidth-bound**（点乘为主 + tall-skinny $L\gg d$）但**中间物化内存巨大**。融合做法：一次 load $x$，同时流过 $W_1,W_2$ 两个 GEMM，在寄存器/shared memory 算 $\text{SiLU}(a)$ 并立即乘 $b$，**只把最终输出写回 global**——省掉 $a,b$ 的写回与重读，且把 $x$ 读取减半。反向：**重算 SiLU**（廉价点乘），两分支梯度用 tiled reduction 就地聚合，消除临时 global 缓冲。

---

## 二、实验效果

### 2.1 实验设置

| 项 | 详情 |
|---|---|
| 硬件 | **单张 NVIDIA H100**（利用 warp-group MMA / TMA / cluster tiling） |
| 软件 | PyTorch 2.0.1 + CUDA 12.1 |
| 测量 | **单个 MoE 层**端到端 fwd+bwd（**不含 optimizer**）；激活内存用 PyTorch saved-tensor hooks 精确统计 |
| Baseline | **仅 Megablocks**（当前 SOTA 稀疏训练系统） |
| 配置 | 7 个 config（Table 1），`ffn_hidden = 4 × input_d`，覆盖 $d\in\{512,1024,2048\}$、$E\in\{4,8,16\}$、$k\in\{1,2,4\}$ |
| 激活 | ReLU/SiLU 与 SwiGLU 两组 |

### 2.2 主要结果

| 场景 | 指标 | MoEBlaze vs Megablocks |
|---|---|---|
| **SiLU** | 峰值激活内存 | 最高 **3.6×**（conf4：6100 MB vs 22000 MB） |
| **SiLU** | 端到端加速 | **1.4×–3.7×**（conf4 最大） |
| **SwiGLU** | 峰值激活内存 | **~4×**（conf3：~10000 MB vs >40000 MB） |
| **SwiGLU** | 端到端加速 | **2×–6.2×**（比 SiLU 更高更稳） |

**关键发现：**
- 内存/加速收益随 $L$（序列）与 $k$（激活专家数）**成比例放大**：conf1（$k{=}1$、小 $L$）收益最小，大 $d$/大 $E$（conf4/conf3）收益最大。
- **SwiGLU 收益 > SiLU**：复杂激活的中间物化更多，融合 + 重算省下的带宽更关键。
- 加速三来源：① 轻量 dispatch（省 permute 延迟）；② atomic-free dispatch 构建（避免多趟 sort kernel + CPU 侧瓶颈）；③ 融合 batched-GEMM 吃满 H100 的 warp-group MMA / TMA。

### 2.3 消融

论文未给逐组件的表格化消融；从设计与结果反推：**dispatch 元数据化**主要贡献「消除 routing buffer + 省 permute」，**SwiGLU 融合 + SiLU 重算**主要贡献「SwiGLU 组更高的内存与加速」。（原文未提供独立 ablation，属复现待补项。）

---

## 三、业界类似方案

### 3.1 方案对比

| 方案 | 年份 | 核心思路 | 相对定位 |
|---|---|---|---|
| **Megablocks** | 2023 | 把 MoE 重构成 block-sparse GEMM，免 padding/dropping | 本文唯一 baseline |
| **Tutel** | 2023 | 运行时自适应并行 + pipelining 应对路由波动 | 侧重并行/调度 |
| **DeepSpeed-MoE** | 2022 | 训练+推理系统级优化 + 压缩 | 侧重系统/压缩 |
| **TurboMoE** | 2025 | gating 路径是瓶颈：融合 + metadata 驱动 kernel + 数据布局变换 | 与本文最近亲：都做 metadata 驱动 + 融合 |
| **MoEBlaze**（本文） | 2026 | 元数据索引消除 routed buffer + SwiGLU 融合 + SiLU 重算 | **专攻 routed 激活内存**，正交于稀疏映射/编排 |

### 3.2 技术路线对比

- **路线 A：稀疏计算映射**（Megablocks block-sparse / grouped GEMM）——解决「变长 per-expert 计算怎么高效跑」，但辅助 routing buffer 仍在。
- **路线 B：gating/dispatch 融合**（TurboMoE、本文）——把 routing 变成轻量索引 + 融合 kernel，直接砍激活内存与访存。MoEBlaze 在 B 上进一步**彻底不物化 routed token**，并把 SwiGLU 融合 + backward 重算纳入同一 co-design。

### 3.3 本文定位

- **相对 Megablocks**：不再需要 per-expert 物化缓冲区；dispatch 构建 atomic-free、免多趟 sort。
- **相对 TurboMoE**：不止融合 gating，而是 **routing + expert compute + 激活 epilogue** 全链路 co-design，且量化了 SwiGLU 的内存收益。
- **独特贡献**：形式化两处内存瓶颈（94 GB routing / 98 GB FFN 实例）；atomic-free 3 步 dispatch；SwiGLU 融合 + SiLU 重算把「存 vs 重算」的传统 trade-off 打破（重算廉价、省的是带宽）。

### 3.4 推荐进一步阅读

| 论文 | 理由 |
|---|---|
| Megablocks（MLSys'23） | 唯一 baseline，block-sparse MoE 的基线实现 |
| TurboMoE（2025） | 最近亲，metadata 驱动 kernel + 数据布局变换 |
| MegaScale-MoE（EuroSys'26，[`./megascale-moe.md`](./megascale-moe.md)） | 互补：MoEBlaze 单卡内存/kernel，MegaScale-MoE 多机通信 overlap |

---

## 四、关键章节精译（摘要 + 方法要点）

> 全文 8 节，此处精译最能决定复现的摘要 + §3/§5 方法要点；实验数字见上表，不重复翻译。

**摘要。** 「内存墙」瓶颈在大规模 MoE 架构里被显著放大。MoE 固有的架构稀疏性导致稀疏算术计算，同时引入巨大的激活内存开销——源于庞大的 token routing 缓冲区，以及物化、缓冲中间张量的需求。这种内存压力限制了 GPU 上可容纳的最大 batch size 与序列长度，还带来过量数据搬运，阻碍性能与高效扩展。我们提出 MoEBlaze，一个通过系统协同设计解决上述问题的内存高效 MoE 训练框架：(i) 一套端到端 token dispatch 与训练方法，用优化数据结构消除中间缓冲区与激活物化；(ii) 协同设计的 kernel 配合智能激活检查点，在降低内存足迹的同时获得更好性能。我们证明 MoEBlaze 相较现有 MoE 框架实现 **4× 以上加速与 50% 以上内存节省**。

**§3 内存高效路由算法。** 给定 $(L,d)$ 输入，核心是用 dispatch 阶段生成的辅助索引列表贯穿整个 MoE 计算，实现 on-the-fly 的 token 访问与结果归约。融合 kernel：① 消费 gating 决策、构建 expert-token 索引列表；② 用 on-the-fly gather（从原始未重排张量按 `expert_token_indices` 取数）执行 expert MLP；③ 用 `token_expert_indices` 直接把 MLP 结果 sum-reduce 进输出张量。仅存两层 back-to-back MLP 之间的中间结果供 backward。

**§5 kernel co-design。** 观察：现代复杂激活（SiLU/SwiGLU）计算轻但物化重，且在 $L\gg d$ 的 tall-skinny 形状下 memory-bandwidth-bound。做法：把两个第一层投影 + SwiGLU epilogue 融进单 kernel，input 只读一次、寄存器/shared 内算 SiLU 并立即乘、只写最终输出；backward 重算 SiLU、tiled 就地聚合两分支梯度。

---

## 五、局限与复现清单

**局限：**
- **仅单卡**：所有实验单张 H100、单个 MoE 层。多机分布式明确列为 future work（其索引/融合原语作者称也适用于分布式，但未验证）。
- **baseline 单一**：只对 Megablocks；未对 DeepSpeed-MoE / Megatron-Core / TurboMoE。
- **无收敛证据**：只测速度与内存，未给 loss/perplexity/下游精度曲线佐证「不掉精度」。
- **负载不均、numerical stability、long-context 极端场景**均未系统评测（见原文社区整理的 knowledge gaps）。
- **通信影响**：单卡场景不涉及 all-to-all；作者指出内存省下来后，多机时 all-to-all 可能成为新瓶颈。

**复现清单：**
- [ ] 代码开源：**否**（截至 2026-07）
- [ ] 数据/配置：Table 1 给了 7 个 shape，可复现 microbenchmark
- [ ] 硬件依赖：H100（warp-group MMA / TMA），非 Hopper 收益可能下降
- [ ] 关键实现：atomic-free 3 步 dispatch + SwiGLU 融合 kernel（低层代码未给）

---

## 六、对 monolith-moe / rocmoe 的启示（Our take）

MoEBlaze 是我们 super-kernel 工作在**「单卡计算/内存 kernel」维度上的强互证**——它没做 comm overlap，恰好是我们的差异化留白，但它的每一招我们几乎都在 IPC/kernel 层踩过对应的坑。

| MoEBlaze 的招 | 我们的对应（monolith-moe / rocmoe） | 可借鉴点 |
|---|---|---|
| 元数据索引、不物化 routed token、on-the-fly gather | Layout-P + receiver-pull + `pack_perm`（sender counting-sort） | 思路同源：都拒绝物化中间大缓冲 |
| **atomic-free 3 步 dispatch 构建** | 我们的 sender-side pack 用 **counting-sort + `atomicAdd`**（非确定序） | ⭐ 值得对标：他们主张 atomic-free 更快、且确定性；我们的 `atomicAdd` 非确定序曾逼我们「backward 必须直接读 `ws.pack_perm`」。评估把 pack 换成 dense-map + prefix-sum 的 atomic-free 构建 |
| **SwiGLU 融合：一次 load，寄存器算 SiLU，只写最终输出** | Phase 2.1 **SwiGLU pre-compute → FC2 全 DTOLDS**（FC2 swiglu_tiles −64%，lds_write 全消） | 结论一致：SwiGLU 中间量不该落 HBM。我们已落地等价优化，互证方向对 |
| **backward 重算 SiLU（廉价点乘换内存）** | decomposed backward（复用 forward 保存量手算 per-expert bwd） | 可评估：backward 里 SiLU 也走重算而非保存，进一步省激活 |
| 融合 batched-GEMM 吃 warp-group MMA / TMA | `mfma_tile.h`（99.3% MFMA）+ DirectToLDS | H100 的 wgmma→MI355X MFMA、TMA→DTOLDS，招式可平移 |

**三条最有用的结论：**
1. **atomic-free dispatch 构建**是我们没试过的方向。当前 `atomicAdd` pack 的非确定序给 backward 带来强耦合（必须复用 sender 的 `pack_perm`）；改成 dense-map + 两阶段 prefix-sum 的确定性构建，可能既省 atomic 争用又解耦 backward。列为 rocmoe 一个候选实验。
2. **SwiGLU 融合 + backward 重算**被独立验证有效——我们 Phase 2.1 已经在 forward 侧做了 pre-compute，backward 侧的 SiLU 重算还没上，是低风险增量。
3. **定位再确认**：MoEBlaze 4–6× 是**单卡、无通信**的内存/kernel 收益；我们的战场是 XGMI in-kernel comm-compute overlap，两者正交、可叠加。

> 相关笔记：[`../notes/monolith-moe/2026-05-13_1245_swiglu_precompute_fc2_full_dtolds.md`](../notes/monolith-moe/2026-05-13_1245_swiglu_precompute_fc2_full_dtolds.md)（SwiGLU pre-compute）、[`../notes/monolith-moe/2026-05-13_2220_decomposed_backward_landed_plus55pct_vs_eager.md`](../notes/monolith-moe/2026-05-13_2220_decomposed_backward_landed_plus55pct_vs_eager.md)（decomposed backward 与 `pack_perm` 耦合）。

---

*据 arXiv:2601.05296 v1 全文重写于 2026-07-07（原摘要版笔记 2026-03-07）。HTML 版：[`moeblaze.html`](./moeblaze.html)。*
