# Motif 3: Technical Report

> [arXiv 2608.09119](https://arxiv.org/abs/2608.09119) (v1, 2026-08-10, cs.AI) · Motif Technologies（韩国）· 27 作者
> 训练代码示例：<https://github.com/MotifTechnologies/motif3-training-example>（TorchTitan 基座，标注 "NVIDIA B200"）
> 硬件：**NVIDIA B200**（§3.1 只提到"B200 的显存容量"，**全文未给卡数/集群规模**）；MXFP8 grouped GEMM 用 DeepGEMM（§3.2 称在 Blackwell 上优于 TransformerEngine 与 torchAO）
> 软件栈：自研 TorchTitan 训练栈 + HybridEP（DeepEP `hybrid-ep` 分支）+ Liger fused CE + FlashAttention-4（改过）+ NeMo RL / NeMo Gym / vLLM / SGLang / Ray
> 模型：314B 总参 / 13.2B 激活 · 53 层（2 dense + 51 MoE）· 384 routed + 1 shared expert · top-8 · 12.5T tokens · 256K 上下文
> 资助方：韩国科学信息通信部（MSIT）Sovereign AI Foundation Model Project
> 论文未给权重下载链接，只开源了训练示例仓库

## TL;DR

**一篇双线技术报告：架构线（GDLA / mHC 改造 / Expert-Specific PolyNorm）+ 系统线（§3）。对我们有用的几乎全在系统线，而系统线的致命缺陷是——一个吞吐数字都没有。**

§3 五个小节描述了十几项训练系统优化，每一项都写得足够细可以复刻，但**没有 MFU、没有 TFLOPS、没有 tokens/s、没有加速比、没有卡数**。§3 里唯一的定量结论全是**解析式通信量比值**（1,024× / 1,536× / 2/3 / 4/3）和一个 profiling 观测（3.54×），不是实测收益。所以这篇的正确读法是**当成一份"设计决策清单"而不是"优化收益清单"**。

**最该记住的四条：**

1. **mHC 的 post-mapping 乘子必须从 2 退火到 1**（§2.3）。原始 mHC 用 `H_post = 2σ(z)`，值域 `(0,2)`；在 Motif 3 的深度上，`>1` 的映射会**逐层反复放大** sublayer 输出，导致激活离群值累积。他们把乘子 `s_t: 2 → 1` 在预训练期间退火，保留早期优化行为、消掉后期的持续放大。
   → **我们本地 `Primus/primus/backends/megatron/core/transformer/hyper_connection.py:359` 就是硬编码的 `post = 2.0 * torch.sigmoid(post_logit)`**，即未退火形式。这是全篇对 Primus DeepSeek-V4 最直接可用的一条。详见 §8.1。

2. **QK-Clip 在 GQA/MLA 下不能对称劈分**（§3.3）。常规做法是 Q 侧和 K 侧各乘 `√γ_h`；他们发现这会**把共享的 K 投影压到零**——因为 `G` 个 query head 共享一个 KV head，共享 K 收到的梯度方向部分相消，恢复信号只有 `1/√G` 量级，而每个 query head 保有全额恢复信号。修正：K 侧比例 `r = 1/(1+√G)`，Q 侧乘 `γ_h^(1-r)`、K 侧乘 `γ_h^r`；rotary K 跨 head 共享所以完全不动，改由 rotary Q 承担全额 `γ_h`。

3. **在他们的规模上，"消除"专家权重 All-Gather 比"隐藏"它更划算**（§3.1）。原文明确说通信 kernel 会吃 SM、显存和互连带宽，因此会拖慢与之重叠的计算。做法：梯度累积期间中间 microbatch 不 reshard 专家参数；51 个 MoE 层里挑 22 层让专家在 forward 与 backward 之间也保持 unsharded，把权重收集降到**每个 optimizer step 恰好一次 All-Gather**。
   → 这和 [Tessera](./tessera.md) 在生产里量到的"EP 通信 kernel 占 ~20 SM 导致 10–20% 减速"是同一个现象的两次独立确认。

4. **长上下文 CP 要逐层选算法，且要在 DP × microbatch 维度重排负载**（§3.5）。full-attention 层用 Ulysses、sliding-window 层用 window-aware Ring（单次 halo 交换）。理由是 SWA 把计算从 `Θ(L²)` 降到 `Θ(LW)`，而 Ulysses 的 all-to-all 量还是 `Θ(L)`，小 `W` 下通信/计算比很难看。另外 document-masked packing 会让**同样标称长度的序列 attention FLOPs 差 3.54×**，他们用 LPT 贪心在全局 pool 上重排。

**一个必须打问号的地方**：§4.2 提到早期 MoE 层路由塌缩时，他们的处置是**把该层的 router 参数和 expert-selection bias 直接替换成相邻稳定层的**。这是人工干预训练中的权重，论文没说触发了几次、在哪些 step、对最终模型有什么影响。见 §7.2 第 2 条。

## 1. Problem 与定位

### 1.1 这不是一篇优化论文，是一篇模型报告

先把定位说清楚，避免误读。这篇的主张是「架构与系统协同设计能让 314B/13.2B 激活的模型打得过 428B–1.6T 的同代开源模型」。它不提出可复用的单点优化并证明其收益，而是**把一次真实的 314B 训练里所有决策摊开**。因此：

- **值得读的**：§2.3（mHC 改造）、§3 全部、§4.2（MoE 稳定化与专家健康度指标）。这三处含有别处拿不到的工程细节。
- **可以跳的**：§5.1–5.2（post-training 数据与 RL 配方，除非在做 RL 后训练）、§4.1（数据配比）。
- **不能当依据的**：任何"这项优化带来 X% 提升"的推断——论文没给。

### 1.2 它想解决的三个问题

| 问题 | 表现 | 论文的答案 |
|---|---|---|
| 细粒度 MoE 的专家可用性 | 384 专家 top-8，路由塌缩 / 专家饿死 / 功能同质化 | 分层均衡（§4.2）+ 6 项专家健康度指标（表 3） |
| 大深度 + 多流残差的数值稳定性 | mHC 的 `(0,2)` post-mapping 逐层放大出激活离群值；FFN 门控乘积放大极值 | post-mapping `2→1` 退火 + FFN 幅值软惩罚 |
| 256K 上下文的并行效率 | 混合 attention 下没有单一 CP 算法通吃；document packing 造成 rank 间负载不均 | 逐层选 CP 算法 + attention 负载感知重排 |

## 2. 架构

### 2.1 配置表（§4.3 + 表 1）

| 项 | 值 |
|---|---|
| 总参 / 激活参 | ~314B / ~13.2B per token |
| 层数 | 53（前 2 层 dense FFN，其余 51 层 MoE） |
| hidden | 4,096 |
| Q / KV heads | 80 / 16 |
| signal / noise query heads | 64 / 16（grouped ratio `g=4`） |
| d_qk / d_v / rotary dim | 192 / 128 / 64 |
| Q / KV 低秩维 | 1,024 / 512 |
| attention 排布 | 每 4 层 1 层 full causal（层 0,4,8,…），其余 3 层 SWA，**窗口 128 token** |
| FFN 中间维（dense / expert） | 12,288 / 1,280 |
| 专家 | 384 routed（top-8）+ 1 shared |
| 专家激活函数 | Expert-Specific PolyNorm |
| 残差 | 修改版 mHC，**4 条并行残差流** |
| MTP | 1 层（自投机解码用） |
| 词表 / 最大上下文 | 220,160 / 262,144 |

### 2.2 GDLA = GDA + MLA + 输出门控

三件已有工作的组合：

- **Differential Attention**（[2410.05258](https://arxiv.org/abs/2410.05258)）：两条 attention 分布相减，抵消 signal/noise 共有的模式。缺点是 1:1 对称劈分，一半 head 容量花在噪声估计上。
- **GDA**（Motif 自己的 [2510.06949](https://arxiv.org/abs/2510.06949)）：改成 `g:1` 非对称，`n_N = n_H/(g+1)`，`n_S = g·n_N`，noise head 输出按组重复 `g` 次配给 signal head。GDA 原文结论是 3:1 / 4:1 是最优区间，Motif 3 取 `g=4`。
- **MLA**：KV 压成低秩 latent。

GDLA 的**具体缝法**（§2.2.1，这是实现上唯一需要小心的地方）：

```
c^Q  = RMSNorm(x W^Q_a);            [q^C; q^R] = c^Q W^Q_b
[c^KV; k^R] = x W^KV_a
c̄^KV = RMSNorm(c^KV);   [K^C; V] = Reshape_heads(c̄^KV W^KV_b)   ← 只做一次
K = [K^C; Repeat(RoPE(k^R))]                                    ← rotary key 广播给全部 KV head
H_S = Attn(Q_S, K, V);  H_N = Attn(Q_N, K, V)
λ_t = σ(x_t W^λ) ∈ (0,1)^{n_S}                                  ← 每 signal head 一个,token 相关
D_t = H_S,t − λ_t ⊙ Repeat_g(H_N,t)
G_t = Reshape_heads(c^Q_t W^G)
GDLA(x_t) = vec(σ(G_t) ⊙ D_t) W^O
```

三个值得注意的设计选择：

1. **KV latent 只归一化 + 上投影一次，signal/noise 两条路只在展开后的 head 维上分开。** 16 个展开的 content-K / V head 被两条 query 路共享。这避免了维护两份 KV state，推理时仍是 MLA 的紧凑状态。
   → 对照本地笔记里 LAGA 的发现（Megatron-Core 之所以在训练路径上 hard-assert 禁掉 absorbed MLA，是因为中间量 `n_h × d_kv/token` 比被替代的 per-head K/V 更大，DSv3 规模激活显存 +20–34%），GDLA 这个"展开一次、两路共享"的写法正是显存友好的那一侧。
2. **`λ` 从原始 DA 的静态逐层系数改成 token 相关、逐 signal head 的 `σ(x_t W^λ)`。** sigmoid 把系数锁在 `(0,1)`，所以差分操作**只能抑制噪声路，永远不会反转或放大它**——这是个有意的安全约束。
3. **输出门控来自归一化后的 query latent `c^Q`**（不是 `x`），`W^G ∈ R^{r_Q × (n_S d_V)}`，逐元素门控每个 signal head 输出的每个通道。这条来自 Qwen3-Next / gated attention（[2505.06708](https://arxiv.org/abs/2505.06708)）。

**证据强度**：图 2 显示 GDLA 损失低于 GDA 和 MLA，"到达 loss 3.2 比 MLA 少用 9.2% 训练 token"。但**这是在 ~10B 参数的对照实验上测的**（§2.2 明确说图 2 和图 6 的三组诊断对比都是 ~10B 模型），不是 314B 上的消融。全文没有 314B 规模的架构消融——对一篇模型报告这很正常，但引用这个 9.2% 时必须带上"10B 对照实验"的限定。

### 2.3 mHC 改造：唯一一处真正的"修正而非组合"

这是全篇架构部分最有价值的一段，因为它是**踩坑后的修正**，不是设计偏好。

原始 mHC（DeepSeek，[2512.24880](https://arxiv.org/abs/2512.24880)）把单条残差流扩成 `n` 条，三个逐 token 生成的映射：

```
X_{ℓ+1,t} = H_res,ℓ,t X_{ℓ,t} + H_post,ℓ,t^⊤ F_ℓ(H_pre,ℓ,t X_{ℓ,t}; W_ℓ)
```

- `H_pre ∈ R^{1×n}`：跨流加权归约，喂给 sublayer
- `H_post ∈ R^{1×n}`：sublayer 输出广播回各流时的缩放
- `H_res ∈ R^{n×n}`：约束在 **Birkhoff 多胞形**（双随机：非负、行和列和均为 1），保证谱范数 ≤1、层间复合仍双随机，从而恢复恒等映射行为

三个映射由一次合并投影动态生成：`q = RMSNorm(vec(X))`，`[p_pre; p_post; vec(P_res)] = W_mHC q`；`H_pre = σ(α_pre p_pre + b_pre)`，`H_res = Sinkhorn(exp(α_res P_res + B_res))`。**投影输出、映射 logits、Sinkhorn 迭代全在 FP32。**

**Motif 3 改的就一处**：原始 post-mapping 是

```
H_post^original = 2 σ(Z_post)          值域 (0, 2)
```

logits 接近 0 时给出 identity-scale，看起来合理；但**大于 1 的映射值会在连续残差块里反复放大 sublayer 输出**。在 Motif 3 的规模和深度上，这导致激活离群值渐进累积。改成时间相关的乘子：

```
H_post^(t) = s_t σ(Z_post),      s_t : 2 → 1（预训练期间渐进退火）
```

值域从 `(0,2)` 过渡到 `(0,1)`。原文的理由写得很清楚：**保留早期优化行为，同时移除训练后期的持续放大，且不对前向激活做硬截断。**

### 2.4 Expert-Specific PolyNorm

把 SiLU 门控换成**每个专家独立学习多项式系数和偏置**的 PolyNorm：

```
PolyNorm_i(z) = Σ_{n=1..3} a_{i,n} · z^n / RMS(z^n) + b_i
a_{i,n} = σ(ã_{i,n}) ∈ (0,1),    b_i = clip(b̃_i, −0.5, 0.5)
```

动机：路由让不同专家看到不同的 token 分布，所有专家共用同一个非线性响应是不必要的约束。归一化在 token 维（hidden 维上）算，所以对专家分组不敏感——只有系数和偏置随专家变。sigmoid / clip 参数化是为了限制激活离群值。

**证据**：用 gate 矩阵的**有效秩** `r_eff(W) = Σ_j σ_j²/max_j σ_j²` 度量学到的门控方向多样性。`rank(W)=512`，所以有效秩上限 512。图 6(a) 显示 Expert-Specific PolyNorm 在测量的各层上都保持比 SwiGLU 更高的有效秩。同样是 ~10B 对照实验，且**只报了有效秩这个代理指标，没报 loss 或下游分数**。

> 工程代价在 §3.4：因为专家跑 grouped GEMM，激活作用在**按专家 pad 过的 token buffer** 上，pad 行不能进任何 per-group 统计。他们写了一个 pad-aware 融合 kernel，吃 per-expert group offsets 和 lengths，保证 pad 行既不进归一化统计也不被写回垃圾值。这个细节任何人复刻 grouped-GEMM 上的 per-expert 归一化都会撞到。

## 3. 训练系统（§3，本篇核心）

### 3.1 并行布局

| 维度 | 取值 | 理由 |
|---|---|---|
| EP | **8**（最内层） | 锁在单个 8-GPU 节点内，dispatch/combine 走 NVLink；实现用 **HybridEP** |
| DP-shard | 8 | FSDP 参数分片 |
| DP-replicate | 剩余 rank | — |
| PP | **不用** | 专家+参数分片 + 优化器状态 CPU offload + B200 显存，314B 能装下 |
| CP | 8，**仅长上下文阶段** | 复用与 EP 相同的 8-rank 进程维 |

"MoE 层锁节点内 + 不用 PP"这个组合与 [MegaScale-MoE](./megascale-moe.md) 的核心判断一致。CP 复用 EP 的进程维是个漂亮的小设计——它让 §3.5 里 Ulysses 的 all-to-all **只走节点内 NVLink**，这是他们敢接受"all-to-all 无法与 attention 重叠"这个代价的关键前提。

### 3.2 显存与重计算（§3.1）

**算子级选择性重计算**：每个 transformer block 包一个 checkpoint region，backward 时按 per-operator 策略重放 forward。多数中间量重算，少数保留——包括**用于导出低精度缩放因子的归约输出**（这条很实用：scale 的归约结果重算会引入不一致）。

**MoE 路径的两个耦合优化**：

1. **显存高效 permutation**（来自 Megatron-Core，[2603.07685](https://arxiv.org/abs/2603.07685)）。常规做法是 routing weight 在专家 down-projection **之后**乘：`p_i W_2^(i) h_i`，这要求完整专家输出留着算 router 权重梯度。因为他们的 down-projection 是无 bias 的线性映射，改成数学等价的 `W_2^(i)(p_i h_i)`——**把 routing weight 折进 permuted 激活**。于是 router 权重梯度可以从专家激活拿到，逆 permutation 退化成 routed 贡献的无权求和。
2. **把这个融合的 unpermute+combine 暴露给选择性 checkpoint 策略并保留其输出**。post-combine 张量比它的 top-k 展开输入小约一个 routing fan-out（这里 8×），所以保留它显存代价不大，但**保证 backward 重算既不保留展开后的专家激活、也不重放 combine all-to-all**。dispatch all-to-all 仍会重放以重建专家输入，但**复用 forward 时缓存的通信状态和路由元数据**，避免额外 host 同步。

> 这两条加上 §2.4 的 pad-aware 融合，构成一套完整的"grouped-GEMM MoE 层怎么做选择性重计算"的配方。对照 [MoEBlaze](./moeblaze.md) 走的是"元数据索引不物化 routed token"的路子，目标相同、手段不同。

**256K 阶段改为对每个 transformer block 全量重计算**——选择性重计算下激活显存仍然占主导，直接用算力换最小激活占用。

**输出投影与 loss**：220K 词表下，仅在预训练序列长度就需要 >13 GB/microbatch（BF16）来物化 logits。用 **Liger fused linear cross-entropy**（[2410.10989](https://arxiv.org/abs/2410.10989)）分块处理 hidden state，forward/backward 都不物化完整 logits 张量。kernel 返回未归一化的 token 级 loss 和，他们再除以跨 DP rank 和梯度累积 microbatch 聚合的**全局有效 token 数**——让 loss 和梯度尺度对 batch 切分方式不变。

### 3.3 三项 FSDP 改造（§3.1，我认为是全篇最可直接搬的部分）

| 改造 | 原生 FSDP 的问题 | 他们的做法 |
|---|---|---|
| **梯度 Reduce-Scatter 重叠** | 每个 MoE block 贡献最多 3 个 FSDP 参数组（dense 块参数 / MoE 非专家参数 / routed 专家，后者在独立 device mesh 上分片）。默认调度只跟踪**一个** in-flight RS，导致进入 post-backward 阶段的组必须等前一个操作完成，专家组和 dense 组的归约被串行化，反复 stall backward | 维护**有界的 in-flight RS 窗口**；post-backward hook 入队不立即等待，完成事件在 backward 末尾一次性 drain。窗口上界同时限住 staging buffer 显存 |
| **跨 microbatch 梯度同步** | — | 每个 microbatch 在 DP-shard 维做 RS，梯度累积在**分片的 FP32 buffer**里而不是完整非分片张量；DP-replicate 维的 All-Reduce 对中间 microbatch 抑制，只在最后一个 microbatch 对已累积的梯度分片发一次。**副本间梯度流量降低一个梯度累积因子** |
| **消除专家权重 All-Gather** | 默认 reshard-after-forward/backward 策略下，专家权重每 MoE 层每 microbatch 要 gather **两次** | 梯度累积期间中间 microbatch backward 后不 reshard 专家参数，让上一个 microbatch gather 的副本被下一个复用，reshard 推迟到最后一个 microbatch。**选定的 22/51 个 MoE 层**（按 per-GPU 显存预算挑）的专家在 forward 与 backward 之间也保持 unsharded，权重收集降到**每 optimizer step 恰好 1 次 All-Gather**。dense 参数保持标准 reshard 策略 |

第三条的**显存论证很值得学**：这个保留不增加峰值显存，因为 backward 推进时每个 checkpointed 层的保存激活在该层梯度算完后释放，而**保留的 MXFP8 专家权重比同一时间窗内释放的激活更小**，于是常驻参数副本占的是 backward 腾出来的显存而非抬高峰值。另外——专家权重以 MXFP8 gather（FP32 master weight 在发送侧量化，All-Gather 量相对 BF16 减半，gather 到的副本直接给 grouped FP8 GEMM 用、不反量化），所以一份非分片常驻副本只需约 BF16 对应物一半的显存。

**这里有一句话值得单独摘出来**（§3.1 原文大意）：FSDP 通常靠预取把参数 All-Gather 藏在计算后面，但**这个重叠不是免费的，因为通信 kernel 会消耗 SM 资源、显存容量和互连带宽，从而拖慢与之重叠的计算**；在他们的规模上，消除这些集合通信比仅仅隐藏它们更高效。

> 这是本篇与我们知识库最强的一次交叉验证。[Tessera](./tessera.md)（OSDI'26，阿里生产环境）量到 **EP 通信 kernel 占 ~20 SM，导致 10–20% 减速**，并因此在生产里否掉了 Comet 式融合。两篇从完全不同的角度（一个是 FSDP 参数 AG，一个是 EP dispatch/combine）得出同一个结论：**"重叠"在大规模下不是零成本，有时正确答案是消除通信而不是隐藏它。**

### 3.4 低精度配方（§3.2）

| 状态 | 精度 |
|---|---|
| master weights / main gradients / router logits / Muon 优化器状态 | **FP32** |
| MoE 专家权重、激活、dispatch 通信 | **MXFP8** |
| 梯度同步（跨 rank 传输） | **BF16**，本地归约在 FP32 |
| grouped GEMM 输出 | BF16（**不额外下转**给 combine 路） |

四个具体决策：

1. **MXFP8 grouped GEMM 后端选 DeepGEMM。** 试过 TransformerEngine 和 torchAO，在他们的模型规模和 Blackwell 上 DeepGEMM 最快。
2. **MoE 激活预量化**：不在 grouped GEMM 前量化，而是**在 EP dispatch 之前**量化，于是 dispatch 通信本身也走 MXFP8。三重收益：dispatch 通信量减半、量化成本降到 `1/top-k`（这里 1/8，因为量化一次而不是每个选中专家各来一次）、且**不引入额外精度损失**（本来就要量化）。反方向不做：grouped GEMM 输出已经是 BF16，强行下转只是多一次量化。为支持这条，他们改了 HybridEP 让它原生处理 MXFP8 张量的 dispatch。
3. **梯度同步用 BF16 传、FP32 归约**：main gradient 是 FP32，常规集合通信会按 FP32 传。改成——RS 时每个 rank 按目的地切分梯度、以 BF16 经 all-to-all 交换、**本地用 FP32 求和**；AR 用同样的分片归约再 All-Gather 复制回去。
4. **row→column MXFP8 本地转码**：backward 需要 row-wise 和 column-wise 两份 MXFP8 权重（分别给 forward 重算和梯度计算）。FSDP 下这本来要 All-Gather 两份。改成**只 All-Gather row-wise，column-wise 由 gather 到的表示本地构造**，通信减半。代价：两种 layout 的缩放轴不同，转换需要重量化、引入额外舍入误差。原文说**观测到这个误差对训练稳定性和最终模型质量没有可测量的影响**——注意这是定性陈述，没给对照曲线。

### 3.5 Muon（§3.3）

Muon 作为主优化器（矩阵形参数），embedding / 输出投影 / 向量形参数用 AdamW。三个系统挑战：

**(a) Parallel Muon 扩到 MoE。** 基础版来自 Motif 2（[2511.07464](https://arxiv.org/abs/2511.07464)）：不像 Distributed Muon 那样在每个参与 rank 上重建完整矩阵并冗余执行 Newton–Schulz，而是**每个参数指定单一 owner rank**；参数按正交化计算代价排序后 round-robin 分配以均衡负载；梯度分片经 all-to-all 收到 owner、在那里正交化、再经第二次 all-to-all 分发回去；按 chunk 处理并流水化 gather/compute/scatter。通信和 Newton–Schulz 都用 BF16。

MoE 的扩展很干净：**专家权重已经按专家维被 EP 和专家 DP 分片，所以每个 rank 本来就持有其本地专家子集的完整矩阵，gather 和 scatter 两个阶段都不需要**。每个专家是独立的 Muon 正交化单元，更新按其权重矩阵形状归一化。他们**把本地持有的全部专家权重 stack 起来，对整个 batch 做 batched GEMM 形式的 Newton–Schulz**，而不是对每个专家单独调二维 kernel。专家更新因此零优化器通信。再加一层：**这个 batched 专家计算与 dense 参数流水的第一次 all-to-all gather 重叠**。

> 与 [DMuon](./dmuon.md) 的 owner-centric Muon 是同一思路的独立实现；DMuon 多了 Gram SYRK NS 和 MILP 负载均衡，Motif 3 用的是按代价排序 round-robin。Motif 3 多的是**MoE 专家 batched 路径**这一块，DMuon 笔记里没有。

**(b) QK-Clip 的两个工程点。**

第一个是**怎么便宜地监控 `S_max^h`**（每个 head 的 pre-softmax logit 最大值）。这个值是 pre-softmax attention score 矩阵的最大值，**FlashAttention 不物化它**，显式重算 `QK^⊤` 等于每层加一趟二次开销。他们**扩展了 FlashAttention-4 forward kernel 直接吐出每个 head 的最大值**：online softmax 本来就维护 running row maxima，把它归约成每 head 一个标量，对 attention forward 的额外开销可忽略。这些值跨 microbatch 和 DP rank 做 max 归约，逐层供给优化器，优化器在参数更新后立刻缩放其本地权重分片。**每 10 步监控并 clip 一次**，`τ=100`（主预训练）/ `τ=200`（256K 长上下文阶段）。

第二个就是 TL;DR 第 2 条的非对称劈分。完整规则：

| 权重行 | 缩放因子 |
|---|---|
| non-rotary query | `γ_h^{1-r}`，其中 `r = 1/(1+√G)` |
| non-rotary key（共享） | `γ_h^r` |
| rotary query | `γ_h`（全额） |
| rotary key（跨 head 共享） | **不动** |
| value | **永不修改** |

`γ_h = τ/S_max^h`。这样每个 logit 分量都恰好被衰减 `γ_h`，但非对称劈分防止反复 clip 压塌共享 K。多个 query head 共享一个 key 投影时，共享 K 行用组内**最小的 `γ_h`**（即最强的所需 clip）。

**(c) 优化器状态 CPU offload。** Muon 每参数只有一个 momentum buffer。每次 optimizer step 后 offload 到 host，下次需要时再 load。状态张量打包进扁平的 pinned host buffer，按张量在专用 stream 上传输，offload 和 reload 都避免临时的 device 侧 staging buffer。

**reload 的时机是这里最巧的一点**：朴素调度会把 H2D reload 放在 optimizer step 前的关键路径上。他们改成**从 backward 期间每个层边界的 activation-checkpoint post-forward hook 发起 reload**——在该层参数 All-Gather 完成之后、backward GEMM 开始之前。于是专家优化器状态随 backward 计算逐层 reload，传输能与 backward GEMM 重叠；其余状态在 optimizer step 前批量 reload。一个 hook 点同时覆盖外层的 activation-checkpointed block 和内层的专家 FSDP 组，save/load 顺序防止 checkpoint save 触发多余的 offload–reload 循环。

### 3.6 Kernel 融合（§3.4）

| Kernel | 做什么 | 关键约束 |
|---|---|---|
| **Pad-aware fused GroupedPolyNorm** | 在 grouped、padded buffer 上一趟算完整个 multiply + PolyNorm 链；提供 forward 和**闭式 backward**；作为 eager 模块的 drop-in（TP/EP 分片与 checkpoint key layout 不变） | 吃 per-expert group offsets/lengths，pad 行既不进归一化统计也不写回垃圾值 |
| **mHC kernels** | 把 mHC 的 per-step "apply" 从 matmul 形式改写成**逐元素乘 + 乘-加和**形式，配一个 Triton residual kernel；整层支持 full-graph `torch.compile` | Sinkhorn-Knopp 与相关投影在 FP32 |
| **Fused PolyNorm–FP8（推理）** | PolyNorm 与 FP8 量化融在一起 | decode 强 bandwidth-bound，融合省掉一次 hidden state 的读写；**原文明确说这个融合在推理侧收益不成比例地大于训练侧** |

> mHC kernel 这条与我们 Primus 的实现路线完全撞上，而且**我们做得更远**——见 §7.2。另外原始 mHC 论文用 TileLang 实现大部分 kernel，Motif 3 用 Triton + torch.compile，这是两条不同的工程选择。

### 3.7 长上下文 CP（§3.5，全篇最完整的一节）

记号：`L` = packed 序列长度、`W` = 窗口、`P` = CP degree。配置 `L=256K`、`W=128`、`P=8`。

**为什么必须逐层选算法：**

| | full attention 层 | sliding-window 层 |
|---|---|---|
| **选择** | **Ulysses** | **window-aware Ring** |
| 计算量 | `Θ(L²)` | `Θ(LW)` |
| 该算法通信量 | `Θ(L)`，被 `Θ(L²)` 计算摊薄 | 单次 halo，`Θ(W)` |
| 排除另一个的理由 | Ring 需要逐迭代动态块调度才能均衡 document mask 的负载 | Ulysses 的 all-to-all 仍是 `Θ(L)`，小 `W` 下通信/计算比很难看 |

- **Ulysses**（[2309.14509](https://arxiv.org/abs/2309.14509)）用 all-to-all 转置序列维和 head 维。`h_q=80`、`h_k=16` 都能被 `P=8` 整除，所以每个 rank 处理**完整 packed 序列的一个等量 head 子集**，从构造上消掉了 rank 级不均衡。代价是这个转置与 attention 之间的数据依赖**阻止 all-to-all 与 attention 计算重叠**（原文说包括他们自己在内考察过的实现都如此）。他们接受这个代价的两个理由：`Θ(L)` 通信被 `Θ(L²)` 计算摊薄；且因为 CP 组与 EP 组共享同一个 8-GPU 节点，**all-to-all 只走节点内 NVLink**。
- **window-aware Ring**：`W ≤ L/P` 时每个 rank 只需要前一个 rank 的 KV 分片的**尾部 `W` 个 token`**。这一次 halo 交换替代最多 `P-1` 次 rotation step，rank 平均收到的 KV 量从 `Θ(L)` 降到 `Θ(W)`——在 `W=128` 的工作点上是解析的 **`L/(2W) = 1,024×`**。固定窗口同时限住 rank 级负载波动，所以这些层不需要输入相关的块调度。

**通信量的诚实记账（附录 B）**，这段写得比正文更值得读：

- 论文**明确说通信量不是 Ring vs Ulysses 的延迟代理**，两个原因：均衡的 Ring rotation 能与计算重叠而 Ulysses 的 all-to-all 不能；且 Ring 的通信量**不是每个 rank 相同**——带 early exit 时 rank `r` 收 `r` 个 KV 分片，而 Ulysses 的 all-to-all 在每个 rank 上相同，**所以取平均还是取最忙 rank，两者的比较方向会反转**。
- `C_Ring / C_Ulysses = P(h_k d_qk + h_v d_v) / (2[(h_q+h_k)d_qk + (h_v+h_o)d_v])`。因为 `h_o = h_q`（构造上）且 `h_v = h_k`（他们的配置），full attention 下比值精确化简为 `P/[2(1+h_q/h_k)] = 2/3`——**rank 平均意义上 Ring 比 Ulysses 少搬 33% 数据**。但最后一个 rank 的比值是 `4/3`，**序关系反过来**。`d_qk=192` 与 `d_v=128` 的不等在这些 head 数等式下恰好抵消。
- 所以他们**没有把 full-attention 的算法选择建立在通信量对比上**，而是建立在实测行为上：「在我们的长上下文运行中，Ulysses 下的 full-attention 层没有成为 step 级 straggler，而 document-masked 的 Ring 变体成为了。」
- `W=128` 工作点上 window-aware Ring 与 Ulysses 的差距是 **1,536×**（图 4c）。

**attention 负载感知重排**（这是我认为最容易被低估的一条）：

document-masked packing 把多个文档拼进固定长度序列并禁止跨文档 attention。标称长度相同，但文档边界会造成**显著不同的 attention 工作量**——图 4a 那个含 473 个文档的 packed 序列里，**rank 7 的 attention FLOPs 是跨 rank 均值的 3.54×**。同步训练下这直接变成 step 级 straggler。

代价代理（用未被 mask 的 attention score 条目数近似 FLOPs）：

```
n_full(d) = d(d+1)/2
n_W(d)    = Σ_{q=0}^{d-1} min(W, q+1) = W(W+1)/2 + (L−W)W      （单文档闭式）
Ĉ({d_i})  = Σ_i [ ¼ n_full(d_i) + ¾ n_W(d_i) ]                  （¼ full / ¾ SWA 层配比）
```

固定 `W` 和相同标称长度下，SWA 项几乎与 token 数线性，**跨文档布局的差异主要由二次的 full-attention 项驱动**。

调度算法：每个副本预取一个 optimizer step 所需的序列，形成 `N_DP × N_acc × B` 个 packed 序列的全局池（他们 `B=1`）。副本们 all-gather 每序列的 `Ĉ`；给定相同输入，**每个副本独立跑同一个确定性的 LPT（longest-processing-time）贪心**——按估计代价降序排序，依次分配给当前负载最轻的 replica–microbatch 槽位。确定性 tie-break 让每个副本产出相同分配，**不需要额外广播**。然后经**固定大小的 all-to-all** 交换序列（所有 packed 序列标称长度相同，所以尺寸固定）。

正确性论证：与 CP 正交（CP 在每个 DP 副本内部应用）；因为同一组序列以同样权重贡献给 optimizer step，**累积梯度在浮点归约顺序意义下被保持**，改变的只是到 DP 副本和 microbatch 的分配。

## 4. 预训练稳定化（§4.2）

### 4.1 分层的专家均衡

| 机制 | 细节 |
|---|---|
| FP32 路由 | router 分数和 expert-selection bias **全程 FP32**，避免离散选择决策的数值误差 |
| sigmoid 路由 + auxiliary-loss-free bias | 沿用 DSv3 / [2408.15664](https://arxiv.org/abs/2408.15664)；bias 更新系数 `1e-3` |
| 序列级 auxiliary LB loss | sum reduction，默认权重 `1e-4`；**层 2 提到 `2.4e-4`、层 3–5 提到 `2.0e-4`**（早期 MoE 层更易失衡） |
| 衰减 router 噪声 | FP32 router logits 加高斯噪声 `ε ~ N(0, σ_t²)`，`σ_t` 走 cosine 衰减到 `σ_min`。图 6(b)：加噪声后每专家最大 token 负载明显更早落到中位负载区间；且**训练初期 loss 下降更快** |
| FFN 幅值软正则 | `L_ffn-clip = w_ffn · mean[ReLU(|y| − τ)²]`，`w_ffn = 2e-4`、`τ = 128`（**最后一层 `τ = 1024`**） |

**FFN 正则那一段的推理值得完整记下来**，因为它是对一个流行做法的反对：gpt-oss-120b 的参考实现在 SwiGLU 内部硬截断（gate 输入上截、linear 输入双侧截）。Motif 3 **拒绝这种前向干预**，理由是激活离群值不必然是无意义的数值伪影——引用 [2601.22966](https://arxiv.org/abs/2601.22966)（attention sink 与 residual sink 的统一视角）指出极端激活会与归一化交互提供有用的重缩放行为，直接截断会同时损害训练稳定性和模型性能；硬截断还丢弃幅值信息，且激活越过阈值后梯度为零。

**而且他们诚实报告了这个软惩罚的局限**：引入后不再观察到孤立的极端离群值，但**激活增长变成分布在更广的一组通道上，FFN 输出整体 RMS 在训练中继续上升**。也就是说这个辅助 loss 抑制了集中的尖峰，但本身**并不能阻止全局激活尺度漂移**——后者要靠逐层 RMS 统计单独跟踪。这种"我的方法只解决了一半"的陈述在技术报告里不常见，可信度加分。

### 4.2 专家健康度指标（表 3，我认为这是全篇最可复用的单块内容）

在与专家利用率热图相同的间隔采集，且**只在 step 的最后一个 microbatch 上采**以限制监控开销；跨 DP worker 聚合，记录成一张 layer × metric 表，另附跨全部 MoE 层的全局 min/max 供时序告警。

| 指标 | 定义 | 健康阈值 | 指示的失效模式 |
|---|---|---|---|
| **dispatch min/median** | `R_dispatch = min_i N_i / median_i N_i` | **>0.7 健康，<0.3 可能饿死** | 路由失衡导致的饥饿或死专家 |
| 最大专家 token 数 | `max_i N_i` | — | 流量集中、专家过载 |
| **输出权重 min/median** | `‖W_2^(i)‖` 的 min/median | **>0.8 健康，急跌到 <0.5 = 专家在死** | 隐性塌缩：专家仍收 token，但输出投影贡献越来越小 |
| **routed/shared RMS 比** | `RMS(y_routed)/RMS(y_shared)`，其中 `y_routed = y_final − y_shared` | **>~0.3 表示 routed 有实质贡献** | shared expert 独揽工作 / routed 塌缩 |
| **专家输出余弦相似度最大值** | 每专家对其 routed token 的输出均值向量，两两余弦相似度 | **<0.5 健康，>0.8 严重冗余** | 功能塌缩：名义不同的专家学成同一个函数 |
| abs-max / RMS | routed 与 shared 各自 | **>~10× 是离群值预警** | 激活离群值、数值不稳前兆 |

三个设计理由值得注意：

1. `R_dispatch` 用 **min/median 而不是 max 或方差**——它直接度量"最不被用的专家相对典型专家如何"，且**对单个异常热门专家鲁棒**。
2. **dispatch 均衡不足以判断专家是否功能健康**：一个专家可以持续收 token 而其对残差流的贡献趋于消失。所以要监控 `W_2^(i)` 的范数——输出投影"特别有信息量，因为它作用在 PolyNorm 之后，直接控制写回残差流的尺度"。
3. **功能多样性用输出激活而不是参数向量度量**：这个基于激活的统计直接检测在各自分配的 token 上行为相似的专家，而**高维展平权重矩阵之间的余弦相似度一般没有信息量**。

### 4.3 训练配置（§4.3）

| 项 | 值 |
|---|---|
| 全局 batch | 最多 **75M tokens** |
| 上下文课程 | 4K → （LR decay 阶段）32K → （专门阶段）256K |
| LR 调度 | WSD：`3e-4` → 稳定阶段跑完大部分预训练 → decay 到 `7.5e-5`（同时 4K→32K）→ 256K 阶段继续降到 `3e-5` |
| RoPE 外推 | full-attention 层用 **DeepSeek-YaRN，4K → 256K，扩展因子 64**；**SWA 层保持局部 RoPE 不做长程外推** |
| 长上下文数据 | 重构约 5% 的整体预训练语料作为专门长上下文数据集 |
| 推理类数据占比 | **<5%**，明确为了不让 base model 分布过度集中在推理轨迹上 |
| MTP loss 权重 | 0.2 |

**tokenizer**：SuperBPE 两阶段（阶段 1 常规 BPE，阶段 2 在 pre-tokenization pattern 里加"重复的空格分隔字母串"，允许一个 pre-token 单元跨多个空格分隔的词，例如英文 "of the"、韩文 "수 있다"）。完整正则在附录 A.1。压缩率（bytes/token，23 MB 多语言多领域探针）：

| Tokenizer | Vocab | en | fr | ko | ja | zh | code | math |
|---|---|---|---|---|---|---|---|---|
| **Motif** | 220,160 | **5.68** | 4.02 | **5.31** | 3.87 | 3.51 | **4.07** | **4.55** |
| Qwen3.5 | 248,066 | 4.51 | 3.91 | 4.03 | **4.22** | **4.13** | 3.68 | 3.54 |
| Gemma-4 | 262,144 | 4.61 | 3.96 | 3.73 | 4.18 | 3.67 | 3.57 | 3.58 |
| DeepSeek-V4 | 129,280 | 4.73 | 3.76 | 3.34 | 3.79 | 4.20 | 3.80 | 3.86 |
| gpt-4o (o200k) | 200,000 | 4.70 | **4.14** | 3.65 | 3.43 | 3.31 | 3.92 | 3.75 |

据此他们主张 12.5T Motif token **相当于用任一对比 tokenizer 计量的 15T+ token**。原文自己标明这是**基于探针的估计，不是对预训练语料的完整重 tokenize**。中日文上不占优。

## 5. Post-training（§5，简述）

三阶段：general SFT → 7 个专家教师 → **MOPD（Multi-teacher On-Policy Distillation）**。

一个流程上的讲究：先训一个**初步 SFT 模型专门用来识别能力相关的失败模式并生成针对性监督数据**（尤其是 agentic 轨迹里易错的决策点），但**这个初步模型只用于构造数据，不作为最终 student 的初始化**——构造完整 SFT 语料后**从预训练 checkpoint 重新开始**训 general SFT student。

**RL 基础设施**（§5.2.1，对做框架的人有参考价值）：模型在自研 TorchTitan 栈上预训练和 SFT，但 RL 想用 NVIDIA NeMo RL。NeMo RL 原生后端是 Megatron Core 和 AutoModel(DTensor)，**在原生后端里重实现 Motif 3 等于引入第二份模型实现、有与预训练/SFT 用的模型发散的风险**。他们的做法是**把 TorchTitan 实现成 NeMo RL 的外部训练后端**——在 import 时挂上所需的 trainer 和 model 接口，**不修改上游 NeMo RL 代码库**。于是同一份实现和 checkpoint 可以直接被 NeMo RL 的 GRPO 优化、被 NeMo Gym 的 verifier 评测。

**异步 GRPO**：生成和训练在**不相交的 GPU 池**上作为独立 Ray actor；异步 collector 持续发 vLLM rollout，learner 从 replay buffer 取轨迹训练。每条轨迹标记产生它的策略权重版本和它有效的 learner step；**轨迹年龄上界 1 个 optimizer step**，超界丢弃，目标版本 batch 未就绪时等待。权重更新**不先排空所有 pending rollout**：生成 worker 在一个共同的 wave 边界短暂暂停，**in-flight 请求及其 KV cache 在暂停期间保持驻留**，避免集合通信排序死锁同时让生成在 refit 后立刻恢复；他们**保留现有 KV cache 而不在每次更新后重算**。异步产生的轨迹会偏离当前 learner 策略，所以缓存其 behavior-policy log prob 并做 token 级重要性采样校正，**重要性权重截断到 `[0.2, 5.0]`**。

**7 个教师**：13 个 verifier 域分成 6 个 GRPO job + 1 个 SFT 训的软件工程教师。**不把单一策略同时对所有 verifier 优化**，理由是 verifier 延迟和奖励方差跨域差异大，混合不相关的奖励面会把本来独立的失败模式耦合起来。教师覆盖：agentic tool use / professional work / software engineering / long-context reasoning & abstention / mathematics / code and science / chat。

配方要点：token 级 GRPO（基于 DAPO），**非对称 ratio clipping `[0.8, 1.28]`**（下 0.2 / 上 0.28），组内奖励归一化 + leave-one-out baseline + advantage clip `[-4, 4]`，全部教师 LR `3e-6`，batch 256–1,152 轨迹，rollout 长度 8K–192K，无熵正则；**离线 prompt 过滤**（4–8 次 rollout 估经验通过率 `p`，保留 `0 < p ≤ 0.8`）替代在线 dynamic sampling；序列打包用改良 first-fit decreasing。

**MOPD**（§5.2.4）：student 生成 on-policy 轨迹，按域路由到对应专家教师，**关掉环境评分**（verifier 奖励不进优化目标）。实现能蒸馏全词表分布但**本文没用**——每个 student 采样的 token 只取被路由教师赋给该 token 的**标量 log 概率**。

```
w_t = π_old(y_t|·) / π_gen(y_t|·);   w̃_t = w_t if 0.5 ≤ w_t ≤ 5.0 else 0     （ICE-POP 过滤）
d_t = sg[ log( π_T(x)(y_t|·) / π_old(y_t|·) ) ]                              （detached OPD 信号）
L_MOPD = − E_t[ w̃_t · d_t · log π_θ(y_t|·) ]
```

环境奖励和参考策略 KL 项都省掉。global batch 512 条 on-policy 轨迹、每 prompt 1 次生成、Muon 无 weight decay、LR `5e-6 → 3e-6`、最大序列 163,840。

**贯穿教师训练和 MOPD 全程冻结**：shared-weight MTP 参数、MoE router 权重、expert-selection bias。这保住了 SFT 学到的路由配置和辅助预测头；他们报告 MTP draft token 接受率在训练过程中无可测退化，且**最终 MOPD 模型的 MTP 权重与初始化 MOPD 的 student 完全相同**。

> 「RL 和蒸馏阶段冻结 router」这个决定，对我们做 MoE 后训练框架是个值得记下的先例：它把 RL 期间的专家负载分布变成静态的，**从系统角度看这让 EP 通信形状在整个 RL 阶段可预测**。论文没从这个角度论证（他们的理由是保住能力），但这是个免费的系统性质。

## 6. Experiments

### 6.1 Base model（表 4）

| MMLU 5-shot | MMLU-Pro 5-shot CoT | ARC-C 25-shot | WinoGrande 5-shot | HellaSwag 10-shot | PIQA 0-shot | GSM8K 8-shot CoT | MATH 4-shot CoT | HumanEval 0-shot | MBPP 3-shot |
|---|---|---|---|---|---|---|---|---|---|
| 86.20 | 68.56 | 94.71 | 80.90 | 88.30 | 85.14 | 93.93 | 70.58 | 73.70 | 84.60 |

原文明确说**这些只用于刻画预训练获得的能力，不试图与用可能不同 harness 和 prompting 协议评测的其他模型直接比较**。

### 6.2 最终模型（表 6）

Motif 3 全部评测用 `temperature=1.0`、`top-p=0.95`、最大序列 262,144。**基线分数取自各 benchmark 排行榜的报告值**，Motif 3 自己在内部评测。

| Benchmark | **Motif 3 314B-A13B** | MiniMax-3 428B-A23B | GLM-5.1 744B-A40B | Kimi-K2.6 1T-A32B | Qwen-3.7 Max | DS-v4-Pro 1.6T-A49B |
|---|---|---|---|---|---|---|
| *Agentic* | | | | | | |
| GDPval-AA v2 | 38.7 | **44.4** | 37.8 | 34.4 | 39.0 | 40.2 |
| τ²-Bench Telecom | 94.7 | 88.9 | **97.7** | 95.9 | 94.7 | 96.2 |
| τ³-Banking | **35.3** | 15.3 | 13.6 | 23.3 | 12.0 | 30.1 |
| ITBench-AA | **51.5\*** | – | 40.3 | 31.2 | 42.5 | 38.3 |
| *Coding* | | | | | | |
| SWE-bench Verified | 76.2 | 75.0 | 76.4 | 76.2 | **80.4** | 77.4 |
| Terminal-Bench 2.1 | 74.9 | 65.2 | 61.8 | 65.9 | **75.0** | 64.0 |
| SciCode | 40.6 | 45.4 | 43.8 | **53.5** | **53.5** | 50.0 |
| *Reasoning & Knowledge* | | | | | | |
| IMO-AnswerBench | 83.2 | – | 83.8 | 81.8 | **90.0** | 89.8 |
| Apex Shortlist | 75.5 | – | 71.1 | 77.4 | 44.5 | **85.8** |
| GPQA Diamond | 83.4 | **92.9** | 86.8 | 91.1 | 92.4 | 88.8 |
| HLE | 37.0 | 39.0 | 30.1 | 37.5 | **41.4** | 37.5 |
| CritPt | 6.6 | 3.7 | 4.6 | 8.0 | 11.4 | **12.9** |
| AA-Omniscience Accuracy | 30.1 | 16.7 | 23.7 | 32.6 | 31.0 | **42.9** |
| AA-Omniscience Non-Hallucination | 71.6 | **81.6** | 70.1 | 59.5 | 74.0 | 5.9 |
| *Long Context & IF* | | | | | | |
| AA-LCR | 72.3 | **80.3** | 68.0 | 76.7 | 75.0 | 70.0 |
| IFBench | 78.2 | **82.9** | 76.3 | 76.0 | 79.1 | 76.5 |

`*` = 仅在公开子集上评测。

**读法**：Motif 3 是这张表里**总参最小的模型**（314B vs 428B / 744B / 1T / 1.6T），激活参也最小（13.2B vs 23B / 40B / 32B / 49B）。在这个前提下：

- **强项集中在 agentic / 终端**：τ³-Banking 35.3 是表中最高（第二名 DS-v4-Pro 30.1，MiniMax-3 只有 15.3）；ITBench-AA 51.5 是可得结果中最高；Terminal-Bench 2.1 74.9 仅次于 Qwen-3.7 Max 的 75.0。这与 MOPD 的目标一致（把 agentic tool use、professional work、SWE 等教师合并进一个 student）。
- **弱项也很明确，作者自己点出来了**：SciCode 40.6 和 CritPt 6.6 明显低于最强模型，"表明在科学编码和专门科学推理上仍有改进空间"。GPQA Diamond 83.4 是这组里最低。
- **AA-Omniscience 上的取舍是有意的**：accuracy 30.1 低于最好的模型，但 non-hallucination 71.6 属于表中较高——这与他们专门训了一个"long-context reasoning & abstention"教师、并用四档评分阶梯（correct 1 / partial 0.666 / not attempted 0.333 / incorrect 0）**明确奖励弃答优于猜测**直接对应。注意 DS-v4-Pro 那一格的 5.9 看起来异常，用这张表时要小心。

## 7. Limitations

### 7.1 作者声明的

- 训练和评测未覆盖真实任务、领域、语言、交互模式和部署条件的全部多样性，在训练配比中欠代表或评测套件中缺失的任务上表现会有波动。
- **主要是文本模型**，不能直接理解视觉输入。
- 虽然支持长上下文，但许多长程应用要求在**比这里评测的轨迹长得多**的范围上可靠地做状态跟踪、规划、恢复和环境交互。

### 7.2 我认为需要打问号的

1. **§3 没有一个性能数字。** 这是最大的问题。十几项系统优化，没有 MFU、没有吞吐、没有加速比、没有卡数、没有 step time。三条尤其需要数字支撑却完全没有：
   - "在我们的规模上消除专家权重 All-Gather 比隐藏它更高效"——**多高效？**
   - "row→column MXFP8 转换的重量化误差不影响训练稳定性或最终模型质量"——**没有对照曲线**，只是定性陈述。
   - "DeepGEMM 在我们的模型规模和 Blackwell 上优于 TransformerEngine 和 torchAO"——**没有基准表**。
   对比之下 §3.5 反而给了严谨的解析记账（而且附录 B 主动声明通信量不是延迟代理），这个反差说明**不是不会做定量分析，而是选择不公布系统性能数字**。
2. **路由塌缩的人工干预。** §4.2：早期 MoE 层"路由均衡偶尔仍会塌缩。检测到塌缩时，我们立即把受影响层的 router 参数和 expert-selection bias 替换成相邻的、专家利用率稳定的层的"。这是**训练中途人手改权重**。论文没说：触发了几次、在哪些 step、被替换的层是哪些、对最终模型有无影响、这个操作是否可自动化。对任何想复刻的人这是个黑洞——它意味着"这套均衡机制足以稳定训练 12.5T token"这个隐含主张**并不成立**，需要人工救援。
3. **架构消融全在 ~10B 上做。** GDLA 的 9.2% token 效率、PolyNorm 的有效秩、router 噪声的负载曲线，三组都是 ~10B 对照实验。314B 上没有任何架构消融——可以理解（成本），但引用这些数字时不能省略这个限定。
4. **mHC 退火的关键细节缺失。** `s_t: 2 → 1` 是"在预训练期间渐进降低"，但**退火的调度形式、起止 step、是否与 LR 阶段对齐，全都没给**。这恰好是想复刻的人最需要的参数。同样，Sinkhorn 迭代次数也没给（原始 mHC 论文用 `t_max = 20`）。
5. **表 6 的评测口径不可比。** Motif 3 自己评，基线取排行榜报告值。原文在 §4.4 对 base model 明确声明了这个 caveat，但对表 6 只说"为提供语境对比"，没有同等强度的声明。ITBench-AA 那格还只是公开子集。
6. **FFN 幅值正则被作者自己判定只解决了一半问题**（全局激活尺度仍在漂移），但**没有给出他们最终怎么处理这个漂移**——只说"通过逐层 RMS 统计单独跟踪"。跟踪不是解决。

## 8. Our take

### 8.1 立刻可用：Primus DeepSeek-V4 的 mHC post-mapping 应该做退火

这是全篇对我们最直接的一条。本地 `Primus/primus/backends/megatron/core/transformer/hyper_connection.py` 的 `HyperMixer.compute_weights` 里：

```python
# hyper_connection.py L358-360, HyperMixer.compute_weights 的 eager 分支
pre  = torch.sigmoid(pre_logit) + self.eps        # (eps, 1+eps]
post = 2.0 * torch.sigmoid(post_logit)            # (0, 2)  no eps   ← 就是这个 2.0
comb = torch.softmax(comb_logit, dim=-1) + self.eps
```

`2.0` 是硬编码的常数，也就是原始 mHC 的 `H_post = 2σ(z)` 形式。Motif 3 报告的正是这个形式在大深度下的病理：**`>1` 的 post-mapping 会逐层反复放大 sublayer 输出，导致激活离群值渐进累积**。

同样的常数也在 Triton 快路径 `hc_glue_compute_tail_triton` 里（`_hc_glue_enabled()` 分支），所以两条路径都要改。

具体建议：

| 项 | 建议 |
|---|---|
| 参数化 | 把 `2.0` 提成 config 项（例如 `hc_post_scale`，默认 2.0 保持 checkpoint 兼容），Triton kernel 侧接同一个标量参数 |
| 退火 | 加一个 `hc_post_scale_final: 1.0` + 退火区间，由 trainer 逐 step 写入 |
| 验证 | 我们的 V4 跑的层数和 token 量都远小于 Motif 3，**很可能观察不到这个病理**。所以别直接照抄退火，先**加逐层 FFN / sublayer 输出 abs-max 与 RMS 的监控**（正好是 Motif 3 §4.2 那套指标里的 `abs-max/RMS > 10×` 预警），确认在我们的规模上有没有累积趋势 |
| 风险 | 退火改变前向数值，会打破 bit-wise 复现和现有 checkpoint 的等价性。默认关，作为长跑（>1T token）配置的开关 |

注意 `comb` 那行的 `softmax(...) + eps` 与 Motif 3 的 `Sinkhorn(exp(α P + B))` 有个细微差异：softmax 本身等价于 Sinkhorn 的第一次行归一化，所以序列上基本一致，但 `+eps` 会轻微破坏双随机性。Sinkhorn 迭代 20 次后影响应该可忽略——不过既然 Motif 3 明确说"投影输出、映射 logits、Sinkhorn 迭代全在 FP32"（我们也是这样做的），这条一致，不用改。

### 8.2 我们在 mHC kernel 上已经走得比这篇远

Motif 3 §3.4 对 mHC kernel 的全部描述是：把 per-step apply 从 matmul 形式改写成逐元素乘 + 乘-加和形式，配一个 Triton residual kernel，整层支持 full-graph `torch.compile`，Sinkhorn 与投影在 FP32。

我们本地已有的：

| Motif 3 | Primus V4 现状 |
|---|---|
| "Triton residual kernel" | `hc_expand_triton`（`post·out + comb·x` 融合写回）+ `hc_collapse_triton`（`(pre·x).sum(-2)` 融合，省掉 `[..., K, D]` 临时张量与 `K·D` 额外 HBM 流量） |
| "逐元素乘 + 乘-加和形式" | 同一思路，但还多了 `hc_glue_compute_tail_triton`（slice + scale + base + sigmoid/softmax + eps 的 elemwise 尾巴融成一个 kernel） |
| "full-graph torch.compile" | 三条路径：手写 Triton（`PRIMUS_SINKHORN_TRITON`，默认开）> `torch.compile(fullgraph=True, dynamic=True)` > eager |
| — | `fused_rms_norm` 处理 packed `K*D` 轴上的无参数 RMS |

而且我们的 `sinkhorn.py` 注释里记录了一个 Motif 3 完全没提的问题：**eager 路径每次调用发 `1 + 2*(n_iters-1)` 次独立的 FP32 `aten::sum`，在 V4-Flash 生产宽度下每次归约跑在内存受限下界的约 250× 之上**，因为 HIP 默认的 `reduce_kernel<512,1,...>` 是给巨型归约设计的，而 `[1,4096,4,4] → [1,4096,4,1]` 每个输出只有 4 个元素。

**这条值得对外讲。** 原始 mHC 论文用 TileLang，Motif 3 用 Triton + torch.compile，**都没有报告小归约在 ROCm 上的这个病理**。我们有实测归因，而且给出了三档 fallback 的路由策略。如果要写一篇 mHC on ROCm 的工程记录，这是现成的差异化点。

### 8.3 QK-Clip：我们现在关着它，而 Motif 3 给出了打开它需要的两块拼图

本地状态：`Primus/primus/backends/megatron/core/transformer/fla_flash_attention.py` 里 `current_max_attn_logits` 被显式设为 `None`，注释是"为 MLA 可选的 qk_clip 路径保留，在我们的配置里禁用"。`kimi_k3_base.yaml` 里也明确记着 K2 weight clipping"report-only, deliberately not implemented"。

**这正是 Motif 3 §3.3 解决的两个问题**：

1. **为什么拿不到 `S_max`**：FlashAttention 不物化 pre-softmax score 矩阵，显式重算 `QK^⊤` 等于每层加一趟二次开销。Motif 3 的做法是**改 FA forward kernel 直接吐每 head 的最大值**——online softmax 本来就维护 running row maxima，归约成每 head 一个标量，额外开销可忽略。在我们这边对应的动作是改 AITER / CK / Triton FA 的 forward epilogue，**不需要新算法，只需要把已有的 row max 再归约一次并写出去**。这是个小改动，而且它解锁的不只是 QK-Clip——per-head logit 最大值本身就是训练稳定性监控的好信号。
2. **为什么直接开会出事**：对称 `√γ_h` 劈分会把 GQA/MLA 里共享的 K 投影压到零。如果我们在 K3 或 V4 路径上简单打开上游 Megatron 的 `qk_clip.py`（它是 K2 的 QK-Clip），**很可能撞上同一个坑**。`r = 1/(1+√G)` 这个修正连同"rotary K 不动、rotary Q 承全额 `γ_h`、value 永不修改、组内取最小 `γ_h`"的完整规则，是可以直接实现的。

优先级判断：这不紧急（我们目前没有观察到 attention logit 增长导致的不稳），但**如果之后要在 ROCm 上跑 Muon 长跑，这条会从"nice to have"变成必需**——QK-Clip 存在的理由就是控制 Muon 下的 attention logit 增长，而我们已经有 `per_head_muon.py` 了。

### 8.4 AMD / ROCm 移植性盘点

这是我的分析，不是论文内容。Motif 3 的系统栈是**全 NVIDIA Blackwell**，逐项看在 ROCm 上的对应物：

| Motif 3 组件 | ROCm 侧现状 | 缺口评估 |
|---|---|---|
| MXFP8 grouped GEMM（DeepGEMM） | gfx950 有 MX block-scaled MFMA（`mfma_scale_f32_*_f8f6f4`）；torchtitan 已支持 gfx950 mxfp8（#2222）；Primus-Turbo 有 FlyDSL 的 **FP8 grouped GEMM**（PR #384）和 **MXFP8 dense GEMM**（PR #390） | **MXFP8 × grouped 这个交叉点是缺的**。这是 Motif 3 配方里最核心的一块，也是我们最该补的一块 |
| HybridEP（DeepEP `hybrid-ep` 分支） | 该分支用 **TMA 指令**做最小 SM 占用，CDNA 没有 TMA 对应物。我们这边走的是 Primus 的 deepep 路径 | 架构性缺口。AMD 侧的等价目标（最小 SM 占用的 dispatch/combine）要用别的手段达成 —— 这正好是 ROCmoe / MonolithEP 的题目 |
| MXFP8 dispatch（激活预量化） | 与后端无关的**调度决策**，只要 EP 库能搬 MXFP8 张量就能做 | **最便宜的一条，应该直接抄**。量化成本降到 1/top-k + dispatch 量减半，且不引入额外精度损失 |
| FlashAttention-4 吐 per-head max | FA4 是 Blackwell 专用；我们的 attention 走 AITER / CK / FLA | 需要自己改 epilogue，但改动很小（见 §8.3） |
| Liger fused linear CE | Triton 实现，可移植 | 无缺口 |
| mHC kernels | 见 §8.2，我们更完整 | **我们领先** |
| Parallel Muon（含 MoE batched 路径） | Primus 有 `per_head_muon.py`（Kimi K3 路线） | 两条不同的 Muon 变体。Motif 3 的"本地专家 stack 起来做 batched Newton–Schulz、零优化器通信"这块值得单独看，它利用的是**专家已按专家维分片**这个与 EP 无关的事实，在 ROCm 上同样成立 |
| 优化器状态 CPU offload + 从 checkpoint hook 逐层 reload | 通用 PyTorch 机制 | 无架构缺口；`hook 点同时覆盖外层 activation-checkpointed block 与内层专家 FSDP 组`这个细节值得照抄 |
| 逐层选 CP 算法 + LPT 负载重排 | 纯调度层，与后端无关 | **无缺口，可直接实现**；见 §8.5 |

### 8.5 逐层选 CP 算法 + 负载重排：最可移植、最被低估的一条

§3.5 整节没有一行硬件相关代码。它的两个结论都是纯调度层的：

1. **混合 attention（full + SWA）下不存在通吃的 CP 算法**，要逐层选。判据很清晰：SWA 把计算降到 `Θ(LW)` 而 Ulysses 通信仍是 `Θ(L)`，小 `W` 下比值难看；full attention 的 `Θ(L²)` 计算能摊薄 Ulysses 的 `Θ(L)`。
2. **document-masked packing 会造成 3.54× 的 rank 间 attention FLOPs 差异**，用 `Ĉ = Σ[¼ n_full + ¾ n_W]` 做代价代理 + 确定性 LPT 贪心在 `N_DP × N_acc × B` 全局池上重排，靠确定性 tie-break 省掉广播、靠标称长度相同用固定尺寸 all-to-all 交换。

我们现在做长上下文（V4-Flash 这类混合 attention 模型）时，**如果 CP 算法是全局一个选择，那就一定在其中一类层上是错的**。这条应该进 Primus Projection 的决策逻辑：CP 算法选择要下沉到 per-layer-type，而不是一个全局 flag。

第二条的价值更容易被低估——它是**梯度等价的**（同一组序列以同样权重贡献 optimizer step，累积梯度在浮点归约顺序意义下保持），所以是纯收益的调度优化，只要 dataloader 层能拿到文档边界就能做。3.54× 这个数值本身就值得我们在自己的长上下文 packing 上量一遍。

### 8.6 与知识库里已有结论的三处交叉

| 本篇的观察 | 已有笔记里的对应 | 结论 |
|---|---|---|
| 通信 kernel 吃 SM/显存/带宽，会拖慢与之重叠的计算；大规模下**消除**优于**隐藏** | [Tessera](./tessera.md)：生产环境量到 **EP 通信 kernel 占 ~20 SM，10–20% 减速**，因此否掉 Comet 式融合 | 两次独立确认。"重叠是免费的"这个默认假设在大规模下不成立，这是我们做 overlap 类工作时必须先回答的问题 |
| GDLA 把 KV latent 展开一次、signal/noise 两路共享 16 个展开 KV head | LAGA（[2607.17644](https://arxiv.org/abs/2607.17644)）：Megatron-Core hard-assert 禁掉训练路径的 absorbed MLA，因为中间量 `n_h × d_kv/token` 比被替代的 per-head K/V 更大，DSv3 规模激活显存 +20–34% | GDLA 的写法落在显存友好的那一侧。做 GDLA 类架构时"展开一次共享"是正确的默认 |
| MoE 层锁在节点内（EP=8 走 NVLink），不用 PP | [MegaScale-MoE](./megascale-moe.md)：MoE 层锁节点内 + SP(attn)/EP(FFN) | 同一判断的第二次出现。EP 域不跨节点这条在 314B 规模上仍然成立 |

### 8.7 一句话的方法论提醒

这篇报告展示了一个我们自己也容易犯的问题：**§3.5 给了严谨的解析记账并主动声明"通信量不是延迟代理"，而 §3.1–3.4 十几项优化一个数字都没有。** 同一篇报告里两种截然不同的严谨度。

对我们的启示是反过来的：写 MonolithEP / ROCmoe 的对外材料时，如果某项优化只有定性描述，**要么补上测量，要么像他们附录 B 那样明确声明"这个数字不是收益代理"**。含糊地并列定性描述和定量结论，读者会默认全部都有实测支撑。

## 9. 延伸阅读

| 论文 | 为什么值得读 |
|---|---|
| **mHC: Manifold-Constrained Hyper-Connections**（[2512.24880](https://arxiv.org/abs/2512.24880)，DeepSeek） | Motif 3 §2.3 改的就是这篇。原文有完整的 I/O 开销分析（HC 把访存代价放大约 `n` 倍）、DualPipe 扩展、以及用 TileLang 实现的 5 个 kernel 的分工。**我们 V4 的 mHC 实现的直接上游** |
| **Grouped Differential Attention**（[2510.06949](https://arxiv.org/abs/2510.06949)，Motif 自己） | GDLA 的 G 那一半。含 `g` 的消融（3:1 / 4:1 最优）和 group-differentiated growth（放大时只复制 signal head） |
| **Scalable training of MoE models with Megatron Core**（[2603.07685](https://arxiv.org/abs/2603.07685)，NVIDIA） | Motif 3 §3.1 显存高效 permutation 的来源。已在 `knowledge/systems/industry-training-optimization-2026.md` 里被列为最该细读的三篇之一 |
| **Recipes for pre-training LLMs with MXFP8**（[2506.08027](https://arxiv.org/abs/2506.08027)，NVIDIA） | Motif 3 MXFP8 选型的依据 |
| **A unified view of attention and residual sinks**（[2601.22966](https://arxiv.org/abs/2601.22966)） | §4.2 拒绝硬截断激活的理论依据。如果我们要做激活离群值治理，这篇是先决 |
| **Motif 2 12.7B Technical Report**（[2511.07464](https://arxiv.org/abs/2511.07464)） | Parallel Muon 的原始出处，以及 dynamic data-mixture scheduling |
| **motif3-training-example** 仓库 | TorchTitan 基座的参考实现。§3 那些没给数字的优化，**代码在这里**——FSDP 的有界 in-flight RS 窗口、专家 AG 消除、CPU offload 的 hook 点、window-aware Ring 的 halo 交换，都值得直接读源码 |

## 参考

- 论文：<https://arxiv.org/abs/2608.09119>
- 代码：<https://github.com/MotifTechnologies/motif3-training-example>
- 本地相关实现：`Primus/primus/backends/megatron/core/transformer/hyper_connection.py`、`Primus/primus/backends/megatron/core/optimizer/per_head_muon.py`、`Primus/primus/configs/models/megatron/deepseek_v4_base.yaml`
