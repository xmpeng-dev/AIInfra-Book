# SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining

> [arXiv 2602.10718](https://arxiv.org/abs/2602.10718) (v3) · 美团 Longcat + 清华 · 代码 [meituan-longcat/SGLang-FluentLLM](https://github.com/meituan-longcat/SGLang-FluentLLM)
> 硬件：8x NVIDIA Hopper（论文称"受限不便披露具体型号"，见下文推断）· NVLink
> 模型：DeepSeek-V3.1 (671B) · LongCat-Flash-Thinking (560B)

## TL;DR

**一句话：把 MLA 的 KV cache 做成 FP8，同时不让 RoPE 那 64 维掉精度。** 最高 1.91x 端到端解码吞吐（对 FlashMLA BF16 基线），11 个 benchmark 上基本平手。

它是**精度杠杆**，不是稀疏杠杆——全程是稠密 MLA，没有 indexer、没有 top-k。而且 1.91x 里**大部分来自 FP8 让 KV cache 减半、batch 能开更大**，不是来自算得更快（最佳点在 DP8/TP1，正是显存最紧的配置）。

对我们最直接的一条：论文的 Config A（RoPE 也一起量化）被证明会在深层"误差爆炸"，而 **FlyDSL 现在的 `mla_fwd_decode_m16x8_fp8_fp8` 看起来就是 Config A**（见 §6）。

## 1. Problem

MLA 用低秩压缩把 KV cache 压小了，但高并发下光靠架构压缩不够，还想把 cache 量化到 FP8 换更大 batch。naive 地上 FP8 会撞三堵墙：

**(1) MLA 的 KV cache 是异质的。** 每个 token 的 cache 分两段：content latent `c_KV`（512 维，低秩压缩出来的内容）和 decoupled RoPE `k^R`（64 维，位置信息，所有 head 共享）。论文在 LongCat-Flash-Thinking 上实测（Fig 3）：

| 分量 | 动态范围 | FP8 量化后 MSE |
|---|---|---|
| content | 集中在 ±10^1 | 正常 |
| RoPE | 达 ±10^3，有明显离群尾部 | 高一个数量级 |

统一量化对这两段同等对待，等于被 RoPE 的离群值绑架。

**(2) 只量化 content 又会引入混合精度累加。** FlashMLA 把 QK GEMM 的归约维（576）切成 9 个 thread group（每组 64 维），用 permuted schedule 做指令级并行。RoPE-aware 量化后前 8 组是 FP8（要反量化）、第 9 组是 BF16，混合精度累加需要一道同步屏障，打断交错执行、产生流水气泡。

**(3) PV GEMM 的 scale 维度对不上。** Hopper 上 FP8 WGMMA 要求 V 必须 k-major（沿序列维连续）。但 MLA 里 V 和 K 共用同一份 latent cache，所以 **V 继承的是 per-token 量化 scale，而这些 scale 恰好排在 GEMM 的归约维上**。归约求和的每一项 scale 都不同，没法提到 GEMM 外面——常规的"算完再反量化"范式直接失效。

## 2. Method

### 2.1 RoPE-Aware Per-Token 量化

**算法侧**：只把 content 量化成 FP8，RoPE 保持 BF16。粒度选 per-token 而不是 per-block，理由是自回归解码下 block-wise 会留下"page tail"——未填满的量化块，需要复杂的尾巴缓冲管理。per-token 的好处是新 token 生成即可立刻量化，且所有 token 走统一逻辑，好接 vLLM / SGLang。

**硬件侧的关键技巧 —— Scale Domain Alignment**：不去改 kernel 的累加逻辑，而是在**数据准备阶段**把 BF16 的 RoPE 项预先除以 content 的量化 scale：

$$Q^R \leftarrow Q^R / S^{Q_c}, \quad K^R \leftarrow K^R / S^{K_c}$$

这样 BF16 的 RoPE 段被"伪装"成已经处在 FP8 量化域里的值，kernel 就能对 9 个组一视同仁，**原封不动复用 FlashMLA 已经调好的累加顺序**，中间不需要反量化也不需要同步。这是全文我认为最漂亮的一手——用数据侧的一次预缩放，换掉了 kernel 侧的一道屏障。

### 2.2 Scale-Fusion PV 计算流水重构

针对第 3 堵墙，三件事连在一起做：

1. **Scale Fusion**：利用乘法结合律，把 V 的量化 scale 提前融进注意力概率矩阵，`P' = P ⊙ S_V`。scale 从归约维挪到了外面。
2. **Block-wise 动态量化 P'**：融了 `S_V` 之后 P' 的动态范围被拉宽，所以对 P' 做块级动态量化，块大小取 PV GEMM 的 tiling 参数 `BlockN = 64`，跟 kernel 的分块执行天然对齐。
3. **隐式反量化**：块级量化让各块 scale 不同，他们把这个缩放**直接并进 softmax**，让 softmax 顺手完成反量化，不额外付出缩放开销。

重构后的流水：softmax（含隐式反量化）→ 融 `S_V` 进 P → 块级量化 P' → 分块 PV GEMM（scale-aware 累加）。

### 2.3 端到端数据流优化

三层：

**Layer 1 融合算子。** `Fused-Q-Quant`（per-token scale 计算 + 混合精度转换 + scale domain alignment 三合一）；`Fused-K-Append`（量化 + 对齐 + PagedAttention 式非连续写入 cache，一次 launch 搞定，消掉中间缓冲）；`Fused-Fetch-Dequant`（取数时在寄存器级顺手反量化，服务 chunk prefill / prefix cache 复用，避免"先加载到 SMEM 再单独跑反量化 kernel"两步走）。

**Layer 2 访存对齐。** content 维的 tile 从 64 加到 128，正好对上 128 B 的 L2 cache line 和 Hopper 的 Swizzle-128B SMEM 布局，产生完全合并的 TMA descriptor。

**Layer 3 零开销布局变换。** V-tile 走 SMEM→RF→SMEM 转置，P 累加器做字节级寄存器置换以匹配 WGMMA 输出布局；两者都调度到**前一个 QK GEMM 的计算区间里**用异步执行盖掉。

## 3. Experiments

**端到端吞吐**：8x Hopper，DeepSeek-V3.1 与 LongCat-Flash-Thinking，上下文 16k-128k，扫 DP1/TP8、DP4/TP2、DP8/TP1。相对 FlashMLA BF16 **最高 1.91x**，且**最大增益出现在 DP8/TP1** —— 论文明说原因是 FP8 减小的显存占用让 batch 能开得大得多。

**精度**（Table 1，11 个 benchmark）：基本平手。降得最多的几个：AIME-25 87.92 → 85.42、GPQA-Diamond 84.15 → 82.57、Arena-Hard 57.10 → 55.50（均为 DSv3.1）；也有涨的，IFEval 86.32 → 87.25。

**数值消融**（Fig 4 / Table 2，32k 上下文逐层测 RMSE / cosine diff / relative L2）：

| 配置 | content | RoPE | 结论 |
|---|---|---|---|
| **SnapMLA** | per-token | 不量化 | 误差最低，与 BF16 基线持平 |
| Config A | per-token | per-token | **深层"误差爆炸"**，实证 RoPE 的量化敏感性 |
| Config B | per-tensor 静态 (scale=1.0) | 不量化 | 抓不住 token 间的动态范围变化 |
| Config C | per-tensor 动态 | 不量化 | 同上 |
| Config D | per-block | 不量化 | 略差于 per-token |

**Kernel roofline**：核心是 16 个 FP8 tile（content）+ 1 个 BF16 tile（RoPE），BF16 等效算力成本从 17 降到 `16/2 + 1 = 9`，所以有效 FP8 峰值 = `148 x 17/9 ≈ 279.6 TFLOPS`。实测紧贴这条线，H>=64 时约达有效峰值的 85%。

## 4. 一个值得注意的推断：那块 GPU 大概率是 H20

论文说"受限不便披露具体 Hopper 型号"，但给了 **BF16 峰值 148 TFLOPS**。H20 的 BF16 稠密算力正是 148 TFLOPS，FP8 是 296 TFLOPS —— 跟他们推出的有效 FP8 峰值 279.6 高度吻合（差值正来自那 1 个保 BF16 的 tile）。

这件事影响结论的可迁移性：**H20 是算力贫、带宽富的配置**（148 TFLOPS BF16 / 4.0 TB/s / 96 GB）。在这种卡上 FP8 换来的算力收益相对更显眼。MI355X 的算力带宽比完全不同（FP8 稠密 5 PFLOPS / 8 TB/s / 288 GB），所以：

- **算力那部分收益到 MI355X 上会缩水**（本来就不缺算力）
- **显存那部分收益照样成立**——FP8 让 KV cache 从每 token 1152 B 降到约 644 B（512x1 content + 64x2 RoPE + scale），约 1.8x，batch 照样能开大。而这恰恰是 1.91x 的主要来源

## 5. Limitations

- **Hopper 专属实现**。WGMMA 的 k-major 约束、TMA、Swizzle-128B 都是 Hopper 的东西；§2.1 那个预缩放技巧也是为了绕开 FlashMLA 特定的 9-group schedule。**问题**（异质敏感性、scale 维度错配）是架构无关的，**解法**不是。
- **只做 decode**。prefill 的 FP8 交给 FA3，没碰。
- **稠密 MLA，与稀疏正交**。没有 indexer、没有 top-k、没有 DSA。
- GPU 型号不披露，复现时的算力口径要自己对。

## 6. Our take

**它和 MegaAttn 是两个不冲突的杠杆，可以叠乘。** SnapMLA 让**每个被注意到的 token 更便宜**（精度）；MegaAttn 让**被注意到的 token 更少 + 选择过程更便宜**（稀疏 + 融合）。SnapMLA 是稠密 MLA，它的位置其实是坐在 MegaAttn 第三段（sparse MLA）**里面**，而不是替代 MegaAttn。

**对 FlyDSL 有一条可以马上查的：现有 MLA kernel 疑似就是被证伪的 Config A。**
`kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py` 结构上确实按 FlashMLA 的方式切了 nope / rope（`NUM_NOPE_ITERS = 8`、`NUM_ROPE_ITERS = 1`，`q_nope_packs` / `q_rope_packs` 分开），但 **`QK_HEAD_DIM = 576` 全程走 FP8，两段没有精度区分**；而且 `flydsl_mla_fwd_decode()` 的入参里只有 `softmax_scale`，**没有任何 per-token 反量化 scale 张量**。若上游确实是对 576 维统一量化，那就正好落在论文实测会"深层误差爆炸"的 Config A 上。

值得做的验证（成本很低）：拿长上下文样本逐层测 attention 输出对 BF16 的 RMSE / relative L2，对比"576 维统一 FP8" vs "512 content FP8 + 64 RoPE BF16"两种配置。如果误差曲线复现了论文 Fig 4 的形状，那么把 RoPE 那 64 维拉回 BF16 是个纯赚的改动——而且 §2.1 的预缩放技巧正好能让 kernel 的累加顺序不用动。

**可以直接搬的**：RoPE-aware per-token 量化（算法层，架构无关）、Scale Domain Alignment 预缩放（思路可搬，MFMA 上的具体形式要重做）、Scale Fusion 把 `S_V` 折进 P 再由 softmax 隐式反量化（这个问题在 absorbed 模式下同样存在，与是不是 Hopper 无关）。

**不用搬的**：TMA / Swizzle-128B / WGMMA 布局那一层，CDNA4 上对应的是另一套（MFMA + LDS bank conflict + buffer load），得重新设计。

## 参考

- FlashMLA（BF16 基线）：[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- FlashAttention-3（FP8 attention 与 k-major 布局问题的来源）：NeurIPS 2024
- MegaAttn 设计文档：[`../notes/megaattn/2026-08-05_1650_megaattn_v1_architecture_design.md`](../notes/megaattn/2026-08-05_1650_megaattn_v1_architecture_design.md)
