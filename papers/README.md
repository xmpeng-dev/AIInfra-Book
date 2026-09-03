# papers — 论文阅读笔记

每篇论文一个文件(或子目录,当有复现脚本/补充材料时)。
slug 是 kebab-case;年份只在需要消歧义时加。

> 100+ 篇论文的全景分类索引在 [`../knowledge/moe/paper-landscape.md`](../knowledge/moe/paper-landscape.md);
> **arXiv 2025–2026 训练/推理优化速查**见 [`../knowledge/systems/training-optimization-landscape-2026.md`](../knowledge/systems/training-optimization-landscape-2026.md);
> **2026-08 增量扫描（27 篇待读 + 本轮精读的 5 篇）**见 [`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md);
> 这里只列**已有详细笔记**的论文。

## 论文清单

### 训练优化

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| MoEBlaze | MLSys'26 | MoE 训练内存 | 元数据索引不物化 routed token + SwiGLU 融合 + SiLU backward 重算,单 H100 单层 vs Megablocks 内存↓4× / 加速≤6.2×;[html](./moeblaze.html) | [`moeblaze.md`](./moeblaze.md) |
| LAER-MoE / FSEP | ASPLOS'26 | MoE 并行 | FSEP 全分片专家并行 + 动态重排,1.69× 端到端 | [`laer-moe-fsep.md`](./laer-moe-fsep.md) |
| SwiftMoE | arXiv'25 | MoE 训练 | 参数-优化器解耦 + 动态 Expert 放置,+30.5% 收敛 | [`swiftmoe.md`](./swiftmoe.md) |
| MemFine | arXiv'25 | MoE 内存 | 细粒度 chunk 激活调度 + 选择性重计算,48% 内存↓ | [`memfine.md`](./memfine.md) |
| MoE Parallel Folding | arXiv'25 | MoE 并行 | 五维混合并行,Attn/MoE 解耦,49.3% MFU | [`moe-parallel-folding.md`](./moe-parallel-folding.md) |
| Comet | MLSys'25 | MoE 通信重叠 | shared-tensor 依赖分解 + thread-block 专用化(单融合 kernel),1.96× 单层 / 1.71× 端到端;[html](./comet.html) | [`comet.md`](./comet.md) |
| MegaScale-MoE | EuroSys'26 | MoE 大规模训练 | MoE 层锁节点内 + SP(attn)/EP(FFN) + inter/intra-op overlap + 通信压缩,1440×H800 训 352B 达 1.41M tok/s / 1.88× vs Megatron;[html](./megascale-moe.html) | [`megascale-moe.md`](./megascale-moe.md) |
| UniEP | arXiv'26 | MoE MegaKernel 训练 | Dispatch+GroupGEMM / GroupGEMM+Combine 融成单 kernel,SM 动态角色 + token scoreboard + 确定性映射保 bit-wise,vs COMET 1.03–1.38×;开源可移植 AMD;[html](./uniep/uniep.html) | [`uniep/`](./uniep/README.md) |
| UltraEP | arXiv'26 | MoE 负载均衡(RSN) | Rack-Scale 节点上 exact-load 实时(每 microbatch/层)均衡,quota planner + RSN-native 通信,不均 1.3–4→1.0,训练 1.42×/serving 1.56×;点名 AMD Helios;[html](./ultraep/ultraep.html) | [`ultraep/`](./ultraep/README.md) |
| DisagMoE | arXiv'26 | MoE 训练 overlap | attention/FFN 解耦到不同 GPU 组 + AF-Pipe(all-to-all→M2N 一等流水阶段)+ roofline/MILP 分 GPU/NIC,up to 1.81× vs Megatron;[html](./disagmoe/disagmoe.html) | [`disagmoe/`](./disagmoe/README.md) |
| Piper | arXiv'26 | MoE 训练(AMD/HPC) | Frontier(MI250X+RCCL+Dragonfly)上资源建模 + PP×EP 局部化通信 + 拓扑感知 all-to-all + expert migration,2–3.5× MFU vs X-MoE;[html](./piper/piper.html) | [`piper/`](./piper/README.md) |
| AutoOverlap | arXiv'26 | comm-compute 编译器 | communication chunk 抽象 + 源到源 Triton 编译器自动 kernel 内细粒度 overlap(后端/chunk/tile 三维自适应),平均 1.3× 最高 4.7×;[html](./autooverlap/autooverlap.html) | [`autooverlap/`](./autooverlap/README.md) |
| FlowMoE | NeurIPS'25 | MoE 流水线 | 统一流水线调度 + chunk 优先级,-57% 训练时间 | [`flowmoe.md`](./flowmoe.md) |
| Megatron-Core MoE | -- | 工程参考 | NVIDIA 官方 MoE 实现细节(grouped GEMM / token dispatcher / load balance) | [`megatron-core-moe.md`](./megatron-core-moe.md) |
| veScale FSDP | -- | 分布式训练 | veScale 的 FSDP 设计与实现要点 | [`vescale-fsdp.md`](./vescale-fsdp.md) |
| DMuon | arXiv'26 | 分布式优化器 | Owner-centric Muon + Gram SYRK NS + MILP LB,FSDP2 drop-in 3 行,端到端 avg +2% vs AdamW / optim 6.85–163× vs Muon-AG | [`dmuon.md`](./dmuon.md) |
| MatrixFSDP | arXiv'26 | 分布式优化器 | 改 ZeRO-3 分片放置(每矩阵一个 owner 持整块)让 backward 归约天然落在 owner,optim step 零矩阵集合通信;64×A100 optim 4.2×→54.6× / E2E 1.37×→2.15×;不支持 TP、未谈 MoE | [`matrixfsdp.md`](./matrixfsdp.md) |
| AGoQ | arXiv'26 | 低精度梯度通信 | 激活近 4-bit + 8-bit 梯度,把 AllReduce 改成 A2A→本地 FP32 归约→AG 以避开"通信中做低精度加法";显存 −52%、1.34×(8B–32B LLaMA, ≤64 卡)。**但要打折看**:52% 里梯度量化只占 2.4 GB(5.2%),1.34× 几乎全来自激活省显存换掉重算(R=10→0)且该配置 DP=2、重构基本没起作用;3.4× 全来自位宽而非结构,从未做原生低精度 RS 对照组;Table 7/Table 4/Eq.21 三处内部矛盾。**最关键**:该结构上游已出货(`--gtp-remat-reduce-scatter-with-fp32-accumulation`,"same bytes on the wire"),ZeRO++ qgZ(2306.10209, 2023)更早发表且多 2-hop;**真缺口是格式(通信库无块缩放 dtype)不是结构**,真约束是**尾数宽度**(加 W 个数需 log₂W 位,W=72→6.2 位 vs FP8 E4M3 的 3 位) | [`agoq.md`](./agoq.md) |
| DynamiQ | SIGCOMM'26 | 低精度梯度通信 | 把单 chunk 的 RS 抽象成 in-arborescence,每跳误差正比子树大小(ring `O(εSM²n³)` / butterfly `O(εSM²n²)`);按 super-group 在**聚合后**梯度里的量级分 2/4/8/16 位 + 层次化 scale + 相关舍入 + decompress-accumulate-recompress 融合 kernel(瓶颈是 HBM 带宽非线速)。LLaMA-1B MMLU 达 BF16 的 99% 时比 MXFP8 快 34.5% / 比 BF16 快 40.8%;**消融里变量位宽分配单项 3.5–5.1×,超其余三项之和**。⚠ 只到 8 GPU / 1B 级微调,DP=8192 那组是纯模拟+合成数据;**testbed 的 HBM:线速 = 61:1,加速比不可外推**(盈亏门槛 9.5,MI455X 机架内只有 5.4–6.5→亏,跨机架 33–39→盈);开源 | [`dynamiq.md`](./dynamiq.md) |
| GIFT | arXiv'26 | 低精度梯度通信 | 用 K-FAC 输入侧因子把梯度白化到近各向同性坐标系再做 FP8 量化与归约(K-FAC 只当度量不当优化器),rank-32 低秩 + 只对 profile 出的 13 个脆弱层(**全是 fc2**)开、每 50 步刷新。FP8 单步往返 RelL2 −67.4%(对角近似几乎无效→有用几何是**跨维耦合**的)。⚠ **质量结论站不住**:同表里直接欧氏 FP8 端到端 −10.79% 而 GIFT 只 −7.6%(慢 3.19 pt);头对头 600M 上欧氏赢 8 项 GIFT 赢 6 项、14 任务均值欧氏 0.5186 > FP32 0.5060 > GIFT 0.5032;基线是 FP32 非 BF16,换 BF16 后直接 FP8 只 +3.9%、GIFT 只 +0.4%;**缺"欧氏+EF"这个关键消融**(GIFT 分支有 EF 和局部 scale,欧氏基线两样都没有);只到 300M/600M 且两点用了不同序列长度。**原创发现**:EF 缓冲累积在变换坐标里,刷新时旧基残差被新 `L_A^⊤` 映射回去→**基不匹配,注入 ∝(L_A^new−L_A^old)^⊤R 的伪更新**,论文一字未提 | [`gift.md`](./gift.md) |
| Tessera | OSDI'26 | 异构 MoE 流水并行(阿里) | 切分与 overlap 调度是循环依赖:先真机 profile 每个 overlap pair 的 post-overlap cost,再用 MILP 按实测边代价选切分,外加 DBO 用路由元数据预测气泡填 Wgrad;Qwen3/Qwen3-Next 生产 4,096–12,288 GPU **+20%~33%**,万亿 **39% MFU**;**§5 工程经验含金量最高**——生产里否掉了 Comet 式融合(后端耦合)、**EP 通信 kernel 占 ~20 SM 致 10–20% 减速**、代价模型尾部误差 15% 会翻转 MILP 排序 | [`tessera.md`](./tessera.md) |
| Motif 3 | arXiv'26 | 314B MoE 模型报告(训练系统) | 韩国 Motif,314B-A13B / 384 专家 top-8 / B200 / TorchTitan。**mHC 的 post-mapping 必须从 `2σ(z)` 退火到 `1σ(z)`**,否则 `>1` 的映射逐层反复放大 sublayer 输出、累积激活离群值——**我们 V4 的 `hyper_connection.py:359` 正是硬编码 `2.0`**;**QK-Clip 在 GQA/MLA 下必须非对称劈分** `r=1/(1+√G)`,对称 `√γ` 会把共享 K 压到零;改 FA4 forward 直接吐 per-head logit max;**大规模下"消除"专家权重 All-Gather 优于"隐藏"**(通信 kernel 吃 SM,与 Tessera 的 20-SM 发现互证);长上下文**逐层选 CP**(full→Ulysses / SWA→window-aware Ring halo,解析 1024×)+ LPT 负载重排(rank 间 attention FLOPs 差 **3.54×**);MXFP8 在 EP dispatch **之前**量化(通信减半 + 量化成本降到 1/top-k);6 项专家健康度指标带阈值。**但 §3 十几项系统优化零吞吐/MFU/加速比数字**,只能当设计决策清单读 | [`motif-3.md`](./motif-3.md) |
| HyperParallel-MoE | arXiv'26 | MoE 训练(昇腾) | 算子串行→编译期静态调度的 tile 级 AIC/AIV 异构 taskflow,AIV 驱动单边通信 + SSC 离线编译,单次 kernel launch;Dispatch-to-Combine 1.49–1.58× / E2E 1.08–1.09×;**静态 vs 动态派发 0.1 vs 2.36 µs/任务**,ROCmoe P4 的量化论据 | [`hyperparallel-moe.md`](./hyperparallel-moe.md) |
| MXFP4 原生预训练 | arXiv'26 | FP4 训练数值(AMD) | Penn State + AMD;MI355X **原生** MXFP4 逐段打开,到 ppl 3.3 的 token 开销 Fprop 8–9% → +Dgrad 10–11% → **+Wgrad 跳 26–27%**;随机舍入与随机 Hadamard 在全流水**不收敛**,**确定性** Hadamard 打回 8–9%(H16/H32 同尺寸对照,已排除"旋转尺寸"这个变量);单步吞吐 +20% 但**端到端只剩 +9–10%**——FP4 收益被稳定性闸住;**dense-only,未碰 MoE/grouped GEMM** | [`mxfp4-pretraining.md`](./mxfp4-pretraining.md) · [全文中译](./mxfp4-pretraining-zh.md) |

### 推理系统

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| Perseus | arXiv'26 | 多节点 megakernel 通信 | proxy RDMA 上 `put-with-signal` 展开成 `Put→Fence→Signal`,每次 fence 排空 NIC 流水;96 并发/8 节点吞吐掉到 2%;decoupled signaling(每 PE 一个 fence)+ NIC 侧保序,Libfabric 10.3× / IBRC 2.47×,proxy 反超 IBGDA 1.2×;**节点内无此问题** | [`perseus.md`](./perseus.md) |
| X-Stage | arXiv'26 | 发射后流水阶段建模 | remote store 发出到远端可见之间存在软件可见的 X-Stage;三参数 Burst–Gap 模型 `(T_iss⁰, R=717 GB/s, Q=4.25 MiB)` 预测背压拐点;据此重排 DeepGEMM MegaMoE 的 L1/L2 交错,84 配置几何均值 1.18× / 最大 1.62× / 最差 0.94×(L2 局部性反噬) | [`x-stage.md`](./x-stage.md) |
| ExpertPlex | arXiv'26 | MoE 服务(自适应 APK) | 共享专家 + 分离 attention;Adaptive Persistent Kernel 在 tile 边界抢占(界 = 1 tile + 1 检查 epoch,与序列长无关),抢占决策沿 system→device→DSMEM→warp 传播;vs Green Context prefill 只慢 1.12×(对方 4.07×);goodput 2.01× vs PDD;**唯一质疑静态调度的一篇,但只反固定空间划分** | [`expertplex.md`](./expertplex.md) |
| Fleet | arXiv'26 | Chiplet megakernel | Chiplet-task + 协作 L2 tiling,MI350 decode 1.3–1.5× vs vLLM eager | [`fleet.md`](./fleet.md) |
| MoE Tile Signaling | ICPP'26 | MoE 推理 overlap | remote-owner layout + persistent producer/consumer + tile epilogue signal,combine A2A 与 GEMM overlap,4×A100 最高 2.64× E2E / 2.74× MoE layer vs FasterMoE | [`moe-tile-signaling.md`](./moe-tile-signaling.md) |
| MegaScale-Infer | SIGCOMM'25 | MoE 推理 | 分离式 EP,Prefill/Decode/Expert 解耦,3.2× 吞吐 / 55% 成本↓ | [`megascale-infer.md`](./megascale-infer.md) |
| SnapMLA | arXiv'26 | MLA FP8 解码 | RoPE-aware per-token 量化(RoPE 保 BF16)+ 预缩放域对齐 + S_V 折进 P 由 softmax 隐式反量化,8×Hopper 最高 1.91× 吞吐(主要来自 KV cache 减半后 batch 变大) | [`snapmla.md`](./snapmla.md) |
| KTransformers | SOSP'25 | 异构推理 | CPU+GPU 异构推理,$5K 跑 DeepSeek-V3 671B | [`ktransformers.md`](./ktransformers.md) |
| UBEP | SIGCOMM'26 | MoE EP 通信库(超节点) | 南大 + 华为;BSP 阶段+全局 barrier → 依赖驱动细粒度任务,**Data-as-Flag** 用 512B 原子写把 flag 融进 payload(TFF/DC/**SP 哨兵轮询**三变体),层次化 token 调度同时均衡负载与 hop 距离(Two-Hop 延迟 = 本地 **11.5×**);CM384/256 die 上 dispatch ↓34.7–52.4% / 带宽 +35.3–40.8% / P99 TPOT ↓11.1%;**MoE 通信占 per-token 延迟 ~50% 但只占硬件执行时间 ~20%**——"重排执行而非加带宽"的第三方证据;§6 点名批评 fused persistent kernel 的刚性资源划分与软件轮询,是 ROCmoe 路线要正面回答的一篇 | [`ubep.md`](./ubep.md) |

### 架构创新

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| OmniMoE | arXiv'26 | MoE 路由 | 原子专家 + 笛卡尔积路由 O(√N),10.9× 推理加速 | [`omnimoe.md`](./omnimoe.md) |
| LatentMoE | -- | MoE 架构 | Latent space MoE 设计要点 | [`latent-moe.md`](./latent-moe.md) |

### 编译器 / 内核生成

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| HipKittens | MLSys'26 | AMD kernel 原语(Stanford+AMD) | ⭐ tile 抽象可移植到 AMD,但**实例化算法必须重做**。**wave specialization 在 AMD 是负优化**(寄存器静态均分→producer 白占寄存器压低输出 tile:0-producer/256×256 得 1610 vs 4-producer/128×256 得 893 TFLOPS);替代方案 8-wave ping-pong(48 行热循环即追平裸汇编)/4-wave interleave;register pinning 绕过 HIPCC +20%;chiplet grid swizzle +19%;**附录逆出了 CDNA ISA 未文档化的 LDS phase/bank 表**;已产品化进 AITER | [`hipkittens.md`](./hipkittens.md) · [全文中译](./hipkittens-zh.md) |
| Swizzled Head-first Attention | arXiv'26 | AMD chiplet NUMA 调度(Duke+AMD) | FA2 的 workgroup 网格重排,把一个 head 的全部 Q-block 锁到单个 XCD 以复用私有 L2;提出 **ACC(共享同一份 K/V 的 WG 组)** 抽象;MI300X 上 MHA 最高 **+50%**,L2 命中 **90–96% vs block-first ≈1%**——**AITER 现用的 Swizzled Block-first 在多头+长序列下几乎全 miss**;但 GQA 上无优势(8 KV head 恰等于 8 XCD)、backward 仅 1.10×、**causal 下亏 7%**(FlyDSL 实测,论文未提);全文归一化无绝对 TFLOPS;**AMD 已实现进 FlyDSL** | [`swizzled-head-first-attention.md`](./swizzled-head-first-attention.md) |
| AutoMegaKernel | arXiv'26 | megakernel 合成 + 静态校验 | HF Llama → 单个常驻 cooperative kernel,零手写 CUDA;frozen schedule-IR 验证器用 9 条静态图检查证无死锁/无竞争,7160 个对抗调度**零假接受**;同源重定向 sm_80/90/120;**负面结果更有用**:瓶颈是每 tile 跨 SM 同步,且在带宽最高的训练级芯片上最严重 | [`automegakernel.md`](./automegakernel.md) |

### 其他

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| LEANN | -- | 向量索引 | Low-storage vector index for RAG / on-device | [`leann-low-storage-vector-index/`](./leann-low-storage-vector-index/README.md) |

## 编辑约定

- **slug**: kebab-case,通常去掉 "moe" 后缀(因为同类论文聚在一起)。
- **单文件 vs 子目录**: 默认单文件 `<slug>.md`;当有复现脚本、补充材料、原始 PDF 时升级为 `<slug>/` 目录。
- **新增论文**: 用 `.cursor/skills/read-paper/SKILL.md` 或 `paper-deep-analysis/SKILL.md`,写完后**回写本 README 的清单**(挑对应方向的表格加一行)。
- **必含字段**: Problem · Contribution · Method(含数据流) · Experiments · Limitations · Our take。
- 引用其他论文笔记用 `./<slug>.md` 相对路径。

## 历史索引

旧版 100+ 篇分类索引已搬到 [`../knowledge/moe/paper-landscape.md`](../knowledge/moe/paper-landscape.md)。
