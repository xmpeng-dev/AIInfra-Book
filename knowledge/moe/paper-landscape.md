# 🔥 MoE 优化论文全景图 2025-2026（第二版）

> **更新时间:** 2026-03-07  
> **覆盖论文总数:** 100+ 篇（精选深度笔记 12 篇，分类索引全部）  
> **研究方向:** 训练优化 · 推理系统 · 通信优化 · 架构创新 · 理论分析 · 生产部署

---

## 📖 深度阅读笔记目录

### 🔵 训练优化类（Training Optimization）

| 论文 | 发表 | 核心贡献 | 关键数字 | 笔记 |
|------|------|---------|---------|------|
| **MoEBlaze** | arXiv'26 | 内存墙突破：数据结构+Kernel融合+Smart AC | 4x加速，50%内存↓ | [笔记](../../papers/moeblaze.md) |
| **LAER-MoE** | ASPLOS'26 | FSEP全分片+动态重排负载均衡 | 1.69x端到端 | [笔记](../../papers/laer-moe-fsep.md) |
| **SwiftMoE** | arXiv'25 | 参数-优化器解耦，动态Expert放置 | +30.5% 收敛 | [笔记](../../papers/swiftmoe.md) |
| **MemFine** | arXiv'25 | 细粒度Chunk激活调度+选择性重计算 | 48%内存↓ | [笔记](../../papers/memfine.md) |
| **MoE Parallel Folding** | arXiv'25 | 五维混合并行，Attn/MoE解耦 | 49.3% MFU | [笔记](../../papers/moe-parallel-folding.md) |
| **Comet** | MLSys'25 | Tile级计算-通信Overlap，Warp专用化 | 1.8x端到端 | [笔记](../../papers/comet.md) |
| **MegaScale-MoE** | EuroSys'26 | 万卡生产训练系统，容错+拓扑感知 | 42% MFU@10K GPU | [笔记](../../papers/megascale-moe.md) |
| **FlowMoE** | NeurIPS'25 | 统一流水线调度，Chunk优先级 | -57%训练时间 | [笔记](../../papers/flowmoe.md) |

### 🟠 推理系统类（Inference Systems）

| 论文 | 发表 | 核心贡献 | 关键数字 | 笔记 |
|------|------|---------|---------|------|
| **MegaScale-Infer** | SIGCOMM'25 | 分离式EP，Prefill/Decode/Expert解耦 | 3.2x吞吐，55%成本↓ | [笔记](../../papers/megascale-infer.md) |
| **KTransformers** | SOSP'25 | CPU+GPU异构推理，消费级硬件跑671B | $5K跑DeepSeek-V3 | [笔记](../../papers/ktransformers.md) |

### 🟢 架构创新类（Architecture）

| 论文 | 发表 | 核心贡献 | 关键数字 | 笔记 |
|------|------|---------|---------|------|
| **OmniMoE** | arXiv'26 | 原子专家+笛卡尔积路由O(√N) | 10.9x推理加速 | [笔记](../../papers/omnimoe.md) |

### ⚪ 基础原理文档（Fundamentals）

| 文档 | 内容 | 链接 |
|------|------|------|
| **MoE 详细计算过程与数据流** | 前向/反向计算、张量形状、A2A 通信与真实数据流示例 | [文档](./dataflow.md) |

---

## 📋 100+ 篇论文分类索引

> 以下论文按方向分类，标有 ✅ 的有详细笔记，其他提供摘要导读

---

### 🔵 A. 训练系统优化

#### A1. 通信优化（Communication）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **MegaScale-MoE** | EuroSys'26 | 万卡拓扑感知路由+分层EP | 45% A2A延迟↓ |
| **Comet** | MLSys'25 | Tile级计算-通信Overlap | 90%+ Overlap率 |
| **FlowMoE** | NeurIPS'25 | Chunk级流水线调度 | -57%训练时间 |
| **ScaleMoE** | PACT'25 | 大规模可扩展训练框架 | — |
| **FUSCO** | arXiv'25 | 数据shuffle通信-变换融合 | — |
| **UCCL-EP** | arXiv'25 | 可移植的Expert并行通信库 | — |
| **FarSkip-Collective** | arXiv'25 | 解除MoE中阻塞通信的限制 | — |
| **HybridEP** | arXiv'25 | 跨数据中心的混合EP/DP传输 | — |
| **MixNet** | SIGCOMM'25 | 可重构光电混合网络用于MoE训练 | — |
| **Optimizing All-to-All on Torus** | MICRO'25 | 环形网络中容错All-to-All | — |
| **BigMac** | arXiv'25 | 通信高效的MoE模型结构 | — |
| **FSMoE** | ASPLOS'25 | 灵活可扩展MoE训练系统 | — |
| **X-MoE** | SC'25 | HPC平台MoE可扩展训练 | — |
| **EfficientMoE** | TPDS'25 | 自适应负载均衡训练优化 | — |

#### A2. 负载均衡（Load Balancing）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **LAER-MoE** | ASPLOS'26 | FSEP+动态重排 | 1.69x |
| **SwiftMoE** | arXiv'25 | 参数-优化器解耦 | +30.5% |
| **Least-Loaded EP** | arXiv'26 | 最小负载Expert并行 | — |
| **NetMoE** | ICLR'25 | 动态Sample放置 | — |
| **PopFetcher** | ATC'25 | 基于热度的Expert预取 | — |
| **HierMoE** | arXiv'25 | 层次化Token去重+Expert交换 | — |

#### A3. 内存优化（Memory）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **MoEBlaze** | arXiv'26 | 数据结构+Kernel+Smart AC | 4x，50%↓ |
| **MemFine** | arXiv'25 | Chunk激活调度 | 48%↓ |
| **Hecate** | arXiv'25 | 全分片稀疏数据并行 | — |
| **MoE-DisCo** | arXiv'26 | 低经济成本MoE训练 | — |
| **Dense Backprop** | arXiv'25 | 稠密反向传播改善MoE训练 | — |
| **Batch Tiling Attention** | SC Workshop'25 | Wafer-Scale处理器上MoE训练 | — |

#### A4. 并行策略（Parallelism）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **MoE Parallel Folding** | arXiv'25 | 5D并行+Attn/MoE解耦 | 49.3% MFU |
| **FOLDMOE** | ACL'25 | Attention-MoE流水线，长序列 | — |
| **Dense Training, Sparse Inference** | arXiv'25 | 训练稠密推理稀疏的新范式 | — |

---

### 🟠 B. 推理系统优化

#### B1. Expert 卸载与缓存（Offloading & Caching）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **KTransformers** | SOSP'25 | CPU+GPU异构，消费级硬件 | 15-20 tok/s |
| **Taming Latency-Memory** | EuroSys'26 | 细粒度Expert卸载延迟-内存权衡 | — |
| **Oracle-MoE** | ICML'25 | 局部路由一致性的内存约束推理 | — |
| **FloE** | ICML'25 | 飞行中的MoE推理，内存约束GPU | — |
| **fMoE** | arXiv'25 | 细粒度Expert卸载大规模服务 | — |
| **HybriMoE** | DAC'25 | CPU-GPU混合调度+缓存管理 | — |
| **eMoE** | arXiv'25 | 任务感知内存高效MoE推理 | — |
| **MoE-Gen** | arXiv'25 | 单GPU高吞吐MoE推理，模块级批处理 | — |
| **PreMoe** | arXiv'25 | 内存受限下的Expert剪枝+检索 | — |
| **BuddyMoE** | arXiv'25 | 利用Expert冗余加速内存受限推理 | — |
| **Accelerating with Speculative Decoding** | arXiv'25 | 推测解码隐藏卸载延迟 | — |
| **MoE-SpeQ** | arXiv'25 | 推测量化解码+主动Expert预取卸载 | — |

#### B2. Expert 预取（Prefetching）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **PROBE** | arXiv'26 | 实时预测预取，计算通信共平衡 | 35-42%延迟↓ |
| **Pre-Attention Expert Prediction** | arXiv'25 | 注意力前预测Expert | — |
| **Efficient MoE Inference** | arXiv'25 | 分离式EP+细粒度调度 | — |
| **MoEs Are Stronger** | arXiv'25 | 超并行推理扩展RoE | — |

#### B3. 服务系统（Serving Systems）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **MegaScale-Infer** | SIGCOMM'25 | 分离式EP，万卡服务 | 3.2x吞吐 |
| **MixServe** | arXiv'26 | 自动分布式MoE服务，融合通信 | — |
| **Janus** | arXiv'25 | 分离Attention和Expert | — |
| **Tarragon** | arXiv'26 | MoE推理弹性容错 | — |
| **GRACE-MoE** | arXiv'25 | 局部感知路由+分组复制推理 | — |
| **HAP** | arXiv'25 | 混合自适应并行高效MoE推理 | — |
| **MoETuner** | arXiv'25 | 均衡Expert放置+Token路由 | — |
| **Samoyeds** | EuroSys'25 | 结构化稀疏+稀疏Tensor Core | — |
| **Priority Scheduling** | EuroMLSys'25 | 混合优先级MoE推理抢占调度 | — |
| **Expert Sharding** | EuroMLSys'25 | 推理中的Expert分片加速 | — |
| **MoE-Prism** | arXiv'25 | 弹性MoE服务，模型-系统协设计 | — |
| **BrownoutServe** | arXiv'25 | SLO感知的MoE突发负载服务 | — |

#### B4. 负载均衡推理（Inference Load Balancing）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **Efficient MoE Serving** | arXiv'25 | 均衡激活Expert数而非Token | — |
| **MicroMoE** | arXiv'25 | 细粒度负载均衡Token调度 | — |
| **Capacity-Aware Inference** | arXiv'25 | 缓解Straggler效应 | — |
| **Opportunistic Expert Activation** | arXiv'25 | 批次感知Expert路由加速 | — |
| **Speculative MoE** | arXiv'25 | 推测Token+Expert预调度 | — |

---

### 🟢 C. 模型架构与路由

#### C1. 路由创新（Routing）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **OmniMoE** | arXiv'26 | 原子专家+笛卡尔积O(√N)路由 | 10.9x推理 |
| **Ada-K Routing** | ICLR'25 | 自适应K动态路由 | — |
| **GatePro** | arXiv'25 | 无参数Expert选择优化 | — |
| **Load Balancing Similarity** | arXiv'25 | 相似度保持路由器负载均衡 | — |
| **MoE-GPS** | arXiv'25 | 动态Expert复制预测策略指南 | — |
| **Long-Tailed Router** | arXiv'25 | 长尾分布感知的大视觉语言模型路由 | — |
| **Steering MoE LLMs** | arXiv'25 | 通过Expert激活/去激活控制LLM | — |
| **LExI** | arXiv'25 | 层自适应激活Expert高效推理 | — |
| **MoDES** | arXiv'25 | 动态Expert跳过加速多模态LLM | — |
| **Context-Aware MoE** | arXiv'25 | CXL GPU-NDP系统上的上下文感知推理 | — |
| **C3PO** | arXiv'25 | 测试时Expert重混合路径优化 | — |
| **Not All Models Suit Offloading** | arXiv'25 | 本地路由一致性与Expert卸载分析 | — |

#### C2. 稀疏性与压缩（Sparsity & Compression）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **LatentMoE** | arXiv'26 | 最优精度/FLOP/参数比 | — |
| **Parameters vs FLOPs** | arXiv'25 | 最优稀疏性缩放律 | — |
| **DualSparse-MoE** | arXiv'25 | 张量/神经元级稀疏+Expert分区重建 | — |
| **PuzzleMoE** | arXiv'25 | 稀疏Expert合并+位压缩推理 | — |
| **DSMoE** | arXiv'25 | 矩阵分区Expert+动态路由Dense转MoE | — |
| **DiEP** | arXiv'25 | 可微Expert剪枝自适应压缩 | — |
| **Sub-MoE** | arXiv'25 | 子空间Expert合并压缩 | — |
| **MergeMoE** | arXiv'25 | Expert输出合并压缩 | — |
| **ReXMoE** | arXiv'25 | 最小开销的Expert复用 | — |
| **MoEQuant** | arXiv'25 | Expert均衡采样+亲和性引导量化 | — |
| **MoE-Compression** | SC'25 | Expert压缩误差与推理精度分析 | — |
| **Compression Error Sensitivity** | SC Workshop'25 | 不同Expert的压缩误差敏感性 | — |
| **EAC-MoE** | ACL'25 | Expert选择感知压缩 | — |

---

### 🟣 D. 动态 Expert / 共享（Dynamic Expert Sharing）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **Dynamic Expert Sharing** | arXiv'26 | 扩散LLM中解耦内存与并行性 | 40%内存↓ |
| **ElasticMoE** | arXiv'25 | MoE模型高效自动扩缩 | — |
| **MoE-Prism** | arXiv'25 | 解构单体Expert弹性服务 | — |
| **HD-MoE** | arXiv'25 | 3D近存处理的混合动态并行 | — |
| **Mixture of Lookup Experts** | arXiv'25 | 查找表Expert设计 | — |
| **Stratum** | MICRO'25 | 分层3D堆叠DRAM系统硬件协设计 | — |
| **BrainMoE** | NeurIPS'25 | 认知联合嵌入脑基础模型 | — |

---

### 🔴 E. 参数高效微调（PEFT with MoE）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **S'MoRE** | NeurIPS'25 | 残差Expert结构混合参数高效微调 | — |
| **MoLA** | NAACL'25 | LoRA层次Expert分配 | — |
| **FlyLoRA** | NeurIPS'25 | 隐式秩级混合Expert LoRA | — |
| **PT-MoE** | arXiv'25 | Prompt Tuning集成MoE框架 | — |
| **CoMoE** | arXiv'25 | 对比表示PEFT MoE | — |
| **SimSMoE** | NAACL'25 | 表示崩溃感知MoE高效训练 | — |
| **MoLA** | NAACL'25 | LoRA Expert分配层 | — |

---

### ⚫ F. 理论与缩放律（Theory & Scaling）

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **Parameters vs FLOPs** | arXiv'25 | 最优稀疏性的缩放律 | — |
| **Towards Greater Leverage** | arXiv'25 | MoE高效模型缩放律 | — |
| **Joint MoE Scaling Laws** | arXiv'25 | MoE可以内存高效 | — |
| **A Survey on MoE** | TKDE'25 | MoE综述 | — |
| **Every FLOP Counts** | arXiv'25 | 300B MoE无高端GPU训练 | — |
| **The New LLM Bottleneck** | arXiv'25 | 系统视角的潜在注意力和MoE | — |
| **Muon is Scalable** | arXiv'25 | LLM训练的Muon优化器可扩展性 | — |

---

### 🔵 G. 多模态与特殊场景

| 论文 | 发表 | 核心思路 | 关键数字 |
|------|------|---------|---------|
| **MoE-Inference-Bench** | arXiv'25 | MoE大模型推理性能评测基准 | — |
| **I2MoE** | ICML'25 | 可解释多模态交互MoE | — |
| **Efficient Mixture-of-Agents** | arXiv'25 | 树结构路由+自适应剪枝 | — |
| **EvoMoE** | arXiv'25 | 多模态LLM Expert进化 | — |
| **SonicMoE** | arXiv'25 | IO和Tile感知MoE加速 | — |
| **MoES Are Stronger** | arXiv'25 | 超并行推理扩展 | — |

---

## 🗺️ 技术优化全景图

### 核心优化维度总结

```
MoE 系统优化维度（2025-2026）：

                    【训练阶段】
    ┌──────────────────────────────────────────────────┐
    │                                                   │
    │  内存维度                                          │
    │  ├─ 激活内存：MoEBlaze, MemFine                    │
    │  └─ 参数内存：LAER-MoE (FSEP), SwiftMoE           │
    │                                                   │
    │  通信维度                                          │
    │  ├─ 单节点：Comet, FlowMoE                         │
    │  ├─ 跨节点：MegaScale-MoE, ScaleMoE, FUSCO        │
    │  └─ 跨数据中心：HybridEP, MixNet                  │
    │                                                   │
    │  并行策略                                          │
    │  ├─ 5D并行：MoE Parallel Folding                   │
    │  └─ 生产级：MegaScale-MoE (万卡)                   │
    │                                                   │
    └──────────────────────────────────────────────────┘

                    【推理阶段】
    ┌──────────────────────────────────────────────────┐
    │                                                   │
    │  系统架构                                          │
    │  ├─ 分离式：MegaScale-Infer, Janus                │
    │  └─ 异构：KTransformers, HybriMoE                 │
    │                                                   │
    │  Expert 管理                                       │
    │  ├─ 卸载缓存：KTransformers, FloE, fMoE           │
    │  ├─ 预取：PROBE, PopFetcher, Pre-Attn Pred        │
    │  └─ 负载均衡：MicroMoE, Capacity-Aware            │
    │                                                   │
    │  服务框架                                          │
    │  ├─ 大规模：MegaScale-Infer, MixServe             │
    │  └─ 边缘：D2MoE, Remoe (Serverless)              │
    │                                                   │
    └──────────────────────────────────────────────────┘

                    【架构层面】
    ┌──────────────────────────────────────────────────┐
    │  路由优化：OmniMoE, Ada-K, GatePro               │
    │  稀疏压缩：DualSparse, PuzzleMoE, MoEQuant       │
    │  动态共享：Dynamic Expert Sharing, ElasticMoE    │
    │  PEFT：S'MoRE, MoLA, FlyLoRA                    │
    └──────────────────────────────────────────────────┘
```

---

## 🎯 场景化选型指南（更新版）

### 按场景快速选型

| 你的场景 | 第一选择 | 第二选择 | 第三选择 |
|---------|---------|---------|---------|
| **个人/小团队跑 671B MoE** | **KTransformers** | llama.cpp | — |
| **公司 GPU 集群训练 100B MoE** | **SwiftMoE** | **FlowMoE** | LAER-MoE |
| **万卡生产训练** | **MegaScale-MoE** | MoE Parallel Folding | — |
| **高并发推理服务** | **MegaScale-Infer** | Janus | MixServe |
| **单机多卡推理** | **KTransformers** | vLLM+Expert Sharding | — |
| **降低训练内存** | **MoEBlaze** | MemFine | — |
| **消除训练通信瓶颈** | **Comet** | FlowMoE | MegaScale-MoE |
| **新模型架构设计** | **Parameters vs FLOPs** | OmniMoE | LatentMoE |
| **量化/压缩部署** | **MoEQuant** | PuzzleMoE | Sub-MoE |

---

## 📊 深度笔记论文性能速查

| 论文 | 场景 | 最重要指标 | vs 基线 |
|------|------|---------|--------|
| MoEBlaze | 训练 | 速度 + 内存 | 4x 快，50% 内存↓ |
| LAER-MoE | 训练 | 端到端吞吐 | 1.69x |
| SwiftMoE | 训练 | 收敛速度 | +30.5% vs DeepSpeed |
| MemFine | 训练 | 激活内存 | -48% |
| FlowMoE | 训练 | 训练时间 | -57% |
| MoE Parallel Folding | 训练 | MFU | 49.3% @ H100 |
| Comet | 训练 | 吞吐量 | 2.3x, 90%+ overlap |
| MegaScale-MoE | 训练(生产) | 万卡 MFU | ~42% @ 10K GPU |
| OmniMoE | 推理 | 延迟 | 10.9x vs PEER |
| MegaScale-Infer | 推理(服务) | 吞吐+成本 | 3.2x 吞吐, 55% 成本↓ |
| KTransformers | 推理(个人) | 可行性 | $5K 跑 671B 模型 |

---

## 📚 推荐阅读路径（更新版）

### 路径 A：全栈 AI Infra 工程师（4 周）

**第 1 周：理解 MoE 基础和问题**
1. `Parameters vs FLOPs` — 理解 MoE 的理论基础
2. `MoEBlaze` — 训练内存问题本质
3. `MemFine` — 激活内存精细管理

**第 2 周：训练系统优化**
4. `LAER-MoE` — 负载均衡的系统设计
5. `SwiftMoE` — 参数解耦思想
6. `Comet` — 计算通信 Overlap 的极致
7. `FlowMoE` — 调度层面的优化

**第 3 周：大规模系统**
8. `MoE Parallel Folding` — 并行策略设计
9. `MegaScale-MoE` — 工业实践（万卡）

**第 4 周：推理系统**
10. `OmniMoE` — 架构级推理优化
11. `MegaScale-Infer` — 分离式推理架构
12. `KTransformers` — 异构推理的落地

### 路径 B：专注推理系统（2 周）

1. `KTransformers` — 异构推理基础
2. `OmniMoE` — 架构创新
3. `PROBE` — 预取策略
4. `MegaScale-Infer` — 生产系统
5. 选读：`Janus`, `MixServe`, `FloE`

### 路径 C：专注训练优化（2 周）

1. `MoEBlaze` + `MemFine` — 内存优化
2. `Comet` + `FlowMoE` — 通信优化
3. `LAER-MoE` + `SwiftMoE` — 负载均衡
4. `MoE Parallel Folding` — 并行策略
5. `MegaScale-MoE` — 生产落地

---

## 🔮 2025-2026 研究趋势总结

### 主流热点（高关注度）

1. **推理-训练分离架构** ← MegaScale-Infer, Janus 代表
2. **CPU+GPU 异构推理** ← KTransformers 代表
3. **万卡生产系统** ← MegaScale-MoE 代表
4. **细粒度计算-通信 Overlap** ← Comet 代表

### 新兴方向（快速增长）

5. **Expert 智能缓存/卸载** ← 大量 arXiv 论文
6. **MoE 量化与压缩** ← MoEQuant, PuzzleMoE 等
7. **MoE PEFT 微调** ← S'MoRE, MoLA 等
8. **路由架构创新** ← OmniMoE, Ada-K 等

### 相对成熟（研究放缓）

- 简单负载均衡 loss（已有 Auxiliary Loss 标准方案）
- 基础 Expert Parallel（已集成进主流框架）

---

*总览更新于 2026-03-07 | 覆盖 100+ 篇 MoE 优化论文 | AIInfra-Book*
