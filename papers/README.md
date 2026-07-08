# papers — 论文阅读笔记

每篇论文一个文件(或子目录,当有复现脚本/补充材料时)。
slug 是 kebab-case;年份只在需要消歧义时加。

> 100+ 篇论文的全景分类索引在 [`../knowledge/moe/paper-landscape.md`](../knowledge/moe/paper-landscape.md);
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

### 推理系统

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| Fleet | arXiv'26 | Chiplet megakernel | Chiplet-task + 协作 L2 tiling,MI350 decode 1.3–1.5× vs vLLM eager | [`fleet.md`](./fleet.md) |
| MegaScale-Infer | SIGCOMM'25 | MoE 推理 | 分离式 EP,Prefill/Decode/Expert 解耦,3.2× 吞吐 / 55% 成本↓ | [`megascale-infer.md`](./megascale-infer.md) |
| KTransformers | SOSP'25 | 异构推理 | CPU+GPU 异构推理,$5K 跑 DeepSeek-V3 671B | [`ktransformers.md`](./ktransformers.md) |

### 架构创新

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| OmniMoE | arXiv'26 | MoE 路由 | 原子专家 + 笛卡尔积路由 O(√N),10.9× 推理加速 | [`omnimoe.md`](./omnimoe.md) |
| LatentMoE | -- | MoE 架构 | Latent space MoE 设计要点 | [`latent-moe.md`](./latent-moe.md) |

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
