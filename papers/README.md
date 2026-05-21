# papers — 论文阅读笔记

每篇论文一个文件(或子目录,当有复现脚本/补充材料时)。
slug 是 kebab-case;年份只在需要消歧义时加。

> 100+ 篇论文的全景分类索引在 [`../knowledge/moe/paper-landscape.md`](../knowledge/moe/paper-landscape.md);
> 这里只列**已有详细笔记**的论文。

## 论文清单

### 训练优化

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| MoEBlaze | arXiv'26 | MoE 训练内存 | 数据结构 + Kernel 融合 + Smart AC,4× 加速 / 50% 内存↓ | [`moeblaze.md`](./moeblaze.md) |
| LAER-MoE / FSEP | ASPLOS'26 | MoE 并行 | FSEP 全分片专家并行 + 动态重排,1.69× 端到端 | [`laer-moe-fsep.md`](./laer-moe-fsep.md) |
| SwiftMoE | arXiv'25 | MoE 训练 | 参数-优化器解耦 + 动态 Expert 放置,+30.5% 收敛 | [`swiftmoe.md`](./swiftmoe.md) |
| MemFine | arXiv'25 | MoE 内存 | 细粒度 chunk 激活调度 + 选择性重计算,48% 内存↓ | [`memfine.md`](./memfine.md) |
| MoE Parallel Folding | arXiv'25 | MoE 并行 | 五维混合并行,Attn/MoE 解耦,49.3% MFU | [`moe-parallel-folding.md`](./moe-parallel-folding.md) |
| Comet | MLSys'25 | MoE 通信重叠 | Tile 级 compute-comm overlap + warp 专用化,1.8× 端到端 | [`comet.md`](./comet.md) |
| MegaScale-MoE | EuroSys'26 | MoE 大规模 | 万卡生产训练系统 + 容错 + 拓扑感知,42% MFU @ 10K GPU | [`megascale-moe.md`](./megascale-moe.md) |
| FlowMoE | NeurIPS'25 | MoE 流水线 | 统一流水线调度 + chunk 优先级,-57% 训练时间 | [`flowmoe.md`](./flowmoe.md) |
| Megatron-Core MoE | -- | 工程参考 | NVIDIA 官方 MoE 实现细节(grouped GEMM / token dispatcher / load balance) | [`megatron-core-moe.md`](./megatron-core-moe.md) |
| veScale FSDP | -- | 分布式训练 | veScale 的 FSDP 设计与实现要点 | [`vescale-fsdp.md`](./vescale-fsdp.md) |

### 推理系统

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
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
