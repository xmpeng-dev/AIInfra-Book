# knowledge/systems — 训练系统/框架知识

## 文件索引

| 文件 | 内容 |
|---|---|
| [`primus-pipeline-runtime-megatron-integration.md`](./primus-pipeline-runtime-megatron-integration.md) | **Primus 流水线 runtime 如何接入 Megatron** — 以 PP=4/1f1b 走通 plan→adapter→executor→handler 五层调用链；执行权移交的绑定点、plan 兼数据流图、延迟 wgrad 机制、接新后端的三样交付物、当前层泄漏 |
| [`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md) | **arXiv 2025–2026 训练/推理优化全景**（MoE overlap · 优化器 · 基础设施 · 推理→训练迁移 · 机构索引 · 优先阅读队列） |
| [`arxiv-digest-2026-08.md`](./arxiv-digest-2026-08.md) | **arXiv 2026-08 增量扫描**（2026-05~08 投稿，1157→92→27 篇；执行模型 / EP 通信 / AMD 建模 / 内核 DSL / 训练并行 / 低精度；含最该先读的五篇及其精读笔记入口） |
| [`industry-training-optimization-2026.md`](./industry-training-optimization-2026.md) | **大厂训练侧优化动向**（2026-03~08，按机构组织）——**双来源**：arXiv（作者单位逐篇核实）+ 开源仓库/工程博客，含两者的交叉印证与冲突。国内字节 / 阿里 / 腾讯 / 华为 / 百度 / 月之暗面 / 美团 等，海外 Meta / NVIDIA / AMD / 微软+OpenAI / Google / AWS 等；横切主线：FP4 训练 · RL 后训练栈 · 超节点通信库 · 弹性状态迁移 · Muon 跨栈共识；附对 ROCmoe / FlyDSL / Primus 的具体差距清单 |
| [`torchtitan-diff-2025-10-vs-2026-04.md`](./torchtitan-diff-2025-10-vs-2026-04.md) | TorchTitan 半年跨度的 diff 总结(API / 训练循环 / 并行 / 优化器变更) |
| [`training-1024g-stability-interview-notes.md`](./training-1024g-stability-interview-notes.md) | 1024-GPU 训练稳定性面试笔记(checkpoint / failover / 监控 / RCCL 调优) |

## 什么时候来查

- 想知道某家大厂最近在训练侧做什么 / 找对标基线 → 看 industry-training-optimization-2026
- 准备升级 TorchTitan / 排查跨版本回归 → 看 diff
- 设计大规模训练的容错与监控 → 看 stability 笔记
- 比较 Primus / TorchTitan / Megatron 三个 backend 的工程差异 → 配合 `.cursor/skills/backend-gap-report/SKILL.md`
- 改 Primus 流水线调度 / 加新调度算法 / 给新后端接 runtime → 看 primus-pipeline-runtime-megatron-integration

## TODO

- 增补 Megatron-Core MoE 的工程实现要点(已有 paper note `papers/megatron-core-moe.md`,但工程经验单独成文更好)
- Primus 的执行层设计要点已成文(见上表首行);后续补 comm / precision / memory 三块的对应走读
