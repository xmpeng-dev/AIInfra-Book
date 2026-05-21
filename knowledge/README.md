# knowledge/ — 稳定的领域知识

这一层是**不易变的领域 know-how**，区别于:

- `papers/` — 论文阅读笔记(每篇一文件,只增不改)
- `notes/<project>/` — 项目工作日志(append-only)
- `weekly/` — 周报存档
- `.cursor/skills/` — 任务驱动的标准操作流程

agent 在写代码、设计、回答问题时,默认先 `Grep` 这里。

## 子领域

| 目录 | 内容 | 何时来查 |
|---|---|---|
| [`hardware/`](./hardware/README.md) | AMD MI300X/MI325X/MI355X + NVIDIA H100/H200/B200 规格、CDNA 架构、MFMA 指令、参考资料 | 写 kernel、对照硬件能力、做选型表 |
| [`systems/`](./systems/README.md) | TorchTitan / Megatron-Core / Primus 等训练框架的工程要点、跨版本 diff、大规模训练稳定性 | 集成 backend、debug 跨版本回归、做架构选型 |
| [`moe/`](./moe/README.md) | MoE 的算法、数据流、并行方案、论文全景图、研究方向 | 实现 MoE 训练系统、设计新的并行/通信策略、写 MoE 论文笔记前的 sanity check |
| [`kernels/`](./kernels/README.md) | GEMM / FP8 / comm-compute overlap 等可复用 kernel know-how | 写或优化 HIP/CUDA kernel |
| [`libraries/`](./libraries/README.md) | 第三方算子库(CK / AITER / hipBLAS / primus-turbo 等)的设计蒸馏 | 选库、读上游代码前的 sanity check、跨库 pattern 比较 |
| [`pilot/`](./pilot/README.md) | Primus Pilot v2 自动调优系统的设计文档(workflow / state machine / sub-skills) | 实现 / 调试 Pilot,或想理解它的搜索策略 |

## 编辑约定

- 每个子目录恰好一个 `README.md`,作为本目录的索引/路由。
- 文件名 kebab-case,一篇文档只覆盖一个概念。
- 超过 30KB 的文档要拆分:把每个二级标题独立成文件,README 做索引。
- 增加新文件时,**同步更新所在目录的 `README.md` 表格**。
- 引用论文笔记一律用 `../../papers/<slug>.md` 相对路径。
