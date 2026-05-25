# notes — 项目工作日志

按**项目**组织,每个项目目录下:

- 一份 `README.md`(项目总览: 状态 / 时间线 / 下一步 / 文件索引)
- 多篇按时间顺序的 progress / archive note(`YYYY-MM-DD[_HHMM]_<topic>.md`)

> 论文笔记不在这里,在 [`../papers/`](../papers/README.md);周报不在这里,在 [`../weekly/`](../weekly/README.md);
> 稳定的领域知识不在这里,在 [`../knowledge/`](../knowledge/README.md)。

## 当前项目

| 项目 | 主题 | 状态 |
|---|---|---|
| [`career-strategy/`](./career-strategy/README.md) | Agent 时代职业路线: Pilot 主线 + MMOE 硬核副线 | active — 长期职业资产规划 |
| [`gpt-oss/`](./gpt-oss/README.md) | GPT-OSS-20B MLPerf 调优 (MI355X) | active — best 9963 s |
| [`monolith-moe/`](./monolith-moe/README.md) | MoE super-kernel + CCO (MI355X) | 阶段性收尾 — HIP C++ IPC kernel 待落地 |
| [`monolith-ep/`](./monolith-ep/) | MoE expert parallel super-kernel | placeholder — 待启动 |
| [`rocmoe/`](./rocmoe/README.md) | MoE super-kernel v3：Layout-P + receiver-pull + MonolithEP hot loop (MI355X) | active — 设计冻结 (M0 待启动) |
| [`mlperf-llama/`](./mlperf-llama/README.md) | Llama-2-70B LoRA SFT, NeMo vs Primus | active — DataLoader fix + `fp8_param` A/B 待跑 |
| [`pilot/`](./pilot/README.md) | Primus Pilot v2: agentic training-tuning system | active — bootstrap 工具落地,BASELINE 入口待排查 |

## 写 note 的两条 skill

| skill | 用途 | 文件名 |
|---|---|---|
| `progress-note` | 工作过程中即时记一笔(实验跑完 / bug 定位 / 决策点) | `YYYY-MM-DD_HHMM_<topic>.md`(精确到分钟) |
| `archive-notes` | 一段调研或分析结束后做总结归档 | `YYYY-MM-DD_<topic>.md` |

两条 skill 都要求写完后**回写所在项目的 README**(进展时间线 + 下一步)。

## 项目目录命名

`kebab-case`,≤ 3 个词。新项目直接新建目录 + `README.md`,并在本文件的"当前项目"表里加一行。
