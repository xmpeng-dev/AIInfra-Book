# notes — 工作笔记

按**项目**组织，每个项目目录下：
- 一份 `README.md`（项目总览：状态 / 时间线 / 下一步 / 文件索引）
- 多篇按时间顺序的 progress / archive note（`YYYY-MM-DD[_HHMM]_<topic>.md`）

## 当前项目

| 项目 | 主题 | 状态 |
|---|---|---|
| [`career-strategy/`](./career-strategy/README.md) | Agent 时代职业路线：Pilot 主线 + MMOE 硬核副线 | active — 长期职业资产规划 |
| [`gpt-oss/`](./gpt-oss/README.md) | GPT-OSS-20B MLPerf 调优 (MI355X) | active — best 9963 s |
| [`monolith-moe/`](./monolith-moe/README.md) | MoE super-kernel + CCO (MI355X) | 阶段性收尾 — HIP C++ IPC kernel 待落地 |
| [`mlperf-llama/`](./mlperf-llama/README.md) | Llama-2-70B LoRA SFT, NeMo vs Primus | active — DataLoader fix + `fp8_param` A/B 待跑 |
| [`weekly-reports/`](./weekly-reports/README.md) | 跨项目周报 | 每周一篇 |

## 写 note 的两条 skill

| skill | 用途 | 文件名 |
|---|---|---|
| `progress-note` | 工作过程中即时记一笔（实验跑完 / bug 定位 / 决策点） | `YYYY-MM-DD_HHMM_<topic>.md`（精确到分钟） |
| `archive-notes` | 一段调研或分析结束后做总结归档 | `YYYY-MM-DD_<topic>.md` |

两条 skill 都要求写完后**回写所在项目的 README**（进展时间线 + 下一步）。

## 项目目录命名

`kebab-case`，≤ 3 个词。新项目直接新建目录 + `README.md`，并在本文件的"当前项目"表里加一行。
