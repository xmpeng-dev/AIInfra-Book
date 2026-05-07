---
name: archive-notes
description: >-
  Archive analysis, research notes, and technical findings into the notes/
  directory organized by project. Use when the user asks to save, archive,
  record, or write up analysis results, or says "归档", "记录", "保存笔记",
  "写到 notes".
---

# Archive Notes

将当前对话中的分析、调研、实验结果等内容归档到 `notes/` 目录。

## 目录结构

```
notes/
├── README.md                    # 顶层入口：项目清单
└── <project-slug>/
    ├── README.md                # 项目总览：状态 / 时间线 / 下一步 / 文件索引
    └── YYYY-MM-DD_topic_slug.md
```

- 按 **项目** 分子目录（kebab-case slug），方便跟踪同一项目的进展，例如：
  - `notes/monolith-moe/`     — MonolithMoE super-kernel + MoE comm-overlap
  - `notes/gpt-oss/`          — GPT-OSS-20B MLPerf 调优系列
  - `notes/mlperf-llama/`     — Llama2-70B LoRA (NeMo vs Primus) MLPerf
  - `notes/weekly-reports/`   — 跨项目周报
- 文件名格式：`YYYY-MM-DD_简短英文主题_slug.md`（下划线分隔，全小写）
- 日期取**今天的日期**，前缀保留以便项目内按时间排序
- **每个项目目录必须有 `README.md` 项目总览**（参考现有项目的 README）

## 选择项目目录

1. 如果当前对话明显延续某个已有项目（看 `notes/` 现有子目录），归档进去。
2. 如果是全新项目，新建一个 kebab-case 项目目录，名字简短（≤ 3 个词）。
3. 如果内容横跨多个项目（如周报、综述），放 `notes/weekly-reports/` 或新建一个明确的总览目录。
4. 不确定时先列出现有 `notes/*/` 子目录，再决定。

## 归档流程

1. **确定内容**：从当前对话中提取要归档的分析内容。如果对话中有多个独立主题，分别归档为不同文件。
2. **确定项目**：根据上面的规则选定项目目录（必要时新建）。
3. **生成文件名**：`YYYY-MM-DD_topic.md`，topic 用简短英文，下划线分隔，例如 `2026-04-14_moe_e2e_performance_benchmark.md`。
4. **创建目录**：确保 `notes/<project-slug>/` 目录存在；新项目时同时建 `README.md`（参考其他项目）。
5. **写入内容**：按下方模板组织内容。
6. **回写项目 README**（**必做**）：在 `notes/<project>/README.md` 的 **进展时间线** 加一行（日期 / 里程碑 / 关键数字 / 链接），并按需更新 **下一步** 和 **状态** 节。新项目时同时回写 `notes/README.md` 的"当前项目"清单。
7. **确认**：告知用户文件路径 + 已更新的 README。

## 内容模板

```markdown
# 标题（中文）

**日期**: YYYY-MM-DD

## 背景 / 目标

简要说明分析的背景和目标。

## 主要发现 / 结论

核心结论，用表格或列表呈现关键数据。

## 详细分析

分节展开，包含：
- 实验配置（硬件、软件、参数）
- 数据/表格/代码片段
- 原因分析

## 下一步 / 建议

（可选）后续行动建议。

## 相关文件

（可选）关联的代码、文档路径。
```

## 内容规范

- **语言**：正文用中文，代码/命令/技术术语保留英文原文
- **数据优先**：尽量用表格呈现定量数据，避免纯文字描述性能数字
- **精简**：去除对话中的试探、纠错、重复内容，只保留最终结论和关键推导
- **可追溯**：保留实验命令、脚本路径、环境信息，方便日后复现
- **自包含**：读者无需回溯对话即可理解全部内容

## 示例

用户说："把今天的 MoE overlap 分析归档一下"

→ 已有 `notes/monolith-moe/` 项目目录 → 创建 `notes/monolith-moe/2026-04-14_moe_comm_overlap_analysis.md`，内容从对话中提取整理 → 回写 `notes/monolith-moe/README.md` 时间线追加一行。

用户说："这是一个新项目 attn-fp8 的初步调研，归档"

→ 新建 `notes/attn-fp8/`，写 `notes/attn-fp8/README.md`（项目总览空架子）+ `notes/attn-fp8/YYYY-MM-DD_initial_survey.md` → 在 `notes/README.md` 的"当前项目"清单加一行。
