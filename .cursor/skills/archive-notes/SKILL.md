---
name: archive-notes
description: >-
  Archive analysis, research notes, and technical findings into the notes/
  directory organized by project. Use when the user asks to save, archive,
  record, or write up analysis results, or says "归档", "记录", "保存笔记",
  "写到 notes".
---

# Archive Notes

Archive analysis, research notes, experiment results, and technical findings from the current conversation into the `notes/` directory.

## Directory Structure

```
notes/
├── README.md                    # Top-level index: project list
└── <project-slug>/
    ├── README.md                # Project overview: status / timeline / next steps / file index
    └── YYYY-MM-DD_HHMM_topic_slug.md
```

- Organize notes by **project** using kebab-case slugs so progress for each project is easy to track. Examples:
  - `notes/monolith-moe/`     — MonolithMoE super-kernel + MoE comm-overlap
  - `notes/gpt-oss/`          — GPT-OSS-20B MLPerf tuning series
  - `notes/mlperf-llama/`     — Llama2-70B LoRA (NeMo vs Primus) MLPerf
  - `notes/pilot/`            — Primus Pilot v2 auto-tuner project
- **Cross-project weekly reports go to `weekly/<year>/`, NOT into `notes/`.** Use the existing files there as the template.
- **Single-paper reading notes go to `papers/<slug>.md`, NOT into `notes/`.** Use the `read-paper` or `paper-deep-analysis` skill.
- File naming format: `YYYY-MM-DD_HHMM_short_english_topic_slug.md` using local time, 24-hour `HHMM`, underscores, and lowercase words.
- Use **today's date and current minute** as the prefix so files sort chronologically within a project and are easy to review in sequence.
- **Every project directory must have a `README.md` project overview**. Follow existing project README patterns.

## Choose the Project Directory

1. If the conversation clearly continues an existing project, archive the note under that existing `notes/<project>/` directory.
2. If it is a new project, create a short kebab-case project directory name, preferably no more than three words.
3. If the content spans multiple projects (weekly summary, cross-project strategy), put it under `weekly/<year>/` or create a clear overview project under `notes/`. Single-paper notes go to `papers/`, not `notes/`.
4. When unsure, inspect the existing `notes/*/` directories first, then choose the best fit.

## Archive Workflow

1. **Identify the content**: Extract the final analysis, research findings, or experiment results from the current conversation. If the conversation contains multiple independent topics, archive them as separate files.
2. **Choose the project**: Select the project directory using the rules above. Create a new project directory when needed.
3. **Generate the filename**: Use `YYYY-MM-DD_HHMM_topic.md`, where `HHMM` is the current local time to minute precision and `topic` is a short English slug separated by underscores, for example `2026-04-14_1530_moe_e2e_performance_benchmark.md`.
4. **Create directories**: Ensure `notes/<project-slug>/` exists. For a new project, also create its `README.md` using existing project READMEs as examples.
5. **Write the note**: Organize the note using the template below.
6. **Update the project README** (**required**): Add one row to the **progress timeline** in `notes/<project>/README.md` with date, milestone, key numbers, and link. Update **next steps** and **status** when appropriate. For a new project, also update the "current projects" table in `notes/README.md`.
7. **Confirm**: Tell the user the note path and which README files were updated.

## Content Template

```markdown
# Title in Chinese

**日期**: YYYY-MM-DD HH:MM

## 背景 / 目标

Briefly explain the background and goal of the analysis.

## 主要发现 / 结论

Summarize the core conclusions. Use tables or lists for key data.

## 详细分析

Expand by section. Include:
- Experiment setup: hardware, software, parameters.
- Data, tables, or code snippets.
- Root-cause analysis.

## 下一步 / 建议

Optional follow-up actions or recommendations.

## 相关文件

Optional related code paths or document paths.
```

## Content Guidelines

- **Language**: Write the archived note body in Chinese. Keep code, commands, paths, and technical terms in their original English form.
- **Data first**: Prefer tables for quantitative data. Avoid describing performance numbers only in prose.
- **Concise**: Remove exploratory dialogue, corrections, and repetition. Keep only final conclusions and key reasoning.
- **Traceable**: Preserve experiment commands, script paths, environment details, and relevant file paths so the work can be reproduced later.
- **Self-contained**: The reader should not need to revisit the conversation to understand the note.

## Examples

User request: "Archive today's MoE overlap analysis."

→ Existing project directory: `notes/monolith-moe/`  
→ Create `notes/monolith-moe/2026-04-14_1530_moe_comm_overlap_analysis.md`  
→ Extract and condense the conversation into the note  
→ Update the timeline in `notes/monolith-moe/README.md`

User request: "This is the initial survey for a new attn-fp8 project. Archive it."

→ Create `notes/attn-fp8/`  
→ Write `notes/attn-fp8/README.md` as the project overview  
→ Write `notes/attn-fp8/YYYY-MM-DD_HHMM_initial_survey.md`  
→ Add one row to the "current projects" table in `notes/README.md`
