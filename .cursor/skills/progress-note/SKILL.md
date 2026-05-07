---
name: progress-note
description: >-
  Write a progress note into notes/<project>/ with minute-precision filename
  and a detailed timestamp+environment header, covering when / what problem /
  what was done / achieved effect / next directions, then update the project
  README. Use when the user asks to write a note, log progress, record
  current work, or says "写一个 note", "记一下进展", "把这次实验写下来", "log 一下".
---

# Progress Note

把当前工作进展写成一篇 note，存进 `notes/<project>/` 目录，并同步更新该项目的 `README.md`。强调**精确到分钟的文件名 + 完整时间环境头**，方便日后回忆与跟踪同一项目的演进。

跟 `archive-notes` 的区别：

| skill | 触发时机 | 粒度 |
|---|---|---|
| `archive-notes` | 一段调研/分析结束后做总结归档 | 按主题，1 篇 = 1 个完整结论 |
| `progress-note` | 工作过程中即时记一笔（实验跑完、bug 定位完、决策点） | 按时间，1 天可多篇 |

## 工作流

### 1. 取当前时间戳（精确到分钟）

跑一条 `date` 命令拿到本地时间，不要凭印象：

```bash
date "+%Y-%m-%d %H:%M %A"
# 例: 2026-05-07 11:02 Thursday
```

### 2. 选项目目录

按 `notes/<project-slug>/` 放，复用 `archive-notes` 的项目划分约定：

- 看 `notes/` 下已有的子目录（参考顶层 `notes/README.md`），归到对应项目。
- 全新项目就新建一个 kebab-case 目录，名字 ≤ 3 个词；同时建项目 `README.md`（参考已有项目）+ 在顶层 `notes/README.md` 加一行。
- 不确定先列一下 `notes/*/`。

### 3. 生成文件名

格式：`YYYY-MM-DD_HHMM_<topic_slug>.md`

- `HHMM` 为 24 小时制分钟，例：`1102`
- `topic_slug` 用下划线分隔的全小写英文，3–6 个词，能让文件名一眼看出主题
- 例：
  - `2026-05-07_1102_notes_dir_reorg_by_project.md`
  - `2026-04-29_2343_idle_191ms_dataloader_root_cause.md`
  - `2026-04-24_1530_fused_residual_rmsnorm_v1_smoke.md`

### 4. 写入 note 内容（按下方模板）

### 5. 回写项目 README（必做）

写完 note 后，**同时更新 `notes/<project>/README.md`**：

- 在 **进展时间线** 表格里追加一行：`日期 [HH:MM] | 里程碑 | 关键数字 | [链接到本 note]`
- 按需更新 **状态** 节里的"当前最佳 / 项目状态 / 上次更新"
- 按需更新 **下一步（按 ROI）** 表格（移除已完成的，加新的方向）

新项目时还要：
- 建 `notes/<project>/README.md`（参考已有项目结构：状态 / 环境 / 进展时间线 / 下一步 / 文件索引 / 维护约定）
- 在顶层 `notes/README.md` 的"当前项目"清单加一行

### 6. 告知用户最终路径（含 README 改动）

---

## Note 内容模板

```markdown
# <一句话中文标题>

> 时间: YYYY-MM-DD HH:MM (Asia/Shanghai)
> 项目: <project-slug>
> 硬件: <e.g. 8x AMD Instinct MI355X (gfx950), XGMI 全互联 / 1 节点 8 GPU>
> 容器: <镜像名:tag, 例 rocm/mlperf-training:llama2_70b_training_6.0_2026-04-27-22-49-59>
> 软件: <PyTorch / ROCm / Triton / 其他关键栈版本>
> 代码: <repo / branch / commit hash 或 worktree 路径>

## 1. 时间点 / 上下文

- 上一次相关进展：链接到上一篇 note（同项目）或简述 1 行
- 触发本次工作的事件：例如某次 trace 跑完、用户提出新需求、上游 PR 合入

## 2. 问题

要解决的问题是什么。**一句话 + 量化指标**：

- 现状：<指标 = 数值>（例：稳态 step = 862 ms / 479 TFLOP/s/GPU）
- 目标：<指标 = 数值>（例：把 step 压到 < 770 ms）
- 卡点：<已知瓶颈 / 假设 / 风险>

## 3. 做了什么

按时间或步骤列，**保留可复现的命令、脚本路径、关键 diff**：

| # | 动作 | 关键命令 / 文件 | 备注 |
|---|---|---|---|
| 1 | … | … | … |

或用步骤列表：

1. **<动作 1>** — 命令 / patch
2. **<动作 2>** — …

## 4. 效果

**用表格呈现定量数据**，避免"提升了一些"这类描述：

| 指标 | Before | After | Δ |
|---|---|---|---|
| step (ms) | … | … | … |
| TFLOP/s/GPU | … | … | … |
| HBM peak (GB) | … | … | … |
| 收敛 (val loss) | … | … | … |

定性观察：

- ✅ 达到的目标
- ⚠️ 副作用 / 待确认
- ❌ 没达成的目标

## 5. 可持续方向

下一步可以继续推的方向，按 ROI 排序：

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| P0 | … | … | … |
| P1 | … | … | … |
| P2 | … | … | … |

## 相关文件

- 代码 / patch：`<path>`
- 上游 note：`notes/<project>/<prev-note>.md`
- 原始 trace / log：`<path>`
```

---

## 内容规范

- **语言**：正文中文，命令/代码/术语保留英文
- **量化优先**：所有效果用表格 + 数字，不用"明显加快"
- **可复现**：保留命令、脚本路径、commit hash、镜像 tag
- **自包含**：读者只看这一篇 + 链接的上一篇就能跟上进度
- **不删旧 note**：进展写新 note，老 note 不改（除非加 superseded 横幅）
- **时间用 24h**：`14:30`，不是 `2:30 PM`
- **README 是单一进展真源**：项目当前状态以 `notes/<project>/README.md` 为准，旧 note 描述的"当前最佳"自然过期不必回改

## 反例

❌ 文件名 `2026-05-07_optimization.md` — 没分钟、topic 太泛
❌ 文件名 `note_today.md` — 既无时间也无主题
❌ "把 step 压低了" — 没数字
❌ 跨项目混在一篇里 — 拆成多篇分别归项目
❌ 写完 note 没回写项目 README — 进展跟踪就断了

## 示例

用户说："把刚才 fused residual rmsnorm 的 80-iter smoke 结果记一下"

→ `date` 拿到 `2026-04-24 18:42`
→ 项目 = `gpt-oss`（已有目录）
→ 文件名：`notes/gpt-oss/2026-04-24_1842_fused_residual_rmsnorm_v1_80iter_smoke.md`
→ 头部：时间 `2026-04-24 18:42`，硬件 `8x MI355X`，容器/软件填实际版本
→ 5 节按模板写满
→ **回写** `notes/gpt-oss/README.md`：进展时间线追加 `2026-04-24 18:42 | Tier 1A V1 80-iter smoke | step −1.04% / +1.04% TFLOP/s, 0 NaN | [链接]`
