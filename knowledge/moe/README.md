# knowledge/moe — MoE 系统知识

> 论文笔记不在这里,在 [`../../papers/`](../../papers/README.md);
> 项目进度也不在这里,在 [`../../notes/monolith-moe/`](../../notes/monolith-moe/README.md)。

## 文件索引

| 文件 | 类型 | 内容 |
|---|---|---|
| [`dataflow.md`](./dataflow.md) | 原理 | MoE 前向/反向的详细计算流程,张量形状,All-to-All 通信示例 |
| [`overview-detailed.md`](./overview-detailed.md) | 综述 | "MoE 是什么,为什么需要它" + 计算原理 × 论文优化地图 |
| [`paper-landscape.md`](./paper-landscape.md) | 综述 | 2025-2026 MoE 论文全景图(100+ 篇分类索引,链接到 `papers/`) |
| [`research-overview.md`](./research-overview.md) | 综述 | MoE 研究方向分析:覆盖地图 / 空白 / 创新机会(AI Infra + AMD 视角) |
| [`research-direction-b.md`](./research-direction-b.md) | 深度 | 研究方向 B 的详细展开 |
| [`research-direction-c.md`](./research-direction-c.md) | 深度 | 研究方向 C 的详细展开(关联 [`papers/laer-moe-fsep.md`](../../papers/laer-moe-fsep.md)) |
| [`recent-arxiv.md`](./recent-arxiv.md) | 速记 | MoE 训练工程最近的 arxiv 速览(短笔记,未单独成 paper note) |

## 阅读路径推荐

| 目的 | 推荐顺序 |
|---|---|
| 第一次接触 MoE | `overview-detailed.md` → `dataflow.md` → 看 `papers/comet.md`、`papers/megascale-moe.md` 各一篇 |
| 找新的研究方向 | `paper-landscape.md`(看空白)→ `research-overview.md` → 对应 `research-direction-*.md` |
| 想搞清楚某个工程问题 | 直接搜 `papers/` 索引(`papers/README.md`) |
| 实现 / 调试 monolith MoE | `dataflow.md`(参考) + `notes/monolith-moe/README.md`(状态) + `.cursor/skills/cco-pipeline-overlap/SKILL.md` |

## 编辑约定

- 这里只放**稳定的、跨实验复用**的知识。
- 单次实验或决策记录写到 `notes/<project>/`。
- 单篇论文笔记写到 `papers/<slug>.md`(用 `.cursor/skills/read-paper/` 或 `paper-deep-analysis/`)。
- 引用论文用 `../../papers/<slug>.md` 相对路径。
