# career-strategy — Agent 时代职业路线

> **目标**：把当前的训练系统、GPU 性能优化、分布式训练经验，转化成 agent 时代更长期、更有壁垒的职业主线。
> **核心定位**：Agentic AI Infrastructure / Agentic Training Systems Engineer。

## 状态

| 维度 | 判断 |
|---|---|
| 主线方向 | `Primus/pilot`：agentic training tuning system |
| 硬核副线 | `MMOE`：AMD MoE super-kernel / GPU kernel optimization |
| 长期叙事 | 用 agent 自动化训练调优闭环，用 kernel 能力解决真正困难的性能问题 |
| 当前策略 | Pilot 作为职业资产主线推进，MMOE 作为底层硬实力与高价值 case study |

## 进展时间线

| 日期 | 里程碑 | 关键结论 | 来源 note |
|---|---|---|---|
| 2026-05-08 | Agent 时代职业路线梳理 | `Primus/pilot` 是最适合作为主线押注的项目之一；`MMOE` 是高壁垒 GPU kernel 副线/杀手锏 | [agentic_infra_and_gpu_kernel_career_strategy](./2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md) |

## 下一步

| 优先级 | 方向 | 说明 |
|---|---|---|
| P0 | 把 `Primus/pilot` 推到真实多节点训练闭环 | 打通 preflight / submit / observe / diagnose / replan / report |
| P1 | 给 Pilot 加 profiler-driven diagnosis | 让 trace、log、metrics 驱动瓶颈判断和下一轮配置选择 |
| P2 | 把 `MMOE` 作为 Pilot 的高价值 benchmark case | 用 agent 管理 kernel sweep、性能归因、报告生成 |
| P3 | 建 agent eval | 衡量 agent 是否能稳定减少 GPU-hour 浪费、定位失败、提出有效 next step |

## 文件索引

| 主题 | 文件 |
|---|---|
| Agentic infra + GPU kernel 职业策略 | [`2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md`](./2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md) |

## 维护约定

每次出现职业路线、项目取舍、长期技术定位相关的重要判断时，写一篇 archive note，并同步更新本 README。
