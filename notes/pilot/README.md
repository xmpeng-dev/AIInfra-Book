# pilot — Primus Pilot v2：agentic training-tuning system

> **目标**：把训练系统调优做成由 LLM agent 在 Cursor / Claude Code / Codex 里循环驱动的 closed loop —— PREFLIGHT → PROJECTION → SMOKE → BASELINE → OPTIMIZE_LOOP（OBSERVE → DIAGNOSE → REPLAN → EXECUTE → CORRECTNESS_LITE → SETTLE，可选 ENV_SWEEP）→ REPORT → LEARN
> **位置**：`Primus/pilot/`（knowledge + tool 包，本身无 runtime；runtime 由具体 agent 框架提供）
> **关键入口**：
> - `.cursor/rules/00-pilot-core.mdc`（Cursor 自动加载；Part I 规则 + Part II 五步 recipe + Part III 退出条件）
> - `pilot/AGENTS.md`（角色契约 / reading scope 权威）
> - `pilot/skills/workflow/{state_machine,orchestration}.md`（路由 + bootstrap 协议）
> - `pilot/tools/`（Tool 集，统一 `python -m pilot <subcmd>`）
> - `pilot/schemas/`（数据契约）

## 状态

| 维度 | 状态 |
|---|---|
| Bootstrap (`pilot session init`) | ✅ 单条命令完成（schema 校验 + r0 checkpoint） |
| Cursor agent 启动链 | ✅ rules + AGENTS + orchestration 去重，reading scope 单点权威 |
| PREFLIGHT / PROJECTION / SMOKE | ✅ 已在 mi355-gpu-26 (1×8 MI355X) 跑通 |
| BASELINE | 🟡 入口出现非确定性现象，待排查 |
| OPTIMIZE_LOOP / REPORT / LEARN | 🔴 未跑通端到端 |
| 下游 tool `--tuning-config` 串联 | 🔴 submit / tune_single / profiler / observe / preflight 都待接 |

项目状态：**active**，2026-05-11 起把"Cursor agent 主导的 Pilot tuning closed loop"作为主线推进。

## 进展时间线

| 日期 | 里程碑 | 关键结论 | 来源 note |
|---|---|---|---|
| 2026-05-11 20:29 | `pilot session init` 工具 + Cursor rule 梳理 | bootstrap 收成 1 条命令；rules ↔ doc 不再循环引用 README / 彼此；在 mi355-gpu-26 走通 PREFLIGHT / PROJECTION / SMOKE，BASELINE 入口暂停 | [`2026-05-11_2029_pilot_session_init_tool_and_rule_hygiene`](./2026-05-11_2029_pilot_session_init_tool_and_rule_hygiene.md) |

## 下一步（按 ROI）

| 优先级 | 方向 | 预期 | 备注 |
|---|---|---|---|
| P0 | 排查 BASELINE 入口非确定性 | `dsv2_lite_fp8_mi355x_*` session 能跑过 BASELINE 进 OPTIMIZE_LOOP | 看 `r0/baseline_run/*.log` 与 SMOKE 的 metric 差异 |
| P0 | `--tuning-config` 串进 `pilot.tools.submit.run` | agent 调用面缩成 `submit run --tuning-config <path> --stage <name>` | 改 submit argparse + 默认值解析 |
| P1 | `--tuning-config` 串进 `tune_single.run / profiler.run / observe.* / preflight.*` | 各 stage 自动落点 `state/<sid>/<trace_subdir>/<stage>/`，r0 / r1 / ... 分桶 | 每个 tool 接 `--stage` + `--trial-id` |
| P1 | 端到端 smoke：`session init` → `submit run --tuning-config --stage smoke` | 证实 trace_subdir / stage / trial_id 串通 | 等 P0 完成 |
| P2 | `prompts/orchestrator.md` 与 `00-pilot-core.mdc` Part II 去重 | 让 framework-agnostic（Python harness）与 Cursor-specific 入口各司其职 | |
| P2 | 清掉 `orchestration.md` 残留 `README.md §X.X` 溯源引用 | 与 rules 对齐，skill 也只引 schema 文件 | doc-only |

## 文件索引

| 主题 | 文件 |
|---|---|
| `pilot session init` 工具 + Cursor rule 梳理（首篇） | [`2026-05-11_2029_pilot_session_init_tool_and_rule_hygiene.md`](./2026-05-11_2029_pilot_session_init_tool_and_rule_hygiene.md) |

## 维护约定

- 每次推进 Pilot（新增 tool / 调整 state machine / 跑通新阶段 / 修一类 agent 行为 / 重大 doc-rule 重构）都写一篇 progress note，并回写本 README 的 **进展时间线** 与 **下一步**。
- 试运行 session 落在 `pilot/state/<session_id>/`；note 引用具体 session 时直接给路径，不抄内容。
- Pilot v2 的 source-of-truth 始终在仓库 `pilot/` 下；本目录只是工作过程笔记，不要在这里复述 spec。
