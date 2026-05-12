# Pilot session init 工具 + Orchestrator 规则梳理

> 时间: 2026-05-11 下午 ~ 2026-05-12 上午 (UTC+8)
> 项目: pilot
> 硬件: 试运行节点 mi355-gpu-26 / 1 × 8 MI355X (gfx950)；工具/规则改动本身无硬件依赖
> 容器: xiaoming-dev (Podman) 在节点上运行；Pilot tool 在容器外的 workspace 直接 `python -m`
> 软件: Pilot v2 in-repo（`pilot/`）；Python 3.11，jsonschema，pyyaml
> 代码: `/shared/amdgpu/home/xiaoming_peng_qle/workspace/Primus`（本地 worktree，未提交）

## 1. 时间点 / 上下文

- **起因**：要在 mi355-gpu-26 上跑 DeepSeek-V2-Lite (FP8) 的 Pilot tuning，触发一次完整 PREFLIGHT → PROJECTION → SMOKE → BASELINE 验证流程。开干之后暴露两类问题：
  1. Session 启动**没有 session-wide 配置文件**，每个 stage 都要重复传 `--plan / --cluster-config / --log-dir / 各种 override`，agent ↔ tool 之间 contract 太松。
  2. Cursor 启动 agent 时加载的 `.cursor/rules/` 与 `pilot/AGENTS.md` / `pilot/skills/workflow/orchestration.md` 之间**循环引用 + `pilot/README.md` 多次出现**，agent 实际能 follow 的"最小读取集合"模糊。
- **前一篇相关 note**：暂无（本项目首篇）。

## 2. 问题

| 子问题 | 现状 | 目标 |
|---|---|---|
| Session 启动 | agent 要手算 `session_id` 字符串、手写两份 yaml、再手动 `state.checkpoint` | 一条命令完成：建目录 + 写三份 yaml + r0 checkpoint，全部 schema 校验 |
| `tuning_config` 缺位 | 没有 session-wide 配置，每个 stage 重传同一组参数 | 引入 `state/<session_id>/tuning.yaml`：`log_dir_prefix` / `trace_subdir` / `target` / `base_overrides` / per-stage 默认值都集中 |
| Cursor rule 重复 | `00-pilot-core.mdc` 写不变式、`10-orchestrator-role.mdc` 写五步 recipe，两份都是 `alwaysApply: true` 且 audience 完全一致 | 合并成一份 `00-pilot-core.mdc`：Part I 规则 / Part II 五步 recipe / Part III 退出条件 |
| `README.md` 在 rules 里满天飞 | 多份 `.cursor/rules/*.mdc` 引用 `pilot/README.md §X.X`，README 1789 行太大不适合 always-on 引用 | rule 只引 `AGENTS.md` 和 `pilot/schemas/*.schema.json`，不再绕 README |
| Reading list 循环 | `00-` → `AGENTS.md §4` → 列必读 3 份；`orchestration.md §0.1` 又把 `00-` 列回去 | reading scope 权威单点 = `AGENTS.md §4`；`orchestration.md` 不再管 reading list |
| `session_id` 漂移 | doc 写单下划线 `<model>_<cluster>_<ts>`、code 用双下划线 `<model>__<cluster>__<ts>` | doc 与 code 对齐，并把 doc 改成调用 `pilot session init`，agent 不再手算 |

## 3. 做了什么

### 3.1 新增 `pilot session init` 工具链

- **`pilot/schemas/tuning_config.schema.json`**（新建）：session-wide 配置 schema，约束 `log_dir_prefix` / `trace_subdir` / `target` / 每个 stage 默认值 / `optimize_loop.dir_pattern` 必须含 `{trial_id}` 等。
- **`pilot/tools/session.py`**（新建）：
  - `_derive_model_id`（plan yaml stem → snake_case）+ `_derive_cluster_id`（读 `cluster.yaml` 的 `cluster_id`）+ `_default_session_id() = f"{model}__{cluster}__{ts}"`
  - `init()`：建 `state/<session_id>/`，写 `tuning.yaml` / `target_vector.yaml` / `tuning_state.yaml`（`current_stage=PREFLIGHT`），schema 校验，落 `r0/` checkpoint，幂等（默认拒覆盖，`--force` 强写）
  - CLI 入口接 `--plan / --cluster-config / --session-id / --primary / --rounds / --candidates-per-round / --smoke-iters / --train-iters / --timeout-s / --base-override / --constraint / --trace-subdir / --node / --notes / --force`
- **`pilot/cli/subcommands/session.py`**（新建）：把 `pilot session` 挂到统一 `python -m pilot` 入口，与 `python -m pilot.tools.session` 等价。
- **`pilot/tools/_tuning_config.py`**（新建）：loader + `TuningConfig` 类，给后续每个 tool 用 `--tuning-config` 时统一解析 `plan` / `cluster_config` / `base_overrides` / `stage_trace_dir(stage, trial_id=...)`。

### 3.2 Cursor rule 与 pilot doc 梳理

- **`pilot/AGENTS.md`**：删 §1 Language + 原始中英文混用的 rationale；§2–§7 重新编号成 §1–§6（reading list 移到新 §4）。
- **`.cursor/rules/00-pilot-core.mdc`**：
  - 把原 `10-orchestrator-role.mdc` 整体并入，结构 = Part I（3 条规则：state-first / 角色隔离 / tool boundary）+ Part II（`decide → spawn → apply → checkpoint → trim` 五步 recipe）+ Part III（退出条件）
  - 删 `pilot/README.md` 全部 cross-ref（§2.2 / §8.11 / §7.7 等）；schema 引用改成 `pilot/schemas/<x>.schema.json`
- 删 **`.cursor/rules/10-orchestrator-role.mdc`**。
- **`.cursor/rules/90-tool-invocation.mdc`** + 它的镜像 `pilot/integrations/cursor/rules/90-tool-invocation.mdc`：删 README 链；`see 10-orchestrator-role.mdc` → `see 00-pilot-core.mdc Part I §2`。
- **`.cursor/rules/30-worker-{preflight,diagnose}.mdc`**：`pilot/README.md §8.x` → `pilot/schemas/<x>_report.schema.json`。
- **`pilot/integrations/cursor/{README.md,AGENTS.md}`**：所有 `10-orchestrator-role.mdc` → `00-pilot-core.mdc`（必要处加 `Part II` 后缀）。
- **`pilot/skills/workflow/orchestration.md`**：
  - 删 §0.1 Required reading order（与 `AGENTS.md §4` 重复 → 循环引用根源）
  - §0.2–§0.5 → §0.1–§0.4，内部交叉引用同步
  - §0.3 New-session protocol：从"手算 `session_id` → 写两份 yaml → checkpoint → handoff" 4 步改成"`python -m pilot session init ...` 一条命令 → handoff" 2 步
- **`pilot/schemas/tuning_config.schema.json`** L5：`orchestration.md §0.4` → `§0.3`（同步重新编号）。

### 3.3 试运行结果

- 在 mi355-gpu-26 起的会话 `dsv2_lite_fp8_mi355x_20260511T075126Z`（user 显式传的 session_id；新版默认会生成 `<model>__<cluster>__<ts>`）走通 PREFLIGHT (cache hit) → PROJECTION → SMOKE。
- SMOKE 10/10 iter，rc=0，loss=10.29 finite，per-iter ≈ 1.40 s，tps ≈ 23.3 K。
- 在 BASELINE 入口暂停——用户观察到"运行有点随机"，留作下次排查。

## 4. 效果

| 维度 | Before | After | Δ |
|---|---:|---:|---|
| Session 启动步数（agent 视角）| 4 步（手算 sid + 写两份 yaml + checkpoint + handoff） | 2 步（`pilot session init` + handoff） | −2 步，全部 schema 校验 |
| `.cursor/rules/` 文件数 | 13（含 `10-orchestrator-role.mdc`） | 12 | −1（合并入 `00-`） |
| `.cursor/rules/` 内 `pilot/README.md` 引用数 | 6 | 0 | −6 |
| `00-pilot-core.mdc` 大小 | 57 行（只含 invariants） | 139 行（Part I + II + III） | 单文件，结构清晰 |
| Reading scope 权威 | 2 处（`AGENTS.md §4` + `orchestration.md §0.1`，互相回指） | 1 处（`AGENTS.md §4`） | 循环断开 |
| Doc vs code 一致性 | `session_id` 双下划线 vs 单下划线；`§0.4` vs `§0.3` | 全部对齐 | 0 漂移项 |

定性：

- ✅ **Agent 启动链可演练**：用户一句话 → Cursor 自动加载 `00-` → 跟着 `AGENTS.md §4` 读 3 份 framework-agnostic 文件（`prompts/orchestrator.md`、`state_machine.md`、`orchestration.md`）→ `orchestration.md §0.1` 二分 new / resume → `pilot session init` → 进 `00-` Part II 五步 recipe 循环。
- ✅ Bootstrap 工具自带 schema 校验，后续加字段也有硬约束。
- ⚠️ BASELINE 入口的"运行有点随机"现象未排查，下次接续要先看。
- ❌ `--tuning-config / --stage / --trial-id` 还没串进 `submit / tune_single / profiler / observe / preflight` 等下游 tool，目前 stage 间仍要重复传参。

## 5. 可持续方向

| 优先级 | 方向 | 预期 | 风险 / 前置 |
|---|---|---|---|
| P0 | 排查 BASELINE 入口的非确定性 | 让 `dsv2_lite_fp8_mi355x_*` session 能跑过 BASELINE 进入 OPTIMIZE_LOOP | 需要复现一次问题 + 看 `r0/baseline_run/*.log` |
| P0 | 把 `--tuning-config` 串进 `pilot.tools.submit.run` | agent 调用面缩成 `submit run --tuning-config <path> --stage <name>`；plan / cluster / log_dir / base_overrides 自动注入 | 改 submit 的 argparse 与默认值解析 |
| P1 | `--tuning-config` 串进 `tune_single.run / profiler.run / observe.* / preflight.*` | 各 stage 自动落到 `state/<sid>/<trace_subdir>/<stage>/`，r0 / r1 / ... 分桶 | 每个 tool 接 `--stage` + `--trial-id` |
| P1 | 端到端 smoke：`session init` → `submit run --tuning-config --stage smoke` → 验文件落点 | 证实 trace_subdir / stage / trial_id 串通 | 等 P0 完成 |
| P2 | `prompts/orchestrator.md` 与 `00-pilot-core.mdc` Part II 内容去重 | 一份 framework-agnostic（Python harness 用），一份 Cursor-specific；目前重叠 | 拆 source-of-truth：Part II 引用 prompts，或在 `AGENTS.md §4` 标 Cursor 路径可跳过 prompts |
| P2 | 把 `orchestration.md` 残留的 `README.md §7.7` 等溯源引用清掉 | 与 rules 一致：skill 也只引 schema 文件 | doc-only，低优先 |

## 相关文件

- 新增工具：`pilot/tools/session.py`、`pilot/tools/_tuning_config.py`、`pilot/cli/subcommands/session.py`
- 新增 schema：`pilot/schemas/tuning_config.schema.json`
- 合并 / 清理后的 rules：`.cursor/rules/00-pilot-core.mdc`、`.cursor/rules/90-tool-invocation.mdc`、`.cursor/rules/30-worker-{preflight,diagnose}.mdc`
- 已删除：`.cursor/rules/10-orchestrator-role.mdc`
- 同步更新：`pilot/AGENTS.md`、`pilot/integrations/cursor/{README.md,AGENTS.md}`、`pilot/integrations/cursor/rules/90-tool-invocation.mdc`、`pilot/skills/workflow/orchestration.md`、`pilot/schemas/tuning_config.schema.json`
- 试运行 session：`pilot/state/dsv2_lite_fp8_mi355x_20260511T075126Z/`（停在 BASELINE 入口）
