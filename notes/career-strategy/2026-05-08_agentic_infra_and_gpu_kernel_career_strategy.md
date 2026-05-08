# Agent 时代职业路线：Primus Pilot 与 MMOE

**日期**: 2026-05-08

## 背景 / 目标

当前主要有两个技术方向：

- `Primus/pilot`：面向训练调优的 agentic training tuning system。
- `MMOE`：面向 AMD MI355X / MI300X 的 MonolithMoE persistent super-kernel。

目标是判断在 agent 快速发展的背景下，哪个方向更适合作为职业路线主线，哪个方向更适合作为长期技术壁垒与高价值补充。

## 主要结论

| 方向 | 定位 | 职业价值 | 风险 | 建议角色 |
|---|---|---|---|---|
| `Primus/pilot` | Agentic training infrastructure | 顺应 agent 发展趋势，贴合已有训练系统经验，能形成可复用系统闭环 | 需要真实多节点闭环和 agent eval 支撑价值 | 主线押注 |
| `MMOE` | AMD MoE GPU kernel / comm-compute overlap | 技术壁垒极高，展示 GPU kernel、MFMA、XGMI、IPC、persistent kernel 能力 | 成败更依赖最终性能结果，研究风险更高 | 硬核副线 / 杀手锏 |

一句话判断：

> `Primus/pilot` 应该作为 agent 时代职业主线推进，`MMOE` 应该作为底层硬实力和高价值性能 case study 保持投入。最理想的组合是：用 Pilot 自动化训练/性能调优闭环，用 MMOE 证明自己能处理真正困难的 GPU 系统问题。

## 为什么 `Primus/pilot` 值得作为主线

`Primus/pilot` 和个人背景高度匹配：large-scale training systems、distributed training、GPU performance、profiling、benchmarking、reproducibility、Megatron-style workflow。这不是普通 agent 应用开发，而是把已有训练系统经验转化成 agent 时代的新形态。

它天然适合 agentic workflow：

```text
preflight -> submit -> observe -> diagnose -> replan -> rerun -> report
```

这个流程需要读上下文、调用工具、观察训练结果、更新计划、沉淀状态，正是 agent 的优势场景。训练调优也有明确经济价值：错误配置、hang、OOM、通信瓶颈、低效并行策略都会浪费 GPU-hour。一个能自动诊断和缩短调优周期的系统，对训练平台团队有直接价值。

`Primus/pilot` 的长期壁垒不来自某个 agent 框架，而来自：

- 真实训练日志和实验数据。
- 分布式训练 failure mode。
- profiler trace / run log / metrics 的诊断规则。
- 并行策略、显存策略、通信策略的经验沉淀。
- 与 Primus / Megatron / SLURM / ROCm / RCCL 的集成深度。
- 可复现、可审计、可中断的 filesystem state。

可对外讲述的职业叙事：

> Built an agentic training tuning system that automates preflight checks, distributed job submission, runtime observation, profiler-driven diagnosis, and benchmark-guided configuration optimization for large-scale model training.

这比泛泛地会 LangChain、Cursor agent 或普通 RAG demo 更有职业区分度。

## `Primus/pilot` 的关键补强方向

| 优先级 | 方向 | 目标 |
|---|---|---|
| P0 | 真实集群闭环 | 让 slurm 模式稳定跑通 preflight / submit / observe / diagnosis / report |
| P1 | Profiler 驱动优化 | 用 trace / log / metrics 判断 compute、comm、memory、pipeline bubble、dataloader、checkpoint 瓶颈 |
| P2 | Experiment Memory | 把每轮实验沉淀为可查询知识：模型、硬件、配置、失败原因、有效/无效策略 |
| P3 | Agent Eval | 衡量 agent 是否能稳定诊断训练问题、减少 GPU-hour 浪费、提出有效 next step |
| P4 | Production Guardrails | 加预算上限、危险操作隔离、配置 diff 审查、失败回滚、参数白名单 |

## `MMOE` 的价值与风险

`MMOE` 是更硬核的底层性能项目，覆盖：

- AMD MI355X / MI300X。
- persistent super-kernel。
- device-side all-to-all。
- XGMI / HIP IPC。
- MFMA GEMM。
- MoE dispatch / expert GEMM / weighted combine fusion。
- comm-compute overlap。
- CDNA3 / CDNA4 架构差异。
- LDS / VGPR / AGPR / occupancy 调优。

这类能力非常稀缺，普通 agent 很难替代，因为它要求理解硬件、指令、内存层级、同步协议、benchmark 和正确性门禁。

但 `MMOE` 的风险也更高：它更像 research / kernel prototype，职业叙事强弱高度依赖最终性能结果。当前 README 中 DSV3 目标和现状显示出明显性能爬坡压力：

| 项 | 数字 |
|---|---|
| PyTorch 8-GPU baseline | 8.466 ms |
| 理想融合目标 | 4.797 ms |
| 当前 DSV3 super-kernel | 134.1 ms |

如果最终能证明 persistent super-kernel 在 AMD MoE all-to-all + expert GEMM 上有显著收益，它会非常亮眼；如果长期无法超过成熟 RCCL + CK / hipBLASLt 路线，职业叙事会弱一些。

因此更合适的定位是：`MMOE` 不是替代 `Primus/pilot` 的主线，而是作为高壁垒底层能力和 Pilot 的高价值 case study。

## 两个方向如何组合

最佳组合不是二选一，而是形成上下游关系：

```text
Primus/pilot
  -> 管理实验
  -> 收集日志 / trace / metrics
  -> 判断瓶颈
  -> 自动生成 sweep
  -> 归档结果和报告

MMOE
  -> 作为一个极难但高价值的性能优化对象
  -> 提供真实 kernel benchmark / correctness / profiler 数据
  -> 检验 Pilot 是否能辅助 GPU performance engineering
```

这样形成完整职业故事：

> 我不仅能做 agent infrastructure，还能让 agent 服务真正困难的 training systems 和 GPU performance 问题。

## 最终建议

短中期主线：推进 `Primus/pilot`，把它从设计/单机调优工具推进到真实多节点训练调优平台。

长期硬实力：持续保留 `MMOE`，重点争取一个足够有说服力的性能突破，哪怕只在特定 MoE shape 或特定 AMD 硬件上成立。

职业定位应从 Training Systems & AI Infrastructure Engineer 进化为：

> Agentic AI Infrastructure / Agentic Training Systems Engineer with deep GPU performance expertise.

## 相关文件

- `/shared/amdgpu/home/xiaoming_peng_qle/workspace/Primus/pilot/README.md`
- `/shared/amdgpu/home/xiaoming_peng_qle/workspace/Primus/pilot/agent/README.md`
- `/shared/amdgpu/home/xiaoming_peng_qle/workspace/Primus/pilot/tools/tune_single.py`
- `/shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/README.md`
