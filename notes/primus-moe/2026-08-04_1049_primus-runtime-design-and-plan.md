# Primus-Runtime：设计与实施规划

> **When**: 2026-08-04 10:49 UTC+8
> **Where**: `~/workspace/Primus`（dev 分支，HEAD `11400635`）— 静态代码分析，无集群运行
> **Context**: [框架战略备忘](./2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md)中选项 C 的落地方案。战略层面的取舍论证见该备忘；本文只谈「怎么做」

---

## TL;DR

**架构不需要设计——它已经存在。** `primus/core/pipeline_parallel/`（2,277 行）已是一个后端无关的 plan/execute 运行时内核：IR（`SchedulerNode`）→ 算法注册表（`produce_schedule_instance`）→ 执行器（`ScheduleRunner`，43 行）→ 后端 handler dict。它已有两个消费者：`primuspipe` 训练路径与 Projection 的性能 simulator。

**但它不在生产路径上。** `patch_primus_pipeline: true` 只出现在 2 个测试 yaml；生产 recipe（`examples/customer_package/run_qwen3_*_mi355x.sh` feature case 7，默认 `PP_STRATEGY="zbv"`）走的是 `--patch_zero_bubble True`，即 legacy 的 7,542 行 megatron-绑定栈。**所以「架构已验证」只在测试与 Projection 层面成立，不在生产规模成立**——这同时抬高了 P0 的价值（好架构还没吃到生产）和风险（迁移目标未经生产检验）。

**真正的工作是收敛，不是新建。** 当前有**三套并行的流水线调度栈**（core+primuspipe、legacy zerobubble、torch.distributed.pipelining+patch），两套 IR 定义，其中 `R` 这个 op 在两套 IR 中同名不同义。重构停在半路，两份并存本身就是持续的维护危害。

**规划分五期**：P0 收敛调度栈（把停下的重构做完）→ P1 通信层归位 → P2 精度层 → P3 内存与 HIP graph → P4 对外化。P0 不写新功能，只做归位与去重，以现有生产 recipe 吞吐零回退为硬门。

---

## 1. 现状盘点

### 1.1 已经成型的契约（不需重新设计）

| 契约 | 位置 | 行数 | 形态 |
|---|---|---|---|
| **IR** | `core/pipeline_parallel/scheduler/scheduler_node.py` | 69 | `FuncType`（11 个 op）+ `SchedulerNode(func_type, mini_batch, chunk, args, meta)` |
| **Plan** | 算法产出 | — | `list[list[SchedulerNode]]`，按 rank 索引，可序列化 |
| **算法接口** | `scheduler/algorithms/base.py` | 302 | `PipelineScheduleAlgo.generate_schedule_table()`；子类 `VFoldScheduleAlgo` 处理 V 形调度 |
| **算法注册表** | `scheduler/schedule_table_factory.py` | 56 | `produce_schedule_instance(name, pp, vpp, mbs)`，6 个算法类 / 7 个注册名 |
| **执行器** | `scheduler/scheduler.py` | **43** | `ScheduleRunner(handler_dict).run(table, rank)` |
| **后端接缝** | `backends/megatron/.../primuspipe/handlers/__init__.py` | 36 | `{FuncType: callable}`，11 个 op 中 4 个复用 core 默认实现 |
| **默认 handler** | `core/pipeline_parallel/handler/` | 339 | `offload_handler`（261）、`wgrad_handler`（78） |

执行器全文只有 43 行，核心是一个按 op 类型分派的循环：

```24:35:primus/core/pipeline_parallel/scheduler/scheduler.py
    def run(self, scheduler_table, rank: int):
        for idx, node in enumerate(scheduler_table[rank]):
            if node.args is not None and "combined_group" in node.args:
                func = self.handle_func_dict[FuncType.FB]
                func(node, idx, scheduler_table[rank])
            else:
                if self.pre_process_func is not None:
                    self.pre_process_func(node, idx, scheduler_table[rank])
                func = self.handle_func_dict[node.func_type]
                func(node, idx, scheduler_table[rank])
```

接入点也已存在——Megatron 的 `forward_backward_func` 被整体替换：

```120:121:primus/backends/megatron/core/pipeline_parallel/schedules.py
def get_primus_pipeline_parallel_fwd_backward_func():
    return PrimusPipelineParallelLauncher().run
```

### 1.2 三套并行的调度栈（主要债务）

| 栈 | 位置 | 行数 | IR | 谁在用 | 状态 |
|---|---|---|---|---|---|
| **A. core + primuspipe** | `core/pipeline_parallel/` + `backends/megatron/.../primuspipe/` | 2,277 + 1,085 | `SchedulerNode`（11 op） | **仅 2 个测试 yaml** + Projection simulator | **目标架构**，算法覆盖不全，**未上生产** |
| **B. legacy zerobubble** | `backends/megatron/.../zerobubble/` | **7,542** | `ScheduledNode`（18 op） | **生产**：customer_package 的 Qwen3-235B / 30B 脚本（`PP_STRATEGY="zbv"`） | 功能最全（自动求解器、通信规划、细粒度 offload），但完全绑定 megatron |
| **C. torch pipelining** | `backends/torchtitan/patches/pipelining_schedule_patches.py` | 79 | torch 自有 | torchtitan | 只 patch 进 DualPipe-V，与 A/B 无关 |
| **（默认）原生 Megatron** | 上游 | — | — | 大部分 recipe，含 DSv3 671B（PP16/VPP2） | 两个开关都不开时的缺省 |

**迁移方向已定但停在测试层**：`tests/trainer/test_megatron_trainer_zero_bubble.yaml` 里 `patch_zero_bubble: true` 被注释掉、换成了 `patch_primus_pipeline: true`。重复也已泄漏到用户配置面——`v-half` / `v-min` 在 A 是 `pp_algorithm` 取值，在 B 是 `zero_bubble_v_schedule_mem_setup: half|min`。

栈 B 的独有能力是 A 目前缺的：`scheduler/v_auto_schedule.py`（928 行自动调度求解器）、`scheduler/communication.py`（700 行通信规划）、`scheduler/{graph,passes}.py`（414 行调度 IR + pass 框架）、`offload.py` + `scheduler/offloading.py`（1,007 行）。

算法层面 A 与 B 大量重复：A 有 `basic_1f1b` / `interleaved_1f1b` / `zerobubble` / `zerobubble_heuristic` / `zbv_formatted` / `zbv_greedy`；B 有 `basic1f1b` / `seq1f1b` / `v1f1b` / `vpp` / `zb` / `zbv` / `zbv_greedy` / `group_interleaved_1f1b`。

### 1.3 架构的验证程度：够用，但不是生产验证

`primus/core/projection/`（13,394 行）直接复用同一套调度内核：`performance_projection/simulator.py` 导入 `scheduler_node`，`projection.py` 在约 20 处以全限定类名引用 core 的算法类（`...algorithms.zerobubble.ScheduleZeroBubble` 等）作为已验证案例的配置。

**同一个 plan 既驱动真实执行又驱动性能预测**——这是 plan/execute 分离在设计上正确的最强证据，也是 Projection 在 MoE 案例上达到 1.4% 误差的原因。任何重构都必须保住这个双消费者性质。

但要如实区分两件事：

| 主张 | 是否成立 |
|---|---|
| 这套抽象能表达真实的调度语义 | **成立** —— Projection 用它做出了 1.4% 误差的预测 |
| 这套抽象能跑通训练 | **成立** —— 2 个测试 yaml 走的就是这条路 |
| 这套抽象在生产规模、生产 recipe 上验证过 | **不成立** —— 生产走的是栈 B |

对 P0 的含义是双向的：**价值更高**（好架构还没吃到生产收益，收敛后立刻兑现），**风险也更高**（迁移目标本身未经生产检验，不能假设「照搬就行」）。因此 P0 必须双跑对拍，而不是直接切换。

---

## 2. 目标架构

```
primus/runtime/
├── ir/            op 类型、schedule graph、passes      ← 合并 scheduler_node.py 与 zerobubble/scheduler/{graph,passes}.py
├── planner/       调度算法、通信规划、内存规划          ← scheduler/algorithms/ + zerobubble/scheduler/communication.py
├── executor/      按 plan 驱动 handler                  ← ScheduleRunner + pipeline_launcher + zerobubble/runtime.py
├── handlers/      后端无关的默认 handler                ← core/pipeline_parallel/handler/*（已存在）
├── comm/          RCCL / SDMA / symmetric-mem / DeepEP  ← R2
├── precision/     FP8/MXFP8 scale 状态机与 cache 失效   ← R5
├── memory/        offload / recompute / HIP graph 捕获  ← R6
├── platform/      gfx942 / gfx950 能力与拓扑描述        ← 新增
└── adapters/{megatron,torchtitan,maxtext}/   仅 handler dict + spec provider
```

四条组织规则（前两条已被现有代码验证，后两条待建立）：

1. **plan / execute 分离** — planner 产出可序列化 plan，executor 只按 plan 驱动 handler。
2. **handler dict 是唯一后端接缝** — 后端只提供 `{op_type: callable}`，不参与调度决策。
3. **依赖方向单向**：adapters → runtime → platform / Turbo。当前 `backends/transformer_engine/` 有 4 个文件反向 `import megatron`，需 CI 规则（import-linter）强制。
4. **能力发现而非硬编码** — platform 层声明能力，planner 查询后决策，不在算法里写 `if gfx950`。

---

## 3. 终局形态：与 Megatron 如何配合

> 当前接入链路的逐层代码走读（PP=4 / 1f1b 完整例子）见 `slab/knowledge/systems/primus-pipeline-runtime-megatron-integration.md`。本节只讲终局形态与接缝。

**分工一句话：Megatron 回答「模型是什么」，Runtime 回答「这个 step 怎么跑」。**

关键前提是**模型的前向反向计算仍然是 Megatron 的代码**——Runtime 只决定顺序、通信与内存，不碰模型实现。这是模型覆盖能继续白拿上游的原因：DeepSeek-V4 的 mHC / DSA 等上游一落地，Primus 立即可用，因为那发生在 handler 内部，与调度无关。

### 3.1 接缝：从 26 处语义重写收敛到 3 个绑定点

今天 `megatron.pp.schedule` patch 的做法是直接改写两处模块绑定：

```39:47:primus/backends/megatron/patches/parallelism/schedule_patches.py
    ori_pp.get_forward_backward_func = get_primus_pipeline_parallel_fwd_backward_func
    ...
    megatron_training.get_forward_backward_func = get_primus_pipeline_parallel_fwd_backward_func
```

终局保持同样的形态，但整个执行层只剩三个绑定点：

| 绑定点 | 交出什么 | 上游接受扩展点后的形态 | 对应模块 |
|---|---|---|---|
| `get_forward_backward_func` | 一个 step 的**执行权** | 调度器注册表，按名取实现 | R1 / R3 |
| spec / provider | 层内算子实现 | 已有机制（`TransformerLayerSpec`），补完整即可 | R4 / R5 |
| collective backend | 通信后端（SDMA / symm-mem / DeepEP） | process-group 级插件 | R2 / R6 |

**终局不是零 patch。** 即使上游全盘接受扩展点，启动时仍需把 Primus 的实现注册进去。差别在于那是三处注册，而非 26 处语义重写，且注册点是上游承诺的 API——lock bump 不会把它冲掉。

### 3.2 一个 step 的时序

启动期（一次）：Primus 读 config 决定调度算法 / comm backend / precision recipe → Megatron 正常构建模型 → 三个绑定点接上 → planner 产出 plan（`list[list[SchedulerNode]]`，可序列化）。

每个 step：

1. Megatron 的 `train_step` 调 `get_forward_backward_func()`，拿到 Runtime 的 executor
2. executor 遍历 plan 中的 `SchedulerNode`，按 `func_type` 分派到 handler dict
3. `F` / `B` 节点调回 **Megatron 的** `forward_step` / `backward_step`——模型计算在此发生
4. `W` / `O` / `R` / 通信节点走 Runtime 的默认实现
5. Megatron 拿回 loss，继续 optimizer step

控制流是 **Megatron → Runtime → Megatron 的三明治**，不是单向调用。这也是「执行层不等于组件库」的具体含义。

### 3.3 用户视角

**现有 Megatron recipe 不需要修改**——drop-in 兼容性是整件事的前提，不能因重构丢失。

用户可感知的唯一变化是配置收敛：今天 `primus/configs/modules/megatron/` 下 `primus_pipeline.yaml` 与 `zero_bubble.yaml` 概念重叠（`debug_scheduler_table` 两边都有，offload 各有一套参数、命名不同），这正是三套调度栈在用户面前的投影。P0 完成后合并为一套。

---

## 4. 关键技术任务的细节

### 4.1 IR 合一：11 op vs 18 op，且有一处同名不同义

| 差异 | core `FuncType`（11） | legacy `FuncType`（18） | 处理 |
|---|---|---|---|
| 组合前反向 | `FB` | 无 | 保留 core 的 `FB` |
| zero-bubble 后验证通信 | 无 | `POST_VALIDATION` / `SEND_POST_VALIDATION` / `RECV_POST_VALIDATION` | core 需补 3 个 op |
| offload 粒度 | `O` / `R`（2 个，粗） | `OFFLOAD_BARRIER` / `SEND_START` / `SEND_END` / `RECV_PREP` / `RECV_START` / `RECV_END`（6 个，细） | 采用细粒度，否则 offload 与 comm 无法在同一 plan 内排序 |
| **命名冲突** | `R = "RELOAD"`（offload 回载） | `R = "R"`，且 `is_computation()` 判定为真，即 **recompute** | **同名不同义，合并时必须显式改名**（建议 `RELOAD` / `RECOMPUTE` 全拼） |

这个冲突是 P0 最容易埋雷的地方：两边都叫 `FuncType.R`，静态检查不会报错，但语义相反。

### 4.2 需要新增的契约

| 契约 | 目的 | 现状 |
|---|---|---|
| **platform capability** | planner 查询「有无 SDMA 引擎 / symmetric memory / XGMI 全连接 / HIP graph 支持」后决策 | 不存在，能力判断散落在各 patch 的 `condition` 里 |
| **comm backend 接口** | RCCL / SDMA / symm-mem / DeepEP 可替换，两个后端共用一份实现 | 不存在，`sdma_symm_mem_collectives` 在两后端各写一份（142 + 163 行，零共享） |
| **precision state** | FP8/MXFP8 scale cache 的失效时机作为 plan 的一部分，而非隐式全局状态 | 隐式，散在 `primus_turbo_float8_local.py`（1,703 行）与多个 patch |
| **plan 序列化格式** | HIP graph 捕获、离线调度分析、Projection 输入共用 | 事实上已是 `list[list[SchedulerNode]]`，但无稳定序列化契约 |

### 4.3 HIP graph 捕获对 IR 的要求

full-iteration 捕获要求 plan 中**不含数据依赖的控制流**——所有分支（哪个 micro-batch offload、哪层 recompute、scale 是否刷新）必须在 planner 阶段静态决定并落进 plan。这反过来给 IR 提了两个要求：细粒度 offload op（见 3.1）与显式的 precision op。**这是把 HIP graph 排在 P3 而非更早的原因**：它依赖 P0 的 IR 合一与 P2 的精度显式化。

---

## 5. 工作分解

### P0 — 收敛调度栈（前置：无）

把停在半路的重构做完。**不写新功能。**

- 合并两套 IR，解决 `R` 的命名冲突，core `FuncType` 扩到覆盖 legacy 全部语义
- 把 legacy 独有能力迁入 core：自动调度求解器（928）、通信规划（700）、pass 框架（414）、细粒度 offload（1,007）
- 去重算法：`zbv_greedy` / `basic1f1b` / `zb`↔`zerobubble` / `interleaved_1f1b`↔`group_interleaved_1f1b`
- 迁移后删除 `backends/megatron/.../zerobubble/`，`primuspipe` 收为 adapters/megatron 的 handler dict

**验收**：调度栈从 3 套降到 2 套（A + torchtitan 的 torch 路径）；`zerobubble/` 目录清空；Projection simulator 与真实执行仍共用同一 IR；现有 recipe 吞吐零回退（基线见第 6 节）。

**风险**：legacy 栈功能最全且在生产路径上，迁移期需双跑对拍。建议按算法逐个迁移并保留开关，而非一次性切换。

### P1 — 通信层归位（前置：P0 的 IR）

- 定义 comm backend 接口，把两份 `sdma_symm_mem_collectives` 合一
- `sdma_param_gather`（248）、`fsdp2_fp8_all_gather`（654）纳入 `runtime/comm/`
- 建立 platform capability 契约，把「有无 SDMA / symm-mem」从 patch condition 提到 platform 层
- torchtitan 接入同一 comm backend

**验收**：TorchTitan 复用 runtime 的 comm 实现而非自带；SDMA 相关 patch 从 patch 层消失；两后端共用一份实现。**这是「跨后端复用」第一次被真正证明。**

### P2 — 精度层（前置：P0）

- FP8/MXFP8 scale 状态机从隐式全局改为 plan 内显式 op
- `primus_turbo_float8_local.py`（1,703）与相关 patch 归入 `runtime/precision/`
- 明确与 Turbo 的边界：量化 kernel 属 Turbo，何时量化 / cache 何时失效属 runtime

**验收**：FP8 recipe 的 scale 刷新时机可从 plan 读出；相关 patch 退出 patch 层；FP8 收敛与吞吐零回退。

### P3 — 内存与 HIP graph（前置：P0、P2）

- offload / recompute 决策统一到 planner
- **full-iteration HIP graph 捕获**——第一个「patch 形态做不出来」的能力

**验收**：端到端 step 完整捕获成功；host launch 开销降低可测量；MoE 小模型区间吞吐提升。这是整个方案的论证支点。

### P4 — 对外化（前置：P1 至少完成）

- 独立可交付物（独立仓或独立 package），三后端为消费者
- 向上游提扩展点 PR（调度器注册、collective 后端替换）
- super-kernel 经 R4 产品化

**验收**：三后端至少两个在生产 recipe 中调用 runtime；上游 PR 拿到明确 accept/reject 信号。

### 依赖关系

```
P0 ──┬── P1 ──┐
     ├── P2 ──┼── P3
     └────────┘
              └── P4（需 P1）
```

---

## 6. 回归基线与验收门

每一期结束都必须过同一组门，任何一项回退即视为未完成：

| 基线 | 来源 | 门 |
|---|---|---|
| Qwen3-235B-A22B，32 GPU，FP8-CS，TP1/PP1/CP4/EP8 | MoE package 2.0 blog | ≥ 4,809.1 tokens/s |
| Qwen3-30B-A3B，8 GPU，FP8-CS，EP8 | 同上 | ≥ 27,581.3 tokens/s |
| GPT-OSS 20B，8 GPU，FP8-CS，DP8 | 同上 | ≥ 28,136.7 tokens/s |
| DeepSeek-V3 671B，TP1/PP16/VPP2/EP8 | 同上 | loss parity + 吞吐不回退 |
| Projection 精度 | MoE 案例（Mixtral 8x22B，EP8/PP4/VPP2） | 误差仍 ≤ 1.4% |

最后一项容易被忽略但很关键：它是「plan/execute 分离没有在重构中被破坏」的直接检验。

---

## 7. 未决问题（需要人决策）

1. **P0 的迁移方向**：把 legacy 的能力搬进 core，还是把 core 的接口套到 legacy 上？前者代码量大但终局干净，后者快但可能把 megatron 耦合带进 core。倾向前者，需与 pipeline owner 确认。
2. **torchtitan 的调度栈是否收编**：它现在走 `torch.distributed.pipelining`。收编能统一，但要与 PyTorch 上游的演进赛跑；不收编则 R1 的「跨后端」只在 comm 层成立。
3. **独立仓还是 package**：P4 时 runtime 是拆成独立 repo（如 Primus-Turbo 那样）还是留在 Primus 内作为独立 package。影响 CI、发版与外部可见性。
4. **`primuspipe` 与 `zerobubble` 的 owner 是否同一人**：两套并存可能反映的是组织边界而非技术选择。若是，P0 的阻力主要在协调而非代码。
5. **绝对工时与人力**：本文只给依赖顺序与验收标准，不含工时估算——需与团队一起估。

---

## 8. 与 H2 2026 roadmap 的映射

| roadmap 条目 | 对应期 |
|---|---|
| DP/PP/TP/EP comm overlap 统一调度器 | P0（IR/planner）+ P1（comm） |
| Full-iteration HIP graph capture | **P3** |
| 1F1B EP A2A overlap | P0 的 planner + P1 |
| MoE-Native IR（研究线 B） | P0 的 pass 框架之上，非从零 |
| Backward comm scheduling（研究线 D） | P0（wgrad op 已在 IR 中）+ P1 |
| HybridEP-AMD、DeepEP v2 | P1 的 comm backend |
| MXFP8 GroupedGEMM | P2（runtime 侧）+ Turbo（kernel 侧） |
| MoE super-kernel 产品化 | P4 |

**roadmap 上大部分系统类条目都落在 P0–P1 之后**——这说明收敛调度栈不是「重构的成本」，而是这些条目的共同前置。

---

## Next

- [ ] 与 pipeline owner 确认第 7 节问题 1 与 4（迁移方向、组织边界）
- [ ] 逐算法对比 A / B 两套调度栈，产出精确的去重清单与迁移顺序
- [ ] 起草合并后的 `FuncType`，明确 `RELOAD` / `RECOMPUTE` 改名方案
- [ ] 搭 P0 的对拍框架：同 config 下 A / B 两栈的 schedule table 差异比对
- [ ] 加 import-linter 规则，冻结当前的反向依赖不再增加
