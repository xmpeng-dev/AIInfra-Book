# MoE Train Runtime v2 架构设计：Plan IR + 可替换 OverlapEngine

> **When**: 2026-08-19 12:29 UTC+8
> **Where**: 设计阶段，repo 未创建；首要目标平台 8× MI355X (gfx950) + SLURM 多节点
> **Context**: [v1 设计](./2026-08-19_1046_moe_train_runtime_v1_architecture_design.md) 的演进（非替换）。通读 `3rd/pith-train` engine 核心（`dualpipe/{dualpipev,overlap,execution}.py` ~1800 行）+ agent 层（`.agents/skills/`、`tools/memory_estimator/`）后，与 slab 已有 AMD 实测结论对账，发现 v1 的 `OverlapPolicy` 枚举粒度不足以承载「哪种 overlap 在 AMD 上有效」这个仍然开放的实验问题

## TL;DR

PithTrain 的性能架构建立在一个 NVIDIA 专属假设上：**a2a 放到第二条 comm stream 就能被 compute 藏掉**。slab 实测证明该假设在 AMD 上为假（RCCL 侧流 overlap ≈ 0%，最好比顺序执行慢 3%）。因此 5-stage 切分**存在的理由**在 AMD 上消失了 —— 但切分本身仍有价值，只是身份从「stream 调度技巧」变为「kernel 融合边界」。

v2 的核心提议是一句话：**把「有哪些 phase、按什么顺序跑」做成可打印 / 可 diff / 可断言的 `StepPlan` IR，把「怎么 overlap」做成消费该 IR 的可替换 `OverlapEngine`。**

相对 v1 的实质增量五处：

1. **Plan IR 取代 `OverlapPolicy` 枚举** —— 策略不是 enum，而是消费同一份 IR 的 executor；顺带解决 schedule 不可测 + simulator 漂移。
2. **融合粒度钉死在 stage-pair**，直接绑定 FlyDSL 已验证的 L1/L2 分解（v1 的 `FUSED_MOE` 仍偏 whole-layer，会踩 C2）。
3. **补 `SdmaPeerEngine`** —— 「不写 kernel 也能 overlap」的中间档，v1 缺这一级。
4. **明确接受 `csrc/` HIP 层**，并写清这是对 PithTrain "Python-native / no in-tree C++" 原则的**有意偏离**及其理由（C3）。
5. **`Capability` 探测 + `platform/` arch 事实层 + plan↔trace join** —— 针对「agent 在 AMD 上比在 NVIDIA 上更容易幻觉」的具体对策。

---

## 1. Background：PithTrain 在设计什么

它的所有取舍指向一个目标：**最小化 agent 的「阅读预算」**。这比「代码整洁」是个更精确的目标函数。

### 1.1 值得继承的 DNA

| DNA | 具体做法 | 为什么对 agent 有效 |
|---|---|---|
| locality > reuse | 无 plugin registry、无 string dispatch、一 model 一文件、接受重复 | 每层 indirection = 一次 grep + 一次 read；局部可读 = 更少 tool call 建立 ground truth |
| greppability 即 API | `PretrainLMCfg` 而非 `Cfg`；`launch` 是唯一通用动词 | `grep` 成为全函数，不漏 |
| Protocol 而非基类 | `LayerProtocol` 是结构契约，不是继承树 | 没有 `super()` 链要追 |
| reference 作为可执行 oracle | 每个 layer / operator 都带 `reference_forward` | **最重要一条**：把「我改对了吗」从判断题变成一条命令。agent 无法肉眼校验数值 |
| skill 带 PASS/FAIL 闸门 | `validate-correctness` 门禁 loss，性能只报告不门禁 | 把项目的**认识论**编码进仓库 |
| 自我解析模型 | `tools/memory_estimator` 符号化重放 8-step schedule | 不占 GPU 就能回答「pp=4 ep=8 会不会 OOM」；SLURM 排队环境里这是节奏问题 |

两条元级设计同样值得抄：CI 跑 4 路 Claude review（correctness / performance / **compactness** / consistency），compactness reviewer 专门抓「新增 indirection」和「为不可能发生的条件写防御」—— 仓库自带对抗架构漂移的免疫系统。

### 1.2 不能照抄的（多数是它自己违反自己原则处）

| # | 问题 | 位置 / 证据 |
|---|---|---|
| P1 | **scheduler 拥有了 MoE 数据流**。stage 2/4 声明为 framework-owned，a2a 住在 pipeline 引擎里 | `dualpipe/execution.py`；换 dispatch 实现必须改 pipeline |
| P2 | **手工展开的指令序列**，并用 asymmetric-None 作为「stage5+stage1 是否合并」的**带内信号** | `dualpipe/overlap.py` 331 行；`use_merged = outs is not None and args is None` |
| P3 | **`comm_stream` 是把实现当抽象** | `ExecutionCtx` 七字段：`comp_stream / comm_stream / fwd_event / bwd_event / fwd_comm_work / bwd_comm_work / fwd_comm_deferred_free` |
| P4 | **内存安全靠散文保证** | 手写 `untyped_storage().resize_(0)`，正确性写在注释：「only safe for MoE layers with EP where `padded_index_gather` is the first consumer」 |
| P5 | **schedule 有两份实现** | `dualpipev.py` 8 段循环 vs `tools/memory_estimator/schedule_simulator.py` 346 行重放 |
| P6 | **model 文件泄漏 EP 分支** | `qwen3_moe.py:forward_stage5` 的 `if distributed.ep_size == 1: ... else: scatter_add_` |
| P7 | **vendor 耦合无接缝** | `deep-gemm` / `flash-attn-4[cu13]` / `torch@cu130` 是顶层依赖；唯一替换点 `training.Linear = FP8Linear if fp8` 只覆盖 GEMM，attention 硬绑 FA4 在 model 里，MoE comm 硬绑 `direct_all_to_all` 在引擎里 |
| P8 | mesh 固定 `(PP,DP,CP,EP)` 但**不强制** EP intra-node | `modules/distributed.py:setup_device_mesh`；ep=16 在 8 卡节点静默跨 IB |
| P9 | 其他 | 无 TP / 无 activation checkpointing；同步 ckpt；只有 pp_rank 0 读数据；`num_chunks >= 2*pp_size` 把 global batch 与 PP 度耦死 |

---

## 2. 三条 AMD 硬约束（决定结构，不是调参）

| # | 约束 | 证据 | 架构后果 |
|---|---|---|---|
| **C1** | **RCCL 侧流不 overlap**（架构限制，非 bug） | 双流 0.90 ms vs 顺序 0.83 ms；最好 0.85 ms = **-3%**；`GPU_MAX_HW_QUEUES` 4/8/16 均 0% overlap（[`monolith-moe/2026-04-14_rccl_overlap_analysis.md`](../monolith-moe/2026-04-14_rccl_overlap_analysis.md)） | 删除 stream 作为 overlap 抽象；stream 引擎降级为 NVIDIA 专用 + parity 对照 |
| **C2** | **整层单核融合在训练规模翻盘** | super-kernel vs PyTorch+RCCL：512 t/g **1.46×**、2048 t/g **0.53×**、8192 t/g **0.29×**（[`monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md`](../monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md)）；stage 配对路线 fwd **2.6–3.1×**（[`notes/peer-tiles/`](../peer-tiles/)、[`notes/MegaMoeFlydsl/`](../MegaMoeFlydsl/)） | 融合粒度 = **stage-pair**，不是 whole-layer |
| **C3** | **AMD 缺关键库，引擎必须自己拥有 kernel** | hipBLASLt grouped GEMM 不支持 FP8（ROCm 7.1）；DeepEP / rocSHMEM 仅 gfx942；AITER 是 inference-only 无 backward（`knowledge/kernels/fp8-expert-gemm.md`、`knowledge/libraries/`） | **放弃 PithTrain 的 no-in-tree-C++ 原则** —— 本设计最诚实的一处偏离 |

### 2.1 C1 的精确推论

DualPipeV 在 AMD 上仍然有用，但**用途变了**。这一区分是整个 v2 的支点：

| 机制 | 作用 | AMD 上是否有效 |
|---|---|---|
| DualPipeV V 形 + 8-step | 减少 pipeline bubble（rank 空转） | **有效**，与 stream 无关 |
| 5-stage 切分 + `comm_stream` | 单 rank 内 comm/compute 重叠 | **无效**（C1） |
| 5-stage 切分作为**融合边界** | 让 comm 进 kernel 与 MFMA 重叠 | **有效**，且是唯一有效路径（C2） |

结论：切分保留，`comm_stream` 删除，overlap 机制变成待实验的变量。**「AMD 上哪种 overlap 机制有效」本身是未知的**，架构必须能承载这个实验 —— 这是 v1 用 enum、v2 用 IR + engine 的根本原因。

---

## 3. 核心一招：Plan IR

把 schedule 从「代码」变成「值」。

```python
class Phase(StrEnum):
    EMBED         = "embed"
    ATTN          = "attn"           # LN -> attn -> LN -> residual
    ROUTE         = "route"          # gate + splits + dedup
    SHARED_EXPERT = "shared_mlp"     # dense；DISPATCH 的天然重叠伙伴
    DISPATCH      = "dispatch"       # comm
    EXPERT_UP     = "expert_up"      # gate+up grouped GEMM
    EXPERT_ACT    = "expert_act"
    EXPERT_DOWN   = "expert_down"
    COMBINE       = "combine"        # comm（含 weighted scatter-add）
    RESIDUAL      = "residual"
    HEAD          = "head"
    LOSS          = "loss"

@dataclass(frozen=True, slots=True)
class Op:
    kind:   Literal["fwd", "bwd", "wgrad", "send", "recv", "free"]
    chunk:  int                      # micro-batch id
    branch: int                      # V 形分支 0/1
    layer:  int | None
    phases: tuple[Phase, ...]        # len > 1 == fusion group
    op_id:  str                      # 稳定字符串，可 join 到 trace

def build_plan(spec: ScheduleSpec) -> StepPlan: ...   # 纯函数
```

`build_plan` 是**纯函数**：输入 `(pp_rank, pp_size, num_chunks, num_layers, engine_caps)`，输出 `StepPlan`。

PithTrain 的两处隐式协议在 plan 里变成显式 op：

| PithTrain | v2 |
|---|---|
| asymmetric-None 标记 stage5+stage1 合并 | `phases=(RESIDUAL, ATTN)` |
| `WeightGradStore.enabled` 全局开关 + 末尾 `assert funcs_queue.empty()` 兜底 | `kind="wgrad"` 的被调度 op，配对关系静态可查 |
| 手写 `resize_(0)` + 注释保证 | `kind="free"`，由 plan liveness 分析生成 |

---

## 4. OverlapEngine：四个可互换实现

```python
class OverlapEngine(Protocol):
    caps: EngineCaps                                    # 声明支持哪些 fusion group
    def run(self, plan: StepPlan, ctx: ExecCtx) -> StepResult: ...
```

| Engine | comm 机制 | 适用 | 依据 | 里程碑 |
|---|---|---|---|---|
| `SerialEngine` | 默认流阻塞 a2a | 任意平台，正确性锚 | — | M0 |
| `StreamEngine` | 侧流 + event（= PithTrain） | 仅 NVIDIA / parity 对照 | C1 反证 | M2 |
| `SdmaPeerEngine` | SDMA queue peer copy，**不吃 CU** | AMD，无需写 kernel | Primus 已有 `sdma_symm_mem_collectives`（且重复实现两遍，见 `notes/primus-moe/`） | M3a |
| `FusedStageEngine` | in-kernel XGMI push 与 MFMA 同核重叠 | AMD 上限 | FlyDSL L1 **1.64×** / L2 **1.36×**；e2e fwd **2.6–3.1×**；`fc1_dgrad_combine` 省 1.71 ms、overlap 效率 ~87% | M3b |

`FusedStageEngine` 消费的 fusion group 正是 slab 已验证的两对，不是猜的分法：

| Fusion group | 对应 FlyDSL 段 | 实测 |
|---|---|---|
| `(DISPATCH, EXPERT_UP)` | L1 = dispatch + fc1 | 1.64× |
| `(EXPERT_DOWN, COMBINE)` | L2 = fc2 + combine | 1.36×，两腿近等长、overlap 效率 ~71% |
| bwd `(EXPERT_UP_DGRAD, COMBINE)` | `fc1_dgrad_combine` | 省 1.71 ms（serial 的 40%），效率 ~87% |

---

## 5. Backend 与 model 接缝

### 5.1 EP-invariant 引擎契约（关键取舍）

```python
class MoEEngine(Protocol):
    def forward(self, normed_h: Tensor, routing: Routing) -> Tensor:
        """返回 [T, H]：已按 topk_weight 加权求和，与 ep_size 无关。"""
```

`topk_weight` 作为输入交给引擎，缩放在 combine 内完成（融合 kernel 本来就想这么做）。于是 model 文件里：

```python
def residual_add(self, moe_out, residual):
    return residual + moe_out        # ep=1 与 ep=8 完全同一行
```

P6 的 `if ep_size == 1: ... else: ...` 分支彻底消失。**model 拥有数学，backend 拥有布局。**

代价：引擎拥有 topk 缩放，exotic aggregation 需要 hook（见 §10 Q1）。

model 作者只写 4 个方法 —— `attn_block` / `route` / `residual_add` / `reference_forward` —— **零通信代码、零 EP 分支**。

### 5.2 BackendBundle（setup 期绑定，无 runtime 查表）

```python
@dataclass(frozen=True, slots=True)
class Backend:
    linear:         LinearFactory          # dense GEMM
    grouped_linear: GroupedLinearFactory   # per-expert GEMM
    attention:      AttnFn                 # flash / varlen / ring
    moe:            MoEEngine
    collective:     Collective             # a2a / p2p
    caps:           Capability             # 探测结果，启动即打印
```

`Capability` 是 PithTrain 完全没有的东西：启动时探测一次并写进日志 —— `gfx950` / `hipblaslt_grouped_fp8=False` / `deepep=False` / `sdma_peer=True`。agent 读一份 log 就知道走了哪条路，不用猜。

### 5.3 `platform/`：把 ROCm 事实变成本地可引用常量

agent 关于 AMD 的训练数据远少于 CUDA，最常见幻觉是 warp=32、忘记 LDS 64 KB、编错 MFMA 指令名。单独一层存 arch 表（`gfx942` / `gfx950`、`WAVEFRONT = 64` 且 assert、LDS 容量、`v_mfma_*_f8f6f4` 指令族）与 `Topology`（XGMI clique、node group）。**让 agent 去读，而不是去回忆。**

---

## 6. 分层与目录

```
moe_train/
├── tasks/         pretrain_lm, convert_ckpt, tokenize        # launch(cfg)
├── plan/          phases.py  build.py  liveness.py  cost.py  golden/
├── exec/          executor.py  engines/{serial,stream,sdma_peer,fused_stage}.py
├── model/         protocol.py  qwen3_moe.py  deepseek_v2.py
├── backend/       bundle.py  capability.py  rocm/  cuda/
├── kernels/
│   ├── triton/    grouped_gemm, silu_mul, scatter, quantize
│   ├── hip/       csrc/ — fused dispatch×FC1, FC2×combine
│   └── reference/ 每个 kernel 的 torch 参考实现（强制）
├── platform/      arch.py  topology.py
├── data/  ckpt/  obs/  contexts/
```

比 PithTrain 只多四个边界：`plan/`、`backend/`、`kernels/hip/`、`platform/`。目标仍是 agent 一次读完（~15K 行 Python + ~3K 行 HIP）。

**并行度**：`ParallelSpec.ep_scope: Literal["intra_node","global"] = "intra_node"`，mesh builder 跨节点时**报错而非静默跨 IB**。AMD 上这条比 NVIDIA 更要紧 —— XGMI 525 GB/s vs NVLink 900 GB/s，跨 IB 再掉一个数量级。TP 维预留不实现。

**Compile 边界**：`attn_block` / `residual_add` 走 `@compile_region(fullgraph=True)`；整个 MoE group 是**单个 opaque custom_op**。不重演 PithTrain「S1/S5 编译、S3 裸奔」。ROCm 上 `torch.compile` 可靠性弱于 CUDA，因此 **no-compile 是一等运行模式**，不是 debug 后门。

---

## 7. 这个设计的优势

### 7.1 一句话

**它把「不确定的东西」变成变量，把「确定的东西」变成可断言的值。** AMD 上唯一真正未知的是「哪种 overlap 机制有效」；已确定的是 phase 分解和 DualPipeV 调度。PithTrain 把两者都写死在同一批文件里，所以任一改动都要动另一个。

### 7.2 优势逐条

| # | 优势 | 机制 | 为什么重要 |
|---|---|---|---|
| A1 | **schedule 改动的验证成本降一个数量级** | plan 是纯函数输出 → golden 文本快照 + 静态不变式（fwd/bwd 配对、send/recv 配对、无 use-after-free） | PithTrain 改 schedule 只能 8 卡 torchrun 跑 25 步比 loss；v2 先秒级 diff，再上 GPU |
| A2 | **消灭 schedule 漂移** | executor / memory estimator / cost model / trace 分析消费**同一份** plan | PithTrain 的 P5：两份实现必然漂移，且漂移时 estimator 静默给错答案 |
| A3 | **overlap 机制可 A/B 而不动 model 与 schedule** | 四 engine 同 plan、同 loss 曲线、同 parity 门禁 | 这是 AMD 研究引擎的**核心需求**：答案未知，架构必须承载实验。v1 的 enum 只能承载「已知的三种」 |
| A4 | **内存安全从散文变机械推导** | liveness 分析生成 `free` op | 消除 P4 —— 全仓最高风险、最不可 review 的代码 |
| A5 | **model 文件与并行度解耦** | EP-invariant 引擎契约（§5.1） | 加模型的成本不随 mesh 复杂度增长；消除 P6 |
| A6 | **无 GPU 可迭代** | plan + liveness（memory）+ roofline cost model（time） | SLURM 排队环境下这是**节奏问题**不是便利问题；PithTrain 已证明这条路值得走，v2 只是让它不会漂 |
| A7 | **profile 从启发式变成 join** | roctx/NVTX range 名 = `op_id` | PithTrain 的 nsys skill 要靠 kernel 名纯度 + NVTX 一致性投票去**猜** stage 归属；v2 直接接 slab `gpu-trace-analysis` |
| A8 | **AMD 现实被编码进结构而非注释** | C1 → 无 stream 抽象；C2 → fusion group 粒度；C3 → `kernels/hip/` + 强制 reference | 结构性约束不会被下一个改动者（人或 agent）无意违反 |
| A9 | **幻觉抑制** | `platform/` arch 事实 + `Capability` 启动探测并打印 | agent 在 AMD 上比在 NVIDIA 上更容易编造 API/规格；给它可读的事实源 |

### 7.3 量化：改一件事要付多少代价

| 任务 | PithTrain | v2 |
|---|---|---|
| 加一个 phase（拆 `EXPERT` 为 UP/ACT/DOWN） | 新增 `Args`/`Outs`/`Record` + `_f`/`_b` 共 5 处 → 改 `overlap.py` 手工展开循环 → 改 `execution.py` 的 `layer_forward`/`layer_backward` → 改 `schedule_simulator.py` | 加一个 `Phase` 成员 + `build_plan` 一行 + 更新 golden |
| 换 overlap 机制 | 改 `execution.py` 里 7 个 `ExecutionCtx` 字段的用法 + 改 `overlap.py` | 新增一个 `engines/*.py` 文件 |
| 验证 schedule 改动 | 8 卡 torchrun × 25 步比 loss（~20 min，且要抢到节点） | golden diff + 不变式（~1 s），再跑 loss |
| 回答「pp=4 ep=8 会 OOM 吗」 | 跑 estimator（独立实现，可能已漂移） | 同一 plan 上跑 liveness + cost |
| 定位 loss 散在哪一步 | 只有 loss 级 `compare.py` | plan diff → phase fixture → loss（三级二分） |
| 加一个模型 | 4 方法，含 `prepare_dispatch` + EP 分支 | 4 方法，零通信、零 EP 分支 |

### 7.4 代价（必须写清）

| 代价 | 缓解 |
|---|---|
| **多一层 IR**，字面违反 PithTrain 的 minimal-indirection 原则 | indirection 的真实成本是「多一跳才知道执行什么」；plan 是**数据不是代码**，`print(plan)` 把这一跳变成 O(1)。这是唯一可接受的加层理由，不可推广到其他模块 |
| plan 是纯函数 → 无法表达数据依赖的动态调度（如按实际 token 数动态改 chunk） | 第一版接受静态 plan；动态需求出现时再引入 `plan.specialize(runtime_shape)`，不提前设计 |
| HIP 层引入构建系统，CI 变重，agent 需能编译 | 硬规则：每个 HIP kernel 必须同时有 Python reference + Triton fallback，否则拒收。`SerialEngine` 全程不依赖 HIP |
| 四 engine = 四份维护成本 | 只有 `SerialEngine` 必须永远正确；其余按里程碑上线，共享 plan + parity 门禁；任一 engine 掉队就删 |
| 引擎拥有 topk 缩放（§5.1） | 留 `AggregateHook`，第一版不实现（Q1） |

### 7.5 不构成优势的（避免自我说服）

- **不比 PithTrain 更快**，直到 M3b 落地。M0/M1 明确不追性能。
- **不比 PithTrain 更小**：多 `plan/` + `platform/` + `csrc/`，行数是增加的。省下的是**改动成本**，不是行数。
- **不解决 P9**（无 TP、无 activation checkpointing）—— 那是独立工作项，v2 只预留 mesh 维。

---

## 8. Agent-native 层：超出 PithTrain 的四处

PithTrain 的 agent 层已很强，以下四条是仍缺的：

| # | 缺口 | 补法 |
|---|---|---|
| 1 | **oracle 是单点** —— `compare.py` 只告诉你 loss 散了 | 做成层级：plan（文本）→ phase 输出（张量 fixture）→ loss。配一个二分 phase 的 `parity` skill。agent 最贵的成本是定位「在哪一步」 |
| 2 | plan / trace / cost 三者无法对账 | 同一 `op_id` 串起「计划怎么排 / 实际怎么跑 / 理论该多快」；一条命令给出「哪个 op 离 roofline 最远」 |
| 3 | 库版本差异不可追溯 | 每次 run 落 `manifest.json`：`{config hash, git sha, topology, ROCm/RCCL/hipBLASLt 版本, capability probe}` + diff manifest 的 skill。AMD 上库版本是性能回归主因之一 |
| 4 | 指标只有人读日志，`compare.py` 正则解析 | JSONL + 人读日志双写；工具读 JSONL |

---

## 9. 里程碑

| 阶段 | 交付 | 验收门禁 |
|---|---|---|
| **M0** | `plan/` + `SerialEngine` + Rocm BF16 backend + Qwen3-30B-A3B | 单节点 ep=8 pp=1 loss 下降；plan golden 测试通过 |
| **M1** | DualPipeV plan（V 形 + 8-step + wgrad）+ liveness free | 与 PithTrain 同 mesh loss 曲线对齐（相对差 < 5e-3）；plan 不变式全绿 |
| **M2** | cost model + memory estimator（共享 plan）+ manifest | 预测 peak memory 与实测误差 < 10%；`StreamEngine` 在 NVIDIA 上跑通做交叉验证 |
| **M3a** | `SdmaPeerEngine` | 与 Serial 同 loss；rocprof 证明 peer copy 与 compute 有重叠且未抢 CU |
| **M3b** | `FusedStageEngine`（HIP，两对 fusion group） | T=8192 下 MoE 段 wall 优于 Serial；每核 reference 对拍通过 |
| **M4** | FP8 via `f8f6f4` + async ckpt + 多节点 pp>1 | FP8/BF16 parity；resharding 测试 `save(pp2,ep4) → load(pp1,ep8)` |

每阶段必须过 `reference_forward` + micro-batch loss parity。M0/M1 **不追性能**。

---

## 10. 明确不做 / 开放问题

**不做（第一版）**：Megatron 式 plugin registry / YAML spec；全模型族覆盖（先 Qwen3-MoE + DeepSeek-V2-Lite）；in-run elastic FT；TP（mesh 预留）；替代 Primus 作为 Megatron 后端。

| # | 开放问题 | 倾向 |
|---|---|---|
| Q1 | EP-invariant 契约让引擎拥有 topk 缩放，exotic aggregation 怎么办 | 留 `AggregateHook`，第一版不实现 |
| Q2 | HIP kernel 从零写还是 port MonolithEP / FlyDSL 已有核 | **port**，不重造；FlyDSL 的 L1/L2 分解直接对应两个 fusion group |
| Q3 | `SdmaPeerEngine` 是否已足够（不写 HIP 就拿到 overlap） | 先做 —— 若 SDMA 拿到 60%+ overlap，M3b 性价比需重估 |
| Q4 | Rocm GEMM 走 Primus-Turbo 薄壳还是 CK 直连 | Primus-Turbo 优先（符合 slab library 策略）；grouped FP8 需自己补 |
| Q5 | plan golden 快照粒度（op 全量 vs 摘要） | 全量 —— plan 应小到可以全量存 |
| Q6 | 工作名 `moe-train` vs 并入 `RocMoE-v3` | 独立 repo 保持 compact；kernel 从 rocmoe / MegaMoeFlydsl 复用 |

---

## 11. Next

1. Stub `plan/phases.py` + `plan/build.py` + `plan/golden/` 三件套（无 kernel、无 GPU，纯函数 + 快照测试）。
2. 从 PithTrain `overlap.py` 的调用图机械导出 pp=1/2/4 的 op 序列，作为 golden 的初始基线 —— 这同时是「行为 parity」的形式化。
3. 写 `plan/liveness.py`，对 PithTrain 的 `resize_(0)` 位置做交叉验证（应完全重合，若不重合则其中一方有 bug）。
4. M0：`SerialEngine` + AITER attention + Primus-Turbo grouped GEMM，单 node Qwen3-30B-A3B。

---

## 12. 参考

- PithTrain 源码：[`3rd/pith-train/`](../../3rd/pith-train/)，架构文档 [`docs/architecture.md`](../../3rd/pith-train/docs/architecture.md)
- v1 设计：[2026-08-19_1046_moe_train_runtime_v1_architecture_design.md](./2026-08-19_1046_moe_train_runtime_v1_architecture_design.md)
- RCCL overlap 实测：[`notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md`](../monolith-moe/2026-04-14_rccl_overlap_analysis.md)
- super-kernel 训练规模翻盘：[`notes/monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md`](../monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md)、[`notes/peer-tiles/`](../peer-tiles/)
- stage-pair 融合实测：[`notes/MegaMoeFlydsl/`](../MegaMoeFlydsl/)
- MoE comm 五问：[`knowledge/kernels/memory-access-patterns.md`](../../knowledge/kernels/memory-access-patterns.md)
- FP8 expert GEMM：[`knowledge/kernels/fp8-expert-gemm.md`](../../knowledge/kernels/fp8-expert-gemm.md)
- Primus 执行层战略：[`notes/primus-moe/2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md`](../primus-moe/2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md)
