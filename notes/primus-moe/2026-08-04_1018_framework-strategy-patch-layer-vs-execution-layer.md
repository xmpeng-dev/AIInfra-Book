# Primus 框架战略：加固 patch 层、自研框架、还是自研执行层

> **When**: 2026-08-04 10:18 UTC+8
> **Where**: `~/workspace/Primus`（dev 分支，HEAD `11400635`）— 静态代码分析，无集群运行
> **Context**: 面向管理层的技术备忘，回答「AMD 是否需要一个自己的训练框架」。数据源为 Primus 仓库当前状态；roadmap 上下文见 [`2026-05-29_roadmap_h2_2026.md`](./2026-05-29_roadmap_h2_2026.md)

---

## TL;DR

**「要不要自研」是个伪问题——Primus 里已经有约 3.3 万行执行层代码在跑生产，只是没被当成执行层来管理。** `primus/backends/megatron/core/pipeline_parallel` 一个目录 8,975 行，就已经超过 megatron 全部 patch 的 7,975 行；`primus/backends/transformer_engine/` 另有 4,045 行 TE 局部重实现。patch 只占 megatron 后端代码的 **17%**（7,975 / 48,054）。

三条结论：

1. **patch 数量不是核心问题。** 97 个注册 patch 中 42% 是配置与 UX 改写、27% 是 Turbo kernel 注入、27% 是语义重写，真正的 ROCm 缺陷绕过只有 3 个（3%）。配置类那 42% 确实养不起人，但它们也不是负债，迁出配置层即可。
2. **反对自研一个 Megatron 替代品（选项 B）。** drop-in 兼容性是 Primus 的核心销售卖点，模型覆盖是靠上游白拿的；自研会把迁移成本还给客户，并把团队的工作变成无差异化的追赶——编制增长了，但增长的是跑步机岗位。
3. **建议把已有的 3.3 万行 + 26 个语义重写 patch 归位成一个独立执行层（选项 C，工作名 Primus-Runtime）。** 借 NVIDIA Transformer Engine 的**组织位置**（被多个上游框架调用的横向层，上游是消费者而非对手），但不是借它的技术范围——范围上 Primus-Turbo 才是 TE 的对位物，Runtime 在其之上管「一个 step 怎么跑」。立项论证因此不是「请批准造一个框架」，而是「请批准把已有的执行层从三个后端命名空间里合并出来」——后者的说服难度完全不同。

最直接的证据：`sdma_symm_mem_collectives` 这一个 AMD 硬件能力，在 megatron 和 torchtitan 下各有一份独立实现（142 + 163 行，**共享代码为零**）。同一件事已经在写第二遍了。

---

## 背景

Primus 不 fork 上游后端，而是用 monkey-patch 叠加：每个 patch 用 `@register_patch` 注册，由 `run_patches` 在训练生命周期的某个 phase 应用（引擎在 `primus/core/patches/`，6 文件 / 812 行）。这个设计比 fork 一份 Megatron 然后永久 rebase 要好——有 registry、有 phase、有 condition 门控、有 `PRIMUS_PATCHES` 环境变量级开关。

但两个现象促使我们重新评估这层的定位：

- roadmap H2 2026 上的若干条目（MoE-Native IR、DP/PP/TP/EP 统一 comm overlap 调度器、full-iteration HIP graph capture）**在 monkey-patch 形态下无法干净实现**——捕获整个 step 需要控制执行流，而执行流目前在 Megatron 手里。
- 团队规模的扩张需要与之匹配的技术纵深。「维护 N 个 patch 让 Megatron 在 MI355X 上跑起来」这个叙事的编制天花板很低。

---

## 数据来源与方法

全部数字来自对 `~/workspace/Primus` HEAD `11400635` 的静态分析，可复现：

```bash
# patch 规模（文件数 / 行数）
find primus/backends/megatron/patches -name '*.py' | wc -l
find primus/backends/megatron/patches -name '*.py' -exec cat {} + | wc -l

# 注册 patch 精确计数：AST 解析 @register_patch 装饰器
# （直接 grep '@register_patch' 会得到 111，其中 4 处是 primus/core 里的
#   docstring 示例，其余差额为多行装饰器的重复计数；以 AST 结果为准）

# 维护开销
git log --since="6 months ago" --oneline -- primus/backends/*/patches | wc -l
git log --since="6 months ago" --oneline | wc -l

# 版本门控覆盖率
rg -c 'backend_versions' --glob 'primus/backends/**/*.py'

# 各后端 patch vs 非 patch 行数（3.3 万行执行层资产的来源）
#   遍历 primus/backends/<b>/**/*.py，按路径是否含 /patches/ 分两类计数
```

分类是按 patch 的**归属**（这段逻辑长期应该属于谁）而非功能做的，规则基于文件路径与 patch id，脚本可重跑。

---

## 数据

### 规模

| 维度 | 值 |
|---|---|
| 注册 patch 总数 | **97**（megatron 78 / torchtitan 16 / maxtext 2 / megatron_bridge 1） |
| patch 代码量 | 约 **10,100 行**（megatron 76 文件 7,975 行；torchtitan 20 文件 1,762 行；maxtext 157 行；bridge 234 行） |
| patch 引擎 | 6 文件 / 812 行（`primus/core/patches/`，本身不含 patch） |
| Primus Python 总量 | 100,482 行（patch 层占约 10%） |
| 近半年 commit 触及 patch 目录 | **49 / 195（25%）** |

### 关键发现：patch 只占后端代码的 17%

讨论「要不要自研」之前必须先看清一件事——**Primus 里已经有约 3.3 万行执行层代码，只是没被当成执行层来管理。**

| 位置 | 非 patch 行数 | 是什么 |
|---|---|---|
| `primus/backends/megatron/core/pipeline_parallel` | **8,975** | 一套完整的流水线调度器（比 megatron 全部 patch 的 7,975 行还大） |
| `primus/backends/megatron/core/extensions` | 7,254 | 后端扩展实现 |
| `primus/backends/megatron/core/transformer` | 4,572 | transformer 层实现 |
| `primus/backends/transformer_engine/` | **4,045** | **一份局部的 TE 重实现**（permutation 1,761、comm_overlap 542、gemm 751、module/base 304） |
| `primus/backends/megatron/core/models` | 3,357 | 模型实现 |
| `primus/backends/megatron/core/optimizer` | 2,343 | 优化器实现 |
| `primus/backends/megatron/core/distributed` | 1,418 | 分布式实现 |
| 小计（上述 + `fp8_utils` / `fp4_utils` / `tensor_parallel` / `parallel_state`） | **约 33,000** | |

对比：`primus/backends/megatron/` 非 patch 代码 **40,079 行** vs patch 代码 7,975 行——**patch 只占这个后端的 17%**。

三条推论：

1. **「自研有没有能力/预算」是个伪问题。** 代码已经写完并已在付维护成本，只是按「它影子化了谁的命名空间」归档（`backends/megatron/core/...` 镜像上游路径），而不是按「它提供什么能力」归档。
2. **TorchTitan 一行都用不上。** 它的非 patch 代码只有 1,201 行。最直接的证据是 `sdma_symm_mem_collectives` 在两个后端各有一份独立实现（megatron 142 行 + torchtitan 163 行，**共享代码为零**，两边只共用 patch registry 和 logging）。同一个硬件能力写了两遍。
3. **它作为产品是不可见的。** 3.3 万行核心执行逻辑归档在 `backends/` 下，对外读起来是「适配胶水」，讲不出产品故事，也支撑不起编制。

### 构成（按归属分类）

| 类别 | patch 数 | 占比 | 能支撑团队纵深？ | 长期归属 |
|---|---|---|---|---|
| **配置与 UX 改写**<br>args 路径 / wandb / tensorboard / mock data / tokenizer / profiler / checkpoint 路径 | 41 | 42% | 否 | Primus 配置层，或推上游 |
| **Turbo kernel 注入**<br>`turbo/` 与 `te_patches/` 系列 | 26 | 27% | 部分（纵深在 Turbo 侧，不在 patch 侧） | 走 Mcore spec / provider 钩子 |
| **语义重写**<br>schedule / v_schedule / zero_bubble / train_step / forward_step / pp layout / MLA / moe_dispatcher / precision-aware optimizer / FSDP | **26** | **27%** | **是** | **抽成独立执行层** |
| **ROCm 缺陷绕过** | 3 | 3% | 否 | 推上游，数量应随时间递减 |
| 未分类 | 1 | 1% | — | — |

### 工程卫生

| 指标 | 值 | 判读 |
|---|---|---|
| 带 `condition=` 门控 | 68 / 97 | 良好 |
| 带 `backend_versions=` 版本门 | **1 / 97** | 见下（naive 补齐是**危险的**） |
| 用 `inspect.getsource` 源码重写 | 2 文件 | 克制（这是最脆弱的 patch 风格） |
| patch 声明所替换的上游符号 | **0 / 97** | `FunctionPatch` 无 `targets` 字段——**bump 时无法自动化影响面分析** |

三点需要在建议里体现的引擎细节：

- **上游按 commit 钉死**（`primus/_thirdparty.lock`，Megatron-LM `d3528a21`），所以风险不是「持续漂移」，而是**bump 那一刻没有任何机器可读的信号**告诉你哪些 patch 需要复核。
- **补 `backend_versions` 在当前引擎里是陷阱。** `_get_backend_version()` 吞掉所有异常返回 `None`（`primus/core/runtime/train_runtime.py:192-195`），而 `_match_backend_version()` 在「有 pattern 且版本为 None」时返回 `False`（`primus/core/patches/patch.py:68-74`）——即**静默跳过**。megatron 的版本探测靠在 `sys.path` 里找 `megatron/core/package_info.py`（`megatron_adapter.py:142-147`），上游一旦挪动该文件，探测失败 → 所有带版本门的 patch 集体消失，训练照常启动，只留一行 `⊘ Skipped` 日志。今天只影响 1 个 patch；若无脑铺到 97 个，一次探测失败就能静默停用整个 patch 层。
- **唯一的正面先例**：`turbo/te_spec_provider_patches.py` 带 `backend_versions=["<0.17"]`，含义是上游 Megatron ≥ 0.17 后此 patch 自动退休。这正是「推上游 → patch 到期自删」闭环可行的证据，也是每个 patch 都该有的形态。

---

## 论证

### 三条路的取舍

| 选项 | 技术差异化 | 团队纵深 | 客户迁移成本 | 上游依赖 | 主要风险 |
|---|---|---|---|---|---|
| **A. 维持现状，加固 patch 层** | 低 | 低 | 零（卖点保留） | 完全被动 | 编制天花板低；lock bump 时无影响面分析；重复实现随后端数量线性增长 |
| **B. 自研训练框架**（AMD 的 Megatron） | 中 | 高但脆弱 | 高（卖点丧失） | 无 | 模型覆盖从零开始；工作变成无差异化追赶 |
| **C. 归位为独立执行层**（Primus-Runtime） | **高** | **高且可辩护** | 零 | 仅 API 面 | 需要上游接受扩展点；需要跨后端抽象设计能力 |

选项 C 的完整展开见下方[「选项 C 展开」](#选项-c-展开执行层的边界模块与交付)一节。

### 为什么反对 B

**模型覆盖是上游的礼物。** roadmap 上 DeepSeek-V4 的 mHC、DSA、sqrtsoftplus 全部标注为 "port from `Megatron-LM#XXXX`"。自研意味着这些要自己实现、自己验收敛、自己维护 HF 互转。以 TAS 的人力对标 Megatron-Core 的投入，这个账算不过来。

**drop-in 兼容性就是产品价值本身。** 客户手上是 Megatron 配置，AMD 的卖点是「你的 recipe 在 MI355X 上直接跑」。自研框架把迁移摩擦还给客户，而消除这个摩擦恰恰是 Primus 存在的理由。

**B 的编制故事比看上去脆弱。** 它确实创造大量岗位，但创造的是**追赶型义务工作**：实现每个新模型、验证收敛、维护数据管线与 checkpoint 转换。对标物是一个成熟且免费的竞品，管理层随时可以问「为什么不直接用 Megatron」，而团队永远只能回答「我们在追平」。团队变大了，但工作不可发表、不可对外、也不构成个人履历。

### 为什么推荐 C

**参照物选错了。** NVIDIA 没有去 fork PyTorch 或 JAX，而是造了 Transformer Engine——一个被 PyTorch、JAX、Megatron、NeMo 共同调用的横向层。TE 团队规模大、mandate 硬、差异化明确，而且**上游是它的消费者而不是它的对手**。值得借的是这个**组织位置**，不是 TE 的技术范围（范围上的对位物是 Primus-Turbo，见[下节](#2-三层边界以及与-te--turbo-的关系)）。AMD 需要的是「Turbo + 一个真正的 runtime」，不是一个 Megatron 克隆。

**它不是新建，而是归位。** 3.3 万行既有实现（`core/pipeline_parallel` 8,975 行的调度器、`transformer_engine/` 4,045 行的 TE 重实现等）、26 个语义重写 patch、加上 roadmap 上的 MoE-Native IR、统一 comm overlap 调度器、full-iteration HIP graph capture、super-kernel 产品化、HybridEP-AMD、MXFP8——这本来就是一个完整的 execution / runtime 层。它现在按「影子化了谁的命名空间」归档在 `backends/` 下，所以既讲不出产品故事，也支撑不起编制，还导致同一能力被跨后端重复实现。

**它解决了 A 和 B 都解决不了的技术问题。** HIP graph 捕获整个 step、跨 PP/EP 的统一通信调度、comm-aware 编译期调度，都要求对执行流有控制权。这些能力在 patch 形态下做不干净，在 B 形态下能做但要付模型覆盖的代价。C 形态下，Megatron 负责模型定义，执行层负责怎么跑——两边都做自己擅长的事。

**它的差异化天然成立。** gfx950 的 XGMI 拓扑、SDMA、tile 级 comm-compute 重叠没有上游竞品，因为 NVIDIA 不会替 AMD 做。这类工作可发表、可对外、可作为硬件卖点，编制是可辩护的。

---

## 选项 C 展开：执行层的边界、模块与交付

工作名 **Primus-Runtime**（下称「执行层」）。

### 1. 它不是新建项目，而是把已有资产归位

这是 C 与 B 最本质的区别，也是 C 风险远低于直觉的原因：**待归位的 3.3 万行代码已经存在、已经在跑生产 recipe、已经在付维护成本**（见上文[关键发现](#关键发现patch-只占后端代码的-17)）。C 的第一期不写新功能，只做三件事——把 `backends/megatron/core/*` 与 `backends/transformer_engine/*` 中与后端无关的部分抽出、定义调用接口、让 TorchTitan 复用。

因此 C 的立项论证不是「请批准我们造一个框架」，而是「我们已经有一个执行层，请批准把它从三个后端命名空间里合并出来，以消除重复实现并使其可对外」。后者在管理层面前是完全不同量级的说服难度。

### 2. 三层边界（以及与 TE / Turbo 的关系）

| 层 | 归属 | 职责 | 现状 |
|---|---|---|---|
| **算子层** | Primus-Turbo（已存在） | 单个 kernel 怎么算快：grouped GEMM、FP8/MXFP8、permute、attention、DeepEP 原语 | 已是独立仓，边界清晰 |
| **执行层** | **Primus-Runtime（待立项）** | 一个 step 怎么跑：调度、通信编排、内存规划、graph 捕获、精度状态机 | **散落在 3 个后端命名空间 + 26 个 patch** |
| **框架层** | Megatron / TorchTitan / MaxText（上游） | 模型定义、数据管线、checkpoint、收敛验证 | 保持上游，不动 |

判据一句话：**「换一块 GPU 就要重写」的东西属于下两层；「换一个模型就要重写」的东西属于框架层。** 按这条线切，模型覆盖继续白拿上游，硬件差异化完全归自己。

**注意：「AMD 的 TE」这个说法只指组织位置，不指技术范围。** 前文借 TE 类比，指的是「被多个上游框架调用的横向层」这一定位。范围上必须分清：TE 的实际内容是融合算子 + FP8 recipe + comm overlap（userbuffers），**这部分在 AMD 侧已由 Primus-Turbo 承担——严格说 Turbo 才是 AMD 的 TE**。TE 从不管调度，管调度的是 Megatron。因此 **Primus-Runtime 没有 NVIDIA 的直接对位物**，它是「Megatron 中已被 AMD 事实接管的执行逻辑」+「TE 中 comm overlap 那一块」的合并体：

| NVIDIA 侧 | AMD 当前位置 | 目标归属 |
|---|---|---|
| TE：融合算子 + FP8 recipe | Primus-Turbo（独立仓）+ `backends/transformer_engine/` 4,045 行 TE 局部 fork | Turbo |
| TE userbuffers：comm overlap | `backends/transformer_engine/transformer_engine_torch/comm_overlap.py` 542 行 | **Runtime R2** |
| Megatron-Core：schedules / distributed / optimizer | `backends/megatron/core/*` 约 29,000 行 | **Runtime R1 / R3 / R5 / R6** |
| Megatron-Core：模型定义 / 数据 / checkpoint | 上游 | 上游，不动 |

**当前这几层已经缠死，这是「归位」而非「新建」的最强证据。** `primus/backends/transformer_engine/` 的 4,045 行是**vendor 了 TE 源码再改**（文件头均为 "Modification Copyright© 2025 AMD"，目录逐一镜像 TE 的 `pytorch/` 与 `transformer_engine_torch/`），且：

- **4 个文件反向 `import megatron`**（`pytorch/module/base.py`、`pytorch/cpp_extensions/gemm.py`、`transformer_engine_torch/gemm.py`、`transformer_engine_torch/comm_overlap.py`）——一个影子化 TE 命名空间的层，依赖了它上面两级的框架。
- `comm_overlap.py` 一个文件同时 import `transformer_engine_torch`、`primus_turbo`、`hip`、`megatron.core.utils`——四层全穿。

换句话说，今天的分层不是「Turbo → TE → Megatron」，而是一团双向依赖。R2 之所以被选为 C0 的首个搬迁目标，正是因为它同时是重复实现最明显（两份 SDMA collectives）和层级错乱最严重的地方。

### 3. 它具体做什么：一个 step 的八个决策点

「执行层」这个词太抽象，落到实处它就是**回答下面八个问题的那组策略与机制**——全部与「模型是什么」无关，全部与「跑在哪块 GPU 上」有关。七个今天已经在做，第八个做不了：

| 决策点 | 现有实现 | 行数 |
|---|---|---|
| micro-batch 按什么顺序跑 | `pipeline_parallel/zerobubble/scheduler/` 下 8 种策略（`basic1f1b` / `seq1f1b` / `v1f1b` / `vpp` / `zb` / `zbv` / `zbv_greedy` / `group_interleaved_1f1b`）+ 自动求解器 `v_auto_schedule.py` (928) | 约 3,000 |
| fwd/bwd 之间通信何时发、走哪条链路 | `zerobubble/scheduler/communication.py` | 700 |
| 反向是否拆 dX/dW、dW 推迟到何时算 | `extensions/zbpp_gemm.py` (869)、`te_gemm_patch_wgrad.py` (736)、`te_group_gemm_patch_wgrad.py` (372)、`te_wgrad_store.py` (130) | 2,107 |
| 激活留着、重算、还是 offload 到 host | `zerobubble/offload.py` (547) + `scheduler/offloading.py` (460) | 1,007 |
| 参数 all-gather 走 RCCL 还是 SDMA 引擎 | `distributed/sdma_param_gather.py` (248) + `fsdp2_fp8_all_gather.py` (654) | 902 |
| FP8 量化在哪一步做、scale cache 何时失效 | `extensions/primus_turbo_float8_local.py` | 1,703 |
| 哪些算子走 Turbo、哪些走 TE | `extensions/primus_turbo.py` (2,056) + `primus_turbo_local_spec.py` (318) + `transformer_engine_spec_provider.py` (222) | 2,596 |
| **整个 step 能否被 HIP graph 捕获** | **尚不存在**——roadmap 项，且在 patch 形态下做不出来 | **0** |

两个细节改变了这件事的性质：

- **已经有一个调度 IR 和 pass 框架。** `zerobubble/scheduler/graph.py` (240) + `passes.py` (174)，配 928 行自动调度求解器和 193 行调度可视化器 `pp_visualizer.py`。这不是适配胶水，是编译器基础设施。roadmap 上的「MoE-Native IR」因此也不是从零起步——它是给这套已有 IR 加通信感知的代价模型。
- **`pipeline_parallel/` 里其实是两套运行时。** `zerobubble/` 约 7,542 行 + `primuspipe/`（自有 launcher + fwd/bwd/combined/communication handlers）1,085 行。两套并存本身就说明这一层缺少归属和统一设计。

**没有层边界的代价已经在账上。** `core/transformer/moe/` 下有 `deprecated_20251209/` 与 `deprecated_2caa681a1/` 两个目录，共 **3,145 行按上游日期与 commit 冻结的 Megatron MoE 实现快照**（experts / token_dispatcher / router / moe_layer 各一份，两代并存）。因为无处安放自己的 MoE 执行逻辑，只能整份 vendor 上游代码再改，上游每变动一次就多冻一代。这类债务在 A 方案下永远还不完，而它正是 R4 要消除的东西。

### 4. 模块划分

由上述决策点、26 个语义重写 patch、3.3 万行既有实现、以及 H2 roadmap 交叉得出。「现有资产」列说明每个模块不是从零开始：

| 模块 | 现有资产 | 现有 patch | roadmap 关联 | 差异化来源 |
|---|---|---|---|---|
| **R1 调度器**<br>pipeline / 1F1B / zero-bubble / DualPipe-V / warmup | `core/pipeline_parallel` 8,975 行 | 8 个（`pp.schedule`、`pp.v_schedule`、`pp.zero_bubble_optimizer`、`forward_step.zero_bubble`、`pipeline_parallel_layer_layout`、`pp_warmup`、`train_step_seq_split`、torchtitan `pipelining_schedules_dualpipev`） | 统一 comm overlap 调度器、1F1B EP A2A overlap | 调度与 XGMI/RDMA 带宽比强相关，NV 的最优解不是 AMD 的最优解 |
| **R2 通信编排**<br>collective 后端 / SDMA / symmetric memory / A2A | `transformer_engine/comm_overlap.py` 542 行 | 3 个（`sdma_symm_mem_collectives` ×2 后端、`sdma_param_all_gather`、`skip_redundant_mp_sync`） | HybridEP-AMD、DeepEP v2、token-level chunked A2A | **纯 AMD 独有**：XGMI 拓扑、SDMA 引擎、rocSHMEM |
| **R3 梯度与反向分解**<br>dX/dW 拆分、wgrad 调度 | 部分在 `core/pipeline_parallel` | 3 个（`pp.linear_grad_split`、`pp.te_wgrad_split`、`legacy_grouped_mlp_wgrad`） | backward comm scheduling（研究线 D） | dW AllReduce 完全隐藏依赖对 SDMA 的控制 |
| **R4 MoE 层执行**<br>dispatcher / router / permute | `transformer_engine/permutation.py` 1,761 行 | 3 个（`turbo.moe_dispatcher`、`moe.primus_topk_router`、`moe.skip_identity_sort`） | super-kernel 产品化、MoE ECHO、comm-aware routing | tile 级 comm-compute 融合（RocMoE / MonolithEP 已验证 4.82 ms / 8×MI355X） |
| **R5 精度与优化器状态**<br>FP8 cache / precision-aware / Muon | `core/optimizer` 2,343 行、`fp8_utils` + `fp4_utils` 639 行 | 5 个（`precision_aware_fp8_tensorwise`、`fsdp2_fp32_param`、`fsdp2_bf16_master_weight`、`train_step_fp8_cache_update`、`optimizer.muon`） | MXFP8、FP8 primary weights for Muon、CPU-offload optimizer | MXFP8 block scaling 在 gfx950 上的 layout 与 NV 不同 |
| **R6 内存与 graph**<br>recompute / offload / HIP graph 捕获 | `core/distributed` 1,418 行 | 4 个（`custom_recompute_layer_ids`、`fsdp.device_mesh`、`fsdp.torch_fsdp2`、`megatron_fsdp`） | full-iteration HIP graph、fine-grained offloading、paged stashing | HIP graph 与 CUDA graph 的捕获语义差异 |

六个模块，每个都有既有代码、既有 patch、既有 roadmap 条目和明确的硬件差异化来源。R2 和 R4 是最不可能被上游抢掉的——因为它们的价值直接来自 AMD 硬件特性。

### 5. 怎么组织：结构、规则，以及一个停在半路的重构

**这个组织方式不需要设计——它已经在仓库里跑着。** `primus/core/pipeline_parallel/`（2,277 行）已经是一个后端无关的运行时内核：`scheduler/scheduler_node.py` 定义 IR 的 op 类型，`scheduler/algorithms/` 放 7 个后端无关的调度算法（1,738 行），`scheduler/schedule_table_factory.py` 是策略注册表，`handler/offload_handler.py` 与 `handler/wgrad_handler.py` 是默认 handler。

而 megatron 侧的 `primuspipe/` 只提供**一张按 op 类型索引的 handler dict**（`primuspipe/handlers/__init__.py`）：11 个 op 中 4 个直接复用 core 的默认实现（`W` → `default_wgrad_handler`、`O`/`R` → `default_offload_handler`/`default_reload_handler`），后端只负责 fwd / bwd / combined / p2p 四类约 700 行。

**plan/execute 分离已经产生了第二个消费者。** `primus/core/projection/`（13,394 行）的 simulator 使用**同一套 scheduler 内核**——同一个调度计划既驱动真实执行，也驱动性能预测。这是抽象正确性的最好证据，也是 Projection 在 MoE 案例上做到 1.4% 误差的原因。

#### 目标结构

```
primus/runtime/
├── ir/            op 类型、schedule graph、passes      ← 合并 core/scheduler_node.py 与 zerobubble/scheduler/{graph,passes}.py
├── planner/       调度算法、通信规划、内存规划          ← core/scheduler/algorithms/ + zerobubble/scheduler/communication.py
├── executor/      按 plan 驱动 handler                  ← primuspipe/pipeline_launcher.py + zerobubble/runtime.py
├── handlers/      后端无关的默认 handler                ← core/pipeline_parallel/handler/*（已存在）
├── comm/          RCCL / SDMA / symmetric-mem / DeepEP  ← R2，现散在 3 处、重复 2 份
├── precision/     FP8/MXFP8 scale 状态机与 cache 失效   ← R5
├── memory/        offload / recompute / HIP graph 捕获  ← R6
├── platform/      gfx942 / gfx950 能力与拓扑描述
└── adapters/{megatron,torchtitan,maxtext}/   仅 handler dict + spec provider
```

#### 四条组织规则

| 规则 | 状态 | 说明 |
|---|---|---|
| **plan / execute 分离** | 已验证 | planner 产出可序列化 plan，executor 只按 plan 驱动 handler。双消费者（真实执行 + Projection simulator）已证明可行 |
| **handler dict 是唯一后端接缝** | 已验证 | 后端只提供 `{op_type: callable}`，不参与调度决策。`megatron_primuspipe_handler_dict` 是现成样板 |
| **依赖方向单向**：adapters → runtime → platform / Turbo | **当前被违反** | `backends/transformer_engine/` 有 4 个文件反向 `import megatron`。需 CI 规则（import-linter 之类）强制，否则会再缠回去 |
| **能力发现而非硬编码** | **缺失** | platform 层声明「有无 SDMA / symmetric memory / XGMI 全连接」，planner 查询能力后决策，而不是在调度算法里写 `if gfx950`。这是同一套 planner 同时服务 MI300X 与 MI355X 的前提 |

#### 重要前提：这个重构停在半路，现在是两份并存

调度算法目前有**两套**：`core/pipeline_parallel/scheduler/algorithms/`（7 个，后端无关）与 `backends/megatron/core/pipeline_parallel/zerobubble/scheduler/`（8 个，绑定 megatron）。按文件名看 `zbv_greedy`、`basic1f1b`/`basic_1f1b`、`zb`/`zerobubble`、`interleaved_1f1b`/`group_interleaved_1f1b` 均为重复。

因此 **C0 的准确描述不是「启动一次重构」，而是「把已经停下的那次重构做完」**。这不只是措辞——架构可行性已被现有代码走通一次，风险画像与从零重构完全不同。不做的代价则是持续的：两份调度算法并存本身就是维护危害，且重复已泄漏到用户配置面（`v-half` / `v-min` 在两套栈里是两套配置写法）。

**但必须如实标注验证边界**：干净架构（core + `primuspipe`）目前**只被 2 个测试 yaml 使用**，生产 recipe（customer_package 的 Qwen3 脚本，默认 `PP_STRATEGY="zbv"`）走的是 legacy 的 `--patch_zero_bubble` 栈。所以「已走通」指测试与 Projection 层面，不含生产规模。这抬高了 C0 的价值（好架构尚未兑现生产收益），同时也抬高了风险（迁移目标未经生产检验），因此 C0 必须双跑对拍而非直接切换。

### 6. API 面：三类接入点

**先排除一个高概率误读：这不是「开发一堆组件给 Megatron 调用」。** 那个描述准确刻画的是 Primus-Turbo——grouped GEMM、permute、attention 是组件，Megatron 持有控制流并逐个调用它们。执行层是**控制反转**的：`get_primus_pipeline_parallel_fwd_backward_func()` 返回 `PrimusPipelineParallelLauncher().run`，即 Megatron 的 `forward_backward_func` 被整体替换，Megatron 交出一个 step 的执行权，再由执行层按自己的 plan 回调后端的 `megatron_fwd_handler` / `megatron_bwd_handler`（`backends/megatron/core/pipeline_parallel/schedules.py:120-121`）。

这个区别决定能力上限：组件库的每次调用彼此独立，只能在单次调用内优化，得到 per-op 局部最优。[八个决策点](#3-它具体做什么一个-step-的八个决策点)中的第八项（full-iteration HIP graph 捕获）在组件形态下永远做不到——捕获整个 step 要求拥有整个 step 的控制流。跨 PP stage 的通信编排、把 dW 推迟数个 micro-batch、按全局 schedule 决定哪层 offload，同理都需要全局视野。

**推论（也是执行风险）**：若该项目被当作「开发一堆组件」来推进，它会退化成一个零件箱——各组件局部最优、没有统一 plan、最终既无差异化也无产品叙事。守住 plan/execute 分离是成立前提，不是实现细节。

在此前提下，执行层不 fork、不改写上游，只暴露三种被调用的方式。按侵入性从低到高：

| 接入方式 | 机制 | 已有先例 | 适用模块 |
|---|---|---|---|
| **Spec / provider 注入** | 上游用 `TransformerLayerSpec` / submodule spec 描述层结构，执行层提供替换 provider | `turbo/te_spec_provider_patches.py`、`gpt_decoder_layer_specs_patches.py`（已走通一半） | R4、R5 |
| **调度器注册** | 上游从注册表按名取调度实现，而非硬编码 `forward_backward_func` | torchtitan 的 `pipelining_schedules` 已接近此形态 | R1、R3 |
| **Collective 后端替换** | 上游经 process-group / 抽象通信接口调用，执行层提供 AMD 后端 | `sdma_symm_mem_collectives`（现为 patch，本应是后端插件） | R2、R6 |

**这三类里前两类上游已部分具备，这是 C 可行性的核心依据**——不是要求上游为 AMD 新增架构，而是要求把已有的扩展点补完整。第三类是需要真正推动的新扩展点，建议以「多加速器厂商共同受益」而非「AMD 专用」的形式提出。

上游接受度的验证成本很低：先提 2–3 个 PR 试水，一个季度内就能拿到明确信号，不需要先投入重构。

### 7. 为什么这个 mandate 支撑得住团队扩张

关键在于**每个模块产出的都是可对外、可发表、可作为硬件卖点的工作**，而不是追赶型义务工作：

| 模块 | 岗位性质 | 可对外产出 |
|---|---|---|
| R1 调度器 | 分布式系统 | 论文 / 上游 PR / blog |
| R2 通信编排 | 通信 + 硬件底层 | rocSHMEM / DeepEP 上游贡献、MLPerf 成绩 |
| R3 反向分解 | 编译 / 调度 | 论文（研究线 D 已在 roadmap） |
| R4 MoE 执行 | kernel + 系统协同 | super-kernel 论文、tokens/s 硬指标 |
| R5 精度 | 数值 + 系统 | FP8/MXFP8 收敛报告、recipe |
| R6 内存与 graph | 运行时 | 端到端显存与吞吐指标 |

对比 B（自研框架）产出的是「实现了第 N 个模型、验证了收敛」——这类工作既无法发表也无法差异化。**同样的编制增长，C 给出的是可辩护的岗位，B 给出的是跑步机岗位。**

### 8. 分期与验收标准

> 更细的工作分解、IR 合并细节、依赖顺序与未决问题见配套规划：[`2026-08-04_1049_primus-runtime-design-and-plan.md`](./2026-08-04_1049_primus-runtime-design-and-plan.md)。下表为管理层视角的摘要。

| 期 | 交付物 | 验收标准（可量化） |
|---|---|---|
| **C0**（数周） | 归位可行性验证：从 `backends/megatron/core/pipeline_parallel` 与 `backends/transformer_engine` 中抽出与后端无关的部分，形成 R1/R2 骨架 | 消除 `sdma_symm_mem_collectives` 的两份重复实现（305 行 → 单份）；现有 recipe 吞吐**零回退**（基线：Qwen3-235B-A22B 4,809.1 tokens/s、DSv3 671B 配方） |
| **C1**（一个季度） | R2 作为 collective 后端插件被两个后端共同调用；向上游提 2–3 个扩展点 PR | TorchTitan 复用 R2 而非自带实现；上游 PR 收到明确 accept/reject 信号（**这是 C 的 go/no-go 决策点**） |
| **C2**（两个季度） | 首个「patch 形态做不出来」的能力落地：**full-iteration HIP graph 捕获 + 统一 comm overlap 调度器**（R1 + R6） | 端到端 step 完整捕获成功；host launch 开销降低可测量；MoE 小模型区间吞吐提升 |
| **C3**（战略） | 执行层作为独立可交付物，Megatron / TorchTitan / MaxText 均为消费者；super-kernel 经 R4 产品化 | 三后端至少两个在生产 recipe 中调用执行层；super-kernel 达 roadmap 目标（BF16 ≤ 7 ms、FP8 ≤ 5 ms，DSv3 T_src=2048 / 8×MI355X） |

**C2 是整个方案的论证支点**：选它作为第一个新能力，是因为 full-iteration HIP graph 需要控制整个 step 的执行流，在 monkey-patch 形态下做不干净——它一旦做成，就用工程事实而非 PPT 证明了这一层的必要性。

### 9. C 的风险与反驳

- **上游拒绝扩展点。** 后果：R2/R6 退化为更结构化的 patch，R1/R4/R5 仍可通过已有 spec 机制走通。缓解：C1 阶段先低成本试水拿信号，再决定是否投重构；扩展点设计对所有非 NV 加速器普适。
- **抽象引入性能回退。** patch 是贴着热点写的，多一层间接就可能吃掉收益。缓解：C0 起把现有 recipe 设为硬性回归门，任何抽象不得回退；这也是 C0 不写新功能只做归位的原因——先证明抽象是零成本的。
- **归位过程中破坏生产 recipe。** 3.3 万行代码正在跑生产。缓解：按模块逐个搬，每次搬完跑一遍 DSv3 与 Qwen3-235B 的 loss parity + 吞吐；先搬重复实现最明显的 R2（收益清楚、面积小）。
- **和 Primus-Turbo 边界打架。** 「算子 vs 编排」在 MoE super-kernel 这类跨层融合工作上本来就模糊。缓解：以「是否需要跨层调度信息」为判据——需要知道 PP stage / 通信时序的归执行层，只看单个 tensor 的归 Turbo。
- **人力在归位期被占用而无新特性产出。** 这是真实代价。缓解：C0 控制在数周内并只做 R2 一个模块；其余模块的归位与新功能开发并行，不设「先重构完再做事」的门。

---

## 建议

**阶段 0：先修可观测性（数日，任何路线的前提）**

顺序很重要——在 patch 层「静默失效」可能发生的情况下补版本门，是把风险放大而非缩小。

- `_get_backend_version()` 改为 fail-loud：区分「探测失败」与「版本不匹配」两种语义，前者应报错而非静默把 patch 全部跳过。
- 启动时输出 applied-patch manifest（预期数 vs 实际数），数量不符即失败退出。今天 patch 是否生效只能靠翻日志里的 `⊘ Skipped`。
- `PatchRegistry.register` 的重复 id 从 `log.warning` 升级为报错（`patch_registry.py:57-59` 当前是静默覆盖）。
- `_patch_guard` 从 `backends/megatron/patches/` 提升到 core，或至少覆盖 torchtitan——其 docstring 已说明 core runner 无 re-apply 保护，wrapping 类 patch 二次应用会静默双重生效。

**阶段 1：让归属变得机器可读（数周，与战略选择无关）**
- 给 `register_patch` 增加 `targets=[...]` 字段，声明所替换的上游符号，并回填 97 个 patch。当前 0/97 有此声明，这是 bump 时无法做影响面分析的根因。
- CI 增加 lock-bump 影响报告：`_thirdparty.lock` 变更时，diff 新旧 commit 间 `targets` 涉及符号的源码，列出需人工复核的 patch。这比版本门更贴合「按 commit 钉死」的实际做法。
- 在阶段 0 完成后，为 patch 补 `backend_versions`，形式对齐 `te_spec_provider_patches.py` 的 `["<0.17"]`——即每个 patch 都声明自己的**到期条件**。
- 把 41 个配置与 UX patch 迁出 patch 层，进 `primus/configs/modules/megatron`。这批占 42% 的数量但几乎不含技术纵深，迁出后 patch 层的真实性质才看得清。
- 把 3 个 ROCm 绕过推上游，并建立「ROCm 修复必须有对应上游 PR」的规则，让这个数字长期趋零。

**阶段 2：收敛注入方式（一个季度内，可验证）**
- 把 26 个 Turbo 注入 patch 改造成走 Megatron 的 spec / submodule provider 钩子。目标是从「Primus 改写 Megatron」变成「Megatron 调用 Primus」。
- 对 26 个语义重写 patch 逐个写出所需的上游扩展点，提 2–3 个 PR 试水。**上游的 accept/reject 信号是选项 C 的 go/no-go 决策点**，且获取成本远低于先做重构。

**阶段 3：执行层立项**

见[选项 C 展开](#选项-c-展开执行层的边界模块与交付)的 C0–C3 分期与验收标准。要点：C0 只做归位不写新功能（先证明抽象零成本），首个新能力选 full-iteration HIP graph + 统一 comm overlap 调度器（在 patch 形态下做不出来，因此天然论证这一层的必要性）。

---

## 风险与反对意见（诚实列出）

选项 C 自身的风险见[C 的风险与反驳](#9-c-的风险与反驳)。以下是本备忘整体层面的：

- **本备忘的分类含判断成分。** 归属分类由路径与 patch id 规则得出，边界 case（如 FSDP 相关 patch 归入语义重写）可争议。分类脚本可重跑，欢迎按不同规则复核。
- **3.3 万行的统计口径可质疑。** 该数字是 `primus/backends/megatron/core/*` 与 `primus/backends/transformer_engine/*` 的非 patch 行数之和，其中必然有一部分是真正的后端适配代码而非可复用执行逻辑。真实可归位比例需在 C0 阶段逐模块核实；结论「已有资产远大于 patch 层」对这个比例不敏感。
- **抽象设计能力是真门槛。** 跨三后端的执行层比 97 个 patch 难写得多，做失败的内部框架比 patch 层更糟。缓解：C0 只搬一个模块（R2）验证。
- **本文未评估人力与排期。** 各期的绝对工时需与团队一起估；本备忘只提供分期结构与验收标准。

---

## Next

- [ ] 阶段 0 全部四项（可立即开工，且是补版本门的前提）
- [ ] 给 `register_patch` 加 `targets=` 字段并回填，接 CI lock-bump 影响报告
- [ ] 输出 41 个配置类 patch 的迁出清单，评估工作量
- [ ] 逐个走查 26 个语义重写 patch，列出所需的上游扩展点，选 2–3 个提 PR 试水
- [ ] C0 可行性验证：核实 `core/pipeline_parallel` 与 `transformer_engine/` 中真正与后端无关的比例
- [ ] 若需对外沟通，本备忘出英文版
