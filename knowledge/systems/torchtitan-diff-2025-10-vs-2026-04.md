# TorchTitan 演进分析：2025-10 vs 2026-04

> 报告生成日期：2026-04-21
>
> 对比基准：
> - **旧版本（参考）**：`4db8f6e6` (2025-10-16, "ci: add codespell pre-commit hook" #1899)
> - **新版本（HEAD/main）**：`70b044c4` (2026-04-21, "Bump tj-actions/changed-files...")
> - **跨度**：约 6 个月，**496 个 commits**，**446 个文件改动**，**+56,069 / −17,680 行**
>
> 来源：`https://github.com/pytorch/torchtitan`

---

## 0. TL;DR

过去半年 torchtitan 经历了一次 **架构级 BC-breaking 重构**，核心信号包括：

1. **配置系统从 TOML 迁移到 Python Dataclass Registry**（PR #2386），`CONFIG_FILE=*.toml` → `MODULE=xxx CONFIG=yyy` 的全新 CLI；
2. **模型代码目录扁平化**（`models/<x>/model/model.py` → `models/<x>/model.py`，新增 `config_registry.py`），并引入新的 `Configurable` / `Module` 协议；
3. **`ParallelDims` 用 `DeviceMesh._unflatten` 重写**（PR #1660），三套子 mesh（dataloading / dense / sparse）替代原来手写 EP+非EP 两套分支；
4. **EP/MoE 大跃进**：`token_dispatcher` 抽象、DeepEP / HybridEP（GB200/NVL72）、MoE node-limited routing、float32 router、Per-layer compile；
5. **训练新形态**：MXFP8（B200/GB200，最高 28% 加速）、BF16 优化器状态、SFT（ChatDataset）、Full DTensor、Compute-Comm Overlap；
6. **新模型/能力**：`gpt_oss` 与 `flux` 从 experiments 毕业进入 core；新增 `qwen3_vl`（含 30B-A3B / 235B-A22B MoE 多模态）；Llama3 weight-tying；DeepSeek-V3 16B/671B 完整 toml；
7. **实验区彻底改组**：`flux`/`simple_fsdp`/`compiler_toolkit`/`torchcomms`/`moe_symm_mem_kernels` 全部下线或并入 core；新增 `graph_trainer`（含 SimpleFSDP、precompile）、`rl`（vLLM + Monarch + TorchStore，GRPO）、`transformers_modeling_backend`、`ft`（DiLoCo/MCCL，从 components 移到 experiments）；
8. **趋势**：core 越来越薄、可组合，编译路线 = `graph_trainer`，规模化路线 = `autoparallel` + Full DTensor，RL 路线 = `rl`，多 vendor 路线 = ROCm CI / AMD Fork / XPU 讨论。

---

## 1. 外部接口（CLI / 配置 / 训练入口）

### 1.1 CLI 启动方式（BC Break）

| 维度 | 2025-10 | 2026-04 |
|---|---|---|
| 入口脚本 | `CONFIG_FILE=./.../llama3_8b.toml ./run_train.sh` | `MODULE=llama3 CONFIG=llama3_8b ./run_train.sh` |
| 训练模块 | `TRAIN_FILE=torchtitan.train` (可换) | 固定 `-m torchtitan.train --module ${MODULE} --config ${CONFIG}` |
| Debug 模式 | 无 | 新增 `COMM_MODE=fake_backend\|local_tensor`（无 GPU 验证 / 单卡模拟多卡） |
| Pre-train 模式 | torchrun + NCCL | `LOCAL_RANK=0 python3 -m torchtitan.train ... --comm.mode=...`（dry-run） |

`run_train.sh` 现在支持两种执行路径：常规 `torchrun` 路径与不需要 GPU 的 `COMM_MODE=fake_backend` / `local_tensor` 路径。后者来自 [Local Tensor]/[Fake mode] 工作（PR #2057），可在单卡上做配置校验和数值验证。

### 1.2 配置系统：TOML → Python Dataclass Registry（核心 BC Break）

**PR #2386 [BC Breaking] Config System Refactor: TOML to Python Dataclass Registry** 是过去半年最重要的接口变化。

**旧路径**：`torchtitan/config/job_config.py`（910 行），`torchtitan/models/<x>/train_configs/*.toml` 提供模型配置。

**新路径**：
- `torchtitan/config/configs.py`（397 行）：拆出 `TrainingConfig / ParallelismConfig / ActivationCheckpointConfig / CompileConfig / CommConfig / DebugConfig` 等 dataclass。
- `torchtitan/config/configurable.py`（113 行）：新基类 `Configurable`，所有可配置组件（tokenizer / dataloader / optimizer / 模型层）继承它，提供 `Config` 嵌套类与自动 `build()` 机制（要求 `@dataclass(kw_only=True, slots=True)`）。
- 每个模型新增 `torchtitan/models/<x>/config_registry.py`，以 Python 函数返回 `Trainer.Config`，例如 `llama3_debugmodel()`、`llama3_8b()`、`deepseek_v3_16b()`。所有 `train_configs/*.toml` **已删除**。
- CLI 通过 `tyro` 进行 dataclass → 命令行映射（`--training.steps 1`、`--parallelism.tensor_parallel_degree 8` 等）。

**优点**：
- 类型安全、IDE 友好、可继承复用配置（Trainer.Config 可被实验子类化）；
- 配置即代码：support imports、partial / lambda、字典字段（`per_op_sac_force_recompute_mm_shapes_by_fqns` 等）；
- 实验自定义 `Trainer.Config` 子类即可注入新字段（见 `docs/extension.md`）。

**代价**：
- 已有的 `*.toml` 用户需要迁移；
- 新写实验时强制要求 `Configurable` 模式 + `kw_only=True, slots=True`；
- 学习曲线较陡，配置到底是哪些参数得读 `configs.py`。

### 1.3 入口与 Trainer 拆分

旧版本 `train.py` 一个文件 778 行，包含完整 Trainer 类。新版本：

- `torchtitan/train.py`（~80 行）只做 `init_logger → ConfigManager().parse_args → Trainer().train`；
- `torchtitan/trainer.py`（**新增 878 行**）独立的 `Trainer` 类：构造组件、device/mesh/PG 初始化、训练循环、checkpoint、PP forward/backward；
- `init_distributed_env` 从 `Trainer.__init__` 拆出（PR #2003 `[RFC] Seperate init_distributed_env from the Trainer.__init__`）；
- 实验可继承 `Trainer` + `Trainer.Config`，例如 `experiments/ft/trainer.py` 重写训练循环以支持 DiLoCo/HSDP+TorchFT。

### 1.4 模型注册接口：`TrainSpec` → `ModelSpec`

| 旧 | 新 |
|---|---|
| `torchtitan/protocols/train_spec.py` | `torchtitan/protocols/model_spec.py` |
| `TrainSpec` 包含模型类、所有 builder（dataloader、optimizer、lr_scheduler、metrics、validator、tokenizer） | `ModelSpec` 只保留模型 + parallelize + pipeline + loss + state_dict_adapter |
| 全局 `register_train_spec()` 注册 | 模型 `__init__.py` 提供 `model_registry(flavor) -> ModelSpec`；其余组件通过 `Trainer.Config` 中的 `Configurable.Config` 注入 |

新增 `protocols/module.py`（172 行）定义 `Module(nn.Module, Configurable)` 基类，`init_states` 自动递归子模块、按 `param_init` 字典初始化（PR #2633 `[Module] Refactor init_weights to config-based param_init system`）。这意味着模型层级别的初始化策略（trunc_normal、depth-scaled std、weight tying 顺序）都由配置驱动而不是硬编码。

### 1.5 数据集接口

- 旧的 `torchtitan/datasets/{__init__,hf_datasets}.py` 已废弃，统一搬迁到 `torchtitan/hf_datasets/`；
- 新增 `text_datasets.py`：除 `HuggingFaceTextDataLoader` 之外，新增 **`ChatDataset` / `ChatDataLoader`** 用于 SFT（PR #2556 `Add ChatDataset and ChatDataLoader for SFT training`），结合 `BaseTokenizer.apply_chat_template`（PR #2455）；
- 新增 `hf_datasets/multimodal/` 用于 VLM/Qwen3-VL；
- `BaseTokenizer` 增强：`apply_chat_template` + `IGNORE_INDEX` 标签 mask。

---

## 2. 并行策略变更

### 2.1 `ParallelDims` 重写（PR #1660 / 2025-12）

旧实现把"是否启用 EP"分成 `_build_mesh_with_ep` / `_build_mesh_without_ep` 两条手写路径，使用 `init_device_mesh` 一次性构建多维 mesh，再通过 `_flatten` 拼出 `dp` / `dp_shard_cp` / `dp_cp` / `ep` 子 mesh，对 EP 还需额外引入 `dp_shard_mod_ep` / `dp_shard_in_ep` 两个 dim。代码相当复杂（200+ 行）。

新实现使用 `DeviceMesh._unflatten` 一次性 build 出三套 mesh：

```text
dataloading_mesh: ["pp", "batch", "cp", "tp"]
dense_mesh:      ["pp", "dp_replicate", "fsdp",  "tp"]
sparse_mesh:     ["pp", "dp_replicate", "efsdp", "ep", "etp"]
```

其中 `batch = dp_replicate * dp_shard`，`fsdp = dp_shard * cp`，`efsdp = fsdp * tp / (etp * ep)`。

- 对于不存在的维度（degree=1 或 batch dim）使用 `backend_override = "fake"`，避免 PG 创建开销；
- `loss` 通过 batch+cp flatten 得到；
- 不再需要 `dp_shard_mod_ep` / `dp_shard_in_ep` 显式概念，约束放宽（旧版要求 `ep % cp == 0` 等已不再断言）；
- 暴露 `ParallelDims.from_config(parallelism_config, world_size)` 方法。

收益：可读性、扩展性显著上升；为 Full DTensor、HybridEP、AutoParallel 提供干净的 mesh 抽象。

### 2.2 `ParallelismConfig` 字段对比

新增字段（2026-04 相对 2025-10）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `enable_sequence_parallel` | `True` | 显式开关 SequenceParallel（之前隐式跟随 TP） |
| `context_parallel_load_balancer` | `"headtail"` | 新增 `headtail` (SDPA) / `ptrr` (FlexAttention) / `None` |
| `expert_parallel_comm_backend` | `"standard"` | 新增 `deepep` (H100/NVLink Switch)、`hybridep` (GB200/NVL72) 后端 |

被删除字段：
- `enable_compiled_autograd`（移到 `CompileConfig` / GraphTrainer）。

EP 相关约束放宽：旧版 `expert_parallel_degree` 字段下记录了一长串硬约束，新版只保留 `etp == tp or etp == 1`，其余由 mesh 计算自动满足。

### 2.3 `distributed/` 目录新增模块

| 新文件 | 功能 |
|---|---|
| `compile.py` | 抽出 `apply_compile(model, compile_config)`，含 FakeTensor monkey-patch；支持 per-block 或 whole-model 编译 |
| `context_parallel.py` | 新封装 `apply_cp_to_attention_module`，使用 PyTorch 新 CP API（`_context_parallel_shard`、`_HeadTailLoadBalancer`、`_PTRRLoadBalancer`）；PR #2144 `[CP] Refactor Context Parallel to use new PyTorch CP APIs` |
| `fsdp.py` | `disable_fsdp_gradient_division`、`get_fsdp_reshard_after_forward_policy`，统一与 ReplicateModule 的处理 |
| `deepep/` | DeepEP 后端集成的支持代码（PR #2107 集成 DeepEP，#2310 shared_experts overlap 与 deepep.combine()） |

`tensor_parallel.py` / `pipeline_parallel.py` / `expert_parallel.py` 都做了 refactor，去掉了死代码（如 `create_context_parallel_ctx`），并支持 PP/EP overlap（PR #1721）。

### 2.4 Pipeline Parallel

仍然以 PyTorch `_PipelineSchedule` 为基础，关键变化：
- `pipeline_parallel_first_stage_less_layers` / `pipeline_parallel_last_stage_less_layers` 进入 stable；
- `module_fqns_per_model_part` 显式列出每段 PP 的模块 FQN；
- `pipeline_llm` 函数从 `models/<x>/infra/pipeline.py` 上移到 `torchtitan/distributed/pipeline_parallel.py`，所有模型共享；
- 修复 `apply_compile` 在 PP init 中被多次调用（PR #2135）。

### 2.5 FSDP

- **FSDP1 → FSDP2** 已默认（之前已是，但本期把 `amp` / `replicate` 在 FSDP2 中统一为 `fully_shard`，PR #2900）；
- `fsdp_reshard_after_forward = "default" | "always" | "never"` 三档（之前只有 bool）；
- **FSDP 始终保留以应用 MixedPrecisionPolicy**：`_mesh_exist("fsdp", ...)` 总是返回 True，即使 degree=1，从而保证混合精度训练规则一致；
- 当 `ep=1` 时支持非 dim-0 FSDP 分片 MoE experts（PR #2668）；
- 启用 per-param mesh FSDP2 for MoE（PR #2281）；
- BF16 优化器状态：`OptimizersContainer.Config(implementation="fused_opt_states_bf16")`（PR #2732）：参数/梯度仍 fp32，仅 optimizer state 用 bf16，参考 DeepSeek-V3 671B 训练实践。

### 2.6 MoE / EP

过去半年 MoE 是改动最密集的子系统：

- **`models/moe/` 整目录搬到 `models/common/`**：`moe.py` / `kernels.py` / `utils.py` 都成为 common 组件，新增 `token_dispatcher.py`（PR #2842 `[MoE][1/n] Introduce token dispatcher and replace token reorderer`）；
- EP 路由配置从 trainer 移到 config registry（PR #2960 `[MoE][2/n] Move EP setup from trainer to config registry`）；
- **DeepEP** 集成（PR #2107），shared_experts 与 `deepep.combine()` overlap（PR #2310），DCP metadata bug 修复（PR #2227），refactor + 内存泄漏修复（PR #2296）；
- **HybridEP** 支持 GB200 / NVL72（PR #2207 `[HybridEP] Support hybridEP for GB200 with NVL72`），可通过 `HYBRIDEP_NUM_SMS_DISPATCH/COMBINE` 环境变量调节 SM 占用；
- **node-limited routing**（PR #2111）；
- MoE router gate 强制 fp32（PR #2389）、router 边界 all-reduce 修复（PR #2416）；
- gpt-oss MoE router gate bias + top-k renorm 修复（PR #2319）；
- MoE 模块化重构（PR #2615 `[Module] Modularize MoE components`）；
- 移除 MoE 不必要 padding（PR #2774），bmm → scatter add 回滚（PR #2775）；
- MoE 模型支持 non-strict tracing（PR #2612）。

### 2.7 Context Parallel

- 引入 PyTorch 新 CP API（PR #2144）；
- 新增 FlexCP（PR #2145）；
- `[Qwen3][GPT-OSS][DSV3] Enable FlexAttention with Context Parallel`（PR #2541）。

### 2.8 通信层

新增 `CommConfig.mode`：`"default" | "fake_backend" | "local_tensor" | "torchcomms"`：
- `fake_backend`：完全无 GPU 配置校验（CI dry-run）；
- `local_tensor`：单卡 LocalTensor 串行模拟所有 ranks，保证数值一致，便于在小规模上验证 5D 并行；
- `torchcomms`：替换 PG 后端为 torchcomms（PR #2510 `[torchcomms] Remove torchcomms experiment and integrate via comm.use_torchcomms config`）。原 `experiments/torchcomms/` 实验已下线，能力直接整合进 core 配置。

---

## 3. 新模型与新能力

### 3.1 模型矩阵对比

| 模型 | 2025-10 位置 | 2026-04 位置 | 变更 |
|---|---|---|---|
| llama3 / llama4 | `models/llama3` 4 子目录 | 扁平化 `model.py`/`parallelize.py`/`config_registry.py`/`state_dict_adapter.py` | 配置改为 Python；llama3 加 weight tying（#2580） |
| llama3_ft | `models/llama3_ft` 独立 | 已删除，并入 `experiments/ft/llama3` | DiLoCo 走实验路径 |
| deepseek_v3 | `models/deepseek_v3` 仅 16B/671B toml | 同上扁平化；toml 删除，改 Python | 加 FlexAttention+CP；test 4D+EP+PP（#2234） |
| qwen3 | `models/qwen3` (0.6B/1.7B/32B/moe_debug) | 加 14B（#2461）、varlen 14B（#2941） | 注意力修复多次；MoE 路径成熟 |
| gpt_oss | `experiments/gpt_oss` | 已**毕业**到 `models/gpt_oss/`（PR #2203） | 加 YaRN RoPE（#2216）、torch.compile（#2687） |
| flux | `experiments/flux` | 已**毕业**到 `models/flux/`（PR #1858） | CP 支持（#1851）、torch.compile + MXFP8（#2579） |
| **qwen3_vl** | 不存在 | **新增** `models/qwen3_vl/`（#2409 `[Qwen3VL] add qwen3 vl`） | ViT + Patch Merger + DeepStack + MRoPE，含 30B-A3B / 235B-A22B MoE |
| llama3_70b / 405b | toml | toml 删 → Python registry | — |

### 3.2 训练特性新增

| 特性 | 入口 | 关键 PR |
|---|---|---|
| **MXFP8 训练**（B200 dense + MoE） | `model_converters=[MXLinearConverter.Config(...)]`，`docs/mxfp8.md` | `Crusoe B200 1.28× 加速` Blog；MoE token group padding kernels in torchao（#2520） |
| **MXFP8 on AMD gfx950 (MI355X)** | `[ROCm] Support mxfp8 on gfx950`（#2222） | — |
| **BF16 优化器状态** | `optimizer.implementation="fused_opt_states_bf16"`，`docs/bf16_optimizer_states.md` | #2732 |
| **SFT** | `ChatDataset` + `ChatDataLoader` + `apply_chat_template` + `IGNORE_INDEX` | #2556 / #2455 |
| **Per-layer compile（含 MoE）** | `compile.enable=True`, `compile.components=["model","loss"]` | #2741 |
| **Full DTensor** | TP 区域全部使用 DTensor（Qwen3 / Llama4） | PR #2149（高优先级，仍在开发） |
| **Compute-Comm Overlap** | `Feature/rfc 2408 compute comms overlap` | #3020 |
| **fused RoPE** | llama3 8B tp2/fsdp4 promotion | #3039 / #3040 (CUDA Graph replay) |
| **Fused Linear projection within MLP** | qkv 融合（#3036 类） | #2931 |
| **weight tying** | Llama3 / Qwen3 fix | #2580 / #2253 / #2522 |
| **per-op Selective AC + memory_budget** | `ActivationCheckpointConfig.mode="memory_budget"` + Pareto SVG | refactor #2357 |
| **AC `preserve_rng_state=True` 默认** | 默认确定性 | #2380 |
| **Verl 集成** | TorchTitan 模型可被 Verl 调用 | #2333 |
| **多源 checkpoint loading** | `additional_load_path` | #2949 |
| **HF state dict 双向适配** | `state_dict_adapter.py` 各模型独立提供 | gpt-oss #2021 等 |

### 3.3 文档矩阵

| 文档 | 状态 |
|---|---|
| `docs/mxfp8.md` | **新增**（210 行），B200/GB200 案例 + 28% 加速 |
| `docs/bf16_optimizer_states.md` | **新增** |
| `docs/composability.md` | **新增**（PP-friendly model writing、TORCH_NCCL_AVOID_RECORD_STREAMS 等） |
| `docs/fsdp.md` | **新增**（FSDP1 → FSDP2 迁移指南） |
| `docs/torchft.md` | **删除**（迁移到 `experiments/ft/torchft.md`） |
| `docs/debugging.md` | 大幅扩充（+122 行），含 COMM_MODE 用法 |
| `docs/float8.md` / `checkpoint.md` / `extension.md` 等 | 全面 refresh |

---

## 4. 当前 experiments：核心定位与活跃度

新版本的 `experiments/README.md` 重新约定了"实验区"的契约：实验对 core 是单向依赖、有明确 owner、`torchtitan` team 保留下线权。当前活跃实验：

| 实验 | 核心定位 | Owner | 关键技术 |
|---|---|---|---|
| **graph_trainer** | 编译器路线主战场，把 SimpleFSDP / 编译工具链组合 | @ruisizhang123, @SherlockNoMad, @yiming0416 | AOT / JIT / aot_fx_trace 三种模式；CooR precompile；FX graph passes（remove-noop / detach / identity_view 等）；CUDAGraph；PR #2457 把原 `simple_fsdp` + `compiler_toolkit` 合并进来 |
| **autoparallel** | 把并行策略交给 PyTorch AutoParallel 自动决策 | @wconstab, @xmfan | `local_map_deepseek_v3` 演进；ROCm CI 支持（#2248） |
| **ft** | DiLoCo / HSDP + TorchFT 故障容忍训练 | @tushar00jain, @fegin | 从 `components/ft/` 移到这里；新增 `MCCL` 后端（#2829）；含独立 `trainer.py` |
| **vlm** | 通用 VLM 训练框架 | @lkhphuc, @shuhuayu | 与 `models/qwen3_vl` 互补 |
| **transformers_modeling_backend** | 直接训练 HuggingFace Transformers 模型 | @3outeille | `transformers==4.57.1`，4D parallelism + torch.compile；已验证 Llama3.2/Phi-2/Qwen2.5/Mistral/SeedCoder/Qwen3/Helium/OLMo/Ministral；MoE 进行中（#2679） |
| **rl** | RL 训练（GRPO + vLLM rollout） | @wwwjn | TorchTitan 模型定义被 vLLM wrapper 共享 → bitwise parity；Monarch 做 actor 编排；TorchStore 做权重同步（GPU-to-GPU RDMA）；Qwen3-14B GRPO（#2941）；batch-invariant kernels |
| **forge** | RL/Agent toolkit（孵化中） | @allenwang28, @joecummings 等 | TBA |

已下线 / 并入 core 的实验：
- `experiments/flux/` → `models/flux/`（毕业）
- `experiments/simple_fsdp/` → `experiments/graph_trainer/simple_fsdp.py`（合并）
- `experiments/compiler_toolkit/` → `experiments/graph_trainer/`（合并 #2457）
- `experiments/torchcomms/` → `comm.use_torchcomms` 配置（#2510）
- `experiments/moe_symm_mem_kernels/` → `models/common/`（#2426 cleanup）
- `experiments/gpt_oss/` → `models/gpt_oss/`（毕业 #2203）

---

## 5. 热点讨论与社区方向

### 5.1 高 reaction Open PR（截至 2026-04-21）

| PR | 主题 | 含义 |
|---|---|---|
| #3032 | quantize on config instead of on model | 量化进一步走 Configurable 路线 |
| #3007 | [HybridEP] Enable HybridEP with graph_trainer | HybridEP × 编译路线汇合 |
| #3050 / #3049 | GraphTrainer SAC pass refactor / enable_cudagraph 配置 | CUDA Graph 在编译路线上的成熟 |
| #2916 / #3042 | GraphTrainer CooR precompile for DSv3 / + CP support | 编译路线对 DSv3 与 CP 的扩展 |
| #2149 | [full DTensor] Use all DTensor for Qwen3 and llama4 at TP region | 高优先级，"Full DTensor" 是 Composability 的目标 |
| #2963 | [Module][Full DTensor] Config-based sharding infrastructure with llama3 adoption | 配置化的 sharding plan |
| #2898 | Refactor checkpoint system with StateDictTransforms and converter protocol | DCP/HF 互通 |
| #2983 / #3001 / #3000 | RL 系列（compiled loss / load-balancing aux loss） | RL 训练成熟化 |
| #2783 | [RL] Add support for Compiled loss | 编译 × RL |
| #2949 | Add multi-source checkpoint loading with additional_load_path | SFT/继续训练场景 |
| #2679 | [WIP] Add MoE support to the transformers modeling backend | HF 路径补 MoE |

### 5.2 活跃 Open Issues / Discussions

| 主题 | 来源 | 摘要 |
|---|---|---|
| Expert Parallel Fast Path / Router dtype（来自 solar-open 102B） | #2225 | 工业界实战反馈 EP fast path |
| DeepSeek-V3 671B 基准 + profile | #1558 | 大规模 DSv3 性能数据需求很高 |
| `[DeepEP] shared_experts cannot overlap with deepep.combine()` | #2298 | DeepEP 核心 overlap 痛点（虽然 #2310 已实现一部分） |
| FLUX 加 PP | #1264 | 多模态扩散模型 PP 仍是缺口 |
| 加入 MLPerf Llama 3 8B | #2046 | benchmark 标准化诉求 |
| PP × MoE aux_loss | #1979 | PP+MoE 数学一致性问题 |
| SFT mask 怎么加 | #1781 | SFT 仍然在补能力 |
| Duplicated AG/RS in SimpleFSDP | #2133 | 编译路线下的通信冗余 |
| Discussions: MFU calc for quantized workloads | Apr 8 | MXFP8 后 MFU 口径 |
| Discussions: nn.Identity usage | Apr 9 | 配置 / param_init 的 corner case |
| Discussions: support MXFP8 All-gather | Dec 17 | 期待 FSDP2 fp8 all-gather |
| Discussions: AMP for pure TP | Jan 29 | TP-only 训练的混合精度问题 |
| Discussions: Early stopping | Feb 4 | 训练流程功能性诉求 |

### 5.3 release 节奏

- **0.1.0**：2025-06-18（首个 pre-release）
- **0.2.0**：2025-10-18（参考点附近，pytorch 2.10 nightly）
- **0.3.0**：2025-12-26（pytorch 2.11 nightly，DeepEP / GPT-OSS / Compiler Toolkit / Autoparallel / RL 全部入场）
- **0.4.0**：2026-02-20（pytorch 2.12 nightly，CP refactor / FlexCP / mxfp8 on gfx950 / Verl 集成 / ROCm 全面 CI）

### 5.4 Vendor / 硬件趋势

- **AMD**：`[2025/11] AMD released an optimized fork of torchtitan for AMD GPUs`，已正式写入 README News；ROCm CI 已扩展至 H100 测试镜像（#2202）、SimpleFSDP（#2220）、AutoParallel/CompilerToolkit（#2248）、Transformers/VLM（#2276）；MI355X (gfx950) MXFP8（#2222）；ROCm Pt 7.1 nightly（#2491）；
- **NVIDIA Blackwell**：MXFP8（B200/GB200，28% 加速 blog），HybridEP（GB200 NVL72，NVL72 SM 配置可调）；
- **Intel XPU**：Discussion 中讨论 MXLinear on XPU；
- **H100**：DeepEP 是主战场；
- **MI300X / MI355X**：AMD-AGI fork 是社区关注重点。

---

## 6. 关键迁移指南（如果你 6 个月前 fork 过）

### 6.1 必改动作（按优先级）

1. **替换启动命令**：`CONFIG_FILE=... ./run_train.sh` → `MODULE=<模型目录名> CONFIG=<config_registry函数名> ./run_train.sh`。
2. **删除 `train_configs/*.toml`，新建 `config_registry.py`**：把每个 toml 翻译成返回 `Trainer.Config` 的 Python 函数。可参考 `torchtitan/models/llama3/config_registry.py`。
3. **模型代码搬家**：
   - `models/<x>/model/model.py` → `models/<x>/model.py`
   - `models/<x>/model/args.py` → 已合并到 `model.py` / `__init__.py`
   - `models/<x>/infra/parallelize.py` → `models/<x>/parallelize.py`
   - `models/<x>/infra/pipeline.py` → 删除，统一用 `torchtitan/distributed/pipeline_parallel.py:pipeline_llm`
   - `models/<x>/model/state_dict_adapter.py` → `models/<x>/state_dict_adapter.py`
4. **`__init__.py` 提供 `model_registry(flavor) -> ModelSpec`**（替代旧的 `register_train_spec`）。
5. **模型层继承 `Module` 基类**（`from torchtitan.protocols.module import Module`），实现 `Config` 嵌套类，使用 `param_init` 字典而不是构造函数里手动 `nn.init.*`。
6. **CommConfig**：`comm.mode=fake_backend` 是 CI 校验配置的最小负担方式，建议接入。
7. **EP 配置**：`expert_parallel_comm_backend` 选 `standard / deepep / hybridep`；老的 `expert_parallel_degree` 约束已经放宽，按文档检查实际 mesh 形状即可。
8. **FT/DiLoCo**：从 `from torchtitan.components.ft import FTManager` → 用 `torchtitan/experiments/ft/` 下独立 trainer，并在启动时加上 `TRAIN_FILE=torchtitan.experiments.ft.train`。

### 6.2 可选优化

- 升 PyTorch nightly 到 cu130（README 默认）；记得装 `torchdata` nightly。
- 启用 BF16 optimizer states：`OptimizersContainer.Config(implementation="fused_opt_states_bf16")`。
- B200 上启用 MXFP8 dense + MoE（参考 `docs/mxfp8.md`）。
- 用 `MODULE=transformers_modeling_backend` 直接训 HF 上现成 dense 模型。
- RL pipeline 走 `torchtitan/experiments/rl`（vLLM + Monarch + TorchStore + GRPO）。
- 编译路线走 `MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh`，体验 SimpleFSDP + AOT compile + precompile。

---

## 7. 架构演进总结（一图流文字版）

```
                         2025-10                                  2026-04
                ┌─────────────────────────┐         ┌────────────────────────────────────┐
入口            │ run_train.sh + TOML     │   →     │ MODULE=xx CONFIG=yy + Python       │
                │  CONFIG_FILE=...        │         │  Configurable + tyro CLI           │
                └─────────────────────────┘         └────────────────────────────────────┘

Trainer         train.py (778 行)              →    train.py (~80) + trainer.py (878)
                                                    + Trainer.Config 子类化扩展点

Mesh            手写 _build_mesh_with_ep        →    DeviceMesh._unflatten 三套子 mesh
                / _build_mesh_without_ep            (dataloading / dense / sparse)

Model layout    models/x/{model/args.py,        →    扁平 + config_registry.py
                  model/model.py,                    + Configurable Module 协议
                  infra/parallelize.py,              + Per-model state_dict_adapter
                  train_configs/*.toml}

Models          llama3/4, deepseek_v3,          →    + flux + gpt_oss (毕业)
                qwen3, qwen3_vl 雏形                + qwen3_vl 完整 (含 235B-A22B MoE)
                                                    + llama3 weight tying
                                                    Llama3-8B 仍是基线

MoE / EP        EP toml 静态约束                →    token_dispatcher 抽象
                                                    DeepEP (H100), HybridEP (GB200)
                                                    node-limited routing
                                                    fp32 router, per-layer compile

Quantization    Float8                          →    Float8 + MXFP8 (dense+MoE, B200/GB200)
                                                    + MXFP8 on gfx950
                                                    + BF16 optimizer states

Comm / Debug    NCCL                            →    + fake_backend / local_tensor / torchcomms

Experiments     flux, simple_fsdp, torchcomms,  →    graph_trainer, autoparallel, ft,
                moe_symm_mem_kernels, autop,         vlm, transformers_modeling_backend, rl,
                ft (in components), vlm, forge      forge

Vendor          NVIDIA H100 主, ROCm 起步       →    AMD 官方 fork, ROCm CI 全覆盖
                                                    Blackwell B200/GB200 一线
                                                    XPU 起步讨论
```

---

## 附录：参考链接

- 源码与对比 commit：
  - 旧：[`4db8f6e6` (2025-10-16)](https://github.com/pytorch/torchtitan/commit/4db8f6e6)
  - 新：[`70b044c4` (2026-04-21)](https://github.com/pytorch/torchtitan/commit/70b044c4)
- 关键 BC-breaking PR：
  - [#2386 Config System Refactor](https://github.com/pytorch/torchtitan/pull/2386)
  - [#1660 ParallelDims rewrite](https://github.com/pytorch/torchtitan/pull/1660)
  - [#2510 torchcomms 整合](https://github.com/pytorch/torchtitan/pull/2510)
  - [#2457 Merge simple_fsdp + compiler_toolkit](https://github.com/pytorch/torchtitan/pull/2457)
  - [#2003 Separate init_distributed_env](https://github.com/pytorch/torchtitan/pull/2003)
- Releases：
  - [v0.2.0 (2025-10-18)](https://github.com/pytorch/torchtitan/releases) · [v0.3.0 (2025-12-26)](https://github.com/pytorch/torchtitan/releases) · [v0.4.0 (2026-02-20)](https://github.com/pytorch/torchtitan/releases)
- Discussions：[Vote on new features](https://github.com/pytorch/torchtitan/discussions/694)
- 博客：Accelerating 2K+ Scale Pre-training up to 1.28x with TorchAO MXFP8 and TorchTitan on Crusoe B200 Cluster

