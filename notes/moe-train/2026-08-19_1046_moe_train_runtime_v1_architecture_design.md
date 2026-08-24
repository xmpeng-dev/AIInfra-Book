# MoE Train Runtime v1 架构设计：PithTrain DNA + topology-aware 重写

> **When**: 2026-08-19 10:46 UTC+8
> **Where**: 设计阶段，尚未落码；首要目标平台 8× MI355X (gfx950) + SLURM 多节点
> **Context**: 通读 [`3rd/pith-train`](../../3rd/pith-train/)（CMU MLC PithTrain）后的架构取舍；不 fork 全抄，借 compact + schedule-aware 思路重写一条 **AMD-first、可 scale-out** 的 MoE 预训练 runtime
> **参考对话**: PithTrain 设计分析、DualPipeV/5-stage/FP8 注入、AMD comm_stream 问题、大规模集群瓶颈、架构级重写方向

## TL;DR

新建 **MoE Train Runtime**（工作名 `moe-train`，待定），继承 PithTrain 的四条 DNA——**schedule-aware layer protocol、V 形 PP 调度器、Config→Context 注入、canonical checkpoint**——同时做四处架构级替换：

1. **MoE 收成 `MoEExecutionUnit`**：dispatch / GEMM / combine 不再散落在 `execution.py` + RCCL + stream，对外仍暴露 5 个 schedule slot，对内 model 只写 3 个 macro block（attn / moe / post-moe）。
2. **`BackendBundle` 替代隐式 `training.Linear`**：setup 时绑定 Gemm / Attn / MoEComm backend（NVIDIA DeepGEMM、AMD hipBLASLt/Primus-Turbo），无 runtime registry。
3. **`OverlapPolicy` 替代 `comm_stream`**：`SEQUENTIAL | INTERLEAVED | FUSED_MOE`，硬件与 topology 决定策略，不把 stream 当 overlap 抽象。
4. **`ParallelSpec` topology-aware**：EP/CP 默认 **intra-node**，mesh builder 读 XGMI/NVLink domain，避免万卡 EP all-to-all 踩坑。

第一版验收：**单节点 MI355X BF16 Qwen3-30B-A3B loss 曲线与 PithTrain reference 对齐**；第二版再加 FUSED_MOE overlap 与 FP8 backend swap。

---

## 1. Background

### 1.1 为什么要重写而不是 patch PithTrain

PithTrain（~11K 行 Python）的价值在于 **agent-readable 的结构性取舍**，不是某几个 kernel 的峰值：

| 优点 | 局限（架构层，非性能 tuning） |
|---|---|
| 5-stage 切分 ↔ DualPipeV F/B 交错天然对齐 | `comm_stream` 把 overlap **实现** 当成 **抽象**；AMD 上 CU 互抢 |
| `LayerProtocol` 让 model 不为 pipeline 写 generic forward | stage2/4 逻辑泄漏到 model 周边；MoE 路径跨 4+ 文件 |
| `training.Linear/GroupedLinear` 注入，model FP8-agnostic | 仅覆盖 GEMM；Attention/MoE comm 仍硬绑 NVIDIA |
| mesh `(PP,DP,CP,EP)` 内层 CP/EP = 假设 NVLink 域 | 跨节点 EP 无 first-class 约束 |
| 同步 DCP checkpoint + fail-fast | 大规模长跑运维成本高 |
| 无 TP、无 activation checkpointing | 大模型只能堆 PP |

目标：**保留「小、可读、schedule-first」**，补齐 **multi-backend、topology、MoE 执行单元、scale-out 子系统**——代码量可控在 ~15K 行，不滑向 Megatron 式 plugin 沼泽。

### 1.2 与现有 AMD 资产的关系

| 资产 | 角色 |
|---|---|
| [`3rd/pith-train`](../../3rd/pith-train/) | **Schedule 与 protocol 参考实现**；DualPipeV 8-step 行为 parity 门禁 |
| `notes/rocmoe/` | FUSED_MOE / super-kernel overlap 思路；`MoEExecutionUnit` 的 AMD 实现候选 |
| `notes/monolith-moe/`、`notes/MegaMoeFlydsl/` | dispatch+GEMM+combine 分段 profile 与数值对拍基准 |
| `notes/primus-moe/` | 执行层 vs patch 层战略；不与 Primus-Megatron patch 重复造轮子 |
| `knowledge/kernels/memory-access-patterns.md` | MoE comm 五问 checklist |
| `.cursor/skills/cco-pipeline-overlap/` | FUSED_MOE policy 的实现 SOP |

**定位**：独立 compact runtime，**不是** Primus 第四后端，也 **不是** PithTrain fork。Primus 继续吃 Megatron 生态；本 runtime 吃「从零可读 + MI355X 原生 MoE overlap」。

---

## 2. 继承的 DNA（不可妥协）

### 2.1 Schedule-aware `LayerProtocol`

Transformer layer 在 **EP 通信边界** 切 phase，使 pipeline 能把 **一个 micro-batch 的 compute** 与 **另一个的 comm** 交错。

对外（scheduler）仍映射为 **5 个 slot**（与 PithTrain / DualPipeV 兼容，便于 port schedule 与 validate-correctness）：

| Slot | 内容 | 执行 stream / 归属 |
|---|---|---|
| S1 | Attn + router | compute |
| S2 | dispatch | moe unit / comm |
| S3 | expert GEMM | moe unit / compute |
| S4 | combine | moe unit / comm |
| S5 | aggregate + residual | compute |

对内（model 作者）只实现 **3 个方法**（见 §3.2）。

### 2.2 `PipelineScheduler`（DualPipeV 等价物）

- V 形切分：`2 × pp_size` chunk，rank `r` 持 `(r, 2pp-1-r)`
- 8-step 主循环：`nF0 → nF0F1 → nB1W1F1 → nF0B1F1B0 → …`
- `WeightGradStore` zero-bubble W 阶段保留
- FSDP2：loop 内 suppress hook，loop 后 manual `post_backward`

### 2.3 Config → Setup → Context

```python
# 模式不变；context 模块扩展 backends
TrainCfg → setup_training() → contexts.{distributed, training, backends}
```

### 2.4 Canonical checkpoint

- 磁盘：global layer 名、per-expert 索引、与 PP/EP 无关
- 运行时：localized layout；DCP 存取
- **增强**：async `CheckpointCoordinator`（见 §3.6）

### 2.5 Operator 契约

每个性能算子：**kernel + reference + `torch.library.custom_op`**；`reference_forward` 保留为 correctness anchor。

### 2.6 Compact 原则

- 无 plugin registry / string dispatch
- 一 model 一文件，接受少量重复
- 可重复流程 ship 为 agent skill

---

## 3. 架构 redesign

### 3.1 四层 + 两横切面

```
Tasks/              pretrain_lm, convert_ckpt, tokenize
Runtime/
  scheduler/        PipelineScheduler (DualPipeV)
  mesh/             ParallelSpec, HardwareTopology, process groups
  execution/        ExecutionPlan: slot 驱动, OverlapPolicy
  loop/             train_step, optim, metrics hooks
Model/              protocol.py + qwen3_moe.py …
Backends/           BackendBundle + 各硬件实现
Operators/          Triton/HIP kernels
────────── 横切 ──────────
contexts/           distributed, training, backends
skills/             validate-correctness, add-model, …
```

比 PithTrain **只多三个明确边界**：`backends/`、`checkpoint/`（子模块）、`data/`（子模块）。

### 3.2 Model protocol：3 macro block × 5 schedule slot

```python
class LayerProtocol(Protocol):
    idx: int

    def forward_attn(self, hidden, rope, cu_seqlens=None) -> AttnOut:
        """S1: LN → Attn → LN → route; 返回 residual + routing metadata."""

    def forward_moe(self, dispatch_tokens, routing: RoutingInfo) -> Tensor:
        """S2+S3+S4: 委托 MoEExecutionUnit；model 文件不含 A2A 细节."""

    def forward_post_moe(self, moe_outs, routing, residual) -> Tensor:
        """S5: weighted sum + residual."""

    def reference_forward(self, hidden, rope) -> Tensor: ...
```

`AttnOut` / `RoutingInfo` 字段与 PithTrain `interface.py` 对齐，降低 port 成本。

**ExecutionPlan** 负责把 scheduler 的 `stage2_f … stage5_f` 调用链映射到上述三方法 + unit 内部 phase；F/B 交错表可 largely 从 PithTrain `overlap.py` 移植。

### 3.3 `BackendBundle`（setup-time 绑定）

```python
@dataclass
class BackendBundle:
    gemm: GemmBackend           # dense_linear, grouped_linear
    attn: AttnBackend           # flash / ring / mla / linear-attn
    moe: MoEExecutionUnit       # 见 §3.4
    overlap: OverlapPolicy

def setup_backends(cfg: TrainCfg, topo: HardwareTopology) -> BackendBundle:
    if cfg.platform == "rocm":
        gemm = RocmGemmBackend(primus_turbo=cfg.use_primus_turbo, fp8=cfg.fp8)
        attn = AiterAttnBackend(...)
        moe = RocmMoEUnit(gemm=gemm, overlap=cfg.overlap_policy, topo=topo)
    elif cfg.platform == "cuda":
        ...
    return BackendBundle(gemm, attn, moe, cfg.overlap_policy)
```

Model 构造：

```python
self.q_proj = backends.gemm.dense_linear(in, out, bias=...)
# forward_moe 内:
return backends.moe.execute(dispatch_tokens, routing)
```

**与 PithTrain `training.Linear = FP8Linear if fp8`** 同哲学，扩展到 attention + MoE comm；**仍然无 runtime 查表**。

### 3.4 `MoEExecutionUnit`

MoE 路径从 4 个文件收到 1 个 execution unit：

```
MoEExecutionUnit
  prepare(routing)       ← ep_dispatch dedup, splits
  dispatch(...)          ← comm phase (S2)
  compute(...)           ← grouped GEMM + act (S3)
  combine(...)           ← comm phase (S4)
  execute(...)           ← 按 OverlapPolicy 编排上述 phase
  reference(...)         ← 对拍 PithTrain prepare_dispatch + grouped_mm + a2a
```

| 实现 | Policy | 说明 |
|---|---|---|
| `SequentialMoEUnit` | SEQUENTIAL | phase 串行；M1 默认；AMD correctness baseline |
| `InterleavedMoEUnit` | INTERLEAVED | 保留 micro-batch 级 F/B 交错；comm 可不单独 stream |
| `FusedMoEUnit` | FUSED_MOE | chunk 级 dispatch+GEMM+combine；RocMoE/Monolith 路线 |

Unit 对外 **`execute()` 一次调用**；ExecutionPlan 若需 fine-grain F/B 交错，可调 unit 内部 phase hook（与 PithTrain stage2_f/stage3_f 等等价）。

### 3.5 `OverlapPolicy`

```python
class OverlapPolicy(Enum):
    SEQUENTIAL = "sequential"       # 默认 AMD / 跨节点 EP
    INTERLEAVED = "interleaved"     # PithTrain 式 F/B 交错，comm 不强制第二 stream
    FUSED_MOE = "fused_moe"         # super-kernel / chunk pipeline
```

**架构规则**：Policy 由 `(platform, ep_scope, cp_scope)` 在 `setup_backends` 时选定，**禁止** model 或 task 直接创建 `comm_stream`。

### 3.6 `ParallelSpec` + topology

```python
@dataclass
class ParallelSpec:
    pp: int
    dp: int  # inferred
    cp: int
    ep: int
    ep_scope: Literal["intra_node", "global"] = "intra_node"
    cp_scope: Literal["intra_node", "global"] = "intra_node"
    sharding: Literal["fsdp", "hsdp"] = "fsdp"
```

`HardwareTopology` 提供 `node_local_rank_groups`, `xgmi_clique`, `ib_clique`。

Mesh builder 规则：

1. `ep_scope=intra_node` → EP group 不得跨 node；`world_size` 不够则报错，不 silent 跨 IB A2A。
2. `cp_scope` 同理。
3. PP 可跨节点；DP 扩 scale-out。

预留 **TP 维**（v2）：`ParallelSpec.tp`，第一版不实现，mesh 命名留空位。

### 3.7 Compile 边界（架构声明）

| Region | 策略 |
|---|---|
| `forward_attn` | `@compile_region(fullgraph=True)` |
| `forward_moe` | **整段 opaque**：`backends.moe.execute` = 单一 custom_op |
| `forward_post_moe` | `@compile_region(fullgraph=True)` |

不再出现 PithTrain 式「S1/S5 compile、S3 裸奔」——MoE 要么全 opaque，要么 FUSED unit 内 kernel 化。

### 3.8 子系统：Checkpoint & Data

**CheckpointCoordinator**

- async save；rank stagger；hook `on_step_end`
- 仍用 DCP + canonical format（与 PithTrain 磁盘语义兼容）
- RNG / optimizer / scheduler 同目录布局

**ShardDataset**

- DP rank 读各自 shard；block shuffle（GPU `randperm` 仅当 index 可放入显存）
- PP rank 0 不再唯一 reader；activation 仍 PP P2P

### 3.9 可靠性（optional 外层）

- Core runtime **保持 fail-fast**（与 PithTrain 一致，debug 友好）
- `ElasticLauncher` 包装在 `tasks/`，非 core 依赖

---

## 4. 与 PithTrain 对照表

| 模块 | PithTrain | MoE Train Runtime v1 |
|---|---|---|
| PP 调度 | `dualpipe/dualpipev.py` | `runtime/scheduler/` — 行为 parity |
| F/B 交错 | `overlap.py` | `runtime/execution/plan.py` |
| Stage 执行 | `execution.py` + comm_stream | ExecutionPlan + MoEUnit + OverlapPolicy |
| Model API | 5 方法分散在 layer | 3 macro + unit 委托 |
| GEMM FP8 | `training.Linear` 注入 | `BackendBundle.gemm` |
| MoE comm | `all_to_all.py` + stream | `MoEExecutionUnit` |
| Mesh | 固定 `(PP,DP,CP,EP)` | `ParallelSpec` + topology |
| Ckpt | 同步 `save_checkpoint` | Coordinator async |
| Data | memmap + GPU shuffle | ShardDataset + block shuffle |

---

## 5. 目录结构（目标 ~15K LOC）

```
moe_train/
├── tasks/
├── runtime/
│   ├── scheduler/      # PipelineScheduler, WeightGradStore, FP8WeightCache
│   ├── mesh/           # ParallelSpec, topology, setup_distributed
│   ├── execution/      # ExecutionPlan, slot records
│   └── loop/           # train_step, clip_grad, metrics
├── model/
│   ├── protocol.py
│   └── qwen3_moe.py
├── backends/
│   ├── bundle.py
│   ├── gemm/           # rocm.py, cuda_deepgemm.py
│   ├── attn/           # aiter.py, flash.py, ring.py
│   └── moe/            # sequential.py, fused_rocm.py
├── operators/
├── checkpoint/
├── data/
└── contexts/
```

---

## 6. 分阶段里程碑

| 阶段 | 交付 | 验收 |
|---|---|---|
| **M0** | Protocol + Mesh(intra-node EP) + SequentialMoEUnit + BF16 RocmGemm | 单 node EP=8 pp=1 loss 降 |
| **M1** | PipelineScheduler port + ExecutionPlan | `validate-correctness` vs PithTrain 同 mesh loss 曲线 |
| **M2** | BackendBundle FP8 swap | 同模型 FP8/ BF16 parity（DeepGEMM 或 hipBLASLt） |
| **M3** | FusedMoEUnit (MI355X) | rocprof：MoE 段 MFMA util ↑ vs Sequential |
| **M4** | Async ckpt + ShardDataset | 256+ GPU 长跑无 shuffle OOM；ckpt 不阻塞整 step |
| **M5** | 多节点 pp>1 + ep intra-node | 8n8g benchmark 对比 PithTrain 同配置 tokens/s |

每阶段 **必须** 过 `reference_forward` + micro-batch loss parity；不追求 M0 性能。

---

## 7. 明确不做（第一版）

- Megatron 式 plugin registry / YAML spec 解析
- 全模型族覆盖（先 Qwen3-MoE + DeepSeek-V2-Lite）
- In-run elastic fault tolerance（仅文档化 fail-fast）
- TP（mesh 预留，不实现）
- 替代 Primus 作为 Megatron 后端

---

## 8. 开放问题

| # | 问题 | 倾向 |
|---|---|---|
| Q1 | 工作 repo 名 `moe-train` vs 并入 `RocMoE-v3` | 独立 repo 保持 compact；kernel 从 rocmoe 复用 |
| Q2 | FUSED_MOE v1 是否直接复用 MonolithEP push 还是全新 unit | 先 Sequential parity，再 port 已有 super-kernel 为 FusedMoEUnit |
| Q3 | DualPipeV 8-step 是否原样 port 还是简化 | **原样 port**；schedule 是 PithTrain 已验证资产 |
| Q4 | Primus-Turbo 作为唯一 Rocm GEMM 还是 CK 直连 | Primus-Turbo 薄壳优先，符合 slab library 策略 |
| Q5 | Agent skills 放 `.agents/skills` 还是 slab `.cursor/skills` | 代码 repo 内 `.agents/skills`；交叉引用 slab validate SOP |

---

## 9. Next

1. 在目标 repo  stub `model/protocol.py` + `backends/bundle.py` + `contexts/`（无 kernel，接口 only）。
2. 从 PithTrain 移植 `interface.py` 字段定义 + `overlap.py` 调用图，标注 → 新 ExecutionPlan 映射。
3. M0：Rocm `SequentialMoEUnit` + AITER attention + Primus-Turbo grouped GEMM，单 node Qwen3-30B-A3B。
4. 写 agent skill `validate-correctness-moe-train`（fork PithTrain skill，换 launch 入口）。

---

## 10. 参考

- PithTrain 源码：[`3rd/pith-train/`](../../3rd/pith-train/)
- PithTrain 架构文档：[`3rd/pith-train/docs/architecture.md`](../../3rd/pith-train/docs/architecture.md)
- MoE comm 五问：[`knowledge/kernels/memory-access-patterns.md`](../../knowledge/kernels/memory-access-patterns.md)
- CCO overlap SOP： [`.cursor/skills/cco-pipeline-overlap/SKILL.md`](../../.cursor/skills/cco-pipeline-overlap/SKILL.md)
- Primus 执行层战略：[`notes/primus-moe/2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md`](../primus-moe/2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md)
