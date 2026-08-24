# Primus 定位再思考：从集成层到边界消解层

> **When**: 2026-08-06 09:30 UTC+8
> **Where**: slab 知识库 / 战略判断，非实验记录
> **Context**: 对 [`2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md`](./2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md) 的修正。起因是对 Primus "多 backend 封装" 定位的不满：它是 AMD 训练入口，但架构上只是多后端封装，既没提升 Primus 影响力，对个人职业纵深加成也不够。

## TL;DR

三条结论：

1. **推翻 2026-05-08 的主线判断。** 把 `Primus/pilot`（agentic training tuning）当职业主线是错的——agent 能力是当前商品化最快的技能，且 Pilot 推了 15 个月仍卡在 BASELINE 非确定性。
2. **真正稀缺的不是 agent 能力也不是 kernel 能力，是两者的组合权限**：同时握有 kernel 级执行权（`mfma_tile.h` 99% BF16 峰值）和框架级调度权（Primus 第一贡献者，~173/586 commits ≈ 30%）。NVIDIA 在组织上跨不过 CUTLASS / cuBLAS / NCCL / Megatron 四个团队，做不到这件事。
3. **新主张：Primus 的深度不在"再抽象一层"，在"成为唯一有权把库与库之间的缝焊死的层"。** 性能不是丢在库里面，是丢在库之间的缝里；四条主要的缝都是组织边界造成的，而 Primus 正好坐在上面。

## Primus 现状（有据可查）

Primus 不是 fork，是 AMD 自研编排层。首次提交 2025-02-24，586 commits，一方代码约 129K Python LOC。三层生态：`Primus-SaFE`(平台) → `Primus-LM`(框架) → `Primus-Turbo`(算子)。

| 事实 | 位置 | 含义 |
|---|---|---|
| Megatron / TorchTitan 是未修改的 pinned submodule，全部适配靠运行时 monkey-patch | `primus/backends/megatron/patches/` **184 个 patch 文件 / 7,975 LOC**；TorchTitan 20 个 | "封装感"的技术根源 |
| 已存在 13,394 LOC 的解析式性能/显存模拟器 | `primus/core/projection/` | 被埋成 `docs/projection.md` 的边缘功能 |
| 已存在 3,737 LOC 调优 agent | `primus/agents/tuning_agent/` | 同上 |
| Megatron 落后上游约 403 commits，patch 需版本门控 | `backend_version_patterns` 机制 | 维护成本随上游速度线性增长 |
| 配置为无类型 `SimpleNamespace` + `deep_merge`，无 schema | `primus/core/config/` | 字段拼错静默流进 backend args |

一个细节最说明问题：`primus/backends/megatron/patches/moe_patches/topk_router_patches.py` 为替换一个 `TopKRouter`，用了**六种方式**打同一个补丁（patch `sys.modules`、patch `moe_layer.TopKRouter`、改 dataclass 字段默认值、包 `get_moe_module_spec_for_backend`、包 `MoELayer.__init__`、再 patch 废弃模块）。这不是代码质量问题，是**架构没有所有权**的必然结果：Primus 不拥有 parallel state，不拥有训练循环，只能从外面往里够。

## 三个前提修正

### 前提 1：「入口」是滩头堡，不是终点

集成层结构上就是低信用位置——模型架构别人定，kernel 别人写，硬件别人设计，你负责粘合。高投入、低署名、可替代。

关键在于：**把集成层做得更聪明并不能改变这一点**。更聪明的 wrapper 还是 wrapper。出路是拿入口当滩头堡往上游走，占住一个做决策的位置。

### 前提 2：职业问题和 Primus 影响力问题不是同一个问题

两者成本、周期、受众都不同，不该用一个 artifact 同时解决：

| 问题 | 性质 | 解法 | 成本 |
|---|---|---|---|
| Primus 影响力不足 | 组织 / 生态问题 | 上游化、标准、公开证据 | 低 |
| 个人技术纵深不足 | 技术问题 | 占住一个别人做不了的品类 | 高 |

### 前提 3：对自身稀缺性的判断是错的

2026-05-08 那篇把选择框成 "agent 能力 vs kernel 能力"，押 agent 因为它是趋势。但 agent 技能是当前商品化最快的技能，是 2026 年最不该押注的方向。单纯 kernel 能力也不算稀缺，AMD 内部还有别的 kernel 高手。

稀缺的是**组合权限**：kernel 级执行权 + 框架级调度权同时在一个人手上。

## 主张：Primus 的深度 = 消解边界

这个组合唯一解锁的能力是**删掉库与库之间的缝**。这个论点其实已经在 `monolith-moe/2026-04-14_rccl_overlap_analysis.md` 里被自己发现了，只是当时没把它当论点：RCCL 多流**架构上不可能** overlap——不是实现 bug，是集合通信 API 的语义（collective 是原子的、全 rank 参与的操作）从定义上禁止了目标行为。

### MoE 训练今天的四条缝

| 缝 | 两侧 | 损失 | 证据 |
|---|---|---|---|
| collective ↔ compute | RCCL ↔ hipBLASLt / CK | A2A 完全暴露；dispatch + combine 占 MoE 层 40–60% | `notes/moe_perf/turbo/README.md` |
| operator ↔ operator | Primus-Turbo 算子之间 | 中间 buffer 往返 HBM | MegaMoeFlydsl L2Y round-trip 938 MB |
| framework ↔ library | Megatron 调度 ↔ 算子实现 | 框架不知道算子的时间结构，无法在算子内部插通信 | 184 个 monkey-patch |
| quant ↔ GEMM | 量化 kernel ↔ GEMM | transpose / colwise requant 带宽受限 ~35% HBM，把 FP8 的 2× 稀释到 1.15–1.29× | `notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md` |

**每一条缝都是组织边界造成的，不是技术边界。** 184 个补丁的真正含义不是技术债，是"想焊缝但手不够长"的物理证据。

### 为什么 AMD 结构性有利

- 8 卡 XGMI 全互联 + 大 HBM + chiplet，比 NVLink + NVSwitch 更适合 device-side 直接寻址（HIP IPC 工作已验证）。
- NVIDIA 组织上跨不过 CUTLASS / cuBLAS / NCCL / Megatron 四个团队、四个发布节奏。
- 学术空白：2026-07-07 的 paper scan 结论是"没有一篇针对 AMD CDNA"的 in-kernel XGMI overlap 工作。

### 关键升级：从 kernel 变成 execution model

三代 super-kernel（`monolith-moe` / `rocmoe` / `MegaMoeFlydsl`）没长成东西，根因是被当作 **kernel** 做。kernel 是点解——512 t/g 赢，2048 t/g 就输（见 `notes/monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md`）。

**execution model 是一份契约**：规定通信与计算如何交错，框架调度它，算子实现它，再加一个决定何时启用的 predicate。

叙事从"我写了个快的融合 kernel"升级为"我定义了 AMD 训练上通信与计算融合的执行方式"。

这个升级同时把成本模型（`primus/core/projection/`）从"主张"降级为"判据"——正好是自己的数据所要求的：融合路径只在某些 regime 赢，所以必须有 predicate 判断。

### 在 Primus 里的具体形态（不是重写）

1. **Primus-Turbo**：一族带明确 overlap 契约的可组合原语——peer-pull dispatch、in-kernel A2A、tile-level ready flag、persistent worker queue。这些在 `.cursor/skills/cco-pipeline-overlap/SKILL.md` 里已经编码好了。
2. **Primus-LM**：一个能感知该契约的调度层。这是唯一需要真架构工作的地方，也正是能消灭一批 monkey-patch 的地方。
3. **Predicate**：一个小而明确的成本判据，决定何时走融合路径。复用 `primus/core/projection/`，但范围收窄、用途明确。

## 双轨落地

### 轨道 A —— 技术纵深（贵，12–18 个月）

上述边界消解线。优势：建在最强能力上；不依赖 Pilot（阻塞 15 个月）；不依赖多节点集群（目前没有）；已有三代原型与真实数字（MonolithEP 4.82 ms / 1.76×、RocMoE dispatch 2.23×、MegaMoeFlydsl FP8 1.36×）。

### 轨道 B —— 影响力（便宜，3 个月，并行）

| 动作 | 理由 | 现状 |
|---|---|---|
| **上游化** | 184 个补丁里挑可上游的推 Megatron / TorchTitan。每个合并的 PR = 永久、可引用、署名的影响力，顺便减债 | 完全为零，无 OSS PR 记录 |
| **公开证据** | AMD 的商业问题不是"框架像 wrapper"，是没人相信 AMD 能训前沿模型，因为缺公开证据。连续、可复现、带 trace 的公开性能记录成本极低 | 只有零散 ROCm blog |
| **恢复 weekly** | 最便宜的复利 | 2026-04 后断更 |

## 认真考虑过但排在后面的两个角度

| 角度 | 吸引力 | 为什么排后 |
|---|---|---|
| **RL / post-training 抢滩** | Primus 定位里写了 RL 但零代码；品类爆发中、AMD 无人占；用得上大 HBM 优势 | inference 侧积累为零（无 vLLM / SGLang / KV-cache / paged attention 笔记），等于放弃现有全部积累从头开始。天花板可能更高，但换赛道成本极大 |
| **硬件协同设计**（用 Primus 数据影响 MI400 / MI450） | 天生 Fellow-track；数据唯一 | 对外不可见（NDA）；反馈周期 2–3 年；更像是轨道 A 做成之后自然获得的权力，而非可直接立项的事 |

## 什么会证伪这个主张

- **融合路径在训练规模上再次失败。** 已经发生过一次（512 t/g 赢、2048 t/g 输）。缓解：接受"只在某些 regime 赢"，但必须让 predicate 判得准；判不准则整条线不成立。
- **测量不可信。** MI355X sclk 抖 ±30%，`rocm-smi --setperflevel high` 不支持（见 `notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md`），意味着任何 <10% 的结论都是噪声。这也是 `notes/pilot/` BASELINE 非确定性的同一个根因。**这是轨道 A 开工前必须先解决的前置项。**
- **组织不认可。** 若管理层只要下季度 MLPerf 数字，需要用轨道 B 的短期产出换轨道 A 的时间。

## 职业定位修正

从：

> Agentic AI Infrastructure / Agentic Training Systems Engineer with deep GPU performance expertise.

改为：

> The person who defined how communication and computation fuse on AMD training hardware.

前者绑在一个正在快速贬值的技能标签上；后者绑在硬件结构上，不会贬值。

## 相关文件

- [`2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md`](./2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md) — 被本篇修正的前一版判断
- `notes/primus-moe/2026-05-29_roadmap_h2_2026.md` — H2 2026 Primus MoE 路线图，含 super-kernel 产品化条目
- `notes/monolith-moe/2026-04-14_rccl_overlap_analysis.md` — RCCL 语义上不可 overlap 的原始发现
- `notes/monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md` — 融合路径在训练规模上失败的证据
- `notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md` — FP8 收益被量化开销稀释；sclk 噪声
- `notes/moe_perf/turbo/README.md` — dispatch + combine 占 MoE 层 40–60%
- `.cursor/skills/cco-pipeline-overlap/SKILL.md` — 已沉淀的 overlap 原语
- `knowledge/moe/research-overview.md` — 六个可发表研究方向（A–F）
- `primus/core/projection/`、`primus/agents/tuning_agent/`、`primus/backends/megatron/patches/`
