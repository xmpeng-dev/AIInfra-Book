# primus-moe — Primus MoE roadmap & planning notes

> Strategic / planning artifacts for the **Primus MoE story** (Primus-LM + Primus-Turbo).
> Hardware-line research notes live under sibling folders: `slab/notes/rocmoe/`, `slab/notes/monolith-moe/`, `slab/notes/monolith-ep/`.

## Status

| 维度 | 值 |
|---|---|
| Owner | AMD AI Brain — Training at Scale (TAS) team |
| Hardware focus | MI300X / MI325X (gfx942), MI350X / MI355X (gfx950) |
| Backends | Megatron-LM, TorchTitan, JAX MaxText |
| Companion repos | `~/workspace/Primus/`, `~/workspace/Primus-Turbo/`, `~/workspace/Primus-dev/` |
| Last roadmap | [2026-09-03 全框架级 2026 Q4 – 2027 H2](./2026-09-03_primus-roadmap-2026q4-2027h2.md) — MoE 细项仍以 [2026-05-29 H2 2026 draft](./2026-05-29_roadmap_h2_2026.md) (mirrors `NVIDIA/Megatron-LM#4815` format) 为准 |

## Index

| 日期 | 文件 | 内容 |
|---|---|---|
| 2026-09-03 | [`2026-09-03_primus-roadmap-2026q4-2027h2.md`](./2026-09-03_primus-roadmap-2026q4-2027h2.md) | **全框架级 roadmap（2026 Q4 – 2027 H2）** — 四路调研合成（slab / 上游 TorchTitan / 上游 Megatron-LM / Primus 现状）。核心判断：①上游欠债是最大风险（Megatron 落后 1391 commits、core 0.16→0.20；TorchTitan 落后 937 commits；156 patch 仅 1 个版本门；`GPTModel` 已废弃、`megatron/legacy/` 已删）②两个上游官方扩展点刚就位（Megatron `BackendSpecProvider`、TorchTitan `overrides/`）是时间敏感窗口 ③护城河经实证（上游 Megatron core 内 ROCm 命中 1 / `torch.cuda` 796；TorchTitan 已停 ROCm CI 并删 golden；zero-bubble 上游 0 命中）。分期 C0 可信度 → C1 插件化 → C2 存量性能 → C3 规模化可靠性 → C4 执行层收拢，附 13 条反证护栏 |
| 2026-08-04 | [`2026-08-04_1049_primus-runtime-design-and-plan.md`](./2026-08-04_1049_primus-runtime-design-and-plan.md) | Primus-Runtime 设计与实施规划 — core/pipeline_parallel 已是 plan/execute 内核且双消费者验证；现存三套并行调度栈、两套 IR（`R` 同名不同义）；P0 收敛 → P1 comm → P2 精度 → P3 HIP graph → P4 对外化 |
| 2026-08-04 | [`2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md`](./2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md) | 框架战略备忘 — patch 只占 megatron 后端代码 17%，已有约 3.3 万行执行层资产散落在 3 个后端命名空间；反对自研 Megatron 替代品，建议归位成独立执行层 Primus-Runtime（R1–R6 六模块 + C0–C3 分期验收） |
| 2026-07-22 | [`2026-07-22_moe_10k_gpu_gap_analysis.md`](./2026-07-22_moe_10k_gpu_gap_analysis.md) | MoE @ 万卡:瓶颈 × Primus 现状 gap 分析 — 把 10k 卡瓶颈逐条对到 DeepEP/1F1B overlap/分层 recompute/projection,识别容错·动态负载均衡·拓扑感知通信三大空白 |
| 2026-05-29 | [`2026-05-29_roadmap_h2_2026.md`](./2026-05-29_roadmap_h2_2026.md) | Primus MoE Roadmap H2 2026 (Q3 + Q4) — full mirror of the Megatron Mcore-MoE Q2 2026 roadmap, AMD-flavored |

## Cross-references

- **Production line**: `slab/notes/monolith-moe/` (4.82 ms / 8× MI355X, DSv3 e2e + loss parity)
- **Research line (super-kernel)**: `slab/notes/rocmoe/` (Layout-P + receiver-pull, BF16 target ≤ 7 ms)
- **Research directions**: `slab/knowledge/moe/research-overview.md` (comm-aware routing, AMD-FSEP, MoE-IR, backward comm scheduling)
- **Megatron MoE comparison reference**: [`NVIDIA/Megatron-LM#4815`](https://github.com/NVIDIA/Megatron-LM/issues/4815)

## Maintenance

- One roadmap doc per release cycle (H1 / H2). Old roadmaps stay in place; the latest is linked from this README.
- Update `Status` row + `Index` table whenever a new roadmap iteration lands.
- Cross-link to the per-feature progress notes in `monolith-moe/` and `rocmoe/` rather than duplicating numbers here.
