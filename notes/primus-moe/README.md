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
| Last roadmap | [2026-05-29 H2 2026 draft](./2026-05-29_roadmap_h2_2026.md) (mirrors `NVIDIA/Megatron-LM#4815` format) |

## Index

| 日期 | 文件 | 内容 |
|---|---|---|
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
