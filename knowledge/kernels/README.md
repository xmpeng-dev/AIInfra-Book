# knowledge/kernels — Kernel 优化 know-how

## 文件索引

| 文件 | 内容 |
|---|---|
| [`memory-access-patterns.md`](./memory-access-patterns.md) | 数据搬运的 5 问 review checklist (跨 row 读连续? 跨 row 写连续? Register 是否同时背 read+write? Wave 锁步还是独立? 数据流方向跟硬件匹配吗?) — 写或 review 任何搬数据 kernel 时默认先过这张表 |
| [`fp8-expert-gemm.md`](./fp8-expert-gemm.md) | FP8 Expert GEMM 优化总结(grouped GEMM、scale 选择、AMD/NV 路径差异) |

## 什么时候来查

- 写或 review **任何在 GPU 上搬数据的 kernel** (dispatch / combine / permute / scatter / staging) → `memory-access-patterns.md` 必读 (默认 review checklist)
- 写或优化 GEMM / GroupedGEMM kernel → 这里 + `.cursor/skills/amd-gemm-optimization/SKILL.md` 或 `cuda_gemm_optimization/SKILL.md`
- 计算-通信重叠(in-kernel overlap)→ 配合 `.cursor/skills/cco-pipeline-overlap/SKILL.md` (`memory-access-patterns.md` Q3/Q4/Q5 是它的前置)
- AMD MI355X 上的 MFMA / LDS 细节 → 配合 `.cursor/skills/mi355_hardware_aware/SKILL.md`

## TODO

- 增补 cco-overlap 的核心要点摘录(目前只在 skill 里)
- ~~增补 LDS swizzle / async-direct-to-LDS / wave specialization 的速查表~~ (2026-05-21 由 `memory-access-patterns.md` 覆盖 async-direct-to-LDS 和 wave 锁步/独立, LDS swizzle 仍可独立成篇)
