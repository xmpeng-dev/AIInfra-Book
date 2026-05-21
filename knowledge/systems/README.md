# knowledge/systems — 训练系统/框架知识

## 文件索引

| 文件 | 内容 |
|---|---|
| [`torchtitan-diff-2025-10-vs-2026-04.md`](./torchtitan-diff-2025-10-vs-2026-04.md) | TorchTitan 半年跨度的 diff 总结(API / 训练循环 / 并行 / 优化器变更) |
| [`training-1024g-stability-interview-notes.md`](./training-1024g-stability-interview-notes.md) | 1024-GPU 训练稳定性面试笔记(checkpoint / failover / 监控 / RCCL 调优) |

## 什么时候来查

- 准备升级 TorchTitan / 排查跨版本回归 → 看 diff
- 设计大规模训练的容错与监控 → 看 stability 笔记
- 比较 Primus / TorchTitan / Megatron 三个 backend 的工程差异 → 配合 `.cursor/skills/backend-gap-report/SKILL.md`

## TODO

- 增补 Primus 的设计要点(目前散落在 `notes/pilot/` 中)
- 增补 Megatron-Core MoE 的工程实现要点(已有 paper note `papers/megatron-core-moe.md`,但工程经验单独成文更好)
