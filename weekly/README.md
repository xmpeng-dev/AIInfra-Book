# weekly — 跨项目周报

每周一篇,聚合当周在所有项目(`monolith-moe` / `gpt-oss` / `mlperf-llama` / …)上的进展。
详细 note 在各自项目目录 `../notes/<project>/` 下。

## 目录约定

```
weekly/
├── README.md
└── <year>/
    └── YYYY-MM-DD_weekly_report_<MMDD>_<MMDD>[_EN].md
```

按年份分桶,文件名沿用既有约定: 周末日期 + 起止月日 + 可选语言后缀。

## 周报清单

| 周期 | 主题 | 状态摘要 | 文件 |
|---|---|---|---|
| 2026-04-09 → 2026-04-14 | MoE CCO 优化 (MI355X) | Hand-written GG vs CK +32%/+46%; RCCL/rocSHMEM/Triton sys-atomic 三条路全否; 选定 HIP C++ IPC 路径 | [`2026/2026-04-14_weekly_report_0409_0414.md`](./2026/2026-04-14_weekly_report_0409_0414.md) |
| 2026-04-19 → 2026-04-22 | GPT-OSS-20B MLPerf 调优 (MI355X) | E2E 11916 → **9963 s** (−16.4%, val 3.3247 ✓); 半自动 ablation loop 跑通; mbs=4 路径反直觉发现两条 | [`2026/2026-04-22_weekly_report_0419_0422.md`](./2026/2026-04-22_weekly_report_0419_0422.md) (中文) · [`2026/2026-04-22_weekly_report_0419_0422_EN.md`](./2026/2026-04-22_weekly_report_0419_0422_EN.md) (English) |

## 各项目当前状态

> 链向各项目 README 的"状态"节,避免本目录重复维护。

- [`../notes/gpt-oss/`](../notes/gpt-oss/README.md) — best 9963 s, V1/V2 fused residual+RMSNorm 待 800-iter 终判
- [`../notes/monolith-moe/`](../notes/monolith-moe/README.md) — Layout C + GEMM 完成, HIP C++ IPC kernel 待落地
- [`../notes/mlperf-llama/`](../notes/mlperf-llama/README.md) — Primus vs NeMo gap 定位完成, DataLoader spawn fix + `fp8_param` A/B 待跑

## 维护约定

- 每周写一篇,覆盖周一到周日,放到当年份的子目录。
- **周报里不写新结论**——只汇总当周各项目的 progress / archive notes,加跨项目排序。
- 写完后更新本 README 的清单和"各项目当前状态"链接。
