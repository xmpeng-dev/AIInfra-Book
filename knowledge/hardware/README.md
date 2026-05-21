# knowledge/hardware — GPU 硬件知识

## 文件索引

| 文件 | 内容 |
|---|---|
| [`gpu-comparison.md`](./gpu-comparison.md) | NVIDIA(A100/H100/H200/B200/GB200/L40S) vs AMD(MI250X/MI300X/MI325X/MI355X)横向对照 + MI355X vs B200 详细对比 |
| [`refs/PTA.pdf`](./refs/PTA.pdf) | 参考手册(二进制) |

## 什么时候来查

- 选型对比 / 报价表 → `gpu-comparison.md`
- 写 kernel 前确认硬件参数(HBM/带宽/MFMA 峰值) → `gpu-comparison.md` 的相应表格
- 想了解 CDNA4 微架构细节 → 配合 `.cursor/skills/mi355_hardware_aware/SKILL.md` 一起读

## 编辑约定

- 数字必须有出处(链接到厂商 datasheet 或 platform 页面)。
- 同一卡的不同封装(SXM / PCIe / OAM)分行列出,不要含糊带过。
- 不在这里放任何实验性数据 —— 那是 `notes/` 的事。
