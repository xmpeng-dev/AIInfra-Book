# knowledge/hardware — GPU 硬件知识

## 文件索引

| 文件 | 内容 |
|---|---|
| [`gpu-comparison.md`](./gpu-comparison.md) | NVIDIA(A100/H100/H200/B200/GB200/L40S) vs AMD(MI250X/MI300X/MI325X/MI355X)横向对照 + MI355X vs B200 详细对比 |
| [`refs/PTA.pdf`](./refs/PTA.pdf) | 参考手册(二进制) |

## 什么时候来查

- 选型对比 / 报价表 → `gpu-comparison.md`
- 写 kernel 前确认 chip 级参数(HBM 容量 / 带宽 / 板级算力对比) → `gpu-comparison.md`
- 算子开发要的微架构细节 (XCD / NPS / MFMA 指令 / VGPR-AGPR / LDS / async direct-to-LDS / Triton-AMD 调参 / HIP intrinsic ↔ ISA) → `.cursor/skills/mi355_hardware_aware/SKILL.md` 是入口

## 编辑约定

- 数字必须有出处(链接到厂商 datasheet 或 platform 页面)。
- 同一卡的不同封装(SXM / PCIe / OAM)分行列出,不要含糊带过。
- 不在这里放任何实验性数据 —— 那是 `notes/` 的事。
