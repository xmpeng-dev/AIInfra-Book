# career-strategy — 长期技术定位与职业路线

> **目标**：把当前的训练系统、GPU 性能优化、分布式训练经验，转化成有长期壁垒的职业主线。
> **核心定位**（2026-08-06 立，2026-08-12 15:05 定落地形态）：定义 AMD 训练硬件上通信与计算的融合方式——kernel 级执行权 + 框架级调度权的组合。落地形态是**用融合 kernel 删掉跨 op 的格式契约**，不是加一层派发。

## 状态

| 维度 | 判断（2026-08-12 15:05） |
|---|---|
| 主线方向 | **边界消解**：让 Primus 从集成层变成唯一有权焊死库间缝隙的层 |
| **落地形态** | **删掉量化↔GEMM 的格式缝**：MXFP8 反向里 GEMM 本体已拿到 1.84–1.96×，净收益被 colwise requant 稀释到 1.15–1.29×；剩余开销不是 kernel 不够好，是格式契约横跨三个 op，无人有权删 |
| **已作废的落地形态** | ~~占判定层（dispatcher + tuned CSV）~~ —— 是更聪明的 wrapper，且绑死 Turbo，离开 AMD 归零；与 8-06「集成层是低信用位置」自相矛盾 |
| **靶子的选法** | 用自测 dense GEMM 当尺子（`mfma_tile.h` bf16 1290 TF → MXFP8 上限 ~2580 TF）：dW1 ragged wgrad 已达 **85%**，bf16 腿也 85% → **GEMM 本体接近做完，剩 <10% 且 sclk ±30% 下不可验证** |
| **站位（已核）** | Primus-LM **173/646 = 26.8% #1**；Primus-Turbo **96/441 = 21.8% #2**；Turbo `benchmark/` **44/103 = 42.7% #1**。**同时在算子层与框架层前二的人只有一个** → 8-06 的"组合权限"已在手，缺的是 claim 不是位置 |
| 影响力副线 | 上游化 184 个 Megatron patch、公开可复现性能证据、恢复 weekly；判定数据天然是"公开可复现证据"的载体 |
| 已降级 | `Primus/pilot` 作为职业主线（agent 技能商品化过快，且阻塞 15 个月）。但判定层的 J0 **解掉** Pilot 卡住的同一个根因 |
| 已否决 | ①以"移植某个 kernel"为立项单位（HipKittens 那条线的教训：找了三轮才确认无落点）；②"写更优的 GEMM"的朴素版（当前靶子 85% 已达，剩余落在噪声里） |
| HipKittens 的位置 | **备用手段**：其唯一独有能力 register pinning 对症 dW1 wgrad 的 VGPR 248/512 → 2 waves/SIMD；但 GEMM 只剩 <10%，现在是"没有病的药"。取出时机 = 融合 kernel 的 epilogue 加量化数学后撞寄存器墙 |
| 长期叙事 | 性能丢在库与库的缝里，而这些缝是组织边界造成的；同时握有两层权限的人才能删掉它们 |

## 进展时间线

| 日期 | 里程碑 | 关键结论 | 来源 note |
|---|---|---|---|
| 2026-05-08 | Agent 时代职业路线梳理 | `Primus/pilot` 是最适合作为主线押注的项目之一；`MMOE` 是高壁垒 GPU kernel 副线/杀手锏 | [agentic_infra_and_gpu_kernel_career_strategy](./2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md) |
| 2026-08-06 | **Primus 定位再思考，推翻上一版主线** | 稀缺性不在 agent 也不在 kernel，在两者的组合权限；Primus 的深度是消解库间边界而非再抽象一层；职业与影响力应双轨分开解决 | [primus-positioning-boundary-dissolution](./2026-08-06_primus-positioning-boundary-dissolution.md) |
| 2026-08-12 14:28 | **定位续篇：占判定层**（主张已作废，事实部分保留） | ①"都已卡位"前提被 git 史推翻，组合权限已在手（LM #1 26.8% / Turbo #2 21.8%）；②AITER 自述不写 kernel，其 gfx950 ASM 层唯一 backward 是 `fmha_v3_bwd` → AMD 训练侧峰值投入 = 一个算子；③Turbo 已有测量/派发/知识三层，缺判定数据与噪声地板 | [own-the-judgment-layer](./2026-08-12_1428_own-the-judgment-layer.md) |
| 2026-08-12 15:05 | **靶子：量化↔GEMM 的格式缝**（**当前判断**） | ①判定层是更聪明的 wrapper 且绑 Turbo，作废；②但"更快的 GEMM"也枯竭——dW1 ragged wgrad 达自测上限 **85%**（两种精度都是），autotune 后四次尝试全无效或高风险，且 sclk ±30% 使 <10% 不可验证；③钱在格式缝：GEMM 本体 1.84–1.96× 被 colwise requant 稀释到 1.15–1.29×，~1.15 ms 量化开销跑在 35% HBM 带宽；④撞墙原因是"改 layout 会破坏下游 dW1/STEP3 契约"——**跨 op 契约无人有权改，而这正是站位的兑现点**；⑤HK 降为备用手段 | [the-quant-gemm-seam](./2026-08-12_1505_the-quant-gemm-seam.md) |

## 下一步

分期见 [the-quant-gemm-seam](./2026-08-12_1505_the-quant-gemm-seam.md) §分期（S0–S4）。

| 优先级 | 方向 | 说明 |
|---|---|---|
| **P0（S0）** | **可判性** | 钉死的 GPU4 上目标 stage 重复 30+ 次出 run-to-run 分布，定"最小可判差异"。**是 claim 的许可证，不是独立主张**；天级；若结论是"<15% 不可判"则整条线可验证性成疑 |
| **P0（S1）** | **契约测绘** | 画清 量化 kernel → dW1/dW2 GEMM → STEP3 combine 的 layout / scale 格式契约，标出哪段是"为迁就下游"而存在。纯阅读 + 归属分析，不跑 kernel |
| **P0（S2）** | **单点验证** | 选开销最大一段（dW1 或 dW2 requant 各 0.365 ms），原型化"GEMM prologue 直接吃未量化输入"或"epilogue 直接吐目标格式"；验收 byte-exact 或梯度 SNR ≥15 dB，且净收益 > S0 阈值 |
| P1（S3） | 推广 + 契约固化进 Turbo 算子接口 | 反向净 FP8 收益 1.29× → 目标 ≥1.6×；这一步兑现框架侧站位 |
| P1 | 三代 super-kernel 收敛为一条 | `monolith-moe` / `rocmoe` / `MegaMoeFlydsl` 并行稀释资产，建议保 `MegaMoeFlydsl` |
| P2（S4） | HK register pinning | 仅当 S2/S3 的 epilogue 撞寄存器墙时取用；目标 VGPR ≤170 / 3 waves per SIMD |
| P2 | `fc1_dgrad_combine` 的 0.32 ms overlap gap | 第二处钱，收益小于格式缝（0.32 vs 0.5–0.9 ms），排后 |
| P2 | 184 个 Megatron patch 债务分类 | 打标为 应上游 / 应内化 / 应删除 |
| P2（搭车） | attention 接入 `AutoKernelDispatcher` | 官方 TODO、成本低，可作影响力轨道产出；**不是职业主张** |
| P2 | 影响力轨道 | 首批 upstream PR、公开可复现性能记录、恢复 weekly |

## 文件索引

| 主题 | 文件 |
|---|---|
| 量化↔GEMM 格式缝 / 靶子选法 / S0–S4 分期（**当前判断**） | [`2026-08-12_1505_the-quant-gemm-seam.md`](./2026-08-12_1505_the-quant-gemm-seam.md) |
| 占判定层（**主张已作废**；站位数据与 AITER/Turbo 现状核查仍可引用） | [`2026-08-12_1428_own-the-judgment-layer.md`](./2026-08-12_1428_own-the-judgment-layer.md) |
| Primus 定位再思考 / 边界消解（母论点仍成立，落地形态由 15:05 那篇给出） | [`2026-08-06_primus-positioning-boundary-dissolution.md`](./2026-08-06_primus-positioning-boundary-dissolution.md) |
| Agentic infra + GPU kernel 职业策略（已被 08-06 修正） | [`2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md`](./2026-05-08_agentic_infra_and_gpu_kernel_career_strategy.md) |

## 维护约定

每次出现职业路线、项目取舍、长期技术定位相关的重要判断时，写一篇 archive note，并同步更新本 README。
