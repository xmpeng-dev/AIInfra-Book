# megaattn — DSA attention megakernel (indexer + top-k + sparse MLA)

> **目标**: 把 DeepSeek Sparse Attention 的 decode 三段 —— lightning indexer (paged MQA logits) → top-k 2048 → sparse MLA —— 融进一个 persistent kernel，复用 MegaMoE v2 已验证的 role-split 骨架
> **平台**: 8x MI355X (gfx950, CDNA4)
> **栈**: ROCm / FlyDSL / Primus
> **目标模型**: DeepSeek-V3.2 → V4 家族。MLA absorbed (QK 576 = 512 latent + 64 rope, V 512, 128 heads)，indexer 64 heads x 128 dim，index_topk = 2048
> **工作仓库**: `~/workspace/FlyDSL/kernels/attention/`
> **参考仓库**:
>   - `~/workspace/FlyDSL/kernels/mega_moe/` — 骨架来源（role-split grid / DispatchSlot 信号表 / 原子工作队列）
>   - `~/workspace/MegaMOE/` — DeepGEMM，indexer 语义与对拍基准（`sm100_fp4_paged_mqa_logits.cuh`）

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | **设计 2026-08-05 16:50 — v1 架构设计成文，M0 摸底待启动**。核心论点：`[next_n, S_kv]` fp32 logits 张量只有 top-k 一个消费者，不该落 HBM；用 LDS 在线阈值剪枝把 top-k 输入从 `N x 4B` 压到 `~24 KB`。骨架与 MegaMoE 同构（打分 → top-k → gather → 算），且**无跨卡通信**，比 MegaMoE 简单一档。详见 [v1 架构设计](./2026-08-05_1650_megaattn_v1_architecture_design.md);**下一步 P0**: M0 分段 profile，若 top-k + logits 往返合计 < 15% 则就地叫停 |
| 前置缺口 | `pa_metadata.py:1242` 硬断言 `topk == -1`，FlyDSL 稀疏路径为零；attention 全栈无 backward |
| 已有资产 | MLA m16x8 FP8 tile loop（`mla_fwd_decode_m16x8_fp8_fp8.py`，2096 行）与 split-KV reduce 已就绪，缺的只是「按索引取 KV」 |
| M0 门槛 | N = 8K / 32K / 128K 三档分段 rocprof，量出 indexer / top-k / sparse MLA 实际 wall 占比 |
| M3 目标 | 单层 attention wall 相对「三独立 kernel」基线降 ≥ 25% @ 128K，精确 top-k 逐位一致 |

## 设计核心（3 个改动 vs 三独立 kernel 基线）

| # | 改动 | 砍掉的开销 | 来源 |
|---|---|---|---|
| 1 | indexer 边算边在 LDS 维护近似阈值，只回写越阈候选 | logits 全量写 + radix 多趟读（N=128K 时约 2.3 MB / 层 / 序列） | LiteTopK ([2607.11976](https://arxiv.org/abs/2607.11976)) |
| 2 | decode 用上一 step 的第 2048 名分数做初始阈值 | 相邻 step top-k 重叠 35-50%，阈值一上来就接近真值 | NVIDIA Guess-Verify-Refine |
| 3 | scorer / selector / attn 三角色共处一个 persistent grid | 3 次 launch → 1 次，2 次全局同步 → 0，段间可流水 | MegaMoE v2 `mega_moe_stage1.py` |

## 阶段计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 三独立 kernel 基线 + 分段 profile | 拿到占比表；< 15% 则叫停 |
| M1 | 独立实现 paged MQA logits (FP8) + k=2048 radix top-k | 与 DeepGEMM 逐位对拍；不差于 AITER / ROCm blog |
| M2 | 融合 indexer + 在线剪枝 + top-k | 相对 M1 两段之和降 ≥ 30%；回退率 < 1% |
| M3 | 接 sparse MLA，role-split persistent grid | 相对 M0 降 ≥ 25% @ 128K |
| M4 | prefill 路径、FP4 indexer (V4)、CSA/HCA | 按 V4 落地节奏 |

## 文件索引

| 文件 | 内容 |
|---|---|
| [2026-08-05_1650_megaattn_v1_architecture_design.md](./2026-08-05_1650_megaattn_v1_architecture_design.md) | v1 架构设计：瓶颈分析、role-split 结构、与 MegaMoE 的同构映射、`work_info` 接口改动、分阶段计划与风险表 |

## 相关

- [H2 2026 roadmap](../primus-moe/2026-05-29_roadmap_h2_2026.md) — L96 DSA / L120 fused MQA / L121 CSA-HCA / L260 FP8 indexer
- [rocmoe v2 架构设计](../rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md) — role-split + scoreboard 同步的先例
- [训练优化 landscape 2026](../../knowledge/systems/training-optimization-landscape-2026.md)
