# kernel-substrate — 自主可控的 kernel 底座（以 HipKittens 为切入）

> **目标**: 拿到一个**我们能改的** kernel 底座（tile 抽象 + 近峰值 GEMM 内环 + CDNA 布局知识），使 megakernel / 各类算子融合不再受制于外部供给方排期。attention 只是它的一个消费者。
> **平台**: MI355X (gfx950, CDNA4)
> **栈**: ROCm / HipKittens / Primus-Turbo / Megatron
> **工作仓库**: `~/workspace/hipKitten/`（MIT 许可；`include/cdna4/` 原语层 + `kernels/cdna4/{gemm,attn}` + `training/llama/`）
> **对接仓库**: `~/workspace/Primus/`（Primus-Turbo 算子层）· 外环骨架复用 `~/workspace/FlyDSL/kernels/mega_moe/`
> **目录名说明**: `hk-attn-bwd` 是历史遗留（原为 attention backward 专项），2026-08-13 重写后范围已扩大，尚未改名。

## 1. 定位（2026-08-13 重写）

**原立项把这件事当成「一个 attention 项目」，用「能否打赢 AITER」当门槛。这个框架是错的。** 真正的出发点是**代码库的自主可控**，四条约束单独任何一条都足以立项：

1. **融合需要能改的代码。** megakernel、fused kernel 的本质是改内核内部。`hipBLASLt` 与 AITER 都不允许我们改——碰不到的 epilogue 塞不进 quant，不受控的算子放不进 persistent kernel。
2. **PoC KPI 交付被卡在这里。** 若一个 PoC 需要融合而底层 GEMM 不可改，这个 PoC **根本做不出来**，不是做得慢。所以底座是交付的**前置条件**，不是与交付争资源的平行项目。
3. **MegaMoE 的性能地基是 GEMM 内环。** 没有高效 GEMM 的积累，融合实现的性能会很差——外环编排做得再对也救不回内环。
4. **未来的融合机会无法响应。** 业界出现新的优化方式（典型如需要 GEMM fuse 的），手里没有可改的 GEMM 就没有入口。这条直接制约 Primus 的后续发展。

**判据由此改变：不是「性能是否占优」，而是「能否做到今天做不到的融合」。** 性能持平不构成叫停理由。

## 2. 稀缺资产：HK 提供什么外面给不了的

| 资产 | 为什么外面拿不到 | 出处 |
|---|---|---|
| **逆出来的 CDNA LDS phase/bank 行为表** | **CDNA ISA 文档根本没写**，HK 作者写了个 solver 反推，结果列在论文附录 | [`papers/hipkittens.md`](../../papers/hipkittens.md) §4.2 |
| **按共现布局给的无冲突 swizzle 集合** | AMD MFMA 布局没有 NVIDIA 那种可组合 16×16 core matrix 结构，**单一 swizzle 不可能**；HK 的取舍是为常见共现组合各给一套 | 同上 |
| **绕过 HIPCC 的 register pinning** | HIPCC 不让开发者把 AGPR 用作矩阵指令输入操作数；HK 提供显式钉寄存器的机制。**FlyDSL「交给 LLVM 分配」在此结构性吃亏** | 同上 §4.1 |
| **近峰值 GEMM 内环（纯 HIP，可改）** | 见下表；hipBLASLt / AITER 同等性能但不可改 | 同上 §5 |
| **MIT 许可 + 实际接收外部 PR** | merge 记录含 AMD 与社区贡献者，连 `ds_read_b64_tr`、`mubuf ops` 这类底层指令支持都是外部提进去的 | `hipKitten/LICENSE`、`git log --merges` |

**GEMM 不是研究水平，是可直接当底座的数字：**

| kernel | 数值 | 对照 |
|---|---|---|
| HK BF16 GEMM @ MI355X（0-producer / 256×256） | **1610 TFLOPS** | B200 上 TK 1538、CUTLASS profiler 最优 1570 |
| HK FP8 GEMM（8-wave ping-pong，热循环 48 行） | **3222 TFLOPS** | 4-wave interleave 更快但代码量爆炸（183 行） |

## 3. 接缝：内环 HK / 外环复用现有骨架

**原判断「HK 在 CDNA 上没有跨 WG 同步，所以别在它上面做 megakernel」是错的读法——那恰好是我们已经拥有的一半。**

| 层 | 谁提供 | 依据 |
|---|---|---|
| **内环**：tile 布局 / swizzle / 寄存器调度 / MFMA GEMM | **HK** | 上表 |
| **外环**：persistent grid、role-split、跨 WG 信号、原子工作队列、跨卡通信 | **我们已有** | `FlyDSL/kernels/mega_moe/` 的 role-split grid + `DispatchSlot` 信号表 + `WORK_HEAD/WORK_TAIL`；MonolithEP 同源 |

**已核实的 HK 缺口（属外环，不属内环）**：`include/cdna4/ops/` 下只有 `memory / register / shared / group`，**无 sync、无 barrier、无 cluster**；`cdna4` 子树内 `__hip_atomic_*` 与 `threadfence` 零出现；`distributed-kernels/` 只有一个经 Iris 挂出的 BF16 GEMM。→ **这些是接缝的位置，不是障碍。**

## 4. 消费者与优先级

排序依据是「当下哪个融合被卡得最疼且最值钱」，**不是哪个模型更旗舰**。

| # | 消费者 | 被卡在哪 | 备注 |
|---|---|---|---|
| **1** | **GEMM epilogue 融 quant** | FP8 训练里每个 linear 最多 4 次不可缓存的激活/梯度 cast（x/grad_y 各 rowwise+colwise）。**收益最大的一类是 GEMM 算完直接吐 FP8**，而这需要可改的 GEMM | 权重侧已由「首 microbatch 量化 + 缓存」「专家权重合并 + grouped 量化」治过；剩下的都在激活侧。norm/SwiGLU 侧已有 `silu_and_mul_fq`、`qk_norm_rope_quant` 等先例，缺的是 GEMM 侧 |
| **2** | **MoE grouped GEMM 内环** | MegaMoE 的性能地基。我们在 MMOE/MonolithEP 上靠 DTOLDS + XOR swizzle + LDS pad 自己重走过这条路，成本很高 | HK 的布局知识可直接复用，避免再逆一遍 |
| **3** | **attention backward（MLA 优先）** | 见 §6 修正：训练态 MLA 是 **qk 192 / v 128**，唯一障碍是应用层常量，**不需要 gather、不需要 576/512 swizzle** | 四个主流后端里三个断言两头维相等，**结构性空白** |
| 4 | attention（GQA） | AITER 已达 902–1047 且不可改 | 价值在「摘掉外部依赖」，不在超过它 |

## 5. 阶段计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M0** | **交付一个当前做不到的融合**，规模可以很小。首选「GEMM + FP8 quant epilogue」 | 能跑 + 数值对拍通过 + 相对「GEMM 后接独立 cast kernel」有可度量下降。**不要求打赢任何在位实现** |
| M0.5 | 量出奖池：一个 microbatch 内 quant kernel 的 launch 次数与时间占比；标出已有 fused 变体 / 未覆盖路径 | 一张覆盖率表；顺带判定「转置双重量化」值多少（若值，换 2D block 量化可让 colwise 那次不必存在） |
| M1 | 解掉 HK 编译期模板固定（`ATTN_B/ATTN_H/ATTN_H_KV/ATTN_N`），吃变长与多形状 | 生产形状全绿 + 与在位实现数值对拍。**这是主要工程量，也是最不可替代的积累** |
| M2 | attention：拆 `ATTN_D` 为 `D_QK`/`D_V`，跑 DSv3 训练态 MLA backward（qk192/v128） | 跑得起来 + 与 CK 路径对拍（另三个后端根本跑不了，无 TFLOPS 及格线可言） |
| M3 | 内环接外环：把 HK 的 GEMM 内环嵌进 mega_moe 的 role-split 骨架 | 相对现有 MegaMoE 内环有增量；目标可发表 |

**二维门槛（替代原「不占优则叫停」）**：①性能进入可用区间（到在位实现的合理比例、生产形状跑得动，**不要求打赢**）；②可控性收益可兑现（列出在 HK 上做得到、在供给方实现上做不到的动作，至少两条落地）。**两条都不成立才叫停。**

## 6. 已修正的判断（存档，避免重犯）

| 原判断 | 实际情况 | 依据 |
|---|---|---|
| 「配置修好后 turbo ≈ TE 差 <2%，说明内核层已被 CK 汇编吃满」 | **同源自比，不能作为内核已优化的依据。** 原文自己写了「因两者跑同一批 aiter/CK 汇编 kernel」——11.29 vs 11.10 ms 只证明两个 wrapper 一致。「已吃满」是推断，**仍缺 roofline 或异构实现的对照** | [plan §0.1(b)](./2026-08-12_1006_hk_primus_attn_bwd_plan.md) L83 |
| 「AITER 902–1047 已接近上限，及格线 ~1050」 | **1047 约为 BF16 峰值的 40%**（MI355X 密集 OCP-FP8 标称 5 PFLOPS → BF16 约 2.5 PFLOPS 一档，**此为量级估算**，硬件 note 未直接列 BF16）。同芯片上 HK 的 GEMM 做到 1610（约 65%）。**汇编版本也没到硬件上限** | [`knowledge/hardware/gpu-comparison.md`](../../knowledge/hardware/gpu-comparison.md)；[`papers/hipkittens.md`](../../papers/hipkittens.md) |
| 「MLA 要 576/512，且需要 gather 原语，成本极高」 | **576/512 是 absorbed 的 decode 形态（megaattn 的地盘）。训练态 MLA 是 `qk_head_dim 128 + qk_pos_emb 64 = 192` / `v_head_dim 128`。** 训练 attention 在序列上稠密，**不需要 gather**；192/128 也不构成新一档 swizzle。唯一障碍是应用层 `constexpr int ATTN_D = 128` | `Primus/primus/configs/models/megatron/deepseek_v3.yaml:18-20`；`hipKitten/training/llama/csrc/attn_fwd_causal.cpp:22` |
| 「FlyDSL 是可控的一侧」 | **FlyDSL attention 与 AITER 同源**；「AMD 官方」不等于「我们可控」。原依据是 FP8 GEMM 先例（Primus commit `fa391f32`），**该先例不能外推到 attention** | 供给方约束，见 §7 |
| 「MLA backward 是真空，值得抢」 | **不是真空**：`Turbo/primus_turbo/flydsl/attention/` 已有 sparse MLA 前向 1068 行 / 反向 2759 行（`D=512 / DQK=576`，含 topk 分块、KV 预取、QK4 门控），PR #420 | `Turbo` 仓实测 |
| 「quant 覆盖率表与本项目无关，可独立先做」 | **只对一半成立。** norm/SwiGLU 侧可独立做；**融进 GEMM epilogue 的那一类需要可改的 GEMM**，即本项目 | §4 消费者 1 |

## 7. 供给方约束（本项目的根本理由）

AITER 与 Turbo 内 FlyDSL attention **同源**，且该方向优先级在**推理**，训练算子排期长。**我们背训练 KPI，却不持有训练关键路径上任何一条可改的实现。** 三条具体代价：

- **新形状要排队**：GQA 4:1 / 8:1 / 16:1、变长、FP8-FP4 变体。
- **异常无法定性**：黑盒里只能从外面测，读不到 ISA。现存待查格子——16 heads（TP=8 每卡份额）反向掉到 **~655 TFLOPS**，但绝对时间仅 1.31 ms，更可能是 dQ 原子归约 / launch 固定开销占比放大而非调度不良。**先 rocprof 定性**：调度问题 HK 的 ping-pong / register pinning 能帮，原子归约或 launch 开销完全帮不上。
- **对 fusion 路线有架构否决权**：见 §1。

## 8. attention 消费者的形状输入

| 模型 | H_Q | H_KV | GQA 比 | head_dim | seq | 状态 |
|---|---|---|---|---|---|---|
| DeepSeek-V3（**M2 目标**） | 128 | MLA | — | **qk 192 / v 128** | 4096 | 需拆 `D_QK`/`D_V` |
| Llama3.1-8B | 32 | 8 | 4:1 | 128 | 2048 / max 131072 | 待验证 |
| Llama3.1-70B | 64 | 8 | 8:1 | 128 | max 131072 | 待验证（**HK Makefile 默认值恰为此形状**，说明 GQA 路径本身支持） |
| Qwen3-235B-A22B | 64 | 4 | 16:1 | 128 | 4096 | 待验证 |

HK `setup_kernels.sh` 当前只构建 `B=8 H=16 H_KV=16 N=2048`（MHA），与生产形状均不匹配——这是 M1 的输入。

**已有资产**：HK 的 `training/llama/llama/models/attentions/` 本身就是可插拔 attention 后端抽象，并排放着 `aiter.py` 对照组，与 Primus 的 `use_turbo_attention` 开关同构，**测台几乎不用搭**。

## 9. 风险

| 风险 | 对冲 |
|---|---|
| **接手 = 事实上成为一个 kernel 底座的维护者**，而 HK 主干不完整（FP6 只在 `fp6_experimental` 分支） | **这不是附带成本，是 megakernel 路线的入场费**——不付这笔，后面每次融合都要等排期。应作为常设人力显式决定，不要当成某个项目的附属风险滑进去 |
| 押在研究产物上 | MIT 许可可自由 fork/vendor；且**实际接收外部 PR**，不是 fork 到孤岛。长期可把 register pinning 等反向提案出去 |
| HK 编译期模板与生产变长冲突 | 即 M1 主要工程量，也是最不可替代的积累 |
| 底座价值难以在单季度 KPI 上体现 | M0 刻意设成「交付一个当前做不到的融合」而非性能表——它同时是 PoC 的前置条件，能直接对上交付 |
| 论文性能主张已被实测削弱（AITER 在 MI355X 达 902–1047 vs HK 论文 1024） | **本项目已不依赖该主张**。性能论证换成 roofline 口径（§6 第二行） |

## 10. 需要盯住的单一变量

**若 attention / GEMM 的供给排期变得可预期，或我们拿到了往里提代码的通道，本项目的理由即消失**——因为性能那条已经不成立。这是唯一会推翻立项的变化。

## 文件索引

| 文件 | 内容 |
|---|---|
| [2026-08-12_1006_hk_primus_attn_bwd_plan.md](./2026-08-12_1006_hk_primus_attn_bwd_plan.md) | 原 attention backward 专项计划：缺口分析、AITER 基线过时的三轮证据、形状缺口、风险与对冲。**其 §0.1(b) 的「内核层已吃满」结论见 §6 第一行修正** |

## 相关

- [`papers/hipkittens.md`](../../papers/hipkittens.md) · [全文中译](../../papers/hipkittens-zh.md) — 稀缺资产与 GEMM 数字的出处；**注意 §7.1「wave specialization 在 AMD 是负优化」针对的是 workgroup 内的 wave，不否定跨 CU 的 WG 角色分区**
- [`papers/swizzled-head-first-attention.md`](../../papers/swizzled-head-first-attention.md) — XCD/L2 感知调度，M3 的先行工作
- [`megaattn/`](../megaattn/README.md) — DSA decode 三段融合（absorbed 576/512 + 稀疏 gather），与本项目**互补不重叠**；其「FlyDSL 稀疏路径为零」前提需按 §6 第五行修正
- [`papers/ubep.md`](../../papers/ubep.md) — 「改写通信库」而非「写 megakernel」的对照路线；其 §6 对 fused persistent kernel 的两条批评值得正面回答
- [`knowledge/systems/primus-pipeline-runtime-megatron-integration.md`](../../knowledge/systems/primus-pipeline-runtime-megatron-integration.md) — Primus 接入机制
