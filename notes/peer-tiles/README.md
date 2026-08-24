# peer-tiles — 自有 repo：AMD 上融合 MoE 训练 kernel 的 tile 原语库

> **论点**（2026-08-12 15:52 收紧）: **AMD 的 mega 范式是「stage 级成对融合」——一次通信配一个保持库级 tile 几何的 GEMM，两条腿等长。整层单核融合会切碎 GEMM 的算术强度，而该损失随 token 数线性增长，必然在生产规模翻盘**
> **实现工具**: tile 抽象的第四条原语「所有权与可见性」（谁写、何时对谁可见、代价多少）——是写出上述范式的手段，不是主论点
> **对照物**: HipKittens（检验并重做了 ThunderKittens 的三条原语；kernel 是论证的证据而非产品）
> **平台**: 8× MI355X (gfx950, CDNA4, XGMI 全互联) + MI300X (gfx942)
> **载体**（2026-08-12 19:30 再修正，第三版）: **混合** —— **HIP C++ 写融合骨架（通信 / 角色 / 调度 / combine）+ FlyDSL 生成的 GEMM tile 以 AMDGPU bitcode 交付**，经 `-Xclang -mlink-builtin-bitcode` 链入、wrapper `alwaysinline` → tile 内联进调用方 WG、跑满寄存器文件、无 call-ABI spill。这是 `3rd/ROCMega` 已在用的架构，分界恰落在两者强项上。FlyDSL 是 `github.com/ROCm/FlyDSL`，**公开可 pip 装**（0.3.1）。~~v1 纯 HIP C++ header 错~~ ~~v2 纯 FlyDSL 不完整~~
> **范围**: MoE 层 fwd+bwd 的融合 super-kernel。**不做**通用 GEMM / attention / 新 DSL / 推理
> **命名**: `peer-tiles` 是占位名，见立项文档 §8

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | **M0 待跑 2026-08-12 15:52** — 三方 A/B（gen-1 / gen-3 / PyTorch+RCCL，同日同机，直接测 forward）。**它决定项目叙事**，见下一行 |
| **已经赢了（19:30 由 ROCMega 文档确证）** | DSv3 T=8k per-rank / EP8 forward：PyTorch+RCCL **18.64 ms**（1.00×）· **gen-1** monolith-moe **64.47 ms（0.29×）** · **ROCMega 7.105 ms（2.6×）** · **gen-3 FlyDSL mega 5.978 ms（3.1×）**。后两者是**同环境、钉时钟 2400 MHz、逐字节匹配 routing** 的直接对照（`3rd/ROCMega/doc/fwd_vs_flydsl.md`）。⚠ PyTorch 那行仍是 5 月异节点未钉时钟 → **M0 缩小为"只刷新这一条基线"** |
| ~~gen-3 forward 由减法估 ~7.80 ms~~ | **作废**，直接测得 **5.978 ms / 971 TFLOP·s⁻¹·GPU⁻¹** |
| 范式命题：**已独立复现** | gen-1 = 单核全融（4 role / 5 sub-phase / grid barrier / chunk 流水）→ 0.29×。**ROCMega 与 gen-3 是两个独立实现、不同语言，却收敛到同一个四腿结构** `prologue → dispatch+FC1 → SwiGLU → FC2+combine+reduce`，每 kernel 只融「一次通信 × 一个 GEMM」，两腿近等长（ROCMega 3.595 ‖ 3.181；gen-3 3.178 ‖ 2.445），双双 2.6–3.1×。算式：收益 ≤ 被藏起的通信（有界），代价 = FLOPs×(1/η_fused−1/η_lib)（随 T 线性）；gen-1 @T=8192 代价 ≈53 ms vs 收益 ≤7.39 ms |
| **共驻代价的实测形态** | ROCMega 与 gen-3 的全部 1.1 ms 差距 = **FC2 NT tile 在 1 WG/CU régime 下缺 `ks→ks+1` lookahead**：融合 kernel 的 128 KB 静态 LDS 把占用压到 1 WG/CU → 没有 sibling wave 掩盖 `wait_lgkm` → 独立场景下成立的流水假设失配 → **GEMM −20%（971→812）、forward −15%**。FlyDSL tile 是显式单 WG 软流水（4-buffer distance-2 + `s_setprio` + `vmcnt(3)`）。**这就是"HK 贡献纪律而非 GEMM"的实测版** |
| **MegaMoE + HK 的判断（08-13 07:20）** | ⚠ 先更正一处读错：HK 的 `Occupancy 2 waves/SIMD` 是**单 WG 8 waves 铺满 4 个 SIMD = 1 WG/CU**，**不依赖** sibling-WG 重叠。其循环体是条件 barrier + `s_setprio` + 手调 vmcnt 的 **8-wave ping-pong** —— **与 FlyDSL tile 同属"单 WG 显式流水"类别，与 ROCMega 失配的那个不是**（只是 2-buffer vs 4-buffer 浅一档）。落点判断：**换 gen-3 的 tile = 低价值**（那槽位已 971 TFLOP/s）；**补 ROCMega 的 FC2 tile = 较高价值**（有已确诊的 20% 缺口 + HIP 骨架更适合接 C++ tile）；**作为"融合régime 下的 tile 设计"研究 = 最高价值**。硬缺口仍是 grouped/变长-M 与 device-function 重构。**顺序：先做不需要 HK 的 S0**（压 FC2 tile 的 LDS 到 2 WG/CU，看缺口是否闭合）—— 闭合则病因是占用、HK 无落点；不闭合则病因是流水深度、HK 才有戏 |
| **HK 的角色（已修正 + 16:30 实测确认）** | **不是"更快的 GEMM"**，而是**让 GEMM 与别的 role 共驻而不掉效率的纪律**：register pinning（`WG_PER_CU=1` 拿 −35%、dW1 卡 VGPR 248/512 = 同一约束三次现身）、8-wave 角色轮换、Table 5 LDS phase/bank、单 kernel 混用多 MFMA 形状 |
| **实测（2026-08-12 16:30 + 17:45 调优）** | **出厂配置弱、调完反超**。出厂 `BLOCK_SIZE=256`：per-expert M=2048 打平偏输、分 chunk 崩塌到 **0.37×**，机制是**网格饥饿**（`grid=(N/256)*(M/256)`，吞吐 ≈ grid占CU比例 × 峰值）。**只改 tile 256→128 + chiplet window WGM**：chunk2 **0.68→1.31×**、FC2 dgrad **0.70→1.28×**、chunk4 **0.49→1.07×**、chunk8 **0.37→0.84×** —— **5 个 MoE 形状赢 3 个**。→ [实测 note](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md) |
| **关键副产物：没有通吃配置** | BS=128 让大方阵从 ~1.0× 掉到 0.69–0.75×。hipBLASLt 靠运行时按形状选 tile 取胜，HK 只出厂一份。**但 mega kernel 里形状是编译期已知的 → 逐点特化本来就是对的做法，hipBLASLt 的选型优势在此失效，而 HK「手调配置 + 编译期特化」的模型恰好对口。** 有效旋钮 3 个（`BLOCK_SIZE` / `WGM` / 大形状留 256）；`WARPS_M/N` 不是旋钮（改了数值就错），`BS=64` 编译失败 |
| **仍然成立的缺口** | ①**HK 没有 grouped / 变长-M GEMM，只有 dense** —— 要进 mega 得自己写 grouped driver，而 gen-3 的 `gemm_helper.py`(1496 行) 已有（grouped mxfp8 wgrad 2007–2199 TFLOPS）；②HK **无自动选型**（论文批判点 5 自己承认）；③编出来是 **AGPR=0 / VGPR=210 / 2 waves/SIMD** 的编译器托管版，连 HK 自己的 register pinning 在这个 GEMM 里都没启用 |
| 还在饥饿里的一格 | chunk8 `M=256` 仅 0.84×：BS=128 时 grid 仍只 64 WG（25% CU）。需**非对称 tile**（BM=64/BN=128），而 `micros/192x256/kernel.cpp` 已是 `BLOCK_SIZE_M`/`_N` 分离 → 机制存在，换基底文件即可试。另 FC1 gateup 0.88× 是**唯一非饥饿的落后格**（满占用），怀疑 torch 走 split-K，值得单独 rocprof |
| 护城河 | **RCCL 语义上不可能 overlap**（架构限制非实现 bug）→ 必须进 kernel；而 8 卡 XGMI 全互联 + HIP IPC 让 peer HBM 可直接寻址，NVIDIA 要靠 TMA + symmetric memory 绕；**AMD CDNA 上 in-kernel XGMI overlap 是全空白**（2026-07-07 paper scan） |
| 最大的既有资产（**16:20 修正**） | **库的内容一半是散文、一半已经是代码，而且是跑出 2.4× 的那一半**：`prims.py`+`fp8/prims.py`(340) 已把 scope/order 参数化 = F2 的核心语义；`symm_buffer.py`×2(1150) = F1 的 `peer` tier；`gemm_helper.py`(1496)+`gemm_mxfp8_tile.py`(363) = 库级 GEMM tile；`barrier.py`×2(221)。散文那半是 `cco-pipeline-overlap/SKILL.md` 494 行。外加三代 79 篇 note |
| 最诚实的一条 | 论点必须把 regime 边界写进去：gen-1 在 512 t/g **1.46×** → 2048 t/g **0.53×** → 8192 t/g **0.29×**。别人只发赢的 regime，这张表无法被复制；**而若 M0 确认 gen-3 已把它翻过来，"我们做出了第一个在生产规模打败库路径的 AMD mega kernel + 这是让它成立的范式"是更强的资产** |
| 项目性质（**16:20 修正**） | 不是"建一个库"，是**"让一个已经存在的库毕业"**：代码有了（~10k 行）、结果有了（2.4×，待 M0 确认）。缺的是独立 repo / 名字 / 论点 / regime 判据 / API 纪律 / 与库路径的权威对照 |
| ~~生死线：抽象税~~ | **作废** —— 抽象税已被 gen-3 自己证明为零（wgrad 2199 TFLOPS = 自测上限 85%）。新的生死线是 **M1 能否在 Turbo 之外独立跑出同一数字** |
| 前置项 | 噪声地板（`career-strategy/2026-08-12_1505` 的 S0）。sclk ±30% 下 <10% 不可判 |

## 四条原语（第 4 条是实现范式的工具，不是主论点）

| # | 原语 | 出处 | 状态 |
|---|---|---|---|
| 1 | Tiles | TK → HK（**通用**） | 复用 |
| 2 | Overlapping | HK 重做（wave spec 是负优化 → 8-wave ping-pong / 4-wave interleave） | 复用，但 §待决 2 有矛盾 |
| 3 | Grid scheduling | HK 重做（chiplet 感知 +19%） | 复用 |
| **4** | **Ownership & Visibility** | **本项目** | 待设计；**在「stage 级成对融合」里它的作用面缩小到 stage 之间与两条腿之间的 publish/acquire**——这反而让 API 更小、更可能做对 |

> 范式收紧的一个副作用：原设想的 `chunk_pipeline<Chunks,Phases>`（5 个 sub-phase 的深流水）是 gen-1 的形状，正是被 §0.2 判为失败原因的那个结构。**F3 应该重心转向"两条腿的等长与 CU 配比"，而不是多级 chunk 流水。**

## 原语族（API 草案见立项文档 §3）

| 族 | 内容 | 来自 skill 的哪部分 |
|---|---|---|
| **F1 放置与流水** | `tile<T,M,N,Layout,Tier>`（`Tier` 含 `peer`）、`layout::padded/xor_swizzled`、`staged<Depth>`、`load_async` | Principle 1（技法 1.1–1.4）+ Principle 2（技法 2.1–2.5）+ Pattern D |
| **F2 所有权与可见性** | `scope{wave,wg,agent,system}`（代价进类型）、`publish/acquire`、`ready_matrix<Chunks,Phases>`、`system_barrier`（全 kernel 只允许 2 个）、pull-based fanout 为默认 | Principle 3（技法 3.1–3.6）+ Pattern C；方向性语义来自 `memory-access-patterns.md` Q1/Q5 |
| **F3 角色与骨架** | `roles<Comm,Compute,Tail>`、`chunk_pipeline<Chunks,Phases>`、`work_queue` | Pattern A / B + 技法 1.3 / 3.4 |
| **F4 验证模式** | 把 skill 的三个 Diagnostic 变成库的 verify 开关 + rocprof wrapper | Principle 1/2/3 的 Diagnostic 段 |

库相对散文的两个真实增量：①`Layout` 类型让 writer/reader 共享 swizzle 常量，**off-by-one 在类型系统里不可能发生**（消灭 skill Quick Reference 里那条"numerical drift after fuse"）；②`scope` 进模板参数，编译期就能拒绝"chunk 循环里用 system scope"。

## 里程碑

| M | 内容 | 验收 |
|---|---|---|
| **M0**（19:30 缩小） | ~~三方 A/B~~ —— ROCMega 已给出 gen-3 与 ROCMega 的同环境钉时钟对照。**只剩：在同一钉时钟环境刷新 PyTorch+RCCL 基线**（现基线 2026-05-13、异节点、未钉时钟） | 把 2.6× / 3.1× 从"量级可信"变成"精确可引" |
| **M0'** | 论点 + API 定稿，**零代码** | 一页别人能反驳的东西；三个待决问题有答案 |
| **M1**（16:20 改写） | **毕业**：把 `primus_turbo/flydsl/mega/` 抽成独立 public repo，只依赖 `pip install flydsl`，在 Turbo 之外复现 M0 的数字 | 零环境 `pip install` 后跑出同一数字（噪声内）。**这才是"换工作带得走"的真实测试** |
| M2 | regime 扫描 + 判据模型 | 能预测 crossover；含"精度越低融合越该赢"这个可证伪推论 |
| M3 | 用库写一个原来写不出来的 kernel：量化融进 GEMM prologue/epilogue | 反向净 FP8 1.29× → ≥1.6×（= `career-strategy/2026-08-12_1505` 的 S2/S3） |
| M4 | 论文 | MLSys / ASPLOS |

**M1 之前不写任何新 kernel。**

## 待决问题（M0 的输入）

1. 论点表述用 **A 跨设备 tile** 还是 **B 所有权与可见性**？倾向 B（能统一解释 push/pull、flag matrix、barrier 代价表、同卡跨 WG），建议标题用 B、图和第一节用 A 落地。
2. **wave specialization 该不该是默认？** skill 技法 1.3 说窄条件下该用，HK §7.1 说 AMD 上是负优化。假设是"comm role 不能被计算 wave 吸收 + `buffer_load_lds` 不占 VGPR"让 HK 结论在此不适用。需要一次实测（= rocmoe 悬着的 M4-α vs M4-β），它同时是 F3 的默认值和论文的一个 finding。
3. repo 名。不要用 `*-kittens`——蹭 lineage 会削弱"第四条原语"这个独立贡献。

## 风险

| 风险 | 对冲 |
|---|---|
| **第四代问题**（三代已失败） | artifact 类型不同（库 + 判据 vs 点解 kernel）；论点含输的 regime。若生产 regime 全输，论文转为"in-kernel 融合的边界在哪"，**仍成立只是结论换向** |
| ~~抽象税~~ | 已被 gen-3 证伪（2199 TFLOPS = 上限 85%），不再是风险 |
| **绑 FlyDSL 版本** | 吃它的 version churn：`2026-08-01_1440_..._regression_flydsl024_...` 就是一次 0.2.4 引起的回归；aiter pin 在 0.1.7 而上游已 0.3.1。对冲照抄 aiter：**pin + CI 双跑** |
| **代码住在 AMD 产品仓** | gen-3 现在是 `primus_turbo/` 的子目录 → 与"判定层绑 Turbo"同一个问题。对冲即 M1：抽成独立 repo + 论文（HK 的 kernel 进了 AITER，但 repo 与论文永远是 Stanford 的） |
| 单人维护 | 范围锁死，核心 < 5000 行 |
| AMD 官方自己做 | 空白已存在一年以上；本论点需多卡 + 训练 + 生产 workload 三者齐备，Turbo/FlyDSL 目前是单机单卡视角 |
| 测量噪声 | S0 前置；M2 的 crossover 是大效应（0.53× vs 1.46×），不怕噪声 |

## 文件索引

| 文件 | 内容 |
|---|---|
| [2026-08-12_1530_repo-charter-and-primitive-api.md](./2026-08-12_1530_repo-charter-and-primitive-api.md) | 立项文档：§0 修订（范式命题 + gen-3 可能已赢 + 载体改为 FlyDSL）、与七个在位者的差异化、范围与不做清单、F1–F4 原语 API 草案（已存在 vs 待纪律化）、证据→论文章节编排、regime 判据要扫的五个轴、M0–M4、风险、三个待决问题 |
| [2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md) | **实测**：HK bf16 GEMM 在 DSV3 MoE 形状（per-expert M=2048 与 chunk M=256/512/1024）vs hipBLASLt；出厂配置 0.37–0.86×，调 tile+WGM 后 3/5 格反超（1.07–1.31×）；网格饥饿的定量诊断；有效旋钮只有 3 个；噪声地板（单向下漂 2–6.8%） |
| [2026-08-12_1930_rocmega_analysis.md](./2026-08-12_1930_rocmega_analysis.md) | **ROCMega 分析**：混合载体（HIP 骨架 + FlyDSL bitcode tile，`-mlink-builtin-bitcode` + `alwaysinline`）；范式命题的独立复现；gen-3 forward 的直接测量 5.978 ms；1 WG/CU 共驻代价的 root cause 与它对 HK 主张的确证；§7 MegaMoE+HK 的可行性判断 |
| [2026-08-14_1450_attn_model_shape_sweep.md](./2026-08-14_1450_attn_model_shape_sweep.md) | **20 个主流模型真实形状横测（40 行）**：①HK 能跑 **13/40**，全胜 **1.04–1.23×，几何平均 1.12×**；②**HK 的增量集中在 aiter 的弱格**（llama4 B=1 → 1.23×、gpt_oss d=64 → 1.20×；而 aiter 饱和的 grok/mixtral 只 1.04×），与 GEMM 那轮同构；③**`d=256` 是三方都没覆盖的无人区**（ck 494–556 = d128 的一半、aiter Triton 崩到 114–134、HK 无实例），glm5 + qwen3_5_35B 在用；④**fp8 全线 169–454 TFLOPS，慢 2–4×**（占 20/40 行）；⑤HK 三条硬限制：bf16 only / `d_qk==d_v` / `d∈{64,128}` |
| [2026-08-14_1330_hk_attn_vs_turbo_blocked.md](./2026-08-14_1330_hk_attn_vs_turbo_blocked.md) | **attention 全后端横测（一张表，同进程背靠背）**：①**HK / 最快 turbo = 1.04–1.16×，几何平均 1.10×，6/6 全胜**；"最快 turbo"全 6 行都是 aiter CK/asm；②turbo 三路径 **CK/asm > aiter Triton 0.59–0.80× > Triton fp8 0.19–0.47×**；③**FlyDSL 只有 sparse MLA 一个算子、跑不了 GQA**，在其自身形状上 vs 仓库内 Triton oracle **8/8 领先，fwd 1.43–2.03× / bwd 1.52–2.54×**；④编译歧义的**根因定位**（别名模板 `attn_tile<D,…>` 里 `D` 未被使用 → clang 22 判不出三个 concept 互斥）与语义等价修法 |

## 相关

- `.cursor/skills/cco-pipeline-overlap/SKILL.md` — **494 行原语清单，库的内容本体**
- [`knowledge/kernels/memory-access-patterns.md`](../../knowledge/kernels/memory-access-patterns.md) — 五问；F2 方向性语义的依据
- [`papers/hipkittens.md`](../../papers/hipkittens.md) — §2 三条原语 · §7.1 wave spec 为何亏 · §9 批判 3（无 MoE 无多卡）
- [`notes/career-strategy/2026-08-12_1505_the-quant-gemm-seam.md`](../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md) — M3 的内容；S0 是 M2 前置
- [`notes/career-strategy/2026-08-06_primus-positioning-boundary-dissolution.md`](../career-strategy/2026-08-06_primus-positioning-boundary-dissolution.md) — execution model 母论点，本项目是它的库形态
- `notes/monolith-moe/` · `notes/rocmoe/` · `notes/MegaMoeFlydsl/` — Eval / Ablation / 负结果的数据来源
