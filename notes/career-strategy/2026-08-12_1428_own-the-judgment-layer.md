# 定位续篇：占判定层，而不是再写一个 kernel

> **When**: 2026-08-12 14:28 UTC+8（**15:05 自我修正，见文首「修正」**）
> **Where**: 登录机，静态代码与 git 史核查（未跑 GPU）
> **Context**: 对 [`2026-08-06_primus-positioning-boundary-dissolution.md`](./2026-08-06_primus-positioning-boundary-dissolution.md) 的续篇。起因是"推理 AITER、训练 Turbo 都已卡位，还能做什么"这个判断，以及连续两周 Turbo commit 全是 refactor 带来的"只在打小补丁"的感觉

---

## 修正（2026-08-12 15:05）：主张作废，判定层降级为工具

**本篇的主张（占判定层）被否决，理由是它撞在 8-06 自己的论断上**：「集成层结构上就是低信用位置……更聪明的 wrapper 还是 wrapper」。dispatcher + tuned CSV 绑死在 Primus-Turbo 上，而 Turbo 在 AMD 之外无人使用 → **离开 AMD 资产归零**。AITER 的 dispatcher 值钱是因为 AITER 值钱，这个前提不可复制。

**替代主张见 [`2026-08-12_1505_the-quant-gemm-seam.md`](./2026-08-12_1505_the-quant-gemm-seam.md)**：消解缝隙的正确落法是**融合 kernel 把缝删掉**，不是加一层派发。靶子是量化↔GEMM 的格式契约（MegaMoE MXFP8 反向：GEMM 本体 1.84–1.96×，净收益被 colwise requant 稀释到 1.15–1.29×）。

**本篇仍然成立的部分**（不要连带丢弃）：

| 保留 | 内容 |
|---|---|
| §核查一 站位数据 | Primus-LM 173/646 #1、Turbo 96/441 #2、Turbo `benchmark/` 44/103 #1；跨两层前二只有一人。"都已卡位"前提确实不成立 |
| §核查二 AITER 事实 | AITER 自述不写 kernel；gfx950 ASM 层唯一 backward 是 `fmha_v3_bwd`（128 个 `.co`），`fmoe` 只有前向 → **AMD 训练侧峰值投入 = 一个算子** |
| §核查三 Turbo 现状 | 测量管道 / 派发框架 / 优化知识三层已有；判定数据、噪声地板、裁决为零 |
| J0 噪声地板 | **降级**：不再是"主线第一段"，而是"证明 kernel claim 所需的最小测量"。sclk ±30% 下 <10% 的收益无法验证，这是继续在 GEMM 上抠的硬约束，也是新主张必须先解决的事 |
| J2 attention dispatcher | **降级为搭车项**：官方 TODO、成本低、可作为影响力轨道的产出，但不是职业主张 |

---

## TL;DR（原文，主张部分已作废）

三条结论：

1. **"都已卡位"这个前提不成立。** git 史显示：Primus-LM 173/646 = **26.8% #1**（第二名 88，不到一半），Primus-Turbo 96/441 = **21.8% #2**，Turbo 的 `benchmark/` 44/103 = **42.7% #1**。同时在算子层与框架层都排前二的人只有一个。8-06 认定为稀缺的"组合权限"**已经在手**，缺的不是位置，是挂在位置上的 claim。
2. **发力方向是判定层，不是第五个写 kernel 的人。** AITER 自述不写 kernel（`knowledge/libraries/aiter.md` §1），它靠多后端 dispatch + tuned CSV 入仓 + autotuning 入 CI + API 稳定性成为 vLLM/SGLang 默认后端。**AMD 在推理侧赢在拥有判定数据。训练侧的判定数据是零。**
3. **前三层已铺好，缺最后一公里。** Turbo 已有测量管道（`benchmark/ops/training/run_suite.py`，主要是自己写的）、派发框架（`core/backend.py` 的 `AutoKernelDispatcher` 四级优先级）、优化知识（`agent/skills/` 17 个）。缺的是判定数据、噪声地板协议、和任何权威的跨后端裁决。

## 背景：为什么会有"只在打小补丁"的感觉

这个感觉有确切的来源。最近 15 个 Primus-Turbo commit（2026-08-07 → 08-12）全部是同一类：

```
refactor(mega/mxfp8): give each combine direction its own body, not a role flag to branch on
refactor(mega/mxfp8): drop five fp8 combine knobs, and say what the double barrier really is
refactor(mega/mxfp8): drop the preshuffled LDS branches and the forked scaled MFMA
refactor(mega/mxfp8): drop the fp8 knobs and paths production never takes
refactor(mega/mxfp8): keep only the prims that make the fp8 path different
```

删旋钮、拆死路径、收敛入口。工程上都对，但**没有一个 claim 挂在上面**。这不是"位置不够"的症状，是"有位置没有主张"的症状。

同期 `notes/hk-attn-bwd/` 那条线的教训是同一件事的另一面：找了三轮才确认 HK 在 dense attention backward 上没有落点，而失败的根因是**把"能不能移植一个 kernel"当成了立项的单位**。kernel 是点解——这一条 8-06 已经写过（512 t/g 赢、2048 t/g 输），但当时只应用在 super-kernel 上，没有应用到"选题"这一层。

## 核查一：站位数据

| 仓库 | 我的 commit | 占比 | 排名 |
|---|---|---|---|
| Primus-LM (`a6c7fcd8`) | 173 / 646 | **26.8%** | **#1** |
| Primus-Turbo (`bdd96e69`) | 93 (`xmpeng-dev`) + 3 (`Xiaoming Peng`) = 96 / 441 | **21.8%** | **#2** |
| Primus-Turbo `benchmark/` | 44 / 103 | **42.7%** | **#1** |

别人的跨仓分布（这是关键对照）：

| 人 | Primus-Turbo | Primus-LM | 形态 |
|---|---|---|---|
| xiaobochen-amd | 125 (#1) | 13 | 纯算子 |
| wenxie-amd | 3 | 88 (#2) | 纯框架 |
| RuibinCheung | 56 | 38 | 两侧但各约半量 |
| kyle-256（FlyDSL） | 56 | 榜外 | 纯 kernel |
| **本人** | **96 (#2)** | **173 (#1)** | **两侧均前二，唯一** |

→ 8-06 §「主张」里那句"稀缺的是组合权限：kernel 级执行权 + 框架级调度权同时在一个人手上"，**在 git 史上已经成立**。这改变了问题的性质：不是"怎么进场"，是"进场了要主张什么"。

## 核查二：AITER 的护城河不是 kernel

`knowledge/libraries/aiter.md` §1「它不是什么」原文：

> 不是单一 kernel 库 —— 它本质是 "facade + dispatcher"，底下大量 kernel 来自 CK / ck_tile / hipBLASLt / Triton / 手写 ASM

它凭什么成为默认后端，同一份文档 §3.3 已经答了：接口稳定性、支持矩阵按 production 模型为单位反向调优、**autotuning 入 CI + tuned CSV 入仓作为版本化兼容性资产**。§4 的可借鉴模式表六行里，有三行（多后端 dispatch 收进 op、JIT cache key、autotuning 入 CI + CSV 入仓）的 "applies to us" 一栏写的都是 primus-turbo。

**两个月前已经写下答案，当时当成"可借鉴的模式"，没当成"我的定位"。**

另有一条支撑（本次核查）：AITER pinned 版本 `v0.1.14.post1` 的手写 ASM 层（`hsa/gfx950/`），`find -iname "*bwd*" -o -iname "*dgrad*" -o -iname "*wgrad*"` **只命中 `fmha_v3_bwd` 一个目录**（128 个 `.co`）。`fmoe` / `fmoe_2stages` 只有 stage1/stage2 前向。即**AMD 在训练侧的峰值 kernel 投入 = attention backward 一个算子**，其余训练 backward 都在 Triton 层。

→ 这条既说明训练侧确实有空缺，也说明**空缺不在"某个 kernel 慢"**：自测数据反着说，手写 grouped GEMM 在 DSV3 ragged 形状上只有 CK 的 0.58–0.60×（643T vs 1050T），CK 没把性能留在桌上。真正的空缺在跨库的缝（dispatch+combine 占 MoE 层 40–60%），以及**在"谁来判定"这一层**。

## 核查三：判定层在 Turbo 里的现状

| 层 | 状态 | 证据 |
|---|---|---|
| 测量管道 | **已有** | `benchmark/ops/training/{run_suite.py, benchmark_suite.yaml}`：flat task list、按 `group` 过滤、`shardable` 跨 GPU 切分合并、8 GPU 调度、输出 CSV。**44/103 commits 是自己的** |
| 派发框架 | **已有** | `primus_turbo/pytorch/core/backend.py`：`GlobalBackendManager` + `AutoKernelDispatcher.dispatch()` 四级优先级（code > env > auto-tune > default > fallback），GEMM / GroupedGEMM / MoE dispatch-combine 已接 |
| 优化知识 | **已有** | `agent/skills/` 17 个：`kernel-optimize`、`gemm-optimization`、`lds-optimization`、`prefetch-data-load`、`bisect-perf-regression`、`capture-kernel-trace`、`kernel-trace-analysis`、`flydsl-kernel-authoring`、`flydsl-tile-programming`、`add-target-atom-op` 等；`agent/{historical_experience,rules,workspace}` |
| **判定数据** | **零** | `primus_turbo/` 下无任何 tuned config CSV；无 autotune 产出物入仓目录 |
| **噪声地板** | **零** | MI355X sclk 抖 ±30%，`rocm-smi --setperflevel high` 不支持（`notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md`）；只有 `scripts/dev_bench_with_keepalive.sh` 这个雏形 |
| **裁决** | **零** | attention 未接 `AutoKernelDispatcher`，后端硬编码 `if sink is None`（`attention_aiter_impl.py:146/265`），两处 `TODO(ruibin): Add unified attention kernel dispatcher`（`:373/:478`）；无 `PRIMUS_TURBO_ATTN_BACKEND` |

**前三层铺好了，其中两层主要是自己铺的。缺的是最后一公里，而走它需要的正是那个唯一具备的组合。**

## 主张

> 不去抢"写 kernel"的位，去占"判定"的位：**定义 AMD 训练算子如何被选中。**

三件可交付物，逐级依赖：

1. **测量协议**（许可证）。噪声地板刻画、最小可判差异、统计口径、环境指纹（gfx / ROCm / hipcc flags / aiter 版本 / sclk 策略）入 CSV header。没有这个，后面所有数字都不成立。
2. **判定数据**（护城河）。`(op, shape, dtype, arch, regime) → 哪条后端赢`，CSV 入仓、CI 周期刷新。这是 AITER 那套模式移到训练侧，`aiter.md` §4 已经写了怎么做。
3. **判据 / predicate**（主张）。什么时候走融合路径而不是库路径。这就是 8-06 的 execution model 契约的另一半——**契约规定通信与计算如何交错，判定层规定何时启用哪种交错。**

## 为什么这不是补丁：有一个组织必须做决定的靶子

**DSV4 的 11 个 attention 后端**（`Primus/primus/backends/megatron/core/transformer/v4_attention_kernels/`：`_eager` / `_triton_v0_deprecated` / `_triton_v1` / `_triton_v2` / `_gluon_dsa` / `_gluon_v2` / `_gluon_v3` / `_flydsl_v0_deprecated` / `_flydsl_v1` / `_tilelang` / `_turbo_flydsl`），每一路都带 backward，2026-07-17 起四个 PR 迭代到第三代，目录 README 已经过期（未收录 gluon_v2/v3、flydsl_v1、turbo）。

**现在没有任何人能说清哪个赢。** 这个决定本季度必须有人做，而做出它就等于拥有它。

且 harness 的 80% 已经在手：`benchmark/ops/training/bench_attention_dsv3.py` 的「每后端独立子进程（避开 import 前环境变量污染）+ 对 fp32 SDPA 算 SNR + 版本环境固化」正是裁决 11 个后端所需的形状。

## 分期规划

| 阶段 | 内容 | 验收门槛 | 成本 |
|---|---|---|---|
| **J0** | **噪声地板协议**。刻画 MI355X 在 keepalive 下的 run-to-run 分布，定出"最小可判差异"；把环境指纹写进 `run_suite.py` 的 CSV header | 能声明"本平台上 X% 以下的差异不可判"，并被别人复用 | 1–2 周 |
| **J1** | **DSV4 11 后端裁决**。扩 `bench_attention_dsv3.py` 到 DSA 三形状（cr=0 / 128 / 4）× 11 后端 × 生产 shape，出权威对比 + 数值 SNR | 一份能直接决定 `use_v4_attention_backend` 默认值的报告；结论带 J0 的可判性标注 | 2–4 周 |
| **J2** | **attention 接入 `AutoKernelDispatcher`**。补上 `PRIMUS_TURBO_ATTN_BACKEND`，把 J1 的结论固化成 tuned CSV 入仓 | 官方 TODO 关闭；后端可 A/B 且默认值有数据支撑 | 2–3 周 |
| **J3** | **判定数据扩到 MoE 训练路径**。grouped GEMM / dispatch-combine / mega-moe 的 `(regime) → backend` 表，CI 周期刷新 | tuned CSV 成为版本化资产；回归可自动发现 | 1–2 月 |
| **J4** | **predicate**：融合路径 vs 库路径的判据，落进 8-06 的 execution model | 512 t/g 赢、2048 t/g 输这类 regime 反转能被判据提前预测 | 与轨道 A 合流 |

J0→J2 是 2 个月量级、全程有可交付、且每一步都在关别人的 TODO 或做别人在等的决定。J3→J4 才是纵深。

## 与 8-06 的接口

8-06 把测量可信度列为"轨道 A 开工前必须先解决的前置项"。本篇的修正是：**它不是前置项，它是主线的第一段。** 理由是判定层同时满足 8-06 里的三个要求——

- 建在最强能力上（算子 + 框架双前二）
- 不依赖 Pilot（反而**解掉** Pilot 卡 15 个月的 BASELINE 非确定性，同一个根因）
- 不依赖多节点集群（单机就能做）

而 8-06 的轨道 B（影响力，便宜并行）与它天然复合：判定数据本身就是"连续、可复现、带 trace 的公开性能记录"最好的载体。

## 什么会证伪这个主张

- **噪声地板降不下来。** 若 J0 结论是"MI355X 上 <15% 不可判"，则大量 kernel 优化工作在这个平台上无法被裁决，判定层的价值也随之下降。这是真实风险，`notes/pilot/` 卡 15 个月就是先例。
- **组织把裁决权收走。** 若后端默认值由管理层按供应商关系而非数据决定，则判定数据不产生权力。缓解：先用 J1 把数据摆到桌上，数据先于权力。
- **11 后端的赛马自然收敛。** 若 gluon_v3 在本季度直接胜出并被定为默认，J1 的裁决价值缩水。缓解：J1 的产出物是**协议 + 可复跑的 harness**，不只是这一轮的表；下一个 op 还要用。
- **被读成 tooling 而非技术纵深。** 缓解：交付物必须是**数字与规则**，harness 是副产品。对外叙事是"我刻画了 AMD 训练算子的选择规律"，不是"我写了个 benchmark 框架"。

## 与另一条高地的关系

真正无人占据、且 AITER / FlyDSL / HipKittens **结构上都到不了**的地方是**多节点**：AITER 是单机 op 库，FlyDSL 是单卡 kernel，HK 论文完全没有多卡内容（`papers/hipkittens.md` §9 批判 3），Turbo 也以单机为主。而 8-06 认定的四条缝在跨节点上全部恶化。

但 8-06 已记录"目前没有多节点集群"。所以定位是：**多节点是判定层长大之后的延伸方向，不是现在可立项的事。** 判定层的好处是它天然向那里扩——判据一旦要跨节点，就必须先有跨节点的测量与判定数据。

## 下一步

**P0**：J0 噪声地板。先只做一件事——在 keepalive 下把同一个 workload 重复 30+ 次，出 run-to-run 分布，定"最小可判差异"。这一步不写 kernel、不碰 CI，一两天就能有第一版数字，且它决定后面所有阶段的可信度。

**P1**：把 J1 的范围与 Primus 侧对齐——确认 `use_v4_attention_backend` 的默认值决定权在谁手上，以及本季度是否真的需要这个决定（若不需要，J1 降级为 J3 的预演）。

## 相关文件

- [`2026-08-06_primus-positioning-boundary-dissolution.md`](./2026-08-06_primus-positioning-boundary-dissolution.md) — 本篇的母文；轨道 A/B、四条缝、execution model
- [`../hk-attn-bwd/2026-08-12_1354_turbo_attention_ground_truth.md`](../hk-attn-bwd/2026-08-12_1354_turbo_attention_ground_truth.md) — Turbo attention 无 dispatcher、DSV4 11 后端、FlyDSL↔AITER 关系的证据来源
- [`../../knowledge/libraries/aiter.md`](../../knowledge/libraries/aiter.md) — §1 "AITER 不写 kernel"、§3.3 autotuning 入 CI、§4 三行 applies-to-primus-turbo
- `notes/MegaMoeFlydsl/mxfp8_moe_bwd_perf_summary.md` — sclk ±30% 噪声与 `setperflevel` 不可用
- `notes/monolith-moe/2026-05-13_2340_apples_to_apples_super_kernel_loses_at_training_scale.md` — regime 反转，predicate 存在的理由
- `notes/moe_perf/turbo/README.md` — dispatch+combine 占 MoE 层 40–60%
- Primus-Turbo：`benchmark/ops/training/{run_suite.py,benchmark_suite.yaml}`、`primus_turbo/pytorch/core/backend.py`、`agent/skills/`
