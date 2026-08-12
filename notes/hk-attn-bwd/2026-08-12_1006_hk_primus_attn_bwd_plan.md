# HipKittens 切入 Primus：attention backward 缺口分析与规划

**日期**: 2026-08-12 10:06

## 背景 / 目标

对 HipKittens（下称 HK）有持续投入的意愿，需要找一条**既能在 HK 上积累技术资产、又能给 Primus 带来可度量收益**的路径，而不是做一个孤立的复现或调研。

本文回答三个问题：切口在哪、为什么这个切口是结构性的而非偶然的、分几步走。

结论先行：**切口是 attention backward。** 它是 Primus 作为训练框架的刚需，是 HK 已发表的最强项，而 FlyDSL——当前 Primus 的外部内核供给方——在这一块是完全空白。

---

## 主要发现 / 结论

### 0. 重要修正（2026-08-12 10:56 补）：论文的 AITER 基线已过时

拿到 Primus-Turbo 内部性能页的实测数据（AMD Confluence，页面需登录，本节数据来自截图）后，**本规划的量化前提必须下修**。

BF16、`[B, S, H, D]`、单位 TFLOPS：

| 形状 | causal | FA-v3 H200 F/B | FA-v4 B200 F/B | Turbo(AITER) MI300X F/B | **Turbo(AITER) MI355X F/B** |
|---|---|---|---|---|---|
| [1, 4096, 32, 128] | True | 615 / 489 | 1042 / 228※ | 433 / 267 | **991 / 902** |
| [8, 4096, 32, 128] | True | 563 / 515 | 1190 / 868 | 448 / 360 | **1046 / 925** |
| [1, 8192, 64, 128] | True | 643 / 515 | 1325 / 954 | 426 / 403 | **1110 / 1047** |
| [8, 8192, 64, 128] | True | 618 / 530 | 1252 / 991 | 480 / 401 | **1119 / 1026** |
| [1, 65536, 3, 128] | False | — | 1309 / 1102 | — | **1196 / 1221** |
| [1, 65536, 8, 128] | False | — | 1381 / 1139 | — | **1201 / 1208** |
| [4, 131072, 3, 128] | False | — | 1372 / 1127 | — | **1224 / 1212** |
| [4, 131072, 8, 128] | False | — | 1398 / 1141 | — | **1249 / 1193** |

※ B200 该格 BWD=228 明显偏离同列其他行（868–991），疑为坏点，不作为有效数据。

**三条结论：**

1. **HK 相对 AITER 的优势已基本消失。** 论文报的 AITER backward（seqlen 8192：causal 272 / non-causal 384）与本表 **MI300X** 列（403 / 401）吻合，而非 MI355X 列。MI355X 上 AITER backward 已达 **902–1047**（causal），而 **HK 论文 register pinning 后的最好成绩是 1024**。即：**AITER 现在持平或反超 HK 的已发表数字**。
2. **M0 及格线从「打败 272–403」变为「打败约 1050」**，是完全不同量级的挑战。
3. **方向性反转**：非 causal 长序列上 MI355X 的 **BWD 反超 B200**（1193–1221 vs 1102–1141），但 **FWD 落后**（1196–1249 vs 1309–1398）。**AMD 当前落后 B200 的是前向而非反向**——而前向是 FlyDSL 的地盘。

**仍未被这张表覆盖、因而 M0 仍有信息价值的两点：**

- **无 GQA 比例**：列头只有一个 `H`，应为 MHA。生产是 4:1 / 8:1 / **16:1**（Qwen3-235B），高 GQA 比下 K/V 复用模式不同，AITER 表现未知。
- **causal + 中等序列是全表最弱格**（[1,4096,32,128] 的 902），而 4096 正是 Qwen3-235B-A22B 的生产 `seq_length`。
  **注意混淆**：causal 行序列长为 4096/8192，non-causal 行为 65536/131072，长序列本身摊薄开销更好，故 902–1047 vs 1193–1221 不能干净归因于 causal，需同序列长的对照实验才能定论。

**待落实**：该表的 ROCm / AITER / Turbo 版本与测试日期未知，需向页面维护者确认后回填，否则无法判断其时效性。

### 0.1 再次修正（2026-08-12 11:54 补）：DSV3 实测报告，建议本项目转入 on-hold

依据自测报告《DeepSeek-V3 8K Attention 后端对比（MI355X）》（Primus-Turbo `main` @ `bdd96e69`，2026-08-12，原始数据 `/perf_apps/xiaoming/attn_dsv3_8k_all.csv`）。

**(a) HK 结构上跑不了 DSV3 的 MLA 形状——源码层已确认。**

DSV3 训练形状为 `head_dim_qk=192`（nope 128 + rope 64）/ `head_dim_v=128`，qk 与 v 头维不等。而 HK 只有单一头维常量：

```cpp
// hipKitten/training/llama/csrc/attn_fwd_causal.cpp:22（attn_bkwd_prep.cpp:22 同）
constexpr int ATTN_D = 128; // dimension
```

K 与 V 的 shared tile 复用同一个 `ATTN_D`（`attn_fwd_causal.cpp:165-166`），无 D_QK / D_V 之分。HK 与报告 §3.4 中跑不了该形状的三个后端属同一失败类别：

| 后端 | 失败原因 |
|---|---|
| Dao-AILab flash-attn 2.8.3 | 要求 `head_dim_v == head_dim_qk` |
| TE AOTriton | 无可用后端，仅 CK 路径支持头维不等 |
| 原生 `aiter.flash_attn_func` | 断言两头维相等 |
| **HipKittens** | **只有单一 `ATTN_D=128`** |

→ **MLA 出范围的结论从「推断」升级为「源码确认」。**

**(b) 剩余空间在配置层，不在内核层。**

报告的两项主要发现均为配置/版本问题，收益远超 HK 的潜在贡献且成本极低：

| 问题 | 收益 | 性质 |
|---|---|---|
| aiter pin (`v0.1.14.post1`) 过期，causal 前向拿不到 causal 加速（5.19 → 2.75 ms） | **1.89×** | 版本 pin + 去掉 `out=` |
| `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` 默认 `1`，gfx950 上是慢路径（14.35 → 8.61 ms） | **1.68×** | 默认值按 arch 决定 |

**配好后 turbo 与 TE 打平**（11.29 vs 11.10 ms，< 2%），因两者跑同一批 aiter/CK 汇编 kernel。即该形状上**内核层已被 CK 汇编吃满**。

**(c) 唯一仍待查的薄弱格子：低头数反向。**

报告 §3.3：16 heads（TP=8 每卡份额）时前向仍 ~1000 TFLOPS，**反向掉到 ~655 TFLOPS**，报告归因为并行度不足。

**但机制存疑**：16 heads × b=1 × seq 8192 按 KV block 切分后仍有约 10³ 量级 workgroup，对 256 CU 不算少；且反向绝对时间仅 1.31 ms，更可能是**固定开销（launch、dQ 原子归约）占比放大**而非调度不良。

→ 这个区别决定 HK 是否有用：调度问题 HK 的 ping-pong / register pinning 能帮；原子归约或 launch 开销则完全帮不上。**先用 rocprof 确认机制，成本远低于上 HK。**

**(d) 结论：本项目转入 on-hold。**

三轮证据累积后，HK 在 dense attention backward 上的边际价值已很低：论文的缺口前提（AITER 30%）在 MI300X 级软件上才成立；MI355X 上 AITER 已达 902–1047（MHA/GQA）；MLA 形状 HK 无法运行；配置修好后 turbo≈TE。

**保留一个低成本的收口动作**（见「下一步」P0'），确认两个未测格子后再决定彻底关闭还是重启。

### 1. 缺口是实在的：FlyDSL attention 全栈无 backward

`FlyDSL/kernels/attention/` 下 15 个文件，全部是 forward 或 decode：

| 类别 | 文件 |
|---|---|
| flash attention 前向 | `flash_attn_generic.py`、`flash_attn_gfx950.py`、`flash_attn_fp8_gfx950.py` |
| MLA decode | `mla_fwd_decode.py`、`mla_fwd_decode_m16x8_fp8_fp8.py` |
| paged attention decode | `pa_decode_fp8.py`、`pa_decode_swa.py`、`pa_decode_tile.py`、`pa_metadata.py`、`pa_common.py` |
| 融合算子 | `fused_rope_cache_kernel.py`、`qk_norm_rope_quant.py` |

全仓搜 `backward|bwd|dq|dkv` 只命中一处，是前向 kernel 的参数注释：

> `# [B, num_heads, Sq]. Needed by backward; not supported for fp8.`（`flash_attn_interface.py:620`）

即 FlyDSL 只负责**产出 backward 所需的 LSE**，把 backward 本身留给了别人。这一点与 [`megaattn/README.md`](../megaattn/README.md) 里已经记过的「attention 全栈无 backward」相互印证。

### 2. HK 的最强项恰好就是这块，且桥已搭好一半

| HK 资产 | 路径 |
|---|---|
| GQA backward 内核（含 causal） | `kernels/cdna4/attn/{gqa_backwards, gqa_causal_backwards}` |
| 编译成 PyTorch 扩展 | `training/llama/csrc/{attn_bkwd_causal_HBN.cpp, attn_bkwd_causal_HNB.cpp, attn_bkwd_prep.cpp}` |
| torch autograd 封装 | `training/llama/llama/models/attentions/hipkittens.py` |
| **对 AITER 的 A/B 对照已内置** | 同目录并排的 `base.py` / `aiter.py` |

最后一条是关键：HK 的训练 harness 本身就是**一个可插拔的 attention 后端抽象，且已有 AITER 作为对照组**。这与 Primus 的 `use_turbo_attention` 开关是同构的，M0 几乎不需要额外搭台。

论文侧的支撑数据：AITER 的 GQA backward 只有 SoTA 的 30%，PyTorch SDPA 24%；HK 靠 register pinning 把 backward 从 855 推到 **1024 TFLOPS**。

### 3. 通道已存在，且有先例

Primus 生态分三层：Primus-LM（训练框架）/ **Primus-Turbo（高性能算子）** / Primus-SaFE（稳定性平台）。Turbo 通过一组开关挂载到 LM：

```yaml
# primus/configs/modules/megatron/primus_turbo.yaml
enable_primus_turbo: false
use_turbo_attention: false
use_turbo_gemm: false
use_turbo_grouped_gemm: false
use_turbo_deepep: false
use_turbo_rms_norm: false
```

**先例是 FlyDSL**（Primus commit `fa391f32`，AMD Kyle Zhao，2026-07-28）：

> Point the low-precision section at the FlyDSL repo as the source of the FP8 GEMM and grouped-GEMM kernels, and add the FlyDSL team to the acknowledgments.

模式很清楚：**外部 DSL 编写内核 → 落进 Primus-Turbo → Primus-LM 用 flag 挂载 → 文档致谢作者团队**。HK 走同一条路，换成 attention backward。

### 4. 为什么这个缺口是结构性的，不会被 FlyDSL 顺手补掉

backward 是 attention 里**寄存器压力最高**的环节——dQ / dK / dV 三个累加器同时存活。而这恰好是：

- HK **唯一独有**能力（register pinning，可把 tile 钉到指定 VGPR/AGPR）的用武之地；
- FlyDSL「fragment 声明 + 交给 LLVM 分配，用户只能给 `waves_per_eu` 提示」这一设计选择**最吃亏**的地方。

FlyDSL 要补上这块，得引入一个与其 CuTe 式布局代数哲学相冲突的逃生舱。所以这不是抢地盘，是**补一块对方因为设计取向而不会去补的洞**——这样的定位才可持续。

---

## 详细分析：形状缺口表

这是 M0 必须先解决的问题。HK 的内核用**编译期模板参数**实例化，每个形状要重新编译：

```makefile
# hipKitten/training/llama/csrc/Makefile:11-14
ATTN_B    ?= 8
ATTN_H    ?= 64
ATTN_H_KV ?= 8
ATTN_N    ?= 2048
```

```bash
# hipKitten/training/llama/csrc/setup_kernels.sh:5-7 —— 实际只构建了一组
make SRC=attn_bkwd_causal_HBN.cpp TARGET=tk_kernel_bkwd ATTN_B=8 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
```

**一个有利的发现**：Makefile 的默认值 `ATTN_H=64 / ATTN_H_KV=8` 正好就是 Llama3-70B 的形状，说明 GQA 路径本身是支持的，`setup_kernels.sh` 建的 `16/16` 只是它自测用的 MHA 配置。这把 M1 的风险降了一档。

### Primus 生产形状（从 model config 抽取）

| 模型 | H_Q | H_KV | GQA 比 | head_dim | seq_length | 配置来源 |
|---|---|---|---|---|---|---|
| Llama3.1-8B | 32 | 8 | 4:1 | 128 | 2048 (sft) / max 131072 | `models/megatron/llama3_8B.yaml` |
| Llama3.1-70B | 64 | 8 | **8:1** | 128 | max 131072 | `models/megatron/llama3_70B.yaml` |
| Qwen3-235B-A22B | 64 | 4 | **16:1** | 128 (`kv_channels`) | 4096 | `models/megatron/qwen3_235B_A22B.yaml` |
| DeepSeek-V3 | 128 | **MLA** | — | qk 128 / v 128 | 4096 | `models/megatron/deepseek_v3.yaml` |

DeepSeek-V3 走 MLA（`q_lora_rank: 1536`、`kv_lora_rank: 512`），**不是 GQA，HK 没有对应的 backward**。这是必须承认的范围限制：

> **本项目范围限定在 Llama / Qwen 系的 GQA backward，不覆盖 DeepSeek MLA。**

MLA backward 是另一条线，与 [`megaattn`](../megaattn/README.md) 有交集，本项目不并入。

### 缺口小结

| 维度 | HK 现状 | Primus 需求 | 差距 |
|---|---|---|---|
| GQA 比例 | 模板支持，实建 1:1 (MHA) | 4:1 / 8:1 / 16:1 | 需按比例实例化验证 |
| 序列长 | 建了 2048 | 2048 / 4096 / 8K+ | 需扩展 + 变长支持 |
| batch | 编译期固定 B=8 | 变动 (mbs 1/2) | **编译期固定是主要障碍** |
| causal | 有（causal + non-causal） | 训练需 causal | 覆盖 |
| MLA | 无 | DeepSeek-V3 需要 | **不在范围内** |

---

## 阶段计划

| 阶段 | 内容 | 验收门槛 |
|---|---|---|
| **M0** | 用 HK 现成 harness 在 MI355X 上跑 HK vs AITER 的 attention backward，**用 Primus 生产形状（重点补 GQA 4:1/8:1/16:1，及 causal @ seq 4096）** | 拿到形状 × 后端 TFLOPS 表；**及格线为 AITER 的 ~1050（非论文的 272–403）**；若无任一生产格子上 HK 明显占优则就地叫停 |
| **M1** | 形状覆盖：GQA 4:1/8:1/16:1 × seq 2048/4096/8192，解决 batch 编译期固定 | 生产形状全绿，数值与 AITER 对拍通过 |
| **M2** | 接入 Primus-Turbo，挂到 `use_turbo_attention` 后 | 端到端 Llama3.1-70B / Qwen3-235B 训练步时下降可度量 |
| **M3** | 研究增量：XCD 感知调度 + register pinning 在 backward 上的结合 | 相对 M1 有增量收益；目标可发表 |

### M0 是最重要的一步

论文的数字是在作者选定的形状上测的，与 Primus 生产形状不一定重合。**如果 HK 在你们的真实形状上赢不了 AITER，后面所有事情的前提都变了。** 所以 M0 必须最先做，且门槛要设成"可以叫停"。

产出格式建议直接套 Primus 仓库里已有的 `skills/backend-gap-report/SKILL.md`——那是团队内已经存在的沟通语言，比自造格式更容易被接受。

### M3 的研究机会

[Swizzled Head-first Mapping 那篇](../../papers/swizzled-head-first-attention.md)在 backward 上只拿到 **1.10×**，作者明确写了「怀疑本优化引入了新瓶颈，留给未来工作」。而 HK 有 register-pinned 的 backward。

**XCD 感知调度与寄存器钉住在 backward 上的结合是一个公开空白**：有明确的先行工作可引、有真实的收益空间、正好落在本项目的主线上。这件事同时满足「HK 上的积累」「Primus 的收益」「可发表」三个条件。

---

## 风险与对冲

| 风险 | 说明 | 对冲 |
|---|---|---|
| **⚠ 论文的价值主张已被实测数据削弱**（见 §0） | AITER 在 MI355X 上 backward 已达 902–1047，持平/反超 HK 论文的 1024。「AITER backward 很差」这一前提**只在 MI300X 级软件上成立** | M0 及格线上调到 ~1050；把 M0 的目的从「确认 HK 赢」改为「**找出是否存在仍然薄弱的生产格子**」（高 GQA 比、causal @ 4096）。找不到就停 |
| **AMD 投入方向在 FlyDSL** | FlyDSL 是 ROCm 官方（781 commits / ~60 作者 / PR 到 #956），已进 Primus 文档致谢；HK 是斯坦福研究产物 | **把资产定义为「AMD attention backward 的专家能力」而非 HK 代码库**。寄存器钉住技法、backward 调度算法、形状覆盖经验换框架都带得走 |
| HK 编译期模板 | 每形状重编译，生产要变长/多形状 | 这正是 M1 的主要工程量，也是最难被替代的积累 |
| 范围限制 | DeepSeek-V3 走 MLA，不覆盖 | 明确写进范围声明，不承诺 |
| HK 主干不完整 | FP6 GEMM 只在 `fp6_experimental` 分支，论文附录 F 的结果主干不可复现 | 不要把论文全部结果当作主干可复现 |

**长期的一个可能性**：若对 register pinning 理解足够深，可以把它作为逃生舱**反向提案给 FlyDSL**。那是跨项目的高价值贡献，也是对冲"押错框架"风险的最好方式。

---

## 下一步 / 建议

> **状态已变更为 on-hold（见 §0.1）。以下 P0'/P1' 是收口动作，不是原计划的 M0。**

**P0'（收口，约 1 天）**：用 rocprof 确认 16-head 反向 655 TFLOPS 的**机制**——是调度不良，还是 dQ 原子归约 / launch 固定开销占比放大。这一条直接决定 HK 有没有落点。

**P1'（收口，约 1 天）**：补测 Confluence 表未覆盖的 **GQA 比例**（4:1 / 8:1 / **16:1**）下 AITER 的 causal backward。若某个高 GQA 比格子明显偏弱，才有重启本项目的理由。

**决策点**：P0' 若确认是固定开销问题，且 P1' 未发现薄弱格子 → **彻底关闭本项目**，HK 的投入转向非 dense-attention 方向。

**与本项目无关但优先级更高的两件事**（来自同一份报告，收益远大于本项目）：
1. 升 aiter pin 并去掉 `out=` —— 所有形状的 causal 前向 1.89×
2. gfx950 上 `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` 默认改 `0` —— 反向 1.68×

---

## 相关文件

**HipKittens**
- `~/workspace/hipKitten/kernels/cdna4/attn/{gqa_backwards, gqa_causal_backwards}` — backward 内核
- `~/workspace/hipKitten/training/llama/csrc/` — PyTorch 扩展与构建脚本
- `~/workspace/hipKitten/training/llama/llama/models/attentions/` — 可插拔后端 + AITER 对照

**Primus**
- `primus/configs/modules/megatron/primus_turbo.yaml` — Turbo 开关
- `examples/customer_package/run_qwen3_235b_a22b_pretrain_mi355x.sh` — 生产配置
- `primus/configs/models/megatron/{llama3_70B, qwen3_235B_A22B, deepseek_v3}.yaml` — 模型形状
- `skills/backend-gap-report/SKILL.md` — M0 产出格式
- commit `fa391f32` — FlyDSL 供给内核的先例

**FlyDSL**
- `~/workspace/FlyDSL/kernels/attention/` — 确认无 backward 的现场

**slab**
- [`papers/hipkittens.md`](../../papers/hipkittens.md) · [全文中译](../../papers/hipkittens-zh.md)
- [`papers/swizzled-head-first-attention.md`](../../papers/swizzled-head-first-attention.md) — M3 的先行工作
- [`knowledge/systems/primus-pipeline-runtime-megatron-integration.md`](../../knowledge/systems/primus-pipeline-runtime-megatron-integration.md)
