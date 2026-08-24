# Primus-Turbo attention 代码核查：对 HK 切入计划的五处更正

> **When**: 2026-08-12 13:54 UTC+8
> **Where**: 登录机，纯静态代码核查（未跑 GPU）
> **Context**: 复核 [2026-08-12_1006_hk_primus_attn_bwd_plan.md](./2026-08-12_1006_hk_primus_attn_bwd_plan.md) 中关于 attention 的事实前提

## TL;DR

原计划关于 attention 的五条事实前提有四条不成立、一条有重大遗漏。核心两点：

1. **「FlyDSL attention 全栈无 backward、且因设计取向不会去补」是错的。** FlyDSL 写的 attention backward 已经存在，只是不落在 FlyDSL 仓库里——`Primus-Turbo/primus_turbo/flydsl/attention/sparse_mla_bwd.py` 有 2759 行，直接用 `rocdl.mfma_*` / `rocdl.s_setprio` / 手工 LDS 分配写成，抽象层级**比 HK 更低**；FlyDSL 仓库里还并排放着 `fp8_gemm_8wave.py` 和 `fp8_gemm_4wave.py`，正是 HK 论文的两个调度。原计划「结构性缺口」的两条论据同时失效。
2. **切口选在了一个已经结束的战场。** Primus 栈上 attention 的活跃前沿是 DeepSeek-V4 sparse MLA（DSA），`Primus/.../v4_attention_kernels/` 下有 11 个并列后端目录（eager、Triton v0/v1/v2、Gluon dsa/v2/v3、FlyDSL v0/v1、TileLang、turbo_flydsl），**每一路都带 backward**，2026-07-17 起四个 PR 连续迭代到第三代。而 dense GQA/MHA backward 这块已被 aiter/CK 汇编吃满。

另外原计划把接入成本算低了：`use_turbo_attention` 不是一个 attention 内核后端开关，Turbo 内部**没有**后端派发机制（两处 `TODO(ruibin): Add unified attention kernel dispatcher` 至今未做）。这反过来是一个真实、低风险、已被官方 TODO 认领的可贡献点。

## 背景

原计划的 attention 论证链是四段：(1) FlyDSL attention 无 backward → 存在缺口；(2) backward 是 HK 独有的 register pinning 的用武之地、也是 FlyDSL 设计最吃亏处 → 缺口是结构性的；(3) `use_turbo_attention` 开关 + FlyDSL 先例 → 通道现成；(4) Primus 生产形状是 Llama/Qwen GQA + DSV3 MLA → 需求明确。

§0 / §0.1 两轮修正已经推翻了性能前提（AITER 在 MI355X 上 backward 已达 902–1047，且 MLA 形状 HK 源码层跑不了），但**没有复核前提 (1)(3)(4)**——那三条是从仓库目录结构和 YAML 开关推断的，没有读过 Primus-Turbo 的 attention 实现，也没有读过 Primus 侧的 attention 后端注册表。本次核查补这一块。

## 核查范围与版本

| 仓库 | 路径 | commit | 日期 |
|---|---|---|---|
| Primus-Turbo（旧 clone，即 `slab/Primus-Turbo` 符号链接目标） | `/perf_apps/xiaoming/Primus-Turbo` | `9c1e61c1` (#440) | 2026-07-31 |
| Primus-Turbo（新 clone，DSV3 报告的基线） | `/perf_apps/xiaoming/MegaMoE` | `bdd96e69` (#441) | 2026-08-11 |
| Primus | `/perf_apps/xiaoming/Primus` | `a6c7fcd8` | 2026-08-06 |
| FlyDSL（本地 checkout，可能落后 upstream） | `/perf_apps/xiaoming/flydsl` | `1afc0b0f` (#788) | — |

两个 Primus-Turbo clone 的 attention 源码 **逐字节相同**（`diff -rq` 对 `primus_turbo/{flydsl,pytorch/ops,pytorch/kernels}/attention` 均为空），所以下文行号在报告基线 `bdd96e69` 上同样成立。

---

## 更正一：FlyDSL 的 attention backward 已经存在，只是不在 FlyDSL 仓库里

原文（§1）：「`FlyDSL/kernels/attention/` 下 15 个文件，全部是 forward 或 decode…全仓搜 `backward|bwd|dq|dkv` 只命中一处」。

**对仓库的观察是对的**：本地 checkout `1afc0b0f` 的 `kernels/attention/` 仍是 14 个 fwd / decode / 融合算子文件，无 attention backward。

**但推论错了**——错在把「FlyDSL 仓库的内容」当成「用 FlyDSL 写出来的东西」。FlyDSL 是 DSL 工具，kernel 写在下游仓库：

| 位置 | 文件 | 行数 | 内容 |
|---|---|---|---|
| Primus-Turbo | `primus_turbo/flydsl/attention/sparse_mla_bwd.py` | 2759 | DSV4 sparse-MLA 反向，产出 `dq` / `dkv` / `d_sink`（`:2758-2759`） |
| Primus-Turbo | `primus_turbo/flydsl/attention/sparse_mla_fwd.py` | 1068 | 同形状前向 |
| Primus | `v4_attention_kernels/_flydsl_v1/dsa_bwd_dq_flydsl_kernel.py` | 595 | DSA 反向 dQ |
| Primus | `v4_attention_kernels/_flydsl_v0_deprecated/kernels/v4_*_bwd_*.py` | 9 个文件 | SLA / CSA / HCA 反向（已弃用） |

作者与时间：Primus-Turbo 的两个文件由 **kyle-256 (Kyle.Zhao@amd.com)** 在 `b919358f` / **2026-07-20** / `[feat] add triton & flydsl impl for sparse attention (#420)` 一次引入，`502f3f35`（07-28，#431）补 FlyDSL 的 Apache-2.0 许可头。**这正是原计划 §3 引为「先例」的那位 AMD Kyle Zhao。** 即先例的真实形态不是「FlyDSL 供给 FP8 GEMM」这一次性事件，而是「FlyDSL 团队持续在 Turbo/Primus 里写 kernel，含 attention 反向」。

顺带一条：FlyDSL 仓库里有 `kernels/norm/rmsnorm_bwd_kernel.py`，所以「FlyDSL 不写 backward」在 DSL 能力层面也不成立。

## 更正二：FlyDSL 的抽象层级比 HK 更低，"结构性缺口"的论据不成立

原文（§4）：「FlyDSL『fragment 声明 + 交给 LLVM 分配，用户只能给 `waves_per_eu` 提示』这一设计选择最吃亏……FlyDSL 要补上这块，得引入一个与其 CuTe 式布局代数哲学相冲突的逃生舱。」

这是把 Triton 的能力边界套到了 FlyDSL 上。`sparse_mla_bwd.py` 的热循环（`:446-509`）实际长这样：

```python
acc_s0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), acc_s0])
acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), acc_dp0])
```

配套用到的东西：显式 `rocdl.ds_read_tr16_b64` 转置读、`rocdl.s_setprio(1/0)` 包住 MFMA 簇（`:1278, :1327, :1515, :1529`）、手工 `SmemAllocator` + `D_LDS=528` padding 消 bank conflict、按 build 参数调的多级 prefetch 深度（`pf_qk` / `pf_pv`）、多累加器拆 RAW 链、XCD-aware token remap。

而 FlyDSL 仓库里已经有 HK 论文的两个核心调度，是文件名级别的直接对应：

| HK 论文贡献 | FlyDSL 仓库对应物 |
|---|---|
| 8-wave ping-pong | `kernels/gemm/fp8_gemm_8wave.py`、`kernels/conv/conv3d_implicit_8wave{,_fp8}.py` |
| 4-wave interleave | `kernels/gemm/fp8_gemm_4wave.py`、`kernels/gemm/fp4_gemm_4wave.py` |
| `s_setprio` 包 MFMA 簇 | `kernels/attention/flash_attn_interface.py` 的 `dualwave_swp_setprio` 旋钮（`:359, :609`） |
| chiplet 感知 grid 重排 | `sparse_mla_fwd.py` 的 `xcd_remap` build 参数 |

**结论**：HK 相对 FlyDSL 的独有能力被压缩到 **register pinning 一项**，而不是「整个 backward 领域」。原计划把「HK 唯一独有能力」与「一整块无人区」画了等号，这个等号不成立。

## 更正三：`use_turbo_attention` 不是内核后端开关，M2 的工程量被严重低估

原文（§3）：「Turbo 通过一组开关挂载到 LM……模式很清楚：外部 DSL 编写内核 → 落进 Primus-Turbo → Primus-LM 用 flag 挂载」。

实际的 Turbo dense attention 供给链只有一条真实路径，且**内部没有派发层**：

| 层 | 事实 | 证据 |
|---|---|---|
| 派发策略 | 唯一分支是 `sink is None`：None 走 aiter csrc/CK，非 None 走 aiter Triton | `attention_aiter_impl.py:10-13` 的策略注释 + `:146-195` / `:265-324` |
| 底层入口 | 固定 `aiter.ops.mha._flash_attn_{forward,backward}`（varlen 同族） | `attention_aiter_impl.py:33-44` |
| CK vs 汇编 v3 | **Turbo 不选**，只透传 `is_v3_atomic_fp32` / `how_v3_bf16_cvt`，选择发生在 aiter 内部 | `attention_aiter_impl.py:286-287` |
| 环境变量后端选择 | **不存在**。`common/constants.py` 只有 `PRIMUS_TURBO_{GEMM,GROUPED_GEMM,MOE_DISPATCH_COMBINE}_BACKEND`（`:21,25,29`），attention 相关只有 `ENV_ATTN_V3_ATOMIC_FP32`（`:37`），那是数值精度旋钮不是后端选择 |
| 自动派发框架 | attention 的 `KernelBackend` 子类**未注册进** `AutoKernelDispatcher`，被自己的 `custom_op` 直接调用 | `attention_aiter_impl.py:374-380` |
| 官方待办 | 两处 `# TODO(ruibin): Add unified attention kernel dispatcher` | `attention_aiter_impl.py:373`、`:478` |

所以要把任何第三方 attention 内核（HK 或别的）挂进 Turbo，**第一步是先给 Turbo 建 attention 后端派发器**——这件事在原计划 M2「接入 Primus-Turbo，挂到 `use_turbo_attention` 后」里完全没被计价。

反过来看，这是本次核查里性价比最高的发现：**它是一个官方 TODO 已认领但未做的缺口，纯框架层、零内核风险、且是任何后续内核工作的前置件。**

## 更正四：漏掉了 Primus 上真正活跃的 attention 前沿——DSV4 sparse MLA

原计划的形状表只列了 Llama3.1-8B/70B、Qwen3-235B、DeepSeek-V3。实际 Primus 上 attention 的开发热度全部集中在 **DeepSeek-V4**，且是一个多 DSL 赛马场：

`primus/backends/megatron/core/transformer/v4_attention_kernels/` 下的后端目录：

| 后端 | 目录 | backward | 说明 |
|---|---|---|---|
| eager | `_eager/` | 有 | PyTorch 参考基线 |
| triton_v0 | `_triton_v0_deprecated/` | 有 | 已弃用，CSA only，标量 GEMV（慢 30–260×）|
| triton_v1 | `_triton_v1/` | 有 | **生产默认** |
| triton_v2 | `_triton_v2/` | 有 | 融合单 latent sparse-MLA，`tl.dot` |
| gluon | `_gluon_dsa/` | 有 | 手调，gfx950 only |
| gluon_v2 | `_gluon_v2/` | 有 | 第二代 fwd+bwd |
| gluon_v3 | `_gluon_v3/` | 有 | 第三代，含 `aiter_lse_fwd.py` / `aiter_mla_gluon.py` |
| flydsl_v0 | `_flydsl_v0_deprecated/` | 有（9 个 bwd 文件） | 已弃用 |
| flydsl_v1 | `_flydsl_v1/` | 有 | 原生 FlyDSL MFMA |
| turbo | `_turbo_flydsl/` | 有（转调 Turbo 的 `sparse_mla_bwd`） | 绑到安装的 primus_turbo |
| tilelang | `_tilelang/` | 有 | 实验，未接选择器 |

三种形状按 `compress_ratio` 逐层选：`cr=0` dense/SWA、`cr=128` HCA（层级压缩池，joint softmax）、`cr=4` CSA（per-query top-K gather + joint softmax）。派发优先级 `use_turbo_attention > use_v4_triton_attention > eager`（`deepseek_v4_attention.py:689-693`）。

迭代节奏（`git log -- v4_attention_kernels/`）：

| commit | 日期 | 作者 | 主题 |
|---|---|---|---|
| `4999e928` (#882) | 2026-07-17 | wenxie-amd | Add DeepSeek-V4 training support（模型 + attention/MoE 内核 + Muon + FP8/FP4） |
| `e71393dc` (#898) | 2026-07-23 | wenxie-amd | Update turbo flydsl sparse attn |
| `0523e9da` (#891) | 2026-07-28 | RuibinCheung | grouped gemm fp4 + 清理 |
| `02b4fa69` (#929) | 2026-07-30 | WangLingxun | 第三方归属声明 |

**这对 HK 是三重坏消息：**

1. 需求侧不是 dense GQA backward，而是 sparse top-k gather 反向（含 `d_sink`、CSR 反向 scatter、可变有效 KV）。HK 论文全部是规则形状（已记在 [`papers/hipkittens.md`](../../papers/hipkittens.md) §11 未决问题 3），**现成的 GQA backward 资产不可复用，要从零写**。
2. 供给侧已有 4 种 DSL（Triton / Gluon / FlyDSL / TileLang）× 最多 3 代迭代在竞争同一个 op。HK 进来是第 5 条腿。
3. 而 dense（`cr=0`）那一路在 V4 里也被 `use_turbo_attention` 抢先，落到 aiter 汇编上——即 HK 已发表的强项对应的正是 V4 里最不吃紧的那一格。

## 更正五：漏掉了 Turbo attention 真正的框架侧资产，其中 ring 负载均衡是一个确定缺口

原计划只把 Turbo 看成「一个挂内核的地方」，没有盘点它自己有什么。实际上 Turbo 相对原生 aiter 的价值全在 wrapper 层，而这些正是 HK 编译期模板路线接入时的真实门槛：

| Turbo 的框架侧能力 | 实现 | HK 现状 |
|---|---|---|
| head_dim pad 到 8 的倍数（**这是 MLA 192/128 能落到 aiter 上的唯一原因**） | `flash_attn_interface.py:76-80`，输出 `:117` 切回 | 无，单一 `ATTN_D=128` |
| 三种 layout（bshd / sbhd / bhsd）的 stride 正确性 | `attention_utils.py:28-81` 推断 + `flash_attn_interface.py:143-168` 按 format 构造梯度 stride | 无 |
| varlen（thd packed） | `AiterFlashAttnVarlenFunc`，`flash_attn_interface.py:437-644` | 无 |
| 上下文并行（Ulysses A2A + ring 混合 USP） | `flash_attn_usp_interface.py:730/771/823` 三入口 | 无 |
| 测试矩阵已覆盖生产 GQA 与 MLA 形状 | `tests/pytorch/ops/test_attention.py:28-43`：64/8、32/8、28/4、48/8 GQA 与 192/128 MLA | 需按形状重编译 |

最后一行值得单独强调：原计划 §「缺口小结」把「GQA 比例需按比例实例化验证」列为 HK 的机会，但 Turbo 的测试矩阵**本来就已经覆盖** 8:1 / 4:1 / 7:1 / 6:1 的 GQA 与 MLA 192/128，跨 `batch ∈ {1,2,3,4}` × `causal` × `window_size` × 三种 layout × `is_v3_atomic_fp32` 全组合。这条「机会」实际上是原计划的信息缺失。

**而 CP 路径里有一个确定的、纯算法层的缺口：ring attention 没有做 causal 负载均衡。**

- 序列切分是连续 chunk：`t.chunk(cp_size, seq_dim)[cp_rank]`（`tests/pytorch/ops/test_attention_with_cp.py:38-46`）。
- ring 循环用「按 rank 整块跳过」处理 causal：前向 `if not arg_causal or step <= comm.rank`（`usp/attention_ring.py:134`），反向 `if step <= kv_comm.rank or not arg_causal`（`:202`）。
- 标准的 `2*cp_size` chunk zigzag 重排 **不存在**（全仓无重排 helper）。

后果是 causal + CP 下各 rank 计算量随 rank 单调递增，尾部 rank 成为整个 CP 组的 straggler。这是长序列训练的刚需项，且收益可以在不碰任何 kernel 的前提下量化。

另外两条 CP 限制：varlen 不支持 ring（`flash_attn_usp_interface.py:537-538`），fp8 不支持 ring（`:341-343`）；ring 的 comm/compute 重叠只靠异步 P2P `batch_isend_irecv` 预取，无独立 stream（`usp/attention_ring.py:128-149`）。

---

## 附：两个配置问题在最新 commit 上仍未修

DSV3 报告（§4 / §5）指出的两条，在 `bdd96e69`（2026-08-11）上复核仍然成立：

| 问题 | 现状 | 收益 |
|---|---|---|
| aiter pin 过期 | `common/aiter_utils.py:20` 仍 `AITER_VERSION = "0.1.14.post1"` | causal 前向 1.89× |
| `ATOMIC_FP32` 默认值未按 arch 决定 | `ops/attention/attention_utils.py:19` 仍 `os.getenv(ENV_ATTN_V3_ATOMIC_FP32, "1")` | 反向 1.68× |

## 解读：更正后 HK 的落点在哪

把五条更正叠起来，地形是这样的：

| 战场 | 现状 | HK 能带来什么 |
|---|---|---|
| dense MHA/GQA backward | aiter/CK 汇编吃满（MI355X causal 902–1047 TFLOPS） | 已发表最好成绩 1024，持平或落后 |
| MLA（DSV3 192/128） | Turbo wrapper + aiter 汇编，turbo≈TE 差 <2% | **源码层跑不了**（单一 `ATTN_D=128`） |
| sparse MLA / DSA（DSV4） | 4 种 DSL × 10 个后端在竞争，已到第三代 | 无现成资产，要从零写；且论文未验证 ragged 形状 |
| Turbo attention 后端派发 | 官方 TODO 已认领未做 | 与 HK 无关，但是 HK 接入的前置件 |
| ring attention causal 负载均衡 | 缺口确定，无 zigzag | 与 HK 无关，纯算法层 |

**HK 作为「内核供给方」的落点基本没有了。** 它作为「技法来源」还有一个位置：register pinning 是 HK 相对 FlyDSL 唯一独有的能力（更正二），而 DSA 反向的 dQ 累加正是寄存器压力最高、且 `_flydsl_v1/dsa_bwd_dq_flydsl_kernel.py` 已经在用手工多累加器拆 RAW 链去缓解的地方。**把 register pinning 作为技法提案给 FlyDSL、落在 DSA 反向 dQ 上**，是原计划 §「长期的一个可能性」里那条对冲，现在看它反而是主线上唯一还站得住的一条。

而如果目标里「正向支撑 Primus」的权重高于「基于 HK」，那么更正三和更正五各给出了一个更好的落点：都是缺口确定、纯框架/算法层、零内核风险、且可度量。

## 下一步

原计划的 P0' / P1' 收口动作不变（rocprof 查 16-head 反向机制、补测 GQA 比例下的 causal backward），但优先级下调——即使发现薄弱格子，更正一至四说明 HK 也很难成为填补者。

替代的三个候选（按「缺口确定性 × 可度量 × 成本」排序）：

1. **Turbo attention 后端派发器**（更正三）。对齐 GEMM 侧已有的 `GlobalBackendManager` + `AutoKernelDispatcher` 四级优先级设计，把 attention 的 `KernelBackend` 子类注册进去，加 `PRIMUS_TURBO_ATTN_BACKEND`。官方 TODO 已认领方向明确，纯 Python，无内核风险；做完之后任何第三方内核（含 HK）才有挂载点。
2. **ring attention 的 causal zigzag 负载均衡**（更正五）。先量化 causal + CP 下各 rank 的负载偏斜，再实现 `2*cp_size` chunk 重排。纯算法层，收益可在不碰 kernel 的前提下测出来。
3. **两个配置问题**（附节）。成本最低、收益已实测（1.89× / 1.68×），但属于别人报告里已经提出的建议，技术含量低，适合作为搭车项而非主线。

**决策点**：本项目（HK dense attention backward）建议从 on-hold 改为**关闭**，另起一条以 Primus attention 框架层为主线的项目；HK 的投入收敛为「register pinning 技法」这一个可迁移资产，等 DSA 反向 dQ 出现明确的寄存器瓶颈证据时再取出来用。

## 相关文件

**Primus-Turbo**（`/perf_apps/xiaoming/MegaMoE` = `bdd96e69`，或 `slab/Primus-Turbo` = `9c1e61c1`，attention 源码相同）
- `primus_turbo/pytorch/kernels/attention/attention_aiter_impl.py` — 唯一真实 dense 路径 + 两处派发器 TODO
- `primus_turbo/pytorch/ops/attention/flash_attn_interface.py` — head_dim pad / layout / varlen
- `primus_turbo/pytorch/ops/attention/{flash_attn_usp_interface.py, usp/attention_ring.py}` — CP
- `primus_turbo/flydsl/attention/sparse_mla_{fwd,bwd}.py` — FlyDSL 写的 attention 反向
- `agent/skills/kernel-optimize/knowledge/ops/attention/{overview,optimization-directions}.md` — Turbo 自带的 attention 优化知识库（含 §8 backward dQ 归约方案）

**Primus**（`a6c7fcd8`）
- `primus/backends/megatron/core/transformer/v4_attention_kernels/README.md` + 10 个后端目录
- `primus/backends/megatron/core/transformer/deepseek_v4_attention.py` — 后端派发与优先级

**FlyDSL**（`1afc0b0f`，本地 checkout 可能落后）
- `kernels/attention/` — 确认无 attention backward
- `kernels/gemm/fp8_gemm_{8wave,4wave}.py` — HK 两个调度的对应物

**slab**
- [`2026-08-12_1006_hk_primus_attn_bwd_plan.md`](./2026-08-12_1006_hk_primus_attn_bwd_plan.md) — 被更正的对象
- [`papers/hipkittens.md`](../../papers/hipkittens.md) — §11 未决问题 3 已预警 ragged 形状未验证
- DSV3 实测报告：`/perf_apps/xiaoming/MegaMoE/docs/attention_dsv3_8k_backends.md`
