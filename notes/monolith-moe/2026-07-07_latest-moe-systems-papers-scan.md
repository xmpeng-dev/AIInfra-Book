# 最新 MoE 系统论文扫描（2026-01 ~ 2026-02 batch）

> **When**: 2026-07-07 09:20 UTC+8
> **Where**: slab 知识库 / 文献调研（无 GPU）
> **Context**: 以 [`xmpeng-dev/ml-systems-papers`](https://github.com/xmpeng-dev/ml-systems-papers) 的 MoE 章节为索引，抓取其中最新一批（`arxiv'26` + `EuroSys'26`）10 篇 MoE 系统论文的真实摘要，为 [`monolith-moe`](./README.md) / [`rocmoe`](../rocmoe/README.md) 的 super-kernel + CCO 方向做外部对标。摘要与数字均来自各 arxiv/会议原文，非臆测。

## TL;DR

- 这批最新工作在**推理侧**明显扎堆：预测式 prefetch（PROBE）、细粒度 offload（FineMoE）、负载均衡（LLEP）、容错（Tarragon）——共同前提都是「MoE 推理是 memory-bound + 负载高度不均」。
- **训练侧**只有 3 篇，但每一篇都踩在我们的核心命题上：MoEBlaze（co-designed kernel + 消除中间 buffer）、MegaScale-MoE（inter/intra-op comm-compute overlap + 通信压缩，生产级 1.88×）、MoE-DisCo（拆成 dense 子模型省钱）。
- **和我们最相关的两条主线**：(1) **fused comm 算法**——MixServe 的 fused AR-A2A、MegaScale-MoE 的 holistic overlap，都是我们 super-kernel「把 A2A 塞进 GEMM kernel」思路的 host-side/collective-side 对照组；(2) **降低搬运量的正交杠杆**——LatentMoE 用 latent 投影把 all-to-all 体积和 expert weight 字节直接 ÷(d/ℓ)，和我们 backlog 里的「FP8/mxfp8 weights」是同一类正交优化。
- 全部为 NVIDIA/通用 GPU 语境，**没有一篇针对 AMD CDNA**——我们在 MI355X 上做 in-kernel XGMI overlap 仍是空白区。

## 一、训练系统（和 super-kernel 最相关）

| 论文 | 场景 | 核心手段 | 关键数字 |
|---|---|---|---|
| **MoEBlaze** (MLSys'26, `2601.05296`) | MoE 训练显存墙 | 端到端 dispatch + 消除中间 buffer/activation 物化；co-designed kernel + smart activation checkpoint | >4× 加速 / >50% 省显存 vs Tutel/MegaBlocks；已上 Meta 推荐生产 |
| **MegaScale-MoE** (EuroSys'26, `2505.11432`) | 生产级大规模 MoE 训练 | attention/FFN 各自定制 comm-efficient 并行；inter- + intra-op comm-compute overlap；通信压缩到低精度 | 352B MoE / 1440 Hopper → 1.41M tok/s，**1.88× vs Megatron-LM**（ByteDance） |
| **MoE-DisCo** (`2601.06857`) | 低成本训练 | 把 MoE 拆成多个 dense 子模型（shared backbone + 单 expert），无监督聚类分数据，各自在廉价设备无通信独立训，最后大 GPU 上短 finetune 合并 | 性能追平/超过全参训练，成本 ↓47.6–69.5%（Qwen1.5-MoE-2.7B / Llama-MoE-3.5B） |

**逐篇要点：**

- **MoEBlaze** — 论点与我们 P2「消除中间 buffer / 单相 FC1-FC2」高度重合，但它是在 NVIDIA H100 上用 wgmma + TMA 做的，且强调「重算比搬运便宜」（activation checkpoint overhead 近乎为 0）。**对我们**：验证了「co-design kernel + 砍中间物化」这条路在训练侧值 4×，是我们 super-kernel 训练工况反超 PyTorch+RCCL 的理论背书；它没有做 comm overlap，这块正是我们的差异化。
- **MegaScale-MoE** — 目前训练侧「comm-compute overlap + 通信压缩」的生产级标杆，1.88× 是 host-side/collective-side 能到的上限参照。**对我们**：它证明 overlap 有真实收益（不像 RCCL multi-stream 反慢），但仍停留在 op 级 overlap；我们要论证 in-kernel chunk 级 overlap 能比 op 级再进一步。「通信压缩到低精度」和我们 backlog 的 FP8 weights 思路一致。
- **MoE-DisCo** — 与 super-kernel 正交（是训练范式而非 kernel），但「拆 dense 子模型省通信」的思路对成本受限场景有参考价值，暂不影响我们主线。

## 二、推理服务：预测 / overlap / offload

| 论文 | 痛点 | 核心手段 | 关键数字 |
|---|---|---|---|
| **PROBE** (`2602.00509`) | EP 下 straggler + A2A 拥塞「双重惩罚」 | Continuous Lookahead Pipelining：把 predict/plan/prefetch 移出关键路径；gate-init 预测器 + 硬件感知均衡求解 + phase-locked split-phase 传输（不撞 A2A） | prefill ↓1.32× / decode ↑1.26× |
| **MixServe** (`2601.08800`) | 分布式 MoE 服务通信瓶颈 | 自动选 TP-EP 混合并行 + **fused AR-A2A**（overlap 节点内 AR 与节点间 A2A） | DeepSeek-R1/Qwen3：TTFT 1.08–3.80× / ITL 1.03–1.66× / 吞吐 +5.2–50.3% |
| **FineMoE** (EuroSys'26, `2502.05370`) | expert 稀疏激活 → 显存低效 | 细粒度 GPU↔CPU expert offload；expert map（迭代级选择模式）+ prompt 语义提示指导 prefetch/cache/offload | 延迟 ↓47% / expert 命中率 ↑39% vs SOTA |

**对我们**：MixServe 的 **fused AR-A2A** 是我们「把 A2A 融进 kernel」思路在服务侧、collective 层面的最直接对照——它靠 overlap 节点内外通信拿到 up to 3.8× TTFT，但仍是两个 collective 的 overlap，没有和 GEMM 融合。PROBE 的「compute-comm co-balance」在问题定义上和我们的 comm_ratio 调优同源（都是分 CU/带宽给 compute vs comm）。FineMoE / offload 与我们的全驻留 super-kernel 场景不同，参考价值较低。

## 三、负载均衡 / 容错 / 架构

| 论文 | 类别 | 核心手段 | 关键数字 |
|---|---|---|---|
| **LLEP** (`2601.17111`, Salesforce) | EP 负载均衡 | 把过载设备的超额 token + expert 参数动态迁到空闲设备；**不改路由、数学等价**，支持 backward | up to 5–6× 加速 / 4× 峰值显存 ↓；gpt-oss-120b ~1.9× |
| **Tarragon** (`2601.01310`) | MoE 推理容错 | AW/EW 分为独立故障域；可重配数据通路 rerouting + 自愈（AW 异步增量 KV checkpoint、EW 用残余显存部署 shadow experts） | 故障 stall ↓160–213×（~64s → 0.3–0.4s），无故障时 <3% overhead |
| **LatentMoE** (`2601.18089`, NVIDIA) | 架构（accuracy/FLOP） | 路由 + expert 计算 + 通信全在低维 latent 空间（d→ℓ→d），把路由字节和 expert weight 字节 ÷~(d/ℓ)，省下的预算换更多 expert + 更高 top-k | iso-accuracy 下 up to 3.5× 加速；已用于 Nemotron-3 Super/Ultra |
| **DES** (`2602.00879`) | MoE diffusion LLM | 并行解码致「expert 爆炸」；序列级 coreset 选择（DES-Seq intra-seq + DES-Vote 显著性投票）复用 expert | unique expert 激活 ↓55% / 延迟 ↓38%，保 99% 精度 |

**对我们**：
- **LLEP** 直击我们真实痛点——DSV3 per-(src,expert) 的 token 数不均导致 GEMM tile 利用率坍塌（见时间线 A2 失败 note）。它在算法层做 token/weight 迁移；我们可以借「按实际负载动态分配 compute WG」而非固定 round-robin（对应 backlog P2「work-stealing tile counter」）。
- **LatentMoE** 是继 FP8 之后又一个**降低搬运量的正交杠杆**：d/ℓ 直接砍 all-to-all 体积和 weight 字节，且与 tile 布局、swizzle 完全正交。若未来目标模型采用 latent MoE，我们的 dispatch/combine 流量和 FC weight HBM 流量会同步下降，super-kernel 的 comm_ratio 甜点会再漂移。
- Tarragon / DES 与当前主线关系较弱（容错、diffusion 解码），仅登记。

## 对 monolith-moe / rocmoe 的启示（按 ROI）

1. **差异化定位更清晰了**：MegaScale-MoE（op 级 overlap）、MixServe（collective 级 fused AR-A2A）已经把「host 侧 / 两个 collective 之间的 overlap」做到 1.88–3.8×；我们的价值主张必须锁定在**它们做不到的 in-kernel chunk 级 GEMM×A2A 融合**，且是在 **AMD CDNA / XGMI** 这个全空白区。
2. **正交杠杆值得优先排**：FP8/mxfp8 weights（backlog P0）+ LatentMoE 式 latent 投影，都是「和 kernel 布局正交、直接砍 HBM/XGMI 流量」的招，收益可叠加，风险低于继续动 compute 排布（P1/A1/A2 已连续证伪 compute 重排）。
3. **负载不均要在系统层解**：LLEP 说明「即使训练时加了 load-balance loss，推理/后训练时 expert 仍严重不均」——印证我们不能假设 per-expert token 数均匀，work-stealing / 动态 WG 分配比固定切分更稳。

## 来源

- 索引：`xmpeng-dev/ml-systems-papers` 的 Mixture of Experts 章节（本地快照 `ml-systems-papers-0.md`，行 1088–1160）。
- 摘要抓取：各论文 arxiv `abs/html` 页 + EuroSys'26 会议页（arxiv id 见上表）。本 note 只覆盖该 batch 中最新的 `arxiv'26` + `EuroSys'26` 条目；`NeurIPS'25 / SC'25 / arxiv'25` 等更早条目未纳入。
