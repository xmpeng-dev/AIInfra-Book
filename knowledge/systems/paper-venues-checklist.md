# 论文检索会议清单（ML Systems）

> 来源：[xmpeng-dev/ml-systems-papers](https://github.com/xmpeng-dev/ml-systems-papers)（fork 自 [byungsoo-oh/ml-systems-papers](https://github.com/byungsoo-oh/ml-systems-papers)，作者 Byungsoo Oh 即 [Perseus](../../papers/perseus.md) 一作）
> 统计口径：该 README 共 **1,101 条带会议标签的论文**、约 120 个不同标签；其中落在与我们直接相关的 8 个分区（Training System / MoE / Communication / GPU Kernel / Compiler / Attention / GPU Memory / Inference）的有 **785 条**。
> 该仓库的 arXiv 预印本另存在 `README_arxiv.md`，本文件只讨论**正式会议**。

## 为什么需要这份清单

我们之前的检索是 arXiv 单一来源，[industry-training-optimization-2026](./industry-training-optimization-2026.md) 已经记录过这个结构性缺口：**中国大厂的生产级工作大量以会议论文形式出现、且从不投 arXiv**。最典型的就是 [Tessera](../../papers/tessera.md)（OSDI '26，阿里云，Qwen3/Qwen3-Next 生产预训练，万亿模型 39% MFU）——2,498 篇的 arXiv 全量扫描完全没扫到它。

所以本清单的核心用途不是"再列一遍所有会议"，而是回答一个问题：**哪些会议的论文 arXiv 上搜不到，必须去会议本身捞。**

## Tier 0：arXiv 搜不到，必须直接翻会议

这是本文件最重要的一段。这几个会议的系统论文（尤其是工业界的 Operational Systems / Industry track）**经常从不上 arXiv**，只能从会议官网或 DBLP 拿。

| 会议 | 核心统计 | 覆盖主题 | 出版时间（大致） | 取用方式 |
|---|---|---|---|---|
| **OSDI** | 59 | 系统设计与实现，训练/推理框架、编译器（Compiler 分区 14/26 都是 OSDI） | 7 月 | **USENIX 全站开放获取**，PDF 直接下 |
| **SOSP** | 24 | 同 OSDI，偏经典系统；已从双年改为**每年** | 10–11 月 | ACM DL，近年多为开放获取 |
| **EuroSys** | 50 | 训练系统、调度、MoE 工程（MegaScale-MoE 即此处） | 3–4 月 | ACM DL，近年多为开放获取 |
| ~~USENIX ATC~~ | 28 | **已停办** — USENIX 董事会 2025 年决定 sunset，**ATC '25（2025-07，Boston）是最后一届**（[官方公告](https://www.usenix.org/blog/usenix-atc-announcement)：2024 年仅 165 人参会）。历史论文仍在 USENIX 开放获取，部分主题并入其他 USENIX 会议 | — | 只翻存量 |
| **NSDI** | 41 | **Training System 分区第一（24 篇）**，集群调度、网络 | 4–5 月 | USENIX 开放获取 |
| **ASPLOS** | 53 | 软硬件协同，内存管理（GPU Memory 分区第一） | 3–4 月 | ACM DL；一年多轮投稿 |
| **SC** | 34 | HPC 侧大规模训练、通信；AMD/HPC 相关工作常在此 | 11 月 | ACM/IEEE；作者常自挂 PDF |
| **FAST** | 5 | 存储、checkpoint | 2 月 | USENIX 开放获取 |

**优先级判断**：如果只能盯三个，选 **OSDI / EuroSys / NSDI**——前两个是异构 MoE 训练系统的主场（Tessera、MegaScale-MoE 都在这儿），NSDI 是集群与通信侧论文密度最高的地方。

## Tier 1：与我们主题强相关，但多数也会上 arXiv

这些会议的论文通常有 arXiv 版本，现有的 arXiv 扫描已能覆盖大部分，**去会议官网的作用是补漏和确认最终版**。

| 会议 | 核心统计 | 对我们的价值 | 时间 |
|---|---|---|---|
| **MLSys** | 62（**核心分区总数第一**） | ML 系统的主场，Comet 等 comm-compute overlap 工作在此 | 5 月，proceedings.mlsys.org 免费 |
| **PPoPP** | 16 | 并行编程模型、kernel 级并行 | 2–3 月 |
| **SIGCOMM** | 21（**Communication 分区第一**） | 集合通信、网络拓扑、超节点互联 | 8 月 |
| **ISCA / MICRO / HPCA** | 27 / 14 / 20 | 体系结构侧；低精度数值格式、片上互联、NPU 设计 | 6 月 / 10–11 月 / 2–3 月 |
| **CGO / PLDI** | 7 / 4 | **编译器与 kernel DSL 的主场**，FlyDSL 路线的对标在这里 | 2–3 月 / 6 月 |
| **ICS / HPDC / IPDPS / ICPP** | 6 / 5 / 8 / 9 | HPC 系统，通信与调度的二线来源 | 6 月 / 6 月 / 5 月 / 8 月 |
| **SoCC / Middleware / APNET** | 11 / 2 / 3 | 云侧调度与资源管理，优先级低 | — |

## Tier 2：ML 会议——已被 arXiv 覆盖，不必单独翻

| 会议 | 核心统计 | 说明 |
|---|---|---|
| NeurIPS / ICML / ICLR | 45 / 44 / 23 | **MoE 分区里 NeurIPS(12)+ICML(10) 排前二**，但这些论文几乎 100% 有 arXiv 版本。继续用 arXiv 扫描即可，不需要单独去 OpenReview 翻 |
| ACL / EMNLP / NAACL / COLM | 12 / 7 / 5 / 7 | MoE 的**算法侧**（路由、稀疏化）出现在这里；系统侧价值有限 |

> **一条经验**：MoE 分区的会议分布明显偏 ML 会议（NeurIPS/ICML/ICLR/ACL/NAACL/EMNLP 合计 42 篇 vs MLSys 8 篇），说明 MoE 的**系统**论文密度其实低于算法论文。找 MoE 系统工作，去 OSDI/EuroSys/MLSys 比去 NeurIPS 有效得多。

## 明确排除

这些在原 README 里有但与我们无关，检索时应主动过滤，避免重复之前 arXiv 扫描里"医学影像/联邦学习"污染候选集的问题：

联邦学习（Federated Learning 分区）、隐私保护 ML（CCS / USENIX Security / S&P）、CV 会议（CVPR / ECCV / ACM MM）、推荐系统（RecSys / KDD / SIGIR / WWW）、数据库（VLDB / SIGMOD / ICDE，除非是 checkpoint / 存储主题）、软件工程（ICSE / FSE / ESEC）、移动端（MobiSys / MobiCom / SenSys / IMWUT）、Agentic AI（SAA @ SOSP'25 等）。

## 检索方法

**DBLP 是最省事的入口**——它按会议按年索引，且有 API，适合脚本化，比逐个会议官网翻页可靠。

```bash
# 某会议某年的全部论文（HTML/XML/BibTeX 都有）
https://dblp.org/db/conf/osdi/osdi2026.html
https://dblp.org/db/conf/eurosys/eurosys2026.html
https://dblp.org/db/conf/nsdi/nsdi2026.html

# 关键词跨会议检索 API（JSON）
curl 'https://dblp.org/search/publ/api?q=mixture-of-experts+training&h=200&format=json'
```

DBLP 会议路径速查：`conf/osdi`、`conf/sosp`、`conf/eurosys`、`conf/usenix`（ATC）、`conf/nsdi`、`conf/fast`、`conf/asplos`、`conf/sc`、`conf/mlsys`、`conf/ppopp`、`conf/sigcomm`、`conf/isca`、`conf/micro`、`conf/hpca`、`conf/cgo`、`conf/pldi`、`conf/ics`、`conf/hpdc`、`conf/ipps`（IPDPS）。

**USENIX 系（OSDI / ATC / NSDI / FAST）可以直接抓**，全站开放获取，PDF 链接形如：

```
https://www.usenix.org/conference/osdi26/technical-sessions      # 会议目录
https://www.usenix.org/system/files/osdi26-hu-weifang.pdf        # 论文 PDF
```

**上游仓库本身也值得定期同步**——[byungsoo-oh/ml-systems-papers](https://github.com/byungsoo-oh/ml-systems-papers) 维护得相当活跃（README 里已有 '26 年的 EuroSys/OSDI/ASPLOS 条目），直接 diff 它的 commit 比自己从零扫会议便宜。

## 检索日历

按出版月份排，用来决定"这个月该去翻哪几个会议"：

| 月份 | 该翻的会议 |
|---|---|
| 2–3 月 | FAST、HPCA、PPoPP、CGO |
| 3–4 月 | **EuroSys**、ASPLOS |
| 4–5 月 | **NSDI**、IPDPS |
| 5–6 月 | **MLSys**、ISCA、PLDI、ICS、HPDC |
| 7 月 | **OSDI** |
| 8 月 | SIGCOMM、ICPP |
| 10–11 月 | **SOSP**、MICRO、SC |

> 当前是 2026-08。**OSDI '26、EuroSys '26、ASPLOS '26、NSDI '26、MLSys '26、FAST '26、PPoPP '26、CGO '26、HPCA '26 均已发布并已完成一轮扫描** → [`venue-scan-2026-08.md`](./venue-scan-2026-08.md)。SOSP '26 / SC '26 / MICRO '26 / ISCA '26 要等到 10–11 月（ISCA '26 6 月已开但 DBLP 尚未索引）。

## 与现有文档的关系

- [`industry-training-optimization-2026.md`](./industry-training-optimization-2026.md) — 双来源调研，本清单补的正是它记录的"arXiv 系统性漏掉会议论文"这个缺口
- [`arxiv-digest-2026-08.md`](./arxiv-digest-2026-08.md) — arXiv 侧的增量扫描，与本清单互补
- [`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md) — 全景速查与机构索引
