# AutoOverlap:chunk 抽象 + 自动 kernel 内细粒度通信计算重叠编译器
# AutoOverlap: Enabling Fine-Grained Overlap of Computation and Communication with Chunk-Based Scheduling

> **arXiv:** [2601.20595](https://arxiv.org/abs/2601.20595) (2026-01) · **机构:** UC San Diego + Meta
> **平台:** 多 GPU（NVLink，H100 语境；源到源 Triton 编译器）
> **领域:** 分布式编译器 · comm-compute overlap · kernel 融合 · 自动调度
> **核心贡献:** 提出 **communication chunk 抽象**——把「通信粒度」从 kernel 结构和后端机制中解耦，让 chunk 级 overlap plan 可从别的分布式编译器移植、用户手写或模板实例化。基于 Triton 的**源到源编译器 + runtime**，自动把「本地 Triton kernel + chunk 调度」变成融合的分布式 kernel，做 inter/intra-chunk autotuning。平均 **1.3×**、最高 **4.7×**。

---

## 一、问题分析
- 现有分布式编译器（Alpa / Mercury）只在 **kernel/stream 级** overlap：每个通信阶段一次 kernel launch + kernel 边界 device-wide 同步 + 额外 launch/sync 开销；把计算切成多个短 kernel 又加剧 wave quantization（tail wave 空转）；粗粒度还在时间轴末尾留一长段几乎无 overlap 的通信尾巴。
- 手写细粒度方案（Flux / AsyncTP / FlashOverlap / COMET / Triton-Distributed）虽好，但**每个算子/架构/硬件都要专家手工**设计融合 kernel、选 tiling/buffer、推同步协议 → 难泛化、难 retarget。

## 二、方法
- **Communication chunk 抽象**：一个 chunk = 某通信操作关联的一块逻辑数据 + 产生/消费它的 tile。把「chunk 大小、后端选择、tile 顺序」暴露成少数原则性 knob，编译器据此推理「chunk 何时就绪、哪些 tile 产/用它、如何与融合多阶段程序 + 不规则集合通信交互」。
- **源到源 Triton 编译器 + runtime**：吃标注过的本地 Triton kernel + 高层 chunk 通信 plan，重排 kernel 的 tile 执行以跟随通信进度，为每个 chunk 选后端；runtime 无缝对接 PyTorch distributed，最小改动替换标准算子。
- **三个自适应维度**（新设计空间）：
  1. **自适应通信后端**：copy engine（独立于 SM、~400GB/s，但 host launch + 只连续、每次 2-3µs）/ TMA（16 SM 就 300+GB/s，但需 SM 发起、限节点内点对点）/ load-store（灵活、可 in-network reduce，但吃 SM 且同步）——按 chunk 特性选。
  2. **自适应 chunk 大小**：平衡链路吞吐 vs 同步开销。
  3. **自适应 intra-chunk tile 调度**：重排 tile 顺序跟踪通信进度，同时保 register/shared/cache 局部性。
- **inter/intra-chunk autotuning**：调 chunk 大小、后端、SM 分配、tile 顺序。

## 三、实验效果
| 维度 | 结果 |
|---|---|
| 端到端平均加速（常见算子） | **1.3×** |
| 最高 | **4.7×** |
| 定位 | Chunk 粒度 + Compute/Comm/Schedule 全 Auto + Template（对比表：Alpa kernel-auto-template；Flux/AsyncTP tile-manual；FlashOverlap/Triton-Dist chunk-manual；AutoOverlap chunk-auto） |

## 四、业界定位
- vs kernel 级编译器（Alpa/Mercury）：把通信当 full-kernel 黑盒，看不到 kernel 内 overlap；AutoOverlap 下探到 chunk。
- vs 手写细粒度（Flux/COMET/FlashOverlap/Triton-Distributed）：它们要专家逐算子手工；AutoOverlap 提供**通用编译器抽象 + 自动化**。
- **独特贡献**：chunk 抽象解耦「overlap 意图」与「底层实现」；自动化 + 后端自适应；plan 可复用/移植。

## 五、局限与复现
- 建在 Triton 源到源之上（Triton 表达力/性能边界之内）；评测多为常见算子级。
- 后端 TMA 限节点内点对点，跨节点仍需 load-store / 集合库。

## 六、对 monolith-moe / rocmoe 的启示（Our take）

AutoOverlap 是这批里唯一的**编译器/自动化视角**，正好对照我们「全手写 HIP super-kernel」的另一极。

| AutoOverlap | 我们（monolith-moe / rocmoe） | 关系 |
|---|---|---|
| **chunk 抽象**解耦通信粒度与 kernel 结构 | 手写死 chunk / block 粒度在 kernel 里 | ⭐ 若把我们的 dispatch/combine 参数化成 chunk plan,可复用/移植/autotune |
| **自适应通信后端**（copy engine / TMA / load-store 按 chunk 选） | 手写 HIP IPC（单一机制）+ DirectToLDS | ⭐ 值得评估:MI355X 上 XGMI DMA vs async-copy-to-LDS vs load/store 按 chunk 选 |
| **自适应 chunk 大小 + intra-chunk tile 调度** | 固定 tile / `kSubWGs` | 与 UniEP autotuner 呼应:tile/chunk 粒度不该拍死 |
| 源到源 Triton 编译器,PyTorch distributed 无缝替换 | 全手写 HIP C++ | ⚠️ 权衡:我们手写换极致性能,但可维护性/泛化差;AutoOverlap 是「自动化换泛化」的另一极 |

**两条最有用的结论：**
1. **chunk 抽象**给了我们一个把 super-kernel「参数化 + autotune」的框架思路——现在 dispatch/combine 的 chunk 粒度、走哪个搬运机制都写死在 HIP 里；若抽象成 chunk plan，就能像 UniEP/AutoOverlap 那样自动搜最优，而非人工返工。
2. **通信后端按 chunk 自适应**在 AMD 上尤其值得试:MI355X 有 XGMI DMA、async direct-to-LDS、普通 load/store 三条搬运路,当前我们基本只用一条;按 chunk 特性(大小/连续性/是否跨 rank)选后端可能是免费的 overlap 增益。

> 相关：[`../comet.md`](../comet.md)（手写细粒度对照）、[`./`](../uniep/README.md)（UniEP 的 autotuner 与此呼应）、`.cursor/skills/cco-pipeline-overlap/SKILL.md`。

---

*据 arXiv:2601.20595 全文（2026-01）整理于 2026-07-07。HTML：[`autooverlap.html`](./autooverlap.html)。*
