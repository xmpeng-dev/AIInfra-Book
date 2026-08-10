# arXiv 速览 2026-08 — MoE 融合核与通算重叠

> **检索日期：** 2026-08-10
> **检索范围：** 2026-05-01 ~ 2026-08-06 投稿，22 组查询词跨 cs.DC / cs.AR / cs.PF / cs.LG
> **方法：** arXiv Atom API 按投稿日倒序抓取，去重 1157 篇 → 关键词加权打分过阈 92 篇 → 人工复核 27 篇
> **去重基准：** `papers/README.md` 已有笔记 + `knowledge/` 中出现过的 arXiv ID
> **对应画布：** `~/.cursor/projects/home-xiaoming-workspace/canvases/arxiv-moe-kernel-digest-2026-08.canvas.tsx`
> **常设索引：** [`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md)（本文是该文档 8 月的增量扫描）

---

## 0. 一句话结论

这三个月社区正好在集中回答我们已经押上的那个赌注。**「persistent megakernel + 核内确定性 overlap」在单节点上被反复验证有效**，但同时冒出两类反证：

1. **跨节点 RDMA 会把它的优势整个吃掉** — [2605.00686](https://arxiv.org/abs/2605.00686)，8 节点上退化最多 10×；
2. **静态资源划分在负载波动下会退化** — [2607.18002](https://arxiv.org/abs/2607.18002)，直接质疑编译期定死 WG 角色比例。

另一边，昇腾阵营用 HyperParallel-MoE、UBEP、StrataCL、Relay-Buffer-Free 四篇走了一条几乎同构的「静态调度 + 设备侧单边通信 + 去 host 同步」路线，说明 ROCmoe 的执行模型判断不是 AMD 特产，可以引用作旁证。

---

## 1. 最该先读的五篇

| # | 论文 | arXiv | 为什么排在最前 | 精读笔记 |
|---|------|-------|----------------|----------|
| 1 | Perseus: Eliminating Hidden Serialization in Multi-Node Megakernel Communication | [2605.00686](https://arxiv.org/abs/2605.00686) | MegaMonolith 出单节点会正面撞上的墙 | [`perseus.md`](../../papers/perseus.md) |
| 2 | X-Stage: An Overlooked Pipeline Stage for Communication-Computation Overlap in DiT Inference | [2607.23264](https://arxiv.org/abs/2607.23264) | 检验「COMM_DISPATCH 与 COMPUTE free overlap」的边界 | [`x-stage.md`](../../papers/x-stage.md) |
| 3 | HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs | [2605.23764](https://arxiv.org/abs/2605.23764) | ROCmoe 同构论点在昇腾上的完整实现，最强旁证 | [`hyperparallel-moe.md`](../../papers/hyperparallel-moe.md) |
| 4 | ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels | [2607.18002](https://arxiv.org/abs/2607.18002) | 反方论据，ROCmoe P4 必须回应 | [`expertplex.md`](../../papers/expertplex.md) |
| 5 | AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis | [2606.09682](https://arxiv.org/abs/2606.09682) | ROCmoe P2 缺的调度 IR 与安全校验层 | [`automegakernel.md`](../../papers/automegakernel.md) |

---

## 2. 执行模型 / megakernel（5 篇）

| 论文 | arXiv | 核心 | 与我们的关系 |
|------|-------|------|--------------|
| Eliminating Hidden Serialization in Multi-Node Megakernel Communication | [2605.00686](https://arxiv.org/abs/2605.00686) | proxy-based RDMA 里「tile 传输先于完成信号」的顺序约束强制 fence 排空 NIC 流水，代价随并发传输数增长 | MonolithEP/MegaMonolith 扩到多节点前必读 |
| X-Stage | [2607.23264](https://arxiv.org/abs/2607.23264) | persistent kernel 发出 remote store 之后、远端可见之前的软件可见流水阶段；Burst-Gap 模型预测背压拐点 | 检验 free overlap 前提；给 XGMI 写背压建模 |
| HyperParallel-MoE | [2605.23764](https://arxiv.org/abs/2605.23764) | 逐 kernel 串行 → 跨 AIC/AIV 静态调度 tile 级异构 taskflow；AIV 驱动单边通信消 host 同步 | 与 ROCmoe P1/P2/P4 同构，可引用 |
| ExpertPlex | [2607.18002](https://arxiv.org/abs/2607.18002) | adaptive persistent kernel 做 PD 共置；指出 Green Context 式静态切分无法跟踪算子间资源变化与逐层专家负载 | 反方论据，P4 需回应 |
| AutoMegaKernel | [2606.09682](https://arxiv.org/abs/2606.09682) | schedule-IR 静态校验器证明无死锁/无竞争；同源 retarget sm_80/90/120 | P2 缺的那层 |

## 3. EP 通信层（6 篇）

| 论文 | arXiv | 核心 | 与我们的关系 |
|------|-------|------|--------------|
| UBEP | [2607.06202](https://arxiv.org/abs/2607.06202) | 超节点上重构 EP 通信库：干掉 BSP 粗粒度串行、同步开销、距离无关调度 | 「不写 megakernel、改写通信库」路线的最强代表 |
| Relay Buffer Independent Communication over Pooled HBM | [2605.06055](https://arxiv.org/abs/2605.06055) | 池化 HBM + 对称内存；直接写目标专家窗口 / 直接读远端窗口，去掉中继与重排缓冲 | 与 MonolithEP 的 IPC scatter/gather + SymmWorkspace 同路线 |
| StrataCL | [2607.26444](https://arxiv.org/abs/2607.26444) | registration-on-allocation 用户 buffer 直通；负载均衡核划分 + 设备侧 SDMA offload | CloudMatrix384 上 MoE dispatch/combine 带宽 +1.4× |
| EEP（Surviving Partial Rank Failures） | [2605.10670](https://arxiv.org/abs/2605.10670) | 把 EP 成员关系当可变的活性问题；CUDA graph 里烧死的路由元数据是障碍 | 对静态化路线的容错警告 |
| DODOCO | [2605.20982](https://arxiv.org/abs/2605.20982) | 5 个真实 MoE checkpoint 实测推翻两个假设：路由不均不能靠系统层纠正；mock-token benchmark 不代表生产路由 | 负载均衡类工作的评测方法论检验 |
| TAOT | [2608.03676](https://arxiv.org/abs/2608.03676) | 拓扑感知最优传输做训练期专家副本动态放置 | 与 UltraEP 同问题不同解法 |

## 4. AMD / 性能建模 / 工具（4 篇）

| 论文 | arXiv | 核心 | 与我们的关系 |
|------|-------|------|--------------|
| Kerncap | [2605.03208](https://arxiv.org/abs/2605.03208) | HSA runtime 层拦截 dispatch（HIP + Triton），地址空间闭包快照，产自包含 reproducer | 调 MMOE/FlyDSL kernel 时可直接用 |
| Microbenchmark-Driven Analytical Performance Modeling | [2605.04178](https://arxiv.org/abs/2605.04178) | B200（TMEM/TMA）+ MI300A（Infinity Cache/VGPR/占用率）解析模型；MI300A 27 kernel MAE 0.09%，naive roofline >95% 误差 | ROCmoe P5 的 roofline 判据需要的正是这个 |
| TileSight | [2607.22432](https://arxiv.org/abs/2607.22432) | tile 从编程原语升为分析原语，贯通核内流水 / cache 层次 / 跨 GPU 互连 | FlyDSL 性能模型可照搬 |
| UltraQuant | [2606.20474](https://arxiv.org/abs/2606.20474) | 4-bit KV cache，含 AMD GPU 优化的 decode-attention kernel 与 FP4 近似路径 | 少见把 AMD 当一等目标的量化工作 |

## 5. 内核 DSL / codegen（5 篇，FlyDSL 相关）

| 论文 | arXiv | 核心 | 与我们的关系 |
|------|-------|------|--------------|
| Tile-Level Activation Overlap | [2607.02521](https://arxiv.org/abs/2607.02521) | SwiGLU 中间张量物化占 MLP 时间 9–37%；两个 CUTLASS SM90 kernel（Pingpong warp specialization / Epilogue Visitor Tree）tile 级融进 GEMM，最高 2.47×，torch.compile 慢 3–7× | FlyDSL 的 MoE FC1+SwiGLU 融合直接对标 |
| DA-MoE（Decoding the Skew） | [2607.23099](https://arxiv.org/abs/2607.23099) | 最优 fused-MoE kernel 随路由偏斜与 token 数变化，静态 token-count 分桶选核是错的；Effective Experts 指标 + GPU-resident 运行期核分派 | FlyDSL grouped GEMM autotune 策略 |
| ComFuse | [2608.03537](https://arxiv.org/abs/2608.03537) | 复杂访存密集子图融进计算密集 kernel | 编译管线「哪些子图值得融」这一层 |
| Correct but Slow | [2607.04454](https://arxiv.org/abs/2607.04454) | DSL 生成 kernel「编译过了、结果对了、但就是慢」的系统性实证 | FlyDSL 对外评测方法论 |
| CommBench | [2608.04450](https://arxiv.org/abs/2608.04450) | 100+ 专家编写/生产蒸馏的 GPU 通信编程题，覆盖 EP 通信与通算融合；多卡真跑的防作弊评测 | 也可当通信 kernel 参考实现集 |

## 6. 训练并行 / 框架（4 篇，Primus 相关）

| 论文 | arXiv | 核心 | 关键数据 |
|------|-------|------|----------|
| Mixture-of-Parallelisms | [2607.01844](https://arxiv.org/abs/2607.01844) | 分层组合并行 + 新 optimizer step 策略 | <12 个 8×H200 节点跑万亿参数 / 百万上下文，per-GPU 吞吐 4.7–8.2× |
| moefs | [2607.18631](https://arxiv.org/abs/2607.18631) | 把「工具链能否真的 emit」作为一等搜索约束；跨并行/调度/kernel 三层搜索，同时产 Megatron + SGLang 栈 | 8×H100 上超最强手调基线 |
| LAGA（MLA 序列并行显存回归） | [2607.17644](https://arxiv.org/abs/2607.17644) | 解释 Megatron-Core 为何 hard-assert 禁掉训练路径的 absorbed MLA：中间量 n_h×d_kv/token 比被替代的 per-head K/V 更大 | DSv3 规模激活显存 +20–34%，融合 kernel 下差距扩到 19.2 GB |
| SLAI T-Rex | [2607.20145](https://arxiv.org/abs/2607.20145) | 昇腾 SuperPOD 上 DeepSeek-V4 全参后训练，模型并行/通算编排/kernel 三层优化 | MFU 34.22%，2.93× vs 开源基线 |

## 7. 低精度（3 篇）

| 论文 | arXiv | 核心 |
|------|-------|------|
| Stable FP4 Training via Transposition-Invariant Block Quantization | [2607.24953](https://arxiv.org/abs/2607.24953) | 前向/反向对同一权重分块方向不同是 FP4 训练不稳的根源，用转置不变分块量化解决 |
| FOCUS | [2608.01847](https://arxiv.org/abs/2608.01847) | 耦合松弛 + 双粒度缩放的 FP4 方案 |
| QUADS | [2607.15810](https://arxiv.org/abs/2607.15810) | MoE 上 NVFP4 的双侧量化误差对齐，稳定 RL 后训练 |

---

## 8. 检索覆盖度自检

同一套查询词还捞回了 7 篇已有笔记或已在 `knowledge/` 索引里的论文，说明检索没漏主线（这 7 篇已从上面清单剔除）：

Piper `2605.05049` · DisagMoE `2605.11005` · MoE Tile Signaling `2607.19539` · UltraEP `2606.04101` · MoE-Hub `2605.05888` · Ada-MK `2605.11581` · Resource-aware Overlap `2606.09200`

## 9. 复现检索

```bash
# 查询词与打分逻辑见本次会话；核心 API 调用形如
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.DC+AND+abs:%22mixture-of-experts%22\
&sortBy=submittedDate&sortOrder=descending&max_results=80"
```

阈值：关键词加权分 ≥ 12（megakernel / MoE / 通算重叠 / ROCm 各 5–6 分，
cs.DC|cs.AR|cs.PF 分类 +4，标题命中 NLP/医疗/联邦等无关词 −6）。
