# Piper:资源建模 + 流水线混合并行的大规模 MoE 训练（Frontier / AMD）
# Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism

> **arXiv:** [2605.05049](https://arxiv.org/abs/2605.05049) (2026-05) · **机构:** Oak Ridge National Lab（Sajal Dash, Feiyi Wang）
> **平台:** **Frontier 超算 = AMD MI250X + Slingshot / Dragonfly + RCCL** ← 这是唯一一篇正面 AMD 的
> **领域:** MoE 训练 · HPC · PP×EP 混合并行 · 拓扑感知 all-to-all · 解析资源建模
> **核心贡献:** 建一套解析资源模型（显存/算力/通信），用 **PP×EP 设备网格把 EP 通信约束在拓扑局部小组** + **Dragonfly 拓扑感知分层 all-to-all** + **expert migration 负载均衡（<5% 开销）**。vs X-MoE **2–3.5× MFU**，all-to-all **1.5–4×**（摘要称 up to 9×）厂商带宽，在 Frontier 上把**万亿参数 MoE 训到 20% MFU**（X-MoE 545B 仅 5.23%）。

---

## 一、问题分析
- HPC 平台（Frontier）为模拟负载设计，**跨节点是非均匀 Dragonfly 网络**（组内高带宽、组间稀疏），topology-oblivious 的 all-to-all 会在慢的组间链路上撞车。
- 细粒度 MoE（DeepSeek-MoE 式）产生大量 tall-and-skinny GEMM（硬件利用率低）、激活膨胀、all-to-all 涉及大量进程。
- 现有框架（DeepSpeed-MoE/TED、X-MoE、Tutel）**把所有组件铺在大 GPU 组上**，逼昂贵集合通信（TP 4 次 all-reduce、sharded-DP 2 次 all-gather、EP 4 次 all-to-all）同时跨很多 rank；且**缺平台感知的混合并行规划**。X-MoE 在 500B+ 只有 5% MFU。

## 二、方法（四组件）
1. **解析资源模型**：量化任意 `(PP, EP)` 配置下的显存（参数+优化器+激活，含 SwiGLU 3 矩阵、per-expert 激活 `2·bsk/E·(3d_ffn+d_model)`）、算力、通信；micro-benchmark + 代码插桩 + 硬件 profiling 标定。
2. **PP×EP 混合并行（核心）**：`P` GPU 组成 `PP×EP` 网格——`PP` 个流水段，每段 `EP` 张卡处理 `L/PP` 层的一部分 expert（expert-data 并行）。**把 EP 通信约束在拓扑局部小组**（理想情况单节点或单跳 Rosetta switch 组）→ 用快的组内互联、躲开大规模 all-to-all 的高延迟。首次把 PP 用到 MoE 的**层内 EP 轴**。
3. **Dragonfly 拓扑感知分层 all-to-all**：显式建三层结构（intra-node / intra-group / inter-group），协调异步 P2P 消除空闲、均匀打满 NIC。
4. **Expert migration 负载均衡**：同层 GPU 周期性交换 expert 物理重分布，摊销 <5% 训练时间。

## 三、实验效果（Frontier / AMD MI250X）
| 维度 | 结果 |
|---|---|
| MFU | **2–3.5× vs X-MoE** |
| all-to-all 带宽 | 1.5–4×（摘要 up to 9×）vs 厂商实现 |
| 万亿参数 MoE | Frontier 上 **20% MFU**（对比 X-MoE 545B 仅 5.23%） |
| expert migration | 负载均衡开销 **<5%** |
| 覆盖模型 | DeepSeek-V2/V3、Mixtral、Qwen3、Llama-4、Kimi-K2 等（Table I，含 SwiGLU/shared-expert 细节） |

## 四、业界定位
- vs DeepSpeed-MoE/TED / X-MoE：它们无平台感知混合并行规划、把组件铺满大组；Piper 用 PP 约束通信域 + 资源模型自动选配置。
- vs 分层 all-to-all（Tutel/FasterMoE/HetuMoE）：它们把 inter-node 当均匀网络；Piper 显式建 Dragonfly 三层。
- **独特贡献**：首个统一显存+算力+通信、在真实 HPC 平台验证的 MoE 解析模型；PP-over-EP 把通信局部化。

## 五、局限与复现
- 强绑 Dragonfly / Frontier 拓扑；expert migration 是物理重排（与路由级 balance 互补）。
- 基于 Tutel 扩 EP；DeepSpeed-TED/MoE、X-MoE 对照。

## 六、对 monolith-moe / rocmoe 的启示（Our take）

**这是 5 篇里唯一跑在 AMD（MI250X + RCCL + Frontier）上的**，平台最贴近你的日常栈。

| Piper | 我们（monolith-moe / rocmoe / Primus on AMD） | 关系 |
|---|---|---|
| 跑在 **AMD MI250X + RCCL + Frontier** | MI355X + RCCL + Primus/Megatron | ⭐ 同厂商栈,RCCL all-to-all 行为、Dragonfly 拓扑经验直接可参考 |
| 统一显存/算力/通信解析模型选 (PP,EP) | 我们靠实测 sweep 定并行配置 | ⭐ 可借鉴:建 MI355X 的解析模型先剪枝配置空间 |
| **PP×EP 把 EP 通信约束在拓扑局部组** | 单节点 8×MI355X EP8（本就局部） | 我们已在最局部域;但多节点扩展时这条是模板 |
| Dragonfly 拓扑感知分层 all-to-all（1.5–9× vs 厂商） | RCCL 默认 all-to-all | ⭐ 若做多节点,RCCL 的拓扑感知优化空间被证实很大 |
| tall-and-skinny GEMM 利用率低是核心痛点 | 我们 GEMM 才 62% peak（hipBLASLt-class gap） | 同痛点,佐证 grouped-GEMM 优化的价值 |

**两条最有用的结论：**
1. **同栈平台对标**——Piper 是少见的 AMD/RCCL MoE 训练公开工作,它的 Dragonfly 拓扑感知 all-to-all 拿到 1.5–9× 厂商带宽,说明 RCCL 默认 all-to-all 在非均匀网络上有巨大优化空间;若 rocmoe/Primus 扩到多节点,这是直接可抄的方向。
2. **解析资源模型先剪枝**——与 UniEP 的 autotuner、DisagMoE 的 roofline 一脉相承:与其盲 sweep,不如先建 MI355X 的显存/算力/XGMI 解析模型把配置空间剪到可行子集,再实测精调。tall-and-skinny GEMM 利用率低这条也再次印证我们 grouped-GEMM 优化的必要性。

> 相关：`.cursor/skills/mi355_hardware_aware/SKILL.md`（MI355X 硬件参数可用于建 Piper 式解析模型）、[`../../notes/monolith-moe/README.md`](../../notes/monolith-moe/README.md)（GEMM 62% peak）。

---

*据 arXiv:2605.05049 全文（2026-05）整理于 2026-07-07。HTML：[`piper.html`](./piper.html)。*
