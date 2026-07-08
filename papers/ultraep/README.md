# UltraEP:Rack-Scale 节点上的实时精确负载均衡
# UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing

> **arXiv:** [2606.04101](https://arxiv.org/abs/2606.04101) (2026-06) · **机构:** 北京大学 + 小红书 + 上海 AI Lab
> **平台:** Rack-Scale Node（RSN，64+ GPU scale-up 域；显式点名 **AMD Helios** / NVIDIA Rubin / UALink）
> **领域:** MoE 负载均衡 · EP · 训练 + serving prefill · RSN 通信
> **核心贡献:** 首个在 RSN 上对 large-EP MoE 做 **exact-load、实时（每 microbatch + 每层）** 负载均衡的系统。用 **quota-driven planner**（联合优化 replication + reroute）+ **RSN-native 通信**（persistent tile streaming + chunk-streaming relay 树），把 inter-rank 不均从 1.30–4.01 压到 **1.01–1.04**，达 **94.6% 训练 / 93.9% serving prefill 理想吞吐**，训练 **1.42× vs Megatron-LM**、serving **1.56× vs SGLang**。

---

## 一、问题分析

### 1.1 背景
- Large-EP（32/64-way）是数千亿 MoE 的标配，但**放大专家负载不均** → 计算 straggler、token all-to-all 瓶颈、激活显存尖峰。EP degree 越大，per-rank 专家越少，路由抖动越直接翻译成 inter-rank skew。
- **RSN（rack-scale node）** 把 scale-up 域从单机 4/8 GPU 扩到整机架 64+ GPU（load/store 语义、几百 GB/s），使「把整个 EP 组放进一个高带宽域、在热路径上做均衡」第一次物理可行。

### 1.2 现有方案的坑
- EPLB / LPLB 等靠**历史负载预测**周期性重排 expert。但作者实测（§3）：前沿细粒度 MoE 的专家热度在 **microbatch / 层 / 数据域间剧烈漂移**（prefill 尤甚），stale 预测常失效甚至**加剧**不均。
- 转向 **exact load** 需在 gating 之后才拿到真实负载 → 均衡操作被逼上**关键路径**，暴露 plan solving + 重的 expert 搬运开销。RSN 之外（跨机架 scale-out）搬 expert 状态代价过高。

### 1.3 解决方案
两大创新对应控制面 / 数据面：
1. **Quota-driven planner**：直接对「每个 expert 实例的最终 token 负载 quota」求解，把 replication 与 reroute **耦合**（只有能带来有效负载的 replica 才被创建），阈值二分 + 贪心可行性 oracle，**GPU-native**（单 CTA/SM，shared memory）。
2. **RSN-native 通信**：把动态 expert 传输编成 device 端 tile 级任务（persistent tile streaming，double-buffer），对多 replica 的热 expert 建 **chunk-streaming relay 树**分摊 fan-out。

---

## 二、方法

### 2.1 专家布局与显存
- **logical / physical expert**：每 rank 固定 main slot + `N_slot` 个 redundant slot。**Replication-only**（不重排 main expert——large-EP 下每 rank main expert 少，重排收益小代价大）。
- **跨层 buffer 复用**：redundant slot **不存 optimizer state**、weight/grad buffer 跨层共享。Qwen3-235B（94 层）单 redundant slot 从 3.3GB weights/6.6GB grads 降到 **36MB/72MB**——代价是前向关键路径上有严格的 per-layer weight 物化 deadline。

### 2.2 前向 / 反向流水
- **前向（Fig.8）**：复用已有 notify-dispatch 拿全局路由 → 每 rank 确定性地算出**相同**的 replication + reroute plan（无额外同步）→ 分发 main-expert weight 到远端 replica（可与 reroute overlap，但 token dispatch 必须等它完成以免抢带宽）→ **planning + weight replication 都在关键路径**。
- **反向（Fig.9）**：先恢复 redundant expert weight（可与 Wgrad overlap），MoE backward 后每个 main expert **聚合所有 replica 的梯度**回 main buffer（保训练等价，且必须在下一层前完成，因 buffer 跨层复用）。反向复用前向缓存的 plan。

### 2.3 Quota-driven planning（Algorithm 1）
- 找最小负载阈值 τ 使所有 rank 靠 replication 都能压到 τ 以下：二分 τ + 贪心 oracle（按残余 excess 降序访问过载 rank、按 λ_e 降序访问其 main expert，把负载迁到 slack 最大的可行 rank，受 quota 下限 `u_min`=1024、slot 预算、no-duplicate 约束）。**用 quota 当耦合变量**：每次 probe 在选 replica 的同时预留 reroute 容量，避免枚举 replica 集或 token 级路由，也避免无效 replica。
- **Reroute**：quota 定后只做 source-wise split，优先让本地 token 消费本地 quota（省跨 rank 流量），残余按剩余容量按比例分 + 确定性 rounding。per-token 分配用 cumulative-quota 前缀扫描的上界查找。
- **GPU-native 求解**：单 CTA 在一个 SM 上，负载矩阵/放置状态入 shared memory，多 warp 并行评估阈值 probe，warp 级归约找高-slack 目标。

### 2.4 RSN-native 通信
- **Persistent tile streaming**：weight 分发 + 梯度归约共用；expert 权重切固定 tile，plan 编译成 device 端任务流，持久 kernel 反复 pull 下一 tile；double-buffer（传 tile i 时预取 i+1），把任务查找/寻址/同步折进 tile 流水。
- **Chunk-streaming relay 树**：replica 数超阈值（4）的热 expert 建两级 relay，relay frontier ≈ √(|H(e)|−1)；按 chunk 而非整 expert 转发（relay 收到一个 chunk 就转，无全局 barrier）；load-aware 选发送量最小的 rank 当 relay。

---

## 三、实验效果

**设置**：公有云 RSN（64 GPU/机架，scale-up 带宽为 scale-out 的 8–10×）；模型 GLM4.5-106B / Qwen3-235B / GLM4.7-358B / DeepSeek-V3-671B；up to 256 GPU（4 机架）；bf16。Baseline：Megatron-LM、SGLang、EPLB、LPLB、EPLB+（喂 exact load）、Ideal（强制均衡上界）。

| 指标 | 结果 |
|---|---|
| 训练理想吞吐占比 | **94.6%**（平均，三模型 imbalance 压到 1.01–1.03） |
| serving prefill 理想吞吐占比 | **93.9%**（90–97%） |
| 训练加速 | 平均 **1.42× vs Megatron-LM**（EPLB/LPLB/EPLB+ 分别 +20/12/29%） |
| serving 加速 | **1.56× vs SGLang**、1.29× vs EPLB |
| inter-rank 不均 | 1.30–4.01 → **1.01–1.04** |
| expert 复制通信 | **3.1–5.5× vs torch.distributed / DeepEP**；relay 再 +1.3–1.8×，高 fan-out 下 latency 近常数 ~0.28ms |
| 激活显存尖峰 | 无均衡时训练 2× / serving 11× 高于 ideal；UltraEP 拉回接近 ideal |
| 生产（RefMoE-288B，多机架） | 92%+ ideal，+9.6% vs no-balance，loss 轨迹不变 |
| 消融 vs EPLB+ | imbalance 1.19→1.03，solving −27.4%，slot −57.9%，流量 −3.9% |

**延迟拆解（Qwen3-235B）**：UltraEP 的热路径额外开销仅 0.33ms 前向（占总 1.8%）；MoE compute 已接近 ideal；剩余 token all-to-all +33%/10%（源于真实路由不均，非 UltraEP 引入）。

---

## 四、业界定位

| 方案 | 时机 | 手段 | 局限 |
|---|---|---|---|
| EPLB | 周期性、历史负载 | 冗余 expert 启发式复制 | 追不上非平稳负载 |
| LPLB | per-microbatch | LP 求解 reroute，≤1 replica/expert | replica 预算受限 + 求解慢 |
| **UltraEP** | **每 microbatch/层、exact load** | **quota 联合 replication+reroute + RSN-native 通信** | 依赖 RSN scale-up 域 |

**独特贡献**：首个 RSN 上 exact-load 实时均衡；quota 耦合 replication+reroute（直接优化 post-reroute 负载而非 pre-reroute hotness）；RSN 专用动态通信。同时覆盖训练 + 推理，可自然扩展到 RL。

---

## 五、局限与复现
- **强依赖 RSN scale-up 域**：标准 RDMA 集群（EP 组跨多节点走慢速 scale-out）上不适用。
- 需 GPU-native、one-sided peer-memory 语义（symmetric buffer + peer handle）。
- 9.6K 行 C++/Python，集成 Megatron-LM + SGLang（各 <1K 行），用 DeepEP hybrid-ep 分支做 token dispatch。代码未见开源说明。

## 六、对 monolith-moe / rocmoe 的启示（Our take）

UltraEP 和我们**平台高度对口**：它显式点名 **AMD Helios RSN**，而我们正是在单节点 8×MI355X / XGMI 上做（一个小型 scale-up 域）。它解决的是我们一直回避的**负载不均**这条正交轴。

| UltraEP | 我们（monolith-moe / rocmoe） | 关系 |
|---|---|---|
| exact-load 实时均衡（gating 后即算 plan） | 固定 round-robin 分 compute WG，假设 per-expert token 均匀 | ⭐⭐ 直击我们 A2 失败根因（token 数不均致 GEMM tile 利用率坍塌） |
| quota 联合 replication + reroute，GPU-native 求解（单 CTA/SM） | 暂无系统级均衡 | ⭐ 可移植:在 super-kernel 内部先算 quota 再分 tile |
| replication-only + 跨层 buffer 复用（不存 optimizer state） | — | 若上均衡,这套省显存布局值得抄 |
| persistent tile streaming + double buffer + chunk relay 树 | persistent super-kernel + receiver-pull | 通信手法同源;relay 树是我们没做的 fan-out 分摊 |
| **RSN = scale-up 域内均衡才可行** 的核心论断 | 我们本就在 XGMI scale-up 域内 | ⭐ 我们天然满足前提,可在 kernel 内直接做热路径均衡 |

**三条最有用的结论：**
1. **负载不均要在系统层解**——UltraEP 实测「即使有 balance loss，microbatch/层级仍严重不均」，正是我们 A1/A2「动 compute 排布无效」的互补答案:不要试图让 compute 排布吸收不均，而应在 dispatch 前**按 exact load 分 quota / 迁移 token**。
2. **quota-driven 的 GPU-native 求解**（单 CTA/SM、shared memory、warp 归约、~0.1ms）证明热路径上做균衡决策可行——我们的 super-kernel 里完全可以塞一个类似的 planning phase。
3. **我们天然满足 RSN 前提**（XGMI scale-up 域），且 UltraEP 点名 AMD Helios，说明这条路线对 AMD rack 明确有效;是 rocmoe 之后引入负载均衡时的首选参考。

> 相关：[`../../notes/monolith-moe/README.md`](../../notes/monolith-moe/README.md)（A1/A2 「compute 是 parallelism-bound、不要动排布」+ work-stealing tile counter backlog）。

---

*据 arXiv:2606.04101 全文（2026-06）整理于 2026-07-07。HTML：[`ultraep.html`](./ultraep.html)。*
