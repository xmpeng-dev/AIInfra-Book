# UniEP:统一专家并行的 MoE MegaKernel 训练系统
# UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training

> **arXiv:** [2604.19241](https://arxiv.org/abs/2604.19241) (2026-04-21) · **代码:** [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)
> **机构:** ByteDance Seed + 清华大学 · **平台:** NVIDIA Hopper（Triton-distributed，作者称可移植 AMD）
> **领域:** MoE 训练 · comm-compute overlap · MegaKernel · 数值可复现
> **核心贡献:** 首次把 **MegaKernel** 用到 MoE 训练——把 Dispatch+GroupGEMM 与 GroupGEMM+Combine 各融成一个持久化单 kernel，SM 动态分 Comm/Comp/Relay 角色、token 级 scoreboard 同步、单 CUDA stream 内细粒度 overlap，且靠**确定性 token 映射保证 bit-wise 与串行执行一致**。vs SOTA（COMET）**1.03–1.38× 端到端**、kernel 级最高 18.4× vs 非重叠基线。

> **⭐ 与本项目关系最紧密的一篇。** UniEP 基本是 NVIDIA/Hopper 侧版本的 RocMoE-v2 super-kernel：MegaKernel + SM 角色分配 + 片上 scoreboard + 单 stream + 确定性排序，逐条能和我们的设计对上。详见 §6「Our take」。

---

## 一、问题分析

### 1.1 背景与动机
- MoE 已是主流（GPT-5 / Gemini-3 / DeepSeek-V3 / Qwen3，专家数 64–384），EP 成为跨卡分发标准。MoE 组件（dispatch + expert compute + combine）吃掉训练总时间的 **30%–80%**。
- 根因：算力增速远快于 NVLink/IB 带宽 → EP 被 all-to-all / all-gather 卡住。

### 1.2 现有方案的两个致命缺陷
1. **计算与通信当成两个独立算子，靠 CUDA 多流粗粒度 overlap** → 需 host 侧同步、跨 rank 产生 bubble；更严重的是**为了 overlap 要切 micro-batch，改变了反向 Transposed GroupGEMM 的累加顺序**（BF16 非结合律）→ **破坏 bit-wise 可复现性**，工业训练不可接受。
2. **缺乏统一通信原语**：top-k 大时 AllGather 省带宽、稀疏路由时 AllToAll 更优，逼开发者维护多套 heuristic + 多套 kernel。

### 1.3 解决方案（两个洞察）
- **洞察 1**：最优 overlap 不需要 host 多流——用 GPU 的 SM 做 block 级调度。把此前只用于**推理**的 MegaKernel 搬到训练，但**只融 MoE 子图**（Dispatch+GroupGEMM、GroupGEMM+Combine），不融整模型。SM 动态分 Comm/Comp 角色，token 级 scoreboard 管依赖，单 stream 内激进 overlap，**不切 micro-batch → 保持确定性**。
- **洞察 2**：用参数化抽象统一 AllGather/AllToAll，配一个硬件性能模型自动选最优配置。

---

## 二、方法

### 2.1 持久化 Worker + 三角色
启动时线程块数 = 物理 SM 数，一一映射常驻（无上下文切换、无因超发导致的死锁）。每个 worker 动态担任：
- **Comm-Worker**：发 token、发起跨 GPU 事务（`putmem_warp` push，实测 push > get）；
- **Comp-Worker**：跑 padding-free GroupGEMM tile（MegaBlocks/vLLM 式动态指针）；
- **Relay Worker**：管同步信号 + 做 rank 内 multicast。

### 2.2 确定性 token 映射（Algorithm 1，保 bit-wise）
把本地 token 的 `(target_rank, expert_id, destination_offset)` 用 **AllGather 的每-rank-每-expert 计数表 + 沿 rank 维的 exclusive prefix sum** 算出全局写偏移（式 1），再叠本地 stable-sort 索引 → 每个 Comm-Worker 无锁、无冲突地算出唯一目标地址。**token 到达顺序固定** → 后续归约顺序固定 → 反向 Transposed GroupGEMM 累加树不变 → bit-wise 等价。

### 2.3 Scoreboard 同步 + 动态调度
- Scoreboard = `[S_token（到达）| S_tile（tile 就绪）]`。Relay 轮询 `S_token`，攒够一个 tile（如 128）就置 `S_tile`；Comp 轮询 `S_tile` 就绪即算。用 `ld_acquire/st_release` 保跨 SM 可见性。
- **动态角色**：全局 atomic 计数器当任务队列游标，SM 原子自增取任务 ID 决定角色；初始全做通信，数据到齐后陆续转计算，天然负载均衡。

### 2.4 Relay 带宽优化 + 优先级调度
- **Relay intra-rank multicast**：一个 token 若路由到同一目标 GPU 的多个 expert，只跨 NVLink 传一次、本地 HBM 复制。Top-8 / 8 GPU 下期望只需发到 **平均 5.25 个不同 rank → ~34% 流量削减**。
- **优先级 token 调度**：Comm-Worker 按 expert 升序发（与 Comp 消费顺序一致），避免 head-of-line blocking；靠 prefix-sum offset 隐式实现零开销。

### 2.5 AutoTuning
配置空间 `C=(N_disp, N_comb, N_relay, N_red, warps)` ≈ **10^5**（如 209,088）。解析性能模型（roofline 式，含 GEMM/SwiGLU/dispatch/combine 各段 latency 公式 + overlap 模拟）预测 latency，C++/OpenMP 求解器 **144ms** 搜完，配 4096-token 分桶 memoization 摊销到可忽略。模型预测 vs 穷举误差 0.5–6.5%（均值 3.8%）。

---

## 三、实验效果

**设置**：2 个 Hopper 集群（Cluster1：200GB/s NVLink；Cluster2：400GB/s NVLink），8 GPU/节点。12 个 MoE 配置（DeepSeek/Qwen/Kimi 家族，专家 64–512），seqlen 8k/32k/128k。Baseline：**Serial（DeepEP + TransformerEngine，非重叠）** 与 **COMET（双流重叠 SOTA）**。

| 维度 | 结果 |
|---|---|
| Dispatch+GroupGEMM kernel | vs Serial 8k **18.4×** / 32k 11.2×；vs COMET **1.30–1.57×** |
| GroupGEMM+Combine kernel | vs Serial 8.2–11.6×；vs COMET 1.07–1.87× |
| 层级（fwd+bwd） | vs Serial fwd 11.98×；vs COMET fwd 1.08–1.22×（seqlen 越长优势越大） |
| **数值精度** | **UniEP 对参考实现 bit-wise 完全一致（max_diff=0）**；COMET max_diff 高达 **0.25**、22–29% 元素不一致 |
| 端到端（128 GPU，512k seqlen 生产训练） | 127 → **138 B tokens/day = 1.09×**，且保持 bit-wise 可复现 |
| 消融 | O→B（Relay 带宽）1.06–1.36×；B→A（autotune）1.15–1.68×；全系统 A vs COMET **1.24–1.73×** |
| 非 bit-wise 变体（放松确定性，切 2 子 batch） | 反向再快 2–8%（多数配置），但 low-intensity / compute-heavy 两例反退 |

---

## 四、业界定位

| 路线 | 代表 | 与 UniEP 差异 |
|---|---|---|
| 多流 kernel 级 overlap | Megatron-LM / Centauri / TorchTitan | UniEP 单 stream，无 host 干预、无 wave quantization |
| device-side 信号双流 | COMET / FLUX / TileLink / CoCoNet | 仍双流；UniEP 融成单 MegaKernel 消除 alignment bubble |
| 跨 micro-batch overlap | DeepSeek-V3 DualPipe | 破坏 bit-wise + 2× 参数；UniEP 单 micro-batch 内 overlap 保确定性 |
| 推理 MegaKernel | Mirage / HazyMega | 融整模型，仅 batch-1 推理划算；UniEP 只融 MoE 子图适配大 batch 训练 |

**独特贡献**：首个 MoE 训练 MegaKernel；bit-wise 确定性 token 映射；统一 AllGather/AllToAll + 10^5 空间的解析 autotuner。

---

## 五、局限与复现
- 目前 Triton 实现，最新架构上 Triton 可能弱于手写 CUDA（作者称设计语言无关，可重写 CUDA 补齐，参考 DeepGEMM）；未用 TMA。
- 评测限单节点 8 GPU（跨节点 IB 场景未测）。
- **代码开源**（Triton-distributed），~21k 行 Python。作者明确**可移植到 AMD**（Triton-distributed 支持 AMD）。

## 六、对 monolith-moe / rocmoe 的启示（Our take）

这是目前和 RocMoE-v2 super-kernel **最同源**的一篇，几乎逐条对得上：

| UniEP | 我们（monolith-moe / rocmoe） | 关系 |
|---|---|---|
| 持久化 worker（1 threadblock / SM）+ 动态 Comm/Comp/Relay 角色 | persistent super-kernel + `comm_ratio` 分 CU + `__launch_bounds__(_,1)` 物理隔离 | 高度同源；他们**动态**分角色（atomic 任务队列），我们是 **build-time `kSubWGs`** 静态分 → ⭐ 可借鉴动态化 |
| **确定性 token 映射（AllGather 计数 + prefix-sum 全局 offset）保 bit-wise** | sender counting-sort + `atomicAdd` 写 `pack_perm`（**非确定序**），逼得 backward 必须复用 `pack_perm` | ⭐⭐ **最该借鉴**：UniEP 用 prefix-sum 全局 offset 得到确定序，正好解掉我们 backward 与 `pack_perm` 的强耦合 |
| token 级 scoreboard（`S_token`/`S_tile`）+ `ld_acquire/st_release` | 64-bit `block_ready` 位图 + release-acquire | 收敛一致 |
| Relay intra-rank multicast（Top-8→5.25 rank，−34% 流量） | 我们暂无 | ⭐ 新点：DSV3 top_k=8 / EP8 下同样有「一 token 多 expert 落同 rank」，可省 XGMI |
| 10^5 空间解析 autotuner（C++ 144ms + 分桶缓存） | `kSubWGs` 五档人工 sweep + 三次返工 | ⭐ 可借鉴：把 sub-WG / tile / ratio 做成解析模型 + 运行时按 (T, skew) 选 |
| push（putmem）> get 实测 | 我们 receiver-pull（比 push 每 token 慢 ~43%，靠 overlap 藏） | ⚠️ 反向结论：UniEP 明确 push 更快；值得复测我们 pull vs push 在 MI355X XGMI 上的差 |

**三条最有用的结论：**
1. **确定性 token 映射**是我们 backward `pack_perm` 耦合问题的现成解——用 AllGather 计数 + 沿 rank prefix-sum 得确定全局 offset，替掉 `atomicAdd` 非确定序，backward 不必再复用 forward 的 perm。列为 rocmoe 高优先级实验。
2. **动态 SM 角色 + 解析 autotuner** 比我们 build-time `kSubWGs` 静态值更能吸收 skew，尤其 hot_cov50 下 dispatch 变 critical path 时。
3. **Relay multicast** 是我们没做的正交省流量招（DSV3 top_k=8 场景收益可观），且 UniEP 已开源 Triton 实现、明确可移植 AMD，可直接读它的 Triton-distributed 代码对照。

> 相关：[`../comet.md`](../comet.md)（UniEP 的双流前身对照）、[`../../notes/monolith-moe/README.md`](../../notes/monolith-moe/README.md)（`pack_perm` 非确定序 / `kSubWGs` / launch_bounds CU 隔离）。

---

*据 arXiv:2604.19241 全文（2026-04-21）整理于 2026-07-07。HTML：[`uniep.html`](./uniep.html)。*
