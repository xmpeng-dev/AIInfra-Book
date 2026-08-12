# 大厂训练侧优化动向（2026-03 ~ 2026-08）

> **定位：** 不看学术热点，只看**大厂在训练工程上实际投了人力的方向**。按机构组织，每条都核实过论文的**作者单位归属**（拉 arXiv HTML 的作者块 + 邮箱域名，不靠摘要里出现公司名）。
> **口径：** 只收训练相关（含明确回灌训练的推理工作）；纯推理/serving 工作除非有训练含义否则不收。
> **配套：** 主题维度的全景见 [`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md)；最近一轮 arXiv 增量扫描见 [`arxiv-digest-2026-08.md`](./arxiv-digest-2026-08.md)。
> **检索方法与可复现命令：** 见文末 §5。

---

## 0. 结论先行

> 本文档有**两个独立来源**：§1–§2 来自 arXiv（作者单位逐篇核实），§6 来自开源仓库与工程博客。交叉印证与冲突见 [§6.3](#63-两个来源的交叉印证与冲突)。

**如果只记七件事：**

1. **FP4 已经从"能不能推理"变成"能不能训练"，而且战线推到了 RL。** 华为把 HiFloat4 从预训练一路做到**端到端 FP4 RL 后训练**（rollout + training 双侧 4-bit），阿里 Qwen 做 NVFP4 rollout，蚂蚁重新审视 E2M1 的收缩偏差，腾讯连出三篇 FP4/FP8 量化，AMD 在 MI355X 原生 FP4 上定位到 **Wgrad 是发散主因**。这是目前投入最密集的一条线。
2. **RL 后训练正在长成独立的一套系统栈。** 阿里 RollArt、NVIDIA Molt、腾讯 ForeMoE / shared-prefix 复用，都在解同一类问题：rollout 与 training 阶段负载特性完全不同，不能沿用预训练的调度假设。
3. **超节点（rack-scale / supernode）正在逼各家重写通信库。** 华为 StrataCL + UBEP（SIGCOMM'26）针对 CloudMatrix384，NVIDIA 给 NCCL 加 EP 专用 API，Meta 为 MTIA 300 自研 HCCL，Intel 出 PCCL。NCCL/RCCL 的"buffer-centric + 通用 collective"抽象在超节点上被集体质疑。
4. **弹性训练的实现路径从 checkpoint 转向直接状态迁移。** 华为/中科院 ETC 和无问芯穹 DynaTrain 都绕开存储持久化走 P2P 直传，DynaTrain 把 235B MoE 的并行切换压到 **4.36 秒**。
5. **Meta 罕见地把训练硬件叙事摆到了台前。** MTIA 300 是 Meta 第一颗把后端网络集成进封装的芯片，配套 HCCL 用**编译式通信模型**把 collective 完全卸载到片上 message engine；且 MTIA 300 **已量产用于排序/推荐训练**。
6. **Muon 从算法选择变成了系统约束。** 字节、DeepSeek、月之暗面、微软、NVIDIA、AMD 的训练栈在这半年里全部接入 Muon。它要求分布式框架保住参数的 2D 结构，与现有"把梯度当扁平 buffer"的设计直接冲突（详见 [§6.5](#65-两侧共同的新主线muon-成为跨栈共识)）。
7. **低精度的战场已从 GEMM 转到参数存储与通信。** NVIDIA `--fp4-param-gather` 让分片以 native NVFP4 packed 4-bit 直接 all-gather，**前向路径零 per-microbatch 量化**；这是一条独立于算力的收益来源，也是 AMD 侧最明确的差距。

**最值得我们细读的三篇：** Meta HCCL（[2608.00358](https://arxiv.org/abs/2608.00358)）、华为 UBEP（[2607.06202](https://arxiv.org/abs/2607.06202)）、NVIDIA Megatron-Core MoE 训练报告（[2603.07685](https://arxiv.org/abs/2603.07685)）。

**一处必须自己拿主意的开放问题：** FP4 训练里随机性到底有没有用——NVIDIA 默认开 stochastic rounding + RHT，AMD 两次明确说它稳不住 Wgrad。两边都是生产级证据，没有公开材料给出定论。见 [§6.3](#63-两个来源的交叉印证与冲突)。

---

## 1. 国内大厂

### 1.1 字节跳动 Seed —— MoE 训练全栈，且开始往编译器走

| 工作 | arXiv | 单位构成 | 核心 | 关键数据 |
|------|-------|----------|------|----------|
| **MegaScale-Omni** | [2605.08962](https://arxiv.org/abs/2605.08962) | ByteDance（上交合作） | 多模态 LLM 训练：encoder 与 LLM backbone **解耦并行**（encoder 长短序列 SP + backbone 5D 并行）、encoder-LLM 联合流水、数据加载端去中心化重排 | 生产动态负载下 **1.27×–7.57×**，千卡级在产 |
| **UniEP** | [2604.19241](https://arxiv.org/abs/2604.19241) | ByteDance Seed + 清华 | Dispatch+GEMM / GEMM+Combine 融合成单 MegaKernel | vs COMET **1.03–1.38×**（已有笔记） |
| **DisagMoE** | [2605.11005](https://arxiv.org/abs/2605.11005) | ByteDance Seed + UW + Cornell | Attn/FFN GPU 组解耦 + AF-Pipe | **1.8×** @128×H800（已有笔记） |
| **DITRON** | [2605.02953](https://arxiv.org/abs/2605.02953) | ByteDance Seed + 浙大/北大/清华 | **分布式多级 tiling 编译器**：不再靠 cuBLAS+NCCL 拼装，直接编译并行张量程序 | ⭐ 与 FlyDSL 直接对位 |

**读法：** 字节这半年的重心从"手写融合 kernel"（Comet/UniEP）明显往**编译器化**移动（TileLink → DITRON）。MegaScale-Omni 则说明另一件事：多模态训练的瓶颈已经不是并行策略本身，而是**动态负载下资源分配与并行策略的静态耦合**。

> **对已有笔记的修正：** landscape 里 DisagMoE 的机构标为"—"，实为 **ByteDance Seed 主导**；UltraEP 标为"—"，实为**北大主导 + 小红书/上海**（见 §1.5）。

### 1.2 阿里（通义 Qwen / 蚂蚁）—— RL 训练系统 + 低精度 rollout

| 工作 | arXiv | 单位 | 核心 | 关键数据 |
|------|-------|------|------|----------|
| **RollArt** | [2512.22560](https://arxiv.org/abs/2512.22560) | 阿里（ROLL 团队） | Agentic RL 的**解耦式多任务训练**：prefill 计算密集、decode 带宽密集、环境执行 CPU 密集、奖励评估突发——四类负载分池 | 2025-12 首发 / 2026-06 更新 |
| **QUADS** | [2607.15810](https://arxiv.org/abs/2607.15810) | **Qwen Team, Alibaba** | MoE 的 NVFP4 rollout 稳定化：双侧量化误差对齐 | 面向 W4A4 FP4 GEMM |
| **UFP4 / 收缩偏差** | [2606.20381](https://arxiv.org/abs/2606.20381) | **蚂蚁 Ant Group** | 指出 Blackwell/Rubin 与 AMD MI350 系列的 FP4 路径都押在 **E2M1** 上，而 E2M1 的非均匀格式带来系统性收缩偏差 | ⭐ 对 AMD FP4 训练直接相关 |

**读法：** 阿里这条线两头都在 RL——**系统侧**（RollArt 的四类负载解耦）和**数值侧**（QUADS 的 rollout 低精度）。蚂蚁那篇是少见的**质疑硬件格式选择**的工作，值得注意的是它把 AMD MI350 系列和 NVIDIA 放在同一批判范围内。

### 1.3 腾讯 —— 集中火力在 RL 后训练的负载与精度

| 工作 | arXiv | 单位构成 | 核心 |
|------|-------|----------|------|
| **ForeMoE** | [2606.11867](https://arxiv.org/abs/2606.11867) | 北大 + 上交 + 腾讯 | RL 后训练的 **micro-step 级 MoE 负载均衡**：step 级负载稳定但 micro-step 因 batch 极小而剧烈抖动，改用 rollout 阶段的**可预见路由信息**前瞻调度。64 GPU 上 **1.45×** |
| Shared-Prefix 复用 | [2606.01143](https://arxiv.org/abs/2606.01143) | HKUST + 腾讯 | GRPO 同 prompt 多轨迹的**调度级前缀复用** |
| 在线动态批处理 | [2606.19989](https://arxiv.org/abs/2606.19989) | 腾讯 | 样本真实训练开销只有预处理后才可见，离线 batch sampler 的核心假设失效 |
| P-Cast FP8 Attention | [2606.06521](https://arxiv.org/abs/2606.06521) | 腾讯 | Attention Sink 导致 FP8 softmax 矩阵 cast 崩塌，论证 S=2^8 最优 |
| SOAR / FOCUS | [2605.12245](https://arxiv.org/abs/2605.12245) · [2608.01847](https://arxiv.org/abs/2608.01847) | 腾讯（部分为实习产出） | NVFP4/MXFP4 的 scale 选择与双粒度缩放 |

**读法：** ForeMoE 那个观察很关键——**预训练的负载均衡方法在 RL 后训练里会直接失效**，因为统计口径从 step 掉到了 micro-step。任何打算把训练侧均衡策略复用到 RL 的方案都要先过这一关。

### 1.4 华为（昇腾）—— 投入密度最高，且是唯一把 FP4 打通到 RL 的

这半年检索到的**训练系统类论文数量，华为是国内第一**（24 篇命中、11 篇训练系统相关）。

| 工作 | arXiv | 单位构成 | 核心 | 关键数据 |
|------|-------|----------|------|----------|
| **HiFloat4 预训练** | [2604.08826](https://arxiv.org/abs/2604.08826) | 华为 | HiF4 格式 vs MXFP4 的大规模训练对比，linear + expert GEMM **全 FP4** | 相对误差控制在全精度 **1%** 内 |
| **HiFloat4 RL 后训练** | [2607.26515](https://arxiv.org/abs/2607.26515) | 华为 | **首个端到端 FP4 RL 后训练**（rollout+training 双侧、含反向）。核心发现：主要退化源不是训练侧量化，而是 **rollout 激活量化**；只把训练侧恢复高精度反而更差（rollout-training 失配）。用 Rollout-ResQ 残差修正 | 与 BF16 差距 **4.9% → 1.1%**；MXFP4 上 13.6% → 5.3% |
| **UBEP** | [2607.06202](https://arxiv.org/abs/2607.06202) | 华为 + 南京大学（**SIGCOMM'26**） | 为生产超节点（NVL72/576、CloudMatrix384）重构 EP 通信库；指出统一地址空间+高带宽 fabric 并不自动带来稀疏 MoE 通信收益，瓶颈在**执行严格串行化**等三点 | ⭐⭐ 强烈建议精读 |
| **StrataCL** | [2607.26444](https://arxiv.org/abs/2607.26444) | 华为 | **fabric-native 零冗余**通信库：现有库是 buffer-centric（用户 buffer 与通信 buffer 分离，导致多余拷贝或昂贵注册），改为 **registration-on-allocation** 实现用户 buffer 直通 |
| **CommFuse** | [2604.24013](https://arxiv.org/abs/2604.24013) | 华为 Toronto Ascend Team | 用**分解后的 P2P** 替代 reduce-scatter / all-gather，消除切分式 overlap 的尾延迟 | 对 TPSP / UP 均适用 |
| **ETC（弹性状态迁移）** | [2607.04749](https://arxiv.org/abs/2607.04749) | 中科院计算所 + 华为 | 抛弃 checkpoint，靠**状态局部性 + P2P 直传**做混合并行的在线重配；已集成进 Megatron-LM | 迁移开销降 **2.33×–6.37×** |
| **HyperParallel-MoE** | [2605.23764](https://arxiv.org/abs/2605.23764) | 中科大 + 华为 | AIC/AIV 异构核交错调度 + 设备侧单边通信 | 已有精读笔记 |

**读法：** 华为这套组合拳的逻辑非常完整：**格式（HiF4）→ 预训练 → RL 后训练 → 超节点通信库（UBEP/StrataCL）→ 弹性（ETC）**。对我们最有价值的是两点：(1) FP4 RL 的失败模式定位到 rollout 侧而非训练侧，这个结论跨硬件通用；(2) UBEP/StrataCL 都在说同一件事——**超节点的通信库不能沿用 NCCL 那套 buffer 语义**。

### 1.5 其他国内单位

| 工作 | arXiv | 单位 | 核心 | 关键数据 |
|------|-------|------|------|----------|
| **DynaTrain** | [2605.18815](https://arxiv.org/abs/2605.18815) | **无问芯穹** + 中科院计算所 + 北大 | **亚秒级在线并行切换**：Virtual Parameter Space 把任意并行配置变成统一逻辑坐标系下的确定性映射，把切换化约为几何求交 | 70B dense **<2s**、235B MoE **4.36s**；比 checkpoint 类方案快**三个数量级** |
| **UltraEP** | [2606.04101](https://arxiv.org/abs/2606.04101) | 北大主导 + 小红书 + 上海 | rack-scale 精确负载均衡，**每 microbatch 每层**重均衡 | 256 GPU / 106B–671B；达理想均衡的 **94.3%**，**1.49×** |
| **LongCat-Flash-Thinking** | [2601.16725](https://arxiv.org/abs/2601.16725) | **美团 LongCat 团队** | 技术报告，含训练基础设施披露 | — |
| **HCMS** | [2607.01817](https://arxiv.org/abs/2607.01817) | **B站 Bilibili** | 长序列 SP 的 head 分块多流流水，打破 all-to-all SP 的串行执行 | — |
| **SLAI T-Rex** | [2607.20145](https://arxiv.org/abs/2607.20145) | 深圳河套研究院 AI 训练平台团队 | 在**昇腾 SuperPOD 上对 DeepSeek-V4 全参数后训练** | ⭐ 国产芯片大模型后训练的少见公开记录 |
| **Mach-Mind-4-Flash** | [2607.09375](https://arxiv.org/abs/2607.09375) | **理想汽车**基础模型团队 | 35B-A3B MoE，纯后训练达到 100B 级表现 | — |
| **DisagFusion** | [2605.25550](https://arxiv.org/abs/2605.25550) | 商汤相关 | 扩散模型解耦服务的异步流水 + 弹性调度 | 吞吐 **3.4×–20.5×** |

**读法：** DynaTrain 是这批里工程价值最高的——它把"改并行策略"从**分钟级停机**变成了**秒级在线操作**，这直接改变了弹性训练和 RL 阶段切换的设计空间。

---

## 2. 海外大厂

### 2.1 Meta —— MTIA 300 + HCCL，硬件-通信协同设计

| 工作 | arXiv | 核心 |
|------|-------|------|
| **HCCL** | [2608.00358](https://arxiv.org/abs/2608.00358) （**SC'26**） | 与 **MTIA 300** 协同设计的集合通信库。MTIA 300 是 Meta **第一颗把后端网络直接集成进芯片封装**的芯片，带专用 message engine（ME）+ 近存计算（NMC），**把 collective 执行完全从计算网格卸载出去**，从而获得大幅 comp/comm overlap。HCCL 采用**编译式通信模型** |
| PRISM | [2607.21746](https://arxiv.org/abs/2607.21746) | 评估 POSIX 存储系统对 AI 研究工作流的影响——AI 研究优先的是**研究员迭代效率**，与传统 HPC 负载假设不同 |
| FlashAttention-4 | [2603.05451](https://arxiv.org/abs/2603.05451) | Princeton/**Meta**/Colfax/NVIDIA/Together 联合；面向非对称硬件扩展的算法-kernel 流水协同设计 |

**读法：** HCCL 这篇是本轮最值得读的。它给出的架构主张——**把 collective 从计算单元彻底卸载到专用引擎**——和我们在 megakernel 里纠结的"通信占用 CU"问题是同一个矛盾的两种解法：Meta 加硬件，我们只能在软件里挤。读它主要是看**编译式通信模型**怎么组织。

### 2.2 NVIDIA —— Megatron-Core 成体系输出 + 给 NCCL 补 EP

| 工作 | arXiv | 核心 |
|------|-------|------|
| **Megatron-Core MoE 训练报告** | [2603.07685](https://arxiv.org/abs/2603.07685) | 系统性讲清 MoE 训练在**显存/通信/计算三者耦合约束**下的权衡；覆盖十亿到万亿参数、千卡级集群 | 
| **NCCL EP** | [2603.13606](https://arxiv.org/abs/2603.13606) | 面对 DeepEP、Hybrid-EP 等设备发起通信库的既成事实，**给 NCCL 做统一的 EP 通信 API** |
| **Molt** | [2607.21653](https://arxiv.org/abs/2607.21653) | PyTorch 原生的 agentic RL 训练框架；出发点是主流框架里每改一个算法都要穿透 trainer/后端/rollout 三层胶水 |
| SOAP / Muon 扩展 | [2607.20548](https://arxiv.org/abs/2607.20548) | 把高阶优化器推到大规模预训练，定位并修复数值不稳定 |
| MoX | [2607.20220](https://arxiv.org/abs/2607.20220) | Technion + NVIDIA；直连/光交换拓扑上的 MoE 路由，**离线优化路由**即可，无需动态重配拓扑 |

**读法：** NCCL EP 值得单独注意——这是 NVIDIA 承认 DeepEP 那条**设备发起 RDMA** 路线赢了，开始把它收编进官方 collective 栈。RCCL 侧迟早要回答同一个问题。

### 2.3 AMD —— FP4 训练的数值定位 + 工具链

| 工作 | arXiv | 单位构成 | 核心 |
|------|-------|----------|------|
| **MXFP4 原生硬件预训练** | [2605.09825](https://arxiv.org/abs/2605.09825) | Penn State + **AMD** | 在 **MI355X 原生 MXFP4**（非软件模拟）上逐段打开 FP4：结论是 **Wgrad 量化才是收敛退化主因**，Fprop/Dgrad 影响温和；随机舍入和随机 Hadamard 旋转都救不回来，**确定性 Hadamard 旋转**才能恢复稳定 |
| Eidola | [2606.12638](https://arxiv.org/abs/2606.12638) | Wisconsin + **AMD Research** | 建模多 GPU 通信流量；重点是 kernel fusion 与 overlap 反而制造了**难以建模的不规则瞬态流量** |
| Kerncap | [2605.03208](https://arxiv.org/abs/2605.03208) | AMD 相关 | AMD GPU 上的**自动 kernel 抽取与隔离**，免去手工重建 build flag / dispatch 配置 |
| dMX | [2606.04115](https://arxiv.org/abs/2606.04115) | AMD 相关 | 可微的混合精度位宽分配 |

**读法：** MXFP4 那篇是我们这条线上**最该立刻读**的——它是少数在真实 MI355X 原生 FP4 上做的受控实验，而且结论是可操作的：**先保住 Wgrad，Fprop/Dgrad 可以激进**。Kerncap 则是现成的工程工具，能省掉不少 kernel 调优的隔离成本。

### 2.4 Microsoft + OpenAI —— 十万卡网络是联合叙事

| 工作 | arXiv | 核心 |
|------|-------|------|
| **MRC + SRv6** | [2605.04333](https://arxiv.org/abs/2605.04333) | **OpenAI + Microsoft + AMD + NVIDIA + Broadcom 联合**。同步预训练在超大规模下**由尾延迟主导**。三招：(1) 新 RDMA 传输 **MRC**，多路径喷洒 + 主动负载均衡，消除流冲突；(2) 多平面 Clos，让 **10 万卡以上**集群仍能做成两层拓扑并提高物理冗余；(3) **SRv6 静态源路由**让 MRC 自行绕开故障。已在 OpenAI 与 Microsoft 生产环境运行 |
| TileSight | [2607.22432](https://arxiv.org/abs/2607.22432) | 帝国理工 + 北大 + **MSRA**；面向 Triton/TileLang/CUDA Tile 这类 **tile 为一等公民**的框架的第一性原理解析性能模型，从核到集群 |

**读法：** MRC/SRv6 这篇的份量在于它是**五家联合署名的生产经验**，而且明确说了在 OpenAI 和微软生产环境跑。TileSight 与 FlyDSL 直接对位——tile 级编程范式已成主流，但性能分析工具还停在 roofline，这是个明显的工具缺口。

### 2.5 Google —— 基础设施组件化输出

| 工作 | arXiv | 核心 |
|------|-------|------|
| **Orbax** | [2605.23066](https://arxiv.org/abs/2605.23066) | JAX 原生的模块化分布式 checkpoint 库。保存比 PyTorch 对标方案快 **3.5×**、加载快 **2×**；已开源 |
| JAXBench | [2607.20466](https://arxiv.org/abs/2607.20466) | TPU 上的 AI 生成 kernel 优化基准，50 个 JAX workload——GPU 侧早有类似基准，TPU 侧此前空白 |

### 2.6 其他海外

| 工作 | arXiv | 单位 | 核心 | 关键数据 |
|------|-------|------|------|----------|
| **Optimus / Aurora** | [2604.00785](https://arxiv.org/abs/2604.00785) | Intel + Argonne | 在 Aurora（127,488 个 Intel PVC GPU tile）上预训练 MoE；自研 Optimus 库，含**EP 感知的分片优化器** | 12288 tile 上扩展效率 **~90%**，训练加速 **1.71×** |
| PCCL | [2606.07019](https://arxiv.org/abs/2606.07019) | Georgia Tech + **Intel Labs** | 进程组感知的通用集合通信**算法综合器** | — |
| **Expert Upcycling** | [2604.19835](https://arxiv.org/abs/2604.19835) | **Amazon Stores Foundation AI**（+CMU/Anthropic） | 用 upcycling 移动 MoE 的算力-效率前沿 | — |
| Mixture-of-Parallelisms | [2607.01844](https://arxiv.org/abs/2607.01844) | **Salesforce AI Research** | MoE 训练栈按**层与阶段分别特化**并行策略，而非全局统一 | — |

---

## 3. 横切主线

### 3.1 FP4 训练：战场从格式转到"哪一段不能量化"

| 谁 | 结论 |
|----|------|
| AMD + Penn State | **Wgrad 是发散主因**；确定性 Hadamard 旋转有效，随机化无效 |
| 华为 | RL 场景下**rollout 激活量化**才是主因；只提升训练侧精度反而更差 |
| 蚂蚁 | **E2M1 格式本身**带来收缩偏差，而这是 NVIDIA 与 AMD 共同的硬件押注 |
| 腾讯 | FP8 attention 里 Attention Sink 导致 P 矩阵 cast 崩塌 |

**共同点：** 四家都从"整体能不能 FP4"下沉到了**逐段定位敏感路径**。这条线上如果要做 AMD 侧工作，起点应该是复现 Wgrad 结论并检查 MI355X 上的 Hadamard 旋转开销。

### 3.2 RL 后训练：不能沿用预训练的调度假设

- **负载统计口径变了**：腾讯 ForeMoE——step 级稳定 / micro-step 级剧烈抖动。
- **阶段异构**：阿里 RollArt——prefill 计算密集、decode 带宽密集、环境执行 CPU 密集、奖励突发，四类必须分池。
- **精度失配**：华为——rollout 与 training 的量化必须**同步**考虑，单侧提精度会更差。
- **框架分层成本**：NVIDIA Molt——算法迭代被 trainer/后端/rollout 三层胶水拖累。

### 3.3 超节点重写通信库

NCCL/RCCL 的两个假设在超节点上同时失效：**buffer-centric 的内存语义**（StrataCL 攻击点）和**通用 collective 抽象对稀疏 MoE 流量的不适配**（UBEP、NCCL EP、MoX 攻击点）。

| 方案 | 谁 | 改什么 |
|------|----|--------|
| StrataCL | 华为 | registration-on-allocation，用户 buffer 直通，零冗余拷贝 |
| UBEP | 华为 + 南大 | 针对 CloudMatrix384 重构 EP 通信，破执行串行化 |
| NCCL EP | NVIDIA | 官方收编设备发起 RDMA 的 EP 语义 |
| HCCL | Meta | 硬件卸载 + 编译式通信模型 |
| PCCL | Intel | 进程组感知的算法综合 |

### 3.4 弹性：从 checkpoint 到状态直传

DynaTrain（VPS 抽象，235B MoE **4.36s**）与 ETC（P2P 直传，**2.33–6.37×**）殊途同归：**存储持久化不该出现在并行重配的关键路径上**。

### 3.5 编译器化

字节 DITRON、MSRA 系 TileSight、Google JAXBench——共同信号是 **tile 级抽象已成事实标准，但配套的性能模型与基准仍是缺口**。

---

## 4. 对我们三条线的启示

### ROCmoe / MonolithEP

1. **[高优先级] 读 Meta HCCL。** 它把"collective 占用计算单元"这个我们在 megakernel 里绕不开的矛盾，用专用 message engine 卸载解决了。我们没有这个硬件，但**编译式通信模型**的组织方式可以借鉴——尤其是它如何在编译期确定通信调度。
2. **[高优先级] 读华为 UBEP。** 超节点上"统一地址空间 + 高带宽 ≠ 稀疏 MoE 高性能"，它列的三个瓶颈（首要是**执行严格串行化**）几乎肯定在 MI300/MI400 的 XGMI 域内同样成立。
3. **RCCL 的缺口比想象的大。** 不只是 EP API：NCCL 2.28 里对 MoE 训练最值钱的是 **Copy Engine collectives**（用 copy engine 驱动传输，把 SM 从 alltoall/allgather 里释放出来）和 **device API**（kernel 内发起通信）。ROCm 7.2 公开的 RCCL 进展只到"4-NIC 拓扑感知 + backport NCCL 2.28 算法"。AMD 已有 rocSHMEM 和 mori SDMA allgather 作基础，缺的是**框架可直接消费的 API 面**。
4. **sync-free 和整迭代 graph capture 应该合流。** AMD 两件事都做了但没合并：`--turbo_sync_free_moe_stage` 消除了 D2H 同步，MLPerf 里的 Flux 用 HIP graph 整迭代捕获拿到 +5.3%。NVIDIA 的经验说明**二者相乘才是大头**——消除同步的真正价值不在省那点开销，而在于它让整迭代 graph capture 成为可能（GPT-OSS +93%）。
5. **参数与通信的原生低精度化是最明确的差距。** Primus v26.5 已在 Flux 路径做了 FSDP2 的 fp8 all-gather，说明基础设施存在；把它推广到 MoE/dense 主线并推进到 FP4，对标 NVIDIA 的 `--fp4-param-gather`。
6. **负载均衡不要直接从预训练复用到 RL。** ForeMoE 的 micro-step 抖动结论要先验证。

### FlyDSL

1. **DITRON（字节）是最直接的对位工作**——同样是"不靠 cuBLAS+NCCL 拼装、直接编译分布式张量程序"。必须读，确认我们的差异化在哪。
2. **TileSight 是我们缺的那块**：tile 级 DSL 有了，但性能模型还是 roofline。如果 FlyDSL 想自动调度，需要类似的解析模型。
3. **Kerncap（AMD）可以直接用**：自动抽取隔离 kernel，省掉调优时重建 build flag 的成本。
4. **要抢的生态位是"框架团队的快速补位能力"，不只是"写 kernel 更快"。** NVIDIA 那个 +93% 的 MoE 融合内核就是 CuTe DSL 的产物，且有 DSL → cuDNN Frontend → TE → Megatron-Core 的完整分发链路。FlyDSL 在 Primus-Turbo 里已承担 GEMM/GroupedGEMM/MoE 主力后端，方向对；差的是**分发链路的完整性**，以及像 AWS NKI Compiler 那样完整开源以吸引社区。

### Primus

1. **弹性训练：DynaTrain 的 VPS 抽象值得评估**。Primus 的 pipeline runtime 如果要支持在线并行切换，VPS 那套"统一逻辑坐标系 + 几何求交"是目前最干净的形式化。
2. **ETC 已经集成进 Megatron-LM**，迁移路径清晰，可作为 checkpoint-free 弹性的参考实现。
3. **NVIDIA Megatron-Core MoE 报告**应该当作对标基线通读一遍——它把显存/通信/计算的耦合约束讲得比任何单篇论文都系统；其 **Parallel Folding**（解耦 attention 与 MoE 的并行配置，打破 `EP ≤ DP` 约束）是 Primus 并行层可直接对照的设计。
4. **容错的算法层是完全空白，而这对 AMD 尤其有价值。** Primus-SaFE 目前是"检测 + 调度 + 重启"范式；Decoupled DiLoCo 展示的是另一层——quorum + grace window 让单 learner 故障不阻塞其他 learner，且**不同代 TPU 混在同一次训练里无退化**。多云、多站点、MI300X/MI325X/MI355X 多代混跑本来就是 AMD 客户的现实处境。Meta Monarch 的**分级恢复**（先进程级重启，必要时才升级到作业重分配，健康 replica 继续步进）是更容易落地的中间态。
5. **MI300A 的 Superchip offloading 是被浪费的独有优势。** 微软 SuperOffload 明确把 MI300A 列为目标平台，但 GraceAdam（比 PyTorch CPU Adam 快 3×）只实现了 Grace 版。MI300A 的 APU 统一内存在架构上比 GH200 的 NVLink-C2C 更激进，却没有对应的 ROCm 侧 CPU 优化器实现。

---

## 5. 检索方法与可复现性

**为什么不能只用 arXiv API：** 本轮检索中 `export.arxiv.org` 持续返回 **HTTP 429**（每次请求挂起约 32 秒后失败）。改用 arXiv **网页高级检索**（`https://arxiv.org/search/advanced`）绕开，网页端不受该限流影响。

**三步流程：**

1. **机构定向检索** —— 26 个机构名 × 日期区间 2026-02-01 ~ 2026-09-01，限 CS 分类 → 2498 篇去重候选。
2. **训练系统相关度打分** —— 关键词加权 + **硬门槛**（必须命中并行策略/集合通信/checkpoint/低精度格式/框架名/集群规模之一），避免"只是提了一句 we train"的模型论文混入 → 209 篇。
3. **作者单位核实** —— 逐篇拉 `https://arxiv.org/html/<id>` 的 `ltx_authors` 块，**截断到摘要标记之前且限长 1500 字符**，只在作者块内匹配机构名与**邮箱域名**。

**这一步不能省。** 早期版本把摘要正文一起送进匹配，导致正文里的 "NVIDIA A100 GPU"、"DeepSeek-R1" 被误判成作者单位；即便只取作者块，若不限长也会串进正文。最终人工抽查还纠正了两处：`Expert Upcycling` 实为 **Amazon** 主导（非 Anthropic），`UCCL-Zip` 作者明确声明与其 Amazon 职务无关，**不应计为亚马逊成果**。

```bash
# 机构定向检索（网页端，绕开 API 限流）
python3 /tmp/org_websearch.py      # -> /tmp/org_pool2.json

# 训练系统相关度打分 + 硬门槛
python3 /tmp/org_rank.py           # -> /tmp/org_ranked.json

# 作者单位核实（只匹配作者块 + 邮箱域名）
python3 /tmp/org_affil.py /tmp/org_merged.json /tmp/org_final.json

# 高分但未识别单位的，打印上下文窗口人工判读
python3 /tmp/org_resolve.py
```

**已知局限：**
- 约 113/854 篇无 HTML 渲染（仅 PDF），作者单位无法自动提取，这部分靠 abs 页人工补。
- §1–§2 只覆盖 arXiv，而**顶会论文与厂商技术报告不一定进 arXiv**：阿里 Tessera（OSDI '26）、DeepSeek-V4 基础设施章节、Kimi K3 的 MoonEP、美团 LongCat-2.0 都被这一步系统性漏掉，全靠 §6 的开源/博客检索补回。只扫 arXiv 会**结构性低估中国大厂的生产级工作**。→ 补救办法见 [`paper-venues-checklist.md`](./paper-venues-checklist.md)：按会议直接检索，Tier 0 那批（OSDI/SOSP/EuroSys/ATC/NSDI/ASPLOS/SC/FAST）就是 arXiv 覆盖不到的部分。
- 单位归属按论文作者块判定，因此"某公司员工的个人研究"与"公司项目产出"无法严格区分（UCCL-Zip 即为一例，已手工排除）。

---

## 6. 开源与工程博客侧

> 本节覆盖 arXiv 之外的产出：开源框架 release notes、官方工程博客、会议分享。**与 §1–§2 的 arXiv 结论互为独立来源**，交叉印证与冲突见 §6.3。

### 6.1 国内开源栈

| 公司 | 关键产出 | 时间 | 要点 |
|------|----------|------|------|
| **字节** | [veScale-FSDP](https://www.arxiv.org/pdf/2602.22437) | 2026-03 | 重写 PyTorch FSDP2，`RaggedShard` 抽象支持 **Muon 等非逐元素优化器**；吞吐 +5~66%、峰值显存 -16~30%；内部最大 **2.4T 参数 / 10K GPU** |
| | [verl v0.8.0](https://github.com/verl-project/verl/releases/tag/v0.8.0) | 2026-06 | 统一 engine 抽象（废弃旧 worker）；Megatron-FSDP 模式、动态 CP、**NVFP4 (W4A16) QAT**、**MoE router replay (R2/R3)**、**Ascend 950 上 MXFP8 rollout** |
| | [VeOmni](https://github.com/ByteDance-Seed/VeOmni/issues/271) | 2026 Q1–Q2 | 昇腾原生适配：NPU 专用 CI、EP / Async Ulysses CP、GMM/FA/RMSNorm/RoPE 融合算子 |
| **阿里** | **Tessera**（[OSDI '26](https://www.usenix.org/system/files/osdi26-hu-weifang.pdf)） | 2026 | ⭐ 本轮数字最硬的一篇。为**异构 MoE**（稀疏 MoE + 多种 attention 变体混合）重做流水并行：用**重叠后实测开销**做切分、运行时用可移动任务填补路由空泡。**4,096–12,288 GPU 生产负载上 +20%~33%，万亿模型 39% MFU**；已用于 Qwen3 / Qwen3-Next 预训练。**已精读** → [`papers/tessera.md`](../../papers/tessera.md)（§5 工程经验最有价值：生产里否掉 Comet 式融合、EP 通信 kernel 占 ~20 SM 致 10–20% 减速、代价模型尾部误差翻转 MILP 排序） |
| **腾讯** | [官方发布稿](https://www.tencent.com/zh-cn/tencent-hunyuan-officially-releases-hy3-advancing-agent-capabilities-and-deeper-product-integration/) | 2026-01~07 | 承认自 2026-01 底**重建预训练与 RL 基础设施**；Hy3 的 RL 用 **verl + Megatron-LM + vLLM 开源栈**而非自研 AngelPTM。**预训练系统细节未公开** |
| **华为** | [hyper-parallel](https://gitcode.com/mindspore/hyper-parallel/blob/master/docs/guide/multicore_moe.md) | 2026-05 | HyperParallel-MoE 的 MindSpore 实现，含 O0/O1 两档调度下沉 |
| | [MindSpeed 26.0/26.1](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/release_notes_core.md) | 2026-03/06 | Atlas A3 的 **AI QoS 训练流量优先级配置**、KVAllgather CP、FSDP 后端 Device 解耦 |
| **百度** | [ERNIE 5.0 报告](https://arxiv.org/abs/2602.04705) | 2026-02 | 2.4T 参数；**TP4 + PP12 + EP64 + ZeRO-1 + CP**，跨机用 **DeepEP**，FP8 混合精度，**tokenizer 与 MoE backbone 解耦部署在独立 GPU 节点** |
| | [飞桨 3.3](https://github.com/PaddlePaddle/Paddle/releases/tag/v3.3.0) | 2026-01/03 | FlashMask V3（前向持久化抢占式 Tile 调度器做 SM 间均衡，单卡领先 FlexAttention 2.1×）；**VMM Allocator** 把 MoE 显存碎片率压到 3% |
| **月之暗面** | [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 2026-07 | ⭐ 训练侧信息密度最高的一次发布。**Quantile Balancing 直接去掉负载均衡辅助损失**；**Per-Head Muon**；**从 SFT 起 MXFP4 权重 + MXFP8 激活 QAT**；同时开源 **MoonEP**（完全均衡执行 / 静态计算形状 / 零拷贝 / 关键路径无 host 同步）、FlashKDA、AgentEnv |
| **智谱** | [slime](https://github.com/THUDM/slime/releases) | 月级迭代 | GLM-5/5.1/5.2 的 RL 栈。v0.3.1 新增 **FLOPs 均衡的 micro-batching**、top-p masking 让训练复现 rollout 采样分布。官方文档诚实标注 **FP8 训练 + FP8 rollout 仍是 experimental** |
| **DeepSeek** | [V4 报告](https://arxiv.org/abs/2606.19348) | 2026-06 | **单 fused MoE kernel 完全重叠计算/通信/访存**；用 **TileLang** 写 kernel；提供 **batch-invariant 确定性 kernel 库保证训推位级可复现**；Muon 的 hybrid ZeRO（背包算法做矩阵到 rank 均衡）；**MoE 专家权重 FP4 QAT**——关键巧思是 MXFP4→FP8 反量化无损，既有 FP8 管线零改动复用 |
| **美团** | [LongCat-2.0](https://github.com/meituan-longcat/LongCat-2.0) | 2026-06 | ⭐ 国产芯片规模化训练的存在性证明：**1.6T 参数 / 35T tokens / 5 万张国产 ASIC 全流程训练，无回滚**。**6D 并行**（新增 EMBP 加速 N-gram Embedding）；超节点额外 +30% 吞吐；MFU 提升 1.5×；稳态日吞吐 >1T tokens |
| | [DORA](https://doi.org/10.48550/arxiv.2604.26256) | 2026-04 | 异步 RL 的三约束（轨迹内策略一致 / 数据完整 / 有界 staleness）+ 多版本流式 rollout 消除空泡；生产环境端到端 2–4× |
| **小米** | [MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash) | 2026-01 | R3 + Request-Level Prefix Cache（缓存 KV **与路由专家**）+ 序列级细粒度调度 + partial rollout |

**未找到公开产出（已核实，非遗漏）：** 腾讯 Hy3 的预训练系统细节；阿里 PAI 工具链（ChatLearn / Pai-Megatron-Patch）2026 年的重大更新；快手 KRL/KwaiEnv 的开源版本；阶跃 2026-03~08 的独立训练系统披露；小米 2026-03 后的新产出。

### 6.2 海外开源栈

| 公司 | 关键产出 | 时间 | 要点 |
|------|----------|------|------|
| **AMD** | [MLPerf Training v6.0](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html) | 2026-06 | ⭐⭐ 首次用 **Primus** 提交 + 生产级 MXFP4 配方。**整条量化流水线融成单个 HIP kernel**（Hadamard 旋转→scale→FP4 cast→packing→shuffling→scale swizzling）；**确定性 16-point Hadamard 旋转**；两阶段 healing（MXFP4 → 单步切 FP8 → FP8 收尾，FP8 权重副本预置 pinned CPU 内存避免显存尖峰）。相对 v5.1 **+13~19%**；对比 B200 同卡数差距约 5% |
| | [FlyDSL](https://github.com/ROCm/FlyDSL) | 2026-02/03 | `fly` MLIR dialect + CuTe 风格 layout algebra；已在 Primus-Turbo 承担 FP8 grouped GEMM（[PR #384](https://github.com/AMD-AGI/Primus-Turbo/pull/384)）与 MXFP8 dense GEMM（[PR #390](https://github.com/AMD-AGI/Primus-Turbo/pull/390)）主力后端 |
| | [1K GPU MoE 预训练](https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/) | 2025-12 | 与 Meta PyTorch 合作。分项收益：AITER attention +15% → tensorwise FP8 GEMM +102% → FP8 grouped GEMM +60%，累计 **2.77×**；1024 卡 96% scaling。**DeepEP 的量化价值**：EP 8→32 时朴素 A2A 吞吐 2000→750 TPS、通信占比 10%→50%，换 DeepEP 后稳在 2000–2100 TPS、通信封顶 18% |
| | [Primus v26.5.0](https://github.com/AMD-AGI/Primus/releases) | 2026-07 | **DeepSeek-V4 训练支持**（含 Muon 优化器、FP8/FP4）；Flux 训练后端的 **FSDP2 fp8 all-gather**；移除 grouped MLP 的 D2H 同步 |
| **NVIDIA** | [GTP / `--fp4-param-gather`](https://docs.nvidia.com/megatron-core/developer-guide/nightly/api-guide/core/generalized_tensor_parallel.html) | 2026 | ⭐⭐ 分片保持为 native `NVFP4Tensor`，以 **packed 4-bit 直接 all-gather**，distributed optimizer 每 step 只写一次分片，**前向零 per-microbatch 量化** |
| | [CuTe DSL MoE 融合内核](https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/) | 2026-06 | kernel 级 1.3–2×，但真正的收益来自**消除同步后整个迭代可被 CUDA Graph 捕获**：DeepSeek-V3 预训练 **+8%**、GPT-OSS **+93%** |
| | [NCCL 2.28](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/) | — | **Device API**（kernel 内发起通信，LSA / Multimem / **GIN** 三后端）+ **Copy Engine collectives**（用 copy engine 驱动传输，把 SM 从 alltoall/allgather 里释放出来） |
| | [NeMo-RL v0.6.0](https://github.com/NVIDIA-NeMo/RL) | 2026-04 | 端到端 FP8 GRPO（linear W8A8 >15%，加 FP8 KV/attention 后 rollout 额外 ~30%，端到端 ~48%）；**Muon 优化器**；non-colocated GRPO 让 2048 GPU 任务启动加速 2× |
| **Meta** | [MTIA 300](https://about.fb.com/news/2026/03/expanding-metas-custom-silicon-to-power-our-ai-workloads/) | 2026-03 | **已量产用于排序/推荐训练**。反主流策略：先为 GenAI 推理优化，再支持排序推荐训练与 GenAI 训练。软件栈全押 PyTorch 原生 + **Triton-MTIA**（[arXiv:2608.00325](https://arxiv.org/html/2608.00325)） |
| | [torchtitan v0.2.2](https://github.com/pytorch/torchtitan/releases) | 2026-02 | CP 重构到新 API；Compiler Toolkit **为 FSDP 的 AG/RS 分离独立 process group**；DeepEP `shared_experts` 与 combine 重叠（**由 AMD 提交**）；ROCm gfx950 mxfp8 支持 |
| | [PyTorch 2.10/2.11](https://pytorch.org/blog/pytorch-2-11-release-blog/) | — | 为分布式 RL 做确定性：`torch.compile` 尊重 deterministic mode、DebugMode 定位数值发散；**Differentiable Collectives**；c10d 暴露 `shrink_group`（弹性基础） |
| **Google** | [Ironwood 训练指南](https://cloud.google.com/blog/products/compute/training-large-models-on-ironwood-tpus) | 2026-03 | 原生 FP8 MXU；Tokamax 内核（Splash Attention、**Megablox GMM** 处理 MoE ragged 张量避免 padding）；**把 All-Gather/Reduce-Scatter 卸载到 SparseCore**，TensorCore 专注计算 |
| | [Decoupled DiLoCo](https://deepmind.google/blog/decoupled-diloco/) | 2026-04 | ⭐ **把容错做进算法层**：独立异步 learner unit + **minimum quorum + adaptive grace window + token-weighted merging**。12B 模型跨 4 个 region、仅 2–5 Gbps 广域网训练快 20 倍以上；高故障率下 goodput **88% vs 数据并行 27%**（后者为二手数字） |
| **Microsoft** | [SuperOffload](https://pytorch.org/blog/superoffload-unleashing-the-power-of-large-scale-llm-training-on-superchips/)（ASPLOS'26） | 2026-03 | 重新审视 offload 的前提：NVLink-C2C 900 GB/s vs PCIe Gen4 64 GB/s。**GraceAdam** 比 PyTorch CPU Adam 快 3×。**官方把 MI300A 列为目标平台**，但 CPU 优化器只实现了 Grace 版 |
| | [DeepSpeed + Muon](https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/) | 2026-06 | 把 Muon 更新挪进 ZeRO stage 1/2 的 `get_flat_partition`（此时梯度尚未扁平化）；ZeRO-3 走 **mori 的 SDMA allgather**（AMD 相关） |
| **OpenAI** | [MRC](https://openai.com/index/mrc-supercomputer-networking/) | 2026-05 | 与 AMD/Broadcom/Intel/Microsoft/NVIDIA 共同开发两年，**已贡献给 OCP**。把 800Gb/s 接口拆成 8×100Gb/s 提高交换机 radix，使 **~131,000 GPU 只需两层 Clos**。已在其**全部最大 GB200 集群**上训练前沿模型——某次前沿训练期间重启 4 台 tier-1 交换机，**无需与训练团队协调** |
| **AWS** | [Neuron 2.30/2.31](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html) | 2026-05/07 | NKI Library 新增 CP、**MXFP8 training**、fused optimizers、**MoE dispatch / MoE training collectives**、ring attention；**NKI Compiler 以 Apache 2.0 开源，基于 MLIR** |

**未找到公开产出：** xAI、Mistral、Cerebras 的 2026 年训练系统一手技术披露；Anthropic 自身的训练系统细节（只有与 AMD 的 2 GW MI455X 合作公告）。

### 6.3 两个来源的交叉印证与冲突

**互相印证（可信度显著提高）：**

1. **AMD MXFP4 的 Wgrad 结论**：arXiv [2605.09825](https://arxiv.org/abs/2605.09825)（Penn State + AMD，MI355X 受控实验）与 MLPerf v6.0 官方博客（生产提交）**独立给出同一结论**——Wgrad 量化是发散主因，确定性 Hadamard 旋转有效而随机化无效。学术与生产两侧对上，这条可以当结论用。
2. **OpenAI MRC**：arXiv [2605.04333](https://arxiv.org/abs/2605.04333) 与 OpenAI 官方博客 + OCP 贡献互证，且补上了"已在全部最大 GB200 集群运行"这个部署事实。
3. **Meta MTIA**：arXiv 的 HCCL（SC'26，与 MTIA 300 协同设计）与 Meta 官方博客（MTIA 300 已量产用于排序推荐训练）拼成完整图景。
4. **华为 HyperParallel-MoE**：arXiv 论文与 MindSpore 仓库实现对得上，且仓库补出了 O0/O1 调度下沉档位。

**⚠️ 一处实质冲突，值得深挖：**

**FP4 训练中随机性的作用，AMD 与 NVIDIA 结论相反。** NVIDIA Transformer Engine 的 `NVFP4BlockScaling` **默认开启 stochastic rounding + Random Hadamard Transform**；AMD 在 MLPerf v6.0 博客与 arXiv 论文中**两次明确指出 stochastic rounding 和 randomized Hadamard 都稳不住 Wgrad，必须用确定性旋转**。两边都是生产级证据。可能的解释是格式差异（NVFP4 的 E4M3 block scale vs MXFP4 的 E8M0）或 block size 差异（16 vs 32），但没有公开材料给出定论——**这是一个真实的开放问题，也是我们做 AMD FP4 训练时绕不开的第一个决策点。**

**arXiv 完全漏掉的重要工作：** 阿里 **Tessera**（OSDI '26，未上 arXiv）、DeepSeek-V4 的基础设施章节、Kimi K3 的 MoonEP、美团 LongCat-2.0 的 5 万卡国产 ASIC 训练。**教训：顶会论文与厂商技术报告不一定进 arXiv，只扫 arXiv 会系统性漏掉中国大厂的生产级工作。**

### 6.4 生命周期告警（会影响选型）

- **Meta torchforge 开发已暂停**，PyTorch 的 LLM 训练正在向 **torchtitan 收敛**。若有跟进 TorchForge 的计划需重新评估。
- **AWS NeuronX Distributed Training (NxDT) 自 2.29 起 EOS**，2.31 起移出 DLAMI；`pytorch-training-neuronx` DLC 不再发布。取而代之的 **TorchNeuron** 走 PyTorch PrivateUse1 backend，原生 FSDP/DTensor，**兼容 TorchTitan**。
- **AMD `rocm/pytorch-training` 镜像将被 `rocm/primus` 取代**（[ROCm 文档](https://rocmdocs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-pytorch.html)，2026-06-30）。
- **字节 verl v0.8.0 废弃旧 FSDP/Megatron worker**，全部迁到统一 engine 抽象（破坏性变更）。

### 6.5 两侧共同的新主线：Muon 成为跨栈共识

这是只看 arXiv 看不出来的一条。Muon 已经同时落进：字节 veScale-FSDP（`RaggedShard` 专为非逐元素优化器设计）、DeepSeek-V4（hybrid ZeRO + 背包算法均衡矩阵分配 + BF16 梯度量化通信减半）、Kimi K3（Per-Head Muon）、微软 DeepSpeed（ZeRO 2/3 已合并）、NVIDIA NeMo-RL v0.6、AMD Primus v26.5。**优化器从算法选择变成了系统约束**——它要求分布式框架能保住参数的 2D 结构，这与现有把梯度当扁平 buffer 的设计直接冲突。
