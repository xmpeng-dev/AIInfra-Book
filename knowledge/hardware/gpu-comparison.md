# 数据中心 AI 加速卡参数对照：NVIDIA vs AMD

> **范围**：面向 **训练 / 推理 / HPC** 的数据中心 GPU 与 **AMD Instinct** 系列；**不包含** GeForce 游戏显卡。  
> **说明**：下列为厂商规格页 / datasheet 中的**标称值**；同一芯片常有 **SXM、PCIe、NVL** 等不同封装，功耗多为可配置范围。Tensor Core / Matrix Core 的峰值算力高度依赖 **稀疏、精度与测试定义**，跨厂商数字**不可直接等同为实际吞吐**，应以目标框架与实测为准。

---

## 1. NVIDIA 数据中心 GPU（训练 / 推理主力）

| 型号 | 架构 | GPU HBM（典型） | 标称 GPU 内存带宽 | 峰值 FP8 Tensor（标称，常含稀疏） | 典型功耗 (TDP/TGP) | 常见形态 |
|------|------|-----------------|-------------------|-----------------------------------|----------------------|----------|
| A100 80GB | Ampere | 80 GB HBM2e | 2.0 TB/s | —（一代 Tensor，常用 FP16/BF16 对比） | 约 400 W（SXM） | SXM4 / PCIe |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 3,958 TFLOPS | 最高约 700 W（可配置） | SXM5 |
| H100 NVL | Hopper | 94 GB HBM3 | 3.9 TB/s | 3,341 TFLOPS | 约 350–400 W（可配置） | PCIe，可 NVLink 桥接多卡 |
| H200 SXM | Hopper | 141 GB HBM3e | 4.8 TB/s | 3,958 TFLOPS（与 H100 同算力档，内存更大更快） | 最高约 700 W（可配置） | SXM5 |
| H200 NVL | Hopper | 141 GB HBM3e | 4.8 TB/s | 3,341 TFLOPS | 最高约 600 W（可配置） | PCIe，多卡 NVLink |
| B200 | Blackwell | 192 GB HBM3e | 约 8 TB/s | Blackwell 一代 Tensor（含 FP4 等；见 datasheet） | 约 1000 W 量级 | SXM（机架方案） |
| GB200 Grace Blackwell Superchip | Grace CPU + **2×** Blackwell GPU | 两颗 Blackwell 各带 HBM3e（整机架 NVL72 等方案中再经 NVLink 扩展） | 单 Superchip 的 GPU HBM 侧标称约 **13.4 TB/s**（NVIDIA GB200 NVL72 规格表） | 单 Superchip FP8 标称约 **20 PFLOPS**（稀疏口径以官方表为准） | 液冷机架 / Superchip 模块 | Grace 与 GPU 经 NVLink-C2C |

**Blackwell 架构共性（摘要）**：双芯粒（reticle-limited die）经高带宽片间互联拼成一颗逻辑 GPU；第二代 Transformer Engine，强调 FP8 / FP4 等低精度与大规模 MoE / 长上下文场景。具体峰值请以 [Blackwell 技术简报 / datasheet](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) 为准。

### 1.1 常见「非 HBM」推理卡（仍属数据中心 AI 场景）

| 型号 | 架构 | 显存 | 带宽量级 | 备注 |
|------|------|------|----------|------|
| L40S | Ada Lovelace | 48 GB GDDR6（ECC） | 864 GB/s | 功耗较低、PCIe 易部署，适合部分推理与图形/视频类负载；大模型单卡显存通常弱于 HBM 卡 |

---

## 2. AMD Instinct（CDNA）AI / HPC 加速卡

| 型号 | 架构 | HBM | 标称内存带宽 | 峰值 FP8（标称） | 典型板卡功耗 (TBP) | 形态 |
|------|------|-----|--------------|------------------|--------------------|------|
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | 2.61 PFLOPS（5.22 PFLOPS 结构化稀疏） | 750 W | OAM（UBB） |
| MI325X（单 OAM） | CDNA 3（MI300 系列演进） | 256 GB HBM3E | 6 TB/s（AMD MI325X 平台页：per OAM） | 与 MI300X 同档标称 FP8（见该系列 datasheet） | 以 datasheet 为准 | OAM，与 MI300X 平台兼容演进 |
| MI355X | **CDNA 4** | **288 GB HBM3E** | **8 TB/s** | OCP-FP8 **5 PFLOPS** / **10.1 PFLOPS**（结构化稀疏）；另标称 MXFP4/MXFP6 矩阵 **10.1 PFLOPS**、MXFP8 矩阵 **5 PFLOPS** | **1400 W**（TBP） | OAM（UBB 2.0），被动/主动散热 |

**平台级参考（8× MI325X）**：总 HBM3E **2.048 TB**，单 OAM 带宽 **6 TB/s**（[MI325X Platform](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x/platform.html)）。

**平台级参考（8× MI355X）**：总 HBM3E **2.3 TB**，单 OAM 带宽 **8 TB/s**（[MI355X Platform](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html)）。

上一代 **MI250X**（CDNA 2，HBM2e、双 GCD）在部分旧集群仍可见；新采购对照一般以 **MI300X / MI325X / MI355X** 与 **H100 / H200 / B200** 为主。

---

## 3. AMD MI355X 与 NVIDIA B200 对比

两者均为 **单卡 HBM3E、约 8 TB/s 内存带宽** 量级，但 **容量、功耗、架构代际与软件生态** 差异很大；下表仅列 **厂商公开标称**，算力口径（密集 / 稀疏、OCP-FP8 / NVFP4 等）**不一致时不可直接比大小**。

| 项目 | AMD Instinct **MI355X** | NVIDIA **B200** |
|------|-------------------------|-----------------|
| **GPU 架构** | CDNA 4（TSMC **3 nm** + 6 nm） | Blackwell（TSMC **4NP**，双芯粒拼单 GPU） |
| **单卡 HBM 容量** | **288 GB** | **192 GB** |
| **标称内存带宽** | **8 TB/s** | **约 8 TB/s**（常见 datasheet / 摘要表述） |
| **典型板卡功耗** | **1400 W**（TBP） | **约 1000 W**（TDP，可因配置略变） |
| **低精度矩阵峰值（标称）** | OCP-FP8：**5 PFLOPS**（密集）/ **10.1 PFLOPS**（结构化稀疏）；MXFP4/MXFP6 矩阵 **10.1 PFLOPS**；MXFP8 矩阵 **5 PFLOPS**（[MI355X 规格页](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)） | 第五代 Tensor Core；业界常见引用约 **FP8：4.5 PFLOPS（密集）/ 9 PFLOPS（稀疏）**，**FP4（NVFP4）稀疏峰值更高**（具体以 [Blackwell / DGX B200 资料](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) 为准） |
| **形态与互联** | OAM；**7×** Infinity Fabric（单链路最高 **153 GB/s**），含 scale-out 链路 | 数据中心 **SXM** 等；多卡依赖 **NVLink / NVSwitch** 与机架方案（与 Hopper 代际相比为第五代 NVLink 体系） |
| **软件栈** | **ROCm**；强调 MXFP6/MXFP4 等与开放微缩格式 | **CUDA**、TensorRT-LLM、NeMo 等；NVFP4 / Transformer Engine 与 NVIDIA AI 软件绑定深 |

**简要解读（选型向）**

- **显存**：MI355X **288 GB** 对单卡超大模型、长上下文、少卡切分更友好；B200 **192 GB** 更依赖多卡并行或量化。  
- **机柜与供电**：MI355X **1400 W** 单卡 TBP 明显高于 B200 **约 1000 W**，对风道、液冷、机柜功率密度与配电要求更苛刻。  
- **性能**：纸面 PFLOPS 需对齐 **精度（FP8 / MXFP8 / NVFP4）** 与 **是否稀疏**；真实吞吐还受 **内存带宽、通信、算子融合与框架支持** 支配。  
- **生态**：训练/推理栈若深度依赖 **CUDA 与 NVIDIA 闭源优化路径**，B200 往往「默认路径」更顺；若集群已统一 **ROCm** 或追求 **开放格式与 AMD 平台报价**，MI355X 是同期对位选项之一。

---

## 4. 选型时「看什么」——与游戏卡无关的维度

| 维度 | 说明 |
|------|------|
| **单卡 HBM 容量** | 决定能否单卡放下权重、能否开更大 batch / 更长上下文；**MI355X 288 GB**、**MI325X 256 GB** 与 **B200 192 GB**、**H200 141 GB** 常是横向对比焦点。 |
| **内存带宽** | LLM / 大矩阵算子多为 memory-bound，**TB/s 级带宽**与 **HBM 代际**（HBM3 / HBM3e）直接影响吞吐。 |
| **互联与拓扑** | NVLink / NVSwitch、InfiniBand、以及 AMD Infinity Fabric 等多卡/多机扩展能力，往往比单卡峰值算力更决定集群效率。 |
| **软件栈** | NVIDIA：**CUDA / cuDNN / TensorRT / Triton** 等与主流框架绑定深；AMD：**ROCm**，需核对发行版、驱动与具体模型/算子支持矩阵。 |
| **精度与稀疏** | FP8 / FP4、结构化稀疏等会显著改变「纸面 TFLOPS」与可部署模型，对比时要看 **是否同精度、同稀疏定义**。 |

---

## 5. 官方参考（更新规格请以 datasheet 为准）

- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) · [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) · [Blackwell 架构](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) · [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) · [L40S](https://www.nvidia.com/en-us/data-center/l40s/)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) · [MI325X Platform](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x/platform.html) · [MI355X GPU](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) · [MI355X Platform](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html)

---

*文档侧重「AI 训练/推理用数据中心卡」；若需补充 **RTX PRO 数据中心版**、**边缘 AI 卡** 或 **云厂商定制 SKU**，可按同一表格字段再扩展一节。*
