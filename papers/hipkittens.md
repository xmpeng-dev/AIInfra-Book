# HipKittens: Fast and Furious AMD Kernels

> [MLSys '26](https://proceedings.mlsys.org/paper_files/paper/2026/file/bc75fa9843a7905bbed9d83895a88f7f-Paper-Conference.pdf) · [arXiv 2511.08083](https://arxiv.org/abs/2511.08083)（2025-11-11）· [博客](https://hazyresearch.stanford.edu/blog/2025-11-09-hk) · [代码](https://github.com/HazyResearch/HipKittens)
> William Hu, Drew Wadsworth（Stanford）· Sean Siddens, Stanley Winata, **Ryan Swann, Muhammad Osama（AMD）** · Daniel Y. Fu（UCSD）· Christopher Ré, Simran Arora（Stanford）
> 硬件：MI325X（CDNA3）· MI350X / MI355X（CDNA4）；对照 NVIDIA B200（跑 TK / CUTLASS）
> 环境：ROCm 7.0 preview docker（`rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta`），500 次 warmup + 100 次测量
> **MLSys 正式版摘要里加了一句 arXiv 版没有的话：HipKittens 已产品化进 AITER。**

> **全文中译见 [`hipkittens-zh.md`](./hipkittens-zh.md)**（原文 CC0 1.0，作者已放弃著作权）。本篇是按论文章节顺序做的精读与延伸讨论，覆盖全部小节、表格数字与附录工程细节，并附对 MonolithEP / ROCmoe 的影响判断。

## TL;DR

**论文要回答的问题只有一个：ThunderKittens 那套 tile 原语是 NVIDIA 特有的，还是通用的？**

结论分两半，这个区分是全文的骨架：

- **前端抽象是通用的**——tile 数据结构 + PyTorch 风格的批量算子（`mma` / `exp` / `add`）原样搬到 AMD 就能用。
- **实例化这些抽象的算法必须重做**——调度模式、访存布局、cache 优化三样全部要换，因为硬件差异是根本性的。

**最该记住的一条：NVIDIA 那套 wave specialization（producer-consumer）在 AMD 上是负优化。** 原因不是实现问题，是架构决定的——**AMD 硬件把寄存器静态均分给同一 SIMD 上的所有 wave**，所以 producer wave 白占寄存器却不产出计算，直接压低了每个 thread block 能算的输出 tile 尺寸，进而压低算术强度。实测 BF16 GEMM（M=N=K=8192，MI355X）：

| 配置 | 输出 tile | TFLOPS |
|---|---|---|
| 4 producer / 8 consumer | 128×256 | 893 |
| 4 producer / 12 consumer | 192×256 | 1278 |
| **0 producer** / 8 consumer | 192×256 | 1281 |
| **0 producer / 8 consumer** | **256×256** | **1610** |

**producer 数量一增，性能就掉**；把 producer 砍到零、把输出 tile 顶到 256×256 才摸到峰值。作为对照，B200 上 TK 是 1538、CUTLASS profiler 选出的最优是 1570。

**替代方案是两个新调度**：**8-wave ping-pong**（每 SIMD 两个 wave，一个只发计算、一个只发访存，靠条件 barrier 周期性互换角色）和 **4-wave interleave**（每 SIMD 一个 wave，自己细粒度交错计算与访存指令）。前者简单且够用，后者更快但代码量爆炸——FP8 GEMM 热循环 48 行 vs 183 行、MHA backward 331 行 vs 989 行。

**对我们最直接的三条**：
1. **MonolithEP 的 WG 角色分区设计要重新评估**——这篇给出了 AMD 上「专职 producer」为什么亏的机制性解释和量化证据（§7.1）。
2. **AMD 的 MFMA 布局没有 NVIDIA 那种可组合的 16×16 core matrix 结构，导致布局爆炸，单一 swizzle 策略不可能**；而且 shared memory 的 phase/bank 行为**在 CDNA ISA 文档里根本没写**，作者是写了个 solver 逆出来的，结果列在附录 Table 5——这张表可以直接拿来用（§4.2）。
3. **chiplet 感知的 grid 调度值 +19%**，而默认 row-major 在「输出 tile 宽度与 XCD 数互质」时会踩到最坏情况——这是个我们大概率没做的免费优化（§5）。

---

## 1. Problem：CUDA 护城河的具体形态

论文开篇给的是硬件对比（Table 2 左半），这张表是整个工作的动机：

| 规格 | NVIDIA B200 SXM5 | AMD MI355X OAM |
|---|---|---|
| BF16 matrix/tensor | 2.2 PFLOPs | **2.5 PFLOPs** |
| MXFP8 | 4.5 PFLOPs | **5.0 PFLOPs** |
| **MXFP6** | 4.5 PFLOPs | **10.1 PFLOPs（2.2×）** |
| MXFP4 | 9.0 PFLOPs | **10.1 PFLOPs** |
| 显存容量 | 180 GB | **288 GB** |
| 显存带宽 | 8.0 TB/s | 8.0 TB/s |

**硬件全面不吃亏，个别项（FP6）领先一倍以上**，但软件生态把这些峰值锁死了。论文用了「hardware lottery / CUDA 护城河」的说法，并给出具体证据：

- **AITER 和 PyTorch 在 MI355X 上跑 Llama GQA backward，分别只有 SoTA 的 30% 和 24%**
- PyTorch SDPA 的 GQA backward 在 MI355X 上只有 **259 TFLOPS**（ROCm 7.0.0，2025-10）
- AITER 的 GQA backward 在序列长 8192 时，causal / non-causal 分别只有 **272 / 384 TFLOPS**
- Mojo 的 MHA kernel 因为 bank conflict，只有峰值的 **50%**
- TileLang 在 AMD 上只报了单个 attention kernel、MI300X 上 **257 TFLOPs**
- Triton 在 AMD 上寄存器生命期跟踪有问题、访存降不到最优 intrinsic；**截至 2025-09，buffer load 甚至还不是 Triton 在 AMD 上的默认加载方式**；编译出的 LayerNorm 比 HK 的 L2 命中率低 **23%**

**核心矛盾**：AMD 的峰值性能 kernel 是**极少数专家用裸汇编写的**（AITER、Composable Kernel），做法是在汇编里细粒度交错计算与访存指令的发射。这条路和 tile 化编程是正交的，且**无法规模化覆盖 AI 工作负载的广度**——GQA backward 就是活例子，AITER 根本没做好。

论文顺带提了 NVIDIA 侧的历史类比：H100 发布到开源峰值 attention kernel 出现，**中间隔了两年**。

## 2. 三条既有原语（要检验的对象）

论文把 TK 这类 DSL 的贡献归纳成三条，然后逐条检验它们在 AMD 上是否成立：

1. **Tiles**——基本数据类型是带优化访存模式的 tile，配一套 PyTorch 风格的批量算子（包装 PTX）。帮开发者显式管理内存层级各级的数据。
2. **Overlapping**——少量 kernel 模式帮开发者把 worker（AMD 的 wave / NVIDIA 的 warp）调度到不同硬件执行单元上。现代 NVIDIA kernel 已收敛到 **wave specialization（producer-consumer）**。
3. **Grid scheduling**——按合适顺序把工作分配给 thread block，最大化不可编程 cache 的复用。

**检验结果：第 1 条通用，第 2、3 条必须重做。**

## 3. 背景：AMD 硬件与术语

| 概念 | NVIDIA | AMD |
|---|---|---|
| 执行单元 | Warp（32 线程） | **Wave（64 线程）** |
| 线程块 | Thread Block / CTA | Workgroup |
| 处理器 | SM | **CU（compute unit）** |
| 共享内存 | SMEM | **LDS** |
| 寄存器 | Registers | **AGPR / VGPR** |
| Cache | L2（全 GPU） | **L2（chiplet 级）+ LLC（全 GPU）** |
| 矩阵计算 | Tensor core | **Matrix core** |
| 矩阵指令 | WGMMA / WMMA / TCGEN05 | **MFMA** |
| 异步访存 | TMA | **buffer_load to LDS** |

关键硬件参数：

- **MI355X：256 CU，组织成 8 个 XCD（accelerator complex die），每 XCD 32 个 CU**（CDNA3 是 38 个）
- 每 CU 4 个 SIMD；**每 SIMD 512 个 32-bit 向量寄存器**（每 CU 共 512KB）
- **每 XCD 共享一个 4MB 的 L2；LLC 位于 L2 与 HBM 之间**
- **L2 miss 最坏惩罚 300ns，LLC miss 500ns**
- 单 wave/SIMD 时，硬件把 512 个寄存器**劈成 256 VGPR + 256 AGPR**

## 4. 方法一：可编程内存的访存优化

### 4.1 开发者控制的寄存器调度（register pinning）

**问题很具体**：硬件支持 AGPR 作为 matrix core 指令的输入，**但 HIPCC 不支持**。于是对同时含矩阵和向量操作的负载（attention backward 是典型），HIPCC 编出来的 kernel 必须插入冗余的 `v_accvgpr_read` 把数据从 AGPR 搬到 VGPR 才能发 MFMA。

**解法**：让开发者绕过编译器，把每个 tile 归属的寄存器**显式钉死**。接口设计得很克制——pinned register tile 的用法和编译器托管的完全一致，两种都保留，开发者自选控制粒度。

```cpp
using Q_ranges = split_many_t<type_list<range<24,39>>, 4>;  // v[24:27] v[28:31] v[32:35] v[36:39]
rt<bf16, 16, 128, row_l, rt_16x32_s, Q_ranges> Q_i;          // 每个 range 装一个 base tile
```

**效果**（MHA non-causal backward，batch 16 / heads 16 / head dim 128）：

| 方法 | seq 4096 | seq 8192 |
|---|---|---|
| HK（编译器托管） | 855 | 909 |
| **HK + pinned registers** | **1024** | **1091** |
| AMD 汇编（AITER） | 1018 | 1169 |

**+20% 左右，且正是这一步让 HK 的 backward attention 追平 AITER 的裸汇编。**

### 4.2 异构 matrix core 形状的 tile 布局

这一节是全文工程含量最高的部分。

**NVIDIA 的结构性优势**：所有矩阵指令形状都由同一个 **16×16 core matrix** 反复拼出来，所以 TK 和 Linear Layouts 能用**一套 swizzle 策略通吃所有形状**。

**AMD 没有这个性质**：每条 MFMA 指令用**完全不同的布局**，彼此之间没有共同的底层结构 → **布局数量爆炸**。

更麻烦的是第二层差异：**wave 访问 shared memory 是分 phase 进行的**（一个 wave 里的一部分线程并发访问），而

- NVIDIA 的 phase 是**顺序**分配线程的（线程 0-7 第一 phase，8-15 第二 phase……）
- **AMD 的 phase 是非顺序的，而且随指令不同而不同**

**而这些 phase 在 CDNA ISA 文档里根本没有记载**——作者写了个 solver 把它们逆出来（做法：遍历 wave 内每对线程，让它们访问同一个 bank，若发生冲突则判定二者同 phase），结果记在 Table 5：

| 指令 | 可用 bank 数 | phase 数 | 各 phase 的活跃线程 |
|---|---|---|---|
| `ds_read_b128` | 64 | 4 | 0-3,12-15,20-27 / 4-11,16-19,28-31 / 32-35,44-47,52-59 / 36-43,48-51,60-63 |
| `ds_read_b96` | 32 | 8 | 0-3,20-23 / 4-7,16-19 / 8-11,28-31 / 12-15,24-27 / … |
| `ds_write_b64` | 32 | 4 | 0-15 / 16-31 / 32-47 / 48-63 |
| `ds_read_b64` | 64 | 2 | 0-31 / 32-63 |

> **这张表可以直接拿来用。** 它是公开材料里少见的 CDNA4 LDS phase 行为文档，写任何手工 HIP kernel 做 bank conflict 分析都用得上。

**「单一 swizzle 不可能」的证明**（附录 D.1 给的反例，值得记住）：

- 行布局 16×16 BF16 tile **写入** LDS：每线程持 4 个连续 BF16 = 64 bit，最优指令是 `ds_write_b64`，无冲突的 swizzle 是 `offset ^= ((offset % 512) >> 7) << 3`，**它把内存打散成 64-bit 块**
- 行布局 16×32 BF16 tile **读出** LDS：每线程持 8 个连续 BF16 = 128 bit，最优指令是 `ds_read_b128`，**它要求至少 128 bit 连续**

两者的粒度直接冲突——`ds_write_b64` 的 swizzle 破坏了 `ds_read_b128` 需要的连续性。**所以必须用不同的 swizzle 模式。**

**HK 的工程取舍**：不为每种布局都实现独立 swizzle（代码复杂度不可控），而是**识别出常见的共现布局组合，为这些组合提供无 bank conflict 的 swizzle**。论文给的例子是一个同时让 16×32 行布局（`ds_read_b128`）和列布局（`ds_read_b64_tr_b16`）都无冲突的 swizzle：从第 8 行开始，把前 8 列与后 8 列互换。

**三级内存的处理**：

1. **Register**：默认用**最小的 MFMA 指令形状**，因为这给调度控制留的余地最大；边缘情况允许开发者按 MFMA 形状参数化
2. **Shared**：如上，按共现组合提供 swizzle；支持 shared↔register 之间形状互为倍数的加载（16×32 SMEM → 32×16 寄存器**不支持**，16×16 → 32×16 **支持**）
3. **Global**：AMD 支持 HBM→LDS 的直接异步加载（类比 TMA），绕过寄存器文件。**但有个反直觉的点：TK 在 NVIDIA 上是直接 swizzle shared memory 地址，AMD 上则要 swizzle HBM 地址**

## 5. 方法二：计算与访存的重叠调度

### 5.1 为什么 wave specialization 在 AMD 上不行

NVIDIA 能做 wave specialization，靠的是一整套配套硬件特性，**AMD 一样都没有**：

| NVIDIA 依赖的特性 | AMD 状况 |
|---|---|
| 专用访存硬件 `tma` | 无对应物（有 buffer_load 但语义不同） |
| 异步矩阵乘、操作数直取 shared/tensor memory（`wgmma`/`tcgen05`） | 无 |
| 大 SRAM 撑起的深流水（**B200 每处理器 SRAM 比 MI355X 大 40%**） | 更小 |
| 寄存器再分配（TMA 省寄存器 → producer 把寄存器让给 consumer） | **寄存器静态均分，无法转让** |
| 硬件同步原语 `mbarrier` | 无（用 shared memory atomics 替代） |

**最致命的是寄存器静态分配**。论文的实验设计很干净：固定其他变量，扫 producer/consumer 比例和输出 tile 尺寸，结论是两条——**要把每 thread block 的输出 tile 顶到最大以提高算术强度，同时要把流水做深以掩盖访存延迟**。而 wave specialization 与第一条直接冲突。

**两个补充结论也值得记**：

- **AMD 用 shared memory atomics 替代 mbarrier，开销可忽略**——192×256 的 producer-consumer kernel（用 atomics）和无 wave specialization 的 kernel 性能相当，说明**输出 tile 形状才是主导因素**，同步机制不是
- **代偿关系**：NVIDIA 靠大 SRAM 撑深流水 + 大矩阵形状（256×256×16），AMD 则靠**小矩阵形状（16×16×32）实现更细粒度的加载/计算分级**来做深流水；NVIDIA 靠操作数直取 shared memory 缓解寄存器压力，AMD 则靠**2× 大的寄存器文件**代偿

### 5.2 两个可行的调度模式

**8-wave ping-pong（适合计算/访存均衡的负载）**

每 thread block 八个 wave，每 SIMD 常驻两个。八个 wave 分成两组四个（每组每 SIMD 一个）。**同一 SIMD 上的两个 wave 分工互补：一个只发计算指令，另一个只发访存指令，然后靠条件 barrier 周期性互换角色**，来回翻转。计算 wave 跑 MFMA 时，配对的访存 wave 预取下一批数据。

**4-wave interleave（适合不均衡的负载）**

每个 SIMD 恰好一个 wave，**每个 wave 自己同时发计算和访存指令，靠精心错开的顺序最大化各硬件单元的占用率**。计算重或访存重的场景下，它能更好地同时打满 MFMA 和 LDS 流水；wave 可以动态调整指令配比。

**取舍是明确的**（Table 3）：

| Kernel | 模式 | 热循环行数 | TFLOPS |
|---|---|---|---|
| FP8 GEMM | 8-wave | **48** | 3222 |
| FP8 GEMM | 4-wave | 183 | **3327** |
| MHA backward | 8-wave | **331** | 894 |
| MHA backward | 4-wave | 989 | **1091** |

8-wave 能用大 tile 原语（和 wave specialization 类似的粒度），4-wave 必须用小 base tile 编程，**代码量涨 3–4 倍**。

**论文自己强调的意外发现：8-wave 这个简单模式已经足以在 BF16 GEMM、FP8 GEMM、attention forward 上追平或超过 AMD 的裸汇编 kernel。** GQA non-causal backward 上 8-wave 超基线 1.8×，4-wave 更是 2.3×。

## 6. 方法三：不可编程 cache 的 chiplet 感知调度

**动机**：chiplet 正在成为 GPU 扩展的主流路径（Blackwell 2 chip，MI355X 8 chip），但现有框架完全忽略其层次化 cache 结构。

**硬件事实**：硬件调度器把 thread block 以 **round-robin 方式**分配给 XCD。带宽模型：

```
Bandwidth = LLC 带宽 × LLC 命中率 + L2 带宽 × L2 命中率     （L2 带宽约为 LLC 的 3×）
```

**两条互相冲突的优化原则**：

1. **L2 复用**：映射到同一 XCD（共享 L2）的 thread block 应当覆盖输出矩阵的一个**矩形区域**（"L2 tile"），这样连续的 block 复用相同的 A 行和 B 列
2. **LLC 复用**：但纯粹为 L2 优化会导致各 XCD 抓取 A、B 的**互不相交**部分，在下一级 cache 造成冗余加载。理想情况下所有 XCD 的合并访问足迹（"LLC tile"）应在 A 和 B 上都有重叠

**实测数据**（Table 4）把这个张力展示得很清楚：

| 形状 | 调度 | L2 命中 | LLC 命中 | 带宽 | TFLOPS |
|---|---|---|---|---|---|
| 9216³ | row-major | 55% | 95% | 15.1 TB/s | 1113 |
| 9216³ | XCD(W7/C216) | **79%** | 24% | 14.9 TB/s | 991 ↓ |
| 9216³ | **XCD(W5/C25)** | 75% | 93% | **18.3 TB/s** | **1145** |
| 14592³ | row-major | 36% | 76% | 10.7 TB/s | 900 |
| 14592³ | XCD(W8/C542) | 79% | **7%** | 13.9 TB/s | 980 |
| 14592³ | **XCD(W8/C64)** | 78% | 55% | **16.6 TB/s** | **1068（+19%）** |

**注意中间那两行**：把 L2 命中率从 55% 拉到 79%，性能反而从 1113 掉到 991——**因为 LLC 命中率从 95% 崩到 24%**。只优化一级 cache 是会倒退的。

**HK 的算法**（Algorithm 1）两步走，两个可调参数 `W`（window 高度）和 `C`（chunk 大小）：

1. **XCD 分组**：把 2D grid 拍平成线性序列，重映射 block ID 使得**连续 C 个 ID 落在同一 XCD**，减少跨 chiplet 流量
2. **层次化窗口遍历**：不按行遍历，而是按**高度为 W 的竖直窗口**处理，等效于把 block ID 空间「折叠」成矩形 tile，优化 L2 复用

调参经验：因为 L2 带宽约为 LLC 的 3×，**W 应优先最大化 L2 命中率**；MI355X 每 XCD 32 CU，实测 **L2 tile 取 8×4 或 4×8 最好**；再调 C 改善 LLC。

> **最坏情况的判据值得单独记**：当**输出矩阵的 tile 宽度与 XCD 数量互质**时（例如 MI355X 上 57 个 tile 配 8 个 XCD），默认调度会踩到最差复用模式。这是个可以直接写进 kernel 选参逻辑的规则。

## 7. 实验

**基线**：PyTorch（compiled 与 SDPA）、AITER、Composable Kernel、ROCm Triton、hipBLASLt。评测 MI325X（CDNA3）与 MI355X（CDNA4）。

| 负载 | 结果 |
|---|---|
| **BF16 / FP8 GEMM** | 追平 AITER / hipBLASLt / PyTorch（均为汇编实现）；**超 Triton 1.3–3.0×**；且**单一 8-wave 调度通吃所有评测形状** |
| **Attention forward**（MHA/GQA，causal/非 causal，head dim 64/128） | 平均超所有 AMD 基线（含裸汇编 AITER）：**1.0–2.1× AITER**、1.3–4.5× PyTorch SDPA、1.0–1.4× CK、1.2–4.5× Triton。论文称在可比设置下与 FlashAttention-3 有竞争力 |
| **Attention backward** | GQA causal/非 causal **超基线 1.8–2.5×**；MHA 与最强汇编基线持平 |
| **访存受限**（fused dropout-residual-layernorm、rotary） | **超 AITER 与 torch.compile 1.1–2.2×** |

**attention backward 的实现细节值得注意**：这是出了名的寄存器重负载，HK 的 kernel **同时用了多种 MFMA 形状（16×16×32 和 32×32×16）、对同一个 shared tile 同时做行布局和列布局的读取、外加显式寄存器 pinning**。混用 tensor core 形状这一点，论文明确说是 Linear Layouts 等工作没有演示过的。

**正确性验证**：用这些 kernel 在 SlimPajama 上预训练 **Llama 1B 和 BERT 110M，10B token 后困惑度与 PyTorch/AITER 训练的模型一致**。

## 8. 附录里的两个硬核工程细节

### 8.1 MI325X（CDNA3）的降级路径

**MI325X 只有 65KB shared memory，做不了 SMEM 双缓冲**。HK 的应对：保持同样的 8-wave 结构，但**改用寄存器文件做双缓冲**——不走 HBM→LDS 直接加载，而是 HBM→寄存器缓冲，wave 在此期间对上一批已存入的寄存器 tile 做 MFMA，算完再用 `ds_write` 把寄存器缓冲写下到 shared memory。

### 8.2 FP6 案例研究（附录 F，标注为初步结果）

**FP6 是 AMD 最被低估的硬件优势——matrix core 峰值是 NVIDIA 的 2.2×（10.1 vs 4.5 PFLOPs）。但访存把它锁死了。**

FP6 的问题在于 6 bit 不对齐字节边界。128×128 的 tile 是 12,288 字节、每线程 48 字节，三种加载指令全有问题：

- **`buffer_load_dwordx4`**：每 tile 每线程只需 3 条指令，最省。但 `ds_read_b128` 要求 16 字节对齐，而每行第二个线程的读取偏移是 24 字节 → kernel 变成 shared memory 受限。补救是让奇数 lane 交换两条 LDS 读指令的顺序，但**目标寄存器依赖 thread ID 会触发慢速 scratch 存储**，于是只能事后按 thread ID 条件交换寄存器区域——这会「打断 wave」，产生两条跳转指令。**实测跳转 + VALU 占了热循环 49% 的周期，kernel 只有 2430 TFLOPS。** 换成两条 `ds_read_b96` 则与全局加载的 16 字节块错位，无法 swizzle，导致 **4 路 bank conflict**
- **`buffer_load_dwordx3`**：能让一个 wave 恰好加载 8×128 的 tile，且支持 swizzle。缺点是指令数并不比 FP8 少（每线程 4 条，和 FP8 GEMM 一样），且 12 字节数据按 16 字节跨步加载，**浪费 25% 的 shared memory tile，32 个 bank 里有 8 个用不上**。尽管如此，**这是三者中最可行的**
- **`buffer_load_dwordx1`**：无浪费无对齐问题，但被指令发射数拖垮

**还有一个 HIPCC 的坑**：16384³ 规模下 HIPCC **把 54 个寄存器 spill 到 scratch，产出又慢又错的 kernel**。靠显式寄存器调度才修掉。而且要手工保证 `v_mov_b32_e32` 与依赖它的 MFMA 之间至少隔 8 个周期的指令，**有时得手工插 `v_nop`**。

**结果**：HK 的 FP6 GEMM 超过 AMD CK 实现，性能与 HK 自己的 FP8 GEMM 相当（**也就是说 2× 的 FP6 峰值优势目前一点没吃到**）。

### 8.3 编译器提示的用法

HK 在 HIP 层调度之外，还用 LLVM 的三类 intrinsic 辅助：

- `__builtin_amdgcn_sched_barrier(mask)` — 在编译后的调度里建立指令类别的硬边界
- `__builtin_amdgcn_sched_group_barrier(mask, size, syncid)` — 建立调度流水。掩码：`MFMA=0x08`、`VMEM=0x20`、`DS=0x100`。语义是「向后找最近的、尚未被前一个 group barrier 收编的对应类型指令」
- `__builtin_amdgcn_s_setprio(0-3)` — 设定 wave 相对于竞争硬件资源的其他 wave 的优先级。**用在 8-wave ping-pong 的计算簇周围**

**局限**：`asm volatile` 包住的代码对编译器是黑盒；部分指令（如 `v_cvt_pk_bf16_f32`）缺 LLVM builtin。

论文顺带批评了 Modular 的路线：Modular 的 GEMM 也靠 `sched_group_barrier`，但**那要求开发者逐条指令地思考，而非提供「用 tile 原语思考」的选项**。HK 的主张是**在 cluster 尺度用调度提示、在顶层用 tile 原语组织 kernel 调度**。

## 9. 批判

**论文自己诚实的地方**（值得肯定）：FP6 明确标注为初步结果且 CK 基线未优化；对「超过 AITER」的解释也点明了部分原因是 AITER 覆盖不全而非同台竞技更快。

**需要打折扣的地方**：

1. **摘要说 1.2–2.4×，引言说 1.2–10×，差距来自选择性。** 那些大倍数几乎全部来自**基线基本缺失**的场景（GQA backward、head dim 64、访存受限 kernel）。在 AMD 真正投入优化过的路径（BF16/FP8 GEMM、MHA forward）上，HK 是**追平**汇编而非超越。这个区分论文写清楚了，但摘要的措辞会让人记住大数。

2. **Table 2 拿 MI355X 上的 HK 和 B200 上的 TK/CUTLASS 并列**，用来论证「AMD 能追平」。跨厂商比绝对 TFLOPS 在方法论上是不干净的（峰值本来就不同：BF16 2.5 vs 2.2 PFLOPs），当方向性参考可以，当结论不行。

3. **完全没有 MoE kernel，也没有任何多卡/分布式内容。** 全部是单卡 kernel。dispatch/combine、all-to-all、EP 这些我们最关心的东西一个没碰。

4. **register pinning 是把双刃剑**：绕过编译器意味着寄存器预算的正确性由开发者背，换硬件代次、换寄存器数量配置都可能要重调。论文把它做成可选项是对的，但没讨论维护成本。

5. **chiplet swizzle 的 `W`/`C` 需要按形状调参，论文没给自动调参器**，只给了经验值（8×4 或 4×8）。实际用起来这是个 autotuning 问题。

6. **4-wave interleave 只给了模式描述，没给可复用的抽象。** 代码量从 331 行涨到 989 行这个数字本身说明，这条路目前仍然更接近「有 tile 原语辅助的手写汇编」。

## 10. 对我们的意义

### 10.1 直接冲击 MonolithEP 的 WG 角色分区

这是本篇对我们最重要的一条。**MonolithEP 的设计是 WG 角色分区——一部分 WG 做通信/搬运，一部分做计算。这在结构上就是 wave specialization。** 而这篇论文给出了 AMD 上这个模式为什么亏的**机制性解释加量化证据**：

> AMD hardware statically divides registers across all waves, meaning **producers consume registers without contributing to output computation**. This limits the usable output tile size when using wave specialization.

0 producer + 256×256 输出 = 1610 TFLOPS，而 4 producer + 128×256 = 893 TFLOPS。**接近 2× 的差距，且方向是单调的。**

**但要注意适用边界，不能直接照搬结论**：HipKittens 测的是 GEMM，producer 干的是纯访存搬运；MonolithEP 的通信 WG 干的是**跨卡通信**（IPC scatter/gather、device-scope atomic flag 轮询），这类工作**无法被计算 wave 顺带做掉**——它不像预取那样可以塞进计算 wave 的空隙。所以「砍掉 producer」在 MoE megakernel 里未必可行。

**真正该做的动作是量化**：测出 MonolithEP 里通信 WG 占用的寄存器预算，代入「这些寄存器如果还给计算 WG 能把输出 tile 撑多大」，算出这笔交易到底亏多少。如果亏得像 GEMM 这么多，那 8-wave ping-pong 式的**角色轮换**（而非固定分区）就值得试——让同一个 wave 交替干通信和计算，而不是常驻分工。

### 10.2 8-wave ping-pong 是可以直接抄的默认调度

论文的实证结论很明确：**8-wave ping-pong 这个简单模式已经足以在 BF16 GEMM、FP8 GEMM、attention forward 上追平 AMD 裸汇编**，而且热循环只有几十行。对 ROCmoe 来说这是个现成的起点，比从零设计调度靠谱。

关键实现要素就三个：每 SIMD 两个 wave、条件 barrier 控制角色互换、计算簇周围用 `s_setprio` 调优先级。

### 10.3 Table 5 的 phase/bank 表可以直接用

CDNA ISA 没有文档化 LDS 指令的 phase 行为，作者逆出来并公开了。**任何手写 HIP kernel 做 bank conflict 分析都需要这张表**，我们之前的 kernel 优化基本是靠试。建议把它抄进 `knowledge/kernels/`，并把作者那个 solver 思路（遍历线程对、令其访问同 bank、观察是否冲突）实现一份，将来换硬件代次可以自己重测。

同样值得记的是那条「单一 swizzle 不可能」的反例——`ds_write_b64` 的 64-bit 打散与 `ds_read_b128` 的 128-bit 连续性要求直接冲突。这解释了为什么我们之前在混布局场景下调 swizzle 老是按下葫芦浮起瓢。

### 10.4 chiplet grid swizzle 大概率是我们的免费午餐

+19% 的量级，实现成本只是一个 block ID 重映射函数。**特别要检查的是「输出 tile 宽度与 XCD 数互质」这个最坏情况判据**——MI355X 是 8 个 XCD，任何输出宽度为奇数 tile 的 GEMM 都会踩。我们的 GEMM 和 MoE grouped GEMM 有没有做 XCD 感知的 block 重排，值得立刻查一遍。

注意那个陷阱：**只优化 L2 会让性能倒退**（55%→79% L2 命中，性能 1113→991）。必须两级联合调。

### 10.5 对 FlyDSL 路线选型的输入

现在有了三个可对照的点，构成完整的谱系：

| 方案 | 抽象层次 | 性能 | 代价 |
|---|---|---|---|
| [avelang](../knowledge/libraries/avelang.md) | 零抽象，`s_waitcnt` 是一等语法 | 理论上等于手写 | 只有开发体验差异，**代码质量上限等同手写 HIP** |
| **HipKittens** | **C++ 嵌入式 tile 原语 + 选择性绕过编译器** | **追平/超过裸汇编** | 需要理解 phase/bank/布局，但热循环仅几十行 |
| Wave（MLSys '26） | 符号化 Python DSL + MLIR | 未知（待读） | 编译器路线的全部风险 |

**HipKittens 证明了中间路线是可行的**：不需要写裸汇编，但确实需要在**寄存器分配**这一个点上绕过 HIPCC。这个结论对 FlyDSL 很关键——它说明「DSL 必须能开后门」，纯粹的高层抽象（Triton 那种）在 AMD 上拿不到峰值，而后门只需要开在少数几个位置。

### 10.6 FP6 是个被忽略的机会窗口，但门槛很高

**AMD 的 FP6 峰值是 NVIDIA 的 2.2×，而且目前谁都没吃到**（HK 自己的 FP6 GEMM 只做到和 FP8 相当）。障碍是 6-bit 不对齐字节边界导致的加载困局——最优指令 `buffer_load_dwordx4` 会因为 shuffle 吃掉 49% 的热循环周期。

这是个**高风险高回报**的方向：如果能解决 FP6 的加载问题，拿到的是硬件层面 2× 的优势，且是 NVIDIA 结构上给不了的。但从论文描述看，这个问题很可能需要硬件或 ISA 层面的配合（比如一条对齐友好的 FP6 加载指令），纯软件绕不过去。**建议先当作观察项，不要立刻投入。**

### 10.7 一条战略信号

**HipKittens 已经产品化进 AITER**（只在 MLSys 正式版摘要里写了，arXiv 版没有）。这意味着 AMD 官方库正在从「专家手写汇编」转向「tile 原语 + 选择性汇编」。我们如果要在 AMD 上做 kernel 工作，**跟着这个方向走比对抗它更省力**——也意味着 AITER 的基线性能在未来会快速上升，我们做对比实验时要注意版本。

## 11. 未决问题

1. **MonolithEP 的通信 WG 到底占多少寄存器预算？** 这是把 §10.1 的定性警告变成决策的唯一途径，也是个低成本实验。
2. **角色轮换（8-wave ping-pong 式）能不能用在跨卡通信上？** 通信有延迟不可控的特性，条件 barrier 的周期性互换未必成立。
3. **HK 的 tile 原语能不能承载 MoE 的 ragged 形状？** 论文全部是规则形状，MoE 的可变 token 数是另一回事。
4. **Wave（MLSys '26）与 HipKittens 的定位差异**——同届会议两篇 AMD 相关的 kernel 编程工作，值得对读。
5. 论文没有多卡内容，所以 [Perseus](./perseus.md) 记录的跨节点问题与本文完全正交，两者可以叠加。
