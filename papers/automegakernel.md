# AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis

> [arXiv 2606.09682](https://arxiv.org/abs/2606.09682) (v1, 2026-06-08, cs.LG) · Jaber Jaber, Osama Jaber（RightNow AI，自筹经费） · CC BY 4.0
> 代码：<https://github.com/RightNow-AI/AutoMegaKernel>（开源，含 agent harness 与数据）
> 硬件：RTX 5090 Laptop (sm_120) · A100-40GB (sm_80) · H100-80GB (sm_90) · L4 / L40S / A10G / T4
> 范围：**HF Llama 系单流 batch-1 decode**；MoE 明确列在 Layer 3 roadmap，不在本文范围内

## TL;DR

**这篇论文的贡献不是速度，是「一个 agent 提出的调度方案，怎么在启动前被静态证明不会死锁、不会竞争」。** 作者自己在摘要第一段就写了 "The contribution is the system, not raw speed."

它把 HF Llama 模型编译成一个常驻 cooperative kernel（一次 launch = 一次前向 = 一个 token），中间加了一层 **frozen 的 schedule-IR 验证器**，用 9 条静态图检查证明无死锁、无竞争。在 **7,160 个对抗构造的调度**（其中 6,091 个被独立 oracle 判定不安全）上，验证器**零假接受**，同时接受了全部 360 个真实 lowering。

**对 ROCmoe 最有价值的两条：**

1. **P2 缺的那层就是这个。** MonolithEP 现在改一次 WG 角色划分或依赖关系，验证方式是「跑一下看会不会挂」。这篇给出了一套可直接移植的 9 条检查（§2.2），成本极低（CPU 上 ~5,150 个调度/秒）。其中「shared-counter all-join」那条尤其重要——**它抓的是一类跑测试根本发现不了的 bug**（作者的动态 oracle 在 350 个该类变异体上一个都没抓到，静态验证器全抓到了）。

2. **它的负面结果比正面结果更值钱。** 作者 A/B 测掉了两个显而易见的优化方向（cp.async 权重暂存环、split-KV attention）后都退步，从而定位出真正的结构性瓶颈是**每 tile 一次的跨 SM 同步**——而且**在带宽最高的训练级芯片上最严重**（A100 0.79→0.55×、H100 0.72→0.60×，而低带宽的推理级 L4 反而赢到 1.33×）。**MI355X 是训练级芯片**，按这个 regime 论证，MonolithEP 如果有每 tile 的 grid 级同步，在 MI355X 上只会更糟。

## 1. Problem

batch-1 解码时算术强度接近 1，下界是 `weight_bytes / HBM_bandwidth`。常规 PyTorch/cuBLAS 执行每算子一个 kernel，每层付几十次 CPU launch 延迟，每个算子边界都要把激活往 HBM 上倒一圈。CUDA Graph 摊薄了 launch 开销，但**算子间的 kernel 边界和它们的 HBM 往返还在**。

megakernel 去掉这些边界。作者指出现有工作留下的缺口是**信任与可移植性**：

- **MPK**（Mirage Persistent Kernel）自动把张量程序转成单个常驻 kernel，但**没有静态的死锁/竞争闸门**；
- Spector 等人手写的 Llama-1B megakernel 只针对一个模型一个架构；
- 两者都不是**可以让编码 agent 安全驱动的编辑面**。

最后一条是本文真正的动机：如果要让一个自动搜索循环去改调度，那**「改错了会挂住 GPU」是不可接受的失败模式**，必须变成「改错了在验证阶段被干净地 REJECTED」。

## 2. Method

### 2.1 四层架构与两个循环

| 层 | 内容 | 谁写 |
|---|---|---|
| **Layer 3** | 动态性（连续批处理、动态 shape、**MoE**） | roadmap 占位 |
| **Layer 2** | scheduler：HF 模型 → 图 IR → task-DAG + `validate()` 安全闸门（`ir.py` 1150 行 + `graph.py` 548 行） | **agent 搜索，验证器把关** |
| **Layer 1** | instructions：符合 ABI 的微内核（GEMV、attention、RMSNorm、RoPE…），各自对参考算子隔离校验 | **agent 搜索，隔离测试把关** |
| **Layer 0** | VM：常驻 cooperative megakernel，每 SM 一个 block，per-SM 调度器、计数器、页（`scheduler.cu` **仅 132 行** + `loader.py` 857 行） | **手写、逐架构冻结的可信基座** |

- **Loop 1** 调一个微内核（隔离测试把关）
- **Loop 2** 调 `ScheduleConfig`（frozen VM 确定性 lowering，验证器把关）

> 「正确性是架构的属性，不是产物的属性」——生成被限制在一个已验证的结构内部。这个思路对我们很重要：**不是去验证 agent 产出的每份代码，而是把生成限制在一个能被静态检查的表示上。**

### 2.2 同步模型与 9 条静态检查（可直接移植的部分）

**同步模型被刻意收窄到只剩一种形式：**

- 跨任务信号**只能**通过单调递增的 `uint32` 计数器；
- 任务完成时先发一次 device-scope release fence（给自己所有输出缓冲的 store 定序），然后**恰好把一个 `out_counter` 加 1**，语义是「我的输出全写完且可见」；
- 任务执行前在一组 `(counter, threshold)` 上自旋等待，**阈值必须是静态已知的**，用 acquire-load（编译器不得外提），带指数退避和 `abort_flag` 轮询（WDDM 看门狗逃生）；
- **没有锁，没有任意信号**。

正因为收窄成这样，安全性才归约成一小组静态图检查：

| 检查 | 保证 | 拒绝条件 |
|---|---|---|
| 引用完整性 / arity / 参数 | 良构 | 缺 buffer/counter、arity 错、缺参数 |
| ABI 容量上限 | 良构 | inputs>8、outputs>4、waits>8、rank>4 |
| **等待可满足性** | 无死锁 | 阈值 <1、无生产者、或 `t > #producers` |
| **DAG 无环** | 无死锁 | 生产者→消费者图有环（Kahn + 迭代 DFS 取环见证，5000+ 节点安全） |
| **per-SM 队列序** | 无死锁 | 存在边 `a→b`、同一 SM、队列里 `b` 排在 `a` 前面 |
| **共享计数器全联接** | 无竞争 | 多生产者计数器上出现 `1 < t < #producers` |
| **传递 happens-before** | 无竞争 | 某次读的写者不是其有序前驱 |
| KV_CACHE 定序 | 无竞争 | 本轮 append 的读者没有定序边 |
| 输出可达性 | 正确性 | 有 IO_OUTPUT buffer 没有任务生产 |

**「共享计数器全联接」是全文最精妙的一条**，值得单独解释：

> 计数器携带的是**一个计数**，不是**哪个生产者完成了**。所以一个有多个生产者的计数器就是一次真正的 join，任何在它上面的等待都必须用 `threshold = #producers`。`1 < t < #producers` 的部分等待就是一个 **first-k-of-N 竞争**——你知道有 k 个完成了，但不知道是哪 k 个。

这类 bug 在运行时几乎抓不到：作者那个基于计数器的动态 oracle 在 350 个该类变异体上**判定为 0 个不安全**（结构上就观察不到），而静态验证器**全部 350 个都拒绝了**。作者因此说「静态证明比运行时采样更严格也更正确」。

**传递 happens-before 检查**的实现也很朴素有效：沿拓扑序走，为每个任务维护「传递前驱写过的 buffer 位掩码」，任何读操作只要它的写者不在有序前驱里就拒绝。

### 2.3 ABI 与编辑面

每个 Task 一一对应一个定长 POD `amk_instruction_t`：op、≤8 输入 / 4 输出 / 8 等待、一个 `out_counter`、SM 索引、一个带类型的标量参数块。Buffer 携带 `{ptr, numel, rank, dtype, space, shape[4], stride[4]}`。

**关键纪律：指令是纯计算——不得触碰计数器或任何未声明的 buffer，不得发起工作；同步全归 VM 所有。** 枚举值和容量常量在 `ir.py` 与 `abi.h` 两边都是权威定义，`tests/test_abi_sync.py` 在任何漂移时让构建失败。

Layer-2 的 agent 提出的是一个**结构化对象**而不是 kernel 代码：

```
ScheduleConfig {
  tiling,               # 每算子 tile 尺寸
  fusion_grouping,
  sm_assignment,        # round_robin / load_balance / explicit
  pipelining_depth,     # 权重预取前瞻
  page_allocation,      # linear / graph_color / none
  threads_per_block,
  smem_bytes_per_block,
}
```

Frozen VM 把任意一点确定性地 lower 成 `MegakernelProgram`，`validate()` 保证**无论选哪一点结果都是安全的**。

> **「一块新 GPU 是一条新的 `GpuTarget` 数据记录，永远不是一次 scheduler 修改。」** 这句话是很好的设计约束，值得写进 FlyDSL 的设计原则。

## 3. Experiments

### 3.1 验证器可靠性（本文的核心结果）

7,160 个调度 = 360 个真实 lowering + 2,800 个单点注入变异体（8 个不安全类，每类 350 个）+ 4,000 个随机 DAG，由一个**不调用 `validate()` 的独立结构+动态 oracle** 打标。

| 不安全类 | oracle 判定不安全 | 验证器拒绝 | 假接受 |
|---|---|---|---|
| cycle | 342 | 342 | 0 |
| drop_wait（竞争） | 331 | 331 | 0 |
| kv_before_append | 171 | 171 | 0 |
| self_wait | 197 | 197 | 0 |
| oob_counter | 350 | 350 | 0 |
| oob_buffer | 350 | 350 | 0 |
| capacity_overflow | 350 | 350 | 0 |
| partial_shared | **0**（oracle 观察不到） | **350**（仍全拒） | 0 |
| **全体** | **6,091** | **6,091** | **0（0.0000%）** |

同时接受全部 **360/360** 真实 lowering，其中 24 个重新 lower 后在 ReferenceVM 上跑，与 eager PyTorch **逐位相同**。CPU 上验证速度约 **5,150 个调度/秒**。

作者的自我限定很克制：这是**「在一个可信基座之上的经验性可靠性，不是形式化验证」**——可信基座指验证器本身和逐架构的 VM。

### 3.2 生成覆盖与自重定向

10/10 支持的模型自动生成正确 megakernel，**零手写 CUDA**：

| 模型 | 来源 | 层数 | IR 任务数 | logit 误差 | == HF greedy |
|---|---|---|---|---|---|
| SmolLM2-135M | 真实 ckpt | 30 | 1,716 | 3.9e-5 | yes |
| SmolLM2-360M | 真实 ckpt | 32 | 2,690 | 2.5e-5 | yes |
| TinyLlama-1.1B-Chat | 真实 ckpt | 22 | **3,410** | 1.3e-5 | yes |
| Llama h2048 L8 | from-config | 8 | 1,634 | 8.6e-6 | n/a |

IR 任务数随深度结构性增长（2 层 182 → 8 层 1,634）。4 个故意不兼容的变体里 3 个在导入时被明确拒绝（带 bias 的投影、linear RoPE scaling、GELU 激活）；**第 4 个（Qwen2 硬编码 q/k/v bias）被静默接受**，logit 误差 2.47——作者把这个 config-inspection 盲点如实报告了，并说一次 state-dict bias 扫描就能补上。

**自重定向**：同一份源码在 sm_80 / sm_90 / sm_120 上构建并运行正确的 megakernel，gencode 从活动设备自动推导。fp32 下最大绝对误差 ≤4.2e-7（toy/合成），真实 SmolLM2-135M 上 3.8e-5。

真实 checkpoint 上：teacher-forced 困惑度 **14.948473 vs 14.948473**（差 2.45e-7），64-token greedy 解码与 `model.generate` **逐字节相同**。

### 3.3 性能：赢在哪、输在哪

搜索找到的 int8 weight-only（W8A16）megakernel 对 CUDA-graph 化的 cuBLAS bf16：

| GPU | 类别 | 带宽 | 结果 |
|---|---|---|---|
| L4 (sm_89) | 推理级 | 300 GB/s | **1.18→1.33×**（1.3B→4B，仍在涨） |
| L40S (sm_89) | 推理级 | 864 GB/s | **1.25–1.27×** |
| A10G (sm_86) | 推理级 | 600 GB/s | 1.04–1.08×（规模大了才过线） |
| RTX 5090 Laptop | 消费级 | 896 GB/s | 1.19–1.23× |
| T4 (sm_75) | 推理级 | 320 GB/s | 0.95–0.97×（**占用率受限**：64 KB SMEM → 每 SM 1 block） |
| **A100 (sm_80)** | **训练级** | 1555 GB/s | **0.79× → 0.55×**（随规模变差） |
| **H100 (sm_90)** | **训练级** | 3350 GB/s | **0.72× → 0.60×** |

作者反复强调：**排序不是带宽的干净函数**（864 GB/s 的 L40S 赢得比 600 GB/s 的 A10G 多），分界线是**推理级 vs 训练级 regime**。

等精度对照（bf16 vs bf16）是诚实的：AMK 的 GEMV 达 ≈460 GB/s（实测峰值的 63%），cuBLAS 的天花板是 ≈661 GB/s（实测峰值的 90%）。整体解码上输给 CUDA-graph cuBLAS **1.13×**、输给默认 vLLM **1.65×**。

### 3.4 瓶颈定位（方法论值得抄）

作者没有停在「我们慢」，而是 **A/B 测掉了两个显而易见的杠杆，两个都退步**：

| 假设 | A/B 结果 | 结论 |
|---|---|---|
| GEMV 是 load-latency 受限 → 加 cp.async 权重暂存环 | ring/sync = **0.82×**(A100) / **0.87×**(L4)，更慢 | **否**，不是访存延迟受限 |
| attention 是瓶颈 → split-KV | 同样退步（也增加跨 SM 同步） | **否** |

两个否定合起来把结构性瓶颈锁定在**每 tile 一次的跨 SM 同步**——megakernel 要付而 cuBLAS 不付的固定成本，**恰好在最快的训练级硅片上最严重**。剩下的杠杆是**更粗粒度的同步调度器**（每层更少的 grid 级屏障），不是 GEMV 流水也不是更激进的量化。

### 3.5 量化与自调优

| 精度 | µs/token | vs bf16 | 质量 |
|---|---|---|---|
| bf16 | 1537.4 | 1.00× | 参考 |
| **int8** | 1371.2 | **1.12×**（kernel-only 1.18×） | **greedy 无损**（32/32 token） |
| int4 | 1450.6 | 1.06× | 有损（~22% token 一致，文本仍连贯） |

int4 的权重流量下界降 **2.42×**（不是朴素的 4×——tied embedding、fp16 dequant scale、所有非 GEMV buffer 仍是 bf16，混合字节比约 0.41×），但每元素 dequant 的 ALU 开销和非 GEMV 部分的 Amdahl 定律把加速吃掉了。

自调优循环（propose → validate → 正确性闸门 → 测量 → keep/revert）在自己的默认配置上提升 **1.25–1.72×**；L4 上从落后（0.97×）到越过 cuBLAS 只用了约 **50 秒**搜索。

## 4. Limitations

**作者声明的（这篇的自我披露密度在系统论文里罕见）：**

- **不是形式化验证**，是「在可信基座之上的经验性可靠性」；可信基座 = 验证器本身 + 逐架构手写的 VM。
- **只有 Llama 系**：MoE 路由、滑窗/融合 QKV attention、部分或缩放 RoPE、带 bias 的投影全部超出范围，导入时拒绝。**MoE 在 Layer 3 roadmap 上，本文完全没做。**
- **只测 batch-1、position-0、空 KV**：按构造就是权重主导的，正好对准带宽下界这个论点，但**不覆盖长上下文下增长的 attention/KV 读取成本**。
- **没有硬件计数器**：Modal 账号上 ncu/Nsight 不可用（LibraryNotLoaded），所有利用率数字都是墙钟 + 解析 roofline 推出来的。
- **时钟未锁**：笔记本 GPU 从 180 MHz 起爬坡，数据中心 SM 时钟未 pin（A100 上 15.6 ms 的解码 std 高达 4.5 ms）。锁频重测显示 roofline 占比在 ±0.8 个百分点内，所以差距是 kernel 质量不是降频。
- **自筹经费、算力受限**：全部数据中心测量加起来「几个 GPU 小时」量级，跑不了长时间搜索也做不了 tensor-core/DP4A GEMV 的迭代开发。
- 两条 AMK 计时路径（kernel-only 配对交错 vs 整体解码含 host 重打包）**明确声明不混用**，不把更有利的数字悄悄挪过去。

**我要补充的：**

- 「零假接受」的强度取决于变异体生成器的覆盖面。8 个不安全类是作者自己定义的；**类之外的不安全模式不在测试范围内**。不过考虑到同步模型被收窄到只有单调计数器，可能的不安全模式空间确实有限，这个论证是站得住的。
- `partial_shared` 那一类 oracle 判定 0 个不安全、验证器拒绝 350 个——作者解释成「静态更严格」，我倾向同意，但严格说这也可能包含过度保守的成分，只是 360/360 真实 lowering 全通过间接排除了这个疑虑。
- vLLM 对比用了 `dtype=float32`（不是 vLLM 默认的 bf16），A100 那行只有 enforce_eager。作者自己披露了，但这意味着表 11 里唯一「AMK 赢 vLLM」的那行（A100 2.06×）含金量很低。

## 5. Our take

### 5.1 这就是 ROCmoe P2 缺的那一层

`rocmoe_DESIGN.md` 的 P2 说 megakernel 是一等构造，但**没有说这个 megakernel 的调度是怎么被验证的**。MonolithEP 现在的实际状态是：改一次 WG 角色划分、改一次 chunk 依赖、加一个新的 flag，验证手段是「跑一下看会不会挂」。

这篇给出的方案代价极低（CPU 上 5,150 调度/秒），而且**我们的同步模型已经很接近它的前提**：device-scope atomic + flag 轮询。要采纳，需要接受三条纪律：

1. 生产者**只能**递增计数器，不能做任意信号；
2. 消费者**只能**在静态已知的阈值上等待；
3. 每个任务**恰好**递增一个 `out_counter`。

**这对 MonolithEP 是真实的约束**，值得先做一次审计：现在的 dispatch/combine flag 机制符不符合这三条？如果不符合，改造成本是多少？

### 5.2 最该立刻自查的一个 bug 类

**「共享计数器全联接」规则**：如果 MonolithEP 里有任何地方是「等待 k 个 chunk 到达」，而那个计数器的生产者数量 > k，那我们就有一个 **first-k-of-N 竞争**——而且**跑测试永远发现不了**（作者的动态 oracle 在 350 个这类变异体上一个都没抓到）。

具体到我们的代码，需要检查的是：COMPUTE WG 等待 dispatch 完成的那个条件，阈值是不是恰好等于会递增该计数器的 COMM_DISPATCH WG 数量。如果我们用的是「等到至少 N 个 token 块可用」这种形式，就要确认拿到的是**哪些**块，而不只是**多少个**。

这一条我认为是这次五篇精读里**最可能直接抓出现有缺陷的一条**。

### 5.3 负面结果对 MI355X 的推论（比正面结果更重要）

作者定位出的结构性瓶颈是**每 tile 一次的跨 SM 同步**，并且明确指出它**在带宽最高的训练级硅片上最严重**：A100 (1.55 TB/s) 0.79→0.55×，H100 (3.35 TB/s) 0.72→0.60×。逻辑是同步成本基本固定，而字节流时间随带宽变短，所以固定成本的占比在快硅片上更高。

**MI355X 的 HBM3E 带宽在 8 TB/s 量级，比 H100 还高一倍多。** 按这个 regime 论证：

> 一个带每 tile grid 级同步的 megakernel，在 MI355X 上的相对代价只会比 H100 更差。

而且 AMD 侧的跨 CU 同步条件更不利：
- 没有 `grid.sync()` 的同等硬件支持（cooperative launch 在 ROCm 上语义和代价都不同）；
- 没有 DSMEM，跨 CU 只能过 L2 / Infinity Fabric；
- **跨 XCD 的原子操作要过 Infinity Fabric**，比同 XCD 内贵一个量级。

**结论：对 MonolithEP，「粗粒度同步」不是一个优化选项，是一个硬性设计约束。** 需要立刻确认两件事：
1. MonolithEP 当前的同步粒度是什么？每 tile？每 chunk？每层？
2. 有没有任何跨 XCD 的 grid 级屏障？如果有，它出现的频率是多少？

这条应该作为一条新的设计原则（或 P4 的子条款）写进文档：**同步点的数量和作用域是一等预算项，和 CU 数、LDS 一样要显式记账。**

### 5.4 值得抄的方法论：用 A/B 否定来定位瓶颈

作者遇到墙之后没有直接归因，而是把两个最显而易见的假设各做一次 A/B，**两个都退步**，从而反推出真正的绑定项。这比「我们认为瓶颈是 X」有说服力得多，而且成本很低。

MonolithEP 现在如果还有说不清的性能缺口，应该照这个模式做：列出 2–3 个显而易见的假设，各做一次 A/B，**明确报告哪些退步了**。

### 5.5 对 FlyDSL 的两条直接输入

1. **`ScheduleConfig` 的字段表可以直接当 FlyDSL 调度配置的起点**：tiling / fusion_grouping / sm_assignment / pipelining_depth / page_allocation / threads_per_block / smem_bytes_per_block。特别是 `sm_assignment` 的三档（round_robin / load_balance / explicit）——对应到 CDNA 就是 WG→CU 映射策略，而我们因为有 XCD 分层，可能需要第四档 `xcd_local`（呼应 [Fleet](./fleet.md) 的结论）。
2. **「一块新 GPU 是一条新的数据记录，永远不是一次 scheduler 修改」**——gfx942 与 gfx950 的差异（MFMA 形状、LDS 大小、DTOLDS 支持）应该全部收进一个 target 描述表，而不是散在 codegen 的 `if` 里。

### 5.6 需要注意的边界

- **它不做 MoE**（Layer 3 roadmap）。我们借的是 harness，不是应用。把它的验证器搬过来时，MoE 特有的动态性（每步变化的每专家 token 数）会带来新的问题：任务数本身是数据相关的，而 AMK 的 IR 任务数是静态的（随层数增长，不随数据）。**这是移植时最大的未知**——可能需要「按 shape bucket 预编译多份 SSC」的路子（参考 [HyperParallel-MoE](./hyperparallel-moe.md)）。
- 「correctness by construction」是相对于一个手写的、逐架构的 132 行 `scheduler.cu`。对我们其实是好消息：**可信基座小到可以人工完整审查**。我们的 CDNA VM 如果也能控制在几百行，同样的论证就成立。

## 6. 延伸阅读

1. **本文代码**（GitHub 开源）——`schedule/ir.py` 的 `validate()`（1150 行）是最该直接读的一个文件，9 条检查的实现都在里面。
2. **MPK / Mirage Persistent Kernel**（[2512.22219](https://arxiv.org/abs/2512.22219)）——本文最近的对照系统，有编译器和运行时但没有静态安全闸门。
3. **[HyperParallel-MoE](./hyperparallel-moe.md)**——同样把调度提到编译期，但用的是 SSC 而不是带验证器的 IR；两者互补，前者有 MoE 覆盖，后者有安全性论证。
4. **AutoKernel**（同作者前作）——单 kernel 层面的 agent 驱动优化，本文是它上升一个抽象层级的产物。

## 参考

- 论文：<https://arxiv.org/abs/2606.09682>
- 代码：<https://github.com/RightNow-AI/AutoMegaKernel>
- 本次检索的完整清单：[`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md)
