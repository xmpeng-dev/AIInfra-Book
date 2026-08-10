# Perseus: Eliminating Hidden Serialization in Multi-Node Megakernel Communication

> [arXiv 2605.00686](https://arxiv.org/abs/2605.00686) (v1, 2026-05-01, cs.DC) · Byungsoo Oh, Rachee Singh (Cornell) · CC BY 4.0
> 基线系统：FlashMoE（NeurIPS'25，Aimuyo/Oh/Singh，同组）+ NVSHMEM v3.5.21
> 硬件：NERSC Perlmutter（4×A100/节点，Slingshot-11 Cassini NIC，200 Gb/s dragonfly，最多 16 节点 / 64 GPU）· 商用云（8×H100/节点，ConnectX-7，InfiniBand NDR 400 Gb/s，最多 4 节点 / 32 GPU）
> 模型：Qwen3-30B-A3B（通信受限）· GPT-OSS-120B（均衡）· DeepSeek-V3（计算受限侧）

## TL;DR

**单节点 MoE megakernel 的胜利不能自动跨节点。** 把 SOTA megakernel 摊到 8 节点上，通信受限的 Qwen3-30B 比单节点趋势外推慢 **10×**，16 节点上弱扩展退化 **19×**。

根因不在 megakernel 抽象，而在 `putmem_signal_nbi` 落到 proxy-based RDMA 传输时被展开成 `Put → Fence → Signal`，**每个 Fence 都要把 NIC 流水里所有在飞的 Put 排空**，而 megakernel 恰恰要发大量并发小传输。实测：96 并发传输、8 节点下，coupled put+signal 的吞吐掉到 put-only 的 **2%**。

Perseus 用两个正交手段修掉：**decoupled signaling**（把 fence 从「每专家一次」降到「每远端 PE 一次」）+ **NIC-side ordering**（用 `FI_FENCE`/`IBV_SEND_FENCE` 让 NIC 硬件保序，proxy 不再阻塞）。Libfabric 上最高 **10.3×**，IBRC 上最高 **2.47×**，且让 proxy 路径反超 GPU-direct 的 IBGDA 最多 **1.2×**。

**对我们最关键的一条：这个病理在节点内不存在**（节点内 Put 走 NVLink/XGMI，proxy 不在路径上，Put-with-Signal 没有可观测开销）。所以 MonolithEP 现在的单节点数字是安全的；但它精确地定义了 MegaMonolith 出单节点时会撞上的那堵墙——而且 AMD 侧大概率**更容易**撞上（见 §5）。

## 1. Problem

### 1.1 megakernel 的执行模型（背景）

一个 MoE 层在 CPU 驱动模型下会派发几百个短 kernel（论文引 FlashMoE 的数据：DeepSpeedMoE 550 个/层，Megatron-LM 261 个/层），加上 AllToAll 的全局屏障，实测 A100 上 MoE 前向 GPU 空转 60–90%。

megakernel 的做法是把整个子图融成一个常驻 kernel，CTA 分角色：多数当 **processor**（跑 tile 上的 GEMM），少数当 **OS**（`subscriber` 解码对端到达通知 + `scheduler` 维护就绪队列）。tile 粒度（如 128×64）跟踪依赖，一个 tile 的输入就绪就能推进，不等其他 tile。FlashMoE 报告 SM 利用率 >93%，对比 CPU 驱动基线的 9–60%。

通信必须是 **device-initiated + one-sided**：host 发起的集合通信在常驻 kernel 里够不着，且集合的粒度是整层、完成即全局屏障，正是 megakernel 要消灭的东西。落地靠 PGAS 对称内存 + `put-with-signal`：把 token 写进远端对称 buffer，再写一个 flag；远端 subscriber 轮询 flag。

> 这套东西和 MonolithEP 的 WG 角色分区 + IPC scatter/gather + device-scope atomic flag 是同一个设计空间，只是它们跨节点、我们在节点内。

### 1.2 多节点弱扩展的塌方

固定每 GPU 工作量、节点数 1→8 扫描（弱扩展，理想是平线）：

| 模型 | TFLOPs/GB | 8 节点相对单节点趋势 |
|---|---|---|
| Qwen3-30B-A3B | 4.6（通信受限） | **慢 10×** |
| GPT-OSS-120B | 17.3（均衡） | 每翻倍退化 1.3–1.7× |
| Llama4-Scout-17B | 49.2（计算受限） | 仅 1.1–1.3× |

退化程度和通信/计算比强相关，SM trace 显示 processor CTA 在 tile 之间空等迟到的 signal。这是第一条指向传输层的证据。

### 1.3 根因：fence 引发的串行化

**并发传输数的公式**（值得记住，是设计期的预测器）：设 EP 组共 `P` 个 PE、模型 `E` 个专家、发送方本节点有 `P_local` 个 PE，则每次 dispatch 一个 PE 通过其**单条 proxy channel** 发出

```
(P - P_local) · (E / P)   个并发传输
```

Qwen3-30B、4 节点 16 GPU、Perlmutter 每节点 4 PE：`(16-4) × 8 = 96` 个并发传输挤同一条 channel。

**proxy 如何展开 put-with-signal**：NVSHMEM 每个 PE 只分配一条 proxy channel，GPU 上所有 CTA 的请求都写进同一个 host 侧队列，一个 CPU proxy 线程 FIFO 地喂给 NIC。为保证「signal 可见时 payload 必已到达」，proxy 把每次调用展开成 `Put → Fence → Signal`，**Fence 会等本 channel 上所有先前 Put 从 NIC 返回 completion**，期间 proxy 停止排空新请求。

节点内没有 proxy，所以这个 Fence 不存在——这正是单节点看不到问题的原因。

**微基准量化**（4 KB–4 MB，并发 1–128，2–8 节点）：

- 96 并发 / 8 节点：coupled put+signal 吞吐 = put-only 上限的 **2%**（Fig 5a）
- 96 并发下聚合 fence 时间：4 KB 消息 2 节点 0.96 ms → 8 节点 **6.1 ms**；1 MB 消息 3.5 ms → **9.2 ms**
- fence 占总通信时间：小消息最高 **98%**，即使 4 MB 仍 >19%；signal 本身 <0.2%

两个叠加效应：(a) 并发 CTA 把 `Put→Fence→Signal` 三元组交织进同一 FIFO，每个 fence 要等前面所有 CTA 的 Put；(b) 节点数增加后 Put 打向更多目的地，单个 fence 等的是**跨所有目的地的尾延迟**。

**megakernel 最想赢的区间（大量并发小 tile 传输，如 128×64 BF16 = 16 KB）正是 proxy signaling 退化最狠的区间。**

### 1.4 为什么朴素解法不行

- **bulk batching**：合并小传输能摊薄 fence，但等于退回粗粒度同步，overlap 收益归零。
- **加 proxy 线程/channel**：消掉单 FIFO 瓶颈，但跨 channel 保序又要引入新的同步。

## 2. Method

### 2.1 Decoupled signaling（megakernel 层）

**关键洞察：Fence 只需要在 Signal 之前，不需要在每个 Put 之后。**

两阶段协议（Algorithm 1）：

```
Phase 1（所有 CTA）
  stage tokens → put_nbi(...)            # 不带 signal
  atomic_add(counter[g], 1)              # 通知 leader
  if not is_leader(g): 让出，回调度器去跑计算

Phase 2（仅 group leader CTA）
  wait_until(counter[g] == group_size)
  fence()                                # 每组一次
  for e' in group g: signal(sig[e'], peer)
```

正确性靠传递性：原子计数器保证组内所有 Put 都已进入 proxy FIFO → 单 proxy 线程保序 → Fence 落在所有 Put 之后 → 所有 Signal 落在 Fence 之后。

**group 粒度**默认取 **per-PE**（一组 = 发往同一远端 PE 的所有专家），Qwen3 例子里 fence 数从 112 降到 4。扫描确认该默认值在 S=1K~64K 上都在最优的 2% 以内。

收益拆解（S=1K，8 节点）很有意思：

| 来源 | 延迟 |
|---|---|
| coupled 基线 | 22.7 ms |
| 仅解耦、group size=1（fence 数不变） | 19.9 ms（**−12%**） |
| group size=28（fence 112→4） | 12.3 ms（再 **−38%**） |

即**光是把提交顺序从「三元组交织」改成「PUT 批 + Signal 批」，不减 fence 数量，就有 12%**——因为 NIC 能自由流水连续的 PUT。

### 2.2 NIC-side ordering（传输层）

**关键洞察：保序不一定要 proxy 排空。** 现代 RDMA NIC 提供 per-request fence 标志（Libfabric `FI_FENCE`、IB verbs `IBV_SEND_FENCE`），NIC 遇到带标志的请求会用内部硬件寄存器推迟它直到同连接上所有先前请求完成。

于是 proxy 收到 fence 请求时只置一个 pending 标志就继续提交，下一个 Signal 带上标志交给 NIC 执行保序。**proxy 永不阻塞**；和解耦组合后，每组只有第一个 Signal 需要带标志。

实现量：NVSHMEM transport 模块改动 **<100 行**，只替换 `fi_cntr_wait`（Libfabric）/ `check_poll_avail`（IBRC）为 fence 标志。改动完全在传输共享库 `nvshmem_transport_*.so` 内，用户换库即可，**不需要改应用代码或重编译**。

**多 QP 适配（易踩的坑）**：IBRC 默认 round-robin 把发往同一 peer 的操作散到多个 QP，而 `IBV_SEND_FENCE` 只在 QP 内保序 → 依赖的 Put 与 signal 可能落到不同 QP，语义就破了。Perseus 用 `qp = pe % num_qps` 把同一 peer 的所有操作钉到确定 QP，既继承 QP 内 FIFO 又保留跨 peer 的带宽分散。

### 2.3 两者为什么互补

- **解耦**降的是 fence **频次**，在 fence 数量主导时更有效（小规模、小消息）。
- **NIC 保序**降的是**单次 fence 成本**，在 per-fence 开销主导时更有效（大规模）。

## 3. Experiments

### 3.1 端到端

| 传输 | 最高加速 | 备注 |
|---|---|---|
| Libfabric（proxy，Slingshot-11） | **10.3×** | Qwen3；GPT-OSS 2.8×，DSv3 2.2× |
| IBRC（proxy，ConnectX-7） | **2.47×** | S=64K、4 节点 |
| vs vanilla IBGDA（GPU-direct） | **1.2×** | Perseus 跑在 proxy 上反超 GPU-direct |

最后一行是全文最强的论点：**限制多节点 megakernel 的是保序机制，不是「proxy vs GPU-direct」这个选择**。作者还指出 IBGDA 要消耗 SM 周期做 NIC 提交，在 megakernel 里会和计算抢 SM；CPU proxy 没这个干扰，只是之前被 fence 卡死。

### 3.2 消融（Qwen3，Perlmutter）

| 配置 | fence 数 | 单次 fence 成本 | 8 节点加速 |
|---|---|---|---|
| vanilla | 112 | 高（全流水排空） | 1.0× |
| 仅 NIC 保序 | 112 | 低（硬件标志） | 1.3–2.6× |
| 仅解耦 | 28 | 高 | 1.2–1.6× |
| Perseus | 28 | 低 | **1.5–3.5×** |

2 节点时反过来：解耦（1.2–1.5×）优于 NIC 保序（1.1–1.4×），因为此时 fence 数量比单次成本更重要。

### 3.3 扩展性恢复与利用率

- 微基准：96 并发 / 4 KB / 8 节点，吞吐从 put-only 的 2% 恢复到 **74%**；大消息下与 put-only 齐平。
- 弱扩展：Qwen3 16 节点退化从 **19× 降到 3.5×**；GPT-OSS 基本拉平。
- TensorCore 利用率（4 节点，归一到各自单节点 = 100%）：

| 模型 | vanilla | Perseus |
|---|---|---|
| GPT-OSS-120B | 75% | **98%** |
| Qwen3-30B | 31% | **95%** |

### 3.4 路由倾斜鲁棒性

注入 Zipf 路由（指数 0→1.5，1.5 时前 10/128 个专家吃 82% token），加速保持：S=1K/8 节点 2.7×→2.0×；S=8K 时加速反而**随倾斜上升**（1.5×→2.1×），因为 Perseus 的优势在每字节成本上，传输变大时优势放大。

### 3.5 通用性：Triton-distributed

把 NIC 保序应用到 Triton-distributed 的 AllToAll（纯通信、无计算可重叠），**不改一行应用代码**：

- 固定开销 α 降低约 **99%**（4 节点 68.8 ms → 0.7 ms），最高 **79×** 加速（4 节点均值 59.6×）
- 对比 NCCL：vanilla 的 GPU-initiated AllToAll 平均比 NCCL **慢 18.7×**；上了 Perseus 后比 NCCL **快 11×**

这条很重要——它说明「GPU-initiated 点对点在多节点小消息上打不过 NCCL」这个业界常见观察，**很可能是传输层保序的伪影，而不是范式本身的问题**。

### 3.6 α-β 分解（可直接借用的方法论）

用 `T = α + β·M` 拟合（所有配置 R² > 0.99，M = EC×H×2 字节，EC = S·k/E）：

- Libfabric：α 主导。vanilla 的 α 随节点数增长，Perseus 基本平；Qwen3 16 节点 α 从 22.28 ms → **2.21 ms（−90%）**。
- IBRC：α 本来就小（1–5 ms，硬件 CQ 轮询比 Libfabric 的软件计数器排空轻），问题出在多 QP 排空抬高了 β；Perseus 把 β 降 **60%**（Qwen3），追平甚至超过 IBGDA 的 β。

这也解释了两个传输上加速趋势相反的现象：Libfabric 上 α 差距大 → 小 S 加速最高；IBRC 上 β 差距大 → 加速随 S 增长。

## 4. Limitations

**作者声明的：**

- 只做**推理前向**；与 vLLM / SGLang / Megatron 的集成列为 future work。
- GPU-direct（IBGDA）上收益有限：解耦 + warp 并行 signaling 最高 1.25×（均值 1.06×），且只在 2 节点 16×H100 做了初步评估。
- 多 proxy 线程不解决问题（跨线程保序会重新引入串行），作者选择直接攻击保序成本。

**我认为需要打问号的：**

- 「correctness 通过验证，error rate <1%」这个表述含混——BF16 下和 vanilla 同量级，但没说清是数值比对的相对误差还是别的。megakernel 做 bit-wise 可复现（对比 UniEP 的 token scoreboard + 确定性映射）在训练场景是硬需求，这篇没碰。
- 依赖 NIC 暴露 per-request fence 标志。作者说 `FI_FENCE`/`IBV_SEND_FENCE` 是标准 API，但对 EFA、Broadcom 等 fabric 没有实测。
- 多 QP 的确定性钉扎（`pe % num_qps`）在 peer 数不是 QP 数整数倍、或 peer 间流量不均时会造成 QP 负载不均，论文没量化这个代价。
- 弱扩展固定的是「每 GPU token 数」，同时 EP 度在涨，所以 E/P 在变——这让「退化纯粹来自扩展开销」的解读没有那么干净。

## 5. Our take

### 5.1 好消息：这个病理不威胁 MonolithEP 现有结论

论文明确写了：**节点内 Put 走 NVLink，proxy 不在路径上，Put-with-Signal 没有可观测开销**。MonolithEP / MMOE 是单节点 8 卡 IPC + XGMI，对应的就是这个「无 proxy」区间。所以 18.41 ms → ≤12 ms 那条基线不受影响。

### 5.2 坏消息：AMD 侧大概率**更容易**撞上这堵墙

- NVIDIA 至少还有 IBGDA 这条 GPU-direct 逃生通道；AMD 生态里 rocSHMEM（论文脚注 [5] 引了 ROCm rocSHMEM 和 ROCm DeepEP）在多数 fabric 上走的是 **proxy 路径**，也就是说这个 fence 病理对我们是**默认路径**而非边缘情况。
- 用论文的并发公式估一下 MegaMonolith 的目标场景：DSv3 `E=256`，8 节点 × 8 GPU = `P=64`，`P_local=8`，则每次 dispatch 每 PE 发 `(64-8) × (256/64) = 224` 个并发传输——**比论文里出问题的 96 还差一倍多**。

**待验证的三件事**（下次多节点实验前必须查）：
1. rocSHMEM 的 `putmem_signal_nbi` 等价物在 proxy 后端是否同样展开成 `Put→Fence→Signal`？每 PE 是否也只有一条 proxy channel？
2. AMD 常用的 NIC 栈（Broadcom Thor / ConnectX）在 ROCm 路径上是否暴露 `IBV_SEND_FENCE`？
3. ROCm DeepEP 的 dispatch 是否已经在做类似的批量 signaling？

### 5.3 可以直接抄的一条：decoupled signaling 是纯 kernel 层改动

Phase 1/Phase 2 + leader CTA + 每组原子计数器，**完全不依赖传输层**，而且和我们的 WG 角色分区天然对齐：per-PE grouping 就是「每个远端 rank 一个 leader COMM_DISPATCH WG」。

更值得注意的是它的副作用：非 leader CTA 在 Phase 1 结束后**立刻回到调度器去跑计算**，而不是卡在 put-with-signal 上。这个「让出」在节点内也有价值——MonolithEP 里 COMM_DISPATCH WG 发完就闲着，如果能回收成 COMPUTE，`224:16:16` 这个静态比例就可以更激进。这条值得单独做一次 A/B。

### 5.4 对 ROCmoe 设计文档的影响

- **P3（通信是可调度的一等流水阶段）得到强化**：这篇把「保序」也变成了一个可调度、可分组、可下放到硬件的对象，比我们文档里写的粒度更细。建议在 P3 下面加一条子原则：*signal 的保序开销要和数据传输分开记账*。
- **P6（核里 native，边界上标准）需要打补丁**：Perseus 的收益有一半来自改 NVSHMEM 传输层。如果 ROCmoe 坚持「通信库复用生态」，就等于把这一半收益让出去。需要明确 rocSHMEM 是「复用」还是「我们要 fork 的边界内组件」。
- **α-β 分解值得纳入 P5 的判据工具箱**：比单点 speedup 有用得多，且拟合成本极低（R²>0.99）。我们做 XGMI 建模时可以照抄这个模板。

### 5.5 与已读论文的关系

- **[UniEP](./uniep/README.md)**：同为 megakernel 融合 dispatch/combine，但 UniEP 只做节点内、强调 bit-wise 可复现；Perseus 补的是它跨节点时的传输层。两篇合起来才是 MegaMonolith 的完整路线图。
- **[MoE Tile Signaling](./moe-tile-signaling.md)**：tile epilogue signal 的思路和这里的 per-tile signal 同源，但那篇是 4×A100 单节点，没暴露 fence 问题。
- **[Comet](./comet.md)**：shared-tensor 依赖分解 + thread-block 专用化，也是单节点。
- **[DisagMoE](./disagmoe/README.md)**：走的是「把 all-to-all 变成 M2N 一等流水阶段」的另一条路，不依赖 device-initiated put-with-signal，因此天然绕开了这个病理——值得作为多节点方案的对照组重新评估。

## 6. 延伸阅读

1. **FlashMoE**（NeurIPS'25，同组）——本文的基线系统，必读，因为 Perseus 的所有改动都是在它上面做的。
2. **Mirage Persistent Kernel (MPK)**（[2512.22219](https://arxiv.org/abs/2512.22219)）——megakernel 编译器路线，和 AutoMegaKernel 一起看。
3. **TransferEngine / UCCL-EP**——用 WriteImm 通知绕开发送侧 fence，但接口和 memory-based put-with-signal 不同；如果我们能接受换接口，这是第三条路。
4. **Triton-distributed**（[ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)）——本文的通用性验证载体，79× 那组数字就出在它的 AllToAll benchmark 上。

## 参考

- 论文：<https://arxiv.org/abs/2605.00686>
- 本次检索的完整清单：[`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md)
