# MoE @ 万卡:瓶颈 × Primus 现状 gap 分析

> **定位:** 以「10000 卡 MoE 预训练」为场景,把已知瓶颈逐条对到 Primus(Primus-LM + Primus-Turbo)当前能力上,识别真正的空白与优先级。
> **视角:** AI Infra + AMD(MI300X/MI355X)· Megatron-LM 后端为主
> **关联:** [`../../knowledge/systems/training-1024g-stability-interview-notes.md`](../../knowledge/systems/training-1024g-stability-interview-notes.md) · [`../../knowledge/moe/paper-landscape.md`](../../knowledge/moe/paper-landscape.md) · [`2026-05-29_roadmap_h2_2026.md`](./2026-05-29_roadmap_h2_2026.md)
> **更新:** 2026-07-22

---

## 0. 核心判断(TL;DR)

- 几百~千卡,Primus 的优化重心是 **step time / MFU**;到 10k 卡,决定成败的是 **goodput = MFU × 有效时间占比(ETTR)**。
- Primus 在**通信重叠、显存、kernel、性能预估**四块已经很强(DeepEP、1F1B overlap、分层 recompute、projection)。
- Primus 在 **①容错/自愈 ②拓扑感知通信 ③运行时动态负载均衡** 三块相对 MegaScale 级系统有明显空白——而这三块恰好是 10k 尺度上吃掉 goodput 的主因。
- 一句话:**Primus 目前把「单步跑多快」做到了第一梯队,但「一次万卡长跑能保住多少有效算力」还缺一套系统化答案。**

---

## 1. 场景与什么变了

10k 卡相对 1k 卡的本质变化:

1. **同步语义 → 木桶效应**:单步时间 = 最慢 rank 的时间。任何长尾(慢卡/慢 NIC/GC/负载不均)都被放大成一等瓶颈。
2. **MTBF 坍缩**:万卡尺度平均每几小时一次故障(参考 Llama 3:16384×H100,54 天 419 次非预期中断,约一半 GPU/HBM 相关)。故障恢复开销直接决定 goodput。
3. **通信跨域加深**:EP/DP 度数变大,更多 all-to-all / all-reduce peer 走 IB 而非 NVLink;物理网络拓扑与拥塞开始限制有效带宽。
4. **气泡对 batch 敏感**:固定 global batch 下卡越多 → 每 iter 的 GA 越小 → pipeline bubble 上升。

> 衡量指标要从「峰值 MFU」升级到「goodput / ETTR」。这是本分析的主线。

---

## 2. 瓶颈 × Primus 现状 × Gap(核心表)

| # | 瓶颈(@10k) | Primus 现状(已有) | Gap(10k 尺度还缺什么) | 缺口 |
|---|---|---|---|---|
| 1 | MoE all-to-all 通信墙 | DeepEP 加速 dispatch(GPU 侧索引、sync-free、EP scaling 1.05–7.66×)、1F1B a2a overlap、Turbo grouped GEMM | overlap 是「聚合式」(attn+FFN 同组);长序列下 attn(二次)/FFN(线性)compute-comm 比失衡仍留残尾;无 attn/FFN 解耦(AFD) | 中 |
| 2 | DP/ZeRO 侧 all-reduce | BF16 梯度规约(`grad_reduce_in_bf16`)、BF16 优化器状态 | 万卡 DP 下梯度同步对慢节点极敏感;无分层/拓扑感知 all-reduce、无梯度压缩通信 | 中 |
| 3 | Pipeline 气泡 | 任意 PP 切分(`pipeline_model_parallel_layout`)、interleaved VPP、pp bubble projection、`pp_warmup` | 气泡建模是静态理想值,不含慢节点导致的动态气泡;无按实测重平衡 stage 的闭环 | 低-中 |
| 4 | 专家负载不均(straggler) | aux-loss 均衡 + DeepEP;sync-free MoE | **无运行时动态 expert 放置/迁移**;万卡下热点专家 → 全局等待。UltraEP/LAER-MoE/SwiftMoE 这类能力缺位 | **高** |
| 5 | 显存(激活主导) | 分层 recompute(`recompute_layer_ids`/`RECOMP_IDS`)、精度感知优化器、CP、memory projection | projection 覆盖显存但不建模 checkpoint IO 峰值;激活主导下 CP/recompute 的自动寻优仍靠人 | 低 |
| 6 | **容错 / 自愈 / goodput** | 手工节点池剔除、manual GC、NUMA 绑定、fast-exit;冒烟/稳态/长跑三阶段(均为**人工 SOP**) | **无自动故障检测 + 弹性重构 + 快速/异步/内存内 checkpoint**;无 straggler 自动识别驱逐;无 SDC(静默数据损坏)防护 | **高** |
| 7 | 网络拓扑 / 拥塞 | 手工绑定 `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` | 无 rail-aware / 层次化 all-to-all 放置、无拥塞感知路由;10k 尺度理论带宽会被哈希冲突打折 | 中-高 |
| 8 | 性能预估的适用性 | 混合 bench/仿真 projection,多节点误差 <10%;pp bubble projection | projection 目标是**理想吞吐**,不建模故障率/straggler/goodput;10k 决策更需要 goodput 预估 | 中 |

图例:缺口 = 相对 MegaScale/DisagMoE/UltraEP 级系统的差距。

---

## 3. Gap 优先级(影响 × 当前覆盖)

```
          高影响
            │  [6] 容错/goodput ★★★        [4] 动态负载均衡 ★★★
            │  [7] 拓扑感知通信 ★★
            │
            │  [1] AFD 通信解耦 ★★          [2] 分层 all-reduce ★
            │  [8] goodput projection ★★
低覆盖 ─────┼─────────────────────────────── 高覆盖
            │                               [3] 气泡(VPP 已强)
            │                               [5] 显存/recompute(已强)
            │                               [1] a2a overlap(DeepEP 已强)
          低影响
```

**结论:优先补 [6] 容错/goodput 和 [4] 运行时动态负载均衡**——影响最高、当前覆盖最低。[7] 拓扑感知通信次之。[1][2][3][5] 属于「已强,做增量」。

---

## 4. 三个高优先缺口的落地建议(结合 Primus)

### P0 — 容错 / goodput(缺口 6)
把现有的**人工三阶段 SOP** 沉淀为**自动化能力**:
- **快速/异步 checkpoint**:对标 ByteCheckpoint / Gemini(内存内 checkpoint),缩短「故障→恢复」窗口;projection 里加入 checkpoint IO 与恢复时间项。
- **自动故障检测 + 节点池闭环**:把 1024g 笔记里的「剔除异常节点」从手工变成在线健康检查 + 自动替换(对标 Unicron 自愈)。
- **弹性重构**:节点丢失后不整体重启,按 pipeline template 降级续跑(对标 Oobleck)。
- **goodput/ETTR 作为一等指标**:训练脚本与 dashboard 直接产出 ETTR,而非只看 step time。

### P0 — 运行时动态负载均衡(缺口 4)
- aux-loss 只治「统计均衡」,治不了「瞬时热点」。引入**动态 expert 放置/迁移**或 capacity-aware 调度(对标 UltraEP 的 rack-scale exact-load、LAER-MoE 的动态重排)。
- 与 DeepEP / sync-free 路径打通,避免迁移引入新的 CPU 同步。

### P1 — 拓扑感知通信(缺口 7)
- 把 HCA/接口绑定从「手工固定」升级为 **rail-aware 的 EP/DP rank 放置** + 层次化 all-to-all(节点内 NVLink 聚合、节点间 IB),对标 HPN / Meta RoCE@Scale。
- 与 DeepEP 的多节点路径结合,减少跨 spine 的冗余流量。

### 研究性储备 — AFD / DisagMoE(缺口 1)
- 长序列 + 大 EP 场景下,Primus 的聚合式 1F1B overlap 会遇到 attn/FFN compute-comm 比失衡的残尾。可评估 **DisagMoE 式 attn/FFN 解耦 + roofline/MILP 分 GPU/NIC** 作为 H2 研究项(注:单节点 XGMI 带宽充裕,收益上限低于跨机 IB 稀缺场景,需先算 R 判据)。

---

## 5. 相关论文与交叉引用

**Primus 已强的方向(增量参考):**
- 通信重叠:[`../../papers/comet.md`](../../papers/comet.md) · [`../../papers/disagmoe/`](../../papers/disagmoe/README.md)([arXiv:2605.11005](https://arxiv.org/abs/2605.11005))
- 并行/气泡:[`../../papers/moe-parallel-folding.md`](../../papers/moe-parallel-folding.md)

**高优缺口的对标论文(建议补深度笔记):**
- 容错/万卡工程:**MegaScale**(NSDI'24,>10k GPU)、**Gemini**(SOSP'23,内存内 checkpoint)、**Oobleck**(SOSP'23,弹性 pipeline)、**Unicron**(自愈)、**ByteCheckpoint**;Meta《The Llama 3 Herd of Models》infra 章 —— 目前 `paper-landscape.md` **未收录容错/可靠性一类**,是索引空白。
- 动态负载均衡:[`../../papers/ultraep/`](../../papers/ultraep/README.md) · [`../../papers/laer-moe-fsep.md`](../../papers/laer-moe-fsep.md) · [`../../papers/swiftmoe.md`](../../papers/swiftmoe.md)
- 网络:Alibaba **HPN**(SIGCOMM'24)、Meta **RDMA over Ethernet at Meta Scale**(SIGCOMM'24)。
- 生产 MoE 万卡:[`../../papers/megascale-moe.md`](../../papers/megascale-moe.md)(EuroSys'26,~42% MFU @10K)。

**后续动作建议:**
1. 在 [`../../knowledge/moe/paper-landscape.md`](../../knowledge/moe/paper-landscape.md) 新增「H. 大规模可靠性 & 网络」分类,先补索引。
2. 为 MegaScale / Gemini / Oobleck 各写一篇深度笔记(按 `papers/README.md` 字段规范)。
3. 把本文的 P0(容错、动态均衡)接入 [`2026-05-29_roadmap_h2_2026.md`](./2026-05-29_roadmap_h2_2026.md) 的 H2 计划。

---

*本分析基于 Primus MoE blog(`Primus/docs/tech_blogs/moe_package_2.0/` + `examples/moe_package/README.md`)、1024 卡稳定性笔记与 MoE 论文库整理于 2026-07-22。数字为工程判断,非官方性能声明。*
