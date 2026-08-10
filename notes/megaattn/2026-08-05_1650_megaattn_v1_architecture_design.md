# 2026-08-05 16:50  MegaAttn v1 架构设计：indexer + top-k + sparse MLA 三段融合 on gfx950

> **When**: 2026-08-05 16:50 UTC+8
> **Where**: 设计阶段，尚未落码；目标平台 8x MI355X (gfx950, CDNA4)
> **Context**: MegaMoE v2 (`FlyDSL/kernels/mega_moe/`, commit `dc8e1539`) 收口后的下一个 megakernel 目标。把 MegaMoE 在 MoE 路径上验证过的 role-split persistent kernel 打法，搬到 DeepSeek Sparse Attention 的 decode 链路上。
> **模型目标**: DeepSeek-V3.2 / V4 家族，MLA absorbed 模式 (QK 576 = 512 latent + 64 rope, V 512, 128 heads)，DSA indexer 64 heads x 128 dim，index_topk = 2048

## TL;DR

把 DSA 的三段 —— **lightning indexer (paged MQA logits) → top-k 2048 → sparse MLA** —— 融进一个 persistent kernel，命名 **MegaAttn v1**。

三条核心论点：

1. **中间那个 `[next_n, S_kv]` fp32 logits 张量不该落 HBM**。它是 indexer 的输出、top-k 的唯一输入，除此之外没有任何消费者。融合后用 LDS 里的 running threshold 在线剪枝，只回写越过阈值的候选（约 `3k` 个），把 top-k 的输入从 `N x 4B` 压到 `~24 KB`。
2. **top-k 的 wall time 占比远高于它的字节占比**。indexer 搬的字节多但跑在 streaming roofline 上；top-k 搬的字节少但受全局同步、histogram atomic 竞争拖累，有效带宽低得多。所以这一段是靠算法（在线阈值 + 跨 step 时序复用）而不是靠带宽优化拿收益。
3. **它和 MegaMoE 是同一个骨架**。打分 → top-k 选择 → 按索引 gather → 算，四段一一对应；`mega_moe_stage1.py` 的 role-split grid、`DispatchSlot` 信号表、`WORK_HEAD/WORK_TAIL` 原子工作队列可以直接搬，不需要从零设计调度层。

预期收益需要 M0 实测后再拍。设计目标：decode @ 128K context 单层 attention wall 相对「三个独立 kernel」基线降 25-35%，prefill chunk 场景降幅更大（logits 中间张量在 prefill 下大两三个数量级）。

---

## 1. Background

### 1.1 FlyDSL attention 现状

`kernels/attention/` 共 18,338 行，覆盖 FMHA（gfx950 FP8 已做过两轮优化，`ca99de5e` / `05d785e5`）、PA decode（tile / fp8 / swa）、MLA decode、以及 `qk_norm_rope_quant` 这类前处理融合。

两个硬缺口：

**稀疏路径完全没有**，而且是显式挡掉的：

```1241:1243:/home/xiaoming/workspace/FlyDSL/kernels/attention/pa_metadata.py
    assert is_causal, "FlyDSL pa_metadata only supports causal"
    assert topk == -1, "FlyDSL pa_metadata does not support sparse (topk)"
    assert uni_seqlen_qo >= 1, "FlyDSL pa_metadata requires uniform qo length"
```

**backward 没有**。全仓只有 `flash_attn_interface.py:620` 一处注释提到 LSE "Needed by backward"，没有对应实现。MegaAttn v1 同样只做前向，跟 MegaMoE 保持一致；backward 是独立议题。

现有 MLA decode 的形状约定（`mla_fwd_decode_m16x8_fp8_fp8.py:45-47`）已经跟 DSv3.2 对齐：

| 量 | 值 | 说明 |
|---|---|---|
| `QK_HEAD_DIM` | 576 | 512 nope (KV lora latent) + 64 rope |
| `V_HEAD_DIM` | 512 | `KV_LORA_RANK`，absorbed 模式 |
| `NUM_QO_HEADS` | 128 | |
| `PAGE_SIZE` | 1 | 每页一个 token |
| dtype | FP8 q + FP8 kv | |

也就是说 **sparse MLA 那一段的算子本体已经在了**，缺的是「按索引取 KV」而不是「按连续区间取 KV」。

### 1.2 DSA 的数据流与真实形状

从 DeepGEMM (`~/workspace/MegaMOE/`) 的 indexer 实现反推语义（`tests/test_attention.py:113-222`，`deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh`）：

```
q       : [next_n, 64, 128]      indexer 有独立的 64 个 head，dim 128
kv      : [S_kv, 128]            MQA，单 KV head，与 MLA 共用压缩 latent cache
weights : [next_n, 64]           fp32，每 head 一个标量权重

logits[t, j] = sum_h relu(q[t,h,:] . kv[j,:]) * weights[t,h]     -> [next_n, S_kv] fp32
             (再按 causal / cu_seqlen_ks,ke 掩码)

topk_idx = topk(logits[t], k=2048)                               -> [next_n, 2048] int32
out      = MLA(q_mla[t], kv_latent[topk_idx])                    -> 只算 2048 个位置
```

关键结构事实：**DeepGEMM 只提供到 logits 为止**（`fp8_mqa_logits` / `fp8_fp4_paged_mqa_logits` / `get_paged_mqa_logits_metadata`），top-k 在库外（FlashInfer / AITER）。也就是说 indexer 和 top-k 之间今天是一条**库边界**，而这条边界恰好是访存上最该消掉的地方 —— 这是 MegaAttn 的立足点。

### 1.3 为什么是现在

H2 2026 roadmap（[`../primus-moe/2026-05-29_roadmap_h2_2026.md`](../primus-moe/2026-05-29_roadmap_h2_2026.md)）已经把这些列成散件需求：

| roadmap 行 | 内容 |
|---|---|
| L96 | DSA (DeepSeek Sparse Attention) 支持，从 Megatron-LM#2440 移植 |
| L97 | Hybrid Attention with CSA + HCA |
| L120 | Fused dense MQA kernel for AMD (AITER backend) |
| L121 | Fused CSA / HCA kernel for AMD |
| L260 | FP8 router / **FP8 indexer** for DeepSeek-DSA family |
| L241 | MLA RoPE fusion (FP8-aware) |

MegaAttn 不是新增需求，是把 L120 / L121 / L260 这三件**当成一个 kernel 一次做完**，而不是三个算子各自优化后再拼。

### 1.4 外部工作定位

| 工作 | 出处 | 做了什么 | 对 MegaAttn 的价值 |
|---|---|---|---|
| SnapMLA | [2602.10718](https://arxiv.org/abs/2602.10718)，美团 Longcat | FP8 MLA decode：RoPE 部分保高精度的 per-token KV 量化、PV GEMM scale 对齐、fused Q-quant + K-append | 直接借数值方案；Hopper 实现，AMD 侧需重做。报 1.91x 吞吐 |
| LiteTopK | [2607.11976](https://arxiv.org/abs/2607.11976) | 融合 indexer-TopK：用上一 chunk 的 top-3k 做采样估计分布，等宽分箱在线维持紧阈值，只回写有希望的候选，保持精确 top-k | **核心可借鉴项**，就是本设计的在线剪枝依据。报 GLM 5.2 prefill 1.2x |
| Guess-Verify-Refine | NVIDIA 技术报告，2026-04-30 | 利用相邻 decode step 的 top-k 集合重叠（层间 35-50%，个别层近 60%）做猜测-验证-修正的精确 top-k | decode 场景的算法层加速，与融合正交可叠加。报比生产 radix-select 再快 1.88x 均值 / 2.42x 峰值 |
| ROCm Adaptive Top-K | [ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/adaptive-topk/README.html) | K<=128 走寄存器 bitonic（DPP + med3），K>128 走 radix_11bits | **注意**：DSA 的 k=2048 落在 radix 那一侧，blog 里最漂亮的寄存器 bitonic 优化用不上。AMD 侧目前只有独立算子，没有融合实现 |

AMD 侧的融合实现是空白，这是做 MegaAttn 的窗口。

---

## 2. 瓶颈分析

### 2.1 单层单序列 decode 访存量（估算）

设上下文 `N` tokens，`next_n = 1`，k = 2048，indexer dim 128 FP8，MLA latent 576 B/token FP8。

| 段 | 读 | 写 | N = 32K | N = 128K |
|---|---|---|---|---|
| indexer (MQA logits) | `N x 128 B` | `N x 4 B` | 4.2 MB + 128 KB | 16.8 MB + 512 KB |
| top-k (radix, R≈3-4 pass) | `R x N x 4 B` | `2048 x 4 B` | 384-512 KB | 1.5-2.0 MB |
| sparse MLA | `2048 x 576 B` | `128 x 512 x 2 B` | 1.18 MB | 1.18 MB（与 N 无关） |
| **合计** | | | **~6.0 MB** | **~20.0 MB** |
| 其中可被融合消掉 | logits 往返 | | ~0.6 MB (10%) | ~2.3 MB (12%) |

按 MI355X 8 TB/s 标称带宽，N=128K 时 20 MB 的理论下限约 2.5 µs/层/序列。

### 2.2 为什么 top-k 值得单独治

字节上 top-k 只占 8-10%，但 wall time 占比要高得多，原因是三段的有效带宽差距很大：

- **indexer** 是干净的 streaming MQA GEMM，KV 顺序读，能接近 roofline。
- **sparse MLA** 是 2048 个非连续 page 的 gather，`PAGE_SIZE=1` 意味着每个 token 一次独立寻址，但总量固定且可以预取。
- **top-k** 是多趟 radix：每趟之间要全局同步，histogram 阶段有 atomic 竞争，且必须等 indexer 整个 kernel 结束才能启动。它搬的字节少，但达到的有效带宽远低于前两者。

NVIDIA GVR 报告的定性结论与此一致：top-k 在 8K 时无关紧要，到 128K 成为主要瓶颈，且他们的收益来自**算法**（跨 step 复用）而非带宽优化。这说明单纯把 top-k 写快没用，得改它的输入规模和启动时机。

### 2.3 融合能拿到什么

三件事，按预期贡献排序：

1. **在线阈值剪枝**（LiteTopK 思路）：indexer 边算边在 LDS 维护一个近似阈值，只把越过阈值的候选写回。top-k 输入从 `N x 4 B` 降到约 `3k x 4 B ≈ 24 KB`，与 `N` 解耦。这同时消掉了 logits 的写和 radix 的多趟读。
2. **消灭 kernel 边界**：三次 launch → 一次，两次全局同步 → 零。decode 场景下单层三次 launch 的固定开销在小 batch 时不可忽略。
3. **段间流水**：indexer 处理第 `i` 个 KV 分块时，第 `i-1` 块的候选已经可以进入 top-k 的分箱统计；top-k 一旦定出 2048 个索引，sparse MLA 的 KV gather 可以立刻起步。这就是 MegaMoE 里 dispatch 与 FC1 chunk-overlap 的同一套东西。

**prefill 场景收益更大**：logits 是 `[chunk_len, S_kv]`，chunk_len=4096 / S_kv=128K 时 fp32 中间张量 2 TB 量级，实现上必然分块，但分块本身就是被迫的额外往返。融合后这个张量根本不存在。LiteTopK 在 GLM 5.2 prefill 上报 1.2x 佐证了这一点。

---

## 3. 设计

### 3.1 与 MegaMoE 的同构映射

这是整个设计的立论基础 —— 不是「再写一个大 kernel」，而是同一个骨架换负载：

| MegaMoE v2 | MegaAttn v1 | 复用程度 |
|---|---|---|
| router 打分（每 token 对 256 expert） | indexer MQA 打分（每 query 对 N token） | 结构同，负载不同 |
| top-k 选 8 个 expert | top-k 选 2048 个 token | k 大三个数量级，算法要换 |
| dispatch：按 expert 分组 + permute + 跨卡搬运 | gather：按 top-k 索引取 latent KV page | 单卡，无通信，**更简单** |
| grouped GEMM1 + SwiGLU + GEMM2 | sparse MLA (QK + softmax + PV) | 算子不同，tile 调度同 |
| combine：加权求和回原顺序 | split-KV reduce（已有，`split_output`/`split_lse`） | 已存在 |
| `DispatchSlot` 32 槽设备端信号表 | 同一套，改语义 | **直接搬** |
| `WORK_HEAD`/`WORK_TAIL` 原子工作队列 + `EPOCH_GATE` generation ticket | 同 | **直接搬** |
| planner(1) + dispatch(32) + consumer(剩余) 的 role-split grid | scorer + selector + attn 三角色 | **直接搬** |

关键差异是 **MegaAttn 没有跨卡通信**（单卡 decode 内），所以 `mori.shmem` 那一层整个不需要。这让 v1 比 MegaMoE 简单一档：MegaMoE 最难的部分（P2P 信号、fixedslot vs compact 两套 dispatch、跨 rank epoch 同步）在这里全部消失。

### 3.2 Role-split persistent grid

沿用 `mega_moe_stage1.py:79-84` 的做法 —— 固定 CU 倍数、角色按 block 序号划分、不留 tail：

```
launch_grid_x = scorer_blocks + selector_blocks + attn_blocks   (= num_cu * grid_mult)

scorer   (多数 CU)  : 流式扫 KV，算 MQA logits，在线分箱 + 阈值剪枝，
                      候选写入 device-resident 候选池，原子推进 WORK_TAIL
selector (少数 CU)  : 消费候选池，维护全局 top-k 堆 / 二次 radix，
                      定稿后写 topk_idx 并 release SELECT_READY
attn     (多数 CU)  : spin 在 SELECT_READY 上；一旦就绪按 topk_idx gather latent KV，
                      跑现有 MLA m16x8 fp8 tile loop，split-KV 部分和走已有 reduce
```

`scorer` 与 `attn` 的 CU 配比需要 autotune。参考 MegaMoE 的先例：`num_dispatch_cu=32 / num_cu=256` 是 1:7，且 rocmoe 那边的经验是**角色间物理隔离 CU 比共享调度更好**（[`../rocmoe/2026-05-23_2216_UP_m4a_wg_per_cu_1_drops_super_dispatch_35pct.md`](../rocmoe/2026-05-23_2216_UP_m4a_wg_per_cu_1_drops_super_dispatch_35pct.md)，`__launch_bounds__(_, 1)` 物理隔离让 dispatch wall 降 35%）。

同步原语上沿用 rocmoe v2 的结论：**不用 cross-WG barrier，用 per-block scoreboard + `atomic_load_acquire` spin**，避免打 atomic。

### 3.3 三处关键设计选择

**(a) 候选池的结构。** scorer 不写全量 logits，写 `(score, token_idx)` 二元组到一个环形候选池。阈值来源两条路，decode 和 prefill 分开：

- **decode**：走 GVR 思路，用**上一个 decode step 的第 2048 名分数**作为初始阈值。相邻 step 的 top-k 集合重叠 35-50%，这个阈值天然接近真值，一上来就能剪掉绝大多数候选。上一步的索引和分数本来就在设备上，零额外成本。
- **prefill**：走 LiteTopK 思路，用上一个 chunk 的 top-3k 分数做采样估计，等宽分箱在线收紧。

两条路都必须保证**精确 top-k**：阈值只用来决定「是否写回候选」，若最终候选数不足 2048 则回退重扫（记录 fallback 计数器，autotune 时监控回退率）。

**(b) `PAGE_SIZE=1` 的 gather 代价。** 现有 MLA decode 是 `PAGE_SIZE=1`，2048 个 top-k 索引意味着 2048 次独立寻址、每次 576 B。这不算差（576 B 已经够一次 wide load），但索引是无序的，会打乱 L2 局部性。设计上让 selector **对 topk_idx 排序后再交给 attn**（2048 个 int32 的排序在 LDS 里很便宜），换回顺序访问。这一点值得在 M1 单独 A/B。

**(c) 数值方案。** indexer 走 FP8（V3.2）/ FP4（V4）；MLA 走 FP8。直接采纳 SnapMLA 的两条结论：RoPE 那 64 维保高精度不量化（KV cache 的量化敏感度是异质的），per-token 粒度对齐自回归解码。scale 对齐问题在 absorbed 模式下同样存在，需要在 M1 阶段用 DeepGEMM 的 `fp8_fp4_mqa_logits` 做逐位对拍。

### 3.4 接口改动：work_info 从「区间」到「索引表」

这是对现有代码影响最大的一处。`pa_metadata` 现在的工作项是一个**连续 KV 区间**：

```26:29:/home/xiaoming/workspace/FlyDSL/kernels/attention/pa_metadata.py
work_info layout (8 x int32 per work), matching ``PaWorkInfo``:
  [0] batch_idx  [1] partial_qo_loc(-1 if no split)  [2] qo_start  [3] qo_end
  [4] kv_start   [5] kv_end                          [6] kv_offset(=0)
```

稀疏路径下 `[kv_start, kv_end)` 不再成立，工作项要变成「topk_idx 数组的一个切片」。两种改法：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 保留 range 语义 | 让 `kv_start/kv_end` 索引进 `topk_idx` 数组而不是 KV cache 本身，attn 端多一层间接寻址 | 改动最小，`pa_metadata` 的 split-KV 负载均衡逻辑原样可用 |
| B. 新增 sparse work_info 字段 | 扩 8 字段布局，加 `topk_base` / `topk_len` | 语义清晰但要动 `PaWorkInfo` ABI 和 AITER 兼容层 |

**倾向 A**。因为 k=2048 固定，split-KV 的负载均衡在稀疏路径下反而更简单（每个 work 分到的 token 数是确定的），`pa_metadata.py` 那套 `num_splits_per_khead` 的分配逻辑可以不动，只是被索引的对象换了一层。

---

## 4. 分阶段计划

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **M0 摸底** | 在 gfx950 上跑通「三个独立 kernel」基线：AITER/现有 indexer + ROCm adaptive topk + FlyDSL MLA decode。分段 rocprof，量出 N = 8K / 32K / 128K 下三段的实际 wall 占比 | 拿到分段占比表。**若 top-k + logits 往返合计 < 15%，整个设计的前提不成立，就地叫停改做别的**（见 §6 风险 R1） |
| **M1 独立算子** | 在 FlyDSL 里各自实现：(a) paged MQA logits kernel，FP8 先行；(b) k=2048 的 radix top-k。与 DeepGEMM `fp8_fp4_mqa_logits` 逐位对拍 | 精度对齐；单算子性能不差于 AITER / ROCm blog 实现 |
| **M2 融合前两段** | indexer + 在线阈值剪枝 + top-k 合成一个 kernel。decode 走 GVR 上步阈值，prefill 走 LiteTopK 分箱。带 fallback 计数器 | 相对 M1 两段之和降 ≥ 30%；精确 top-k 逐位一致；回退率 < 1% |
| **M3 全链路** | 接入 sparse MLA，role-split persistent grid，`pa_metadata` 走方案 A | 单层 attention wall 相对 M0 基线降 ≥ 25% @ 128K；端到端 DSv3.2 decode 精度不掉 |
| **M4 (可选)** | prefill 路径、FP4 indexer (V4)、CSA/HCA 变体 | 按 V4 落地节奏定 |

M0 是硬门槛，不做完不进 M1。这一点吸取 rocmoe 的教训 —— 那边在 overlap 上限上翻盘过两次（[`../rocmoe/2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md`](../rocmoe/2026-05-22_1545_FLAT_fc1_fc2_roofline_recalibrates_m2_g_overlap_budget.md)），都是因为先动手写 kernel、后补 roofline。

---

## 5. 复用清单

落码时直接搬的东西，避免重复设计：

| 来源 | 组件 | 用途 |
|---|---|---|
| `kernels/mega_moe/dispatch.py:17-52` | `DispatchSlot` IntEnum（32 槽）+ 设备端信号表 | 改语义即可：`SELECT_READY` / `CAND_HEAD` / `CAND_TAIL` / `EPOCH_GATE` |
| `kernels/mega_moe/mega_moe_stage1.py:79-84` | role-split grid 划分，固定 CU 倍数不留 tail | 直接照抄结构 |
| `kernels/mega_moe/mega_moe_stage1.py:164-182` | `EPOCH_GATE` generation ticket，跨 launch 持久化 | persistent kernel 复用启动必备 |
| `kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py` | MLA m16x8 FP8 tile loop、split-KV reduce | 只改取数路径，算子本体不动 |
| `kernels/attention/pa_metadata.py` | worklist 负载均衡调度器 | 方案 A 下逻辑不动 |
| `kernels/attention/qk_norm_rope_quant.py` | 融合的 norm + RoPE + 量化 | indexer 的 q 侧前处理 |
| slab skill | `.cursor/skills/cco-pipeline-overlap/SKILL.md` | 段间流水设计 |
| slab knowledge | `knowledge/kernels/memory-access-patterns.md` 五问清单 | 候选池写入 / KV gather 的访存审查 |

---

## 6. 风险

| # | 风险 | 判定方式 | 缓解 |
|---|---|---|---|
| R1 | **indexer 本身就占了绝大部分时间**，融合掉的 logits 往返只是零头，天花板不到 15% | M0 分段 profile | 这是最大的风险。M0 直接证伪就停，转 §7 的备选方向 |
| R2 | 在线阈值在分布尖锐/平坦的极端 batch 上回退率高，反而变慢 | M2 用真实 DSv3.2 分数分布压测，统计回退率 | 保留「阈值失效即退化为标准 radix」的路径，最坏等于不融合 |
| R3 | role-split 下 scorer 和 attn 抢 L2，各自都掉速（rocmoe M2-G α 出现过 dispatch +53% 的先例） | M3 A/B：融合 vs 独立 | 用 `__launch_bounds__` 做 CU 物理隔离，rocmoe M4-α 证明有效 |
| R4 | V4 的 CSA + HCA 混合注意力把 indexer 语义改了，M1-M3 白做 | 跟 Megatron-LM#2440 的移植进度 | v1 只锁 V3.2 语义；把 indexer 打分做成可替换模块 |
| R5 | 无序 gather 打乱 L2，sparse MLA 比 dense 的每 token 成本高很多 | M1 单独 A/B 排序 vs 不排序 | selector 端排序 topk_idx（2048 个 int32，LDS 内很便宜） |

---

## 7. Next

**P0**：M0 摸底。在 gfx950 上搭「三个独立 kernel」基线并分段 profile，N = 8K / 32K / 128K 三档。这一步决定整个方向是否成立（R1）。

**P1**：M0 通过后进 M1，先做 FP8 paged MQA logits，拿 DeepGEMM `~/workspace/MegaMOE/tests/test_attention.py` 的用例做对拍基准。

**备选方向**（M0 证伪时切过去，优先级从高到低）：

1. **Attention 通信融合**（CP 方向）—— Ulysses 的 head↔seq all-to-all 或 ring attention P2P 吃进 kernel。这才是 MegaMoE「把 all-to-all 吃进 kernel」在 attention 上的严格对应，对万卡长上下文训练价值最直接，先例也最少。难度高于 MegaAttn v1。
2. **Mega-MLA**（decode 全链路：q_down/up proj + RoPE + quant + paged MLA + o_proj 融一个 kernel）—— SnapMLA 已把 FP8 数值部分做完并开源，主要工作是移植 + AMD 调优，论文性弱但工程收益确定。
3. **FMHA backward megakernel** —— 训练侧真正的缺口，但工程量大。

**不跟的方向**：通用 megakernel 编译器（MPK / OSDI'26、Event Tensor、AutoMegaKernel）。它们与 FlyDSL 本身是竞争关系而非补充。

---

## 参考

- MegaMoE v2 实现：`~/workspace/FlyDSL/kernels/mega_moe/`（commit `dc8e1539`，4199 行）
- DeepGEMM indexer 参考实现：`~/workspace/MegaMOE/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh`
- SnapMLA: [arXiv 2602.10718](https://arxiv.org/abs/2602.10718)
- LiteTopK: [arXiv 2607.11976](https://arxiv.org/abs/2607.11976)
- NVIDIA Guess-Verify-Refine Top-K for DSA Decoding（2026-04-30 技术报告）
- ROCm Adaptive Top-K: [ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/adaptive-topk/README.html)
- H2 2026 roadmap：[`../primus-moe/2026-05-29_roadmap_h2_2026.md`](../primus-moe/2026-05-29_roadmap_h2_2026.md)
- rocmoe v2 架构设计（role-split / scoreboard 先例）：[`../rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md`](../rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md)
- 训练优化 landscape：[`../../knowledge/systems/training-optimization-landscape-2026.md`](../../knowledge/systems/training-optimization-landscape-2026.md)
