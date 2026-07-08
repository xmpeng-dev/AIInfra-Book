# 参考设计研读 — DeepGEMM Mega MoE 的 FC2 → combine 重叠机制 (SM100)

> **When**: 2026-06-25 10:56 UTC+8
> **Where**: slab 知识研读 (无 GPU 运行, 纯代码阅读)
> **Context**: 接续 [dispatch→FC1 研读](./2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md), 把同一套 "详细设计 / 流程 / overlap 原因 / ROCm 移植" 的分析延伸到 MoE 后半段 FC2 + combine; 为 rocmoe-v2 的 combine 方向抉择 (pull vs push) 提供参考实现对照

## TL;DR

DeepGEMM 的 FC2 **复用 FC1 的同一条 GEMM 流水线**(同 MMA warp + stage barrier, 经 L2 ring `l2_full`/`l2_empty` 解耦)。combine 被拆成两半且方向与 dispatch **相反**: (a) **FC2 epilogue 直接 push-scatter**——用 dispatch 写好的 `TokenSrcMetadata` 把每行结果 fire-and-forget 写回源 token 所在 rank 的 combine buffer(融合了 "FC2 输出写回" 和 "all-to-all combine scatter", NVLink 写分散在整个 GEMM 时间线上被 MMA 藏掉); (b) **本地 reduce**——所有 rank push 完后, 各 rank 本地把 kNumTopk 份 partial 双缓冲归约写 `y`(无跨卡流量, 且与 dispatch workspace clean 错峰重叠)。**最值得 rocmoe 关注**: DeepGEMM 用 dispatch=pull / combine=push 的非对称设计, 而 rocmoe 现在 combine 也用 pull(为躲 8-outbound 写竞争)——DeepGEMM 说明 "融合式 push" 把写突发摊平后可能反而更快, 这是应实测推翻/确认的假设, 不该先验排除。

## Background

rocmoe-v2 设计核心 #2 是 "sender push → receiver pull", 理由是砍 "FC2 出口 8 outbound XGMI 写竞争 ~0.7 ms"(见 [架构设计 note](./2026-05-21_1252_rocmoe_v2_architecture_design.md)); M5 规划 "atomic-free combine pull(寄存器归约)" 移植 MonolithEP `combine.hip`。DeepGEMM Mega MoE 给出了一个反例: combine 用 push, 但把它**融进 FC2 epilogue** 而非 standalone kernel。本篇逐行研读其 FC2 + combine 路径, 对照 rocmoe 的 pull-combine 抉择。

研读对象(`github.com/deepseek-ai/DeepGEMM`, main):
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`: FC2 = `Linear2` 分支(epilogue 段), combine reduce = 末尾段
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`: `combine_token_buffer`(`[kNumTopk][kNumMaxTokensPerRank]` bf16)、`TokenSrcMetadata`、`l2_full`/`l2_empty` 计数器

## 主要发现 / 结论

### 1. FC2: 复用 FC1 的同一条 GEMM 流水线

FC2 不是新逻辑, 而是同一 `MegaMoEScheduler` 在 `BlockPhase::Linear2` 下走一遍, **复用同一 MMA warp + 同一组 stage barrier**。每个 wave 内调度顺序: 先算完该 wave 所有 expert 的 L1 块, 再算所有 L2 块(`next_phase` 切换)。

| 角色 | FC1 (Linear1) | FC2 (Linear2) |
|---|---|---|
| act-load warp(4)等待点 | `while (ld_acq(l1_full_count) != ...)` | `while (ld_acq(l2_full_count) != ...)` |
| act 来源 | L1 ring(dispatch 拉来的 FP8 token) | L2 ring(FC1 epilogue 写的 FP8 中间激活) |
| weight-load warp(5) | L1 权重(FP4) | L2 权重(FP4) |
| MMA warp(6) | 同一 `tcgen05` UMMA | 同一 |
| epilogue | SwiGLU+量化 → 写 L2 ring + `l2_full↑` + `l1_empty↑` | **combine-scatter**(见下)+ `l2_empty↑` |

L2 ring 的 `l2_full`/`l2_empty` 把 FC1(生产者)和 FC2(消费者)以 BLOCK_M 粒度解耦 + 双向背压, 与 L1 ring 解耦 dispatch↔FC1 完全同构。FC1 epilogue 一产出 FP8 块 FC2 就能消费。

### 2. combine 第一半: FC2 epilogue 的 push-scatter(走 NVLink 写远端)

FC2 epilogue 不写本地输出, 而是把每行 GEMM 结果直接 push 回源 token 所在 rank:

```cpp
// Linear2 epilogue: TMEM → cast bf16 → STSM 进 smem → 写远端 combine buffer
const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
//                          ↑ dispatch 阶段写好的 (rank_idx, token_idx, topk_idx)
const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;   // ← PUSH 到源 rank combine buffer[topk][token]
```

- 用 dispatch 阶段写的 `TokenSrcMetadata` 知道这行回哪个 rank / 哪个原始 token / 哪个 topk slot。
- `sym_buffer.map(..., dst_rank_idx) = packed` 是 **fire-and-forget NVLink 写**, 不等返回。
- **fuse 了 "FC2 输出写回" 和 "all-to-all combine scatter"**, 省掉一次写 HBM 再读出来 scatter 的往返。
- 同时 `l2_empty↑` 释放 L2 ring 槽给 FC1。

### 3. combine 第二半: 本地 reduce(无 NVLink)

所有 rank FC2 push 完(一个 `nvlink_barrier(kBeforeCombineReduceBarrierTag)` 保证)后, 各 rank 本地归约自己收到的 kNumTopk 份 partial:

```cpp
for (token_idx = sm*kNumEpilogueWarps + epi_warp; token_idx < num_tokens;
     token_idx += kNumSMs*kNumEpilogueWarps) {
    读 topk_idx → mask(哪些 topk 有效)
    for chunk:
        do_reduce = move_mask_and_load(load_stage);     // TMA load 一份 partial(本地 combine buffer)
        while (do_reduce) {
            do_reduce = move_mask_and_load(load_stage^1);  // 预取下一份
            combine_load_barriers[load_stage]->wait();
            accumulate(reduced, ...);                      // fp32 寄存器累加
        }
        cast bf16 → st_shared → tma_store_1d 到最终 y
}
```

- reduce 输入 `combine_token_buffer` 是**本地**内存(刚被远端 push 填满)→ 归约阶段**无跨卡流量**。
- 双缓冲: `kNumChunkSlots = 3`(2 load stage + 1 store), `move_mask_and_load` 在累加当前份时预取下一份 → TMA load ↔ 寄存器累加 ↔ TMA store 软件流水。
- token 划分: 全局 epilogue-warp round-robin(起 `sm·kNumEpilogueWarps+epi_warp`, 步 `kNumSMs·kNumEpilogueWarps`), 1 warp 归约 1 个输出 token。

## 详细分析

### 时序图(FC1→FC2→combine 接力)

```
时间 →   ... wave W 算完 FC1 ...        wave W FC2          所有 rank FC2 push 完     combine reduce
         |--------------------|------------------------|----------------------|------------------|

FC1 EPI  [swiglu→写L2 ring, l2_full↑, l1_empty↑]
                    │ (L2 ring 第 k 块满)
                    ▼
ACT-LD4         (等 l2_full) [ld L2 blk ]...
                                 │
MMA6                          [mma FC2 ]...              ← 与 dispatch 拉 wave W+1、
                                 │                          FC1 算 wave W+1 在 MMA 流水里交错
FC2 EPI                       [TMEM→bf16→push 远端]...   ← NVLink 写分散在整个 GEMM 时间线上
                                 │ *sym_buffer.map(dst_rank)=packed (fire-and-forget)
                                 │ l2_empty↑ (释放 L2 ring 给 FC1)
                                 ▼
              ┌── nvlink_barrier: 等所有 rank FC2 push 完 ──┐
                                 ▼
DISP(clean)                                          [清 workspace 计数器]  ← 与 reduce 错峰重叠
COMBINE                                              [ld topk0│ld topk1│reduce│store y] 双缓冲流水
                                                       └─ 全本地, 无 NVLink ─┘

         └ FC2 的 NVLink 写(push)与后续 MMA 重叠; reduce 的本地 TMA 与 dispatch clean 重叠 ┘
```

### 为什么 FC2+combine 能高效 overlap

| # | 机制 | 说明 |
|---|---|---|
| 1 | **FC2 复用 FC1 的 GEMM 流水线** | 同一 MMA warp + stage barrier, FC2 MMA 在指令流水级与 FC1/dispatch 的后续工作交错; 无独立 kernel |
| 2 | **L2 ring 解耦 FC1↔FC2** | `l2_full`/`l2_empty` BLOCK_M 粒度握手 + 双向背压, FC1 一产出 FP8 块 FC2 就能算 |
| 3 | **combine-scatter 融进 FC2 epilogue + push** | FC2 输出直接 fire-and-forget 写远端, 省掉 HBM 往返; NVLink 写**分散在整个 GEMM 时间线**上, 与后续 MMA 重叠, 避免了 standalone combine kernel 的写突发 |
| 4 | **reduce 全本地** | combine 第二半只读本地 combine buffer, 无跨卡流量; 跨卡流量全在 FC2 epilogue 的 push 阶段已发生并被 MMA 藏掉 |
| 5 | **reduce 内部双缓冲** | 3 槽(2 load + 1 store)让 TMA load ↔ 寄存器累加 ↔ TMA store 软件流水 |
| 6 | **reduce 与 dispatch clean 错峰重叠** | `kDispatchWithEpilogueBarrierIdx` 让 dispatch 清 workspace 计数器与 combine reduce 并行 |

核心: **把 combine 拆成 "push-scatter(藏在 FC2 GEMM 后)+ 本地 reduce(藏在 dispatch clean 后)", 两段跨卡/本地流量分别被不同阶段的计算吸收掉。**

### dispatch 与 combine 的方向非对称(重点)

| 阶段 | DeepGEMM 方向 | 为什么这个方向自然 |
|---|---|---|
| dispatch | **receiver-pull**(TMA load remote) | 消费者(目标 expert 的 rank)知道要拉哪些 token, pull 自然; TMA 深队列藏 RTT |
| combine | **sender-push**(FC2 epilogue 写 remote)+ 本地 reduce | 生产者(expert 的 rank)知道结果该回哪(`TokenSrcMetadata`), push 自然; 融进 epilogue 后写突发被摊平 + 被 MMA 藏 |

rocmoe 现在 dispatch 和 combine **都用 pull**(combine-pull 为躲 FC2 出口 8-outbound 写竞争)。DeepGEMM 说明 **融合式 push 可能反而更快**——这是应实测推翻/确认的设计假设。

## ROCm / MI355X 移植建议

| 优先级 | 建议 | 为什么 |
|---|---|---|
| **P0(设计抉择)** | 重新评估 rocmoe pull-combine vs DeepGEMM 融合式 push-combine | rocmoe #2 把 combine 改 pull 是为躲 "FC2 出口 8 outbound XGMI 写竞争 ~0.7 ms"; 但 DeepGEMM 把 push 融进 FC2 epilogue、分散到整个 GEMM 时间线, 写突发被摊平 + 被 MMA 藏, 可能不再有那 0.7 ms。值得用 rocprof 量 "融合式 push" vs "pull" 而非先验排除 |
| **P0** | FC2 epilogue 用 `global_store` / `buffer_store` fire-and-forget 写 peer, 不等返回 | push 的好处是不吃 XGMI RTT(对照 dispatch pull ~43%/tok RTT 惩罚); AMD 上 peer write 是普通 global store 到 IPC 映射地址, 天然 fire-and-forget |
| **P1** | combine reduce 用 `global_load_lds` 双缓冲 + 寄存器归约 | DeepGEMM reduce 是本地 TMA + fp32 寄存器累加; AMD 对应 `global_load_lds` 多份在途 + `__reduce_add` / 寄存器累加(rocmoe M5 "atomic-free combine 寄存器归约" 方向一致), 避免 atomic combine |
| **P1** | L2 ring 加 `l2_empty` 回环(同 L1 ring 建议) | FC1↔FC2 同样需有界 ring + 背压, 复用 `grid_sync_v2` |
| **P2** | combine-reduce 与 dispatch workspace clean 错峰重叠 | 用 rocmoe 已有 phase barrier 排成并行(DeepGEMM `kDispatchWithEpilogueBarrierIdx` 等价) |
| **P2** | FC2 输出精度与下一段对齐 | DeepGEMM FC2 输出 bf16 直接 combine; rocmoe 若上 mxfp8 需在 FC2 epilogue 处理量化(M6) |

## 下一步 / 建议

- **P0 实验**: 在 rocmoe 上做 "融合式 push-combine"(FC2 epilogue 直接 `global_store` 写 peer combine buffer)vs 现有 pull-combine 的 A/B, 用 rocprof 量 XGMI 写突发是否真被 GEMM 时间线摊平 —— 这是 DeepGEMM 给出的最有价值的反例, 直接影响 rocmoe 设计核心 #2 的成立与否。
- combine reduce 移到 M5 时, 参考 DeepGEMM 的 3-槽双缓冲(2 load + 1 store)+ 本地 fp32 寄存器归约。
- 本研读为代码阅读, 无 perf 实测; DeepGEMM 在 MI355X 等价工况的基线需 H100/H800 上跑 `tests/test_mega_moe.py`。

## 相关文件

- 源码(已 clone 到 `/tmp/DeepGEMM`, 非持久): `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`(`Linear2` 分支 + combine 末尾段)
- 上游: `https://github.com/deepseek-ai/DeepGEMM`(PR #304 Mega MoE, #316 benchmarks)
- 前篇 dispatch→FC1 研读: [`2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md`](./2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md)
- rocmoe 架构设计(combine pull 抉择来源): [`2026-05-21_1252_rocmoe_v2_architecture_design.md`](./2026-05-21_1252_rocmoe_v2_architecture_design.md)
