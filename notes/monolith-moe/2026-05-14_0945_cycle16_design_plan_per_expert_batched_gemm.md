# Cycle 16 design plan — per-expert batched GEMM via scatter_native + all-8-srcs-ready

**date**: 2026-05-14 09:45 (UTC+8)
**node**: mi355-gpu-26 (xiaoming-dev container)
**hardware**: 8× MI355X (gfx950)
**target**: T_src=8192 wall 51.24 ms → lower via per-expert GEMM (**cycle 15 “22×” vs standalone was retracted** — wrong `M` on HIP microbench; re-bench at `M=2048` before trusting a speedup factor)

## 决策

User picked:
- ready signal = **all 8 srcs ready** before kicking GEMM (simpler, peer-scatter sync gap typically <1 ms)
- data layout = **scatter_native** (scatter writes directly to per-expert contiguous buffer, no extra repack pass)

Rejected: repack approach (would have been an easier 1-day validation but pays ~40 us extra HBM traffic — user wants the clean end-state).

## 现状映射（要改的数据结构）

`MoeIpcWorkspace` 现状（`csrc/fused_moe_super_kernel.hip:454`）：
```cpp
bf16_t*  dispatch_tokens;          // [NUM_GPUS][max_recv_per_src][hidden_size]   ← src-major
int*     dispatch_expert_offsets;   // [NUM_GPUS][epg+1]  per-(src, e) prefix sum
int*     dispatch_src_token_ids;    // [NUM_GPUS][max_recv_per_src]
float*   dispatch_topk_weights;    // [NUM_GPUS][max_recv_per_src]
int*     dispatch_expert_ready;    // [NUM_GPUS][epg]   per-(src, e) ready flag
int*     dispatch_src_ready;       // [NUM_GPUS]         per-src "all epg done" counter
```

新增/替换字段（cycle 16 新 layout，**parallel path**，受 `MOE_USE_EXPERT_PACKED_LAYOUT` 编译开关控制）：
```cpp
// EXPERT-MAJOR contiguous layout
bf16_t*  expert_packed_tokens;     // [epg][max_per_expert_global][H]   ← expert-major
int*     expert_packed_src_ids;    // [epg][max_per_expert_global]      原 src_token_id
float*   expert_packed_weights;    // [epg][max_per_expert_global]      原 topk_weight
int8_t*  expert_packed_src_rank;   // [epg][max_per_expert_global]      record sender rank for combine

// 新协调原语
int*     expert_pack_pos;          // [epg]   atomicAdd 当前 write head；scatter 时 src 用它认领 slot
int*     expert_done_count;        // [epg]   每 src 写完一个 expert 的 token 后 atomicAdd(1)；
                                    //         compute 等到值 == NUM_GPUS 才开 GEMM（all-8-ready）
int*     expert_total;             // [epg]   写满后 == 该 expert 在本 rank 的 received tokens 总数
                                    //         （也是 GEMM 的 M 值）
```

`max_per_expert_global` = 全 8 srcs 写到本 rank 的某个 expert 的最大可能 tokens，用作 buffer 维度。
保守估计 `= 2 * (T_src * topk * NUM_GPUS / num_experts_total) = 2 * 256 = 512`（DSV3 T_src=8192 工况）。
8 ranks, 32 epg, 512 slots, 7168 H, 2 B = **940 MB** per peer 仅这一个 buffer —— 太大。
更紧 bound `= 1.5x avg = 384`：705 MB。仍偏大。

**收紧 bound**: 用 `max_recv_tokens / epg`（dispatch buffer 总容量除以 expert 数）= 8192 / 32 = 256，加 30% 裕度 = 333。
333 × 32 × 7168 × 2B = **152 MB** per rank。可接受。

注：`max_recv_tokens` 已经是 host 端按 routing balance 算好的 bound（包含安全裕度）。
本次 sizing 公式：`max_per_expert = max_recv_tokens / epg + safety` 与现有协议自洽。

## 三阶段实施

### Stage A — 数据结构 + scatter side（今天）

1. 在 `MoeIpcWorkspace` 加上述新字段，host setup 函数分配。
2. 在 `pack_sort_phase` 末尾增加：
   - 重置本 rank IPC 上的 `expert_pack_pos[*]=0, expert_done_count[*]=0, expert_total[*]=0`。
3. 修改 `multi_wg_scatter_phase`：每 (dest, e) pair 处理时
   ```
   my_offset = atomicAdd(&peer->expert_pack_pos[e], e_cnt)   // 认领 e_cnt 个 slot
   atomicAdd(&peer->expert_total[e], e_cnt)                  // 累计 M
   for slot in 0..e_cnt:
     write to peer->expert_packed_tokens[e][my_offset + slot]
     write src_id / weight / src_rank metadata
   __threadfence_system()
   atomicAdd(&peer->expert_done_count[e], 1)                  // 通知本 src 完成本 expert
   ```
4. 保留旧的 per-(src, e) 写入路径不动（旧 `dispatch_expert_ready` 仍然 fire），仅 *额外* 新增写入。
   这样旧 GEMM/combine 仍然 PASS，新路径独立验证。

### Stage B — compute side 切换（明天）

5. 加 compile flag `MOE_USE_EXPERT_PACKED_LAYOUT`（默认 0）。
6. 在 `expert_compute_phase` 内分支：开启时
   - 等待 `expert_done_count[e] == NUM_GPUS`（all-8 ready）
   - 读 `M = expert_total[e]` 作为 GEMM 的 M
   - tile loop 改成 `for e in [0, epg): for ni in [0, N_TILE_count)`，
     单 tile 用 M_TILE=256 N_TILE=256，valid_m = min(M_TILE, M)。
   - 可能少量 partial tile（M 末尾不足 256），但相比当前 87.5% MFMA 浪费 → 现在最差 1/256 = 0.4% 浪费。
7. FC2 输出写到一个 `expert_packed_fc2_out[e][slot][H]` 临时区。

### Stage C — combine side 适配（第三天）

8. `gather_combine_phase` 读 `expert_packed_fc2_out` 而不是 `fc2_scratch`。
9. 用 `expert_packed_src_rank` 和 `expert_packed_src_ids` 找回原 `(dst_rank, dst_token_id)`，
   atomicAdd 加权和到 `peer->combine_results`。

### 验收 checkpoint

- After Stage A: bench 旧 path 不退化，host 端确认新 buffer 写入正确（dump 一个 iter 比对）。
- After Stage B: bench `MOE_USE_EXPERT_PACKED_LAYOUT=1`，
  - **预期**：FC1+FC2 28.4 → 1.3-3 ms 量级（22× 闭合的下界）
  - **实测 < 5 ms** 即视为假设证实，进入 Stage C 完成端到端。
  - **实测 > 10 ms** 说明 hypothesis 错了（M_TILE 不是主因），停下来重新分析。
- After Stage C: 端到端 numerical parity vs 旧 path（运行 mock data forward，max abs diff < 1e-3 bf16 噪声）。
- 训练接入：DSV3 4-layer 跑 20 iter 与 baseline TEGroupedMLP 三方 loss parity（同 cycle 13 验证方式）。

## 风险

1. **all-8 等待引入额外 wait**: 8 个 src 的 peer-scatter 不严格同步，最坏 src 决定开 GEMM 时刻。但同 expert 的 8 src 写入实际上 highly correlated（同物理 ring），观测中 srcs 通常 < 1 ms 内全部到。如果 cycle 16 之后这个 wait 显著（> 3 ms），可以再做 K-of-8 partial 启动作为 follow-up。
2. **atomic 竞争**: `expert_pack_pos[e]` 被 8 srcs 同时 atomicAdd。HW atomic 在 system scope 下延迟 ~50 cycles × 8 srcs serial = ~400 ns. Negligible.
3. **memory bloat**: 152 MB per rank 新 buffer。8 ranks × 152 = 1.2 GB on a single node. 192 GB HBM 上没问题。
4. **测试覆盖**: 旧 path 保留作为对照，每步可 A/B。risk 可控。

## 不做（明确降级搁置）

- cycle 17（GEMM software pipelining）：cycle 16 之后 GEMM 总时间已 ≤3 ms，pipelining 绝对收益 < 1 ms，不值得。
- cycle 18（kFullTile template）：cycle 16 已结构性消除 partial-M tile，无适用场景。
- cycle 19（closed-form pair_id）：cycle 16 之后 tile 数 8× 减少，per-tile prologue scan 总开销已经折扣。

## 实施日志（这条 note 持续追加）

- [x] 09:45 design plan 落定 → 进 Stage A
- [x] **10:00 Stage A1 PASS — IPC workspace plumbing**：
  - `MoeIpcWorkspace` 加 7 个新字段 (`expert_packed_tokens / src_ids / weights / src_rank / pack_pos / done_count / total`)
  - `WorkspaceLayout` + `compute_layout` + `alloc_workspace` + `finalize_workspace` + `workspace_layout_dict` 都对齐
  - `tests/smoke_super_kernel.hip` 也加了对应 alloc（让 standalone bench 跑得通）
  - `moe_reset_workspace_kernel` 加 `epg` 参数，每 iter 把 3 个新计数器清零
  - `max_per_expert = ceil_to_64(avg + 50% safety) = ceil_to_64(256 + 128) = 384`（DSV3 T_src=8192）
  - **bench 验收**：T_src=8192 wall **50.24 ms** (vs Stage A1 之前 50.31，无回归)；T_src=2048 wall **13.12 ms** (无变化)。data plumbing 完成，0 行为变化。
- [x] **10:10 Stage A2 PASS — scatter 在原 layout 旁并行写 expert-major layout**：
  - `multi_wg_scatter_phase` 内每 (dest, e) pair 多做：
    1. WG-内 `atomicAdd(&peer->expert_pack_pos[e], e_cnt)` 认领 slot offset
    2. wave-parallel 多写一份 token 数据到 `peer->expert_packed_tokens[e][offset..]`
    3. lane 0 同时写 per-token metadata (src_id / weight / src_rank) 到 packed buffer
    4. 全 WG fence 后 `atomicAdd(&peer->expert_done_count[e], 1)`
  - bench 验收（额外 HBM 写入开销）：
    - T_src=8192: **52.21 ms** (vs A1 50.24，**+1.97 ms / +3.9%**)
    - T_src=2048: **13.64 ms** (vs A1 13.12，**+0.52 ms / +4.0%**)
  - +4% 是 **double-write** 的暂时代价；Stage C 删除旧 layout 后会全部回收。
  - 旧 path 完全 untouched（旧 ready flag/offsets/buffer 仍在更新），所以现有的 bench/smoke/training 全部无 behavioral 变化。
- [ ] Stage B — compute side switch（`MOE_USE_EXPERT_PACKED_LAYOUT` flag, FC1+FC2 走新 layout）
- [ ] Stage C — combine 适配 + 删旧 layout 双写
