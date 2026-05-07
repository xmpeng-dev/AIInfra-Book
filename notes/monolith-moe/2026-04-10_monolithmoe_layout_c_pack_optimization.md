# MonolithMoE: Layout C 重构 + Pack Phase 深度优化 — MI355X

> 日期: 2026-04-10 (Friday)
> 硬件: 8x AMD Instinct MI355X (gfx950), XGMI 全互联
> 软件: PyTorch 2.12.0.dev20260408+rocm7.1
> 配置: DeepSeek-V3 671B, 256E (EP8→32 local), top_k=8, H=7168, F=2048

---

## 1. 今日工作概览

延续 4/9 的 MonolithMoE super-kernel 骨架，完成三个迭代：

| # | Commit | 内容 | 改动量 |
|---|--------|------|--------|
| 1 | `b1c41b5` | IPC workspace 重构为 Layout C (src × expert 2D 排序) | +1358 / -197 |
| 2 | `986bbce` | Pack phase HIP kernel 深度优化 (OPT-1..7) | +672 / -65 |
| 3 | `fece2b1` | Multi-WG scatter (OPT-8): 所有 comm WG 协作 IPC 数据搬运 | +379 / -127 |

产出文件：
- `csrc/fused_moe_super_kernel.hip` — 核心 kernel 代码
- `3rd/lab/notes/monolith-moe/2026-04-14_data_layout_analysis.md` — 5 种 IPC buffer layout (A-E) 对比分析
- `docs/pack_phase_baseline_mi355x.txt` — Layout C baseline 性能数据
- `docs/pack_phase_optimized_mi355x.txt` — 优化后性能数据 + 理论分析
- `benchmarks/pack_phase_bench.py` — Pack phase micro-benchmark

---

## 2. Layout C: Source × Expert 二维排序

### 2.1 动机

4/9 的 Layout A（按 source GPU 分 chunk，expert 不排序）有两个致命问题：
- 接收端每个 chunk 需要 scan + gather 按 expert 重排 token → 增加 ~2.4ms 到 critical path
- 8-stage pipeline → fill+drain 开销 25%

Layout C 在发送端做 (dest_gpu, local_expert) 二维排序，token 到达接收端时已按 expert 预排好，实现 **zero-gather compute**。

### 2.2 5 种 Layout 对比结论

| Layout | Pipeline stages | Gather 开销 | XGMI 效率 | 跨 GPU 原子操作 | 预估端到端 |
|--------|:-:|:-:|:-:|:-:|:-:|
| A (Source-Chunked) | 8 | **2.4ms** | ~95% | 无 | ~9.85ms (比不 overlap 还慢) |
| B (Expert-First) | 32 | 0 | ~40% | **有** | ~5.10ms |
| **C (Src×Expert)** | **8 → 256** | **0** | **~95%** | **无** | **~4.83ms** |
| D (Tile-Aligned) | ~512 | 0 | ~75% | 有 | ~4.96ms |
| E (Expert+Zone) | 256 | 0 | ~35% | 无 | ~5.50ms |

**选择 Layout C + sub-chunk signaling**：8 source × 32 expert = 256 个 `expert_ready` flag，fill+drain < 1%，XGMI 写仍然是大块连续写（~7MB/pair），无跨 GPU 原子竞争。

### 2.3 关键数据结构变化

**Pack workspace 简化** — 从 3 个数组缩减为 1 个：
```
Before: pack_offsets[NUM_GPUS+1], pack_token_ids[T*K], pack_expert_ids[T*K]
After:  pack_perm[T*K]  (sorted pair index → original pair index)
```

**IPC workspace** — 从 per-source flat 改为 per-(source, expert) 分段：
```
dispatch_expert_offsets[NUM_GPUS * (epg+1)]   // 每个 source 的 expert prefix sum
dispatch_expert_ready[NUM_GPUS * epg]          // 256 个 per-(src, expert) flag
```

**Combine phase** — 增加 fp32 accumulator，避免 bf16 精度损失：
```
gather_combine_phase(args, output_accum)  // float* output_accum [T_local, H]
```

---

## 3. Pack Phase 深度优化 (OPT-1 ~ OPT-8)

### 3.1 Bottleneck 分析

Baseline 性能 (PyTorch emulation, 4096 tokens):

| 步骤 | 耗时 | 占比 |
|------|------|------|
| Sort (histogram + prefix sum + counting sort) | 0.112ms | 24% |
| Scatter (token data gather + IPC write) | 0.333ms | 73% |
| 端到端 | 0.458ms | 100% |

关键发现：HIP kernel 中 scatter 内循环的 **整数除法是最大瓶颈**：

```
for (int i = threadIdx.x; i < e_count * H; i += WG_SIZE)
    int slot = i / H;    // ← H=7168, 非 2 的幂, ~40 cycles!
    int h    = i % H;    // ← ~40 cycles!
```

32K pairs × 7168 elements/pair = 234.9M 个元素，每个元素 ~80 cycles 除法 → **仅除法就耗 ~4.4ms**，远超实际内存拷贝时间。

### 3.2 优化清单

| OPT | 描述 | 影响 |
|-----|------|------|
| OPT-1 | bucket = expert_id (恒等变换，消除 2 次 div/mod by epg) | ★★★ |
| OPT-2 | K bit-shift (K=8 → >>3, &7) | ★★ |
| **OPT-3** | **2D loop (slot × h) 消除 div by H=7168** | **★★★★★ (关键)** |
| OPT-4 | Wave-parallel scatter (4 waves × 独立行) | ★★★ |
| OPT-5 | 128-bit vectorized loads/stores (uint4 = 8 bf16) | ★★★★ |
| OPT-6 | `__restrict__` + const compiler hints | ★ |
| OPT-7 | Fused metadata write (lane 0 写 metadata，其他 lane 拷数据) | ★★ |
| **OPT-8** | **Multi-WG scatter: 所有 C 个 comm WG 协作搬运** | **★★★★** |

### 3.3 OPT-3 核心变化 (消除除法)

```
// BEFORE: 线性化循环，每个元素做除法
for (int i = threadIdx.x; i < e_count * H; i += WG_SIZE) {
    int slot = i / H;    // ~40 cycles
    int h    = i % H;    // ~40 cycles
    ...
}

// AFTER: 2D 循环，零除法
for (int s0 = 0; s0 < e_cnt; s0 += WAVES_PER_WG) {
    int my_slot = s0 + wave_id;
    int token_id = pair_idx >> k_log2;    // bit-shift
    const uint4* src = ...;               // 128-bit vector
    for (int v = lane_id; v < vec_per_row; v += WARP_SZ)
        dst_vec[v] = src_vec[v];          // 14 iters, vectorized
}
```

### 3.4 OPT-8: Multi-WG Scatter

将 `pack_and_scatter_phase` 拆分为两个子 phase：

| 子 phase | 执行者 | 作用 |
|----------|--------|------|
| `pack_sort_phase` | WG 0 only (LDS) | 2D histogram + prefix sum + counting sort |
| `multi_wg_scatter_phase` | 所有 C 个 comm WG | Round-robin 256 个 (dest, expert) 对 |

WG 0 完成排序后写 `pack_offsets` 到 global memory，设置 `scatter_ready` flag。
其他 comm WG polling 该 flag 后并行搬运数据，C 倍的 CU 并行度。

### 3.5 理论加速预估

| 组件 | Baseline 预估 | 优化后预估 | 加速 |
|------|:---:|:---:|:---:|
| 除法开销 (i/H, i%H) | ~4.4ms | 0 | ∞ |
| HBM scatter reads | ~0.33ms | ~0.26ms (vec) | 1.3× |
| HBM writes | ~0.18ms | ~0.18ms | 1.0× |
| Sort | ~0.11ms | ~0.11ms | 1.0× |
| **端到端 pack phase** | **~5.0ms** | **~0.51ms** | **~10×** |

实际 HIP kernel 加速需 `rocprof` 验证（除法延迟可能被 memory latency 部分隐藏）。

---

## 4. PyTorch Benchmark 结果 (MI355X)

PyTorch 已使用优化的 HIP kernel，因此 PyTorch 级别的 emulation 无法观测 OPT-1~8 的收益（PyTorch roofline 就是优化后的下限）：

### 4.1 Pack Phase Baseline (Layout C)

| tokens | pairs | sort(ms) | scatter(ms) | e2e(ms) | scatter BW | 占 MoE 比例 |
|--------|-------|----------|-------------|---------|------------|-----------|
| 1024 | 8192 | 0.097 | 0.075 | 0.178 | 1564 GB/s | 2.1% |
| 2048 | 16384 | 0.099 | 0.178 | 0.286 | 1322 GB/s | 3.4% |
| **4096** | **32768** | **0.112** | **0.333** | **0.458** | **1411 GB/s** | **5.4%** |
| 8192 | 65536 | 0.122 | 0.659 | 0.797 | 1426 GB/s | 9.4% |
| 16384 | 131072 | 0.145 | 1.385 | 1.533 | 1358 GB/s | 18.1% |

核心特征：
- Sort 几乎恒定 (~0.1ms)，kernel launch overhead 主导
- Scatter 线性增长，bandwidth-bound at ~1400 GB/s (MI355X peak 5300 GB/s 的 26%)
- 4096 tokens 时 pack phase 仅占整个 MoE 的 5.4% (~8.47ms)

### 4.2 Optimization Pattern 验证

| Pattern | 耗时 | 说明 |
|---------|------|------|
| OPT-1: old bucket (2×div) | 0.027ms | |
| OPT-1: new bucket (identity) | 0.005ms | **5.4× 加速** |
| OPT-3+5: gather (random rows) | 0.329ms | ~2856 GB/s |
| OPT-3+5: contiguous copy | 0.183ms | ~5134 GB/s (97% of peak) |
| gather_overhead | 1.80× | scattered reads 的硬性下限 |

---

## 5. 8-GPU MoE 端到端参考数据

| 组件 | 耗时 | 类别 |
|------|------|------|
| prep (gate GEMM) | 0.035ms | Overhead |
| dispatch | 1.427ms | A2A |
| sort+index | 0.116ms | Overhead |
| fc1 | 2.727ms | GEMM |
| act (SwiGLU) | 0.161ms | Overhead |
| fc2 | 1.719ms | GEMM |
| combine | 2.242ms | A2A |
| topk_sum | 0.038ms | Overhead |
| **TOTAL** | **8.466ms** | |
| A2A | 3.669ms | **43.3%** |
| GEMM | 4.446ms | **52.5%** |

GEMM (4.45ms) > A2A (3.67ms) → **100% A2A 理论上可隐藏**。
Layout C + 256-stage pipeline 预估端到端 **~4.83ms (1.75× speedup)**。

---

## 6. 资源分析 (MI355X CDNA4)

| 资源 | 使用量 | 限制 | 状态 |
|------|--------|------|------|
| LDS (pack phase) | ~6 KB | 128 KB/CU | ✅ 充裕 |
| LDS (GEMM WG) | ~14 KB | 128 KB/CU | ✅ 4 WG/CU |
| VGPR (pack) | ~20 | 256/thread | ✅ 充裕 |
| VGPR (GEMM) | ~44 | 256/thread | ✅ < 96 目标 |
| Occupancy | 4 WGs/CU | 8 WGs/CU | ✅ 足够隐藏 latency |
| 瓶颈 | HBM scattered reads | 5.3 TB/s peak | ~1.4 TB/s (26%) |

---

## 7. 剩余工作 & Next Steps

### 已完成
- [x] Layout C IPC workspace 重构 (dispatch + combine)
- [x] Pack phase 2D counting sort (histogram + prefix sum + pack_perm)
- [x] Per-(src, expert) sub-chunk signaling (256 flags)
- [x] OPT-1~7: 消除除法、bit-shift、2D loop、wave-parallel、128-bit vec、restrict、fused metadata
- [x] OPT-8: Multi-WG scatter (pack_sort + multi_wg_scatter 拆分)
- [x] fp32 combine accumulator
- [x] Pack phase benchmark + MI355X baseline profiling

### 待做 (Phase 2 性能)
- [ ] **编译 + rocprof 实测**: 验证 OPT-1~8 的实际 HIP kernel 加速
- [ ] M_TILE 自适应: expert token 少时用 M_TILE=32/16
- [ ] Work stealing: compute WG 间 atomic tile counter
- [ ] Combine 阶段 bf16 atomicAdd (MI355X 原生支持)
- [ ] 功能正确性验证 (对比 PyTorch reference MoE)

---

## 附录: Commit 详情

```
fece2b1  2026-04-10 02:16  Implement multi-WG scatter (OPT-8)
986bbce  2026-04-10 01:32  Optimize pack_and_scatter_phase (OPT-1~7)
b1c41b5  2026-04-10 01:18  Refactor IPC workspace to Layout C
3eeb40d  2026-04-09 23:53  Add 8-GPU support and MI355X baseline results
4f3837f  2026-04-09 23:47  Add MoE component-level performance decomposition benchmark
```
