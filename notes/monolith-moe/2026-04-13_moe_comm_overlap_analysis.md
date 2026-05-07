# MoE 通信-计算 Overlap 可行性分析总结

## 目标

验证在 `topk=8, ep=8` 配置下，MoE 通信（dispatch + combine All-to-All）与 CK GEMM 计算能否通过 stream-level overlap 获得加速。

## 测试环境

- **硬件**: 8× MI355X (gfx950)，单节点
- **软件**: ROCm 7.2, RCCL, primus_turbo (CK)
- **容器**: `xiaoming-dev-fix` (rocm/primus:v26.2)

---

## 1. 通信量分析

### 1.1 模型配置 (DeepSeek-V3)

| 参数 | 值 |
|------|-----|
| tokens_per_mbs | 8192 |
| topk | 8 |
| ep | 8 |
| hidden_size | 7168 |

### 1.2 通信量对比

| 场景 | 通信量/rank | 说明 |
|------|------------|------|
| topk=1, 单次 A2A | 117.4 MB | 微基准对照 |
| topk=8, dispatch | 939.5 MB | token → expert |
| topk=8, combine | 939.5 MB | expert → token |
| **topk=8, 总计** | **1,879 MB** | dispatch + combine |

---

## 2. 基准性能实测

### 2.1 topk=1 单次 A2A 基线

| N_tiles | Data/rank (MB) | 时间 (ms) | 带宽 (GB/s) |
|---------|----------------|-----------|-------------|
| 1 | 117.4 | 0.353 | 333 |
| 2 | 58.7 | 0.193 | 305 |
| 4 | 29.4 | 0.104 | 281 |

### 2.2 topk=8 真实流程 (dispatch + combine)

`torchrun --nproc_per_node=8 benchmarks/bench_ck_overlap_real.py`

| 组件 | 时间 (ms) |
|------|-----------|
| Comm (Dispatch+Combine) | **5.460** |
| Compute (CK GEMM) | **0.348** |
| Baseline (Serial) | **5.566** |

**关键发现**: 通信时延是计算的 **15.7x**，通信绝对主导。

---

## 3. Overlap 实测结果

### 3.1 async_op + work.wait() 方案

测试脚本: `benchmarks/bench_ck_overlap_real.py`

流程:
1. `dispatch_async()` → `work.wait()` → GEMM → `combine_async()`
2. 下一 tile 的 dispatch 在当前 tile wait 前发起

| N_tiles | Overlap 时间 (ms) | vs Baseline |
|---------|------------------|-------------|
| 2 | 6.016 | **-7%** |
| 4 | 6.256 | **-11%** |

### 3.2 Stream-based 方案

测试脚本: `benchmarks/bench_ck_stream_overlap.py`

| N_tiles | Simple (ms) | Pipeline (ms) | Best Speedup |
|---------|------------|---------------|--------------|
| 2 | 0.729 | 0.769 | +5% |
| 4 | 1.346 | 1.036 | -26% |
| 8 | 1.876 | 1.598 | -52% |

> 注: 2 GPU 时有 +37% 收益，8 GPU 时收益消失或负收益。

---

## 4. 原因分析

### 4.1 为什么 Overlap 没起来？

1. **通信主导，计算占比太小**
   - Comm: 5.46 ms (94%)
   - Compute: 0.35 ms (6%)
   - 理论最优 overlap 也只能省 ~0.35 ms（6%）

2. **`work.wait()` 是 host 侧阻塞**
   - 每个 tile 都在 host 上 wait，打碎调度节奏
   - 引入 CPU-side 同步开销（约 0.3-0.4 ms/iter）

3. **RCCL 多 stream 并不总能与 GEMM 真并行**
   - 通信放到独立流并没有线性重叠
   - RCCL 内核/调度和 GEMM 在 CU/带宽/队列层面存在竞争

4. **Tile 化固有开销**
   - N 增加 → collective 次数增加 + kernel launch 增加
   - 小 shape 导致 GEMM 效率下降

### 4.2 数据量对比验证

| 配置 | 预期时延 | 实测时延 | 匹配度 |
|------|---------|---------|--------|
| topk=1, 单次 A2A | 0.35 ms | 0.353 ms | ✓ |
| topk=8, dispatch+combine | 0.35×8×2 = 5.6 ms | 5.46 ms | ✓ |

**结论**: 通信时延主要来自数据量放大，RCCL 本身没有退化。

---

## 5. 结论

### 5.1 单节点 topk=8 + ep=8 场景

- **Overlap 收益有限或负收益**
- 原因: 通信绝对主导（94%），计算窗口太小
- 同步/调度开销吃掉了理论收益

### 5.2 何时 Overlap 有效？

| 条件 | 当前状态 | 需要改善方向 |
|------|---------|-------------|
| Comm ≈ Compute | Comm >> Compute | 跨节点场景 / 更大计算量 |
| 低同步开销 | host wait 开销大 | persistent kernel / device-side sync |
| 高带宽效率 | 小 tile 带宽下降 | 减少 tile 数 / 更大 tile |

### 5.3 推荐方向

1. **跨节点场景**（IB HDR）: Comm 时延会更长，overlap 窗口更大
2. **Persistent Kernel**: 用 device-side flag/signal 替代 host wait
3. **减少通信**: 考虑 All-Reduce 替代 All-to-All、或 routing 优化
4. **增大计算**: FC1+FC2 fused、或 batch 增大

---

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `benchmarks/bench_ck_overlap_real.py` | topk=8 真实通信 + CK overlap 测试 |
| `benchmarks/bench_ck_stream_overlap.py` | stream-based overlap 测试 |
| `3rd/lab/notes/monolith-moe/2026-04-14_tile_overlap_analysis.md` | 完整分析文档 |
