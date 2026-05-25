# MoE forward 算子级分解（8466 μs 基准）

**日期**: 2026-05-14 12:10 (UTC+8)  
**归档**: 与 README 中 **PyTorch+RCCL baseline 8.466 ms / layer** 口径对齐的 **operator 级 us 表**（总时长 **8466 μs**）。

## 背景 / 目标

- 将 **prep / dispatch / sort / fc1 / act / fc2 / combine / topk sum** 拆成 **overhead / comm / compute** 三类，便于与 super-kernel profile bucket（`dispatch_wait`、`fc1_tiles`、`fc2_tiles`、`scatter` 等）对照。
- 百分比以 **8466 μs** 为分母（与 `8.466 ms` 一致）。

## 原始分解表

| Operator | group | us | % of 8466 us |
|----------|-------|-----|----------------|
| prep · gate Linear + softmax | overhead | 35 | 0.4 % |
| dispatch · all_to_all | comm | 1427 | 16.9 % |
| sort + indexing | overhead | 116 | 1.4 % |
| fc1 · grouped GEMM (×256 experts) | compute | 2727 | 32.2 % |
| act · SwiGLU | compute | 161 | 1.9 % |
| fc2 · grouped GEMM (×256 experts) | compute | 1719 | 20.3 % |
| combine · all_to_all + top-k sum | comm | 2242 | 26.5 % |
| topk weighted sum | overhead | 38 | 0.4 % |

**合计**: 35+1427+116+2727+161+1719+2242+38 = **8465 μs**（与 8466 差 1 μs，四舍五入/计时边界）。

## 按 group 汇总

| group | 包含项 | Σ us | % of 8466 |
|-------|--------|------|-----------|
| **overhead** | prep + sort + topk weighted sum | 189 | **2.2 %** |
| **comm** | dispatch all_to_all + combine（表中 combine 行含 a2a + top-k sum 的 comm 部分，与脚本拆分口径可能略有重叠，以原表为准） | 3669 | **43.3 %** |
| **compute** | fc1 + SwiGLU + fc2 | 4607 | **54.5 %** |

说明：若将 **「combine」行** 严格只算作 **RCCL all_to_all**，则 **top-k sum** 更接近 **本地 compute**；上表沿用用户给出的 **group 标签** 做归档。

## 与 `bench_pytorch_rccl_dsv3.py` 粗对照

仓库内微基准（`T_src=8192`，rank-0 median，mi355 一次采样）大致为：

- **RCCL `all_to_all` 合计 ~5.4 ms / iter**（counts + x + eid + combine）→ **comm ~29% wall**（~18.7 ms wall）。
- **本地 prep + GEMM + post_combine ~13.1 ms** → **comp ~70% wall**。

与上表 **43% comm / 54% compute** 的差异来自：** workload（512 vs 8192 tokens）、是否含 gate、profiler 归属（combine 行是否含 top-k sum）** 等；本 note **以表中 8466 μs 分解为归档真值**，微基准仅作交叉参考。

## 相关文件

- `benchmarks/bench_pytorch_rccl_dsv3.py` — comm/comp 细分（`a2a_*` / `prep_ms` / `post_combine_ms`）
- `slab/notes/monolith-moe/README.md` — 状态表里 **8.466 ms vs super-kernel**  headline 出处

## 下一步

- 若需 **与 super-kernel 同一列命名**，可把 `dispatch_src_ready_wait` / `fc1_tiles` / `scatter` 等 bucket **映射到上表 Operator 行** 做一张对照表（另开 note 或附录）。
