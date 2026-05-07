# GPT-OSS-20B / MI355X — Turbo grouped MLP 反慢 -7% 的 trace 实证

**日期**：2026-04-25
**仓库**：`/home/xiaompen/mlperf-training/small_llm_moe_pretraining/primus`
**状态**：复核 note 20 的"Turbo grouped MLP 在 EP=1 反慢"结论。本 note
补充 Kineto trace 级别的 per-stream / per-category 证据，明确 root cause
是 **stream serialization**（不是 kernel 算力）。

## 1. 背景

note 20 修了 `use_turbo_grouped_mlp` 的 string-truthy bug，并通过 200-iter
实跑确认 yaml 设 `false` 后 step 更快（A 1233 ms → 1182 ms）。但当时只
有标量 step time，没有 trace 级数据解释为什么 Turbo 会慢。

本次跑了一组三段对比 trace，把 stream-level 行为彻底锁死，避免后续再
被 "Turbo kernel 看起来更猛、想再开一次" 的直觉骗到。

## 2. 实验设置（自包含可复现）

```
small_llm_moe_pretraining/primus/run-trace/20260425_b_v1fused/
  A_fuse0/        PRIMUS_FUSED_RESIDUAL_NORM=0  use_turbo_grouped_mlp=false
  B_fuse1/        PRIMUS_FUSED_RESIDUAL_NORM=1  use_turbo_grouped_mlp=false
  C_turbo_grouped/PRIMUS_FUSED_RESIDUAL_NORM=1  use_turbo_grouped_mlp=true   ← 本次新增
```

24 train iter，profile steps 16..19 on rank 2，GBS=32 MBS=4，TP1 PP1 EP1，
seq=8192，FP8 hybrid。每个目录下 `run.sh` + `*.yaml` + `*.sh` 配齐，
单一 `bash run.sh` 即可重现。

## 3. 三段 trace 对比

| | A（baseline） | B（V1 fused） | C（V1 fused + Turbo grouped） |
|---|---|---|---|
| ProfilerStep#17 wall | 1127.35 ms | 1125.12 ms | **1232.42 ms** |
| Δ vs B | +0.20% | — | **+9.5%** |
| iter 20 训练 ms | 1188.4 | 1186.5 | **1276.2** |
| iter 20 TFLOPS | 694.9 | 696.0 | **647.1** |
| HIP mem | 244 GiB | 244 GiB | **251 GiB (+3%)** |
| `(experts):` 类型 | GroupedMLP | GroupedMLP | **PrimusTurboGroupedMLP** ✓ |
| 数值（loss/grad） | — | 一致 | 与 B 偏差 ~0.03%（FP8 噪声） |

C 的 Turbo 路径正确激活（log: `(experts): PrimusTurboGroupedMLP()`），
数值正确，但 wall +9.5% / TFLOPS -7%。

## 4. 根因：stream-level 行为

### 4.1 Per-stream busy（最关键）

| stream | B（GroupedMLP） | C（PrimusTurboGroupedMLP） |
|---|---|---|
| stream 0（主） | **600.24 ms (53.3%)** | **1181.43 ms (95.9%)** ← 几乎打满 |
| stream 13（gemm 副） | 209.62 ms | — |
| stream 14（gemm 副） | 224.92 ms | — |
| stream 15（gemm 副） | 186.59 ms | — |
| stream 16（gemm 副） | 200.93 ms | — |
| 4 副 stream 合计 | **822 ms work / ~250 ms wall** | **0 ms** |
| stream 11（NCCL） | 234 ms | 272 ms |
| stream 4（h2d） | 56 ms | 56 ms |

→ B 的 legacy `GroupedMLP` 把 32 experts 拆成 N 个 cuBLAS GEMM，**散到 4
个副 stream 并行跑**（4 lane × ~205 ms = 822 ms work，wall 不到 250 ms）。
C 的 `PrimusTurboGroupedMLP` 用 `_grouped_bf16_persistent_gemm_kernel` 一
个 kernel 把全部 expert 算完，**只能在 stream 0 上串行**。

**stream 0 从 53% 占用直接打到 96% 占用**，其余 stream 几乎空转 —— 这是
wall +107 ms 的全部解释。

### 4.2 Per-category：算力其实更高，wall 反而更慢

| category | B ms | C ms | Δ |
|---|---|---|---|
| `gemm` | 947.93 | 128.03 | **-819.9**（被 grouped_gemm 取代） |
| **`grouped_gemm`** | **0** | **569.57** | **+569.6**（新出现） |
| **gemm + grouped_gemm 合计** | **947.93** | **697.60** | **-250.3** |
| `nccl_generic` | 125.51 | 167.51 | +42（overlap 变差） |
| `attn_kernel` | 202.21 | 210.66 | +8.5 |
| `elementwise` | 158.14 | 158.27 | +0.1 |
| `norm` | 41.09 | 40.96 | -0.1 |
| 其余 | 几乎不变 | | |

→ Turbo persistent kernel **真的省了 250 ms 算力**（少了 fused vs N×cuBLAS
的 launch/scheduler 重复开销），但因为同时**抹掉了 4-stream 并行结构**，
省下来的 work 从 4 个 lane 折回 1 个 lane，wall 反而 +107 ms。

NCCL overlap 也微跌：B 80.8% hidden → C 82.5% hidden（看起来更高，但实际
nccl_generic 总量从 125 涨到 167 ms，绝对暴露在 critical path 上的部分
变多了 6 ms）。

### 4.3 Top-3 kernel in C 解释一切

```
375.35 ms  (30.5% of step)  _grouped_bf16_persistent_gemm_kernel
274.73 ms  (22.3% of step)  ncclDevKernel_Generic_1
194.22 ms  (15.8% of step)  _grouped_variable_k_gemm_kernel
```

forward + backward 两个 grouped_gemm kernel 共 569.6 ms 全在 stream 0 上
顺序排队；NCCL 在 stream 11 上 272 ms 几乎打满，但因为 stream 0 自己已经
塞满，NCCL 无法被有效 hide（hidden 282 ms / 占 critical path 的部分 49 ms）。

对比 B 时 stream 0 只占 600 ms，剩下 525 ms wall 内有充足空间让副 stream
gemm 与 NCCL 同时跑。

## 5. 与 note 20 的关系

note 20 已经在 200-iter 实测中给出"Turbo 慢 ~50 ms/step"的标量结论，并
在 yaml 里把 `use_turbo_grouped_mlp` 硬钉为 `false`。本 note 是把那条结
论的根因从 black-box 推到 stream-level 白盒：

- note 20：现象（step time 大）+ 配置（yaml 锁 false）+ truthy bug fix。
- note 21：trace 证据（per-stream / per-category）+ 算力 vs 并行的取舍模
  型，**为后续任何"想动 grouped MLP 路径"的优化建立判据**。

## 6. 后续判据（写给未来的自己）

任何修改 grouped MLP / MoE expert 计算路径的 PR，**审稿前必须先看 trace
里 stream 0 是否被打满到 >85%**：

1. 如果是，Wall 会 regress（哪怕 GPU work 总量减少）。本场景 EP=1 / 32
   experts 全本地，**stream 并行性比单 kernel 算力更值钱**。
2. 如果对端是 EP > 1 / experts 跨节点的情况，alltoall 通信会让 stream 0
   不会被 grouped GEMM 完全占满，那时 Turbo persistent kernel 可能反胜。
   **不要把这个场景的结论盲推到 EP > 1**。
3. yaml 里第 175–178 行已有 inline 注释解释这个 trade-off，未来若想再开
   `use_turbo_grouped_mlp: true`，先复现一组本 note 的 A/B trace 验证。

## 7. 不动 grouped MLP 的下一步优化候选（从同一份 trace 提取）

| # | 候选 | 理由 | 估计省 wall |
|---|---|---|---|
| ① | `patch_moe_overlap: true` | nccl_generic 230 ms 仅 80% hidden，pipeline 化 alltoall 可压暴露部分 | 30–50 ms |
| ② | V2 fused residual（ADD#2） | V1 已砍 ADD#1（-2.3 ms add + -3.2 ms norm bwd），V2 同思路 | 5–10 ms |
| ③ | SwiGLU bwd `_to_copy + cat + split` 去冗余 | kernel 名暴露 view tax 30 ms | 5–15 ms |
| ④ | dtype/view tax audit（直接 copy 22 ms + manual_unroll 22 ms + AUnaryFunctor 23 ms） | note 16 元素税 audit 的下一波 | 10–25 ms |

下一组实验从 ① 开始（已在 `D_moe_overlap/` 启动，本 note 写完时正在跑）。

## 8. 工件位置

- A trace + breakdown：`small_llm_moe_pretraining/primus/run-trace/20260425_b_v1fused/A_fuse0/`
- B trace + breakdown：`small_llm_moe_pretraining/primus/run-trace/20260425_b_v1fused/B_fuse1/`
- C trace + breakdown：`small_llm_moe_pretraining/primus/run-trace/20260425_b_v1fused/C_turbo_grouped/`

每个目录下：`run.sh` + `*.yaml` + `*.sh` + `run.log` + `_outer.log` +
`output/.../pt.trace.json` (78 / 81 / 47 MB) + `breakdown.txt`。

复现：`bash <dir>/run.sh`，~8 min 训练 + ~6 min ckpt save（ckpt 跑完可
`docker exec ... rm -rf <dir>/output/.../checkpoints` 释放 268 GB）。
