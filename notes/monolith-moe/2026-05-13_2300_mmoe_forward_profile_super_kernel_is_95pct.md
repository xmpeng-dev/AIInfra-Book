## 时间 / 环境

- **时间**: 2026-05-13 23:00 +0800
- **机器**: `mi355-gpu-26` (8× MI355X / gfx950 / XGMI 全互联)
- **容器**: `xiaoming-dev` (podman)
- **配置**: PP=1 / EP=8 / TP=1 / 4 层（1 dense + 3 MoE）/ DSV3 256E / **top_k=8**（`deepseek_v3.yaml` 把 base 的 6 覆写成 8，runtime log 确认）/ H=7168 / F=2048 / seq=2048 / micro-batch=1 → **T_src=2048 / T_recv=16384 / per-slot=2048** / `MMOE_BACKWARD=decomposed`
- **方法**: 在 `MonolithMoELayer.forward` 内放入 `MMOE_FWD_PROFILE=1` 触发的细粒度 rank-0 timer（`torch.cuda.synchronize` 包夹每个 step，写 stderr）

## 什么问题

`forward-compute` baseline TEGroupedMLP **37 ms** vs MMOE decomposed **93 ms**，差 56 ms。最初怀疑是 Python wrapper 的 weight stack / routing reconstruct / workspace 检查在累计开销，但 `_ensure_transposed_weights` 上版本缓存修复后 forward 只降 4 ms（97→93）——证明开销不在 wrapper 上。需要拆出 forward 的每一步到底花在哪。

## 做了什么

加了一个 env-var gated 的 per-step profiler 到 `python/mmoe/megatron.py:forward()`，rank-0 上对每个 MoE layer 的 7 个 sub-step 各打一行 `[mmoe-fwd-prof]` 到 stderr，前 12 次调用：

```
route → ensure_workspace → ensure_weights → routing_to_topk → shared_experts → super_kernel → reshape+shared_add
```

每个 stamp 前都 `cuda.synchronize()`，所以测的是真实 GPU 完成时间（包括 launch + kernel + 任何上游未完事项的尾流）。

## 取得了什么效果（steady state，iter≥3，跨 3 MoE layer × 多 iter 平均）

```
  route                  avg=  0.767 ms  n=30
  ensure_workspace       avg=  0.013 ms  n=30
  ensure_weights         avg=  0.070 ms  n=30   <- version cache 已生效
  routing_to_topk        avg=  0.112 ms  n=30
  shared_experts         avg=  0.395 ms  n=30
  super_kernel           avg= 26.512 ms  n=30   <- 占 layer 95%
  reshape+shared_add     avg=  0.067 ms  n=30
  PER-LAYER-SUM           27.937 ms
  x 3 MoE layers          83.812 ms
```

结合 `forward-compute = 93–95 ms`，剩下的 ~10 ms 在 attention + dense + 非 MoE 层。

## 结论：forward overhead 全在 super-kernel 自己里，不在集成层

| 项 | 时间 | 备注 |
|---|---|---|
| super-kernel kernel wall × 3 layer | **80 ms** | 95% of MMOE forward |
| route + topk reconstruct + shared MLP + reshape × 3 | ~5 ms | wrapper 总开销 |
| weight stack（version cached）× 3 | ~0.2 ms | 已不是问题 |
| workspace check（cached）× 3 | ~0.04 ms | 已不是问题 |
| non-MoE layer | ~10 ms | attention + dense |

**MMOE 与 baseline 的 56 ms 差距 = super-kernel kernel 本身的 (26.5 − 7.8) × 3 ≈ 56 ms**，跟 forward-compute 差值正好对上。

### 为什么 super-kernel 在训练里这么慢？—— **bench 用了错误的 token 数**

| 工况 | T_src/GPU | top_k | T_recv/GPU (= T_src × top_k) | per-slot avg (= T_recv/8) | seq × mb | vs bench (per-slot) |
|---|---|---|---|---|---|---|
| 历史 bench | **512** | 8 | 4096 | 512 | (n/a) | 1× |
| 4-layer 单机验证（我们现在跑的） | 2048 | 8 | 16384 | **2048** | 2048 × 1 | **4×** |
| **真实生产 DSV3** | **8192** | 8 | 65536 | **8192** | **4096 × 2** | **16×** |

- 历史 bench 全部跑在 `--tokens 512 --topk 8` 上（`benchmarks/results/*.sh` 全部硬编）
- top_k 实际是 8（`deepseek_v3.yaml` 把 base 的 6 覆写成 8，runtime log 印证 `moe_router_topk: 8 (int)`）
- 生产 DSV3：`seq=4096 × mb=2 = 8192 t/gpu`，TP=1 → SP 不省 seq

**之前 README 把 `4.82 ms / 598 TFLOP/s` 当 headline 是误导**——那是 `T_src=512` 下的，跟训练（更跟生产）不是一回事，per-slot 流量差 **4× / 16×**。

实测 4× scale 对照：

| 工况 | per-slot avg | wall | 实测吞吐 |
|---|---|---|---|
| Standalone bench | 512 | 4.82 ms | 598 TFLOP/s |
| 训练实测（4-layer） | 2048 | **26.5 ms** | **~437 TFLOP/s** |

per-slot **4×**，wall **5.5×**（线性应该 4×），吞吐**降 27%**——sub-linear。生产 per-slot 是 8192（再 4× of 4-layer），按 5.5×/4× = 1.375× sub-linear 比例外推：

- 线性外推：26.5 × 4 = 106 ms / call
- sub-linear 外推：26.5 × (4 × 1.375) ≈ 146 ms / call
- 3 MoE layer × ~110-150 ms = **330-450 ms forward** 只算 MoE 部分。**生产侧很可能不可接受**。

但这都是推算，需要**直接跑 bench @ 2048 / 8192 t/g 来 confirm**。下面一节就是这件事。

sub-linear 的可能根因：

1. **Cross-WG / cross-GPU barrier 不随 M 缩**——每个 phase barrier 是固定常数延迟，M 大时相对占比变小但绝对延迟相同
2. **Routing imbalance 放大**：avg_Te ∝ T_src × top_k，最慢 expert 跟平均的偏差按比例放大，kernel 是 lockstep 必须等它
3. **Fixed K_TILE=128 / 小 MFMA tile**：当前 tile 选择是为 512 t/g 调的，更大 M 下应该上更大的 tile（32×32 MFMA、K_TILE=256），现在小 tile 在大 M 下 K-loop 数 4×/16×，accumulator 压力大
4. **LDS / register 压力**：M 大，每个 WG 内部 phase 完成度差异放大，cross-WG barrier 等待变长

baseline `TEGroupedMLP+alltoall dispatcher` 同样 workload 大概 **~7.8 ms / layer**（推算自 baseline forward 37 ms − non-MoE 10 ms − 路由 1 ms × 3 layer，再 ÷3）——这条路径用的是 hipBLASLt / Aiter 的 autotuned grouped GEMM，对每个 M 形状都能挑到最优 tile。

## 下一步方向

| 优先级 | 方向 | 预期收益 | 难度 |
|---|---|---|---|
| **P0** | **bench `--tokens 8192 --topk 6` 对齐生产 DSV3**；同时 `--tokens 2048 --topk 6` 对齐单机验证 | 必做，统一口径 | 低（改两个参数重跑 30 iter） |
| P1 | 基于 P0 结果 sweep 最优 `comm_wg_ratio` / `target_wgs_per_cu` @ per-slot=6144；很可能不再是 `0.25` | 5-15% | 低 |
| P1 | 大 MFMA tile（16×16 → 32×32）+ K_TILE=256，针对生产 large-M 重调 tile | 15-25% wall | 中 |
| P2 | Phase pipelining：FC1 phase 没全跑完就开 FC2（producer-consumer barrier 替换 lockstep） | 20-30% | 高 |
| - | 接受现状，对生产 large-M 训练 fall back baseline grouped GEMM | -大量 forward | 工程简单但等于放弃训练侧 large-M 性能 |

不过 forward 不是当前最大瓶颈—— **decomposed backward 190 ms 占 iter 47%** 仍是更大的提升空间，对应 todo `decomposed_bwd_perf_opt`。但下一步 P0 必须先把 bench 对齐到真实工况，否则后续所有 sweep 都建立在错误的 baseline 上。

## 教训

- README headline `4.82 ms / 598 TFLOP/s` 必须明确标注 **`@ T_src=512, top_k=8`**，**不是训练工况**（更不是生产工况）
- 至少 **4 个独立维度**（T_src、top_k、seq、micro_batch）都要 cross-check，**这次连续 3 次被打脸**：(1) 起初说 bench = 训练（错）；(2) 改口 2048 t/g 是训练（4-layer 是 2048 没错，但生产是 **8192**）；(3) 用 top_k=8 算 per-slot（生产/4-layer 都是 **top_k=6**）。
- Hard rule：**每次发 perf 数字前，先把 `T_src` / `top_k` / `EP` / `seq` / `mb` 五个数字写出来跟当前讨论的工况显式核对一遍**。如果 bench 的这五个数字跟讨论的工况不同，明确标注"这不是 X 工况"
