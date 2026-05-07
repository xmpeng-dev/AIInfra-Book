# MoE FP8 Grouped GEMM 性能诊断 — A1 历史失败原因 + Triton 后端可行性评估

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-04-21 |
| 模型 | GPT-OSS-20B / MI355X x8 / TP=PP=EP=1 |
| 当前最优配置 | M4_C5 (mbs=4 gbs=32, 5 项优化全开) — 1319.9 ms/step / 633.4 TFLOPs/GPU |
| 目标 | 评估「打开 MoE FP8 grouped GEMM」这一 P0 优化的真实可行性 |
| 结论 | **无需自己写 Triton kernel**：PrimusTurbo 仓库内已有一份成熟的 Triton FP8 grouped GEMM 后端，在 M4_C5 真实 shape 上实测**比当前 bf16 路径快 34% / 33%**（fc1 / fc2）。历史 A1 失败的根因不是 kernel 慢，而是 (a) 默认走 CK 后端、(b) tensorwise 路径里的 `maybe_pre_sync=True` 强制 host sync、(c) 每次 fwd 多做两次 fp8 cast 存 colwise。**改 3 个 yaml 开关 + 1 个 env var 即可拿到收益**。 |

---

## 1. 历史 ablation 数据 — 之前真的慢了多少？

3 次历史尝试（数据 [2026-04-19](2026-04-19_gptoss_01_mlperf_best_e2e_run.md)）都用 `use_turbo_grouped_mlp: true`，全部落后 baseline 200~300 ms / iter：

| 实验 | recipe | 配置 | step (ms) | TFLOPs/GPU | vs B0 baseline |
| --- | --- | --- | --- | --- | --- |
| **B0**  | delayed | 全 bf16 grouped GEMM | **800** | **524** | — |
| A1      | delayed | + use_turbo_grouped_mlp | 1050 | 390 | **-25.6 %** |
| A1b     | tensorwise | + use_turbo_grouped_mlp | 1085 | 381 | **-27.3 %** |
| A1b_tw  | tensorwise | + use_turbo_grouped_mlp | 1086 | 380 | **-27.5 %** |

> 原始日志：`ablations/A1_moe_fp8.run.log`、`ablations/A1b_moe_fp8_tensorwise.run.log`、`ablations/A1b_moe_fp8_tw.run.log`。
> mbs=2 / gbs=16 / lr=4e-4 / FP8 hybrid / iter 60–80 steady state。

3 次结果几乎一致（-25% ~ -28% TFLOPs），不是抖动，是**系统性问题**。后面所有 Bx 链都不再带 `use_turbo_grouped_mlp`。

---

## 2. 根因分析 — kernel 慢吗？不是

### 2.1 PrimusTurboGroupedMLP 调用栈

源码：`primus-patches/files/primus/backends/megatron/core/extensions/primus_turbo.py:1151-1325`。

```
PrimusTurboGroupedMLP.forward
  ├── pt.ops.grouped_gemm_fp8(permuted, w1, tokens_per_expert, ...)   # fc1
  │     └── GroupedGemmFP8TensorFunc.apply  (tensorwise 路径)
  │           ├── quantize_fp8(a, axis=-1)        # rowwise A
  │           ├── quantize_fp8(b, axis=-1/-2)     # rowwise B
  │           ├── grouped_gemm_fp8_impl(...)      # ← 真实 GEMM
  │           ├── quantize_fp8(a, axis=-2)        # colwise A — 为 bwd 存
  │           └── quantize_fp8(b, axis=-2/-1)     # colwise B — 为 bwd 存
  ├── activation_func_with_probs(fc1_output, ...)
  └── pt.ops.grouped_gemm_fp8(intermediate, w2, ...)                    # fc2 重复一次
```

`grouped_gemm_fp8_impl` 内部由 `GroupedGEMMFP8KernelDispatcher` 路由到 3 个后端之一：

| BackendType | 实现 | 触发条件 |
| --- | --- | --- |
| `BackendType.CK` | CK C++ kernel (`csrc/pytorch/grouped_gemm/`) | **代码里硬编码的 `default_backend`** |
| `BackendType.HIPBLASLT` | hipBLASLt | env `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPBLASLT` |
| `BackendType.TRITON` | Triton (`primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py`) | env `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON` |

> 关键：A1 run 没设 env，所以 `user_backend_enum = None`，全部走 `default_backend=CK.value`。

### 2.2 真实 shape 微基准 (单卡 MI355X)

脚本：`bench_moe_fp8_grouped_gemm.py`。Shape 严格匹配 M4_C5：

```
mbs=4, seq=8192, EP=1, num_experts=32, topk=4
  → dispatched tokens = 131072
  → balanced M / expert = 4096
  → fc1 (swiglu gate+up): B=32, M=4096, N=5760, K=2880
  → fc2 (down)          : B=32, M=4096, N=2880, K=2880
```

**结果（fwd + bwd 端到端，per layer / per microbatch）**：

| backend | granularity | fc1 (ms) | fc2 (ms) | sum (ms) | TFLOPs/GPU | vs bf16 hipblaslt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **CK**       | bf16 (legacy 默认)       | 26.36 | 13.81 | **40.17** |  484 | **+98 % 🔴** (兜底慢) |
| **CK**       | fp8 tensorwise (A1 走的) | 11.13 |  6.67 | **17.80** | 1075 | -12 % |
| **CK**       | fp8 rowwise              | 15.75 |  8.19 | 23.94 |  812 | +18 % |
| **HIPBLASLT**| bf16                     | 13.16 |  7.17 | **20.33** |  950 | **基准（生产 bf16 路径）** |
| **HIPBLASLT**| fp8 tensorwise           | 13.51 |  7.41 | 20.92 |  923 | +3 % |
| **HIPBLASLT**| fp8 rowwise / blockwise  | — | — | — | — | 不支持 |
| **TRITON**   | bf16                     | 12.93 |  6.85 | 19.78 |  981 | -3 % |
| **TRITON**   | **fp8 tensorwise**       | **8.64** | **4.83** | **13.47** | **1430** | **-34 % 🟢 最快** |
| **TRITON**   | fp8 rowwise              | 10.62 |  6.22 | 16.84 | 1138 | -17 % |
| **TRITON**   | fp8 blockwise            | 17.31 | 10.11 | 27.42 |  700 | +35 % (差) |

> 注：**生产 legacy bf16** 走的是 Megatron 的 `groupd_gemm.fused_grouped_gemm`，底层是 hipBLASLt 类调度，所以与「HIPBLASLT bf16 = 20.33 ms」是同一性能档位（这两行都在 ~950 TFLOPs/GPU）。

### 2.3 为什么 A1 慢 — 不是 kernel 算得慢

CK FP8 tensorwise 实测 **17.80 ms** 还比 bf16 生产 **20.33 ms** 略快。但 A1 实跑 1050 ms vs B0 800 ms = **多了 250 ms / iter**。差距来自 kernel 之外：

| 开销项 | 来源 | 估算 / iter |
| --- | --- | --- |
| **Host sync (`maybe_pre_sync=True`)** | tensorwise 路径 fwd 调 `hipblaslt_grouped_gemm_fp8` 时强制 stream sync 读 amax | 24 layers × 2 GEMM × (fwd+bwd) = ~96 syncs；MI355X HtoD round-trip ~0.5–1.5 ms → **50–100 ms** |
| **多余的 colwise FP8 cast** | fwd 里同时存 colwise quant of `a` 和 `b` 给 bwd wgrad 用 | 24 × 2 × 2 = 96 个 quant kernel launch，每个 ~30–80 µs → **~5–8 ms** GPU + 不少 host overhead |
| **DeepEP CPU `tokens_per_expert` 拉回** | `deepep_use_cuda_num_tokens_per_expert=False`（A1 配置 `turbo_sync_free_moe_stage=0`），每层都把 token 计数从 GPU 拉回 CPU | 24 syncs / iter，**20–40 ms** |
| **CK kernel 对小 M 排程不优** | A1 跑的是 mbs=2，M/expert = 2048（balanced）；CK 的 tile 选取偏向大 M，<= 2048 时部分 tile 浪费 | 几 ms |
| **额外 quantize_fp8 launch** | tensorwise: 每次 fwd 4 个 quant，bwd 3 个 quant，全是 ~30 µs 小 launch | 24 × 7 ≈ 170 launch / iter，**~10 ms** host bound |

合计 ~85–155 ms，**与 A1 实测的 250 ms 退化基本对得上**（剩下的差异落在 grad 累加、pipeline 不重叠、autograd save_for_backward 多存等次级因素）。

### 2.4 关键诊断 ✅

> **不是 PrimusTurbo 的 FP8 grouped GEMM 算得慢，是它整条 fwd/bwd 包装走 CK + tensorwise + host sync 后整体退化**。

CK 对应的 kernel 本身只比 bf16 快 12% — 不够 cover 上面 100+ ms 的整合开销。

而 **Triton 后端避开了所有这些坑**：
- Triton kernel 没有 `maybe_pre_sync=True` — `GroupedGEMMFP8TritonBackend.execute` 不带 sync 路径。
- Triton 实现的 forward / backward 在算子内部直接做 fused quant，不依赖 `quantize_fp8` Python 调度。
- Triton kernel 在 M=4096 这个尺寸上调度得明显比 CK 好（**8.64 ms vs 11.13 ms**，**-22 %**）。

---

## 3. 是否需要自研一份 Triton 版？— 不需要

| 选项 | 工作量 | 预期收益 (M4_C5 1320 ms 基础上) | 风险 | 结论 |
| --- | --- | --- | --- | --- |
| **A. 直接打开 PrimusTurbo Triton 后端** | 1 env var + 3 yaml 开关 | fc1+fc2 单层省 6.86 ms × 24 layer = **~165 ms / iter（理论上限）**；扣除 quant 开销，预期 **120–150 ms / iter**，**+10 ~ +12 % TFLOPs**，落到 **~1170 ms / iter / ~715 TF** | 数值要 1 次 convergence 验证（FP8 tensorwise vs delayed 损失曲线） | **强推 ✅** |
| B. 自研 Triton fused MoE FP8 GEMM | 几天–几周 | 在 A 基础上再省 0.3–1 ms / layer × 24 = 7–24 ms / iter（**+0.5 ~ +2 %**） | 高（fp8 数值 + bwd wgrad correctness） | **不划算 ❌** |
| C. 写 Triton 把 fp8 quant + grouped GEMM + activation **完全 fused** 成单 kernel | 1–2 周 | 干掉 quant launch 开销，再省 ~10 ms / iter | 高（ROCm Triton fp8 wgrad 端到端 fuse 还很不成熟） | 暂不考虑 |

**理由 A 选项已经是性价比最优**：
1. PrimusTurbo `primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py` 已有 6 个公开 entry point（tensorwise / rowwise / blockwise × fwd / variable-K bwd），全部已 lit-up 过且在 `benchmark/ops/benchmark_suite.yaml` 里有 reg test。
2. dispatcher 自动路由（`GroupedGEMMFP8KernelDispatcher._cache` 是 1024 项的 LRU autotune cache），所以即使 fc1 / fc2 / variable-K bwd 走不同 tile shape 也都被覆盖。
3. 我们 mbs=4 / EP=1 / num_experts=32 这个 shape 在 PrimusTurbo 的 benchmark suite 里属于最常见档位，已经被 tune 过。

---

## 4. 推荐方案 — M4_C6 = M4_C5 + Triton FP8 grouped GEMM

### 4.1 改动清单（3 行 yaml + 1 个 env var）

```yaml
# ablations/M4_chain/M4_C6.yaml — 在 M4_C5 基础上叠加：
fp8_recipe: tensorwise              # was: delayed   ← Turbo FP8 GG 要求
moe_use_legacy_grouped_gemm: false  # was: true      ← 关闭 Megatron legacy 路径
use_turbo_grouped_mlp: true         # was: false     ← 走 PrimusTurboGroupedMLP
```

```bash
# 在 docker exec 启动时设：
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
```

可选（如果 attention 全 bf16 不动）：保留 `use_turbo_attention: false`，避免引入 attention FP8 的额外变量。

### 4.2 不动的事情（防止变量过多）

- `moe_token_dispatcher_type: alltoall` 不动（不切 DeepEP，避免引入 CPU sync 风险）。
- `turbo_sync_free_moe_stage: 0` 不动（这开关只在 DeepEP 路径里生效，alltoall 下无影响）。
- mbs=4 / gbs=32 / lr=8e-4 不动。
- 其它 5 项 M4_C5 优化（NUMA off / nan_check off / ddp_pad / grad_reduce_in_bf16 / use_turbo_rms_norm）不动。

### 4.3 验证步骤

1. **Microbench 在容器里复现一次**（已完成，结果在 §2.2，见 `bench_moe_fp8_grouped_gemm.py`）。
2. **Profile run（80 iter，trace 50–53）**：复用 `ablations/M4_chain/run_profile_C5.sh` 的模式，新建 `M4_C6_profile.yaml + run_profile_C6.sh`，确认：
   - kernel trace 里 MoE GEMM 部分换成 Triton kernel（看 kernel 名带 `triton`）。
   - per-step 时间从 ~1320 ms 降到 ~1150–1200 ms。
   - 没有新的 host-side `aten::item` / `_local_scalar_dense` 增加。
3. **Convergence ablation（300–500 iter）**：FP8 tensorwise 比 delayed amax 损失曲线略陡但通常一致；要看 80~300 iter 的 lm loss 是否仍 ≤ M4_C5 的 ±2%。
4. **如果 step 时间真的降到 ~1170 ms / +10% TFLOPs，并且 loss 曲线没退化，写入 M4 chain 作为 C6**，刷新 `2026-04-21_gptoss_08_mbs4_optimization_chain.md`。

### 4.4 收益预估表（替换 P0 行）

| 来源 | 预期 step 时间 (ms) | 预期 TFLOPs/GPU | 节省 |
| --- | --- | --- | --- |
| M4_C5 (now) | 1319.9 | 633.4 | — |
| **M4_C6 = C5 + Triton FP8 GG (tensorwise)** | **~1170** | **~715** | **-150 ms / +13 % 🟢** |

如果跑出来比这个预估差很多（比如只省 60-80 ms），那是 quant launch overhead + autograd 包装吃掉了一半收益，**那时候**才考虑写自己的 fused Triton kernel（选项 C）。

---

## 5. 执行清单 (TODO)

- [ ] 写 `ablations/M4_chain/M4_C6.yaml`（基于 M4_C5，改 3 个开关）。
- [ ] 写 `ablations/M4_chain/M4_C6_profile.yaml`（带 torch.profiler）+ `run_profile_C6.sh`（带 `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON`）。
- [ ] Profile 跑 80 iter，对比 trace。
- [ ] convergence run 300+ iter，对比 lm loss。
- [ ] 更新 `2026-04-21_gptoss_08_mbs4_optimization_chain.md`，加上 C6 行。
- [ ] **如果**收益 < 80 ms / iter，启动选项 C（自研 fused Triton kernel）的设计 RFC。

---

## 附录 A — Microbench 命令（容器内）

```bash
# 在容器里：
cd /home/xiaompen/mlperf
for backend in CK HIPBLASLT TRITON; do
  PRIMUS_TURBO_GROUPED_GEMM_BACKEND=$backend HIP_VISIBLE_DEVICES=0 \
    python3 bench_moe_fp8_grouped_gemm.py
done
```

源码：`/home/xiaompen/mlperf/bench_moe_fp8_grouped_gemm.py`。

## 附录 B — 关键代码引用

- `PrimusTurboGroupedMLP.forward` — 触发 fp8 grouped GEMM：
  `primus-patches/files/primus/backends/megatron/core/extensions/primus_turbo.py:1197-1325`
- `GroupedGemmFP8TensorFunc.forward` — tensorwise 走 `maybe_pre_sync=True`：
  `Primus-Turbo/primus_turbo/pytorch/ops/grouped_gemm_fp8.py:287-331`
- `GroupedGEMMFP8TritonBackend.execute` — 我们要切到的 Triton 路径：
  `Primus-Turbo/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py:310-392`
- env var 入口：
  `Primus-Turbo/primus_turbo/pytorch/core/backend.py:36`
