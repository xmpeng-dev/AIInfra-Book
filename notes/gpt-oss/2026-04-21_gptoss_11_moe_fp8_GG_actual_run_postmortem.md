# MoE FP8 Grouped GEMM 实跑结果 + 失败 postmortem

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-04-22 |
| 实验 | M4_C6 (v1: TE+tensorwise) / M4_C6_v2 (TE+delayed) — 在 M4_C5 之上加 MoE FP8 grouped GEMM |
| 结论 | **失败 ❌**：v1 = 1354 ms（-3 % TF），v2 = 1373 ms（-4 % TF）。**M4_C5 (1217 ms / 98 % GPU util) 仍然是最优**。MoE 切到 FP8 grouped GEMM 在 GPU 端确实省了 ~167 ms / iter，但被 (a) FP8 amax all-reduce 让 comm 时间几乎翻倍 (256 → 483 ms/iter) 和 (b) `aten::item` 主机同步爆涨 (10 → 826 ms/iter, 不上关键路径但侵蚀 launch 余量) 完全吃掉。**根本原因 = M4_C5 已经 98 % GPU 利用率，进一步省 GPU compute 立刻让 comm 暴露**。 |

---

## 0. 微基准的预测 vs 实跑结果

| 来源 | per-iter 预测 / 实测 | 对比 M4_C5 |
| --- | --- | --- |
| Microbench (`bench_moe_fp8_grouped_gemm.py`) | 节省 ~150 ms / iter | -11 % step time |
| **实跑 M4_C6 v1** (TE FP8 tensorwise) | **+50 ms / iter（变慢）** | **-3 % TF** |
| **实跑 M4_C6 v2** (TE FP8 delayed) | **+62 ms / iter（变慢）** | **-4 % TF** |

> 微基准只测了 GEMM kernel 本身在单卡 idle 上的耗时；实跑里 GEMM 被 comm 重叠 + 触发 amax all-reduce + 触发 cast_fp8 + 触发 .item() 同步。这些都是端到端层面的系统效应，单算子 microbench 看不到。**这是一个清晰的 microbench 失真案例。**

---

## 1. 实跑数据（profile iter 50–53，3 step 平均）

```
                       step_avg  GEMM_FP8  GEMM_bf16  comm    comm_exp  cast_fp8  item_sync  gpu_util
                         (ms)    (ms/it)   (ms/it)   (ms/it)   (%)      (ms/it)   (ms/it)
M4_C5  (baseline)        1217      69        678      256      19.6%     ~70       10        98.0%
M4_C6_v1 (tensorwise)    1354     463         33      387      35.5%    136       755        95.4%
M4_C6_v2 (delayed)       1373     511         33      483      28.5%    228       826        96.7%
```

> 注：`gemm_bf16` 这一列把 `Cijk_*_BBS_*` 归为 bf16；`Cijk_*_F8*` 归为 FP8。M4_C5 的 678 ms 几乎全是 MoE bf16 grouped GEMM；v1/v2 的 463/511 ms 是 MoE FP8 grouped GEMM（rocBLAS Tensile `Cijk_*_F8B8BS_MT256x256x128`）。

### 1.1 GPU 端：MoE GEMM 确实快了 ~170 ms / iter ✅

```
M4_C5    : MoE bf16 GG = 678 ms/iter
M4_C6_v2 : MoE FP8  GG = 511 ms/iter   →  -167 ms/iter  ✅
```

### 1.2 但是 comm 几乎翻倍 ⚠️

```
M4_C5    : comm = 256 ms/iter  (exposed 19.6 % = 50 ms exposed)
M4_C6_v2 : comm = 483 ms/iter  (exposed 28.5 % = 138 ms exposed)
                                                  ↑ +88 ms exposed comm
```

新增的 ~227 ms/iter comm = **TE FP8 grouped GEMM 的 amax all-reduce**（每个 FP8 tensor 都要在 DP 组里同步 amax 给 delayed scaling 用）。M4_C5 没有这块开销因为 MoE 是 bf16，不需要 amax。

### 1.3 cast_fp8 多了 ~160 ms / iter ⚠️

```
M4_C5    : cast_fp8 = ~70 ms/iter
M4_C6_v2 : cast_fp8 = 228 ms/iter   →  +158 ms/iter
```

每次 MoE FP8 grouped GEMM 前都要 cast_transpose `permuted_local_hidden_states` 到 FP8 + cast_transpose weight 到 FP8 + 反向时 cast 梯度。24 layer × 2 GEMM × ~2 cast = ~96 个 cast_transpose / iter。

### 1.4 `aten::item` 主机同步爆涨 — 但**不上关键路径** ✅（surprising）

```
M4_C5    : aten::item = 10 ms/iter      (n=??)
M4_C6_v1 : aten::item = 755 ms/iter     (n=81 calls/iter)  ← tensorwise per-tensor scale
M4_C6_v2 : aten::item = 826 ms/iter     (n=81 calls/iter)  ← delayed amax history reduce
```

- delayed scaling 比 tensorwise 还差 → 是因为 MoE FP8 后多了 24 × 2 = 48 个 amax buffer，每个都要 `.item()` 读历史窗口。
- 但 GPU 利用率仍 96.7 %（v2）说明这些 .item() 都被异步 launch 余量 hide 掉了，**不算在 wall time 里**。
- 但 launch budget 几乎用尽 — 任何后续会增加 launch 数的优化都会立即把这个潜伏开销暴露出来。

### 1.5 Amdahl ceiling: GPU 已经 98 % 利用率

M4_C5 的 GPU 利用率 = **98 %**，这是个很满的状态。  
- comp = 1066 ms/iter（GPU compute 占满 98 % 的 wall=1217 ms）。  
- comm = 256 ms/iter，但 80 % 跟 comp 重叠了。  
- 所以 wall ≈ max(comp_alone + exposed_comm) = 1066 + 50 ≈ 1116 ms（实测 1217，剩 100 ms 是 launch gap / kernel-launch 排队）。

切到 FP8 后：  
- comp 名义 GPU 时间下降到 1080 ms（FP8 GEMM 省 167 ms，但 cast_fp8 多 158 ms，几乎抵消）。  
- comm 增加到 483 ms（amax all-reduce），exposed 上升到 28.5 % = 138 ms exposed。  
- wall = 1079 + 138 + launch gap = ~1373 ms。**结果就是退化。**

---

## 2. 三件意外发现 / Primus 内部的暗坑

下面三个都是在 source code + 运行日志里挖出来的，跟最初的诊断报告 (`gptoss_10_moe_fp8_grouped_gemm_diagnosis.md`) 的假设不一致，需要 update：

### 2.1 Primus 有一个 patch 在 TE>=1.9.0 时**无条件禁用** PrimusTurbo 的 MoE 路径

源码：`/workspace/Primus/primus/backends/megatron/patches/turbo/te_spec_provider_patches.py:51-62`

```python
if (use_turbo_grouped_mlp and moe_grouped_gemm
    and not moe_use_legacy_grouped_gemm
    and is_te_min_version("1.9.0")):
    log_rank_0("[Patch:megatron.turbo.te_spec_provider] PrimusTurbo not support TEGroupedMLP (TE>=1.9.0); using TE backend...")
    return False   # ← 整个 PrimusTurboSpecProvider 不挂载
```

含义：要让 `pt.ops.grouped_gemm_fp8` 真正被调用，必须  
- `use_turbo_grouped_mlp: true` **AND**  
- `moe_use_legacy_grouped_gemm: true` （不是 false！跟我之前推荐的相反）

（我们用 TE 2.x，所以这个 patch 100 % 生效。）

### 2.2 PrimusTurbo 不支持 `delayed` recipe — 静默退化到 bf16

源码：`/workspace/Primus/primus/backends/megatron/core/fp8_utils.py:140-143`

```python
if config.fp8_recipe == Fp8Recipe.delayed:
    # NOTE: Primus-Turbo not support delayed scaling.
    fp8_quant_config_none_reason = "Primus-Turbo not support delayed scaling."
    fp8_quant_config = None     # ← 然后 enabled_turbo=False
```

进一步导致：

```python
# primus_turbo.py:1236
if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():  # = False
    fc1 = pt.ops.grouped_gemm_fp8(...)
else:
    fc1 = self.grouped_gemm(...)   # ← bf16 pt.ops.grouped_gemm 静默走这里
```

**这就是为什么历史 A1 (`use_turbo_grouped_mlp=true + delayed`) 反而比 B0 慢 25 %**：A1 走的是 PrimusTurbo 的 **bf16** grouped GEMM (`pt.ops.grouped_gemm`)，而这个 bf16 grouped GEMM (内部走 CK) 在我们的 shape 下比 Megatron 的 legacy bf16 grouped GEMM (走 hipblaslt) **慢 ~2x**（microbench: 26 ms vs 13 ms /  fc1 layer）。**根本不是 FP8 vs bf16 的对比**。

### 2.3 想要 PrimusTurbo Triton FP8 GG 的唯一组合 = 全模型 tensorwise，会让 TE Linear 全员 amax sync

唯一能合法触发 `pt.ops.grouped_gemm_fp8` + `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON` 的 yaml 组合：

```yaml
fp8_recipe: tensorwise                  # 必须，否则 PrimusTurbo 走 bf16 fallback
moe_use_legacy_grouped_gemm: true       # 必须，否则路由到 TEGroupedMLP
use_turbo_grouped_mlp: true             # 必须，否则路由到 GroupedMLP（bf16）
```

但 `fp8_recipe: tensorwise` 同时把 **所有非-MoE TE Linear** (QKV proj, attn output, lm_head) 切到 Float8CurrentScaling，于是 24 layer × 3-4 GEMM × `.item()` per iter ≈ 81 个 host sync / iter。我们 v1 就是这么跑的（虽然走的是 TE 不是 Turbo），结果 wall 反而退化。

---

## 3. 真正的下一步建议（更新 diagnosis 报告 §3）

| 选项 | 预期收益 | 工作量 | 风险 | 备注 |
| --- | --- | --- | --- | --- |
| **A. 维持 M4_C5 不动**，把精力转到 comm | — | 0 | 0 | **现状最优**，1217 ms / step (profile) ≈ 1320 ms 训练期 |
| **B. 减少 comm — `--use_distributed_optimizer` ZeRO-1 改 grain** / 更大 NCCL bucket / 切成 RS+AG 重叠 | 50–100 ms（comm exposed 50→0） | 中 | 中 | comm 是当前真正的暴露瓶颈 |
| **C. 干掉 attn bwd `recompile`** — `aiter::fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile` 提示 JIT 重编 | 30–60 ms | 中 | 低 | 看是否能命中预编译 kernel |
| **D. mbs=4 → mbs=8** 让 comp 更长，把 comm 完全 hide | 100–150 ms（comm exposed 50→0 + amortize bias) | 小（仅 yaml） | 内存风险 | M4_C5 已 90 % HBM 利用率，可能 OOM；要量 |
| ~~E. MoE FP8 grouped GEMM~~ | ~~+150 ms 预测~~ | ~~小~~ | — | **实测 -50 ms 退化，弃** |
| F. 自研 fused Triton MoE FP8（cast+gemm+activation 合一） | 30–80 ms | 大（1–2 周） | 高 | 只有当 MoE GEMM 重新成为瓶颈才考虑 |

**优先级推荐**：D > B > C，先做 D（最小改动验证内存余量），之后做 B。E/F 暂时下架。

---

## 4. 文件 & 数据索引

- M4_C6_v1 yaml：`ablations/M4_chain/M4_C6.yaml`、`M4_C6_profile.yaml`
- M4_C6_v2 yaml：`ablations/M4_chain/M4_C6_v2_profile.yaml`
- profile 启动脚本：`run_profile_C6.sh`（v1, env=TRITON）/ `run_profile_C6_v2.sh`（v2，无 env）
- run.log：`ablations/M4_chain/logs/M4_C6_profile_20260421_195433.log` & `M4_C6_v2_profile_20260421_*.log`
- trace（容器内）：
  - `/workspace/code/output/amd/root/M4_C6_profile/tensorboard/...pt.trace.json` (639 MB)
  - `/workspace/code/output/amd/root/M4_C6_v2_profile/tensorboard/...pt.trace.json` (602 MB)
  - `/workspace/code/output/amd/root/M4_C5_profile/tensorboard/...pt.trace.json` (260 MB)
- microbench：`bench_moe_fp8_grouped_gemm.py`（**注意：单卡 microbench 严重高估了端到端收益，因为忽略了 amax all-reduce 和 GPU saturation**）

---

## 5. 总结 — 一句话

**M4_C5 的 GPU 利用率已经 98 %，瓶颈不再是 GEMM 算得快不快，而是 GPU compute 一旦减少，藏在 comp 后面的 comm 就立刻暴露。MoE FP8 grouped GEMM 在 GPU 上省 167 ms，但触发 +227 ms FP8 amax all-reduce + 158 ms cast_fp8 launch，净效应是 +50 ~ +60 ms / iter（变慢）。下一步真正的杠杆是 comm（B/D 选项），而不是 MoE GEMM。**
