## 时间 / 环境

- **时间**: 2026-05-13 21:50 +0800
- **机器**: `mi355-gpu-26` (8× MI355X / gfx950 / XGMI 全互联)
- **容器**: `xiaoming-dev` (podman)
- **配置**: PP=1 / EP=8 / TP=1 / 4 层 (1 dense + 3 MoE) / DSV3 256E / top_k=8 / H=7168 / F=2048 / seq=2048 / micro-batch=1 / global-batch=8 / mock data / `log_interval=1`
- **种子**: Megatron 默认 `seed=1234`，micro-batch=1 单步 batch 完全确定
- **MMOE 路径**: `MMOE_BACKWARD=eager`（forward 走 super-kernel，backward 用 eager replay）
- **Baseline 路径**: `TEColumnParallelGroupedLinear` + 原生 Megatron `MoELayer`

## 什么问题

20 iter e2e 跑通只能证明「**没崩**」，没法证明 **MMOE forward + eager_bwd 跟原生 MoE 数值一致**：

- 用 mock data + 默认 seed，理论上同 seed 同 batch 应该有可比 loss 趋势（但 bf16 + 多 stream MoE atomic-combine 注定不可能 bit-similar）。
- 上一次跑 e2e 时压根没拿到 iter loss，因为 Megatron 默认 `log_interval=100`，我们只跑 20 iter；Primus 的 throughput patch 又把 `print_rank_last` 的 hook 拦在 ROCm mem stats 上，loss 行去了 stdout 但 last-rank 7 的 stdout.log 只有 5 行启动信息（torchrun `--redirects 3` 把 stdout 也接住了但内容空）。

只对了 wall + VRAM + no-NaN 还不够，必须看 **每个 iter 的 lm_loss / grad_norm / loss_scale 是否在 baseline 1‰ 内**，否则 eager_bwd 哪里漏 grad 就要靠下游训出垃圾权重才会发现。

## 做了什么

### 1) Loss mirror patch（dev-only，env 控制）

在 `mmoe_super_kernel_patches.py` 里加了一条 priority=10 patch（`MMOE_DEBUG_LOSS_TO_STDERR=1` 启用），把 `megatron.training.utils.print_rank_last` 包一层：

```python
def _mirror_print_rank_last(message):
    if megatron_utils.is_last_rank():
        sys.stderr.write(f"[MMOE-LOSS] {message}\n")
        sys.stderr.flush()
    original_print_rank_last(message)
```

**priority=10** 是关键：Primus 自己的 `unified_patch`（priority=50）在 `before_train` 阶段会拿 `original_print_rank_last = megatron_training.print_rank_last`。我们的 mirror 先跑，Primus 抓到的就是我们 mirror 出去的版本，调用链变成：

```
training_log()
  → primus_training_log (Primus wrapper)
  → set megatron_training.print_rank_last = primus_print_rank_last
  → call print_rank_last(log_string) [= primus_print_rank_last]
  → enrich with mem/throughput stats → updated
  → call original_print_rank_last(updated)  [= our mirror]
  → our mirror writes "[MMOE-LOSS] ..." to stderr → forwards to megatron.training.utils.print_rank_last → print() on rank 7
```

stderr 上每个 iter 一行 `[MMOE-LOSS] ...`，被 `torchrun --redirects 3` 落到 `rank{R}/stderr.log`，grep 起来就是结构化数据。

### 2) yaml 同时支持 baseline / MMOE 切换

```yaml
use_mmoe_super_kernel: ${PRIMUS_USE_MMOE:true}
log_interval: 1                # force per-iter log
log_avg_skip_iterations: 0     # don't skip warm-up iters in throughput avg
```

Primus 的 yaml loader 把 `${VAR:default}` 展开成**字符串**，所以传 `PRIMUS_USE_MMOE=false` 后 `use_mmoe_super_kernel` 是 `"false" (str)`，对 `bool("false") == True`。在 patch 里加 `_coerce_bool()` 把 `{"0","false","no","off","none",""}` 显式转 False，否则 baseline 仍然会被 patch。

### 3) 两次跑

- `PRIMUS_USE_MMOE=false MMOE_DEBUG_LOSS_TO_STDERR=1` → `/tmp/dsv3_baseline_logs`
- `PRIMUS_USE_MMOE=true  MMOE_DEBUG_LOSS_TO_STDERR=1 MMOE_BACKWARD=eager` → `/tmp/dsv3_mmoe_logs`

raw logs 落 `slab/notes/monolith-moe/raw/{baseline,mmoe}_rank7.log`。

## 实测：loss + grad-norm 对比

| iter | baseline lm_loss | MMOE lm_loss | abs diff   | rel diff   | base gn | mmoe gn | base TFLOP/s/GPU | mmoe TFLOP/s/GPU |
|------|------------------|--------------|------------|------------|---------|---------|------------------|------------------|
|    1 | 12.011050        | 12.010880    | −1.70e-04  | −1.42e-05  | 6.922   | 6.922   |   0.8 (startup)  |   0.9 (startup)  |
|    2 | 12.010020        | 12.009820    | −2.00e-04  | −1.67e-05  | 6.751   | 6.750   |   1.6            |   1.9            |
|    3 | 11.424120        | 11.423700    | −4.20e-04  | −3.68e-05  | 7.217   | 7.218   |  97.0            |  53.3            |
|    4 |  9.687332        |  9.686249    | −1.08e-03  | −1.12e-04  | 6.900   | 6.900   | 166.3            |  68.0            |
|    5 |  6.487745        |  6.488276    | +5.31e-04  | +8.18e-05  |12.022   |12.024   | 179.2            |  67.2            |
|   10 |  1.667311        |  1.665281    | −2.03e-03  | −1.22e-03  | 5.972   | 5.967   | 182.6            |  68.8            |
|   15 |  0.728094        |  0.728003    | −9.08e-05  | −1.25e-04  | 3.313   | 3.312   | 183.8            |  68.7            |
|   20 |  0.567783        |  0.568154    | +3.71e-04  | +6.53e-04  | 2.964   | 2.967   | 184.3            |  68.9            |

(完整 20 行见 `slab/notes/monolith-moe/raw/{baseline,mmoe}_rank7.log`)

## 达成的效果

**数值正确性：PASS**

- iter 1 loss 差 **1.7e-4**（≈ bf16 1-2 ULP）—— 强证据：forward 在第一步就跟 Megatron baseline 等价（差异只能来自 atomic-combine 顺序和 bf16 round-to-nearest 噪声）。
- 整个 20 iter 区间，loss 相对误差 **max = 1.22e-3 @ iter 10**，最后落在 **6.5e-4 @ iter 20**；grad_norm max abs diff **3e-3**（baseline 与 MMOE 在 12.022/12.024、5.972/5.967 这种数对上）。
- 训练曲线 12.01 → 0.57 单调下降一致，无 NaN，无 skipped/nan iter，loss_scale 全程 1.0（bf16 路径）。
- **eager_bwd 路径完整传递了 grad**：如果 backward 漏 grad，最多撑 2-3 iter loss 就会跟 baseline 拉开（experts 接收不到 grad → 退化成只训 attention/dense）。这里 20 iter 都贴着，证明 expert weights 在收 grad。

**性能 (eager_bwd)：184 → 69 TFLOP/s/GPU（2.7× slower）**

- baseline TEGroupedMLP 路径 wall ~232 ms/iter；MMOE eager_bwd wall ~615 ms/iter。
- eager_bwd 完整重跑了 forward 一遍（autograd-aware all_to_all_single + index_add 走 reference path），所以 backward 大约是 2-3× forward。
- 这跟规划完全吻合：eager_bwd 只为正确性兜底，不为性能；下一步 P0 是 `_backward_decomposed`。

## 关键观察

1. **同 seed mock data 不能给 bit-similar 但能给 ~1e-3 量级**。
   - 真实分歧来源：atomic_add combine 顺序（每个 rank scatter 顺序受 IPC scatter latency 影响）+ bf16 round 到不同 expert 求和顺序 + grouped-GEMM 内部 K-loop 顺序变了（MMOE 用 M-flat tile vs TEGroupedMLP 用 per-expert）。
   - rel diff 在 iter 10 处最大（loss 1.6，绝对差 2e-3）—— 模型刚开始 fit、grad 量级最大、bf16 cancellation 风险最高的区间。20 iter 后期回落到 4e-4，趋势良性。

2. **`use_mmoe_super_kernel` 的 yaml env 替换是字符串。**
   - Primus yaml loader `_resolve_env_in_string()` 只做 int/float 强转，bool 是字符串，调用方必须显式 bool 化。已修：`_coerce_bool()` 在 `_is_mmoe_enabled()` 处理。
   - 不修就会出现「`PRIMUS_USE_MMOE=false` baseline 跑里仍然 ready 出 mmoe layer / patch 真应用」这种很难发现的 bug。

3. **Primus 的 throughput patch 把 last-rank 训练 log 行 swallow 进了 stdout（而非 stderr）。**
   - torchrun `--redirects 3` 同时接 stdout 和 stderr，正常情况下应该能拿到。但 rank 7 stdout.log 只有 5 行，而 stderr.log 47 行 —— 推测是 rank 7 stdout 走的是 print(stdout) → buffered → torchrun watcher 在我们查看之前没 flush 完整。
   - mirror 到 stderr 一劳永逸：stderr 立即 flush，外加 `[MMOE-LOSS]` 前缀方便 grep。

## 下一步

- **P0 实现 `_backward_decomposed`**：当前 eager_bwd 是 615 ms/iter，对比 baseline 232 ms/iter 是 2.7× 慢。用 forward 时保留的 `permuted_input` / `fc1_gate_up_save` / `recv_pr` / `send_indices` 拼 hand-written bwd：
  - **dC/dY** (combine 反向): 拿 `out_recv_weighted = out_recv * recv_pr`，反 all_to_all + 反 permute（cheap，主要 IO）。
  - **dW2**: 用 `fc1_gate_up_save` (silu(gate)\*up 之后) 和 `dY` 做 grouped-GEMM-transpose-A。
  - **d(fc1_act)**: `dY @ W2.T`，再走 SwiGLU 反向 (`silu(gate)\*up` 的 dgate / dup) 拼出 `d_fc1_pre`。
  - **dW1**: `permuted_input.T @ d_fc1_pre`。
  - **dx_local** (dispatch 反向): `d_fc1_pre @ W1.T`，再反 all_to_all + 反 sort by topk。
  - 复用 super-kernel forward 的 weight layout，避免再次 transpose；预期 backward ≈ 1× forward，MMOE 整体 throughput 应该回到 ≥ 184 TFLOP/s/GPU（甚至更高，因为 forward 端单 iter 已经 598 TFLOPS）。
- **P1 端到端 throughput benchmark** (mock data, micro-batch=2/4, train_iters=50)：让 eager 和 baseline 都跑稳态吞吐对比；同时跟 `benchmarks/results/dsv3_sparse_8gpu_phase2p1_swiglu_precompute_2026-05-13.txt` 的 forward-only TFLOPS 拉通看「整训练 vs forward only」一致性。
- **P2 把 `MMOE_DEBUG_LOSS_TO_STDERR` mirror 升级为 CI 钩子**: 跑 baseline + MMOE 两路 + 自动算 abs/rel diff，阈值 5e-3 → CI 失败，避免后续 backward / FP8 / mxfp8 路径偷偷改坏 forward。

## 复现命令

```bash
ssh mi355-gpu-26
podman exec -it xiaoming-dev bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/3rd/Primus
export PYTHONPATH=/shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/python:$(pwd)
export PYTHONUNBUFFERED=1
export MMOE_DEBUG_LOSS_TO_STDERR=1
# baseline
PRIMUS_USE_MMOE=false torchrun --standalone --nproc_per_node=8 --redirects 3 \
  --log-dir /tmp/dsv3_baseline_logs \
  primus/cli/main.py train pretrain \
  --config /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/examples/primus/dsv3_4layer_mmoe.yaml \
  --backend_path /shared/amdgpu/home/xiaoming_peng_qle/workspace/MMOE/3rd/Primus/third_party/Megatron-LM
# MMOE
PRIMUS_USE_MMOE=true MMOE_BACKWARD=eager torchrun ... --log-dir /tmp/dsv3_mmoe_logs ...
# grep loss
grep -nE 'MMOE-LOSS' /tmp/dsv3_baseline_logs/*/attempt_0/7/stderr.log
grep -nE 'MMOE-LOSS' /tmp/dsv3_mmoe_logs/*/attempt_0/7/stderr.log
```
