# gptoss_17 — Tier 1A 实现报告：fused (residual + RMSNorm) Triton kernel

**日期**: 2026-04-24
**前置**: [`gptoss_16` Tier 1 elementwise tax audit](./2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md)
**任务来源**: note 16 列出的 Tier 1A 必做项 — “kill bf16 add = 42 ms / 3.8 % step”
**状态**: 代码落地完成，等待容器内 microbench + 80-iter A/B smoke

---

## 1. 这一刀切在哪

GPT-OSS-20B `TransformerLayer.forward` 每一层 fwd 路径上有两个独立的 `bf16 add`：

```
hidden_states = input_layernorm(hidden_states)            # CALL #1
attn_out = self_attention(hidden_states)
hidden_states = self_attn_bda(attn_out, residual)         # ADD #1 ← 本次目标
hidden_states = pre_mlp_layernorm(hidden_states)          # CALL #2
mlp_out = mlp(hidden_states)
hidden_states = mlp_bda(mlp_out, residual)                # ADD #2 ← 跨层，留给 V2
```

trace 看到的 `vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>` 一共 **42.36 ms / step (3.8 %)**，按 24 层 × 2 add × fwd+bwd 摊算 ≈ 880 us / launch × 48 launch。本版本 **只把 ADD #1 + CALL #2 合一**（同一层内），所以理论上限是 **2 个 add 中砍掉 1 个 ≈ 21 ms ≈ −1.9 % step**。

> 跨层的 ADD #2 → 下一层 CALL #1 也可融合，但需要改 `TransformerBlock` 把状态传过层边界，列入 V2 工作。

---

## 2. 改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `patches/triton_rmsnorm.py` | 新增 4 个 kernel（fwd / bwd × single-row / multi-row）+ `TritonRMSNormResidualFn` autograd Function + `triton_rmsnorm_residual()` 公开 API |
| 2 | `patches/fused_residual_rmsnorm.py` | **新文件** — 运行时 monkeypatch installer：(a) 给 `PrimusTurboRMSNorm.forward` 加 `residual=` 参数；(b) 替换 `TransformerLayer.forward` 走融合路径 |
| 3 | `src/train.py` | 在 `install_runtime_patches()` 里调用 `fused_residual_rmsnorm.install()` |
| 4 | `config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh` | 新增 `PRIMUS_FUSED_RESIDUAL_NORM` 环境开关（默认 0，便于 A/B） |
| 5 | `patches/bench_fused_residual_rmsnorm.py` | **新文件** — 数值正确性 + 单 kernel 性能 microbench |

零 yaml 改动；零 Primus 源码改动；全部走 monkeypatch。失败时自动回退原路径。

---

## 3. Triton kernel 数学校验

### 3.1 Fwd

输入：`x, residual ∈ bf16[B, H]`, `gamma ∈ bf16[H]`，`eps`。
计算：
```
xpr = (x.fp32 + residual.fp32)            # 写回 bf16，作为下一 bda 的 residual
var = mean(xpr^2)
rstd = 1 / sqrt(var + eps)
y = (xpr * rstd * gamma.fp32).bf16
保存：xpr (bf16), gamma (ref), rstd (fp32)
返回：(y, xpr)
```

精度路径与现有 `_rmsnorm_fwd_kernel` 完全一致（fp32 内部累加，仅最终结果 cast 回 bf16），所以 vs `F.rms_norm(x + residual)` 的最大偏差应该 ≤ 2⁻⁵（bf16 的 ULP 量级）— 已写进 microbench tolerance 5e-3。

### 3.2 Bwd

记 `xhat = xpr * rstd`，标准 RMSNorm 反传：

```
dxhat = dy * gamma
m = mean(dxhat * xhat)
dx_norm = (dxhat - xhat * m) * rstd          ← gradient flowing through y
```

融合关键：`xpr` 既是 norm 的输入，也是下一 bda 的输入 → autograd backward 同时收到 `dy` 和 `dxpr_external`。kernel 直接做：

```
dx = dx_norm + dxpr_external
dgp = dy * xhat                              # per-row partial dgamma
```

最后 `xpr = x + residual` 的 Jacobian 是 `[I, I]`，所以 autograd 返回 `(dx, dx, dg, None)` —— **同一个 dx 同时给 x 和 residual**。无需分配两份。

> 这一点很重要：如果 backward 给 x 和 residual 各申请一个 dx tensor，反而新增了一次 `bf16 copy`，把节省抵消掉。当前实现复用同一个 tensor。

### 3.3 dgamma reduction

跟现有路径一致：kernel 写 fp32 partial `[B, H]`，外面一句 `.sum(dim=0).to(bf16)`。这一步本来就是 cheap 的 single-call torch reduce，不动。

---

## 4. Monkeypatch 设计

### 4.1 入口

`fused_residual_rmsnorm.install()`：

1. 读 `PRIMUS_FUSED_RESIDUAL_NORM`，未开则 no-op。
2. 替换 `PrimusTurboRMSNorm.forward(self, x)` → `forward(self, x, residual=None)`：
   - `residual is None` → 原路径（不破坏 q_norm / k_norm / final_layernorm 等单输入调用点）。
   - `residual is not None` → `triton_rmsnorm_residual(x, residual, gamma, eps)`。
3. 替换 `TransformerLayer.forward` → `_fused_layer_forward`：
   - `_can_fuse(layer)` 静态前提检查（见下）。
   - 任意 layer 失败 → 整体回退原 `forward`（一次性，避免反复抖动）。

### 4.2 启用前提（`_can_fuse`）

回退条件（任一命中即不融合）：

* `config.fp32_residual_connection`：原路径会先把 residual cast 到 fp32，融合 kernel 不支持。
* `config.inference_fuse_tp_communication`：推理优化路径里 attn 输出已经隐含了 residual add，融合反而会重算。
* `recompute_input_layernorm` / `recompute_pre_mlp_layernorm`：会再把 norm 当 checkpoint 重算，融合后没办法分离 norm-only 输出。
* `offload_attn_norm` / `offload_mlp_norm`：`FineGrainedActivationOffloadingInterface` 会包 bda 输入，融合路径不识别。
* `hidden_dropout != 0`：bda 是 `bias + dropout(x) + residual`，dropout 不为零时 add 不再是单 add。
* `pre_mlp_layernorm` 不是 `PrimusTurboRMSNorm`：必须是我们能塞 residual 入参的子类。
* 有 cross-attention：本路径只覆盖 decoder-only 标准层。

GPT-OSS-20B 当前配置满足全部条件 → **每一层都会走融合路径**。

### 4.3 fwd 重写

`_do_fused_forward(layer, hidden_states, **kw)` 复刻 `_forward_attention + _forward_mlp` 但去掉 ADD #1 的独立调用，关键 3 行：

```python
attn_out, attn_bias = attention_output_with_bias        # 解 tuple
if attn_bias is not None: attn_out = attn_out + attn_bias  # GPT-OSS 走不到
pre_mlp_layernorm_output, hidden_states = layer.pre_mlp_layernorm(attn_out, residual=residual)
```

`pre_mlp_layernorm(...)` 由前面的 monkeypatch 接管，返回 `(y, xpr)`，分别喂给 `mlp(...)` 和 `mlp_bda(...)` —— 结构上跟原代码一一对应，`mlp_bda` 的 ADD #2 完全没动。

---

## 5. 风险与回滚

| 风险 | 触发条件 | 回滚 |
|---|---|---|
| Triton kernel 数值漂移 | bf16 累加路径不一致 | 跑 `bench_fused_residual_rmsnorm.py`；超 tol 直接关 env var |
| autograd grad 错配（dx ≠ dresidual） | kernel bwd 写错 | `gradcheck` 在 microbench 里走 `torch.autograd.backward([y, xpr], [gy, gxpr])` 对照参考路径 |
| 某层条件不满足 | 未来加 cross-attention / dropout | `_can_fuse` 单层 False → 该层走原 forward；首次抛异常 → 全局退回 |
| profile path 名变化 | downstream 看 NVTX 区域名 | NVTX `fused_residual_pre_mlp_layernorm` 用唯一前缀，不与原 `self_attn_bda` 撞 |
| DDP grad bucket 顺序变化 | 减少了 1 个 op，autograd 图微变 | `gradient_accumulation_fusion` + `overlap_grad_reduce` 都走的是 per-param hook，op 数量与图无关，不影响 |

**一行禁用**：`PRIMUS_FUSED_RESIDUAL_NORM=0 ./run...`（默认值就是 0）。

---

## 6. 预期收益（保守 + 进取）

| 项 | 估计 |
|---|---|
| 本版砍掉的 add 数（fwd） | 24 layer × 1 = 24 |
| 占当前 `bf16 add` 总量 | ≈ 24 / 48 = 50 %（假设 fwd:bwd 1:1）→ ~21 ms |
| step 节省 | **−1.5 ~ −2.0 %**（保守 17 ms / 进取 22 ms） |
| E2E 锚（vs 1129 ms baseline） | **1108 ~ 1112 ms / step**（−1.9 ~ −2.2 %） |

**比 note 16 给的 −3.4 ~ −4.3 % 少一半** —— 那个数字是把跨层 ADD #2 也算进去的；本版本明确只切单层内 add。等 V2 把 mlp_bda → 下一层 input_layernorm 也融合，就能把另一半收回来。

> **不应期望一次 V1 完成 note 15 给的 −7~9 % Tier 1 总目标**。Tier 1B (SwiGLU bwd 去 cat)、Tier 1C (direct_copy 排查) 是 Tier 1 的剩余 ROI 来源；fused residual+norm 单独占 Tier 1 的 1/3 上下。

---

## 7. 验收门槛

### 7.1 容器内 microbench（先跑）

```bash
docker run --rm -it $CONT bash -lc \
  "cd /workspace/code/patches && python3 bench_fused_residual_rmsnorm.py"
```

通过条件：每个 shape 都打印 `OK` 且 `speedup >= 1.5x`（vs 当前 `add() + triton_rmsnorm`）。

### 7.2 80-iter smoke（A/B）

```bash
# A：禁用
PRIMUS_FUSED_RESIDUAL_NORM=0 EXP=...gpt_oss_20B-pretrain-fp8.yaml \
  PRIMUS_TRAIN_ITERS=80 ./launch.sh

# B：启用
PRIMUS_FUSED_RESIDUAL_NORM=1 EXP=...gpt_oss_20B-pretrain-fp8.yaml \
  PRIMUS_TRAIN_ITERS=80 ./launch.sh
```

通过条件（看 `run_base.log` 里 `iter [60-80]` 的 step time）：

* B vs A 稳态 step time 下降 **≥ 12 ms (−1.0 %)**（保守阈值，给抖动 0.5 % 余量）。
* B 跑出来的 train loss 在前 80 step 与 A 同步（Δ ≤ 1e-3）。

不达标的处理：

* B 比 A 慢 → kernel 融合反而引入新瓶颈（很可能是 `_pick_config` 误判 BLOCK_H），跑 microbench 看 per-shape 耗时。
* 数值偏差 → 只关 env，不撤代码（kernel 本身有单元测试守住）。

### 7.3 收敛验证（通过 7.2 后）

跑 1× 完整 mlperf E2E，验证 val loss ≤ 3.34 在 ≤ 16 个 eval 内命中（参考 `gptoss_06`）。

---

## 8. V2 / 后续

* **V2 — 跨层 ADD #2 + 下一层 CALL #1 融合**：要改 `TransformerBlock.forward` 让相邻层之间传 `(mlp_out, residual)` tuple；估再 −1.9 %（剩下另一半 add）。
* **V3 — `_rmsnorm_fwd_residual_kernel` 直接吃 bf16 SwiGLU MLP 输出**：把 MLP 末端的 bf16 cast 也吞进 norm pre-add，再省 ~5 ms (Tier 1 剩余预算之一)。
* **V4 — final_layernorm 之前的 add 融合**：单独 1 ms 量级，等 V2 落地后顺手。

合在一起，note 16 给的 Tier 1 合计 −4.4 ~ −6.5 % 仍然成立，**本 V1 是其中第 1 步**。

---

## 9. 落地文件路径

```
small_llm_moe_pretraining/primus/
  patches/
    triton_rmsnorm.py                    (M)  +160 行 kernel + autograd
    fused_residual_rmsnorm.py            (A)  monkeypatch installer (~210 行)
    bench_fused_residual_rmsnorm.py      (A)  正确性 + perf microbench
  src/
    train.py                             (M)  install_runtime_patches() 加 1 处调用
  config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh   (M)  新增 PRIMUS_FUSED_RESIDUAL_NORM
```

无 Primus / Megatron 源码 patch，无 Dockerfile 改动，无 yaml 改动。
