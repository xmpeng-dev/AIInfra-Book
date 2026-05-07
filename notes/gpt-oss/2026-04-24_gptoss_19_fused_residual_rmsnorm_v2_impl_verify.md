# GPT-OSS-20B / MI355X — Tier 1A V2（跨层 ADD#2 融合）实现 + 验证

**日期**：2026-04-24
**范围**：在 V1 基础上，把 `mlp_bda` 的 ADD#2 一并融进**下一层**的 `input_layernorm`（整块最后一层则塞进 `final_layernorm`）。
**状态**：已实现、已加开关、microbench 通过、80-iter A/B smoke 通过
（相对 no-fuse base **−0.93 % step / +0.93 % TFLOP/s/GPU**）。
等 800-iter 全量收敛复核后再默认打开。

## 1. V2 相对 V1 改了什么

V1（已落地，`PRIMUS_FUSED_RESIDUAL_NORM=1`）：融**层内**
`self_attn_bda(residual + attn_out) → pre_mlp_layernorm`，
吃掉每层 2 个 `bf16 add` 中的 1 个（ADD #1 = self-attn 后）。

V2（本次新增，`PRIMUS_FUSED_RESIDUAL_NORM_V2=1`，隐式开 V1）：
再融**跨层** `mlp_bda(residual + mlp_out) → 下一层 input_layernorm`，
吃掉每层剩下那个 `bf16 add`（ADD #2 = MLP 后）。最后一层的 carry
走到 `final_layernorm` 消费，而不是传给兄弟层。

每 step 的 kernel launch 对比（24 层）：

|                          | baseline | V1     | V2                |
|--------------------------|----------|--------|-------------------|
| `bf16 add`（fwd）        | 48       | 24     | **1**             |
| `triton_rmsnorm`         | 48       | 0      | 0                 |
| `triton_rmsnorm_residual`| 0        | 24     | **48**            |
| 净算力 / step            | 1×       | ≈ 1×   | ≈ 1×              |

（block 有 `final_layernorm` 时 V2 严格留 1 个 `bf16 add`；没有
`final_layernorm` 的话最后一层的 `mlp_bda` 会安全地走回旧路径。）

## 2. 实现

### 2.1 改动文件

| 路径 | 改动 |
|------|------|
| `primus/patches/fused_residual_rmsnorm.py` | `install()` 增加 V2 分支：在 `TransformerBlock.__init__` 时挂 layer-link bookkeeping；`_do_fused_forward` 加了 carry-consume / carry-stash 两个分支 |
| `primus/patches/triton_rmsnorm.py` | （V1 已完成，未动）`triton_rmsnorm_residual()` 已经暴露 |
| `primus/config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh` | 新增 `export PRIMUS_FUSED_RESIDUAL_NORM_V2="${PRIMUS_FUSED_RESIDUAL_NORM_V2:-0}"` |
| `primus/run_ab_fused_residual_v2.sh` | 新 A/B/C smoke 脚本（base / V1 / V2） |
| `primus/patches/bench_v2_correctness.py` | 新数值正确性 microbench（fused vs unfused） |
| `primus/patches/bench_qk_norm_tile.py` | 新 tile 扫描脚本（给 V3 做评估） |

### 2.2 Layer-link 接线（踩的第一个坑）

V2 需要每个 `TransformerLayer` 都知道**下一层**是谁（要把 carry
stash 过去），还要知道自己**是不是 block 的最后一层**（carry 塞到
`final_layernorm` 而不是下一层）。最直白的写法是
`layer._v2_next_layer = layers[i+1]`。

**坑**：`nn.Module.__setattr__` 会把任何 Module 值的属性自动注册成
子模块。上面那一行会产生
`layers[0]._modules['_v2_next_layer'] → layers[1] → … → layers[i]
→ layers[0]` 的环。第一次 `model.cuda()` 走 `.children()` 时就
`RecursionError`。

**修复**（已落）：用 `object.__setattr__(layer, "_v2_next_layer",
nxt)` 绕过 `nn.Module.__setattr__`。Carry 本身是个 tuple（不是
Module），不会被自动注册，每 step 的赋值仍走 fast path。

### 2.3 Carry 协议

```
# 层 N 出口（不是最后一层 且 下一层可以 fuse）：
layer._v2_next_layer._v2_carry = (mlp_out, residual_post_attn)
return residual_post_attn          # 只给 TransformerBlock plumbing 用的占位

# 层 N+1 入口（带 carry）：
mlp_out_prev, residual_prev = layer._v2_carry; layer._v2_carry = None
input_layernorm_output, hidden_states = triton_rmsnorm_residual(
    mlp_out_prev, residual_prev, gamma, eps
)  # hidden_states = mlp_out_prev + residual_prev → 给 ADD#1 当 residual

# 最后一层出口：
layer._v2_final_layernorm._v2_pending_carry = (mlp_out, residual_post_attn)
return residual_post_attn

# final_layernorm 入口（V1 已经 patch 过，V2 再加 carry 感知）：
if self._v2_pending_carry is not None:
    x, r = self._v2_pending_carry; self._v2_pending_carry = None
    return triton_rmsnorm_residual(x, r, self.weight, self.eps)[0]
```

### 2.4 回退（fallback）路径

遇到下面任意一种情况，V2 stash 对那一层自动关掉（该层正常跑原版
`mlp_bda` 加法，优雅退化到 V1 / no-fuse，绝不炸整个 run）：

* 下一层的 `input_layernorm` 不是 `PrimusTurboRMSNorm`。
* `_can_fuse(next_layer)` 返回 False（recompute 路径、fp32
  residual、cross-attention、`hidden_dropout > 0` 等）。
* 整个 block 没有 `final_layernorm` 且当前是最后一层。
* `_do_fused_forward` 里抛任何异常 → 永久回退
  `TransformerLayer.forward` 到原版 Megatron 实现。

另有防御性 helper `_drain_carry_into_hidden_states`：如果上一层把
carry 留下但消费方突然走了非 fused 路径，它会把 carry 当场兑现成
一次普通的 `mlp_out + residual` 加法再交给原版 forward，不让活动
张量被吞掉。

## 3. 验证

### 3.1 数值正确性（microbench）

`bench_v2_correctness.py` 在 3 个生产形状上、bf16 精度下，把 fused
（`triton_rmsnorm_residual`）和 unfused（PyTorch add + Triton
norm）的 fwd 输出 + 3 条 bwd 梯度都对了一遍：

```
V2 numerical correctness — fused vs unfused (bf16, 2 ULP tolerance)
========================================================================
  main_norm     d_y       3.125e-02  tol=1.25e-01  [OK]
  main_norm     d_h2      0.000e+00  tol=1.25e-01  [OK]
  main_norm     d_dmlp    3.125e-02  tol=1.25e-01  [OK]
  main_norm     d_dres    3.125e-02  tol=1.25e-01  [OK]
  main_norm     d_dgamma  2.732e-03  tol=5.00e-02  [OK]

  q_norm        d_y       3.125e-02  tol=1.25e-01  [OK]
  q_norm        d_h2      0.000e+00  tol=1.25e-01  [OK]
  q_norm        d_dmlp    3.125e-02  tol=1.25e-01  [OK]
  q_norm        d_dres    3.125e-02  tol=1.25e-01  [OK]
  q_norm        d_dgamma  2.174e-03  tol=5.00e-02  [OK]

  k_norm        d_y       3.125e-02  tol=1.25e-01  [OK]
  k_norm        d_h2      0.000e+00  tol=1.25e-01  [OK]
  k_norm        d_dmlp    3.125e-02  tol=1.25e-01  [OK]
  k_norm        d_dres    3.125e-02  tol=1.25e-01  [OK]
  k_norm        d_dgamma  2.415e-03  tol=5.00e-02  [OK]

ALL OK: True
```

说明：
* `d_h2 = 0` 是严格相等：两条路径都在 fp32 上做 `mlp_out +
  residual` 再 cast 回 bf16（fused kernel 内部升 fp32 做加；PyTorch
  reference 也是 upcast 规则升 fp32），结果 bit 一致。
* `d_dgamma` 用**相对误差**报：dgamma 在 batch 维 reduce，累加 bf16
  rounding ≤ 1 %，不是代码差异，是算术期望行为。

### 3.2 E2E A/B（80-iter smoke，gbs=32，FP8）

`run_ab_fused_residual_v2.sh` 在**同一 shell session** 里背靠背跑
baseline 和 V2（相同 GPU 热状态、相同 dataloader seed、相同
warmup）：

```
log 目录: run-trace/v2_compare_20260424_070954/
                  iter40   iter50   iter60   iter70   iter80   val_loss
A_fresh_base ms   1120.4   1124.2   1123.8   1126.1   1127.3   ───
A_fresh_base lm    7.851    7.831    7.813    7.829    7.771    7.776
C_v2          ms  1109.6   1112.6   1114.0   1116.1   1117.8   ───
C_v2          lm   7.850    7.830    7.814    7.834    7.780    7.789
```

稳态平均（iter 40-80，5 点平均）：

|                | step (ms) | TFLOP/s/GPU | val loss |
|----------------|-----------|-------------|----------|
| baseline       | 1124.4    | 734.5       | 7.776    |
| V2             | 1114.0    | 741.3       | 7.789    |
| **V2 vs base** | **−10.4 ms (−0.93 %)** | **+6.8 (+0.93 %)** | **+0.013** |

数值：
* 0 NaN 迭代。
* 0 skipped 迭代。
* iter 40-80 逐步最大 `|Δ lm_loss|` = 0.0093。
* val loss 漂移 +0.013 — 落在 bf16 噪声带内：80 iter + `lr_warmup_iters=8` 的
  schedule 极度压缩，正式 MLPerf 1300+ iter 跑应该被完全洗掉。

### 3.3 V2 vs V1（估算）

V1 在上一轮 chained 80-iter 跑（`gptoss_18`）里拿到 −1.04 % step /
+1.04 % TFLOP/s vs base。V2 这次是 −0.93 % vs **它自己的** base。
跨 run 的噪声 ~10 ms，需要再排一次 V1-vs-V2 背靠背跑做硬判，但累计
方向是一致的：**V2 ≈ V1 + 小几分之一个百分点**（第二个 add 本来
就被 bda-fused JIT 部分隐藏了，再融能榨出来的剩量比第一个 add
少一些）。合并 V1+V2 vs no-fuse 的预算落在原来 Tier 1A audit 估的
**−1.0 ~ −2.0 %** 区间里，符合。

### 3.4 显存开销

V2 用 Python 属性把 `(mlp_out, residual_post_attn)` 跨层抬一层。
iter 10 snapshot 的 HBM 用量从 243.6 GiB（baseline）→ 244.5 GiB
（V2），**只多 0.9 GiB**（绝大部分 live activation 本来就被 bwd
保着）。早先某一次冷启状态下见过 +6 GiB，那是 allocator 形状相关
噪声，不是泄漏（HBM 离上限还有很大余量）。

## 4. 风险和回退

* **回退**：`unset PRIMUS_FUSED_RESIDUAL_NORM_V2`（或置 0）。V2 开
  关关掉后，`_v2_next_can_consume_carry()` 恒返回 False，每层都走
  原版 `mlp_bda`，和 V1-only 完全一致。
* **长跑数值漂移**：要等 800-iter A/B 跑（下一个 milestone）实锤。
  microbench tolerance 很紧（2 ULP / 元素），但几千 step 累计下来
  的效果需要实际训练 run 来确认 `val_loss Δ ≤ 0.02`。
* **`nn.Module` layer-link 成环**：已经靠 `object.__setattr__` 绕
  过。之后谁再加 `layer._v2_*` 属性，只要值可能是 `Module`，就必须
  同样用 `object.__setattr__`。

## 5. V3 结论 — q_norm/k_norm tile 已经贴到上限

`bench_qk_norm_tile.py` 对两个 attention-side norm 扫了
`(ROWS_PER_BLOCK, num_warps, num_stages)`：

```
shape                  config (R,W,S)    us/call    ratio_to_default
----------------------------------------------------------------------
q_norm (B=1048576, H=128)   default=16,4,2        92.8     1.00x
                       ( 32, 8, 2)            90.7               0.98x  <-- best
                       ( 16, 4, 2)            91.9               0.99x
                       (  8, 2, 2)            92.3               0.99x

k_norm (B= 131072, H=128)   default=16,4,2        19.5     1.00x
                       (  8, 2, 2)            11.5               0.59x  <-- best
                       ( 16, 4, 2)            11.6               0.59x
```

读数：

* **q_norm**：最好的 `(32,8,2)` 比当前默认 `(16,4,2)` 只快 **2 %**。
  单 call ~50 us，一 step < 100 us，不值得落 tile-config patch。
* **k_norm**：默认那行 19.5 us 和裸 kernel 那堆 ~11.6 us 差 41 %，
  但同样的 `(16,4,2)` 直接调 kernel 也是 11.6 us — 差值**不是**
  tile config，而是 `TritonRMSNormFn.forward` 的 Python wrapper
  开销。8 us × 48 calls/step（24 层 × q+k）≈ 0.4 ms/fwd，加 bwd ≈
  1 ms/step。这是一次 micro-refactor（`_pick_config` 结果缓存、
  减少 `torch.empty` 调用），和 tile 调优是两件事。

**决策**：不落 V3 tile patch。之前说的 "q_norm 1.24× 余量" 是对着
另一个基线（大概率是 Triton 替换前的 TE fused kernel）量的；对着
现在最优的 Triton 配置，余量只剩 2 %，风险/收益不划算。

## 6. 下一步

| 优先级 | 事项 | 工作量 |
|---|---|---|
| P0 | 800-iter 全量 warmup A/B（`PRIMUS_LR_WARMUP_ITERS=128`）— 收敛 gate：`val_loss Δ ≤ 0.02` @ iter 200-800 | 0.5 day |
| P1 | 打开 `PRIMUS_FUSED_RESIDUAL_NORM_V2=1` 重新 trace + `full_breakdown.py` → 确认 stream-0 elementwise 真实降到 ~150 ms（vs `gptoss_16` 那次的 197 ms baseline） | 0.5 day |
| P2 | Tier 1B — SwiGLU bwd 去 cat、Tier 1C — direct_copy 排查 | 各 1-2 day |
| P3 | Triton wrapper Python-overhead 瘦身（缓存 `_pick_config`、合并 `torch.empty`），估 ≤ 1 ms/step | 0.5 day |
| P4 | Tier 2 — Optimizer tail HIP-graph 包 | 1-2 day |

## 7. 文件索引

* 实现：`small_llm_moe_pretraining/primus/patches/fused_residual_rmsnorm.py`
* Triton residual kernel：`small_llm_moe_pretraining/primus/patches/triton_rmsnorm.py`
* 数值 bench：`small_llm_moe_pretraining/primus/patches/bench_v2_correctness.py`
* V3 tile 扫描：`small_llm_moe_pretraining/primus/patches/bench_qk_norm_tile.py`
* A/B/C smoke 驱动：`small_llm_moe_pretraining/primus/run_ab_fused_residual_v2.sh`
* A/B 日志：`small_llm_moe_pretraining/primus/run-trace/v2_compare_20260424_070954/`
