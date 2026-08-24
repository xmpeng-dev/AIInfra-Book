# MXFP8 Mega MoE — 反向(backward)性能汇总 fp8 vs bf16（T=8192）@ MI355X

> **What**: fp8(MXFP8) mega MoE 反向各 stage 延迟 vs 同族 bf16 参考,逐 stage 对比 + 优化靶点。
> **Where**: `smci355-ccs-aus-n01-21`（MI355X / gfx950 ×8），容器 `xiaoming-dev`，job 22522。
> **配置**: DeepSeek-V3，EP8，**T=8192**，H=7168，I=2048，E=256，topk=8，BM=BN=256，routing=load_balanced。
> **通信**: 已迁到 epoch 自复位；CU 配比已重调（L2 combine=32、fc1_dgrad_combine combine=24）。日期 2026-07-24。
> **测量**: 跨 8 rank 取最慢；fp8 wgrad 报 **FULL**(meta+quant+requant+GEMM) 和 **GEMM-only**；bf16 wgrad 是 Triton
> variable-K GEMM(合成操作数,GEMM-only,bf16 无需 quant);combine/dispatch 为整 kernel 延迟。

## 逐 stage 对比（fp8 vs bf16，bf16/fp8 = 加速比）

| 阶段名 | fp8 (ms / TFLOPS) | bf16 (ms / TFLOPS) | bf16/fp8 | fp8 细分 |
|---|---|---|---|---|
| `dispatch_fc2_dgrad`（dispatch(dy)+fc2-dgrad） | **1.678** / 1218 | 2.504 / 817 | **1.49×** | fused dispatch PUSH + GEMM |
| `fc2_wgrad`（dW2, variable-K） | **1.846** / 1107 (FULL) | 2.115 / 967 | FULL **1.15×** / GEMM **1.96×** | GEMM 1.080(1893 TF) + requant 0.365 + quant 0.144 + meta 0.122 |
| `fc1_wgrad`（dW1, variable-K, LOCAL） | **2.898** / 1411 (FULL) | 3.743 / 1092 | FULL **1.29×** / GEMM **1.84×** | GEMM 2.038(2007 TF) + requant 0.365 + quant 0.278 + meta 0.123 |
| `fc1_dgrad_combine`（fc1 dgrad + combine + reduce） | **3.031** / 1349 | 3.807 / 1074 | **1.26×** | fc1-dgrad GEMM(K=2I) + fp8-PUSH combine + reduce |
| **反向合计**（4 stage，kernel-only） | **9.453 ms** | 12.169 ms | **1.29×** | (未含 SwiGLU^T 等 glue) |

- e2e 完整 autograd(fwd+bwd)参考:fp8 **14.71 ms** vs bf16 19.97 ms = **1.36×**(梯度 SNR dx/d_topk_w/dW1/dW2 全 ≥15 dB PASS)。
- bf16 `fc1_dgrad_combine` 在 load_balanced/T=8192 本次跑通(3.807ms);历史上 K=2I 大 T 偶发 OOB,与 fp8 路径无关。
- wgrad 的 bf16 腿是 GEMM-only(合成操作数,值无关);fp8 FULL 含 colwise requant/quant。故“FULL vs bf16”是净训练口径,“GEMM vs bf16”看纯算力优势。

## 关键观察
1. **fp8 wgrad GEMM 本体 ~2×**(dW2 1.96× / dW1 1.84×),但 **colwise requant/quant 把 FULL 稀释到 1.15–1.29×**。
   - dW2 quant/requant/meta ≈ 0.63ms(占 FULL 34%);dW1 ≈ 0.77ms(占 26%)。两个 wgrad 共 ~1.15ms 的量化开销。
   - 这些量化是**访存受限的转置量化 kernel**(`colwise_(re)quant_mxfp8_grouped_flydsl`),之前测只有 ~35% HBM 带宽 → 有明显 headroom。
2. **`fc1_dgrad_combine`(3.03ms)是单项最大**,只领先 bf16 1.26×;它是 dgrad GEMM(K=2I)+ combine PUSH + reduce 三合一。
3. **`dispatch_fc2_dgrad`(1.49×)最健康**,dispatch/preshuffle CU 已是最优 24/8。
4. wgrad GEMM 本体(dW1 2.04 / dW2 1.08)是变长-K “瘦组”regime,占用受限(LDS 128KB → 1 block/CU),深水区(LDS 4→2 buffer)高风险已暂缓(见 `2026-07-24_0956_..._wgrad_gemm` note)。

## FUSED dual-quant of grad_l1 优化 loop（目标 0.1ms;单卡,grad_l1 P=65536 F=4096）

grad_l1 前向后被量化两次(rowwise-preshuffled E4M3 给 STEP3 + colwise-grouped E5M2 给 dW1)。新 FLYDSL
`rowcol_dual_quant_mxfp8_grouped_flydsl`(`quant_colwise_trans_flydsl.py`)一次读 grad_l1 双输出,byte-exact
于两个原 kernel。**注:HBM 底 ≈0.136ms**(读 536MB + 写 268+268 + scale ≈1.09GB / 8TB/s),0.1ms 低于物理底,
朝底优化到停滞为止。

| round | 改动 | fused ms | vs split | byte-exact | 备注 |
|---|---|---|---|---|---|
| 1 | baseline 2D-tile LDS + 2-phase rowwise | 0.331 | 1.28× | ✅ | profile: LDSBankConflict 11.85%, VALUBusy 58%, occ 10.6/CU (LDS 33KB-capped), MemStall 1.4% |

## 优化靶点（优先级）
1. **colwise requant/quant(共享,~1.15ms 跨 dW1+dW2,访存受限 ~35% BW)** —— 一处优化双收益,风险低,估计可省 ~0.4–0.5ms。**首选**。
2. **`fc1_dgrad_combine`(3.03ms,单项最大,仅 1.26×)** —— 需 rocprof 拆 GEMM(K=2I dgrad NT)vs combine PUSH vs reduce,分别定方向。
3. **wgrad GEMM 占用(dW1/dW2)** —— LDS 4→2 buffer 冲占用翻倍,高风险/收益不确定,暂缓。

## 复现
```bash
cd /perf_apps/xiaoming/MegaMoE; export PYTHONPATH=$PWD
# fp8 反向全 breakdown(单进程 8 卡):
MASTER_PORT=$((20000+RANDOM%20000)) python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage bwd --mode load_balanced
# bf16 反向参考:
MASTER_PORT=$((20000+RANDOM%20000)) python benchmark/ops/training/bench_mega_moe_bf16.py --num-tokens 8192 --stage bwd --mode load_balanced
# e2e fwd+bwd + 梯度 SNR:
MASTER_PORT=$((20000+RANDOM%20000)) python benchmark/ops/bench_mega_moe_fused_fp8_bwd.py --num-processes 8 --num-tokens 8192
```

数据采集: 2026-07-24, feat/xiaompen/mega_moe_flydsl_mxfp8 @ commit b41054f6（CU retune）+ bench 增加 `--stage bwd`。
| 2 | Phase-A 32-feature read lane-swizzle (break LDS bank conflict) | ~0.33 (noisy) | ~1.34× | ✅ | **LDSBankConflict 11.85%→0.00%**, VALUBusy 58%→72.9% (now VALU-bound), occ ~11. sclk bouncing → use ratio + PMC as metric |
| 3 | bf16 LDS tile (32KB→16KB, occupancy) | — | — | (rolled back) | FLYDSL extf-from-bf16-LDS type friction; AND low value: after R2 the kernel is VALU-bound (72.9%), so occupancy is not the cap. Reverted to R2. |

### Loop conclusion (stopped)
- **Achieved**: FLYDSL fused dual-quant of grad_l1 — byte-exact to both shipped kernels, **~1.36× vs the split**
  (rowwise-preshuffled E4M3 + colwise-grouped E5M2 in one read), **LDS-bank-conflict-free** (R2 swizzle).
  Post-R2 the kernel is **VALU-bound (VALUBusy 72.9%, MemUnitStalled 1.4%, LDSBankConflict 0%)**.
- **0.1ms target is physically unreachable**: the dual moves ~1.09 GB (read grad_l1 536MB + write rowwise
  268 + colwise 268 + scales), so the **HBM floor is ~0.136 ms** at 8 TB/s. 0.1 ms is below the floor.
- **Stopping** at diminishing returns: the remaining lever is VALU reduction (inherent dual-quant math;
  Phase-B cvt_pk pairing ~+5% best case), which is (a) hard, (b) unmeasurable on this node — **sclk bounces
  (level 0/1) → absolutes swing ±30%**, so <10% gains can't be validated. The clean wins (fused + conflict-free)
  are banked; ratio-vs-split is the stable metric used.
- **Net realized (once wired into backward)**: ~0.09 ms/backward saved by fusing grad_l1's two quants into one read.
- Next (not perf-loop): wire the fused kernel into STEP3 (accept pre-quant rowwise) + dW1 (accept pre-quant colwise) + e2e gradcheck.

## FUSED dual-quant — loop 续跑 (GPU4 pinned, 目标 0.2ms, 2026-07-24)

重启 /loop,先解决抖动:`rocm-smi --setperflevel high` 本机 **Not supported**(sclk 仍 level0/1 跳)→
扫 8 卡取 spread 最紧的 **GPU4 (±0.16%)** 全程钉住(`HIP_VISIBLE_DEVICES=4`)。基线(round-3 swizzle 版) **0.3339ms**。

| round | 改动 | fused ms | 决策 | 关键发现 |
|---|---|---|---|---|
| 4 | rowwise phase-B `1/scale`→**pow2 指数倒数** `(254-biased)<<23` | **0.3077** | ✅ −7.8% | fdiv 是真 VALU 成本(32/thread);确认 VALU-bound。byte-exact |
| 5 | colwise `_e8m0_quant_pack` 同样 pow2 倒数 | 0.3066 | ✅ 噪声内 | colwise 的 inv 被提到循环外(1/thread)→ 收益可忽略 |
| 6 | BT 256→128(减半 LDS tile 提占用) | 0.3507 | ✗ 回退 | **非占用受限**:WG 数翻倍→prologue+barrier 开销压倒占用收益 |
| 7 | phase-B 复用 `vals[]` 寄存器(免 32 ds_read/thread) | **0.3024** | ✅ −1.4% | 缩短依赖链有效;VGPR 52→56(占用无碍) |
| 8 | BT 256→512(减 WG 数) | 0.3037 | ✗ 回退 | BT=256 是甜点;占用损失抵消开销收益 |
| 9 | `cvt_f32_fp8` 单转换 | — | ✗ 无效 | 该 op 是 fp8→f32 反量化方向,不存在标量 f32→fp8 |
| 10 | amax **树规约**(深度 32→5) | 0.3280 | ✗ 回退 | 树需 32 leaf 同时存活→**寄存器压力** > ILP 收益;顺序 fold 更优 |

### Loop 续跑 conclusion
- **本轮净收益 0.3339 → 0.3024 ms(−9.4%)**,全部 byte-exact、K1 kernel-internal、1:1 迁移真实训练。主力是
  **R4 pow2 指数倒数**(−7.8%);R7 寄存器复用(−1.4%)。
- **0.2ms 目标对本 kernel 不可达**:双输出 traffic ≈1.07GB,理想 HBM 底 ~0.13ms、现实 BW-bound ~0.19–0.25ms;
  但 kernel 是 **compute/barrier-latency-bound(VALUBusy 60%、MemUnitStalled 1.5%、非占用受限)**,实际底 ≈0.30ms。
- **已排除的方向**(见 `agent/historical_experience/gfx950/mxfp8_dual_quant/flydsl/tips.md`):占用(小/大 tile)、
  树规约(寄存器压力)、cross-lane phase-A(thread=column 需 32 规约/thread,比现设计差 5×)。跌破 0.25ms 需改输出 layout,
  会破坏下游 dW1 GEMM / STEP3 combine 契约,超出本 kernel 范围。
- **最终 kernel = round-7**,已落回 `quant_colwise_trans_flydsl.py`。

### 已接进 backward(wired in, 2026-07-24)
把 fused `rowcol_dual_quant_mxfp8_grouped_flydsl` 接进 `mega_moe_backward_fp8_impl`:swiglu_backward 出 grad_l1 后
**一次读**出双输出 → `(q_row,a_sp)` 喂 STEP3 combine(新增 `x_fp8_rowwise=` 参数跳过其内部 rowwise quant)、`(q_col,s_col)`
喂 dW1(`_mxfp8_variable_k_wgrad_dw1` 改收 pre-quant colwise + 共享 meta)。顺带把 grouped `meta` 提到一次算好,
在 fused-quant / dW1 / dW2 间共享(省 2 次 meta D2H ≈0.24ms)。

- **e2e 验证(8卡,T=8192,byte-exact 保持)**:fp8 fwd+bwd **14.25 / 14.33 ms**(两次 clean run)vs 之前 14.71ms →
  **省 ~0.4ms/step**(fused 一次读 ~0.13 + 共享 meta ~0.24)。梯度 SNR dx≈20-22 / d_topk_w≈23 / dW1≈19.4 / dW2≈19.7 dB **全 PASS(≥15)**。
- 改动文件:`mega_moe_backward_fp8_impl.py`(orchestration + dW1/dW2 helper 签名)、`grouped_gemm_combine_fp8_kernel.py`
  (combine 加 `x_fp8_rowwise=`)、`quant_colwise_trans_flydsl.py`(fused kernel,R4/R7)、fp8 `__init__` 导出、
  `bench_mega_moe_fp8.py`(dW1 stage 适配新签名)。
- **坑**:e2e 别在一个 container invocation 里 back-to-back 连跑多次——symm buffer/IPC 显存不释放会 OOM;每次单独跑或先 pkill 清理。

## fc1_dgrad_combine rocprof 拆解 (2026-07-28, n05-29, T=8192 EP8)

**Stage 延迟**: ~2.99–3.01 ms (kernel-only, combine_cu=24–32 flat); e2e bwd-only stage ~3.02 ms。

### 三腿 breakdown (`bench_step3_fp8.py --breakdown`)

| 模式 | ms | 说明 |
|---|---|---|
| GEMM_ONLY | 2.200 | fc1 dgrad NT mxfp8 GEMM + CShuffle fp8 epilogue |
| PUSH_ONLY | 2.029 | combine XGMI push (skip GEMM spin) |
| NO_REDUCE (GEMM‖PUSH) | 2.522 | overlap 实测 |
| full (含 reduce+gate) | 2.615 | reduce+gate 估 ~0.09 ms |
| serial GEMM+PUSH | 4.229 | 无 overlap 上界 |

- overlap 省 **1.71 ms (40% of serial)**; vs ideal max(G,P)=2.2 ms 仍有 **~0.32 ms gap** (overlap 效率 ~87%)。
- **combine_cu sweep (16–48)**: 24–32 ≈2.99 ms flat; 不是大杠杆(两腿等长 ~2.0–2.2 ms)。

### ATT hotspot (kern_1, `bench_fc1_dgrad_combine_trace.py`)

| 模式 | 主瓶颈 | Top line |
|---|---|---|
| **full** | 50% reduce-flag spin (VMEM-wait) | `grouped_gemm_combine_fp8_kernel.py:298` (`s_sleep` on reduce flag) |
| **GEMM_ONLY** | 47% post-tile `s_barrier` | `:509` (after `gemm_mxfp8_nt_tile`, before combine_flag bump) |
| **GEMM_ONLY** | 27% MFMA | `gemm_helper.py:366` |
| **PUSH_ONLY** | 97% VMEM-wait | `:227–257` (`combine_push_fp8_segment` load-all-then-store) |

### 优化方向 (按 ROI)

1. **提 GEMM‖PUSH overlap** (~0.3 ms): 两腿等长; reduce spin 占 ATT 但 wall 仅 ~0.09 ms。
2. **GEMM**: 减 post-tile barrier 成本(`:509`); VMEM prefetch in `gemm_mxfp8_nt_tile` (11% VMEM-load)。
3. **PUSH**: `combine_push_fp8_segment` load/store 交错(现 load-all→store-all); XGMI 本质 bound。
4. **reduce spin**: 专用 reduce CU 或更快 flag path; wall 小,优先级低。

复现:
```bash
PYTHONPATH=$PWD python benchmark/ops/bench_step3_fp8.py --num-processes 8 --breakdown
PYTHONPATH=$PWD python benchmark/ops/bench_fc1_dgrad_combine_trace.py --num-processes 8
# ATT: PT_COMBINE_GEMM_ONLY=1 / PT_COMBINE_PUSH_ONLY=1 + rocprofv3 kern_1
```

### GEMM post-tile barrier 优化尝试 (2026-07-28, 结论: 无效)

ATT 显示 GEMM-only 模式 `:509` `s_barrier` 占 ~47%,但改动 sync 策略 **wall time 不变**:

| 变体 | full STEP3 | GEMM_ONLY | dx |
|---|---|---|---|
| baseline (`s_waitcnt(0)` + `barrier`) | 2.60 ms | 2.39 ms | finite ✅ |
| `_wait_mem()` + `barrier` | 2.68 ms | — | finite ✅ |
| `_wait_mem()` + release atomic **无 barrier** | 2.70 ms | 2.41 ms | finite ✅ (同 norm) |
| 仅 `barrier` (无 s_waitcnt) | 2.69 ms | — | finite ✅ |

**结论**: barrier stall 是 epilogue 跨 wave 完成时间差（straggler）的反映,减 sync 指令不能缩短 wall time; 且去掉 barrier 无收益甚至有噪声级回退。**PUSH load/store 交错**同样 flat (~0.5%)。

下一步 ROI: **CShuffle epilogue** 本身(VALU/amax/cvt)或 **GEMM/PUSH 腿本身** 的优化; overlap CU 配比已调。

### overlap CU 配比调优 (2026-07-28, cc=20)

扫 `num_combine_cu` × `PT_COMBINE_NO_REDUCE`（GEMM‖PUSH overlap 指标）:

| cc | NO_REDUCE | 备注 |
|---|---|---|
| 4–12 | 5–14 ms | combine 跟不上 GEMM，严重 stall |
| 16 | 2.62 ms | 可用下界 |
| **20** | **2.47–2.52 ms** | **甜点** |
| 28 (旧默认) | 2.51–2.53 ms | |
| 40+ | 2.67+ ms | combine 占 CU 过多 |

- **full STEP3 / fc1 stage**: cc=20 vs 28 均在 ~2.60 / ~2.99 ms（噪声内 flat）
- **s_sleep(2) 不可改**: 改成 0/1 会让 combine 自旋占满 CU，NO_REDUCE 劣化 ~40%
- **已落地**: `mega_moe_backward_fp8_impl.py` `num_combine_cu=20`（fwd 仍 28）

## dW1/dW2 优化 (2026-07-28, n05-29, P1 + dy_act prep + 重排)

### 改动

1. **P1 forward 预 colwise quant pool_x** — forward clone pool 后立即 `colwise_requant`, 存 `ctx.pool_x_colwise_fp8`; backward triple_prep 跳过 pool 腿 (~0.36 ms)。
2. **`colwise_dy_act_prep_mxfp8_grouped_flydsl`** — dy requant + act quant 单 JIT (P1 路径); triple_prep 在 `pool_x_colwise_fp8` 给定时不跑 pool 腿。
3. **Backward 重排** — `dW2 → dW1 → STEP3` (dW1 与 STEP3 独立, 逻辑更清晰; 单 stream wall time 不变)。
4. **Stage bench 更新** — fc2/fc1_wgrad 走 production fused 路径; fc1 FULL = GEMM-only (dual_quant 共享 prep)。

### 性能 (T=8192 EP8 load_balanced, n05-29)

| 指标 | 优化前 | 优化后 | Δ |
|---|---|---|---|
| **e2e bwd-only** | ~8.86 ms | **8.734 ms** | **−0.13 ms** |
| **e2e fwd+bwd** | ~14.25–14.71 ms | **14.117 ms** | **−0.13–0.59 ms** |
| fc2_wgrad FULL | ~1.74 ms | **1.54 ms** | −0.20 ms (dy_act 0.48 + GEMM 1.05) |
| fc1_wgrad GEMM | ~2.04 ms | **~2.00 ms** | flat (GEMM 瓶颈) |
| triple_prep | ~0.80 ms | **0.78 ms** (full) / **0.44 ms** (dy_act P1) | pool 腿移到 fwd |

- P1 把 pool_x colwise (~0.36 ms) 从 backward 挪到 forward; **fwd+bwd 净零和**, backward 孤立路径省 ~0.36 ms。
- dW2 prep 从 triple(含 pool) → dy_act, 省 ~0.30 ms。
- dW1 GEMM (~2.0 ms) 仍是 wgrad 最大单项; 下一步 ROI = variable-K GEMM autotune / ATT。

复现:
```bash
docker exec xiaoming-dev bash -lc 'cd /perf_apps/xiaoming/MegaMoE && export PYTHONPATH=$PWD
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_bwd_only.py --num-processes 8 --num-tokens 8192 --routing-mode load_balanced
python3 benchmark/ops/training/bench_mega_moe_fp8.py --stage fc2_wgrad --mode load_balanced --num-tokens 8192
python3 benchmark/ops/training/bench_mega_moe_fp8.py --stage fc1_wgrad --mode load_balanced --num-tokens 8192
python3 benchmark/ops/bench_triple_prep_fp8.py'
```

```bash
# 复现 overlap sweep
PT_COMBINE_NO_REDUCE=1 python benchmark/ops/bench_step3_fp8.py --combine-cu 20 --breakdown
```

## dW1 variable-K wgrad GEMM — ATT + autotune (2026-07-28, n05-29)

### Autotune sweep (synthetic, M_pad=69632, G=32)

| shape | OUT_M×OUT_N | base (256,256,4,8,0) | best cfg | best ms | Δ |
|---|---|---|---|---|---|
| **dW1** | 4096×7168 | 2.062 ms / 1983 TF | **(256,256,2,1,0)** | **1.859 ms / 2199 TF** | **−10%** |
| **dW2** | 7168×2048 | 1.038 ms / 1969 TF | **(256,256,8,4,0)** | **1.004 ms / 2036 TF** | **−3%** |

- bm=128 候选 **byte-exact FAIL** → 已从候选列表移除。
- 已落地: `_GWG_WGRAD_DEFAULT_CFG = (256,256,2,1,0)`; 候选重排 (dW1 ref 在前); autotune 仍 per-shape 选最优。

### ATT hotspot (`kernel_grouped_mxfp8_wgrad`, cfg=256,256,2,1,0)

| 类型 | 占比 | Top line |
|---|---|---|
| **barrier** | 34% | `mxfp8_grouped_kernel.py:1327` (K-chunk `_wgrad_ssa_chunk` loop) |
| **MFMA/FMA stall** | 26% | `gemm_helper.py:349` (MFMA 等数据) |
| **VMEM-load** | 18% | `gemm_helper.py:194` |
| LDS | 6% | `gemm_helper.py:218` |
| VMEM-wait | 8% | chunk 边界 `s_waitcnt` |

- **Occupancy**: VGPR 248/512 → **2 waves/SIMD** (VGPR-bound, 非 LDS-bound); 3 waves 需 combined VGPR ≤170。
- **81% total stall** — 主因 K-pipeline chunk 间 `s_barrier` (1327/1365) + MFMA 等 VMEM/LDS。

### 优化方向 (按 ROI / 风险)

1. **已做**: autotune 默认/候选 → dW1 GEMM ~2.04→**~1.97 ms** (stage bench)。
2. **VMEM prefetch 重排 (2026-07-28, 无效)**: 提前 issue `a_g2s`/`ScaleS2R` load + prologue 补 `a_next1` → isolated GEMM **1.86→1.93 ms (+4%)**, e2e flat; **已回滚**。pipeline 已是 distance-2 双缓冲，compiler 调度已饱和。
3. **asm_mma inline MFMA (无效)**: `MfmaScale16x16x128(asm_mma=True)` → FlyDSL 编译 **"not a valid operand"**; mxfp8 scaled wgrad 与 tensorwise wgrad 路径不兼容; **未落地**。
4. **K-chunk barrier 减 sync**: `:1327` 占 29% ATT stall, 改 pipeline 同步 **高风险** (同 fc1_dgrad barrier 结论)。
5. **VGPR 压到 3 waves / LDS 4→2 buffer**: 历史结论 — 暂缓。

复现 ATT / sweep:
```bash
# cfg sweep
python benchmark/ops/bench_dw1_wgrad_sweep.py
# ATT trace (single GPU)
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PT_MXGG_AUTOTUNE=0 \\
  python benchmark/ops/bench_dw1_gemm_trace.py --cfg 256,256,2,1,0
# hotspot
python .cursor/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py \\
  /tmp/dw1_att_trace/ui_output_agent_*_dispatch_* --topk 15 --mode both
```

