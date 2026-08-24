# Mega MoE fp8 反向瓶颈拆解 + combine top-k reduce 向量化（T=8192）@ MI355X

> **用途**: 在 `2026-08-01_1440` 的回归基础上，把**反向**的 9.07 ms 拆到 stage 级 + 重叠级，定位可回收的结构性浪费，
> 并完成第一轮优化（combine top-k reduce 的 payload load 从 4 B/lane 拓宽到 16 B/lane）。
> **Where**: `smci355-ccs-aus-n04-33`（MI355X / gfx950 ×8），容器 `xiaoming-dev`，flydsl 0.2.4，job 23835。
> **配置**: DeepSeek-V3，EP8，T=8192，H=7168，I=2048，E=256，topk=8，BM=BN=256。
> **Campaign**: `agent/workspace/mega_moe_combine_reduce_flydsl_gfx950_20260801/`（round-1 基线 / round-2 已接受）。

---

## Part A — 反向瓶颈账（生产口径，load_balanced）

| 阶段 | ms | 占比 | 实测 TFLOPS |
|---|---|---|---|
| STEP3：fc1 dgrad + combine PUSH + reduce | 2.99 | 33.0% | 1374 |
| dW1 variable-K wgrad（纯 GEMM，requant 已由 P1 移到 forward） | 2.08 | 22.9% | 1968 |
| L2 dgrad：dispatch(dy) + fc2 GEMM | 1.67 | 18.4% | 1225 |
| dW2：dual quant 0.49 + GEMM 1.06 | 1.55 | 17.1% | GEMM 1925 |
| SwiGLU^T + rowcol dual quant（残差，未单独测） | ~0.61 | 6.7% | — |
| colwise meta D2H（每 step 一次） | 0.18 | 2.0% | — |
| **合计** | **9.07** | | |

账是闭合的（bwd-only 实测 9.07 ms）。

### roofline

四个 GEMM 合计 **12.27 TFLOP/rank**，MXFP8 密集峰值 5000 TFLOPS → **2.45 ms 理论地板**；实际 9.07 ms = **峰值 27%**。
拆成两个池子：

- **GEMM 本身 6.31 ms**，跑在 1877–1968 TFLOPS = **~38% 峰值**
- **非 GEMM 开销 2.76 ms**：通信暴露 ~0.98、量化 ~0.79、reduce 0.42、SwiGLU^T ~0.28、meta 0.18

### 关键：STEP3 的重叠拆解（`bench_step3_fp8.py --breakdown`, cc=28）

| leg | ms |
|---|---|
| full（GEMM + PUSH + reduce + gate） | 2.977 |
| GEMM_ONLY | 2.178 |
| PUSH_ONLY | 1.969 |
| NO_REDUCE（GEMM ∥ PUSH） | 2.553 |
| **reduce+gate = full − NO_REDUCE** | **0.424** |

串行和 4.147 → 重叠后 2.553（省 1.594，1.62× vs 理想 1.90×）。结论：**0.375 ms 的 PUSH 没藏住**，
另有 **0.424 ms 的 reduce+gate 是完全串行的尾巴**。

### 一并修掉的 bench bug

`--stage fc1_wgrad` 之前一跑就 assert 挂掉：bench 把 rowwise 的 `pool_x` 传给
`_mxfp8_variable_k_wgrad_dw1`，而该函数默认 `pool_x_is_colwise=True`。补上 `pool_x_is_colwise=False` 后才拿到 dW1 的数。

---

## Part B — Round 2：combine top-k reduce 向量化（ACCEPTED）

### 判据先于动手

reduce 腿的真实流量：payload 读 8192·8·7168 B = 469 MB，scale 读 ~15 MB，输出写 8192·7168·2 B = 117 MB，
合计 ~601 MB。0.424 ms → **1.42 TB/s，仅 HBM 峰值 8 TB/s 的 18%**。**这是发射受限，不是带宽受限**，
所以拓宽 load 是第一顺位。反例参照：7/29 的 `colwise_requant` campaign 在 ~35% 峰值上打了 12 轮全 REJECT。

### 单一改动

`_make_topk_reduce_fp8`（`grouped_gemm_combine_fp8_kernel.py`）——**前向 L2 combine 和反向 STEP3 共用**这一段：

- payload `buffer_load(vec_width=1)`（4 B/lane）→ `vec_width=4`（16 B/lane，b128），
  lane 索引 `lane + k*64` → `lane*4 + k*256`（一个 warp step 覆盖 1024 B 连续）。
- 4 个 4 对齐的 i32 word = 16 个 fp8 落在**同一个 E8M0 32 块**内，所以 `w//32` 和 `8*((w//8)%4)` 恒定，
  **一次 scale load + 一个 shift 服务 4 个 word**（scale load 也降到 1/4）。
- store 侧**故意不动**（仍是 4 次 `bf16_v4`），保证本轮只有 load 一个变量。

同一文件里的 PUSH 路径**本来就在用 b128**，这个不对称纯属历史遗留。

### 结果

| 指标 | 基线 | round 2 | Δ |
|---|---|---|---|
| **reduce+gate 腿** | **0.424** | **0.135** | **−0.289（−68%）** |
| STEP3 full | 2.977 | 2.697 | −0.280 |
| 前向 L2 combine | 1.912 | 1.870 | −0.042 |
| **e2e fwd+bwd（LB）** | **14.640** | **14.346** | **−0.294（−2.01%）** |
| **e2e fwd+bwd（RR，skew）** | **13.814** | **13.465** | **−0.349（−2.53%）** |
| bwd-only（LB / RR） | 9.080 / 8.502 | 8.774 / 8.215 | −0.306 / −0.287 |

reduce 腿现在是 601 MB / 0.135 ms = **4.45 TB/s = 56% 峰值**（原 18%）。

**对照组全部持平** —— NO_REDUCE 2.553→2.562、GEMM_ONLY 2.178→2.127、PUSH_ONLY 1.969→1.980，
证明改动确实只落在 reduce 路径上。

**正确性逐位不变** —— 前向 `y_norm` 6.892e+02、反向 `dx_norm` 9.888e+02 与基线精确一致；
e2e 梯度 SNR（T=2048）dx 21.9 / d_topk_w 23.1 / dW1 19.5 / dW2 19.7 dB，与基线**完全相同**，PASS。

**skew 稳健性**（Rule 11）—— round_robin（all-to-few 病态分布）的收益比 load_balanced **更大**（−2.53% vs −2.01%），
不是靠均匀分布吃出来的。bucket = K1（kernel 内部），无缓存，收益 1:1 transfer 到真实训练。

### 为什么前向只省 0.042 而反向省 0.289

同一段 reduce，反向 STEP3 里它是**完全串行的尾巴**，缩短多少就直接落在关键路径上；
前向 L2 里它已经和 GEMM/PUSH 重叠得不错。**融合 kernel 里某一腿的收益取决于重叠情况，不取决于它自身大小** ——
预测前必须先做重叠拆解。

---

---

## Part C — Round 3/4：STEP3 未隐藏 PUSH，两轮均 ROLLBACK（但拿到了关键诊断）

### Round 3：`num_combine_cu` 重扫 → ROLLBACK

假设：round-2 让 tail-reduce（由 combine CU 承担）便宜了 3.1 倍，所以 CU 配比最优点应该往更多 combine 偏。

结果**平的**：cc=20/24/28 → 2.699/2.697/2.702 ms（噪声内），32 以上单调变差（2.720/2.817/2.853）。

**诊断（这轮真正的产出）**：

| cc | GEMM_ONLY | PUSH_ONLY | NO_REDUCE（重叠后） |
|---|---|---|---|
| 20 | 2.183 | 2.348 | **2.562** |
| 28 | 2.127 | 1.980 | **2.562** |
| 40 | 2.192 | 1.695 | **2.680** |

PUSH 腿单独看对 CU **高度敏感**（2.348 → 1.980 → 1.695），但重叠后被钉在 ~2.56。把短腿加速 0.37 ms 换来**零**收益
→ 这是**共享内存/fabric 带宽饱和**，不是调度或 CU 分配问题。杠杆是**减少字节数**。

> 可复用判据：`overlap vs max(leg) < 1` **不能**单独证明调度差。先把短腿单独加速，看重叠时间是否变化；
> 不变就是带宽受限，重扫 CU 配比是浪费一轮。

### Round 4：打开 L2 tile swizzle `PT_COMBINE_GROUP_M` → ROLLBACK

顺着"减流量"：STEP3 的 GEMM 有 28 个列 tile（A 最坏重读 28 次），而 swizzle 默认是关的。

| GROUP_M | full | GEMM_ONLY | NO_REDUCE |
|---|---|---|---|
| **0（关）** | 2.693 | **2.112** | **2.510** |
| 2 | 2.657 | 2.208 | 2.573 |
| 4 | 2.668 | 2.196 | 2.572 |
| 8 | 2.838 | 2.254 | 2.746 |
| 16 | 2.724 | 2.303 | 2.654 |

GEMM 腿**单调变差**（2.112 → 2.303），信号明确。gm=2/4 的 `full` 看似略好属于 ~0.05 ms 噪声，且被两条底层腿否掉。

**根因**：这是**分组（per-expert）GEMM**。经典 GROUP_M swizzle 假设一组行块共享同一个 B 矩阵；
但这里连续的 `block_m` **属于不同 expert**，一个 GROUP_M 组会横跨多个 per-expert 权重 slab，
于是**把 L2 打乱而不是摊薄**——组越大越差。稠密 GEMM 的 swizzle 启发式要移植到分组 GEMM 必须做成 group-aware。

### 状态

工作 kernel 仍是 round-2（已校验与快照一致）。连续两轮 rollback → 按 iteration_rules Rule 7，
下一次动这个目标前**必须先做完整 rocprofv3 ATT + rocprof-compute 重剖面**，确认到底是谁饱和：
HBM 读（GEMM 算子）、XGMI 写（PUSH）、还是 L2。

剖面之后才评估的候选：group-aware swizzle；提高 STEP3 的 `BLOCK_N`（28 个列 tile 偏多）；
**删掉 L2Y 本地往返**（epilogue 写 469 MB，PUSH 再读 469 MB，共 ~938 MB，是已识别的最大单项流量，
但与现有 3-role 设计初衷冲突）。

---

## 下一步排序（反向现在 ~8.78 ms）

1. **STEP3 未隐藏的 PUSH ~0.44 ms** —— `overlap vs max(GEMM,PUSH)` 只有 0.83×（2.562 vs 理想 2.127）。
2. **dW1 GEMM 2.08 ms @ 1968 TFLOPS（39% 峰值）** —— 单项最大，但历史 4 轮全 rollback（VGPR-bound，2 waves/SIMD）。
3. **L2 dgrad 未隐藏通信 ~0.6 ms** —— 需要先给 dispatch stage 补一个 `--breakdown` 等价拆解。
4. **dW2 dual quant 0.49 ms** —— requant 融进 L2-dgrad GEMM 热循环，R1 已评为高风险低回报，暂缓。

同 kernel 内最便宜的后续：把 reduce 的 **store** 侧从 `bf16_v4`（8 B）拓到 16 B —— load 便宜 4× 之后，
store 在剩余的 0.135 ms 里占比变大了。

## 复现

```bash
docker exec xiaoming-dev bash -lc 'cd /perf_apps/xiaoming/MegaMoE && export PYTHONPATH=$PWD
# 反向分阶段
T=8192 MODE=load_balanced STAGES="dispatch_fc2_dgrad fc2_wgrad fc1_wgrad fc1_dgrad_combine" \
  bash benchmark/ops/training/run_stages_fp8.sh
# STEP3 重叠拆解（含 reduce 腿）
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_step3_fp8.py \
  --num-processes 8 --num-tokens 8192 --breakdown
# 主指标
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_bwd_only.py \
  --num-processes 8 --num-tokens 8192 --routing-mode both --only fp8 --warmup 8 --iters 25
# A/B 回退到标量 load
PT_COMBINE_REDUCE_VW=1 MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_step3_fp8.py \
  --num-processes 8 --num-tokens 8192 --breakdown'
```
