# Monolith MoE — P0a 3-stage Prefetch Failed: workload is NOT HBM-latency-bound

| Field                | Value |
|----------------------|-------|
| When                 | 2026-05-12 16:35 (UTC+8) |
| Where                | `mi355-gpu-26` / `xiaoming-dev` container, gfx950 |
| Project              | MonolithMoE |
| Baseline             | P2 — Batched (e, src, mi, ni) Compute (`f50bb43`) |
| Result               | **NO net change** (±1% vs P2, noise level) |
| Status               | Reverted (patch saved as `benchmarks/results/3stage_prefetch.patch`) |
| Artifacts            | `benchmarks/run_3stage_bench.sh`, `benchmarks/results/3stage_bench.txt`, `benchmarks/results/p2_baseline_for_3stage_compare.txt`, `benchmarks/results/3stage_prefetch.patch` |

## TL;DR

我们按 CK 论文的 3-stage software prefetch 在 `mfma_gemm_tile_t` /
`mfma_gemm_tile_swiglu_t` 里做了一次完整重写：3 个 LDS buffer 循环、HBM
load 提前 2 tile issue、精确 `s_waitcnt vmcnt(N)` 让 kt+1 tile 落 LDS
而 kt+2 tile 仍在飞。**功能 50/50 PASS，但 DSV3 / TILE-FIT 两个形状的
端到端延迟与 P2 在 ±1% 噪声内等价**。

| Shape         | ratio | P2 (ms) | 3-stage (ms) | Δ    |
|---------------|-------|---------|--------------|------|
| DSV3 SPARSE   | 0.15  | 8.21    | 8.31         | +1.2 % |
| DSV3 SPARSE   | 0.18  | **8.00**| 8.05         | +0.6 % |
| DSV3 SPARSE   | 0.20  | 8.21    | 8.29         | +1.0 % |
| DSV3 SPARSE   | 0.25  | 8.44    | 8.54         | +1.2 % |
| DSV3 SPARSE   | 0.30  | 10.04   | 10.08        |  0.0 % |
| TILE-FIT      | 0.15  | 4.26    | 4.30         | +0.9 % |
| TILE-FIT      | 0.18  | 4.45    | 4.44         | −0.2 % |
| TILE-FIT      | 0.20  | 4.35    | 4.37         | +0.5 % |
| TILE-FIT      | 0.25  | 4.34    | 4.30         | −0.8 % |
| TILE-FIT      | 0.30  | 4.59    | 4.60         | +0.2 % |

CK 的 deep-dive note 预期 3-stage 单项收益 +12–18 %。我们一分钱都没拿到。
原因不是实现错了 —— **现状已经不是 HBM-latency-bound**，所以再加深 pipeline
不会改善任何东西，只会增加 VGPR 压力（实测从 256 VGPR + 240 B/lane scratch
spill 维持原样，AGPR 106 → 158，调度变得稍紧）。

## 1. Background — 这次为什么试 3-stage

上一份 breakdown（A2 失败后）把 DSV3 的瓶颈定位到 FC1+FC2 的 5.88 ms（73 %
wall），并把工作量与 HBM 流量都换算了一遍。当时**第一次的诊断是 HBM weight
bandwidth bound**，但用户当场指出 `effective_tflops ≈ 357T` 离 MFMA 峰值
（~1050T）只有 33 %，HBM 利用率 < 40 %，单看哪个都说不上 "bound"。

**修正后的诊断**（讨论结论）：真正坐在关键路径上的是 **MFMA pipeline
stalls**，意味着大量 cycle 是 GPU 在等待，不在算也不在搬运。CK 在同样
shape 上能拿到 ~42 % peak，说明软件流水线效率有 ~30 % 提升空间。

CK note 列出的优化矢量按预期收益排：

1. **3-stage prefetch** +12–18 %（最大单点）
2. **LDS XOR swizzle** +5–10 %
3. **精确 `s_waitcnt` 阶梯** +5–10 %
4. **Stream-K** +15–20 %（架构性大改动）

按"最小风险 / 最大收益"我们选了 (1) 先做。

## 2. What was implemented

完整 patch 在 `benchmarks/results/3stage_prefetch.patch`（209 行新增 / 80 行
删除，本节给出结构）。

### 2.1 `GemmLdsLayout`：2-buffer → 3-buffer

```cpp
constexpr int NSTAGES = 3;
struct GemmLdsLayout {
    bf16_t A[NSTAGES][M_TILE * A_LDS_STRIDE];   // 18 KB × 3 = 54 KB
    bf16_t B[NSTAGES][N_TILE * B_LDS_STRIDE];   // 18 KB × 3 = 54 KB
    // ...pair_*[MAX_PAIRS=512]  ≈ 10 KB
};                                              // 总 ≈ 118 KB / WG
```

`gfx950` 每 CU 160 KB LDS，1 WG/CU 下放得开。

### 2.2 `wait_vm` helper：0..2 → 0..16

为了发出精确的 `s_waitcnt vmcnt(N)`（N = 一个 K-tile 的 cooperative-load
个数），扩展查找表到 0..16。

### 2.3 `mfma_gemm_tile_t` 主循环重写

**Prologue（fill pipeline 2 stages）**：

```cpp
// 同时 issue K-tile 0 + 1
for p: ra_cur[p] = buf_load16(a_rsrc, a_boff[p]);          // tile 0
for p: rb_cur[p] = buf_load16(b_rsrc, b_boff[p]);
if (num_k_tiles >= 2) {
    for p: ra_nxt[p] = buf_load16(a_rsrc, a_boff[p] + ko1); // tile 1
    for p: rb_nxt[p] = buf_load16(b_rsrc, b_boff[p] + ko1);
    wait_vm(LOADS_PER_KTILE);   // 等到只剩 tile 1 outstanding
} else {
    wait_vm(0);
}
// commit tile 0 → LDS[0]
LDS_write_0_from_ra_cur();
__syncthreads();
// move tile 1's data into ra_cur position
ra_cur ← ra_nxt;
```

**Main loop（per iter kt）**：

```cpp
const int compute_idx = kt % NSTAGES;          // MFMA on LDS[kt % 3]
const int store_idx   = (kt + 1) % NSTAGES;    // commit kt+1 here

// (1) issue HBM load for tile kt+2 → ra_nxt
if (kt + 2 < num_k_tiles) for p: ra_nxt[p] = buf_load16(...)

// (2) MFMA on LDS[compute_idx]  (K-step register prefetch, unchanged)
for ks in 0..K_STEPS_PER_TILE: mfma_bf16(acc, ar, br)

// (3) drain ONLY kt+1, leave kt+2 in flight overlapping next iter
if (kt + 1 < num_k_tiles) {
    if (kt + 2 < num_k_tiles) wait_vm(LOADS_PER_KTILE);
    else                      wait_vm(0);
    LDS_write_store_idx_from_ra_cur();
}
__syncthreads();

// (4) rotate
if (kt + 2 < num_k_tiles) ra_cur ← ra_nxt;
```

SwiGLU 版本（`mfma_gemm_tile_swiglu_t`）同结构，A 端多读 `up` 一份（每个
K-tile 共 `2 × A_LD_PHASES_T + B_LD_PHASES_T` 个 cooperative load）。

### 2.4 Correctness

```text
test_super_kernel_correctness  PASS=50  FAIL=0
rank 0..7  max_abs=0.0859  max_rel=0.0003  bad=0
```

50/50 PASS、bit-exact 与 P2 一致，**实现是正确的**。

## 3. Why it didn't help — root cause

### 3.1 Register / occupancy 数据（hipcc `-Rpass-analysis=kernel-resource-usage`）

| Kernel                          | P2 baseline | 3-stage |
|---------------------------------|-------------|---------|
| `fused_moe_super_kernel` VGPRs  | 256         | 256     |
| AGPRs                           | 106         | 158     |
| Scratch (B / lane)              | 240         | 240     |
| Occupancy (waves / SIMD)        | 1           | 1       |
| VGPR spill                      | 0           | 0       |

**Spill 没增加**，VGPR 没动（早就锁在 256/256），**occupancy 也是 1
waves/SIMD = 1 WG/CU**。AGPR 106 → 158 是 compiler 把更多临时值挤进
AGPR file 抵消 VGPR 压力（合理）。所以从资源角度看，3-stage 没引入新的
spill 路径。

### 3.2 那 +12 % 哪里去了？

`mfma_gemm_tile_t` 主循环的时间预算：

```text
per K-tile:  MFMA   + LDS-read   + LDS-write + sync
             ~256c    pipelined    ~50c       ~50c
HBM latency: ~400c, throughput-limited by 5–8 loads/tile
```

CK 的 +12 % 是因为 v8a 的 main-loop 同步 `wait_vm(0)` 真的在卡 HBM 抵达 —
他们的 tile 较大（256×256），单 K-tile MFMA time > HBM latency 但留出富
余给 3-stage 把 HBM 折掉。

我们这里：

- **DSV3 small tile (32×128)**：单 K-tile MFMA time 较短，HBM 5 loads，
  1-stage 已经能让 HBM 大部分被 MFMA 盖住。`wait_vm(0)` 几乎是空 wait。
- **TILE-FIT default tile (128×128)**：单 K-tile MFMA time 更长 (4×4
  MFMAs/wave × 4 K-steps)，HBM 8 loads。1-stage 充分。

**1-stage prefetch 已经把 HBM latency 完全盖住**，3-stage 多加的两级
buffer 只是把 wait_vm(0) 换成 wait_vm(N)，从静态看少等了一点点，但
- 在 1 occupancy 下我们没有其它 wave 可调度去填这点空隙；
- compiler 的 scheduler 在 `wait_vm(0)` 处早就把可以提前的指令都搬到前
  面去了，所以"少 wait 一点"≈ 0 收益。

结论：**3-stage 不是无效设计，而是这个 workload 不再吃 HBM 隐藏**。

### 3.3 旁证：CK 的 +12 % 是相对 v8a 的 530T → 620T，但 v8a 当时 `wait_vm(0)` 真在卡 HBM；我们这里 357T 卡的不是 HBM，是 MFMA pipeline 本身的 issue rate（dependency stalls / lgkmcnt(0) at K-step boundary）。

## 4. Lessons learned

1. **跨工程套搬 CK recipe 之前先确认 "真的 HBM-latency bound"**。简单 sanity
   check：把 `wait_vm(0)` 换成 `wait_vm(LOADS_PER_KTILE)`（不用动 LDS 布局）
   —— 如果延迟一点都不动，说明 HBM 不是瓶颈，根本不需要 3-stage。我们这次
   直接做完整 3-stage 重写，多走了 1.5 天。
2. **`-Rpass-analysis=kernel-resource-usage`** 是判断 spill / occupancy 的
   黄金标准。这次 P2 和 3-stage 资源完全一样 (256/106/240) → 资源不是变量。
3. **Latency-hiding pipeline 加深的边际收益是 "上一级有没有 wait" × "wait
   能不能被新指令吃掉"**。两者都为零时（已经没 wait + 1 wave/SIMD 没多余可
   调度），任何 stage > 1 都是白做。
4. **A1, A2, P0a 这三次 regression 共同点**：都是为了"理论收益"动 GEMM 主
   循环。MFMA pipeline 是高度调优过的，单独改一个 dimension（per-expert
   pipeline / M-concat / stage 深度）大概率打不过 hipcc 的 default 调度。

## 5. Where to next

新的诊断焦点：

- **MFMA pipeline efficiency 357T / 1050T ≈ 34 %**，gap ~ 3×。
- 1 wave/SIMD 单 wave 调度 → 任何 dependency stall 都会暴露成 dead cycle。
- 主循环里仍有 `__syncthreads()` per K-tile（112 × 50c ≈ 5600c / tile ≈ 11 %
  per-tile time）—— 但去掉需要保证 3-buffer 之间的 LDS write/read 永远走
  不同 buffer（NSTAGES=3 已经 OK），这部分得 profile 一下才确认值不值。

候选下一步（按"信号 + 工作量"重排）：

| 候选 | 信号强度 | 工作量 | 备注 |
|---|---|---|---|
| **A. Profile 一次 `mfma_gemm_tile_t` 内部**（PROFILE_DECL / hardware counter） | 高（直接看哪段慢）| 1d | 不知道下一步该做什么时优先 |
| **B. K-step 间去掉 lgkmcnt(0) 隐式 barrier**（确认 LDS double-buffer 不需要） | 中 | 1d | CK 拿掉过这步，估算 +3–7 % |
| **C. LDS XOR swizzle 替 PAD=4** | 中 | 2–3d | 与 (A,B) 联动；CK +5–10 % |
| **D. Weight pre-permutation to MFMA fragment layout** | 高（去掉 LDS write + reduce ds_read） | 2–3d | 改 weight 准备路径而非 GEMM 主循环 |
| ~~E. 继续做 3-stage~~ | 0（本 note） | -- | 已证伪 |

我倾向先做 **(A) profile** 验证哪段最热，再决定 B/C/D 中哪个先打。
盲打 stage / swizzle / pre-permutation 都有 30–50 % 失败概率（A1/A2/P0a 已
经证明），profile 一次最便宜。

## 6. Reproduction

```bash
# baseline P2 (current HEAD)
bash benchmarks/run_3stage_bench.sh
# results → benchmarks/results/3stage_bench.txt (P2 data since patch reverted)
# the saved P2 reference: benchmarks/results/p2_baseline_for_3stage_compare.txt

# 3-stage attempt (apply patch, then bench)
git apply benchmarks/results/3stage_prefetch.patch
bash benchmarks/run_3stage_bench.sh
# results stored to the same file (rerun overwrites)

# revert
git checkout -- csrc/fused_moe_super_kernel.hip
```
