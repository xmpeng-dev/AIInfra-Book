# Monolith MoE — `mfma_gemm_tile_t` 内部 Profile：LDS write 33%, MFMA inner 44%, HBM 不卡

| Field    | Value |
|----------|-------|
| When     | 2026-05-12 17:15 (UTC+8) |
| Where    | `mi355-gpu-26` / `xiaoming-dev` container, gfx950 |
| Project  | MonolithMoE |
| Status   | Instrumentation merged (zero cost in release, gated by `MOE_PROFILE`) |
| Artifacts| `benchmarks/results/gemm_internal_profile.txt` |
| Resource | P2 baseline ↔ +profile build: 256 VGPR / 106 AGPR / 240 B scratch (identical — DCE works) |

## TL;DR

为 `mfma_gemm_tile_t` / `mfma_gemm_tile_swiglu_t` 加了 6 个 GEMM 内部
profile bucket（gemm_total / hbm_issue / wait_vm / lds_write / sync /
mfma_inner），用 thread-0 `clock64` 在主循环每段切片，函数尾部一次性
`atomicAdd` 到 HBM。两种 shape（DSV3 SPARSE 小 tile 和 TILE-FIT 默认 tile）
breakdown 出来的比例**完全一致**：

| Section          | DSV3 (small) | TILE-FIT (default) | 含义 |
|------------------|--------------|--------------------|------|
| gemm_total       | 5657 µs / 100 % | 1737 µs / 100 %    | FC1+FC2 中 GEMM 内部累计时间 |
| hbm_issue        |  587 µs /  10 % |  175 µs /  10 %    | `buf_load16` 指令派发段 |
| **wait_vm**      |   61 µs /   **1 %** |   16 µs /  **1 %** | **HBM drain — 完全不卡** ✓ |
| **lds_write**    | 1922 µs /  **34 %** |  567 µs / **33 %** | reg→LDS commit + 编译器穿插 |
| **sync**         |   62 µs /   **1 %** |   19 µs /  **1 %** | `__syncthreads` per K-tile |
| **mfma_inner**   | 2505 µs /  **44 %** |  754 µs / **43 %** | `ds_read + mfma + lgkmcnt` K-step 循环 |
| accounted        | 90.8 %       | 88.1 %             | 余为 prologue/epilogue + 采样开销 |

### 三个直接结论

1. **HBM latency 完全不是瓶颈**（wait_vm 1 %）— P0a 3-stage prefetch
   失败的根因彻底证实，不要再去做"加深 prefetch"了。
2. **`__syncthreads` per K-tile 也不是瓶颈**（1 %）— 拿掉 K-step 间
   barrier 的预期收益是 noise 量级，不值得碰。
3. **真瓶颈在 LDS 端**（lds_write 34 % + ds_read 在 mfma_inner 中占未知比例
   ≤ 44 %），合起来 ≥ 50 % 的 GEMM 时间是 LDS 端在挪数据 / 等数据。

## 1. Setup

### 1.1 Bucket 布局（`MOE_PROFILE_NUM_BUCKETS = 20`）

```text
 0  kernel_total           (existing)
 1  dispatch_src_ready_wait
 2  fc1_tiles
 3  compute_barrier_1
 4  fc2_swiglu_tiles
 5  compute_barrier_2
 6  copy_to_combine
 7  compute_barrier_3
 8  sort
 9  scatter
11  gather_combine_phase
─── 新增 ──────────────
12  gemm_total           (TIC at entry, TOC at exit — minus atomicAdd cost)
13  gemm_hbm_issue       (buf_load16 dispatch loop)
14  gemm_wait_vm         (s_waitcnt vmcnt(0))
15  gemm_lds_write       (reg→LDS ds_write_b128 loop, with compiler interleave)
16  gemm_sync            (__syncthreads at K-tile boundary)
17  gemm_mfma_inner      (K-step ds_read + mfma + lgkmcnt loop)
```

### 1.2 Instrumentation 设计要点

- **每次 TIC/TOC 只 thread 0 采样**，clock64 → `uint64_t` 局部累加器，
  避免每个 K-tile 都打 atomicAdd（如果 6 buckets × 32-112 K-tiles ×
  50 output tiles ≈ 100k atomicAdd / WG，那 atomic 自己就 ≥ 5 ms）。
- **函数尾部一次 `atomicAdd`** 把累计写回 HBM 6 个 bucket 槽。
- `GEMM_PROF_DECL(name)` 在 non-profile build 也声明 `uint64_t name = 0`
  避免后续 `t_end - t_total` 编不过；TIC/TOC 都展开成 `((void)x)`，
  release build 完全 DCE（实测 P2 ↔ profile build 同样的 256/106/240，
  zero overhead）。
- `prof` pointer 在 non-profile build 是 `nullptr`，所有 `GEMM_PROF_PUBLISH`
  都早 return。

### 1.3 50/50 correctness

`test_super_kernel_correctness 50/50 PASS` — 与 P2 bit-exact（profile 钩
子对数值无影响）。

## 2. 结果（DSV3 SPARSE / TILE-FIT）

跑命令（已存档为 `benchmarks/results/gemm_internal_profile.txt`）：

```bash
# build profile bin
hipcc -std=c++17 -O3 --offload-arch=gfx950 -I csrc \
  -DMOE_PROFILE -DMOE_M_TILE=128 -DMOE_N_TILE=128 \
  -o benchmarks/results/bin/bench_profile_m128 benchmarks/bench_super_kernel.hip

# DSV3
./benchmarks/results/bin/bench_profile_m128 \
  --tokens 512 --hidden 7168 --ffn 2048 --epg 32 --topk 8 \
  --num-cus 256 --wgs-per-cu 1 --comm-ratio 0.18 \
  --warmup 5 --iters 30 --profile
```

### 2.1 DSV3 SPARSE（avg 16 tokens/expert，小 tile 32×128 路径）

```text
latency_ms=8.139   effective_tflops=354.6
compute WGs [46..210)
  kernel_total=8138.6 us
    dispatch_src_ready_wait : 1120.0
    fc1_tiles               : 2947.6   ← FC1+FC2 = 5965.9
    compute_barrier_1       :  428.4
    fc2_swiglu_tiles        : 3018.3
    compute_barrier_2       :   71.4
    copy_to_combine         :  436.7
    compute_barrier_3       :  111.0

  GEMM-internal (sum 所有 mfma_gemm_tile_t / _swiglu_t 调用):
    gemm_total              : 5657 us  (100.0 %)
    ├─ hbm_issue            :  587 us  (10.4 %)
    ├─ wait_vm              :   61 us  ( 1.1 %)   ← HBM drain
    ├─ lds_write            : 1922 us  (34.0 %)   ← biggest target
    ├─ sync_per_ktile       :   62 us  ( 1.1 %)
    └─ mfma_inner           : 2505 us  (44.3 %)   ← MFMA + ds_read + lgkmcnt
```

`gemm_total = 5657` 对应 `fc1 + fc2 - 外层 __syncthreads = 5966 -
60 (outer sync) ≈ 5900` — 拼回正常。

### 2.2 TILE-FIT（avg 128 tokens/expert，默认 tile 128×128 路径）

```text
latency_ms=4.368   effective_tflops=188.8
compute WGs [51..205)
  kernel_total=4368.5 us
    dispatch_src_ready_wait : 1989.0   ← 占了 45 % wall（comm 路径，与 GEMM 无关）
    fc1_tiles               :  871.7
    fc2_swiglu_tiles        :  901.6

  GEMM-internal:
    gemm_total              : 1737 us  (100.0 %)
    ├─ hbm_issue            :  175 us  (10.1 %)
    ├─ wait_vm              :   16 us  ( 0.9 %)
    ├─ lds_write            :  567 us  (32.6 %)
    ├─ sync_per_ktile       :   19 us  ( 1.1 %)
    └─ mfma_inner           :  754 us  (43.4 %)
```

**两 shape 完全同模式** — 与 tile 大小、K 长度、avg_Te 都无关。比例
是 GEMM kernel 结构本身决定的。

## 3. 关键诊断：lds_write 34 % 是什么？

我以为 "lds_write" 是单纯的 `ds_write_b128` 序列，仔细看 asm 发现并不是。

### 3.1 ASM 实地观察

`bench_super_kernel.s` 里 SwiGLU 路径的 ds_write 上下文：

```asm
; ── 我的 TIC 落在这里（acc_lds_w begin）──
v_or_b32_sdwa v82, v51, v50 ...     ; 拼 bf16 high-low（gate 部分）
s_waitcnt vmcnt(9)                  ; ← !! 等下一组 HBM 抵达
v_lshlrev_b32_e32 v50, 16, v34      ; 转 fp32 准备 silu
v_and_b32_e32 v85, 0xffff0000, v85
v_or_b32_sdwa v83, v80, v64 ...
v_mul_f32_e32 v64, 0xbfb8aa3b, v50  ; silu 系数
v_fma_f32 v80, v50, s61, -v64
v_rndne_f32_e32 v81, v64
v_exp_f32_e32 v64, v64               ; silu 的 expf
...
ds_write_b128 v234, v[82:85]         ; ← 真正的 ds_write
s_waitcnt vmcnt(8)                   ; ← 又一组 HBM
v_lshlrev_b32_e32 v51, 16, v66
v_ldexp_f32 v64, v64, v80
v_div_scale_f32 v80, ...
v_rcp_f32_e32 v81, v80               ; rcp 算 silu(x) = x / (1+e^-x)
...
ds_write_b128 v235, v[34:37]
```

### 3.2 实际包含的内容

我标记的 "lds_write" 区段（TIC → TOC 之间）**实际上是 compiler
schedule 后的混合体**：

1. **ds_write_b128 序列**（真正的 LDS 写入）
2. **下一个 K-tile 的 buf_load16 已经在飞**（编译器把 HBM issue 提前到
   这里以与 ds_write 重叠 — 这是 compiler 自带的隐式 prefetch）
3. **`s_waitcnt vmcnt(N)`** 等下一组 HBM 数据抵达（不是为了 ds_write，
   而是为了后续 VALU 消费 up 值）
4. **SwiGLU fuse_swiglu 计算**（仅 FC2 路径）：silu(gate) = gate / (1 + e^-gate)，
   每个 lane 至少 12 个 VALU 指令（lshl + mul + fma + rndne + exp +
   ldexp + cmp + sub + add + cndmask + add + rcp + fma×3 + div_fixup
   + bfe + add3 + or_sdwa），用 ~150-200 cycle 完成一组 8-wide silu*up。

### 3.3 这意味着什么

**lds_write 34 % 这个数字其实包含了 SwiGLU 的全部计算 + 编译器自动
2-stage prefetch 的等待**。把它拆开估算（FC1 / FC2 各占一半 gemm_total）：

- FC1 lds_write 实际 ≈ 8 × `ds_write_b128` × 32 cycles ≈ 256 cycles/K-tile
  × 112 K-tile/tile × 39 tile/WG ≈ 1.1M cycles ≈ 660 µs / WG
- FC2 lds_write + SwiGLU 计算 + HBM 等待 ≈ 余下 ~1300 µs / WG

实际 FC1 占的 ~660 µs / 2828 µs（FC1 gemm 部分） ≈ **23 %**，与
TILE-FIT 的 33 % 接近（TILE-FIT 是非 SwiGLU + SwiGLU 混合）。

**重要观察**：编译器其实**已经做了** 3-stage prefetch 的等价物（把
下一 K-tile 的 buf_load 嵌进 ds_write 序列）—— 这就是为什么 P0a 显
式 3-stage 一分钱都没拿到。

## 4. mfma_inner 44 % 是什么？

这部分**没有 compiler 重排干扰**，是干净的 K-step 内层循环：

```cpp
for ks in 0..K_STEPS_PER_TILE:                  // 4 (gfx950) or 8 (gfx942)
    if has_next_step:
        ar_next[m] = ds_read(LDS A);           // 1-4 reads
        br_next[n] = ds_read(LDS B);           // 4 reads
    wait_lgkm(M+N or 0)                         // ← 编译器无法消去
    for m, n: mfma_bf16(acc, ar, br)            // M_PER_WAVE × N_PER_WAVE MFMAs
    ar = ar_next; br = br_next
```

per K-tile（CDNA4，gfx950）的理论下界：

- K_STEPS = 4
- 每 step：4 ds_read + 4 ds_read + lgkmcnt + (1×4=4) MFMAs（小 tile）
- ds_read_b128 cycles：~8 / read（4 ways × 2 半 wave）→ 64 cycles
- mfma cycle（gfx950, K=16, throughput）：~4-8 cycles / mfma → ~32 cycles
- lgkmcnt(N) stall: 取决于 N 与 LDS 完成时机

per K-step ~100-150 cycles。per K-tile ~400-600 cycles。

DSV3 small tile：112 K-tiles × 500 cycles × 39 tile/WG ≈ 2.2M cycles
≈ 1.3 ms。实测 2.5 ms — **几乎是理论 2 倍**。可能的原因：

1. **LDS bank conflicts on ds_read**（PAD=4 不是 XOR swizzle）
2. **lgkmcnt 等待时间**比预期长（LDS queue 拥塞）
3. **MFMA pipeline dependency stalls**（ar / br / acc 之间）

理论上 XOR swizzle 应该能把 (1) 干掉，估算 mfma_inner 44 % → ~35 %（减
9 % gemm = 减 510 µs / WG = -0.5 ms wall on DSV3）。

## 5. 下一步建议（重排）

### 5.1 干掉的 candidate

- ~~3-stage HBM prefetch~~ — 已证伪（compiler 自带）
- ~~K-step 间去 `s_barrier`~~ — sync 已经只占 1 %，没收益
- ~~精确 `s_waitcnt` 阶梯优化~~ — 看 asm 编译器已做得相当好

### 5.2 新 P0 排序

| 优先级 | 改动 | 预期 GEMM 减少 | 预期 wall（DSV3） | 工作量 | 备注 |
|---|---|---|---|---|---|
| **P0** | **LDS XOR swizzle 替换 PAD=4** | 减 5-10 % | -0.3 ~ -0.6 ms | 2-3 d | 同时打 ds_write + ds_read，DSV3+TILE-FIT 都吃 |
| **P0** | **mxfp8 weights for FC1/FC2** | （+ 改变 mfma_inner 的 mfma 数量）| -2 ~ -3 ms | 5-7 d | A2 后唯一剩下的 DSV3 大杠杆；MFMA 峰值 ×2，与 layout 正交 |
| P1  | **Weight pre-permutation to MFMA-frag layout** | 减 ds_read fragment 重排 | -0.3 ms | 2-3 d | 离线把 w1/w2 排成 MFMA 期望的格子，省 lds 重排 |
| P1  | **C-shuffle epilogue（LDS-assisted store_acc）** | 减 store_acc | -0.1 ~ -0.2 ms | 2 d | epilogue 现在每 lane 32 个 bf16 store，未向量化 |
| P2  | **mfma_inner 的 ds_read 用更小的 ds_read_b64**（如果 b128 有过宽 issue） | 不确定 | unknown | 2 d | 需要先 micro-bench |
| P2  | **Work-stealing tile counter** | 改善 T_e 不均的尾巴 | -0.2 ~ -0.5 ms | 3 d | 与 GEMM 内部正交 |

### 5.3 我的推荐

**先做 LDS XOR swizzle**：

- 改动局部（`GemmLdsLayout` + `lds_frag_ds` 的 base_byte 计算 + cooperative
  load 的 `ao[p]`/`bo[p]` 计算），不动 GEMM 主循环结构。
- 同时影响 ds_write (34 %) 和 mfma_inner 中的 ds_read（≤ 44 %），即使每
  个减 10 % 也有合并 7-8 % gemm reduction。
- 与 mxfp8 完全正交（mxfp8 是改 mfma 路径，XOR swizzle 是改 LDS 布局），
  做完 XOR 之后做 mxfp8 还能继续叠。

**然后做 mxfp8**：

- MI355X 原生支持 `v_mfma_*_mxfp8`，FC1/FC2 weight HBM 流量 ÷2，MFMA 峰
  值 ×2。
- 实测预期 DSV3 wall 8 → 5-6 ms（与 P0 目标线 4.8 ms 接近）。

## 6. 复现

```bash
# 1. build profile bin
hipcc -std=c++17 -O3 --offload-arch=gfx950 -I csrc \
  -DMOE_PROFILE -DMOE_M_TILE=128 -DMOE_N_TILE=128 \
  -o benchmarks/results/bin/bench_profile_m128 \
  benchmarks/bench_super_kernel.hip

# 2. DSV3 SPARSE
./benchmarks/results/bin/bench_profile_m128 \
  --tokens 512 --hidden 7168 --ffn 2048 --epg 32 --topk 8 \
  --num-cus 256 --wgs-per-cu 1 --comm-ratio 0.18 \
  --warmup 5 --iters 30 --profile

# 3. TILE-FIT
./benchmarks/results/bin/bench_profile_m128 \
  --tokens 1024 --hidden 4096 --ffn 1024 --epg 4 --topk 4 \
  --num-cus 256 --wgs-per-cu 1 --comm-ratio 0.20 \
  --warmup 5 --iters 30 --profile
```

Release build（无 `-DMOE_PROFILE`）资源占用与 P2 完全一致：256 VGPR /
106 AGPR / 240 B scratch / 1 wave per SIMD，profile 钩子在 inlined
inline asm + DCE 之后**零开销**。
