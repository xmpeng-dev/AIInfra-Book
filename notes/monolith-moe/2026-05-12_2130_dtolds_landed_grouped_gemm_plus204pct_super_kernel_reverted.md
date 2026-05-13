# DirectToLds：grouped_gemm 飞跃 +204%，super-kernel 反而 −12% 已回滚

> 时间: 2026-05-12 21:30 (Asia/Shanghai)
> 项目: monolith-moe
> 硬件: 1× AMD Instinct MI355X (gfx950, single-node)
> 容器: xiaoming-dev podman (rocm/dev-ubuntu-22.04:7.2)
> 软件: hipcc 19 / ROCm 7.2.0 / LLVM 19 + LLD
> 代码: MMOE@`c282a77` (grouped_gemm DirectToLds 已 push) + 一份本地实验（super-kernel DTOLDS 已 revert）

## 1. 时间点 / 上下文

- 上一次相关进展：[2026-05-12_2230 LDS PAD 4→8 解锁 25% Grouped GEMM](./2026-05-12_2230_lds_pad_8_unlocks_25pct_grouped_gemm.md)（landed PAD=8）+ K_TILE 64→128 +7% DSV3（已 push commit `1e6dfe8`）
- 触发本次：[2026-05-12_1715 GEMM 内部 profile](./2026-05-12_1715_mfma_gemm_tile_internal_profile_lds_write_44pct_mfma.md) 锁死 `lds_write` 33–34 %、`mfma_inner` 含 ds_read 43–44 %，两段都吃 LDS。下一刀本应砍 `lds_write` 这条；DirectToLds (`buffer_load_dwordx4_lds`) 是干掉它的最直接武器——把 HBM→VGPR→LDS 三步合成一条 HW DMA 指令。

## 2. 问题

| 维度 | 数值 |
|---|---|
| Grouped GEMM (DSV3 GateUP, 静态/持久) 现状 | 210 T / 7.86 ms |
| Grouped GEMM 目标 | ≥ 400 T（封住 `lds_write` 33 %） |
| Super-kernel (DSV3_SPARSE m128_n128 ratio=0.20) 现状 | 421 T / 6.83 ms |
| Super-kernel 目标 | 接近 grouped_gemm 比例（500 T+） |

卡点假设：`lds_write` 是 critical path，DirectToLds 干掉它后应该接近 grouped_gemm baseline + 50% 提升。

## 3. 做了什么

### 3.1 grouped_gemm.hip：DirectToLds 落地 + K-对齐 template dispatch

| # | 动作 | 关键文件 / commit |
|---|---|---|
| 1 | 探测 gfx950 builtin：`__builtin_amdgcn_raw_ptr_buffer_load_lds`，确认 `size=16` 生成 `buffer_load_dwordx4 ... lds`；inline-asm 版本则会被 LLVM s_waitcnt pass 改写（与之前 P0 失败原因一致），所以走 builtin | `csrc/grouped_gemm.hip` DTOLDS_X4 macro |
| 2 | 将 prologue + 主 K 循环的 `buf_load16 → wait_vm → ds_write_b128` 三段塌成一条 DTOLDS_X4 | 同上，line ~385–462 |
| 3 | PAD=8 → 0：DTOLDS 每 wave 写 1024 字节必须连续，row stride 不能有 pad；MFMA-read 端在 K_TILE=64 时被验证容忍 32-way conflict | 同上，`PAD` 常量 |
| 4 | 边界处理踩坑：DTOLDS 写到 OOB voff（`0x80000000`）会 **silently 跳过 LDS write**，persistent kernel 跨 tile 复用 LDS 时会留 stale → partial-MN / partial-K 测试在 persistent 路径上炸 max_abs=14 | 复现 + 诊断在本 note 里 |
| 5 | Hybrid 修复：full-vector 走 DTOLDS_X4，partial 走 per-element scalar zero-fill + `ds_write_b128`。**单测全过，但 perf 跌回 210 T**——slow path 把 A/B 指针钉在 VGPR 里跨整个 K 循环 | 同上 |
| 6 | 关键 fix：模板参数 `bool kKAlignedKT`，`if constexpr` 把 slow path 编译期消掉。Host 端按 `K % kK == 0` dispatch 到 fast / slow 两个 `__global__` 模板实例化 | 同上，line ~387–462 + 753–784 |
| 7 | 编译期坑：`__global__` 模板要在 host 端 parse body，`gemm_core` 原本只是 `__device__` 导致 SFINAE 失败 → 改成 `__host__ __device__` + body 包 `#ifdef __HIP_DEVICE_COMPILE__` | 同上，line ~256–263, ~556–561 |
| 8 | 提交：commit `c282a77` "DirectToLds in grouped_gemm: HBM→LDS DMA, +210% TFLOPS"，已 push origin/main | — |

### 3.2 super-kernel：同 pattern 移植 → 失败

| # | 动作 | 结果 |
|---|---|---|
| 1 | `LDS_PAD = 8 → 0`，加 DTOLDS_X4 macro，模板 `bool kKAlignedKT=true`（production F_swiglu/H 都是 128 倍数） | smoke + correctness 全过 |
| 2 | `mfma_gemm_tile_t` (FC1)：prologue + 主 K 循环全部塌成 DTOLDS_X4；profile 上 `lds_write` 33% → 22% | — |
| 3 | `mfma_gemm_tile_swiglu_t` (FC2)：B 操作数走 DTOLDS，A 必须先把 gate/up 读到 VGPR 做 silu/rcp 才能写 LDS（DTOLDS 不能 fuse 计算），所以 A 保留 legacy | — |
| 4 | 全套 sweep：DSV3_SPARSE m128_n128 ratio=0.20 **6.83 → 7.86 ms（−13%）**；TILE_FIT m128_n128 ratio=0.20 **3.90 → 4.22 ms（−8%）** | **REGRESSION** |
| 5 | 隔离实验：把 FC2 也 revert 到 legacy（只留 FC1 DTOLDS），m128_n128 DSV3_SPARSE 7.81 ms / 369 T —— 仍然 regress 12 %。说明 regression 来自 PAD=0 本身，不是 FC2 partial-DTOLDS | — |
| 6 | 回滚 super-kernel 全部改动，仅保留 grouped_gemm.hip commit | — |

### 3.3 失败根因分析（**经 PAD=0-only 隔离实验最终锁定**）

> ⚠️ 这一节先后经历两版假设：
> - **H1**: "PAD=0 → 32-way bank conflict 是主因"——但同公式在 grouped_gemm K_TILE=64 PAD=0 下成立且 +204%，方向对但解释不完整
> - **H2**: "DTOLDS 把 LDS 写挪进 MFMA 循环 → 端口争用"——**被 PAD=0-only 实验否定**
> - **H3（最终）**: bank conflict **是真因**，但 grouped_gemm 高 MFMA 密度（16/K-step）能 hide，super-kernel small tile（1/K-step）hide 不住

**三路 bucket 对比（per WG per iter，DSV3 SPARSE m128_n128 ratio=0.18）：**

| Bucket | Baseline PAD=8 | **PAD=0 only**（无 DTOLDS） | DTOLDS (PAD=0) |
|---|---|---|---|
| `gemm_total` | **4104 us** | **5474 us** | 5306 us |
| `hbm_issue` | 703 us (17 %) | 647 us (11.8 %) | 1019 us (19 %) |
| `wait_vm` | 164 us (4 %) | 28 us (0.5 %) | 26 us (0.5 %) |
| `lds_write` | **2136 us (52 %)** | 1859 us (34 %) | **1291 us (24 %)** ✅ |
| `sync_per_ktile` | 36 us | 37 us | 32 us |
| `mfma_inner` | **588 us (14 %)** | **2470 us (45 %)** ← 已暴 | 2505 us (47 %) |
| **wall** | **6.83 ms / 421T** | **7.94 ms / 364T** | 7.86 ms / 367T |

**决定性证据**：只把 `LDS_PAD` 从 8 改 0、**根本没动 DTOLDS**，`mfma_inner` 就已经从 588 → 2470 us（4×）。DTOLDS 再加上去 mfma_inner 几乎不变（2470 → 2505），但成功把 `lds_write` 从 1859 → 1291 砍掉 568 us。证明：
1. **DTOLDS 移植代码本身没问题**——它干活了（lds_write 单独看是 −40%）
2. **mfma_inner 4× 爆炸的真因是 PAD=0 的 ds_read bank conflict**——跟 DTOLDS 无关
3. **DTOLDS 被 PAD=0 prereq 绑架而 regress**：bank conflict 代价（+1882 us mfma_inner）远 > DTOLDS 收益（−568 us lds_write − 138 us wait_vm = −706 us）

历史 bucket 对比（删除）：

**真正机制：PAD=0 引发 32-way bank conflict → ds_read 暴露在临界路径上**

`mfma_inner` bucket 量的是 `ds_read_b128 + mfma_bf16` 两件事合起来的 K-step 循环 wall。MFMA 指令数完全没动（同 K 维度同 M/N），唯一可能涨的是 `ds_read_b128` 等待时间。

PAD=0 配置下的 LDS bank pattern：
- K_TILE=128 时 row stride = 128 bf16 = 64 dwords，(64 mod 32) = 0 → 所有 32 lane 落到 bank 0 → **32-way conflict**
- 同样 K_TILE=64 时 stride = 32 dwords，(32 mod 32) = 0 → 也是 **32-way conflict**

PAD=8 时 (K=128) stride = 68 dwords，(68 mod 32) = 4 → lane k 落到 bank (4k) mod 32，周期 8 → **4-way conflict**（不是之前说的 2-way）。

也就是 PAD=0 把 ds_read 的实际 latency 推到大约 32×/4× = 8× 长，**单 read 大约从 ~20 cyc 涨到 ~100+ cyc**。

**为什么 grouped_gemm 同样 PAD=0 + DTOLDS 反而 +204 %？因为 MFMA 计算密度是 4–16×：**

| | super-kernel **small tile**（DSV3 SPARSE Te=16 走这条） | super-kernel default tile (Te ≥ 32) | grouped_gemm |
|---|---|---|---|
| Tile 形状 | M=32 N=128 1×4 waves | M=N=128 2×2 waves | M=N=256 2×2 waves |
| `MFMA_PER_WAVE_M × N` / wave | **1 × 1 = 1 MFMA** | 2 × 2 = 4 MFMA | 4 × 4 = **16 MFMA** |
| Compute / K-step / wave | ~32 cyc | ~128 cyc | ~512 cyc |
| 100+ cyc ds_read 能不能 hide？ | **❌ 完全暴露** | 临界 | ✅ 充分 |
| PAD=0 → mfma_inner 影响 | 4 × 爆炸（small tile 占绝大多数） | 中等 | 几乎无 |

**bank conflict 不是次要项，是真的主因——只是要看它能不能被 MFMA hide**：

- grouped_gemm 16 MFMA / K-step → bank conflict 加重的 ds_read 完全被 compute 盖住 → DTOLDS 收益直通
- super-kernel small tile 1 MFMA / K-step → bank conflict 加重的 ds_read 完全暴露 → mfma_inner 4 × 爆炸，吞掉 DTOLDS 收益

而且 PAD=8 baseline 6.83 ms 跑得动是因为 4-way conflict 下 ds_read latency ~50–80 cyc，跟 1 MFMA compute ~32 cyc 同量级——勉强能在软件 pipelining 里塞下；PAD=0 把 latency 推到 ~100+ cyc，就 hide 不住了。

简言之：**super-kernel DTOLDS 失败是 "PAD=0 prereq + small tile 低 MFMA 密度" 两件事相乘的结果**。DTOLDS 这个改动本身没问题，是 small tile 路径吃不下它的 prereq。

## 4. 效果

### 4.1 grouped_gemm.hip（landed）

| Shape | Before (PAD=8, K_TILE=128, 旧 staging) | After (PAD=0, K_TILE=128, DTOLDS, kKAlignedKT) | Δ |
|---|---|---|---|
| DSV3-GateUP M=8192 K=7168 N=4096 | 199 T / 38.5 ms | **643 T / 12.0 ms** | **+222 %** |
| DSV3-GateUP M=16384 K=7168 N=4096 | 204 T / 75.5 ms | **643 T / 23.9 ms** | **+215 %** |
| DSV3-Down M=8192 K=7168 N=2048 | 213 T / 18.0 ms | **558 T / 6.99 ms** | **+162 %** |
| DSV3-Down M=16384 K=7168 N=2048 | 211 T / 36.4 ms | **560 T / 27.5 ms** | **+165 %** |
| MoE-sparse stress (M=128 K=7168 N=4096) | 113 T | **321 T** | +184 % |
| **vs CK 参考（1050 T GateUP / 960 T Down）** | 20 % CK | **60 % CK** | — |

所有测试 PASS：static + persistent + mfma standalone + partial-MN + K-tail + tiny-K（K=70 / 80 / 31 全覆盖）。

### 4.2 super-kernel（reverted）

| 配置 | Before (baseline `1e6dfe8`) | After DTOLDS 实验 | Δ | 状态 |
|---|---|---|---|---|
| m128_n128 DSV3_SPARSE ratio=0.20 | 421 T / 6.83 ms | 367 T / 7.86 ms | **−13 %** | reverted |
| m128_n128 DSV3_SPARSE ratio=0.25 | 425 T / 6.80 ms | 362 T / 7.97 ms | **−15 %** | reverted |
| m128_n128 TILE_FIT ratio=0.20 | 212 T / 3.90 ms | 195 T / 4.22 ms | **−8 %** | reverted |
| m128_n128 TILE_FIT ratio=0.25 | 217 T / 3.80 ms | 198 T / 4.16 ms | **−8.5 %** | reverted |
| m64_n64 DSV3_SPARSE ratio=0.20 | 297 T / 9.72 ms | 253 T / 11.42 ms | **−15 %** | reverted |
| m64_n64 TILE_FIT ratio=0.20 | 176 T / 4.70 ms | 156 T / 5.28 ms | **−11 %** | reverted |

定性观察：

- ✅ grouped_gemm 单 kernel 收益巨大（+204 % 平均），CK gap 直接砍半
- ✅ 学到 DTOLDS 的三条硬约束：①per-wave 1024 B 连续写 → PAD=0；②OOB voff silently 跳过 LDS write → persistent 路径会留 stale data；③slow path 不 template 出去会钉 A/B 指针 3 ×↓ TFLOPS
- ❌ super-kernel 上 DTOLDS regress 12 %，**真因经 PAD=0-only 隔离实验最终锁定**：PAD=0 引起 32-way bank conflict，把 ds_read latency 推到 ~100+ cyc，DSV3 SPARSE 走的 small tile 路径只有 1 MFMA / K-step (~32 cyc compute) 完全 hide 不住，`mfma_inner` 单独 4×（588→2470 us）即使没动 DTOLDS。DTOLDS 这一改动本身没问题（lds_write 单独看 −40%），是被它自带的 PAD=0 prereq 绑架
- ⚠️ Density-bump 验证（M=N=256 K=64 PAD=8 不动 DTOLDS）：TILE_FIT (Te=128 默认 tile) GEMM 内部 −40%；DSV3 SPARSE (Te=16 small tile) wall −11%，因为 small tile 不走默认 tile，density 跟它无关，而 K_TILE 减半把 small tile 的 K-tile 数 2× → per-tile overhead 2× → 输

详细 bucket 对比和机制分析见 §3.3。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| ✅ **完成** | PAD=0-only 隔离实验 | 锁定 bank conflict 是真因（mfma_inner 588→2470 us 完全归因于 PAD=0，跟 DTOLDS 无关） | — |
| ✅ **完成** | M=N=256 K=64 PAD=8 density-bump 实验 | TILE_FIT GEMM −40% 但 DSV3 SPARSE wall −11%；证明 density 对 small tile 无效 | — |
| **P0** | **LDS XOR swizzle**（**只配 legacy ds_write_b128 路径**，不能加 DTOLDS） | DSV3 SPARSE: `lds_write` 不变（2136 us 仍在），但 `mfma_inner` 4-way → 1-way conflict → 588→~150 us → **gemm_total 4104→~3660 us, wall 6.83→~6.1 ms (−11 %)**。⚠️ **硬件约束**：DTOLDS 是 lane k→M0+k*16 强制线性 LDS DMA，XOR swizzle 需要非线性物理布局，两者**不能直接结合**；要砍 `lds_write` 的 52 % 大头需要重新设计 wave-block LDS layout 才能同时容纳 DTOLDS + 反 bank-conflict | 索引重写（写入和读取需对齐 XOR mask）；先 standalone grouped_gemm（legacy slow-path）验证再 port super-kernel；1-2 d；参考 [ck_deep_dive](./2026-05-08_ck_implementation_deep_dive.md) |
| P1 | per-tile-class K_TILE template（让 default tile 用 M=N=256 K=64 + small tile 用 K=128） | 默认 tile GEMM −40 %（Step 1 已验证），DSV3 SPARSE 不动 | LDS region union + kK/kPad 模板参数；1-2 d；等 P0 落地后衡量 |
| P1 | **FP8 / mxfp8 weights for FC1 / FC2** | HBM weight 流量 ÷ 2，与 swizzle/density 正交；DSV3 FC 路径再 −1 ms | mxfp8 quantizer + scale 路径；2-3 d |
| **P0** | **FP8 / mxfp8 weights for FC1 / FC2**（与 LDS swizzle 正交） | HBM weight 流量 ÷2，DSV3 FC 5.86 → ~3 ms（−2.8 ms） | mxfp8 quantizer + scale 路径；2-3 d |
| P1 | 把 grouped_gemm 这套 `kKAlignedKT` template + `__host__ __device__` 模式当模板 | 后续任何 GEMM kernel 加 DTOLDS 都遵循 | 已写在 commit message |
| P1 | Wave-block LDS layout（4 行 1024 B 一组 + group 间 PAD） | 让 PAD>0 与 DTOLDS 共存，4-way conflict 而不是 32-way；理论上能恢复 super-kernel DTOLDS 收益 | 索引函数变复杂；先验 XOR swizzle，简单实现走通再考虑 |
| P2 | C-shuffle epilogue（+3–5 %） | FC2 影响小，可与 FP8 联动 | 不阻塞 |
| P2 | Work-stealing tile counter（grouped_gemm 已有 persistent 路径，super-kernel 也可移植） | −0.2~0.5 ms | 已在 grouped_gemm 里跑通 |

## 相关文件

- 代码 / commit：`c282a77` (csrc/grouped_gemm.hip)，已 push `origin/main`
- bench 验证脚本：`benchmarks/results/verify_gg_dtolds.sh`
- bench 数据：`benchmarks/results/grouped_gemm_dtolds_landed.txt`
- 上游 note：[`2026-05-12_2230_lds_pad_8_unlocks_25pct_grouped_gemm.md`](./2026-05-12_2230_lds_pad_8_unlocks_25pct_grouped_gemm.md), [`2026-05-12_1715_mfma_gemm_tile_internal_profile_lds_write_44pct_mfma.md`](./2026-05-12_1715_mfma_gemm_tile_internal_profile_lds_write_44pct_mfma.md)
- 反例代码（已 revert，未 commit）：super-kernel 上的 DTOLDS 移植；patch 形态可由 `git stash` 历史恢复

