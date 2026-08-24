# 主流模型 attention 形状横测：HK vs Primus-Turbo 全后端

> **When**: 2026-08-14 14:50 UTC+8
> **Where**: `smci355-ccs-aus-n04-21`（MI355X / gfx950），容器 `xiaoming-dev`（`rocm/primus:v26.3`，ROCm 7.2.1 / clang 22 / torch 2.10.0），单卡 `HIP_VISIBLE_DEVICES=0`
> **Context**: 在 20 个主流模型的真实 attention 形状上横测。方法与 HK 的编译修法见 [同日前一篇](./2026-08-14_1330_hk_attn_vs_turbo_blocked.md)

## TL;DR

1. **HK 能跑 13/40 行**，全部胜出：**1.04–1.23×，几何平均 1.12×**（对最快 turbo，即 aiter CK/asm）。
2. **HK 的优势恰好集中在 aiter 最弱的格子**：llama4（B=1, h_q=40）**1.23×**、gpt_oss（d=64）**1.20–1.21×**；而在 aiter 已接近饱和的 grok/mixtral（B=2, s=8192，ck 1044–1053 TFLOPS）只有 **1.04×**。**与 08-12 GEMM 那轮同构 —— HK 补的是低并行度/非主流 head_dim 的短板，不是普遍更快。**
3. **`d=256` 是全栈短板，且没人覆盖**：glm5 **ck=494**、qwen3_5_35B **ck=556**（对比 d=128 的 ~940–1050，掉一半）；aiter Triton 更崩到 **114 / 134**（0.23×）。**HK 也不支持 d=256。** glm5 与 qwen3.5 都用它 → **真实缺口**。
4. **`d=64` 也偏弱**：gpt_oss ck **620–621** vs d=128 的 ~940。
5. **fp8 attention 全线不可用**：**169–454 TFLOPS**，对应 bf16 636–1053，普遍慢 2–4×。与 DSV3 报告"只适合正确性验证"一致。
6. **MLA(192/128) 上 turbo 很强**（947–1086），HK 完全缺席（单一 `ATTN_D`，`d_qk≠d_v` 跑不了）。

## HK 的支持边界（三条硬限制）

| 限制 | 来源 | 挡掉的行 |
|---|---|---|
| **bf16 only**（无 fp8 attention） | `kernels/cdna4/attn/` 只有 `gqa{,_causal}{,_backwards}`，全 bf16；全仓无 fp8 attention | **20 行** |
| **`d_qk == d_v`**（单一 `ATTN_D`，K/V 共用） | `attn_fwd_causal.cpp:22` `constexpr int ATTN_D` | MLA 192/128 的 **5 个 bf16 行** |
| **`d ∈ {64, 128}`** | `Makefile`：`ifeq ($(ATTN_D),64) SRC=kernel_d64.cpp else SRC=kernel.cpp`，无其他分支 | d=256 的 **2 个 bf16 行** |
| （另）**无稀疏路径** | 无 top-k gather 实现 | V4-sparse **2 行** |

## 结果表

口径：**causal 前向**（训练默认），bf16（fp8 列除外），30 warmup / 20 iters，单位 **TFLOPS**。
`FLOPs = 2·B·s²·h_q·(d_qk+d_v) / 2`（兼容 `d_qk≠d_v`，与 DSV3 报告一致）。
`ck` = turbo `flash_attn_func`（aiter csrc/CK+asm-v3，生产默认）· `tri` = turbo `flash_attn_func(sink=…)`（aiter Triton，**算的是带 sink 的 attention**）· `fp8` = turbo `flash_attn_fp8_func`（Triton blockwise，**精度不同**）。

### HK 可跑的 13 行

| model | attn | B | s | h_q | h_kv | d | **HK** | ck | tri | **HK / 最快 turbo** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| llama4_17B128E | GQA | 1 | 4096 | 40 | 8 | 128 | **783** | 636 | 474 | **1.23×** |
| llama4_17B16E | GQA | 1 | 4096 | 40 | 8 | 128 | **777** | 636 | 474 | **1.22×** |
| gpt_oss_20B | GQA | 8 | 4096 | 64 | 8 | 64 | **762** | 620 | 630 | **1.21×** |
| gpt_oss_120B | GQA | 8 | 4096 | 64 | 8 | 64 | **758** | 621 | 630 | **1.20×** |
| minimax_m2.5 | GQA | 3 | 4096 | 48 | 8 | 128 | **972** | 861 | 652 | **1.13×** |
| qwen3_235B_A22B | GQA | 4 | 4096 | 64 | 4 | 128 | **1028** | 938 | 671 | **1.10×** |
| qwen3_30B_A3B | GQA | 8 | 4096 | 32 | 4 | 128 | **1029** | 941 | 667 | **1.09×** |
| lfm2_8B_A1B | GQA | 8 | 4096 | 32 | 8 | 128 | **1025** | 938 | 668 | **1.09×** |
| mixtral_8x7B_v0.1 | GQA | 4 | 4096 | 32 | 8 | 128 | **975** | 900 | 650 | **1.08×** |
| deepseek_v2_lite | MHA | 12 | 4096 | 16 | 16 | 128 | **992** | 933 | 675 | **1.06×** |
| grok2 | GQA | 2 | 8192 | 64 | 8 | 128 | **1117** | 1053 | 758 | **1.06×** |
| grok1 | GQA | 2 | 8192 | 48 | 8 | 128 | **1086** | 1044 | 745 | **1.04×** |
| mixtral_8x22B_v0.1 | GQA | 2 | 8192 | 48 | 8 | 128 | **1087** | 1045 | 743 | **1.04×** |

> **HK 13/13 全胜，1.04–1.23×，几何平均 1.12×。** 最快 turbo 在全部行都是 aiter CK/asm。

### HK 不支持的行（含原因）

| model | attn | dtype | B | s | h_q | h_kv | d_qk/d_v | HK 不支持的原因 | ck | tri | fp8 |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| kimi_k3_8L_official | MLA | bf16 | 2 | 7168 | 96 | 96 | 192/128 | `d_qk≠d_v` | **1086** | — | — |
| kimi_k2 | MLA | bf16 | 4 | 4096 | 64 | 64 | 192/128 | `d_qk≠d_v` | **979** | — | — |
| deepseek_v3 | MLA | bf16 | 2 | 4096 | 128 | 128 | 192/128 | `d_qk≠d_v` | **975** | — | — |
| deepseek_v2 | MLA | bf16 | 1 | 4096 | 128 | 128 | 192/128 | `d_qk≠d_v` | **947** | — | — |
| kimi_k3_curve | MLA | bf16 | 2 | 2048 | 16 | 16 | 192/128 | `d_qk≠d_v` | 478 | — | — |
| **qwen3_5_35B_A3B** | GQA | bf16 | 2 | 8192 | 16 | 2 | **256/256** | **d=256 无该实例** | **556** | **134** | — |
| **glm5** | MLA | bf16 | 1 | 4096 | 64 | 64 | **256/256** | **d=256 无该实例** | **494** | **114** | — |
| qwen3_5_35B_A3B | GQA | fp8 | 2 | 8192 | 16 | 2 | 256/256 | 无 fp8 | — | — | 454 |
| grok2 | GQA | fp8 | 2 | 8192 | 64 | 8 | 128/128 | 无 fp8 | — | — | 413 |
| mixtral_8x22B_v0.1 | GQA | fp8 | 2 | 8192 | 48 | 8 | 128/128 | 无 fp8 | — | — | 409 |
| grok1 | GQA | fp8 | 2 | 8192 | 48 | 8 | 128/128 | 无 fp8 | — | — | 405 |
| qwen3_235B_A22B | GQA | fp8 | 4 | 4096 | 64 | 4 | 128/128 | 无 fp8 | — | — | 313 |
| qwen3_30B_A3B | GQA | fp8 | 6 | 4096 | 32 | 4 | 128/128 | 无 fp8 | — | — | 296 |
| lfm2_8B_A1B | GQA | fp8 | 8 | 4096 | 32 | 8 | 128/128 | 无 fp8 | — | — | 294 |
| minimax_m2.5 | GQA | fp8 | 3 | 4096 | 48 | 8 | 128/128 | 无 fp8 | — | — | 291 |
| gpt_oss_20B | GQA | fp8 | 2 | 8192 | 64 | 8 | 64/64 | 无 fp8 | — | — | 285 |
| glm5 | MLA | fp8 | 1 | 4096 | 64 | 64 | 256/256 | 无 fp8 | — | — | 239 |
| gpt_oss_120B | GQA | fp8 | 8 | 4096 | 64 | 8 | 64/64 | 无 fp8 | — | — | 229 |
| gpt_oss_20B | GQA | fp8 | 6 | 4096 | 64 | 8 | 64/64 | 无 fp8 | — | — | 225 |
| mixtral_8x7B_v0.1 | GQA | fp8 | 2 | 4096 | 32 | 8 | 128/128 | 无 fp8 | — | — | 211 |
| kimi_k2 | MLA | fp8 | 4 | 4096 | 64 | 64 | 192/128 | 无 fp8 | — | — | 209 |
| deepseek_v3 | MLA | fp8 | 2 | 4096 | 128 | 128 | 192/128 | 无 fp8 | — | — | 208 |
| deepseek_v2_lite | MLA | fp8 | 14 | 4096 | 16 | 16 | 192/128 | 无 fp8 | — | — | 208 |
| deepseek_v2 | MLA | fp8 | 1 | 4096 | 128 | 128 | 192/128 | 无 fp8 | — | — | 197 |
| llama4_17B128E | GQA | fp8 | 1 | 4096 | 40 | 8 | 128/128 | 无 fp8 | — | — | 171 |
| llama4_17B16E | GQA | fp8 | 1 | 4096 | 40 | 8 | 128/128 | 无 fp8 | — | — | 169 |

### V4-sparse（独立 API，FlyDSL vs Triton）

`deepseek_v4_flash` 走的是 sparse MLA（单 latent MQA + per-token top-k），HK 无稀疏路径。
在生产 seqlen 上跑 turbo 的两个实现（`primus_turbo/{flydsl,triton}/attention/sparse_mla*`）：

| model | s | h_q | d | cr | topk 宽 | FlyDSL | Triton | **FlyDSL/Triton** |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| deepseek_v4_flash | 4096 | 64 | 512 | 0（纯 SWA） | 128 | 0.244 ms | 0.316 ms | **1.30×** |
| deepseek_v4_flash | 4096 | 64 | 512 | 4（random pool） | 640 | 0.572 ms | 0.924 ms | **1.62×** |

与 08-14 早前在 s=1024/2048 上的测量（1.43–2.03×）同向，倍数随 seqlen 增大而收窄。

## 解读

### HK 的增量集中在 aiter 的弱格

把 13 行按比值排序，规律很干净：

| 比值段 | 行 | 共同特征 | aiter 的绝对水平 |
|---|---|---|---|
| **1.20–1.23×** | llama4 ×2（B=1, h_q=40）、gpt_oss ×2（d=64） | **低并行度**（B=1）或 **非主流 head_dim**（64） | **620–636**（低） |
| 1.06–1.13× | minimax、qwen3 ×2、lfm2、mixtral_8x7B、dsv2_lite、grok2 | 中等 | 861–1053 |
| **1.04×** | grok1、mixtral_8x22B（B=2, s=8192） | **高并行度长序列** | **1044–1045**（接近饱和） |

**aiter 越强的地方，HK 的增量越小。** 这与 08-12 GEMM 那轮的结论完全同构（HK 赢在网格饥饿区，在 aiter 跑满的大方阵上只是追平）。**HK 的价值定位应该是"补短板"而不是"普遍更快"。**

### `d=256` 是一个所有人都没覆盖的格子

| d | turbo ck 典型值 | aiter Triton | HK |
|---:|---:|---:|---|
| 64 | 620–621 | 630 | ✓ 758–762 |
| 128 | 861–1053 | 650–758 | ✓ 972–1117 |
| 192/128（MLA） | 947–1086 | ✗ | ✗ |
| **256** | **494–556** | **114–134** | **✗ 无实例** |

`d=256` 上 aiter CK 掉到 d=128 的一半，aiter Triton 直接崩（0.23×），**而 HK 连实例都没有**。
用它的是 **glm5**（78 层）和 **qwen3_5_35B_A3B**（40 层）—— 两个都是新模型。
**这是本轮唯一一个"三方都没覆盖好"的格子，也是最值得动手的缺口。**

### fp8 attention 仍然不可用

19 个 fp8 行全部落在 **169–454 TFLOPS**，而同形状 bf16 的 aiter CK 是 636–1053 —— **普遍慢 2–4×**。
DSV3 报告"fp8 attention 慢 4.0×、精度低 25 dB，目前只适合做正确性验证"的结论，在全部主流模型形状上都成立。
注意 fp8 是这批模型里**一半的行**（20/40），意味着**如果这些配置真要跑 fp8，attention 会成为瓶颈**。

## 下一步

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | **`d=256` 缺口**：先用 rocprof 确认 aiter CK 在 d=256 掉一半的机制（LDS 容量？tile 选型？），再判断是 aiter 配置问题还是真需要新 kernel | 三方都没覆盖；glm5 + qwen3.5 两个新模型在用；是本轮唯一的"无人区" |
| P1 | HK 反向（`gqa{,_causal}_backwards`）在这 13 行上测 | 论文最强主张在反向；前向已确认 HK 的增量集中在弱格，反向是否同规律未知 |
| P1 | HK 加 `d=256` 实例（新增 `kernel_d256.cpp`）可行性评估 | 若 LDS/寄存器允许，这是 HK 唯一能填的"无人区" |
| P2 | fp8 attention：确认是 Triton 实现问题还是 aiter 无 fp8 汇编路径 | 20/40 行受影响，但属 turbo 侧问题，非 HK |

## 复现

```bash
ssh smci355-ccs-aus-n04-21
docker exec xiaoming-dev bash -lc '
  cd /perf_apps/xiaoming/scratch/hk_attn && bash run_models.sh'
```

脚本：`/perf_apps/xiaoming/scratch/hk_attn/{run_models.sh, bench_model.py, patch_nodep.py}`
（`patch_nodep.py` 是 HK 编译歧义的绕过，见[前一篇 §2](./2026-08-14_1330_hk_attn_vs_turbo_blocked.md)）

## 相关

- [2026-08-14_1330_hk_attn_vs_turbo_blocked.md](./2026-08-14_1330_hk_attn_vs_turbo_blocked.md) — 方法学、编译歧义根因与修法、turbo 三后端排序、FlyDSL sparse MLA
- [2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md) — GEMM 侧的同构结论（HK 赢在弱格）
- `/perf_apps/xiaoming/MegaMoE/docs/attention_dsv3_8k_backends.md` — fp8 attention 与 MLA 的原始报告
