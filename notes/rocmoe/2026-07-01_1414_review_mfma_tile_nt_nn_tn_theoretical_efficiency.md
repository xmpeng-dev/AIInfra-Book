# mfma_tile NT/NN/TN 三方向 GEMM tile 纯理论效率评估

> **When**: 2026-07-01 14:14 UTC+8
> **Where**: login node / 无 GPU 分配 (纯理论分析, 未实测)
> **Context**: 评估 `ROCMega-dev/csrc/gemm/mfma_tile_{nt,nn,tn}.h` (shape-generic v2 GEMM tile, 是 `mfma_tile.h` 的模板化后继) 的实现效率 / 流程合理性 / 性能上限; 对照 AMD CDNA4 官方 GEMM 优化路线图

## TL;DR

三个 tile (NT 前向 / NN 反向 dY / TN 反向 dW) 流程正确、技术栈几乎完全对齐 AMD CDNA4 官方博客 (DTOLDS + 双缓冲 + XOR swizzle + K-doubled MFMA + `ds_read_tr16` 转置读), 属于路线图上"倒数第二档"。**离 hipBLASLt/CK 级别的最后差距集中在 wave specialization / 8-wave ping-pong 这一项**; 三者里 **NN 宽 leg 是最弱环节** (AGPR 128/lane 占用率 ~2 waves/SIMD + 默认关掉 K-lookahead)。纯理论最高 ROI 改动: 给 1 WG/CU 的大 tile 上 ping-pong, 并把 NN 的 `ROCMOE_MFMA_NN_K_LOOKAHEAD` 默认打开。

## Background

- 被评估文件 (均在 `ROCMega-dev/csrc/gemm/`, 共享 `mfma_common.h` 原语):
  - `mfma_tile_nt.h` — C = A·Bᵀ, 前向 FC1/FC2
  - `mfma_tile_nn.h` — C = A·B, 反向 dY (FC2-bwd / FC1-bwd+scatter)
  - `mfma_tile_tn.h` — C = Aᵀ·B, 反向 dW (grad_w)
- 目标硬件 gfx950 (MI355X, CDNA4): bf16 dense 峰值 ~2.5 PFLOPS, HBM 8 TB/s, LDS 160 KB/CU, bf16 roofline 拐点 AI\* ≈ 315。
- 无机器可跑, 全部为静态代码 + roofline/occupancy 推算。

## 相关论文 / 参考 (GEMM 优化)

此方向权威材料主要在厂商博客 + 库源码, 纯学术论文少。

| 来源 | 类型 | 与本代码的关系 |
|---|---|---|
| ROCm Blog: *FP8 GEMM Optimization on CDNA4 (MI355X)* | 厂商博客 (最相关) | double-buffer→matrix core→DTOLDS→LDS swizzle→细粒度调度→**8-wave ping-pong** 追平 hipBLASLt; 本代码走到倒数第二步。单→双缓冲 +134%, swizzle 再 +10% (M=N=K=4096) |
| ROCm Blog: *Accelerating LLM Inference with Low-Latency GEMMs* | 厂商博客 | Split-K / intra-CTA K-slice / 多级 LDS ring (STAGES); 本代码仅 2 级 ring 无 split-K |
| Osama et al., *Stream-K: Work-centric Parallel Decomposition for GEMM on the GPU* (2023) | 学术论文 | K 维负载均衡, 针对 small-M/大-K (MoE per-expert 痛点); 本代码未用 |
| NVIDIA CUTLASS docs + AMD Composable Kernel 源码 | 参考实现 | 本代码 WG→wave→MFMA 层次分解即 CUTLASS-style; 见 `knowledge/libraries/composable-kernel.md` |

## 主要发现 / 结论

### 用到的优化技术 (对照 CDNA4 SOTA)

| 技术 | 代码中 | 官方推荐 | 评价 |
|---|---|---|---|
| Direct-to-LDS (HBM→LDS 单 DMA, 绕过 VGPR+ds_write) | ✅ `ROCMOE_DTOLDS_X4` | ✅ 关键 | 到位 |
| Double buffering (K-tile ring) | ✅ `kLdsBufs=2`, 可选 3 | ✅ 关键 | 到位 |
| XOR swizzle 消 bank conflict | ✅ NT/NN 的 A-half (K 连续侧) | ✅ | 到位 (仅覆盖 K-连续侧) |
| K-doubled MFMA `32x32x16_bf16` | ✅ gfx950 分支 | ✅ 必须 (拿 2× 峰值) | 到位 |
| gfx950 转置读 `ds_read_tr16` (LDS 读 -4×) | ✅ TN 全路径 + NN B-half | ✅ | 到位, 关键 |
| OOB 谓词 `voff=0x80000000` 免边界 mask | ✅ | ✅ buffer_load no-trap | 到位 |
| K-lookahead (ks→ks+1 喂满 MFMA) | TN 默认开 / **NN 默认关** | ✅ | 见下 |
| **Wave specialization / 8-wave ping-pong** | ❌ 全 wave 同时 load+compute | ✅ **追平 hipBLASLt 最后一步** | **最大缺口** |
| Split-K / Stream-K | ❌ | ✅ (low-latency GEMM) | small-M MoE 有头款 |
| 向量化 epilogue (经 LDS 聚合再写) | ❌ 逐元素 2B 写 | — | 小头款 |

### 流程合理性: 合理, 无正确性问题

- 标准 CUTLASS-style 三层分解 + 软件流水: prologue 预取 `kLdsBufs-1` 块 → 主循环 issue 下一块同时算当前块 → epilogue drain; `wait_vm`/`wait_lgkm`/`__syncthreads` 摆放符合 CDNA counter 语义, 无 torn-read 风险。
- NT 两侧 K 连续 → 统一 swizzle; NN/TN 的 K-主侧受 `buffer_load_lds` wave-uniform 约束不能 swizzle/pad (`kLdsPadTN=0`), 改用 `ds_read_tr16` 硬件转置读 — 取舍正确 (注释交代 swizzle 在 `[K,M]` 上实测反而退化)。
- 三文件复用同一套 `TileXX` traits + 同一 K-loop 骨架, 一致性好。TN 保证跨 layout 位一致, 利于测试/确定性。

### 效率 / 性能理论评估 (逐 tile 配置)

| 路径 | tile / 波数 | 每 lane AGPR (acc) | 占用率 (waves/SIMD) | 定位 |
|---|---|---|---|---|
| NT 窄 | 128², W=2×2, 4 wave, 2 WG/CU | acc[1][1]=16 | 高, sibling WG 掩盖 wait | **最健康** |
| NT 宽 | 256², W=4×4, 16 wave, 1 WG/CU | acc[2][2]=64 | ~4/SIMD (OK) | 大 M/N/K 可达 bf16 峰值较高比例 (~60-75%) |
| NN 宽 | 256², W=4×2, 8 wave, 1 WG/CU | acc[2][4]=**128** | 仅 ~2/SIMD | **占用率受限 + 默认无 lookahead**, 最弱 |
| TN 宽 (dW) | W=2×2 默认 | 取决于 tile | 1 WG/CU + lookahead | 转置读是主瓶颈, 已用 tr16 缓解 |

要点:
- **前向 NT** 技术栈与博客里追到 ~1166 TFLOPS FP8 (≈hipBLASLt) 那份 kernel 结构几乎一致, 只差 ping-pong; bf16 大 GEMM 下应能到 ~2.5 PFLOPS 峰值不错比例。三者最强。
- **MoE 现实约束**: per-expert M (路由 token) 常几十到几百, bf16 roofline 拐点 AI\*≈315 (K 要 ≳315 才 compute-bound) → 很多 expert-GEMM 实际**访存受限**, 决定性能的是"少一次 HBM 往返 + 融合 epilogue"而非 MFMA duty cycle。代码的 scatter epilogue + `n_off` 融合正冲此设计, 方向对。
- **反向 dW (TN)** 最难: A 为 `[K,M]` 需转置读, 1 WG/CU; 已用 `ds_read_tr16` 把 LDS 读指令砍 4×, 是该路径最重要优化, 做到位。

## 详细分析: 潜在瓶颈 & 可改进点 (按 ROI 排序)

1. **无 wave specialization / 8-wave ping-pong (最大头)**。博客里这一步把 HIP/C++ kernel 从"接近"追到"追平 hipBLASLt"。当前所有 wave 既搬数又算, 1 WG/CU 的大 tile (NN 宽、TN) 靠 lookahead+双缓冲掩盖延迟, duty cycle 有天花板。参照 `.cursor/skills/cco-pipeline-overlap/SKILL.md` 做 producer/consumer 分工。
2. **NN 默认 `ROCMOE_MFMA_NN_K_LOOKAHEAD=0`, 而 TN 默认 =1** (`mfma_tile_nn.h:70` vs `mfma_tile_tn.h:51`)。NN 宽同样 1 WG/CU (注释明说 fc1 dY "exposed, NOT hidden under transport"), 又是所有路径里 AGPR 压力最大 (128/lane)、占用率最低 (~2 waves/SIMD) 的, 却默认关 lookahead → per-K-step `wait_lgkm(0)` 暴露。建议在 1 WG/CU 的 NN 宽 leg 默认打开 (需实测: 它同时吃 VGPR 会压占用率, 是权衡)。这是三文件间最明显的策略不一致。
3. **默认只 2 级 buffer**。MI355X 160 KB LDS (CDNA3 的 2.5×), 硬件 skill 明确 `num_stages=3` 现实可行; 1 WG/CU 大 tile 有预算上三级 ring 多藏一层 HBM 延迟。已留 opt-in 3, 可对大 tile 默认 3。
4. **epilogue 未向量化** (`mfma_common.h:store_acc_block`)。每 lane 逐元素 16 个 2-byte scatter 写 (row-strided 不合并)。对 compute-bound 大 GEMM 无所谓, 但对 small-M MoE 众多小 tile 会累积。可先在 LDS 把 32×32 acc 重排成行连续再做 128-bit 合并写 (scatter 版天生做不了, contiguous 版可以)。
5. **无 Split-K / Stream-K**。small-M + 大 K 的 expert-GEMM 会让单 WG 的 K-loop 长而 CU 利用不均; 但 MoE grid 横跨 experts×tiles 天然有并行度, 优先级低。

## 下一步 / 建议

- 纯理论最高 ROI: **给 1 WG/CU 大 tile (NN 宽、TN) 上 8-wave ping-pong + 把 NN 的 K-lookahead 默认打开** — 收益最高、风险可控。
- 需机器时优先实测: (a) NN 宽 leg 打开 lookahead 后 occupancy vs wait-stall 净效果; (b) 大 tile 3-stage ring; (c) rocprof 看 `MFMAUtilization` / `LDSBankConflict` 定位 TN 转置读是否有残余 bank conflict。
- 与 rocmoe 主线的关系: 这三个 tile 是 M7 (per-tile-class K_TILE template) / M8 (decomposed backward) 的 GEMM 资产; 若上 ping-pong 亦是 M4 (wave specialization) 的落地点。

## 相关文件

- `ROCMega-dev/csrc/gemm/mfma_tile_nt.h` · `mfma_tile_nn.h` · `mfma_tile_tn.h`
- `ROCMega-dev/csrc/gemm/mfma_common.h` (DTOLDS / swizzle / lds_frag / ds_read_tr16 / waits / epilogue)
- `.cursor/skills/mi355_hardware_aware/SKILL.md` (roofline / occupancy / MFMA 表)
- `.cursor/skills/amd-gemm-optimization/SKILL.md` · `.cursor/skills/cco-pipeline-overlap/SKILL.md`
- ROCm Blog: FP8 GEMM Optimization on CDNA4 — <https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html>
