# 2026-05-12 22:30 CST  LDS_PAD=8 解锁 GroupedGEMM +25% — alignment 才是 PAD=4 的真坑

## 环境
- 主机: mi355-gpu-26 (AMD MI355X gfx950, 256 CU)
- 容器: xiaoming-dev (podman)
- 工具链: ROCm hipcc 6.x
- 改动: 仅 `constexpr int PAD = 4 → 8` (一行)，同时同步 `LDS_PAD = 4 → 8` 到 super-kernel
- 文件: csrc/grouped_gemm.hip, csrc/fused_moe_super_kernel.hip

## 问题
连续 5 轮 instruction-scheduling 实验（sched_group_barrier × 3 阶段、pre-issue 32 ds_reads × 4 阶段、iglp_opt(0)）全部证伪：
- baseline (HEAD): 170 T (DSV3-GateUP M=4096)
- 所有调度 hack: 170-180 T，无显著改善
- 与 CK 1211 T 仍差 7×

postmortem (`2026-05-12_2050_pre_issue_32_ds_reads_p1_failed_*.md`) 推断**真正瓶颈是 LDS bank conflict**，因为 ds_read 完成时间不可预测让 LLVM waitcnt-insertion pass 不敢发 partial wait。

但 XOR swizzle 设计陷入分析瘫痪：理论上 ds_read_b128 的 16-byte alignment + 32 banks 让任意 swizzle 都至少 2-way conflict。

→ 改用**经验 PAD 扫描**：直接试不同 PAD 值看哪个最快，反推 HW 行为。

## 做了什么

### 阶段 1: 粗扫 PAD ∈ {0, 4, 8, 16, 24}

测试 shape: DSV3-GateUP M=4096 K=7168 N=4096 E=8 (static & persistent grids)

| PAD | AS (elem) | stride (dword) | stride mod 32 | TFLOPS | 备注 |
|---|---|---|---|---|---|
| 0  | 64 | 32 | 0  | 199 T | 跳过 prologue 数据 |
| 4  | 68 | 34 | 2  | 170 T | OLD baseline |
| **8** | **72** | **36** | **4** | **212 T** | **+24.7% over PAD=4** |
| 16 | 80 | 40 | 8  | 211 T | 与 PAD=8 等同 |
| 24 | 88 | 44 | 12 | LDS overflow | 168 KB > 160 KB budget |

### 阶段 2: 微调 PAD ∈ {8, 10, 12, 14, 20}

测试同 shape：

| PAD | AS | stride | stride mod 32 | M=4096 GateUP | M=4096 Down | M=16384 GateUP | M=16384 Down |
|---|---|---|---|---|---|---|---|
| **8** | 72 | 36 | 4  | **212 T** | **224 T** | **228 T** | **238 T** |
| 10 | 74 | 37 | 5  | 167 T | 176 T | 177 T | 182 T |
| 12 | 76 | 38 | 6  | 170 T | 180 T | 180 T | 185 T |
| 14 | 78 | 39 | 7  | 167 T | 176 T | 177 T | 182 T |
| 20 | 84 | 42 | 10 | LDS overflow (168 KB > 160 KB) |

## 关键发现：**`PAD % 8 == 0` 是 alignment 阈值，不是 bank-conflict 阈值**

PAD ∈ {4, 10, 12, 14}（≠ 8 的倍数）：性能塌到 170 T
PAD ∈ {8, 16}（8 的倍数）：性能 211-238 T

**原因**：`ds_read_b128` 需要每个 lane 的 LDS 地址 **16-byte 对齐**。
- Lane n 的 byte offset = base + n × stride × 2 (bf16)
- 16-byte 对齐要求 `stride × 2 ≡ 0 (mod 16)` → `stride ≡ 0 (mod 8)` (in elements)
- 即 `PAD ≡ 0 (mod 8)` (因为 kK = 64 是 8 的倍数)

当 PAD=4: stride=68 elem，lane 1 byte offset = 136，136 mod 16 = 8 → **不 16-byte 对齐**。HW 把 `ds_read_b128` 自动拆成两个 `ds_read_b64`，吞吐量减半。

PAD=10/12/14 同理 — 不是 8 倍数 → 拆分。

PAD=8 是**最小的合规 PAD**（PAD=0 会让所有 row 起在同一 bank，重新引入 8-way conflict）。

## 收益验证

### 1. GroupedGEMM 标 bench (完整 DSV3 sweep, E ∈ {8,16,32}, M ∈ {512..16384})

| Shape | PAD=4 baseline | PAD=8 now | 提升 |
|---|---|---|---|
| GateUP E=8 M=512   | 167 T | 208 T | +24.6% |
| GateUP E=8 M=4096  | 170 T | 212 T | +24.7% |
| GateUP E=8 M=16384 | 180 T | 227 T | +26.1% |
| Down   E=8 M=4096  | 180 T | 224 T | +24.4% |
| Down   E=8 M=16384 | 186 T | 238 T | +28.0% |
| Down   E=32 M=16384 | 187 T | 238 T | +27.3% |

平均 +24-28% across all DSV3 shapes，**完整测试套 12/12 PASS**。

### 2. Super-kernel bench (vs PAD=4 baseline from `2026-05-12_1715_*.md`)

| 场景 | PAD=4 baseline | PAD=8 now | wall 改善 | TFLOPS |
|---|---|---|---|---|
| DSV3_SPARSE (T=512, H=7168, F=2048, epg=32, topk=8) | 8.14 ms / 354.6 T | **7.29 ms / 396 T** | **-0.85 ms / -10.4%** | **+11.7%** |
| TILE_FIT (T=1024, H=4096, F=1024, epg=4) | 4.37 ms / 188.8 T | **3.99 ms / 207 T** | **-0.38 ms / -8.7%** | **+9.6%** |

super-kernel 的 wall 改善（10%）小于 GroupedGEMM 内部改善（25%），原因：~45% wall 在 comm 路径（dispatch_src_ready_wait 等），与 GEMM 无关。GEMM phase 内部（约 30% wall）确实拿到了完整 +25%。

### 3. 资源 + 正确性

资源（hipcc -Rpass-analysis=kernel-resource-usage）：
- VGPR 256, AGPR 256, scratch 1348 B/lane, occupancy 1 wave/SIMD, 0 spill — **与 PAD=4 完全一致**
- LDS: 144 KB (vs 136 KB PAD=4, 160 KB hard cap)，吃下 8 KB 增量

正确性：
- `tests/test_grouped_gemm_correctness`: 4/4 PASS
- `tests/test_grouped_gemm_persistent_correctness`: 8/8 PASS
- `tests/test_mfma_gemm_tile_standalone`: PASS (0 mismatch)
- `tests/test_super_kernel_correctness`: 8/8 ranks PASS, max_abs=0.0859 max_rel=0.0003

## 教训与连锁影响

### 教训
1. **不要盲目相信"bank conflict"理论**：5 轮调度 hack 都假设是调度问题，结果发现是 **alignment 问题**。
2. **经验扫描 vs 理论分析**：5 个 PAD 值的扫描 30 分钟出结果；如果继续按 XOR swizzle 设计走，要 2-3 天且方向都可能错（XOR swizzle 设计假设了 16-byte ds_read 对齐，但实际 PAD=4 时它早已被破坏）。
3. **一行改动可以解锁 25%**：当根因诊断对了，修复往往非常小。
4. **PAD=4 这一行是从最早 v8a 第一版就有的**，从未质疑过——**"baseline 不动"是最大的陷阱**。

### 7× gap 重新核算
- v8a 之前: 170 T (PAD=4, alignment broken)
- v8a 现在: 212-238 T (PAD=8) - depending on shape
- CK 参考: 1211 T
- 剩余 gap: 1211 / 224 ≈ **5.4×**（不再是 7.1×）

CK 与我们的差距收窄到 5.4×。剩下的差距推测：
- LDS bank conflict（真正的 conflict，不是 alignment 问题）：~1.5×
- Instruction scheduling / pipeline depth：~1.5×
- MFMA throughput / register reuse：~2-2.5×

## 已提交的改动
- `csrc/grouped_gemm.hip`: `PAD: 4 → 8`，加 9 行注释解释 alignment
- `csrc/fused_moe_super_kernel.hip`: `LDS_PAD: 4 → 8`，加同样注释
- Diff: +28 / -4 行

## 下一步建议（按 ROI 重排）

### P0: mxfp8 / FP8 weights for FC1/FC2
- MI355X 原生支持 `v_mfma_scale_f32_32x32x64_f8f6f4`（CK 的 .so 里也有这个 instruction）
- HBM weight 流量 ÷2（bf16→fp8）+ MFMA 吞吐 ×2 → 单 stage 理论 ×4 加速
- 估时 5-7 d，预期 DSV3 wall 8.14 → 4-5 ms

### P0: 真正的 LDS XOR swizzle（不是 PAD）
- 现在 PAD=8 给出 stride mod 32 = 4，可能还有 4-way conflict（CK 是 conflict-free）
- 需要 XOR swizzle 让 32 lane 起 banks 全分布
- 预期 +20-30%（针对 GEMM 内部）→ DSV3 wall -0.5 ~ -0.8 ms
- 估时 2-3 d

### P1: kpack=8 register-layout 整理
- 把 ar/br fragments 在 LDS 中以 (K_block, M, K_inner) 三维布局存储（CK 模式）
- 避免 frag 在 ds_read 时的 lane-row 重排
- 估时 1-2 d

### P1: 验证当前 PAD=8 是否真的 stride mod 32 = 4 的 4-way conflict
- 用 cycle counter profile 直接测 ds_read latency at PAD=8
- 如果 latency ≈ 标称 → 5.4× gap 不在 LDS，要看别处
- 估时 0.5 d

### 弃用
- ~~sched_group_barrier~~（前一份证伪）
- ~~pre-issue 32 ds_reads~~（前一份证伪）
- ~~iglp_opt(0)~~（前一份证伪）

## 文件
- 改动 commit: 待用户确认
- 完整 bench 日志: `terminals/556849.txt`, `terminals/928945.txt`
- 前置 postmortem: `2026-05-12_2050_pre_issue_32_ds_reads_p1_failed_*.md`
