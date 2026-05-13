# 2026-05-12 19:55 CST  P0 sched_group_barrier 实验失败：编译器 waitcnt-insertion pass 锁死 lgkmcnt(0)

## 环境
- 主机: mi355-gpu-26 (AMD MI355X gfx950, 256 CU)
- 容器: xiaoming-dev (podman)
- 工具链: ROCm hipcc 6.x, LLVM AMDGPU pre-RA scheduler
- Baseline: `csrc/grouped_gemm.hip` HEAD = `c70d6a3a` (intrinsic MFMA + 8-phase 协同加载，正确性 12/12 PASS)
- 参照: `primus_turbo` CK GroupedGEMM `~1211 TFLOPS` for DSV3-GateUP M=4096 K=7168 N=4096 E=8
- v8a 当前: `170 TFLOPS`（同 shape）

## 问题（背景）
前一份分析（`2026-05-12_1920_ck_pipeline_v3_root_cause_sched_group_barrier.md`）得出结论：

> v8a 和 CK V3 在 tile/warp/MFMA/per-K-tile 指令数完全一致，唯一差异是 CK V3 用 `__builtin_amdgcn_sched_group_barrier` 显式告诉 LLVM AMDGPU 调度器如何 interleave `buf_load / ds_write / ds_read / MFMA`，预期 v8a 加上同样的 hint 可以拿到 3-4× 提升。

按此假设实施。

## 做了什么

### 阶段 1：原地添加 hint（最小侵入）
- 在 `csrc/grouped_gemm.hip` 新增 `gemm_v8a_hotloop_sched_hint()`，emit 96 个 `__builtin_amdgcn_sched_group_barrier(mask, size, 0)`，参数完全对齐 CK V3 HotLoopScheduler：
  - Stage 1（×16, A+B buf_load）：`[DS_WRITE, MFMA, VMEM, MFMA×2]` 重复
  - Stage 2（×16, A+B ds_read 对）：`[DS_READ×2, MFMA]` 重复
  - 总计 16 DS_WRITE + 16 VMEM + 32 DS_READ + 64 MFMA = 完全匹配 K-tile 指令数
- 调用点：放在 K-tile body 末尾、`if (has_next) { … }` 的 Phase 3 块内、`__syncthreads()` 之前。
- 资源不变：256 V + 256 A，1 wave/SIMD，无 spill，1348 byte scratch。
- 正确性：`test_grouped_gemm_correctness` 4/4 PASS，`test_grouped_gemm_persistent_correctness` 8/8 PASS。
- **性能：~180 TFLOPS（DSV3-GateUP），完全没动。**

诊断 asm（`benchmarks/results/v8a_sgb.s`）：
- 16 条 `ds_write_b128` 在 line 2371-2409；
- 96 条 `sched_group_barrier mask(…)` 注释在 line 5201-5296；
- 之间有 `s_barrier` (line 2411, 3101)。

→ **`sched_group_barrier` 指令落在了与 ds_writes 不同的 basic block（被 `s_barrier` 隔开），LLVM 调度器无法跨 BB 重排**。

### 阶段 2：重构主循环让 K-tile body 成为单一 straight-line BB
仿照 CK V3 主循环结构：
- 把 `__syncthreads()` 从 K-tile body 末尾**搬到开头**（drain 上一轮 ds_writes）；
- 去掉两个 `if (has_next)` 守卫——Phase 1 的 buf_load 始终发出（依赖 `make_buffer_rsrc` 的 num_records 自动 clamp 越界为 0），Phase 3 的 ds_write 始终写入（最后一轮写到没人读的 buffer，浪费 16 LDS 写）；
- HotLoopScheduler hint + `__builtin_amdgcn_sched_barrier(0)` 放在 K-tile body 末尾。

资源不变。`test_grouped_gemm_persistent_correctness` 8/8 PASS。

诊断 asm（`benchmarks/results/v8a_sgb2.s`）：
- 16 条 ds_writes 在 line 2366-2404；
- sched_group_barrier 注释和 mfma/ds_read **真正交错出现** 在 line 3221-3329 区域；
- 仍然有 `s_barrier` 在两者之间 — 但比阶段 1 距离近了。

→ 部分起效，scheduler **确实在重排** ds_read/mfma 顺序；但还有一个深层问题。

**性能：~180 TFLOPS，仍然没动。**

### 阶段 3：去掉显式 wait_lgkm，让 compiler 自行决定 lgkm 值
原代码每个 K-step mfma 前有 `wait_lgkm(has_next_step ? (kM+kN) : 0)`。这是显式的 `s_waitcnt lgkmcnt(8)`。怀疑此显式 wait 会让 compiler 重复添加 `lgkmcnt(0)`，于是把 explicit wait 删掉，让 compiler 完全自由。

正确性继续 8/8 PASS。**性能：仍然 ~180 TFLOPS。**

## 根因诊断

统计阶段 3 .s 文件中 `s_waitcnt lgkmcnt(N)` 的分布（`benchmarks/results/v8a_v3.s`）：

```
     53 	s_waitcnt lgkmcnt(0)
      2 	s_waitcnt lgkmcnt(1)
      1 	s_waitcnt lgkmcnt(2)
```

**95%+ 的 LDS 等待都是 `lgkmcnt(0)`**。LLVM AMDGPU 的 waitcnt-insertion pass 在每个 `v_mfma` 前都插入 `s_waitcnt lgkmcnt(0)`，强制 LDS 队列完全排空。

原因（在 asm 区段 line 3200-3330 观察到）：
```
ds_read_b128 v[240:243], v7              # 第 1 个
ds_read_b128 v[228:231], v7 offset:4352  # 第 2 个
ds_read_b128 v[180:183], v7 offset:8704  # 第 3 个
ds_read_b128 v[144:147], v7 offset:13056 # 第 4 个
ds_read_b128 v[176:179], v7              # 第 5 个
s_waitcnt lgkmcnt(0)                     # ← compiler 插的
v_mfma_f32_32x32x16_bf16 a[0:15], v[240:243], v[176:179]
```

这里 mfma 用第 1 个 ds_read 的 v[240:243] 和第 5 个 ds_read 的 v[176:179]。理论上 compiler 可以发 `lgkmcnt(0)`，但若用 LDS in-order 完成假设，发 `lgkmcnt(0)` 也合理（要等第 5 个完成，意味着前 5 个都要完成）。

但 compiler 没有发更激进的 partial wait（例如 `lgkmcnt(3)` 等到第 5 个 = ≤3 outstanding），即使理论上 LDS-read-to-same-thread 是 in-order。这是 LLVM AMDGPU 当前版本 waitcnt-insertion pass 的保守策略。

**结论**：sched_group_barrier 能控制 **指令顺序**，但控制不了 **waitcnt 插入策略**。即使把 32 个 ds_reads 和 64 个 mfmas 完美 interleave，每个 mfma 前的 `lgkmcnt(0)` 仍然让 MFMA pipeline 串行化，故 v8a 卡在 ~170-180 TFLOPS（≈ MFMA 峰值的 13%）。

## 关键发现
1. **CK V3 ~1211T 不是单靠 sched_group_barrier 拿到的**。CK 必然还做了一件我们没做的事——很可能是：
   - **手写 inline-asm 的 `s_waitcnt lgkmcnt(N)`**（N 是合理的 partial wait），跳过 LLVM 的保守 waitcnt pass；或
   - **更深的 ds_read 寄存器堆叠**——一次性 issue 完所有 32 个 ds_reads 到 32 组不冲突的寄存器，然后做 64 个 mfmas（没有中间 ds_read），让 compiler 容易发 partial waitcnt。
2. 我们当前 ar[]/br[] + ar_next[]/br_next[] 是 2-deep 双 buffer，prefetch 深度不够；compiler 看到 mfma 紧贴对应 ds_read，只能保守发 lgkm(0)。

## 已撤销的实验
- `git stash`：`sgb-experiment-failed-no-gain`（包含 hint helper + 主循环重构 + 去掉 wait_lgkm）
- 当前 HEAD = `c70d6a3a`，未引入回归

## 数据
| 阶段 | 改动 | DSV3-GateUP M=4096 |
|---|---|---|
| baseline | HEAD | 169-170 T |
| 1 | + sched_group hint（保留 if/else 结构） | 170-183 T |
| 2 | + 重构 main loop（无 if/else, sync 移到顶部） | 169-181 T |
| 3 | + 删除 explicit wait_lgkm | 170-178 T |
| CK 参考 | — | **1211 T** |

## 下一步建议（按 ROI 重排）

### P0a: 全 K-tile ds_read 一次性 issue + 全 mfma 拉直
- 把 4 个 K-step 的 32 个 ds_reads 全部 issue 到 32 个**不同的**寄存器（拆 `ar[m]` → `ar_all[KS][kM]`）
- 然后做 64 mfmas，期间不再有 ds_read
- compiler 看到完整的 64 mfma + 32 已 issue 的 ds_reads，可能 emit `lgkmcnt(31), lgkmcnt(30), …`，pipeline 起来
- 风险：register pressure ↑（128 dword for ar + 128 for br = 256 V），可能 spill；如果 spill，需要从 256+256 V/A 重新平衡
- 预期：3-5× 提升（若 compiler 配合）

### P0b: 手写 inline-asm `s_waitcnt`（仿 CK）
- 在每个 mfma 前用 inline asm 显式发 `s_waitcnt lgkmcnt(N)`，N 根据 ds_read 数手算
- 风险：脆弱，每次改 K-step 数都要重算 N
- 预期：1.5-2× 提升

### P0c: 升级 ROCm/LLVM 试试新版 waitcnt pass
- 新版 LLVM 可能有更激进的 partial-waitcnt 策略
- 低风险但可能无效

### P1: 改 N_Tile / K_Tile 形状降低 ds_read/mfma 比
- 把 K_Tile 从 64 升到 128 → mfma 数 ×2，ds_read 数 ×2，ratio 不变，但单 K-tile 内 MFMA 更长可以 amortize waitcnt 开销
- 风险：占用更多 LDS，可能 occupancy 下降

## 教训
1. **不要假设 compiler 听话**——sched_group_barrier 只能控制顺序，不能控制 waitcnt
2. **微观对齐指令数 ≠ 性能对齐**：CK 和 v8a 指令数相同，但 CK 的"看不见的细节"（可能是 inline-asm waitcnt 或更深的 prefetch）才是 7× gap 的来源
3. asm 分析时要看 **lgkmcnt(N) 的具体 N 值分布**，而不是只数指令条数

## 文件
- 失败实验 stash: `git stash list | grep sgb-experiment-failed`
- asm 对比: `benchmarks/results/v8a_sgb.s`（阶段 1）`v8a_sgb2.s`（阶段 2）`v8a_v3.s`（阶段 3）
- bench 对比: `benchmarks/results/v8a_now.txt` `ck_now.txt`
