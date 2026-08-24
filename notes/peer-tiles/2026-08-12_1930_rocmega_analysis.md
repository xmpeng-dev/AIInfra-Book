# ROCMega 分析：范式命题的独立复现 + 载体问题的第三个答案

> **When**: 2026-08-12 19:30 UTC+8
> **Where**: 登录机，纯代码与文档阅读（本轮未跑 GPU）
> **Context**: 分析 `/perf_apps/xiaoming/slab/3rd/ROCMega`（HEAD `e67710f`）—— 另一个 mega MoE 实现。它同时回答了 [立项文档](./2026-08-12_1530_repo-charter-and-primitive-api.md) 里的两个悬案：范式命题是否站得住、载体到底选什么

## TL;DR

1. **架构是混合的，这是载体问题的第三个答案**（我前两次都答错了）：**HIP C++ 写融合骨架 + FlyDSL 生成的 GEMM tile 以 AMDGPU bitcode 交付**，用 `-Xclang -mlink-builtin-bitcode` 链进 HIP TU，wrapper 是 `alwaysinline` → tile **内联进调用方 WG，跑在完整寄存器文件上，无 call-ABI spill**。
2. **范式命题得到独立复现。** ROCMega 的结构是 `prologue → dispatch+FC1 → SwiGLU → FC2+combine+reduce` —— 与 gen-3 (FlyDSL mega) **完全相同的四腿成对融合**。两个独立实现、不同语言、同一结构，都在生产规模上大幅赢过库路径；而 gen-1 的单核全融是 0.29×。
3. **拿到了 gen-3 forward 的直接测量：5.978 ms**（同环境、钉时钟 2400 MHz、逐字节匹配 routing）。此前我是用 `fwd+bwd − bwd` 减出来的 ~7.80 ms，**该估计作废**。
4. **它自带的 root-cause 分析，正是"HK 贡献纪律而非 GEMM"这一主张的实测版**：ROCMega 与 FlyDSL 的全部 1.1 ms 差距 = **FC2 NT tile 在 1 WG/CU régime 下缺 `ks→ks+1` K-step lookahead**。值 20% GEMM 吞吐、15% 整个 forward。

## 1. 它是什么

`README.md` 自述：「a development repository for exploring **mega kernels in the ROCm ecosystem**」，第一个目标是 Mega MoE —— 把 dispatch / 跨 rank 通信 / GEMM / 激活 / combine / 反向「fused into **a small number of** high-performance kernels」。

**注意措辞：a small number of kernels，不是 one kernel。** 这与 gen-1 的单核全融是不同的设计意图。

### 代码结构 = 成对融合

| 文件 | 融的是什么 |
|---|---|
| `csrc/moe/dispatch_fc1_fwd.hip` | dispatch（XGMI push）× FC1 grouped GEMM |
| `csrc/moe/fc2_combine_fwd.hip` | FC2 GEMM × combine push × reduce |
| `csrc/moe/combine_fc2_bwd.hip` | combine-bwd × FC2-bwd dY × dW（bwd leg 1） |
| `csrc/moe/fc1_dispatch_bwd.hip` | FC1-bwd dgrad × grad_x dispatch-back push/reduce × wgrad（bwd leg 2） |
| `csrc/moe/swiglu/swiglu_{fwd,bwd}.hip` | 单独 |
| `csrc/moe/dispatch/{prologue,transport}.hip` | 路由元数据 / 传输 |

**每个 kernel 恰好融「一次通信 × 一个 GEMM」。** 这正是 [立项文档 §0.3](./2026-08-12_1530_repo-charter-and-primitive-api.md) 的范式命题。

### GEMM tile 的交付方式（载体问题的答案）

`csrc/gemm/gemm_tiles.h` 头注释写得很清楚：

> The tiles are shipped as committed AMDGPU bitcode in `csrc/gemm/tiles/*.bc` and linked into the consuming HIP translation unit via `-Xclang -mlink-builtin-bitcode`. Each tile is a 256x256, 512-thread MFMA (`v_mfma_f32_32x32x16`) software-pipeline that a consumer calls ONCE PER WORKGROUP; **the linked wrapper is alwaysinline, so the tile inlines into the caller WG and runs with the full register file (no call-ABI spill)**.

已提交的 tile（7 个 `.bc`，141–182 KB）：

```
gemm_nt_bf16_mfma32x32x16_256x256_k{2048,4096,7168,rt}.bc
gemm_fc1bwd_nn_k4096_tn_m4096n7168_sharedlds.bc
gemm_fc2bwd_nn_k7168_tn_m7168n2048_sharedlds.bc
gemm_tn_wgrad_bf16_mfma16x16x32_256x256_m4096n7168.bc
```

**bitcode 的符号表证明它们是 FlyDSL 出的**，不是 hipcc 从 HIP 源码编的：

```
__shared_alloc_0 ... __shared_alloc_7      ← 8 个手工 LDS 槽（4-buffer × 2 操作数）
llvm.amdgcn.s.setprio                      ← s_setprio
llvm.amdgcn.raw.ptr.buffer.load.lds        ← async direct-to-LDS
llvm.amdgcn.mfma.f32.32x32x16.bf16         ← 前向/dgrad tile
llvm.amdgcn.mfma.f32.16x16x32.bf16         ← wgrad tile（换了 MFMA 形状）
producer: 22.0.0git f58b06dce…（= 容器内 ROCm 7.2.1 的 clang）· llvm-link
```

`__shared_alloc_N` 是 FlyDSL `SmemAllocator` 的签名（HIP 源码会产出 C++ mangled 名）；`s_setprio` + direct-to-LDS 也都是此前在 `primus_turbo/flydsl/` 里盘过的 FlyDSL 原语。CMakeLists 亦直接标注两条 bwd 腿为 `flydsl` / `flydsl-only`。

一个值得记的工程细节：两个 128 KB LDS tile 预合并进一个模块时，**同名 LDS 全局是 `linkonce_odr` → 两块 128 KB ODR 折叠成一块**，融合 kernel 才装得下（`gemm_tiles.h:27` 注释）。

## 2. 性能

环境：`chi2761`（SLURM `mi355x` 独占），8× MI355X gfx950，EP8，容器 `xiaoming-dev`（`tasimage/primus:pr-867`），ROCm 7.2.1 / clang 22，**`perf_determinism` + SCLK 钉在 2400 MHz** —— 这是本项目所有数据里测量纪律最好的一份。

### DSv3 T=8k forward，同环境、逐字节匹配 routing（`doc/fwd_vs_flydsl.md`，2026-07-11）

| leg | ROCMega (µs) | FlyDSL-mega (µs) | delta |
|---|---:|---:|---|
| prologue | 142 | 157 | ROCMega −11% |
| dispatch + FC1 | 3595 | 3178 | **FlyDSL −12%** |
| SwiGLU | 146 | 185 | ROCMega −21% |
| FC2 + combine + reduce | 3181 | 2445 | **FlyDSL −23%** |
| **GPU 合计** | **7065** | **5978** | **FlyDSL −15%** |
| FC1+FC2 GEMM | 812 TFLOP/s/GPU | **971** | FlyDSL +20% |

复测（`doc/notes/2026-07-14.md`，换到 `chi2866`）：ROCMega 7.200 ms / 801.7 TFLOP/s，与 7.105 差 1.3%（run-to-run）。
反向：**fwd+bwd clean wall 20.84 ms**（balanced）/ 22.01 ms（random）。
T=2k forward：2.250 ms（balanced）/ 2.478（random）。

### 放进四代对照（⚠ 见下方限定）

| 实现 | forward @ DSv3 T=8k EP8 | 相对 PyTorch+RCCL | 结构 | 载体 |
|---|---:|---:|---|---|
| PyTorch+RCCL（2026-05-13） | 18.64 ms | 1.00× | 库路径，无 overlap | — |
| **gen-1** monolith-moe（2026-05-13） | **64.47 ms** | **0.29×** | **单核全融**（4 role / 5 sub-phase） | HIP C++ |
| **ROCMega**（2026-07-11） | **7.105 ms** | **2.6×** | **四腿成对融合** | HIP 骨架 + FlyDSL bitcode tile |
| **gen-3** FlyDSL mega（2026-07-11） | **5.978 ms** | **3.1×** | **四腿成对融合** | 纯 FlyDSL |

⚠ **限定**：PyTorch+RCCL 那一行是 2026-05-13 在 `mi355-gpu-26` 上测的，**未钉时钟、不同节点、不同日期**，而 ROCMega/gen-3 两行是同环境钉 2400 MHz。所以 2.6× / 3.1× 是**量级可信、精确值待 M0 复核**。这不改变结论方向（2 倍以上的差距远超时钟漂移的 ±30%）。

## 3. 对范式命题的意义

[立项文档 §0.3](./2026-08-12_1530_repo-charter-and-primitive-api.md) 的命题是：

> AMD 上的 mega 范式是「stage 级成对融合」：一次通信配一个保持库级 tile 几何的 GEMM，两条腿等长。整层单核融合会切碎 GEMM 的算术强度，该损失随 token 数线性增长，必然在生产规模翻盘。

ROCMega 把这条从「一次观察」升级为「**独立复现**」：

- **同一结构**：两个实现都是 `prologue → dispatch+FC1 → SwiGLU → FC2+combine+reduce`，都是每 kernel 融一次通信 × 一个 GEMM。
- **不同语言**：一个纯 FlyDSL，一个 HIP C++ 骨架 + bitcode tile。
- **同一量级的胜利**：2.6× 与 3.1×，而单核全融的 gen-1 是 0.29×。
- **两条腿都接近等长**：ROCMega dispatch+FC1 3.595 ms vs FC2+combine 3.181 ms；gen-3 是 3.178 vs 2.445。

两个人各自摸索、收敛到同一个结构，比一个实现赢了更能说明这是**范式**而非**调参运气**。

## 4. 对 HK 角色的意义：主张的实测版

[立项文档 §0.4](./2026-08-12_1530_repo-charter-and-primitive-api.md) 主张 HK 贡献的是「让 GEMM 与别的 role 共驻而不掉效率的纪律」，而不是更快的 GEMM。当时这是推理。**`fwd_vs_flydsl.md` 的 root cause 把它变成了实测**：

> **The FC2 NT (write-through) tile lacks a `ks→ks+1` K-step lookahead in the 1-WG/CU regime.** ROCMega's NT tile normally hides `wait_lgkm` stalls by running 2 WG/CU (sibling-wave overlap) — `mfma_common.h:25-29` states that assumption outright. **But the fused FC2 tile carries 128 KB static LDS, so it runs at 1 WG/CU with no sibling wave to hide the stall** … FlyDSL's tile is an explicit single-WG software pipeline (**4-buffer distance-2 + `s_setprio` + `vmcnt(3)`**).

机制链条：

```
融合 kernel 的 LDS 预算（128 KB 静态）
   → 占用降到 1 WG/CU
   → 没有 sibling wave 掩盖 wait_lgkm
   → 独立 GEMM 上成立的流水假设在融合上下文里失效
   → GEMM 掉 20%（971 → 812），整个 forward 掉 15%
```

**这就是「共驻代价」的精确形态**，而且是可量化的：一个在独立场景下调好的 GEMM tile，搬进 mega kernel 后因为占用régime 变了而失配。

**与本项目今天的 HK 实测直接咬合**（见 [HK GEMM note](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md)）。

> **⚠ 2026-08-13 07:20 更正**：本段初稿写「HK 出厂 GEMM 也是 2 waves/SIMD 假设 → 丢进 mega 会撞同一堵墙」，**这是把占用régime 读反了**。`WARPS_M=2 × WARPS_N=4 = 8 waves = 512 线程`，一个 WG 铺在 4 个 SIMD 上就是 2 waves/SIMD —— 编译器报的 `Occupancy: 2 waves/SIMD` 是**单 WG 占满 CU（1 WG/CU）**，不是两个 WG 共驻。HK **不依赖** sibling-WG 重叠。

HK 的循环体（`256_256_64_32_with16x32.cpp:121-149`）实际是**单 WG 8-wave ping-pong**：

```cpp
if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }   // 条件 barrier = 角色互换
asm volatile("s_waitcnt vmcnt(4)");                     // 手调流水深度
__builtin_amdgcn_s_setprio(1);
mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
__builtin_amdgcn_s_setprio(0);
```

即论文 §5.2 的 8-wave ping-pong + §8.3 的 `s_setprio`。**三个 tile 的对照因此是：**

| tile | 靠什么掩盖延迟 | 1 WG/CU 下 |
|---|---|---|
| ROCMega HIP NT tile | **2 WG/CU 的 sibling wave**（`mfma_common.h:25-29` 明写此假设） | **失效**，−20% GEMM |
| FlyDSL bitcode tile | 单 WG 显式软流水：**4-buffer** distance-2 + `s_setprio` + `vmcnt(3)` | 有效（971 TFLOP/s） |
| **HK tile** | 单 WG **8-wave ping-pong**：条件 barrier + `s_setprio` + 手调 `vmcnt/lgkmcnt`，**2-buffer** tic/toc + 循环内 lookahead | **结构上属于正确类别，缓冲深度浅一档（2 vs 4）；融合上下文内未测** |

→ 修正后的结论：**HK 的 tile 与 FlyDSL 的 tile 属于同一设计类别（单 WG 显式流水），与 ROCMega 失配的那个不是。** 这反而是「HK 进 mega」少见的正面论据，但需实测（见 [§7](#7-megamoe--hk-的可行性2026-08-13-0720)）。

**一个可立刻验证的联动**：今天测出 HK 的 `BLOCK_SIZE=128` 在 MoE 形状上反超 hipBLASLt，而 **BS=128 同时把 LDS 用量降到约 1/4** —— 这正对上 ROCMega 自己列的第 2 条 actionable：「raise FC2 tile occupancy to 2 WG/CU (smaller K-tile / fewer buffers / less LDS) so sibling waves hide the stall — no pipeline rewrite」。**小 tile 可能同时解决 MoE 形状的网格饥饿与融合上下文的占用失配。**

## 5. 对立项文档的修订

| 条目 | 原判断 | 依 ROCMega 修订 |
|---|---|---|
| **载体** | 先写 HIP C++ header（错），改为纯 FlyDSL（不完整） | **混合**：HIP C++ 骨架（通信 / 融合 / 角色 / 调度）+ FlyDSL bitcode tile（GEMM 内循环），以 `-mlink-builtin-bitcode` + `alwaysinline` 为 ABI 边界。**这个分界恰好落在两者各自的强项上**，且有实测支撑（FlyDSL tile +20% GEMM） |
| 范式命题 | 一次观察（gen-1 vs gen-3） | **独立复现**（ROCMega 与 gen-3 同结构、不同语言、同量级胜利） |
| gen-3 forward | 由减法估得 ~7.80 ms | **直接测得 5.978 ms**（同环境钉时钟），原估计作废 |
| M0（三方 A/B） | 仍需 | **范围可缩小**：ROCMega 已提供 gen-3 与 ROCMega 的同环境对照，**M0 只剩"刷新 PyTorch+RCCL 基线"这一条**（现基线是 5 月、异节点、未钉时钟） |
| HK 的角色 | 推理：贡献纪律非 GEMM | **实测确证**，且给出了代价的量化（1 WG/CU 失配 = 20% GEMM / 15% forward） |

## 6. 待查

1. **ROCMega 与 gen-3 的 backward 对照缺失。** 只有 ROCMega 的 fwd+bwd 20.84 ms，没有 gen-3 的同环境反向数。而 MegaMoE 那边 fp8 fwd+bwd 是 14.117 ms（不同精度、不同节点）。要比得先对齐精度。
2. **`3rd/kgraft` 不在本机**（`gen_fc1bwd_shared_lds.sh` 等生成脚本）。要复现 tile 生成流程得先找到它。
3. **前向 tile 的 provenance 未在仓库内标注**（只有 bwd 两腿标了 flydsl）；符号表证据指向 FlyDSL，但没有直接文档。
4. **ROCMega 与 `notes/rocmoe/` 的关系待厘清** —— 是同一条线的演进，还是并行的两个 checkout？本项目的「三代」叙事可能要改成四代或两支。

## 7. MegaMoE + HK 的可行性（2026-08-13 07:20）

问题：把 HK 的 GEMM 放进 mega MoE，值不值。分三种落点评估。

### 7.1 HK 带进来的与缺的

| 带进来 | 证据 |
|---|---|
| **单 WG 8-wave ping-pong**，正是融合上下文强制的 1 WG/CU régime 的正解 | §4 更正后的对照表 |
| MoE 形状上调优后**反超 hipBLASLt 3/5 格**（1.07–1.31×） | [HK GEMM note §调优](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md) |
| C++ tile 原语，可读、可组合、可引用；不需要 kgraft 那套 bitcode ABI | `gemm_tiles.h` 的 ABI 复杂度即代价对照 |

| 缺的 / 成本 | 说明 |
|---|---|
| **无 grouped / 变长-M** | MoE 要 per-expert 变长 M。**这是最大的单项工程** |
| **不是 device function** | HK GEMM 是 `__global__`，自带 grid / chiplet swizzle。要融合必须把 K-loop 抽成 `__device__`，接受外部传入的 LDS 指针与 tile 坐标 |
| **LDS 预算冲突** | `dynamic_shared_memory()` 直接要 `MAX_SHARED_MEMORY`。BS=256 时 `As[2][2]+Bs[2][2]` = 8 × `st_bf<128,64>` × 2 B = **128 KB** —— 与 ROCMega 那个逼到 1 WG/CU 的数字**完全一致**。BS=128 降到 **64 KB** |
| 缓冲深度浅一档 | 2-buffer vs FlyDSL 的 4-buffer distance-2 |

**一个耐人寻味的重合**：BS=128 同时(a)在 MoE 形状上反超 hipBLASLt、(b)把 LDS 从 128 KB 砍到 64 KB 给 comm staging 腾地方。**两个独立约束指向同一个配置。**

### 7.2 三种落点的判断

| 落点 | 判断 | 理由 |
|---|---|---|
| **换掉 gen-3（纯 FlyDSL）的 tile** | **低价值** | 那个槽位已经 971 TFLOP/s；HK 要连本带利地补上 grouped driver + device-function 重构，才可能打平。期望收益 ≈ 0 |
| **补 ROCMega 的 FC2 NT tile** | **较高价值** | ROCMega 有**已确诊的 20% GEMM 缺口**，病因正是"依赖 2 WG/CU"，而 HK 属于正确设计类别；且 ROCMega 是 HIP 骨架，接 C++ tile 比取 FlyDSL bitcode 自然 |
| **作为"融合régime 下的 tile 设计"研究** | **最高价值** | 现在有三个数据点（HIP 2-WG/CU tile 失败、FlyDSL 4-buffer 成功、HK 8-wave 2-buffer 未测）。**"一个 GEMM tile 要长成什么样才能住进 mega kernel"是一个尚无答案的问题**，而 HK/CUTLASS/CK 全是独立régime 的设计 |

### 7.3 建议的顺序（先做不需要 HK 的那一步）

**S0 —— 不碰 HK，先定机制。** 执行 ROCMega 自己列的 actionable #2：把 FC2 NT tile 的 LDS 压到能跑 2 WG/CU，看 20% GEMM 缺口是否闭合。

- 闭合 ⇒ 病因是**占用**，与谁的 tile 无关，HK 的结构优势不成立 → 不必上 HK
- 不闭合 ⇒ 病因是**流水深度**，需要单 WG 深流水 tile → HK 才有落点

**S1 —— 只在 S0 指向流水深度时做。** 把 HK 的 K-loop 抽成 `__device__`（接外部 LDS 指针），塞进 ROCMega 融合 FC2 腿的同一个槽位，与 FlyDSL bitcode tile 同槽对照。**这是"融合上下文内的 tile 对决"，是目前谁都没有的测量**，也是 §7.2 第三条研究价值的具体形态。

**S2 —— grouped driver。** 无论 S1 结果如何，HK 要真正服务 MoE 都得有变长-M driver。这一项独立于 S0/S1，且是 HK 相对 FlyDSL `gemm_helper.py` 的唯一硬缺口。

## 相关文件

- `3rd/ROCMega/doc/fwd_vs_flydsl.md` —— **本篇最重要的来源**：同环境逐腿对照 + root cause
- `3rd/ROCMega/doc/fwd_perf.md` —— 环境、正确性（max_rel_err 3.72e-3）、T=2k/8k × balanced/random
- `3rd/ROCMega/doc/notes/2026-07-14.md` —— 复测 + 反向 20.84 ms + **测量纪律教训**（冷时钟要先跑一次丢弃、balanced vs random、timer 会拉长 wall）
- `3rd/ROCMega/csrc/gemm/gemm_tiles.h` —— bitcode tile 的 ABI 与 LDS ODR 折叠
- [立项文档](./2026-08-12_1530_repo-charter-and-primitive-api.md) · [HK GEMM 实测](./2026-08-12_1630_hk_gemm_on_dsv3_moe_shapes.md)
