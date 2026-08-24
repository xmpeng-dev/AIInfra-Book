# Mega MoE MXFP8 前向 — 阶段性性能 & 正确性 @ MI355X（n03-33）

> **When**: 2026-07-20 15:20 UTC+8
> **Where**: `smci355-ccs-aus-n03-33`（MI355X / gfx950 ×8），容器 `xiaoming-dev`（`rocm/primus:v26.3`）
> **Context**: 把 mega MoE 的 fp8(MXFP8) 版本从 Primus-Turbo 移植进 `/perf_apps/xiaoming/MegaMoE`；本 note 记录**前向**(L1 dispatch+fc1 → SwiGLU → L2 fc2+combine) 接通后的阶段性性能与正确性。反向尚未移植。
> **Repo**: `/perf_apps/xiaoming/MegaMoE` @ `66dacc6`（feat(moe): wire fp8 mega MoE forward），branch `feat/xiaompen/mega_moe_flydsl_mxfp8`
> **软件**: torch `2.10.0+git94c6e04`，**flydsl `0.2.4`**（镜像自带 `0.1.1.dev409` egg 需先升级),triton `3.6.0`
> **配置**: DeepSeek-V3，EP8，H=7168，I=2048，E=256，topk=8。参考基线 = 本仓库 bf16 `mega_moe_fused`。

## TL;DR

- **fp8 前向端到端跑通、正确**：8 rank 全 finite,fp8 vs bf16 **SNR 20.8–22.3 dB / cos 0.996–0.997**（和源 repo 报的 ~23 dB 一致;fp8 用 SNR gate 而非 allclose）。
- **DSv3(T=8192) 前向 fp8 = 5.22 ms,比 bf16 7.06 ms 快 1.35×**。分段:L1 ~1.6×、fc2(L2) 1.39×。
- **关键修复**:signal pad 改 `hipMallocUncached` —— 修掉 fc2 fp8-combine 的**跨 rank NaN 竞争**(cached L2 读到过期 E8M0 → +inf)。
- **小 T(2048) 前向只 ~1.0–1.08×**:不是 fc2 退化,是固定 host 开销(4 次 sync+barrier、prologue、量化、launch)在小 compute 下占比大。

## Background

- 之前(2026-07-17)已验证 bf16 两融合算子 EP8 正确 + 重叠有效(见 `2026-07-17_1411_..._srcbuild_mi355x.md`)。
- 本轮把 fp8 路径 vendored 进自包含的 `flydsl/mega/fp8/` 子包(SymLayout + scoreboard + 双堆),不动 bf16。L1(dispatch+fc1)上一轮已验证 cos=1.0;本轮补齐 **L2(fc2 mxfp8 GEMM + fp8 combine PUSH + bf16-out dequant reduce)** + SwiGLU,并在 `mega_moe_fused_fp8` 里接通完整前向。

## What I did（本轮改动)

- **L2 + SwiGLU 移植** 进 `flydsl/mega/fp8/`,接通 `mega_moe_fused_fp8` 前向(L1→SwiGLU→L2)。
- **根因修复**:`primus_turbo/pytorch/core/symm_mem.py` 的 signal pad 由 cached `hipMalloc` 改 `hipMallocUncached`。fp8 combine 把 fp8 payload+E8M0 PUSH 进 peer signal 堆的 combine buffer,cached 下留在过期 L2 line → reduce 读到垃圾 E8M0 指数(+inf) → 少数 rank 随机 NaN。bf16(单堆 `signal_pad_size=0`)不受影响。
- **纯化**:`fp8/` 下不带 bf16 GEMM/combine kernel(删 `PT_FP8_COMBINE_GEMM=bf16` 回退,共享线程常量抽到 `combine_config`);`grouped_gemm_combine_fp8` 变纯计算 kernel(收 pre-quantized w2)。
- **权重量化统一在 op 层版本缓存维护**(w1/w2 各按 `_version`),计算 kernel 不做量化判断;`token/act` 量化在 kernel 内(per-call)。
- 量化算法:OCP MXFP8(E4M3 元素 + 每 1×32 块 E8M0 scale),fc2 输出量化融合在 GEMM 的 CShuffle epilogue(`StoreCQuantMxfp8CShuffle`)里 → combine 传输天然 fp8、零额外量化 kernel。

复现:

```bash
cd /perf_apps/xiaoming/MegaMoE
# e2e 前向 fp8 vs bf16(SNR gate)
MASTER_PORT=8xxx MEGA_BENCH_TIMEOUT_S=300 PYTHONPATH=$PWD \
  python benchmark/ops/bench_mega_moe_fused_fp8.py --num-processes 8 --num-tokens 8192
# 隔离 L1(dispatch+fc1),含 token quant,vs torch dequant-GEMM 参考
PYTHONPATH=$PWD python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_l1.py --num-processes 8 --num-tokens 8192
# 隔离 fc2(L2 combine)fp8 延迟
PYTHONPATH=$PWD python benchmark/ops/bench_grouped_gemm_combine_fp8_l2.py --num-processes 8 --num-tokens 8192
```

## Result

取跨 8 rank 最慢值。fp8 用 SNR(dB)= `10·log10(‖ref‖²/‖ref−out‖²)`,参考为 bf16。

### 完整前向(fp8 vs bf16 `mega_moe_fused`)

| T | fp8 fwd (ms) | bf16 fwd (ms) | 加速 | 精度(fp8 vs bf16) |
|---|---|---|---|---|
| 2048 | 2.38–2.42 | 2.42–2.58 | ~1.0–1.08× | SNR 20.8–21.2 dB, cos 0.996 |
| **8192 (DSv3)** | **5.22** | **7.06** | **1.35×** | SNR 22.31 dB, cos 0.997 |

### 分段(DSv3 T=8192)

| 段 | fp8 | bf16 | 加速 | 备注 |
|---|---|---|---|---|
| L1 dispatch+fc1(含 token quant) | 2.35 ms @ 1737 TFLOPS | ~3.8 ms | ~1.6× | 隔离 bench cos=1.0 vs torch 参考;token_quant 0.059 ms(~2.5%) |
| fc2 L2 combine(整条) | 2.16 ms | 3.01 ms | **1.39×** | 隔离 bench,SNR 25.9 dB,finite |
| — fc2 纯 GEMM(down-proj) | ~1.12 ms @ 1828 TFLOPS | — | **~2×**(硬件/源实测) | f8f6f4 MFMA ≈ bf16 峰值 2× |
| 整条前向 | 5.22 ms | 7.06 ms | 1.35× | = L1 + fc2 + swiglu |

### L1 隔离(token-quant-inside,vs torch dequant-GEMM 参考,cos=1.0)

| T | token_quant (ms) | fused kernel (ms) | L1 total (ms / TFLOPS) |
|---|---|---|---|
| 2048 | 0.048 | 0.894 | 0.942 / 1245 |
| 8192 | 0.059 | 2.294 | 2.353 / 1737 |

### 训练 bench `--stage fwd`（`bench_mega_moe_fp8.py`，同一 `_bench` 方法学，2026-07-21 补测）

把完整前向做成 training bench 的一个 stage：fp8 `mega_moe_fused_fp8` vs 现成 bf16 `mega_moe_fused`，同输入，持久 W1/W2（fp8 op 的版本化权重量化缓存命中，贴训练）。每次调用都是整条 per-forward（各自 prologue/dispatch/combine/reset），CUDA-event 计时、每次 `sync+barrier` 串行化。`load_balanced`（真实路由）干净：

| T | fp8 fwd (ms) | bf16 fwd (ms) | 加速 (bf16/fp8) | SNR(fp8 vs bf16) |
|---|---|---|---|---|
| 2048 | 2.202 | 2.450 | 1.113× | 20.55 dB PASS |
| **8192 (DSv3)** | **5.246** | **7.202** | **1.373×** | 22.28 dB PASS |

和上面 e2e bench 一致（8192 ~1.35–1.37×、2048 ~1.1×）。**`round_robin` 会撞上 fp8 L2 combine 的跨 rank 竞争(NaN/GPU fault)**——完整前向含 L2,故 fwd stage 只在 `--mode load_balanced` 下干净;round_robin 待 L2 竞争修复后再放开。复现:

```bash
PYTHONPATH=$PWD python3 benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fwd --mode load_balanced
```

### 前向 breakdown — 为什么整体只 ~1.37×（2026-07-22 干净环境实测，load_balanced T=8192, M_pool=69632）

| 段 | fp8 | bf16 | 加速 | fp8 TFLOPS |
|---|---|---|---|---|
| **L1**（dispatch PUSH + fc1 GEMM，含 token quant） | 2.624 ms | 4.295 ms | **1.64×** | 1558 |
| **L2**（fc2 GEMM + combine PUSH + reduce，kernel-only） | 2.140 ms | 2.899 ms | **1.36×** | 955 |
| **完整 fwd** | 5.218 ms | 7.143 ms | **1.37×** | — |

L2 内部拆解 —— **实测**（给 fp8 combine kernel 加 `PT_COMBINE_GEMM_ONLY` / `PT_COMBINE_PUSH_ONLY` 隔离，照 backward kernel 已有的；load_balanced T=8192）：

| 隔离 | 时间 | 含义 |
|---|---|---|
| GEMM_ONLY（只 fc2 GEMM，**含 CShuffle mxfp8-quant epilogue**） | **1.599 ms** | GEMM 腿 |
| PUSH_ONLY（只 combine PUSH，GEMM idle） | **1.568 ms** | PUSH 腿 |
| NO_REDUCE（GEMM+PUSH overlap，无 reduce） | **2.056 ms** | 两腿重叠 |
| full L2（+reduce） | 2.14 ms | reduce ≈ 0.085 ms（可忽略） |

⚠️ **纠正之前引用的"纯 fc2 GEMM ~1.12ms"**：那是纯 grouped GEMM 的旧数；combine kernel 里的 fc2 GEMM 带 **mxfp8-quant epilogue**（量化 C→fp8 L2Y），实测 **1.60ms**，比纯 GEMM 重。

**机制（实测校准）**：L1/L2 都是**单 kernel launch**，combine block（`block_index < num_combine_cu=48`）和 GEMM block（`>= gemm_base`）在**同一 grid 并发**、共享 CU；GEMM 每算完一 tile `atomic_add(sb_l2)`，combine block `spin sb_l2>=n_blocks` 就 push —— **producer-consumer overlap**。overlap **确实有效**：串行 1.60+1.57=3.17ms，实测 overlap 到 2.056ms（省 ~1.1ms）；但**不完美**（理想 max=1.60ms，实测 2.056ms → 效率 ~71%），因 combine 与 GEMM 共享 CU 竞争。

**根因（为什么 L2 只 1.36×，进而 fwd 只 1.37×）**：L2 是**两条几乎等长的腿（GEMM 1.60 / PUSH 1.57）overlap（71% 效率）**的系统 —— 不是谁藏谁：
1. fp8 缩短 GEMM 腿，但被 **mxfp8-quant epilogue 稀释**（没到 2×）。
2. fp8 缩短 PUSH 腿，但被**跨 rank 通信固定开销稀释**（没到 2×）。
3. **overlap 不完美**（CU 竞争，71%）。
三者叠加 → L2 1.36×。对比 **L1 是 GEMM-bound（1.64×）**：L1 的 dispatch PUSH 短于 fc1 GEMM，GEMM 主导 → fp8 GEMM 2× 生效更多。
→ 可挖方向：**同时缩两条腿**（GEMM 的 quant epilogue 优化、PUSH 降精度/字节）+ **提 overlap 效率**（调 `num_combine_cu` 平衡 GEMM/PUSH 的 CU 占用），而不是只盯某一条。SwiGLU 走 bf16（不加速）也稀释一点。

**`num_combine_cu` sweep（load_balanced L2 @ 8192，`PT_COMBINE_CU`）**：≤24 **HANG**（combine CU 不足，PUSH 跟不上 GEMM，scoreboard 堆积自旋）；28=2.087 / **32=2.072（甜点）** / 48=2.121（默认）/ 64=2.178 / 80=2.323 / 96=2.352。单调：**combine CU 越少越快**（GEMM 腿 1.60 略长于 PUSH 腿 1.57，多给 GEMM CU 更优），但 <28 hang。→ 甜点 32 比默认 48 **只快 ~2.3%（0.05ms）**，传导到 fwd 加速 1.37→~1.38×，**几乎不动**。结论：`num_combine_cu` 不是大优化点（两腿等长，可调窗口窄 28–32）；真正要缩的是 GEMM 的 mxfp8-quant epilogue（纯 GEMM ~1.1 → 含 epilogue 1.6）和 PUSH 精度/字节。

## Interpretation

1. **fc2 的 ~2× 纯 GEMM 被 combine 稀释成整条 fc2 的 1.39×**:L2 的 combine PUSH + top-k reduce 是带宽 bound(输出/reduce 走 bf16 量级),GEMM 只占 L2 一部分。这与源设计笔记一致(combine 才是瓶颈)。**注意**:源旧笔记曾说"fp8 L2 拿不到收益/更慢"(那是旧的逐-block `l2_writeback` 版本 4.68ms);vendored 的最终版(write-through + mxfp8 GEMM)实测 fp8 L2 = 2.16ms < bf16 3.01ms,**已经赢**,以实测为准。
2. **小 T 前向 ~1.0×**:前向为正确性放了 4 次 `torch.cuda.synchronize()+group.barrier()`(L1/L2 scoreboard/flag 复位),加 prologue/量化/launch 的固定成本;T=2048 时 compute 太小、固定开销占比大,把 fp8 的算力优势拉平。大 T(真实训练 shape)下 1.35×。→ 后续可试源 op 的 `PT_MEGA_BARRIER_MODE=reduced`(只留 2 个 load-bearing barrier)降小 T 固定开销。
3. **精度 ~20–22 dB**:两级 fp8 GEMM(fc1+fc2)+ SwiGLU 量化噪声累积,和源 e2e ~23 dB 一致;fp8 判定用 SNR,不用 allclose。

## 注意 / 坑

- **e2e 对比 bench 的 bf16 参考腿会偶发自旋**:`bench_mega_moe_fused_fp8.py` 在**同一进程**里跑 bf16 `mega_moe_fused` + fp8(两套对称内存 + 首次 JIT),bf16 腿曾出现 8 卡 100% 但不推进(跨 rank flag 等不到)的偶发死锁。**fp8 一侧已单独验证正确**(`--only fp8` + 隔离 L1/L2 bench),bf16 腿的 SNR 数是能跑通时取到的。跑挂用 `kill -9 $(ps -eo pid,cmd | grep '[s]pawn_main' ...)` 清残留 worker、换 `MASTER_PORT`。
- 新起的容器要先 `bash slab/notes/MegaMoeFlydsl/_repro/fix_flydsl.sh` 升 flydsl 到 0.2.4。

## Next

1. **移植反向**(HANDOFF 里的下一步):STEP1 dispatch(dy)+fc2 dgrad(fp8,会赢)→ dW2/dW1 mxfp8 variable-K → STEP3 fc1 dgrad(fp8)+combine。前向目前**未 save_for_backward**,接反向时补 ctx 保存。
2. 可选:前向 barrier 收敛(`reduced` 模式)提升小 T 加速;e2e bench 的 bf16 参考腿改成独立进程,避免同进程两套 symm 的偶发死锁。
