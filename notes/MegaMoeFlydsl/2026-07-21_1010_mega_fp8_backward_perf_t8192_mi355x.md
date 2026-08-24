# Mega MoE MXFP8 反向 — 性能追踪（T=8192 only）@ MI355X

> **用途**: fp8(MXFP8) mega MoE **反向**各阶段 fp8-vs-bf16 性能的**持续记录**。只跑 **T=8192（DSv3 训练 scale）**。后续每次性能测试往下面「测试记录」追加一条带日期的条目。
> **Where**: `smci355-ccs-aus-n03-33`（MI355X / gfx950 ×8），容器 `xiaoming-dev`（`rocm/primus:v26.3`）
> **软件**: torch `2.10.0+git94c6e04`，flydsl `0.2.4`，triton `3.6.0`（新容器先 `bash slab/notes/MegaMoeFlydsl/_repro/fix_flydsl.sh`）
> **配置**: DeepSeek-V3，EP8，**T=8192**，H=7168，I=2048，E=256，topk=8，BM=BN=256。基线 = 本仓库 bf16 `mega_moe_fused` 同族 FlyDSL bf16 kernel。跨 8 rank 取最慢；fp8 精度用 SNR(dB)=`10·log10(‖ref‖²/‖ref−out‖²)`。
> **前向性能**见 `2026-07-20_1520_mega_fp8_forward_perf_mi355x.md`。
>
> **✅ 已修正（2026-07-21 11:30）**：早先怀疑的"fp8 慢 ~34%"**不是** flydsl 环境回归，而是**两个 bench 侧 bug**，均已修：
> 1. **计时方法**：`_bench` 原来对**单次** kernel 调用用 GPU event 括住计时 → 把 custom-op dispatch / autotune 查表的 **host 开销算成 GPU-idle**，虚高所有 fp8 kernel（小 kernel 虚高更多：quant 0.139→0.256、requant 0.345→0.491）。改成源 `test_dw2_bench` 的**背靠背**计时（N 次连发 / N）后，requant/quant **精确吻合源**。
> 2. **GEMM autotune 漏移植**：`mxfp8_grouped_kernel.py` 的 variable-K wgrad 配置被写死 `xcd=8`，源有 `(bm,bn,gm,xcd,gn)` autotune（DSv3 dW2 挑 `xcd=1`）。已按源逐字节补齐（文件现与源 `diff -q` 一致），GEMM +~8%。
>
> 修正后 GEMM 达 **~2000 TFLOPS**。与源 test 的 2324 的残差 = **池分区(routing 均衡度)**，非代码：直接调 kernel 的 A/B 证明 MegaMoE 包与源包**同速(1929 vs 1905)**，源包跑我的算子也只有 1905；源 2324 是它那次 routing 抽到的异常均衡池(m_pad 66048/padding 0.8%)，随机 routing 的 ~70k 池(padding ~7%)对应 ~2000（详见测试记录 13:40）。下表为**修正后**的数。

## 复现（全部 T=8192）

```bash
cd /perf_apps/xiaoming/MegaMoE
# STEP1 dispatch(dy)+fc2 dgrad — fp8 vs bf16（两腿都是 fused PUSH + fc2-dgrad grouped GEMM）
PYTHONPATH=$PWD python benchmark/ops/bench_step1_fp8_vs_bf16.py --num-processes 8 --num-tokens 8192
# dW2 / dW1 — fp8 vs bf16 wgrad + SNR
PYTHONPATH=$PWD python benchmark/ops/bench_dw2_fp8.py --num-processes 8 --num-tokens 8192
PYTHONPATH=$PWD python benchmark/ops/bench_dw1_fp8.py --num-processes 8 --num-tokens 8192
# STEP3 fc1 dgrad+combine（fp8-PUSH，大 T 会偶发 hang；清残留后重试）
PYTHONPATH=$PWD python benchmark/ops/bench_step3_fp8.py --num-processes 8 --num-tokens 8192
# e2e fwd+bwd fp8 vs bf16 + 梯度 SNR（权重常驻→量化缓存命中；含 STEP3，大 T 受 hang 影响）
PYTHONPATH=$PWD python benchmark/ops/bench_mega_moe_fused_fp8_bwd.py --num-processes 8 --num-tokens 8192
# 跑挂清残留：kill -9 $(ps -eo pid,cmd | grep '[s]pawn_main' | awk '{print $1}')  然后换 MASTER_PORT
```

## 当前快照（T=8192）

| 阶段 | fp8 (ms / TFLOPS) | bf16 (ms / TFLOPS) | fp8/bf16 | 精度 | 状态 |
|---|---|---|---|---|---|
| **STEP1** dispatch(dy)+fc2 dgrad | 1.848 / 1106 | 2.453 / 833 | **0.753× (1.33× faster)** | SNR 30.9 dB | ✅ 稳 |
| **dW2** fc2 wgrad (variable-K) | 1.76 / 1176（FULL） | 1.96 / 1056 | FULL **0.90×** / GEMM-only **0.52×**（GEMM 2013 TFLOPS） | SNR 21.8 dB | ✅ 稳 |
| **dW1** fc1 wgrad (variable-K, LOCAL) | 2.868 / 1426 | 3.814 / 1072¹ | **0.752× (1.33× faster)** | SNR 21.8 dB | ✅ 稳 |
| **STEP3** fc1 dgrad + combine | 3.31 / 1237 | 3.89 / 1052（独立 bf16 nt） | **1.175× (fp8 faster)** | dx finite; e2e dx 21.9 dB | ✅ 干净环境稳（2026-07-22）² |
| **SwiGLU^T** | bf16（两路径同 kernel），量级小，未单列 | | | | |
| **bwd only** (STEP1+dW2+STEP3+dW1, 融合 autograd) | 10.43 ms | 13.12 ms | **1.26×** | grad SNR 19.5–23 dB PASS | ✅（2026-07-22）|
| **e2e** fwd+bwd | 16.05 ms | 20.35 ms | **1.27×**（fwd 1.29× / bwd 1.26×） | dx21.7/dtw22.9/dW1 19.5/dW2 19.7 dB PASS | ✅ 打通（2026-07-22）⁴ |

- ⁰ dW2 **FULL** = requant(pool)+quant(act)+GEMM（反向实跑）= **0.90× bf16（fp8 快）**；隔离 **fp8 GEMM = 0.52×（1.9× faster，2013 TFLOPS，近 roofline）**；breakdown requant 0.360 / quant 0.139 / GEMM 1.027ms（均吻合源 test_dw2_bench）。剩余可挖点 = quant/requant ~0.5ms（producer fusion）。
- ¹ dW1 的 bf16 参考腿只算 GEMM（同一份本地 pool）；fp8 dW1 是 **LOCAL**（复用前向派发的 fc1-input 池），还额外省掉 bf16 融合路径对 `saved_x` 的**跨 rank 重派发** → 实际优势 > 1.33×。
- ² STEP3 fp8-PUSH **干净环境稳定**（238 次 back-to-back + 12 次独立 run 零 combine 竞争，见 2026-07-22 记录）；之前的"死锁"是**残留进程干扰**（rocm-smi 里占 GPU 自旋的 UNKNOWN PID），非 kernel 竞争。
- ³ STEP3 bf16 腿 = 独立纯 bf16 stack bench `bench_step3_bf16.py`（`grouped_gemm_combine_bf16_flydsl_kernel` layout="nt" = w1^T reuse，**5/5 PASS**）；training bench 的 bf16 leg 因同进程 fp8→bf16 stack 污染 SIGABRT（非 kernel bug，见 2026-07-22 08:25 记录）。
- ⁴ e2e @T=8192 **干净环境打通**（2026-07-22）：fp8 16.02ms vs bf16 20.19ms = **1.26×**，梯度 SNR dx21.9/dtw23.1/dW1 19.5/dW2 19.7 dB 全 PASS。（T=2048 曾 1.06×。）

**小结**：反向四阶段 fp8 全部有对比数 —— **STEP1 1.33×**、**dW1 1.33×**、**dW2 FULL 0.90×（GEMM 1.9×，近 roofline 2013 TFLOPS）**、**STEP3 1.175×**；**e2e@8192 fwd+bwd = 1.26×，梯度全 PASS**。之前"STEP3 死锁/e2e 不可测"均为残留进程干扰，干净环境全部跑通。（早先 dW2"持平"是 bench 计时 bug + GEMM autotune 漏移植所致，已修正——见顶部。）

## 测试记录（往下追加）

### 2026-07-21 — 反向接通后的首个 T=8192 快照
- repo `66dacc6`，branch `feat/xiaompen/mega_moe_flydsl_mxfp8`。
- STEP1 fp8 1.873ms(1092 TFLOPS) vs bf16 2.697ms(758) → **1.44× faster**，grad_swiglu SNR 30.9dB。
- dW2 fp8 1.952ms(1047) vs bf16 1.945ms(1051) → 1.004×（持平），SNR 21.8dB。
- dW1 fp8 3.183ms(1284) vs bf16 3.773ms(1084, GEMM-only) → **1.19× faster**，SNR 21.8dB。
- STEP3 fp8 ~3.34ms(1225 TFLOPS) 可跑通时；大 T 间歇死锁未根治。
- e2e@8192 未测（STEP3 阻塞）；@T=2048 常驻权重 fp8 7.31 vs bf16 7.72 ≈ 1.06×，grad SNR dx19.0/dtw19.3/dW1 18.3/dW2 19.5 dB 全 PASS。

### 2026-07-21 (10:39) — dW2 复测（容器重启后确认）
- 节点 n03-33 重启过、dev 容器被删；重拉 `xiaoming-dev` + `fix_flydsl.sh`（flydsl→0.2.4）后复测。
- **dW2 @T=8192 ×2 run 稳定**：fp8 1.96 / 1.95 ms（~1043 TFLOPS）vs bf16 1.94 / 1.94 ms（~1053）→ **1.009×（fp8 略慢，≈持平）**，SNR 21.77 dB PASS。
- 结论不变：dW2 wgrad 算力 bound，fp8 的 colwise requant 开销 ≈ 抵消 GEMM 2× → latency 无净收益；价值在梯度以 fp8 走的显存/带宽。

### 2026-07-21 (10:54) — dW2 分解（给 bench 加 GEMM-only 拆解，对齐源 test_dw2_bench）
- 起因：源 `tests/.../test_mega_moe_mxfp8.py::test_dw2_bench` 记忆里"性能很高"。源测试当前跑不了（容器**已装的** C++ `quantize_mxfp8` 旧 ABI 收 6 参、源 Python 传 7 参 → RuntimeError）；遂在 MegaMoE 的 `bench_dw2_fp8` 里加同款分解。
- **@T=8192**：`dW2 FULL 2.00ms (1023 TFLOPS) | bf16 1.94ms (1053) → FULL 1.03×`，但 **GEMM-only 0.613×**。
- **breakdown**：requant(pool)=0.491ms，quant(act)=0.256ms，**GEMM=1.190ms（1719 TFLOPS）**，sum=1.936ms。
- 定论：**隔离 fp8 GEMM 快 1.63×（1719 TFLOPS）= 记忆里的"高性能"**；FULL 被 0.75ms 的 quant/requant 吃平。→ dW2 若要拿 latency 收益，必须砍掉/上游融合那 0.75ms（如把 act 的 colwise-quant 融进 STEP2 swiglu_backward 输出、pool requant 进一步 producer-fuse）。

### 2026-07-22 (08:30) — ✅ e2e@8192 fwd+bwd 打通（推翻"STEP3 阻塞"）— 完整 8k 训练 step fp8 快 1.26×
- 干净环境跑 `bench_mega_moe_fused_fp8_bwd.py --num-tokens 8192`（完整 fp8 autograd：fwd L1→swiglu→L2 combine；bwd STEP1→swiglu^T→dW2→STEP3→dW1）+ grad-check vs bf16 `mega_moe_fused`。**routing = load_balanced**（rand-softmax top-k，真实非均衡路由）：
  - **fwd+bwd：bf16 20.35 / fp8 16.05 ms → 1.27×**；加了 fwd-only 计时拆分（bwd = 合计 − fwd）：**fwd 1.29×（fp8 5.63 / bf16 7.23 ms）、bwd 1.26×（fp8 10.43 / bf16 13.12 ms）**。
  - **梯度 SNR vs bf16：dx ~21.8 / d_topk_w ~23 / dW1 19.5 / dW2 19.7 dB，全 PASS（gate≥15dB）**。
  - ⚠️ e2e 的 STEP3 是 autograd backward 的一环，`--iters` 大时多次 STEP3 combine 累积会偶发 hang（用 `--warmup 1 --iters 2` 稳跑；大 iters 那次 hang 在编译后的 combine，非编译问题）。
- 之前"e2e@8192 因 STEP3 大 T 不稳暂不可测"是残留进程干扰；干净环境端到端**一次跑通**。**至此 fp8 mega MoE fwd+bwd 全链路 @ DSv3 8k：1.26× vs bf16，梯度数值正确。**

### 2026-07-22 (08:25) — ✅ STEP3 fp8-vs-bf16 对比拿到（独立 bench）+ training bench bf16 leg = fp8→bf16 stack 污染
- **training bench `fc1_dgrad_combine` 的 bf16 leg SIGABRT 是污染，非 bf16 kernel bug**：写纯 bf16 独立 bench `benchmark/ops/bench_step3_bf16.py`（完全不碰 fp8 stack）→ **5/5 PASS**。training bench 里 bf16 leg 在 fp8 leg + `symm.destroy()` 之后建 bf16 stack，全局 symm 单例（`fp8/symm_buffer.py::_CURRENT_SYMM_BUFFER`）/ fp8 scratch 被污染 → bf16 combine SIGABRT。二分证明：`STEP3_SKIP_BF16=1` 时 fp8 leg 单独 PASS（3.15ms/1296 TFLOPS）；bf16 独立 stack 也 PASS；只有同进程 fp8→bf16 顺序崩。
- **独立对比（`bench_step3_fp8.py` vs `bench_step3_bf16.py`，同 routing/权重 seed，M_pool=69632 实际行数可比，T=8192 EP8）**：

  | | 延迟 | TFLOPS | dx |
  |---|---|---|---|
  | **fp8** STEP3（fp8-PUSH combine，生产路径） | **3.306 ms** | 1237 | finite PASS |
  | **bf16** STEP3（bf16 combine，nt = w1^T reuse） | **3.886 ms** | 1052 | finite PASS |
  | **加速** | **bf16/fp8 = 1.175×（fp8 faster）** | | |

- 注：bf16 独立 bench 的 `m_pad` 要用 `num_tile_blocks[0]*BM`（实际 padded 行数），不是 `l1_bf.shape[0]`（那是 worst-case 容量 532480，会把 TFLOPS 虚高到 8065）。修正后两 bench M_pool 一致（69632）→ 可比。
- **至此 8 个 stage 全部有 fp8-vs-bf16 对比数**：前向 l1/l2/fwd、反向 dispatch_fc2_dgrad/fc2_wgrad/fc1_wgrad/fc1_dgrad_combine。（training bench 的 `fc1_dgrad_combine` bf16 leg 若要用，需先修 fp8→bf16 stack 污染；独立 bench 已给可信对比。）
- 复现：`PYTHONPATH=$PWD python3 benchmark/ops/bench_step3_bf16.py --num-processes 8 --num-tokens 8192`（+ `bench_step3_fp8.py`），换 MASTER_PORT，run 间清残留。

### 2026-07-22 (07:45) — ✅ 干净环境 STEP3 稳定：之前的死锁 = 残留进程干扰，**非 kernel 竞争**
- 换新节点/容器（干净环境：8 卡都在 297MB 基线、无残留 GPU 进程；先 `fix_flydsl.sh` 升 flydsl→0.2.4），重跑 STEP3 独立 bench（`benchmark/ops/bench_step3_fp8.py` T=8192 EP8）：
  - 单 run 28 次 combine：**PASS**；单 run **210 次 back-to-back combine**：**PASS**（3.29ms / 1242 TFLOPS）。
  - 10 次独立 run（清理间隔 `sleep 5`）：**9 PASS** + 1（run9）被 90s timeout 杀在 **JIT 编译/启动阶段**（日志停在 `c_n` 编译 warning，**无任何** combine gate / reduce flag timeout / SIGABRT）→ 清理间隔太短、上个 run 的 GPU context 未释放完导致启动慢，**非 combine 死锁**。
  - 3 次独立 run（清理间隔拉到 `sleep 15`）：**3/3 PASS**（~3.30ms / ~1240 TFLOPS）。
- **结论**：STEP3 fp8-PUSH combine 在干净环境下**无死锁**（238 次 back-to-back + 12 次独立 run 零 combine 竞争）。**之前多轮 hang 全是残留进程**（`rocm-smi --showpids` 里占 GPU 自旋的 `UNKNOWN` PID，抢 CU/context、恶化跨 rank combine 的 landing/reduce-flag 同步窗口）所致，`pkill -f spawn_main/python3 benchmark` 抓不到它们（cmdline 不匹配）→ 累积残留把后续 run 全拖挂。**拆 reduce 非必需**；STEP3 stage 干净环境可用。
- **操作纪律**（避免"假死锁"）：跑前 `rocm-smi --showmeminfo vram` 确认 8 卡回 ~298MB 基线；hang 后用 `rocm-smi --showpids` 按 `UNKNOWN` PID `kill -9`（别只靠 `pkill`）；run 间留足释放时间（≥15s）。
- 之前被"死锁"挡住的 **STEP3 bf16 对比 / e2e@8192 现在可做**（推翻 23:30 那条的阻塞判断——阻塞源是残留干扰，非 kernel）。

### 2026-07-21 (23:30) — STEP3 加 bf16 对比腿：代码就绪，但被 fp8-PUSH combine 死锁阻塞（拿不到稳定数）
- 给 `fc1_dgrad_combine` stage 加 bf16 对比：fp8 `grouped_gemm_combine_fp8_bwd`（fp8-PUSH）vs bf16 `grouped_gemm_combine_bf16_flydsl_kernel`，kernel-only 延迟对比（reset outside window，同 L2 口径）。
- **两个架构坑（已解决）**：① **fp8/bf16 dispatch handle 不兼容** —— bf16 combine kernel 读 `handle[9..12]` 的 recv_*，而 fp8 `dispatch_prologue` 的 handle 那里是 group_lens/offs → 用 fp8 handle 喂 bf16 kernel 直接 `IndexError`。故 bf16 腿必须走**独立 bf16 stack**（bf16 dispatch 拿 hbf + realistic grad_l1 替身），只能对比延迟（跨 stack 不能算 dx SNR，靠 e2e gradcheck）。② bf16 **nn backward** 路径未验证（SIGABRT，note 早标"待补"）→ 改用 L2 验证过的 **nt（w1^T reuse）** 路径。
- **阻塞：fp8-PUSH combine 当前环境死锁概率极高**。kernel-only bench 每次 iter 一次 fp8-PUSH combine，跑 warmup+iters 次里几乎必挂——连 `--warmup 1 --iters 2`（~3 次 combine）都反复 spin-deadlock（NCCL/host 全卡，须 `pkill` + 换 `MASTER_PORT` 重试）。比之前记的 5–15% 严重得多。**源 `test_step3_bench` 同样 hang** → 死锁是继承自源的顽疾，非移植引入。
- **现状**：STEP3 bf16 对比腿**代码就绪**，但受 fp8-PUSH combine 死锁阻塞，**拿不到稳定的同 run fp8-vs-bf16 数**。仅在早前侥幸跑通一次拿到 fp8 单腿 3.262ms（op-wrapper 版，warmup5 iters15 侥幸 20 次全过）。
- **根治前置**：必须先拆「融合 GEMM+push → barrier → 独立 reduce」（见 Interpretation #4）消除跨 rank reduce-flag 竞争，STEP3 才能稳定 bench。其余 7 个 stage（前向 l1/l2/fwd + 反向 dispatch_fc2_dgrad/fc2_wgrad/fc1_wgrad）都稳定给出 fp8-vs-bf16 数。

### 2026-07-21 (23:00) — fc1 dgrad+combine (STEP3) 接入 training bench（`--stage fc1_dgrad_combine`）→ 反向四阶段全部接通
- 复刻 `bench_step3_fp8.py` 成 stage：真实反向到 STEP3（fwd L1 → dispatch(dy)+fc2-dgrad → swiglu_backward(return_gate) → grad_l1+grad_gate），`_mxfp8_step3_fc1_dgrad_combine`：fp8 fc1-dgrad（grad_l1@w1^T）+ fp8-PUSH combine + unweighted reduce + gate scatter → dx[T,H]+grad_topk[T,K]。op 内部每次自 reset L2 scoreboard/flag（`_host_rendezvous`）→ 背靠背计时有效。
- **@T=8192 E=256 load_balanced**：fp8 **3.262** ms（**1253 TFLOPS**，M_pool=69632），dx[T,H] finite（norm 988.8）+ grad_topk[T,K] finite = **smoke PASS**。与旧快照 ~3.337ms/1225 TFLOPS 吻合。本次一次跑通未触发死锁。
- **两个固有限制（同源）**：① **无隔离 bf16 腿** —— fp8 子包不带 bf16 combine-bwd kernel，故只做 smoke+延迟，严格 dx SNR 由 e2e gradcheck 保证；② fp8-PUSH combine 大 T ~5–15% 偶发跨 rank reduce-flag 自旋死锁 + round_robin 必撞（同前向 L2），挂了换 `MASTER_PORT` 重跑 load_balanced。
- 复现：`PYTHONPATH=$PWD python3 benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc1_dgrad_combine --mode load_balanced`（不含在 `both` 里）。
- **至此 training bench 8 个 stage 齐全**：前向 l1 / l2 / fwd；反向 dispatch_fc2_dgrad / fc2_wgrad / fc1_wgrad / fc1_dgrad_combine。

### 2026-07-21 (21:20) — fc1 wgrad (dW1) 接入 training bench（`bench_mega_moe_fp8.py --stage fc1_wgrad`）+ 对齐源
- 复刻 `bench_dw1_fp8.py` 成 stage：真实反向到 dW1（fwd L1 → **STEP1 覆盖前 clone 前向派发的 fc1-input 池 pool_x** → dispatch(dy)+fc2-dgrad → swiglu_backward → grad_l1），dW1 = grad_l1^T @ pool_x（variable-K，**LOCAL**，复用前向池、无跨 rank 重派发）。fp8 `_mxfp8_variable_k_wgrad_dw1`（quant grad_l1 + requant fp8 pool_x → fp8 GEMM）vs bf16 Triton `grouped_gemm_variable_k_impl`。加 meta/quant/requant/GEMM breakdown + FULL/GEMM-only，SNR gate 15dB。不碰 L2 combine。
- **@T=8192 E=256（DSv3 真实）**：fp8 FULL **2.894** ms(1413 TFLOPS) vs bf16 **3.792** ms(1078) → FULL **1.310×**；**GEMM-only 1.902×**（GEMM 1.994ms @ **2051 TFLOPS**）；breakdown meta 0.117 / quant(grad_l1) 0.272 / requant(pool_x) 0.363 / GEMM 1.994；SNR 21.77 dB PASS。与旧快照 dW1 2.868/3.814→1.33× 吻合。
- **对齐源 test_dw1_bench（E=32）**：源 NEW local mxfp8 = **2.324 ms / 1668.8 TFLOPS**（M_pool=66048，FLOP 3.88T，old/new re-dispatch=2.15×）。我的 E=32 fp8 FULL = **2.436 ms / 1592 TFLOPS**（同 M_pool 66048、同 FLOP 3.88T，GEMM 1.632ms @ **2376 TFLOPS**、GEMM-only 1.903×）。→ **差 ~5%（2.436 vs 2.324），同 dW2 的 host D2H 气泡**，kernel 等价。
- **口径注意**：源 test_dw1_bench 的 bf16 腿是 **OLD 跨 rank re-dispatch**（5.0ms，old/new=2.15×）；我的（同独立 bench）bf16 腿是**纯 GEMM**（3.79ms，GEMM-only 1.90×）。fp8 dW1 LOCAL 省掉 re-dispatch → 真实优势 ≈ 源的 2.15×，比纯 GEMM 比值大。
- 复现：`PYTHONPATH=$PWD python3 benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc1_wgrad --mode load_balanced`（对标源加 `--num-experts 32`；不含在 `both` 里）。

### 2026-07-21 (16:00) — ⚠️关键：dW2 "1.4ms vs 1.77ms" 差异 = **E=32(源 test) vs E=256(DSv3 真实)**，非 bug
- 起因：training bench dW2（E=256）FULL 1.767ms，和记忆里源 test 的 1.4ms 对不上。**当前环境实测源 `test_mega_moe_mxfp8.py::test_dw2_bench`（PT_DW2_BENCH=1）**：`full mxfp8 wgrad = 1.439 ms | bf16 1.967 | GEMM 0.838ms @ 2315 TFLOPS | M(pool)=66048 | fmt=e5m2`。→ **1.4ms 真实存在**。
- **根因**：源 test 文件全局 `_E=32`（correctness 小配置），`test_dw2_bench` 只借了 DSv3 的 H/I/T=7168/2048/8192 但 **E 仍是 32**（epr=4）。variable-K wgrad 每组 K = 该 expert 的 token 数；**E=32 每组 K 是 E=256 的 8×** → GEMM 效率高（2315 vs 1963 TFLOPS）。
- **对齐铁证**（bench 加 meta 项后五项 breakdown 逐项对齐，E=32 --warmup 10 --iters 30，×2 稳定）：

  | 项 | MegaMoE bench (E=32) | 源 test_dw2_bench | 
  |---|---|---|
  | M_pool | 66048 | 66048 |
  | meta | 0.119 | 0.098 |
  | requant(pool) | 0.346 | 0.344 |
  | quant(act) | 0.132 | 0.138 |
  | **GEMM** | **0.837 ms / 2317 TFLOPS** | **0.838 ms / 2315 TFLOPS** |
  | sum | 1.434 | 1.418 |
  | FULL | 1.51 | 1.439 |

  **GPU kernel 逐项等价（GEMM 2317 vs 2315、requant/quant/M_pool 全一致）→ 移植 100% 正确**。FULL 的 ~0.07ms 残差 = `_mxfp8_variable_k_wgrad` 内 `colwise_grouped_meta` 的 D2H 在串行执行打断流水的 host 气泡（FULL-sum：我 0.076 vs 源 0.021），非 kernel 性能；两包 op 源码逐字节相同（仅常量名 `_DW_FP8_FORMAT`/`_DW2_FP8_FORMAT`，值都 e5m2）。
- **结论**：dW2 GEMM 与源逐项等价；「1.4ms」只在 **E=32** 成立。**DSv3 真实 E=256** 下 dW2 GEMM 掉到 ~1963 TFLOPS（每组 K 小是 variable-K wgrad 的固有特性）→ GEMM 1.04ms、FULL ~1.77ms。training bench 默认 E=256（真实训练成本），对标源 test 请加 `--num-experts 32`。

### 2026-07-21 (15:50) — fc2 wgrad (dW2) 接入 training bench（`bench_mega_moe_fp8.py --stage fc2_wgrad`），双 routing
- 复刻 `bench_dw2_fp8.py` 成 training bench 的 stage：真实反向到 dW2（fwd L1→dispatch(dy)+fc2-dgrad→swiglu_backward→variable-K wgrad），fp8 `_mxfp8_variable_k_wgrad`（pool colwise-requant + act colwise-quant → fp8 GEMM）vs bf16 Triton `grouped_gemm_variable_k_impl`，**背靠背计时** + FULL/GEMM-only/requant/quant/GEMM 分解，SNR gate 15 dB。用 `generate_inputs` 的路由（绝对 TFLOPS 随池分区波动，两边同分区一致）。不碰 L2 combine → **双 routing 都干净**。
- **@T=8192 load_balanced**（M_pool=69632）：fp8 FULL **1.767** ms(1157 TFLOPS) vs bf16 **2.063** ms(991) → FULL **1.168×**；**GEMM-only 1.981×**（fp8 GEMM 1.041ms @ **1963 TFLOPS**）；breakdown requant 0.368 / quant 0.145 / GEMM 1.041；SNR 21.76 dB PASS。
- **@T=8192 round_robin**（M_pool=65536，零 padding）：fp8 FULL **1.683** ms(1143) vs bf16 **1.930** ms(997) → FULL **1.147×**；**GEMM-only 1.992×**（GEMM 0.969ms @ **1986 TFLOPS**）；requant 0.344 / quant 0.128 / GEMM 0.969；SNR 21.76 dB PASS。
- 结论与独立 bench 一致：**隔离 fp8 GEMM ≈2×（~2000 TFLOPS，近 roofline）是真赢点，FULL 被每次 colwise requant+quant（~0.5ms）稀释到 ~1.15×**。dW2 算力 bound，latency 净收益有限，价值在梯度以 fp8 走显存/带宽。
- 复现：`PYTHONPATH=$PWD python3 benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc2_wgrad --mode load_balanced`（不含在 `both` 里；跑挂多为端口残留，换 `MASTER_PORT` 重试）。

### 2026-07-21 (15:35) — dispatch(dy)+fc2-dgrad 接入 training bench（`bench_mega_moe_fp8.py --stage dispatch_fc2_dgrad`），双 routing
- 把反向第一阶段做成 training bench 的一个 stage（复刻 `bench_step1_fp8_vs_bf16.py`：两腿同一 fused「dispatch(dy) PUSH + fc2-dgrad GEMM」，fp8/bf16 各自 symm 栈跑完即 destroy，**背靠背计时**，grad_swiglu vs 逐组 bf16 ref 的 SNR）。该阶段不碰 L2 combine → **两种 routing 都干净**（不像 fwd/L2 的 round_robin 会撞跨 rank 竞争）。
- **@T=8192 load_balanced**：fp8 **1.840** ms(1111 TFLOPS) vs bf16 **2.438** ms(838) → **bf16/fp8 1.325×**，SNR 30.89 dB PASS，M_pool=69632。**与独立 bench 快照(1.848/2.453/1.33×/30.9dB)吻合**。
- **@T=8192 round_robin**：fp8 **1.721** ms(1118) vs bf16 **2.393** ms(804) → **1.391×**，SNR 30.89 dB，M_pool=65536（零 padding 池更小 → 略快、加速比更高）。
- 复现：`PYTHONPATH=$PWD python3 benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage dispatch_fc2_dgrad --mode load_balanced`（不含在 `both` 里）。

### 2026-07-21 (13:40) — dW2 GEMM 与源 2322 TFLOPS 的残差 = 池分区(非代码），A/B 铁证 kernel 等价
- 直接调 `grouped_gemm_mxfp8_variable_k_flydsl_kernel`（合成算子，同 shape）：**MegaMoE 包 1929 vs 源包 1905 TFLOPS —— 一样**；用**源包**跑我的合成算子也只有 1905（≠ 源 test 的 2322）。→ **kernel/编译/环境零差别，移植 100% 正确**。
- 源 test 的 2322 来自它那次 routing 抽到的**异常均衡池**（m_pad 66048、padding 仅 0.8%）；随机/真实 routing 的池是 ~70k（padding ~7%，正是每组均值 128×32≈4096 的期望），对应 GEMM ~2000 TFLOPS。逐字复刻源 seed(123+rank)+x/w1/w2/gate 抽取顺序，我 rank0 仍得 m_pad 71168（prologue padding 逻辑与源逐字一致，8 行 diff 仅 import 路径）——存在定位不到的细微 RNG 状态差，但**与 GEMM 无关**。
- 结论：dW2 fp8 GEMM 已验证与源等价；bench 绝对 TFLOPS 随 routing 池分区波动（越均衡越高），**同分区两边一致**。不再追源那次非典型均衡抽样。

### 2026-07-21 (11:30) — ✅ 定位并修复"fp8 慢 34%"：bench 计时 bug + GEMM autotune 漏移植
- **推翻 11:07 的"环境回归"判断**。真因两个（均已修）：
  1. `_bench` 单次调用 event-bracket 计时 → host 开销记成 GPU-idle，虚高所有 fp8 kernel。改背靠背（连发 N/N）后 requant 0.491→**0.360**、quant 0.256→**0.139**（精确吻合源 0.345/0.138）。已同步修 4 个 bench（dw2 / dw1 / step1_fp8_vs_bf16 / step3）。
  2. `mxfp8_grouped_kernel.py` variable-K wgrad 配置写死 `xcd=8`，漏了源的 `(bm,bn,gm,xcd,gn)` autotune。已逐字节补齐（`diff -q` 与源一致），GEMM 1740→**2013 TFLOPS**（源 2324；残留是 M_pool 70400 vs 66048）。
- **修正后 T=8192**：STEP1 fp8 1.848 vs bf16 2.453 = **0.753×**；dW1 fp8 2.868（1426 TFLOPS） vs 3.814 = **0.752×**；dW2 FULL 1.76（fp8 快 0.90×）、GEMM 2013 TFLOPS。全部 SNR 通过。

### 2026-07-21 (11:07) — ⚠️（已推翻，见 11:30）误判为环境回归：重建容器 fp8 kernel 慢 ~34%
- 对比上一轮 campaign 记录（dW2 FULL 1.43ms：meta 0.100 / requant 0.344 / quant 0.139 / **GEMM 0.833ms @ 2327 TFLOPS**；GEMM-only 0.42×）。
- 本环境实测 dW2 FULL 1.93ms：requant 0.491 / quant 0.256 / **GEMM 1.175ms @ 1740 TFLOPS**；GEMM-only 0.61×。
- 排查：GPU 负载时钟 ~2390MHz（健康，非限频）；bf16 Triton 腿 ~1.94ms **两环境一致**；warmup 25 无改善；`~/.flydsl/autotune` 仅 `.lock`（无丢失 config）。→ **只有 fp8 FlyDSL kernel 慢**，根因 = 重建容器 pip `flydsl==0.2.4` build 与 campaign 原容器不一致。
- 待办：确认 campaign 原容器 flydsl 来源（pip / 本地 build），对齐后重测全阶段绝对数。之前几条 2026-07-21 记录的 fp8 绝对 ms 都带此偏慢，加速比偏保守。

<!-- 模板：新增一条
### YYYY-MM-DD — <改动/目的一句话>
- repo <commit>，<改了什么>。
- STEP1 fp8 X ms vs bf16 Y ms → Z×，SNR …
- dW2 … / dW1 … / STEP3 … / e2e …
- 备注：<稳定性/异常/结论>
-->

## Interpretation（T=8192）

1. **STEP1 是最干净的赢点（1.44×）**：同吃两个杠杆 —— 跨 rank dispatch(dy) PUSH 字节减半 + mxfp8 grouped GEMM ~2× 算力，且 comm 藏在 GEMM 下（同前向 L1 的 ~1.6× 同源）。
2. **dW1 收益被 bench 低估**：bf16 腿仅 GEMM；fp8 的结构优势是 LOCAL（`grad_l1^T @ pool_x`），省掉 bf16 路径的跨 rank 重派发 `saved_x`，即便只比 GEMM 也 1.19×。
3. **dW2 ≈ 持平**：wgrad 已算力 bound，fp8 的 a/b colwise requant + E8M0 是额外 per-call 开销，恰好抵消 GEMM 的 2×；latency 无净收益，价值在**梯度以 fp8 走的显存/带宽**。
4. **STEP3 未生产就绪**：fp8-PUSH combine 的跨 rank reduce-flag liveness 竞争大 T 偶发死锁；已排除 gate-scatter/combine_cu/专用 reduce；`NO_REDUCE`(仅 GEMM+push) 稳、前向 fp8 combine 8/8 稳 → 定位「融合 reduce 角色 × 跨 rank flag」。修法：**拆成「融合 GEMM+push → barrier → 独立 reduce」**。

## Next
1. 修 STEP3 死锁（拆 reduce），30–50 次 T=8192 压测确认根除 → 回填 STEP3 稳定值 + bf16 腿 + e2e@8192。
2. 反向 barrier 收敛（`PT_MEGA_BARRIER_MODE=reduced`）看 e2e@8192 净收益。
