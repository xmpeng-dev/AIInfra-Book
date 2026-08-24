# Mega MoE MXFP8 — fc1_dgrad 提前 + (fc1_wgrad ∥ combine) 单 kernel CU 分角色 overlap 设计 note（T=8192）@ MI355X

> **What**: `fc1_dgrad_combine`(3.03ms)是 fp8 反向单项最大、仅 1.26× vs bf16 的最弱腿。本 note 记录一个
> 重构方案:把 **fc1-dgrad GEMM 拆成独立 kernel 提前算完**,然后在**一个 kernel、一个 stream** 里按 **CU 分角色**
> 让 **fc1_wgrad(dW1,LOCAL 纯算力)** 与 **combine(PUSH+reduce,纯 XGMI 通信)** 各占一批 CU 各跑各的、天然并发 ——
> overlap 阶段的 combine 不再和 GEMM 有生产者-消费者耦合。
> **Where**: MI355X / gfx950 ×8,容器 `xiaoming-dev`。DSv3 EP8 T=8192 H=7168 I=2048 E=256 topk=8。
> **Status**: 设计/计划(未实现)。日期 2026-07-24。前置:fused grad_l1 dual-quant 已接入(commit a8d15503)。
> **关键决定**: overlap 用**单 kernel 内 CU 分角色**(复用现 combine kernel 的 block_index 切角色架构),
> **不用两 stream**(避免 ROCm 双 kernel 被 CU 争用串行化的不确定性;单 kernel 内不同 CU 由硬件天然并发)。

---

## 1. 现状:`fc1_dgrad_combine` 为什么是最弱腿

当前 `grouped_gemm_combine_mxfp8_flydsl_kernel`(`grouped_gemm_combine_fp8_kernel.py`)是 **一个 kernel 三角色**,
按 block_index 切 CU:

- **GEMM role**(`[gemm_base, ...)`): mxfp8 NT tile(`gemm_mxfp8_nt_tile`,K=2I)+ CShuffle mxfp8-quant
  epilogue → 写 **LOCAL fp8 L2Y**(`L2Y_FP8`/`L2Y_SCALE`),然后 `atomic_add(combine_flag[block_m])` 通知。
- **COMBINE role**(`[0, ncomb)`): **spin `combine_flag[block_m]`(等 GEMM 出该 tile)** → 读 LOCAL fp8 L2Y →
  push 到 peer `comb[slot]`(XGMI)+ 置 `reduce_flag`。
- **REDUCE role**: spin `reduce_flag` → fp8-dequant topk 求和 → `output`(dx)。

**瓶颈根因**:GEMM 和 combine 在同一 kernel 里做**生产者-消费者**——combine 的 CU 要 **自旋等** GEMM 的 CU
逐 tile 产出(`combine_flag`/`sb_l2` rendezvous)。两个角色抢同一批 CU/占用:
- combine CU 早期空转等 GEMM,GEMM CU 又被 combine/reduce 挤掉一部分 → 谁都跑不满。
- 还带来 co-scheduling deadlock 风险(dedicated-reduce 会死锁,只能 tail-reduce;见 CU 重扫 note)。
- CU 配比只能折中(STEP3 最优 24 combine),GEMM 拿不到全部 CU。

隔离 stage 数(T=8192, load_balanced):`fc1_dgrad_combine` **3.03ms / 1349 TF**;相邻的 `fc1_wgrad`(dW1)
**2.88ms**(其中 GEMM 2.037 + quant 0.258 + requant 0.372 + meta 0.119),**dW1 是 LOCAL,零跨卡通信**。
两条腿当前**串行** ≈ 5.9ms。

---

## 2. 数据依赖(overlap 可行性的关键)

```
grad_l1 (SwiGLU^T 出) ──┬─ fc1_dgrad = grad_l1 @ w1^T ──→ combine(PUSH 跨卡)──→ reduce ──→ dx
                        │        (本地 fp8 结果)
                        └─ dW1 = grad_l1^T @ pool_x  (LOCAL, 只需 grad_l1 + fwd pool_x)
```

- `fc1_dgrad → combine`:**有依赖**(combine 消费 dgrad 的本地 fp8 输出)。
- `dW1`:**独立**——只依赖 grad_l1(已 fused 量化)+ pool_x(前向 pool,已 clone),**不依赖 fc1_dgrad,也无通信**。

⇒ `dW1`(算力)与 `combine`(通信)天然可 overlap;二者都在 `fc1_dgrad` 之后。

---

## 3. 方案:拆 fc1_dgrad + 单 kernel 内 (dW1 ∥ combine) CU 分角色

**Step 1 — fc1_dgrad 独立 kernel(提前算)**:把 combine kernel 的 GEMM role 抽成独立 kernel,用 **全部 CU** 跑
mxfp8 NT GEMM(K=2I)+ CShuffle fp8 epilogue → 写满 LOCAL fp8 L2Y(+scale)。不需要 combine_flag rendezvous
(自己一个 kernel 全跑完)。全 CU → GEMM 跑在自己的 roofline 上。

**Step 2 — 单 kernel 内 dW1 ∥ combine(CU 分角色,一个 stream)**:新的融合 kernel,按 `block_index` 切两类 CU
角色(和现 combine kernel 的 GEMM/combine/reduce 切法同构):
- **WGRAD role**(占大部分 CU):dW1 = grad_l1^T @ pool_x variable-K wgrad(LOCAL,纯 MFMA 算力,无通信)。
- **COMBINE+REDUCE role**(占少部分 CU):读**已算好**的 L2Y → XGMI push 到 peer → reduce → dx。
- 一个 kernel launch、一个 stream:**不同 CU 由硬件天然并发**,wgrad 的 MFMA 和 combine 的 XGMI 各跑各的,
  彼此不同步、不共享 flag(除各自 role 内部的)。
- 因为 fc1_dgrad 已算完,**combine role 不再自旋等任何 GEMM**(`combine_flag` 由 Step1 结束即就绪) ——
  正是这次想要的"overlap 阶段根本不和 GEMM 通信"。combine 的 XGMI 延迟被同 kernel 里 wgrad 的算力盖住。

**为什么单 kernel 而非两 stream**:两 kernel 分 stream 依赖 ROCm 运行时把它们 co-schedule 到不同 CU,常被
CU 争用/队列串行化(不确定);**单 kernel 内 block_index 切 CU 是确定的硬件并发**,且复用现成架构。

**串行 vs overlap 估算**(隔离数外推,需 e2e 实测):
- 现状:`fc1_dgrad_combine`(3.03) + `dW1`(2.88) ≈ **5.9ms**。
- 方案:`fc1_dgrad`(独立全 CU GEMM,估 ~1.5–2.0) + 单 kernel max(`dW1` wgrad-role, `combine` role)
  ≈ 1.7 + ~2.6 ≈ **~4.3ms** → **省 ~1.3–1.6ms/backward**(粗估,乐观)。
- 收益两部分:(a) fc1_dgrad 拿满 CU 比在融合 kernel 里被挤更快;(b) combine 通信被 dW1 算力盖住 + 免 GEMM 自旋。

---

## 4. 实现要点 / 改造面

1. **拆 GEMM role → fc1_dgrad 独立 kernel**:复用 `gemm_mxfp8_nt_tile` + `StoreCQuantMxfp8CShuffle` 写
   `L2Y_FP8/L2Y_SCALE`;新独立 launcher(全 CU,grid = tiles)。从 `_do_gemm_tile` 抽出。结束时把
   `combine_flag` 全置位(或 epoch 直接标全就绪),让后续 combine role 无需自旋。
2. **新 wgrad+combine 融合 kernel**(单 kernel,一个 stream,`block_index` 切角色):
   - `block_index < ncomb`:COMBINE role(读已算好的 L2Y → push,**去掉 GEMM 自旋**)。
   - `[ncomb, ncomb+nreduce)`:REDUCE role(不变,spin reduce_flag → dx)。
   - `>= wgrad_base`:**WGRAD role** —— dW1 variable-K wgrad tile(grad_l1^T @ pool_x)。这是最大改造:
     要把 `grouped_gemm_fp8_variable_k_impl` 的 tile 逻辑作为一个 role 塞进这个 kernel(共享 LDS/grid 分配)。
   - CU 配比:combine ~24(重扫)、reduce 走 tail 或 0、wgrad 拿其余大头。
3. **编排**(`mega_moe_backward_fp8_impl`):`fc1_dgrad`(kernel1)→ `wgrad_combine_fused`(kernel2,一个 stream)。
   dW1 不再单独 launch,并进 kernel2。
4. **buffer**:fc1_dgrad 的 LOCAL fp8 L2Y 是现融合 kernel 已有的中间产物(write-through sc1),拆开后仍是本地
   buffer,combine role 读它 —— **不新增 HBM 往返**(只多一次 launch)。scratch `_L2Y_FP8_SCRATCH` 按 (M,H) 复用。

---

## 5. 风险 / 开放问题(实现前要验)

- **单 kernel 内两 role 真并发?** 同一 kernel 的不同 CU 本就并行,但要确认 **wgrad role(MFMA、可能高 LDS/VGPR)
  与 combine role(XGMI)的占用不互相饿死** —— rocprofv3 trace 看两 role 的 wall 是否 ≈ max 而非 sum。
  wgrad tile 的 LDS 需求(mxfp8 GEMM)较大,和 combine 的 LDS 合到一个 kernel 可能压占用,需权衡 CU 配比。
- **融合难度**:把 variable-K wgrad GEMM 塞进 combine kernel 是最大工作量(两套 tile/LDS 逻辑共存于一 kernel);
  若太复杂,退一步可先验证 Step1(拆 fc1_dgrad)单独收益,再评估 role 融合。
- **fc1_dgrad 独立后是否真更快**:K=2I 瘦-K,占用受限,加满 CU 也可能吃不满 → 收益打折;要实测。
- **combine role 的 CU 需求**:去掉 GEMM 自旋后,纯 push 可能更少 CU 就够(重扫)。
- **reduce 的 flag 依赖**:reduce 仍等 combine push 完(reduce_flag),同 kernel 内 role 间协作,和现在一致。
- **正确性**:byte 级不变(GEMM/quant/push/reduce/wgrad 数学都不动,只是拆 kernel + role 重组);e2e 梯度 SNR 必须全 PASS。
- **Rule 11 迁移**:overlap 是结构级,和 benchmark idiom 无关,收益应 1:1 迁移真实训练。
- **对比基线**:必须同 session e2e A/B(现融合 vs 拆分+role-overlap),隔离 stage 数不忠实(combine 单跑没 wgrad 可盖)。

---

## 6. 验证计划

1. 先实现 Step1(fc1_dgrad 独立 kernel)+ combine-only kernel,dW1 仍单独 launch(**不 role-overlap**),
   e2e 确认 byte-exact / SNR PASS 且不慢(纯拆分基线,先验证 fc1_dgrad 独立收益 + combine 免自旋)。
2. 再把 dW1 wgrad 作为 role 并进 combine kernel(单 kernel CU 分角色),rocprofv3 trace 确认 wgrad role 与
   combine role 在同一 kernel 内 wall ≈ max(而非 sum),两 role 未互相饿死。
3. e2e A/B(同 session,交替跑,pkill 清理防 OOM):fp8 fwd+bwd 现状 ~14.3ms vs overlap 后;目标省 ~1ms+。
4. 全量梯度 SNR(dx/d_topk_w/dW1/dW2 ≥15dB)PASS。
5. 记录到 `mxfp8_moe_bwd_perf_summary.md`。

---

## 7. 实现进度（Step 1 = 拆分,未 role-overlap）

**已实现**(`grouped_gemm_combine_fp8_kernel.py` + `mega_moe_backward_fp8_impl.py`):
- combine kernel 加 `phase` constexpr:phase=1 = fc1_dgrad GEMM-only(全 CU 写 L2Y + 置 combine_flag);
  phase=2 = combine PUSH + dedicated reduce(无 GEMM,combine_flag 已由 phase1 置好→免自旋等 GEMM);phase=0 = 原融合。
- wrapper 加 `split=True`:同 stream 依次 launch phase1 → phase2(epoch bump 只在 phase1 做,phase2 复用同 parity/bank)。
- backward `_STEP3_SPLIT=True`(可切回 False 做 A/B),combine=24 / reduce=16。dW1 仍单独 launch(Step 2 再 role 融合)。
- **踩坑**:(a) `block_m/block_n` 在 phase1 与 phase0 两个 const_expr 兄弟分支都赋值 → AST rewriter None-init 报错;
  改成嵌套 const_expr + phase1 用独立变量名 `bm_g/bn_g`。(b) `if phase==..` 放进 `@flyc.jit launch` 内会被重写导致
  `grid_size` 不绑定 → 改成闭包 ternary 算 grid_size + 按 `_do_bump` 在 _compile 层(Python)条件定义两份 launch。

**已验证(EP4,4 卡,clean GPU 4-7)**:e2e 梯度 SNR **dx=21.9 / d_topk_w=23.0 / dW1=19.5 / dW2=19.7 dB PASS**,
与融合版**逐位相同**(证明拆分 byte-correct)。EP4 perf 不可比(配置不同)。

**⚠️ 阻塞**:EP8/T=8192 的 perf A/B(拆分 vs 融合)需要 8 卡,但 **GPU 0-3 被之前 e2e 连跑崩溃的 MPI 残留进程
wedge 住**(每卡 ~283GiB 驻留,SDMA 卡死;本机 `rocm-smi --gpureset`/`--setperflevel` 全 "Not supported",
残留进程跨 PID namespace 杀不掉)。GPU 4-7 干净。需要 node/container 重置后再跑 EP8 perf。
- **教训**:e2e 千万别在一个 container invocation 里 back-to-back 连跑;崩溃会 wedge GPU,且本机无法 reset。

## 8. EP8 perf A/B 结果(2026-07-24, flydsl 0.2.4, 8 卡干净, 同 session)

| variant | fp8 fwd+bwd | vs bf16 | 梯度 SNR |
|---|---|---|---|
| **fused**(现状,`_STEP3_SPLIT=False`) | **14.34 ms** | 1.42× | PASS |
| **split**(Step 1,phase1 fc1_dgrad + phase2 combine) | **23.11 ms** | 0.88× | PASS(正确) |

- flydsl 0.2.4 **不是**变慢原因(fused 14.34 与升级前 14.25/14.33 一致)。
- **split 是 +8.8ms 大回退。** ⇒ **本方案前提被证伪**:融合 kernel 的 combine role 不是"空转自旋等 GEMM",
  而是**流水线**——每个 GEMM tile 一算完,combine CU 立刻 push 它(算力与 XGMI 通信重叠)。拆成
  phase1(全 GEMM)→ phase2(全 combine)把这条流水线**串行化**了,还被迫用**更慢的 dedicated reduce**
  (phase2 无 GEMM tile → 不能 tail-reduce;dedicated reduce 本就完败 tail-reduce,见 CU 重扫 note)。
- Step 2 的 dW1‖combine overlap 最多只能盖住 dW1 的 ~2.9ms,**远补不回**拆分丢掉的 8.8ms。⇒ **方案作废**。

### 结论 & 后续方向
- **放弃"拆 fc1_dgrad + 拆 combine"**。融合 fc1_dgrad_combine 的 GEMM↔combine 流水线已经很好,别拆。
- 若还想 overlap:更靠谱的是**保留融合 fc1_dgrad_combine 不动**,让它整体与 **dW1(LOCAL 纯算力)** 并行
  (max(3.03, 2.9)≈3.03 而非 5.9,省 ~2.9ms),而不是拆 fc1_dgrad。但两个 GEMM+comm 大 kernel 抢 CU 能否真并行
  需实测(和之前两-stream 顾虑一样)。
- 代码:`_STEP3_SPLIT` 默认 False(走融合),phase 基础设施保留但休眠(不影响 fused 路径)。可按需删。

## 9. 单卡 dgrad/wgrad GEMM 微基准(评估 dw+combine fuse, 2026-07-24, GPU 单卡, EP8-per-rank shape)

`bench_dw1_dgrad_singlegpu.py`(G=32, P=65536, 2I=4096, H=7168, 单卡合成操作数,无通信):

| num_cu | wgrad ms | wgrad TF | dgrad ms | dgrad TF |
|---|---|---|---|---|
| None | 1.866 | 2062 | **1.664** | 2312 |
| 304 | 1.868 | 2060 | 2.098 | 1834 |
| 192 | 1.867 | 2061 | 2.116 | 1818 |
| 64  | 1.871 | 2057 | 2.119 | 1817 |

**发现**:
1. 两个 GEMM 本体已经很强:**dW1 wgrad ≈1.87ms(2060 TF)**、**fc1_dgrad ≈1.66ms(2312 TF, num_cu=None)**。
2. **两者对 `num_cu` 完全不敏感(304→64 一模一样)** ⇒ grouped GEMM kernel **不理会 num_cu 上限**,硬件把 grid
   铺满所有 CU。附带:dgrad 传了 num_cu(哪怕 304)比 None 慢 26%(1.66→2.10),是另一个 quirk。

**对 dw+combine fuse 的评估(结论:收益小、代价大,不建议做)**:
- 因为 GEMM 不认 num_cu,**无法用 num_cu 给 combine 让出 CU** 来做 CU 划分;真要 CU 分角色,只能把 wgrad tile
  逻辑当成一个 role **手写进 combine kernel**(用 block_index 划分)——大工程,还要和 combine 的 LDS/占用共存。
- 两-stream 并行也不行:wgrad 和 fc1_dgrad_combine 都是铺满 CU 的大 kernel,互相抢 CU 会串行化(和之前顾虑一致)。
- 上限收益:overlap 最多盖住 combine 的**通信**部分。fc1_dgrad_combine=3.03ms 里 dgrad GEMM≈1.66ms,
  ⇒ combine+reduce(通信)≈1.3ms;即 dw‖combine 理想最多省 ~1.3ms。
- 但第 8 节已证:一旦为了 overlap 去**拆 fc1_dgrad**(让 combine 不等 GEMM),就丢掉融合 kernel 的 GEMM↔combine
  流水线,直接 +8.8ms —— 远大于能省的 1.3ms。**综合:dw+combine fuse 净收益大概率为负,性价比低,不做。**

### 最终结论
- fc1 反向的两个 GEMM 已接近算力上限(2000-2300 TF),combine 已有良好的 GEMM↔combine 流水线。
- 想要的 overlap 省不回拆分/CU-划分的代价 ⇒ **放弃 dw+combine / 拆 fc1_dgrad 这条线**。
- `_STEP3_SPLIT` 保持 False(走原融合);split/phase 代码休眠或删除(见第 10 节待办)。

## 10. 干净复测(n10-29, flydsl 0.2.4, 8 卡干净) + reduce_cu 修复(2026-07-25)

之前 +8.8ms 一半是**污染**、一半是**dedicated reduce CU 太少**。干净机器复测:

| config | fp8 fwd+bwd | vs bf16 | 备注 |
|---|---|---|---|
| fused(baseline) | 14.34 ms | 1.42× | 原融合 |
| split, reduce_cu=16 | 23.19 ms | 0.86× | 干净机器也慢 → reduce 瓶颈 |
| **split, reduce_cu=128** | **16.67 ms** | 1.20× | reduce CU 16→128 **省 6.5ms** |

- **split(Step 1)真实 de-pipeline 代价仅 ~+2.3ms**(16.67 vs 14.34),不是 +8.8。之前结论作废(reduce 配错 + 污染)。
- **关键洞察**:fused `fc1_dgrad_combine`(3.03ms)是 **combine/comm-bound**(dgrad GEMM 1.66ms 已被藏在里面)。
  ⇒ 把这 ~3ms 的 combine 通信与 dW1 的 ~2.9ms 算力 **overlap(Step 2)**,理论省 ~2.9ms > de-pipeline 的 2.3ms
  ⇒ **净胜、能超过 fused**。用户直觉成立,继续做 Step 2。
- **持续优化(不否决)**:reduce_cu 还可再调;Step 2 用 dW1‖phase2(combine) 双 stream 或单 mega-kernel CU 分角色。

## 11. Step-2 overlap 实测(n10-29, flydsl 0.2.4, 8卡干净, 2026-07-25)

dW1(LOCAL 纯算力)‖ combine(XGMI 通信)双 stream overlap,两种分解:

| 方案 | fp8 fwd+bwd | vs bf16 | 说明 |
|---|---|---|---|
| fused baseline(无 overlap) | 14.34 ms | 1.42× | 基线 |
| split(phase1+phase2) ‖ dW1 | 15.08 ms | 1.33× | 拆分暴露了 fc1_dgrad GEMM(~2ms),得不偿失 |
| **fused fc1_dgrad_combine ‖ dW1** | **14.18–14.23 ms** | 1.42× | **稳定 +0.14ms 胜**(3 次 14.18/14.18/14.23) |

- **overlap 确实有效但被 CU 争用卡住**:dW1 GEMM 和 fused combine 都想铺满 CU(且 GEMM 不认 num_cu),
  双 stream 上大部分串行,只有 combine 的 XGMI 空隙漏进一点 dW1 → 只省 0.14ms(理论可省 ~1.87ms dW1 GEMM)。
- **不拆 fc1_dgrad 更好**(14.18 < 15.08):保留 combine 内 GEMM↔push 流水线,dgrad 仍被藏住。
- 当前最优配置已设为 `_STEP3_SPLIT=False, _STEP3_OVERLAP_DW1=True`(安全 +0.14ms,不回退)。

### 要拿到大头(~1.87ms)需要 CU 分角色的 mega-kernel
两 stream 争用是天花板。真要让 dW1 与 combine 并行,得把 **dW1 variable-K wgrad tile 逻辑作为一个 role
按 block_index 塞进 combine kernel**(combine 24 + reduce + wgrad 各占一批 CU),硬件保证并行。工程量大、
风险高(dW1 在部分 CU 上是否还快取决于它的 num_cu 敏感性——单卡测是 flat,但那是 num_cu 被忽略的假象,
需在 mega-kernel 里实测)。待用户确认是否投入。

## 12. 单 stream mega-kernel 方案(硬约束:绝不用双 stream, 2026-07-25)

用户约束:**绝对不能双 stream**。⇒ overlap 只能靠**单 kernel 内 block_index 分角色**(dW1 wgrad 作为一个 role
和 combine/reduce/dgrad-GEMM 角色在同一 kernel、同一 stream,靠不同 CU 硬件并行)。已把两-stream 路径关掉
(`_STEP3_OVERLAP_DW1=False`,回到安全 fused 14.34)。

### 可行性(已核实)
- **block size 一致**:combine `_BLOCK_THREADS=512`,wgrad kernel block=512 ✅。
- **LDS**:combine 的 GEMM role LDS ~128KB(8 buffer fp8);wgrad LDS 也 ~128KB(a/b 各 4 buffer,BLOCK_M/N=256,
  BLOCK_K=128 → a_lds=16KB×4 + b_lds=16KB×4 = 128KB)。两者都是 1 block/CU。mega-kernel 必须**union LDS**
  (取 max≈128KB,combine-role 和 wgrad-role WG 复用同一块 LDS 的不同 view)→ 占用与现在持平,不恶化 ✅。
- **wgrad tile 不是可复用原语**:combine 的 dgrad role 用 `gemm_mxfp8_nt_tile`(NT 定长 K);wgrad 是 **TN 变长 K**
  (`grad_l1^T@pool_x`),tile 逻辑埋在 `kernel_grouped_mxfp8_wgrad._do_tile`(mxfp8_grouped_kernel.py,~370 行
  双缓冲 MFMA:G2SLoader/S2RLoader/MfmaScale16x16x128/ScaleS2R/ScaleBComb/_wgrad_ssa_chunk/tail)。**要抽成
  可复用 role**。

### 工程量 & ROI(诚实评估)
- **收益上限**:把 dW1 GEMM ~1.87ms 藏进 combine 通信 → e2e 14.34 → 理论 ~12.5ms(1.6× vs bf16)。
- **工作量大**:抽 370 行变长-K wgrad tile 成 role + union LDS + block_index 分派 + 处理 wgrad 的 preshuffle
  pre-kernel + dW1/dx/dW2 byte-exact + CU 配比调优。多轮、高风险(寄存器压力、correctness、真并行度未知)。

### 增量计划(降风险)
1. **R1 抽原语**:把 `kernel_grouped_mxfp8_wgrad._do_tile` 抽成 `gemm_mxfp8_variable_k_tile(lds_view, ...)`
   可复用函数;用它重建 standalone wgrad kernel,验证 byte-exact + perf 不退化(纯重构,不碰 combine)。
2. **R2 加 wgrad role**:combine kernel 加 block_index 段 → 调用该原语;union LDS;wgrad 操作数(colwise grad_l1
   + colwise pool_x + 预 preshuffle scale)从 orchestration 传入。correctness(dW1+dx+dW2 SNR)。
3. **R3 perf + CU 配比**:combine/reduce/dgrad/wgrad 四角色 CU 分配调优,目标 e2e < 14.34,兑现 dW1 隐藏收益。

## 13. mega-kernel 关键 de-risk + 精化计划(用户选 "full", 2026-07-25)

**LDS 已确认 byte-identical(大幅简化,无需 union)**:
- combine GEMM-role LDS:`_mx_a_lds=_mx_b_lds=(256/2)*_MXFP8_BLOCK_K(128)=16384` × 8 buffer + C_lds_shuffle。
- wgrad LDS:`a_lds_size=b_lds_size=(256/2)*128=16384` × 8 buffer。
- ⇒ 两者 8 个 fp8 buffer **完全同尺寸**。wgrad role 可**直接复用** combine 的 `A_lds_cur_0..B_lds_next_1`,
  不需要 union。SharedStorage 保持 combine 现有的即可。occupancy 不变(仍 1 block/CU)。

**wgrad tile 依赖(都在 mxfp8_grouped_kernel.py, 可 import)**:G2SLoader / S2RLoader / MfmaScale16x16x128 /
ScaleS2R / ScaleBComb / StoreCPerTensor / _wgrad_ssa_chunk / _wgrad_mx_body_4buf / compute_global_swizzle /
xcd_remap_pid / _wgrad_block_mn / make_fp8_buffer_tensor_rebased / _load_go。`_do_tile`(1078-1255)只是编排它们
+ 闭包常量(BLOCK_M/N/K, N_TILES, offsets)+ LDS + 操作数。

**preshuffle**:wgrad 需要先把 scale 预 preshuffle(pre_kern)进 a_sp/b_sp workspace。mega 里在 orchestration
里先跑这个便宜的 pre-kernel(不是 combine,单独 launch,快),wgrad role 读预好的 scale。

**四角色 mega-kernel(单 stream,保留 fc1_dgrad 融合以隐藏 dgrad)**:
grid = combine_cu + reduce_cu + dgrad_tiles + wgrad_tiles;block_index 分派:
combine / reduce(tail 或 dedicated)/ dgrad-GEMM(gemm_mxfp8_nt_tile)/ wgrad(抽出的 variable_k tile)。

**构建步骤**:
1. 抽 `_do_tile` → 模块级 `gemm_mxfp8_variable_k_tile(pid, lds, A,B,C, A_scale,B_scale, go_div, m_total, cfg)`;
   用它重建 standalone `kernel_grouped_mxfp8_wgrad`,验证 byte-exact + perf 不退化(不碰 combine)。
2. combine kernel:加 wgrad-role 分支调用该原语(复用 8 buffer LDS);combine 签名加 A/B/C/scale/go/m_total 参数;
   orchestration 传入 colwise grad_l1(a)+ colwise pool_x(b)+ 预 preshuffle scale + dW1 输出 buffer。
3. correctness(dW1+dx+dW2 SNR)→ CU 配比调优。

**状态**:设计/de-risk 完成;开始 step 1 抽原语。当前 live 代码 = 安全 fused baseline(两-stream 已关)。

## 14. Step-1 抽取失败 → flydsl 架构约束(2026-07-25)

试把 `kernel_grouped_mxfp8_wgrad._do_tile` 抽成模块级 `gemm_mxfp8_variable_k_tile`,编译报错
`cannot evaluate dynamic 'Boolean' as Python bool during tracing`。

**根因(重要 flydsl 约束)**:`@flyc.kernel` 只对 **kernel 自身源码 + 其内嵌 def** 做 AST 改写(把动态
`if`/`for range(dyn)` 转成 scf.if/scf.for)。wgrad tile 里有动态控制流:`if wave_m==1: s_barrier()`、
`for _c in range(_nfull)`(_nfull 运行时)、`if k_abs<k_iters`。抽到**模块级函数**后这些退化成普通 Python
→ 报错。模块级 helper(`_wgrad_ssa_chunk`/`_wgrad_mx_body_4buf`)能用是因为只含 `range_constexpr`(编译期)。

**已还原**:standalone wgrad kernel 恢复原内嵌 `_do_tile`,e2e dW1/dW2/dx SNR 全 PASS(dx 曾出现 14.4 是
随机数据方差,复跑 21.7/21.8)。生产未坏。(遗留:模块级 dead `gemm_mxfp8_variable_k_tile`+`_VKWgradCfg`
+`import collections`,无害,下轮清理。)

**修订 mega-kernel 方案(二选一)**:
- (A) wgrad tile 作为**内嵌 def** 直接写进 combine kernel 的 `kern`(复制 ~175 行编排,复用模块级
  `_wgrad_ssa_chunk`/`_wgrad_mx_body_4buf`;combine 模块 import 那 ~15 个 helper/class)。可靠但重复代码。
- (B) 把 tile 的动态控制流改写成 flydsl 显式 API(`_emit_if_then` 替 `if`,动态 for 用 flydsl 的 scf.for
  wrapper),让模块级原语可用 → 无重复。需先确认 flydsl 有动态 for 的显式 API。
- 下轮先查 flydsl 动态 for API;有则走 (B),没有走 (A)。

## 15. 单-stream mega-kernel 实现进度(2026-07-25, ~90% 完成,卡在 flydsl tracer)

**已实现(加法式,`wgrad` 编译开关 + `_STEP3_MEGA` 运行开关门控)**:
- `grouped_gemm_combine_fp8_kernel.py`:import wgrad helper(utils.gemm_helper + gemm_fp8_grouped + mxfp8_grouped);
  `_compile` 加 wgrad cfg 常量(OUT_M=2I, OUT_N=H, m_total=Ppad, BM/BN/BK=256/256/128, cbsz/blgp=1 e5m2)+
  `wgrad` 参数;`kern` 加 7 个 wgrad 参数 + 内嵌 `_do_wgrad_tile(t)`(镜像 standalone `_do_tile`,复用 combine
  的 8 个 fp8 LDS buffer + 模块级 `_wgrad_ssa_chunk`/`_wgrad_mx_body_4buf`);block_index 分派加 wgrad 段
  (`[gemm_base+dgrad_tiles, ...)`)+ grid 扩 `_WG_TOTAL`;launch 两变体 + wrapper 传 wgrad 操作数(缺省 dummy)。
- `mxfp8_grouped_kernel.py`:加 `prepare_variable_k_wgrad_operands`(prep + 跑 preshuffle pre-kernel,不含 GEMM)。
- `mega_moe_backward_fp8_impl.py`:`_STEP3_MEGA` 分支 —— 建 wgrad 操作数(A=colwise grad_l1 dual-quant,
  B=colwise-requant pool_x)+ preshuffle,一次 launch combine(带 wgrad_operands)→ dx + dW1(从 WG_C 取)。

**已验证**:非-mega 路径(wgrad 编译 out)**完全 inert**,e2e 14.19-14.25ms、dW1/dx/dW2 SNR 全 PASS。生产安全。

**当前卡点(mega 路径)**:e2e 编译 mega-kernel 时报 `UnboundLocalError: cannot access local variable '_c'`
—— 出在内嵌 `_do_wgrad_tile` 的动态循环 `for _c in range(_nfull)`(_nfull 运行时)。standalone 的同款
`_do_tile` 能跑,差异是 mega 里 `_do_wgrad_tile` 被放在 runtime `if block_index >= wgrad_base:`(scf.if)里,
带 8 个 loop-carried buffer 的 scf.for 嵌在 scf.if 内疑似触发 flydsl tracer 问题(或 nested-def re-trace,
见文件内 `_wave4_do_tile_tn` 的 re-trace 注释)。

**已修掉的坑**:preshuffle `pre_kern` 不能在纯 Python 里 `.launch`(需包进 @flyc.jit)→ 已改成 cached @flyc.jit +
`run_eager_or_capture`;dW1 e5m2 → cbsz/blgp=1(之前错写 0)。

**下一步调试**:让 `_do_wgrad_tile` 的动态 for 不嵌在 scf.if 里(或规避 nested-def re-trace)。候选:
(a) 把 wgrad tile 的 `for _c` 主体也抽成模块级 helper(像 `_wgrad_ssa_chunk`,但外层 for 是动态的——需确认能否);
(b) 换 dispatch 结构让 wgrad tile 在 kernel 顶层调用(用 mask/no-op 处理非 wgrad block);
(c) 看 flydsl 是否有显式 scf.for API 供 module-level 用。
`_STEP3_MEGA=False` 保持生产安全。

## 16. tracer 修复 + mega 路径首次 launch OK(2026-07-25 晚)

**根因确认**:Python `for _c in range(_nfull)`(带 loop-carried LDS buffer tuple)在 runtime `scf.if` 分支内
(AST 改写后 scf.for ⊂ scf.if)触发 flydsl tracer `UnboundLocalError: '_c'`。standalone 能跑是因为
`_do_tile(pid)` 在 kernel 顶层**无条件**调用,不在 scf.if 里。

**修复(两处,`grouped_gemm_combine_fp8_kernel.py`)**:
1. **动态 K-loop 改 `_emit_for`**:把 `for _c in range(_nfull): _wgrad_ssa_chunk(...)` 换成
   `_emit_for(0, _nfull, 1, _wgrad_k_chunk)`(body 用 `nonlocal` 更新 LDS buffer,loop-carry-free,
   见 `gemm_helper._emit_for` 文档)。不再走 Python `for range(dyn)` AST 改写。
2. **dispatch 展平为 if/elif 链**:wgrad 段不再 `if wgrad: ... else: _phase0_roles()`,改为与
   `dispatch_grouped_gemm_mxfp8` 同型的 flat `if block_index >= wgrad_base: ... elif combine: ... else: gemm/reduce`。

**验证**:
- `_compile(..., wgrad=True)` AST 改写 OK。
- EP8 smoke(`agent/workspace/test_mega_wgrad_compile.py`, T=512):**mega combine+wgrad 编译 + 一次 launch OK**,
  `dx=(512,7168) dW1=(32,512,7168) finite=True`。
- `_STEP3_MEGA=True` 已打开;完整 e2e gradcheck(SNR)待跑(8 卡 torchrun 首次 JIT 较慢,需 pre-warm 或更长 timeout)。

**下一步**:e2e dW1/dx/dW2 SNR vs bf16 → 若 PASS 则 CU 配比调优( combine/reduce/dgrad/wgrad 四角色)测 e2e perf。

## 17. 干净机 n02-33 对比(flydsl 0.2.4, EP8 T=8192, 2026-07-26)

**环境**:job 22831 `smci355-ccs-aus-n02-33`, 新容器 `xiaoming-dev`(rocm/primus:v26.3),
flydsl `0.1.1.dev409` → **`pip install flydsl==0.2.4`**, 8× MI355X 空闲。

**对比脚本**:`agent/workspace/bench_mega_vs_baseline_bwd.py`(同一 e2e, `_STEP3_MEGA` False/True)。

| 路径 | fp8 fwd+bwd | vs bf16 | dx SNR | dW1 SNR | dW2 SNR | d_topk_w SNR | 判定 |
|------|-------------|---------|--------|---------|---------|--------------|------|
| **baseline fused** (`_STEP3_MEGA=False`) | **9.769 ms** | 1.34× | 22.0 | 19.5 | 19.7 | 23.1 | **PASS** |
| **mega-kernel** (`_STEP3_MEGA=True`) | **9.770 ms** | 1.34× | **6.8** | 17.7 | 19.4 | 19.9 | **FAIL** (dx) |

**结论(修正)**:
- **Correctness**:mega 与 baseline **dx byte-identical**(single-spawn T=8192 I=256/2048: max_abs=0, SNR~180 dB)。
  两者 vs bf16 dx SNR **均为 ~22 dB PASS**。先前 dual-spawn 对比脚本报 mega dx=6.8 FAIL 是**测试方法假阳性**
  (两次独立 `mp.spawn` 之间 symm/epoch 状态不一致),不是 mega-kernel 逻辑 bug。
- **Perf**:mega 与 baseline **几乎相同**(9.770 vs 9.769 ms)—— dW1 尚未藏进 combine 通信,CU 配比未调。
- `_STEP3_MEGA=True` 可继续 perf 调优;对比请用 single-spawn `bench_mega_vs_baseline_bwd.py`(已修)。

## 附:相关文件
- `primus_turbo/flydsl/mega/fp8/grouped_gemm_combine_fp8_kernel.py` — 现融合三角色 kernel(拆分源)。
- `primus_turbo/pytorch/kernels/mega_moe/mega_moe_backward_fp8_impl.py` — 反向编排(加 overlap)。
- `primus_turbo/flydsl/mega/fp8/gemm_mxfp8_tile.py` — `gemm_mxfp8_nt_tile`(GEMM role 本体)。
- `primus_turbo/flydsl/mega/fp8/gemm_helper.py` — `StoreCQuantMxfp8CShuffle`(fp8 epilogue)。
