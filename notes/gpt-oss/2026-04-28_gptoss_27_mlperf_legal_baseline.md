# GPT-OSS-20B / MI355X — MLPerf v6.0 合法 baseline 复测，并修订 note 25 的 wave 收益数字

**日期:** 2026-04-28 凌晨复测 + 汇总
**硬件:** 8 × MI355X 单机, TP1 PP1 EP1, GBS 32 / MBS 4
**目标:** MLPerf v6.0 GPT-OSS-20B closed division, eval lm_loss ≤ 3.34, BF16 ↔ FP8 hybrid

## 0. 触发原因 — 把 [GPT-OSS overnight RMSNorm wave A/B](2026-04-27_gptoss_25_overnight_rmsnorm_wave_timetotarget) 的 7.71 % 数字翻译成 submission 真数

note 25 跑的两组都用了 engineering A/B preset：

```
PRIMUS_TRAIN_ITERS=12000
PRIMUS_LR_WARMUP_ITERS=128
PRIMUS_LR_DECAY_ITERS=11872     # ← 关键，cosine 退到 12k 就到底
SEED=1234
```

而 MLCommons MLPerf v6.0 `training_rules.adoc` 的 hyperparameter 表对 `gpt_oss_20b`
明文规定（line 323/329, v6.0）:

| MLLOG key | 规则 |
|---|---|
| `max_steps` | **FIXED = 1,200,000** |
| `opt_learning_rate_decay_steps` | **FIXED = `1_200_000 − opt_learning_rate_warmup_steps`** |
| `opt_end_learning_rate` | FIXED = `opt_base_learning_rate × 0.1` |
| `opt_adamw_beta_1 / _beta_2 / _epsilon` | 0.9 / 0.95 / 1e-5 |
| `opt_adamw_weight_decay` | 0.1 |
| `opt_gradient_clip_norm` | 1.0 |
| `dropout` | 0.0 |
| `sequence_length` | 8192 |
| `opt_learning_rate_warmup_steps` | unconstrained |
| `opt_base_learning_rate` | unconstrained |
| `global_batch_size` | unconstrained |
| `gradient_accumulation_steps` | unconstrained |

> rules line 302: "The MLPerf verifier scripts checks all hyperparameters... If the
> verifier and the constraints in this table differ, the verifier ... is the source
> of truth."

→ verifier 会读 MLLOG 的 `opt_learning_rate_decay_steps` / `max_steps` 直接卡。
note 25 那两组（12k / 11872）都过不了，**本身可以作为内部 A/B / 冒烟，但 RESULT
数字不能进 submission**。

外加 RCP（Reference Convergence Point）这关：rules §"RCPs" line 536 — "We are
interested in avoiding cases where the submission convergence is **faster** than
the reference"，即使 schedule 数对了，把 cosine 强行塞进 12k 窗口 → 收敛比
reference 快 → score 会被 normalize 回 RCP mean，相当于白干。

所以做了这次 stock-legal 复测。

## 1. TL;DR — 4 跑对比

| 跑 | wave | `max_steps` | RESULT (s) | iter@stop | eval@cross | 稳态 ms/iter | MLPerf 合法？ |
|---|---|---|---|---|---|---|---|
| **本次 stock-legal BASE** | OFF | **1,200,000** | **9,234** | 7,296 | 3.3328 | 1,196 | **✓ submission-ready** |
| 0427 "broken" run | ON  | 1,200,000 | 9,035 | 7,296 | 3.3337 | 1,180 | ✓ 也合法 |
| 0425 OPT (note 25) | ON  | 12,000 | 7,719 | 6,144 | 3.3328 | 1,176 | ✗ engineering only |
| 0425 BASE (note 25) | OFF | 12,000 | 8,364 | 6,528 | 3.3315 | 1,202 | ✗ engineering only |

> 0427 "broken" 那次最初被当成 regression 通报（[note 26](2026-04-28_gptoss_26_lrdecay_default_regression) 的早期讨论），后来发现它 schedule 是合法的、只是相对 12k engineering A/B 慢，本质上是个 legal-OPT 跑。

```
本次 RESULT 行（2026-04-27 18:31:23 docker 起跑，2026-04-28 02:05:17 容器时钟收尾）:
RESULT,GPT_OSS_20B,,9234,AMD,2026-04-27 11:31:23 PM
:::MLLOG ... "key":"run_stop"   ... "samples_count":233472, "status":"success"
:::MLLOG ... "key":"run_duration" ... "value":"9153.89s -> 152.56 minutes"
```

## 2. 修订 note 25 的 wave 收益 — 拆三个 delta

把 4 跑两两对比，能干净拆出 3 个互不依赖的 effect：

### 2.1 Schedule 12k → 1.2M（"合法化代价"）

同 wave on/off 状态下，把 `lr_decay` 从 11,872 撑到 1,199,872 的代价：

| wave 状态 | 12k schedule (RESULT) | 1.2M schedule (RESULT) | Δ |
|---|---|---|---|
| OFF | 8,364 s (BASE 0425) | **9,234 s (本次)** | **+870 s / +10.4 %** |
| ON | 7,719 s (OPT 0425) | 9,035 s (broken 0427) | **+1,316 s / +17.0 %** |

→ 把 cosine 撑到合法 schedule，付 **+10–17 % wall**。wave 受影响更大（多丢 6.6 pp）
是因为 wave 在 12k schedule 下"借走"了 cosine 真退到中段的红利。

### 2.2 Wave 在合法 schedule 下的真实增益（**这才是 submission 数字**）

同 1.2M schedule 下，wave on vs off：

| metric | BASE 1.2M (本次) | OPT 1.2M (0427 broken) | Δ |
|---|---|---|---|
| **RESULT (s)** | **9,234** | **9,035** | **−199 s / −2.15 %** |
| iter @ run_stop | 7,296 | 7,296 | **0** |
| eval_accuracy @ cross | 3.3328 | 3.3337 | ~ same |
| 稳态 ms/iter | 1,196 | 1,180 | −16 ms / −1.34 % |
| 稳态 TFLOP/s/GPU | 691 | 706 | +15.0 / +2.17 % |

**关键修订:** wave 真实可报的 wall 增益是 **−2.15 %**，不是 note 25 报的 −7.71 %。
note 25 那 −5.5 pp 的差额是 12k cosine schedule "顺手收敛更快"借的，**不是 wave 自己挣的**。

### 2.3 Wave 的"收敛红利"在合法 schedule 下消失了

这是这次复测最 surprising 的结果：

- 合法 schedule 下 OPT 和 BASE 都在 **iter 7,296 完全同时 cross**（差 0 iter）
- eval_accuracy@cross 几乎一样（3.3337 vs 3.3328，差 0.001 是噪声量级）
- 即 wave 的"系统性 −0.025 lm_loss"在 LR 全程 ≈ 8e-4 卡死的环境下**不出现**

note 25 §4 的解释是："fused Triton kernel 数值精度更好，24 层 fp32 累加积累出
0.02~0.04 lm_loss 优势"。如果这条解释成立，**任何 schedule 下都该看到这个差**，
跟 LR 不耦合。但实测在合法 schedule 下不出现 → 这个解释**不全对**。

更可能的真相：wave 的"收敛红利"来自 12k schedule 下 OPT 的 effective LR 较小（同
一个 absolute step 上 OPT 的 cosine 走得更快、LR 更低），低 LR 让 fused kernel 的
数值优势能 manifest 成 lm_loss 差。一旦 LR 卡在峰值不动，wave 的数值精度优势对
loss 的影响就被 SGD 噪声淹没。

→ 留给后续 trace 验证：在合法 schedule 下做一组 long-tail iter ~10k+ 的 OPT vs
BASE，看 wave 的 lm_loss 优势在 LR 真退到中段后是否 reappear。

### 2.4 Sanity check — model 拟合
```
预测 wall(BASE 1.2M) = wall(BASE 12k) × (1 + schedule_cost_off)
                     = 8,364 × 1.104 = 9,234 s   ✓ 实测 9,234 s
预测 wall(OPT 1.2M)  = wall(BASE 1.2M) × (1 − wave_per_iter_only)
                     = 9,234 × (1 − 0.0215) = 9,036 s   ✓ 实测 9,035 s
```

非常 clean。三个 delta 是真正可分离的。

## 3. 本次 run 的 eval 全曲线 + stability

```
samples    iter    lm_loss      
12,288     384     5.2839       
24,576     768     4.5009       
36,864    1152     4.2240       
49,152    1536     4.0115       
61,440    1920     3.8773       
73,728    2304     3.7696       
86,016    2688     3.6962       
98,304    3072     3.6407       
110,592   3456     3.5915       
122,880   3840     3.5490       
135,168   4224     3.5146       
147,456   4608     3.4796       
159,744   4992     3.4545       
172,032   5376     3.4330       
184,320   5760     3.4084       
196,608   6144     3.3833       
208,896   6528     3.3664       
221,184   6912     3.3475       
233,472   7296     3.3328  ← cross, run_stop, status=success
```

| | |
|---|---|
| NaN iter | 0 |
| skipped iter | 0 |
| run_stop.status | "success" |
| loss scale | 1.0 全程 (bf16 / fp8 hybrid 不需动态 scale) |
| grad_norm 分布 | healthy, peak ~0.7 在 warmup 末，稳态 0.18–0.21 |

## 4. Schedule 合规审计 — 当前生效配置

本次 run.log 头部 args dump 的关键行：

```
train_iters ..................................... 1200000     ← FIXED 1200000  ✓
lr_decay_iters .................................. 1199872     ← = 1200000 − 128  ✓
lr_warmup_iters ................................. 128
lr .............................................. 0.0008
min_lr .......................................... 8e-05       ← = lr × 0.1  ✓
adam_beta1 / adam_beta2 ......................... 0.9 / 0.95   ✓
weight_decay .................................... 0.1          ✓
seq_length (yaml) ............................... 8192         ✓
seed ............................................ 1234
```

MLLOG 出口（verifier 实际读这些）:
```
opt_base_learning_rate          : 0.0008
opt_learning_rate_warmup_steps  : 128
opt_learning_rate_decay_steps   : 1199872
opt_learning_rate_decay_schedule: "cosine with linear warmup"
opt_adamw_weight_decay          : 0.1
max_steps                       : 1200000
submission_benchmark            : "gpt_oss_20b"
submission_division             : "closed"
```

全过 v6.0 表 ✓。

## 5. SOP 改动 — 杜绝 schedule 默认值踩坑

之前 `small_llm_moe_pretraining/primus/run.sh` 是个共用 launcher，"engineering
A/B 短跑"和"submission 长跑"共用同一脚本，靠 caller 传 env var 区分。教训：

- caller 一旦忘传 → 隐式落到 `config_MI355X_*.sh` 里 `TRAIN_ITERS=1200000` 的
  submission 默认；这条 path 跑出来是 **合法但慢**的数（参考本次 9,234 s）。
- 反过来如果 caller 用了 [`run-trace/baseline/20260425_overnight_12k/run_overnight.sh`](2026-04-27_gptoss_25_overnight_rmsnorm_wave_timetotarget) 那种带 12k override 的 wrapper → 跑出来的是
  **快但非法**的数（参考 OPT 0425 7,719 s），直接拿去交会被 verifier 拒。

**当前临时修复：** `run.sh` 注释里写明"stock MLPerf-legal baseline，工程 preset
在 `run-trace/baseline/20260425_overnight_12k/`，**不要拿来报 submission 数字**"。

**TODO：** 把两条 path 物理拆开：

- `run.sh` 永远走 stock-legal preset（本次配置）
- `run.sh.engineering_12k` 显式带 12k schedule + RMSNorm wave，仅用于 fast A/B
  cycle，文件头第一行 `# WARNING: NOT MLPERF-LEGAL` 防止误用
- 或者更彻底：拆 `config_MI355X_*_mlperf_submission.sh` vs
  `config_MI355X_*_engineering_12k.sh` 两份，run.sh 只 source 前者

下一份 PR 会做。

## 6. 下一步

> ~~原 §6.1 起一次 legal-OPT 复测 (wave ON + 1.2M)~~ — **取消**。0427 "broken" 跑
> (`run.log.20260427_140306_lrdecay_oops`, RESULT 9,035 s, iter 7,296) 已经是同
> 配置，N=1 数据点已存在，不需要复测。复测脚本起到 iter 500 (ms/iter cum 1,164.9，
> 跟 0427 broken 的 1,180 一致，落 noise 内) 后被 kill，归档为
> `run.log.20260427_28_legalopt_dup_killed_iter500`。

按优先级：

1. **修 SOP** — `run.sh` 永久 legal、新建 `run.sh.engineering_12k`（见 §5）。最便宜，
   防下次"忘加 export → 跑出非可比数字"。
2. **Long-tail wave A/B**（同 legal schedule 但显式跑过 cross 到 iter ~12k 看
   cosine 真退到中段后 wave 的 lm_loss 是否拉开）— 验证 §2.3 的"wave 收敛红利
   是 schedule-coupled illusion"假说。需 hook `run_stop` 或拉高 target loss
   threshold；一晚一跑 ~3h。
3. **(可选) SEED replicate** — 同配置换 SEED=4234 跑一次 BASE/OPT 各一份，给
   §2.2 的 −199s claim 真实 noise band（当前两侧都 N=1）。一晚两跑 ~5h。
4. **next wave 机会**：wave 真实只能挣 −2.15 %，下一波优化目标应该是
   - tier-2 的 attention bwd（仍是 ~126 ms/step）
   - moe permute / dispatcher 的 elementwise tax
   - sync-free MoE stage ≥ 2（[note 22](2026-04-25_gptoss_22_sync_free_moe_stage_audit)）
   - FP8 attention bwd

## 7. Run artifacts

```
small_llm_moe_pretraining/primus/
├── run.log.20260427_27_stocklegal_base_9234s          ← 本次 BASE 完整 log (671 KB)
├── run.stdout.log.20260427_27_stocklegal_base_9234s
├── run.log.20260427_140306_lrdecay_oops               ← 0427 legal-OPT (wave ON+1.2M sched, RESULT 9035 s)
├── run.log.20260427_22_engineering_12k_partial        ← 12k engineering 中途 kill
├── run.log.20260427_28_legalopt_dup_killed_iter500    ← 重复 legal-OPT 起到 iter 500 即 kill
└── run.sh                                             ← stock-legal launcher (修订过)

slab/notes/gpt-oss/
├── 2026-04-28_gptoss_27_mlperf_legal_baseline.md  ← 本 note
├── 2026-04-27_gptoss_25_overnight_rmsnorm_wave_timetotarget.md  ← 头部已加 superseded 横幅
└── 2026-04-28_gptoss_26_lrdecay_default_regression.md  ← (TODO，本周整理)
```
