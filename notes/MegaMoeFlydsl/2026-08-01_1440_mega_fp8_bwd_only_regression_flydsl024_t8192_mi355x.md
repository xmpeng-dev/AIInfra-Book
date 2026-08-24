# Mega MoE fp8 — 换机 + flydsl 0.2.4 后各阶段性能回归（T=8192）@ MI355X

> **用途**: 换到新节点、把容器里的 flydsl 从 `0.1.1.dev409` 升到 `0.2.4` 之后，用 `bench_mega_moe_bwd_only.py`
> 回归 fwd+bwd / fwd-only / bwd-only 三段的性能，并与 7/24–7/30 的历史读数对齐口径。
> **Where**: `smci355-ccs-aus-n04-33`（MI355X / gfx950 ×8，8 卡全空闲），容器 `xiaoming-dev`（`rocm/primus:v26.3`），job 23835。
> **配置**: DeepSeek-V3，EP8，T=8192，H=7168，I=2048，E=256，topk=8；warmup=8，iters=25，跨 8 rank 取最慢。
> **代码**: `feat/xiaompen/mega_moe_flydsl_mxfp8` @ `60962853`，**working tree 带 7/30 未提交改动**
> （forward 拆成 `l1_impl → colwise_requant → l2_impl`，6 文件 +151/−56）。

---

## 0. 环境变更：flydsl 升级

容器里原来是 **easy_install 装的 egg `0.1.1.dev409`**，它在 `easy-install.pth` 里注册，会**盖住** pip 装的版本，
所以单纯 `pip install` 无效，必须先摘掉 egg：

```bash
SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
pip uninstall -y flydsl
grep -v flydsl $SP/easy-install.pth > /tmp/ei.pth && mv /tmp/ei.pth $SP/easy-install.pth
rm -rf $SP/flydsl-*.egg
pip install flydsl==0.2.4
rm -rf ~/.flydsl            # 清 JIT 盘缓存，强制重编 + 重 autotune
```

升级后 `import flydsl` 落在 `/opt/venv/lib/python3.12/site-packages/flydsl/`（不再是 egg），`pip show` = 0.2.4。
仓库版 `primus_turbo` 靠 `PYTHONPATH=/perf_apps/xiaoming/MegaMoE` 覆盖 `/opt/venv` 里的安装版（后者不含 `flydsl.mega`）。

---

## 1. 主回归：`bench_mega_moe_bwd_only.py`（两次独立 run）

两次 run 差 ≤0.06 ms，噪声很小，下表取两次均值。

### load_balanced

| | fwd+bwd | fwd-only | bwd-only |
|---|---|---|---|
| **fp8** | **14.64 ms** | **5.52 ms** | **9.08 ms** |
| bf16 | 20.54 ms | 7.30 ms | 13.93 ms |
| bf16/fp8 | **1.40×** | 1.32× | **1.53×** |

单次读数：run1 = 14.663 / 5.537 / 9.061，run2 = 14.608 / 5.509 / 9.099（bf16: 20.475→20.612 / 7.307→7.283 / 13.895→13.970）。

### round_robin

| | fwd+bwd | fwd-only | bwd-only |
|---|---|---|---|
| **fp8** | **13.81 ms** | **5.24 ms** | **8.50 ms** |
| bf16 | 19.79 ms | 7.00 ms | 13.41 ms |
| bf16/fp8 | **1.43×** | 1.34× | **1.58×** |

单次读数：run1 = 13.790 / 5.265 / 8.523，run2 = 13.837 / 5.215 / 8.480。

**cross-check**: fp8 `fwd-only + bwd-only` = 14.608（LB）/ 13.695（RR），与 `fwd+bwd` 14.608 / 13.837 基本闭合；
bf16 两段和比 fwd+bwd 大 ~0.6 ms（bf16 反向里有一段和前向重叠，历史一致，非本轮问题）。

**跨脚本一致性**：`bench_mega_moe_fused_fp8_bwd.py` 三次读到 fp8 fwd+bwd = 14.624 / 14.633 / 14.688 ms，
与 `bwd_only` 的 14.61–14.66 完全对齐——7/30 那次"两个脚本口径打架"的问题不再出现。

---

## 2. 前向分阶段拆解（`bench_fwd_breakdown_fp8.py`, load_balanced）

| stage | ms | 占 FULL_no_grad |
|---|---|---|
| L1_dispatch_fc1 | 2.468 | 52.3% |
| SwiGLU+quant_fused | 0.260 | 5.5% |
| L2_combine_xfp8 | 1.905 | 40.3% |
| w2_prep_cached | 0.006 | — |
| **isolated sum** | **4.633** | — |
| **FULL_no_grad**（纯 forward） | **4.724** | op overhead +0.091 |
| **FULL_autograd**（训练 forward） | **5.366** | autograd prep **+0.642** |

参考：`SwiGLU_bf16 + quant_rowwise` 不融合是 0.435 ms，融合后 0.260 ms（−0.175 ms，融合收益仍在）。

**口径提醒**（7/30 踩过的坑，继续沿用）：`bwd_only` 表里的 **fwd-only 列 = `FULL_autograd`（5.37–5.52 ms）**，
含 `save_for_backward` 的 meta + clone + colwise requant，**不是**纯 forward 的 4.72 ms。对比性能务必分开报这两个数。

---

## 3. 与历史读数对比

| 日期 | 节点 / flydsl | fp8 fwd+bwd | fwd-only | bwd-only |
|---|---|---|---|---|
| 2026-07-24 | n01-21 / 容器原 build | 14.71 | — | — |
| 2026-07-28（dW1/dW2 优化后，历史最好） | n05-29 / 容器原 build | **14.117** | — | **8.734** |
| 2026-07-30（requant 位置修回 L1 后） | n04-25 / pip 0.2.4 | 14.49–14.62 | 5.44 | 9.29 |
| **2026-08-01（本次）** | **n04-33 / pip 0.2.4** | **14.64** | **5.52** | **9.08** |

- **本次 ≈ 7/30**，同代码同口径一致 → **不是新引入的代码回归**，7/30 之后没有掉性能。
- 对 7/28 的 **14.117 有 +0.52 ms 差距**（bwd-only +0.35 ms）。历史已记录过：**pip 装的 flydsl 0.2.4 与 campaign 原容器里那个
  0.2.4 build 不是同一份**，fp8 FlyDSL kernel 会慢（当时 dW2 GEMM 1740 vs 2327 TF），bf16/Triton 不受影响。
  本次 bf16 侧 20.54 ms 与历史 19.97–20.75 同区间，也吻合"只有 fp8 腿受影响"这个特征。
- **未做同环境 A/B**（升级前的 0.1.1.dev409 已被覆盖、缓存已清），所以 +0.52 ms 归因到环境**尚未证实**，只是与已知现象一致。
  要证实需在同一容器里 0.2.4 ↔ 原 build 来回切换重测。

---

## 4. ⚠️ 发现：dx 梯度 SNR 在 T=8192 下不稳定（非确定性）

`bench_mega_moe_fused_fp8_bwd.py` 同一份代码、同一 routing seed，连跑四次：

| run | T | dx | d_topk_w | dW1 | dW2 | gate(≥15dB) |
|---|---|---|---|---|---|---|
| 1 | 8192 | **14.6** | 22.9 | 19.5 | 19.7 | **FAIL** |
| 2 | 8192 | **19.3** | 22.9 | 19.5 | 19.7 | PASS |
| 3 | 8192 | **14.3** | 23.0 | 19.5 | 19.7 | **FAIL** |
| 4 | 2048 | 21.9 | 23.1 | 19.5 | 19.7 | PASS |

- **只有 dx 抖**（14.3 ↔ 19.3 dB，跨 5 dB），`d_topk_w / dW1 / dW2` **三次逐位稳定**。
  确定性 kernel 在固定 seed 下不该有 run-to-run 差异 → **dx 那条路径（STEP3 / L1-dgrad combine + reduce）存在非确定性**，
  怀疑是 combine reduce 的原子/写入顺序，或高 token 数下 symm workspace slot 的残留没清干净。
- T=2048 下 dx=21.9 dB 正常 → 与**规模相关**，T=8192 才暴露。
- 历史记录里 dx 一直是 **20–22 dB 稳定 PASS**，所以这是**新暴露的问题**（本轮变量：换机 + flydsl 0.2.4 + 7/30 未提交的 forward 拆分，三者未拆开）。
- 性能数不受影响（三次 fwd+bwd 14.62/14.63/14.69 完全一致），但**这条在合入前必须查清**。

**下一步建议**（按优先级）：
1. 固定其它变量，单独回退 7/30 那笔未提交改动，看 dx 抖动是否消失（最快的三选一排除法）。
2. 若与改动无关，在 combine reduce 后加一次跨 rank barrier / 清 workspace 复测，定位是否 symm 残留。
3. 连跑 10 次统计 dx 分布，确认是双峰（种子无关的竞态）还是连续抖动。

---

## 5. 复现

```bash
# 环境（fresh 容器必做）
docker exec xiaoming-dev bash -lc '
SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
pip uninstall -y flydsl
grep -v flydsl $SP/easy-install.pth > /tmp/ei.pth && mv /tmp/ei.pth $SP/easy-install.pth
rm -rf $SP/flydsl-*.egg && pip install flydsl==0.2.4 && rm -rf ~/.flydsl'

# 主回归（两种 routing，fp8 + bf16）
docker exec xiaoming-dev bash -lc 'cd /perf_apps/xiaoming/MegaMoE && export PYTHONPATH=$PWD
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_bwd_only.py \
  --num-processes 8 --num-tokens 8192 --routing-mode both --only both --warmup 8 --iters 25'

# 前向分阶段
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_fwd_breakdown_fp8.py \
  --num-processes 8 --num-tokens 8192 --warmup 8 --iters 25

# 梯度 SNR（dx 抖动复现，需多跑几次）
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_fused_fp8_bwd.py \
  --num-processes 8 --num-tokens 8192
```
