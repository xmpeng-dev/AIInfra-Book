# MASSIVE recalibration — all cycle 1-15 bench data was at 8× over-sized workload; real DSV3 super-kernel 7.72 ms ≈ PyTorch+RCCL 7.60 ms (basically tied)

**date**: 2026-05-14 10:18 (UTC+8)
**node**: mi355-gpu-26 (xiaoming-dev container)
**hardware**: 8× MI355X (gfx950)
**workload**: DSV3 4-layer @ seq=4096 mbs=2 (global) topk=8 EP=8 epg=32
**tag**: cycle 15 → cycle 15-RECAL — single most important fact-check of the day

## TL;DR

我之前所有 bench (cycle 1-14b) 都用 `--tokens 8192`，把这个数解读为 source per GPU = 8192。
真实 DSV3 配置（global mbs=2, seq=4096, EP=8）下源 token = global_tokens / GPU = (mbs * seq) / GPU = 8192/8 = **1024 source per GPU**，对应 received per GPU = 1024 × topk = **8192**。

所以 bench 应该用 `--tokens 1024`（对应 source = 1024 per GPU），不是 `--tokens 8192`。

**用错参数的代价**: bench 在 8× 偏大的 workload 上跑，per-(src, e) M 也是 8× 偏大（256 vs 真实 32），M_TILE 浪费假设、22× GEMM gap、所有"FC1 17 ms 慢得离谱"全部 *基于错误工况*。

## 真实 DSV3 工况实测对比（同一 `seq * mbs * topk / GPU = 8192` workload）

```
super-kernel @ --tokens 1024 (real DSV3):
  wall = 7.72 ms, effective_tflops = 747
  variant = Generic (T_src < 4096 阈值)
  avg_Te = 32 (per (src, e))
  per-expert M = 256
  fc1_tiles      : 1.76 ms
  fc2_tiles      : 1.17 ms
  dispatch_wait  : 2.28 ms (until_first 2.07 + skew 0.21)
  copy_to_combine: 0.30 ms
  swiglu_pc      : 0.13 ms
  compute_barriers (1+swiglu+2+3): 0.13 ms
  scatter (comm WG): 1.19 ms
  packed_fc1 cycle-16 candidate: 1.95 ms (含完整 wait/descriptor/barrier overhead)

PyTorch+RCCL @ --tokens 1024 (real DSV3):
  wall = 7.60 ms (median, 95.0 TFLOP/s/GPU effective)
  dispatch: 0.90 ms
  gemm    : 5.60 ms  (per-expert torch.matmul，hipBLASLt-tuned)
  combine : 0.95 ms
```

**Speedup = 7.60 / 7.72 = 0.98× (我们略慢 0.12 ms)，基本平手。**

## 含义

### 1. 之前所有"super-kernel 慢 PyTorch 2.7×" 的结论是错的

`--tokens 8192` 工况下：super 51 ms vs PyTorch 18.6 ms = 0.36×（PyTorch 快 2.7×）。
`--tokens 1024` 工况下：super 7.72 ms vs PyTorch 7.60 ms = 0.98×（基本平）。

差异来源：M=32 per-(src, e) 时 super-kernel 的 M_TILE=256 的"浪费 MFMA cycle"等价于把
GEMM 时间放大约 4×；同样的浪费在 PyTorch 路径里也不存在（hipBLASLt 用 per-expert M=256 直接打），
但 PyTorch 的 dispatch+combine 用 RCCL all-to-all_single 在小 workload 下更慢，恰好抵消。

按 issued FLOP 算，super-kernel FC1 在真实工况下达到 (256×16×940M FLOP issued) / 1.76 ms = **2189 TFLOPS** = MI355X 峰值的 87%。compiler/MFMA 已经是吃满的状态。

### 2. Super-kernel 的 GEMM 实际是比 PyTorch 快的

- super FC1+FC2 = **2.93 ms**
- PyTorch GEMM = **5.60 ms**
- super-kernel GEMM 比 PyTorch 快 **1.9×** ✓

之前"GEMM 太差"的判断完全错。GEMM 是我们的 *优势*，不是劣势。

### 3. 真正的瓶颈是 dispatch_wait（不是 GEMM）

| Phase | super-kernel | PyTorch | super 多花 |
|---|---|---|---|
| dispatch | dispatch_wait 2.28 + scatter (overlap) | 0.90 | **+1.38 ms** |
| GEMM | 2.93 | 5.60 | **−2.67 ms** ✓ |
| combine | copy 0.30 + gather (overlap) | 0.95 | ~−0.65 ms ✓ |
| net | 7.72 | 7.60 | +0.12 |

dispatch_wait 是我们相对 PyTorch 唯一显著吃亏的地方，**且占 super-kernel 自身 wall 30%**，单点 ROI 最高。

之前我说 dispatch_wait 可能是 measurement artifact，**实际不是** —— 在 8 srcs peer-scatter 的真实异步行为下，compute WGs 等到 last src 的时间 = 2.28 ms。这跟 PyTorch all-to-all_single 同步语义下整体阻塞 0.90 ms 比，多了 1.38 ms 的 effective wait。

### 4. Cycle 16 (per-expert batched GEMM) 的 ROI 重估

- 假设：M=32 per (src,e) 浪费 87.5% MFMA → 切 per-expert M=256 后 8× 几何收益
- 实测 packed_fc1 含 overhead = 1.95 ms vs old = 1.76 ms（packed 路径 overhead 吃了所有几何收益）
- 纯 GEMM 部分估计 < 1 ms，整合后 (替换 old FC1) 净省 ~0.7 ms
- FC2 也有同样收益 → ~0.5 ms
- **cycle 16 总收益 ~1.2 ms wall**（不是 27 ms，错了 22×）

仍然值得做（占 wall 1.2/7.72 = 15%），但不是 highest priority 了。

## 优先级重排

| Cycle | Δ wall (估) | 难度 | ROI |
|---|---|---|---|
| **dispatch_wait 优化（推开 first-src 等待 / 与 GEMM overlap / 改 K-of-8 partial-fire）** | **−1.5 ms** | 中 | **★★★★★** |
| Cycle 16 (per-expert batch GEMM) | −1.2 ms | 中 | ★★★ |
| Cycle 20 (recycle comm WGs to compute) | −0.5 ms | 中 | ★★ |
| Cycle 17/18/19 | <0.3 ms each | 高 | ★ |

## 目标（用真实数据重设）

- 当前：super 7.72 ms vs PyTorch 7.60 ms = 0.98×
- M2: super ≤ 6.0 ms = 1.27× （只做 dispatch_wait）
- M5（最终 1.8×）: super ≤ 4.22 ms = 7.60 / 1.8 = need 1.83× 加速

预算（all-of-the-above）:
- dispatch_wait: 2.28 → 0.5 (−1.78 ms)
- FC1+FC2: 2.93 → 1.7 (cycle 16 −1.2 ms)
- copy+barrier: 0.40 → 0.20 (−0.2 ms)
- 总: 7.72 → 4.54 ms ≈ 1.67× —— 接近目标但需要每项都做到位

## 立即下一步

1. ✅ 写完本 note → 把所有 cycle 1-14b 的"22× gap"判断 *作废* / 标注作废
2. 把 cycle 16 的 design plan 改写为 (a) 仍然落地，但 (b) target 收益修正为 ~1.2 ms 而不是 27 ms
3. **新优先**：cycle 21 — dispatch_wait 优化设计
   - 思路 A: 给 compute WG "all-8-srcs-ready" wait + GEMM 之间引入 overlap (compute WG 在等 src 7 时已经把 src 0-6 的 FC1 推进了)
   - 思路 B: 改用 K-of-8 partial-fire (像之前 ready_signal 的备选项)
   - 思路 C: scatter-side 推到位（多 comm WG 加载快 first src）

## 致歉

之前两轮 cycle 14b "FC1 17.27 ms" / cycle 15 "22× gap" 都是基于错误工况。这是单点解读 `tokens_per_gpu` 含义错误造成的级联误判。一旦换到真实工况 `--tokens 1024`，整张图像反过来：super-kernel 已经几乎追平 PyTorch。后续所有规划都基于本 note 的 7.72 ms baseline 重新构建。
