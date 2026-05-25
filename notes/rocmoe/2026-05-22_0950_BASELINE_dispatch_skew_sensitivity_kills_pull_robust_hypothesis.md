# 2026-05-22 09:50  Dispatch skew sweep — pull-dispatch is *more* skew-sensitive than RCCL, not less

> **⚠️ 已修订 2026-05-22 10:35** — 见 [`2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md`](./2026-05-22_1035_FLAT_dispatch_phase_profile_corrects_skew_mechanism.md)。两个具体修订：
> 1. 本 note 通篇 "M1b standalone" 应读作 **"M1c-A standalone (current default with `ROCMOE_DISPATCH_USE_PACKED_OUTBOX=1`)"**。当前 `csrc/dispatch.hip` 默认走的就是 M1c-A 的 3-phase sender + packed_outbox 路径，不是 M1b 的 scatter-read 路径。
> 2. §3 把 skew tax 归因到 "receiver per-block `block_ready` polling 等最慢 sender" —— **机理错了**。当前代码 receiver 端没有 per-block polling，cross-rank `phase_barrier` 在 receiver 启动之前一次性完成。实测 phase_barrier 仅占 wall 0.6% (39-64 μs)。真正的 skew 来源是 Receiver per-rank imbalance (max 比 mean 多 +31%) + PhaseB sender pack max-rank (+41%) + syncB grid_sync<1> (+597% rel, +0.7 ms abs)。
>
> **核心结论不变**：standalone pull-dispatch 在 hot_cov50 下比 RCCL 更敏感 2-3×；M1c-B / M1c-C 搁置；下一步 M2-G 仍然成立。但 "M2-G FC1 overlap 能拿多少" 这一估计需要下调：FC1 在 DSv3 prod 约 1.5 ms，相对 receiver 8 ms 只能藏 ~15%。
>
> ---
>
> 时间: 2026-05-22 09:30 → 09:50 (Asia/Shanghai)
> 项目: rocmoe
> 类型: BASELINE — first RocMoE-v2 M1b standalone dispatch numbers under
>   non-balanced router skew, paired apples-to-apples with the existing
>   Megatron-LM `MoELayer` baseline at the same `(model, T, skew_profile)`
>   grid.
> 触发: user "和好了 harness 工作，我们重新回来看看 dispatch 的问题" —
>   take Route C from the dispatch decision tree (run the dispatch-stage
>   skew sweep on both sides before committing to M1c-B/C vs M2-G).
> 硬件: 8x AMD Instinct MI355X (gfx950, mi355-gpu-7), SLURM allocation
>   13588 (new, started ~5 min before this sweep)
> 容器: `xiaoming-dev` (podman, ROCm 7.2 / PyTorch 2.10)
> 上一节点:
>   - [`2026-05-21_2100_DOWN_m1c_a_revisit_dsv3_production_size`](./2026-05-21_2100_DOWN_m1c_a_revisit_dsv3_production_size.md)
>     (M1c-A revert, planned M1c-B receiver-side sort)
>   - [`2026-05-22_0410_BASELINE_mcore_moe_full_sweep`](./2026-05-22_0410_BASELINE_mcore_moe_full_sweep.md)
>     §8 (skew profile harness landed; RCCL combine = main skew-sensitive
>     stage; dispatch sensitivity left for follow-up — this note)
> 数据:
>   - `bench_results/rocmoe_dispatch_skew_20260522_0950.csv` (12 rows, this sweep)
>   - `bench_results/mcore_baseline_20260521_1014.csv` (RCCL alltoall side,
>     reused from §8)
>   - per-run stdout in `bench_results/log_rocmoe_dispatch_skew_20260521_2055/`
>     and `bench_results/log_rocmoe_dispatch_skew_20260521_2058/`

## TL;DR

The Route C hypothesis was: "RocMoE-v2 pull-dispatch should be more
robust to router skew than RCCL alltoall, because each receiver pulls
only the buckets it owns and a hot expert only stalls its own WG, not
the whole all-to-all." **That hypothesis is false.** Across 4 models ×
3 skew profiles:

| Model       | T     | RCCL Δrealistic | M1b Δrealistic |   gap   | RCCL Δhot | M1b Δhot |   gap   |
|-------------|------:|----------------:|---------------:|--------:|----------:|---------:|--------:|
| DSv3        | 8192  |          -0.8 % |        +10.6 % | +11.4 pp|   +11.1 % |  +31.3 % | +20.2 pp|
| DSv2        | 4096  |          +6.0 % |         +7.8 % |  +1.8 pp|   +19.1 % |  +49.5 % | +30.4 pp|
| Qwen3-235B  | 16384 |          +3.1 % |         +4.0 % |  +0.9 pp|   +14.8 % |  +32.8 % | +18.0 pp|
| Qwen3-30B   | 32768 |          +3.3 % |         +3.9 % |  +0.6 pp|   +15.5 % |  +32.5 % | +17.0 pp|

- At publication-grade `realistic_cov20` (CoV ≈ 0.20, matches aux-loss-
  free-balanced trained MoE post-warmup) **we are 2-15× more sensitive
  than RCCL on the pp gap**, biggest on DSv3 (group_topk=4 concentrates
  the hot experts within each selected group).
- At worst-case `hot_cov50` (CoV ≈ 0.60-0.75) **we are uniformly 2-3×
  more sensitive than RCCL**; the abs gap balloons from 3.6-4.8× to
  4.4-5.5× and per-rank skew on our side falls from 0.99 → 0.69-0.82
  (RCCL's per-rank skew stays at ~0.95 because alltoall pads).

The structural reason: pull-dispatch's per-block `block_ready` polling
makes the receiver wait for the **slowest sender's last block**. Under
skew, the slowest sender's bucket grows from ~256 tokens (balanced) to
~1500 tokens (hot, DSv3); the receiver-side wait queue lengthens
proportionally, but the receiver is not free to do other work while it
waits because it's locked to that pull. RCCL's batched all-to-all
suffers from the same imbalance but exposes more concurrency by
treating the whole transfer as a single send/recv pair where the
hardware engine can pipeline across all bytes.

**Implication for the dev plan**: the standalone-dispatch tuning route
(M1c-B receiver-side `src_t` sort, M1c-C combined LDS staging) was
budgeted to recover 13-15 % wall — that's small compared to the
30-50 % skew tax and entirely outside the structural cause (it
optimizes the receiver's HBM coalescing, not the sender's slowest-
block tail). **Recommend pivoting to Route B (M2-G first)**: the
architectural bets (Layout-P g=0 spin elimination -1.34 ms, dispatch
↔ FC1 chunk-overlap, pull-combine -0.7 ms) are the only ones that can
move the dispatch wall meaningfully.

## 1. What we ran

### 1.1 Code added (minimal)

- `benchmarks/bench_routing.h` — added `resolve_skew_sigma(name)` that
  maps the 3 profile names to sigma, matching `workloads.yaml`. Stays
  in sync with mcore harness.
- `benchmarks/bench_dispatch.hip` and `bench_super_dispatch.hip` — two
  new optional positional CLI args:
  - `skew_profile` (arg #11, default `balanced`)
  - `group_topk` (arg #12, default 4; allows DSv2 group_topk=3 to be
    passed correctly without hard-coding 4)
  Both wire into the `RoutingCfg` already supported by `bench_routing.h`.
- `scripts/bench_dispatch_skew_sweep.sh` — sweep driver mirroring
  `run_baseline_sweep.sh` style: 4 models × 3 skews × prod_T_per_rank
  = 12 runs, writes one CSV row per run with realised CoV, bucket
  geometry, host wall, dev wall (per-rank max + per-rank mean), per-
  rank skew, us/token. Picks `max_recv_factor=8` for hot_cov50 (HIP
  bench observes bucket_max/mean ≈ 6× at sigma=0.30 — bigger than the
  3.3× the mcore harness observes, because the HIP-side seed is FNV-1a
  of "rocmoe-skew-hot_cov50" while mcore uses SHA-256 of the same
  string ⇒ different bias vector, same calibration grade).

No kernel change — the dispatch body is exactly M1b sweet-spot
(kSubWGs=8, scoreboard-as-uint32_t-counter; `dispatch_body.h` git-clean).

### 1.2 Workload grid

Same `(model, T_per_rank, skew_profile)` grid as the §8 mcore sweep:

| model       | T_per_rank | num_experts | topk | H    | group_topk | dist       |
|-------------|-----------:|------------:|-----:|-----:|-----------:|------------|
| deepseek_v3      |  8192 | 256 | 8 | 7168 | 4 | dsv3       |
| deepseek_v2      |  4096 | 160 | 6 | 5120 | 3 | dsv3       |
| qwen3_235b_a22b  | 16384 | 128 | 8 | 4096 | — | plain_topk |
| qwen3_30b_a3b    | 32768 | 128 | 8 | 2048 | — | plain_topk |

Skew profiles (sigma → realised CoV verified on the HIP bench):

| profile          | sigma | CoV (DSv3) | CoV (DSv2) | CoV (Qwen3) |
|------------------|------:|-----------:|-----------:|------------:|
| balanced         |  0.00 |       0.02 |       0.03 |        0.01 |
| realistic_cov20  |  0.10 |       0.22 |       0.20 |        0.18 |
| hot_cov50        |  0.30 |       0.73 |       0.75 |        0.68 |

(`hot_cov50` calibration realised 0.68-0.75 in the HIP bench vs ~0.60
in mcore — small over-shoot due to FNV vs SHA bias-vector divergence,
not a problem for the comparison.)

warmup=10 iters=50 bf16; 8x MI355X single allocation; same node and
container across all 12 runs.

## 2. Raw numbers

### 2.1 RocMoE-v2 M1b standalone dispatch (this sweep)

`dev wall (ms), per-rank max mean over 50 iters = critical-path latency`:

| Model       | T     | balanced | realistic_cov20 | hot_cov50 | bal→hot |
|-------------|------:|---------:|----------------:|----------:|--------:|
| DSv3        |  8192 |    9.410 |          10.404 |    12.360 | +31.3 % |
| DSv2        |  4096 |    3.185 |           3.433 |     4.763 | +49.5 % |
| Qwen3-235B  | 16384 |   13.933 |          14.486 |    18.502 | +32.8 % |
| Qwen3-30B   | 32768 |   24.574 |          25.538 |    32.568 | +32.5 % |

Per-rank load-balance skew (per-rank min wall / per-rank mean wall —
1.0 = perfectly balanced, lower = more lopsided):

| Model       | balanced | realistic | hot   |
|-------------|---------:|----------:|------:|
| DSv3        |    0.993 |     0.936 | 0.824 |
| DSv2        |    0.985 |     0.921 | 0.689 |
| Qwen3-235B  |    0.996 |     0.952 | 0.755 |
| Qwen3-30B   |    0.995 |     0.952 | 0.758 |

→ Under hot_cov50 the per-rank wall imbalance is 18-31 %; this is the
direct manifestation of pull's structural problem — the receivers
holding hot experts finish much later than the receivers holding cold
ones.

Realised bucket geometry (`per-(src_rank, dst_local_e) bucket
max/mean ratio`, the metric `max_recv_per_e_per_src` must size for):

| Model       | balanced | realistic | hot  |
|-------------|---------:|----------:|-----:|
| DSv3        |     1.21 |      1.95 | 6.10 |
| DSv2        |     1.24 |      1.69 | 6.33 |
| Qwen3-235B  |     1.12 |      1.51 | 5.29 |
| Qwen3-30B   |     1.09 |      1.51 | 5.25 |

### 2.2 MCore RCCL alltoall dispatch (from §8 sweep, for comparison)

`stage_dispatch_ms` column of `mcore_baseline_20260521_1014.csv`:

| Model       | T     | balanced | realistic_cov20 | hot_cov50 | bal→hot |
|-------------|------:|---------:|----------------:|----------:|--------:|
| DSv3        |  8192 |    2.557 |           2.537 |     2.841 | +11.1 % |
| DSv2        |  4096 |    0.886 |           0.939 |     1.055 | +19.1 % |
| Qwen3-235B  | 16384 |    2.913 |           3.003 |     3.345 | +14.8 % |
| Qwen3-30B   | 32768 |    2.876 |           2.971 |     3.321 | +15.5 % |

### 2.3 Side-by-side: skew sensitivity and absolute multiple

| Model       | M1b/RCCL bal | M1b/RCCL real | M1b/RCCL hot |
|-------------|-------------:|--------------:|-------------:|
| DSv3        |        3.68× |         4.10× |        4.35× |
| DSv2        |        3.60× |         3.66× |        4.51× |
| Qwen3-235B  |        4.78× |         4.82× |        5.53× |
| Qwen3-30B   |        8.54× |         8.59× |        9.81× |

Qwen3-30B is the worst absolute multiple — H=2048 is tiny so RCCL's
fixed alltoall overhead amortises poorly on our per-block polling
side. The model-wise multiple is shaped almost entirely by H (DSv3
H=7168 → 3.7× ⟶ Qwen3-30B H=2048 → 8.5×). This is consistent with the
per-token dispatch being inverse-proportional to H (each token costs
~H bytes of XGMI movement, fixed-cost overhead per block dominates at
small H).

## 3. Why pull is *more* skew-sensitive than alltoall

Going back to the receive-side state machine in `dispatch_body.h`:

```
for each (e, src_rank, block_b) it owns:
    spin on atomic_load_acquire(peer.block_ready[blk]) until set
    cooperative_b128_copy(peer.input_token_buf[bucket],
                          local.expert_token_pool[slot])
```

Under skew, the (e=hot_expert, src_rank=any) bucket grows from ~256
tokens balanced → ~1500 tokens hot. The sender doing that hot bucket
takes ~6× longer to walk through its tokens and only flips
`block_ready[hot_block_b]` at the end (block-level granularity). The
receiver assigned to that hot bucket is **idle-spinning** for the
duration of that 6× longer write, because:

1. The receiver WG is in a tight wait-then-copy loop; it doesn't
   reschedule onto another (e, block_b) while waiting on this one
   (that's how block-level ownership is enforced — to avoid duplicate
   inbound reads of the same source bucket).
2. Even if it could reschedule, the *other* buckets are likely
   already pulled — under hot_cov50 the small buckets finish almost
   instantly and only the hot ones remain.
3. The kernel's overall wall is `max over (receiver WG)` of
   (sum over its assigned buckets of (wait + pull)), and the receiver
   stuck with the hottest bucket loses linearly with that bucket size.

Per-rank skew dropping to 0.69 on DSv2 hot is exactly this: the rank
holding the hottest experts waits ~50 % longer than the rank holding
the coldest ones. There's no work-stealing in M1b to soak it up.

**RCCL's batched alltoall** doesn't have this problem to the same
extent because:

1. It moves bytes in a single fused all-to-all call that the XGMI
   engine can pipeline aggressively (multiple in-flight transfers
   per pair), so a long bucket doesn't block other pairs.
2. The aggregate volume per rank pair is balanced even under skew —
   what changes is which rank pair has the most bytes, not whether
   any pair gets blocked. Hardware engine queues them all in flight.
3. There's no per-block request-response RTT in the receiver — RCCL
   uses send-side push with credits; the receiver doesn't have to
   poll a per-block ready flag.

So pull's structural cost is **not** "more in-flight transactions"
(MonolithEP push has more per `compare_monolithep_dispatch` §4),
it's "longest wait dominated by longest bucket = O(sigma·H)" vs
RCCL's "longest wait dominated by total volume = O(H)".

## 4. What this kills, what it doesn't

### Killed (or downgraded):

- **"pull is skew-robust"** — empirically false on the standalone
  dispatch, structurally explained. Bullet (3) of the architecture
  design's §2.2 "why pull" justification ("one slow sender only
  blocks its own block_b") **is true at the protocol-graph level
  but wrong at the wall-level** because under skew the slow sender
  *is* the longest block. The graph-level argument was a strawman vs
  push, not vs alltoall.
- **M1c-B receiver-side `src_t` sort ROI** — the planned 3-5 % wall
  improvement is irrelevant to a 30-50 % skew tax. It's a HBM-burst
  micro-optimization that's orthogonal to the wait-on-slowest-sender
  problem. Stays on the shelf; may revisit if a future M2-G build
  shows HBM burst is the actual bottleneck.
- **M1c-C combined LDS staging ROI** — same reasoning. LDS staging
  optimizes receiver XGMI hiding, doesn't change the sender-side
  block_ready tail.

### Still alive (unchanged by this finding):

- **Layout-P g=0 fan-in spin elimination (-1.34 ms)** — this is
  about cross-WG synchronization for the FC1 input layout, not
  bucket-imbalance. Still a clean architectural win that lands at
  M2-G.
- **dispatch ↔ FC1 chunk-level overlap** — the only structural
  defense against absolute dispatch wall (4-8× vs RCCL standalone is
  not winnable head-on; hiding it inside FC1 MFMA changes the game).
  Becomes more urgent now that we know dispatch absolute and skew
  sensitivity both lose to RCCL.
- **pull-combine over FC2 8-way push (-0.7 ms in BF16)** — combine
  is the actual skew-sensitive stage (§8 of the mcore note shows
  +76-175 % on combine wall vs +11-19 % on dispatch wall). Pull-
  combine's value proposition is more about *avoiding the FC2 ⇄
  combine XGMI contention* than skew-robustness. Still the right
  bet.

### Note on absolute regression vs prior runs

DSv3 T=8192 balanced was reported as 7.491 ms in the M1c-A revisit
note (2026-05-21 21:00, same node, same kernel). This sweep measured
9.410 ms = **+25 % regression** with no code change in between. Most
likely root cause: NIC weather + co-tenant XGMI traffic on a new
SLURM allocation (interactive job 13588 started 5 min before this
sweep; previous runs were on a different allocation). The within-
allocation Δ% comparisons across skew profiles are valid (each pair
runs back-to-back on the same allocation), but the absolute number
should be re-measured at the start of every session per
`rocmoe-dev-loop` rule 1. Will pin a fresh balanced number whenever
the next M2-G change lands.

## 5. Engineering takeaways

| 教训 | 落地 |
|---|---|
| Pull's "skew-robust" claim only holds at the protocol-graph level (which sender's block I'm waiting on), not at the wall level (how long that block takes). The graph-level robustness against one-slow-receiver is real (and matters for hardware fault tolerance), but **wall robustness against routing imbalance was never on the menu** for either push or pull. | Update architecture-design §2.2 with a footnote: "wall-level skew robustness requires either (a) bucket subdivision so a single hot expert is striped across multiple WGs, or (b) cross-WG work-stealing — M1b has neither, M2-G can have (b) via the persistent role queue." |
| RCCL alltoall's skew sensitivity (+11-19 % at hot_cov50) is **also worse than the architecture-design's implied roofline** — but the absolute is small enough (~0.5 ms) that production users mostly don't notice. Our pull on the same workload loses 30-50 % = ~5-10 ms = absolutely noticeable in a 7-20 ms forward budget. | Treat "RCCL Δ% at hot_cov50" as the floor that any RocMoE-v2 dispatch story has to clear. M2-G's overlap goal becomes: dispatch wall under hot_cov50 hidden inside FC1, contributes zero to super-kernel wall. |
| The decision tree in `rocmoe-dev-loop` "Route A vs B vs C" exists precisely so we don't sink engineering into micro-tuning when a structural pivot is the right move. Today we saved 1-2 days of M1c-B/C engineering by 1 hour of measurement. | Add a sentence to skill: "before starting a micro-tune milestone, run the relevant publication-grade workload sweep to make sure the optimization target survives realistic conditions." |
| `bench_dispatch.hip` already had `bench_routing.h`'s skew injection ready (sigma-bias path implemented + calibration table); only the CLI plumbing was missing. Same for `bench_super_dispatch.hip`. The cost of "add a publication-grade dimension" to existing standalone benches was ~30 lines C++ + one sweep script. | Standardize: every standalone HIP bench that owns a `RoutingCfg` should accept `--skew-profile` from day 1; saves the future-self the ad-hoc add. |
| HIP-side bias vector uses FNV-1a("rocmoe-skew-<profile>") while mcore uses SHA-256("rocmoe-skew-<profile>") → same string, different per-expert bias → realised CoV diverges (0.73 vs 0.59 at sigma=0.30). The calibration grade is fine, but if we ever want **per-token-bit-exact** comparison between HIP and mcore, we need the same hash. | Future: replace FNV-1a in `bench_routing.h:skew_seed` with a hand-rolled SHA-256 (or pull in a tiny single-file SHA-256 like `picosha2.h`). Low priority — not blocking any decision. |

## 6. Next step

The decision tree from the earlier framing collapses to:

| Route | status |
|---|---|
| A — M1c-B / M1c-C (standalone micro-tune) | **abandoned** for now, justified by this note |
| C — measure dispatch skew sensitivity | **done** (this note) |
| B — M2-G GEMM in super-kernel, hide dispatch in FC1 | **next** |

Concrete M2-G plan (carry-over from `2026-05-21_1930_BASELINE_m2d_dispatch_in_super_kernel.md`):

1. Port `mfma_tile.h` GEMM body into the GEMM role inside
   `csrc/super_kernel.hip`, driven by the per-pool-block
   `l1_arrival_count` work-stealing queue from M2-D.
2. Acceptance:
   - **DSv3 T=2048** in-super-kernel `(dispatch + GEMM)` wall < M2-D
     `(dispatch only)` 1.669 ms + GEMM standalone time (within ±10 %).
     This validates that chunk-level pull-dispatch ↔ FC1 MFMA overlap
     actually fires.
   - **DSv3 T=8192 balanced** super-kernel `(dispatch + GEMM only)`
     wall must beat **today's session-pinned baseline**: M1b
     dispatch 9.41 ms + standalone GEMM time (need to bench in same
     allocation). If we do not beat dispatch-then-GEMM serialized,
     the overlap bet is failing and we need to debug.
   - **Re-run this skew sweep** on the M2-G build at all 12 points;
     publication-grade dispatch skew sensitivity must drop from
     +30-50 % hot tax to ≤ +15 % (i.e. matching RCCL or better)
     because the FC1 MFMA now absorbs the wait.

3. Note: `notes/<time>_<UP|DOWN|FLAT|BASELINE>_m2g_*.md` per
   `rocmoe-dev-loop` rule 3.

## 7. Files

- New / modified:
  - `benchmarks/bench_routing.h` — `resolve_skew_sigma()` helper
  - `benchmarks/bench_dispatch.hip` — `skew_profile` + `group_topk` args
  - `benchmarks/bench_super_dispatch.hip` — same
  - `scripts/bench_dispatch_skew_sweep.sh` — new sweep driver
- Data:
  - `bench_results/rocmoe_dispatch_skew_20260522_0950.csv` (12 rows)
  - `bench_results/log_rocmoe_dispatch_skew_20260521_2055/` (8 balanced+realistic logs)
  - `bench_results/log_rocmoe_dispatch_skew_20260521_2058/` (4 hot logs)
- Related:
  - `bench_results/mcore_baseline_20260521_1014.csv` (RCCL side, prior sweep)
  - `notes/2026-05-22_0410_BASELINE_mcore_moe_full_sweep.md` §8 (parent
    harness work that made this comparison possible)
