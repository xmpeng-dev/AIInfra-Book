# GPT-OSS-20B / MI355X — V2 fused residual+norm 800-iter A/B verdict

**Date:** 2026-04-25
**Hardware:** 8× MI355X single-node, TP1 PP1 EP1, GBS 32 / MBS 4
**Patches under test:**
- Baseline (`OPT-V1`) — `PRIMUS_FUSED_RESIDUAL_NORM=1` + `PRIMUS_MOE_SWIGLU_NOCAT=1`
- Treatment (`OPT-V1+V2`) — adds `PRIMUS_FUSED_RESIDUAL_NORM_V2=1`
**Status:** ❌ V2 is **neutral** at 800-iter scale; do NOT enable by default.

## 1. TL;DR

| metric | V1 (baseline) | V1+V2 | Δ |
|---|---|---|---|
| median ms/iter (steady, eval-trimmed, n=58) | 1177.30 | 1176.10 | **−1.20 ms (−0.10 %)** |
| trim10 ms/iter | 1185.32 | 1183.88 | **−1.44 ms (−0.12 %)** |
| median TFLOP/s/GPU | 701.5 | 702.2 | **+0.7 (+0.10 %)** |
| iters faster than V1 | — | 49 / 58 (84.5 %) | direction OK, magnitude trivial |
| eval #1 lm_loss @ 384 | 5.24200 | 5.24531 | +0.003 (noise) |
| eval #2 lm_loss @ 768 | 4.75609 | 4.73775 | **−0.018 (V2 ~1.6 % lower PPL)** |
| eval #3 lm_loss @ 800 | 4.73636 | 4.71879 | **−0.018** |
| NaN / skipped | 0 / 0 | 0 / 0 | clean |

**Expected gain (from yesterday's 80-iter smoke test, note 19): −0.93 % step (−10 ms). Realised gain at 800 iter: −0.10 %, an order of magnitude below noise floor.** The 80-iter
result was within noise; the asymptotic effect of V2 on this config is essentially nil.

## 2. What V2 does (recap)

V1 (`PRIMUS_FUSED_RESIDUAL_NORM=1`) fuses the **in-layer** ADD#1
(`self_attn_bda(residual + attn_out) → pre_mlp_layernorm`) into one Triton
RMSNorm-with-residual kernel.

V2 (`PRIMUS_FUSED_RESIDUAL_NORM_V2=1`, implies V1) additionally fuses the
**cross-layer** ADD#2 (`mlp_bda(residual + mlp_out) → next layer's
input_layernorm`) by stashing a `(mlp_out, residual)` carry on the next layer
and consuming it inside that layer's first norm. Last-layer carry goes to
`final_layernorm`. Per step (24 layers):

|                          | baseline | V1     | V2  |
|--------------------------|----------|--------|-----|
| `bf16 add` launches       | 48       | 24     | **1** |
| `triton_rmsnorm`           | 48       | 0      | 0    |
| `triton_rmsnorm_residual`  | 0        | 24     | **48** |

Trace evidence: the H_nocat baseline trace
(`run-trace/20260425_b_v1fused/H_nocat_s4_breakdown.txt`) showed the surviving
ADD#2 as a 39.85 ms `vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>`
on stream 0 — V2 was supposed to take that to ~0.

## 3. Patch lifecycle confirmation (V2 was actually applied)

From `opt_v2/opt_v2.log` at iter 0:

```
[Patch] Applying megatron.turbo.fused_residual_norm: Fuse residual+add into PrimusTurboRMSNorm via Triton triton_rmsnorm_residual; gated by PRIMUS_FUSED_RESIDUAL_NORM(_V2).
[Patch:megatron.turbo.fused_residual_norm] Installing fused residual+RMSNorm (V2=True, V1=True)
[fused_residual_rmsnorm] patched PrimusTurboRMSNorm.forward (residual= arg + V2 _pending_carry)
[fused_residual_rmsnorm] patched TransformerBlock.__init__ (V2 layer-link wiring active)
[fused_residual_rmsnorm] patched TransformerLayer.forward (V1 ADD#1 + V2 cross-layer ADD#2 fusion active)
[Patch] ✓ Applied: megatron.turbo.fused_residual_norm (priority=60)
```

No `fused forward raised`, no `falling back`, no `V2 layer-link wiring raised`
in the entire 800-iter log — V2 ran the full carry-fusion path on every layer,
on every step, with no per-layer fallback to V1.

`use_turbo_rms_norm=true` — confirmed in resolved args. All `_can_fuse`
preconditions are also satisfied (no fp32_residual, no recompute, hidden_dropout=0,
no offload, no cross_attention).

## 4. Plumbing under `@register_patch`

`Primus/primus/backends/megatron/patches/turbo/fused_residual_norm_patches.py` already
gated on either env var, so no code change was needed for V2 — flipping
`PRIMUS_FUSED_RESIDUAL_NORM_V2=1` activates it via `@register_patch`'s
auto-discovery, exactly the same way V1 / `swiglu_nocat` / `permute_fusion` are
applied.

```python
def _is_fused_residual_norm_enabled(ctx: PatchContext) -> bool:
    return _env_truthy("PRIMUS_FUSED_RESIDUAL_NORM") or _env_truthy(
        "PRIMUS_FUSED_RESIDUAL_NORM_V2"
    )

@register_patch(
    "megatron.turbo.fused_residual_norm",
    backend="megatron", phase="before_train",
    condition=_is_fused_residual_norm_enabled,
    priority=60,
)
def patch_fused_residual_norm(ctx: PatchContext): ...
```

## 5. Why no measurable speedup (hypothesis)

V2 *is* eliminating the ADD#2 launches, but:

1. The 39.85 ms ADD#2 in the V1 trace was running **on stream 0** in series with
   compute. On MI355X with the current memory bandwidth headroom, replacing 24
   bf16 add launches with adding 24 add-residual ops to the existing
   `triton_rmsnorm_residual` kernel **doesn't actually shrink the
   `triton_rmsnorm_residual` kernel by an equivalent amount** — the fused kernel
   becomes very slightly heavier (one extra add per element), and the savings
   in launch overhead + DRAM round-trip are largely re-spent inside the merged
   kernel.
2. Per-layer Python carry-stash bookkeeping (`object.__setattr__` on
   `_v2_carry`, `_v2_pending_carry`) costs ~tens of µs per layer × 24 layers ×
   800 iter, which is not free relative to a 1-2 ms savings target.
3. Confirmation would require a fresh trace under V1+V2 to see the
   `triton_rmsnorm_residual` kernel time, but given that the wall-clock
   delta is below noise (−0.12 % trim10), the kernel-level question is
   academic — the user-visible answer is "no benefit".

## 6. Decision

- **Do NOT default `PRIMUS_FUSED_RESIDUAL_NORM_V2=1`**. Keep it as an off-by-default
  experimental knob (which it already is — both `_v2_enabled()` and the
  `@register_patch` gate default to `0`).
- The V2 implementation in
  `Primus/primus/backends/megatron/core/extensions/fused_residual_rmsnorm.py`
  stays in place — it's correct, has graceful per-layer fallback, and may help
  on different model shapes (more layers, larger hidden dim, different memory
  pressure regimes). Just not on this 24-layer GPT-OSS-20B / MI355X config.
- This closes the Tier-1A optimization queue. **All Tier-1 / Tier-2 candidates
  on this trace have now been verified as either already enabled (V1, swiglu_nocat,
  permute_fusion, turbo_rms_norm, primus_topk_router, te_spec_provider) or
  ineffective on this config (V2, use_turbo_attention, use_turbo_grouped_mlp,
  sync_free_moe_stage, …)**.

## 7. Run artifacts

- V1 baseline log:
  `small_llm_moe_pretraining/primus/ab_runs/20260425_register_patch_800iter/opt/opt.log`
- V1+V2 log:
  `small_llm_moe_pretraining/primus/ab_runs/20260425_v2_verify/opt_v2/opt_v2.log`
- runner / snapshot scripts:
  `small_llm_moe_pretraining/primus/ab_runs/20260425_v2_verify/{run_v2.sh,snapshot_v2.py}`
- trimmed comparison: `/tmp/v2_clean.py`
