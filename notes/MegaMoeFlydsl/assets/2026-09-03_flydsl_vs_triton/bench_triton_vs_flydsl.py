"""Timed Triton-vs-FlyDSL comparison for the ops that register both backends.

One process per (op, backend): the backend is pinned through the PRIMUS_TURBO_*_BACKEND
env var, so AutoKernelDispatcher raises rather than silently falling back and every timed
row is guaranteed to be the named backend. Correctness runs first and gates the timing
(allclose for bf16, SNR for fp8/fp4) using the same helpers as benchmark/ops/training.

Shapes are a representative subset of config.py's grid: 4 MoE models x 2 EP sizes x
3 token counts x {GateUP, Down} for the grouped ops, 3 dense models x {1,4} MBS for
dense fp8 GEMM, and the DSV4 flash/pro variants for sparse MLA.
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

REPO = "/perf_apps/xiaoming/MegaMoE"
sys.path.insert(0, os.path.join(REPO, "benchmark/ops/training"))
sys.path.insert(0, os.path.join(REPO, "tests/pytorch/ops"))

from config import (  # noqa: E402
    DenseModelConfigs,
    MoEModelConfigs,
    check_allclose,
    compute_snr,
    gen_gemm_test_cases,
    gen_grouped_gemm_group_lens,
    get_platform_info,
    grouped_gemm_ref,
)

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)

DEV = "cuda"
BF16 = torch.bfloat16

FP8_TW = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
FP8_MX = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    block_size=32,
    scale_dtype=ScaleDtype.E8M0,
)
FP4_MX = Float4QuantConfig()

# SNR floors per the project's accuracy gate: E4M3 25 dB, E5M2 20 dB, FP4 10 dB.
SNR_FP8, SNR_FP4 = 25.0, 10.0

# Subset of config.py's MoE grid. EP sizes are the two largest that divide the expert
# count (smallest B), which keeps per-GPU token counts in a realistic training range.
MOE_MODELS = ["DeepSeek-V3", "Qwen3-235B-A22B", "Mixtral-8x22B", "Kimi-K2"]
MOE_EP_SIZES = [32, 16, 8]
MOE_M_SIZES = [1024, 4096, 8192]

DENSE_MODELS = ["Llama-3.1-8B", "Llama-3.1-405B", "Qwen2.5-72B"]
DENSE_MBS = [1, 4]

# DSV4 sparse MLA: (variant, num_heads) x seqlen, cr=4 = random pool on top of the SWA band.
SPARSE_MLA_CASES = [(v, s) for v in ("flash", "pro") for s in (1024, 2048, 4096)]

WARMUP, ITERS = 20, 100


def moe_cases():
    cases = []
    for name in MOE_MODELS:
        cfg = MoEModelConfigs[name]
        experts, inter, hidden = (
            cfg["n_routed_experts"],
            cfg["moe_intermediate_size"],
            cfg["hidden_size"],
        )
        eps = [ep for ep in MOE_EP_SIZES if experts % ep == 0 and experts // ep >= 1][:2]
        for ep in eps:
            b = experts // ep
            for m in MOE_M_SIZES:
                for label, (n, k) in (
                    (f"{name}-GateUP", (2 * inter, hidden)),
                    (f"{name}-Down", (hidden, inter)),
                ):
                    cases.append({"Case": label, "EP": ep, "B": b, "M": m, "N": n, "K": k})
    return cases


def dense_cases():
    cases = []
    for name in DENSE_MODELS:
        for mbs in DENSE_MBS:
            for m, n, k in gen_gemm_test_cases(DenseModelConfigs[name]):
                cases.append({"Case": name, "MBS": mbs, "M": m * mbs, "N": n, "K": k})
    return cases


def _time_ms(fn):
    return benchmark.Timer(stmt="fn()", globals={"fn": fn}).timeit(ITERS).mean * 1e3


def _measure(fwd, bwd, fwd_flops):
    """fwd/bwd wall time and TFLOPS; bwd counts 2x the forward FLOPs (dgrad + wgrad)."""
    for _ in range(WARMUP):
        fwd()
        bwd()
    torch.cuda.synchronize()
    fwd_ms, bwd_ms = _time_ms(fwd), _time_ms(bwd)
    return {
        "Forward Time (ms)": fwd_ms,
        "Forward TFLOPS": fwd_flops / (fwd_ms * 1e-3) / 1e12,
        "Backward Time (ms)": bwd_ms,
        "Backward TFLOPS": 2 * fwd_flops / (bwd_ms * 1e-3) / 1e12,
    }


# --------------------------------------------------------------------------------------
# Grouped GEMM (bf16 / fp8 tensorwise / fp8 mxfp8 / fp4 mxfp4)
# --------------------------------------------------------------------------------------


def run_grouped(case, quant_config, snr_floor):
    b, m, n, k = case["B"], case["M"], case["N"], case["K"]
    group_lens = gen_grouped_gemm_group_lens(b, m, balance=True).to(DEV)
    a = torch.randn(b * m, k, dtype=BF16, device=DEV, requires_grad=True)
    w = torch.randn(b, n, k, dtype=BF16, device=DEV, requires_grad=True)

    if quant_config is None:
        call = lambda: turbo.ops.grouped_gemm(a, w, group_lens, trans_b=True)
    elif isinstance(quant_config, Float4QuantConfig):
        call = lambda: turbo.ops.grouped_gemm_fp4(a, w, group_lens, trans_b=True, config=quant_config)
    else:
        call = lambda: turbo.ops.grouped_gemm_fp8(a, w, group_lens, trans_b=True, config=quant_config)

    out = call()
    grad_out = torch.randn_like(out)
    check = _check_grouped(a, w, out, grad_out, group_lens, snr_floor)

    bwd = lambda: out.backward(grad_out, retain_graph=True)
    stats = _measure(call, bwd, 2 * b * m * n * k)
    return check, stats


def _check_grouped(a, w, out, grad_out, group_lens, snr_floor):
    ref = grouped_gemm_ref(a.detach(), w.detach(), group_lens, trans_b=True)
    a_ref = a.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    grouped_gemm_ref(a_ref, w_ref, group_lens, trans_b=True).backward(grad_out)
    out.backward(grad_out, retain_graph=True)

    if snr_floor is None:
        ok = all(
            check_allclose(got.detach(), exp, BF16)
            for got, exp in ((out, ref), (a.grad, a_ref.grad), (w.grad, w_ref.grad))
        )
    else:
        snrs = [
            compute_snr(ref, out.detach()),
            compute_snr(a_ref.grad, a.grad),
            compute_snr(w_ref.grad, w.grad),
        ]
        ok = all(s > snr_floor for s in snrs)
    a.grad = w.grad = None
    return ok


# --------------------------------------------------------------------------------------
# Dense fp8 GEMM (tensorwise)
# --------------------------------------------------------------------------------------


def run_dense_fp8(case, quant_config, snr_floor):
    m, n, k = case["M"], case["N"], case["K"]
    a = torch.randn(m, k, dtype=BF16, device=DEV, requires_grad=True)
    b = torch.randn(n, k, dtype=BF16, device=DEV, requires_grad=True)

    call = lambda: turbo.ops.gemm_fp8(a, b, trans_b=True, config=quant_config)
    out = call()
    grad_out = torch.randn_like(out)

    ref = a.detach() @ b.detach().T
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    (a_ref @ b_ref.T).backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    snrs = [
        compute_snr(ref, out.detach()),
        compute_snr(a_ref.grad, a.grad),
        compute_snr(b_ref.grad, b.grad),
    ]
    check = all(s > snr_floor for s in snrs)
    a.grad = b.grad = None

    bwd = lambda: out.backward(grad_out, retain_graph=True)
    stats = _measure(call, bwd, 2 * m * n * k)
    return check, stats


# --------------------------------------------------------------------------------------
# DSV4 sparse MLA
# --------------------------------------------------------------------------------------


def run_sparse_mla(case, *_):
    from test_attention import _build_sparse_mla, _sparse_mla_topk

    variant, seqlen = case["Variant"], case["Seqlen"]
    heads = 64 if variant == "flash" else 128
    pool, topk_pool, _ = _sparse_mla_topk(variant, 4, seqlen)
    q, kv, topk_idx, sink, grad_out = _build_sparse_mla(4, heads, seqlen, pool, topk_pool)
    q = q.detach().requires_grad_()
    kv = kv.detach().requires_grad_()
    sink = sink.detach().requires_grad_()

    call = lambda: turbo.ops.sparse_mla_func(q, kv, topk_idx, attn_sink=sink, kv_lora_rank=512)
    out = call()
    check = _check_sparse_mla(q, kv, topk_idx, sink, out)

    bwd = lambda: out.backward(grad_out, retain_graph=True)
    # QK over the 576-wide latent+rope plus PV over the 512-wide latent, per selected kv row.
    topk = topk_idx.shape[1]
    fwd_flops = 2 * seqlen * heads * topk * (576 + 512)
    stats = _measure(call, bwd, fwd_flops)
    return check, stats


@torch.no_grad()
def _check_sparse_mla(q, kv, topk_idx, sink, out, chunk=256):
    """fp32 gather-and-softmax reference over the selected kv rows, chunked over tokens."""
    kv_flat = kv.detach().squeeze(1).float()
    q_f = q.detach().float()
    scale = 1.0 / (q.shape[-1] ** 0.5)
    ok = True
    for lo in range(0, q.shape[0], chunk):
        hi = min(lo + chunk, q.shape[0])
        idx = topk_idx[lo:hi].long()
        valid = idx >= 0
        gathered = kv_flat[idx.clamp(min=0)]  # [c, topk, 576]
        scores = torch.einsum("thd,tkd->thk", q_f[lo:hi], gathered) * scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        aug = torch.cat([scores, sink.detach().float().view(1, -1, 1).expand(hi - lo, -1, 1)], -1)
        probs = aug.softmax(-1)[..., :-1]
        ref = torch.einsum("thk,tkd->thd", probs, gathered[..., :512])
        ok &= check_allclose(out[lo:hi].detach(), ref.to(BF16), BF16)
    return ok


OPS = {
    "gemm_fp8_tw": ("PRIMUS_TURBO_GEMM_BACKEND", dense_cases, run_dense_fp8, FP8_TW, SNR_FP8),
    "gg_bf16": ("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", moe_cases, run_grouped, None, None),
    "gg_fp8_tw": ("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", moe_cases, run_grouped, FP8_TW, SNR_FP8),
    "gg_fp8_mx": ("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", moe_cases, run_grouped, FP8_MX, SNR_FP8),
    "gg_fp4_mx": ("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", moe_cases, run_grouped, FP4_MX, SNR_FP4),
    "sparse_mla": (
        "PRIMUS_TURBO_SPARSE_ATTN_BACKEND",
        lambda: [{"Case": f"DSV4-{v}", "Variant": v, "Seqlen": s} for v, s in SPARSE_MLA_CASES],
        run_sparse_mla,
        None,
        None,
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", required=True, choices=sorted(OPS))
    p.add_argument("--backend", required=True, choices=["TRITON", "FLYDSL"])
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    env_key, case_fn, run_fn, quant_config, snr_floor = OPS[args.op]
    if os.environ.get(env_key) != args.backend:
        raise SystemExit(f"{env_key} must be set to {args.backend} before launch (got {os.environ.get(env_key)!r})")

    platform, gpu = get_platform_info()
    cases = case_fn()
    rows = []
    for i, case in enumerate(cases):
        if i % args.num_shards != args.shard_id:
            continue
        row = {"Op": args.op, "Backend": args.backend, "Platform": platform, "GPU": gpu, **case}
        try:
            check, stats = run_fn(case, quant_config, snr_floor)
            row["Check"] = "PASS" if check else "FAIL"
            row.update({k: round(v, 4) for k, v in stats.items()})
        except Exception as e:
            row["Check"] = "ERROR"
            row["Error"] = str(e).strip().splitlines()[0][:200]
        rows.append(row)
        print(f"[{i + 1}/{len(cases)}] {row}", flush=True)
        torch.cuda.empty_cache()

    out = args.output or (
        f"{args.op}_{args.backend}_{datetime.now():%Y%m%d}_{gpu}.part-{args.shard_id}.csv"
    )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
