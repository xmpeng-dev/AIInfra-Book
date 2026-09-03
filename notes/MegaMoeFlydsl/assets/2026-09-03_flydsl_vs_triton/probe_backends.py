"""Probe which (op, granularity, backend) combinations actually run on this GPU.

Each candidate op has both TRITON and FLYDSL registered in its dispatcher. Forcing a
backend that cannot handle the inputs raises from AutoKernelDispatcher.dispatch, so a
clean fwd+bwd here is the gate for including that combination in the timed comparison.
"""

import os
import sys
import traceback

import torch

sys.path.insert(0, "/perf_apps/xiaoming/MegaMoE/tests/pytorch/ops")

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
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
FP8_RW = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.ROWWISE)
FP8_BW = Float8QuantConfig(
    format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128
)
FP4_MX = Float4QuantConfig()


def _fwd_bwd(out, grad):
    out.backward(grad)
    torch.cuda.synchronize()


def op_gemm_fp8(config):
    m, n, k = 4096, 4096, 4096
    a = torch.randn(m, k, dtype=BF16, device=DEV, requires_grad=True)
    b = torch.randn(n, k, dtype=BF16, device=DEV, requires_grad=True)
    out = turbo.ops.gemm_fp8(a, b, trans_b=True, config=config)
    _fwd_bwd(out, torch.randn_like(out))


def op_grouped_gemm_bf16(_config=None):
    g, m, n, k = 8, 512, 4096, 7168
    x = torch.randn(g * m, k, dtype=BF16, device=DEV, requires_grad=True)
    w = torch.randn(g, n, k, dtype=BF16, device=DEV, requires_grad=True)
    group_lens = torch.full((g,), m, dtype=torch.int64, device=DEV)
    out = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    _fwd_bwd(out, torch.randn_like(out))


def op_grouped_gemm_fp8(config):
    g, m, n, k = 8, 512, 4096, 7168
    a = torch.randn(g * m, k, dtype=BF16, device=DEV, requires_grad=True)
    b = torch.randn(g, n, k, dtype=BF16, device=DEV, requires_grad=True)
    group_lens = torch.full((g,), m, dtype=torch.int64, device=DEV)
    out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    _fwd_bwd(out, torch.randn_like(out))


def op_grouped_gemm_fp4(config):
    g, m, n, k = 8, 512, 4096, 7168
    a = torch.randn(g * m, k, dtype=BF16, device=DEV, requires_grad=True)
    b = torch.randn(g, n, k, dtype=BF16, device=DEV, requires_grad=True)
    group_lens = torch.full((g,), m, dtype=torch.int64, device=DEV)
    out = turbo.ops.grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)
    _fwd_bwd(out, torch.randn_like(out))


def op_sparse_mla(_config=None):
    from test_attention import _build_sparse_mla, _sparse_mla_topk

    seqlen, variant, cr = 2048, "flash", 4
    num_heads = 64
    pool, topk_pool, _ = _sparse_mla_topk(variant, cr, seqlen)
    q, kv, topk_idx, sink, grad_out = _build_sparse_mla(cr, num_heads, seqlen, pool, topk_pool)
    q = q.detach().requires_grad_()
    kv = kv.detach().requires_grad_()
    sink = sink.detach().requires_grad_()
    out = turbo.ops.sparse_mla_func(q, kv, topk_idx, attn_sink=sink, kv_lora_rank=512)
    _fwd_bwd(out, grad_out)


# (label, env var, op fn, quant config)
CASES = [
    ("gemm_fp8/tensorwise", "PRIMUS_TURBO_GEMM_BACKEND", op_gemm_fp8, FP8_TW),
    ("gemm_fp8/rowwise", "PRIMUS_TURBO_GEMM_BACKEND", op_gemm_fp8, FP8_RW),
    ("gemm_fp8/blockwise", "PRIMUS_TURBO_GEMM_BACKEND", op_gemm_fp8, FP8_BW),
    ("gemm_fp8/mx_blockwise", "PRIMUS_TURBO_GEMM_BACKEND", op_gemm_fp8, FP8_MX),
    ("grouped_gemm/bf16", "PRIMUS_TURBO_GROUPED_GEMM_BACKEND", op_grouped_gemm_bf16, None),
    ("grouped_gemm_fp8/tensorwise", "PRIMUS_TURBO_GROUPED_GEMM_BACKEND", op_grouped_gemm_fp8, FP8_TW),
    ("grouped_gemm_fp8/mx_blockwise", "PRIMUS_TURBO_GROUPED_GEMM_BACKEND", op_grouped_gemm_fp8, FP8_MX),
    ("grouped_gemm_fp4/mx_blockwise", "PRIMUS_TURBO_GROUPED_GEMM_BACKEND", op_grouped_gemm_fp4, FP4_MX),
    ("sparse_mla", "PRIMUS_TURBO_SPARSE_ATTN_BACKEND", op_sparse_mla, None),
]

BACKENDS = ["TRITON", "FLYDSL"]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = []
    for label, env_key, fn, config in CASES:
        for backend in BACKENDS:
            os.environ[env_key] = backend
            GlobalBackendManager.reset()
            try:
                fn(config) if config is not None else fn()
                status, detail = "OK", ""
            except Exception as e:
                status = "UNSUPPORTED" if isinstance(e, (ValueError, AssertionError)) else "ERROR"
                detail = str(e).strip().splitlines()[0][:150]
                if status == "ERROR":
                    traceback.print_exc(limit=3)
            finally:
                os.environ.pop(env_key, None)
                GlobalBackendManager.reset()
                torch.cuda.empty_cache()
            print(f"{label:32s} {backend:8s} {status:12s} {detail}")
            results.append((label, backend, status))

    print("\n=== combinations runnable on BOTH backends ===")
    by_label = {}
    for label, backend, status in results:
        by_label.setdefault(label, {})[backend] = status
    for label, per_backend in by_label.items():
        if all(s == "OK" for s in per_backend.values()) and len(per_backend) == len(BACKENDS):
            print(f"  {label}")


if __name__ == "__main__":
    main()
