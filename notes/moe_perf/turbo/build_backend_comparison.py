#!/usr/bin/env python3
"""Render the *Per-model breakdown* and *Backend comparison* sections of
`README.md` from the per-backend CSVs produced by `run_all_models.sh`
under `archive_backends/<backend>/<slug>-spread-breakdown.csv`.

Output layout per model:

    ### <Model>
    `<shape>`

    #### Forward — Triton
    | ... full breakdown table ... |
    #### Forward — HIPBLASLT
    | ... |
    #### Forward — CK
    | ... |
    #### Backward — Triton
    | ... |
    #### Backward — HIPBLASLT
    | ... |
    #### Backward — CK
    | ... |
    #### Step time — Triton vs HIPBLASLT vs CK
    | ... step time + Δ vs Triton ... |

The earlier "Backend comparison" headline tables (PROD step / GEMM
TFLOPS / DeepEP A2A side-effect) are still printed at the top as a
quick summary.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

BACKENDS = ["triton", "hipblaslt", "ck"]
BACKEND_LABEL = {
    "triton": "Triton",
    "hipblaslt": "HIPBLASLT",
    "ck": "CK",
}

# (display name, slug, shape string, PROD BS)
MODELS = [
    (
        "DeepSeek-V2-Lite",
        "deepseek-v2-lite",
        "experts=64 top_k=6 hidden=2048 intermediate=1408 ep_size=8 local_experts=8",
        49152,
    ),
    (
        "DeepSeek-V3",
        "deepseek-v3",
        "experts=256 top_k=8 hidden=7168 intermediate=2048 ep_size=8 local_experts=32",
        8192,
    ),
    (
        "Qwen3-30B-A3B",
        "qwen3-30b-a3b",
        "experts=128 top_k=8 hidden=2048 intermediate=768 ep_size=8 local_experts=16",
        32768,
    ),
    (
        "Qwen3-235B-A22B",
        "qwen3-235b-a22b",
        "experts=128 top_k=8 hidden=4096 intermediate=1536 ep_size=8 local_experts=16",
        16384,
    ),
]


def load(backend: str, slug: str) -> dict[int, dict]:
    p = HERE / "archive_backends" / backend / f"{slug}-spread-breakdown.csv"
    if not p.exists():
        return {}
    out = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            out[int(r["num_tokens"])] = r
    return out


def pct(base: float, new: float) -> str:
    if base == 0:
        return "—"
    return f"{(new - base) / base * 100:+.1f}%"


# ---------------------------------------------------------------------------
# Per-model breakdown
# ---------------------------------------------------------------------------


def _bs_label(bs: int, prod_bs: int) -> str:
    # The PROD row's cells are wrapped in **...** by the caller, so don't
    # pre-bold here (otherwise we'd get ****49,152 ★****).
    return f"{bs:,} ★" if bs == prod_bs else f"{bs:,}"


def render_fwd_table(backend: str, rows: dict[int, dict], prod_bs: int) -> str:
    out = ["#### Forward — " + BACKEND_LABEL[backend], ""]
    if not rows:
        out.append("(no data)")
        return "\n".join(out) + "\n"
    out.append(
        "| Batch Size | Time (us) | Compute (TFLOPS) | Global Memory (GB/s) | "
        "sort (us) | dispatch (us) | fc1 (us / TFLOPS) | "
        "swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | "
        "combine (us) | misc (us) | all_kernels (us) |"
    )
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bs in sorted(rows):
        r = rows[bs]
        is_prod = bs == prod_bs
        cells = [
            _bs_label(bs, prod_bs),
            f"{float(r['time_us']):.1f}",
            f"{float(r['tflops']):.1f}",
            f"{float(r['gbps']):.0f}",
            f"{float(r['sort_us']):.1f}",
            f"{float(r['dispatch_us']):.1f}",
            f"{float(r['fc1_us']):.1f} / {float(r['fc1_tflops']):.1f}",
            f"{float(r['swiglu_us']):.1f} / {float(r['swiglu_tflops']):.1f}",
            f"{float(r['fc2_us']):.1f} / {float(r['fc2_tflops']):.1f}",
            f"{float(r['combine_us']):.1f}",
            f"{float(r['misc_us']):.1f}",
            f"{float(r['all_kernels_us']):.1f}",
        ]
        if is_prod:
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_bwd_table(backend: str, rows: dict[int, dict], prod_bs: int) -> str:
    out = ["#### Backward — " + BACKEND_LABEL[backend], ""]
    if not rows:
        out.append("(no data)")
        return "\n".join(out) + "\n"
    out.append(
        "| Batch Size | Time (us) | Compute (TFLOPS) | "
        "sort (us) | dispatch (us) | fc1 (us / TFLOPS) | "
        "swiglu (us / TFLOPS) | fc2 (us / TFLOPS) | "
        "combine (us) | misc (us) | all_kernels (us) |"
    )
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bs in sorted(rows):
        r = rows[bs]
        is_prod = bs == prod_bs
        cells = [
            _bs_label(bs, prod_bs),
            f"{float(r['bwd_time_us']):.1f}",
            f"{float(r['bwd_tflops']):.1f}",
            f"{float(r['bwd_sort_us']):.1f}",
            f"{float(r['bwd_dispatch_us']):.1f}",
            f"{float(r['bwd_fc1_us']):.1f} / {float(r['bwd_fc1_tflops']):.1f}",
            f"{float(r['bwd_swiglu_us']):.1f} / {float(r['bwd_swiglu_tflops']):.1f}",
            f"{float(r['bwd_fc2_us']):.1f} / {float(r['bwd_fc2_tflops']):.1f}",
            f"{float(r['bwd_combine_us']):.1f}",
            f"{float(r['bwd_misc_us']):.1f}",
            f"{float(r['bwd_all_kernels_us']):.1f}",
        ]
        if is_prod:
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_step_table(rows_by_backend: dict[str, dict[int, dict]], prod_bs: int) -> str:
    out = [
        "#### Step time — Triton vs HIPBLASLT vs CK",
        "",
        "Forward + Backward wall time per BS. **Δ** columns compare against Triton at the same BS.",
        "",
        (
            "| BS | Triton fwd | Triton bwd | Triton step | "
            "HBLAS fwd | HBLAS bwd | HBLAS step | HBLAS Δ | "
            "CK fwd | CK bwd | CK step | CK Δ |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_bs = sorted(set().union(*(set(r) for r in rows_by_backend.values())))
    for bs in all_bs:
        is_prod = bs == prod_bs

        def trio(b: str) -> tuple[float | None, float | None, float | None]:
            r = rows_by_backend.get(b, {}).get(bs)
            if r is None:
                return None, None, None
            f = float(r["time_us"])
            bw = float(r["bwd_time_us"])
            return f, bw, f + bw

        tf, tb, ts = trio("triton")
        hf, hb, hs = trio("hipblaslt")
        cf, cb, cs = trio("ck")

        def fmt(v: float | None) -> str:
            return f"{v:,.0f}" if v is not None else "—"

        def delta(s: float | None) -> str:
            if ts is None or s is None:
                return "—"
            return pct(ts, s)

        cells = [
            f"{bs:,}" + (" ★" if is_prod else ""),
            fmt(tf), fmt(tb), f"**{ts:,.0f}**" if ts is not None else "—",
            fmt(hf), fmt(hb), fmt(hs), delta(hs),
            fmt(cf), fmt(cb), fmt(cs), delta(cs),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_per_model_section() -> str:
    """Replacement for the README's `## Per-model breakdown` section."""
    blocks: list[str] = []
    blocks.append("## Per-model breakdown — spread routing, BF16\n")
    blocks.append(
        "Each model has **seven** tables: forward breakdown for each "
        "grouped-GEMM backend (Triton / HIPBLASLT / CK), backward "
        "breakdown for each backend, and a step-time comparison across "
        "the three. All rows are spread routing, no autotune, "
        "`--sync-free-stage 2`, BF16 weights + activations. The ★ row in "
        "each table marks that model's production token count per rank "
        "(`mbs × seq_length`). Bold numbers in the breakdown tables "
        "highlight the PROD row.\n"
    )
    blocks.append(
        "Column semantics (forward): `Time (us)` is the e2e CUDA-event "
        "wall time. `Compute (TFLOPS)` is FC1+FC2 matmul throughput "
        "against `Time (us)` (backward counts the 2× FLOPs from "
        "`grad_input + grad_weight` GEMMs). `fc1 / swiglu / fc2` are "
        "`mean_us / TFLOPS`; per-stage TFLOPS uses each kernel's own "
        "wall-time and its theoretical FLOPs (fwd fc1 = `4·N·H·F`, "
        "swiglu = `5·N·F`, fc2 = `2·N·H·F`, `N = num_tokens × topk`; "
        "bwd = 2× fwd FLOPs). `Global Memory (GB/s)` is the "
        "forward-only e2e-averaged DRAM traffic. `all_kernels (us)` is "
        "the sum of per-stage CUDA-event medians; the residual vs "
        "`Time (us)` is launch-queue overhead. In the backward table, "
        "`sort` is a residual (e2e backward minus the sum of all other "
        "stages) because PyTorch leaf-tensor hooks fire at unpredictable "
        "times.\n"
    )
    blocks.append(
        "Raw CSVs live in `archive_backends/{triton,hipblaslt,ck}/`. "
        "This whole section is regenerated by "
        "`python3 build_backend_comparison.py`.\n"
    )

    for name, slug, shape, prod_bs in MODELS:
        blocks.append(f"### {name}\n")
        blocks.append(f"`{shape}`\n")
        rows_by_backend = {b: load(b, slug) for b in BACKENDS}
        for b in BACKENDS:
            blocks.append(render_fwd_table(b, rows_by_backend[b], prod_bs))
        for b in BACKENDS:
            blocks.append(render_bwd_table(b, rows_by_backend[b], prod_bs))
        blocks.append(render_step_table(rows_by_backend, prod_bs))

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------


def render_headline_step_table() -> str:
    out = [
        "### End-to-end step time at PROD batch size (us)\n",
        (
            "| Model | PROD BS | Triton fwd | HBLAS fwd | CK fwd | "
            "Triton bwd | HBLAS bwd | CK bwd | "
            "Triton step | HBLAS step | CK step | HBLAS Δstep | CK Δstep |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, slug, _, prod_bs in MODELS:
        rows = {b: load(b, slug) for b in BACKENDS}
        if not all(prod_bs in rows[b] for b in BACKENDS):
            # Render what we have, mark missing as —.
            tf = float(rows["triton"][prod_bs]["time_us"]) if prod_bs in rows["triton"] else None
            tb = float(rows["triton"][prod_bs]["bwd_time_us"]) if prod_bs in rows["triton"] else None
            hf = float(rows["hipblaslt"][prod_bs]["time_us"]) if prod_bs in rows["hipblaslt"] else None
            hb = float(rows["hipblaslt"][prod_bs]["bwd_time_us"]) if prod_bs in rows["hipblaslt"] else None
            cf = float(rows["ck"][prod_bs]["time_us"]) if prod_bs in rows["ck"] else None
            cb = float(rows["ck"][prod_bs]["bwd_time_us"]) if prod_bs in rows["ck"] else None
        else:
            tf = float(rows["triton"][prod_bs]["time_us"])
            hf = float(rows["hipblaslt"][prod_bs]["time_us"])
            cf = float(rows["ck"][prod_bs]["time_us"])
            tb = float(rows["triton"][prod_bs]["bwd_time_us"])
            hb = float(rows["hipblaslt"][prod_bs]["bwd_time_us"])
            cb = float(rows["ck"][prod_bs]["bwd_time_us"])

        def s(a, b):
            return None if a is None or b is None else a + b

        ts = s(tf, tb)
        hs = s(hf, hb)
        cs = s(cf, cb)

        def fmt(v):
            return f"{v:,.0f}" if v is not None else "—"

        def delta(new):
            return pct(ts, new) if ts is not None and new is not None else "—"

        out.append(
            "| " + " | ".join([
                name,
                f"{prod_bs:,}",
                fmt(tf), fmt(hf), fmt(cf),
                fmt(tb), fmt(hb), fmt(cb),
                f"**{ts:,.0f}**" if ts is not None else "—",
                fmt(hs), fmt(cs),
                delta(hs), delta(cs),
            ]) + " |"
        )
    return "\n".join(out) + "\n"


def render_gemm_headline() -> str:
    out = [
        "### Grouped-GEMM kernels (TFLOPS at PROD BS)\n",
        (
            "| Model | PROD BS | fc1_fwd Tri / HBLAS / CK | "
            "fc2_fwd Tri / HBLAS / CK | fc1_bwd Tri / HBLAS / CK | "
            "fc2_bwd Tri / HBLAS / CK |"
        ),
        "|---|---:|---|---|---|---|",
    ]
    for name, slug, _, prod_bs in MODELS:
        rows = {b: load(b, slug) for b in BACKENDS}
        if not all(prod_bs in rows[b] for b in BACKENDS):
            continue

        def trio(key: str) -> str:
            t = float(rows["triton"][prod_bs][key])
            h = float(rows["hipblaslt"][prod_bs][key])
            c = float(rows["ck"][prod_bs][key])
            return f"{t:.0f} / {h:.0f} / {c:.0f}"

        out.append(
            "| "
            + " | ".join([
                name,
                f"{prod_bs:,}",
                trio("fc1_tflops"),
                trio("fc2_tflops"),
                trio("bwd_fc1_tflops"),
                trio("bwd_fc2_tflops"),
            ])
            + " |"
        )
    return "\n".join(out) + "\n"


def render_combine_headline() -> str:
    out = [
        "### DeepEP combine/dispatch side-effect (us at PROD BS)\n",
        "| Model | PROD BS | dispatch_fwd | combine_fwd | dispatch_bwd | combine_bwd |",
        "|---|---:|---|---|---|---|",
    ]
    for name, slug, _, prod_bs in MODELS:
        rows = {b: load(b, slug) for b in BACKENDS}
        if not all(prod_bs in rows[b] for b in BACKENDS):
            continue

        def trio_pct(key: str) -> str:
            t = float(rows["triton"][prod_bs][key])
            h = float(rows["hipblaslt"][prod_bs][key])
            c = float(rows["ck"][prod_bs][key])
            return f"{t:.0f} / {h:.0f} ({pct(t,h)}) / {c:.0f} ({pct(t,c)})"

        out.append(
            "| "
            + " | ".join([
                name,
                f"{prod_bs:,}",
                trio_pct("dispatch_us"),
                trio_pct("combine_us"),
                trio_pct("bwd_dispatch_us"),
                trio_pct("bwd_combine_us"),
            ])
            + " |"
        )
    return "\n".join(out) + "\n"


def render_headline_section() -> str:
    """Top-level Backend comparison summary (PROD-point only)."""
    blocks = [
        "## Backend comparison summary — Triton vs HIPBLASLT vs CK (no autotune)\n",
        (
            "Same setup as the per-model tables above: `BACKEND=TRITON|HIPBLASLT|CK "
            "bash run_all_models.sh`, autotune off, raw CSVs in "
            "`archive_backends/<backend>/`. The full per-BS breakdown for each "
            "backend is in the *Per-model breakdown* section.\n"
        ),
        render_headline_step_table(),
        render_gemm_headline(),
        render_combine_headline(),
    ]
    return "\n".join(blocks)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--section":
        which = sys.argv[2]
        if which == "per-model":
            print(render_per_model_section(), end="")
        elif which == "headline":
            print(render_headline_section(), end="")
        else:
            print(f"unknown section: {which}", file=sys.stderr)
            sys.exit(1)
    else:
        print(render_per_model_section())
        print(render_headline_section())


if __name__ == "__main__":
    main()
