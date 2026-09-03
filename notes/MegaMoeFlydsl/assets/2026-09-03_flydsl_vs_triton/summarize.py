"""Aggregate the sharded CSVs into a Triton-vs-FlyDSL comparison.

Per the project's perf convention the headline score is the geometric mean of
Combined Step TFLOPS (6*M*N*K / (fwd+bwd)) across shapes, with forward and backward
kept as diagnostics. Rows that did not pass their correctness gate are dropped from
the score and reported separately, since a FAIL makes the timing meaningless.
"""

import glob
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

OP_LABELS = {
    "gemm_fp8_tw": "dense GEMM fp8 (tensorwise)",
    "gg_bf16": "grouped GEMM bf16",
    "gg_fp8_tw": "grouped GEMM fp8 (tensorwise)",
    "gg_fp8_mx": "grouped GEMM mxfp8",
    "gg_fp4_mx": "grouped GEMM mxfp4",
    "sparse_mla": "sparse MLA (DSV4)",
}
KEY = ["Op", "Case", "Shape"]


def shape_label(row):
    """One string per measured configuration: the GEMM ops are keyed on M/N/K, sparse MLA
    on its sequence length, so a single pivot key covers both."""
    if row["Op"] == "sparse_mla":
        return f"seqlen={int(row['Seqlen'])}"
    return f"M={int(row['M'])} N={int(row['N'])} K={int(row['K'])}"


def load():
    frames = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.csv"))):
        # Ops with fewer cases than shards leave the tail shards with nothing to write.
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True)
    for col in ("B", "EP", "Seqlen", "MBS"):
        if col not in df:
            df[col] = np.nan
    df["Combined Time (ms)"] = df["Forward Time (ms)"] + df["Backward Time (ms)"]
    df["Shape"] = df.apply(shape_label, axis=1)

    # ops/grouped_gemm.py short-circuits len(group_lens)==1 to a dense hipBLASLt GEMM,
    # so single-expert bf16 rows never reach either backend and time identically. They
    # would only dilute the bf16 score, so drop them.
    shortcut = (df["Op"] == "gg_bf16") & (df["B"] == 1)
    if shortcut.any():
        print(f"dropped {int(shortcut.sum())} gg_bf16 rows with B=1 (dense hipBLASLt short-circuit)")
    return df[~shortcut].copy()


def geomean(s):
    s = s.dropna()
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def main():
    df = load()
    bad = df[df["Check"] != "PASS"]
    ok = df[df["Check"] == "PASS"].copy()

    print(f"rows: {len(df)}  (PASS {len(ok)}, non-PASS {len(bad)})\n")
    if len(bad):
        cols = [c for c in ("Op", "Backend", "Case", "M", "N", "K", "Check", "Error") if c in bad]
        print("=== non-PASS rows (excluded from the score) ===")
        print(bad[cols].to_string(index=False), "\n")

    # Only shapes measured on BOTH backends are comparable.
    pivot = ok.pivot_table(
        index=KEY,
        columns="Backend",
        values=["Forward TFLOPS", "Backward TFLOPS", "Combined Time (ms)"],
        aggfunc="first",
    ).dropna()
    print(f"shapes with both backends: {len(pivot)}\n")

    print("=== per-op summary (geomean over shapes; speedup = FlyDSL / Triton) ===")
    hdr = f"{'op':32s} {'n':>3s} {'fwd TF/s T':>11s} {'fwd TF/s F':>11s} {'fwd x':>6s} {'bwd TF/s T':>11s} {'bwd TF/s F':>11s} {'bwd x':>6s} {'step x':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for op, label in OP_LABELS.items():
        sub = pivot[pivot.index.get_level_values("Op") == op]
        if sub.empty:
            print(f"{label:32s} {'--':>3s}")
            continue
        ft, ff = geomean(sub[("Forward TFLOPS", "TRITON")]), geomean(sub[("Forward TFLOPS", "FLYDSL")])
        bt, bf = geomean(sub[("Backward TFLOPS", "TRITON")]), geomean(sub[("Backward TFLOPS", "FLYDSL")])
        # Combined step speedup is a time ratio, so geomean the per-shape ratio.
        step = geomean(
            sub[("Combined Time (ms)", "TRITON")] / sub[("Combined Time (ms)", "FLYDSL")]
        )
        print(
            f"{label:32s} {len(sub):3d} {ft:11.1f} {ff:11.1f} {ff / ft:6.2f} "
            f"{bt:11.1f} {bf:11.1f} {bf / bt:6.2f} {step:7.2f}"
        )

    print("\n=== per-shape combined-step speedup (FlyDSL vs Triton) ===")
    flat = pivot.copy()
    flat["step_x"] = (
        flat[("Combined Time (ms)", "TRITON")] / flat[("Combined Time (ms)", "FLYDSL")]
    )
    # reset_index on the pivot leaves the column MultiIndex behind, which writes a blank
    # second header row to CSV; flattening to plain names drops it.
    flat = flat.reset_index()[["Op", "Case", "Shape", "step_x"]]
    flat.columns = ["Op", "Case", "Shape", "step_x"]
    for op in OP_LABELS:
        sub = flat[flat["Op"] == op].sort_values("step_x")
        if sub.empty:
            continue
        print(f"\n-- {OP_LABELS[op]} --")
        print(sub.drop(columns="Op").to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = os.path.join(os.path.dirname(RESULTS), "comparison.csv")
    flat.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
