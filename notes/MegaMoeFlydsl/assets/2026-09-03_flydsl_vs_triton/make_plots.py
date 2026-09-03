"""Render the Triton-vs-FlyDSL comparison charts embedded in the slab note.

Labels are English on purpose: the container's matplotlib has no CJK font, so Chinese
axis text would render as boxes. The note's prose carries the Chinese explanation.
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/perf_apps/xiaoming/slab/notes/MegaMoeFlydsl/assets/2026-09-03_flydsl_vs_triton"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_data.json")

TRITON_C, FLYDSL_C = "#8c8c8c", "#1f77b4"
GOOD_C, BAD_C = "#2e7d32", "#c62828"

SHORT = {
    "gemm_fp8_tw": "dense GEMM\nfp8 tensorwise",
    "gg_bf16": "grouped GEMM\nbf16",
    "gg_fp8_tw": "grouped GEMM\nfp8 tensorwise",
    "gg_fp8_mx": "grouped GEMM\nmxfp8",
    "gg_fp4_mx": "grouped GEMM\nmxfp4",
    "sparse_mla": "sparse MLA\n(DSV4)",
}
GROUPED = ["gg_bf16", "gg_fp8_tw", "gg_fp8_mx", "gg_fp4_mx"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if ax.get_yaxis().get_label_text() == "" else "y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def plot_speedup(summary, path, n_shapes):
    """Headline: combined-step speedup (fwd+bwd time ratio) per op family."""
    labels = [SHORT[s["op"]].replace("\n", " ") for s in summary]
    vals = [s["x"] for s in summary]
    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    bars = ax.barh(labels, vals, color=[GOOD_C if v >= 1.25 else FLYDSL_C for v in vals], height=0.6)
    ax.axvline(1.0, color=BAD_C, ls="--", lw=1.2, label="parity (1.0x)")
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}x", va="center", fontsize=10)
    ax.set_xlim(0.9, max(vals) * 1.12)
    ax.set_xlabel("Combined-step speedup, FlyDSL / Triton (x, geomean over shapes)")
    ax.set_title(
        "FlyDSL vs Triton: combined forward+backward step speedup\n"
        f"MI355X (gfx950), {n_shapes} shapes, all correctness-gated PASS",
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_tflops(summary, path):
    """Forward and backward throughput side by side, both backends."""
    labels = [SHORT[s["op"]] for s in summary]
    x = np.arange(len(labels))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160, sharey=True)
    for ax, (tk, fk, name) in zip(axes, [("tf", "ff", "Forward"), ("tb", "fb", "Backward")]):
        t = [s[tk] for s in summary]
        f = [s[fk] for s in summary]
        ax.bar(x - w / 2, t, w, label="Triton", color=TRITON_C)
        ax.bar(x + w / 2, f, w, label="FlyDSL", color=FLYDSL_C)
        for xi, (a, b) in enumerate(zip(t, f)):
            ax.text(xi + w / 2, b + 40, f"{b / a:.2f}x", ha="center", fontsize=8.5, color=FLYDSL_C)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"{name} throughput", fontsize=11, loc="left")
        style(ax)
    axes[0].set_ylabel("TFLOPS (geomean over shapes)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Throughput by operator and backend -- backward FLOPs counted as 2x forward (dgrad + wgrad)",
        fontsize=11,
        x=0.008,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_vs_m(rows, path):
    """Grouped-GEMM speedup against tokens-per-expert: where the bf16 small-M loss sits."""
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
    markers = {"gg_bf16": "o", "gg_fp8_tw": "s", "gg_fp8_mx": "^", "gg_fp4_mx": "D"}
    colors = {"gg_bf16": BAD_C, "gg_fp8_tw": "#ef8a3a", "gg_fp8_mx": FLYDSL_C, "gg_fp4_mx": GOOD_C}

    for op in GROUPED:
        sub = [r for r in rows if r["op"] == op]
        by_m = {}
        for r in sub:
            m = int(re.search(r"M=(\d+)", r["shape"]).group(1))
            by_m.setdefault(m, []).append(r["x"])
        ms = sorted(by_m)
        geo = [float(np.exp(np.mean(np.log(by_m[m])))) for m in ms]
        ax.plot(
            ms,
            geo,
            marker=markers[op],
            color=colors[op],
            lw=1.8,
            ms=6,
            label=SHORT[op].replace("\n", " "),
        )
        for m, g in zip(ms, geo):
            ax.annotate(f"{g:.2f}", (m, g), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")

    ax.axhline(1.0, color="#555", ls="--", lw=1.1)
    ax.text(1024, 1.005, "parity", fontsize=8.5, color="#555", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 8192])
    ax.set_xticklabels(["1024", "4096", "8192"])
    ax.set_xlabel("Tokens per expert (M)")
    ax.set_ylabel("Combined-step speedup, FlyDSL / Triton (x)")
    ax.set_title(
        "Grouped GEMM: FlyDSL's advantage grows with M; bf16 is the only loss, at M=1024",
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(DATA))
    summary, rows = d["summary"], d["rows"]
    plot_speedup(summary, os.path.join(OUT, "speedup-by-op.png"), len(rows))
    plot_tflops(summary, os.path.join(OUT, "throughput-fwd-bwd.png"))
    plot_vs_m(rows, os.path.join(OUT, "grouped-gemm-speedup-vs-m.png"))
    print("wrote:", *sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
