import re
import sys
from pathlib import Path

ROOT = Path("/tmp")

# (label, log dir prefix)
RUNS = [
    ("baseline",  "dsv3_baseline_logs"),
    ("eager",     "dsv3_eager_logs"),
    ("decomposed","dsv3_decomposed_logs"),
]

# pull from rank 7 stderr (timers.log() prints on last rank)

TIMERS = [
    "forward-backward",
    "forward-compute",
    "backward-compute",
    "optimizer",
    "batch-generator",
]

PAT = {t: re.compile(rf"^\s+{re.escape(t)}\s+\.+:\s+\((\d+\.\d+),\s*(\d+\.\d+)\)") for t in TIMERS}

def parse(stderr_path):
    """Return list of dicts, one per iteration block (excluding model-setup)."""
    blocks = []
    cur = {}
    for ln in open(stderr_path):
        # Strip the ANSI / Primus prefix to expose the raw timers.log() body
        for t, pat in PAT.items():
            m = pat.search(ln)
            if m:
                # use max time (worst rank)
                cur[t] = float(m.group(2))
        if "number of nan iterations" in ln and cur:
            blocks.append(cur)
            cur = {}
    return blocks

print(f"{'run':12s} | iter | fwd-bwd | forward | backward | optim  | batch-gen")
print("-" * 80)
for label, dirpre in RUNS:
    paths = sorted(ROOT.glob(f"{dirpre}/*/attempt_0/7/stderr.log"))
    if not paths:
        print(f"{label:12s} | (no logs found at {ROOT}/{dirpre})")
        continue
    blocks = parse(paths[0])
    for i, b in enumerate(blocks, start=1):
        fb = b.get("forward-backward", float("nan"))
        f  = b.get("forward-compute", float("nan"))
        bw = b.get("backward-compute", float("nan"))
        op = b.get("optimizer", float("nan"))
        bg = b.get("batch-generator", float("nan"))
        print(f"{label:12s} | {i:4d} | {fb:7.1f} | {f:7.1f} | {bw:8.1f} | {op:6.1f} | {bg:6.1f}")
    print()

# Steady-state average (iter 5-20, skipping JIT / warm-up first 2-3 iters)
print("\n=== steady-state averages (iter 5..end) ===")
print(f"{'run':12s} | fwd-bwd | forward | backward | optim  | batch-gen | iter_total")
print("-" * 86)
for label, dirpre in RUNS:
    paths = sorted(ROOT.glob(f"{dirpre}/*/attempt_0/7/stderr.log"))
    if not paths:
        continue
    blocks = parse(paths[0])
    tail = blocks[4:]  # iter 5 onward
    if not tail:
        continue
    n = len(tail)
    fb = sum(b.get("forward-backward", 0) for b in tail) / n
    f  = sum(b.get("forward-compute",  0) for b in tail) / n
    bw = sum(b.get("backward-compute", 0) for b in tail) / n
    op = sum(b.get("optimizer",        0) for b in tail) / n
    bg = sum(b.get("batch-generator",  0) for b in tail) / n
    print(f"{label:12s} | {fb:7.1f} | {f:7.1f} | {bw:8.1f} | {op:6.1f} | {bg:6.1f}    | {fb+op+bg:7.1f}")
