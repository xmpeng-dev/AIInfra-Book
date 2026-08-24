#!/usr/bin/env bash
set -uo pipefail
echo "===== pip show ====="
pip show primus-turbo | sed -n '1,3p;/Location/p'

echo; echo "===== resolve imports ====="
python - <<'PY'
import importlib, importlib.util as u
import primus_turbo
print("primus_turbo.__file__ ->", primus_turbo.__file__)
print("primus_turbo.__version__ ->", getattr(primus_turbo, "__version__", "?"))
for mod in ["primus_turbo.pytorch._C",
            "primus_turbo.flydsl.mega",
            "primus_turbo.flydsl.mega.tune_utils",
            "primus_turbo.pytorch.ops.moe.mega_moe_fused"]:
    spec = u.find_spec(mod)
    print(f"{mod:48s} ->", (spec.origin if spec else "NOT FOUND"))

import torch
print("torch.cuda.device_count ->", torch.cuda.device_count())
import primus_turbo.pytorch as ptp  # noqa
print("import primus_turbo.pytorch OK")
PY
