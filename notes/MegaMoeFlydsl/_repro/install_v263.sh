#!/usr/bin/env bash
# Install repo editable into the v26.3 container venv (reuse prebuilt .so, no compile).
set -uo pipefail
REPO=/perf_apps/xiaoming/MegaMoE
LOG=/perf_apps/xiaoming/MegaMoE/slab/notes/MegaMoeFlydsl/logs/v263_install.log
cd "$REPO"
{
  echo "===== uninstall existing primus-turbo ====="
  pip uninstall -y primus-turbo primus_turbo || true

  echo; echo "===== editable install (skip ext build, no deps, no build isolation) ====="
  PRIMUS_TURBO_SKIP_EXT_BUILD=1 pip install -e . --no-build-isolation --no-deps 2>&1 | tail -n 15

  echo; echo "===== pip show ====="
  pip show primus-turbo 2>/dev/null | sed -n '1,3p;/Location/p'

  echo; echo "===== resolve imports + ABI check ====="
  python - <<'PY'
import importlib.util as u
import primus_turbo
print("primus_turbo.__file__   ->", primus_turbo.__file__)
print("primus_turbo.__version__ ->", getattr(primus_turbo, "__version__", "?"))
for mod in ["primus_turbo.pytorch._C",
            "primus_turbo.flydsl.mega",
            "primus_turbo.flydsl.mega.tune_utils",
            "primus_turbo.pytorch.ops.moe.mega_moe_fused"]:
    spec = u.find_spec(mod)
    print(f"{mod:48s} ->", (spec.origin if spec else "NOT FOUND"))
import flydsl
from flydsl.autotune import Autotuner
print("flydsl", flydsl.__version__, "_run_config" , hasattr(Autotuner, "_run_config"))
import torch
print("torch", torch.__version__, "devices", torch.cuda.device_count())
import primus_turbo.pytorch as ptp  # loads _C -> real ABI test
print("import primus_turbo.pytorch OK (ABI ok)")
PY
} 2>&1 | tee "$LOG"
