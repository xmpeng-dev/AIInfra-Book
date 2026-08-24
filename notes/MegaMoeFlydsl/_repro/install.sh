#!/usr/bin/env bash
# Install the repo checkout into the container venv (editable, reuse prebuilt .so).
set -euo pipefail
REPO=/perf_apps/xiaoming/MegaMoE
cd "$REPO"

echo "===== uninstall existing primus-turbo (dev23) ====="
pip uninstall -y primus-turbo || true

echo; echo "===== editable install (skip ext build, no deps, no build isolation) ====="
PRIMUS_TURBO_SKIP_EXT_BUILD=1 pip install -e . --no-build-isolation --no-deps -v 2>&1 | tail -n 40

echo; echo "===== pip show ====="
pip show primus-turbo | head -n 6

echo; echo "===== resolve import ====="
python - <<'PY'
import primus_turbo, importlib.util as u
print("primus_turbo.__file__ ->", primus_turbo.__file__)
for mod in ["primus_turbo.pytorch._C", "primus_turbo.flydsl.mega", "primus_turbo.flydsl.mega.tune_utils"]:
    spec = u.find_spec(mod)
    print(f"{mod:40s} ->", (spec.origin if spec else "NOT FOUND"))
import primus_turbo.pytorch as ptp
print("import primus_turbo.pytorch OK")
PY
