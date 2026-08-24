#!/usr/bin/env bash
# Discovery: environment + prebuilt-.so vs installed-.so comparison.
set -uo pipefail
REPO=/perf_apps/xiaoming/MegaMoE
SP=/opt/venv/lib/python3.12/site-packages

echo "===== python / pip ====="
which python pip
python --version

echo; echo "===== versions ====="
python - <<'PY'
import importlib
for name in ["torch", "triton", "flydsl", "primus_turbo"]:
    try:
        m = importlib.import_module(name)
        print(f"{name:14s} {getattr(m,'__version__','?'):20s} {getattr(m,'__file__','?')}")
    except Exception as e:
        print(f"{name:14s} IMPORT-ERROR: {e}")
PY

echo; echo "===== flydsl.mega present in installed pkg? ====="
python - <<'PY'
import importlib.util as u
for mod in ["primus_turbo.flydsl.mega", "flydsl"]:
    spec = u.find_spec(mod)
    print(mod, "->", spec.origin if spec else "NOT FOUND")
PY

echo; echo "===== prebuilt .so in repo vs installed site-packages (sha256) ====="
for rel in pytorch/_C.cpython-312-x86_64-linux-gnu.so lib/libprimus_turbo_kernels.so; do
  echo "--- $rel ---"
  sha256sum "$REPO/primus_turbo/$rel" 2>/dev/null || echo "  repo:   MISSING"
  sha256sum "$SP/primus_turbo/$rel"   2>/dev/null || echo "  site:   MISSING"
done

echo; echo "===== whoami inside container ====="
whoami; id

echo; echo "===== build deps present? ====="
python -c "import setuptools, setuptools_scm, wheel; print('setuptools', setuptools.__version__); print('setuptools_scm', setuptools_scm.__version__)" 2>&1 | head
