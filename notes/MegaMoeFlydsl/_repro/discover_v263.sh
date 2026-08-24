#!/usr/bin/env bash
# Discovery for the new rocm/primus:v26.3 container.
set -uo pipefail
REPO=/perf_apps/xiaoming/MegaMoE

echo "===== host / whoami ====="
hostname; whoami; id

echo; echo "===== python / pip ====="
which python pip
python --version
python -c "import sys; print('abi tag guess: cpython-%d%d' % sys.version_info[:2])"

echo; echo "===== versions ====="
python - <<'PY'
import importlib
for name in ["torch", "triton", "flydsl", "primus_turbo"]:
    try:
        m = importlib.import_module(name)
        print(f"{name:14s} {str(getattr(m,'__version__','?')):22s} {getattr(m,'__file__','?')}")
    except Exception as e:
        print(f"{name:14s} IMPORT-ERROR: {type(e).__name__}: {e}")
PY

echo; echo "===== flydsl API: has _run_config on Autotuner? ====="
python - <<'PY'
try:
    from flydsl.autotune import Autotuner
    print("Autotuner methods:", [m for m in ("_run_config","_run_with_hints") if hasattr(Autotuner, m)])
except Exception as e:
    print("flydsl.autotune import error:", e)
PY

echo; echo "===== is flydsl.mega present in the INSTALLED primus_turbo? ====="
python - <<'PY'
import importlib.util as u
for mod in ["primus_turbo.flydsl.mega", "primus_turbo.pytorch._C"]:
    try:
        spec = u.find_spec(mod)
        print(f"{mod:34s} ->", (spec.origin if spec else "NOT FOUND"))
    except Exception as e:
        print(f"{mod:34s} -> ERROR {e}")
PY

echo; echo "===== primus_turbo install location / .so tags ====="
python - <<'PY'
try:
    import primus_turbo, os, glob
    root = os.path.dirname(primus_turbo.__file__)
    print("primus_turbo root:", root)
    for so in glob.glob(os.path.join(root, "**", "*.so"), recursive=True):
        print("  ", so)
except Exception as e:
    print("err:", e)
PY

echo; echo "===== repo mounted + repo .so tags ====="
ls -d "$REPO" >/dev/null 2>&1 && echo "repo mounted: YES" || echo "repo mounted: NO"
echo "repo .so files:"; ls -la "$REPO"/primus_turbo/pytorch/*.so "$REPO"/primus_turbo/lib/*.so 2>/dev/null

echo; echo "===== torch cuda ====="
python -c "import torch; print('torch', torch.__version__, 'devices', torch.cuda.device_count())" 2>&1 | head

echo; echo "===== pip show primus-turbo ====="
pip show primus-turbo 2>/dev/null | sed -n '1,3p;/Location/p' || echo "not pip-visible"
