#!/usr/bin/env bash
# n01-25 ships flydsl 0.1.1.dev409 (egg) which lacks TargetAddressSpace needed by #412.
# Replace it with 0.2.4 (matches the working n03-33 container). C++ build is untouched.
set -uo pipefail
SP=/opt/venv/lib/python3.12/site-packages
PTH="$SP/easy-install.pth"

echo "===== remove old flydsl egg ====="
pip uninstall -y flydsl || true
# egg installed via easy_install: drop its easy-install.pth line + dir if still present
if [ -f "$PTH" ]; then
  sed -i '/flydsl-0.1.1.dev409/d' "$PTH"
fi
rm -rf "$SP"/flydsl-0.1.1.dev409-py3.12-linux-x86_64.egg
echo "remaining flydsl entries in site-packages:"; ls -d "$SP"/flydsl* 2>/dev/null || echo "(none)"

echo; echo "===== install flydsl 0.2.4 ====="
pip install "flydsl==0.2.4" 2>&1 | tail -n 15

echo; echo "===== clear stale ~/.flydsl cache ====="
rm -rf /root/.flydsl; echo cleared

echo; echo "===== verify ====="
python - <<'PY'
import flydsl
print("flydsl", flydsl.__version__, flydsl.__file__)
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace  # the failing symbol
print("TargetAddressSpace OK")
import primus_turbo
print("primus_turbo", primus_turbo.__file__)
import primus_turbo.pytorch          # loads freshly-built _C
import primus_turbo.flydsl.mega      # mega kernels
print("mega import OK")
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__, "devices", torch.cuda.device_count())
PY
