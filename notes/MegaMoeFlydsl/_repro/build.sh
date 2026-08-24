#!/usr/bin/env bash
# From-source build of primus_turbo (#412) in the container (compiles CK/HIP csrc).
set -uo pipefail
REPO=/perf_apps/xiaoming/MegaMoE
LOGS="$REPO/slab/notes/MegaMoeFlydsl/logs"
LOG="$LOGS/build.log"
MARKER="$LOGS/build_DONE.marker"
mkdir -p "$LOGS"
rm -f "$MARKER"
cd "$REPO"

{
  echo "############ BUILD START $(date -u +%Y-%m-%dT%H:%M:%SZ) ############"
  git config --global --add safe.directory "$REPO"

  echo "===== git state ====="
  git rev-parse --short HEAD; git branch --show-current

  echo; echo "===== submodule init (CK + hipify_torch) ====="
  git submodule sync --recursive
  git submodule update --init --recursive
  echo "composable_kernel entries: $(ls -A 3rdparty/composable_kernel 2>/dev/null | wc -l)"
  echo "hipify_torch entries: $(ls -A 3rdparty/hipify_torch 2>/dev/null | wc -l)"

  echo; echo "===== uninstall old primus-turbo ====="
  pip uninstall -y primus-turbo primus_turbo || true

  echo; echo "===== from-source editable build (compiles ext; NO skip flag) ====="
  echo "start compile: $(date -u +%H:%M:%SZ)"
  MAX_JOBS=128 PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 \
    pip install -e . --no-build-isolation --no-deps -v
  rc=$?
  echo "pip rc=$rc  end: $(date -u +%H:%M:%SZ)"

  echo; echo "===== built artifacts ====="
  ls -la primus_turbo/pytorch/*.so primus_turbo/lib/*.so 2>/dev/null

  echo; echo "===== import + ABI check ====="
  python - <<'PY'
import importlib.util as u
import primus_turbo
print("primus_turbo.__file__ ->", primus_turbo.__file__)
for mod in ["primus_turbo.pytorch._C","primus_turbo.flydsl.mega","primus_turbo.pytorch.ops.moe.mega_moe_fused"]:
    spec = u.find_spec(mod); print(f"{mod:44s} ->", (spec.origin if spec else "NOT FOUND"))
import flydsl, torch
print("flydsl", flydsl.__version__, "torch", torch.__version__, "devices", torch.cuda.device_count())
import primus_turbo.pytorch as ptp
print("import primus_turbo.pytorch OK (ABI ok)")
PY
  echo "BUILD exit=$rc" >> "$MARKER"
  echo "############ BUILD END $(date -u +%Y-%m-%dT%H:%M:%SZ) ############"
} 2>&1 | tee "$LOG"
