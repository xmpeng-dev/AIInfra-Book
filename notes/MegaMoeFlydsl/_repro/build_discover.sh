#!/usr/bin/env bash
# Feasibility check for a from-source build of primus_turbo (#412) in the container.
set -uo pipefail
REPO=/perf_apps/xiaoming/MegaMoE
cd "$REPO"

echo "===== host / user ====="; hostname; whoami

echo; echo "===== toolchain ====="
echo -n "hipcc: "; (command -v hipcc && hipcc --version 2>/dev/null | head -n 2) || echo MISSING
echo -n "cmake: "; (command -v cmake && cmake --version | head -n1) || echo MISSING
echo -n "ninja: "; (command -v ninja && ninja --version) || echo MISSING
echo -n "ROCM_HOME/ROCM_PATH: "; echo "${ROCM_HOME:-} ${ROCM_PATH:-}"
ls -d /opt/rocm* 2>/dev/null
echo -n "python setuptools_scm: "; python -c "import setuptools_scm,setuptools;print('setuptools',setuptools.__version__)" 2>&1 | head -n1

echo; echo "===== gpu arch ====="
(rocminfo 2>/dev/null | grep -m1 -i gfx) || echo "rocminfo n/a"

echo; echo "===== resources ====="
echo "nproc: $(nproc)"; free -g | head -n2

echo; echo "===== git + submodule state ====="
git -C "$REPO" rev-parse --short HEAD 2>&1 | head -n1
git -C "$REPO" branch --show-current 2>&1 | head -n1
echo "3rdparty/composable_kernel entries: $(ls -A 3rdparty/composable_kernel 2>/dev/null | wc -l)"
echo "3rdparty/hipify_torch entries: $(ls -A 3rdparty/hipify_torch 2>/dev/null | wc -l)"
git -C "$REPO" submodule status 2>&1 | head -n 10

echo; echo "===== github reachability (submodule fetch) ====="
git ls-remote --heads https://github.com/ROCm/composable_kernel.git 2>&1 | head -n 2 || echo "NO NETWORK"

echo; echo "===== existing prebuilt .so (will be overwritten by build) ====="
ls -la primus_turbo/pytorch/*.so primus_turbo/lib/*.so 2>/dev/null
