#!/usr/bin/env bash
# n01-25: run both fused mega-MoE benchmarks on the FROM-SOURCE build (flydsl 0.2.4).
set -uo pipefail
REPO=/perf_apps/xiaoming/MegaMoE
BENCH="$REPO/benchmark/ops/training"
LOGS="$REPO/slab/notes/MegaMoeFlydsl/logs"
STAMP=20260717_n0125src
MARKER="$LOGS/n0125_src_DONE.marker"
rm -f "$MARKER"; mkdir -p "$LOGS"
cd "$BENCH"
export PYTORCH_ROCM_ARCH=gfx950

run() {
  local mode="$1"
  local log="$LOGS/n0125_src_${mode}.log"
  echo "############ MODE=$mode START $(date -u +%Y-%m-%dT%H:%M:%SZ) ############" | tee "$log"
  timeout 2400 python bench_mega_moe.py --mode "$mode" --models DeepSeek-V3 --num-processes 8 \
      -o "$LOGS/${mode}_${STAMP}_MI355X.csv" >>"$log" 2>&1
  local rc=$?
  echo "############ MODE=$mode EXIT=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ############" | tee -a "$log"
  echo "$mode exit=$rc" >> "$MARKER"
}

run dispatch_grouped_gemm
run grouped_gemm_combine
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MARKER"
