#!/bin/bash
# Full sharded comparison: 6 ops x {TRITON, FLYDSL}, each split 8 ways across the 8 MI355X.
# Pairs run one at a time so a GPU never hosts two timed processes at once.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/perf_apps/xiaoming/MegaMoE

declare -A ENVKEY=(
  [gemm_fp8_tw]=PRIMUS_TURBO_GEMM_BACKEND
  [gg_bf16]=PRIMUS_TURBO_GROUPED_GEMM_BACKEND
  [gg_fp8_tw]=PRIMUS_TURBO_GROUPED_GEMM_BACKEND
  [gg_fp8_mx]=PRIMUS_TURBO_GROUPED_GEMM_BACKEND
  [gg_fp4_mx]=PRIMUS_TURBO_GROUPED_GEMM_BACKEND
  [sparse_mla]=PRIMUS_TURBO_SPARSE_ATTN_BACKEND
)
OPS=(gg_bf16 gg_fp8_tw gg_fp8_mx gg_fp4_mx gemm_fp8_tw sparse_mla)
NSHARD=8

mkdir -p results logs
for op in "${OPS[@]}"; do
  for be in TRITON FLYDSL; do
    echo "=== $op / $be  ($(date +%H:%M:%S)) ==="
    for s in $(seq 0 $((NSHARD - 1))); do
      HIP_VISIBLE_DEVICES=$s env "${ENVKEY[$op]}=$be" \
        python bench_triton_vs_flydsl.py --op "$op" --backend "$be" \
          --num-shards $NSHARD --shard-id "$s" \
          -o "results/${op}_${be}.part-${s}.csv" \
          >"logs/${op}_${be}.shard${s}.log" 2>&1 &
    done
    wait
    ok=$(grep -l '^saved ' logs/${op}_${be}.shard*.log 2>/dev/null | wc -l)
    echo "    shards finished: $ok/$NSHARD"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
