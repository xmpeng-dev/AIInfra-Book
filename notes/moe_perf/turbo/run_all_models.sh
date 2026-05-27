#!/usr/bin/env bash
# Drive the turbo MoE breakdown benchmark for every MI355X pretrain config
# we care about and dump one CSV per model. Run from anywhere inside the
# xiaoming-dev container with primus_turbo installed:
#
#   bash slab/notes/moe_perf/turbo/run_all_models.sh
#
# Output: $OUT_DIR/<model>-spread-breakdown.csv (one row per --sweep point).
set -u

export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON

# Resolve this script's own directory so BENCH / OUT_DIR are independent
# of the user's cwd.
SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-$SELF_DIR}
mkdir -p "$OUT_DIR"

BENCH=$SELF_DIR/bench_turbo_moe_e2e.py
NP=8
WARMUP=6
ITERS=25

# Per-model shape (hidden ffn experts topk deepep_cu).
# Production num_tokens per rank = mbs * seq_length.
declare -A MODELS
MODELS[deepseek_v2_lite]="2048 1408 64 6 64"     # mbs=12 seq=4096 -> 49152
MODELS[deepseek_v3]="7168 2048 256 8 80"          # mbs=2  seq=4096 -> 8192
MODELS[qwen3_30B_A3B]="2048 768 128 8 80"         # mbs=8  seq=4096 -> 32768
MODELS[qwen3_235B_A22B]="4096 1536 128 8 80"      # mbs=4  seq=4096 -> 16384

# Sweep covers each model's production point (marked PROD below) plus
# surrounding sizes so we can read the scaling. Sub-256 / sub-1024 hits DeepEP
# intranode-buffer edge cases at small EP and is skipped.
declare -A SWEEPS
SWEEPS[deepseek_v2_lite]="4096 8192 16384 32768 49152"   # PROD = 49152
SWEEPS[deepseek_v3]="1024 2048 4096 8192 16384"          # PROD = 8192
SWEEPS[qwen3_30B_A3B]="2048 4096 8192 16384 32768"       # PROD = 32768
SWEEPS[qwen3_235B_A22B]="2048 4096 8192 16384 32768"     # PROD = 16384

for model in deepseek_v2_lite deepseek_v3 qwen3_30B_A3B qwen3_235B_A22B; do
    read -r H FFN E K NUMCU <<<"${MODELS[$model]}"
    # kebab-case CSV name matching slab/notes conventions.
    slug=$(printf '%s' "$model" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
    CSV="$OUT_DIR/${slug}-spread-breakdown.csv"
    echo "=================================="
    echo "model=$model H=$H ffn=$FFN E=$E topk=$K deepep_num_cu=$NUMCU"
    echo "  sweep BS = ${SWEEPS[$model]}"
    echo "  csv -> $CSV"
    echo "=================================="

    {
        echo "model,num_tokens,time_us,tflops,gbps,sort_us,dispatch_us,fused_moe_us,combine_us,misc_us,all_kernels_us"
    } >"$CSV"

    for BS in ${SWEEPS[$model]}; do
        echo "--- $model BS=$BS ---"
        # Capture only the table body line ("|  <BS> | ...").
        OUT=$(timeout 180 torchrun \
            --standalone \
            --nproc-per-node=$NP \
            $BENCH \
            --sweep "$BS" \
            --warmup $WARMUP \
            --iters $ITERS \
            --routing spread \
            --sync-free-stage 2 \
            --hidden-size $H \
            --moe-ffn-hidden-size $FFN \
            --num-experts $E \
            --topk $K \
            --deepep-num-cu $NUMCU 2>&1)

        # Find the breakdown line. Should look like:
        # |       8192 |    12500.6 |            461.8 |  ...
        ROW=$(printf '%s\n' "$OUT" | grep -E "^\|\s+$BS\s+\|" | head -1)
        if [ -z "$ROW" ]; then
            echo "FAILED BS=$BS"
            printf '%s\n' "$OUT" | tail -8
            echo "$model,$BS,,,,,,,,," >>"$CSV"
            continue
        fi

        echo "$ROW"
        # Parse "|       8192 | 12500.6 | 461.8 | 591 | 11.3 | 2564.8 | 6270.5 | 2571.8 | 1014.8 | 12433.1 |"
        VALS=$(printf '%s\n' "$ROW" | sed -E 's/^\|//; s/\|$//' | tr '|' ',' | sed -E 's/ //g')
        echo "$model,$VALS" >>"$CSV"
        # Brief pause to let GPU rings drain between runs.
        sleep 2
    done
done

echo
echo "DONE. CSVs in $OUT_DIR:"
ls -l "$OUT_DIR"
