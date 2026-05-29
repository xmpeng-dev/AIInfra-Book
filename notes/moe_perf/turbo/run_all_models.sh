#!/usr/bin/env bash
# Drive the turbo MoE breakdown benchmark for every MI355X pretrain config
# we care about and dump one CSV per model. Run from anywhere inside the
# xiaoming-dev container with primus_turbo installed:
#
#   bash slab/notes/moe_perf/turbo/run_all_models.sh
#
# Output: $OUT_DIR/<model>-spread-breakdown.csv (one row per --sweep point).
set -u

# --- Backend selection -------------------------------------------------
#
# Set BACKEND=TRITON|HIPBLASLT|CK to control the grouped-GEMM backend
# primus-turbo uses (default: TRITON, the canonical baseline). Autotune
# stays disabled — earlier experiments showed `PRIMUS_TURBO_AUTO_TUNE=1`
# has a global side effect that regresses DeepEP combine latency by
# 10–15%, even though only the `TURBO` backend variant of
# MoEDispatch/Combine passes `can_handle()` (the `DEEP_EP` variant gates
# on a standalone `deep_ep` import that is not installed). See
# `README.md` §Autotune experiment for the analysis. We also observed
# that HIPBLASLT / CK trigger a similar (smaller) combine regression vs
# Triton, so Triton remains the canonical choice unless a per-shape
# study shows otherwise.
BACKEND=${BACKEND:-TRITON}
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=$BACKEND
unset PRIMUS_TURBO_AUTO_TUNE 2>/dev/null || true

# Resolve this script's own directory so BENCH / OUT_DIR are independent
# of the user's cwd.
SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-$SELF_DIR}
mkdir -p "$OUT_DIR"

BENCH=$SELF_DIR/bench_turbo_moe_e2e.py
NP=8
# Autotune samples each backend in the first few calls; give it room.
WARMUP=10
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
    # The bench script auto-derives `<output>_breakdown.csv` from --output-csv;
    # we point it at a scratch file and let it accumulate rows there, then rename.
    RAW="$OUT_DIR/.${slug}_raw.csv"
    RAW_BD="$OUT_DIR/.${slug}_raw_breakdown.csv"
    echo "=================================="
    echo "model=$model H=$H ffn=$FFN E=$E topk=$K deepep_num_cu=$NUMCU"
    echo "  sweep BS = ${SWEEPS[$model]}"
    echo "  csv -> $CSV"
    echo "=================================="

    # Reset the per-model accumulator so re-runs don't append stale rows.
    rm -f "$RAW" "$RAW_BD"

    for BS in ${SWEEPS[$model]}; do
        echo "--- $model BS=$BS ---"
        # 480s timeout: autotune first-call sweeps every backend during
        # warmup, which at BS >= 16k for DSV3 / 49k for DSV2-Lite can take
        # well over 240s on the slowest backend before the timed loop starts.
        timeout 480 torchrun \
            --standalone \
            --nproc-per-node=$NP \
            $BENCH \
            --sweep "$BS" \
            --warmup $WARMUP \
            --iters $ITERS \
            --mode fwd_bwd \
            --routing spread \
            --sync-free-stage 2 \
            --hidden-size $H \
            --moe-ffn-hidden-size $FFN \
            --num-experts $E \
            --topk $K \
            --deepep-num-cu $NUMCU \
            --output-csv "$RAW" 2>&1 \
          | grep -E "^\|\s+$BS\s+\|" | head -2 \
          || echo "FAILED BS=$BS"
        sleep 2
    done

    if [ -s "$RAW_BD" ]; then
        mv "$RAW_BD" "$CSV"
        rm -f "$RAW"
        echo "wrote $CSV"
    else
        echo "WARN: no breakdown rows captured for $model"
    fi
done

echo
echo "DONE [BACKEND=$BACKEND]. CSVs in $OUT_DIR:"
ls -l "$OUT_DIR"/*.csv 2>/dev/null
