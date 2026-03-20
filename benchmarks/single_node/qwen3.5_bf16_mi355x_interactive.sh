#!/usr/bin/env bash
#
# Interactive benchmark script for Qwen3.5-397B-A17B on MI355X.
# Runs inside Docker with CLI args for TP, sequence lengths, and concurrency.
#
# Usage:
#   ./qwen3.5_bf16_mi355x_interactive.sh [OPTIONS]
#
# Options:
#   -tp N              Tensor parallel size (default: 8)
#   -sl ISL,OSL ...    Space-separated isl,osl pairs (default: 1024,1024 8192,1024 1024,8192)
#   -conc N N ...      Space-separated concurrency values (default: 8)
#   -result-dir DIR    Output directory (default: /workspace/)
#
# Examples:
#   ./qwen3.5_bf16_mi355x_interactive.sh
#   ./qwen3.5_bf16_mi355x_interactive.sh -tp 8 -sl 1024,1024 8192,1024 -conc 8 16 32
#   ./qwen3.5_bf16_mi355x_interactive.sh -result-dir /workspace/results

set -e

# Defaults
TP=8
SL_LIST=("1024,1024" "8192,1024" "1024,8192")
CONC_LIST=(8)
RESULT_DIR="/workspace/"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -tp N              Tensor parallel size (default: 8)"
    echo "  -sl ISL,OSL ...    Space-separated isl,osl pairs (default: 1024,1024 8192,1024 1024,8192)"
    echo "  -conc N N ...      Space-separated concurrency values (default: 8)"
    echo "  -result-dir DIR   Output directory (default: /workspace/)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 -tp 8 -sl 1024,1024 8192,1024 -conc 8 16 32"
    echo "  $0 -result-dir /workspace/results"
    exit 1
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        -tp)
            TP="$2"
            shift 2
            ;;
        -sl)
            shift
            SL_LIST=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                SL_LIST+=("$1")
                shift
            done
            [[ ${#SL_LIST[@]} -eq 0 ]] && SL_LIST=("1024,1024" "8192,1024" "1024,8192")
            ;;
        -conc)
            shift
            CONC_LIST=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                CONC_LIST+=("$1")
                shift
            done
            [[ ${#CONC_LIST[@]} -eq 0 ]] && CONC_LIST=(8)
            ;;
        -result-dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate TP
if ! [[ "$TP" =~ ^[0-9]+$ ]]; then
    echo "Error: -tp must be a positive integer, got: $TP"
    exit 1
fi

# Validate each sl pair (isl,osl)
for pair in "${SL_LIST[@]}"; do
    if ! [[ "$pair" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "Error: -sl pair must be isl,osl (e.g. 1024,1024), got: $pair"
        exit 1
    fi
done

# Validate each conc
for c in "${CONC_LIST[@]}"; do
    if ! [[ "$c" =~ ^[0-9]+$ ]]; then
        echo "Error: -conc values must be positive integers, got: $c"
        exit 1
    fi
done

# Ensure result dir exists and ends with /
[[ "${RESULT_DIR}" != */ ]] && RESULT_DIR="${RESULT_DIR}/"
mkdir -p "$RESULT_DIR"

# Set optional env vars
export MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.1}"

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars MODEL RANDOM_RANGE_RATIO

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "Config: TP=$TP, SL=${SL_LIST[*]}, CONC=${CONC_LIST[*]}, RESULT_DIR=$RESULT_DIR"

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Start GPU monitoring
start_gpu_monitor

python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --mem-fraction-static 0.8 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Loop over (isl,osl) and conc
for sl_pair in "${SL_LIST[@]}"; do
    IFS=',' read -r ISL OSL <<< "$sl_pair"
    for CONC in "${CONC_LIST[@]}"; do
        RESULT_FILENAME="result_TP${TP}_CONC${CONC}_ISL${ISL}_OSL${OSL}.json"
        echo "Running: ISL=$ISL OSL=$OSL CONC=$CONC -> $RESULT_FILENAME"

        run_benchmark_serving \
            --model "$MODEL" \
            --port "$PORT" \
            --backend vllm \
            --input-len "$ISL" \
            --output-len "$OSL" \
            --random-range-ratio "$RANDOM_RANGE_RATIO" \
            --num-prompts "$((CONC * 10))" \
            --max-concurrency "$CONC" \
            --result-filename "$RESULT_FILENAME" \
            --result-dir "$RESULT_DIR"

        if [ "${RUN_EVAL}" = "true" ]; then
            run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
            append_lm_eval_summary
        fi
    done
done

stop_gpu_monitor
echo "Done. Results in $RESULT_DIR"
echo ""
echo "To summarize results into a markdown table, run (from repo root):"
echo "  python3 utils/summarize_interactive_results.py $RESULT_DIR -o summary.md"
