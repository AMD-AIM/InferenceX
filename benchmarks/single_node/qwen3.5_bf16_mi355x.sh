#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MEM_FRAC_STATIC=0.82
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
CUDA_GRAPH_MAX_BATCH_SIZE=$CONC
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=$((ISL + OSL + 20))

# Default: recv every ~10 requests; if CONC ≥ 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

export SGLANG_FUSED_QK_NORM_ROPE_CACHE_PTS_QUANT_SHUFFLE=1

cd /sgl-workspace/sglang
set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --mem-fraction-static $MEM_FRAC_STATIC \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --enable-aiter-allreduce-fusion \
    --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
    --tokenizer-worker-num 6 \
    --stream-interval 30 \
    --context-length $CONTEXT_LENGTH > $SERVER_LOG 2>&1 &

SERVER_PID=$!
cd -
# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
