#!/usr/bin/env bash

# Sudo rm only for paths under workspace; guards against path injection / escaping
safe_sudo_rm() {
    local target="$1"
    local workspace="${2:-$GITHUB_WORKSPACE}"
    if [[ -z "$workspace" || -z "$target" ]]; then return 0; fi
    if [[ "$workspace" != /* ]]; then return 0; fi
    if [[ "$target" != "$workspace"/* ]]; then return 0; fi
    if [[ "$target" == *".."* ]]; then return 0; fi
    sudo rm -rf "$target" 2>/dev/null || true
}

scancel_sync() {
    local jobid=$1
    local timeout=${2:-600}
    local interval=10
    local start
    start=$(date +%s)

    echo "[scancel_sync] Requesting cancel of job $jobid"
    scancel "$jobid" || true

    while [[ -n "$(squeue -j "$jobid" --noheader 2>/dev/null)" ]]; do
        local now
        now=$(date +%s)
        if (( now - start >= timeout )); then
            echo "[scancel_sync][WARN] job $jobid still present after ${timeout}s"
            return 1
        fi
        echo "[scancel_sync] waiting for job $jobid to exit. $((timeout-(now-start))) secs remaining..."
        sleep "$interval"
    done
    echo "[scancel_sync] job $jobid exited"
    return 0
}

if [[ "$IS_MULTINODE" == "true" ]]; then
    # This sets up the environment and launches multi-node benchmarks

    set -x

    # Set up environment variables for SLURM
    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"
    export SLURM_JOB_NAME="benchmark-sglang-disagg.job"

    export MODEL_NAME=${MODEL##*/}
    export MODEL_PATH="/it-share/data"
    export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
    export MORI_RDMA_TC=104

    # Set additional required env vars for multi_node scripts
    export MODEL_DIR="$MODEL_PATH"  # job.slurm uses MODEL_DIR
    export GPUS_PER_NODE=8          # MI355X has 8 GPUs (set to 4 for MI325X)

    export ISL="$ISL"
    export OSL="$OSL"

    # Logs go to BENCHMARK_LOGS_DIR (must be under workspace - no host modification outside)
    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    if [[ -z "$GITHUB_WORKSPACE" || -z "$BENCHMARK_LOGS_DIR" ]] || [[ "$BENCHMARK_LOGS_DIR" != "$GITHUB_WORKSPACE"/* ]]; then
        echo "ERROR: BENCHMARK_LOGS_DIR must be under GITHUB_WORKSPACE. Got BENCHMARK_LOGS_DIR=$BENCHMARK_LOGS_DIR" >&2
        exit 1
    fi
    mkdir -p "$BENCHMARK_LOGS_DIR"
    safe_sudo_rm "$BENCHMARK_LOGS_DIR/logs" "$GITHUB_WORKSPACE"

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi
    JOB_ID=$(bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}")

    # Wait for job to complete
    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"

    # Give slurm time to start the job and create log file
    sleep 10

    # Wait for log file to appear (also check job is still alive)
    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            echo "ERROR: Job $JOB_ID failed before creating log file"
            scontrol show job "$JOB_ID"
            exit 1
        fi
        sleep 5
    done

    set +x

    # Poll for job completion in background
    (
        while squeue -u $USER --noheader --format='%i' | grep -q "$JOB_ID"; do
            sleep 10
        done
    ) &
    POLL_PID=$!

    # Tail the log file until job completes (-F follows by name, polls instead of inotify for NFS)
    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

    wait $POLL_PID

    set -x

    # FIXME: The below is bad and is a result of the indirection of the ways in which
    # Dynamo jobs are launched. In a follow-up PR, the location of the result file should not
    # depend on the runner, it should always be in the same spot in the GH workspace.

    # Process results from all configurations

    # search for "FRAMEWORK_DIFF_IF_STATEMENT #3" for this if-statement
    # Find the latest log directory that contains the data

    COLLECT_SCRIPT=$(mktemp)
    trap "rm -f '$COLLECT_SCRIPT'" EXIT
    cat > "$COLLECT_SCRIPT" <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

    LOGS_DIR=$(python3 "$COLLECT_SCRIPT" "$BENCHMARK_LOGS_DIR" "$ISL" "$OSL" 1)
    if [ -z "$LOGS_DIR" ]; then
        echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"
    ls -la "$LOGS_DIR"

    # Result JSON are contained within the result directory (copy only into workspace)
    for result_file in $(find "$LOGS_DIR" -type f); do
        file_name=$(basename "$result_file")
        if [[ -f "$result_file" && -n "$GITHUB_WORKSPACE" && "$GITHUB_WORKSPACE" == /* ]]; then
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying it to ${WORKSPACE_RESULT_FILE}"
            cp "$result_file" "$WORKSPACE_RESULT_FILE"
        fi
    done

    echo "All result files processed"
    # Use sync scancel to ensure nfs file handle is released in time
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    safe_sudo_rm "$BENCHMARK_LOGS_DIR/logs" "$GITHUB_WORKSPACE"

    # Upload logs as artifact if running in GitHub Actions (workspace only)
    if [[ -n "${GITHUB_ACTIONS:-}" && -n "$GITHUB_WORKSPACE" && "$GITHUB_WORKSPACE" == /* ]]; then
        ARTIFACT_DIR="$GITHUB_WORKSPACE/benchmark_artifacts"
        mkdir -p "$ARTIFACT_DIR"
        cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$ARTIFACT_DIR/" 2>/dev/null || true
        echo "Logs copied to $ARTIFACT_DIR for artifact upload"
    fi

else

    export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/docker/huggingface/hub}"
    export PORT_OFFSET=${RUNNER_NAME: -1}
    export PORT=$(( 8888 + ${PORT_OFFSET} ))
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
    BENCHMARK_SCRIPT="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_mi355x${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
    if [[ -z "$WORKSPACE" || "$WORKSPACE" != /* ]]; then
        echo "ERROR: WORKSPACE must be an absolute path. Got: $WORKSPACE" >&2
        exit 1
    fi
    mkdir -p "$HF_HUB_CACHE_MOUNT"

    if command -v salloc &>/dev/null && command -v srun &>/dev/null && command -v squeue &>/dev/null; then
        # SLURM path: allocate via salloc, run via srun with enroot/squash container
        PARTITION="compute"
        SQUASH_FILE="/var/lib/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
        LOCK_FILE="${SQUASH_FILE}.lock"

        set -x
        salloc --partition=$PARTITION --gres=gpu:$TP --cpus-per-task=128 --time=180 --no-shell --job-name="$RUNNER_NAME"
        JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)
        if [[ -z "$JOB_ID" ]]; then
            echo "ERROR: salloc failed or no job found for $RUNNER_NAME. Check partition=$PARTITION and GPU availability." >&2
            exit 1
        fi

        srun --jobid=$JOB_ID bash -c "docker rm -f bmk-server 2>/dev/null || true"

        # Note: This block runs on the compute node and modifies /var/lib/squash (enroot cache).
        # Only squash files under /var/lib/squash/ are touched - no user data or workspace.
        srun --jobid=$JOB_ID bash -c "
            exec 9>\"$LOCK_FILE\"
            flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
            if [[ \"$SQUASH_FILE\" != /var/lib/squash/* ]]; then exit 1; fi
            if [[ \"$FRAMEWORK\" == \"atom\" ]]; then
                rm -f \"$SQUASH_FILE\"
            fi
            if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
                echo 'Squash file already exists and is valid, skipping import'
            else
                rm -f \"$SQUASH_FILE\"
                enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
            fi
        "

        export VLLM_CACHE_ROOT="/it-share/gharunners/.cache/vllm"

        # Reinstall aiter/sglang in /sgl-workspace when AITER_REF or SGLANG_REF are set, then run benchmark
        RUN_CMD="bash benchmarks/single_node/patch_sgl_components.sh && exec bash $BENCHMARK_SCRIPT"
        srun --jobid=$JOB_ID \
            --container-image=$SQUASH_FILE \
            --container-mounts=$WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
            --container-mount-home \
            --container-writable \
            --container-workdir=/workspace/ \
            --no-container-entrypoint --export=ALL \
            bash -c "$RUN_CMD"

        scancel $JOB_ID

        # Remove gpucore temp files only within workspace (no host modification outside)
        if ls "$WORKSPACE"/gpucore.* 1> /dev/null 2>&1; then
            echo "gpucore files exist. not good"
            for f in "$WORKSPACE"/gpucore.*; do
                [[ -e "$f" ]] && safe_sudo_rm "$f" "$WORKSPACE"
            done
        fi
    else
        # Non-SLURM path: run directly with Docker (no salloc/srun)
        if ! command -v docker &>/dev/null; then
            echo "ERROR: Neither SLURM nor Docker found. Install SLURM (for cluster) or Docker (for standalone)." >&2
            exit 1
        fi
        echo "SLURM not available; using Docker directly."

        server_name="bmk-server"
        docker rm -f "$server_name" 2>/dev/null || true

        set -x
        # Reinstall aiter/sglang in /sgl-workspace when AITER_REF or SGLANG_REF are set, then run benchmark
        RUN_CMD="bash benchmarks/single_node/patch_sgl_components.sh && exec bash $BENCHMARK_SCRIPT"
        docker run --rm --ipc=host --shm-size=16g --network=host --name=$server_name \
            --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
            --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
            -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
            -v "$WORKSPACE:/workspace/" -w /workspace/ \
            -e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e EP_SIZE -e DP_ATTENTION -e CONC \
            -e MAX_MODEL_LEN -e PORT=$PORT -e ISL -e OSL -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
            -e RANDOM_RANGE_RATIO -e RESULT_FILENAME -e RUN_EVAL -e RUNNER_TYPE \
            -e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
            -e SPEC_DECODING -e DISAGG \
            -e AITER_REMOTE -e AITER_REF -e SGLANG_REMOTE -e SGLANG_REF \
            --entrypoint=/bin/bash \
            "$IMAGE" \
            -c "$RUN_CMD"
    fi
fi
