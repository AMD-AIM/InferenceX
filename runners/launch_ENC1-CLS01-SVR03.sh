#!/usr/bin/bash

# Launch script for ENC1-CLS01-SVR03 (local MI300X runner)
# Paths adapted for device: /mnt/nvme8n1p1 for cache, /workspace for build

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# HF cache: use NVMe storage, override via HF_HUB_CACHE_MOUNT env if needed
HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/mnt/nvme1n1p1/huggingface/hub/}"
mkdir -p "$HF_HUB_CACHE_MOUNT"
PORT=8888

server_name="bmk-server"

set -x
docker run --rm --ipc=host --shm-size=16g --network=host --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e ISL -e OSL -e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e RANDOM_RANGE_RATIO -e RESULT_FILENAME -e RUN_EVAL -e RUNNER_TYPE \
-e WORK_DIR=/workspace \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/single_node/"${EXP_NAME%%_*}_${PRECISION}_mi300x.sh"
