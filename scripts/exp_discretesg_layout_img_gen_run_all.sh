#!/bin/bash
set -euo pipefail

RUN_GROUP="${1:-phase2-run1}"
LOG_DIR="${2:-logs}"

mkdir -p "$LOG_DIR"

echo "Starting layout-image-generation experiment group: $RUN_GROUP"

# ---------------------------------------
# Define experiment -> GPU assignment
# ---------------------------------------
EXP1_NAME="simple_factorized"
EXP1_GPUS="0"

# EXP2_NAME="factorized_edge"
# EXP2_GPUS="1"

PIDS=()

launch_job () {
    local GPUS="$1"
    local EXP_NAME="$2"

    local LOG_FILE="$LOG_DIR/${RUN_GROUP}_${EXP_NAME}_gpus-${GPUS//,/}.log"

    echo "Launching $EXP_NAME on GPUs $GPUS"
    bash ./scripts/exp_discretesg_layout_img_gen_run.sh "$GPUS" "$RUN_GROUP" "$EXP_NAME" > "$LOG_FILE" 2>&1 &
    local PID=$!
    PIDS+=($PID)
    echo "  PID=$PID log=$LOG_FILE"
}

launch_job "$EXP1_GPUS" "$EXP1_NAME"
# launch_job "$EXP2_GPUS" "$EXP2_NAME"

echo "Waiting for all launched jobs..."
FAIL=0

for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo "One or more experiments failed."
    exit 1
fi

echo "All experiments completed successfully."