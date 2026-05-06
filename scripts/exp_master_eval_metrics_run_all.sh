#!/bin/bash
set -euo pipefail

RUN_GROUP="master-eval-run1"
LOG_DIR="logs_metrics"

mkdir -p "$LOG_DIR"

echo "Starting metrics evaluation group: $RUN_GROUP"

# ---------------------------------------
# Define experiment -> GPU assignment
# ---------------------------------------
EXP1_NAME="text-sg"
EXP1_GPUS="0"

EXP2_NAME="text-sg-img"
EXP2_GPUS="1"

EXP3_NAME="text-sg-layout"
EXP3_GPUS="2"

EXP4_NAME="text-sg-layout-img"
EXP4_GPUS="3"

PIDS=()

launch_job () {
    local GPUS="$1"
    local EXP_NAME="$2"

    local LOG_FILE="$LOG_DIR/${RUN_GROUP}_${EXP_NAME}_gpus-${GPUS//,/}.log"

    echo "Launching metrics for $EXP_NAME on GPUs $GPUS"
    bash ./scripts/exp_eval_metrics_run.sh "$GPUS" "$RUN_GROUP" "$EXP_NAME" > "$LOG_FILE" 2>&1 &
    local PID=$!
    PIDS+=($PID)

    echo "  PID=$PID log=$LOG_FILE"
}

launch_job "$EXP1_GPUS" "$EXP1_NAME"
launch_job "$EXP2_GPUS" "$EXP2_NAME"
launch_job "$EXP3_GPUS" "$EXP3_NAME"
launch_job "$EXP4_GPUS" "$EXP4_NAME"

echo "Waiting for all jobs..."

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo "One or more metric jobs failed."
    exit 1
fi

echo "All metric evaluations completed successfully."