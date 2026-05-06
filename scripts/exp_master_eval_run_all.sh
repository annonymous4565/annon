#!/bin/bash
set -euo pipefail

# --------------------------------------------------
# Hardcoded config (no CLI args)
# --------------------------------------------------
RUN_GROUP="master-eval-run1"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "Starting master evaluation group: $RUN_GROUP"

# ---------------------------------------
# Define experiment -> GPU assignment
# ---------------------------------------

EXP1_NAME="coco_text_sg"
EXP1_GPUS="0"

EXP2_NAME="coco_text_sg_img"
EXP2_GPUS="1"

EXP3_NAME="coco_text_sg_layout"
EXP3_GPUS="2"

EXP4_NAME="coco_text_sg_layout_img"
EXP4_GPUS="3"

# Uncomment if needed
# EXP5_NAME="vg_text_sg"
# EXP5_GPUS="0"

# EXP6_NAME="vg_text_sg_img"
# EXP6_GPUS="1"

# EXP7_NAME="vg_text_sg_layout"
# EXP7_GPUS="2"

# EXP8_NAME="vg_text_sg_layout_img"
# EXP8_GPUS="3"

PIDS=()

launch_job () {
    local GPUS="$1"
    local EXP_NAME="$2"

    local LOG_FILE="$LOG_DIR/${RUN_GROUP}_${EXP_NAME}_gpus-${GPUS//,/}.log"

    echo "Launching $EXP_NAME on GPUs $GPUS"
    bash ./scripts/exp_master_eval_run.sh "$GPUS" "$RUN_GROUP" "$EXP_NAME" > "$LOG_FILE" 2>&1 &
    local PID=$!
    PIDS+=($PID)
    echo "  PID=$PID log=$LOG_FILE"
}

launch_job "$EXP1_GPUS" "$EXP1_NAME"
launch_job "$EXP2_GPUS" "$EXP2_NAME"
launch_job "$EXP3_GPUS" "$EXP3_NAME"
launch_job "$EXP4_GPUS" "$EXP4_NAME"

# launch_job "$EXP5_GPUS" "$EXP5_NAME"
# launch_job "$EXP6_GPUS" "$EXP6_NAME"
# launch_job "$EXP7_GPUS" "$EXP7_NAME"
# launch_job "$EXP8_GPUS" "$EXP8_NAME"

echo "Waiting for all launched jobs..."
FAIL=0

for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo "One or more evaluations failed."
    exit 1
fi

echo "All evaluations completed successfully."