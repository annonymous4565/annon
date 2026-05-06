#!/bin/bash

RUN_GROUP="${1:-phase2-run1}"
LOG_DIR="${2:-logs}"

mkdir -p "$LOG_DIR"

echo "Starting parallel DiscreteSG experiment group: $RUN_GROUP"

# ---------------------------------------
# Define experiment -> GPU set assignment
# ---------------------------------------

EXP1_NAME="simple_factorized"
EXP1_GPUS="0"

# EXP2_NAME="baseline_effective_num_neg_0.3"
# EXP2_GPUS="4,5"

# EXP3_NAME="inverse_freq_neg"
# EXP3_GPUS="4,5"

# EXP4_NAME="baseline_effective_num_neg_0.2_max"
# EXP4_GPUS="6,7"

PIDS=()

launch_job () {
    local GPUS="$1"
    local EXP_NAME="$2"

    local LOG_FILE="$LOG_DIR/${RUN_GROUP}_${EXP_NAME}_gpus-${GPUS//,/}.log"

    echo "Launching $EXP_NAME on GPUs $GPUS"
    bash ./scripts/exp_discretesg_eval_text_run.sh "$GPUS" "$RUN_GROUP" "$EXP_NAME" > "$LOG_FILE" 2>&1 &
    local PID=$!
    PIDS+=($PID)
    echo "  PID= $PID log=$LOG_FILE"
}

launch_job "$EXP1_GPUS" "$EXP1_NAME"
# launch_job "$EXP2_GPUS" "$EXP2_NAME"
# launch_job "$EXP3_GPUS" "$EXP3_NAME"
# launch_job "$EXP4_GPUS" "$EXP4_NAME"

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