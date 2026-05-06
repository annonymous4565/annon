#!/bin/bash

CUDA_VISIBLE_DEVICES="$1"
RUN_GROUP="$2"
EXP_NAME="$3"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

NUM_GPUS=$(python - <<EOF
gpus = "${CUDA_VISIBLE_DEVICES}".split(",")
print(len(gpus))
EOF
)

echo "=============================================="
echo "Running experiment: $EXP_NAME"
echo "Run group: $RUN_GROUP"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Num GPUs: $NUM_GPUS"
echo "=============================================="

# Only configs that change across runs
REL_WEIGHTING_STRATEGY="simple"
REL_EFFECTIVE_NUM_BETA="0.99"
NO_REL_LOSS_WEIGHT="0.1"
REL_WEIGHT_MAX="5.0"
USE_NEGATIVE_EDGE_SAMPLING="True"
NEG_EDGE_SAMPLE_STRATEGY="ratio"
NEG_POS_RATIO="3.0"
LAMBDA_REL="5.0"
NUM_EPOCHS="15"
CKPT="./checkpoints/phase5a-run5-full-rev-sampling/simple_factorized/best_total.pt"
TRAIN_MODE="False"

WANDB_RUN_NAME="${RUN_GROUP}_${EXP_NAME}"
CHECKPOINT_DIR="./checkpoints/${RUN_GROUP}/${EXP_NAME}"
FLOWCHART_OUT_DIR="./visualization/sg_flowcharts/${RUN_GROUP}/${EXP_NAME}"

DEMON_SEL_MODE="argmax"
TEXT_PROMPT="a person riding a horse"

case "$EXP_NAME" in
    factorized_edge)
        REL_WEIGHTING_STRATEGY="effective_num"
        REL_EFFECTIVE_NUM_BETA="0.99"
        USE_NEGATIVE_EDGE_SAMPLING="True"
        NEG_EDGE_SAMPLE_STRATEGY="ratio"
        NEG_POS_RATIO="3.0"
        NO_REL_LOSS_WEIGHT="0.2"
        REL_WEIGHT_MAX="5.0"
        LAMBDA_REL="5.0"
        ;;

    simple_factorized)
        REL_WEIGHTING_STRATEGY="simple"
        USE_NEGATIVE_EDGE_SAMPLING="True"
        NEG_EDGE_SAMPLE_STRATEGY="ratio"
        NEG_POS_RATIO="3.0"
        NO_REL_LOSS_WEIGHT="0.1"
        LAMBDA_REL="5.0"
        ;;
    *)
        echo "Unknown experiment name: $EXP_NAME"
        exit 1
        ;;
esac

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.text_run_full_reverse \
    --rel_weighting_strategy "$REL_WEIGHTING_STRATEGY" \
    --rel_effective_num_beta "$REL_EFFECTIVE_NUM_BETA" \
    --no_rel_loss_weight "$NO_REL_LOSS_WEIGHT" \
    --use_negative_edge_sampling "$USE_NEGATIVE_EDGE_SAMPLING" \
    --neg_edge_sample_strategy "$NEG_EDGE_SAMPLE_STRATEGY" \
    --neg_pos_ratio "$NEG_POS_RATIO" \
    --lambda_rel "$LAMBDA_REL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --num_epochs "$NUM_EPOCHS" \
    --flowchart_out_dir "$FLOWCHART_OUT_DIR" \
    --ckpt "$CKPT" \
    --train_mode "$TRAIN_MODE" \
    --demon_selection_mode "$DEMON_SEL_MODE" \
    --text_prompt "$TEXT_PROMPT"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Experiment failed: $EXP_NAME"
    exit $EXIT_CODE
fi

echo "Finished experiment: $EXP_NAME"