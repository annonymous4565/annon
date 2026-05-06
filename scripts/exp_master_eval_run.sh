#!/bin/bash

CUDA_VISIBLE_DEVICES="$1"
RUN_GROUP="$2"
EXP_NAME="$3"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "=============================================="
echo "Running master eval: $EXP_NAME"
echo "Run group: $RUN_GROUP"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# --------------------------------------------------
# Only configs that change across modes
# --------------------------------------------------
EVAL_DATASET_TYPE="coco"
EVAL_BATCH_SIZE="1"
MODE="text-sg"

CKPT="./checkpoints/phase5c-run3-training-rev-sampler-deterministic/simple_factorized/best_total.pt"
OUTPUT_ROOT="./output/master_eval/${RUN_GROUP}/${EXP_NAME}"

# --------------------------------------------------
# Mode selection
# --------------------------------------------------
case "$EXP_NAME" in
    coco_text_sg)
        EVAL_DATASET_TYPE="coco"
        MODE="text-sg"
        ;;

    coco_text_sg_img)
        EVAL_DATASET_TYPE="coco"
        MODE="text-sg-img"
        ;;

    coco_text_sg_layout)
        EVAL_DATASET_TYPE="coco"
        MODE="text-sg-layout"
        ;;

    coco_text_sg_layout_img)
        EVAL_DATASET_TYPE="coco"
        MODE="text-sg-layout-img"
        ;;

    vg_text_sg)
        EVAL_DATASET_TYPE="vg"
        MODE="text-sg"
        ;;

    vg_text_sg_img)
        EVAL_DATASET_TYPE="vg"
        MODE="text-sg-img"
        ;;

    vg_text_sg_layout)
        EVAL_DATASET_TYPE="vg"
        MODE="text-sg-layout"
        ;;

    vg_text_sg_layout_img)
        EVAL_DATASET_TYPE="vg"
        MODE="text-sg-layout-img"
        ;;
    *)
        echo "Unknown experiment name: $EXP_NAME"
        exit 1
        ;;
esac

python -m scripts.evaluate_master \
    --eval_dataset_type "$EVAL_DATASET_TYPE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --mode "$MODE" \
    --output_root "$OUTPUT_ROOT" \
    --ckpt "$CKPT"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Evaluation failed: $EXP_NAME"
    exit $EXIT_CODE
fi

echo "Finished evaluation: $EXP_NAME"