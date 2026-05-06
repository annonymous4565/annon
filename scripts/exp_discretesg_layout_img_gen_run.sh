#!/bin/bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="$1"
RUN_GROUP="$2"
EXP_NAME="$3"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "=============================================="
echo "Running layout image generation experiment: $EXP_NAME"
echo "Run group: $RUN_GROUP"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# --------------------------------------------------
# Defaults that may vary across runs
# --------------------------------------------------
CKPT="./checkpoints/phase7b3-run2/simple_factorized/best_total.pt"
OUTPUT_DIR="./output/layout_image_gen/${RUN_GROUP}/${EXP_NAME}"


MAX_IMAGES="8"
SHAPE_SOURCE_INDEX="0"

UNCONDITIONAL_STOCHASTIC_OBJ="False"
UNCONDITIONAL_STOCHASTIC_EDGE="False"
UNCONDITIONAL_STOCHASTIC_REL="False"
UNCONDITIONAL_USE_REVERSE_VOCAB_HEADS="True"
UNCONDITIONAL_OBJ_TEMP="1.0"
UNCONDITIONAL_REL_TEMP="1.0"
UNCONDITIONAL_EDGE_LOGIT_THRESHOLD="0.5"
UNCONDITIONAL_RELATION_EDGE_LOGIT_THRESHOLD="0.0"

UNCONDITIONAL_USE_DEGREE_PRUNING="False"
UNCONDITIONAL_MAX_OUT_DEGREE="0"
UNCONDITIONAL_MAX_IN_DEGREE="0"

DRAW_FLOWCHART="True"

SAVE_LAYOUT_BOXES_ONLY="True"
LAYOUT_BOX_IMAGE_SIZE=256


python -m scripts.generate_from_discretesg_layout_text \
    --ckpt "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --max_images "$MAX_IMAGES" \
    --shape_source_index "$SHAPE_SOURCE_INDEX" \
    --unconditional_stochastic_obj "$UNCONDITIONAL_STOCHASTIC_OBJ" \
    --unconditional_stochastic_edge "$UNCONDITIONAL_STOCHASTIC_EDGE" \
    --unconditional_stochastic_rel "$UNCONDITIONAL_STOCHASTIC_REL" \
    --unconditional_use_reverse_vocab_heads "$UNCONDITIONAL_USE_REVERSE_VOCAB_HEADS" \
    --unconditional_obj_temp "$UNCONDITIONAL_OBJ_TEMP" \
    --unconditional_rel_temp "$UNCONDITIONAL_REL_TEMP" \
    --unconditional_edge_logit_threshold "$UNCONDITIONAL_EDGE_LOGIT_THRESHOLD" \
    --unconditional_relation_edge_logit_threshold "$UNCONDITIONAL_RELATION_EDGE_LOGIT_THRESHOLD" \
    --unconditional_use_degree_pruning "$UNCONDITIONAL_USE_DEGREE_PRUNING" \
    --unconditional_max_out_degree "$UNCONDITIONAL_MAX_OUT_DEGREE" \
    --unconditional_max_in_degree "$UNCONDITIONAL_MAX_IN_DEGREE" \
    --draw_flowchart "$DRAW_FLOWCHART" \
    --save_layout_boxes_only "$SAVE_LAYOUT_BOXES_ONLY" \
    --layout_box_image_size "$LAYOUT_BOX_IMAGE_SIZE" 

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Experiment failed: $EXP_NAME"
    exit $EXIT_CODE
fi

echo "Finished experiment: $EXP_NAME"