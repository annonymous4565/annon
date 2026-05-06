#!/bin/bash

CUDA_VISIBLE_DEVICES="$1"
RUN_GROUP="$2"
MODE="$3"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

NUM_GPUS=$(python - <<EOF
gpus = "${CUDA_VISIBLE_DEVICES}".split(",")
print(len(gpus))
EOF
)

echo "=============================================="
echo "Running metrics for mode: $MODE"
echo "Run group: $RUN_GROUP"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Num GPUs: $NUM_GPUS"
echo "=============================================="

# ---------------------------------------
# Core paths
# ---------------------------------------
ROOT_DIR="./output/master_eval/${RUN_GROUP}"
MODE_DIR="${ROOT_DIR}/${MODE}"

GT_SG_DIR="${ROOT_DIR}/gt_sg"
GT_LAYOUT_DIR="${ROOT_DIR}/gt_layouts"
GT_IMAGE_DIR="${ROOT_DIR}/gt_images"

# ---------------------------------------
# Mode-specific toggles
# ---------------------------------------
COMPUTE_GRAPH="True"
COMPUTE_LAYOUT="True"
COMPUTE_IMAGE="True"

case "$MODE" in
    text-sg)
        COMPUTE_GRAPH="True"
        COMPUTE_LAYOUT="False"
        COMPUTE_IMAGE="False"
        ;;

    text-sg-img)
        COMPUTE_GRAPH="False"
        COMPUTE_LAYOUT="False"
        COMPUTE_IMAGE="True"
        ;;

    text-sg-layout)
        COMPUTE_GRAPH="False"
        COMPUTE_LAYOUT="True"
        COMPUTE_IMAGE="False"
        ;;

    text-sg-layout-img)
        COMPUTE_GRAPH="False"
        COMPUTE_LAYOUT="True"
        COMPUTE_IMAGE="True"
        ;;

    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo "----------------------------------------------"
echo "Graph metrics:  $COMPUTE_GRAPH"
echo "Layout metrics: $COMPUTE_LAYOUT"
echo "Image metrics:  $COMPUTE_IMAGE"
echo "----------------------------------------------"

python -m scripts.evaluate_metrics \
    --root_dir "$ROOT_DIR" \
    --mode "$MODE" \
    --gt_sg_dir "$GT_SG_DIR" \
    --gt_layout_dir "$GT_LAYOUT_DIR" \
    --gt_image_dir "$GT_IMAGE_DIR" \
    --compute_graph "$COMPUTE_GRAPH" \
    --compute_layout "$COMPUTE_LAYOUT" \
    --compute_image "$COMPUTE_IMAGE"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Metrics failed for mode: $MODE"
    exit $EXIT_CODE
fi

echo "Finished metrics for mode: $MODE"