#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#                USER CONFIGURATION (EDIT HERE)
# ============================================================

# -------------------------
# GPU assignment
# -------------------------
gpu_graph=0
gpu_layout=1
gpu_image=2

# -------------------------
# Optional: wait for job
# -------------------------
wait_pid=""   # e.g. 2930664, leave "" to disable

# -------------------------
# Graph inputs
# -------------------------
pred_sg_json="./eval_inputs/pred_sg.json"
gt_sg_json="./eval_inputs/gt_sg.json"

# -------------------------
# Layout inputs
# -------------------------
pred_layout_json="./eval_inputs/pred_layout.json"
gt_layout_json="./eval_inputs/gt_layout.json"

# -------------------------
# Image inputs
# -------------------------
gen_image_dir="./generated_images"
ref_image_dir="./reference_images"
graph_text_json="./eval_inputs/graph_texts.json"

# -------------------------
# BLIP-VQA inputs
# -------------------------
blip_prompt_json="./eval_inputs/prompts.json"
blip_out_dir="${out_dir}/blip_vqa"
blip_np_num=4
blip_project_dir="./scripts/evaluation/BLIPvqa_eval"

# -------------------------
# Output directory
# -------------------------
out_dir="./eval_outputs"

# -------------------------
# SPAN (optional)
# Leave empty to disable
# -------------------------
span_repo_root=""
span_ckpt=""
span_img_folder=""
span_prediction_records_json=""


# ============================================================
#                    SCRIPT START
# ============================================================

mkdir -p "${out_dir}"

# -------------------------
# Wait for PID if needed
# -------------------------
if [[ -n "${wait_pid}" ]]; then
    echo "Waiting for PID ${wait_pid} to finish..."
    while kill -0 "${wait_pid}" 2>/dev/null; do
        sleep 10
    done
    echo "PID ${wait_pid} finished."
fi

# ============================================================
# GRAPH METRICS
# ============================================================
echo "============================================================"
echo "Evaluating GRAPH metrics"
echo "GPU: ${gpu_graph}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${gpu_graph} python scripts/evaluate_graph_metrics.py \
    --pred_sg_json "${pred_sg_json}" \
    --gt_sg_json "${gt_sg_json}" \
    --out_dir "${out_dir}"

# ============================================================
# LAYOUT METRICS
# ============================================================
echo "============================================================"
echo "Evaluating LAYOUT metrics"
echo "GPU: ${gpu_layout}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${gpu_layout} python scripts/evaluate_layout_metrics.py \
    --pred_layout_json "${pred_layout_json}" \
    --gt_layout_json "${gt_layout_json}" \
    --out_dir "${out_dir}" \
    --iou_thresh 0.5

# ============================================================
# IMAGE METRICS
# ============================================================
echo "============================================================"
echo "Evaluating IMAGE metrics"
echo "GPU: ${gpu_image}"
echo "============================================================"

IMAGE_CMD=(
    python scripts/evaluate_image_metrics.py
    --gen_image_dir "${gen_image_dir}"
    --ref_image_dir "${ref_image_dir}"
    --graph_text_json "${graph_text_json}"
    --out_dir "${out_dir}"
)

# -------------------------
# SPAN (optional)
# -------------------------
if [[ -n "${span_repo_root}" && -n "${span_ckpt}" && -n "${span_img_folder}" && -n "${span_prediction_records_json}" ]]; then
    echo "SPAN enabled"
    IMAGE_CMD+=(
        --span_repo_root "${span_repo_root}"
        --span_ckpt "${span_ckpt}"
        --span_img_folder "${span_img_folder}"
        --span_prediction_records_json "${span_prediction_records_json}"
    )
else
    echo "SPAN disabled"
fi

CUDA_VISIBLE_DEVICES=${gpu_image} "${IMAGE_CMD[@]}"

echo "============================================================"
echo "Evaluating BLIP-VQA"
echo "GPU: ${gpu_image}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${gpu_image} python scripts/evaluate_blip_vqa.py \
    --image_dir "${gen_image_dir}" \
    --prompt_json "${blip_prompt_json}" \
    --out_dir "${blip_out_dir}" \
    --np_num "${blip_np_num}"

# ============================================================
# DONE
# ============================================================
echo "============================================================"
echo "All evaluations completed."
echo "Results saved to: ${out_dir}"
echo "============================================================"