#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# USER CONFIGURATION
# ============================================================

gpu=0

# Directory with generated images
image_dir="./generated_images"

# JSON list of prompts aligned with generated images
prompt_json="./eval_inputs/prompts.json"

# Output folder
out_dir="./eval_outputs/blip_vqa"

# Number of noun phrase slots to test
np_num=4

# BLIP project root
project_dir="./scripts/evaluation/BLIPvqa_eval"

# ============================================================
# RUN
# ============================================================

export CUDA_VISIBLE_DEVICES=${gpu}

cd "${project_dir}"

python ../../evaluate_blip_vqa.py \
  --image_dir "${image_dir}" \
  --prompt_json "${prompt_json}" \
  --out_dir "${out_dir}" \
  --np_num "${np_num}"