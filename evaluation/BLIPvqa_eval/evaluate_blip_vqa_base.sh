# !/bin/bash
# catch arguments
device="$1"
sample_root="$2"
categories="$3"
np_num="$4"

export project_dir="BLIPvqa_eval/"
export CUDA_VISIBLE_DEVICES=$device

cd $project_dir
categories="${categories}"

for category in $categories
do
    out_dir="../${sample_root}/${category}/"
    echo " Evaluating blip-vqa on $out_dir"
    python BLIP_vqa.py --out_dir=$out_dir --np_num="${np_num}"
done