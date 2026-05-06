import os
import re
import json
import argparse
from pathlib import Path

import torch
import pickle
from evaluation.image_metrics import (
    evaluate_image_generation,
    sorted_image_paths,
)



def load_coco_caption_index(input_file, dataset):
    if dataset == "COCO":
        with open(input_file, 'rb') as f:
                    data = pickle.load(f)

        out = {}
        for e in data:
            image_id = int(e["img_id"])
            caption = str(e["captions"])
            out[image_id] = caption
        return out
    elif dataset == "CompSGBench":
        with open(input_file, 'r') as f:
            data = json.load(f)
            data = data[:1000]
            out = {}
        for e in data:
            image_id = int(e["img_id"])
            caption = str(e["top_caption"])
            name = str(e["name"])
            out[image_id] = (caption,name)
        return out



def parse_generated_filename(path):
    """
    Expected:
      gen_root/<sanitized_caption>_<id>_<image_id>.png

    We parse from the end so sanitized caption may contain underscores.
    """
    stem = Path(path).stem
    image_id = stem.rsplit(".")[0]

    return int(image_id)


def coco_image_path(gt_root, image_id, dataset):
    if dataset == "COCO":
        fname = f"{int(image_id):012d}.jpg"
    elif dataset == "CompSGBench":
        fname = f"{image_id}"
    return os.path.join(gt_root, fname)


def collect_aligned_items(gen_root, gt_root, input_file, dataset):
    caption_index = load_coco_caption_index(input_file, dataset)
    gen_paths = sorted_image_paths(gen_root)

    aligned_gen = []
    aligned_gt = []
    prompts = []
    skipped = []

    for gen_path in gen_paths:
        try:
            image_id = parse_generated_filename(gen_path)
        except Exception as e:
            skipped.append((gen_path, str(e)))
            continue

        key = image_id
        if key not in caption_index:
            skipped.append((gen_path, f"caption not found for key={key}"))
            continue
        if dataset == "COCO":
            gt_path = coco_image_path(gt_root, image_id, dataset)
            prompts.append(caption_index[key])
        elif dataset == "CompSGBench":
            _, name = caption_index[key]
            gt_path = coco_image_path(gt_root, name, dataset)
            prompts.append(caption_index[key][0])
        if not os.path.exists(gt_path):
            skipped.append((gen_path, f"GT image missing: {gt_path}"))
            prompts.pop()
            continue

        aligned_gen.append(gen_path)
        aligned_gt.append(gt_path)
        

    return aligned_gen, aligned_gt, prompts, skipped


def save_metrics(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "fid": result.fid,
        "clip_text_score": result.clip_text_score,
        "clip_image_score": result.clip_image_score,
        "image_reward": result.image_reward,
        "blip_vqa": result.blip_vqa,
    }

    with open(os.path.join(output_dir, "image_metrics_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(output_dir, "image_metrics_summary.txt"), "w") as f:
        for k, v in payload.items():
            f.write(f"{k}: {v}\n")

    print(payload)



def main():

    pp =[
    "five_category",
    "mix_relation",
    "only_numeral",
    "sampled_non_relation",
    "sampled_only_semantic",
    "sampled_only_spatial",

]
    ii = 0

    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gen_root", default=f"./output/ImageGen/ComposeDiff/COCO/{pp[ii]}/cfg5.0_/samples")
    parser.add_argument("--gen_root", default="./output/ImageGen/ComposeDiff/CompSGBench/cfg5.0_/samples")
    # parser.add_argument("--gt_root", default="./data/coco/images/val2017")
    parser.add_argument("--gt_root", default="../SG_baselines/LAION-SG/data")
    # parser.add_argument("--input_file", default=f"./data/final/coco_{pp[ii]}.pkl")
    parser.add_argument("--input_file", default=f"./data/final/CompSGBench_final.json")
    # parser.add_argument("--output_dir", default=f"./output/ImageGen/ComposeDiff/COCO/{pp[ii]}")
    parser.add_argument("--output_dir", default=f"./output/ImageGen/ComposeDiff/CompSGBench/")
    # parser.add_argument("--dataset", default="COCO") #CompSGBench
    parser.add_argument("--dataset", default="CompSGBench") #CompSGBench
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--no_clip_text", action="store_true")
    parser.add_argument("--no_clip_image", action="store_true")
    parser.add_argument("--image_reward", action="store_true", default=True)
    parser.add_argument("--blip_vqa", action="store_true", default=True)
    parser.add_argument("--blip_np_num", type=int, default=4)

    args = parser.parse_args()

    gen_paths, gt_paths, prompts, skipped = collect_aligned_items(
        gen_root=args.gen_root,
        gt_root=args.gt_root,
        input_file=args.input_file,
        dataset=args.dataset
    )

    if args.max_samples > 0:
        gen_paths = gen_paths[: args.max_samples]
        gt_paths = gt_paths[: args.max_samples]
        prompts = prompts[: args.max_samples]

    print(f"aligned samples: {len(gen_paths)}")
    print(f"skipped samples: {len(skipped)}")

    if skipped:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "skipped.json"), "w") as f:
            json.dump(
                [{"gen_path": p, "reason": r} for p, r in skipped],
                f,
                indent=2,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = evaluate_image_generation(
        generated_image_dir=args.gen_root,
        reference_image_dir=args.gt_root,
        generated_image_paths=gen_paths,
        reference_image_paths=gt_paths,
        prompts=prompts,
        compute_fid_flag=not args.no_fid,
        compute_clip_text_flag=not args.no_clip_text,
        compute_clip_image_flag=not args.no_clip_image,
        compute_image_reward_flag=args.image_reward,
        compute_blip_vqa_flag=args.blip_vqa,
        blip_out_dir=args.output_dir,
        blip_np_num=args.blip_np_num,
        device=device,
    )

    save_metrics(result, args.output_dir)


if __name__ == "__main__":
    main()


















































