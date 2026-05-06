import argparse
import json
import os
from typing import List, Tuple

import torch
import spacy
from tqdm.auto import tqdm

from evaluation.BLIPvqa_eval.BLIP.train_vqa_func import VQA_main


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def get_sorted_image_files(image_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    files.sort()
    return files


def load_prompts(prompt_json: str) -> List[str]:
    """
    Accepts either:
      1. a JSON list of strings
      2. a JSON list of dicts with key 'prompt'
      3. a JSON dict mapping string/int ids -> prompt
    """
    with open(prompt_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], str):
            return obj
        if isinstance(obj[0], dict):
            if "prompt" in obj[0]:
                return [str(x["prompt"]) for x in obj]
            raise ValueError("Prompt dict list must contain key 'prompt'.")

    if isinstance(obj, dict):
        # sort keys numerically if possible, else lexicographically
        def sort_key(x):
            try:
                return int(x)
            except Exception:
                return str(x)

        keys = sorted(obj.keys(), key=sort_key)
        return [str(obj[k]) for k in keys]

    raise ValueError("Unsupported prompt JSON format.")


def extract_noun_phrases(prompt: str, nlp) -> List[str]:
    doc = nlp(prompt)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if text.lower() not in ["top", "the side", "the left", "the right"]:
            noun_phrases.append(text)
    return noun_phrases


def create_annotation_for_blip(
    image_dir: str,
    prompt_json: str,
    outpath: str,
    np_index: int = 0,
):
    """
    Creates vqa_test.json for a single noun-phrase index.
    """
    nlp = spacy.load("en_core_web_sm")

    image_files = get_sorted_image_files(image_dir)
    prompts = load_prompts(prompt_json)

    if len(image_files) != len(prompts):
        raise ValueError(
            f"Number of images ({len(image_files)}) does not match number of prompts ({len(prompts)})."
        )

    annotations = []

    for idx, (file_name, prompt) in enumerate(zip(image_files, prompts)):
        image_dict = {}
        image_dict["image"] = os.path.join(image_dir, file_name)
        image_dict["question_id"] = idx

        noun_phrases = extract_noun_phrases(prompt, nlp)
        if len(noun_phrases) > np_index:
            q_tmp = noun_phrases[np_index]
            image_dict["question"] = f"{q_tmp}?"
        else:
            image_dict["question"] = ""

        image_dict["dataset"] = "custom"
        annotations.append(image_dict)

    print("Number of processed images:", len(annotations))

    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "vqa_test.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f)


def parse_args():
    parser = argparse.ArgumentParser(description="BLIP VQA evaluation for prompt-image alignment.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing generated images.",
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        required=True,
        help="JSON file containing prompts aligned to the generated images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write BLIP-VQA outputs.",
    )
    parser.add_argument(
        "--np_num",
        type=int,
        default=8,
        help="Maximum noun phrase index to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np_num = args.np_num

    image_files = get_sorted_image_files(args.image_dir)
    sample_num = len(image_files)

    if sample_num == 0:
        raise ValueError(f"No images found in: {args.image_dir}")

    reward = torch.zeros((sample_num, np_num), device="cuda" if torch.cuda.is_available() else "cpu")

    order = "_blip"

    for i in tqdm(range(np_num), desc="BLIP-VQA noun phrase sweep"):
        print(f"start VQA {i+1}/{np_num}")

        ann_dir = os.path.join(args.out_dir, f"annotation{i + 1}{order}")
        vqa_dir = os.path.join(ann_dir, "VQA")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(vqa_dir, exist_ok=True)

        create_annotation_for_blip(
            image_dir=args.image_dir,
            prompt_json=args.prompt_json,
            outpath=ann_dir,
            np_index=i,
        )

        _ = VQA_main(
            ann_dir + "/",
            vqa_dir + "/",
        )

        with open(os.path.join(vqa_dir, "result", "vqa_result.json"), "r", encoding="utf-8") as f:
            result_json = json.load(f)

        with open(os.path.join(ann_dir, "vqa_test.json"), "r", encoding="utf-8") as f:
            ann_json = json.load(f)

        for k in range(len(result_json)):
            if ann_json[k]["question"] != "":
                reward[k][i] = float(result_json[k]["answer"])
            else:
                reward[k][i] = 1.0

        print(f"end VQA {i+1}/{np_num}")

    # Product across noun phrase questions, matching your original logic
    reward_final = reward[:, 0]
    for i in range(1, np_num):
        reward_final *= reward[:, i]

    final_ann_dir = os.path.join(args.out_dir, f"annotation{order}")
    os.makedirs(final_ann_dir, exist_ok=True)

    # reuse last result template
    last_result_path = os.path.join(args.out_dir, f"annotation{np_num}{order}", "VQA", "result", "vqa_result.json")
    with open(last_result_path, "r", encoding="utf-8") as f:
        result_json = json.load(f)

    reward_avg = 0.0
    for k in range(len(result_json)):
        result_json[k]["answer"] = f"{reward_final[k].item():.4f}"
        reward_avg += float(result_json[k]["answer"])

    with open(os.path.join(final_ann_dir, "vqa_result.json"), "w", encoding="utf-8") as f:
        json.dump(result_json, f)

    reward_avg /= max(len(result_json), 1)

    print("BLIP-VQA score:", reward_avg)
    with open(os.path.join(final_ann_dir, "blip_vqa_score.txt"), "w", encoding="utf-8") as f:
        f.write("BLIP-VQA score:" + str(reward_avg))


if __name__ == "__main__":
    main()