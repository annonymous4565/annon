from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union, List

import torch
from PIL import Image
import tempfile
import shutil


try:
    import clip
except ImportError:
    clip = None

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None

try:
    import ImageReward as RM
except ImportError:
    RM = None

import spacy
from evaluation.BLIPvqa_eval.BLIP.train_vqa_func import VQA_main
# try:
#     import spacy
#     from evaluation.BLIPvqa_eval.BLIP.train_vqa_func import VQA_main
# except ImportError:
#     spacy = None
#     VQA_main = None



def _make_clean_image_dir(image_paths, image_size=1024, suffix=".png"):
    tmpdir = tempfile.mkdtemp(prefix="clean_fid_")

    for i, p in enumerate(image_paths):
        out = os.path.join(tmpdir, f"{i:06d}{suffix}")
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize((image_size, image_size), Image.BICUBIC)
            im.save(out)

    return tmpdir

def compute_fid_from_paths(generated_image_paths, reference_image_paths) -> float:
    if calculate_metrics is None:
        raise ImportError("torch-fidelity is not installed.")

    gen_tmp = _make_clean_image_dir(generated_image_paths, image_size=1024)
    ref_tmp = _make_clean_image_dir(reference_image_paths, image_size=1024)

    try:
        metrics = calculate_metrics(
            input1=gen_tmp,
            input2=ref_tmp,
            cuda=torch.cuda.is_available(),
            fid=True,
            kid=False,
            isc=False,
            verbose=False,
            samples_find_deep=True,
            batch_size=32,
            dataloader_num_workers=0,
        )
        return float(metrics["frechet_inception_distance"])
    finally:
        shutil.rmtree(gen_tmp, ignore_errors=True)
        shutil.rmtree(ref_tmp, ignore_errors=True)

@dataclass
class ImageEvalResult:
    fid: Optional[float] = None
    clip_text_score: Optional[float] = None
    clip_image_score: Optional[float] = None
    image_reward: Optional[float] = None
    blip_vqa: Optional[float] = None


def sorted_image_paths(image_dir: Union[str, os.PathLike]) -> List[str]:
    image_dir = str(image_dir)
    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    ]

    def key_fn(x):
        nums = re.findall(r"\d+", x)
        return int(nums[-1]) if nums else x

    return [os.path.join(image_dir, f) for f in sorted(files, key=key_fn)]


def compute_fid(generated_image_dir, reference_image_dir) -> float:
    if calculate_metrics is None:
        raise ImportError("torch-fidelity is not installed.")
    metrics = calculate_metrics(
        input1=str(Path(generated_image_dir).resolve()),
        input2=str(Path(reference_image_dir).resolve()),
        cuda=torch.cuda.is_available(),
        fid=True,
        kid=False,
        isc=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


@torch.no_grad()
def compute_clip_text_score(
    image_paths: Sequence[str],
    prompts: Sequence[str],
    device: Optional[torch.device] = None,
    model_name: str = "ViT-B/32",
) -> float:
    if clip is None:
        raise ImportError("clip is not installed.")

    assert len(image_paths) == len(prompts)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    sims = []
    for image_path, prompt in zip(image_paths, prompts):
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text = clip.tokenize([prompt], truncate=True).to(device)

        image_feat = model.encode_image(image)
        text_feat = model.encode_text(text)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sims.append((image_feat * text_feat).sum(dim=-1).item())

    return float(sum(sims) / max(len(sims), 1))


@torch.no_grad()
def compute_clip_image_score(
    generated_image_paths: Sequence[str],
    reference_image_paths: Sequence[str],
    device: Optional[torch.device] = None,
    model_name: str = "ViT-B/32",
) -> float:
    if clip is None:
        raise ImportError("clip is not installed.")

    assert len(generated_image_paths) == len(reference_image_paths)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    sims = []
    for gen_path, ref_path in zip(generated_image_paths, reference_image_paths):
        gen_img = preprocess(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)
        ref_img = preprocess(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)

        gen_feat = model.encode_image(gen_img)
        ref_feat = model.encode_image(ref_img)

        gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)

        sims.append((gen_feat * ref_feat).sum(dim=-1).item())

    return float(sum(sims) / max(len(sims), 1))


@torch.no_grad()
def compute_image_reward(
    image_paths: Sequence[str],
    prompts: Sequence[str],
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> float:
    if RM is None:
        raise ImportError("ImageReward is not installed.")

    assert len(image_paths) == len(prompts)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RM.load("ImageReward-v1.0")
    model = model.to(device)
    model.eval()

    records = []
    scores = []

    for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
        reward = model.score(prompt, image_path)
        reward = float(reward)

        scores.append(reward)
        records.append({"question_id": i, "answer": reward, "prompt": prompt, "image": image_path})

        if (i + 1) % 100 == 0:
            print(f"ImageReward: {i + 1} images processed")

    avg = float(sum(scores) / max(len(scores), 1))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vqa_result.json"), "w") as f:
            json.dump(records, f, indent=2)
        with open(os.path.join(save_dir, "score_avg.txt"), "w") as f:
            f.write("score avg:" + str(avg))

    return avg


def create_blip_annotations_from_prompts(
    image_paths: Sequence[str],
    prompts: Sequence[str],
    outpath: str,
    np_index: int,
):
    if spacy is None:
        raise ImportError("spacy is not installed.")

    assert len(image_paths) == len(prompts)

    nlp = spacy.load("en_core_web_sm")
    os.makedirs(outpath, exist_ok=True)

    annotations = []

    for qid, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
        doc = nlp(prompt)

        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in ["top", "the side", "the left", "the right"]:
                noun_phrases.append(chunk.text)

        if len(noun_phrases) > np_index:
            question = f"{noun_phrases[np_index]}?"
        else:
            question = ""

        annotations.append({
            "image": image_path,
            "question_id": qid,
            "question": question,
            "dataset": "color",
            "prompt": prompt,
        })

    with open(os.path.join(outpath, "vqa_test.json"), "w") as f:
        json.dump(annotations, f)

    print("Number of Processed Images:", len(annotations))


def compute_blip_vqa(
    image_paths: Sequence[str],
    prompts: Sequence[str],
    out_dir: str,
    np_num: int = 8,
) -> float:
    if VQA_main is None:
        raise ImportError("BLIP.train_vqa_func.VQA_main could not be imported.")

    assert len(image_paths) == len(prompts)

    os.makedirs(out_dir, exist_ok=True)

    sample_num = len(image_paths)
    reward = torch.zeros((sample_num, np_num), device="cuda" if torch.cuda.is_available() else "cpu")

    order = "_blip"

    for i in range(np_num):
        print(f"start VQA {i + 1}/{np_num}")

        ann_dir = os.path.join(out_dir, f"annotation{i + 1}{order}")
        vqa_dir = os.path.join(ann_dir, "VQA")

        os.makedirs(vqa_dir, exist_ok=True)

        create_blip_annotations_from_prompts(
            image_paths=image_paths,
            prompts=prompts,
            outpath=ann_dir,
            np_index=i,
        )

        VQA_main(ann_dir + "/", vqa_dir + "/")

        with open(os.path.join(vqa_dir, "result", "vqa_result.json"), "r") as f:
            r = json.load(f)

        with open(os.path.join(ann_dir, "vqa_test.json"), "r") as f:
            ann = json.load(f)

        for k in range(len(r)):
            if ann[k]["question"] != "":
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1.0

        print(f"end VQA {i + 1}/{np_num}")

    reward_final = reward[:, 0]
    for i in range(1, np_num):
        reward_final *= reward[:, i]

    records = []
    score_sum = 0.0

    for k in range(sample_num):
        val = float(reward_final[k].item())
        score_sum += val
        records.append({"question_id": k, "answer": f"{val:.4f}"})

    avg = score_sum / max(sample_num, 1)

    final_dir = os.path.join(out_dir, f"annotation{order}")
    os.makedirs(final_dir, exist_ok=True)

    with open(os.path.join(final_dir, "vqa_result.json"), "w") as f:
        json.dump(records, f)

    with open(os.path.join(final_dir, "blip_vqa_score.txt"), "w") as f:
        f.write("BLIP-VQA score:" + str(avg))

    print("BLIP-VQA score:", avg)
    return float(avg)


def evaluate_image_generation(
    generated_image_dir: Optional[str] = None,
    reference_image_dir: Optional[str] = None,
    generated_image_paths: Optional[Sequence[str]] = None,
    reference_image_paths: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
    compute_fid_flag: bool = True,
    compute_clip_text_flag: bool = True,
    compute_clip_image_flag: bool = True,
    compute_image_reward_flag: bool = False,
    compute_blip_vqa_flag: bool = False,
    blip_out_dir: Optional[str] = None,
    blip_np_num: int = 8,
    device: Optional[torch.device] = None,
) -> ImageEvalResult:
    if generated_image_paths is None and generated_image_dir is not None:
        generated_image_paths = sorted_image_paths(generated_image_dir)

    if reference_image_paths is None and reference_image_dir is not None:
        reference_image_paths = sorted_image_paths(reference_image_dir)

    result = ImageEvalResult()

    if compute_fid_flag:
        if generated_image_paths is not None and reference_image_paths is not None:
            n = min(len(generated_image_paths), len(reference_image_paths))
            result.fid = compute_fid_from_paths(
                generated_image_paths[:n],
                reference_image_paths[:n],
            )
        elif generated_image_dir is not None and reference_image_dir is not None:
            result.fid = compute_fid(generated_image_dir, reference_image_dir)

    if compute_clip_text_flag:
        if generated_image_paths is None or prompts is None:
            raise ValueError("CLIP text-image needs generated_image_paths and prompts.")
        result.clip_text_score = compute_clip_text_score(generated_image_paths, prompts, device=device)

    if compute_clip_image_flag:
        if generated_image_paths is None or reference_image_paths is None:
            raise ValueError("CLIP image-image needs generated_image_paths and reference_image_paths.")
        n = min(len(generated_image_paths), len(reference_image_paths))
        result.clip_image_score = compute_clip_image_score(
            generated_image_paths[:n],
            reference_image_paths[:n],
            device=device,
        )

    if compute_image_reward_flag:
        if generated_image_paths is None or prompts is None:
            raise ValueError("ImageReward needs generated_image_paths and prompts.")
        save_dir = None
        if generated_image_dir is not None:
            save_dir = os.path.join(generated_image_dir, "..", "annotation_imageReward")
        result.image_reward = compute_image_reward(
            generated_image_paths,
            prompts,
            device=device,
            save_dir=save_dir,
        )

    if compute_blip_vqa_flag:
        if generated_image_paths is None or prompts is None:
            raise ValueError("BLIP-VQA needs generated_image_paths and prompts.")
        if blip_out_dir is None:
            if generated_image_dir is None:
                raise ValueError("Provide blip_out_dir when generated_image_dir is None.")
            blip_out_dir = os.path.join(generated_image_dir, "..")
        result.blip_vqa = compute_blip_vqa(
            generated_image_paths,
            prompts,
            out_dir=blip_out_dir,
            np_num=blip_np_num,
        )

    return result


# image_paths = sorted_image_paths("./output/mode/gen_images")
# prompts = ["a person riding a horse", "a dog beside a chair", ...]

# res = evaluate_image_generation(
#     generated_image_paths=image_paths,
#     reference_image_paths=sorted_image_paths("./output/gt_images"),
#     prompts=prompts,
#     generated_image_dir="./output/mode/gen_images",
#     reference_image_dir="./output/gt_images",
#     compute_image_reward_flag=True,
#     compute_blip_vqa_flag=True,
#     blip_out_dir="./output/mode",
# )