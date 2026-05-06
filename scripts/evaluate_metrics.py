import os
import ast
import json
from dataclasses import dataclass
from typing import Optional, List

import pyrallis
import torch
import numpy as np

from evaluation.graph_metrics import SceneGraphSample, evaluate_graph_generation
from evaluation.image_metrics import evaluate_image_generation
from evaluation.layout_metrics import LayoutObject, LayoutSample, evaluate_layout_f1


@dataclass
class SavedOutputEvalConfig:
    root_dir: str = "./output/master_eval/run1"
    mode: str = "text-sg-layout-img"

    # reference folders/files
    gt_sg_dir: Optional[str] = None
    gt_layout_dir: Optional[str] = None
    gt_image_dir: Optional[str] = None

    # generated subfolders, relative to root_dir/mode
    gen_sg_subdir: str = "gen_sg"
    gen_layout_subdir: str = "gen_layouts"
    gen_image_subdir: str = "gen_images"

    # output
    metrics_out_name: str = "metrics_summary.txt"

    # toggles
    compute_graph: bool = True
    compute_layout: bool = True
    compute_image: bool = True

    layout_iou_thresh: float = 0.5


def parse_sg_txt(path: str) -> SceneGraphSample:
    triplets = []
    nodes = {}

    if not os.path.exists(path):
        return SceneGraphSample(nodes=[], triplets=[])

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "(none)":
                continue

            t = ast.literal_eval(line)
            s_idx, s_name, rel, o_idx, o_name = t

            s_idx = int(s_idx)
            o_idx = int(o_idx)
            nodes[s_idx] = str(s_name)
            nodes[o_idx] = str(o_name)
            triplets.append((s_idx, str(s_name), str(rel), o_idx, str(o_name)))

    if len(nodes) == 0:
        node_list = []
    else:
        max_idx = max(nodes.keys())
        node_list = ["__pad__"] * (max_idx + 1)
        for k, v in nodes.items():
            node_list[k] = v

    return SceneGraphSample(nodes=node_list, triplets=triplets)


def parse_layout_txt(path: str) -> List[LayoutObject]:
    objs = []
    if not os.path.exists(path):
        return objs

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("idx"):
            continue

        parts = line.split("\t")
        if len(parts) < 6:
            continue

        label = parts[1]
        box = np.array(
            [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
            dtype=np.float32,
        )
        objs.append(LayoutObject(label=label, bbox=box))

    return objs


def list_ids_from_prefix(folder: str, prefix: str, suffix: str):
    ids = []
    if not os.path.isdir(folder):
        return ids

    for name in os.listdir(folder):
        if name.startswith(prefix) and name.endswith(suffix):
            x = name[len(prefix):-len(suffix)]
            ids.append(x)
    return sorted(ids)


def read_prompt_map(root_dir: str):
    path = os.path.join(root_dir, "text_prompts.json")
    if not os.path.exists(path):
        return {}

    with open(path, "r") as f:
        rows = json.load(f)

    out = {}
    for r in rows:
        out[str(r["file_id"])] = r["text_prompt"]
    return out


@pyrallis.wrap()
def main(opt: SavedOutputEvalConfig):
    mode_dir = os.path.join(opt.root_dir, opt.mode)
    out_path = os.path.join(mode_dir, opt.metrics_out_name)

    results = {}

    # -------------------------
    # Graph metrics
    # -------------------------
    if opt.compute_graph:
        gen_sg_dir = os.path.join(mode_dir, opt.gen_sg_subdir)

        if opt.gt_sg_dir is not None and os.path.isdir(opt.gt_sg_dir):
            gen_ids = list_ids_from_prefix(gen_sg_dir, "sg_", ".txt")
            ref_ids = list_ids_from_prefix(opt.gt_sg_dir, "sg_", ".txt")
            ids = sorted(set(gen_ids) & set(ref_ids))

            gen_graphs = [
                parse_sg_txt(os.path.join(gen_sg_dir, f"sg_{i}.txt"))
                for i in ids
            ]
            ref_graphs = [
                parse_sg_txt(os.path.join(opt.gt_sg_dir, f"sg_{i}.txt"))
                for i in ids
            ]

            graph_metrics = evaluate_graph_generation(
                generated_graphs=gen_graphs,
                reference_graphs=ref_graphs,
            )
            results.update({f"graph/{k}": v for k, v in graph_metrics.items()})
        else:
            results["graph/skipped"] = "missing gt_sg_dir"

    # -------------------------
    # Layout metrics
    # -------------------------
    if opt.compute_layout:
        gen_layout_dir = os.path.join(mode_dir, opt.gen_layout_subdir)

        if opt.gt_layout_dir is not None and os.path.isdir(opt.gt_layout_dir):
            gen_ids = list_ids_from_prefix(gen_layout_dir, "layout_", ".txt")
            ref_ids = list_ids_from_prefix(opt.gt_layout_dir, "layout_", ".txt")
            ids = sorted(set(gen_ids) & set(ref_ids))

            samples = []
            for i in ids:
                pred = parse_layout_txt(os.path.join(gen_layout_dir, f"layout_{i}.txt"))
                gt = parse_layout_txt(os.path.join(opt.gt_layout_dir, f"layout_{i}.txt"))
                samples.append(LayoutSample(pred_objects=pred, gt_objects=gt))

            layout_metrics = evaluate_layout_f1(
                samples=samples,
                iou_thresh=opt.layout_iou_thresh,
            )
            results.update({f"layout/{k}": v for k, v in layout_metrics.items()})
        else:
            results["layout/skipped"] = "missing gt_layout_dir"

    # -------------------------
    # Image metrics
    # -------------------------
    if opt.compute_image:
        gen_image_dir = os.path.join(mode_dir, opt.gen_image_subdir)
        gt_image_dir = opt.gt_image_dir or os.path.join(opt.root_dir, "gt_images")

        if os.path.isdir(gen_image_dir) and os.path.isdir(gt_image_dir):
            prompt_map = read_prompt_map(opt.root_dir)
            image_ids = list_ids_from_prefix(gen_image_dir, "img_", ".png")

            image_paths = [
                os.path.join(gen_image_dir, f"img_{i}.png")
                for i in image_ids
            ]
            graph_texts = [
                prompt_map.get(i, "")
                for i in image_ids
            ]

            image_res = evaluate_image_generation(
                generated_image_dir=gen_image_dir,
                reference_image_dir=gt_image_dir,
                image_paths=image_paths,
                graph_texts=graph_texts,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

            results["image/fid"] = image_res.fid
            results["image/clip_score"] = image_res.clip_score
            results["image/span_similarity"] = image_res.span_similarity
        else:
            results["image/skipped"] = "missing gen_image_dir or gt_image_dir"

    os.makedirs(mode_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    with open(os.path.join(mode_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()