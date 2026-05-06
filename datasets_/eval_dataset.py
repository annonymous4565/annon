import os
import json
import random
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from datasets_.visual_genome.dataset import (
    SceneGraphDataset,
    decode_item,
    format_decoded_graph,
)

from configs import DiscreteSGEvalConfig


# ============================================================
# Common helpers
# ============================================================

def pil_to_tensor(img: Image.Image, image_size: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return tfm(img.convert("RGB"))


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1).clamp(0.0, 1.0)


def graph_to_prompt(nodes, triplets, max_triplets: int = 30) -> str:
    objects = [str(n) for n in nodes if str(n) != "__pad__"]
    obj_text = ", ".join(objects)

    if triplets is None or len(triplets) == 0:
        rel_text = "no relations"
    else:
        rel_text = "; ".join(
            f"{s_name} {rel} {o_name}"
            for s_idx, s_name, rel, o_idx, o_name in triplets[:max_triplets]
        )

    return f"Objects: {obj_text}. Relations: {rel_text}."


def find_vg_image_path(image_dirs: List[str], image_id: Optional[int], url_tail: Optional[str]):
    candidates = []

    if image_id is not None:
        candidates.extend([
            f"{image_id}.jpg",
            f"{image_id}.png",
            f"{int(image_id):012d}.jpg",
            f"{int(image_id):012d}.png",
        ])

    if url_tail is not None:
        candidates.append(os.path.basename(url_tail))

    for d in image_dirs:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p

    return None


# ============================================================
# Visual Genome eval dataset
# ============================================================

class VisualGenomeEvalDataset(Dataset):
    """
    Eval wrapper around your existing processed VG npz.

    Returns common schema:
      {
        "sample_id": int,
        "source": "vg",
        "text_prompt": str,
        "gt_image": Tensor [3,H,W] or None,
        "gt_image_path": str or None,
        "gt_boxes": Tensor [N,4] cxcywh or None,
        "gt_boxes_xyxy": Tensor [N,4] normalized xyxy or None,
        "gt_scene_graph": dict or None,
        "obj_labels": Tensor [N],
        "rel_labels": Tensor [N,N],
        "node_mask": Tensor [N],
        "edge_mask": Tensor [N,N],
      }
    """

    def __init__(
        self,
        npz_path: str,
        image_dirs: Optional[List[str]] = None,
        image_data_json_path: Optional[str] = None,
        prompt_json_path: Optional[str] = None,
        image_size: int = 512,
        return_images: bool = True,
        fallback_prompt_from_graph: bool = True,
    ):
        super().__init__()

        self.sg_dataset = SceneGraphDataset(
            npz_path=npz_path,
            return_boxes=True,
            return_metadata=True,
        )

        self.image_dirs = image_dirs or []
        self.image_size = image_size
        self.return_images = return_images
        self.fallback_prompt_from_graph = fallback_prompt_from_graph

        self.image_data = None
        if image_data_json_path is not None and image_data_json_path != "":
            with open(image_data_json_path, "r") as f:
                self.image_data = json.load(f)

        self.prompt_map = {}
        if prompt_json_path is not None and prompt_json_path != "":
            with open(prompt_json_path, "r") as f:
                self.prompt_map = json.load(f)

        self.object_vocab = self.sg_dataset.object_vocab
        self.relation_vocab = self.sg_dataset.relation_vocab

    def __len__(self):
        return len(self.sg_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.sg_dataset[idx]

        image_index = item.get("image_index", idx)
        image_id = None
        url_tail = None

        if self.image_data is not None and image_index is not None:
            meta = self.image_data[int(image_index)]
            image_id = meta.get("image_id", None)
            url = meta.get("url", None)
            if url is not None:
                url_tail = os.path.basename(url)

        image_path = find_vg_image_path(
            image_dirs=self.image_dirs,
            image_id=image_id,
            url_tail=url_tail,
        )

        gt_image = None
        if self.return_images and image_path is not None:
            img = Image.open(image_path).convert("RGB")
            gt_image = pil_to_tensor(img, self.image_size)

        nodes, triplets = decode_item(
            obj_labels=item["obj_labels"],
            rel_labels=item["rel_labels"],
            node_mask=item["node_mask"],
            edge_mask=item["edge_mask"],
            object_vocab=self.object_vocab,
            relation_vocab=self.relation_vocab,
            no_rel_token="__no_relation__",
        )

        graph_text = format_decoded_graph(nodes, triplets)
        graph_prompt = graph_to_prompt(nodes, triplets)

        # Prompt priority:
        # 1. prompt_json by image_id
        # 2. prompt_json by image_index
        # 3. graph-derived prompt
        text_prompt = None
        if image_id is not None and str(image_id) in self.prompt_map:
            text_prompt = self.prompt_map[str(image_id)]
        elif str(image_index) in self.prompt_map:
            text_prompt = self.prompt_map[str(image_index)]
        elif self.fallback_prompt_from_graph:
            text_prompt = graph_prompt
        else:
            text_prompt = ""

        gt_boxes = item.get("boxes", None)             # cxcywh normalized from your dataset
        gt_boxes_xyxy = item.get("boxes_xyxy", None)   # raw 1024 xyxy from your dataset

        if gt_boxes_xyxy is not None:
            gt_boxes_xyxy_norm = gt_boxes_xyxy.float() / 1024.0
            gt_boxes_xyxy_norm = gt_boxes_xyxy_norm.clamp(0.0, 1.0)
        else:
            gt_boxes_xyxy_norm = None

        return {
            "sample_id": int(idx),
            "source": "vg",
            "image_index": int(image_index) if image_index is not None else int(idx),
            "image_id": int(image_id) if image_id is not None else -1,
            "text_prompt": text_prompt,
            "gt_image": gt_image,
            "gt_image_path": image_path,
            "gt_boxes": gt_boxes,
            "gt_boxes_xyxy": gt_boxes_xyxy_norm,
            "gt_scene_graph": {
                "nodes": nodes,
                "triplets": triplets,
                "graph_text": graph_text,
            },
            "obj_labels": item["obj_labels"],
            "rel_labels": item["rel_labels"],
            "node_mask": item["node_mask"],
            "edge_mask": item["edge_mask"],
        }


# ============================================================
# COCO eval dataset
# ============================================================

class COCOEvalDataset(Dataset):
    """
    COCO eval dataset with text, image, boxes, but no GT scene graph.

    Requires:
      - instances_json
      - captions_json for text prompts

    Returns same common schema as VG where possible.
    """

    def __init__(
        self,
        image_dir: str,
        instances_json: str,
        captions_json: str,
        image_size: int = 512,
        max_objects_per_image: int = 20,
        min_objects_per_image: int = 1,
        min_object_size: float = 0.0,
        max_samples: Optional[int] = None,
        caption_strategy: str = "first",  # first | random
    ):
        super().__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        self.max_objects_per_image = max_objects_per_image
        self.min_objects_per_image = min_objects_per_image
        self.min_object_size = min_object_size
        self.max_samples = max_samples
        self.caption_strategy = caption_strategy

        with open(instances_json, "r") as f:
            instances = json.load(f)

        with open(captions_json, "r") as f:
            captions = json.load(f)

        self.image_id_to_info = {}
        for im in instances["images"]:
            self.image_id_to_info[int(im["id"])] = im

        self.cat_id_to_name = {
            int(c["id"]): c["name"] for c in instances["categories"]
        }

        self.image_id_to_anns = {}
        for ann in instances["annotations"]:
            image_id = int(ann["image_id"])
            self.image_id_to_anns.setdefault(image_id, []).append(ann)

        self.image_id_to_captions = {}
        for ann in captions["annotations"]:
            image_id = int(ann["image_id"])
            self.image_id_to_captions.setdefault(image_id, []).append(ann["caption"])

        self.image_ids = []
        for image_id, info in self.image_id_to_info.items():
            anns = self._filtered_annotations(image_id)
            has_caption = image_id in self.image_id_to_captions
            if (
                has_caption
                and self.min_objects_per_image <= len(anns) <= self.max_objects_per_image
            ):
                self.image_ids.append(image_id)

        self.image_ids.sort()

        if self.max_samples is not None:
            self.image_ids = self.image_ids[: self.max_samples]

        # COCO has no relations. These are only for compatibility.
        self.object_vocab = ["__pad__"] + sorted(set(self.cat_id_to_name.values()))
        self.object_to_idx = {n: i for i, n in enumerate(self.object_vocab)}
        self.relation_vocab = ["__no_relation__"]

    def _filtered_annotations(self, image_id: int):
        info = self.image_id_to_info[image_id]
        W = float(info["width"])
        H = float(info["height"])

        anns = []
        for ann in self.image_id_to_anns.get(image_id, []):
            x, y, w, h = ann["bbox"]
            area_frac = (w * h) / max(W * H, 1.0)
            if area_frac >= self.min_object_size:
                anns.append(ann)
        return anns

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        info = self.image_id_to_info[image_id]

        image_path = os.path.join(self.image_dir, info["file_name"])
        img = Image.open(image_path).convert("RGB")
        gt_image = pil_to_tensor(img, self.image_size)

        W = float(info["width"])
        H = float(info["height"])

        anns = self._filtered_annotations(image_id)
        anns = anns[: self.max_objects_per_image]

        obj_names = []
        boxes_xyxy = []

        for ann in anns:
            cat_name = self.cat_id_to_name[int(ann["category_id"])]
            obj_names.append(cat_name)

            x, y, w, h = ann["bbox"]
            x1 = x / W
            y1 = y / H
            x2 = (x + w) / W
            y2 = (y + h) / H
            boxes_xyxy.append([x1, y1, x2, y2])

        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32).clamp(0.0, 1.0)
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)

        captions = self.image_id_to_captions[image_id]
        if self.caption_strategy == "random":
            text_prompt = random.choice(captions)
        else:
            text_prompt = captions[0]

        return {
            "sample_id": int(idx),
            "source": "coco",
            "image_index": int(idx),
            "image_id": int(image_id),
            "text_prompt": text_prompt,
            "gt_image": gt_image,
            "gt_image_path": image_path,
            "gt_boxes": boxes_cxcywh,
            "gt_boxes_xyxy": boxes_xyxy,
            "gt_scene_graph": None,

            # COCO has no GT SG in your setup.
            # Keep these None so downstream can branch cleanly.
            "obj_labels": None,
            "rel_labels": None,
            "node_mask": None,
            "edge_mask": None,

            "object_names": obj_names,
        }


# ============================================================
# Collate
# ============================================================

def eval_collate_fn(batch):
    """
    Keep variable-length / optional fields as lists.
    Downstream evaluation usually writes per-sample outputs anyway.
    """
    out = {}

    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]

        if k == "gt_image" and vals[0] is not None:
            out[k] = torch.stack(vals, dim=0)
        elif torch.is_tensor(vals[0]):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                out[k] = vals
        else:
            out[k] = vals

    return out


# ============================================================
# Factory
# ============================================================

def build_eval_dataset(opt: DiscreteSGEvalConfig):
    """
    opt.eval_dataset_type:
      - "vg"
      - "coco"
    """
    dtype = getattr(opt, "eval_dataset_type", "vg").lower()

    if dtype == "vg":
        image_dirs = getattr(opt, "vg_image_dirs", None)
        if isinstance(image_dirs, str):
            image_dirs = [x.strip() for x in image_dirs.split(",") if x.strip()]

        return VisualGenomeEvalDataset(
            npz_path=opt.eval_npz_path,
            image_dirs=image_dirs or [
                "./data/visual_genome/VG_100K",
                "./data/visual_genome/VG_100K_2",
            ],
            image_data_json_path=getattr(opt, "vg_image_data_json_path", None),
            prompt_json_path=getattr(opt, "eval_prompt_json_path", None),
            image_size=getattr(opt, "eval_image_size", 512),
            return_images=getattr(opt, "eval_return_images", True),
            fallback_prompt_from_graph=getattr(opt, "vg_fallback_prompt_from_graph", True),
        )

    if dtype == "coco":
        return COCOEvalDataset(
            image_dir=opt.coco_image_dir,
            instances_json=opt.coco_instances_json,
            captions_json=opt.coco_captions_json,
            image_size=getattr(opt, "eval_image_size", 512),
            max_objects_per_image=getattr(opt, "coco_max_objects_per_image", 20),
            min_objects_per_image=getattr(opt, "coco_min_objects_per_image", 1),
            min_object_size=getattr(opt, "coco_min_object_size", 0.0),
            max_samples=getattr(opt, "eval_max_samples", None),
            caption_strategy=getattr(opt, "coco_caption_strategy", "first"),
        )

    raise ValueError(f"Unknown eval_dataset_type: {dtype}")


def build_eval_dataloader(opt):
    dataset = build_eval_dataset(opt)
    loader = DataLoader(
        dataset,
        batch_size=getattr(opt, "eval_batch_size", 1),
        shuffle=False,
        num_workers=getattr(opt, "eval_num_workers", 0),
        pin_memory=True,
        collate_fn=eval_collate_fn,
    )
    return loader