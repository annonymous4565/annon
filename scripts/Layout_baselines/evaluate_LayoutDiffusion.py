import os
import json
import argparse
from typing import List, Optional, Dict

import numpy as np

from evaluation.layout_metrics import LayoutObject, LayoutSample, evaluate_layout_f1

DATASET = "coco_stuff"

def safe_label(cls_id: int, id_to_label: Optional[Dict[int, str]] = None) -> str:
    if id_to_label is None:
        return f"cls_{int(cls_id)}"
    return str(id_to_label.get(int(cls_id), f"cls_{int(cls_id)}"))


def xyxy_grid_to_norm_box(x1, y1, x2, y2, coord_grid: int) -> np.ndarray:
    denom = float(coord_grid - 1)
    return np.clip(
        np.array([x1 / denom, y1 / denom, x2 / denom, y2 / denom], dtype=np.float32),
        0.0,
        1.0,
    )


def cxcywh_grid_to_norm_box(cx, cy, w, h, coord_grid: int) -> np.ndarray:
    denom = float(coord_grid - 1)
    cx, cy, w, h = cx / denom, cy / denom, w / denom, h / denom
    return np.clip(
        np.array(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )


def decode_layoutdiffusion_pred_row(
    row: np.ndarray,
    coord_grid: int = 128,
    layout_dim: int = 5,
    pad_id: int = 0,
    eos_id: int = 1,
    id_to_label: Optional[Dict[int, str]] = None,
) -> List[LayoutObject]:
    """
    LayoutDiffusion converted prediction format:
        [cls, x1, y1, x2, y2, cls, x1, y1, x2, y2, ...]

    Coordinates are xyxy discrete grid coords.
    """
    row = np.asarray(row, dtype=np.int64).reshape(-1)
    objects: List[LayoutObject] = []

    usable_len = (len(row) // layout_dim) * layout_dim

    for i in range(0, usable_len, layout_dim):
        cls_id, x1, y1, x2, y2 = row[i : i + layout_dim].tolist()

        if cls_id in (pad_id, eos_id):
            continue
        if min(x1, y1, x2, y2) < 0:
            continue
        if max(x1, y1, x2, y2) >= coord_grid:
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        bbox = xyxy_grid_to_norm_box(x1, y1, x2, y2, coord_grid)

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        objects.append(
            LayoutObject(
                label=safe_label(cls_id, id_to_label),
                bbox=bbox,
            )
        )

    return objects


def decode_blt_gt_row(
    row: np.ndarray,
    coord_grid: int = 128,
    layout_dim: int = 5,
    pad_id: int = 0,
    eos_id: int = 1,
    id_to_label: Optional[Dict[int, str]] = None,
) -> List[LayoutObject]:
    """
    BLT real/GT format:
        [cls, cx, cy, w, h, cls, cx, cy, w, h, ...]

    Coordinates are cxcywh discrete grid coords.
    """
    row = np.asarray(row, dtype=np.int64).reshape(-1)
    objects: List[LayoutObject] = []

    usable_len = (len(row) // layout_dim) * layout_dim

    for i in range(0, usable_len, layout_dim):
        cls_id, cx, cy, w, h = row[i : i + layout_dim].tolist()

        if cls_id in (pad_id, eos_id):
            continue
        if min(cx, cy, w, h) < 0:
            continue
        if max(cx, cy, w, h) >= coord_grid:
            continue
        if w <= 0 or h <= 0:
            continue

        bbox = cxcywh_grid_to_norm_box(cx, cy, w, h, coord_grid)

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        objects.append(
            LayoutObject(
                label=safe_label(cls_id, id_to_label),
                bbox=bbox,
            )
        )

    return objects


def load_id_to_label(label_map_json: str):
    if not label_map_json:
        return None

    with open(label_map_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload.get("id_to_label", payload)
    return {int(k): str(v) for k, v in raw.items()}


def save_metrics(metrics, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    txt_path = os.path.join(output_path, "layout_metrics_summary.txt")
    json_path = os.path.join(output_path, "layout_metrics_summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] saved {txt_path}")
    print(f"[OK] saved {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate converted LayoutDiffusion predictions against BLT GT layouts."
    )

    parser.add_argument("--pred_npy", default=f"../SG_baselines/LayoutGeneration/LayoutDiffusion/results/checkpoint/{DATASET}/samples/generated.npy")
    parser.add_argument("--gt_npy", default=f"../SG_baselines/google-research/layout-blt/ckpt/{DATASET}/test/reference_samples.npy", help="Reference BLT .npy, shape [B,L].")
    parser.add_argument("--output_path", default=f'./output/Layout_baselines/LayoutDiffusion/{DATASET}')

    parser.add_argument("--coord_grid", type=int, default=128)
    parser.add_argument("--layout_dim", type=int, default=5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--eos_id", type=int, default=1)
    parser.add_argument("--label_map_json", type=str, default="")

    args = parser.parse_args()

    pred_arr = np.load(args.pred_npy)
    gt_arr = np.load(args.gt_npy)

    if pred_arr.ndim != 2:
        raise ValueError(f"Expected pred_npy shape [B,L], got {pred_arr.shape}")
    if gt_arr.ndim != 2:
        raise ValueError(f"Expected gt_npy shape [B,L], got {gt_arr.shape}")

    n = min(pred_arr.shape[0], gt_arr.shape[0])
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    pred_arr = pred_arr[:n]
    gt_arr = gt_arr[:n]

    id_to_label = load_id_to_label(args.label_map_json)

    pred_layouts = [
        decode_layoutdiffusion_pred_row(
            row,
            coord_grid=args.coord_grid,
            layout_dim=args.layout_dim,
            pad_id=args.pad_id,
            eos_id=args.eos_id,
            id_to_label=id_to_label,
        )
        for row in pred_arr
    ]

    gt_layouts = [
        decode_blt_gt_row(
            row,
            coord_grid=args.coord_grid,
            layout_dim=args.layout_dim,
            pad_id=args.pad_id,
            eos_id=args.eos_id,
            id_to_label=id_to_label,
        )
        for row in gt_arr
    ]

    samples = [
        LayoutSample(pred_objects=pred_layouts[i], gt_objects=gt_layouts[i])
        for i in range(n)
    ]

    pred_counts = [len(x) for x in pred_layouts]
    gt_counts = [len(x) for x in gt_layouts]

    print(f"num samples: {n}")
    print(f"pred shape: {pred_arr.shape}")
    print(f"gt shape: {gt_arr.shape}")
    print(f"avg pred objects: {float(np.mean(pred_counts)):.3f}")
    print(f"avg gt objects: {float(np.mean(gt_counts)):.3f}")
    print(f"max pred objects: {int(np.max(pred_counts)) if pred_counts else 0}")
    print(f"max gt objects: {int(np.max(gt_counts)) if gt_counts else 0}")

    metrics = evaluate_layout_f1(samples, iou_thresh=args.iou_thresh)

    print("\n=== LayoutDiffusion layout metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()