# scripts/Layout_baselines/evaluate_BLT.py
import os
import json
import argparse
from typing import List

import numpy as np

from evaluation.layout_metrics import LayoutObject, LayoutSample, evaluate_layout_f1

DATASET = 'coco_stuff'

def cxcywh_to_xyxy_one(cx, cy, w, h, coord_grid: int):
    denom = float(coord_grid - 1)

    cx = float(cx) / denom
    cy = float(cy) / denom
    w = float(w) / denom
    h = float(h) / denom

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return np.array(
        [
            np.clip(x1, 0.0, 1.0),
            np.clip(y1, 0.0, 1.0),
            np.clip(x2, 0.0, 1.0),
            np.clip(y2, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def decode_blt_offset_layout(
    seq: np.ndarray,
    coord_grid: int = 128,
    layout_dim: int = 5,
    pad_id: int = 0,
    eos_id: int = 1,
) -> List[LayoutObject]:
    """
    Decodes BLT test() outputs after offset subtraction.

    Expected repeated format:
        category, x, y, w, h, category, x, y, w, h, ...

    Coordinates are local discrete ids in [0, coord_grid - 1].
    """
    seq = np.asarray(seq, dtype=np.int64).reshape(-1)

    usable_len = (len(seq) // layout_dim) * layout_dim
    objects = []

    for i in range(0, usable_len, layout_dim):
        cls_id, x, y, w, h = seq[i : i + layout_dim].tolist()

        if cls_id in (pad_id, eos_id):
            continue

        if min(x, y, w, h) < 0:
            continue

        if max(x, y, w, h) >= coord_grid:
            continue

        if w <= 0 or h <= 0:
            continue

        bbox = cxcywh_to_xyxy_one(x, y, w, h, coord_grid=coord_grid)

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        objects.append(
            LayoutObject(
                label=f"cls_{int(cls_id)}",
                bbox=bbox,
            )
        )

    return objects


def save_metrics(metrics, output_path):
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
        description="Evaluate BLT generated layouts against BLT real/reference layouts."
    )

    parser.add_argument("--pred_npy", default=f"../SG_baselines/google-research/layout-blt/ckpt/{DATASET}/test/generated_samples.npy", help="Generated BLT .npy, shape [B,L].")
    parser.add_argument("--gt_npy", default=f"../SG_baselines/google-research/layout-blt/ckpt/{DATASET}/test/reference_samples.npy", help="Reference BLT .npy, shape [B,L].")
    parser.add_argument("--output_path", default=f'./output/Layout_baselines/BLT/{DATASET}')

    parser.add_argument("--coord_grid", type=int, default=128)
    parser.add_argument("--layout_dim", type=int, default=5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--eos_id", type=int, default=1)

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

    pred_layouts = [
        decode_blt_offset_layout(
            seq,
            coord_grid=args.coord_grid,
            layout_dim=args.layout_dim,
            pad_id=args.pad_id,
            eos_id=args.eos_id,
        )
        for seq in pred_arr
    ]

    gt_layouts = [
        decode_blt_offset_layout(
            seq,
            coord_grid=args.coord_grid,
            layout_dim=args.layout_dim,
            pad_id=args.pad_id,
            eos_id=args.eos_id,
        )
        for seq in gt_arr
    ]

    samples = [
        LayoutSample(pred_objects=pred_layouts[i], gt_objects=gt_layouts[i])
        for i in range(n)
    ]

    print(f"num samples: {n}")
    print(f"pred shape: {pred_arr.shape}")
    print(f"gt shape: {gt_arr.shape}")
    print(f"avg pred objects: {np.mean([len(x) for x in pred_layouts]):.3f}")
    print(f"avg gt objects: {np.mean([len(x) for x in gt_layouts]):.3f}")

    metrics = evaluate_layout_f1(samples, iou_thresh=args.iou_thresh)

    print("\n=== Layout metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()

