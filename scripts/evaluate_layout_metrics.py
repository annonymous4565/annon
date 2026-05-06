import os
import json
import argparse
import numpy as np

from evaluation.layout_metrics import LayoutObject, LayoutSample, evaluate_layout_f1


def load_layout_samples(pred_path: str, gt_path: str):
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_obj = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_obj = json.load(f)

    if len(pred_obj) != len(gt_obj):
        raise ValueError(
            f"Prediction and GT layout files must have same number of samples: "
            f"{len(pred_obj)} vs {len(gt_obj)}"
        )

    samples = []
    for idx, (pred_item, gt_item) in enumerate(zip(pred_obj, gt_obj)):
        if "objects" not in pred_item:
            raise ValueError(f"pred sample {idx} missing 'objects'")
        if "objects" not in gt_item:
            raise ValueError(f"gt sample {idx} missing 'objects'")

        pred_objects = []
        for o in pred_item["objects"]:
            pred_objects.append(
                LayoutObject(
                    label=str(o["label"]),
                    bbox=np.array(o["bbox"], dtype=float),
                )
            )

        gt_objects = []
        for o in gt_item["objects"]:
            gt_objects.append(
                LayoutObject(
                    label=str(o["label"]),
                    bbox=np.array(o["bbox"], dtype=float),
                )
            )

        samples.append(
            LayoutSample(
                pred_objects=pred_objects,
                gt_objects=gt_objects,
            )
        )

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_layout_json", type=str, required=True)
    parser.add_argument("--gt_layout_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    samples = load_layout_samples(args.pred_layout_json, args.gt_layout_json)
    metrics = evaluate_layout_f1(samples, iou_thresh=args.iou_thresh)

    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, "layout_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== LAYOUT METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()