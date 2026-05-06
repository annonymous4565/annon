import os
import json
import pickle
import argparse
from typing import List

import numpy as np

from evaluation.layout_metrics import LayoutObject, LayoutSample, evaluate_layout_f1

DATASET="visual_genome"

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ind_to_classes(idx_to_word_path):
    obj = load_pickle(idx_to_word_path)
    if isinstance(obj, dict):
        return obj["ind_to_classes"]
    return obj[0]


def safe_lookup(vocab, idx):
    try:
        return str(vocab[int(idx)])
    except Exception:
        return f"cls_{int(idx)}"


def cxcywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.stack(
        [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ],
        axis=-1,
    )
    return np.clip(out, 0.0, 1.0)


def layoutdm_result_to_objects(result, ind_to_classes) -> List[LayoutObject]:
    """
    result:
        (boxes_cxcywh, labels)
    boxes_cxcywh:
        [N, 4], normalized cx, cy, w, h
    labels:
        [N]
    """
    boxes, labels = result
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    boxes_xyxy = cxcywh_to_xyxy(boxes)

    objects = []
    for box, lab in zip(boxes_xyxy, labels):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        objects.append(
            LayoutObject(
                label=safe_lookup(ind_to_classes, int(lab)),
                bbox=box.astype(np.float32),
            )
        )

    return objects


def load_reference_npz(reference_npz):
    data = np.load(reference_npz, allow_pickle=True)

    if "gt_x" not in data:
        raise KeyError(f"Missing gt_x in {reference_npz}. Keys: {list(data.keys())}")
    if "gt_x_bbox" not in data:
        raise KeyError(f"Missing gt_x_bbox in {reference_npz}. Keys: {list(data.keys())}")

    gt_x = data["gt_x"]
    gt_bbox = data["gt_x_bbox"]

    if "gt_node_flags" in data:
        gt_flags = data["gt_node_flags"].astype(bool)
    else:
        gt_flags = np.ones(gt_x.shape, dtype=bool)

    return gt_x, gt_bbox, gt_flags


def npz_gt_to_objects(labels, boxes, flags, ind_to_classes):
    labels = np.asarray(labels, dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    flags = np.asarray(flags).astype(bool)

    boxes_xyxy = cxcywh_to_xyxy(boxes)

    objects = []
    for lab, box, keep in zip(labels, boxes_xyxy, flags):
        if not keep:
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        objects.append(
            LayoutObject(
                label=safe_lookup(ind_to_classes, int(lab)),
                bbox=box.astype(np.float32),
            )
        )

    return objects


def save_metrics(metrics, output_path):
    os.makedirs(output_path, exist_ok=True)

    txt_path = os.path.join(output_path, "layout_metrics_summary.txt")
    json_path = os.path.join(output_path, "layout_metrics_summary.json")

    with open(txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] saved {txt_path}")
    print(f"[OK] saved {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_pkl", default='../SG_baselines/layout-dm/multirun/results/unconditional_temperature_1.0_name_random_num_timesteps_100/seed_0_new.pkl')
    parser.add_argument("--reference_npz", default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_8_patch_1_node_bits_edge_bits_sample_1000_Apr-27-10-19-37/sampling_during_evaluation/{DATASET}_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-10-19-46_model_inference/final_samples_array_before_eval.npz")
    # parser.add_argument("--reference_npz", default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_10_patch_1_node_bits_edge_bits_sample_1000_Apr-27-11-57-46/sampling_during_evaluation/coco_stuff_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-11-57-49_model_inference/final_samples_array_before_eval.npz")
    parser.add_argument("--idx_to_word_path", default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/idx_to_word.pkl")
    parser.add_argument("--output_path", default=f'./output/Layout_baselines/LayoutDM/{DATASET}')

    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0)

    args = parser.parse_args()

    pred_obj = load_pickle(args.pred_pkl)
    # if "results" not in pred_obj:
    #     raise KeyError(f"Expected key 'results' in {args.pred_pkl}")

    pred_results = pred_obj

    gt_x, gt_bbox, gt_flags = load_reference_npz(args.reference_npz)
    ind_to_classes = load_ind_to_classes(args.idx_to_word_path)

    n = min(len(pred_results), gt_x.shape[0])
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    pred_layouts = [
        layoutdm_result_to_objects(pred_results[i], ind_to_classes)
        for i in range(n)
    ]

    gt_layouts = [
    npz_gt_to_objects(
        labels=gt_x[i],
        boxes=gt_bbox[i],
        flags=gt_flags[i],
        ind_to_classes=ind_to_classes,
    )
    for i in range(n)
]

    samples = [
        LayoutSample(pred_objects=pred_layouts[i], gt_objects=gt_layouts[i])
        for i in range(n)
    ]

    print(f"num samples: {n}")
    print(f"avg pred objects: {np.mean([len(x) for x in pred_layouts]):.3f}")
    print(f"avg gt objects: {np.mean([len(x) for x in gt_layouts]):.3f}")

    metrics = evaluate_layout_f1(samples, iou_thresh=args.iou_thresh)

    print("\n=== LayoutDM layout metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()