import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np

DATASET = 'coco_stuff'
from evaluation.layout_metrics import (
    LayoutObject,
    LayoutSample,
    evaluate_layout_f1,
)


def load_idx_to_word(idx_to_word_path: str):
    import pickle

    with open(idx_to_word_path, "rb") as f:
        idx_to_word = pickle.load(f)

    if "ind_to_classes" not in idx_to_word:
        raise KeyError("idx_to_word.pkl must contain key 'ind_to_classes'")

    return idx_to_word


def safe_lookup(vocab, idx: int, fallback_prefix: str = "cls"):
    try:
        return str(vocab[int(idx)])
    except Exception:
        return f"{fallback_prefix}_{int(idx)}"


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    boxes: [N,4] normalized cx,cy,w,h
    returns: [N,4] normalized x1,y1,x2,y2
    """
    boxes = boxes.astype(np.float32)

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    out = np.stack([x1, y1, x2, y2], axis=-1)
    return np.clip(out, 0.0, 1.0)


def valid_box_mask_xyxy(boxes_xyxy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    return (w > eps) & (h > eps)


def build_layout_objects(
    labels: np.ndarray,
    boxes_cxcywh: np.ndarray,
    node_flags: np.ndarray,
    ind_to_classes,
) -> List[LayoutObject]:
    """
    labels: [N]
    boxes_cxcywh: [N,4]
    node_flags: [N]
    """
    labels = np.asarray(labels)
    boxes_cxcywh = np.asarray(boxes_cxcywh, dtype=np.float32)
    node_flags = np.asarray(node_flags).astype(bool)

    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    box_valid = valid_box_mask_xyxy(boxes_xyxy)

    objects = []
    for i in range(labels.shape[0]):
        if not node_flags[i]:
            continue
        if not box_valid[i]:
            continue

        label = safe_lookup(ind_to_classes, int(labels[i]))
        bbox = boxes_xyxy[i].astype(np.float32)

        objects.append(LayoutObject(label=label, bbox=bbox))

    return objects


def infer_node_flags_from_boxes(boxes_cxcywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_cxcywh, dtype=np.float32)
    return np.any(np.abs(boxes) > 1e-8, axis=-1)


def load_diffusesg_layout_samples(
    npz_path: str,
    idx_to_word_path: str,
    pred_x_key: str = "samples_x",
    pred_bbox_key: str = "samples_x_bbox",
    pred_flags_key: str = "samples_node_flags",
    gt_x_key: str = "gt_x",
    gt_bbox_key: str = "gt_x_bbox",
    gt_flags_key: str = "gt_node_flags",
) -> List[LayoutSample]:
    data = np.load(npz_path, allow_pickle=True)

    idx_to_word = load_idx_to_word(idx_to_word_path)
    ind_to_classes = idx_to_word["ind_to_classes"]

    required = [pred_x_key, pred_bbox_key, gt_x_key, gt_bbox_key]
    for k in required:
        if k not in data:
            raise KeyError(
                f"Missing key '{k}' in {npz_path}. Available keys: {list(data.keys())}"
            )

    pred_x = data[pred_x_key]
    pred_bbox = data[pred_bbox_key]
    gt_x = data[gt_x_key]
    gt_bbox = data[gt_bbox_key]

    if pred_bbox is None or gt_bbox is None:
        raise ValueError(
            "NPZ does not contain bbox arrays. "
            "Expected samples_x_bbox and gt_x_bbox to be present."
        )

    if pred_flags_key in data:
        pred_flags = data[pred_flags_key]
    else:
        pred_flags = np.stack([infer_node_flags_from_boxes(b) for b in pred_bbox], axis=0)

    if gt_flags_key in data:
        gt_flags = data[gt_flags_key]
    else:
        gt_flags = np.stack([infer_node_flags_from_boxes(b) for b in gt_bbox], axis=0)

    B = min(pred_x.shape[0], gt_x.shape[0])

    samples = []
    for b in range(B):
        pred_objects = build_layout_objects(
            labels=pred_x[b],
            boxes_cxcywh=pred_bbox[b],
            node_flags=pred_flags[b],
            ind_to_classes=ind_to_classes,
        )

        gt_objects = build_layout_objects(
            labels=gt_x[b],
            boxes_cxcywh=gt_bbox[b],
            node_flags=gt_flags[b],
            ind_to_classes=ind_to_classes,
        )

        samples.append(
            LayoutSample(
                pred_objects=pred_objects,
                gt_objects=gt_objects,
            )
        )

    return samples


def save_metrics(metrics: Dict[str, Any], output_path: str):
    os.makedirs(output_path, exist_ok=True)

    txt_path = os.path.join(output_path, "layout_metrics_summary.txt")
    json_path = os.path.join(output_path, "layout_metrics_summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved: {txt_path}")
    print(f"[OK] Saved: {json_path}")


def print_quick_stats(samples: List[LayoutSample]):
    pred_counts = [len(s.pred_objects) for s in samples]
    gt_counts = [len(s.gt_objects) for s in samples]

    print("\n=== Layout Eval Stats ===")
    print(f"num_samples: {len(samples)}")
    print(f"pred objects mean: {np.mean(pred_counts):.3f}")
    print(f"pred objects min/max: {np.min(pred_counts)} / {np.max(pred_counts)}")
    print(f"gt objects mean: {np.mean(gt_counts):.3f}")
    print(f"gt objects min/max: {np.min(gt_counts)} / {np.max(gt_counts)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DiffuseSG layout predictions using layout F1 metrics."
    )

    parser.add_argument(
        "--npz_path",
        type=str,
        # default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_8_patch_1_node_bits_edge_bits_sample_1000_Apr-27-10-19-37/sampling_during_evaluation/{DATASET}_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-10-19-46_model_inference/final_samples_array_before_eval.npz",
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_10_patch_1_node_bits_edge_bits_sample_1000_Apr-27-11-57-46/sampling_during_evaluation/coco_stuff_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-11-57-49_model_inference/final_samples_array_before_eval.npz",
        help="Path to final_samples_array_before_eval.npz",
    )

    parser.add_argument(
        "--idx_to_word_path",
        type=str,
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/idx_to_word.pkl",
        help="Path to idx_to_word.pkl containing ind_to_classes.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=f'./output/Layout_baselines/DiffuseSG/{DATASET}',
        help="Directory to save layout metrics.",
    )

    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--pred_x_key", type=str, default="samples_x")
    parser.add_argument("--pred_bbox_key", type=str, default="samples_x_bbox")
    parser.add_argument("--pred_flags_key", type=str, default="samples_node_flags")

    parser.add_argument("--gt_x_key", type=str, default="gt_x")
    parser.add_argument("--gt_bbox_key", type=str, default="gt_x_bbox")
    parser.add_argument("--gt_flags_key", type=str, default="gt_node_flags")

    return parser.parse_args()


def main():
    args = parse_args()

    samples = load_diffusesg_layout_samples(
        npz_path=args.npz_path,
        idx_to_word_path=args.idx_to_word_path,
        pred_x_key=args.pred_x_key,
        pred_bbox_key=args.pred_bbox_key,
        pred_flags_key=args.pred_flags_key,
        gt_x_key=args.gt_x_key,
        gt_bbox_key=args.gt_bbox_key,
        gt_flags_key=args.gt_flags_key,
    )

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print_quick_stats(samples)

    metrics = evaluate_layout_f1(
        samples=samples,
        iou_thresh=args.iou_thresh,
    )

    print("\n=== LAYOUT METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()