from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np


@dataclass
class LayoutObject:
    label: str
    bbox: np.ndarray  # [x1, y1, x2, y2]


@dataclass
class LayoutSample:
    """
    One conditional example.
    pred_objects and gt_objects are lists of detected/generated boxes and GT boxes.
    """
    pred_objects: List[LayoutObject]
    gt_objects: List[LayoutObject]


def bbox_area(box: np.ndarray) -> float:
    w = max(float(box[2] - box[0]), 0.0)
    h = max(float(box[3] - box[1]), 0.0)
    return w * h


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))

    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih

    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def greedy_match_by_iou(
    pred_objects: List[LayoutObject],
    gt_objects: List[LayoutObject],
    class_aware: bool,
    iou_thresh: float,
) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """
    Greedy 1-1 matching by IoU.
    Returns:
        tp, fp, fn, matches
    """
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []

    for pi, pobj in enumerate(pred_objects):
        best_gj = None
        best_iou = -1.0

        for gj, gobj in enumerate(gt_objects):
            if gj in used_gt:
                continue
            if class_aware and pobj.label != gobj.label:
                continue

            iou = bbox_iou(pobj.bbox, gobj.bbox)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gj = gj

        if best_gj is not None:
            used_gt.add(best_gj)
            matches.append((pi, best_gj, best_iou))

    tp = len(matches)
    fp = len(pred_objects) - tp
    fn = len(gt_objects) - tp
    return tp, fp, fn, matches


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def compute_class_frequency_weights(samples: Sequence[LayoutSample]) -> Dict[str, float]:
    counts = Counter()
    for s in samples:
        for obj in s.gt_objects:
            counts[obj.label] += 1

    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def compute_class_area_weights(samples: Sequence[LayoutSample]) -> Dict[str, float]:
    area_sum = defaultdict(float)
    count = defaultdict(int)

    for s in samples:
        for obj in s.gt_objects:
            area_sum[obj.label] += bbox_area(obj.bbox)
            count[obj.label] += 1

    avg_area = {k: area_sum[k] / max(count[k], 1) for k in area_sum}
    total = sum(avg_area.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in avg_area.items()}


def get_all_gt_classes(samples: Sequence[LayoutSample]) -> List[str]:
    classes = sorted({obj.label for s in samples for obj in s.gt_objects})
    return classes


def per_class_f1_counts(
    samples: Sequence[LayoutSample],
    iou_thresh: float = 0.5,
) -> Dict[str, Dict[str, int]]:
    """
    Compute TP/FP/FN per class by matching within each class separately.
    """
    classes = get_all_gt_classes(samples)
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for s in samples:
        for c in classes:
            pred_c = [o for o in s.pred_objects if o.label == c]
            gt_c = [o for o in s.gt_objects if o.label == c]

            tp, fp, fn, _ = greedy_match_by_iou(
                pred_objects=pred_c,
                gt_objects=gt_c,
                class_aware=False,   # already filtered by class
                iou_thresh=iou_thresh,
            )
            stats[c]["tp"] += tp
            stats[c]["fp"] += fp
            stats[c]["fn"] += fn

    return stats


def evaluate_layout_f1(
    samples: Sequence[LayoutSample],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    bbox metrics:
      - F1-Vanilla: unweighted mean across classes
      - F1-Area: class F1 weighted by average GT box area
      - F1-Frequency: class F1 weighted by GT class frequency
      - F1-BO: bbox-only F1, class-agnostic
    """
    if len(samples) == 0:
        return {
            "f1_vanilla": 0.0,
            "f1_area": 0.0,
            "f1_frequency": 0.0,
            "f1_bo": 0.0,
        }

    classes = get_all_gt_classes(samples)
    class_stats = per_class_f1_counts(samples, iou_thresh=iou_thresh)

    class_f1 = {}
    for c in classes:
        tp = class_stats[c]["tp"]
        fp = class_stats[c]["fp"]
        fn = class_stats[c]["fn"]
        class_f1[c] = f1_from_counts(tp, fp, fn)

    # F1-Vanilla
    f1_vanilla = float(np.mean([class_f1[c] for c in classes])) if classes else 0.0

    # F1-Frequency
    freq_w = compute_class_frequency_weights(samples)
    f1_frequency = 0.0
    for c in classes:
        f1_frequency += freq_w.get(c, 0.0) * class_f1[c]

    # F1-Area
    area_w = compute_class_area_weights(samples)
    f1_area = 0.0
    for c in classes:
        f1_area += area_w.get(c, 0.0) * class_f1[c]

    # F1-BO (class-agnostic)
    total_tp_bo = 0
    total_fp_bo = 0
    total_fn_bo = 0
    for s in samples:
        tp, fp, fn, _ = greedy_match_by_iou(
            pred_objects=s.pred_objects,
            gt_objects=s.gt_objects,
            class_aware=False,
            iou_thresh=iou_thresh,
        )
        total_tp_bo += tp
        total_fp_bo += fp
        total_fn_bo += fn

    f1_bo = f1_from_counts(total_tp_bo, total_fp_bo, total_fn_bo)

    return {
        "f1_vanilla": float(f1_vanilla),
        "f1_area": float(f1_area),
        "f1_frequency": float(f1_frequency),
        "f1_bo": float(f1_bo),
    }