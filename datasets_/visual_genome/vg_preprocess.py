import json
import h5py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

# -----------------------------
# Utilities
# -----------------------------

def _count_repeated_labels(labels: np.ndarray) -> Tuple[int, int]:
    """
    labels: [K]
    returns:
      repeated_count = total repeated instances beyond first occurrence
      max_label_count = maximum count of any one label
    """
    if labels.size == 0:
        return 0, 0
    _, counts = np.unique(labels, return_counts=True)
    repeated_count = int(np.maximum(counts - 1, 0).sum())
    max_label_count = int(counts.max())
    return repeated_count, max_label_count


def _mean_pairwise_box_iou(boxes: np.ndarray) -> float:
    """
    boxes: [K,4] xyxy
    mean IoU over off-diagonal pairs
    """
    if boxes.shape[0] <= 1:
        return 0.0
    iou = bbox_overlaps(boxes, boxes)
    mask = ~np.eye(iou.shape[0], dtype=bool)
    if mask.sum() == 0:
        return 0.0
    return float(iou[mask].mean())


def _count_isolated_nodes_from_relations(
    rel_labels: np.ndarray,
    edge_mask: np.ndarray,
    node_mask: np.ndarray,
    no_rel_id: int,
) -> Tuple[int, float]:
    """
    rel_labels: [N,N]
    edge_mask:  [N,N]
    node_mask:  [N]
    """
    valid_nodes = node_mask.astype(bool)
    valid_pairs = edge_mask.astype(bool)
    pos_edges = valid_pairs & (rel_labels != no_rel_id)

    out_deg = pos_edges.sum(axis=1)
    in_deg = pos_edges.sum(axis=0)
    total_deg = out_deg + in_deg

    isolated = valid_nodes & (total_deg == 0)
    num_valid = int(valid_nodes.sum())
    num_isolated = int(isolated.sum())
    isolated_frac = float(num_isolated / max(num_valid, 1))
    return num_isolated, isolated_frac

def _invert_mapping(d: Dict[str, int]) -> List[str]:
    """Return index -> token list for a 0/1-based mapping dict."""
    max_idx = max(d.values())
    out = [""] * (max_idx + 1)
    for k, v in d.items():
        out[v] = k
    return out


def _bbox_xywh_center_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [xc, yc, w, h] to [x1, y1, x2, y2].
    Matches the convention used in neural-motifs loader.
    """
    boxes = boxes.astype(np.float32).copy()
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2.0
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes


def bbox_overlaps(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Simple IoU matrix for optional overlap filtering.
    boxes1: [N, 4], boxes2: [M, 4] in xyxy
    returns: [N, M]
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    out = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        x11, y11, x12, y12 = boxes1[i]
        a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
        for j in range(m):
            x21, y21, x22, y22 = boxes2[j]
            a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)

            ix1 = max(x11, x21)
            iy1 = max(y11, y21)
            ix2 = min(x12, x22)
            iy2 = min(y12, y22)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = a1 + a2 - inter
            out[i, j] = inter / union if union > 0 else 0.0

    return out


def _box_centers_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return np.stack([cx, cy], axis=-1).astype(np.float32)


def _box_areas_xyxy(boxes: np.ndarray) -> np.ndarray:
    w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return (w * h).astype(np.float32)


def _min_distance_to_selected(candidate_center: np.ndarray, selected_centers: np.ndarray) -> float:
    """
    candidate_center: [2]
    selected_centers: [K,2]
    returns minimum Euclidean distance
    """
    if selected_centers.shape[0] == 0:
        return 0.0
    d = np.linalg.norm(selected_centers - candidate_center[None, :], axis=1)
    return float(d.min())


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = x.mean() if x.size > 0 else 0.0
    sd = x.std() if x.size > 0 else 0.0
    if sd < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / sd


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class RawSceneGraph:
    image_index: int
    boxes: np.ndarray              # [num_obj, 4] xyxy
    obj_labels: np.ndarray         # [num_obj]
    relations: np.ndarray          # [num_rel, 3] => [subj_idx, obj_idx, pred]
    split: str                     # train/val/test


@dataclass
class ProcessedSceneGraph:
    image_index: int
    split: str
    obj_labels: np.ndarray         # [N_max]
    rel_labels: np.ndarray         # [N_max, N_max]
    node_mask: np.ndarray          # [N_max]
    edge_mask: np.ndarray          # [N_max, N_max]
    kept_old_indices: np.ndarray   # [num_kept]
    boxes: Optional[np.ndarray]    # [N_max, 4], padded with zeros
    triplets: List[Tuple[int, int, int]]  # kept triplets in new indexing


# -----------------------------
# Main loader / preprocessor
# -----------------------------

class VGSceneGraphPreprocessor:
    """
    Preprocess Stanford-filtered Visual Genome scene graph H5/JSON files into
    a fixed-size tensor representation for discrete diffusion.

    New object_selection modes:
      - degree
      - original
      - degree_spread
      - degree_spread_softcap
      - anchor_degree_spread
      - anchor_degree_spread_softcap
    """

    def __init__(
        self,
        h5_path: str,
        dict_json_path: str,
        image_data_json_path: Optional[str] = None,
        n_max: int = 20,
        box_scale_key: str = "boxes_1024",
        filter_empty_rels: bool = False,
        filter_non_overlap: bool = False,
        drop_background: bool = True,
        pad_token: str = "__pad__",
        no_rel_token: str = "__no_relation__",
        keep_first_relation: bool = True,
        object_selection: str = "degree",
        # ----- new selection hyperparameters -----
        degree_weight: float = 1.0,
        area_weight: float = 0.5,
        center_weight: float = 0.25,
        spread_weight: float = 1.0,
        repeat_penalty_weight: float = 0.75,
        repeat_penalty_type: str = "log",   # "linear" or "log"
        image_canvas_size: float = 1024.0,
                # ----- graph filtering -----
        enable_graph_filtering: bool = False,
        filter_min_nodes_for_low_rel: int = 10,
        filter_max_pos_edges_if_dense_nodes: int = 2,
        filter_max_repeated_labels: Optional[int] = None,
        filter_max_single_label_count: Optional[int] = None,
        filter_max_mean_box_iou: Optional[float] = None,
        filter_max_isolated_frac: Optional[float] = None,
    ) -> None:
        self.h5_path = h5_path
        self.dict_json_path = dict_json_path
        self.n_max = n_max
        self.box_scale_key = box_scale_key
        self.filter_empty_rels = filter_empty_rels
        self.filter_non_overlap = filter_non_overlap
        self.drop_background = drop_background
        self.pad_token = pad_token
        self.no_rel_token = no_rel_token
        self.keep_first_relation = keep_first_relation
        self.object_selection = object_selection
        self.image_data_json_path = image_data_json_path

        self.degree_weight = float(degree_weight)
        self.area_weight = float(area_weight)
        self.center_weight = float(center_weight)
        self.spread_weight = float(spread_weight)
        self.repeat_penalty_weight = float(repeat_penalty_weight)
        self.repeat_penalty_type = str(repeat_penalty_type)
        self.image_canvas_size = float(image_canvas_size)

        self.enable_graph_filtering = bool(enable_graph_filtering)
        self.filter_min_nodes_for_low_rel = int(filter_min_nodes_for_low_rel)
        self.filter_max_pos_edges_if_dense_nodes = int(filter_max_pos_edges_if_dense_nodes)
        self.filter_max_repeated_labels = filter_max_repeated_labels
        self.filter_max_single_label_count = filter_max_single_label_count
        self.filter_max_mean_box_iou = filter_max_mean_box_iou
        self.filter_max_isolated_frac = filter_max_isolated_frac

        self._load_dicts()
        self._open_h5()
        self._prepare_split_index()
        self._load_image_data()

        self.diff_obj_to_idx, self.diff_idx_to_obj = self._build_object_vocab()
        self.diff_rel_to_idx, self.diff_idx_to_rel = self._build_relation_vocab()

    # ---- loading ----

    def _load_dicts(self) -> None:
        with open(self.dict_json_path, "r") as f:
            info = json.load(f)

        if "label_to_idx" not in info or "predicate_to_idx" not in info:
            raise ValueError("dict json must contain 'label_to_idx' and 'predicate_to_idx'.")

        self.label_to_idx = info["label_to_idx"]
        self.predicate_to_idx = info["predicate_to_idx"]

        self.idx_to_label = _invert_mapping(self.label_to_idx)
        self.idx_to_predicate = _invert_mapping(self.predicate_to_idx)
        self.attribute_to_idx = info.get("attribute_to_idx", None)

    def _open_h5(self) -> None:
        self.h5 = h5py.File(self.h5_path, "r")

        required = [
            "split",
            "img_to_first_box",
            "img_to_last_box",
            "img_to_first_rel",
            "img_to_last_rel",
            "labels",
            "relationships",
            "predicates",
        ]
        for key in required:
            if key not in self.h5:
                raise KeyError(f"Missing required H5 key: {key}")

        if self.box_scale_key not in self.h5:
            available = [k for k in self.h5.keys() if k.startswith("boxes_")]
            raise KeyError(f"Missing '{self.box_scale_key}'. Available box keys: {available}")

        self.data_split = self.h5["split"][:]
        self.img_to_first_box = self.h5["img_to_first_box"][:]
        self.img_to_last_box = self.h5["img_to_last_box"][:]
        self.img_to_first_rel = self.h5["img_to_first_rel"][:]
        self.img_to_last_rel = self.h5["img_to_last_rel"][:]

        self.all_labels = self.h5["labels"][:, 0]
        self.all_boxes = _bbox_xywh_center_to_xyxy(self.h5[self.box_scale_key][:])
        self.all_relationships = self.h5["relationships"][:]
        self.all_predicates = self.h5["predicates"][:, 0]

    def _load_image_data(self) -> None:
        if self.image_data_json_path is not None:
            with open(self.image_data_json_path, "r") as f:
                self.image_data = json.load(f)
        else:
            self.image_data = None

    def _prepare_split_index(self) -> None:
        self.trainval_indices = np.where(
            (self.data_split == 0) & (self.img_to_first_box >= 0)
        )[0]
        self.test_indices = np.where(
            (self.data_split == 2) & (self.img_to_first_box >= 0)
        )[0]

        if self.filter_empty_rels:
            self.trainval_indices = self.trainval_indices[
                self.img_to_first_rel[self.trainval_indices] >= 0
            ]
            self.test_indices = self.test_indices[
                self.img_to_first_rel[self.test_indices] >= 0
            ]

    def _build_object_vocab(self) -> Tuple[Dict[str, int], List[str]]:
        tokens = [self.pad_token]
        for idx, name in enumerate(self.idx_to_label):
            if idx == 0 and self.drop_background:
                continue
            tokens.append(name)
        token_to_idx = {tok: i for i, tok in enumerate(tokens)}
        return token_to_idx, tokens

    def _build_relation_vocab(self) -> Tuple[Dict[str, int], List[str]]:
        tokens = [self.no_rel_token]
        for idx, name in enumerate(self.idx_to_predicate):
            if idx == 0 and self.drop_background:
                continue
            tokens.append(name)
        token_to_idx = {tok: i for i, tok in enumerate(tokens)}
        return token_to_idx, tokens

    # ---- split helpers ----

    def get_indices(self, split: str = "trainval") -> np.ndarray:
        split = split.lower()
        if split in ("train", "trainval", "val"):
            return self.trainval_indices.copy()
        if split == "test":
            return self.test_indices.copy()
        raise ValueError("split must be one of: train, val, trainval, test")

    # ---- raw graph extraction ----

    def load_raw_graph(self, image_index: int, split_name: str = "trainval") -> RawSceneGraph:
        fb = self.img_to_first_box[image_index]
        lb = self.img_to_last_box[image_index]
        fr = self.img_to_first_rel[image_index]
        lr = self.img_to_last_rel[image_index]

        if fb < 0 or lb < fb:
            raise ValueError(f"Image {image_index} has no valid boxes.")

        boxes = self.all_boxes[fb:lb + 1]
        labels = self.all_labels[fb:lb + 1]

        if fr >= 0 and lr >= fr:
            predicates = self.all_predicates[fr:lr + 1]
            rel_pairs_global = self.all_relationships[fr:lr + 1]
            rel_pairs_local = rel_pairs_global - fb
            rels = np.column_stack((rel_pairs_local, predicates)).astype(np.int64)
        else:
            rels = np.zeros((0, 3), dtype=np.int64)

        if self.filter_non_overlap and len(rels) > 0:
            overlaps = bbox_overlaps(boxes, boxes)
            keep = overlaps[rels[:, 0], rels[:, 1]] > 0.0
            rels = rels[keep]

        return RawSceneGraph(
            image_index=image_index,
            boxes=boxes,
            obj_labels=labels.astype(np.int64),
            relations=rels,
            split=split_name,
        )

    # ---- selection helpers ----

    def _repeat_penalty(self, current_count: int) -> float:
        if current_count <= 0:
            return 0.0
        if self.repeat_penalty_type == "linear":
            return float(current_count)
        elif self.repeat_penalty_type == "log":
            return float(np.log1p(current_count))
        else:
            raise ValueError("repeat_penalty_type must be 'linear' or 'log'")

    def _select_object_indices(
        self,
        raw_obj_labels: np.ndarray,   # [num_obj]
        raw_boxes: np.ndarray,        # [num_obj,4]
        rels: np.ndarray,             # [num_rel,3]
        valid_obj_mask: np.ndarray,   # [num_obj] bool
    ) -> np.ndarray:
        """
        Returns selected old indices, length <= n_max.
        """

        candidate_indices = np.where(valid_obj_mask)[0]
        if candidate_indices.size == 0:
            return np.zeros((0,), dtype=np.int64)

        # degree
        num_obj = raw_obj_labels.shape[0]
        degree = np.zeros(num_obj, dtype=np.int64)
        for s, o, _ in rels:
            degree[s] += 1
            degree[o] += 1

        if self.object_selection == "original":
            return np.array(candidate_indices[: self.n_max], dtype=np.int64)

        if self.object_selection == "degree":
            order = sorted(candidate_indices.tolist(), key=lambda i: (-degree[i], i))
            return np.array(order[: self.n_max], dtype=np.int64)

        # shared geometric stats for spread/anchor modes
        cand_boxes = raw_boxes[candidate_indices]                      # [C,4]
        cand_centers = _box_centers_xyxy(cand_boxes)                  # [C,2]
        cand_areas = _box_areas_xyxy(cand_boxes)                      # [C]
        cand_degrees = degree[candidate_indices].astype(np.float32)   # [C]

        deg_z = _safe_zscore(cand_degrees)
        area_z = _safe_zscore(cand_areas)

        # higher if closer to image center
        canvas_center = np.array([0.5 * self.image_canvas_size, 0.5 * self.image_canvas_size], dtype=np.float32)
        center_dist = np.linalg.norm(cand_centers - canvas_center[None, :], axis=1).astype(np.float32)
        center_score = -center_dist
        center_z = _safe_zscore(center_score)

        base_score = (
            self.degree_weight * deg_z
            + self.area_weight * area_z
            + self.center_weight * center_z
        ).astype(np.float32)

        if self.object_selection == "anchor_degree_spread":
            # seed by strong anchor score, then spread
            seed_order = np.argsort(-base_score)
            selected_local = [int(seed_order[0])]
            selected_set = {selected_local[0]}

            while len(selected_local) < min(self.n_max, candidate_indices.size):
                selected_centers = cand_centers[selected_local]
                best_local = None
                best_score = None
                for local_idx in range(candidate_indices.size):
                    if local_idx in selected_set:
                        continue
                    spread_gain = _min_distance_to_selected(cand_centers[local_idx], selected_centers)
                    score = float(base_score[local_idx] + self.spread_weight * spread_gain / self.image_canvas_size)
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_local = local_idx
                selected_local.append(int(best_local))
                selected_set.add(int(best_local))

            return np.array(candidate_indices[selected_local], dtype=np.int64)

        if self.object_selection == "degree_spread":
            # seed by degree only, then spread
            seed_order = np.argsort(-cand_degrees)
            selected_local = [int(seed_order[0])]
            selected_set = {selected_local[0]}

            while len(selected_local) < min(self.n_max, candidate_indices.size):
                selected_centers = cand_centers[selected_local]
                best_local = None
                best_score = None
                for local_idx in range(candidate_indices.size):
                    if local_idx in selected_set:
                        continue
                    spread_gain = _min_distance_to_selected(cand_centers[local_idx], selected_centers)
                    score = float(cand_degrees[local_idx] + self.spread_weight * spread_gain / self.image_canvas_size)
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_local = local_idx
                selected_local.append(int(best_local))
                selected_set.add(int(best_local))

            return np.array(candidate_indices[selected_local], dtype=np.int64)

        if self.object_selection in ("degree_spread_softcap", "anchor_degree_spread_softcap"):
            if self.object_selection == "degree_spread_softcap":
                seed_order = np.argsort(-cand_degrees)
            else:
                seed_order = np.argsort(-base_score)

            selected_local = [int(seed_order[0])]
            selected_set = {selected_local[0]}

            label_counts: Dict[int, int] = defaultdict(int)
            first_label = int(raw_obj_labels[candidate_indices[selected_local[0]]])
            label_counts[first_label] += 1

            while len(selected_local) < min(self.n_max, candidate_indices.size):
                selected_centers = cand_centers[selected_local]
                best_local = None
                best_score = None

                for local_idx in range(candidate_indices.size):
                    if local_idx in selected_set:
                        continue

                    old_idx = int(candidate_indices[local_idx])
                    label_id = int(raw_obj_labels[old_idx])
                    repeat_count = label_counts[label_id]
                    repeat_pen = self._repeat_penalty(repeat_count)

                    spread_gain = _min_distance_to_selected(
                        cand_centers[local_idx],
                        selected_centers,
                    ) / self.image_canvas_size

                    score = (
                        float(base_score[local_idx])
                        + self.spread_weight * float(spread_gain)
                        - self.repeat_penalty_weight * float(repeat_pen)
                    )

                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_local = local_idx

                selected_local.append(int(best_local))
                selected_set.add(int(best_local))

                chosen_old_idx = int(candidate_indices[best_local])
                chosen_label = int(raw_obj_labels[chosen_old_idx])
                label_counts[chosen_label] += 1

            return np.array(candidate_indices[selected_local], dtype=np.int64)

        raise ValueError(
            "object_selection must be one of: "
            "'degree', 'original', 'degree_spread', 'degree_spread_softcap', "
            "'anchor_degree_spread', 'anchor_degree_spread_softcap'"
        )

    # ---- conversion to diffusion-ready fixed-size tensors ----

    def process_raw_graph(self, raw: RawSceneGraph) -> ProcessedSceneGraph:
        num_obj = raw.obj_labels.shape[0]
        rels = raw.relations.copy()

        valid_obj_mask = np.ones(num_obj, dtype=bool)
        if self.drop_background:
            valid_obj_mask &= (raw.obj_labels != 0)

        if rels.shape[0] > 0:
            keep_rel = valid_obj_mask[rels[:, 0]] & valid_obj_mask[rels[:, 1]]
            rels = rels[keep_rel]

        kept_old_indices = self._select_object_indices(
            raw_obj_labels=raw.obj_labels,
            raw_boxes=raw.boxes,
            rels=rels,
            valid_obj_mask=valid_obj_mask,
        )

        old_to_new = {
            old_i: new_i for new_i, old_i in enumerate(kept_old_indices.tolist())
        }

        kept_triplets: List[Tuple[int, int, int]] = []
        pair_to_pred: Dict[Tuple[int, int], int] = {}

        for s_old, o_old, pred_old in rels.tolist():
            if s_old not in old_to_new or o_old not in old_to_new:
                continue
            s_new = old_to_new[s_old]
            o_new = old_to_new[o_old]
            if s_new == o_new:
                continue
            key = (s_new, o_new)
            if key in pair_to_pred:
                if self.keep_first_relation:
                    continue
            pair_to_pred[key] = int(pred_old)

        for (s_new, o_new), pred_old in pair_to_pred.items():
            kept_triplets.append((s_new, o_new, pred_old))

        obj_out = np.full(
            (self.n_max,),
            fill_value=self.diff_obj_to_idx[self.pad_token],
            dtype=np.int64,
        )
        rel_out = np.full(
            (self.n_max, self.n_max),
            fill_value=self.diff_rel_to_idx[self.no_rel_token],
            dtype=np.int64,
        )
        node_mask = np.zeros((self.n_max,), dtype=np.bool_)
        edge_mask = np.zeros((self.n_max, self.n_max), dtype=np.bool_)
        boxes_out = np.zeros((self.n_max, 4), dtype=np.float32)

        for new_i, old_i in enumerate(kept_old_indices.tolist()):
            raw_label_id = int(raw.obj_labels[old_i])
            raw_label_name = self.idx_to_label[raw_label_id]
            if self.drop_background and raw_label_id == 0:
                continue
            diff_obj_id = self.diff_obj_to_idx[raw_label_name]
            obj_out[new_i] = diff_obj_id
            node_mask[new_i] = True
            boxes_out[new_i] = raw.boxes[old_i]

        for i in range(self.n_max):
            for j in range(self.n_max):
                if i != j and node_mask[i] and node_mask[j]:
                    edge_mask[i, j] = True

        for s_new, o_new, pred_old in kept_triplets:
            pred_name = self.idx_to_predicate[int(pred_old)]
            diff_rel_id = self.diff_rel_to_idx[pred_name]
            rel_out[s_new, o_new] = diff_rel_id

        return ProcessedSceneGraph(
            image_index=raw.image_index,
            split=raw.split,
            obj_labels=obj_out,
            rel_labels=rel_out,
            node_mask=node_mask,
            edge_mask=edge_mask,
            kept_old_indices=kept_old_indices,
            boxes=boxes_out,
            triplets=kept_triplets,
        )

    def process_image_index(self, image_index: int, split_name: str = "trainval") -> ProcessedSceneGraph:
        raw = self.load_raw_graph(image_index=image_index, split_name=split_name)
        return self.process_raw_graph(raw)

    # ---- decoding / inspection ----

    def decode_object_id(self, diff_obj_id: int) -> str:
        return self.diff_idx_to_obj[int(diff_obj_id)]

    def decode_relation_id(self, diff_rel_id: int) -> str:
        return self.diff_idx_to_rel[int(diff_rel_id)]

    def decode_processed_graph(self, item: ProcessedSceneGraph) -> Dict[str, Any]:
        node_names = []
        for i in range(self.n_max):
            if item.node_mask[i]:
                node_names.append(self.decode_object_id(item.obj_labels[i]))
            else:
                node_names.append(self.pad_token)

        triplets = []
        for i in range(self.n_max):
            if not item.node_mask[i]:
                continue
            for j in range(self.n_max):
                if not item.edge_mask[i, j]:
                    continue
                rel_name = self.decode_relation_id(item.rel_labels[i, j])
                if rel_name == self.no_rel_token:
                    continue
                triplets.append((i, node_names[i], rel_name, j, node_names[j]))

        meta = None
        if self.image_data is not None:
            meta = self.image_data[item.image_index]

        out = {
            "image_index": item.image_index,
            "split": item.split,
            "nodes": node_names,
            "triplets": triplets,
            "node_mask": item.node_mask.astype(int).tolist(),
            "kept_old_indices": item.kept_old_indices.tolist(),
        }
        if meta is not None:
            out["image_id"] = meta.get("image_id")
            out["url"] = "/".join(meta.get("url").split("/")[-2:])
        return out

    def print_processed_graph(self, item: ProcessedSceneGraph) -> None:
        decoded = self.decode_processed_graph(item)
        print(f"\nImage index: {decoded['image_index']} | split={decoded['split']}")
        if "image_id" in decoded:
            print(f"Image id: {decoded['image_id']}")
        if "url" in decoded:
            print(f"URL tail: {decoded['url']}")
        print("Nodes:")
        for i, name in enumerate(decoded["nodes"]):
            if item.node_mask[i]:
                print(f"  [{i:02d}] {name}")
        print("Triplets:")
        if len(decoded["triplets"]) == 0:
            print("  (none)")
        else:
            for s_idx, s_name, rel, o_idx, o_name in decoded["triplets"]:
                print(f"  [{s_idx:02d}] {s_name} --{rel}--> [{o_idx:02d}] {o_name}")

    # ---- dataset export ----

    def export_split_packed(
        self,
        out_path: str,
        split: str = "trainval",
        limit: Optional[int] = None,
    ) -> None:
        indices = self.get_indices(split)
        if limit is not None:
            indices = indices[:limit]

        obj_labels_list = []
        rel_labels_list = []
        node_mask_list = []
        edge_mask_list = []
        boxes_list = []
        image_index_list = []
        kept_old_indices_list = []

        skipped = 0

        for idx in indices:
            try:
                proc = self.process_image_index(int(idx), split_name=split)
                if self.enable_graph_filtering:
                    should_filter, filter_stats = self.should_filter_processed_graph(proc)
                    if should_filter:
                        skipped += 1
                        continue
            except Exception as e:
                print(f"Skipping image_index={idx} because: {e}")
                skipped += 1
                continue
            

            obj_labels_list.append(proc.obj_labels)
            rel_labels_list.append(proc.rel_labels)
            node_mask_list.append(proc.node_mask)
            edge_mask_list.append(proc.edge_mask)
            boxes_list.append(proc.boxes)
            image_index_list.append(proc.image_index)

            kept = np.full((self.n_max,), fill_value=-1, dtype=np.int64)
            kept[: len(proc.kept_old_indices)] = proc.kept_old_indices
            kept_old_indices_list.append(kept)

        if len(obj_labels_list) == 0:
            raise RuntimeError("No valid processed examples found.")

        obj_labels = np.stack(obj_labels_list, axis=0).astype(np.int64)         # [M, N]
        rel_labels = np.stack(rel_labels_list, axis=0).astype(np.int64)         # [M, N, N]
        node_mask = np.stack(node_mask_list, axis=0).astype(np.bool_)           # [M, N]
        edge_mask = np.stack(edge_mask_list, axis=0).astype(np.bool_)           # [M, N, N]
        boxes = np.stack(boxes_list, axis=0).astype(np.float32)                 # [M, N, 4]
        image_index = np.array(image_index_list, dtype=np.int64)                # [M]
        kept_old_indices = np.stack(kept_old_indices_list, axis=0).astype(np.int64)

        np.savez_compressed(
            out_path,
            obj_labels=obj_labels,
            rel_labels=rel_labels,
            node_mask=node_mask,
            edge_mask=edge_mask,
            boxes=boxes,
            image_index=image_index,
            kept_old_indices=kept_old_indices,
            n_max=np.array([self.n_max], dtype=np.int64),
            object_vocab=np.array(self.diff_idx_to_obj, dtype=object),
            relation_vocab=np.array(self.diff_idx_to_rel, dtype=object),
            split=np.array([split], dtype=object),
        )

        print(f"Saved {obj_labels.shape[0]} items to {out_path} (skipped {skipped}).")
        print(f"obj_labels shape: {obj_labels.shape}")
        print(f"rel_labels shape: {rel_labels.shape}")

    def close(self) -> None:
        if hasattr(self, "h5") and self.h5 is not None:
            self.h5.close()
    
    def should_filter_processed_graph(
        self,
        proc: ProcessedSceneGraph,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns:
          should_filter: bool
          stats: dict of graph stats / reasons
        """
        node_mask = proc.node_mask.astype(bool)
        valid_labels = proc.obj_labels[node_mask]
        valid_boxes = proc.boxes[node_mask] if proc.boxes is not None else np.zeros((0, 4), dtype=np.float32)

        num_nodes = int(node_mask.sum())
        no_rel_id = self.diff_rel_to_idx[self.no_rel_token]

        pos_edges = (proc.rel_labels != no_rel_id) & proc.edge_mask.astype(bool)
        num_pos_edges = int(pos_edges.sum())

        repeated_count, max_label_count = _count_repeated_labels(valid_labels)
        mean_box_iou = _mean_pairwise_box_iou(valid_boxes)
        num_isolated, isolated_frac = _count_isolated_nodes_from_relations(
            rel_labels=proc.rel_labels,
            edge_mask=proc.edge_mask,
            node_mask=proc.node_mask,
            no_rel_id=no_rel_id,
        )

        reasons = []

        if (
            num_nodes >= self.filter_min_nodes_for_low_rel
            and num_pos_edges <= self.filter_max_pos_edges_if_dense_nodes
        ):
            reasons.append("dense_nodes_few_relations")

        if (
            self.filter_max_repeated_labels is not None
            and repeated_count > int(self.filter_max_repeated_labels)
        ):
            reasons.append("too_many_repeated_labels")

        if (
            self.filter_max_single_label_count is not None
            and max_label_count > int(self.filter_max_single_label_count)
        ):
            reasons.append("single_label_dominates")

        if (
            self.filter_max_mean_box_iou is not None
            and mean_box_iou > float(self.filter_max_mean_box_iou)
        ):
            reasons.append("high_box_overlap")

        if (
            self.filter_max_isolated_frac is not None
            and isolated_frac > float(self.filter_max_isolated_frac)
        ):
            reasons.append("too_many_isolated_nodes")

        stats = {
            "num_nodes": num_nodes,
            "num_pos_edges": num_pos_edges,
            "repeated_count": repeated_count,
            "max_label_count": max_label_count,
            "mean_box_iou": mean_box_iou,
            "num_isolated": num_isolated,
            "isolated_frac": isolated_frac,
            "reasons": reasons,
        }

        return len(reasons) > 0, stats