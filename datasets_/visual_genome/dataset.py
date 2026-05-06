from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets_.visual_genome.collate import scene_graph_collate_fn


class SceneGraphDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        return_boxes: bool = False,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        blob = np.load(npz_path, allow_pickle=True)

        self.obj_labels = blob["obj_labels"]          # [M, N]
        self.rel_labels = blob["rel_labels"]          # [M, N, N]
        self.node_mask = blob["node_mask"]            # [M, N]
        self.edge_mask = blob["edge_mask"]            # [M, N, N]

        self.boxes = blob["boxes"] if "boxes" in blob else None
        self.image_index = blob["image_index"] if "image_index" in blob else None
        self.kept_old_indices = blob["kept_old_indices"] if "kept_old_indices" in blob else None

        self.n_max = int(blob["n_max"][0])
        self.object_vocab = blob["object_vocab"].tolist()
        self.relation_vocab = blob["relation_vocab"].tolist()
        self.split = blob["split"][0] if "split" in blob else None

        self.return_boxes = return_boxes
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return self.obj_labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = {
            "obj_labels": torch.as_tensor(self.obj_labels[idx], dtype=torch.long),
            "rel_labels": torch.as_tensor(self.rel_labels[idx], dtype=torch.long),
            "node_mask": torch.as_tensor(self.node_mask[idx], dtype=torch.bool),
            "edge_mask": torch.as_tensor(self.edge_mask[idx], dtype=torch.bool),
        }

        if self.return_boxes and self.boxes is not None:
            boxes_xyxy = torch.as_tensor(self.boxes[idx], dtype=torch.float32)  # [N,4]

            x1 = boxes_xyxy[:, 0]
            y1 = boxes_xyxy[:, 1]
            x2 = boxes_xyxy[:, 2]
            y2 = boxes_xyxy[:, 3]

            # Stored preprocessing used boxes_1024, so coordinates are in 1024-scale image space.
            W = 1024.0
            H = 1024.0

            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H

            boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)

            # Keep valid boxes in [0,1]; padded all-zero boxes remain all-zero.
            boxes_cxcywh = boxes_cxcywh.clamp(min=0.0, max=1.0)

            # IMPORTANT: keep the same key so the rest of the code does not break.
            out["boxes"] = boxes_cxcywh
            out["boxes_xyxy"] = boxes_xyxy

        if self.return_metadata:
            if self.image_index is not None:
                out["image_index"] = int(self.image_index[idx])
            if self.kept_old_indices is not None:
                out["kept_old_indices"] = torch.as_tensor(self.kept_old_indices[idx], dtype=torch.long)
            if self.split is not None:
                out["split"] = self.split

        return out
    


def build_scene_graph_dataloader(
    npz_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    return_boxes: bool = False,
    return_metadata: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = SceneGraphDataset(
        npz_path=npz_path,
        return_boxes=return_boxes,
        return_metadata=return_metadata,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=scene_graph_collate_fn,
    )
    return loader


def decode_item(
    obj_labels,
    rel_labels,
    node_mask,
    edge_mask,
    object_vocab,
    relation_vocab,
    no_rel_token="__no_relation__",
    mask_obj_token_id=None,
    mask_obj_token_name="[MASK_OBJ]",
):
    nodes = []
    for i in range(len(obj_labels)):
        if bool(node_mask[i]):
            obj_id = int(obj_labels[i])

            if mask_obj_token_id is not None and obj_id == mask_obj_token_id:
                nodes.append(mask_obj_token_name)
            else:
                nodes.append(object_vocab[obj_id])
        else:
            nodes.append("__pad__")

    triplets = []
    N = len(obj_labels)
    for i in range(N):
        if not bool(node_mask[i]):
            continue
        for j in range(N):
            if not bool(edge_mask[i, j]):
                continue
            rel = relation_vocab[int(rel_labels[i, j])]
            if rel == no_rel_token:
                continue
            triplets.append((i, nodes[i], rel, j, nodes[j]))

    return nodes, triplets

def format_decoded_graph(nodes, triplets, max_triplets: int = 30) -> str:
    lines = []
    lines.append("Nodes:")
    for i, n in enumerate(nodes):
        if n != "__pad__":
            lines.append(f"  [{i:02d}] {n}")
    lines.append("Triplets:")
    if len(triplets) == 0:
        lines.append("  (none)")
    else:
        for t in triplets[:max_triplets]:
            s_idx, s_name, rel, o_idx, o_name = t
            lines.append(f"  [{s_idx:02d}] {s_name} --{rel}--> [{o_idx:02d}] {o_name}")
    return "\n".join(lines)


def format_graph_triplets_only(triplets, max_triplets: int = 25) -> str:
    if len(triplets) == 0:
        return "(none)"
    lines = []
    for t in triplets[:max_triplets]:
        s_idx, s_name, rel, o_idx, o_name = t
        lines.append(f"[{s_idx:02d}] {s_name} --{rel}--> [{o_idx:02d}] {o_name}")
    return "\n".join(lines)


def format_nodes_with_boxes(
        nodes,
        boxes,
        node_mask,
        box_valid_mask=None,
        max_nodes: int = 30,
        box_format: str = "cxcywh",
        precision: int = 3,
    ) -> str:
        """
        nodes: list[str]
        boxes: tensor/list/ndarray [N,4]
        node_mask: tensor/list [N]
        box_valid_mask: tensor/list [N] or None
        """
        if boxes is None:
            return "(none)"

        lines = []
        n = min(len(nodes), max_nodes)

        for i in range(n):
            if not bool(node_mask[i]):
                continue

            valid = True if box_valid_mask is None else bool(box_valid_mask[i])

            if not valid:
                lines.append(f"[{i:02d}] {nodes[i]} : box=None")
                continue

            b = boxes[i]
            x0 = float(b[0])
            x1 = float(b[1])
            x2 = float(b[2])
            x3 = float(b[3])

            if box_format == "cxcywh":
                lines.append(
                    f"[{i:02d}] {nodes[i]} : "
                    f"(cx={x0:.{precision}f}, cy={x1:.{precision}f}, "
                    f"w={x2:.{precision}f}, h={x3:.{precision}f})"
                )
            else:
                lines.append(
                    f"[{i:02d}] {nodes[i]} : "
                    f"({x0:.{precision}f}, {x1:.{precision}f}, "
                    f"{x2:.{precision}f}, {x3:.{precision}f})"
                )

        return "\n".join(lines) if len(lines) > 0 else "(none)"

