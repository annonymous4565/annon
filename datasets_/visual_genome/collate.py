from typing import List, Dict, Any
import torch



def scene_graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "obj_labels": torch.stack([x["obj_labels"] for x in batch], dim=0),   # [B, N]
        "rel_labels": torch.stack([x["rel_labels"] for x in batch], dim=0),   # [B, N, N]
        "node_mask": torch.stack([x["node_mask"] for x in batch], dim=0),     # [B, N]
        "edge_mask": torch.stack([x["edge_mask"] for x in batch], dim=0),     # [B, N, N]
    }

    if "boxes" in batch[0]:
        out["boxes"] = torch.stack([x["boxes"] for x in batch], dim=0)  # [B, N, 4]

    if "boxes_xyxy" in batch[0]:
        out["boxes_xyxy"] = torch.stack([x["boxes_xyxy"] for x in batch], dim=0)  # [B, N, 4]

    if "boxes_cxcywh" in batch[0]:
        out["boxes_cxcywh"] = torch.stack([x["boxes_cxcywh"] for x in batch], dim=0)  # [B, N, 4]

    if "box_valid_mask" in batch[0]:
        out["box_valid_mask"] = torch.stack([x["box_valid_mask"] for x in batch], dim=0)  # [B, N]

    if "image_index" in batch[0]:
        out["image_index"] = [x["image_index"] for x in batch]

    if "split" in batch[0]:
        out["split"] = [x["split"] for x in batch]

    if "kept_old_indices" in batch[0]:
        out["kept_old_indices"] = torch.stack([x["kept_old_indices"] for x in batch], dim=0)

    return out