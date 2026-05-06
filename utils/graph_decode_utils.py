# utils/graph_decode_utils.py

from typing import List, Optional, Tuple
import torch


def safe_obj_name(obj_id: int, obj_classes: Optional[List[str]], mask_obj_token_id: Optional[int] = None) -> str:
    if mask_obj_token_id is not None and obj_id == mask_obj_token_id:
        return "[MASK_OBJ]"
    if obj_classes is not None and 0 <= obj_id < len(obj_classes):
        return obj_classes[obj_id]
    return f"[OBJ_{obj_id}]"


def safe_rel_name(rel_id: int, rel_classes_pos: Optional[List[str]], mask_rel_token_id: Optional[int] = None) -> str:
    if mask_rel_token_id is not None and rel_id == mask_rel_token_id:
        return "[MASK_REL]"
    if rel_classes_pos is not None and 0 <= rel_id < len(rel_classes_pos):
        return rel_classes_pos[rel_id]
    return f"[REL_{rel_id}]"


def decode_obj_list(
    obj_ids: torch.Tensor,
    node_mask: torch.Tensor,
    obj_classes: Optional[List[str]] = None,
    mask_obj_token_id: Optional[int] = None,
) -> List[Tuple[int, str]]:
    names = []
    valid_idx = torch.where(node_mask.bool())[0]
    for i in valid_idx.tolist():
        oid = int(obj_ids[i].item())
        names.append((i, safe_obj_name(oid, obj_classes, mask_obj_token_id)))
    return names


def decode_triplets_from_structured_state(
    obj_ids: torch.Tensor,
    edge_t: torch.Tensor,
    rel_pos_t: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    obj_classes: Optional[List[str]] = None,
    rel_classes_pos: Optional[List[str]] = None,
    mask_obj_token_id: Optional[int] = None,
    mask_rel_token_id: Optional[int] = None,
):
    triplets = []
    valid_nodes = node_mask.bool()
    valid_pairs = edge_mask.bool()

    valid_node_ids = torch.where(valid_nodes)[0].tolist()
    for i in valid_node_ids:
        for j in valid_node_ids:
            if not valid_pairs[i, j]:
                continue
            if int(edge_t[i, j].item()) != 1:
                continue

            s_id = int(obj_ids[i].item())
            o_id = int(obj_ids[j].item())
            r_id = int(rel_pos_t[i, j].item())

            s_name = safe_obj_name(s_id, obj_classes, mask_obj_token_id)
            o_name = safe_obj_name(o_id, obj_classes, mask_obj_token_id)
            r_name = safe_rel_name(r_id, rel_classes_pos, mask_rel_token_id)

            triplets.append((i, s_name, r_name, j, o_name))

    return triplets


def format_nodes_block(
    title: str,
    obj_ids: torch.Tensor,
    node_mask: torch.Tensor,
    obj_classes: Optional[List[str]] = None,
    mask_obj_token_id: Optional[int] = None,
) -> str:
    lines = [title, "Nodes:"]
    decoded = decode_obj_list(
        obj_ids=obj_ids,
        node_mask=node_mask,
        obj_classes=obj_classes,
        mask_obj_token_id=mask_obj_token_id,
    )
    if len(decoded) == 0:
        lines.append("  (none)")
    else:
        for idx, name in decoded:
            lines.append(f"  [{idx:02d}] {name}")
    return "\n".join(lines)


def format_triplets_block(
    obj_ids: torch.Tensor,
    edge_t: torch.Tensor,
    rel_pos_t: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    obj_classes: Optional[List[str]] = None,
    rel_classes_pos: Optional[List[str]] = None,
    mask_obj_token_id: Optional[int] = None,
    mask_rel_token_id: Optional[int] = None,
) -> str:
    lines = ["Triplets:"]
    triplets = decode_triplets_from_structured_state(
        obj_ids=obj_ids,
        edge_t=edge_t,
        rel_pos_t=rel_pos_t,
        node_mask=node_mask,
        edge_mask=edge_mask,
        obj_classes=obj_classes,
        rel_classes_pos=rel_classes_pos,
        mask_obj_token_id=mask_obj_token_id,
        mask_rel_token_id=mask_rel_token_id,
    )
    if len(triplets) == 0:
        lines.append("  (none)")
    else:
        for s_idx, s_name, r_name, o_idx, o_name in triplets:
            lines.append(f"  [{s_idx:02d}] {s_name} --{r_name}--> [{o_idx:02d}] {o_name}")
    return "\n".join(lines)