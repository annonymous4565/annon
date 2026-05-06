# sampling/reward_terms.py

from typing import Dict, List, Optional, Sequence, Tuple
import torch


# ---------------------------------------------------------
# Box helpers
# ---------------------------------------------------------

@torch.no_grad()
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [...,4] in normalized cx,cy,w,h
    returns: [...,4] in normalized x1,y1,x2,y2
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def pairwise_iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    boxes_a: [M,4]
    boxes_b: [K,4]
    returns: [M,K]
    """
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_a = ((ax2 - ax1).clamp(min=0.0) * (ay2 - ay1).clamp(min=0.0))
    area_b = ((bx2 - bx1).clamp(min=0.0) * (by2 - by1).clamp(min=0.0))

    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-12)


@torch.no_grad()
def build_relation_group_ids(relation_vocab: Sequence[str]) -> Dict[str, List[int]]:
    """
    Conservative relation grouping for geometry rewards.
    """
    relation_vocab = [str(x).strip().lower() for x in relation_vocab]

    def ids_for_exact(names: Sequence[str]) -> List[int]:
        out = []
        names = set([str(x).strip().lower() for x in names])
        for k, name in enumerate(relation_vocab):
            if name in names:
                out.append(k)
        return out

    def ids_for_substrings(names: Sequence[str]) -> List[int]:
        out = []
        for k, name in enumerate(relation_vocab):
            if any(alias in name for alias in names):
                out.append(k)
        return sorted(list(set(out)))

    left_ids = ids_for_substrings(["left of", "to the left of", "on the left of", "left"])
    right_ids = ids_for_substrings(["right of", "to the right of", "on the right of", "right"])
    above_ids = ids_for_exact(["above", "over", "on top of"])
    below_ids = ids_for_exact(["below", "under", "beneath"])
    inside_ids = ids_for_exact(["in", "inside", "inside of"])

    return {
        "left": left_ids,
        "right": right_ids,
        "above": above_ids,
        "below": below_ids,
        "inside": inside_ids,
    }


# ---------------------------------------------------------
# Graph-only reward terms
# ---------------------------------------------------------

@torch.no_grad()
def reward_isolated_node_penalty(
    edge_t: torch.Tensor,      # [B,N,N]
    node_mask: torch.Tensor,   # [B,N]
) -> torch.Tensor:
    """
    Penalize isolated valid nodes.
    Returns reward in [-1, 0], where higher is better.
    """
    valid_nodes = node_mask.bool()
    out_deg = edge_t.sum(dim=-1)
    in_deg = edge_t.sum(dim=-2)
    total_deg = out_deg + in_deg

    isolated = valid_nodes & (total_deg == 0)
    num_valid = valid_nodes.sum(dim=-1).clamp(min=1)
    frac_isolated = isolated.sum(dim=-1).float() / num_valid.float()

    return -frac_isolated


@torch.no_grad()
def reward_bidirectional_edge_penalty(
    edge_t: torch.Tensor,      # [B,N,N]
    edge_mask: torch.Tensor,   # [B,N,N]
) -> torch.Tensor:
    """
    Penalize mutual directed edges i->j and j->i.
    Returns reward in [-1, 0], higher is better.
    """
    keep = edge_t.bool() & edge_mask.bool()
    mutual = keep & keep.transpose(1, 2)

    # each mutual pair counted twice; remove diagonal and divide by 2
    B, N, _ = edge_t.shape
    diag = torch.eye(N, device=edge_t.device, dtype=torch.bool).unsqueeze(0)
    mutual = mutual & (~diag)

    mutual_count = mutual.sum(dim=(1, 2)).float() / 2.0
    active_edge_count = keep.sum(dim=(1, 2)).float().clamp(min=1.0)

    frac_mutual = mutual_count / active_edge_count
    return -frac_mutual


@torch.no_grad()
def reward_dense_graph_penalty(
    edge_t: torch.Tensor,      # [B,N,N]
    edge_mask: torch.Tensor,   # [B,N,N]
) -> torch.Tensor:
    """
    Penalize overly dense graphs intrinsically.
    Returns reward in [-1, 0], higher is better.
    """
    keep = edge_t.bool() & edge_mask.bool()
    edge_frac = keep.sum(dim=(1, 2)).float() / edge_mask.sum(dim=(1, 2)).clamp(min=1).float()
    return -edge_frac


# ---------------------------------------------------------
# Layout reward terms
# ---------------------------------------------------------

@torch.no_grad()
def reward_box_bounds(
    layout_box_pred: torch.Tensor,   # [B,N,4] cxcywh normalized
    box_valid_mask: torch.Tensor,    # [B,N]
) -> torch.Tensor:
    """
    Reward high when boxes stay inside [0,1] and widths/heights are positive.
    Range roughly <= 0, higher is better.
    """
    valid = box_valid_mask.bool()
    if valid.sum() == 0:
        return torch.zeros(layout_box_pred.shape[0], device=layout_box_pred.device, dtype=layout_box_pred.dtype)

    xyxy = cxcywh_to_xyxy(layout_box_pred)
    x1, y1, x2, y2 = xyxy.unbind(dim=-1)
    _, _, w, h = layout_box_pred.unbind(dim=-1)

    penalty = (
        (-x1).clamp(min=0.0)
        + (-y1).clamp(min=0.0)
        + (x2 - 1.0).clamp(min=0.0)
        + (y2 - 1.0).clamp(min=0.0)
        + (-w).clamp(min=0.0)
        + (-h).clamp(min=0.0)
    )

    penalty = penalty * valid.float()
    denom = valid.sum(dim=-1).clamp(min=1).float()
    return -(penalty.sum(dim=-1) / denom)


@torch.no_grad()
def reward_layout_overlap(
    layout_box_pred: torch.Tensor,   # [B,N,4]
    box_valid_mask: torch.Tensor,    # [B,N]
) -> torch.Tensor:
    """
    Penalize pairwise overlap among valid boxes.
    Returns reward <= 0, higher is better.
    """
    B, N, _ = layout_box_pred.shape
    xyxy = cxcywh_to_xyxy(layout_box_pred)

    rewards = []
    for b in range(B):
        valid = box_valid_mask[b].bool()
        idx = torch.where(valid)[0]
        if idx.numel() <= 1:
            rewards.append(torch.zeros((), device=layout_box_pred.device, dtype=layout_box_pred.dtype))
            continue

        boxes = xyxy[b, idx]  # [M,4]
        iou = pairwise_iou_xyxy(boxes, boxes)

        eye = torch.eye(iou.shape[0], device=iou.device, dtype=torch.bool)
        iou = iou.masked_fill(eye, 0.0)

        # average over off-diagonal terms
        denom = max(iou.numel() - iou.shape[0], 1)
        rewards.append(-(iou.sum() / float(denom)))

    return torch.stack(rewards, dim=0)


@torch.no_grad()
def reward_layout_spread(
    layout_box_pred: torch.Tensor,   # [B,N,4]
    box_valid_mask: torch.Tensor,    # [B,N]
) -> torch.Tensor:
    """
    Reward spatial spread of box centers.
    Higher is better.
    """
    centers = layout_box_pred[..., :2]   # [B,N,2]
    B = centers.shape[0]

    rewards = []
    for b in range(B):
        valid = box_valid_mask[b].bool()
        idx = torch.where(valid)[0]
        if idx.numel() <= 1:
            rewards.append(torch.zeros((), device=layout_box_pred.device, dtype=layout_box_pred.dtype))
            continue

        pts = centers[b, idx]  # [M,2]
        dmat = torch.cdist(pts, pts, p=2)

        eye = torch.eye(dmat.shape[0], device=dmat.device, dtype=torch.bool)
        dmat = dmat.masked_fill(eye, 0.0)

        denom = max(dmat.numel() - dmat.shape[0], 1)
        rewards.append(dmat.sum() / float(denom))

    return torch.stack(rewards, dim=0)


# ---------------------------------------------------------
# Relation-geometry reward
# ---------------------------------------------------------

@torch.no_grad()
def reward_relation_geometry(
    rel_full_t: torch.Tensor,        # [B,N,N] full relation ids including no-rel
    layout_box_pred: torch.Tensor,   # [B,N,4] cxcywh normalized
    box_valid_mask: torch.Tensor,    # [B,N]
    relation_group_ids: Dict[str, List[int]],
) -> torch.Tensor:
    """
    Soft relation-geometry consistency reward.
    Higher is better.
    """
    centers = layout_box_pred[..., :2]
    xyxy = cxcywh_to_xyxy(layout_box_pred)
    B, N, _ = rel_full_t.shape

    device = rel_full_t.device
    dtype = layout_box_pred.dtype

    out = torch.zeros((B,), device=device, dtype=dtype)

    for b in range(B):
        valid_node = box_valid_mask[b].bool()

        term_sum = torch.zeros((), device=device, dtype=dtype)
        term_count = 0.0

        for i in range(N):
            if not valid_node[i]:
                continue
            for j in range(N):
                if i == j or (not valid_node[j]):
                    continue

                r = int(rel_full_t[b, i, j].item())

                ci = centers[b, i]
                cj = centers[b, j]
                bi = xyxy[b, i]
                bj = xyxy[b, j]

                if r in relation_group_ids["left"]:
                    term_sum = term_sum + (cj[0] - ci[0])   # subj should be left of obj => obj.cx - subj.cx positive
                    term_count += 1.0
                elif r in relation_group_ids["right"]:
                    term_sum = term_sum + (ci[0] - cj[0])
                    term_count += 1.0
                elif r in relation_group_ids["above"]:
                    term_sum = term_sum + (cj[1] - ci[1])   # obj below subj => obj.cy - subj.cy positive
                    term_count += 1.0
                elif r in relation_group_ids["below"]:
                    term_sum = term_sum + (ci[1] - cj[1])
                    term_count += 1.0
                elif r in relation_group_ids["inside"]:
                    # reward if subj box lies inside obj box
                    inside_x = torch.minimum(
                        (bi[0] - bj[0]),
                        (bj[2] - bi[2]),
                    )
                    inside_y = torch.minimum(
                        (bi[1] - bj[1]),
                        (bj[3] - bi[3]),
                    )
                    term_sum = term_sum + torch.minimum(inside_x, inside_y)
                    term_count += 1.0

        if term_count > 0:
            out[b] = term_sum / term_count
        else:
            out[b] = torch.zeros((), device=device, dtype=dtype)

    return out


# ---------------------------------------------------------
# Composite reward
# ---------------------------------------------------------

@torch.no_grad()
def compute_sg_layout_reward_terms(
    obj_t: torch.Tensor,                    # [B,N]
    edge_t: torch.Tensor,                   # [B,N,N]
    rel_full_t: torch.Tensor,               # [B,N,N]
    node_mask: torch.Tensor,                # [B,N]
    edge_mask: torch.Tensor,                # [B,N,N]
    layout_box_pred: Optional[torch.Tensor] = None,   # [B,N,4]
    box_valid_mask: Optional[torch.Tensor] = None,    # [B,N]
    relation_group_ids: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns per-example reward terms [B].
    """
    device = obj_t.device
    dtype = torch.float32

    reward_dict = {
        "reward_isolated_node": reward_isolated_node_penalty(edge_t=edge_t, node_mask=node_mask).to(dtype),
        "reward_bidirectional_edge": reward_bidirectional_edge_penalty(edge_t=edge_t, edge_mask=edge_mask).to(dtype),
        "reward_dense_graph": reward_dense_graph_penalty(edge_t=edge_t, edge_mask=edge_mask).to(dtype),
    }

    if layout_box_pred is not None and box_valid_mask is not None:
        reward_dict["reward_box_bounds"] = reward_box_bounds(
            layout_box_pred=layout_box_pred,
            box_valid_mask=box_valid_mask,
        ).to(dtype)

        reward_dict["reward_layout_overlap"] = reward_layout_overlap(
            layout_box_pred=layout_box_pred,
            box_valid_mask=box_valid_mask,
        ).to(dtype)

        reward_dict["reward_layout_spread"] = reward_layout_spread(
            layout_box_pred=layout_box_pred,
            box_valid_mask=box_valid_mask,
        ).to(dtype)

        if relation_group_ids is not None:
            reward_dict["reward_relation_geometry"] = reward_relation_geometry(
                rel_full_t=rel_full_t,
                layout_box_pred=layout_box_pred,
                box_valid_mask=box_valid_mask,
                relation_group_ids=relation_group_ids,
            ).to(dtype)
        else:
            reward_dict["reward_relation_geometry"] = torch.zeros(
                obj_t.shape[0], device=device, dtype=dtype
            )
    else:
        zeros = torch.zeros(obj_t.shape[0], device=device, dtype=dtype)
        reward_dict["reward_box_bounds"] = zeros.clone()
        reward_dict["reward_layout_overlap"] = zeros.clone()
        reward_dict["reward_layout_spread"] = zeros.clone()
        reward_dict["reward_relation_geometry"] = zeros.clone()

    return reward_dict


@torch.no_grad()
def combine_reward_terms(
    reward_terms: Dict[str, torch.Tensor],
    reward_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Weighted sum of per-example reward terms.
    """
    total = None
    for k, v in reward_terms.items():
        w = float(reward_weights.get(k, 0.0))
        contrib = w * v
        total = contrib if total is None else (total + contrib)

    if total is None:
        any_term = next(iter(reward_terms.values()))
        total = torch.zeros_like(any_term)

    out = dict(reward_terms)
    out["reward_total"] = total
    return out