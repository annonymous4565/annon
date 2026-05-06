# diffusion/sg_state_utils.py

from typing import Tuple
import torch


def build_structured_targets(
    rel_full: torch.Tensor,          # [B,N,N] or [N,N]
    edge_mask: torch.Tensor,         # [B,N,N] or [N,N], bool
    no_rel_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        edge_0:          binary existence target, same shape as rel_full
        rel_pos_0:       positive-only relation ids, zero-filled where no edge
        gt_pos_edge_mask: mask of true positive edges
    """
    edge_mask = edge_mask.bool()
    gt_pos_edge_mask = edge_mask & (rel_full != no_rel_token_id)
    edge_0 = gt_pos_edge_mask.long()

    rel_pos_0 = rel_full.clone()

    # map full labels -> positive-only labels
    rel_pos_0 = torch.where(
        rel_pos_0 > no_rel_token_id,
        rel_pos_0 - 1,
        rel_pos_0,
    )

    # fill non-edges with 0 placeholder; they are always masked out later
    rel_pos_0 = torch.where(gt_pos_edge_mask, rel_pos_0, torch.zeros_like(rel_pos_0))
    return edge_0, rel_pos_0.long(), gt_pos_edge_mask


def remap_full_rel_to_pos(rel_full: torch.Tensor, no_rel_token_id: int) -> torch.Tensor:
    rel_pos = torch.where(rel_full > no_rel_token_id, rel_full - 1, rel_full)
    return rel_pos.long()


def remap_pos_rel_to_full(rel_pos: torch.Tensor, no_rel_token_id: int) -> torch.Tensor:
    rel_full = torch.where(rel_pos >= no_rel_token_id, rel_pos + 1, rel_pos)
    return rel_full.long()


def reconstruct_full_relations(
    pred_edge_exists: torch.Tensor,  # bool [B,N,N]
    pred_rel_pos: torch.Tensor,      # long [B,N,N]
    no_rel_token_id: int,
) -> torch.Tensor:
    pred_rel_full_if_pos = remap_pos_rel_to_full(pred_rel_pos, no_rel_token_id)
    pred_rel_full = torch.full_like(pred_rel_full_if_pos, fill_value=no_rel_token_id)
    pred_rel_full[pred_edge_exists] = pred_rel_full_if_pos[pred_edge_exists]
    return pred_rel_full


def build_valid_pair_mask(node_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    """
    Combines node validity and provided edge_mask.
    """
    nm_i = node_mask.unsqueeze(-1)
    nm_j = node_mask.unsqueeze(-2)
    return edge_mask.bool() & nm_i & nm_j