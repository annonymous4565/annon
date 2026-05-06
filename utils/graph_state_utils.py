import torch

@torch.no_grad()
def build_full_relation_from_structured_state(
    edge_t: torch.Tensor,       # [B,N,N] or [N,N]
    rel_pos_t: torch.Tensor,    # [B,N,N] or [N,N]
    no_rel_token_id: int,
    num_rel_pos_classes: int,
) -> torch.Tensor:
    """
    Convert structured state (edge_t, rel_pos_t) into full relation labels
    expected by decode_item(...).

    Convention:
    - no edge => no_rel_token_id
    - positive edge with valid positive relation => rel_pos + 1
    - positive edge with masked relation token => no_rel_token_id
        (since decode_item has no explicit [MASK_REL] handling)
    """
    rel_full = torch.full_like(rel_pos_t, fill_value=no_rel_token_id)

    pos_mask = edge_t.bool()

    # valid positive relation ids are 0 .. num_rel_pos_classes-1
    valid_rel_mask = pos_mask & (rel_pos_t >= 0) & (rel_pos_t < num_rel_pos_classes)

    rel_full[valid_rel_mask] = rel_pos_t[valid_rel_mask] + 1

    # masked / invalid rel ids stay mapped to no_rel_token_id
    return rel_full