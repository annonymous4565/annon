# sampling/node_gibbs.py

from typing import Optional, Tuple

import torch


@torch.no_grad()
def build_fixed_structure_predictions(
    edge_logits: torch.Tensor,      # [B,N,N]
    rel_logits_pos: torch.Tensor,   # [B,N,N,K_rel_pos]
    edge_mask: torch.Tensor,        # [B,N,N]
    no_rel_token_id: int,
    edge_exist_thres: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build fixed discrete structure predictions from model logits.

    Returns:
        edge_fixed: [B,N,N] in {0,1}
        rel_full_fixed: [B,N,N] in full relation vocab
    """
    edge_prob = torch.sigmoid(edge_logits)
    edge_fixed = (edge_prob >= edge_exist_thres).long()
    edge_fixed = edge_fixed * edge_mask.long()

    rel_pos_pred = rel_logits_pos.argmax(dim=-1)   # [B,N,N], positive-rel index 0..K_pos-1
    rel_full_fixed = rel_pos_pred + 1              # shift back to full vocab assuming 0=no_rel

    rel_full_fixed = torch.where(
        edge_fixed.bool(),
        rel_full_fixed,
        torch.full_like(rel_full_fixed, fill_value=no_rel_token_id),
    )

    rel_full_fixed = rel_full_fixed * edge_mask.long() + \
        (~edge_mask.bool()).long() * no_rel_token_id

    return edge_fixed, rel_full_fixed


@torch.no_grad()
def full_rel_to_rel_pos(
    rel_full: torch.Tensor,         # [B,N,N]
    no_rel_token_id: int,
    mask_rel_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert full relation labels to positive-only relation labels used by model input.

    Assumes no_rel_token_id = 0 and positive relations occupy 1..K.
    For non-edge/no-rel entries, fill with mask_rel_token_id if provided, else 0.
    """
    rel_pos = rel_full.clone()

    pos_mask = rel_full != no_rel_token_id
    rel_pos[pos_mask] = rel_full[pos_mask] - 1

    if mask_rel_token_id is not None:
        rel_pos[~pos_mask] = mask_rel_token_id
    else:
        rel_pos[~pos_mask] = 0

    return rel_pos.long()


@torch.no_grad()
def sample_node_labels_from_logits(
    node_logits: torch.Tensor,      # [B,K]
    temp: float = 1.0,
) -> torch.Tensor:
    logits = node_logits / max(temp, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled


@torch.no_grad()
def run_single_sweep_node_gibbs(
    model,
    obj_init: torch.Tensor,         # [B,N]
    edge_fixed: torch.Tensor,       # [B,N,N]
    rel_pos_fixed: torch.Tensor,    # [B,N,N]
    t: torch.Tensor,                # [B]
    node_mask: torch.Tensor,        # [B,N]
    edge_mask: torch.Tensor,        # [B,N,N]
    sample_temp: float = 1.0,
    random_order: bool = True,
) -> torch.Tensor:
    """
    One conditional node sweep:
      - initialize obj_cur = obj_init
      - for each valid node position i:
          rerun model on current obj_cur and fixed structure
          sample node i from obj_logits[:, i, :]
          write back into obj_cur[:, i]
    """
    B, N = obj_init.shape
    obj_cur = obj_init.clone()

    if random_order:
        order = torch.randperm(N, device=obj_init.device)
    else:
        order = torch.arange(N, device=obj_init.device)

    for i in order.tolist():
        valid_b = node_mask[:, i].bool()
        if valid_b.sum() == 0:
            continue

        model_out_i = model(
            obj_t=obj_cur,
            edge_t=edge_fixed,
            rel_pos_t=rel_pos_fixed,
            t=t,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        sampled_i = sample_node_labels_from_logits(
            model_out_i["obj_logits"][:, i, :],
            temp=sample_temp,
        )

        obj_cur[valid_b, i] = sampled_i[valid_b]

    return obj_cur


@torch.no_grad()
def run_node_gibbs_sampler(
    model,
    obj_t: torch.Tensor,            # [B,N]
    edge_t: torch.Tensor,           # [B,N,N]
    rel_pos_t: torch.Tensor,        # [B,N,N]
    t: torch.Tensor,                # [B]
    node_mask: torch.Tensor,        # [B,N]
    edge_mask: torch.Tensor,        # [B,N,N]
    no_rel_token_id: int,
    mask_rel_token_id: Optional[int],
    edge_exist_thres: float = 0.5,
    num_sweeps: int = 1,
    sample_temp: float = 1.0,
    use_fixed_structure: bool = True,
    random_order: bool = True,
):
    """
    Phase 3E.1:
      1. predict structure once from noisy state
      2. fix structure
      3. initialize obj_cur = obj_t
      4. run one or more node sweeps

    Returns:
        pred_obj_full: [B,N]
        edge_fixed: [B,N,N]
        rel_full_fixed: [B,N,N]
        rel_pos_fixed: [B,N,N]
    """
    model_out_struct = model(
        obj_t=obj_t,
        edge_t=edge_t,
        rel_pos_t=rel_pos_t,
        t=t,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

    edge_fixed, rel_full_fixed = build_fixed_structure_predictions(
        edge_logits=model_out_struct["edge_logits"],
        rel_logits_pos=model_out_struct["rel_logits_pos"],
        edge_mask=edge_mask,
        no_rel_token_id=no_rel_token_id,
        edge_exist_thres=edge_exist_thres,
    )

    rel_pos_fixed = full_rel_to_rel_pos(
        rel_full=rel_full_fixed,
        no_rel_token_id=no_rel_token_id,
        mask_rel_token_id=mask_rel_token_id,
    )

    obj_cur = obj_t.clone()

    for _ in range(num_sweeps):
        obj_cur = run_single_sweep_node_gibbs(
            model=model,
            obj_init=obj_cur,
            edge_fixed=edge_fixed if use_fixed_structure else edge_t,
            rel_pos_fixed=rel_pos_fixed if use_fixed_structure else rel_pos_t,
            t=t,
            node_mask=node_mask,
            edge_mask=edge_mask,
            sample_temp=sample_temp,
            random_order=random_order,
        )

    return obj_cur, edge_fixed, rel_full_fixed, rel_pos_fixed