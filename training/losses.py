import torch
import torch.nn.functional as F



# -----------------------------
# Plausible Prior - soft labels
# -----------------------------

def gather_soft_plausibility_prior_biases(
    obj_probs: torch.Tensor,                      # [B, N, Kobj]
    edge_pair_logit_prior: torch.Tensor = None,  # [Kobj, Kobj]
    rel_pair_logprob_prior: torch.Tensor = None, # [Kobj, Kobj, Krel_pos]
):
    """
    Compute expected plausibility prior biases under soft object distributions.

    Returns:
        edge_prior_bias: [B, N, N] or None
        rel_prior_bias:  [B, N, N, Krel_pos] or None
    """
    edge_prior_bias = None
    rel_prior_bias = None

    # edge: sum_cd p_i(c) p_j(d) edge_prior[c,d]
    if edge_pair_logit_prior is not None:
        edge_prior_bias = torch.einsum(
            "bic,bjd,cd->bij",
            obj_probs,
            obj_probs,
            edge_pair_logit_prior,
        )

    # rel: sum_cd p_i(c) p_j(d) rel_prior[c,d,r]
    if rel_pair_logprob_prior is not None:
        rel_prior_bias = torch.einsum(
            "bic,bjd,cdr->bijr",
            obj_probs,
            obj_probs,
            rel_pair_logprob_prior,
        )

    return edge_prior_bias, rel_prior_bias

# -----------------------------
# Relation class weighting
# -----------------------------

@torch.no_grad()
def _count_relation_classes(dataset) -> torch.Tensor:
    num_rel_classes = len(dataset.relation_vocab)
    counts = torch.zeros(num_rel_classes, dtype=torch.float32)

    for i in range(len(dataset)):
        item = dataset[i]
        rel = item["rel_labels"]       # [N, N]
        edge_mask = item["edge_mask"]  # [N, N]

        vals = rel[edge_mask]
        counts.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))

    counts = torch.clamp(counts, min=1.0)
    return counts

@torch.no_grad()
def remap_full_rel_to_pos_labels(
    rel_targets_full: torch.Tensor,
    no_rel_token_id: int,
) -> torch.Tensor:
    """
    Map full relation labels (with NO_REL) to positive-only labels [0, K_pos-1].

    Assumes input contains no NO_REL entries.
    """
    pos_targets = rel_targets_full.clone()

    # shift labels above NO_REL down by 1
    pos_targets = torch.where(
        pos_targets > no_rel_token_id,
        pos_targets - 1,
        pos_targets,
    )

    # if NO_REL somehow slipped in, this is an error
    if (pos_targets == no_rel_token_id).any() and (rel_targets_full == no_rel_token_id).any():
        raise ValueError("NO_REL found in remap_full_rel_to_pos_labels input.")

    return pos_targets.long()


@torch.no_grad()
def remap_pos_labels_to_full(
    rel_pos_labels: torch.Tensor,
    no_rel_token_id: int,
) -> torch.Tensor:
    """
    Map positive-only labels [0, K_pos-1] back to full relation ids.
    """
    rel_full = rel_pos_labels.clone()
    rel_full = torch.where(
        rel_full >= no_rel_token_id,
        rel_full + 1,
        rel_full,
    )
    return rel_full.long()

@torch.no_grad()
def build_simple_relation_class_weights(
    num_rel_classes: int,
    no_rel_token_id: int,
    no_rel_weight: float,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(num_rel_classes, dtype=torch.float32, device=device)
    weights[no_rel_token_id] = no_rel_weight
    return weights


@torch.no_grad()
def build_relation_class_weights_inverse_freq(
    dataset,
    no_rel_token_id: int,
    no_rel_weight: float = 0.1,
    alpha: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    counts = _count_relation_classes(dataset)

    weights = 1.0 / counts.pow(alpha)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=min_weight, max=max_weight)

    weights[no_rel_token_id] = no_rel_weight

    if device is not None:
        weights = weights.to(device)

    return weights


@torch.no_grad()
def build_relation_class_weights_effective_num(
    dataset,
    no_rel_token_id: int,
    no_rel_weight: float = 0.1,
    beta: float = 0.999,
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    counts = _count_relation_classes(dataset)

    beta_t = torch.tensor(beta, dtype=torch.float32)
    effective_num = (1.0 - beta_t.pow(counts)) / (1.0 - beta_t)
    weights = 1.0 / effective_num

    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=min_weight, max=max_weight)

    weights[no_rel_token_id] = no_rel_weight

    if device is not None:
        weights = weights.to(device)

    return weights


@torch.no_grad()
def build_relation_class_weights(
    dataset,
    strategy: str,
    no_rel_token_id: int,
    no_rel_weight: float,
    device: torch.device,
    alpha: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    effective_num_beta: float = 0.999,
):
    strategy = strategy.lower()

    if strategy == "none":
        return None

    if strategy == "simple":
        return build_simple_relation_class_weights(
            num_rel_classes=len(dataset.relation_vocab),
            no_rel_token_id=no_rel_token_id,
            no_rel_weight=no_rel_weight,
            device=device,
        )

    if strategy == "inverse_freq":
        return build_relation_class_weights_inverse_freq(
            dataset=dataset,
            no_rel_token_id=no_rel_token_id,
            no_rel_weight=no_rel_weight,
            alpha=alpha,
            min_weight=min_weight,
            max_weight=max_weight,
            device=device,
        )

    if strategy == "effective_num":
        return build_relation_class_weights_effective_num(
            dataset=dataset,
            no_rel_token_id=no_rel_token_id,
            no_rel_weight=no_rel_weight,
            beta=effective_num_beta,
            min_weight=min_weight,
            max_weight=max_weight,
            device=device,
        )

    raise ValueError(f"Unknown relation weighting strategy: {strategy}")


# -----------------------------
# Negative-edge subsampling
# -----------------------------

@torch.no_grad()
def build_relation_loss_mask(
    rel_targets: torch.Tensor,
    edge_mask: torch.Tensor,
    no_rel_token_id: int,
    use_negative_edge_sampling: bool = False,
    neg_edge_sample_strategy: str = "ratio",
    neg_pos_ratio: float = 3.0,
    neg_keep_prob: float = 0.1,
) -> torch.Tensor:
    """
    Build mask used for relation training loss.

    Positive edges are always kept.
    Negative edges (NO_REL) are optionally subsampled.
    """
    positive_mask = edge_mask & (rel_targets != no_rel_token_id)
    negative_mask = edge_mask & (rel_targets == no_rel_token_id)

    if not use_negative_edge_sampling:
        return edge_mask

    B, N, _ = rel_targets.shape
    rel_loss_mask = positive_mask.clone()

    for b in range(B):
        pos_idx = positive_mask[b].nonzero(as_tuple=False)
        neg_idx = negative_mask[b].nonzero(as_tuple=False)

        num_pos = pos_idx.shape[0]
        num_neg = neg_idx.shape[0]

        if num_neg == 0:
            continue

        if neg_edge_sample_strategy == "prob":
            keep = torch.rand(num_neg, device=rel_targets.device) < neg_keep_prob
            kept_neg_idx = neg_idx[keep]

        elif neg_edge_sample_strategy == "ratio":
            max_neg = int(round(neg_pos_ratio * num_pos)) if num_pos > 0 else 0
            max_neg = min(max_neg, num_neg)

            if max_neg > 0:
                perm = torch.randperm(num_neg, device=rel_targets.device)[:max_neg]
                kept_neg_idx = neg_idx[perm]
            else:
                kept_neg_idx = neg_idx[:0]

        else:
            raise ValueError(f"Unknown neg_edge_sample_strategy: {neg_edge_sample_strategy}")

        if kept_neg_idx.numel() > 0:
            rel_loss_mask[b, kept_neg_idx[:, 0], kept_neg_idx[:, 1]] = True

    return rel_loss_mask


# -----------------------------
# Base masked losses
# -----------------------------

def masked_object_loss_sum_and_count(
    obj_logits: torch.Tensor,
    obj_targets: torch.Tensor,
    node_mask: torch.Tensor,
):
    per_node = F.cross_entropy(
        obj_logits.transpose(1, 2),  # [B, K_obj, N]
        obj_targets,
        reduction="none",
    )  # [B, N]

    mask = node_mask.float()
    loss_sum = (per_node * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return loss_sum, count


def masked_relation_loss_sum_and_count(
    rel_logits: torch.Tensor,
    rel_targets: torch.Tensor,
    rel_loss_mask: torch.Tensor,
    class_weights: torch.Tensor = None,
):
    B, N, _, K = rel_logits.shape

    logits_flat = rel_logits.reshape(B * N * N, K)
    targets_flat = rel_targets.reshape(B * N * N)
    mask_flat = rel_loss_mask.reshape(B * N * N).float()

    per_edge = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
        weight=class_weights,
    )

    loss_sum = (per_edge * mask_flat).sum()
    count = mask_flat.sum().clamp(min=1.0)
    return loss_sum, count


# -----------------------------
# Diagnostic masks
# -----------------------------

@torch.no_grad()
def build_positive_relation_mask(
    rel_targets: torch.Tensor,
    edge_mask: torch.Tensor,
    no_rel_token_id: int,
) -> torch.Tensor:
    return edge_mask & (rel_targets != no_rel_token_id)


@torch.no_grad()
def build_negative_relation_mask(
    rel_targets: torch.Tensor,
    edge_mask: torch.Tensor,
    no_rel_token_id: int,
) -> torch.Tensor:
    return edge_mask & (rel_targets == no_rel_token_id)



# -----------------------------
# Main loss - Plausible - soft label
# -----------------------------

def compute_discrete_sg_factorized_loss(
    model_out: dict,
    batch_t: dict,
    no_rel_token_id: int,
    lambda_obj: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_rel: float = 1.0,
    edge_exist_thres: float = 0.5,
    edge_pos_weight: float = 1.0,
    edge_pair_logit_prior: torch.Tensor = None,
    rel_pair_logprob_prior: torch.Tensor = None,
    obj_probs_for_prior: torch.Tensor = None,
    edge_prior_strength: float = 0.0,
    rel_prior_strength: float = 0.0,
):

    """
    Factorized SG loss:
      - object loss
      - edge existence loss (binary)
      - positive relation classification loss (only on GT-positive edges)

    Returns:
      training loss terms +
      diagnostic relation losses compatible with earlier logging
    """

    # -------------------------
    # Object loss
    # -------------------------
    obj_loss_sum, obj_count = masked_object_loss_sum_and_count(
        obj_logits=model_out["obj_logits"],
        obj_targets=batch_t["obj_0"],
        node_mask=batch_t["node_mask"],
    )
    obj_loss = obj_loss_sum / obj_count

    # -------------------------
    # GT masks
    # -------------------------
    rel_targets = batch_t["rel_0"]                  # [B, N, N]
    edge_mask = batch_t["edge_mask"].bool()         # [B, N, N]

    gt_pos_mask = edge_mask & (rel_targets != no_rel_token_id)
    gt_neg_mask = edge_mask & (rel_targets == no_rel_token_id)

    gt_edge_exists = gt_pos_mask.float()            # [B, N, N]

    # -------------------------
    # Edge existence loss
    # -------------------------
    edge_logits = model_out["edge_logits"]          # [B, N, N]
    rel_logits_pos = model_out["rel_logits_pos"]    # [B, N, N, K_rel_pos]

    if (
        obj_probs_for_prior is not None
        and (edge_pair_logit_prior is not None or rel_pair_logprob_prior is not None)
    ):
        edge_prior_bias, rel_prior_bias = gather_soft_plausibility_prior_biases(
            obj_probs=obj_probs_for_prior,
            edge_pair_logit_prior=edge_pair_logit_prior,
            rel_pair_logprob_prior=rel_pair_logprob_prior,
        )

        if edge_prior_bias is not None and edge_prior_strength != 0.0:
            edge_logits = edge_logits + edge_prior_strength * edge_prior_bias

        if rel_prior_bias is not None and rel_prior_strength != 0.0:
            rel_logits_pos = rel_logits_pos + rel_prior_strength * rel_prior_bias

    pos_weight = torch.tensor(
        edge_pos_weight,
        device=edge_logits.device,
        dtype=edge_logits.dtype,
    )

    per_edge_bce = F.binary_cross_entropy_with_logits(
        edge_logits,
        gt_edge_exists,
        reduction="none",
        pos_weight=pos_weight,
    )

    edge_loss_sum = (per_edge_bce * edge_mask.float()).sum()
    edge_count = edge_mask.float().sum().clamp(min=1.0)
    edge_loss = edge_loss_sum / edge_count

    # -------------------------
    # Positive relation type loss
    # -------------------------

    if gt_pos_mask.any():
        pos_logits = rel_logits_pos[gt_pos_mask]    # [num_pos, K_rel_pos]
        pos_targets_full = rel_targets[gt_pos_mask] # [num_pos]
        pos_targets = remap_full_rel_to_pos_labels(
            pos_targets_full,
            no_rel_token_id=no_rel_token_id,
        )

        rel_pos_cls_sum = F.cross_entropy(
            pos_logits,
            pos_targets,
            reduction="sum",
        )
        rel_pos_count = torch.tensor(
            float(pos_targets.numel()),
            device=rel_targets.device,
        ).clamp(min=1.0)
        rel_pos_cls_loss = rel_pos_cls_sum / rel_pos_count
    else:
        rel_pos_cls_sum = edge_logits.new_tensor(0.0)
        rel_pos_count = edge_logits.new_tensor(1.0)
        rel_pos_cls_loss = edge_logits.new_tensor(0.0)

    # -------------------------
    # Total training loss
    # -------------------------
    total_loss = (
        lambda_obj * obj_loss
        + lambda_edge * edge_loss
        + lambda_rel * rel_pos_cls_loss
    )

    # -------------------------
    # Diagnostics: reconstruct full relation prediction
    # -------------------------
    pred_edge_exists = (torch.sigmoid(edge_logits) >= edge_exist_thres) & edge_mask

    pred_pos_labels = rel_logits_pos.argmax(dim=-1)   # [B, N, N], in [0, K_rel_pos-1]
    pred_rel_full_if_pos = remap_pos_labels_to_full(
        pred_pos_labels,
        no_rel_token_id=no_rel_token_id,
    )

    pred_rel_full = torch.full_like(rel_targets, fill_value=no_rel_token_id)
    pred_rel_full[pred_edge_exists] = pred_rel_full_if_pos[pred_edge_exists]

    # full unweighted relation CE diagnostics against rel_0
    logits_full = torch.zeros(
        *rel_logits_pos.shape[:-1], rel_logits_pos.shape[-1] + 1,
        device=rel_logits_pos.device,
        dtype=rel_logits_pos.dtype,
    )  # [B,N,N,K_rel]

    # place positive logits into the correct full-class slots
    if no_rel_token_id == 0:
        logits_full[..., 1:] = rel_logits_pos
        logits_full[..., 0] = edge_logits
    else:
        # general case
        logits_full[..., :no_rel_token_id] = rel_logits_pos[..., :no_rel_token_id]
        logits_full[..., no_rel_token_id] = edge_logits
        logits_full[..., no_rel_token_id + 1:] = rel_logits_pos[..., no_rel_token_id:]

    rel_unweighted_sum, rel_unweighted_count = masked_relation_loss_sum_and_count(
        rel_logits=logits_full,
        rel_targets=rel_targets,
        rel_loss_mask=edge_mask,
        class_weights=None,
    )
    rel_loss_unweighted = rel_unweighted_sum / rel_unweighted_count

    rel_pos_sum, rel_pos_count_diag = masked_relation_loss_sum_and_count(
        rel_logits=logits_full,
        rel_targets=rel_targets,
        rel_loss_mask=gt_pos_mask,
        class_weights=None,
    )
    rel_loss_positive_only = rel_pos_sum / rel_pos_count_diag

    rel_neg_sum, rel_neg_count = masked_relation_loss_sum_and_count(
        rel_logits=logits_full,
        rel_targets=rel_targets,
        rel_loss_mask=gt_neg_mask,
        class_weights=None,
    )
    rel_loss_negative_only = rel_neg_sum / rel_neg_count

    return {
        "loss": total_loss,

        "obj_loss": obj_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "rel_pos_cls_loss": rel_pos_cls_loss.detach(),

        "obj_loss_sum": obj_loss_sum.detach(),
        "obj_count": obj_count.detach(),

        "edge_loss_sum": edge_loss_sum.detach(),
        "edge_count": edge_count.detach(),

        "rel_pos_cls_sum": rel_pos_cls_sum.detach(),
        "rel_pos_count": rel_pos_count.detach(),

        # compatibility diagnostics
        "rel_loss": rel_pos_cls_loss.detach(),  # use this slot if trainer still expects "rel_loss"
        "rel_loss_sum": rel_pos_cls_sum.detach(),
        "rel_count": rel_pos_count.detach(),

        "rel_loss_unweighted": rel_loss_unweighted.detach(),
        "rel_loss_positive_only": rel_loss_positive_only.detach(),
        "rel_loss_negative_only": rel_loss_negative_only.detach(),

        "rel_unweighted_sum": rel_unweighted_sum.detach(),
        "rel_unweighted_count": rel_unweighted_count.detach(),

        "rel_pos_sum": rel_pos_sum.detach(),
        "rel_pos_count": rel_pos_count_diag.detach(),

        "rel_neg_sum": rel_neg_sum.detach(),
        "rel_neg_count": rel_neg_count.detach(),

        "pred_rel_full": pred_rel_full.detach(),
    }

def compute_discrete_sg_loss(
    model_out: dict,
    batch_t: dict,
    no_rel_token_id: int,
    rel_class_weights: torch.Tensor = None,
    lambda_obj: float = 1.0,
    lambda_rel: float = 1.0,
    use_negative_edge_sampling: bool = False,
    neg_edge_sample_strategy: str = "ratio",
    neg_pos_ratio: float = 3.0,
    neg_keep_prob: float = 0.1,
):
    """
    Returns:
      training loss terms +
      diagnostic relation losses
    """
    obj_loss_sum, obj_count = masked_object_loss_sum_and_count(
        obj_logits=model_out["obj_logits"],
        obj_targets=batch_t["obj_0"],
        node_mask=batch_t["node_mask"],
    )

    # Main training relation mask
    rel_loss_mask = build_relation_loss_mask(
        rel_targets=batch_t["rel_0"],
        edge_mask=batch_t["edge_mask"],
        no_rel_token_id=no_rel_token_id,
        use_negative_edge_sampling=use_negative_edge_sampling,
        neg_edge_sample_strategy=neg_edge_sample_strategy,
        neg_pos_ratio=neg_pos_ratio,
        neg_keep_prob=neg_keep_prob,
    )

    rel_loss_sum, rel_count = masked_relation_loss_sum_and_count(
        rel_logits=model_out["rel_logits"],
        rel_targets=batch_t["rel_0"],
        rel_loss_mask=rel_loss_mask,
        class_weights=rel_class_weights,
    )

    obj_loss = obj_loss_sum / obj_count
    rel_loss = rel_loss_sum / rel_count
    total_loss = lambda_obj * obj_loss + lambda_rel * rel_loss

    # -------------------------
    # Diagnostics
    # -------------------------

    # 1) unweighted relation loss on same mask as training
    rel_unweighted_sum, rel_unweighted_count = masked_relation_loss_sum_and_count(
        rel_logits=model_out["rel_logits"],
        rel_targets=batch_t["rel_0"],
        rel_loss_mask=rel_loss_mask,
        class_weights=None,
    )
    rel_loss_unweighted = rel_unweighted_sum / rel_unweighted_count

    # 2) positive-only relation loss
    pos_mask = build_positive_relation_mask(
        rel_targets=batch_t["rel_0"],
        edge_mask=batch_t["edge_mask"],
        no_rel_token_id=no_rel_token_id,
    )
    rel_pos_sum, rel_pos_count = masked_relation_loss_sum_and_count(
        rel_logits=model_out["rel_logits"],
        rel_targets=batch_t["rel_0"],
        rel_loss_mask=pos_mask,
        class_weights=None,
    )
    rel_loss_positive_only = rel_pos_sum / rel_pos_count

    # 3) negative-only relation loss
    neg_mask = build_negative_relation_mask(
        rel_targets=batch_t["rel_0"],
        edge_mask=batch_t["edge_mask"],
        no_rel_token_id=no_rel_token_id,
    )
    rel_neg_sum, rel_neg_count = masked_relation_loss_sum_and_count(
        rel_logits=model_out["rel_logits"],
        rel_targets=batch_t["rel_0"],
        rel_loss_mask=neg_mask,
        class_weights=None,
    )
    rel_loss_negative_only = rel_neg_sum / rel_neg_count

    return {
        "loss": total_loss,
        "obj_loss": obj_loss.detach(),
        "rel_loss": rel_loss.detach(),

        "obj_loss_sum": obj_loss_sum.detach(),
        "obj_count": obj_count.detach(),

        "rel_loss_sum": rel_loss_sum.detach(),
        "rel_count": rel_count.detach(),

        # diagnostics
        "rel_loss_unweighted": rel_loss_unweighted.detach(),
        "rel_loss_positive_only": rel_loss_positive_only.detach(),
        "rel_loss_negative_only": rel_loss_negative_only.detach(),

        "rel_unweighted_sum": rel_unweighted_sum.detach(),
        "rel_unweighted_count": rel_unweighted_count.detach(),

        "rel_pos_sum": rel_pos_sum.detach(),
        "rel_pos_count": rel_pos_count.detach(),

        "rel_neg_sum": rel_neg_sum.detach(),
        "rel_neg_count": rel_neg_count.detach(),
    }

