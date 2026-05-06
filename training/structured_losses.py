# training/structured_losses.py

import torch
import torch.nn.functional as F

from diffusion.sg_state_utils import reconstruct_full_relations


def masked_object_loss_sum_and_count(
    obj_logits: torch.Tensor,          # [B,N,K]
    obj_targets: torch.Tensor,         # [B,N]
    obj_loss_mask: torch.Tensor,       # [B,N] bool
    class_weights: torch.Tensor = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 1.0,
):
    logits = obj_logits.transpose(1, 2)  # [B,K,N]

    per_node_ce = F.cross_entropy(
        logits,
        obj_targets,
        reduction="none",
        weight=class_weights,
    )  # [B,N]

    if use_focal_loss:
        pt = torch.exp(-per_node_ce).clamp(min=1e-8, max=1.0)
        focal_factor = focal_alpha * (1.0 - pt).pow(focal_gamma)
        per_node = focal_factor * per_node_ce
    else:
        per_node = per_node_ce

    mask = obj_loss_mask.float()
    loss_sum = (per_node * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return loss_sum, count


def masked_bce_loss_sum_and_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor = None,
):
    per = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
        pos_weight=pos_weight,
    )
    mf = mask.float()
    loss_sum = (per * mf).sum()
    count = mf.sum().clamp(min=1.0)
    return loss_sum, count


def masked_relation_ce_sum_and_count(
    rel_logits_pos: torch.Tensor,      # [B,N,N,K_rel_pos]
    rel_targets_pos: torch.Tensor,     # [B,N,N]
    rel_loss_mask: torch.Tensor,       # [B,N,N] bool
):
    logits = rel_logits_pos[rel_loss_mask]
    targets = rel_targets_pos[rel_loss_mask]

    if targets.numel() == 0:
        z = rel_logits_pos.new_tensor(0.0)
        o = rel_logits_pos.new_tensor(1.0)
        return z, o

    loss_sum = F.cross_entropy(logits, targets, reduction="sum")
    count = torch.tensor(float(targets.numel()), device=rel_logits_pos.device).clamp(min=1.0)
    return loss_sum, count


@torch.no_grad()
def compute_edge_metrics(
    edge_logits: torch.Tensor,
    edge_0: torch.Tensor,
    edge_mask: torch.Tensor,
    thres: float = 0.5,
):
    pred_edge = (torch.sigmoid(edge_logits) >= thres) & edge_mask.bool()
    gt_edge = edge_0.bool() & edge_mask.bool()

    tp = (pred_edge & gt_edge).sum().float()
    fp = (pred_edge & (~gt_edge)).sum().float()
    fn = ((~pred_edge) & gt_edge).sum().float()

    return {
        "tp_edges": tp,
        "fp_edges": fp,
        "fn_edges": fn,
        "pred_pos_edges": pred_edge.sum().float(),
        "gt_pos_edges": gt_edge.sum().float(),
        "pred_edge": pred_edge,
    }

   
def compute_structured_sg_loss(
    model_out: dict,
    batch_t: dict,
    no_rel_token_id: int,
    lambda_obj: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_rel: float = 1.0,
    lambda_layout: float = 0.0,
    lambda_rel_geometry: float = 0.0,
    relation_vocab: list = None,
    rel_geom_margin: float = 0.02,
    edge_exist_thres: float = 0.5,
    edge_pos_weight: float = 1.0,
    obj_class_weights: torch.Tensor = None,
    rel_class_weights: torch.Tensor = None,
    node_loss_mode: str = "corrupted",
    use_object_focal_loss: bool = False,
    object_focal_gamma: float = 2.0,
    object_focal_alpha: float = 1.0,
    pred_obj_override: torch.Tensor = None,
    pred_rel_full_override: torch.Tensor = None,
    use_layout_supervision: bool = False,
    layout_loss_type: str = "smooth_l1",
    use_layout_giou_loss: bool = False,
    lambda_layout_giou: float = 0.5,
    use_relation_geometry_loss: bool = False,
    use_layout_class_priors: bool = False,
    lambda_layout_class_prior: float = 0.0,
    layout_prior_mean: torch.Tensor = None,
    layout_prior_var: torch.Tensor = None,
    layout_prior_valid: torch.Tensor = None,
    layout_class_prior_eps: float = 1e-4,
    use_layout_regularization: bool = False,
    layout_overlap_reg_weight: float = 0.10,
    layout_spread_reg_weight: float = 0.05,
    layout_min_center_spread: float = 0.18,
    use_relation_geometry_reg: bool = False,
    lambda_relation_geometry_reg: float = 0.0,
    relation_geometry_margin: float = 0.03,
    use_graph_law_reg: bool = False,
    lambda_graph_law_reg: float = 0.0,
    graph_law_edge_weight: float = 1.0,
    graph_law_degree_weight: float = 0.5,
    graph_law_rel_weight: float = 0.5,
    graph_law_eps: float = 1e-6,
    object_only_sanity: bool = False,
    topk_obj_loss_only: bool = False,
    topk_obj_classes: int = 0
):
    """
    Phase 3E.1-compatible SG loss.

    Training losses are still computed from the one-shot model_out.
    Final node / relation predictions used for metrics can be overridden
    by Gibbs sampler outputs.

    rel_class_weights:
        Optional tensor of shape [K_rel_pos] for weighting positive relation
        classes in the structured relation loss.
    """
    obj_logits = model_out["obj_logits"]          # [B,N,K_obj]
    edge_logits = model_out["edge_logits"]        # [B,N,N]
    rel_logits_pos = model_out["rel_logits_pos"]  # [B,N,N,K_rel_pos]

    layout_box_pred = model_out.get("layout_box_pred", None)   # [B,N,4] or None

    obj_targets = batch_t["obj_0"]                # [B,N]
    edge_targets = batch_t["edge_0"].float()      # [B,N,N]
    rel_pos_targets = batch_t["rel_pos_0"]        # [B,N,N]

    node_mask = batch_t["node_mask"].bool()       # [B,N]
    edge_mask = batch_t["edge_mask"].bool()       # [B,N,N]

    obj_corrupt_mask = batch_t["obj_corrupt_mask"].bool()
    obj_mask_token_mask = batch_t["obj_mask_token_mask"].bool()

    gt_pos_edge_mask = batch_t["gt_pos_edge_mask"].bool()  # [B,N,N]

    target_boxes = batch_t.get("boxes", None)              # [B,N,4] or None
    box_valid_mask = batch_t.get("box_valid_mask", None)   # [B,N] or None

    device = obj_logits.device

    # -------------------------------------------------
    # Object loss mask
    # -------------------------------------------------
    if node_loss_mode == "all":
        obj_loss_mask = node_mask
    elif node_loss_mode == "corrupted":
        obj_loss_mask = node_mask & obj_corrupt_mask
    elif node_loss_mode == "masked_only":
        obj_loss_mask = node_mask & obj_mask_token_mask
    else:
        raise ValueError(f"Unknown node_loss_mode: {node_loss_mode}")

    # -------------------------------------------------
    # Top-K class filtering (NEW)
    # -------------------------------------------------
    if topk_obj_loss_only:
        topk = topk_obj_classes.to(obj_targets.device)
        topk_mask = torch.isin(obj_targets, topk)

        obj_loss_mask = obj_loss_mask & topk_mask

    # If nothing selected, fall back to all valid nodes to avoid NaNs
    if obj_loss_mask.sum() == 0:
        # fallback but KEEP topk constraint if enabled
        if topk_obj_loss_only:
            obj_loss_mask = node_mask & torch.isin(obj_targets, topk)
        else:
            obj_loss_mask = node_mask

    obj_logits_flat = obj_logits[obj_loss_mask]     # [M,K]
    obj_targets_flat = obj_targets[obj_loss_mask]   # [M]

    if use_object_focal_loss:
        ce = F.cross_entropy(
            obj_logits_flat,
            obj_targets_flat,
            reduction="none",
            weight=obj_class_weights,
        )
        pt = torch.exp(-ce)
        obj_loss = (object_focal_alpha * ((1 - pt) ** object_focal_gamma) * ce).mean()
    else:
        obj_loss = F.cross_entropy(
            obj_logits_flat,
            obj_targets_flat,
            reduction="mean",
            weight=obj_class_weights,
        )

    # -------------------------------------------------
    # Edge loss
    # -------------------------------------------------
    edge_logits_flat = edge_logits[edge_mask]
    edge_targets_flat = edge_targets[edge_mask]

    if edge_pos_weight != 1.0:
        pos_weight = torch.tensor(edge_pos_weight, dtype=edge_logits.dtype, device=device)
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits_flat,
            edge_targets_flat,
            reduction="mean",
            pos_weight=pos_weight,
        )
    else:
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits_flat,
            edge_targets_flat,
            reduction="mean",
        )

    # -------------------------------------------------
    # Relation loss (only on true positive GT edges)
    # -------------------------------------------------
    if gt_pos_edge_mask.sum() > 0:
        rel_logits_valid = rel_logits_pos[gt_pos_edge_mask]     # [M,K_rel_pos]
        rel_targets_valid = rel_pos_targets[gt_pos_edge_mask]   # [M]

        if rel_class_weights is not None:
            rel_class_weights = rel_class_weights.to(device=device, dtype=rel_logits_valid.dtype)

            per_example_rel_loss = F.cross_entropy(
                rel_logits_valid,
                rel_targets_valid,
                reduction="none",
            )  # [M]

            per_example_weights = rel_class_weights[rel_targets_valid]  # [M]

            rel_loss = (
                (per_example_rel_loss * per_example_weights).sum()
                / per_example_weights.sum().clamp(min=1e-12)
            )
        else:
            rel_loss = F.cross_entropy(
                rel_logits_valid,
                rel_targets_valid,
                reduction="mean",
            )
    else:
        rel_loss = torch.zeros((), device=device, dtype=obj_logits.dtype)
    
    # -------------------------------------------------
    # Layout loss
    # -------------------------------------------------
    if (
        use_layout_supervision
        and (layout_box_pred is not None)
        and (target_boxes is not None)
        and (box_valid_mask is not None)
    ):
        layout_dict = compute_layout_loss(
            model_out=model_out,
            batch_t=batch_t,
            layout_loss_type=layout_loss_type,
            use_layout_giou_loss=use_layout_giou_loss,
            lambda_layout_giou=lambda_layout_giou,
        )
        layout_loss = layout_dict["layout_loss"]
        layout_l1 = layout_dict["layout_l1"]
        layout_giou_loss = layout_dict["layout_giou_loss"]
        layout_mean_giou = layout_dict["layout_mean_giou"]
    else:
        # must stay connected to layout head when it exists, for DDP safety
        if layout_box_pred is not None:
            zero = layout_box_pred.sum() * 0.0
        else:
            zero = obj_logits.sum() * 0.0

        layout_loss = zero
        layout_l1 = zero.detach()
        layout_giou_loss = zero.detach()
        layout_mean_giou = zero.detach()
    
    # --------------------------------------------------
    # 8E.2 / 8E.3 layout regularization
    # --------------------------------------------------
    if use_layout_regularization and layout_box_pred is not None:
        layout_reg_terms = compute_layout_regularizers(
            layout_box_pred=layout_box_pred,
            node_mask=node_mask,
            min_center_spread=layout_min_center_spread,
        )

        layout_overlap_reg = layout_reg_terms["layout_overlap_reg"]
        layout_spread_reg = layout_reg_terms["layout_spread_reg"]
        layout_center_spread = layout_reg_terms["layout_center_spread"]

        layout_reg_loss = (
            float(layout_overlap_reg_weight) * layout_overlap_reg
            + float(layout_spread_reg_weight) * layout_spread_reg
        )
    else:
        layout_reg_loss = torch.zeros((), device=device, dtype=obj_logits.dtype)
        layout_overlap_reg = torch.zeros((), device=device, dtype=obj_logits.dtype)
        layout_spread_reg = torch.zeros((), device=device, dtype=obj_logits.dtype)
        layout_center_spread = torch.zeros((), device=device, dtype=obj_logits.dtype)
        layout_reg_loss = torch.zeros((), device=device, dtype=obj_logits.dtype)
    

    # -------------------------------------------------
    # Relation-aware geometry loss
    # -------------------------------------------------
    if (
        use_layout_supervision
        and use_relation_geometry_loss
        and lambda_rel_geometry > 0.0
        and relation_vocab is not None
        and ("layout_box_pred" in model_out)
        and (model_out["layout_box_pred"] is not None)
        and ("rel_full_0" in batch_t)
    ):
        rel_geom_dict = compute_relation_geometry_loss(
            model_out=model_out,
            batch_t=batch_t,
            relation_vocab=relation_vocab,
            geom_margin=rel_geom_margin,
        )
        rel_geometry_loss = rel_geom_dict["rel_geom_loss"]
        rel_geometry_count = rel_geom_dict["rel_geom_count"]
        rel_geom_left_count = rel_geom_dict["rel_geom_left_count"]
        rel_geom_right_count = rel_geom_dict["rel_geom_right_count"]
        rel_geom_above_count = rel_geom_dict["rel_geom_above_count"]
        rel_geom_below_count = rel_geom_dict["rel_geom_below_count"]
        rel_geom_inside_count = rel_geom_dict["rel_geom_inside_count"]
    else:
        rel_geometry_loss = torch.zeros((), device=device, dtype=obj_logits.dtype)
        rel_geometry_count = torch.zeros((), device=device, dtype=obj_logits.dtype)
        rel_geom_left_count = torch.zeros((), device=device, dtype=obj_logits.dtype)
        rel_geom_right_count = torch.zeros((), device=device, dtype=obj_logits.dtype)
        rel_geom_above_count = torch.zeros((), device=device, dtype=obj_logits.dtype)
        rel_geom_below_count = torch.zeros((), device=device, dtype=obj_logits.dtype)
        rel_geom_inside_count = torch.zeros((), device=device, dtype=obj_logits.dtype)
    
    # -------------------------------------------------
    # Layout class prior loss
    # -------------------------------------------------
    if (
        use_layout_class_priors
        and layout_box_pred is not None
        and layout_prior_mean is not None
        and layout_prior_var is not None
        and layout_prior_valid is not None
        and target_boxes is not None
        and box_valid_mask is not None
    ):
        prior_dict = compute_layout_class_prior_loss(
            layout_box_pred=layout_box_pred,
            obj_targets=obj_targets,
            node_mask=node_mask,
            box_valid_mask=box_valid_mask.bool(),
            prior_mean=layout_prior_mean.to(device=device, dtype=layout_box_pred.dtype),
            prior_var=layout_prior_var.to(device=device, dtype=layout_box_pred.dtype),
            prior_valid=layout_prior_valid.to(device=device),
            eps=layout_class_prior_eps,
        )
        layout_class_prior_loss = prior_dict["layout_class_prior_loss"]
        layout_class_prior_count = prior_dict["layout_class_prior_count"]
    else:
        layout_class_prior_loss = torch.zeros((), device=device, dtype=obj_logits.dtype)
        layout_class_prior_count = torch.zeros((), device=device, dtype=obj_logits.dtype)

    
    # -------------------------------------------------
    # Final predicted structure
    # -------------------------------------------------
    edge_prob = torch.sigmoid(edge_logits)
    edge_pred = (edge_prob >= edge_exist_thres).long()
    edge_pred = edge_pred * edge_mask.long()

    rel_pos_pred = rel_logits_pos.argmax(dim=-1)     # [B,N,N]
    pred_rel_full = rel_pos_pred + 1                 # assumes 0=no_rel in full vocab
    pred_rel_full = torch.where(
        edge_pred.bool(),
        pred_rel_full,
        torch.full_like(pred_rel_full, fill_value=no_rel_token_id),
    )
    pred_rel_full = torch.where(
        edge_mask,
        pred_rel_full,
        torch.full_like(pred_rel_full, fill_value=no_rel_token_id),
    )

    if pred_rel_full_override is not None:
        pred_rel_full = pred_rel_full_override

    # -------------------------------------------------
    # Final predicted nodes
    # -------------------------------------------------
    if pred_obj_override is not None:
        pred_obj_full = pred_obj_override
    else:
        pred_obj_full = obj_logits.argmax(dim=-1)

    # -------------------------------------------------
    # Edge metrics
    # -------------------------------------------------
    gt_edge = edge_targets.bool() & edge_mask
    pred_edge = edge_pred.bool() & edge_mask

    tp_edges = (pred_edge & gt_edge).sum().float()
    fp_edges = (pred_edge & (~gt_edge) & edge_mask).sum().float()
    fn_edges = ((~pred_edge) & gt_edge).sum().float()

    gt_num_pos_edges = gt_edge.sum().float()
    pred_num_pos_edges = pred_edge.sum().float()

    edge_precision = tp_edges / pred_num_pos_edges.clamp(min=1.0)
    edge_recall = tp_edges / gt_num_pos_edges.clamp(min=1.0)
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall).clamp(min=1e-8)
    pred_gt_edge_ratio = pred_num_pos_edges / gt_num_pos_edges.clamp(min=1.0)

    # -------------------------------------------------
    # Relation accuracy on true-positive predicted edges
    # -------------------------------------------------
    tp_edge_mask = pred_edge & gt_edge
    if tp_edge_mask.sum() > 0:
        rel_pred_full = pred_rel_full
        rel_gt_full = batch_t["rel_full_0"]
        relation_accuracy_on_true_positive_edges = (
            (rel_pred_full[tp_edge_mask] == rel_gt_full[tp_edge_mask]).float().mean()
        )
    else:
        relation_accuracy_on_true_positive_edges = torch.zeros((), device=device, dtype=obj_logits.dtype)

    # -------------------------------------------------
    # Node metrics
    # -------------------------------------------------
    node_metric_dict = compute_node_accuracy_metrics(
        obj_logits=obj_logits,
        obj_targets=obj_targets,
        node_mask=node_mask,
        obj_corrupt_mask=obj_corrupt_mask,     # true corruption for evaluation
        obj_mask_token_mask=obj_mask_token_mask,
        pred_obj=pred_obj_full,
    )

    # -------------------------
    # 8E.4 relation-geometry consistency regularizer
    # -------------------------
    if (
        use_relation_geometry_reg
        and lambda_relation_geometry_reg > 0.0
        and "layout_box_pred" in model_out
        and "rel_full_0" in batch_t
    ):

        rel_geom_reg_dict = compute_relation_geometry_regularizer(
            pred_rel_full=pred_rel_full.detach(),   # ← IMPORTANT CHANGE
            layout_box_pred=model_out["layout_box_pred"],
            node_mask=batch_t["node_mask"],              # ← NEW
            edge_mask=batch_t["edge_mask"],
            relation_vocab=relation_vocab,
            no_rel_token_id=no_rel_token_id,
            margin=relation_geometry_margin,
        )

        relation_geometry_reg = rel_geom_reg_dict["relation_geometry_reg"]
        rel_geom_reg_behind_count = rel_geom_reg_dict["rel_geom_reg_behind_count"]
        rel_geom_reg_front_count = rel_geom_reg_dict["rel_geom_reg_front_count"]
        rel_geom_reg_above_count = rel_geom_reg_dict["rel_geom_reg_above_count"]
        rel_geom_reg_below_count = rel_geom_reg_dict["rel_geom_reg_below_count"]
        rel_geom_reg_inside_count = rel_geom_reg_dict["rel_geom_reg_inside_count"]
        rel_geom_reg_on_count = rel_geom_reg_dict["rel_geom_reg_on_count"]

    else:
        relation_geometry_reg = torch.zeros((), device=model_out["obj_logits"].device)
        rel_geom_reg_behind_count = torch.zeros((), device=model_out["obj_logits"].device)
        rel_geom_reg_front_count = torch.zeros((), device=model_out["obj_logits"].device)
        rel_geom_reg_above_count = torch.zeros((), device=model_out["obj_logits"].device)
        rel_geom_reg_below_count = torch.zeros((), device=model_out["obj_logits"].device)
        rel_geom_reg_inside_count = torch.zeros((), device=model_out["obj_logits"].device)
        rel_geom_reg_on_count = torch.zeros((), device=model_out["obj_logits"].device)


    graph_law_reg = torch.zeros((), device=model_out["obj_logits"].device)
    graph_edge_density_reg = torch.zeros((), device=model_out["obj_logits"].device)
    graph_degree_reg = torch.zeros((), device=model_out["obj_logits"].device)
    graph_rel_marginal_reg = torch.zeros((), device=model_out["obj_logits"].device)
    pred_edge_density = torch.zeros((), device=model_out["obj_logits"].device)
    gt_edge_density = torch.zeros((), device=model_out["obj_logits"].device)

    if use_graph_law_reg and lambda_graph_law_reg > 0.0:
        graph_reg_dict = compute_graph_law_regularizer(
            edge_logits=model_out["edge_logits"],
            rel_logits_pos=model_out["rel_logits_pos"],
            rel_full_target=batch_t["rel_full_0"],
            edge_mask=batch_t["edge_mask"],
            no_rel_token_id=no_rel_token_id,
            edge_weight=graph_law_edge_weight,
            degree_weight=graph_law_degree_weight,
            rel_weight=graph_law_rel_weight,
            eps=graph_law_eps,
        )

        graph_law_reg = graph_reg_dict["graph_law_reg"]

        graph_edge_density_reg = graph_reg_dict["graph_edge_density_reg"]
        graph_degree_reg = graph_reg_dict["graph_degree_reg"]
        graph_rel_marginal_reg = graph_reg_dict["graph_rel_marginal_reg"]
        pred_edge_density = graph_reg_dict["pred_edge_density"]
        gt_edge_density = graph_reg_dict["gt_edge_density"]

    if object_only_sanity:


        layout_loss = layout_loss.detach() * 0.0
        rel_geometry_loss = rel_geometry_loss.detach() * 0.0
        layout_class_prior_loss = layout_class_prior_loss.detach() * 0.0
        layout_reg_loss = layout_reg_loss.detach() * 0.0 
        relation_geometry_reg = relation_geometry_reg.detach() * 0.0
        graph_law_reg = graph_law_reg.detach() * 0.0
    
    # else:
        
    # -------------------------------------------------
    # Total loss
    # -------------------------------------------------
    loss = lambda_obj * obj_loss + lambda_edge * edge_loss + lambda_rel * rel_loss + lambda_layout * layout_loss + lambda_rel_geometry * rel_geometry_loss + lambda_layout_class_prior * layout_class_prior_loss + layout_reg_loss + float(lambda_relation_geometry_reg) * relation_geometry_reg + float(lambda_graph_law_reg) * graph_law_reg



    return {
        "loss": loss,
        "obj_loss": obj_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "rel_loss": rel_loss.detach(),
        "layout_loss": layout_loss.detach(),
        "layout_l1": layout_l1.detach(),
        "layout_giou_loss": layout_giou_loss.detach(),
        "layout_mean_giou": layout_mean_giou.detach(),

        # 8E.2 / 8E.3 layout regularization terms
        "layout_overlap_reg": layout_overlap_reg.detach(),
        "layout_spread_reg": layout_spread_reg.detach(),
        "layout_center_spread": layout_center_spread.detach(),
        "layout_reg_loss": layout_reg_loss.detach(),

        "rel_geometry_loss": rel_geometry_loss.detach(),
        "rel_geometry_count": rel_geometry_count.detach(),

        "relation_geometry_reg": relation_geometry_reg.detach(),
        "rel_geom_reg_behind_count": rel_geom_reg_behind_count.detach(),
        "rel_geom_reg_front_count": rel_geom_reg_front_count.detach(),
        "rel_geom_reg_above_count": rel_geom_reg_above_count.detach(),
        "rel_geom_reg_below_count": rel_geom_reg_below_count.detach(),
        "rel_geom_reg_inside_count": rel_geom_reg_inside_count.detach(),
        "rel_geom_reg_on_count": rel_geom_reg_on_count.detach(),


        "layout_class_prior_loss": layout_class_prior_loss.detach(),
        "layout_class_prior_count": layout_class_prior_count.detach(),

        "rel_geometry_loss": rel_geometry_loss.detach(),
        "rel_geometry_count": rel_geometry_count.detach(),
        "rel_geom_left_count": rel_geom_left_count.detach(),
        "rel_geom_right_count": rel_geom_right_count.detach(),
        "rel_geom_above_count": rel_geom_above_count.detach(),
        "rel_geom_below_count": rel_geom_below_count.detach(),
        "rel_geom_inside_count": rel_geom_inside_count.detach(),

        "gt_num_pos_edges": gt_num_pos_edges.detach(),
        "pred_num_pos_edges": pred_num_pos_edges.detach(),
        "pred_gt_edge_ratio": pred_gt_edge_ratio.detach(),
        "edge_precision": edge_precision.detach(),
        "edge_recall": edge_recall.detach(),
        "edge_f1": edge_f1.detach(),
        "relation_accuracy_on_true_positive_edges": relation_accuracy_on_true_positive_edges.detach(),

        "node_acc_all": node_metric_dict["node_acc_all"].detach(),
        "node_count_all": node_metric_dict["node_count_all"].detach(),
        "node_acc_corrupted": node_metric_dict["node_acc_corrupted"].detach(),
        "node_count_corrupted": node_metric_dict["node_count_corrupted"].detach(),
        "node_acc_masked": node_metric_dict["node_acc_masked"].detach(),
        "node_count_masked": node_metric_dict["node_count_masked"].detach(),

        "pred_obj_full": pred_obj_full.detach(),
        "pred_rel_full": pred_rel_full.detach(),
        "layout_box_pred": layout_box_pred.detach() if layout_box_pred is not None else None,

        "tp_edges": tp_edges.detach(),
        "fp_edges": fp_edges.detach(),
        "fn_edges": fn_edges.detach(),

        "graph_law_reg": graph_law_reg.detach(),
        "graph_edge_density_reg": graph_edge_density_reg,
        "graph_degree_reg": graph_degree_reg,
        "graph_rel_marginal_reg": graph_rel_marginal_reg,
        "pred_edge_density": pred_edge_density,
        "gt_edge_density": gt_edge_density,
    }


def compute_reverse_vocab_step_loss(
    model_out,
    batch_rev,
    lambda_rev_obj: float,
    lambda_rev_edge: float,
    lambda_rev_rel: float,
    edge_pos_weight: float = 1.0,
    obj_rev_class_weights: torch.Tensor = None,
):
    """
    Reverse-step loss using separate reverse-vocab heads.
    Expected model_out keys:

obj_rev_logits:      [B,N,K_obj+1]
edge_rev_logits:     [B,N,N]
rel_rev_logits_pos:  [B,N,N,K_rel_pos+1]
    Expected batch_rev keys:

obj_prev_target_rev: [B,N]
edge_prev_target:    [B,N,N]
rel_prev_target_rev: [B,N,N]
obj_reverse_mask:    [B,N]
edge_reverse_mask:   [B,N,N]
rel_reverse_mask:    [B,N,N]
    """
    obj_rev_logits = model_out["obj_rev_logits"]
    edge_rev_logits = model_out["edge_rev_logits"]
    rel_rev_logits = model_out["rel_rev_logits_pos"]
    device = obj_rev_logits.device
    zero = obj_rev_logits.sum() * 0.0
    # -------------------------
    # Object reverse loss
    # -------------------------
    obj_loss_mask = batch_rev["obj_reverse_mask"]
    if obj_loss_mask.any():
        obj_logits_flat = obj_rev_logits[obj_loss_mask]             # [M, K_obj+1]
        obj_targets_flat = batch_rev["obj_prev_target_rev"][obj_loss_mask].long()
        obj_loss = F.cross_entropy(
            obj_logits_flat,
            obj_targets_flat,
            weight=obj_rev_class_weights,
            reduction="mean",
        )
    else:
        obj_loss = zero
    # -------------------------
    # Edge reverse loss
    # -------------------------
    edge_loss_mask = batch_rev["edge_reverse_mask"]
    if edge_loss_mask.any():
        edge_logits_flat = edge_rev_logits[edge_loss_mask]          # [M]
        edge_targets_flat = batch_rev["edge_prev_target"][edge_loss_mask].float()
        pos_weight = torch.tensor(
            edge_pos_weight,
            device=device,
            dtype=edge_logits_flat.dtype,
        )
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits_flat,
            edge_targets_flat,
            pos_weight=pos_weight,
            reduction="mean",
        )
    else:
        edge_loss = zero
    # -------------------------
    # Relation reverse loss
    # -------------------------
    rel_loss_mask = batch_rev["rel_reverse_mask"]
    if rel_loss_mask.any():
        rel_logits_flat = rel_rev_logits[rel_loss_mask]             # [M, K_rel_pos+1]
        rel_targets_flat = batch_rev["rel_prev_target_rev"][rel_loss_mask].long()
        rel_loss = F.cross_entropy(
            rel_logits_flat,
            rel_targets_flat,
            reduction="mean",
        )
    else:
        rel_loss = zero
    loss = (
        lambda_rev_obj * obj_loss
        + lambda_rev_edge * edge_loss
        + lambda_rev_rel * rel_loss
    )
    return {
        "loss": loss,
        "obj_loss": obj_loss,
        "edge_loss": edge_loss,
        "rel_loss": rel_loss,
    }

@torch.no_grad()
def count_object_classes(dataset) -> torch.Tensor:
    num_obj_classes = len(dataset.object_vocab)
    counts = torch.zeros(num_obj_classes, dtype=torch.float32)

    for i in range(len(dataset)):
        item = dataset[i]
        obj = item["obj_labels"]
        node_mask = item["node_mask"].bool()
        vals = obj[node_mask]
        counts.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))

    counts = torch.clamp(counts, min=1.0)
    return counts

@torch.no_grad()
def build_object_class_weights_effective_num(
    dataset,
    beta: float = 0.999,
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    device: torch.device = None,
):
    counts = count_object_classes(dataset)

    beta_t = torch.tensor(beta, dtype=torch.float32)
    effective_num = (1.0 - beta_t.pow(counts)) / (1.0 - beta_t)
    weights = 1.0 / effective_num
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=min_weight, max=max_weight)

    if device is not None:
        weights = weights.to(device)
    return weights

@torch.no_grad()
def compute_node_accuracy_metrics(
    obj_logits: torch.Tensor,           # [B,N,K]
    obj_targets: torch.Tensor,          # [B,N]
    node_mask: torch.Tensor,            # [B,N]
    obj_corrupt_mask: torch.Tensor,     # [B,N]
    obj_mask_token_mask: torch.Tensor,  # [B,N]
    pred_obj: torch.Tensor = None,      # [B,N] optional split prediction
):
    if pred_obj is None:
        pred = obj_logits.argmax(dim=-1)
    else:
        pred = pred_obj

    valid = node_mask.bool()
    corrupted = valid & obj_corrupt_mask.bool()
    masked_only = valid & obj_mask_token_mask.bool()

    total_valid = valid.sum().float().clamp(min=1.0)
    total_corr = corrupted.sum().float().clamp(min=1.0)
    total_masked = masked_only.sum().float().clamp(min=1.0)

    valid_acc = ((pred == obj_targets) & valid).sum().float() / total_valid
    corrupted_acc = ((pred == obj_targets) & corrupted).sum().float() / total_corr
    masked_acc = ((pred == obj_targets) & masked_only).sum().float() / total_masked

    return {
        "node_acc_all": valid_acc,
        "node_acc_corrupted": corrupted_acc,
        "node_acc_masked": masked_acc,
        "node_count_all": total_valid,
        "node_count_corrupted": total_corr,
        "node_count_masked": total_masked,
    }

@torch.no_grad()
def build_split_obj_prediction(
    obj_logits: torch.Tensor,       # [B,N,K]
    obj_t: torch.Tensor,            # [B,N]
    node_mask: torch.Tensor,        # [B,N]
    obj_update_mask: torch.Tensor,  # [B,N] bool
    value_update_mode: str = "argmax",   # "argmax" or "sample"
    sample_temp: float = 1.0,
    node_sample_conf_thresh: float = 0.7
) -> torch.Tensor:
    """
    Split reverse for nodes:
      - update only nodes in obj_update_mask
      - copy through obj_t elsewhere

    For selected nodes:
      - argmax mode: deterministic MAP update
      - sample mode: stochastic categorical sample from softened logits
    """
    valid = node_mask.bool()
    update = obj_update_mask.bool() & valid

    if value_update_mode == "argmax":
        pred_obj_selected = obj_logits.argmax(dim=-1)   # [B,N]

    elif value_update_mode == "sample":

        logits = obj_logits / max(sample_temp, 1e-8)
        probs = torch.softmax(logits, dim=-1)

        max_prob, argmax = probs.max(dim=-1)

        B, N, K = probs.shape
        sampled = torch.multinomial(
            probs.view(B * N, K),
            num_samples=1,
        ).view(B, N)

        # NEW: confidence gating
        pred_obj_selected = torch.where(
            max_prob > node_sample_conf_thresh,
            argmax,
            sampled
        )

    else:
        raise ValueError(f"Unknown node_value_update_mode: {value_update_mode}")

    pred_obj_full = torch.where(update, pred_obj_selected, obj_t)
    pred_obj_full = torch.where(valid, pred_obj_full, obj_t)
    return pred_obj_full  

@torch.no_grad()
def infer_obj_update_mask_from_logits(
    obj_logits: torch.Tensor,   # [B,N,K]
    obj_t: torch.Tensor,        # [B,N]
    node_mask: torch.Tensor,    # [B,N]
    conf_thresh: float = 0.80,
    strategy: str = "confidence_or_mismatch",
) -> torch.Tensor:
    """
    Infer which nodes should be updated without oracle corruption mask.

    Strategies:
      - confidence_or_mismatch:
            update if predicted class != current obj_t OR confidence < threshold
      - confidence_only:
            update if confidence < threshold
      - mismatch_only:
            update if predicted class != current obj_t
    """
    probs = torch.softmax(obj_logits, dim=-1)          # [B,N,K]
    conf, pred = probs.max(dim=-1)                     # [B,N]

    valid = node_mask.bool()
    mismatch = (pred != obj_t) & valid
    low_conf = (conf < conf_thresh) & valid

    if strategy == "confidence_or_mismatch":
        update = mismatch | low_conf
    elif strategy == "confidence_only":
        update = low_conf
    elif strategy == "mismatch_only":
        update = mismatch
    else:
        raise ValueError(f"Unknown node_update_strategy: {strategy}")

    return update

@torch.no_grad()
def infer_obj_update_mask_stochastic(
    obj_logits: torch.Tensor,   # [B,N,K]
    obj_t: torch.Tensor,        # [B,N]
    node_mask: torch.Tensor,    # [B,N]
    prob_scale: float = 2.0,
    prob_power: float = 1.0,
    min_prob: float = 0.0,
    max_prob: float = 0.75,
) -> torch.Tensor:
    """
    Stochastic node-update mask from model uncertainty.

    Steps:
      1. compute confidence = max softmax prob
      2. uncertainty = 1 - confidence
      3. map uncertainty to update probability
      4. sample Bernoulli(update_prob)

    Returns:
        update_mask: [B,N] bool
    """
    probs = torch.softmax(obj_logits, dim=-1)   # [B,N,K]
    conf, _ = probs.max(dim=-1)                 # [B,N]

    uncertainty = 1.0 - conf                    # [B,N]

    # scaled uncertainty -> update probability
    update_prob = prob_scale * uncertainty
    update_prob = update_prob.clamp(min=0.0, max=1.0)

    if prob_power != 1.0:
        update_prob = update_prob.pow(prob_power)

    update_prob = update_prob.clamp(min=min_prob, max=max_prob)

    valid = node_mask.bool()
    update_prob = torch.where(valid, update_prob, torch.zeros_like(update_prob))

    sampled = torch.bernoulli(update_prob).bool()
    sampled = sampled & valid
    return sampled

@torch.no_grad()
def compute_update_probability_map(
    obj_logits: torch.Tensor,   # [B,N,K]
    node_mask: torch.Tensor,    # [B,N]
    prob_scale: float = 2.0,
    prob_power: float = 1.0,
    min_prob: float = 0.0,
    max_prob: float = 0.75,
) -> torch.Tensor:
    probs = torch.softmax(obj_logits, dim=-1)
    conf, _ = probs.max(dim=-1)
    uncertainty = 1.0 - conf

    update_prob = prob_scale * uncertainty
    update_prob = update_prob.clamp(min=0.0, max=1.0)

    if prob_power != 1.0:
        update_prob = update_prob.pow(prob_power)

    update_prob = update_prob.clamp(min=min_prob, max=max_prob)
    update_prob = torch.where(node_mask.bool(), update_prob, torch.zeros_like(update_prob))
    return update_prob

def compute_conditional_node_loss(
    obj_logits: torch.Tensor,          # [B,N,K]
    obj_targets: torch.Tensor,         # [B,N]
    query_mask: torch.Tensor,          # [B,N] bool
    node_mask: torch.Tensor,           # [B,N] bool
    obj_class_weights: torch.Tensor = None,
    use_object_focal_loss: bool = False,
    object_focal_gamma: float = 2.0,
    object_focal_alpha: float = 1.0,
):
    """
    Conditional node objective:
    supervise only the selected query nodes.
    """
    loss_mask = query_mask.bool() & node_mask.bool()

    if loss_mask.sum() == 0:
        # safety fallback
        zero = obj_logits.sum() * 0.0
        return {
            "cond_node_loss": zero,
            "cond_node_acc": zero.detach(),
            "cond_node_count": zero.detach(),
            "cond_pred_obj": obj_logits.argmax(dim=-1).detach(),
        }

    logits_flat = obj_logits[loss_mask]
    targets_flat = obj_targets[loss_mask]

    if use_object_focal_loss:
        ce = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
            weight=obj_class_weights,
        )
        pt = torch.exp(-ce)
        cond_node_loss = (object_focal_alpha * ((1 - pt) ** object_focal_gamma) * ce).mean()
    else:
        cond_node_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean",
            weight=obj_class_weights,
        )

    pred_obj = obj_logits.argmax(dim=-1)
    cond_node_acc = (pred_obj[loss_mask] == obj_targets[loss_mask]).float().mean()
    cond_node_count = loss_mask.sum().float()

    return {
        "cond_node_loss": cond_node_loss,
        "cond_node_acc": cond_node_acc.detach(),
        "cond_node_count": cond_node_count.detach(),
        "cond_pred_obj": pred_obj.detach(),
    }

def build_discrete_predictions_from_model_out(
    model_out: dict,
    edge_exist_thres: float,
):
    """
    Convert model outputs to discrete graph predictions for refinement pass.

    Returns:
        pred_obj:      [B,N]
        pred_edge:     [B,N,N] in {0,1}
        pred_rel_pos:  [B,N,N] in {0,...,K_rel_pos-1}
    """
    obj_logits = model_out["obj_logits"]
    edge_logits = model_out["edge_logits"]
    rel_logits_pos = model_out["rel_logits_pos"]

    pred_obj = obj_logits.argmax(dim=-1)
    pred_edge = (torch.sigmoid(edge_logits) >= edge_exist_thres).long()
    pred_rel_pos = rel_logits_pos.argmax(dim=-1)

    return {
        "pred_obj": pred_obj,
        "pred_edge": pred_edge,
        "pred_rel_pos": pred_rel_pos,
    }



def compute_refinement_sg_loss(
    model_out_refine: dict,
    batch_t: dict,
    first_pass_pred_obj: torch.Tensor,
    no_rel_token_id: int,
    lambda_obj: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_rel: float = 1.0,
    edge_exist_thres: float = 0.5,
    edge_pos_weight: float = 1.0,
    obj_class_weights: torch.Tensor = None,
    refine_obj_wrong_weight: float = 3.0,
    refine_obj_base_weight: float = 0.25,
    use_object_focal_loss: bool = False,
    object_focal_gamma: float = 2.0,
    object_focal_alpha: float = 1.0,
):
    """
    Phase 4F.2:
    Same SG refinement loss as before, but node loss is reweighted so that
    nodes that pass 1 got wrong receive higher weight.
    """
    obj_logits = model_out_refine["obj_logits"]
    edge_logits = model_out_refine["edge_logits"]
    rel_logits_pos = model_out_refine["rel_logits_pos"]

    obj_0 = batch_t["obj_0"]
    edge_0 = batch_t["edge_0"].float()
    rel_pos_0 = batch_t["rel_pos_0"]
    node_mask = batch_t["node_mask"].bool()
    edge_mask = batch_t["edge_mask"].bool()

    # -------------------------
    # Weighted object refinement loss
    # -------------------------
    obj_valid = node_mask
    wrong_mask = (first_pass_pred_obj != obj_0) & obj_valid

    obj_weights = torch.where(
        wrong_mask,
        torch.full_like(obj_0, fill_value=refine_obj_wrong_weight, dtype=torch.float32),
        torch.full_like(obj_0, fill_value=refine_obj_base_weight, dtype=torch.float32),
    )

    logits_flat = obj_logits[obj_valid]
    targets_flat = obj_0[obj_valid]
    weights_flat = obj_weights[obj_valid]

    if use_object_focal_loss:
        ce = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
            weight=obj_class_weights,
        )
        pt = torch.exp(-ce)
        obj_loss_vec = object_focal_alpha * ((1 - pt) ** object_focal_gamma) * ce
    else:
        obj_loss_vec = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
            weight=obj_class_weights,
        )

    obj_loss = (obj_loss_vec * weights_flat).sum() / weights_flat.sum().clamp(min=1e-12)

    # -------------------------
    # Edge loss (unchanged for first 4F.2)
    # -------------------------
    edge_valid = edge_mask
    edge_targets = edge_0[edge_valid]
    edge_logits_flat = edge_logits[edge_valid]

    pos_weight = torch.tensor(edge_pos_weight, device=edge_logits.device, dtype=edge_logits.dtype)
    edge_loss = F.binary_cross_entropy_with_logits(
        edge_logits_flat,
        edge_targets,
        reduction="mean",
        pos_weight=pos_weight,
    )

    # -------------------------
    # Relation loss on true positive edges
    # -------------------------
    gt_edge = (edge_0 > 0.5) & edge_mask
    if gt_edge.any():
        rel_loss = F.cross_entropy(
            rel_logits_pos[gt_edge],
            rel_pos_0[gt_edge],
            reduction="mean",
        )
    else:
        rel_loss = edge_logits.sum() * 0.0

    loss = lambda_obj * obj_loss + lambda_edge * edge_loss + lambda_rel * rel_loss

    # -------------------------
    # Diagnostics, same style as structured SG loss
    # -------------------------
    pred_obj = obj_logits.argmax(dim=-1)

    pred_edge = (torch.sigmoid(edge_logits) >= edge_exist_thres) & edge_mask
    gt_edge_bool = gt_edge

    tp_edges = (pred_edge & gt_edge_bool).sum().float()
    fp_edges = (pred_edge & (~gt_edge_bool) & edge_mask).sum().float()
    fn_edges = ((~pred_edge) & gt_edge_bool).sum().float()

    pred_num_pos_edges = pred_edge.sum().float()
    gt_num_pos_edges = gt_edge_bool.sum().float()

    node_count_all = node_mask.sum().float()
    node_acc_all = ((pred_obj == obj_0) & node_mask).sum().float() / node_count_all.clamp(min=1.0)

    corrupted_mask = (batch_t["obj_t"] != obj_0) & node_mask
    node_count_corrupted = corrupted_mask.sum().float()
    if node_count_corrupted.item() > 0:
        node_acc_corrupted = ((pred_obj == obj_0) & corrupted_mask).sum().float() / node_count_corrupted
    else:
        node_acc_corrupted = obj_loss.detach() * 0.0

    # relation accuracy on TP edges
    if tp_edges.item() > 0:
        pred_rel = rel_logits_pos.argmax(dim=-1)
        rel_acc_tp = ((pred_rel == rel_pos_0) & pred_edge & gt_edge_bool).sum().float() / tp_edges.clamp(min=1.0)
    else:
        rel_acc_tp = obj_loss.detach() * 0.0

    return {
        "loss": loss,
        "obj_loss": obj_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "rel_loss": rel_loss.detach(),
        "tp_edges": tp_edges.detach(),
        "fp_edges": fp_edges.detach(),
        "fn_edges": fn_edges.detach(),
        "pred_num_pos_edges": pred_num_pos_edges.detach(),
        "gt_num_pos_edges": gt_num_pos_edges.detach(),
        "relation_accuracy_on_true_positive_edges": rel_acc_tp.detach(),
        "node_acc_all": node_acc_all.detach(),
        "node_count_all": node_count_all.detach(),
        "node_acc_corrupted": node_acc_corrupted.detach(),
        "node_count_corrupted": node_count_corrupted.detach(),
    }


def compute_final_graph_metrics(
    pred_obj: torch.Tensor,
    pred_edge: torch.Tensor,
    pred_rel_pos: torch.Tensor,
    batch_clean: dict,
    batch_start: dict,
):
    obj_0 = batch_clean["obj_0"]
    edge_0 = batch_clean["edge_0"].bool()
    rel_pos_0 = batch_clean["rel_pos_0"]
    node_mask = batch_clean["node_mask"].bool()
    edge_mask = batch_clean["edge_mask"].bool()

    pred_edge = pred_edge.bool() & edge_mask
    gt_edge = edge_0 & edge_mask

    tp_edges = (pred_edge & gt_edge).sum().float()
    fp_edges = (pred_edge & (~gt_edge) & edge_mask).sum().float()
    fn_edges = ((~pred_edge) & gt_edge).sum().float()

    pred_num_pos_edges = pred_edge.sum().float()
    gt_num_pos_edges = gt_edge.sum().float()

    node_count_all = node_mask.sum().float()
    node_acc_all = ((pred_obj == obj_0) & node_mask).sum().float() / node_count_all.clamp(min=1.0)

    corrupted_mask = (batch_start["obj_t"] != obj_0) & node_mask
    node_count_corrupted = corrupted_mask.sum().float()
    if node_count_corrupted.item() > 0:
        node_acc_corrupted = ((pred_obj == obj_0) & corrupted_mask).sum().float() / node_count_corrupted
    else:
        node_acc_corrupted = tp_edges.detach() * 0.0

    if tp_edges.item() > 0:
        rel_acc_tp = ((pred_rel_pos == rel_pos_0) & pred_edge & gt_edge).sum().float() / tp_edges.clamp(min=1.0)
    else:
        rel_acc_tp = tp_edges.detach() * 0.0

    return {
        "tp_edges": tp_edges.detach(),
        "fp_edges": fp_edges.detach(),
        "fn_edges": fn_edges.detach(),
        "pred_num_pos_edges": pred_num_pos_edges.detach(),
        "gt_num_pos_edges": gt_num_pos_edges.detach(),
        "relation_accuracy_on_true_positive_edges": rel_acc_tp.detach(),
        "node_acc_all": node_acc_all.detach(),
        "node_count_all": node_count_all.detach(),
        "node_acc_corrupted": node_acc_corrupted.detach(),
        "node_count_corrupted": node_count_corrupted.detach(),
    }

import torch
import torch.nn.functional as F


def compute_layout_loss(
    model_out: dict,
    batch_t: dict,
    layout_loss_type: str = "smooth_l1",
    use_layout_giou_loss: bool = False,
    lambda_layout_giou: float = 0.5,
):
    """
    Layout loss on normalized boxes in (cx, cy, w, h).

    Expected:
        model_out["layout_box_pred"] : [B, N, 4]
        batch_t["boxes"]         : [B, N, 4]
        batch_t["box_valid_mask"]: [B, N]

    Supported layout_loss_type:

"l1"
"smooth_l1"
"mse"
    """
    if "layout_box_pred" not in model_out:
        raise KeyError("model_out must contain 'layout_box_pred'")
    if "boxes" not in batch_t:
        raise KeyError("batch_t must contain 'boxes'")
    if "box_valid_mask" not in batch_t:
        raise KeyError("batch_t must contain 'box_valid_mask'")

    pred_boxes = model_out["layout_box_pred"]                  # [B,N,4]
    target_boxes = batch_t["boxes"].float()                # [B,N,4]
    box_valid_mask = batch_t["box_valid_mask"].bool()      # [B,N]

    if pred_boxes.shape != target_boxes.shape:
        raise ValueError(
            f"layout_box_pred shape {pred_boxes.shape} does not match target boxes shape {target_boxes.shape}"
        )

    if box_valid_mask.shape != pred_boxes.shape[:2]:
        raise ValueError(
            f"box_valid_mask shape {box_valid_mask.shape} does not match first two dims of layout_box_pred {pred_boxes.shape[:2]}"
        )

    valid_count = box_valid_mask.sum()

    if valid_count.item() == 0:
        zero = pred_boxes.sum() * 0.0
        return {
            "layout_loss": zero,
            "layout_l1": zero.detach(),
            "layout_giou_loss": zero.detach(),
            "layout_mean_giou": zero.detach(),
        }

    pred_valid = pred_boxes[box_valid_mask]        # [M,4]
    target_valid = target_boxes[box_valid_mask]    # [M,4]

    layout_loss_type = layout_loss_type.lower()

    if layout_loss_type == "l1":
        coord_loss = F.l1_loss(pred_valid, target_valid, reduction="mean")
    elif layout_loss_type == "smooth_l1":
        coord_loss = F.smooth_l1_loss(pred_valid, target_valid, reduction="mean")
    elif layout_loss_type == "mse":
        coord_loss = F.mse_loss(pred_valid, target_valid, reduction="mean")
    else:
        raise ValueError(
            f"Unknown layout_loss_type='{layout_loss_type}'. "
            f"Supported: 'l1', 'smooth_l1', 'mse'."
        )

    layout_l1 = F.l1_loss(pred_valid, target_valid, reduction="mean")

    if use_layout_giou_loss:
        pred_xyxy = box_cxcywh_to_xyxy(pred_valid)
        target_xyxy = box_cxcywh_to_xyxy(target_valid)

        giou = generalized_box_iou(pred_xyxy, target_xyxy)   # [M]
        giou_loss = (1.0 - giou).mean()
        mean_giou = giou.mean()

        layout_loss = coord_loss + lambda_layout_giou * giou_loss
    else:
        giou_loss = coord_loss.new_zeros(())
        mean_giou = coord_loss.new_zeros(())
        layout_loss = coord_loss

    return {
        "layout_loss": layout_loss,
        "layout_l1": layout_l1.detach(),
        "layout_giou_loss": giou_loss.detach(),
        "layout_mean_giou": mean_giou.detach(),
    }


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [..., 4] in normalized (cx, cy, w, h)
    returns: [..., 4] in normalized (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    return out.clamp(min=0.0, max=1.0)



def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1, boxes2: [M,4] in xyxy
    returns: [M] pairwise aligned GIoU
    """
    x11, y11, x12, y12 = boxes1.unbind(dim=-1)
    x21, y21, x22, y22 = boxes2.unbind(dim=-1)

    # intersection
    xi1 = torch.maximum(x11, x21)
    yi1 = torch.maximum(y11, y21)
    xi2 = torch.minimum(x12, x22)
    yi2 = torch.minimum(y12, y22)

    inter_w = (xi2 - xi1).clamp(min=0.0)
    inter_h = (yi2 - yi1).clamp(min=0.0)
    inter = inter_w * inter_h

    # areas
    area1 = (x12 - x11).clamp(min=0.0) * (y12 - y11).clamp(min=0.0)
    area2 = (x22 - x21).clamp(min=0.0) * (y22 - y21).clamp(min=0.0)

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-8)

    # enclosing box
    xc1 = torch.minimum(x11, x21)
    yc1 = torch.minimum(y11, y21)
    xc2 = torch.maximum(x12, x22)
    yc2 = torch.maximum(y12, y22)

    enc_w = (xc2 - xc1).clamp(min=0.0)
    enc_h = (yc2 - yc1).clamp(min=0.0)
    enc_area = enc_w * enc_h

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-8)
    return giou

def compute_relation_geometry_loss(
    model_out: dict,
    batch_t: dict,
    relation_vocab: list,
    geom_margin: float = 0.02,
):
    device = batch_t["node_mask"].device

    if "layout_box_pred" not in model_out or model_out["layout_box_pred"] is None:
        z = torch.zeros((), device=device)
        return {
            "rel_geom_loss": z,
            "rel_geom_count": z,
            "rel_geom_left_count": z,
            "rel_geom_right_count": z,
            "rel_geom_above_count": z,
            "rel_geom_below_count": z,
            "rel_geom_inside_count": z,
        }

    if "rel_full_0" not in batch_t:
        z = torch.zeros((), device=device)
        return {
            "rel_geom_loss": z,
            "rel_geom_count": z,
            "rel_geom_left_count": z,
            "rel_geom_right_count": z,
            "rel_geom_above_count": z,
            "rel_geom_below_count": z,
            "rel_geom_inside_count": z,
        }

    pred_boxes = model_out["layout_box_pred"]
    rel_full = batch_t["rel_full_0"]
    edge_mask = batch_t["edge_mask"].bool()
    node_mask = batch_t["node_mask"].bool()

    cx = pred_boxes[..., 0]
    cy = pred_boxes[..., 1]
    w = pred_boxes[..., 2].clamp(min=1e-6)
    h = pred_boxes[..., 3].clamp(min=1e-6)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    sx1 = x1.unsqueeze(2)
    sy1 = y1.unsqueeze(2)
    sx2 = x2.unsqueeze(2)
    sy2 = y2.unsqueeze(2)

    ox1 = x1.unsqueeze(1)
    oy1 = y1.unsqueeze(1)
    ox2 = x2.unsqueeze(1)
    oy2 = y2.unsqueeze(1)

    valid_pair = edge_mask & node_mask.unsqueeze(2) & node_mask.unsqueeze(1)

    total_loss = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)

    left_count = torch.zeros((), device=device)
    right_count = torch.zeros((), device=device)
    above_count = torch.zeros((), device=device)
    below_count = torch.zeros((), device=device)
    inside_count = torch.zeros((), device=device)

    def ids_for_names(name_list):
        out = []
        for k, name in enumerate(relation_vocab):
            lname = str(name).strip().lower()
            if lname in name_list:
                out.append(k)
        return out
    def ids_for_substrings(name_list):
        out = []
        for k, name in enumerate(relation_vocab):
            lname = str(name).strip().lower()
            if any(alias in lname for alias in name_list):
                out.append(k)
        return out

    # left_ids = ids_for_names(["left"])
    # right_ids = ids_for_names(["right"])
    left_ids = ids_for_substrings([
            "left of",
            "to the left of",
            "on the left of",
            "left",
        ])

    right_ids = ids_for_substrings([
        "right of",
        "to the right of",
        "on the right of",
        "right",
    ])
    above_ids = ids_for_names(["above", "over", "on top of"])
    below_ids = ids_for_names(["below", "under", "beneath"])
    inside_ids = ids_for_names(["in", "inside", "inside of"])

    def add_group_loss(rel_ids, per_pair_loss):
        if len(rel_ids) == 0:
            return torch.zeros((), device=device), torch.zeros((), device=device)

        rel_mask = torch.zeros_like(rel_full, dtype=torch.bool)
        for rid in rel_ids:
            rel_mask |= (rel_full == rid)

        rel_mask = rel_mask & valid_pair
        if rel_mask.any():
            return per_pair_loss[rel_mask].mean(), rel_mask.sum().float()
        return torch.zeros((), device=device), torch.zeros((), device=device)

    left_loss = F.relu(sx2 + geom_margin - ox1)
    left_group_loss, left_count = add_group_loss(left_ids, left_loss)

    right_loss = F.relu(ox2 + geom_margin - sx1)
    right_group_loss, right_count = add_group_loss(right_ids, right_loss)

    above_loss = F.relu(sy2 + geom_margin - oy1)
    above_group_loss, above_count = add_group_loss(above_ids, above_loss)

    below_loss = F.relu(oy2 + geom_margin - sy1)
    below_group_loss, below_count = add_group_loss(below_ids, below_loss)

    inside_loss = (
        F.relu(ox1 - sx1 + geom_margin) +
        F.relu(oy1 - sy1 + geom_margin) +
        F.relu(sx2 - ox2 + geom_margin) +
        F.relu(sy2 - oy2 + geom_margin)
    ) * 0.25
    inside_group_loss, inside_count = add_group_loss(inside_ids, inside_loss)

    group_losses = []
    for loss_i, count_i in [
        (left_group_loss, left_count),
        (right_group_loss, right_count),
        (above_group_loss, above_count),
        (below_group_loss, below_count),
        (inside_group_loss, inside_count),
    ]:
        if count_i.item() > 0:
            group_losses.append(loss_i)

    if len(group_losses) == 0:
        z = torch.zeros((), device=device)
        return {
            "rel_geom_loss": z,
            "rel_geom_count": z,
            "rel_geom_left_count": left_count,
            "rel_geom_right_count": right_count,
            "rel_geom_above_count": above_count,
            "rel_geom_below_count": below_count,
            "rel_geom_inside_count": inside_count,
        }

    total_loss = torch.stack(group_losses).mean()
    total_count = left_count + right_count + above_count + below_count + inside_count

    return {
        "rel_geom_loss": total_loss,
        "rel_geom_count": total_count,
        "rel_geom_left_count": left_count,
        "rel_geom_right_count": right_count,
        "rel_geom_above_count": above_count,
        "rel_geom_below_count": below_count,
        "rel_geom_inside_count": inside_count,
    }

def compute_layout_class_prior_loss(
    layout_box_pred: torch.Tensor,      # [B,N,4]
    obj_targets: torch.Tensor,          # [B,N]
    node_mask: torch.Tensor,            # [B,N]
    box_valid_mask: torch.Tensor,       # [B,N]
    prior_mean: torch.Tensor,           # [K,4]
    prior_var: torch.Tensor,            # [K,4]
    prior_valid: torch.Tensor,          # [K]
    eps: float = 1e-4,
):
    device = layout_box_pred.device
    dtype = layout_box_pred.dtype

    valid_mask = node_mask.bool() & box_valid_mask.bool()
    if valid_mask.sum() == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "layout_class_prior_loss": zero,
            "layout_class_prior_count": zero,
        }

    cls_ids = obj_targets[valid_mask]         # [M]
    pred_boxes = layout_box_pred[valid_mask]  # [M,4]

    class_is_valid = prior_valid[cls_ids]
    if class_is_valid.sum() == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "layout_class_prior_loss": zero,
            "layout_class_prior_count": zero,
        }

    cls_ids = cls_ids[class_is_valid]
    pred_boxes = pred_boxes[class_is_valid]

    mu = prior_mean[cls_ids]                  # [M,4]
    var = prior_var[cls_ids].clamp(min=eps)   # [M,4]

    loss = (((pred_boxes - mu) ** 2) / var).mean()

    return {
        "layout_class_prior_loss": loss,
        "layout_class_prior_count": class_is_valid.sum().to(dtype),
    }

# def compute_layout_regularizers(layout_box_pred, node_mask):
#     valid = node_mask.bool()
#     valid_f = valid.float()

#     cx, cy = layout_box_pred[..., 0], layout_box_pred[..., 1]
#     w = layout_box_pred[..., 2].clamp(min=1e-6)
#     h = layout_box_pred[..., 3].clamp(min=1e-6)

#     x1, y1 = cx - 0.5 * w, cy - 0.5 * h
#     x2, y2 = cx + 0.5 * w, cy + 0.5 * h

#     # pairwise IoU
#     inter_x1 = torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))
#     inter_y1 = torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))
#     inter_x2 = torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1))
#     inter_y2 = torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1))

#     inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
#     area = ((x2 - x1).clamp(min=1e-6) * (y2 - y1).clamp(min=1e-6))
#     union = area.unsqueeze(2) + area.unsqueeze(1) - inter
#     iou = inter / union.clamp(min=1e-6)

#     B, N = valid.shape
#     pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
#     eye = torch.eye(N, device=valid.device, dtype=torch.bool).unsqueeze(0)
#     pair_mask = pair_mask & ~eye

#     overlap_reg = (iou * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1.0)

#     # spread anti-collapse
#     denom = valid_f.sum(dim=1).clamp(min=1.0)
#     mean_cx = (cx * valid_f).sum(dim=1) / denom
#     mean_cy = (cy * valid_f).sum(dim=1) / denom

#     var_x = ((cx - mean_cx[:, None]) ** 2 * valid_f).sum(dim=1) / denom
#     var_y = ((cy - mean_cy[:, None]) ** 2 * valid_f).sum(dim=1) / denom
#     spread = torch.sqrt((var_x + var_y).clamp(min=1e-8))

#     return overlap_reg, spread

def compute_layout_regularizers(
    layout_box_pred: torch.Tensor,   # [B,N,4], cxcywh normalized
    node_mask: torch.Tensor,         # [B,N]
    min_center_spread: float = 0.18,
):
    valid = node_mask.bool()
    valid_f = valid.float()

    cx = layout_box_pred[..., 0]
    cy = layout_box_pred[..., 1]
    w = layout_box_pred[..., 2].clamp(min=1e-6)
    h = layout_box_pred[..., 3].clamp(min=1e-6)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    # -------------------------
    # Pairwise overlap regularizer
    # -------------------------
    inter_x1 = torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))
    inter_y1 = torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))
    inter_x2 = torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1))
    inter_y2 = torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1))

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area = ((x2 - x1).clamp(min=1e-6) * (y2 - y1).clamp(min=1e-6))
    union = area.unsqueeze(2) + area.unsqueeze(1) - inter
    iou = inter / union.clamp(min=1e-6)

    B, N = valid.shape
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(N, device=valid.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & (~eye)

    overlap_reg = (
        iou * pair_mask.float()
    ).sum() / pair_mask.float().sum().clamp(min=1.0)

    # -------------------------
    # Spread anti-collapse regularizer
    # -------------------------
    denom = valid_f.sum(dim=1).clamp(min=1.0)

    mean_cx = (cx * valid_f).sum(dim=1) / denom
    mean_cy = (cy * valid_f).sum(dim=1) / denom

    var_x = (((cx - mean_cx[:, None]) ** 2) * valid_f).sum(dim=1) / denom
    var_y = (((cy - mean_cy[:, None]) ** 2) * valid_f).sum(dim=1) / denom

    center_spread = torch.sqrt((var_x + var_y).clamp(min=1e-8))

    spread_reg = F.relu(
        float(min_center_spread) - center_spread
    ).mean()

    return {
        "layout_overlap_reg": overlap_reg,
        "layout_spread_reg": spread_reg,
        "layout_center_spread": center_spread.mean(),
    }

# def compute_relation_geometry_regularizer(
#     layout_box_pred: torch.Tensor,      # [B,N,4], cxcywh
#     rel_full_target: torch.Tensor,      # [B,N,N], full relation ids, no_rel included
#     edge_mask: torch.Tensor,            # [B,N,N]
#     relation_vocab,
#     no_rel_token_id: int,
#     margin: float = 0.03,
# ):
#     device = layout_box_pred.device

#     cx = layout_box_pred[..., 0]
#     cy = layout_box_pred[..., 1]
#     w = layout_box_pred[..., 2].clamp(min=1e-6)
#     h = layout_box_pred[..., 3].clamp(min=1e-6)

#     x1 = cx - 0.5 * w
#     y1 = cy - 0.5 * h
#     x2 = cx + 0.5 * w
#     y2 = cy + 0.5 * h

#     # subject i, object j
#     cx_i = cx.unsqueeze(2)
#     cy_i = cy.unsqueeze(2)
#     x1_i = x1.unsqueeze(2)
#     y1_i = y1.unsqueeze(2)
#     x2_i = x2.unsqueeze(2)
#     y2_i = y2.unsqueeze(2)

#     cx_j = cx.unsqueeze(1)
#     cy_j = cy.unsqueeze(1)
#     x1_j = x1.unsqueeze(1)
#     y1_j = y1.unsqueeze(1)
#     x2_j = x2.unsqueeze(1)
#     y2_j = y2.unsqueeze(1)

#     valid_rel = edge_mask.bool() & (rel_full_target != no_rel_token_id)

#     rel_name_to_id = {name: idx for idx, name in enumerate(relation_vocab)}

#     def rel_mask(names):
#         ids = [rel_name_to_id[n] for n in names if n in rel_name_to_id]
#         if len(ids) == 0:
#             return torch.zeros_like(valid_rel)
#         ids_t = torch.tensor(ids, device=device, dtype=rel_full_target.dtype)
#         return valid_rel & torch.isin(rel_full_target, ids_t)

#     losses = []
#     counts = {}

#     # i left of j: cx_i < cx_j
#     m_left = rel_mask(["left of", "to the left of"])
#     if m_left.any():
#         loss_left = F.relu(cx_i - cx_j + margin)[m_left].mean()
#         losses.append(loss_left)
#         counts["rel_geom_reg_left_count"] = m_left.float().sum()
#     else:
#         counts["rel_geom_reg_left_count"] = torch.zeros((), device=device)

#     # i right of j: cx_i > cx_j
#     m_right = rel_mask(["right of", "to the right of"])
#     if m_right.any():
#         loss_right = F.relu(cx_j - cx_i + margin)[m_right].mean()
#         losses.append(loss_right)
#         counts["rel_geom_reg_right_count"] = m_right.float().sum()
#     else:
#         counts["rel_geom_reg_right_count"] = torch.zeros((), device=device)

#     # i above j: cy_i < cy_j
#     m_above = rel_mask(["above", "over"])
#     if m_above.any():
#         loss_above = F.relu(cy_i - cy_j + margin)[m_above].mean()
#         losses.append(loss_above)
#         counts["rel_geom_reg_above_count"] = m_above.float().sum()
#     else:
#         counts["rel_geom_reg_above_count"] = torch.zeros((), device=device)

#     # i below/under j: cy_i > cy_j
#     m_below = rel_mask(["below", "under", "beneath"])
#     if m_below.any():
#         loss_below = F.relu(cy_j - cy_i + margin)[m_below].mean()
#         losses.append(loss_below)
#         counts["rel_geom_reg_below_count"] = m_below.float().sum()
#     else:
#         counts["rel_geom_reg_below_count"] = torch.zeros((), device=device)

#     # i inside/in/on j: center of i should lie inside j box, with margin
#     m_inside = rel_mask(["in", "inside", "inside of", "on"])
#     if m_inside.any():
#         inside_loss = (
#             F.relu(x1_j - cx_i + margin)
#             + F.relu(cx_i - x2_j + margin)
#             + F.relu(y1_j - cy_i + margin)
#             + F.relu(cy_i - y2_j + margin)
#         )
#         loss_inside = inside_loss[m_inside].mean()
#         losses.append(loss_inside)
#         counts["rel_geom_reg_inside_count"] = m_inside.float().sum()
#     else:
#         counts["rel_geom_reg_inside_count"] = torch.zeros((), device=device)

#     if len(losses) == 0:
#         reg = torch.zeros((), device=device)
#     else:
#         reg = torch.stack(losses).mean()

#     out = {
#         "relation_geometry_reg": reg,
#     }
#     out.update(counts)
#     return out

def compute_relation_geometry_regularizer(
    pred_rel_full: torch.Tensor,      # [B,N,N], full relation ids, 0 = no_rel
    layout_box_pred: torch.Tensor,    # [B,N,4], cxcywh normalized
    node_mask: torch.Tensor,          # [B,N]
    edge_mask: torch.Tensor,          # [B,N,N]
    relation_vocab,
    no_rel_token_id: int = 0,
    margin: float = 0.02,
):
    device = pred_rel_full.device
    rel_full = pred_rel_full.long()

    name_to_id = {name: i for i, name in enumerate(relation_vocab)}

    def ids(names):
        return [name_to_id[x] for x in names if x in name_to_id]

    above_ids = ids(["above", "over"])
    below_ids = ids(["below", "under", "beneath", "underneath"])
    behind_ids = ids(["behind"])
    front_ids = ids(["in front of"])
    inside_ids = ids(["in", "inside", "inside of"])
    on_ids = ids(["on", "on top of", "sitting on", "standing on", "lying on"])


    # print("geom ids:",
    # "above", above_ids,
    # "below", below_ids,
    # "behind", behind_ids,
    # "front", front_ids,
    # "inside", inside_ids,
    # "on", on_ids)
    # print("pred unique rel_full:", torch.unique(pred_rel_full.detach())[:30])

    cx = layout_box_pred[..., 0]
    cy = layout_box_pred[..., 1]
    w = layout_box_pred[..., 2].clamp(min=1e-6)
    h = layout_box_pred[..., 3].clamp(min=1e-6)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    subj_cx = cx.unsqueeze(2)
    subj_cy = cy.unsqueeze(2)
    obj_cx = cx.unsqueeze(1)
    obj_cy = cy.unsqueeze(1)

    subj_x1 = x1.unsqueeze(2)
    subj_y1 = y1.unsqueeze(2)
    subj_x2 = x2.unsqueeze(2)
    subj_y2 = y2.unsqueeze(2)

    obj_x1 = x1.unsqueeze(1)
    obj_y1 = y1.unsqueeze(1)
    obj_x2 = x2.unsqueeze(1)
    obj_y2 = y2.unsqueeze(1)

    valid_pair = (
        node_mask.bool().unsqueeze(2)
        & node_mask.bool().unsqueeze(1)
        & edge_mask.bool()
        & (rel_full != no_rel_token_id)
    )

    B, N, _ = rel_full.shape
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair & (~eye)

    def rel_mask(rel_ids):
        m = torch.zeros_like(rel_full, dtype=torch.bool)
        for rid in rel_ids:
            m |= (rel_full == rid)
        return m & valid_pair

    above_mask = rel_mask(above_ids)
    below_mask = rel_mask(below_ids)
    behind_mask = rel_mask(behind_ids)
    front_mask = rel_mask(front_ids)
    inside_mask = rel_mask(inside_ids)
    on_mask = rel_mask(on_ids)

    losses = []
    counts = {}

    def masked_mean_loss(loss_mat, mask, name):
        count = mask.float().sum()
        counts[name] = count
        if count > 0:
            losses.append((loss_mat * mask.float()).sum() / count)

    # subject above object: subj center should be above object center
    masked_mean_loss(
        F.relu(subj_cy - obj_cy + margin),
        above_mask,
        "above",
    )

    # subject below/under object: subj center should be below object center
    masked_mean_loss(
        F.relu(obj_cy - subj_cy + margin),
        below_mask,
        "below",
    )

    # approximate "behind" as subject higher in image than object
    masked_mean_loss(
        F.relu(subj_cy - obj_cy + margin),
        behind_mask,
        "behind",
    )

    # approximate "in front of" as subject lower in image than object
    masked_mean_loss(
        F.relu(obj_cy - subj_cy + margin),
        front_mask,
        "front",
    )

    # subject inside object: subject box should lie inside object box
    inside_loss = (
        F.relu(obj_x1 - subj_x1 + margin)
        + F.relu(obj_y1 - subj_y1 + margin)
        + F.relu(subj_x2 - obj_x2 + margin)
        + F.relu(subj_y2 - obj_y2 + margin)
    )
    masked_mean_loss(
        inside_loss,
        inside_mask,
        "inside",
    )

    # subject on object: subject should be vertically above object and horizontally near it
    on_vertical = F.relu(subj_cy - obj_cy + margin)
    on_horizontal = F.relu((subj_cx - obj_cx).abs() - 0.5 * (w.unsqueeze(1) + margin))
    on_loss = on_vertical + 0.25 * on_horizontal
    masked_mean_loss(
        on_loss,
        on_mask,
        "on",
    )

    if len(losses) == 0:
        relation_geometry_reg = layout_box_pred.sum() * 0.0
    else:
        relation_geometry_reg = torch.stack(losses).mean()

    return {
        "relation_geometry_reg": relation_geometry_reg,
        "rel_geom_reg_above_count": counts.get("above", torch.tensor(0.0, device=device)),
        "rel_geom_reg_below_count": counts.get("below", torch.tensor(0.0, device=device)),
        "rel_geom_reg_behind_count": counts.get("behind", torch.tensor(0.0, device=device)),
        "rel_geom_reg_front_count": counts.get("front", torch.tensor(0.0, device=device)),
        "rel_geom_reg_inside_count": counts.get("inside", torch.tensor(0.0, device=device)),
        "rel_geom_reg_on_count": counts.get("on", torch.tensor(0.0, device=device)),
    }

# def compute_graph_law_regularizer(
#     edge_logits: torch.Tensor, 
#     rel_logits: torch.Tensor,       # [B,N,N,R_full], includes no-rel at id 0
#     rel_full_target: torch.Tensor,  # [B,N,N], full relation ids, no-rel=0
#     edge_mask: torch.Tensor,        # [B,N,N]
#     no_rel_token_id: int = 0,
#     edge_weight: float = 1.0,
#     degree_weight: float = 0.5,
#     rel_weight: float = 0.5,
#     eps: float = 1e-6,
# ):
#     device = rel_logits.device
#     edge_mask_f = edge_mask.float()

#     rel_probs = torch.softmax(rel_logits, dim=-1)
#     # pred_edge_prob = 1.0 - rel_probs[..., no_rel_token_id]  # [B,N,N]
#     pred_edge_prob = torch.sigmoid(edge_logits)
#     # gt_edge = (rel_full_target != no_rel_token_id).float() * edge_mask_f
#     gt_edge = (rel_full_target != no_rel_token_id).float()
#     pred_edge = pred_edge_prob * edge_mask_f

#     valid_edges_per_graph = edge_mask_f.sum(dim=(1, 2)).clamp(min=1.0)

#     # -------------------------
#     # 1. Edge density law
#     # -------------------------
#     pred_density = pred_edge.sum(dim=(1, 2)) / valid_edges_per_graph
#     gt_density = gt_edge.sum(dim=(1, 2)) / valid_edges_per_graph

#     edge_density_reg = F.smooth_l1_loss(pred_density, gt_density)

#     # -------------------------
#     # 2. Degree / hub law
#     # -------------------------
#     pred_out_deg = pred_edge.sum(dim=2)
#     pred_in_deg = pred_edge.sum(dim=1)
#     gt_out_deg = gt_edge.sum(dim=2)
#     gt_in_deg = gt_edge.sum(dim=1)

#     # Normalize by possible neighbors so this is scale-stable.
#     N = rel_full_target.shape[1]
#     denom = max(float(N - 1), 1.0)

#     pred_deg = (pred_out_deg + pred_in_deg) / denom
#     gt_deg = (gt_out_deg + gt_in_deg) / denom

#     node_valid = (edge_mask_f.sum(dim=2) > 0).float()
#     deg_denom = node_valid.sum().clamp(min=1.0)

#     degree_reg = (
#         F.smooth_l1_loss(pred_deg * node_valid, gt_deg * node_valid, reduction="sum")
#         / deg_denom
#     )

#     # -------------------------
#     # 3. Relation marginal law
#     # -------------------------
#     # Exclude no-rel from relation marginal.
#     rel_probs_pos = rel_probs[..., 1:] * edge_mask_f.unsqueeze(-1)
#     pred_rel_hist = rel_probs_pos.sum(dim=(0, 1, 2))

#     num_rel = rel_logits.shape[-1]
#     gt_rel_onehot = F.one_hot(
#         rel_full_target.clamp(min=0, max=num_rel - 1),
#         num_classes=num_rel,
#     ).float()[..., 1:]

#     gt_rel_hist = (gt_rel_onehot * edge_mask_f.unsqueeze(-1)).sum(dim=(0, 1, 2))

#     pred_rel_dist = pred_rel_hist / pred_rel_hist.sum().clamp(min=eps)
#     gt_rel_dist = gt_rel_hist / gt_rel_hist.sum().clamp(min=eps)

#     rel_marginal_reg = torch.abs(pred_rel_dist - gt_rel_dist).sum()

#     graph_law_reg = (
#         float(edge_weight) * edge_density_reg
#         + float(degree_weight) * degree_reg
#         + float(rel_weight) * rel_marginal_reg
#     )

#     return {
#         "graph_law_reg": graph_law_reg,
#         "graph_edge_density_reg": edge_density_reg.detach(),
#         "graph_degree_reg": degree_reg.detach(),
#         "graph_rel_marginal_reg": rel_marginal_reg.detach(),
#         "pred_edge_density": pred_density.mean().detach(),
#         "gt_edge_density": gt_density.mean().detach(),
#     }

def compute_graph_law_regularizer(
    edge_logits: torch.Tensor,
    rel_logits_pos: torch.Tensor,   # [B,N,N,R_pos], excludes no-rel
    rel_full_target: torch.Tensor,  # [B,N,N], 0=no-rel, positive ids otherwise
    edge_mask: torch.Tensor,
    no_rel_token_id: int = 0,
    edge_weight: float = 1.0,
    degree_weight: float = 0.5,
    rel_weight: float = 0.5,
    eps: float = 1e-6,
):
    edge_mask_f = edge_mask.float()

    pred_edge_prob = torch.sigmoid(edge_logits)
    gt_edge = (rel_full_target != no_rel_token_id).float() * edge_mask_f
    pred_edge = pred_edge_prob * edge_mask_f

    valid_edges_per_graph = edge_mask_f.sum(dim=(1, 2)).clamp(min=1.0)

    pred_density = pred_edge.sum(dim=(1, 2)) / valid_edges_per_graph
    gt_density = gt_edge.sum(dim=(1, 2)) / valid_edges_per_graph
    edge_density_reg = F.smooth_l1_loss(pred_density, gt_density)

    pred_out_deg = pred_edge.sum(dim=2)
    pred_in_deg = pred_edge.sum(dim=1)
    gt_out_deg = gt_edge.sum(dim=2)
    gt_in_deg = gt_edge.sum(dim=1)

    N = rel_full_target.shape[1]
    denom = max(float(N - 1), 1.0)

    pred_deg = (pred_out_deg + pred_in_deg) / denom
    gt_deg = (gt_out_deg + gt_in_deg) / denom

    node_valid = (edge_mask_f.sum(dim=2) > 0).float()
    deg_denom = node_valid.sum().clamp(min=1.0)

    degree_reg = (
        F.smooth_l1_loss(pred_deg * node_valid, gt_deg * node_valid, reduction="sum")
        / deg_denom
    )

    # relation marginal over positive relations only
    rel_probs_pos = torch.softmax(rel_logits_pos, dim=-1)

    # Weight predicted relation histogram by predicted edge probability
    pred_rel_hist = (
        rel_probs_pos * pred_edge.unsqueeze(-1)
    ).sum(dim=(0, 1, 2))

    # Convert full ids: 1..R_pos -> 0..R_pos-1, only for positive GT edges
    R_pos = rel_logits_pos.shape[-1]
    gt_rel_pos = (rel_full_target - 1).clamp(min=0, max=R_pos - 1)

    gt_rel_onehot = F.one_hot(gt_rel_pos, num_classes=R_pos).float()
    gt_rel_hist = (
        gt_rel_onehot * gt_edge.unsqueeze(-1)
    ).sum(dim=(0, 1, 2))

    pred_rel_dist = pred_rel_hist / pred_rel_hist.sum().clamp(min=eps)
    gt_rel_dist = gt_rel_hist / gt_rel_hist.sum().clamp(min=eps)

    rel_marginal_reg = torch.abs(pred_rel_dist - gt_rel_dist).sum()

    graph_law_reg = (
        float(edge_weight) * edge_density_reg
        + float(degree_weight) * degree_reg
        + float(rel_weight) * rel_marginal_reg
    )

    return {
        "graph_law_reg": graph_law_reg,
        "graph_edge_density_reg": edge_density_reg.detach(),
        "graph_degree_reg": degree_reg.detach(),
        "graph_rel_marginal_reg": rel_marginal_reg.detach(),
        "pred_edge_density": pred_density.mean().detach(),
        "gt_edge_density": gt_density.mean().detach(),
    }