# sampling/full_reverse_sampler.py

from typing import Dict, List, Optional, Sequence, Callable
import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_categorical_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [..., K]
    returns: [...]
    """
    orig_shape = probs.shape[:-1]
    K = probs.shape[-1]
    flat = probs.reshape(-1, K)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.reshape(orig_shape)


@torch.no_grad()
def argmax_categorical_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.argmax(dim=-1)


@torch.no_grad()
def build_model_x0_probs(model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Build p_theta(x0 | xt) distributions from model outputs.

    Returns:
      obj_probs:        [B,N,K_obj]
      edge_probs_2:     [B,N,N,2]   state order: 0=no-edge, 1=edge
      rel_probs_ext:    [B,N,N,Kp+1] state order: 0..Kp-1 positive rels, Kp = mask-rel state
    """
    obj_probs = F.softmax(model_out["obj_logits"], dim=-1)                # [B,N,K_obj]

    edge_pos = torch.sigmoid(model_out["edge_logits"])                    # [B,N,N]
    edge_probs_2 = torch.stack([1.0 - edge_pos, edge_pos], dim=-1)       # [B,N,N,2]

    rel_probs = F.softmax(model_out["rel_logits_pos"], dim=-1)           # [B,N,N,Kp]
    B, N, _, Kp = rel_probs.shape

    # extend with zero probability on mask-rel x0 state
    zero_mask_col = torch.zeros(B, N, N, 1, device=rel_probs.device, dtype=rel_probs.dtype)
    rel_probs_ext = torch.cat([rel_probs, zero_mask_col], dim=-1)        # [B,N,N,Kp+1]

    return {
        "obj_probs": obj_probs,
        "edge_probs_2": edge_probs_2,
        "rel_probs_ext": rel_probs_ext,
    }


@torch.no_grad()
def compute_reverse_posterior_from_x0_probs(
    x0_probs: torch.Tensor,          # [M,K]
    current_state: torch.Tensor,     # [M] integer in [0,K-1]
    Q_t: torch.Tensor,               # [K,K], row-stochastic, source->target
    Qbar_prev: torch.Tensor,         # [K,K], row-stochastic, source x0 -> prev state
) -> torch.Tensor:
    """
    Compute posterior over z_{t-1} given current z_t and model p(z0 | zt).

    posterior[a] ∝ Q_t[a,b] * sum_c p(x0=c|xt) * Qbar_prev[c,a]

    Returns:
      posterior: [M,K]
    """
    # predictive_prev[m,a] = sum_c p_x0[m,c] * Qbar_prev[c,a]
    predictive_prev = x0_probs @ Qbar_prev   # [M,K]

    # likelihood[m,a] = Q_t[a, current_state[m]]
    likelihood = Q_t[:, current_state].transpose(0, 1)  # [M,K]

    posterior = predictive_prev * likelihood
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return posterior


@torch.no_grad()
def compute_prev_marginal_from_x0_probs(
    x0_probs: torch.Tensor,          # [M,K]
    Qbar_prev: torch.Tensor,         # [K,K]
) -> torch.Tensor:
    """
    Marginal over z_{t-1} induced only by model p(z0|xt), without conditioning
    on a current observed state. Useful when the current variable is undefined
    (e.g. relation when current edge_t=0).
    """
    prev_probs = x0_probs @ Qbar_prev
    prev_probs = prev_probs / prev_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return prev_probs


@torch.no_grad()
def sample_or_argmax(
    probs: torch.Tensor,
    stochastic: bool,
) -> torch.Tensor:
    if stochastic:
        return sample_categorical_from_probs(probs)
    return argmax_categorical_from_probs(probs)


@torch.no_grad()
def reverse_step_via_discrete_posterior(
    model,
    obj_gen,
    obj_t: torch.Tensor,
    edge_t: torch.Tensor,
    rel_pos_t: torch.Tensor,
    t_cur: int,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    stochastic_obj: bool = True,
    stochastic_edge: bool = True,
    stochastic_rel: bool = True,
):
    """
    Proper posterior-style reverse step:
      xt -> p_theta(x0|xt) -> posterior over x_{t-1}
    """
    device = obj_t.device
    B, N = obj_t.shape

    assert t_cur >= 1, "Posterior reverse step should be used for t_cur >= 1"

    t_tensor = torch.full((B,), int(t_cur), device=device, dtype=torch.long)

    model_out = model(
        obj_t=obj_t,
        edge_t=edge_t,
        rel_pos_t=rel_pos_t,
        t=t_tensor,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

    x0_probs = build_model_x0_probs(model_out)
    kernels_t = obj_gen.get_all_kernels_t(t_tensor)

    # previous cumulative kernels at t-1
    prev_t_tensor = torch.full((B,), int(t_cur - 1), device=device, dtype=torch.long)
    kernels_prev = obj_gen.get_all_kernels_t(prev_t_tensor)

    # ------------------------------------------------------------------
    # OBJECTS
    # ------------------------------------------------------------------
    obj_prev = obj_t.clone()

    for b in range(B):
        valid_nodes = node_mask[b].bool()
        idx = torch.where(valid_nodes)[0]
        if idx.numel() == 0:
            continue

        obj_probs_b = x0_probs["obj_probs"][b, idx]              # [M,K_obj]
        obj_cur_b = obj_t[b, idx]                                # [M]
        Q_obj_t_b = kernels_t["Q_obj_t"][b]                      # [K,K]
        Qbar_obj_prev_b = kernels_prev["Qbar_obj_t"][b]          # [K,K]

        obj_post = compute_reverse_posterior_from_x0_probs(
            x0_probs=obj_probs_b,
            current_state=obj_cur_b,
            Q_t=Q_obj_t_b,
            Qbar_prev=Qbar_obj_prev_b,
        )                                                        # [M,K]

        obj_prev[b, idx] = sample_or_argmax(obj_post, stochastic_obj)

    # ------------------------------------------------------------------
    # EDGES
    # ------------------------------------------------------------------
    edge_prev = edge_t.clone()
    rel_prev = rel_pos_t.clone()

    valid_pairs = edge_mask.bool()

    for b in range(B):
        pair_idx = torch.where(valid_pairs[b])
        if pair_idx[0].numel() == 0:
            continue

        i_idx, j_idx = pair_idx
        M = i_idx.numel()

        edge_probs_b = x0_probs["edge_probs_2"][b, i_idx, j_idx]         # [M,2]
        edge_cur_b = edge_t[b, i_idx, j_idx]                              # [M]
        Q_edge_t_b = kernels_t["Q_edge_t"][b]                             # [2,2]
        Qbar_edge_prev_b = kernels_prev["Qbar_edge_t"][b]                 # [2,2]

        edge_post = compute_reverse_posterior_from_x0_probs(
            x0_probs=edge_probs_b,
            current_state=edge_cur_b,
            Q_t=Q_edge_t_b,
            Qbar_prev=Qbar_edge_prev_b,
        )                                                                 # [M,2]

        edge_prev_b = sample_or_argmax(edge_post, stochastic_edge)        # [M]
        edge_prev[b, i_idx, j_idx] = edge_prev_b

        # ------------------------------------------------------------------
        # RELATIONS (conditional on sampled previous edge)
        # State space here is:
        #   0..Kp-1 = positive relation classes
        #   Kp      = mask relation token
        # If sampled prev edge = 0, relation state is irrelevant and we set 0.
        # ------------------------------------------------------------------
        rel_probs_b = x0_probs["rel_probs_ext"][b, i_idx, j_idx]          # [M,Kp+1]
        rel_cur_b = rel_pos_t[b, i_idx, j_idx]                            # [M]
        Q_rel_t_b = kernels_t["Q_rel_t"][b]                               # [Ks,Ks]
        Qbar_rel_prev_b = kernels_prev["Qbar_rel_t"][b]                   # [Ks,Ks]

        # whether relation is actually observed at current step
        cur_edge_exists_b = edge_t[b, i_idx, j_idx].bool()
        prev_edge_exists_b = edge_prev_b.bool()

        rel_prev_b = rel_prev[b, i_idx, j_idx].clone()

        # Case A: current edge exists and previous sampled edge exists
        mask_full_post = cur_edge_exists_b & prev_edge_exists_b
        if mask_full_post.any():
            rel_post = compute_reverse_posterior_from_x0_probs(
                x0_probs=rel_probs_b[mask_full_post],
                current_state=rel_cur_b[mask_full_post],
                Q_t=Q_rel_t_b,
                Qbar_prev=Qbar_rel_prev_b,
            )
            rel_prev_b[mask_full_post] = sample_or_argmax(rel_post, stochastic_rel)

        # Case B: current edge absent but previous sampled edge exists
        # current relation state is undefined; use predictive marginal only
        mask_no_current_obs = (~cur_edge_exists_b) & prev_edge_exists_b
        if mask_no_current_obs.any():
            rel_prev_probs = compute_prev_marginal_from_x0_probs(
                x0_probs=rel_probs_b[mask_no_current_obs],
                Qbar_prev=Qbar_rel_prev_b,
            )
            rel_prev_b[mask_no_current_obs] = sample_or_argmax(rel_prev_probs, stochastic_rel)

        # Case C: previous sampled edge absent -> relation irrelevant
        mask_prev_no_edge = ~prev_edge_exists_b
        if mask_prev_no_edge.any():
            rel_prev_b[mask_prev_no_edge] = 0

        rel_prev[b, i_idx, j_idx] = rel_prev_b

    # enforce invalid pairs to zero
    edge_prev = edge_prev * edge_mask.long()
    rel_prev = torch.where(edge_prev.bool(), rel_prev, torch.zeros_like(rel_prev))

    return {
        "obj_t": obj_prev.long(),
        "edge_t": edge_prev.long(),
        "rel_pos_t": rel_prev.long(),
        "model_out": model_out,
        "x0_probs": x0_probs,
    }

@torch.no_grad()
def reverse_step_via_reverse_vocab_heads(
    model,
    obj_t: torch.Tensor,
    edge_t: torch.Tensor,
    rel_pos_t: torch.Tensor,
    t_cur: int,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    stochastic_obj: bool = False,
    stochastic_edge: bool = False,
    stochastic_rel: bool = False,
    obj_temp: float = 1.0,
    rel_temp: float = 1.0,
    edge_logit_threshold: float = 0.0,
    relation_edge_logit_threshold: float = 0.0,
    use_degree_pruning: bool = False,
    max_out_degree: int = 0,
    max_in_degree: int = 0,
    use_final_step_cleanup: bool = False,
    final_edge_logit_threshold: float = 0.5,
    final_rel_conf_threshold: float = 0.0,
    generic_obj_ids: Optional[Sequence[int]] = None,
    generic_attachment_rel_ids: Optional[Sequence[int]] = None,
    generic_attachment_edge_logit_threshold: float = 1.0,
    reward_fn: Optional[Callable] = None,
    use_reward_tilting: bool = False,
    reward_tilt_alpha: float = 1.0,
    reward_tilt_temperature: float = 1.0,
    reward_tilt_num_sweeps: int = 1,
    reward_tilt_objects: bool = False,
    reward_tilt_edges: bool = False,
    reward_tilt_relations: bool = False,
    reward_tilt_use_layout: bool = False,
    reward_tilt_obj_topk: int = 5,
    reward_tilt_rel_topk: int = 5,
    reward_weights: Optional[dict] = None,
    reward_tilt_edge_logit_band: float = 0.75,
    reward_w_hub_degree: float = 0.50,
    reward_hub_degree_threshold: int = 4,
    reward_relation_group_pos_ids: Optional[dict] = None,
    reward_tilt_relation_alpha: float = 0.5,
    reward_w_relation_geometry_tilt: float = 1.0,
    reward_obj_log_prior: Optional[torch.Tensor] = None,
    reward_tilt_object_alpha: float = 0.25,
    reward_w_object_class_prior_tilt: float = 0.50,
    reward_w_object_relation_support_tilt: float = 0.25,
    reward_tilt_obj_logit_margin: float = 1.0,
    reward_tilt_layout_alpha: float = 0.25,
    reward_w_layout_overlap_tilt: float = 1.0,
    reward_w_layout_spread_tilt: float = 0.5,
    reward_w_box_bounds_tilt: float = 0.5,
):
    
    device = obj_t.device
    B, N = obj_t.shape

    base_model = model.module if hasattr(model, "module") else model

    t_tensor = torch.full((B,), int(t_cur), device=device, dtype=torch.long)

    model_out = model(
        obj_t=obj_t,
        edge_t=edge_t,
        rel_pos_t=rel_pos_t,
        t=t_tensor,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

    if "obj_rev_logits" not in model_out:
        raise RuntimeError("Model output does not contain obj_rev_logits. Did you enable use_reverse_vocab_heads?")

    obj_rev_logits = model_out["obj_rev_logits"]
    edge_rev_logits = model_out["edge_rev_logits"]
    rel_rev_logits = model_out["rel_rev_logits_pos"]

    num_obj_clean = base_model.num_obj_classes
    num_rel_clean = base_model.num_rel_pos_classes

    # --------------------------------------------------
    # Edge update
    # --------------------------------------------------
    edge_prev = sample_binary_from_logits(
        edge_rev_logits,
        stochastic=stochastic_edge,
        threshold=edge_logit_threshold,
    )
    edge_prev = edge_prev * edge_mask.long()

    # Optional Phase 6A.2: degree-aware pruning
    if use_degree_pruning:
        edge_keep_mask = edge_prev.bool() & edge_mask.bool()

        # score relative to the decision threshold, so positive means stronger keep
        edge_conf_scores = edge_rev_logits - float(edge_logit_threshold)

        pruned_keep_mask = prune_edges_by_degree(
            edge_keep_mask=edge_keep_mask,
            edge_scores=edge_conf_scores,
            edge_mask=edge_mask,
            max_out_degree=max_out_degree,
            max_in_degree=max_in_degree,
        )

        edge_prev = pruned_keep_mask.long()

        ## debug
        # if use_degree_pruning:
        #     out_deg = edge_prev.sum(dim=-1).max().item()
        #     in_deg = edge_prev.sum(dim=-2).max().item()
        #     print("max out degree after pruning:", out_deg)
        #     print("max in degree after pruning:", in_deg)

    # --------------------------------------------------
    # Object update
    # --------------------------------------------------
    obj_prev = obj_t.clone()

    valid_nodes = node_mask.bool()
    if t_cur == 1:
        # final x0 must be clean, so exclude MASK_OBJ class
        obj_logits_use = obj_rev_logits[..., :num_obj_clean]
    else:
        obj_logits_use = obj_rev_logits

    if valid_nodes.any():
        obj_prev_valid = sample_categorical_from_logits(
            obj_logits_use[valid_nodes],
            stochastic=stochastic_obj,
            temperature=obj_temp,
        )
        obj_prev[valid_nodes] = obj_prev_valid.long()

    # --------------------------------------------------
    # Relation-confidence gating
    # --------------------------------------------------
    # Only edges with stronger confidence are allowed to keep a relation.
    relation_keep_mask = (
        edge_prev.bool()
        & edge_mask.bool()
        & (edge_rev_logits > relation_edge_logit_threshold)
    )

    # Since your representation expects positive edges to carry relation labels,
    # suppress weak positive edges that do not pass the relation-confidence gate.
    edge_prev = torch.where(
        relation_keep_mask,
        edge_prev,
        torch.zeros_like(edge_prev),
    )

    # --------------------------------------------------
    # Relation update
    # --------------------------------------------------
    rel_prev = torch.zeros_like(rel_pos_t)

    if t_cur == 1:
        # final x0 must be clean, so exclude MASK_REL class
        rel_logits_use = rel_rev_logits[..., :num_rel_clean]
    else:
        rel_logits_use = rel_rev_logits

    if relation_keep_mask.any():
        rel_prev_valid = sample_categorical_from_logits(
            rel_logits_use[relation_keep_mask],
            stochastic=stochastic_rel,
            temperature=rel_temp,
        )
        rel_prev[relation_keep_mask] = rel_prev_valid.long()

    rel_prev = torch.where(edge_prev.bool(), rel_prev, torch.zeros_like(rel_prev))

    # --------------------------------------------------
    # 8A.3 Reward-tilted local Gibbs refinement
    # --------------------------------------------------
    if use_reward_tilting and reward_tilt_edges:
        edge_prev, rel_prev = reward_tilt_edges_local_delta_step(
            edge_prev=edge_prev,
            rel_prev=rel_prev,
            edge_logits_use=edge_rev_logits,
            rel_logits_use=rel_logits_use,
            node_mask=node_mask,
            edge_mask=edge_mask,
            alpha=reward_tilt_alpha,
            temperature=reward_tilt_temperature,
            stochastic_edge=stochastic_edge,
            w_isolated_node=reward_weights.get("reward_isolated_node", 0.25) if reward_weights is not None else 0.25,
            w_dense_graph=reward_weights.get("reward_dense_graph", 0.10) if reward_weights is not None else 0.10,
            w_bidirectional_edge=reward_weights.get("reward_bidirectional_edge", 0.10) if reward_weights is not None else 0.10,
            edge_logit_band=reward_tilt_edge_logit_band,
            w_hub_degree=reward_w_hub_degree,
            hub_degree_threshold=reward_hub_degree_threshold,
        )
    
    # --------------------------------------------------
    # 8A.3b Relation-only reward tilt
    # --------------------------------------------------
    if (
        use_reward_tilting
        and reward_tilt_relations
        and reward_relation_group_pos_ids is not None
    ):
        layout_box_pred = None

        # Your model likely returns one of these keys depending on implementation.
        if "layout_box_pred" in model_out:
            layout_box_pred = model_out["layout_box_pred"]
        elif "boxes_pred" in model_out:
            layout_box_pred = model_out["boxes_pred"]
        elif "layout_boxes" in model_out:
            layout_box_pred = model_out["layout_boxes"]

        if layout_box_pred is not None:
            rel_prev = reward_tilt_relations_local_geometry_step(
                rel_prev=rel_prev,
                edge_prev=edge_prev,
                rel_logits_use=rel_logits_use,
                layout_box_pred=layout_box_pred,
                node_mask=node_mask,
                edge_mask=edge_mask,
                relation_group_pos_ids=reward_relation_group_pos_ids,
                alpha=reward_tilt_relation_alpha,
                temperature=reward_tilt_temperature,
                stochastic_rel=stochastic_rel,
                rel_topk=reward_tilt_rel_topk,
                w_relation_geometry=reward_w_relation_geometry_tilt,
            )

        rel_prev = torch.where(edge_prev.bool(), rel_prev, torch.zeros_like(rel_prev))
    
    # --------------------------------------------------
    # 8A.3c Object-only reward tilt
    # --------------------------------------------------
    if (
        use_reward_tilting
        and reward_tilt_objects
        and reward_obj_log_prior is not None
    ):
        obj_prev = reward_tilt_objects_local_prior_step(
            obj_prev=obj_prev,
            obj_rev_logits=obj_logits_use,
            node_mask=node_mask,
            edge_prev=edge_prev,
            rel_prev=rel_pos_t,
            obj_log_prior=reward_obj_log_prior,
            alpha=reward_tilt_object_alpha,
            temperature=reward_tilt_temperature,
            stochastic_obj=stochastic_obj,
            obj_topk=reward_tilt_obj_topk,
            obj_logit_margin=reward_tilt_obj_logit_margin,
            w_object_class_prior=reward_w_object_class_prior_tilt,
            w_object_relation_support=reward_w_object_relation_support_tilt,
        )
    
    # --------------------------------------------------
    # 8A.4 Layout-induced geometric safeguard
    # --------------------------------------------------
    if (
        use_reward_tilting
        and reward_tilt_use_layout
        and reward_tilt_layout_alpha > 0.0
    ):
        layout_box_pred = None

        if "layout_box_pred" in model_out:
            layout_box_pred = model_out["layout_box_pred"]
        elif "boxes_pred" in model_out:
            layout_box_pred = model_out["boxes_pred"]
        elif "layout_boxes" in model_out:
            layout_box_pred = model_out["layout_boxes"]

        if layout_box_pred is not None:
            layout_reward = compute_layout_global_reward(
                layout_box_pred=layout_box_pred,
                node_mask=node_mask,
                w_overlap=reward_w_layout_overlap_tilt,
                w_spread=reward_w_layout_spread_tilt,
                w_bounds=reward_w_box_bounds_tilt,
            )

            # Store for logging/debugging.
            model_out["reward_layout_global"] = layout_reward.detach()
    # --------------------------------------------------
    # 6A.3 Final-step cleanup
    # --------------------------------------------------
    if use_final_step_cleanup and t_cur == 1:
        edge_prev, rel_prev = apply_final_step_cleanup(
            obj_prev=obj_prev,
            edge_prev=edge_prev,
            rel_prev=rel_prev,
            edge_rev_logits=edge_rev_logits,
            rel_logits_use=rel_logits_use,
            edge_mask=edge_mask,
            final_edge_logit_threshold=final_edge_logit_threshold,
            final_rel_conf_threshold=final_rel_conf_threshold,
            generic_obj_ids=generic_obj_ids,
            generic_attachment_rel_ids=generic_attachment_rel_ids,
            generic_attachment_edge_logit_threshold=generic_attachment_edge_logit_threshold,
        )

    # print("edge survive count:", edge_prev.sum().item())
    # print("relation keep count:", relation_keep_mask.sum().item())

    return {
        "obj_t": obj_prev.long(),
        "edge_t": edge_prev.long(),
        "rel_pos_t": rel_prev.long(),
        "model_out": model_out,
    }

@torch.no_grad()
def run_full_reverse_chain(
    model,
    obj_gen,
    batch_clean,
    T,
    edge_exist_thres=0.5,
    stochastic_obj=False,
    stochastic_edge=False,
    stochastic_rel=False,
    return_trace=False,
    use_reverse_vocab_heads=False,
    obj_temp=1.0,
    rel_temp=1.0,
    edge_logit_threshold: float = 0.0,
    relation_edge_logit_threshold: float = 0.0,
    use_degree_pruning: bool = False,
    max_out_degree: int = 0,
    max_in_degree: int = 0,
    use_final_step_cleanup: bool = False,
    final_edge_logit_threshold: float = 0.5,
    final_rel_conf_threshold: float = 0.0,
    generic_obj_ids: Optional[Sequence[int]] = None,
    generic_attachment_rel_ids: Optional[Sequence[int]] = None,
    generic_attachment_edge_logit_threshold: float = 1.0,
    reward_fn: Optional[Callable] = None,
    use_reward_tilting: bool = False,
    reward_tilt_alpha: float = 1.0,
    reward_tilt_temperature: float = 1.0,
    reward_tilt_num_sweeps: int = 1,
    reward_tilt_objects: bool = False,
    reward_tilt_edges: bool = False,
    reward_tilt_relations: bool = False,
    reward_tilt_use_layout: bool = False,
    reward_tilt_obj_topk: int = 5,
    reward_tilt_rel_topk: int = 5,
    reward_weights: Optional[dict] = None,
    reward_tilt_edge_logit_band: float = 0.75,
    reward_w_hub_degree: float = 0.50,
    reward_hub_degree_threshold: int = 4,
    reward_relation_group_pos_ids: Optional[dict] = None,
    reward_tilt_relation_alpha: float = 0.5,
    reward_w_relation_geometry_tilt: float = 1.0,
    reward_obj_log_prior: Optional[torch.Tensor] = None,
    reward_tilt_object_alpha: float = 0.25,
    reward_w_object_class_prior_tilt: float = 0.50,
    reward_w_object_relation_support_tilt: float = 0.25,
    reward_tilt_obj_logit_margin: float = 1.0,
    reward_tilt_layout_alpha: float = 0.25,
    reward_w_layout_overlap_tilt: float = 1.0,
    reward_w_layout_spread_tilt: float = 0.5,
    reward_w_box_bounds_tilt: float = 0.5,
):
    """
    Conditional reverse reconstruction:
      start from q(x_T | x0_clean)
      run posterior-based reverse chain down to x_0
    """
    device = batch_clean["obj_0"].device
    B = batch_clean["obj_0"].shape[0]

    t_start = torch.full((B,), int(T), device=device, dtype=torch.long)

    start_state = obj_gen.sample_graph_at_t_from_clean(
        batch_clean=batch_clean,
        t=t_start,
    )
    
    layout_global_rewards = []

    cur_obj = start_state["obj_t"]
    cur_edge = start_state["edge_t"]
    cur_rel_pos = start_state["rel_pos_t"]

    # print("using reverse vocab heads sampler:", use_reverse_vocab_heads)
    # print("obj unique max:", cur_obj.max().item())
    # print("edge unique:", cur_edge.unique())
    # print("rel unique max:", cur_rel_pos.max().item())

    trace: List[Dict[str, torch.Tensor]] = []

    # reverse from T down to 1; output is state at 0
    for t_cur in range(T, 0, -1):
        if use_reverse_vocab_heads:
            step_out = reverse_step_via_reverse_vocab_heads(
                model=model,
                obj_t=cur_obj,
                edge_t=cur_edge,
                rel_pos_t=cur_rel_pos,
                t_cur=t_cur,
                node_mask=batch_clean["node_mask"],
                edge_mask=batch_clean["edge_mask"],
                stochastic_obj=stochastic_obj,
                stochastic_edge=stochastic_edge,
                stochastic_rel=stochastic_rel,
                obj_temp=obj_temp,
                rel_temp=rel_temp,
                edge_logit_threshold=edge_logit_threshold,
                relation_edge_logit_threshold=relation_edge_logit_threshold,
                use_degree_pruning=use_degree_pruning,
                max_out_degree=max_out_degree,
                max_in_degree=max_in_degree,
                use_final_step_cleanup=use_final_step_cleanup,
                final_edge_logit_threshold=final_edge_logit_threshold,
                final_rel_conf_threshold=final_rel_conf_threshold,
                generic_obj_ids=generic_obj_ids,
                generic_attachment_rel_ids=generic_attachment_rel_ids,
                generic_attachment_edge_logit_threshold=generic_attachment_edge_logit_threshold,
                reward_fn=reward_fn,
                use_reward_tilting=use_reward_tilting,
                reward_tilt_alpha=reward_tilt_alpha,
                reward_tilt_temperature=reward_tilt_temperature,
                reward_tilt_num_sweeps=reward_tilt_num_sweeps,
                reward_tilt_objects=reward_tilt_objects,
                reward_tilt_edges=reward_tilt_edges,
                reward_tilt_relations=reward_tilt_relations,
                reward_tilt_use_layout=reward_tilt_use_layout,
                reward_tilt_obj_topk=reward_tilt_obj_topk,
                reward_tilt_rel_topk=reward_tilt_rel_topk,
                reward_weights=reward_weights,
                reward_tilt_edge_logit_band=reward_tilt_edge_logit_band,
                reward_w_hub_degree=reward_w_hub_degree,
                reward_hub_degree_threshold=reward_hub_degree_threshold,
                reward_relation_group_pos_ids=reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=reward_tilt_relation_alpha,
                reward_w_relation_geometry_tilt=reward_w_relation_geometry_tilt,
                reward_obj_log_prior=reward_obj_log_prior,
                reward_tilt_object_alpha=reward_tilt_object_alpha,
                reward_w_object_class_prior_tilt=reward_w_object_class_prior_tilt,
                reward_w_object_relation_support_tilt=reward_w_object_relation_support_tilt,
                reward_tilt_obj_logit_margin=reward_tilt_obj_logit_margin,
                reward_tilt_layout_alpha=reward_tilt_layout_alpha,
                reward_w_layout_overlap_tilt=reward_w_layout_overlap_tilt,
                reward_w_layout_spread_tilt=reward_w_layout_spread_tilt,
                reward_w_box_bounds_tilt=reward_w_box_bounds_tilt,
            )
            if "model_out" in step_out and "reward_layout_global" in step_out["model_out"]:
                layout_global_rewards.append(step_out["model_out"]["reward_layout_global"].detach().cpu())
        else:
            step_out = reverse_step_via_discrete_posterior(
                model=model,
                obj_gen=obj_gen,
                obj_t=cur_obj,
                edge_t=cur_edge,
                rel_pos_t=cur_rel_pos,
                t_cur=t_cur,
                node_mask=batch_clean["node_mask"],
                edge_mask=batch_clean["edge_mask"],
                stochastic_obj=stochastic_obj,
                stochastic_edge=stochastic_edge,
                stochastic_rel=stochastic_rel,
            )
        
        
        # print("next obj max:", step_out["obj_t"].max().item())
        # print("next rel max:", step_out["rel_pos_t"].max().item())

        cur_obj = step_out["obj_t"]
        cur_edge = step_out["edge_t"]
        cur_rel_pos = step_out["rel_pos_t"]

        if return_trace:
            trace.append({
                "t": torch.full((B,), t_cur - 1, device=device, dtype=torch.long),
                "obj_t": cur_obj.clone(),
                "edge_t": cur_edge.clone(),
                "rel_pos_t": cur_rel_pos.clone(),
            })

    # --------------------------------------------------
    # Final layout readout from sampled x0 graph
    # --------------------------------------------------
    final_model_out = model(
        obj_t=cur_obj,
        edge_t=cur_edge,
        rel_pos_t=cur_rel_pos,
        t=torch.zeros((B,), device=device, dtype=torch.long),
        node_mask=batch_clean["node_mask"],
        edge_mask=batch_clean["edge_mask"],
    )

    layout_box_final = final_model_out.get("layout_box_pred", None)

    out = {
        "obj_final": cur_obj,
        "edge_final": cur_edge,
        "rel_pos_final": cur_rel_pos,
        "layout_box_final": layout_box_final.detach() if layout_box_final is not None else None,
        "start_state": start_state,
    }

    if return_trace:
        out["trace"] = trace
    
    if len(layout_global_rewards) > 0:
        out["layout_global_reward"] = torch.stack(layout_global_rewards, dim=0).mean(dim=0)

    return out

@torch.no_grad()
def build_unconditional_start_state(
    obj_gen,
    node_mask: torch.Tensor,   # [B,N]
    edge_mask: torch.Tensor,   # [B,N,N]
    T: int,
):
    """
    Build a terminal noisy state x_T without conditioning on any clean graph.

    Phase 6D.1b:
    If empirical unconditional priors are available on obj_gen, use them as the
    clean x0 prior and diffuse them to time T:
        p(x_T) = p_data(x0) @ Qbar_T

    Otherwise fall back to the old row-averaged kernel prior.
    """
    device = node_mask.device
    B, N = node_mask.shape

    t_tensor = torch.full((B,), int(T), device=device, dtype=torch.long)
    kernels_T = obj_gen.get_all_kernels_t(t_tensor)

    num_obj_classes = obj_gen.num_obj_classes
    num_rel_pos_classes = obj_gen.num_rel_pos_classes
    rel_state_size = num_rel_pos_classes + 1

    obj_t = torch.zeros((B, N), dtype=torch.long, device=device)
    edge_t = torch.zeros((B, N, N), dtype=torch.long, device=device)
    rel_pos_t = torch.zeros((B, N, N), dtype=torch.long, device=device)

    use_empirical = (
        getattr(obj_gen, "obj_empirical_prior", None) is not None
        and getattr(obj_gen, "edge_empirical_prior", None) is not None
        and getattr(obj_gen, "rel_empirical_prior", None) is not None
    )

    for b in range(B):
        valid_nodes = node_mask[b].bool()
        valid_pairs = edge_mask[b].bool()

        # --------------------------------------------------
        # Objects
        # --------------------------------------------------
        Qbar_obj_T = kernels_T["Qbar_obj_t"][b]  # [Kobj, Kobj]
        if use_empirical:
            obj_terminal = obj_gen.obj_empirical_prior @ Qbar_obj_T
        else:
            obj_terminal = Qbar_obj_T.mean(dim=0)

        obj_terminal = obj_terminal / obj_terminal.sum().clamp(min=1e-12)

        if valid_nodes.any():
            obj_samples = sample_categorical_from_probs(
                obj_terminal.unsqueeze(0).expand(valid_nodes.sum(), -1)
            )
            obj_t[b, valid_nodes] = obj_samples.long()

        # --------------------------------------------------
        # Edges
        # --------------------------------------------------
        Qbar_edge_T = kernels_T["Qbar_edge_t"][b]  # [2,2]
        if use_empirical:
            edge_terminal = obj_gen.edge_empirical_prior @ Qbar_edge_T
        else:
            edge_terminal = Qbar_edge_T.mean(dim=0)

        edge_terminal = edge_terminal / edge_terminal.sum().clamp(min=1e-12)

        if valid_pairs.any():
            edge_samples = sample_categorical_from_probs(
                edge_terminal.unsqueeze(0).expand(valid_pairs.sum(), -1)
            )
            edge_t[b][valid_pairs] = edge_samples.long()

        # --------------------------------------------------
        # Relations
        # --------------------------------------------------
        Qbar_rel_T = kernels_T["Qbar_rel_t"][b]   # [Krel_state, Krel_state]
        if use_empirical:
            rel_terminal = obj_gen.rel_empirical_prior @ Qbar_rel_T
        else:
            rel_terminal = Qbar_rel_T[:num_rel_pos_classes].mean(dim=0)

        rel_terminal = rel_terminal / rel_terminal.sum().clamp(min=1e-12)

        # For active sampled edges, prefer positive relation classes over mask-rel
        rel_terminal_active = rel_terminal.clone()
        rel_terminal_active[-1] = 0.0
        rel_terminal_active = rel_terminal_active / rel_terminal_active.sum().clamp(min=1e-12)

        active_pairs = valid_pairs & edge_t[b].bool()
        if active_pairs.any():
            rel_samples = sample_categorical_from_probs(
                rel_terminal_active.unsqueeze(0).expand(active_pairs.sum(), -1)
            )
            rel_pos_t[b][active_pairs] = rel_samples.long()

    edge_t = edge_t * edge_mask.long()
    rel_pos_t = torch.where(edge_t.bool(), rel_pos_t, torch.zeros_like(rel_pos_t))

    return {
        "obj_t": obj_t.long(),
        "edge_t": edge_t.long(),
        "rel_pos_t": rel_pos_t.long(),
    }


@torch.no_grad()
def run_full_reverse_chain_unconditional(
    model,
    obj_gen,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    T: int,
    stochastic_obj: bool = True,
    stochastic_edge: bool = True,
    stochastic_rel: bool = True,
    return_trace: bool = False,
    use_reverse_vocab_heads: bool = False,
    obj_temp: float = 1.0,
    rel_temp: float = 1.0,
    edge_logit_threshold: float = 0.0,
    relation_edge_logit_threshold: float = 0.0,
    use_degree_pruning: bool = False,
    max_out_degree: int = 0,
    max_in_degree: int = 0,
    use_final_step_cleanup: bool = False,
    final_edge_logit_threshold: float = 0.5,
    final_rel_conf_threshold: float = 0.0,
    generic_obj_ids: Optional[Sequence[int]] = None,
    generic_attachment_rel_ids: Optional[Sequence[int]] = None,
    generic_attachment_edge_logit_threshold: float = 1.0,
    reward_fn: Optional[Callable] = None,
    use_reward_tilting: bool = False,
    reward_tilt_alpha: float = 1.0,
    reward_tilt_temperature: float = 1.0,
    reward_tilt_num_sweeps: int = 1,
    reward_tilt_objects: bool = False,
    reward_tilt_edges: bool = False,
    reward_tilt_relations: bool = False,
    reward_tilt_use_layout: bool = False,
    reward_tilt_obj_topk: int = 5,
    reward_tilt_rel_topk: int = 5,
    reward_weights: Optional[dict] = None,
    reward_tilt_edge_logit_band: float = 0.75,
    reward_w_hub_degree: float = 0.50,
    reward_hub_degree_threshold: int = 4,
    reward_relation_group_pos_ids: Optional[dict] = None,
    reward_tilt_relation_alpha: float = 0.5,
    reward_w_relation_geometry_tilt: float = 1.0,
    reward_obj_log_prior: Optional[torch.Tensor] = None,
    reward_tilt_object_alpha: float = 0.25,
    reward_w_object_class_prior_tilt: float = 0.50,
    reward_w_object_relation_support_tilt: float = 0.25,
    reward_tilt_obj_logit_margin: float = 1.0,
    reward_tilt_layout_alpha: float = 0.25,
    reward_w_layout_overlap_tilt: float = 1.0,
    reward_w_layout_spread_tilt: float = 0.5,
    reward_w_box_bounds_tilt: float = 0.5,
):
    """
    Unconditional reverse generation:
      start from a terminal noisy state x_T
      run posterior-based reverse chain down to x_0
    """
    device = node_mask.device
    B = node_mask.shape[0]

    start_state = build_unconditional_start_state(
        obj_gen=obj_gen,
        node_mask=node_mask,
        edge_mask=edge_mask,
        T=T,
    )

    cur_obj = start_state["obj_t"]
    cur_edge = start_state["edge_t"]
    cur_rel_pos = start_state["rel_pos_t"]

    trace: List[Dict[str, torch.Tensor]] = []

    for t_cur in range(T, 0, -1):
        if use_reverse_vocab_heads:
            step_out = reverse_step_via_reverse_vocab_heads(
                model=model,
                obj_t=cur_obj,
                edge_t=cur_edge,
                rel_pos_t=cur_rel_pos,
                t_cur=t_cur,
                node_mask=node_mask,
                edge_mask=edge_mask,
                stochastic_obj=stochastic_obj,
                stochastic_edge=stochastic_edge,
                stochastic_rel=stochastic_rel,
                obj_temp=obj_temp,
                rel_temp=rel_temp,
                edge_logit_threshold=edge_logit_threshold,
                relation_edge_logit_threshold=relation_edge_logit_threshold,
                use_degree_pruning=use_degree_pruning,
                max_out_degree=max_out_degree,
                max_in_degree=max_in_degree,
                use_final_step_cleanup=use_final_step_cleanup,
                final_edge_logit_threshold=final_edge_logit_threshold,
                final_rel_conf_threshold=final_rel_conf_threshold,
                generic_obj_ids=generic_obj_ids,
                generic_attachment_rel_ids=generic_attachment_rel_ids,
                generic_attachment_edge_logit_threshold=generic_attachment_edge_logit_threshold,
                reward_fn=reward_fn,
                use_reward_tilting=use_reward_tilting,
                reward_tilt_alpha=reward_tilt_alpha,
                reward_tilt_temperature=reward_tilt_temperature,
                reward_tilt_num_sweeps=reward_tilt_num_sweeps,
                reward_tilt_objects=reward_tilt_objects,
                reward_tilt_edges=reward_tilt_edges,
                reward_tilt_relations=reward_tilt_relations,
                reward_tilt_use_layout=reward_tilt_use_layout,
                reward_tilt_obj_topk=reward_tilt_obj_topk,
                reward_tilt_rel_topk=reward_tilt_rel_topk,
                reward_weights=reward_weights,
                reward_tilt_edge_logit_band=reward_tilt_edge_logit_band,
                reward_w_hub_degree=reward_w_hub_degree,
                reward_hub_degree_threshold=reward_hub_degree_threshold,
                reward_relation_group_pos_ids=reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=reward_tilt_relation_alpha,
                reward_w_relation_geometry_tilt=reward_w_relation_geometry_tilt,
                reward_obj_log_prior=reward_obj_log_prior,
                reward_tilt_object_alpha=reward_tilt_object_alpha,
                reward_w_object_class_prior_tilt=reward_w_object_class_prior_tilt,
                reward_w_object_relation_support_tilt=reward_w_object_relation_support_tilt,
                reward_tilt_obj_logit_margin=reward_tilt_obj_logit_margin,
                reward_tilt_layout_alpha=reward_tilt_layout_alpha,
                reward_w_layout_overlap_tilt=reward_w_layout_overlap_tilt,
                reward_w_layout_spread_tilt=reward_w_layout_spread_tilt,
                reward_w_box_bounds_tilt=reward_w_box_bounds_tilt,
            )
        else:
            step_out = reverse_step_via_discrete_posterior(
                model=model,
                obj_gen=obj_gen,
                obj_t=cur_obj,
                edge_t=cur_edge,
                rel_pos_t=cur_rel_pos,
                t_cur=t_cur,
                node_mask=node_mask,
                edge_mask=edge_mask,
                stochastic_obj=stochastic_obj,
                stochastic_edge=stochastic_edge,
                stochastic_rel=stochastic_rel,
            )

        cur_obj = step_out["obj_t"]
        cur_edge = step_out["edge_t"]
        cur_rel_pos = step_out["rel_pos_t"]

        if return_trace:
            trace.append({
                "t": torch.full((B,), t_cur - 1, device=device, dtype=torch.long),
                "obj_t": cur_obj.clone(),
                "edge_t": cur_edge.clone(),
                "rel_pos_t": cur_rel_pos.clone(),
            })

    # --------------------------------------------------
    # Final layout readout from sampled x0 graph
    # --------------------------------------------------
    final_model_out = model(
        obj_t=cur_obj,
        edge_t=cur_edge,
        rel_pos_t=cur_rel_pos,
        t=torch.zeros((B,), device=device, dtype=torch.long),
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

    layout_box_final = final_model_out.get("layout_box_pred", None)

    out = {
        "obj_final": cur_obj,
        "edge_final": cur_edge,
        "rel_pos_final": cur_rel_pos,
        "layout_box_final": layout_box_final.detach() if layout_box_final is not None else None,
        "start_state": start_state,
    }

    if return_trace:
        out["trace"] = trace

    return out

@torch.no_grad()
def sample_prev_state_from_current_batch(
    model,
    obj_gen,
    obj_t: torch.Tensor,
    edge_t: torch.Tensor,
    rel_pos_t: torch.Tensor,
    t: torch.Tensor,                 # [B]
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    stochastic_obj: bool = False,
    stochastic_edge: bool = False,
    stochastic_rel: bool = False,
):
    """
    5C-lite helper:
    For each example in a batch, sample a single reverse step x_{t-1} from the current x_t.

    For t=0 examples, leave state unchanged.
    """
    B = obj_t.shape[0]

    obj_prev = obj_t.clone()
    edge_prev = edge_t.clone()
    rel_prev = rel_pos_t.clone()

    valid_prev_mask = (t > 0)

    for b in range(B):
        t_b = int(t[b].item())
        if t_b <= 0:
            continue

        step_out = reverse_step_via_discrete_posterior(
            model=model,
            obj_gen=obj_gen,
            obj_t=obj_t[b:b+1],
            edge_t=edge_t[b:b+1],
            rel_pos_t=rel_pos_t[b:b+1],
            t_cur=t_b,
            node_mask=node_mask[b:b+1],
            edge_mask=edge_mask[b:b+1],
            stochastic_obj=stochastic_obj,
            stochastic_edge=stochastic_edge,
            stochastic_rel=stochastic_rel,
        )

        obj_prev[b] = step_out["obj_t"][0]
        edge_prev[b] = step_out["edge_t"][0]
        rel_prev[b] = step_out["rel_pos_t"][0]

    return {
        "obj_prev": obj_prev.long(),
        "edge_prev": edge_prev.long(),
        "rel_prev": rel_prev.long(),
        "valid_prev_mask": valid_prev_mask,
    }

import torch
import torch.nn.functional as F



@torch.no_grad()
def sample_categorical_from_logits(logits: torch.Tensor, stochastic: bool = False, temperature: float = 1.0):
    """
    logits: [..., K]
    returns integer classes: [...]
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if not stochastic:
        return logits.argmax(dim=-1)

    probs = F.softmax(logits / temperature, dim=-1)
    orig_shape = probs.shape[:-1]
    K = probs.shape[-1]
    flat = probs.reshape(-1, K)
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.reshape(orig_shape)



@torch.no_grad()
def sample_binary_from_logits(
    logits: torch.Tensor,
    stochastic: bool = False,
    threshold: float = 0.0,
):
    """
    logits: [...]
    returns 0/1 tensor of same shape

    Deterministic mode:
        edge = 1 iff logit > threshold

    Stochastic mode:
        sample from sigmoid(logit - threshold), so positive threshold
        also makes stochastic sampling more conservative.
    """
    if not stochastic:
        return (logits > threshold).long()

    probs = torch.sigmoid(logits - threshold)
    return torch.bernoulli(probs).long()

@torch.no_grad()
def prune_edges_by_degree(
    edge_keep_mask: torch.Tensor,      # [B,N,N] bool
    edge_scores: torch.Tensor,         # [B,N,N] float, higher = more likely keep
    edge_mask: torch.Tensor,           # [B,N,N] bool
    max_out_degree: int,
    max_in_degree: int,
) -> torch.Tensor:
    """
    Confidence-based degree pruning.

    Keeps only the strongest edges while enforcing:
      out-degree(i) <= max_out_degree
      in-degree(j)  <= max_in_degree

    Strategy:

apply outgoing top-k pruning row-wise
apply incoming top-k pruning column-wise
final keep = row_keep & col_keep


    edge_scores should already reflect confidence for the current proposed edge.
    """
    B, N, _ = edge_keep_mask.shape
    valid_keep = edge_keep_mask & edge_mask.bool()

    if max_out_degree <= 0 and max_in_degree <= 0:
        return torch.zeros_like(valid_keep)

    neg_inf = torch.tensor(-1e9, device=edge_scores.device, dtype=edge_scores.dtype)
    masked_scores = torch.where(valid_keep, edge_scores, neg_inf)

    row_keep = torch.ones_like(valid_keep)
    col_keep = torch.ones_like(valid_keep)

    # ---------------------------------
    # Out-degree pruning
    # ---------------------------------
    if max_out_degree > 0:
        row_keep = torch.zeros_like(valid_keep)
        k_out = min(max_out_degree, N)

        for b in range(B):
            for i in range(N):
                row_valid = valid_keep[b, i]   # [N]
                num_valid = int(row_valid.sum().item())
                if num_valid == 0:
                    continue

                if num_valid <= max_out_degree:
                    row_keep[b, i, row_valid] = True
                    continue

                vals = masked_scores[b, i]  # [N]
                top_idx = torch.topk(vals, k=k_out, dim=-1).indices
                row_keep[b, i, top_idx] = True

    # ---------------------------------
    # In-degree pruning
    # ---------------------------------
    if max_in_degree > 0:
        col_keep = torch.zeros_like(valid_keep)
        k_in = min(max_in_degree, N)

        for b in range(B):
            for j in range(N):
                col_valid = valid_keep[b, :, j]   # [N]
                num_valid = int(col_valid.sum().item())
                if num_valid == 0:
                    continue

                if num_valid <= max_in_degree:
                    col_keep[b, col_valid, j] = True
                    continue

                vals = masked_scores[b, :, j]  # [N]
                top_idx = torch.topk(vals, k=k_in, dim=-1).indices
                col_keep[b, top_idx, j] = True

    pruned_keep = valid_keep
    if max_out_degree > 0:
        pruned_keep = pruned_keep & row_keep
    if max_in_degree > 0:
        pruned_keep = pruned_keep & col_keep

    return pruned_keep

@torch.no_grad()
def _tensor_membership_mask(x: torch.Tensor, values: Optional[Sequence[int]]) -> torch.Tensor:
    """
    x: arbitrary integer tensor
    values: python list / tuple of ints
    returns bool mask of same shape as x
    """
    if values is None or len(values) == 0:
        return torch.zeros_like(x, dtype=torch.bool)

    mask = torch.zeros_like(x, dtype=torch.bool)
    for v in values:
        mask |= (x == int(v))
    return mask

@torch.no_grad()
def apply_final_step_cleanup(
    obj_prev: torch.Tensor,                 # [B,N]
    edge_prev: torch.Tensor,                # [B,N,N]
    rel_prev: torch.Tensor,                 # [B,N,N]
    edge_rev_logits: torch.Tensor,          # [B,N,N]
    rel_logits_use: torch.Tensor,           # [B,N,N,K]
    edge_mask: torch.Tensor,                # [B,N,N]
    final_edge_logit_threshold: float = 0.5,
    final_rel_conf_threshold: float = 0.0,
    generic_obj_ids: Optional[Sequence[int]] = None,
    generic_attachment_rel_ids: Optional[Sequence[int]] = None,
    generic_attachment_edge_logit_threshold: float = 1.0,
):
    """
    Final-step cleanup applied only at t_cur == 1.

    Rules:

Remove any surviving edge whose edge logit is below final_edge_logit_threshold.
Remove any surviving edge whose chosen relation confidence is below final_rel_conf_threshold.
For generic-attachment edges, require a stronger edge logit threshold.


    Generic attachment edge:
      target object class in generic_obj_ids
      AND relation in generic_attachment_rel_ids
    """
    # --------------------------------------------------
    # Base keep mask from final edge confidence
    # --------------------------------------------------
    keep_mask = edge_prev.bool() & edge_mask.bool()
    keep_mask = keep_mask & (edge_rev_logits > final_edge_logit_threshold)

    # --------------------------------------------------
    # Relation confidence cleanup
    # --------------------------------------------------
    if final_rel_conf_threshold > 0.0:
        rel_probs = F.softmax(rel_logits_use, dim=-1)                     # [B,N,N,K]
        rel_conf = rel_probs.max(dim=-1).values                           # [B,N,N]
        keep_mask = keep_mask & (rel_conf > final_rel_conf_threshold)

    # --------------------------------------------------
    # Generic attachment cleanup
    # --------------------------------------------------
    if generic_obj_ids is not None and len(generic_obj_ids) > 0 and \
       generic_attachment_rel_ids is not None and len(generic_attachment_rel_ids) > 0:

        # target/object-side generic class mask
        obj_is_generic = _tensor_membership_mask(obj_prev, generic_obj_ids)   # [B,N]
        target_is_generic = obj_is_generic.unsqueeze(1).expand_as(edge_prev)  # [B,N,N]

        # relation class mask
        rel_is_generic_attach = _tensor_membership_mask(
            rel_prev, generic_attachment_rel_ids
        )                                                                     # [B,N,N]

        generic_attach_mask = target_is_generic & rel_is_generic_attach

        # require stronger edge logit for those edges
        keep_mask = keep_mask & (
            (~generic_attach_mask) |
            (edge_rev_logits > generic_attachment_edge_logit_threshold)
        )

    edge_prev = torch.where(keep_mask, edge_prev, torch.zeros_like(edge_prev))
    rel_prev = torch.where(keep_mask, rel_prev, torch.zeros_like(rel_prev))

    return edge_prev.long(), rel_prev.long()

@torch.no_grad()
def _ensure_current_in_candidates(candidates: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """
    candidates: [K]
    current: scalar tensor
    """
    if (candidates == current).any():
        return candidates
    return torch.cat([current.view(1), candidates], dim=0)


@torch.no_grad()
def _sample_from_tilted_scores(
    scores: torch.Tensor,  # [K]
    stochastic: bool,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("reward_tilt_temperature must be > 0")

    if not stochastic:
        return scores.argmax(dim=-1)

    probs = F.softmax(scores / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def reward_tilt_local_gibbs_step(
    obj_prev: torch.Tensor,                 # [B,N]
    edge_prev: torch.Tensor,                # [B,N,N]
    rel_prev: torch.Tensor,                 # [B,N,N]
    obj_logits_use: torch.Tensor,           # [B,N,K_obj]
    edge_logits_use: torch.Tensor,          # [B,N,N]
    rel_logits_use: torch.Tensor,           # [B,N,N,K_rel]
    node_mask: torch.Tensor,                # [B,N]
    edge_mask: torch.Tensor,                # [B,N,N]
    reward_fn: Callable,
    alpha: float = 1.0,
    temperature: float = 1.0,
    num_sweeps: int = 1,
    tilt_objects: bool = True,
    tilt_edges: bool = True,
    tilt_relations: bool = True,
    obj_topk: int = 5,
    rel_topk: int = 5,
    stochastic_obj: bool = False,
    stochastic_edge: bool = False,
    stochastic_rel: bool = False,
):
    """
    8A.3 reward-tilted local Gibbs refinement.

    This approximates:

        p_tilt(x_{t-1}|x_t) ∝ p_theta(x_{t-1}|x_t) exp(alpha R(x_{t-1}))

    by repeatedly updating local variables with conditionals:

        p_tilt(x_i | x_-i) ∝ p_theta(x_i) exp(alpha R(x_i, x_-i))
    """
    B, N = obj_prev.shape
    device = obj_prev.device

    obj_prev = obj_prev.clone()
    edge_prev = edge_prev.clone()
    rel_prev = rel_prev.clone()

    obj_logprob = F.log_softmax(obj_logits_use, dim=-1)
    rel_logprob = F.log_softmax(rel_logits_use, dim=-1)

    edge_logprob_0 = F.logsigmoid(-edge_logits_use)
    edge_logprob_1 = F.logsigmoid(edge_logits_use)

    for _ in range(int(num_sweeps)):
        # --------------------------------------------------
        # Object updates
        # --------------------------------------------------
        if tilt_objects:
            K_obj = obj_logits_use.shape[-1]
            k_obj = min(int(obj_topk), K_obj)

            for b in range(B):
                valid_nodes = torch.where(node_mask[b].bool())[0]

                for i in valid_nodes.tolist():
                    current_val = obj_prev[b, i]

                    top_candidates = torch.topk(
                        obj_logits_use[b, i],
                        k=k_obj,
                        dim=-1,
                    ).indices

                    candidates = _ensure_current_in_candidates(top_candidates, current_val)

                    scores = []
                    for cand in candidates:
                        obj_cand = obj_prev.clone()
                        obj_cand[b, i] = cand

                        reward_val = reward_fn(
                            obj_t=obj_cand,
                            edge_t=edge_prev,
                            rel_pos_t=rel_prev,
                            node_mask=node_mask,
                            edge_mask=edge_mask,
                        )[b]

                        base_lp = obj_logprob[b, i, cand]
                        scores.append(base_lp + float(alpha) * reward_val)

                    scores = torch.stack(scores, dim=0)
                    chosen_idx = _sample_from_tilted_scores(
                        scores=scores,
                        stochastic=stochastic_obj,
                        temperature=temperature,
                    )
                    obj_prev[b, i] = candidates[chosen_idx]

        # --------------------------------------------------
        # Edge updates
        # --------------------------------------------------
        if tilt_edges:
            for b in range(B):
                pair_idx = torch.where(edge_mask[b].bool())
                if pair_idx[0].numel() == 0:
                    continue

                for ii, jj in zip(pair_idx[0].tolist(), pair_idx[1].tolist()):
                    scores = []

                    # candidate edge = 0
                    edge_cand_0 = edge_prev.clone()
                    rel_cand_0 = rel_prev.clone()
                    edge_cand_0[b, ii, jj] = 0
                    rel_cand_0[b, ii, jj] = 0

                    reward_0 = reward_fn(
                        obj_t=obj_prev,
                        edge_t=edge_cand_0,
                        rel_pos_t=rel_cand_0,
                        node_mask=node_mask,
                        edge_mask=edge_mask,
                    )[b]
                    scores.append(edge_logprob_0[b, ii, jj] + float(alpha) * reward_0)

                    # candidate edge = 1
                    edge_cand_1 = edge_prev.clone()
                    rel_cand_1 = rel_prev.clone()
                    edge_cand_1[b, ii, jj] = 1

                    # If relation is currently zero, seed it with model's best relation.
                    if rel_cand_1[b, ii, jj].item() == 0:
                        rel_cand_1[b, ii, jj] = rel_logits_use[b, ii, jj].argmax()

                    reward_1 = reward_fn(
                        obj_t=obj_prev,
                        edge_t=edge_cand_1,
                        rel_pos_t=rel_cand_1,
                        node_mask=node_mask,
                        edge_mask=edge_mask,
                    )[b]
                    scores.append(edge_logprob_1[b, ii, jj] + float(alpha) * reward_1)

                    scores = torch.stack(scores, dim=0)
                    chosen = _sample_from_tilted_scores(
                        scores=scores,
                        stochastic=stochastic_edge,
                        temperature=temperature,
                    )

                    if chosen.item() == 0:
                        edge_prev[b, ii, jj] = 0
                        rel_prev[b, ii, jj] = 0
                    else:
                        edge_prev[b, ii, jj] = 1
                        if rel_prev[b, ii, jj].item() == 0:
                            rel_prev[b, ii, jj] = rel_logits_use[b, ii, jj].argmax()

        # --------------------------------------------------
        # Relation updates
        # --------------------------------------------------
        if tilt_relations:
            K_rel = rel_logits_use.shape[-1]
            k_rel = min(int(rel_topk), K_rel)

            for b in range(B):
                active_pairs = torch.where(edge_prev[b].bool() & edge_mask[b].bool())
                if active_pairs[0].numel() == 0:
                    continue

                for ii, jj in zip(active_pairs[0].tolist(), active_pairs[1].tolist()):
                    current_val = rel_prev[b, ii, jj]

                    top_candidates = torch.topk(
                        rel_logits_use[b, ii, jj],
                        k=k_rel,
                        dim=-1,
                    ).indices

                    candidates = _ensure_current_in_candidates(top_candidates, current_val)

                    scores = []
                    for cand in candidates:
                        rel_cand = rel_prev.clone()
                        rel_cand[b, ii, jj] = cand

                        reward_val = reward_fn(
                            obj_t=obj_prev,
                            edge_t=edge_prev,
                            rel_pos_t=rel_cand,
                            node_mask=node_mask,
                            edge_mask=edge_mask,
                        )[b]

                        base_lp = rel_logprob[b, ii, jj, cand]
                        scores.append(base_lp + float(alpha) * reward_val)

                    scores = torch.stack(scores, dim=0)
                    chosen_idx = _sample_from_tilted_scores(
                        scores=scores,
                        stochastic=stochastic_rel,
                        temperature=temperature,
                    )
                    rel_prev[b, ii, jj] = candidates[chosen_idx]

    edge_prev = edge_prev * edge_mask.long()
    rel_prev = torch.where(edge_prev.bool(), rel_prev, torch.zeros_like(rel_prev))

    return obj_prev.long(), edge_prev.long(), rel_prev.long()

# @torch.no_grad()
# def reward_tilt_edges_local_delta_step(
#     edge_prev: torch.Tensor,          # [B,N,N]
#     rel_prev: torch.Tensor,           # [B,N,N]
#     edge_logits_use: torch.Tensor,    # [B,N,N]
#     rel_logits_use: torch.Tensor,     # [B,N,N,K]
#     node_mask: torch.Tensor,          # [B,N]
#     edge_mask: torch.Tensor,          # [B,N,N]
#     alpha: float = 1.0,
#     temperature: float = 1.0,
#     stochastic_edge: bool = False,
#     w_isolated_node: float = 0.25,
#     w_dense_graph: float = 0.10,
#     w_bidirectional_edge: float = 0.10,
# ):
#     """
#     Fast 8A.3 edge-only reward-tilted local Gibbs.

#     Uses local analytic reward deltas for:
# isolated nodes
# dense graph penalty
# bidirectional edge penalty

#     Does NOT call full reward_fn.
#     Does NOT use layout reward.
#     """
#     if temperature <= 0:
#         raise ValueError("reward_tilt_temperature must be > 0")

#     B, N, _ = edge_prev.shape

#     edge_prev = edge_prev.clone()
#     rel_prev = rel_prev.clone()

#     edge_logprob_0 = F.logsigmoid(-edge_logits_use)
#     edge_logprob_1 = F.logsigmoid(edge_logits_use)

#     for b in range(B):
#         valid_pairs = torch.where(edge_mask[b].bool())
#         if valid_pairs[0].numel() == 0:
#             continue

#         for i, j in zip(valid_pairs[0].tolist(), valid_pairs[1].tolist()):
#             old_e = int(edge_prev[b, i, j].item())

#             # current degrees before candidate change
#             deg_i_old = int((edge_prev[b, i, :] & edge_mask[b, i, :]).sum().item()) \
#                       + int((edge_prev[b, :, i] & edge_mask[b, :, i]).sum().item())

#             deg_j_old = int((edge_prev[b, j, :] & edge_mask[b, j, :]).sum().item()) \
#                       + int((edge_prev[b, :, j] & edge_mask[b, :, j]).sum().item())

#             total_edges_old = float((edge_prev[b] & edge_mask[b]).sum().item())
#             valid_edge_count = float(edge_mask[b].sum().item())
#             valid_edge_count = max(valid_edge_count, 1.0)

#             reverse_e = int(edge_prev[b, j, i].item()) if bool(edge_mask[b, j, i]) else 0

#             scores = []

#             for cand_e in (0, 1):
#                 delta = cand_e - old_e

#                 deg_i_new = deg_i_old + delta
#                 deg_j_new = deg_j_old + delta

#                 # isolated reward term is negative isolated fraction.
#                 # Local delta: if degree changes 0<->1, isolated count changes.
#                 old_iso_i = 1.0 if deg_i_old == 0 else 0.0
#                 old_iso_j = 1.0 if deg_j_old == 0 else 0.0
#                 new_iso_i = 1.0 if deg_i_new == 0 else 0.0
#                 new_iso_j = 1.0 if deg_j_new == 0 else 0.0

#                 num_nodes = max(float(node_mask[b].sum().item()), 1.0)
#                 delta_isolated_reward = -((new_iso_i + new_iso_j) - (old_iso_i + old_iso_j)) / num_nodes

#                 # dense graph reward term is negative edge density.
#                 total_edges_new = total_edges_old + delta
#                 delta_dense_reward = -((total_edges_new - total_edges_old) / valid_edge_count)

#                 # bidirectional reward is negative bidirectional fraction/proxy.
#                 # Adding i->j while j->i exists creates bidirectional penalty.
#                 # Removing i->j while j->i exists removes it.
#                 delta_bidir_reward = 0.0
#                 if reverse_e == 1:
#                     delta_bidir_reward = -float(delta) / valid_edge_count

#                 local_reward_delta = (
#                     float(w_isolated_node) * delta_isolated_reward
#                     + float(w_dense_graph) * delta_dense_reward
#                     + float(w_bidirectional_edge) * delta_bidir_reward
#                 )

#                 base_lp = edge_logprob_1[b, i, j] if cand_e == 1 else edge_logprob_0[b, i, j]
#                 scores.append(base_lp + float(alpha) * base_lp.new_tensor(local_reward_delta))

#             scores = torch.stack(scores, dim=0)

#             if stochastic_edge:
#                 probs = F.softmax(scores / temperature, dim=0)
#                 chosen = torch.multinomial(probs, num_samples=1).item()
#             else:
#                 chosen = int(scores.argmax().item())

#             edge_prev[b, i, j] = chosen

#             if chosen == 0:
#                 rel_prev[b, i, j] = 0
#             else:
#                 if rel_prev[b, i, j].item() == 0:
#                     rel_prev[b, i, j] = rel_logits_use[b, i, j].argmax()

#     edge_prev = edge_prev * edge_mask.long()
#     rel_prev = torch.where(edge_prev.bool(), rel_prev, torch.zeros_like(rel_prev))

#     return edge_prev.long(), rel_prev.long()

# @torch.no_grad()
# def reward_tilt_edges_local_delta_step(
#     edge_prev,
#     rel_prev,
#     edge_logits_use,
#     rel_logits_use,
#     node_mask,
#     edge_mask,
#     alpha=1.0,
#     temperature=1.0,
#     stochastic_edge=False,
#     w_isolated_node=0.25,
#     w_dense_graph=0.10,
#     w_bidirectional_edge=0.10,
#     reward_tilt_edge_logit_band: float = 0.75,
#     reward_w_hub_degree: float = 0.50,
#     reward_hub_degree_threshold: int = 4,
# ):
#     edge_prev = edge_prev.long()
#     rel_prev = rel_prev.long()
#     edge_mask_b = edge_mask.bool()

#     B, N, _ = edge_prev.shape
#     device = edge_prev.device
#     dtype = edge_logits_use.dtype

#     # Current undirected-ish node degree: outgoing + incoming
#     out_deg = (edge_prev * edge_mask.long()).sum(dim=-1)  # [B,N]
#     in_deg = (edge_prev * edge_mask.long()).sum(dim=-2)   # [B,N]
#     deg = out_deg + in_deg                                # [B,N]

#     old_e = edge_prev.float()                             # [B,N,N]

#     deg_i_old = deg.unsqueeze(2).expand(B, N, N).float()
#     deg_j_old = deg.unsqueeze(1).expand(B, N, N).float()

#     # Candidate deltas for edge=0 and edge=1
#     delta0 = 0.0 - old_e
#     delta1 = 1.0 - old_e

#     num_nodes = node_mask.float().sum(dim=1).clamp(min=1.0).view(B, 1, 1)
#     valid_edge_count = edge_mask.float().sum(dim=(1, 2)).clamp(min=1.0).view(B, 1, 1)

#     def candidate_reward_delta(delta):
#         deg_i_new = deg_i_old + delta
#         deg_j_new = deg_j_old + delta

#         old_iso_i = (deg_i_old == 0).float()
#         old_iso_j = (deg_j_old == 0).float()
#         new_iso_i = (deg_i_new == 0).float()
#         new_iso_j = (deg_j_new == 0).float()

#         delta_isolated = -((new_iso_i + new_iso_j) - (old_iso_i + old_iso_j)) / num_nodes
#         delta_dense = -(delta / valid_edge_count)

#         reverse_e = edge_prev.transpose(1, 2).float()
#         delta_bidir = -(delta * reverse_e) / valid_edge_count

#         return (
#             float(w_isolated_node) * delta_isolated
#             + float(w_dense_graph) * delta_dense
#             + float(w_bidirectional_edge) * delta_bidir
#         )

#     reward_delta0 = candidate_reward_delta(delta0)
#     reward_delta1 = candidate_reward_delta(delta1)

#     score0 = F.logsigmoid(-edge_logits_use) + float(alpha) * reward_delta0
#     score1 = F.logsigmoid(edge_logits_use) + float(alpha) * reward_delta1

#     scores = torch.stack([score0, score1], dim=-1)  # [B,N,N,2]
#     scores = torch.where(
#         edge_mask_b.unsqueeze(-1),
#         scores,
#         torch.full_like(scores, -1e9),
#     )

#     if stochastic_edge:
#         probs = F.softmax(scores / temperature, dim=-1)
#         edge_new = torch.multinomial(
#             probs.reshape(-1, 2),
#             num_samples=1,
#         ).reshape(B, N, N)
#     else:
#         edge_new = scores.argmax(dim=-1)

#     edge_new = edge_new.long() * edge_mask.long()

#     # Preserve old relations where edge survives; seed new edges with best relation.
#     best_rel = rel_logits_use.argmax(dim=-1).long()
#     rel_new = torch.where(edge_new.bool(), rel_prev, torch.zeros_like(rel_prev))
#     rel_new = torch.where(
#         edge_new.bool() & (rel_new == 0),
#         best_rel,
#         rel_new,
#     )
#     rel_new = torch.where(edge_new.bool(), rel_new, torch.zeros_like(rel_new))

#     return edge_new.long(), rel_new.long()

@torch.no_grad()
def reward_tilt_edges_local_delta_step(
    edge_prev: torch.Tensor,          # [B,N,N]
    rel_prev: torch.Tensor,           # [B,N,N]
    edge_logits_use: torch.Tensor,    # [B,N,N]
    rel_logits_use: torch.Tensor,     # [B,N,N,K]
    node_mask: torch.Tensor,          # [B,N]
    edge_mask: torch.Tensor,          # [B,N,N]
    alpha: float = 1.0,
    temperature: float = 1.0,
    stochastic_edge: bool = False,
    w_isolated_node: float = 0.25,
    w_dense_graph: float = 0.10,
    w_bidirectional_edge: float = 0.10,
    edge_logit_band: float = 0.75,
    w_hub_degree: float = 0.50,
    hub_degree_threshold: int = 4,
):
    """
    Vectorized 8A.3 edge-only reward-tilted local update.

    Tilted local conditional:

        score(e_ij=k) = log p_theta(e_ij=k) + alpha * Delta R_ij(k)

    Boundary mask:
        only apply tilt when abs(edge_logit) < edge_logit_band.

    Hub penalty:
        discourages adding edges to already high-degree nodes.
    """
    if temperature <= 0:
        raise ValueError("reward_tilt_temperature must be > 0")

    edge_prev = edge_prev.long()
    rel_prev = rel_prev.long()

    B, N, _ = edge_prev.shape
    edge_mask_bool = edge_mask.bool()
    edge_mask_f = edge_mask.float()

    # Existing model-sampled edge state.
    old_e = edge_prev.float() * edge_mask_f

    # Degrees from current sampled state.
    out_deg = old_e.sum(dim=-1)       # [B,N]
    in_deg = old_e.sum(dim=-2)        # [B,N]
    deg = out_deg + in_deg            # [B,N]

    deg_i_old = deg.unsqueeze(2).expand(B, N, N)
    deg_j_old = deg.unsqueeze(1).expand(B, N, N)

    # Candidate deltas: edge=0 and edge=1.
    delta0 = 0.0 - old_e
    delta1 = 1.0 - old_e

    num_nodes = node_mask.float().sum(dim=1).clamp(min=1.0).view(B, 1, 1)
    valid_edge_count = edge_mask_f.sum(dim=(1, 2)).clamp(min=1.0).view(B, 1, 1)

    reverse_e = old_e.transpose(1, 2)

    hub_thr = float(hub_degree_threshold)

    def candidate_reward_delta(delta: torch.Tensor) -> torch.Tensor:
        deg_i_new = deg_i_old + delta
        deg_j_new = deg_j_old + delta

        # 1. Isolated-node reward: reward = -isolated_frac.
        old_iso_i = (deg_i_old == 0).float()
        old_iso_j = (deg_j_old == 0).float()
        new_iso_i = (deg_i_new == 0).float()
        new_iso_j = (deg_j_new == 0).float()

        delta_isolated = -((new_iso_i + new_iso_j) - (old_iso_i + old_iso_j)) / num_nodes

        # 2. Dense-graph reward: reward = -edge_density.
        delta_dense = -(delta / valid_edge_count)

        # 3. Bidirectional reward: penalize creating reciprocal edge.
        delta_bidir = -(delta * reverse_e) / valid_edge_count

        # 4. Hub reward: penalize degree above threshold.
        old_hub_i = torch.clamp(deg_i_old - hub_thr, min=0.0)
        old_hub_j = torch.clamp(deg_j_old - hub_thr, min=0.0)
        new_hub_i = torch.clamp(deg_i_new - hub_thr, min=0.0)
        new_hub_j = torch.clamp(deg_j_new - hub_thr, min=0.0)

        delta_hub = -((new_hub_i + new_hub_j) - (old_hub_i + old_hub_j)) / num_nodes

        return (
            float(w_isolated_node) * delta_isolated
            + float(w_dense_graph) * delta_dense
            + float(w_bidirectional_edge) * delta_bidir
            + float(w_hub_degree) * delta_hub
        )

    reward_delta0 = candidate_reward_delta(delta0)
    reward_delta1 = candidate_reward_delta(delta1)

    score0_base = F.logsigmoid(-edge_logits_use)
    score1_base = F.logsigmoid(edge_logits_use)

    score0_tilted = score0_base + float(alpha) * reward_delta0
    score1_tilted = score1_base + float(alpha) * reward_delta1

    # Boundary mask: only tilt uncertain edges.
    tilt_mask = edge_mask_bool & (edge_logits_use.abs() < float(edge_logit_band))

    score0 = torch.where(tilt_mask, score0_tilted, score0_base)
    score1 = torch.where(tilt_mask, score1_tilted, score1_base)

    scores = torch.stack([score0, score1], dim=-1)  # [B,N,N,2]

    invalid_scores = torch.full_like(scores, -1e9)
    scores = torch.where(edge_mask_bool.unsqueeze(-1), scores, invalid_scores)

    if stochastic_edge:
        probs = F.softmax(scores / float(temperature), dim=-1)
        edge_new = torch.multinomial(
            probs.reshape(-1, 2),
            num_samples=1,
        ).reshape(B, N, N)
    else:
        edge_new = scores.argmax(dim=-1)

    edge_new = edge_new.long() * edge_mask.long()

    # Preserve old relations where edge survives; seed newly active edges.
    best_rel = rel_logits_use.argmax(dim=-1).long()

    rel_new = torch.where(edge_new.bool(), rel_prev, torch.zeros_like(rel_prev))
    rel_new = torch.where(
        edge_new.bool() & (rel_new == 0),
        best_rel,
        rel_new,
    )
    rel_new = torch.where(edge_new.bool(), rel_new, torch.zeros_like(rel_new))

    return edge_new.long(), rel_new.long()

@torch.no_grad()
def reward_tilt_relations_local_geometry_step(
    rel_prev: torch.Tensor,             # [B,N,N]
    edge_prev: torch.Tensor,            # [B,N,N]
    rel_logits_use: torch.Tensor,       # [B,N,N,K]
    layout_box_pred: torch.Tensor,      # [B,N,4], cxcywh normalized
    node_mask: torch.Tensor,            # [B,N]
    edge_mask: torch.Tensor,            # [B,N,N]
    relation_group_pos_ids: dict,
    alpha: float = 0.5,
    temperature: float = 1.0,
    stochastic_rel: bool = False,
    rel_topk: int = 5,
    w_relation_geometry: float = 1.0,
):
    """
    Relation-only local reward tilt on existing active edges.

    Does NOT create/remove edges.
    Does NOT change objects.
    Only changes relation class among top-k model candidates.

    score(r_ij=k) =
        log p_theta(r_ij=k)
        + alpha * local_geometry_reward(k, box_i, box_j)

    rel ids here are positive relation ids, i.e. rel_pos ids.
    """
    if layout_box_pred is None:
        return rel_prev.long()

    if temperature <= 0:
        raise ValueError("reward_tilt_temperature must be > 0")

    B, N, _ = rel_prev.shape
    device = rel_prev.device
    K = rel_logits_use.shape[-1]

    rel_prev = rel_prev.clone().long()

    active_mask = edge_prev.bool() & edge_mask.bool()

    if not active_mask.any():
        return rel_prev.long()

    # --------------------------------------------------
    # Candidate relation ids: top-k plus current relation
    # --------------------------------------------------
    k_rel = min(int(rel_topk), K)
    topk_ids = torch.topk(rel_logits_use, k=k_rel, dim=-1).indices  # [B,N,N,k]

    current_rel = rel_prev.unsqueeze(-1)  # [B,N,N,1]
    candidate_ids = torch.cat([current_rel, topk_ids], dim=-1)      # [B,N,N,k+1]
    C = candidate_ids.shape[-1]

    candidate_ids = candidate_ids.clamp(min=0, max=K - 1)

    rel_logprob = F.log_softmax(rel_logits_use, dim=-1)
    base_scores = torch.gather(rel_logprob, dim=-1, index=candidate_ids)  # [B,N,N,C]

    # --------------------------------------------------
    # Box geometry
    # layout_box_pred: cx, cy, w, h
    # --------------------------------------------------
    cx = layout_box_pred[..., 0]
    cy = layout_box_pred[..., 1]
    bw = layout_box_pred[..., 2].clamp(min=1e-6)
    bh = layout_box_pred[..., 3].clamp(min=1e-6)

    cx_i = cx.unsqueeze(2).expand(B, N, N)
    cy_i = cy.unsqueeze(2).expand(B, N, N)
    w_i = bw.unsqueeze(2).expand(B, N, N)
    h_i = bh.unsqueeze(2).expand(B, N, N)

    cx_j = cx.unsqueeze(1).expand(B, N, N)
    cy_j = cy.unsqueeze(1).expand(B, N, N)
    w_j = bw.unsqueeze(1).expand(B, N, N)
    h_j = bh.unsqueeze(1).expand(B, N, N)

    x1_i = cx_i - 0.5 * w_i
    y1_i = cy_i - 0.5 * h_i
    x2_i = cx_i + 0.5 * w_i
    y2_i = cy_i + 0.5 * h_i

    x1_j = cx_j - 0.5 * w_j
    y1_j = cy_j - 0.5 * h_j
    x2_j = cx_j + 0.5 * w_j
    y2_j = cy_j + 0.5 * h_j

    # --------------------------------------------------
    # Local geometry scores per pair
    # Positive means relation is geometrically plausible.
    # Shape: [B,N,N]
    # --------------------------------------------------
    margin = 0.02

    score_above = (cy_j - cy_i) / (h_i + h_j + 1e-6)
    score_below = (cy_i - cy_j) / (h_i + h_j + 1e-6)
    score_left = (cx_j - cx_i) / (w_i + w_j + 1e-6)
    score_right = (cx_i - cx_j) / (w_i + w_j + 1e-6)

    # subject inside object
    inside_x = torch.minimum(x2_i, x2_j) - torch.maximum(x1_i, x1_j)
    inside_y = torch.minimum(y2_i, y2_j) - torch.maximum(y1_i, y1_j)
    inter = inside_x.clamp(min=0.0) * inside_y.clamp(min=0.0)
    area_i = (x2_i - x1_i).clamp(min=1e-6) * (y2_i - y1_i).clamp(min=1e-6)
    score_inside = inter / area_i

    geometry_reward = torch.zeros(B, N, N, C, device=device, dtype=rel_logits_use.dtype)

    def add_group_reward(group_name: str, pair_score: torch.Tensor):
        ids = relation_group_pos_ids.get(group_name, [])
        if ids is None or len(ids) == 0:
            return

        group_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
        for rid in ids:
            if 0 <= int(rid) < K:
                group_mask |= (candidate_ids == int(rid))

        geometry_reward[group_mask] = pair_score.unsqueeze(-1).expand_as(geometry_reward)[group_mask]

    add_group_reward("above", score_above)
    add_group_reward("below", score_below)
    add_group_reward("left", score_left)
    add_group_reward("right", score_right)
    add_group_reward("inside", score_inside)

    # tilted_scores = base_scores + float(alpha) * float(w_relation_geometry) * geometry_reward
    layout_reward_scale = float(alpha) * float(w_relation_geometry)
    tilted_scores = base_scores + layout_reward_scale * geometry_reward

    # Penalize geometrically bad spatial relation candidates.
    # This prevents choosing "above/on/in/under" when boxes disagree.
    bad_geom_mask = geometry_reward < -0.05
    tilted_scores = torch.where(
        bad_geom_mask,
        tilted_scores - layout_reward_scale,
        tilted_scores,
    )

    # Only active edges are eligible for relation update.
    invalid = torch.full_like(tilted_scores, -1e9)
    tilted_scores = torch.where(active_mask.unsqueeze(-1), tilted_scores, invalid)

    if stochastic_rel:
        probs = F.softmax(tilted_scores / float(temperature), dim=-1)
        chosen_local = torch.multinomial(
            probs.reshape(-1, C),
            num_samples=1,
        ).reshape(B, N, N)
    else:
        chosen_local = tilted_scores.argmax(dim=-1)

    chosen_rel = torch.gather(
        candidate_ids,
        dim=-1,
        index=chosen_local.unsqueeze(-1),
    ).squeeze(-1)

    rel_prev = torch.where(active_mask, chosen_rel.long(), torch.zeros_like(rel_prev))
    return rel_prev.long()

@torch.no_grad()
def reward_tilt_objects_local_prior_step(
    obj_prev: torch.Tensor,              # [B,N]
    obj_rev_logits: torch.Tensor,        # [B,N,K]
    node_mask: torch.Tensor,             # [B,N]
    edge_prev: torch.Tensor,             # [B,N,N]
    rel_prev: torch.Tensor,              # [B,N,N]
    obj_log_prior: torch.Tensor,         # [K]
    alpha: float = 0.25,
    temperature: float = 1.0,
    stochastic_obj: bool = False,
    obj_topk: int = 5,
    obj_logit_margin: float = 1.0,
    w_object_class_prior: float = 0.50,
    w_object_relation_support: float = 0.25,
):
    """
    Conservative object-only reward tilt.

    Does NOT create/remove edges.
    Does NOT change relations.
    Only changes object class among top-k candidates for uncertain nodes.

    score(o_i=k) =
        log p_theta(o_i=k)
        + alpha * [
            w_prior * log p_data(k)
            + w_support * relation_support(i, k)
        ]

    Boundary/uncertainty rule:
        only tilt nodes whose top1-top2 logit margin < obj_logit_margin.
    """
    if temperature <= 0:
        raise ValueError("reward_tilt_temperature must be > 0")

    B, N = obj_prev.shape
    K = obj_rev_logits.shape[-1]
    device = obj_prev.device

    obj_prev = obj_prev.clone().long()
    obj_log_prior = obj_log_prior.to(device=device, dtype=obj_rev_logits.dtype)

    valid_node_mask = node_mask.bool()
    if not valid_node_mask.any():
        return obj_prev.long()

    # --------------------------------------------------
    # Uncertainty mask: only tilt ambiguous object choices
    # --------------------------------------------------
    top2 = torch.topk(obj_rev_logits, k=min(2, K), dim=-1).values
    if top2.shape[-1] == 1:
        margin = torch.zeros(B, N, device=device, dtype=obj_rev_logits.dtype)
    else:
        margin = top2[..., 0] - top2[..., 1]

    tilt_node_mask = valid_node_mask & (margin < float(obj_logit_margin))

    if not tilt_node_mask.any():
        return obj_prev.long()

    # --------------------------------------------------
    # Candidate ids: current + top-k
    # --------------------------------------------------
    k_obj = min(int(obj_topk), K)
    topk_ids = torch.topk(obj_rev_logits, k=k_obj, dim=-1).indices  # [B,N,k]

    current_obj = obj_prev.unsqueeze(-1)                            # [B,N,1]
    candidate_ids = torch.cat([current_obj, topk_ids], dim=-1)      # [B,N,C]
    C = candidate_ids.shape[-1]
    candidate_ids = candidate_ids.clamp(min=0, max=K - 1)

    obj_logprob = F.log_softmax(obj_rev_logits, dim=-1)
    base_scores = torch.gather(obj_logprob, dim=-1, index=candidate_ids)  # [B,N,C]

    # --------------------------------------------------
    # Prior reward: prefer plausible dataset object classes
    # --------------------------------------------------
    prior_reward = obj_log_prior[candidate_ids]  # [B,N,C]

    # Normalize prior reward to avoid huge negative shifts.
    prior_reward = prior_reward - obj_log_prior.mean()

    # --------------------------------------------------
    # Relation-support reward
    # Conservative heuristic:
    #   Nodes with incident edges should prefer common/less-rare object labels.
    #   Isolated nodes receive no relation support boost.
    #
    # This avoids hard-coding class names and avoids needing object-pair stats.
    # --------------------------------------------------
    edge_f = edge_prev.float()
    incident_degree = edge_f.sum(dim=-1) + edge_f.sum(dim=-2)  # [B,N]
    has_relation = (incident_degree > 0).float().unsqueeze(-1) # [B,N,1]

    relation_support_reward = has_relation * prior_reward

    reward_score = (
        float(w_object_class_prior) * prior_reward
        + float(w_object_relation_support) * relation_support_reward
    )

    tilted_scores = base_scores + float(alpha) * reward_score

    # Only tilt uncertain valid nodes.
    invalid = torch.full_like(tilted_scores, -1e9)
    tilted_scores = torch.where(
        tilt_node_mask.unsqueeze(-1),
        tilted_scores,
        invalid,
    )

    if stochastic_obj:
        probs = F.softmax(tilted_scores / float(temperature), dim=-1)
        chosen_local = torch.multinomial(
            probs.reshape(-1, C),
            num_samples=1,
        ).reshape(B, N)
    else:
        chosen_local = tilted_scores.argmax(dim=-1)

    chosen_obj = torch.gather(
        candidate_ids,
        dim=-1,
        index=chosen_local.unsqueeze(-1),
    ).squeeze(-1)

    obj_prev = torch.where(tilt_node_mask, chosen_obj.long(), obj_prev)
    obj_prev = torch.where(valid_node_mask, obj_prev, torch.zeros_like(obj_prev))

    return obj_prev.long()

@torch.no_grad()
def compute_layout_global_reward(
    layout_box_pred: torch.Tensor,   # [B,N,4], cxcywh normalized
    node_mask: torch.Tensor,         # [B,N]
    w_overlap: float = 1.0,
    w_spread: float = 0.5,
    w_bounds: float = 0.5,
):
    """
    Returns per-graph layout reward [B].

    Higher is better.

    Penalizes:

pairwise box overlap
boxes outside [0,1]
center collapse / low spatial spread
    """
    B, N, _ = layout_box_pred.shape
    device = layout_box_pred.device
    dtype = layout_box_pred.dtype

    valid = node_mask.bool()

    cx = layout_box_pred[..., 0]
    cy = layout_box_pred[..., 1]
    bw = layout_box_pred[..., 2].clamp(min=1e-6)
    bh = layout_box_pred[..., 3].clamp(min=1e-6)

    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh

    # -------------------------
    # Bounds penalty
    # -------------------------
    bounds_violation = (
        F.relu(-x1)
        + F.relu(-y1)
        + F.relu(x2 - 1.0)
        + F.relu(y2 - 1.0)
    )
    bounds_penalty = (
        bounds_violation * valid.float()
    ).sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1.0)

    # -------------------------
    # Pairwise IoU overlap
    # -------------------------
    x1_i = x1.unsqueeze(2)
    y1_i = y1.unsqueeze(2)
    x2_i = x2.unsqueeze(2)
    y2_i = y2.unsqueeze(2)

    x1_j = x1.unsqueeze(1)
    y1_j = y1.unsqueeze(1)
    x2_j = x2.unsqueeze(1)
    y2_j = y2.unsqueeze(1)

    inter_w = (torch.minimum(x2_i, x2_j) - torch.maximum(x1_i, x1_j)).clamp(min=0.0)
    inter_h = (torch.minimum(y2_i, y2_j) - torch.maximum(y1_i, y1_j)).clamp(min=0.0)
    inter = inter_w * inter_h

    area = ((x2 - x1).clamp(min=1e-6) * (y2 - y1).clamp(min=1e-6))
    area_i = area.unsqueeze(2)
    area_j = area.unsqueeze(1)
    union = (area_i + area_j - inter).clamp(min=1e-6)
    iou = inter / union

    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    pair_valid = pair_valid & (~eye)

    overlap_penalty = (
        iou * pair_valid.float()
    ).sum(dim=(1, 2)) / pair_valid.float().sum(dim=(1, 2)).clamp(min=1.0)

    # -------------------------
    # Spread reward
    # -------------------------
    valid_f = valid.float()
    denom = valid_f.sum(dim=-1).clamp(min=1.0)

    mean_cx = (cx * valid_f).sum(dim=-1) / denom
    mean_cy = (cy * valid_f).sum(dim=-1) / denom

    var_x = (((cx - mean_cx.unsqueeze(-1)) ** 2) * valid_f).sum(dim=-1) / denom
    var_y = (((cy - mean_cy.unsqueeze(-1)) ** 2) * valid_f).sum(dim=-1) / denom

    spread = torch.sqrt((var_x + var_y).clamp(min=1e-8))

    reward = (
        -float(w_overlap) * overlap_penalty
        -float(w_bounds) * bounds_penalty
        +float(w_spread) * spread
    )

    return reward