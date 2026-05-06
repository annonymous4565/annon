# # diffusion/objective_generator.py

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch

from diffusion.schedules import DiscreteNoiseSchedules
from diffusion.transitions import (
    CategoricalTransitionKernel,
    make_uniform_prior,
    apply_transition_to_labels,
)
from diffusion.priors import (
    compute_object_prior_from_dataset,
    compute_relation_prior_from_dataset,
)

from configs import DiscreteSGConfig




class DiscreteSGObjectiveGenerator:
    def __init__(
        self,
        cfg,
        num_obj_classes: int,
        num_rel_classes: int,
        device: torch.device,
        dataset_for_priors: Optional[object] = None,
    ):
        """
        cfg is your DiscreteSGConfig
        """
        self.cfg = cfg
        self.device = device

        self.num_steps = cfg.num_diffusion_steps
        self.pad_token_id = cfg.pad_token_id
        self.no_rel_token_id = cfg.no_rel_token_id

        self.num_obj_classes = num_obj_classes
        self.mask_obj_token_id = num_obj_classes

        self.use_masked_node_diffusion = getattr(cfg, "use_masked_node_diffusion", False)
        self.node_random_corruption_prob = getattr(cfg, "node_random_corruption_prob", 0.0)

        self.obj_betas = torch.linspace(
            cfg.obj_beta_start,
            cfg.obj_beta_end,
            cfg.num_diffusion_steps,
            dtype=torch.float32,
            device=device,
        )

        self.rel_betas = torch.linspace(
            cfg.rel_beta_start,
            cfg.rel_beta_end,
            cfg.num_diffusion_steps,
            dtype=torch.float32,
            device=device,
        )

        # priors
        if cfg.use_empirical_obj_prior:
            if dataset_for_priors is None:
                raise ValueError("dataset_for_priors must be provided when use_empirical_obj_prior=True")
            obj_prior = compute_object_prior_from_dataset(
                dataset_for_priors,
                pad_token_id=cfg.pad_token_id,
            ).to(device)
        else:
            obj_prior = make_uniform_prior(num_obj_classes, device=device)
            obj_prior[cfg.pad_token_id] = 0.0
            obj_prior = obj_prior / obj_prior.sum()

        if cfg.use_empirical_rel_prior:
            if dataset_for_priors is None:
                raise ValueError("dataset_for_priors must be provided when use_empirical_rel_prior=True")
            rel_prior_all, rel_prior_nonnull = compute_relation_prior_from_dataset(
                dataset_for_priors,
                no_rel_token_id=cfg.no_rel_token_id,
            )
            rel_prior_all = rel_prior_all.to(device)
            rel_prior_nonnull = rel_prior_nonnull.to(device)
        else:
            rel_prior_all = make_uniform_prior(num_rel_classes, device=device)
            rel_prior_nonnull = rel_prior_all.clone()
            rel_prior_nonnull[cfg.no_rel_token_id] = 0.0
            rel_prior_nonnull = rel_prior_nonnull / rel_prior_nonnull.sum()

        # kernels
        self.obj_kernel = CategoricalTransitionKernel(
            num_classes=num_obj_classes,
            prior_probs=obj_prior,
            special_absorbing_id=cfg.pad_token_id,
            no_rel_token_id=None,
            no_rel_leak_scale=0.0,
            use_sticky_no_rel=False,
            nonnull_prior_probs=None,
            device=device,
        )

        self.rel_kernel = CategoricalTransitionKernel(
            num_classes=num_rel_classes,
            prior_probs=rel_prior_all,
            special_absorbing_id=None,
            no_rel_token_id=cfg.no_rel_token_id,
            no_rel_leak_scale=cfg.no_rel_leak_scale,
            use_sticky_no_rel=cfg.use_sticky_no_rel,
            nonnull_prior_probs=rel_prior_nonnull,
            device=device,
        )

        # precompute cumulative transitions
        self.obj_Q_bars = self.obj_kernel.precompute_Q_bars(self.obj_betas)  # [T, Ko, Ko]
        self.rel_Q_bars = self.rel_kernel.precompute_Q_bars(self.rel_betas)  # [T, Kr, Kr]

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        returns t in [0, T-1], shape [B]
        """
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            dtype=torch.long,
            device=self.device,
        )

    def q_sample_objects(self, obj_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        obj_0: [B, N]
        t: [B]
        """
        B, N = obj_0.shape
        obj_t = torch.empty_like(obj_0)

        for b in range(B):
            Q_bar_t = self.obj_Q_bars[t[b]]   # [Ko, Ko]
            obj_t[b] = apply_transition_to_labels(obj_0[b], Q_bar_t)

        return obj_t

    def q_sample_relations(self, rel_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        rel_0: [B, N, N]
        t: [B]
        """
        B, N, _ = rel_0.shape
        rel_t = torch.empty_like(rel_0)

        for b in range(B):
            Q_bar_t = self.rel_Q_bars[t[b]]   # [Kr, Kr]
            rel_t[b] = apply_transition_to_labels(rel_0[b], Q_bar_t)

        return rel_t

    def get_training_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input batch should contain:
            obj_labels, rel_labels, node_mask, edge_mask

        Returns:
            obj_0, rel_0, obj_t, rel_t, t, node_mask, edge_mask
        """
        obj_0 = batch["obj_labels"].to(self.device)
        rel_0 = batch["rel_labels"].to(self.device)
        node_mask = batch["node_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device)

        B = obj_0.shape[0]
        t = self.sample_timesteps(B)

        if self.use_masked_node_diffusion:
            obj_t = self.q_sample_objects_with_mask(
                obj_0=obj_0,
                node_mask=node_mask,
                t=t,
            )
        else:
            obj_t = self.q_sample_objects(obj_0, t)

        rel_t = self.q_sample_relations(rel_0, t)

        return {
            "obj_0": obj_0,
            "rel_0": rel_0,
            "obj_t": obj_t,
            "rel_t": rel_t,
            "t": t,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
        }
    
    def q_sample_objects_with_mask(
        self,
        obj_0: torch.Tensor,
        node_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask-based object corruption.

        For each valid node at timestep t:
          p_mask = t / (T - 1)
          p_rand = node_random_corruption_prob * p_mask
          p_keep = 1 - p_mask - p_rand

        Invalid/padded nodes are left unchanged.
        """
        device = obj_0.device
        B, N = obj_0.shape

        # tau = t.float() / max(self.num_steps - 1, 1)          # [B]
        # p_mask = tau[:, None].expand(B, N)                    # [B, N]
        # p_rand = (self.node_random_corruption_prob * tau)[:, None].expand(B, N)
        # p_keep = 1.0 - p_mask - p_rand
        # p_keep = torch.clamp(p_keep, min=0.0)

        tau = t.float() / max(self.num_steps - 1, 1)          # [B]
        tau = tau[:, None].expand(B, N)                       # [B, N]

        rho = self.node_random_corruption_prob

        p_keep = 1.0 - tau
        p_rand = rho * tau
        p_mask = (1.0 - rho) * tau

        u = torch.rand(B, N, device=device)

        obj_t = obj_0.clone()
        valid = node_mask.bool()

        # mask corruption
        mask_sel = valid & (u >= p_keep) & (u < p_keep + p_mask)
        obj_t[mask_sel] = self.mask_obj_token_id

        # optional random corruption to real object classes only
        rand_sel = valid & (u >= p_keep + p_mask)
        if rand_sel.any() and self.node_random_corruption_prob > 0.0:
            rand_vals = torch.randint(
                low=0,
                high=self.num_obj_classes,
                size=(rand_sel.sum().item(),),
                device=device,
            )
            obj_t[rand_sel] = rand_vals

        return obj_t


    
def corrupt_object_labels_with_mask(
    obj_0: torch.Tensor,         # [B, N]
    node_mask: torch.Tensor,     # [B, N] bool
    t: torch.Tensor,             # [B]
    num_steps: int,
    mask_obj_token_id: int,
    node_random_corruption_prob: float = 0.0,
) -> torch.Tensor:
    """
    Corrupt object labels using an absorbing MASK token.

    At timestep t:
    p_mask = t / (num_steps - 1)
    p_keep = 1 - p_mask - p_rand
    p_rand = node_random_corruption_prob * (t / (num_steps - 1))

    Returns:
    obj_t: [B, N]
    """
    device = obj_0.device
    B, N = obj_0.shape

    tau = t.float() / max(num_steps - 1, 1)               # [B]
    p_mask = tau[:, None].expand(B, N)                    # [B, N]
    p_rand = (node_random_corruption_prob * tau)[:, None].expand(B, N)
    p_keep = 1.0 - p_mask - p_rand
    p_keep = torch.clamp(p_keep, min=0.0)

    u = torch.rand(B, N, device=device)

    obj_t = obj_0.clone()

    # valid nodes only
    valid = node_mask.bool()

    # mask corruption
    mask_sel = valid & (u >= p_keep) & (u < p_keep + p_mask)
    obj_t[mask_sel] = mask_obj_token_id

    # optional random corruption
    rand_sel = valid & (u >= p_keep + p_mask)
    if rand_sel.any() and node_random_corruption_prob > 0.0:
        num_obj_classes = mask_obj_token_id
        rand_vals = torch.randint(
            low=0,
            high=num_obj_classes,
            size=(rand_sel.sum().item(),),
            device=device,
        )
        obj_t[rand_sel] = rand_vals

    return obj_t
