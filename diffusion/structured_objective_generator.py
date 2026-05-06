# diffusion/structured_objective_generator.py

from typing import Dict, Any, Optional
import torch

from diffusion.sg_state_utils import build_structured_targets, build_valid_pair_mask


class StructuredSGObjectiveGenerator:
    def __init__(
        self,
        cfg,
        num_obj_classes: int,
        num_rel_classes_full: int,
        device: torch.device,
        obj_similarity_matrix: torch.Tensor = None,
    ):
        self.cfg = cfg
        self.device = device

        self.num_steps = cfg.num_diffusion_steps
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes_full = num_rel_classes_full
        self.no_rel_token_id = cfg.no_rel_token_id

        self.num_rel_pos_classes = num_rel_classes_full - 1

        self.mask_obj_token_id = num_obj_classes
        self.mask_rel_token_id = self.num_rel_pos_classes

        self.obj_similarity_matrix = obj_similarity_matrix

        self.obj_empirical_prior = None          # [K_obj]
        self.edge_empirical_prior = None         # [2]
        self.rel_empirical_prior = None          # [K_rel_state] where last index is mask-rel state

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            dtype=torch.long,
            device=self.device,
        )

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() / max(self.num_steps - 1, 1)

    # def q_sample_objects(
    #     self,
    #     obj_0: torch.Tensor,        # [B,N]
    #     node_mask: torch.Tensor,    # [B,N]
    #     t: torch.Tensor,            # [B]
    # ):
    #     """
    #     Phase 3A:
    #     Semantic discrete node corruption via a class-transition kernel.
    #     No absorbing mask token is used as the main corruption mechanism.
    #     """
    #     B, N = obj_0.shape
    #     tau = self._tau(t)[:, None]  # [B,1]

    #     valid = node_mask.bool()
    #     obj_t = obj_0.clone()
    #     obj_corrupt_mask = torch.zeros_like(valid)

    #     # In Phase 3A, no absorbing node mask token corruption by default
    #     obj_mask_token_mask = torch.zeros_like(valid)

    #     if not getattr(self.cfg, "use_semantic_node_corruption", False):
    #         raise ValueError("Phase 3A expects use_semantic_node_corruption=True.")

    #     if self.obj_similarity_matrix is None:
    #         raise ValueError("obj_similarity_matrix must be provided for semantic node corruption.")

    #     S = self.obj_similarity_matrix.to(obj_0.device)  # [K,K]
    #     K = S.shape[0]
    #     I = torch.eye(K, device=obj_0.device, dtype=S.dtype)

    #     beta = torch.clamp(self.cfg.node_corrupt_intensity * tau, max=1.0)  # [B,1]


    #     for b in range(B):
    #         idx = torch.where(valid[b])[0]
    #         if idx.numel() == 0:
    #             continue

    #         beta_b = float(beta[b, 0].item())
    #         Q_b = (1.0 - beta_b) * I + beta_b * S  # [K,K]

    #         clean_classes = obj_0[b, idx]          # [M]
    #         probs = Q_b[clean_classes]             # [M,K]

    #         sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [M]

    #         obj_t[b, idx] = sampled
    #         obj_corrupt_mask[b, idx] = (sampled != clean_classes)

    #     return obj_t.long(), obj_corrupt_mask, obj_mask_token_mask

    def q_sample_objects(
    self,
    obj_0: torch.Tensor,        # [B,N]
    node_mask: torch.Tensor,    # [B,N]
    t: torch.Tensor,            # [B]
):
        """
        Hybrid object corruption:
    mask corruption with probability node_mask_ratio
    random corruption with probability node_rand_ratio
    semantic corruption with remaining probability

        Corruption strength is controlled by beta = node_corrupt_intensity * tau(t).
        """
        B, N = obj_0.shape
        tau = self._tau(t)[:, None]  # [B,1]

        valid = node_mask.bool()
        obj_t = obj_0.clone()

        obj_corrupt_mask = torch.zeros_like(valid)
        obj_mask_token_mask = torch.zeros_like(valid)
        obj_rand_mask = torch.zeros_like(valid)
        obj_semantic_mask = torch.zeros_like(valid)

        if self.obj_similarity_matrix is None:
            raise ValueError("obj_similarity_matrix must be provided for semantic node corruption.")

        S = self.obj_similarity_matrix.to(obj_0.device)  # [K,K]
        K = S.shape[0]
        I = torch.eye(K, device=obj_0.device, dtype=S.dtype)

        beta = torch.clamp(self.cfg.node_corrupt_intensity * tau, max=1.0)  # [B,1]

        mask_ratio = float(getattr(self.cfg, "node_mask_ratio", 0.0))
        rand_ratio = float(getattr(self.cfg, "node_rand_ratio", 0.0))

        if mask_ratio < 0 or rand_ratio < 0 or mask_ratio + rand_ratio > 1.0:
            raise ValueError(
                f"Invalid node corruption ratios: "
                f"node_mask_ratio={mask_ratio}, node_rand_ratio={rand_ratio}. "
                f"They must be nonnegative and sum to <= 1."
            )

        for b in range(B):
            idx = torch.where(valid[b])[0]
            if idx.numel() == 0:
                continue

            beta_b = float(beta[b, 0].item())
            Q_b = (1.0 - beta_b) * I + beta_b * S  # [K,K]

            clean_classes = obj_0[b, idx]          # [M]
            probs = Q_b[clean_classes]             # [M,K]

            # Decide which valid nodes are actually corrupted at this timestep.
            # beta_b controls total corruption probability.
            corrupt_sel = torch.rand(
                clean_classes.shape,
                device=obj_0.device,
            ) < beta_b

            if not corrupt_sel.any():
                continue

            # Among corrupted nodes, choose corruption type.
            u = torch.rand(clean_classes.shape, device=obj_0.device)

            mask_sel = corrupt_sel & (u < mask_ratio)
            rand_sel = corrupt_sel & (u >= mask_ratio) & (u < mask_ratio + rand_ratio)
            sem_sel = corrupt_sel & ~(mask_sel | rand_sel)

            sampled = clean_classes.clone()

            # 1. Mask corruption
            if mask_sel.any():
                sampled[mask_sel] = self.mask_obj_token_id

            # 2. Random corruption
            if rand_sel.any():
                sampled[rand_sel] = torch.randint(
                    low=0,
                    high=K,
                    size=(int(rand_sel.sum().item()),),
                    device=obj_0.device,
                )

            # 3. Semantic corruption
            if sem_sel.any():
                sampled[sem_sel] = torch.multinomial(
                    probs[sem_sel],
                    num_samples=1,
                ).squeeze(-1)

            changed = sampled != clean_classes

            obj_t[b, idx] = sampled
            obj_corrupt_mask[b, idx] = changed
            obj_mask_token_mask[b, idx] = mask_sel & changed
            obj_rand_mask[b, idx] = rand_sel & changed
            obj_semantic_mask[b, idx] = sem_sel & changed

        return (
            obj_t.long(),
            obj_corrupt_mask,
            obj_mask_token_mask,
            obj_rand_mask,
            obj_semantic_mask,
        )

    def q_sample_edges(
        self,
        edge_0: torch.Tensor,       # [B,N,N], {0,1}
        pair_mask: torch.Tensor,    # [B,N,N], bool
        t: torch.Tensor,            # [B]
    ):
        B, N, _ = edge_0.shape
        tau = self._tau(t)[:, None, None].expand(B, N, N)

        p_pos_to_neg = self.cfg.edge_pos_flip_max * tau
        p_neg_to_pos = self.cfg.edge_neg_flip_max * tau

        u = torch.rand(B, N, N, device=edge_0.device)
        valid = pair_mask.bool()

        edge_t = edge_0.clone()
        edge_corrupt_mask = torch.zeros_like(valid)

        pos = valid & (edge_0 == 1)
        neg = valid & (edge_0 == 0)

        flip_pos = pos & (u < p_pos_to_neg)
        flip_neg = neg & (u < p_neg_to_pos)

        edge_t[flip_pos] = 0
        edge_t[flip_neg] = 1

        edge_corrupt_mask[flip_pos] = True
        edge_corrupt_mask[flip_neg] = True
        return edge_t.long(), edge_corrupt_mask

    def q_sample_relations(
        self,
        rel_pos_0: torch.Tensor,        # [B,N,N]
        gt_pos_edge_mask: torch.Tensor, # [B,N,N]
        t: torch.Tensor,                # [B]
    ):
        B, N, _ = rel_pos_0.shape
        tau = self._tau(t)[:, None, None].expand(B, N, N)

        rand_frac = self.cfg.rel_rand_ratio
        mask_frac = self.cfg.rel_mask_ratio
        frac_sum = rand_frac + mask_frac
        if frac_sum <= 0:
            raise ValueError("rel_mask_ratio + rel_rand_ratio must be > 0")

        rand_frac = rand_frac / frac_sum
        mask_frac = mask_frac / frac_sum

        p_corrupt = tau
        p_mask = mask_frac * p_corrupt
        p_rand = rand_frac * p_corrupt
        p_keep = 1.0 - p_corrupt

        u = torch.rand(B, N, N, device=rel_pos_0.device)
        valid = gt_pos_edge_mask.bool()

        rel_t = rel_pos_0.clone()
        rel_corrupt_mask = torch.zeros_like(valid)

        mask_sel = valid & (u >= p_keep) & (u < p_keep + p_mask)
        rand_sel = valid & (u >= p_keep + p_mask)

        rel_t[mask_sel] = self.mask_rel_token_id
        if rand_sel.any():
            rand_vals = torch.randint(
                low=0,
                high=self.num_rel_pos_classes,
                size=(rand_sel.sum().item(),),
                device=rel_pos_0.device,
            )
            rel_t[rand_sel] = rand_vals

        rel_corrupt_mask[mask_sel] = True
        rel_corrupt_mask[rand_sel] = True
        return rel_t.long(), rel_corrupt_mask

    def get_training_batch(
    self,
    batch: Dict[str, Any],
    force_t: Optional[int] = None,
) -> Dict[str, Any]:
        obj_0 = batch["obj_labels"].to(self.device)
        rel_full_0 = batch["rel_labels"].to(self.device)
        node_mask = batch["node_mask"].to(self.device).bool()
        edge_mask = batch["edge_mask"].to(self.device).bool()
        boxes_0 = batch["boxes"].to(self.device)

        pair_mask = build_valid_pair_mask(node_mask, edge_mask)

        edge_0, rel_pos_0, gt_pos_edge_mask = build_structured_targets(
            rel_full=rel_full_0,
            edge_mask=pair_mask,
            no_rel_token_id=self.no_rel_token_id,
        )

        B = obj_0.shape[0]
        if force_t is None:
            t = self.sample_timesteps(B)
        else:
            t = torch.full(
                (B,),
                int(force_t),
                dtype=torch.long,
                device=self.device,
            )

        # obj_t, obj_corrupt_mask, obj_mask_token_mask = self.q_sample_objects(obj_0, node_mask, t)
        obj_t, obj_corrupt_mask, obj_mask_token_mask, obj_rand_mask, obj_semantic_mask = (
            self.q_sample_objects(obj_0, node_mask, t)
        )

        edge_t, edge_corrupt_mask = self.q_sample_edges(edge_0, pair_mask, t)
        rel_pos_t, rel_corrupt_mask = self.q_sample_relations(rel_pos_0, gt_pos_edge_mask, t)

        if getattr(self.cfg, "object_one_node_sanity", False):
            obj_t = obj_0.clone()
            obj_corrupt_mask = torch.zeros_like(node_mask, dtype=torch.bool)
            obj_mask_token_mask = torch.zeros_like(node_mask, dtype=torch.bool)

            B, N = obj_0.shape
            for b in range(B):
                valid = torch.where(node_mask[b])[0]
                if valid.numel() == 0:
                    continue
                j = valid[torch.randint(valid.numel(), (1,), device=self.device)]
                obj_t[b, j] = self.mask_obj_token_id
                obj_corrupt_mask[b, j] = True
                obj_mask_token_mask[b, j] = True

        if getattr(self.cfg, "object_only_sanity", False):
            edge_t = edge_0.clone()
            rel_pos_t = rel_pos_0.clone()

            edge_corrupt_mask = torch.zeros_like(edge_corrupt_mask, dtype=torch.bool)
            rel_corrupt_mask = torch.zeros_like(rel_corrupt_mask, dtype=torch.bool)

        # relation tokens only meaningful where edge_t==1
        rel_pos_t = torch.where(edge_t.bool(), rel_pos_t, torch.zeros_like(rel_pos_t))

        return {
            "obj_0": obj_0,
            "rel_full_0": rel_full_0,
            "edge_0": edge_0,
            "rel_pos_0": rel_pos_0,

            "boxes_0":boxes_0,

            "obj_t": obj_t,
            "edge_t": edge_t,
            "rel_pos_t": rel_pos_t,

            "obj_corrupt_mask": obj_corrupt_mask,
            "obj_mask_token_mask": obj_mask_token_mask,
            "obj_rand_mask": obj_rand_mask,
            "obj_semantic_mask": obj_semantic_mask,
            "edge_corrupt_mask": edge_corrupt_mask,
            "rel_corrupt_mask": rel_corrupt_mask,

            "gt_pos_edge_mask": gt_pos_edge_mask,
            "node_mask": node_mask,
            "edge_mask": pair_mask,
            "t": t,
        }
    
    @torch.no_grad()
    def sample_graph_at_t_from_clean(
        self,
        batch_clean: Dict[str, Any],
        t: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Reuse the exact same forward corruption process on an explicit clean graph batch.

        batch_clean must contain:
        obj_0      [B,N]
        edge_0     [B,N,N]
        rel_pos_0  [B,N,N]
        node_mask  [B,N]
        edge_mask  [B,N,N]
        """
        obj_0 = batch_clean["obj_0"].to(self.device)
        edge_0 = batch_clean["edge_0"].to(self.device)
        rel_pos_0 = batch_clean["rel_pos_0"].to(self.device)
        node_mask = batch_clean["node_mask"].to(self.device).bool()
        edge_mask = batch_clean["edge_mask"].to(self.device).bool()
        t = t.to(self.device).long()

        # forward corruption exactly as in get_training_batch(...)
        obj_t, obj_corrupt_mask, obj_mask_token_mask, obj_rand_mask, obj_semantic_mask = self.q_sample_objects(obj_0, node_mask, t)
        edge_t, edge_corrupt_mask = self.q_sample_edges(edge_0, edge_mask, t)

        # relations only meaningful on positive clean edges for corruption
        gt_pos_edge_mask = edge_0.bool() & edge_mask
        rel_pos_t, rel_corrupt_mask = self.q_sample_relations(rel_pos_0, gt_pos_edge_mask, t)

        # relation tokens only meaningful where edge_t == 1
        rel_pos_t = torch.where(edge_t.bool(), rel_pos_t, torch.zeros_like(rel_pos_t))

        return {
            "obj_0": obj_0,
            "edge_0": edge_0,
            "rel_pos_0": rel_pos_0,

            "obj_t": obj_t,
            "edge_t": edge_t,
            "rel_pos_t": rel_pos_t,

            "obj_corrupt_mask": obj_corrupt_mask,
            "obj_mask_token_mask": obj_mask_token_mask,
            "edge_corrupt_mask": edge_corrupt_mask,
            "obj_rand_mask": obj_rand_mask,
            "obj_semantic_mask": obj_semantic_mask,
            "rel_corrupt_mask": rel_corrupt_mask,

            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "t": t,
        }

    def _ensure_t_tensor(self, t) -> torch.Tensor:
        if torch.is_tensor(t):
            return t.to(self.device).long().view(-1)
        return torch.as_tensor([t], device=self.device, dtype=torch.long)

    def _project_row_stochastic(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix or batch of matrices to valid row-stochastic form.
        Q: [K,K] or [B,K,K]
        """
        squeeze_back = False
        if Q.dim() == 2:
            Q = Q.unsqueeze(0)
            squeeze_back = True

        B, K, _ = Q.shape
        Q = torch.clamp(Q, min=0.0)
        row_sum = Q.sum(dim=-1, keepdim=True)  # [B,K,1]

        # Normalize nonzero rows
        Q = Q / row_sum.clamp(min=1e-12)

        # If any row is numerically zero, replace with identity row
        zero_row_mask = (row_sum.squeeze(-1) <= 1e-12)  # [B,K]
        if zero_row_mask.any():
            eye = torch.eye(K, device=Q.device, dtype=Q.dtype).unsqueeze(0).expand(B, K, K)
            for b in range(B):
                if zero_row_mask[b].any():
                    Q[b, zero_row_mask[b]] = eye[b, zero_row_mask[b]]

        if squeeze_back:
            Q = Q.squeeze(0)
        return Q

    def _derive_adjacent_kernel_from_cumulative(
        self,
        Qbar_prev: torch.Tensor,   # [B,K,K]
        Qbar_t: torch.Tensor,      # [B,K,K]
    ) -> torch.Tensor:
        """
        Derive an adjacent-step kernel Q_t from cumulative kernels by solving:
            Qbar_t ≈ Qbar_prev @ Q_t
        then projecting to row-stochastic form.
        """
        if Qbar_prev.dim() == 2:
            Qbar_prev = Qbar_prev.unsqueeze(0)
        if Qbar_t.dim() == 2:
            Qbar_t = Qbar_t.unsqueeze(0)

        B, K, _ = Qbar_prev.shape
        Q_t = []

        for b in range(B):
            prev = Qbar_prev[b]   # [K,K]
            cur = Qbar_t[b]       # [K,K]

            prev_pinv = torch.linalg.pinv(prev)
            q = prev_pinv @ cur
            q = self._project_row_stochastic(q)
            Q_t.append(q)

        return torch.stack(Q_t, dim=0)
    
    # ============================================================
    # 5B.1: OBJECT KERNELS
    # ============================================================

    def get_object_Qbar_t(self, t) -> torch.Tensor:
        """
        Exact cumulative object kernel matching q_sample_objects:
            Qbar_obj(t) = (1 - beta_t) I + beta_t S
        Returns:
            [B,K,K] if t is batched, else [1,K,K]
        """
        t = self._ensure_t_tensor(t)
        tau = self._tau(t)  # [B]
        beta = torch.clamp(self.cfg.node_corrupt_intensity * tau, max=1.0)  # [B]

        if self.obj_similarity_matrix is None:
            raise ValueError("obj_similarity_matrix must be provided for object kernels.")

        S = self.obj_similarity_matrix.to(self.device).float()   # [K,K]
        K = S.shape[0]
        I = torch.eye(K, device=self.device, dtype=torch.float32).unsqueeze(0)  # [1,K,K]

        Qbar = (1.0 - beta)[:, None, None] * I + beta[:, None, None] * S.unsqueeze(0)
        Qbar = self._project_row_stochastic(Qbar)
        return Qbar

    def get_object_Q_t(self, t) -> torch.Tensor:
        """
        Approximate adjacent object kernel derived from cumulative family.
        Returns:
            [B,K,K]
        """
        t = self._ensure_t_tensor(t)
        B = t.shape[0]

        Qbar_t = self.get_object_Qbar_t(t)
        Q_t = []

        K = self.num_obj_classes
        I = torch.eye(K, device=self.device, dtype=torch.float32)

        for b in range(B):
            if int(t[b].item()) == 0:
                Q_t.append(I)
            else:
                t_prev = torch.as_tensor([int(t[b].item()) - 1], device=self.device, dtype=torch.long)
                Qbar_prev = self.get_object_Qbar_t(t_prev)[0]
                q = self._derive_adjacent_kernel_from_cumulative(Qbar_prev, Qbar_t[b])[0]
                Q_t.append(q)

        return torch.stack(Q_t, dim=0)

    # ============================================================
    # 5B.1: EDGE KERNELS
    # ============================================================

    def get_edge_Qbar_t(self, t) -> torch.Tensor:
        """
        Exact cumulative edge kernel matching q_sample_edges.

        State order:
            0 = no edge
            1 = edge
        Returns:
            [B,2,2]
        """
        t = self._ensure_t_tensor(t)
        tau = self._tau(t)  # [B]

        p10 = self.cfg.edge_pos_flip_max * tau   # edge -> no edge
        p01 = self.cfg.edge_neg_flip_max * tau   # no edge -> edge

        B = t.shape[0]
        Qbar = torch.zeros(B, 2, 2, device=self.device, dtype=torch.float32)

        # row 0 = source no-edge
        Qbar[:, 0, 0] = 1.0 - p01
        Qbar[:, 0, 1] = p01

        # row 1 = source edge
        Qbar[:, 1, 0] = p10
        Qbar[:, 1, 1] = 1.0 - p10

        Qbar = self._project_row_stochastic(Qbar)
        return Qbar

    def get_edge_Q_t(self, t) -> torch.Tensor:
        """
        Approximate adjacent edge kernel derived from cumulative family.
        Returns:
            [B,2,2]
        """
        t = self._ensure_t_tensor(t)
        B = t.shape[0]

        Qbar_t = self.get_edge_Qbar_t(t)
        Q_t = []

        I = torch.eye(2, device=self.device, dtype=torch.float32)

        for b in range(B):
            if int(t[b].item()) == 0:
                Q_t.append(I)
            else:
                t_prev = torch.as_tensor([int(t[b].item()) - 1], device=self.device, dtype=torch.long)
                Qbar_prev = self.get_edge_Qbar_t(t_prev)[0]
                q = self._derive_adjacent_kernel_from_cumulative(Qbar_prev, Qbar_t[b])[0]
                Q_t.append(q)

        return torch.stack(Q_t, dim=0)

    # ============================================================
    # 5B.1: RELATION KERNELS
    # ============================================================

    def get_relation_Qbar_t(self, t) -> torch.Tensor:
        """
        Exact cumulative positive-relation/mask kernel matching q_sample_relations.

        State order:
            0 .. K_pos-1 = positive relation classes
            K_pos        = mask relation token

        Note:
            This kernel is only for the relation state *conditional on edge existing*.
            If edge=0, relation should be forced to no-relation externally.
        Returns:
            [B, K_pos+1, K_pos+1]
        """
        t = self._ensure_t_tensor(t)
        tau = self._tau(t)  # [B]

        rand_frac = self.cfg.rel_rand_ratio
        mask_frac = self.cfg.rel_mask_ratio
        frac_sum = rand_frac + mask_frac
        if frac_sum <= 0:
            raise ValueError("rel_mask_ratio + rel_rand_ratio must be > 0")

        rand_frac = rand_frac / frac_sum
        mask_frac = mask_frac / frac_sum

        p_corrupt = tau
        p_mask = mask_frac * p_corrupt
        p_rand = rand_frac * p_corrupt
        p_keep = 1.0 - p_corrupt

        B = t.shape[0]
        Kp = self.num_rel_pos_classes
        Ks = Kp + 1
        mask_idx = self.mask_rel_token_id  # should equal Kp

        Qbar = torch.zeros(B, Ks, Ks, device=self.device, dtype=torch.float32)

        # rows 0..Kp-1 are clean positive relation sources
        base_rand = (p_rand / float(Kp)).unsqueeze(-1)  # [B,1]

        for b in range(B):
            # all positive-source rows get uniform random mass over positive states
            Qbar[b, :Kp, :Kp] = base_rand[b].item()

            # add keep mass to the diagonal
            diag = torch.arange(Kp, device=self.device)
            Qbar[b, diag, diag] += p_keep[b]

            # add mask mass
            Qbar[b, :Kp, mask_idx] = p_mask[b]

            # absorbing extension for mask row
            Qbar[b, mask_idx, mask_idx] = 1.0

        Qbar = self._project_row_stochastic(Qbar)
        return Qbar

    def get_relation_Q_t(self, t) -> torch.Tensor:
        """
        Approximate adjacent relation kernel derived from cumulative family.
        Returns:
            [B, K_pos+1, K_pos+1]
        """
        t = self._ensure_t_tensor(t)
        B = t.shape[0]

        Qbar_t = self.get_relation_Qbar_t(t)
        Q_t = []

        Ks = self.num_rel_pos_classes + 1
        I = torch.eye(Ks, device=self.device, dtype=torch.float32)

        for b in range(B):
            if int(t[b].item()) == 0:
                Q_t.append(I)
            else:
                t_prev = torch.as_tensor([int(t[b].item()) - 1], device=self.device, dtype=torch.long)
                Qbar_prev = self.get_relation_Qbar_t(t_prev)[0]
                q = self._derive_adjacent_kernel_from_cumulative(Qbar_prev, Qbar_t[b])[0]
                Q_t.append(q)

        return torch.stack(Q_t, dim=0)
    
    def get_all_kernels_t(self, t):
        """
        Convenience accessor for 5B sampler code.
        """
        return {
            "Qbar_obj_t": self.get_object_Qbar_t(t),
            "Q_obj_t": self.get_object_Q_t(t),

            "Qbar_edge_t": self.get_edge_Qbar_t(t),
            "Q_edge_t": self.get_edge_Q_t(t),

            "Qbar_rel_t": self.get_relation_Qbar_t(t),
            "Q_rel_t": self.get_relation_Q_t(t),
        }
    
    def get_training_batch_pair(
        self,
        batch: Dict[str, Any],
        force_t: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Phase 5C-full-1

        Build a matched pair:
            batch_t    ~ q(x_t   | x_0)
            batch_prev ~ q(x_t-1 | x_0)

        using the SAME clean graph and the SAME timestep pair.
        """

        obj_0 = batch["obj_labels"].to(self.device)
        rel_full_0 = batch["rel_labels"].to(self.device)
        node_mask = batch["node_mask"].to(self.device).bool()
        edge_mask = batch["edge_mask"].to(self.device).bool()
        boxes_0 = batch["boxes"].to(self.device)

        pair_mask = build_valid_pair_mask(node_mask, edge_mask)

        edge_0, rel_pos_0, gt_pos_edge_mask = build_structured_targets(
            rel_full=rel_full_0,
            edge_mask=pair_mask,
            no_rel_token_id=self.no_rel_token_id,
        )

        B = obj_0.shape[0]
        if force_t is None:
            t = self.sample_timesteps(B)
        else:
            t = torch.full(
                (B,),
                int(force_t),
                dtype=torch.long,
                device=self.device,
            )

        t_prev = torch.clamp(t - 1, min=0)

        # -------------------------
        # sample x_t
        # -------------------------
        # obj_t, obj_corrupt_mask, obj_mask_token_mask = self.q_sample_objects(
        #     obj_0, node_mask, t
        # )
        obj_t, obj_corrupt_mask, obj_mask_token_mask, obj_rand_mask, obj_semantic_mask = (
            self.q_sample_objects(obj_0, node_mask, t)
        )
        edge_t, edge_corrupt_mask = self.q_sample_edges(
            edge_0, pair_mask, t
        )
        rel_pos_t, rel_corrupt_mask = self.q_sample_relations(
            rel_pos_0, gt_pos_edge_mask, t
        )

        if getattr(self.cfg, "object_one_node_sanity", False):
            obj_t = obj_0.clone()
            obj_corrupt_mask = torch.zeros_like(node_mask, dtype=torch.bool)
            obj_mask_token_mask = torch.zeros_like(node_mask, dtype=torch.bool)

            B, N = obj_0.shape
            for b in range(B):
                valid = torch.where(node_mask[b])[0]
                if valid.numel() == 0:
                    continue
                j = valid[torch.randint(valid.numel(), (1,), device=self.device)]
                obj_t[b, j] = self.mask_obj_token_id
                obj_corrupt_mask[b, j] = True
                obj_mask_token_mask[b, j] = True

        if getattr(self.cfg, "object_only_sanity", False):
            edge_t = edge_0.clone()
            rel_pos_t = rel_pos_0.clone()

            edge_corrupt_mask = torch.zeros_like(edge_corrupt_mask, dtype=torch.bool)
            rel_corrupt_mask = torch.zeros_like(rel_corrupt_mask, dtype=torch.bool)

        # relations only meaningful where sampled edge_t == 1
        rel_pos_t = torch.where(edge_t.bool(), rel_pos_t, torch.zeros_like(rel_pos_t))

        batch_t = {
            "obj_0": obj_0,
            "rel_full_0": rel_full_0,
            "edge_0": edge_0,
            "rel_pos_0": rel_pos_0,

            "boxes_0":boxes_0,

            "obj_t": obj_t,
            "edge_t": edge_t,
            "rel_pos_t": rel_pos_t,

            "obj_corrupt_mask": obj_corrupt_mask,
            "obj_mask_token_mask": obj_mask_token_mask,
            "obj_rand_mask": obj_rand_mask,
            "obj_semantic_mask": obj_semantic_mask,
            "edge_corrupt_mask": edge_corrupt_mask,
            "rel_corrupt_mask": rel_corrupt_mask,

            "gt_pos_edge_mask": gt_pos_edge_mask,
            "node_mask": node_mask,
            "edge_mask": pair_mask,
            "t": t,
        }

        # -------------------------
        # sample x_{t-1}
        # -------------------------
        # obj_prev, obj_corrupt_mask_prev, obj_mask_token_mask_prev = self.q_sample_objects(
        #     obj_0, node_mask, t_prev
        # )
        obj_prev, obj_corrupt_mask_prev, obj_mask_token_mask_prev, obj_rand_mask_prev, obj_semantic_mask_prev = (
            self.q_sample_objects(obj_0, node_mask, t_prev)
        )
        edge_prev, edge_corrupt_mask_prev = self.q_sample_edges(
            edge_0, pair_mask, t_prev
        )
        rel_pos_prev, rel_corrupt_mask_prev = self.q_sample_relations(
            rel_pos_0, gt_pos_edge_mask, t_prev
        )

        if getattr(self.cfg, "object_only_sanity", False):
            edge_prev = edge_0.clone()
            rel_pos_prev = rel_pos_0.clone()

            edge_corrupt_mask_prev = torch.zeros_like(edge_corrupt_mask_prev, dtype=torch.bool)
            rel_corrupt_mask_prev = torch.zeros_like(rel_corrupt_mask_prev, dtype=torch.bool)

        # relations only meaningful where sampled edge_prev == 1
        rel_pos_prev = torch.where(edge_prev.bool(), rel_pos_prev, torch.zeros_like(rel_pos_prev))

        batch_prev = {
            "obj_0": obj_0,
            "rel_full_0": rel_full_0,
            "edge_0": edge_0,
            "rel_pos_0": rel_pos_0,

            "obj_t": obj_prev,
            "edge_t": edge_prev,
            "rel_pos_t": rel_pos_prev,

            "obj_corrupt_mask": obj_corrupt_mask_prev,
            "obj_mask_token_mask": obj_mask_token_mask_prev,
            "obj_rand_mask": obj_rand_mask_prev,
            "obj_semantic_mask": obj_semantic_mask_prev,
            "edge_corrupt_mask": edge_corrupt_mask_prev,
            "rel_corrupt_mask": rel_corrupt_mask_prev,

            "gt_pos_edge_mask": gt_pos_edge_mask,
            "node_mask": node_mask,
            "edge_mask": pair_mask,
            "t": t_prev,
        }

        return {
            "batch_t": batch_t,
            "batch_prev": batch_prev,
        }

    @torch.no_grad()
    def build_clean_state_batch(self, batch):
        """
        Build the clean graph state as a batch dict in the same format as get_training_batch.
        Useful for t=0 or analysis/debugging.
        """
        obj_0 = batch["obj_labels"].to(self.device)
        rel_full_0 = batch["rel_labels"].to(self.device)
        node_mask = batch["node_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device)

        edge_0 = (rel_full_0 != self.no_rel_token_id).long()
        rel_pos_0 = torch.clamp(rel_full_0 - 1, min=0)

        B = obj_0.shape[0]
        t0 = torch.zeros(B, device=self.device, dtype=torch.long)

        return {
            "obj_0": obj_0,
            "edge_0": edge_0,
            "rel_pos_0": rel_pos_0,
            "obj_t": obj_0.clone(),
            "edge_t": edge_0.clone(),
            "rel_pos_t": rel_pos_0.clone(),
            "t": t0,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
        }
    
    @torch.no_grad()
    def set_empirical_unconditional_priors(
        self,
        obj_prior: torch.Tensor,   # [K_obj]
        edge_prior: torch.Tensor,  # [2]
        rel_prior: torch.Tensor,   # [K_rel_state]
    ):
        self.obj_empirical_prior = obj_prior.detach().to(self.device)
        self.edge_empirical_prior = edge_prior.detach().to(self.device)
        self.rel_empirical_prior = rel_prior.detach().to(self.device)