# # diffusion/transitions.py

from dataclasses import dataclass
from typing import Optional

import torch

from configs import DiscreteSGConfig


# # def normalize_prob_vector(x: torch.Tensor) -> torch.Tensor:
# #     x = x.float()
# #     x = x.clamp(min=0)
# #     s = x.sum()
# #     if s <= 0:
# #         raise ValueError("Probability vector has non-positive sum.")
# #     return x / s


# # def make_uniform_prior(num_classes: int, device: torch.device = None) -> torch.Tensor:
# #     return torch.full((num_classes,), 1.0 / num_classes, device=device, dtype=torch.float32)


# # def make_empirical_prior(
# #     counts: torch.Tensor,
# #     device: torch.device = None,
# #     eps: float = 1e-8,
# # ) -> torch.Tensor:
# #     counts = counts.to(device=device, dtype=torch.float32)
# #     return normalize_prob_vector(counts + eps)


# # class CategoricalTransitionKernel:
# #     """
# #     Builds one-step transition matrices:
# #         Q_t = (1 - beta_t) I + beta_t U
# #     where U is a rank-1 row-constant matrix induced by a prior vector.

# #     Optionally makes one special token absorbing (e.g. PAD).
# #     """

# #     def __init__(
# #         self,
# #         num_classes: int,
# #         prior_probs: torch.Tensor,
# #         special_absorbing_id: Optional[int] = None,
# #         device: torch.device = None,
# #     ):
# #         self.num_classes = num_classes
# #         self.device = device if device is not None else prior_probs.device
# #         self.prior_probs = normalize_prob_vector(prior_probs).to(self.device)
# #         self.special_absorbing_id = special_absorbing_id

# #         self.identity = torch.eye(num_classes, device=self.device, dtype=torch.float32)

# #     def get_Q_t(self, beta_t: torch.Tensor) -> torch.Tensor:
# #         """
# #         beta_t: scalar tensor
# #         returns: [K, K]
# #         """
# #         U = self.prior_probs.unsqueeze(0).repeat(self.num_classes, 1)  # [K, K]
# #         Q_t = (1.0 - beta_t) * self.identity + beta_t * U

# #         if self.special_absorbing_id is not None:
# #             i = self.special_absorbing_id
# #             Q_t[i, :] = 0.0
# #             Q_t[i, i] = 1.0

# #         return Q_t

# #     def get_Q_bar(self, betas: torch.Tensor, t_index: int) -> torch.Tensor:
# #         """
# #         Compute cumulative transition:
# #             Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t
# #         using 0-based indexing for t_index.
# #         returns [K, K]
# #         """
# #         Q_bar = torch.eye(self.num_classes, device=self.device, dtype=torch.float32)
# #         for s in range(t_index + 1):
# #             Q_s = self.get_Q_t(betas[s])
# #             Q_bar = Q_bar @ Q_s
# #         return Q_bar


# # def sample_categorical_from_probs(probs: torch.Tensor) -> torch.Tensor:
# #     """
# #     probs: [..., K]
# #     returns integer samples: [...]
# #     """
# #     flat = probs.reshape(-1, probs.shape[-1])
# #     samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
# #     return samples.reshape(probs.shape[:-1])


# # def apply_transition_to_labels(
# #     x0: torch.Tensor,
# #     Q_bar_t: torch.Tensor,
# # ) -> torch.Tensor:
# #     """
# #     x0: arbitrary shape of integer labels
# #     Q_bar_t: [K, K]
# #     returns xt with same shape as x0
# #     """
# #     probs = Q_bar_t[x0]  # [..., K]
# #     return sample_categorical_from_probs(probs)



# def normalize_prob_vector(x: torch.Tensor) -> torch.Tensor:
#     x = x.float().clamp(min=0)
#     s = x.sum()
#     if s <= 0:
#         raise ValueError("Probability vector has non-positive sum.")
#     return x / s


# def make_uniform_prior(num_classes: int, device=None) -> torch.Tensor:
#     return torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32, device=device)


# class CategoricalTransitionKernel:
#     def __init__(
#         self,
#         num_classes: int,
#         prior_probs: torch.Tensor,
#         special_absorbing_id: Optional[int] = None,
#         no_rel_token_id: Optional[int] = None,
#         no_rel_leak_scale: float = 0.01,
#         use_sticky_no_rel: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         self.num_classes = num_classes
#         self.device = device if device is not None else prior_probs.device
#         self.prior_probs = normalize_prob_vector(prior_probs).to(self.device)
#         self.special_absorbing_id = special_absorbing_id

#         self.no_rel_token_id = no_rel_token_id
#         self.no_rel_leak_scale = no_rel_leak_scale
#         self.use_sticky_no_rel = use_sticky_no_rel

#         self.identity = torch.eye(num_classes, dtype=torch.float32, device=self.device)

#         # prior over non-null classes only
#         if no_rel_token_id is not None:
#             nonnull = self.prior_probs.clone()
#             nonnull[no_rel_token_id] = 0.0
#             self.nonnull_prior_probs = normalize_prob_vector(nonnull)
#         else:
#             self.nonnull_prior_probs = self.prior_probs.clone()

#     def get_Q_t(self, beta_t: torch.Tensor) -> torch.Tensor:
#         """
#         beta_t: scalar tensor
#         returns Q_t of shape [K, K]
#         """
#         U = self.prior_probs.unsqueeze(0).repeat(self.num_classes, 1)  # [K, K]
#         Q_t = (1.0 - beta_t) * self.identity + beta_t * U

#         # absorbing token, e.g. PAD for objects
#         if self.special_absorbing_id is not None:
#             i = self.special_absorbing_id
#             Q_t[i, :] = 0.0
#             Q_t[i, i] = 1.0

#         # sticky NO_REL for relations
#         if self.use_sticky_no_rel and self.no_rel_token_id is not None:
#             i = self.no_rel_token_id
#             eps_t = self.no_rel_leak_scale * beta_t

#             Q_t[i, :] = eps_t * self.nonnull_prior_probs
#             Q_t[i, i] = 1.0 - eps_t

#         # numerical cleanup
#         Q_t = Q_t / Q_t.sum(dim=-1, keepdim=True)
#         return Q_t

#     def precompute_Q_bars(self, betas: torch.Tensor) -> torch.Tensor:
#         """
#         betas: [T]
#         returns Q_bars: [T, K, K]
#         """
#         T = betas.shape[0]
#         Q_bars = []
#         Q_bar = torch.eye(self.num_classes, dtype=torch.float32, device=self.device)
#         for t in range(T):
#             Q_t = self.get_Q_t(betas[t])
#             Q_bar = Q_bar @ Q_t
#             Q_bars.append(Q_bar.clone())
#         return torch.stack(Q_bars, dim=0)  # [T, K, K]


# def sample_categorical_from_probs(probs: torch.Tensor) -> torch.Tensor:
#     flat = probs.reshape(-1, probs.shape[-1])
#     samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
#     return samples.reshape(probs.shape[:-1])


# def apply_transition_to_labels(x0: torch.Tensor, Q_bar_t: torch.Tensor) -> torch.Tensor:
#     probs = Q_bar_t[x0]   # [..., K]
#     return sample_categorical_from_probs(probs)




def normalize_prob_vector(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float().clamp(min=0)
    s = x.sum()
    if s <= eps:
        raise ValueError("Probability vector has non-positive sum.")
    return x / s


def make_uniform_prior(num_classes: int, device=None) -> torch.Tensor:
    return torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32, device=device)


def sample_categorical_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [..., K]
    returns: [...]
    """
    flat = probs.reshape(-1, probs.shape[-1])
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(probs.shape[:-1])


def apply_transition_to_labels(x0: torch.Tensor, Q_bar_t: torch.Tensor) -> torch.Tensor:
    """
    x0: arbitrary integer tensor with values in [0, K-1]
    Q_bar_t: [K, K]
    """
    probs = Q_bar_t[x0]   # [..., K]
    return sample_categorical_from_probs(probs)


class CategoricalTransitionKernel:
    """
    One-step categorical transition:
        Q_t = (1 - beta_t) I + beta_t U

    Supports:
    - absorbing special token (e.g. PAD for objects)
    - sticky NO_REL row for relations
    """

    def __init__(
        self,
        num_classes: int,
        prior_probs: torch.Tensor,
        special_absorbing_id: Optional[int] = None,
        no_rel_token_id: Optional[int] = None,
        no_rel_leak_scale: float = 0.01,
        use_sticky_no_rel: bool = False,
        nonnull_prior_probs: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.num_classes = num_classes
        self.device = device if device is not None else prior_probs.device

        self.prior_probs = normalize_prob_vector(prior_probs).to(self.device)
        self.special_absorbing_id = special_absorbing_id

        self.no_rel_token_id = no_rel_token_id
        self.no_rel_leak_scale = no_rel_leak_scale
        self.use_sticky_no_rel = use_sticky_no_rel

        self.identity = torch.eye(num_classes, dtype=torch.float32, device=self.device)

        if nonnull_prior_probs is not None:
            self.nonnull_prior_probs = normalize_prob_vector(nonnull_prior_probs).to(self.device)
        elif no_rel_token_id is not None:
            tmp = self.prior_probs.clone()
            tmp[no_rel_token_id] = 0.0
            self.nonnull_prior_probs = normalize_prob_vector(tmp)
        else:
            self.nonnull_prior_probs = self.prior_probs.clone()

    def get_Q_t(self, beta_t: torch.Tensor) -> torch.Tensor:
        """
        beta_t: scalar tensor
        returns Q_t: [K, K]
        """
        U = self.prior_probs.unsqueeze(0).repeat(self.num_classes, 1)   # [K, K]
        Q_t = (1.0 - beta_t) * self.identity + beta_t * U

        # absorbing special token, e.g. PAD
        if self.special_absorbing_id is not None:
            i = self.special_absorbing_id
            Q_t[i, :] = 0.0
            Q_t[i, i] = 1.0

        # sticky NO_REL row
        if self.use_sticky_no_rel and self.no_rel_token_id is not None:
            i = self.no_rel_token_id
            eps_t = self.no_rel_leak_scale * beta_t

            Q_t[i, :] = eps_t * self.nonnull_prior_probs
            Q_t[i, i] = 1.0 - eps_t

        Q_t = Q_t / Q_t.sum(dim=-1, keepdim=True)
        return Q_t

    def precompute_Q_bars(self, betas: torch.Tensor) -> torch.Tensor:
        """
        betas: [T]
        returns Q_bars: [T, K, K]
        where Q_bars[t] = Q_0 @ Q_1 @ ... @ Q_t
        """
        T = betas.shape[0]
        Q_bars = []
        Q_bar = torch.eye(self.num_classes, dtype=torch.float32, device=self.device)

        for t in range(T):
            Q_t = self.get_Q_t(betas[t])
            Q_bar = Q_bar @ Q_t
            Q_bars.append(Q_bar.clone())

        return torch.stack(Q_bars, dim=0)