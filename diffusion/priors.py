# diffusion/priors.py

import torch


def normalize_prob_vector(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float().clamp(min=0)
    s = x.sum()
    if s <= eps:
        raise ValueError("Probability vector has non-positive sum.")
    return x / s


@torch.no_grad()
def compute_object_prior_from_dataset(dataset, pad_token_id: int) -> torch.Tensor:
    """
    Returns empirical object prior over all object classes.
    PAD mass is zeroed out and then renormalized.
    """
    counts = torch.zeros(len(dataset.object_vocab), dtype=torch.float32)

    for i in range(len(dataset)):
        item = dataset[i]
        obj = item["obj_labels"]       # [N]
        node_mask = item["node_mask"]  # [N]

        vals = obj[node_mask]
        counts.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))

    counts[pad_token_id] = 0.0
    return normalize_prob_vector(counts)


@torch.no_grad()
def compute_relation_prior_from_dataset(dataset, no_rel_token_id: int):
    """
    Returns:
        rel_prior_all: empirical prior over all relation classes, including NO_REL
        rel_prior_nonnull: empirical prior over non-null predicates only
    """
    counts_all = torch.zeros(len(dataset.relation_vocab), dtype=torch.float32)

    for i in range(len(dataset)):
        item = dataset[i]
        rel = item["rel_labels"]       # [N, N]
        edge_mask = item["edge_mask"]  # [N, N]

        vals = rel[edge_mask]
        counts_all.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))

    rel_prior_all = normalize_prob_vector(counts_all)

    counts_nonnull = counts_all.clone()
    counts_nonnull[no_rel_token_id] = 0.0
    rel_prior_nonnull = normalize_prob_vector(counts_nonnull)

    return rel_prior_all, rel_prior_nonnull