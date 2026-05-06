# diffusion/node_semantic_kernel.py

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_object_context_features(
    dataset,
    num_obj_classes: int,
    num_rel_pos_classes: int,
    no_rel_token_id: int,
) -> torch.Tensor:
    """
    Build a context feature vector for each object class using:
      - outgoing relation histogram
      - incoming relation histogram
      - neighbor object histogram

    Returns:
        feat: [K_obj, 2*K_rel_pos + K_obj]
    """
    obj_rel_out = torch.zeros(num_obj_classes, num_rel_pos_classes, dtype=torch.float32)
    obj_rel_in = torch.zeros(num_obj_classes, num_rel_pos_classes, dtype=torch.float32)
    obj_neighbor = torch.zeros(num_obj_classes, num_obj_classes, dtype=torch.float32)

    for i in range(len(dataset)):
        item = dataset[i]

        obj = item["obj_labels"]            # [N]
        rel = item["rel_labels"]            # [N,N]
        node_mask = item["node_mask"].bool()
        edge_mask = item["edge_mask"].bool()

        valid_nodes = torch.where(node_mask)[0]
        if valid_nodes.numel() == 0:
            continue

        for s in valid_nodes.tolist():
            c_s = int(obj[s])

            for t in valid_nodes.tolist():
                if not edge_mask[s, t]:
                    continue

                r = int(rel[s, t])
                if r == no_rel_token_id:
                    continue

                # full rel id -> positive-only rel id
                r_pos = r - 1 if r > no_rel_token_id else r
                c_t = int(obj[t])

                obj_rel_out[c_s, r_pos] += 1.0
                obj_rel_in[c_t, r_pos] += 1.0
                obj_neighbor[c_s, c_t] += 1.0

    feat = torch.cat([obj_rel_out, obj_rel_in, obj_neighbor], dim=-1)
    feat = F.normalize(feat, p=2, dim=-1)
    return feat


@torch.no_grad()
def build_similarity_matrix_from_features(
    feat: torch.Tensor,
    temperature: float = 0.1,
    self_bias: float = 0.0,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a row-stochastic similarity / transition-support matrix from class features.

    Args:
        feat: [K_obj, D_feat], assumed normalized or at least comparable
        temperature: smaller => sharper similarities
        self_bias: additive bias on the diagonal before softmax
        topk: if provided, keep only top-k neighbors per row before softmax

    Returns:
        sim: [K_obj, K_obj], row-stochastic
    """
    sim = feat @ feat.t()  # cosine-like if feat normalized
    sim = sim / max(temperature, 1e-8)

    K = sim.shape[0]
    if self_bias != 0.0:
        sim = sim + torch.eye(K, dtype=sim.dtype) * self_bias

    if topk is not None and topk < K:
        values, indices = torch.topk(sim, k=topk, dim=-1)
        masked = torch.full_like(sim, fill_value=-1e9)
        masked.scatter_(dim=-1, index=indices, src=values)
        sim = masked

    sim = torch.softmax(sim, dim=-1)
    return sim


def build_node_transition_matrix(
    sim_matrix: torch.Tensor,
    beta_t: float,
) -> torch.Tensor:
    """
    Q_t = (1 - beta_t) I + beta_t S
    where S is the semantic similarity transition matrix.
    """
    K = sim_matrix.shape[0]
    I = torch.eye(K, device=sim_matrix.device, dtype=sim_matrix.dtype)
    return (1.0 - beta_t) * I + beta_t * sim_matrix