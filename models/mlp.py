import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: [B]
    returns: [B, dim]
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeAwareMPBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = MLP(2 * d_model, d_model, d_model, dropout=dropout)
        self.upd_mlp = MLP(2 * d_model, d_model, d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, rel_emb: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, D]
        rel_emb: [B, N, N, D]  (edge from i -> j stored at [i,j])
        edge_mask: [B, N, N]
        """
        B, N, D = h.shape

        # sender features h_j for messages j -> i
        h_sender = h.unsqueeze(1).expand(B, N, N, D)   # receiver i, sender j? not yet
        h_sender = h.unsqueeze(2).expand(B, N, N, D)   # [B, i, j, D] using sender j after transpose below
        h_sender = h.unsqueeze(1).expand(B, N, N, D)   # let's use rel_emb[:, j, i] below instead

        # Build messages from j -> i using edge (j, i)
        h_j = h.unsqueeze(1).expand(B, N, N, D)              # [B, i, j, D] where last N is sender j
        e_ji = rel_emb.transpose(1, 2)                       # [B, i, j, D], edge j->i
        msg_in = torch.cat([h_j, e_ji], dim=-1)              # [B, i, j, 2D]
        msgs = self.msg_mlp(msg_in)                          # [B, i, j, D]

        mask = edge_mask.transpose(1, 2).unsqueeze(-1).float()  # [B, i, j, 1], valid j->i
        msgs = msgs * mask

        agg = msgs.sum(dim=2)                                # [B, i, D]

        upd_in = torch.cat([h, agg], dim=-1)
        delta = self.upd_mlp(upd_in)
        h = self.norm(h + delta)
        return h


class SceneGraphDenoiser(nn.Module):
    def __init__(
        self,
        num_obj_classes: int,
        num_rel_classes: int,
        d_model: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.d_model = d_model

        self.obj_embedding = nn.Embedding(num_obj_classes, d_model)
        self.rel_embedding = nn.Embedding(num_rel_classes, d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            EdgeAwareMPBlock(d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.obj_head = nn.Linear(d_model, num_obj_classes)
        self.rel_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_rel_classes),
        )

    def forward(
        self,
        obj_t: torch.Tensor,
        rel_t: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ):
        """
        obj_t: [B, N]
        rel_t: [B, N, N]
        t: [B]
        node_mask: [B, N]
        edge_mask: [B, N, N]
        """
        B, N = obj_t.shape

        obj_emb = self.obj_embedding(obj_t)                  # [B, N, D]
        rel_emb = self.rel_embedding(rel_t)                  # [B, N, N, D]

        t_emb = sinusoidal_timestep_embedding(t, self.d_model)
        t_emb = self.time_mlp(t_emb)                         # [B, D]

        h = obj_emb + t_emb[:, None, :]

        # zero padded nodes for safety
        h = h * node_mask.unsqueeze(-1).float()

        for layer in self.layers:
            h = layer(h, rel_emb, edge_mask)
            h = h * node_mask.unsqueeze(-1).float()

        obj_logits = self.obj_head(h)                        # [B, N, K_obj]

        hi = h.unsqueeze(2).expand(B, N, N, self.d_model)
        hj = h.unsqueeze(1).expand(B, N, N, self.d_model)
        pair_feat = torch.cat([hi, hj, rel_emb], dim=-1)    # [B, N, N, 3D]
        rel_logits = self.rel_head(pair_feat)                # [B, N, N, K_rel]

        return {
            "obj_logits": obj_logits,
            "rel_logits": rel_logits,
        }