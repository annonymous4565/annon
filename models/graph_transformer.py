import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EdgeAwareSelfAttention(nn.Module):
    """
    Node self-attention with additive edge bias derived from relation embeddings.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.edge_bias_proj = nn.Linear(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        rel_emb: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:        [B, N, D]
        rel_emb:  [B, N, N, D]   edge embedding for i -> j at [i, j]
        node_mask:[B, N]
        edge_mask:[B, N, N]
        """
        B, N, D = h.shape
        H = self.num_heads
        Hd = self.head_dim

        q = self.q_proj(h).view(B, N, H, Hd).transpose(1, 2)  # [B, H, N, Hd]
        k = self.k_proj(h).view(B, N, H, Hd).transpose(1, 2)  # [B, H, N, Hd]
        v = self.v_proj(h).view(B, N, H, Hd).transpose(1, 2)  # [B, H, N, Hd]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # additive edge bias from relation embedding
        edge_bias = self.edge_bias_proj(rel_emb)  # [B, N, N, H]
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        attn_logits = attn_logits + edge_bias

        # disallow invalid targets
        pair_mask = edge_mask.unsqueeze(1)  # [B, 1, N, N]
        attn_logits = attn_logits.masked_fill(~pair_mask, float("-inf"))

        # if a node is padded, it should neither send nor receive meaningful attention
        valid_nodes = node_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        attn_logits = attn_logits.masked_fill(~valid_nodes, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, Hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = EdgeAwareSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, h, rel_emb, node_mask, edge_mask):
        h = h + self.attn(self.norm1(h), rel_emb, node_mask, edge_mask)
        h = h + self.ff(self.norm2(h))
        h = h * node_mask.unsqueeze(-1).float()
        return h


class SceneGraphTransformer(nn.Module):
    def __init__(
        self,
        num_obj_classes: int,
        num_rel_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.d_model = d_model

        self.rel_embedding = nn.Embedding(num_rel_classes, d_model)

        self.num_obj_classes = num_obj_classes
        self.mask_obj_token_id = num_obj_classes

        self.obj_embedding = nn.Embedding(num_obj_classes + 1, d_model)
        self.obj_head = nn.Linear(d_model, num_obj_classes)

        self.subj_proj = nn.Linear(d_model, d_model)
        self.obj_proj  = nn.Linear(d_model, d_model)
        self.pair_feat_dim = 4 * d_model

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            GraphTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        pair_feat_dim = 4 * d_model

        self.edge_head = nn.Sequential(
            nn.Linear(pair_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.rel_head_pos = nn.Sequential(
            nn.Linear(pair_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_rel_classes - 1),
        )

    def forward(self, obj_t, rel_t, t, node_mask, edge_mask):
        """
        obj_t: [B, N]
        rel_t: [B, N, N]
        t: [B]
        node_mask: [B, N]
        edge_mask: [B, N, N]
        """
        B, N = obj_t.shape

        obj_emb = self.obj_embedding(obj_t)          # [B, N, D]
        rel_emb = self.rel_embedding(rel_t)          # [B, N, N, D]

        t_emb = sinusoidal_timestep_embedding(t, self.d_model)
        t_emb = self.time_mlp(t_emb)                 # [B, D]

        h = obj_emb + t_emb[:, None, :]
        h = h * node_mask.unsqueeze(-1).float()

        for layer in self.layers:
            h = layer(h, rel_emb, node_mask, edge_mask)

        obj_logits = self.obj_head(h)                # [B, N, K_obj]

        subj_h = self.subj_proj(h)
        obj_h  = self.obj_proj(h)

        hi = subj_h.unsqueeze(2).expand(B, N, N, self.d_model)
        hj = obj_h.unsqueeze(1).expand(B, N, N, self.d_model)

        pair_feat = torch.cat([hi, hj, hi - hj, hi * hj], dim=-1)


        edge_logits = self.edge_head(pair_feat).squeeze(-1)   # [B, N, N]
        rel_logits_pos = self.rel_head_pos(pair_feat)         # [B, N, N, K_rel-1]

        return {
            "obj_logits": obj_logits,
            "edge_logits": edge_logits,
            "rel_logits_pos": rel_logits_pos,
        }
