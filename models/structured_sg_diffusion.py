# models/structured_sg_diffusion.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class NodePairBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.node_from_pair = MLP(3 * d_model, 2 * d_model, d_model, dropout)
        self.pair_from_node = MLP(5 * d_model, 2 * d_model, d_model, dropout)

        self.node_norm1 = nn.LayerNorm(d_model)
        self.node_norm2 = nn.LayerNorm(d_model)
        self.pair_norm1 = nn.LayerNorm(d_model)
        self.pair_norm2 = nn.LayerNorm(d_model)

        self.node_ff = MLP(d_model, 4 * d_model, d_model, dropout)
        self.pair_ff = MLP(d_model, 4 * d_model, d_model, dropout)

        self.out_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.in_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, node_h, pair_h, node_mask, edge_mask):
        B, N, D = node_h.shape
        valid_pair = edge_mask.bool()
        pair_h_masked = pair_h * valid_pair.unsqueeze(-1).float()

        out_deg = valid_pair.sum(dim=2).clamp(min=1).unsqueeze(-1).float()
        in_deg = valid_pair.sum(dim=1).clamp(min=1).unsqueeze(-1).float()

        out_ctx_raw = pair_h_masked.sum(dim=2) / out_deg
        in_ctx_raw = pair_h_masked.sum(dim=1) / in_deg

        out_gate = self.out_gate(torch.cat([node_h, out_ctx_raw], dim=-1))
        in_gate = self.in_gate(torch.cat([node_h, in_ctx_raw], dim=-1))

        out_ctx = out_gate * out_ctx_raw
        in_ctx = in_gate * in_ctx_raw

        node_update = self.node_from_pair(torch.cat([node_h, out_ctx, in_ctx], dim=-1))
        node_h = self.node_norm1(node_h + node_update)
        node_h = self.node_norm2(node_h + self.node_ff(node_h))
        node_h = node_h * node_mask.unsqueeze(-1).float()

        hi = node_h.unsqueeze(2).expand(B, N, N, D)
        hj = node_h.unsqueeze(1).expand(B, N, N, D)
        pair_update = self.pair_from_node(torch.cat([pair_h, hi, hj, hi - hj, hi * hj], dim=-1))
        pair_h = self.pair_norm1(pair_h + pair_update)
        pair_h = self.pair_norm2(pair_h + self.pair_ff(pair_h))
        pair_h = pair_h * valid_pair.unsqueeze(-1).float()

        return node_h, pair_h


class StructuredSceneGraphDiffusionModel(nn.Module):
    """
    Phase 4E.1:
    Relation-bucket-aware node conditioning.

    Compared to 4A, node prediction now uses multiple outgoing/incoming
    neighbor summaries grouped by relation-role buckets.
    """
    def __init__(
        self,
        num_obj_classes: int,
        num_rel_classes_full: int,
        d_model: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_relation_buckets: int = 5,
        rel_bucket_map_pos: torch.Tensor = None,   # [K_rel_pos]
        use_relation_bucket_node_conditioning: bool = False,
        use_reverse_vocab_heads: bool = False,
        use_layout_head: bool = False,
        layout_hidden_dim: int = 256,
        n_max: int = 20,
        object_from_structure_only = False,
        object_condition_on_gt_structure = False,
        use_direct_obj_head = False
    ):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes_full = num_rel_classes_full
        self.num_rel_pos_classes = num_rel_classes_full - 1

        self.mask_obj_token_id = num_obj_classes
        self.mask_rel_token_id = self.num_rel_pos_classes

        self.num_obj_classes = num_obj_classes

        self.num_relation_buckets = num_relation_buckets

        self.n_max = n_max
        self.object_from_structure_only = object_from_structure_only
        self.object_condition_on_gt_structure = object_condition_on_gt_structure
        
        # -------------------------
        # Learn relation buckets as projection
        # -------------------------
        # use_relation_bucket_node_conditioning = True
        self.use_relation_bucket_node_conditioning = use_relation_bucket_node_conditioning
        self.num_relation_buckets = num_relation_buckets

        # learned projection from relation-probability simplex to role channels
        self.rel_bucket_proj = nn.Linear(self.num_rel_pos_classes, num_relation_buckets, bias=False)

        # bucket-aware neighbor projections
        self.out_bucket_neighbor_proj = nn.Linear(num_relation_buckets * d_model + d_model, d_model)
        self.in_bucket_neighbor_proj = nn.Linear(num_relation_buckets * d_model + d_model, d_model)

        # class-side projections for bucket-aware scores
        self.class_out_bucket_neighbor_proj = nn.Linear(d_model, d_model)
        self.class_in_bucket_neighbor_proj = nn.Linear(d_model, d_model)

        # -------------------------
        # Input embeddings
        # -------------------------
        self.obj_embedding = nn.Embedding(num_obj_classes + 1, d_model)
        self.edge_embedding = nn.Embedding(2, d_model)
        self.rel_embedding = nn.Embedding(self.num_rel_pos_classes + 1, d_model)

        self.node_pos_embedding = nn.Embedding(self.n_max, self.obj_embedding.embedding_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.node_init = MLP(2 * d_model, 2 * d_model, d_model, dropout)
        self.pair_init = MLP(5 * d_model, 2 * d_model, d_model, dropout)

        self.layers = nn.ModuleList([
            NodePairBlock(d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        # -------------------------
        # Edge / relation heads
        # -------------------------
        self.edge_head = MLP(d_model, 2 * d_model, 1, dropout)
        self.rel_head_pos = MLP(d_model, 2 * d_model, self.num_rel_pos_classes, dropout)

        # -------------------------
        # Soft structure projections
        # -------------------------
        self.rel_prob_proj = nn.Linear(self.num_rel_pos_classes, d_model)
        self.edge_prob_proj = nn.Linear(1, d_model)

        # -------------------------
        # Object class embeddings
        # -------------------------
        self.obj_class_embed = nn.Embedding(num_obj_classes, d_model)
        nn.init.normal_(self.obj_class_embed.weight, mean=0.0, std=0.02)

        # -------------------------
        # Node-side latent projections
        # -------------------------
        self.local_node_proj = nn.Linear(2 * d_model, d_model)
        self.out_latent_proj = nn.Linear(3 * d_model, d_model)
        self.in_latent_proj = nn.Linear(3 * d_model, d_model)

        self.out_neighbor_proj = nn.Linear(2 * d_model, d_model)
        self.in_neighbor_proj = nn.Linear(2 * d_model, d_model)

        # -------------------------
        # Class-side projections
        # -------------------------
        self.class_local_proj = nn.Linear(d_model, d_model)
        self.class_out_latent_proj = nn.Linear(d_model, d_model)
        self.class_in_latent_proj = nn.Linear(d_model, d_model)
        self.class_out_neighbor_proj = nn.Linear(d_model, d_model)
        self.class_in_neighbor_proj = nn.Linear(d_model, d_model)

        # -------------------------
        # Residual bias head
        # -------------------------
        if self.use_relation_bucket_node_conditioning:
            bias_in_dim = (8 + 2 * num_relation_buckets) * d_model
        else:
            bias_in_dim = 8 * d_model

        self.obj_bias_head = nn.Sequential(
            nn.Linear(bias_in_dim, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, num_obj_classes),
        )

        self.use_reverse_vocab_heads = use_reverse_vocab_heads

        # -------------------------
        # Phase 7A.2: layout head
        # predicts normalized (cx, cy, w, h) per valid node
        # -------------------------
        self.use_layout_head = use_layout_head

        if self.use_layout_head:
            self.layout_head = nn.Sequential(
                nn.Linear(d_model, layout_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(layout_hidden_dim, layout_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(layout_hidden_dim, 4),
            )

        if self.use_reverse_vocab_heads:
            # Reverse object head:
            # predicts clean object classes + [MASK_OBJ]
            self.obj_rev_head = nn.Sequential(
                nn.Linear(bias_in_dim, 2 * d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, num_obj_classes + 1),
            )

            # Reverse edge head:
            # same feature input as clean edge head (pair_h)
            self.edge_rev_head = MLP(d_model, 2 * d_model, 1, dropout)

            # Reverse relation head:
            # predicts positive relation classes + [MASK_REL]
            # same feature input as clean relation head (pair_h)
            self.rel_rev_head_pos = MLP(
                d_model,
                2 * d_model,
                self.num_rel_pos_classes + 1,
                dropout,
            )
        
        ########## sanity ######################
        self.use_direct_obj_head = use_direct_obj_head

        direct_obj_in_dim = (
            d_model * 8  # adjust if your hidden dim variable is not D
        )

        self.direct_obj_head = nn.Sequential(
            nn.Linear(direct_obj_in_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_obj_classes),
        )

    def bucketize_rel_probs(self, rel_prob: torch.Tensor) -> torch.Tensor:
        """
        rel_prob: [B,N,N,K_rel_pos]
        returns:  [B,N,N,Bk]
        """
        bucket_logits = self.rel_bucket_proj(rel_prob)   # [B,N,N,Bk]
        bucket_prob = torch.softmax(bucket_logits, dim=-1)
        return bucket_prob

    def forward(
        self,
        obj_t,
        edge_t,
        rel_pos_t,
        t,
        node_mask,
        edge_mask,
        edge_input_override=None,
        rel_input_override=None,
    ):
        B, N = obj_t.shape
        D = self.obj_embedding.embedding_dim

        pos_ids = torch.arange(N, device=obj_t.device).unsqueeze(0).expand(B, N)
        pos_emb = self.node_pos_embedding(pos_ids)  # [B, N, D]

        # -------------------------
        # Timestep embedding
        # -------------------------
        t_emb = sinusoidal_timestep_embedding(t, D)
        t_emb = self.time_mlp(t_emb)

        # -------------------------
        # Node input stream
        # -------------------------
        obj_label_emb = self.obj_embedding(obj_t)

        if getattr(self, "object_from_structure_only", False):
            obj_emb = pos_emb + 0.0 * obj_label_emb
        else:
            obj_emb = obj_label_emb + pos_emb

        node_h = self.node_init(
            torch.cat([obj_emb, t_emb[:, None, :].expand(B, N, D)], dim=-1)
        )
        node_h = node_h * node_mask.unsqueeze(-1).float()

        # -------------------------
        # Pair input stream
        # -------------------------
        edge_input = edge_t if edge_input_override is None else edge_input_override
        rel_input = rel_pos_t if rel_input_override is None else rel_input_override

        edge_emb = self.edge_embedding(edge_input.long())
        rel_emb = self.rel_embedding(rel_input.long())

        hi = obj_emb.unsqueeze(2).expand(B, N, N, D)
        hj = obj_emb.unsqueeze(1).expand(B, N, N, D)
        tt = t_emb[:, None, None, :].expand(B, N, N, D)

        pair_h = self.pair_init(torch.cat([hi, hj, edge_emb, rel_emb, tt], dim=-1))
        pair_h = pair_h * edge_mask.unsqueeze(-1).float()

        # -------------------------
        # Message passing
        # -------------------------
        for layer in self.layers:
            node_h, pair_h = layer(node_h, pair_h, node_mask, edge_mask)

        # -------------------------
        # Structure prediction
        # -------------------------
        edge_logits = self.edge_head(pair_h).squeeze(-1)      # [B,N,N]
        rel_logits_pos = self.rel_head_pos(pair_h)            # [B,N,N,K_rel_pos]

        if self.use_reverse_vocab_heads:
            edge_rev_logits = self.edge_rev_head(pair_h).squeeze(-1)          # [B,N,N]
            rel_rev_logits_pos = self.rel_rev_head_pos(pair_h)                # [B,N,N,K_rel_pos+1]
        else:
            edge_rev_logits = None
            rel_rev_logits_pos = None

        pair_mask_f = edge_mask.unsqueeze(-1).float()

        edge_prob = torch.sigmoid(edge_logits).unsqueeze(-1)  # [B,N,N,1]
        rel_prob = torch.softmax(rel_logits_pos, dim=-1)      # [B,N,N,K_rel_pos]

        edge_prob = edge_prob * pair_mask_f
        rel_prob = rel_prob * pair_mask_f

        # -------------------------
        # Standard latent structure summaries
        # -------------------------
        edge_feat = self.edge_prob_proj(edge_prob)
        rel_feat = self.rel_prob_proj(rel_prob)

    
        pair_h_masked = pair_h * pair_mask_f

        out_deg = edge_mask.sum(dim=2).clamp(min=1).unsqueeze(-1).float()
        in_deg = edge_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()

        final_out_ctx = pair_h_masked.sum(dim=2) / out_deg
        final_in_ctx = pair_h_masked.sum(dim=1) / in_deg

        out_rel_feat = rel_feat.sum(dim=2) / out_deg
        in_rel_feat = rel_feat.sum(dim=1) / in_deg

        out_edge_feat = edge_feat.sum(dim=2) / out_deg
        in_edge_feat = edge_feat.sum(dim=1) / in_deg

        # -------------------------
        # 4A coarse explicit neighbor summaries
        # -------------------------
        rel_conf = rel_prob.max(dim=-1, keepdim=True).values
        neighbor_weight = edge_prob * rel_conf
        neighbor_weight = neighbor_weight * pair_mask_f

        out_neighbor_raw = (neighbor_weight * hj).sum(dim=2) / out_deg
        in_neighbor_raw = (neighbor_weight * hi).sum(dim=1) / in_deg

        # -------------------------
        # Phase 4E.1 Learned relation-role channels
        # -------------------------
        rel_bucket_prob = self.bucketize_rel_probs(rel_prob)
        rel_bucket_prob = rel_bucket_prob.unsqueeze(-1)              # [B,N,N,Bk,1]

        hj_exp = hj.unsqueeze(3)                                     # [B,N,N,1,D]
        hi_exp = hi.unsqueeze(3)                                     # [B,N,N,1,D]

        # weight by both predicted edge existence and learned relation-role channel
        # edge_bucket_weight = edge_prob_for_neighbors.unsqueeze(3) * rel_bucket_prob

        edge_bucket_weight = edge_prob.unsqueeze(3) * rel_bucket_prob
        edge_bucket_weight = edge_bucket_weight * edge_mask.unsqueeze(-1).unsqueeze(-1).float()

        # outgoing role-aware neighbor summaries
        out_bucket_sum = (edge_bucket_weight * hj_exp).sum(dim=2)    # [B,N,Bk,D]
        # incoming role-aware neighbor summaries
        in_bucket_sum = (edge_bucket_weight * hi_exp).sum(dim=1)     # [B,N,Bk,D]

        out_bucket_sum = out_bucket_sum / out_deg.unsqueeze(2)       # [B,N,Bk,D]
        in_bucket_sum = in_bucket_sum / in_deg.unsqueeze(2)          # [B,N,Bk,D]

        out_bucket_flat = out_bucket_sum.reshape(B, N, self.num_relation_buckets * D)
        in_bucket_flat = in_bucket_sum.reshape(B, N, self.num_relation_buckets * D)

        # -------------------------
        # Node-side features
        # -------------------------
        local_feat = self.local_node_proj(torch.cat([node_h, obj_emb], dim=-1))
        out_latent_feat = self.out_latent_proj(torch.cat([final_out_ctx, out_rel_feat, out_edge_feat], dim=-1))
        in_latent_feat = self.in_latent_proj(torch.cat([final_in_ctx, in_rel_feat, in_edge_feat], dim=-1))

        out_neighbor_feat = self.out_neighbor_proj(torch.cat([out_neighbor_raw, obj_emb], dim=-1))
        in_neighbor_feat = self.in_neighbor_proj(torch.cat([in_neighbor_raw, obj_emb], dim=-1))

        out_bucket_neighbor_feat = self.out_bucket_neighbor_proj(
            torch.cat([out_bucket_flat, obj_emb], dim=-1)
        )
        in_bucket_neighbor_feat = self.in_bucket_neighbor_proj(
            torch.cat([in_bucket_flat, obj_emb], dim=-1)
        )

        # -------------------------
        # Candidate class embeddings
        # -------------------------
        class_embed = self.obj_class_embed.weight

        class_local = self.class_local_proj(class_embed)
        class_out_latent = self.class_out_latent_proj(class_embed)
        class_in_latent = self.class_in_latent_proj(class_embed)
        class_out_neighbor = self.class_out_neighbor_proj(class_embed)
        class_in_neighbor = self.class_in_neighbor_proj(class_embed)
        class_out_bucket_neighbor = self.class_out_bucket_neighbor_proj(class_embed)
        class_in_bucket_neighbor = self.class_in_bucket_neighbor_proj(class_embed)

        # -------------------------
        # Compatibility scores
        # -------------------------
        score_local = torch.einsum("bnd,kd->bnk", local_feat, class_local)
        score_out_latent = torch.einsum("bnd,kd->bnk", out_latent_feat, class_out_latent)
        score_in_latent = torch.einsum("bnd,kd->bnk", in_latent_feat, class_in_latent)
        score_out_neighbor = torch.einsum("bnd,kd->bnk", out_neighbor_feat, class_out_neighbor)
        score_in_neighbor = torch.einsum("bnd,kd->bnk", in_neighbor_feat, class_in_neighbor)

        # NEW
        score_out_bucket_neighbor = torch.einsum(
            "bnd,kd->bnk", out_bucket_neighbor_feat, class_out_bucket_neighbor
        )
        score_in_bucket_neighbor = torch.einsum(
            "bnd,kd->bnk", in_bucket_neighbor_feat, class_in_bucket_neighbor
        )
        
        # -------------------------
        # Residual bias
        # -------------------------
        if self.use_relation_bucket_node_conditioning:
            bias_input = torch.cat(
                [
                    node_h,
                    obj_emb,
                    final_out_ctx,
                    final_in_ctx,
                    out_rel_feat,
                    in_rel_feat,
                    out_neighbor_raw,
                    in_neighbor_raw,
                    out_bucket_flat,
                    in_bucket_flat,
                ],
                dim=-1,
            )
        else:
            bias_input = torch.cat(
                [
                    node_h,
                    obj_emb,
                    final_out_ctx,
                    final_in_ctx,
                    out_rel_feat,
                    in_rel_feat,
                    out_neighbor_raw,
                    in_neighbor_raw,
                ],
                dim=-1,
            )


        bias_logits = self.obj_bias_head(bias_input)

        if self.use_reverse_vocab_heads:
            obj_rev_logits = self.obj_rev_head(bias_input)   # [B,N,K_obj+1]
        else:
            obj_rev_logits = None

        # obj_logits = (
        #     score_local
        #     + score_out_latent
        #     + score_in_latent
        #     + score_out_neighbor
        #     + score_in_neighbor
        #     + score_out_bucket_neighbor
        #     + score_in_bucket_neighbor
        #     + bias_logits
        # )

        if getattr(self, "use_direct_obj_head", False):
            direct_obj_input = torch.cat(
                [
                    local_feat,
                    out_latent_feat,
                    in_latent_feat,
                    out_neighbor_feat,
                    in_neighbor_feat,
                    out_bucket_neighbor_feat,
                    in_bucket_neighbor_feat,
                    node_h,
                ],
                dim=-1,
            )

            obj_logits = self.direct_obj_head(direct_obj_input)
        else:
            obj_logits = (
                score_local
                + score_out_latent
                + score_in_latent
                + score_out_neighbor
                + score_in_neighbor
                + score_out_bucket_neighbor
                + score_in_bucket_neighbor
                + bias_logits
            )

        obj_logits = obj_logits * node_mask.unsqueeze(-1).float()
        
        # -------------------------
        # Phase 7A.2: layout prediction
        # output format: normalized (cx, cy, w, h) in [0,1]
        # -------------------------
        if self.use_layout_head:
            layout_box_pred = torch.sigmoid(self.layout_head(node_h))   # [B,N,4]
            layout_box_pred = layout_box_pred * node_mask.unsqueeze(-1).float()
        else:
            layout_box_pred = None
        
        # ddp_keepalive = 0.0 * edge_logits.sum() + 0.0 * rel_logits_pos.sum()

        # if self.use_reverse_vocab_heads:
        #     ddp_keepalive = (
        #         ddp_keepalive
        #         + 0.0 * edge_rev_logits.sum()
        #         + 0.0 * rel_rev_logits_pos.sum()
        #     )

        # obj_logits = obj_logits + ddp_keepalive

        out = {
            "obj_logits": obj_logits,
            "edge_logits": edge_logits,
            "rel_logits_pos": rel_logits_pos,
        }

        if self.use_layout_head:
            out["layout_box_pred"] = layout_box_pred

        if self.use_reverse_vocab_heads:
            out["obj_rev_logits"] = obj_rev_logits
            out["edge_rev_logits"] = edge_rev_logits
            out["rel_rev_logits_pos"] = rel_rev_logits_pos

        return out