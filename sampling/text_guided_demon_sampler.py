from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from sampling.full_reverse_sampler import (
    build_unconditional_start_state,
    reverse_step_via_reverse_vocab_heads,
)
from utils.graph_state_utils import build_full_relation_from_structured_state
from datasets_.visual_genome.dataset import decode_item


def serialize_decoded_graph(nodes, triplets, max_triplets: int = 50) -> str:
    objects = [str(n) for n in nodes if str(n) != "__pad__"]
    obj_text = ", ".join(objects)

    if triplets is None or len(triplets) == 0:
        rel_text = "no relations"
    else:
        rel_text = "; ".join(
            f"{s_name} {rel} {o_name}"
            for _, s_name, rel, _, o_name in triplets[:max_triplets]
        )

    return f"Objects: {obj_text}. Relations: {rel_text}."


class KeywordTextGraphScorer:
    def __init__(self):
        self.stopwords = {
            "a", "an", "the", "on", "in", "of", "with", "and", "to",
            "is", "are", "photo", "image", "picture", "scene"
        }

    def _tok(self, s):
        s = s.lower()
        for ch in ",.;:!?()[]{}":
            s = s.replace(ch, " ")
        return [x for x in s.split() if x and x not in self.stopwords]

    def score(self, prompt: str, graph_texts: Sequence[str]) -> torch.Tensor:
        p = set(self._tok(prompt))
        if len(p) == 0:
            return torch.zeros(len(graph_texts))

        out = []
        for g in graph_texts:
            gt = set(self._tok(g))
            out.append(len(p & gt) / max(len(p), 1))
        return torch.tensor(out, dtype=torch.float32)


class CLIPTextGraphScorer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
    ):
        from transformers import CLIPModel, AutoTokenizer

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        feat = self.model.get_text_features(**batch)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def score(self, prompt: str, graph_texts: Sequence[str]) -> torch.Tensor:
        p = self._encode([prompt])          # [1,D]
        g = self._encode(graph_texts)       # [K,D]
        return (g @ p.t()).squeeze(-1).detach().cpu()


@torch.no_grad()
def x0_proxy_from_candidate(
    model,
    candidate: Dict[str, torch.Tensor],
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    obj_temp: float,
    rel_temp: float,
    edge_logit_threshold: float,
    relation_edge_logit_threshold: float,
    use_degree_pruning: bool,
    max_out_degree: int,
    max_in_degree: int,
    use_final_step_cleanup: bool,
    final_edge_logit_threshold: float,
    final_rel_conf_threshold: float,
    generic_obj_ids: Optional[Sequence[int]],
    generic_attachment_rel_ids: Optional[Sequence[int]],
    generic_attachment_edge_logit_threshold: float,
):
    """
    Deterministic clean proxy x0_hat from candidate state.

    Uses your reverse vocab heads at t_cur=1, because that is the final clean
    denoising step in your implementation.
    """
    return reverse_step_via_reverse_vocab_heads(
        model=model,
        obj_t=candidate["obj_t"],
        edge_t=candidate["edge_t"],
        rel_pos_t=candidate["rel_pos_t"],
        t_cur=1,
        node_mask=node_mask,
        edge_mask=edge_mask,
        stochastic_obj=False,
        stochastic_edge=False,
        stochastic_rel=False,
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
        use_reward_tilting=False,
    )


@torch.no_grad()
def state_to_graph_text(
    state: Dict[str, torch.Tensor],
    obj_gen,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    object_vocab,
    relation_vocab,
    mask_obj_token_id: int,
) -> str:
    obj = state["obj_t"][0].detach().cpu()
    edge = state["edge_t"][0].detach().cpu()
    rel_pos = state["rel_pos_t"][0].detach().cpu()

    rel_full = build_full_relation_from_structured_state(
        edge_t=edge,
        rel_pos_t=rel_pos,
        no_rel_token_id=0,
        num_rel_pos_classes=obj_gen.num_rel_pos_classes,
    )

    nodes, triplets = decode_item(
        obj_labels=obj,
        rel_labels=rel_full,
        node_mask=node_mask[0].detach().cpu(),
        edge_mask=edge_mask[0].detach().cpu(),
        object_vocab=object_vocab,
        relation_vocab=relation_vocab,
        no_rel_token="__no_relation__",
        mask_obj_token_id=mask_obj_token_id,
    )

    return serialize_decoded_graph(nodes, triplets)


def select_candidate(
    scores: torch.Tensor,
    mode: str,
    guidance_scale: float,
    temperature: float,
) -> int:
    if mode == "argmax":
        return int(scores.argmax().item())

    if mode == "softmax":
        logits = guidance_scale * scores / max(temperature, 1e-8)
        probs = torch.softmax(logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    raise ValueError(f"Unknown demon_selection_mode: {mode}")


@torch.no_grad()
def run_full_reverse_chain_text_guided_demon_unconditional(
    model,
    obj_gen,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    T: int,
    prompt: str,
    scorer,
    object_vocab,
    relation_vocab,
    mask_obj_token_id: int,

    stochastic_obj: bool = True,
    stochastic_edge: bool = True,
    stochastic_rel: bool = True,
    return_trace: bool = False,

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

    # keep your existing reward-tilting knobs
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

    # demon knobs
    demon_num_candidates: int = 4,
    demon_selection_mode: str = "argmax",  # argmax | softmax
    demon_guidance_scale: float = 10.0,
    demon_softmax_temperature: float = 1.0,
    demon_every_n_steps: int = 5,
    demon_start_t: Optional[int] = None,
    demon_end_t: int = 1,
    demon_use_x0_proxy: bool = True,
    demon_verbose: bool = False,
):
    """
    Demon-style text-guided unconditional SG sampling.

    At selected timesteps:
      x_t -> K candidates x_{t-1}
      score x0_proxy(candidate) against prompt
      choose candidate by argmax or softmax
    """
    device = node_mask.device
    B = node_mask.shape[0]

    if B != 1:
        raise ValueError("This implementation assumes batch size 1 for candidate scoring.")

    start_state = build_unconditional_start_state(
        obj_gen=obj_gen,
        node_mask=node_mask,
        edge_mask=edge_mask,
        T=T,
    )

    cur_obj = start_state["obj_t"]
    cur_edge = start_state["edge_t"]
    cur_rel_pos = start_state["rel_pos_t"]

    trace = []
    demon_logs = []

    if demon_start_t is None:
        demon_start_t = T

    def should_guide(t_cur: int) -> bool:
        if t_cur > demon_start_t:
            return False
        if t_cur < demon_end_t:
            return False
        if demon_every_n_steps <= 1:
            return True
        return (t_cur % demon_every_n_steps == 0) or (t_cur == 1)

    def one_reverse_step(
        obj_in,
        edge_in,
        rel_in,
        t_cur: int,
        force_stochastic: bool,
    ):
        return reverse_step_via_reverse_vocab_heads(
            model=model,
            obj_t=obj_in,
            edge_t=edge_in,
            rel_pos_t=rel_in,
            t_cur=t_cur,
            node_mask=node_mask,
            edge_mask=edge_mask,
            stochastic_obj=True if force_stochastic else stochastic_obj,
            stochastic_edge=True if force_stochastic else stochastic_edge,
            stochastic_rel=True if force_stochastic else stochastic_rel,
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

    for t_cur in range(T, 0, -1):
        if not should_guide(t_cur):
            step_out = one_reverse_step(
                cur_obj,
                cur_edge,
                cur_rel_pos,
                t_cur=t_cur,
                force_stochastic=False,
            )

        else:
            candidates = []
            graph_texts = []

            for _ in range(demon_num_candidates):
                cand = one_reverse_step(
                    cur_obj,
                    cur_edge,
                    cur_rel_pos,
                    t_cur=t_cur,
                    force_stochastic=True,
                )

                if demon_use_x0_proxy:
                    score_state = x0_proxy_from_candidate(
                        model=model,
                        candidate=cand,
                        node_mask=node_mask,
                        edge_mask=edge_mask,
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
                    )
                else:
                    score_state = cand

                graph_texts.append(
                    state_to_graph_text(
                        state=score_state,
                        obj_gen=obj_gen,
                        node_mask=node_mask,
                        edge_mask=edge_mask,
                        object_vocab=object_vocab,
                        relation_vocab=relation_vocab,
                        mask_obj_token_id=mask_obj_token_id,
                    )
                )
                candidates.append(cand)

            scores = scorer.score(prompt, graph_texts)

            chosen_idx = select_candidate(
                scores=scores,
                mode=demon_selection_mode,
                guidance_scale=demon_guidance_scale,
                temperature=demon_softmax_temperature,
            )

            step_out = candidates[chosen_idx]

            log_item = {
                "t": int(t_cur),
                "chosen_idx": int(chosen_idx),
                "scores": [float(x) for x in scores.tolist()],
                "chosen_score": float(scores[chosen_idx].item()),
                "chosen_graph_text": graph_texts[chosen_idx],
            }
            demon_logs.append(log_item)

            if demon_verbose:
                print(
                    f"[DEMON] t={t_cur} chosen={chosen_idx} "
                    f"score={scores[chosen_idx].item():.4f} "
                    f"scores={[round(float(x), 4) for x in scores.tolist()]}"
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
        "demon_logs": demon_logs,
    }

    if return_trace:
        out["trace"] = trace

    return out