import os
import json
import random

import pyrallis
import torch
from tqdm import tqdm

from configs import DiscreteSGConfig
from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint

from datasets_.visual_genome.dataset import (
    build_scene_graph_dataloader,
    decode_item,
    format_decoded_graph,
)

from utils.graph_state_utils import build_full_relation_from_structured_state
from utils.wandb_utils import render_graph_text_block_to_image, ensure_dir

from sampling.text_guided_demon_sampler import (
    run_full_reverse_chain_text_guided_demon_unconditional,
    CLIPTextGraphScorer,
    KeywordTextGraphScorer,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text_scorer(opt, device):
    scorer_type = getattr(opt, "demon_scorer", "clip")
    if scorer_type == "keyword":
        return KeywordTextGraphScorer()
    if scorer_type == "clip":
        return CLIPTextGraphScorer(
            model_name=getattr(opt, "clip_text_model", "openai/clip-vit-base-patch32"),
            device=device,
        )
    raise ValueError(f"Unknown demon_scorer: {scorer_type}")


@torch.no_grad()
def main_impl(opt: DiscreteSGConfig):
    set_seed(getattr(opt, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, obj_gen, trainer, loaded_opt, ckpt = load_model_and_objgen_from_checkpoint(
        ckpt_path=opt.ckpt,
        device=device,
        strict=True,
    )
    model = model.to(device)
    model.eval()

    loader = build_scene_graph_dataloader(
        npz_path=opt.val_npz_path,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    dataset = loader.dataset

    scorer = build_text_scorer(opt, device)

    output_dir = getattr(opt, "output_dir", "./output/text_guided_sg")
    ensure_dir(output_dir)

    draw_flowchart = getattr(opt, "draw_flowchart", True)
    max_images = getattr(opt, "max_images", 8)
    prompt = getattr(opt, "text_prompt", "a person riding a horse")

    sample_item = dataset[getattr(opt, "shape_source_index", 0)]
    node_mask = sample_item["node_mask"].unsqueeze(0).to(device).bool()
    edge_mask = sample_item["edge_mask"].unsqueeze(0).to(device).bool()

    T = opt.num_diffusion_steps - 1

    for sample_idx in tqdm(range(max_images), desc="Text-guided SG generation"):
        sample_out = run_full_reverse_chain_text_guided_demon_unconditional(
            model=model,
            obj_gen=obj_gen,
            node_mask=node_mask,
            edge_mask=edge_mask,
            T=T,
            prompt=prompt,
            scorer=scorer,
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            mask_obj_token_id=opt.mask_obj_token_id,

            stochastic_obj=getattr(opt, "unconditional_stochastic_obj", False),
            stochastic_edge=getattr(opt, "unconditional_stochastic_edge", False),
            stochastic_rel=getattr(opt, "unconditional_stochastic_rel", False),

            obj_temp=getattr(opt, "unconditional_obj_temp", 1.0),
            rel_temp=getattr(opt, "unconditional_rel_temp", 1.0),
            edge_logit_threshold=getattr(opt, "unconditional_edge_logit_threshold", 0.5),
            relation_edge_logit_threshold=getattr(opt, "unconditional_relation_edge_logit_threshold", 0.0),

            use_degree_pruning=getattr(opt, "unconditional_use_degree_pruning", False),
            max_out_degree=getattr(opt, "unconditional_max_out_degree", 0),
            max_in_degree=getattr(opt, "unconditional_max_in_degree", 0),

            use_final_step_cleanup=getattr(opt, "full_reverse_use_final_step_cleanup", False),
            final_edge_logit_threshold=getattr(opt, "full_reverse_final_edge_logit_threshold", 0.5),
            final_rel_conf_threshold=getattr(opt, "full_reverse_final_rel_conf_threshold", 0.0),
            generic_obj_ids=getattr(opt, "full_reverse_generic_obj_ids", []),
            generic_attachment_rel_ids=getattr(opt, "full_reverse_generic_attachment_rel_ids", []),
            generic_attachment_edge_logit_threshold=getattr(opt, "full_reverse_generic_attachment_edge_logit_threshold", 1.0),

            demon_num_candidates=getattr(opt, "demon_num_candidates", 4),
            demon_selection_mode=getattr(opt, "demon_selection_mode", "argmax"),
            demon_guidance_scale=getattr(opt, "demon_guidance_scale", 10.0),
            demon_softmax_temperature=getattr(opt, "demon_softmax_temperature", 1.0),
            demon_every_n_steps=getattr(opt, "demon_every_n_steps", 5),
            demon_start_t=getattr(opt, "demon_start_t", None),
            demon_end_t=getattr(opt, "demon_end_t", 1),
            demon_use_x0_proxy=getattr(opt, "demon_use_x0_proxy", True),
            demon_verbose=getattr(opt, "demon_verbose", False),
        )

        obj_pred = sample_out["obj_final"][0].detach().cpu()
        edge_pred = sample_out["edge_final"][0].detach().cpu()
        rel_pos_pred = sample_out["rel_pos_final"][0].detach().cpu()

        rel_full_pred = build_full_relation_from_structured_state(
            edge_t=edge_pred,
            rel_pos_t=rel_pos_pred,
            no_rel_token_id=0,
            num_rel_pos_classes=obj_gen.num_rel_pos_classes,
        )

        nodes, triplets = decode_item(
            obj_labels=obj_pred,
            rel_labels=rel_full_pred,
            node_mask=node_mask[0].detach().cpu(),
            edge_mask=edge_mask[0].detach().cpu(),
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            no_rel_token="__no_relation__",
            mask_obj_token_id=opt.mask_obj_token_id,
        )

        graph_text = format_decoded_graph(nodes, triplets)

        ex_dir = os.path.join(output_dir, f"example_{sample_idx:03d}")
        ensure_dir(ex_dir)

        with open(os.path.join(ex_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        with open(os.path.join(ex_dir, "sampled_graph.txt"), "w") as f:
            f.write(graph_text)

        with open(os.path.join(ex_dir, "demon_logs.json"), "w") as f:
            json.dump(sample_out.get("demon_logs", []), f, indent=2)

        torch.save(
            {
                "obj_final": sample_out["obj_final"].detach().cpu(),
                "edge_final": sample_out["edge_final"].detach().cpu(),
                "rel_pos_final": sample_out["rel_pos_final"].detach().cpu(),
                "layout_box_final": sample_out["layout_box_final"].detach().cpu()
                if sample_out["layout_box_final"] is not None else None,
                "node_mask": node_mask.detach().cpu(),
                "edge_mask": edge_mask.detach().cpu(),
            },
            os.path.join(ex_dir, "sampled_state.pt"),
        )

        if draw_flowchart:
            render_graph_text_block_to_image(
                graph_text=graph_text,
                out_path_no_ext=os.path.join(ex_dir, "rendered_graph"),
                title=f"Text-guided SG | {prompt}",
                rankdir=getattr(opt, "flowchart_rankdir", "LR"),
                format=getattr(opt, "flowchart_format", "png"),
                show_node_ids=getattr(opt, "flowchart_show_node_ids", True),
            )

        print(f"[OK] saved {ex_dir}")


@pyrallis.wrap()
def main(opt: DiscreteSGConfig):
    main_impl(opt)


if __name__ == "__main__":
    main()