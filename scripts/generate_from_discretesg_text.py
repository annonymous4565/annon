import os
import json
import random
from typing import Dict, Any, List

import pyrallis
import torch

from configs import ImageGeneratorConfig
from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint

from datasets_.visual_genome.dataset import (
    build_scene_graph_dataloader,
    decode_item,
    format_decoded_graph,
)

from utils.graph_state_utils import build_full_relation_from_structured_state
from utils.wandb_utils import render_graph_text_block_to_image, ensure_dir

from models.sg_image_generator import SGConditionedImageGenerator

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
    if getattr(opt, "demon_scorer", "clip") == "keyword":
        return KeywordTextGraphScorer()
    return CLIPTextGraphScorer(
        model_name=getattr(opt, "clip_text_model", "openai/clip-vit-base-patch32"),
        device=device,
    )


def convert_decoded_graph_to_conditioning(nodes, triplets) -> Dict[str, List[List[Dict[str, Any]]]]:
    triplet_list = []
    global_id_list = []
    used_node_ids = set()

    for s_idx, s_name, rel, o_idx, o_name in triplets:
        triplet_list.append({
            "item1": str(s_name),
            "relation": str(rel),
            "item2": str(o_name),
        })
        global_id_list.append({
            "item1": int(s_idx),
            "item2": int(o_idx),
        })
        used_node_ids.add(int(s_idx))
        used_node_ids.add(int(o_idx))

    isolated_items = []
    for idx, name in enumerate(nodes):
        if str(name) != "__pad__" and idx not in used_node_ids:
            isolated_items.append(str(name))

    return {
        "all_triplets": [triplet_list],
        "all_isolated_items": [isolated_items],
        "all_global_ids": [global_id_list],
    }


@torch.no_grad()
def sample_text_guided_sg(opt, model, obj_gen, dataset, node_mask, edge_mask, device, scorer):
    prompt = getattr(opt, "text_prompt", "a person riding a horse")
    T = opt.num_diffusion_steps - 1

    return run_full_reverse_chain_text_guided_demon_unconditional(
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


@torch.no_grad()
def sample_and_generate_images(opt, model, obj_gen, loader, device, image_generator):
    os.makedirs(opt.output_dir, exist_ok=True)

    dataset = loader.dataset
    scorer = build_text_scorer(opt, device)

    sample_item = dataset[getattr(opt, "shape_source_index", 0)]
    node_mask = sample_item["node_mask"].unsqueeze(0).to(device).bool()
    edge_mask = sample_item["edge_mask"].unsqueeze(0).to(device).bool()

    saved = 0
    attempts = 0
    max_attempts = max(opt.max_images * 10, 50)

    while saved < opt.max_images and attempts < max_attempts:
        attempts += 1

        sample_out = sample_text_guided_sg(
            opt=opt,
            model=model,
            obj_gen=obj_gen,
            dataset=dataset,
            node_mask=node_mask,
            edge_mask=edge_mask,
            device=device,
            scorer=scorer,
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

        if triplets is None or len(triplets) == 0:
            print(f"[SKIP] attempt {attempts}: no triplets")
            continue

        graph_text = format_decoded_graph(nodes, triplets)
        conditioning = convert_decoded_graph_to_conditioning(nodes, triplets)

        ex_dir = os.path.join(opt.output_dir, f"example_{saved:03d}")
        ensure_dir(ex_dir)

        with open(os.path.join(ex_dir, "prompt.txt"), "w") as f:
            f.write(opt.text_prompt)

        with open(os.path.join(ex_dir, "sampled_graph.txt"), "w") as f:
            f.write(graph_text)

        with open(os.path.join(ex_dir, "conditioning.json"), "w") as f:
            json.dump(conditioning, f, indent=2)

        with open(os.path.join(ex_dir, "demon_logs.json"), "w") as f:
            json.dump(sample_out.get("demon_logs", []), f, indent=2)

        if getattr(opt, "draw_flowchart", False):
            render_graph_text_block_to_image(
                graph_text=graph_text,
                out_path_no_ext=os.path.join(ex_dir, "rendered_graph"),
                title=f"Text-guided SG | {opt.text_prompt}",
                rankdir=getattr(opt, "flowchart_rankdir", "LR"),
                format=getattr(opt, "flowchart_format", "png"),
                show_node_ids=getattr(opt, "flowchart_show_node_ids", True),
            )

        image_path = os.path.join(ex_dir, "generated_image.jpg")
        image_generator.generate(
            all_triplets=conditioning["all_triplets"],
            all_isolated_items=conditioning["all_isolated_items"],
            all_global_ids=conditioning["all_global_ids"],
            output_path=image_path,
            seed=opt.seed + saved,
        )

        print(f"[OK] saved text→SG→image sample {saved}: {ex_dir}")
        saved += 1


@pyrallis.wrap()
def main(opt: ImageGeneratorConfig):
    set_seed(opt.seed)
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

    image_generator = SGConditionedImageGenerator(opt, device=device)

    sample_and_generate_images(
        opt=opt,
        model=model,
        obj_gen=obj_gen,
        loader=loader,
        device=device,
        image_generator=image_generator,
    )


if __name__ == "__main__":
    main()