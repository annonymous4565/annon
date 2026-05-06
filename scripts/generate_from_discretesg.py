import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple

import pyrallis
import torch

from configs import ImageGeneratorConfig
from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint

from datasets_.visual_genome.dataset import (
    build_scene_graph_dataloader,
    decode_item,
    format_decoded_graph,
)

from sampling.full_reverse_sampler import run_full_reverse_chain_unconditional
from utils.graph_state_utils import build_full_relation_from_structured_state

from models.sg_image_generator import SGConditionedImageGenerator

from utils.wandb_utils import render_graph_text_block_to_image, ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





def convert_decoded_graph_to_conditioning(
    nodes,
    triplets,
) -> Dict[str, List[List[Dict[str, Any]]]]:
    """
    Convert decoded graph output into the SG-conditioned image generator format.

    Expected decoded triplet format from your setup:
        (s_idx, s_name, rel, o_idx, o_name)

    Output format:
        all_triplets:
            [[{"item1": s_name, "relation": rel, "item2": o_name}, ...]]

        all_isolated_items:
            [["cabinet", "sink", ...]]

        all_global_ids:
            [[{"item1": s_idx, "item2": o_idx}, ...]]

    Here, the ids are taken directly from your sampled graph decoding, not from
    any external dataset mapping.
    """
    triplet_list = []
    global_id_list = []
    used_node_ids = set()

    for trip in triplets:
        if len(trip) != 5:
            raise ValueError(
                f"Expected decoded triplet of length 5 "
                f"(s_idx, s_name, rel, o_idx, o_name), got: {trip}"
            )

        s_idx, s_name, rel, o_idx, o_name = trip

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
        if name == "__pad__":
            continue
        if idx not in used_node_ids:
            isolated_items.append(str(name))

    conditioning = {
        "all_triplets": [triplet_list],
        "all_isolated_items": [isolated_items],
        "all_global_ids": [global_id_list],
    }
    return conditioning




@torch.no_grad()
def sample_and_generate_images(
    opt: ImageGeneratorConfig,
    model,
    obj_gen,
    dataloader,
    device: torch.device,
    image_generator: SGConditionedImageGenerator,
):
    os.makedirs(opt.output_dir, exist_ok=True)

    if getattr(opt, "draw_flowchart", False):
        ensure_dir(os.path.join(opt.flowchart_out_dir, "image_generator"))

    dataset = dataloader.dataset
    sample_item = dataset[getattr(opt, "shape_source_index", 0)]

    node_mask = sample_item["node_mask"].unsqueeze(0).to(device).bool()
    edge_mask = sample_item["edge_mask"].unsqueeze(0).to(device).bool()

    T = opt.num_diffusion_steps - 1

    for sample_idx in range(opt.max_images):
        
        sample_out = run_full_reverse_chain_unconditional(
            model=model,
            obj_gen=obj_gen,
            node_mask=node_mask,
            edge_mask=edge_mask,
            T=T,
            stochastic_obj=getattr(opt, "unconditional_stochastic_obj", False),
            stochastic_edge=getattr(opt, "unconditional_stochastic_edge", False),
            stochastic_rel=getattr(opt, "unconditional_stochastic_rel", False),
            return_trace=False,
            use_reverse_vocab_heads=getattr(opt, "unconditional_use_reverse_vocab_heads", True),
            obj_temp=getattr(opt, "unconditional_obj_temp", 1.0),
            rel_temp=getattr(opt, "unconditional_rel_temp", 1.0),
            edge_logit_threshold=getattr(opt, "unconditional_edge_logit_threshold", 0.5),
            relation_edge_logit_threshold=getattr(opt, "unconditional_relation_edge_logit_threshold", 0.0),
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

        pred_nodes, pred_trips = decode_item(
            obj_labels=obj_pred,
            rel_labels=rel_full_pred,
            node_mask=node_mask[0].detach().cpu(),
            edge_mask=edge_mask[0].detach().cpu(),
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            no_rel_token="__no_relation__",
            mask_obj_token_id=getattr(model, "mask_obj_token_id", len(dataset.object_vocab)),
        )

        pred_text = format_decoded_graph(pred_nodes, pred_trips)

        conditioning = convert_decoded_graph_to_conditioning(
            pred_nodes,
            pred_trips,
        )

        ex_dir = os.path.join(opt.output_dir, f"example_{sample_idx:03d}")
        os.makedirs(ex_dir, exist_ok=True)

        with open(os.path.join(ex_dir, "sampled_graph.txt"), "w", encoding="utf-8") as f:
            f.write(pred_text)

        with open(os.path.join(ex_dir, "conditioning.json"), "w", encoding="utf-8") as f:
            json.dump(conditioning, f, indent=2, ensure_ascii=False)

        if getattr(opt, "draw_flowchart", False):
            try:
                flow_dir = os.path.join(
                    opt.flowchart_out_dir,
                    "image_generator",
                    f"example_{sample_idx:03d}",
                )
                ensure_dir(flow_dir)

                render_graph_text_block_to_image(
                    graph_text=pred_text,
                    out_path_no_ext=os.path.join(flow_dir, "sampled_sg"),
                    title=f"Sampled SG | ex={sample_idx}",
                    rankdir=opt.flowchart_rankdir,
                    format=opt.flowchart_format,
                    show_node_ids=opt.flowchart_show_node_ids,
                )
            except Exception as e:
                print(f"[WARN] Flowchart render failed for generated example {sample_idx}: {e}")

        image_path = os.path.join(ex_dir, "generated_image.jpg")
        image_generator.generate(
            all_triplets=conditioning["all_triplets"],
            all_isolated_items=conditioning["all_isolated_items"],
            all_global_ids=conditioning["all_global_ids"],
            output_path=image_path,
            seed=opt.seed + sample_idx,
        )

        print(f"[OK] Saved unconditional sample {sample_idx} to: {ex_dir}")


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
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
    )

    image_generator = SGConditionedImageGenerator(opt, device=device)

    sample_and_generate_images(
        opt=opt,
        model=model,
        obj_gen=obj_gen,
        dataloader=loader,
        device=device,
        image_generator=image_generator,
    )


if __name__ == "__main__":
    main()