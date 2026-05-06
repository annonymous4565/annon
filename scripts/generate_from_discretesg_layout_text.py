import os
import json
import random

import pyrallis
import torch

from configs import SGLayoutImageGenConfig
from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint

from datasets_.visual_genome.dataset import (
    build_scene_graph_dataloader,
    decode_item,
    format_decoded_graph,
)

from utils.graph_state_utils import build_full_relation_from_structured_state
from utils.wandb_utils import render_graph_text_block_to_image, ensure_dir
from utils.layout_vis import draw_layout_boxes, draw_layout_boxes_on_image

from models.sg_layout_image_generator import LayoutConditionedImageGenerator

from sampling.text_guided_demon_sampler import (
    run_full_reverse_chain_text_guided_demon_unconditional,
    CLIPTextGraphScorer,
    KeywordTextGraphScorer,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cxcywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def build_layout_conditioning(
    obj_final: torch.Tensor,          # [N]
    layout_box_final: torch.Tensor,   # [N,4] cxcywh
    node_mask: torch.Tensor,          # [N]
    image_obj_class_id: int = 0,
):
    boxes_xyxy = cxcywh_to_xyxy_tensor(layout_box_final)

    obj_class = torch.cat([
        torch.tensor([image_obj_class_id], dtype=torch.long),
        obj_final.long(),
    ], dim=0)

    obj_bbox = torch.cat([
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        boxes_xyxy.float(),
    ], dim=0)

    is_valid_obj = torch.cat([
        torch.tensor([1], dtype=torch.long),
        node_mask.long(),
    ], dim=0)

    return {
        "obj_class": obj_class.unsqueeze(0),
        "obj_bbox": obj_bbox.unsqueeze(0),
        "is_valid_obj": is_valid_obj.unsqueeze(0),
    }


def save_layout_conditioning_json(cond, path):
    payload = {
        "obj_class": cond["obj_class"].detach().cpu().tolist(),
        "obj_bbox": cond["obj_bbox"].detach().cpu().tolist(),
        "is_valid_obj": cond["is_valid_obj"].detach().cpu().tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_text_scorer(opt, device):
    if getattr(opt, "demon_scorer", "clip") == "keyword":
        return KeywordTextGraphScorer()
    return CLIPTextGraphScorer(
        model_name=getattr(opt, "clip_text_model", "openai/clip-vit-base-patch32"),
        device=device,
    )


@torch.no_grad()
def sample_text_guided_sg(opt, model, obj_gen, dataset, node_mask, edge_mask, scorer):
    return run_full_reverse_chain_text_guided_demon_unconditional(
        model=model,
        obj_gen=obj_gen,
        node_mask=node_mask,
        edge_mask=edge_mask,
        T=opt.num_diffusion_steps - 1,
        prompt=opt.text_prompt,
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
            scorer=scorer,
        )

        layout_box_final = sample_out["layout_box_final"]
        if layout_box_final is None:
            raise ValueError("layout_box_final is None. Enable/use trained layout head.")

        obj_pred = sample_out["obj_final"][0].detach().cpu()
        edge_pred = sample_out["edge_final"][0].detach().cpu()
        rel_pos_pred = sample_out["rel_pos_final"][0].detach().cpu()
        layout_i = layout_box_final[0].detach().cpu()
        node_mask_i = node_mask[0].detach().cpu().long()

        rel_full_pred = build_full_relation_from_structured_state(
            edge_t=edge_pred,
            rel_pos_t=rel_pos_pred,
            no_rel_token_id=0,
            num_rel_pos_classes=obj_gen.num_rel_pos_classes,
        )

        nodes, triplets = decode_item(
            obj_labels=obj_pred,
            rel_labels=rel_full_pred,
            node_mask=node_mask_i.bool(),
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

        cond = build_layout_conditioning(
            obj_final=obj_pred,
            layout_box_final=layout_i,
            node_mask=node_mask_i,
            image_obj_class_id=getattr(opt, "layout_image_obj_class_id", 0),
        )

        ex_dir = os.path.join(opt.output_dir, f"example_{saved:03d}")
        ensure_dir(ex_dir)

        with open(os.path.join(ex_dir, "prompt.txt"), "w") as f:
            f.write(opt.text_prompt)

        with open(os.path.join(ex_dir, "sampled_graph.txt"), "w") as f:
            f.write(graph_text)

        with open(os.path.join(ex_dir, "demon_logs.json"), "w") as f:
            json.dump(sample_out.get("demon_logs", []), f, indent=2)

        save_layout_conditioning_json(
            cond,
            os.path.join(ex_dir, "layout_conditioning.json"),
        )

        torch.save(
            {
                "obj_final": sample_out["obj_final"].detach().cpu(),
                "edge_final": sample_out["edge_final"].detach().cpu(),
                "rel_pos_final": sample_out["rel_pos_final"].detach().cpu(),
                "layout_box_final": sample_out["layout_box_final"].detach().cpu(),
                "node_mask": node_mask.detach().cpu(),
                "edge_mask": edge_mask.detach().cpu(),
            },
            os.path.join(ex_dir, "sampled_layout_state.pt"),
        )

        if getattr(opt, "draw_flowchart", False):
            render_graph_text_block_to_image(
                graph_text=graph_text,
                out_path_no_ext=os.path.join(ex_dir, "rendered_graph"),
                title=f"Text-guided SG | {opt.text_prompt}",
                rankdir=getattr(opt, "flowchart_rankdir", "LR"),
                format=getattr(opt, "flowchart_format", "png"),
                show_node_ids=getattr(opt, "flowchart_show_node_ids", True),
            )

        if getattr(opt, "save_layout_boxes_only", True):
            draw_layout_boxes(
                obj_class=cond["obj_class"][0],
                obj_bbox=cond["obj_bbox"][0],
                is_valid_obj=cond["is_valid_obj"][0],
                output_path=os.path.join(ex_dir, "layout_boxes.png"),
                class_names=dataset.object_vocab,
                image_size=getattr(opt, "layout_box_image_size", 256),
                draw_text=True,
                skip_first_object=True,
            )

        image_path = os.path.join(ex_dir, "generated_image.jpg")
        image_generator.generate(
            obj_class=cond["obj_class"],
            obj_bbox=cond["obj_bbox"],
            is_valid_obj=cond["is_valid_obj"],
            output_path=image_path,
            seed=opt.seed + saved,
        )

        if getattr(opt, "save_images_with_bboxs", False):
            draw_layout_boxes_on_image(
                image_path=image_path,
                obj_class=cond["obj_class"][0],
                obj_bbox=cond["obj_bbox"][0],
                is_valid_obj=cond["is_valid_obj"][0],
                output_path=os.path.join(ex_dir, "generated_image_with_boxes.jpg"),
                class_names=dataset.object_vocab,
                draw_text=True,
                skip_first_object=True,
            )

        print(f"[OK] saved text→SG→layout→image sample {saved}: {ex_dir}")
        saved += 1


@pyrallis.wrap()
def main(opt: SGLayoutImageGenConfig):
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

    image_generator = LayoutConditionedImageGenerator(opt)

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