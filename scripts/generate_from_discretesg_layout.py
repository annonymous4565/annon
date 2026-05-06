import json
import os
import random

import pyrallis
import torch

from configs import SGLayoutImageGenConfig
from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint

from utils.layout_vis import draw_layout_boxes, draw_layout_boxes_on_image

from datasets_.visual_genome.dataset import (
    build_scene_graph_dataloader,
    decode_item,
    format_decoded_graph,
)

from sampling.full_reverse_sampler import run_full_reverse_chain_unconditional
from utils.graph_state_utils import build_full_relation_from_structured_state

from models.sg_layout_image_generator import LayoutConditionedImageGenerator
from utils.wandb_utils import render_graph_text_block_to_image, ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def canonicalize_boxes_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)

def cxcywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    out = torch.stack([x1, y1, x2, y2], dim=-1)
    return out.clamp(0.0, 1.0)

def build_layout_conditioning(
    obj_final: torch.Tensor,   # [N]
    layout_box_final: torch.Tensor,        # [N,4] in cxcywh
    node_mask: torch.Tensor,    # [N]
    image_obj_class_id: int = 0,
):

    # Convert boxes to xyxy (you already do this correctly)
    boxes_xyxy = cxcywh_to_xyxy_tensor(layout_box_final)

    # --------------------------------------------------
    # Prepend global image object
    # --------------------------------------------------
    obj_class = torch.cat([
        torch.tensor([image_obj_class_id], dtype=torch.long),
        obj_final.long()
    ], dim=0)

    obj_bbox = torch.cat([
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        boxes_xyxy.float()
    ], dim=0)

    is_valid_obj = torch.cat([
        torch.tensor([1], dtype=torch.long),
        node_mask.long()
    ], dim=0)

    return {
        "obj_class": obj_class.unsqueeze(0),      # [1, N+1]
        "obj_bbox": obj_bbox.unsqueeze(0),        # [1, N+1, 4]
        "is_valid_obj": is_valid_obj.unsqueeze(0),
    }


@torch.no_grad()
def sample_and_generate_images(
    opt: SGLayoutImageGenConfig,
    model,
    obj_gen,
    dataloader,
    device: torch.device,
    image_generator: LayoutConditionedImageGenerator,
):
    os.makedirs(opt.output_dir, exist_ok=True)

    if getattr(opt, "draw_flowchart", False):
        ensure_dir(os.path.join(opt.flowchart_out_dir, "layout_image_generator"))

    dataset = dataloader.dataset
    sample_item = dataset[getattr(opt, "shape_source_index", 0)]

    node_mask = sample_item["node_mask"].unsqueeze(0).to(device).bool()
    edge_mask = sample_item["edge_mask"].unsqueeze(0).to(device).bool()

    T = opt.num_diffusion_steps - 1
    saved = 0
    attempts = 0
    max_attempts = max(opt.max_images * 10, 50)

    while saved < opt.max_images and attempts < max_attempts:
        attempts += 1

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
            use_degree_pruning=getattr(opt, "unconditional_use_degree_pruning", False),
            max_out_degree=getattr(opt, "unconditional_max_out_degree", 0),
            max_in_degree=getattr(opt, "unconditional_max_in_degree", 0),
        )

        obj_pred = sample_out["obj_final"]                 # [B,N]
        edge_pred = sample_out["edge_final"]               # [B,N,N]
        rel_pos_pred = sample_out["rel_pos_final"]         # [B,N,N]
        layout_box_final = sample_out["layout_box_final"]  # [B,N,4] or None

        if layout_box_final is None:
            raise ValueError(
                "layout_box_final is None. Your sampler must return layout_box_final."
            )

        obj_pred_i = obj_pred[0].detach().cpu()
        edge_pred_i = edge_pred[0].detach().cpu()
        rel_pos_pred_i = rel_pos_pred[0].detach().cpu()

        rel_full_pred = build_full_relation_from_structured_state(
            edge_t=edge_pred_i,
            rel_pos_t=rel_pos_pred_i,
            no_rel_token_id=0,
            num_rel_pos_classes=obj_gen.num_rel_pos_classes,
        )

        pred_nodes, pred_trips = decode_item(
            obj_labels=obj_pred_i,
            rel_labels=rel_full_pred,
            node_mask=node_mask[0].detach().cpu(),
            edge_mask=edge_mask[0].detach().cpu(),
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            no_rel_token="__no_relation__",
            mask_obj_token_id=getattr(opt, "mask_obj_token_id", len(dataset.object_vocab)),
        )

        if pred_trips is None or len(pred_trips) == 0:
            print(f"[SKIP] sample attempt {attempts} has no triplets")
            continue

        pred_text = format_decoded_graph(pred_nodes, pred_trips)

        cond = build_layout_conditioning(
            obj_final=obj_pred,
            layout_box_final=layout_box_final,
            node_mask=node_mask,
        )

        ex_dir = os.path.join(opt.output_dir, f"example_{saved:03d}")
        os.makedirs(ex_dir, exist_ok=True)

        with open(os.path.join(ex_dir, "sampled_graph.txt"), "w", encoding="utf-8") as f:
            f.write(pred_text)

        torch.save(
            {
                "obj_final": obj_pred.detach().cpu(),
                "edge_final": edge_pred.detach().cpu(),
                "rel_pos_final": rel_pos_pred.detach().cpu(),
                "layout_box_final": layout_box_final.detach().cpu(),
                "node_mask": node_mask.detach().cpu(),
                "edge_mask": edge_mask.detach().cpu(),
            },
            os.path.join(ex_dir, "sampled_layout_state.pt"),
        )

        if getattr(opt, "draw_flowchart", False):
            try:
                flow_dir = os.path.join(
                    opt.flowchart_out_dir,
                    "layout_image_generator",
                    f"example_{saved:03d}",
                )
                ensure_dir(flow_dir)

                render_graph_text_block_to_image(
                    graph_text=pred_text,
                    out_path_no_ext=os.path.join(flow_dir, "sampled_sg"),
                    title=f"Sampled SG | ex={saved}",
                    rankdir=opt.flowchart_rankdir,
                    format=opt.flowchart_format,
                    show_node_ids=opt.flowchart_show_node_ids,
                )
            except Exception as e:
                print(f"[WARN] Flowchart render failed for generated example {saved}: {e}")
        
        # Save boxes-only visualization
        if getattr(opt, "save_layout_boxes_only", True):
            try:
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
            except Exception as e:
                print(f"[WARN] Failed to save boxes-only visualization for example {saved}: {e}")

        image_path = os.path.join(ex_dir, "generated_image.jpg")
        image_generator.generate(
            obj_class=cond["obj_class"],
            obj_bbox=cond["obj_bbox"],
            is_valid_obj=cond["is_valid_obj"],
            output_path=image_path,
            seed=opt.seed + saved,
        )

        print(f"[OK] Saved layout-conditioned sample {saved} to: {ex_dir}")
        saved += 1

        if getattr(opt, "save_images_with_bboxs", False):
            try:
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
            except Exception as e:
                print(f"[WARN] Failed to overlay boxes on generated image for example {saved}: {e}")

    if saved < opt.max_images:
        print(f"[WARN] Only generated {saved}/{opt.max_images} valid samples after {attempts} attempts")



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
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
    )

    image_generator = LayoutConditionedImageGenerator(opt)

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