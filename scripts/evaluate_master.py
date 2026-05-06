import os
import json
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

from datasets_ import build_eval_dataloader
from datasets_.visual_genome.dataset import decode_item, format_decoded_graph
from utils.graph_state_utils import build_full_relation_from_structured_state
from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint
from utils.wandb_utils import ensure_dir

from sampling.text_guided_demon_sampler import (
    run_full_reverse_chain_text_guided_demon_unconditional,
    CLIPTextGraphScorer,
    KeywordTextGraphScorer,
)

from models.sg_image_generator import SGConditionedImageGenerator
from models.sg_layout_image_generator import LayoutConditionedImageGenerator


from configs import DiscreteSGEvalConfig

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_image_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_gt_image(batch, i: int, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    img = batch["gt_image"][i]
    tensor_image_to_pil(img).save(out_path)

def save_tensor_image(x, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)


def save_prompts_json(prompts: List[Dict[str, Any]], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def save_triplets_tuple_format(triplets, out_path: str):
    """
    Compatible with graph metrics: one tuple per line:
      (s_idx, s_name, rel, o_idx, o_name)
    """
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        if triplets is None or len(triplets) == 0:
            f.write("(none)\n")
        else:
            for t in triplets:
                s_idx, s_name, rel, o_idx, o_name = t
                f.write(f"({int(s_idx)}, {repr(str(s_name))}, {repr(str(rel))}, {int(o_idx)}, {repr(str(o_name))})\n")


def save_layout_txt(obj_labels, boxes, node_mask, object_vocab, out_path: str):
    """
    Saves layout as:
      idx, object_name, x1, y1, x2, y2

    Assumes boxes are already normalized xyxy if passed to LayoutDiffusion.
    """
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("idx\tobject\tx1\ty1\tx2\ty2\n")
        for i in range(obj_labels.shape[0]):
            if int(node_mask[i].item()) == 0:
                continue
            obj_id = int(obj_labels[i].item())
            name = object_vocab[obj_id] if 0 <= obj_id < len(object_vocab) else str(obj_id)
            b = boxes[i].detach().cpu().tolist()
            f.write(
                f"{i}\t{name}\t"
                f"{float(b[0]):.6f}\t{float(b[1]):.6f}\t"
                f"{float(b[2]):.6f}\t{float(b[3]):.6f}\n"
            )


def cxcywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def build_layout_conditioning_for_layoutdiffusion(
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


def save_layout_conditioning_json(cond, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    payload = {
        "obj_class": cond["obj_class"].detach().cpu().tolist(),
        "obj_bbox": cond["obj_bbox"].detach().cpu().tolist(),
        "is_valid_obj": cond["is_valid_obj"].detach().cpu().tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def convert_decoded_graph_to_sg_image_conditioning(nodes, triplets):
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


def build_text_scorer(opt, device):
    if opt.demon_scorer == "keyword":
        return KeywordTextGraphScorer()
    return CLIPTextGraphScorer(
        model_name=opt.clip_text_model,
        device=device,
    )


def get_sampling_masks_from_eval_item(opt, batch, i, device):
    """
    VG has node_mask/edge_mask from dataset.
    COCO does not, so create a fixed fully-connected mask.
    """
    node_mask_i = None
    edge_mask_i = None

    if "node_mask" in batch and batch["node_mask"][i] is not None:
        node_mask_i = batch["node_mask"][i].to(device).bool().unsqueeze(0)
        edge_mask_i = batch["edge_mask"][i].to(device).bool().unsqueeze(0)
    else:
        N = int(opt.eval_num_nodes)
        node_mask = torch.ones(N, dtype=torch.bool, device=device)
        edge_mask = torch.ones(N, N, dtype=torch.bool, device=device)
        edge_mask.fill_diagonal_(False)
        node_mask_i = node_mask.unsqueeze(0)
        edge_mask_i = edge_mask.unsqueeze(0)

    return node_mask_i, edge_mask_i


@torch.no_grad()
def sample_text_guided_sg(opt, model, obj_gen, dataset, prompt, node_mask, edge_mask, scorer):
    return run_full_reverse_chain_text_guided_demon_unconditional(
        model=model,
        obj_gen=obj_gen,
        node_mask=node_mask,
        edge_mask=edge_mask,
        T=opt.num_diffusion_steps - 1,
        prompt=prompt,
        scorer=scorer,
        object_vocab=dataset.object_vocab,
        relation_vocab=dataset.relation_vocab,
        mask_obj_token_id=opt.mask_obj_token_id,

        stochastic_obj=opt.unconditional_stochastic_obj,
        stochastic_edge=opt.unconditional_stochastic_edge,
        stochastic_rel=opt.unconditional_stochastic_rel,

        obj_temp=opt.unconditional_obj_temp,
        rel_temp=opt.unconditional_rel_temp,
        edge_logit_threshold=opt.unconditional_edge_logit_threshold,
        relation_edge_logit_threshold=opt.unconditional_relation_edge_logit_threshold,

        use_degree_pruning=opt.unconditional_use_degree_pruning,
        max_out_degree=opt.unconditional_max_out_degree,
        max_in_degree=opt.unconditional_max_in_degree,

        demon_num_candidates=opt.demon_num_candidates,
        demon_selection_mode=opt.demon_selection_mode,
        demon_guidance_scale=opt.demon_guidance_scale,
        demon_softmax_temperature=opt.demon_softmax_temperature,
        demon_every_n_steps=opt.demon_every_n_steps,
        demon_start_t=opt.demon_start_t,
        demon_end_t=opt.demon_end_t,
        demon_use_x0_proxy=opt.demon_use_x0_proxy,
        demon_verbose=opt.demon_verbose,
    )


def decode_sampled_sg(opt, sample_out, obj_gen, dataset, node_mask, edge_mask):
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
    return nodes, triplets, graph_text

def load_or_create_eval_index(opt, dataset):
    eval_index_path = opt.eval_index_json or os.path.join(opt.output_root, "eval_index.json")

    if getattr(opt, "reuse_eval_index", True) and os.path.exists(eval_index_path):
        with open(eval_index_path, "r", encoding="utf-8") as f:
            eval_index = json.load(f)

        eval_ids = [int(x["dataset_idx"]) for x in eval_index]
        print(f"[OK] Loaded eval index: {eval_index_path}")
        return eval_index, eval_ids, eval_index_path

    max_items = min(len(dataset), int(opt.max_eval_items))
    eval_ids = list(range(max_items))

    eval_index = []
    for idx in eval_ids:
        item = dataset[idx]
        eval_index.append({
            "file_id": f"{idx:06d}",
            "dataset_idx": int(idx),
            "source": str(item["source"]),
            "image_id": int(item["image_id"]),
            "text_prompt": str(item["text_prompt"]),
            "gt_image_path": item.get("gt_image_path", None),
        })

    os.makedirs(os.path.dirname(eval_index_path) or ".", exist_ok=True)
    with open(eval_index_path, "w", encoding="utf-8") as f:
        json.dump(eval_index, f, indent=2, ensure_ascii=False)

    print(f"[OK] Created eval index: {eval_index_path}")
    return eval_index, eval_ids, eval_index_path

@pyrallis.wrap()
def main(opt: DiscreteSGEvalConfig):
    set_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.mode not in {
        "text-sg",
        "text-sg-img",
        "text-sg-layout",
        "text-sg-layout-img",
    }:
        raise ValueError(f"Unknown mode: {opt.mode}")

    loader = build_eval_dataloader(opt)
    dataset = loader.dataset

    eval_index_path = opt.eval_index_json or os.path.join(opt.output_root, "eval_index.json")

    if opt.reuse_eval_index and os.path.exists(eval_index_path):
        with open(eval_index_path, "r") as f:
            eval_index = json.load(f)
        eval_ids = [x["dataset_idx"] for x in eval_index]
    else:
        eval_ids = list(range(min(len(dataset), opt.max_eval_items)))
        eval_index = []
        for idx in eval_ids:
            item = dataset[idx]
            eval_index.append({
                "dataset_idx": int(idx),
                "source": item["source"],
                "image_id": int(item["image_id"]),
                "text_prompt": item["text_prompt"],
                "gt_image_path": item["gt_image_path"],
            })
        os.makedirs(os.path.dirname(eval_index_path) or ".", exist_ok=True)
        with open(eval_index_path, "w") as f:
            json.dump(eval_index, f, indent=2)

    model, obj_gen, trainer, loaded_opt, ckpt = load_model_and_objgen_from_checkpoint(
        ckpt_path=opt.ckpt,
        device=device,
        strict=True,
    )
    model = model.to(device)
    model.eval()

    scorer = build_text_scorer(opt, device)

    sg_image_generator = None
    if opt.mode == "text-sg-img":
        sg_image_generator = SGConditionedImageGenerator(opt, device=device)

    layout_image_generator = None
    if opt.mode == "text-sg-layout-img":
        layout_image_generator = LayoutConditionedImageGenerator(opt)

    root = opt.output_root
    gt_img_dir = os.path.join(root, "gt_images")
    mode_root = os.path.join(root, opt.mode)

    ensure_dir(root)
    ensure_dir(gt_img_dir)
    ensure_dir(mode_root)

    if opt.mode in {"text-sg-img", "text-sg-layout-img"}:
        ensure_dir(os.path.join(mode_root, "gen_images"))

    if opt.mode in {"text-sg", "text-sg-img", "text-sg-layout", "text-sg-layout-img"}:
        ensure_dir(os.path.join(mode_root, "gen_sg"))

    if opt.mode in {"text-sg-layout", "text-sg-layout-img"}:
        ensure_dir(os.path.join(mode_root, "gen_layouts"))

    prompts_records = []

    global_idx = 0

    eval_index, eval_ids, eval_index_path = load_or_create_eval_index(opt, dataset)

    for global_idx, dataset_idx in enumerate(tqdm(eval_ids, desc=f"MasterEval[{opt.mode}]")):
        if global_idx >= opt.max_eval_items:
            break

        file_id = f"{global_idx:06d}"
        item = dataset[dataset_idx]

        prompt = item["text_prompt"]

        # -------------------------
        # Save GT image
        # -------------------------
        # use `item` instead of `batch`
        # save GT image
        if item["gt_image"] is not None:
            save_tensor_image(
                item["gt_image"],
                os.path.join(opt.output_root, "gt_images", f"img_{file_id}.png"),
            )
        
        node_mask, edge_mask = get_sampling_masks_from_eval_item(
            opt=opt,
            item=item,
            device=device,
        )

        # -------------------------
        # Sample text-conditioned SG
        # -------------------------

        sample_out = sample_text_guided_sg(
            opt=opt,
            model=model,
            obj_gen=obj_gen,
            dataset=dataset,
            prompt=prompt,
            node_mask=node_mask,
            edge_mask=edge_mask,
            scorer=scorer,
        )

        nodes, triplets, graph_text = decode_sampled_sg(
            opt=opt,
            sample_out=sample_out,
            obj_gen=obj_gen,
            dataset=dataset,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # -------------------------
        # Save generated SG
        # -------------------------
        save_triplets_tuple_format(
            triplets=triplets,
            out_path=os.path.join(mode_root, "gen_sg", f"sg_{file_id}.txt"),
        )

        with open(os.path.join(mode_root, "gen_sg", f"sg_pretty_{file_id}.txt"), "w") as f:
            f.write(graph_text)

        # -------------------------
        # Optional SG-conditioned image
        # -------------------------
        if opt.mode == "text-sg-img":
            if triplets is None or len(triplets) == 0:
                print(f"[WARN] sample {file_id} has no triplets; skipping image")
            else:
                cond = convert_decoded_graph_to_sg_image_conditioning(nodes, triplets)
                with open(os.path.join(mode_root, "gen_sg", f"conditioning_{file_id}.json"), "w") as f:
                    json.dump(cond, f, indent=2)

                sg_image_generator.generate(
                    all_triplets=cond["all_triplets"],
                    all_isolated_items=cond["all_isolated_items"],
                    all_global_ids=cond["all_global_ids"],
                    output_path=os.path.join(mode_root, "gen_images", f"img_{file_id}.png"),
                    seed=opt.seed + global_idx,
                )

        # -------------------------
        # Optional layout
        # -------------------------
        if opt.mode in {"text-sg-layout", "text-sg-layout-img"}:
            layout_box_final = sample_out.get("layout_box_final", None)
            if layout_box_final is None:
                raise ValueError("layout_box_final is None. Use a checkpoint with layout head enabled.")

            obj_final = sample_out["obj_final"][0].detach().cpu()
            layout_cxcywh = layout_box_final[0].detach().cpu()
            node_mask_i = node_mask[0].detach().cpu().long()

            cond_layout = build_layout_conditioning_for_layoutdiffusion(
                obj_final=obj_final,
                layout_box_final=layout_cxcywh,
                node_mask=node_mask_i,
                image_obj_class_id=opt.layout_image_obj_class_id,
            )

            save_layout_txt(
                obj_labels=cond_layout["obj_class"][0],
                boxes=cond_layout["obj_bbox"][0],
                node_mask=cond_layout["is_valid_obj"][0],
                object_vocab=dataset.object_vocab,
                out_path=os.path.join(mode_root, "gen_layouts", f"layout_{file_id}.txt"),
            )

            save_layout_conditioning_json(
                cond_layout,
                os.path.join(mode_root, "gen_layouts", f"layout_conditioning_{file_id}.json"),
            )

            if opt.mode == "text-sg-layout-img":
                layout_image_generator.generate(
                    obj_class=cond_layout["obj_class"],
                    obj_bbox=cond_layout["obj_bbox"],
                    is_valid_obj=cond_layout["is_valid_obj"],
                    output_path=os.path.join(mode_root, "gen_images", f"img_{file_id}.png"),
                    seed=opt.seed + global_idx,
                )

            global_idx += 1

        if global_idx >= opt.max_eval_items:
            break

    save_prompts_json(
        prompts_records,
        os.path.join(root, "text_prompts.json"),
    )

    print(f"[DONE] Saved {global_idx} examples to: {root}")


if __name__ == "__main__":
    main()