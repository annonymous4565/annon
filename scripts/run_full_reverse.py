import os
import torch
from tqdm import tqdm
import pyrallis

try:
    import wandb
except ImportError:
    wandb = None

from utils.checkpoint_loading import load_model_and_objgen_from_checkpoint

from sampling.reward_terms import build_relation_group_ids, compute_sg_layout_reward_terms, combine_reward_terms


from datasets_.visual_genome.dataset import (
    build_scene_graph_dataloader,
    decode_item,
    format_decoded_graph,
    format_graph_triplets_only,
    format_nodes_with_boxes
)
from sampling.full_reverse_sampler import run_full_reverse_chain

from configs import DiscreteSGConfig

from utils.graph_state_utils import build_full_relation_from_structured_state

# NEW: flowchart helpers
from utils.wandb_utils import (
    WandBLogger,
    render_graph_text_block_to_image,
    ensure_dir,
    build_eval_graph_comparison_table,
    build_graph_image_comparison_table,
    render_layout_boxes_to_image,
)

@torch.no_grad()
def evaluate_sampler(
    model,
    opt,
    obj_gen,
    dataloader,
    trainer,
    device,
    T,
    max_batches=3,
    stochastic_obj=False,
    stochastic_edge=False,
    stochastic_rel=False,
    wandb_logger=None,
):
    model.eval()

    total = {
        "node_correct": 0.0,
        "node_total": 0.0,
        "edge_tp": 0.0,
        "edge_fp": 0.0,
        "edge_fn": 0.0,
    }

    rows = []
    img_rows = []
    examples_logged = 0
    max_examples_to_log = min(getattr(opt, "wandb_num_val_fullrev_graphs_to_log", 3), 16)

    draw_flowchart = getattr(opt, "draw_flowchart", False)
    flowchart_rankdir = getattr(opt, "flowchart_rankdir", "LR")
    flowchart_format = getattr(opt, "flowchart_format", "png")
    flowchart_show_node_ids = getattr(opt, "flowchart_show_node_ids", True)
    flowchart_log_table = getattr(opt, "flowchart_log_table", True)
    flowchart_log_individual_images = getattr(opt, "flowchart_log_individual_images", True)
    flowchart_out_dir = getattr(opt, "flowchart_out_dir", "./visualization")

    save_layout_boxes_only = getattr(opt, "save_layout_boxes_only", False)
    layout_box_image_size = getattr(opt, "layout_box_image_size", 256)
    layout_log_individual_images = getattr(opt, "layout_log_individual_images", True)

    eval_vis_dir = os.path.join(flowchart_out_dir, "full_reverse_eval")
    if draw_flowchart:
        ensure_dir(eval_vis_dir)

    iterator = tqdm(dataloader, desc="FullReverseEval")
    for bidx, batch in enumerate(iterator):
        if max_batches is not None and bidx >= max_batches:
            break

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        batch_clean_ = trainer.obj_gen.get_training_batch(batch, force_t=0)
        
        batch_clean = {
            "obj_0": batch_clean_["obj_0"],
            "edge_0": batch_clean_["edge_0"],
            "boxes_0": batch_clean_["boxes_0"],
            "rel_pos_0": batch_clean_["rel_pos_0"],
            "node_mask": batch_clean_["node_mask"],
            "edge_mask": batch_clean_["edge_mask"],
        }

        if "boxes" in batch:
            batch_clean["boxes"] = batch["boxes"].to(device)  # [B,N,4]
            batch_clean["box_valid_mask"] = batch["node_mask"].bool().to(device)

        reward_relation_group_ids = build_relation_group_ids(trainer.train_dataset.relation_vocab)

        reward_weights = {
                "reward_isolated_node": getattr(opt, "reward_w_isolated_node", 0.25),
                "reward_bidirectional_edge": getattr(opt, "reward_w_bidirectional_edge", 0.10),
                "reward_dense_graph": getattr(opt, "reward_w_dense_graph", 0.10),
                "reward_box_bounds": getattr(opt, "reward_w_box_bounds", 0.25),
                "reward_layout_overlap": getattr(opt, "reward_w_layout_overlap", 1.00),
                "reward_layout_spread": getattr(opt, "reward_w_layout_spread", 0.50),
                "reward_relation_geometry": getattr(opt, "reward_w_relation_geometry", 0.50),
            }

        reward_relation_group_pos_ids = {}

        for group_name, ids in reward_relation_group_ids.items():
            pos_ids = []
            for rid in ids:
                # reward_relation_group_ids are full relation vocab ids.
                # rel_pos ids are shifted by -1 because full id 0 is __no_relation__.
                if int(rid) != opt.no_rel_token_id:
                    pos_ids.append(int(rid) - 1)
            reward_relation_group_pos_ids[group_name] = pos_ids

        reward_obj_log_prior = trainer.build_reward_object_log_prior()

        sample_out = run_full_reverse_chain(
            model=model,
            obj_gen=obj_gen,
            batch_clean=batch_clean,
            T=T,
            stochastic_obj=stochastic_obj,
            stochastic_edge=stochastic_edge,
            stochastic_rel=stochastic_rel,
            return_trace=False,
            use_reverse_vocab_heads=getattr(opt, "full_reverse_use_reverse_vocab_heads", False),
            obj_temp=getattr(opt, "full_reverse_obj_temp", 1.0),
            rel_temp=getattr(opt, "full_reverse_rel_temp", 1.0),
            edge_logit_threshold=getattr(opt, "full_reverse_edge_logit_threshold", 0.0),
            relation_edge_logit_threshold=getattr(opt, "full_reverse_relation_edge_logit_threshold", 0.0),
            use_degree_pruning=getattr(opt, "full_reverse_use_degree_pruning", False),
            max_out_degree=getattr(opt, "full_reverse_max_out_degree", 0),
            max_in_degree=getattr(opt, "full_reverse_max_in_degree", 0),
            use_final_step_cleanup=getattr(opt, "full_reverse_use_final_step_cleanup", False),
            final_edge_logit_threshold=getattr(opt, "full_reverse_final_edge_logit_threshold", 0.5),
            final_rel_conf_threshold=getattr(opt, "full_reverse_final_rel_conf_threshold", 0.0),
            generic_obj_ids=getattr(opt, "full_reverse_generic_obj_ids", []),
            generic_attachment_rel_ids=getattr(opt, "full_reverse_generic_attachment_rel_ids", []),
            generic_attachment_edge_logit_threshold=getattr(opt, "full_reverse_generic_attachment_edge_logit_threshold", 1.0,),
            reward_fn=None,
            use_reward_tilting=getattr(opt, "use_reward_tilting", False),
            reward_tilt_alpha=getattr(opt, "reward_tilt_alpha", 1.0),
            reward_tilt_temperature=getattr(opt, "reward_tilt_temperature", 1.0),
            reward_tilt_num_sweeps=getattr(opt, "reward_tilt_num_sweeps", 1),
            reward_tilt_objects=getattr(opt, "reward_tilt_objects", False),
            reward_tilt_edges=getattr(opt, "reward_tilt_edges", False),
            reward_tilt_relations=getattr(opt, "reward_tilt_relations", False),
            reward_tilt_use_layout=getattr(opt, "reward_tilt_use_layout", False),
            reward_tilt_obj_topk=getattr(opt, "reward_tilt_obj_topk", 5),
            reward_tilt_rel_topk=getattr(opt, "reward_tilt_rel_topk", 5),
            reward_weights=reward_weights,
            reward_tilt_edge_logit_band=getattr(opt, "reward_tilt_edge_logit_band", 0.75),
            reward_w_hub_degree=getattr(opt, "reward_w_hub_degree", 0.50),
            reward_hub_degree_threshold=getattr(opt, "reward_hub_degree_threshold", 4),
            reward_relation_group_pos_ids=reward_relation_group_pos_ids,
            reward_tilt_relation_alpha=getattr(opt, "reward_tilt_relation_alpha", 0.5),
            reward_w_relation_geometry_tilt=getattr(opt, "reward_w_relation_geometry_tilt", 1.0),
            reward_obj_log_prior=reward_obj_log_prior,
            reward_tilt_object_alpha=getattr(opt, "reward_tilt_object_alpha", 0.25),
            reward_w_object_class_prior_tilt=getattr(opt, "reward_w_object_class_prior_tilt", 0.50),
            reward_w_object_relation_support_tilt=getattr(opt, "reward_w_object_relation_support_tilt", 0.25),
            reward_tilt_obj_logit_margin=getattr(opt, "reward_tilt_obj_logit_margin", 1.0),
            reward_tilt_layout_alpha=getattr(opt, "reward_tilt_layout_alpha", 0.25),
            reward_w_layout_overlap_tilt=getattr(opt, "reward_w_layout_overlap_tilt", 1.0),
            reward_w_layout_spread_tilt=getattr(opt, "reward_w_layout_spread_tilt", 0.5),
            reward_w_box_bounds_tilt=getattr(opt, "reward_w_box_bounds_tilt", 0.5),
        )

        obj_final = sample_out["obj_final"]
        edge_final = sample_out["edge_final"]
        rel_pos_final = sample_out["rel_pos_final"]
        layout_box_final = sample_out["layout_box_final"]

        start_state = sample_out["start_state"]

        obj_gt = batch_clean["obj_0"]
        edge_gt = batch_clean["edge_0"]
        rel_pos_gt = batch_clean["rel_pos_0"]

        node_mask = batch_clean["node_mask"]
        edge_mask = batch_clean["edge_mask"]

        # aggregate metrics
        correct = (obj_final == obj_gt) & node_mask
        total["node_correct"] += float(correct.sum().item())
        total["node_total"] += float(node_mask.sum().item())

        tp = ((edge_final == 1) & (edge_gt == 1) & edge_mask).sum()
        fp = ((edge_final == 1) & (edge_gt == 0) & edge_mask).sum()
        fn = ((edge_final == 0) & (edge_gt == 1) & edge_mask).sum()

        total["edge_tp"] += float(tp.item())
        total["edge_fp"] += float(fp.item())
        total["edge_fn"] += float(fn.item())

        # log a few examples in trainer-like format
        B = obj_final.shape[0]
        for i in range(B):
            if examples_logged >= max_examples_to_log:
                break

            rel_full_clean_i = build_full_relation_from_structured_state(
                edge_t=edge_gt[i].detach().cpu(),
                rel_pos_t=rel_pos_gt[i].detach().cpu(),
                no_rel_token_id=0,
                num_rel_pos_classes=obj_gen.num_rel_pos_classes,
            )

            rel_full_start_i = build_full_relation_from_structured_state(
                edge_t=start_state["edge_t"][i].detach().cpu(),
                rel_pos_t=start_state["rel_pos_t"][i].detach().cpu(),
                no_rel_token_id=0,
                num_rel_pos_classes=obj_gen.num_rel_pos_classes,
            )

            rel_full_final_i = build_full_relation_from_structured_state(
                edge_t=edge_final[i].detach().cpu(),
                rel_pos_t=rel_pos_final[i].detach().cpu(),
                no_rel_token_id=0,
                num_rel_pos_classes=obj_gen.num_rel_pos_classes,
            )

            clean_nodes, clean_trips = decode_item(
                obj_labels=obj_gt[i].detach().cpu(),
                rel_labels=rel_full_clean_i,
                node_mask=node_mask[i].detach().cpu(),
                edge_mask=edge_mask[i].detach().cpu(),
                object_vocab=dataloader.dataset.object_vocab,
                relation_vocab=dataloader.dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=opt.mask_obj_token_id,
            )

            noisy_nodes, noisy_trips = decode_item(
                obj_labels=start_state["obj_t"][i].detach().cpu(),
                rel_labels=rel_full_start_i,
                node_mask=node_mask[i].detach().cpu(),
                edge_mask=edge_mask[i].detach().cpu(),
                object_vocab=dataloader.dataset.object_vocab,
                relation_vocab=dataloader.dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=opt.mask_obj_token_id,
            )

            pred_nodes, pred_trips = decode_item(
                obj_labels=obj_final[i].detach().cpu(),
                rel_labels=rel_full_final_i,
                node_mask=node_mask[i].detach().cpu(),
                edge_mask=edge_mask[i].detach().cpu(),
                object_vocab=dataloader.dataset.object_vocab,
                relation_vocab=dataloader.dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=opt.mask_obj_token_id,
            )

            clean_text = format_decoded_graph(clean_nodes, clean_trips)
            noisy_text = format_decoded_graph(noisy_nodes, noisy_trips)
            pred_text = format_decoded_graph(pred_nodes, pred_trips)

            gt_boxes = batch_clean.get("boxes", None)
            box_valid_mask = batch_clean.get("box_valid_mask", None)
            pred_boxes = sample_out.get("layout_box_final", None)
            clean_boxes_text = format_nodes_with_boxes(
                nodes=clean_nodes,
                boxes=gt_boxes[i].detach().cpu() if gt_boxes is not None else None,
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )
            pred_boxes_text = format_nodes_with_boxes(
                nodes=pred_nodes,
                boxes=pred_boxes[i].detach().cpu() if pred_boxes is not None else None,
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            # per-example diagnostics
            gt_pos = (rel_full_clean_i != 0) & edge_mask[i].detach().cpu().bool()
            start_pos = (rel_full_start_i != 0) & edge_mask[i].detach().cpu().bool()
            pred_pos = (rel_full_final_i != 0) & edge_mask[i].detach().cpu().bool()

            tp_i = int((pred_pos & gt_pos).sum().item())
            fp_i = int((pred_pos & (~gt_pos)).sum().item())
            fn_i = int(((~pred_pos) & gt_pos).sum().item())

            edge_precision_i = tp_i / max(tp_i + fp_i, 1)
            edge_recall_i = tp_i / max(tp_i + fn_i, 1)
            edge_f1_i = 2.0 * edge_precision_i * edge_recall_i / max(edge_precision_i + edge_recall_i, 1e-12)

            node_mask_i = node_mask[i].detach().cpu().bool()
            obj_gt_i = obj_gt[i].detach().cpu()
            obj_start_i = start_state["obj_t"][i].detach().cpu()
            obj_pred_i = obj_final[i].detach().cpu()

            corrupt_mask_i = (obj_start_i != obj_gt_i) & node_mask_i
            num_nodes_i = int(node_mask_i.sum().item())
            num_corrupt_i = int(corrupt_mask_i.sum().item())
            pred_match_gt_i = int(((obj_pred_i == obj_gt_i) & node_mask_i).sum().item())
            pred_match_noisy_i = int(((obj_pred_i == obj_start_i) & node_mask_i).sum().item())

            node_acc_all_i = pred_match_gt_i / max(num_nodes_i, 1)
            node_acc_corr_i = (
                int(((obj_pred_i == obj_gt_i) & corrupt_mask_i).sum().item()) / num_corrupt_i
                if num_corrupt_i > 0 else 0.0
            )

            diag_text = (
                f"=== DIAGNOSTICS ===\n"
                f"gt_pos_edges: {int(gt_pos.sum().item())}\n"
                f"start_noisy_pos_edges: {int(start_pos.sum().item())}\n"
                f"final_pos_edges: {int(pred_pos.sum().item())}\n"
                f"tp_edges: {tp_i}\n"
                f"fp_edges: {fp_i}\n"
                f"fn_edges: {fn_i}\n"
                f"edge_precision: {edge_precision_i:.4f}\n"
                f"edge_recall: {edge_recall_i:.4f}\n"
                f"edge_f1: {edge_f1_i:.4f}\n"
                f"\n"
                f"num_nodes: {num_nodes_i}\n"
                f"true_corrupt_nodes: {num_corrupt_i}\n"
                f"noisy_diff_from_gt: {int((obj_start_i[node_mask_i] != obj_gt_i[node_mask_i]).sum().item())}\n"
                f"pred_match_gt: {pred_match_gt_i}\n"
                f"pred_match_noisy: {pred_match_noisy_i}\n"
                f"node_acc_all: {node_acc_all_i:.4f}\n"
                f"node_acc_corrupted: {node_acc_corr_i:.4f}"
            )

            joined = (
                f"{diag_text}\n\n"
                f"=== CLEAN ===\n{clean_text}\n\n"
                f"=== CLEAN BOXES ===\n{clean_boxes_text}\n\n"
                f"=== START NOISY (t={T}) ===\n{noisy_text}\n\n"
                f"=== FINAL SAMPLED x0 ===\n{pred_text}"
                f"=== FINAL SAMPLED BOXES ===\n{pred_boxes_text}"
            )

            if wandb_logger is not None:
                wandb_logger.log_text(
                    key=f"eval_fullrev_graphs/example_{examples_logged}",
                    text=joined,
                    step=0,
                )

            rows.append({
                "example_id": examples_logged,
                "timestep": int(T),
                "gt_pos_edges": int(gt_pos.sum().item()),
                "start_noisy_pos_edges": int(start_pos.sum().item()),
                "final_pos_edges": int(pred_pos.sum().item()),
                "tp_edges": tp_i,
                "fp_edges": fp_i,
                "fn_edges": fn_i,
                "edge_precision": round(edge_precision_i, 4),
                "edge_recall": round(edge_recall_i, 4),
                "edge_f1": round(edge_f1_i, 4),
                "num_nodes": num_nodes_i,
                "true_corrupt_nodes": num_corrupt_i,
                "noisy_diff_from_gt": int((obj_start_i[node_mask_i] != obj_gt_i[node_mask_i]).sum().item()),
                "pred_match_gt": pred_match_gt_i,
                "pred_match_noisy": pred_match_noisy_i,
                "node_acc_all": round(node_acc_all_i, 4),
                "node_acc_corrupted": round(node_acc_corr_i, 4),
                "clean_graph": format_graph_triplets_only(clean_trips),
                "start_noisy_graph": format_graph_triplets_only(noisy_trips),
                "final_graph": format_graph_triplets_only(pred_trips),
                "clean_boxes": clean_boxes_text,
                "final_boxes": pred_boxes_text,
            })

            if draw_flowchart:
                try:
                    ex_dir = os.path.join(eval_vis_dir, f"example_{examples_logged:03d}")
                    ensure_dir(ex_dir)

                    clean_img = render_graph_text_block_to_image(
                        graph_text=clean_text,
                        out_path_no_ext=os.path.join(ex_dir, "clean"),
                        title=f"Clean Graph | ex={examples_logged}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    noisy_img = render_graph_text_block_to_image(
                        graph_text=noisy_text,
                        out_path_no_ext=os.path.join(ex_dir, "start_noisy"),
                        title=f"Start Noisy Graph | ex={examples_logged} | t={T}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    pred_img = render_graph_text_block_to_image(
                        graph_text=pred_text,
                        out_path_no_ext=os.path.join(ex_dir, "final"),
                        title=f"Final Sampled x0 | ex={examples_logged}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    img_rows.append({
                        "example_id": examples_logged,
                        "timestep": int(T),
                        "clean_graph_img": clean_img,
                        "noisy_graph_img": noisy_img,
                        "pred_graph_img": pred_img,
                    })

                    if wandb_logger is not None and flowchart_log_individual_images:
                        wandb_logger.log(
                            {
                                f"eval_fullrev_graphs_visual/example_{examples_logged}/clean": wandb.Image(clean_img, caption=f"clean ex={examples_logged}"),
                                f"eval_fullrev_graphs_visual/example_{examples_logged}/start_noisy": wandb.Image(noisy_img, caption=f"start noisy ex={examples_logged}"),
                                f"eval_fullrev_graphs_visual/example_{examples_logged}/final": wandb.Image(pred_img, caption=f"final ex={examples_logged}"),
                            },
                            step=0,
                        )

                except Exception as e:
                    print(f"[WARN] Flowchart render failed for example {examples_logged}: {e}")
            
            if save_layout_boxes_only:
                try:
                    ex_dir = os.path.join(eval_vis_dir, f"example_{examples_logged:03d}")
                    ensure_dir(ex_dir)

                    clean_layout_img = render_layout_boxes_to_image(
                        obj_class=batch_clean["obj_0"][i].detach().cpu(),
                        obj_bbox=batch_clean["boxes_0"][i].detach().cpu(),
                        is_valid_obj=batch_clean["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(ex_dir, f"example_{i}_layout_clean.png"),
                        class_names=dataloader.dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    pred_layout_img = render_layout_boxes_to_image(
                        obj_class=obj_final[i].detach().cpu(),
                        obj_bbox=layout_box_final[i].detach().cpu(),
                        is_valid_obj=batch_clean["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(ex_dir, f"example_{i}_layout_pred.png"),
                        class_names=dataloader.dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    if layout_log_individual_images:
                        wandb_logger.log(
                            {
                                f"eval_fullrev_layout_visual/example_{i}/clean_boxes": wandb.Image(
                                    clean_layout_img,
                                    caption=f"clean layout ex={i}"
                                ),
                                f"eval_fullrev_layout_visual/example_{i}/pred_boxes": wandb.Image(
                                    pred_layout_img,
                                    caption=f"pred layout ex={i}"
                                ),
                            },
                            step=0,
                        )

                except Exception as e:
                    print(f"[WARN] Full reverse Layout box render failed for example {i}: {e}")

            examples_logged += 1

    node_acc = total["node_correct"] / max(total["node_total"], 1.0)
    precision = total["edge_tp"] / max(total["edge_tp"] + total["edge_fp"], 1.0)
    recall = total["edge_tp"] / max(total["edge_tp"] + total["edge_fn"], 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    print("\n=== FULL REVERSE METRICS ===")
    print(f"node_acc: {node_acc:.4f}")
    print(f"edge_precision: {precision:.4f}")
    print(f"edge_recall: {recall:.4f}")
    print(f"edge_f1: {f1:.4f}")

    if wandb_logger is not None:
        wandb_logger.log(
            {
                "eval/fullrev_node_acc": node_acc,
                "eval/fullrev_edge_precision": precision,
                "eval/fullrev_edge_recall": recall,
                "eval/fullrev_edge_f1": f1,
            },
            step=0,
        )

        if not draw_flowchart:
            table = build_eval_graph_comparison_table(rows)
            if table is not None:
                wandb_logger.log(
                    {"eval/fullrev_graph_comparisons": table},
                    step=0,
                )
        elif flowchart_log_table:
            img_table = build_graph_image_comparison_table(img_rows)
            if img_table is not None:
                wandb_logger.log(
                    {"eval/fullrev_graph_comparisons_flowchart": img_table},
                    step=0,
                )

@pyrallis.wrap()
def main(opt: DiscreteSGConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger = WandBLogger(opt)
    wandb_logger.init()

    # Make sure this unpacking matches your checkpoint loader implementation
    model, obj_gen, trainer, cfg, ckpt = load_model_and_objgen_from_checkpoint(
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

    evaluate_sampler(
        model=model,
        opt=opt,
        obj_gen=obj_gen,
        dataloader=loader,
        trainer=trainer,
        device=device,
        T=opt.num_diffusion_steps - 1,
        max_batches=opt.full_reverse_eval_max_batches,
        stochastic_obj=opt.full_reverse_stochastic_obj,
        stochastic_edge=opt.full_reverse_stochastic_edge,
        stochastic_rel=opt.full_reverse_stochastic_rel,
        wandb_logger=wandb_logger,
    )

    wandb_logger.finish()


if __name__ == "__main__":
    main()