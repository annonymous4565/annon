import os
from dataclasses import asdict
import time
import math
from typing import Optional, Dict, Any, List, Tuple
import json
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

from datasets_.visual_genome.dataset import SceneGraphDataset
from datasets_.visual_genome.collate import scene_graph_collate_fn

from diffusion.structured_objective_generator import StructuredSGObjectiveGenerator
from diffusion.sg_state_utils import reconstruct_full_relations

from utils.layout_vis import draw_layout_boxes, draw_layout_boxes_on_image

from models.structured_sg_diffusion import StructuredSceneGraphDiffusionModel

from sampling.reward_terms import build_relation_group_ids, compute_sg_layout_reward_terms, combine_reward_terms


from sampling.full_reverse_sampler import (
    run_full_reverse_chain,
    sample_prev_state_from_current_batch,
    run_full_reverse_chain_unconditional
)

from evaluation.graph_metrics import SceneGraphSample, evaluate_graph_generation
from utils.graph_state_utils import build_full_relation_from_structured_state

from evaluation.nearest_neighbor_analysis import (
    rank_nearest_neighbors,
    summarize_nn_set,
)

from training.structured_losses import (
    compute_structured_sg_loss,
    compute_reverse_vocab_step_loss,
    compute_conditional_node_loss,
    compute_refinement_sg_loss,
    build_discrete_predictions_from_model_out,
    build_object_class_weights_effective_num,
    compute_node_accuracy_metrics,
    compute_conditional_node_loss,
    build_discrete_predictions_from_model_out,
    compute_final_graph_metrics,
    compute_layout_regularizers
)


from sampling.node_gibbs import run_node_gibbs_sampler

from diffusion.node_semantic_kernel import (
    build_object_context_features,
    build_similarity_matrix_from_features
)

from training.distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    reduce_scalar_sum,
    barrier,
)

from utils.wandb_utils import (
    WandBLogger,
    build_graph_comparison_table,
    build_eval_graph_comparison_table,
    build_epoch_metrics_table,
    build_graph_image_comparison_table,
    render_graph_text_block_to_image,
    render_layout_boxes_to_image,
    ensure_dir
)

from datasets_.visual_genome.dataset import (
    decode_item,
    format_decoded_graph,
    format_graph_triplets_only,
    format_nodes_with_boxes
)

def _t():
    return time.time()

def _log_time(msg, start):
    print(f"[TIMER] {msg}: {time.time() - start:.2f}s")

def unwrap_dataset(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds

    
def compute_topk_obj_classes(dataset, k: int, num_obj_classes: int):
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

    obj = base.obj_labels
    mask = base.node_mask.astype(bool)

    vals = obj[mask]
    vals = vals[vals >= 0]

    counts = np.bincount(vals, minlength=num_obj_classes)
    topk = np.argsort(-counts)[:k]

    return torch.tensor(topk, dtype=torch.long)

class StructuredSGDDPTrainer:
    def __init__(self, opt):
        self.opt = opt

        (
            self.is_distributed,
            self.rank,
            self.world_size,
            self.local_rank,
            self.device,
        ) = setup_distributed()
        if is_main_process():
            print("Loading Train dataset")
        self.train_dataset = SceneGraphDataset(
            npz_path=opt.train_npz_path,
            return_boxes=True,
            return_metadata=False,
        )
        if is_main_process():
            print("Loading Val dataset")
        self.val_dataset = None
        if opt.val_npz_path is not None:
            self.val_dataset = SceneGraphDataset(
                npz_path=opt.val_npz_path,
                return_boxes=True,
                return_metadata=False,
            )
        
        if getattr(opt, "sanity_overfit_tiny", False):
            n_train = getattr(opt, "sanity_overfit_num_graphs_train", 128)
            n_val = getattr(opt, "sanity_overfit_num_graphs_val", 128)
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, list(range(n_train)))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, list(range(n_val)))

        if is_main_process():
            print("Initializing Train dataloader")
        self.train_sampler = (
            DistributedSampler(self.train_dataset, shuffle=True)
            if self.is_distributed else None
        )
        if is_main_process():
            print("Initializing Val dataloader")
        self.val_sampler = (
            DistributedSampler(self.val_dataset, shuffle=False)
            if (self.is_distributed and self.val_dataset is not None) else None
        )


        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=opt.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=opt.num_workers,
            pin_memory=True,
            collate_fn=scene_graph_collate_fn,
            drop_last=False,
            persistent_workers=(opt.num_workers > 0),
        )

        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=opt.eval_batch_size,
                shuffle=False,
                sampler=self.val_sampler,
                num_workers=opt.num_workers,
                pin_memory=True,
                collate_fn=scene_graph_collate_fn,
                drop_last=False,
                persistent_workers=(opt.num_workers > 0),
            )
        
        # self.object_vocab = self.val_loader.dataset.object_vocab
        # self.relation_vocab = self.val_loader.dataset.relation_vocab
        # self.opt.num_obj_classes=len(self.train_dataset.object_vocab)
        # self.opt.num_rel_pos_classes = len(self.train_dataset.relation_vocab) - 1

        base_ds = unwrap_dataset(self.val_loader.dataset)
        self.object_vocab = base_ds.object_vocab
        self.relation_vocab = base_ds.relation_vocab

        base_train = unwrap_dataset(self.train_dataset)
        self.train_dataset = base_train

        base_val = unwrap_dataset(self.val_dataset)
        self.val_dataset = base_val

        self.opt.num_obj_classes=len(self.train_dataset.object_vocab)
        self.opt.num_rel_pos_classes = len(self.train_dataset.relation_vocab) - 1

        self.best_val_node_corr_by_t = {10: 0.0, 25: 0.0, 30: 0.0, 35: 0.0, 40: 0.0, 49: 0.0}
        self.current_curriculum_t = 10

        self.best_val_node_corr_t10 = 0.0
        self.best_val_node_corr_t25 = 0.0
        self.best_val_node_corr_t49 = 0.0
        
        self.topk_obj_classes = None

        if getattr(self.opt, "topk_obj_loss_only", False):
            self.topk_obj_classes = compute_topk_obj_classes(
                self.train_dataset,
                self.opt.topk_obj_k,
                self.opt.num_obj_classes
            )

        # t0 = time.time()
        obj_feat = build_object_context_features(
            dataset=self.train_dataset,
            num_obj_classes=len(self.train_dataset.object_vocab),
            num_rel_pos_classes=len(self.train_dataset.relation_vocab) - 1,
            no_rel_token_id=self.opt.no_rel_token_id,
        )
        # print(f"[TIMING] build_object_context_features: {time.time() - t0:.2f}s")

        # t1 = time.time()
        self.obj_similarity_matrix = build_similarity_matrix_from_features(
            obj_feat,
            temperature=getattr(self.opt, "node_semantic_temp", 0.1),
            self_bias=getattr(self.opt, "node_semantic_self_bias", 0.0),
            topk=getattr(self.opt, "node_semantic_topk", None),
        ).to(self.device)
        # print(f"[TIMING] build_similarity_matrix_from_features: {time.time() - t1:.2f}s")

        self.model = StructuredSceneGraphDiffusionModel(
            num_obj_classes=len(self.train_dataset.object_vocab),
            num_rel_classes_full=len(self.train_dataset.relation_vocab),
            d_model=self.opt.d_model,
            num_layers=self.opt.num_layers,
            dropout=self.opt.dropout,
            num_relation_buckets=getattr(self.opt, "num_relation_buckets", 8),
            use_relation_bucket_node_conditioning=getattr(self.opt, "use_relation_bucket_node_conditioning", False),
            use_reverse_vocab_heads=getattr(self.opt, "use_reverse_vocab_heads", False),
            use_layout_head=opt.use_layout_head,
            layout_hidden_dim=opt.layout_hidden_dim,
            n_max=opt.n_max,
            object_from_structure_only=getattr(self.opt, "object_from_structure_only", False),
            object_condition_on_gt_structure = getattr(self.opt, "object_condition_on_gt_structure", False),
            use_direct_obj_head = getattr(self.opt, "use_direct_obj_head", False), 


        ).to(self.device)

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )

        self.obj_gen = StructuredSGObjectiveGenerator(
            cfg=opt,
            num_obj_classes=len(self.train_dataset.object_vocab),
            num_rel_classes_full=len(self.train_dataset.relation_vocab),
            device=self.device,
            obj_similarity_matrix=self.obj_similarity_matrix,
        )

        self.obj_class_weights = build_object_class_weights_effective_num(
            dataset=self.train_dataset,
            beta=getattr(self.opt, "obj_effective_num_beta", 0.999),
            min_weight=getattr(self.opt, "obj_weight_min", 0.25),
            max_weight=getattr(self.opt, "obj_weight_max", 5.0),
            device=self.device,
        )

        self.object_class_counts = self.compute_object_class_counts()
        self.object_freq_bucket = self.build_object_frequency_buckets(self.object_class_counts)

        self.opt.mask_obj_token_id = self.unwrap_model().mask_obj_token_id
        self.opt.mask_rel_token_id = self.unwrap_model().mask_rel_token_id

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_rel_loss = float("inf")
        self.best_val_obj_loss = float("inf")

        self.best_fullrev_recon_composite = float("-inf")
        self.best_fullrev_structure_composite = float("-inf")

        self.reverse_rel_class_weights = None
        if getattr(self.opt, "use_reverse_rel_class_weights", False):
            self.reverse_rel_class_weights = self.build_reverse_relation_class_weights()
        
        # if self.reverse_rel_class_weights is not None:
        #     print("Reverse rel weights:", self.reverse_rel_class_weights[:10])
        #     print("min/max:",
        #         self.reverse_rel_class_weights.min().item(),
        #         self.reverse_rel_class_weights.max().item())

        if getattr(self.opt, "use_empirical_unconditional_priors", True):
            prior_dict = self.estimate_empirical_unconditional_priors(
                max_batches=getattr(self.opt, "empirical_prior_max_batches", None)
            )
            self.obj_gen.set_empirical_unconditional_priors(
                obj_prior=prior_dict["obj_prior"],
                edge_prior=prior_dict["edge_prior"],
                rel_prior=prior_dict["rel_prior"],
            )

            # if is_main_process():
            #     print("[EmpiricalPriors] obj_prior sum:", float(prior_dict["obj_prior"].sum().item()))
            #     print("[EmpiricalPriors] edge_prior:", prior_dict["edge_prior"].detach().cpu().tolist())
            #     print("[EmpiricalPriors] rel_prior sum:", float(prior_dict["rel_prior"].sum().item()))
            #     print("[EmpiricalPriors] top obj prior ids:",
            #         torch.topk(prior_dict["obj_prior"], k=10).indices.detach().cpu().tolist())
            #     print("[EmpiricalPriors] edge prior:",
            #         prior_dict["edge_prior"].detach().cpu().tolist())
            #     print("[EmpiricalPriors] top rel prior ids:",
            #         torch.topk(prior_dict["rel_prior"][:-1], k=10).indices.detach().cpu().tolist())

        if opt.train_mode:
            self.wandb_logger = WandBLogger(opt)
            self.wandb_logger.init()
            self.epoch_metrics_history = []
        
        # if is_main_process() and getattr(self.opt, "use_relation_geometry_loss", False):
        #     self.log_relation_geometry_groups()

        self.layout_prior_mean = None
        self.layout_prior_var = None
        self.layout_prior_valid = None
        self.layout_prior_count = None
        self.reward_weights = None
        self.reward_obj_log_prior = None
        self.reward_relation_group_pos_ids = {}

        if getattr(self.opt, "use_layout_class_priors", False):
            self.build_layout_class_priors()
        
        

        
        if opt.use_reward_tilting:
            self.reward_relation_group_ids = build_relation_group_ids(self.train_dataset.relation_vocab)

            self.reward_weights = {
                "reward_isolated_node": getattr(self.opt, "reward_w_isolated_node", 0.25),
                "reward_bidirectional_edge": getattr(self.opt, "reward_w_bidirectional_edge", 0.10),
                "reward_dense_graph": getattr(self.opt, "reward_w_dense_graph", 0.10),
                "reward_box_bounds": getattr(self.opt, "reward_w_box_bounds", 0.25),
                "reward_layout_overlap": getattr(self.opt, "reward_w_layout_overlap", 1.00),
                "reward_layout_spread": getattr(self.opt, "reward_w_layout_spread", 0.50),
                "reward_relation_geometry": getattr(self.opt, "reward_w_relation_geometry", 0.50),
            }

            for group_name, ids in self.reward_relation_group_ids.items():
                pos_ids = []
                for rid in ids:
                    # reward_relation_group_ids are full relation vocab ids.
                    # rel_pos ids are shifted by -1 because full id 0 is __no_relation__.
                    if int(rid) != self.opt.no_rel_token_id:
                        pos_ids.append(int(rid) - 1)
                self.reward_relation_group_pos_ids[group_name] = pos_ids

            self.reward_obj_log_prior = self.build_reward_object_log_prior()


        if is_main_process():
            self.wandb_logger.watch(self.unwrap_model(), log="gradients", log_freq=500)

        if is_main_process():
            os.makedirs(opt.checkpoint_dir, exist_ok=True)

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model
    
    def slice_batch_dict(self, batch_dict: dict, keep_mask: torch.Tensor) -> dict:
        """
        Slice all batch-first tensors in a batch_dict by a boolean mask.
        Non-batched items are passed through unchanged.
        """
        out = {}
        B = keep_mask.shape[0]

        for k, v in batch_dict.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
                out[k] = v[keep_mask]
            else:
                out[k] = v
        return out

    def train(self):
        assert self.opt.val_every_epoch >= 1
        assert self.opt.save_every_epoch >= 1

        for epoch in range(self.opt.num_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            train_metrics = self.run_epoch(epoch=epoch, training=True)

            graphgen_metrics = {}
            uncond_graphgen_metrics = {}
            nn_metrics = {}

            if is_main_process():
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_obj={train_metrics['obj_loss']:.4f} "
                    f"train_edge={train_metrics['edge_loss']:.4f} "
                    f"train_rel={train_metrics['rel_loss']:.4f} "
                    f"train_edge_p={train_metrics['edge_precision']:.4f} "
                    f"train_edge_r={train_metrics['edge_recall']:.4f} "
                    f"train_edge_f1={train_metrics['edge_f1']:.4f} "
                    f"train_edge_ratio={train_metrics['pred_gt_edge_ratio']:.4f} "
                    f"train_rel_acc_tp={train_metrics['relation_accuracy_on_true_positive_edges']:.4f} "
                    f"train_node_all={train_metrics['node_acc_all']:.4f} "
                    f"train_node_corr={train_metrics['node_acc_corrupted']:.4f} "
                    # f"train_lay={train_metrics['layout_loss']:.4f} "
                    # f"train_lay_l1={train_metrics['layout_l1']:.4f} "
                    # f"train_lay_giou={train_metrics['layout_giou_loss']:.4f} "
                    # f"train_lay_giou_mean={train_metrics['layout_mean_giou']:.4f} "
                    # f"train_lay_overg={train_metrics['layout_overlapg']:.4f} "
                    # f"train_lay_spreadg={train_metrics['layout_spreadg']:.4f} "
                    # f"train_lay_cspread={train_metrics['layout_center_spread']:.4f} "
                    # f"train_lay_reg_loss={train_metrics['layout_reg_loss']:.4f} "
                    # f"train_rel_geo={train_metrics['rel_geometry_loss']:.4f} "
                    # f"train_lay_prior={train_metrics['layout_class_prior_loss']:.4f} "
                    # f"train_slay={train_metrics['sampled_layout_loss']:.4f} "
                    # f"train_slay_l1={train_metrics['sampled_layout_l1']:.4f} "
                    # f"train_slay_giou={train_metrics['sampled_layout_giou']:.4f} "
                    # f"train_reward={train_metrics['reward_total_eval']:.4f} "
                    # f"train_rel_geom={train_metrics['rel_geometry_loss']:.4f} "
                    # f"train_rel_geo_reg={train_metrics['relation_geometry_reg']:.4f} "
                    # f"train_graph_law_reg={train_metrics['graph_law_reg']:.4f} "
                    # f"train_graph_edge_density_reg={train_metrics['graph_edge_density_reg']:.4f} "
                    # f"train_graph_degree_reg={train_metrics['graph_degree_reg']:.4f} "
                    # f"train_graph_rel_marginal_reg={train_metrics['graph_rel_marginal_reg']:.4f} "
                    # f"train_pred_edge_density_avg={train_metrics['pred_edge_density_avg']:.4f} "
                    # f"train_gt_edge_density_avg={train_metrics['gt_edge_density_avg']:.4f} "
                    
                )

                self.wandb_logger.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/obj_loss": train_metrics["obj_loss"],
                    "train/edge_loss": train_metrics["edge_loss"],
                    "train/rel_loss": train_metrics["rel_loss"],
                    "train/gt_num_pos_edges": train_metrics["gt_num_pos_edges"],
                    "train/pred_num_pos_edges": train_metrics["pred_num_pos_edges"],
                    "train/pred_gt_edge_ratio": train_metrics["pred_gt_edge_ratio"],
                    "train/tp_edges": train_metrics["tp_edges"],
                    "train/fp_edges": train_metrics["fp_edges"],
                    "train/fn_edges": train_metrics["fn_edges"],
                    "train/edge_precision": train_metrics["edge_precision"],
                    "train/edge_recall": train_metrics["edge_recall"],
                    "train/edge_f1": train_metrics["edge_f1"],
                    "train/relation_accuracy_on_true_positive_edges": train_metrics["relation_accuracy_on_true_positive_edges"],
                    "train/node_acc_all": train_metrics["node_acc_all"],
                    "train/node_acc_corrupted": train_metrics["node_acc_corrupted"],
                    "train/edge_errors_total": train_metrics["fp_edges"] + train_metrics["fn_edges"],
                    "train/cond_node_loss": train_metrics.get("cond_node_loss", 0.0),
                    "train/cond_node_acc": train_metrics.get("cond_node_acc", 0.0),
                    "train/cond_query_prob": train_metrics.get("cond_query_prob", 0.0),
                    "train/refine_loss": train_metrics.get("refine_loss", 0.0),
                    "train/sampled_state_loss": train_metrics.get("sampled_state_loss", 0.0),
                    "train/reverse_step_loss": train_metrics.get("reverse_step_loss", 0.0),
                    "train/reverse_vocab_loss": train_metrics.get("reverse_vocab_loss", 0.0),
                    "train/reverse_obj_loss": train_metrics.get("reverse_obj_loss", 0.0),
                    "train/reverse_edge_loss": train_metrics.get("reverse_edge_loss", 0.0),
                    "train/reverse_rel_loss": train_metrics.get("reverse_rel_loss", 0.0),
                    "train/layout_loss": train_metrics.get('layout_loss', 0.0),
                    "train/layout_l1": train_metrics.get('layout_l1', 0.0),
                    "train/layout_giou_loss": train_metrics.get('layout_giou_loss', 0.0),
                    "train/layout_mean_giou": train_metrics.get('layout_mean_giou', 0.0),
                    "train/rel_geometry_loss": train_metrics.get('rel_geometry_loss', 0.0),
                    "train/layout_class_prior_loss": train_metrics.get('layout_class_prior_loss', 0.0),
                    "train/sampled_layout_loss": train_metrics.get('sampled_layout_loss', 0.0),
                    "train/sampled_layout_l1": train_metrics.get('sampled_layout_l1', 0.0),
                    "train/sampled_layout_giou": train_metrics.get('sampled_layout_giou', 0.0),
                    "train/reward_total_eval": train_metrics.get('reward_total_eval', 0.0),
                    "train/layout_overlapg": train_metrics.get('layout_overlapg', 0.0),
                    "train/layout_spreadg": train_metrics.get('layout_spreadg', 0.0),
                    "train/layout_center_spread": train_metrics.get('layout_center_spread', 0.0),
                    "train/layout_reg_loss": train_metrics.get('layout_reg_loss', 0.0),
                    "train/layout_overlapg": train_metrics.get('layout_overlapg', 0.0),
                    "train/relation_geometry_reg": train_metrics.get('relation_geometry_reg', 0.0),
                    "train/graph_law_reg": train_metrics.get('graph_law_reg', 0.0),
                    "train/graph_edge_density_reg": train_metrics.get('graph_edge_density_reg', 0.0),
                    "train/graph_degree_reg": train_metrics.get('graph_degree_reg', 0.0),
                    "train/graph_rel_marginal_reg": train_metrics.get('graph_rel_marginal_reg', 0.0),
                    "train/gt_edge_density_avg": train_metrics.get('gt_edge_density_avg', 0.0),

                }, step=self.global_step)

                lr = self.optimizer.param_groups[0]["lr"]
                self.wandb_logger.log({"train/lr": lr}, step=self.global_step)

                self.epoch_metrics_history.append({
                    "epoch": epoch,
                    "split": "train",
                    "loss": float(train_metrics["loss"]),
                    "obj_loss": float(train_metrics["obj_loss"]),
                    "edge_loss": float(train_metrics["edge_loss"]),
                    "rel_loss": float(train_metrics["rel_loss"]),
                    "gt_num_pos_edges": float(train_metrics["gt_num_pos_edges"]),
                    "pred_num_pos_edges": float(train_metrics["pred_num_pos_edges"]),
                    "pred_gt_edge_ratio": float(train_metrics["pred_gt_edge_ratio"]),
                    "tp_edges": float(train_metrics["tp_edges"]),
                    "fp_edges": float(train_metrics["fp_edges"]),
                    "fn_edges": float(train_metrics["fn_edges"]),
                    "edge_precision": float(train_metrics["edge_precision"]),
                    "edge_recall": float(train_metrics["edge_recall"]),
                    "edge_f1": float(train_metrics["edge_f1"]),
                    "relation_accuracy_on_true_positive_edges": float(train_metrics["relation_accuracy_on_true_positive_edges"]),
                    "node_acc_all": float(train_metrics["node_acc_all"]),
                    "node_acc_corrupted": float(train_metrics["node_acc_corrupted"]),
                    "cond_node_loss": float(train_metrics.get("cond_node_loss", 0.0)),
                    "cond_node_acc": float(train_metrics.get("cond_node_acc", 0.0)),
                    "refine_loss": float(train_metrics.get("refine_loss", 0.0)),
                    "sampled_state_loss": float(train_metrics.get("sampled_state_loss", 0.0)),
                    "reverse_step_loss": float(train_metrics.get("reverse_step_loss", 0.0)),
                    "reverse_vocab_loss": float(train_metrics.get("reverse_vocab_loss", 0.0)),
                    "layout_loss": float(train_metrics.get("layout_loss", 0.0)),
                    "layout_l1": float(train_metrics.get("layout_l1", 0.0)),
                    "layout_giou_loss": float(train_metrics.get("layout_giou_loss", 0.0)),
                    "layout_mean_giou": float(train_metrics.get("layout_mean_giou", 0.0)),
                    "rel_geometry_loss": float(train_metrics.get("rel_geometry_loss", 0.0)),
                    "layout_class_prior_loss": float(train_metrics.get("layout_class_prior_loss", 0.0)),
                    "sampled_layout_loss": float(train_metrics.get("sampled_layout_loss", 0.0)),
                    "sampled_layout_l1": float(train_metrics.get("sampled_layout_l1", 0.0)),
                    "sampled_layout_giou": float(train_metrics.get("sampled_layout_giou", 0.0)),
                    "reward_total_eval": float(train_metrics.get("reward_total_eval", 0.0)),
                    "layout_overlapg": float(train_metrics.get("layout_overlapg", 0.0)),
                    "layout_spreadg": float(train_metrics.get("layout_spreadg", 0.0)),
                    "layout_center_spread": float(train_metrics.get("layout_center_spread", 0.0)),
                    "layout_reg_loss": float(train_metrics.get("layout_reg_loss", 0.0)),
                    "relation_geometry_reg": float(train_metrics.get("relation_geometry_reg", 0.0)),
                    "graph_law_reg": float(train_metrics.get("graph_law_reg", 0.0)),
                    "graph_edge_density_reg": float(train_metrics.get("graph_edge_density_reg", 0.0)),
                    "graph_degree_reg": float(train_metrics.get("graph_degree_reg", 0.0)),
                    "graph_rel_marginal_reg": float(train_metrics.get("graph_rel_marginal_reg", 0.0)),
                    "pred_edge_density_avg": float(train_metrics.get("pred_edge_density_avg", 0.0)),
                    "gt_edge_density_avg": float(train_metrics.get("gt_edge_density_avg", 0.0)),
                })

            should_validate = (
                self.val_loader is not None
                and (
                    ((epoch + 1) % self.opt.val_every_epoch == 0)
                    or (epoch == self.opt.num_epochs - 1)
                )
            )

            if should_validate:
                # t0 = _t()
                val_metrics = self.run_epoch(epoch=epoch, training=False)
                # _log_time("run_epoch (val)", t0)


                if getattr(self.opt, "use_t_curriculum", False):
                    # stage_t = self.get_curriculum_t()
                    # val_node_corr = val_metrics.get("node_acc_corrupted", None)
                    # if val_node_corr is None:
                    #     val_node_corr = val_metrics.get("node_corr", 0.0)

                    # if stage_t == 10:
                    #     self.best_val_node_corr_t10 = max(self.best_val_node_corr_t10, val_node_corr)
                    # elif stage_t == 25:
                    #     self.best_val_node_corr_t25 = max(self.best_val_node_corr_t25, val_node_corr)
                    # elif stage_t == 49:
                    #     self.best_val_node_corr_t49 = max(self.best_val_node_corr_t49, val_node_corr)
                    val_node_corr = val_metrics["node_acc_corrupted"]  # or your val_node_corr key
                    curr_t = self.current_curriculum_t

                    self.best_val_node_corr_by_t[curr_t] = max(
                        self.best_val_node_corr_by_t[curr_t],
                        val_node_corr,
                    )

                    if is_main_process():
                        print(
                            f"[Curriculum] t={curr_t} "
                            f"val_node_corr={val_node_corr:.4f} "
                            f"best_val_corr={self.best_val_node_corr_by_t[curr_t]:.4f}"
                        )


                run_fullrev = (epoch % getattr(self.opt, "full_reverse_eval_every", 5) == 0)
                run_fullrev =  False

                if run_fullrev:
                    # t1 = _t()
                    fullrev_metrics = self.evaluate_full_reverse_reconstruction(
                        epoch=epoch,
                        max_batches=getattr(self.opt, "full_reverse_eval_max_batches", 20),
                    )
                    # _log_time("full_reverse_eval", t1)
                    # t2 = _t()
                    fullrev_composites = self.compute_fullrev_composite_metrics(fullrev_metrics)
                    fullrev_metrics.update(fullrev_composites)
                    # _log_time("fullrev_composites", t2)

                    if fullrev_metrics["fullrev_recon_composite"] > self.best_fullrev_recon_composite:
                        self.best_fullrev_composite = fullrev_metrics["fullrev_recon_composite"]
                        self.save_checkpoint(
                            epoch=epoch,
                            is_best=True,
                            best_name="best_fullrev_recon_composite.pt",
                            save_epoch_file=False,
                        )

                    if fullrev_metrics["fullrev_structure_composite"] > self.best_fullrev_structure_composite:
                        self.best_fullrev_structure_composite = fullrev_metrics["fullrev_structure_composite"]
                        self.save_checkpoint(
                            epoch=epoch,
                            is_best=True,
                            best_name="best_fullrev_structure_composite.pt",
                            save_epoch_file=False,
                        )
                    
                    if getattr(self.opt, "run_graph_generation_eval", False) and is_main_process():
                        # t3 = _t()
                        graphgen_metrics = self.evaluate_graph_generation_realism(
                            max_graphs=getattr(self.opt, "graph_generation_eval_max_graphs", 256),
                            include_motif_metrics=getattr(self.opt, "include_motif_metrics", False),
                            topk_triplets_k=getattr(self.opt, "graph_metrics_topk_triplets_k", 50),
                            out_degree_thresh=getattr(self.opt, "graph_metrics_hub_out_thresh", 3),
                            in_degree_thresh=getattr(self.opt, "graph_metrics_hub_in_thresh", 3),
                        )
                        # _log_time("graphgen_eval", t3)
                        t3b = _t()
                        graphgen_metrics.update(
                                self.compute_graphgen_composite(graphgen_metrics)
                            )
                        # _log_time("graphgen_composite", t3b)
                    
                    if getattr(self.opt, "run_unconditional_graph_generation_eval", False) and is_main_process():
                        # t4 = _t()
                        uncond_graphgen_metrics = self.evaluate_unconditional_graph_generation_realism(
                            max_graphs=getattr(self.opt, "unconditional_graph_generation_eval_max_graphs", 256),
                            include_motif_metrics=getattr(self.opt, "include_motif_metrics", False),
                            topk_triplets_k=getattr(self.opt, "graph_metrics_topk_triplets_k", 50),
                            out_degree_thresh=getattr(self.opt, "graph_metrics_hub_out_thresh", 3),
                            in_degree_thresh=getattr(self.opt, "graph_metrics_hub_in_thresh", 3),
                        )
                        # _log_time("uncond_graphgen_eval", t4)
                        # t4b = _t()
                        uncond_graphgen_metrics.update(
                                self.compute_unconditional_graphgen_composite(uncond_graphgen_metrics)
                            )
                        # _log_time("uncond_graphgen_composite", t4b)
                else:
                    fullrev_metrics = None
                
                if getattr(self.opt, "run_nn_eval", True):
                    # t5 = _t()
                    nn_metrics = self.evaluate_nearest_neighbor_analysis(
                        epoch=epoch,
                        max_generated_graphs=getattr(self.opt, "nn_eval_num_generated", 128),
                        max_reference_graphs=getattr(self.opt, "nn_eval_num_reference", 512),
                        top_k=getattr(self.opt, "nn_eval_top_k", 5),
                        save_dir="nn_analysis",
                    )
                    # _log_time("nn_eval", t5)
                # _log_time("TOTAL validation block", t0)
                
                           

                if is_main_process():
                    print(
                        f"[Epoch {epoch:03d}] "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"val_obj={val_metrics['obj_loss']:.4f} "
                        f"val_edge={val_metrics['edge_loss']:.4f} "
                        f"val_rel={val_metrics['rel_loss']:.4f} "
                        f"val_edge_p={val_metrics['edge_precision']:.4f} "
                        f"val_edge_r={val_metrics['edge_recall']:.4f} "
                        f"val_edge_f1={val_metrics['edge_f1']:.4f} "
                        f"val_edge_ratio={val_metrics['pred_gt_edge_ratio']:.4f} "
                        f"val_rel_acc_tp={val_metrics['relation_accuracy_on_true_positive_edges']:.4f} "
                        f"val_node_all={val_metrics['node_acc_all']:.4f} "
                        f"val_node_corr={val_metrics['node_acc_corrupted']:.4f} "
                        # f"val_lay={val_metrics['layout_loss']:.4f} "
                        # f"val_lay_l1={val_metrics['layout_l1']:.4f} "
                        # f"val_lay_giou={val_metrics['layout_giou_loss']:.4f} "
                        # f"val_lay_giou_mean={val_metrics['layout_mean_giou']:.4f} "
                        # f"val_lay_overg={val_metrics['layout_overlapg']:.4f} "
                        # f"val_lay_spreadg={val_metrics['layout_spreadg']:.4f} "
                        # f"val_lay_cspread={val_metrics['layout_center_spread']:.4f} "
                        # f"val_lay_reg_loss={val_metrics['layout_reg_loss']:.4f} "
                        # f"val_rel_geo={val_metrics['rel_geometry_loss']:.4f} "
                        # f"val_lay_prior={val_metrics['layout_class_prior_loss']:.4f} "
                        # f"val_slay={val_metrics['sampled_layout_loss']:.4f} "
                        # f"val_slay_l1={val_metrics['sampled_layout_l1']:.4f} "
                        # f"val_slay_giou={val_metrics['sampled_layout_giou']:.4f} "
                        # f"val_reward={val_metrics['reward_total_eval']:.4f} "
                        # f"val_rel_geo_reg={val_metrics['relation_geometry_reg']:.4f} "
                        # f"val_graph_law_reg={val_metrics['graph_law_reg']:.4f} "
                        # f"val_graph_edge_density_reg={val_metrics['graph_edge_density_reg']:.4f} "
                        # f"val_graph_degree_reg={val_metrics['graph_degree_reg']:.4f} "
                        # f"val_graph_rel_marginal_reg={val_metrics['graph_rel_marginal_reg']:.4f} "
                        # f"val_pred_edge_density_avg={val_metrics['pred_edge_density_avg']:.4f} "
                        # f"val_gt_edge_density_avg={val_metrics['gt_edge_density_avg']:.4f} "
                    )

                    if getattr(self.opt, "use_dedicated_reverse_branch", False):
                        print(
                            f"[Epoch {epoch:03d}] "
                            f"rev_vocab={val_metrics.get('reverse_vocab_loss', 0.0):.4f} "
                            f"rev_obj={val_metrics.get('reverse_obj_loss', 0.0):.4f} "
                            f"rev_edge={val_metrics.get('reverse_edge_loss', 0.0):.4f} "
                            f"rev_rel={val_metrics.get('reverse_rel_loss', 0.0):.4f}"
                        )
                    
                    if graphgen_metrics:
                        print(
                            f"[Epoch {epoch:03d}] "
                            f"graphgen_valid={graphgen_metrics['graphgen_valid_graph_rate']:.4f} "
                            f"graphgen_node_mmd={graphgen_metrics['graphgen_node_count_mmd']:.4f} "
                            f"graphgen_edge_mmd={graphgen_metrics['graphgen_edge_count_mmd']:.4f} "
                            f"graphgen_obj_tv={graphgen_metrics['graphgen_object_label_tv']:.4f} "
                            f"graphgen_rel_tv={graphgen_metrics['graphgen_relation_label_tv']:.4f} "
                            f"graphgen_triplet_tv={graphgen_metrics['graphgen_triplet_label_tv']:.4f} "
                            f"graphgen_unique={graphgen_metrics['graphgen_uniqueness_ratio']:.4f} "
                            f"graphgen_novel={graphgen_metrics['graphgen_novelty_ratio']:.4f} "
                            f"graphgen_tripdiv={graphgen_metrics['graphgen_triplet_diversity']:.4f} "
                            f"graphgen_topk_tripcov={graphgen_metrics['graphgen_topk_triplet_coverage']:.4f} "
                            f"graphgen_hub_tv={graphgen_metrics['graphgen_hub_signature_tv']:.4f} "
                            f"graphgen_attach_tv={graphgen_metrics['graphgen_attachment_motif_tv']:.4f} "
                            f"graphgen_degpair_tv={graphgen_metrics['graphgen_degree_pair_tv']:.4f} "
                        )
                    
                    if uncond_graphgen_metrics:
                        print(
                            f"[Epoch {epoch:03d}] "
                            f"uncond_valid={uncond_graphgen_metrics['uncond_graphgen_valid_graph_rate']:.4f} "
                            f"uncond_node_mmd={uncond_graphgen_metrics['uncond_graphgen_node_count_mmd']:.4f} "
                            f"uncond_edge_mmd={uncond_graphgen_metrics['uncond_graphgen_edge_count_mmd']:.4f} "
                            f"uncond_obj_tv={uncond_graphgen_metrics['uncond_graphgen_object_label_tv']:.4f} "
                            f"uncond_rel_tv={uncond_graphgen_metrics['uncond_graphgen_relation_label_tv']:.4f} "
                            f"uncond_triplet_tv={uncond_graphgen_metrics['uncond_graphgen_triplet_label_tv']:.4f} "
                            f"uncond_unique={uncond_graphgen_metrics['uncond_graphgen_uniqueness_ratio']:.4f} "
                            f"uncond_novel={uncond_graphgen_metrics['uncond_graphgen_novelty_ratio']:.4f} "
                            f"uncond_tripdiv={uncond_graphgen_metrics['uncond_graphgen_triplet_diversity']:.4f} "
                            f"uncond_topk_tripcov={uncond_graphgen_metrics['uncond_graphgen_topk_triplet_coverage']:.4f} "
                            f"uncond_hub_tv={uncond_graphgen_metrics['uncond_graphgen_hub_signature_tv']:.4f} "
                            f"uncond_attach_tv={uncond_graphgen_metrics['uncond_graphgen_attachment_motif_tv']:.4f} "
                            f"uncond_degpair_tv={uncond_graphgen_metrics['uncond_graphgen_degree_pair_tv']:.4f} "
                        )

                    self.wandb_logger.log({
                        "epoch": epoch,
                        "val/loss": val_metrics["loss"],
                        "val/obj_loss": val_metrics["obj_loss"],
                        "val/edge_loss": val_metrics["edge_loss"],
                        "val/rel_loss": val_metrics["rel_loss"],
                        "val/gt_num_pos_edges": val_metrics["gt_num_pos_edges"],
                        "val/pred_num_pos_edges": val_metrics["pred_num_pos_edges"],
                        "val/pred_gt_edge_ratio": val_metrics["pred_gt_edge_ratio"],
                        "val/tp_edges": val_metrics["tp_edges"],
                        "val/fp_edges": val_metrics["fp_edges"],
                        "val/fn_edges": val_metrics["fn_edges"],
                        "val/edge_precision": val_metrics["edge_precision"],
                        "val/edge_recall": val_metrics["edge_recall"],
                        "val/edge_f1": val_metrics["edge_f1"],
                        "val/relation_accuracy_on_true_positive_edges": val_metrics["relation_accuracy_on_true_positive_edges"],
                        "val/node_acc_all": val_metrics["node_acc_all"],
                        "val/node_acc_corrupted": val_metrics["node_acc_corrupted"],
                        "val/edge_errors_total": val_metrics["fp_edges"] + val_metrics["fn_edges"],
                        "val/cond_node_loss": val_metrics.get("cond_node_loss", 0.0),
                        "val/cond_node_acc": val_metrics.get("cond_node_acc", 0.0),
                        "val/cond_query_prob": val_metrics.get("cond_query_prob", 0.0),
                        "val/refine_loss": val_metrics.get("refine_loss", 0.0),
                        "val/sampled_state_loss": val_metrics.get("sampled_state_loss", 0.0),
                        "val/reverse_step_loss": val_metrics.get("reverse_step_loss", 0.0),
                        "val/reverse_vocab_loss": val_metrics.get("reverse_vocab_loss", 0.0),
                        "val/reverse_obj_loss": val_metrics.get("reverse_obj_loss", 0.0),
                        "val/reverse_edge_loss": val_metrics.get("reverse_edge_loss", 0.0),
                        "val/reverse_rel_loss": val_metrics.get("reverse_rel_loss", 0.0),
                        "val/layout_loss": val_metrics.get('layout_loss', 0.0),
                        "val/layout_l1": val_metrics.get('layout_l1', 0.0),
                        "val/layout_giou_loss": val_metrics.get('layout_giou_loss', 0.0),
                        "val/layout_mean_giou": val_metrics.get('layout_mean_giou', 0.0),
                        "val/rel_geometry_loss": val_metrics.get('rel_geometry_loss', 0.0),
                        "val/layout_class_prior_loss": val_metrics.get('layout_class_prior_loss', 0.0),
                        "val/sampled_layout_loss": val_metrics.get('sampled_layout_loss', 0.0),
                        "val/sampled_layout_l1": val_metrics.get('sampled_layout_l1', 0.0),
                        "val/sampled_layout_giou": val_metrics.get('sampled_layout_giou', 0.0),
                        "val/reward_total_eval": val_metrics.get('reward_total_eval', 0.0),
                        "val/layout_overlapg": val_metrics.get('layout_overlapg', 0.0),
                        "val/layout_spreadg": val_metrics.get('layout_spreadg', 0.0),
                        "val/layout_center_spread": val_metrics.get('layout_center_spread', 0.0),
                        "val/layout_reg_loss": val_metrics.get('layout_reg_loss', 0.0),
                        "val/relation_geometry_reg": val_metrics.get('relation_geometry_reg', 0.0),
                        "val/graph_law_reg": val_metrics.get('graph_law_reg', 0.0),
                        "val/graph_edge_density_reg": val_metrics.get('graph_edge_density_reg', 0.0),
                        "val/graph_degree_reg": val_metrics.get('graph_degree_reg', 0.0),
                        "val/graph_rel_marginal_reg": val_metrics.get('graph_rel_marginal_reg', 0.0),
                        "val/gt_edge_density_avg": val_metrics.get('gt_edge_density_avg', 0.0),

                    }, step=self.global_step)

                    if graphgen_metrics:
                            self.wandb_logger.log(
                                {f"val/{k}": v for k, v in graphgen_metrics.items()}
                            , step=self.global_step)
                    
                    if uncond_graphgen_metrics:
                            self.wandb_logger.log(
                                {f"val/{k}": v for k, v in uncond_graphgen_metrics.items()}
                                , step=self.global_step)
                    
                    if nn_metrics:
                        self.wandb_logger.log({
                            "nn_mean_distance": nn_metrics["nn_mean_distance"],
                            "nn_median_distance": nn_metrics["nn_median_distance"],
                            "nn_mean_obj_distance": nn_metrics["nn_mean_obj_distance"],
                            "nn_mean_rel_distance": nn_metrics["nn_mean_rel_distance"],
                            "nn_mean_triplet_distance": nn_metrics["nn_mean_triplet_distance"],
                            "nn_mean_node_count_distance": nn_metrics["nn_mean_node_count_distance"],
                            "nn_mean_edge_count_distance": nn_metrics["nn_mean_edge_count_distance"],
                        }, step=self.global_step)

                    self.epoch_metrics_history.append({
                        "epoch": epoch,
                        "split": "val",
                        "loss": float(val_metrics["loss"]),
                        "obj_loss": float(val_metrics["obj_loss"]),
                        "edge_loss": float(val_metrics["edge_loss"]),
                        "rel_loss": float(val_metrics["rel_loss"]),
                        "gt_num_pos_edges": float(val_metrics["gt_num_pos_edges"]),
                        "pred_num_pos_edges": float(val_metrics["pred_num_pos_edges"]),
                        "pred_gt_edge_ratio": float(val_metrics["pred_gt_edge_ratio"]),
                        "tp_edges": float(val_metrics["tp_edges"]),
                        "fp_edges": float(val_metrics["fp_edges"]),
                        "fn_edges": float(val_metrics["fn_edges"]),
                        "edge_precision": float(val_metrics["edge_precision"]),
                        "edge_recall": float(val_metrics["edge_recall"]),
                        "edge_f1": float(val_metrics["edge_f1"]),
                        "relation_accuracy_on_true_positive_edges": float(val_metrics["relation_accuracy_on_true_positive_edges"]),
                        "node_acc_all": float(val_metrics["node_acc_all"]),
                        "node_acc_corrupted": float(val_metrics["node_acc_corrupted"]),
                        "cond_node_loss": float(val_metrics.get("cond_node_loss", 0.0)),
                        "cond_node_acc": float(val_metrics.get("cond_node_acc", 0.0)),
                        "refine_loss": float(val_metrics.get("refine_loss", 0.0)),
                        "sampled_state_loss": float(val_metrics.get("sampled_state_loss", 0.0)),
                        "reverse_step_loss": float(val_metrics.get("reverse_step_loss", 0.0)),
                        "reverse_vocab_loss": float(val_metrics.get("reverse_vocab_loss", 0.0)),
                        "layout_loss": float(val_metrics.get("layout_loss", 0.0)),
                        "layout_l1": float(val_metrics.get("layout_l1", 0.0)),
                        "layout_giou_loss": float(val_metrics.get("layout_giou_loss", 0.0)),
                        "layout_mean_giou": float(val_metrics.get("layout_mean_giou", 0.0)),
                        "rel_geometry_loss": float(val_metrics.get("rel_geometry_loss", 0.0)),
                        "layout_class_prior_loss": float(val_metrics.get("layout_class_prior_loss", 0.0)),
                        "sampled_layout_loss": float(val_metrics.get("sampled_layout_loss", 0.0)),
                        "sampled_layout_l1": float(val_metrics.get("sampled_layout_l1", 0.0)),
                        "sampled_layout_giou": float(val_metrics.get("sampled_layout_giou", 0.0)),
                        "reward_total_eval": float(val_metrics.get("reward_total_eval", 0.0)),
                        "layout_overlapg": float(val_metrics.get("layout_overlapg", 0.0)),
                        "layout_spreadg": float(val_metrics.get("layout_spreadg", 0.0)),
                        "layout_center_spread": float(val_metrics.get("layout_center_spread", 0.0)),
                        "layout_reg_loss": float(val_metrics.get("layout_reg_loss", 0.0)),
                        "relation_geometry_reg": float(val_metrics.get("relation_geometry_reg", 0.0)),
                        "graph_law_reg": float(train_metrics.get("graph_law_reg", 0.0)),
                        "graph_edge_density_reg": float(val_metrics.get("graph_edge_density_reg", 0.0)),
                        "graph_degree_reg": float(val_metrics.get("graph_degree_reg", 0.0)),
                        "graph_rel_marginal_reg": float(val_metrics.get("graph_rel_marginal_reg", 0.0)),
                        "pred_edge_density_avg": float(val_metrics.get("pred_edge_density_avg", 0.0)),
                        "gt_edge_density_avg": float(val_metrics.get("gt_edge_density_avg", 0.0)),
                    })

                    if epoch % self.opt.wandb_num_epochs_val == 0:
                        self.log_validation_graph_examples(epoch)
                        self.log_validation_graph_table(epoch)

                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.save_checkpoint(
                            epoch,
                            is_best=True,
                            best_name="best_total.pt",
                            save_epoch_file=False,
                        )

                    if val_metrics["rel_loss"] < self.best_val_rel_loss:
                        self.best_val_rel_loss = val_metrics["rel_loss"]
                        self.save_checkpoint(
                            epoch,
                            is_best=True,
                            best_name="best_rel.pt",
                            save_epoch_file=False,
                        )
                    
                    if val_metrics["obj_loss"] < self.best_val_obj_loss:
                        self.best_val_obj_loss = val_metrics["obj_loss"]
                        self.save_checkpoint(
                            epoch,
                            is_best=True,
                            best_name="best_obj.pt",
                            save_epoch_file=False,
                        )

                    if fullrev_metrics is not None:
                        self.wandb_logger.log({
                            "val/fullrev_node_acc_all": fullrev_metrics["fullrev_node_acc_all"],
                            "val/fullrev_node_acc_corrupted": fullrev_metrics["fullrev_node_acc_corrupted"],
                            "val/fullrev_edge_f1": fullrev_metrics["fullrev_edge_f1"],
                            "val/fullrev_edge_precision": fullrev_metrics["fullrev_edge_precision"],
                            "val/fullrev_edge_recall": fullrev_metrics["fullrev_edge_recall"],
                            "val/fullrev_rel_acc_tp": fullrev_metrics["fullrev_rel_acc_tp"],
                            "val/fullrev_recon_composite": fullrev_metrics.get("fullrev_recon_composite", 0.0),
                            "val/fullrev_structure_composite": fullrev_metrics.get("fullrev_structure_composite", 0.0),
                            "epoch": epoch,
                        }, step=self.global_step)

                        print(
                            f"[Epoch {epoch:03d}] "
                            f"fullrev_node_acc_all={fullrev_metrics['fullrev_node_acc_all']:.4f} "
                            f"fullrev_node_acc_corrupted={fullrev_metrics['fullrev_node_acc_corrupted']:.4f} "
                            f"fullrev_edge_f1={fullrev_metrics['fullrev_edge_f1']:.4f} "
                            f"fullrev_edge_precision={fullrev_metrics['fullrev_edge_precision']:.4f} "
                            f"fullrev_edge_recall={fullrev_metrics['fullrev_edge_recall']:.4f} "
                            f"fullrev_rel_acc_tp={fullrev_metrics['fullrev_rel_acc_tp']:.4f} "
                            f"fullrev_recon_comp={fullrev_metrics['fullrev_recon_composite']:.4f} "
                            f"fullrev_struct_comp={fullrev_metrics['fullrev_structure_composite']:.4f} "
                        )

                        self.epoch_metrics_history.append({
                            "epoch": epoch,
                            "split": "val",
                            "fullrev_node_acc_all": fullrev_metrics["fullrev_node_acc_all"],
                            "fullrev_node_acc_corrupted": fullrev_metrics["fullrev_node_acc_corrupted"],
                            "fullrev_edge_f1": fullrev_metrics["fullrev_edge_f1"],
                            "fullrev_edge_precision": fullrev_metrics["fullrev_edge_precision"],
                            "fullrev_edge_recall": fullrev_metrics["fullrev_edge_recall"],
                            "fullrev_rel_acc_tp": fullrev_metrics["fullrev_rel_acc_tp"],
                            "fullrev_recon_composite": fullrev_metrics.get("fullrev_recon_composite", 0.0),
                            "fullrev_structure_composite": fullrev_metrics.get("fullrev_structure_composite", 0.0),
                        })

                        if epoch % self.opt.wandb_num_epochs_fullrev == 0:
                            self.log_full_reverse_graph_examples(epoch)
                            self.log_full_reverse_graph_table(epoch)
                self.update_curriculum_t(epoch)

            should_save_regular = (
                ((epoch + 1) % self.opt.save_every_epoch == 0)
                or (epoch == self.opt.num_epochs - 1)
            )


            if is_main_process() and (((epoch + 1) % self.opt.val_every_epoch == 0) or should_save_regular):
                self.log_epoch_metrics_table()

            if is_main_process() and should_save_regular:
                self.save_checkpoint(epoch, is_best=False, save_epoch_file=True)

            self.global_step += 1
            barrier()

        if is_main_process():
            self.wandb_logger.finish()
        cleanup_distributed()

    def run_epoch(self, epoch: int, training: bool):
        if training:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_obj_loss_sum = 0.0
        total_obj_count = 0.0

        total_edge_loss_sum = 0.0
        total_edge_count = 0.0

        total_rel_loss_sum = 0.0
        total_rel_count = 0.0

        total_layout_loss_sum = 0.0
        total_layout_count = 0.0

        total_layout_l1_sum = 0.0
        total_layout_l1_count = 0.0

        total_layout_giou_loss_sum = 0.0
        total_layout_mean_giou_sum = 0.0

        total_sampled_layout_loss_sum = 0.0
        total_sampled_layout_loss_count = 0.0
        total_sampled_layout_l1_sum = 0.0
        total_sampled_layout_giou_val_sum = 0.0

        total_layout_overlap_reg_sum = 0.0
        total_layout_spread_reg_sum = 0.0
        total_layout_center_spread_sum = 0.0
        total_layout_reg_loss_sum = 0.0
        total_layout_reg_count = 0.0

        total_rel_geometry_loss_sum = 0.0
        total_rel_geometry_loss_count = 0.0
        total_rel_geometry_loss_sum = 0.0
        total_rel_geometry_loss_count = 0.0

        total_rel_geom_left_count = 0.0
        total_rel_geom_right_count = 0.0
        total_rel_geom_above_count = 0.0
        total_rel_geom_below_count = 0.0
        total_rel_geom_inside_count = 0.0

        total_relation_geometry_reg_sum = 0.0
        total_relation_geometry_reg_count = 0.0

        total_rel_geom_reg_behind_count = 0.0
        total_rel_geom_reg_front_count = 0.0
        total_rel_geom_reg_above_count = 0.0
        total_rel_geom_reg_below_count = 0.0
        total_rel_geom_reg_inside_count = 0.0
        total_rel_geom_reg_on_count  = 0.0

        total_layout_class_prior_loss_sum = 0.0
        total_layout_class_prior_loss_count = 0.0

        total_loss_sum = 0.0
        total_loss_count = 0.0

        total_tp_edges = 0.0
        total_fp_edges = 0.0
        total_fn_edges = 0.0
        total_pred_pos_edges = 0.0
        total_gt_pos_edges = 0.0
        total_rel_acc_sum = 0.0
        total_rel_acc_count = 0.0

        total_node_correct_all = 0.0
        total_node_count_all = 0.0

        total_node_correct_corrupted = 0.0
        total_node_count_corrupted = 0.0

        total_cond_node_loss_sum = 0.0
        total_cond_node_count = 0.0
        total_cond_node_correct = 0.0

        total_refine_loss_sum = 0.0
        total_refine_loss_count = 0.0

        total_sampled_state_loss_sum = 0.0
        total_sampled_state_loss_count = 0.0

        total_reverse_step_loss_sum = 0.0
        total_reverse_step_loss_count = 0.0

        total_reverse_vocab_loss_sum = 0.0
        total_reverse_vocab_loss_count = 0.0

        total_reverse_obj_loss_sum = 0.0
        total_reverse_edge_loss_sum = 0.0
        total_reverse_rel_loss_sum = 0.0
        total_reverse_branch_count = 0.0

        total_reward_sum = 0.0
        total_reward_count = 0.0

        total_graph_law_reg_sum = 0.0
        total_graph_edge_density_reg_sum = 0.0
        total_graph_degree_reg_sum = 0.0
        total_graph_rel_marginal_reg_sum = 0.0
        total_pred_edge_density_sum = 0.0
        total_gt_edge_density_sum = 0.0
        total_graph_law_count = 0.0

        iterator = loader
        if is_main_process():
            desc = f"Train {epoch}" if training else f"Eval {epoch}"
            iterator = tqdm(loader, desc=desc, leave=False)

        context = torch.enable_grad() if training else torch.no_grad()
        tt = 0
        tt1 = 0
        with context:
            for batch in iterator:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                
                

                if getattr(self.opt, "object_fixed_t_sanity", False):
                    batch_t = self.obj_gen.get_training_batch(batch,
                        force_t=getattr(self.opt, "object_fixed_t", None),
                    )
                else:
                    batch_t = self.obj_gen.get_training_batch(batch)
                
                # # --------------------------------------------------
                # # Phase 5C-full-3: 
                # # --------------------------------------------------
                if getattr(self.opt, "object_fixed_t_sanity", False):
                    batch_pair = self.obj_gen.get_training_batch_pair(batch,
                        force_t=getattr(self.opt, "object_fixed_t", None),
                    )
                else:
                    batch_pair = self.obj_gen.get_training_batch_pair(batch)

                # only curriculum during training
                # if getattr(self.opt, "use_t_curriculum", False):
                #     force_t = self.current_curriculum_t
                #     batch_t = self.obj_gen.get_training_batch(batch,force_t)
                #     batch_pair = self.obj_gen.get_training_batch_pair(batch,force_t)
                # else:
                #     batch_t = self.obj_gen.get_training_batch(batch)
                #     batch_pair = self.obj_gen.get_training_batch_pair(batch)
  

                batch_t = batch_pair["batch_t"]
                batch_prev = batch_pair["batch_prev"]

                

                # --------------------------------------------------
                # Phase 7A.3: clean-graph layout supervision inputs
                # --------------------------------------------------
                if "boxes" in batch:
                    batch_t["boxes"] = batch["boxes"].to(self.device)  # [B,N,4]
                    batch_t["box_valid_mask"] = batch["node_mask"].bool().to(self.device)

                    if batch_prev is not None:
                        batch_prev["boxes"] = batch["boxes"].to(self.device)
                        batch_prev["box_valid_mask"] = batch["node_mask"].bool().to(self.device)
                # --------------------------------------------------
                # Pass 1: normal forward
                # --------------------------------------------------
                model_out_1 = self.model(
                    obj_t=batch_t["obj_t"],
                    edge_t=batch_t["edge_t"],
                    rel_pos_t=batch_t["rel_pos_t"],
                    t=batch_t["t"],
                    node_mask=batch_t["node_mask"],
                    edge_mask=batch_t["edge_mask"],
                    edge_input_override=batch_t["edge_0"],
                    rel_input_override=batch_t["rel_pos_0"],
                )

                if is_main_process() and tt == 0 and epoch % 5 == 0:
                    with torch.no_grad():
                        cm = batch_t["obj_corrupt_mask"] & batch_t["node_mask"]

                        if cm.any():
                            pred = model_out_1["obj_logits"].argmax(dim=-1)

                            copy_acc = (batch_t["obj_t"][cm] == batch_t["obj_0"][cm]).float().mean()
                            pred_acc = (pred[cm] == batch_t["obj_0"][cm]).float().mean()

                            rand_frac = batch_t.get("obj_rand_mask", torch.zeros_like(cm))[cm].float().mean()
                            mask_frac = batch_t.get("obj_mask_token_mask", torch.zeros_like(cm))[cm].float().mean()

                            print(
                                "[OBJ BASELINE DEBUG]",
                                "t=", batch_t["t"].unique().detach().cpu().tolist(),
                                "num_corr=", int(cm.sum().item()),
                                "copy_acc=", float(copy_acc.item()),
                                "pred_acc=", float(pred_acc.item()),
                                "mask_frac=", float(mask_frac.item()),
                                "rand_frac=", float(rand_frac.item()),
                                "target_unique=", int(batch_t["obj_0"][cm].unique().numel()),
                                "input_unique=", int(batch_t["obj_t"][cm].unique().numel()),
                                "pred_unique=", int(pred[cm].unique().numel()),
                            )

                            print(
                                "[OBJ BASELINE SAMPLE]",
                                "target=", batch_t["obj_0"][cm].detach().cpu().tolist()[:25],
                                "input=", batch_t["obj_t"][cm].detach().cpu().tolist()[:25],
                                "pred=", pred[cm].detach().cpu().tolist()[:25],
                            )
                        else:
                            print(
                                "[OBJ BASELINE DEBUG]",
                                "t=", batch_t["t"].unique().detach().cpu().tolist(),
                                "num_corr=0",
                            )
                    with torch.no_grad():
                        pred = model_out_1["obj_logits"].argmax(dim=-1)
                        cm = batch_t["obj_corrupt_mask"] & batch_t["node_mask"]

                        topk = self.topk_obj_classes.to(batch_t["obj_0"].device)
                        topk_mask = torch.isin(batch_t["obj_0"], topk)
                        m = cm & topk_mask

                        if m.any():
                            topk_corr = (pred[m] == batch_t["obj_0"][m]).float().mean()
                        else:
                            topk_corr = torch.tensor(0.0, device=batch_t["obj_0"].device)
                        
                        print(
                                "[OBJ TOPK DEBUG]",
                                "t=", batch_t["t"].unique().detach().cpu().tolist(),
                                "num_corr=",topk_corr.item(),
                            )
                        tt = 1
                    
                    with torch.no_grad():
                        bt = batch_pair["batch_t"]
                        cm = bt["obj_corrupt_mask"] & bt["node_mask"]

                        logits = model_out_1["obj_logits"]  # adjust key if needed
                        pred = logits.argmax(dim=-1)

                        if cm.any():
                            tgt = bt["obj_0"][cm]
                            inp = bt["obj_t"][cm]
                            prd = pred[cm]

                            print("[OBJ PRED DEBUG]")
                            print("num_corr=", cm.sum().item())
                            print("target=", tgt[:20].detach().cpu().tolist())
                            print("input=", inp[:20].detach().cpu().tolist())
                            print("pred=", prd[:20].detach().cpu().tolist())
                            print("acc=", (prd == tgt).float().mean().item())
                            print("[LOGIT DEBUG]", logits[cm].std().item(), logits[cm].mean().item())
                        tt1 = 1

                # print(batch_t["boxes"].shape)
                # print(batch_t["boxes"].min().item(), batch_t["boxes"].max().item())
                # print(batch_t["boxes"][batch_t["box_valid_mask"]][:5])
                # print(model_out_1["layout_box_pred"].shape)
                # print(model_out_1["layout_box_pred"].min().item(), model_out_1["layout_box_pred"].max().item())

                loss_dict_1 = compute_structured_sg_loss(
                    model_out=model_out_1,
                    batch_t=batch_t,
                    no_rel_token_id=self.opt.no_rel_token_id,
                    lambda_obj=self.opt.lambda_obj,
                    lambda_edge=self.opt.lambda_edge,
                    lambda_rel=self.opt.lambda_rel,
                    lambda_layout=getattr(self.opt, "lambda_layout", 1.0),
                    edge_exist_thres=self.opt.edge_exist_thres,
                    edge_pos_weight=self.opt.edge_pos_weight,
                    obj_class_weights=self.obj_class_weights
                    if getattr(self.opt, "use_object_class_weights", False) and training
                    else None,
                    node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
                    use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
                    object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                    object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                    pred_obj_override=None,
                    pred_rel_full_override=None,
                    use_layout_supervision=getattr(self.opt, "use_layout_supervision", False),
                    layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
                    use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
                    lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),
                    lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
                    relation_vocab=self.train_dataset.relation_vocab,
                    rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
                    use_relation_geometry_loss=getattr(self.opt, "use_relation_geometry_loss", False),
                    use_layout_class_priors=getattr(self.opt, "use_layout_class_priors", False),
                    lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0),
                    layout_prior_mean=self.layout_prior_mean,
                    layout_prior_var=self.layout_prior_var,
                    layout_prior_valid=self.layout_prior_valid,
                    layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),
                    use_layout_regularization=getattr(self.opt, "use_layout_regularization", False),
                    layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.10),
                    layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.05),
                    layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),
                    use_relation_geometry_reg=getattr(self.opt, "use_relation_geometry_reg", False),
                    lambda_relation_geometry_reg=getattr(self.opt, "lambda_relation_geometry_reg", 0.0),
                    relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.03),
                    use_graph_law_reg=getattr(self.opt, "use_graph_law_reg", False),
                    lambda_graph_law_reg=getattr(self.opt, "lambda_graph_law_reg", 0.0),
                    graph_law_edge_weight=getattr(self.opt, "graph_law_edge_weight", 1.0),
                    graph_law_degree_weight=getattr(self.opt, "graph_law_degree_weight", 0.5),
                    graph_law_rel_weight=getattr(self.opt, "graph_law_rel_weight", 0.5),
                    graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
                    object_only_sanity=getattr(self.opt, "object_only_sanity", False),
                    topk_obj_loss_only=getattr(self.opt, "topk_obj_loss_only", False),
                    topk_obj_classes=self.topk_obj_classes,
                )

                loss = loss_dict_1["loss"]

                # --------------------------------------------------
                # Phase 8A.1: reward tilting
                # --------------------------------------------------
                if not training and self.opt.use_reward_tilting:
                    pred_obj_eval = loss_dict_1["pred_obj_full"]
                    pred_rel_full_eval = loss_dict_1["pred_rel_full"]
                    pred_rel_pos_eval = torch.where(
                        pred_rel_full_eval == self.opt.no_rel_token_id,
                        torch.zeros_like(pred_rel_full_eval),
                        pred_rel_full_eval - 1,
                    )
                    pred_edge_eval = (pred_rel_full_eval != self.opt.no_rel_token_id).long()

                    reward_terms_eval = self.compute_reward_terms_for_state(
                        obj_t=pred_obj_eval,
                        edge_t=pred_edge_eval,
                        rel_pos_t=pred_rel_pos_eval,
                        node_mask=batch_t["node_mask"],
                        edge_mask=batch_t["edge_mask"],
                        layout_box_pred=loss_dict_1.get("layout_box_pred", None),
                        box_valid_mask=batch_t.get("box_valid_mask", None),
                    )

                # --------------------------------------------------
                # Phase 5C-full-2: reverse-step-aligned training
                # --------------------------------------------------
                reverse_step_loss_val = None

                if getattr(self.opt, "use_reverse_step_training", False):
                    batch_rev, valid_prev_mask = self.build_reverse_step_target_batch(batch_t, batch_prev)

                    if batch_rev is not None:
                        model_out_rev = self.slice_model_out(model_out_1, valid_prev_mask)

                        loss_dict_rev = compute_structured_sg_loss(
                            model_out=model_out_rev,
                            batch_t=batch_rev,
                            no_rel_token_id=self.opt.no_rel_token_id,
                            lambda_obj=self.opt.lambda_obj,
                            lambda_edge=self.opt.lambda_edge,
                            lambda_rel=self.opt.lambda_rel,
                            lambda_layout=getattr(self.opt, "lambda_layout", 1.0),
                            edge_exist_thres=self.opt.edge_exist_thres,
                            edge_pos_weight=self.opt.edge_pos_weight,
                            obj_class_weights=self.obj_class_weights
                            if getattr(self.opt, "use_object_class_weights", False) and training
                            else None,
                            rel_class_weights=self.reverse_rel_class_weights
                            if getattr(self.opt, "use_reverse_rel_class_weights", False)
                            else None,
                            node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
                            use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
                            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                            pred_obj_override=None,
                            pred_rel_full_override=None,
                            use_layout_supervision=getattr(self.opt, "use_layout_supervision", False),
                            layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
                            use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
                            lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),
                            lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
                            relation_vocab=self.train_dataset.relation_vocab,
                            rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
                            use_relation_geometry_loss=getattr(self.opt, "use_relation_geometry_loss", False),
                            lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0),
                            layout_prior_mean=self.layout_prior_mean,
                            layout_prior_var=self.layout_prior_var,
                            layout_prior_valid=self.layout_prior_valid,
                            layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),
                            use_layout_regularization=getattr(self.opt, "use_layout_regularization", False),
                            layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.10),
                            layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.05),
                            layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),
                            use_relation_geometry_reg=getattr(self.opt, "use_relation_geometry_reg", False),
                            lambda_relation_geometry_reg=getattr(self.opt, "lambda_relation_geometry_reg", 0.0),
                            relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.03),
                            use_graph_law_reg=getattr(self.opt, "use_graph_law_reg", False),
                            lambda_graph_law_reg=getattr(self.opt, "lambda_graph_law_reg", 0.0),
                            graph_law_edge_weight=getattr(self.opt, "graph_law_edge_weight", 1.0),
                            graph_law_degree_weight=getattr(self.opt, "graph_law_degree_weight", 0.5),
                            graph_law_rel_weight=getattr(self.opt, "graph_law_rel_weight", 0.5),
                            graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
                            object_only_sanity=getattr(self.opt, "object_only_sanity", False),
                        )

                        reverse_step_loss_val = loss_dict_rev["loss"]
                        loss = loss + getattr(self.opt, "lambda_reverse_step", 0.5) * reverse_step_loss_val
                
                # --------------------------------------------------
                # Phase 5D.1: dedicated reverse branch
                # --------------------------------------------------
                reverse_vocab_loss_val = None

                if getattr(self.opt, "use_dedicated_reverse_branch", False) and getattr(self.opt, "use_reverse_vocab_step_training", False):
                    reverse_out = self.compute_reverse_branch_loss(
                        model_out=model_out_1,
                        batch_t=batch_t,
                        batch_prev=batch_prev,
                        training=training,
                    )

                    if reverse_out is not None:
                        loss_dict_rev = reverse_out["loss_dict_rev"]

                        reverse_vocab_loss_val = loss_dict_rev["loss"]
                        loss = loss + getattr(self.opt, "lambda_reverse_vocab_step", 0.5) * reverse_vocab_loss_val

                        total_reverse_obj_loss_sum += float(loss_dict_rev["obj_loss"].item())
                        total_reverse_edge_loss_sum += float(loss_dict_rev["edge_loss"].item())
                        total_reverse_rel_loss_sum += float(loss_dict_rev["rel_loss"].item())
                        total_reverse_branch_count += 1.0


                # --------------------------------------------------
                # Phase 4B conditional node objective
                # --------------------------------------------------
                cond_node_loss_val = None
                cond_node_acc_val = None
                cond_node_count_val = None

                if getattr(self.opt, "use_conditional_node_objective", False):
                    query_mask = self.sample_conditional_query_mask(batch_t=batch_t, epoch=epoch)

                    obj_cond_t, edge_cond_t, rel_pos_cond_t = self.build_conditional_node_inputs(
                        batch_t=batch_t,
                        query_mask=query_mask,
                    )

                    cond_model_out = self.model(
                        obj_t=obj_cond_t,
                        edge_t=edge_cond_t,
                        rel_pos_t=rel_pos_cond_t,
                        t=batch_t["t"],
                        node_mask=batch_t["node_mask"],
                        edge_mask=batch_t["edge_mask"],
                    )

                    cond_dict = compute_conditional_node_loss(
                        obj_logits=cond_model_out["obj_logits"],
                        obj_targets=batch_t["obj_0"],
                        query_mask=query_mask,
                        node_mask=batch_t["node_mask"],
                        obj_class_weights=self.obj_class_weights
                        if getattr(self.opt, "use_object_class_weights", False) and training
                        else None,
                        use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
                        object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                        object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                    )

                    loss = loss + getattr(self.opt, "lambda_cond_node", 1.0) * cond_dict["cond_node_loss"]

                    cond_node_loss_val = cond_dict["cond_node_loss"]
                    cond_node_acc_val = cond_dict["cond_node_acc"]
                    cond_node_count_val = cond_dict["cond_node_count"]

                # --------------------------------------------------
                # Phase 5C-lite: sampled-state auxiliary training
                # --------------------------------------------------
                sampled_state_loss_val = None

                if getattr(self.opt, "use_sampled_state_training", False):
                    sampled_prev = sample_prev_state_from_current_batch(
                        model=self.model,
                        obj_gen=self.obj_gen,
                        obj_t=batch_t["obj_t"],
                        edge_t=batch_t["edge_t"],
                        rel_pos_t=batch_t["rel_pos_t"],
                        t=batch_t["t"],
                        node_mask=batch_t["node_mask"],
                        edge_mask=batch_t["edge_mask"],
                        stochastic_obj=getattr(self.opt, "sampled_state_stochastic_obj", False),
                        stochastic_edge=getattr(self.opt, "sampled_state_stochastic_edge", False),
                        stochastic_rel=getattr(self.opt, "sampled_state_stochastic_rel", False),
                    )

                    valid_prev_mask = sampled_prev["valid_prev_mask"]

                    if valid_prev_mask.any():
                        batch_prev = self.slice_batch_dict(batch_t, valid_prev_mask)

                        batch_prev["obj_t"] = sampled_prev["obj_prev"][valid_prev_mask]
                        batch_prev["edge_t"] = sampled_prev["edge_prev"][valid_prev_mask]
                        batch_prev["rel_pos_t"] = sampled_prev["rel_prev"][valid_prev_mask]
                        batch_prev["t"] = batch_t["t"][valid_prev_mask] - 1

                        model_out_prev = self.model(
                            obj_t=batch_prev["obj_t"],
                            edge_t=batch_prev["edge_t"],
                            rel_pos_t=batch_prev["rel_pos_t"],
                            t=batch_prev["t"],
                            node_mask=batch_prev["node_mask"],
                            edge_mask=batch_prev["edge_mask"],
                        )

                        loss_dict_prev = compute_structured_sg_loss(
                            model_out=model_out_prev,
                            batch_t=batch_prev,
                            no_rel_token_id=self.opt.no_rel_token_id,
                            lambda_obj=self.opt.lambda_obj,
                            lambda_edge=self.opt.lambda_edge,
                            lambda_rel=self.opt.lambda_rel,
                            lambda_layout=getattr(self.opt, "lambda_layout", 1.0),
                            edge_exist_thres=self.opt.edge_exist_thres,
                            edge_pos_weight=self.opt.edge_pos_weight,
                            obj_class_weights=self.obj_class_weights
                            if getattr(self.opt, "use_object_class_weights", False) and training
                            else None,
                            node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
                            use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
                            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                            pred_obj_override=None,
                            pred_rel_full_override=None,
                            use_layout_supervision=getattr(self.opt, "use_layout_supervision", False),
                            layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
                            use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
                            lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),
                            lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
                            relation_vocab=self.train_dataset.relation_vocab,
                            rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
                            use_relation_geometry_loss=getattr(self.opt, "use_relation_geometry_loss", False),
                            lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0),
                            layout_prior_mean=self.layout_prior_mean,
                            layout_prior_var=self.layout_prior_var,
                            layout_prior_valid=self.layout_prior_valid,
                            layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),
                            use_layout_regularization=getattr(self.opt, "use_layout_regularization", False),
                            layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.10),
                            layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.05),
                            layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),
                            use_relation_geometry_reg=getattr(self.opt, "use_relation_geometry_reg", False),
                            lambda_relation_geometry_reg=getattr(self.opt, "lambda_relation_geometry_reg", 0.0),
                            relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.03),
                            use_graph_law_reg=getattr(self.opt, "use_graph_law_reg", False),
                            lambda_graph_law_reg=getattr(self.opt, "lambda_graph_law_reg", 0.0),
                            graph_law_edge_weight=getattr(self.opt, "graph_law_edge_weight", 1.0),
                            graph_law_degree_weight=getattr(self.opt, "graph_law_degree_weight", 0.5),
                            graph_law_rel_weight=getattr(self.opt, "graph_law_rel_weight", 0.5),
                            graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
                            object_only_sanity=getattr(self.opt, "object_only_sanity", False),
                        )

                        sampled_state_loss_val = loss_dict_prev["loss"]
                        loss = loss + getattr(self.opt, "lambda_sampled_state", 0.25) * sampled_state_loss_val
                
                # --------------------------------------------------
                # Phase 4F.2 refinement pass
                # --------------------------------------------------
                refine_loss_val = None

                if getattr(self.opt, "use_refinement_pass", False):
                    pred_1 = build_discrete_predictions_from_model_out(
                        model_out=model_out_1,
                        edge_exist_thres=self.opt.edge_exist_thres,
                    )

                    pred_obj_1 = pred_1["pred_obj"]
                    pred_edge_1 = pred_1["pred_edge"]
                    pred_rel_pos_1 = pred_1["pred_rel_pos"]

                    if getattr(self.opt, "refine_detach_first_pass", True):
                        pred_obj_1 = pred_obj_1.detach()
                        pred_edge_1 = pred_edge_1.detach()
                        pred_rel_pos_1 = pred_rel_pos_1.detach()

                    if getattr(self.opt, "refine_use_pred_structure", True):
                        edge_refine_t = pred_edge_1
                        rel_pos_refine_t = pred_rel_pos_1
                    else:
                        edge_refine_t = batch_t["edge_t"]
                        rel_pos_refine_t = batch_t["rel_pos_t"]

                    model_out_2 = self.model(
                        obj_t=pred_obj_1,
                        edge_t=edge_refine_t,
                        rel_pos_t=rel_pos_refine_t,
                        t=batch_t["t"],
                        node_mask=batch_t["node_mask"],
                        edge_mask=batch_t["edge_mask"],
                    )

                    if getattr(self.opt, "use_residual_refine_weighting", False):
                        loss_dict_2 = compute_refinement_sg_loss(
                            model_out_refine=model_out_2,
                            batch_t=batch_t,
                            first_pass_pred_obj=pred_obj_1,
                            no_rel_token_id=self.opt.no_rel_token_id,
                            lambda_obj=self.opt.lambda_obj,
                            lambda_edge=self.opt.lambda_edge,
                            lambda_rel=self.opt.lambda_rel,
                            edge_exist_thres=self.opt.edge_exist_thres,
                            edge_pos_weight=self.opt.edge_pos_weight,
                            obj_class_weights=self.obj_class_weights
                            if getattr(self.opt, "use_object_class_weights", False) and training
                            else None,
                            refine_obj_wrong_weight=getattr(self.opt, "refine_obj_wrong_weight", 3.0),
                            refine_obj_base_weight=getattr(self.opt, "refine_obj_base_weight", 0.25),
                            use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
                            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                        )
                    else:
                        loss_dict_2 = compute_structured_sg_loss(
                            model_out=model_out_2,
                            batch_t=batch_t,
                            no_rel_token_id=self.opt.no_rel_token_id,
                            lambda_obj=self.opt.lambda_obj,
                            lambda_edge=self.opt.lambda_edge,
                            lambda_rel=self.opt.lambda_rel,
                            lambda_layout=getattr(self.opt, "lambda_layout", 1.0),
                            edge_exist_thres=self.opt.edge_exist_thres,
                            edge_pos_weight=self.opt.edge_pos_weight,
                            obj_class_weights=self.obj_class_weights
                            if getattr(self.opt, "use_object_class_weights", False) and training
                            else None,
                            node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
                            use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
                            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                            pred_obj_override=None,
                            pred_rel_full_override=None,
                            use_layout_supervision=getattr(self.opt, "use_layout_supervision", False),
                            layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
                            use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
                            lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),
                            lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
                            relation_vocab=self.train_dataset.relation_vocab,
                            rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
                            use_relation_geometry_loss=getattr(self.opt, "use_relation_geometry_loss", False),
                            lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0),
                            layout_prior_mean=self.layout_prior_mean,
                            layout_prior_var=self.layout_prior_var,
                            layout_prior_valid=self.layout_prior_valid,
                            layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),
                            use_layout_regularization=getattr(self.opt, "use_layout_regularization", False),
                            layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.10),
                            layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.05),
                            layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),
                            use_relation_geometry_reg=getattr(self.opt, "use_relation_geometry_reg", False),
                            lambda_relation_geometry_reg=getattr(self.opt, "lambda_relation_geometry_reg", 0.0),
                            relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.03),
                            use_graph_law_reg=getattr(self.opt, "use_graph_law_reg", False),
                            lambda_graph_law_reg=getattr(self.opt, "lambda_graph_law_reg", 0.0),
                            graph_law_edge_weight=getattr(self.opt, "graph_law_edge_weight", 1.0),
                            graph_law_degree_weight=getattr(self.opt, "graph_law_degree_weight", 0.5),
                            graph_law_rel_weight=getattr(self.opt, "graph_law_rel_weight", 0.5),
                            graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
                            object_only_sanity=getattr(self.opt, "object_only_sanity", False),
                        )

                    refine_loss_val = loss_dict_2["loss"]
                    loss = loss + getattr(self.opt, "lambda_refine", 0.5) * refine_loss_val
                
                # --------------------------------------------------
                # 8E.6: sampled-state layout denoising
                # Train layout head on model-sampled graph states, supervised by GT boxes.
                # This targets P(L_clean | G_sampled), not graph reconstruction.
                # --------------------------------------------------
                sampled_layout_loss_val = None
                sampled_layout_l1_val = None
                sampled_layout_giou_val = None

                if (
                    training
                    and getattr(self.opt, "use_sampled_layout_training", False)
                    and getattr(self.opt, "use_layout_supervision", False)
                    and ("boxes" in batch_t)
                ):
                    with torch.no_grad():
                        sampled_prev = sample_prev_state_from_current_batch(
                            model=self.model,
                            obj_gen=self.obj_gen,
                            obj_t=batch_t["obj_t"],
                            edge_t=batch_t["edge_t"],
                            rel_pos_t=batch_t["rel_pos_t"],
                            t=batch_t["t"],
                            node_mask=batch_t["node_mask"],
                            edge_mask=batch_t["edge_mask"],
                            stochastic_obj=getattr(self.opt, "sampled_layout_stochastic_obj", True),
                            stochastic_edge=getattr(self.opt, "sampled_layout_stochastic_edge", True),
                            stochastic_rel=getattr(self.opt, "sampled_layout_stochastic_rel", True),
                        )

                    valid_prev_mask = sampled_prev["valid_prev_mask"]

                    if valid_prev_mask.any():
                        batch_layout = self.slice_batch_dict(batch_t, valid_prev_mask)

                        obj_sampled = sampled_prev["obj_prev"][valid_prev_mask]
                        edge_sampled = sampled_prev["edge_prev"][valid_prev_mask]
                        rel_sampled = sampled_prev["rel_prev"][valid_prev_mask]

                        if getattr(self.opt, "sampled_layout_detach_state", True):
                            obj_sampled = obj_sampled.detach()
                            edge_sampled = edge_sampled.detach()
                            rel_sampled = rel_sampled.detach()

                        batch_layout["obj_t"] = obj_sampled
                        batch_layout["edge_t"] = edge_sampled
                        batch_layout["rel_pos_t"] = rel_sampled
                        batch_layout["t"] = (batch_t["t"][valid_prev_mask] - 1).clamp(min=0)

                        # Keep GT layout target attached to the same original scene.
                        batch_layout["boxes"] = batch_t["boxes"][valid_prev_mask]
                        batch_layout["box_valid_mask"] = batch_t["box_valid_mask"][valid_prev_mask]

                        model_out_layout = self.model(
                            obj_t=batch_layout["obj_t"],
                            edge_t=batch_layout["edge_t"],
                            rel_pos_t=batch_layout["rel_pos_t"],
                            t=batch_layout["t"],
                            node_mask=batch_layout["node_mask"],
                            edge_mask=batch_layout["edge_mask"],
                        )

                        loss_dict_layout = compute_structured_sg_loss(
                            model_out=model_out_layout,
                            batch_t=batch_layout,
                            no_rel_token_id=self.opt.no_rel_token_id,

                            # Important: this auxiliary branch is layout-only.
                            lambda_obj=0.0,
                            lambda_edge=0.0,
                            lambda_rel=0.0,
                            lambda_layout=getattr(self.opt, "lambda_layout", 1.0),

                            edge_exist_thres=self.opt.edge_exist_thres,
                            edge_pos_weight=self.opt.edge_pos_weight,
                            obj_class_weights=None,
                            rel_class_weights=None,
                            node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
                            use_object_focal_loss=False,
                            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
                            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
                            pred_obj_override=None,
                            pred_rel_full_override=None,

                            use_layout_supervision=True,
                            layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
                            use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
                            lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),

                            # Usually OFF for this auxiliary branch at first.
                            lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
                            relation_vocab=self.train_dataset.relation_vocab,
                            rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
                            use_relation_geometry_loss=False,

                            use_layout_class_priors=getattr(self.opt, "sampled_layout_use_class_prior", True),
                            lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0)
                            if getattr(self.opt, "sampled_layout_use_class_prior", True)
                            else 0.0,
                            layout_prior_mean=self.layout_prior_mean,
                            layout_prior_var=self.layout_prior_var,
                            layout_prior_valid=self.layout_prior_valid,
                            layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),

                            use_layout_regularization=False,
                            layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.02),
                            layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.01),
                            layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),

                            use_relation_geometry_reg=False,
                            lambda_relation_geometry_reg=0.0,
                            relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.02),
                            use_graph_law_reg=False,
                            lambda_graph_law_reg=0.0,
                            graph_law_edge_weight=0.0,
                            graph_law_degree_weight=0.0,
                            graph_law_rel_weight=0.0,
                            graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
                            object_only_sanity=getattr(self.opt, "object_only_sanity", False),
                        )

                        sampled_layout_loss_val = loss_dict_layout["layout_loss"]
                        sampled_layout_l1_val = loss_dict_layout.get("layout_l1", None)
                        sampled_layout_giou_val = loss_dict_layout.get("layout_giou_loss", None)

                        loss = loss + getattr(self.opt, "lambda_sampled_layout", 0.25) * sampled_layout_loss_val

                # --------------------------------------------------
                # Backward
                # --------------------------------------------------
                if training:
                    loss.backward()
                    if getattr(self.opt, "grad_clip_norm", 0.0) and self.opt.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip_norm)
                    self.optimizer.step()

                # --------------------------------------------------
                # Accumulate main pass metrics
                # --------------------------------------------------
                total_obj_loss_sum += float(loss_dict_1["obj_loss"].item())
                total_obj_count += 1.0

                total_edge_loss_sum += float(loss_dict_1["edge_loss"].item())
                total_edge_count += 1.0

                total_rel_loss_sum += float(loss_dict_1["rel_loss"].item())
                total_rel_count += 1.0

                total_layout_loss_sum += float(loss_dict_1["layout_loss"].item())
                total_layout_count += 1.0

                total_layout_l1_sum += float(loss_dict_1["layout_l1"].item())
                total_layout_l1_count += 1.0

                total_loss_sum += float(loss.item())
                total_loss_count += 1.0

                total_tp_edges += float(loss_dict_1["tp_edges"].item())
                total_fp_edges += float(loss_dict_1["fp_edges"].item())
                total_fn_edges += float(loss_dict_1["fn_edges"].item())
                total_pred_pos_edges += float(loss_dict_1["pred_num_pos_edges"].item())
                total_gt_pos_edges += float(loss_dict_1["gt_num_pos_edges"].item())

                total_rel_acc_sum += float(loss_dict_1["relation_accuracy_on_true_positive_edges"].item())
                total_rel_acc_count += 1.0

                total_node_correct_all += float(
                    (loss_dict_1["node_acc_all"] * loss_dict_1["node_count_all"]).item()
                )
                total_node_count_all += float(loss_dict_1["node_count_all"].item())

                total_node_correct_corrupted += float(
                    (loss_dict_1["node_acc_corrupted"] * loss_dict_1["node_count_corrupted"]).item()
                )
                total_node_count_corrupted += float(loss_dict_1["node_count_corrupted"].item())

                if cond_node_loss_val is not None:
                    total_cond_node_loss_sum += float(cond_node_loss_val.item())
                    total_cond_node_count += float(cond_node_count_val.item())
                    total_cond_node_correct += float((cond_node_acc_val * cond_node_count_val).item())

                if refine_loss_val is not None:
                    total_refine_loss_sum += float(refine_loss_val.item())
                    total_refine_loss_count += 1.0
                
                if sampled_state_loss_val is not None:
                    total_sampled_state_loss_sum += float(sampled_state_loss_val.item())
                    total_sampled_state_loss_count += 1.0
                
                if reverse_step_loss_val is not None:
                    total_reverse_step_loss_sum += float(reverse_step_loss_val.item())
                    total_reverse_step_loss_count += 1.0
                
                if reverse_vocab_loss_val is not None:
                    total_reverse_vocab_loss_sum += float(reverse_vocab_loss_val.item())
                    total_reverse_vocab_loss_count += 1.0
                
                if sampled_layout_loss_val is not None:
                    total_sampled_layout_loss_sum += float(sampled_layout_loss_val.item())
                    total_sampled_layout_loss_count += 1.0

                if sampled_layout_l1_val is not None:
                    total_sampled_layout_l1_sum += float(sampled_layout_l1_val.item())

                if sampled_layout_giou_val is not None:
                    total_sampled_layout_giou_val_sum += float(sampled_layout_giou_val.item())
                
                if "layout_overlap_reg" in loss_dict_1:
                    total_layout_overlap_reg_sum += float(loss_dict_1["layout_overlap_reg"].detach().item())
                    total_layout_spread_reg_sum += float(loss_dict_1["layout_spread_reg"].detach().item())
                    total_layout_center_spread_sum += float(loss_dict_1["layout_center_spread"].detach().item())
                    total_layout_reg_loss_sum += float(loss_dict_1["layout_reg_loss"].detach().item())
                    total_layout_reg_count += 1.0
                
                if "relation_geometry_reg" in loss_dict_1:
                    total_relation_geometry_reg_sum += float(loss_dict_1["relation_geometry_reg"].item())
                    total_relation_geometry_reg_count += 1.0

                    total_rel_geom_reg_behind_count += float(loss_dict_1["rel_geom_reg_behind_count"].item())
                    total_rel_geom_reg_front_count += float(loss_dict_1["rel_geom_reg_front_count"].item())
                    total_rel_geom_reg_above_count += float(loss_dict_1["rel_geom_reg_above_count"].item())
                    total_rel_geom_reg_below_count += float(loss_dict_1["rel_geom_reg_below_count"].item())
                    total_rel_geom_reg_inside_count += float(loss_dict_1["rel_geom_reg_inside_count"].item())
                    total_rel_geom_reg_on_count += float(loss_dict_1["rel_geom_reg_on_count"].item())
                
                if "layout_giou_loss" in loss_dict_1:
                    total_layout_giou_loss_sum += float(loss_dict_1["layout_giou_loss"].item())
                    total_layout_mean_giou_sum += float(loss_dict_1["layout_mean_giou"].item())
                
                if "rel_geometry_loss" in loss_dict_1:
                    total_rel_geometry_loss_sum += float(loss_dict_1["rel_geometry_loss"].item())
                    total_rel_geometry_loss_count += 1.0

                    total_rel_geom_left_count += float(loss_dict_1["rel_geom_left_count"].item())
                    total_rel_geom_right_count += float(loss_dict_1["rel_geom_right_count"].item())
                    total_rel_geom_above_count += float(loss_dict_1["rel_geom_above_count"].item())
                    total_rel_geom_below_count += float(loss_dict_1["rel_geom_below_count"].item())
                    total_rel_geom_inside_count += float(loss_dict_1["rel_geom_inside_count"].item())
                
                if "layout_class_prior_loss" in loss_dict_1:
                    total_layout_class_prior_loss_sum += float(loss_dict_1["layout_class_prior_loss"].item())
                    total_layout_class_prior_loss_count += 1.0
                
                if not training and self.opt.use_reward_tilting:
                    total_reward_sum += float(reward_terms_eval["reward_total"].mean().item())
                    total_reward_count += 1.0
                
                if "graph_law_reg" in loss_dict_1:
                    total_graph_law_reg_sum += float(loss_dict_1["graph_law_reg"].detach().item())
                    total_graph_edge_density_reg_sum += float(loss_dict_1["graph_edge_density_reg"].detach().item())
                    total_graph_degree_reg_sum += float(loss_dict_1["graph_degree_reg"].detach().item())
                    total_graph_rel_marginal_reg_sum += float(loss_dict_1["graph_rel_marginal_reg"].detach().item())
                    total_pred_edge_density_sum += float(loss_dict_1["pred_edge_density"].detach().item())
                    total_gt_edge_density_sum += float(loss_dict_1["gt_edge_density"].detach().item())
                    total_graph_law_count += 1.0
                    
                

                # --------------------------------------------------
                # Postfix
                # --------------------------------------------------
                if is_main_process():
                    cur_loss = total_loss_sum / max(total_loss_count, 1.0)
                    cur_obj = total_obj_loss_sum / max(total_obj_count, 1.0)
                    cur_edge = total_edge_loss_sum / max(total_edge_count, 1.0)
                    cur_rel = total_rel_loss_sum / max(total_rel_count, 1.0)

                    cur_layout = total_layout_loss_sum / max(total_layout_count, 1.0)
                    cur_layout_l1 = total_layout_l1_sum / max(total_layout_l1_count, 1.0)

                    cur_edge_precision = total_tp_edges / max(total_tp_edges + total_fp_edges, 1.0)
                    cur_edge_recall = total_tp_edges / max(total_tp_edges + total_fn_edges, 1.0)
                    cur_edge_f1 = (
                        2.0 * cur_edge_precision * cur_edge_recall
                        / max(cur_edge_precision + cur_edge_recall, 1e-12)
                    )
                    cur_edge_ratio = total_pred_pos_edges / max(total_gt_pos_edges, 1.0)

                    cur_node_all = total_node_correct_all / max(total_node_count_all, 1.0)
                    cur_node_corr = total_node_correct_corrupted / max(total_node_count_corrupted, 1.0)

                    postfix = {
                        "loss": f"{cur_loss:.4f}",
                        "obj": f"{cur_obj:.4f}",
                        "edge": f"{cur_edge:.4f}",
                        "rel": f"{cur_rel:.4f}",
                        "edge_p": f"{cur_edge_precision:.4f}",
                        "edge_r": f"{cur_edge_recall:.4f}",
                        "edge_f1": f"{cur_edge_f1:.4f}",
                        "edge_ratio": f"{cur_edge_ratio:.4f}",
                        "node_all": f"{cur_node_all:.4f}",
                        "node_corr": f"{cur_node_corr:.4f}",
                        "layout": f"{cur_layout:.4f}",
                        "box_l1": f"{cur_layout_l1:.4f}",
                    }

                    if total_cond_node_count > 0:
                        cur_cond_acc = total_cond_node_correct / max(total_cond_node_count, 1.0)
                        postfix["cond_acc"] = f"{cur_cond_acc:.4f}"

                    if total_refine_loss_count > 0:
                        cur_ref = total_refine_loss_sum / max(total_refine_loss_count, 1.0)
                        postfix["ref"] = f"{cur_ref:.4f}"
                    
                    if total_sampled_state_loss_count > 0:
                        cur_sampled = total_sampled_state_loss_sum / max(total_sampled_state_loss_count, 1.0)
                        postfix["samp"] = f"{cur_sampled:.4f}"
                    
                    if total_reverse_step_loss_count > 0:
                        cur_rev = total_reverse_step_loss_sum / max(total_reverse_step_loss_count, 1.0)
                        postfix["rev"] = f"{cur_rev:.4f}"
                    
                    if total_reverse_vocab_loss_count > 0:
                        cur_revh = total_reverse_vocab_loss_sum / max(total_reverse_vocab_loss_count, 1.0)
                        postfix["revh"] = f"{cur_revh:.4f}"

                    if total_reverse_branch_count > 0:
                        cur_rev_obj = total_reverse_obj_loss_sum / max(total_reverse_branch_count, 1.0)
                        cur_rev_edge = total_reverse_edge_loss_sum / max(total_reverse_branch_count, 1.0)
                        cur_rev_rel = total_reverse_rel_loss_sum / max(total_reverse_branch_count, 1.0)
                        postfix["rev_obj"] = f"{cur_rev_obj:.4f}"
                        postfix["rev_edge"] = f"{cur_rev_edge:.4f}"
                        postfix["rev_rel"] = f"{cur_rev_rel:.4f}"
                    
                    if total_sampled_layout_loss_count > 0:
                        cur_slay = total_sampled_layout_loss_sum / max(total_sampled_layout_loss_count, 1.0)
                        postfix["slay"] = f"{cur_slay:.4f}"

                        cur_slay_l1 = total_sampled_layout_l1_sum / max(total_sampled_layout_loss_count, 1.0)
                        postfix["slay_l1"] = f"{cur_slay_l1:.4f}"

                        cur_slay_giou = total_sampled_layout_giou_val_sum / max(total_sampled_layout_loss_count, 1.0)
                        postfix["slay_giou"] = f"{cur_slay_giou:.4f}"
                    
                    if self.opt.use_layout_giou_loss:
                        cur_lay_giou = total_layout_giou_loss_sum / max(total_layout_count, 1.0)
                        cur_mean_giou = total_layout_mean_giou_sum / max(total_layout_count, 1.0)
                        postfix["lay_giou"] = f"{cur_lay_giou:.4f}"
                        postfix["mean_giou"] = f"{cur_mean_giou:.4f}"
                    
                    if "layout_overlap_reg" in loss_dict_1:
                        cur_layout_overlap_reg_sum = total_layout_overlap_reg_sum / max(total_layout_reg_count, 1.0)
                        cur_layout_spread_reg_sum = total_layout_spread_reg_sum / max(total_layout_reg_count, 1.0)
                        cur_layout_center_spread_reg_sum = total_layout_center_spread_sum / max(total_layout_reg_count, 1.0)
                        cur_layout_reg_loss_sum = total_layout_reg_loss_sum / max(total_layout_reg_count, 1.0)
                        postfix["lay_ovreg"] = f"{cur_layout_overlap_reg_sum:.4f}"
                        postfix["lay_spreg"] = f"{cur_layout_spread_reg_sum:.4f}"
                        postfix["lay_spread"] = f"{cur_layout_center_spread_reg_sum:.4f}"
                        postfix["lay_reg_loss"] = f"{cur_layout_reg_loss_sum:.4f}"
                    
                    if total_rel_geometry_loss_count > 0:
                        cur_rel_geom = total_rel_geometry_loss_sum / max(total_rel_geometry_loss_count, 1.0)
                        postfix["rgeom"] = f"{cur_rel_geom:.4f}"

                    if total_relation_geometry_reg_count > 0:
                        cur_rel_geom_reg = total_relation_geometry_reg_sum / max(total_relation_geometry_reg_count, 1.0)
                        postfix["rgeom_reg"] = f"{cur_rel_geom_reg:.4f}"
                    
                    if total_layout_class_prior_loss_count > 0:
                        cur_lprior = total_layout_class_prior_loss_sum / max(total_layout_class_prior_loss_count, 1.0)
                        postfix["lay_prior"] = f"{cur_lprior:.4f}"
                    
                    if not training and self.opt.use_reward_tilting:
                        cur_reward = total_reward_sum / max(total_reward_count, 1.0)
                        postfix["reward"] = f"{cur_reward:.4f}"
                    
                    if total_graph_law_count > 0:
                        postfix["glaw"] = f"{total_graph_law_reg_sum / max(total_graph_law_count, 1.0):.4f}"
                        postfix["gden"] = f"{total_graph_edge_density_reg_sum / max(total_graph_law_count, 1.0):.4f}"
                        postfix["gdeg"] = f"{total_graph_degree_reg_sum / max(total_graph_law_count, 1.0):.4f}"
                        postfix["grel"] = f"{total_graph_rel_marginal_reg_sum / max(total_graph_law_count, 1.0):.4f}"

                    iterator.set_postfix(**postfix)

        # --------------------------------------------------
        # Reduce across ranks
        # --------------------------------------------------
        total_obj_loss_sum = reduce_scalar_sum(total_obj_loss_sum, self.device)
        total_obj_count = reduce_scalar_sum(total_obj_count, self.device)

        total_edge_loss_sum = reduce_scalar_sum(total_edge_loss_sum, self.device)
        total_edge_count = reduce_scalar_sum(total_edge_count, self.device)

        total_rel_loss_sum = reduce_scalar_sum(total_rel_loss_sum, self.device)
        total_rel_count = reduce_scalar_sum(total_rel_count, self.device)

        total_layout_loss_sum = reduce_scalar_sum(total_layout_loss_sum, self.device)
        total_layout_count = reduce_scalar_sum(total_layout_count, self.device)

        total_layout_l1_sum = reduce_scalar_sum(total_layout_l1_sum, self.device)
        total_layout_l1_count = reduce_scalar_sum(total_layout_l1_count, self.device)

        total_layout_giou_loss_sum = reduce_scalar_sum(total_layout_giou_loss_sum, self.device)
        total_layout_mean_giou_sum = reduce_scalar_sum(total_layout_mean_giou_sum, self.device)

        total_rel_geometry_loss_sum = reduce_scalar_sum(total_rel_geometry_loss_sum, self.device)
        total_rel_geometry_loss_count = reduce_scalar_sum(total_rel_geometry_loss_count, self.device)

        total_layout_class_prior_loss_sum = reduce_scalar_sum(total_layout_class_prior_loss_sum, self.device)
        total_layout_class_prior_loss_count = reduce_scalar_sum(total_layout_class_prior_loss_count, self.device)

        total_loss_sum = reduce_scalar_sum(total_loss_sum, self.device)
        total_loss_count = reduce_scalar_sum(total_loss_count, self.device)

        total_tp_edges = reduce_scalar_sum(total_tp_edges, self.device)
        total_fp_edges = reduce_scalar_sum(total_fp_edges, self.device)
        total_fn_edges = reduce_scalar_sum(total_fn_edges, self.device)
        total_pred_pos_edges = reduce_scalar_sum(total_pred_pos_edges, self.device)
        total_gt_pos_edges = reduce_scalar_sum(total_gt_pos_edges, self.device)

        total_rel_acc_sum = reduce_scalar_sum(total_rel_acc_sum, self.device)
        total_rel_acc_count = reduce_scalar_sum(total_rel_acc_count, self.device)

        total_node_correct_all = reduce_scalar_sum(total_node_correct_all, self.device)
        total_node_count_all = reduce_scalar_sum(total_node_count_all, self.device)

        total_node_correct_corrupted = reduce_scalar_sum(total_node_correct_corrupted, self.device)
        total_node_count_corrupted = reduce_scalar_sum(total_node_count_corrupted, self.device)

        total_cond_node_loss_sum = reduce_scalar_sum(total_cond_node_loss_sum, self.device)
        total_cond_node_count = reduce_scalar_sum(total_cond_node_count, self.device)
        total_cond_node_correct = reduce_scalar_sum(total_cond_node_correct, self.device)

        total_refine_loss_sum = reduce_scalar_sum(total_refine_loss_sum, self.device)
        total_refine_loss_count = reduce_scalar_sum(total_refine_loss_count, self.device)

        total_sampled_state_loss_sum = reduce_scalar_sum(total_sampled_state_loss_sum, self.device)
        total_sampled_state_loss_count = reduce_scalar_sum(total_sampled_state_loss_count, self.device)

        total_reverse_step_loss_sum = reduce_scalar_sum(total_reverse_step_loss_sum, self.device)
        total_reverse_step_loss_count = reduce_scalar_sum(total_reverse_step_loss_count, self.device)

        total_reverse_vocab_loss_sum = reduce_scalar_sum(total_reverse_vocab_loss_sum, self.device)
        total_reverse_vocab_loss_count = reduce_scalar_sum(total_reverse_vocab_loss_count, self.device)

        total_reverse_obj_loss_sum = reduce_scalar_sum(total_reverse_obj_loss_sum, self.device)
        total_reverse_edge_loss_sum = reduce_scalar_sum(total_reverse_edge_loss_sum, self.device)
        total_reverse_rel_loss_sum = reduce_scalar_sum(total_reverse_rel_loss_sum, self.device)
        total_reverse_branch_count = reduce_scalar_sum(total_reverse_branch_count, self.device)

        total_sampled_layout_loss_sum = reduce_scalar_sum(total_sampled_layout_loss_sum, self.device)
        total_sampled_layout_loss_count = reduce_scalar_sum(total_sampled_layout_loss_count, self.device)
        total_sampled_layout_l1_sum = reduce_scalar_sum(total_sampled_layout_l1_sum, self.device)
        total_sampled_layout_giou_val_sum = reduce_scalar_sum(total_sampled_layout_giou_val_sum, self.device)

        total_layout_overlap_reg_sum = reduce_scalar_sum(total_layout_overlap_reg_sum, self.device)
        total_layout_spread_reg_sum = reduce_scalar_sum(total_layout_spread_reg_sum, self.device)
        total_layout_center_spread_sum = reduce_scalar_sum(total_layout_center_spread_sum, self.device)
        total_layout_reg_loss_sum = reduce_scalar_sum(total_layout_reg_loss_sum, self.device)

        total_layout_reg_count = reduce_scalar_sum(total_layout_reg_count, self.device)

        total_reward_sum = reduce_scalar_sum(total_reward_sum, self.device)
        total_reward_count = reduce_scalar_sum(total_reward_count, self.device)

        total_relation_geometry_reg_sum = reduce_scalar_sum(total_relation_geometry_reg_sum, self.device)
        total_relation_geometry_reg_count = reduce_scalar_sum(total_relation_geometry_reg_count, self.device)

        total_rel_geom_reg_behind_count = reduce_scalar_sum(total_rel_geom_reg_behind_count, self.device)
        total_rel_geom_reg_front_count = reduce_scalar_sum(total_rel_geom_reg_front_count, self.device)
        total_rel_geom_reg_above_count = reduce_scalar_sum(total_rel_geom_reg_above_count, self.device)
        total_rel_geom_reg_below_count = reduce_scalar_sum(total_rel_geom_reg_below_count, self.device)
        total_rel_geom_reg_inside_count = reduce_scalar_sum(total_rel_geom_reg_inside_count, self.device)
        total_rel_geom_reg_on_count = reduce_scalar_sum(total_rel_geom_reg_on_count, self.device)

        total_graph_law_reg_sum = reduce_scalar_sum(total_graph_law_reg_sum, self.device)
        total_graph_edge_density_reg_sum = reduce_scalar_sum(total_graph_edge_density_reg_sum, self.device)
        total_graph_degree_reg_sum = reduce_scalar_sum(total_graph_degree_reg_sum, self.device)
        total_graph_rel_marginal_reg_sum = reduce_scalar_sum(total_graph_rel_marginal_reg_sum, self.device)
        total_pred_edge_density_sum = reduce_scalar_sum(total_pred_edge_density_sum, self.device)
        total_gt_edge_density_sum = reduce_scalar_sum(total_gt_edge_density_sum, self.device)
        total_graph_law_count = reduce_scalar_sum(total_graph_law_count, self.device)

        # --------------------------------------------------
        # Final metrics
        # --------------------------------------------------
        obj_loss = total_obj_loss_sum / max(total_obj_count, 1e-12)
        edge_loss = total_edge_loss_sum / max(total_edge_count, 1e-12)
        rel_loss = total_rel_loss_sum / max(total_rel_count, 1e-12)

        layout_loss = total_layout_loss_sum / max(total_layout_count, 1e-12)
        layout_l1 = total_layout_l1_sum / max(total_layout_l1_count, 1e-12)

        layout_giou_loss = total_layout_giou_loss_sum / max(total_layout_count, 1e-12)
        layout_mean_giou = total_layout_mean_giou_sum / max(total_layout_count, 1e-12)

        total_loss = total_loss_sum / max(total_loss_count, 1e-12)

        edge_precision = total_tp_edges / max(total_tp_edges + total_fp_edges, 1e-12)
        edge_recall = total_tp_edges / max(total_tp_edges + total_fn_edges, 1e-12)
        edge_f1 = 2.0 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12)

        pred_gt_edge_ratio = total_pred_pos_edges / max(total_gt_pos_edges, 1e-12)
        relation_accuracy_on_true_positive_edges = total_rel_acc_sum / max(total_rel_acc_count, 1e-12)

        node_acc_all = total_node_correct_all / max(total_node_count_all, 1e-12)
        node_acc_corrupted = total_node_correct_corrupted / max(total_node_count_corrupted, 1e-12)

        cond_node_loss = total_cond_node_loss_sum / max(total_cond_node_count, 1e-12) if total_cond_node_count > 0 else 0.0
        cond_node_acc = total_cond_node_correct / max(total_cond_node_count, 1e-12) if total_cond_node_count > 0 else 0.0

        refine_loss = total_refine_loss_sum / max(total_refine_loss_count, 1e-12) if total_refine_loss_count > 0 else 0.0
        sampled_state_loss = (
            total_sampled_state_loss_sum / max(total_sampled_state_loss_count, 1e-12)
            if total_sampled_state_loss_count > 0 else 0.0
        )

        reverse_step_loss = (
            total_reverse_step_loss_sum / max(total_reverse_step_loss_count, 1e-12)
            if total_reverse_step_loss_count > 0 else 0.0
        )

        reverse_vocab_loss = (
            total_reverse_vocab_loss_sum / max(total_reverse_vocab_loss_count, 1e-12)
            if total_reverse_vocab_loss_count > 0 else 0.0
        )

        reverse_obj_loss = (
            total_reverse_obj_loss_sum / max(total_reverse_branch_count, 1e-12)
            if total_reverse_branch_count > 0 else 0.0
        )
        reverse_edge_loss = (
            total_reverse_edge_loss_sum / max(total_reverse_branch_count, 1e-12)
            if total_reverse_branch_count > 0 else 0.0
        )
        reverse_rel_loss = (
            total_reverse_rel_loss_sum / max(total_reverse_branch_count, 1e-12)
            if total_reverse_branch_count > 0 else 0.0
        )

        sampled_layout_loss = (
            total_sampled_layout_loss_sum / max(total_sampled_layout_loss_count, 1e-12)
            if total_sampled_layout_loss_count > 0 else 0.0
        )
        sampled_layout_l1 = (
            total_sampled_layout_l1_sum / max(total_sampled_layout_loss_count, 1e-12)
            if total_sampled_layout_loss_count > 0 else 0.0
        )

        sampled_layout_giou = (
            total_sampled_layout_giou_val_sum / max(total_sampled_layout_loss_count, 1e-12)
            if total_sampled_layout_loss_count > 0 else 0.0
        )

        layout_overlap = (
            total_layout_overlap_reg_sum / max(total_layout_reg_count, 1e-12)
            if total_layout_reg_count > 0 else 0.0
        )

        layout_spreadg = (
            total_layout_spread_reg_sum / max(total_layout_reg_count, 1e-12)
            if total_layout_reg_count > 0 else 0.0
        )

        layout_center_spread = (
            total_layout_center_spread_sum / max(total_layout_reg_count, 1e-12)
            if total_layout_reg_count > 0 else 0.0
        )

        layout_reg_loss = (
            total_layout_reg_loss_sum / max(total_layout_reg_count, 1e-12)
            if total_layout_reg_count > 0 else 0.0
        )

        rel_geometry_loss = (
            total_rel_geometry_loss_sum / max(total_rel_geometry_loss_count, 1e-12)
            if total_rel_geometry_loss_count > 0 else 0.0
        )

        layout_class_prior_loss = (
            total_layout_class_prior_loss_sum / max(total_layout_class_prior_loss_count, 1e-12)
            if total_layout_class_prior_loss_count > 0 else 0.0
        )

        reward_total_eval = total_reward_sum / max(total_reward_count, 1e-12)

        relation_geometry_reg = (
            total_relation_geometry_reg_sum / max(total_relation_geometry_reg_count, 1e-12)
            if total_relation_geometry_reg_count > 0 else 0.0
        )

        graph_law_reg = (
            total_graph_law_reg_sum / max(total_graph_law_count, 1e-12)
            if total_graph_law_count > 0 else 0.0
        )
        graph_edge_density_reg = (
            total_graph_edge_density_reg_sum / max(total_graph_law_count, 1e-12)
            if total_graph_law_count > 0 else 0.0
        )
        graph_degree_reg = (
            total_graph_degree_reg_sum / max(total_graph_law_count, 1e-12)
            if total_graph_law_count > 0 else 0.0
        )
        graph_rel_marginal_reg = (
            total_graph_rel_marginal_reg_sum / max(total_graph_law_count, 1e-12)
            if total_graph_law_count > 0 else 0.0
        )
        pred_edge_density_avg = (
            total_pred_edge_density_sum / max(total_graph_law_count, 1e-12)
            if total_graph_law_count > 0 else 0.0
        )
        gt_edge_density_avg = (
            total_gt_edge_density_sum / max(total_graph_law_count, 1e-12)
            if total_graph_law_count > 0 else 0.0
        )

        out = {
            "loss": total_loss,
            "obj_loss": obj_loss,
            "edge_loss": edge_loss,
            "rel_loss": rel_loss,
            "gt_num_pos_edges": total_gt_pos_edges,
            "pred_num_pos_edges": total_pred_pos_edges,
            "pred_gt_edge_ratio": pred_gt_edge_ratio,
            "tp_edges": total_tp_edges,
            "fp_edges": total_fp_edges,
            "fn_edges": total_fn_edges,
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
            "relation_accuracy_on_true_positive_edges": relation_accuracy_on_true_positive_edges,
            "node_acc_all": node_acc_all,
            "node_acc_corrupted": node_acc_corrupted,
            "cond_node_loss": cond_node_loss,
            "cond_node_acc": cond_node_acc,
            "refine_loss": refine_loss,
            "sampled_state_loss": sampled_state_loss,
            "reverse_step_loss": reverse_step_loss,
            "reverse_vocab_loss": reverse_vocab_loss,
            "reverse_obj_loss": reverse_obj_loss,
            "reverse_edge_loss": reverse_edge_loss,
            "reverse_rel_loss": reverse_rel_loss,
            "layout_loss": layout_loss,
            "layout_l1": layout_l1,
            "sampled_layout_loss": sampled_layout_loss,
            "sampled_layout_l1": sampled_layout_l1,
            "sampled_layout_giou": sampled_layout_giou,
            "layout_giou_loss": layout_giou_loss,
            "layout_mean_giou": layout_mean_giou,
            "rel_geometry_loss": rel_geometry_loss,
            "rel_geom_left_count": total_rel_geom_left_count,
            "rel_geom_right_count": total_rel_geom_right_count,
            "rel_geom_above_count": total_rel_geom_above_count,
            "rel_geom_below_count": total_rel_geom_below_count,
            "rel_geom_inside_count": total_rel_geom_inside_count,
            "layout_class_prior_loss": layout_class_prior_loss,
            "reward_total_eval": reward_total_eval,
            "layout_overlapg":layout_overlap,
            "layout_spreadg":layout_spreadg,
            "layout_center_spread":layout_center_spread,
            "layout_reg_loss": layout_reg_loss,
            "relation_geometry_reg": relation_geometry_reg,
            "rel_geom_reg_behind_count": total_rel_geom_reg_behind_count,
            "rel_geom_reg_front_count": total_rel_geom_reg_front_count,
            "rel_geom_reg_above_count": total_rel_geom_reg_above_count,
            "rel_geom_reg_below_count": total_rel_geom_reg_below_count,
            "rel_geom_reg_inside_count": total_rel_geom_reg_inside_count,
            "rel_geom_reg_on_count": total_rel_geom_reg_on_count,
            "graph_law_reg": graph_law_reg,
            "graph_edge_density_reg": graph_edge_density_reg,
            "graph_degree_reg": graph_degree_reg,
            "graph_rel_marginal_reg": graph_rel_marginal_reg,
            "pred_edge_density_avg": pred_edge_density_avg,
            "gt_edge_density_avg": gt_edge_density_avg,
        }


        return out

    def save_checkpoint(self, epoch: int, is_best: bool = False, best_name: str = "best.pt", save_epoch_file: bool = True):
        if not is_main_process():
            return

        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_rel_loss": self.best_val_rel_loss,
            "config": asdict(self.opt) if hasattr(self.opt, "__dataclass_fields__") else None,
            "num_obj_classes": len(self.train_dataset.object_vocab),
            "num_rel_classes": len(self.train_dataset.relation_vocab),
            "object_vocab": self.train_dataset.object_vocab,
            "relation_vocab": self.train_dataset.relation_vocab,
        }

        if getattr(self.opt, "save_latest_every_epoch", True):
            latest_path = os.path.join(self.opt.checkpoint_dir, "latest.pt")
            torch.save(ckpt, latest_path)

        if save_epoch_file:
            epoch_path = os.path.join(self.opt.checkpoint_dir, f"epoch_{epoch:03d}.pt")
            torch.save(ckpt, epoch_path)

        if is_best:
            best_path = os.path.join(self.opt.checkpoint_dir, best_name)
            torch.save(ckpt, best_path)

    @torch.no_grad()
    def build_full_relation_prediction(self, model_out: dict, edge_mask: torch.Tensor) -> torch.Tensor:
        pred_edge_exists = (torch.sigmoid(model_out["edge_logits"]) >= self.opt.edge_exist_thres) & edge_mask.bool()
        pred_rel_pos = model_out["rel_logits_pos"].argmax(dim=-1)
        pred_rel_full = reconstruct_full_relations(
            pred_edge_exists=pred_edge_exists,
            pred_rel_pos=pred_rel_pos,
            no_rel_token_id=self.opt.no_rel_token_id,
        )
        return pred_rel_full

    @torch.no_grad()
    def build_full_relation_from_noisy_state(self, batch_t: dict) -> torch.Tensor:
        """
        Reconstruct a full-vocabulary relation tensor from the structured noisy state
        for logging / decoding.

        We only decode an edge as positive if ALL of the following hold:
        1. edge_t says the edge exists
        2. the pair is a valid edge_mask location
        3. the edge was a GT-positive edge originally (so rel_pos_t is meaningful)
        4. rel_pos_t is not MASK_REL

        Everything else is mapped to NO_REL for safe decoding.
        """
        edge_mask = batch_t["edge_mask"].bool()              # [B,N,N]
        edge_t = batch_t["edge_t"].bool()                    # [B,N,N]
        gt_pos_edge_mask = batch_t["gt_pos_edge_mask"].bool()# [B,N,N]
        rel_pos_t = batch_t["rel_pos_t"].clone()             # [B,N,N]

        mask_rel_id = self.opt.mask_rel_token_id
        no_rel_id = self.opt.no_rel_token_id

        # relation token is meaningful only on originally positive edges
        meaningful_rel_token = gt_pos_edge_mask

        # masked relation token should not be decoded as a real predicate
        not_masked_rel = (rel_pos_t != mask_rel_id)

        # only decode edges that are valid, currently on, originally positive,
        # and not carrying MASK_REL
        # pred_edge_exists = edge_mask & edge_t & meaningful_rel_token & not_masked_rel
        pred_edge_exists = gt_pos_edge_mask & edge_t & not_masked_rel

        # safe placeholder where pred_edge_exists is False
        rel_pos_safe = rel_pos_t.clone()
        rel_pos_safe[~pred_edge_exists] = 0

        rel_full = reconstruct_full_relations(
            pred_edge_exists=pred_edge_exists,
            pred_rel_pos=rel_pos_safe,
            no_rel_token_id=no_rel_id,
        )
        return rel_full

    @torch.no_grad()
    def log_validation_graph_examples(self, epoch: int):
        if not is_main_process():
            return
        if self.val_loader is None:
            return

        batch = next(iter(self.val_loader))
        

        batch_t = self.obj_gen.get_training_batch(batch)

        if "boxes" in batch:
            batch_t["boxes"] = batch["boxes"].to(self.device)  # [B,N,4]
            batch_t["box_valid_mask"] = batch["node_mask"].bool().to(self.device)


        self.model.eval()

        model_out = self.model(
            obj_t=batch_t["obj_t"],
            edge_t=batch_t["edge_t"],
            rel_pos_t=batch_t["rel_pos_t"],
            t=batch_t["t"],
            node_mask=batch_t["node_mask"],
            edge_mask=batch_t["edge_mask"],
            edge_input_override=batch_t["edge_0"],
            rel_input_override=batch_t["rel_pos_0"],
        )

        pred_obj_override = None
        pred_rel_full_override = None

        if getattr(self.opt, "use_node_gibbs", False):
            pred_obj_override, edge_fixed, pred_rel_full_override, rel_pos_fixed = run_node_gibbs_sampler(
                model=self.model,
                obj_t=batch_t["obj_t"],
                edge_t=batch_t["edge_t"],
                rel_pos_t=batch_t["rel_pos_t"],
                t=batch_t["t"],
                node_mask=batch_t["node_mask"],
                edge_mask=batch_t["edge_mask"],
                no_rel_token_id=self.opt.no_rel_token_id,
                mask_rel_token_id=getattr(self.opt, "mask_rel_token_id", None),
                edge_exist_thres=self.opt.edge_exist_thres,
                num_sweeps=getattr(self.opt, "num_node_gibbs_sweeps", 1),
                sample_temp=getattr(self.opt, "node_gibbs_sample_temp", 1.0),
                use_fixed_structure=getattr(self.opt, "node_gibbs_use_fixed_structure", True),
                random_order=getattr(self.opt, "node_gibbs_random_order", True),
                edge_input_override=batch_t["edge_0"],
                rel_input_override=batch_t["rel_pos_0"],
            )

        loss_dict = compute_structured_sg_loss(
            model_out=model_out,
            batch_t=batch_t,
            no_rel_token_id=self.opt.no_rel_token_id,
            lambda_obj=self.opt.lambda_obj,
            lambda_edge=self.opt.lambda_edge,
            lambda_rel=self.opt.lambda_rel,
            lambda_layout=getattr(self.opt, "lambda_layout", 1.0),
            edge_exist_thres=self.opt.edge_exist_thres,
            edge_pos_weight=self.opt.edge_pos_weight,
            obj_class_weights=None,
            node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
            use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
            pred_obj_override=pred_obj_override,
            pred_rel_full_override=pred_rel_full_override,
            use_layout_supervision=getattr(self.opt, "use_layout_supervision", False),
            layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
            use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
            lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),
            lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
            relation_vocab=self.train_dataset.relation_vocab,
            rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
            use_relation_geometry_loss=getattr(self.opt, "use_relation_geometry_loss", False),
            lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0),
            layout_prior_mean=self.layout_prior_mean,
            layout_prior_var=self.layout_prior_var,
            layout_prior_valid=self.layout_prior_valid,
            layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),
            use_layout_regularization=getattr(self.opt, "use_layout_regularization", False),
            layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.10),
            layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.05),
            layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),
            use_relation_geometry_reg=getattr(self.opt, "use_relation_geometry_reg", False),
            lambda_relation_geometry_reg=getattr(self.opt, "lambda_relation_geometry_reg", 0.0),
            relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.03),
            use_graph_law_reg=getattr(self.opt, "use_graph_law_reg", False),
            lambda_graph_law_reg=getattr(self.opt, "lambda_graph_law_reg", 0.0),
            graph_law_edge_weight=getattr(self.opt, "graph_law_edge_weight", 1.0),
            graph_law_degree_weight=getattr(self.opt, "graph_law_degree_weight", 0.5),
            graph_law_rel_weight=getattr(self.opt, "graph_law_rel_weight", 0.5),
            graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
            object_only_sanity=getattr(self.opt, "object_only_sanity", False),
        )

        obj_pred = loss_dict["pred_obj_full"]
        rel_pred = loss_dict["pred_rel_full"]
        layout_box_pred = loss_dict["layout_box_pred"]
        rel_noisy = self.build_full_relation_from_noisy_state(batch_t)

        num_to_log = min(self.opt.wandb_num_val_graphs_to_log, obj_pred.shape[0])

        draw_flowchart = getattr(self.opt, "draw_flowchart", False)
        flowchart_rankdir = getattr(self.opt, "flowchart_rankdir", "LR")
        flowchart_format = getattr(self.opt, "flowchart_format", "png")
        flowchart_show_node_ids = getattr(self.opt, "flowchart_show_node_ids", True)
        flowchart_log_individual_images = getattr(self.opt, "flowchart_log_individual_images", True)
        flowchart_out_dir = getattr(self.opt, "flowchart_out_dir", "/tmp/sg_flowcharts")

        save_layout_boxes_only = getattr(self.opt, "save_layout_boxes_only", False)
        layout_box_image_size = getattr(self.opt, "layout_box_image_size", 256)
        layout_log_individual_images = getattr(self.opt, "layout_log_individual_images", True)

        for i in range(num_to_log):
            clean_nodes, clean_triplets = decode_item(
                obj_labels=batch_t["obj_0"][i].detach().cpu(),
                rel_labels=batch_t["rel_full_0"][i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            noisy_nodes, noisy_triplets = decode_item(
                obj_labels=batch_t["obj_t"][i].detach().cpu(),
                rel_labels=rel_noisy[i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            pred_nodes, pred_triplets = decode_item(
                obj_labels=obj_pred[i].detach().cpu(),
                rel_labels=rel_pred[i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            masked_nodes = self.count_masked_nodes(
                batch_t["obj_t"][i].detach().cpu(),
                batch_t["node_mask"][i].detach().cpu(),
            )

            clean_text = format_decoded_graph(clean_nodes, clean_triplets)
            noisy_text = format_decoded_graph(noisy_nodes, noisy_triplets)
            pred_text = format_decoded_graph(pred_nodes, pred_triplets)

            pred_boxes = loss_dict.get("layout_box_pred", None)   # [B,N,4] or None
            gt_boxes = batch_t.get("boxes", None)                 # [B,N,4] or None
            box_valid_mask = batch_t.get("box_valid_mask", None)  # [B,N] or None


            diag = self.compute_example_edge_diagnostics(
                rel_gt=batch_t["rel_full_0"][i].detach().cpu(),
                rel_noisy=rel_noisy[i].detach().cpu(),
                rel_pred=rel_pred[i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
            )
            node_diag = self.compute_example_node_diagnostics(
                obj_gt=batch_t["obj_0"][i].detach().cpu(),
                obj_noisy=batch_t["obj_t"][i].detach().cpu(),
                obj_pred=obj_pred[i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
            )

            node_metrics = compute_node_accuracy_metrics(
                obj_logits=model_out["obj_logits"][i:i+1],
                obj_targets=batch_t["obj_0"][i:i+1],
                node_mask=batch_t["node_mask"][i:i+1],
                obj_corrupt_mask=batch_t["obj_corrupt_mask"][i:i+1],
                obj_mask_token_mask=batch_t["obj_mask_token_mask"][i:i+1],
                pred_obj=obj_pred[i:i+1],
            )

            clean_boxes_text = format_nodes_with_boxes(
                nodes=clean_nodes,
                boxes=gt_boxes[i].detach().cpu() if gt_boxes is not None else None,
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            pred_boxes_text = format_nodes_with_boxes(
                nodes=pred_nodes,
                boxes=pred_boxes[i].detach().cpu() if pred_boxes is not None else None,
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            diag_text = (
                f"=== DIAGNOSTICS ===\n"
                f"gt_pos_edges: {diag['num_gt_pos_edges']}\n"
                f"noisy_pos_edges: {diag['num_noisy_pos_edges']}\n"
                f"pred_pos_edges: {diag['num_pred_pos_edges']}\n"
                f"tp_edges: {diag['tp_edges']}\n"
                f"fp_edges: {diag['fp_edges']}\n"
                f"fn_edges: {diag['fn_edges']}\n"
                f"edge_precision: {diag['edge_precision']:.4f}\n"
                f"edge_recall: {diag['edge_recall']:.4f}\n"
                f"edge_f1: {diag['edge_f1']:.4f}\n"
                f"\n"
                f"num_nodes: {node_diag['num_nodes']}\n"
                f"masked_nodes: {masked_nodes}\n"
                f"noisy_diff_from_gt: {node_diag['noisy_diff_from_gt']}\n"
                f"pred_match_gt: {node_diag['pred_match_gt']}\n"
                f"pred_match_noisy: {node_diag['pred_match_noisy']}\n"
                f"node_acc_all: {node_metrics['node_acc_all'].item():.4f}\n"
                f"node_acc_corrupted: {node_metrics['node_acc_corrupted'].item():.4f}\n"
                f"node_acc_masked: {node_metrics['node_acc_masked'].item():.4f}"
            )

            joined = (
                f"{diag_text}\n\n"
                f"=== CLEAN ===\n{clean_text}\n\n"
                f"=== CLEAN BOXES ===\n{clean_boxes_text}\n\n"
                f"=== NOISY (t={int(batch_t['t'][i].item())}) ===\n{noisy_text}\n\n"
                f"=== PREDICTED x0 ===\n{pred_text}\n\n"
                f"=== PREDICTED BOXES ===\n{pred_boxes_text}"
            )

            # Keep your current text logging
            self.wandb_logger.log_text(
                key=f"val_graphs/example_{i}",
                text=joined,
                step=self.global_step,
            )

            # Optional graphviz rendering
            if draw_flowchart:
                try:
                    example_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                    )
                    ensure_dir(example_dir)

                    clean_img = render_graph_text_block_to_image(
                        graph_text=clean_text,
                        out_path_no_ext=os.path.join(example_dir, f"example_{i}_clean"),
                        title=f"Clean Graph | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    noisy_img = render_graph_text_block_to_image(
                        graph_text=noisy_text,
                        out_path_no_ext=os.path.join(example_dir, f"example_{i}_noisy"),
                        title=f"Noisy Graph | ex={i} | t={int(batch_t['t'][i].item())}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    pred_img = render_graph_text_block_to_image(
                        graph_text=pred_text,
                        out_path_no_ext=os.path.join(example_dir, f"example_{i}_pred"),
                        title=f"Predicted x0 | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    if flowchart_log_individual_images:
                        self.wandb_logger.log(
                            {
                                f"val_graphs_visual/example_{i}/clean": wandb.Image(clean_img, caption=f"clean ex={i}"),
                                f"val_graphs_visual/example_{i}/noisy": wandb.Image(noisy_img, caption=f"noisy ex={i}"),
                                f"val_graphs_visual/example_{i}/pred": wandb.Image(pred_img, caption=f"pred ex={i}"),
                                "epoch": epoch,
                            },
                            step=self.global_step,
                        )

                except Exception as e:
                    print(f"[WARN] Graphviz rendering failed for example {i}: {e}")
            
            if save_layout_boxes_only:
                try:
                    example_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                    )
                    ensure_dir(example_dir)

                    clean_layout_img = render_layout_boxes_to_image(
                        obj_class=batch_t["obj_0"][i].detach().cpu(),
                        obj_bbox=batch_t["boxes_0"][i].detach().cpu(),
                        is_valid_obj=batch_t["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(example_dir, f"example_{i}_layout_clean.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    pred_layout_img = render_layout_boxes_to_image(
                        obj_class=obj_pred[i].detach().cpu(),
                        obj_bbox=layout_box_pred[i].detach().cpu(),
                        is_valid_obj=batch_t["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(example_dir, f"example_{i}_layout_pred.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    if layout_log_individual_images:
                        self.wandb_logger.log(
                            {
                                f"val_layout_visual/example_{i}/clean_boxes": wandb.Image(
                                    clean_layout_img,
                                    caption=f"clean layout ex={i}"
                                ),
                                f"val_layout_visual/example_{i}/pred_boxes": wandb.Image(
                                    pred_layout_img,
                                    caption=f"pred layout ex={i}"
                                ),
                                "epoch": epoch,
                            },
                            step=self.global_step,
                        )

                except Exception as e:
                    print(f"[WARN] Layout box render failed for example {i}: {e}")


    @torch.no_grad()
    def log_validation_graph_table(self, epoch: int):
        if not is_main_process():
            return
        if self.val_loader is None:
            return
        if self.wandb_logger is None or not self.wandb_logger.enabled:
            return

        batch = next(iter(self.val_loader))
        batch_t = self.obj_gen.get_training_batch(batch)

        if "boxes" in batch:
            batch_t["boxes"] = batch["boxes"].to(self.device)  # [B,N,4]
            batch_t["box_valid_mask"] = batch["node_mask"].bool().to(self.device)

        self.model.eval()

        model_out = self.model(
            obj_t=batch_t["obj_t"],
            edge_t=batch_t["edge_t"],
            rel_pos_t=batch_t["rel_pos_t"],
            t=batch_t["t"],
            node_mask=batch_t["node_mask"],
            edge_mask=batch_t["edge_mask"],
        )

        pred_obj_override = None
        pred_rel_full_override = None

        if getattr(self.opt, "use_node_gibbs", False):
            pred_obj_override, edge_fixed, pred_rel_full_override, rel_pos_fixed = run_node_gibbs_sampler(
                model=self.model,
                obj_t=batch_t["obj_t"],
                edge_t=batch_t["edge_t"],
                rel_pos_t=batch_t["rel_pos_t"],
                t=batch_t["t"],
                node_mask=batch_t["node_mask"],
                edge_mask=batch_t["edge_mask"],
                no_rel_token_id=self.opt.no_rel_token_id,
                mask_rel_token_id=getattr(self.opt, "mask_rel_token_id", None),
                edge_exist_thres=self.opt.edge_exist_thres,
                num_sweeps=getattr(self.opt, "num_node_gibbs_sweeps", 1),
                sample_temp=getattr(self.opt, "node_gibbs_sample_temp", 1.0),
                use_fixed_structure=getattr(self.opt, "node_gibbs_use_fixed_structure", True),
                random_order=getattr(self.opt, "node_gibbs_random_order", True),
            )

        loss_dict = compute_structured_sg_loss(
            model_out=model_out,
            batch_t=batch_t,
            no_rel_token_id=self.opt.no_rel_token_id,
            lambda_obj=self.opt.lambda_obj,
            lambda_edge=self.opt.lambda_edge,
            lambda_rel=self.opt.lambda_rel,
            lambda_layout=getattr(self.opt, "lambda_layout", 1.0),
            edge_exist_thres=self.opt.edge_exist_thres,
            edge_pos_weight=self.opt.edge_pos_weight,
            obj_class_weights=None,
            node_loss_mode=getattr(self.opt, "node_loss_mode", "corrupted"),
            use_object_focal_loss=getattr(self.opt, "use_object_focal_loss", False),
            object_focal_gamma=getattr(self.opt, "object_focal_gamma", 2.0),
            object_focal_alpha=getattr(self.opt, "object_focal_alpha", 1.0),
            pred_obj_override=pred_obj_override,
            pred_rel_full_override=pred_rel_full_override,
            use_layout_supervision=getattr(self.opt, "use_layout_supervision", False),
            layout_loss_type=getattr(self.opt, "layout_loss_type", "smooth_l1"),
            use_layout_giou_loss=getattr(self.opt, "use_layout_giou_loss", False),
            lambda_layout_giou=getattr(self.opt, "lambda_layout_giou", 0.5),
            lambda_rel_geometry=getattr(self.opt, "lambda_rel_geometry", 0.0),
            relation_vocab=self.train_dataset.relation_vocab,
            rel_geom_margin=getattr(self.opt, "rel_geom_margin", 0.02),
            use_relation_geometry_loss=getattr(self.opt, "use_relation_geometry_loss", False),
            lambda_layout_class_prior=getattr(self.opt, "lambda_layout_class_prior", 0.0),
            layout_prior_mean=self.layout_prior_mean,
            layout_prior_var=self.layout_prior_var,
            layout_prior_valid=self.layout_prior_valid,
            layout_class_prior_eps=getattr(self.opt, "layout_class_prior_eps", 1e-4),
            use_layout_regularization=getattr(self.opt, "use_layout_regularization", False),
            layout_overlap_reg_weight=getattr(self.opt, "layout_overlap_reg_weight", 0.10),
            layout_spread_reg_weight=getattr(self.opt, "layout_spread_reg_weight", 0.05),
            layout_min_center_spread=getattr(self.opt, "layout_min_center_spread", 0.18),
            use_relation_geometry_reg=getattr(self.opt, "use_relation_geometry_reg", False),
            lambda_relation_geometry_reg=getattr(self.opt, "lambda_relation_geometry_reg", 0.0),
            relation_geometry_margin=getattr(self.opt, "relation_geometry_margin", 0.03),
            use_graph_law_reg=getattr(self.opt, "use_graph_law_reg", False),
            lambda_graph_law_reg=getattr(self.opt, "lambda_graph_law_reg", 0.0),
            graph_law_edge_weight=getattr(self.opt, "graph_law_edge_weight", 1.0),
            graph_law_degree_weight=getattr(self.opt, "graph_law_degree_weight", 0.5),
            graph_law_rel_weight=getattr(self.opt, "graph_law_rel_weight", 0.5),
            graph_law_eps=getattr(self.opt, "graph_law_eps", 1e-6),
            object_only_sanity=getattr(self.opt, "object_only_sanity", False),
        )

        obj_pred = loss_dict["pred_obj_full"]
        rel_pred = loss_dict["pred_rel_full"]
        layout_box_pred = loss_dict["pred_rel_full"]
        rel_noisy = self.build_full_relation_from_noisy_state(batch_t)

        ############### 8A.1 ##############################

        pred_boxes = loss_dict.get("layout_box_pred", None)

        # reconstruct rel_pos prediction from rel_full prediction
        pred_rel_pos = torch.where(
            rel_pred == self.opt.no_rel_token_id,
            torch.zeros_like(rel_pred),
            rel_pred - 1,
        )

        pred_edge = (rel_pred != self.opt.no_rel_token_id).long()

        if self.opt.use_reward_tilting:
            reward_terms = self.compute_reward_terms_for_state(
                obj_t=obj_pred,
                edge_t=pred_edge,
                rel_pos_t=pred_rel_pos,
                node_mask=batch_t["node_mask"],
                edge_mask=batch_t["edge_mask"],
                layout_box_pred=pred_boxes,
                box_valid_mask=batch_t.get("box_valid_mask", None),
            )

            reward_log = {}
            for k, v in reward_terms.items():
                if torch.is_tensor(v):
                    reward_log[f"val_reward/{k}"] = v.detach().float().mean().item()

            self.wandb_logger.log(
                {
                    **reward_log,
                    "epoch": epoch,
                },
                step=self.global_step,
            )
        else:
            reward_terms = None

        ###############  ##############################

        pred_boxes = loss_dict.get("layout_box_pred", None)   # [B,N,4] or None
        gt_boxes = batch_t.get("boxes", None)                 # [B,N,4] or None
        box_valid_mask = batch_t.get("box_valid_mask", None)  # [B,N] or None

        rows = []
        img_rows = []
        num_to_log = min(self.opt.wandb_num_val_graphs_to_log, obj_pred.shape[0])

        draw_flowchart = getattr(self.opt, "draw_flowchart", False)
        flowchart_rankdir = getattr(self.opt, "flowchart_rankdir", "LR")
        flowchart_format = getattr(self.opt, "flowchart_format", "png")
        flowchart_show_node_ids = getattr(self.opt, "flowchart_show_node_ids", True)
        flowchart_log_table = getattr(self.opt, "flowchart_log_table", True)
        flowchart_out_dir = getattr(self.opt, "flowchart_out_dir", "/tmp/sg_flowcharts")

        save_layout_boxes_only = getattr(self.opt, "save_layout_boxes_only", False)
        layout_box_image_size = getattr(self.opt, "layout_box_image_size", 256)
        layout_log_individual_images = getattr(self.opt, "layout_log_individual_images", True)

        for i in range(num_to_log):
            clean_nodes, clean_triplets = decode_item(
                obj_labels=batch_t["obj_0"][i].detach().cpu(),
                rel_labels=batch_t["rel_full_0"][i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            noisy_nodes, noisy_triplets = decode_item(
                obj_labels=batch_t["obj_t"][i].detach().cpu(),
                rel_labels=rel_noisy[i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            pred_nodes, pred_triplets = decode_item(
                obj_labels=obj_pred[i].detach().cpu(),
                rel_labels=rel_pred[i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            diag = self.compute_example_edge_diagnostics(
                rel_gt=batch_t["rel_full_0"][i].detach().cpu(),
                rel_noisy=rel_noisy[i].detach().cpu(),
                rel_pred=rel_pred[i].detach().cpu(),
                edge_mask=batch_t["edge_mask"][i].detach().cpu(),
            )
            node_diag = self.compute_example_node_diagnostics(
                obj_gt=batch_t["obj_0"][i].detach().cpu(),
                obj_noisy=batch_t["obj_t"][i].detach().cpu(),
                obj_pred=obj_pred[i].detach().cpu(),
                node_mask=batch_t["node_mask"][i].detach().cpu(),
            )
            node_metrics = compute_node_accuracy_metrics(
                obj_logits=model_out["obj_logits"][i:i+1],
                obj_targets=batch_t["obj_0"][i:i+1],
                node_mask=batch_t["node_mask"][i:i+1],
                obj_corrupt_mask=batch_t["obj_corrupt_mask"][i:i+1],
                obj_mask_token_mask=batch_t["obj_mask_token_mask"][i:i+1],
                pred_obj=obj_pred[i:i+1],
            )

            clean_text = format_decoded_graph(clean_nodes, clean_triplets)
            noisy_text = format_decoded_graph(noisy_nodes, noisy_triplets)
            pred_text = format_decoded_graph(pred_nodes, pred_triplets)

            clean_boxes_text = format_nodes_with_boxes(
                nodes=clean_nodes,
                boxes=gt_boxes[i].detach().cpu() if gt_boxes is not None else None,
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            pred_boxes_text = format_nodes_with_boxes(
                nodes=pred_nodes,
                boxes=pred_boxes[i].detach().cpu() if pred_boxes is not None else None,
                node_mask=batch_t["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            rows.append({
                "example_id": i,
                "timestep": int(batch_t["t"][i].item()),
                "gt_pos_edges": diag["num_gt_pos_edges"],
                "noisy_pos_edges": diag["num_noisy_pos_edges"],
                "pred_pos_edges": diag["num_pred_pos_edges"],
                "tp_edges": diag["tp_edges"],
                "fp_edges": diag["fp_edges"],
                "fn_edges": diag["fn_edges"],
                "edge_precision": round(diag["edge_precision"], 4),
                "edge_recall": round(diag["edge_recall"], 4),
                "edge_f1": round(diag["edge_f1"], 4),
                "num_nodes": node_diag["num_nodes"],
                "noisy_diff_from_gt": node_diag["noisy_diff_from_gt"],
                "pred_match_gt": node_diag["pred_match_gt"],
                "pred_match_noisy": node_diag["pred_match_noisy"],
                "node_acc_all": round(float(node_metrics["node_acc_all"].item()), 4),
                "node_acc_corrupted": round(float(node_metrics["node_acc_corrupted"].item()), 4),
                "node_acc_masked": round(float(node_metrics["node_acc_masked"].item()), 4),
                "clean_graph": format_graph_triplets_only(clean_triplets),
                "noisy_graph": format_graph_triplets_only(noisy_triplets),
                "pred_graph": format_graph_triplets_only(pred_triplets),
                "clean_boxes": clean_boxes_text,
                "pred_boxes": pred_boxes_text,
                "reward_total": (round(float(reward_terms["reward_total"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_iso": (round(float(reward_terms["reward_isolated_node"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_bidir": (round(float(reward_terms["reward_bidirectional_edge"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_dense": (round(float(reward_terms["reward_dense_graph"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_bounds": (round(float(reward_terms["reward_box_bounds"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_overlap": (round(float(reward_terms["reward_layout_overlap"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_spread": (round(float(reward_terms["reward_layout_spread"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_relgeom": (round(float(reward_terms["reward_relation_geometry"][i].item()), 4) if reward_terms is not None else 0.0),
            })

            if draw_flowchart:
                try:
                    table_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                        "table",
                    )
                    ensure_dir(table_dir)

                    clean_img = render_graph_text_block_to_image(
                        graph_text=clean_text,
                        out_path_no_ext=os.path.join(table_dir, f"example_{i}_clean"),
                        title=f"Clean Graph | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    noisy_img = render_graph_text_block_to_image(
                        graph_text=noisy_text,
                        out_path_no_ext=os.path.join(table_dir, f"example_{i}_noisy"),
                        title=f"Noisy Graph | ex={i} | t={int(batch_t['t'][i].item())}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    pred_img = render_graph_text_block_to_image(
                        graph_text=pred_text,
                        out_path_no_ext=os.path.join(table_dir, f"example_{i}_pred"),
                        title=f"Predicted x0 | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    img_rows.append({
                        "example_id": i,
                        "timestep": int(batch_t["t"][i].item()),
                        "clean_graph_img": clean_img,
                        "noisy_graph_img": noisy_img,
                        "pred_graph_img": pred_img,
                    })

                except Exception as e:
                    print(f"[WARN] Flowchart table render failed for example {i}: {e}")

            if save_layout_boxes_only:
                try:
                    table_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                        "table"
                    )
                    ensure_dir(table_dir)

                    clean_layout_img = render_layout_boxes_to_image(
                        obj_class=batch_t["obj_0"][i].detach().cpu(),
                        obj_bbox=batch_t["boxes_0"][i].detach().cpu(),
                        is_valid_obj=batch_t["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(table_dir, f"example_{i}_layout_clean.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    pred_layout_img = render_layout_boxes_to_image(
                        obj_class=obj_pred[i].detach().cpu(),
                        obj_bbox=layout_box_pred[i].detach().cpu(),
                        is_valid_obj=batch_t["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(table_dir, f"example_{i}_layout_pred.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    img_rows.append({
                        "example_id": i,
                        "timestep": int(batch_t["t"][i].item()),
                        "clean_layout_img": clean_layout_img,
                        "pred_layout_img": pred_layout_img,
                    })

                except Exception as e:
                    print(f"[WARN] Layout box table render failed for example {i}: {e}")

        # Original text-table path
        if not draw_flowchart and not save_layout_boxes_only:
            table = build_graph_comparison_table(rows)
            if table is not None:
                self.wandb_logger.log(
                    {"val/graph_comparisons": table, "epoch": epoch},
                    step=self.global_step,
                )
            return

        # Flowchart image-table path
        if (draw_flowchart or save_layout_boxes_only) and flowchart_log_table:
            img_table = build_graph_image_comparison_table(img_rows)
            if img_table is not None:
                self.wandb_logger.log(
                    {"val/graph_comparisons_flowchart": img_table, "epoch": epoch},
                    step=self.global_step,
                )

    @torch.no_grad()
    def log_full_reverse_graph_examples(self, epoch: int):
        if not is_main_process():
            return
        if self.val_loader is None:
            return

        batch = next(iter(self.val_loader))
        batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

        if "boxes" in batch:
            batch_clean["boxes"] = batch["boxes"].to(self.device)  # [B,N,4]
            batch_clean["box_valid_mask"] = batch["node_mask"].bool().to(self.device)

        batch_clean_min = {
            "obj_0": batch_clean["obj_0"],
            "edge_0": batch_clean["edge_0"],
            "boxes_0": batch_clean["boxes_0"],
            "rel_pos_0": batch_clean["rel_pos_0"],
            "node_mask": batch_clean["node_mask"],
            "edge_mask": batch_clean["edge_mask"],
        }

        self.model.eval()

        sample_out = run_full_reverse_chain(
                model=self.model,
                obj_gen=self.obj_gen,
                batch_clean=batch_clean_min,
                T=self.opt.num_diffusion_steps - 1,
                edge_exist_thres=self.opt.edge_exist_thres,
                stochastic_obj=getattr(self.opt, "full_reverse_stochastic_obj", False),
                stochastic_edge=getattr(self.opt, "full_reverse_stochastic_edge", False),
                stochastic_rel=getattr(self.opt, "full_reverse_stochastic_rel", False),
                return_trace=False,
                use_reverse_vocab_heads=getattr(self.opt, "full_reverse_use_reverse_vocab_heads", False),
                obj_temp=getattr(self.opt, "full_reverse_obj_temp", 1.0),
                rel_temp=getattr(self.opt, "full_reverse_rel_temp", 1.0),
                edge_logit_threshold=getattr(self.opt, "full_reverse_edge_logit_threshold", 0.0),
                relation_edge_logit_threshold=getattr(self.opt, "full_reverse_relation_edge_logit_threshold", 0.0),
                use_degree_pruning=getattr(self.opt, "full_reverse_use_degree_pruning", False),
                max_out_degree=getattr(self.opt, "full_reverse_max_out_degree", 0),
                max_in_degree=getattr(self.opt, "full_reverse_max_in_degree", 0),
                use_final_step_cleanup=getattr(self.opt, "full_reverse_use_final_step_cleanup", False),
                final_edge_logit_threshold=getattr(self.opt, "full_reverse_final_edge_logit_threshold", 0.5),
                final_rel_conf_threshold=getattr(self.opt, "full_reverse_final_rel_conf_threshold", 0.0),
                generic_obj_ids=getattr(self.opt, "full_reverse_generic_obj_ids", []),
                generic_attachment_rel_ids=getattr(self.opt, "full_reverse_generic_attachment_rel_ids", []),
                generic_attachment_edge_logit_threshold=getattr(self.opt,"full_reverse_generic_attachment_edge_logit_threshold",1.0,),
                reward_fn=None,
                use_reward_tilting=getattr(self.opt, "use_reward_tilting", False),
                reward_tilt_alpha=getattr(self.opt, "reward_tilt_alpha", 1.0),
                reward_tilt_temperature=getattr(self.opt, "reward_tilt_temperature", 1.0),
                reward_tilt_num_sweeps=getattr(self.opt, "reward_tilt_num_sweeps", 1),
                reward_tilt_objects=getattr(self.opt, "reward_tilt_objects", False),
                reward_tilt_edges=getattr(self.opt, "reward_tilt_edges", False),
                reward_tilt_relations=getattr(self.opt, "reward_tilt_relations", False),
                reward_tilt_use_layout=getattr(self.opt, "reward_tilt_use_layout", False),
                reward_tilt_obj_topk=getattr(self.opt, "reward_tilt_obj_topk", 5),
                reward_tilt_rel_topk=getattr(self.opt, "reward_tilt_rel_topk", 5),
                reward_weights=self.reward_weights,
                reward_tilt_edge_logit_band=getattr(self.opt, "reward_tilt_edge_logit_band", 0.75),
                reward_w_hub_degree=getattr(self.opt, "reward_w_hub_degree", 0.50),
                reward_hub_degree_threshold=getattr(self.opt, "reward_hub_degree_threshold", 4),
                reward_relation_group_pos_ids=self.reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=getattr(self.opt, "reward_tilt_relation_alpha", 0.5),
                reward_w_relation_geometry_tilt=getattr(self.opt, "reward_w_relation_geometry_tilt", 1.0),
                reward_obj_log_prior=self.reward_obj_log_prior,
                reward_tilt_object_alpha=getattr(self.opt, "reward_tilt_object_alpha", 0.25),
                reward_w_object_class_prior_tilt=getattr(self.opt, "reward_w_object_class_prior_tilt", 0.50),
                reward_w_object_relation_support_tilt=getattr(self.opt, "reward_w_object_relation_support_tilt", 0.25),
                reward_tilt_obj_logit_margin=getattr(self.opt, "reward_tilt_obj_logit_margin", 1.0),
                reward_tilt_layout_alpha=getattr(self.opt, "reward_tilt_layout_alpha", 0.25),
                reward_w_layout_overlap_tilt=getattr(self.opt, "reward_w_layout_overlap_tilt", 1.0),
                reward_w_layout_spread_tilt=getattr(self.opt, "reward_w_layout_spread_tilt", 0.5),
                reward_w_box_bounds_tilt=getattr(self.opt, "reward_w_box_bounds_tilt", 0.5),
            )

        start_state = sample_out["start_state"]

        obj_final = sample_out["obj_final"]
        edge_final = sample_out["edge_final"]
        rel_pos_final = sample_out["rel_pos_final"]
        layout_box_final = sample_out["layout_box_final"]
        rel_final_full = build_full_relation_from_structured_state(
            edge_t=edge_final,
            rel_pos_t=rel_pos_final,
            no_rel_token_id=self.opt.no_rel_token_id,
            num_rel_pos_classes=len(self.train_dataset.relation_vocab) - 1,
        )

        rel_start_full = build_full_relation_from_structured_state(
            edge_t=start_state["edge_t"],
            rel_pos_t=start_state["rel_pos_t"],
            no_rel_token_id=self.opt.no_rel_token_id,
            num_rel_pos_classes=len(self.train_dataset.relation_vocab) - 1,
        )

        num_to_log = min(self.opt.wandb_num_val_fullrev_graphs_to_log, obj_final.shape[0])

        draw_flowchart = getattr(self.opt, "draw_flowchart", False)
        flowchart_rankdir = getattr(self.opt, "flowchart_rankdir", "LR")
        flowchart_format = getattr(self.opt, "flowchart_format", "png")
        flowchart_show_node_ids = getattr(self.opt, "flowchart_show_node_ids", True)
        flowchart_log_individual_images = getattr(self.opt, "flowchart_log_individual_images", True)
        flowchart_out_dir = getattr(self.opt, "flowchart_out_dir", "/tmp/sg_flowcharts")

        save_layout_boxes_only = getattr(self.opt, "save_layout_boxes_only", False)
        layout_box_image_size = getattr(self.opt, "layout_box_image_size", 256)
        layout_log_individual_images = getattr(self.opt, "layout_log_individual_images", True)

        for i in range(num_to_log):
            clean_nodes, clean_triplets = decode_item(
                obj_labels=batch_clean["obj_0"][i].detach().cpu(),
                rel_labels=batch_clean["rel_full_0"][i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            noisy_nodes, noisy_triplets = decode_item(
                obj_labels=start_state["obj_t"][i].detach().cpu(),
                rel_labels=rel_start_full[i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            final_nodes, final_triplets = decode_item(
                obj_labels=obj_final[i].detach().cpu(),
                rel_labels=rel_final_full[i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            clean_text = format_decoded_graph(clean_nodes, clean_triplets)
            noisy_text = format_decoded_graph(noisy_nodes, noisy_triplets)
            final_text = format_decoded_graph(final_nodes, final_triplets)

            gt_boxes = batch_clean.get("boxes", None)
            box_valid_mask = batch_clean.get("box_valid_mask", None)
            final_boxes = sample_out.get("layout_box_final", None)
            clean_boxes_text = format_nodes_with_boxes(
                nodes=clean_nodes,
                boxes=gt_boxes[i].detach().cpu() if gt_boxes is not None else None,
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )
            final_boxes_text = format_nodes_with_boxes(
                nodes=final_nodes,
                boxes=final_boxes[i].detach().cpu() if final_boxes is not None else None,
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            diag = self.compute_example_edge_diagnostics(
                rel_gt=batch_clean["rel_full_0"][i].detach().cpu(),
                rel_noisy=rel_start_full[i].detach().cpu(),
                rel_pred=rel_final_full[i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
            )

            node_diag = self.compute_example_node_diagnostics(
                obj_gt=batch_clean["obj_0"][i].detach().cpu(),
                obj_noisy=start_state["obj_t"][i].detach().cpu(),
                obj_pred=obj_final[i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
            )

            node_mask_i = batch_clean["node_mask"][i].detach().cpu().bool()
            obj_gt_i = batch_clean["obj_0"][i].detach().cpu()
            obj_start_i = start_state["obj_t"][i].detach().cpu()
            obj_final_i = obj_final[i].detach().cpu()

            corrupt_mask_i = (obj_start_i != obj_gt_i) & node_mask_i
            num_nodes_i = int(node_mask_i.sum().item())
            num_corrupt_i = int(corrupt_mask_i.sum().item())

            pred_match_gt_i = int(((obj_final_i == obj_gt_i) & node_mask_i).sum().item())
            node_acc_all_i = pred_match_gt_i / max(num_nodes_i, 1)

            if num_corrupt_i > 0:
                node_acc_corr_i = int(((obj_final_i == obj_gt_i) & corrupt_mask_i).sum().item()) / num_corrupt_i
            else:
                node_acc_corr_i = 0.0
            
        

            diag_text = (
                f"=== DIAGNOSTICS ===\n"
                f"gt_pos_edges: {diag['num_gt_pos_edges']}\n"
                f"start_noisy_pos_edges: {diag['num_noisy_pos_edges']}\n"
                f"final_pos_edges: {diag['num_pred_pos_edges']}\n"
                f"tp_edges: {diag['tp_edges']}\n"
                f"fp_edges: {diag['fp_edges']}\n"
                f"fn_edges: {diag['fn_edges']}\n"
                f"edge_precision: {diag['edge_precision']:.4f}\n"
                f"edge_recall: {diag['edge_recall']:.4f}\n"
                f"edge_f1: {diag['edge_f1']:.4f}\n"
                f"\n"
                f"num_nodes: {num_nodes_i}\n"
                f"true_corrupt_nodes: {num_corrupt_i}\n"
                f"noisy_diff_from_gt: {node_diag['noisy_diff_from_gt']}\n"
                f"pred_match_gt: {node_diag['pred_match_gt']}\n"
                f"pred_match_noisy: {node_diag['pred_match_noisy']}\n"
                f"node_acc_all: {node_acc_all_i:.4f}\n"
                f"node_acc_corrupted: {node_acc_corr_i:.4f}"
            )

            joined = (
                f"{diag_text}\n\n"
                f"=== CLEAN ===\n{clean_text}\n\n"
                f"=== CLEAN BOXES ===\n{clean_boxes_text}\n\n"
                f"=== START NOISY (t={self.opt.num_diffusion_steps - 1}) ===\n{noisy_text}\n\n"
                f"=== FINAL SAMPLED x0 ===\n{final_text}\n\n"
                f"=== FINAL SAMPLED BOXES ===\n{final_boxes_text}"
            )

            self.wandb_logger.log_text(
                key=f"val_fullrev_graphs/example_{i}",
                text=joined,
                step=self.global_step,
            )

            if draw_flowchart:
                try:
                    example_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                        "fullrev",
                    )
                    ensure_dir(example_dir)

                    clean_img = render_graph_text_block_to_image(
                        graph_text=clean_text,
                        out_path_no_ext=os.path.join(example_dir, f"example_{i}_clean"),
                        title=f"Clean Graph | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    noisy_img = render_graph_text_block_to_image(
                        graph_text=noisy_text,
                        out_path_no_ext=os.path.join(example_dir, f"example_{i}_start_noisy"),
                        title=f"Start Noisy Graph | ex={i} | t={self.opt.num_diffusion_steps - 1}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    final_img = render_graph_text_block_to_image(
                        graph_text=final_text,
                        out_path_no_ext=os.path.join(example_dir, f"example_{i}_final"),
                        title=f"Final Sampled x0 | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    if flowchart_log_individual_images:
                        self.wandb_logger.log(
                            {
                                f"val_fullrev_graphs_visual/example_{i}/clean": wandb.Image(clean_img, caption=f"clean ex={i}"),
                                f"val_fullrev_graphs_visual/example_{i}/start_noisy": wandb.Image(noisy_img, caption=f"start noisy ex={i}"),
                                f"val_fullrev_graphs_visual/example_{i}/final": wandb.Image(final_img, caption=f"final ex={i}"),
                                "epoch": epoch,
                            },
                            step=self.global_step,
                        )

                except Exception as e:
                    print(f"[WARN] Full reverse Graphviz rendering failed for example {i}: {e}")

            if save_layout_boxes_only:
                try:
                    example_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                    )
                    ensure_dir(example_dir)

                    clean_layout_img = render_layout_boxes_to_image(
                        obj_class=batch_clean["obj_0"][i].detach().cpu(),
                        obj_bbox=batch_clean["boxes_0"][i].detach().cpu(),
                        is_valid_obj=batch_clean["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(example_dir, f"example_{i}_layout_clean.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    pred_layout_img = render_layout_boxes_to_image(
                        obj_class=obj_final[i].detach().cpu(),
                        obj_bbox=layout_box_final[i].detach().cpu(),
                        is_valid_obj=batch_clean["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(example_dir, f"example_{i}_layout_pred.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    if layout_log_individual_images:
                        self.wandb_logger.log(
                            {
                                f"val_fullrev_layout_visual/example_{i}/clean_boxes": wandb.Image(
                                    clean_layout_img,
                                    caption=f"clean layout ex={i}"
                                ),
                                f"val_fullrev_layout_visual/example_{i}/pred_boxes": wandb.Image(
                                    pred_layout_img,
                                    caption=f"pred layout ex={i}"
                                ),
                                "epoch": epoch,
                            },
                            step=self.global_step,
                        )

                except Exception as e:
                    print(f"[WARN] Full reverse Layout box render failed for example {i}: {e}")
    
    @torch.no_grad()
    def log_full_reverse_graph_table(self, epoch: int):
        if not is_main_process():
            return
        if self.val_loader is None:
            return
        if self.wandb_logger is None or not self.wandb_logger.enabled:
            return

        batch = next(iter(self.val_loader))
        batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

        if "boxes" in batch:
            batch_clean["boxes"] = batch["boxes"].to(self.device)  # [B,N,4]
            batch_clean["box_valid_mask"] = batch["node_mask"].bool().to(self.device)

        batch_clean_min = {
            "obj_0": batch_clean["obj_0"],
            "edge_0": batch_clean["edge_0"],
            "boxes_0": batch_clean["boxes_0"],
            "rel_pos_0": batch_clean["rel_pos_0"],
            "node_mask": batch_clean["node_mask"],
            "edge_mask": batch_clean["edge_mask"],
        }

        self.model.eval()

        sample_out = run_full_reverse_chain(
                model=self.model,
                obj_gen=self.obj_gen,
                batch_clean=batch_clean_min,
                T=self.opt.num_diffusion_steps - 1,
                edge_exist_thres=self.opt.edge_exist_thres,
                stochastic_obj=getattr(self.opt, "full_reverse_stochastic_obj", False),
                stochastic_edge=getattr(self.opt, "full_reverse_stochastic_edge", False),
                stochastic_rel=getattr(self.opt, "full_reverse_stochastic_rel", False),
                return_trace=False,
                use_reverse_vocab_heads=getattr(self.opt, "full_reverse_use_reverse_vocab_heads", False),
                obj_temp=getattr(self.opt, "full_reverse_obj_temp", 1.0),
                rel_temp=getattr(self.opt, "full_reverse_rel_temp", 1.0),
                edge_logit_threshold=getattr(self.opt, "full_reverse_edge_logit_threshold", 0.0),
                relation_edge_logit_threshold=getattr(self.opt, "full_reverse_relation_edge_logit_threshold", 0.0),
                use_degree_pruning=getattr(self.opt, "full_reverse_use_degree_pruning", False),
                max_out_degree=getattr(self.opt, "full_reverse_max_out_degree", 0),
                max_in_degree=getattr(self.opt, "full_reverse_max_in_degree", 0),
                use_final_step_cleanup=getattr(self.opt, "full_reverse_use_final_step_cleanup", False),
                final_edge_logit_threshold=getattr(self.opt, "full_reverse_final_edge_logit_threshold", 0.5),
                final_rel_conf_threshold=getattr(self.opt, "full_reverse_final_rel_conf_threshold", 0.0),
                generic_obj_ids=getattr(self.opt, "full_reverse_generic_obj_ids", []),
                generic_attachment_rel_ids=getattr(self.opt, "full_reverse_generic_attachment_rel_ids", []),
                generic_attachment_edge_logit_threshold=getattr(self.opt,"full_reverse_generic_attachment_edge_logit_threshold",1.0,),
                reward_fn=None,
                use_reward_tilting=getattr(self.opt, "use_reward_tilting", False),
                reward_tilt_alpha=getattr(self.opt, "reward_tilt_alpha", 1.0),
                reward_tilt_temperature=getattr(self.opt, "reward_tilt_temperature", 1.0),
                reward_tilt_num_sweeps=getattr(self.opt, "reward_tilt_num_sweeps", 1),
                reward_tilt_objects=getattr(self.opt, "reward_tilt_objects", False),
                reward_tilt_edges=getattr(self.opt, "reward_tilt_edges", False),
                reward_tilt_relations=getattr(self.opt, "reward_tilt_relations", False),
                reward_tilt_use_layout=getattr(self.opt, "reward_tilt_use_layout", False),
                reward_tilt_obj_topk=getattr(self.opt, "reward_tilt_obj_topk", 5),
                reward_tilt_rel_topk=getattr(self.opt, "reward_tilt_rel_topk", 5),
                reward_weights=self.reward_weights,
                reward_tilt_edge_logit_band=getattr(self.opt, "reward_tilt_edge_logit_band", 0.75),
                reward_w_hub_degree=getattr(self.opt, "reward_w_hub_degree", 0.50),
                reward_hub_degree_threshold=getattr(self.opt, "reward_hub_degree_threshold", 4),
                reward_relation_group_pos_ids=self.reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=getattr(self.opt, "reward_tilt_relation_alpha", 0.5),
                reward_w_relation_geometry_tilt=getattr(self.opt, "reward_w_relation_geometry_tilt", 1.0),
                reward_obj_log_prior=self.reward_obj_log_prior,
                reward_tilt_object_alpha=getattr(self.opt, "reward_tilt_object_alpha", 0.25),
                reward_w_object_class_prior_tilt=getattr(self.opt, "reward_w_object_class_prior_tilt", 0.50),
                reward_w_object_relation_support_tilt=getattr(self.opt, "reward_w_object_relation_support_tilt", 0.25),
                reward_tilt_obj_logit_margin=getattr(self.opt, "reward_tilt_obj_logit_margin", 1.0),
                reward_tilt_layout_alpha=getattr(self.opt, "reward_tilt_layout_alpha", 0.25),
                reward_w_layout_overlap_tilt=getattr(self.opt, "reward_w_layout_overlap_tilt", 1.0),
                reward_w_layout_spread_tilt=getattr(self.opt, "reward_w_layout_spread_tilt", 0.5),
                reward_w_box_bounds_tilt=getattr(self.opt, "reward_w_box_bounds_tilt", 0.5),
            )

        start_state = sample_out["start_state"]

        obj_final = sample_out["obj_final"]
        edge_final = sample_out["edge_final"]
        rel_pos_final = sample_out["rel_pos_final"]
        layout_box_final = sample_out["layout_box_final"]
        rel_final_full = build_full_relation_from_structured_state(
            edge_t=edge_final,
            rel_pos_t=rel_pos_final,
            no_rel_token_id=self.opt.no_rel_token_id,
            num_rel_pos_classes=len(self.train_dataset.relation_vocab) - 1,
        )

        rel_start_full = build_full_relation_from_structured_state(
            edge_t=start_state["edge_t"],
            rel_pos_t=start_state["rel_pos_t"],
            no_rel_token_id=self.opt.no_rel_token_id,
            num_rel_pos_classes=len(self.train_dataset.relation_vocab) - 1,
        )
        layout_global_reward = None
        ############### 8A.1 ##############################
        if self.opt.use_reward_tilting:
            reward_terms = self.compute_reward_terms_for_state(
                obj_t=obj_final,
                edge_t=edge_final,
                rel_pos_t=rel_pos_final,
                node_mask=batch_clean["node_mask"],
                edge_mask=batch_clean["edge_mask"],
                layout_box_pred=layout_box_final,
                box_valid_mask=batch_clean.get("box_valid_mask", None),
            )

            reward_log = {}
            for k, v in reward_terms.items():
                if torch.is_tensor(v):
                    reward_log[f"val_fullrev_reward/{k}"] = v.detach().float().mean().item()

            self.wandb_logger.log(
                {
                    **reward_log,
                    "epoch": epoch,
                },
                step=self.global_step,
            )
            layout_global_reward = sample_out.get("layout_global_reward", None)
        else:
            reward_terms = None
    
        ###############  ##############################

        final_boxes = sample_out.get("layout_box_final", None)
        gt_boxes = batch_clean.get("boxes", None)
        box_valid_mask = batch_clean.get("box_valid_mask", None)

        rows = []
        img_rows = []
        num_to_log = min(self.opt.wandb_num_val_fullrev_graphs_to_log, obj_final.shape[0])

        draw_flowchart = getattr(self.opt, "draw_flowchart", False)
        flowchart_rankdir = getattr(self.opt, "flowchart_rankdir", "LR")
        flowchart_format = getattr(self.opt, "flowchart_format", "png")
        flowchart_show_node_ids = getattr(self.opt, "flowchart_show_node_ids", True)
        flowchart_log_table = getattr(self.opt, "flowchart_log_table", True)
        flowchart_out_dir = getattr(self.opt, "flowchart_out_dir", "/tmp/sg_flowcharts")

        save_layout_boxes_only = getattr(self.opt, "save_layout_boxes_only", False)
        layout_box_image_size = getattr(self.opt, "layout_box_image_size", 256)
        layout_log_individual_images = getattr(self.opt, "layout_log_individual_images", True)

        for i in range(num_to_log):
            clean_nodes, clean_triplets = decode_item(
                obj_labels=batch_clean["obj_0"][i].detach().cpu(),
                rel_labels=batch_clean["rel_full_0"][i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            noisy_nodes, noisy_triplets = decode_item(
                obj_labels=start_state["obj_t"][i].detach().cpu(),
                rel_labels=rel_start_full[i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            final_nodes, final_triplets = decode_item(
                obj_labels=obj_final[i].detach().cpu(),
                rel_labels=rel_final_full[i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
                object_vocab=self.val_dataset.object_vocab,
                relation_vocab=self.val_dataset.relation_vocab,
                no_rel_token="__no_relation__",
                mask_obj_token_id=self.opt.mask_obj_token_id,
            )

            diag = self.compute_example_edge_diagnostics(
                rel_gt=batch_clean["rel_full_0"][i].detach().cpu(),
                rel_noisy=rel_start_full[i].detach().cpu(),
                rel_pred=rel_final_full[i].detach().cpu(),
                edge_mask=batch_clean["edge_mask"][i].detach().cpu(),
            )

            node_diag = self.compute_example_node_diagnostics(
                obj_gt=batch_clean["obj_0"][i].detach().cpu(),
                obj_noisy=start_state["obj_t"][i].detach().cpu(),
                obj_pred=obj_final[i].detach().cpu(),
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
            )

            clean_boxes_text = format_nodes_with_boxes(
                nodes=clean_nodes,
                boxes=gt_boxes[i].detach().cpu() if gt_boxes is not None else None,
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )

            final_boxes_text = format_nodes_with_boxes(
                nodes=final_nodes,
                boxes=final_boxes[i].detach().cpu() if final_boxes is not None else None,
                node_mask=batch_clean["node_mask"][i].detach().cpu(),
                box_valid_mask=box_valid_mask[i].detach().cpu() if box_valid_mask is not None else None,
                box_format="cxcywh",
            )


            node_mask_i = batch_clean["node_mask"][i].detach().cpu().bool()
            obj_gt_i = batch_clean["obj_0"][i].detach().cpu()
            obj_start_i = start_state["obj_t"][i].detach().cpu()
            obj_final_i = obj_final[i].detach().cpu()

            corrupt_mask_i = (obj_start_i != obj_gt_i) & node_mask_i
            num_nodes_i = int(node_mask_i.sum().item())
            num_corrupt_i = int(corrupt_mask_i.sum().item())

            pred_match_gt_i = int(((obj_final_i == obj_gt_i) & node_mask_i).sum().item())
            node_acc_all_i = pred_match_gt_i / max(num_nodes_i, 1)

            if num_corrupt_i > 0:
                node_acc_corr_i = int(((obj_final_i == obj_gt_i) & corrupt_mask_i).sum().item()) / num_corrupt_i
            else:
                node_acc_corr_i = 0.0

            clean_text = format_decoded_graph(clean_nodes, clean_triplets)
            noisy_text = format_decoded_graph(noisy_nodes, noisy_triplets)
            final_text = format_decoded_graph(final_nodes, final_triplets)

            rows.append({
                "example_id": i,
                "timestep": self.opt.num_diffusion_steps - 1,
                "gt_pos_edges": diag["num_gt_pos_edges"],
                "start_noisy_pos_edges": diag["num_noisy_pos_edges"],
                "final_pos_edges": diag["num_pred_pos_edges"],
                "tp_edges": diag["tp_edges"],
                "fp_edges": diag["fp_edges"],
                "fn_edges": diag["fn_edges"],
                "edge_precision": round(diag["edge_precision"], 4),
                "edge_recall": round(diag["edge_recall"], 4),
                "edge_f1": round(diag["edge_f1"], 4),
                "num_nodes": num_nodes_i,
                "true_corrupt_nodes": num_corrupt_i,
                "noisy_diff_from_gt": node_diag["noisy_diff_from_gt"],
                "pred_match_gt": node_diag["pred_match_gt"],
                "pred_match_noisy": node_diag["pred_match_noisy"],
                "node_acc_all": round(node_acc_all_i, 4),
                "node_acc_corrupted": round(node_acc_corr_i, 4),
                "clean_graph": format_graph_triplets_only(clean_triplets),
                "start_noisy_graph": format_graph_triplets_only(noisy_triplets),
                "final_graph": format_graph_triplets_only(final_triplets),
                "clean_boxes": clean_boxes_text,
                "final_boxes": final_boxes_text,
                "reward_total": (round(float(reward_terms["reward_total"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_iso": (round(float(reward_terms["reward_isolated_node"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_bidir": (round(float(reward_terms["reward_bidirectional_edge"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_dense": (round(float(reward_terms["reward_dense_graph"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_bounds": (round(float(reward_terms["reward_box_bounds"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_overlap": (round(float(reward_terms["reward_layout_overlap"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_spread": (round(float(reward_terms["reward_layout_spread"][i].item()), 4) if reward_terms is not None else 0.0),
                "r_relgeom": (round(float(reward_terms["reward_relation_geometry"][i].item()), 4) if reward_terms is not None else 0.0),
                "layout_global_reward": (round(float(layout_global_reward[i].item()), 4) if layout_global_reward is not None else None),
            })

            if draw_flowchart:
                try:
                    table_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                        "fullrev_table",
                    )
                    ensure_dir(table_dir)

                    clean_img = render_graph_text_block_to_image(
                        graph_text=clean_text,
                        out_path_no_ext=os.path.join(table_dir, f"example_{i}_clean"),
                        title=f"Clean Graph | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    noisy_img = render_graph_text_block_to_image(
                        graph_text=noisy_text,
                        out_path_no_ext=os.path.join(table_dir, f"example_{i}_start_noisy"),
                        title=f"Start Noisy Graph | ex={i} | t={self.opt.num_diffusion_steps - 1}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )
                    final_img = render_graph_text_block_to_image(
                        graph_text=final_text,
                        out_path_no_ext=os.path.join(table_dir, f"example_{i}_final"),
                        title=f"Final Sampled x0 | ex={i}",
                        rankdir=flowchart_rankdir,
                        format=flowchart_format,
                        show_node_ids=flowchart_show_node_ids,
                    )

                    img_rows.append({
                        "example_id": i,
                        "timestep": self.opt.num_diffusion_steps - 1,
                        "clean_graph_img": clean_img,
                        "noisy_graph_img": noisy_img,
                        "pred_graph_img": final_img,
                    })

                except Exception as e:
                    print(f"[WARN] Full reverse flowchart table render failed for example {i}: {e}")
            
            if save_layout_boxes_only:
                try:
                    table_dir = os.path.join(
                        flowchart_out_dir,
                        f"epoch_{epoch:04d}",
                        f"step_{self.global_step}",
                        "table"
                    )
                    ensure_dir(table_dir)

                    clean_layout_img = render_layout_boxes_to_image(
                        obj_class=batch_clean["obj_0"][i].detach().cpu(),
                        obj_bbox=batch_clean["boxes_0"][i].detach().cpu(),
                        is_valid_obj=batch_clean["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(table_dir, f"example_{i}_layout_clean.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    pred_layout_img = render_layout_boxes_to_image(
                        obj_class=obj_final[i].detach().cpu(),
                        obj_bbox=layout_box_final[i].detach().cpu(),
                        is_valid_obj=batch_clean["node_mask"][i].detach().cpu(),
                        out_path=os.path.join(table_dir, f"example_{i}_layout_pred.png"),
                        class_names=self.val_dataset.object_vocab,
                        image_size=layout_box_image_size,
                        skip_first_object=True,
                    )

                    img_rows.append({
                        "example_id": i,
                        "timestep": self.opt.num_diffusion_steps - 1,
                        "clean_layout_img": clean_layout_img,
                        "pred_layout_img": pred_layout_img,
                    })

                except Exception as e:
                    print(f"[WARN] Full reverse Layout box table render failed for example {i}: {e}")


        if not draw_flowchart and not save_layout_boxes_only:
            table = build_eval_graph_comparison_table(rows)
            if table is not None:
                self.wandb_logger.log(
                    {"val/fullrev_graph_comparisons": table, "epoch": epoch},
                    step=self.global_step,
                )
            return

        if (draw_flowchart or save_layout_boxes_only) and flowchart_log_table:
            img_table = build_graph_image_comparison_table(img_rows)
            if img_table is not None:
                self.wandb_logger.log(
                    {"val/fullrev_graph_comparisons_flowchart": img_table, "epoch": epoch},
                    step=self.global_step,
                )

    def log_epoch_metrics_table(self):
        if not is_main_process():
            return
        if self.wandb_logger is None or not self.wandb_logger.enabled:
            return

        table = build_epoch_metrics_table(self.epoch_metrics_history)
        if table is not None:
            self.wandb_logger.log(
                {"summary/epoch_metrics_table": table},
                step=self.global_step,
            )

    @torch.no_grad()
    def compute_example_edge_diagnostics(
        self,
        rel_gt: torch.Tensor,
        rel_noisy: torch.Tensor,
        rel_pred: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> dict:
        no_rel_id = self.opt.no_rel_token_id
        edge_mask = edge_mask.bool()

        gt_pos = edge_mask & (rel_gt != no_rel_id)
        noisy_pos = edge_mask & (rel_noisy != no_rel_id)
        pred_pos = edge_mask & (rel_pred != no_rel_id)

        tp = pred_pos & gt_pos
        fp = pred_pos & (~gt_pos)
        fn = (~pred_pos) & gt_pos

        num_gt_pos = int(gt_pos.sum().item())
        num_noisy_pos = int(noisy_pos.sum().item())
        num_pred_pos = int(pred_pos.sum().item())
        num_tp = int(tp.sum().item())
        num_fp = int(fp.sum().item())
        num_fn = int(fn.sum().item())

        precision = num_tp / max(num_tp + num_fp, 1)
        recall = num_tp / max(num_tp + num_fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

        return {
            "num_gt_pos_edges": num_gt_pos,
            "num_noisy_pos_edges": num_noisy_pos,
            "num_pred_pos_edges": num_pred_pos,
            "tp_edges": num_tp,
            "fp_edges": num_fp,
            "fn_edges": num_fn,
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
        }

    @torch.no_grad()
    def compute_example_node_diagnostics(
        self,
        obj_gt: torch.Tensor,
        obj_noisy: torch.Tensor,
        obj_pred: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict:
        node_mask = node_mask.bool()

        gt = obj_gt[node_mask]
        noisy = obj_noisy[node_mask]
        pred = obj_pred[node_mask]

        num_nodes = int(node_mask.sum().item())
        noisy_diff_from_gt = int((noisy != gt).sum().item())
        pred_match_gt = int((pred == gt).sum().item())
        pred_match_noisy = int((pred == noisy).sum().item())

        return {
            "num_nodes": num_nodes,
            "noisy_diff_from_gt": noisy_diff_from_gt,
            "pred_match_gt": pred_match_gt,
            "pred_match_noisy": pred_match_noisy,
        }

    @torch.no_grad()
    def count_masked_nodes(self, obj_t: torch.Tensor, node_mask: torch.Tensor) -> int:
        valid = node_mask.bool()
        return int(((obj_t == self.opt.mask_obj_token_id) & valid).sum().item())
    

    @torch.no_grad()
    def compute_object_class_counts(self):
        num_obj_classes = len(self.train_dataset.object_vocab)
        counts = torch.zeros(num_obj_classes, dtype=torch.float32)

        for i in range(len(self.train_dataset)):
            item = self.train_dataset[i]
            obj = item["obj_labels"]
            node_mask = item["node_mask"].bool()
            vals = obj[node_mask]
            counts.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))
        return counts

    @torch.no_grad()
    def build_object_frequency_buckets(self, counts: torch.Tensor):
        """
        Returns bucket id per class:
        0 = rare
        1 = medium
        2 = frequent
        """
        q1 = torch.quantile(counts, 0.33)
        q2 = torch.quantile(counts, 0.66)

        bucket = torch.zeros_like(counts, dtype=torch.long)
        bucket[counts > q1] = 1
        bucket[counts > q2] = 2
        return bucket
    
    @torch.no_grad()
    def compute_node_bucket_metrics(self, model_out: dict, batch_t: dict, loss_dict: dict):
        pred = loss_dict["pred_obj_full"]
        gt = batch_t["obj_0"]
        valid = batch_t["node_mask"].bool()
        masked = batch_t["obj_mask_token_mask"].bool() & valid

        result = {}
        for bucket_id, name in [(0, "rare"), (1, "medium"), (2, "freq")]:
            class_mask = (self.object_freq_bucket.to(gt.device)[gt] == bucket_id)
            eval_mask = masked & class_mask

            correct = ((pred == gt) & eval_mask).sum().float()
            count = eval_mask.sum().float().clamp(min=1.0)

            result[f"node_acc_masked_{name}"] = correct / count
            result[f"node_count_masked_{name}"] = count
        return result
    
    def get_teacher_force_prob(self, epoch: int) -> float:
        if not getattr(self.opt, "use_teacher_forced_structure", False):
            return 0.0

        start_p = getattr(self.opt, "teacher_force_start_prob", 1.0)
        end_p = getattr(self.opt, "teacher_force_end_prob", 0.0)
        decay_epochs = max(getattr(self.opt, "teacher_force_decay_epochs", 1), 1)

        progress = min(float(epoch) / float(decay_epochs), 1.0)
        prob = start_p + (end_p - start_p) * progress
        return float(prob)
    
    @torch.no_grad()
    def build_teacher_forced_structure_inputs(self, batch_t: dict, epoch: int):
        """
        Build edge/relation inputs that blend clean and noisy structure for training.
        Uses per-edge Bernoulli masking with epoch-dependent teacher-force probability.
        """
        tf_prob = self.get_teacher_force_prob(epoch)

        if tf_prob <= 0.0:
            return batch_t["edge_t"], batch_t["rel_pos_t"], tf_prob

        edge_t = batch_t["edge_t"]
        rel_pos_t = batch_t["rel_pos_t"]
        edge_0 = batch_t["edge_0"]
        rel_pos_0 = batch_t["rel_pos_0"]
        edge_mask = batch_t["edge_mask"].bool()

        device = edge_t.device
        use_clean = (torch.rand_like(edge_t.float()) < tf_prob) & edge_mask

        if not getattr(self.opt, "teacher_force_edge", True):
            edge_input = edge_t
        else:
            edge_input = torch.where(use_clean, edge_0, edge_t)

        if not getattr(self.opt, "teacher_force_rel", True):
            rel_input = rel_pos_t
        else:
            rel_input = torch.where(use_clean, rel_pos_0, rel_pos_t)

        # relation token is meaningful only where edge_input is on
        rel_input = torch.where(edge_input.bool(), rel_input, torch.zeros_like(rel_input))

        return edge_input.long(), rel_input.long(), tf_prob
    
    def sample_conditional_query_mask(self, batch_t: dict, epoch: int) -> torch.Tensor:
        node_mask = batch_t["node_mask"].bool()
        edge_0 = batch_t["edge_0"].bool()

        B, N = node_mask.shape
        device = node_mask.device

        if getattr(self.opt, "use_cond_query_curriculum", False):
            q = self.get_current_cond_query_prob(epoch)
        else:
            q = float(getattr(self.opt, "cond_query_prob", 0.20))

        min_q = int(getattr(self.opt, "cond_min_queries", 1))
        use_degree_bias = getattr(self.opt, "use_degree_biased_query_sampling", False)
        mix = float(getattr(self.opt, "cond_degree_bias_mix", 0.5))

        query_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            valid = node_mask[b]
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

            if valid_idx.numel() == 0:
                continue

            target_q = max(min_q, int(round(q * valid_idx.numel())))
            target_q = min(target_q, valid_idx.numel())

            if not use_degree_bias:
                chosen = valid_idx[torch.randperm(valid_idx.numel(), device=device)[:target_q]]
                query_mask[b, chosen] = True
                continue

            deg = edge_0[b].sum(dim=0) + edge_0[b].sum(dim=1)
            deg = deg.float()

            deg_valid = deg[valid_idx] + 1.0
            deg_prob = deg_valid / deg_valid.sum()
            uni_prob = torch.full_like(deg_prob, 1.0 / deg_prob.numel())

            probs = mix * deg_prob + (1.0 - mix) * uni_prob
            probs = probs / probs.sum()

            chosen_local = torch.multinomial(probs, num_samples=target_q, replacement=False)
            chosen = valid_idx[chosen_local]
            query_mask[b, chosen] = True

        return query_mask


    def build_conditional_node_inputs(self, batch_t: dict, query_mask: torch.Tensor):
        """
        Build the conditional-node training inputs:
        - query nodes use noisy obj_t
        - context nodes use clean obj_0
        - structure can be clean or noisy depending on config
        """
        obj_cond_t = torch.where(query_mask, batch_t["obj_t"], batch_t["obj_0"])

        if getattr(self.opt, "cond_use_clean_structure", True):
            edge_cond_t = batch_t["edge_0"]
            rel_pos_cond_t = batch_t["rel_pos_0"]
        else:
            edge_cond_t = batch_t["edge_t"]
            rel_pos_cond_t = batch_t["rel_pos_t"]

        return obj_cond_t, edge_cond_t, rel_pos_cond_t
    
    def get_current_cond_query_prob(self, epoch: int) -> float:
        """
        Phase 4D.1: linearly increase query probability over training.
        """
        if not getattr(self.opt, "use_cond_query_curriculum", False):
            return float(getattr(self.opt, "cond_query_prob", 0.20))

        q0 = float(getattr(self.opt, "cond_query_prob_start", 0.10))
        q1 = float(getattr(self.opt, "cond_query_prob_end", 0.35))
        T = int(getattr(self.opt, "cond_query_curriculum_epochs", 20))

        if T <= 0:
            return q1

        alpha = min(max(epoch / float(T), 0.0), 1.0)
        return q0 + alpha * (q1 - q0)
    
    @torch.no_grad()
    def evaluate_full_reverse_reconstruction(self, epoch: int, max_batches: int = 20):
        self.model.eval()

        total_tp_edges = 0.0
        total_fp_edges = 0.0
        total_fn_edges = 0.0
        total_pred_pos_edges = 0.0
        total_gt_pos_edges = 0.0
        total_rel_acc_sum = 0.0
        total_rel_acc_count = 0.0

        total_node_correct_all = 0.0
        total_node_count_all = 0.0
        total_node_correct_corrupted = 0.0
        total_node_count_corrupted = 0.0

        n_batches = 0

        iterator = self.val_loader
        if is_main_process():
            total_for_bar = len(self.val_loader) if max_batches is None else min(len(self.val_loader), max_batches)
            iterator = tqdm(self.val_loader, desc=f"FullReverseEval {epoch}", total=total_for_bar, leave=False)

        for batch in iterator:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            batch_clean_min = {
                "obj_0": batch_clean["obj_0"],
                "edge_0": batch_clean["edge_0"],
                "rel_pos_0": batch_clean["rel_pos_0"],
                "node_mask": batch_clean["node_mask"],
                "edge_mask": batch_clean["edge_mask"],
            }

            sample_out = run_full_reverse_chain(
                model=self.model,
                obj_gen=self.obj_gen,
                batch_clean=batch_clean_min,
                T=self.opt.num_diffusion_steps - 1,
                edge_exist_thres=self.opt.edge_exist_thres,
                stochastic_obj=getattr(self.opt, "full_reverse_stochastic_obj", False),
                stochastic_edge=getattr(self.opt, "full_reverse_stochastic_edge", False),
                stochastic_rel=getattr(self.opt, "full_reverse_stochastic_rel", False),
                return_trace=False,
                use_reverse_vocab_heads=getattr(self.opt, "full_reverse_use_reverse_vocab_heads", False),
                obj_temp=getattr(self.opt, "full_reverse_obj_temp", 1.0),
                rel_temp=getattr(self.opt, "full_reverse_rel_temp", 1.0),
                edge_logit_threshold=getattr(self.opt, "full_reverse_edge_logit_threshold", 0.0),
                relation_edge_logit_threshold=getattr(self.opt, "full_reverse_relation_edge_logit_threshold", 0.0),
                use_degree_pruning=getattr(self.opt, "full_reverse_use_degree_pruning", False),
                max_out_degree=getattr(self.opt, "full_reverse_max_out_degree", 0),
                max_in_degree=getattr(self.opt, "full_reverse_max_in_degree", 0),
                use_final_step_cleanup=getattr(self.opt, "full_reverse_use_final_step_cleanup", False),
                final_edge_logit_threshold=getattr(self.opt, "full_reverse_final_edge_logit_threshold", 0.5),
                final_rel_conf_threshold=getattr(self.opt, "full_reverse_final_rel_conf_threshold", 0.0),
                generic_obj_ids=getattr(self.opt, "full_reverse_generic_obj_ids", []),
                generic_attachment_rel_ids=getattr(self.opt, "full_reverse_generic_attachment_rel_ids", []),
                generic_attachment_edge_logit_threshold=getattr(self.opt,"full_reverse_generic_attachment_edge_logit_threshold",1.0,),
                reward_fn=None,
                use_reward_tilting=getattr(self.opt, "use_reward_tilting", False),
                reward_tilt_alpha=getattr(self.opt, "reward_tilt_alpha", 1.0),
                reward_tilt_temperature=getattr(self.opt, "reward_tilt_temperature", 1.0),
                reward_tilt_num_sweeps=getattr(self.opt, "reward_tilt_num_sweeps", 1),
                reward_tilt_objects=getattr(self.opt, "reward_tilt_objects", False),
                reward_tilt_edges=getattr(self.opt, "reward_tilt_edges", False),
                reward_tilt_relations=getattr(self.opt, "reward_tilt_relations", False),
                reward_tilt_use_layout=getattr(self.opt, "reward_tilt_use_layout", False),
                reward_tilt_obj_topk=getattr(self.opt, "reward_tilt_obj_topk", 5),
                reward_tilt_rel_topk=getattr(self.opt, "reward_tilt_rel_topk", 5),
                reward_weights=self.reward_weights,
                reward_tilt_edge_logit_band=getattr(self.opt, "reward_tilt_edge_logit_band", 0.75),
                reward_w_hub_degree=getattr(self.opt, "reward_w_hub_degree", 0.50),
                reward_hub_degree_threshold=getattr(self.opt, "reward_hub_degree_threshold", 4),
                reward_relation_group_pos_ids=self.reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=getattr(self.opt, "reward_tilt_relation_alpha", 0.5),
                reward_w_relation_geometry_tilt=getattr(self.opt, "reward_w_relation_geometry_tilt", 1.0),
                reward_obj_log_prior=self.reward_obj_log_prior,
                reward_tilt_object_alpha=getattr(self.opt, "reward_tilt_object_alpha", 0.25),
                reward_w_object_class_prior_tilt=getattr(self.opt, "reward_w_object_class_prior_tilt", 0.50),
                reward_w_object_relation_support_tilt=getattr(self.opt, "reward_w_object_relation_support_tilt", 0.25),
                reward_tilt_obj_logit_margin=getattr(self.opt, "reward_tilt_obj_logit_margin", 1.0),
                reward_tilt_layout_alpha=getattr(self.opt, "reward_tilt_layout_alpha", 0.25),
                reward_w_layout_overlap_tilt=getattr(self.opt, "reward_w_layout_overlap_tilt", 1.0),
                reward_w_layout_spread_tilt=getattr(self.opt, "reward_w_layout_spread_tilt", 0.5),
                reward_w_box_bounds_tilt=getattr(self.opt, "reward_w_box_bounds_tilt", 0.5),
                
            )

            metrics = compute_final_graph_metrics(
                pred_obj=sample_out["obj_final"],
                pred_edge=sample_out["edge_final"],
                pred_rel_pos=sample_out["rel_pos_final"],
                batch_clean=batch_clean_min,
                batch_start=sample_out["start_state"],
            )

            total_tp_edges += float(metrics["tp_edges"].item())
            total_fp_edges += float(metrics["fp_edges"].item())
            total_fn_edges += float(metrics["fn_edges"].item())
            total_pred_pos_edges += float(metrics["pred_num_pos_edges"].item())
            total_gt_pos_edges += float(metrics["gt_num_pos_edges"].item())

            total_rel_acc_sum += float(metrics["relation_accuracy_on_true_positive_edges"].item())
            total_rel_acc_count += 1.0

            total_node_correct_all += float((metrics["node_acc_all"] * metrics["node_count_all"]).item())
            total_node_count_all += float(metrics["node_count_all"].item())

            total_node_correct_corrupted += float((metrics["node_acc_corrupted"] * metrics["node_count_corrupted"]).item())
            total_node_count_corrupted += float(metrics["node_count_corrupted"].item())

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

        total_tp_edges = reduce_scalar_sum(total_tp_edges, self.device)
        total_fp_edges = reduce_scalar_sum(total_fp_edges, self.device)
        total_fn_edges = reduce_scalar_sum(total_fn_edges, self.device)
        total_pred_pos_edges = reduce_scalar_sum(total_pred_pos_edges, self.device)
        total_gt_pos_edges = reduce_scalar_sum(total_gt_pos_edges, self.device)
        total_rel_acc_sum = reduce_scalar_sum(total_rel_acc_sum, self.device)
        total_rel_acc_count = reduce_scalar_sum(total_rel_acc_count, self.device)
        total_node_correct_all = reduce_scalar_sum(total_node_correct_all, self.device)
        total_node_count_all = reduce_scalar_sum(total_node_count_all, self.device)
        total_node_correct_corrupted = reduce_scalar_sum(total_node_correct_corrupted, self.device)
        total_node_count_corrupted = reduce_scalar_sum(total_node_count_corrupted, self.device)

        edge_precision = total_tp_edges / max(total_tp_edges + total_fp_edges, 1e-12)
        edge_recall = total_tp_edges / max(total_tp_edges + total_fn_edges, 1e-12)
        edge_f1 = 2.0 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12)

        rel_acc_tp = total_rel_acc_sum / max(total_rel_acc_count, 1e-12)
        node_acc_all = total_node_correct_all / max(total_node_count_all, 1e-12)
        node_acc_corrupted = total_node_correct_corrupted / max(total_node_count_corrupted, 1e-12)

        return {
            "fullrev_gt_num_pos_edges": total_gt_pos_edges,
            "fullrev_pred_num_pos_edges": total_pred_pos_edges,
            "fullrev_tp_edges": total_tp_edges,
            "fullrev_fp_edges": total_fp_edges,
            "fullrev_fn_edges": total_fn_edges,
            "fullrev_edge_precision": edge_precision,
            "fullrev_edge_recall": edge_recall,
            "fullrev_edge_f1": edge_f1,
            "fullrev_rel_acc_tp": rel_acc_tp,
            "fullrev_node_acc_all": node_acc_all,
            "fullrev_node_acc_corrupted": node_acc_corrupted,
        }
    
    def slice_batch_dict(self, batch_dict: dict, keep_mask: torch.Tensor) -> dict:
        """
        Slice all batch-first tensors in a batch_dict by a boolean mask.
        Non-batched items are passed through unchanged.
        """
        out = {}
        B = keep_mask.shape[0]
        for k, v in batch_dict.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
                out[k] = v[keep_mask]
            elif isinstance(v, list):
                if len(v) == B:
                    out[k] = [vv for vv, keep in zip(v, keep_mask.tolist()) if keep]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def build_reverse_step_target_batch(self, batch_t: dict, batch_prev: dict):
        valid_prev_mask = batch_t["t"] > 0
        if not valid_prev_mask.any():
            return None, None

        bt = self.slice_batch_dict(batch_t, valid_prev_mask)
        bp = self.slice_batch_dict(batch_prev, valid_prev_mask)

        valid_obj_target_mask = (
            (bp["obj_t"] >= 0)
            & (bp["obj_t"] < len(self.train_dataset.object_vocab))
            & bt["node_mask"]
        )

        valid_edge_target_mask = bt["edge_mask"]

        valid_rel_target_mask = (
            bp["edge_t"].bool()
            & (bp["rel_pos_t"] >= 0)
            & (bp["rel_pos_t"] < (len(self.train_dataset.relation_vocab) - 1))
            & bt["edge_mask"]
        )

        obj_step_mask = (bt["obj_t"] != bp["obj_t"]) & valid_obj_target_mask
        edge_step_mask = (bt["edge_t"] != bp["edge_t"]) & valid_edge_target_mask
        rel_step_mask = (bt["rel_pos_t"] != bp["rel_pos_t"]) & valid_rel_target_mask

        obj_prev_target = torch.where(
            valid_obj_target_mask,
            bp["obj_t"],
            torch.zeros_like(bp["obj_t"]),
        )

        edge_prev_target = bp["edge_t"]

        rel_prev_target = torch.where(
            valid_rel_target_mask,
            bp["rel_pos_t"],
            torch.zeros_like(bp["rel_pos_t"]),
        )

        batch_rev = {
            "obj_0": obj_prev_target,
            "edge_0": edge_prev_target,
            "rel_pos_0": rel_prev_target,

            "obj_t": bt["obj_t"],
            "edge_t": bt["edge_t"],
            "rel_pos_t": bt["rel_pos_t"],

            "obj_corrupt_mask": obj_step_mask,
            "obj_mask_token_mask": torch.zeros_like(obj_step_mask, dtype=torch.bool),
            "edge_corrupt_mask": edge_step_mask,
            "rel_corrupt_mask": rel_step_mask,

            "gt_pos_edge_mask": valid_rel_target_mask,

            "node_mask": bt["node_mask"],
            "edge_mask": bt["edge_mask"],
            "t": bt["t"],
        }

        if "boxes" in bp:
            batch_rev["boxes"] = bp["boxes"]

        if "box_valid_mask" in bp:
            batch_rev["box_valid_mask"] = bp["box_valid_mask"]
        else:
            batch_rev["box_valid_mask"] = bt["node_mask"].bool()

        for k in ["rel_full_0"]:
            if k in bt:
                batch_rev[k] = bt[k]

        return batch_rev, valid_prev_mask

    def slice_model_out(self, model_out: dict, keep_mask: torch.Tensor) -> dict:
        """
        Slice batch-first tensors in model_out by a boolean batch mask.
        """
        out = {}
        B = keep_mask.shape[0]

        for k, v in model_out.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
                out[k] = v[keep_mask]
            else:
                out[k] = v
        return out    
    
    def build_reverse_vocab_target_batch(self, batch_t: dict, batch_prev: dict):
        """
        Build reverse-step targets for separate reverse vocab heads.
        Reverse heads predict:

        objects: clean object classes + [MASK_OBJ]
        relations: positive relation classes + [MASK_REL]
        edges: binary
        Returns:
            batch_rev, valid_prev_mask
        """
        valid_prev_mask = batch_t["t"] > 0
        if not valid_prev_mask.any():
            return None, None
        bt = self.slice_batch_dict(batch_t, valid_prev_mask)
        bp = self.slice_batch_dict(batch_prev, valid_prev_mask)
        # -------------------------
        # Object reverse targets
        # -------------------------
        obj_prev_raw = bp["obj_t"]   # may contain clean obj id or mask_obj_token_id
        obj_is_clean = (obj_prev_raw >= 0) & (obj_prev_raw < self.opt.num_obj_classes)
        obj_is_mask = (obj_prev_raw == self.opt.mask_obj_token_id)
        valid_obj_target_mask = (obj_is_clean | obj_is_mask) & bt["node_mask"]
        # map [MASK_OBJ] -> final class slot num_obj_classes
        obj_prev_target_rev = torch.where(
            obj_is_mask,
            torch.full_like(obj_prev_raw, self.opt.num_obj_classes),
            obj_prev_raw,
        )
        obj_prev_target_rev = torch.where(
            valid_obj_target_mask,
            obj_prev_target_rev,
            torch.zeros_like(obj_prev_target_rev),
        )
        # -------------------------
        # Edge reverse targets
        # -------------------------
        edge_prev_target = bp["edge_t"]
        valid_edge_target_mask = bt["edge_mask"]
        # -------------------------
        # Relation reverse targets
        # -------------------------
        rel_prev_raw = bp["rel_pos_t"]   # may contain clean rel id or mask_rel_token_id
        rel_is_clean = (rel_prev_raw >= 0) & (rel_prev_raw < self.opt.num_rel_pos_classes)
        rel_is_mask = (rel_prev_raw == self.opt.mask_rel_token_id)
        valid_rel_target_mask = (
            bp["edge_t"].bool()
            & (rel_is_clean | rel_is_mask)
            & bt["edge_mask"]
        )
        # map [MASK_REL] -> final class slot num_rel_pos_classes
        rel_prev_target_rev = torch.where(
            rel_is_mask,
            torch.full_like(rel_prev_raw, self.opt.num_rel_pos_classes),
            rel_prev_raw,
        )
        rel_prev_target_rev = torch.where(
            valid_rel_target_mask,
            rel_prev_target_rev,
            torch.zeros_like(rel_prev_target_rev),
        )
        # -------------------------
        # Only train where x_t differs from x_{t-1}
        # -------------------------
        # obj_reverse_mask = (bt["obj_t"] != bp["obj_t"]) & valid_obj_target_mask
        obj_reverse_mask = valid_obj_target_mask
        # edge_reverse_mask = (bt["edge_t"] != bp["edge_t"]) & valid_edge_target_mask
        edge_reverse_mask = valid_edge_target_mask
        # rel_reverse_mask = (bt["rel_pos_t"] != bp["rel_pos_t"]) & valid_rel_target_mask
        rel_reverse_mask = valid_rel_target_mask
        batch_rev = {
            "obj_prev_target_rev": obj_prev_target_rev,
            "edge_prev_target": edge_prev_target,
            "rel_prev_target_rev": rel_prev_target_rev,
            "obj_reverse_mask": obj_reverse_mask,
            "edge_reverse_mask": edge_reverse_mask,
            "rel_reverse_mask": rel_reverse_mask,
            "node_mask": bt["node_mask"],
            "edge_mask": bt["edge_mask"],
            "t": bt["t"],
        }
        return batch_rev, valid_prev_mask
    
    def compute_reverse_branch_loss(
        self,
        model_out: dict,
        batch_t: dict,
        batch_prev: dict,
        training: bool,
    ):
        """
        Dedicated reverse branch for Phase 5D.1.
        Builds reverse targets from (x_t, x_{t-1}) and computes reverse-vocab loss.
        """
        batch_rev, valid_prev_mask = self.build_reverse_vocab_target_batch(batch_t, batch_prev)
        if batch_rev is None:
            return None

        model_out_rev = self.slice_model_out(model_out, valid_prev_mask)

        obj_rev_class_weights = None
        if getattr(self.opt, "use_object_class_weights", False) and training:
            base_w = self.obj_class_weights
            if base_w is not None:
                obj_rev_class_weights = torch.cat(
                    [
                        base_w,
                        torch.ones(
                            1,
                            device=base_w.device,
                            dtype=base_w.dtype,
                        ),
                    ],
                    dim=0,
                )

        loss_dict_rev = compute_reverse_vocab_step_loss(
            model_out=model_out_rev,
            batch_rev=batch_rev,
            lambda_rev_obj=getattr(self.opt, "lambda_rev_obj", 1.0),
            lambda_rev_edge=getattr(self.opt, "lambda_rev_edge", 1.0),
            lambda_rev_rel=getattr(self.opt, "lambda_rev_rel", 1.0),
            edge_pos_weight=getattr(self.opt, "reverse_edge_pos_weight", self.opt.edge_pos_weight),
            obj_rev_class_weights=obj_rev_class_weights,
        )

        return {
            "loss_dict_rev": loss_dict_rev,
            "batch_rev": batch_rev,
            "valid_prev_mask": valid_prev_mask,
        }

    def compute_fullrev_composite_metrics(self, metrics: dict) -> dict:
        """
        Phase-level reverse reconstruction composites.
        These are for tracking reverse sampler quality, not final SG generator quality.
        """
        edge_f1 = float(metrics.get("fullrev_edge_f1", 0.0))
        node_corr = float(metrics.get("fullrev_node_acc_corrupted", 0.0))
        rel_acc_tp = float(metrics.get("fullrev_rel_acc_tp", 0.0))

        gt_edges = float(metrics.get("fullrev_gt_num_pos_edges", 0.0))
        pred_edges = float(metrics.get("fullrev_pred_num_pos_edges", 0.0))

        # Penalize both over-generation and under-generation of edges
        edge_count_score = math.exp(
            -abs(math.log((pred_edges + 1.0) / (gt_edges + 1.0)))
        )

        fullrev_recon_composite = (
            0.40 * edge_f1
            + 0.30 * node_corr
            + 0.15 * rel_acc_tp
            + 0.15 * edge_count_score
        )

        fullrev_structure_composite = (
            0.60 * edge_f1
            + 0.20 * rel_acc_tp
            + 0.20 * edge_count_score
        )

        return {
            "fullrev_edge_count_score": edge_count_score,
            "fullrev_recon_composite": fullrev_recon_composite,
            "fullrev_structure_composite": fullrev_structure_composite,
        }
    
    @torch.no_grad()
    def decode_structured_graph_to_scene_graph_sample(
        self,
        obj_labels: torch.Tensor,     # [N]
        edge_t: torch.Tensor,         # [N,N]
        rel_pos_t: torch.Tensor,      # [N,N]
        node_mask: torch.Tensor,      # [N]
        edge_mask: torch.Tensor,      # [N,N]
    ) -> SceneGraphSample:
        """
        Convert structured graph tensors into SceneGraphSample for graph realism metrics.
        """
        # dataset = self.val_loader.dataset
        dataset = unwrap_dataset(self.val_loader.dataset)

        rel_full = build_full_relation_from_structured_state(
            edge_t=edge_t,
            rel_pos_t=rel_pos_t,
            no_rel_token_id=0,
            num_rel_pos_classes=self.obj_gen.num_rel_pos_classes,
        )

        nodes, triplets = decode_item(
            obj_labels=obj_labels.detach().cpu(),
            rel_labels=rel_full.detach().cpu(),
            node_mask=node_mask.detach().cpu(),
            edge_mask=edge_mask.detach().cpu(),
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            no_rel_token="__no_relation__",
            mask_obj_token_id=getattr(
                self.model.module if hasattr(self.model, "module") else self.model,
                "mask_obj_token_id",
                len(dataset.object_vocab),
            ),
        )

        return SceneGraphSample(
            nodes=list(nodes),
            triplets=list(triplets),
        )
    
    @torch.no_grad()
    def decode_clean_batch_item_to_scene_graph_sample(
        self,
        batch_clean: Dict[str, Any],
        b: int,
    ) -> SceneGraphSample:
        """
        Decode one clean graph from batch_clean into SceneGraphSample.
        """
        return self.decode_structured_graph_to_scene_graph_sample(
            obj_labels=batch_clean["obj_0"][b],
            edge_t=batch_clean["edge_0"][b],
            rel_pos_t=batch_clean["rel_pos_0"][b],
            node_mask=batch_clean["node_mask"][b],
            edge_mask=batch_clean["edge_mask"][b],
        )
    
    @torch.no_grad()
    def build_reference_scene_graph_pool(
        self,
        max_graphs: int = 256,
    ):
        """
        Build a pool of real validation scene graphs for graph-generation realism metrics.
        """
        self.model.eval()

        reference_graphs = []
        collected = 0

        for batch in self.val_loader:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            B = batch_clean["obj_0"].shape[0]
            for b in range(B):
                reference_graphs.append(
                    self.decode_clean_batch_item_to_scene_graph_sample(batch_clean, b)
                )
                collected += 1
                if collected >= max_graphs:
                    return reference_graphs

        return reference_graphs
    
    @torch.no_grad()
    def build_generated_scene_graph_pool_from_full_reverse(
        self,
        max_graphs: int = 256,
    ):
        """
        Build a generated graph pool using the current full reverse sampler
        starting from corrupted validation graphs.
        This is the first realism harness for Phase 6C.1.
        """
        self.model.eval()

        generated_graphs = []
        collected = 0

        for batch in self.val_loader:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            batch_clean_min = {
                "obj_0": batch_clean["obj_0"],
                "edge_0": batch_clean["edge_0"],
                "rel_pos_0": batch_clean["rel_pos_0"],
                "node_mask": batch_clean["node_mask"],
                "edge_mask": batch_clean["edge_mask"],
            }

            sample_out = run_full_reverse_chain(
                model=self.model,
                obj_gen=self.obj_gen,
                batch_clean=batch_clean_min,
                T=self.opt.num_diffusion_steps - 1,
                edge_exist_thres=self.opt.edge_exist_thres,
                stochastic_obj=getattr(self.opt, "full_reverse_stochastic_obj", False),
                stochastic_edge=getattr(self.opt, "full_reverse_stochastic_edge", False),
                stochastic_rel=getattr(self.opt, "full_reverse_stochastic_rel", False),
                return_trace=False,
                use_reverse_vocab_heads=getattr(self.opt, "full_reverse_use_reverse_vocab_heads", False),
                obj_temp=getattr(self.opt, "full_reverse_obj_temp", 1.0),
                rel_temp=getattr(self.opt, "full_reverse_rel_temp", 1.0),
                edge_logit_threshold=getattr(self.opt, "full_reverse_edge_logit_threshold", 0.0),
                relation_edge_logit_threshold=getattr(self.opt, "full_reverse_relation_edge_logit_threshold", 0.0),
                use_degree_pruning=getattr(self.opt, "full_reverse_use_degree_pruning", False),
                max_out_degree=getattr(self.opt, "full_reverse_max_out_degree", 0),
                max_in_degree=getattr(self.opt, "full_reverse_max_in_degree", 0),
                use_final_step_cleanup=getattr(self.opt, "full_reverse_use_final_step_cleanup", False),
                final_edge_logit_threshold=getattr(self.opt, "full_reverse_final_edge_logit_threshold", 0.5),
                final_rel_conf_threshold=getattr(self.opt, "full_reverse_final_rel_conf_threshold", 0.0),
                generic_obj_ids=getattr(self.opt, "full_reverse_generic_obj_ids", []),
                generic_attachment_rel_ids=getattr(self.opt, "full_reverse_generic_attachment_rel_ids", []),
                generic_attachment_edge_logit_threshold=getattr(self.opt,"full_reverse_generic_attachment_edge_logit_threshold",1.0,),
                reward_fn=None,
                use_reward_tilting=getattr(self.opt, "use_reward_tilting", False),
                reward_tilt_alpha=getattr(self.opt, "reward_tilt_alpha", 1.0),
                reward_tilt_temperature=getattr(self.opt, "reward_tilt_temperature", 1.0),
                reward_tilt_num_sweeps=getattr(self.opt, "reward_tilt_num_sweeps", 1),
                reward_tilt_objects=getattr(self.opt, "reward_tilt_objects", False),
                reward_tilt_edges=getattr(self.opt, "reward_tilt_edges", False),
                reward_tilt_relations=getattr(self.opt, "reward_tilt_relations", False),
                reward_tilt_use_layout=getattr(self.opt, "reward_tilt_use_layout", False),
                reward_tilt_obj_topk=getattr(self.opt, "reward_tilt_obj_topk", 5),
                reward_tilt_rel_topk=getattr(self.opt, "reward_tilt_rel_topk", 5),
                reward_weights=self.reward_weights,
                reward_tilt_edge_logit_band=getattr(self.opt, "reward_tilt_edge_logit_band", 0.75),
                reward_w_hub_degree=getattr(self.opt, "reward_w_hub_degree", 0.50),
                reward_hub_degree_threshold=getattr(self.opt, "reward_hub_degree_threshold", 4),
                reward_relation_group_pos_ids=self.reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=getattr(self.opt, "reward_tilt_relation_alpha", 0.5),
                reward_w_relation_geometry_tilt=getattr(self.opt, "reward_w_relation_geometry_tilt", 1.0),
                reward_obj_log_prior=self.reward_obj_log_prior,
                reward_tilt_object_alpha=getattr(self.opt, "reward_tilt_object_alpha", 0.25),
                reward_w_object_class_prior_tilt=getattr(self.opt, "reward_w_object_class_prior_tilt", 0.50),
                reward_w_object_relation_support_tilt=getattr(self.opt, "reward_w_object_relation_support_tilt", 0.25),
                reward_tilt_obj_logit_margin=getattr(self.opt, "reward_tilt_obj_logit_margin", 1.0),
                reward_tilt_layout_alpha=getattr(self.opt, "reward_tilt_layout_alpha", 0.25),
                reward_w_layout_overlap_tilt=getattr(self.opt, "reward_w_layout_overlap_tilt", 1.0),
                reward_w_layout_spread_tilt=getattr(self.opt, "reward_w_layout_spread_tilt", 0.5),
                reward_w_box_bounds_tilt=getattr(self.opt, "reward_w_box_bounds_tilt", 0.5),
            )

            B = batch_clean["obj_0"].shape[0]
            for b in range(B):
                generated_graphs.append(
                    self.decode_structured_graph_to_scene_graph_sample(
                        obj_labels=sample_out["obj_final"][b],
                        edge_t=sample_out["edge_final"][b],
                        rel_pos_t=sample_out["rel_pos_final"][b],
                        node_mask=batch_clean["node_mask"][b],
                        edge_mask=batch_clean["edge_mask"][b],
                    )
                )
                collected += 1
                if collected >= max_graphs:
                    return generated_graphs

        return generated_graphs
    
    @torch.no_grad()
    def evaluate_graph_generation_realism(
        self,
        max_graphs: int = 256,
        include_motif_metrics: bool = False,
        topk_triplets_k: int = 50,
        out_degree_thresh: int = 3,
        in_degree_thresh: int = 3,
    ) -> Dict[str, float]:
        """
        Phase 6C.1:
        Evaluate graph-generation realism using the existing graph metrics suite.
        """
        reference_graphs = self.build_reference_scene_graph_pool(max_graphs=max_graphs)
        generated_graphs = self.build_generated_scene_graph_pool_from_full_reverse(max_graphs=max_graphs)

        metrics = evaluate_graph_generation(
            generated_graphs=generated_graphs,
            reference_graphs=reference_graphs,
            include_motif_metrics=include_motif_metrics,
            topk_triplets_k=topk_triplets_k,
            out_degree_thresh=out_degree_thresh,
            in_degree_thresh=in_degree_thresh,

        )

        return {f"graphgen_{k}": float(v) for k, v in metrics.items()}
    
    def compute_graphgen_composite(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Higher is better. MMD/TV are converted to closeness scores via (1 - value) clipping.
        """
        valid_rate = float(metrics.get("graphgen_valid_graph_rate", 0.0))
        node_mmd = float(metrics.get("graphgen_node_count_mmd", 1.0))
        edge_mmd = float(metrics.get("graphgen_edge_count_mmd", 1.0))
        obj_tv = float(metrics.get("graphgen_object_label_tv", 1.0))
        rel_tv = float(metrics.get("graphgen_relation_label_tv", 1.0))
        triplet_tv = float(metrics.get("graphgen_triplet_label_tv", 1.0))
        unique = float(metrics.get("graphgen_uniqueness_ratio", 0.0))
        trip_div = float(metrics.get("graphgen_triplet_diversity", 0.0))

        def closeness(x):
            return max(0.0, 1.0 - x)

        graphgen_composite = (
            0.20 * valid_rate
            + 0.15 * closeness(node_mmd)
            + 0.15 * closeness(edge_mmd)
            + 0.10 * closeness(obj_tv)
            + 0.10 * closeness(rel_tv)
            + 0.10 * closeness(triplet_tv)
            + 0.10 * unique
            + 0.10 * trip_div
        )

        return {"graphgen_composite": graphgen_composite}
    
    @torch.no_grad()
    def build_generated_scene_graph_pool_unconditional(
        self,
        max_graphs: int = 256,
    ):
        """
        Phase 6D.1:
        Build unconditional generated scene graphs using the terminal-noise reverse chain.
        Uses validation graph masks only as shape templates.
        """
        self.model.eval()

        generated_graphs = []
        collected = 0

        for batch in self.val_loader:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            node_mask = batch_clean["node_mask"]
            edge_mask = batch_clean["edge_mask"]

            sample_out = run_full_reverse_chain_unconditional(
                model=self.model,
                obj_gen=self.obj_gen,
                node_mask=node_mask,
                edge_mask=edge_mask,
                T=self.opt.num_diffusion_steps - 1,
                stochastic_obj=getattr(self.opt, "unconditional_stochastic_obj", False),
                stochastic_edge=getattr(self.opt, "unconditional_stochastic_edge", False),
                stochastic_rel=getattr(self.opt, "unconditional_stochastic_rel", False),
                return_trace=False,
                use_reverse_vocab_heads=getattr(self.opt, "unconditional_use_reverse_vocab_heads", False),
                obj_temp=getattr(self.opt, "unconditional_obj_temp", 1.0),
                rel_temp=getattr(self.opt, "unconditional_rel_temp", 1.0),
                edge_logit_threshold=getattr(self.opt, "unconditional_edge_logit_threshold", 0.0),
                relation_edge_logit_threshold=getattr(self.opt, "unconditional_relation_edge_logit_threshold", 0.0),
                use_degree_pruning=getattr(self.opt, "full_reverse_use_degree_pruning", False),
                max_out_degree=getattr(self.opt, "full_reverse_max_out_degree", 0),
                max_in_degree=getattr(self.opt, "full_reverse_max_in_degree", 0),
                use_final_step_cleanup=getattr(self.opt, "full_reverse_use_final_step_cleanup", False),
                final_edge_logit_threshold=getattr(self.opt, "full_reverse_final_edge_logit_threshold", 0.5),
                final_rel_conf_threshold=getattr(self.opt, "full_reverse_final_rel_conf_threshold", 0.0),
                generic_obj_ids=getattr(self.opt, "full_reverse_generic_obj_ids", []),
                generic_attachment_rel_ids=getattr(self.opt, "full_reverse_generic_attachment_rel_ids", []),
                generic_attachment_edge_logit_threshold=getattr(self.opt,"full_reverse_generic_attachment_edge_logit_threshold",1.0,),
                reward_fn=None,
                use_reward_tilting=getattr(self.opt, "use_reward_tilting", False),
                reward_tilt_alpha=getattr(self.opt, "reward_tilt_alpha", 1.0),
                reward_tilt_temperature=getattr(self.opt, "reward_tilt_temperature", 1.0),
                reward_tilt_num_sweeps=getattr(self.opt, "reward_tilt_num_sweeps", 1),
                reward_tilt_objects=getattr(self.opt, "reward_tilt_objects", False),
                reward_tilt_edges=getattr(self.opt, "reward_tilt_edges", False),
                reward_tilt_relations=getattr(self.opt, "reward_tilt_relations", False),
                reward_tilt_use_layout=getattr(self.opt, "reward_tilt_use_layout", False),
                reward_tilt_obj_topk=getattr(self.opt, "reward_tilt_obj_topk", 5),
                reward_tilt_rel_topk=getattr(self.opt, "reward_tilt_rel_topk", 5),
                reward_weights=self.reward_weights,
                reward_tilt_edge_logit_band=getattr(self.opt, "reward_tilt_edge_logit_band", 0.75),
                reward_w_hub_degree=getattr(self.opt, "reward_w_hub_degree", 0.50),
                reward_hub_degree_threshold=getattr(self.opt, "reward_hub_degree_threshold", 4),
                reward_relation_group_pos_ids=self.reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=getattr(self.opt, "reward_tilt_relation_alpha", 0.5),
                reward_w_relation_geometry_tilt=getattr(self.opt, "reward_w_relation_geometry_tilt", 1.0),
                reward_obj_log_prior=self.reward_obj_log_prior,
                reward_tilt_object_alpha=getattr(self.opt, "reward_tilt_object_alpha", 0.25),
                reward_w_object_class_prior_tilt=getattr(self.opt, "reward_w_object_class_prior_tilt", 0.50),
                reward_w_object_relation_support_tilt=getattr(self.opt, "reward_w_object_relation_support_tilt", 0.25),
                reward_tilt_obj_logit_margin=getattr(self.opt, "reward_tilt_obj_logit_margin", 1.0),
                reward_tilt_layout_alpha=getattr(self.opt, "reward_tilt_layout_alpha", 0.25),
                reward_w_layout_overlap_tilt=getattr(self.opt, "reward_w_layout_overlap_tilt", 1.0),
                reward_w_layout_spread_tilt=getattr(self.opt, "reward_w_layout_spread_tilt", 0.5),
                reward_w_box_bounds_tilt=getattr(self.opt, "reward_w_box_bounds_tilt", 0.5),
            )

            B = node_mask.shape[0]
            for b in range(B):
                generated_graphs.append(
                    self.decode_structured_graph_to_scene_graph_sample(
                        obj_labels=sample_out["obj_final"][b],
                        edge_t=sample_out["edge_final"][b],
                        rel_pos_t=sample_out["rel_pos_final"][b],
                        node_mask=node_mask[b],
                        edge_mask=edge_mask[b],
                    )
                )
                collected += 1
                if collected >= max_graphs:
                    return generated_graphs

        return generated_graphs
    
    @torch.no_grad()
    def evaluate_unconditional_graph_generation_realism(
        self,
        max_graphs: int = 256,
        include_motif_metrics: bool = False,
        topk_triplets_k: int = 50,
        out_degree_thresh: int = 3,
        in_degree_thresh: int = 3,
    ) -> Dict[str, float]:
        """
        Phase 6D.1:
        Evaluate unconditional SG generation realism against real validation graphs.
        """
        reference_graphs = self.build_reference_scene_graph_pool(max_graphs=max_graphs)
        generated_graphs = self.build_generated_scene_graph_pool_unconditional(max_graphs=max_graphs)

        metrics = evaluate_graph_generation(
            generated_graphs=generated_graphs,
            reference_graphs=reference_graphs,
            include_motif_metrics=include_motif_metrics,
            topk_triplets_k=topk_triplets_k,
            out_degree_thresh=out_degree_thresh,
            in_degree_thresh=in_degree_thresh,
        )

        return {f"uncond_graphgen_{k}": float(v) for k, v in metrics.items()}
    
    def compute_unconditional_graphgen_composite(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Higher is better.
        """
        valid_rate = float(metrics.get("uncond_graphgen_valid_graph_rate", 0.0))
        node_mmd = float(metrics.get("uncond_graphgen_node_count_mmd", 1.0))
        edge_mmd = float(metrics.get("uncond_graphgen_edge_count_mmd", 1.0))
        obj_tv = float(metrics.get("uncond_graphgen_object_label_tv", 1.0))
        rel_tv = float(metrics.get("uncond_graphgen_relation_label_tv", 1.0))
        triplet_tv = float(metrics.get("uncond_graphgen_triplet_label_tv", 1.0))
        unique = float(metrics.get("uncond_graphgen_uniqueness_ratio", 0.0))
        novel = float(metrics.get("uncond_graphgen_novelty_ratio", 0.0))
        trip_div = float(metrics.get("uncond_graphgen_triplet_diversity", 0.0))

        def closeness(x):
            return max(0.0, 1.0 - x)

        composite = (
            0.15 * valid_rate
            + 0.10 * closeness(node_mmd)
            + 0.10 * closeness(edge_mmd)
            + 0.10 * closeness(obj_tv)
            + 0.10 * closeness(rel_tv)
            + 0.15 * closeness(triplet_tv)
            + 0.10 * unique
            + 0.10 * novel
            + 0.10 * trip_div
        )

        return {"uncond_graphgen_composite": composite}
    
    @torch.no_grad()
    def estimate_empirical_unconditional_priors(self, max_batches: int = None):
        """
        Estimate empirical clean-graph marginals from the training loader.

        Object prior:
            p(obj label on valid nodes)

        Edge prior:
            p(no-edge), p(edge) over valid pairs

        Relation prior:
            p(positive relation class | positive edge), represented in the
            relation state space [0..K_rel_pos-1, mask_rel], with mask_rel mass = 0.
        """
        obj_counts = None
        edge_counts = torch.zeros(2, dtype=torch.float64, device=self.device)
        rel_counts = None

        n_batches = 0

        for batch in self.train_loader:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            obj_0 = batch_clean["obj_0"]                  # [B,N]
            edge_0 = batch_clean["edge_0"]                # [B,N,N]
            rel_pos_0 = batch_clean["rel_pos_0"]          # [B,N,N]
            node_mask = batch_clean["node_mask"].bool()   # [B,N]
            edge_mask = batch_clean["edge_mask"].bool()   # [B,N,N]

            # -----------------------------
            # Objects
            # -----------------------------
            obj_valid = obj_0[node_mask]                  # [M]
            K_obj = self.obj_gen.num_obj_classes
            if obj_counts is None:
                obj_counts = torch.zeros(K_obj, dtype=torch.float64, device=self.device)
            obj_counts += torch.bincount(obj_valid, minlength=K_obj).to(torch.float64)

            # -----------------------------
            # Edges
            # -----------------------------
            edge_valid = edge_0[edge_mask]                # [M]
            edge_counts += torch.bincount(edge_valid, minlength=2).to(torch.float64)

            # -----------------------------
            # Relations
            # Only count positive relations on positive edges
            # -----------------------------
            pos_rel_mask = edge_mask & edge_0.bool()
            rel_valid = rel_pos_0[pos_rel_mask]           # [M]
            K_rel_state = self.obj_gen.num_rel_pos_classes + 1
            if rel_counts is None:
                rel_counts = torch.zeros(K_rel_state, dtype=torch.float64, device=self.device)

            # positive relation ids occupy [0 .. K_rel_pos-1]
            if rel_valid.numel() > 0:
                rel_counts[:self.obj_gen.num_rel_pos_classes] += torch.bincount(
                    rel_valid,
                    minlength=self.obj_gen.num_rel_pos_classes,
                ).to(torch.float64)

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

        # -----------------------------
        # DDP reduce raw counts first
        # -----------------------------
        obj_counts = obj_counts if obj_counts is not None else torch.zeros(
            self.obj_gen.num_obj_classes, dtype=torch.float64, device=self.device
        )
        rel_counts = rel_counts if rel_counts is not None else torch.zeros(
            self.obj_gen.num_rel_pos_classes + 1, dtype=torch.float64, device=self.device
        )

        obj_counts = obj_counts.to(self.device)
        edge_counts = edge_counts.to(self.device)
        rel_counts = rel_counts.to(self.device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(obj_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(edge_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(rel_counts, op=dist.ReduceOp.SUM)

        # -----------------------------
        # Normalize
        # -----------------------------
        obj_prior = obj_counts / obj_counts.sum().clamp(min=1.0)

        edge_prior = edge_counts / edge_counts.sum().clamp(min=1.0)

        rel_prior = rel_counts.clone()
        rel_prior[-1] = 0.0
        rel_prior = rel_prior / rel_prior.sum().clamp(min=1.0)

        return {
            "obj_prior": obj_prior.float(),
            "edge_prior": edge_prior.float(),
            "rel_prior": rel_prior.float(),
        }
    
    @torch.no_grad()
    def _decode_structured_graph_batch_item(
        self,
        obj_t,
        edge_t,
        rel_pos_t,
        node_mask,
        edge_mask,
    ):
        # dataset = self.val_loader.dataset
        dataset = unwrap_dataset(self.val_loader.dataset)

        rel_full = build_full_relation_from_structured_state(
            edge_t=edge_t,
            rel_pos_t=rel_pos_t,
            no_rel_token_id=0,
            num_rel_pos_classes=self.obj_gen.num_rel_pos_classes,
        )

        nodes, triplets = decode_item(
            obj_labels=obj_t.detach().cpu(),
            rel_labels=rel_full.detach().cpu(),
            node_mask=node_mask.detach().cpu(),
            edge_mask=edge_mask.detach().cpu(),
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            no_rel_token="__no_relation__",
            mask_obj_token_id=getattr(self.model.module if hasattr(self.model, "module") else self.model,
                                    "mask_obj_token_id",
                                    len(dataset.object_vocab)),
        )

        return SceneGraphSample(nodes=nodes, triplets=triplets)
    
    @torch.no_grad()
    def build_reference_graph_bank(self, max_graphs: int = 512):
        ref_graphs = []
        ref_texts = []

        n_seen = 0
        for batch in self.val_loader:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            B = batch_clean["obj_0"].shape[0]
            for b in range(B):
                g = self._decode_structured_graph_batch_item(
                    obj_t=batch_clean["obj_0"][b],
                    edge_t=batch_clean["edge_0"][b],
                    rel_pos_t=batch_clean["rel_pos_0"][b],
                    node_mask=batch_clean["node_mask"][b],
                    edge_mask=batch_clean["edge_mask"][b],
                )
                ref_graphs.append(g)
                ref_texts.append(format_decoded_graph(g.nodes, g.triplets))
                n_seen += 1
                if max_graphs is not None and n_seen >= max_graphs:
                    return ref_graphs, ref_texts

        return ref_graphs, ref_texts
    
    @torch.no_grad()
    def evaluate_nearest_neighbor_analysis(
        self,
        epoch: int,
        max_generated_graphs: int = 128,
        max_reference_graphs: int = 512,
        top_k: int = 5,
        save_dir: str = "nn_analysis",
    ):
        self.model.eval()

        out_dir = os.path.join(self.opt.output_dir, save_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

        reference_graphs, reference_texts = self.build_reference_graph_bank(
            max_graphs=max_reference_graphs
        )

        generated_graphs = []
        generated_texts = []

        n_gen = 0
        for batch in self.val_loader:
            batch_clean = self.obj_gen.get_training_batch(batch, force_t=0)

            batch_clean_min = {
                "obj_0": batch_clean["obj_0"],
                "edge_0": batch_clean["edge_0"],
                "rel_pos_0": batch_clean["rel_pos_0"],
                "node_mask": batch_clean["node_mask"],
                "edge_mask": batch_clean["edge_mask"],
            }

            sample_out = run_full_reverse_chain(
                model=self.model,
                obj_gen=self.obj_gen,
                batch_clean=batch_clean_min,
                T=self.opt.num_diffusion_steps - 1,
                edge_exist_thres=self.opt.edge_exist_thres,
                stochastic_obj=getattr(self.opt, "full_reverse_stochastic_obj", False),
                stochastic_edge=getattr(self.opt, "full_reverse_stochastic_edge", False),
                stochastic_rel=getattr(self.opt, "full_reverse_stochastic_rel", False),
                return_trace=False,
                use_reverse_vocab_heads=getattr(self.opt, "full_reverse_use_reverse_vocab_heads", False),
                obj_temp=getattr(self.opt, "full_reverse_obj_temp", 1.0),
                rel_temp=getattr(self.opt, "full_reverse_rel_temp", 1.0),
                edge_logit_threshold=getattr(self.opt, "full_reverse_edge_logit_threshold", 0.0),
                relation_edge_logit_threshold=getattr(self.opt, "full_reverse_relation_edge_logit_threshold", 0.0),
                use_degree_pruning=getattr(self.opt, "full_reverse_use_degree_pruning", False),
                max_out_degree=getattr(self.opt, "full_reverse_max_out_degree", 0),
                max_in_degree=getattr(self.opt, "full_reverse_max_in_degree", 0),
                use_final_step_cleanup=getattr(self.opt, "full_reverse_use_final_step_cleanup", False),
                final_edge_logit_threshold=getattr(self.opt, "full_reverse_final_edge_logit_threshold", 0.5),
                final_rel_conf_threshold=getattr(self.opt, "full_reverse_final_rel_conf_threshold", 0.0),
                generic_obj_ids=getattr(self.opt, "full_reverse_generic_obj_ids", []),
                generic_attachment_rel_ids=getattr(self.opt, "full_reverse_generic_attachment_rel_ids", []),
                generic_attachment_edge_logit_threshold=getattr(self.opt,"full_reverse_generic_attachment_edge_logit_threshold",1.0,),
                reward_fn=None,
                use_reward_tilting=getattr(self.opt, "use_reward_tilting", False),
                reward_tilt_alpha=getattr(self.opt, "reward_tilt_alpha", 1.0),
                reward_tilt_temperature=getattr(self.opt, "reward_tilt_temperature", 1.0),
                reward_tilt_num_sweeps=getattr(self.opt, "reward_tilt_num_sweeps", 1),
                reward_tilt_objects=getattr(self.opt, "reward_tilt_objects", False),
                reward_tilt_edges=getattr(self.opt, "reward_tilt_edges", False),
                reward_tilt_relations=getattr(self.opt, "reward_tilt_relations", False),
                reward_tilt_use_layout=getattr(self.opt, "reward_tilt_use_layout", False),
                reward_tilt_obj_topk=getattr(self.opt, "reward_tilt_obj_topk", 5),
                reward_tilt_rel_topk=getattr(self.opt, "reward_tilt_rel_topk", 5),
                reward_weights=self.reward_weights,
                reward_tilt_edge_logit_band=getattr(self.opt, "reward_tilt_edge_logit_band", 0.75),
                reward_w_hub_degree=getattr(self.opt, "reward_w_hub_degree", 0.50),
                reward_hub_degree_threshold=getattr(self.opt, "reward_hub_degree_threshold", 4),
                reward_relation_group_pos_ids=self.reward_relation_group_pos_ids,
                reward_tilt_relation_alpha=getattr(self.opt, "reward_tilt_relation_alpha", 0.5),
                reward_w_relation_geometry_tilt=getattr(self.opt, "reward_w_relation_geometry_tilt", 1.0),
                reward_obj_log_prior=self.reward_obj_log_prior,
                reward_tilt_object_alpha=getattr(self.opt, "reward_tilt_object_alpha", 0.25),
                reward_w_object_class_prior_tilt=getattr(self.opt, "reward_w_object_class_prior_tilt", 0.50),
                reward_w_object_relation_support_tilt=getattr(self.opt, "reward_w_object_relation_support_tilt", 0.25),
                reward_tilt_obj_logit_margin=getattr(self.opt, "reward_tilt_obj_logit_margin", 1.0),
                reward_tilt_layout_alpha=getattr(self.opt, "reward_tilt_layout_alpha", 0.25),
                reward_w_layout_overlap_tilt=getattr(self.opt, "reward_w_layout_overlap_tilt", 1.0),
                reward_w_layout_spread_tilt=getattr(self.opt, "reward_w_layout_spread_tilt", 0.5),
                reward_w_box_bounds_tilt=getattr(self.opt, "reward_w_box_bounds_tilt", 0.5),
            )

            B = batch_clean["obj_0"].shape[0]
            for b in range(B):
                g = self._decode_structured_graph_batch_item(
                    obj_t=sample_out["obj_final"][b],
                    edge_t=sample_out["edge_final"][b],
                    rel_pos_t=sample_out["rel_pos_final"][b],
                    node_mask=batch_clean["node_mask"][b],
                    edge_mask=batch_clean["edge_mask"][b],
                )
                generated_graphs.append(g)
                generated_texts.append(format_decoded_graph(g.nodes, g.triplets))
                n_gen += 1
                if max_generated_graphs is not None and n_gen >= max_generated_graphs:
                    break

            if max_generated_graphs is not None and n_gen >= max_generated_graphs:
                break

        summary = summarize_nn_set(
            generated_graphs=generated_graphs,
            reference_graphs=reference_graphs,
            top_k=1,
        )

        records = []
        for i, g in enumerate(generated_graphs):
            ranked = rank_nearest_neighbors(
                query_graph=g,
                reference_graphs=reference_graphs,
                top_k=top_k,
            )

            best = ranked[0]
            best_idx = best["ref_index"]

            record = {
                "generated_index": i,
                "generated_graph_text": generated_texts[i],
                "nearest_neighbor_index": int(best_idx),
                "nearest_neighbor_graph_text": reference_texts[best_idx],
                "best_distance": best,
                "top_k": ranked,
            }
            records.append(record)

        with open(os.path.join(out_dir, "nearest_neighbors.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # also save a short human-readable txt file
        txt_path = os.path.join(out_dir, "nearest_neighbors_readable.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for rec in records[: min(len(records), 25)]:
                f.write("=" * 100 + "\n")
                f.write(f"Generated index: {rec['generated_index']}\n")
                f.write(f"Best NN index: {rec['nearest_neighbor_index']}\n")
                f.write(f"Best distance: {rec['best_distance']['distance_total']:.4f}\n")
                f.write(f"Obj dist: {rec['best_distance']['distance_obj']:.4f}\n")
                f.write(f"Rel dist: {rec['best_distance']['distance_rel']:.4f}\n")
                f.write(f"Triplet dist: {rec['best_distance']['distance_triplet']:.4f}\n")
                f.write(f"Node count dist: {rec['best_distance']['distance_node_count']:.4f}\n")
                f.write(f"Edge count dist: {rec['best_distance']['distance_edge_count']:.4f}\n\n")

                f.write("[GENERATED GRAPH]\n")
                f.write(rec["generated_graph_text"] + "\n\n")
                f.write("[NEAREST REAL GRAPH]\n")
                f.write(rec["nearest_neighbor_graph_text"] + "\n\n")

        return {
            "nn_mean_distance": summary["nn_mean_distance"],
            "nn_median_distance": summary["nn_median_distance"],
            "nn_mean_obj_distance": summary["nn_mean_obj_distance"],
            "nn_mean_rel_distance": summary["nn_mean_rel_distance"],
            "nn_mean_triplet_distance": summary["nn_mean_triplet_distance"],
            "nn_mean_node_count_distance": summary["nn_mean_node_count_distance"],
            "nn_mean_edge_count_distance": summary["nn_mean_edge_count_distance"],
        }
    
    def build_reverse_relation_class_weights(self) -> torch.Tensor:
        """
        Build weights for positive relation classes only (structured rel_pos state).

        We count relation frequencies only on true positive edges from the training set.
        Output shape: [num_rel_pos_classes]
        """
        import numpy as np
        import torch

        rel_labels = self.train_dataset.rel_labels      # [M, N, N], full relation labels
        edge_mask = self.train_dataset.edge_mask        # [M, N, N]

        num_rel_pos = self.obj_gen.num_rel_pos_classes
        no_rel_token_id = self.opt.no_rel_token_id

        counts = np.zeros(num_rel_pos, dtype=np.float64)

        # full relation convention assumed:
        #   0 = no relation
        #   1..num_rel_pos = positive relation ids
        for i in range(rel_labels.shape[0]):
            rel_i = rel_labels[i]
            edge_mask_i = edge_mask[i].astype(bool)

            valid_pos = edge_mask_i & (rel_i != no_rel_token_id)

            rel_vals = rel_i[valid_pos]
            if rel_vals.size == 0:
                continue

            # map full relation ids -> rel_pos ids in [0, num_rel_pos-1]
            rel_pos_vals = rel_vals - 1

            rel_pos_vals = rel_pos_vals[
                (rel_pos_vals >= 0) & (rel_pos_vals < num_rel_pos)
            ]

            if rel_pos_vals.size == 0:
                continue

            binc = np.bincount(rel_pos_vals, minlength=num_rel_pos).astype(np.float64)
            counts += binc

        counts = np.maximum(counts, 1.0)

        probs = counts / counts.sum()
        weights = (1.0 / probs) ** float(self.opt.reverse_rel_class_weight_power)

        # normalize so mean weight = 1
        weights = weights / weights.mean()

        weights = np.clip(
            weights,
            float(self.opt.reverse_rel_class_weight_min),
            float(self.opt.reverse_rel_class_weight_max),
        )

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # print("[ReverseRelWeights] min/max:",
        #     float(weights.min().item()), float(weights.max().item()))
        # print("[ReverseRelWeights] first 10:",
        #     weights[:10].detach().cpu().tolist())

        return weights
    
    def log_relation_geometry_groups(self):
        """
        Print which relation vocab entries are matched by the 7B.2 geometry groups.
        Call once after datasets are built.
        """
        rel_vocab = self.train_dataset.relation_vocab

        groups = {
            "left": ["left of", "to the left of"],
            "right": ["right of", "to the right of"],
            "above": ["above", "over", "on top of"],
            "below": ["below", "under", "beneath"],
            "inside": ["in", "inside", "inside of"],
        }

        print("\n[RelationGeometryDebug] matched relation groups:")
        for group_name, aliases in groups.items():
            matched = []
            for idx, name in enumerate(rel_vocab):
                lname = str(name).strip().lower()
                if lname in aliases:
                    matched.append((idx, name))
            print(f"  {group_name}: {matched if matched else 'NONE'}")
        
        print(self.debug_relation_vocab_keywords())
    
    def debug_relation_vocab_keywords(self):
        rel_vocab = self.train_dataset.relation_vocab
        keywords = ["left", "right", "above", "over", "under", "below", "in", "inside", "front", "behind", "on"]

        print("\n[RelationVocabKeywordDebug]")
        for kw in keywords:
            matched = [(i, r) for i, r in enumerate(rel_vocab) if kw in str(r).strip().lower()]
            print(f"  keyword='{kw}': {matched if matched else 'NONE'}")
    
    def build_layout_class_priors(self):
        """
        Build class-conditional priors over normalized boxes:
            b = [cx, cy, w, h] in [0,1]^4

        Stores:
            self.layout_prior_mean      [K_obj, 4]
            self.layout_prior_var       [K_obj, 4]
            self.layout_prior_valid     [K_obj]
            self.layout_prior_count     [K_obj]
        """
        num_obj_classes = len(self.train_dataset.object_vocab)

        sum_boxes = torch.zeros(num_obj_classes, 4, dtype=torch.float64)
        sum_sq_boxes = torch.zeros(num_obj_classes, 4, dtype=torch.float64)
        counts = torch.zeros(num_obj_classes, dtype=torch.long)

        loader = self.train_loader

        for batch in loader:
            if "boxes" not in batch:
                continue

            obj_labels = batch["obj_labels"]            # [B,N]
            boxes = batch["boxes"]                      # [B,N,4], assumed normalized cxcywh
            node_mask = batch["node_mask"].bool()       # [B,N]

            obj_flat = obj_labels[node_mask]            # [M]
            box_flat = boxes[node_mask]                 # [M,4]

            for cls_id in range(num_obj_classes):
                cls_mask = (obj_flat == cls_id)
                if cls_mask.any():
                    cls_boxes = box_flat[cls_mask].double()
                    sum_boxes[cls_id] += cls_boxes.sum(dim=0)
                    sum_sq_boxes[cls_id] += (cls_boxes ** 2).sum(dim=0)
                    counts[cls_id] += int(cls_boxes.shape[0])

        mean = torch.zeros(num_obj_classes, 4, dtype=torch.float32)
        var = torch.ones(num_obj_classes, 4, dtype=torch.float32)
        valid = torch.zeros(num_obj_classes, dtype=torch.bool)

        min_count = getattr(self.opt, "layout_class_prior_min_count", 50)
        eps = getattr(self.opt, "layout_class_prior_eps", 1e-4)
        max_var = getattr(self.opt, "layout_class_prior_max_var", 0.25)

        for cls_id in range(num_obj_classes):
            cnt = int(counts[cls_id].item())
            if cnt >= min_count:
                mu = (sum_boxes[cls_id] / cnt).float()
                ex2 = (sum_sq_boxes[cls_id] / cnt).float()
                va = (ex2 - mu ** 2).clamp(min=eps, max=max_var)

                mean[cls_id] = mu
                var[cls_id] = va
                valid[cls_id] = True

        self.layout_prior_mean = mean.to(self.device)
        self.layout_prior_var = var.to(self.device)
        self.layout_prior_valid = valid.to(self.device)
        self.layout_prior_count = counts.to(self.device)

        # if is_main_process():
        #     num_valid = int(valid.sum().item())
        #     print(f"[LayoutClassPriors] valid classes: {num_valid}/{num_obj_classes}")
        #     top_counts = torch.topk(counts.float(), k=min(10, num_obj_classes)).indices.tolist()
        #     for cls_id in top_counts:
        #         name = self.train_dataset.object_vocab[cls_id]
        #         print(
        #             f"  cls={cls_id:03d} name={name} count={int(counts[cls_id].item())} "
        #             f"mean={mean[cls_id].tolist()} var={var[cls_id].tolist()} valid={bool(valid[cls_id].item())}"
        #         )
    
    @torch.no_grad()
    def compute_reward_terms_for_state(
        self,
        obj_t: torch.Tensor,                 # [B,N]
        edge_t: torch.Tensor,                # [B,N,N]
        rel_pos_t: torch.Tensor,             # [B,N,N]
        node_mask: torch.Tensor,             # [B,N]
        edge_mask: torch.Tensor,             # [B,N,N]
        layout_box_pred: Optional[torch.Tensor] = None,   # [B,N,4]
        box_valid_mask: Optional[torch.Tensor] = None,    # [B,N]
    ):

        rel_full_t = build_full_relation_from_structured_state(
            edge_t=edge_t,
            rel_pos_t=rel_pos_t,
            no_rel_token_id=self.opt.no_rel_token_id,
            num_rel_pos_classes=len(self.train_dataset.relation_vocab) - 1,
        )

        reward_terms = compute_sg_layout_reward_terms(
            obj_t=obj_t,
            edge_t=edge_t,
            rel_full_t=rel_full_t,
            node_mask=node_mask,
            edge_mask=edge_mask,
            layout_box_pred=layout_box_pred,
            box_valid_mask=box_valid_mask,
            relation_group_ids=self.reward_relation_group_ids,
        )

        reward_terms = combine_reward_terms(
            reward_terms=reward_terms,
            reward_weights=self.reward_weights,
        )

        return reward_terms
    
    @torch.no_grad()
    def reward_fn_for_sampling(
        self,
        obj_t: torch.Tensor,
        edge_t: torch.Tensor,
        rel_pos_t: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        ):
        """
        Reward callback used by 8A.3 sampling.

        For 8A.3, keep this graph-only.
        Layout-conditioned reward comes in 8A.4.
        """
        layout_box_pred = None
        box_valid_mask = None

        if getattr(self.opt, "reward_tilt_use_layout", False):
            # Keep disabled for 8A.3 unless you intentionally want expensive model calls.
            raise NotImplementedError(
                "reward_tilt_use_layout=True is reserved for 8A.4."
            )

        reward_terms = self.compute_reward_terms_for_state(
            obj_t=obj_t,
            edge_t=edge_t,
            rel_pos_t=rel_pos_t,
            node_mask=node_mask,
            edge_mask=edge_mask,
            layout_box_pred=layout_box_pred,
            box_valid_mask=box_valid_mask,
        )

        return reward_terms["reward_total"]

    @torch.no_grad()
    def build_reward_object_log_prior(self):
        obj_labels = torch.as_tensor(
            self.train_dataset.obj_labels,
            dtype=torch.long,
        )

        node_mask = torch.as_tensor(
            self.train_dataset.node_mask,
            dtype=torch.bool,
        )

        num_obj_classes = len(self.train_dataset.object_vocab)
        counts = torch.ones(num_obj_classes, dtype=torch.float32)  # smoothing

        valid_obj = obj_labels[node_mask]
        valid_obj = valid_obj[
            (valid_obj >= 0) & (valid_obj < num_obj_classes)
        ]

        counts.scatter_add_(
            dim=0,
            index=valid_obj.cpu(),
            src=torch.ones_like(valid_obj.cpu(), dtype=torch.float32),
        )

        prior = counts / counts.sum().clamp(min=1.0)
        log_prior = torch.log(prior.clamp(min=1e-12))

        return log_prior
    
    # -------------------------
    # curriculum force_t based on best val_node_corr
    # -------------------------
    def get_curriculum_t(self):
        # default stage thresholds
        s1 = getattr(self.opt, "curriculum_stage1_node_corr", 0.80)
        s2 = getattr(self.opt, "curriculum_stage2_node_corr", 0.80)

        best_t10 = getattr(self, "best_val_node_corr_t10", 0.0)
        best_t25 = getattr(self, "best_val_node_corr_t25", 0.0)

        if best_t10 < s1:
            return 10
        elif best_t25 < s2:
            return 25
        else:
            return 49

    def update_curriculum_t(self, epoch):
        curr_t = self.current_curriculum_t
        # self.current_curriculum_t = 25
        best_corr = self.best_val_node_corr_by_t.get(curr_t, 0.0)

        if curr_t == 10 and (best_corr >= 0.35 or epoch >= 50):
            self.current_curriculum_t = 25

        elif curr_t == 25 and (best_corr >= 0.40 or epoch >= 100):
            self.current_curriculum_t = 35

    