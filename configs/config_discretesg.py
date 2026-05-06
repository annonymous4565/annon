from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DiscreteSGConfig:
    # paths
    h5_path: str = "./data/visual_genome/VG-SGG.h5"
    dict_json_path: str = "./data/visual_genome/VG-SGG-dicts.json"
    image_data_json_path: Optional[str] = "./data/visual_genome/image_data.json"
    output_dir: str = "./data/visual_genome/processed"

    # representation
    n_max: int = 20
    box_scale_key: str = "boxes_1024"
    object_selection: str = "degree"   # ["degree", "original", "degree_spread", "degree_spread_softcap", "anchor_degree_spread", "anchor_degree_spread_softcap"]
    degree_weight: float = 1.0
    area_weight: float = 0.5
    center_weight: float = 0.25
    spread_weight: float = 1.0
    repeat_penalty_weight: float = 0.75
    repeat_penalty_type: str ="log"
    enable_graph_filtering: bool = True
    filter_min_nodes_for_low_rel: int = 10
    filter_max_pos_edges_if_dense_nodes: int = 2
    filter_max_repeated_labels: Optional[int] = 5
    filter_max_single_label_count: Optional[int] = 4
    filter_max_mean_box_iou: Optional[float] = 0.20
    filter_max_isolated_frac: Optional[float] = 0.35

    # filtering
    filter_empty_rels: bool = False
    filter_non_overlap: bool = False
    drop_background: bool = True
    keep_first_relation: bool = True

    # split/export
    export_split: str = "test"     # ["trainval", "test"]
    limit: Optional[int] = None
    train_filename: str = "vg_sg_degree_filter_new_train_nmax20_seed42.npz"
    val_filename: str = "vg_sg_degree_filter_new_val_nmax20_seed42.npz"
    # processed_npz_path: str = './data/visual_genome/processed/vg_sg_trainval_nmax20.npz'
    processed_npz_path: str = './data/visual_genome/processed/vg_sg_degree_filter_new_trainval_nmax20.npz'

    train_npz_path: str = './data/visual_genome/processed/vg_sg_degree_filter_new_train_nmax20_seed42.npz'
    val_npz_path: str = './data/visual_genome/processed/vg_sg_degree_filter_new_val_nmax20_seed42.npz'
    # train_npz_path: str = './data/visual_genome/processed/vg_sg_train_nmax20_seed42.npz'
    # val_npz_path: str = './data/visual_genome/processed/vg_sg_val_nmax20_seed42.npz'

    val_ratio: float = 0.1
    split_seed: int = 42
    shuffle: bool = True
    train_mode: bool = True

    # vocab / special tokens
    pad_token: str = "__pad__"
    no_rel_token: str = "__no_relation__"
    num_obj_classes: int = 50
    num_rel_pos_classes: int = 150

    # debugging / inspection
    inspect_index: Optional[int] = None
    inspect_nth: Optional[int] = None

    # random / reproducibility
    seed: int = 0

#  DiscreteNoiseScheduleConfig(DiscreteSGConfig):
    obj_beta_start: float = 1e-3
    obj_beta_end: float = 1e-1
    rel_beta_start: float = 5e-4
    rel_beta_end: float = 1e-2


#  DiscreteTransitionConfig(DiscreteSGConfig):
    pad_token_id: int = 0
    no_rel_token_id: int = 0
    make_pad_absorbing: bool = True


# @dataclass
# class DiscreteSGObjectiveConfig:

    obj_beta_start: float = 1e-3
    obj_beta_end: float = 5e-2


    sample_timesteps_uniformly: bool = True

# @dataclass
# class StickyRelationConfig:
    no_rel_leak_scale: float = 0.01   # epsilon_t = scale * beta_t
    use_sticky_no_rel: bool = True
    use_empirical_obj_prior: bool = True
    use_empirical_rel_prior: bool = False

    # model
    d_model: int = 256
    num_mp_layers: int = 4
    dropout: float = 0.1
    num_heads: int = 8
    num_layers: int = 8

    # loss
    lambda_obj: float = 1.0
    lambda_rel: float = 0.0
    lambda_edge: float = 0.0

    # relation weighting
    rel_weighting_strategy: str = "effective_num"   # ["none", "simple", "inverse_freq", "effective_num"]
    no_rel_loss_weight: float = 0.1

    rel_weight_alpha: float = 0.5
    rel_weight_min: float = 0.25
    rel_weight_max: float = 5.0
    rel_effective_num_beta: float = 0.99

    # negative edge subsampling
    use_negative_edge_sampling: bool = True
    neg_edge_sample_strategy: str = "ratio"   # ["ratio", "prob"]
    neg_pos_ratio: float = 3.0                # if strategy == "ratio"
    neg_keep_prob: float = 0.1                # if strategy == "prob"
    use_negative_edge_sampling_in_val: bool = False


    #edge
    edge_exist_thres: float = 0.35
    edge_pos_weight: float = 1.0

    #node-refine
    lambda_obj_stage1: float = 0.3
    lambda_edge_stage1: float = 0.3
    lambda_rel_stage1: float = 0.3

    #plausibility
    use_plausibility_prior: bool = False
    edge_prior_strength: float = 0.15
    rel_prior_strength: float = 0.15
    prior_use_gt_objects_in_train: bool = False
    prior_smoothing_eps: float = 1.0

    #node-diffusion-new
    use_masked_node_diffusion: bool = True
    node_mask_corruption_schedule: str = "linear"
    node_random_corruption_prob: float = 0.2   # keep 0.0 for first run
    mask_obj_token_id: int = -1  # set at runtime to num_obj_classes


    num_diffusion_steps: int = 50
    batch_size: int = 64
    eval_batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    val_every_epoch: int = 1
    save_every_epoch: int = 1
    save_latest_every_epoch: bool = True
    checkpoint_dir: str = "checkpoints/phase1"
    ckpt: str = './checkpoints/phase5a-run5-full-rev-sampling/simple_factorized/best_total.pt'

    # wandb
    use_wandb: bool = True
    wandb_project: str = "DiscreteSG"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_mode: str = "online"   # "online", "offline", "disabled"
    wandb_log_every_steps: int = 50
    wandb_num_val_graphs_to_log: int = 3
    wandb_num_val_fullrev_graphs_to_log: int = 8
    wandb_num_val_fullrev_graphs_to_log: int = 2
    wandb_num_epochs_fullrev: int = 2
    wandb_num_epochs_val: int = 2

    draw_flowchart: bool = False
    flowchart_rankdir: str = "TB"          # "LR" or "TB"
    flowchart_format: str = "png"          # png / svg / pdf
    flowchart_show_node_ids: bool = True
    flowchart_log_individual_images: bool = True
    flowchart_log_table: bool = False
    flowchart_out_dir: str = "./visualization/sg_flowcharts"



    # structured diffusion
    use_structured_diffusion: bool = True

    # node corruption
    node_mask_ratio: float = 1.0
    node_rand_ratio: float = 0.0

    # edge corruption
    edge_pos_flip_max: float = 0.5
    edge_neg_flip_max: float = 0.05

    # relation corruption
    rel_mask_ratio: float = 0.8
    rel_rand_ratio: float = 0.2

    # structured model
    structured_num_layers: int = 4

    # structured loss
    node_loss_on_corrupted_only: bool = True
    node_loss_mode: str = "masked_only"   # options: "corrupted", "masked_only"
    use_object_class_weights: bool = False
    obj_effective_num_beta: float = 0.999
    obj_weight_min: float = 0.25
    obj_weight_max: float = 5.0
    use_object_focal_loss: bool = False
    object_focal_gamma: float = 2.0
    object_focal_alpha: float = 1.0

    # node corruption
    node_corrupt_intensity: float = 1.0

    # teacher-forced structure blending
    use_teacher_forced_structure: bool = True
    teacher_force_edge: bool = True
    teacher_force_rel: bool = True

    # linear decay schedule over training
    teacher_force_start_prob: float = 1.0
    teacher_force_end_prob: float = 0.0
    teacher_force_decay_epochs: int = 30

    # Phase 3A: semantic node corruption
    use_semantic_node_corruption: bool = True
    node_semantic_temp: float = 0.1
    node_semantic_self_bias: float = 0.0
    node_semantic_topk: int = 20   # or None if you prefer dense matrix

    # Phase 3B: split reverse
    log_node_update_stats: bool = True

    # Phase 3E.1: true split posterior node sweeps
    use_node_gibbs: bool = False
    num_node_gibbs_sweeps: int = 3
    node_gibbs_sample_temp: float = 1.0
    node_gibbs_use_fixed_structure: bool = True
    node_gibbs_random_order: bool = True

    # Phase 4
    # Phase 4B: conditional node objective
    use_conditional_node_objective: bool = False
    lambda_cond_node: float = 1.0

    cond_query_prob: float = 0.20
    cond_min_queries: int = 1

    # structure used in conditional pass
    cond_use_clean_structure: bool = False

    # logging
    log_conditional_node_metrics: bool = True

    # -------------------------
    # Phase 4D.1: query curriculum
    # -------------------------
    use_cond_query_curriculum: bool = False
    cond_query_prob_start: float = 0.10
    cond_query_prob_end: float = 0.30
    cond_query_curriculum_epochs: int = 20

    # -------------------------
    # Phase 4D.2: degree-biased query selection
    # -------------------------
    use_degree_biased_query_sampling: bool = False
    cond_degree_bias_mix: float = 0.5   # 0 = uniform, 1 = pure degree-proportional

    # Phase 4E.1
    use_relation_bucket_node_conditioning: bool = False
    num_relation_buckets: int = 12


    # Phase 4F.1
    use_refinement_pass: bool = False
    lambda_refine: float = 0.5
    refine_use_pred_structure: bool = True
    refine_detach_first_pass: bool = True

    # Phase 4F.2
    use_residual_refine_weighting: bool = False

    refine_obj_wrong_weight: float = 3.0
    refine_obj_base_weight: float = 0.25

    # optional later, keep off for first test
    refine_edge_wrong_weight: float = 1.0
    refine_edge_base_weight: float = 1.0
    refine_rel_wrong_weight: float = 1.0
    refine_rel_base_weight: float = 1.0

    # Phase 5A
    full_reverse_eval_every: int = 9999
    full_reverse_eval_max_batches: int = 3

    # Phase 5B
    full_reverse_stochastic_obj: bool = False
    full_reverse_stochastic_edge: bool = False
    full_reverse_stochastic_rel: bool = False

    # Phase 5C-lite: sampled-state auxiliary training
    use_sampled_state_training: bool = False
    lambda_sampled_state: float = 0.25

    # Phase 5C-full-2: reverse-step-aligned training
    use_reverse_step_training: bool = False
    lambda_reverse_step: float = 0.5

    # Phase 5C-full-3: separate reverse-step heads
    use_reverse_vocab_heads: bool = False
    use_reverse_vocab_step_training: bool = False
    lambda_reverse_vocab_step: float = 0.5

    # Phase 5D.1: dedicated reverse branch
    use_dedicated_reverse_branch: bool = False

    # Phase 5D.2: use reverse heads directly in full reverse sampler
    full_reverse_use_reverse_vocab_heads: bool = False

    # optional temperatures for stochastic sampling
    full_reverse_obj_temp: float = 1.0
    full_reverse_rel_temp: float = 1.0

    # Phase 5D.4
    reverse_edge_pos_weight: float = 10.0

    # Phase 5D.5
    full_reverse_edge_logit_threshold: float = 0.5

    # Phase 6A.1: relation-confidence gating
    full_reverse_relation_edge_logit_threshold: float = 0.0

    # Phase 6B.1: separate reverse modality weights
    lambda_rev_obj: float = 1.0
    lambda_rev_edge: float = 1.25
    lambda_rev_rel: float = 1.25

    # Phase 6C.1: graph realism evaluation
    run_graph_generation_eval: bool = False
    graph_generation_eval_max_graphs: int = 256

    # Phase 6D.1: unconditional graph generation eval
    run_unconditional_graph_generation_eval: bool = False
    unconditional_graph_generation_eval_max_graphs: int = 256

    # unconditional sampler settings
    unconditional_use_reverse_vocab_heads: bool = False
    unconditional_stochastic_obj: bool = False
    unconditional_stochastic_edge: bool = False
    unconditional_stochastic_rel: bool = False
    unconditional_obj_temp: float = 1.0
    unconditional_rel_temp: float = 1.0
    unconditional_edge_logit_threshold: float = 0.5
    unconditional_relation_edge_logit_threshold: float = 0.0

    # Phase 6D.1b
    use_empirical_unconditional_priors: bool = False
    empirical_prior_max_batches: Optional[int] = None

    # Phase 6A.2
    full_reverse_use_degree_pruning: bool = False
    full_reverse_max_out_degree: int = 4
    full_reverse_max_in_degree: int = 4

    # 6A.3 final-step cleanup
    full_reverse_use_final_step_cleanup: bool = False
    full_reverse_final_edge_logit_threshold: float = 0.5
    full_reverse_final_rel_conf_threshold: float = 0.20

    # object ids that are often generic attachment targets
    full_reverse_generic_obj_ids: List[int] = field(default_factory=list)

    # relation ids that are often attachment-like
    full_reverse_generic_attachment_rel_ids: List[int] = field(default_factory=list)

    # stronger threshold for generic attachment edges
    full_reverse_generic_attachment_edge_logit_threshold: float = 1.0

    # 6D.2 Nearest Neighbor qualitative analysis
    run_nn_eval: bool = False
    nn_eval_num_generated: int = 128
    nn_eval_num_reference: int = 512
    nn_eval_top_k: int = 5

    # Phase 6B.2: reverse rare-relation weighting
    use_reverse_rel_class_weights: bool = False
    reverse_rel_class_weight_power: float = 0.5
    reverse_rel_class_weight_min: float = 1.0
    reverse_rel_class_weight_max: float = 4.0

    # Phase 6C.2: Motif metrics
    include_motif_metrics: bool = True
    graph_metrics_topk_triplets_k: int = 50
    graph_metrics_hub_out_thresh: int = 3
    graph_metrics_hub_in_thresh: int = 3

    # Phase 7A.1-3
    use_layout_head: bool = False
    layout_hidden_dim: int = 256

    use_layout_supervision: bool = False
    lambda_layout: float = 1.0
    layout_loss_type: str = "smooth_l1"   # choices: "l1", "smooth_l1"

    # Phase 7A.4

    # Phase 7B.1
    use_layout_giou_loss: bool = False
    lambda_layout_giou: float = 0.1

    # Phase 7B.2
    use_relation_geometry_loss: bool = False
    lambda_rel_geometry: float = 0.05
    rel_geom_margin: float = 0.02

    # -------- Phase 7B.3: object-type geometric priors --------
    use_layout_class_priors: bool = False
    lambda_layout_class_prior: float = 0.05
    layout_class_prior_min_count: int = 50
    layout_class_prior_eps: float = 1e-4
    layout_class_prior_max_var: float = 0.25

    # Layout visualization
    save_layout_boxes_only: bool = False
    layout_box_image_size: int = 256
    layout_log_individual_images: bool = False

    # -------------------------
    # Phase 8A.1 reward terms
    # -------------------------
    use_reward_tilting: bool = False

    reward_w_isolated_node: float = 0.05
    reward_w_bidirectional_edge: float = 0.25
    reward_w_dense_graph: float = 1.0

    reward_w_box_bounds: float = 0.25
    reward_w_layout_overlap: float = 1.00
    reward_w_layout_spread: float = 0.50
    reward_w_relation_geometry: float = 1.0

    # -------------------------
    # 8A.3 reward-tilted reverse sampling
    # -------------------------
    reward_tilt_alpha: float = 0.0
    reward_tilt_temperature: float = 1.0
    reward_tilt_num_sweeps: int = 1

    reward_tilt_objects: bool = True
    reward_tilt_edges: bool = False
    reward_tilt_relations: bool = True
    reward_tilt_use_layout: bool = False

    reward_tilt_obj_topk: int = 5
    reward_tilt_rel_topk: int = 5

    # For 8A.3a edge reward only.
    reward_tilt_edge_logit_band: float = 0.75
    reward_w_hub_degree: float = 0.50
    reward_hub_degree_threshold: int = 4

    # For 8A.3b rel reward only
    reward_tilt_relation_alpha: float = 0.5
    reward_w_relation_geometry_tilt: float = 1.0

    # For 8A.3c obj+rel reward
    reward_tilt_object_alpha: float = 0.25
    reward_w_object_class_prior_tilt: float = 0.5
    reward_w_object_relation_support_tilt: float = 0.25
    reward_tilt_obj_logit_margin:float = 1.0

    # 8A.4 layout-induced geometric reward
    reward_tilt_layout_alpha: float = 0.25
    reward_w_layout_overlap_tilt: float = 1.0
    reward_w_layout_spread_tilt: float = 0.5
    reward_w_box_bounds_tilt: float = 0.5

    # 8E.2 / 8E.3 layout regularization
    use_layout_regularization: bool = False
    layout_overlap_reg_weight: float = 0.02
    layout_spread_reg_weight: float = 0.01
    layout_min_center_spread: float = 0.18


    # -------------------------
    # 8E.4 relation-geometry consistency regularizer
    # -------------------------
    use_relation_geometry_reg: bool = False
    lambda_relation_geometry_reg: float = 0.02
    relation_geometry_margin: float = 0.02

    # -------------------------
    # 8E.6 sampled-layout denoising
    # -------------------------
    use_sampled_layout_training: bool = False
    lambda_sampled_layout: float = 0.1

    sampled_layout_stochastic_obj: bool = False
    sampled_layout_stochastic_edge: bool = False
    sampled_layout_stochastic_rel: bool = False
    sampled_layout_detach_state: bool = True
    sampled_layout_use_regularization: bool = False
    sampled_layout_use_relation_geometry: bool = False
    sampled_layout_use_class_prior: bool = False

    # -------------------------
    # 8E.1 graph law regularization
    # -------------------------
    use_graph_law_reg: bool = False
    lambda_graph_law_reg: float = 0.005
    graph_law_edge_weight: float = 1.0
    graph_law_degree_weight: float = 0.5
    graph_law_rel_weight: float = 0.5
    graph_law_eps: float = 1e-6

    ######### sanity check 1
    object_only_sanity: bool = True
    object_one_node_sanity: bool = False
    object_copy_sanity_t0: bool = False
    object_fixed_t_sanity: bool = True
    object_fixed_t: int = 10
    sanity_overfit_tiny: bool = True
    sanity_overfit_num_graphs_train: int = 4096
    sanity_overfit_num_graphs_val: int = 1024

    use_t_curriculum = False
    curriculum_stage1_node_corr = 0.80
    curriculum_stage2_node_corr = 0.80
    object_from_structure_only: bool = False
    object_condition_on_gt_structure: bool = True

    use_direct_obj_head: bool = True
    topk_obj_loss_only: bool = True
    topk_obj_k: int = 50




    
# use_plausibility_prior = False
# use_empirical_rel_prior = False
# use_empirical_unconditional_priors = False

# prior_use_gt_objects_in_train = False
# cond_use_clean_structure = False

# use_reverse_rel_class_weights = False  # at least one run

# Layout: OFF for sanity check
# use_layout_head: bool = False
# use_layout_supervision: bool = False
# use_layout_giou_loss: bool = False
# use_relation_geometry_loss: bool = False
# use_layout_class_priors: bool = False
# use_sampled_layout_training: bool = False
# sampled_layout_use_class_prior: bool = False
# sampled_layout_use_regularization: bool = False
# sampled_layout_use_relation_geometry: bool = False
# use_conditional_node_objective: bool = False
# use_reverse_vocab_step_training: bool = False
# use_sampled_state_training: bool = False
# use_reverse_step_training: bool = False
# use_reverse_vocab_heads = False
# full_reverse_use_reverse_vocab_heads = False
# unconditional_use_reverse_vocab_heads = False
# use_dedicated_reverse_branch = False



















