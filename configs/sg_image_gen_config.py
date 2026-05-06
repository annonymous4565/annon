# configs/sg_image_gen_config.py
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List
from configs import DiscreteSGConfig


@dataclass
class SGImageGenConfig:
    # -------------------------
    # Dataset / paths
    # -------------------------
    image_dir: str = "./data"
    train_json_path: str = "./dataset/dataset/train_all.json"
    val_json_path: str = "./dataset/dataset/val_all.json"

    max_objects_per_image: int = 10
    use_orphaned_objects: bool = True
    include_relationships: bool = True
    model_config_json: str = ""
    image_size: int = 512

    # -------------------------
    # Graph tower config
    # -------------------------
    graph_width: int = 512
    num_graph_layer: int = 5
    embed_dim: int = 512

    # -------------------------
    # Diffusion model checkpoints
    # -------------------------
    stable_diffusion_checkpoint: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_checkpoint: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    compile_model: bool = False

    # -------------------------
    # Runtime / inference
    # -------------------------
    seed: int = 42
    num_inference_steps: int = 40
    high_noise_fraction: float = 0.8
    img_size: int = 512
    accusteps: int = 64
    cache_dir: str = "./cache/huggingface"
    val_times_per_epoch: int = 1

    # -------------------------
    # Training-style fields kept for compatibility
    # -------------------------
    name: Optional[str] = None
    workers: int = 1
    batch_size: int = 1
    val_batch_size: int = 1
    epochs: int = 59
    lr: float = 5.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    wd: float = 0.2
    warmup: int = 10000
    use_bn_sync: bool = False
    skip_scheduler: bool = False
    save_frequency: int = 1
    save_most_recent: bool = False
    logs: str = "./logs"
    log_local: bool = False

    precision: str = "amp_bfloat16"
    pretrained: str = ""
    pretrained_image: bool = False

    lock_image: bool = False
    lock_image_unlocked_groups: int = 0
    lock_image_freeze_bn_stats: bool = False
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None
    grad_checkpointing: bool = False
    local_loss: bool = False
    gather_with_grad: bool = False
    force_quick_gelu: bool = False

    dist_url: str = "env://"
    dist_backend: str = "nccl"
    report_to: str = "tensorboard"
    debug: bool = False
    ddp_static_graph: bool = False
    no_set_device_rank: bool = False
    norm_gradient_clip: float = 10.0

    # -------------------------
    # New inference-only fields
    # -------------------------
    sg_encoder_ckpt: str = "./checkpoints/image_gen/baseline3_100.pt"
    conditioning_json: str = "./input/conditioning/sample.json"
    output_path: str = "./output/image_gen/generated.png"
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0


    # Val loader gen
    val_npz_path: str = './data/visual_genome/processed/vg_sg_val_nmax20_seed42.npz'
    batch_size = 16
    mask_obj_token_id: int = -1  # set at runtime to num_obj_classes
    output_dir: str = "./output/image_gen/"
    max_images: int = 50
    draw_flowchart: bool = True
    flowchart_rankdir: str = "TB"          # "LR" or "TB"
    flowchart_format: str = "png"          # png / svg / pdf
    flowchart_show_node_ids: bool = True

    #### Text-conditioning
    demon_num_candidates = 4
    demon_selection_mode = "argmax"
    demon_every_n_steps = 10
    demon_use_x0_proxy = True
    # demon_selection_mode = "softmax"
    demon_guidance_scale = 20.0
    demon_softmax_temperature = 1.0

    text_prompt: str = "a person riding a horse"
    demon_scorer: str = "clip"  # clip | keyword
    clip_text_model: str = "openai/clip-vit-base-patch32"

    demon_start_t: Optional[int] = None
    demon_end_t: int = 1
    demon_verbose: bool = True



@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int]
    width: int
    head_width: int
    image_size: int
    mlp_ratio: float
    patch_size: int = None
    timm_model_name: str = None
    timm_model_pretrained: bool = None
    timm_pool: str = None
    timm_proj: str = None


@dataclass
class CLIPGraphCfg:
    layers: int
    width: int


@dataclass
class ImageGeneratorConfig(DiscreteSGConfig):
    # SG model

    # how many unconditional samples to generate
    max_images: int = 4

    # graph shape source
    # use a real dataset item only to get node_mask / edge_mask shape and vocab
    shape_source_index: int = 0

    # SG-conditioned image model
    stable_diffusion_checkpoint: str = "stabilityai/stable-diffusion-xl-base-1.0"
    cache_dir: str = "/.cache/huggingface"
    model_config_json: str = ""
    sg_encoder_ckpt: str = "./checkpoints/image_gen/baseline3_100.pt"

    precision: str = "amp_bfloat16"
    force_quick_gelu: bool = False
    pretrained_image: bool = False
    compile_model: bool = False

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50

    output_dir: str = "./output/image_gen/"


    # Compatibility fields for create_model_and_transforms
    image_dir: str = "./data"
    train_json_path: str = "./dataset/dataset/train_all.json"
    val_json_path: str = "./dataset/dataset/val_all.json"
    max_objects_per_image: int = 10
    use_orphaned_objects: bool = True
    include_relationships: bool = True
    image_size: int = 512

    graph_width: int = 512
    num_graph_layer: int = 5
    embed_dim: int = 512

    refiner_checkpoint: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    high_noise_fraction: float = 0.8
    img_size: int = 512
    accusteps: int = 64
    val_times_per_epoch: int = 1

    name: Optional[str] = None
    workers: int = 1
    val_batch_size: int = 1
    epochs: int = 59
    lr: float = 5.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    wd: float = 0.2
    warmup: int = 10000
    use_bn_sync: bool = False
    skip_scheduler: bool = False
    save_frequency: int = 1
    save_most_recent: bool = False
    logs: str = "./logs"
    log_local: bool = False

    pretrained: str = ""
    lock_image: bool = False
    lock_image_unlocked_groups: int = 0
    lock_image_freeze_bn_stats: bool = False
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None
    grad_checkpointing: bool = False
    local_loss: bool = False
    gather_with_grad: bool = False

    dist_url: str = "env://"
    dist_backend: str = "nccl"
    report_to: str = "tensorboard"
    debug: bool = False
    ddp_static_graph: bool = False
    no_set_device_rank: bool = False
    norm_gradient_clip: float = 10.0


@dataclass
class LayoutImageGenConfig(DiscreteSGConfig):
    # -------------------------
    # LayoutDiffusion paths
    # -------------------------
    layoutdiffusion_config: str = "./checkpoints/layout_gen/LayoutDiffusion_large.yaml"
    layoutdiffusion_ckpt: str = "./checkpoints/layout_gen/VG_256x256_LayoutDiffusion_large_ema_1450000.pt"

    # -------------------------
    # Runtime
    # -------------------------
    seed: Optional[int] = 42
    compile_model: bool = False
    conditioning_json: str = "./input/conditioning/layout_sample.json"
    # -------------------------
    # Output
    # -------------------------
    output_path: str = "./output/layout_image_gen/generated.png"

    # -------------------------
    # Sampling override knobs
    # These should match your YAML unless you want to override.
    # -------------------------
    layout_sample_method: str = "ddim"




@dataclass
class SGLayoutImageGenConfig(DiscreteSGConfig):
    # =========================================================
    # LayoutDiffusion backend
    # =========================================================
    layoutdiffusion_config: str = "./checkpoints/layout_gen/LayoutDiffusion_large.yaml"
    layoutdiffusion_ckpt: str = "./checkpoints/layout_gen/VG_256x256_LayoutDiffusion_large_ema_1450000.pt"
    layout_sample_method: str = "ddim"   # dpm_solver | ddpm | ddim

    # =========================================================
    # Runtime / output
    # =========================================================
    seed: int = 42
    output_dir: str = "./output/layout_image_gen/"
    max_images: int = 3
    shape_source_index: int = 0

    # =========================================================
    # Sampler controls for unconditional SG sampling
    # =========================================================
    unconditional_stochastic_obj: bool = False
    unconditional_stochastic_edge: bool = False
    unconditional_stochastic_rel: bool = False

    unconditional_use_reverse_vocab_heads: bool = True
    unconditional_obj_temp: float = 1.0
    unconditional_rel_temp: float = 1.0
    unconditional_edge_logit_threshold: float = 0.5
    unconditional_relation_edge_logit_threshold: float = 0.0

    unconditional_use_degree_pruning: bool = False
    unconditional_max_out_degree: int = 0
    unconditional_max_in_degree: int = 0

    # =========================================================
    # Optional visualization
    # =========================================================
    draw_flowchart: bool = True
    flowchart_rankdir: str = "TB"          # "LR" or "TB"
    flowchart_format: str = "png"          # png / svg / pdf
    flowchart_show_node_ids: bool = True
    save_images_with_bboxs: bool = True
    save_layout_boxes_only: bool = True
    layout_box_image_size: int = 256
    layout_log_individual_images: bool = True

    # Val loader gen
    val_npz_path: str = './data/visual_genome/processed/vg_sg_val_nmax20_seed42.npz'
    batch_size = 16
    mask_obj_token_id: int = -1  # set at runtime to num_obj_classes
    output_dir: str = "./output/val_layout_image_gen/"
    max_images: int = 5

    #### Text-conditioning
    demon_num_candidates = 4
    demon_selection_mode = "argmax"
    demon_every_n_steps = 10
    demon_use_x0_proxy = True
    # demon_selection_mode = "softmax"
    demon_guidance_scale = 20.0
    demon_softmax_temperature = 1.0

    text_prompt: str = "a person riding a horse"
    demon_scorer: str = "clip"  # clip | keyword
    clip_text_model: str = "openai/clip-vit-base-patch32"

    demon_start_t: Optional[int] = None
    demon_end_t: int = 1
    demon_verbose: bool = True



@dataclass 
class DiscreteSGEvalConfig(DiscreteSGConfig): #, SGLayoutImageGenConfig, SGImageGenConfig):
    # eval dataset
    eval_dataset_type: str = "vg"   # vg | coco
    eval_batch_size: int = 1
    eval_num_workers: int = 0
    eval_image_size: int = 512
    eval_return_images: bool = True
    eval_max_samples: Optional[int] = None

    eval_index_json: str = ""          # if empty, create/use output_root/eval_index.json
    reuse_eval_index: bool = True

    # VG eval
    eval_npz_path: str = "./data/visual_genome/processed/vg_sg_degree_filter_new_test_nmax20.npz"
    vg_image_dirs: str = "./data/visual_genome/VG_100K,./data/visual_genome/VG_100K_2"
    vg_image_data_json_path: Optional[str] = "./data/visual_genome/image_data.json"
    eval_prompt_json_path: Optional[str] = None
    vg_fallback_prompt_from_graph: bool = True

    # COCO eval
    coco_image_dir: str = "./data/coco/images/val2017"
    coco_instances_json: str = "./data/coco/annotations/instances_val2017.json"
    coco_captions_json: str = "./data/coco/annotations/captions_val2017.json"
    coco_max_objects_per_image: int = 20
    coco_min_objects_per_image: int = 1
    coco_min_object_size: float = 0.0
    coco_caption_strategy: str = "first"  # first | random


    # -------------------------
    # Mode / output
    # -------------------------
    mode: str = "text-sg"  # text-sg | text-sg-img | text-sg-layout | text-sg-layout-img
    output_root: str = "./output/master_eval"
    max_eval_items: int = 100

    # -------------------------
    # DiscreteSG model
    # -------------------------
    ckpt: str = ""
    num_diffusion_steps: int = 50
    mask_obj_token_id: int = 0
    seed: int = 42

    # Shape for COCO text-SG sampling
    eval_num_nodes: int = 20

    # -------------------------
    # Text-guided demon sampler
    # -------------------------
    demon_scorer: str = "clip"  # clip | keyword
    clip_text_model: str = "openai/clip-vit-base-patch32"

    demon_num_candidates: int = 4
    demon_selection_mode: str = "argmax"  # argmax | softmax
    demon_guidance_scale: float = 10.0
    demon_softmax_temperature: float = 1.0
    demon_every_n_steps: int = 5
    demon_start_t: Optional[int] = None
    demon_end_t: int = 1
    demon_use_x0_proxy: bool = True
    demon_verbose: bool = False

    unconditional_stochastic_obj: bool = False
    unconditional_stochastic_edge: bool = False
    unconditional_stochastic_rel: bool = False
    unconditional_obj_temp: float = 1.0
    unconditional_rel_temp: float = 1.0
    unconditional_edge_logit_threshold: float = 0.5
    unconditional_relation_edge_logit_threshold: float = 0.0
    unconditional_use_degree_pruning: bool = False
    unconditional_max_out_degree: int = 0
    unconditional_max_in_degree: int = 0

    # -------------------------
    # SG-conditioned image generator fields
    # Keep same names expected by SGConditionedImageGenerator
    # -------------------------
    stable_diffusion_checkpoint: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_checkpoint: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    compile_model: bool = False
    num_inference_steps: int = 40
    high_noise_fraction: float = 0.8
    image_size: int = 512
    accusteps: int = 64
    cache_dir: str = "/.cache/huggingface"
    val_times_per_epoch: int = 1
    model_config_json: str = ""
    precision: str = "amp_bfloat16"
    force_quick_gelu: bool = False
    pretrained_image: bool = False
    sg_encoder_ckpt: str = "./checkpoints/image_gen/baseline3_100.pt"
    
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0


    # Compatibility fields used by SG encoder config
    image_dir: str = "./data"
    max_objects_per_image: int = 10
    use_orphaned_objects: bool = True
    include_relationships: bool = True
    
    graph_width: int = 512
    num_graph_layer: int = 5
    embed_dim: int = 512

    # -------------------------
    # LayoutDiffusion generator fields
    # -------------------------
    layoutdiffusion_config: str = "./checkpoints/layout_gen/LayoutDiffusion_large.yaml"
    layoutdiffusion_ckpt: str = "./checkpoints/layout_gen/VG_256x256_LayoutDiffusion_large_ema_1450000.pt"
    layout_sample_method: str = "ddim"
    layout_image_obj_class_id: int = 0
