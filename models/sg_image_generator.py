import torch
from typing import Optional
import os

from transformers import AutoTokenizer, PretrainedConfig
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel

from models.sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms

from configs import ImageGeneratorConfig

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class SGConditionedImageGenerator:
    def __init__(self, opt: ImageGeneratorConfig, device: torch.device):
        self.opt = opt
        self.device = device
        self.weight_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        tokenizer_one = AutoTokenizer.from_pretrained(
            opt.stable_diffusion_checkpoint,
            subfolder="tokenizer",
            use_fast=False,
            cache_dir=opt.cache_dir,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            opt.stable_diffusion_checkpoint,
            subfolder="tokenizer_2",
            use_fast=False,
            cache_dir=opt.cache_dir,
        )

        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            opt.stable_diffusion_checkpoint,
            subfolder="text_encoder",
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            opt.stable_diffusion_checkpoint,
            subfolder="text_encoder_2",
        )

        text_encoder_one = text_encoder_cls_one.from_pretrained(
            opt.stable_diffusion_checkpoint,
            subfolder="text_encoder",
            variant="fp16" if self.device.type == "cuda" else None,
            cache_dir=opt.cache_dir,
        ).to(self.device)

        text_encoder_two = text_encoder_cls_two.from_pretrained(
            opt.stable_diffusion_checkpoint,
            subfolder="text_encoder_2",
            variant="fp16" if self.device.type == "cuda" else None,
            cache_dir=opt.cache_dir,
        ).to(self.device)

        vae = AutoencoderKL.from_pretrained(
            opt.stable_diffusion_checkpoint,
            subfolder="vae",
            variant="fp16" if self.device.type == "cuda" else None,
            cache_dir=opt.cache_dir,
        ).to(self.device)

        unet = UNet2DConditionModel.from_pretrained(
            opt.stable_diffusion_checkpoint,
            subfolder="unet",
            variant="fp16" if self.device.type == "cuda" else None,
            cache_dir=opt.cache_dir,
        ).to(self.device)

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            opt.stable_diffusion_checkpoint,
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            unet=unet,
            torch_dtype=self.weight_dtype,
            cache_dir=opt.cache_dir,
        ).to(self.device)

        self.sg_encoder = create_model_and_transforms(
            opt,
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=[tokenizer_one, tokenizer_two],
            model_config_json=opt.model_config_json,
            precision=opt.precision,
            device=self.device,
            force_quick_gelu=opt.force_quick_gelu,
            pretrained_image=opt.pretrained_image,
        ).to(self.device)

        checkpoint = torch.load(opt.sg_encoder_ckpt, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.sg_encoder.load_state_dict(state_dict)
        self.sg_encoder.eval()

        if getattr(opt, "compile_model", False) and hasattr(torch, "compile"):
            self.sg_encoder = torch.compile(self.sg_encoder)
            self.pipeline.unet = torch.compile(self.pipeline.unet)

    @torch.no_grad()
    def generate(
        self,
        all_triplets,
        all_isolated_items,
        all_global_ids,
        output_path: str,
        seed: Optional[int] = None,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        prompt_embeds, pooled_embeds = self.sg_encoder(
            all_triplets,
            all_isolated_items,
            all_global_ids,
        )

        image = self.pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            num_inference_steps=self.opt.num_inference_steps,
            width=self.opt.width,
            height=self.opt.height,
            guidance_scale=self.opt.guidance_scale,
            generator=generator,
        ).images[0]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        return image
