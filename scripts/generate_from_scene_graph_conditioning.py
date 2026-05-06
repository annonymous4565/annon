# generate_from_scene_graph_conditioning.py
import os
import json
from typing import Any, Optional

import pyrallis
import torch
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel

from configs import SGImageGenConfig
from models.sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    def __init__(self, opt: SGImageGenConfig):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    @torch.no_grad()
    def generate(
        self,
        all_triplets,
        all_isolated_items,
        all_global_ids,
        output_path: Optional[str] = None,
    ):
        output_path = output_path or self.opt.output_path

        generator = None
        if self.opt.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.opt.seed)

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


@pyrallis.wrap()
def main(opt: SGImageGenConfig):
    conditioning = load_json(opt.conditioning_json)

    all_triplets = conditioning["all_triplets"]
    all_isolated_items = conditioning["all_isolated_items"]
    all_global_ids = conditioning["all_global_ids"]

    generator = SGConditionedImageGenerator(opt)
    generator.generate(
        all_triplets=all_triplets,
        all_isolated_items=all_isolated_items,
        all_global_ids=all_global_ids,
        output_path=opt.output_path,
    )

    print(f"Saved generated image to: {opt.output_path}")


if __name__ == "__main__":
    main()