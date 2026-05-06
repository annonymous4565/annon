import os
from typing import Optional

import torch
import torch as th
from PIL import Image
from omegaconf import OmegaConf

from configs import LayoutImageGenConfig

# Adjust these imports to your local LayoutDiffusion package structure if needed
from models.layout_diffusion.layout_diffusion_unet import build_model
from models.layout_diffusion.respace import build_diffusion
from models.layout_diffusion.util import fix_seed
from models.layout_diffusion.dataset.util import image_unnormalize_batch

from models.layout_diffusion.dpm_solver.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


def save_tensor_image(img_tensor: th.Tensor, output_path: str) -> None:
    """
    img_tensor: [3,H,W], expected in [0,1] after unnormalize
    """
    x = img_tensor.detach().cpu().clamp(0.0, 1.0)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    Image.fromarray(x).save(output_path)


class LayoutConditionedImageGenerator:
    def __init__(self, opt: LayoutImageGenConfig):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = OmegaConf.load(opt.layoutdiffusion_config)

        if getattr(self.cfg.sample, "fix_seed", False):
            fix_seed()

        # Build model
        self.model = build_model(self.cfg)
        self.model.to(self.device)

        ckpt_path = opt.layoutdiffusion_ckpt
        if ckpt_path is None or ckpt_path == "":
            raise ValueError("layoutdiffusion_ckpt is empty")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        try:
            self.model.load_state_dict(checkpoint, strict=True)
        except Exception:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)

        if getattr(self.cfg.sample, "use_fp16", False):
            self.model.convert_to_fp16()

        self.model.eval()

        # Build sampling utilities
        if self.opt.layout_sample_method == "dpm_solver":
            self.noise_schedule = NoiseScheduleVP(schedule="linear")
            self.diffusion = None
        elif self.opt.layout_sample_method in ["ddpm", "ddim"]:
            self.diffusion = build_diffusion(
                self.cfg,
                timestep_respacing=self.cfg.sample.timestep_respacing,
            )
            self.noise_schedule = None
        else:
            raise NotImplementedError(
                f"Unsupported layout_sample_method: {self.opt.layout_sample_method}"
            )

    def model_fn(
        self,
        x,
        t,
        obj_class=None,
        obj_bbox=None,
        obj_mask=None,
        is_valid_obj=None,
        **kwargs,
    ):
        """
        Matches LayoutDiffusion classifier-free sampling logic.
        """
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = self.model(
            x,
            t,
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj,
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        # unconditional branch
        uncond_obj_class = th.ones_like(obj_class).fill_(
            self.model.layout_encoder.num_classes_for_layout_object - 1
        )
        uncond_obj_class[:, 0] = 0

        uncond_obj_bbox = th.zeros_like(obj_bbox)
        uncond_obj_bbox[:, 0] = th.tensor([0, 0, 1, 1], dtype=obj_bbox.dtype, device=obj_bbox.device)

        uncond_is_valid_obj = th.zeros_like(is_valid_obj)
        uncond_is_valid_obj[:, 0] = 1

        uncond_obj_mask = None
        if obj_mask is not None:
            uncond_obj_mask = th.zeros_like(obj_mask)
            uncond_obj_mask[:, 0] = th.ones(obj_mask.shape[-2:], device=obj_mask.device)

        uncond_image, uncond_extra_outputs = self.model(
            x,
            t,
            obj_class=uncond_obj_class,
            obj_bbox=uncond_obj_bbox,
            obj_mask=uncond_obj_mask,
            is_valid_obj=uncond_is_valid_obj,
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + self.cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if self.opt.layout_sample_method in ["ddpm", "ddim"]:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    @torch.no_grad()
    def generate(
        self,
        obj_class: th.Tensor,
        obj_bbox: th.Tensor,
        is_valid_obj: th.Tensor,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
        obj_mask: Optional[th.Tensor] = None,
    ):
        output_path = output_path or self.opt.output_path

        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        obj_class = obj_class.to(self.device)
        obj_bbox = obj_bbox.to(self.device)
        is_valid_obj = is_valid_obj.to(self.device)

        model_kwargs = {
            "obj_class": obj_class,
            "obj_bbox": obj_bbox,
            "is_valid_obj": is_valid_obj,
        }

        if obj_mask is not None and "obj_mask" in self.cfg.data.parameters.used_condition_types:
            model_kwargs["obj_mask"] = obj_mask.to(self.device)

        B = obj_class.shape[0]
        H = int(self.cfg.data.parameters.image_size)
        W = int(self.cfg.data.parameters.image_size)

        if self.opt.layout_sample_method == "dpm_solver":
            wrapped_model_fn = model_wrapper(
                self.model_fn,
                self.noise_schedule,
                is_cond_classifier=False,
                total_N=1000,
                model_kwargs=model_kwargs,
            )

            dpm_solver = DPM_Solver(wrapped_model_fn, self.noise_schedule)

            x_T = th.randn((B, 3, H, W), device=self.device)
            sample = dpm_solver.sample(
                x_T,
                steps=int(self.cfg.sample.timestep_respacing[0]),
                eps=float(self.cfg.sample.eps),
                adaptive_step_size=self.cfg.sample.adaptive_step_size,
                fast_version=self.cfg.sample.fast_version,
                clip_denoised=False,
                rtol=self.cfg.sample.rtol,
            )
            sample = sample.clamp(-1, 1)

        elif self.opt.layout_sample_method in ["ddpm", "ddim"]:
            sample_fn = (
                self.diffusion.p_sample_loop
                if self.opt.layout_sample_method == "ddpm"
                else self.diffusion.ddim_sample_loop
            )

            all_results = sample_fn(
                self.model_fn,
                (B, 3, H, W),
                clip_denoised=self.cfg.sample.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=self.device,
            )

            last_result = all_results[-1]
            sample = last_result["sample"].clamp(-1, 1)

        else:
            raise NotImplementedError

        sample = image_unnormalize_batch(sample)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_tensor_image(sample[0], output_path)
        return sample[0]