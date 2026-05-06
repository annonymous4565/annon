# diffusion/schedules.py

from dataclasses import dataclass
import torch
from configs import DiscreteSGConfig



def make_linear_beta_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Returns betas of shape [T].
    """
    return torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=torch.float32)


class DiscreteNoiseSchedules:
    def __init__(self, cfg: DiscreteSGConfig, device: torch.device = None):
        self.num_steps = cfg.num_diffusion_steps
        self.obj_betas = make_linear_beta_schedule(
            cfg.num_diffusion_steps, cfg.obj_beta_start, cfg.obj_beta_end, device=device
        )
        self.rel_betas = make_linear_beta_schedule(
            cfg.num_diffusion_steps, cfg.rel_beta_start, cfg.rel_beta_end, device=device
        )