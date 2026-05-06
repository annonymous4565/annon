from typing import Any, Dict, Tuple
import torch

from configs import DiscreteSGConfig
from training.trainer_ddp import StructuredSGDDPTrainer


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _extract_model_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]

    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise KeyError(
        "Could not find model weights in checkpoint. "
        "Expected one of: 'model_state_dict', 'model', 'state_dict', or raw state_dict."
    )


def _extract_cfg(ckpt: Dict[str, Any]):
    if "opt" in ckpt and ckpt["opt"] is not None:
        return ckpt["opt"]

    if "config" in ckpt and ckpt["config"] is not None:
        cfg_val = ckpt["config"]
        if isinstance(cfg_val, dict):
            return DiscreteSGConfig(**cfg_val)
        return cfg_val

    raise KeyError(
        "Could not find config in checkpoint. Expected one of: 'opt' or 'config'."
    )


def build_trainer_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[StructuredSGDDPTrainer, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = _extract_cfg(ckpt)
    trainer = StructuredSGDDPTrainer(cfg)

    state_dict = _extract_model_state_dict(ckpt)
    state_dict = _strip_module_prefix(state_dict)

    # IMPORTANT: load into the underlying nn.Module, not the DDP wrapper
    trainer.unwrap_model().load_state_dict(state_dict, strict=strict)

    trainer.unwrap_model().to(device)
    trainer.unwrap_model().eval()

    # optional: keep wrapper in eval too
    trainer.model.eval()

    return trainer, ckpt


def load_model_and_objgen_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
):
    trainer, ckpt = build_trainer_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        strict=strict,
    )
    return trainer.model, trainer.obj_gen, trainer, trainer.opt, ckpt