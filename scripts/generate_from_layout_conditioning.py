import os
import json
from typing import Any, Optional

import pyrallis
import torch

from configs import LayoutImageGenConfig
from models.sg_layout_image_generator import LayoutConditionedImageGenerator


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_condition_tensors(layout_conditioning: dict, device: torch.device):
    """
    Expected JSON format:
    {
      "obj_class": [[1, 5, 10, 0, 0]],
      "obj_bbox": [[[0.0,0.0,1.0,1.0],
                    [0.1,0.2,0.4,0.6],
                    [0.5,0.2,0.8,0.7],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0]]],
      "is_valid_obj": [[1,1,1,0,0]]
    }

    Optional:
      "obj_mask": ...
    """
    obj_class = torch.tensor(layout_conditioning["obj_class"], dtype=torch.long, device=device)
    obj_bbox = torch.tensor(layout_conditioning["obj_bbox"], dtype=torch.float32, device=device)
    is_valid_obj = torch.tensor(layout_conditioning["is_valid_obj"], dtype=torch.long, device=device)

    obj_mask = None
    if "obj_mask" in layout_conditioning:
        obj_mask = torch.tensor(layout_conditioning["obj_mask"], dtype=torch.float32, device=device)

    return obj_class, obj_bbox, is_valid_obj, obj_mask


@pyrallis.wrap()
def main(opt: LayoutImageGenConfig):
    if not hasattr(opt, "conditioning_json"):
        raise ValueError("LayoutImageGenConfig must have a conditioning_json field")

    conditioning = load_json(opt.conditioning_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj_class, obj_bbox, is_valid_obj, obj_mask = build_condition_tensors(conditioning, device)

    generator = LayoutConditionedImageGenerator(opt)
    generator.generate(
        obj_class=obj_class,
        obj_bbox=obj_bbox,
        is_valid_obj=is_valid_obj,
        obj_mask=obj_mask,
        output_path=opt.output_path,
        seed=opt.seed,
    )

    print(f"Saved generated image to: {opt.output_path}")


if __name__ == "__main__":
    main()