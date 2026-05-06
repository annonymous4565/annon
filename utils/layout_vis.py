import os
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


def _cxcywh_to_xyxy(box: Sequence[float], image_size: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized [cx, cy, w, h] in [0,1] to pixel xyxy.
    """
    cx, cy, w, h = [float(v) for v in box]

    # clamp width/height and centers to sane range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    x1 = int(round(x1 * image_size))
    y1 = int(round(y1 * image_size))
    x2 = int(round(x2 * image_size))
    y2 = int(round(y2 * image_size))

    x1 = max(0, min(image_size - 1, x1))
    y1 = max(0, min(image_size - 1, y1))
    x2 = max(0, min(image_size - 1, x2))
    y2 = max(0, min(image_size - 1, y2))

    # just in case of numerical weirdness
    x0 = min(x1, x2)
    y0 = min(y1, y2)
    x1_ = max(x1, x2)
    y1_ = max(y1, y2)

    return x0, y0, x1_, y1_


def _default_color(idx: int) -> Tuple[int, int, int]:
    palette = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 0),
        (255, 140, 0),
        (180, 0, 180),
        (0, 180, 180),
        (220, 50, 50),
        (120, 120, 0),
        (80, 80, 255),
        (255, 0, 150),
    ]
    return palette[idx % len(palette)]


def draw_layout_boxes(
    obj_class: torch.Tensor,         # [N]
    obj_bbox: torch.Tensor,          # [N,4]
    is_valid_obj: torch.Tensor,      # [N]
    output_path: str,
    class_names: Optional[Sequence[str]] = None,
    image_size: int = 256,
    background: str = "white",
    draw_text: bool = True,
    skip_first_object: bool = True,
):
    """
    Draw boxes on a blank canvas.
    Improved label placement to avoid all text piling up in the same corner.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if background == "white":
        img = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    else:
        img = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    obj_class = obj_class.detach().cpu()
    obj_bbox = obj_bbox.detach().cpu()
    is_valid_obj = is_valid_obj.detach().cpu().long()

    for i in range(obj_class.shape[0]):
        if skip_first_object and i == 0:
            continue
        if int(is_valid_obj[i].item()) == 0:
            continue

        color = _default_color(i)

        try:
            raw_box = obj_bbox[i].tolist()
            box = _cxcywh_to_xyxy(raw_box, image_size)

            # Skip degenerate boxes after conversion
            if box[2] <= box[0] or box[3] <= box[1]:
                print(f"[WARN] Skipping degenerate box[{i}] raw={raw_box} xyxy={box}")
                continue

            draw.rectangle(box, outline=color, width=3)

            if draw_text:
                cls_id = int(obj_class[i].item())
                label = str(cls_id)
                if class_names is not None and 0 <= cls_id < len(class_names):
                    label = class_names[cls_id]

                # prepend object slot index for debugging/readability
                label = f"{i}:{label}"

                # Prefer drawing above the box; otherwise draw inside.
                tx = min(box[0] + 3, image_size - 80)
                ty = box[1] - 14

                if ty < 0:
                    ty = box[1] + 3

                # Stagger overlapping labels a bit
                ty = min(ty + (i % 5) * 10, image_size - 12)

                if font is not None:
                    try:
                        text_bbox = draw.textbbox((tx, ty), label, font=font)
                        draw.rectangle(text_bbox, fill=(255, 255, 255))
                    except Exception:
                        pass

                draw.text((tx, ty), label, fill=color, font=font)

        except Exception as e:
            print(f"[ERROR] Failed box[{i}] = {obj_bbox[i].tolist()}")
            print(f"[ERROR] is_valid[{i}] = {is_valid_obj[i].item()}")
            raise e

    img.save(output_path)
    return output_path


def draw_layout_boxes_on_image(
    image_path: str,
    obj_class: torch.Tensor,         # [N]
    obj_bbox: torch.Tensor,          # [N,4]
    is_valid_obj: torch.Tensor,      # [N]
    output_path: str,
    class_names: Optional[Sequence[str]] = None,
    draw_text: bool = True,
    skip_first_object: bool = True,
):
    """
    Overlay boxes on an already generated image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    image_size = img.size[0]
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    obj_class = obj_class.detach().cpu()
    obj_bbox = obj_bbox.detach().cpu()
    is_valid_obj = is_valid_obj.detach().cpu().bool()

    for i in range(obj_class.shape[0]):
        if skip_first_object and i == 0:
            continue
        if not bool(is_valid_obj[i].item()):
            continue

        color = _default_color(i)
        box = _cxcywh_to_xyxy(obj_bbox[i].tolist(), image_size)
        draw.rectangle(box, outline=color, width=3)

        if draw_text:
            cls_id = int(obj_class[i].item())
            label = str(cls_id)
            if class_names is not None and 0 <= cls_id < len(class_names):
                label = class_names[cls_id]

            tx, ty = box[0], max(0, box[1] - 12)
            draw.text((tx, ty), label, fill=color, font=font)

    img.save(output_path)
    return output_path