from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union, Dict, Any
from pathlib import Path


import re
import json
import pickle
import subprocess
import tempfile

import torch
from PIL import Image

try:
    import clip
except ImportError:
    clip = None

# Optional FID package:
# pip install torch-fidelity
try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None



@dataclass
class ImageEvalResult:
    fid: Optional[float]
    clip_score: Optional[float]
    span_similarity: Optional[float]


def serialize_scene_graph_text(nodes, triplets) -> str:
    """
    Simple SG -> text serialization for CLIP.
    """
    node_text = ", ".join(str(n) for n in nodes if str(n) != "__pad__")
    if len(triplets) == 0:
        trip_text = "no relations"
    else:
        rels = []
        for s_idx, s_name, rel, o_idx, o_name in triplets:
            rels.append(f"{s_name} {rel} {o_name}")
        trip_text = "; ".join(rels)
    return f"Objects: {node_text}. Relations: {trip_text}."

def _validate_image_dir(path_like: Union[str, os.PathLike], name: str) -> str:
    if path_like is None:
        raise ValueError(f"{name} is None")

    path_str = str(Path(path_like).expanduser().resolve())

    if not os.path.exists(path_str):
        raise ValueError(f"{name} does not exist: {path_str}")
    if not os.path.isdir(path_str):
        raise ValueError(f"{name} is not a directory: {path_str}")

    files = [
        f for f in os.listdir(path_str)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    ]
    if len(files) == 0:
        raise ValueError(f"{name} contains no image files: {path_str}")

    return path_str


@torch.no_grad()
def compute_clip_score(
    image_paths: Sequence[str],
    graph_texts: Sequence[str],
    device: Optional[torch.device] = None,
    model_name: str = "ViT-B/32",
) -> float:
    if clip is None:
        raise ImportError("clip is not installed. Install openai-clip.")

    assert len(image_paths) == len(graph_texts)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    sims = []
    for img_path, text in zip(image_paths, graph_texts):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        tokens = clip.tokenize([text]).to(device)

        image_feat = model.encode_image(image)
        text_feat = model.encode_text(tokens)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sim = (image_feat * text_feat).sum(dim=-1).item()
        sims.append(sim)

    return float(sum(sims) / max(len(sims), 1))


def compute_fid(
    generated_image_dir: Union[str, os.PathLike],
    reference_image_dir: Union[str, os.PathLike],
) -> float:
    if calculate_metrics is None:
        raise ImportError("torch-fidelity is not installed.")

    gen_dir = _validate_image_dir(generated_image_dir, "generated_image_dir")
    ref_dir = _validate_image_dir(reference_image_dir, "reference_image_dir")

    metrics = calculate_metrics(
        input1=gen_dir,
        input2=ref_dir,
        cuda=torch.cuda.is_available(),
        fid=True,
        kid=False,
        isc=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


class SPANScorer:
    """
    Wrapper scaffold for a pretrained SPAN similarity model.

    You can adapt this once you decide exactly how you want to load the
    pretrained model from:
    https://github.com/yrcong/Learning_Similarity_between_Graphs_Images

    Expected public method:
        score(image_paths, scene_graph_payloads) -> float
    """

    def __init__(self, model_root: str, device: Optional[torch.device] = None):
        self.model_root = model_root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO:
        # Load pretrained SPAN model here.
        # Keep this class thin so the rest of your evaluation code does not change.
        self.model = None

    @torch.no_grad()
    def score(
        self,
        image_paths: Sequence[str],
        scene_graph_payloads: Sequence[dict],
    ) -> float:
        """
        scene_graph_payloads can be your conditioning dicts, or decoded graph dicts.
        """
        if self.model is None:
            raise NotImplementedError(
                "SPANScorer loading is not implemented yet. "
                "Wire this to the pretrained SPAN repo/model."
            )

        # TODO:
        # Return mean SG-image similarity score.
        raise NotImplementedError


def evaluate_image_generation(
    generated_image_dir: Optional[str] = None,
    reference_image_dir: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    graph_texts: Optional[Sequence[str]] = None,
    span_scorer: Optional[SPANScorer] = None,
    scene_graph_payloads: Optional[Sequence[dict]] = None,
    device: Optional[torch.device] = None,
) -> ImageEvalResult:
    fid = None
    clip_score = None
    span_similarity = None

    if generated_image_dir is not None and reference_image_dir is not None:
        fid = compute_fid(generated_image_dir, reference_image_dir)

    if image_paths is not None and graph_texts is not None:
        clip_score = compute_clip_score(image_paths, graph_texts, device=device)

    if span_scorer is not None and image_paths is not None and scene_graph_payloads is not None:
        span_similarity = span_scorer.score(image_paths, scene_graph_payloads)

    return ImageEvalResult(
        fid=fid,
        clip_score=clip_score,
        span_similarity=span_similarity,
    )



def build_span_prediction_record(
    image_id: int,
    boxes,
    labels,
    rel_annotations,
) -> Dict[str, Any]:
    """
    Expected by the SPAN benchmark repo README / benchmark pipeline:
      {
        "image_id": int,
        "boxes": np.ndarray or array-like,
        "labels": np.ndarray or array-like,
        "rel_annotations": np.ndarray or array-like,
      }
    """
    return {
        "image_id": int(image_id),
        "boxes": boxes,
        "labels": labels,
        "rel_annotations": rel_annotations,
    }


def save_span_prediction_pkl(
    prediction_records: List[Dict[str, Any]],
    output_pkl: str,
) -> str:
    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(prediction_records, f)
    return output_pkl


def parse_span_r_precision(stdout: str) -> Dict[str, float]:
    """
    Parses lines like:
      R Precision for image-graph matching:  0.73
      R Precision for graph-image matching:  0.71
    """
    out = {}

    m1 = re.search(r"R Precision for image-graph matching:\s*([0-9.]+)", stdout)
    m2 = re.search(r"R Precision for graph-image matching:\s*([0-9.]+)", stdout)

    if m1:
        out["span_r_precision_image_to_graph"] = float(m1.group(1))
    if m2:
        out["span_r_precision_graph_to_image"] = float(m2.group(1))

    if not out:
        raise ValueError("Could not parse SPAN R-Precision output.")

    # primary metric: mean of both directions
    vals = list(out.values())
    out["span_similarity"] = sum(vals) / len(vals)
    return out


def compute_span_r_precision(
    span_repo_root: str,
    span_ckpt_path: str,
    span_img_folder: str,
    prediction_records: List[Dict[str, Any]],
    node_bbox: bool = False,
    batch_size: int = 100,
    image_layer_num: int = 6,
    graph_layer_num: int = 6,
    python_exec: str = "python",
) -> Dict[str, float]:
    """
    Uses the official repo's benchmark.py entrypoint.
    """
    span_repo_root = str(Path(span_repo_root).expanduser().resolve())
    span_ckpt_path = str(Path(span_ckpt_path).expanduser().resolve())
    span_img_folder = str(Path(span_img_folder).expanduser().resolve())

    if not os.path.exists(os.path.join(span_repo_root, "benchmark.py")):
        raise ValueError(f"SPAN repo root invalid: {span_repo_root}")
    if not os.path.exists(span_ckpt_path):
        raise ValueError(f"SPAN checkpoint missing: {span_ckpt_path}")
    if not os.path.isdir(span_img_folder):
        raise ValueError(f"SPAN image folder missing: {span_img_folder}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_pkl = os.path.join(tmpdir, "span_predictions.pkl")
        save_span_prediction_pkl(prediction_records, pred_pkl)

        cmd = [
            python_exec,
            os.path.join(span_repo_root, "benchmark.py"),
            "--batch_size", str(batch_size),
            "--image_layer_num", str(image_layer_num),
            "--graph_layer_num", str(graph_layer_num),
            "--resume", span_ckpt_path,
            "--eval",
            "--prediction", pred_pkl,
            "--img_folder", span_img_folder,
        ]
        if node_bbox:
            cmd.append("--node_bbox")

        proc = subprocess.run(
            cmd,
            cwd=span_repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "SPAN benchmark failed.\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

        return parse_span_r_precision(proc.stdout)








