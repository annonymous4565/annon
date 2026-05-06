import os
import json
import argparse
from pathlib import Path

from evaluation.image_metrics import (
    compute_fid,
    compute_clip_score,
    compute_span_r_precision,
)


def load_graph_texts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, list):
        raise ValueError("graph_text_json must be a JSON list of strings")

    for i, x in enumerate(obj):
        if not isinstance(x, str):
            raise ValueError(f"graph_text_json entry {i} is not a string")

    return obj


def load_span_prediction_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, list):
        raise ValueError("SPAN prediction records JSON must be a list")

    required = {"image_id", "boxes", "labels", "rel_annotations"}
    for i, row in enumerate(obj):
        if not isinstance(row, dict):
            raise ValueError(f"SPAN prediction record {i} is not a dict")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"SPAN prediction record {i} missing keys: {sorted(missing)}")

    return obj


def get_sorted_image_files(image_dir: str):
    image_dir = str(Path(image_dir).expanduser().resolve())
    if not os.path.isdir(image_dir):
        raise ValueError(f"Not a directory: {image_dir}")

    files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    ]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_image_dir", type=str, required=True)
    parser.add_argument("--ref_image_dir", type=str, required=True)
    parser.add_argument("--graph_text_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # optional SPAN block
    parser.add_argument("--span_repo_root", type=str, default="")
    parser.add_argument("--span_ckpt", type=str, default="")
    parser.add_argument("--span_img_folder", type=str, default="")
    parser.add_argument("--span_prediction_records_json", type=str, default="")
    parser.add_argument("--span_node_bbox", action="store_true")
    parser.add_argument("--span_batch_size", type=int, default=100)
    parser.add_argument("--span_image_layer_num", type=int, default=6)
    parser.add_argument("--span_graph_layer_num", type=int, default=6)
    parser.add_argument("--python_exec", type=str, default="python")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = {}

    # -------------------------
    # FID
    # -------------------------
    try:
        fid = compute_fid(
            generated_image_dir=args.gen_image_dir,
            reference_image_dir=args.ref_image_dir,
        )
        metrics["fid"] = fid
    except Exception as e:
        metrics["fid_error"] = str(e)
        print(f"[WARN] FID failed: {e}")

    # -------------------------
    # CLIP
    # -------------------------
    try:
        image_files = get_sorted_image_files(args.gen_image_dir)
        graph_texts = load_graph_texts(args.graph_text_json)

        if len(image_files) != len(graph_texts):
            raise ValueError(
                f"Number of generated images ({len(image_files)}) does not match "
                f"number of graph texts ({len(graph_texts)})"
            )

        clip_score = compute_clip_score(
            image_paths=image_files,
            graph_texts=graph_texts,
        )
        metrics["clip_score"] = clip_score
    except Exception as e:
        metrics["clip_score_error"] = str(e)
        print(f"[WARN] CLIP score failed: {e}")

    # -------------------------
    # SPAN
    # -------------------------
    span_requested = (
        args.span_repo_root != ""
        and args.span_ckpt != ""
        and args.span_img_folder != ""
        and args.span_prediction_records_json != ""
    )

    if span_requested:
        try:
            prediction_records = load_span_prediction_records(
                args.span_prediction_records_json
            )

            span_metrics = compute_span_r_precision(
                span_repo_root=args.span_repo_root,
                span_ckpt_path=args.span_ckpt,
                span_img_folder=args.span_img_folder,
                prediction_records=prediction_records,
                node_bbox=args.span_node_bbox,
                batch_size=args.span_batch_size,
                image_layer_num=args.span_image_layer_num,
                graph_layer_num=args.span_graph_layer_num,
                python_exec=args.python_exec,
            )
            metrics.update(span_metrics)

        except Exception as e:
            metrics["span_error"] = str(e)
            print(f"[WARN] SPAN failed: {e}")
    else:
        metrics["span_skipped"] = True

    out_path = os.path.join(args.out_dir, "image_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== IMAGE METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()