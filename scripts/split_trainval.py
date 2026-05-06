import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyrallis

from configs import DiscreteSGConfig

def _subset_blob(blob: dict, indices: np.ndarray) -> dict:
    out = {}

    # Per-example arrays: subset these
    per_example_keys = [
        "obj_labels",
        "rel_labels",
        "node_mask",
        "edge_mask",
        "boxes",
        "image_index",
        "kept_old_indices",
    ]

    # Metadata arrays: copy as-is if present
    metadata_keys = [
        "n_max",
        "object_vocab",
        "relation_vocab",
    ]

    for key in per_example_keys:
        if key in blob:
            out[key] = blob[key][indices]

    for key in metadata_keys:
        if key in blob:
            out[key] = blob[key]

    return out


def _save_npz(path: str, data: dict, split_name: str) -> None:
    payload = dict(data)
    payload["split"] = np.array([split_name], dtype=object)
    np.savez_compressed(path, **payload)


def _print_stats(name: str, data: dict) -> None:
    print(f"{name}:")
    print(f"  num_examples: {data['obj_labels'].shape[0]}")
    print(f"  obj_labels shape: {data['obj_labels'].shape}")
    print(f"  rel_labels shape: {data['rel_labels'].shape}")
    if "image_index" in data:
        print(f"  first 5 image_index: {data['image_index'][:5].tolist()}")


@pyrallis.wrap()
def main(opt: DiscreteSGConfig) -> None:
    os.makedirs(opt.output_dir, exist_ok=True)

    blob_npz = np.load(opt.processed_npz_path, allow_pickle=True)
    blob = {k: blob_npz[k] for k in blob_npz.files}

    num_examples = blob["obj_labels"].shape[0]
    if num_examples <= 1:
        raise ValueError("Need at least 2 examples to create a train/val split.")

    if not (0.0 < opt.val_ratio < 1.0):
        raise ValueError("val_ratio must be strictly between 0 and 1.")

    indices = np.arange(num_examples)
    if opt.shuffle:
        rng = np.random.default_rng(opt.split_seed)
        rng.shuffle(indices)

    num_val = int(round(num_examples * opt.val_ratio))
    num_val = max(1, num_val)
    num_val = min(num_examples - 1, num_val)

    val_indices = np.sort(indices[:num_val])
    train_indices = np.sort(indices[num_val:])

    train_data = _subset_blob(blob, train_indices)
    val_data = _subset_blob(blob, val_indices)

    train_path = os.path.join(opt.output_dir, opt.train_filename)
    val_path = os.path.join(opt.output_dir, opt.val_filename)

    _save_npz(train_path, train_data, split_name="train")
    _save_npz(val_path, val_data, split_name="val")

    print(f"Input file: {opt.processed_npz_path}")
    print(f"Total examples: {num_examples}")
    print(f"Split seed: {opt.split_seed}")
    print(f"Validation ratio: {opt.val_ratio:.4f}")
    print()
    _print_stats("Train split", train_data)
    print()
    _print_stats("Val split", val_data)
    print()
    print(f"Saved train split to: {train_path}")
    print(f"Saved val split to:   {val_path}")


if __name__ == "__main__":
    main()