import os
import json
import pickle
import argparse
from typing import List

import numpy as np

from evaluation.graph_metrics import (
    SceneGraphSample,
    evaluate_graph_generation,
)

DATASET="coco_stuff"

def load_idx_to_word(idx_to_word_path: str):
    with open(idx_to_word_path, "rb") as f:
        idx_to_word = pickle.load(f)

    if "ind_to_classes" not in idx_to_word:
        raise KeyError("idx_to_word.pkl must contain key 'ind_to_classes'")
    if "ind_to_predicates" not in idx_to_word:
        raise KeyError("idx_to_word.pkl must contain key 'ind_to_predicates'")

    return idx_to_word


def safe_lookup(vocab, idx: int, fallback_prefix: str):
    try:
        return str(vocab[int(idx)])
    except Exception:
        return f"{fallback_prefix}_{int(idx)}"


def build_graphs_from_arrays(
    X: np.ndarray,
    A: np.ndarray,
    node_flags: np.ndarray,
    ind_to_classes,
    ind_to_predicates,
    no_relation_id: int = 0,
    allow_self_edges: bool = False,
) -> List[SceneGraphSample]:
    """
    X:          [B, N] object ids
    A:          [B, N, N] relation ids, A[i,j] is i -> j
    node_flags: [B, N] valid-node mask

    Based on your visualization code:
      object name = ind_to_classes[X[i]]
      relation   = ind_to_predicates[A[i,j]]
      A[i,j] <= 0 means no relation
    """
    graphs = []

    B, N = X.shape

    for b in range(B):
        flags_b = node_flags[b].astype(bool)

        nodes = []
        for i in range(N):
            if flags_b[i]:
                nodes.append(safe_lookup(ind_to_classes, int(X[b, i]), "obj"))
            else:
                nodes.append("__pad__")

        triplets = []
        for i in range(N):
            if not flags_b[i]:
                continue
            for j in range(N):
                if not flags_b[j]:
                    continue
                if (not allow_self_edges) and i == j:
                    continue

                rel_id = int(A[b, i, j])
                if rel_id <= no_relation_id:
                    continue

                rel = safe_lookup(ind_to_predicates, rel_id, "rel")

                triplets.append(
                    (
                        int(i),
                        nodes[i],
                        rel,
                        int(j),
                        nodes[j],
                    )
                )

        graphs.append(SceneGraphSample(nodes=nodes, triplets=triplets))

    return graphs


def load_graphs_from_npz(
    npz_path: str,
    idx_to_word_path: str,
    no_relation_id: int = 0,
    allow_self_edges: bool = False,
):
    data = np.load(npz_path, allow_pickle=True)
    idx_to_word = load_idx_to_word(idx_to_word_path)

    required = [
        "samples_x",
        "samples_a",
        "samples_node_flags",
        "gt_x",
        "gt_a",
        "gt_node_flags",
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in npz: {npz_path}")

    ind_to_classes = idx_to_word["ind_to_classes"]
    ind_to_predicates = idx_to_word["ind_to_predicates"]

    generated_graphs = build_graphs_from_arrays(
        X=data["samples_x"],
        A=data["samples_a"],
        node_flags=data["samples_node_flags"],
        ind_to_classes=ind_to_classes,
        ind_to_predicates=ind_to_predicates,
        no_relation_id=no_relation_id,
        allow_self_edges=allow_self_edges,
    )

    reference_graphs = build_graphs_from_arrays(
        X=data["gt_x"],
        A=data["gt_a"],
        node_flags=data["gt_node_flags"],
        ind_to_classes=ind_to_classes,
        ind_to_predicates=ind_to_predicates,
        no_relation_id=no_relation_id,
        allow_self_edges=allow_self_edges,
    )

    return generated_graphs, reference_graphs


def save_metrics(metrics: dict, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    txt_path = os.path.join(output_path, "graph_metrics_summary.txt")
    json_path = os.path.join(output_path, "graph_metrics_summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved: {txt_path}")
    print(f"[OK] Saved: {json_path}")


def save_debug_graphs(graphs, output_path: str, prefix: str, max_graphs: int = 5):
    debug_dir = os.path.join(output_path, "debug_graphs")
    os.makedirs(debug_dir, exist_ok=True)

    for idx, g in enumerate(graphs[:max_graphs]):
        path = os.path.join(debug_dir, f"{prefix}_{idx:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Nodes:\n")
            for i, n in enumerate(g.nodes):
                if str(n) != "__pad__":
                    f.write(f"  [{i:02d}] {n}\n")

            f.write("Triplets:\n")
            if len(g.triplets) == 0:
                f.write("  (none)\n")
            else:
                for s_idx, s_name, rel, o_idx, o_name in g.triplets:
                    f.write(
                        f"  [{s_idx:02d}] {s_name} --{rel}--> "
                        f"[{o_idx:02d}] {o_name}\n"
                    )


def print_quick_stats(name: str, graphs: List[SceneGraphSample]):
    node_counts = [sum(str(n) != "__pad__" for n in g.nodes) for g in graphs]
    edge_counts = [len(g.triplets) for g in graphs]

    print(f"\n=== {name} ===")
    print(f"num_graphs: {len(graphs)}")
    print(f"node_count mean: {np.mean(node_counts):.3f}")
    print(f"node_count min/max: {np.min(node_counts)} / {np.max(node_counts)}")
    print(f"edge_count mean: {np.mean(edge_counts):.3f}")
    print(f"edge_count min/max: {np.min(edge_counts)} / {np.max(edge_counts)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate scene graphs saved in final_samples_array_before_eval.npz."
    )

    parser.add_argument(
        "--npz_path",
        type=str,
        # required=True,
        help="Path to file",
        # default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_8_patch_1_node_bits_edge_bits_sample_1000_Apr-27-10-19-37/sampling_during_evaluation/{DATASET}_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-10-19-46_model_inference/final_samples_array_before_eval.npz"
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_10_patch_1_node_bits_edge_bits_sample_1000_Apr-27-11-57-46/sampling_during_evaluation/coco_stuff_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-11-57-49_model_inference/final_samples_array_before_eval.npz"
    )
    parser.add_argument(
        "--idx_to_word_path",
        type=str,
        # required=True,
        help="Path to file",
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/idx_to_word.pkl"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        # required=True,
        help="Folder to save graph_metrics_summary.txt/json.",
        default=f'./output/SGgraph_baselines/DiffuseSG/{DATASET}'
    )

    parser.add_argument(
        "--max_graphs",
        type=int,
        default=0,
        help="Optional cap on number of generated/GT graphs. 0 means all.",
    )

    parser.add_argument(
        "--no_relation_id",
        type=int,
        default=0,
        help="Relation id treated as no edge/no relation.",
    )

    parser.add_argument(
        "--allow_self_edges",
        action="store_true",
        help="Keep self-loop edges if present.",
    )

    parser.add_argument(
        "--include_motif_metrics",
        action="store_true",
        help="Also compute optional motif metrics.",
        default=True
    )

    parser.add_argument(
        "--topk_triplets_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--out_degree_thresh",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--in_degree_thresh",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--save_debug_graphs",
        action="store_true",
        help="Save a few decoded generated and GT graphs as txt for inspection.",
    )

    parser.add_argument(
        "--debug_num_graphs",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generated_graphs, reference_graphs = load_graphs_from_npz(
        npz_path=args.npz_path,
        idx_to_word_path=args.idx_to_word_path,
        no_relation_id=args.no_relation_id,
        allow_self_edges=args.allow_self_edges,
    )

    if args.max_graphs > 0:
        generated_graphs = generated_graphs[: args.max_graphs]
        reference_graphs = reference_graphs[: args.max_graphs]

    print_quick_stats("Generated", generated_graphs)
    print_quick_stats("Reference", reference_graphs)

    if args.save_debug_graphs:
        save_debug_graphs(
            generated_graphs,
            output_path=args.output_path,
            prefix="generated",
            max_graphs=args.debug_num_graphs,
        )
        save_debug_graphs(
            reference_graphs,
            output_path=args.output_path,
            prefix="reference",
            max_graphs=args.debug_num_graphs,
        )

    metrics = evaluate_graph_generation(
        generated_graphs=generated_graphs,
        reference_graphs=reference_graphs,
        include_motif_metrics=args.include_motif_metrics,
        topk_triplets_k=args.topk_triplets_k,
        out_degree_thresh=args.out_degree_thresh,
        in_degree_thresh=args.in_degree_thresh,
    )

    # print("\n=== GRAPH METRICS ===")
    # for k, v in metrics.items():
    #     print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()