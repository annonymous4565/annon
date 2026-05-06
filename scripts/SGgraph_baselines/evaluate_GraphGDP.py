import os
import re
import json
import pickle
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np

DATASET = "coco_stuff"
from evaluation.graph_metrics import (
    SceneGraphSample,
    evaluate_graph_generation,
)


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


def parse_GraphGDP_samples_txt(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parses GraphGDP generated txt format:

        N=6
        X:
        112 66 ...
        E:
        0 0 ...
        ...

    Returns:
        list of (X, E)
          X: [N]
          E: [N,N]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = re.split(r"\n\s*\n", raw.strip())
    graphs = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) == 0:
            continue

        n = None
        x = None
        e_rows = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("N="):
                n = int(line.split("=")[1].strip())
                i += 1
                continue

            if line.startswith("X"):
                i += 1
                if i >= len(lines):
                    raise ValueError(f"Malformed X block:\n{block}")
                x = np.array([int(v) for v in lines[i].split()], dtype=np.int64)
                i += 1
                continue

            if line.startswith("E"):
                i += 1
                if n is None:
                    raise ValueError(f"E block appeared before N:\n{block}")

                e_rows = []
                for _ in range(n):
                    if i >= len(lines):
                        raise ValueError(f"Malformed E block:\n{block}")
                    e_rows.append([int(v) for v in lines[i].split()])
                    i += 1
                continue

            i += 1

        if n is None or x is None or len(e_rows) == 0:
            continue

        e = np.array(e_rows, dtype=np.int64)

        if x.shape[0] != n:
            raise ValueError(f"X length {x.shape[0]} does not match N={n}")

        if e.shape != (n, n):
            raise ValueError(f"E shape {e.shape} does not match {(n, n)}")

        graphs.append((x, e))

    return graphs


def GraphGDP_arrays_to_scene_graphs(
    graph_arrays: List[Tuple[np.ndarray, np.ndarray]],
    idx_to_word: Dict[str, Any],
    no_relation_id: int = 0,
    allow_self_edges: bool = False,
    edge_id_shift: int = 0,
    node_id_shift: int = 0,
) -> List[SceneGraphSample]:
    """
    Converts parsed GraphGDP samples into SceneGraphSample.

    Assumptions:
      X[i] is object id.
      E[i,j] is relation id for directed edge i -> j.
      E[i,j] == no_relation_id means no edge.

    edge_id_shift:
      Use 0 if relation ids directly index ind_to_predicates.
      Use -1 if relation ids are 1-based and vocab is 0-based.
    node_id_shift:
      Use 0 if node ids directly index ind_to_classes.
      Use -1 if node ids are 1-based.
    """
    ind_to_classes = idx_to_word["ind_to_classes"]
    ind_to_predicates = idx_to_word["ind_to_predicates"]

    out = []

    for X, E in graph_arrays:
        n = len(X)

        nodes = []
        for i in range(n):
            obj_id = int(X[i]) + int(node_id_shift)
            nodes.append(safe_lookup(ind_to_classes, obj_id, "obj"))

        triplets = []
        for i in range(n):
            for j in range(n):
                if (not allow_self_edges) and i == j:
                    continue

                rel_id_raw = int(E[i, j])
                if rel_id_raw <= no_relation_id:
                    continue

                rel_id = rel_id_raw + int(edge_id_shift)
                rel = safe_lookup(ind_to_predicates, rel_id, "rel")

                triplets.append(
                    (
                        int(i),
                        nodes[i],
                        str(rel),
                        int(j),
                        nodes[j],
                    )
                )

        out.append(SceneGraphSample(nodes=nodes, triplets=triplets))

    return out


def load_reference_graphs_from_diffusesg_pkl(
    pkl_path: str,
    idx_to_word: Dict[str, Any],
    no_relation_id: int = 0,
    allow_self_edges: bool = False,
) -> List[SceneGraphSample]:
    """
    Loads DiffuseSG-format reference pkl.

    Each item:
        node_labels: [N]
        edge_map: [N,N]
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    ind_to_classes = idx_to_word["ind_to_classes"]
    ind_to_predicates = idx_to_word["ind_to_predicates"]

    graphs = []

    for item in data:
        X = np.asarray(item["node_labels"], dtype=np.int64)
        E = np.asarray(item["edge_map"], dtype=np.int64)

        n = len(X)

        nodes = [
            safe_lookup(ind_to_classes, int(X[i]), "obj")
            for i in range(n)
        ]

        triplets = []
        for i in range(n):
            for j in range(n):
                if (not allow_self_edges) and i == j:
                    continue

                rel_id = int(E[i, j])
                if rel_id <= no_relation_id:
                    continue

                rel = safe_lookup(ind_to_predicates, rel_id, "rel")
                triplets.append((i, nodes[i], rel, j, nodes[j]))

        graphs.append(SceneGraphSample(nodes=nodes, triplets=triplets))

    return graphs


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

    if len(graphs) == 0:
        return

    print(f"node_count mean: {np.mean(node_counts):.3f}")
    print(f"node_count min/max: {np.min(node_counts)} / {np.max(node_counts)}")
    print(f"edge_count mean: {np.mean(edge_counts):.3f}")
    print(f"edge_count min/max: {np.min(edge_counts)} / {np.max(edge_counts)}")


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GraphGDP generated scene graph txt samples using project graph metrics."
    )

    parser.add_argument(
        "--GraphGDP_samples_txt",
        type=str,
        # default="../SG_baselines/GraphGDP/outputs/2026-04-28/07-17-26-graph-tf-model/generated_samples1.txt",
        default="../SG_baselines/GraphGDP/outputs/2026-04-28/09-41-53-graph-tf-model/generated_samples1.txt",
        help="Path to GraphGDP generated txt samples.",
    )

    parser.add_argument(
        "--idx_to_word_path",
        type=str,
        # default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/idx_to_word.pkl",
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/idx_to_word.pkl",
        help="Path to DiffuseSG idx_to_word.pkl.",
    )

    parser.add_argument(
        "--reference_pkl",
        type=str,
        # default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/validation_data_bbox_dbox32_np.pkl",
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/coco_blt_validation_data_dbox32_np.pkl",
        help="Path to DiffuseSG validation/test pkl used as reference.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=f'./output/SGgraph_baselines/GraphGDP/{DATASET}',
        help="Directory to save metrics.",
    )

    parser.add_argument(
        "--max_generated_graphs",
        type=int,
        default=0,
        help="Optional cap on GraphGDP generated graphs. 0 means all.",
    )

    parser.add_argument(
        "--max_reference_graphs",
        type=int,
        default=0,
        help="Optional cap on reference graphs. 0 means all.",
    )

    parser.add_argument(
        "--no_relation_id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--allow_self_edges",
        action="store_true",
    )

    parser.add_argument(
        "--edge_id_shift",
        type=int,
        default=0,
        help="Use -1 if GraphGDP relation ids are 1-based but vocab is 0-based.",
    )

    parser.add_argument(
        "--node_id_shift",
        type=int,
        default=0,
        help="Use -1 if GraphGDP node ids are 1-based but vocab is 0-based.",
    )

    parser.add_argument(
        "--include_motif_metrics",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_debug_graphs",
        action="store_true",
    )

    parser.add_argument(
        "--debug_num_graphs",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    idx_to_word = load_idx_to_word(args.idx_to_word_path)

    GraphGDP_arrays = parse_GraphGDP_samples_txt(args.GraphGDP_samples_txt)

    generated_graphs = GraphGDP_arrays_to_scene_graphs(
        graph_arrays=GraphGDP_arrays,
        idx_to_word=idx_to_word,
        no_relation_id=args.no_relation_id,
        allow_self_edges=args.allow_self_edges,
        edge_id_shift=args.edge_id_shift,
        node_id_shift=args.node_id_shift,
    )

    reference_graphs = load_reference_graphs_from_diffusesg_pkl(
        pkl_path=args.reference_pkl,
        idx_to_word=idx_to_word,
        no_relation_id=args.no_relation_id,
        allow_self_edges=args.allow_self_edges,
    )

    if args.max_generated_graphs > 0:
        generated_graphs = generated_graphs[: args.max_generated_graphs]

    if args.max_reference_graphs > 0:
        reference_graphs = reference_graphs[: args.max_reference_graphs]

    print_quick_stats("GraphGDP generated", generated_graphs)
    print_quick_stats("DiffuseSG reference", reference_graphs)

    if args.save_debug_graphs:
        save_debug_graphs(
            generated_graphs,
            output_path=args.output_path,
            prefix="GraphGDP_generated",
            max_graphs=args.debug_num_graphs,
        )
        save_debug_graphs(
            reference_graphs,
            output_path=args.output_path,
            prefix="diffusesg_reference",
            max_graphs=args.debug_num_graphs,
        )

    metrics = evaluate_graph_generation(
        generated_graphs=generated_graphs,
        reference_graphs=reference_graphs,
        include_motif_metrics=args.include_motif_metrics,
    )

    print("\n=== GRAPH METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()