# scripts/Graph_baselines/evaluate_varscene.py

import os
import json
import pickle
import argparse
from typing import List

from evaluation.graph_metrics import SceneGraphSample, evaluate_graph_generation

DATASET="visual_genome"

def load_graphs(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def nx_multigraph_to_scene_graph(g) -> SceneGraphSample:
    # Stable node ordering
    node_ids = list(g.nodes())

    old_to_new = {old: i for i, old in enumerate(node_ids)}

    nodes = []
    for old in node_ids:
        attrs = g.nodes[old]
        label = attrs.get("label", attrs.get("name", str(old)))
        nodes.append(str(label))

    triplets = []

    # MultiGraph edges may be (u,v,key,data) if keys=True
    if g.is_multigraph():
        edge_iter = g.edges(keys=True, data=True)
        for u, v, _key, attrs in edge_iter:
            if u not in old_to_new or v not in old_to_new:
                continue

            rel = attrs.get("label", attrs.get("relation", attrs.get("predicate", "edge")))

            s_idx = old_to_new[u]
            o_idx = old_to_new[v]

            triplets.append(
                (
                    s_idx,
                    nodes[s_idx],
                    str(rel),
                    o_idx,
                    nodes[o_idx],
                )
            )
    else:
        for u, v, attrs in g.edges(data=True):
            if u not in old_to_new or v not in old_to_new:
                continue

            rel = attrs.get("label", attrs.get("relation", attrs.get("predicate", "edge")))

            s_idx = old_to_new[u]
            o_idx = old_to_new[v]

            triplets.append(
                (
                    s_idx,
                    nodes[s_idx],
                    str(rel),
                    o_idx,
                    nodes[o_idx],
                )
            )

    return SceneGraphSample(nodes=nodes, triplets=triplets)


def load_scene_graph_samples(path: str, max_graphs: int = 0) -> List[SceneGraphSample]:
    graphs = load_graphs(path)

    if max_graphs and max_graphs > 0:
        graphs = graphs[:max_graphs]

    return [nx_multigraph_to_scene_graph(g) for g in graphs]


def inspect_graph_file(path: str, num_graphs: int = 3):
    graphs = load_graphs(path)

    print(f"path: {path}")
    print(f"num graphs: {len(graphs)}")

    for i, g in enumerate(graphs[:num_graphs]):
        print("\n" + "=" * 80)
        print(f"graph {i}")
        print("type:", type(g))
        print("is_multigraph:", g.is_multigraph())
        print("num_nodes:", g.number_of_nodes())
        print("num_edges:", g.number_of_edges())

        print("\nNodes:")
        for n, attrs in list(g.nodes(data=True))[:20]:
            print(f"  {n}: {attrs}")

        print("\nEdges:")
        if g.is_multigraph():
            for u, v, k, attrs in list(g.edges(keys=True, data=True))[:30]:
                print(f"  {u} -- {v} key={k}: {attrs}")
        else:
            for u, v, attrs in list(g.edges(data=True))[:30]:
                print(f"  {u} -- {v}: {attrs}")


def save_debug_graphs(graphs: List[SceneGraphSample], output_dir: str, prefix: str, n: int = 5):
    debug_dir = os.path.join(output_dir, "debug_graphs")
    os.makedirs(debug_dir, exist_ok=True)

    for i, g in enumerate(graphs[:n]):
        path = os.path.join(debug_dir, f"{prefix}_{i:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Nodes:\n")
            for idx, label in enumerate(g.nodes):
                f.write(f"  [{idx:02d}] {label}\n")

            f.write("Triplets:\n")
            if len(g.triplets) == 0:
                f.write("  (none)\n")
            else:
                for s_idx, s_name, rel, o_idx, o_name in g.triplets:
                    f.write(f"  [{s_idx:02d}] {s_name} --{rel}--> [{o_idx:02d}] {o_name}\n")


def save_metrics(metrics, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "graph_metrics_summary.txt")
    json_path = os.path.join(output_dir, "graph_metrics_summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] saved {txt_path}")
    print(f"[OK] saved {json_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--generated_pkl", default="../SG_baselines/varscene/ckpt/unc_mmd_log_model_1000.0_star_0.001_generated.pkl")
    parser.add_argument("--reference_pkl", default="../SG_baselines/varscene/vg_data/data/graphs_train.pkl")
    parser.add_argument("--output_path", default=f'./output/SGgraph_baselines/varscene/{DATASET}')

    parser.add_argument("--max_generated_graphs", type=int, default=0)
    parser.add_argument("--max_reference_graphs", type=int, default=0)
    parser.add_argument("--include_motif_metrics", action="store_true", default=True)

    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--inspect_num_graphs", type=int, default=3)
    parser.add_argument("--save_debug_graphs", action="store_true")
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

    args = parser.parse_args()

    if args.inspect_only:
        inspect_graph_file(args.generated_pkl, num_graphs=args.inspect_num_graphs)
        return

    generated_graphs = load_scene_graph_samples(
        args.generated_pkl,
        max_graphs=args.max_generated_graphs,
    )

    reference_graphs = load_scene_graph_samples(
        args.reference_pkl,
        max_graphs=args.max_reference_graphs,
    )

    print("generated graphs:", len(generated_graphs))
    print("reference graphs:", len(reference_graphs))
    print("avg generated edges:", sum(len(g.triplets) for g in generated_graphs) / max(len(generated_graphs), 1))
    print("avg reference edges:", sum(len(g.triplets) for g in reference_graphs) / max(len(reference_graphs), 1))

    if args.save_debug_graphs:
        save_debug_graphs(generated_graphs, args.output_path, "generated")
        save_debug_graphs(reference_graphs, args.output_path, "reference")

    metrics = evaluate_graph_generation(
        generated_graphs=generated_graphs[:1000],
        reference_graphs=reference_graphs[:5000],
        include_motif_metrics=args.include_motif_metrics,
        topk_triplets_k=args.topk_triplets_k,
        out_degree_thresh=args.out_degree_thresh,
        in_degree_thresh=args.in_degree_thresh,
    )

    print("\n=== Graph metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()