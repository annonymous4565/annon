import os
import ast
import json
import pickle
import argparse
from typing import List, Tuple

import numpy as np

from evaluation.graph_metrics import (
    SceneGraphSample,
    evaluate_graph_generation,
)

from datasets_.visual_genome.dataset import (
    SceneGraphDataset,
    decode_item,
)

DATASET = "coco_stuff"

def change_format(graph, ind_to_classes, ind_to_predicates):
    """
    graph: [X, F]
      X: object ids, usually 1-based
      F: relation matrix, relation ids, with 51 treated as no relation
    """
    X, F = graph
    X = list(X)

    F = np.array(F).copy()
    F[F == 51] = 0

    objs = [ind_to_classes[int(x) - 1] for x in X]

    to_idx_lst, from_idx_lst = F.nonzero()

    triples = []
    for to_idx, from_idx in zip(to_idx_lst, from_idx_lst):
        rel_id = int(F[to_idx, from_idx])
        if rel_id <= 0:
            continue

        rel = ind_to_predicates[rel_id - 1]
        triples.append([int(to_idx), str(rel), int(from_idx)])

    return objs, triples


def external_graph_to_scene_graph_sample(graph, ind_to_classes, ind_to_predicates):
    objs, triples_raw = change_format(graph, ind_to_classes, ind_to_predicates)

    triplets = []
    for s_idx, rel, o_idx in triples_raw:
        if s_idx < 0 or s_idx >= len(objs):
            continue
        if o_idx < 0 or o_idx >= len(objs):
            continue

        s_name = str(objs[s_idx])
        o_name = str(objs[o_idx])

        triplets.append(
            (
                int(s_idx),
                s_name,
                str(rel),
                int(o_idx),
                o_name,
            )
        )

    return SceneGraphSample(
        nodes=[str(x) for x in objs],
        triplets=triplets,
    )


def load_external_generated_graphs(
    generated_graphs_p: str,
    categories_p: str,
) -> List[SceneGraphSample]:
    ind_to_classes, ind_to_predicates, _ = pickle.load(open(categories_p, "rb"))

    generated_graphs = pickle.load(open(generated_graphs_p, "rb"))

    out = []
    for graph in generated_graphs:
        out.append(
            external_graph_to_scene_graph_sample(
                graph=graph,
                ind_to_classes=ind_to_classes,
                ind_to_predicates=ind_to_predicates,
            )
        )

    return out


def parse_sg_tuple_txt(path: str) -> SceneGraphSample:
    """
    Parses files with one tuple per line:
      (s_idx, 's_name', 'rel', o_idx, 'o_name')
    """
    nodes = {}
    triplets = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line == "(none)":
                continue

            t = ast.literal_eval(line)
            s_idx, s_name, rel, o_idx, o_name = t

            s_idx = int(s_idx)
            o_idx = int(o_idx)

            nodes[s_idx] = str(s_name)
            nodes[o_idx] = str(o_name)

            triplets.append(
                (
                    s_idx,
                    str(s_name),
                    str(rel),
                    o_idx,
                    str(o_name),
                )
            )

    if len(nodes) == 0:
        return SceneGraphSample(nodes=[], triplets=[])

    max_idx = max(nodes.keys())
    node_list = ["__pad__"] * (max_idx + 1)

    for idx, name in nodes.items():
        node_list[idx] = name

    return SceneGraphSample(nodes=node_list, triplets=triplets)


def load_reference_graphs_from_sg_dir(reference_sg_dir: str) -> List[SceneGraphSample]:
    files = [
        f for f in os.listdir(reference_sg_dir)
        if f.startswith("sg_") and f.endswith(".txt")
    ]
    files = sorted(files)

    graphs = []
    for name in files:
        graphs.append(parse_sg_tuple_txt(os.path.join(reference_sg_dir, name)))

    return graphs
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

def load_reference_graphs_from_npz(reference_npz: str) -> List[SceneGraphSample]:
    dataset = SceneGraphDataset(
        npz_path=reference_npz,
        return_boxes=False,
        return_metadata=False,
    )

    graphs = []

    for i in range(len(dataset)):
        item = dataset[i]

        nodes, triplets = decode_item(
            obj_labels=item["obj_labels"],
            rel_labels=item["rel_labels"],
            node_mask=item["node_mask"],
            edge_mask=item["edge_mask"],
            object_vocab=dataset.object_vocab,
            relation_vocab=dataset.relation_vocab,
            no_rel_token="__no_relation__",
        )

        graphs.append(
            SceneGraphSample(
                nodes=[str(x) for x in nodes],
                triplets=[
                    (
                        int(s_idx),
                        str(s_name),
                        str(rel),
                        int(o_idx),
                        str(o_name),
                    )
                    for s_idx, s_name, rel, o_idx, o_name in triplets
                ],
            )
        )

    return graphs


def maybe_limit(graphs: List[SceneGraphSample], max_graphs: int):
    if max_graphs is None or max_graphs <= 0:
        return graphs
    return graphs[:max_graphs]


def save_metrics(metrics: dict, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    txt_path = os.path.join(output_path, "graph_metrics_summary.txt")
    json_path = os.path.join(output_path, "graph_metrics_summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved metrics txt:  {txt_path}")
    print(f"[OK] Saved metrics json: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate external generated scene graphs against project graph metrics."
    )

    parser.add_argument(
        "--generated_graphs_p",
        type=str,
        # required=True,
        help="Path to generated_objects.p. Each element should be [X, F].",
        default="../SG_baselines/usggen/generated_samples/coco_stuff/new_usggen_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710/generated_objects.p"
    )

    parser.add_argument(
        "--categories_p",
        type=str,
        default="../SG_baselines/usggen/data/coco_stuff/categories.p",
        help="Path to categories.p containing ind_to_classes, ind_to_predicates, _.",
    )

    parser.add_argument(
        "--reference_npz",
        type=str,
        default=f"./data/visual_genome/processed/vg_sg_degree_filter_new_test_nmax20.npz",
        help="Optional reference VG npz path.",
    )

    parser.add_argument(
        "--npz_path",
        type=str,
        # required=True,
        help="Path to file",
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/exp/edm_diffuse_sg_regular/{DATASET}_diffuse_sg_edm_self-cond-ON_feat_dim_96_window_10_patch_1_node_bits_edge_bits_sample_1000_Apr-27-11-57-46/sampling_during_evaluation/coco_stuff_200.pth_weight_model_model_pure_noise_exp_eval_Apr-27-11-57-49_model_inference/final_samples_array_before_eval.npz",
        )
    parser.add_argument(
        "--idx_to_word_path",
        type=str,
        # required=True,
        help="Path to file",
        default=f"../SG_baselines/DiffuseSG/DiffuseSG/data_scenegraph/{DATASET}/idx_to_word.pkl"
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
        "--reference_sg_dir",
        type=str,
        default="",
        help="Optional reference SG txt folder containing sg_XXXXXX.txt files.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        # required=True,
        help="Folder to save graph_metrics_summary.txt/json.",
        default=f'./output/SGgraph_baselines/SceneGraphGen/{DATASET}'
    )

    parser.add_argument(
        "--max_generated_graphs",
        type=int,
        default=0,
        help="Optional cap on generated graphs. 0 means all.",
    )

    parser.add_argument(
        "--max_reference_graphs",
        type=int,
        default=0,
        help="Optional cap on reference graphs. 0 means all.",
    )

    parser.add_argument(
        "--include_motif_metrics",
        action="store_true",
        help="Also compute optional motif realism metrics.",
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

    return parser.parse_args()


def main():
    args = parse_args()

    generated_graphs = load_external_generated_graphs(
        generated_graphs_p=args.generated_graphs_p,
        categories_p=args.categories_p,
    )
    generated_graphs = maybe_limit(generated_graphs, args.max_generated_graphs)

    if args.reference_sg_dir:
        reference_graphs = load_reference_graphs_from_sg_dir(args.reference_sg_dir)
    elif args.reference_npz:
        # reference_graphs = load_reference_graphs_from_npz(args.reference_npz)
        _, reference_graphs = load_graphs_from_npz(
        npz_path=args.npz_path,
        idx_to_word_path=args.idx_to_word_path,
        no_relation_id=args.no_relation_id,
        allow_self_edges=args.allow_self_edges,
    )
    else:
        raise ValueError(
            "You must provide either --reference_sg_dir or --reference_npz."
        )

    reference_graphs = maybe_limit(reference_graphs, args.max_reference_graphs)

    print(f"[INFO] Generated graphs: {len(generated_graphs)}")
    print(f"[INFO] Reference graphs: {len(reference_graphs)}")

    metrics = evaluate_graph_generation(
        generated_graphs=generated_graphs,
        reference_graphs=reference_graphs,
        include_motif_metrics=args.include_motif_metrics,
        topk_triplets_k=args.topk_triplets_k,
        out_degree_thresh=args.out_degree_thresh,
        in_degree_thresh=args.in_degree_thresh,
    )

    save_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()