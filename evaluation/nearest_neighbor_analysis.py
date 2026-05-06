from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from evaluation.graph_metrics import SceneGraphSample



def _object_counter(g: SceneGraphSample) -> Counter:
    return Counter(str(n) for n in g.nodes if str(n) != "__pad__")



def _relation_counter(g: SceneGraphSample) -> Counter:
    return Counter(str(rel) for _, _, rel, _, _ in g.triplets)



def _triplet_counter(g: SceneGraphSample) -> Counter:
    return Counter((str(sn), str(rel), str(on)) for _, sn, rel, _, on in g.triplets)



def _normalized_l1_counter_distance(a: Counter, b: Counter) -> float:
    vocab = sorted(set(a.keys()) | set(b.keys()))
    if len(vocab) == 0:
        return 0.0

    va = np.array([float(a.get(k, 0.0)) for k in vocab], dtype=np.float64)
    vb = np.array([float(b.get(k, 0.0)) for k in vocab], dtype=np.float64)

    sa = va.sum()
    sb = vb.sum()

    if sa > 0:
        va = va / sa
    if sb > 0:
        vb = vb / sb

    return float(0.5 * np.abs(va - vb).sum())  # TV distance



def _safe_ratio_abs_diff(x: int, y: int, denom_floor: float = 1.0) -> float:
    denom = max(float(max(x, y)), denom_floor)
    return abs(float(x) - float(y)) / denom



def graph_nn_distance(
    gen_graph: SceneGraphSample,
    ref_graph: SceneGraphSample,
    w_obj: float = 0.20,
    w_rel: float = 0.15,
    w_trip: float = 0.35,
    w_node_count: float = 0.15,
    w_edge_count: float = 0.15,
) -> Dict[str, float]:
    """
    Lower is better.
    All components are in [0,1] approximately.
    """
    obj_dist = _normalized_l1_counter_distance(_object_counter(gen_graph), _object_counter(ref_graph))
    rel_dist = _normalized_l1_counter_distance(_relation_counter(gen_graph), _relation_counter(ref_graph))
    trip_dist = _normalized_l1_counter_distance(_triplet_counter(gen_graph), _triplet_counter(ref_graph))

    gen_num_nodes = sum(str(n) != "__pad__" for n in gen_graph.nodes)
    ref_num_nodes = sum(str(n) != "__pad__" for n in ref_graph.nodes)

    gen_num_edges = len(gen_graph.triplets)
    ref_num_edges = len(ref_graph.triplets)

    node_count_dist = _safe_ratio_abs_diff(gen_num_nodes, ref_num_nodes)
    edge_count_dist = _safe_ratio_abs_diff(gen_num_edges, ref_num_edges)

    total = (
        w_obj * obj_dist
        + w_rel * rel_dist
        + w_trip * trip_dist
        + w_node_count * node_count_dist
        + w_edge_count * edge_count_dist
    )

    return {
        "distance_total": float(total),
        "distance_obj": float(obj_dist),
        "distance_rel": float(rel_dist),
        "distance_triplet": float(trip_dist),
        "distance_node_count": float(node_count_dist),
        "distance_edge_count": float(edge_count_dist),
    }



def rank_nearest_neighbors(
    query_graph: SceneGraphSample,
    reference_graphs: Sequence[SceneGraphSample],
    top_k: int = 5,
) -> List[Dict]:
    scored = []
    for idx, ref in enumerate(reference_graphs):
        d = graph_nn_distance(query_graph, ref)
        rec = {"ref_index": idx, **d}
        scored.append(rec)

    scored.sort(key=lambda x: x["distance_total"])
    return scored[:top_k]



def summarize_nn_set(
    generated_graphs: Sequence[SceneGraphSample],
    reference_graphs: Sequence[SceneGraphSample],
    top_k: int = 1,
) -> Dict[str, float]:
    if len(generated_graphs) == 0 or len(reference_graphs) == 0:
        return {
            "nn_mean_distance": 0.0,
            "nn_median_distance": 0.0,
            "nn_mean_obj_distance": 0.0,
            "nn_mean_rel_distance": 0.0,
            "nn_mean_triplet_distance": 0.0,
            "nn_mean_node_count_distance": 0.0,
            "nn_mean_edge_count_distance": 0.0,
        }

    bests = []
    for g in generated_graphs:
        ranked = rank_nearest_neighbors(g, reference_graphs, top_k=max(top_k, 1))
        bests.append(ranked[0])

    return {
        "nn_mean_distance": float(np.mean([x["distance_total"] for x in bests])),
        "nn_median_distance": float(np.median([x["distance_total"] for x in bests])),
        "nn_mean_obj_distance": float(np.mean([x["distance_obj"] for x in bests])),
        "nn_mean_rel_distance": float(np.mean([x["distance_rel"] for x in bests])),
        "nn_mean_triplet_distance": float(np.mean([x["distance_triplet"] for x in bests])),
        "nn_mean_node_count_distance": float(np.mean([x["distance_node_count"] for x in bests])),
        "nn_mean_edge_count_distance": float(np.mean([x["distance_edge_count"] for x in bests])),
    }