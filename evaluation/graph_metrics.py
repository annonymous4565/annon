from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Optional, Set

import numpy as np

@dataclass
class SceneGraphSample:
    """
    nodes: list[str]
    triplets: list of tuples in your decoded format:
        (s_idx, s_name, rel, o_idx, o_name)
    """
    nodes: List[str]
    triplets: List[Tuple[int, str, str, int, str]]


def canonical_graph_hash(g: SceneGraphSample) -> str:
    node_part = tuple((i, str(n)) for i, n in enumerate(g.nodes) if str(n) != "__pad__")
    trip_part = tuple(
        sorted((int(s), str(sn), str(r), int(o), str(on)) for s, sn, r, o, on in g.triplets)
    )
    payload = repr((node_part, trip_part)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def valid_node_indices(g: SceneGraphSample) -> List[int]:
    return [i for i, n in enumerate(g.nodes) if str(n) != "__pad__"]


def object_labels(g: SceneGraphSample) -> List[str]:
    return [str(g.nodes[i]) for i in valid_node_indices(g)]


def relation_labels(g: SceneGraphSample) -> List[str]:
    return [str(rel) for _, _, rel, _, _ in g.triplets]


def triplet_labels(g: SceneGraphSample) -> List[Tuple[str, str, str]]:
    return [(str(sn), str(rel), str(on)) for _, sn, rel, _, on in g.triplets]


def node_count(g: SceneGraphSample) -> int:
    return len(valid_node_indices(g))


def edge_count(g: SceneGraphSample) -> int:
    return len(g.triplets)


def in_degrees(g: SceneGraphSample) -> Dict[int, int]:
    deg = {i: 0 for i in valid_node_indices(g)}
    for s, _, _, o, _ in g.triplets:
        if o in deg:
            deg[o] += 1
    return deg


def out_degrees(g: SceneGraphSample) -> Dict[int, int]:
    deg = {i: 0 for i in valid_node_indices(g)}
    for s, _, _, o, _ in g.triplets:
        if s in deg:
            deg[s] += 1
    return deg


def degree_histogram(deg_dict: Dict[int, int]) -> np.ndarray:
    vals = list(deg_dict.values())
    if len(vals) == 0:
        return np.array([1.0], dtype=np.float64)
    max_deg = max(vals)
    hist = np.zeros(max_deg + 1, dtype=np.float64)
    for v in vals:
        hist[v] += 1.0
    hist /= max(hist.sum(), 1.0)
    return hist


def tv_distance_from_counters(a: Counter, b: Counter, vocab: Sequence) -> float:
    pa = np.array([float(a.get(v, 0.0)) for v in vocab], dtype=np.float64)
    pb = np.array([float(b.get(v, 0.0)) for v in vocab], dtype=np.float64)
    pa /= max(pa.sum(), 1.0)
    pb /= max(pb.sum(), 1.0)
    return float(0.5 * np.abs(pa - pb).sum())


def pad_to_same(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = max(len(x), len(y))
    if len(x) < m:
        x = np.pad(x, (0, m - len(x)))
    if len(y) < m:
        y = np.pad(y, (0, m - len(y)))
    return x, y


def gaussian_tv_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    x, y = pad_to_same(x, y)
    x = x / max(x.sum(), 1.0)
    y = y / max(y.sum(), 1.0)
    tv = 0.5 * np.abs(x - y).sum()
    return float(np.exp(-(tv ** 2) / (2.0 * sigma * sigma)))


def compute_mmd(
    samples_a: List[np.ndarray],
    samples_b: List[np.ndarray],
    kernel=gaussian_tv_kernel,
    sigma: float = 1.0,
) -> float:
    if len(samples_a) == 0 or len(samples_b) == 0:
        return 0.0

    def mean_kernel(xs, ys):
        total = 0.0
        for x in xs:
            for y in ys:
                total += kernel(x, y, sigma=sigma)
        return total / (len(xs) * len(ys))

    return float(
        mean_kernel(samples_a, samples_a)
        + mean_kernel(samples_b, samples_b)
        - 2.0 * mean_kernel(samples_a, samples_b)
    )


def graph_validity_rate(
    graphs: Sequence[SceneGraphSample],
    allow_self_loops: bool = False,
) -> float:
    valid_count = 0
    for g in graphs:
        valid_ids = set(valid_node_indices(g))
        ok = True
        for s, _, _, o, _ in g.triplets:
            if s not in valid_ids or o not in valid_ids:
                ok = False
                break
            if (not allow_self_loops) and s == o:
                ok = False
                break
        if ok:
            valid_count += 1
    return float(valid_count / max(len(graphs), 1))


def uniqueness_ratio(graphs: Sequence[SceneGraphSample]) -> float:
    hashes = [canonical_graph_hash(g) for g in graphs]
    return float(len(set(hashes)) / max(len(hashes), 1))


def novelty_ratio(
    generated_graphs: Sequence[SceneGraphSample],
    reference_graphs: Sequence[SceneGraphSample],
) -> float:
    ref_hashes = {canonical_graph_hash(g) for g in reference_graphs}
    novel = sum(canonical_graph_hash(g) not in ref_hashes for g in generated_graphs)
    return float(novel / max(len(generated_graphs), 1))


def triplet_diversity(graphs: Sequence[SceneGraphSample]) -> float:
    if len(graphs) < 2:
        return 0.0

    trip_sets = [set(triplet_labels(g)) for g in graphs]
    vals = []
    for i in range(len(trip_sets)):
        for j in range(i + 1, len(trip_sets)):
            a, b = trip_sets[i], trip_sets[j]
            union = len(a | b)
            inter = len(a & b)
            vals.append(1.0 - (inter / union if union > 0 else 1.0))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_graph_generation(
    generated_graphs: Sequence[SceneGraphSample],
    reference_graphs: Sequence[SceneGraphSample],
    include_motif_metrics: bool = False,
    topk_triplets_k: int = 50,
    out_degree_thresh: int = 3,
    in_degree_thresh: int = 3,
) -> Dict[str, float]:
    gen_node_hists = [np.bincount([node_count(g)], minlength=max(node_count(g) + 1, 2)).astype(np.float64)
                      for g in generated_graphs]
    ref_node_hists = [np.bincount([node_count(g)], minlength=max(node_count(g) + 1, 2)).astype(np.float64)
                      for g in reference_graphs]

    gen_edge_hists = [np.bincount([edge_count(g)], minlength=max(edge_count(g) + 1, 2)).astype(np.float64)
                      for g in generated_graphs]
    ref_edge_hists = [np.bincount([edge_count(g)], minlength=max(edge_count(g) + 1, 2)).astype(np.float64)
                      for g in reference_graphs]

    gen_out_deg = [degree_histogram(out_degrees(g)) for g in generated_graphs]
    ref_out_deg = [degree_histogram(out_degrees(g)) for g in reference_graphs]
    gen_in_deg = [degree_histogram(in_degrees(g)) for g in generated_graphs]
    ref_in_deg = [degree_histogram(in_degrees(g)) for g in reference_graphs]

    gen_obj_counter = Counter()
    ref_obj_counter = Counter()
    gen_rel_counter = Counter()
    ref_rel_counter = Counter()
    gen_trip_counter = Counter()
    ref_trip_counter = Counter()

    for g in generated_graphs:
        gen_obj_counter.update(object_labels(g))
        gen_rel_counter.update(relation_labels(g))
        gen_trip_counter.update(triplet_labels(g))

    for g in reference_graphs:
        ref_obj_counter.update(object_labels(g))
        ref_rel_counter.update(relation_labels(g))
        ref_trip_counter.update(triplet_labels(g))

    obj_vocab = sorted(set(gen_obj_counter.keys()) | set(ref_obj_counter.keys()))
    rel_vocab = sorted(set(gen_rel_counter.keys()) | set(ref_rel_counter.keys()))
    trip_vocab = sorted(set(gen_trip_counter.keys()) | set(ref_trip_counter.keys()))

    metrics = {
        "valid_graph_rate": graph_validity_rate(generated_graphs),
        "node_count_mmd": compute_mmd(gen_node_hists, ref_node_hists),
        "edge_count_mmd": compute_mmd(gen_edge_hists, ref_edge_hists),
        "in_degree_mmd": compute_mmd(gen_in_deg, ref_in_deg),
        "out_degree_mmd": compute_mmd(gen_out_deg, ref_out_deg),
        "object_label_tv": tv_distance_from_counters(gen_obj_counter, ref_obj_counter, obj_vocab),
        "relation_label_tv": tv_distance_from_counters(gen_rel_counter, ref_rel_counter, rel_vocab),
        "triplet_label_tv": tv_distance_from_counters(gen_trip_counter, ref_trip_counter, trip_vocab),
        "uniqueness_ratio": uniqueness_ratio(generated_graphs),
        "novelty_ratio": novelty_ratio(generated_graphs, reference_graphs),
        "triplet_diversity": triplet_diversity(generated_graphs),
    }

    if include_motif_metrics:
        motif_metrics = compute_motif_realism_metrics(
            generated_graphs=generated_graphs,
            reference_graphs=reference_graphs,
            topk_triplets_k=topk_triplets_k,
            out_degree_thresh=out_degree_thresh,
            in_degree_thresh=in_degree_thresh,
        )
        metrics.update(motif_metrics)

    return metrics

def subject_object_degree_pairs(g: SceneGraphSample) -> List[Tuple[int, int]]:
    """
    For each triplet (s, r, o), collect:
      (out_degree(subject), in_degree(object))
    This captures simple hub / attachment structure.
    """
    out_deg = out_degrees(g)
    in_deg = in_degrees(g)

    pairs = []
    for s, _, _, o, _ in g.triplets:
        pairs.append((int(out_deg.get(s, 0)), int(in_deg.get(o, 0))))
    return pairs



def hub_signature_labels(
    g: SceneGraphSample,
    out_degree_thresh: int = 3,
    in_degree_thresh: int = 3,
) -> List[Tuple[str, str]]:
    """
    Coarse motif labels for high-degree nodes:
      ("hub_out", object_label) for nodes with high outgoing degree
      ("hub_in",  object_label) for nodes with high incoming degree
    """
    out_deg = out_degrees(g)
    in_deg = in_degrees(g)

    labels = []
    for i in valid_node_indices(g):
        obj = str(g.nodes[i])
        if out_deg.get(i, 0) >= out_degree_thresh:
            labels.append(("hub_out", obj))
        if in_deg.get(i, 0) >= in_degree_thresh:
            labels.append(("hub_in", obj))
    return labels



def attachment_motif_labels(g: SceneGraphSample) -> List[Tuple[str, str]]:
    """
    Edge-local structural motif:
      (relation_label, 'leaf_object') if object indegree == 1 and outdegree == 0
      (relation_label, 'nonleaf_object') otherwise

    This helps distinguish 'attribute-like attachments' from more structural links.
    """
    out_deg = out_degrees(g)
    in_deg = in_degrees(g)

    motifs = []
    for _, _, rel, o, _ in g.triplets:
        obj_out = out_deg.get(o, 0)
        obj_in = in_deg.get(o, 0)

        kind = "leaf_object" if (obj_in == 1 and obj_out == 0) else "nonleaf_object"
        motifs.append((str(rel), kind))
    return motifs



def topk_triplet_coverage(
    generated_graphs: Sequence[SceneGraphSample],
    reference_graphs: Sequence[SceneGraphSample],
    k: int = 50,
) -> float:
    """
    Fraction of reference top-k frequent triplets that appear at least once
    in generated graphs.
    """
    ref_counter = Counter()
    gen_counter = Counter()

    for g in reference_graphs:
        ref_counter.update(triplet_labels(g))
    for g in generated_graphs:
        gen_counter.update(triplet_labels(g))

    if len(ref_counter) == 0:
        return 0.0

    ref_topk = [lab for lab, _ in ref_counter.most_common(k)]
    covered = sum(1 for lab in ref_topk if gen_counter.get(lab, 0) > 0)
    return float(covered / max(len(ref_topk), 1))



def motif_tv_distance_from_counters(
    gen_counter: Counter,
    ref_counter: Counter,
) -> float:
    vocab = sorted(set(gen_counter.keys()) | set(ref_counter.keys()))
    return tv_distance_from_counters(gen_counter, ref_counter, vocab)



def compute_motif_realism_metrics(
    generated_graphs: Sequence[SceneGraphSample],
    reference_graphs: Sequence[SceneGraphSample],
    topk_triplets_k: int = 50,
    out_degree_thresh: int = 3,
    in_degree_thresh: int = 3,
) -> Dict[str, float]:
    """
    Phase 6C.2:

frequent triplet coverage
hub pattern similarity
attachment motif similarity
degree-pair motif similarity
    Lower TV is better. Higher coverage is better.
    """
    gen_hub_counter = Counter()
    ref_hub_counter = Counter()

    gen_attach_counter = Counter()
    ref_attach_counter = Counter()

    gen_degpair_counter = Counter()
    ref_degpair_counter = Counter()

    for g in generated_graphs:
        gen_hub_counter.update(
            hub_signature_labels(
                g,
                out_degree_thresh=out_degree_thresh,
                in_degree_thresh=in_degree_thresh,
            )
        )
        gen_attach_counter.update(attachment_motif_labels(g))
        gen_degpair_counter.update(subject_object_degree_pairs(g))

    for g in reference_graphs:
        ref_hub_counter.update(
            hub_signature_labels(
                g,
                out_degree_thresh=out_degree_thresh,
                in_degree_thresh=in_degree_thresh,
            )
        )
        ref_attach_counter.update(attachment_motif_labels(g))
        ref_degpair_counter.update(subject_object_degree_pairs(g))

    return {
        "topk_triplet_coverage": topk_triplet_coverage(
            generated_graphs,
            reference_graphs,
            k=topk_triplets_k,
        ),
        "hub_signature_tv": motif_tv_distance_from_counters(gen_hub_counter, ref_hub_counter),
        "attachment_motif_tv": motif_tv_distance_from_counters(gen_attach_counter, ref_attach_counter),
        "degree_pair_tv": motif_tv_distance_from_counters(gen_degpair_counter, ref_degpair_counter),
    }