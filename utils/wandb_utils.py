from dataclasses import asdict
from typing import Optional, List, Dict, Tuple

import os
import re
import html


try:
    from graphviz import Digraph
except Exception:
    Digraph = None


import torch

from training.distributed_utils import is_main_process

from utils.layout_vis import draw_layout_boxes

try:
    import wandb
except ImportError:
    wandb = None


# -----------------------------
# Parsing / formatting helpers
# -----------------------------

_NODE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+(.*)$")
_TRIPLET_LINE_RE = re.compile(
    r"^\s*\[(\d+)\]\s+(.*?)\s+--(.*?)-->\s+\[(\d+)\]\s+(.*)$"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-\.]+", "_", text)
    return text.strip("_")


def graphviz_available() -> bool:
    return Digraph is not None


def build_epoch_metrics_table(rows):
    """
    Supports both:
    1. old baseline schema
    2. new structured diffusion schema
    """
    if wandb is None or len(rows) == 0:
        return None

    sample = rows[0]

    # Old baseline schema
    if "rel_loss_weighted" in sample:
        table = wandb.Table(
            columns=[
                "epoch",
                "split",
                "loss",
                "obj_loss",
                "rel_loss_weighted",
                "rel_loss_unweighted",
                "rel_loss_positive_only",
                "rel_loss_negative_only",
            ]
        )

        for row in rows:
            table.add_data(
                row.get("epoch"),
                row.get("split"),
                row.get("loss"),
                row.get("obj_loss"),
                row.get("rel_loss_weighted"),
                row.get("rel_loss_unweighted"),
                row.get("rel_loss_positive_only"),
                row.get("rel_loss_negative_only"),
            )
        return table

    # New structured diffusion schema
    table = wandb.Table(
        columns=[
            "epoch",
            "split",
            "loss",
            "obj_loss",
            "edge_loss",
            "rel_loss",
            "gt_num_pos_edges",
            "pred_num_pos_edges",
            "pred_gt_edge_ratio",
            "tp_edges",
            "fp_edges",
            "fn_edges",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "relation_accuracy_on_true_positive_edges",
            "node_acc_all",
            "node_acc_corrupted",
        ]
    )

    for row in rows:
        table.add_data(
            row.get("epoch"),
            row.get("split"),
            row.get("loss"),
            row.get("obj_loss"),
            row.get("edge_loss"),
            row.get("rel_loss"),
            row.get("gt_num_pos_edges"),
            row.get("pred_num_pos_edges"),
            row.get("pred_gt_edge_ratio"),
            row.get("tp_edges"),
            row.get("fp_edges"),
            row.get("fn_edges"),
            row.get("edge_precision"),
            row.get("edge_recall"),
            row.get("edge_f1"),
            row.get("relation_accuracy_on_true_positive_edges"),
            row.get("node_acc_all"),
            row.get("node_acc_corrupted"),
        )


def build_graph_comparison_table(rows):
    """
    Existing text table helper, unchanged.
    rows: list of dicts with keys:
        example_id, timestep, clean_graph, noisy_graph, pred_graph
    """
    if wandb is None:
        return None

    table = wandb.Table(
        columns=["example_id", "timestep", "clean_graph", "noisy_graph", "pred_graph"]
    )

    for row in rows:
        table.add_data(
            row["example_id"],
            row["timestep"],
            row["clean_graph"],
            row["noisy_graph"],
            row["pred_graph"],
        )

    return table

def build_eval_graph_comparison_table(rows):
    """
    Existing text table helper, unchanged.
    rows: list of dicts with keys:
        example_id, timestep, clean_graph, start_noisy_graph, final_graph
    """
    if wandb is None:
        return None

    table = wandb.Table(
        columns=["example_id", "timestep", "clean_graph", "start_noisy_graph", "final_graph"]
    )

    for row in rows:
        table.add_data(
            row["example_id"],
            row["timestep"],
            row["clean_graph"],
            row["start_noisy_graph"],
            row["final_graph"],
        )

    return table


def build_graph_image_comparison_table(rows):
    """
    Image-based W&B table for clean / noisy / pred rendered graphs.
    Each row should have:
        example_id, timestep, clean_graph_img, noisy_graph_img, pred_graph_img
    """
    if wandb is None:
        return None

    table = wandb.Table(
        columns=["example_id", "timestep", "clean_graph", "noisy_graph", "pred_graph"]
    )

    for row in rows:
        table.add_data(
            row["example_id"],
            row["timestep"],
            wandb.Image(row["clean_graph_img"]) if row.get("clean_graph_img") else None,
            wandb.Image(row["noisy_graph_img"]) if row.get("noisy_graph_img") else None,
            wandb.Image(row["pred_graph_img"]) if row.get("pred_graph_img") else None,
        )

    return table


# -----------------------------
# Indexed graph rendering
# -----------------------------

def build_indexed_graph_data(
    nodes: List[str],
    triplets: List[Tuple[str, str, str]],
) -> Tuple[List[Dict], List[Tuple[int, str, int]]]:
    """
    Converts:
        nodes = ["man", "boy", "boy", ...]
        triplets = [("man", "on", "sidewalk"), ...]
    into indexed structures.

    NOTE:
    This only works reliably if labels are unique.
    Since you have repeated object labels, this helper is NOT the preferred route.
    It is included only for completeness.

    Preferred route is to use indexed triplets directly, or parse from your
    formatted decoded text where ids are already present.
    """
    indexed_nodes = [{"id": i, "label": label} for i, label in enumerate(nodes)]

    # fragile when repeated labels exist
    label_to_first_idx = {}
    for i, label in enumerate(nodes):
        if label not in label_to_first_idx:
            label_to_first_idx[label] = i

    indexed_triplets = []
    for subj_label, rel, obj_label in triplets:
        if subj_label not in label_to_first_idx or obj_label not in label_to_first_idx:
            continue
        indexed_triplets.append(
            (label_to_first_idx[subj_label], rel, label_to_first_idx[obj_label])
        )

    return indexed_nodes, indexed_triplets


def parse_graph_text_block(graph_text: str) -> Tuple[List[Dict], List[Tuple[int, str, int]]]:
    """
    Parses blocks like:

    Nodes:
      [00] sidewalk
      [01] street
    Triplets:
      [02] man --on--> [00] sidewalk

    Returns:
      nodes = [{"id": 0, "label": "sidewalk"}, ...]
      triplets = [(2, "on", 0), ...]
    """
    lines = graph_text.splitlines()

    nodes: List[Dict] = []
    triplets: List[Tuple[int, str, int]] = []

    in_nodes = False
    in_triplets = False

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.strip() == "Nodes:":
            in_nodes = True
            in_triplets = False
            continue

        if line.strip() == "Triplets:":
            in_nodes = False
            in_triplets = True
            continue

        if in_nodes:
            m = _NODE_LINE_RE.match(line)
            if m:
                node_id = int(m.group(1))
                label = m.group(2).strip()
                nodes.append({"id": node_id, "label": label})

        elif in_triplets:
            m = _TRIPLET_LINE_RE.match(line)
            if m:
                subj_id = int(m.group(1))
                rel = m.group(3).strip()
                obj_id = int(m.group(4))
                triplets.append((subj_id, rel, obj_id))

    return nodes, triplets


def parse_validation_log_text(full_text: str) -> Dict[str, str]:
    """
    Extracts CLEAN / NOISY / PREDICTED sections from your saved validation text.

    Returns dict with keys:
        diagnostics, clean, noisy, predicted
    """
    sections = {
        "diagnostics": "",
        "clean": "",
        "noisy": "",
        "predicted": "",
    }

    clean_match = re.search(
        r"=== CLEAN ===\n(.*?)(?=\n=== NOISY)", full_text, flags=re.DOTALL
    )
    noisy_match = re.search(
        r"=== NOISY.*?===\n(.*?)(?=\n=== PREDICTED x0 ===)", full_text, flags=re.DOTALL
    )
    pred_match = re.search(
        r"=== PREDICTED x0 ===\n(.*)$", full_text, flags=re.DOTALL
    )
    diag_match = re.search(
        r"^(.*?)(?=\n=== CLEAN ===)", full_text, flags=re.DOTALL
    )

    if diag_match:
        sections["diagnostics"] = diag_match.group(1).strip()
    if clean_match:
        sections["clean"] = clean_match.group(1).strip()
    if noisy_match:
        sections["noisy"] = noisy_match.group(1).strip()
    if pred_match:
        sections["predicted"] = pred_match.group(1).strip()

    return sections


def extract_pre_text_from_html(html_path: str) -> str:
    """
    Reads W&B saved html text export and extracts the <pre>...</pre> contents.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r"<pre>(.*?)</pre>", content, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not find <pre>...</pre> block in {html_path}")

    text = html.unescape(m.group(1))
    return text


# -----------------------------
# Graphviz rendering
# -----------------------------

# def render_scene_graph_graphviz(
#     nodes: List[Dict],
#     triplets: List[Tuple[int, str, int]],
#     out_path_no_ext: str,
#     title: str = "Scene Graph",
#     rankdir: str = "LR",
#     format: str = "png",
#     show_node_ids: bool = True,
#     relation_fillcolor: str = "lightblue",
#     object_fillcolor: str = "lightyellow",
#     graph_bgcolor: str = "white",
# ) -> str:
#     """
#     nodes: [{"id": int, "label": str}, ...]
#     triplets: [(subj_id, rel, obj_id), ...]

#     Saves and returns rendered image path.
#     """
#     if Digraph is None:
#         raise ImportError("graphviz python package is not available.")

#     dot = Digraph(name=title, format=format)
#     dot.attr(
#         rankdir=rankdir,
#         splines="true",
#         nodesep="0.45",
#         ranksep="0.75",
#         bgcolor=graph_bgcolor,
#         pad="0.2",
#     )
#     dot.attr(label=title, labelloc="t", fontsize="20", fontname="Helvetica-Bold")

#     # Object nodes
#     for node in nodes:
#         node_id = int(node["id"])
#         label = str(node["label"])

#         visible_label = f"[{node_id:02d}] {label}" if show_node_ids else label

#         dot.node(
#             f"obj_{node_id}",
#             label=visible_label,
#             shape="box",
#             style="rounded,filled",
#             fillcolor=object_fillcolor,
#             color="gray30",
#             fontname="Helvetica",
#             fontsize="12",
#             margin="0.15,0.08",
#         )

#     # Relation nodes
#     for edge_idx, (subj_id, rel, obj_id) in enumerate(triplets):
#         rel_node_id = f"rel_{edge_idx}"

#         dot.node(
#             rel_node_id,
#             label=str(rel),
#             shape="ellipse",
#             style="filled",
#             fillcolor=relation_fillcolor,
#             color="steelblue4",
#             fontname="Helvetica",
#             fontsize="11",
#             margin="0.08,0.04",
#         )

#         dot.edge(f"obj_{subj_id}", rel_node_id, color="gray35", penwidth="1.3")
#         dot.edge(rel_node_id, f"obj_{obj_id}", color="gray35", penwidth="1.3")

#     ensure_dir(os.path.dirname(out_path_no_ext) or ".")
#     rendered_path = dot.render(out_path_no_ext, cleanup=True)
#     return rendered_path

def render_scene_graph_graphviz(
    nodes: List[Dict],
    triplets: List[Tuple[int, str, int]],
    out_path_no_ext: str,
    title: str = "Scene Graph",
    rankdir: str = "LR",
    format: str = "png",
    show_node_ids: bool = True,
    relation_fillcolor: str = "lightblue1",
    object_fillcolor: str = "lightpink1",
    graph_bgcolor: str = "transparent",
) -> str:
    """
    nodes: [{"id": int, "label": str}, ...]
    triplets: [(subj_id, rel, obj_id), ...]

    Saves and returns rendered image path.
    Aesthetic matched to LayoutDiffusion-style scene graph rendering:
        pink object boxes
        blue relation boxes
        relation as intermediate node
        thick arrows
        transparent background
    """
    if Digraph is None:
        raise ImportError("graphviz python package is not available.")

    ensure_dir(os.path.dirname(out_path_no_ext) or ".")

    dot = Digraph(name=title, format=format)

    # Match reference style
    dot.attr(
        size="5,3",
        ratio="compress",
        dpi="300",
        bgcolor=graph_bgcolor,
        rankdir=rankdir,
        nodesep="0.5",
        ranksep="0.5",
        pad="0.05",
    )

    # No big title by default; keeps graph clean for papers/slides.
    # If you still want title, uncomment below.
    # dot.attr(label=title, labelloc="t", fontsize="20", fontname="Helvetica-Bold")

    # Object nodes
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontsize="48",
        color="none",
        fillcolor=object_fillcolor,
        fontname="Helvetica",
        margin="0.15,0.08",
    )

    for node in nodes:
        node_id = int(node["id"])
        label = str(node["label"])
        visible_label = f"[{node_id:02d}] {label}" if show_node_ids else label

        dot.node(
            f"obj_{node_id}",
            label=visible_label,
        )

    # Relation nodes
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontsize="48",
        color="none",
        fillcolor=relation_fillcolor,
        fontname="Helvetica",
        margin="0.15,0.08",
    )

    edge_width = "6"
    arrow_size = "1.5"
    binary_edge_weight = "1.2"

    for edge_idx, (subj_id, rel, obj_id) in enumerate(triplets):
        rel_node_id = f"rel_{edge_idx}"

        dot.node(
            rel_node_id,
            label=str(rel),
        )

        dot.edge(
            f"obj_{int(subj_id)}",
            rel_node_id,
            penwidth=edge_width,
            arrowsize=arrow_size,
            weight=binary_edge_weight,
        )
        dot.edge(
            rel_node_id,
            f"obj_{int(obj_id)}",
            penwidth=edge_width,
            arrowsize=arrow_size,
            weight=binary_edge_weight,
        )

    rendered_path = dot.render(out_path_no_ext, cleanup=True)
    return rendered_path

def render_scene_graph_graphviz_edges(
    nodes,
    triplets,
    out_path_no_ext,
    title: str = "Scene Graph",
    rankdir: str = "LR",
    format: str = "png",
    show_node_ids: bool = True,
    object_fillcolor: str = "lightpink1",
    graph_bgcolor: str = "transparent",
    edge_label_bgcolor: str = "white",
) -> str:
    """
    nodes: [{"id": int, "label": str}, ...]
    triplets: [(subj_id, rel, obj_id), ...]

    Draws object nodes as boxes and relations as labels placed on a cut in the edge.
    """
    if Digraph is None:
        raise ImportError("graphviz python package is not available.")

    ensure_dir(os.path.dirname(out_path_no_ext) or ".")

    dot = Digraph(name=title, format=format)

    dot.attr(
        size="5,3",
        ratio="compress",
        dpi="300",
        bgcolor=graph_bgcolor,
        rankdir=rankdir,
        nodesep="0.6",
        ranksep="0.8",
        pad="0.05",
    )

    # Object nodes
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontsize="48",
        color="none",
        fillcolor=object_fillcolor,
        fontname="Helvetica",
        margin="0.15,0.08",
    )

    for node in nodes:
        node_id = int(node["id"])
        label = str(node["label"])
        visible_label = f"[{node_id:02d}] {label}" if show_node_ids else label

        dot.node(
            f"obj_{node_id}",
            label=visible_label,
        )

    # Edge style
    edge_width = "6"
    arrow_size = "1.5"

    dot.attr(
        "edge",
        penwidth=edge_width,
        arrowsize=arrow_size,
        fontsize="36",
        fontname="Helvetica",
        color="black",
        fontcolor="black",
    )

    # Relations as plaintext intermediate nodes.
    # This creates a true visual "cut" in the edge and works with transparent backgrounds.
    dot.attr(
        "node",
        shape="plaintext",
        style="",
        fontsize="36",
        fontname="Helvetica",
        margin="0.02,0.02",
        color="none",
    )

    for edge_idx, (subj_id, rel, obj_id) in enumerate(triplets):
        rel_node_id = f"rel_{edge_idx}"

        dot.node(
            rel_node_id,
            label=str(rel),
        )

        dot.edge(
            f"obj_{int(subj_id)}",
            rel_node_id,
            penwidth=edge_width,
            arrowsize="0.0",
        )

        dot.edge(
            rel_node_id,
            f"obj_{int(obj_id)}",
            penwidth=edge_width,
            arrowsize=arrow_size,
        )

    rendered_path = dot.render(out_path_no_ext, cleanup=True)
    return rendered_path

import textwrap

def wrap_label(label, width=10):
    label = str(label)

    # Handles names like "traffic light" better
    label = label.replace("_", " ")

    return "\n".join(
        textwrap.wrap(
            label,
            width=width,
            break_long_words=True,
            break_on_hyphens=True,
        )
    )

def render_scene_graph_graphviz_edges_new(
    nodes,
    triplets,
    out_path_no_ext,
    title: str = "Scene Graph",
    format: str = "png",
    show_node_ids: bool = False,
    graph_bgcolor: str = "transparent",
    auto_node_colors: bool = True,
    default_node_color: str = "lightgray",
) -> str:

    if Digraph is None:
        raise ImportError("graphviz python package is not available.")

    ensure_dir(os.path.dirname(out_path_no_ext) or ".")

    dot = Digraph(name=title, format=format, engine="neato")

    dot.attr(
        bgcolor=graph_bgcolor,
        dpi="300",
        overlap="false",
        splines="curved",
        outputorder="edgesfirst",
        sep="+6",
        esep="+4",
        pad="0.1",
    )

    palette = [
        "lightpink1",
        "lightblue1",
        "lightgoldenrod1",
        "palegreen1",
        "plum1",
        "lightsalmon1",
        "khaki1",
        "aquamarine1",
        "thistle1",
        "mistyrose1",
    ]

    for node in nodes:
        node_id = int(node["id"])
        label = str(node["label"])
        visible_label = f"{node_id}: {label}" if show_node_ids else label

        color = palette[node_id % len(palette)] if auto_node_colors else default_node_color

        dot.node(
            f"obj_{node_id}",
            label=wrap_label(visible_label,width=0.4),
            shape="box",
            style="filled,rounded",
            fillcolor=color,
            color="gray40",
            penwidth="1.2",
            fontsize="5",
            fontname="Helvetica",
            fixedsize="true",
            width="0.4",
            height="0.4",
        )

    dot.attr(
        "edge",
        penwidth="1.2",
        arrowsize="0.4",
        # color="gray35",
        color="black"
    )

    for edge_idx, (subj_id, rel, obj_id) in enumerate(triplets):
        rel_node_id = f"rel_{edge_idx}"

        dot.node(
            rel_node_id,
            label=str(rel),
            shape="plain",
            width="0",
            height="0",
            margin="0",
            fontsize="7",
            fontname="Helvetica",
        )

        dot.edge(
            f"obj_{int(subj_id)}",
            rel_node_id,
            arrowsize="0",
            len="0.3",
            weight="10",
        )

        dot.edge(
            rel_node_id,
            f"obj_{int(obj_id)}",
            arrowsize="0.4",
            len="0.3",
            weight="10",
        )

    return dot.render(out_path_no_ext, cleanup=True)

import numpy as np

def data_item_to_graphviz_inputs(
    item,
    ind_to_classes,
    ind_to_predicates,
    no_rel_id: int = 0,
):
    labels = np.asarray(item["node_labels"])
    edge_map = np.asarray(item["edge_map"])

    nodes = []
    for i, lab in enumerate(labels):
        nodes.append({
            "id": int(i),
            "label": str(ind_to_classes[int(lab)]),
        })

    triplets = []
    N = len(labels)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            rel_id = int(edge_map[i, j])
            if rel_id == no_rel_id:
                continue

            triplets.append((
                int(i),
                str(ind_to_predicates[rel_id]),
                int(j),
            ))

    return nodes, triplets

def render_graph_text_block_to_image(
    graph_text: str,
    out_path_no_ext: str,
    title: str,
    rankdir: str = "LR",
    format: str = "png",
    show_node_ids: bool = True,
) -> str:
    """
    Parse one graph text block and render it.
    """
    nodes, triplets = parse_graph_text_block(graph_text)
    return render_scene_graph_graphviz(
        nodes=nodes,
        triplets=triplets,
        out_path_no_ext=out_path_no_ext,
        title=title,
        rankdir=rankdir,
        format=format,
        show_node_ids=show_node_ids,
    )


def render_validation_log_html_to_images(
    html_path: str,
    out_dir: str,
    prefix: Optional[str] = None,
    rankdir: str = "LR",
    format: str = "png",
    show_node_ids: bool = True,
) -> Dict[str, Optional[str]]:
    """
    For already completed W&B HTML log files:
    - extract <pre> text
    - parse CLEAN / NOISY / PREDICTED sections
    - render each section to an image

    Returns:
        {
            "clean": "...png",
            "noisy": "...png",
            "predicted": "...png",
            "raw_text": "...txt",
        }
    """
    ensure_dir(out_dir)

    base = prefix or sanitize_filename(os.path.splitext(os.path.basename(html_path))[0])

    full_text = extract_pre_text_from_html(html_path)
    txt_path = os.path.join(out_dir, f"{base}_extracted.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    sections = parse_validation_log_text(full_text)

    outputs = {
        "clean": None,
        "noisy": None,
        "predicted": None,
        "raw_text": txt_path,
    }

    if sections["clean"]:
        outputs["clean"] = render_graph_text_block_to_image(
            graph_text=sections["clean"],
            out_path_no_ext=os.path.join(out_dir, f"{base}_clean"),
            title="Clean Graph",
            rankdir=rankdir,
            format=format,
            show_node_ids=show_node_ids,
        )

    if sections["noisy"]:
        outputs["noisy"] = render_graph_text_block_to_image(
            graph_text=sections["noisy"],
            out_path_no_ext=os.path.join(out_dir, f"{base}_noisy"),
            title="Noisy Graph",
            rankdir=rankdir,
            format=format,
            show_node_ids=show_node_ids,
        )

    if sections["predicted"]:
        outputs["predicted"] = render_graph_text_block_to_image(
            graph_text=sections["predicted"],
            out_path_no_ext=os.path.join(out_dir, f"{base}_predicted"),
            title="Predicted x0 Graph",
            rankdir=rankdir,
            format=format,
            show_node_ids=show_node_ids,
        )

    return outputs


def render_layout_boxes_to_image(
    obj_class,
    obj_bbox,
    is_valid_obj,
    out_path: str,
    class_names=None,
    image_size: int = 256,
    skip_first_object: bool = True,
) -> str:
    return draw_layout_boxes(
        obj_class=obj_class,
        obj_bbox=obj_bbox,
        is_valid_obj=is_valid_obj,
        output_path=out_path,
        class_names=class_names,
        image_size=image_size,
        draw_text=True,
        skip_first_object=skip_first_object,
    )

class WandBLogger:
    def __init__(self, opt):
        self.opt = opt
        self.enabled = bool(opt.use_wandb) and is_main_process() and (wandb is not None)
        self.run = None

    def init(self):
        if not self.enabled:
            return

        self.run = wandb.init(
            project=self.opt.wandb_project,
            entity=self.opt.wandb_entity,
            name=self.opt.wandb_run_name,
            mode=self.opt.wandb_mode,
            config=asdict(self.opt) if hasattr(self.opt, "__dataclass_fields__") else None,
        )

    def log(self, data: dict, step: Optional[int] = None):
        if not self.enabled:
            return
        wandb.log(data, step=step)

    def watch(self, model: torch.nn.Module, log: str = None, log_freq: int = 500): #"gradients"
        # if not self.enabled:
        #     return
        # wandb.watch(model, log=log, log_freq=log_freq)
        return

    def finish(self):
        if not self.enabled:
            return
        wandb.finish()

    def log_text(self, key: str, text: str, step: Optional[int] = None):
        if not self.enabled:
            return
        wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)