import os
import pickle
from typing import Dict, List, Any, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_lookup(vocab, idx: int, offset: int = 0):
    idx = int(idx) + offset
    try:
        if isinstance(vocab, dict):
            return str(vocab[idx])
        return str(vocab[idx])
    except Exception:
        return f"cls_{idx}"

def clean_obj_name(name: str):
    if name.endswith("-other"):
        return name.replace("-other", "")
    return name

def item_text(label: str, attributes=None) -> str:
    # LAION-SG style: "male middle-aged person"
    if attributes:
        attrs = [str(a) for a in attributes if str(a)]
        return " ".join(attrs + [str(label)])
    return str(label)


class COCODataset(Dataset):
    """
    Converts COCO/DiffuseSG-style pkl entries into LAION-SG-style outputs.

    Returns:
        triplets: List[Dict]
          [{"item1": str, "relation": str, "item2": str}, ...]

        isolated_items: List[str]

        global_ids: List[Dict]
          [{"item1": int, "item2": int}, ...]

    Expected pkl entry:
        {
            "node_labels": np.ndarray [N],
            "edge_map": np.ndarray [N,N],
            "caption": str,
            "image_id": int,
            ...
        }
    """

    def __init__(
        self,
        pkl_path: str,
        vocab_pkl: str,
        no_relation_id: int = 0,
        class_index_offset: int = 0,
        predicate_index_offset: int = 0,
        global_id_offset: int = 0,
        include_self_edges: bool = False,
        return_metadata: bool = False,
    ):
        self.data = load_pickle(pkl_path)
        self.no_relation_id = int(no_relation_id)
        self.class_index_offset = int(class_index_offset)
        self.predicate_index_offset = int(predicate_index_offset)
        self.global_id_offset = int(global_id_offset)
        self.include_self_edges = bool(include_self_edges)
        self.return_metadata = bool(return_metadata)

        vocab = load_pickle(vocab_pkl)

        if isinstance(vocab, dict):
            self.ind_to_classes = vocab.get("ind_to_classes")
            self.ind_to_predicates = vocab.get("ind_to_predicates")
        else:
            # categories.p style: ind_to_classes, ind_to_predicates, ...
            self.ind_to_classes = vocab[0]
            self.ind_to_predicates = vocab[1]

        if self.ind_to_classes is None:
            raise KeyError("Could not find ind_to_classes in vocab_pkl.")
        if self.ind_to_predicates is None:
            raise KeyError("Could not find ind_to_predicates in vocab_pkl.")
        

    def __len__(self):
        return len(self.data)

    def _global_item_id(self, image_id: int, node_idx: int) -> int:
        # Stable synthetic global id.
        # Avoid collision by reserving 1000 node slots per image.
        return int(self.global_id_offset + int(image_id) * 1000 + int(node_idx))

    def _decode_nodes(self, entry: Dict[str, Any]) -> Tuple[List[str], List[int]]:
        labels = np.asarray(entry["node_labels"], dtype=np.int64)
        image_id = int(entry.get("image_id", entry.get("filename", 0)))

        node_texts = []
        global_item_ids = []

        for i, lab in enumerate(labels):
            label = clean_obj_name(safe_lookup(
                self.ind_to_classes,
                int(lab),
                offset=self.class_index_offset,
            ))
            node_texts.append(item_text(label))
            global_item_ids.append(self._global_item_id(image_id, i))

        return node_texts, global_item_ids


    def __getitem__(self, idx: int):
        entry = self.data[idx]
        print(entry)
        node_texts, global_item_ids = self._decode_nodes(entry)
        edge_map = np.asarray(entry["edge_map"], dtype=np.int64)

        n = len(node_texts)
        used_nodes = set()
        triplets = []
        global_ids = []

        for i in range(n):
            for j in range(n):
                if i == j and not self.include_self_edges:
                    continue

                rel_id = int(edge_map[i, j])
                if rel_id == self.no_relation_id:
                    continue

                rel = safe_lookup(
                    self.ind_to_predicates,
                    rel_id,
                    offset=self.predicate_index_offset,
                )

                triplets.append({
                    "item1": node_texts[i],
                    "relation": rel,
                    "item2": node_texts[j],
                })

                global_ids.append({
                    "item1": int(global_item_ids[i]),
                    "item2": int(global_item_ids[j]),
                })

                used_nodes.add(i)
                used_nodes.add(j)

        isolated_items = [
            node_texts[i]
            for i in range(n)
            if i not in used_nodes
        ]

        # print(type(self.ind_to_classes))
        # print(list(self.ind_to_classes.items())[:5] if isinstance(self.ind_to_classes, dict) else self.ind_to_classes[:5])
        # print("label 134 direct:", self.ind_to_classes[134] if not isinstance(self.ind_to_classes, dict) else self.ind_to_classes.get(134))
        # print("label 133 direct:", self.ind_to_classes[133] if not isinstance(self.ind_to_classes, dict) else self.ind_to_classes.get(133))
        # print("class_index_offset:", self.class_index_offset)
        # print("node_labels:", entry["node_labels"])
        # for lab in entry["node_labels"]:
        #     print("raw:", int(lab), "lookup_idx:", int(lab) + self.class_index_offset,
        #         "name:", safe_lookup(self.ind_to_classes, int(lab), self.class_index_offset))
        # breakpoint()

        # all_id = str(entry.get("filename", entry.get("image_id", idx)))
        all_id = str(entry["img_id"])

        return [0], triplets, global_ids, isolated_items, [0], [0], [0], all_id

def laion_sg_collate(batch):
    imgs, triplets, global_ids, isolated_items, texts, sizes, crops, all_ids = zip(*batch)
    return (
        list(imgs),
        list(triplets),
        list(global_ids),
        list(isolated_items),
        list(texts),
        list(sizes),
        list(crops),
        list(all_ids),
    )


def build_coco_sg_dataloader(args):
    print(args.val_pkl_path)
    dataset = COCODataset(
        pkl_path=args.val_pkl_path,
        vocab_pkl=args.val_vocab_path,
        no_relation_id=0,
        class_index_offset=0,
        predicate_index_offset=0,
        return_metadata=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=laion_sg_collate,
    )

    return loader