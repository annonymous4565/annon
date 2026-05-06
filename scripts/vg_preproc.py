import pyrallis
import random
import os
import numpy as np
from configs import DiscreteSGConfig
from datasets_ import VGSceneGraphPreprocessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@pyrallis.wrap()
def main(opt: DiscreteSGConfig) -> None:
    set_seed(opt.seed)
    os.makedirs(opt.output_dir, exist_ok=True)

    proc = VGSceneGraphPreprocessor(
        h5_path=opt.h5_path,
        dict_json_path=opt.dict_json_path,
        image_data_json_path=opt.image_data_json_path,
        n_max=opt.n_max,
        box_scale_key=opt.box_scale_key,
        filter_empty_rels=opt.filter_empty_rels,
        filter_non_overlap=opt.filter_non_overlap,
        drop_background=opt.drop_background,
        pad_token=opt.pad_token,
        no_rel_token=opt.no_rel_token,
        keep_first_relation=opt.keep_first_relation,
        object_selection=opt.object_selection,
        degree_weight=opt.degree_weight,
        area_weight=opt.area_weight,
        center_weight=opt.center_weight,
        spread_weight=opt.spread_weight,
        repeat_penalty_weight=opt.repeat_penalty_weight,
        repeat_penalty_type=opt.repeat_penalty_type,
        enable_graph_filtering=opt.enable_graph_filtering,
        filter_min_nodes_for_low_rel=opt.filter_min_nodes_for_low_rel,
        filter_max_pos_edges_if_dense_nodes=opt.filter_max_pos_edges_if_dense_nodes,
        filter_max_repeated_labels=opt.filter_max_repeated_labels,
        filter_max_single_label_count=opt.filter_max_single_label_count,
        filter_max_mean_box_iou=opt.filter_max_mean_box_iou,
        filter_max_isolated_frac=opt.filter_max_isolated_frac,
    )

    try:
        if opt.inspect_index is not None:
            item = proc.process_image_index(opt.inspect_index, split_name=opt.export_split)
            proc.print_processed_graph(item)

        if opt.inspect_nth is not None:
            indices = proc.get_indices(opt.export_split)
            image_index = int(indices[opt.inspect_nth])
            item = proc.process_image_index(image_index, split_name=opt.export_split)
            proc.print_processed_graph(item)
        out_name = f"vg_sg_{opt.object_selection}_filter_new_{opt.export_split}_nmax{opt.n_max}.npz"
        out_path = os.path.join(opt.output_dir, out_name)
        proc.export_split_packed(
            out_path=out_path,
            split=opt.export_split,
            limit=opt.limit,
        )
        print(f"Saved packed dataset to: {out_path}")

    finally:
        proc.close()

if __name__ == "__main__":
    main()
