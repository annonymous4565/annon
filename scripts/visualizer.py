import os
from glob import glob

from utils.wandb_utils import (
    render_validation_log_html_to_images
)



def visualize_one_logged_html(
    html_path: str,
    out_dir: str,
    rankdir: str = "LR",
    format: str = "png",
    show_node_ids: bool = True,
):
    outputs = render_validation_log_html_to_images(
        html_path=html_path,
        out_dir=out_dir,
        rankdir=rankdir,
        format=format,
        show_node_ids=show_node_ids,
    )

    print("Saved:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")

    return outputs


def visualize_many_logged_htmls(
    input_dir: str,
    out_dir: str,
    pattern: str = "*.html",
    rankdir: str = "LR",
    format: str = "png",
    show_node_ids: bool = True,
):
    html_files = sorted(glob(os.path.join(input_dir, pattern)))
    if not html_files:
        print(f"No html files found in {input_dir} matching {pattern}")
        return []

    all_outputs = []
    for html_path in html_files:
        base = os.path.splitext(os.path.basename(html_path))[0]
        curr_out_dir = os.path.join(out_dir, base)

        try:
            outputs = render_validation_log_html_to_images(
                html_path=html_path,
                out_dir=curr_out_dir,
                prefix=base,
                rankdir=rankdir,
                format=format,
                show_node_ids=show_node_ids,
            )
            all_outputs.append(outputs)
            print(f"[OK] {html_path}")
        except Exception as e:
            print(f"[FAIL] {html_path}: {e}")

    return all_outputs