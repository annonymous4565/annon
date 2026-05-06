import os
import json
import argparse

from evaluation.graph_metrics import SceneGraphSample, evaluate_graph_generation


def load_graphs(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    graphs = []
    for item in obj:
        graphs.append(
            SceneGraphSample(
                nodes=item["nodes"],
                triplets=[tuple(t) for t in item["triplets"]],
            )
        )
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_sg_json", type=str, required=True)
    parser.add_argument("--gt_sg_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    pred_graphs = load_graphs(args.pred_sg_json)
    gt_graphs = load_graphs(args.gt_sg_json)

    metrics = evaluate_graph_generation(pred_graphs, gt_graphs)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "graph_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()