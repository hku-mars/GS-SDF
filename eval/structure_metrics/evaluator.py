import argparse
import json
from eval_utils import eval_mesh
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pcd", help="folder containing the depth images")
    parser.add_argument("--pred_mesh", help="folder containing the rgb images")
    # evaluation parameters
    parser.add_argument("--down_sample_vox", type=float, default=0.01)
    parser.add_argument("--dist_thre", type=float, default=0.02)
    parser.add_argument("--truncation_dist_acc", type=float, default=0.2)
    parser.add_argument("--truncation_dist_com", type=float, default=0.2)
    args = parser.parse_args()

    output_json_path = Path(args.pred_mesh).parent

    # evaluation
    eval_metric = eval_mesh(
        args.pred_mesh,
        args.gt_pcd,
        down_sample_res=args.down_sample_vox,
        threshold=args.dist_thre,
        truncation_acc=args.truncation_dist_acc,
        truncation_com=args.truncation_dist_com,
        gt_bbx_mask_on=True,
    )

    try:
        with open(output_json_path / "structure_eval.json", "w") as fp:
            json.dump(args.pred_mesh, fp, indent=True)
            json.dump(eval_metric, fp, indent=True)
            for key in eval_metric:
                print(key + ": " + str(eval_metric[key]))
            print(f"Structure evaluation results are written into {output_json_path}")

        with open(output_json_path / "evaluation_results.json", "a") as fp:
            json.dump(args.pred_mesh, fp, indent=True)
            eval_metric = {k: round(v, 3) for k, v in eval_metric.items()}
            json.dump(eval_metric, fp, indent=True)
            fp.write("\n")

        with open(output_json_path / "../../all_evaluation_results.json", "a") as fp:
            # add dataset name
            json.dump(args.pred_mesh, fp, indent=True)
            eval_metric = {k: round(v, 3) for k, v in eval_metric.items()}
            json.dump(eval_metric, fp, indent=True)
            fp.write("\n")
    except IOError:
        print("I/O error")
