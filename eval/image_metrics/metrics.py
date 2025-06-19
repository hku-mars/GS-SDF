#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from loss_utils import ssim
from lpipsPyTorch import Lpips
import json
from tqdm import tqdm
from image_utils import psnr
from argparse import ArgumentParser


def evaluate_render(renders_dir, gt_dir, fname, lpips_inferer: Lpips):
    render = Image.open(renders_dir / fname)
    gt = Image.open(gt_dir / fname)
    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

    # [1, chaneel, H, W]
    return (
        fname,
        ssim(render, gt),
        psnr(render, gt),
        lpips_inferer.forward(render, gt),
    )


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        try:
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # color_dir = Path(scene_dir) / "color"
            scene_dir_path = Path(scene_dir)

            gt_color_dir = scene_dir_path / "gt"
            renders_color_dir = scene_dir_path / "renders"

            image_names = []
            ssims = []
            psnrs = []
            lpipss = []

            lpips_inferer = Lpips(net_type="vgg")
            list = os.listdir(renders_color_dir)
            if len(list) == 0:
                print("No renders found in", renders_color_dir)
                continue
            for fname in tqdm(os.listdir(renders_color_dir)):
                fname, ssim, psnr, lpips = evaluate_render(
                    renders_color_dir, gt_color_dir, fname, lpips_inferer
                )

                image_names.append(fname)
                ssims.append(ssim)
                psnrs.append(psnr)
                lpipss.append(lpips)

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            # print max PSNR and min PSNR and their filename
            max_psnr = torch.tensor(psnrs).max()
            min_psnr = torch.tensor(psnrs).min()
            max_psnr_index = torch.tensor(psnrs).argmax()
            min_psnr_index = torch.tensor(psnrs).argmin()

            print(
                "  Max PSNR: {:>12.7f} for {}".format(
                    max_psnr, image_names[max_psnr_index]
                )
            )
            print(
                "  Min PSNR: {:>12.7f} for {}".format(
                    min_psnr, image_names[min_psnr_index]
                )
            )
            print("")

            full_dict[scene_dir].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "Max PSNR {}:".format(image_names[max_psnr_index]): max_psnr.item(),
                    "Min PSNR {}:".format(image_names[min_psnr_index]): min_psnr.item(),
                }
            )
            per_view_dict[scene_dir].update(
                {
                    "SSIM": {
                        name: ssim
                        for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                    },
                    "PSNR": {
                        name: psnr
                        for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                    },
                    "LPIPS": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    },
                }
            )

            with open(scene_dir_path / "../render_eval.json", "w") as fp:
                json.dump(full_dict, fp, indent=True)
            with open(scene_dir_path / "../render_eval_per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
            with open(scene_dir_path / "../../evaluation_results.json", "a") as fp:
                full_dict[scene_dir] = {
                    k: round(v, 3) for k, v in full_dict[scene_dir].items()
                }
                json.dump(full_dict, fp, indent=True)
                fp.write("\n")
            
            with open(scene_dir_path / "../../../../all_evaluation_results.json", "a") as fp:
                # add dataset name
                json.dump(full_dict, fp, indent=True)
                fp.write("\n")
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir_path)
            print(e)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    args = parser.parse_args()
    evaluate(args.model_paths)
