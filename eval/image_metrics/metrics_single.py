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


def evaluate_render(renders, gt):
    render = Image.open(renders)
    gt = Image.open(gt)
    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

    # [1, chaneel, H, W]
    return (
        render,
        psnr(render, gt),
    )


def evaluate(args):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    full_dict[args.renders_color] = {}
    per_view_dict[args.renders_color] = {}
    full_dict_polytopeonly[args.renders_color] = {}
    per_view_dict_polytopeonly[args.renders_color] = {}

    scene_dir_path = Path(args.renders_color)

    gt_color_dir = Path(args.gt_color)
    renders_color = Path(args.renders_color)

    image_names = []
    psnrs = []

    fname, psnr = evaluate_render(renders_color, gt_color_dir)
    psnrs.append(psnr)

    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))

    full_dict[args.renders_color].update(
        {
            "PSNR": torch.tensor(psnrs).mean().item(),
        }
    )
    per_view_dict[args.renders_color].update(
        {
            "PSNR": {
                name: psnr
                for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
            },
        }
    )

    with open(scene_dir_path.parent / "../render_eval.json", "w") as fp:
        json.dump(full_dict, fp, indent=True)
    with open(scene_dir_path.parent / "../../evaluation_results.json", "a") as fp:
        full_dict[args.renders_color] = {
            k: round(v, 3) for k, v in full_dict[args.renders_color].items()
        }
        json.dump(full_dict, fp, indent=True)
        fp.write("\n")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--gt_color", type=str, default="")
    parser.add_argument("--renders_color", type=str, default="")
    args = parser.parse_args()
    evaluate(args)
