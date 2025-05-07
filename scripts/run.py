# -------- scripts/run.py --------
import sys, os
# add project sub-packages to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import argparse
from models import MODEL_REGISTRY
from train import train_model
import torch

#put parths in dictionary
DATA_DIRS = {
    "radial_distortion": {
        "distorted": r"D:\distortion_data\Kaggle_Images4k_radial_distortion\distorted",
        "uv": r"D:\distortion_data\Kaggle_Images4k_radial_distortion\uv_maps",
        "clean": r"D:\distortion_data\Kaggle_Images4k_dataset"
    },
    "random_field": {
        "distorted": r"D:\distortion_data\Kaggle_Images4k_random_field\distorted",
        "uv": r"D:\distortion_data\Kaggle_Images4k_random_field\uv_maps",
        "clean": r"D:\distortion_data\Kaggle_Images4k_dataset"
    }
}

distortion_type = "random_field"  # or "random_field"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--distorted',      default=DATA_DIRS[distortion_type]["distorted"])  # distorted images directory
    p.add_argument('--uv',             default=DATA_DIRS[distortion_type]["uv"])       # uv maps directory
    p.add_argument('--clean',          default=DATA_DIRS[distortion_type]["clean"])      # clean images directory
    p.add_argument('--transform_type', default=distortion_type,
                   help="suffix for distorted PNG files, e.g. 'radial_distortion' or 'random_field'")
    p.add_argument('--ckpt',           default='checkpoints')  # checkpoint directory
    p.add_argument('--model',          default='unet', choices=list(MODEL_REGISTRY.keys()))  # model name
    p.add_argument('--size',           type=int,   default=256)    # image size
    p.add_argument('--bs',             type=int,   default=8)      # batch size
    p.add_argument('--lr',             type=float, default=1e-4)   # learning rate
    p.add_argument('--epochs',         type=int,   default=10)     # number of epochs
    p.add_argument('--val_split',      type=float, default=0.2)    # validation split
    p.add_argument('--img_w',          type=float, default=0.1)    # weight for image loss
    p.add_argument('--workers',        type=int,   default=4)      # number of workers for data loading
    p.add_argument('--multi_gpu',      action='store_true')        # use multiple GPUs
    p.add_argument('--distributed',    action='store_true')        # use distributed training
    p.add_argument('--local_rank',     type=int,   default=0)      # local rank for distributed training
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
    train_model(args)
