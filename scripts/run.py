#!/usr/bin/env python3
import sys, os
# add project sub-packages to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import argparse
from models import MODEL_REGISTRY
from train import train_model
import torch

# put paths in dictionary
DATA_DIRS = {
    "radial_distortion": {
        "distorted": r"D:\distortion_data\Kaggle_Images4k_radial_distortion\distorted",
        "uv":        r"D:\distortion_data\Kaggle_Images4k_radial_distortion\uv_maps",
        "clean":     r"D:\distortion_data\Kaggle_Images4k_dataset"
    },
    "random_field": {
        "distorted": r"D:\distortion_data\Kaggle_Images4k_random_field\distorted",
        "uv":        r"D:\distortion_data\Kaggle_Images4k_random_field\uv_maps",
        "clean":     r"D:\distortion_data\Kaggle_Images4k_dataset"
    }
}

distortion_type = "radial_distortion"  # or "radial_distortion"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--distorted',
        default=DATA_DIRS[distortion_type]["distorted"],
        help="distorted images directory"
    )
    p.add_argument(
        '--uv',
        default=DATA_DIRS[distortion_type]["uv"],
        help="uv maps directory"
    )
    p.add_argument(
        '--clean',
        default=DATA_DIRS[distortion_type]["clean"],
        help="clean images directory"
    )
    p.add_argument(
        '--transform_type',
        default=distortion_type,
        help="suffix for distorted PNG files"
    )
    p.add_argument('--ckpt',        default='checkpoints', help="checkpoint dir")
    p.add_argument(
        '--model',
        default='unet',
        choices=list(MODEL_REGISTRY.keys()),
        help="model name"
    )
    p.add_argument('--size',        type=int,   default=256,   help="image size")
    p.add_argument('--bs',          type=int,   default=8,     help="batch size")
    p.add_argument('--lr',          type=float, default=1e-4,  help="learning rate")
    p.add_argument('--epochs',      type=int,   default=10,    help="number of epochs")
    p.add_argument('--val_split',   type=float, default=0.2,   help="validation split")
    p.add_argument(
        '--img_w',
        type=float,
        default=0.1,
        help="weight for reconstruction loss"
    )
    p.add_argument(
        '--workers',
        type=int,
        default=4,
        help="num workers for data loading"
    )
    p.add_argument(
        '--multi_gpu',
        action='store_true',
        help="use multiple GPUs via DataParallel"
    )
    p.add_argument(
        '--distributed',
        action='store_true',
        help="use DistributedDataParallel"
    )
    p.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help="local rank for distributed training"
    )
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
    train_model(args)
