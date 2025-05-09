# train.py
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data.dataset import UVMapDataset
from utils.warp_function import warp  # ← new
from torchvision.utils import save_image
from models import get_model
from metrics.metrics import psnr
from tqdm import tqdm

def train_model(args):
    # 1) Transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop((args.size, args.size), scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 2) Dataset & split
    ds = UVMapDataset(
        args.distorted,
        args.uv,
        args.clean,
        transform,
        transform_type=args.transform_type
    )
    val_n   = int(len(ds) * args.val_split)
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    # 3) Samplers
    train_sampler = DistributedSampler(train_ds) if args.distributed else None
    val_sampler   = DistributedSampler(val_ds)   if args.distributed else None

    # 4) DataLoaders
    tr_loader = DataLoader(
        train_ds, batch_size=args.bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True
    )
    vl_loader = DataLoader(
        val_ds, batch_size=args.bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True
    )

    # 5) Device selection
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 6) Model now only predicts UV (2 channels)
    model = get_model(args.model, in_ch=3, out_ch=2).to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 7) Optimizer & losses
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_uv = nn.L1Loss()
    loss_img = nn.MSELoss()

    os.makedirs(args.ckpt, exist_ok=True)
    best_psnr = 0.0

    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for dist, uv_gt, clean in pbar:
            dist, uv_gt, clean = dist.to(device), uv_gt.to(device), clean.to(device)

            # 8) Forward → UV → warp → reconstruction
            pred_uv = model(dist)                 # B×2×H×W
            recon   = warp(dist, pred_uv)         # B×3×H×W

            l_uv  = loss_uv(pred_uv, uv_gt)
            l_img = loss_img(recon, clean)
            loss  = l_uv + args.img_w * l_img

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * dist.size(0)
            pbar.set_postfix(loss=f"{train_loss/((pbar.n+1)*args.bs):.4f}")

        train_loss /= len(train_ds)

        # ---- VALIDATE ----
        model.eval()
        val_loss   = 0.0
        total_psnr = 0.0
        pbar = tqdm(vl_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for dist, uv_gt, clean in pbar:
                dist, uv_gt, clean = dist.to(device), uv_gt.to(device), clean.to(device)

                pred_uv = model(dist)
                recon   = warp(dist, pred_uv)

                l_uv  = loss_uv(pred_uv, uv_gt)
                l_img = loss_img(recon, clean)
                loss  = l_uv + args.img_w * l_img

                val_loss   += loss.item() * dist.size(0)
                total_psnr += psnr(recon, clean) * dist.size(0)
                pbar.set_postfix(
                    val_loss=f"{val_loss/((pbar.n+1)*args.bs):.4f}",
                    psnr=f"{(total_psnr/((pbar.n+1)*args.bs)):.2f}"
                )

        val_loss /= len(val_ds)
        avg_psnr = total_psnr / len(val_ds)

        print(f"Epoch {epoch}/{args.epochs} — "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

        # ---- Save best checkpoint ----
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(
                model.state_dict(),
                os.path.join(args.ckpt, f'best_{args.model}.pth')
            )

    # ---- final save ----
    torch.save(
        model.state_dict(),
        os.path.join(args.ckpt, f'last_{args.model}.pth')
    )
    print("Training complete, checkpoints saved.")
