import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data.dataset import UVMapDataset
from models import get_model
from metrics.metrics import psnr
from tqdm import tqdm

def train_model(args):
    # Transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop((args.size, args.size), scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Dataset & split (now pass transform_type)
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

    # Samplers
    train_sampler = DistributedSampler(train_ds) if args.distributed else None
    val_sampler   = DistributedSampler(val_ds)   if args.distributed else None

    # DataLoaders
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

    # Device selection
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = get_model(args.model, in_ch=3, out_ch=5).to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer & losses
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_uv  = nn.MSELoss()
    loss_img = nn.L1Loss()

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
            out       = model(dist)
            pred_uv   = out[:, :2]
            pred_img  = out[:, 2:]
            l_uv      = loss_uv(pred_uv, uv_gt)
            l_img     = loss_img(pred_img, clean)
            loss      = l_uv + args.img_w * l_img

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
                out      = model(dist)
                pred_uv  = out[:, :2]
                pred_img = out[:, 2:]
                l_uv     = loss_uv(pred_uv, uv_gt)
                l_img    = loss_img(pred_img, clean)
                loss     = l_uv + args.img_w * l_img

                val_loss   += loss.item() * dist.size(0)
                total_psnr += psnr(pred_img, clean) * dist.size(0)
                pbar.set_postfix(
                    val_loss=f"{val_loss/((pbar.n+1)*args.bs):.4f}",
                    psnr=f"{(total_psnr/((pbar.n+1)*args.bs)):.2f}"
                )

        val_loss /= len(val_ds)
        avg_psnr = total_psnr / len(val_ds)

        print(f"Epoch {epoch}/{args.epochs} — "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(args.ckpt, 'best_%s.pth'%args.model))

    torch.save(model.state_dict(), os.path.join(args.ckpt, 'last_%s.pth'%args.model))
    print("Training complete, checkpoints 'best.pth' & 'last_%s.pth' saved."%args.model)
