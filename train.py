"""
Train NAFNetSR on paired (degraded, ground-truth) images.

Example:
    python train.py \
        --gt_dir /path/to/train/gt \
        --degraded_dir /path/to/train/degraded \
        --epochs 200 --batch_size 8 --amp

Checkpoints are written to checkpoints/last.pth (every epoch) and
checkpoints/best.pth (best validation pSNR so far) -- evaluation.py
loads architecture config from the checkpoint automatically.
Use --resume checkpoints/last.pth to continue a run that got interrupted.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import PairedRestorationDataset
from losses import RestorationLoss
from model import NAFNetSR


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--channels", type=int, default=1, help="1=grayscale, 3=RGB")
    p.add_argument("--patch_size", type=int, default=96, help="crop size on the degraded image")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ssim_weight", type=float, default=0.2)
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--width", type=int, default=32, help="NAFNet feature channel width")
    p.add_argument("--enc_blocks", type=str, default="1,1,1,1",
                   help="NAFBlocks per encoder level, comma-separated (e.g. 2,2,4,8)")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="mixed precision (faster on modern GPUs)")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Parse enc_blocks from string to list
    enc_blocks = [int(x) for x in args.enc_blocks.split(",")]
    dec_blocks = enc_blocks  # mirror encoder for decoder

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected -- training will be very slow. "
              "Use Colab/Kaggle if you don't have local GPU access.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create separate train and val datasets (no shared backing dataset)
    train_full = PairedRestorationDataset(
        args.gt_dir, args.degraded_dir, channels=args.channels,
        patch_size=args.patch_size, train=True,
    )
    val_full = PairedRestorationDataset(
        args.gt_dir, args.degraded_dir, channels=args.channels,
        patch_size=args.patch_size, train=False,
    )

    n_total = len(train_full)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Generate deterministic index split
    indices = list(range(n_total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    print(f"Train pairs: {n_train} | Val pairs: {n_val}")
    print(f"Model: NAFNetSR(width={args.width}, enc={enc_blocks})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = NAFNetSR(
        channels=args.channels, width=args.width, scale=2,
        enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks,
    ).to(device)
    criterion = RestorationLoss(
        ssim_weight=args.ssim_weight, lpips_weight=args.lpips_weight, device=device
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    best_psnr = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for deg, gt in train_loader:
            deg, gt = deg.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(deg)
                loss, l1_val, ssim_val, lpips_val = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_psnr_total = 0.0
        with torch.no_grad():
            for deg, gt in val_loader:
                deg, gt = deg.to(device), gt.to(device)
                pred = model(deg).clamp(0, 1)
                val_psnr_total += psnr(pred, gt)
        val_psnr = val_psnr_total / len(val_loader)

        dt = time.time() - t0
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train_loss {avg_loss:.4f} | "
            f"val_psnr {val_psnr:.2f} dB | {dt:.1f}s/epoch"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": max(best_psnr, val_psnr),
            "args": {
                **vars(args),
                "width": args.width,
                "enc_blocks": enc_blocks,
                "dec_blocks": dec_blocks,
            },
        }
        torch.save(ckpt, out_dir / "last.pth")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  -> new best ({best_psnr:.2f} dB), saved {out_dir}/best.pth")


if __name__ == "__main__":
    main()
