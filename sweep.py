"""
Resumable Hyperparameter Sweep for NAFNetSR
25 configurations × 100 epochs each.

Designed for multi-session Colab use:
  - Each completed config is saved to sweep_log.json
  - Re-running this script skips already-finished configs automatically
  - Best checkpoint per config saved to checkpoints/sweep/cfg_<id>/best.pth
  - Final ranked leaderboard printed and saved to sweep_results.csv

Usage:
    # First session (and any resumed sessions):
    !python sweep.py \
        --gt_dir dataset/train/GT \
        --degraded_dir dataset/train/NoisyLR \
        --channels 1 \
        --probe_epochs 100 \
        --amp

    # After all configs complete, train the winner:
    !python train.py --gt_dir ... --degraded_dir ... --epochs 300 \
        --lr <best_lr> --batch_size <best_bs> ... --amp
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import PairedRestorationDataset
from losses import RestorationLoss
from model import NAFNetSR

class EMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def store_and_apply(self, model):
        msd = model.state_dict()
        self.backup = {k: v.detach().clone() for k, v in msd.items()}
        model.load_state_dict({k: self.shadow[k].to(msd[k].dtype) for k in msd})

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup)
            self.backup = None


# ──────────────────────────────────────────────────────────────
# 25 Configurations
# Strategy: explore lr, batch_size, patch_size, width, depth,
# ssim_weight, and meaningful combinations of the above.
# LPIPS is disabled during sweep (too slow) - used only in final training.
# ──────────────────────────────────────────────────────────────
CONFIGS = [
    # id   lr       bs   patch  width  enc_blocks    ssim_w
    (101, 2e-4,    8,   96,   32,  [1,1,1,1],     0.20),
    (102, 5e-4,    8,   96,   32,  [1,1,1,1],     0.20),
    (103, 1e-3,    8,   96,   32,  [1,1,1,1],     0.20),
]


def psnr_metric(pred, target):
    pred = pred.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def run_config(cfg_id, lr, batch_size, patch_size, width, enc_blocks,
               ssim_w, args, device, out_dir):
    print(f"\n{'='*65}")
    print(f"  CONFIG {cfg_id:02d}/{len(CONFIGS)} | lr={lr} bs={batch_size} "
          f"patch={patch_size} width={width} enc={enc_blocks} ssim={ssim_w}")
    print(f"{'='*65}")

    torch.manual_seed(42)
    random.seed(42)

    ckpt_dir = out_dir / f"cfg_{cfg_id:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    full_ds = PairedRestorationDataset(
        args.gt_dir, args.degraded_dir,
        channels=args.channels, patch_size=patch_size, train=True,
        synth_prob=args.synth_prob,
    )
    n_val   = max(1, int(len(full_ds) * 0.05))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.dataset = PairedRestorationDataset(
        args.gt_dir, args.degraded_dir,
        channels=args.channels, patch_size=patch_size, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    TARGET_STEPS = 12000
    args.probe_epochs = max(1, TARGET_STEPS // len(train_loader))

    model = NAFNetSR(
        channels=args.channels, width=width, scale=2,
        enc_blk_nums=enc_blocks, dec_blk_nums=enc_blocks,
    ).to(device)

    # No LPIPS during sweep — too slow. L1 + SSIM only.
    criterion = RestorationLoss(
        ssim_weight=ssim_w, lpips_weight=0.0, device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.9), weight_decay=1e-3)
    total_steps  = args.probe_epochs * len(train_loader)
    warmup_steps = min(500, max(1, total_steps // 20))

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(model, decay=0.999)

    best_psnr   = -1.0
    no_improve  = 0
    history     = []
    t_start     = time.time()

    for epoch in range(args.probe_epochs):
        model.train()
        running_loss = 0.0
        for deg, gt in train_loader:
            deg = deg.to(device, non_blocking=True)
            gt  = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(deg)
            # Loss in fp32: SSIM's variance terms lose all precision in fp16
            # once pred ~= target, giving negative variance -> NaN.
            loss, _, _, _ = criterion(pred.float(), gt.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            scheduler.step()
            running_loss += loss.item()

        # Validation
        ema.store_and_apply(model)
        model.eval()
        val_psnr_total = 0.0
        with torch.no_grad():
            for deg, gt in val_loader:
                deg, gt = deg.to(device), gt.to(device)
                pred = model(deg)
                val_psnr_total += psnr_metric(pred, gt)
        ep_psnr = val_psnr_total / len(val_loader)
        avg_loss = running_loss / len(train_loader)

        history.append({"epoch": epoch + 1, "loss": round(avg_loss, 5),
                         "val_psnr": round(ep_psnr, 4)})

        if ep_psnr > best_psnr:
            best_psnr  = ep_psnr
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr,
                "cfg": dict(lr=lr, batch_size=batch_size, patch_size=patch_size,
                            width=width, enc_blocks=enc_blocks, ssim_w=ssim_w),
            }, ckpt_dir / "best.pth")
            no_improve += 1
            
        ema.restore(model)

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / (epoch + 1)) * (args.probe_epochs - epoch - 1)
            print(f"  Ep {epoch+1:03d}/{args.probe_epochs} | "
                  f"loss={avg_loss:.4f} | val_psnr={ep_psnr:.2f} dB | "
                  f"best={best_psnr:.2f} dB | ETA {eta/60:.0f}m")

        # Early stopping: give up after 30 straight epochs without improvement
        if no_improve >= 30:
            print(f"  Early stop at epoch {epoch+1} (no improvement for 30 epochs)")
            break

    elapsed = time.time() - t_start
    print(f"  ★ Config {cfg_id:02d} done: best_psnr={best_psnr:.2f} dB | "
          f"time={elapsed/60:.1f}m")

    return best_psnr, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir",         required=True)
    p.add_argument("--degraded_dir",   required=True)
    p.add_argument("--channels",       type=int,   default=1)
    p.add_argument("--synth_prob",     type=float, default=0.0)
    p.add_argument("--probe_epochs",   type=int,   default=100,
                   help="Epochs per config (100-120 recommended)")
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--out_dir",        default="checkpoints/sweep",
                   help="Directory for per-config checkpoints")
    p.add_argument("--out_csv",        default="sweep_results.csv")
    p.add_argument("--log_file",       default="sweep_log.json",
                   help="Progress log — allows resuming across sessions")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load progress log (so we can resume)
    log_path = Path(args.log_file)
    if log_path.exists():
        with open(log_path) as f:
            done_log = json.load(f)
        print(f"Resuming sweep — {len(done_log)} config(s) already complete.")
    else:
        done_log = {}

    n_remaining = sum(1 for c in CONFIGS if str(c[0]) not in done_log)
    print(f"\nDevice: {device}")
    print(f"Total configs: {len(CONFIGS)} | Remaining: {n_remaining}")
    print(f"Epochs per config: {args.probe_epochs}")
    print(f"Estimated time remaining: "
          f"~{n_remaining * args.probe_epochs * 35 / 3600:.1f}h "
          f"(rough T4 estimate)\n")

    all_results = list(done_log.values())

    for cfg in CONFIGS:
        cfg_id = cfg[0]
        if str(cfg_id) in done_log:
            print(f"  Config {cfg_id:02d} — already done "
                  f"({done_log[str(cfg_id)]['best_psnr']:.2f} dB), skipping.")
            continue

        cfg_id, lr, bs, ps, width, enc, ssim_w = cfg
        best_psnr, history = run_config(
            cfg_id, lr, bs, ps, width, enc, ssim_w, args, device, out_dir
        )

        entry = {
            "cfg_id": cfg_id, "lr": lr, "batch_size": bs,
            "patch_size": ps, "width": width, "enc_blocks": str(enc),
            "ssim_weight": ssim_w, "best_psnr": round(best_psnr, 4),
            "history": history,
        }
        all_results.append(entry)
        done_log[str(cfg_id)] = entry

        # Save progress immediately after each config
        with open(log_path, "w") as f:
            json.dump(done_log, f, indent=2)

        # Append to CSV
        csv_exists = Path(args.out_csv).exists()
        with open(args.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "cfg_id", "lr", "batch_size", "patch_size", "width",
                "enc_blocks", "ssim_weight", "best_psnr"
            ])
            if not csv_exists:
                writer.writeheader()
            writer.writerow({k: entry[k] for k in writer.fieldnames})

    # ── Final Leaderboard ──
    all_results.sort(key=lambda r: r["best_psnr"], reverse=True)
    print(f"\n{'='*70}")
    print("  SWEEP COMPLETE — Full Leaderboard")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'ID':<4} {'lr':<8} {'bs':<4} {'patch':<6} "
          f"{'width':<6} {'enc':<12} {'ssim':<6} {'PSNR':>8}")
    print("-" * 70)
    for rank, r in enumerate(all_results, 1):
        marker = " ★" if rank == 1 else ""
        print(f"  {rank:<4} {r['cfg_id']:<4} {r['lr']:<8} {r['batch_size']:<4} "
              f"{r['patch_size']:<6} {r['width']:<6} {r['enc_blocks']:<12} "
              f"{r['ssim_weight']:<6} {r['best_psnr']:>8.2f} dB{marker}")

    best = all_results[0]
    print(f"\n  ★ WINNER: Config {best['cfg_id']} → {best['best_psnr']:.2f} dB")
    print(f"\n  Run full 300-epoch training with winner:")
    print(f"    python train.py \\")
    print(f"      --gt_dir {args.gt_dir} \\")
    print(f"      --degraded_dir {args.degraded_dir} \\")
    print(f"      --channels {args.channels} \\")
    print(f"      --epochs 300 \\")
    print(f"      --lr {best['lr']} \\")
    print(f"      --batch_size {best['batch_size']} \\")
    print(f"      --patch_size {best['patch_size']} \\")
    print(f"      --ssim_weight {best['ssim_weight']} \\")
    print(f"      --amp")
    print(f"\n  Checkpoints per config: checkpoints/sweep/cfg_<id>/best.pth")
    print(f"  Full results: {args.out_csv}")
    print(f"  Progress log: {args.log_file}")


if __name__ == "__main__":
    main()
