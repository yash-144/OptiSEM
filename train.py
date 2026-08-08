"""
Train NAFNetSR on paired (degraded, ground-truth) images.

    python train.py --gt_dir dataset/train/GT --degraded_dir dataset/train/NoisyLR \
        --channels 1 --epochs 250 --lr 5e-4 --batch_size 8 --patch_size 96 \
        --width 32 --enc_blocks 1,1,1,1 --ssim_weight 0.2 --lpips_weight 0.1 \
        --synth_prob 0.75 --amp

Resume an interrupted run with --resume checkpoints/last.pth. Keep every
other flag IDENTICAL on resume: --epochs defines the cosine horizon, so
changing it silently changes the schedule.

  best.pth  -> EMA weights, best val PSNR. THIS IS WHAT YOU SUBMIT.
  last.pth  -> raw weights + EMA shadow + optimizer + scaler + step count.
               For resuming only.
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import PairedRestorationDataset
from losses import RestorationLoss
from model import NAFNetSR


class EMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
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


def psnr(pred, target):
    mse = torch.mean((pred.clamp(0, 1) - target) ** 2).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--patch_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ssim_weight", type=float, default=0.2)
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--enc_blocks", type=str, default="1,1,1,1")
    p.add_argument("--hr_blocks", type=int, default=2)
    p.add_argument("--synth_prob", type=float, default=0.75,
                   help="fraction of training samples degraded synthetically")
    p.add_argument("--noise_a", type=float, default=0.1673)
    p.add_argument("--noise_p", type=float, default=0.811)
    p.add_argument("--val_stride", type=int, default=20,
                   help="every Nth image is held out. MUST match metrics.py.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    enc_blocks = [int(x) for x in args.enc_blocks.split(",")]
    dec_blocks = enc_blocks

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training set: synthetic degradation on. Val set: REAL pairs only —
    # never validate against your own noise model.
    train_full = PairedRestorationDataset(
        args.gt_dir, args.degraded_dir, channels=args.channels,
        patch_size=args.patch_size, train=True,
        synth_prob=args.synth_prob, noise_a=args.noise_a, noise_p=args.noise_p,
    )
    val_full = PairedRestorationDataset(
        args.gt_dir, args.degraded_dir, channels=args.channels,
        patch_size=args.patch_size, train=False,
    )

    # Deterministic stride split — identical rule to metrics.py, so the
    # numbers you put on Slide 6 come from data the model never saw.
    # Val must cover SEM pairs ONLY — identical protocol to Model C.
    # DIV2K images (indices >= n_pairs) are train-only.
    n_pairs = len(train_full.pairs)
    val_indices = list(range(0, n_pairs, args.val_stride))
    val_set = set(val_indices)
    train_indices = [i for i in range(len(train_full)) if i not in val_set]

    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} (stride {args.val_stride})")
    print(f"Model: NAFNetSR(width={args.width}, enc={enc_blocks}) "
          f"| synth_prob={args.synth_prob}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = NAFNetSR(channels=args.channels, width=args.width, scale=2,
                     enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks,
                     hr_blocks=args.hr_blocks).to(device)

    criterion = RestorationLoss(ssim_weight=args.ssim_weight,
                                lpips_weight=args.lpips_weight,
                                device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.9), weight_decay=1e-3)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(500, max(1, total_steps // 20))

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch, best_psnr = 0, -1.0
    if args.resume:
        # weights_only=False: the checkpoint holds lists/dicts, and PyTorch
        # >=2.6 refuses to unpickle those under the new default.
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])            # RAW weights
        optimizer.load_state_dict(ck["optimizer"])
        if "ema" in ck:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        else:
            print("  WARNING: no EMA in checkpoint, restarting the average")
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_psnr = ck.get("best_psnr", -1.0)
        # Fast-forward the LR schedule. Without this the run re-runs warmup
        # and never finishes the cosine decay.
        done = ck.get("global_step", start_epoch * steps_per_epoch)
        for _ in range(done):
            scheduler.step()
        print(f"Resumed at epoch {start_epoch}, step {done}, "
              f"best {best_psnr:.2f} dB, lr {scheduler.get_last_lr()[0]:.2e}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_ok, n_skip = 0.0, 0, 0

        for deg, gt in train_loader:
            deg = deg.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # Guard: a single bad batch must not poison the epoch or the
            # weights. Skip it, count it, keep training.
            if not (torch.isfinite(deg).all() and torch.isfinite(gt).all()):
                n_skip += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(deg)
            # Loss in fp32: SSIM's variance terms lose all precision in fp16
            # once pred ~= target, giving negative variance -> NaN.
            loss, _, _, _ = criterion(pred.float(), gt.float())

            if not torch.isfinite(loss):
                n_skip += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm):
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                running_loss += loss.item()
                n_ok += 1
            else:
                scaler.update()
                n_skip += 1
            scheduler.step()

        avg_loss = running_loss / max(n_ok, 1)

        # Validate on EMA weights — that is what gets submitted.
        ema.store_and_apply(model)
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for deg, gt in val_loader:
                deg, gt = deg.to(device), gt.to(device)
                tot += psnr(model(deg), gt)
        val_psnr = tot / len(val_loader)

        meta = {"epoch": epoch, "best_psnr": max(best_psnr, val_psnr),
                "args": {**vars(args), "enc_blocks": enc_blocks,
                         "dec_blocks": dec_blocks}}
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            meta["best_psnr"] = best_psnr
            torch.save({"model": model.state_dict(), **meta},
                       out_dir / "best.pth")          # EMA weights
        ema.restore(model)

        # Raw state for resuming — saved OUTSIDE the EMA block on purpose.
        torch.save({
            "model": model.state_dict(),              # RAW weights
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": (epoch + 1) * steps_per_epoch,
            **meta,
        }, out_dir / "last.pth")

        skip_note = f" | SKIPPED {n_skip}" if n_skip else ""
        star = "  <- best" if val_psnr >= best_psnr else ""
        print(f"Ep {epoch+1}/{args.epochs} | loss {avg_loss:.4f} | "
              f"val_psnr {val_psnr:.3f} dB | lr {scheduler.get_last_lr()[0]:.2e} "
              f"| {time.time()-t0:.0f}s{skip_note}{star}")

    print(f"\nDone. Best {best_psnr:.3f} dB -> {out_dir}/best.pth")


if __name__ == "__main__":
    main()
