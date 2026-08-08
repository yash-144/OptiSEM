"""
ceiling.py — how much headroom is actually left?

Splits your error into two independent budgets:

  A. SUPER-RESOLUTION loss  — information the 2x downsample destroyed.
     Irrecoverable without hallucinating detail.
  B. DENOISING loss         — information the noise destroyed.
     Recoverable, in principle, with a better model.

Rows printed:

  bicubic on real LR      your published baseline
  YOUR MODEL on real LR   your current number
  ---
  bicubic on CLEAN LR     no noise, no learning. Pure downsample loss.
  YOUR MODEL on CLEAN LR  <-- YOUR SR CEILING. Perfect denoising cannot beat
                              this with the current architecture.
  YOUR MODEL, noise 0.5x  what half the noise would be worth
  YOUR MODEL, noise 1.0x  synthetic sanity check, should ~= real LR row

    python ceiling.py --gt_dir dataset/train/GT \
        --degraded_dir dataset/train/NoisyLR \
        --checkpoint checkpoints/best.pth
"""

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn

from model import NAFNetSR

VAL_STRIDE = 20
NOISE_A, NOISE_P = 0.1673, 0.811


def _list(root):
    out = []
    for e in ("*.npy", "*.png", "*.tif", "*.tiff"):
        out.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    return out


def load(p):
    if p.endswith(".npy"):
        a = np.load(p).astype(np.float32)
    else:
        from PIL import Image
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = a[None]
    elif a.shape[-1] in (1, 3):
        a = a.transpose(2, 0, 1)
    return torch.from_numpy(a)[None]


def psnr(x, y):
    mse = torch.mean((x.clamp(0, 1) - y) ** 2).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--channels", type=int, default=1)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    gt_m = {Path(f).stem: f for f in _list(a.gt_dir)}
    dg_m = {Path(f).stem: f for f in _list(a.degraded_dir)}
    stems = sorted(set(gt_m) & set(dg_m))[::VAL_STRIDE]
    print(f"Held-out images: {len(stems)}\n")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("args") or ck.get("cfg") or {}
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    enc = cfg.get("enc_blocks", [1, 1, 1, 1])
    if isinstance(enc, str):
        enc = [int(t) for t in enc.strip("[]").replace(" ", "").split(",") if t]
    st = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in ck.get("model", ck).items()}
    model = NAFNetSR(channels=a.channels, width=int(cfg.get("width", 32)),
                     scale=2, enc_blk_nums=list(enc), dec_blk_nums=list(enc))
    model.load_state_dict(st); model = model.to(dev).eval()
    n_par = sum(q.numel() for q in model.parameters())
    print(f"Model: width={cfg.get('width', 32)} enc={list(enc)} "
          f"| {n_par/1e6:.2f}M params\n")

    rows = {k: [] for k in ["bic_real", "mod_real", "bic_clean", "mod_clean",
                            "mod_n05", "mod_n10"]}

    with torch.no_grad():
        for s in stems:
            g = load(gt_m[s]).to(dev)
            d = load(dg_m[s]).to(dev)
            clean = F.interpolate(g, scale_factor=0.5, mode="area").clamp_min(0.0)

            def up(x):
                return F.interpolate(x, scale_factor=2, mode="bicubic",
                                     align_corners=False).clamp(0, 1)

            def noisy(scale):
                return clean + torch.randn_like(clean) * (
                    NOISE_A * scale * clean.pow(NOISE_P))

            rows["bic_real"].append((psnr(up(d), g),
                                     ssim_fn(up(d), g, data_range=1.0).item()))
            rows["mod_real"].append((psnr(model(d), g),
                                     ssim_fn(model(d).clamp(0, 1), g, data_range=1.0).item()))
            rows["bic_clean"].append((psnr(up(clean), g),
                                      ssim_fn(up(clean), g, data_range=1.0).item()))
            rows["mod_clean"].append((psnr(model(clean), g),
                                      ssim_fn(model(clean).clamp(0, 1), g, data_range=1.0).item()))
            for tag, sc in (("mod_n05", 0.5), ("mod_n10", 1.0)):
                o = model(noisy(sc))
                rows[tag].append((psnr(o, g),
                                  ssim_fn(o.clamp(0, 1), g, data_range=1.0).item()))

    LABELS = [
        ("bic_real",  "bicubic on real LR"),
        ("mod_real",  "YOUR MODEL on real LR"),
        (None, None),
        ("bic_clean", "bicubic on CLEAN LR (no noise)"),
        ("mod_clean", "YOUR MODEL on CLEAN LR  <== SR CEILING"),
        ("mod_n05",   "YOUR MODEL, noise 0.5x"),
        ("mod_n10",   "YOUR MODEL, noise 1.0x"),
    ]
    print(f"{'':<38}{'PSNR (dB)':>12}{'SSIM':>10}")
    print("-" * 60)
    means = {}
    for k, lab in LABELS:
        if k is None:
            print("-" * 60); continue
        m = np.mean(rows[k], axis=0); means[k] = m
        print(f"{lab:<38}{m[0]:>12.3f}{m[1]:>10.4f}")

    sr_ceiling = means["mod_clean"][0]
    actual = means["mod_real"][0]
    print()
    print("=" * 60)
    print(f"  SR ceiling (perfect denoising) : {sr_ceiling:6.2f} dB")
    print(f"  You are at                     : {actual:6.2f} dB")
    print(f"  Headroom from denoising alone  : {sr_ceiling - actual:6.2f} dB")
    print("=" * 60)
    print()
    print("HOW TO READ THIS")
    print("  If the SR ceiling is BELOW your target, the target is unreachable")
    print("  without a bigger model or hallucinated detail — no amount of")
    print("  denoising work gets you there. Spend effort on capacity instead.")
    print("  If the ceiling is well ABOVE your number, noise is your bottleneck")
    print("  and better denoising (capacity, longer training, TTA) will pay.")
    print()
    print("  Caveat: the model was trained on noisy input, so CLEAN input is")
    print("  mildly out of distribution. Treat the ceiling as approximate.")


if __name__ == "__main__":
    main()
