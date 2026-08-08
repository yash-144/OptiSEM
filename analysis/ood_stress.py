"""
ood_stress.py — does the synthetic-degradation training actually generalize?

Applies degradations to held-out GT under conditions the model NEVER saw,
and reports PSNR/SSIM/LPIPS for your model against a bicubic baseline.

Conditions:
  real          the actual KLA NoisyLR pairs (in-distribution reference)
  noise 0.3x    bottom edge of the training range
  noise 1.0x    centre of the training range
  noise 2.5x    top edge of the training range
  noise 4.0x    BEYOND training  <- this is the OOD claim
  noise 6.0x    BEYOND training  <- this is the OOD claim
  gauss-blur    blur-then-subsample: a downsample operator never trained on
  additive      pure additive Gaussian: the WRONG noise model entirely

    python ood_stress.py --gt_dir dataset/train/GT \
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
import lpips

from model import NAFNetSR

VAL_STRIDE = 20          # must match train.py / metrics.py
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


def gaussian_blur(x, sigma=1.0):
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x


def degrade(gt, kind, scale):
    """gt: [1,C,H,W] in [0,1] -> degraded [1,C,H/2,W/2]."""
    if kind == "gauss-blur":
        lr = gaussian_blur(gt, 1.0)[:, :, ::2, ::2].clamp_min(0.0)
    else:
        lr = F.interpolate(gt, scale_factor=0.5, mode="area").clamp_min(0.0)

    if kind == "additive":               # wrong noise model: flat sigma
        return lr + torch.randn_like(lr) * (NOISE_A * scale * 0.5)
    sigma = NOISE_A * scale * lr.pow(NOISE_P)
    return lr + torch.randn_like(lr) * sigma


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
    hr_blocks = int(cfg.get("hr_blocks", 0))
    model = NAFNetSR(channels=a.channels, width=int(cfg.get("width", 32)),
                     scale=2, enc_blk_nums=list(enc), dec_blk_nums=list(enc),
                     hr_blocks=hr_blocks)
    model.load_state_dict(st); model = model.to(dev).eval()

    lp = lpips.LPIPS(net="alex").to(dev)

    CONDITIONS = [
        ("real (in-dist)", None,         None),
        ("noise 0.3x",     "powerlaw",   0.3),
        ("noise 1.0x",     "powerlaw",   1.0),
        ("noise 2.5x",     "powerlaw",   2.5),
        ("noise 4.0x  OOD","powerlaw",   4.0),
        ("noise 6.0x  OOD","powerlaw",   6.0),
        ("gauss-blur  OOD","gauss-blur", 1.0),
        ("additive    OOD","additive",   1.0),
    ]

    print(f"{'condition':<17}{'bicubic PSNR':>14}{'model PSNR':>12}"
          f"{'gain':>8}{'SSIM':>9}{'LPIPS':>9}")
    print("-" * 69)

    with torch.no_grad():
        for name, kind, scale in CONDITIONS:
            rows = []
            for s in stems:
                g = load(gt_m[s]).to(dev)
                d = load(dg_m[s]).to(dev) if kind is None else degrade(g, kind, scale)
                bic = F.interpolate(d, scale_factor=2, mode="bicubic",
                                    align_corners=False).clamp(0, 1)
                out = model(d).clamp(0, 1)
                rows.append((
                    psnr(bic, g), psnr(out, g),
                    ssim_fn(out, g.clamp(0, 1), data_range=1.0).item(),
                    lp(out.repeat(1, 3, 1, 1) * 2 - 1,
                       g.repeat(1, 3, 1, 1) * 2 - 1).item(),
                ))
            m = np.mean(rows, axis=0)
            print(f"{name:<17}{m[0]:>14.3f}{m[1]:>12.3f}{m[1]-m[0]:>+8.2f}"
                  f"{m[2]:>9.4f}{m[3]:>9.4f}")

    print("\nRead it this way: the 'gain' column on the OOD rows is your claim.")
    print("If it stays clearly positive at 4x/6x and on the unseen operators,")
    print("your degradation pipeline generalizes and you can say so on Slide 5.")
    print("If it collapses, say that honestly and show the limit — a measured")
    print("failure boundary reads far better to a reviewer than no test at all.")


if __name__ == "__main__":
    main()
