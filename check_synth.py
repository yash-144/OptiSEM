"""
check_synth.py — is the NaN coming from the data or the model?

Runs your synthetic degradation over the whole GT set, many times, with no
model involved. Reports any non-finite output and the exact conditions.

    python check_synth.py --gt_dir dataset/train/GT --repeats 5
"""

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
    return torch.from_numpy(a)


def degrade(gt, noise_a, noise_p, clamp):
    mode = random.choice(["bicubic", "bilinear", "area"])
    x = gt.unsqueeze(0)
    if mode == "area":
        lr = F.interpolate(x, scale_factor=0.5, mode="area")
    else:
        lr = F.interpolate(x, scale_factor=0.5, mode=mode,
                           align_corners=False, antialias=True)
    lr = lr.squeeze(0)
    neg = (lr < 0).sum().item()
    if clamp:
        lr = lr.clamp_min(0.0)
    sigma = noise_a * random.uniform(0.3, 2.5) * lr.pow(noise_p)
    return lr + torch.randn_like(lr) * sigma, mode, neg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--noise_a", type=float, default=0.1673)
    p.add_argument("--noise_p", type=float, default=0.811)
    a = p.parse_args()

    files = []
    for e in ("*.npy", "*.png", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(a.gt_dir, "**", e), recursive=True))
    files = sorted(files)
    print(f"{len(files)} GT images x {a.repeats} repeats\n")

    for clamp in (False, True):
        random.seed(0); torch.manual_seed(0)
        bad, neg_imgs, neg_px, lo, hi = 0, 0, 0, 1e9, -1e9
        first = None
        for r in range(a.repeats):
            for f in files:
                out, mode, neg = degrade(load(f), a.noise_a, a.noise_p, clamp)
                if neg:
                    neg_imgs += 1
                    neg_px += neg
                if not torch.isfinite(out).all():
                    bad += 1
                    if first is None:
                        first = (Path(f).name, mode, neg)
                lo = min(lo, out.min().item()) if torch.isfinite(out).all() else lo
                hi = max(hi, out.max().item()) if torch.isfinite(out).all() else hi
        tag = "WITH clamp_min(0)" if clamp else "WITHOUT clamp_min(0)"
        print(f"--- {tag} ---")
        print(f"  non-finite outputs : {bad}")
        print(f"  crops w/ negatives : {neg_imgs}  ({neg_px} pixels total)")
        print(f"  output range       : [{lo:.4f}, {hi:.4f}]")
        if first:
            print(f"  first failure      : {first[0]} kernel={first[1]} negpx={first[2]}")
        print()

    print("VERDICT")
    print("  non-finite > 0 WITHOUT clamp and == 0 WITH clamp")
    print("     -> your dataset.py is missing clamp_min(0.0). That is the NaN.")
    print("  zero in BOTH cases")
    print("     -> the data is clean; the NaN is in the model forward under")
    print("        fp16. The guard in the new train.py will skip and log it.")


if __name__ == "__main__":
    main()
