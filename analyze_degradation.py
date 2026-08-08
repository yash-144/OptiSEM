"""
analyze_degradation.py — reverse-engineer the KLA degradation model.

Goal: figure out exactly how NoisyLR was produced from GT, so you can
synthesize unlimited (GT, degraded) pairs with RANDOMIZED noise levels.
That is the single highest-leverage thing you can do for an
out-of-distribution test set.

Model assumed:
    LR_clean = downsample(GT, kernel, factor=2)
    LR_noisy = LR_clean * (1 + s)  +  n
        s ~ N(0, speckle_var)      <- multiplicative (speckle)
        n ~ N(0, gauss_var)        <- additive (Gaussian)

Consequence:  Var(residual | mu) = speckle_var * mu^2 + gauss_var
So we bin residuals by local clean intensity mu, compute variance per bin,
and fit a straight line in mu^2. Slope = speckle_var, intercept = gauss_var.

Usage:
    python analyze_degradation.py \
        --gt_dir dataset/train/GT \
        --degraded_dir dataset/train/NoisyLR \
        --n_samples 200
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def list_pairs(gt_dir, deg_dir):
    def listing(root):
        out = []
        for ext in ("*.npy", "*.png", "*.tif", "*.tiff"):
            out.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        return out

    gt = {Path(f).stem: f for f in listing(gt_dir)}
    dg = {Path(f).stem: f for f in listing(deg_dir)}
    common = sorted(set(gt) & set(dg))
    if not common:
        raise RuntimeError("No matching stems between GT and degraded dirs.")
    return [(gt[k], dg[k]) for k in common]


def load(path):
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr[0]
    return arr


def downsample(gt, mode, factor=2):
    t = torch.from_numpy(gt)[None, None]
    if mode == "area":
        out = F.interpolate(t, scale_factor=1 / factor, mode="area")
    else:
        out = F.interpolate(t, scale_factor=1 / factor, mode=mode,
                            align_corners=False, antialias=True)
    return out[0, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--bins", type=int, default=24)
    args = p.parse_args()

    pairs = list_pairs(args.gt_dir, args.degraded_dir)
    print(f"Found {len(pairs)} pairs. Using {min(args.n_samples, len(pairs))}.\n")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), size=min(args.n_samples, len(pairs)), replace=False)

    # ── Step 0: shape / range sanity ──────────────────────────────────
    g0, d0 = load(pairs[idx[0]][0]), load(pairs[idx[0]][1])
    print(f"GT  shape={g0.shape} range=[{g0.min():.4f}, {g0.max():.4f}]")
    print(f"LR  shape={d0.shape} range=[{d0.min():.4f}, {d0.max():.4f}]")
    ratio = g0.shape[0] / d0.shape[0]
    print(f"Scale ratio: {ratio:.2f}x")
    if abs(ratio - 2.0) > 1e-6:
        print("  !! Not a clean 2x pair — your dataset.py silently resizes GT "
              "when this happens. Check whether you have mixed 2x / 4x data.")
    print()

    # ── Step 1: which downsample kernel? ──────────────────────────────
    # The true kernel is the one leaving the smallest residual mean-abs error.
    print("Step 1 — identifying the downsample kernel")
    print(f"  {'kernel':<10} {'mean |residual|':>16}")
    kernel_scores = {}
    for mode in ("bicubic", "bilinear", "area", "nearest-exact"):
        errs = []
        for i in idx[:50]:
            gt, lr = load(pairs[i][0]), load(pairs[i][1])
            try:
                clean = downsample(gt, mode)
            except Exception:
                break
            if clean.shape != lr.shape:
                break
            errs.append(np.abs(lr - clean).mean())
        if errs:
            kernel_scores[mode] = float(np.mean(errs))
            print(f"  {mode:<10} {kernel_scores[mode]:>16.6f}")
    best_kernel = min(kernel_scores, key=kernel_scores.get)
    print(f"  -> best guess: {best_kernel}\n")

    # ── Step 2: fit the noise model ───────────────────────────────────
    print("Step 2 — fitting Var(residual) = speckle_var * mu^2 + gauss_var")
    mus, residuals = [], []
    for i in idx:
        gt, lr = load(pairs[i][0]), load(pairs[i][1])
        clean = downsample(gt, best_kernel)
        if clean.shape != lr.shape:
            continue
        mus.append(clean.ravel())
        residuals.append((lr - clean).ravel())
    mu = np.concatenate(mus)
    r = np.concatenate(residuals)

    lo, hi = np.percentile(mu, 1), np.percentile(mu, 99)
    edges = np.linspace(lo, hi, args.bins + 1)
    which = np.digitize(mu, edges) - 1
    bin_mu, bin_var = [], []
    for b in range(args.bins):
        m = which == b
        if m.sum() < 500:
            continue
        bin_mu.append(mu[m].mean())
        bin_var.append(r[m].var())
    bin_mu, bin_var = np.array(bin_mu), np.array(bin_var)

    A = np.stack([bin_mu ** 2, np.ones_like(bin_mu)], axis=1)
    coef, *_ = np.linalg.lstsq(A, bin_var, rcond=None)
    speckle_var, gauss_var = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = ((bin_var - pred) ** 2).sum()
    ss_tot = ((bin_var - bin_var.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"  {'mu':>8} {'Var(resid)':>14} {'fit':>14}")
    for m, v, q in zip(bin_mu, bin_var, pred):
        print(f"  {m:>8.4f} {v:>14.6f} {q:>14.6f}")
    print()
    print(f"  speckle_var = {speckle_var:.6f}   (speckle_std = "
          f"{max(speckle_var, 0) ** 0.5:.4f})")
    print(f"  gauss_var   = {gauss_var:.6f}   (gauss_sigma = "
          f"{max(gauss_var, 0) ** 0.5:.4f})")
    print(f"  R^2         = {r2:.4f}")
    print()
    if r2 < 0.8:
        print("  R^2 is low — the simple speckle+Gaussian model does not explain")
        print("  the residual. Likely extras: blur before downsampling, Poisson")
        print("  (shot) noise, or a non-Gaussian speckle distribution. Try")
        print("  plotting Var vs mu (linear, not mu^2) to test for Poisson.")
    else:
        print("  Good fit. Use these as the CENTER of your randomized range.")
    print()

    # ── Step 3: the generator you should train with (REJECTED) ────────
    print("=" * 68)
    print("Step 3 — NOTE: THIS GENERATOR WAS REJECTED (1106% dark-bin error).")
    print("It contradicts the power-law model we actually trained on.")
    print("=" * 68)
    print(f'''
def degrade(gt, rng):
    """gt: float32 tensor [C,H,W] in [0,1]. Returns degraded LR tensor."""
    import torch.nn.functional as F
    # randomize the kernel so the model never memorizes one resampler
    mode = rng.choice(["bicubic", "bilinear", "area"])
    if mode == "area":
        lr = F.interpolate(gt[None], scale_factor=0.5, mode="area")[0]
    else:
        lr = F.interpolate(gt[None], scale_factor=0.5, mode=mode,
                           align_corners=False, antialias=True)[0]

    # randomize noise strength 0.3x - 2.5x the fitted values.
    # THIS is what buys you out-of-distribution robustness.
    sp = {max(speckle_var, 1e-8) ** 0.5:.5f} * rng.uniform(0.3, 2.5)
    gs = {max(gauss_var, 1e-8) ** 0.5:.5f} * rng.uniform(0.3, 2.5)

    lr = lr * (1 + torch.randn_like(lr) * sp) + torch.randn_like(lr) * gs
    return lr          # deliberately NOT clamped — matches the real data
''')
    print("Then: 20-30%% of batches from the ORIGINAL fixed pairs (so you stay")
    print("calibrated to the real distribution), 70-80%% synthesized on the fly.")


if __name__ == "__main__":
    main()
