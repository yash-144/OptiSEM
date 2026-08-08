#!/usr/bin/env python3
"""
degradation_engine.py

Two jobs:

  probe   Disambiguate the exact downsample operator and the speckle
          distribution shape, using the KLA training pairs.
              python degradation_engine.py probe --gt_dir GT --lr_dir LR --n 100

  verify  Synthesise degraded images from GT with the matched engine and check
          the synthetic statistics against the real ones.
              python degradation_engine.py verify --gt_dir GT --lr_dir LR --n 100

The Degrader class at the bottom is what you import into your dataloader.

Measured from 200 pairs (256->128, float32, GT min-max normalised to [0,1]):
  scale                 exactly 2.0
  sub-pixel shift       0.000 +- 0.09 px  -> none
  mean preservation     1.00008           -> noise is zero-mean multiplicative
  sigma = sqrt(a)       median 0.153, p10 0.065, p90 0.221
  additive gaussian     indistinguishable from zero
  clipping              none (LR reaches 1.36)

Deps: numpy, opencv-python, Pillow, scipy. torch optional.
"""

import argparse
import glob
import os

import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

EXTS = (".npy", ".png", ".tif", ".tiff", ".bmp")

# Measured speckle strength. Widened ~25% beyond the observed p10/p90 so the
# model sees degradations harsher and milder than anything in training.
SIGMA_LO, SIGMA_HI = 0.055, 0.315
SIGMA_MED = 0.171


# ------------------------------------------------------------------ downsamplers

def ds_cv2(x, f, interp):
    h, w = x.shape
    return cv2.resize(x.astype(np.float32), (w // f, h // f),
                      interpolation=interp).astype(np.float64)


def ds_pil(x, f, resample):
    h, w = x.shape
    im = Image.fromarray(x.astype(np.float32), mode="F")
    return np.asarray(im.resize((w // f, h // f), resample), dtype=np.float64)


def ds_torch(x, f, mode, antialias):
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    h, w = x.shape
    o = F.interpolate(t, size=(h // f, w // f), mode=mode,
                      align_corners=False, antialias=antialias)
    return o[0, 0].numpy().astype(np.float64)


def downsamplers():
    d = {
        "cv2_cubic":        lambda x, f: ds_cv2(x, f, cv2.INTER_CUBIC),
        "cv2_area":         lambda x, f: ds_cv2(x, f, cv2.INTER_AREA),
        "cv2_lanczos4":     lambda x, f: ds_cv2(x, f, cv2.INTER_LANCZOS4),
        "pil_bicubic":      lambda x, f: ds_pil(x, f, Image.BICUBIC),
        "pil_lanczos":      lambda x, f: ds_pil(x, f, Image.LANCZOS),
        "pil_bilinear":     lambda x, f: ds_pil(x, f, Image.BILINEAR),
    }
    if HAS_TORCH:
        d["torch_bicubic_aa"]   = lambda x, f: ds_torch(x, f, "bicubic", True)
        d["torch_bicubic_noaa"] = lambda x, f: ds_torch(x, f, "bicubic", False)
        d["torch_bilinear_aa"]  = lambda x, f: ds_torch(x, f, "bilinear", True)
    return d


# ------------------------------------------------------------------------- utils

def imread_any(p):
    if p.lower().endswith(".npy"):
        a = np.load(p)
    else:
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if a is None:
        raise IOError(p)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float64)


def find_pairs(gt_dir, lr_dir):
    def index(d):
        out = {}
        for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if p.lower().endswith(EXTS):
                out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
        return out
    g, l = index(gt_dir), index(lr_dir)
    common = sorted(set(g) & set(l))
    if common:
        return [(g[k], l[k]) for k in common]
    return list(zip(sorted(g.values()), sorted(l.values())))


def lag1_autocorr(r):
    r = r - r.mean()
    v = float(np.mean(r * r)) + 1e-12
    return 0.5 * (float(np.mean(r[:, :-1] * r[:, 1:]) / v)
                  + float(np.mean(r[:-1, :] * r[1:, :]) / v))


# ------------------------------------------------------------------------- probe

def cmd_probe(args):
    pairs = find_pairs(args.gt_dir, args.lr_dir)[:args.n]
    print(f"probing {len(pairs)} pairs\n")

    dss = downsamplers()
    mses = {k: [] for k in dss}
    acs = {k: [] for k in dss}
    skews, kurts = [], []

    for gp, lp in pairs:
        gt, lr = imread_any(gp), imread_any(lp)
        f = gt.shape[0] // lr.shape[0]
        best_name, best_mse, best_down = None, np.inf, None
        for name, fn in dss.items():
            try:
                d = fn(gt, f)
            except Exception:
                continue
            r = lr - d
            m = float(np.mean(r ** 2))
            mses[name].append(m)
            acs[name].append(lag1_autocorr(r))
            if m < best_mse:
                best_name, best_mse, best_down = name, m, d

        # speckle shape: normalise residual by local intensity, in bright areas
        mu = best_down
        m = mu > np.percentile(mu, 60)
        g = (lr[m] - mu[m]) / (mu[m] + 1e-6)
        g = g[np.isfinite(g)]
        if g.size > 500:
            s = g.std() + 1e-12
            skews.append(float(np.mean(((g - g.mean()) / s) ** 3)))
            kurts.append(float(np.mean(((g - g.mean()) / s) ** 4) - 3.0))

    print(f"{'operator':<22}{'median MSE':>14}{'rel.':>9}{'lag1 autocorr':>16}")
    print("-" * 61)
    rank = sorted([(float(np.median(v)), k) for k, v in mses.items() if v])
    base = rank[0][0]
    for m, k in rank:
        print(f"{k:<22}{m:>14.3e}{m/base:>9.3f}{float(np.median(acs[k])):>16.4f}")

    print(f"\nWINNER: {rank[0][1]}")
    print("  If the winner's |lag1 autocorr| is < ~0.01, the operator is exact.")
    print("  If it stays near -0.05 for every candidate, the pipeline did")
    print("  something extra (e.g. blur before decimation) - come back to it.")
    if len(rank) > 1:
        print(f"  Runner-up is {rank[1][0]/base:.3f}x worse. Under ~1.02x means")
        print("  the two are indistinguishable at this noise level; use either.")

    if skews:
        print(f"\nspeckle shape (residual / local intensity, bright regions):")
        print(f"  skewness  median {np.median(skews):+.3f}")
        print(f"  ex.kurt   median {np.median(kurts):+.3f}")
        print("  |skew| < 0.15 and |kurt| < 0.3  -> gaussian multiplicative:")
        print("      y = x * (1 + sigma * randn)")
        print("  skew > 0.4                      -> gamma speckle:")
        print("      y = x * gamma(L, 1/L),  L ~ 1/sigma^2")


# ------------------------------------------------------------------ the degrader

class Degrader:
    """Matched degradation engine. Call on a float32 GT crop in [0, 1]."""

    def __init__(self, downsampler="cv2_cubic", scale=2,
                 sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
                 gamma_speckle_prob=1.0, additive_hi=0.03,
                 blur_prob=0.0, kernel_jitter_prob=0.15, rng=None):
        self.ds_name = downsampler
        self.ds = downsamplers()[downsampler]
        self.alts = [k for k in downsamplers() if k != downsampler]
        self.scale = scale
        self.sigma_lo, self.sigma_hi = sigma_lo, sigma_hi
        self.gamma_p = gamma_speckle_prob
        self.additive_hi = additive_hi
        self.blur_p = blur_prob
        self.jitter_p = kernel_jitter_prob
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def minmax(x):
        """Match KLA's convention: every GT image is min-max scaled to [0,1]."""
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def __call__(self, gt_crop):
        gt = np.asarray(gt_crop, dtype=np.float64)
        # 0. gamma / contrast jitter BEFORE min-max
        gt = np.clip(gt, 0, None)
        if gt.max() > 0:
            gt = np.power(gt / gt.max(), self.rng.uniform(0.6, 1.7))
            
        # 1. renormalise the CROP, not the parent image
        gt = self.minmax(gt)

        x = gt
        # 2. optional pre-blur - off by default, the data shows none.
        #    Enable at low probability as OOD insurance only.
        if self.blur_p and self.rng.random() < self.blur_p:
            s = self.rng.uniform(0.3, 1.0)
            x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s).astype(np.float64)

        # 3. downsample; occasionally swap the kernel so the model is not
        #    brittle to the exact resampler used on the test set
        ds = self.ds
        if self.jitter_p and self.rng.random() < self.jitter_p:
            ds = downsamplers()[self.alts[self.rng.integers(len(self.alts))]]
        lr = ds(x, self.scale)

        # 4. zero-mean multiplicative speckle - the dominant term by far
        sigma = self.rng.uniform(self.sigma_lo, self.sigma_hi)
        if self.gamma_p and self.rng.random() < self.gamma_p:
            L = max(1.0, 1.0 / (sigma ** 2))
            g = self.rng.gamma(L, 1.0 / L, size=lr.shape)
        else:
            g = 1.0 + sigma * self.rng.standard_normal(lr.shape)
        lr = lr * g

        # 5. tiny additive term - measured as ~0, kept as a small OOD hedge
        if self.additive_hi > 0:
            lr = lr + self.rng.uniform(0, self.additive_hi) \
                 * self.rng.standard_normal(lr.shape)

        # 6. NO clipping on the input. GT stays in [0,1].
        return lr.astype(np.float32), gt.astype(np.float32)


# ------------------------------------------------------------------------ verify

def cmd_verify(args):
    pairs = find_pairs(args.gt_dir, args.lr_dir)[:args.n]
    deg = Degrader(downsampler=args.downsampler, additive_hi=0.0,
                   kernel_jitter_prob=0.0,
                   sigma_lo=SIGMA_MED, sigma_hi=SIGMA_MED)

    def stats(lr, gt_dn):
        r = lr - gt_dn
        m = gt_dn > np.percentile(gt_dn, 60)
        return (float((lr > 1.0).mean() + (lr < 0.0).mean()),
                float(lr.max()),
                float(np.std(r[m] / (gt_dn[m] + 1e-6))))

    real, synth = [], []
    ds = downsamplers()[args.downsampler]
    for gp, lp in pairs:
        gt, lr = imread_any(gp), imread_any(lp)
        f = gt.shape[0] // lr.shape[0]
        real.append(stats(lr, ds(gt, f)))
        slr, sgt = deg(gt)
        synth.append(stats(slr.astype(np.float64), ds(sgt.astype(np.float64), f)))

    real, synth = np.array(real), np.array(synth)
    names = ["out-of-range pixel frac", "max value", "relative noise std"]
    print(f"\n{'statistic':<26}{'REAL':>12}{'SYNTHETIC':>14}")
    print("-" * 52)
    for i, n in enumerate(names):
        print(f"{n:<26}{np.median(real[:, i]):>12.4f}{np.median(synth[:, i]):>14.4f}")
    print("\nIf these three lines match, your engine reproduces KLA's degradation")
    print("and you can generate unlimited training pairs from ANY grayscale image.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--gt_dir", required=True)
        p.add_argument("--lr_dir", required=True)
        p.add_argument("--n", type=int, default=100)
        if name == "verify":
            p.add_argument("--downsampler", default="pil_bicubic")
    args = ap.parse_args()
    (cmd_probe if args.cmd == "probe" else cmd_verify)(args)


if __name__ == "__main__":
    main()
