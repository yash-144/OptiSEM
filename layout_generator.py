#!/usr/bin/env python3
"""
layout_generator.py

Generates synthetic grayscale "SEM-like" images of periodic semiconductor
structures, to cover the content gap in the KLA PS01 training set (which is
100% organic/natural imagery and contains no circuit layouts at all).

Output: float32 .npy, min-max normalised to [0,1], matching KLA's GT convention.
Feed these straight into Degrader() as ground truth.

    python layout_generator.py --n 4000 --out data/layouts --size 256

Styles: dram, finfet, contacts, logic, grating  (default: all, evenly mixed)

Everything is drawn at SS x supersampling then box-downsampled, so edges are
properly antialiased rather than jagged - jagged synthetic edges are a strong
tell that a network can latch onto instead of learning real structure.

Deps: numpy, opencv-python.
"""

import argparse
import os

import numpy as np
import cv2

SS = 4  # supersample factor


# ------------------------------------------------------------------- primitives

def _canvas(n, rng):
    """Dark field with a faint low-frequency charging gradient."""
    g = rng.uniform(0.0, 0.10)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * xx + np.sin(ang) * yy
    return (rng.uniform(0.02, 0.12) + g * grad).astype(np.float32)


def _ler(length, amp, rng, corr=8.0):
    """Line-edge roughness: correlated 1D jitter along a feature."""
    w = rng.standard_normal(length).astype(np.float32)
    k = max(3, int(corr) | 1)
    w = cv2.GaussianBlur(w.reshape(-1, 1), (1, k), 0).ravel()
    s = w.std() + 1e-6
    return (w / s) * amp


def _bars(img, pitch, width, vertical, level, rng, ler_amp):
    """Draw a periodic set of bars with line-edge roughness."""
    n = img.shape[0]
    off = rng.integers(0, max(1, pitch))
    for c in range(-pitch, n + pitch, pitch):
        x0 = c + off
        jit = _ler(n, ler_amp, rng)
        for i in range(n):
            a = int(round(x0 + jit[i]))
            b = a + width
            if b <= 0 or a >= n:
                continue
            a, b = max(a, 0), min(b, n)
            if vertical:
                img[i, a:b] = level
            else:
                img[a:b, i] = level
    return img


# ----------------------------------------------------------------------- styles

def gen_dram(n, rng):
    img = _canvas(n, rng)
    p1 = int(rng.integers(12, 40)) * SS
    p2 = int(rng.integers(12, 40)) * SS
    w1 = max(2, int(p1 * rng.uniform(0.30, 0.55)))
    w2 = max(2, int(p2 * rng.uniform(0.30, 0.55)))
    lv = rng.uniform(0.45, 0.75)
    ler = rng.uniform(0.0, 1.6) * SS
    _bars(img, p1, w1, False, lv, rng, ler)
    _bars(img, p2, w2, True, lv * rng.uniform(0.85, 1.15), rng, ler)
    # contact/via dot at every intersection
    r = max(4 * SS, int(min(w1, w2) * rng.uniform(0.60, 0.95)))
    dot = lv * rng.uniform(1.2, 1.5)
    o1, o2 = rng.integers(0, p1), rng.integers(0, p2)
    for y in range(o1, n, p1):
        for x in range(o2, n, p2):
            cv2.circle(img, (int(x + w2 // 2), int(y + w1 // 2)), r, float(dot), -1)
    return img


def gen_finfet(n, rng):
    img = _canvas(n, rng)
    p = int(rng.integers(8, 26)) * SS
    w = max(2, int(p * rng.uniform(0.28, 0.50)))
    _bars(img, p, w, True, rng.uniform(0.40, 0.70), rng, rng.uniform(0, 1.4) * SS)
    # one or two horizontal gate bars crossing the fins
    for _ in range(int(rng.integers(1, 3))):
        gw = int(rng.integers(max(4, p), max(6, 2 * p)))
        gy = int(rng.integers(0, max(1, n - gw)))
        img[gy:gy + gw, :] = rng.uniform(0.72, 1.00)
    return img


def gen_contacts(n, rng):
    img = _canvas(n, rng)
    p = int(rng.integers(12, 40)) * SS
    r = max(2, int(p * rng.uniform(0.18, 0.36)))
    lv = rng.uniform(0.65, 1.00)
    hexed = rng.random() < 0.4
    for j, y in enumerate(range(int(rng.integers(0, p)), n + p, p)):
        sh = (p // 2) if (hexed and j % 2) else 0
        for x in range(int(rng.integers(0, p)) + sh, n + p, p):
            cv2.circle(img, (int(x), int(y)), r, float(lv * rng.uniform(0.9, 1.0)), -1)
    return img


def gen_logic(n, rng):
    """Manhattan polygons - irregular but strictly axis-aligned."""
    img = _canvas(n, rng)
    for _ in range(int(rng.integers(14, 70))):
        w = int(rng.integers(3 * SS, 26 * SS))
        h = int(rng.integers(3 * SS, 26 * SS))
        if rng.random() < 0.5:
            w, h = h, w
        x = int(rng.integers(0, max(1, n - w)))
        y = int(rng.integers(0, max(1, n - h)))
        img[y:y + h, x:x + w] = rng.uniform(0.35, 1.00)
    return img


def gen_grating(n, rng):
    img = _canvas(n, rng)
    p = int(rng.integers(6, 40)) * SS
    w = max(2, int(p * rng.uniform(0.25, 0.60)))
    _bars(img, p, w, rng.random() < 0.5, rng.uniform(0.5, 1.0), rng,
          rng.uniform(0, 2.0) * SS)
    return img


STYLES = {"dram": gen_dram, "finfet": gen_finfet, "contacts": gen_contacts,
          "logic": gen_logic, "grating": gen_grating}


# ------------------------------------------------------------------- sem realism

def sem_finish(img, n_out, rng):
    n = img.shape[0]

    # multi-layer structure: add a second faint layer at different pitch
    if rng.random() < 0.4:
        bg = _canvas(n, rng)
        p = int(rng.integers(16, 50)) * SS
        w = max(2, int(p * rng.uniform(0.2, 0.5)))
        _bars(bg, p, w, rng.random() < 0.5, rng.uniform(0.3, 0.6), rng, rng.uniform(0, 2.0)*SS)
        img = img + bg * rng.uniform(0.4, 0.7)

    # defects: occasional bridging (bright blobs) or missing features (dark blobs)
    if rng.random() < 0.6:
        for _ in range(int(rng.integers(1, 8))):
            r = int(rng.integers(2 * SS, 12 * SS))
            x, y = int(rng.integers(0, n)), int(rng.integers(0, n))
            if rng.random() < 0.5:
                cv2.circle(img, (x, y), r, float(rng.uniform(0.7, 1.2)), -1)
            else:
                cv2.circle(img, (x, y), r, 0.0, -1)

    # global rotation before downsampling, so edges stay antialiased
    if rng.random() < 0.7:
        ang = rng.uniform(-90, 90) if rng.random() < 0.3 else rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)

    img = cv2.resize(img, (n_out, n_out), interpolation=cv2.INTER_AREA)

    # beam spot blur
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.4))

    # edge brightening: SEM secondary-electron yield peaks at feature edges
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag /= (mag.max() + 1e-6)
    img = img + rng.uniform(0.15, 0.65) * cv2.GaussianBlur(
        mag, (0, 0), rng.uniform(0.4, 1.1))

    # local charging halo around bright structures
    if rng.random() < 0.5:
        halo = cv2.GaussianBlur(img, (0, 0), rng.uniform(4, 12))
        img = img + rng.uniform(0.05, 0.20) * halo

    # gamma / contrast jitter BEFORE min-max, so the histogram shape varies
    img = np.clip(img, 0, None)
    img = np.power(img / (img.max() + 1e-6), rng.uniform(0.6, 1.7))

    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return img.astype(np.float32)


def generate(style, n_out, rng):
    img = STYLES[style](n_out * SS, rng)
    return sem_finish(img, n_out, rng)


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="layouts")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--style", default="all", choices=["all"] + list(STYLES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true",
                    help="also write a 5x5 png contact sheet")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    styles = list(STYLES) if args.style == "all" else [args.style]

    tiles = []
    for i in range(args.n):
        s = styles[i % len(styles)]
        img = generate(s, args.size, rng)
        np.save(os.path.join(args.out, f"{s}_{i:06d}.npy"), img)
        if args.preview and len(tiles) < 25:
            tiles.append(img)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.n}")

    if args.preview and tiles:
        while len(tiles) < 25:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(5)])
        cv2.imwrite(os.path.join(args.out, "preview.png"),
                    (np.clip(sheet, 0, 1) * 255).astype(np.uint8))
        print("wrote preview.png")

    print(f"wrote {args.n} layouts to {args.out}/")


if __name__ == "__main__":
    main()
