"""
size_check.py — does the model behave differently on 256px input than on 128px?

WHY THIS EXISTS
    KLA's test set may contain 512->256 pairs. You have only ever trained on
    96px LR patches and measured on 128px LR images. NAFNet's SCA block uses a
    GLOBAL average pool, so a block's output can depend on the size and the
    overall statistics of the entire input, not just the local neighbourhood.
    If that dependence is strong, your 512->256 outputs are unmeasured and
    possibly much worse than your reported numbers.

WHAT IT DOES
    Two independent tests, both comparing "run alone at 128" against
    "run as part of a 256px input".

    MIRROR test   one image, mirror-tiled to 256x256. Content is continuous
                  across the boundary and the global mean/variance are
                  IDENTICAL to the solo run. So this isolates the effect of
                  spatial size alone.

    MOSAIC test   four different held-out images tiled into one 256x256 input.
                  Global statistics genuinely change, which is what a real
                  larger image looks like to a global pool. Seams are
                  artificial, so the script reports interior and seam-band
                  differences separately.

HOW TO READ THE OUTPUT
    interior diff tiny (< ~0.002 mean abs, < ~0.3 dB PSNR drop)
        -> size-independent. Ship as-is. Delete this worry.

    diff large ONLY in the seam band, interior clean
        -> receptive field spilling across an artificial boundary. Benign,
           it is an artefact of the test, not of your model.

    diff large in the quadrant INTERIORS
        -> global statistics are leaking. Your 512->256 behaviour differs from
           everything you have measured. FIX: in evaluation.py, split any input
           larger than 128px into 128x128 tiles with ~16px overlap, run each
           tile, and blend the overlaps. That forces inference to happen at the
           size the model was validated at.

    MIRROR clean but MOSAIC dirty
        -> confirms it is the content statistics (global pool), not the size.
           Same fix.

USAGE
    python size_check.py --gt_dir dataset/train/GT \
        --degraded_dir dataset/train/NoisyLR \
        --checkpoint /content/drive/MyDrive/kla_backup/run_C_hr/best.pth

    Add --checkpoint pointing at model_A_p96.pth to test the old model too.
    hr_blocks is read from the checkpoint cfg and defaults to 0, so BOTH
    Model A and Run C load correctly regardless of model.py's default.
"""

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluation import run_tiled
from model import NAFNetSR

VAL_STRIDE = 20          # must match train.py / metrics.py / ceiling.py
N_IMAGES = 8             # how many held-out images to test (mirror test)
SEAM_PX = 24             # HR pixels each side of a seam treated as "seam band"


# ----------------------------------------------------------------- io helpers

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


def build_model(ckpt_path, channels, dev):
    """Loads a checkpoint whether or not it has hr_blocks, and whether or not
    model.py defaults hr_blocks to 2. This is the part that breaks in your
    other scripts."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("args") or ck.get("cfg") or {}
    if not isinstance(cfg, dict):
        cfg = vars(cfg)

    enc = cfg.get("enc_blocks", [1, 1, 1, 1])
    if isinstance(enc, str):
        enc = [int(t) for t in enc.strip("[]").replace(" ", "").split(",") if t]

    st = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in ck.get("model", ck).items()}

    hr = int(cfg.get("hr_blocks", 0))     # 0 = pre-hr_blocks checkpoint
    kw = dict(channels=channels, width=int(cfg.get("width", 32)), scale=2,
              enc_blk_nums=list(enc), dec_blk_nums=list(enc))
    try:
        model = NAFNetSR(**kw, hr_blocks=hr)
    except TypeError:                     # model.py predates the hr_blocks arg
        model = NAFNetSR(**kw)
    model.load_state_dict(st)
    model = model.to(dev).eval()

    n_par = sum(q.numel() for q in model.parameters())
    print(f"Loaded {Path(ckpt_path).name}: width={cfg.get('width', 32)} "
          f"enc={list(enc)} hr_blocks={hr} | {n_par/1e6:.2f}M params\n")
    return model


def pad16(x):
    """NAFNetSR needs H,W divisible by 16 (4 stride-2 stages)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def run(model, x):
    dev = next(model.parameters()).device
    use_amp = (dev.type == "cuda")
    return run_tiled(model, x, dev, use_amp)


# ------------------------------------------------------------------- the tests

def mirror_tile(x):
    """[1,C,H,W] -> [1,C,2H,2W]; quadrant (0,0) is the original, content is
    continuous across both boundaries, global mean/variance unchanged."""
    top = torch.cat([x, x.flip(-1)], dim=-1)
    return torch.cat([top, top.flip(-2)], dim=-2)


def diff_report(a, b, label, seam_sides):
    """a, b: [1,C,H,W] HR outputs of the same content. seam_sides: which edges
    of this quadrant are artificial boundaries, e.g. ('right', 'bottom')."""
    d = (a - b).abs()
    h, w = d.shape[-2:]

    mask = torch.ones_like(d, dtype=torch.bool)
    if "right" in seam_sides:
        mask[..., :, w - SEAM_PX:] = False
    if "left" in seam_sides:
        mask[..., :, :SEAM_PX] = False
    if "bottom" in seam_sides:
        mask[..., h - SEAM_PX:, :] = False
    if "top" in seam_sides:
        mask[..., :SEAM_PX, :] = False

    interior = d[mask].mean().item()
    seam = d[~mask].mean().item() if (~mask).any() else float("nan")
    return dict(label=label, mean_all=d.mean().item(), mean_interior=interior,
                mean_seam=seam, max=d.max().item())


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
    print(f"Held-out images available: {len(stems)}\n")

    model = build_model(a.checkpoint, a.channels, dev)

    # ---------------------------------------------------------- MIRROR test
    print("=" * 72)
    print("MIRROR TEST — same content and same global statistics, 2x the size.")
    print("A nonzero interior difference here means SIZE ALONE changes the")
    print("model's behaviour.")
    print("=" * 72)

    mir_rows = []
    for s in stems[:N_IMAGES]:
        d = load(dg_m[s]).to(dev)
        g = load(gt_m[s]).to(dev)
        if d.shape[-1] != d.shape[-2]:
            n = min(d.shape[-2:])
            d = d[..., :n, :n]
            g = g[..., : n * 2, : n * 2]

        solo = run(model, d)
        big = run(model, mirror_tile(d))
        quad = big[..., : solo.shape[-2], : solo.shape[-1]]

        r = diff_report(solo, quad, s, seam_sides=("right", "bottom"))
        r["psnr_solo"] = psnr(solo, g)
        r["psnr_big"] = psnr(quad, g)
        mir_rows.append(r)

    print(f"{'image':<16}{'mean|diff|':>12}{'interior':>11}{'seam':>10}"
          f"{'PSNR solo':>11}{'PSNR@256':>11}{'drop':>8}")
    print("-" * 79)
    for r in mir_rows:
        print(f"{r['label'][:15]:<16}{r['mean_all']:>12.5f}"
              f"{r['mean_interior']:>11.5f}{r['mean_seam']:>10.5f}"
              f"{r['psnr_solo']:>11.3f}{r['psnr_big']:>11.3f}"
              f"{r['psnr_solo'] - r['psnr_big']:>+8.3f}")
    mi = float(np.mean([r["mean_interior"] for r in mir_rows]))
    md = float(np.mean([r["psnr_solo"] - r["psnr_big"] for r in mir_rows]))
    print("-" * 79)
    print(f"MIRROR mean interior |diff| = {mi:.5f}   mean PSNR drop = {md:+.3f} dB\n")

    # ---------------------------------------------------------- MOSAIC test
    print("=" * 72)
    print("MOSAIC TEST — four different images in one 256px input, so the")
    print("global statistics genuinely change. This is the realistic case.")
    print("=" * 72)

    quads = stems[:4]
    tiles_d, tiles_g = [], []
    n = min(min(load(dg_m[s]).shape[-2:]) for s in quads)
    for s in quads:
        d = load(dg_m[s]).to(dev)[..., :n, :n]
        g = load(gt_m[s]).to(dev)[..., : n * 2, : n * 2]
        tiles_d.append(d)
        tiles_g.append(g)

    mosaic = torch.cat([torch.cat(tiles_d[:2], dim=-1),
                        torch.cat(tiles_d[2:], dim=-1)], dim=-2)
    big = run(model, mosaic)
    H = n * 2
    got = [big[..., :H, :H], big[..., :H, H:],
           big[..., H:, :H], big[..., H:, H:]]
    sides = [("right", "bottom"), ("left", "bottom"),
             ("right", "top"), ("left", "top")]

    print(f"{'image':<16}{'mean|diff|':>12}{'interior':>11}{'seam':>10}"
          f"{'PSNR solo':>11}{'PSNR@256':>11}{'drop':>8}")
    print("-" * 79)
    mos_rows = []
    for s, d, g, q, sd in zip(quads, tiles_d, tiles_g, got, sides):
        solo = run(model, d)
        r = diff_report(solo, q, s, seam_sides=sd)
        r["psnr_solo"] = psnr(solo, g)
        r["psnr_big"] = psnr(q, g)
        mos_rows.append(r)
        print(f"{r['label'][:15]:<16}{r['mean_all']:>12.5f}"
              f"{r['mean_interior']:>11.5f}{r['mean_seam']:>10.5f}"
              f"{r['psnr_solo']:>11.3f}{r['psnr_big']:>11.3f}"
              f"{r['psnr_solo'] - r['psnr_big']:>+8.3f}")
    oi = float(np.mean([r["mean_interior"] for r in mos_rows]))
    os_ = float(np.mean([r["mean_seam"] for r in mos_rows]))
    od = float(np.mean([r["psnr_solo"] - r["psnr_big"] for r in mos_rows]))
    print("-" * 79)
    print(f"MOSAIC mean interior |diff| = {oi:.5f}   seam |diff| = {os_:.5f}"
          f"   mean PSNR drop = {od:+.3f} dB\n")

    # ------------------------------------------------------------- verdict
    print("=" * 72)
    if mi < 0.002 and oi < 0.002 and abs(md) < 0.3 and abs(od) < 0.3:
        print("  VERDICT: size-independent. Your 128px numbers should transfer")
        print("  to 512->256 inputs. No change needed. Move on to the deck.")
    elif oi > 0.002 and oi > 0.5 * os_:
        print("  VERDICT: the difference is in the quadrant INTERIORS, not just")
        print("  at the seams. Global statistics are leaking. Your 512->256")
        print("  behaviour is NOT what you measured.")
        print("  FIX: tile inference in evaluation.py — split anything larger")
        print("  than 128px into 128x128 tiles with 16px overlap, run each,")
        print("  blend the overlaps. Then re-run this script to confirm.")
    elif os_ > 0.002:
        print("  VERDICT: difference is concentrated at the artificial seams,")
        print("  interiors are clean. That is an artefact of the mosaic, not a")
        print("  model defect. You are fine. Move on to the deck.")
    else:
        print("  VERDICT: mixed / borderline. Look at the per-image rows above")
        print("  and at the PSNR drop column — that is the number that matters.")
        print("  A drop under 0.3 dB is not worth spending a day on with 8 left.")
    print("=" * 72)
    print()
    print("NOTE: this tests input SIZE sensitivity only. It does not test")
    print("KLA's genuinely different image sources. Do not claim it does.")


if __name__ == "__main__":
    main()
