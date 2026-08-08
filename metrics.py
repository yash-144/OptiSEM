"""
metrics.py — PSNR / SSIM / LPIPS on a deterministic held-out split,
with a bicubic baseline row. Produces the numbers for Slide 6.

python metrics.py --gt_dir dataset/train/GT --degraded_dir dataset/train/NoisyLR \
                  --checkpoint checkpoints/best.pth
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn
import lpips
from model import NAFNetSR

VAL_STRIDE = 20          # every 20th image = 5% holdout, deterministic


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
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--channels", type=int, default=1)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = {Path(f).stem: f for f in _list(a.gt_dir)}
    dg = {Path(f).stem: f for f in _list(a.degraded_dir)}
    stems = sorted(set(gt) & set(dg))[::VAL_STRIDE]
    print(f"Held-out split: {len(stems)} images\n")

    lp = lpips.LPIPS(net="alex").to(dev)

    model = None
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck.get("args") or ck.get("cfg") or {}
        if not isinstance(cfg, dict):
            cfg = vars(cfg)
        enc = cfg.get("enc_blocks", [1, 1, 1, 1])
        if isinstance(enc, str):
            enc = [int(t) for t in enc.strip("[]").replace(" ", "").split(",") if t]
        st = ck.get("model", ck.get("state_dict", ck))
        st = {k.replace("_orig_mod.", "").replace("module.", ""): v
              for k, v in st.items()}
        model = NAFNetSR(channels=a.channels, width=int(cfg.get("width", 32)),
                         scale=2, enc_blk_nums=list(enc), dec_blk_nums=list(enc))
        model.load_state_dict(st); model = model.to(dev).eval()

    rows = {"bicubic": [], "model": []}
    with torch.no_grad():
        for s in stems:
            g = load(gt[s]).to(dev)
            d = load(dg[s]).to(dev)
            preds = {"bicubic": F.interpolate(d, scale_factor=2, mode="bicubic",
                                              align_corners=False).clamp(0, 1)}
            if model is not None:
                preds["model"] = model(d).clamp(0, 1)
            for k, pr in preds.items():
                l3 = lp(pr.repeat(1, 3, 1, 1) * 2 - 1,
                        g.repeat(1, 3, 1, 1) * 2 - 1).item()
                rows[k].append((psnr(pr, g),
                                ssim_fn(pr, g.clamp(0, 1), data_range=1.0).item(),
                                l3))

    print(f"{'method':<10}{'PSNR (dB)':>12}{'SSIM':>10}{'LPIPS':>10}")
    print("-" * 42)
    for k, v in rows.items():
        if not v:
            continue
        m = np.mean(v, axis=0)
        print(f"{k:<10}{m[0]:>12.3f}{m[1]:>10.4f}{m[2]:>10.4f}")


if __name__ == "__main__":
    main()
