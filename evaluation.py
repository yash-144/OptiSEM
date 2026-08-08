"""
KLA Hackathon Evaluation Script (Standalone)

    python evaluation.py /path/to/test_images /path/to/output

Runs from any working directory. Handles .npy and image inputs, arbitrary
input sizes, and mixed resolutions (128x128 and 256x256) in one directory.

Fixes over the previous version:
  1. torch.load(weights_only=False)  -- PyTorch >=2.6 defaults to True and
     will refuse to unpickle a checkpoint containing lists/Namespaces.
  2. Checkpoint path resolves relative to THIS FILE, not the cwd.
  3. Reads architecture from either the "args" key (train.py) or the
     "cfg" key (sweep.py). The two scripts disagreed.
  4. Removed torch.compile: it recompiles per input shape, and the test
     set has more than one shape. Inference time is scored.
  5. Pads inputs to a multiple of 16 (the model has 4 stride-2 stages,
     so it needs /16, not /8) and crops the output back.
  6. Batches same-shape images instead of one-at-a-time host syncs.
  7. autocast only when CUDA is actually present.
  8. Writes output in the SAME format as the input.
"""

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import NAFNetSR

HERE = Path(__file__).resolve().parent
PAD_MULTIPLE = 16          # 4 encoder stages, each stride 2
TILE, OVERLAP = 128, 16
IMG_EXTS = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_image(path, channels):
    """-> (float32 array [C,H,W], metadata for writing it back out)"""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 3:
            # accept HWC or CHW
            arr = arr.transpose(2, 0, 1) if arr.shape[-1] in (1, 3) else arr
        return arr, {"kind": "npy"}

    from PIL import Image
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32) / 65535.0
        if arr.ndim == 2:
            arr = arr[None]
        return arr, {"kind": "img", "suffix": path.suffix,
                     "dtype": np.uint16, "scale": 65535.0}
    img = img.convert("L" if channels == 1 else "RGB")
    arr = np.asarray(img, dtype=np.float32)
    scale = 255.0
    arr = arr / scale
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return arr, {"kind": "img", "suffix": path.suffix, "dtype": np.uint8, "scale": scale}


def save_image(arr, out_path, meta):
    """arr: float32 [C,H,W] already clamped to [0,1]."""
    if meta["kind"] == "npy":
        out = arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
        np.save(out_path.with_suffix(".npy"), out.astype(np.float32))
        return

    from PIL import Image
    a = arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
    a = np.clip(a * meta["scale"] + 0.5, 0, meta["scale"]).astype(meta["dtype"])
    Image.fromarray(a).save(out_path.with_suffix(meta["suffix"]))


def build_model(ckpt_path, channels, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # train.py stores "args"; sweep.py stores "cfg". Accept either.
    cfg = ckpt.get("args") or ckpt.get("cfg") or {}
    if not isinstance(cfg, dict):
        cfg = vars(cfg)                      # argparse.Namespace
    width = int(cfg.get("width", 32))
    enc = cfg.get("enc_blocks", [1, 1, 1, 1])
    if isinstance(enc, str):                 # "2,2,4,8" or "[2, 2, 4, 8]"
        enc = [int(t) for t in enc.strip("[]").replace(" ", "").split(",") if t]
    dec = cfg.get("dec_blocks", enc)
    if isinstance(dec, str):
        dec = [int(t) for t in dec.strip("[]").replace(" ", "").split(",") if t]
    hr_blocks = int(cfg.get("hr_blocks", 0))     # 0 = old checkpoints

    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}

    model = NAFNetSR(channels=channels, width=width, scale=2,
                     enc_blk_nums=list(enc), dec_blk_nums=list(dec),
                     hr_blocks=hr_blocks)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"Loaded {ckpt_path.name}: width={width} enc={list(enc)}")
    return model


@torch.no_grad()
def run_batch(model, batch, device, use_amp):
    """batch: float32 tensor [N,C,H,W]. Returns [N,C,2H,2W] clamped."""
    _, _, h, w = batch.shape
    ph = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pw = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    if ph or pw:
        batch = F.pad(batch, (0, pw, 0, ph), mode="reflect")

    batch = batch.to(device, non_blocking=True)
    if device == "cuda":
        batch = batch.to(memory_format=torch.channels_last)

    if use_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
        out = out.float()
    else:
        out = model(batch)

    return out[:, :, : h * 2, : w * 2].clamp_(0, 1)


@torch.no_grad()
def run_tiled(model, batch, device, use_amp):
    n, c, h, w = batch.shape
    if h <= TILE and w <= TILE:
        return run_batch(model, batch, device, use_amp)
    step = TILE - OVERLAP
    out  = torch.zeros(n, c, h * 2, w * 2, device=device)
    wsum = torch.zeros(1, 1, h * 2, w * 2, device=device)
    ys = sorted({*range(0, max(h - TILE, 0) + 1, step), max(h - TILE, 0)})
    xs = sorted({*range(0, max(w - TILE, 0) + 1, step), max(w - TILE, 0)})
    r = OVERLAP * 2
    for y in ys:
        for x in xs:
            win_y = torch.ones(TILE * 2, device=device)
            if y > 0:
                win_y[:r] = torch.linspace(0, 1, r, device=device)
            if y < ys[-1]:
                win_y[-r:] = torch.linspace(1, 0, r, device=device)
                
            win_x = torch.ones(TILE * 2, device=device)
            if x > 0:
                win_x[:r] = torch.linspace(0, 1, r, device=device)
            if x < xs[-1]:
                win_x[-r:] = torch.linspace(1, 0, r, device=device)
                
            win = (win_y[:, None] * win_x[None, :])[None, None]
            
            t = run_batch(model, batch[:, :, y:y+TILE, x:x+TILE], device, use_amp)
            out [:, :, y*2:(y+TILE)*2, x*2:(x+TILE)*2] += t * win
            wsum[:, :, y*2:(y+TILE)*2, x*2:(x+TILE)*2] += win
    return (out / wsum.clamp_min(1e-8)).clamp_(0, 1)


def main():
    p = argparse.ArgumentParser(description="KLA Hackathon Evaluation Script")
    p.add_argument("--input_directory", dest="input_dir", required=True)
    p.add_argument("--output_directory", dest="output_dir", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Defaults to <script_dir>/checkpoints/best.pth")
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_path = Path(args.checkpoint) if args.checkpoint else HERE / "checkpoints" / "best.pth"
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(ckpt_path, args.channels, device)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in in_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if not files:
        print(f"No images found in {in_dir}")
        return
    print(f"Found {len(files)} images on {device}. Running inference...")

    # Group by shape so a batch is always uniform (test set mixes 128 and 256).
    groups = {}
    for f in files:
        arr, meta = load_image(f, args.channels)
        groups.setdefault(arr.shape, []).append((f, arr, meta))

    # Optimal batch sizes determined via timing sweeps
    BS_BY_PIXELS = {128*128: 8, 256*256: 4, 512*512: 4}

    # warm up cudnn autotuning on each distinct (shape, batch_size) pair outside the timer
    if device == "cuda":
        for shape, items in groups.items():
            bs = BS_BY_PIXELS.get(shape[-1] * shape[-2], args.batch_size)
            bs = min(bs, len(items))
            run_tiled(model, torch.zeros(bs, *shape), device, use_amp)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    t_compute = 0.0
    
    for shape, items in groups.items():
        bs = BS_BY_PIXELS.get(shape[-1] * shape[-2], args.batch_size)
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            batch = torch.from_numpy(np.stack([c[1] for c in chunk]))
            
            c0 = time.perf_counter()
            out = run_tiled(model, batch, device, use_amp)
            if device == "cuda":
                torch.cuda.synchronize()
            t_compute += (time.perf_counter() - c0)
            
            out = out.cpu().numpy()
            for (path, _, meta), pred in zip(chunk, out):
                rel_path = path.relative_to(in_dir)
                out_path = out_dir / rel_path.parent / path.stem
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(pred, out_path, meta)
                n += 1
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(f"Done. {n} images -> {out_dir}")
    print(f"Total: {dt:.3f}s | {dt / max(n, 1) * 1000:.2f} ms/image (I/O + Compute)")
    print(f"Compute: {t_compute:.3f}s | {t_compute / max(n, 1) * 1000:.2f} ms/image")


if __name__ == "__main__":
    main()
