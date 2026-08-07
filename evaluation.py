"""
KLA Hackathon Evaluation Script (Standalone)

Accepts an input directory of test images and an output directory.
Loads the trained NAFNetSR model and runs inference, saving restored .npy files.
Optimized for NVIDIA H100 inference benchmarking.

Usage:
    python evaluation.py /path/to/test_images /path/to/output
    python evaluation.py /path/to/test_images /path/to/output --checkpoint checkpoints/best.pth
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
import torch
from model import NAFNetSR

def main():
    p = argparse.ArgumentParser(description="KLA Hackathon Evaluation Script")
    p.add_argument("input_dir", help="Path to test images directory")
    p.add_argument("output_dir", help="Path to output directory")
    p.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to model weights")
    p.add_argument("--channels", type=int, default=1, help="1=grayscale, 3=RGB")
    args = p.parse_args()

    # Inference Optimizations for H100
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint and extract architecture config
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Architecture params are saved inside checkpoint by train.py
    saved_args = ckpt.get("args", {})
    width = saved_args.get("width", 32)
    enc_blocks = saved_args.get("enc_blocks", [1, 1, 1, 1])
    dec_blocks = saved_args.get("dec_blocks", enc_blocks)

    model = NAFNetSR(
        channels=args.channels, width=width, scale=2,
        enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Try to compile model for faster inference if PyTorch 2.0+ is available
    try:
        model = torch.compile(model)
    except Exception:
        pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    exts = ("*.npy", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    test_files = []
    for ext in exts:
        test_files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))

    if not test_files:
        print(f"No test images found in {args.input_dir}")
        return

    print(f"Found {len(test_files)} images. Running inference...")

    with torch.no_grad():
        for filepath in test_files:
            stem = Path(filepath).stem
            is_npy = filepath.endswith(".npy")
            
            if is_npy:
                arr = np.load(filepath).astype(np.float32)
            else:
                from PIL import Image
                mode = "L" if args.channels == 1 else "RGB"
                img = Image.open(filepath).convert(mode)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                
            if arr.ndim == 2:
                arr = arr[:, :, None]
            
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(x)
            
            # Clamp output to valid [0, 1] range before saving
            pred = pred.clamp(0, 1)
            
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if args.channels == 1:
                pred_np = pred_np.squeeze(-1)
                
            np.save(out_dir / f"{stem}.npy", pred_np)

    print(f"Done! {len(test_files)} images restored to {args.output_dir}")

if __name__ == "__main__":
    main()
