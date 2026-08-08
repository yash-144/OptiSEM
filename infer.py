"""
Inference script — generates visual .png files for quick inspection.

Usage:
    python infer.py \
        --input_directory dataset/train/NoisyLR \
        --output_directory dataset/quick_results \
        --checkpoint checkpoints/best.pth \
        --channels 1
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import NAFNetSR

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_directory", required=True, help="Path to degraded test images")
    p.add_argument("--output_directory", required=True, help="Path to save restored images")
    p.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to model weights")
    p.add_argument("--channels", type=int, default=1, help="1=grayscale, 3=RGB")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint and extract architecture config
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Did you train the model?")
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", ckpt.get("cfg", {}))
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    width = int(saved_args.get("width", 32))
    enc_blocks = saved_args.get("enc_blocks", [1, 1, 1, 1])
    if isinstance(enc_blocks, str):
        enc_blocks = [int(t) for t in enc_blocks.strip("[]").replace(" ", "").split(",") if t]
    dec_blocks = saved_args.get("dec_blocks", enc_blocks)
    hr_blocks = int(saved_args.get("hr_blocks", 0))

    model = NAFNetSR(
        channels=args.channels, width=width, scale=2,
        enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks,
        hr_blocks=hr_blocks,
    ).to(device)
    model.load_state_dict(ckpt.get("model", ckpt.get("state_dict", ckpt)))
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')} "
          f"(width={width}, enc={enc_blocks})")

    out_dir = Path(args.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.npy")
    test_files = []
    for ext in exts:
        test_files.extend(glob.glob(os.path.join(args.input_directory, "**", ext), recursive=True))

    if not test_files:
        print(f"No images found in {args.input_directory}!")
        return

    print(f"Found {len(test_files)} images. Starting inference...")

    with torch.no_grad():
        for i, filepath in enumerate(test_files):
            stem = Path(filepath).stem
            is_npy = filepath.endswith(".npy")
            
            if is_npy:
                arr = np.load(filepath).astype(np.float32)
            else:
                mode = "L" if args.channels == 1 else "RGB"
                img = Image.open(filepath).convert(mode)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                
            if arr.ndim == 2:
                arr = arr[:, :, None]
            
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Pad to multiple of 16
            h, w = x.shape[2:]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
            pred = model(x)
            
            # Unpad
            if pad_h > 0 or pad_w > 0:
                pred = pred[:, :, :h, :w]
            
            # Clamp to valid range
            pred = pred.clamp(0, 1)
            
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if args.channels == 1:
                pred_np = pred_np.squeeze(-1)
                
            # Save as .npy (for scoring)
            np.save(out_dir / f"{stem}.npy", pred_np)
            
            # Save as .png (for visual inspection)
            img_out = Image.fromarray((pred_np * 255.0).astype(np.uint8))
            img_out.save(out_dir / f"{stem}.png")
            
            print(f"[{i+1}/{len(test_files)}] Restored {stem}")
            
    print("Inference complete!")

if __name__ == "__main__":
    main()
