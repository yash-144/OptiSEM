"""
Compare script: puts NoisyLR, GT, and Restored images side-by-side as PNGs.

Usage (on Colab, after training + inference):
    python compare.py \
        --samples 000000 000001 000002 \
        --gt_dir dataset/train/GT \
        --noisylr_dir dataset/train/NoisyLR \
        --restored_dir dataset/quick_results \
        --out_dir dataset/comparison
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image


def npy_to_png(npy_path, out_path):
    arr = np.load(npy_path).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", nargs="+", default=["000000", "000001", "000002"])
    p.add_argument("--gt_dir", default="dataset/train/GT")
    p.add_argument("--noisylr_dir", default="dataset/train/NoisyLR")
    p.add_argument("--restored_dir", default="dataset/quick_results")
    p.add_argument("--out_dir", default="dataset/comparison")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for stem in args.samples:
        print(f"\nProcessing {stem}...")

        # 1. NoisyLR (input) — upscale to GT size for visual comparison
        nlr_path = os.path.join(args.noisylr_dir, f"{stem}.npy")
        if os.path.exists(nlr_path):
            arr = np.load(nlr_path).astype(np.float32)
            arr = np.clip(arr, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            # Upscale to GT size so all 3 images are the same dimensions
            img_up = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
            img_up.save(out / f"{stem}_1_noisylr.png")
            print(f"  Saved {stem}_1_noisylr.png ({img_up.size})")

        # 2. Ground Truth
        gt_path = os.path.join(args.gt_dir, f"{stem}.npy")
        if os.path.exists(gt_path):
            npy_to_png(gt_path, out / f"{stem}_2_gt.png")
            arr = np.load(gt_path)
            print(f"  Saved {stem}_2_gt.png ({arr.shape})")

        # 3. Restored output
        restored_path = os.path.join(args.restored_dir, f"{stem}.npy")
        if os.path.exists(restored_path):
            npy_to_png(restored_path, out / f"{stem}_3_restored.png")
            arr = np.load(restored_path)
            print(f"  Saved {stem}_3_restored.png ({arr.shape})")

    print(f"\nDone! All comparison images saved to: {args.out_dir}/")
    print("Files are numbered 1/2/3 so they sort as: NoisyLR → GT → Restored")


if __name__ == "__main__":
    main()
