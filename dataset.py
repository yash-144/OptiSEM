"""
PairedRestorationDataset

Pairs degraded and ground-truth images by matching filename stem
(e.g. "sample_014.png" in both folders), searched recursively so it
doesn't matter if your dataset organizes images into subfolders by
category/source.

If your dataset instead uses a manifest / CSV to map degraded->GT
filenames, swap out `_list_images` + the matching logic below for a
`pandas.read_csv` -- everything downstream (crops, augmentation,
tensor format) stays the same.
"""

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_images(root):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.npy")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return files


class PairedRestorationDataset(Dataset):
    def __init__(self, gt_dir, degraded_dir, channels=1, patch_size=96, train=True):
        self.channels = channels
        self.patch_size = patch_size
        self.train = train

        gt_files = _list_images(gt_dir)
        deg_files = _list_images(degraded_dir)
        if not gt_files:
            raise RuntimeError(f"No images found under {gt_dir}")
        if not deg_files:
            raise RuntimeError(f"No images found under {degraded_dir}")

        gt_by_stem = {Path(f).stem: f for f in gt_files}
        deg_by_stem = {Path(f).stem: f for f in deg_files}

        common = sorted(set(gt_by_stem) & set(deg_by_stem))
        if not common:
            raise RuntimeError(
                f"No matching filenames between {gt_dir} and {degraded_dir}. "
                "GT and degraded images must share the same base filename "
                "(e.g. img001.png in both folders)."
            )
        # deterministic order -> train/val split stays consistent across runs
        self.pairs = [(deg_by_stem[k], gt_by_stem[k]) for k in common]

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        if path.endswith('.npy'):
            arr = np.load(path).astype(np.float32)
        else:
            from PIL import Image
            mode = "L" if self.channels == 1 else "RGB"
            img = Image.open(path).convert(mode)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W

    def __getitem__(self, idx):
        deg_path, gt_path = self.pairs[idx]
        deg = self._load(deg_path)
        gt = self._load(gt_path)

        _, dh, dw = deg.shape
        _, gh, gw = gt.shape
        if (gh, gw) != (dh * 2, dw * 2):
            # keeps training robust if a few pairs in the dataset don't
            # follow the exact 2x rule
            gt = torch.nn.functional.interpolate(
                gt.unsqueeze(0), size=(dh * 2, dw * 2), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)

        if self.train:
            ps = self.patch_size
            if dh < ps or dw < ps:
                pad_h, pad_w = max(0, ps - dh), max(0, ps - dw)
                deg = torch.nn.functional.pad(deg, (0, pad_w, 0, pad_h), mode="reflect")
                gt = torch.nn.functional.pad(gt, (0, pad_w * 2, 0, pad_h * 2), mode="reflect")
                dh, dw = deg.shape[1], deg.shape[2]

            top = random.randint(0, dh - ps)
            left = random.randint(0, dw - ps)
            deg = deg[:, top:top + ps, left:left + ps]
            gt = gt[:, top * 2:(top + ps) * 2, left * 2:(left + ps) * 2]

            if random.random() < 0.5:
                deg, gt = deg.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                deg, gt = deg.flip(-2), gt.flip(-2)

        return deg, gt
