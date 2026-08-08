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
import torch.nn.functional as F
from torch.utils.data import Dataset


def _list_images(root):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.npy")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return files


class PairedRestorationDataset(Dataset):
    def __init__(self, gt_dir, degraded_dir, channels=1, patch_size=96, train=True,
                 synth_prob=0.0, noise_a=0.1673, noise_p=0.811):
        self.channels = channels
        self.patch_size = patch_size
        self.train = train
        self.synth_prob = synth_prob
        self.noise_a = noise_a      # sigma(mu) = noise_a * mu**noise_p
        self.noise_p = noise_p      # fitted: Var = 0.0280 * mu^1.622

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

    def _synth_degrade(self, gt_patch):
        """gt_patch: [C, 2*ps, 2*ps] in [0,1]  ->  degraded [C, ps, ps]."""
        x = gt_patch.unsqueeze(0)
        
        # 30% chance to use the empirically fitted power-law noise
        if random.random() < 0.3:
            mode = random.choice(["bicubic", "bilinear", "area"])
            if mode == "area":
                lr = F.interpolate(x, scale_factor=0.5, mode="area")
            else:
                lr = F.interpolate(x, scale_factor=0.5, mode=mode, align_corners=False, antialias=True)
            lr = lr.squeeze(0)
            scale = random.uniform(0.0, 2.5)
            # Use .abs() instead of clamp_min(0.0) to preserve signal range
            sigma = self.noise_a * scale * lr.abs().pow(self.noise_p)
            return lr + torch.randn_like(lr) * sigma

        # 70% chance to use the official 3 degradations in random order
        order = [0, 1, 2]
        random.shuffle(order)
        
        curr = x
        for op in order:
            if op == 0:
                # Additive Gaussian
                sigma = random.uniform(0.01, 0.15)
                curr = curr + torch.randn_like(curr) * sigma
            elif op == 1:
                # Speckle (Multiplicative)
                sigma = random.uniform(0.05, 0.3)
                curr = curr + curr * torch.randn_like(curr) * sigma
            elif op == 2:
                # Downsample
                mode = random.choice(["bicubic", "bilinear", "area"])
                if mode == "area":
                    curr = F.interpolate(curr, scale_factor=0.5, mode="area")
                else:
                    curr = F.interpolate(curr, scale_factor=0.5, mode=mode, align_corners=False, antialias=True)
                    
        return curr.squeeze(0)

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
            if not getattr(PairedRestorationDataset, "_warned_scale", False):
                print(f"[dataset] WARNING: non-2x pair {Path(gt_path).name}: "
                      f"GT {gh}x{gw} vs deg {dh}x{dw}. GT is being resized — "
                      f"check whether your dataset mixes 2x and 4x pairs.")
                PairedRestorationDataset._warned_scale = True
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

            if self.synth_prob > 0 and random.random() < self.synth_prob:
                deg = self._synth_degrade(gt)

            if random.random() < 0.5:
                deg, gt = deg.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                deg, gt = deg.flip(-2), gt.flip(-2)
            if random.random() < 0.5:
                deg, gt = deg.transpose(-2, -1), gt.transpose(-2, -1)

        return deg, gt
