"""
L1 + SSIM combined loss.

L1 alone gives blurry-but-safe results; SSIM alone can ignore absolute
brightness. Combining both is a standard, cheap way to push both pixel
fidelity (pSNR) and structural similarity (SSIM) -- two of the three
metrics you're actually graded on -- without needing a perceptual
network (LPIPS) in the loop, which would slow down training.
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim as ssim_fn
import lpips
import math

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))

class RestorationLoss(nn.Module):
    def __init__(self, ssim_weight=0.2, lpips_weight=0.1, device="cuda"):
        super().__init__()
        self.l1 = CharbonnierLoss()
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight

        # Only load the LPIPS network if we're actually using it
        if lpips_weight > 0:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        else:
            self.lpips_fn = None

    def forward(self, pred, target):
        # 1. L1 Loss
        l1_loss = self.l1(pred, target)

        # 2. SSIM Loss (clamp pred to [0,1] so data_range=1.0 is valid)
        pred_st = pred + (pred.clamp(0, 1) - pred).detach()
        ssim_val = ssim_fn(pred_st, target.clamp(0, 1), data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val

        # 3. LPIPS Loss (skipped if weight is 0)
        if self.lpips_fn is not None and self.lpips_weight > 0:
            pred_scaled   = (pred * 2) - 1
            target_scaled = (target * 2) - 1
            if pred_scaled.shape[1] == 1:
                pred_scaled   = pred_scaled.repeat(1, 3, 1, 1)
                target_scaled = target_scaled.repeat(1, 3, 1, 1)
            lpips_loss = self.lpips_fn(pred_scaled, target_scaled).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=pred.device)

        total = l1_loss + (self.ssim_weight * ssim_loss) + (self.lpips_weight * lpips_loss)
        return total, l1_loss.detach(), ssim_val.detach(), lpips_loss.detach()
