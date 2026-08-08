# SEM Image Restoration — Mini-NAFNet

**Team Node Zero** | KLA Hackathon 2026 | SRM University, Haryana Delhi-NCR

A deep learning pipeline for joint denoising and 2× super-resolution of degraded Scanning Electron Microscope (SEM) images. Built on the **NAFNet (Nonlinear Activation Free Network)** architecture.

---

## Problem Statement

Modern semiconductor manufacturing uses SEM to inspect microscopic circuit structures. Increasing inspection speed reduces electron exposure, producing images that are:

- **Noisy** — speckle and Gaussian noise corrupt pixel values (intensity can exceed `[0, 1]` range)
- **Low resolution** — downsampled from 256×256 → 128×128 or 512×512 → 256×256

These degradations obscure nanoscale defects, reducing inspection accuracy and manufacturing yield.

**Our objective:** Restore degraded SEM images to ground-truth quality in a single forward pass — removing noise while simultaneously doubling spatial resolution.

---

## Project Structure

```
sem-model/
├── model.py          # NAFNetSR architecture (SimpleGate, SCA, NAFBlock, UNet, PixelShuffle)
├── dataset.py        # Paired data loading with augmentation (.npy and image formats)
├── losses.py         # Combined loss: L1 + SSIM + LPIPS (perceptual)
├── train.py          # Main training script (configurable architecture + loss)
├── sweep.py          # Hyperparameter sweep (25 configs × 100 epochs, resumable)
├── infer.py          # Visual inference — generates .png files for inspection
├── compare.py        # Side-by-side comparison (NoisyLR vs GT vs Restored)
├── evaluation.py     # Official KLA submission script (H100-optimized)
├── README.md         # This file
└── dataset/
    └── train/
        ├── GT/       # Ground truth images (256×256, float32 .npy, range [0, 1])
        └── NoisyLR/  # Degraded images (128×128, float32 .npy, range [-0.08, 1.45])
```

---

## Architecture — How It Works

### NAFNetSR (model.py)

The model performs **joint denoising + 2× super-resolution** in a single forward pass.

```
Input (128×128 NoisyLR)
    │
    ├──── Bicubic Upsample 2× ──────────────────────────┐
    │                                                     │  (global skip)
    ▼                                                     │
┌─────────────────────────────┐                          │
│  NAFNet Feature Extractor   │                          │
│  (UNet with NAFBlocks)      │                          │
│                             │                          │
│  Encoder: [NAFBlocks] × 4   │                          │
│      ↓ downsample (strided conv)                       │
│  Middle: [NAFBlock] × 1     │                          │
│      ↑ upsample (PixelShuffle)                         │
│  Decoder: [NAFBlocks] × 4   │                          │
│  + skip connections          │                          │
└──────────┬──────────────────┘                          │
           │                                              │
           ▼                                              │
    PixelShuffle 2× ──── Learned Residual ───── (+) ◄───┘
                                                  │
                                                  ▼
                                        Output (256×256 Restored)
```

### NAFBlock — The Core Building Block

Each NAFBlock replaces traditional activation functions (ReLU, GELU) with two innovations from the NAFNet paper (ECCV 2022):

1. **SimpleGate**: Splits channels in half, multiplies them (`x₁ * x₂`). This is a parameter-free gating mechanism that provides non-linearity without activation functions.

2. **Simplified Channel Attention (SCA)**: Global average pooling → 1×1 conv → element-wise multiply. Lightweight channel recalibration.

```
NAFBlock(x):
    ┌── LayerNorm₁ → 1×1 Conv → 3×3 DepthwiseConv → SimpleGate → SCA → 1×1 Conv → ×β ──┐
    x ──────────────────────────────────────────────────────────────────────────── (+) ── x'
    ┌── LayerNorm₂ → 1×1 Conv → SimpleGate → 1×1 Conv → ×γ ──────────────────────────────┐
    x' ─────────────────────────────────────────────────────────────────────────── (+) ── out
```

- `β` and `γ` are learnable scalars (initialized to 0 for stable training start)
- Two **independent** LayerNorm instances (`norm1`, `norm2`) for each branch

### Why This Architecture?

| Design choice | Reason |
|---|---|
| Global bicubic skip | Model only needs to learn the residual (what bicubic gets wrong). Loss starts low, converges faster. |
| SimpleGate over ReLU | Avoids dead neurons and information loss. Shown to match or beat GELU/ReLU on restoration benchmarks. |
| PixelShuffle for upsampling | Produces sharper edges than transposed convolution. No checkerboard artifacts. |
| UNet encoder-decoder | Multi-scale features capture both fine textures (shallow) and global structure (deep). |

### Configurable Parameters

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `width` | `--width` | 32 | Feature channels at the first encoder level (doubles at each level) |
| `enc_blocks` | `--enc_blocks` | `1,1,1,1` | Number of NAFBlocks at each of the 4 encoder levels |

Example: `--width 48 --enc_blocks 2,2,4,8` creates a deeper, wider model with more capacity.

---

## Data Pipeline (dataset.py)

### Loading

- Supports `.npy` (NumPy arrays) and standard image formats (`.png`, `.jpg`, etc.)
- Files are matched by filename stem — `000042.npy` in `GT/` pairs with `000042.npy` in `NoisyLR/`
- Degraded images are **not** clipped to `[0, 1]` since speckle noise naturally exceeds this range

### Training Augmentation

1. **Padding** — if image is smaller than patch size, reflect-pad to minimum size
2. **Random crop** — extract `patch_size × patch_size` from degraded, corresponding `2×patch_size × 2×patch_size` from GT (aligned)
3. **Random horizontal flip** (50% probability)
4. **Random vertical flip** (50% probability)

### Validation

- No cropping or flipping — full-resolution images used as-is
- Train/val split uses **separate dataset instances** with deterministic index-based subsetting (no data leakage)

### Data Characteristics (Verified)

| | GT | NoisyLR |
|---|---|---|
| Resolution | 256×256 | 128×128 |
| Data type | float32 | float32 |
| Value range | `[0.0, 1.0]` | `[-0.08, 1.45]` |
| Format | `.npy` | `.npy` |
| Total pairs | 3200 | 3200 |

---

## Loss Function (losses.py)

The loss directly optimizes for all three KLA evaluation metrics:

```
Total Loss = L1 + (ssim_weight × SSIM_loss) + (lpips_weight × LPIPS_loss)
```

| Component | What it optimizes | Weight | Details |
|---|---|---|---|
| **L1 Loss** | Pixel-level fidelity → **PSNR** | 1.0 (fixed) | Mean absolute error between pred and GT |
| **SSIM Loss** | Structural similarity → **SSIM** | `--ssim_weight` (default: 0.2) | `1 - SSIM(pred, GT)`. Pred clamped to `[0,1]` before computation for mathematical correctness |
| **LPIPS Loss** | Perceptual quality → **LPIPS** | `--lpips_weight` (default: 0.1) | AlexNet backbone. Grayscale images are replicated to 3 channels. Set to 0.0 to disable (saves memory and time) |

---

## Complete Training Workflow

### Phase 1: Hyperparameter Sweep (sweep.py)

Explores 25 configurations across 7 dimensions, each trained for 100 epochs:

| Dimension | Values explored |
|---|---|
| Learning rate | `2e-4`, `5e-4`, `1e-3`, `2e-3`, `3e-3` |
| Batch size | `4`, `8`, `16`, `32` |
| Patch size | `64`, `96`, `128`, `160` |
| Model width | `16`, `32`, `48`, `64` |
| Encoder depth | `[1,1,1,1]`, `[2,2,2,2]`, `[2,2,4,4]`, `[2,2,4,8]` |
| SSIM weight | `0.0`, `0.1`, `0.2`, `0.25`, `0.3`, `0.5` |
| Combinations | 4 hand-crafted mixed configs |

**Key features:**
- **Resumable** — progress saved to `sweep_log.json` after every config. Safe across Colab disconnects.
- **Early stopping** — skips configs that plateau for 30 epochs
- **LPIPS disabled** during sweep (L1 + SSIM only) for ~2× speed
- **Per-config checkpoints** saved to `checkpoints/sweep/cfg_<id>/best.pth`
- **Final leaderboard** printed and saved to `sweep_results.csv`

```bash
python sweep.py \
  --gt_dir dataset/train/GT \
  --degraded_dir dataset/train/NoisyLR \
  --channels 1 \
  --probe_epochs 100 \
  --amp
```

**Estimated time:** ~20-25 hours on T4 GPU (across multiple Colab sessions)

### Phase 2: Full Training (train.py)

Train the winning config for 200+ epochs with LPIPS enabled:

```bash
python train.py \
  --gt_dir dataset/train/GT \
  --degraded_dir dataset/train/NoisyLR \
  --channels 1 \
  --epochs 200 \
  --lr <best_lr> \
  --batch_size <best_bs> \
  --patch_size <best_ps> \
  --width <best_width> \
  --enc_blocks <best_enc> \
  --ssim_weight <best_ssim> \
  --lpips_weight 0.1 \
  --amp
```

**Features:**
- AdamW optimizer with CosineAnnealingLR scheduler
- Automatic Mixed Precision (AMP) via `--amp`
- Checkpoints saved every epoch (`last.pth`) and on best val PSNR (`best.pth`)
- Architecture config saved inside checkpoint (evaluation.py reads it automatically)
- Resume interrupted runs with `--resume checkpoints/last.pth`

**Estimated time:** ~1.5-6 hours on T4 GPU (depends on winning config size)

### Phase 3: Visual Inspection (infer.py + compare.py)

Generate restored images and compare side-by-side:

```bash
# Pick a few samples for quick visual check
mkdir -p dataset/quick_test
cp dataset/train/NoisyLR/000000.npy dataset/quick_test/
cp dataset/train/NoisyLR/000001.npy dataset/quick_test/
cp dataset/train/NoisyLR/000002.npy dataset/quick_test/

# Run inference
python infer.py \
  --input_dir dataset/quick_test \
  --output_dir dataset/quick_results \
  --checkpoint checkpoints/best.pth \
  --channels 1

# Generate comparison images
python compare.py \
  --samples 000000 000001 000002 \
  --gt_dir dataset/train/GT \
  --noisylr_dir dataset/train/NoisyLR \
  --restored_dir dataset/quick_results \
  --out_dir dataset/comparison
```

Output: `dataset/comparison/` folder with `_1_noisylr.png`, `_2_gt.png`, `_3_restored.png` per sample.

### Phase 4: Official Evaluation (evaluation.py)

Standalone script matching KLA submission requirements:

```bash
python evaluation.py /path/to/test_images /path/to/output
```

**Optimizations for H100 benchmarking:**
- `torch.backends.cudnn.benchmark = True`
- `torch.compile()` for graph-level optimization (PyTorch 2.0+)
- Automatic Mixed Precision during inference
- Architecture config loaded from checkpoint (no hardcoded values)
- Output clamped to `[0, 1]` for valid metric computation

---

## Setup & Requirements

**Python 3.10+** with the following libraries:

```bash
pip install torch torchvision numpy Pillow pytorch_msssim lpips
```

### Running on Google Colab

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Cell 2: Setup
!cp /content/drive/MyDrive/sem-model.zip /content/
!unzip -q /content/sem-model.zip -d /content/
%cd /content/sem-model
!pip install -q pytorch_msssim lpips
```

```bash
# Cell 3: Run sweep (resumable across sessions)
!python sweep.py \
  --gt_dir dataset/train/GT \
  --degraded_dir dataset/train/NoisyLR \
  --channels 1 \
  --probe_epochs 100 \
  --amp
```

---

## References

1. **NAFNet: Simple Baselines for Image Restoration** — [github.com/megvii-research/nafnet](https://github.com/megvii-research/nafnet)
2. **Deep Learning Based SEM Image Denoising** — [TechRxiv](https://www.techrxiv.org/doi/10.36227/techrxiv.22296655)
3. **Automated Semiconductor Defect Inspection in SEM Images** — [arXiv:2308.08376](https://arxiv.org/abs/2308.08376)
