# Node Zero — KLA PS01 Action Plan

**Written 7 Aug 2026. Deadline 16 Aug 2026 → 9 days.**

Confidence tags: `[C]` = certain (verified in your code/data), `[L]` = likely
(strong inference), `[G]` = guessing.

---

## 0. The one-paragraph version

Stop the sweep. It cannot answer the question you're asking it — your
epoch-to-epoch validation noise (±1 dB) is larger than the gaps between your
configs (0.4 dB). `[C]` Fix `evaluation.py`, which currently crashes on a fresh
PyTorch install and would make you **unscorable**. `[C]` Then spend the
remaining days on the thing that actually matches how you're judged:
a **synthetic degradation pipeline** that makes your model robust to the
out-of-distribution test set, plus the submission artifacts (PDF, repo, video)
that Round 1 is actually scored on by human experts.

Your model architecture is fine. Don't touch `model.py`.

---

## 1. Status of every file

| File | Verdict | Action |
|---|---|---|
| `model.py` | Correct. Matches official NAFNet (LayerNorm-via-permute ✓, SCA ✓, decoder skip channel counts ✓). Good speed/quality tradeoff. | **No changes.** |
| `evaluation.py` | **Broken.** 4 bugs, one fatal. | **Replace** with the version I wrote. |
| `losses.py` | 2 real bugs. | Patch (§2.2). |
| `dataset.py` | Missing augmentation; silently corrupts non-2× pairs; no synthetic degradation. | Patch (§2.3). |
| `sweep.py` | Methodologically invalid for its purpose. | Patch (§2.4), then use for **3 short runs only**. |
| `train.py` | **I have not seen this file.** Same patches almost certainly apply. | Send it to me. |
| `README.md` | Claims MS-SSIM; code uses single-scale SSIM. | Fix the sentence. |
| `infer.py`, `compare.py` | Not reviewed. | Low priority. |

---

## 2. Exact fixes

### 2.1 `evaluation.py` — REPLACE ENTIRELY

Use the `evaluation.py` file I gave you. What it fixes:

1. **FATAL** `[C]` — `torch.load(path, map_location=device)`. PyTorch ≥2.6
   defaults `weights_only=True` and refuses to unpickle your checkpoint
   (it contains a list under `enc_blocks`). On KLA's H100, from your
   `requirements.txt`: `UnpicklingError` → unscored → cannot win.
2. `[C]` — `ckpt.get("args", {})` but `sweep.py` saves under `"cfg"`. Silent
   fallback to width=32, then a shape-mismatch crash.
3. `[C]` — `--checkpoint` default is the *relative* path `checkpoints/best.pth`.
   Run from any other directory → `FileNotFoundError`.
4. `[C]` — `torch.compile` is lazy, so your `try/except` catches nothing, and it
   **recompiles per input shape**. The test set has both 128×128 and 256×256.
   Inference time is explicitly benchmarked. Removed.
5. `[C]` — no padding logic. Your model has 4 stride-2 stages, so it needs
   inputs divisible by **16**, not 8. Any OOD image of an odd size crashes.
   Now reflect-padded and cropped back.
6. `[C]` — one image at a time with a `.cpu()` sync each. Now batched by shape.
7. `[C]` — `autocast('cuda')` called even when `device == "cpu"`.
8. `[C]` — always writes `.npy`. Now writes back in the input's format.

**Still open:** you must confirm with KLA whether test inputs are `.npy` or
image files, and what output format their scorer expects. See §6.

### 2.2 `losses.py`

**Fix A — the clamp kills your SSIM gradient.** `[L]` Your inputs reach 1.45 and
the bicubic skip passes that through, so a real fraction of predicted pixels sit
outside [0,1] and get **zero gradient** from the SSIM term.

```python
# FIND:
        pred_clamped = pred.clamp(0, 1)
        target_clamped = target.clamp(0, 1)
        ssim_val = ssim_fn(pred_clamped, target_clamped, data_range=1.0, size_average=True)

# REPLACE WITH:
        # Straight-through clamp: forward value is clamped (so data_range=1.0
        # is valid) but gradient flows to out-of-range pixels.
        pred_st = pred + (pred.clamp(0, 1) - pred).detach()
        ssim_val = ssim_fn(pred_st, target.clamp(0, 1),
                           data_range=1.0, size_average=True)
```

**Fix B — three GPU→CPU syncs every step.** `[C]` `.item()` forces a device
sync. You call it 3× per iteration.

```python
# FIND:
        return total, l1_loss.item(), ssim_val.item(), lpips_loss.item()

# REPLACE WITH:
        return total, l1_loss.detach(), ssim_val.detach(), lpips_loss.detach()
```

Then in `train.py`, wherever you log these, add `.item()` at the logging site
(once every N steps, not every step).

### 2.3 `dataset.py`

**Fix A — complete the D4 augmentation.** `[C]` You have h-flip and v-flip:
4 of the 8 dihedral symmetries. Adding transpose gives all 8, free, and
generalization is the criterion you're graded on.

```python
# FIND:
            if random.random() < 0.5:
                deg, gt = deg.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                deg, gt = deg.flip(-2), gt.flip(-2)

# REPLACE WITH:
            if random.random() < 0.5:
                deg, gt = deg.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                deg, gt = deg.flip(-2), gt.flip(-2)
            if random.random() < 0.5:                      # completes D4
                deg, gt = deg.transpose(-2, -1), gt.transpose(-2, -1)
```

(Safe because your crops are square.)

**Fix B — stop silently corrupting mismatched pairs.** `[C]` Right now a
512-GT/128-LR pair gets GT quietly bicubic-*downscaled* to 256 and you'd never
know.

```python
# FIND:
        if (gh, gw) != (dh * 2, dw * 2):
            # keeps training robust if a few pairs in the dataset don't
            # follow the exact 2x rule
            gt = torch.nn.functional.interpolate(

# REPLACE WITH:
        if (gh, gw) != (dh * 2, dw * 2):
            if not getattr(PairedRestorationDataset, "_warned_scale", False):
                print(f"[dataset] WARNING: non-2x pair {Path(gt_path).name}: "
                      f"GT {gh}x{gw} vs deg {dh}x{dw}. GT is being resized — "
                      f"check whether your dataset mixes 2x and 4x pairs.")
                PairedRestorationDataset._warned_scale = True
            gt = torch.nn.functional.interpolate(
```

**Fix C — on-the-fly synthetic degradation.** This is the headline change.
Add `import torch.nn.functional as F` at the top, then:

```python
# ADD to __init__ signature:
    def __init__(self, gt_dir, degraded_dir, channels=1, patch_size=96, train=True,
                 synth_prob=0.0, speckle_std=0.0, gauss_std=0.0):
        ...
        self.synth_prob  = synth_prob
        self.speckle_std = speckle_std
        self.gauss_std   = gauss_std

# ADD as a method:
    def _synth_degrade(self, gt_patch):
        """gt_patch: [C, 2*ps, 2*ps] in [0,1]  ->  degraded [C, ps, ps]."""
        mode = random.choice(["bicubic", "bilinear", "area"])
        x = gt_patch.unsqueeze(0)
        if mode == "area":
            lr = F.interpolate(x, scale_factor=0.5, mode="area")
        else:
            lr = F.interpolate(x, scale_factor=0.5, mode=mode,
                               align_corners=False, antialias=True)
        lr = lr.squeeze(0)
        sp = self.speckle_std * random.uniform(0.3, 2.5)
        gs = self.gauss_std   * random.uniform(0.3, 2.5)
        lr = lr * (1 + torch.randn_like(lr) * sp) + torch.randn_like(lr) * gs
        return lr          # deliberately NOT clamped — matches real data range

# ADD in __getitem__, immediately AFTER the crop and BEFORE the flips:
            if self.synth_prob > 0 and random.random() < self.synth_prob:
                deg = self._synth_degrade(gt)
```

Get `speckle_std` and `gauss_std` from `analyze_degradation.py` (§3.1).

### 2.4 `sweep.py` / `train.py` — training loop

**Fix A — gradient clipping.** `[C]` You have none. This is why lr=1e-3
produced `loss=0.957` at epoch 5 and why 2e-3/3e-3 NaN'd. `[L]` Your entire
"2e-4 is best" conclusion is an artifact of this missing line.

```python
# FIND:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

# REPLACE WITH:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
```

**Fix B — per-step warmup + cosine.** Add `import math` at the top.

```python
# FIND:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.probe_epochs
    )

# REPLACE WITH:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.9), weight_decay=1e-3)
    total_steps  = args.probe_epochs * len(train_loader)
    warmup_steps = min(500, max(1, total_steps // 20))

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
```

Then move the scheduler step **inside** the batch loop:

```python
# FIND:
            running_loss += loss.item()
        scheduler.step()

# REPLACE WITH:
            scheduler.step()                    # per STEP, not per epoch
            running_loss += loss.item()
```

**Fix C — clamp before computing PSNR.** `[C]` `evaluation.py` clamps; your
sweep didn't. You were ranking on a metric that isn't the scored metric.

```python
# FIND:
def psnr_metric(pred, target):
    mse = torch.mean((pred - target) ** 2).item()

# REPLACE WITH:
def psnr_metric(pred, target):
    pred = pred.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
```

**Fix D — EMA of weights.** `[L]` +0.1–0.2 dB, and more importantly it collapses
your ±1 dB val noise so "best checkpoint" stops being a lottery. Add near the
top of the file:

```python
class EMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def store_and_apply(self, model):
        msd = model.state_dict()
        self.backup = {k: v.detach().clone() for k, v in msd.items()}
        model.load_state_dict({k: self.shadow[k].to(msd[k].dtype) for k in msd})

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup)
            self.backup = None
```

Wire it in:

```python
# after `model = NAFNetSR(...).to(device)`
    ema = EMA(model, decay=0.999)

# after `scaler.update()` in the batch loop
            ema.update(model)

# wrap the validation block
        ema.store_and_apply(model)      # <-- before `model.eval()`
        model.eval()
        ...  (existing validation loop)
        # <-- after computing ep_psnr and AFTER torch.save if it's the best
        ema.restore(model)
```

Save the EMA weights (that's what you submit), not the raw ones.

**Fix E — budget in iterations, not epochs.** `[C]` With 3040 training images
and `drop_last=True`: bs=4 → 760 steps/epoch, bs=8 → 380, bs=16 → 190,
bs=32 → 95. Over 100 epochs that's 76,000 vs 9,500 optimizer steps — an **8×
spread**. Your batch-size sweep measured training length, not batch size. The
same confound corrupts the width and depth sweeps.

For the three calibration runs in §3.2, replace the epoch loop bound so every
run gets the **same number of optimizer steps** (use 12,000):

```python
    TARGET_STEPS = 12000
    args.probe_epochs = max(1, TARGET_STEPS // len(train_loader))
```

### 2.5 `README.md`

`[C]` It says MS-SSIM in three places; `losses.py` imports single-scale
`ssim`. A KLA engineer will read both files. Pick one and make them agree.
(Silver lining: single-scale SSIM has no 160px minimum, so patch=64 would not
actually have crashed. My earlier warning about that was wrong.)

---

## 3. What to do with the two new files

### 3.1 `analyze_degradation.py` — **run this tonight, it takes 2 minutes**

```bash
python analyze_degradation.py \
  --gt_dir dataset/train/GT \
  --degraded_dir dataset/train/NoisyLR \
  --n_samples 200
```

It answers three things:

- **Step 0** — do you actually have clean 2× pairs, or is your dataset mixing
  512→256 and 256→128? (Your `dataset.py` hides this today.)
- **Step 1** — which downsample kernel KLA used (bicubic / bilinear / area).
- **Step 2** — fits `Var(residual) = speckle_var·μ² + gauss_var`, giving you
  the speckle and Gaussian standard deviations, plus an R² telling you whether
  the model actually explains the data.

If **R² > 0.8**: you have the degradation recipe. Feed `speckle_std` and
`gauss_std` into `dataset.py` Fix C and train with `synth_prob=0.75`.

If **R² < 0.8**: the residual has structure the simple model misses (blur
before downsampling, or Poisson shot noise). Send me the output table and
I'll extend the model.

**This plot and these numbers go on Slide 4 or 5.** "We reverse-engineered the
forward degradation operator and trained against randomized samples from it"
is a real contribution. It is also the direct answer to KLA's stated OOD
requirement. `[L]` this is worth more than every remaining sweep config
combined.

### 3.2 `evaluation.py` — drop-in replacement

```bash
cp evaluation.py sem-model/evaluation.py
```

Then **test it the way KLA will**, which is not the way you've been testing it:

```bash
cd /tmp && rm -rf fresh && git clone <your-public-repo> fresh
cd /tmp && python fresh/evaluation.py fresh/dataset/quick_test /tmp/out
```

Note the `cd /tmp` — running from outside the repo is the whole point. If that
command works on a clean Colab runtime with a fresh `pip install -r
requirements.txt`, you are scorable. Until you've run exactly that, you are not.

### 3.3 `metrics.py` — you need this, it doesn't exist yet

Slide 6 requires SSIM, PSNR **and** LPIPS on your test split. You currently
measure only PSNR. Create this file:

```python
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
```

**Run it with `--checkpoint` omitted first.** That gives you the bicubic
baseline row. `[L]` It'll land near 24.7 dB — which is roughly where your
epoch-1 val PSNR was, meaning 100 epochs of training bought you about
**+1.8 dB over doing nothing**. Judges will ask for this comparison. Have it.

---

## 4. The 9-day plan

### Day 1 — Fri 7 Aug (tonight)
- [ ] **Kill the sweep.** Keep cfg 1–3 results; they're a legitimate LR ablation
      for the deck. Do not run configs 6–25.
- [ ] Run `analyze_degradation.py`. Save the output.
- [ ] Attend the KLA Q&A session (5 PM). Ask the questions in §6.
- [ ] Swap in the new `evaluation.py`.

### Day 2 — Sat 8 Aug
- [ ] Apply every patch in §2.
- [ ] Create `metrics.py`, run it with no checkpoint → bicubic baseline row.
- [ ] **Three calibration runs**, 12,000 steps each (~30 min each on T4):
      lr ∈ {2e-4, 5e-4, 1e-3}, everything else fixed at bs=8, patch=96,
      width=32, enc=[1,1,1,1]. `[L]` With clipping + warmup in place, 1e-3
      should now win comfortably. Pick the winner and **stop deciding.**
- [ ] Kick off the long run overnight.

### Days 3–6 — Sun 9 → Wed 12 Aug
- [ ] **The long run.** Winning LR, `synth_prob=0.75`, EMA on, cosine over your
      real horizon (aim 100k+ steps). Resume across Colab sessions. Don't
      babysit it — check in twice a day.
- [ ] In parallel, build your **own OOD test set**: grab grayscale
      microscopy/texture images from a different public source, apply your
      fitted degradation, evaluate. Nobody else will do this. It is the single
      most direct evidence you can offer against KLA's stated OOD criterion.
- [ ] Generate before/after visual triples (degraded → yours → GT) for Slide 6.
      Pick 3–4 that show a *structure* being recovered, not just less grain.

### Days 7–8 — Thu 13 → Fri 14 Aug
- [ ] Freeze the model. Whatever it is on Thursday morning, that's the model.
- [ ] Final `metrics.py` run: in-distribution + your OOD split + bicubic
      baseline, all three metrics. That's your Slide 6 table.
- [ ] Benchmark inference time per image (the new `evaluation.py` prints it).
- [ ] Write the 9-slide PDF. Naming: `NodeZero_KLA_PS01.pdf`.
- [ ] `pip freeze > requirements.txt`. Weights to HuggingFace or Drive.

### Day 9 — Sat 15 Aug
- [ ] **The fresh-machine test** (§3.2). New Colab runtime, `git clone`,
      `pip install -r requirements.txt`, run `evaluation.py` from outside the
      repo directory. This is the step that catches the `torch.load` bug.
- [ ] Record the ≤5 min demo video.
- [ ] Commit the restored-outputs folder.

### Sun 16 Aug
- [ ] **Submit in the morning.** Not at 11 PM.

---

## 5. Round 1 deliverables checklist

Round 1 (17–26 Aug) is human experts scoring innovation, technical approach,
feasibility, impact and scalability. It is **not** a PSNR leaderboard. Weight
your effort accordingly.

**Repo (all mandatory):**
- [ ] `README.md` — a reviewer must clone and run inference without contacting you
- [ ] `evaluation.py` — standalone `.py`, not a notebook, runs with zero edits
- [ ] Training script — reproduces training from scratch
- [ ] Trained weights — downloadable (Git LFS / Drive / HF)
- [ ] Restored test outputs folder
- [ ] `requirements.txt` — full `pip freeze`
- [ ] Public repo

**PDF (max 8–9 slides, remove the instruction slide):**

| Slide | Your content |
|---|---|
| 3 Idea | NAFNet + joint denoise/SR in one pass. Say *why* NAFNet: activation-free, fast at inference — tie it to the speed criterion. |
| 4 Solution | Architecture diagram + **the degradation-inversion analysis**. Loss design. Augmentation strategy. |
| 5 Innovation | Randomized synthetic degradation for OOD robustness. Your degradation-fitting method. Speed-oriented design (UNet at LR resolution, single pixel-shuffle). |
| 6 Results | PSNR/SSIM/LPIPS table with **three rows**: bicubic baseline, in-distribution, your self-built OOD split. Before/after triples. |
| 7 Feasibility | PyTorch, Colab T4, training hours, param count, ms/image. Be honest about the T4 — it reads as resourcefulness, not weakness. |

---

## 6. Questions to ask KLA (5 PM session)

1. **Are test inputs `.npy` or image files, and what output format does your
   scorer read?** `[C]` If you write `.npy` and they glob for `.png`, your
   output directory is empty as far as scoring is concerned. This is the
   highest-risk unknown you have.
2. **How are PSNR, SSIM, LPIPS and inference time weighted against each
   other?** You cannot make a rational speed/quality tradeoff without this.
3. Is inference timed per image or over the whole directory? (Determines
   whether warm-up cost matters.)
4. Does the test set contain both 512→256 and 256→128 pairs? Should one model
   handle both?

---

## 7. What I still need from you

- `train.py` — I've never seen it. §2.4's patches almost certainly apply but I
  can't confirm without reading it.
- The output of `analyze_degradation.py`.
- The checkpoint key schema `train.py` actually writes (`"args"` vs `"cfg"`),
  so I can confirm the new `evaluation.py` reads it correctly.

Send those and I'll write the modified `dataset.py` and `train.py` for you
directly.
