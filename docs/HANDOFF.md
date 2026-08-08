# HANDOFF — Team Node Zero / KLA Hackathon 2026 (PS01: SEM Image Restoration)

**Written 8 Aug 2026. Submission deadline 16 Aug 2026.**
Paste this whole file into a new chat to resume. It is self-contained.

Confidence tags used throughout: `[C]` certain (verified in code or data),
`[L]` likely (strong inference), `[G]` guessing.

---

## 0. Who/what

- Team Node Zero, SRM University Haryana Delhi-NCR. KLA Hackathon 2026, PS01.
- Task: joint denoising + 2× super-resolution of degraded SEM images.
- Model: NAFNet-based `NAFNetSR`, custom. Training on Colab T4.
- User has no prior ML training experience. Explain reasoning, don't just
  hand over commands.
- **User's advisor preferences: challenge assumptions in the first sentence,
  tag every claim with confidence, lead with the uncomfortable truth, never
  open with agreement, hold position under pushback unless given new
  information.** These are strong preferences — follow them.

### Hackathon facts (verified from https://i4c.in/hackathon-2026/)
- Deadline **16 Aug 2026**. Round 1 evaluation **17–26 Aug**.
- Round 1 is **human subject-matter experts** scoring innovation, technical
  approach, feasibility, impact, scalability. **NOT a PSNR leaderboard.**
- Test set is explicitly **out-of-distribution** — images from different
  sources than training data.
- Inference time is explicitly benchmarked (H100).
- Submission requires: public repo, standalone `evaluation.py` that runs with
  zero edits, training script, downloadable weights, restored test outputs,
  `requirements.txt`, ≤9-slide PDF, ≤5-min demo video.
- Test inputs may be 512→256 **and** 256→128. Training data is 256→128 only.

---

## 1. CURRENT STATE — measured numbers

### Model A (`model_A_p96.pth`) — the current best, frozen and backed up
`width=32, enc=[1,1,1,1], 4.54M params`, 250 epochs, lr 5e-4, bs 8, patch 96,
`synth_prob=0.75`, ssim_w 0.2, lpips_w 0.1, EMA 0.999, AMP.
Location: `/content/drive/MyDrive/kla_backup/model_A_p96.pth`

Held-out split: **160 images, stride-20 rule** (`sorted(stems)[::20]`).
This rule is used identically by `train.py`, `metrics.py`, `ood_stress.py`,
`ceiling.py`. Do not change it.

```
method            PSNR (dB)     SSIM    LPIPS
bicubic              22.676   0.5316   0.4358
Model A              27.673   0.7768   0.1407      <- +5.00 dB over bicubic
```

### Ceiling decomposition — THE KEY FINDING
```
bicubic on real LR                     22.676   0.5316
Model A on real LR                     27.673   0.7768
---
bicubic on CLEAN LR (no noise)         31.403   0.8731
Model A on CLEAN LR                    31.244   0.8703   <- BELOW bicubic
Model A, synthetic noise 0.5x          28.963   0.8078
Model A, synthetic noise 1.0x          27.207   0.7510
```

`[C]` **Model A is worse than bicubic on clean input.** It has learned
essentially zero super-resolution. All +5.00 dB of its gain is denoising.
`[L]` A properly trained 2× SR net beats bicubic on clean data by 2–4 dB, so
the true ceiling is ~33–35 dB, not 31.24, and real headroom is ~6 dB.
`[C]` Caveat: clean input is mildly OOD for a model trained on noisy input,
so some of the −0.16 dB is distribution shift, not incapacity. Direction is
still unambiguous.

**Root cause** `[L]`: in `NAFNetSR.forward`, after `pixel_shuffle` there is
exactly ONE 3×3 conv (`upconv2`, width→channels) at high resolution. The
entire 4.54M-param UNet runs at LOW resolution. All HR detail synthesis falls
on that single conv. Under capacity pressure the model spends everything on
denoising (worth ~5 dB) and lets the bicubic skip handle upsampling.

**Secondary cause** `[C]`: `scale = random.uniform(0.3, 2.5)` in
`_synth_degrade` means the model never sees a near-clean image, so it was
never asked to do SR without denoising.

### OOD stress test (Model A)
```
condition          bicubic PSNR  model PSNR    gain     SSIM    LPIPS
real (in-dist)           22.676      27.673   +5.00   0.7768   0.1407
noise 0.3x               28.288      29.933   +1.65   0.8366   0.1068
noise 1.0x               21.999      27.190   +5.19   0.7506   0.1680
noise 2.5x               15.460      24.640   +9.18   0.6567   0.2508
noise 4.0x  OOD          12.436      23.267  +10.83   0.6066   0.3059
noise 6.0x  OOD          10.458      20.793  +10.34   0.5165   0.4157
gauss-blur  OOD          21.349      25.926   +4.58   0.6975   0.2126
additive    OOD          22.143      23.670   +1.53   0.5469   0.3922
```

`[C]` Strong result for the deck: +10.34 dB at 6× noise (far outside the
0.3–2.5× training range), +4.58 dB on an unseen downsample operator.
`[C]` **Honest limitation to state on the slide:** the `additive` row (+1.53
dB, SSIM 0.5469) shows the model is robust to noise *strength* shifts but not
to noise *shape* shifts — it learned the power-law structure specifically.
State this. A measured failure boundary reads as rigor.

`[C]` Also state: these OOD rows use *our* degradation model at unseen
strengths, not KLA's genuinely different image sources. It measures
degradation-shift robustness, NOT source-shift robustness. Do not overclaim.

---

## 2. THE DEGRADATION MODEL (reverse-engineered)

Verified via `analyze_degradation.py` on 200 pairs.

- Scale ratio confirmed exactly 2.00×. GT [0,1], LR [0.0009, 1.1665].
- **Downsample kernel: `area`** (mean|residual| 0.0629 vs bicubic 0.0649,
  bilinear 0.0667). `[L]` — a 3% margin, but real.
- Noise model: the script's built-in speckle+Gaussian fit reported
  `speckle_var=0.026365, gauss_var=0.001797, R²=0.9781`. **This fit was
  REJECTED.** `[C]` R² was masking an 1106% relative error in the darkest
  bin — it injected σ=0.0432 where reality has 0.0150 (2.9× too noisy in dark
  regions, exactly where SEM defects hide).
- **Accepted model:** `Var(μ) = 0.0280 · μ^1.622`, i.e.
  **`σ(μ) = 0.1673 · μ^0.811`**. Max relative error 19.6% across all 24 bins
  (vs 1106% for the rejected fit).
- `[L]` Exponent 1.62 sits between Poisson shot noise (1.0) and pure
  multiplicative speckle (2.0) — physically consistent with SEM
  (electron-counting shot noise + detector gain fluctuation). **Good Slide 4
  material.**
- Constants in code: `noise_a=0.1673`, `noise_p=0.811`.

---

## 3. FILE INVENTORY

| File | State |
|---|---|
| `model.py` | Working. **hr_blocks patch PENDING — see §5.** |
| `dataset.py` | Patched: D4 aug, non-2× warning, `_synth_degrade` with clamp_min(0). **noise-range patch PENDING — see §5.** |
| `losses.py` | Patched: straight-through clamp, `.detach()` instead of `.item()`. Uses single-scale `ssim`, NOT MS-SSIM. |
| `train.py` | Rewritten. Working resume, NaN guard, stride-20 split, EMA. **hr_blocks flag PENDING.** |
| `evaluation.py` | Rewritten. **hr_blocks read PENDING — mandatory or submission breaks.** |
| `metrics.py` | Working. PSNR/SSIM/LPIPS + bicubic baseline on stride-20 split. |
| `analyze_degradation.py` | Done its job. Keep for the deck. |
| `check_synth.py` | Done its job. |
| `ood_stress.py` | Working. |
| `ceiling.py` | Working. |
| `sweep.py` | **RETIRED. Do not run.** |
| `infer.py`, `compare.py` | Never reviewed. May be broken. |
| `README.md` | MS-SSIM→SSIM fixed. |

---

## 4. BUGS FOUND AND FIXED (do not reintroduce)

**`evaluation.py` (all `[C]`, was pass/fail risk):**
1. `torch.load` without `weights_only=False` — PyTorch ≥2.6 refuses to
   unpickle the checkpoint. Would have made the submission unscorable.
2. Read arch from `"args"` but `sweep.py` saved `"cfg"` — silent wrong width.
3. `--checkpoint` default was a *relative* path — FileNotFoundError when run
   from another directory (which is how KLA runs it).
4. `torch.compile` is lazy so the try/except caught nothing, and it recompiles
   per input shape (test set has both 128 and 256). Removed.
5. No padding — model needs input divisible by **16** (4 stride-2 stages),
   not 8. Now reflect-pads and crops back.
6. One image at a time with a `.cpu()` sync each. Now batched by shape.
7. `autocast('cuda')` called on CPU.
8. Always wrote `.npy`. Now writes back in the input's format.

**`losses.py`:** `pred.clamp(0,1)` gave zero gradient outside [0,1] (inputs
reach 1.45 via the bicubic skip) → straight-through clamp. Three `.item()`
calls forced GPU→CPU sync every step → `.detach()`.

**`dataset.py`:** missing transpose (only 4 of 8 D4 symmetries); silent
bicubic *downscale* of GT on non-2× pairs (now warns).

**`train.py`:** resume restarted the LambdaLR at step 0 while `start_epoch`
jumped ahead (warmup re-run, cosine never completed); EMA shadow and scaler
not restored; `last.pth` contained EMA weights not raw weights; `torch.load`
weights_only; split rule disagreed with `metrics.py`.

**`sweep.py` methodological flaws:** no gradient clipping (caused all NaNs
above 2e-4, which invalidated the "2e-4 is best" conclusion); epoch-based
budget confounded batch size (bs=4 → 760 steps/epoch vs bs=32 → 95, an 8×
spread); patch 160 > image size 128 so config 11 trained on mirrored padding;
PSNR computed unclamped while `evaluation.py` clamps.

**The NaN mystery — SOLVED:** `[L]` SSIM computed in fp16. `pytorch_msssim`
derives local variance as `E[x²] − (E[x])²`; once `pred ≈ target` these terms
nearly cancel in fp16, giving small *negative* variance → division/sqrt → NaN.
Explains: why it struck around epoch 12 (needs pred≈target), why the LOWEST
LR died first (not LR-driven at all), why config 102 survived (still improving,
never plateaued), and why the original sweep never saw it (different tensor fed
to SSIM before the straight-through clamp was added).
**Fix, already applied in `train.py`:** compute the forward in autocast, then
the loss OUTSIDE it in fp32:
```python
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(deg)
            loss, _, _, _ = criterion(pred.float(), gt.float())
```
Result: SKIPPED count dropped to ~1 per epoch (≈1 in 380 batches).
`[L]` Those residual skips are blank/uniform crops where SSIM degenerates —
benign, the guard eats them.

---

## 5. PENDING PATCHES — apply before the next run

### `model.py` — add HR-resolution capacity
```python
# __init__ signature:
    def __init__(self, channels=1, width=32, scale=2,
                 enc_blk_nums=None, dec_blk_nums=None, hr_blocks=2):

# FIND:
        self.upconv1 = nn.Conv2d(width, width * (scale ** 2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.upconv2 = nn.Conv2d(width, channels, 3, 1, 1, bias=True)
# REPLACE WITH:
        self.upconv1 = nn.Conv2d(width, width * (scale ** 2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        # HR-resolution processing. Previously a single 3x3 conv had to
        # synthesize ALL high-frequency detail — the reason this model
        # scored BELOW bicubic on clean input.
        self.hr_blocks = nn.Sequential(
            *[NAFBlock(width) for _ in range(hr_blocks)]) if hr_blocks > 0 \
            else nn.Identity()
        self.upconv2 = nn.Conv2d(width, channels, 3, 1, 1, bias=True)

# In forward, FIND:
        residual = self.pixel_shuffle(residual)
        residual = self.upconv2(residual)
# REPLACE WITH:
        residual = self.pixel_shuffle(residual)
        residual = self.hr_blocks(residual)
        residual = self.upconv2(residual)
```

### `dataset.py` — force SR learning
```python
# FIND:
        scale = random.uniform(0.3, 2.5)          # OOD robustness
# REPLACE WITH:
        scale = random.uniform(0.0, 2.5)          # 0.0 forces SR learning
```

### `train.py`
```python
    p.add_argument("--hr_blocks", type=int, default=2)
```
and pass `hr_blocks=args.hr_blocks` into the `NAFNetSR(...)` call.
(`train.py` saves `vars(args)` into the checkpoint, so it propagates.)

### `evaluation.py` — MANDATORY, or the submission breaks
In `build_model`, after the `dec` parsing:
```python
    hr_blocks = int(cfg.get("hr_blocks", 0))     # 0 = old checkpoints
```
and add `hr_blocks=hr_blocks` to the `NAFNetSR(...)` call.
The `0` default keeps `model_A_p96.pth` loadable.

`[G]` Expected inference cost: +40–60% (2 NAFBlocks at 256×256 = 4× the
spatial area of the LR path). Inference time is scored — watch the ms/image
number `evaluation.py` prints.

---

## 6. NEXT COMMANDS

```bash
# Run C — the HR-capacity experiment
!python train.py --gt_dir dataset/train/GT --degraded_dir dataset/train/NoisyLR \
  --channels 1 --epochs 250 --lr 5e-4 --batch_size 8 --patch_size 96 \
  --width 32 --enc_blocks 1,1,1,1 --hr_blocks 2 \
  --ssim_weight 0.2 --lpips_weight 0.1 --synth_prob 0.75 --val_stride 20 \
  --out_dir /content/drive/MyDrive/kla_backup/run_C_hr --amp \
  2>&1 | tee -a /content/drive/MyDrive/kla_backup/run_C_hr/train_log.txt

# Then re-run ceiling.py on it. The row that matters is
# "YOUR MODEL on CLEAN LR" — if it moves above 31.403 (bicubic), the SR
# path is finally working and the gain will carry into the noisy case.
!python ceiling.py --gt_dir dataset/train/GT --degraded_dir dataset/train/NoisyLR \
  --checkpoint /content/drive/MyDrive/kla_backup/run_C_hr/best.pth
```

**Resume rule** `[C]`: identical flags plus `--resume .../last.pth`.
`--epochs` defines the cosine horizon; changing it silently changes the
schedule.

**Timing reference:** 250 epochs at patch 96 / bs 8 / width 32 ≈ **2.2 hours**
on T4 (31–33 s/epoch). Add ~40–60% for `hr_blocks=2`.

### STILL UNVERIFIED — the pass/fail test
Colab's `!` uses `sh`, not bash — brace expansion `{0,1,2}` fails. Use:
```bash
!mkdir -p /content/quick_test
!cp dataset/train/NoisyLR/000000.npy dataset/train/NoisyLR/000001.npy \
    dataset/train/NoisyLR/000002.npy dataset/train/NoisyLR/000003.npy /content/quick_test/
%cd /content
!python /content/sem-model/evaluation.py /content/quick_test /content/quick_out \
    --checkpoint /content/drive/MyDrive/kla_backup/model_A_p96.pth
!ls -la /content/quick_out
%cd /content/sem-model
```
Must print a ms/image number and produce 4 output files. The checkpoint load
and path resolution from `/content` already verified working; only the file
copy failed.

**Before submission, the real test:** fresh Colab runtime, `git clone` the
public repo, `pip install -r requirements.txt`, run `evaluation.py` from
OUTSIDE the cloned directory. This is the step that catches everything.

---

## 7. REMAINING LEVERS (ranked, with honest estimates)

1. **HR blocks + noise range 0.0** — the run above. `[G]` +0.5–1.5 dB. This is
   the only one addressing the actual diagnosed problem.
2. **Drop LPIPS from the loss.** `[L]` +0.2–0.4 dB PSNR. `[C]` LPIPS is a
   perceptual loss and deliberately trades pixel fidelity. But LPIPS is also a
   scored metric — **blocked on KLA's metric weighting.**
3. **Narrow the synthetic noise range** (e.g. 0.6–1.6×). `[L]` +0.3–0.5 dB
   in-distribution, at the cost of the OOD robustness that is the innovation
   story. Deliberate tradeoff — measure with `ood_stress.py` before choosing.
4. **Switch L1 → PSNR/Charbonnier loss.** `[L]` +0.1–0.3 dB.
5. **Capacity: width 48/64 or enc [2,2,4,8].** `[L]` +0.3–0.6 dB, costs
   inference time.
6. **Self-ensemble ×8 D4 TTA at inference.** `[C]` +0.1–0.3 dB reliably,
   costs 8× inference time.
7. **Patch 128.** `[L]` +0.1–0.3 dB. The one axis the original sweep never
   actually tested (configs 10 and 11 were both broken).

`[G]` Realistic stacked ceiling: ~29–30 dB. 30+ likely requires a much larger
model, which directly damages the benchmarked speed score.

**Do not restart hyperparameter sweeping.** `[C]` Validation noise in the
original sweep was ±1 dB epoch-to-epoch while config gaps were 0.17–0.42 dB.
The measurement had no statistical power. Config selection is settled:
lr 5e-4, bs 8, patch 96, width 32, enc [1,1,1,1], EMA 0.999, AMP.

---

## 8. OPEN QUESTIONS FOR KLA — still unanswered, both are blocking

1. **`[C]` Are test inputs `.npy` or image files, and what output format does
   the scorer read?** If we write `.npy` and they glob for `.png`, the output
   directory is empty as far as scoring is concerned. Highest-risk unknown.
2. **How are PSNR, SSIM, LPIPS and inference time weighted?** This has now
   blocked two separate decisions (the LPIPS loss weight and the
   capacity-vs-speed tradeoff).
3. Is inference timed per image or over the whole directory?
4. Does the test set contain both 512→256 and 256→128?

There was a KLA Q&A session (Akshat Singh, ML Research Engineer) listed for
7 Aug 5 PM. Unknown whether the user attended. Check for a recording.

---

## 9. SCHEDULE — 8 days left, and the deck is EMPTY

Priority order `[C]`: (1) a runnable submission — binary pass/fail;
(2) OOD generalization — explicitly tested; (3) inference speed — explicitly
benchmarked; (4) absolute metrics; (5) hyperparameters — do not touch.

- **8–9 Aug:** apply §5 patches, launch Run C in the background. Complete the
  pass/fail test. **Start the deck.**
- **9–11 Aug:** visual before/after triples (pick images where a *structure*
  is recovered, not just where grain disappears — a judge cannot see 27.6 vs
  26.6 but can see an edge return). Slide 6 tables from `metrics.py` +
  `ood_stress.py`. Draft all 9 slides.
- **12–13 Aug:** freeze the model. Final metrics run. `pip freeze >
  requirements.txt`. Weights to HuggingFace/Drive. Restored outputs folder.
- **14–15 Aug:** fresh-machine test. Demo video. README.
- **16 Aug:** submit in the morning.

### Slide plan
| Slide | Content |
|---|---|
| 3 Idea | NAFNet + joint denoise/SR in one pass. Say *why* NAFNet: activation-free, fast at inference — tie explicitly to the speed criterion. |
| 4 Solution | Architecture diagram. **The degradation-inversion analysis** (power law, exponent 1.62 between Poisson and speckle). Loss design. D4 augmentation. |
| 5 Innovation | Randomized synthetic degradation for OOD robustness. The reverse-engineering method. Deliberate over-coverage of the noise range beyond the observed training distribution. Speed-oriented design. |
| 6 Results | Three-row table: bicubic baseline / in-distribution / self-built OOD stress test. All three metrics. Before/after triples. |
| 7 Feasibility | PyTorch, Colab T4, ~2.2 h training, 4.54M params, ms/image. Be honest about the T4 — it reads as resourcefulness. |

---

## 10. CORRECTIONS LOG — advisor calls that were WRONG

Recorded so the next chat doesn't repeat them and so confidence is calibrated:

1. Predicted config 9 (patch 64) would crash on MS-SSIM's 160px minimum.
   **Wrong** — `losses.py` uses single-scale `ssim`, which has no such floor.
2. Estimated the bicubic baseline at ~24.7 dB from the epoch-1 val PSNR, and
   claimed training bought "+1.8 dB." **Wrong** — actual baseline 22.676,
   actual gain +5.00 dB.
3. Diagnosed the calibration-run NaNs as a missing `clamp_min(0.0)` before
   `.pow()`. **Wrong** — the clamp was already present. Real cause was fp16
   SSIM catastrophic cancellation.
4. Estimated 250 epochs would take 6–10 h on T4. **Wrong by 3–4×** — actual
   2.2 h.
5. Forecast a final 27.5–28.2 dB, then revised down to 27.3–27.5. Actual
   27.673. First forecast was correct; the revision was too pessimistic.
6. Said "both edits" when only one edit existed (the second was for the
   retired `sweep.py`). Ambiguous wording.

Things that were right and load-bearing: the `torch.load` weights_only bug,
the epoch-vs-iteration confound, the missing gradient clipping, the R²
misdiagnosis of the noise model, and the ceiling decomposition that revealed
the model does no super-resolution.
