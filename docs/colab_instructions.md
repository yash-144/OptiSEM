# Colab Backup & Resume Instructions

This guide contains the exact commands you need to run in Google Colab to safely back up your sweep progress before your compute instance dies, and how to resume it later in a fresh instance.

---

## 💾 1. HOW TO BACKUP (Before your runtime expires)

Run these cells in Colab to save your current sweep progress to Google Drive. You can run this even if a configuration is only halfway done.

**Cell 1: Connect to Drive (if not already connected)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2: Zip the progress and move to Drive**
```bash
# Zip only the logs and checkpoints (excluding the heavy dataset)
!cd /content/sem-model && zip -r /content/sem-model-progress.zip sweep_log.json sweep_results.csv checkpoints/

# Copy the zip to your Google Drive root folder
!cp /content/sem-model-progress.zip /content/drive/MyDrive/
```

*Wait for the cell to finish, then verify that `sem-model-progress.zip` is in your Google Drive before closing the tab.*

---

## 🚀 2. HOW TO RESUME (In a brand new Colab session)

When you come back tomorrow with fresh compute hours, run these cells in order to restore everything and pick up exactly where you left off.

**Cell 1: Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2: Extract original code/dataset + overlay saved progress**
```bash
# Extract your original project (code + dataset)
!cp /content/drive/MyDrive/sem-model.zip /content/
!unzip -q /content/sem-model.zip -d /content/

# Extract your SAVED PROGRESS over the original files
!cp /content/drive/MyDrive/sem-model-progress.zip /content/
!unzip -o -q /content/sem-model-progress.zip -d /content/sem-model/
```

**Cell 3: Install dependencies**
```bash
%cd /content/sem-model
!pip install -q pytorch_msssim lpips
```

**Cell 4: Resume the sweep!**
```bash
!python sweep.py \
  --gt_dir dataset/train/GT \
  --degraded_dir dataset/train/NoisyLR \
  --channels 1 \
  --probe_epochs 100 \
  --amp
```

*(The script will automatically read `sweep_log.json`, skip the configurations you've already completed, and start training the next one in the queue.)*
