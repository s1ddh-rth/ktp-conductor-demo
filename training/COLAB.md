# Training on Colab — quick guide

The training script (`training/train.py`) is Colab-friendly. Here's the recipe:

## Cell 1 — clone + install

```python
!git clone https://github.com/s1ddh-rth/ktp-conductor-demo.git
%cd ktp-conductor-demo
!pip install -q uv
!uv pip install --system -e ".[training]"
```

## Cell 2 — get the dataset

TTPLA is hosted on GitHub by the original authors. As of writing, the
canonical link is `https://github.com/r3ab/ttpla_dataset` — confirm
before running (URLs drift). The repo provides image + JSON-annotation
download links.

```python
# Option A: from a Drive folder you've prepared
from google.colab import drive
drive.mount('/content/drive')
!ls /content/drive/MyDrive/ttpla

# Option B: download with kaggle / wget / similar
# (filled in based on the current dataset host)
```

The script expects:

```
ttpla/
  images/   *.jpg
  masks/    *.png   (binary 0/255)
```

If TTPLA ships with COCO/YOLO-format JSON annotations, run a one-time
`scripts/ttpla_to_masks.py` (you'll write this once and reuse) to
rasterise the polygon annotations to PNG masks.

## Cell 3 — train

```python
!python -m training.train \
    --data /content/drive/MyDrive/ttpla \
    --epochs 50 \
    --batch-size 12 \
    --resolution 768 \
    --split-mode canonical \
    --out /content/drive/MyDrive/ktp-weights
```

The `--resolution 768` and `--batch-size 12` defaults match the v2
recipe (see `docs/methodology.md` §10). `--split-mode canonical` (the
default) reads TTPLA's official three-way partition from
`splitting_dataset_txt/` (905 train / 109 val / 220 test); pass
`--split-mode random` only when re-running the v1 baseline.

Optional flags:

- `--limit N` — train on the first `N` (image, mask) pairs only. Use
  this for a 1-epoch smoke test before committing to the full run. In
  canonical mode, `--limit` trims the training split only; val and
  test integrity are preserved.
- `--resume /content/drive/MyDrive/ktp-weights/last.ckpt` — continue
  from a Lightning checkpoint (loads optimiser state too). Useful if
  the Colab session disconnects mid-training; the `ModelCheckpoint`
  callback writes to `--out` on every val-IoU improvement.

T4 GPU on Colab free tier should complete 50 epochs in ~3 hours at
768×768 (≈1.5× the v1 512×512 wall-clock). Watch out for the 12-hour
Colab session limit. If a session times out before 50 epochs complete,
the procedure is:

1. Note the latest `last.ckpt` Lightning wrote into `--out`.
2. Reconnect Colab and re-run the same training command with
   `--resume <path-to-last.ckpt>` appended.
3. Optimiser state, LR-schedule position, and epoch counter resume
   from the checkpoint, so the cosine schedule continues smoothly.

## Cell 4 — copy weights to the laptop

After training, you'll have `unet_resnet34_ttpla.pth` (~95 MB) in your
Drive. On the Fedora laptop:

```bash
# Place into the deployment directory
mkdir -p weights
cp ~/Downloads/unet_resnet34_ttpla.pth weights/
# Restart the server; it picks up the new weights at startup
sudo systemctl restart ktp-demo
# or, in dev: just re-run uvicorn
```

The inference path (`app/ml/model.py`) loads this file automatically on
startup.
