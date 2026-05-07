"""TTPLA U-Net training — Colab-ready.

Usage on Colab (T4)
-------------------
1. Upload TTPLA images + masks to your Drive (or download in-notebook).
2. Mount Drive, clone this repo, install [training] extras.
3. Run:  python -m training.train --data /content/drive/MyDrive/ttpla --epochs 50

The v2 defaults are tuned for a 16 GB T4 at 768×768, batch size 12
(matches the production sliding-window tile size; see
``docs/methodology.md`` §10). For a GTX 1050 Ti (4 GB) you would set
``--batch-size 2 --resolution 512`` which works but trains substantially
more slowly and re-introduces the v1 inference-path drift.

CLI flags
---------
--data        path to TTPLA root.
              Layout is auto-detected:
                <data>/images/ + <data>/masks/   (preferred, back-compat)
                <data>/*.jpg   + <data>/masks/   (TTPLA's actual flat layout)
              Override either with --images-dir / --masks-dir.
--split-mode  'canonical' (default, v2) uses TTPLA's official three-way
              partition from <data>/splitting_dataset_txt/
              (train.txt: 905, val.txt: 109, test.txt: 220 = 1234 images).
              'random' reproduces v1's seed-42 random_split over all
              images (preserved for re-running v1 only).
--resolution  training crop / validation resize size (default 768).
              Matches the production sliding-window tile so train- and
              serve-time see cables at the same pixel width.
--limit N     trim training subset to first N pairs (default 0 = all);
              useful for smoke-testing the pipeline before a full run.
              In canonical mode this trims the training split only —
              val and test integrity are preserved.
--resume PATH continue training from a Lightning checkpoint (.ckpt)
              instead of starting fresh; loads optimiser state too.
              Useful when a Colab session disconnects mid-training.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import albumentations as A  # noqa: N812 — community convention
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics import JaccardIndex

from training.canonical import discover_images, load_canonical_splits


# ── dataset ─────────────────────────────────────────────────────────────
class TTPLADataset(Dataset):
    """Pairs of (image, binary cable mask) at fixed crop size.

    Decoupled from directory discovery: callers pass an explicit list
    of image paths so that canonical-split logic can construct the
    train / val / test buckets up front and hand each one a fresh
    dataset with its own transform pipeline.

    Images without a matching mask (under ``masks_dir``) are skipped at
    construction time with a warning. This guards against partially-
    labelled releases where some annotation JSONs failed to convert.
    """

    def __init__(
        self,
        images: list[Path],
        masks_dir: Path,
        transform: A.Compose,
    ):
        if not images:
            raise FileNotFoundError("TTPLADataset received an empty image list")

        kept: list[Path] = []
        n_missing = 0
        for img_path in images:
            if (masks_dir / f"{img_path.stem}.png").exists():
                kept.append(img_path)
            else:
                n_missing += 1
        if n_missing:
            warnings.warn(
                f"{n_missing}/{len(images)} images have no matching mask "
                f"under {masks_dir}; they will be skipped. Run "
                f"scripts/ttpla_to_masks.py first if this is unexpected.",
                stacklevel=2,
            )
        if not kept:
            raise FileNotFoundError(
                f"no (image, mask) pairs from the requested {len(images)} images "
                f"under {masks_dir}; check that masks/ contains *.png matching the image stems."
            )

        self.images = sorted(kept)
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img_path = self.images[i]
        mask_path = self.masks_dir / (img_path.stem + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)


def resolve_images_dir(data: Path, override: Path | None) -> Path:
    """Pick the images directory.

    Honours an explicit override; otherwise prefers ``<data>/images``
    (the convention assumed by v1 and by ``scripts/ttpla_to_masks.py``)
    and falls back to ``<data>`` itself for the flat layout TTPLA
    actually ships (.jpg + .json side-by-side in the dataset root).
    """
    if override is not None:
        return override
    nested = data / "images"
    if nested.is_dir():
        return nested
    return data


def build_transforms(image_size: int) -> tuple[A.Compose, A.Compose]:
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# ── model ───────────────────────────────────────────────────────────────
class ConductorModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        # Focal handles class imbalance; Dice rewards overlap on thin objects
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.iou = JaccardIndex(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = 0.5 * self.focal(logits, y) + 0.5 * self.dice(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).int()
        iou = self.iou(preds, y.int())
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/iou", iou, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):  # noqa: D401
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


# ── entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    # v2 defaults: bs 12 fits 768×768 on a 16 GB T4 alongside AMP
    # activations; bs 16 OOMs at this resolution. Resolution 768 matches
    # the production sliding-window tile so train- and serve-time see
    # cables at the same pixel width.
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--resolution", type=int, default=768,
                   help="training crop / validation resize size (default 768).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("weights"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--split-mode",
        choices=["random", "canonical"],
        default="canonical",
        help=(
            "How to partition images. 'canonical' (default, v2) uses "
            "TTPLA's official splitting_dataset_txt/ partition: train.txt "
            "(905), val.txt (109), test.txt (220). 'random' reproduces "
            "the v1 seed-42 random_split for re-running v1 exactly."
        ),
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="image directory (default: <data>/images, fallback <data>).",
    )
    p.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="mask directory (default: <data>/masks).",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="canonical-splits directory (default: <data>/splitting_dataset_txt).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "trim the train subset to first N pairs (0 = all). Useful "
            "for smoke tests; in canonical mode val/test stay intact."
        ),
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume training from a Lightning .ckpt (loads optimiser state too).",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)

    image_size = args.resolution

    images_dir = resolve_images_dir(args.data, args.images_dir)
    masks_dir = args.masks_dir or (args.data / "masks")
    splits_dir = args.splits_dir or (args.data / "splitting_dataset_txt")

    train_tf, val_tf = build_transforms(image_size)

    if args.split_mode == "canonical":
        train_imgs, val_imgs, _test_imgs = load_canonical_splits(splits_dir, images_dir)
        if args.limit > 0:
            train_imgs = train_imgs[: args.limit]
        train_ds: Dataset = TTPLADataset(train_imgs, masks_dir, transform=train_tf)
        val_ds: Dataset = TTPLADataset(val_imgs, masks_dir, transform=val_tf)
    else:
        all_images = discover_images(images_dir)
        if args.limit > 0:
            all_images = all_images[: args.limit]
        full = TTPLADataset(all_images, masks_dir, transform=train_tf)
        n_val = max(1, len(full) // 10)
        n_train = len(full) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        # The random_split shares a single underlying transform; for the
        # validation subset we substitute a no-augmentation pipeline.
        val_ds.dataset = TTPLADataset(all_images, masks_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = ConductorModule(lr=args.lr)
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=args.out,
        filename="unet_resnet34_ttpla-{epoch:02d}-{val/iou:.3f}",
        monitor="val/iou",
        mode="max",
        save_top_k=2,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # only helps on Ampere+; harmless on T4
        callbacks=[ckpt, pl.callbacks.LearningRateMonitor()],
        gradient_clip_val=1.0,
    )
    resume_path = str(args.resume) if args.resume else None
    if resume_path and not Path(resume_path).exists():
        raise FileNotFoundError(f"--resume path not found: {resume_path}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    # Save a clean state_dict for the inference server. v2 weights live
    # alongside v1 weights; the inference path can be pointed at either.
    best_path = ckpt.best_model_path
    print(f"\nBest checkpoint: {best_path}")
    best = ConductorModule.load_from_checkpoint(best_path)
    out_name = "unet_resnet34_ttpla.pth" if args.split_mode == "random" else "unet_resnet34_ttpla_v2.pth"
    torch.save(best.model.state_dict(), args.out / out_name)
    print(f"Saved deployable weights → {args.out / out_name}")


if __name__ == "__main__":
    main()
