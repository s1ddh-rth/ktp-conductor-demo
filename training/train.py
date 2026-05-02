"""TTPLA U-Net training — Colab-ready.

Usage on Colab (T4)
-------------------
1. Upload TTPLA images + masks to your Drive (or download in-notebook).
2. Mount Drive, clone this repo, install [training] extras.
3. Run:  python -m training.train --data /content/drive/MyDrive/ttpla --epochs 50

The defaults below fit comfortably on a 16 GB T4 at 512×512, batch size
16. For a GTX 1050 Ti (4 GB) you would set --batch-size 4 --image-size 512
which works but trains ~5× more slowly.

CLI flags
---------
--data        path to TTPLA root (must contain images/ and masks/)
--limit N     train on the first N image/mask pairs (default 0 = all);
              useful for smoke-testing the pipeline before a full run.
--resume PATH continue training from a Lightning checkpoint (.ckpt)
              instead of starting fresh; loads optimiser state too.
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

# Image extensions the dataset will pick up. TTPLA releases vary in
# whether they ship .jpg or .JPG; we match all common variants.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ── dataset ─────────────────────────────────────────────────────────────
class TTPLADataset(Dataset):
    """Pairs of (image, binary cable mask) at fixed crop size.

    Expects a directory layout:
        root/
          images/  *.jpg|*.jpeg|*.png  (case-insensitive)
          masks/   *.png   (same basename; binary, 0=bg, 255=cable)

    Images without a matching mask are skipped at construction time
    (with a warning). This guards against partially-labelled releases
    where some annotation JSONs failed to convert.
    """

    def __init__(self, root: Path, transform: A.Compose, limit: int = 0):
        all_images = sorted(
            p for p in (root / "images").iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTS
        )
        if not all_images:
            raise FileNotFoundError(f"no images in {root/'images'}")

        masks_dir = root / "masks"
        kept: list[Path] = []
        n_missing = 0
        for img_path in all_images:
            if (masks_dir / f"{img_path.stem}.png").exists():
                kept.append(img_path)
            else:
                n_missing += 1
        if n_missing:
            warnings.warn(
                f"{n_missing}/{len(all_images)} images have no matching mask "
                f"under {masks_dir}; they will be skipped. Run "
                f"scripts/ttpla_to_masks.py first if this is unexpected.",
                stacklevel=2,
            )
        if limit > 0:
            kept = kept[:limit]
        if not kept:
            raise FileNotFoundError(
                f"no (image, mask) pairs found under {root}; "
                f"check that masks/ contains *.png matching the image stems."
            )

        self.images = kept
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("weights"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="train on first N image/mask pairs (0 = all). Useful for smoke tests.",
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

    train_tf, val_tf = build_transforms(args.image_size)
    full = TTPLADataset(args.data, transform=train_tf, limit=args.limit)
    n_val = max(1, len(full) // 10)
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset = TTPLADataset(args.data, transform=val_tf, limit=args.limit)

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

    # Save a clean state_dict for the inference server
    best_path = ckpt.best_model_path
    print(f"\nBest checkpoint: {best_path}")
    best = ConductorModule.load_from_checkpoint(best_path)
    torch.save(best.model.state_dict(), args.out / "unet_resnet34_ttpla.pth")
    print(f"Saved deployable weights → {args.out / 'unet_resnet34_ttpla.pth'}")


if __name__ == "__main__":
    main()
