"""
Supervised fine-tuning of TSViT on labelled LC patches.

Workflow:
  1. Load a DINO-pretrained TSViT backbone (optional — trains from scratch if not provided)
  2. Attach a fresh segmentation head (the mlp_head in TSViTdense.TSViT)
  3. Fine-tune on (image, label_mask) pairs with masked cross-entropy loss
  4. Save checkpoints every N epochs

Label format:
  Each labelled patch is an xr.Dataset with variables matching HLSPatchDataset
  PLUS a variable "label" of shape (H, W) with integer LC class indices (0 = ignore).

Usage:
    python -m training.finetune_segmentation \\
        --patch-dir  data/hls_patches_labelled \\
        --output-dir runs/finetune \\
        --config     configs/default.yaml \\
        --backbone   runs/dino_pretrain/dino_backbone_final.pth
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from datasets.hls_loader import HLSPatchDataset, N_CHANNELS
from models.TSViT.TSViTdense import TSViT


# ---------------------------------------------------------------------------
# Labelled dataset
# ---------------------------------------------------------------------------

class LabelledPatchDataset(Dataset):
    """
    Extends HLSPatchDataset to also load per-pixel LC labels.

    Expects each NetCDF patch to contain a variable 'label' of shape (H, W)
    with integer class indices. Background / ignore is class 0.
    """

    def __init__(
        self,
        patch_dir: str,
        end_date: str = "2023-12-31",
        n_steps: int = 24,
        img_size: int = 48,
        augment: bool = True,
    ):
        self._img_ds = HLSPatchDataset(
            patch_dir, end_date=end_date, n_steps=n_steps,
            img_size=img_size, augment=augment, return_path=True,
        )

    def __len__(self):
        return len(self._img_ds)

    def __getitem__(self, idx):
        import xarray as xr
        tensor, path = self._img_ds[idx]

        ds = xr.open_dataset(path, mask_and_scale=False)
        if "label" not in ds:
            raise KeyError(f"No 'label' variable in {path}")
        label = torch.from_numpy(ds["label"].values.astype(np.int64))   # (H, W)

        # Align spatial size with img_size
        H, W = tensor.shape[1], tensor.shape[2]
        if label.shape != (H, W):
            label = label[:H, :W]

        return tensor, label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy ignoring pixels where label == ignore_index (default 0)."""

    def __init__(self, num_classes: int, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: (B, num_classes, H, W)
        # labels: (B, H, W)
        return self.ce(logits, labels)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SegmentationTrainer:
    def __init__(
        self,
        model_config: dict,
        patch_dir: str,
        output_dir: str,
        backbone_path: Optional[str] = None,
        end_date: str = "2023-12-31",
        n_steps: int = 24,
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        val_split: float = 0.15,
        device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs     = epochs
        self.device     = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Dataset / loaders
        full_ds = LabelledPatchDataset(patch_dir, end_date=end_date, n_steps=n_steps, augment=True)
        n_val   = max(1, int(len(full_ds) * val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=4, pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                       num_workers=2, pin_memory=True)

        # Model
        self.model = TSViT(model_config).to(self.device)

        # Load DINO backbone weights if provided (only backbone, skip head)
        if backbone_path and os.path.isfile(backbone_path):
            ckpt = torch.load(backbone_path, map_location="cpu")
            state = ckpt.get("backbone", ckpt)
            # Strip the _full. prefix added by TSViTBackbone wrapper
            remapped = {}
            for k, v in state.items():
                new_k = k.replace("_full.", "")
                remapped[new_k] = v
            missing, unexpected = self.model.load_state_dict(remapped, strict=False)
            print(f"Loaded backbone from {backbone_path}")
            print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

        # Separate LRs: lower for pre-trained backbone, higher for head
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if "mlp_head" not in n
        ]
        head_params = list(self.model.mlp_head.parameters())
        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr},
        ], weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.criterion = MaskedCrossEntropyLoss(model_config["num_classes"]).to(self.device)

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train=True)
            val_loss   = self._run_epoch(train=False)
            self.scheduler.step()

            print(f"Epoch {epoch:3d}/{self.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(epoch, tag="best")

            if epoch % 10 == 0:
                self._save(epoch)

        self._save(self.epochs, tag="final")

    def _run_epoch(self, train: bool) -> float:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        total, n = 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in loader:
                imgs   = imgs.to(self.device)       # (B, T, H, W, C)
                labels = labels.to(self.device)     # (B, H, W)

                logits = self.model(imgs)            # (B, num_classes, H, W)
                loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total += loss.item() * imgs.shape[0]
                n     += imgs.shape[0]

        return total / max(n, 1)

    def _save(self, epoch: int, tag: str = ""):
        suffix = f"_{tag}" if tag else ""
        path   = self.output_dir / f"tsviT_seg_ep{epoch:04d}{suffix}.pth"
        torch.save({
            "epoch":        epoch,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
        }, path)
        print(f"  Saved -> {path}")

    @torch.no_grad()
    def compute_iou(self, loader) -> dict:
        """Compute per-class IoU over a DataLoader."""
        self.model.eval()
        num_classes = self.model.num_classes
        inter = torch.zeros(num_classes)
        union = torch.zeros(num_classes)

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds = self.model(imgs).argmax(dim=1)     # (B, H, W)
            for c in range(num_classes):
                pred_c  = preds   == c
                label_c = labels  == c
                inter[c] += (pred_c & label_c).sum().float()
                union[c] += (pred_c | label_c).sum().float()

        iou = {}
        for c in range(num_classes):
            iou[c] = float(inter[c] / union[c].clamp(min=1))
        iou["mean"] = float(np.mean(list(iou.values())))
        return iou


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir",    required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--backbone",     default=None, help="Path to DINO pretrained backbone .pth")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--lr-backbone",  type=float, default=1e-5)
    p.add_argument("--end-date",     default="2023-12-31")
    p.add_argument("--n-steps",      type=int,   default=24)
    p.add_argument("--device",       default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = SegmentationTrainer(
        model_config=cfg["model"],
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        backbone_path=args.backbone,
        end_date=args.end_date,
        n_steps=args.n_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        device=args.device,
    )
    trainer.train()

    # Report final IoU on validation set
    iou = trainer.compute_iou(trainer.val_loader)
    print("\nValidation IoU per class:")
    for k, v in iou.items():
        print(f"  class {k}: {v:.4f}")
