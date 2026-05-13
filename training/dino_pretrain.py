"""
DINO self-supervised pretraining for TSViT backbone.

Uses DinoSatTileDataset which:
  - Calls get_random_date_tile_as_image() per __getitem__ -> random temporal window
    per epoch across all available dates (multi-year data is automatically exploited)
  - Applies DataAugmentationDINOV2: 2 global spatial crops + N local spatial crops
    with random flips, rotations, spectral jitter, gaussian blur

The collated batch dict has keys:
  "collated_global_crops" : (2*B, T, C+1, H, W)   <- student + teacher
  "collated_local_crops"  : (N*B, T, C+1, H, W)   <- student only

TSViTBackbone receives permuted tensors: (B, T, H, W, C+1).

Usage:
    python -m training.dino_pretrain \\
        --patch-dir data/hls_patches \\
        --output-dir runs/dino_pretrain \\
        --config configs/default.yaml
"""

from __future__ import annotations

import copy
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.TSViT.TSViTdense import TSViT
# datasets imports are deferred to DINOTrainer.__init__ to avoid the
# Windows DLL load issue with rioxarray when importing at module level.


# ---------------------------------------------------------------------------
# DINO projection head
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """MLP projection head + L2-normalised prototypes."""

    def __init__(self, in_dim: int, out_dim: int = 65536,
                 hidden_dim: int = 2048, n_layers: int = 3,
                 norm_last_layer: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                # LayerNorm instead of BatchNorm: works with batch size = 1
                # (necessary because crops are processed one at a time)
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ---------------------------------------------------------------------------
# DINO loss
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        n_crops: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.n_crops         = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.full(1000, teacher_temp),
        ])

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor, epoch: int):
        """
        student_out : (n_crops * B, out_dim)   all crops stacked
        teacher_out : (2 * B, out_dim)          global crops only
        """
        student_out = (student_out / self.student_temp).chunk(self.n_crops)

        t_temp      = float(self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)])
        teacher_out = F.softmax((teacher_out - self.center) / t_temp, dim=-1).detach().chunk(2)

        loss, n_pairs = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    continue
                loss    += torch.mean(torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1))
                n_pairs += 1

        loss /= n_pairs
        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out):
        batch_center = torch.cat(teacher_out, dim=0).mean(dim=0, keepdim=True)
        self.center  = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ---------------------------------------------------------------------------
# EMA + schedules
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)


def cosine_scheduler(base_val: float, final_val: float, epochs: int, warmup_epochs: int = 0):
    schedule = []
    for ep in range(epochs):
        if ep < warmup_epochs:
            val = base_val * ep / max(1, warmup_epochs)
        else:
            t   = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            val = final_val + 0.5 * (base_val - final_val) * (1 + math.cos(math.pi * t))
        schedule.append(val)
    return np.array(schedule)


# ---------------------------------------------------------------------------
# TSViT backbone wrapper: strips mlp_head, returns pooled feature vector
# ---------------------------------------------------------------------------

class TSViTBackbone(nn.Module):
    """TSViT up to space_transformer; returns (B, dim) via mean pooling."""

    def __init__(self, model_config: dict):
        super().__init__()
        self._full = TSViT(model_config)
        self.dim   = model_config['dim']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, H, W, C+1)  — last channel is DOY/365"""
        m = self._full
        x = x.permute(0, 1, 4, 2, 3)             # (B, T, C+1, H, W)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]                    # (B, T) DOY channel
        x  = x[:, :, :-1]                         # (B, T, C, H, W) spectral only

        xt = (xt * 365.0001).to(torch.int64).clamp(0, 365)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32).reshape(-1, 366)
        temporal_pos = m.to_temporal_embedding_input(xt).reshape(B, T, m.dim)

        x = m.to_patch_embedding(x)               # (B*P, T, dim)
        x = x.reshape(B, -1, T, m.dim)
        x += temporal_pos.unsqueeze(1)
        x = x.reshape(-1, T, m.dim)               # (B*P, T, dim)

        from einops import repeat
        cls = repeat(m.temporal_token, '() N d -> b N d', b=B * m.num_patches_1d ** 2)
        x   = torch.cat((cls, x), dim=1)
        x   = m.temporal_transformer(x)
        x   = x[:, :m.num_classes]                # (B*P, num_classes, dim)

        x = (
            x.reshape(B, m.num_patches_1d ** 2, m.num_classes, m.dim)
             .permute(0, 2, 1, 3)
             .reshape(B * m.num_classes, m.num_patches_1d ** 2, m.dim)
        )
        x += m.space_pos_embedding
        x  = m.dropout(x)
        x  = m.space_transformer(x)               # (B*num_classes, P, dim)

        x = x.reshape(B, m.num_classes, m.num_patches_1d ** 2, m.dim).mean(dim=[1, 2])
        return x   # (B, dim)


# ---------------------------------------------------------------------------
# Crop format adapter
# ---------------------------------------------------------------------------

def _to_tsviT_format(crops: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    crops : (batch, T, C+1, H, W)  from collate_data_and_cast
    returns: (batch, T, H, W, C+1)  expected by TSViTBackbone
    """
    return crops.permute(0, 1, 3, 4, 2).to(device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DINOTrainer:
    def __init__(
        self,
        model_config: dict,
        patch_dir: str,
        output_dir: str,
        n_months: int = 6,
        n_bands: int = 7,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 5e-4,
        weight_decay: float = 0.04,
        dino_out_dim: int = 65536,
        teacher_momentum: float = 0.996,
        n_local_crops: int = 6,
        num_workers: int = 0,
        device: Optional[str] = None,
        end_date: Optional[str] = None,
        # GPU options
        use_amp: bool = False,
        crop_batch_size: int = 1,
        resume_from: Optional[str] = None,
        log_path: Optional[str] = None,
        save_every: int = 10,
    ):
        import logging as _logging

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.save_every      = save_every
        self.device          = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp         = use_amp and (self.device.type == "cuda")
        self.crop_batch_size = crop_batch_size if crop_batch_size > 0 else 10 ** 9

        # ── Logging ────────────────────────────────────────────────────────
        handlers = [_logging.StreamHandler(sys.stdout)]
        if log_path:
            handlers.append(_logging.FileHandler(log_path, mode="a"))
        _logging.basicConfig(
            level=_logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,
        )
        self.log = _logging.getLogger(__name__)

        img_size = model_config['img_res']

        # ── Dataset ────────────────────────────────────────────────────────
        from datasets.utils import collate_data_and_cast
        from datasets.transforms.tensor_transforms import DataAugmentationDINOV2

        dino_transform = DataAugmentationDINOV2(
            global_crops_scale=[0.32, 1.0],
            local_crops_scale=[0.05, 0.32],
            local_crops_number=n_local_crops,
            global_crops_size=img_size,
            local_crops_size=img_size,
            std_range=(0.001, 0.005),
            n_bands=n_bands,
        )
        _dino_transform = dino_transform

        # Auto-detect: "hls_patch_*.nc" (from hls_download.py) → HLSPatchDataset.
        # Patches may live in tile sub-directories (tile_000/, tile_001/ …).
        import glob as _glob
        sample_files = (
            _glob.glob(os.path.join(patch_dir, "*.nc")) or
            _glob.glob(os.path.join(patch_dir, "*", "*.nc"))
        )
        recursive = not bool(_glob.glob(os.path.join(patch_dir, "*.nc")))
        use_hls_loader = bool(sample_files) and os.path.basename(
            sample_files[0]
        ).startswith("hls_patch_")

        from torch.utils.data import Dataset as _Dataset

        if use_hls_loader:
            from datasets.hls_loader import HLSPatchDataset
            _end_date = end_date or "2023-12-31"
            n_steps   = model_config.get('max_seq_len', 24)

            class _HLSPatchDinoDataset(_Dataset):
                def __init__(self, patch_dir, end_date, n_steps, img_size,
                             transform, recursive):
                    self._ds = HLSPatchDataset(
                        patch_dir, end_date=end_date, n_steps=n_steps,
                        img_size=img_size, recursive=recursive,
                    )
                    self._transform = transform

                def __len__(self):
                    return len(self._ds)

                def __getitem__(self, idx):
                    t = self._ds[idx].numpy()    # (T, H, W, C+1)
                    t = t.transpose(0, 3, 1, 2)  # (T, C+1, H, W)
                    return self._transform(t)

            dataset = _HLSPatchDinoDataset(
                patch_dir, _end_date, n_steps, img_size, _dino_transform, recursive
            )
            self.log.info(
                f"HLSPatchDataset  patches={len(dataset)}  "
                f"end_date={_end_date}  recursive={recursive}"
            )
        else:
            from datasets.agro_satdata import MltTileData

            class _DinoDataset(_Dataset, MltTileData):
                def __len__(self):
                    return max(0, len(self.all_combinations) - 2)

                def __getitem__(self, idx):
                    satdata = self.get_random_date_tile_as_image(idx)
                    return _dino_transform(satdata)

            dataset = _DinoDataset.__new__(_DinoDataset)
            _Dataset.__init__(dataset)
            MltTileData.__init__(dataset, path=patch_dir,
                                 months_season=n_months, pixel_size=img_size)
            self.log.info(f"MltTileData  combinations={len(dataset)}")

        n_global_crops = 2
        self.n_crops   = n_global_crops + n_local_crops

        collate = partial(
            collate_data_and_cast,
            mask_ratio_tuple=[0.1, 0.5],
            mask_probability=0.5,
            n_tokens=None,
            mask_generator=None,
            dtype=torch.float32,
        )

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        # ── Student / teacher ──────────────────────────────────────────────
        self.student_backbone = TSViTBackbone(model_config).to(self.device)
        self.teacher_backbone = copy.deepcopy(self.student_backbone).to(self.device)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        feat_dim = model_config['dim']
        self.student_head = DINOHead(feat_dim, dino_out_dim).to(self.device)
        self.teacher_head = DINOHead(feat_dim, dino_out_dim).to(self.device)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss(out_dim=dino_out_dim, n_crops=self.n_crops).to(self.device)

        params = (
            list(self.student_backbone.parameters()) +
            list(self.student_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scaler    = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.momentum_schedule = cosine_scheduler(teacher_momentum, 1.0, epochs, warmup_epochs=10)
        self.lr_schedule        = cosine_scheduler(lr, 1e-6, epochs, warmup_epochs=10)

        # ── Resume ────────────────────────────────────────────────────────
        self.start_epoch = 0
        if resume_from and os.path.isfile(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.student_backbone.load_state_dict(ckpt["backbone"])
            self.student_head.load_state_dict(ckpt["head"])
            if "teacher_backbone" in ckpt:
                self.teacher_backbone.load_state_dict(ckpt["teacher_backbone"])
            if "teacher_head" in ckpt:
                self.teacher_head.load_state_dict(ckpt["teacher_head"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and ckpt["scaler"] is not None and self.use_amp:
                self.scaler.load_state_dict(ckpt["scaler"])
            if "dino_center" in ckpt:
                self.dino_loss.center.copy_(ckpt["dino_center"])
            self.start_epoch = ckpt.get("epoch", 0)
            self.log.info(f"Resumed from {resume_from}  (epoch {self.start_epoch})")

    # ── Forward helpers ────────────────────────────────────────────────────

    def _forward_crops(self, crops: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
        """
        crops : (N, T, H, W, C+1)
        Process in chunks of crop_batch_size to balance memory vs. throughput.
        crop_batch_size=1  → one at a time (original CPU-safe behaviour)
        crop_batch_size=N  → all at once  (fastest on GPU)
        """
        feats = []
        N    = crops.shape[0]
        ctx  = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            for start in range(0, N, self.crop_batch_size):
                chunk = crops[start: start + self.crop_batch_size]
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    f = self.student_head(self.student_backbone(chunk))
                feats.append(f)
        return torch.cat(feats, dim=0)

    def _forward_teacher(self, crops: torch.Tensor) -> torch.Tensor:
        feats = []
        N = crops.shape[0]
        with torch.no_grad():
            for start in range(0, N, self.crop_batch_size):
                chunk = crops[start: start + self.crop_batch_size]
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    f = self.teacher_head(self.teacher_backbone(chunk))
                feats.append(f)
        return torch.cat(feats, dim=0)

    # ── Train loop ─────────────────────────────────────────────────────────

    def train(self):
        self.log.info(
            f"device={self.device}  amp={self.use_amp}  "
            f"batches/epoch={len(self.loader)}  "
            f"crop_batch={self.crop_batch_size}"
        )

        for epoch in range(self.start_epoch, self.epochs):
            self._set_lr(self.lr_schedule[epoch])
            momentum   = self.momentum_schedule[epoch]
            total_loss = 0.0

            for batch in self.loader:
                global_crops = _to_tsviT_format(batch["collated_global_crops"], self.device)
                local_crops  = _to_tsviT_format(batch["collated_local_crops"],  self.device)

                all_crops    = torch.cat([global_crops, local_crops], dim=0)
                student_feat = self._forward_crops(all_crops, no_grad=False)
                teacher_feat = self._forward_teacher(global_crops)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.dino_loss(student_feat, teacher_feat, epoch)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.student_backbone.parameters(), 3.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                update_teacher(self.student_backbone, self.teacher_backbone, momentum)
                update_teacher(self.student_head,     self.teacher_head,     momentum)
                total_loss += loss.item()

            avg = total_loss / max(1, len(self.loader))
            self.log.info(
                f"Epoch {epoch+1:4d}/{self.epochs}  loss={avg:.4f}  mom={momentum:.4f}"
            )

            if (epoch + 1) % self.save_every == 0:
                self._save(epoch + 1)

        self._save(self.epochs, final=True)

    def _save(self, epoch: int, final: bool = False):
        tag  = "final" if final else f"ep{epoch:04d}"
        path = self.output_dir / f"dino_{tag}.pth"
        torch.save({
            "epoch":            epoch,
            "backbone":         self.student_backbone.state_dict(),
            "head":             self.student_head.state_dict(),
            "teacher_backbone": self.teacher_backbone.state_dict(),
            "teacher_head":     self.teacher_head.state_dict(),
            "optimizer":        self.optimizer.state_dict(),
            "scaler":           self.scaler.state_dict() if self.use_amp else None,
            "dino_center":      self.dino_loss.center.clone(),
        }, path)
        self.log.info(f"Checkpoint → {path}")

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir",   required=True,  help="Directory with *patch_*.nc files")
    p.add_argument("--output-dir",  required=True,  help="Checkpoint output directory")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--n-months",    type=int,   default=6,
                   help="Temporal window length (months) for each training sample")
    p.add_argument("--n-bands",     type=int,   default=7,
                   help="Number of spectral bands (excl. DOY channel)")
    p.add_argument("--n-local-crops", type=int, default=6)
    p.add_argument("--num-workers", type=int,   default=0)
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dino_cfg = cfg["training"]["dino"]

    trainer = DINOTrainer(
        model_config=cfg["model"],
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        n_months=args.n_months,
        n_bands=args.n_bands,
        epochs=args.epochs      or dino_cfg["epochs"],
        batch_size=args.batch_size or dino_cfg["batch_size"],
        lr=dino_cfg["lr"],
        weight_decay=dino_cfg["weight_decay"],
        dino_out_dim=dino_cfg["out_dim"],
        teacher_momentum=dino_cfg["teacher_momentum"],
        n_local_crops=args.n_local_crops or dino_cfg["n_local_crops"],
        num_workers=args.num_workers,
        device=args.device,
    )
    trainer.train()
