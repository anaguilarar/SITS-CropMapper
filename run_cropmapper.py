"""
SITS-CropMapper — main entry point.

Three modes:
  download   Download HLS patches for a bounding box and date range
  pretrain   DINO self-supervised pretraining on unlabelled patches
  finetune   Supervised fine-tuning on labelled patches
  infer      Run inference (LC segmentation) on a patch directory

Examples
--------
# 1. Download HLS data for Honduras (Dec 2022 - Dec 2023)
python run_cropmapper.py download \\
    --bbox -87.5 13.5 -87.0 14.0 \\
    --start 2022-12-01 --end 2023-12-31 \\
    --output data/hnd_patches

# 2. DINO pretraining on downloaded patches (no labels needed)
python run_cropmapper.py pretrain \\
    --patch-dir data/hnd_patches \\
    --output-dir runs/dino \\
    --epochs 100

# 3. Supervised fine-tuning with labelled patches
python run_cropmapper.py finetune \\
    --patch-dir data/hnd_labelled \\
    --output-dir runs/finetune \\
    --backbone runs/dino/dino_backbone_final.pth \\
    --epochs 50

# 4. Inference
python run_cropmapper.py infer \\
    --patch-dir data/hnd_patches \\
    --weights runs/finetune/tsviT_seg_ep0050_final.pth \\
    --output-dir results/hnd \\
    --end-date 2023-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def cmd_download(args):
    from utils.hls_download import download_hls
    download_hls(
        bbox=tuple(args.bbox),
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        patch_size=args.patch_size,
        strategy=args.strategy,
        stream=not args.no_stream,
        local_dir=args.local_dir,
    )


# ---------------------------------------------------------------------------
# DINO pretrain
# ---------------------------------------------------------------------------

def cmd_pretrain(args):
    from training.dino_pretrain import DINOTrainer
    cfg  = load_config(args.config)
    dino = cfg["training"]["dino"]
    trainer = DINOTrainer(
        model_config=cfg["model"],
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        n_months=args.n_months,
        n_bands=args.n_bands,
        epochs=args.epochs      or dino["epochs"],
        batch_size=args.batch_size or dino["batch_size"],
        lr=dino["lr"],
        weight_decay=dino["weight_decay"],
        dino_out_dim=dino["out_dim"],
        teacher_momentum=dino["teacher_momentum"],
        n_local_crops=dino["n_local_crops"],
        num_workers=args.num_workers,
        device=args.device,
    )
    trainer.train()


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------

def cmd_finetune(args):
    from training.finetune_segmentation import SegmentationTrainer
    cfg = load_config(args.config)
    ft  = cfg["training"]["finetune"]
    trainer = SegmentationTrainer(
        model_config=cfg["model"],
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        backbone_path=args.backbone,
        end_date=args.end_date,
        n_steps=cfg["data"]["n_steps"],
        epochs=args.epochs or ft["epochs"],
        batch_size=args.batch_size or ft["batch_size"],
        lr=ft["lr"],
        lr_backbone=ft["lr_backbone"],
        val_split=ft["val_split"],
        device=args.device,
    )
    trainer.train()
    iou = trainer.compute_iou(trainer.val_loader)
    print("\nValidation IoU:")
    for k, v in iou.items():
        print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def cmd_infer(args):
    import rioxarray
    import xarray as xr
    from torch.utils.data import DataLoader
    from datasets.hls_loader import HLSPatchDataset
    from models.TSViT.TSViTdense import TSViT

    cfg   = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model = TSViT(cfg["model"]).to(device)
    ckpt  = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {args.weights}")

    ds = HLSPatchDataset(
        args.patch_dir,
        end_date=args.end_date,
        n_steps=cfg["data"]["n_steps"],
        img_size=cfg["data"]["img_size"],
        augment=False,
        return_path=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size or 4,
                        shuffle=False, num_workers=2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    all_paths = []

    with torch.no_grad():
        for batch_imgs, batch_paths in loader:
            batch_imgs = batch_imgs.to(device)
            logits     = model(batch_imgs)              # (B, num_classes, H, W)
            preds      = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)
            all_preds.append(preds)
            all_paths.extend(batch_paths)

    all_preds = np.concatenate(all_preds, axis=0)   # (N, H, W)

    # Save one GeoTIFF per patch using the source NetCDF CRS
    n_saved = 0
    for i, (pred, src_path) in enumerate(zip(all_preds, all_paths)):
        try:
            src = rioxarray.open_rasterio(src_path, masked=True)
            out_da = xr.DataArray(
                pred[np.newaxis],           # (1, H, W)
                dims=["band", "y", "x"],
                attrs={"long_name": "LC class"},
            )
            if hasattr(src, "rio") and src.rio.crs is not None:
                out_da = out_da.rio.write_crs(src.rio.crs)
                out_da = out_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

            out_path = out_dir / (Path(src_path).stem + "_lc.tif")
            out_da.rio.to_raster(str(out_path), dtype="uint8")
            n_saved += 1
        except Exception as exc:
            print(f"  Warning: could not save patch {i}: {exc}")

    print(f"Inference complete — {n_saved} LC patches saved to {out_dir}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="run_cropmapper",
        description="SITS-CropMapper: HLS satellite time-series crop segmentation",
    )
    root.add_argument("--config", default="configs/default.yaml",
                      help="Path to YAML config (default: configs/default.yaml)")

    sub = root.add_subparsers(dest="command", required=True)

    # ── download ────────────────────────────────────────────────────────────
    dl = sub.add_parser("download", help="Download HLS patches from NASA Earthdata")
    dl.add_argument("--bbox",       nargs=4, type=float, required=True,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    dl.add_argument("--start",      required=True, help="Start date YYYY-MM-DD")
    dl.add_argument("--end",        required=True, help="End date YYYY-MM-DD")
    dl.add_argument("--output",     required=True, help="Output directory for patches")
    dl.add_argument("--patch-size", type=int, default=48)
    dl.add_argument("--no-stream",  action="store_true",
                    help="Download files locally instead of S3 streaming")
    dl.add_argument("--local-dir",  default=None)
    dl.add_argument("--strategy",   default="netrc",
                    choices=["netrc", "environment", "prompt"])

    # ── pretrain ────────────────────────────────────────────────────────────
    pt = sub.add_parser("pretrain", help="DINO self-supervised pretraining")
    pt.add_argument("--patch-dir",    required=True)
    pt.add_argument("--output-dir",   required=True)
    pt.add_argument("--n-months",     type=int, default=6,
                    help="Temporal window (months) per training sample")
    pt.add_argument("--n-bands",      type=int, default=7,
                    help="Number of spectral bands (excl. DOY)")
    pt.add_argument("--num-workers",  type=int, default=0)
    pt.add_argument("--epochs",       type=int, default=None)
    pt.add_argument("--batch-size",   type=int, default=None)
    pt.add_argument("--device",       default=None)

    # ── finetune ────────────────────────────────────────────────────────────
    ft = sub.add_parser("finetune", help="Supervised fine-tuning on labelled patches")
    ft.add_argument("--patch-dir",  required=True)
    ft.add_argument("--output-dir", required=True)
    ft.add_argument("--backbone",   default=None,
                    help="Path to DINO pretrained backbone .pth")
    ft.add_argument("--end-date",   default="2023-12-31")
    ft.add_argument("--epochs",     type=int,   default=None)
    ft.add_argument("--batch-size", type=int,   default=None)
    ft.add_argument("--device",     default=None)

    # ── infer ───────────────────────────────────────────────────────────────
    inf = sub.add_parser("infer", help="Run LC segmentation inference")
    inf.add_argument("--patch-dir",  required=True)
    inf.add_argument("--weights",    required=True, help="Path to fine-tuned .pth")
    inf.add_argument("--output-dir", required=True)
    inf.add_argument("--end-date",   default="2023-12-31")
    inf.add_argument("--batch-size", type=int, default=4)
    inf.add_argument("--device",     default=None)

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    # Ensure local packages are importable when running from project root
    sys.path.insert(0, str(Path(__file__).parent))

    dispatch = {
        "download": cmd_download,
        "pretrain": cmd_pretrain,
        "finetune": cmd_finetune,
        "infer":    cmd_infer,
    }
    dispatch[args.command](args)
