"""
Honduras HLS download + DINO pretraining pipeline
==================================================
Windows GPU server entry point.

Randomly samples N tiles across Honduras, downloads HLS time-series patches
for each tile, then runs DINO self-supervised pretraining on the full collection.

Stages
------
  download   Download HLS patches for each sampled tile
  pretrain   DINO pretraining on all patches under patch-dir
  all        download then pretrain  (default)

Quick-start
-----------
  # Full pipeline, 50 random tiles, GPU auto-detected
  python run_hnd_pretrain.py

  # Download only (50 tiles)
  python run_hnd_pretrain.py --stage download --n-tiles 50

  # Pretrain only, resuming from last checkpoint
  python run_hnd_pretrain.py --stage pretrain --resume

  # Custom config + explicit paths
  python run_hnd_pretrain.py --config configs/hnd_pretrain.yaml ^
      --patch-dir D:/data/hnd_patches ^
      --output-dir D:/runs/hnd_dino ^
      --epochs 200 --batch-size 16 --workers 4

NASA Earthdata credentials
--------------------------
  Credentials must be available before the download stage.
  Recommended: create %USERPROFILE%\\.netrc with::

    machine urs.earthdata.nasa.gov
    login   YOUR_USERNAME
    password YOUR_PASSWORD

  Alternatively pass --earthdata-strategy environment and set:
    EARTHDATA_USERNAME, EARTHDATA_PASSWORD environment variables.

IMPORTANT — Windows multiprocessing
-------------------------------------
  DataLoader workers on Windows require the if __name__ == '__main__' guard,
  which is already present at the bottom of this file.  Always launch via:
      python run_hnd_pretrain.py ...
  NOT via python -c "import run_hnd_pretrain; ..."
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Honduras geographic grid helpers
# ---------------------------------------------------------------------------

# Approximate bounding box covering all of Honduras
HND_BBOX: Tuple[float, float, float, float] = (-89.35, 13.00, -83.15, 16.52)


def sample_tiles(
    n: int,
    tile_deg: float,
    country_bbox: Tuple[float, float, float, float] = HND_BBOX,
    seed: int = 42,
) -> List[Tuple[float, float, float, float]]:
    """
    Divide country_bbox into a regular grid of (tile_deg × tile_deg) cells
    and return a random sample of N cells without replacement.

    Returns list of (min_lon, min_lat, max_lon, max_lat) tuples.
    """
    min_lon, min_lat, max_lon, max_lat = country_bbox
    n_cols = math.floor((max_lon - min_lon) / tile_deg)
    n_rows = math.floor((max_lat - min_lat) / tile_deg)
    grid = [
        (
            min_lon + c * tile_deg,
            min_lat + r * tile_deg,
            min_lon + (c + 1) * tile_deg,
            min_lat + (r + 1) * tile_deg,
        )
        for r in range(n_rows)
        for c in range(n_cols)
    ]
    rng = random.Random(seed)
    return rng.sample(grid, min(n, len(grid)))


# ---------------------------------------------------------------------------
# Download stage
# ---------------------------------------------------------------------------

def stage_download(args, cfg: dict, log: logging.Logger) -> int:
    """Download HLS patches for each sampled tile. Returns total patch count."""
    from utils.hls_download import download_hls

    dl_cfg   = cfg.get("download", {})
    n_tiles  = args.n_tiles  or dl_cfg.get("n_tiles",    50)
    tile_deg = args.tile_deg or dl_cfg.get("tile_deg",   0.25)
    start    = args.start_date or dl_cfg.get("start_date", "2021-01-01")
    end      = args.end_date   or dl_cfg.get("end_date",   "2023-12-31")
    strategy = args.earthdata_strategy or dl_cfg.get("earthdata_strategy", "netrc")
    bbox_cfg = dl_cfg.get("country_bbox", list(HND_BBOX))
    country_bbox = tuple(bbox_cfg)

    patch_size = cfg["data"]["patch_size"]
    patch_dir  = Path(args.patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    tiles = sample_tiles(n_tiles, tile_deg, country_bbox=country_bbox, seed=args.seed)
    log.info(
        f"Sampled {len(tiles)} tiles  tile_deg={tile_deg}°  "
        f"dates={start} → {end}"
    )

    success = skipped = failed = 0
    for i, bbox in enumerate(tiles):
        tile_dir = patch_dir / f"tile_{i:03d}"

        # Skip tiles already downloaded (idempotent re-runs)
        existing = list(tile_dir.glob("*.nc")) if tile_dir.exists() else []
        if existing:
            log.info(f"[{i+1:3d}/{len(tiles)}] tile_{i:03d}  already done"
                     f"  ({len(existing)} patches)  — skip")
            skipped += 1
            continue

        tile_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"[{i+1:3d}/{len(tiles)}] tile_{i:03d}"
            f"  bbox=({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f})"
        )
        try:
            download_hls(
                bbox=bbox,
                start_date=start,
                end_date=end,
                output_path=str(tile_dir),
                patch_size=patch_size,
                strategy=strategy,
                stream=True,
            )
            n_saved = len(list(tile_dir.glob("*.nc")))
            log.info(f"           → {n_saved} patches")
            success += 1
        except Exception as exc:
            log.warning(f"           → FAILED: {exc}")
            failed += 1

    total = sum(len(list((patch_dir / f"tile_{i:03d}").glob("*.nc")))
                for i in range(len(tiles))
                if (patch_dir / f"tile_{i:03d}").exists())
    log.info(
        f"Download complete  success={success}  skipped={skipped}  failed={failed}"
        f"  total_patches={total}"
    )
    return total


# ---------------------------------------------------------------------------
# Pretrain stage
# ---------------------------------------------------------------------------

def stage_pretrain(args, cfg: dict, log: logging.Logger) -> None:
    """Run DINO self-supervised pretraining on downloaded patches."""
    import torch
    from training.dino_pretrain import DINOTrainer

    dino_cfg   = cfg["training"]["dino"]
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp    = (device_str == "cuda")

    patch_dir = Path(args.patch_dir)
    all_nc    = list(patch_dir.rglob("*patch_*.nc"))
    if not all_nc:
        log.error(
            f"No patches found under {patch_dir}.\n"
            f"Run  python run_hnd_pretrain.py --stage download  first."
        )
        sys.exit(1)
    log.info(f"Found {len(all_nc)} patches across all tiles in {patch_dir}")

    epochs     = args.epochs     or dino_cfg["epochs"]
    batch_size = args.batch_size or dino_cfg["batch_size"]
    workers    = args.workers

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path   = str(output_dir / "train.log")

    # Auto-resume: find latest checkpoint in output_dir if --resume flag set
    resume_from = args.resume_from
    if args.resume and resume_from is None:
        ckpts = sorted(output_dir.glob("dino_ep*.pth"))
        if ckpts:
            resume_from = str(ckpts[-1])
            log.info(f"Auto-resume from {resume_from}")
        else:
            log.info("No checkpoint found — starting from scratch")

    end_date = args.end_date or cfg.get("download", {}).get("end_date", "2023-12-31")

    log.info(
        f"DINO pretrain  device={device_str}  amp={use_amp}  "
        f"epochs={epochs}  batch={batch_size}  workers={workers}  "
        f"crop_batch={dino_cfg.get('crop_batch_size', 1)}"
    )

    trainer = DINOTrainer(
        model_config     = cfg["model"],
        patch_dir        = str(patch_dir),
        output_dir       = str(output_dir),
        n_months         = dino_cfg.get("n_months", 6),
        n_bands          = cfg["data"].get("n_bands", 7),
        epochs           = epochs,
        batch_size       = batch_size,
        lr               = dino_cfg["lr"],
        weight_decay     = dino_cfg["weight_decay"],
        dino_out_dim     = dino_cfg["out_dim"],
        teacher_momentum = dino_cfg["teacher_momentum"],
        n_local_crops    = dino_cfg["n_local_crops"],
        num_workers      = workers,
        device           = device_str,
        end_date         = end_date,
        use_amp          = use_amp,
        crop_batch_size  = dino_cfg.get("crop_batch_size", 1),
        resume_from      = resume_from,
        log_path         = log_path,
        save_every       = dino_cfg.get("save_every", 10),
    )
    trainer.train()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_hnd_pretrain",
        description="Honduras HLS download + DINO pretraining (Windows GPU server)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config", default="configs/hnd_pretrain.yaml",
        help="YAML config file",
    )
    p.add_argument(
        "--stage", default="all", choices=["download", "pretrain", "all"],
        help="Which stage(s) to run",
    )
    p.add_argument(
        "--patch-dir", default="data/hnd_patches",
        help="Root directory for patches. Each tile saved to patch-dir/tile_NNN/",
    )
    p.add_argument(
        "--output-dir", default="runs/hnd_pretrain",
        help="DINO checkpoint and log output directory",
    )

    # ── Download ────────────────────────────────────────────────────────────
    dl = p.add_argument_group("download")
    dl.add_argument("--n-tiles",    type=int,   default=None,
                    help="Number of random Honduras tiles (overrides config)")
    dl.add_argument("--tile-deg",   type=float, default=None,
                    help="Tile side length in degrees (overrides config)")
    dl.add_argument("--start-date", default=None, help="HLS start date YYYY-MM-DD")
    dl.add_argument("--end-date",   default=None, help="HLS end date YYYY-MM-DD")
    dl.add_argument("--earthdata-strategy", default=None,
                    choices=["netrc", "environment", "prompt"])
    dl.add_argument("--seed", type=int, default=42, help="Tile sampling random seed")

    # ── Pretrain ────────────────────────────────────────────────────────────
    pt = p.add_argument_group("pretrain")
    pt.add_argument("--epochs",     type=int,   default=None,
                    help="Training epochs (overrides config)")
    pt.add_argument("--batch-size", type=int,   default=None,
                    help="Batch size (overrides config)")
    pt.add_argument("--workers",    type=int,   default=4,
                    help="DataLoader worker processes")
    pt.add_argument("--device",     default=None,
                    help="cuda | cpu  (auto-detected when omitted)")
    pt.add_argument("--resume",     action="store_true",
                    help="Auto-resume from latest checkpoint in output-dir")
    pt.add_argument("--resume-from", default=None, metavar="PATH",
                    help="Explicit .pth checkpoint to resume from")

    return p


# ---------------------------------------------------------------------------
# Entry point — MUST be under if __name__ == '__main__' on Windows
# (DataLoader spawn multiprocessing requires it)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure project root is on sys.path when launched from any directory
    _root = Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    parser = build_parser()
    args   = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # ── Root logger (console; pretrain stage also adds a file handler) ──────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("hnd_pretrain")

    log.info(f"Config : {args.config}")
    log.info(f"Stage  : {args.stage}")
    log.info(f"Patches: {args.patch_dir}")
    log.info(f"Output : {args.output_dir}")

    # ── Run stages ──────────────────────────────────────────────────────────
    if args.stage in ("download", "all"):
        stage_download(args, cfg, log)

    if args.stage in ("pretrain", "all"):
        stage_pretrain(args, cfg, log)

    log.info("Pipeline finished.")
