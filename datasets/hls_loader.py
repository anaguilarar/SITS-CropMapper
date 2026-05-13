"""
HLS patch dataset for TSViT training and inference.

Reads NetCDF patch files produced by utils/hls_download.py.
Harmonizes observations to a fixed 14-day grid (same cadence as HLS design).
Returns tensors of shape (T, H, W, C) where the last channel is DOY/365.

Expected NetCDF variables: blue, green, red, nir, swir1
Optionally: ndvi, gndvi (computed on-the-fly if absent)
Date dimension: 'date' coordinate of dtype datetime64.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECTRAL_BANDS = ["blue", "green", "red", "nir", "swir1"]
ALL_BANDS      = ["blue", "green", "red", "nir", "swir1", "ndvi", "gndvi"]
N_BANDS        = 7       # spectral bands (no DOY)
N_CHANNELS     = 8       # 7 bands + 1 DOY channel

HLS_INTERVAL_DAYS = 14  # nominal HLS revisit after merging Landsat + S2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_vi(ds: xr.Dataset) -> xr.Dataset:
    """Add ndvi and gndvi to dataset if not already present."""
    nir   = ds["nir"]
    red   = ds["red"]
    green = ds["green"]
    if "ndvi" not in ds:
        ds["ndvi"]  = ((nir - red)   / (nir + red   + 1e-9)).clip(-1, 1).astype("float32")
    if "gndvi" not in ds:
        ds["gndvi"] = ((nir - green) / (nir + green + 1e-9)).clip(-1, 1).astype("float32")
    return ds


def _harmonize_to_grid(
    data: np.ndarray,        # (T_raw, C, H, W)
    dates: np.ndarray,       # (T_raw,) datetime64[D]
    end_date: np.datetime64,
    n_steps: int,
    interval: int = HLS_INTERVAL_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin raw observations into a regular backward-looking grid.

    Each grid cell covers [t - interval, t) relative to end_date.
    Cells with multiple observations are averaged; empty cells get zeros.

    Returns
    -------
    grid_data  : (n_steps, C, H, W)  float32
    grid_doys  : (n_steps,)          int  day-of-year 1..365
    """
    T_grid = n_steps
    grid_data = np.zeros((T_grid,) + data.shape[1:], dtype=np.float32)
    grid_counts = np.zeros(T_grid, dtype=np.int32)

    for t_raw, date in enumerate(dates):
        days_before = int((end_date - date) / np.timedelta64(1, 'D'))
        if days_before < 0 or days_before >= T_grid * interval:
            continue
        bin_idx = T_grid - 1 - (days_before // interval)
        if 0 <= bin_idx < T_grid:
            grid_data[bin_idx]  += np.nan_to_num(data[t_raw], nan=0.0)
            grid_counts[bin_idx] += 1

    nonzero = grid_counts > 0
    grid_data[nonzero] /= grid_counts[nonzero, None, None, None]

    # Compute DOY for each grid slot center
    grid_doys = np.zeros(T_grid, dtype=np.int32)
    for i in range(T_grid):
        slot_center = end_date - np.timedelta64(int((T_grid - 1 - i) * interval), 'D')
        doy = slot_center.astype('datetime64[D]').astype(object).timetuple().tm_yday
        grid_doys[i] = doy

    return grid_data, grid_doys


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HLSPatchDataset(Dataset):
    """
    PyTorch Dataset over a directory of HLS NetCDF patch files.

    Each patch is harmonized to a regular 14-day time grid and returned as a
    tensor of shape ``(T, H, W, C)`` where ``C = N_CHANNELS`` (7 spectral + 1 DOY).

    Parameters
    ----------
    patch_dir   : path to directory containing ``*patch_*.nc`` files
    end_date    : reference end date (ISO string or numpy datetime64)
    n_steps     : number of 14-day time steps to return
    img_size    : expected spatial size; patches are center-cropped/padded
    augment     : apply random horizontal flip during training
    return_path : if True, __getitem__ also returns the file path
    """

    def __init__(
        self,
        patch_dir: str,
        end_date: str = "2023-12-31",
        n_steps: int = 24,
        img_size: int = 48,
        augment: bool = False,
        return_path: bool = False,
        recursive: bool = False,
    ):
        self.patch_dir  = Path(patch_dir)
        self.end_date   = np.datetime64(end_date, 'D')
        self.n_steps    = n_steps
        self.img_size   = img_size
        self.augment    = augment
        self.return_path = return_path

        scan = self.patch_dir.rglob if recursive else self.patch_dir.glob
        self.files = sorted(scan("*patch_*.nc"))
        if not self.files:
            raise FileNotFoundError(f"No patch NetCDF files found in {patch_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        # Open without auto-scaling (values already in [0,1] after download pipeline)
        ds = xr.open_dataset(str(path), mask_and_scale=False)
        ds = _compute_vi(ds)

        dates = ds["date"].values.astype("datetime64[D]")

        # Stack bands into (T, C, H, W)
        arrays = []
        for band in ALL_BANDS:
            arr = ds[band].values  # (T, H, W) or (H, W) for single-date
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            arrays.append(arr)
        data = np.stack(arrays, axis=1).astype(np.float32)  # (T, C, H, W)

        # Harmonize to regular grid
        grid_data, grid_doys = _harmonize_to_grid(
            data, dates, self.end_date, self.n_steps
        )

        # Spatial crop / pad to img_size
        _, C, H, W = grid_data.shape
        if H != self.img_size or W != self.img_size:
            grid_data = _resize_spatial(grid_data, self.img_size)

        # Build DOY channel: (T, 1, H, W) broadcast over spatial dims
        doy_channel = np.zeros((self.n_steps, 1, self.img_size, self.img_size), dtype=np.float32)
        for t in range(self.n_steps):
            doy_channel[t, 0] = grid_doys[t] / 365.0

        # Concatenate: (T, C+1, H, W)
        full = np.concatenate([grid_data, doy_channel], axis=1)

        # Reorder to (T, H, W, C) for TSViT
        full = full.transpose(0, 2, 3, 1)  # (T, H, W, C)

        tensor = torch.from_numpy(full)

        if self.augment and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[2])   # random horizontal flip

        if self.return_path:
            return tensor, str(path)
        return tensor


def _resize_spatial(data: np.ndarray, target: int) -> np.ndarray:
    """Center-crop or zero-pad (T, C, H, W) to (T, C, target, target)."""
    T, C, H, W = data.shape
    out = np.zeros((T, C, target, target), dtype=data.dtype)

    # crop
    h_start = max(0, (H - target) // 2)
    w_start = max(0, (W - target) // 2)
    h_end   = h_start + min(H, target)
    w_end   = w_start + min(W, target)
    src_h   = min(H, target)
    src_w   = min(W, target)

    out[:, :, :src_h, :src_w] = data[:, :, h_start:h_end, w_start:w_end]
    return out


# ---------------------------------------------------------------------------
# Collate with variable-length masking (for inference without labels)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return torch.stack(batch, dim=0)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    patch_dir = sys.argv[1] if len(sys.argv) > 1 else "data/hls_patches"
    ds = HLSPatchDataset(patch_dir, end_date="2023-12-31", n_steps=24)
    print(f"Found {len(ds)} patches")
    sample = ds[0]
    print(f"Sample shape: {sample.shape}")   # (24, 48, 48, 8)
    print(f"DOY range: {sample[:, 0, 0, -1].min():.3f} to {sample[:, 0, 0, -1].max():.3f}")
    print(f"NIR range: {sample[:, 0, 0, 3].min():.4f} to {sample[:, 0, 0, 3].max():.4f}")
