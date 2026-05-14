"""
HLS (Harmonized Landsat Sentinel-2) download pipeline.

Authenticates with NASA Earthdata via earthaccess, searches HLSS30 + HLSL30
granules for a bounding box and date range, applies cloud masking, computes
vegetation indices, tiles the data into patch_size x patch_size NetCDF files,
and saves them ready for HLSPatchDataset.

NASA Earthdata account required. Credentials via:
    ~/.netrc  OR  env vars EARTHDATA_USERNAME / EARTHDATA_PASSWORD

Usage (CLI):
    python -m utils.hls_download \\
        --bbox -87.5 13.5 -87.0 14.0 \\
        --start 2023-01-01 --end 2023-12-31 \\
        --output data/hls_patches
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Band mapping: logical name -> HLS asset suffix
# ---------------------------------------------------------------------------

BAND_MAP: Dict[str, Dict[str, str]] = {
    "HLSS30": {
        "blue": "B02", "green": "B03", "red": "B04",
        "nir": "B8A", "swir1": "B11", "fmask": "Fmask",
    },
    "HLSL30": {
        "blue": "B02", "green": "B03", "red": "B04",
        "nir": "B05", "swir1": "B06", "fmask": "Fmask",
    },
}

# Fmask bits: cloud=3, cloud shadow=4, adjacent cloud=2, snow/ice=5, water=1 (optional)
_BAD_BITS: Tuple[int, ...] = (1, 2, 3, 4)
_HLS_SCALE = 0.0001
MIN_VALID_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login(strategy: str = "netrc") -> None:
    try:
        import earthaccess
    except ImportError:
        raise ImportError("pip install earthaccess")
    earthaccess.login(strategy=strategy, persist=True)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_granules(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    collections: Tuple[str, ...] = ("HLSS30", "HLSL30"),
    max_results: int = 2000,
) -> List:
    import earthaccess
    granules = []
    for short_name in collections:
        found = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox,
            temporal=(start_date, end_date),
            count=max_results,
        )
        granules.extend(found)
        print(f"  [{short_name}] {len(found)} granules")
    print(f"Total: {len(granules)} granules")
    return granules


# ---------------------------------------------------------------------------
# Per-granule processing
# ---------------------------------------------------------------------------

def _product_name(granule) -> str:
    title = granule["umm"]["GranuleUR"]
    return "HLSS30" if "HLSS30" in title else "HLSL30"


def _build_fmask(fmask_array: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(fmask_array, dtype=bool)
    for bit in _BAD_BITS:
        mask |= (fmask_array & (1 << bit)).astype(bool)
    return mask


def _open_band(file_obj, bbox, crs="EPSG:4326"):
    import rioxarray as rxr
    da = rxr.open_rasterio(file_obj, masked=True).squeeze("band", drop=True)
    da = da.rio.reproject(crs)
    if bbox is not None:
        da = da.rio.clip_box(*bbox)
    return da


def _parse_date(granule_ur: str) -> str:
    for part in granule_ur.split("."):
        if len(part) == 14 and part[:4].isdigit() and part[7] == "T":
            year, doy = int(part[:4]), int(part[4:7])
            return (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
    raise ValueError(f"Cannot parse date from: {granule_ur}")


def process_granule(
    granule,
    bbox=None,
    crs: str = "EPSG:4326",
    stream: bool = True,
    local_dir: Optional[str] = None,
):
    import earthaccess
    import xarray as xr

    product  = _product_name(granule)
    band_map = BAND_MAP[product]
    band_files: Dict[str, object] = {}

    try:
        if stream:
            all_links = granule.data_links()
            for logical, suffix in band_map.items():
                matched = [l for l in all_links
                           if f".{suffix}." in l.split("/")[-1]
                           or l.split("/")[-1].endswith(f".{suffix}.tif")]
                if matched:
                    opened = earthaccess.open([matched[0]])
                    if opened:
                        band_files[logical] = opened[0]
        else:
            if local_dir is None:
                raise ValueError("local_dir required when stream=False")
            paths = earthaccess.download([granule], local_path=local_dir)
            for path in paths:
                name = Path(path).name
                for logical, suffix in band_map.items():
                    if f".{suffix}." in name:
                        band_files[logical] = path

        required = {"blue", "green", "red", "nir", "swir1", "fmask"}
        if required - set(band_files):
            warnings.warn(f"Skipping granule — missing bands")
            return None

        arrays: Dict[str, object] = {}
        for band in ("blue", "green", "red", "nir", "swir1"):
            arrays[band] = _open_band(band_files[band], bbox, crs).astype("float32")

        fmask_da  = _open_band(band_files["fmask"], bbox, crs)
        bad_px    = _build_fmask(fmask_da.values.astype(np.int16))

        for band, da in arrays.items():
            scaled = da * _HLS_SCALE
            scaled = scaled.where(~bad_px).where((scaled >= 0) & (scaled <= 1))
            scaled.attrs.pop("scale_factor", None)
            scaled.attrs.pop("add_offset",   None)
            arrays[band] = scaled.astype("float32")

        valid_frac = float(np.mean(~np.isnan(arrays["red"].values)))
        if valid_frac < MIN_VALID_FRACTION:
            return None

        nir, red, green = arrays["nir"], arrays["red"], arrays["green"]
        arrays["ndvi"]  = ((nir - red)   / (nir + red   + 1e-9)).clip(-1, 1).astype("float32")
        arrays["gndvi"] = ((nir - green) / (nir + green + 1e-9)).clip(-1, 1).astype("float32")

        date_str = _parse_date(granule["umm"]["GranuleUR"])
        ds = xr.Dataset(arrays)
        ds = ds.assign_coords(date=np.datetime64(date_str, "D")).expand_dims("date")
        return ds

    except Exception as exc:
        warnings.warn(f"Granule failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Patch saving
# ---------------------------------------------------------------------------

def _save_patches(ds, output_path: str, patch_size: int) -> int:
    import dask
    from dask.diagnostics import ProgressBar

    ny, nx = ds.sizes["y"], ds.sizes["x"]
    tasks, idx = [], 0

    for r in range(0, ny, patch_size):
        for c in range(0, nx, patch_size):
            patch = ds.isel(y=slice(r, min(r+patch_size, ny)),
                            x=slice(c, min(c+patch_size, nx)))
            pad_y = patch_size - patch.sizes["y"]
            pad_x = patch_size - patch.sizes["x"]
            if pad_y or pad_x:
                patch = patch.pad(y=(0, pad_y), x=(0, pad_x), constant_values=0)

            # Write CRS so QGIS and rioxarray readers can locate the patch correctly.
            # set_spatial_dims is required after isel/pad since rio may lose dim context.
            try:
                patch = patch.rio.set_spatial_dims(x_dim="x", y_dim="y")
                patch = patch.rio.write_crs("EPSG:4326", grid_mapping_name="spatial_ref")
                # write_crs may not stamp grid_mapping on existing vars in all rioxarray
                # versions — add it explicitly so CF-compliant readers (QGIS, gdal) see it.
                for var in list(patch.data_vars):
                    if var != "spatial_ref":
                        patch[var].attrs["grid_mapping"] = "spatial_ref"
            except Exception as exc:
                warnings.warn(f"CRS write failed for patch {idx}: {exc}")

            fname    = os.path.join(output_path, f"hls_patch_{idx:05d}.nc")
            encoding = {v: {"zlib": True, "complevel": 4} for v in patch.data_vars
                        if v != "spatial_ref"}
            # spatial_ref is a scalar CRS variable — include uncompressed so dask writes it
            if "spatial_ref" in patch.coords:
                encoding["spatial_ref"] = {}
            if "spatial_ref" in patch.data_vars:
                encoding["spatial_ref"] = {}
            tasks.append(patch.to_netcdf(fname, encoding=encoding, compute=False))
            idx += 1

    print(f"Writing {len(tasks)} patches ...")
    with ProgressBar():
        dask.compute(*tasks)
    return idx


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_time_series(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    output_path: str,
    patch_size: int = 48,
    collections: Tuple[str, ...] = ("HLSS30", "HLSL30"),
    crs: str = "EPSG:4326",
    stream: bool = True,
    local_dir: Optional[str] = None,
    n_workers: int = 8,
) -> str:
    import xarray as xr

    os.makedirs(output_path, exist_ok=True)
    granules = search_granules(bbox, start_date, end_date, collections)
    if not granules:
        raise RuntimeError("No HLS granules found.")

    n_workers = min(n_workers, len(granules))
    print(f"Processing {len(granules)} granules with {n_workers} workers ...")
    scenes: List[xr.Dataset] = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(process_granule, g, bbox, crs, stream, local_dir): g
            for g in granules
        }
        from tqdm import tqdm
        with tqdm(total=len(granules)) as pbar:
            for fut in as_completed(futures):
                ds = fut.result()
                if ds is not None:
                    scenes.append(ds)
                pbar.update(1)

    if not scenes:
        raise RuntimeError("No valid scenes after cloud filtering.")

    print(f"{len(scenes)} scenes. Aligning and merging ...")
    ref = scenes[0]
    aligned = []
    for s in scenes:
        if s.x.size != ref.x.size or not np.allclose(s.x.values, ref.x.values, atol=1e-6):
            s = s.interp_like(ref, method="nearest")
        aligned.append(s)

    combined = xr.concat(aligned, dim="date").sortby("date")
    _, unique_idx = np.unique(combined.date.values, return_index=True)
    combined = combined.isel(date=unique_idx)

    print(f"Tiling into {patch_size}x{patch_size} patches ...")
    n_patches = _save_patches(combined, output_path, patch_size)
    print(f"Done — {n_patches} patches saved to {output_path}")
    return output_path


def download_hls(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    output_path: str,
    patch_size: int = 48,
    strategy: str = "netrc",
    stream: bool = True,
    local_dir: Optional[str] = None,
) -> str:
    login(strategy=strategy)
    return build_time_series(bbox, start_date, end_date, output_path,
                             patch_size=patch_size, stream=stream, local_dir=local_dir)


def repair_crs(patch_dir: str) -> int:
    """
    Add missing grid_mapping / spatial_ref CRS metadata to existing patches.

    Patches downloaded before the CRS fix lack the grid_mapping attribute on
    their data variables, so QGIS cannot place them correctly.  This function
    rewrites each patch in-place with the correct CF metadata.

    Returns the number of patches repaired.
    """
    import netCDF4 as nc4  # use netCDF4 directly to avoid xarray LRU file locking
    import xarray as xr

    files = list(Path(patch_dir).rglob("*patch_*.nc"))
    if not files:
        print(f"No patch files found under {patch_dir}")
        return 0

    repaired = 0
    for fpath in files:
        # Read metadata and data via netCDF4 directly so the file is closed immediately
        try:
            with nc4.Dataset(str(fpath), "r") as _nc:
                needs_fix = any(
                    not hasattr(_nc.variables.get(v, None), "grid_mapping")
                    for v in _nc.variables
                    if v not in ("x", "y", "date", "spatial_ref")
                )
                if not needs_fix:
                    continue
        except Exception as exc:
            warnings.warn(f"Could not inspect {fpath.name}: {exc}")
            continue

        # Re-open via xarray, load all into memory, then let xarray close the handle
        try:
            import rioxarray  # noqa: F401
            ds = xr.open_dataset(str(fpath), mask_and_scale=False, engine="netcdf4")
            ds.load()
            ds = ds.copy(deep=True)
            ds.close()
            # Force xarray backend cache to release the file handle on Windows
            try:
                from xarray.backends.file_manager import FILE_CACHE
                FILE_CACHE.clear()
            except Exception:
                pass

            ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            ds = ds.rio.write_crs("EPSG:4326", grid_mapping_name="spatial_ref")
            for var in list(ds.data_vars):
                if var != "spatial_ref":
                    ds[var].attrs["grid_mapping"] = "spatial_ref"

            encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars
                        if v != "spatial_ref"}
            if "spatial_ref" in ds.coords or "spatial_ref" in ds.data_vars:
                encoding["spatial_ref"] = {}

            tmp = fpath.with_suffix(".tmp.nc")
            ds.to_netcdf(str(tmp), encoding=encoding)
            # os.replace is atomic: either succeeds or leaves original untouched
            os.replace(str(tmp), str(fpath))
            repaired += 1
        except Exception as exc:
            warnings.warn(f"Could not repair {fpath.name}: {exc}")

    print(f"Repaired {repaired}/{len(files)} patches in {patch_dir}")
    return repaired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bbox",   nargs=4, type=float, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--start",  required=True, help="YYYY-MM-DD")
    p.add_argument("--end",    required=True, help="YYYY-MM-DD")
    p.add_argument("--output", required=True, help="Output directory for patches")
    p.add_argument("--patch-size", type=int, default=48)
    p.add_argument("--no-stream", action="store_true",
                   help="Download files locally instead of streaming from S3")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--strategy", default="netrc",
                   choices=["netrc", "environment", "prompt"])
    args = p.parse_args()

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
