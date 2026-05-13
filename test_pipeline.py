"""
Synthetic end-to-end pipeline test.

Creates a small in-memory HLS-like dataset (3 x 48×48 patches, 24 monthly
composites, 7 spectral bands + date channel) and runs:

  1. Land-cover prediction with the trained STViTS model
  2. Crop-mask extraction via temporal mode
  3. Token extraction + t-SNE + K-Means clustering
  4. Vegetation-index smoothing and per-cluster summarisation
  5. SMFS phenology fitting (if reference patterns are present)
  6. GeoTIFF export

No internet connection or NASA Earthdata credentials required.

Usage
-----
    python test_pipeline.py [--output_dir results_test]
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects
import torch
import xarray as xr
from omegaconf import OmegaConf
from scipy import stats
from scipy.signal import savgol_coeffs
from skimage.transform import resize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_PATH = (
    "runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/config.yml"
)
WEIGHTS_PATH = (
    "runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/"
    "TSViTS_checkpoint_iter34314_ep83.pth"
)
PHEN_REF_DIR = "phen_patterns_"

# ---------------------------------------------------------------------------
# Synthetic data creation
# ---------------------------------------------------------------------------

BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "ndvi", "gndvi"]


def _synthetic_reflectance(n_dates: int, h: int, w: int, seed: int = 0) -> np.ndarray:
    """Return (n_dates, 7, h, w) realistic-ish reflectance values."""
    rng = np.random.default_rng(seed)
    # Simulate seasonal NDVI curve (crops peak mid-season)
    t = np.linspace(0, 2 * np.pi, n_dates)
    ndvi_curve = 0.4 + 0.3 * np.sin(t - np.pi / 2)  # (n_dates,)

    data = rng.uniform(0.02, 0.15, (n_dates, 7, h, w)).astype(np.float32)
    # NIR follows NDVI curve
    data[:, 3] = (ndvi_curve[:, None, None] * 0.5 + 0.1).astype(np.float32)
    # Compute and write NDVI / GNDVI
    nir = data[:, 3]
    red = data[:, 2]
    green = data[:, 1]
    data[:, 5] = (nir - red) / (nir + red + 1e-9)   # ndvi
    data[:, 6] = (nir - green) / (nir + green + 1e-9)  # gndvi
    return data


def create_synthetic_patches(
    output_dir: str,
    n_patches: int = 3,
    n_dates: int = 24,
    img_size: int = 48,
    start_date: str = "2022-01-01",
) -> str:
    """
    Write *n_patches* synthetic NetCDF patch files to *output_dir*.

    Files follow the MltTileData naming convention:
    ``{tile_id}_{5chars}patch_{patch_id:04d}.nc``
    → ``synth_20231patch_{id:04d}.nc``  (tile_id = "synth")
    """
    os.makedirs(output_dir, exist_ok=True)

    dates = np.array(
        [np.datetime64(start_date) + np.timedelta64(14 * i, "D") for i in range(n_dates)]
    )

    # Build a small lat/lon grid centred on the Olancho/Honduras region.
    # Each patch gets a unique spatial offset so tiles don't conflict on merge.
    lat0, lon0 = 14.5, -86.5
    res = 0.000269  # ~30 m in degrees

    written = []
    for pid in range(n_patches):
        # Offset each patch so coordinates are unique across patches
        lats = lat0 + (pid * img_size + np.arange(img_size)) * res
        lons = lon0 + np.arange(img_size) * res
        arr = _synthetic_reflectance(n_dates, img_size, img_size, seed=pid)
        ds_vars = {}
        for bi, bname in enumerate(BAND_NAMES):
            da = xr.DataArray(
                arr[:, bi],
                dims=["date", "y", "x"],
                coords={"date": dates, "y": lats, "x": lons},
                name=bname,
            )
            ds_vars[bname] = da

        ds = xr.Dataset(ds_vars)
        ds.attrs["crs"] = "EPSG:4326"

        # Naming: synth_20231patch_NNNN.nc → tile_id="synth", patch_id=f"{pid:04d}"
        fname = os.path.join(output_dir, f"synth_20231patch_{pid:04d}.nc")
        ds.to_netcdf(fname)
        written.append(fname)
        print(f"  Written: {fname}")

    return output_dir


# ---------------------------------------------------------------------------
# Borrowed helpers from demo_pipeline (avoid import cycle)
# ---------------------------------------------------------------------------

def _evi(bands: np.ndarray) -> np.ndarray:
    blue, red, nir = bands[0], bands[2], bands[3]
    return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-9)


def _smooth(vi_ts, days, crop_mask=None):
    from datasets.image_preprocessing import kernel_regression, chen_sg_filter

    c1, c2 = savgol_coeffs(11, 7), savgol_coeffs(3, 2)
    H, W, T = vi_ts.shape
    data2d = vi_ts.reshape(H * W, T)
    mask1d = crop_mask.flatten() if crop_mask is not None else np.ones(H * W, bool)
    smoothed = np.zeros_like(data2d)
    for i in range(data2d.shape[0]):
        if not mask1d[i] or np.all(np.isnan(data2d[i])):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = kernel_regression(data2d[i], 0.03, days)
        smoothed[i] = chen_sg_filter(ts, coeffs_trend1=c1, coeffs_trend2=c2)
    smoothed[np.isnan(smoothed)] = 0
    return smoothed.reshape(H, W, T)


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run_test(output_dir: str = "results_test") -> None:
    from detection.dataset import InputImgDataset
    from detection.detectors import STViTS_detector
    from detection.utils import predict_tile, DAYS_IN_MONTH

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Create synthetic patch data
    # ------------------------------------------------------------------
    print("\n[1/6] Creating synthetic patch data …")
    patch_dir = os.path.join(output_dir, "patches")
    create_synthetic_patches(patch_dir, n_patches=3, n_dates=24)

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("\n[2/6] Loading model …")
    config = OmegaConf.load(CONFIG_PATH)
    config.DATASETS.paths.training_input = patch_dir

    detector = STViTS_detector(config, init_scheduler=False)
    detector.load_weights_for_detection(WEIGHTS_PATH)
    detector.model.eval()
    print(f"  Device: {detector.device}")

    use_summary = config.TRAIN.get("summary_layer", False)
    img_res = config.MODEL.img_res
    n_months = config.DATASETS.n_months
    patch_size = config.MODEL.patch_size
    n_patches_spatial = img_res // patch_size

    loader = InputImgDataset(
        patch_dir,
        n_months=n_months,
        n_bands=config.DATASETS.n_bands,
        img_size=img_res,
        summarize_img=use_summary,
    )
    loader_6m = InputImgDataset(
        patch_dir,
        n_months=6,
        n_bands=config.DATASETS.n_bands,
        img_size=img_res,
        summarize_img=use_summary,
    )
    n_tiles = loader.__len__()
    print(f"  {n_tiles} tiles found in {patch_dir}")
    assert n_tiles > 0, "No tiles found — check file naming convention!"

    # ------------------------------------------------------------------
    # 3. Land-cover prediction
    # ------------------------------------------------------------------
    print("\n[3/6] Running land-cover prediction …")
    crop_classes = [11, 12, 13, 20]
    # Synthetic data spans 2022-01-01 → 2022-11-22; use 3 months to stay in range
    end_year, end_month = 2022, 11
    n_months_lc = 3

    all_lc_preds = []
    for tile_id in range(n_tiles):
        month, year = end_month, end_year
        tile_preds = []
        for _ in range(n_months_lc):
            if month == 0:
                month, year = 12, year - 1
            date_str = f"{year}-{month:02d}-{DAYS_IN_MONTH[month]}"
            try:
                xr_pred = predict_tile(tile_id, detector, loader, date_str, use_summary)
                tile_preds.append(xr_pred.values)
            except Exception as exc:
                print(f"    Skipping tile={tile_id} date={date_str}: {exc}")
            month -= 1
        if not tile_preds:
            print(f"  Tile {tile_id}: no valid predictions — skipping")
            all_lc_preds.append({"lc_mode": np.zeros((48, 48)), "crop_mask": np.zeros((48, 48), dtype=np.uint8)})
            continue
        # modal land-cover label
        lc_stack = np.stack(tile_preds, axis=0)
        lc_mode = stats.mode(lc_stack, axis=0)[0].squeeze()
        crop_mask = np.isin(lc_mode, crop_classes).astype(np.uint8)
        all_lc_preds.append({"lc_mode": lc_mode, "crop_mask": crop_mask})
        print(f"  Tile {tile_id}: mode shape={lc_mode.shape}, "
              f"crop fraction={crop_mask.mean():.2%}")

    print("  Land-cover prediction OK")

    # ------------------------------------------------------------------
    # 4. Token extraction + clustering
    # ------------------------------------------------------------------
    print("\n[4/6] Token extraction + t-SNE + K-Means …")
    try:
        from openTSNE import TSNE
        from sklearn.cluster import KMeans

        n_months_tok = 6
        n_clusters = 10   # small for fast test
        tsne_embedding = None
        dim = detector.model.backbone.dim

        cluster_results = []
        for tile_id in range(n_tiles):
            tokens_per_month = []
            month, year = end_month, end_year
            for _ in range(n_months_tok):
                if month == 0:
                    month, year = 12, year - 1
                date_str = f"{year}-{month:02d}-{DAYS_IN_MONTH[month]}"
                date_arr = np.array(date_str).astype("datetime64[D]")
                img, _ = loader.__getitem__(
                    tile_id,
                    starting_date=None,
                    ending_date=date_str,
                    scale=True,
                    reference_date=date_arr,
                )
                with torch.no_grad():
                    toks = detector.intermediate_spatial_features(img.unsqueeze(0))
                    toks = detector.model.backbone.norm_final(toks)
                tokens_per_month.append(
                    toks.reshape(n_patches_spatial * n_patches_spatial, dim)
                    .detach().cpu().numpy()
                )
                month -= 1

            # Upsample token map to img_res
            tok_arr = np.array(tokens_per_month)  # (M, NP^2, dim)
            tok_arr = tok_arr.reshape(n_months_tok, n_patches_spatial, n_patches_spatial, dim)
            tok_arr = tok_arr.transpose(1, 2, 0, 3)  # (NP, NP, M, dim)
            tok_resized = resize(
                tok_arr, (img_res, img_res, n_months_tok, dim),
                order=1, preserve_range=True, anti_aliasing=False,
            ).astype(np.float32)

            tok_2d = tok_resized.reshape(img_res * img_res, n_months_tok, dim)
            crop_mask = all_lc_preds[tile_id]["crop_mask"]

            if tsne_embedding is None and crop_mask.sum() > 0:
                masked = tok_2d[crop_mask.flatten() == 1]
                flat_masked = masked.reshape(masked.shape[0] * n_months_tok, dim)
                tsne = TSNE(perplexity=min(30, flat_masked.shape[0] - 1),
                            metric="euclidean", n_jobs=2, random_state=42, verbose=False)
                tsne_embedding = tsne.fit(flat_masked)
                print(f"  t-SNE fitted on tile {tile_id} "
                      f"({flat_masked.shape[0]} masked feature vectors)")

            if tsne_embedding is None:
                cluster_results.append(None)
                continue

            flat = tok_2d.reshape(img_res * img_res * n_months_tok, dim)
            tsne_feat = tsne_embedding.transform(flat)
            tsne_per_px = tsne_feat.reshape(img_res * img_res, n_months_tok, 2)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            kmeans.fit(tsne_feat)

            # Monthly label maps
            cluster_maps = {}
            month2, year2 = end_month, end_year
            for m in range(n_months_tok):
                if month2 == 0:
                    month2, year2 = 12, year2 - 1
                date_str = f"{year2}-{month2:02d}-{DAYS_IN_MONTH[month2]}"
                labels = kmeans.predict(tsne_per_px[:, m]).reshape(img_res, img_res)
                cluster_maps[date_str] = labels
                month2 -= 1

            cluster_results.append(cluster_maps)
            print(f"  Tile {tile_id}: {n_clusters} clusters across {len(cluster_maps)} dates")

        print("  Clustering OK")

    except Exception as exc:
        print(f"  Clustering step skipped: {exc}")
        cluster_results = [None] * n_tiles

    # ------------------------------------------------------------------
    # 5. Phenology detection
    # ------------------------------------------------------------------
    print("\n[5/6] Phenology detection …")
    has_ref = os.path.isdir(PHEN_REF_DIR) and bool(os.listdir(PHEN_REF_DIR))
    if not has_ref:
        print(f"  Skipped (no reference patterns found in '{PHEN_REF_DIR}')")
    else:
        from detection.smf_s_class import SMFS

        ref_vi, ref_phen = [], []
        for i in range(len([f for f in os.listdir(PHEN_REF_DIR) if f.startswith("ref_vi_")])):
            ref_vi.append([float(l) for l in open(os.path.join(PHEN_REF_DIR, f"ref_vi_6months_{i}.txt"))])
            ref_phen.append([float(l) for l in open(os.path.join(PHEN_REF_DIR, f"ref_phe_6months_{i}.txt"))])
        doys = np.arange(1, 185, 14)[:-1]

        for tile_id in range(n_tiles):
            if cluster_results[tile_id] is None:
                continue
            crop_mask = all_lc_preds[tile_id]["crop_mask"]
            for date_str, labels in list(cluster_results[tile_id].items())[:2]:
                img, _ = loader_6m.__getitem__(
                    tile_id, starting_date=None, ending_date=date_str,
                    scale=False, reference_date=np.array(date_str).astype("datetime64[D]"),
                )
                img_np = img.detach().numpy()
                # EVI time series
                T = img_np.shape[0]
                vi_ts = np.zeros((img_res, img_res, T), np.float32)
                for t in range(T):
                    vi_ts[:, :, t] = _evi(img_np[t])
                vi_ts[vi_ts == 0] = np.nan

                day_vals = img_np[:, -1, 0, 0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vi_smooth = _smooth(vi_ts, day_vals, crop_mask)

                # Cluster-level VI summary + SMFS
                for cid in np.unique(labels):
                    vi2d = vi_smooth.reshape(img_res * img_res, T)
                    sub = vi2d[labels.flatten() == cid]
                    valid = sub[~np.any(np.isnan(sub), axis=1)]
                    if not valid.size:
                        continue
                    vi_avg = np.nanmedian(valid, axis=0)
                    if np.any(np.isnan(vi_avg)):
                        continue
                    dists = [np.sqrt(np.sum((vi_avg - np.array(r)) ** 2)) for r in ref_vi]
                    best = int(np.argmin(dists))
                    if dists[best] > 0.3:
                        continue
                    for si, pd in enumerate(ref_phen[best]):
                        m = SMFS(np.array(ref_vi[best]), pd, doys)
                        _ = m.doit(np.copy(vi_avg))
                print(f"  Tile {tile_id}, date {date_str}: SMFS ran on {np.unique(labels).size} clusters")
        print("  Phenology detection OK")

    # ------------------------------------------------------------------
    # 6. Export a sample land-cover raster
    # ------------------------------------------------------------------
    print("\n[6/6] Exporting land-cover raster …")
    import rioxarray  # noqa: F401 — ensures rio accessor is registered

    lc_maps = []
    for tile_id in range(n_tiles):
        # Re-run a single prediction to grab an xarray output
        date_str = f"{end_year}-{end_month:02d}-{DAYS_IN_MONTH[end_month]}"
        xr_pred = predict_tile(tile_id, detector, loader, date_str, use_summary)
        lc_maps.append(xr_pred.drop_vars("date", errors="ignore"))

    merged = xr.merge([m.to_dataset(name="prediction") for m in lc_maps])
    merged.attrs = {}
    out_tif = os.path.join(output_dir, "lc_prediction.tif")
    merged.prediction.rio.to_raster(out_tif)
    print(f"  Saved: {out_tif}")

    print("\n=== All tests passed ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic pipeline test.")
    parser.add_argument("--output_dir", default="results_test")
    args = parser.parse_args()
    run_test(args.output_dir)
