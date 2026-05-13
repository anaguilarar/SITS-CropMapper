"""
End-to-end demo: HLS download → crop segmentation → phenology detection.

This script demonstrates the full framework for any area of interest:

1.  Download HLS satellite time-series data via earthaccess (optional).
2.  Run the trained DinoTSViT model to produce land-cover maps.
3.  Identify crop pixels and cluster them using t-SNE + K-Means.
4.  Detect crop phenology stages (Greenup, Maturity, Senescence, Dormancy)
    by matching EVI time-series to reference patterns with the SMFS algorithm.
5.  Export georeferenced phenology maps as GeoTIFF.

Usage
-----
    python demo_pipeline.py --help

    # Minimal example (uses data already in data/)
    python demo_pipeline.py --input_dir olancho48/all_filtered --output_dir results/

    # Full pipeline including HLS download
    python demo_pipeline.py \\
        --download \\
        --bbox -87.5 13.5 -87.0 14.0 \\
        --start_date 2022-01-01 \\
        --end_date 2023-12-31 \\
        --input_dir data/hls_patches \\
        --output_dir results/

References
----------
Tarasiou et al. (2023) ViTs for SITS: Vision Transformers for Satellite Image
    Time Series. CVPR.
Liu et al. (2022) Detecting crop phenology from vegetation index time-series data
    by improved shape model fitting in each phenological stage.
    Remote Sensing of Environment, 278, 113098.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects
import torch
import xarray
from omegaconf import OmegaConf
from scipy import stats
from scipy.signal import savgol_coeffs
from skimage.transform import resize
from tqdm import tqdm

from datasets.image_preprocessing import chen_sg_filter, kernel_regression
from detection.dataset import InputImgDataset
from detection.detectors import STViTS_detector
from detection.smf_s_class import SMFS
from detection.utils import predict_tile, DAYS_IN_MONTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = "runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/config.yml"
DEFAULT_WEIGHTS = (
    "runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/"
    "TSViTS_checkpoint_iter34314_ep83.pth"
)
DEFAULT_PHEN_REF = "phen_patterns_"

# Land-cover class groups (model trained on 21-class scheme)
LC_GROUPS: Dict[str, List[int]] = {
    "crops": [11, 12, 13, 20],
    "trees": [1, 2, 3, 4, 7, 8],
    "water": [17, 18],
    "soil": [16, 15],
    "urban": [14],
    "vegetation": [21, 5, 6],
    "others": [9, 10, 19],
}


# ---------------------------------------------------------------------------
# EVI helpers
# ---------------------------------------------------------------------------

def _evi(bands: np.ndarray) -> np.ndarray:
    """EVI from bands array shaped (C, H, W): blue=0, green=1, red=2, nir=3."""
    blue, red, nir = bands[0], bands[2], bands[3]
    return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-9)


def get_vi_image_ts(mlt_image: np.ndarray, vi: str = "evi") -> np.ndarray:
    """Compute vegetation index time series from a (T, C, H, W) image cube.

    Parameters
    ----------
    mlt_image:
        Multi-temporal image array shaped ``(T, C, H, W)``.
    vi:
        Vegetation index to compute. Currently supports ``"evi"`` and ``"ndvi"``.

    Returns
    -------
    np.ndarray
        Array shaped ``(H, W, T)`` with VI values. Zero-valued pixels are set
        to NaN (they indicate missing observations).
    """
    H, W = mlt_image.shape[2], mlt_image.shape[3]
    T = mlt_image.shape[0]
    vi_ts = np.zeros((H, W, T), dtype=np.float32)

    for t in range(T):
        if vi == "evi":
            vi_ts[:, :, t] = _evi(mlt_image[t])
        elif vi == "ndvi":
            nir = mlt_image[t, 3]
            red = mlt_image[t, 2]
            vi_ts[:, :, t] = (nir - red) / (nir + red + 1e-9)

    vi_ts[vi_ts == 0] = np.nan
    return vi_ts


# ---------------------------------------------------------------------------
# Time-series smoothing
# ---------------------------------------------------------------------------

def smooth_vi_timeseries(
    vi_ts: np.ndarray,
    days: np.ndarray,
    crop_mask: Optional[np.ndarray] = None,
    coeffs1=savgol_coeffs(11, 7),
    coeffs2=savgol_coeffs(3, 2),
) -> np.ndarray:
    """Apply kernel regression + Chen SG filter to a VI time-series image.

    Parameters
    ----------
    vi_ts:
        VI image shaped ``(H, W, T)``.
    days:
        1-D array of day offsets for each time step ``T``.
    crop_mask:
        Optional 2-D boolean mask ``(H, W)`` — only smooth masked pixels.
    coeffs1, coeffs2:
        Savitzky-Golay coefficient arrays for the two SG passes.

    Returns
    -------
    np.ndarray
        Smoothed VI array, same shape as *vi_ts*. Non-crop / all-NaN pixels
        remain zero.
    """
    H, W, T = vi_ts.shape
    data2d = vi_ts.reshape(H * W, T)
    mask1d = crop_mask.flatten() if crop_mask is not None else np.ones(H * W, dtype=bool)
    smoothed = np.zeros_like(data2d)

    for i in tqdm(range(data2d.shape[0]), desc="Smoothing VI", leave=False):
        if not mask1d[i] or np.all(np.isnan(data2d[i])):
            continue
        ts_interp = kernel_regression(data2d[i], 0.03, days)
        smoothed[i] = chen_sg_filter(
            ts_interp, coeffs_trend1=coeffs1, coeffs_trend2=coeffs2
        )

    smoothed[np.isnan(smoothed)] = 0
    return smoothed.reshape(H, W, T)


# ---------------------------------------------------------------------------
# Cluster-level VI summarisation
# ---------------------------------------------------------------------------

def summarize_vi_per_cluster(
    vi_ts: np.ndarray,
    labels: np.ndarray,
    method: str = "median",
) -> Dict[int, np.ndarray]:
    """Compute per-cluster median (or mean) VI time series.

    Parameters
    ----------
    vi_ts:
        Smoothed VI array shaped ``(H, W, T)``.
    labels:
        Integer cluster label map shaped ``(H, W)``.
    method:
        ``"median"`` (default) or ``"mean"``.

    Returns
    -------
    dict
        Mapping ``{cluster_id: vi_timeseries_1d}``.
    """
    agg = np.nanmedian if method == "median" else np.nanmean
    H, W, T = vi_ts.shape
    vi2d = vi_ts.reshape(H * W, T)
    result: Dict[int, np.ndarray] = {}

    for cid in np.unique(labels):
        mask = (labels.flatten() == cid)
        subset = vi2d[mask]
        valid = subset[~np.any(np.isnan(subset), axis=1)]
        valid = valid[np.any(valid != 0, axis=1)]  # exclude all-zero (non-crop) rows
        if valid.size:
            result[int(cid)] = agg(valid, axis=0)

    return result


# ---------------------------------------------------------------------------
# Phenology utilities
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two equal-length 1-D arrays."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def detect_phenology(
    vi_clusters: Dict[int, np.ndarray],
    ts_reference: Dict[str, list],
    similarity_threshold: float = 0.30,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Match cluster VI curves to reference patterns and run SMFS fitting.

    Parameters
    ----------
    vi_clusters:
        Per-cluster VI time series from :func:`summarize_vi_per_cluster`.
    ts_reference:
        Dict with keys ``"vi_ts"``, ``"phen_days"``, and ``"crop_labels"``.
    similarity_threshold:
        Maximum raw Euclidean distance to reference curve; farther clusters
        are skipped. Raised to 0.30 to accommodate lower-EVI crops (beans,
        sorghum) whose absolute VI values differ from the high-EVI maize
        reference training set.

    Returns
    -------
    (matched_vi, phen_days, crop_types)
        ``matched_vi``: ``{cluster_id: vi_ts}`` for matched clusters.
        ``phen_days``: ``{cluster_id: (4,1,1) array}`` with DOY per stage.
        ``crop_types``: ``{cluster_id: str}`` matched crop label.
    """
    ref_vi     = ts_reference["vi_ts"]
    ref_phen   = ts_reference["phen_days"]
    ref_crops  = ts_reference.get("crop_labels", ["unknown"] * len(ref_vi))
    doys = np.arange(1, 185, 14)[:-1]

    matched_vi: Dict[int, np.ndarray] = {}
    phen_days:  Dict[int, np.ndarray] = {}
    crop_types: Dict[int, str]        = {}

    for cid, vi_ts in vi_clusters.items():
        if np.any(np.isnan(vi_ts)):
            continue

        distances = [euclidean_distance(vi_ts, np.array(r)) for r in ref_vi]
        best = int(np.argmin(distances))

        if distances[best] > similarity_threshold:
            continue

        matched_vi[cid] = vi_ts
        crop_types[cid] = ref_crops[best]
        stage_days = np.full((4, 1, 1), np.nan)

        for stage_idx, phen_day in enumerate(ref_phen[best]):
            model = SMFS(np.array(ref_vi[best]), phen_day, doys)
            result = model.doit(np.copy(vi_ts))
            stage_days[stage_idx, 0, 0] = result

        phen_days[cid] = stage_days

    return matched_vi, phen_days, crop_types


def get_julian_day(date: np.datetime64) -> int:
    """Convert a numpy datetime64 to day-of-year (Julian day)."""
    year = date.astype("datetime64[Y]").astype(int) + 1970
    ref = np.array(f"{year}-01-01").astype("datetime64[D]")
    return int((date.astype("datetime64[D]") - ref) / np.timedelta64(1, "D"))


def find_phenology_dates(
    stage_days: np.ndarray,
    reference_dates: List,
) -> Dict[str, Optional[np.datetime64]]:
    """Convert relative-day offsets to absolute phenology dates."""
    stage_names = ["Greenup", "Maturity", "Senescence", "Dormancy"]
    result: Dict[str, Optional[np.datetime64]] = {s: None for s in stage_names}

    for i, day in enumerate(stage_days.flatten()):
        if day and not np.isnan(day):
            result[stage_names[i]] = reference_dates[0] + np.timedelta64(int(day), "D")

    return result


def validate_phenology(
    phen: np.ndarray,
    max_greenup_to_maturity: int = 105,
    max_senescence_to_dormancy: int = 105,
) -> np.ndarray:
    """Remove biologically implausible phenology detections.

    Parameters
    ----------
    phen:
        Shape ``(4, 1, 1)`` array with day values for each stage. NaN = not detected.
    max_greenup_to_maturity:
        Maximum allowed days from Greenup to Maturity.
    max_senescence_to_dormancy:
        Maximum allowed days from Senescence to Dormancy.

    Returns
    -------
    np.ndarray
        Corrected phenology array (same shape).
    """
    corrected = phen.copy()
    g, m, s, d = [phen[i, 0, 0] for i in range(4)]

    def isnan(v):
        return v == 0 or np.isnan(v)

    # If Greenup is missing, clear Maturity and Senescence
    if isnan(g) and not (isnan(m) and isnan(s)):
        m, s = np.nan, np.nan

    # Greenup must precede Maturity within allowed window
    if not isnan(g) and not isnan(m):
        if m - g > max_greenup_to_maturity or g >= m:
            g, m = np.nan, np.nan

    # Senescence without preceding Greenup+Maturity is invalid
    if (isnan(g) and isnan(m)) and (not isnan(s) and isnan(d)):
        s = np.nan

    # Senescence must precede Dormancy within allowed window
    if not isnan(s) and not isnan(d):
        if d - s > max_senescence_to_dormancy:
            s, d = np.nan, np.nan

    for i, v in enumerate([g, m, s, d]):
        corrected[i, 0, 0] = v
    return corrected


def aggregate_to_biweekly(phen_map: np.ndarray, interval: int = 14) -> np.ndarray:
    """Snap raw day values in *phen_map* to bi-weekly bin centres.

    Rare values (< 1 % of pixels) are suppressed to reduce noise.

    Parameters
    ----------
    phen_map:
        2-D array of day-of-year values (NaN = no detection).
    interval:
        Bin width in days (default 14 = bi-weekly).

    Returns
    -------
    np.ndarray
        Snapped phen_map with suppressed rare values.
    """
    bins = [(i * interval, (i + 1) * interval) for i in range(365 // interval)]
    bin_centres = [lo + (hi - lo) // 2 for lo, hi in bins]

    result = phen_map.copy()
    for val in np.unique(phen_map):
        if np.isnan(val):
            continue
        for (lo, hi), centre in zip(bins, bin_centres):
            if lo <= val < hi:
                result[phen_map == val] = centre
                break

    # Suppress statistically rare values
    total = phen_map.size
    for val in np.unique(result):
        if np.isnan(val):
            continue
        if (np.sum(result == val) / total) * 100 < 1.0:
            result[result == val] = np.nan

    return result


# ---------------------------------------------------------------------------
# Multi-date token extraction
# ---------------------------------------------------------------------------

def extract_tokens(
    tile_id: int,
    detector: STViTS_detector,
    data_loader: InputImgDataset,
    end_year: int,
    end_month: int,
    n_months: int,
) -> List[np.ndarray]:
    """Extract transformer patch tokens for *n_months* successive months.

    Iterates backwards in time from ``end_year/end_month``.

    Parameters
    ----------
    tile_id:
        Dataset tile index.
    detector:
        Loaded :class:`STViTS_detector` in eval mode.
    data_loader:
        :class:`InputImgDataset` wrapping the patch NetCDF files.
    end_year, end_month:
        Most recent year and month to include.
    n_months:
        Total number of monthly windows to extract.

    Returns
    -------
    list of np.ndarray
        One ``(N_patches, dim)`` array per month, ordered most-recent first.
    """
    month, year = end_month, end_year
    tokens_per_month = []

    for _ in tqdm(range(n_months), desc="Extracting tokens", leave=False):
        if month == 0:
            month, year = 12, year - 1

        day = DAYS_IN_MONTH[month]
        date_str = f"{year}-{month:02d}-{day}"
        date_arr = np.array(date_str).astype("datetime64[D]")

        try:
            img, _ = data_loader.__getitem__(
                tile_id,
                starting_date=None,
                ending_date=date_str,
                scale=True,
                reference_date=date_arr,
            )

            with torch.no_grad():
                patch_tokens = detector.intermediate_spatial_features(img.unsqueeze(0))
                patch_tokens = detector.model.backbone.norm_final(patch_tokens)

            n_patches = detector.model.backbone.image_size // detector.model.backbone.patch_size
            tokens_per_month.append(
                patch_tokens.reshape(n_patches * n_patches, detector.model.backbone.dim)
                .detach()
                .cpu()
                .numpy()
            )
        except Exception:
            pass  # date outside data range — skip silently

        month -= 1

    return tokens_per_month


# ---------------------------------------------------------------------------
# Land-cover prediction
# ---------------------------------------------------------------------------

def predict_lc_timeseries(
    tile_id: int,
    detector: STViTS_detector,
    data_loader: InputImgDataset,
    end_year: int,
    end_month: int,
    n_months: int,
    use_summary: bool = True,
) -> List[np.ndarray]:
    """Run the land-cover model for *n_months* consecutive months.

    Returns a list of 2-D prediction arrays (H × W), most-recent first.
    """
    month, year = end_month, end_year
    predictions = []

    for _ in range(n_months):
        if month == 0:
            month, year = 12, year - 1

        day = DAYS_IN_MONTH[month]
        date_str = f"{year}-{month:02d}-{day}"

        try:
            xr_pred = predict_tile(tile_id, detector, data_loader, date_str, use_summary)
            predictions.append(xr_pred.values)
        except Exception:
            pass  # date outside data range — skip silently

        month -= 1

    return predictions


# ---------------------------------------------------------------------------
# Reference phenology patterns reader
# ---------------------------------------------------------------------------

def load_phenology_references(ref_dir: str) -> Dict[str, list]:
    """Load reference VI curves, phenology day arrays, and crop labels from text files.

    Files must follow the naming convention:
    ``ref_vi_6months_{i}.txt``, ``ref_phe_6months_{i}.txt``, and optionally
    ``ref_crop_6months_{i}.txt`` (crop type label, defaults to "unknown").

    Parameters
    ----------
    ref_dir:
        Directory containing the reference pattern files.

    Returns
    -------
    dict
        ``{"vi_ts": [...], "phen_days": [...], "crop_labels": [...]}``.
    """
    if not os.path.isdir(ref_dir):
        warnings.warn(
            f"Phenology reference directory '{ref_dir}' not found. "
            "Phenology detection will be skipped."
        )
        return {"vi_ts": [], "phen_days": [], "crop_labels": []}

    files = os.listdir(ref_dir)
    n_refs = len([f for f in files if f.startswith("ref_vi_")]) if files else 0

    vi_ts, phen_days, crop_labels = [], [], []
    for i in range(n_refs):
        vi_path   = os.path.join(ref_dir, f"ref_vi_6months_{i}.txt")
        ph_path   = os.path.join(ref_dir, f"ref_phe_6months_{i}.txt")
        crop_path = os.path.join(ref_dir, f"ref_crop_6months_{i}.txt")

        vi_ts.append([float(line) for line in open(vi_path).readlines()])
        phen_days.append([float(line) for line in open(ph_path).readlines()])
        if os.path.exists(crop_path):
            crop_labels.append(open(crop_path).read().strip())
        else:
            crop_labels.append("unknown")

    if not vi_ts:
        warnings.warn(
            f"No reference phenology files found in {ref_dir}. "
            "Phenology detection will be skipped."
        )

    return {"vi_ts": vi_ts, "phen_days": phen_days, "crop_labels": crop_labels}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_dir: str,
    output_dir: str,
    config_path: str = DEFAULT_CONFIG,
    weights_path: str = DEFAULT_WEIGHTS,
    phen_ref_dir: str = DEFAULT_PHEN_REF,
    end_year: int = 2024,
    end_month: int = 12,
    n_months_lc: int = 24,
    n_months_cluster: int = 12,
    n_months_phen: int = 8,
    n_clusters: int = 120,
    output_prefix: str = "output",
) -> Tuple[str, str]:
    """Execute the full segmentation + phenology pipeline.

    Parameters
    ----------
    input_dir:
        Path to directory containing patch NetCDF files.
    output_dir:
        Directory where output GeoTIFFs are saved.
    config_path:
        Path to model YAML configuration.
    weights_path:
        Path to trained model checkpoint (.pth).
    phen_ref_dir:
        Directory with reference phenology pattern text files.
    end_year, end_month:
        Most recent year/month to use for temporal windows.
    n_months_lc:
        Number of monthly windows for land-cover prediction.
    n_months_cluster:
        Number of monthly windows for token extraction (clustering).
    n_months_phen:
        Number of monthly windows over which to search for phenology.
    n_clusters:
        Number of K-Means clusters for VI grouping.
    output_prefix:
        Filename prefix for output GeoTIFFs.

    Returns
    -------
    Tuple[str, str]
        ``(lc_path, phenology_path)``.  Either path is an empty string if that
        output was not produced (e.g. no crop tiles found, or phenology
        detection failed for all tiles).
    """
    from openTSNE import TSNE
    from sklearn.cluster import KMeans

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    log.info("Loading model from %s", weights_path)
    config = OmegaConf.load(config_path)
    config.DATASETS.paths.training_input = input_dir

    detector = STViTS_detector(config, init_scheduler=False)
    detector.load_weights_for_detection(weights_path)
    detector.model.eval()

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    use_summary = config.TRAIN.get("summary_layer", False)

    loader_full = InputImgDataset(
        input_dir,
        n_months=config.DATASETS.n_months,
        n_bands=config.DATASETS.n_bands,
        img_size=config.MODEL.img_res,
        summarize_img=use_summary,
    )
    loader_6m = InputImgDataset(
        input_dir,
        n_months=6,
        n_bands=config.DATASETS.n_bands,
        img_size=config.MODEL.img_res,
        summarize_img=use_summary,
    )

    # ------------------------------------------------------------------
    # Reference phenology patterns
    # ------------------------------------------------------------------
    ts_reference = load_phenology_references(phen_ref_dir)
    has_references = bool(ts_reference["vi_ts"])

    n_tiles = loader_6m.__len__()
    log.info("Processing %d tiles …", n_tiles)

    img_res = config.MODEL.img_res
    patch_size = config.MODEL.patch_size
    n_patches = img_res // patch_size

    phenology_maps: List[xarray.Dataset] = []
    lc_maps: List[xarray.Dataset] = []
    tsne_embedding = None  # fitted once on the first tile, reused thereafter

    # SG filter coefficients (pre-computed for speed)
    sg_coeff1 = savgol_coeffs(11, 7)
    sg_coeff2 = savgol_coeffs(3, 2)

    for tile_id in tqdm(range(n_tiles), desc="Tiles"):
        try:
            # -- Land-cover prediction (multi-month modal mask) ----------------
            lc_predictions = predict_lc_timeseries(
                tile_id, detector, loader_full,
                end_year, end_month, n_months_lc, use_summary,
            )
            lc_stack = np.stack(lc_predictions, axis=0)
            crop_mask = stats.mode(
                np.isin(lc_stack, LC_GROUPS["crops"]).astype(np.uint8), axis=0
            )[0].squeeze()
            lc_mode = stats.mode(lc_stack, axis=0)[0].squeeze()

            # Capture modal LC class as a spatially-referenced xarray DataArray
            loader_full.get_tiles_data(tile_id)
            sp_lc = xarray.zeros_like(loader_full._xrdata.isel(date=0).blue)
            sp_lc.values = lc_mode
            sp_lc = sp_lc.rename("lc_class")
            lc_maps.append(sp_lc.to_dataset().drop_vars("date", errors="ignore"))

            # Skip tiles with no crop pixels — nothing to cluster or detect
            n_crop_pixels = int(crop_mask.sum())
            if n_crop_pixels == 0:
                log.debug("Tile %d skipped: no crop pixels detected", tile_id)
                continue

            # -- Token extraction and t-SNE + K-Means clustering --------------
            all_tokens = extract_tokens(
                tile_id, detector, loader_full,
                end_year, end_month, n_months_cluster,
            )

            tokens_arr = (
                np.array(all_tokens)
                .reshape(n_months_cluster, n_patches, n_patches, detector.model.backbone.dim)
                .transpose(1, 2, 0, 3)
            )
            tokens_resized = resize(
                tokens_arr, (img_res, img_res, n_months_cluster, detector.model.backbone.dim),
                order=3, preserve_range=True, anti_aliasing=True,
            ).astype(np.float32)

            tokens2d = tokens_resized.reshape(
                img_res * img_res, n_months_cluster, detector.model.backbone.dim
            )

            if tsne_embedding is None:
                masked_tokens = tokens2d[crop_mask.flatten() == 1]
                n_tsne_samples = masked_tokens.shape[0] * n_months_cluster
                if n_tsne_samples < 2:
                    log.debug("Tile %d skipped: too few crop samples for t-SNE (%d)", tile_id, n_tsne_samples)
                    continue
                flat_masked = masked_tokens.reshape(n_tsne_samples, detector.model.backbone.dim)
                perplexity = min(30, n_tsne_samples - 1)
                tsne = TSNE(perplexity=perplexity, metric="euclidean", n_jobs=4, random_state=42, verbose=False)
                tsne_embedding = tsne.fit(flat_masked)
                log.info("t-SNE embedding fitted on tile %d (%d crop samples)", tile_id, n_tsne_samples)

            tokens2d[crop_mask.flatten() == 0] = 0
            flat_tokens = tokens2d.reshape(img_res * img_res * n_months_cluster, detector.model.backbone.dim)
            tsne_features = tsne_embedding.transform(flat_tokens)
            tsne_features_per_px = tsne_features.reshape(img_res * img_res, n_months_cluster, 2)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            kmeans.fit(tsne_features)

            # Build per-month cluster label maps (most-recent month first)
            month, year = end_month, end_year
            cluster_maps: Dict[str, np.ndarray] = {}
            for m in range(n_months_cluster):
                date_str = f"{year}-{month:02d}-{DAYS_IN_MONTH[month]}"
                labels = kmeans.predict(tsne_features_per_px[:, m]).reshape(img_res, img_res).astype(float)
                cluster_maps[date_str] = labels
                month -= 1
                if month == 0:
                    month, year = 12, year - 1

            # -- Phenology detection -------------------------------------------
            if not has_references:
                continue

            phen_per_date: List[np.ndarray] = []
            crop_maps_per_date: List[np.ndarray] = []
            processed_dates: List[str] = []

            for date_str in list(cluster_maps.keys())[:n_months_phen]:
                labels = cluster_maps[date_str]
                img, _ = loader_6m.__getitem__(
                    tile_id,
                    starting_date=None,
                    ending_date=date_str,
                    scale=False,
                    reference_date=np.array(date_str).astype("datetime64[D]"),
                )
                real_dates = loader_6m._new_dates
                vi_ts = get_vi_image_ts(img.detach().numpy(), vi="evi")
                day_vals = img.detach().numpy()[:, -1, 0, 0]
                vi_smooth = smooth_vi_timeseries(vi_ts, day_vals, crop_mask, sg_coeff1, sg_coeff2)

                # Pixel-direct reference matching — avoids cluster-median flattening
                _CROP_CODES = {"unknown": 0, "maize": 1, "beans": 2, "sorghum": 3, "beans_double_season": 4}
                _ref_vi   = ts_reference["vi_ts"]
                _ref_phen = ts_reference["phen_days"]
                _ref_crop = ts_reference.get("crop_labels", ["unknown"] * len(_ref_vi))
                _doys     = np.arange(1, 185, 14)[:-1]
                _PHEN_THRESHOLD = 0.50

                ref_vi_arr = np.array(_ref_vi, dtype=np.float32)  # (n_refs, T)
                vi2d = vi_smooth.reshape(img_res * img_res, vi_smooth.shape[2])
                phen_map = np.zeros((4, img_res * img_res), dtype=np.float32)
                crop_map = np.zeros(img_res * img_res, dtype=np.int16)
                _crop_flat = crop_mask.flatten()

                for px in range(vi2d.shape[0]):
                    if _crop_flat[px] == 0:
                        continue
                    px_vi = vi2d[px]
                    if np.all(px_vi == 0) or np.any(np.isnan(px_vi)):
                        continue
                    dists = np.sqrt(np.sum((ref_vi_arr - px_vi) ** 2, axis=1))
                    best_ref = int(np.argmin(dists))
                    if dists[best_ref] > _PHEN_THRESHOLD:
                        continue
                    crop_map[px] = _CROP_CODES.get(_ref_crop[best_ref], 0)
                    stage_days = np.full((4, 1, 1), np.nan)
                    for stage_idx, phen_day_ref in enumerate(_ref_phen[best_ref]):
                        model = SMFS(np.array(_ref_vi[best_ref]), phen_day_ref, _doys)
                        stage_days[stage_idx, 0, 0] = model.doit(np.copy(px_vi))
                    validated  = validate_phenology(stage_days)
                    px_dates   = find_phenology_dates(validated, real_dates)
                    for stage_idx, stage_name in enumerate(["Greenup", "Maturity", "Senescence", "Dormancy"]):
                        phen_day = px_dates.get(stage_name)
                        if phen_day is not None:
                            phen_map[stage_idx, px] = get_julian_day(phen_day)

                if not crop_map.any():
                    continue

                phen_per_date.append(phen_map.reshape(4, img_res, img_res))
                crop_maps_per_date.append(crop_map.reshape(img_res, img_res))
                processed_dates.append(date_str)

            if not phen_per_date:
                continue

            # -- Temporal aggregation (median across processed dates) ----------
            median_phen = np.zeros((4, img_res, img_res), dtype=np.float32)
            for stage_idx in range(4):
                stage_stack = np.array([p[stage_idx] for p in phen_per_date]).astype(np.float32)
                stage_stack[stage_stack == 0] = np.nan
                median_phen[stage_idx] = aggregate_to_biweekly(np.nanmedian(stage_stack, axis=0))
            # Modal crop type across processed dates
            from scipy import stats as _stats
            modal_crop = _stats.mode(np.stack(crop_maps_per_date, axis=0), axis=0)[0].squeeze()

            # -- Build xarray output for this tile ----------------------------
            stage_names = ["Greenup", "Maturity", "Senescence", "Dormancy"]
            dummy_bands = ["blue", "green", "red", "nir"]
            sp_data = xarray.zeros_like(loader_6m._xrdata[dummy_bands].isel(date=0))
            for idx, (band, stage) in enumerate(zip(dummy_bands, stage_names)):
                sp_data[band].values = median_phen[idx]
            sp_data = sp_data.rename({b: s for b, s in zip(dummy_bands, stage_names)})
            # Add crop_type as an extra variable (same spatial grid, integer codes)
            sp_data["crop_type"] = xarray.zeros_like(sp_data["Greenup"]).astype(np.int16)
            sp_data["crop_type"].values = modal_crop.astype(np.int16)
            phenology_maps.append(sp_data.drop_vars("date", errors="ignore"))

            # -- Periodic checkpoint export ------------------------------------
            if tile_id > 0 and tile_id % 10 == 0:
                _export_mosaic(phenology_maps, output_dir, f"{output_prefix}_partial.tif")

        except Exception as exc:
            log.warning("Tile %d failed: %s", tile_id, exc)
            continue

    # -- Export land-cover map (always, even if phenology failed) -------------
    lc_path = ""
    if lc_maps:
        merged_lc = xarray.merge(lc_maps)
        merged_lc.attrs = {}
        lc_path = os.path.join(output_dir, f"{output_prefix}_lc.tif")
        merged_lc["lc_class"].rio.to_raster(lc_path)
        log.info("Land-cover map saved to %s", lc_path)
    else:
        log.warning("No tiles produced land-cover maps.")

    if not phenology_maps:
        log.error("No tiles produced valid phenology maps.")
        return lc_path, ""

    phen_path = _export_mosaic(phenology_maps, output_dir, f"{output_prefix}_phenology.tif")
    log.info("Phenology map saved to %s", phen_path)
    return lc_path, phen_path


def _export_mosaic(maps: List[xarray.Dataset], output_dir: str, filename: str) -> str:
    """Merge tile datasets and write to GeoTIFF."""
    merged = xarray.merge(maps)
    merged.attrs = {}
    out_path = os.path.join(output_dir, filename)
    merged.rio.to_raster(out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DinoTSViT crop segmentation + phenology detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Download options
    parser.add_argument("--download", action="store_true", help="Download HLS data before inference.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box for HLS download.")
    parser.add_argument("--start_date", default="2022-01-01", help="HLS download start date.")
    parser.add_argument("--end_date", default="2023-12-31", help="HLS download end date.")
    parser.add_argument("--earthdata_strategy", default="netrc",
                        choices=["netrc", "prompt", "environment"],
                        help="Earthdata login strategy.")

    # Inference options
    parser.add_argument("--input_dir", required=True, help="Directory with patch NetCDF files.")
    parser.add_argument("--output_dir", default="results", help="Output directory.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Model config YAML.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Model checkpoint (.pth).")
    parser.add_argument("--phen_ref", default=DEFAULT_PHEN_REF, help="Phenology reference dir.")
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--end_month", type=int, default=12)
    parser.add_argument("--n_months_lc", type=int, default=24)
    parser.add_argument("--n_months_cluster", type=int, default=12)
    parser.add_argument("--n_months_phen", type=int, default=8)
    parser.add_argument("--n_clusters", type=int, default=120)
    parser.add_argument("--prefix", default="output", help="Output filename prefix.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Step 1: optional HLS download
    if args.download:
        if args.bbox is None:
            raise SystemExit("--bbox is required when --download is set.")
        from utils.hls_download import download_hls_for_area

        log.info("Downloading HLS data for bbox %s …", args.bbox)
        download_hls_for_area(
            bbox=tuple(args.bbox),
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.input_dir,
            earthdata_strategy=args.earthdata_strategy,
        )

    # Step 2: inference pipeline
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        weights_path=args.weights,
        phen_ref_dir=args.phen_ref,
        end_year=args.end_year,
        end_month=args.end_month,
        n_months_lc=args.n_months_lc,
        n_months_cluster=args.n_months_cluster,
        n_months_phen=args.n_months_phen,
        n_clusters=args.n_clusters,
        output_prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
