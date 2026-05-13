# SITS-CropMapper: Self-supervised ViT for Satellite Image Time Series

A self-supervised learning framework for crop segmentation and phenology detection from multi-temporal satellite imagery. The system utilizes a **Temporo-Spatial Vision Transformer (TSViT)**—architected specifically to prioritize temporal dynamics over spatial context—pre-trained using **DINO** self-supervision on HLS data, and paired with an **SMF-S** stage-specific phenological detector.

## Overview

The framework handles the complexity of satellite time series by factorizing temporal and spatial information separately, allowing for precise mapping of both land-cover and growth stages:

| Stage | Method | Core Advantage | Output |
|-------|--------|----------------|--------|
| **Land-cover Segmentation** | DinoTSViT + Seg Head | Temporal-then-Spatial factorization | 21-class Crop / Non-crop map |
| **Phenology Detection** | SMF-S Curve Fitting | Stage-specific iterative fitting | Greenup / Maturity / Senescence / Dormancy DOY |

## Processing Pipeline

The system executes a multi-stage workflow to transform raw satellite observations into agricultural insights:

![SITS-CropMapper Workflow](assets/image/workflow.png)

1.  **Download**: Queries NASA Earthdata for HLS (Sentinel-2 + Landsat) granules over a specified bounding box and date range.
2.  **Segmentation**: Loads the pre-trained **DinoTSViT** checkpoint and performs land-cover prediction to isolate cropland.
3.  **Phenology Detection**: Fits EVI time-series to smooth growth curves via the **SMF-S** algorithm to detect the exact Day of Year (DOY) for key phenological stages.
4.  **Export**: Compiles segmentation maps and phenology dates into standardized spatial products (GeoTIFFs).

**Data source:** [NASA Harmonized Landsat Sentinel-2 (HLS)](https://lpdaac.usgs.gov/products/hlss30v002/) — 30 m, cloud-harmonized time series of surface reflectance.

## Scientific Background

This repository implements the methodologies described in two foundational research papers:

- **TSViT (Temporo-Spatial Vision Transformer):** Tarasiou et al. (2023) *ViTs for SITS: Vision Transformers for Satellite Image Time Series*. CVPR. [PDF](bibliography/vits_for_sits.pdf)
    - **Temporal-First Factorization:** Unlike video transformers, TSViT processes the entire temporal sequence of a pixel before looking at spatial neighbors, capturing unique phenological "signatures."
    - **Timing-Aware Encodings:** Uses acquisition-time-specific positional encodings, making the model aware of calendar dates rather than just sequence order.
- **SMF-S (Shape Model Fitting by Separate stage):** Liu et al. (2022) *Detecting crop phenology from vegetation index time-series data by improved shape model fitting in each phenological stage*. Remote Sensing of Environment. [PDF](bibliography/Detecting%20crop%20phenology%20from%20vegetation%20index%20time-series%20data%20by%20improved%20shape%20model%20fitting.pdf)
    - **Stage-Specific Fitting:** Overcomes "unsynchronized growth" by fitting shapes to each phenological stage separately using an adaptive local window.
    - **Iterative Robustness:** Employs a custom iterative search optimization designed to bypass local optima and handle the noise common in 30m HLS data.

---

## Repository structure

```
SITS-CropMapper/
├── models/                         # Neural network definitions
│   ├── TSViT/                      #   TSViT backbone + modules
│   │   ├── architectures.py        #   TSViT, SwinTSViT, DinoTSViT
│   │   ├── module.py               #   Attention blocks, seg heads
│   │   └── swinTSViT.py
│   ├── dino_engine.py              #   DINO pre-training engine
│   ├── dino_enginev2.py            #   Segmentation fine-tuning engine
│   ├── engine.py                   #   Base training loop
│   ├── loss_functions.py           #   Focal, DINO, IID losses
│   └── metrics/numpy_metrics.py    #   Accuracy, F1, IOU
├── datasets/                       # Data loading & augmentation
│   ├── agro_satdata.py             #   NetCDF time-series reader
│   ├── dataloaders.py              #   PyTorch Datasets
│   ├── image_preprocessing.py      #   Kernel regression, Chen-SG filter
│   ├── segmentation.py             #   Segmentation target loader
│   └── transforms/                 #   Tensor/image augmentations
├── detection/                      # Inference engines
│   ├── dataset.py                  #   InputImgDataset (patch loader)
│   ├── detectors.py                #   STViTS_detector
│   ├── smf_s_class.py              #   SMFS phenology algorithm
│   └── utils.py                    #   Prediction utilities
├── utils/                          # Shared utilities
│   ├── hls_download.py             #   HLS data download & preprocessing
│   ├── gis_funs.py                 #   Spatial utilities (tiling, VI calc)
│   ├── general.py                  #   Vegetation index formulas
│   ├── plots.py                    #   Visualisation helpers
│   └── reporters.py                #   Training metrics logger
├── runs_dino_seg/                  # Trained model artefacts
│   └── run2_48px_months12_…/
│       ├── config.yml              #   Model configuration
│       ├── reporter.json           #   Training metrics
│       └── TSViTS_checkpoint_….pth #   Trained weights (ep 83)
├── data/                           # Example / reference data
├── bibliography/                   # Reference PDFs
├── demo_pipeline.py                # End-to-end demo (download → detect)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (recommended for training; CPU inference is possible but slow)
- [NASA Earthdata account](https://urs.earthdata.nasa.gov/users/new) (required for HLS download)

### 2. Environment setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (select the command matching your CUDA version)
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. NASA Earthdata credentials

Create `~/.netrc` with your NASA Earthdata login:

```
machine urs.earthdata.nasa.gov login <YOUR_USERNAME> password <YOUR_PASSWORD>
```

---

## Data

### HLS (Harmonized Landsat Sentinel-2)

| Product | Sensor | Bands used | Resolution |
|---------|--------|-----------|------------|
| HLSS30 v2.0 | Sentinel-2 MSI | B02, B03, B04, B8A, B11, Fmask | 30 m |
| HLSL30 v2.0 | Landsat 8/9 OLI | B02, B03, B04, B05, B06, Fmask | 30 m |

The model was trained on 7-channel inputs: **Blue, Green, Red, NIR, SWIR1, NDVI, GNDVI**.

### Input NetCDF format

Each patch file is a 48 × 48 pixel NetCDF with:
- **Dimensions:** `date` (time), `y` (northing), `x` (easting)
- **Variables:** `blue`, `green`, `red`, `nir`, `swir1`, `ndvi`, `gndvi`
- **Value range:** 0–10 000 (scaled ×0.0001 in the data loader) or 0–1 float

### Training data organisation

```
hls_data48/
└── segmentation/
    ├── training_input/      # Patch NetCDF files (*_patch_*.nc)
    ├── training_target/     # Corresponding label GeoTIFFs
    ├── validation_input/
    └── validation_target/
```

---

## Workflows

### 1. HLS data download

```python
from utils.hls_download import download_hls_for_area

download_hls_for_area(
    bbox=(-87.5, 13.5, -87.0, 14.0),   # (min_lon, min_lat, max_lon, max_lat)
    start_date="2022-01-01",
    end_date="2023-12-31",
    output_path="data/hls_patches",
    patch_size=48,                       # matches model input resolution
)
```

### 2. DINO Pre-training (Self-Supervised)

```bash
python training_svits_dino_model_LINUXv2.py
```

Edit the config section at the top of the script to set data paths and
hyperparameters. Pre-training saves teacher backbone weights to `runs_dino/`.

### 3. Segmentation fine-tuning

```bash
python training_svits_dino_segmentation_modelV3.py
```

Requires a pre-trained backbone (set `backbone_weight_path` in the config).
Checkpoints are saved to `runs_dino_seg/`.

### 4. End-to-End Pipeline Execution

Run the full workflow from data download to phenology export with a single command:

```bash
python demo_pipeline.py \
    --download \
    --bbox -87.7 14.3 -87.5 14.5 \
    --start_date 2023-01-01 \
    --end_date 2023-12-31 \
    --input_dir data/hls_test_run2 \
    --output_dir results_test \
    --end_year 2023 \
    --end_month 12 \
    --n_months_lc 12 \
    --n_months_cluster 12 \
    --n_months_phen 6 \
    --n_clusters 30 \
    --prefix test_hnd > /tmp/pipeline_run.log 2>&1
```

#### Parameter Explanation:
- `--download`: Triggers the HLS (Sentinel-2 + Landsat) data search and download from NASA Earthdata.
- `--bbox`: Defines the bounding box for the area of interest `[min_lon, min_lat, max_lon, max_lat]`.
- `--start_date` / `--end_date`: The time window for querying satellite granules.
- `--input_dir`: Directory where tiled 48x48 pixel NetCDF patches will be stored.
- `--output_dir`: Directory for the final GeoTIFFs and analytics.
- `--end_year` / `--end_month`: The reference date used as the anchor for the analysis windows.
- `--n_months_lc`: Months of data used for **Land-Cover segmentation**.
- `--n_months_cluster`: Months of data used for **Crop Clustering** (token extraction).
- `--n_months_phen`: Months of data used for **Phenology detection** (SMF-S fitting).
- `--n_clusters`: Number of K-Means clusters used to group crop-specific signatures.
- `--prefix`: Prefix for the output files (e.g., `test_hnd_phenology.tif`).
- `> /tmp/pipeline_run.log 2>&1`: Redirects both standard output and errors to a log file for background monitoring.

Outputs:
- `results_test/test_hnd_phenology.tif` — 4-band GeoTIFF (Greenup, Maturity, Senescence, Dormancy DOY)

### 5. Prepare training/validation split

```bash
python split_into_training_and_validation.py
```

---

## Trained model

| Setting | Value |
|---------|-------|
| Architecture | DinoTSViT + SummConv2 segmentation head |
| Backbone | ViT (6 temporal + 4 spatial layers, dim 256, 16 heads) |
| Image resolution | 48 × 48 px (patch size 2) |
| Temporal window | 12 months × 14-day composites |
| Classes | 21 (CGIAR/FAO land cover scheme) |
| Checkpoint | `runs_dino_seg/run2_48px_months12_…/TSViTS_checkpoint_iter34314_ep83.pth` |

### Loading for inference

```python
from omegaconf import OmegaConf
from detection.detectors import STViTS_detector

config = OmegaConf.load("runs_dino_seg/run2_48px_months12_…/config.yml")
config.DATASETS.paths.training_input = "path/to/patches"

detector = STViTS_detector(config, init_scheduler=False)
detector.load_weights_for_detection(
    "runs_dino_seg/run2_48px_months12_…/TSViTS_checkpoint_iter34314_ep83.pth"
)
detector.model.eval()
```

---

## Applications

The framework targets precision agriculture and food security monitoring:

- **Crop mapping** — land-cover classification distinguishing cropland from other land uses at 30 m resolution.
- **Phenology monitoring** — detecting key growth stages (green-up, peak, senescence) to estimate planting/harvest calendars.
- **Food security early warning** — identify delayed green-up or anomalous vegetation dynamics at the pixel level.
- **Yield estimation support** — phenology maps can serve as inputs to process-based crop models.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{tarasiou2023vits,
  title={{ViTs for SITS}: Vision Transformers for Satellite Image Time Series},
  author={Tarasiou, Michail and Chavez, Erik and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2023}
}

@article{liu2022detecting,
  title={Detecting crop phenology from vegetation index time-series data by improved shape model fitting in each phenological stage},
  author={Liu, Licong and Cao, Ruyin and Chen, Jin and Shen, Miaogen and Wang, Siqing and Zhou, Jie and He, Bingzhe},
  journal={Remote Sensing of Environment},
  volume={277},
  pages={113098},
  year={2022}
}
```

---

## License

This project is developed at CGIAR. Contact [andres.aguilar@cgiar.org](mailto:andres.aguilar@cgiar.org) for licensing information.
