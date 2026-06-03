# chestnut-burr-detection-from-drone

Pipeline for estimating chestnut tree burr yield from drone imagery.

---

## Overview

This repository implements an end-to-end pipeline for estimating chestnut (*Castanea* spp.) burr yield at the individual-tree level using drone imagery and YOLO object detection. The workflow proceeds from raw flight data to per-tree burr counts through four sequential modules:

1. **Flight Reconstruction** — process raw drone images into georeferenced 3D products (DSM, DTM, CHM, orthomosaic) using Agisoft Metashape
2. **Canopy Segmentation** — delineate individual tree canopies from the Canopy Height Model using marker-controlled watershed segmentation
3. **Image Selection** — back-project each canopy polygon onto the raw drone image collection and select the highest-quality image per tree
4. **Burr Detection** — detect and count burrs in each canopy image using YOLO; includes training, hyperparameter tuning, and inference modes

```
Raw drone images
      │
      ▼
[Flight Reconstruction]  ──►  DSM / DTM / CHM / Orthomosaic / Point Cloud
      │
      ▼
[Canopy Segmentation]  ──►  Per-tree canopy polygons + refined treetops
      │
      ▼
[Image Selection]  ──►  Best drone image per canopy (JSON)
      │
      ▼
[Burr Detection]  ──►  Per-tree burr count CSV
```

---

## Hardware Requirements

The full pipeline was developed and tested on Windows with the following hardware:

- **OS:** Windows (Metashape Python API is Windows-only)
- **GPU:** 8 GB VRAM minimum; 16+ GB recommended for Metashape depth map generation and YOLO training
- **RAM:** 128 GB recommended (Metashape depth map generation is memory-intensive; the reconstruction script monitors available RAM and can subdivide tasks automatically)
- **CPU:** 24-core recommended
- **Disk:** ~256 GB free space per orchard dataset (raw images + Metashape project files)

---

## Sample Data

A complete sample dataset for one chestnut orchard flight is available on Google Drive.

**Download (~42 GB):**  
[Download sample data (Google Drive)](https://drive.google.com/file/d/13qJbHO3ZU8EeesVM2tbPHkUUXSIAS_aN/view?usp=sharing)

Extract the ZIP **into the repository root** (the folder containing this README). It will populate the correct `sample_data/` subdirectories under each module. Smaller representative subsets are included in this repository for validating outputs without the full download.

---

## Environment Setup

1. **Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)**

2. **Clone this repository**

3. **Create and activate the conda environment:**
   ```powershell
   conda env create -f burr-detection.yml
   conda activate burr-detection
   ```

4. **Install Agisoft Metashape** (Windows wheel included in the repo):
   ```powershell
   pip install flight_reconstruction/Metashape-2.2.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl
   ```

5. **Activate a Metashape Professional license.** See [Agisoft documentation](https://agisoft.freshdesk.com/support/solutions/articles/31000153171-how-to-activate-metashape-license) for license activation instructions.

6. **Install CUDA-enabled PyTorch** (required for GPU-accelerated training and inference):
   ```powershell
   pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
   This installs PyTorch with CUDA 13.0 support. Adjust the index URL for your CUDA version if needed.

**Key dependencies** (from `burr-detection.yml`):

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.11 | — |
| gdal | 3.10.3 | Geospatial I/O |
| geopandas | 1.1.1 | Vector geometry |
| rasterio | 1.4.3 | Raster I/O |
| scikit-image | 0.25.2 | Image processing, watershed |
| shapely | 2.1.2 | Polygon operations |
| psutil | 7.0.0 | Memory estimation (Metashape subdivision) |
| easyidp | latest | 3D back-projection, Metashape wrapper |
| ultralytics | latest | YOLO training and inference |
| ray[tune] | latest | Distributed hyperparameter tuning |
| optuna | latest | Bayesian hyperparameter search |
| pyexiv2 | latest | XMP metadata reading (DJI drone tags) |

---

## Repository Structure

```
chestnut-burr-detection-from-drone/
├── burr-detection.yml                  # Conda environment
├── yolo11n.pt                          # YOLO11 nano base weights
├── yolov8s.pt                          # YOLOv8 small base weights (default training_params)
│
├── flight_reconstruction/
│   ├── reconstruction.py               # Metashape automation pipeline
│   ├── utils.py                        # GPS, export, filtering utilities
│   ├── config.yml                      # Reconstruction parameters
│   ├── Metashape-2.2.2-...-win_amd64.whl
│   └── sample_data/20230823_Orchard4/
│       ├── DJI_*.JPG                   # Raw drone images (DJI Mavic 3M)
│       ├── DJI_*_PPKNAV.nav/.obs/.bin  # PPK GNSS navigation files
│       ├── DJI_*_Timestamp.MRK
│       └── outputs/                    # Reconstruction products (see Step 1)
│
├── canopy_segmentation/
│   ├── segmentation.py                 # Watershed segmentation
│   └── sample_data/
│       ├── inputs/                     # Tree markers + boundary shapefiles
│       └── outputs/                    # Canopy polygons + treetops shapefiles
│
├── image_selection/
│   ├── canopy_to_image.py              # Best-image selection per canopy
│   └── sample_data/outputs/
│       └── best_image_selections.json
│
└── burr_detection/
    ├── detection.py                    # Entry point: train / tune / inference
    ├── training.py                     # Multi-step progressive YOLO training
    ├── tuning.py                       # Ray Tune + Optuna hyperparameter search
    ├── inference.py                    # Tile-and-detect inference pipeline
    ├── dataset.py                      # Dataset splitting and canopy tiling
    ├── utils.py                        # NMS, evaluation, plotting, metrics
    ├── config.yml                      # Training, tuning, and inference config
    └── sample_data/
        ├── training/inputs/            # Labeled images + YOLO-format labels
        ├── training/outputs/           # Training and tuning run artifacts
        └── inference/outputs/          # Inference run artifacts
```

---

## Step 1: Flight Reconstruction

**Script:** `flight_reconstruction/reconstruction.py`  
**Config:** `flight_reconstruction/config.yml`

### What It Does

Automates Agisoft Metashape to generate georeferenced 3D products from raw drone images:

1. Loads images, computes per-image quality scores, and removes images below threshold (default 0.70)
2. Matches tie points across images, aligns cameras, and optimizes camera intrinsics and extrinsics using RTK/GPS reference coordinates
3. Applies USGS-recommended three-stage tie point filtering: reconstruction uncertainty → projection accuracy → reprojection error (percentile-based thresholds)
4. Builds dense depth maps and a colored, confidence-weighted point cloud
5. Classifies ground points (angle/distance/cell-size thresholds) and generates DSM, DTM, and CHM (DSM − DTM)
6. Produces an orthorectified mosaic with mosaic blending and ghosting filter
7. Exports all products as GeoTIFF/LAS with UTM projection auto-detected from the first image's GPS coordinates

Each processing step checks whether it already completed before running; the pipeline is **resumable** from any point by re-running the same command. GPU selection automatically restricts to devices with ≥8 GB VRAM and disables CPU during GPU-intensive steps to reduce memory fragmentation.

### Inputs

| Input | Path | Notes |
|-------|------|-------|
| Raw images | `<folder>/*.JPG` | DJI Mavic 3M; RTK accuracy read from XMP |
| PPK GNSS files | `<folder>/*.nav`, `*.obs`, `*.bin`, `*.MRK` | DJI PPK navigation |
| Config | `flight_reconstruction/config.yml` | See config reference below |

### Outputs

All written to `<folder>/outputs/`:

| File | Description |
|------|-------------|
| `<name>.psx` | Metashape project file (resumable) |
| `<name>_dsm.tif` | Digital Surface Model (GeoTIFF, UTM) |
| `<name>_dtm.tif` | Digital Terrain Model (ground points only) |
| `<name>_chm.tif` | Canopy Height Model = DSM − DTM |
| `<name>_orthomosaic.tif` | Georeferenced aerial image (GeoTIFF) |
| `<name>_point_cloud.las` | Dense point cloud (RGB + confidence) |
| `<name>_camera_positions.txt` | Camera OPK exterior orientations |
| `<name>_report.pdf` | Metashape processing report |

### Usage

```powershell
conda activate burr-detection
$env:METASHAPE_LICENSE_KEY="XXXXX-XXXXX-XXXXX-XXXXX-XXXXX"

python -m flight_reconstruction.reconstruction `
    --config "flight_reconstruction/config.yml" `
    --folder "flight_reconstruction/sample_data/20230823_Orchard4"
```

### Key Config Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `gps.use_rtk` | `true` | Read RTK accuracy from DJI XMP metadata |
| `image_quality.quality_threshold` | `0.70` | Discard images below this Metashape quality score (0–1) |
| `photo_matching.downscale` | `1` | Matching resolution; 1 = full resolution |
| `photo_matching.keypoint_limit` | `80000` | Max keypoints per image |
| `tie_point_filtering.reconstruction_uncertainty.percentile` | `20` | Remove worst 20% by reconstruction uncertainty |
| `tie_point_filtering.projection_accuracy.percentile` | `30` | Remove worst 30% by projection accuracy |
| `tie_point_filtering.reprojection_error.percentile` | `5` | Remove worst 5% by reprojection error |
| `depth_maps.downscale` | `2` | Depth map resolution; 2 = half resolution |
| `depth_maps.filter_mode` | `disabled` | Depth filtering: `mild`, `moderate`, `aggressive`, or `disabled` |
| `classify_ground_points.max_angle` | `15.0` | Maximum slope angle (°) for ground classification |
| `dem.resolution` | `0` | Output raster resolution; `0` = auto from GSD |
| `orthomosaic.blending_mode` | `mosaic` | Blending algorithm for orthomosaic |

### Limitations

- Requires Agisoft Metashape Professional license (proprietary; not open-source)
- Windows-only (Metashape Python API)
- Assumes DJI drone with standard XMP tags for RTK accuracy; other manufacturers use different tag names
- Single-chunk processing; multi-chunk workflows are not automated
- `subdivide_task` is `false` by default in all config sections; enable for datasets that exceed available RAM

---

## Step 2: Canopy Segmentation

**Script:** `canopy_segmentation/segmentation.py`

### What It Does

Segments individual tree canopies from the CHM using **marker-controlled watershed segmentation** with an adaptive proximity penalty:

1. Loads CHM raster and tree marker points (manually digitized from imagery or collected with RTK GNSS)
2. Refines each marker to the local CHM maximum within a buffer radius (corrects imprecise field placement)
3. Constructs a watershed cost surface from: inverted CHM height, CHM gradient magnitude, and a proximity penalty that scales each tree's boundary distance by its height relative to neighboring trees — preventing over-segmentation in dense stands
4. Runs scikit-image watershed with each refined marker as a labeled seed
5. Removes segments smaller than 5 m² and removes any segments that clip the orchard boundary polygon by more than 0.5 m
6. Exports canopy polygons and refined treetop points as shapefiles

### Inputs

| Argument | Description |
|----------|-------------|
| `--chm` | CHM GeoTIFF from flight reconstruction |
| `--tree-markers` | Point shapefile of approximate tree locations |
| `--extent` (optional) | Boundary polygon; segments overlapping it by >0.5 m are removed |

### Outputs

Written to `--outdir`:

| File | Attributes |
|------|------------|
| `<name>_Canopies.shp` | `tree_id`, `area_m2`, `max_h` (m), `mean_h` (m) |
| `<name>_Treetops.shp` | `tree_id`, height at refined location |

### Usage

```powershell
conda activate burr-detection

python -m canopy_segmentation.segmentation `
    --chm "flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_chm.tif" `
    --tree-markers "canopy_segmentation/sample_data/inputs/20230823_Orchard4_tree_markers.shp" `
    --outdir "canopy_segmentation/sample_data/outputs/" `
    --min-height 1.75 `
    --buffer-size 0.5 `
    --id-column Tree_ID `
    --extent "canopy_segmentation/sample_data/inputs/20230823_Orchard4_boundary.shp"
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--chm` | yes | — | Path to CHM GeoTIFF |
| `--tree-markers` | yes | — | Tree marker point shapefile |
| `--outdir` | yes | — | Output directory |
| `--min-height` | no | `1.75` | Minimum CHM height (m) for the segmentation mask |
| `--buffer-size` | no | `1.0` | Buffer radius (m) for local maxima refinement |
| `--id-column` | no | first non-geometry column | Column name for tree IDs in marker file |
| `--extent` | no | — | Boundary polygon shapefile |

*Tree markers can be manually digitized using leaf-off imagery or collected with an RTK GNSS receiver in the field.*

### Limitations

- One marker = one canopy segment; touching or overlapping crowns are not automatically separated without separate markers
- Pairwise inter-tree distance computation scales O(n²); performance degrades above ~10,000 trees
- CHM and markers must share a coordinate reference system (geopandas will attempt auto-reprojection if they differ)

---

## Step 3: Image Selection

**Script:** `image_selection/canopy_to_image.py`

### What It Does

Selects the highest-quality raw drone image for each canopy polygon using Metashape's calibrated 3D camera model for back-projection:

1. Back-projects each canopy polygon onto all candidate raw images using the Metashape camera model (accounts for lens distortion, orientation, and 3D position)
2. Retains images where the polygon projects within a 10 m distance threshold on the image plane
3. Scores each candidate image:
   - **Exposure:** mean pixel value in projected region; filtered to 0.15–0.85 (discards over/underexposed images)
   - **Gimbal pitch:** read from DJI XMP tag; images near nadir (~−90° ± 5°) are preferred
   - **Sharpness:** variance of the Laplacian over the projected canopy region
   - **Contrast:** standard deviation of pixel values in the projected region
4. Selects the image with highest sharpness (within 85% of the per-canopy maximum), breaking ties by contrast

### Inputs

| Argument | Description |
|----------|-------------|
| `--canopy_shapefile` | Canopy polygon shapefile from segmentation |
| `--dsm` | DSM raster (for 3D Z-value assignment during back-projection) |
| `--metashape_project` | Metashape `.psx` project with aligned cameras |
| `--raw_images` | Folder containing raw drone images |
| `--output` | Output directory |

### Output

`best_image_selections.json` — maps each tree ID to its selected image and projected canopy polygon:

```json
{
  "338": {
    "image_path": "flight_reconstruction/sample_data/20230823_Orchard4/DJI_20230823122017_0102_D.JPG",
    "polygon_coords": [[2521.04, 2099.90], [2508.38, 2099.97], ...]
  }
}
```

Polygon coordinates are in **image pixel space** (not geographic coordinates).

### Usage

```powershell
conda activate burr-detection

python -m image_selection.canopy_to_image `
    --canopy_shapefile "canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp" `
    --dsm "flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif" `
    --metashape_project "flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx" `
    --raw_images "flight_reconstruction/sample_data/20230823_Orchard4" `
    --output "image_selection/sample_data/outputs"
```

### Limitations

- Requires a fully-aligned Metashape project (cameras must already be positioned from Step 1)
- Assumes DJI XMP tags for gimbal pitch; other drone brands use different metadata schemas
- The 10 m image-plane distance threshold may need adjustment for different flight altitudes or focal lengths
- Does not check for occlusion by neighboring trees or branches

---

## Step 4: Burr Detection

**Entry point:** `burr_detection/detection.py`  
**Config:** `burr_detection/config.yml`

Three modes are available: `train`, `tune`, and `inference`. All are accessed through the same entry point:

```powershell
conda activate burr-detection

python -m burr_detection.detection `
    --mode <train|tune|inference> `
    --config "burr_detection/config.yml" `
    --plot-mode <all|subset|none>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `inference` | `train`, `tune`, or `inference` |
| `--config` | `burr_detection/config.yml` | Path to YAML config file |
| `--plot-mode` | `subset` | `all` (every image), `subset` (15 random), `none` |

---

### Labeled Dataset Format

Training and tuning expect YOLO-format labeled data organized as:

```
burr_detection/sample_data/training/inputs/
├── images/    # .jpg or .png image tiles (224×224 recommended)
└── labels/    # matching .txt files, one per image
               # each line: <class_id> <cx> <cy> <w> <h>  (normalized 0–1)
```

The config `data.training_dir` points to the `inputs/` parent directory:

```yaml
data:
  training_dir: burr_detection/sample_data/training/inputs
```

Train/val/test splits (70/20/10 by default, seed 666) are created automatically at the start of each training or tuning run. No labeled dataset is distributed with this repository; users must provide their own YOLO-format annotations.

---

### 4a. Training Mode

Trains a YOLO model using **multi-step progressive gradient accumulation**, which simulates large effective batch sizes without exceeding GPU memory:

| Step | Physical Batch | Accumulate | Effective Batch | Max Epochs | Patience | Layers Unfrozen |
|------|---------------|------------|-----------------|------------|----------|-----------------|
| 1 | 8 | 1 | 8 | 50 | 10 | Head only |
| 2 | 8 | 4 | 32 | 50 | 15 | Head + 3 layers |
| 3 | 8 | 16 | 128 | 50 | 20 | Head + 4 layers |
| 4 | 8 | 64 | 512 | 50 | 25 | All layers |

The learning rate is scaled as `lr = lr0 × √(effective_batch / 4)` at each step. Progressive unfreezing starts from the detection head and gradually extends to the backbone. The best checkpoint across all steps and epochs is selected by F1 score on the validation set and saved as `best_model_weights.pt`.

```powershell
python -m burr_detection.detection --mode train `
    --config "burr_detection/config.yml" `
    --plot-mode subset
```

**Outputs** — written to `burr_detection/sample_data/training/outputs/training_<timestamp>/`:

| File | Description |
|------|-------------|
| `best_model_weights.pt` | Best checkpoint by validation F1 across all steps |
| `training_metrics.csv` | Per-epoch: learning rate, losses, precision, recall, mAP50, F1 |
| `test_results.csv` | Held-out test set: precision, recall, F1, mAP50, inference time |
| `train_step{1-4}/` | Per-step YOLO run directories (weights, results.csv, plots) |
| `prediction_plots/` | Side-by-side ground truth vs. prediction visualizations |

**Performance**

The detector is **YOLOv8s** (the `training_params` default; the tuning search may also select `yolo11n/s` or `yolov8n`).

*Full dataset (production model)* — YOLOv8s on the held-out test split (174 images, 1,936 burrs), ~3.6 ms/img inference on an RTX 4060 Laptop GPU:

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|-----------|--------|----|-------|----------|
| 0.818 | 0.748 | 0.782 | 0.820 | 0.431 |

*The full multi-orchard training set is proprietary and not distributed here.*

*Included sample dataset* — the single-orchard sample is small, so test metrics vary noticeably run-to-run (mAP50 ≈ 0.70–0.80). A representative run:

| Precision | Recall | F1 | mAP50 |
|-----------|--------|----|-------|
| 0.791 | 0.743 | 0.767 | 0.801 |

**Key `training_params` in `config.yml`** (default / tuned starting point):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_size` | `yolov8s.pt` | Base model |
| `imgsz` | `416` | Input image size |
| `optimizer` | `SGD` | — |
| `lr0` | `0.00754` | Initial learning rate |
| `momentum` | `0.861` | SGD momentum |
| `weight_decay` | `0.00800` | — |
| `dropout` | `0.113` | — |
| `mosaic` | `0.047` | Mosaic augmentation probability |

---

### 4b. Tuning Mode

Searches the hyperparameter space using **Ray Tune with Optuna (Bayesian) search** and ASHA early stopping:

- Default: 50 trials, up to 2 concurrent
- ASHA scheduler prunes underperforming trials after a 10-epoch grace period
- Primary optimization metric: minimum validation loss (for scheduling); final model selected by maximum validation F1
- The `training_params` entry in the config is used as a warm-start point for Optuna

```powershell
python -m burr_detection.detection --mode tune `
    --config "burr_detection/config.yml" `
    --plot-mode subset
```

**Tuning space** (from `burr_detection/config.yml`):

| Parameter | Range | Distribution |
|-----------|-------|-------------|
| `model_size` | yolo11n, yolo11s, yolov8n, yolov8s | choice |
| `imgsz` | 224, 320, 416 | choice |
| `optimizer` | AdamW, SGD, Adam | choice |
| `lr0` | [0.0005, 0.01] | log-uniform |
| `lrf` | [0.001, 0.1] | log-uniform |
| `momentum` | [0.85, 0.98] | uniform |
| `weight_decay` | [0.0001, 0.01] | log-uniform |
| `warmup_epochs` | 2, 3, 4, 5 | choice |
| `box_gain` | [12.0, 20.0] | uniform |
| `cls_gain` | [0.5, 2.0] | uniform |
| `dfl_gain` | [1.5, 3.0] | uniform |
| `dropout` | [0.0, 0.2] | uniform |
| augmentation params (hsv_h/s/v, degrees, scale, shear, mosaic, mixup, copy_paste, perspective) | see config | uniform |

**Outputs** — written to `burr_detection/sample_data/training/outputs/tuning_<timestamp>/`:

| File | Description |
|------|-------------|
| `best_<model>_model.pt` | Best model weights |
| `best_trial_config.json` | Hyperparameters for the best trial |
| `all_tuning_history.csv` | Metrics for all 50 trials |
| `best_trial_training_history.csv` | Per-epoch metrics for the best trial |
| `test_results.csv` | Test set evaluation of best model |
| `prediction_plots/` | Prediction visualizations |

Tuning checkpoints per-epoch to `<trial_dir>/checkpoint_<epoch>/` (model weights + state), allowing trials to resume after interruption.

---

### 4c. Inference Mode

Detects burrs on unlabeled canopy images using a tile-and-reconstruct strategy:

For each tree in `best_image_selections.json`:
1. Crop the canopy polygon region from the full drone image (mask outside region to black)
2. Tile the cropped canopy into 224×224 patches with 20% overlap (stride = 179 px); skip all-black tiles
3. Run YOLO inference on each tile (confidence threshold 0.5 by default)
4. Reconstruct tile-space detections back to full canopy coordinates by adding tile offsets
5. Apply NMS (IoU threshold 0.45) to suppress duplicate detections from tile overlap
6. Aggregate per tree: total detections, average confidence, bounding box coordinates

The model is **auto-detected** from the most recent `training_*/` or `tuning_*/` output directory when `inference.model_path` is `null`. To use a specific model, set `inference.model_path` to an explicit path.

```powershell
python -m burr_detection.detection --mode inference `
    --config "burr_detection/config.yml" `
    --plot-mode subset
```

**Inference config keys:**

```yaml
inference:
  model_path: null          # null = auto-detect most recent training/tuning output
  conf_threshold: 0.5
  iou_threshold: 0.45

data:
  image_selections: image_selection/sample_data/outputs/best_image_selections.json
```

**Outputs** — written to `burr_detection/sample_data/inference/outputs/inference_<timestamp>/`:

| File | Description |
|------|-------------|
| `tree_burr_detections.csv` | Per tree: `tree_id`, `image_path`, `total_detections`, `avg_confidence`, `detection_coords` |
| `detection_summary.txt` | Summary stats: mean/min/max burrs per tree, overall confidence, model info |
| `preprocessed_trees/` | Cropped canopy images (for visual verification of masking) |
| `prediction_plots/` | Detection visualizations with bounding boxes |

**Sample inference results** from included sample run (`best_yolo11s_model.pt`, conf=0.5, IoU=0.45):

| Trees Processed | Total Detections | Mean Burrs/Tree | Min | Max | Avg Confidence |
|-----------------|-----------------|-----------------|-----|-----|---------------|
| 516 | 71,963 | 139 | 0 | 730 | 0.605 |

---

## Limitations and Known Assumptions

- **Platform:** Windows-only due to the Agisoft Metashape Python API (Steps 1 and 3)
- **Drone hardware:** Optimized for DJI Mavic 3M; RTK accuracy parsing and gimbal pitch metadata both depend on DJI-specific XMP tags
- **Single class:** The detector is configured for one object class (burr); multi-class use requires label and config changes
- **Canopy segmentation:** One-to-one mapping between markers and canopies; touching or overlapping crowns are not automatically split without separate markers per crown and, as such, outputs usually require manual cleanup. ## TODO: instance segmentation from point cloud
- **Image selection:** Does not account for occlusion of a canopy by adjacent trees or branches
- **Inference tiling:** Tile size (224×224) and overlap (20%) are fixed and require source edits to change
- **Metashape license:** Agisoft Metashape Professional is required for Steps 1 and 3; the included wheel is version 2.2.2

---

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---

## Acknowledgments

This project uses [easyidp](https://easyidp.readthedocs.io/en/latest/index.html) for 3D back-projection in the image selection step. If you use this pipeline in your research, please cite:

Wang, Haozhou and Duan, Yulin and Shi, Yun and Kato, Yoichiro and Ninomiya, Seish and Guo, Wei. "EasyIDP: A Python Package for Intermediate Data Processing in UAV-Based Plant Phenotyping." *Remote Sensing* 13, no. 13 (2021): 2622. https://doi.org/10.3390/rs13132622
