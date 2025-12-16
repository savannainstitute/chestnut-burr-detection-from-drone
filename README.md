# chestnut-burr-detection-from-drone

Pipeline for estimating chestnut tree burr yield using drone imagery.

---

## Overview

This repository contains a modular pipeline for estimating chestnut tree burr yield from drone imagery. The pipeline consists of the following steps:

1. **Flight Reconstruction**: Generate georeferenced orthomosaics (UTM projections), elevation surfaces (DSM/DTM/CHM), and point clouds (SfM) from drone imagery using Agisoft Metashape.
2. **Canopy Segmentation**: Segment individual tree canopies from the CHM using a proximity-based watershed algorithm with control markers.
3. **Image Selection**: Select the best drone image for each segmented canopy using image quality and sensor parameters.
4. **Burr Detection**: Detect and count burrs for each tree using YOLO. Also supports training and hyperparameter tuning.

This README documents each step. Sample data for a full test run is available via Google Drive (see below).

---

## Hardware Requirements

- Windows OS
- Dedicated GPU with at least 8 GB VRAM (16+ GB recommended)
- 128 GB RAM
- 24-core CPU recommended
- ~256 GB free disk space per orchard dataset

---


## Sample Data

Sample data for a chestnut orchard is available via Google Drive as a ZIP file.

**Instructions:**
1. Download the sample data ZIP from Google Drive:  
   [Download sample data (Google Drive)](https://drive.google.com/file/d/13qJbHO3ZU8EeesVM2tbPHkUUXSIAS_aN/view?usp=sharing)
2. Extract the ZIP file **into the root of this repository** (the folder containing this README). Overwrite any existing folders if prompted.
3. The sample data will be placed in the correct subdirectories automatically.

---

## Environment Setup and Prerequisites

1. **Install Miniconda:** https://docs.conda.io/en/latest/miniconda.html
2. **Clone this repository.**
3. **Create and activate the conda environment:**
    ```
    conda env create -f burr-detection.yml
    conda activate burr-detection
    ```
4. **Install Agisoft Metashape from the provided wheel file:**
    ```
    pip3 install flight_reconstruction/Metashape-2.2.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl
    ```
5. **Activate a Metashape Professional license.** (See Agisoft documentation)
6. **Download and extract the sample data from Google Drive** (see above).

---

## Step 1: Flight Reconstruction

Generate georeferenced orthomosaics, point clouds, and elevation surfaces from drone imagery using Agisoft Metashape. Optimized for DJI Mavic 3M. 

- **Script:** `flight_reconstruction/reconstruction.py`
- **Config:** `flight_reconstruction/config.yml`
- **Sample Input Data:**  
    - Raw images and navigation files (for RTK): `flight_reconstruction/sample_data/20230823_Orchard4/`
- **Outputs:** 
    - Metashape project: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx` 
    - Camera positions: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_camera_positions.txt`
    - Point cloud: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_point_cloud.las` 
    - DSM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif` 
    - DTM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dtm.tif` 
    - CHM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_chm.tif` 
    - Orthomosaic: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_orthomosaic.tif`
    - Report: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_report.pdf` 

**Usage (PowerShell):**
1. Activate the conda environment:
    ```
    conda activate burr-detection
    ```
2. Set your Metashape license key as an environment variable (replace with your actual key):
    ```
    $env:METASHAPE_LICENSE_KEY="XXXXX-XXXXX-XXXXX-XXXXX-XXXXX"
    ```
3. Run the reconstruction script:
    ```
    python -m flight_reconstruction.reconstruction `
        --config "flight_reconstruction/config.yml" `
        --folder "flight_reconstruction/sample_data/20230823_Orchard4"
    ```

**Arguments:**
- `--config` (required): Path to the YAML configuration file
- `--folder` (required): Path to the input folder containing raw images and navigation files

---

## Step 2: Canopy Segmentation

Segment individual tree canopies from the CHM using a proximity-based watershed algorithm with control markers.

*Note: Tree markers can be manually digitized using leaf-off imagery or collected with an RTK GNSS receiver.*

- **Script:** `canopy_segmentation/segmentation.py`
- **Sample Input Data:**  
    - CHM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_chm.tif`
    - Tree markers: `canopy_segmentation/sample_data/inputs/20230823_Orchard4_tree_markers.shp`  
    - Boundary shapefile (optional): `canopy_segmentation/sample_data/inputs/20230823_Orchard4_boundary.shp`
- **Outputs:**  
    - Canopies: `canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp` (polygons)
    - Treetops: `canopy_segmentation/sample_data/outputs/20230823_Orchard4_Treetops.shp` (points)

**Usage (PowerShell):**
1. Activate the conda environment:
    ```
    conda activate burr-detection
    ```
2. Run the segmentation script:
    ```
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
- `--chm` (required): Path to CHM raster
- `--tree-markers` (required): Tree marker shapefile
- `--outdir` (required): Output directory
- `--min-height` (optional): Minimum CHM height for segmentation (default: 1.75)
- `--buffer-size` (optional): Buffer (meters) for refining tree markers to local maxima (default: 0.5)
- `--id-column` (optional): Column name for original tree IDs in marker shapefile (e.g., Tree_ID). If not provided, the first non-geometry column will be used.
- `--extent` (optional): Polygon shapefile for processing extent

---

## Step 3: Image Selection

Select the best drone image for each segmented canopy using image quality and sensor parameters.

- **Script:** `image_selection/canopy_to_image.py`
- **Sample Input Data:**  
    - Canopy polygons: `canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp`  
    - DSM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif`  
    - Metashape project: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx`  
    - Raw images: `flight_reconstruction/sample_data/20230823_Orchard4/`
- **Outputs:**  
    - Best image selections: `image_selection/sample_data/outputs/best_image_selections.json`

**Usage (PowerShell):**
1. Activate the conda environment:
    ```
    conda activate burr-detection
    ```
2. Run the image selection script:
    ```
    python -m image_selection.canopy_to_image `
        --canopy_shapefile "canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp" `
        --dsm "flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif" `
        --metashape_project "flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx" `
        --raw_images "flight_reconstruction/sample_data/20230823_Orchard4" `
        --output "image_selection/sample_data/outputs"
    ```

**Arguments:**
- `--canopy_shapefile` (required): Path to the canopy polygons shapefile (e.g., output from segmentation step)
- `--dsm` (required): Path to the DSM raster file
- `--metashape_project` (required): Path to the Metashape project file (.psx)
- `--raw_images` (required): Path to the folder containing raw drone images
- `--output` (required): Output directory for results (best image-canopy mapping JSON)

---

## Step 4: Burr Detection

Detect and count burrs for each tree using YOLO object detection models. Supports training, hyperparameter tuning, and inference on new drone imagery.

- **Main Sctipt:** `burr_detection/detection.py` 
- **Supplemental scripts:** `burr_detection/training.py`, `burr_detection/tuning.py`, `burr_detection/dataset.py`, `burr_detection/utils.py`
- **Config:** `burr_detection/config.yml`
- **Sample Input Data (Tuning, Training):**
    - Training images/labels: `burr_detection/sample_data/training/inputs/images/`, `burr_detection/sample_data/training/inputs/labels/`
- **Sample Input Data (Inference):**
    - Canopy selections: `image_selection/sample_data/outputs/best_image_selections.json`
- **Outputs (Training):**
    - Best model: `burr_detection/sample_data/training/outputs/training_<timestamp>/best_training_weights.pt`
    - Training history: `burr_detection/sample_data/training/outputs/training_<timestamp>/train_step*/`
    - Test results: `burr_detection/sample_data/training/outputs/training_<timestamp>/test_results.csv`
    - Dataset config: `burr_detection/sample_data/training/outputs/training_<timestamp>/dataset.yaml`
- **Outputs (Tuning):**
    - Best model: `burr_detection/sample_data/training/outputs/tuning_<timestamp>/best_<model>_model.pt`
    - Tuning history: `burr_detection/sample_data/training/outputs/tuning_<timestamp>/tuning_history.csv`
    - Best model training history: `burr_detection/sample_data/training/outputs/tuning_<timestamp>/best_trial_training_history.csv`
    - Dataset config: `burr_detection/sample_data/training/outputs/tuning_<timestamp>/dataset.yaml`
- **Outputs (Inference):**
    - Burr counts per tree: `burr_detection/sample_data/inference/outputs/inference_<timestamp>/tree_burr_detections.csv`
    - Detection summary: `burr_detection/sample_data/inference/outputs/inference_<timestamp>/detection_summary.txt`
    - Cropped canopies: `burr_detection/sample_data/inference/outputs/inference_<timestamp>/preprocessed_trees/`
    - Prediction plots: `burr_detection/sample_data/inference/outputs/inference_<timestamp>/prediction_plots/`


**Usage (PowerShell):**
1. Activate the conda environment:
    ```
    conda activate burr-detection
    ```

**Note:** If using a CUDA-enabled GPU (recommended), install PyTorch with CUDA from wheel before running:
    ```
    pip3 install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```

### 4a. Training Mode

2. Train a YOLO model using multi-step progressive training:
    ```
    python -m burr_detection.detection --mode train `
        --config "burr_detection/config.yml" `
        --plot-mode subset
    ```

### 4b. Tuning Mode

2. Optimize hyperparameters using Ray Tune with Optuna search:
    ```
    python -m burr_detection.detection --mode tune `
        --config "burr_detection/config.yml" `
        --plot-mode subset
    ```

### 4c. Inference Mode

2. Detect burrs on unlabeled drone imagery:
    ```
    python -m burr_detection.detection --mode inference `
        --config "burr_detection/config.yml" `
        --plot-mode subset
    ```

**Arguments:**
- `--mode` (optional): `tune`, `train`, or `inference` (default: inference)
- `--config` (optional): `/path/to/config/yml` (default: burr_detection/config.yml)
- `--plot-mode` (optional): `all`, `subset` (15 random), or `none` (default: subset)

**Image Selections JSON Format:**
```json
{
  "tree_001": {
    "image_path": "/path/to/drone_image.jpg",
    "polygon_coords": [[x1, y1], [x2, y2], [x3, y3], ...]
  },
  "tree_002": { ... }
}
```

**Inference Pipeline:**
1. Crop canopy regions from drone images using polygon masks
2. Tile cropped canopies into overlapping 224×224 patches
3. Run YOLO detection on each tile
4. Reconstruct detections into full canopy
5. Apply NMS to remove duplicates from tile overlap
6. Aggregate burr counts and confidences per tree
7. Plot predictions (optional) and save results as CSV

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

This project uses [easyidp](https://easyidp.readthedocs.io/en/latest/index.html) for intermediate data processing. If you use this pipeline in your research, please cite:

Wang, Haozhou and Duan, Yulin and Shi, Yun and Kato, Yoichiro and Ninomiya, Seish and Guo, Wei. "EasyIDP: A Python Package for Intermediate Data Processing in UAV-Based Plant Phenotyping." Remote Sensing 13, no. 13 (2021): 2622. https://doi.org/10.3390/rs13132622