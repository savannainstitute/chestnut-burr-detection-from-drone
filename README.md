# chestnut-burr-detection-from-drone

Pipeline for estimating chestnut tree burr yield using drone imagery.

---

## Overview

This repository contains a modular pipeline for estimating chestnut tree burr yield from drone imagery. The pipeline consists of the following steps:

1. **Flight Reconstruction**: Generate georeferenced orthomosaics, digital surface models (DSM), and point clouds from drone imagery using Agisoft Metashape.
2. **Canopy Segmentation**: Segment individual tree canopies from the DSM using proximity-based and morphological methods.
3. **Image Selection**: Select the best drone image for each segmented canopy using image quality and sensor parameters.
4. **Burr Detection**: (Coming soon) Detect and count burrs for each tree.

This README documents steps 1–3. Sample data for a full test run is available (see below).

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
   [Download sample data (Google Drive)](https://drive.google.com/file/d/1oHYCEbzDCp7JmoOPDLNqplWT2zbMd--1/view?usp=sharing)
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
    pip install flight_reconstruction/Metashape-2.2.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl
    ```
5. **Activate a Metashape Professional license.** (See Agisoft documentation.)
6. **Download and extract the sample data from Google Drive** (see above), preserving the folder structure.

---

## Step 1: Flight Reconstruction

Generate georeferenced (UTM projections) orthomosaics, elevation surfaces, and point clouds from drone imagery using Agisoft Metashape.

- **Script:** `flight_reconstruction/reconstruction.py`
- **Config:** `flight_reconstruction/config.yml`
- **Sample Data:**  
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

**Usage:**
1. Activate the conda environment:
    ```
    conda activate burr-detection
    ```
2. Set your Metashape license key as an environment variable (replace with your actual key):
    ```
    set METASHAPE_LICENSE_KEY=XXXXX-XXXXX-XXXXX-XXXXX-XXXXX
    ```
3. Run the reconstruction script:
    ```
    python flight_reconstruction/reconstruction.py --config flight_reconstruction/config.yml --folders '["sample_data/20230823_Orchard4"]'
    ```

---


## Step 2: Canopy Segmentation

Segment individual tree canopies from the CHM using a proximity-based watershed algorithm with control markers.

*Note: In future releases, this step will be replaced with the SEConD model, which will allow canopy segmentation without the need for manually created control markers.*

- **Script:** `canopy_segmentation/segmentation.py`
- **Sample Data:**  
    - CHM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_chm.tif`
    - Tree markers: `canopy_segmentation/sample_data/inputs/20230823_Orchard4_tree_markers.shp`  
        (Tree markers were manually digitized using a leaf-off canopy height model. Alternatively, tree locations can be collected in the field using RTK GPS.)
    - Boundary shapefile: `canopy_segmentation/sample_data/inputs/20230823_Orchard4_boundary.shp`
- **Outputs:**  
    - Canopies: `canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp` (polygons)
    - Treetops: `canopy_segmentation/sample_data/outputs/20230823_Orchard4_Treetops.shp` (points)

**Usage:**
```
python canopy_segmentation/segmentation.py \
    --chm flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_chm.tif \
    --tree-markers canopy_segmentation/sample_data/inputs/20230823_Orchard4_tree_markers.shp
    --extent canopy_segmentation/sample_data/inputs/20230823_Orchard4_boundary.shp \
    --outdir canopy_segmentation/sample_data/outputs/
```

---

## Step 3: Image Selection

Select the best drone image for each segmented canopy using image quality and sensor parameters.

- **Script:** `image_selection/canopy_to_image.py`
- **Sample Data:**  
    - Canopy polygons: `canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp`  
    - DSM: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif`  
    - Metashape project: `flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx`  
    - Raw images: `flight_reconstruction/sample_data/20230823_Orchard4/`
- **Outputs:**  
    - Sorted image/canopy mapping: `image_selection/easyidp_outputs/img_dict_sort.json`  
    - Best cropped images: `image_selection/easyidp_outputs/best_cropped_images/`

**Usage:**
```
python image_selection/canopy_to_image.py \
    --canopy_shapefile canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp \
    --dsm flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif \
    --metashape_project flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx \
    --raw_images flight_reconstruction/sample_data/20230823_Orchard4 \
    --output image_selection/easyidp_outputs
```

---


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

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
