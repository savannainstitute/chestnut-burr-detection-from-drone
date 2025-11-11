
import easyidp as idp
import geopandas as gpd
import os
import json
import numpy as np
import pandas as pd
from skimage import io, color, filters, draw
import pyexiv2
import argparse

# --- Parse CLI arguments ---
parser = argparse.ArgumentParser(description="Select best images for each canopy using 3D geometry and image metadata.")
parser.add_argument("--canopy_shapefile", type=str, required=True, help="Path to canopy polygons shapefile (.shp)")
parser.add_argument("--dsm", type=str, required=True, help="Path to DSM raster (.tif)")
parser.add_argument("--metashape_project", type=str, required=True, help="Path to Metashape project file (.psx)")
parser.add_argument("--raw_images", type=str, required=True, help="Path to folder containing raw images")
parser.add_argument("--output", type=str, required=True, help="Path to output folder")
args = parser.parse_args()

shapefile_path = args.canopy_shapefile
dsm_path = args.dsm
metashape_path = args.metashape_project
raw_images_folder_path = args.raw_images
output_folder_path = args.output

# --- Convert shapefile to geojson ---
gdf = gpd.read_file(shapefile_path)
geojson_path = shapefile_path.replace(".shp", ".geojson")
gdf.to_file(geojson_path, driver="GeoJSON")

# --- Load canopy polygons ---
roi = idp.ROI()
roi.read_geojson(geojson_path, name_field="tree_id")

# --- Load DSM and add z values to polygons ---
dsm = idp.GeoTiff(dsm_path)
roi.get_z_from_dsm(dsm)

# --- Read 3D reconstruction project from Metashape ---
ms = idp.Metashape(project_path=metashape_path, chunk_id=0)

# --- Backward project polygons onto raw images ---
img_dict_ms = roi.back2raw(ms)

# --- Sort images by distance to ROI, keep best 10 per canopy ---
img_dict_sort = ms.sort_img_by_distance(
    img_dict_ms,
    roi,
    distance_thresh=10,  # distance threshold in meters
    num=10  # best 10 images
)

# --- Save img_dict_sort as JSON ---
os.makedirs(output_folder_path, exist_ok=True)
def convert_polygons_to_lists(d):
    out = {}
    for k, v in d.items():
        out[k] = {img: poly.tolist() if hasattr(poly, "tolist") else poly for img, poly in v.items()}
    return out

img_dict_sort_json = convert_polygons_to_lists(img_dict_sort)
json_path = os.path.join(output_folder_path, "img_dict_sort.json")
with open(json_path, "w") as f:
    json.dump(img_dict_sort_json, f, indent=2)

# --- Helper functions for image selection ---
def gimbal_pitch(img_path):
    try:
        img = pyexiv2.Image(img_path)
        xmp = img.read_xmp()
        if "Xmp.drone-dji.GimbalPitchDegree" in xmp:
            return float(xmp["Xmp.drone-dji.GimbalPitchDegree"])
    except Exception:
        pass
    return np.nan  # Always return a float

def sharpness_contrast_exposure(img_path, poly):
    try:
        img = io.imread(img_path)
        if img.ndim == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img.astype(np.float32) / 255.0
        mask = np.zeros(img_gray.shape, dtype=bool)
        rr, cc = draw.polygon(poly[:,1], poly[:,0], img_gray.shape)
        mask[rr, cc] = True
        if not np.any(mask):
            region = img_gray
            lap = filters.laplace(img_gray)
            sharpness = float(lap.var())
        else:
            region = img_gray[mask]
            lap = filters.laplace(img_gray)
            sharpness = float(lap[mask].var())
        contrast = float(region.std())
        exposure = float(region.mean())
        return sharpness, contrast, exposure, img_gray
    except Exception:
        return np.nan, np.nan, np.nan, np.zeros((1, 1))

def canopy_center_distance(polygon, img_shape):
    try:
        centroid = np.mean(polygon, axis=0)
        img_center = np.array([img_shape[1]/2, img_shape[0]/2])  # (x, y)
        return float(np.linalg.norm(centroid - img_center))
    except Exception:
        return np.nan

def select_best_image(image_paths, polygons):
    data = []
    for img_path, poly in zip(image_paths, polygons):
        sharp, cont, exp, img_gray = sharpness_contrast_exposure(img_path, poly)
        pitch = gimbal_pitch(img_path)
        center_dist = canopy_center_distance(poly, img_gray.shape)
        data.append({
            "img_path": img_path,
            "sharpness": sharp,
            "contrast": cont,
            "exposure": exp,
            "pitch": pitch,
            "center_dist": center_dist
        })
    df = pd.DataFrame(data)
    df_filt = df[(df["exposure"] > 0.15) & (df["exposure"] < 0.85)]
    if df_filt.empty:
        df_filt = df

    # 1. Prefer nadir (within 5 degrees of -90)
    nadir = df_filt[df_filt["pitch"].notnull() & (np.abs(df_filt["pitch"] + 90) < 5)]
    if not nadir.empty:
        df_filt = nadir

    # 2. Sharpest image
    best_sharp = df_filt["sharpness"].max()
    close = df_filt[df_filt["sharpness"] >= 0.85 * best_sharp]
    if len(close) > 1:
        best_contrast = close["contrast"].max()
        close2 = close[close["contrast"] >= 0.95 * best_contrast]
        if len(close2) > 1:
            best = close2.loc[close2["center_dist"].idxmin()]
        else:
            best = close.loc[close["contrast"].idxmax()]
    else:
        best = df_filt.loc[df_filt["sharpness"].idxmax()]
    return best["img_path"], best

# --- Create output directory for best images ---
best_images_dir = os.path.join(output_folder_path, "best_cropped_images")
os.makedirs(best_images_dir, exist_ok=True)

canopy_ids = list(img_dict_sort.keys())

for tree_id in canopy_ids:
    images = list(img_dict_sort[tree_id].keys())
    polygons = list(img_dict_sort[tree_id].values())
    image_paths = [os.path.join(os.path.dirname(raw_images_folder_path), img_rel_path) for img_rel_path in images]
    best_img_path, best_row = select_best_image(image_paths, polygons)
    best_img_path_str = str(best_img_path)
    print(f"Tree ID: {tree_id} | Best image: {os.path.basename(best_img_path_str)}")
    print(f"  Sharpness: {best_row['sharpness']:.4f}")
    print(f"  Contrast:  {best_row['contrast']:.4f}")
    print(f"  Exposure:  {best_row['exposure']:.4f}")
    print(f"  Pitch:     {best_row['pitch']}")
    best_img_name = os.path.basename(best_img_path_str)
    new_name = f"{tree_id}_{best_img_name}"
    dest_path = os.path.join(best_images_dir, new_name)

    # Find the polygon for the best image
    idx = image_paths.index(best_img_path)
    poly = np.array(polygons[idx])

    # Crop to polygon bounding box + 100px buffer
    img = io.imread(best_img_path)
    xmin = max(int(np.floor(poly[:, 0].min())) - 100, 0)
    xmax = min(int(np.ceil(poly[:, 0].max())) + 100, img.shape[1])
    ymin = max(int(np.floor(poly[:, 1].min())) - 100, 0)
    ymax = min(int(np.ceil(poly[:, 1].max())) + 100, img.shape[0])
    cropped = img[ymin:ymax, xmin:xmax]
    io.imsave(dest_path, cropped)