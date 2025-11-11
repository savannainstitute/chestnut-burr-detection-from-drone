import os
import argparse

import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from rasterio import mask as rio_mask
from rasterio.transform import rowcol, xy
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import gaussian
import geopandas as gpd
from shapely.geometry import shape, Point

class TreeCanopySegmentation:
    """
    Performs proximity-based segmentation of tree canopies from a CHM raster.
    """
    def __init__(self, chm_path):
        """
        Initialize segmentation parameters and state.
        """
        self.penalty_strength = 0.1                # Controls strength of proximity penalty in segmentation
        self.boundary_penalty_weight = 1.0         # Weight for proximity penalty in the watershed surface
        self.gradient_weight = 0.4                 # Weight for CHM gradient in the watershed surface
        self.watershed_compactness = 0.0001        # Compactness parameter for watershed algorithm
        self.base_height_factor = 0.8              # Base factor for adjusting boundary distance by height
        self.height_factor_scale = 0.2             # Scale for height-based adjustment of boundary distance
        self.min_height = 1.9                      # Minimum CHM height (meters) for segmentation mask
        self.surface_smooth_sigma = 0.5            # Gaussian smoothing sigma for CHM surface

        self.chm_path = chm_path                   # Path to input CHM raster file
        self.chm_data = None                       # Cropped CHM raster data (NumPy array)
        self.original_chm = None                   # Unmodified CHM raster data (NumPy array)
        self.chm_profile = None                    # Raster metadata/profile (dict)
        self.tree_markers = None                   # Marker image for watershed (NumPy array)
        self.segments = None                       # Segmentation result (NumPy array)
        self.resolution_m_per_pixel = None         # Raster resolution (meters per pixel)
        self.extent_gdf = None                     # GeoDataFrame for processing extent polygon

    def load_chm(self, extent_shapefile):
        """
        Loads and crops the CHM raster to the given extent shapefile.
        """
        try:
            with rasterio.open(self.chm_path) as src:
                gdf = gpd.read_file(extent_shapefile)
                if gdf.crs != src.crs:
                    gdf = gdf.to_crs(src.crs)
                self.extent_gdf = gdf
                geoms = [geom for geom in gdf.geometry]
                arr, out_transform = rio_mask.mask(src, geoms, crop=True, nodata=np.nan)
                profile = src.profile.copy()
                profile.update({
                    "height": arr.shape[1],
                    "width": arr.shape[2],
                    "transform": out_transform,
                    "nodata": np.nan
                })
                data = arr[0].astype(np.float64)
                self.chm_profile = profile

                nodata = self.chm_profile.get("nodata", None)
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)

                self.original_chm = data.copy()
                self.chm_data = data.copy()

                transform = self.chm_profile["transform"]
                px_w = abs(transform.a)
                px_h = abs(transform.e)
                self.resolution_m_per_pixel = float((px_w + px_h) / 2.0)

                print(f"Loaded CHM: shape={self.chm_data.shape}, resolution={self.resolution_m_per_pixel:.4f} m/px")
                return True

        except Exception as e:
            print(f"Error loading CHM: {e}")
            return False

    def meters_to_pixels(self, distance_meters):
        """
        Converts a distance in meters to pixels using the raster resolution.
        """
        return int(round(distance_meters / self.resolution_m_per_pixel))

    def load_tree_tops_from_shapefile(self, shapefile_path, buffer_meters=0.25):
        """
        Loads tree top points, refines them to local maxima in the CHM, and creates marker image.
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs != self.chm_profile["crs"]:
                gdf = gdf.to_crs(self.chm_profile["crs"])

            transform = self.chm_profile["transform"]
            refined = []
            skipped = 0
            base_buf_px = max(1, self.meters_to_pixels(buffer_meters))

            for _, row in gdf.iterrows():
                pt = row.geometry
                r, c = rowcol(transform, pt.x, pt.y)
                r = int(r); c = int(c)

                if not (0 <= r < self.chm_data.shape[0] and 0 <= c < self.chm_data.shape[1]):
                    skipped += 1
                    continue

                # Use a square window of size (2*base_buf_px + 1) centered on (r, c)
                rmin = max(0, r - base_buf_px)
                rmax = min(self.original_chm.shape[0], r + base_buf_px + 1)
                cmin = max(0, c - base_buf_px)
                cmax = min(self.original_chm.shape[1], c + base_buf_px + 1)
                window = self.original_chm[rmin:rmax, cmin:cmax]

                finite_mask = np.isfinite(window)
                if not np.any(finite_mask):
                    skipped += 1
                    continue

                local = np.where(finite_mask, window, -np.inf)
                max_idx = np.argmax(local)
                max_local_idx = np.unravel_index(max_idx, local.shape)
                max_r = rmin + max_local_idx[0]
                max_c = cmin + max_local_idx[1]

                # No distance check; always use the local maximum in the window
                refined.append((max_r, max_c))

            if len(refined) == 0:
                print("No valid refined tree tops found.")
                return None

            markers = np.zeros(self.chm_data.shape, dtype=np.int32)
            for i, (rr, cc) in enumerate(refined):
                markers[rr, cc] = i + 1

            self.tree_markers = markers
            rows = np.array([p[0] for p in refined], dtype=int)
            cols = np.array([p[1] for p in refined], dtype=int)
            print(f"Loaded {len(refined)} tree tops (skipped {skipped})")

            print(f"Minimum height threshold: {self.min_height:.3f} m")
            return (rows, cols)

        except Exception as e:
            print(f"Error loading tree tops shapefile: {e}")
            return None

    def watershed_segment(self):
        """
        Performs adaptive watershed segmentation using the tree markers and CHM.
        """
        if self.tree_markers is None:
            print("No markers available for watershed.")
            return False

        if self.min_height is not None:
            mask = np.isfinite(self.chm_data) & (self.chm_data > self.min_height)
        else:
            mask = np.isfinite(self.chm_data) & (self.chm_data > 0)

        if not np.any(mask):
            print("No valid CHM pixels to segment (check minimum height threshold).")
            return False

        marker_positions = []
        marker_heights = []
        marker_ids = np.unique(self.tree_markers)
        marker_ids = marker_ids[marker_ids > 0]
        
        for marker_id in marker_ids:
            pos = np.where(self.tree_markers == marker_id)
            if len(pos[0]) > 0:
                r, c = pos[0][0], pos[1][0]
                marker_positions.append((r, c))
                marker_heights.append(self.original_chm[r, c])
        
        marker_positions = np.array(marker_positions)
        marker_heights = np.array(marker_heights)

        smoothed_chm = gaussian(self.chm_data, sigma=self.surface_smooth_sigma, preserve_range=True)

        grad_y, grad_x = np.gradient(smoothed_chm)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = np.where(np.isfinite(gradient_mag), gradient_mag, 0)
        height_range = np.nanmax(smoothed_chm) - np.nanmin(smoothed_chm[np.isfinite(smoothed_chm)])
        proximity_penalty = np.full_like(smoothed_chm, np.inf)
        
        def path_crosses_low_height(start_r, start_c, end_r, end_c):
            """
            Returns True if the path between two points crosses below min_height.
            """
            from skimage.draw import line
            line_r, line_c = line(start_r, start_c, end_r, end_c)
            for lr, lc in zip(line_r, line_c):
                if (0 <= lr < self.chm_data.shape[0] and 
                    0 <= lc < self.chm_data.shape[1]):
                    val = self.chm_data[lr, lc]
                    if np.isfinite(val) and self.min_height is not None and val < self.min_height:
                        return True
            return False

        for i, marker_id in enumerate(marker_ids):
            marker_pos = marker_positions[i]
            marker_height = marker_heights[i]

            max_search_distance = 25.0
            distances_to_others = []
            neighbor_info = []
            
            for j, other_pos in enumerate(marker_positions):
                if i != j:
                    dist_px = np.sqrt((marker_pos[0] - other_pos[0])**2 + (marker_pos[1] - other_pos[1])**2)
                    dist_m = dist_px * self.resolution_m_per_pixel

                    if (dist_m <= max_search_distance and 
                        not path_crosses_low_height(marker_pos[0], marker_pos[1], other_pos[0], other_pos[1])):
                        distances_to_others.append(dist_m)
                        neighbor_info.append((j, dist_m, marker_heights[j]))
            
            if len(distances_to_others) > 0:
                nearest_idx = np.argmin(distances_to_others)
                nearest_distance = distances_to_others[nearest_idx]
                nearest_neighbor_info = neighbor_info[nearest_idx]
                nearest_neighbor_height = nearest_neighbor_info[2]

                height_ratio = marker_height / nearest_neighbor_height
                height_factor = self.base_height_factor + self.height_factor_scale * np.clip(height_ratio, 0.5, 1.5)

                local_characteristic_distance = (nearest_distance / 2.0) * height_factor
                
                print(f"Tree {marker_id}: height={marker_height:.1f}m, neighbor_dist={nearest_distance:.1f}m, "
                      f"neighbor_height={nearest_neighbor_height:.1f}m, height_factor={height_factor:.2f}, "
                      f"boundary_dist={local_characteristic_distance:.1f}m")

                marker_mask = (self.tree_markers == marker_id)
                distance_from_this_marker = ndimage.distance_transform_edt(~marker_mask)
                distance_m = distance_from_this_marker * self.resolution_m_per_pixel
                
                local_penalty = self.penalty_strength * height_range * (1 - np.exp(-distance_m / local_characteristic_distance))
                
                proximity_penalty = np.minimum(proximity_penalty, local_penalty)
                
            else:
                print(f"Tree {marker_id}: isolated, using natural watershed boundaries")
                continue  

        proximity_penalty = np.where(np.isinf(proximity_penalty), 
                                     self.penalty_strength * height_range, proximity_penalty)
        
        inv_height = np.where(np.isfinite(smoothed_chm), -smoothed_chm, 0.0)
        surface = inv_height + (self.boundary_penalty_weight * proximity_penalty) + (self.gradient_weight * gradient_mag)

        self.segments = watershed(
            surface, 
            self.tree_markers, 
            connectivity=2,
            compactness=self.watershed_compactness,
            mask=mask
        )
        
        n_segments_before = len(np.unique(self.segments)) - 1
        min_area_m2 = 5.0
        min_size_pixels = int(min_area_m2 / (self.resolution_m_per_pixel ** 2))
        
        cleaned_segments = np.zeros_like(self.segments)
        for segment_id in np.unique(self.segments):
            if segment_id == 0:
                continue
            segment_mask = (self.segments == segment_id)
            cleaned_mask = remove_small_objects(segment_mask, min_size=min_size_pixels)
            cleaned_mask = remove_small_holes(cleaned_mask, area_threshold=min_size_pixels//4)
            cleaned_segments[cleaned_mask] = segment_id
        
        self.segments = cleaned_segments
        n_segments_after = len(np.unique(self.segments)) - 1
        
        print(f"Adaptive watershed produced {n_segments_before} segments, cleaned to {n_segments_after} segments")
        
        if self.extent_gdf is not None:
            self._remove_boundary_segments()
        
        return True

    def _remove_boundary_segments(self):
        """
        Removes segments that overlap the raster boundary by more than a threshold.
        """
        max_border_overlap_m = 0.5
        max_border_pixels = int(max_border_overlap_m / self.resolution_m_per_pixel)
        boundary_geoms = []
        for geom in self.extent_gdf.geometry:
            if hasattr(geom, 'exterior'):
                boundary_geoms.append(geom.exterior)
            elif hasattr(geom, 'geoms'):
                for sub_geom in geom.geoms:
                    boundary_geoms.append(sub_geom.exterior)
        if not boundary_geoms:
            return
        boundary_mask = rasterize(
            boundary_geoms,
            out_shape=self.segments.shape,
            transform=self.chm_profile["transform"],
            fill=0,
            default_value=1,
            dtype=np.uint8
        ).astype(bool)
        segments_to_remove = []
        for seg_id in np.unique(self.segments):
            if seg_id == 0:
                continue
            segment_mask = (self.segments == seg_id)
            overlap_pixels = np.sum(segment_mask & boundary_mask)
            if overlap_pixels > max_border_pixels:
                segments_to_remove.append(seg_id)
        for seg_id in segments_to_remove:
            self.segments[self.segments == seg_id] = 0
        print(f"Removed {len(segments_to_remove)} boundary segments")

    @staticmethod
    def _remove_shapefile_if_exists(path_shp):
        """
        Removes all files associated with a shapefile if they exist before overwriting.
        """
        base, _ = os.path.splitext(path_shp)
        exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"]
        for e in exts:
            p = base + e
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def save_refined_tree_tops(self, output_dir, prefix):
        """
        Saves the refined tree top points as a shapefile.
        """
        if self.tree_markers is None:
            return
        marker_coords = np.where(self.tree_markers > 0)
        tree_ids = self.tree_markers[marker_coords]
        points = []
        heights = []
        transform = self.chm_profile["transform"]
        for r, c, _ in zip(marker_coords[0], marker_coords[1], tree_ids):
            x, y = xy(transform, int(r), int(c), offset="center")
            points.append(Point(x, y))
            heights.append(self.original_chm[r, c] if np.isfinite(self.original_chm[r, c]) else np.nan)
        gdf = gpd.GeoDataFrame({
            "tree_id": list(tree_ids),
            "height": heights,
            "geometry": points
        }, crs=self.chm_profile["crs"])
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{prefix}_Treetops.shp")
        self._remove_shapefile_if_exists(out_path)
        gdf.to_file(out_path)
        print(f"Saved refined treetops: {out_path}")

    def save_results(self, output_dir, prefix):
        """
        Saves the segmented canopy polygons as a shapefile with shape statistics.
        """
        if self.segments is None:
            print("No segments to save.")
            return
        os.makedirs(output_dir, exist_ok=True)
        polygons = []
        labels = []
        transform = self.chm_profile["transform"]
        for geom, val in shapes(self.segments.astype(np.int32), transform=transform):
            if val == 0:
                continue
            polygons.append(shape(geom))
            labels.append(int(val))
        if len(polygons) == 0:
            print("No polygons generated from segments.")
            return
        pix_area = self.resolution_m_per_pixel ** 2
        stats = []
        for lbl in labels:
            mask = self.segments == lbl
            area_m2 = np.sum(mask) * pix_area
            heights = self.original_chm[mask]
            max_h = np.nanmax(heights) if np.any(np.isfinite(heights)) else np.nan
            mean_h = np.nanmean(heights) if np.any(np.isfinite(heights)) else np.nan
            stats.append((area_m2, max_h, mean_h))
        gdf = gpd.GeoDataFrame({
            "tree_id": labels,
            "geometry": polygons,
            "area_m2": [s[0] for s in stats],
            "max_h": [s[1] for s in stats],
            "mean_h": [s[2] for s in stats]
        }, crs=self.chm_profile["crs"])
        out_path = os.path.join(output_dir, f"{prefix}_Canopies.shp")
        self._remove_shapefile_if_exists(out_path)
        gdf.to_file(out_path)
        print(f"Saved canopy polygons: {out_path} ({len(gdf)} features)")

def prefix_from_chm(chm_path):
    """
    Returns a file prefix based on the CHM filename.
    """
    name = os.path.basename(chm_path)
    if name.lower().endswith("_chm.tif"):
        return name[:-8]
    elif name.lower().endswith(".tif"):
        return name[:-4]
    else:
        return os.path.splitext(name)[0]

def main():
    parser = argparse.ArgumentParser(description="Proximity-based canopy segmentation")
    parser.add_argument("chm_path", help="Path to CHM TIFF")
    parser.add_argument("--tree-markers", "-t", required=True, help="Point shapefile of tree markers")
    parser.add_argument("--extent", "-e", required=True, help="Polygon shapefile defining processing extent")
    args = parser.parse_args()

    if not os.path.exists(args.chm_path):
        print("CHM not found.")
        return 1
    if not os.path.exists(args.tree_tops):
        print("Tree tops shapefile not found.")
        return 1
    if not os.path.exists(args.extent):
        print("Extent shapefile not found.")
        return 1

    prefix = prefix_from_chm(args.chm_path)
    outdir = prefix

    seg = TreeCanopySegmentation(args.chm_path)

    if not seg.load_chm(extent_shapefile=args.extent):
        return 1

    coords = seg.load_tree_tops_from_shapefile(
        args.tree_tops,
        buffer_meters=0.25
    )

    if coords is None:
        print("No valid refined tree tops found; exiting.")
        return 1

    if not seg.watershed_segment():
        return 1

    seg.save_results(outdir, prefix)
    seg.save_refined_tree_tops(outdir, prefix)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())