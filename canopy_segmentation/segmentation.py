import os
import argparse

import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from rasterio import mask as rio_mask
from rasterio.transform import rowcol, xy
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import gaussian
import geopandas as gpd
from shapely.geometry import shape, Point

class TreeCanopySegmentation:
    """
    Tree canopy segmentation using proximity-based watershed algorithm.
    """
    def __init__(
        self,
        chm_path,
        penalty_strength=0.1,
        boundary_penalty_weight=1.0,
        gradient_weight=0.4,
        watershed_compactness=0.0001
    ):
        self.chm_path = chm_path
        self.chm_data = None            # Working array (NaN preserved for masking)
        self.original_chm = None        # Original copy (NaN preserved)
        self.chm_profile = None         # Rasterio profile with CRS, transform, etc.
        self.tree_markers = None        # Integer array with tree marker positions
        self.segments = None            # Final segmentation result
        self.resolution_m_per_pixel = None
        self.extent_gdf = None          # Processing extent for boundary filtering
        self.min_height_m = None               # Canopy height threshold (auto-estimated if None)
        self.min_height_est_method = "bimodal" # Method used for height estimation
        self.min_height_fallback = 2.0         # Fallback if estimation fails
        self.surface_smooth_sigma = 1.5        # Gaussian smoothing for CHM surface

        # Parameters to tune
        self.penalty_strength = penalty_strength
        self.boundary_penalty_weight = boundary_penalty_weight
        self.gradient_weight = gradient_weight
        self.watershed_compactness = watershed_compactness

    def load_chm(self, extent_shapefile=None, crop_to_extent=True):
        try:
            with rasterio.open(self.chm_path) as src:
                if extent_shapefile:
                    gdf = gpd.read_file(extent_shapefile)
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)
                    self.extent_gdf = gdf
                    geoms = [geom for geom in gdf.geometry]
                    arr, out_transform = rio_mask.mask(src, geoms, crop=crop_to_extent, nodata=np.nan)
                    profile = src.profile.copy()
                    profile.update({
                        "height": arr.shape[1],
                        "width": arr.shape[2],
                        "transform": out_transform,
                        "nodata": np.nan
                    })
                    data = arr[0].astype(np.float64)
                    self.chm_profile = profile
                else:
                    data = src.read(1, masked=True)
                    if hasattr(data, "filled"):
                        data = data.filled(np.nan)
                    data = data.astype(np.float64)
                    profile = src.profile
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

                if self.min_height_m is None:
                    est = self.estimate_min_height()
                    self.min_height_m = est
                    print(f"Estimated min-height threshold: {self.min_height_m:.3f} m (method={self.min_height_est_method})")

                return True

        except Exception as e:
            print(f"Error loading CHM: {e}")
            return False

    def meters_to_pixels(self, distance_meters):
        """Convert distance in meters to pixels based on CHM resolution."""
        return int(round(distance_meters / self.resolution_m_per_pixel))

    def load_tree_tops_from_shapefile(self, shapefile_path, buffer_meters=1.5):
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs != self.chm_profile["crs"]:
                gdf = gdf.to_crs(self.chm_profile["crs"])

            transform = self.chm_profile["transform"]
            refined = []
            skipped = 0
            base_buf_px = max(1, self.meters_to_pixels(buffer_meters))

            def path_crosses_low_height(start_r, start_c, end_r, end_c):
                from skimage.draw import line
                line_r, line_c = line(start_r, start_c, end_r, end_c)
                for lr, lc in zip(line_r, line_c):
                    if (0 <= lr < self.chm_data.shape[0] and 
                        0 <= lc < self.chm_data.shape[1]):
                        val = self.chm_data[lr, lc]
                        if np.isfinite(val) and self.min_height_m is not None and val < self.min_height_m:
                            return True
                return False

            for _, row in gdf.iterrows():
                pt = row.geometry
                r, c = rowcol(transform, pt.x, pt.y)
                r = int(r); c = int(c)

                if not (0 <= r < self.chm_data.shape[0] and 0 <= c < self.chm_data.shape[1]):
                    skipped += 1
                    continue

                h_at_pt = None
                if 0 <= r < self.original_chm.shape[0] and 0 <= c < self.original_chm.shape[1]:
                    val = self.original_chm[r, c]
                    if np.isfinite(val):
                        h_at_pt = float(val)

                if h_at_pt is None:
                    skipped += 1
                    continue

                local_buf_px = base_buf_px

                # Always refine to local maxima within fixed buffer
                if local_buf_px > 0:
                    rmin = max(0, r - local_buf_px)
                    rmax = min(self.chm_data.shape[0], r + local_buf_px + 1)
                    cmin = max(0, c - local_buf_px)
                    cmax = min(self.chm_data.shape[1], c + local_buf_px + 1)
                    window = self.chm_data[rmin:rmax, cmin:cmax]

                    if np.all(np.isnan(window)):
                        refined.append((r, c))
                        continue

                    finite_mask = np.isfinite(window)
                    if np.any(finite_mask):
                        local = np.where(finite_mask, window, -np.inf)
                        flat_indices = np.argsort(local.ravel())[::-1]
                        candidates = []
                        for flat_idx in flat_indices:
                            local_idx = np.unravel_index(flat_idx, local.shape)
                            if local[local_idx] == -np.inf:
                                break
                            candidate_r = rmin + local_idx[0]
                            candidate_c = cmin + local_idx[1]
                            candidates.append((candidate_r, candidate_c, local[local_idx]))
                        best_candidate = None
                        for candidate_r, candidate_c, _ in candidates:
                            candidate_height = self.original_chm[candidate_r, candidate_c]
                            if (np.isfinite(candidate_height) and 
                                self.min_height_m is not None and 
                                candidate_height >= self.min_height_m and
                                not path_crosses_low_height(r, c, candidate_r, candidate_c)):
                                best_candidate = (candidate_r, candidate_c)
                                break
                        if best_candidate is not None:
                            refined.append(best_candidate)
                        else:
                            refined.append((r, c))
                    else:
                        refined.append((r, c))
                else:
                    refined.append((r, c))

            if len(refined) == 0:
                print("No valid refined tree tops found.")
                return None

            # Create integer marker array for watershed
            markers = np.zeros(self.chm_data.shape, dtype=np.int32)
            for i, (rr, cc) in enumerate(refined):
                markers[rr, cc] = i + 1

            self.tree_markers = markers
            rows = np.array([p[0] for p in refined], dtype=int)
            cols = np.array([p[1] for p in refined], dtype=int)
            print(f"Loaded {len(refined)} tree tops (skipped {skipped})")
            return (rows, cols)

        except Exception as e:
            print(f"Error loading tree tops shapefile: {e}")
            return None

    def watershed_segment(self):
        if self.tree_markers is None:
            print("No markers available for watershed.")
            return False

        if self.min_height_m is not None:
            mask = np.isfinite(self.chm_data) & (self.chm_data > self.min_height_m)
        else:
            mask = np.isfinite(self.chm_data) & (self.chm_data > 0)

        if not np.any(mask):
            print("No valid CHM pixels to segment (check min_height threshold).")
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
            from skimage.draw import line
            line_r, line_c = line(start_r, start_c, end_r, end_c)
            for lr, lc in zip(line_r, line_c):
                if (0 <= lr < self.chm_data.shape[0] and 
                    0 <= lc < self.chm_data.shape[1]):
                    val = self.chm_data[lr, lc]
                    if np.isfinite(val) and self.min_height_m is not None and val < self.min_height_m:
                        return True
            return False

        for i, marker_id in enumerate(marker_ids):
            marker_pos = marker_positions[i]
            marker_height = marker_heights[i]

            max_search_distance = 50.0  # meters - computational limit, not biological limit
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
                height_factor = 0.8 + 0.2 * np.clip(height_ratio, 0.5, 1.5)  # Maps to 0.9-1.1 range

                local_characteristic_distance = (nearest_distance / 2.0) * height_factor
                
                print(f"Tree {marker_id}: height={marker_height:.1f}m, neighbor_dist={nearest_distance:.1f}m, "
                      f"neighbor_height={nearest_neighbor_height:.1f}m, height_factor={height_factor:.2f}, "
                      f"boundary_dist={local_characteristic_distance:.1f}m")

                marker_mask = (self.tree_markers == marker_id)
                distance_from_this_marker = distance_transform_edt(~marker_mask)
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

        # Apply watershed algorithm
        self.segments = watershed(
            surface, 
            self.tree_markers, 
            connectivity=2,           # 8-connected neighborhood
            compactness=self.watershed_compactness,      # Now configurable
            mask=mask                # Only segment valid areas
        )
        
        # Post-processing cleanup
        n_segments_before = len(np.unique(self.segments)) - 1
        min_area_m2 = 5.0  # Minimum realistic tree canopy area
        min_size_pixels = int(min_area_m2 / (self.resolution_m_per_pixel ** 2))
        
        # Remove small objects and fill small holes
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
        
        # Remove segments that overlap with processing boundary
        if self.extent_gdf is not None:
            self._remove_boundary_segments()
        
        return True

    def _remove_boundary_segments(self):
        max_border_overlap_m = 0.5  # Maximum allowed overlap with boundary
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

    def estimate_min_height(self):
        vals = self.original_chm[np.isfinite(self.original_chm) & (self.original_chm > 0)]
        hist, bins = np.histogram(vals, bins=64)
        centers = (bins[:-1] + bins[1:]) / 2.0
        smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=1.0)
        peaks, _ = find_peaks(smooth, distance=2)
        if peaks.size >= 2:
            peak_vals = smooth[peaks]
            top_two = peaks[np.argsort(peak_vals)[-2:]]
            p1, p2 = np.sort(top_two)
            valley_idx = p1 + np.argmin(smooth[p1:p2 + 1])
            thresh = float(centers[valley_idx])
            self.min_height_est_method = "bimodal"
            return thresh
        else:
            self.min_height_est_method = "fallback"
            return self.min_height_fallback

    @staticmethod
    def _remove_shapefile_if_exists(path_shp):
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
    parser.add_argument("--tree-tops", "-t", required=True, help="Point shapefile of tree tops")
    parser.add_argument("--extent", "-e", required=True, help="Polygon shapefile defining processing extent")
    parser.add_argument("--buffer", type=float, default=1.0, help="Buffer radius (meters) for local maxima search")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--min-height-fallback", type=float, default=2.0, help="Fallback minimum height if bimodal estimation fails")
    parser.add_argument("--penalty-strength", type=float, default=0.1, help="Penalty strength for proximity boundaries")
    parser.add_argument("--boundary-penalty-weight", type=float, default=1.0, help="Weight for proximity penalty in surface")
    parser.add_argument("--gradient-weight", type=float, default=0.4, help="Weight for gradient magnitude in surface")
    parser.add_argument("--watershed-compactness", type=float, default=0.0001, help="Compactness parameter for watershed")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.chm_path):
        print("CHM not found.")
        return 1
    if not os.path.exists(args.tree_tops):
        print("Tree tops shapefile not found.")
        return 1
    if not os.path.exists(args.extent):
        print("Extent shapefile not found.")
        return 1

    # Set up output directory
    prefix = prefix_from_chm(args.chm_path)
    outdir = args.output_dir if args.output_dir else prefix

    # Initialize and configure segmentation
    seg = TreeCanopySegmentation(
        args.chm_path,
        penalty_strength=args.penalty_strength,
        boundary_penalty_weight=args.boundary_penalty_weight,
        gradient_weight=args.gradient_weight,
        watershed_compactness=args.watershed_compactness
    )
    seg.min_height_m = None  # Always estimate using bimodal; fallback if needed
    seg.min_height_fallback = args.min_height_fallback

    # Load and process data
    if not seg.load_chm(extent_shapefile=args.extent, crop_to_extent=True):
        return 1

    coords = seg.load_tree_tops_from_shapefile(
        args.tree_tops,
        buffer_meters=args.buffer
    )

    if coords is None:
        print("No valid refined tree tops found; exiting.")
        return 1

    # Perform segmentation and save results
    if not seg.watershed_segment():
        return 1

    seg.save_results(outdir, prefix)
    seg.save_refined_tree_tops(outdir, prefix)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())