import Metashape
import os
import gc
import sys
import time
import yaml
import argparse

from metashape_utils import *

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Metashape automated reconstruction')
    parser.add_argument('--config', type=str, default='config.yml', 
                        help='Path to configuration file')
    parser.add_argument('--folders', type=str,
                        help='List of folders to process, in format: ["folder1", "folder2"]')
    args = parser.parse_args()
    return args

def configure_processors():
    """Configure Metashape to use only high VRAM GPUs (>= 8 GB VRAM) and disable CPU for GPU steps."""
    devices = Metashape.app.enumGPUDevices()
    if len(devices) > 0:
        gpu_names = [device.get('name', f'GPU {i}') for i, device in enumerate(devices)]
        gpu_vrams = [device.get('mem_size', 0) for device in devices]  # VRAM in bytes

        print("Available GPUs:")
        for i, (name, vram) in enumerate(zip(gpu_names, gpu_vrams)):
            if vram:
                print(f"  [{i}] {name} ({vram / (1024 ** 3):.1f} GB VRAM)")
            else:
                print(f"  [{i}] {name} (VRAM unknown)")

        # Select only GPUs with >= 8 GB VRAM
        selected_indices = [
            i for i, vram in enumerate(gpu_vrams)
            if vram and vram >= 8 * (1024 ** 3)
        ]
        if not selected_indices:
            print("No high VRAM GPUs found. Using all available GPUs.")
            selected_indices = list(range(len(devices)))

        # Set GPU mask
        gpu_mask = sum(1 << i for i in selected_indices)
        Metashape.app.gpu_mask = gpu_mask
        print(f"Enabled GPU(s): {', '.join(gpu_names[i] for i in selected_indices)} (mask={gpu_mask})")

        print("Setting GPU backend...")
        Metashape.app.settings.setValue("main/gpu_enable_opencl", "1") # true
        Metashape.app.settings.setValue("main/gpu_enable_cuda", "0") # false

        # Always disable CPU for GPU steps - avoids memory fragmentation
        Metashape.app.cpu_enable = False
        print("CPU disabled during dedicated GPU processing steps")

    else:
        # No GPU available, use CPU
        Metashape.app.cpu_enable = True
        print("No GPUs detected, using CPU processing")

def check_processing_status(chunk):
    """Check what processing steps have already been completed"""
    status = {
        'photos_loaded': len(chunk.cameras) > 0,
        'tie_points_exist': len(chunk.tie_points.points) > 0 if chunk.tie_points else False,
        'depth_maps_built': bool(chunk.depth_maps),
        'point_cloud_built': bool(chunk.point_cloud),
        'model_built': bool(chunk.model),
        'elevations_built': len(chunk.elevations) > 0,
        'orthomosaic_built': bool(chunk.orthomosaic)
    }
    return status

def main():
    """
    Main function to process a single mounted folder.
    This script is designed to process one drone flight folder at a time.
    """
    setup_license_cleanup()
    
    # Activate license
    if not activate_license():
        sys.exit(1)

    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)

    folder_name = os.environ.get('FOLDER_NAME')
    if not folder_name:
        print("Error: FOLDER_NAME environment variable not set")
        sys.exit(1)

    input_folder = folder_name
    output_folder = os.path.join(input_folder, "outputs")
    os.makedirs(output_folder, exist_ok=True)
    
    valid_exts = [".JPG", ".JPEG", ".TIF", ".TIFF"]
    
    try:
        # Find photos in the input folder
        photos = find_files(input_folder, valid_exts)
        if not photos:
            print(f"No valid image files found in folder {input_folder}")
            sys.exit(1)
            
        print(f"Processing folder: {input_folder} ({len(photos)} photos found)")

        # Load project if already exists, otherwise create new
        doc = Metashape.Document()
        lowest_folder_name = os.path.basename(os.path.normpath(input_folder))
        project_path = os.path.join(output_folder, f"project_{lowest_folder_name}.psx")

        if os.path.exists(project_path):
            print(f"Existing project found: {project_path}")
            doc.open(project_path, read_only=False, ignore_lock=True)
            if doc.chunks:
                chunk = doc.chunks[0]
                print("Loaded existing chunk from project")
                
                # Check which processing steps are already done
                status = check_processing_status(chunk)
                print("Processing status:")
                for step, completed in status.items():
                    print(f"  {step}: {completed}")
            else:
                print("Project exists but no chunks found, creating new chunk")
                chunk = doc.addChunk()
                status = check_processing_status(chunk)
        else:
            print("Creating new project")
            doc.save(project_path)
            doc.open(project_path, read_only=False, ignore_lock=True)
            chunk = doc.addChunk()
            status = check_processing_status(chunk)
        
        configure_processors()

        # Set coordinate system using image GPS data
        try:
            epsg_code = get_utm_zone_from_gps(photos)
            if epsg_code:
                chunk.crs = Metashape.CoordinateSystem(epsg_code)
                print(f"Coordinate system set to: {chunk.crs}")
        except Exception as e:
            print(f"Error setting coordinate system: {e}")

        if not status['photos_loaded']:
            try:
                print("Loading photos...")
                chunk.addPhotos(filenames=photos, load_xmp_accuracy=True, progress=progress_timer.update)
                progress_timer.reset()
                print(f"{len(chunk.cameras)} images loaded from: {input_folder}.")
                
                # Rename photos to include parent directory
                for camera in chunk.cameras:
                    try:
                        path = os.path.normpath(camera.photo.path)
                        parent_dir = os.path.basename(os.path.dirname(path))
                        camera.label = f"{parent_dir}/{os.path.basename(path)}"
                    except:
                        pass
                print("Camera labels updated with parent directory names.")
                
                # Set up RTK accuracy
                setup_rtk_accuracy(chunk, config)
                doc.save()

                print("Analyzing image quality...")
                chunk.analyzeImages(progress=progress_timer.update)
                progress_timer.reset()
                
                # Get quality estimates
                quality_values = {}
                for camera in chunk.cameras:
                    if camera.meta["Image/Quality"] is not None:
                        quality_values[camera] = float(camera.meta["Image/Quality"])
                
                # Filter out low-quality images
                if quality_values:
                    quality_threshold = config['image_quality']['quality_threshold'] 
                    low_quality_cameras = [camera for camera, quality in quality_values.items() 
                                        if quality < quality_threshold]
                    
                    if low_quality_cameras:
                        print(f"Disabling {len(low_quality_cameras)} low-quality images (quality < {quality_threshold})...")
                        for camera in low_quality_cameras:
                            camera.enabled = False
                        print(f"{len([c for c in chunk.cameras if c.enabled])} images enabled.")
                    else:
                        print("All images passed quality check.")
                else:
                    print("Image quality analysis did not return results. Proceeding with all images.")
                doc.save()
            except Exception as e:
                print(f"Error during photo loading and quality analysis: {e}")
                sys.exit(1)
        else:
            print("Photos already loaded, skipping...")

        # Tie points
        if not status['tie_points_exist']:
            try:
                print("Matching photos...")
                timer_match = time.time()
                match_params = config['photo_matching']
                
                chunk.matchPhotos(downscale=match_params['downscale'],
                                downscale_3d=match_params['downscale_3d'],
                                keypoint_limit=match_params['keypoint_limit'], 
                                keypoint_limit_3d=match_params['keypoint_limit_3d'], 
                                keypoint_limit_per_mpx=match_params['keypoint_limit_per_mpx'],
                                tiepoint_limit=match_params['tiepoint_limit'],
                                generic_preselection=match_params['generic_preselection'], 
                                reference_preselection=match_params['reference_preselection'],
                                filter_mask=match_params['filter_mask'], 
                                mask_tiepoints=match_params['mask_tiepoints'], 
                                filter_stationary_points=match_params['filter_stationary_points'],
                                keep_keypoints=match_params['keep_keypoints'],
                                guided_matching=match_params['guided_matching'],
                                subdivide_task=match_params['subdivide_task'],
                                progress=progress_timer.update)
                progress_timer.reset() 
                print(f"Photos matched in {round(time.time() - timer_match, 1)} seconds.")
                
                # Align cameras
                chunk.alignCameras(
                    adaptive_fitting=config['camera']['align']['adaptive_fitting'], 
                    min_image=config['camera']['align']['min_image'], 
                    progress=progress_timer.update
                )
                progress_timer.reset()
                print(f"{len(chunk.cameras)} cameras aligned.")
                
                # Reset region to prevent point clipping
                reset_region(chunk)
                
                # Optimize cameras
                print("Optimizing cameras...")
                chunk.optimizeCameras(
                    fit_f=config['camera']['optimize']['fit_f'],
                    fit_cx=config['camera']['optimize']['fit_cx'],
                    fit_cy=config['camera']['optimize']['fit_cy'],
                    fit_k1=config['camera']['optimize']['fit_k1'],
                    fit_k2=config['camera']['optimize']['fit_k2'],
                    fit_k3=config['camera']['optimize']['fit_k3'],
                    fit_k4=config['camera']['optimize']['fit_k4'],
                    fit_p1=config['camera']['optimize']['fit_p1'],
                    fit_p2=config['camera']['optimize']['fit_p2'],
                    fit_b1=config['camera']['optimize']['fit_b1'],
                    fit_b2=config['camera']['optimize']['fit_b2'],
                    fit_corrections=config['camera']['optimize']['fit_corrections'],
                    tiepoint_covariance=config['camera']['optimize']['tiepoint_covariance'],
                    progress=progress_timer.update
                )
                progress_timer.reset()
                print(f"{len(chunk.cameras)} cameras optimized")
                doc.save()
                
                print("Filtering tie points...")               
                points_before = len(chunk.tie_points.points)

                ru_threshold, pa_threshold, re_threshold1 = filter_tie_points_usgs_part1(chunk, config)
                re_threshold2 = filter_tie_points_usgs_part2(chunk, config)

                points_after = len(chunk.tie_points.points)
                percent_removed = round((points_before - points_after) / points_before * 100, 1) if points_before > 0 else 0
                print(f"Tie point filtering complete. Removed {points_before - points_after} points ({percent_removed}%).")
                print(f"Applied thresholds: RU={ru_threshold:.1f}, PA={pa_threshold:.1f}, RE1={re_threshold1:.2f}, RE2={re_threshold2:.2f}")
                reset_region(chunk)
                
                # Export camera positions
                camera_file = os.path.join(output_folder, f"{lowest_folder_name}_camera_positions.txt")
                if not os.path.exists(camera_file):
                    chunk.exportCameras(camera_file, format=Metashape.CamerasFormat.CamerasFormatOPK)
                    print("Camera positions exported.")
                else:
                    print("Camera positions file already exists, skipping export.")
                doc.save()
            except Exception as e:
                print(f"Error during camera alignment/matching/optimization: {e}")
                sys.exit(1)
        else:
            print("Tie points already exist, skipping matching/alignment/filtering...")
        
        # Depth maps
        if not status['depth_maps_built']:
            try:
                print("Building depth maps...")

                # Set subdivide_task for depth maps based on estimated RAM usage
                config = adaptive_subdivide(chunk, 'depth_maps', config)

                # Convert filter_mode from string to Metashape enum
                filter_mode_str = config['depth_maps']['filter_mode'].lower()
                if filter_mode_str == "mild":
                    filter_mode = Metashape.FilterMode.MildFiltering
                elif filter_mode_str == "moderate":
                    filter_mode = Metashape.FilterMode.ModerateFiltering
                elif filter_mode_str == "aggressive":
                    filter_mode = Metashape.FilterMode.AggressiveFiltering
                else:
                    filter_mode = Metashape.FilterMode.NoFiltering

                chunk.buildDepthMaps(
                    downscale=config['depth_maps']['downscale'],  
                    filter_mode=filter_mode, 
                    reuse_depth=config['depth_maps']['reuse_depth'],  
                    max_neighbors=config['depth_maps']['max_neighbors'], 
                    subdivide_task=config['depth_maps']['subdivide_task'],
                    max_gpu_multiplier=config['depth_maps']['max_gpu_multiplier'], # hidden setting that allows concurrency on single GPU
                    progress=progress_timer.update
                )
                progress_timer.reset()
                print("Depth maps finished building.")
                doc.save()
            except Exception as e:
                print(f"Error building depth maps: {e}")
                sys.exit(1)
        else:
            print("Depth maps already built, skipping...")

        # Point cloud
        if not status['point_cloud_built']:
            try:
                if has_valid_transform(chunk):
                    # Convert source_data from string to Metashape enum
                    source_data_str = config['point_cloud']['source_data'].lower()
                    if source_data_str == "depth_maps":
                        source_data = Metashape.DataSource.DepthMapsData
                    else:
                        source_data = Metashape.DataSource.PointCloudData # tie points (sparse cloud)

                    # Build point cloud
                    print("Building dense cloud...")
                    chunk.buildPointCloud(
                        source_data=source_data, 
                        point_colors=config['point_cloud']['point_colors'], 
                        point_confidence=config['point_cloud']['point_confidence'], 
                        keep_depth=config['point_cloud']['keep_depth'],
                        max_neighbors=config['point_cloud']['max_neighbors'],
                        subdivide_task=config['point_cloud']['subdivide_task'],
                        progress=progress_timer.update
                    ) 
                    progress_timer.reset()
                    doc.save()
                    print("Point cloud finished building.")
                    
                    # Classify ground points
                    print("Classifying ground points...")
                    ground_config = config['classify_ground_points']
                    chunk.point_cloud.classifyGroundPoints(
                        max_angle=ground_config['max_angle'],
                        max_distance=ground_config['max_distance'], 
                        cell_size=ground_config['cell_size'],
                        progress=progress_timer.update
                    )
                    progress_timer.reset()
                    doc.save()
                    print("Ground points classified.")

                    # Export point cloud
                    pc_file = os.path.join(output_folder, f"{lowest_folder_name}_point_cloud.{config['point_cloud']['export']['format']}")
                    
                    # Convert format from string to Metashape enum
                    format_str = config['point_cloud']['export']['format'].lower()
                    if format_str == "las":
                        export_format = Metashape.PointCloudFormat.PointCloudFormatLAS
                    elif format_str == "laz":
                        export_format = Metashape.PointCloudFormat.PointCloudFormatLAZ
                    elif format_str == "e57":
                        export_format = Metashape.PointCloudFormat.PointCloudFormatE57
                    elif format_str == "ply":
                        export_format = Metashape.PointCloudFormat.PointCloudFormatPLY
                    else:
                        export_format = Metashape.PointCloudFormat.PointCloudFormatXYZ
                    
                    chunk.exportPointCloud(
                        pc_file, 
                        source_data=Metashape.DataSource.PointCloudData, 
                        save_point_color=config['point_cloud']['export']['save_point_color'],
                        save_point_normal=config['point_cloud']['export']['save_point_normal'],
                        save_point_confidence=config['point_cloud']['export']['save_point_confidence'],
                        format=export_format,
                        crs=chunk.crs,
                        progress=progress_timer.update
                    )
                    progress_timer.reset()
                    print("Point cloud exported.")
            except Exception as e:
                print(f"Error building point cloud: {e}")
                sys.exit(1)
        else:
            print("Point cloud already built, skipping...")

        # Mesh (3d model)
        if not status['model_built']:
            try:
                print("Building 3D model...")
                chunk.buildModel(
                    surface_type=Metashape.SurfaceType.Arbitrary,  
                    interpolation=Metashape.Interpolation.EnabledInterpolation,
                    face_count=Metashape.FaceCount.HighFaceCount, 
                    source_data=Metashape.DataSource.DepthMapsData,
                    vertex_colors=config['model']['vertex_colors'], 
                    vertex_confidence=config['model']['vertex_confidence'], 
                    keep_depth=config['model']['keep_depth'],
                    subdivide_task=config['model']['subdivide_task'],
                    progress=progress_timer.update
                )
                progress_timer.reset()
                doc.save()
                print("3D model finished building.")
            except Exception as e:
                print(f"Error building model: {e}")
                sys.exit(1)
        else:
            print("3D model already built, skipping...")

        # Set up projection for DEM and orthomosaic
        projection = Metashape.OrthoProjection()
        projection.crs = Metashape.CoordinateSystem(epsg_code)

        # DEMs
        if not status['elevations_built']:
            try:
                # Build DEMs if enabled
                print("Building DEMs...")
                
                # DEM compression settings
                compression = Metashape.ImageCompression()
                compression.tiff_big = config['dem']['tiff_big']
                compression.tiff_tiled = config['dem']['tiff_tiled']
                compression.tiff_overviews = config['dem']['tiff_overviews']
                
                # Build DSM from model
                print("Building DSM from model...")
                chunk.buildDem(
                    source_data=Metashape.DataSource.ModelData,
                    interpolation=Metashape.Interpolation.EnabledInterpolation,
                    subdivide_task=config['dem']['subdivide_task'],
                    projection=projection,
                    resolution=config['dem']['resolution'],
                    progress=progress_timer.update
                )
                progress_timer.reset()
                chunk.elevation.label = "DSM"
                doc.save()
                
                # Export DSM
                dsm_file = os.path.join(output_folder, f"{lowest_folder_name}_dsm.tif")
                chunk.exportRaster(
                    path=dsm_file,
                    projection=projection,
                    nodata_value=config['dem']['nodata'],
                    source_data=Metashape.DataSource.ElevationData,
                    image_compression=compression,
                    progress=progress_timer.update
                )
                progress_timer.reset()
                print("DSM exported.")
                
                # Build DTM from ground points only
                print("Building DTM from ground points in point cloud...")
                chunk.buildDem(
                    source_data=Metashape.DataSource.PointCloudData,
                    classes=Metashape.PointClass.Ground,
                    subdivide_task=config['dem']['subdivide_task'],
                    projection=projection,
                    resolution=config['dem']['resolution'],
                    progress=progress_timer.update
                )
                progress_timer.reset()
                chunk.elevation.label = "DTM"
                doc.save()
                
                # Export DTM
                dtm_file = os.path.join(output_folder, f"{lowest_folder_name}_dtm.tif")
                chunk.exportRaster(
                    path=dtm_file,
                    projection=projection,
                    nodata_value=config['dem']['nodata'],
                    source_data=Metashape.DataSource.ElevationData,
                    image_compression=compression,
                    progress=progress_timer.update
                )
                progress_timer.reset()
                print("DTM exported.")
                
                # Create and export CHM
                print("Creating Canopy Height Model (CHM)...")
                dsm_asset = None
                dtm_asset = None
                for elevation in chunk.elevations:
                    if elevation.label == "DSM":
                        dsm_asset = elevation.key
                    elif elevation.label == "DTM":
                        dtm_asset = elevation.key

                if dsm_asset is not None and dtm_asset is not None:
                    chm_file = os.path.join(output_folder, f"{lowest_folder_name}_chm.tif")
                    # CHM = DSM - DTM
                    chunk.transformRaster(
                        asset=dsm_asset,
                        operand_asset=dtm_asset,
                        subtract=True,
                        projection=projection,
                        nodata_value=config['dem']['nodata'],
                        resolution=config['dem']['resolution'],
                        replace_asset=False,
                        clip_to_boundary=False
                    )
                    chunk.elevation.label = "CHM"
                    doc.save()
                    # Export CHM
                    chunk.exportRaster(
                        path=chm_file,
                        projection=projection,
                        nodata_value=config['dem']['nodata'],
                        source_data=Metashape.DataSource.ElevationData,
                        image_compression=compression,
                        progress=progress_timer.update
                    )
                    print("CHM exported.")
                else:
                    print("DSM or DTM asset not found, CHM not created.")

                # Set DSM as active elevation surface for orthomosaic
                for elevation in chunk.elevations:
                    if elevation.label == "DSM":
                        chunk.elevation = elevation
                        break
            except Exception as e:
                print(f"Error building DEMs: {e}")
                sys.exit(1)
        else:
            print("DEMs already built, skipping...")
                
        # Orthomosaic
        if not status['orthomosaic_built']:
            try:
                print("Building orthomosaic from DSM...")
                
                # Set up ortho compression
                compression = Metashape.ImageCompression()
                compression.tiff_big = config['orthomosaic']['export']['tiff_big']
                compression.tiff_tiled = config['orthomosaic']['export']['tiff_tiled'] 
                compression.tiff_overviews = config['orthomosaic']['export']['tiff_overviews']
                
                # Convert blending mode
                blend_str = config['orthomosaic']['blending_mode'].lower()
                if blend_str == "mosaic":
                    blend_mode = Metashape.BlendingMode.MosaicBlending
                elif blend_str == "average":
                    blend_mode = Metashape.BlendingMode.AverageBlending
                elif blend_str == "max":
                    blend_mode = Metashape.BlendingMode.MaxBlending
                elif blend_str == "min":
                    blend_mode = Metashape.BlendingMode.MinBlending
                else:
                    blend_mode = Metashape.BlendingMode.DisabledBlending
                
                # Build orthomosaic from DSM
                chunk.buildOrthomosaic(
                    surface_data=Metashape.DataSource.ElevationData, 
                    blending_mode=blend_mode, 
                    ghosting_filter=config['orthomosaic']['ghosting_filter'],  
                    fill_holes=config['orthomosaic']['fill_holes'],
                    cull_faces=config['orthomosaic']['cull_faces'],
                    refine_seamlines=config['orthomosaic']['refine_seamlines'],
                    subdivide_task=config['orthomosaic']['subdivide_task'],
                    projection=projection,
                    progress=progress_timer.update
                )
                progress_timer.reset()
                doc.save()
                print("Orthomosaic finished building.")
                
                # Export orthomosaic 
                ortho_file = os.path.join(output_folder, f"{lowest_folder_name}_orthomosaic.tif")
                chunk.exportRaster(
                    ortho_file, 
                    source_data=Metashape.DataSource.OrthomosaicData,
                    projection=projection,
                    image_compression=compression,
                    white_background=config['orthomosaic']['export']['white_background'],
                    nodata_value=config['orthomosaic']['export']['nodata'],
                    progress=progress_timer.update
                )
                progress_timer.reset()
                print("Orthomosaic exported.")
                
                gc.collect()
            except Exception as e:
                print(f"Error building orthomosaic: {e}")
                sys.exit(1)
        else:
            print("Orthomosaic already built, skipping...")
        
        try:
            # Export report
            report_file = os.path.join(output_folder, f"{lowest_folder_name}_report.pdf")
            chunk.exportReport(report_file)
            print("Report exported.")
                
            print(f"Processing finished for {lowest_folder_name}; results saved to {output_folder}.")

            doc.save()
            doc = None  
            gc.collect()  
            print("Document closed and memory released.")
        except Exception as e:
            print(f"Error during export: {e}")
            doc.save()
            doc = None
            gc.collect()
            sys.exit(1)
        
    except Exception as e:
        print(f"Error processing folder {input_folder}: {e}")
        sys.exit(1)
    finally:
        if 'doc' in locals() and doc is not None:
            doc.save()
            doc = None
        gc.collect()

    print("Processing completed successfully!")
    deactivate_license()

if __name__ == '__main__':
    try:
        gc.collect()
        main()
    except Exception as e:
        print(f"Error in main processing: {e}")