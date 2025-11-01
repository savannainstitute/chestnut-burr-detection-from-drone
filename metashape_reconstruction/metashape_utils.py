import pyexiv2
import signal
import atexit
import Metashape
import os
import sys
import psutil
import time

license_obj = None

def find_files(folder, valid_types):
    try:
        valid_types = [ext.lower() for ext in valid_types]
        return [os.path.join(folder, entry.name)
                for entry in os.scandir(folder)
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_types]
    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
        return []

def get_utm_zone_from_gps(photo_paths):
    """Auto-detect UTM zone from GPS coordinates in first photo"""
    
    # Let Metashape read the GPS data from EXIF
    temp_doc = Metashape.Document()
    temp_chunk = temp_doc.addChunk()
    temp_chunk.addPhotos([photo_paths[0]])
    
    lat = temp_chunk.cameras[0].reference.location.y
    lon = temp_chunk.cameras[0].reference.location.x
    
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    
    if hemisphere == 'N':
        epsg_code = f"EPSG::{32600 + utm_zone}"
    else:
        epsg_code = f"EPSG::{32700 + utm_zone}"
    
    print(f"Auto-detected UTM zone: {utm_zone}{hemisphere} ({epsg_code})")
    return epsg_code

def setup_rtk_accuracy(chunk, config):
    """Set RTK accuracy using actual XMP accuracy values when available, disable GPS otherwise"""
    if not config['gps']['enabled'] or not config['gps']['use_rtk']:
        return
    
    print("Setting up RTK accuracy...")
    accuracy_from_xmp = 0
    gps_disabled = 0
    
    for cam in chunk.cameras:
        try:
            meta = pyexiv2.Image(cam.photo.path)
            xmp_data = meta.read_xmp()
            meta.close()
            
            # Try to get actual accuracy values first
            rtk_std_lon = xmp_data.get('Xmp.drone-dji.RtkStdLon')
            rtk_std_lat = xmp_data.get('Xmp.drone-dji.RtkStdLat') 
            rtk_std_hgt = xmp_data.get('Xmp.drone-dji.RtkStdHgt')
            
            if rtk_std_lon and rtk_std_lat and rtk_std_hgt:
                # Use actual per-image accuracy values
                cam.reference.location_accuracy = Metashape.Vector([
                    float(rtk_std_lon),
                    float(rtk_std_lat), 
                    float(rtk_std_hgt)
                ])
                cam.reference.accuracy = Metashape.Vector([
                    float(rtk_std_lon),
                    float(rtk_std_lat), 
                    float(rtk_std_hgt)
                ])
                accuracy_from_xmp += 1
            else:
                # No RTK data - disable GPS positioning
                cam.reference.location_enabled = False
                gps_disabled += 1
            
        except Exception as e:
            print(f"Warning: RTK setup failed for {cam.label}: {e}")
            cam.reference.location_enabled = False
            gps_disabled += 1
    
    print(f"RTK setup: {accuracy_from_xmp} cameras with XMP accuracy, {gps_disabled} cameras with GPS disabled")
    
    chunk.updateTransform()

def reset_region(chunk):
    """
    Reset the region and make it much larger than the points;
    necessary because if points go outside the region,
    they get clipped when saving
    """
    chunk.resetRegion()
    region_dims = chunk.region.size
    region_dims[2] *= 3 # Increase height by 3x
    chunk.region.size = region_dims
    print("Region reset to prevent point clipping.")
    return True

def filter_tie_points_usgs_part1(chunk, config):
    """
    First stage of USGS point filtering approach - provides better point retention in vegetation
    """
    print("Performing USGS-style point filtering (stage 1)...")
    
    # Get filtering parameters from config
    ru_config = config['tie_point_filtering']['reconstruction_uncertainty']
    pa_config = config['tie_point_filtering']['projection_accuracy']
    re_config = config['tie_point_filtering']['reprojection_error']
    
    # Get camera optimization parameters
    cam_optimize = config['camera']['optimize']
    
    chunk.optimizeCameras(fit_f=cam_optimize['fit_f'], 
                      fit_cx=cam_optimize['fit_cx'], 
                      fit_cy=cam_optimize['fit_cy'],  
                      fit_k1=cam_optimize['fit_k1'], 
                      fit_k2=cam_optimize['fit_k2'], 
                      fit_k3=cam_optimize['fit_k3'], 
                      fit_k4=cam_optimize['fit_k4'], 
                      fit_p1=cam_optimize['fit_p1'], 
                      fit_p2=cam_optimize['fit_p2'],  
                      fit_b1=cam_optimize['fit_b1'], 
                      fit_b2=cam_optimize['fit_b2'],
                      adaptive_fitting=False)
    
    # Filter by reconstruction uncertainty
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
    values = fltr.values.copy()
    values.sort()
    # Remove worst X% of points
    threshold_index = int(len(values) * (1 - ru_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)  # Ensure valid index
    ru_thresh = values[threshold_index]
    # Use absolute minimum if calculated threshold is too low
    if ru_thresh < ru_config['min_threshold']:
        ru_thresh = ru_config['min_threshold']
    fltr.removePoints(ru_thresh)
    print(f"Removed points with reconstruction uncertainty > {ru_thresh:.1f}")

    # Re-optimize cameras
    chunk.optimizeCameras(fit_f=cam_optimize['fit_f'], 
                      fit_cx=cam_optimize['fit_cx'], 
                      fit_cy=cam_optimize['fit_cy'],  
                      fit_k1=cam_optimize['fit_k1'], 
                      fit_k2=cam_optimize['fit_k2'], 
                      fit_k3=cam_optimize['fit_k3'], 
                      fit_k4=cam_optimize['fit_k4'], 
                      fit_p1=cam_optimize['fit_p1'], 
                      fit_p2=cam_optimize['fit_p2'],  
                      fit_b1=cam_optimize['fit_b1'], 
                      fit_b2=cam_optimize['fit_b2'],
                      adaptive_fitting=False)

    # Filter by projection accuracy
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - pa_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)  # Ensure valid index
    pa_thresh = values[threshold_index]
    if pa_thresh < pa_config['min_threshold']:
        pa_thresh = pa_config['min_threshold']
    fltr.removePoints(pa_thresh)
    print(f"Removed points with projection accuracy > {pa_thresh:.1f}")
    
    # Re-optimize cameras
    chunk.optimizeCameras(fit_f=cam_optimize['fit_f'], 
                      fit_cx=cam_optimize['fit_cx'], 
                      fit_cy=cam_optimize['fit_cy'],  
                      fit_k1=cam_optimize['fit_k1'], 
                      fit_k2=cam_optimize['fit_k2'], 
                      fit_k3=cam_optimize['fit_k3'], 
                      fit_k4=cam_optimize['fit_k4'], 
                      fit_p1=cam_optimize['fit_p1'], 
                      fit_p2=cam_optimize['fit_p2'],  
                      fit_b1=cam_optimize['fit_b1'], 
                      fit_b2=cam_optimize['fit_b2'],
                      adaptive_fitting=False)

    # Initial pass of reprojection error filtering
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - re_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)  # Ensure valid index
    re_thresh = values[threshold_index]
    if re_thresh < re_config['min_threshold']:
        re_thresh = re_config['min_threshold']
    fltr.removePoints(re_thresh)
    print(f"Removed points with reprojection error > {re_thresh:.2f}")

    # Final optimization for this stage
    chunk.optimizeCameras(fit_f=cam_optimize['fit_f'], 
                      fit_cx=cam_optimize['fit_cx'], 
                      fit_cy=cam_optimize['fit_cy'],  
                      fit_k1=cam_optimize['fit_k1'], 
                      fit_k2=cam_optimize['fit_k2'], 
                      fit_k3=cam_optimize['fit_k3'], 
                      fit_k4=cam_optimize['fit_k4'], 
                      fit_p1=cam_optimize['fit_p1'], 
                      fit_p2=cam_optimize['fit_p2'],  
                      fit_b1=cam_optimize['fit_b1'], 
                      fit_b2=cam_optimize['fit_b2'],
                      adaptive_fitting=False)
    print("Stage 1 filtering complete")

    return ru_thresh, pa_thresh, re_thresh

def filter_tie_points_usgs_part2(chunk, config):
    """
    Second stage of USGS point filtering - additional pass for reprojection error
    """
    print("Performing USGS-style point filtering (stage 2)...")
    
    re_config = config['tie_point_filtering']['reprojection_error']
    cam_optimize = config['camera']['optimize']
    
    # Re-optimize cameras before second filter pass
    chunk.optimizeCameras(fit_f=cam_optimize['fit_f'], 
                      fit_cx=cam_optimize['fit_cx'], 
                      fit_cy=cam_optimize['fit_cy'],  
                      fit_k1=cam_optimize['fit_k1'], 
                      fit_k2=cam_optimize['fit_k2'], 
                      fit_k3=cam_optimize['fit_k3'], 
                      fit_k4=cam_optimize['fit_k4'], 
                      fit_p1=cam_optimize['fit_p1'], 
                      fit_p2=cam_optimize['fit_p2'],  
                      fit_b1=cam_optimize['fit_b1'], 
                      fit_b2=cam_optimize['fit_b2'],
                      adaptive_fitting=False)
    
    # Second pass of reprojection error filtering
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - re_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)  # Ensure valid index
    re_thresh = values[threshold_index]
    if re_thresh < re_config['min_threshold']:
        re_thresh = re_config['min_threshold']
    fltr.removePoints(re_thresh)
    print(f"Second pass: Removed points with reprojection error > {re_thresh:.2f}")
    
    # Final optimization
    chunk.optimizeCameras(fit_f=cam_optimize['fit_f'], 
                      fit_cx=cam_optimize['fit_cx'], 
                      fit_cy=cam_optimize['fit_cy'],  
                      fit_k1=cam_optimize['fit_k1'], 
                      fit_k2=cam_optimize['fit_k2'], 
                      fit_k3=cam_optimize['fit_k3'], 
                      fit_k4=cam_optimize['fit_k4'], 
                      fit_p1=cam_optimize['fit_p1'], 
                      fit_p2=cam_optimize['fit_p2'],  
                      fit_b1=cam_optimize['fit_b1'], 
                      fit_b2=cam_optimize['fit_b2'],
                      adaptive_fitting=False)
    print("Stage 2 filtering complete")
    
    return re_thresh

def adaptive_subdivide(chunk, config_section_name, config):
    """
    Enable subdivide_task if estimated peak memory > 90% of available physical RAM.
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)

    enabled = [c for c in chunk.cameras if c.enabled]
    if not enabled:
        print(f"[WARN] No enabled cameras for {config_section_name}.")
        return config

    sensor = enabled[0].sensor
    width, height = sensor.width, sensor.height
    mp_per_image = (width * height) / 1e6
    n_images = len(enabled)
    downscale = config[config_section_name].get('downscale', 1)
    neighbors = config[config_section_name].get('max_neighbors', 16)

    # Upper-bound estimate (GB)
    est_needed = 0.8 * n_images * (mp_per_image / downscale**2) * (neighbors / 8) / 1000.0

    threshold = total_gb * 0.9  # 90% of available RAM
    if est_needed > threshold:
        print(f"Estimated {est_needed:.0f} GB > {threshold:.0f} GB (90% of RAM). "
              f"Enabling subdivide_task for {config_section_name}.")
        config[config_section_name]['subdivide_task'] = True
    else:
        config[config_section_name]['subdivide_task'] = False
        print(f"Estimated {est_needed:.0f} GB within ({total_gb:.0f} GB available).")

    print(f"{config_section_name}: n_images={n_images}, mp_per_image={mp_per_image:.1f}, "
          f"downscale={downscale}, neighbors={neighbors}, subdivide_task={config[config_section_name]['subdivide_task']}")

    return config

def activate_license():
    """Activate license using environment variable"""
    global license_obj
    license_key = os.environ.get('METASHAPE_LICENSE_KEY')
    if not license_key:
        print("Error: METASHAPE_LICENSE_KEY environment variable not set")
        return None
        
    print("Activating license...")
    license_obj = Metashape.License()
    license_obj.activate(license_key)
    print("License activated successfully")
    return license_obj

def deactivate_license():
    """Deactivate license - called by signal handlers and normal exit"""
    global license_obj
    if license_obj:
        print("Deactivating license...")
        license_obj.deactivate()
        print("License deactivated successfully")
        license_obj = None

def signal_handler(signum, frame):
    """Handle container shutdown signals"""
    print(f"Received signal {signum}, deactivating license...")
    deactivate_license()
    sys.exit(0)

def setup_license_cleanup():
    """Setup signal handlers and exit cleanup for license"""
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    # Only register SIGHUP on Unix-like systems
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)
    
    atexit.register(deactivate_license)

def has_valid_transform(chunk):
    """Check if chunk has a valid coordinate transform"""
    try:
        transform = chunk.transform
        if not (transform.scale and transform.rotation and transform.translation):
            return False
        # Check if scale is meaningful (not zero or None)
        if not transform.scale or transform.scale == 0:
            return False
        return True
    except:
        return False

class ProgressTimer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.last_printed_percentage = -5
    
    def update(self, p):
        if p - self.last_printed_percentage >= 5 or p >= 100:
            elapsed = float(time.time() - self.start_time)
            if p > 0:
                remaining_sec = (elapsed / p) * (100 - p)
                print('Progress: {:.0f}%, est. time left: {:.0f} sec'.format(p, remaining_sec))
            else:
                print('Progress: {:.0f}%, est. time left: unknown'.format(p))
            self.last_printed_percentage = p

progress_timer = ProgressTimer()