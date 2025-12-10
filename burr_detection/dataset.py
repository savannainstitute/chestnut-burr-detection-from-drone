"""
Preprocessing utilities for YOLO burr detection
Handles dataset preparation, image tiling, and aggregating tree-level detections
"""
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict


def prepare_dataset(images_dir: Path, labels_dir: Path, output_dir: Path, 
                   splits: Tuple[float, float, float] = (0.7, 0.2, 0.1), 
                   seed: int = 666) -> Dict[str, int]:
    """
    Create train/val/test splits and save .txt files with image paths
    
    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing YOLO label files
        output_dir: Directory to save split .txt files
        splits: Tuple of (train, val, test) split ratios
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with counts for each split
    """
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if len(all_image_files) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    missing_labels = []
    for img_file in all_image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            missing_labels.append(img_file.name)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no corresponding labels")
        all_image_files = [f for f in all_image_files if f.name not in missing_labels]
    
    random.shuffle(all_image_files)
    train_ratio, val_ratio, _ = splits
    
    train_split = int(train_ratio * len(all_image_files))
    val_split = int((train_ratio + val_ratio) * len(all_image_files))
    
    train_files = all_image_files[:train_split]
    val_files = all_image_files[train_split:val_split]
    test_files = all_image_files[val_split:]
    
    split_files = {
        'train.txt': train_files,
        'val.txt': val_files,
        'test.txt': test_files
    }
    
    for filename, files in split_files.items():
        with open(output_dir / filename, 'w') as f:
            for file_path in files:
                f.write(f"{file_path.absolute()}\n")
    
    counts = {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files)
    }
    
    print(f"\nDataset splits created:")
    print(f"  Train: {counts['train']} images")
    print(f"  Val:   {counts['val']} images")
    print(f"  Test:  {counts['test']} images")
    print(f"\nSplit files saved to: {output_dir}")
    
    return counts


class CanopyTiler:
    """Handle cropping, tiling, and detection reconstruction for canopy images"""
    
    def __init__(self, tile_size: int = 224, overlap: float = 0.2):
        """
        Args:
            tile_size: Size of each tile (assumes square tiles)
            overlap: Overlap ratio between tiles (0.0 to 1.0)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
    
    def crop_canopy_from_polygon(self, image_path: Path, polygon_coords: List[List[float]]) -> np.ndarray:
        """
        Crop a canopy region from drone image using polygon coordinates and mask outside areas
        
        Args:
            image_path: Path to the full drone image
            polygon_coords: List of [x, y] coordinates defining the polygon boundary
            
        Returns:
            Cropped and masked canopy image as numpy array
        """
        image = Image.open(image_path).convert('RGB')
        
        coords_array = np.array(polygon_coords)
        x_min = int(np.floor(coords_array[:, 0].min()))
        y_min = int(np.floor(coords_array[:, 1].min()))
        x_max = int(np.ceil(coords_array[:, 0].max()))
        y_max = int(np.ceil(coords_array[:, 1].max()))
        
        cropped_img = image.crop((x_min, y_min, x_max, y_max))

        mask = Image.new('L', (x_max - x_min, y_max - y_min), 0)
        draw = ImageDraw.Draw(mask)

        adjusted_coords = [(x - x_min, y - y_min) for x, y in polygon_coords]
        draw.polygon(adjusted_coords, fill=255)

        cropped_array = np.array(cropped_img)
        mask_array = np.array(mask)
        cropped_array[mask_array == 0] = 0
        
        return cropped_array
    
    def tile_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Tile an image with sliding window approach
        
        Args:
            image: Image as numpy array (can be cropped canopy or full image)
            
        Returns:
            tiles: List of image tiles as numpy arrays
            tile_info: List of dictionaries with tile metadata (x, y positions)
        """
        img_height, img_width = image.shape[:2]
        
        padded_height = ((img_height + self.tile_size - 1) // self.tile_size) * self.tile_size
        padded_width = ((img_width + self.tile_size - 1) // self.tile_size) * self.tile_size
        
        padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        padded_image[:img_height, :img_width, :] = image
        
        tiles = []
        tile_info = []

        for tile_y in range(0, padded_height - self.tile_size + 1, self.stride):
            for tile_x in range(0, padded_width - self.tile_size + 1, self.stride):
                tile = padded_image[tile_y:tile_y+self.tile_size, tile_x:tile_x+self.tile_size, :]

                if np.all(tile == 0):
                    continue
                
                tiles.append(tile)
                tile_info.append({
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'original_width': img_width,
                    'original_height': img_height
                })
        
        return tiles, tile_info
    
    def reconstruct_detections(self, tile_detections: List[Dict], 
                               tile_info: List[Dict]) -> List[Dict]:
        """
        Combine tile predictions back to full image coordinates
        
        Args:
            tile_detections: List of detection results from each tile
                Each dict contains: boxes (xyxy), confidences, labels
            tile_info: List of tile metadata from tile_image()
            
        Returns:
            List of detections in full image coordinates
        """
        all_detections = []
        
        for tile_det, info in zip(tile_detections, tile_info):
            if len(tile_det['boxes']) == 0:
                continue

            boxes = tile_det['boxes'].copy()
            boxes[:, [0, 2]] += info['tile_x']
            boxes[:, [1, 3]] += info['tile_y']

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, info['original_width'])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, info['original_height'])
            
            for i, box in enumerate(boxes):
                all_detections.append({
                    'box': box,
                    'confidence': tile_det['confidences'][i],
                    'label': tile_det['labels'][i] if 'labels' in tile_det else 0
                })
        
        return all_detections
