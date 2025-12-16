from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import random
from ultralytics import YOLO

from burr_detection.utils import plot_ground_truth_vs_predictions, apply_nms, get_output_dir
from burr_detection.dataset import CanopyTiler

class YOLOInference:
    def __init__(self, model_path, conf_threshold, iou_threshold, plot_mode='none'):
        self.model_path = self._get_model_path(model_path)
        print(f"\nLoading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.selections = self._load_selections("image_selection/sample_data/outputs/best_image_selections.json")
        self.tiler = CanopyTiler(tile_size=224, overlap=0.2)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.plot_mode = plot_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = get_output_dir("burr_detection/sample_data/inference/outputs", "inference", self.timestamp)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_trees_dir = self.output_dir / 'preprocessed_trees'
        self.preprocessed_trees_dir.mkdir(exist_ok=True)
        self.results_data = []
        self.all_predictions = []
        self.results_df = None
        self.csv_path = None

    def _get_model_path(self, model_path):
        if model_path is not None:
            return Path(model_path)
        # Auto-detect latest model in outputs
        outputs_dir = Path("burr_detection/sample_data/training/outputs")
        tuning_dirs = sorted(outputs_dir.glob("tuning_*"), reverse=True)
        training_dirs = sorted(outputs_dir.glob("training_*"), reverse=True)
        candidate_dirs = tuning_dirs + training_dirs
        if not candidate_dirs:
            raise FileNotFoundError("No model outputs found. Please run training or tuning first.")
        latest_dir = candidate_dirs[0]
        model_files = list(latest_dir.glob("best_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {latest_dir}")
        print(f"Using model from latest run: {latest_dir.name}")
        return model_files[0]

    def _load_selections(self, selections_path):
        selections_path = Path(selections_path)
        print(f"Loading canopy selections: {selections_path}")
        with open(selections_path, 'r') as f:
            return json.load(f)

    def run(self):
        print(f"\nProcessing {len(self.selections)} trees...")
        for tree_id, tree_data in self.selections.items():
            image_path = Path(tree_data['image_path'])
            if not image_path.exists():
                print(f"Warning: Image not found for {tree_id}: {image_path}")
                continue
            print(f"\nProcessing tree {tree_id}...")
            polygon_coords = tree_data['polygon_coords']
            cropped_canopy = self.tiler.crop_canopy_from_polygon(image_path, polygon_coords)
            canopy_save_path = self.preprocessed_trees_dir / f"tree_{tree_id}.jpg"
            Image.fromarray(cropped_canopy).save(canopy_save_path)
            tiles, tile_info = self.tiler.tile_image(cropped_canopy)
            print(f"  {len(tiles)} tiles from cropped canopy")
            tile_detections = []
            for tile in tiles:
                tile_pil = Image.fromarray(tile)
                pred = self.model.predict(
                    tile_pil,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )[0]
                if pred.boxes is not None and len(pred.boxes) > 0:
                    tile_detections.append({
                        'boxes': pred.boxes.xyxy.cpu().numpy(),
                        'confidences': pred.boxes.conf.cpu().numpy(),
                        'labels': np.zeros(len(pred.boxes))
                    })
                else:
                    tile_detections.append({
                        'boxes': np.array([]),
                        'confidences': np.array([]),
                        'labels': np.array([])
                    })
            all_detections = self.tiler.reconstruct_detections(tile_detections, tile_info)
            raw_count = len(all_detections)
            filtered_detections = apply_nms(all_detections, self.iou_threshold)
            nms_count = len(filtered_detections)
            print(f"  Post-NMS detections: {nms_count} (removed {raw_count - nms_count} duplicates)")
            if nms_count > 0:
                avg_conf = np.mean([d['confidence'] for d in filtered_detections])
                print(f"  Average confidence: {avg_conf:.3f}")
            else:
                avg_conf = 0.0
                print(f"  Average confidence: no detections")
            self.all_predictions.append((canopy_save_path, filtered_detections))
            detection_coords = str([
                (float(d['box'][0]), float(d['box'][1]),
                 float(d['box'][2]), float(d['box'][3]),
                 float(d['confidence']))
                for d in filtered_detections
            ])
            self.results_data.append({
                'tree_id': tree_id,
                'image_path': str(image_path),
                'total_detections': nms_count,
                'avg_confidence': avg_conf,
                'detection_coords': detection_coords
            })
        self.results_df = pd.DataFrame(self.results_data)
        self.csv_path = self.output_dir / 'tree_burr_detections.csv'
        self.results_df.to_csv(self.csv_path, index=False)

        if self.plot_mode == 'none':
            predictions_to_plot = []
        elif self.plot_mode == 'subset':
            sample_size = min(15, len(self.all_predictions))
            predictions_to_plot = random.sample(self.all_predictions, sample_size)
            print(f"\nSaving plots for {sample_size} random trees...")
        else:  # 'all'
            predictions_to_plot = self.all_predictions
            print(f"\nSaving plots for all trees...")

        if predictions_to_plot:
            plot_dir = self.output_dir / 'prediction_plots'
            plot_dir.mkdir(exist_ok=True)
            plot_ground_truth_vs_predictions(
                predictions=predictions_to_plot,
                labels_dir=None,  # No ground truth for unlabeled data
                original_images_dir=self.preprocessed_trees_dir,
                save_dir=plot_dir,
                conf_threshold=self.conf_threshold
            )