"""
YOLO Object Detection CLI
Main entry point for training, tuning, and inference operations
"""
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import random
import json
import pandas as pd
import numpy as np
from typing import Dict
import shutil
from PIL import Image
from ultralytics import YOLO

from burr_detection.training import YOLOTrainer
from burr_detection.tuning import YOLOTuner
from burr_detection.dataset import prepare_dataset, CanopyTiler
from burr_detection.utils import format_test_results, plot_ground_truth_vs_predictions, apply_nms


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_model(training_outputs_dir: Path) -> Path:
    """
    Find the latest trained model in training outputs directory
    
    Args:
        training_outputs_dir: Path to burr_detection/sample_data/training/outputs/
        
    Returns:
        Path to the best model .pt file
    """
    tuning_dirs = list(training_outputs_dir.glob("tuning_*"))
    
    if not tuning_dirs:
        raise FileNotFoundError(
            f"No tuning directories found in {training_outputs_dir}. "
            "Please run tuning first or specify --model path."
        )
    tuning_dirs.sort(reverse=True)
    latest_dir = tuning_dirs[0]
    
    # Find best_*.pt file
    model_files = list(latest_dir.glob("best_*.pt"))
    
    if not model_files:
        raise FileNotFoundError(f"No model file found in {latest_dir}")
    
    print(f"Using model from latest tuning run: {latest_dir.name}")
    return model_files[0]


def evaluate_on_test_set(model_path: Path, training_dir: Path, 
                         output_dir: Path):
    """Evaluate model on test set and save results"""
    
    base_path = Path(training_dir).absolute()
    dataset_config = {
        "train": str(base_path / "train.txt"),
        "val": str(base_path / "val.txt"),
        "test": str(base_path / "test.txt"),
        "names": {0: "Chestnut-burr"}
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    model = YOLO(model_path)
    test_results = model.val(data=str(yaml_path), split='test', verbose=False)
    test_results_print = format_test_results(test_results)
    print(test_results_print)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem.replace("best_", "")

    precision = test_results.results_dict['metrics/precision(B)']
    recall = test_results.results_dict['metrics/recall(B)']
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    test_metrics = {
        'timestamp': timestamp,
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1, 
        'mAP50': test_results.results_dict['metrics/mAP50(B)'],
        'mAP50_95': test_results.results_dict['metrics/mAP50-95(B)'],
        'fitness': test_results.results_dict['fitness'],
        'inference_time_ms': test_results.speed['inference'],
        'model_path': str(model_path)
    }
    
    pd.DataFrame([test_metrics]).to_csv(
        output_dir / "test_results.csv", 
        index=False
    )

def run_tuning(args, config: Dict):
    """Run hyperparameter tuning with Ray Tune"""
    print("\n" + "="*80)
    print("Starting Hyperparameter Tuning")
    print("="*80)
    
    training_dir = Path(config['data']['training_dir'])

    if not (training_dir / 'train.txt').exists():
        print("\nDataset splits not found. Creating splits...")
        prepare_dataset(
            images_dir=training_dir / 'images',
            labels_dir=training_dir / 'labels',
            output_dir=training_dir,
            seed=666
        )

    tuner = YOLOTuner(
        num_samples=args.num_samples,
        yolo_data_dir=str(training_dir),
        points_to_evaluate=config['best_known_hparams'], 
    )
    
    results, best_trial = tuner.run()
    
    if not best_trial:
        print("Tuning completed but no best trial found.")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['data']['training_dir']).parent / 'outputs' / f'tuning_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy best model from checkpoint
    with best_trial.checkpoint.as_directory() as checkpoint_dir:
        yolo_model_path = Path(checkpoint_dir) / "yolo_model.pt"
        model_name = best_trial.config["model_size"].replace(".pt", "")
        best_model_path = output_dir / f"best_{model_name}_model.pt"
        shutil.copy2(yolo_model_path, best_model_path)
    
    if hasattr(best_trial, 'metrics_dataframe'):
        history_df = best_trial.metrics_dataframe
        history_df.to_csv(output_dir / "best_trial_training_history.csv", index=False)
    
    all_trial_data = []
    for result in results:
        trial_data = {'trial_path': result.path}
        if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
            trial_data.update(result.metrics)
        if hasattr(result, 'config') and isinstance(result.config, dict):
            trial_data.update(result.config)
        all_trial_data.append(trial_data)
    
    pd.DataFrame(all_trial_data).to_csv(output_dir / "tuning_history.csv", index=False)

    print("\n" + "="*80)
    print("Evaluating Best Model on Test Set")
    print("="*80)
    
    evaluate_on_test_set(best_model_path, training_dir, output_dir)
    
    print(f"\nTuning complete! Results saved to: {output_dir}")

def run_training(config: Dict):
    """Train model using best known hyperparameters from config"""
    print("\n" + "="*80)
    print("Training with Best Known Hyperparameters")
    print("="*80)
    
    training_dir = Path(config['data']['training_dir'])

    if not (training_dir / 'train.txt').exists():
        print("\nDataset splits not found. Creating splits...")
        prepare_dataset(
            images_dir=training_dir / 'images',
            labels_dir=training_dir / 'labels',
            output_dir=training_dir,
            seed=666
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['data']['training_dir']).parent / 'outputs' / f'training_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = YOLOTrainer(
        model_size=config['best_known_hparams']['model_size'],
        prints_per_epoch=5
    )

    training_config = {
        **config['best_known_hparams'],
    }
    
    results = trainer.train(
        yolo_data_dir=str(training_dir),
        output_dir=str(output_dir),
        config=training_config
    )
    
    best_model_path = results['best_model_path']
    
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)
    
    evaluate_on_test_set(best_model_path, training_dir, output_dir)
    
    print(f"\nTraining complete! Results saved to: {output_dir}")


def run_inference(args, config: Dict):
    """Run inference on unlabeled canopy images"""

    print("\n" + "="*80)
    print("Burr detection on unlabeled canopy images")
    print("="*80)
    
    # Load model
    if args.model:
        model_path = Path(args.model)
    else:
        training_outputs = Path(config['data']['training_dir']).parent / 'outputs'
        model_path = find_latest_model(training_outputs)
    
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    selections_path = Path(args.image_selections)
    print(f"Loading canopy selections: {selections_path}")
    
    with open(selections_path, 'r') as f:
        selections = json.load(f)

    tiler = CanopyTiler(
        tile_size=config['inference']['tile_size'],
        overlap=config['inference']['overlap']
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f'inference_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_trees_dir = output_dir / 'preprocessed_trees'
    preprocessed_trees_dir.mkdir(exist_ok=True)
    
    results_data = []
    all_predictions = []  
    
    print(f"\nProcessing {len(selections)} trees...")
    
    for tree_id, tree_data in selections.items():

        
        image_path = Path(tree_data['image_path'])
        
        if not image_path.exists():
            print(f"Warning: Image not found for {tree_id}: {image_path}")
            continue
        
        print(f"\nProcessing tree {tree_id}...")

        polygon_coords = tree_data['polygon_coords']

        cropped_canopy = tiler.crop_canopy_from_polygon(image_path, polygon_coords)
        canopy_save_path = preprocessed_trees_dir / f"tree_{tree_id}.jpg"
        Image.fromarray(cropped_canopy).save(canopy_save_path)

        tiles, tile_info = tiler.tile_image(cropped_canopy)
        print(f"  {len(tiles)} tiles from cropped canopy")

        tile_detections = []
        for tile in tiles:
            tile_pil = Image.fromarray(tile)
            pred = model.predict(
                tile_pil,
                conf=args.conf_threshold,
                iou=args.iou_threshold,
                verbose=False
            )[0]
            
            if pred.boxes is not None and len(pred.boxes) > 0:
                tile_detections.append({
                    'boxes': pred.boxes.xyxy.cpu().numpy(),
                    'confidences': pred.boxes.conf.cpu().numpy(),
                    'labels': np.zeros(len(pred.boxes))  # All class 0 for burrs
                })
            else:
                tile_detections.append({
                    'boxes': np.array([]),
                    'confidences': np.array([]),
                    'labels': np.array([])
                })

        all_detections = tiler.reconstruct_detections(tile_detections, tile_info)
        
        raw_count = len(all_detections)

        filtered_detections = apply_nms(all_detections, args.iou_threshold)
        
        nms_count = len(filtered_detections)
        
        print(f"  Post-NMS detections: {nms_count} (removed {raw_count - nms_count} duplicates)")

        if nms_count > 0:
            avg_conf = np.mean([d['confidence'] for d in filtered_detections])
            print(f"  Average confidence: {avg_conf:.3f}")
        else:
            avg_conf = 0.0
            print(f"  Average confidence: no detections")

        all_predictions.append((canopy_save_path, filtered_detections))

        detection_coords = str([
            (float(d['box'][0]), float(d['box'][1]), 
             float(d['box'][2]), float(d['box'][3]), 
             float(d['confidence']))
            for d in filtered_detections
        ])
        
        results_data.append({
            'tree_id': tree_id,
            'image_path': str(image_path),
            'total_detections': nms_count,
            'avg_confidence': avg_conf,
            'detection_coords': detection_coords
        })

    results_df = pd.DataFrame(results_data)
    csv_path = output_dir / 'tree_burr_detections.csv'
    results_df.to_csv(csv_path, index=False)

    if args.plot_mode == 'none':
        predictions_to_plot = []
    elif args.plot_mode == 'subset':
        sample_size = min(15, len(all_predictions))
        predictions_to_plot = random.sample(all_predictions, sample_size)
        print(f"\nSaving plots for {sample_size} random trees...")
    else:  # 'all'
        predictions_to_plot = all_predictions
        print(f"\nSaving plots for all trees...")
    
    if predictions_to_plot:
        plot_dir = output_dir / 'prediction_plots'
        plot_dir.mkdir(exist_ok=True)

        plot_ground_truth_vs_predictions(
            predictions=predictions_to_plot,
            labels_dir=None,  # No ground truth for unlabeled data
            original_images_dir=preprocessed_trees_dir,
            save_dir=plot_dir,
            conf_threshold=args.conf_threshold
        )
    
    total_burrs = results_df['total_detections'].sum()
    avg_confidence = results_df['avg_confidence'].mean()
    
    summary_text = f"""
Burr Detection Summary
{'='*50}
Processed Trees: {len(results_df)}
Total Burrs Detected: {total_burrs}
Average Burrs per Tree: {int(round(results_df['total_detections'].mean()))}
Min Burrs: {results_df['total_detections'].min()}
Max Burrs: {results_df['total_detections'].max()}
Overall Average Confidence: {avg_confidence:.3f}

Model: {model_path.name}
Confidence Threshold: {args.conf_threshold}
IoU Threshold: {args.iou_threshold}

Results saved to: {csv_path}
Plots saved to: {plot_dir}
{'='*50}
"""
    
    summary_path = output_dir / 'detection_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)


def run_detection():
    parser = argparse.ArgumentParser(
        description='YOLO Burr Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hyperparameter tuning
  python -m burr_detection.detection --mode tune --num-samples 50
  
  # Training with best known hparams
  python -m burr_detection.detection --mode train
  
  # Inference on unlabeled data
  python -m burr_detection.detection --mode inference `
      --image-selections image_selection/sample_data/outputs/best_image_selections.json `
      --output burr_detection/sample_data/inference/outputs
      --conf-threshold 0.5 `
      --iou-threshold 0.45 `
      --plot-mode subset
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['tune', 'train', 'inference'], 
        default='inference',
        help='Operation mode: tune hyperparameters, train model, or run inference'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default='burr_detection/config.yml',
        help='Path to configuration YAML file'
    )
    
    # Plotting arguments
    parser.add_argument(
        '--plot-mode',
        choices=['all', 'subset', 'none'],
        default='subset',
        help='How many prediction plots to generate'
    )

    # Tuning arguments
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=50,
        help='Number of hyperparameter combinations to try (tune mode only)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model', 
        type=str,
        help='Path to model weights (for inference). If not specified, uses latest tuning run.'
    )
    
    # Inference arguments
    parser.add_argument(
        '--image-selections', 
        type=str,
        help='Path to best_image_selections.json (default from config)'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        default='burr_detection/sample_data/inference/outputs',
        help='Output directory for inference results'
    )
    
    parser.add_argument(
        '--conf-threshold', 
        type=float,
        help='Confidence threshold for detections (default from config)'
    )
    
    parser.add_argument(
        '--iou-threshold', 
        type=float,
        help='IoU threshold for NMS (default from config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set defaults from config if not provided
    if not args.image_selections:
        args.image_selections = config['data']['default_image_selections']
    
    if args.conf_threshold is None:
        args.conf_threshold = config['inference']['conf_threshold']
    
    if args.iou_threshold is None:
        args.iou_threshold = config['inference']['iou_threshold']
    
    # Route to appropriate function
    if args.mode == 'tune':
        run_tuning(args, config)
    elif args.mode == 'train':
        run_training(config)
    else:  # inference
        run_inference(args, config)


if __name__ == "__main__":
    run_detection()