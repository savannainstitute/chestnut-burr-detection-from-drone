"""
YOLO Object Detection
Main file for training, tuning, and inference
"""
import argparse

from pathlib import Path
from typing import Dict

from burr_detection.training import YOLOTrainer
from burr_detection.tuning import YOLOTuner
from burr_detection.inference import YOLOInference
from burr_detection.utils import load_config
        

def run_training(args, config: Dict):
    """Train model using best known hyperparameters from config"""
    print("\n" + "="*80)
    print("Training with Best Known Hyperparameters")
    print("="*80)

    trainer = YOLOTrainer(
        model_size=config['training_params'][0]['model_size'],
        prints_per_epoch=5,
        training_steps=config['training_steps']
    )

    trainer.train(
        yolo_data_dir=str(Path(config['data']['training_dir'])),
        config=config['training_params'][0],
        plot_mode=args.plot_mode, 
        conf_threshold=config['inference']['conf_threshold'],
        iou_threshold=config['inference']['iou_threshold']
    )
    
    print(f"\nTraining complete! Results saved to: {trainer.output_dir}")


def run_tuning(args, config: Dict):
    print("\n" + "="*80)
    print("Starting Hyperparameter Tuning")
    print("="*80)

    tuner = YOLOTuner(
        num_samples=config['ray_tune']['num_samples'],
        max_concurrent_trials=config['ray_tune']['max_concurrent_trials'],
        yolo_data_dir=str(Path(config['data']['training_dir'])),
        training_steps=config['training_steps'],
        points_to_evaluate=config['training_params'],
        tuning_space=config['tuning_space'],
        conf_threshold=config['inference']['conf_threshold'],
        iou_threshold=config['inference']['iou_threshold'],
        plot_mode=args.plot_mode
    )

    tuner.run()

    if not tuner.best_trial:
        print("Tuning completed but no best trial found.")
        return

    print(f"\nTuning complete! Results saved to: {tuner.best_output_dir}")


def run_inference(args, config: Dict):
    print("\n" + "="*80)
    print("Burr detection on unlabeled canopy images")
    print("="*80)

    inference = YOLOInference(
        model_path=config['inference']['model_path'], 
        conf_threshold=config['inference']['conf_threshold'],
        iou_threshold=config['inference']['iou_threshold'],
        plot_mode=args.plot_mode
    )
    inference.run()

    if inference.results_df is not None and not inference.results_df.empty:
        total_burrs = inference.results_df['total_detections'].sum()
        avg_confidence = inference.results_df['avg_confidence'].mean()
        processed_trees = len(inference.results_df)
        avg_burrs_per_tree = int(round(inference.results_df['total_detections'].mean()))
        min_burrs = inference.results_df['total_detections'].min()
        max_burrs = inference.results_df['total_detections'].max()
    else:
        total_burrs = avg_confidence = processed_trees = avg_burrs_per_tree = min_burrs = max_burrs = 0

    summary_text = f"""
    Burr Detection Summary
    {'='*50}
    Processed Trees: {processed_trees}
    Total Burrs Detected: {total_burrs}
    Average Burrs per Tree: {avg_burrs_per_tree}
    Min Burrs: {min_burrs}
    Max Burrs: {max_burrs}
    Overall Average Confidence: {avg_confidence:.3f}

    Model: {inference.model_path.name}
    Confidence Threshold: {inference.conf_threshold}
    IoU Threshold: {inference.iou_threshold}

    Results saved to: {inference.csv_path}
    Plots saved to: {inference.output_dir / 'prediction_plots'}
    {'='*50}
    """
    summary_path = Path(inference.output_dir) / 'detection_summary.txt'
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
  python -m burr_detection.detection --mode tune --plot-mode none
  
  # Training with best known hparams
  python -m burr_detection.detection --mode train `
      --plot-mode subset
  
  
  # Inference on unlabeled data
  python -m burr_detection.detection --mode inference `
      --plot-mode all
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
        help='How many prediction plots to save'
    )

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == 'tune':
        run_tuning(args, config)
    elif args.mode == 'train':
        run_training(args, config)
    else:  # inference
        run_inference(args, config)


if __name__ == "__main__":
    run_detection()