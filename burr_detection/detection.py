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
from burr_detection.dataset import prepare_dataset_splits, burr_tile_group_key
from burr_detection.utils import load_config, plot_dataset_samples, get_output_dir
        

def run_training(args, config: Dict):
    """Train model using best known hyperparameters from config"""
    print("\n" + "="*80)
    print("Training with Best Known Hyperparameters")
    print("="*80)

    trainer = YOLOTrainer(
        model_size=config['training_params'][0]['model_size'],
        prints_per_epoch=5,
        training_steps=config['training_steps'],
        score_weights=config.get('score_weights')
    )

    trainer.train(
        yolo_data_dir=str(Path(config['data']['training_dir'])),
        config=config['training_params'][0],
        plot_mode=args.plot_mode,
        conf_threshold=config['inference']['conf_threshold'],
        iou_threshold=config['inference']['iou_threshold'],
        outputs_dir=config['data'].get('outputs_dir')
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
        plot_mode=args.plot_mode,
        score_weights=config.get('score_weights'),
        analysis_enabled=config.get('analysis', {}).get('enabled', True),
        analysis_top_n=config.get('analysis', {}).get('top_n', 10),
        outputs_dir=config['data'].get('outputs_dir', 'burr_detection/sample_data/training/outputs')
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
        image_selections_path=config['data']['image_selections'],
        conf_threshold=config['inference']['conf_threshold'],
        iou_threshold=config['inference']['iou_threshold'],
        plot_mode=args.plot_mode,
        global_nms_iou=config['inference'].get('global_nms_iou', 0.3),
        tile_batch_size=config['inference'].get('tile_batch_size', 96),
        outputs_dir=config['data'].get('outputs_dir', 'burr_detection/sample_data/training/outputs')
    )
    inference.run()

    print(f"Inference complete! Results saved to: {inference.output_dir}")


def run_preprocess(args, config: Dict):
    """Build the training set, then save QA + (advisory) FN-audit reports.

    If config.data.full_res_images_dir + polygon_labels_dir are set, runs the
    in-module polygon tiler (canopy mask -> tile -> clip polygons to bboxes ->
    denylist/quality filters -> group-aware split) to (re)generate training_dir.
    Otherwise just (re)splits an existing tile set. Then saves QA sample overlays
    and, if data.audit_model is set, an advisory FN-audit report (not applied).
    """
    from burr_detection.dataset import create_tiled_dataset
    from burr_detection.utils import augment_labels_with_model

    print("\n" + "="*80)
    print("Preprocessing: tile (optional) + group-aware split + label augmentation + QA")
    print("="*80)

    data = config['data']
    training_dir = Path(data['training_dir'])
    split_cfg = config.get('split', {})
    fracs = tuple(split_cfg.get('fracs', [0.7, 0.2, 0.1]))
    seed = split_cfg.get('seed', 666)

    src_images = data.get('full_res_images_dir')
    src_labels = data.get('polygon_labels_dir')
    if src_images and src_labels:
        tcfg = data.get('tiling', {}) or {}
        denylist = None
        deny_path = data.get('incorrect_tiles_denylist')
        if deny_path and Path(deny_path).exists():
            denylist = set(Path(deny_path).read_text().split())
        print(f"Tiling full-resolution images from: {src_images}")
        create_tiled_dataset(
            images_dir=src_images,
            labels_dir=src_labels,
            output_dir=training_dir,
            canopy_dir=data.get('canopy_labels_dir'),
            denylist=denylist,
            tile_size=tcfg.get('tile_size', 224),
            overlap=tcfg.get('overlap', 0.2),
            min_canopy_frac=tcfg.get('min_canopy_frac', 0.15),
            bg_keep_ratio=tcfg.get('bg_keep_ratio', 0.3),
            dedup_iou=tcfg.get('dedup_iou', 0.8),
            seed=seed,
        )
    else:
        prepare_dataset_splits(
            images_dir=training_dir / 'images',
            labels_dir=training_dir / 'labels',
            output_dir=training_dir,
            splits=fracs, seed=seed, group_key_fn=burr_tile_group_key,
        )
    print(f"\nDataset ready at: {training_dir}")

    out_base = data.get('outputs_dir') or str(training_dir.parent / "outputs")
    out_dir = get_output_dir(out_base, "preprocess")

    if args.plot_mode != 'none':
        num_samples = 24 if args.plot_mode == 'all' else 8
        print(f"\nSaving {num_samples} QA sample plots (train split)...")
        plot_dataset_samples(
            split_txt=training_dir / 'train.txt',
            labels_dir=training_dir / 'labels',
            save_dir=out_dir / "qa_samples",
            num_samples=num_samples, seed=seed, class_names={0: 'Chestnut-burr'},
        )

    audit_model = data.get('audit_model')
    if audit_model and Path(audit_model).exists():
        aug_conf = data.get('augment_conf', 0.4)
        print(f"\nAugmenting labels with {Path(audit_model).name} (conf>={aug_conf}, "
              f"containment-deduped) to recover missed burrs across all splits...")
        for split in ('train', 'val', 'test'):
            augment_labels_with_model(
                tiled_dir=training_dir, model_path=audit_model, split=split,
                conf=aug_conf, viz_dir=out_dir / f"augment_{split}", viz_n=6,
            )
    else:
        print("\nNo data.audit_model set; skipping label augmentation.")

    print(f"\nPreprocess reports -> {out_dir}")


def _apply_data_root(config, data_root):
    """Point the pipeline at a user dataset without editing the committed config.

    Derives the layout produced by the prep script and overrides config['data']:
      <root>/tiled                              -> training_dir
      <root>/outputs                            -> outputs_dir
      <root>/full_canopy/{images,labels,canopy} -> tiling source
      <root>/incorrect_tiles_denylist.txt       -> denylist (if present)
      <root>/reference/best_tuned_yolov8s.pt    -> audit/augmentation model (if present)
    """
    root = Path(data_root)
    d = config.setdefault('data', {})
    d['training_dir'] = str(root / 'tiled')
    d['outputs_dir'] = str(root / 'outputs')
    d['full_res_images_dir'] = str(root / 'full_canopy' / 'images')
    d['polygon_labels_dir'] = str(root / 'full_canopy' / 'labels')
    d['canopy_labels_dir'] = str(root / 'full_canopy' / 'canopy')
    deny = root / 'incorrect_tiles_denylist.txt'
    d['incorrect_tiles_denylist'] = str(deny) if deny.exists() else None
    model = root / 'reference' / 'best_tuned_yolov8s.pt'
    if model.exists():
        d['audit_model'] = str(model)
    print(f"Using --data-root {root} (training_dir={d['training_dir']})")
    return config


def run_detection():
    parser = argparse.ArgumentParser(
        description='YOLO Burr Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess: group-aware splits + QA sample plots (run before train/tune)
  python -m burr_detection.detection --mode preprocess --plot-mode subset

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
        choices=['preprocess', 'tune', 'train', 'inference'],
        default='inference',
        help='Operation mode: preprocess dataset (splits + QA samples), tune hyperparameters, train model, or run inference'
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

    # Point at your own dataset without editing the committed (sample) config.
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Root of your dataset; derives tiled/, full_canopy/, outputs/, the denylist '
             'and reference model, overriding the sample paths. Unset = use sample data.'
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        _apply_data_root(config, args.data_root)

    if args.mode == 'preprocess':
        run_preprocess(args, config)
    elif args.mode == 'tune':
        run_tuning(args, config)
    elif args.mode == 'train':
        run_training(args, config)
    else:  # inference
        run_inference(args, config)


if __name__ == "__main__":
    run_detection()