"""
YOLO Hyperparameter Tuning Module
Uses Ray Tune with Optuna search for hyperparameter optimization
"""
import os
import sys
import tempfile
import pickle
from pathlib import Path
import yaml
import time

import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ultralytics import YOLO

from burr_detection.training import YOLOTrainer
from burr_detection.utils import set_seed

class YOLOTuner:
    def __init__(self, num_samples=50, yolo_data_dir=None, points_to_evaluate=None):
        self.num_samples = num_samples
        self.yolo_data_dir = str(Path(yolo_data_dir).absolute()) if yolo_data_dir else None
        self.points_to_evaluate = points_to_evaluate
    
    def train_yolo_with_ray(self, config):
        """Training function for Ray Tune trials"""
        from ultralytics.utils import SETTINGS
        SETTINGS["raytune"] = False 
        
        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
        set_seed(666)
        
        checkpoint = tune.get_checkpoint()
        start_step = 0
        step_epochs_completed = 0
        total_epochs_so_far = 0
        yolo_checkpoint_path = None
        
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                yolo_model_path = Path(checkpoint_dir) / "yolo_model.pt"
                
                if data_path.exists():
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_step = checkpoint_state["current_step"]
                    step_epochs_completed = checkpoint_state["step_epochs_completed"] 
                    total_epochs_so_far = checkpoint_state["total_epochs_so_far"]
                
                if yolo_model_path.exists():
                    yolo_checkpoint_path = str(yolo_model_path)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "yolo_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_path = Path(self.yolo_data_dir).absolute()
            dataset_config = {
                "path": str(base_path),
                "train": str(base_path / "train.txt"),
                "val": str(base_path / "val.txt"),
                "test": str(base_path / "test.txt"),
                "names": {0: "Chestnut-burr"}
            }
            yaml_path = output_dir / "dataset.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_config, f)

            def ray_tune_callback(metrics):
                current_step = getattr(trainer, 'current_step', 1)
                step_epoch = trainer.current_epoch
                total_epochs = getattr(trainer, 'total_epochs_so_far', 0) + step_epoch
                
                checkpoint_data = {
                    "current_step": current_step,
                    "step_epochs_completed": step_epoch,
                    "total_epochs_so_far": total_epochs,
                    "training_iteration": total_epochs
                }

                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    
                    if hasattr(trainer, 'model') and trainer.model:
                        yolo_model_path = Path(checkpoint_dir) / "yolo_model.pt"
                        try:
                            trainer.model.save(str(yolo_model_path))
                        except:
                            pass
                    
                    report_metrics = {
                        'training_iteration': total_epochs,
                        'step': current_step,
                        'epoch': metrics.get('epoch', total_epochs),
                        'train_loss': metrics.get('train_loss', 0.0),
                        'train_box_loss': metrics.get('train_box_loss', 0.0),
                        'train_cls_loss': metrics.get('train_cls_loss', 0.0),
                        'train_dfl_loss': metrics.get('train_dfl_loss', 0.0),
                        'lr': metrics.get('lr', 0.0),
                        'val_precision': metrics.get('val_precision', 0.0),
                        'val_recall': metrics.get('val_recall', 0.0),
                        'val_f1': metrics.get('val_f1', 0.0),
                        'val_mAP50': metrics.get('val_mAP50', 0.0),
                        'val_mAP': metrics.get('val_mAP', 0.0),
                        'val_loss': metrics.get('val_loss', 0.0),
                        'val_box_loss': metrics.get('val_box_loss', 0.0),
                        'val_cls_loss': metrics.get('val_cls_loss', 0.0),
                        'val_dfl_loss': metrics.get('val_dfl_loss', 0.0)
                    }
                    
                    try:
                        tune.report(
                            report_metrics,
                            checkpoint=tune.Checkpoint.from_directory(checkpoint_dir)
                        )
                    except Exception as e:
                        pass

            trainer = YOLOTrainer(
                model_size=config["model_size"], 
                prints_per_epoch=5, 
                ray_tune_callback=ray_tune_callback
            )

            if checkpoint:
                self._resume_training(
                    trainer, self.yolo_data_dir, output_dir, config, start_step, 
                    step_epochs_completed, total_epochs_so_far, yolo_checkpoint_path,
                    ray_tune_callback
                )
            else:
                trainer.train(self.yolo_data_dir, output_dir, config=config)
            
            return {}

    def _resume_training(self, trainer, yolo_data_dir, output_dir, config, start_step, 
                        step_epochs_completed, total_epochs_so_far, yolo_checkpoint_path, 
                        ray_tune_callback):
        """Resume training from checkpoint"""
        trainer.ray_tune_callback = ray_tune_callback
        trainer._original_stdout = sys.stdout
        trainer._original_stderr = sys.stderr
        
        if yolo_checkpoint_path and os.path.exists(yolo_checkpoint_path):
            trainer.model = YOLO(yolo_checkpoint_path)
        
        resume_config = {
            **config,
            '_resume_from_step': start_step,
            '_resume_step_epochs': step_epochs_completed,
            '_resume_total_epochs': total_epochs_so_far
        }
        
        return trainer.train(yolo_data_dir, output_dir, config=resume_config)

    def run(self, run_name=None):
        """Run hyperparameter tuning with Ray Tune"""
        
        if run_name is None:
            run_name = f"YOLO_Optuna_{time.strftime('%Y%m%d_%H%M%S')}"
            
        config = {
            "model_size": tune.choice(["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"]),
            "imgsz": tune.choice([224, 320, 416]), 
            "optimizer": tune.choice(["AdamW", "SGD", "Adam"]),
            "lr0": tune.loguniform(0.0005, 0.01),
            "lrf": tune.loguniform(0.001, 0.1),
            "momentum": tune.uniform(0.85, 0.98),
            "weight_decay": tune.loguniform(0.0001, 0.01),
            "warmup_epochs": tune.uniform(1.0, 5.0),
            "warmup_momentum": tune.uniform(0.7, 0.9),
            "box_gain": tune.uniform(12.0, 20.0), 
            "cls_gain": tune.uniform(0.5, 2.0),   
            "dfl_gain": tune.uniform(1.5, 3.0),
            "hsv_h": tune.uniform(0.001, 0.005),   
            "hsv_s": tune.uniform(0.1, 0.3),     
            "hsv_v": tune.uniform(0.1, 0.25),      
            "degrees": tune.uniform(0.0, 2.0),    
            "scale": tune.uniform(0.9, 1.0),   
            "shear": tune.uniform(0.0, 0.5),     
            "perspective": tune.uniform(0.0, 0.00001),
            "mosaic": tune.uniform(0.0, 0.2),      
            "mixup": tune.uniform(0.0, 0.05),    
            "copy_paste": tune.uniform(0.0, 0.2),  
            "dropout": tune.uniform(0.0, 0.2),
        }
        
        def trial_dirname_creator(trial):
            return f"trial_{trial.trial_id}"

        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=100,
            grace_period=10,
            reduction_factor=3
        )

        points_to_evaluate = []
        if self.points_to_evaluate:
            initial_point = {**self.points_to_evaluate}
            points_to_evaluate = [initial_point]

        optuna_search = OptunaSearch(
            metric="val_loss",
            mode="min",
            points_to_evaluate=points_to_evaluate
        )
        optuna_search = ConcurrencyLimiter(optuna_search, max_concurrent=1)
        reporter = tune.CLIReporter(
            metric_columns=[
                "epoch", "step", 
                "train_loss", "train_box_loss", "train_cls_loss", "train_dfl_loss",
                "val_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss",
                "val_precision", "val_recall", "val_f1", "val_mAP50", "val_mAP"
            ],
            parameter_columns=[
                "model_size", "lr0", "optimizer", "imgsz", "box_gain", "cls_gain", 
                "weight_decay", "momentum"
            ],
            max_progress_rows=50,
            sort_by_metric=True
        )

        if torch.cuda.is_available():
            resources = {"cpu": 12, "gpu": 1}
        else:
            resources = {"cpu": 12}
            print("WARNING: CUDA not available. Tuning with CPU only.")

        tuner = tune.Tuner(
            tune.with_resources(self.train_yolo_with_ray, resources=resources),
            tune_config=tune.TuneConfig(
                mode="min",
                metric="val_loss",
                search_alg=optuna_search,
                scheduler=asha_scheduler,
                num_samples=self.num_samples,
                trial_dirname_creator=trial_dirname_creator,
            ),
            run_config=tune.RunConfig(
                name=run_name,
                progress_reporter=reporter,
            ),
            param_space=config
        )
        
        results = tuner.fit()
        
        try:
            best_trial = results.get_best_result("val_loss", "min")
            print("Best trial config:", best_trial.config)
            print("Best trial final validation f1:", best_trial.metrics["val_f1"])
            print("Best trial final validation precision:", best_trial.metrics["val_precision"])
            print("Best trial final validation recall:", best_trial.metrics["val_recall"])
            print("Best trial final validation mAP50:", best_trial.metrics["val_mAP50"])
            print("Best trial final validation mAP:", best_trial.metrics["val_mAP"])
            print("Best trial final validation loss:", best_trial.metrics["val_loss"])
            
            return results, best_trial
        except Exception as e:
            print(f"Error getting best trial: {e}")
            return results, None