"""
YOLO Hyperparameter Tuning Module
Uses Ray Tune with Optuna search for hyperparameter optimization
"""
import os
import sys
import tempfile
import pickle
from pathlib import Path
import time
import shutil
import json
import random

import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import session
from ultralytics import YOLO

from burr_detection.training import YOLOTrainer
from burr_detection.utils import (set_seed, is_notebook, convert_tuning_space, get_output_dir,
                                  evaluate_test_set, plot_ground_truth_vs_predictions)

class YOLOTuner:
    def __init__(
        self,
        num_samples=50,
        max_concurrent_trials=1,
        yolo_data_dir=None,
        training_steps=None,
        points_to_evaluate=None,
        tuning_space=None,
        conf_threshold=0.5,
        iou_threshold=0.45,
        plot_mode='subset'
    ):
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.yolo_data_dir = str(Path(yolo_data_dir).absolute()) if yolo_data_dir else None
        self.training_steps = training_steps if training_steps is not None else []
        self.points_to_evaluate = points_to_evaluate if points_to_evaluate is not None else []
        self.tuning_space = tuning_space if tuning_space is not None else {}
        self.plot_mode = plot_mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.results = None
        self.best_trial = None
        self.best_trial_preds = None
        self.best_output_dir = None
        self.best_model_path = None
        self.best_trial_config = None

        if not self.training_steps:
            raise ValueError("training_steps schedule must be provided for hyperparameter tuning.")
        if not self.tuning_space:
            raise ValueError("tuning_space config must be provided for hyperparameter search.")


    def train_yolo_with_ray(self, config):
        """Training function for Ray Tune trials"""
        from ultralytics.utils import SETTINGS
        SETTINGS["raytune"] = False

        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
        os.chdir(str(Path(__file__).parent.parent)) # handle relative paths
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

        trial_dir = Path(session.get_trial_dir())
        output_dir = trial_dir / "yolo_output"
        output_dir.mkdir(parents=True, exist_ok=True)

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
                    except Exception:
                        pass

                report_metrics = {
                    'training_iteration': total_epochs,
                    'step': current_step,
                    'epoch': metrics['epoch'] if 'epoch' in metrics else total_epochs,
                    'train_loss': metrics['train_loss'] if 'train_loss' in metrics else 0.0,
                    'train_box_loss': metrics['train_box_loss'] if 'train_box_loss' in metrics else 0.0,
                    'train_cls_loss': metrics['train_cls_loss'] if 'train_cls_loss' in metrics else 0.0,
                    'train_dfl_loss': metrics['train_dfl_loss'] if 'train_dfl_loss' in metrics else 0.0,
                    'lr': metrics['lr'] if 'lr' in metrics else 0.0,
                    'val_precision': metrics['val_precision'] if 'val_precision' in metrics else 0.0,
                    'val_recall': metrics['val_recall'] if 'val_recall' in metrics else 0.0,
                    'val_f1': metrics['val_f1'] if 'val_f1' in metrics else 0.0,
                    'val_mAP50': metrics['val_mAP50'] if 'val_mAP50' in metrics else 0.0,
                    'val_fitness': metrics['val_fitness'] if 'val_fitness' in metrics else 0.0,
                    'val_loss': metrics['val_loss'] if 'val_loss' in metrics else 0.0,
                    'val_box_loss': metrics['val_box_loss'] if 'val_box_loss' in metrics else 0.0,
                    'val_cls_loss': metrics['val_cls_loss'] if 'val_cls_loss' in metrics else 0.0,
                    'val_dfl_loss': metrics['val_dfl_loss'] if 'val_dfl_loss' in metrics else 0.0
                }

                try:
                    tune.report(
                        report_metrics,
                        checkpoint=tune.Checkpoint.from_directory(checkpoint_dir)
                    )
                except Exception:
                    pass

        trainer = YOLOTrainer(
            model_size=config["model_size"],
            prints_per_epoch=5,
            ray_tune_callback=ray_tune_callback,
            training_steps=self.training_steps
        )

        if checkpoint:
            self._resume_training(
                trainer, self.yolo_data_dir, config, start_step,
                step_epochs_completed, total_epochs_so_far, yolo_checkpoint_path,
                ray_tune_callback
            )
        else:
            trainer.train(self.yolo_data_dir, config=config)

        return {}

    def _resume_training(self, trainer, yolo_data_dir, config, start_step,
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

        return trainer.train(yolo_data_dir, config=resume_config)

    def run(self, run_name=None):
        """Run hyperparameter tuning with Ray Tune"""

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"YOLO_Optuna_{timestamp}"

        param_space = convert_tuning_space(self.tuning_space)

        def trial_dirname_creator(trial):
            return f"trial_{trial.trial_id}"

        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=100,
            grace_period=10,
            reduction_factor=3
        )

        optuna_search = OptunaSearch(
            metric="val_loss",
            mode="min",
            points_to_evaluate=self.points_to_evaluate,
        )
        optuna_search = ConcurrencyLimiter(optuna_search, max_concurrent=self.max_concurrent_trials)

        metric_columns = [
            "epoch", "step",
            "train_loss", "train_box_loss", "train_cls_loss", "train_dfl_loss",
            "val_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss",
            "val_precision", "val_recall", "val_f1", "val_mAP50", "val_fitness"
        ]
        parameter_columns = [
            "model_size", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
            "warmup_epochs", "warmup_bias_lr", "warmup_momentum",
            "box_gain", "cls_gain", "dfl_gain",
            "hsv_h", "hsv_s", "hsv_v",
            "degrees", "scale", "shear", "perspective",
            "mosaic", "mixup", "copy_paste", "dropout"
        ]
        if is_notebook():
            reporter = tune.JupyterNotebookReporter(
                metric_columns=metric_columns,
                parameter_columns=parameter_columns,
                max_progress_rows=50,
                max_column_length=30
                sort_by_metric=True
            )
        else:
            reporter = tune.CLIReporter(
                metric_columns=metric_columns,
                parameter_columns=parameter_columns,
                max_progress_rows=50,
                max_column_length=30,
                sort_by_metric=True
            )

        max_concurrent = self.max_concurrent_trials
        available_gpus = torch.cuda.device_count()
        available_cpus = os.cpu_count() or 2

        if available_gpus > 0:
            gpus_per_trial = available_gpus / max_concurrent
            cpus_per_trial = max(1, available_cpus // max_concurrent)
            resources = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        else:
            cpus_per_trial = max(1, available_cpus // max_concurrent)
            resources = {"cpu": cpus_per_trial}
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
            param_space=param_space
        )

        self.results = tuner.fit()

        try:
            self.best_output_dir = get_output_dir("burr_detection/sample_data/training/outputs", "tuning", timestamp)
            self.best_output_dir.mkdir(parents=True, exist_ok=True)
            results_df = self.results.get_dataframe(filter_metric='val_loss', filter_mode='min')
            results_df.to_csv(self.best_output_dir / "all_tuning_history.csv", index=False)

            # Find the best trial (lowest val loss of any epoch)
            best_overall_trial = min(
                self.results,
                key=lambda trial: trial.metrics_dataframe['val_loss'].min()
            )
            best_overall_trial.metrics_dataframe.to_csv(self.best_output_dir / "best_trial_training_history.csv", index=False)

            # Find best epoch weights by val f1
            best_f1_idx = best_overall_trial.metrics_dataframe['val_f1'].idxmax()
            best_f1_row = best_overall_trial.metrics_dataframe.loc[best_f1_idx]
            checkpoint_dir_name = best_f1_row['checkpoint_dir_name']
            model_size = str(best_f1_row.get('config/model_size', 'yolo_model')).replace('.pt', '')

            best_trial_dir = Path(best_overall_trial.path)
            checkpoint_dir = best_trial_dir / checkpoint_dir_name
            yolo_model_path = checkpoint_dir / "yolo_model.pt"

            self.best_model_path = self.best_output_dir / f"best_{model_size}_model.pt"
            shutil.copy2(yolo_model_path, self.best_model_path)

            best_config_path = best_trial_dir / "params.json"
            if best_config_path.exists():
                shutil.copy2(best_config_path, self.best_output_dir / "best_trial_config.json")
                with open(best_config_path, "r") as f:
                    self.best_trial_config = json.load(f)
            else:
                self.best_trial_config = {}

            self.best_trial = {
                "path": str(best_trial_dir),
                "config": self.best_trial_config,
                "metrics_dataframe": best_overall_trial.metrics_dataframe,
                "model_path": str(self.best_model_path),
                "output_dir": str(self.best_output_dir)
            }

            dataset_yaml = Path(self.yolo_data_dir) / "dataset.yml"
            if dataset_yaml.exists():
                self.best_trial_preds = evaluate_test_set(
                    model_path=self.best_model_path,
                    training_dir=Path(self.yolo_data_dir),
                    output_dir=self.best_output_dir,
                    plot_mode=self.plot_mode,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )
            if self.best_trial_preds:
                images_dir = Path(self.yolo_data_dir) / "images"
                labels_dir = Path(self.yolo_data_dir) / "labels"
                plot_dir = Path(self.best_output_dir) / "prediction_plots"

                if self.plot_mode == 'none':
                    predictions_to_plot = []
                elif self.plot_mode == 'subset':
                    sample_size = min(15, len(self.best_trial_preds))
                    predictions_to_plot = random.sample(self.best_trial_preds, sample_size)
                else:  # 'all'
                    predictions_to_plot = self.best_trial_preds

                if predictions_to_plot:
                    plot_ground_truth_vs_predictions(
                        predictions=predictions_to_plot,
                        labels_dir=labels_dir,
                        original_images_dir=images_dir,
                        save_dir=plot_dir,
                        conf_threshold=self.conf_threshold
                    )

        except Exception as e:
            print(f"Error in post-processing best trial: {e}")
            self.best_trial = None
            self.best_trial_preds = None