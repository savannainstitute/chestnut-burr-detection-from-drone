"""
YOLO Training Module
Handles multi-step progressive training with early stopping
"""
import time
import math
import sys
import os
import shutil
import pandas as pd
import random
from pathlib import Path
from datetime import timedelta
import logging
import torch
from ultralytics import YOLO

from burr_detection.utils import SmoothedValue, MetricLogger, set_seed, evaluate_test_set, plot_ground_truth_vs_predictions


class YOLOTrainer:
    def __init__(self, model_size="yolo11n.pt", prints_per_epoch=5, ray_tune_callback=None, training_steps=None):
        self.model = YOLO(model_size)
        self.model_size = model_size
        self.prints_per_epoch = prints_per_epoch
        self.ray_tune_callback = ray_tune_callback
        self.training_steps = training_steps
        self.batch_idx = 0
        self.num_batches = 0
        self.current_epoch = 0
        self.epochs = 0
        self.end_time = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.train_metric_logger = MetricLogger(delimiter="  ")
        self.train_metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.validation_metrics = {}
        self.metrics_history = []
        self.yaml_path = None
        self.output_dir = None
        self.final_weights_path = None
        self.training_metrics_path = None
        self.test_preds = None
        self.best_model_path = None
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def train(self, yolo_data_dir, output_dir=None, config=None, plot_mode='none', conf_threshold=0.5, iou_threshold=0.45):
        if config is None:
            config = {}
        set_seed(666)
        os.environ['TQDM_DISABLE'] = '1'
        try:
            from tqdm import tqdm
            tqdm.disable = True
        except:
            pass
        yaml_path = Path(yolo_data_dir) / "dataset.yml"
        train_txt_path = Path(yolo_data_dir) / "train.txt"
        val_txt_path = Path(yolo_data_dir)/ "val.txt"
        test_txt_path = Path(yolo_data_dir) / "test.txt"

        for f in [yaml_path, train_txt_path, val_txt_path, test_txt_path]:
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")

        self.yaml_path = yaml_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print(f"CUDA not available, using CPU for training")

        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path(yolo_data_dir).parent / "outputs" / f"training_{timestamp}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        start_step = config.get('_resume_from_step', 0)
        step_epochs_completed = config.get('_resume_step_epochs', 0)
        total_epochs = config.get('_resume_total_epochs', 0)

        training_steps = self.training_steps if self.training_steps is not None else [
            {"batch": 8, "accumulate": 1, "max_epochs": 50, "patience": 25},
            {"batch": 8, "accumulate": 4, "max_epochs": 50, "patience": 20},
            {"batch": 8, "accumulate": 16, "max_epochs": 50, "patience": 15},
            {"batch": 8, "accumulate": 64, "max_epochs": 50, "patience": 10}
        ]

        lr0 = config.get("lr0", 0.001)
        best_model_path = None
        self.metrics_history = []
        print("\n" + "-" * 80)
        print(f"Starting YOLO training: {self.model_size}")
        print("-" * 80)
        with open(train_txt_path, 'r') as f:
            num_train_images = len(f.readlines())
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        for step_idx in range(start_step, len(training_steps)):
            step = training_steps[step_idx]
            print(f"\nStep {step_idx+1}/{len(training_steps)}")
            print("-" * 60)
            effective_batch = step["batch"] * step["accumulate"]
            scaled_lr = lr0 * math.sqrt(effective_batch / 4)
            if step_idx == start_step and step_epochs_completed > 0:
                epochs_to_run = step["max_epochs"] - step_epochs_completed
                if epochs_to_run <= 0:
                    continue
                print(f"Resuming step {step_idx+1} from epoch {step_epochs_completed+1}")
            else:
                epochs_to_run = step["max_epochs"]
            self.epochs = epochs_to_run
            if best_model_path and os.path.exists(best_model_path):
                self.model = YOLO(best_model_path)
            self._set_trainable_layers(num_layers=step_idx)
            self.current_step = step_idx + 1
            self.total_epochs_so_far = total_epochs
            batches_per_epoch = num_train_images // step["batch"] + (1 if num_train_images % step["batch"] > 0 else 0)
            self.num_batches = batches_per_epoch
            print_freq = max(1, batches_per_epoch // self.prints_per_epoch)
            if self.current_step == 1:
                print(f"Dataset: {num_train_images} images, {batches_per_epoch} batches per epoch")
            print(f"Training with effective batch size {int(step['batch']*step['accumulate'])}, lr={scaled_lr:.6f}")
            print()
            self.model.add_callback("on_train_batch_end", lambda trainer: self._on_batch_end(trainer, print_freq))
            self.model.add_callback("on_train_epoch_end", self._on_epoch_end)
            self.model.add_callback("on_val_end", self._on_val_end)
            self.batch_idx = 0
            self._reset_loggers()
            if hasattr(self, '_last_val_epoch'):
                delattr(self, '_last_val_epoch')
            results = self.model.train(
                data=str(yaml_path),
                epochs=epochs_to_run,
                patience=step["patience"],
                batch=step["batch"],
                imgsz=config.get("imgsz", 416),
                lr0=scaled_lr,
                lrf=config.get("lrf", 0.01),
                optimizer=config.get("optimizer", "AdamW"),
                nbs=effective_batch,
                warmup_epochs=config.get("warmup_epochs", 3),
                warmup_momentum=config.get("warmup_momentum", 0.8),
                warmup_bias_lr=config.get("warmup_bias_lr", 0.0005),
                weight_decay=config.get("weight_decay", 0.0005),
                momentum=config.get("momentum", 0.937),
                box=config.get("box_gain", 7.5),
                cls=config.get("cls_gain", 0.5),
                dfl=config.get("dfl_gain", 1.5),
                hsv_h=config.get("hsv_h", 0.015),
                hsv_s=config.get("hsv_s", 0.7),
                hsv_v=config.get("hsv_v", 0.4),
                degrees=config.get("degrees", 0),
                scale=config.get("scale", 0.5),
                shear=config.get("shear", 0),
                perspective=config.get("perspective", 0),
                mosaic=config.get("mosaic", 1.0),
                mixup=config.get("mixup", 0),
                copy_paste=config.get("copy_paste", 0),
                dropout=config.get("dropout", 0),
                project=str(output_dir),
                name=f"train_step{step_idx+1}",
                exist_ok=True,
                device=device,
                workers=0,
                plots=False,
                save=True,
                save_period=1,
                verbose=False
            )
            best_model_path = str(output_dir / f"train_step{step_idx+1}" / "weights" / "best.pt")
            total_epochs += getattr(results, 'epoch', epochs_to_run)
        print("\n" + "=" * 80)
        print(f"Training complete - {total_epochs} epochs")
        print("=" * 80)
        all_metrics_df = pd.DataFrame(self.metrics_history)
        self.training_metrics_path = output_dir / "training_metrics.csv"
        all_metrics_df.to_csv(self.training_metrics_path, index=False)

        # Only run outside of Ray Tune to avoid conflicts
        if self.ray_tune_callback is None:
            best_f1 = -1
            best_step_dir = None
            best_epoch = None
            for step_dir in sorted(output_dir.glob("train_step*/")):
                results_csv = step_dir / "results.csv"
                weights_dir = step_dir / "weights"
                if results_csv.exists() and weights_dir.exists():
                    df = pd.read_csv(results_csv)
                    for _, row in df.iterrows():
                        precision = row.get("metrics/precision(B)", 0.0)
                        recall = row.get("metrics/recall(B)", 0.0)
                        if precision + recall > 0:
                            f1 = self._calculate_f1(precision, recall)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_step_dir = step_dir
                                best_epoch = int(row["epoch"])
            final_weights = None
            if best_step_dir is not None and best_epoch is not None:
                candidate = best_step_dir / "weights" / f"epoch{best_epoch - 1}.pt"
                if candidate.exists():
                    final_weights = candidate
            best_model_path = str(final_weights) if final_weights else None
            print(f"Best model training path: {best_model_path} (F1={best_f1:.4f})")
            final_weights_path = Path(output_dir) / "best_model_weights.pt"
            if final_weights:
                shutil.copy2(final_weights, final_weights_path)
            self.final_weights_path = final_weights_path
            self.best_model_path = best_model_path
            print(f"Final weights saved to: {final_weights_path}")

            self.test_preds = evaluate_test_set(
                model_path=final_weights_path if final_weights else best_model_path,
                training_dir=Path(yolo_data_dir),
                output_dir=output_dir,
                plot_mode=plot_mode,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            if self.test_preds:
                images_dir = Path(yolo_data_dir) / "images"
                labels_dir = Path(yolo_data_dir) / "labels"
                plot_dir = Path(output_dir) / "prediction_plots"

                if plot_mode == 'none':
                    predictions_to_plot = []
                elif plot_mode == 'subset':
                    sample_size = min(15, len(self.test_preds))
                    predictions_to_plot = random.sample(self.test_preds, sample_size)
                else: 
                    predictions_to_plot = self.test_preds

                if predictions_to_plot:
                    plot_ground_truth_vs_predictions(
                        predictions=predictions_to_plot,
                        labels_dir=labels_dir,
                        original_images_dir=images_dir,
                        save_dir=plot_dir,
                        conf_threshold=conf_threshold
                    )

    def _calculate_f1(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _reset_loggers(self):
        self.train_metric_logger = MetricLogger(delimiter="  ")
        self.train_metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.validation_metrics = {}
        self.end_time = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')

    def _set_trainable_layers(self, num_layers):
        for param in self.model.model.parameters():
            param.requires_grad = False
        if num_layers >= 3:
            for param in self.model.model.parameters():
                param.requires_grad = True
        elif num_layers >= 2:
            for param in self.model.model.model[-1].parameters():
                param.requires_grad = True
            for i in range(-4, 0):
                for param in self.model.model.model[i].parameters():
                    param.requires_grad = True
        elif num_layers >= 1:
            for param in self.model.model.model[-1].parameters():
                param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

    def _on_batch_end(self, trainer, print_freq):
        self.batch_idx += 1
        self.current_epoch = trainer.epoch + 1
        batch_time = time.time() - self.end_time
        self.iter_time.update(batch_time)
        self.end_time = time.time()
        box_loss = float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0.0
        cls_loss = float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0.0
        dfl_loss = float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0.0
        total_loss = box_loss + cls_loss + dfl_loss
        for name, value in [('box_loss', box_loss), ('cls_loss', cls_loss), ('dfl_loss', dfl_loss), ('total_loss', total_loss)]:
            if math.isnan(value) or math.isinf(value):
                raise RuntimeError(f"NaN/inf detected in training metric: {name}={value}")
        self.train_metric_logger.update(
            loss=total_loss,
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss
        )
        self.train_metric_logger.update(lr=trainer.optimizer.param_groups[0]["lr"])
        if self.batch_idx == 1 or self.batch_idx % print_freq == 0 or self.batch_idx == self.num_batches:
            total_batches = getattr(trainer, 'nb', self.num_batches)
            remaining_batches = max(0, total_batches - self.batch_idx)
            eta_seconds = self.iter_time.global_avg * remaining_batches
            eta_string = str(timedelta(seconds=int(eta_seconds)))
            gpu_mem = ''
            if torch.cuda.is_available():
                MB = 1024.0 * 1024.0
                gpu_mem = f"max mem: {torch.cuda.max_memory_allocated() / MB:.0f}M"
            header = f'Epoch: [{self.current_epoch}/{self.epochs}] Training'
            progress = f'[{self.batch_idx}/{total_batches}]'
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = self._original_stdout, self._original_stderr
            print(f"{header} {progress} eta: {eta_string} {self.train_metric_logger} time: {self.iter_time} {gpu_mem}")
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _on_epoch_end(self, trainer):
        self.batch_idx = 0
        self.current_epoch = trainer.epoch + 1

    def _on_val_end(self, validator):
        try:
            val_metrics = validator.metrics
            if hasattr(self, '_last_val_epoch') and self._last_val_epoch == self.current_epoch:
                return
            self._last_val_epoch = self.current_epoch
            map50 = float(val_metrics.box.map50) if hasattr(val_metrics.box, 'map50') else 0.0
            precision = float(val_metrics.box.p[0]) if hasattr(val_metrics.box, 'p') and len(val_metrics.box.p) > 0 else 0.0
            recall = float(val_metrics.box.r[0]) if hasattr(val_metrics.box, 'r') and len(val_metrics.box.r) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            fitness = getattr(val_metrics, 'fitness', 0.0)
            if hasattr(validator, 'loss') and validator.loss is not None and hasattr(validator, 'dataloader'):
                num_batches = len(validator.dataloader)
                val_box_loss = float(validator.loss[0]) / num_batches
                val_cls_loss = float(validator.loss[1]) / num_batches
                val_dfl_loss = float(validator.loss[2]) / num_batches
            else:
                val_box_loss = 0.0
                val_cls_loss = 0.0
                val_dfl_loss = 0.0
            val_total_loss = val_box_loss + val_cls_loss + val_dfl_loss
            self.validation_metrics = {
                'val_mAP50': map50,
                'val_fitness': fitness,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_loss': val_total_loss,
                'val_box_loss': val_box_loss,
                'val_cls_loss': val_cls_loss,
                'val_dfl_loss': val_dfl_loss
            }
            for key, value in self.validation_metrics.items():
                if math.isnan(value) or math.isinf(value):
                    raise RuntimeError(f"NaN/inf detected in validation metric: {key}={value}")
            train_metrics = {
                'lr': self.train_metric_logger.meters['lr'].value,
                'train_loss': self.train_metric_logger.meters['loss'].global_avg,
                'train_box_loss': self.train_metric_logger.meters['box_loss'].global_avg,
                'train_cls_loss': self.train_metric_logger.meters['cls_loss'].global_avg,
                'train_dfl_loss': self.train_metric_logger.meters['dfl_loss'].global_avg,
            }
            actual_epoch = self.current_epoch + getattr(self, 'total_epochs_so_far', 0)
            epoch_metrics = {
                **train_metrics,
                **self.validation_metrics,
                'epoch': self.current_epoch,
                'step': self.current_step,
                'training_iteration': actual_epoch,
            }
            self.metrics_history.append(epoch_metrics)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = self._original_stdout, self._original_stderr
            header = f'Epoch: [{self.current_epoch}/{self.epochs}] Validation'
            val_str = f"mAP50: {map50:.4f}  fitness: {fitness:.4f}  precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}  loss: {val_total_loss:.4f}  box loss: {val_box_loss:.4f}  cls loss: {val_cls_loss:.4f}  dfl loss: {val_dfl_loss:.4f}"
            print(f"{header}  {val_str}")
            print()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if self.ray_tune_callback:
                self.ray_tune_callback(epoch_metrics)
        except Exception as e:
            with open("validation_errors.log", "a") as f:
                f.write(f"Error logging validation metrics: {str(e)}\n")
            print(f"Error in _on_val_end: {e}")