import time
from collections import defaultdict, deque
import datetime
import errno
import os
import time
import yaml
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from ultralytics import YOLO

import torch
import torch.distributed as dist
from torchvision.ops import nms
from typing import List, Dict, Optional

from ray import tune

## Adapted from PyTorch  detection utils: https://github.com/pytorch/vision/blob/main/references/detection/utils.py
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def apply_nms(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        detections: List of detection dicts with 'box', 'confidence', 'label'
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    boxes = np.array([d['box'] for d in detections], dtype=np.float32)
    scores = np.array([d['confidence'] for d in detections], dtype=np.float32)
    
    boxes_tensor = torch.from_numpy(boxes)
    scores_tensor = torch.from_numpy(scores)
    
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    
    return [detections[i] for i in keep_indices.tolist()]


## Other
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_output_dir(base: str, run_type: str, timestamp: Optional[str] = None) -> Path:
    """Return a standardized output directory: base/{run_type}_{timestamp}"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"{run_type}_{timestamp}"


def convert_tuning_space(space):
    """Convert config's dict-based tuning space to Ray Tune format"""
    tuning_space = {}
    for k, v in space.items():
        if isinstance(v, list):
            tuning_space[k] = tune.choice(v)
        elif isinstance(v, dict):
            if 'uniform' in v:
                tuning_space[k] = tune.uniform(*v['uniform'])
            elif 'loguniform' in v:
                tuning_space[k] = tune.loguniform(*v['loguniform'])
            else:
                raise ValueError(f"Unknown distribution for {k}: {v}")
        else:
            raise ValueError(f"Unknown type for {k}: {v}")
    return tuning_space


def compute_composite_objective(val_loss, f1, map50, score_weights):
    """Composite tuning/selection objective (lower is better): a weighted blend
    of validation loss and gain-invariant quality metrics.

    Returns:
        score_weights['loss']*val_loss
        + score_weights['f1']*(1-f1)*10
        + score_weights['map50']*(1-map50)*10

    The *10 puts the unit-interval quality terms on a scale comparable to a
    typical YOLO val_loss. Optimizing this instead of raw val_loss avoids the
    confound where tuning box/cls/dfl gains directly rescales val_loss across
    trials. Degenerate trials get a large
    sentinel so the scheduler prunes them.
    """
    val_loss = float(val_loss) if np.isfinite(val_loss) else float("inf")
    f1 = float(f1) if np.isfinite(f1) else 0.0
    map50 = float(map50) if np.isfinite(map50) else 0.0
    if val_loss <= 0:
        return 1e6
    if f1 < 0.01 and map50 < 0.01:
        return 1e6
    return float(
        score_weights["loss"] * val_loss
        + score_weights["f1"] * (1.0 - f1) * 10.0
        + score_weights["map50"] * (1.0 - map50) * 10.0
    )


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_resilient_write_bytes(retries: int = 8, base_delay: float = 0.25):
    """Make ``pathlib.Path.write_bytes`` retry on transient Windows PermissionError.

    On Windows, an on-access AV/indexer scan can briefly hold an open handle to a
    file right after it is written; the next overwrite of that same file then fails
    with ``PermissionError: [Errno 13]``. Ultralytics rewrites last.pt/best.pt every
    epoch, so a single unlucky scan kills the whole trial. Retrying the write a few
    times with a short backoff rides out the scan instead. Idempotent and process-wide.
    """
    if getattr(Path.write_bytes, "_burr_retry_wrapped", False):
        return
    _orig_write_bytes = Path.write_bytes

    def write_bytes(self, data):
        for attempt in range(retries):
            try:
                return _orig_write_bytes(self, data)
            except PermissionError:
                if attempt == retries - 1:
                    raise
                time.sleep(base_delay * (attempt + 1))

    write_bytes._burr_retry_wrapped = True
    Path.write_bytes = write_bytes


def is_notebook():
    """Check if running in a Jupyter notebook"""
    try:
        from IPython.core.getipython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except Exception:
        return False


def evaluate_test_set(
    model_path: Path,
    training_dir: Path,
    output_dir: Path,
    plot_mode: str = 'none',
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45
):
    """Evaluate model on test set, save results, and optionally return predictions for plotting."""

    print("\n" + "="*80)
    print("Evaluating on the test set...")
    print("="*80)

    dataset_yaml = Path(training_dir) / "dataset.yml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yml not found in {training_dir}. Please run prepare_dataset_splits first.")

    model = YOLO(model_path)
    test_results = model.val(data=str(dataset_yaml), split='test', verbose=False)

    model_name = Path(model_path).stem.replace("best_", "")

    precision = test_results.results_dict['metrics/precision(B)']
    recall = test_results.results_dict['metrics/recall(B)']
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    test_metrics = {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1, 
        'mAP50': test_results.results_dict['metrics/mAP50(B)'],
        'fitness': test_results.results_dict['fitness'],
        'inference_time_ms': test_results.speed['inference'],
        'model_path': str(model_path)
    }

    print("Test Performance:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([test_metrics]).to_csv(
        Path(output_dir) / "test_results.csv", 
        index=False
    )

    if plot_mode == 'none':
        return None

    with open(dataset_yaml, 'r') as f:
        ds = yaml.safe_load(f)
    test_txt = Path(ds['test'])
    if not test_txt.is_absolute():
        test_txt = (dataset_yaml.parent / test_txt).resolve()
    with open(test_txt, 'r') as f:
        test_image_paths = [line.strip() for line in f if line.strip()]

    predictions = []
    results = model.predict(
        test_image_paths,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    for img_path, pred in zip(test_image_paths, results):
        detections = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            labels = pred.boxes.cls.cpu().numpy() if hasattr(pred.boxes, 'cls') else np.zeros(len(boxes))
            for box, conf, label in zip(boxes, confs, labels):
                detections.append({
                    'box': [float(x) for x in box],
                    'confidence': float(conf),
                    'label': int(label)
                })
        predictions.append((img_path, detections))

    return predictions


def plot_ground_truth_vs_predictions(predictions, labels_dir, original_images_dir, save_dir=None, conf_threshold=0.5):
    """
    Plot ground truth vs predictions side by side for each test image using original resolution images
    
    Args:
        predictions: List of tuples (image_path, detections) where detections is list of dicts
        labels_dir: Path to directory containing YOLO label files (None for unlabeled inference)
        original_images_dir: Path to directory containing original resolution images
        save_dir: Directory to save plots (optional)
        conf_threshold: Minimum confidence threshold to display boxes
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    if labels_dir:
        labels_dir = Path(labels_dir)
    original_images_dir = Path(original_images_dir)
    
    for img_path, detections in predictions:
        img_path = Path(img_path)
        img = np.array(ImageOps.exif_transpose(Image.open(img_path)).convert('RGB'))
        img_height, img_width = img.shape[:2]

        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
        else:
            avg_conf = 0.0

        if labels_dir: # tuning or training mode - show ground truth vs predictions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title('Ground Truth', fontsize=16, weight='bold')

            label_file = labels_dir / f"{img_path.stem}.txt"
            if label_file.exists():
                try:
                    gt_data = np.loadtxt(label_file, ndmin=2)
                    if gt_data.ndim == 1:
                        gt_data = gt_data[None, :]
                    if gt_data.shape[0] > 0:
                        for row in gt_data:
                            _, cx_norm, cy_norm, w_norm, h_norm = row
                            cx = cx_norm * img_width
                            cy = cy_norm * img_height
                            w = w_norm * img_width
                            h = h_norm * img_height
                            x1 = cx - w/2
                            y1 = cy - h/2
                            rect = patches.Rectangle(
                                (x1, y1), w, h,
                                linewidth=0.75,
                                edgecolor='red', 
                                facecolor='none'
                            )
                            ax1.add_patch(rect)
                    else:
                        print(f"Label file {label_file} loaded but no rows found.")
                except Exception as e:
                    print(f"Error loading label file {label_file}: {e}")

            ax2.imshow(img)
            ax2.axis('off')
            ax2.set_title(f'Predictions (Avg Conf: {avg_conf:.3f})', fontsize=16, weight='bold')

            for det in detections:
                if det['confidence'] >= conf_threshold:
                    x1, y1, x2, y2 = det['box']
                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=0.75,
                        edgecolor='red', 
                        facecolor='none'
                    )
                    ax2.add_patch(rect)
            fig.suptitle(f'{img_path.name}', fontsize=18, weight='bold')
            
        else:  # Inference mode - show predictions only
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Detections: {len(detections)} burrs | Avg Confidence: {avg_conf:.3f}', 
                        fontsize=16, weight='bold')
            
            for det in detections:
                if det['confidence'] >= conf_threshold:
                    x1, y1, x2, y2 = det['box']
                    width = x2 - x1
                    height = y2 - y1
                    
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=0.75, 
                        edgecolor='red', 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
            
            fig.suptitle(f'{img_path.stem}', fontsize=18, weight='bold')
        
        # Save or show
        if save_dir:
            save_path = save_dir / f'{img_path.stem}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)


def compute_gt_iou_statistics(labels_dir, tile_size: int = 224, glob_pattern: str = "*.txt") -> Dict:
    """Compute pairwise IoU between GT boxes within each label tile.

    Helps pick an NMS IoU just above the maximum GT-GT overlap so two genuinely
    distinct burrs are never suppressed as duplicates. Prints percentiles plus a
    suggested threshold and returns a stats dict. Operates on YOLO bbox labels
    (cls cx cy w h, normalized); run it on the training labels as a one-off.
    """
    labels_dir = Path(labels_dir)
    label_files = sorted(labels_dir.glob(glob_pattern))
    all_ious = []
    tiles_with_multi = 0

    for lf in label_files:
        boxes = []
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            px_cx, px_cy, px_w, px_h = cx * tile_size, cy * tile_size, w * tile_size, h * tile_size
            boxes.append((px_cx - px_w / 2, px_cy - px_h / 2, px_cx + px_w / 2, px_cy + px_h / 2))
        if len(boxes) < 2:
            continue
        tiles_with_multi += 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax1, ay1, ax2, ay2 = boxes[i]
                bx1, by1, bx2, by2 = boxes[j]
                inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
                union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > 0:
                    all_ious.append(iou)

    if not all_ious:
        print("No overlapping GT box pairs found; any NMS IoU works. Suggesting 0.5.")
        return {"count": 0, "max": 0.0, "suggested_nms_iou": 0.5}

    arr = np.array(all_ious)
    stats = {
        "count": int(arr.size),
        "tiles_with_multi_obj": tiles_with_multi,
        "max": float(arr.max()),
        "p99": float(np.percentile(arr, 99)),
        "p95": float(np.percentile(arr, 95)),
        "p90": float(np.percentile(arr, 90)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
    }
    stats["suggested_nms_iou"] = float(min(0.7, stats["max"] + 0.05))
    print(f"Pairwise GT IoU over {len(label_files)} tiles ({tiles_with_multi} with >=2 boxes):")
    for k in ("count", "max", "p99", "p95", "p90", "median", "mean"):
        v = stats[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Suggested NMS IoU (max + 0.05, capped at 0.7): {stats['suggested_nms_iou']:.3f}")
    return stats


def export_predictions_as_yolo(model_path, image_paths, output_dir,
                               conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> int:
    """Run a detector over images and save predictions as YOLO txt + confidence.

    Each line: 'class conf cx cy w h' (normalized). Useful for seeding an
    annotation-review / active-learning loop. Detection-only.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    image_paths = list(image_paths)
    results = model.predict(image_paths, conf=conf_threshold, iou=iou_threshold, verbose=False)

    n_written = 0
    for img_path, pred in zip(image_paths, results):
        lines = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            xywhn = pred.boxes.xywhn.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy().astype(int) if hasattr(pred.boxes, 'cls') else np.zeros(len(xywhn), int)
            for (cx, cy, w, h), c, cl in zip(xywhn, confs, clss):
                lines.append(f"{int(cl)} {float(c):.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (output_dir / f"{Path(img_path).stem}.txt").write_text("\n".join(lines))
        n_written += 1

    print(f"Wrote {n_written} prediction files to {output_dir}")
    return n_written


def analyze_ray_results(experiment_dir, output_dir=None, top_n: int = 10):
    """Summarize Ray Tune trial outputs: top trials, HP importance, objective curves.

    Reads each ``trial_*/progress.csv`` + ``params.json`` under ``experiment_dir``,
    ranks by the per-trial best ``objective``, prints the top-N, computes Spearman
    HP importance vs objective, and saves a summary CSV + plots. Best-effort and
    guarded so it never breaks a tuning run.
    """
    import json as _json

    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir) if output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, epoch_dfs = [], {}
    for td in sorted(experiment_dir.glob("trial_*")):
        pf, gf = td / "params.json", td / "progress.csv"
        if not pf.exists() or not gf.exists():
            continue
        try:
            params = _json.loads(pf.read_text())
            prog = pd.read_csv(gf)
        except Exception:
            continue
        if prog.empty or "objective" not in prog.columns:
            continue
        epoch_dfs[td.name] = prog
        best = prog.loc[prog["objective"].idxmin()]
        rec = {"trial": td.name, "objective": float(best["objective"])}
        for col in ("val_loss", "val_f1", "val_mAP50", "val_precision", "val_recall"):
            if col in prog.columns:
                rec[col] = float(best[col])
        for k, v in params.items():
            rec[f"hp_{k}"] = v
        records.append(rec)

    if not records:
        print(f"No analyzable trials in {experiment_dir}")
        return None

    df = pd.DataFrame(records).sort_values("objective").reset_index(drop=True)
    df.to_csv(output_dir / "trial_summary.csv", index=False)

    show = [c for c in ("trial", "objective", "val_loss", "val_f1", "val_mAP50") if c in df.columns]
    print(f"\nTop {min(top_n, len(df))} trials by objective:")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(df[show].head(top_n).to_string(index=False))

    hp_cols = [c for c in df.columns if c.startswith("hp_") and pd.api.types.is_numeric_dtype(df[c])]
    corrs = {}
    for hp in hp_cols:
        sub = df[[hp, "objective"]].dropna()
        if len(sub) >= 5:
            rho = sub[hp].corr(sub["objective"], method="spearman")
            if pd.notna(rho):
                corrs[hp.replace("hp_", "")] = float(rho)

    try:
        if corrs:
            cser = pd.Series(corrs)
            cser = cser.reindex(cser.abs().sort_values().index)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cser))))
            ax.barh(cser.index, cser.values, color=["#2ecc71" if v < 0 else "#e74c3c" for v in cser.values])
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("Spearman corr with objective (negative = lowers objective)")
            ax.set_title("Hyperparameter importance")
            fig.tight_layout()
            fig.savefig(output_dir / "hp_importance.png", dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for t in df["trial"].head(5):
            ep = epoch_dfs.get(t)
            if ep is not None and "objective" in ep.columns:
                ax.plot(range(1, len(ep) + 1), ep["objective"].values, label=t, alpha=0.8)
        ax.set_xlabel("Report")
        ax.set_ylabel("objective")
        ax.set_title("Top-5 trial objective curves")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "top_trial_curves.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Analysis plotting skipped: {e}")

    print(f"Tuning analysis written to {output_dir}")
    return df


def plot_dataset_samples(split_txt, labels_dir, save_dir, num_samples: int = 8,
                         seed: int = 666, class_names=None):
    """Save a few sample tiles with their YOLO bbox labels overlaid.

    Use this to eyeball, before training, that images and labels line up and that
    the data YOLO will ingest looks right. Reads image paths from a split .txt
    (one path per line), loads each EXIF-aware, draws its normalized bbox labels,
    and writes a PNG per sample. Returns the list of plotted image stems.
    """
    split_txt = Path(split_txt)
    labels_dir = Path(labels_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(split_txt) as f:
        image_paths = [line.strip() for line in f if line.strip()]
    if not image_paths:
        print(f"No images listed in {split_txt}; nothing to plot.")
        return []

    name = class_names.get(0, 'label') if isinstance(class_names, dict) else 'label'
    rng = random.Random(seed)
    picks = rng.sample(image_paths, min(num_samples, len(image_paths)))

    plotted = []
    for img_path in picks:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"  (skip, missing image) {img_path}")
            continue
        img = np.array(ImageOps.exif_transpose(Image.open(img_path)).convert('RGB'))
        h, w = img.shape[:2]

        label_file = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append((cx * w, cy * h, bw * w, bh * h))

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        for cx, cy, bw, bh in boxes:
            ax.add_patch(patches.Rectangle((cx - bw / 2, cy - bh / 2), bw, bh,
                                           linewidth=1.0, edgecolor='red', facecolor='none'))
        ax.set_title(f"{img_path.stem}\n{len(boxes)} {name}  |  {w}x{h}", fontsize=9)
        fig.tight_layout()
        fig.savefig(save_dir / f"{img_path.stem}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        plotted.append(img_path.stem)

    print(f"Saved {len(plotted)} QA sample plots to {save_dir}")
    return plotted


def compute_fn_audit(images_dir, labels_dir, model_path, output_dir,
                     conf_threshold: float = 0.4, iou_threshold: float = 0.5,
                     top_k_plots: int = 24, batch_size: int = 64):
    """Report (do NOT apply) where the existing model and the labels disagree.

    Runs `model_path` over the tiles and compares its burr count to the labeled
    count per tile and per tree. Tiles where the model finds many MORE burrs than
    labeled are candidate false-negative regions; far FEWER may be over-labeling.
    Writes ranked CSVs + label-vs-prediction overlay plots for the largest
    disagreements. Advisory only: the model was trained on the under-labeled data,
    so its predictions are a hint for human triage, not ground truth — nothing is
    relabeled here.
    """
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    rows, preds = [], {}
    for start in range(0, len(img_files), max(1, batch_size)):
        batch = img_files[start:start + batch_size]
        results = model.predict([str(f) for f in batch], conf=conf_threshold,
                                iou=iou_threshold, verbose=False)
        for f, r in zip(batch, results):
            n_pred = int(len(r.boxes)) if r.boxes is not None else 0
            lf = labels_dir / f"{f.stem}.txt"
            n_lab = sum(1 for ln in lf.read_text().splitlines() if ln.strip()) if lf.exists() else 0
            rows.append({"tile": f.stem, "tree": f.stem.rsplit("_", 2)[0],
                         "labeled": n_lab, "predicted": n_pred, "diff": n_pred - n_lab})
            preds[f.stem] = r

    df = pd.DataFrame(rows)
    if df.empty:
        print("FN audit: no tiles found.")
        return df
    df["abs_diff"] = df["diff"].abs()
    df.sort_values("abs_diff", ascending=False).to_csv(output_dir / "fn_audit_tiles.csv", index=False)

    tree = df.groupby("tree").agg(labeled=("labeled", "sum"), predicted=("predicted", "sum")).reset_index()
    tree["diff"] = tree["predicted"] - tree["labeled"]
    tree.sort_values("diff", ascending=False).to_csv(output_dir / "fn_audit_trees.csv", index=False)

    plot_dir = output_dir / "fn_audit_plots"
    plot_dir.mkdir(exist_ok=True)
    for _, row in df.sort_values("diff", ascending=False).head(top_k_plots).iterrows():
        stem = row["tile"]
        r = preds.get(stem)
        if r is None:
            continue
        img = np.array(ImageOps.exif_transpose(Image.open(images_dir / f"{stem}.jpg")).convert("RGB"))
        h, w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.axis("off")
        lf = labels_dir / f"{stem}.txt"
        if lf.exists():
            for ln in lf.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                ax.add_patch(patches.Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h,
                                               lw=1.0, edgecolor="red", facecolor="none"))
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes.xyxyn.cpu().numpy():
                ax.add_patch(patches.Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h,
                                               lw=1.0, edgecolor="yellow", facecolor="none"))
        ax.set_title(f"{stem}\nlabeled={int(row['labeled'])} (red)  pred={int(row['predicted'])} (yellow)", fontsize=8)
        fig.savefig(plot_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    likely_fn = int((df["diff"] > 2).sum())
    print(f"FN audit (advisory, not applied): {len(df)} tiles | labeled {int(df['labeled'].sum())} "
          f"vs predicted {int(df['predicted'].sum())} | {likely_fn} tiles where model finds >2 more "
          f"than labeled | report -> {output_dir}")
    return df


def augment_labels_with_model(tiled_dir, model_path, split: str = "train", conf: float = 0.4,
                              containment_thresh: float = 0.6, batch_size: int = 64,
                              viz_dir=None, viz_n: int = 12, seed: int = 666):
    """Add high-confidence model predictions as new burr labels (recover false
    negatives) for the given split — additions only, never removes labels.

    For each tile in `split`, a prediction with confidence >= conf is appended only
    if it is not the same burr as a box already kept (an original label OR an earlier
    addition). "Same burr" is tested by CONTAINMENT — the fraction of the smaller box
    inside the other >= containment_thresh — so a tight model box fully inside a loose
    human box is treated as a duplicate (plain IoU misses this). The source polygon
    labels are left intact and the tiled set is regenerable from `--mode preprocess`,
    so this is reversible. If viz_dir is given, saves overlays of random augmented
    tiles (red = original, lime = added). Returns (tiles_modified, boxes_added).
    """
    tiled_dir = Path(tiled_dir)
    labels_dir, images_dir = tiled_dir / "labels", tiled_dir / "images"
    stems = [Path(l.strip()).stem for l in (tiled_dir / f"{split}.txt").read_text().splitlines() if l.strip()]
    paths = [images_dir / f"{s}.jpg" for s in stems if (images_dir / f"{s}.jpg").exists()]
    model = YOLO(model_path)

    def xyxy(cx, cy, w, h):
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def contained(a, b):
        """Fraction of the SMALLER box that lies inside the other. Catches a tight
        box fully inside a loose box (same burr), which plain IoU misses."""
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return inter / smaller if smaller > 0 else 0.0

    modified = added = 0
    changes = {}  # stem -> (original_boxes, added_boxes) for verification
    for start in range(0, len(paths), max(1, batch_size)):
        batch = paths[start:start + batch_size]
        results = model.predict([str(p) for p in batch], conf=conf, verbose=False)
        for p, r in zip(batch, results):
            lf = labels_dir / f"{p.stem}.txt"
            orig = []
            if lf.exists():
                for ln in lf.read_text().splitlines():
                    v = ln.split()
                    if len(v) >= 5:
                        orig.append(xyxy(*(float(x) for x in v[1:5])))
            # Add a prediction only if it isn't the same burr as a box already kept
            # (an original label OR an earlier addition), tested by containment.
            kept = list(orig)
            new_boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                for cx, cy, w, h in r.boxes.xywhn.cpu().numpy():
                    nb = xyxy(float(cx), float(cy), float(w), float(h))
                    if all(contained(nb, k) < containment_thresh for k in kept):
                        kept.append(nb)
                        new_boxes.append(nb)
            if not new_boxes:
                continue
            lf.write_text("\n".join(
                f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                for (x1, y1, x2, y2) in kept) + "\n")
            changes[p.stem] = (orig, new_boxes)
            added += len(new_boxes)
            modified += 1

    if viz_dir and changes:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        picks = random.Random(seed).sample(list(changes), min(viz_n, len(changes)))
        for stem in picks:
            existing, new_boxes = changes[stem]
            img = np.array(ImageOps.exif_transpose(Image.open(images_dir / f"{stem}.jpg")).convert("RGB"))
            h, w = img.shape[:2]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img)
            ax.axis("off")
            for (x1, y1, x2, y2) in existing:
                ax.add_patch(patches.Rectangle((x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
                                               lw=1.0, edgecolor="red", facecolor="none"))
            for (x1, y1, x2, y2) in new_boxes:
                ax.add_patch(patches.Rectangle((x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
                                               lw=1.2, edgecolor="lime", facecolor="none"))
            ax.set_title(f"{stem}\noriginal={len(existing)} (red)  added={len(new_boxes)} (lime)", fontsize=8)
            fig.savefig(viz_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"Augmented {split} labels (conf>={conf}): added {added} boxes across {modified} tiles"
          f"{'; overlays -> ' + str(viz_dir) if viz_dir else ''}")
    return modified, added