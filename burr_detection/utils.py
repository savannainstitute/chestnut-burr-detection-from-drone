from collections import defaultdict, deque
import datetime
import errno
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from typing import List, Dict


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
    
    from torchvision.ops import nms
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    
    return [detections[i] for i in keep_indices.tolist()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_test_results(test_results):
    """Format YOLO test evaluation results in a clean, readable way"""

    metrics = test_results.results_dict

    print("\nYOLO Model Evaluation on Test Set")
    print("=" * 50)
    print(f"Class Information: {test_results.names}")
    print(f"Test Objects: {test_results.nt_per_class.sum()} instances ({test_results.nt_per_class[0]} {list(test_results.names.values())[0]})")
    
    print("\nPerformance Metrics:")
    print(f"  • Precision:     {metrics['metrics/precision(B)']:.4f}")
    print(f"  • Recall:        {metrics['metrics/recall(B)']:.4f}")
    print(f"  • mAP@0.5:       {metrics['metrics/mAP50(B)']:.4f}")
    print(f"  • mAP@0.5-0.95:  {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"  • Overall Fitness: {metrics['fitness']:.4f}")
    
    print("\nSpeed Performance:")
    print(f"  • Preprocessing: {test_results.speed['preprocess']:.2f}ms")
    print(f"  • Inference:     {test_results.speed['inference']:.2f}ms")
    print(f"  • Postprocessing: {test_results.speed['postprocess']:.2f}ms")
    print(f"  • Total per image: {test_results.speed['preprocess'] + test_results.speed['inference'] + test_results.speed['postprocess']:.2f}ms")
    print("=" * 50)


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
        img = np.array(Image.open(img_path).convert('RGB'))
        img_height, img_width = img.shape[:2]

        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
        else:
            avg_conf = 0.0
        
        # plot predictions
        if labels_dir: # tuning or training mode - show ground truth vs predictions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title('Ground Truth', fontsize=16, weight='bold')

            label_file = labels_dir / f"{img_path.stem}.txt"
            if label_file.exists():
                try:
                    gt_data = np.loadtxt(label_file, ndmin=2)
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
                                linewidth=0.5,
                                edgecolor='red', 
                                facecolor='none'
                            )
                            ax1.add_patch(rect)
                except:
                    pass

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
                        linewidth=0.5,
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
                        linewidth=0.5, 
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