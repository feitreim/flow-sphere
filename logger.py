from typing import Any
import re
import time
import wandb
import torch
from tqdm import tqdm
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance


class MetricTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('metric', torch.zeros(1))
        self.iters = 0

    @torch.no_grad
    def reset(self):
        torch.zero_(self.metric)  # pyright: ignore
        self.iters = 0

    @torch.no_grad
    def average(self) -> float:
        avg = float((self.metric).detach()) / self.iters  # pyright: ignore
        self.reset()
        return avg

    @torch.no_grad
    def update(self, new: torch.Tensor):
        self.metric.add_(new)  # pyright: ignore
        self.iters += 1


def sanitize_artifact_name(name: str) -> str:
    """Remove characters not allowed in wandb artifact names.

    Artifact names may only contain alphanumeric characters, dashes,
    underscores, and dots.
    """
    return re.sub(r'[^a-zA-Z0-9._-]', '', name)


class Logger:
    """Dummy logger that prints all logged data to console."""

    def __init__(self, project, name, config=None, device: str | torch.device = 'cpu'):
        self.project = project
        self.name = name
        self.device = device
        self.bsz = config['data']['batch_size'] if config else 1
        self.train_state = {}
        self.val_state = {}
        self.iters = 0
        self.global_step = 0
        self.t_start = time.monotonic()
        print(f'[Logger] Initialized: project={project}, name={name}')

    def iter(self):
        self.iters += 1

    def log(self, prefix: str, data: dict[str, Any]):
        print(f'[{prefix}]')
        max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
        for key, value in data.items():
            if isinstance(value, float):
                print(f'  {key:<{max_key_len}} : {value:.6f}')
            else:
                print(f'  {key:<{max_key_len}} : {value}')

    def train_log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.train_state:
                self.train_state[k] = MetricTracker().to(self.device)
            self.train_state[k].update(v)

    def train_step(self):
        self.global_step += self.bsz * self.iters
        t_end = time.monotonic()
        throughput = (self.bsz * self.iters) / (t_end - self.t_start)
        to_log = {f'train/{k}': v.average() for k, v in self.train_state.items()}
        to_log['train/training_samples'] = self.global_step
        to_log['train/throughput'] = throughput

        # Add GPU memory utilization if using GPU
        if torch.cuda.is_available() and isinstance(self.device, (str, torch.device)):
            device_idx = 0
            if isinstance(self.device, torch.device):
                device_idx = self.device.index if self.device.index is not None else 0
            elif isinstance(self.device, str) and self.device.startswith('cuda'):
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0

            mem_allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved(device_idx) / 1024**3  # GB
            to_log['train/gpu_memory_allocated_gb'] = mem_allocated
            to_log['train/gpu_memory_reserved_gb'] = mem_reserved

        self.log('train', {k.removeprefix('train/'): v for k, v in to_log.items()})
        self.iters = 0
        self.t_start = t_end

    def val_log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.val_state:
                self.val_state[k] = (
                    MetricTracker().to(self.device) if self.device else MetricTracker()
                )
            self.val_state[k].update(v)

    def val_step(self):
        to_log = {f'val/{k}': v.average() for k, v in self.val_state.items()}
        self.log('val', {k.removeprefix('val/'): v for k, v in to_log.items()})

    def log_image(self, prefix, grid, caption):
        print(f'[{prefix}/image] grid={tuple(grid.shape)}, caption="{caption}"')

    def save_model(self, model, args):
        print(f'[save_model] Would save model "{args.name}" (saving disabled in dummy logger)')

    def setup_fid(self, dataloader, cache_path=None):
        pass

    def update_fid(self, fake_images):
        pass

    def compute_fid(self) -> float | None:
        return None

    def log_args(self, args):
        for name, arg in args.items():
            print(f'[config/{name}]')
            for k, v in arg.items():
                print(f'  {k}: {v}')


class WandbLogger(Logger):
    def __init__(self, project, name, config, device):
        self.project = project
        self.name = name
        if wandb.run is None:
            wandb.init(project=project, name=name, config=config)
        self.device = device
        self.bsz = config['data']['batch_size']
        self.train_state = {}
        self.val_state = {}
        self.iters = 0
        self.global_step = 0
        self.t_start = time.monotonic()
        wandb.define_metric('train/*', 'train/training_samples')
        wandb.define_metric('val/*', 'train/training_samples')

    def iter(self):
        self.iters += 1

    def log(self, prefix: str, data: dict[str, Any]):
        """
        usage of this would be:
        logger.log('train', data)
        """
        wandb.log({f'{prefix}/{key}': data[key] for key in data.keys()})

    def train_log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.train_state:
                self.train_state[k] = MetricTracker().to(self.device)
            self.train_state[k].update(v)

    def train_step(self):
        self.global_step += self.bsz * self.iters
        t_end = time.monotonic()
        throughput = (self.bsz * self.iters) / (t_end - self.t_start)
        to_log = {f'train/{k}': v.average() for k, v in self.train_state.items()}
        to_log.update({'train/training_samples': self.global_step})
        to_log.update({'train/throughput': throughput})
        wandb.log(to_log)
        self.iters = 0
        self.t_start = t_end

    def val_log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.val_state:
                self.val_state[k] = MetricTracker().to(self.device)
            self.val_state[k].update(v)

    def val_step(self):
        to_log = {f'val/{k}': v.average() for k, v in self.val_state.items()}
        wandb.log(to_log)

    def log_image(self, prefix, grid, caption):
        wandb.log({f'{prefix}/image': wandb.Image(grid.float(), caption=caption)})

    def setup_fid(self, dataloader, cache_path=None):
        """Preload real image statistics for FID. Caches to disk if cache_path is given."""
        self.fid = FrechetInceptionDistance(
            feature=2048, normalize=True, reset_real_features=False
        ).to(self.device)
        cache = Path(cache_path) if cache_path else None
        if cache and cache.exists():
            stats = torch.load(cache, map_location=self.device, weights_only=True)
            self.fid.real_features_sum = stats['real_features_sum']
            self.fid.real_features_cov_sum = stats['real_features_cov_sum']
            self.fid.real_features_num_samples = stats['real_features_num_samples']
            print(f'[FID] Loaded real stats from {cache}')
            return
        for batch in tqdm(dataloader, desc='Preloading FID real images'):
            images = batch[0].to(self.device, dtype=torch.float32)
            if images.max() > 1.0:
                images = images / 255.0
            with torch.no_grad():
                self.fid.update(images, real=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'real_features_sum': self.fid.real_features_sum,
                'real_features_cov_sum': self.fid.real_features_cov_sum,
                'real_features_num_samples': self.fid.real_features_num_samples,
            }, cache)
            print(f'[FID] Saved real stats to {cache}')

    def update_fid(self, fake_images):
        """Accumulate generated images for FID. Expects [-1, 1] range."""
        if not hasattr(self, 'fid'):
            return
        with torch.no_grad():
            images = (fake_images.detach() + 1) / 2  # [-1, 1] -> [0, 1]
            self.fid.update(images.clamp(0, 1), real=False)

    def compute_fid(self) -> float | None:
        """Compute FID score and reset fake image accumulator."""
        if not hasattr(self, 'fid'):
            return None
        try:
            score = float(self.fid.compute())
        except RuntimeError:
            return None
        self.fid.reset()  # only resets fake features (reset_real_features=False)
        return score

    def save_model(self, model, args):
        checkpoint_path = Path('./.checkpoints') / f'{args.name}'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = f'{str(checkpoint_path.resolve())}/model_state_dict.pth'
        artifact_name = sanitize_artifact_name(args.name)
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(local_path=args.config_file, name='model_config', is_tmp=True)
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), checkpoint_path)
        elif isinstance(model, dict):
            states = {k: v.state_dict() for k, v in zip(model.keys(), model.values())}
            torch.save(states, checkpoint_path)
        artifact.add_file(local_path=checkpoint_path, name='model_state_dict', is_tmp=True)
        wandb.log_artifact(artifact)

    def log_args(self, args):
        flat = {f'{section}/{k}': v for section, params in args.items() for k, v in params.items()}
        wandb.config.update(flat)
