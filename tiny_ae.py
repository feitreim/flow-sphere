# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.5",
#     "torchvision>=0.20",
#     "einops>=0.7",
#     "wandb>=0.19,<0.25",
#     "tqdm",
#     "torchmetrics[image]",
# ]
# ///
"""
Tiny N-Dim Bottleneck Sphere AE

Architecture:
  Image → Tokenizer → Encoder → [spatial latent (B, T, D)]
      → GlobalBottleneck.compress → [tiny latent (B, N)]  ← spherified here
      → GlobalBottleneck.expand  → [spatial latent (B, T, D)]
      → Decoder → Detokenizer → Reconstruction

Losses (same 3-loss structure as baseline.py):
  L_pix_recon: SmoothL1 + perceptual, decode(z_small_noisy) vs x
  L_pix_con:   SmoothL1 + perceptual, decode(z_small_NOISY) vs sg(decode(z_small_noisy))
  L_lat_con:   1 - cosine_sim(re_encode(x_NOISY), z_small)

Usage:
  uv run tiny_ae.py --data_dir ./data --latent_dim 4 --batch_size 8
"""

import argparse
import dataclasses
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import tqdm

from autoencoder import Decoder, Encoder, Tokenizer
from logger import Logger, WandbLogger


# ── Config ────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data_dir: str = "/cluster/research-groups/wehrwein/webcam/geiranger/biggerframes/"
    image_size: int = 256
    batch_size: int = 112
    patch_size: int = 16
    n_layers: int = 8
    n_heads: int = 8
    embed_dim: int = 1024
    query_dim: int = 64
    ffn_factor: int = 2  # ffn_dim = embed_dim * ffn_factor
    compression_factor: int = 128  # spatial bottleneck: embed_dim // compression_factor
    latent_dim: int = 5  # N-dim global bottleneck (3, 4, or 5)
    lr: float = 3e-4
    total_steps: int = 100_000
    save_every: int = 5_000
    image_every: int = 250
    log_every: int = 5
    w_l1_recon: float = 1.0
    w_perc_recon: float = 0.1
    w_l1_con: float = 1.0
    w_perc_con: float = 0.1
    w_lat_con: float = 1.0


# ── Dataloader ────────────────────────────────────────────────────────────────────
class TimestampImageDataset(Dataset):
    def __init__(self, dir: str, rank: int = 0):
        self.dir = Path(dir)
        self.files = sorted(f for ext in ("*.jpeg", "*.jpg", "*.png") for f in self.dir.glob(ext))
        assert len(self.files) > 0, f"no images found in {self.dir}"
        if rank > 0:
            rng = random.Random(42)
            for _ in range(rank):
                rng.shuffle(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path = self.files[idx]
        timestamp = int(float(path.stem))
        image = read_image(str(path))
        return image, timestamp


def collate(batch):
    return torch.stack([x[0] for x in batch]).squeeze(), torch.tensor([x[1] for x in batch]).squeeze()


def create_dataloader(dir: str, batch_size: int, rank: int = 0) -> DataLoader:
    dataset = TimestampImageDataset(dir, rank=rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate,
    )


# ── Perceptual Loss ───────────────────────────────────────────────────────────────
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        def prep(t: Tensor) -> Tensor:
            t = t * 0.5 + 0.5
            t = (t - self.mean) / self.std
            if t.shape[-1] < 64:
                t = F.interpolate(t, size=64, mode="bilinear", align_corners=False)
            return t

        return F.l1_loss(self.features(prep(x)), self.features(prep(y)))


# ── Global Bottleneck ─────────────────────────────────────────────────────────────
class GlobalBottleneck(nn.Module):
    def __init__(self, spatial_dim: int, latent_dim: int):
        super().__init__()
        self.compress_linear = nn.Linear(spatial_dim, latent_dim, bias=False)
        self.expand_linear = nn.Linear(latent_dim, spatial_dim, bias=False)

    def compress(self, z_flat: Tensor) -> Tensor:
        # (B, T*D) → (B, N), spherified
        return spherify(self.compress_linear(z_flat))

    def expand(self, z_small: Tensor, T: int, D: int) -> Tensor:
        # (B, N) → (B, T, D)
        return self.expand_linear(z_small).reshape(z_small.shape[0], T, D)


# ── Sphere helpers ────────────────────────────────────────────────────────────────
def spherify(z: Tensor) -> Tensor:
    rms = z.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    return z / rms


def sample_sigma(b: int, device, sigma_max: float = 80.0) -> Tensor:
    alpha_deg = torch.rand(b, device=device) * sigma_max
    return torch.tan(alpha_deg * (math.pi / 180))


# ── TinyAE ────────────────────────────────────────────────────────────────────────
class TinyAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C, H = 3, cfg.image_size
        self.T = (H // cfg.patch_size) ** 2
        self.D = cfg.embed_dim // cfg.compression_factor
        self.N = cfg.latent_dim

        ffn_dim = cfg.embed_dim * cfg.ffn_factor

        self.tok = Tokenizer((C, H, H), cfg.patch_size, cfg.embed_dim)
        self.enc = Encoder(
            cfg.n_layers,
            self.T,
            cfg.n_heads,
            cfg.embed_dim,
            cfg.query_dim,
            cfg.query_dim,
            ffn_dim,
            cfg.compression_factor,
        )
        self.dec = Decoder(
            cfg.n_layers,
            self.T,
            cfg.n_heads,
            cfg.embed_dim,
            cfg.query_dim,
            cfg.query_dim,
            ffn_dim,
            cfg.compression_factor,
        )
        self.bottleneck = GlobalBottleneck(self.T * self.D, cfg.latent_dim)

    def encode_spatial(self, x: Tensor) -> Tensor:
        return self.enc(self.tok(x))  # (B, T, D)

    def compress(self, z_flat: Tensor) -> Tensor:
        return self.bottleneck.compress(z_flat)  # (B, N)

    def decode(self, z_small: Tensor) -> Tensor:
        z_spatial = self.bottleneck.expand(z_small, self.T, self.D)
        return self.tok.detokenize(self.dec(z_spatial))

    def encode(self, x: Tensor) -> Tensor:
        return self.compress(self.encode_spatial(x).reshape(x.shape[0], -1))

    @torch.inference_mode()
    def generate(self, b: int, device) -> Tensor:
        z = spherify(torch.randn(b, self.N, device=device))
        return self.decode(z).clamp(-1, 1)

    def forward(self, x: Tensor, percept: PerceptualLoss, cfg: Config) -> tuple[Tensor, dict]:
        b = x.shape[0]

        z_flat = self.encode_spatial(x).reshape(b, -1)  # (B, T*D)
        z_small = self.compress(z_flat)  # (B, N)
        assert z_small.shape == (b, self.N)

        # noise on the spatial latent, shared direction
        e = torch.randn_like(z_flat)
        sigma = sample_sigma(b, x.device)
        sigma_sub = torch.rand(b, device=x.device) * 0.5 * sigma

        z_small_noisy = self.compress(z_flat + sigma_sub[:, None] * e)
        z_small_NOISY = self.compress(z_flat + sigma[:, None] * e)

        x_noisy = self.decode(z_small_noisy)
        x_NOISY = self.decode(z_small_NOISY)

        # L_recon: lightly-noisy decode vs input
        l1_r = F.smooth_l1_loss(x_noisy, x)
        perc_r = percept(x_noisy, x)
        L_recon = cfg.w_l1_recon * l1_r + cfg.w_perc_recon * perc_r

        # L_con: heavily-noisy decode vs sg(lightly-noisy decode)
        l1_c = F.smooth_l1_loss(x_NOISY, x_noisy.detach())
        perc_c = percept(x_NOISY, x_noisy.detach())
        L_con = cfg.w_l1_con * l1_c + cfg.w_perc_con * perc_c

        # L_lat: re-encode heavily-noisy image back to tiny latent
        z_small_rt = self.encode(x_NOISY.detach())
        L_lat = cfg.w_lat_con * (1 - F.cosine_similarity(z_small_rt, z_small, dim=-1).mean())

        loss = L_recon + L_con + L_lat
        return loss, dict(
            loss=loss.detach(), l1_r=l1_r.detach(), perc_r=perc_r.detach(), l1_c=l1_c.detach(), L_lat=L_lat.detach()
        )


# ── Optimizers ────────────────────────────────────────────────────────────────────
def make_optimizers(model: TinyAE, cfg: Config):
    skip_kw = ("embed", "positional", "pos")
    muon_p, adamw_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(k in name for k in skip_kw):
            muon_p.append(p)
        else:
            adamw_p.append(p)
    muon = torch.optim.Muon(muon_p, lr=cfg.lr * 10, momentum=0.95)
    adamw = torch.optim.AdamW(adamw_p, lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95))
    return muon, adamw


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


# ── Training ──────────────────────────────────────────────────────────────────────
def train(cfg: Config, run_name: str, use_wandb: bool):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    loader = create_dataloader(cfg.data_dir, cfg.batch_size)
    model = TinyAE(cfg).to(device)
    percept = PerceptualLoss().to(device)
    muon, adamw = make_optimizers(model, cfg)

    ckpt_dir = Path(f"checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_dir / "config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger_cfg = {
        "data": {"batch_size": cfg.batch_size, "data_dir": cfg.data_dir, "image_size": cfg.image_size},
        "model": {
            "n_layers": cfg.n_layers,
            "embed_dim": cfg.embed_dim,
            "compression_factor": cfg.compression_factor,
            "latent_dim": cfg.latent_dim,
        },
        "training": {"lr": cfg.lr, "total_steps": cfg.total_steps},
    }
    LoggerCls = WandbLogger if use_wandb else Logger
    logger = LoggerCls(project="sphere-flow", name=run_name, config=logger_cfg, device=device)
    logger.log_args(logger_cfg)
    logger.setup_fid(loader, cache_path="./fid_cache/tiny_ae_fid_stats.pt")

    class SaveArgs:
        name = run_name
        config_file = str(config_path)

    n = sum(p.numel() for p in model.parameters())
    print(
        f"[tiny_ae] run={run_name}  device={device}  params={n / 1e6:.2f}M  latent_dim={cfg.latent_dim}  spatial={model.T}×{model.D}={model.T * model.D}"
    )

    def inf_loader():
        while True:
            yield from loader

    step = 0
    pbar = tqdm(total=cfg.total_steps, desc="training", dynamic_ncols=True)
    for images, _ in inf_loader():
        if step >= cfg.total_steps:
            break

        x = images.float() / 127.5 - 1.0
        x = x.to(device)

        lr = cosine_lr(step, cfg.total_steps, warmup=500, base_lr=cfg.lr)
        for pg in muon.param_groups:
            pg["lr"] = lr * 10
        for pg in adamw.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss, info = model(x, percept, cfg)
        muon.zero_grad()
        adamw.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adamw.step()

        logger.train_log(**info, lr=torch.tensor(lr))
        logger.iter()

        if step % cfg.log_every == 0:
            logger.train_step()

        if step > 0 and step % cfg.image_every == 0:
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                samples = model.generate(8, device)
                z = model.encode(x[:8])
                x_recon = model.decode(z)
            mk = lambda imgs: torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
            logger.log_image(
                "train/recon",
                mk(torch.cat([x[:8], x_recon.clamp(-1, 1)])),
                caption=f"step {step} | top: input  bottom: recon",
            )
            logger.log_image("train/samples", mk(samples), caption=f"step {step}")
            logger.update_fid(samples)

        if step > 0 and step % cfg.save_every == 0:
            torch.save(
                {"step": step, "model": model.state_dict(), "muon": muon.state_dict(), "adamw": adamw.state_dict()},
                ckpt_dir / f"ckpt_{step:07d}.pt",
            )
            fid = logger.compute_fid()
            if fid is not None:
                logger.log("val", {"fid": fid})
            logger.save_model(model, SaveArgs)
            pbar.write(f"[tiny_ae] ckpt step={step}" + (f"  fid={fid:.2f}" if fid else ""))

        pbar.set_postfix(loss=f"{info['loss'].item():.4f}", lr=f"{lr:.2e}")
        pbar.update(1)
        step += 1

    pbar.close()
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    logger.save_model(model, SaveArgs)
    print(f"[tiny_ae] done → {ckpt_dir}/final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data_dir")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--query_dim", type=int)
    parser.add_argument("--ffn_factor", type=int)
    parser.add_argument("--compression_factor", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--image_every", type=int)
    parser.add_argument("--log_every", type=int)
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("wandb", "run_name")}
    cfg = Config(**overrides)
    run_name = args.run_name or f"tiny-ae-n{cfg.latent_dim}"
    train(cfg, run_name=run_name, use_wandb=args.wandb)
