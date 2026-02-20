# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.5",
#     "torchvision>=0.20",
#     "einops>=0.7",
#     "wandb>=0.19,<0.25",
#     "tqdm",
#     "torchmetrics[image]",
#     "datasets>=2.20",
# ]
# ///
"""
Spherical Autoencoder baseline — no flow refinement.
Based on "Image Generation with a Sphere Encoder" (https://arxiv.org/pdf/2602.15030)

Architecture:
  SphereAE: ViT encoder → spherify (RMS-norm to sphere) → ViT decoder

AE losses (paper eqs 7-9, Appendix D):
  L_pix_recon: SmoothL1 + perceptual, D(v_noisy) vs x
  L_pix_con:   SmoothL1 + perceptual, D(v_NOISY) vs sg(D(v_noisy))
  L_lat_con:   1 - cosine_sim(E(D(v_NOISY)), v)

Generation: sample random sphere point → AE decode directly (no refinement)

Optimizers:
  Muon  — all 2D weight matrices (linear layer weights)
  AdamW — embeddings, positional, norms, biases

Usage:
  uv run baseline.py cifar10
  uv run baseline.py imagenet
  uv run baseline.py cifar10 --wandb --run-name my-run
"""

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger, WandbLogger
from normal_autoencoder import Decoder, Encoder, Tokenizer


# ── Config ───────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    dataset: str
    data_dir: str
    img_size: int
    patch_size: int
    ae_layers: int
    embed_dim: int
    num_heads: int
    query_dim: int
    value_dim: int
    ffn_dim: int
    latent_tokens: int
    sigma_angle_max: float = 80.0
    sigma_mix_max: float = 85.0
    sigma_mix_prob: float = 0.1
    w_l1_recon: float = 1.0
    w_perc_recon: float = 1.0
    w_l1_con: float = 0.5
    w_perc_con: float = 0.5
    w_lat_con: float = 0.1
    batch_size: int = 1280
    lr: float = 1e-4
    total_steps: int = 100_000
    save_every: int = 50_000
    image_every: int = 500
    log_every: int = 5


CIFAR10_CFG = Config(
    dataset="cifar10",
    data_dir="./data",
    img_size=32,
    patch_size=4,
    ae_layers=6,
    embed_dim=256,
    num_heads=8,
    query_dim=32,
    value_dim=32,
    ffn_dim=512,
    latent_tokens=1,
    sigma_angle_max=80.0,
    sigma_mix_max=85.0,
    sigma_mix_prob=0.1,
    w_l1_recon=1.0,
    w_perc_recon=1.0,
    w_l1_con=0.5,
    w_perc_con=0.5,
    w_lat_con=0.1,
    batch_size=1280,
    lr=3e-3,
    total_steps=100_000,
)

IMAGENET_CFG = Config(
    dataset="imagenet",
    data_dir="",
    img_size=256,
    patch_size=16,
    ae_layers=4,
    embed_dim=768,
    num_heads=8,
    query_dim=64,
    value_dim=64,
    ffn_dim=1536,
    latent_tokens=4,
    sigma_angle_max=85.0,
    sigma_mix_max=89.0,
    sigma_mix_prob=0.1,
    w_l1_recon=50.0,
    w_perc_recon=1.0,
    w_l1_con=25.0,
    w_perc_con=1.0,
    w_lat_con=0.1,
    batch_size=256,
    lr=4e-4,
    total_steps=500_000,
)


# ── Perceptual Loss (VGG16 relu3_3 features) ─────────────────────────────────────
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


# ── Sphere helpers ────────────────────────────────────────────────────────────────
def spherify(z: Tensor) -> Tensor:
    rms = z.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    return z / rms


def sample_sigma(b: int, cfg: Config, device) -> Tensor:
    mix_mask = torch.rand(b, device=device) < cfg.sigma_mix_prob
    lo = torch.where(mix_mask, torch.full((b,), cfg.sigma_angle_max, device=device), torch.zeros(b, device=device))
    hi = torch.where(
        mix_mask,
        torch.full((b,), cfg.sigma_mix_max, device=device),
        torch.full((b,), cfg.sigma_angle_max, device=device),
    )
    alpha_deg = lo + torch.rand(b, device=device) * (hi - lo)
    return torch.tan(alpha_deg * (math.pi / 180))


# ── Spherical Autoencoder ─────────────────────────────────────────────────────────
class SphereAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C, H = 3, cfg.img_size
        self.num_patches = (H // cfg.patch_size) ** 2
        self.tok = Tokenizer((C, H, H), cfg.patch_size, cfg.embed_dim)
        self.enc = Encoder(
            cfg.ae_layers,
            cfg.latent_tokens,
            self.num_patches,
            cfg.num_heads,
            cfg.embed_dim,
            cfg.query_dim,
            cfg.value_dim,
            cfg.ffn_dim,
        )
        self.dec = Decoder(
            cfg.ae_layers, self.num_patches, cfg.num_heads, cfg.embed_dim, cfg.query_dim, cfg.value_dim, cfg.ffn_dim
        )
        self.lat = cfg.latent_tokens
        self.dim = cfg.embed_dim
        self.L = cfg.latent_tokens * cfg.embed_dim

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.enc(self.tok(x))
        z_flat = z.reshape(z.shape[0], -1)
        return z_flat, spherify(z_flat)

    def decode(self, v_flat: Tensor) -> Tensor:
        b = v_flat.shape[0]
        toks = self.dec(v_flat.reshape(b, self.lat, self.dim))
        return self.tok.detokenize(toks)

    def forward(self, x: Tensor, percept: PerceptualLoss, cfg: Config) -> tuple[Tensor, dict]:
        b = x.shape[0]
        z_flat, v = self.encode(x)
        e = torch.randn_like(v)
        sigma = sample_sigma(b, cfg, x.device)
        sigma_sub = torch.rand(b, device=x.device) * 0.5 * sigma
        v_noisy = spherify(z_flat + sigma_sub.unsqueeze(-1) * e)
        v_NOISY = spherify(z_flat + sigma.unsqueeze(-1) * e)
        x_noisy = self.decode(v_noisy)
        x_NOISY = self.decode(v_NOISY)

        l1_r = F.smooth_l1_loss(x_noisy, x)
        perc_r = percept(x_noisy, x)
        L_recon = cfg.w_l1_recon * l1_r + cfg.w_perc_recon * perc_r

        l1_c = F.smooth_l1_loss(x_NOISY, x_noisy.detach())
        perc_c = percept(x_NOISY, x_noisy.detach())
        L_con = cfg.w_l1_con * l1_c + cfg.w_perc_con * perc_c

        _, v_rt = self.encode(x_NOISY.detach())
        L_lat = cfg.w_lat_con * (1 - F.cosine_similarity(v_rt, v, dim=-1).mean())

        loss = L_recon + L_con + L_lat
        info = dict(loss=loss.detach(), l1_r=l1_r.detach(), perc_r=perc_r.detach(), L_lat=L_lat.detach())
        return loss, info

    @torch.inference_mode()
    def generate(self, b: int, device) -> Tensor:
        v = spherify(torch.randn(b, self.L, device=device))
        return self.decode(v).clamp(-1, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────────
def get_loader(cfg: Config) -> DataLoader:
    if cfg.dataset == "cifar10":
        tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        ds = torchvision.datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=tf)
    elif cfg.dataset == "imagenet":
        import datasets as hf

        tf = T.Compose(
            [
                T.Resize(cfg.img_size + 32),
                T.CenterCrop(cfg.img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        hf_ds = hf.load_dataset(
            "ILSVRC/imagenet-1k", split="train", trust_remote_code=True, cache_dir=cfg.data_dir or None
        )

        class _HFWrapper(torch.utils.data.Dataset):
            def __getitem__(self, idx):
                item = hf_ds[idx]
                return tf(item["image"].convert("RGB")), item["label"]

            def __len__(self):
                return len(hf_ds)

        ds = _HFWrapper()
    else:
        assert False, f"Unknown dataset: {cfg.dataset}"
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


# ── Optimizers ────────────────────────────────────────────────────────────────────
def make_optimizers(model: SphereAE, cfg: Config) -> tuple:
    skip_kw = ("embed", "positional", "latent_tokens", "output_tokens", "pos")
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
def train(cfg: Config, run_name: str, use_wandb: bool, save_artifacts: bool = True):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    loader = get_loader(cfg)
    model = SphereAE(cfg).to(device)
    percept = PerceptualLoss().to(device)
    muon, adamw = make_optimizers(model, cfg)

    ckpt_dir = Path(f"checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_dir / "config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger_cfg = {
        "data": {"batch_size": cfg.batch_size, "dataset": cfg.dataset, "img_size": cfg.img_size},
        "model": {"ae_layers": cfg.ae_layers, "embed_dim": cfg.embed_dim, "latent_tokens": cfg.latent_tokens},
        "training": {"lr": cfg.lr, "total_steps": cfg.total_steps},
    }
    LoggerCls = WandbLogger if use_wandb else Logger
    logger = LoggerCls(project="sphere-ae", name=run_name, config=logger_cfg, device=device)
    logger.log_args(logger_cfg)
    logger.setup_fid(loader, cache_path=f"./fid_cache/{cfg.dataset}_fid_stats.pt")

    n = sum(p.numel() for p in model.parameters())
    print(f"[baseline] run={run_name}  dataset={cfg.dataset}  device={device}  params={n / 1e6:.1f}M")

    class SaveArgs:
        name = run_name
        config_file = str(config_path)

    def inf_loader():
        while True:
            yield from loader

    step = 0
    pbar = tqdm(total=cfg.total_steps, desc="training", dynamic_ncols=True)
    for x, _ in inf_loader():
        if step >= cfg.total_steps:
            break
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
                _, v = model.encode(x[:8])
                x_recon = model.decode(v)
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
            if save_artifacts:
                logger.save_model(model, SaveArgs)
            pbar.write(f"[baseline] ckpt step={step}" + (f"  fid={fid:.2f}" if fid else ""))

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        pbar.update(1)
        step += 1

    pbar.close()
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    if save_artifacts:
        logger.save_model(model, SaveArgs)
    print(f"[baseline] done → {ckpt_dir}/final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = CIFAR10_CFG if args.dataset == "cifar10" else IMAGENET_CFG
    run_name = args.run_name or f"baseline-{args.dataset}"
    train(cfg, run_name=run_name, use_wandb=args.wandb)
