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
Spherical Autoencoder + Flow Refinement Trainer
Based on "Image Generation with a Sphere Encoder" (https://arxiv.org/pdf/2602.15030)
And "Sample What You Can't Compress" (https://arxiv.org/pdf/2409.02529)
And "Back to Basics: Let Denoising Generative Models Denoise" (https://arxiv.org/pdf/2511.13720)

Architecture:
  SphereAE:         ViT encoder → spherify (RMS-norm to sphere) → ViT decoder
  SphereFlowTrainer: SphereAE for initial decode + DiT flow conditioned on sphere latent

AE losses (paper eqs 7-9, Appendix D):
  L_pix_recon: SmoothL1 + perceptual, D(v_noisy) vs x
  L_pix_con:   SmoothL1 + perceptual, D(v_NOISY) vs sg(D(v_noisy))
  L_lat_con:   1 - cosine_sim(E(D(v_NOISY)), v)

Flow loss: velocity matching (x-prediction formulation, FlowTrainer-style)

Training phases:
  Phase 1 (ae_warmup_steps): AE losses only
  Phase 2: AE + flow losses jointly

Optimizers:
  Muon  — all 2D weight matrices (linear layer weights)
  AdamW — embeddings, positional, norms, biases

Usage:
  uv run sphere_flow_train.py cifar10
  uv run sphere_flow_train.py imagenet
  uv run sphere_flow_train.py cifar10 --wandb --run-name my-run
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
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import Decoder, Encoder, Tokenizer
from dit import DiT
from logger import Logger, WandbLogger


# ── Config ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    dataset: str
    data_dir: str
    img_size: int
    patch_size: int
    # AE model
    ae_layers: int
    embed_dim: int
    num_heads: int
    query_dim: int
    value_dim: int
    ffn_dim: int
    # Flow DiT
    flow_layers: int
    flow_embed_dim: int
    flow_query_dim: int
    flow_value_dim: int
    flow_ffn_dim: int
    compression_factor: int = 4
    flow_steps: int = 100
    flow_ratio: float = 0.75
    use_bn: bool = True  # BN on encoder output before spherify (Appendix C.3)
    # Sphere noise: σ = tan(α), α jittered from angle ranges (Appendix D)
    sigma_angle_max: float = 80.0  # degrees, upper bound of normal range
    sigma_mix_max: float = 85.0  # degrees, upper bound of overflow range
    sigma_mix_prob: float = 0.1  # prob of sampling from overflow range
    # Loss weights (Appendix D values)
    w_l1_recon: float = 1.0
    w_perc_recon: float = 1.0
    w_l1_con: float = 0.5
    w_perc_con: float = 0.5
    w_lat_con: float = 0.1
    w_flow: float = 1.0
    # Training
    batch_size: int = 128
    lr: float = 1e-4
    total_steps: int = 100_000
    ae_warmup_steps: int = 10_000
    save_every: int = 50_000
    image_every: int = 500
    log_every: int = 5


CIFAR10_CFG = Config(
    dataset="cifar10",
    data_dir="./data",
    img_size=32,
    patch_size=2,
    ae_layers=6,
    embed_dim=256,
    num_heads=8,
    query_dim=32,
    value_dim=32,
    compression_factor=32,
    ffn_dim=512,
    use_bn=True,
    flow_layers=6,
    flow_embed_dim=256,
    flow_query_dim=32,
    flow_value_dim=32,
    flow_ffn_dim=512,
    sigma_angle_max=80.0,
    sigma_mix_max=85.0,
    sigma_mix_prob=0.1,
    w_l1_recon=1.0,
    w_perc_recon=1.0,
    w_l1_con=1.0,
    w_perc_con=1.0,
    w_lat_con=0.1,
    w_flow=1.0,
    batch_size=256,
    lr=6e-4,
    total_steps=100_000,
    ae_warmup_steps=10_000,
    image_every=250,
)

IMAGENET_CFG = Config(
    dataset="imagenet",
    data_dir="",
    img_size=256,
    patch_size=16,
    ae_layers=6,
    embed_dim=768,
    num_heads=8,
    query_dim=64,
    value_dim=64,
    compression_factor=16,
    ffn_dim=3072,
    use_bn=False,
    flow_layers=12,
    flow_embed_dim=768,
    flow_query_dim=64,
    flow_value_dim=64,
    flow_ffn_dim=2048,
    sigma_angle_max=85.0,
    sigma_mix_max=89.0,
    w_l1_recon=50.0,
    w_perc_recon=1.0,
    w_l1_con=50.0,
    w_perc_con=1.0,
    w_lat_con=0.5,
    w_flow=1.0,
    batch_size=80,
    lr=3e-3,
    total_steps=500_000,
    ae_warmup_steps=10_000,
    image_every=500,
)

IMAGENET64_CFG = Config(
    dataset="64imagenet",
    data_dir="",
    img_size=64,
    patch_size=4,
    ae_layers=6,
    embed_dim=512,
    num_heads=8,
    query_dim=64,
    value_dim=64,
    compression_factor=32,  # @ patch_size = 4 this yields compression ratio 3.0
    ffn_dim=2048,
    use_bn=True,
    flow_layers=8,
    flow_embed_dim=512,
    flow_query_dim=64,
    flow_value_dim=64,
    flow_ffn_dim=2048,
    sigma_angle_max=85.0,
    sigma_mix_max=89.0,
    w_l1_recon=10.0,
    w_perc_recon=1.0,
    w_l1_con=10.0,
    w_perc_con=1.0,
    w_lat_con=0.1,
    w_flow=1.0,
    batch_size=6,
    lr=2e-4,
    total_steps=500_000,
    ae_warmup_steps=50_000,
    image_every=500,
)


# ── Perceptual Loss (VGG16 relu3_3 features) ────────────────────────────────────
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
            t = t * 0.5 + 0.5  # [-1,1] → [0,1]
            t = (t - self.mean) / self.std
            if t.shape[-1] < 64:  # VGG needs ≥64px
                t = F.interpolate(t, size=64, mode="bilinear", align_corners=False)
            return t

        return F.l1_loss(self.features(prep(x)), self.features(prep(y)))


# ── Sphere helpers ───────────────────────────────────────────────────────────────
def spherify(z: Tensor) -> Tensor:
    """RMS-normalize a flat vector onto sphere of radius √L.  z: (b, L) → (b, L)"""
    rms = z.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    return z / rms


def sample_sigma(b: int, cfg: Config, device) -> Tensor:
    """
    Sample noise magnitude σ = tan(α) where α is jittered from angle ranges.
    With prob sigma_mix_prob: α ∈ [sigma_angle_max, sigma_mix_max]  (overflow range)
    Otherwise:                α ∈ [0, sigma_angle_max]               (normal range)
    """
    mix_mask = torch.rand(b, device=device) < cfg.sigma_mix_prob
    lo = torch.where(
        mix_mask,
        torch.full((b,), cfg.sigma_angle_max, device=device),
        torch.zeros(b, device=device),
    )
    hi = torch.where(
        mix_mask,
        torch.full((b,), cfg.sigma_mix_max, device=device),
        torch.full((b,), cfg.sigma_angle_max, device=device),
    )
    alpha_deg = lo + torch.rand(b, device=device) * (hi - lo)
    return torch.tan(alpha_deg * (math.pi / 180))


# ── Spherical Autoencoder ────────────────────────────────────────────────────────
class SphereAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C, H = 3, cfg.img_size
        self.num_patches = (H // cfg.patch_size) ** 2
        self.tok = Tokenizer((C, H, H), cfg.patch_size, cfg.embed_dim)
        self.enc = Encoder(
            cfg.ae_layers,
            self.num_patches,
            cfg.num_heads,
            cfg.embed_dim,
            cfg.query_dim,
            cfg.value_dim,
            cfg.ffn_dim,
            cfg.compression_factor,
        )
        self.dec = Decoder(
            cfg.ae_layers,
            self.num_patches,
            cfg.num_heads,
            cfg.embed_dim,
            cfg.query_dim,
            cfg.value_dim,
            cfg.ffn_dim,
            cfg.compression_factor,
        )
        self.lat = self.num_patches
        self.dim = cfg.embed_dim // cfg.compression_factor
        self.L = self.num_patches * self.dim
        self.bn = nn.BatchNorm1d(self.L) if cfg.use_bn else None

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (z_flat, v): raw flat latent and sphere-projected latent."""
        z = self.enc(self.tok(x))  # (b, lat, dim)
        z_flat = z.reshape(z.shape[0], -1)  # (b, L)
        if self.bn is not None:
            z_flat = self.bn(z_flat)
        return z_flat, spherify(z_flat)

    def decode(self, v_flat: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (image, dec_tokens) from sphere-flat vector."""
        b = v_flat.shape[0]
        toks = self.dec(v_flat.reshape(b, self.lat, self.dim))  # (b, patches, dim)
        return self.tok.detokenize(toks), toks

    def forward(self, x: Tensor, cfg: Config) -> tuple:
        """
        Full noisy forward pass.
        Returns: (v, v_noisy, v_NOISY, x_noisy, x_NOISY, dec_toks_noisy)
          v_noisy:  sphere latent with small noise σ_sub = s·σ, s~U[0, 0.5]
          v_NOISY:  sphere latent with large noise σ~U[0, σ_max]
          Both share the same noise direction e.
        """
        b = x.shape[0]
        z_flat, v = self.encode(x)
        e = torch.randn_like(v)  # shared direction
        sigma = sample_sigma(b, cfg, x.device)  # (b,) large noise
        sigma_sub = torch.rand(b, device=x.device) * 0.5 * sigma  # (b,) small noise
        v_noisy = spherify(z_flat + sigma_sub.unsqueeze(-1) * e)
        v_NOISY = spherify(z_flat + sigma.unsqueeze(-1) * e)
        x_noisy, dec_toks = self.decode(v_noisy)
        x_NOISY, _ = self.decode(v_NOISY)
        return v, v_noisy, v_NOISY, x_noisy, x_NOISY, dec_toks


# ── Sphere + Flow Trainer ────────────────────────────────────────────────────────
class SphereFlowTrainer(nn.Module):
    """
    Spherical AE with DiT flow refinement.

    Sphere AE produces an initial decode x_0 from the spherical latent.
    The DiT flow model then refines x_0 → x, conditioned on:
      c        = sphere latent tokens  (b, num_patches, latent_dim)
      x_i_toks = AE decoder output tokens (b, num_patches, embed_dim)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        C, H = 3, cfg.img_size
        self.ae = SphereAE(cfg)
        self.flow = DiT(
            steps=cfg.flow_steps,
            num_layers=cfg.flow_layers,
            heads=cfg.num_heads,
            embed_dim=cfg.flow_embed_dim,
            query_dim=cfg.flow_query_dim,
            value_dim=cfg.flow_value_dim,
            ffn_dim=cfg.flow_ffn_dim,
            img_chw=(C, H, H),
            patch_size=cfg.patch_size,
            ratio=cfg.flow_ratio,
            context_dim=cfg.embed_dim // cfg.compression_factor,
            toks_dim=cfg.embed_dim,
            flow=True,
        )
        self.cfg = cfg

    def forward(self, x: Tensor, perceptual: PerceptualLoss, train_flow: bool = True) -> tuple[Tensor, dict]:
        cfg = self.cfg
        v, v_noisy, v_NOISY, x_noisy, x_NOISY, dec_toks = self.ae(x, cfg)

        # ── AE losses (paper eqs 7-9) ─────────────────────────────────────────
        # L_pix_recon: reconstruct x from lightly-noisy latent
        l1_r = F.smooth_l1_loss(x_noisy, x)
        perc_r = perceptual(x_noisy, x)
        L_recon = cfg.w_l1_recon * l1_r + cfg.w_perc_recon * perc_r

        # L_pix_con: heavily-noisy decode should match lightly-noisy decode (sg on target)
        l1_c = F.smooth_l1_loss(x_NOISY, x_noisy.detach())
        perc_c = perceptual(x_NOISY, x_noisy.detach())
        L_con = cfg.w_l1_con * l1_c + cfg.w_perc_con * perc_c

        # L_lat_con: encoder maps D(v_NOISY) back to original sphere latent v
        # sg on x_NOISY so this loss only trains the encoder, not the decoder
        _, v_rt = self.ae.encode(x_NOISY.detach())
        L_lat = cfg.w_lat_con * (1 - F.cosine_similarity(v_rt, v, dim=-1).mean())

        ae_loss = L_recon + L_con + L_lat
        info = dict(
            ae=ae_loss.detach(),
            l1_r=l1_r.detach(),
            perc_r=perc_r.detach(),
            L_lat=L_lat.detach(),
            l1_c=l1_c.detach(),
            perc_c=perc_c.detach(),
        )

        if not train_flow:
            return ae_loss, info

        # ── Flow loss (x-prediction, JiT-style) ───────────────────────
        b = x.shape[0]
        c = v_noisy.reshape(b, self.ae.lat, self.ae.dim)  # sphere latent as tokens
        # start from ratio*noise + (1-ratio)*x_0, flow toward x
        noise = torch.randn_like(x_noisy)
        x_start = cfg.flow_ratio * noise + (1 - cfg.flow_ratio) * x_noisy.detach()
        t = torch.rand(b, device=x.device)
        x_t = (1 - t.view(b, 1, 1, 1)) * x_start + t.view(b, 1, 1, 1) * x
        x_pred = self.flow(x_t, dec_toks.detach(), c, t)
        v_tgt = (x - x_t) / (1 - t).clamp(0.05).view(b, 1, 1, 1)
        v_pred = (x_pred - x_t) / (1 - t).clamp(0.05).view(b, 1, 1, 1)
        flow_loss = (v_tgt - v_pred).pow(2).mean()

        total = ae_loss + cfg.w_flow * flow_loss
        info["flow"] = flow_loss.detach()
        return total, info

    @torch.inference_mode()
    def generate(self, b: int, device, N: int = 10) -> Tensor:
        """Sample b images: random sphere point → AE decode → N Heun flow-refinement steps."""
        cfg = self.cfg
        v = spherify(torch.randn(b, self.ae.L, device=device))
        x_0, toks = self.ae.decode(v)
        c = v.reshape(b, self.ae.lat, self.ae.dim)
        x_t = cfg.flow_ratio * torch.randn_like(x_0) + (1 - cfg.flow_ratio) * x_0
        t = torch.zeros(b, device=device)
        dt = 1.0 / N
        half_dt = dt / 2

        def V(x, t):
            x_pred = self.flow(x, toks, c, t)
            return (x_pred - x) / (1 - t).clamp(0.05).view(b, 1, 1, 1)

        for _ in range(N - 1):
            v_1 = V(x_t, t)
            x_i = x_t + v_1 * dt
            v_2 = V(x_i, t + dt)
            x_t = x_t + half_dt * (v_1 + v_2)
            t = t + dt
        return x_0.clamp(-1, 1), self.flow(x_t, toks, c, t).clamp(-1, 1)


# ── Dataset ──────────────────────────────────────────────────────────────────────
def get_loader(cfg: Config) -> DataLoader:
    if cfg.dataset == "cifar10":
        tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        ds = torchvision.datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=tf)
    elif cfg.dataset in ("imagenet", "64imagenet"):
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
            "ILSVRC/imagenet-1k",
            split="train",
            trust_remote_code=True,
            cache_dir=cfg.data_dir or None,
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
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="fork",
    )


# ── Optimizers ───────────────────────────────────────────────────────────────────
def make_optimizers(model: SphereFlowTrainer, cfg: Config) -> tuple[torch.optim.Muon, torch.optim.AdamW]:
    """
    torch.optim.Muon for all 2D weight matrices (linear layer weights).
    AdamW for embeddings, positional encodings, layernorm params, biases.
    Muon lr = 10x AdamW lr (orthogonalized steps are more aggressive).
    """
    skip_kw = (
        "embed",
        "positional",
        "latent_tokens",
        "output_tokens",
        "time_embeds",
        "pos",
    )
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


# ── Training ─────────────────────────────────────────────────────────────────────
def train(cfg: Config, run_name: str, use_wandb: bool, save_artifacts: bool = True):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    loader = get_loader(cfg)
    model = SphereFlowTrainer(cfg).to(device)
    percept = PerceptualLoss().to(device)
    muon, adamw = make_optimizers(model, cfg)

    ckpt_dir = Path(f"checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # write config.json for wandb artifact logging
    config_path = ckpt_dir / "config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger_cfg = {
        "data": {
            "batch_size": cfg.batch_size,
            "dataset": cfg.dataset,
            "img_size": cfg.img_size,
        },
        "model": {
            "ae_layers": cfg.ae_layers,
            "embed_dim": cfg.embed_dim,
            "compression_factor": cfg.compression_factor,
            "flow_layers": cfg.flow_layers,
        },
        "training": {
            "lr": cfg.lr,
            "total_steps": cfg.total_steps,
            "ae_warmup_steps": cfg.ae_warmup_steps,
        },
    }
    LoggerCls = WandbLogger if use_wandb else Logger
    logger = LoggerCls(project="sphere-flow", name=run_name, config=logger_cfg, device=device)
    logger.log_args(logger_cfg)
    logger.setup_fid(loader, cache_path=f"./fid_cache/{cfg.dataset}_fid_stats.pt")

    n_ae = sum(p.numel() for p in model.ae.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"[sphere_flow] run={run_name}  dataset={cfg.dataset}  device={device}")
    print(f"[sphere_flow] ae={n_ae / 1e6:.1f}M  flow={n_flow / 1e6:.1f}M  ae_warmup={cfg.ae_warmup_steps}")

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
        train_flow = step >= cfg.ae_warmup_steps

        lr = cosine_lr(step, cfg.total_steps, warmup=500, base_lr=cfg.lr)
        for pg in muon.param_groups:
            pg["lr"] = lr * 10
        for pg in adamw.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss, info = model(x, percept, train_flow=train_flow)
        muon.zero_grad()
        adamw.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adamw.step()

        # accumulate metrics every step
        epoch = step // len(loader)
        logger.train_log(loss=loss.detach(), **info)
        logger.train_log(lr=torch.tensor(lr), phase=torch.tensor(float(train_flow)), epoch=torch.tensor(float(epoch)))
        logger.iter()

        if step % cfg.log_every == 0:
            logger.train_step()

        if step > 0 and step % cfg.image_every == 0:
            with (
                torch.no_grad(),
                torch.autocast(device_type=device, dtype=torch.bfloat16),
            ):
                ae_samples, samples = model.generate(8, device, N=10)
                _, v = model.ae.encode(x[:8])
                x_recon, _ = model.ae.decode(v)
            mk = lambda imgs: torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
            logger.log_image(
                "train/recon",
                mk(torch.cat([x[:8], x_recon.clamp(-1, 1)])),
                caption=f"step {step} | top: input  bottom: recon",
            )
            logger.log_image("train/ae", mk(ae_samples), caption=f"step {step}")
            logger.log_image("train/flow", mk(samples), caption=f"step {step}")
            logger.update_fid(samples)

        if step > 0 and step % cfg.save_every == 0:
            # checkpoint with optimizer states
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "muon": muon.state_dict(),
                    "adamw": adamw.state_dict(),
                },
                ckpt_dir / f"ckpt_{step:07d}.pt",
            )
            fid = logger.compute_fid()
            if fid is not None:
                logger.log("val", {"fid": fid})
            if save_artifacts:
                logger.save_model(model, SaveArgs)
            pbar.write(f"[sphere_flow] ckpt step={step}" + (f"  fid={fid:.2f}" if fid else ""))

        phase_str = "ae+flow" if train_flow else "ae"
        pbar.set_postfix(loss=f"{loss.item():.4f}", phase=phase_str, lr=f"{lr:.2e}", epoch=epoch)
        pbar.update(1)
        step += 1

    pbar.close()
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    if save_artifacts:
        logger.save_model(model, SaveArgs)
    print(f"[sphere_flow] done → {ckpt_dir}/final.pt")


# ── LR Sweep ─────────────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/fid", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
    },
}


def run_sweep(count: int):
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="sphere-flow")

    def sweep_run():
        torch._dynamo.reset()
        run = wandb.init()
        cfg = dataclasses.replace(
            CIFAR10_CFG,
            lr=run.config["lr"],
            total_steps=50_000,
            save_every=10_000,
            image_every=1_000,
        )
        train(cfg, run_name=run.name, use_wandb=True, save_artifacts=False)
        wandb.finish()

    wandb.agent(sweep_id, sweep_run, count=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cifar10", choices=["cifar10", "imagenet", "64imagenet"])
    parser.add_argument("--wandb", action="store_true", help="use WandbLogger instead of console Logger")
    parser.add_argument("--run-name", default=None, help="run name for logging and checkpoints")
    parser.add_argument("--sweep", action="store_true", help="run a wandb LR sweep on cifar10")
    parser.add_argument("--sweep-count", type=int, default=10, help="number of sweep trials")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.sweep_count)
    else:
        cfg = {"cifar10": CIFAR10_CFG, "imagenet": IMAGENET_CFG, "64imagenet": IMAGENET64_CFG}[args.dataset]
        run_name = args.run_name or f"sphere-flow-{args.dataset}"
        train(cfg, run_name=run_name, use_wandb=args.wandb)
