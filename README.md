# Quick Visualization

For a quick demo with an 8-dimensional model trained on landscape imagery:

```bash
uv run latent_explorer.py --wandb-artifact eitreif/sphere-flow/8dim_geir:latest
```

# Spherical Flow

Implementation of a spherical autoencoder with flow-based refinement for image generation, based on:

- [Image Generation with a Sphere Encoder](https://arxiv.org/pdf/2602.15030) (Yue et al., 2026)
- [Sample What You Can't Compress](https://arxiv.org/pdf/2409.02529) (Birodkar et al. 2025)
- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/pdf/2511.13720) (Li et al. 2025)

## Idea

Standard image autoencoders produce latents in unconstrained space, which makes generative modelling of the latent distribution harder. This work constrains the latent to lie on a hypersphere by RMS-normalising the encoder output. As shown in Image Generation with a Sphere Encoder, their system produces great looking results with a latent space that is great for sampling. A criticism I have seen of their approach is that their model is HUGE for imagenet, with around 1B params to achieve the results that they do. That withstanding, I think the basic idea, spherical latent is great.

My idea is to drastically reduce this parameter count by training a small spherical AE, then applying techniques similar to Sample What You Can't Compress and having a flow model that will few-step refine the images to be of very high quality. Relative to SWYCC I have switched things up a bit, their refinement model is just a full diffusion model that start from noise and just concatenates the initial decoded image,
a) my diffusion/FM setup much more closely resembles JiT, from Back to Basics: Let Denoising Generative Models Denoise.
b) I jump start diffusion by instead of diffusing from pure noise to the image, the coupling is (image_generated_by_ae + XX%\_noise) -> output. This is weird idea AFAIK, but in my experience it significantly speeds up training, as the flow has a lot more image structure and information right away, essentially skipping the beginning of reconstruction.

This process still works very well as few-step, but I am hopeful that it can achieve similar results w/ less parameters and less training compute.

Generation is a two-stage process:

1. **AE decode**: a random point on the sphere is decoded directly to a rough image
2. **Flow refinement**: a DiT flow model refines that image, conditioned on the sphere latent

## Architecture

```
Image x
  │
  ▼
ViT Encoder ──────────────────────────────────────────┐
  │                                                   │
  ▼                                                   │
RMS-norm → sphere latent v  (b, latent_tokens × dim)  │
  │                                                   │
  ▼                                                   │
ViT Decoder → x_0 (rough decode)                      │
  │                            context c = v as tokens│
  ▼                                                   │
DiT Flow ◄────────────────────────────────────────────┘
  │
  ▼
x (refined image)
```

Both the encoder and decoder are ViT-style transformers with RoPE, QK-norm, and flash attention. The latent is a flat vector of `latent_tokens × embed_dim` dimensions, RMS-normalised to the sphere. The flow model is a DiT conditioned on both the sphere latent tokens and the AE decoder's internal token representations.

## Training

Training has two phases:

**Phase 1 — AE warmup** (`ae_warmup_steps`): AE losses only, no flow.

**Phase 2 — Joint**: AE + flow losses together.

### AE Losses

The AE is trained with two noise levels sharing the same noise direction `e`:

- `v_noisy = spherify(z + σ_sub · e)` — small noise, `σ_sub ~ U[0, 0.5σ]`
- `v_NOISY = spherify(z + σ · e)` — large noise, `σ = tan(α)`, `α ~ U[0, 80°]`

| Loss          | Formula                                                 | Purpose                                                                           |
| ------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `L_pix_recon` | SmoothL1 + perceptual, `D(v_noisy)` vs `x`              | Reconstruct from lightly noised latent                                            |
| `L_pix_con`   | SmoothL1 + perceptual, `D(v_NOISY)` vs `sg(D(v_noisy))` | Consistency: heavily noised decode should match lightly noised decode             |
| `L_lat_con`   | `1 - cosine_sim(E(D(v_NOISY)), v)`                      | Encoder round-trip: re-encoding a noisy decode recovers the original sphere point |

The mixed noise schedule (10% of samples draw `α` from `[80°, 85°]`) prevents the latent space from collapsing near the equator.

### Flow Loss

Velocity matching in x-prediction form. The flow model starts from a mixture of the AE decode and Gaussian noise:

```
x_start = ratio · noise + (1 - ratio) · x_0
x_t     = (1 - t) · x_start + t · x
```

The model predicts `x` directly, with the target velocity derived from the x-prediction. `ratio=0.75` controls how much of the starting point is pure noise vs. the AE decode.

### Optimizers

Two optimisers run simultaneously:

- **Muon** (lr × 10) — all 2D weight matrices (attention projections, FFN weights)
- **AdamW** (lr) — embeddings, positional encodings, norms, biases

LR schedule: cosine decay with 500-step linear warmup.

## Configs

|               | CIFAR-10 | ImageNet |
| ------------- | -------- | -------- |
| Image size    | 32×32    | 256×256  |
| AE layers     | 6        | 4        |
| Embed dim     | 256      | 768      |
| Latent tokens | 16       | 32       |
| Flow layers   | 6        | 12       |
| Batch size    | 256      | 128      |
| Total steps   | 100k     | 500k     |
| AE warmup     | 10k      | 10k      |

## Usage

```bash
# CIFAR-10
uv run sphere_flow_train.py cifar10

# ImageNet
uv run sphere_flow_train.py imagenet

# With wandb logging
uv run sphere_flow_train.py cifar10 --wandb --run-name my-run

# Hyperparameter sweep (Bayesian search over lr)
uv run sphere_flow_train.py --sweep --sweep-count 20
```

Checkpoints (with optimizer states) are saved to `checkpoints/<run-name>/` every `save_every` steps. FID is computed at each checkpoint.

## Latent Space Explorer

The `latent_explorer.py` script provides an interactive Gradio GUI for exploring the TinyAE latent space. It downloads a trained model from wandb and lets you manipulate each latent dimension with sliders to see how they affect the generated image in real-time.

### Usage

```bash
uv run latent_explorer.py --wandb-artifact <entity/project/name:version>
```

For a quick demo with an 8-dimensional model trained on landscape imagery:

```bash
uv run latent_explorer.py --wandb-artifact eitreif/sphere-flow/8dim_geir:latest
```

This will:

1. Download the artifact from wandb
2. Start a local web server (default: http://127.0.0.1:7860)
3. Open your browser with N sliders (one per latent dimension)

**Controls:**

- Drag sliders (-1 to 1) to adjust the latent vector
- The raw latent magnitude is shown before spherification
- **Randomize** — sample random values for all dimensions
- **Reset** — set all sliders to 0

The latent vector is spherified (normalized to unit sphere) before decoding, following the model's training regime.

### Options

```bash
uv run latent_explorer.py \
    --wandb-artifact eitreif/sphere-flow/8dim_geir:latest \
    --host 0.0.0.0 \
    --port 8080 \
    --device cuda
```

- `--host` — server bind address (default: 127.0.0.1)
- `--port` — server port (default: 7860)
- `--device` — torch device: cuda, mps, or cpu (default: auto-detect)
