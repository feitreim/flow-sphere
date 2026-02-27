# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.5",
#     "torchvision>=0.20",
#     "einops>=0.7",
#     "wandb>=0.19,<0.25",
#     "gradio>=5.0",
#     "torchmetrics",
# ]
# ///

import argparse
import dataclasses
import json
from pathlib import Path

import gradio as gr
import torch
from torch import Tensor

from tiny_ae import Config, TinyAE, spherify


def load_from_wandb(artifact_path: str, device: str) -> tuple[dict, str]:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    local_dir = artifact.download()

    # Match visualize.py behavior: infer files by lightweight type sniffing.
    sd_path, config_path = None, None
    for p in Path(local_dir).iterdir():
        if not p.is_file():
            continue
        header = p.read_bytes()[:4]
        if header[:2] == b"PK":
            sd_path = p
        elif header[:1] == b"{":
            config_path = p

    assert sd_path is not None, f"No model weights found in {local_dir}"
    sd = torch.load(str(sd_path), map_location=device, weights_only=False)
    return sd, str(config_path) if config_path else local_dir


def _load_config_from_artifact(config_path: str) -> Config:
    candidate = Path(config_path)

    # load_from_wandb may return either a json file path or a downloaded dir.
    if candidate.is_file():
        json_candidates = [candidate]
    elif candidate.is_dir():
        json_candidates = sorted(p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    else:
        json_candidates = []

    if not json_candidates:
        return Config()

    with open(json_candidates[0]) as f:
        d = json.load(f)

    valid = {field.name for field in dataclasses.fields(Config)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return Config(**filtered)


def _extract_model_state_dict(sd: dict) -> dict:
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd


def _to_display_image(img: Tensor):
    # img expected as (1, C, H, W) in [-1, 1]
    img = img[0].detach().cpu().clamp(-1, 1)
    img = ((img + 1.0) * 127.5).round().to(torch.uint8)
    return img.permute(1, 2, 0).numpy()


def make_ui(model: TinyAE, device: str) -> gr.Blocks:
    model.eval()
    latent_dim = model.N

    @torch.inference_mode()
    def decode_from_latent(*vals):
        z_raw = torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)
        raw_mag = float(z_raw.norm(dim=-1).item())
        z = spherify(z_raw)
        decoded = model.decode(z).float()
        return _to_display_image(decoded), f"||z_raw||_2 = {raw_mag:.4f}"

    def randomize_sliders():
        vals = torch.empty(latent_dim).uniform_(-1, 1).tolist()
        return vals

    def reset_sliders():
        return [0.0] * latent_dim

    with gr.Blocks(title="TinyAE Latent Explorer") as demo:
        gr.Markdown("# TinyAE Latent Explorer")

        with gr.Row():
            with gr.Column(scale=1):
                sliders = [
                    gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.01, label=f"z[{i}]")
                    for i in range(latent_dim)
                ]
                with gr.Row():
                    random_btn = gr.Button("Randomize")
                    reset_btn = gr.Button("Reset")
                mag_text = gr.Textbox(label="Raw Latent Magnitude", interactive=False)
            with gr.Column(scale=1):
                out_img = gr.Image(label="Decoded Image", type="numpy")

        for s in sliders:
            s.change(fn=decode_from_latent, inputs=sliders, outputs=[out_img, mag_text], queue=False)

        random_btn.click(fn=randomize_sliders, outputs=sliders, queue=False).then(
            fn=decode_from_latent,
            inputs=sliders,
            outputs=[out_img, mag_text],
            queue=False,
        )
        reset_btn.click(fn=reset_sliders, outputs=sliders, queue=False).then(
            fn=decode_from_latent,
            inputs=sliders,
            outputs=[out_img, mag_text],
            queue=False,
        )

        demo.load(fn=decode_from_latent, inputs=sliders, outputs=[out_img, mag_text], queue=False)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Explore TinyAE latent space with Gradio")
    parser.add_argument("--wandb-artifact", required=True, help="Artifact path like entity/project/name:version")
    parser.add_argument("--device", default=None, help="torch device (default: cuda/mps/cpu auto)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    sd, config_path = load_from_wandb(args.wandb_artifact, device)
    cfg = _load_config_from_artifact(config_path)

    model = TinyAE(cfg).to(device)
    model.load_state_dict(_extract_model_state_dict(sd), strict=True)
    model.eval()

    print(
        f"[latent_explorer] loaded artifact={args.wandb_artifact} | latent_dim={cfg.latent_dim} | device={device}"
    )

    ui = make_ui(model, device)
    ui.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
