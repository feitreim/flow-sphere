import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def SDPA(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Scaled dot-product attention using Flash Attention."""
    out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return rearrange(out, "b h s v -> b s (h v)")


def precompute_rope_freqs(head_dim: int, max_seq: int = 4096) -> Tensor:
    assert head_dim % 2 == 0
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq).float(), theta)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    # x: (b, h, t, d)
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs[: x.shape[2]]).flatten(3)
    return out.type_as(x)


def make_sinusoidal_pos_emb(num_tokens: int, dim: int) -> Tensor:
    assert dim % 2 == 0
    pos = torch.arange(num_tokens, dtype=torch.float32)
    i = torch.arange(dim // 2, dtype=torch.float32)
    angles = pos[:, None] / (10000.0 ** (2 * i[None, :] / dim))
    return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (num_tokens, dim)


def init_param(*shape) -> nn.Parameter:
    u = torch.zeros(*shape)
    if len(list(shape)) > 1:
        torch.nn.init.xavier_normal_(u)
    else:
        torch.nn.init.normal_(u)
    return nn.Parameter(u)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_fc1 = nn.Linear(num_tokens, num_tokens * 4)
        self.token_fc2 = nn.Linear(num_tokens * 4, num_tokens)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.channel_fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # token mixing
        h = self.norm1(x).transpose(1, 2)  # (b, dim, tokens)
        h = self.token_fc2(F.gelu(self.token_fc1(h))).transpose(1, 2)
        x = x + h
        # channel mixing
        h = self.norm2(x)
        h = self.channel_fc2(F.gelu(self.channel_fc1(h)))
        return x + h


class MLPMixer(nn.Module):
    """4-layer MLP-Mixer with final RMSNorm, inserted at encoder end and decoder start."""

    def __init__(self, num_tokens: int, embed_dim: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([MLPMixerLayer(num_tokens, embed_dim) for _ in range(num_layers)])
        self.norm = RMSNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ViTLayer(nn.Module):
    def __init__(
        self,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, query_dim * heads, bias=False)
        self.key = nn.Linear(embed_dim, query_dim * heads, bias=False)
        self.value = nn.Linear(embed_dim, value_dim * heads, bias=False)
        self.output = nn.Linear(value_dim * heads, embed_dim, bias=False)
        self.ffn_norm = RMSNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.h = heads
        self.scale = 1 / math.sqrt(query_dim)

        # qk norm
        self.q_norm = RMSNorm(query_dim)
        self.k_norm = RMSNorm(query_dim)

        self.register_buffer("rope_freqs", precompute_rope_freqs(query_dim))

        # zero init b4 residual
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: b t e
        # attention:
        attn_normed = self.attn_norm(inputs)
        q = rearrange(self.query(attn_normed), "b t (h p) -> b h t p", h=self.h)
        q = self.q_norm(q)
        k = rearrange(self.key(attn_normed), "b t (h p) -> b h t p", h=self.h)
        k = self.k_norm(k)
        v = rearrange(self.value(attn_normed), "b t (h p) -> b h t p", h=self.h)
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)
        scores = SDPA(q, k, v, self.scale)
        attn_out = self.output(scores)
        inputs = inputs + attn_out
        # MLP:
        h = self.ffn_norm(inputs)
        residual = self.fc2(F.silu(self.fc1(h)))
        return inputs + residual


class Tokenizer(nn.Module):
    """Patchify and unpatchify images for DiT."""

    def __init__(
        self, img_chw: tuple[int, int, int], patch_size: int, embed_dim: int, bottleneck_dim: int = 128
    ) -> None:
        super().__init__()
        num_patches = (img_chw[1] // patch_size) * (img_chw[2] // patch_size)
        self.num_patches = num_patches
        params = {"kernel_size": (patch_size, patch_size), "stride": patch_size}
        self.unfold = nn.Unfold(**params)
        self.fold = nn.Fold(output_size=img_chw[1:], **params)
        patch_dim = patch_size * patch_size * img_chw[0]
        self.bottleneck = nn.Linear(patch_dim, bottleneck_dim, bias=False)
        self.to_tokens = nn.Linear(bottleneck_dim, embed_dim, bias=False)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.from_tokens = nn.Linear(embed_dim, patch_dim, bias=False)
        self.register_buffer("positional", make_sinusoidal_pos_emb(num_patches, embed_dim).unsqueeze(0))
        # Smooth patch boundaries with ResNet block
        self.smooth = nn.Conv2d(img_chw[0], img_chw[0], 3, padding=1)

    def num_params(self) -> int:
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total

    def tokenize(self, inputs: Tensor) -> Tensor:
        patches = self.unfold(inputs)
        patches = rearrange(patches, "b p t -> b t p")
        patches = self.bottleneck(patches)
        tokens = self.to_tokens(patches)
        return tokens

    def detokenize(self, tokens: Tensor) -> Tensor:
        # should only be given the tokens that are actual image tokens.
        assert tokens.shape[1] == self.num_patches
        tokens = self.out_norm(tokens)
        patches = self.from_tokens(tokens)
        patches = rearrange(patches, "b t e -> b e t")
        images = self.fold(patches)
        images = self.smooth(images)
        return images

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = self.tokenize(inputs)
        tokens = tokens + self.positional
        return tokens


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_tokens: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
        compression_factor: int = 4,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ViTLayer(heads, embed_dim, query_dim, value_dim, ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("positional_enc", make_sinusoidal_pos_emb(input_tokens, embed_dim))
        self.mixer = MLPMixer(input_tokens, embed_dim)
        self.bottleneck = nn.Linear(embed_dim, embed_dim // compression_factor, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = x + self.positional_enc[None, ...]
        for layer in self.layers:
            h = layer(h)
        h = self.mixer(h)
        return self.bottleneck(h)


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
        compression_factor: int = 4,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ViTLayer(heads, embed_dim, query_dim, value_dim, ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.un_bottleneck = nn.Linear(embed_dim // compression_factor, embed_dim, bias=False)
        self.mixer = MLPMixer(num_tokens, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.un_bottleneck(x)
        h = self.mixer(h)
        for layer in self.layers:
            h = layer(h)
        return h


class AE(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
        img_chw: tuple[int, int, int],
        patch_size: int,
        compression_factor: int = 4,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers,
            num_tokens,
            heads,
            embed_dim,
            query_dim,
            value_dim,
            ffn_dim,
            compression_factor,
        )
        self.decoder = Decoder(
            num_layers,
            num_tokens,
            heads,
            embed_dim,
            query_dim,
            value_dim,
            ffn_dim,
            compression_factor,
        )
        self.tokenizer = Tokenizer(img_chw, patch_size, embed_dim)

    @torch.compile()
    def forward(self, x: Tensor) -> Tensor:
        h = self.tokenizer(x)
        h = self.encoder(h)
        h = self.decoder(h)
        h = self.tokenizer.detokenize(h)
        return h

    def num_params(self) -> int:
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total
