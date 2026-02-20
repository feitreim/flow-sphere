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
        self, img_chw: tuple[int, int, int], patch_size: int, embed_dim: int
    ) -> None:
        super().__init__()
        num_patches = (img_chw[1] // patch_size) * (img_chw[2] // patch_size)
        self.num_patches = num_patches
        params = {"kernel_size": (patch_size, patch_size), "stride": patch_size}
        self.unfold = nn.Unfold(**params)
        self.fold = nn.Fold(output_size=img_chw[1:], **params)
        patch_dim = patch_size * patch_size * img_chw[0]
        self.to_tokens = nn.Linear(patch_dim, embed_dim, bias=False)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.from_tokens = nn.Linear(embed_dim, patch_dim, bias=False)
        self.positional = init_param(1, num_patches, embed_dim)
        # Smooth patch boundaries with ResNet block
        # self.smooth = nn.Conv2d(img_chw[0], img_chw[0], 3, padding=1)

    def num_params(self) -> int:
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total

    def tokenize(self, inputs: Tensor) -> Tensor:
        patches = self.unfold(inputs)
        patches = rearrange(patches, "b p t -> b t p")
        tokens = self.to_tokens(patches)
        return tokens

    def detokenize(self, tokens: Tensor) -> Tensor:
        # should only be given the tokens that are actual image tokens.
        assert tokens.shape[1] == self.num_patches
        tokens = self.out_norm(tokens)
        patches = self.from_tokens(tokens)
        patches = rearrange(patches, "b t e -> b e t")
        images = self.fold(patches)
        # images = self.smooth(images)
        return images

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = self.tokenize(inputs)
        tokens = tokens + self.positional
        return tokens


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        latent_tokens: int,
        input_tokens: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ViTLayer(heads, embed_dim, query_dim, value_dim, ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.latent_tokens = init_param(latent_tokens, embed_dim)
        self.positional_enc = init_param(input_tokens, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq = x.shape[:2]
        latent_tokens = self.latent_tokens.expand(bsz, -1, -1)
        pos_enc = self.positional_enc.expand(bsz, -1, -1)
        x = x + pos_enc
        h = torch.cat([x, latent_tokens], dim=-2)
        for layer in self.layers:
            h = layer(h)
        return h[:, seq:]


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        output_tokens: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ViTLayer(heads, embed_dim, query_dim, value_dim, ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.output_tokens = init_param(output_tokens, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq = x.shape[:2]
        output_tokens = self.output_tokens.expand(bsz, -1, -1)
        h = torch.cat([x, output_tokens], dim=-2)
        for layer in self.layers:
            h = layer(h)
        return h[:, seq:]


class AE(nn.Module):
    def __init__(
        self,
        num_layers: int,
        latent_tokens: int,
        output_tokens: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
        img_chw: tuple[int, int, int],
        patch_size: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers,
            latent_tokens,
            output_tokens,
            heads,
            embed_dim,
            query_dim,
            value_dim,
            ffn_dim,
        )
        self.decoder = Decoder(
            num_layers,
            output_tokens,
            heads,
            embed_dim,
            query_dim,
            value_dim,
            ffn_dim,
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
