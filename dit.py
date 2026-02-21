import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


def init_param(*shape) -> nn.Parameter:
    u = torch.zeros(*shape)
    if len(list(shape)) > 1:
        torch.nn.init.xavier_normal_(u)
    else:
        torch.nn.init.normal_(u)
    return nn.Parameter(u)


def SDPA(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Scaled dot-product attention using Flash Attention."""
    out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return rearrange(out, 'b h s v -> b s (h v)')


def precompute_rope_freqs(head_dim: int, max_seq: int = 4096) -> Tensor:
    assert head_dim % 2 == 0
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq).float(), theta)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    # x: (b, h, t, d)
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs[:x.shape[2]]).flatten(3)
    return out.type_as(x)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning and QK-Norm."""

    def __init__(
        self,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.query = nn.Linear(embed_dim, query_dim * heads, bias=False)
        self.key = nn.Linear(embed_dim, query_dim * heads, bias=False)
        self.value = nn.Linear(embed_dim, value_dim * heads, bias=False)
        self.output = nn.Linear(value_dim * heads, embed_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.h = heads
        self.scale = 1 / math.sqrt(query_dim)

        # QK-Norm: normalize queries and keys to prevent attention explosion
        self.q_norm = RMSNorm(query_dim)
        self.k_norm = RMSNorm(query_dim)

        self.register_buffer("rope_freqs", precompute_rope_freqs(query_dim))

        # adaLN-Zero: 6 modulation params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True),
        )

        # Zero-initialize the modulation output so block starts as identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, inputs: Tensor, c: Tensor) -> Tensor:
        """
        inputs: (b, t, e) - sequence of tokens
        c: (b, e) - conditioning vector (timestep embedding)
        """
        # Get modulation parameters
        mod = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Attention with adaLN modulation and QK-Norm
        attn_normed = modulate(self.attn_norm(inputs), shift1, scale1)
        q = rearrange(self.query(attn_normed), 'b t (h p) -> b h t p', h=self.h)
        k = rearrange(self.key(attn_normed), 'b t (h p) -> b h t p', h=self.h)
        v = rearrange(self.value(attn_normed), 'b t (h p) -> b h t p', h=self.h)
        # Apply QK-Norm to bound attention logits
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)
        scores = SDPA(q, k, v, self.scale)
        attn_out = self.output(scores)
        # Apply gate (zero-initialized, so starts as identity)
        inputs = inputs + gate1.unsqueeze(1) * attn_out

        # MLP with adaLN modulation
        ffn_normed = modulate(self.ffn_norm(inputs), shift2, scale2)
        ffn_out = self.fc2(F.silu(self.fc1(ffn_normed)))
        return inputs + gate2.unsqueeze(1) * ffn_out


class Tokenizer(nn.Module):
    """Patchify and unpatchify images for DiT."""

    def __init__(self, img_chw: tuple[int, int, int], patch_size: int, embed_dim: int) -> None:
        super().__init__()
        num_patches = (img_chw[1] // patch_size) * (img_chw[2] // patch_size)
        params = {'kernel_size': (patch_size, patch_size), 'stride': patch_size}
        self.unfold = nn.Unfold(**params)
        self.fold = nn.Fold(output_size=img_chw[1:], **params)
        patch_dim = patch_size * patch_size * img_chw[0]
        self.to_tokens = nn.Linear(patch_dim, embed_dim, bias=False)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.from_tokens = nn.Linear(embed_dim, patch_dim, bias=False)
        self.positional = init_param(1, num_patches, embed_dim)
        # Smooth patch boundaries with ResNet block
        self.smooth = nn.Conv2d(img_chw[0], img_chw[0], 3, padding=1)

        # Zero-initialize output projection for stable residual learning
        # nn.init.zeros_(self.from_tokens.weight)

    def num_params(self) -> int:
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total

    def tokenize(self, inputs: Tensor) -> Tensor:
        patches = self.unfold(inputs)
        patches = rearrange(patches, 'b p t -> b t p')
        tokens = self.to_tokens(patches)
        return tokens

    def detokenize(self, tokens: Tensor) -> Tensor:
        # should only be given the tokens that are actual image tokens.
        tokens = self.out_norm(tokens)
        patches = self.from_tokens(tokens)
        patches = rearrange(patches, 'b t e -> b e t')
        images = self.fold(patches)
        images = self.smooth(images)
        return images

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = self.tokenize(inputs)
        tokens = tokens + self.positional
        return tokens


class DiT(nn.Module):
    """
    Diffusion Transformer with adaLN-Zero conditioning.

    Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
    """

    def __init__(
        self,
        steps: int,
        num_layers: int,
        heads: int,
        embed_dim: int,
        query_dim: int,
        value_dim: int,
        ffn_dim: int,
        img_chw: tuple[int, int, int],
        patch_size: int,
        ratio: float = 0.75,
        context_dim: int | None = None,
        toks_dim: int | None = None,
        flow: bool = False,
        **kwargs,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DiTBlock(heads, embed_dim, query_dim, value_dim, ffn_dim))
        self.layers = nn.ModuleList(layers)

        # Timestep embedding with MLP (like official DiT)
        self.time_embeds = nn.Embedding(steps, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.steps = steps
        self.ratio = ratio
        self.flow = flow
        self.tokenizer = Tokenizer(img_chw, patch_size, embed_dim)

        # Final layer with adaLN-Zero
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True),
        )

        self.context_project = nn.Linear(context_dim, embed_dim) if context_dim else None
        self.toks_project = nn.Linear(toks_dim, embed_dim) if toks_dim else None

        # Zero-initialize final layer
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def tokenize(self, x: Tensor) -> Tensor:
        return self.tokenizer(x)

    def forward(self, x_t: Tensor, x_i_toks: Tensor | None, c: Tensor, steps: Tensor) -> Tensor:
        """
        x_t: noisy image at step t
        x_i_toks: initial decoder tokens
        c: conditioning tokens from encoder + prior
        steps: diffusion step indices

        returns the predicted noise.
        """
        # Get timestep conditioning for adaLN
        if self.flow:
            steps = (steps * self.steps).long()
        t_emb = self.time_embeds(steps)
        t_emb = self.time_mlp(t_emb)  # (b, embed_dim)

        # Tokenize noisy image
        toks = self.tokenizer(x_t)

        # Concatenate all tokens for self-attention
        if self.context_project is not None:
            c = self.context_project(c)
        if self.toks_project is not None and x_i_toks is not None:
            x_i_toks = self.toks_project(x_i_toks)

        if x_i_toks is not None:
            h = torch.cat([c, x_i_toks, toks], dim=-2)
        else:
            h = torch.cat([c, toks], dim=-2)

        # Apply DiT blocks with adaLN-Zero conditioning
        for layer in self.layers:
            h = layer(h, t_emb)

        # Extract only the image tokens
        offset = c.shape[1] + (x_i_toks.shape[1] if x_i_toks is not None else 0)
        h = h[:, offset:]

        # Final layer with adaLN modulation
        final_mod = self.final_adaLN(t_emb)
        shift, scale = final_mod.chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)

        # Detokenize to image
        eps = self.tokenizer.detokenize(h)
        return eps
