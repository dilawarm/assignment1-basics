"""Core components for the transformer model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm, with equivalent performance.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_sq = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_sq + self.eps)
        return x_normed * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    More efficient than GELU/ReLU for transformers.
    """

    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w3 = nn.Linear(dim_hidden, dim_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Better sequence modeling than learned position embeddings.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cache for sin/cos values if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

            inv_freq = self.inv_freq.to(device)
            freqs = torch.einsum("i,j->ij", pos, inv_freq)

            self._cos_cached = freqs.cos().to(dtype)
            self._sin_cached = freqs.sin().to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (..., seq_len, n_heads, head_dim)
            k: Key tensor of shape (..., seq_len, n_heads, head_dim)
            seq_dim: Sequence dimension (default: 1)

        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = q.shape[seq_dim]
        self._update_cache(seq_len, q.device, q.dtype)

        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]

        q_rot = self._apply_rotation(q, cos, sin, seq_dim)
        k_rot = self._apply_rotation(k, cos, sin, seq_dim)

        return q_rot, k_rot

    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dim: int) -> torch.Tensor:
        """Apply the rotation to a tensor."""
        if seq_dim == 1:
            cos = cos[:, None, :]
            sin = sin[:, None, :]
        elif seq_dim == 0:
            cos = cos[:, None, None, :]
            sin = sin[:, None, None, :]

        x1, x2 = x.chunk(2, dim=-1)

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def scaled_init_(tensor: torch.Tensor, scale: float = 1.0, mode: str = "fan_in"):
    """Initialize tensor with scaled normal distribution."""
    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[0] if len(tensor.shape) >= 2 else 1

    if mode == "fan_in":
        scale = scale / math.sqrt(fan_in)
    elif mode == "fan_out":
        scale = scale / math.sqrt(fan_out)
    elif mode == "fan_avg":
        scale = scale / math.sqrt((fan_in + fan_out) / 2)

    with torch.no_grad():
        tensor.normal_(0, scale)

    return tensor
