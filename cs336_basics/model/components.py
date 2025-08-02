"""Core components for the transformer model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm, with equivalent performance.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        # Normalize and scale
        return x / (norm + self.eps) * self.weight


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
        # SwiGLU: (Swish(xW1) * xW2)W3
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

        # Precompute frequencies (only for half the dimensions)
        # RoPE is applied to pairs of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for sin/cos values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cache for sin/cos values if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Create position indices
            pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

            # Compute frequencies (shape: [seq_len, dim/2])
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq)

            # Cache sin/cos (shape: [seq_len, dim/2])
            self._cos_cached = freqs.cos().to(dtype)
            self._sin_cached = freqs.sin().to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Get cached sin/cos values
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]

        # Apply rotation
        q_rot = self._apply_rotation(q, cos, sin, seq_dim)
        k_rot = self._apply_rotation(k, cos, sin, seq_dim)

        return q_rot, k_rot

    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dim: int) -> torch.Tensor:
        """Apply the rotation to a tensor."""
        # Reshape for rotation
        if seq_dim == 1:
            # x shape: (..., seq_len, n_heads, head_dim)
            # cos/sin shape: (seq_len, dim/2)
            cos = cos[:, None, :]  # (seq_len, 1, dim/2)
            sin = sin[:, None, :]
        elif seq_dim == 0:
            # x shape: (seq_len, ..., n_heads, head_dim)
            cos = cos[:, None, None, :]  # (seq_len, 1, 1, dim/2)
            sin = sin[:, None, None, :]

        # Split tensor for rotation
        # x1, x2 shape: (..., seq_len, n_heads, head_dim/2)
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotation: [cos, -sin; sin, cos] @ [x1; x2]
        # Resulting shapes: (..., seq_len, n_heads, head_dim/2) each
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
