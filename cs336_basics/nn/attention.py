"""
Attention mechanisms for Transformer models.

This module implements various attention mechanisms including scaled dot-product attention,
multi-head self-attention, and rotary positional embedding (RoPE).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Optional[Float[Tensor, "... queries keys"]] = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., values, d_v)
        mask: Optional mask tensor of shape (..., queries, keys)

    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.

    RoPE applies rotation to query and key embeddings based on their position,
    enabling the model to understand relative positions naturally.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize RoPE.

        Args:
            theta: Base frequency for RoPE
            d_k: Dimension of the key/query embeddings
            max_seq_len: Maximum sequence length
            device: Device to place tensors on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        if device is not None:
            inv_freq = inv_freq.to(device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        position = torch.arange(max_seq_len, dtype=torch.float32)
        if device is not None:
            position = position.to(device)

        freqs = torch.outer(position, inv_freq)
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        self.register_buffer("cos_freqs", cos_freqs, persistent=False)
        self.register_buffer("sin_freqs", sin_freqs, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape (..., sequence_length, d_k)
            token_positions: Position indices of shape (..., sequence_length)

        Returns:
            Rotated tensor of shape (..., sequence_length, d_k)
        """
        cos = self.cos_freqs[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_freqs[token_positions]  # (..., seq_len, d_k//2)

        x_even = x[..., 0::2]  # (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k//2)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        output = torch.stack([rotated_even, rotated_odd], dim=-1)
        output = output.view_as(x)

        return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with optional RoPE support.

    This implementation uses a fused QKV projection for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize Multi-Head Self-Attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)

        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        rope: Optional[RotaryPositionalEmbedding] = None,
        token_positions: Optional[Int[Tensor, "... sequence_length"]] = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (..., sequence_length, d_model)
            rope: Optional RoPE module for positional encoding
            token_positions: Optional position indices for RoPE

        Returns:
            Output tensor of shape (..., sequence_length, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        qkv = self.qkv_proj(x)  # (..., seq_len, 3 * d_model)

        qkv = qkv.view(*batch_shape, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(*range(len(batch_shape)), -3, -2, -4, -1)  # (..., 3, num_heads, seq_len, d_k)

        q, k, v = qkv.unbind(dim=-4)  # Each: (..., num_heads, seq_len, d_k)

        if rope is not None and token_positions is not None:
            q_rope = []
            k_rope = []
            for head in range(self.num_heads):
                q_head = rope(q[..., head, :, :], token_positions)
                k_head = rope(k[..., head, :, :], token_positions)
                q_rope.append(q_head)
                k_rope.append(k_head)

            q = torch.stack(q_rope, dim=-3)  # (..., num_heads, seq_len, d_k)
            k = torch.stack(k_rope, dim=-3)  # (..., num_heads, seq_len, d_k)

        attn_output = scaled_dot_product_attention(q, k, v)  # (..., num_heads, seq_len, d_k)

        attn_output = attn_output.permute(*range(len(batch_shape)), -2, -3, -1)  # (..., seq_len, num_heads, d_k)
        attn_output = attn_output.contiguous().view(*batch_shape, seq_len, self.d_model)

        output = self.output_proj(attn_output)

        return output
