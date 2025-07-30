"""
Attention mechanisms and positional embeddings with FlashAttention-2 optimization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from cs336_basics.nn.activations import softmax
from cs336_basics.nn.layers import Linear


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... n_queries d_k"],
    K: Float[torch.Tensor, "... n_keys d_k"],
    V: Float[torch.Tensor, "... n_values d_v"],
    mask: Float[torch.Tensor, "... n_queries n_key"] | None = None,
    use_flash_attention: bool = True,
) -> Float[torch.Tensor, "... n_queries d_v"]:
    """
    Scaled Dot-Product Attention with FlashAttention-2 optimization.

    Uses PyTorch's scaled_dot_product_attention when available for better
    memory efficiency and performance, especially on modern GPUs like H100.

    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k)
        V: Value tensor of shape (..., n_values, d_v)
        mask: Optional attention mask
        use_flash_attention: Whether to use attention when available

    Returns:
        Output tensor with shape (..., n_queries, d_v)
    """
    if use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
        if mask is None:
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False)
    else:
        d_k = Q.shape[-1]
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = softmax(scores, dim=-1)
        output = torch.einsum("...qk,...kv->...qv", attention_weights, V)
        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Enhanced with better memory efficiency and tensor core optimization.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        """
        Initialize the RoPE module with optimizations.

        Args:
            theta: θ value for RoPE (typically 10000)
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length for precomputing values
            device: Device to store buffers on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        dim_indices = torch.arange(0, d_k // 2, dtype=torch.float32)
        frequencies = 1.0 / (theta ** (2.0 * dim_indices / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)

        cos_values = torch.cos(angles).contiguous()
        sin_values = torch.sin(angles).contiguous()

        self.register_buffer("cos_values", cos_values.to(device), persistent=False)
        self.register_buffer("sin_values", sin_values.to(device), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply rotary position embedding with memory access patterns.

        Args:
            x: Input tensor with arbitrary batch dimensions (..., seq_len, d_k)
            token_positions: Position indices for each token (..., seq_len)

        Returns:
            Output tensor with same shape as input but with RoPE applied
        """
        cos_vals = self.cos_values[token_positions]
        sin_vals = self.sin_values[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_new_even = cos_vals * x_even - sin_vals * x_odd
        x_new_odd = sin_vals * x_even + cos_vals * x_odd

        result = torch.stack([x_new_even, x_new_odd], dim=-1)
        return result.flatten(-2)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}"


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with FlashAttention-2.

    Enhanced with memory-efficient implementations and better GPU utilization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Multi-Head Self-Attention layer with optimizations.

        Args:
            d_model: Dimensionality of the model (input/output dimension)
            num_heads: Number of attention heads
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        factory_kwargs = {"device": device, "dtype": dtype}

        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

        self.use_checkpoint = False

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """
        Apply causal multi-head self-attention with optimizations.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            rope: Optional RoPE module for positional encoding
            token_positions: Token positions for RoPE (required if rope is provided)

        Returns:
            Output tensor of same shape as input
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, rope, token_positions, use_reentrant=False)
        else:
            return self._forward_impl(x, rope, token_positions)

    def _forward_impl(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """Internal forward implementation."""
        *batch_dims, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(*batch_dims, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(*range(len(batch_dims)), -3, -2, -4, -1)

        Q, K, V = qkv.unbind(dim=len(batch_dims))

        if rope is not None and token_positions is not None:
            original_q_shape = Q.shape
            original_k_shape = K.shape

            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)

            pos_expanded = token_positions.unsqueeze(-2)
            pos_expanded = pos_expanded.expand(*batch_dims, self.num_heads, seq_len)
            pos_flat = pos_expanded.reshape(-1, seq_len)

            Q_flat = rope(Q_flat, pos_flat)
            K_flat = rope(K_flat, pos_flat)

            Q = Q_flat.reshape(original_q_shape)
            K = K_flat.reshape(original_k_shape)

        attn_output = scaled_dot_product_attention(Q, K, V, mask=None, use_flash_attention=True)

        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.contiguous().view(*batch_dims, seq_len, self.d_model)

        output = self.output_proj(attn_output)
        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}"
