"""Attention module without Transformer Engine dependency."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func

    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Warning: Flash Attention not available. Using standard attention.")

from .components import RotaryPositionEmbedding, scaled_init_


class MultiHeadAttentionNoTE(nn.Module):
    """Multi-head attention without Transformer Engine dependency."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash: bool = True,
        layer_idx: int = 0,
        total_layers: int = 24,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout
        self.use_flash = use_flash and FLASH_AVAILABLE
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        # Always use standard PyTorch Linear layers
        self.qkv_proj = nn.Linear(dim, 3 * n_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        # Rotary embeddings
        self.rope = RotaryPositionEmbedding(head_dim)

        # Initialize weights with proper scaling
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with depth-dependent scaling."""
        # Standard initialization for QKV
        std = 0.02
        if hasattr(self.qkv_proj, "weight"):
            scaled_init_(self.qkv_proj.weight, std)

        # Scaled initialization for output projection
        output_std = std / math.sqrt(2 * self.total_layers)
        if hasattr(self.o_proj, "weight"):
            scaled_init_(self.o_proj.weight, output_std)
            # Zero-initialize output projection for better stability
            with torch.no_grad():
                self.o_proj.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for multi-head attention."""
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)

        # Reshape to separate Q, K, V
        qkv = rearrange(qkv, "b s (three h d) -> three b s h d", three=3, h=self.n_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        q, k = self.rope(q, k, seq_dim=1)

        # Handle caching for inference
        if use_cache:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Apply attention
        if self.use_flash:
            # Use Flash Attention
            q = rearrange(q, "b s h d -> b s h d")
            k = rearrange(k, "b s h d -> b s h d")
            v = rearrange(v, "b s h d -> b s h d")

            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
            )

            attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        else:
            # Standard attention as fallback
            attn_output = self._standard_attention(q, k, v, attention_mask)
            attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

        # Output projection
        output = self.o_proj(attn_output)

        return output, present_key_value

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Compute attention scores
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale

        # Apply causal mask
        seq_len = q.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask[None, None, :, :], float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        return attn_output
