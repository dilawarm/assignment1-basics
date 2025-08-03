"""Attention module with native PyTorch FP8 support (no Transformer Engine)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_func

from .components import RotaryPositionEmbedding, scaled_init_
from .fp8_linear import FP8Linear


class MultiHeadAttentionNativeFP8(nn.Module):
    """Multi-head attention with native PyTorch FP8 support."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash: bool = True,
        use_fp8: bool = True,
        layer_idx: int = 0,
        total_layers: int = 24,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout
        self.use_flash = use_flash
        self.use_fp8 = use_fp8
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        # Check if FP8 is available
        if self.use_fp8:
            try:
                # Test FP8 dtypes
                _ = torch.float8_e4m3fn
                _ = torch.float8_e5m2
                # Test scaled_mm
                test_tensor = torch.randn(2, 2, device="cuda", dtype=torch.float8_e4m3fn)
                scale = torch.tensor(1.0, device="cuda")
                _ = torch._scaled_mm(test_tensor, test_tensor, scale_a=scale, scale_b=scale)
            except (AttributeError, RuntimeError, AssertionError) as e:
                print(f"Warning: Native FP8 not available ({e}). Using standard layers.")
                self.use_fp8 = False

        # Projections - use FP8Linear if requested and available
        if self.use_fp8 and torch.cuda.is_available():
            self.qkv_proj = FP8Linear(dim, 3 * n_heads * head_dim, bias=bias)
            self.o_proj = FP8Linear(n_heads * head_dim, dim, bias=bias)
        else:
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
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional mask tensor
            use_cache: Whether to return key/value for caching
            past_key_value: Optional cached key/value tensors

        Returns:
            Output tensor and optionally cached key/value
        """
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

            # Flash attention with FP8 inputs if enabled
            if self.use_fp8 and q.device.type == "cuda":
                # Flash Attention can work with FP8 inputs on H100
                # Note: Flash Attention internally handles mixed precision
                attn_output = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=True,
                )
            else:
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
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention (fallback)."""
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
