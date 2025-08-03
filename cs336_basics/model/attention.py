"""Attention module with Flash Attention 3 and FP8 support."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from einops import rearrange
from flash_attn import flash_attn_func

from .components import RotaryPositionEmbedding, scaled_init_


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention 3 and FP8 support."""

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

        # Projections - use Transformer Engine layers for FP8 support
        if self.use_fp8:
            try:
                # Transformer Engine layers for FP8 support
                self.qkv_proj = te.Linear(
                    dim,
                    3 * n_heads * head_dim,
                    bias=bias,
                )
                self.o_proj = te.Linear(
                    n_heads * head_dim,
                    dim,
                    bias=bias,
                )
            except Exception as e:
                print(f"Warning: Failed to create Transformer Engine layers: {e}")
                print("Falling back to standard PyTorch layers")
                self.use_fp8 = False
                self.qkv_proj = nn.Linear(dim, 3 * n_heads * head_dim, bias=bias)
                self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)
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

        # Ensure input is contiguous for Transformer Engine
        if self.use_fp8 and not x.is_contiguous():
            x = x.contiguous()

        # QKV projection with error handling for FP8
        try:
            qkv = self.qkv_proj(x)
        except RuntimeError as e:
            if self.use_fp8 and "cuBLAS" in str(e):
                print(f"Warning: Transformer Engine error: {e}")
                print("Falling back to FP32 computation for this layer")
                # Convert to standard linear temporarily
                with torch.no_grad():
                    weight = self.qkv_proj.weight.data.float()
                    bias = self.qkv_proj.bias.data.float() if self.qkv_proj.bias is not None else None
                qkv = F.linear(x.float(), weight, bias)
            else:
                raise

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
            # Use Flash Attention 3
            # Reshape for flash_attn_func: (batch, seq_len, n_heads, head_dim)
            q = rearrange(q, "b s h d -> b s h d")
            k = rearrange(k, "b s h d -> b s h d")
            v = rearrange(v, "b s h d -> b s h d")

            # Flash attention expects dropout probability, not rate
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,  # Always use causal mask for autoregressive LM
            )

            # Reshape back
            attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        else:
            # Standard attention as fallback
            attn_output = self._standard_attention(q, k, v, attention_mask)
            attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

        # Output projection with error handling for FP8
        try:
            output = self.o_proj(attn_output)
        except RuntimeError as e:
            if self.use_fp8 and "cuBLAS" in str(e):
                print(f"Warning: Transformer Engine error in output projection: {e}")
                print("Falling back to FP32 computation")
                with torch.no_grad():
                    weight = self.o_proj.weight.data.float()
                    bias = self.o_proj.bias.data.float() if self.o_proj.bias is not None else None
                output = F.linear(attn_output.float(), weight, bias)
            else:
                raise

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
        if attention_mask is None:
            seq_len = q.shape[1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal_mask[None, None, :, :], float("-inf"))
        else:
            scores.masked_fill_(attention_mask[None, None, :, :], float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        return attn_output
