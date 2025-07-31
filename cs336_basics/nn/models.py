"""
Complete Transformer model implementations with performance optimizations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.nn.activations import SwiGLU
from cs336_basics.nn.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.layers import Embedding, Linear, RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with gradient checkpointing support.

    Enhanced with memory efficiency improvements and performance optimizations
    for modern GPU architectures.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Transformer block with optimizations.

        Args:
            d_model: Dimensionality of the model (input/output dimension)
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            eps: Epsilon value for RMSNorm numerical stability
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.attn = MultiHeadSelfAttention(d_model, num_heads, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, eps, **factory_kwargs)

        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)

        self.ln2 = RMSNorm(d_model, eps, **factory_kwargs)

        self.use_checkpoint = False

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """
        Apply the Transformer block with optional gradient checkpointing.

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
        normalized_x = self.ln1(x)
        attn_output = self.attn(normalized_x, rope=rope, token_positions=token_positions)
        z = x + attn_output

        normalized_z = self.ln2(z)
        ffn_output = self.ffn(normalized_z)
        y = z + ffn_output

        return y

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}"


class TransformerLM(nn.Module):
    """
    Transformer Language Model with performance enhancements.

    Enhanced with gradient checkpointing, memory optimizations, and architectural
    improvements for better training efficiency on modern hardware.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Transformer Language Model with optimizations.

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            context_length: Maximum sequence length the model can process
            d_model: Dimensionality of the model (embedding and hidden dimensions)
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads (d_model must be divisible by num_heads)
            d_ff: Dimensionality of the feed-forward inner layer
            rope_theta: RoPE theta parameter (typically 10000.0)
            eps: Epsilon value for RMSNorm numerical stability
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert context_length > 0, f"context_length must be positive, got {context_length}"
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"

        if d_ff % 64 != 0:
            d_ff = ((d_ff + 63) // 64) * 64

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, eps=eps, **factory_kwargs)
                for _ in range(num_layers)
            ]
        )

        self.mix_param = nn.Parameter(torch.zeros(1))

        self.ln_final = RMSNorm(d_model, eps, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

        self.use_gradient_checkpointing = False
        self._gradient_checkpointing_layers = None

    def enable_gradient_checkpointing(self, layers_to_checkpoint: int | None = None) -> None:
        """
        Enable gradient checkpointing for memory efficiency.

        Args:
            layers_to_checkpoint: Number of layers to checkpoint. If None, checkpoints all layers.
        """
        self.use_gradient_checkpointing = True

        if layers_to_checkpoint is None:
            layers_to_checkpoint = self.num_layers
        else:
            layers_to_checkpoint = min(layers_to_checkpoint, self.num_layers)

        self._gradient_checkpointing_layers = layers_to_checkpoint

        for i, layer in enumerate(self.layers):
            if i < layers_to_checkpoint:
                layer.use_checkpoint = True
                layer.attn.use_checkpoint = True
            else:
                layer.use_checkpoint = False
                layer.attn.use_checkpoint = False

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False
        for layer in self.layers:
            layer.use_checkpoint = False
            layer.attn.use_checkpoint = False

    def forward(
        self, input_ids: Int[torch.Tensor, "... batch_size seq_len"]
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        """
        Forward pass with memory and compute efficiency.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        assert seq_len <= self.context_length, (
            f"Input sequence length ({seq_len}) exceeds context length ({self.context_length})"
        )

        token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(input_ids)

        skip: list[torch.Tensor] = []
        half = len(self.layers) // 2
        for i, layer in enumerate(self.layers):
            if i < half:
                x = layer(x, rope=self.rope, token_positions=token_positions)
                skip.append(x)
            else:
                if skip:
                    residual = torch.sigmoid(self.mix_param) * skip.pop()
                    x = layer(x + residual, rope=self.rope, token_positions=token_positions)
                else:
                    x = layer(x, rope=self.rope, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics for monitoring."""
        if not torch.cuda.is_available():
            return {}

        stats = {}
        stats["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        stats["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / 1e9
        stats["parameter_memory_gb"] = param_memory

        return stats

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters by component."""
        counts = {}

        counts["embeddings"] = sum(p.numel() for p in self.token_embeddings.parameters())

        if self.layers:
            layer_params = sum(p.numel() for p in self.layers[0].parameters())
            counts["transformer_layer"] = layer_params
            counts["all_transformer_layers"] = layer_params * self.num_layers

        counts["final_ln"] = sum(p.numel() for p in self.ln_final.parameters())
        counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return counts

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"vocab_size={self.vocab_size}, context_length={self.context_length}, "
            f"d_model={self.d_model}, num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, d_ff={self.d_ff}, "
            f"rope_theta={self.rope_theta}"
        )
