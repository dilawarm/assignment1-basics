"""
Enhanced Transformer models with architectural improvements for faster convergence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from cs336_basics.nn.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.layers import Embedding, Linear, RMSNorm


class SquaredReLUFFN(nn.Module):
    """
    Feed-forward network with squared ReLU activation: w2(max(w1(x), 0)^2)

    This activation function has been shown to provide better performance than
    standard ReLU or GLU variants in some cases.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """Apply squared ReLU transformation."""
        # w1(x)
        hidden = self.w1(x)
        # max(w1(x), 0)^2
        activated = F.relu(hidden).square()
        # w2(activated)
        return self.w2(activated)


class UNetTransformerBlock(nn.Module):
    """
    Transformer block with optional skip connection from earlier layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        use_squared_relu: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        # Attention components
        self.attn = MultiHeadSelfAttention(d_model, num_heads, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, eps, **factory_kwargs)

        # Feed-forward components
        if use_squared_relu:
            self.ffn = SquaredReLUFFN(d_model, d_ff, **factory_kwargs)
        else:
            from cs336_basics.nn.activations import SwiGLU

            self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)

        self.ln2 = RMSNorm(d_model, eps, **factory_kwargs)

        # Skip connection mixing parameter (learnable)
        self.skip_weight = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
        skip_connection: Float[torch.Tensor, "... seq_len d_model"] | None = None,
    ) -> tuple[Float[torch.Tensor, "... seq_len d_model"], Float[torch.Tensor, "... seq_len d_model"]]:
        """
        Forward pass with optional skip connection.

        Returns:
            - Output tensor
            - Hidden state for skip connection
        """
        # Store hidden state before residual for skip connection
        hidden_state = x.clone()

        # Mix with skip connection if provided
        if skip_connection is not None:
            # Sigmoid to bound the weight between 0 and 1
            alpha = torch.sigmoid(self.skip_weight)
            x = alpha * x + (1 - alpha) * skip_connection

        # Standard transformer block
        # Attention
        attn_out = self.attn(self.ln1(x), rope=rope, token_positions=token_positions)
        x = x + attn_out

        # Feed-forward
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x, hidden_state


class UNetTransformerLM(nn.Module):
    """
    Transformer Language Model with U-Net style skip connections.

    The model stores hidden states from the first half of layers and adds them
    to the second half in reverse order, creating a U-shaped architecture.
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
        use_squared_relu: bool = True,
        tie_embeddings: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert num_layers % 2 == 0, f"num_layers ({num_layers}) must be even for U-Net structure"

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings

        factory_kwargs = {"device": device, "dtype": dtype}

        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        # Positional encoding
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                UNetTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    eps=eps,
                    use_squared_relu=use_squared_relu,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.ln_final = RMSNorm(d_model, eps, **factory_kwargs)

        # Output head (untied by default)
        if tie_embeddings:
            self.lm_head = lambda x: F.linear(x, self.token_embeddings.weight)
        else:
            self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with enhanced initialization."""
        # Standard deviation for initialization
        std = 0.02

        # Initialize embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=std)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                # Note: our custom Linear layer doesn't have bias
            elif isinstance(module, UNetTransformerBlock):
                # Special initialization for skip weights
                nn.init.zeros_(module.skip_weight)

    def forward(
        self, input_ids: Int[torch.Tensor, "batch_size seq_len"]
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        """
        Forward pass with U-Net skip connections.
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.context_length, (
            f"Input sequence length ({seq_len}) exceeds context length ({self.context_length})"
        )

        # Token positions
        token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        # Embed tokens
        x = self.token_embeddings(input_ids)

        # Store skip connections from first half
        skip_connections = []
        half_layers = self.num_layers // 2

        # First half of layers (encoder path)
        for i in range(half_layers):
            x, hidden = self.layers[i](x, rope=self.rope, token_positions=token_positions)
            skip_connections.append(hidden)

        # Second half of layers (decoder path with skip connections)
        for i in range(half_layers, self.num_layers):
            skip_idx = self.num_layers - i - 1  # Reverse order
            skip_connection = skip_connections[skip_idx] if skip_idx < len(skip_connections) else None
            x, _ = self.layers[i](x, rope=self.rope, token_positions=token_positions, skip_connection=skip_connection)

        # Final layer norm
        x = self.ln_final(x)

        # Output projection
        if self.tie_embeddings:
            logits = self.lm_head(x)
        else:
            logits = self.lm_head(x)

        return logits

    def get_parameter_groups(self, base_lr: float = 1e-3):
        """
        Get parameter groups with different learning rates.

        Returns groups for:
        - Embeddings (higher lr)
        - LM head (lower lr)
        - 1D parameters (biases, norms)
        - Regular parameters
        """
        # Categorize parameters
        embed_params = set(self.token_embeddings.parameters())
        lm_head_params = set(self.lm_head.parameters()) if hasattr(self.lm_head, "parameters") else set()

        # 1D parameters (biases, layer norms, etc.)
        one_d_params = set()
        regular_params = set()

        for name, param in self.named_parameters():
            if param in embed_params or param in lm_head_params:
                continue

            # Check if parameter is 1D
            if param.ndim == 1 or "ln" in name or "norm" in name or "skip_weight" in name:
                one_d_params.add(param)
            else:
                regular_params.add(param)

        # Create parameter groups with different learning rates
        param_groups = [
            {"params": list(embed_params), "lr": base_lr * 5.0, "name": "embeddings"},
            {"params": list(lm_head_params), "lr": base_lr * 0.5, "name": "lm_head"},
            {"params": list(one_d_params), "lr": base_lr * 2.0, "name": "1d_params"},
            {"params": list(regular_params), "lr": base_lr, "name": "regular"},
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return param_groups
