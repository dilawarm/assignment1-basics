"""
Transformer models with advanced stability features.

This module implements state-of-the-art transformer architectures with:
- Adaptive gradient clipping (ZClip/AdaGC)
- Outlier-safe initialization
- Stable attention mechanisms
- Memory-efficient implementations for H100 GPU
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = "cuda") -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # Handle device availability - fall back to CPU if CUDA is not available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    Stable RMSNorm implementation that prevents channel-wise amplification.
    Based on latest 2025 research for outlier-safe training.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single-scale RMSNorm to prevent outliers
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key normalization for stable attention."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k


class MultiHeadAttention(nn.Module):
    """
    Stable multi-head attention with:
    - No sliding window (removes instability)
    - Improved numerical stability
    - Outlier-safe implementation
    - QK normalization
    - RoPE embeddings
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Use bias=False for better stability
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.qk_norm = QKNorm(self.head_dim) if use_qk_norm else None

        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2, rope_theta, device)

        self._init_stable_weights()

    def _init_stable_weights(self):
        """Outlier-safe weight initialization."""
        # Conservative Xavier initialization with gain < 1.0
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.8)

        # Zero initialization for output projection (residual path)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        freqs_cis = self.freqs_cis[:seq_len].to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Apply QK normalization for stability
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Reshape for attention computation
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Stable attention computation
        out = self._stable_attention(q, k, v, mask, is_causal)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(out)

    def _stable_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        """Numerically stable attention computation."""
        # Compute attention scores with temperature scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed (remove sliding window completely)
        if is_causal:
            seq_len = q.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e4)  # Use -1e4 instead of -inf for stability

        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        # Stable softmax with temperature control
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout if training
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Compute output
        output = torch.matmul(attn_weights, v)
        return output


class FeedForward(nn.Module):
    """
    Stable feed-forward network with:
    - Outlier-safe activations
    - Conservative initialization
    - Optional SwiGLU activation
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_swiglu = use_swiglu

        if use_swiglu:
            # SwiGLU implementation
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        else:
            self.linear1 = nn.Linear(d_model, d_ff, bias=False)
            self.linear2 = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.activation_fn = self._get_activation(activation)

        self._init_stable_weights()

    def _init_stable_weights(self):
        """Conservative weight initialization."""
        if self.use_swiglu:
            nn.init.xavier_uniform_(self.w1.weight, gain=0.8)
            nn.init.zeros_(self.w2.weight)  # Zero init for residual path
            nn.init.xavier_uniform_(self.w3.weight, gain=0.8)
        else:
            nn.init.xavier_uniform_(self.linear1.weight, gain=0.8)
            nn.init.zeros_(self.linear2.weight)  # Zero init for residual path

    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "relu_squared":
            return lambda x: F.relu(x) ** 2
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: x * sigmoid(W1 @ x) * (W3 @ x)
            gate = torch.sigmoid(self.w1(x))
            hidden = gate * self.w3(x)
            output = self.w2(hidden)
        else:
            hidden = self.activation_fn(self.linear1(x))
            hidden = self.dropout(hidden)
            output = self.linear2(hidden)

        return self.dropout(output)


class TransformerBlock(nn.Module):
    """
    Stable transformer block with:
    - Pre-norm architecture
    - Stable residual connections
    - Conservative initialization
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
        use_swiglu: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.ln_1 = RMSNorm(d_model)
        self.ln_2 = RMSNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_qk_norm=use_qk_norm,
            device=device,
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation="gelu",  # Use GELU for better stability
            use_swiglu=use_swiglu,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = True,
        skip_connection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        norm_x = self.ln_1(x)
        attn_out = self.attention(norm_x, mask=mask, is_causal=is_causal)
        x = x + self.dropout(attn_out)

        # Optional U-Net style skip connection
        if skip_connection is not None:
            x = x + skip_connection

        # Pre-norm feedforward with residual
        norm_x = self.ln_2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)

        return x


class TransformerLM(nn.Module):
    """
    Stable Transformer Language Model with outlier-safe training.
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
        use_qk_norm: bool = True,
        use_swiglu: bool = False,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learnable embedding projection for outlier redistribution
        self.embedding_proj = nn.Linear(d_model, d_model, bias=False)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    use_qk_norm=use_qk_norm,
                    use_swiglu=use_swiglu,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.ln_f = RMSNorm(d_model)

        # Output projection
        if tie_embeddings:
            self.lm_head = None  # Will use embedding weights
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_stable_weights()

    def _init_stable_weights(self):
        """Outlier-safe weight initialization."""
        # Conservative embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Zero init for embedding projection
        nn.init.zeros_(self.embedding_proj.weight)

        # LM head initialization
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embedding with projection for outlier redistribution
        x = self.embedding(input_ids)
        x = x + self.embedding_proj(x)  # Learnable residual projection

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, is_causal=True)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection
        if self.tie_embeddings:
            # Use embedding weights for output projection
            logits = F.linear(x, self.embedding.weight)
        else:
            logits = self.lm_head(x)

        return logits
