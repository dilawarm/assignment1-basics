"""
Transformer Models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


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


class ReLUSquared(nn.Module):
    """ReLU² activation function for improved performance."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = F.relu(x)
        return relu * relu


class QKNorm(nn.Module):
    """Query-Key normalization for improved attention stability."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.q_norm = nn.LayerNorm(dim, eps=eps, bias=False)
        self.k_norm = nn.LayerNorm(dim, eps=eps, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention:
    - FlexAttention with sliding window
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
        window_size: int | None = None,
        use_qk_norm: bool = True,
        use_flex_attention: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.window_size = window_size
        self.use_flex_attention = use_flex_attention

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.qk_norm = QKNorm(self.head_dim) if use_qk_norm else None

        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2, rope_theta, device)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with improved strategy."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        freqs_cis = self.freqs_cis[:seq_len].to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_flex_attention and hasattr(F, "scaled_dot_product_attention"):
            try:
                if self.window_size is not None:
                    window_mask = self._create_sliding_window_mask(seq_len, x.device)
                    if mask is not None:
                        mask = mask & window_mask
                    else:
                        mask = window_mask

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal if mask is None else False,
                    )
            except:
                out = self._standard_attention(q, k, v, mask, is_causal)
        else:
            out = self._standard_attention(q, k, v, mask, is_causal)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(out)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window mask for local attention."""
        if self.window_size is None:
            return None

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        for i in range(seq_len):
            start_pos = max(0, i - self.window_size)
            mask[i, start_pos:i] = False

        return mask

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        """Standard attention implementation."""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        elif is_causal:
            seq_len = q.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        return torch.matmul(attn_weights, v)


class FeedForward(nn.Module):
    """
    Feed-forward network with:
    - ReLU² activation
    - SwiGLU option
    - Improved initialization
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu_squared",
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu

        if use_swiglu:
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)
            self.activation = nn.SiLU()
        else:
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)

            if activation == "relu_squared":
                self.activation = ReLUSquared()
            elif activation == "gelu":
                self.activation = nn.GELU()
            elif activation == "swish":
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with improved strategy."""
        if self.use_swiglu:
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.weight)
        else:
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            gate = self.activation(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
        else:
            hidden = self.activation(self.up_proj(x))

        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    """
    Transformer block with:
    - Pre-norm architecture
    - Skip connections with U-Net pattern
    - Modern attention and feed-forward
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        window_size: int | None = None,
        use_qk_norm: bool = True,
        use_flex_attention: bool = True,
        use_swiglu: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model, bias=False)
        self.ln_2 = nn.LayerNorm(d_model, bias=False)

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            window_size=window_size,
            use_qk_norm=use_qk_norm,
            use_flex_attention=use_flex_attention,
            device=device,
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation="relu_squared",
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
        norm_x = self.ln_1(x)
        attn_out = self.attention(norm_x, mask=mask, is_causal=is_causal)
        x = x + self.dropout(attn_out)

        if skip_connection is not None:
            x = x + skip_connection

        norm_x = self.ln_2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)

        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        context_length: int = 2048,
        d_model: int = 2048,
        num_layers: int = 24,
        num_heads: int = 32,
        d_ff: int = 8192,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        window_size: int | None = 1024,
        use_qk_norm: bool = True,
        use_flex_attention: bool = True,
        use_swiglu: bool = False,
        tie_embeddings: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    window_size=window_size,
                    use_qk_norm=use_qk_norm,
                    use_flex_attention=use_flex_attention,
                    use_swiglu=use_swiglu,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model, bias=False)

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.skip_embeddings = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers // 2)])

        self._init_weights()

        self.to(device)

    def _init_weights(self):
        """Initialize weights with modern best practices."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        for skip_emb in self.skip_embeddings:
            nn.init.zeros_(skip_emb.weight)

        if self.lm_head is not None:
            nn.init.zeros_(self.lm_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)

        skip_connections = {}

        for i, layer in enumerate(self.layers):
            skip_connection = None
            if i < len(self.skip_embeddings):
                skip_emb = self.skip_embeddings[i](x)
                skip_connections[i] = skip_emb

            if i >= self.num_layers // 2:
                skip_idx = self.num_layers - 1 - i
                if skip_idx in skip_connections:
                    skip_connection = skip_connections[skip_idx]

            x = layer(x, mask=mask, is_causal=is_causal, skip_connection=skip_connection)

        x = self.ln_f(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.token_embedding.weight)

        return logits

    def configure_optimizers(self, learning_rate: float, weight_decay: float, betas: tuple[float, float]):
        """Configure optimizers using the hybrid Muon approach."""
        from ..training.muon_optimizer import MuonAdamHybrid

        return MuonAdamHybrid(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
