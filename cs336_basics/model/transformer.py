"""Transformer model for TorchAO FP8 training."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

from .components import RMSNorm, RotaryPositionEmbedding, SwiGLU


class MultiHeadAttention(nn.Module):
    """Multi-head attention using standard PyTorch operations."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash: bool = True,
        layer_idx: int = 0,
        total_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_flash = use_flash
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        self.qkv_proj = nn.Linear(dim, 3 * n_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionEmbedding(head_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        std = 0.02
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=std)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)

        output_std = std / math.sqrt(2 * self.total_layers)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=output_std)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_heads * self.head_dim, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_kv = (k, v) if use_cache else None

        if self.use_flash:
            try:
                attn_out = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    softmax_scale=1.0 / math.sqrt(self.head_dim),
                    causal=True,
                )
            except ImportError:
                attn_out = self._standard_attention(q, k, v, attention_mask)
        else:
            attn_out = self._standard_attention(q, k, v, attention_mask)

        attn_out = attn_out.reshape(batch_size, seq_len, -1)

        out = self.o_proj(attn_out)
        out = self.dropout(out)

        return out, present_kv

    def _standard_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is None:
            seq_len = q.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float("-inf"))
        else:
            scores += attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)

        return attn_out.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Standard transformer block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
        use_flash: bool = True,
        layer_idx: int = 0,
        total_layers: int = 1,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=False,
            use_flash=use_flash,
            layer_idx=layer_idx,
            total_layers=total_layers,
        )
        self.ffn = SwiGLU(dim=dim, hidden_dim=intermediate_size, dropout=dropout)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = x
        x = self.attention_norm(x)
        attn_out, present_kv = self.attn(x, attention_mask, use_cache, past_key_value)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        return x, present_kv


class TransformerLM(nn.Module):
    """Transformer Language Model for TorchAO FP8 training."""

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        dim: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        head_dim: int = 64,
        intermediate_size: int = 4096,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        use_flash: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.n_layers = n_layers
        self.tie_embeddings = tie_embeddings
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.token_emb = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    use_flash=use_flash,
                    layer_idx=i,
                    total_layers=n_layers,
                )
                for i in range(n_layers)
            ]
        )

        self.ln_f = RMSNorm(dim)

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

        self._print_model_size()

    def _init_weights(self):
        """Initialize model weights with scaled initialization."""
        std = 0.02
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=std)

        if self.lm_head is not None:
            output_std = std / math.sqrt(2 * self.n_layers)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=output_std)

    def _print_model_size(self):
        """Print model parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")

    def get_input_embeddings(self):
        return self.token_emb

    def get_output_embeddings(self):
        return self.lm_head if not self.tie_embeddings else self.token_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for language modeling loss
            use_cache: Whether to return key/value cache
            past_key_values: Optional past key/values for caching

        Returns:
            Dictionary with 'logits' and optionally 'loss' and 'past_key_values'
        """
        h = self.token_emb(input_ids)

        present_key_values = () if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.use_gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                h, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    h,
                    attention_mask,
                    use_cache,
                    past_kv,
                )
            else:
                h, present_kv = block(h, attention_mask, use_cache, past_kv)

            if use_cache:
                present_key_values = present_key_values + (present_kv,)

        h = self.ln_f(h)

        if self.tie_embeddings:
            logits = F.linear(h, self.token_emb.weight)
        else:
            logits = self.lm_head(h)

        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        if use_cache:
            output["past_key_values"] = present_key_values

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Simple generation method for testing.

        Args:
            input_ids: Starting token IDs
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[torch.arange(logits.shape[0]).unsqueeze(1), indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
