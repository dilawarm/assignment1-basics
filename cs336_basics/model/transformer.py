"""Main transformer model implementation with TorchAO FP8 support."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import TorchAO for FP8
try:
    import torchao

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

from .attention import MultiHeadAttention
from .components import RMSNorm, SwiGLU, scaled_init_


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward."""

    def __init__(
        self,
        dim: int = 1024,
        n_heads: int = 16,
        head_dim: int = 64,
        intermediate_size: int = 4096,
        dropout: float = 0.0,
        use_flash: bool = True,
        layer_idx: int = 0,
        total_layers: int = 24,
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx

        # Layer normalization (RMSNorm is more efficient than LayerNorm)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Multi-head attention
        self.attn = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_flash=use_flash,
            layer_idx=layer_idx,
            total_layers=total_layers,
        )

        # Feedforward with SwiGLU
        self.ffn = SwiGLU(dim, intermediate_size)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for transformer block."""
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, present_kv = self.attn(x, attention_mask, use_cache, past_key_value)
        x = residual + self.dropout(attn_out)

        # FFN with residual
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        return x, present_kv


class TransformerLM(nn.Module):
    """Transformer Language Model with exact 350M parameters."""

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 tokenizer size
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

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Transformer blocks
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

        # Final layer norm
        self.ln_f = RMSNorm(dim)

        # Language modeling head
        if tie_embeddings:
            # Share weights between input and output embeddings
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Print model size
        self._print_model_size()

    def _init_weights(self):
        """Initialize model weights with scaled initialization."""
        # Token embeddings
        std = 0.02
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=std)

        # LM head (if not tied)
        if self.lm_head is not None:
            # Smaller initialization for output layer
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
        return self.lm_head if self.lm_head is not None else self.token_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> Dict[str, torch.Tensor]:
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
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        h = self.token_emb(input_ids)

        # Process through transformer blocks
        present_key_values = () if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
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
                    use_reentrant=True,  # More compatible with torch.compile()
                )
            else:
                h, present_kv = block(h, attention_mask, use_cache, past_kv)

            if use_cache:
                present_key_values = present_key_values + (present_kv,)

        # Final layer norm
        h = self.ln_f(h)

        # Language modeling head
        if self.tie_embeddings:
            # Use shared embeddings
            logits = F.linear(h, self.token_emb.weight)
        else:
            logits = self.lm_head(h)

        # Prepare output
        output = {"logits": logits}

        # Calculate loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for loss calculation
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
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Simple generation method for testing."""
        self.eval()

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature

            # Apply top-k/top-p filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def apply_torchao_optimizations(model: nn.Module, device: torch.device) -> nn.Module:
    """Apply TorchAO optimizations including FP8."""
    if not TORCHAO_AVAILABLE:
        print("Warning: TorchAO not available. Skipping FP8 optimizations.")
        return model

    try:
        # Check if device supports FP8
        if device.type != "cuda":
            print("Warning: FP8 requires CUDA device. Skipping.")
            return model

        # Check GPU compute capability for FP8 support
        props = torch.cuda.get_device_properties(device)
        if props.major < 8 or (props.major == 8 and props.minor < 9):
            print(f"Warning: GPU compute capability {props.major}.{props.minor} doesn't support FP8. Skipping.")
            return model

        # Apply FP8 quantization using TorchAO
        from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_

        # Move model to device first
        model = model.to(device)

        # Quantize the model to FP8 - be more selective about which layers
        print("🔧 Applying TorchAO FP8 quantization...")
        quantize_(model, float8_dynamic_activation_float8_weight())
        print("✅ Applied TorchAO FP8 quantization")

    except Exception as e:
        print(f"Warning: Failed to apply TorchAO optimizations: {e}")
        print("Continuing without FP8...")

    return model
