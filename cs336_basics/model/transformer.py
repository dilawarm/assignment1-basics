"""Main transformer model implementation with selective mixed precision."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import TorchAO for FP8
try:
    import torchao
    from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_

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
        # Keep normalization layers in FP32 for numerical stability
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
    """Transformer Language Model with selective mixed precision."""

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

        # Token embeddings - Keep in FP32 for stability
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Transformer blocks - These will use mixed precision
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

        # Final layer norm - Keep in FP32 for stability
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

        # Token embeddings (kept in FP32)
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
                    use_reentrant=True,  # More compatible with torch.compile() and TorchAO
                )
            else:
                h, present_kv = block(h, attention_mask, use_cache, past_kv)

            if use_cache:
                present_key_values = present_key_values + (present_kv,)

        # Final layer norm (kept in FP32)
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


def apply_selective_mixed_precision(model: nn.Module, use_compile: bool = True) -> nn.Module:
    """
    Apply selective mixed precision following best practices for transformers.
    
    This version is compatible with torch.compile() by avoiding TorchAO FP8 when compiling.

    Based on research and NVIDIA recommendations:
    1. BF16 for most operations (excellent stability and compile compatibility)
    2. FP32 for embeddings and normalization layers (critical for stability)
    3. FP8 only when not using torch.compile() (due to compatibility issues)
    """
    print("🔧 Applying selective mixed precision optimizations...")

    # For torch.compile() compatibility, prefer BF16 over FP8
    if use_compile:
        print("🔧 Using BF16 mixed precision (torch.compile() compatible)")
        
        # Apply BF16 to most operations for excellent performance and stability
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
            print("✅ Applied BF16 mixed precision to transformer blocks")
        else:
            # Fallback to FP16 if BF16 not supported
            model = model.to(dtype=torch.float16)
            print("✅ Applied FP16 mixed precision (BF16 not supported)")
    
    else:
        # Only try FP8 when NOT using torch.compile()
        print("🔧 Using FP8 + BF16 mixed precision (no compilation)")
        
        if TORCHAO_AVAILABLE:
            try:
                # Get all linear layers for FP8 quantization
                linear_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Apply FP8 to linear layers in attention and FFN
                        if any(part in name for part in ["attn", "ffn", "qkv_proj", "o_proj", "w1", "w2", "w3"]):
                            linear_layers.append(name)

                if linear_layers:
                    # Apply FP8 quantization to linear layers
                    quantize_(model, float8_dynamic_activation_float8_weight())
                    print(f"✅ Applied FP8 to {len(linear_layers)} linear layers")
                else:
                    print("⚠️  No suitable linear layers found for FP8")

            except Exception as e:
                print(f"⚠️  Failed to apply FP8 quantization: {e}")
                print("Falling back to BF16 for all operations")
        else:
            print("⚠️  TorchAO not available, using BF16 only")

        # Apply BF16 to remaining operations
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
            print("✅ Applied BF16 mixed precision")
        else:
            model = model.to(dtype=torch.float16)
            print("✅ Applied FP16 mixed precision (BF16 not supported)")

    # Step 3: Keep critical layers in FP32 for numerical stability
    def convert_critical_layers_to_fp32(module):
        """Keep embeddings and normalization layers in FP32."""
        if isinstance(module, (nn.Embedding, nn.LayerNorm, RMSNorm)):
            module.to(dtype=torch.float32)

    model.apply(convert_critical_layers_to_fp32)
    print("✅ Kept critical layers (embeddings, norms) in FP32")

    # Step 4: Print precision summary
    _print_precision_summary(model)

    return model


def _print_precision_summary(model: nn.Module):
    """Print summary of precision usage in the model."""
    precision_counts = {}
    total_params = 0

    for name, param in model.named_parameters():
        dtype = str(param.dtype)
        if dtype not in precision_counts:
            precision_counts[dtype] = {"count": 0, "params": 0}
        precision_counts[dtype]["count"] += 1
        precision_counts[dtype]["params"] += param.numel()
        total_params += param.numel()

    print(f"\n📊 Precision Summary:")
    for dtype, info in precision_counts.items():
        percentage = (info["params"] / total_params) * 100
        print(f"  {dtype}: {info['count']} tensors, {info['params']:,} params ({percentage:.1f}%)")

    # Estimate memory savings
    if "torch.bfloat16" in precision_counts or "torch.float16" in precision_counts:
        mixed_precision_params = sum(
            info["params"] for dtype, info in precision_counts.items() if "float16" in dtype or "bfloat16" in dtype
        )
        memory_savings = (mixed_precision_params / total_params) * 50  # ~50% savings for FP16/BF16
        print(f"  💾 Estimated memory savings: ~{memory_savings:.1f}%")


# For backward compatibility
apply_torchao_optimizations = apply_selective_mixed_precision