#!/usr/bin/env python3
"""Quick test script to verify model implementation."""

import torch

from cs336_basics.model import TransformerLM


def test_model():
    """Test model creation and forward pass."""
    print("Testing 350M transformer model...")

    # Create model
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=1024,
        dim=1024,
        n_layers=24,
        n_heads=16,
        head_dim=64,
        intermediate_size=4096,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=False,  # Disable for CPU testing
        use_fp8=False,  # Disable for CPU testing
        use_gradient_checkpointing=False,
    )

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))

    print(f"\nTesting forward pass with input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, 50257)")

    # Test with labels
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)

    print(f"\nLoss: {outputs['loss'].item():.4f}")

    # Test generation
    print("\nTesting generation...")
    prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Example prompt
    generated = model.generate(prompt, max_length=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_model()
