#!/usr/bin/env python3
"""Test script for the optimized H100 model implementation."""

import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.model import TransformerLM, apply_torchao_optimizations


def test_model_creation():
    """Test model creation and parameter count."""
    print("🧪 Testing model creation...")

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
        use_gradient_checkpointing=False,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters ({total_params / 1e6:.1f}M)")

    return model


def test_forward_pass(model):
    """Test forward pass performance."""
    print("\n🧪 Testing forward pass...")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test input
    batch_size = 4
    seq_len = 1024
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids)
    forward_time = time.time() - start_time

    print(f"✅ Forward pass completed in {forward_time:.3f}s")
    print(f"Output logits shape: {outputs['logits'].shape}")

    # Test with labels
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")

    return forward_time


def test_torchao_optimizations():
    """Test TorchAO FP8 optimizations."""
    print("\n🧪 Testing TorchAO optimizations...")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping TorchAO test")
        return

    try:
        import torchao

        print("✅ TorchAO available")

        # Create model
        model = TransformerLM(
            vocab_size=50257,
            max_seq_len=512,  # Smaller for testing
            dim=512,
            n_layers=6,
            n_heads=8,
            head_dim=64,
            intermediate_size=2048,
            dropout=0.0,
            tie_embeddings=True,
            use_flash=False,
            use_gradient_checkpointing=False,
        )

        device = torch.device("cuda")
        model = model.to(device)

        # Apply TorchAO optimizations
        model = apply_torchao_optimizations(model, device)

        # Test forward pass
        batch_size = 2
        seq_len = 512
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        with torch.no_grad():
            outputs = model(input_ids)

        print(f"✅ TorchAO optimized forward pass successful")
        print(f"Output shape: {outputs['logits'].shape}")

    except ImportError:
        print("⚠️  TorchAO not available")
    except Exception as e:
        print(f"❌ TorchAO test failed: {e}")


def test_memory_usage():
    """Test memory usage with different batch sizes."""
    print("\n🧪 Testing memory usage...")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return

    device = torch.device("cuda")

    # Create smaller model for testing
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=1024,
        dim=768,
        n_layers=12,
        n_heads=12,
        head_dim=64,
        intermediate_size=3072,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=False,
        use_gradient_checkpointing=True,  # Enable for memory testing
    )

    model = model.to(device)

    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        seq_len = 1024
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        try:
            # Forward + backward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()

            # Memory stats
            current_memory = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9

            print(f"Batch size {batch_size:2d}: Current: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB")

        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size:2d}: OOM")
            break


def test_compilation():
    """Test torch.compile() performance."""
    print("\n🧪 Testing model compilation...")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping compilation test")
        return

    device = torch.device("cuda")

    # Create smaller model for testing
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=512,
        dim=512,
        n_layers=6,
        n_heads=8,
        head_dim=64,
        intermediate_size=2048,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=False,
        use_gradient_checkpointing=False,
    )

    model = model.to(device)

    # Test input
    batch_size = 4
    seq_len = 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Time uncompiled model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    uncompiled_time = time.time() - start_time

    print(f"Uncompiled time (10 runs): {uncompiled_time:.3f}s")

    # Compile model
    print("Compiling model...")
    compiled_model = torch.compile(model, mode="default")

    # Warmup compiled model
    for _ in range(3):
        with torch.no_grad():
            _ = compiled_model(input_ids)

    # Time compiled model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = compiled_model(input_ids)
    torch.cuda.synchronize()
    compiled_time = time.time() - start_time

    print(f"Compiled time (10 runs): {compiled_time:.3f}s")
    speedup = uncompiled_time / compiled_time
    print(f"✅ Compilation speedup: {speedup:.2f}x")


def main():
    """Run all tests."""
    print("🚀 Testing optimized H100 model implementation")
    print("=" * 60)

    # Test model creation
    model = test_model_creation()

    # Test forward pass
    test_forward_pass(model)

    # Test TorchAO optimizations
    test_torchao_optimizations()

    # Test memory usage
    test_memory_usage()

    # Test compilation
    test_compilation()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")

    # Print recommendations
    print("\n📋 Recommendations:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({total_memory:.1f}GB)")

        if "H100" in gpu_name:
            print("  ✅ H100 detected - use maximum optimizations")
            print("  ✅ Recommended batch size: 32-64")
        elif "A100" in gpu_name:
            print("  ✅ A100 detected - use most optimizations")
            print("  ✅ Recommended batch size: 16-32")
        else:
            print("  ⚠️  Consider reducing batch size for non-H100/A100 GPUs")
    else:
        print("  ⚠️  CUDA not available - performance will be limited")


if __name__ == "__main__":
    main()
