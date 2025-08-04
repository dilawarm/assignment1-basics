"""Test script to verify TorchAO FP8 implementation."""

import sys

import torch
from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType, convert_to_float8_training

sys.path.append(".")

from cs336_basics.model import TransformerLM


def test_standard_model():
    """Test that the standard model works without FP8."""
    print("Testing standard TransformerLM...")

    use_flash = torch.cuda.is_available()
    if not use_flash:
        print("  Note: Running on CPU, Flash Attention disabled")

    model = TransformerLM(
        vocab_size=1000,
        max_seq_len=128,
        dim=256,
        n_layers=4,
        n_heads=4,
        head_dim=64,
        intermediate_size=1024,
        use_flash=use_flash,
    )

    if torch.cuda.is_available():
        # Use bfloat16 for Flash Attention compatibility
        model = model.cuda().to(torch.bfloat16)
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    outputs = model(input_ids)
    print(f"✓ Model output shape: {outputs['logits'].shape}")
    print(f"✓ Expected shape: ({batch_size}, {seq_len}, 1000)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")


def test_torchao_conversion():
    """Test TorchAO FP8 conversion."""
    print("\nTesting TorchAO FP8 conversion...")

    try:
        print("✓ TorchAO imported successfully")
    except ImportError:
        print("✗ TorchAO not installed")
        print("  Install with: pip install torchao")
        return False

    model = TransformerLM(
        vocab_size=1000,
        max_seq_len=128,
        dim=256,
        n_layers=4,
        n_heads=4,
        head_dim=64,
        intermediate_size=1024,
        use_flash=True,
    )

    if not torch.cuda.is_available():
        print("✗ CUDA not available - skipping FP8 test")
        return False

    device = torch.device("cuda")
    # Use bfloat16 for better stability and Flash Attention compatibility
    model = model.to(device).to(torch.bfloat16)

    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Compute capability: {compute_capability}")

    if compute_capability < 8.9:
        print("✗ FP8 requires compute capability >= 8.9 (H100)")
        return False

    linear_count_before = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"  Linear modules before: {linear_count_before}")

    config = Float8LinearConfig(
        cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
        cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
        cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
    )

    convert_to_float8_training(model, config=config)
    print("✓ Model converted to FP8")

    float8_count = sum(1 for _, m in model.named_modules() if "Float8" in m.__class__.__name__)
    print(f"✓ Float8 modules after: {float8_count}")

    input_ids = torch.randint(0, 1000, (2, 64), device=device)

    try:
        outputs = model(input_ids)
        print("✓ FP8 forward pass successful")
        print(f"  Output shape: {outputs['logits'].shape}")
        return True
    except Exception as e:
        print(f"✗ FP8 forward pass failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TorchAO FP8 Implementation Test")
    print("=" * 60)

    test_standard_model()

    success = test_torchao_conversion()

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed or were skipped")
    print("=" * 60)
