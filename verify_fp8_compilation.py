#!/usr/bin/env python3
"""Quick verification that FP8 + compilation works correctly."""

import sys

import torch

sys.path.append(".")

from cs336_basics.model import TransformerLM

try:
    from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType, convert_to_float8_training

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("❌ TorchAO not available - FP8 will not work")
    sys.exit(1)


def main():
    print("=" * 80)
    print("FP8 + Compilation Verification")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # Create a small model for testing
    print("\n1. Creating model...")
    model = (
        TransformerLM(
            vocab_size=50257,
            max_seq_len=512,
            dim=512,
            n_layers=4,
            n_heads=8,
            head_dim=64,
            intermediate_size=2048,
            use_flash=True,
            use_gradient_checkpointing=False,
        )
        .cuda()
        .to(torch.bfloat16)
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Model size: {total_params:.1f}M parameters")

    # Convert to FP8
    print("\n2. Converting to FP8...")
    config = Float8LinearConfig(
        cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
        cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
        cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
    )

    convert_to_float8_training(model, config=config)

    float8_count = sum(1 for _, m in model.named_modules() if "Float8" in m.__class__.__name__)
    print(f"   Float8 modules: {float8_count}")

    if float8_count == 0:
        print("❌ FP8 conversion failed!")
        return

    # Test without compilation
    print("\n3. Testing FP8 WITHOUT compilation...")
    input_ids = torch.randint(0, 50257, (4, 512)).cuda()
    labels = input_ids.clone()

    # Warmup
    for _ in range(3):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
    model.zero_grad()
    torch.cuda.synchronize()

    # Test
    import time

    start = time.time()
    for _ in range(10):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    end = time.time()

    fp8_no_compile_time = (end - start) / 10
    print(f"   Time per iteration: {fp8_no_compile_time:.3f} seconds")

    # Compile the model
    print("\n4. Compiling FP8 model...")
    compiled_model = torch.compile(model, mode="default")

    # Test with compilation
    print("\n5. Testing FP8 WITH compilation...")

    # First run (compilation)
    print("   First run (includes compilation time)...")
    start = time.time()
    outputs = compiled_model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    compiled_model.zero_grad()
    torch.cuda.synchronize()
    end = time.time()
    print(f"   First run time: {end - start:.3f} seconds")

    # Warmup
    for _ in range(3):
        outputs = compiled_model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
    compiled_model.zero_grad()
    torch.cuda.synchronize()

    # Test
    start = time.time()
    for _ in range(10):
        outputs = compiled_model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        compiled_model.zero_grad()
    torch.cuda.synchronize()
    end = time.time()

    fp8_compile_time = (end - start) / 10
    print(f"   Time per iteration (compiled): {fp8_compile_time:.3f} seconds")

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"FP8 without compilation: {fp8_no_compile_time:.3f} sec/iter")
    print(f"FP8 with compilation:    {fp8_compile_time:.3f} sec/iter")
    speedup = fp8_no_compile_time / fp8_compile_time
    print(f"Compilation speedup:     {speedup:.2f}x")

    if speedup > 1.5:
        print("\n✅ SUCCESS: Compilation significantly improves FP8 performance!")
    else:
        print("\n⚠️  WARNING: Compilation speedup is lower than expected")
        print("   This might be due to the small model size used for testing")

    print("\nFor full training, use:")
    print("  python train_h100.py --use_fp8 --compile_model")


if __name__ == "__main__":
    main()
