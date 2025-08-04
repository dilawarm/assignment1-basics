#!/usr/bin/env python3
"""Diagnostic script to identify performance bottlenecks on H100."""

import sys
import time
from contextlib import contextmanager

import torch

sys.path.append(".")


@contextmanager
def timer(name):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    print(f"{name}: {end - start:.4f} seconds")


def diagnose():
    print("=" * 80)
    print("H100 Performance Diagnostic")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU: {gpu_name}")

    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    print(f"✓ Compute Capability: {compute_capability}")

    if "H100" not in gpu_name:
        print("⚠️  Warning: Not running on H100!")

    print("\nPrecision Support:")
    print(f"  - BF16: {torch.cuda.is_bf16_supported()}")
    print(f"  - FP16: {torch.cuda.get_device_capability()[0] >= 7}")
    print(f"  - FP8: {compute_capability >= 8.9}")

    print("\nDependencies:")
    try:
        import flash_attn

        print("✓ Flash Attention installed")
    except ImportError:
        print("❌ Flash Attention NOT installed")
        print("  Install with: pip install flash-attn --no-build-isolation")

    try:
        from torchao.float8 import convert_to_float8_training

        print("✓ TorchAO installed")
    except ImportError:
        print("❌ TorchAO NOT installed")
        print("  Install with: pip install torchao")

    print("\nModel Performance Test:")

    from cs336_basics.model import TransformerLM

    configs = [
        ("FP32", torch.float32, False),
        ("FP16", torch.float16, False),
        ("BF16", torch.bfloat16, False),
        ("BF16 + Flash", torch.bfloat16, True),
    ]

    for name, dtype, use_flash in configs:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print(f"\n{name}: Skipped (BF16 not supported)")
            continue

        print(f"\n{name}:")

        try:
            # Create small model for testing
            model = (
                TransformerLM(
                    vocab_size=50257,
                    max_seq_len=1024,
                    dim=1024,
                    n_layers=4,
                    n_heads=16,
                    head_dim=64,
                    intermediate_size=4096,
                    use_flash=use_flash,
                    use_gradient_checkpointing=False,
                )
                .cuda()
                .to(dtype)
            )

            for batch_size in [1, 8, 16]:
                input_ids = torch.randint(0, 50257, (batch_size, 1024)).cuda()

                for _ in range(3):
                    _ = model(input_ids)
                torch.cuda.synchronize()

                start = time.time()
                num_iters = 10
                for _ in range(num_iters):
                    outputs = model(input_ids)
                    loss = outputs["loss"] if "loss" in outputs else outputs["logits"].mean()
                torch.cuda.synchronize()
                end = time.time()

                total_tokens = batch_size * 1024 * num_iters
                tokens_per_sec = total_tokens / (end - start)
                ms_per_token = 1000 * (end - start) / total_tokens

                print(f"  Batch {batch_size}: {tokens_per_sec:,.0f} tokens/sec ({ms_per_token:.2f} ms/token)")

                allocated_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                print(f"    Memory: {allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved")

        except Exception as e:
            print(f"  Error: {e}")

    print("\nCompilation Test:")
    try:
        model = (
            TransformerLM(
                vocab_size=50257,
                max_seq_len=512,
                dim=512,
                n_layers=2,
                n_heads=8,
                head_dim=64,
                intermediate_size=2048,
            )
            .cuda()
            .to(torch.bfloat16)
        )

        compiled_model = torch.compile(model, mode="default")

        input_ids = torch.randint(0, 50257, (1, 512)).cuda()

        with timer("First run (compilation)"):
            _ = compiled_model(input_ids)

        with timer("Second run (compiled)"):
            _ = compiled_model(input_ids)

        print("✓ Compilation working")

    except Exception as e:
        print(f"❌ Compilation error: {e}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    if "H100" in gpu_name:
        print("✓ Running on H100 - good!")
        print("\nFor best performance, use:")
        print("  python train_h100.py --batch_size 16 --gradient_accumulation_steps 8 \\")
        print("    --use_fp8 --use_flash_attn --compile_mode default")
    else:
        print("⚠️  Not running on H100 - performance will be limited")

    if tokens_per_sec < 100000:
        print("\n⚠️  Performance is very low. Check:")
        print("  1. Is the model running on GPU (not CPU)?")
        print("  2. Are optimizations enabled (Flash Attention, compilation)?")
        print("  3. Is the batch size too small?")


if __name__ == "__main__":
    diagnose()
