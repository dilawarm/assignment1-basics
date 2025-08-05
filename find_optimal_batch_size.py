#!/usr/bin/env python3
"""Script to find the optimal batch size for H100 with 80GB VRAM."""

import gc
import os
import sys

import torch

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

sys.path.append(".")

from cs336_basics.model import TransformerLM

try:
    from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType, convert_to_float8_training

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


def get_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return allocated, reserved, total
    return 0, 0, 0


def test_batch_size(batch_size, use_fp8=True, compile_model=True):
    """Test if a specific batch size fits in memory."""
    torch.cuda.empty_cache()
    gc.collect()

    try:
        print(f"\nTesting batch size: {batch_size}")

        # Create model (350M parameters)
        model = (
            TransformerLM(
                vocab_size=50257,
                max_seq_len=1024,
                dim=896,
                n_layers=24,
                n_heads=14,
                head_dim=64,
                intermediate_size=3584,
                use_flash=True,
                use_gradient_checkpointing=False,
            )
            .cuda()
            .to(torch.bfloat16)
        )

        # FP8 conversion if requested
        if use_fp8 and TORCHAO_AVAILABLE:
            config = Float8LinearConfig(
                cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
            )
            convert_to_float8_training(model, config=config)

        # Compile if requested
        if compile_model:
            model = torch.compile(model, mode="default")

        # Create optimizer (this doubles memory usage!)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Test forward and backward pass
        input_ids = torch.randint(0, 50257, (batch_size, 1024)).cuda()
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        allocated_after_forward, _, _ = get_memory_info()
        print(f"  After forward: {allocated_after_forward:.1f}GB allocated")

        # Backward pass
        loss.backward()

        allocated_after_backward, _, _ = get_memory_info()
        print(f"  After backward: {allocated_after_backward:.1f}GB allocated")

        # Optimizer step (this is where Adam states are allocated)
        optimizer.step()
        optimizer.zero_grad()

        final_allocated, final_reserved, total = get_memory_info()
        print(f"  After optimizer: {final_allocated:.1f}GB allocated, {final_reserved:.1f}GB reserved")
        print(f"  Total GPU memory: {total:.1f}GB")
        print(f"  Memory usage: {(final_reserved / total) * 100:.1f}%")

        # Clean up
        del model, optimizer, input_ids, labels, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

        return True, final_reserved

    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OOM with batch size {batch_size}")
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0
    except Exception as e:
        print(f"  ❌ Error with batch size {batch_size}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0


def find_optimal_batch_size(use_fp8=True, compile_model=True):
    """Binary search to find the largest batch size that fits."""
    print("=" * 80)
    print("FINDING OPTIMAL BATCH SIZE FOR H100")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No GPU available!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Total memory: {total_memory:.1f}GB")
    print(f"Configuration: FP8={use_fp8}, Compile={compile_model}")

    # Start with reasonable bounds
    min_batch = 1
    max_batch = 512  # Start high for H100
    optimal_batch = min_batch
    optimal_memory = 0

    # First, quickly check if max_batch works
    print(f"\nQuick test with batch_size={max_batch}...")
    success, memory = test_batch_size(max_batch, use_fp8, compile_model)

    if success:
        print(f"\n✓ Batch size {max_batch} works! GPU is very capable.")
        optimal_batch = max_batch
        optimal_memory = memory
        # Try even larger
        max_batch = 1024

    # Binary search
    print("\nPerforming binary search...")
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2

        # Round to nearest power of 2 for better performance
        mid_batch = 2 ** round(torch.log2(torch.tensor(float(mid_batch))).item())
        mid_batch = max(min_batch, min(max_batch, mid_batch))

        if mid_batch == optimal_batch:
            break

        success, memory = test_batch_size(mid_batch, use_fp8, compile_model)

        if success:
            optimal_batch = mid_batch
            optimal_memory = memory
            min_batch = mid_batch + 1
        else:
            max_batch = mid_batch - 1

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Optimal batch size: {optimal_batch}")
    print(f"Memory usage: {optimal_memory:.1f}GB ({(optimal_memory / total_memory) * 100:.1f}%)")

    # Calculate effective batch sizes
    print("\nRecommended configurations:")
    print(f"1. Maximum throughput (no gradient accumulation):")
    print(f"   --batch_size {optimal_batch} --gradient_accumulation_steps 1")
    print(f"   Effective batch size: {optimal_batch * 1024} tokens")

    if optimal_batch >= 64:
        print(f"\n2. Balanced (some gradient accumulation):")
        print(f"   --batch_size {optimal_batch // 2} --gradient_accumulation_steps 2")
        print(f"   Effective batch size: {optimal_batch * 1024} tokens")

    if optimal_batch >= 32:
        print(f"\n3. Conservative (more stable training):")
        print(f"   --batch_size {optimal_batch // 4} --gradient_accumulation_steps 4")
        print(f"   Effective batch size: {optimal_batch * 1024} tokens")

    # Performance tips
    print("\n" + "=" * 80)
    print("PERFORMANCE TIPS")
    print("=" * 80)
    print("1. Larger batch sizes reduce overhead from optimizer steps")
    print("2. Powers of 2 are optimal for GPU kernels")
    print("3. Leave ~10% memory free for PyTorch's caching allocator")
    print("4. First training step will use more memory due to compilation")

    return optimal_batch


def main():
    """Run the batch size finder."""
    import argparse

    parser = argparse.ArgumentParser(description="Find optimal batch size for H100")
    parser.add_argument("--no_fp8", action="store_true", help="Disable FP8")
    parser.add_argument("--no_compile", action="store_true", help="Disable compilation")
    args = parser.parse_args()

    use_fp8 = not args.no_fp8 and TORCHAO_AVAILABLE
    compile_model = not args.no_compile

    find_optimal_batch_size(use_fp8, compile_model)


if __name__ == "__main__":
    main()
