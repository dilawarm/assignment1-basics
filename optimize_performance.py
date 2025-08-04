#!/usr/bin/env python3
"""Script to optimize H100 training performance by testing different configurations."""

import argparse
import sys
import time

import torch

sys.path.append(".")

from cs336_basics.data import create_dataloaders
from cs336_basics.model import TransformerLM
from cs336_basics.training import Trainer, TrainingConfig

try:
    from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType, convert_to_float8_training

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


def test_configuration(name, model, batch_size=16, seq_len=1024, num_iters=20):
    """Test a specific model configuration."""
    print(f"\nTesting: {name}")
    print("-" * 40)

    # Move to CUDA if not already
    if not next(model.parameters()).is_cuda:
        model = model.cuda()

    # Create dummy data
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).cuda()
    labels = input_ids.clone()

    # Warmup
    print("  Warming up...")
    for _ in range(3):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
    torch.cuda.synchronize()

    # Clear gradients
    model.zero_grad()

    # Benchmark
    print("  Benchmarking...")
    torch.cuda.synchronize()
    start = time.time()

    for i in range(num_iters):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if i % 5 == 0:
            model.zero_grad()  # Clear gradients periodically

    torch.cuda.synchronize()
    end = time.time()

    # Calculate metrics
    total_time = end - start
    tokens_processed = batch_size * seq_len * num_iters
    tokens_per_sec = tokens_processed / total_time
    time_per_iter = total_time / num_iters

    # Calculate MFU
    model_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
    flops_per_token = 6 * model_params * 1e6
    flops_per_second = tokens_per_sec * flops_per_token
    tflops = flops_per_second / 1e12
    h100_tflops_peak = 990  # BF16 peak
    mfu = (tflops / h100_tflops_peak) * 100

    print(f"  Time per iteration: {time_per_iter:.3f} seconds")
    print(f"  Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"  TFLOPS: {tflops:.1f}")
    print(f"  MFU: {mfu:.1f}%")

    # Memory usage
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"  Memory: {allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved")

    return tokens_per_sec, mfu


def main():
    parser = argparse.ArgumentParser(description="Optimize H100 performance")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print("=" * 80)
    print("H100 Performance Optimization")
    print("=" * 80)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    if "H100" not in gpu_name:
        print("WARNING: Not running on H100!")

    # Base model configuration
    base_config = {
        "vocab_size": 50257,
        "max_seq_len": 1024,
        "dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "tie_embeddings": True,
    }

    results = {}

    # Test 1: Baseline (BF16, no optimizations)
    print("\n1. Creating baseline model (BF16, no Flash, no checkpoint)...")
    model = (
        TransformerLM(
            **base_config,
            use_flash=False,
            use_gradient_checkpointing=False,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    results["Baseline (BF16)"] = test_configuration("Baseline (BF16)", model, args.batch_size)
    del model
    torch.cuda.empty_cache()

    # Test 2: Flash Attention only
    print("\n2. Creating Flash Attention model...")
    model = (
        TransformerLM(
            **base_config,
            use_flash=True,
            use_gradient_checkpointing=False,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    results["BF16 + Flash"] = test_configuration("BF16 + Flash Attention", model, args.batch_size)
    del model
    torch.cuda.empty_cache()

    # Test 3: Flash + Gradient Checkpointing
    print("\n3. Creating Flash + Gradient Checkpointing model...")
    model = (
        TransformerLM(
            **base_config,
            use_flash=True,
            use_gradient_checkpointing=True,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    results["BF16 + Flash + Checkpoint"] = test_configuration("BF16 + Flash + Grad Checkpoint", model, args.batch_size)
    del model
    torch.cuda.empty_cache()

    # Test 4: Flash + Compilation
    print("\n4. Creating Flash + Compiled model...")
    model = (
        TransformerLM(
            **base_config,
            use_flash=True,
            use_gradient_checkpointing=False,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    model = torch.compile(model, mode="default")
    results["BF16 + Flash + Compile"] = test_configuration("BF16 + Flash + Compile", model, args.batch_size)
    del model
    torch.cuda.empty_cache()

    # Test 5: FP8 if available
    if TORCHAO_AVAILABLE:
        print("\n5. Creating FP8 model...")
        model = (
            TransformerLM(
                **base_config,
                use_flash=True,
                use_gradient_checkpointing=False,
            )
            .cuda()
            .to(torch.bfloat16)
        )

        try:
            config = Float8LinearConfig(
                cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
            )
            convert_to_float8_training(model, config=config)

            float8_count = sum(1 for _, m in model.named_modules() if "Float8" in m.__class__.__name__)
            print(f"  Float8 modules: {float8_count}")

            if float8_count > 0:
                results["FP8 + Flash"] = test_configuration("FP8 + Flash", model, args.batch_size)
            else:
                print("  FP8 conversion failed - no Float8 modules found")
        except Exception as e:
            print(f"  FP8 conversion failed: {e}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    for config_name, (tokens_per_sec, mfu) in results.items():
        print(f"{config_name:30s}: {tokens_per_sec:>10,.0f} tokens/sec, {mfu:>5.1f}% MFU")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_config = max(results.items(), key=lambda x: x[1][0])
    print(f"Best configuration: {best_config[0]}")
    print(f"Performance: {best_config[1][0]:,.0f} tokens/sec ({best_config[1][1]:.1f}% MFU)")

    # Specific recommendations based on results
    baseline_perf = results.get("Baseline (BF16)", (0, 0))[0]
    flash_perf = results.get("BF16 + Flash", (0, 0))[0]
    checkpoint_perf = results.get("BF16 + Flash + Checkpoint", (0, 0))[0]

    if baseline_perf > 0 and checkpoint_perf > 0:
        checkpoint_slowdown = baseline_perf / checkpoint_perf
        if checkpoint_slowdown > 2:
            print(f"\n⚠️  Gradient checkpointing causes {checkpoint_slowdown:.1f}x slowdown!")
            print("   Consider disabling it with: --no_gradient_checkpointing")

    if flash_perf > baseline_perf * 1.2:
        print(f"\n✓ Flash Attention provides {flash_perf / baseline_perf:.1f}x speedup")

    if best_config[1][0] < 100000:
        print("\n❌ Performance is still low. Check:")
        print("   1. Is the GPU being fully utilized? (nvidia-smi)")
        print("   2. Are there any CPU bottlenecks?")
        print("   3. Try larger batch sizes if memory allows")


if __name__ == "__main__":
    main()
