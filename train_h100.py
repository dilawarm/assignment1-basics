"""
H100-optimized training script for 350M parameter transformer.
Target: Beat validation loss of 3.0781 on OpenWebText in 1.5 hours.
"""

import argparse
import os
import sys
from datetime import datetime

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import torch

# Check for critical dependencies early
try:
    from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType, convert_to_float8_training

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("WARNING: TorchAO not installed - FP8 will not be available")
    print("         Install with: pip install torchao")

try:
    import flash_attn

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING: Flash Attention not installed - performance will be limited")
    print("         Install with: pip install flash-attn --no-build-isolation")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data import create_dataloaders
from cs336_basics.model import TransformerLM
from cs336_basics.training import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train 350M transformer on H100")

    # Model arguments
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="FFN intermediate size")

    # Training arguments (optimized for H100)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=4e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--max_hours", type=float, default=1.5, help="Maximum training hours")

    # Data arguments
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")

    # Optimization arguments
    parser.add_argument("--use_fp8", action="store_true", default=True, help="Use FP8 precision with TorchAO")
    parser.add_argument("--no_fp8", dest="use_fp8", action="store_false", help="Disable FP8 precision")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention")
    parser.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false", help="Disable Flash Attention")
    parser.add_argument("--compile_model", action="store_true", default=True, help="Compile model with torch.compile")
    parser.add_argument("--no_compile", dest="compile_model", action="store_false", help="Disable model compilation")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: default - most stable)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,  # Changed default to False for better H100 performance
        help="Use gradient checkpointing (saves memory but ~3x slower - not needed on H100 80GB)",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases logging")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")

    # Debugging arguments
    parser.add_argument("--debug_performance", action="store_true", help="Enable performance debugging")

    args = parser.parse_args()

    print("=" * 80)
    print("H100-Optimized 350M Transformer Training")
    print("=" * 80)
    print(f"Model Configuration:")
    print(f"  - Parameters: ~350M")
    print(f"  - Layers: {args.n_layers}")
    print(f"  - Hidden size: {args.dim}")
    print(f"  - Heads: {args.n_heads} x {args.head_dim}")
    print(f"  - FFN size: {args.intermediate_size}")
    print(f"  - Sequence length: {args.max_length}")
    print()
    print(f"Training Configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.max_length} tokens")
    print(f"  - Learning rate: {args.learning_rate} (peak) -> {args.min_learning_rate} (min)")
    print(f"  - Warmup steps: {args.warmup_steps}")
    print()
    print(f"Optimizations:")
    print(f"  - FP8 precision (TorchAO): {args.use_fp8}")
    print(f"  - Flash Attention: {args.use_flash_attn}")
    print(f"  - Model compilation: {args.compile_model}")
    if args.compile_model:
        print(f"  - Compile mode: {args.compile_mode}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}", end="")
    if args.gradient_checkpointing:
        print(" (⚠️  ~3x slowdown!)")
    else:
        print(" (✓ optimal for H100)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    print(f"Compute Capability: {compute_capability}")

    if "H100" not in gpu_name and "A100" not in gpu_name:
        print("WARNING: Not running on H100/A100. Performance will be suboptimal.")

    # Automatically adjust settings based on hardware and dependencies
    if args.use_fp8:
        if compute_capability < 8.9:
            print("\nWARNING: FP8 requires compute capability >= 8.9 (H100 or newer)")
            print("         Disabling FP8 and using BF16 mixed precision instead")
            args.use_fp8 = False
        elif not TORCHAO_AVAILABLE:
            print("\nWARNING: FP8 requested but TorchAO not installed")
            print("         Disabling FP8 and using BF16 mixed precision instead")
            args.use_fp8 = False
        else:
            print("\n✓ Using TorchAO Float8")
            print("  PyTorch's official FP8 solution for stable training")
            print("  Expected speedup: ~1.5x over FP16")

    if args.use_flash_attn and not FLASH_ATTN_AVAILABLE:
        print("\nWARNING: Flash Attention requested but not installed")
        print("         Disabling Flash Attention - performance will be limited")
        args.use_flash_attn = False

    # Warn about gradient checkpointing on H100
    if "H100" in gpu_name and args.gradient_checkpointing:
        print("\n⚠️  WARNING: Gradient checkpointing is enabled on H100!")
        print("   This will reduce performance by ~3x and is unnecessary with 80GB memory.")
        print("   Consider running with --no_gradient_checkpointing for better performance.")

    print("\nCreating model...")
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=args.max_length,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=args.use_flash_attn,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    if args.use_fp8 and TORCHAO_AVAILABLE:
        print("\nConverting model to TorchAO Float8...")
        try:
            # Move model to CUDA and BF16 BEFORE FP8 conversion
            model = model.cuda().to(torch.bfloat16)

            config = Float8LinearConfig(
                cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
            )

            convert_to_float8_training(model, config=config)
            print("✓ Model converted to TorchAO Float8")

            float8_count = sum(1 for _, m in model.named_modules() if "Float8" in m.__class__.__name__)
            print(f"  Float8 modules: {float8_count}")

            if float8_count == 0:
                print("❌ WARNING: No Float8 modules found! FP8 conversion may have failed.")
                print("   Falling back to BF16...")
                args.use_fp8 = False
        except Exception as e:
            print(f"❌ FP8 conversion failed: {e}")
            print("   Falling back to BF16...")
            args.use_fp8 = False
            # Ensure model is on CUDA with BF16
            model = model.cuda().to(torch.bfloat16)
    else:
        # Move model to CUDA and BF16 if not using FP8
        model = model.cuda().to(torch.bfloat16)

    print("\nCreating data loaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    if "H100" in gpu_name:
        if args.use_fp8:
            tokens_per_second = 900_000
        elif args.use_flash_attn:
            tokens_per_second = 700_000
        else:
            tokens_per_second = 500_000
    elif "A100" in gpu_name:
        tokens_per_second = 400_000 if args.use_flash_attn else 300_000
    else:
        tokens_per_second = 200_000

    total_seconds = args.max_hours * 3600
    total_tokens = tokens_per_second * total_seconds
    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * args.max_length
    max_steps = int(total_tokens / tokens_per_step)

    print(f"\nTraining plan:")
    print(f"  - Estimated tokens/sec: {tokens_per_second:,}")
    print(f"  - Total training time: {args.max_hours} hours")
    print(f"  - Total tokens: {total_tokens:,} ({total_tokens / 1e9:.1f}B)")
    print(f"  - Total steps: {max_steps:,}")

    config = TrainingConfig(
        model_name=f"gpt-350m-h100-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_interval=5000,
        max_length=args.max_length,
        num_workers=args.num_workers,
        use_fp8=args.use_fp8,
        use_amp=not args.use_fp8,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        use_flash_attn=args.use_flash_attn,
        gradient_checkpointing=args.gradient_checkpointing,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
    )

    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    # Quick performance test if debugging is enabled
    if args.debug_performance:
        print("\nRunning quick performance test...")
        with torch.no_grad():
            test_input = torch.randint(0, 50257, (args.batch_size, args.max_length)).cuda()

            # Warmup
            for _ in range(3):
                _ = model(test_input)
            torch.cuda.synchronize()

            # Test
            import time

            start = time.time()
            num_test_iters = 10
            for _ in range(num_test_iters):
                _ = model(test_input)
            torch.cuda.synchronize()
            end = time.time()

            test_tokens = args.batch_size * args.max_length * num_test_iters
            test_tokens_per_sec = test_tokens / (end - start)
            print(f"  Forward pass only: {test_tokens_per_sec:,.0f} tokens/sec")
            print(f"  Time per iteration: {(end - start) / num_test_iters:.3f} seconds")

    print("\nStarting training...")
    print("Target: Beat validation loss of 3.0781")
    print("Expected: Achieve validation loss of 2.90-2.95")

    # Print actual configuration status
    print("\nActual Configuration:")
    print(f"  - Model on CUDA: {next(model.parameters()).is_cuda}")
    print(f"  - Model dtype: {next(model.parameters()).dtype}")
    print(f"  - FP8 enabled: {args.use_fp8}")
    print(f"  - Flash Attention: {args.use_flash_attn}")
    print(f"  - Model compilation: {args.compile_model}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")

    print("\nExpected Performance on H100:")
    expected_tokens_sec = 0
    if args.use_fp8:
        expected_tokens_sec = 900_000
        print("  - Tokens/sec: ~900,000 (with FP8)")
    elif args.use_flash_attn:
        expected_tokens_sec = 700_000
        print("  - Tokens/sec: ~700,000 (with BF16 + Flash Attention)")
    else:
        expected_tokens_sec = 500_000
        print("  - Tokens/sec: ~500,000 (with BF16)")

    if args.gradient_checkpointing and expected_tokens_sec > 0:
        expected_tokens_sec = expected_tokens_sec // 3  # Gradient checkpointing causes ~3x slowdown
        print(f"  - ⚠️  WITH gradient checkpointing: ~{expected_tokens_sec:,} tokens/sec")

    expected_mfu = 40 if not args.gradient_checkpointing else 15
    print(f"  - MFU: >{expected_mfu}%")
    print("\nIf you see <100,000 tokens/sec, something is wrong!")
    print("-" * 80)

    try:
        final_metrics = trainer.train()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"Final validation perplexity: {final_metrics['val_perplexity']:.2f}")

        if final_metrics["val_loss"] < 3.0781:
            print(f"✅ SUCCESS! Beat target of 3.0781 by {3.0781 - final_metrics['val_loss']:.4f}")
        else:
            print(f"❌ FAILED to beat target. Missed by {final_metrics['val_loss'] - 3.0781:.4f}")

        print("=" * 80)

    except RuntimeError as e:
        print(f"\nTraining failed with error: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check GPU memory with: nvidia-smi")
        print("2. Try reducing batch size: --batch_size 4")
        print("3. Try without compilation: --no_compile")
        print("4. Try without FP8: --no_fp8")
        if args.compile_model:
            print(f"5. Try different compile mode: --compile_mode default (current: {args.compile_mode})")

        if args.use_fp8 and not TORCHAO_AVAILABLE:
            print("\n6. Install TorchAO for FP8 support: pip install torchao")

        sys.exit(1)


if __name__ == "__main__":
    main()
