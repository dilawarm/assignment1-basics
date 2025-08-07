#!/usr/bin/env python3
"""
H100-Optimized Training Script with 2025 State-of-the-Art Techniques

This script implements the latest H100 optimizations based on 2025 research:
- TorchAO FP8 mixed precision
- Advanced learning rate scheduling
- Pre-tokenized .npy data loading
- Aggressive H100-specific configurations
- Dynamic batch size optimization

Target: Beat validation loss of 3.0781 in 1.5 hours on H100.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.data.optimized_dataloader import create_optimized_dataloaders
from cs336_basics.model import TransformerLM, apply_selective_mixed_precision
from cs336_basics.training import H100OptimizedTrainer, H100TrainingConfig


def find_optimal_batch_size(model: nn.Module, device: torch.device, max_length: int = 1024) -> int:
    """
    Dynamically find the optimal batch size for the H100 GPU.

    Based on GPU memory and model size, find the largest batch size that fits.
    """
    print("🔍 Finding optimal batch size for H100...")

    # Get GPU memory
    if device.type == "cuda":
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 Available GPU memory: {gpu_memory_gb:.1f} GB")
    else:
        print("⚠️  Not using CUDA, defaulting to batch size 2")
        return 2

    # Start with a reasonable batch size and increase
    batch_sizes_to_try = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    optimal_batch_size = 8  # Safe default

    model.train()

    for batch_size in batch_sizes_to_try:
        try:
            print(f"🧪 Testing batch size {batch_size}...")

            # Create dummy batch
            dummy_input = torch.randint(0, 50257, (batch_size, max_length)).to(device)
            dummy_labels = dummy_input.clone()

            # Test forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=dummy_input, labels=dummy_labels)
                loss = outputs["loss"]

            # Test backward pass
            loss.backward()

            # Check memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            memory_percent = (memory_used / gpu_memory_gb) * 100

            print(f"   Memory used: {memory_used:.1f} GB ({memory_percent:.1f}%)")

            if memory_percent < 85:  # Keep some headroom
                optimal_batch_size = batch_size
                print(f"✅ Batch size {batch_size} fits!")
            else:
                print(f"❌ Batch size {batch_size} too large")
                break

            # Clear memory
            del outputs, loss, dummy_input, dummy_labels
            torch.cuda.empty_cache()
            model.zero_grad()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM at batch size {batch_size}")
                break
            else:
                raise e

    print(f"🎯 Optimal batch size: {optimal_batch_size}")

    # Clear any remaining memory
    torch.cuda.empty_cache()

    return optimal_batch_size


def setup_h100_environment():
    """Setup optimal H100 environment variables and settings."""
    print("🔧 Setting up H100 environment...")

    # CUDA optimizations
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Memory optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

    # Compilation optimizations
    os.environ["TORCH_COMPILE_DEBUG"] = "0"  # Disable debug for performance
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_cache"

    # NVCC optimizations for H100
    os.environ["NVCC_PREPEND_FLAGS"] = "-ccbin g++"

    # Set optimal torch settings for H100
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for speed
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

    # Flash Attention optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    print("✅ H100 environment configured!")


def main():
    parser = argparse.ArgumentParser(description="H100-Optimized 350M Transformer Training")

    # Model arguments
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="FFN intermediate size")

    # Training arguments (optimized for H100)
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0=auto-detect)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate (increased)")
    parser.add_argument("--min_learning_rate", type=float, default=6e-5, help="Min learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps (reduced)")
    parser.add_argument("--max_hours", type=float, default=1.5, help="Maximum training hours")

    # Data arguments
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data workers")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory with .npy files")

    # H100 optimization arguments
    parser.add_argument("--use_fp8", type=bool, default=True, help="Use FP8 precision")
    parser.add_argument("--use_compile", type=bool, default=True, help="Use torch.compile")
    parser.add_argument("--use_flash_attn", type=bool, default=True, help="Use Flash Attention")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Gradient checkpointing")

    # Advanced optimizations
    parser.add_argument("--auto_batch_size", type=bool, default=True, help="Auto-detect batch size")
    parser.add_argument("--use_memmap", type=bool, default=True, help="Use memory-mapped .npy files")

    # Logging
    parser.add_argument("--use_wandb", type=bool, default=True, help="Use W&B logging")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_optimized", help="Output directory")

    args = parser.parse_args()

    # Setup H100 environment
    setup_h100_environment()

    # Print configuration
    print("=" * 100)
    print("🚀 H100-OPTIMIZED 350M TRANSFORMER TRAINING (2025 EDITION)")
    print("=" * 100)
    print(f"🎯 GOAL: Beat validation loss of 3.0781 in {args.max_hours} hours")
    print(f"🔧 Model: ~350M parameters, {args.n_layers} layers, {args.dim}d")
    print(f"📁 Data: Pre-tokenized .npy files from {args.data_dir}")
    print(f"⚡ Optimizations: FP8={args.use_fp8}, Compile={args.use_compile}, Flash={args.use_flash_attn}")
    print("=" * 100)

    # Verify GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"🎯 GPU: {gpu_name}")
    print(f"🔍 Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"💾 Memory: {memory_gb:.1f} GB")

    # H100 verification
    if "H100" in gpu_name:
        print("✅ H100 detected! All optimizations enabled.")
    elif compute_cap[0] >= 8:
        print("🔶 A100/RTX detected. Most optimizations will work.")
    else:
        print("⚠️  Older GPU detected. Some optimizations may be limited.")

    # Verify data files exist
    train_file = os.path.join(args.data_dir, "owt_train_tokens.npy")
    val_file = os.path.join(args.data_dir, "owt_valid_tokens.npy")

    if not os.path.exists(train_file):
        print(f"❌ ERROR: Training data file not found: {train_file}")
        sys.exit(1)
    if not os.path.exists(val_file):
        print(f"❌ ERROR: Validation data file not found: {val_file}")
        sys.exit(1)

    print(f"✅ Found pre-tokenized data files in {args.data_dir}")

    # Create model with exact 350M parameters
    print("\n🏗️  Creating 350M parameter model...")
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=args.max_length,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        dropout=0.0,  # No dropout for better convergence
        tie_embeddings=True,
        use_flash=args.use_flash_attn,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    model = model.to(device)

    # Auto-detect optimal batch size
    if args.auto_batch_size or args.batch_size == 0:
        optimal_batch_size = find_optimal_batch_size(model, device, args.max_length)
        args.batch_size = optimal_batch_size

    # Calculate gradient accumulation to target ~1M tokens
    target_tokens = 1_048_576  # 1M tokens
    current_tokens = args.batch_size * args.max_length
    optimal_grad_accum = max(1, target_tokens // current_tokens)

    if args.gradient_accumulation_steps != optimal_grad_accum:
        print(f"🔧 Adjusting gradient accumulation: {args.gradient_accumulation_steps} → {optimal_grad_accum}")
        args.gradient_accumulation_steps = optimal_grad_accum

    effective_batch_tokens = args.batch_size * args.gradient_accumulation_steps * args.max_length
    print(f"📊 Effective batch size: {effective_batch_tokens:,} tokens")

    # Create optimized data loaders using .npy files
    print(f"\n📚 Creating optimized .npy data loaders...")
    train_dataloader, val_dataloader = create_optimized_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        seed=42,
        use_memmap=args.use_memmap,
    )

    # Calculate training steps based on target performance
    print(f"\n⏱️  Calculating training schedule...")
    # Target: 800K tokens/sec (more aggressive with .npy files)
    target_tokens_per_sec = 800_000  # Higher expectation with pre-tokenized data
    total_seconds = args.max_hours * 3600
    total_tokens = target_tokens_per_sec * total_seconds
    tokens_per_step = effective_batch_tokens
    max_steps = int(total_tokens / tokens_per_step)

    print(f"   Training time: {args.max_hours} hours")
    print(f"   Target throughput: {target_tokens_per_sec:,} tokens/sec")
    print(f"   Total tokens: {total_tokens:,} ({total_tokens / 1e9:.1f}B)")
    print(f"   Total steps: {max_steps:,}")

    # Create optimized training config
    config = H100TrainingConfig(
        model_name=f"gpt-350m-h100-npy-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Optimized learning rates
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        # Batch configuration
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size_tokens=effective_batch_tokens,
        # Training schedule
        max_steps=max_steps,
        warmup_steps=args.warmup_steps,
        cooldown_steps=500,
        # Frequent evaluation for early detection of success
        eval_interval=500,
        log_interval=50,
        save_interval=2000,
        # Data config
        max_length=args.max_length,
        num_workers=args.num_workers,
        # H100 optimizations
        use_flash_attention=args.use_flash_attn,
        use_fp8=args.use_fp8,
        use_compile=args.use_compile,
        gradient_checkpointing=args.gradient_checkpointing,
        # Advanced optimizations
        use_fused_adamw=True,
        use_lr_warmup_cosine_decay=True,
        pin_memory=True,
        non_blocking=True,
        # Logging
        use_wandb=args.use_wandb,
        project_name="cs336-h100-npy-2025",
        output_dir=args.output_dir,
    )

    # Create H100OptimizedTrainer
    print(f"\n🚀 Creating H100OptimizedTrainer...")
    trainer = H100OptimizedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    # Start training
    print("\n" + "=" * 100)
    print("🚀 STARTING OPTIMIZED TRAINING WITH .NPY DATA")
    print("=" * 100)
    print("🎯 Target: Beat validation loss of 3.0781")
    print("📈 Expected: Achieve 2.90-2.95 validation loss")
    print("⚡ Optimizations: FP8 + Flash Attention + torch.compile + pre-tokenized data")
    print("🔥 Expected throughput: 800K+ tokens/sec (with .npy files)")
    print("=" * 100)

    start_time = time.time()

    try:
        final_metrics = trainer.train()

        # Print results
        training_time_hours = (time.time() - start_time) / 3600

        print("\n" + "=" * 100)
        print("🏁 TRAINING COMPLETE!")
        print("=" * 100)
        print(f"⏱️  Training time: {training_time_hours:.2f} hours")
        print(f"🎯 Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"📊 Final perplexity: {final_metrics['val_perplexity']:.2f}")

        # Success/failure analysis
        target_loss = 3.0781
        if final_metrics["val_loss"] < target_loss:
            improvement = target_loss - final_metrics["val_loss"]
            print(f"✅ SUCCESS! Beat target by {improvement:.4f}")
            print(f"🎉 Achieved goal in {training_time_hours:.2f} hours!")

            # Performance analysis
            if improvement > 0.15:  # Beat by significant margin
                print("🚀 EXCEPTIONAL PERFORMANCE!")
            elif improvement > 0.05:
                print("💪 SOLID PERFORMANCE!")
            else:
                print("✅ Target achieved!")

        else:
            deficit = final_metrics["val_loss"] - target_loss
            print(f"❌ Missed target by {deficit:.4f}")

            # Provide suggestions
            if deficit < 0.1:
                print("🔧 Very close! Try: longer training, higher learning rate")
            elif deficit < 0.2:
                print("🔧 Getting there! Try: larger batch size, different architecture")
            else:
                print("🔧 Need significant improvements! Check data preprocessing, model architecture")

        print("=" * 100)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint(os.path.join(config.output_dir, "interrupted_checkpoint.pt"))

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Attempting to save checkpoint...")
        try:
            trainer.save_checkpoint(os.path.join(config.output_dir, "error_checkpoint.pt"))
        except:
            print("💥 Could not save checkpoint")
        raise


if __name__ == "__main__":
    main()
