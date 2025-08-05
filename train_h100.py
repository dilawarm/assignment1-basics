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
    parser.add_argument("--n_heads", type=int, default=64, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="FFN intermediate size")

    # Training arguments (optimized for H100)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU (H100 optimal: 128-256)")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation (1 for best perf)"
    )
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
        default=False,
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

    print("=" * 60)
    print("🚀 H100-Optimized 350M Transformer Training")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"🔧 GPU: {gpu_name}")
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
    print(f"📊 Model: {total_params:.1f}M parameters")

    if args.use_fp8 and TORCHAO_AVAILABLE:
        try:
            # Move model to CUDA and BF16 BEFORE FP8 conversion
            model = model.cuda().to(torch.bfloat16)

            config = Float8LinearConfig(
                cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
                cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
            )

            convert_to_float8_training(model, config=config)
            float8_count = sum(1 for _, m in model.named_modules() if "Float8" in m.__class__.__name__)

            if float8_count == 0:
                print("⚠️  FP8 conversion failed, using BF16")
                args.use_fp8 = False
            else:
                print(f"✅ FP8 enabled ({float8_count} modules)")
                if not args.compile_model:
                    print("⚠️  WARNING: FP8 without compilation will be slower than BF16!")
        except Exception as e:
            print(f"⚠️  FP8 failed ({e}), using BF16")
            args.use_fp8 = False
            model = model.cuda().to(torch.bfloat16)
    else:
        model = model.cuda().to(torch.bfloat16)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    if "H100" in gpu_name:
        if args.use_fp8:
            tokens_per_second = 300_000  # Realistic for 350M model
        elif args.use_flash_attn:
            tokens_per_second = 200_000
        else:
            tokens_per_second = 150_000
    elif "A100" in gpu_name:
        tokens_per_second = 100_000 if args.use_flash_attn else 80_000
    else:
        tokens_per_second = 50_000

    total_seconds = args.max_hours * 3600
    total_tokens = tokens_per_second * total_seconds
    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * args.max_length
    max_steps = int(total_tokens / tokens_per_step)

    print(f"⏱️  Training plan: {max_steps:,} steps (~{total_tokens / 1e9:.1f}B tokens, {args.max_hours}h)")

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

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    # Configuration summary
    precision = "FP8" if args.use_fp8 else "BF16"
    compile_status = f"✅ {args.compile_mode}" if args.compile_model else "❌"
    flash_status = "✅" if args.use_flash_attn else "❌"

    print(f"⚙️  Config: {precision} | Compile: {compile_status} | Flash: {flash_status}")

    # Quick memory check
    if "H100" in gpu_name or "A100" in gpu_name:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        estimated_usage_gb = (total_params * 6) / 1000 + 5  # Rough estimate
        usage_pct = (estimated_usage_gb / total_memory_gb) * 100

        if usage_pct < 50:
            print(
                f"💾 Memory: ~{usage_pct:.0f}% ({estimated_usage_gb:.1f}GB/{total_memory_gb:.1f}GB) - Consider larger batch size"
            )
        elif usage_pct > 90:
            print(f"💾 Memory: ~{usage_pct:.0f}% ({estimated_usage_gb:.1f}GB/{total_memory_gb:.1f}GB) - May hit OOM!")
        else:
            print(f"💾 Memory: ~{usage_pct:.0f}% ({estimated_usage_gb:.1f}GB/{total_memory_gb:.1f}GB)")

    print("\n🚀 Starting training...")
    print("=" * 60)

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
        print(f"\n❌ Training failed with RuntimeError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
