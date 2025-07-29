"""
Optimized Transformer Training Script for Fast Compilation and Training.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import TransformerLM
from cs336_basics.training.checkpoint import save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.muon_optimizer import MuonAdamHybrid


@dataclass
class TrainArgs:
    """Training configuration."""

    # Model parameters
    vocab_size: int = 50304
    context_length: int = 1024
    num_layers: int = 12
    d_model: int = 1280
    num_heads: int = 20
    d_ff: int = 5120
    rope_theta: float = 10000.0
    window_size: int | None = 512
    use_qk_norm: bool = True
    use_flex_attention: bool = False
    use_swiglu: bool = True
    tie_embeddings: bool = True

    # Optimizer parameters
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    muon_momentum: float = 0.95

    # Learning rate schedule
    max_learning_rate: float = 0.01
    min_learning_rate: float = 0.001
    warmup_iters: int = 300
    cosine_cycle_iters: int = 10000

    # Data paths
    training_set: str = "data/encoded/owt_train_tokens.npy"
    validation_set: str = "data/encoded/owt_valid_tokens.npy"

    # Training configuration
    validation_step_interval: int = 120
    checkpoint_step_interval: int = 1200
    steps: int = 10000
    batch_size: int = 40
    gradient_accumulation_steps: int = 2
    gradient_clipping: float = 1.0

    # Optimization flags
    device: str = "cuda"
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"  # Faster compilation
    use_mixed_precision: bool = True
    use_efficient_attention: bool = True
    use_fused_kernels: bool = True

    # Experiment logging
    experiment_name: str = "fast_training"
    experiment_description: str = "Fast compilation + training"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1"
    wandb_entity: str = ""
    log_dir: str = "experiments"

    def __post_init__(self):
        """Validate configuration."""
        assert self.vocab_size > 0
        assert self.context_length > 0
        assert self.d_model > 0
        assert self.d_model % self.num_heads == 0
        assert self.steps > 0
        assert self.batch_size > 0
        assert self.max_learning_rate > 0

        if self.vocab_size % 128 != 0:
            self.vocab_size = ((self.vocab_size + 127) // 128) * 128
            print(f"Padded vocab_size to {self.vocab_size} for efficiency")


class DataLoader:
    """Optimized data loader."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        context_length: int,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.data = np.load(data_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.num_workers = num_workers

        print(f"Loaded data: {len(self.data):,} tokens")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch with efficient memory access patterns."""
        return get_batch(self.data, self.batch_size, self.context_length, device=self.device)


class FastTrainer:
    """Optimized trainer for fast compilation and training."""

    def __init__(self, args: TrainArgs):
        self.args = args
        self.step = 0

        # Handle device availability
        if args.device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            args.device = "cpu"
        else:
            self.device = torch.device(args.device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_experiment_name = f"{args.experiment_name}_{timestamp}"
        self.experiment_timestamp = timestamp

        self.experiment_logger = ExperimentLogger(
            experiment_name=timestamped_experiment_name,
            description=args.experiment_description,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity if args.wandb_entity else None,
        )

        print(f"🚀 Experiment: {timestamped_experiment_name}")
        self.experiment_logger.log_hyperparameters(**asdict(args))

        self.training_integrator = TrainingIntegrator(
            experiment_logger=self.experiment_logger,
            hardware_log_interval=50,
        )

        print("🏗️ Initializing optimized transformer model...")

        # Optimize PyTorch settings for speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            window_size=args.window_size,
            use_qk_norm=args.use_qk_norm,
            use_flex_attention=args.use_flex_attention,
            use_swiglu=args.use_swiglu,
            tie_embeddings=args.tie_embeddings,
            device=self.device,
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model: {total_params:,} parameters ({total_params / 1e9:.2f}B)")
        print(f"🎯 Memory usage: {total_params * 2 / 1e9:.2f} GB (FP16)")

        # Fast compilation strategy
        if args.compile_model:
            print(f"⚡ Compiling model with {args.compile_mode} mode (optimized for speed)...")
            try:
                compile_start = time.time()
                if args.compile_mode == "reduce-overhead":
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                elif args.compile_mode == "max-autotune":
                    torch._inductor.config.max_autotune = True
                    self.model = torch.compile(self.model, mode="max-autotune", dynamic=True)
                else:
                    self.model = torch.compile(self.model, mode=args.compile_mode)

                compile_time = time.time() - compile_start
                print(f"✅ Model compiled in {compile_time:.1f}s")
                self.experiment_logger.add_note(f"Model compiled with {args.compile_mode} in {compile_time:.1f}s")

            except Exception as e:
                print(f"⚠️ Compilation failed: {e}")
                print("🏃 Running in eager mode")
                self.experiment_logger.add_note(f"Compilation failed: {e}")

        print("🔧 Initializing Muon optimizer...")
        self.optimizer = MuonAdamHybrid(
            self.model.parameters(),
            lr=args.max_learning_rate,
            betas=args.betas,
            weight_decay=args.weight_decay,
            muon_momentum=args.muon_momentum,
        )

        print(f"📚 Loading training data from {args.training_set}")
        self.train_loader = DataLoader(
            args.training_set,
            args.batch_size,
            args.context_length,
            device=str(self.device),
        )

        if Path(args.validation_set).exists():
            print(f"📚 Loading validation data from {args.validation_set}")
            self.val_loader = DataLoader(
                args.validation_set,
                args.batch_size,
                args.context_length,
                device=str(self.device),
            )
        else:
            self.val_loader = None

        # Disable mixed precision on CPU
        if self.device.type == "cpu" and args.use_mixed_precision:
            print("⚠️ Mixed precision not supported on CPU, disabling")
            args.use_mixed_precision = False

        self.scaler = torch.amp.GradScaler() if args.use_mixed_precision else None
        if self.scaler:
            print("⚡ Mixed precision training enabled")

    def calculate_mfu(self, tokens_per_sec: float) -> float:
        """Calculate Model FLOPs Utilization."""
        try:
            N = sum(p.numel() for p in self.model.parameters())
            L = self.args.num_layers
            H = self.args.d_model
            Q = self.args.context_length

            flops_per_token_forward = 6 * N + 12 * L * H * Q
            flops_per_token_training = 3 * flops_per_token_forward
            model_flops_per_sec = tokens_per_sec * flops_per_token_training

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                if "H100" in gpu_name:
                    peak_flops = 1979e12
                elif "A100" in gpu_name:
                    peak_flops = 624e12
                else:
                    peak_flops = 100e12
            else:
                peak_flops = 1e12

            return min(model_flops_per_sec / peak_flops, 1.0)
        except:
            return 0.0

    def get_lr(self, step: int) -> float:
        """Get learning rate using cosine schedule."""
        return cosine_learning_rate_schedule(
            iteration=step,
            max_learning_rate=self.args.max_learning_rate,
            min_learning_rate=self.args.min_learning_rate,
            warmup_iters=self.args.warmup_iters,
            cosine_cycle_iters=self.args.cosine_cycle_iters,
        )

    def evaluate(self) -> tuple[float, float]:
        """Fast evaluation on validation set."""
        if self.val_loader is None:
            return float("inf"), float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 30  # Reduced for speed

        with torch.no_grad():
            for batch_idx in range(num_batches):
                try:
                    inputs, targets = self.val_loader.get_batch()
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                        logits = self.model(inputs)
                        loss = cross_entropy(logits, targets)

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                    else:
                        num_batches -= 1
                except Exception as e:
                    num_batches -= 1

        if num_batches == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))
        return avg_loss, perplexity

    def train_step(self) -> dict[str, Any]:
        """Optimized training step."""
        self.model.train()
        step_start_time = time.time()

        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for accum_step in range(self.args.gradient_accumulation_steps):
            inputs, targets = self.train_loader.get_batch()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                logits = self.model(inputs)
                loss = cross_entropy(logits, targets) / self.args.gradient_accumulation_steps

            total_loss += loss.item()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        # Optimizer step
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()

        step_time = time.time() - step_start_time

        return {
            "loss": total_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time": step_time,
        }

    def train(self):
        """Optimized training loop."""
        MAX_TRAINING_TIME = 1.5 * 3600

        print(f"\n🚀 Starting FAST training for {self.args.steps} steps")
        print(f"⏰ TIME LIMIT: 1.5 hours")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"⚡ Device: {self.device}")

        self.experiment_logger.add_note("Fast training started")
        self.training_integrator.start_epoch(0)

        # Initial evaluation
        val_loss, val_perplexity = self.evaluate()
        print(f"📊 Initial validation - loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")

        pbar = tqdm(range(self.args.steps), desc="🚀 Fast Training")
        training_start_time = time.time()
        best_val_loss = float("inf")

        for step in pbar:
            self.step = step

            # Check time limit
            elapsed_time = time.time() - training_start_time
            if elapsed_time >= MAX_TRAINING_TIME:
                print(f"\n⏰ TIME LIMIT REACHED: {elapsed_time / 3600:.2f}h")
                break

            metrics = self.train_step()

            # Calculate performance metrics
            elapsed_time = time.time() - training_start_time
            effective_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
            tokens_processed = (step + 1) * effective_batch_size * self.args.context_length
            tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
            mfu = self.calculate_mfu(tokens_per_sec)

            time_remaining = max(0, MAX_TRAINING_TIME - elapsed_time)

            # Log training metrics
            self.training_integrator.log_training_step(
                step=step,
                train_loss=metrics["loss"],
                learning_rate=metrics["lr"],
                tokens_processed=effective_batch_size * self.args.context_length,
                samples_processed=effective_batch_size,
                step_time=metrics["step_time"],
                tokens_per_sec=tokens_per_sec,
                wallclock_time=elapsed_time / 3600,
                grad_norm=metrics["grad_norm"],
                mfu=mfu,
                effective_batch_size=effective_batch_size,
                sequence_length=self.args.context_length,
            )

            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "MFU": f"{mfu:.3f}",
                    "best_val": f"{best_val_loss:.4f}",
                    "time_left": f"{time_remaining / 3600:.2f}h",
                }
            )

            # Validation
            if step > 0 and step % self.args.validation_step_interval == 0:
                val_loss, val_perplexity = self.evaluate()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.experiment_logger.add_note(f"🎯 NEW BEST validation loss: {val_loss:.4f}")

                print(
                    f"\n📊 Step {step}: train_loss={metrics['loss']:.4f}, "
                    f"val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}, "
                    f"MFU={mfu:.3f}, tok/s={tokens_per_sec:.0f}"
                )

                self.training_integrator.log_validation_step(
                    step=step,
                    val_loss=val_loss,
                    perplexity=val_perplexity,
                    wallclock_time=elapsed_time / 3600,
                    mfu_at_validation=mfu,
                    tokens_per_sec_at_validation=tokens_per_sec,
                )

            # Checkpointing
            if step > 0 and step % self.args.checkpoint_step_interval == 0:
                checkpoint_path = f"checkpoint_{self.experiment_timestamp}_step_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")

        # Final evaluation
        final_elapsed_time = time.time() - training_start_time
        val_loss, val_perplexity = self.evaluate()
        final_tokens_per_sec = ((step + 1) * effective_batch_size * self.args.context_length) / final_elapsed_time
        final_mfu = self.calculate_mfu(final_tokens_per_sec)

        print(f"\n🏁 FINAL RESULTS:")
        print(f"📊 Validation loss: {val_loss:.4f}")
        print(f"⚡ Training time: {final_elapsed_time / 3600:.2f}h")
        print(f"🚀 Final MFU: {final_mfu:.3f}")
        print(f"⚡ Tokens/sec: {final_tokens_per_sec:.0f}")
        print(f"📈 Total steps completed: {step + 1}")

        # Save final checkpoint
        final_checkpoint = f"final_checkpoint_{self.experiment_timestamp}_step_{step}.pt"
        save_checkpoint(self.model, self.optimizer, step, final_checkpoint)
        print(f"💾 Saved final checkpoint: {final_checkpoint}")

        self.experiment_logger.add_note(
            f"Training completed. Final val_loss: {val_loss:.4f}, MFU: {final_mfu:.3f}, Time: {final_elapsed_time / 3600:.2f}h"
        )
        self.experiment_logger.mark_completed(success=True)


def load_config_from_json(config_path: str) -> TrainArgs:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    from dataclasses import fields

    expected_fields = {field.name for field in fields(TrainArgs)}
    provided_fields = set(config_dict.keys())

    # Remove unexpected fields
    unexpected_fields = provided_fields - expected_fields
    if unexpected_fields:
        print(f"Warning: Unexpected fields in config will be ignored: {unexpected_fields}")
        config_dict = {k: v for k, v in config_dict.items() if k in expected_fields}

    try:
        return TrainArgs(**config_dict)
    except TypeError as e:
        raise TypeError(f"Error creating TrainArgs from config: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast Transformer Training")

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["reduce-overhead", "max-autotune", "default"],
        help="Compilation mode",
    )
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")

    args = parser.parse_args()

    if args.config:
        config = load_config_from_json(args.config)
        if hasattr(config, "compile_mode"):
            config.compile_mode = args.compile_mode
        if args.no_compile:
            config.compile_model = False
    else:
        config = TrainArgs(compile_mode=args.compile_mode, compile_model=not args.no_compile)

    print("🚀 FAST TRAINING INITIATED")
    print(f"⚡ Compilation mode: {config.compile_mode}")

    trainer = FastTrainer(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        trainer.experiment_logger.add_note("Training interrupted by user")
        trainer.experiment_logger.mark_completed(success=False)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.experiment_logger.add_note(f"Training failed with error: {e}")
        trainer.experiment_logger.mark_completed(success=False)
        raise


if __name__ == "__main__":
    main()
