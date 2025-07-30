"""
Optimized Transformer Training Script for H100 GPU

Implements:
- Mixed precision training with bfloat16
- U-Net transformer architecture
- Enhanced optimizers with parameter groups
- Optimized data loading
- Advanced torch.compile settings
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.enhanced_models import UNetTransformerLM
from cs336_basics.training.checkpoint import save_checkpoint
from cs336_basics.training.enhanced_optimizers import EnhancedAdam, LionOptimizer
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule


@dataclass
class OptimizedTrainArgs:
    """Optimized training configuration for H100."""

    # Model parameters (optimized for H100)
    vocab_size: int = 32000
    context_length: int = 512  # Increased from 256
    num_layers: int = 16  # Increased from 4
    d_model: int = 1024  # Increased from 512
    num_heads: int = 8  # Decreased from 16 for better head dimension
    d_ff: int = 2816  # ~2.75x d_model for squared ReLU
    rope_theta: float = 10000
    use_squared_relu: bool = True
    tie_embeddings: bool = False  # Untied embeddings

    # Optimizer parameters
    optimizer_type: str = "enhanced_adam"  # "enhanced_adam" or "lion"
    weight_decay: float = 0.1  # Increased weight decay
    betas: tuple[float, float] = (0.9, 0.95)  # Lower beta2 for faster adaptation
    base_lr: float = 3e-3  # Higher base learning rate

    # Learning rate schedule
    min_learning_rate: float = 3e-5
    warmup_iters: int = 1000  # Faster warmup
    cosine_cycle_iters: int = 20000  # Target ~20k iterations

    # Data paths
    training_set: str = "data/encoded/owt_train_tokens.npy"
    validation_set: str = "data/encoded/owt_valid_tokens.npy"

    # Training configuration (optimized for H100)
    validation_step_interval: int = 250
    checkpoint_step_interval: int = 5000
    steps: int = 20000  # Reduced from 40960
    batch_size: int = 128  # Increased from 32
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0

    # Mixed precision and performance
    use_bfloat16: bool = True
    use_amp: bool = False  # Use manual bfloat16 instead of AMP for better control
    compile_model: bool = True
    compile_mode: str = "max-autotune"  # "default", "reduce-overhead", or "max-autotune"
    use_flash_attention: bool = True
    use_fused_optimizers: bool = True

    # Device and optimization
    device: str = "cuda"
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    # Experiment logging
    experiment_name: str = "optimized_transformer_h100"
    experiment_description: str = "Optimized transformer for H100 with U-Net architecture"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1-optimized"
    wandb_entity: str = ""
    log_dir: str = "experiments"

    def __post_init__(self):
        """Validate and adjust configuration."""
        assert self.vocab_size > 0
        assert self.context_length > 0
        assert self.d_model > 0
        assert self.d_model % self.num_heads == 0
        assert self.steps > 0
        assert self.batch_size > 0
        assert self.base_lr > 0

        # Adjust d_ff to be divisible by 64 for tensor core efficiency
        if self.d_ff % 64 != 0:
            self.d_ff = ((self.d_ff + 63) // 64) * 64


class OptimizedTrainer:
    """Optimized trainer with mixed precision and performance enhancements."""

    def __init__(self, args: OptimizedTrainArgs):
        self.args = args
        self.step = 0
        self.device = torch.device(args.device)

        # Set up CUDA optimizations
        if torch.cuda.is_available():
            # Get device index - default to 0 if not specified
            if self.device.type == "cuda":
                device_idx = self.device.index if self.device.index is not None else 0
                torch.cuda.set_device(device_idx)

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Enable flash attention if available
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                torch.backends.cuda.enable_flash_sdp(args.use_flash_attention)
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Note: We don't set default dtype to bfloat16 globally
        # Instead, we'll be explicit about dtypes to avoid issues with indexing

        # Initialize logging
        self.experiment_logger = ExperimentLogger(
            experiment_name=args.experiment_name,
            description=args.experiment_description,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity if args.wandb_entity else None,
        )

        self.experiment_logger.log_hyperparameters(**asdict(args))

        self.training_integrator = TrainingIntegrator(
            experiment_logger=self.experiment_logger,
            hardware_log_interval=50,
        )

        # Initialize model with bfloat16 if enabled
        dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
        self.model = UNetTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            use_squared_relu=args.use_squared_relu,
            tie_embeddings=args.tie_embeddings,
            device=self.device,
            dtype=dtype,
        )

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.experiment_logger.add_note(
            f"Model initialized: {total_params:,} total parameters, {trainable_params:,} trainable"
        )
        print(f"Model size: {total_params:,} parameters")

        # Compile model with optimized settings
        if args.compile_model and self.device.type == "cuda":
            try:
                # Set torch._dynamo config to handle edge cases better
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True

                compile_kwargs = {
                    "mode": args.compile_mode,
                    "fullgraph": False,  # Allow graph breaks for better compatibility
                    "dynamic": True,  # Allow dynamic shapes
                }

                # Only compile the main forward function
                self.model = torch.compile(self.model, **compile_kwargs)

                self.experiment_logger.add_note(f"Model compiled successfully with mode: {args.compile_mode}")
                print(f"✅ Model compiled with {args.compile_mode} mode")
            except Exception as e:
                self.experiment_logger.add_note(f"Model compilation failed: {e}")
                print(f"⚠️ Model compilation failed: {e}")
                print(f"⚠️ Continuing without compilation...")

        # Initialize optimizer with parameter groups
        param_groups = self.model.get_parameter_groups(base_lr=args.base_lr)

        if args.optimizer_type == "enhanced_adam":
            self.optimizer = EnhancedAdam(
                param_groups,
                lr=args.base_lr,
                betas=args.betas,
                weight_decay=args.weight_decay,
                momentum_scale=True,
                adaptive_eps=True,
            )
        elif args.optimizer_type == "lion":
            self.optimizer = LionOptimizer(
                param_groups,
                lr=args.base_lr,
                betas=args.betas,
                weight_decay=args.weight_decay,
            )
        else:
            # Fallback to standard Adam
            from cs336_basics.training.optimizers import Adam

            self.optimizer = Adam(
                param_groups,
                lr=args.base_lr,
                betas=args.betas,
                weight_decay=args.weight_decay,
            )

        # Log optimizer info
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            num_params = sum(p.numel() for p in group["params"])
            self.experiment_logger.add_note(f"Optimizer {name}: {num_params:,} params, lr={group['lr']:.2e}")

        # Initialize gradient scaler for mixed precision (if using AMP)
        self.scaler = GradScaler() if args.use_amp else None

        # Load data with memory mapping
        print(f"Loading training data from {args.training_set}")
        self.training_set = np.load(args.training_set, mmap_mode="r")
        print(f"Training set size: {len(self.training_set):,} tokens")
        self.experiment_logger.add_note(f"Training data loaded: {len(self.training_set):,} tokens")

        if Path(args.validation_set).exists():
            print(f"Loading validation data from {args.validation_set}")
            self.validation_set = np.load(args.validation_set, mmap_mode="r")
            print(f"Validation set size: {len(self.validation_set):,} tokens")
            self.experiment_logger.add_note(f"Validation data loaded: {len(self.validation_set):,} tokens")
        else:
            self.validation_set = None
            self.experiment_logger.add_note("No validation set found")
            print("No validation set found")

        # Effective batch size for gradient accumulation
        self.effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    def calculate_mfu(self, tokens_per_sec: float) -> float:
        """Calculate Model FLOPs Utilization for H100."""
        try:
            N = sum(p.numel() for p in self.model.parameters())
            L = self.args.num_layers
            H = self.args.d_model
            Q = self.args.context_length

            # Approximate FLOPs per token (forward pass)
            # Attention: 4 * L * H * Q (Q, K, V projections + output)
            # FFN: 4 * L * H * d_ff (two linear layers)
            # Plus additional ops
            flops_per_token_forward = 6 * N + 12 * L * H * Q

            # Training includes forward + backward + optimizer step
            flops_per_token_training = 3 * flops_per_token_forward

            model_flops_per_sec = tokens_per_sec * flops_per_token_training

            # H100 SXM peak FP16/BF16 performance
            if self.args.use_bfloat16:
                peak_flops = 989e12  # H100 BF16 tensor core peak
            else:
                peak_flops = 1979e12  # H100 FP16 tensor core peak

            mfu = min(model_flops_per_sec / peak_flops, 1.0)
            return mfu

        except Exception as e:
            self.experiment_logger.add_note(f"MFU calculation failed: {e}")
            return 0.0

    def get_lr(self, step: int) -> float:
        """Get learning rate using cosine schedule."""
        return cosine_learning_rate_schedule(
            iteration=step,
            max_learning_rate=self.args.base_lr,
            min_learning_rate=self.args.min_learning_rate,
            warmup_iters=self.args.warmup_iters,
            cosine_cycle_iters=self.args.cosine_cycle_iters,
        )

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluate the model on validation set."""
        if self.validation_set is None:
            return float("inf"), float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 50  # Fixed number of validation batches

        eval_start_time = time.time()

        for batch_idx in range(num_batches):
            try:
                inputs, targets = get_batch(
                    self.validation_set, self.args.batch_size, self.args.context_length, device=str(self.device)
                )

                # Note: inputs should remain as long integers for indexing
                # Only convert targets to bfloat16 if needed
                # inputs stay as torch.long for embedding indexing

                logits = self.model(inputs)
                loss = cross_entropy(logits, targets)

                if torch.isfinite(loss):
                    total_loss += loss.item()
                else:
                    num_batches -= 1

            except Exception as e:
                self.experiment_logger.add_note(f"Evaluation batch {batch_idx} failed: {e}")
                print(f"⚠️ Evaluation batch failed: {e}")
                num_batches -= 1

        eval_time = time.time() - eval_start_time

        if num_batches == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))

        eval_tokens_per_sec = (num_batches * self.args.batch_size * self.args.context_length) / eval_time
        self.experiment_logger.add_note(
            f"Evaluation completed: {num_batches} batches, {eval_tokens_per_sec:.0f} tokens/sec"
        )

        self.model.train()
        return avg_loss, perplexity

    def train_step(self) -> dict[str, Any]:
        """Single training step with mixed precision."""
        self.model.train()
        step_start_time = time.time()

        # Update learning rates for all parameter groups
        lr = self.get_lr(self.step)
        for i, param_group in enumerate(self.optimizer.param_groups):
            # Scale learning rate based on group
            group_name = param_group.get("name", "")
            if "embeddings" in group_name:
                param_group["lr"] = lr * 5.0  # Higher LR for embeddings
            elif "lm_head" in group_name:
                param_group["lr"] = lr * 0.5  # Lower LR for lm_head
            elif "1d_params" in group_name:
                param_group["lr"] = lr * 2.0  # Higher LR for 1D params
            else:
                param_group["lr"] = lr

        # Gradient accumulation
        accumulated_loss = 0.0
        self.optimizer.zero_grad()

        for micro_step in range(self.args.gradient_accumulation_steps):
            # Get batch
            batch_start_time = time.time()
            inputs, targets = get_batch(
                self.training_set, self.args.batch_size, self.args.context_length, device=str(self.device)
            )
            batch_time = time.time() - batch_start_time

            # Mixed precision forward pass
            forward_start_time = time.time()
            if self.args.use_amp:
                with autocast(dtype=torch.bfloat16):
                    logits = self.model(inputs)
                    loss = cross_entropy(logits, targets)
                loss = loss / self.args.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                # Manual bfloat16
                # Note: inputs must remain as long integers for embedding indexing
                # Do NOT convert inputs to bfloat16

                logits = self.model(inputs)
                loss = cross_entropy(logits, targets)
                loss = loss / self.args.gradient_accumulation_steps

                backward_start_time = time.time()
                loss.backward()
                backward_time = time.time() - backward_start_time

            forward_time = time.time() - forward_start_time - (backward_time if not self.args.use_amp else 0)
            accumulated_loss += loss.item()

        # Gradient clipping and optimizer step
        if self.args.use_amp:
            self.scaler.unscale_(self.optimizer)
            grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)

            optimizer_start_time = time.time()
            self.optimizer.step()
            optimizer_time = time.time() - optimizer_start_time

        step_time = time.time() - step_start_time

        return {
            "loss": accumulated_loss * self.args.gradient_accumulation_steps,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time": step_time,
            "batch_time": batch_time,
            "forward_time": forward_time,
            "backward_time": backward_time if not self.args.use_amp else 0,
            "optimizer_time": optimizer_time if not self.args.use_amp else 0,
        }

    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting optimized training for {self.args.steps} steps")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {'bfloat16' if self.args.use_bfloat16 else 'float32'}")
        print(
            f"Batch size: {self.args.batch_size} x {self.args.gradient_accumulation_steps} = {self.effective_batch_size}"
        )

        self.experiment_logger.add_note("Training started")
        self.training_integrator.start_epoch(0)

        # Initial evaluation
        val_loss, val_perplexity = self.evaluate()
        print(f"Initial validation - loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")

        self.training_integrator.log_validation_step(
            step=0,
            val_loss=val_loss,
            perplexity=val_perplexity,
            wallclock_time=0.0,
        )

        pbar = tqdm(range(self.args.steps), desc="Training")
        training_start_time = time.time()

        for step in pbar:
            self.step = step

            # Training step
            metrics = self.train_step()

            # Calculate throughput metrics
            elapsed_time = time.time() - training_start_time
            tokens_processed = (step + 1) * self.effective_batch_size * self.args.context_length
            tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
            mfu = self.calculate_mfu(tokens_per_sec)

            # Log metrics
            self.training_integrator.log_training_step(
                step=step,
                train_loss=metrics["loss"],
                learning_rate=metrics["lr"],
                tokens_processed=self.effective_batch_size * self.args.context_length,
                samples_processed=self.effective_batch_size,
                step_time=metrics["step_time"],
                tokens_per_sec=tokens_per_sec,
                wallclock_time=elapsed_time / 3600,
                grad_norm=metrics["grad_norm"],
                mfu=mfu,
                batch_time=metrics["batch_time"],
                forward_time=metrics["forward_time"],
                backward_time=metrics["backward_time"],
                optimizer_time=metrics["optimizer_time"],
                effective_batch_size=self.effective_batch_size,
                sequence_length=self.args.context_length,
            )

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "MFU": f"{mfu:.3f}",
                    "grad_norm": f"{metrics['grad_norm']:.3f}",
                }
            )

            # Validation
            if step > 0 and step % self.args.validation_step_interval == 0:
                val_loss, val_perplexity = self.evaluate()

                print(
                    f"\nStep {step}: train_loss={metrics['loss']:.4f}, "
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

                # Early stopping if we hit target
                if val_loss < 3.0781:
                    print(f"🎉 Target validation loss achieved: {val_loss:.4f} < 3.0781")
                    self.experiment_logger.add_note(f"Target achieved at step {step}: val_loss={val_loss:.4f}")
                    break

            # Checkpointing
            if step > 0 and step % self.args.checkpoint_step_interval == 0:
                checkpoint_path = f"checkpoint_optimized_step_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)
                self.experiment_logger.add_note(f"Checkpoint saved: {checkpoint_path}")
                print(f"Saved checkpoint: {checkpoint_path}")

        # Final evaluation
        final_elapsed_time = time.time() - training_start_time
        val_loss, val_perplexity = self.evaluate()
        final_tokens_per_sec = (
            self.args.steps * self.effective_batch_size * self.args.context_length
        ) / final_elapsed_time
        final_mfu = self.calculate_mfu(final_tokens_per_sec)

        print(f"\nFinal evaluation - val_loss: {val_loss:.4f}, val_perplexity: {val_perplexity:.2f}")
        print(
            f"Training completed - {final_elapsed_time / 3600:.2f} hours, {final_tokens_per_sec:.0f} tok/s, MFU: {final_mfu:.3f}"
        )

        self.training_integrator.log_validation_step(
            step=self.args.steps,
            val_loss=val_loss,
            perplexity=val_perplexity,
            wallclock_time=final_elapsed_time / 3600,
            final_mfu=final_mfu,
            final_tokens_per_sec=final_tokens_per_sec,
            training_hours=final_elapsed_time / 3600,
        )

        # Save final checkpoint
        final_checkpoint = f"final_optimized_checkpoint_step_{self.args.steps}.pt"
        save_checkpoint(self.model, self.optimizer, self.args.steps, final_checkpoint)
        print(f"Saved final checkpoint: {final_checkpoint}")

        self.experiment_logger.add_note(
            f"Training completed successfully. Final metrics: "
            f"val_loss={val_loss:.4f}, MFU={final_mfu:.3f}, "
            f"tokens/sec={final_tokens_per_sec:.0f}"
        )
        self.experiment_logger.mark_completed(success=True)


def load_config_from_json(config_path: str) -> OptimizedTrainArgs:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return OptimizedTrainArgs(**config_dict)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimized Transformer Training for H100")

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Base learning rate")
    parser.add_argument("--optimizer", type=str, default="enhanced_adam", choices=["enhanced_adam", "lion", "adam"])

    parser.add_argument("--experiment_name", type=str, default="optimized_transformer_h100")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1-optimized")
    parser.add_argument("--wandb_entity", type=str, default="", help="Wandb entity")
    parser.add_argument("--log_dir", type=str, default="experiments", help="Local logging directory")
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--no_bfloat16", action="store_true", help="Disable bfloat16 training")

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config_from_json(args.config)
        # Override with command line arguments
        if args.train_data:
            config.training_set = args.train_data
        if args.val_data:
            config.validation_set = args.val_data
        if args.optimizer != "enhanced_adam":
            config.optimizer_type = args.optimizer
    else:
        config = OptimizedTrainArgs(
            training_set=args.train_data or "data/encoded/owt_train_tokens.npy",
            validation_set=args.val_data or "data/encoded/owt_valid_tokens.npy",
            steps=args.steps,
            batch_size=args.batch_size,
            base_lr=args.lr,
            optimizer_type=args.optimizer,
            experiment_name=args.experiment_name,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            log_dir=args.log_dir,
            compile_model=not args.no_compile,
            use_bfloat16=not args.no_bfloat16,
        )

    # Initialize and run trainer
    trainer = OptimizedTrainer(config)
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
