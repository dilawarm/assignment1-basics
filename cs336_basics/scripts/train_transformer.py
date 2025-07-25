"""
Training Script for Transformer Language Model

- Linear Decay to Zero (D2Z) learning rate schedule (ICLR 2025)
- Custom FFN activation: w2(max(w1(x), 0)^2)
- U-Net architecture with learnable skip connections
- MixedOptimizerV2: Muon for linear weights, Adam for embeddings/lm_head/1D
- Untied embeddings with different learning rates
- Advanced H100 optimizations for maximum throughput
- Enhanced memory management and gradient checkpointing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import EnhancedTransformerLM
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import linear_decay_to_zero_schedule, warmup_schedule
from cs336_basics.training.optimizers import Adam, AdamW, MixedOptimizerV2, Muon


@dataclass
class TrainingConfig:
    """
    Configuration.
    """

    # Data parameters
    train_data_path: str
    val_data_path: str | None = None
    vocab_size: int = 32000
    context_length: int = 512

    # Model parameters
    d_model: int = 1024
    num_layers: int = 16
    num_heads: int = 8
    d_ff: int = 4096
    rope_theta: float = 10000.0
    eps: float = 1e-5

    # Architecture features
    tie_embeddings: bool = False
    activation: str = "custom"
    use_unet_architecture: bool = True

    # Training parameters
    max_steps: int = 20000
    max_wallclock_hours: float = 1.5
    batch_size: int = 128
    gradient_accumulation_steps: int = 4

    # Optimizer settings
    optimizer: str = "mixed_v2"
    lr_schedule: str = "linear_decay_to_zero"
    learning_rate: float = 8e-3
    muon_lr: float = 8e-3
    adam_lr: float = 6e-3
    embedding_lr: float = 12e-3
    lm_head_lr: float = 4e-3
    min_learning_rate: float = 0.0
    warmup_steps: int = 100
    weight_decay: float = 0.015

    # Muon-specific parameters
    momentum: float = 0.97
    ns_iters: int = 5

    # Adam-specific parameters
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # H100 optimization settings
    use_amp: bool = True
    use_bfloat16: bool = True
    use_gradient_checkpointing: bool = True
    gradient_checkpointing_layers: int = 12
    use_tf32: bool = True
    compile_model: bool = True
    torch_compile_backend: str = "inductor"
    torch_empty_cache_steps: int = 50
    channels_last: bool = False

    # Data loading optimizations
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    dataloader_drop_last: bool = True

    # Logging and evaluation
    log_interval: int = 20
    eval_interval: int = 200
    eval_batches: int = 50
    save_interval: int = 1000

    # Directories and experiment tracking
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "openwebtext_h100_v2"
    experiment_description: str = "OpenWebText training"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1"

    # Device settings
    device: str = "cuda"
    resume_from: str | None = None
    auto_resume: bool = False

    def __post_init__(self) -> None:
        """Validate and optimize configuration."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.max_wallclock_hours > 0, "max_wallclock_hours must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.d_ff % 64 != 0:
            self.d_ff = ((self.d_ff + 63) // 64) * 64
            warnings.warn(f"Adjusted d_ff to {self.d_ff} for optimal tensor core usage")

        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.total_tokens = self.effective_batch_size * self.max_steps * self.context_length


class DataLoader:
    """High-performance data loader for H100."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        context_length: int,
        device: str,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        drop_last: bool = True,
    ) -> None:
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.data_size = len(self.data)

        if self.data_size < context_length + 1:
            raise ValueError(f"Dataset too small: {self.data_size} < {context_length + 1}")

        print(f"Loaded dataset with {self.data_size:,} tokens from {data_path}")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch with memory access patterns."""
        return get_batch(
            dataset=self.data.astype(np.int_),
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=self.device,
        )


class Trainer:
    """
    Trainer
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize trainer."""
        self.config = config
        self.step = 0
        self.start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_experiment_name = f"{config.experiment_name}_{timestamp}"

        self.experiment_logger = ExperimentLogger(
            experiment_name=timestamped_experiment_name,
            description=config.experiment_description,
            log_dir="experiments",
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )

        self.experiment_logger.log_hyperparameters(**asdict(config))

        self.training_integrator = TrainingIntegrator(
            self.experiment_logger,
            hardware_log_interval=self.config.log_interval,
        )

        self._setup_device()
        self._setup_model()
        self._setup_optimizer()
        self._setup_amp()
        self._setup_data_loaders()

        if config.resume_from or config.auto_resume:
            self._try_resume()

        param_counts = self.model.count_parameters()
        memory_stats = self.model.get_memory_stats()

        print(f"\n🚀 TRAINER INITIALIZED 🚀")
        print(f"Model: {param_counts['total']:,} total parameters")
        print(f"Trainable: {param_counts['trainable']:,} parameters")
        print(f"Model memory: {memory_stats.get('parameter_memory_gb', 0):.2f} GB")
        print(f"Architecture: U-Net={self.config.use_unet_architecture}, Activation={self.config.activation}")
        print(f"Optimizer: {self.config.optimizer} with D2Z schedule")
        print(f"Training target: <3.0781 validation loss in {self.config.max_wallclock_hours}h")
        print(f"Effective batch size: {config.effective_batch_size}")
        print("=" * 80)

    def _setup_device(self) -> None:
        """Setup device with H100-specific optimizations."""
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            self.config.use_tf32 = False
            self.config.compile_model = False
            self.config.use_amp = False
        else:
            self.device = torch.device(self.config.device)

        if self.device.type == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✅ Enabled TF32 for maximum H100 throughput")

            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
                print("✅ Enabled FlashAttention backend")

            torch.backends.cuda.enable_mem_efficient_sdp(True)

            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    def _setup_model(self) -> None:
        """Setup model."""
        self.model = EnhancedTransformerLM(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            rope_theta=self.config.rope_theta,
            eps=self.config.eps,
            tie_embeddings=self.config.tie_embeddings,
            activation=self.config.activation,
            use_unet_architecture=self.config.use_unet_architecture,
            device=self.device,
        )

        if self.config.use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing(self.config.gradient_checkpointing_layers)
            print(f"✅ Enabled gradient checkpointing for {self.config.gradient_checkpointing_layers} layers")

        self.original_model = self.model

        if self.config.compile_model and self.device.type == "cuda":
            try:
                print("⚡ Compiling model for execution...")
                self.model = torch.compile(self.model, mode="max-autotune", backend=self.config.torch_compile_backend)
                print(f"✅ Model compiled with {self.config.torch_compile_backend} backend")
            except Exception as e:
                print(f"❌ Model compilation failed: {e}")
                print("Falling back to non-compiled model...")
                self.config.compile_model = False

    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.config.optimizer == "mixed_v2":
            self.optimizer = MixedOptimizerV2(
                model=self.original_model,
                muon_lr=self.config.muon_lr,
                adam_lr=self.config.adam_lr,
                embedding_lr=self.config.embedding_lr,
                lm_head_lr=self.config.lm_head_lr,
                muon_momentum=self.config.momentum,
                adam_betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
                ns_iters=self.config.ns_iters,
                use_optimized_muon=True,
            )
            print("✅ Using MixedOptimizerV2 (Muon + Adam)")

        elif self.config.optimizer == "adam":
            self.optimizer = Adam(
                self.original_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
            )
            print("✅ Using Adam optimizer")

        elif self.config.optimizer == "muon":
            self.optimizer = Muon(
                self.original_model.parameters(),
                lr=self.config.muon_lr,
                momentum=self.config.momentum,
                ns_iters=self.config.ns_iters,
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
                use_optimized_coefficients=True,
            )
            print("✅ Using Muon optimizer")

        else:
            self.optimizer = AdamW(
                self.original_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
            )
            print("✅ Using AdamW optimizer")

    def _setup_amp(self) -> None:
        """Setup Automatic Mixed Precision."""
        if self.config.use_amp and self.device.type == "cuda":
            if self.config.use_bfloat16:
                self.scaler = None
                self.amp_dtype = torch.bfloat16
                print("✅ Enabled AMP with bfloat16")
            else:
                self.scaler = GradScaler()
                self.amp_dtype = torch.float16
                print("✅ Enabled AMP with float16")
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

    def _setup_data_loaders(self) -> None:
        """Setup data loaders."""
        self.train_loader = DataLoader(
            data_path=self.config.train_data_path,
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
            device=str(self.device),
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=self.config.dataloader_drop_last,
        )

        self.val_loader = None
        if self.config.val_data_path and Path(self.config.val_data_path).exists():
            self.val_loader = DataLoader(
                data_path=self.config.val_data_path,
                batch_size=self.config.batch_size,
                context_length=self.config.context_length,
                device=str(self.device),
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor,
                drop_last=False,
            )

    def _try_resume(self) -> None:
        """Try to resume from checkpoint."""
        checkpoint_path = None

        if self.config.resume_from and Path(self.config.resume_from).exists():
            checkpoint_path = self.config.resume_from
        elif self.config.auto_resume:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
                if checkpoints:
                    checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))

        if checkpoint_path:
            try:
                self.step = load_checkpoint(checkpoint_path, self.original_model, self.optimizer)
                print(f"✅ Resumed from {checkpoint_path} at step {self.step}")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")

    def get_lr(self, step: int) -> dict[str, float]:
        """Get learning rate for current step using D2Z schedule."""
        if self.config.lr_schedule == "linear_decay_to_zero":
            base_lr = linear_decay_to_zero_schedule(
                iteration=step,
                max_learning_rate=self.config.learning_rate,
                warmup_iters=self.config.warmup_steps,
                total_iters=self.config.max_steps,
            )
        else:
            base_lr = warmup_schedule(
                iteration=step,
                max_learning_rate=self.config.learning_rate,
                warmup_iters=self.config.warmup_steps,
                total_iters=self.config.max_steps,
                warmup_type="linear",
            )

        lr_factor = base_lr / self.config.learning_rate if self.config.learning_rate > 0 else 0

        return {
            "base_lr": base_lr,
            "muon_lr": self.config.muon_lr * lr_factor,
            "adam_lr": self.config.adam_lr * lr_factor,
            "embedding_lr": self.config.embedding_lr * lr_factor,
            "lm_head_lr": self.config.lm_head_lr * lr_factor,
        }

    def train_step(self) -> dict[str, Any]:
        """Training step."""
        self.model.train()
        total_loss = 0.0

        lrs = self.get_lr(self.step)

        if self.config.optimizer == "mixed_v2":
            muon_lr_factor = lrs["muon_lr"] / self.config.muon_lr if self.config.muon_lr > 0 else 0
            adam_lr_factor = lrs["adam_lr"] / self.config.adam_lr if self.config.adam_lr > 0 else 0
            embedding_lr_factor = lrs["embedding_lr"] / self.config.embedding_lr if self.config.embedding_lr > 0 else 0
            lm_head_lr_factor = lrs["lm_head_lr"] / self.config.lm_head_lr if self.config.lm_head_lr > 0 else 0

            self.optimizer.update_learning_rates(muon_lr_factor, adam_lr_factor, embedding_lr_factor, lm_head_lr_factor)
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrs["base_lr"]

        self.optimizer.zero_grad()

        for accumulation_step in range(self.config.gradient_accumulation_steps):
            try:
                inputs, targets = self.train_loader.get_batch()

                if self.config.use_amp:
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        logits = self.model(inputs)
                        loss = cross_entropy(logits, targets)
                        loss = loss / self.config.gradient_accumulation_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ NaN/Inf loss detected at step {self.step}! Stopping training.")
                        return {"loss": float("nan"), "lr": lrs["base_lr"], "training_stopped": True}

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    logits = self.model(inputs)
                    loss = cross_entropy(logits, targets)
                    loss = loss / self.config.gradient_accumulation_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ NaN/Inf loss detected at step {self.step}! Stopping training.")
                        return {"loss": float("nan"), "lr": lrs["base_lr"], "training_stopped": True}

                    loss.backward()

                total_loss += loss.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  OOM during accumulation step {accumulation_step}. Clearing cache...")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    if accumulation_step > 0:
                        total_loss = total_loss * self.config.gradient_accumulation_steps / accumulation_step
                        break
                    else:
                        return {"loss": float("nan"), "lr": lrs["base_lr"], "training_stopped": True}
                else:
                    raise e

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            gradient_clipping(self.original_model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            gradient_clipping(self.original_model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

        if self.config.torch_empty_cache_steps > 0 and self.step % self.config.torch_empty_cache_steps == 0:
            torch.cuda.empty_cache()

        return {
            "loss": total_loss,
            "lr": lrs["base_lr"],
            **lrs,
        }

    def evaluate(self) -> dict[str, Any]:
        """Evaluation with mixed precision."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _ in range(self.config.eval_batches):
                try:
                    inputs, targets = self.val_loader.get_batch()

                    if self.config.use_amp:
                        with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            logits = self.model(inputs)
                            loss = cross_entropy(logits, targets)
                    else:
                        logits = self.model(inputs)
                        loss = cross_entropy(logits, targets)

                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"⚠️  Evaluation batch failed: {e}")
                    break

        if num_batches == 0:
            return {}

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        save_checkpoint(self.original_model, self.optimizer, self.step, path)

    def train(self) -> None:
        """Main training loop."""
        max_time_seconds = self.config.max_wallclock_hours * 3600
        print(f"\n🚀 STARTING TRAINING 🚀")
        print(f"Target: <3.0781 validation loss in {self.config.max_wallclock_hours:.1f} hours")
        print(f"Schedule: {self.config.lr_schedule} with {self.config.warmup_steps} warmup steps")
        print(f"Optimizer: {self.config.optimizer}")
        print("=" * 80)

        self.training_integrator.start_epoch(0)

        best_val_loss = float("inf")
        last_eval_time = self.start_time

        while self.step < self.config.max_steps:
            step_start_time = time.time()
            elapsed_time = step_start_time - self.start_time

            if elapsed_time >= max_time_seconds:
                print(f"⏰ Reached wallclock time limit of {self.config.max_wallclock_hours:.1f} hours")
                break

            metrics = self.train_step()

            if metrics.get("training_stopped", False):
                print("❌ Training stopped due to NaN/Inf values")
                break

            step_time = time.time() - step_start_time
            elapsed_hours = elapsed_time / 3600
            steps_per_sec = (self.step + 1) / elapsed_time if elapsed_time > 0 else 0
            tokens_per_sec = steps_per_sec * self.config.effective_batch_size * self.config.context_length

            # Logging
            if self.step % self.config.log_interval == 0:
                tokens_this_step = self.config.effective_batch_size * self.config.context_length
                samples_this_step = self.config.effective_batch_size

                self.training_integrator.log_training_step(
                    wallclock_time=elapsed_hours,
                    step=self.step,
                    train_loss=metrics["loss"],
                    learning_rate=metrics["lr"],
                    tokens_processed=tokens_this_step,
                    samples_processed=samples_this_step,
                    step_time=step_time,
                    tokens_per_sec=tokens_per_sec,
                )

                remaining_hours = self.config.max_wallclock_hours - elapsed_hours
                print(
                    f"Step {self.step:5d}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}, "
                    f"{tokens_per_sec:.0f} tok/s, Time: {elapsed_hours:.2f}h/{self.config.max_wallclock_hours:.1f}h"
                )

            if self.step % self.config.eval_interval == 0 and self.step > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    val_loss = eval_metrics["loss"]

                    self.training_integrator.log_validation_step(
                        wallclock_time=elapsed_hours,
                        step=self.step,
                        val_loss=val_loss,
                        perplexity=eval_metrics["perplexity"],
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    status = "🔥 NEW BEST!" if val_loss < best_val_loss else ""

                    print(
                        f"Eval Step {self.step}: val_loss={val_loss:.4f}, perplexity={eval_metrics['perplexity']:.2f} {status}"
                    )

            if self.step % self.config.save_interval == 0 and self.step > 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.step}.pt"
                self.save_checkpoint(str(checkpoint_path))

            self.step += 1

        final_elapsed_hours = (time.time() - self.start_time) / 3600
        final_eval = self.evaluate()

        if final_eval:
            final_val_loss = final_eval["loss"]
            print(f"\n🏁 FINAL RESULTS:")
            print(f"Final validation loss: {final_val_loss:.4f}")
            print(f"Training time: {final_elapsed_hours:.2f} hours")
            print(f"Status: {'🎯 SUCCESS!' if final_val_loss < 3.0781 else '❌ Target not reached'}")

            self.training_integrator.log_validation_step(
                wallclock_time=final_elapsed_hours,
                step=self.step,
                val_loss=final_val_loss,
                perplexity=final_eval["perplexity"],
            )

        final_checkpoint = Path(self.config.checkpoint_dir) / f"checkpoint_final_step_{self.step}.pt"
        self.save_checkpoint(str(final_checkpoint))

        self.experiment_logger.add_note(f"Training completed in {final_elapsed_hours:.2f} hours")
        self.experiment_logger.mark_completed()


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Transformer Training")

    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--train_data", type=str, help="Path to training data (.npy)")
    parser.add_argument("--val_data", type=str, help="Path to validation data (.npy)")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum training steps")
    parser.add_argument("--max_hours", type=float, default=1.5, help="Maximum training time in hours")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-3, help="Learning rate")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        if args.train_data:
            config.train_data_path = args.train_data
        if args.val_data:
            config.val_data_path = args.val_data
    else:
        if not args.train_data:
            raise ValueError("Must specify either --config or --train_data")

        config = TrainingConfig(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            max_steps=args.max_steps,
            max_wallclock_hours=args.max_hours,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    if args.no_compile:
        config.compile_model = False
    if args.no_wandb:
        config.use_wandb = False

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
