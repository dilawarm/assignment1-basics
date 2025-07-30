"""
Transformer Training Script with Advanced Stability Features.

Includes:
- Ultra-stable training configuration
- Adaptive gradient clipping (ZClip/AdaGC)
- Outlier-safe Muon optimizer
- Proactive loss spike detection
- Memory optimizations for H100 GPU
"""

import argparse
import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import TransformerLM
from cs336_basics.training.checkpoint import save_checkpoint
from cs336_basics.training.gradient_clipping import StabilityMonitor, create_adaptive_clipper, safe_gradient_step
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.muon_optimizer import MuonAdamHybrid


@dataclass
class TrainArgs:
    """Ultra-stable training configuration."""

    # Model parameters
    vocab_size: int = 32000
    context_length: int = 1024
    num_layers: int = 16
    d_model: int = 1280
    num_heads: int = 20
    d_ff: int = 5120
    rope_theta: float = 10000.0
    window_size: int | None = None  # Remove sliding window for stability
    use_qk_norm: bool = True
    use_flex_attention: bool = False
    use_swiglu: bool = False
    tie_embeddings: bool = True

    # Optimizer parameters
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    muon_momentum: float = 0.95
    eps: float = 1e-8

    # Learning rate schedule (ultra-conservative)
    max_learning_rate: float = 0.0006
    min_learning_rate: float = 0.0001
    warmup_iters: int = 500
    cosine_cycle_iters: int = 5400

    # Data paths
    training_set: str = "data/encoded/owt_train_tokens.npy"
    validation_set: str = "data/encoded/owt_valid_tokens.npy"

    # Training configuration
    validation_step_interval: int = 100
    checkpoint_step_interval: int = 1000
    steps: int = 5400
    batch_size: int = 48
    gradient_accumulation_steps: int = 3
    gradient_clipping: float = 0.5

    # Optimization settings
    device: str = "cuda"
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    use_mixed_precision: bool = True
    use_efficient_attention: bool = True
    use_fused_kernels: bool = True

    # Ultra-stability configuration
    training_mode: str = "stable"
    enable_stability_monitoring: bool = True
    gradient_norm_threshold: float = 5.0
    loss_spike_threshold: float = 2.5
    nan_tolerance: int = 1
    enable_stable_initialization: bool = True

    # Adaptive gradient clipping
    use_adaptive_gradient_clipping: bool = True
    adaptive_clipping_method: str = "zclip"  # "zclip" or "adagc"
    zclip_zscore_threshold: float = 2.5
    zclip_min_clip: float = 0.1
    zclip_max_clip: float = 2.0
    zclip_warmup_steps: int = 200

    # Muon optimizer
    use_muon_optimizer: bool = True
    muon_ns_iters: int = 5
    enable_outlier_safe_training: bool = True

    # Performance optimization
    enable_torch_optimizations: bool = True
    eval_batch_count: int = 30

    # Memory optimization
    use_activation_checkpointing: bool = True
    checkpoint_pattern: str = "layers\\.([0-9]+)$"
    use_memory_efficient_attention: bool = True
    optimize_memory_layout: bool = True

    # Experiment logging
    experiment_name: str = "h100_ultra_stable_training"
    experiment_description: str = "Ultra-stable training with outlier-safe features"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1-stable"
    wandb_entity: str = ""
    log_dir: str = "experiments"

    def __post_init__(self):
        """Validate and adapt configuration."""
        assert self.vocab_size > 0
        assert self.context_length > 0
        assert self.d_model > 0
        assert self.d_model % self.num_heads == 0
        assert self.steps > 0
        assert self.batch_size > 0
        assert self.max_learning_rate > 0

        # Force stable training mode for safety
        if self.training_mode != "stable":
            warnings.warn("Forcing stable training mode for maximum safety")
            self.training_mode = "stable"

        # Remove sliding window to prevent instability
        if self.window_size is not None:
            warnings.warn("Removing sliding window for stability")
            self.window_size = None

        # Ensure vocab_size is efficient
        if self.vocab_size % 128 != 0:
            self.vocab_size = ((self.vocab_size + 127) // 128) * 128
            print(f"Padded vocab_size to {self.vocab_size} for efficiency")


class CheckpointModule(nn.Module):
    """Wrapper for activation checkpointing."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def apply_activation_checkpointing(module, pattern_re, ancestor_name=""):
    """Apply activation checkpointing to modules matching the pattern."""
    for name, submodule in module._modules.items():
        full_name = f"{ancestor_name}.{name}" if ancestor_name else name
        submodule = apply_activation_checkpointing(submodule, pattern_re, full_name)
        if pattern_re.match(full_name):
            print(f"🔧 Applying activation checkpointing to: {full_name}")
            setattr(module, name, CheckpointModule(submodule))
        else:
            setattr(module, name, submodule)
    return module


class DataLoader:
    """High-performance data loader with memory optimization."""

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
        """Get a batch with optimized memory access patterns."""
        return get_batch(self.data, self.batch_size, self.context_length, device=self.device)


class Trainer:
    """Ultra-stable trainer with all 2025 stability features."""

    def __init__(self, args: TrainArgs):
        self.args = args
        self.step = 0

        # Configure CUDA memory management
        if args.device == "cuda" and torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
            print("🔧 Configured CUDA memory allocation for better efficiency")

        if args.device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            args.device = "cpu"
            args.use_mixed_precision = False
            args.use_activation_checkpointing = False
        else:
            self.device = torch.device(args.device)

        # Initialize stability monitoring
        self.stability_monitor = StabilityMonitor(
            loss_history_size=50,
            spike_threshold=args.loss_spike_threshold,
            nan_tolerance=args.nan_tolerance,
            recovery_lr_scale=0.1,
        )

        # Initialize experiment logging
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
        print(f"🛡️ Training mode: ULTRA-STABLE")
        self.experiment_logger.log_hyperparameters(**asdict(args))

        self.training_integrator = TrainingIntegrator(
            experiment_logger=self.experiment_logger,
            hardware_log_interval=50,
        )

        print("🏗️ Initializing ultra-stable transformer model...")

        if args.enable_torch_optimizations and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("⚡ PyTorch optimizations enabled")

        # Initialize model with stability features
        self.model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            use_qk_norm=args.use_qk_norm,
            use_swiglu=args.use_swiglu,
            tie_embeddings=args.tie_embeddings,
            device=self.device,
        )

        # Apply activation checkpointing for memory efficiency
        if args.use_activation_checkpointing:
            checkpoint_pattern = re.compile(args.checkpoint_pattern)
            self.model = apply_activation_checkpointing(self.model, checkpoint_pattern)
            print(f"✅ Applied activation checkpointing with pattern: {args.checkpoint_pattern}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model: {total_params:,} parameters ({total_params / 1e6:.1f}M), {trainable_params:,} trainable")

        # Move model to device BEFORE compilation
        self.model = self.model.to(self.device)
        print(f"📍 Model moved to device: {self.device}")

        # Model compilation with stability
        if args.compile_model:
            self._compile_model()

        # Initialize optimizer
        print("🔧 Initializing ultra-stable optimizer...")
        if args.use_muon_optimizer:
            self.optimizer = MuonAdamHybrid(
                self.model.parameters(),
                lr=args.max_learning_rate,
                betas=args.betas,
                weight_decay=args.weight_decay,
                muon_momentum=args.muon_momentum,
                eps=args.eps,
                ns_iters=args.muon_ns_iters,
            )
            print("✅ Using MuonAdamHybrid optimizer for outlier-safe training")
        else:
            # Fallback to AdamW
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.max_learning_rate,
                betas=args.betas,
                weight_decay=args.weight_decay,
                eps=args.eps,
            )

        # Initialize adaptive gradient clipping
        if args.use_adaptive_gradient_clipping:
            clipper_kwargs = {}
            if args.adaptive_clipping_method == "zclip":
                clipper_kwargs = {
                    "zscore_threshold": args.zclip_zscore_threshold,
                    "min_clip_value": args.zclip_min_clip,
                    "max_clip_value": args.zclip_max_clip,
                    "warmup_steps": args.zclip_warmup_steps,
                }

            self.adaptive_clipper = create_adaptive_clipper(method=args.adaptive_clipping_method, **clipper_kwargs)
            print(f"✅ Using {args.adaptive_clipping_method.upper()} adaptive gradient clipping")
        else:
            self.adaptive_clipper = None

        # Initialize data loaders
        print(f"📚 Loading training data from {args.training_set}")
        self.train_loader = DataLoader(
            args.training_set,
            args.batch_size,
            args.context_length,
            device=args.device,  # Use device string directly instead of converting torch.device
        )

        if Path(args.validation_set).exists():
            print(f"📚 Loading validation data from {args.validation_set}")
            self.val_loader = DataLoader(
                args.validation_set,
                args.batch_size,
                args.context_length,
                device=args.device,  # Use device string directly instead of converting torch.device
            )
        else:
            self.val_loader = None
            print("❌ No validation set found")

        # Initialize mixed precision
        if self.device.type == "cpu" and args.use_mixed_precision:
            print("⚠️ Mixed precision not supported on CPU, disabling")
            args.use_mixed_precision = False

        self.scaler = torch.amp.GradScaler() if args.use_mixed_precision else None
        if self.scaler:
            print("⚡ Mixed precision training enabled")

        # Print configuration summary
        print("\n🛡️ ULTRA-STABLE CONFIGURATION ENABLED:")
        print(f"   ✅ Removed sliding window (was causing NaN)")
        print(f"   ✅ Conservative learning rate: {args.max_learning_rate}")
        print(f"   ✅ Adaptive gradient clipping: {args.adaptive_clipping_method.upper()}")
        print(f"   ✅ Muon optimizer: {'YES' if args.use_muon_optimizer else 'NO'}")
        print(f"   ✅ Stability monitoring: NaN tolerance = {args.nan_tolerance}")
        print(f"   📊 Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"   📊 Model size: {total_params / 1e6:.1f}M parameters")

        if self.device.type == "cuda":
            print(f"   📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            torch.cuda.reset_peak_memory_stats()

    def _compile_model(self):
        """Conservative model compilation."""
        print(f"⚡ Compiling model with {self.args.compile_mode} mode...")
        try:
            self.model = torch.compile(self.model, mode=self.args.compile_mode)
            print(f"✅ Model compiled successfully")
            self.experiment_logger.add_note(f"Model compiled with {self.args.compile_mode}")
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}, running in eager mode")
            self.experiment_logger.add_note(f"Compilation failed, running in eager mode: {e}")

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
                elif "V100" in gpu_name:
                    peak_flops = 125e12
                else:
                    peak_flops = 100e12
            else:
                peak_flops = 1e12

            return min(model_flops_per_sec / peak_flops, 1.0)
        except Exception as e:
            self.experiment_logger.add_note(f"MFU calculation failed: {e}")
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
        """Ultra-stable evaluation on validation set."""
        if self.val_loader is None:
            return float("inf"), float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = self.args.eval_batch_count

        eval_start_time = time.time()
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
                        print(f"⚠️ Non-finite loss in evaluation batch {batch_idx}")

                except Exception as e:
                    num_batches -= 1
                    print(f"⚠️ Evaluation batch {batch_idx} failed: {e}")
                    self.experiment_logger.add_note(f"Evaluation batch {batch_idx} failed: {e}")

        eval_time = time.time() - eval_start_time

        if num_batches == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))

        eval_tokens_per_sec = (num_batches * self.args.batch_size * self.args.context_length) / eval_time
        self.experiment_logger.add_note(f"Evaluation: {num_batches} batches, {eval_tokens_per_sec:.0f} tokens/sec")

        return avg_loss, perplexity

    def train_step(self) -> dict[str, Any]:
        """Ultra-stable training step with adaptive error handling."""
        self.model.train()
        step_start_time = time.time()

        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for accum_step in range(self.args.gradient_accumulation_steps):
            try:
                batch_start_time = time.time()
                inputs, targets = self.train_loader.get_batch()
                batch_time = time.time() - batch_start_time

                forward_start_time = time.time()

                # Debug: Check device placement
                print(
                    f"🔍 Debug - inputs device: {inputs.device}, model device: {next(self.model.parameters()).device}"
                )

                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler is not None):
                    logits = self.model(inputs)
                    loss = cross_entropy(logits, targets) / self.args.gradient_accumulation_steps
                forward_time = time.time() - forward_start_time

                if not torch.isfinite(loss):
                    print(f"⚠️ Non-finite loss in accumulation step {accum_step}: {loss.item()}")
                    return {
                        "loss": float("nan"),
                        "lr": lr,
                        "grad_norm": float("nan"),
                        "step_time": time.time() - step_start_time,
                        "batch_time": batch_time,
                        "forward_time": forward_time,
                        "backward_time": 0.0,
                        "optimizer_time": 0.0,
                    }

                total_loss += loss.item()

                backward_start_time = time.time()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                backward_time = time.time() - backward_start_time

            except Exception as e:
                print(f"⚠️ Error in accumulation step {accum_step}: {e}")
                return {
                    "loss": float("nan"),
                    "lr": lr,
                    "grad_norm": float("nan"),
                    "step_time": time.time() - step_start_time,
                    "batch_time": 0.0,
                    "forward_time": 0.0,
                    "backward_time": 0.0,
                    "optimizer_time": 0.0,
                }

        optimizer_start_time = time.time()
        try:
            # Use adaptive gradient clipping if available
            if self.adaptive_clipper is not None:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                grad_norm, was_clipped = safe_gradient_step(
                    self.optimizer,
                    self.adaptive_clipper,
                    self.model.parameters(),
                    fallback_clip=self.args.gradient_clipping,
                )

                if self.scaler is not None:
                    self.scaler.update()
            else:
                # Fallback to traditional gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    from cs336_basics.training.gradient_clipping import gradient_clipping

                    grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    from cs336_basics.training.gradient_clipping import gradient_clipping

                    grad_norm = gradient_clipping(self.model.parameters(), self.args.gradient_clipping)
                    self.optimizer.step()

        except Exception as e:
            print(f"⚠️ Error in optimizer step: {e}")
            return {
                "loss": float("nan"),
                "lr": lr,
                "grad_norm": float("nan"),
                "step_time": time.time() - step_start_time,
                "batch_time": batch_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "optimizer_time": 0.0,
            }

        optimizer_time = time.time() - optimizer_start_time
        step_time = time.time() - step_start_time

        return {
            "loss": total_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time": step_time,
            "batch_time": batch_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
        }

    def train(self):
        """Ultra-stable training loop with comprehensive monitoring."""
        MAX_TRAINING_TIME = 1.5 * 3600

        print(f"\n🛡️ Starting ULTRA-STABLE training for {self.args.steps} steps")
        print(f"⏰ TIME LIMIT: 1.5 hours ({MAX_TRAINING_TIME / 3600:.1f}h)")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"⚡ Device: {self.device}")
        print(f"🎯 Target: Beat validation loss 3.0781")

        self.experiment_logger.add_note("Ultra-stable training started with all 2025 safety features")
        self.training_integrator.start_epoch(0)

        val_loss, val_perplexity = self.evaluate()
        print(f"📊 Initial validation - loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")

        self.training_integrator.log_validation_step(
            step=0,
            val_loss=val_loss,
            perplexity=val_perplexity,
            wallclock_time=0.0,
        )

        pbar = tqdm(range(self.args.steps), desc="🛡️ Ultra-Stable Training")
        training_start_time = time.time()
        best_val_loss = float("inf")
        consecutive_stable_steps = 0

        for step in pbar:
            self.step = step

            elapsed_time = time.time() - training_start_time
            if elapsed_time >= MAX_TRAINING_TIME:
                print(f"\n⏰ TIME LIMIT REACHED: {elapsed_time / 3600:.2f}h / {MAX_TRAINING_TIME / 3600:.1f}h")
                print("🛑 Stopping training to stay within 1.5-hour budget")
                self.experiment_logger.add_note(f"Training stopped due to time limit: {elapsed_time / 3600:.2f}h")
                break

            metrics = self.train_step()

            # Advanced stability monitoring
            if self.stability_monitor:
                should_continue, stability_stats, recovery_state = self.stability_monitor.update(
                    metrics["loss"], metrics["grad_norm"], step
                )
                if not should_continue:
                    print("🛑 Training stopped due to stability issues")
                    self.experiment_logger.add_note("Training stopped due to stability issues")
                    break

                if stability_stats.get("nan_count", 0) == 0:
                    consecutive_stable_steps += 1
                else:
                    consecutive_stable_steps = 0

            elapsed_time = time.time() - training_start_time
            effective_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
            tokens_processed = (step + 1) * effective_batch_size * self.args.context_length
            tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
            mfu = self.calculate_mfu(tokens_per_sec)

            time_remaining = max(0, MAX_TRAINING_TIME - elapsed_time)
            hours_remaining = time_remaining / 3600

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
                batch_time=metrics["batch_time"],
                forward_time=metrics["forward_time"],
                backward_time=metrics["backward_time"],
                optimizer_time=metrics["optimizer_time"],
                effective_batch_size=effective_batch_size,
                sequence_length=self.args.context_length,
            )

            progress_info = {
                "loss": f"{metrics['loss']:.4f}",
                "lr": f"{metrics['lr']:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "MFU": f"{mfu:.3f}",
                "best_val": f"{best_val_loss:.4f}",
                "time_left": f"{hours_remaining:.2f}h",
                "stable": f"{consecutive_stable_steps}",
            }

            # Add memory monitoring for CUDA
            if self.device.type == "cuda" and step % 50 == 0:
                current_memory = torch.cuda.memory_allocated() / 1e9
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                progress_info["mem_gb"] = f"{current_memory:.1f}/{peak_memory:.1f}"

            pbar.set_postfix(progress_info)

            if step > 0 and step % self.args.validation_step_interval == 0:
                val_loss, val_perplexity = self.evaluate()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.experiment_logger.add_note(f"🎯 NEW BEST validation loss: {val_loss:.4f}")

                print(
                    f"\n📊 Step {step}: train_loss={metrics['loss']:.4f}, "
                    f"val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}, "
                    f"MFU={mfu:.3f}, tok/s={tokens_per_sec:.0f}, time_left={hours_remaining:.2f}h"
                )

                self.training_integrator.log_validation_step(
                    step=step,
                    val_loss=val_loss,
                    perplexity=val_perplexity,
                    wallclock_time=elapsed_time / 3600,
                    mfu_at_validation=mfu,
                    tokens_per_sec_at_validation=tokens_per_sec,
                )

            if step > 0 and step % self.args.checkpoint_step_interval == 0:
                checkpoint_path = f"checkpoint_{self.experiment_timestamp}_step_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)
                self.experiment_logger.add_note(f"Checkpoint saved: {checkpoint_path}")
                print(f"💾 Saved checkpoint: {checkpoint_path}")

        final_elapsed_time = time.time() - training_start_time
        val_loss, val_perplexity = self.evaluate()
        final_tokens_per_sec = ((step + 1) * effective_batch_size * self.args.context_length) / final_elapsed_time
        final_mfu = self.calculate_mfu(final_tokens_per_sec)

        print(f"\n🏁 ULTRA-STABLE TRAINING FINAL RESULTS:")
        print(f"📊 Final validation loss: {val_loss:.4f}")
        print(f"🎯 Target was: 3.0781 (SUCCESS: {'YES' if val_loss < 3.0781 else 'NO'})")
        print(f"⚡ Training time: {final_elapsed_time / 3600:.2f}h / {MAX_TRAINING_TIME / 3600:.1f}h")
        print(f"🚀 Final MFU: {final_mfu:.3f}")
        print(f"⚡ Tokens/sec: {final_tokens_per_sec:.0f}")
        print(f"📈 Total steps completed: {step + 1}")
        print(f"🛡️ Stable steps: {consecutive_stable_steps}")

        final_checkpoint = f"final_checkpoint_{self.experiment_timestamp}_step_{step}.pt"
        save_checkpoint(self.model, self.optimizer, step, final_checkpoint)
        print(f"💾 Saved final checkpoint: {final_checkpoint}")

        success = val_loss < 3.0781
        self.experiment_logger.add_note(
            f"Ultra-stable training completed. "
            f"Final val_loss: {val_loss:.4f}, MFU: {final_mfu:.3f}, "
            f"Time: {final_elapsed_time / 3600:.2f}h, Success: {success}"
        )
        self.experiment_logger.mark_completed(success=success)


def load_config_from_json(config_path: str) -> TrainArgs:
    """Load configuration from JSON file with robust error handling."""
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

    unexpected_fields = provided_fields - expected_fields
    if unexpected_fields:
        print(f"Warning: Unexpected fields in config will be ignored: {unexpected_fields}")
        config_dict = {k: v for k, v in config_dict.items() if k in expected_fields}

    missing_fields = []
    for field in fields(TrainArgs):
        if field.name not in config_dict and field.default == field.default_factory:
            missing_fields.append(field.name)

    if missing_fields:
        raise ValueError(f"Required fields missing from config: {missing_fields}")

    try:
        return TrainArgs(**config_dict)
    except TypeError as e:
        raise TypeError(
            f"Error creating TrainArgs from config: {e}. "
            f"Expected fields: {sorted(expected_fields)}. "
            f"Provided fields: {sorted(provided_fields)}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ultra-Stable Transformer Training")

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--steps", type=int, help="Training steps")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")

    parser.add_argument(
        "--mode",
        type=str,
        default="stable",
        choices=["stable"],  # Only stable mode supported
        help="Training mode: only 'stable' supported for maximum safety",
    )

    parser.add_argument("--experiment_name", type=str, default="h100_ultra_stable_training", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1-stable", help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, default="", help="Wandb entity")
    parser.add_argument("--log_dir", type=str, default="experiments", help="Local logging directory")

    args = parser.parse_args()

    if args.config:
        config = load_config_from_json(args.config)
        if args.train_data:
            config.training_set = args.train_data
        if args.val_data:
            config.validation_set = args.val_data
        if args.steps:
            config.steps = args.steps
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.max_learning_rate = args.lr
        if args.experiment_name != "h100_ultra_stable_training":
            config.experiment_name = args.experiment_name
        if args.wandb:
            config.use_wandb = True
        if args.wandb_project != "cs336-assignment1-stable":
            config.wandb_project = args.wandb_project
        if args.wandb_entity:
            config.wandb_entity = args.wandb_entity
        if args.log_dir != "experiments":
            config.log_dir = args.log_dir
    else:
        config = TrainArgs(
            training_set=args.train_data or "data/encoded/owt_train_tokens.npy",
            validation_set=args.val_data or "data/encoded/owt_valid_tokens.npy",
            steps=args.steps or 5400,
            batch_size=args.batch_size or 48,
            max_learning_rate=args.lr or 0.0006,
            experiment_name=args.experiment_name,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            log_dir=args.log_dir,
        )

    print("🛡️ ULTRA-STABLE TRAINING INITIATED")
    print("🎯 Goal: Beat validation loss 3.0781 with ZERO NaN losses")
    print("⚡ Features: Adaptive clipping, Muon optimizer, stability monitoring")

    trainer = Trainer(config)
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
