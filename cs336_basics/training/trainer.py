"""H100-optimized trainer with TorchAO FP8 support."""

import math
import os
import time
from dataclasses import dataclass

import pynvml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

pynvml.nvmlInit()


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    model_name: str = "gpt-350m-h100"

    # Optimization
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_steps: int | None = None
    warmup_steps: int = 2000

    # Evaluation
    eval_interval: int = 1000
    log_interval: int = 100
    save_interval: int = 5000

    # Data
    max_length: int = 1024
    num_workers: int = 4

    # Precision
    use_fp8: bool = True
    use_amp: bool = True

    # Hardware
    compile_model: bool = True
    compile_mode: str = "max-autotune"  # Options: "default", "reduce-overhead", "max-autotune"
    use_flash_attn: bool = True
    gradient_checkpointing: bool = True

    # Logging
    use_wandb: bool = True
    project_name: str = "cs336-assignment1"

    # Paths
    output_dir: str = "./checkpoints"
    resume_from: str | None = None


class Trainer:
    """H100-optimized trainer with TorchAO FP8 and Flash Attention support."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            print("Warning: CUDA not available, training will be slow!")

        self.model = self.model.to(self.device)

        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                self.model = self.model.to(torch.bfloat16)
                self.dtype = torch.bfloat16
                print(f"Using bfloat16 precision on {torch.cuda.get_device_name()}")
            else:
                self.model = self.model.to(torch.float16)
                self.dtype = torch.float16
                print(f"Using float16 precision on {torch.cuda.get_device_name()}")
        else:
            self.dtype = torch.float32

        self.fp8_enabled = False
        if config.use_fp8:
            has_float8 = any("Float8" in module.__class__.__name__ for _, module in self.model.named_modules())
            if has_float8:
                self.fp8_enabled = True
                print("TorchAO FP8 mode enabled - model contains Float8 modules")
            else:
                print("FP8 not enabled - install TorchAO and convert model")

        if config.compile_model:
            print(f"Compiling model with torch.compile(mode='{config.compile_mode}')...")
            self.model = torch.compile(self.model, mode=config.compile_mode)
            self.use_cuda_graphs = config.compile_mode == "reduce-overhead"
            if self.use_cuda_graphs:
                print("  Note: Using CUDA graphs with reduce-overhead mode")
        else:
            self.use_cuda_graphs = False

        self.optimizer = self._create_optimizer()

        self.scaler = GradScaler() if config.use_amp and not self.fp8_enabled else None

        self.scheduler = self._create_scheduler()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        if config.use_wandb:
            self._setup_wandb()
        elif config.use_wandb:
            print("Warning: wandb requested but not available. Install with: pip install wandb")

        os.makedirs(config.output_dir, exist_ok=True)

    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "norm", "embedding"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            fused=True,
        )

        return optimizer

    def _create_scheduler(self):
        """Create cosine learning rate scheduler with warmup."""

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps

            progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
            return self.config.min_learning_rate / self.config.learning_rate + (
                1 - self.config.min_learning_rate / self.config.learning_rate
            ) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_wandb(self):
        wandb.init(
            project=self.config.project_name,
            name=self.config.model_name,
            config=self.config.__dict__,
        )
        wandb.watch(self.model, log="all", log_freq=1000)

    def _get_gpu_stats(self) -> dict[str, float]:
        """Get GPU memory and utilization stats."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

            return {
                "gpu_memory_used_gb": mem_info.used / 1e9,
                "gpu_memory_total_gb": mem_info.total / 1e9,
                "gpu_utilization": util_info.gpu,
            }
        except:
            return {}

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        if self.use_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.scaler is not None and not self.fp8_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                outputs = self.model(input_ids=input_ids, labels=labels)
        else:
            outputs = self.model(input_ids=input_ids, labels=labels)

        loss = outputs["loss"]

        loss = loss / self.config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            if self.use_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs["loss"]

            batch_size, seq_len = input_ids.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def save_checkpoint(self, path: str | None = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")

    def train(self, max_steps: int | None = None):
        """Main training loop."""
        if max_steps is not None:
            self.config.max_steps = max_steps

        if self.config.max_steps is None:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            estimated_steps = int(
                5e9 / (self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length)
            )
            self.config.max_steps = estimated_steps

        print(f"Starting training for {self.config.max_steps} steps...")
        print(
            f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length} tokens"
        )

        self.model.train()
        start_time = time.time()
        tokens_processed = 0

        while self.global_step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                metrics = self.train_step(batch)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    tokens_processed += (
                        self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_length
                    )

                    if self.global_step % self.config.log_interval == 0:
                        elapsed_time = time.time() - start_time
                        tokens_per_sec = tokens_processed / elapsed_time

                        log_metrics = {
                            "train_loss": metrics["loss"],
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "tokens_per_second": tokens_per_sec,
                            "global_step": self.global_step,
                        }

                        log_metrics.update(self._get_gpu_stats())

                        if self.config.use_wandb:
                            wandb.log(log_metrics)

                        print(
                            f"Step {self.global_step}: loss={metrics['loss']:.4f}, "
                            f"lr={log_metrics['learning_rate']:.2e}, "
                            f"tokens/sec={tokens_per_sec:.0f}"
                        )

                    if self.global_step % self.config.eval_interval == 0:
                        print("\nRunning evaluation...")
                        eval_metrics = self.evaluate()

                        print(f"Validation loss: {eval_metrics['val_loss']:.4f}")
                        print(f"Validation perplexity: {eval_metrics['val_perplexity']:.2f}")

                        if self.config.use_wandb:
                            wandb.log(eval_metrics)

                        if eval_metrics["val_loss"] < self.best_val_loss:
                            self.best_val_loss = eval_metrics["val_loss"]
                            self.save_checkpoint(os.path.join(self.config.output_dir, "best_model.pt"))

                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint()

                    if self.global_step >= self.config.max_steps:
                        break

            self.epoch += 1

        print("\nTraining complete! Running final evaluation...")
        final_metrics = self.evaluate()
        print(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"Final validation perplexity: {final_metrics['val_perplexity']:.2f}")

        self.save_checkpoint(os.path.join(self.config.output_dir, "final_model.pt"))

        if self.config.use_wandb:
            wandb.finish()

        return final_metrics
