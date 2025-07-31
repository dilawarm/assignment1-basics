"""
Fast H100-optimized transformer training script.
Focuses on achieving val_loss < 3.0781 in 1.5 hours.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.enhanced_models import UNetTransformerLM
from cs336_basics.training.enhanced_optimizers import MuonOptimizer
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.optimizers import Adam


class H100FastTrainer:
    """Optimized trainer for H100 targeting 1.5 hour training."""

    def __init__(self):
        # Fixed optimal configuration based on winner's approach
        self.device = torch.device("cuda")

        # Model config - matching winner
        self.vocab_size = 32000
        self.context_length = 256  # Critical: 256, not 512!
        self.num_layers = 16
        self.d_model = 1024
        self.num_heads = 8
        self.d_ff = 2816

        # Training config
        self.batch_size = 256  # Larger batch for better GPU utilization
        self.steps = 20000
        self.base_lr = 0.001  # Higher base LR
        self.muon_lr = 0.02
        self.warmup_iters = 500  # Faster warmup
        self.val_interval = 250

        # Setup CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Enable bfloat16
        self.dtype = torch.bfloat16

        print(f"H100 Fast Trainer initialized")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Context length: {self.context_length} (2x faster than 512)")
        print(f"Batch size: {self.batch_size}")

    def create_model(self):
        """Create optimized U-Net transformer model."""
        model = UNetTransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=10000,
            use_squared_relu=True,
            tie_embeddings=False,
            device=self.device,
            dtype=self.dtype,
        )

        # Compile with reduce-overhead for best performance
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}, continuing without compile")

        return model

    def create_optimizers(self, model):
        """Create Muon + Adam optimizers following winner's approach."""
        # Get parameter groups
        param_groups = model.get_parameter_groups(base_lr=self.base_lr)

        # Separate parameters for Muon (2D non-embedding) and Adam (rest)
        muon_params = []
        adam_groups = []

        for group in param_groups:
            group_name = group.get("name", "")
            group_params = list(group["params"])

            if group_name == "regular":
                # 2D params go to Muon
                for p in group_params:
                    if p.ndim == 2:
                        muon_params.append(p)
                    else:
                        # Keep in Adam
                        adam_groups.append({"params": [p], "lr": group["lr"], "name": f"adam_{group_name}"})
            else:
                # Embeddings, lm_head, 1d_params stay with Adam
                adam_groups.append(group)

        print(f"Muon params: {sum(p.numel() for p in muon_params) / 1e6:.1f}M")

        # Create optimizers
        muon_opt = MuonOptimizer(
            muon_params,
            lr=self.muon_lr,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.01,
        )

        adam_opt = Adam(
            adam_groups,
            lr=self.base_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        return muon_opt, adam_opt

    def train(self):
        """Main training loop."""
        # Load data
        print("\nLoading data...")
        train_data = np.load("data/encoded/owt_train_tokens.npy", mmap_mode="r")
        val_data = np.load("data/encoded/owt_valid_tokens.npy", mmap_mode="r")
        print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

        # Create model and optimizers
        print("\nCreating model...")
        model = self.create_model()
        muon_opt, adam_opt = self.create_optimizers(model)

        # Training metrics
        best_val_loss = float("inf")
        tokens_processed = 0

        print(f"\n🚀 Starting training for {self.steps} steps")
        print(f"Target: val_loss < 3.0781 in 1.5 hours")

        start_time = time.time()
        pbar = tqdm(range(self.steps), desc="Training")

        for step in pbar:
            model.train()

            # Get learning rate
            lr = cosine_learning_rate_schedule(
                step,
                max_learning_rate=self.base_lr,
                min_learning_rate=self.base_lr * 0.1,
                warmup_iters=self.warmup_iters,
                cosine_cycle_iters=self.steps,
            )

            # Update learning rates
            muon_lr = self.muon_lr * (lr / self.base_lr)
            for pg in muon_opt.param_groups:
                pg["lr"] = muon_lr

            for pg in adam_opt.param_groups:
                base_scale = pg["lr"] / self.base_lr
                pg["lr"] = lr * base_scale

            # Training step
            muon_opt.zero_grad()
            adam_opt.zero_grad()

            # Get batch
            inputs, targets = get_batch(train_data, self.batch_size, self.context_length, str(self.device))

            # Forward pass
            logits = model(inputs)
            loss = cross_entropy(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            all_params = list(muon_opt.param_groups[0]["params"]) + [
                p for g in adam_opt.param_groups for p in g["params"]
            ]
            grad_norm = gradient_clipping(all_params, max_norm=0.5)

            # Optimizer steps
            muon_opt.step()
            adam_opt.step()

            # Update metrics
            tokens_processed += self.batch_size * self.context_length
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tokens_per_sec / 1000:.0f}k",
                    "grad": f"{grad_norm:.3f}",
                }
            )

            # Validation
            if step > 0 and step % self.val_interval == 0:
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for _ in range(50):
                        val_inputs, val_targets = get_batch(
                            val_data, self.batch_size, self.context_length, str(self.device)
                        )
                        val_logits = model(val_inputs)
                        val_loss = cross_entropy(val_logits, val_targets)
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                perplexity = np.exp(min(avg_val_loss, 10))

                hours_elapsed = elapsed / 3600
                print(
                    f"\nStep {step}: train_loss={loss.item():.4f}, val_loss={avg_val_loss:.4f}, "
                    f"ppl={perplexity:.2f}, time={hours_elapsed:.2f}h, tok/s={tokens_per_sec / 1000:.0f}k"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"✓ New best validation loss: {best_val_loss:.4f}")

                if avg_val_loss < 3.0781:
                    print(f"\n🎉 TARGET ACHIEVED! val_loss={avg_val_loss:.4f} < 3.0781")
                    print(f"Time: {hours_elapsed:.2f} hours")
                    break

        # Final stats
        total_hours = (time.time() - start_time) / 3600
        final_tokens_per_sec = tokens_processed / (time.time() - start_time)

        print(f"\nTraining completed in {total_hours:.2f} hours")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Average throughput: {final_tokens_per_sec / 1000:.0f}k tokens/sec")

        # Save model if successful
        if best_val_loss < 3.0781:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "muon_optimizer_state_dict": muon_opt.state_dict(),
                    "adam_optimizer_state_dict": adam_opt.state_dict(),
                    "best_val_loss": best_val_loss,
                    "step": step,
                },
                "h100_fast_model_success.pt",
            )
            print("✓ Model saved to h100_fast_model_success.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast H100 transformer training")
    parser.add_argument("--check", action="store_true", help="Run diagnostics only")
    args = parser.parse_args()

    if args.check:
        from cs336_basics.scripts.diagnose_training import diagnose_model

        diagnose_model()
    else:
        trainer = H100FastTrainer()
        trainer.train()
