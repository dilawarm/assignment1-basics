"""
Diagnostic script to identify training issues with the H100 optimized transformer.
"""

from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.enhanced_models import UNetTransformerLM
from cs336_basics.training.enhanced_optimizers import MuonOptimizer
from cs336_basics.training.optimizers import Adam


def diagnose_model():
    """Run diagnostics on the model and training setup."""
    print("=== H100 Training Diagnostics ===\n")

    # Check CUDA availability and H100 features
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_capability()}")

        # Check for H100 specific features
        if torch.cuda.get_device_capability()[0] >= 9:
            print("✓ H100 GPU detected - FP8 and Transformer Engine available")

        # Check TF32 settings
        print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  Flash attention available: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")
    else:
        print("✗ CUDA not available!")
        return

    print("\n=== Model Configuration ===")

    # Test with both configurations
    configs = [
        {
            "name": "Original (512 context)",
            "context_length": 512,
            "batch_size": 128,
        },
        {
            "name": "Optimized (256 context)",
            "context_length": 256,
            "batch_size": 256,
        },
    ]

    for config in configs:
        print(f"\n{config['name']}:")

        # Create model
        model = UNetTransformerLM(
            vocab_size=32000,
            context_length=config["context_length"],
            d_model=1024,
            num_layers=16,
            num_heads=8,
            d_ff=2816,
            rope_theta=10000,
            use_squared_relu=True,
            tie_embeddings=False,
            device=device,
            dtype=torch.bfloat16,
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params / 1e6:.1f}M")

        # Memory usage
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"  Model memory: {model_memory:.2f} GB")

        # Test forward pass
        try:
            dummy_input = torch.randint(0, 32000, (config["batch_size"], config["context_length"]), device=device)

            # Test without gradients first
            with torch.no_grad():
                output = model(dummy_input)
                print(f"  ✓ Forward pass successful: output shape {output.shape}")

                # Check for NaN
                if torch.isnan(output).any():
                    print("  ✗ WARNING: NaN values in output!")
                else:
                    print("  ✓ No NaN values in output")

                # Check output statistics
                print(f"  Output stats - mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")

        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")

        # Test loss computation
        try:
            targets = torch.randint(0, 32000, (config["batch_size"], config["context_length"]), device=device)

            # Compute loss with the no-grad output for reporting
            with torch.no_grad():
                loss_value = cross_entropy(output, targets).item()
            print(f"  ✓ Loss computation successful: {loss_value:.4f}")

            # Expected initial loss
            expected_loss = np.log(32000)  # Random prediction
            print(f"  Expected random loss: {expected_loss:.4f}")

        except Exception as e:
            print(f"  ✗ Loss computation failed: {e}")

        # Test backward pass
        try:
            # Check if parameters require gradients
            params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
            total_params = sum(1 for p in model.parameters())
            print(f"  Parameters requiring grad: {params_with_grad}/{total_params}")

            # Create loss that requires grad
            if output.requires_grad:
                loss.backward()
            else:
                # Recompute with gradients enabled
                model.train()
                output = model(dummy_input)
                loss = cross_entropy(output, targets)
                loss.backward()

            # Check gradients
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if grad_norm == 0:
                        print(f"  ✗ WARNING: Zero gradient in {name}")
                    elif torch.isnan(param.grad).any():
                        print(f"  ✗ WARNING: NaN gradient in {name}")

            if grad_norms:
                print(f"  ✓ Gradients computed - mean norm: {np.mean(grad_norms):.4f}")
            else:
                print(f"  ✗ No gradients computed!")

        except Exception as e:
            print(f"  ✗ Backward pass failed: {e}")

        # Estimate throughput
        try:
            import time

            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(10):
                with torch.no_grad():
                    output = model(dummy_input)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            tokens_per_batch = config["batch_size"] * config["context_length"]
            tokens_per_sec = (tokens_per_batch * 10) / elapsed

            print(f"  Throughput estimate: {tokens_per_sec / 1000:.0f}k tokens/sec")

            # Expected H100 performance
            expected_tps = 600000 if config["context_length"] == 256 else 300000
            efficiency = (tokens_per_sec / expected_tps) * 100
            print(f"  Efficiency vs expected: {efficiency:.1f}%")

        except Exception as e:
            print(f"  ✗ Throughput test failed: {e}")

        del model
        torch.cuda.empty_cache()

    print("\n=== Optimization Recommendations ===")
    print("1. Use context_length=256 for 2x faster iteration")
    print("2. Increase batch_size to 256 or higher for better GPU utilization")
    print("3. Use base_lr=0.001 or higher (winner likely used ~0.001-0.003)")
    print("4. Ensure torch.compile with mode='reduce-overhead' is working")
    print("5. Check that mixed precision (bfloat16) is properly enabled")
    print("6. Consider using NVIDIA's Transformer Engine for H100")

    # Test data loading
    print("\n=== Data Loading Test ===")
    data_path = Path("data/encoded/owt_train_tokens.npy")
    if data_path.exists():
        data = np.load(data_path, mmap_mode="r")
        print(f"✓ Training data found: {len(data):,} tokens")

        # Test batch loading speed
        import time

        start = time.time()
        for _ in range(100):
            inputs, targets = get_batch(data, batch_size=256, context_length=256, device=str(device))
        elapsed = time.time() - start
        print(f"  Batch loading: {100 / elapsed:.0f} batches/sec")
    else:
        print("✗ Training data not found!")


if __name__ == "__main__":
    diagnose_model()
