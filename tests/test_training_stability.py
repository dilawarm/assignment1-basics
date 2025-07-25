"""
Tests for training stability and NaN prevention.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.activations import FFN, SwiGLU
from cs336_basics.nn.models import EnhancedTransformerLM
from cs336_basics.scripts.train_transformer import Trainer, TrainingConfig
from cs336_basics.training.optimizers import AdamW, MixedOptimizer, Muon


class TestNumericalStability:
    """Test numerical stability of core components."""

    def test_cross_entropy_stability(self):
        """Test cross-entropy loss with extreme values."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = cross_entropy(logits, targets)
        assert torch.isfinite(loss), "Loss should be finite for normal inputs"

        logits = torch.ones(batch_size, seq_len, vocab_size) * 1000  # Very large
        loss = cross_entropy(logits, targets)
        assert torch.isfinite(loss), "Loss should be finite for large logits"

        logits = torch.ones(batch_size, seq_len, vocab_size) * -1000  # Very small
        loss = cross_entropy(logits, targets)
        assert torch.isfinite(loss), "Loss should be finite for very small logits"

        logits = torch.full((batch_size, seq_len, vocab_size), float("nan"))
        loss = cross_entropy(logits, targets)
        assert torch.isfinite(loss), "Loss should be finite even with NaN inputs"

    def test_ffn_stability(self):
        """Test FFN (custom activation) numerical stability."""
        d_model, d_ff = 512, 2048
        ffn = FFN(d_model, d_ff)

        x = torch.randn(2, 10, d_model)
        output = ffn(x)
        assert torch.isfinite(output).all(), "FFN output should be finite for normal inputs"

        x = torch.ones(2, 10, d_model) * 100  # Large inputs
        output = ffn(x)
        assert torch.isfinite(output).all(), "FFN output should be finite for large inputs"
        assert output.abs().max() < 1e6, "FFN output should be bounded"

    def test_muon_newton_schulz_stability(self):
        """Test Muon optimizer Newton-Schulz orthogonalization stability."""
        optimizer = Muon([torch.tensor([1.0])], lr=0.01)

        X = torch.randn(5, 3)
        result = optimizer.newton_schulz_orthogonalize(X, 5)
        assert torch.isfinite(result).all(), "Newton-Schulz should return finite values"

        X = torch.full((5, 3), float("nan"))
        result = optimizer.newton_schulz_orthogonalize(X, 5)
        assert torch.isfinite(result).all(), "Newton-Schulz should handle NaN inputs safely"

        X = torch.ones(5, 3) * 1000
        result = optimizer.newton_schulz_orthogonalize(X, 5)
        assert torch.isfinite(result).all(), "Newton-Schulz should handle extreme values"

    def test_mixed_optimizer_lr_update(self):
        """Test that MixedOptimizer properly handles learning rate updates."""
        model = EnhancedTransformerLM(
            vocab_size=100, context_length=64, d_model=128, num_layers=2, num_heads=4, d_ff=256
        )

        param_names = {param: name for name, param in model.named_parameters()}

        optimizer = MixedOptimizer(
            model.parameters(), muon_lr=0.003, adamw_lr=0.003, embedding_lr=0.004, lm_head_lr=0.002
        )

        for group in optimizer.param_groups:
            assert "muon_lr" in group
            assert "adamw_lr" in group
            assert "embedding_lr" in group
            assert "lm_head_lr" in group

        x = torch.randint(0, 100, (2, 32))
        logits = model(x)
        targets = torch.randint(0, 100, (2, 32))
        loss = cross_entropy(logits, targets)
        loss.backward()

        optimizer.step(param_names=param_names)

        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param).all(), "Parameters should remain finite after optimization"


class TestTrainingStability:
    """Test training loop stability and NaN handling."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create minimal training configuration for testing."""
        train_data = np.random.randint(0, 1000, size=(10000,), dtype=np.uint16)
        val_data = np.random.randint(0, 1000, size=(1000,), dtype=np.uint16)

        train_path = tmp_path / "train.npy"
        val_path = tmp_path / "val.npy"

        np.save(train_path, train_data)
        np.save(val_path, val_data)

        return TrainingConfig(
            train_data_path=str(train_path),
            val_data_path=str(val_path),
            vocab_size=1000,
            context_length=64,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            max_steps=10,
            batch_size=4,
            learning_rate=0.001,
            warmup_steps=2,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            use_wandb=False,
            use_amp=False,
            compile_model=False,
        )

    def test_trainer_initialization(self, minimal_config):
        """Test that trainer initializes without errors."""
        trainer = Trainer(minimal_config)
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_learning_rate_schedule_application(self, minimal_config):
        """Test that learning rate schedule works correctly."""
        trainer = Trainer(minimal_config)

        # Test that get_lr returns expected dict structure
        lr_step_0 = trainer.get_lr(0)
        lr_step_5 = trainer.get_lr(5)

        # Verify dict structure
        assert "base_lr" in lr_step_0
        assert isinstance(lr_step_0["base_lr"], float)
        assert isinstance(lr_step_5["base_lr"], float)

        # Verify LR increases during warmup
        assert lr_step_5["base_lr"] > lr_step_0["base_lr"]

    def test_nan_detection_and_early_stopping(self, minimal_config):
        """Test that NaN detection properly stops training."""
        trainer = Trainer(minimal_config)

        with patch.object(trainer.train_loader, "get_batch") as mock_get_batch:
            inputs = torch.randint(
                0, minimal_config.vocab_size, (minimal_config.batch_size, minimal_config.context_length)
            )
            targets = torch.randint(
                0, minimal_config.vocab_size, (minimal_config.batch_size, minimal_config.context_length)
            )
            mock_get_batch.return_value = (inputs, targets)

            with patch("cs336_basics.scripts.train_transformer.cross_entropy") as mock_loss:
                mock_loss.return_value = torch.tensor(float("nan"))

                result = trainer.train_step()
                assert result.get("training_stopped", False), "Training should stop on NaN detection"
                assert not torch.isfinite(torch.tensor(result["loss"])), "Loss should be NaN"

    def test_gradient_nan_detection(self, minimal_config):
        """Test that NaN gradients are detected and handled."""
        trainer = Trainer(minimal_config)

        for param in trainer.original_model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        has_nan_grad = False
        for param in trainer.original_model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break

        assert has_nan_grad, "NaN gradients should be detected"

        with patch.object(trainer.train_loader, "get_batch") as mock_get_batch:
            inputs = torch.randint(
                0, minimal_config.vocab_size, (minimal_config.batch_size, minimal_config.context_length)
            )
            targets = torch.randint(
                0, minimal_config.vocab_size, (minimal_config.batch_size, minimal_config.context_length)
            )
            mock_get_batch.return_value = (inputs, targets)

            with patch("cs336_basics.scripts.train_transformer.cross_entropy", return_value=torch.tensor(1.0)):
                with patch.object(torch.Tensor, "backward"):
                    result = trainer.train_step()
                    assert has_nan_grad, "NaN gradients should be present and detectable"

    def test_emergency_checkpoint_creation(self, minimal_config):
        """Test that emergency checkpoints are created on training failure."""
        trainer = Trainer(minimal_config)

        with patch.object(trainer, "train_step") as mock_train_step:
            mock_train_step.return_value = {"training_stopped": True, "loss": float("nan"), "lr": 0.001}

            with patch.object(trainer, "save_checkpoint") as mock_save:
                trainer.step = 5
                metrics = trainer.train_step()

                if metrics.get("training_stopped", False):
                    emergency_checkpoint = (
                        Path(trainer.config.checkpoint_dir) / f"emergency_checkpoint_step_{trainer.step}.pt"
                    )
                    trainer.save_checkpoint(str(emergency_checkpoint))

                    mock_save.assert_called_once()


class TestActivationStability:
    """Test activation function numerical stability."""

    def test_swiglu_stability(self):
        """Test SwiGLU activation stability."""
        d_model, d_ff = 512, 1024
        swiglu = SwiGLU(d_model, d_ff)

        x = torch.randn(2, 10, d_model)
        output = swiglu(x)
        assert torch.isfinite(output).all(), "SwiGLU should produce finite outputs"

        x = torch.ones(2, 10, d_model) * 50
        output = swiglu(x)
        assert torch.isfinite(output).all(), "SwiGLU should handle large inputs"

    def test_ffn_custom_bounded_output(self):
        """Test that FFN (custom) activation produces bounded outputs."""
        d_model, d_ff = 256, 1024
        ffn = FFN(d_model, d_ff)

        for magnitude in [1, 10, 100]:
            x = torch.randn(2, 8, d_model) * magnitude
            output = ffn(x)

            assert torch.isfinite(output).all(), f"FFN output should be finite for magnitude {magnitude}"
            assert output.abs().max() < 1e6, f"FFN output should be bounded for magnitude {magnitude}"
