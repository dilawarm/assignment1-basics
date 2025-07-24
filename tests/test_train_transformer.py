"""
Comprehensive tests for train_transformer.py script.

This module tests the complete training infrastructure including:
- TrainingConfig dataclass with validation
- DataLoader with memory mapping and optimization
- Trainer with H100 optimizations and experiment tracking
- Configuration management (load/save/creation)
- Integration testing with mocked components
- Performance and memory efficiency validation
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cs336_basics.scripts.train_transformer import (
    DataLoader,
    Trainer,
    TrainingConfig,
    create_optimized_configs,
    load_config,
    save_config,
)


class TestTrainingConfig:
    """Test the TrainingConfig dataclass with validation and optimization."""

    def test_config_creation_minimal(self) -> None:
        """Test TrainingConfig creation with minimal required parameters."""
        config = TrainingConfig(
            train_data_path="data/train.npy",
            vocab_size=1000,
            context_length=128,
        )

        assert config.train_data_path == "data/train.npy"
        assert config.val_data_path is None
        assert config.vocab_size == 1000
        assert config.context_length == 128
        assert config.d_model == 1024
        assert config.num_layers == 16
        assert config.activation == "swiglu"
        assert config.use_unet_architecture == True
        assert config.tie_embeddings == False
        assert config.optimizer == "muon_adamw"
        assert config.effective_batch_size == config.batch_size * config.gradient_accumulation_steps
        assert config.total_tokens == config.effective_batch_size * config.max_steps * config.context_length

    def test_config_creation_full(self) -> None:
        """Test TrainingConfig creation with all parameters specified."""
        config = TrainingConfig(
            train_data_path="data/train.npy",
            val_data_path="data/val.npy",
            vocab_size=32000,
            context_length=256,
            d_model=768,
            num_layers=6,
            num_heads=12,
            d_ff=2048,
            max_steps=10000,
            batch_size=32,
            learning_rate=1e-4,
            experiment_name="test_experiment",
        )

        assert config.train_data_path == "data/train.npy"
        assert config.val_data_path == "data/val.npy"
        assert config.vocab_size == 32000
        assert config.context_length == 256
        assert config.d_model == 768
        assert config.num_layers == 6
        assert config.num_heads == 12
        assert config.d_ff == 2048
        assert config.max_steps == 10000
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.experiment_name == "test_experiment"

    def test_config_validation_positive_values(self) -> None:
        """Test that config validation catches negative values."""
        with pytest.raises(AssertionError, match="vocab_size must be positive"):
            TrainingConfig(train_data_path="data/train.npy", vocab_size=0)

        with pytest.raises(AssertionError, match="context_length must be positive"):
            TrainingConfig(train_data_path="data/train.npy", context_length=0)

        with pytest.raises(AssertionError, match="d_model must be positive"):
            TrainingConfig(train_data_path="data/train.npy", d_model=0)

        with pytest.raises(AssertionError, match="max_steps must be positive"):
            TrainingConfig(train_data_path="data/train.npy", max_steps=0)

        with pytest.raises(AssertionError, match="batch_size must be positive"):
            TrainingConfig(train_data_path="data/train.npy", batch_size=0)

        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            TrainingConfig(train_data_path="data/train.npy", learning_rate=0)

    def test_config_validation_divisibility(self) -> None:
        """Test that d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError, match="d_model must be divisible by num_heads"):
            TrainingConfig(
                train_data_path="data/train.npy",
                d_model=511,
                num_heads=16,
            )

        config = TrainingConfig(
            train_data_path="data/train.npy",
            d_model=512,
            num_heads=16,
        )
        assert config.d_model == 512
        assert config.num_heads == 16

    def test_config_d_ff_tensor_core_optimization(self) -> None:
        """Test that d_ff is adjusted for optimal tensor core usage."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config = TrainingConfig(
                train_data_path="data/train.npy",
                d_ff=1345,
            )

            assert config.d_ff == 1408

            assert len(w) == 1
            assert "Adjusted d_ff" in str(w[0].message)

    def test_config_checkpoint_dir_creation(self) -> None:
        """Test that checkpoint directory is created during config initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "test_checkpoints")
            config = TrainingConfig(
                train_data_path="data/train.npy",
                checkpoint_dir=checkpoint_dir,
            )

            assert config.checkpoint_dir == checkpoint_dir
            assert os.path.exists(checkpoint_dir)

    def test_config_effective_calculations(self) -> None:
        """Test that effective batch size and total tokens are calculated correctly."""
        config = TrainingConfig(
            train_data_path="data/train.npy",
            batch_size=32,
            gradient_accumulation_steps=4,
            max_steps=1000,
            context_length=256,
        )

        expected_effective_batch_size = 32 * 4
        expected_total_tokens = expected_effective_batch_size * 1000 * 256

        assert config.effective_batch_size == expected_effective_batch_size
        assert config.total_tokens == expected_total_tokens


class TestDataLoader:
    """Test the optimized DataLoader class."""

    def create_test_data(self, temp_dir: str, size: int = 10000) -> str:
        """Create a test dataset file."""
        data_path = os.path.join(temp_dir, "test_data.npy")
        test_data = np.arange(size, dtype=np.uint16)
        memmap_data = np.memmap(data_path, dtype=np.uint16, mode="w+", shape=(size,))
        memmap_data[:] = test_data[:]
        del memmap_data
        return data_path

    def test_dataloader_creation(self) -> None:
        """Test DataLoader creation with valid data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_test_data(temp_dir)

            loader = DataLoader(
                data_path=data_path,
                batch_size=16,
                context_length=64,
                device="cpu",
            )

            assert loader.data_path == data_path
            assert loader.batch_size == 16
            assert loader.context_length == 64
            assert loader.device == "cpu"
            assert loader.data_size == 10000

    def test_dataloader_file_not_found(self) -> None:
        """Test DataLoader handles missing data files."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            DataLoader(
                data_path="nonexistent_file.npy",
                batch_size=16,
                context_length=64,
                device="cpu",
            )

    def test_dataloader_dataset_too_small(self) -> None:
        """Test DataLoader validates dataset size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_test_data(temp_dir, size=10)

            with pytest.raises(ValueError, match="Dataset too small"):
                DataLoader(
                    data_path=data_path,
                    batch_size=16,
                    context_length=15,
                    device="cpu",
                )

    def test_dataloader_get_batch_shapes(self) -> None:
        """Test that get_batch returns correct tensor shapes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_test_data(temp_dir)

            loader = DataLoader(
                data_path=data_path,
                batch_size=8,
                context_length=32,
                device="cpu",
            )

            inputs, targets = loader.get_batch()

            assert inputs.shape == (8, 32)
            assert targets.shape == (8, 32)
            assert inputs.device.type == "cpu"
            assert targets.device.type == "cpu"

    def test_dataloader_get_batch_values(self) -> None:
        """Test that get_batch returns valid sequences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_test_data(temp_dir)

            loader = DataLoader(
                data_path=data_path,
                batch_size=4,
                context_length=16,
                device="cpu",
            )

            inputs, targets = loader.get_batch()

            for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    assert targets[i, j] == inputs[i, j] + 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dataloader_cuda_device(self) -> None:
        """Test DataLoader with CUDA device."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_test_data(temp_dir)

            loader = DataLoader(
                data_path=data_path,
                batch_size=4,
                context_length=16,
                device="cuda",
            )

            inputs, targets = loader.get_batch()

            assert inputs.device.type == "cuda"
            assert targets.device.type == "cuda"


class TestTrainer:
    """Test the Trainer class with mocked dependencies."""

    def create_minimal_config(self, temp_dir: str) -> TrainingConfig:
        """Create minimal config for testing."""
        data_path = os.path.join(temp_dir, "test_data.npy")
        test_data = np.arange(10000, dtype=np.uint16)
        np.save(data_path, test_data)

        return TrainingConfig(
            train_data_path=data_path,
            vocab_size=1000,
            context_length=64,
            d_model=128,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_steps=10,
            batch_size=4,
            log_interval=5,
            eval_interval=10,
            save_interval=10,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            use_wandb=False,
            compile_model=False,
        )

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_initialization(self, mock_integrator, mock_logger) -> None:
        """Test Trainer initialization with mocked dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)

            trainer = Trainer(config)

            assert trainer.config == config
            assert trainer.step == 0
            assert trainer.device.type in ["cuda", "cpu"]
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.train_loader is not None

            mock_logger.assert_called_once()
            mock_integrator.assert_called_once()

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_trainer_device_setup_cuda_available(self, mock_props, mock_name, mock_integrator, mock_logger) -> None:
        """Test device setup when CUDA is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)
            config.device = "cpu"

            mock_name.return_value = "Mock GPU"
            mock_props.return_value = MagicMock(total_memory=8 * 1024**3)

            with patch("torch.cuda.is_available", return_value=True):
                trainer = Trainer(config)

                assert trainer.device.type == "cpu"

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_device_setup_cuda_unavailable(self, mock_integrator, mock_logger) -> None:
        """Test device setup when CUDA is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)
            config.device = "cuda"

            with patch("torch.cuda.is_available", return_value=False):
                trainer = Trainer(config)

                assert trainer.device.type == "cpu"
                assert not trainer.config.use_tf32
                assert not trainer.config.compile_model

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_get_lr_schedule(self, mock_integrator, mock_logger) -> None:
        """Test learning rate scheduling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)
            config.learning_rate = 1e-3
            config.min_learning_rate = 1e-4
            config.warmup_steps = 5
            config.max_steps = 20

            trainer = Trainer(config)

            lr_step_0 = trainer.get_lr(0)
            lr_step_2 = trainer.get_lr(2)
            lr_step_5 = trainer.get_lr(5)

            assert lr_step_0 > 0.0 and lr_step_0 < config.learning_rate * 0.1
            assert lr_step_2 > lr_step_0
            assert lr_step_5 == config.learning_rate

            lr_step_10 = trainer.get_lr(10)
            lr_step_20 = trainer.get_lr(20)

            assert lr_step_10 < lr_step_5
            assert lr_step_20 >= config.min_learning_rate

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_train_step(self, mock_integrator, mock_logger) -> None:
        """Test single training step execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)

            trainer = Trainer(config)

            def mock_get_batch():
                inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
                targets = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
                return inputs, targets

            trainer.train_loader.get_batch = mock_get_batch

            metrics = trainer.train_step()

            assert isinstance(metrics, dict)
            assert "loss" in metrics
            assert "lr" in metrics
            assert isinstance(metrics["loss"], float)
            assert metrics["loss"] >= 0.0

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_evaluate(self, mock_integrator, mock_logger) -> None:
        """Test evaluation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)

            val_data_path = os.path.join(temp_dir, "val_data.npy")
            val_data = np.arange(config.vocab_size, dtype=np.uint16)
            memmap_val_data = np.memmap(val_data_path, dtype=np.uint16, mode="w+", shape=(len(val_data),))
            memmap_val_data[:] = val_data[:]
            del memmap_val_data
            config.val_data_path = val_data_path
            config.eval_batches = 5

            trainer = Trainer(config)

            eval_metrics = trainer.evaluate()

            assert isinstance(eval_metrics, dict)
            assert "loss" in eval_metrics
            assert "perplexity" in eval_metrics
            assert isinstance(eval_metrics["loss"], float)
            assert isinstance(eval_metrics["perplexity"], float)
            assert eval_metrics["loss"] >= 0.0
            assert eval_metrics["perplexity"] >= 1.0

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_evaluate_no_validation_data(self, mock_integrator, mock_logger) -> None:
        """Test evaluation when no validation data is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)

            trainer = Trainer(config)

            eval_metrics = trainer.evaluate()

            assert eval_metrics == {}

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_trainer_save_checkpoint(self, mock_integrator, mock_logger) -> None:
        """Test checkpoint saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_minimal_config(temp_dir)

            trainer = Trainer(config)
            trainer.step = 100

            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            trainer.save_checkpoint(checkpoint_path)

            assert os.path.exists(checkpoint_path)

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "iteration" in checkpoint or "step" in checkpoint
            step_value = checkpoint.get("iteration", checkpoint.get("step"))
            assert step_value == 100


class TestConfigurationManagement:
    """Test configuration loading, saving, and creation utilities."""

    def test_save_and_load_config(self) -> None:
        """Test saving and loading configuration to/from JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                train_data_path="data/train.npy",
                val_data_path="data/val.npy",
                vocab_size=32000,
                context_length=256,
                experiment_name="test_config",
            )

            config_path = os.path.join(temp_dir, "test_config.json")
            save_config(config, config_path)

            assert os.path.exists(config_path)

            with open(config_path) as f:
                config_dict = json.load(f)

            assert config_dict["train_data_path"] == "data/train.npy"
            assert config_dict["val_data_path"] == "data/val.npy"
            assert config_dict["vocab_size"] == 32000
            assert config_dict["context_length"] == 256
            assert config_dict["experiment_name"] == "test_config"

            loaded_config = load_config(config_path)

            assert loaded_config.train_data_path == config.train_data_path
            assert loaded_config.val_data_path == config.val_data_path
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.context_length == config.context_length
            assert loaded_config.experiment_name == config.experiment_name

    def test_load_config_file_not_found(self) -> None:
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_load_config_invalid_json(self) -> None:
        """Test loading configuration from invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid_config.json")

            with open(config_path, "w") as f:
                f.write("invalid json content")

            with pytest.raises(json.JSONDecodeError):
                load_config(config_path)

    def test_create_optimized_configs(self) -> None:
        """Test creation of optimized configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                create_optimized_configs()

                tinystories_config_path = "cs336_basics/scripts/configs/tinystories_h100_v1.json"
                owt_config_path = "cs336_basics/scripts/configs/openwebtext_h100_v1.json"

                assert os.path.exists(tinystories_config_path)
                assert os.path.exists(owt_config_path)

                tinystories_config = load_config(tinystories_config_path)
                assert tinystories_config.vocab_size == 10000
                assert tinystories_config.experiment_name == "tinystories_h100_v1"
                assert tinystories_config.optimizer == "muon_adamw"
                assert tinystories_config.activation == "swiglu"
                assert tinystories_config.use_unet_architecture == True
                assert "tinystories" in tinystories_config.train_data_path

                owt_config = load_config(owt_config_path)
                assert owt_config.vocab_size == 32000
                assert owt_config.experiment_name == "openwebtext_h100_v1"
                assert owt_config.optimizer == "muon_adamw"
                assert owt_config.activation == "swiglu"
                assert owt_config.use_unet_architecture == True
                assert "owt" in owt_config.train_data_path

            finally:
                os.chdir(original_cwd)


class TestIntegration:
    """Integration tests for complete training workflow."""

    def create_integration_config(self, temp_dir: str) -> TrainingConfig:
        """Create configuration for integration testing."""
        vocab_size = 100
        context_length = 32

        train_data_path = os.path.join(temp_dir, "train_data.npy")
        train_data = np.arange(vocab_size, dtype=np.uint16)
        memmap_train_data = np.memmap(train_data_path, dtype=np.uint16, mode="w+", shape=(vocab_size,))
        memmap_train_data[:] = train_data[:]
        del memmap_train_data

        val_data_path = os.path.join(temp_dir, "val_data.npy")
        val_data = np.arange(vocab_size, dtype=np.uint16)
        memmap_val_data = np.memmap(val_data_path, dtype=np.uint16, mode="w+", shape=(vocab_size,))
        memmap_val_data[:] = val_data[:]
        del memmap_val_data

        return TrainingConfig(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=64,
            num_layers=1,
            num_heads=4,
            d_ff=128,
            max_steps=5,
            batch_size=2,
            log_interval=2,
            eval_interval=3,
            save_interval=5,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            use_wandb=False,
            compile_model=False,
            use_tf32=False,
        )

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_complete_training_workflow(self, mock_integrator, mock_logger) -> None:
        """Test complete training workflow from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_integration_config(temp_dir)

            trainer = Trainer(config)

            mock_integrator_instance = MagicMock()
            trainer.training_integrator = mock_integrator_instance

            trainer.train()

            assert trainer.step == config.max_steps

            checkpoint_dir = Path(config.checkpoint_dir)
            assert checkpoint_dir.exists()

            final_checkpoints = list(checkpoint_dir.glob("checkpoint_final_time_*h_step_*.pt"))
            assert len(final_checkpoints) > 0, f"No final checkpoint found in {checkpoint_dir}"

            mock_integrator_instance.start_epoch.assert_called_once()
            mock_integrator_instance.log_training_step.assert_called()

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_training_with_resume(self, mock_integrator, mock_logger) -> None:
        """Test training resume functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_integration_config(temp_dir)

            trainer1 = Trainer(config)
            trainer1.step = 3

            checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_step_3.pt")
            trainer1.save_checkpoint(checkpoint_path)

            config.resume_from = checkpoint_path
            trainer2 = Trainer(config)

            assert trainer2.step == 3

    def test_config_validation_in_workflow(self) -> None:
        """Test that configuration validation catches issues in realistic scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data_path = os.path.join(temp_dir, "train_data.npy")
            small_data = np.arange(10, dtype=np.uint16)
            memmap_small_data = np.memmap(train_data_path, dtype=np.uint16, mode="w+", shape=(10,))
            memmap_small_data[:] = small_data[:]
            del memmap_small_data

            config = TrainingConfig(
                train_data_path=train_data_path,
                context_length=15,
                batch_size=4,
            )

            with pytest.raises(ValueError, match="Dataset too small"):
                DataLoader(
                    data_path=config.train_data_path,
                    batch_size=config.batch_size,
                    context_length=config.context_length,
                    device="cpu",
                )


class TestPerformanceOptimizations:
    """Test H100-specific performance optimizations."""

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_tf32_optimization_setup(self, mock_props, mock_name, mock_integrator, mock_logger) -> None:
        """Test TF32 optimization setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data_path = os.path.join(temp_dir, "train_data.npy")
            train_data = np.arange(1000, dtype=np.uint16)
            memmap_data = np.memmap(train_data_path, dtype=np.uint16, mode="w+", shape=(1000,))
            memmap_data[:] = train_data[:]
            del memmap_data

            config = TrainingConfig(
                train_data_path=train_data_path,
                use_tf32=True,
                device="cpu",
            )

            mock_name.return_value = "Mock GPU"
            mock_props.return_value = MagicMock(total_memory=8 * 1024**3)

            with patch("torch.cuda.is_available", return_value=True):
                trainer = Trainer(config)

                assert trainer.config.use_tf32

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    def test_memory_optimization_settings(self, mock_integrator, mock_logger) -> None:
        """Test memory optimization settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data_path = os.path.join(temp_dir, "train_data.npy")
            train_data = np.arange(1000, dtype=np.uint16)
            memmap_data = np.memmap(train_data_path, dtype=np.uint16, mode="w+", shape=(1000,))
            memmap_data[:] = train_data[:]
            del memmap_data

            config = TrainingConfig(
                train_data_path=train_data_path,
                pin_memory=True,
                channels_last=True,
            )

            trainer = Trainer(config)

            assert trainer.train_loader.pin_memory

    def test_batch_size_scaling_calculations(self) -> None:
        """Test that batch size scaling calculations are correct for different scenarios."""
        config1 = TrainingConfig(
            train_data_path="dummy",
            batch_size=32,
            gradient_accumulation_steps=4,
            max_steps=1000,
            context_length=256,
        )

        assert config1.effective_batch_size == 128
        assert config1.total_tokens == 128 * 1000 * 256

        config2 = TrainingConfig(
            train_data_path="dummy",
            batch_size=64,
            gradient_accumulation_steps=2,
            max_steps=2000,
            context_length=512,
        )

        assert config2.effective_batch_size == 128
        assert config2.total_tokens == 128 * 2000 * 512


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_training_with_corrupted_data(self) -> None:
        """Test handling of corrupted data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "corrupted_data.npy")
            with open(data_path, "w") as f:
                f.write("corrupted content")

            with pytest.raises((ValueError, OSError)):
                DataLoader(
                    data_path=data_path,
                    batch_size=16,
                    context_length=32,
                    device="cpu",
                )

    @patch("cs336_basics.scripts.train_transformer.ExperimentLogger")
    @patch("cs336_basics.scripts.train_transformer.TrainingIntegrator")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_training_with_invalid_device(self, mock_props, mock_name, mock_integrator, mock_logger) -> None:
        """Test handling of invalid device specifications."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data_path = os.path.join(temp_dir, "train_data.npy")
            train_data = np.arange(1000, dtype=np.uint16)
            memmap_data = np.memmap(train_data_path, dtype=np.uint16, mode="w+", shape=(1000,))
            memmap_data[:] = train_data[:]
            del memmap_data

            config = TrainingConfig(
                train_data_path=train_data_path,
                device="cpu",
            )

            mock_name.return_value = "Mock GPU"
            mock_props.return_value = MagicMock(total_memory=8 * 1024**3)

            with patch("torch.cuda.is_available", return_value=True):
                trainer = Trainer(config)
                assert trainer.device.type == "cpu"
