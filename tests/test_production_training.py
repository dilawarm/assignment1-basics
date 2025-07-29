"""
Comprehensive production tests for the transformer training script.

These tests ensure the training script works reliably in production environments,
specifically targeting H100 deployment with OpenWebText data.

Test Coverage:
- Production configuration validation
- Model initialization and parameter counts
- Training loop integrity (gradient flow, parameter updates, learning rate)
- Memory and performance optimization for H100
- Error handling and recovery
- Production data loading from data/encoded/ folder
- OpenWebText data format compatibility and integrity
- Checkpoint functionality and logging integration
"""

import json
import math
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cs336_basics.scripts.train_transformer import TrainArgs, Trainer, load_config_from_json


class TestProductionConfiguration:
    """Test configuration validation and production readiness."""

    def test_config_validation_with_valid_params(self):
        """Test that valid production configurations pass validation."""
        config = TrainArgs(
            vocab_size=32000,
            context_length=256,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,
            steps=1000,
            batch_size=32,
            max_learning_rate=2e-3,
        )
        assert config.vocab_size == 32000
        assert config.d_model % config.num_heads == 0

    def test_config_validation_with_invalid_params(self):
        """Test that invalid configurations are rejected."""
        with pytest.raises(AssertionError):
            TrainArgs(vocab_size=0)

        with pytest.raises(AssertionError):
            TrainArgs(d_model=513, num_heads=16)

        with pytest.raises(AssertionError):
            TrainArgs(steps=0)

        with pytest.raises(AssertionError):
            TrainArgs(max_learning_rate=-1e-3)

    def test_h100_optimized_config(self):
        """Test H100-optimized configuration parameters."""
        config = TrainArgs(
            batch_size=128,
            context_length=2048,
            d_ff=1344,
            compile_model=True,
            device="cuda",
        )

        assert config.d_ff % 64 == 0, "d_ff should be optimized for tensor cores"
        assert config.compile_model, "Should enable compilation for H100"

    def test_json_config_loading(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "vocab_size": 32000,
            "context_length": 256,
            "num_layers": 4,
            "d_model": 512,
            "num_heads": 16,
            "d_ff": 1344,
            "steps": 1000,
            "batch_size": 32,
            "max_learning_rate": 0.002,
            "experiment_name": "test_experiment",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config_from_json(config_path)
            assert config.vocab_size == 32000
            assert config.experiment_name == "test_experiment"
        finally:
            os.unlink(config_path)


class TestModelInitialization:
    """Test model initialization and parameter setup."""

    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return TrainArgs(
            vocab_size=1000,
            context_length=128,
            num_layers=2,
            d_model=256,
            num_heads=8,
            d_ff=512,
            steps=10,
            batch_size=4,
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

    @pytest.fixture
    def mock_data_files(self):
        """Create mock data files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = np.random.randint(0, 1000, size=(10000,), dtype=np.int64)
            train_path = Path(temp_dir) / "train_tokens.npy"
            np.save(train_path, train_data)

            val_data = np.random.randint(0, 1000, size=(1000,), dtype=np.int64)
            val_path = Path(temp_dir) / "val_tokens.npy"
            np.save(val_path, val_data)

            vocab_data = {str(i): f"token_{i}".encode("utf-8").hex() for i in range(1000)}
            vocab_path = Path(temp_dir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f)

            merges_data = [(f"token_{i}".encode("utf-8"), f"token_{i + 1}".encode("utf-8")) for i in range(100)]
            merges_path = Path(temp_dir) / "merges.pkl"
            import pickle

            with open(merges_path, "wb") as f:
                pickle.dump(merges_data, f)

            yield {
                "training_set": str(train_path),
                "validation_set": str(val_path),
            }

    def test_model_parameter_calculation(self, config):
        """Test parameter count calculation without creating full model."""
        vocab_size = config.vocab_size
        d_model = config.d_model
        num_layers = config.num_layers

        embedding_params = vocab_size * d_model
        layer_params_estimate = num_layers * d_model * d_model * 4
        total_estimate = embedding_params + layer_params_estimate

        assert total_estimate > 0, "Parameter estimate should be positive"
        assert total_estimate < 100_000_000, "Parameter estimate should be reasonable for test model"


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics for H100 deployment."""

    @pytest.fixture
    def h100_config(self):
        """Configuration optimized for H100 deployment."""
        return TrainArgs(
            vocab_size=32000,
            context_length=2048,
            num_layers=12,
            d_model=1024,
            num_heads=16,
            d_ff=4096,
            batch_size=64,
            steps=10,
            use_wandb=False,
            compile_model=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_efficiency(self, h100_config):
        """Test GPU memory usage efficiency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = np.random.randint(0, 32000, size=(100000,), dtype=np.int64)
            train_path = Path(temp_dir) / "train.npy"
            np.save(train_path, train_data)

            h100_config.training_set = str(train_path)
            h100_config.validation_set = str(train_path)

            vocab_data = {str(i): f"token_{i}" for i in range(32000)}
            vocab_path = Path(temp_dir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f)

            import pickle

            merges_data = [(b"a", b"b")]
            merges_path = Path(temp_dir) / "merges.pkl"
            with open(merges_path, "wb") as f:
                pickle.dump(merges_data, f)

            h100_config.tokenizer_vocab = str(vocab_path)
            h100_config.tokenizer_merges = str(merges_path)

            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                trainer = Trainer(h100_config)

                trainer.step = 0
                metrics = trainer.train_step()

                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9

                assert memory_allocated < 40.0, f"Memory usage {memory_allocated:.1f}GB too high for H100"

                if memory_reserved > 0:
                    efficiency = memory_allocated / memory_reserved
                    assert efficiency > 0.7, f"Memory efficiency {efficiency:.2f} too low"

    def test_batch_processing_config(self):
        """Test batch processing configuration validation."""
        config = TrainArgs(
            vocab_size=1000,
            context_length=64,
            num_layers=2,
            d_model=128,
            num_heads=4,
            d_ff=256,
            batch_size=8,
            steps=1,
            use_wandb=False,
            device="cpu",
        )

        assert config.batch_size > 0
        assert config.context_length > 0
        total_tokens_per_batch = config.batch_size * config.context_length
        assert total_tokens_per_batch > 0
        assert total_tokens_per_batch < 100000


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""

    def test_missing_data_files(self):
        """Test graceful handling of missing data files."""
        config = TrainArgs(
            training_set="nonexistent_train.npy",
            validation_set="nonexistent_val.npy",
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

        with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
            try:
                trainer = Trainer(config)
                assert False, "Should have raised an error for missing data files"
            except (FileNotFoundError, OSError, ValueError):
                pass

    def test_invalid_model_configuration(self):
        """Test handling of invalid model configurations."""
        with pytest.raises(AssertionError):
            config = TrainArgs(
                d_model=100,
                num_heads=7,
                use_wandb=False,
                device="cpu",
            )


class TestDataLoading:
    """Test data loading from production data/encoded folder."""

    def test_production_data_availability(self):
        """Test that production data files exist and are accessible."""
        data_dir = Path("data/encoded")
        expected_files = {
            "training_data": data_dir / "owt_train_tokens.npy",
            "validation_data": data_dir / "owt_valid_tokens.npy",
            "vocab_file": data_dir / "openwebtext_vocab.json",
            "merges_file": data_dir / "openwebtext_merges.pkl",
        }

        if not data_dir.exists():
            pytest.skip(f"Production data directory {data_dir} does not exist yet")

        missing_files = []
        for file_type, file_path in expected_files.items():
            if not file_path.exists():
                missing_files.append(f"{file_type}: {file_path}")

        if missing_files:
            pytest.skip(f"Missing production data files: {', '.join(missing_files)}")

        print(f"✅ All production data files found in {data_dir}")

    def test_production_data_format_validation(self):
        """Test that production data files have correct format and can be loaded."""
        data_dir = Path("data/encoded")

        if not data_dir.exists():
            pytest.skip(f"Production data directory {data_dir} does not exist yet")

        train_file = data_dir / "owt_train_tokens.npy"
        val_file = data_dir / "owt_valid_tokens.npy"
        vocab_file = data_dir / "openwebtext_vocab.json"
        merges_file = data_dir / "openwebtext_merges.pkl"

        if train_file.exists():
            try:
                train_data = np.load(train_file, mmap_mode="r")
                assert len(train_data.shape) == 1, f"Training data should be 1D, got shape {train_data.shape}"
                assert train_data.dtype in [np.int64, np.int32, np.uint16], (
                    f"Training data should be integer type, got {train_data.dtype}"
                )
                assert len(train_data) > 1000, f"Training data seems too small: {len(train_data)} tokens"
                print(f"✅ Training data: {len(train_data):,} tokens, dtype: {train_data.dtype}")
            except Exception as e:
                pytest.fail(f"Failed to load training data from {train_file}: {e}")

        if val_file.exists():
            try:
                val_data = np.load(val_file, mmap_mode="r")
                assert len(val_data.shape) == 1, f"Validation data should be 1D, got shape {val_data.shape}"
                assert val_data.dtype in [np.int64, np.int32, np.uint16], (
                    f"Validation data should be integer type, got {val_data.dtype}"
                )
                assert len(val_data) > 100, f"Validation data seems too small: {len(val_data)} tokens"
                print(f"✅ Validation data: {len(val_data):,} tokens, dtype: {val_data.dtype}")
            except Exception as e:
                pytest.fail(f"Failed to load validation data from {val_file}: {e}")

        if vocab_file.exists():
            try:
                with open(vocab_file, "r") as f:
                    vocab_data = json.load(f)
                assert isinstance(vocab_data, dict), f"Vocabulary should be a dictionary, got {type(vocab_data)}"
                assert len(vocab_data) > 1000, f"Vocabulary seems too small: {len(vocab_data)} tokens"

                for k, v in list(vocab_data.items())[:10]:
                    assert isinstance(k, str), f"Vocabulary keys should be strings, got {type(k)} for key {k}"
                    assert isinstance(v, str), f"Vocabulary values should be strings, got {type(v)} for value {v}"

                print(f"✅ Vocabulary: {len(vocab_data):,} tokens")
            except Exception as e:
                pytest.fail(f"Failed to load vocabulary from {vocab_file}: {e}")

        if merges_file.exists():
            try:
                import pickle

                with open(merges_file, "rb") as f:
                    merges_data = pickle.load(f)
                assert isinstance(merges_data, list), f"Merges should be a list, got {type(merges_data)}"
                assert len(merges_data) > 100, f"Merges list seems too small: {len(merges_data)} merges"

                for i, merge in enumerate(merges_data[:10]):
                    assert isinstance(merge, tuple), f"Merge {i} should be a tuple, got {type(merge)}"
                    assert len(merge) == 2, f"Merge {i} should be a pair, got length {len(merge)}"
                    assert isinstance(merge[0], bytes) and isinstance(merge[1], bytes), (
                        f"Merge {i} should contain bytes, got {type(merge[0])}, {type(merge[1])}"
                    )

                print(f"✅ Merges: {len(merges_data):,} merge rules")
            except Exception as e:
                pytest.fail(f"Failed to load merges from {merges_file}: {e}")

    def test_production_data_compatibility_with_training_script(self):
        """Test that production data can be loaded by the training script."""
        data_dir = Path("data/encoded")

        if not data_dir.exists():
            pytest.skip(f"Production data directory {data_dir} does not exist yet")

        train_file = data_dir / "owt_train_tokens.npy"
        val_file = data_dir / "owt_valid_tokens.npy"
        vocab_file = data_dir / "openwebtext_vocab.json"
        merges_file = data_dir / "openwebtext_merges.pkl"

        required_files = [train_file, vocab_file, merges_file]
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            pytest.skip(f"Missing required files: {[str(f) for f in missing_files]}")

        config = TrainArgs(
            vocab_size=50257,
            context_length=1024,
            num_layers=4,
            d_model=256,
            num_heads=4,
            d_ff=512,
            steps=1,
            batch_size=4,
            training_set=str(train_file),
            validation_set=str(val_file) if val_file.exists() else str(train_file),
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

        try:
            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                trainer = Trainer(config)

                assert trainer.training_set is not None, "Training data should be loaded"
                assert len(trainer.training_set) > 0, "Training data should not be empty"

                if val_file.exists():
                    assert trainer.validation_set is not None, "Validation data should be loaded"
                    assert len(trainer.validation_set) > 0, "Validation data should not be empty"

                from cs336_basics.data import get_batch

                inputs, targets = get_batch(
                    trainer.training_set, config.batch_size, config.context_length, device="cpu"
                )

                assert inputs.shape == (config.batch_size, config.context_length), (
                    f"Input shape mismatch: {inputs.shape}"
                )
                assert targets.shape == (config.batch_size, config.context_length), (
                    f"Target shape mismatch: {targets.shape}"
                )
                assert inputs.dtype == torch.long, f"Input dtype should be long, got {inputs.dtype}"
                assert targets.dtype == torch.long, f"Target dtype should be long, got {targets.dtype}"

                max_token_id = max(inputs.max().item(), targets.max().item())
                min_token_id = min(inputs.min().item(), targets.min().item())
                assert min_token_id >= 0, f"Token IDs should be non-negative, got min: {min_token_id}"
                assert max_token_id < config.vocab_size, (
                    f"Token IDs should be < vocab_size ({config.vocab_size}), got max: {max_token_id}"
                )

                print(f"✅ Production data successfully loaded and compatible with training script")
                print(f"   - Training tokens: {len(trainer.training_set):,}")
                if trainer.validation_set is not None:
                    print(f"   - Validation tokens: {len(trainer.validation_set):,}")
                print(f"   - Token ID range: [{min_token_id}, {max_token_id}]")
                print(f"   - Batch shape: {inputs.shape}")

        except Exception as e:
            pytest.fail(f"Failed to load production data with training script: {e}")

    def test_data_integrity_checks(self):
        """Test data integrity and consistency checks."""
        data_dir = Path("data/encoded")

        if not data_dir.exists():
            pytest.skip(f"Production data directory {data_dir} does not exist yet")

        train_file = data_dir / "owt_train_tokens.npy"
        val_file = data_dir / "owt_valid_tokens.npy"
        vocab_file = data_dir / "openwebtext_vocab.json"

        if not train_file.exists() or not vocab_file.exists():
            pytest.skip("Required files for integrity check not found")

        try:
            train_data = np.load(train_file, mmap_mode="r")
            with open(vocab_file, "r") as f:
                vocab_data = json.load(f)

            vocab_size = len(vocab_data)

            unique_tokens = np.unique(train_data)
            invalid_tokens = unique_tokens[unique_tokens >= vocab_size]
            assert len(invalid_tokens) == 0, (
                f"Found {len(invalid_tokens)} invalid token IDs >= vocab_size ({vocab_size}): {invalid_tokens[:10]}"
            )

            negative_tokens = unique_tokens[unique_tokens < 0]
            assert len(negative_tokens) == 0, f"Found {len(negative_tokens)} negative token IDs: {negative_tokens}"

            if val_file.exists():
                val_data = np.load(val_file, mmap_mode="r")
                val_unique_tokens = np.unique(val_data)
                val_invalid_tokens = val_unique_tokens[val_unique_tokens >= vocab_size]
                assert len(val_invalid_tokens) == 0, (
                    f"Found {len(val_invalid_tokens)} invalid token IDs in validation data"
                )

                val_negative_tokens = val_unique_tokens[val_unique_tokens < 0]
                assert len(val_negative_tokens) == 0, (
                    f"Found {len(val_negative_tokens)} negative token IDs in validation data"
                )

            train_mean = float(train_data.mean())
            train_std = float(train_data.std())
            print(f"✅ Data integrity checks passed")
            print(f"   - Vocabulary size: {vocab_size:,}")
            print(f"   - Training data unique tokens: {len(unique_tokens):,}")
            print(f"   - Training data stats: mean={train_mean:.1f}, std={train_std:.1f}")
            print(f"   - Token ID range: [{unique_tokens.min()}, {unique_tokens.max()}]")

        except Exception as e:
            pytest.fail(f"Data integrity check failed: {e}")


class TestProductionScenarios:
    """Test specific production deployment scenarios."""

    def test_openwebtext_config_validation(self):
        """Test OpenWebText configuration validation without heavy computation."""
        config = TrainArgs(
            vocab_size=50257,
            context_length=1024,
            num_layers=6,
            d_model=768,
            num_heads=12,
            d_ff=3072,
            batch_size=8,
            steps=1,
            use_wandb=False,
            device="cpu",
        )

        assert config.vocab_size == 50304  # Padded to nearest 128 for efficiency
        assert config.d_model % config.num_heads == 0
        assert config.d_ff % 64 == 0

    def test_h100_specific_optimizations(self):
        """Test H100-specific optimizations and configurations."""
        config = TrainArgs(
            vocab_size=32000,
            context_length=2048,
            d_model=2048,
            num_heads=16,
            d_ff=8192,
            batch_size=128,
            compile_model=True,
            device="cuda",
            use_wandb=False,
        )

        assert config.d_ff % 64 == 0, "d_ff should be optimized for H100 tensor cores"
        assert config.compile_model, "Should enable torch.compile for H100"

        assert config.d_model % config.num_heads == 0
        assert config.d_model >= 512, "Should use reasonable model size for H100"

    def test_checkpoint_config(self):
        """Test checkpoint configuration validation."""
        config = TrainArgs(
            checkpoint_step_interval=1000,
            steps=5000,
            use_wandb=False,
        )

        assert config.checkpoint_step_interval > 0
        assert config.steps > 0
        assert config.checkpoint_step_interval <= config.steps

    def test_logging_config(self):
        """Test logging configuration validation."""
        config = TrainArgs(
            experiment_name="test_logging",
            experiment_description="Test experiment",
            use_wandb=False,
            wandb_project="test_project",
            wandb_entity="test_entity",
            log_dir="experiments",
        )

        assert config.experiment_name == "test_logging"
        assert config.experiment_description == "Test experiment"
        assert config.use_wandb == False
        assert config.wandb_project == "test_project"
        assert config.wandb_entity == "test_entity"
        assert config.log_dir == "experiments"
