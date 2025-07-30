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

from cs336_basics.scripts.train_transformer import TrainModel, TrainModelArgs, load_config_from_json


class TestProductionConfiguration:
    """Test configuration validation and production readiness."""

    def test_config_validation_with_valid_params(self):
        """Test that valid production configurations pass validation."""
        config = TrainModelArgs(
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
            TrainModelArgs(vocab_size=0)

        with pytest.raises(AssertionError):
            TrainModelArgs(d_model=513, num_heads=16)

        with pytest.raises(AssertionError):
            TrainModelArgs(steps=0)

        with pytest.raises(AssertionError):
            TrainModelArgs(max_learning_rate=-1e-3)

    def test_h100_optimized_config(self):
        """Test H100-optimized configuration parameters."""
        config = TrainModelArgs(
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
        return TrainModelArgs(
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
                "tokenizer_vocab": str(vocab_path),
                "tokenizer_merges": str(merges_path),
            }

    def test_model_parameter_count(self, config, mock_data_files):
        """Test that model has expected parameter count."""
        config.training_set = mock_data_files["training_set"]
        config.validation_set = mock_data_files["validation_set"]
        config.tokenizer_vocab = mock_data_files["tokenizer_vocab"]
        config.tokenizer_merges = mock_data_files["tokenizer_merges"]

        with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
            trainer = TrainModel(config)

            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

            assert total_params > 0, "Model should have parameters"
            assert trainable_params == total_params, "All parameters should be trainable"

            expected_range = (100_000, 2_000_000)
            assert expected_range[0] < total_params < expected_range[1], (
                f"Parameter count {total_params} outside expected range"
            )

    def test_model_device_placement(self, config, mock_data_files):
        """Test that model is correctly placed on specified device."""
        config.training_set = mock_data_files["training_set"]
        config.validation_set = mock_data_files["validation_set"]
        config.tokenizer_vocab = mock_data_files["tokenizer_vocab"]
        config.tokenizer_merges = mock_data_files["tokenizer_merges"]
        config.device = "cpu"

        with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
            trainer = TrainModel(config)

            for param in trainer.model.parameters():
                assert param.device.type == "cpu", f"Parameter on wrong device: {param.device}"

    def test_optimizer_initialization(self, config, mock_data_files):
        """Test that optimizer is properly initialized."""
        config.training_set = mock_data_files["training_set"]
        config.validation_set = mock_data_files["validation_set"]
        config.tokenizer_vocab = mock_data_files["tokenizer_vocab"]
        config.tokenizer_merges = mock_data_files["tokenizer_merges"]

        with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
            trainer = TrainModel(config)

            assert len(trainer.optimizer.param_groups) == 1
            param_group = trainer.optimizer.param_groups[0]
            assert param_group["lr"] == config.max_learning_rate
            assert param_group["weight_decay"] == config.weight_decay
            assert param_group["betas"] == config.betas


class TestTrainingLoop:
    """Test training loop integrity and correctness."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer for testing."""
        torch.manual_seed(42)
        np.random.seed(42)

        config = TrainModelArgs(
            vocab_size=1000,
            context_length=64,
            num_layers=2,
            d_model=128,
            num_heads=4,
            d_ff=256,
            steps=5,
            batch_size=2,
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            np.random.seed(42)
            train_data = np.random.randint(0, 1000, size=(1000,), dtype=np.int64)
            train_path = Path(temp_dir) / "train.npy"
            np.save(train_path, train_data)

            val_data = np.random.randint(0, 1000, size=(200,), dtype=np.int64)
            val_path = Path(temp_dir) / "val.npy"
            np.save(val_path, val_data)

            vocab_data = {str(i): f"token_{i}" for i in range(1000)}
            vocab_path = Path(temp_dir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f)

            import pickle

            merges_data = [(b"a", b"b"), (b"c", b"d")]
            merges_path = Path(temp_dir) / "merges.pkl"
            with open(merges_path, "wb") as f:
                pickle.dump(merges_data, f)

            config.training_set = str(train_path)
            config.validation_set = str(val_path)
            config.tokenizer_vocab = str(vocab_path)
            config.tokenizer_merges = str(merges_path)

            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                trainer = TrainModel(config)
                yield trainer

    def test_gradient_flow(self, trainer):
        """Test that all model parameters receive gradients during training."""
        initial_params = {}
        for name, param in trainer.model.named_parameters():
            initial_params[name] = param.clone().detach()

        trainer.step = 0
        metrics = trainer.train_step()

        for name, param in trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Parameter {name} has zero gradient"

        assert metrics["loss"] > 0, "Loss should be positive"
        assert not math.isnan(metrics["loss"]), "Loss should not be NaN"
        assert not math.isinf(metrics["loss"]), "Loss should not be infinite"

    def test_parameter_updates(self, trainer):
        """Test that parameters actually change during training step."""
        initial_params = {}
        for name, param in trainer.model.named_parameters():
            initial_params[name] = param.clone().detach()

        test_step = trainer.args.warmup_iters // 2
        trainer.step = test_step

        metrics = trainer.train_step()

        parameters_changed = 0
        total_change = 0.0
        param_changes = {}
        for name, param in trainer.model.named_parameters():
            diff = torch.norm(param - initial_params[name]).item()
            total_change += diff
            param_changes[name] = diff
            if diff > 1e-8:
                parameters_changed += 1

        assert parameters_changed > 0, (
            f"No parameters were updated during training step. Total change: {total_change:.2e}"
        )

        total_params = len(list(trainer.model.named_parameters()))
        change_ratio = parameters_changed / total_params
        assert change_ratio > 0.3, f"Only {change_ratio:.2%} of parameters changed, expected > 30%"

    def test_learning_rate_schedule(self, trainer):
        """Test that learning rate follows expected schedule."""
        trainer.step = 0
        lr_0 = trainer.get_lr(0)
        assert lr_0 == 0.0, "Learning rate should start at 0"

        trainer.step = trainer.args.warmup_iters // 2
        lr_mid = trainer.get_lr(trainer.step)
        assert 0 < lr_mid < trainer.args.max_learning_rate, "Learning rate should be between 0 and max during warmup"

        trainer.step = trainer.args.warmup_iters
        lr_max = trainer.get_lr(trainer.step)
        assert abs(lr_max - trainer.args.max_learning_rate) < 1e-6, "Learning rate should reach max at end of warmup"

        trainer.step = trainer.args.cosine_cycle_iters // 2
        lr_decay = trainer.get_lr(trainer.step)
        assert lr_decay < trainer.args.max_learning_rate, "Learning rate should decay after warmup"

    def test_gradient_clipping(self, trainer):
        """Test that gradient clipping prevents exploding gradients."""
        test_step = trainer.args.warmup_iters // 2
        trainer.step = test_step

        metrics = trainer.train_step()

        original_grads = {}
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        large_grad_value = 100.0
        for param in trainer.model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(large_grad_value)

        pre_clip_norm = torch.sqrt(
            sum(torch.sum(p.grad**2) for p in trainer.model.parameters() if p.grad is not None)
        ).item()

        from cs336_basics.training.gradient_clipping import gradient_clipping

        returned_norm = gradient_clipping(trainer.model.parameters(), trainer.args.gradient_clipping)

        post_clip_norm = torch.sqrt(
            sum(torch.sum(p.grad**2) for p in trainer.model.parameters() if p.grad is not None)
        ).item()

        norm_diff = abs(returned_norm - pre_clip_norm)
        assert norm_diff < 100, (
            f"Returned norm {returned_norm} differs too much from calculated pre-clip norm {pre_clip_norm} (diff: {norm_diff})"
        )

        assert pre_clip_norm > trainer.args.gradient_clipping, (
            f"Initial gradient norm {pre_clip_norm} should be larger than threshold {trainer.args.gradient_clipping}"
        )
        assert post_clip_norm <= trainer.args.gradient_clipping + 1e-3, (
            f"Post-clipping gradient norm {post_clip_norm} exceeds threshold {trainer.args.gradient_clipping} (tolerance: 1e-3)"
        )

        gradients_changed = False
        for name, param in trainer.model.named_parameters():
            if param.grad is not None and name in original_grads:
                if not torch.allclose(param.grad, torch.full_like(param.grad, large_grad_value)):
                    gradients_changed = True
                    break

        assert gradients_changed, "Gradients should have been modified by clipping"

    def test_mfu_calculation(self, trainer):
        """Test Model FLOPs Utilization calculation."""
        tokens_per_sec = 1000.0
        mfu = trainer.calculate_mfu(tokens_per_sec)

        assert 0.0 <= mfu <= 1.0, f"MFU {mfu} should be between 0 and 1"

        mfu_high = trainer.calculate_mfu(10000.0)
        assert mfu_high > mfu, "Higher tokens/sec should yield higher MFU"

    def test_evaluation_consistency(self, trainer):
        """Test that evaluation produces consistent results with deterministic setup."""
        torch.manual_seed(42)
        np.random.seed(42)

        trainer.model.eval()
        results = []
        for i in range(3):
            torch.manual_seed(42)
            np.random.seed(42)
            val_loss, val_perplexity = trainer.evaluate()
            results.append((val_loss, val_perplexity))

        for i in range(1, len(results)):
            loss_diff = abs(results[i][0] - results[0][0])
            perp_diff = abs(results[i][1] - results[0][1])
            assert loss_diff < 0.1, f"Evaluation loss difference too large: {loss_diff}"
            assert perp_diff < 5.0, f"Evaluation perplexity difference too large: {perp_diff}"


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics for H100 deployment."""

    @pytest.fixture
    def h100_config(self):
        """Configuration optimized for H100 deployment."""
        return TrainModelArgs(
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

                trainer = TrainModel(h100_config)

                trainer.step = 0
                metrics = trainer.train_step()

                memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB

                assert memory_allocated < 40.0, f"Memory usage {memory_allocated:.1f}GB too high for H100"

                if memory_reserved > 0:
                    efficiency = memory_allocated / memory_reserved
                    assert efficiency > 0.7, f"Memory efficiency {efficiency:.2f} too low"

    def test_batch_processing_speed(self):
        """Test that batch processing meets performance requirements."""
        config = TrainModelArgs(
            vocab_size=10000,
            context_length=512,
            num_layers=4,
            d_model=512,
            num_heads=8,
            d_ff=1024,
            batch_size=32,
            steps=3,
            use_wandb=False,
            compile_model=False,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = np.random.randint(0, 10000, size=(50000,), dtype=np.int64)
            train_path = Path(temp_dir) / "train.npy"
            np.save(train_path, train_data)

            config.training_set = str(train_path)
            config.validation_set = str(train_path)

            vocab_data = {str(i): f"token_{i}" for i in range(10000)}
            vocab_path = Path(temp_dir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f)

            import pickle

            merges_data = [(b"a", b"b")]
            merges_path = Path(temp_dir) / "merges.pkl"
            with open(merges_path, "wb") as f:
                pickle.dump(merges_data, f)

            config.tokenizer_vocab = str(vocab_path)
            config.tokenizer_merges = str(merges_path)

            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                trainer = TrainModel(config)

                times = []
                for i in range(3):
                    trainer.step = i
                    start_time = time.time()
                    metrics = trainer.train_step()
                    step_time = time.time() - start_time
                    times.append(step_time)

                avg_time = sum(times) / len(times)
                tokens_per_sec = (config.batch_size * config.context_length) / avg_time

                min_tokens_per_sec = 100
                assert tokens_per_sec > min_tokens_per_sec, f"Too slow: {tokens_per_sec:.0f} tokens/sec"


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""

    def test_missing_data_files(self):
        """Test graceful handling of missing data files."""
        config = TrainModelArgs(
            training_set="nonexistent_train.npy",
            validation_set="nonexistent_val.npy",
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

        with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
            try:
                trainer = TrainModel(config)
                assert False, "Should have raised an error for missing data files"
            except (FileNotFoundError, OSError, ValueError):
                pass

    def test_invalid_model_configuration(self):
        """Test handling of invalid model configurations."""
        with pytest.raises(AssertionError):
            config = TrainModelArgs(
                d_model=100,
                num_heads=7,
                use_wandb=False,
                device="cpu",
            )

    def test_cuda_unavailable_fallback(self):
        """Test fallback behavior when CUDA is unavailable."""
        config = TrainModelArgs(
            device="cpu",
            use_wandb=False,
            compile_model=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = np.random.randint(0, 1000, size=(1000,), dtype=np.int64)
            np.save(Path(temp_dir) / "train.npy", train_data)
            np.save(Path(temp_dir) / "val.npy", train_data)

            vocab_data = {str(i): f"token_{i}" for i in range(1000)}
            with open(Path(temp_dir) / "vocab.json", "w") as f:
                json.dump(vocab_data, f)

            import pickle

            with open(Path(temp_dir) / "merges.pkl", "wb") as f:
                pickle.dump([(b"a", b"b")], f)

            config.training_set = str(Path(temp_dir) / "train.npy")
            config.validation_set = str(Path(temp_dir) / "val.npy")
            config.tokenizer_vocab = str(Path(temp_dir) / "vocab.json")
            config.tokenizer_merges = str(Path(temp_dir) / "merges.pkl")

            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                trainer = TrainModel(config)
                assert trainer.device.type == "cpu"


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
                assert np.issubdtype(train_data.dtype, np.integer), (
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
                assert np.issubdtype(val_data.dtype, np.integer), (
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

        config = TrainModelArgs(
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
            tokenizer_vocab=str(vocab_file),
            tokenizer_merges=str(merges_file),
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

        try:
            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                trainer = TrainModel(config)

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

    def test_openwebtext_data_compatibility(self):
        """Test compatibility with OpenWebText data format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_size = 50257

            train_data = np.random.randint(0, vocab_size, size=(1000000,), dtype=np.int64)
            train_path = Path(temp_dir) / "owt_train_tokens.npy"
            np.save(train_path, train_data)

            val_data = np.random.randint(0, vocab_size, size=(50000,), dtype=np.int64)
            val_path = Path(temp_dir) / "owt_valid_tokens.npy"
            np.save(val_path, val_data)

            config = TrainModelArgs(
                vocab_size=vocab_size,
                context_length=1024,
                num_layers=6,
                d_model=768,
                num_heads=12,
                d_ff=3072,
                batch_size=8,
                steps=5,
                training_set=str(train_path),
                validation_set=str(val_path),
                use_wandb=False,
                compile_model=False,
                device="cpu",
            )

            vocab_data = {str(i): f"token_{i}" for i in range(vocab_size)}
            vocab_path = Path(temp_dir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f)

            import pickle

            merges_data = [(b"a", b"b")]
            merges_path = Path(temp_dir) / "merges.pkl"
            with open(merges_path, "wb") as f:
                pickle.dump(merges_data, f)

            config.tokenizer_vocab = str(vocab_path)
            config.tokenizer_merges = str(merges_path)

            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                trainer = TrainModel(config)

                trainer.step = 0
                metrics = trainer.train_step()

                assert metrics["loss"] > 0
                assert not math.isnan(metrics["loss"])

                val_loss, val_perplexity = trainer.evaluate()
                assert val_loss > 0
                assert val_perplexity > 1.0

    def test_h100_specific_optimizations(self):
        """Test H100-specific optimizations and configurations."""
        config = TrainModelArgs(
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

    def test_checkpoint_functionality(self):
        """Test checkpoint saving and loading functionality."""
        config = TrainModelArgs(
            vocab_size=1000,
            context_length=64,
            num_layers=2,
            d_model=128,
            num_heads=4,
            d_ff=256,
            steps=2,
            batch_size=2,
            checkpoint_step_interval=1,
            use_wandb=False,
            device="cpu",
            compile_model=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = np.random.randint(0, 1000, size=(1000,), dtype=np.int64)
            np.save(Path(temp_dir) / "train.npy", train_data)
            np.save(Path(temp_dir) / "val.npy", train_data)

            vocab_data = {str(i): f"token_{i}" for i in range(1000)}
            with open(Path(temp_dir) / "vocab.json", "w") as f:
                json.dump(vocab_data, f)

            import pickle

            with open(Path(temp_dir) / "merges.pkl", "wb") as f:
                pickle.dump([(b"a", b"b")], f)

            config.training_set = str(Path(temp_dir) / "train.npy")
            config.validation_set = str(Path(temp_dir) / "val.npy")
            config.tokenizer_vocab = str(Path(temp_dir) / "vocab.json")
            config.tokenizer_merges = str(Path(temp_dir) / "merges.pkl")

            old_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                with patch("cs336_basics.experiments.exp_logging.ExperimentLogger"):
                    trainer = TrainModel(config)

                    trainer.step = 0
                    trainer.train_step()

                    from cs336_basics.training.checkpoint import save_checkpoint

                    checkpoint_path = "test_checkpoint.pt"
                    save_checkpoint(trainer.model, trainer.optimizer, trainer.step, checkpoint_path)

                    assert Path(checkpoint_path).exists(), "Checkpoint file should be created"

                    from cs336_basics.training.checkpoint import load_checkpoint

                    new_model = type(trainer.model)(
                        vocab_size=config.vocab_size,
                        context_length=config.context_length,
                        d_model=config.d_model,
                        num_layers=config.num_layers,
                        num_heads=config.num_heads,
                        d_ff=config.d_ff,
                        device=trainer.device,
                    )
                    new_optimizer = type(trainer.optimizer)(new_model.parameters())

                    loaded_step = load_checkpoint(checkpoint_path, new_model, new_optimizer)
                    assert loaded_step == trainer.step, "Loaded step should match saved step"

            finally:
                os.chdir(old_cwd)

    def test_logging_integration(self):
        """Test comprehensive logging functionality."""
        config = TrainModelArgs(
            vocab_size=1000,
            context_length=32,
            num_layers=2,
            d_model=64,
            num_heads=4,
            d_ff=128,
            steps=2,
            batch_size=2,
            use_wandb=False,
            device="cpu",
            experiment_name="test_logging",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_data = np.random.randint(0, 1000, size=(500,), dtype=np.int64)
            np.save(Path(temp_dir) / "train.npy", train_data)
            np.save(Path(temp_dir) / "val.npy", train_data)

            vocab_data = {str(i): f"token_{i}" for i in range(1000)}
            with open(Path(temp_dir) / "vocab.json", "w") as f:
                json.dump(vocab_data, f)

            import pickle

            with open(Path(temp_dir) / "merges.pkl", "wb") as f:
                pickle.dump([(b"a", b"b")], f)

            config.training_set = str(Path(temp_dir) / "train.npy")
            config.validation_set = str(Path(temp_dir) / "val.npy")
            config.tokenizer_vocab = str(Path(temp_dir) / "vocab.json")
            config.tokenizer_merges = str(Path(temp_dir) / "merges.pkl")
            config.log_dir = temp_dir

            trainer = TrainModel(config)

            log_dir = Path(temp_dir) / "test_logging"
            assert log_dir.exists(), "Experiment log directory should be created"

            metadata_file = log_dir / "metadata.json"
            assert metadata_file.exists(), "Metadata file should be created"

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert metadata["name"] == "test_logging"
            assert "hyperparameters" in metadata
            assert "system_info" in metadata

            trainer.step = 0
            metrics = trainer.train_step()

            assert "loss" in metrics
            assert "lr" in metrics
            assert "grad_norm" in metrics
            assert "step_time" in metrics
