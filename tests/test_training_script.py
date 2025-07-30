"""
Unit tests for the training script to ensure stability and prevent configuration errors.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cs336_basics.scripts.train_transformer import TrainArgs, load_config_from_json


class TestTrainArgs(unittest.TestCase):
    """Test TrainArgs dataclass validation."""

    def test_default_trainargs_creation(self):
        """Test that TrainArgs can be created with defaults."""
        config = TrainArgs()
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.d_model, 1280)
        self.assertEqual(config.num_heads, 20)

    def test_trainargs_validation_valid_config(self):
        """Test TrainArgs with valid configuration."""
        # Suppress print statement for vocab_size padding
        with patch("builtins.print"):
            config = TrainArgs(
                vocab_size=1000,
                context_length=512,
                d_model=256,
                num_heads=8,
                steps=100,
                batch_size=4,
                max_learning_rate=0.001,
            )
        # Should not raise any exceptions
        self.assertEqual(config.vocab_size, 1024)  # Should be padded to multiple of 128

    def test_trainargs_validation_invalid_heads(self):
        """Test TrainArgs validation fails with invalid head configuration."""
        with self.assertRaises(AssertionError):
            TrainArgs(d_model=256, num_heads=7)  # 256 % 7 != 0

    def test_trainargs_validation_negative_values(self):
        """Test TrainArgs validation fails with negative values."""
        with self.assertRaises(AssertionError):
            TrainArgs(vocab_size=-1)

        with self.assertRaises(AssertionError):
            TrainArgs(context_length=0)

        with self.assertRaises(AssertionError):
            TrainArgs(steps=-1)

    def test_vocab_size_padding(self):
        """Test that vocab_size gets padded to multiple of 128."""
        # Test padding behavior by suppressing the print statement
        with patch("builtins.print"):
            config = TrainArgs(vocab_size=1000)
            self.assertEqual(config.vocab_size, 1024)  # 1000 -> 1024 (multiple of 128)

        config = TrainArgs(vocab_size=1024)
        self.assertEqual(config.vocab_size, 1024)  # Already multiple of 128


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading from JSON files."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def create_temp_config(self, config_dict):
        """Helper to create temporary config file."""
        config_path = self.temp_path / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
        return str(config_path)

    def test_load_valid_config(self):
        """Test loading a valid configuration."""
        valid_config = {
            "vocab_size": 1024,  # Use multiple of 128 to avoid padding print
            "context_length": 512,
            "d_model": 256,
            "num_heads": 8,
            "steps": 100,
            "batch_size": 4,
            "max_learning_rate": 0.001,
        }
        config_path = self.create_temp_config(valid_config)

        args = load_config_from_json(config_path)
        self.assertIsInstance(args, TrainArgs)
        self.assertEqual(args.context_length, 512)
        self.assertEqual(args.num_heads, 8)

    def test_load_config_with_unexpected_fields(self):
        """Test loading config with unexpected fields (should warn and ignore)."""
        config_with_extra = {
            "vocab_size": 1024,  # Use multiple of 128 to avoid padding print
            "context_length": 512,
            "d_model": 256,
            "num_heads": 8,
            "unexpected_field": "should_be_ignored",
            "another_unknown": 42,
        }
        config_path = self.create_temp_config(config_with_extra)

        with patch("builtins.print") as mock_print:
            args = load_config_from_json(config_path)
            self.assertIsInstance(args, TrainArgs)
            mock_print.assert_called_once()
            self.assertIn("Unexpected fields", mock_print.call_args[0][0])

    def test_load_nonexistent_config(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config_from_json("/nonexistent/path/config.json")

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        invalid_json_path = self.temp_path / "invalid.json"
        with open(invalid_json_path, "w") as f:
            f.write("{invalid json content")

        with self.assertRaises(ValueError) as cm:
            load_config_from_json(str(invalid_json_path))
        self.assertIn("Invalid JSON", str(cm.exception))

    def test_load_config_with_type_mismatch(self):
        """Test loading config with wrong types."""
        config_with_wrong_types = {
            "vocab_size": "not_a_number",  # Should be int
            "context_length": 512,
            "d_model": 256,
            "num_heads": 8,
        }
        config_path = self.create_temp_config(config_with_wrong_types)

        with self.assertRaises(TypeError):
            load_config_from_json(config_path)


class TestTrainingStability(unittest.TestCase):
    """Test training stability components."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_training_components_cpu_fallback(self, mock_cuda):
        """Test that training components work on CPU as fallback."""
        # Test basic config creation (suppress padding print)
        with patch("builtins.print"):
            config = TrainArgs(
                vocab_size=1000,
                context_length=128,
                d_model=64,
                num_heads=4,
                num_layers=2,
                steps=10,
                batch_size=2,
                device="cpu",
            )
        self.assertEqual(config.device, "cpu")

    def test_memory_estimation(self):
        """Test memory estimation calculations."""
        config = TrainArgs(vocab_size=1024, d_model=256, num_layers=6, num_heads=8)  # Use multiple of 128

        # Basic parameter count estimation
        embedding_params = config.vocab_size * config.d_model
        self.assertEqual(embedding_params, 262144)  # 1024 * 256 = 262144

        # Should be reasonable memory footprint
        self.assertGreater(embedding_params, 0)
        self.assertLess(embedding_params, 1e9)  # Less than 1B params for test config


class TestConfigFieldMapping(unittest.TestCase):
    """Test that all TrainArgs fields are properly handled."""

    def test_all_trainargs_fields_documented(self):
        """Test that we know about all TrainArgs fields."""
        from dataclasses import fields

        trainargs_fields = {field.name for field in fields(TrainArgs)}

        # Expected fields based on current TrainArgs
        expected_fields = {
            "vocab_size",
            "context_length",
            "num_layers",
            "d_model",
            "num_heads",
            "d_ff",
            "rope_theta",
            "window_size",
            "use_qk_norm",
            "use_flex_attention",
            "use_swiglu",
            "tie_embeddings",
            "weight_decay",
            "betas",
            "muon_momentum",
            "eps",
            "max_learning_rate",
            "min_learning_rate",
            "warmup_iters",
            "cosine_cycle_iters",
            "training_set",
            "validation_set",
            "validation_step_interval",
            "checkpoint_step_interval",
            "steps",
            "batch_size",
            "gradient_accumulation_steps",
            "gradient_clipping",
            "device",
            "compile_model",
            "compile_mode",
            "use_mixed_precision",
            "use_efficient_attention",
            "use_fused_kernels",
            "training_mode",
            "enable_stability_monitoring",
            "gradient_norm_threshold",
            "loss_spike_threshold",
            "nan_tolerance",
            "enable_stable_initialization",
            "enable_torch_optimizations",
            "eval_batch_count",
            "experiment_name",
            "experiment_description",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
            "log_dir",
            "use_activation_checkpointing",
            "checkpoint_pattern",
            "use_memory_efficient_attention",
            "optimize_memory_layout",
            "use_adaptive_gradient_clipping",
            "adaptive_clipping_method",
            "zclip_zscore_threshold",
            "zclip_min_clip",
            "zclip_max_clip",
            "zclip_warmup_steps",
            "use_muon_optimizer",
            "muon_ns_iters",
            "enable_outlier_safe_training",
        }

        # Check that we're not missing any fields
        self.assertEqual(
            trainargs_fields,
            expected_fields,
            f"Field mismatch. Missing: {expected_fields - trainargs_fields}, "
            f"Extra: {trainargs_fields - expected_fields}",
        )

    def test_config_completeness(self):
        """Test that our configs contain all necessary fields."""
        config_path = "cs336_basics/scripts/configs/h100_optimal_config.json"

        if Path(config_path).exists():
            with open(config_path) as f:
                config_dict = json.load(f)

            from dataclasses import fields

            trainargs_fields = {field.name for field in fields(TrainArgs)}
            config_fields = set(config_dict.keys())

            # Check that all required fields are present
            missing_required = []
            for field in fields(TrainArgs):
                if field.name not in config_dict and field.default == field.default_factory:
                    missing_required.append(field.name)

            self.assertEqual(missing_required, [], f"Required fields missing from config: {missing_required}")
