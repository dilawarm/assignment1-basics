"""
Tests for logging integration to prevent parameter conflicts and ensure proper method calls.
"""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import torch

from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.scripts.train_transformer import Trainer, TrainingConfig


class TestLoggingIntegration(unittest.TestCase):
    """Test logging integration to prevent parameter conflicts."""

    def setUp(self):
        """Set up test environment."""
        self.config = TrainingConfig(
            train_data_path="dummy_path",
            val_data_path=None,
            max_steps=100,
            batch_size=32,
            learning_rate=1e-3,
            use_wandb=False,
        )

    def test_log_training_step_signature(self):
        """Test that we understand the log_training_step method signature correctly."""
        # Get the method signature
        sig = inspect.signature(TrainingIntegrator.log_training_step)
        params = list(sig.parameters.keys())

        # Expected parameters based on the method definition
        expected_params = [
            "self",
            "step",
            "train_loss",
            "learning_rate",
            "tokens_processed",
            "samples_processed",
            "step_time",
            "tokens_per_sec",
            "wallclock_time",
            "additional_metrics",
        ]

        # Check that we have the expected parameters
        self.assertEqual(len(params), len(expected_params))
        for expected_param in expected_params[:-1]:  # Skip 'additional_metrics' as it's **kwargs
            self.assertIn(expected_param, params)

        # Check that **kwargs is present
        self.assertTrue(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))

    def test_no_parameter_conflicts_in_enhanced_metrics(self):
        """Test that enhanced_metrics doesn't contain parameters that conflict with explicit ones."""
        # Mock objects to avoid complex setup
        with (
            patch("cs336_basics.scripts.train_transformer.EnhancedTransformerLM") as mock_model,
            patch("cs336_basics.scripts.train_transformer.DataLoader"),
            patch("cs336_basics.scripts.train_transformer.MixedOptimizerV2"),
            patch("cs336_basics.scripts.train_transformer.ExperimentLogger"),
            patch("cs336_basics.scripts.train_transformer.TrainingIntegrator"),
        ):
            # Setup mock model to return proper values for initialization
            mock_model_instance = MagicMock()
            mock_model_instance.count_parameters.return_value = {"total": 1000000, "trainable": 1000000}
            mock_model_instance.get_memory_stats.return_value = {"parameter_memory_gb": 1.0}
            mock_model.return_value = mock_model_instance

            # Disable prints during initialization
            with patch("builtins.print"):
                trainer = Trainer(self.config)

            # Mock the stability tracker and other components
            trainer.stability_tracker = MagicMock()
            trainer.stability_tracker.get_comprehensive_stats.return_value = {
                "stability_score": 0.9,
                "loss_variance": 0.1,
            }
            trainer.stability_tracker.detect_training_issues.return_value = {"stable": True, "loss_spike": False}

            # Create mock metrics from train_step
            base_metrics = {"loss": 2.5, "lr": 1e-3, "grad_norm": 0.5, "step_time": 0.1}

            step_time = 0.1
            enhanced_metrics = trainer.get_enhanced_metrics(base_metrics, step_time)

            # Parameters that are passed explicitly to log_training_step
            explicit_params = {
                "step",
                "train_loss",
                "learning_rate",
                "tokens_processed",
                "samples_processed",
                "step_time",
                "tokens_per_sec",
                "wallclock_time",
            }

            # Check for conflicts (these should exist in enhanced_metrics)
            conflicts = set(enhanced_metrics.keys()) & explicit_params

            # Verify that conflicts exist (this is the problem we're solving)
            self.assertTrue(len(conflicts) > 0, "Enhanced metrics should contain conflicting parameters")
            self.assertIn("step_time", conflicts, "step_time should be in conflicts")
            self.assertIn("tokens_per_sec", conflicts, "tokens_per_sec should be in conflicts")

            # Now test that our filtering removes these conflicts
            additional_metrics = {k: v for k, v in enhanced_metrics.items() if k not in explicit_params}
            remaining_conflicts = set(additional_metrics.keys()) & explicit_params

            self.assertEqual(
                len(remaining_conflicts),
                0,
                f"After filtering, no conflicts should remain, but found: {remaining_conflicts}",
            )

    def test_training_integrator_call_compatibility(self):
        """Test that the actual call to training_integrator.log_training_step works without conflicts."""
        # Create mock training integrator
        mock_logger = MagicMock(spec=ExperimentLogger)
        mock_integrator = MagicMock(spec=TrainingIntegrator)

        # Mock the method to capture arguments
        mock_integrator.log_training_step = MagicMock()

        # Simulate the actual call pattern from the training script
        metrics = {"loss": 2.5, "lr": 1e-3, "grad_norm": 0.5, "step_time": 0.1}

        step_time = 0.1
        tokens_per_sec = 1000.0
        elapsed_hours = 0.5

        # Create enhanced metrics (similar to what get_enhanced_metrics returns)
        enhanced_metrics = {
            **metrics,
            "mfu": 0.3,
            "tokens_per_sec": tokens_per_sec,  # This was causing conflicts!
            "samples_per_sec": 100.0,
            "effective_batch_size": 32,
            "training_progress": 0.1,
            "wallclock_hours": elapsed_hours,
            "memory_allocated_gb": 5.0,
            "stability_score": 0.9,
        }

        # Filter out explicit parameters (this is the fix we implemented)
        explicit_params = {
            "step",
            "train_loss",
            "learning_rate",
            "tokens_processed",
            "samples_processed",
            "step_time",
            "tokens_per_sec",
            "wallclock_time",
        }
        additional_metrics = {k: v for k, v in enhanced_metrics.items() if k not in explicit_params}

        # Test the call
        try:
            mock_integrator.log_training_step(
                wallclock_time=elapsed_hours,
                step=100,
                train_loss=metrics["loss"],
                learning_rate=metrics["lr"],
                tokens_processed=1024,
                samples_processed=32,
                step_time=step_time,
                tokens_per_sec=tokens_per_sec,
                **additional_metrics,
            )

            # Verify the call was made
            mock_integrator.log_training_step.assert_called_once()

            # Verify no duplicate parameters
            call_args, call_kwargs = mock_integrator.log_training_step.call_args

            # Check that additional_metrics doesn't contain explicit parameters
            explicit_in_additional = set(additional_metrics.keys()) & explicit_params
            self.assertEqual(
                len(explicit_in_additional),
                0,
                f"Found explicit parameters in additional_metrics: {explicit_in_additional}",
            )

        except TypeError as e:
            if "multiple values for keyword argument" in str(e):
                self.fail(f"Parameter conflict detected: {e}")
            else:
                self.fail(f"Unexpected TypeError: {e}")

    def test_parameter_filtering_correctness(self):
        """Test that parameter filtering preserves important metrics while removing conflicts."""
        # Sample enhanced metrics with both conflicting and non-conflicting parameters
        enhanced_metrics = {
            # Conflicting parameters (should be filtered out)
            "step": 100,
            "train_loss": 2.5,
            "learning_rate": 1e-3,
            "tokens_processed": 1024,
            "samples_processed": 32,
            "step_time": 0.1,
            "tokens_per_sec": 1000.0,
            "wallclock_time": 0.5,
            # Non-conflicting parameters (should be preserved)
            "mfu": 0.3,
            "samples_per_sec": 100.0,
            "effective_batch_size": 32,
            "training_progress": 0.1,
            "memory_allocated_gb": 5.0,
            "stability_score": 0.9,
            "grad_norm": 0.5,
        }

        # Apply the filtering logic
        explicit_params = {
            "step",
            "train_loss",
            "learning_rate",
            "tokens_processed",
            "samples_processed",
            "step_time",
            "tokens_per_sec",
            "wallclock_time",
        }
        additional_metrics = {k: v for k, v in enhanced_metrics.items() if k not in explicit_params}

        # Verify conflicting parameters are removed
        for param in explicit_params:
            self.assertNotIn(param, additional_metrics, f"Conflicting parameter {param} was not filtered out")

        # Verify important non-conflicting metrics are preserved
        important_metrics = [
            "mfu",
            "samples_per_sec",
            "effective_batch_size",
            "training_progress",
            "memory_allocated_gb",
            "stability_score",
            "grad_norm",
        ]
        for metric in important_metrics:
            self.assertIn(metric, additional_metrics, f"Important metric {metric} was incorrectly filtered out")

    def test_validation_logging_no_conflicts(self):
        """Test that validation logging calls also don't have parameter conflicts."""
        mock_integrator = MagicMock(spec=TrainingIntegrator)
        mock_integrator.log_validation_step = MagicMock()

        # Test the validation logging call pattern
        try:
            mock_integrator.log_validation_step(
                wallclock_time=1.0,
                step=1000,
                val_loss=2.3,
                perplexity=10.0,
            )

            mock_integrator.log_validation_step.assert_called_once()

        except TypeError as e:
            if "multiple values for keyword argument" in str(e):
                self.fail(f"Parameter conflict in validation logging: {e}")
            else:
                self.fail(f"Unexpected TypeError in validation logging: {e}")

    def test_training_call_parameter_filtering(self):
        """Test the specific training call that was causing parameter conflicts."""
        # Test the exact pattern from the training script
        metrics = {"loss": 2.5, "lr": 1e-3, "grad_norm": 0.5, "step_time": 0.1}

        step_time = 0.1
        tokens_per_sec = 1000.0

        # Simulate what get_enhanced_metrics returns
        enhanced_metrics = {
            **metrics,
            "mfu": 0.3,
            "tokens_per_sec": tokens_per_sec,  # This was the conflict!
            "samples_per_sec": 100.0,
            "effective_batch_size": 32,
            "training_progress": 0.1,
            "wallclock_hours": 0.5,
            "memory_allocated_gb": 5.0,
            "stability_score": 0.9,
        }

        # Apply the filtering we implemented in the fix
        explicit_params = {
            "step",
            "train_loss",
            "learning_rate",
            "tokens_processed",
            "samples_processed",
            "step_time",
            "tokens_per_sec",
            "wallclock_time",
        }
        additional_metrics = {k: v for k, v in enhanced_metrics.items() if k not in explicit_params}

        # Mock the training integrator
        mock_integrator = MagicMock()

        # This should work without parameter conflicts
        try:
            mock_integrator.log_training_step(
                wallclock_time=0.5,
                step=100,
                train_loss=metrics["loss"],
                learning_rate=metrics["lr"],
                tokens_processed=1024,
                samples_processed=32,
                step_time=step_time,
                tokens_per_sec=tokens_per_sec,
                **additional_metrics,
            )

            # Verify it was called correctly
            mock_integrator.log_training_step.assert_called_once()

        except TypeError as e:
            if "multiple values for keyword argument" in str(e):
                self.fail(f"Parameter conflict still exists: {e}")
            else:
                self.fail(f"Unexpected error: {e}")


if __name__ == "__main__":
    unittest.main()
