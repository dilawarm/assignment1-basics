"""
Tests for experiment logging functionality.

These tests verify the basic functionality of the experiment logging system.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from cs336_basics.experiments.exp_logging import (
    ExperimentLogger,
    MemoryMonitor,
    PerformanceMonitor,
    TrainingIntegrator,
)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_memory_monitor():
    """Test memory monitoring functionality."""
    monitor = MemoryMonitor()

    monitor.reset()

    stats = monitor.get_stats()
    assert isinstance(stats, dict)

    if torch.cuda.is_available():
        assert "memory_allocated_gb" in stats
        assert "memory_utilization" in stats
        assert "memory_efficiency" in stats


def test_performance_monitor():
    """Test performance monitoring functionality."""
    monitor = PerformanceMonitor()

    for i in range(5):
        monitor.log_step(
            step_time=0.5 + i * 0.1,
            tokens_per_sec=1000 + i * 100,
            loss=2.0 - i * 0.1,
            batch_size=32,
            sequence_length=128,
        )

    stats = monitor.get_stats()
    assert isinstance(stats, dict)
    assert "avg_step_time" in stats
    assert "avg_tokens_per_sec" in stats
    assert "total_runtime_hours" in stats

    assert stats["avg_step_time"] > 0
    assert stats["avg_tokens_per_sec"] > 0


def test_experiment_logger_basic(temp_log_dir):
    """Test basic experiment logger functionality."""
    logger = ExperimentLogger(
        experiment_name="test_experiment",
        description="Test experiment",
        log_dir=temp_log_dir,
        use_wandb=False,
    )

    logger.log_hyperparameters(learning_rate=1e-4, batch_size=32, model_size="small")

    for i in range(10):
        logger.log_metrics(
            step=i,
            train_loss=2.0 - i * 0.1,
            learning_rate=1e-4 * (1 - i * 0.01),
        )

    logger.add_note("This is a test note")
    logger.add_note("Another test note")

    logger.mark_completed(success=True)

    experiment_dir = Path(temp_log_dir) / "test_experiment"
    assert experiment_dir.exists()
    assert (experiment_dir / "metadata.json").exists()
    assert (experiment_dir / "metrics.jsonl").exists()

    with open(experiment_dir / "metadata.json") as f:
        metadata = json.load(f)

    assert metadata["name"] == "test_experiment"
    assert metadata["status"] == "completed"
    assert "hyperparameters" in metadata
    assert len(metadata["notes"]) == 2


def test_training_integrator(temp_log_dir):
    """Test training integrator functionality."""
    logger = ExperimentLogger(
        experiment_name="training_test",
        log_dir=temp_log_dir,
        use_wandb=False,
    )

    integrator = TrainingIntegrator(logger, hardware_log_interval=5)

    integrator.start_epoch(0)

    for i in range(10):
        integrator.log_training_step(
            step=i,
            train_loss=2.0 - i * 0.1,
            learning_rate=1e-4,
            tokens_processed=1024,
            samples_processed=32,
            step_time=0.5,
            tokens_per_sec=2048,
        )

    integrator.log_validation_step(
        step=10,
        val_loss=1.5,
        perplexity=4.5,
    )

    assert integrator.best_val_loss == 1.5
    assert integrator.steps_since_improvement == 0

    integrator.log_validation_step(
        step=20,
        val_loss=1.6,
        perplexity=5.0,
    )

    assert integrator.best_val_loss == 1.5
    assert integrator.steps_since_improvement == 1

    logger.mark_completed()


def test_experiment_logger_file_operations(temp_log_dir):
    """Test that experiment logger correctly handles file operations."""
    logger = ExperimentLogger(
        experiment_name="file_test",
        log_dir=temp_log_dir,
        use_wandb=False,
    )

    logger.log_hyperparameters(param1=1, param2="test")
    logger.log_metrics(step=0, loss=1.0, accuracy=0.8)
    logger.add_note("Test note")

    metadata_file = Path(temp_log_dir) / "file_test" / "metadata.json"
    assert metadata_file.exists()

    metrics_file = Path(temp_log_dir) / "file_test" / "metrics.jsonl"
    assert metrics_file.exists()

    with open(metrics_file) as f:
        line = f.readline()
        metrics = json.loads(line)
        assert metrics["step"] == 0
        assert metrics["loss"] == 1.0
        assert metrics["accuracy"] == 0.8
        assert "timestamp" in metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_monitor_cuda():
    """Test memory monitor with CUDA."""
    monitor = MemoryMonitor()

    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y

    stats = monitor.get_stats()

    assert stats["memory_allocated_gb"] > 0
    assert stats["total_memory_gb"] > 0
    assert 0 <= stats["memory_utilization"] <= 1
    assert 0 <= stats["memory_efficiency"] <= 1

    del x, y, z
    torch.cuda.empty_cache()


def test_performance_monitor_trends():
    """Test performance monitor trend calculation."""
    monitor = PerformanceMonitor()

    losses = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
    for _, loss in enumerate(losses):
        monitor.log_step(step_time=0.5, tokens_per_sec=1000, loss=loss, batch_size=32, sequence_length=128)

    stats = monitor.get_stats()

    assert "loss_trend" in stats
    assert stats["loss_trend"] < 0


if __name__ == "__main__":
    pytest.main([__file__])
