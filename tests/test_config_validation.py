#!/usr/bin/env python3
"""
Lightweight configuration validation script that can run without heavy PyTorch dependencies.
This script validates that configuration files are properly formatted and compatible.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union


def get_expected_fields() -> Set[str]:
    """Get the expected fields for TrainArgs based on the dataclass definition."""
    return {
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
        "use_mixed_precision",
        "use_efficient_attention",
        "use_fused_kernels",
        "experiment_name",
        "experiment_description",
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "log_dir",
    }


def validate_field_types_and_values(config: Dict[str, Any]) -> List[str]:
    """Validate field types and values."""
    errors = []

    # Define validation rules: (field_name, expected_type, validator_function)
    validations = [
        ("vocab_size", int, lambda x: x > 0, "must be positive integer"),
        ("context_length", int, lambda x: x > 0, "must be positive integer"),
        ("num_layers", int, lambda x: x > 0, "must be positive integer"),
        ("d_model", int, lambda x: x > 0, "must be positive integer"),
        ("num_heads", int, lambda x: x > 0, "must be positive integer"),
        ("d_ff", int, lambda x: x > 0, "must be positive integer"),
        ("rope_theta", (int, float), lambda x: x > 0, "must be positive number"),
        ("window_size", (int, type(None)), lambda x: x is None or x > 0, "must be positive integer or null"),
        ("weight_decay", (int, float), lambda x: x >= 0, "must be non-negative number"),
        (
            "betas",
            list,
            lambda x: len(x) == 2 and all(0 <= b <= 1 for b in x),
            "must be list of 2 values between 0 and 1",
        ),
        ("muon_momentum", (int, float), lambda x: 0 <= x <= 1, "must be between 0 and 1"),
        ("max_learning_rate", (int, float), lambda x: x > 0, "must be positive number"),
        ("min_learning_rate", (int, float), lambda x: x > 0, "must be positive number"),
        ("warmup_iters", int, lambda x: x >= 0, "must be non-negative integer"),
        ("cosine_cycle_iters", int, lambda x: x > 0, "must be positive integer"),
        ("steps", int, lambda x: x > 0, "must be positive integer"),
        ("batch_size", int, lambda x: x > 0, "must be positive integer"),
        ("gradient_accumulation_steps", int, lambda x: x > 0, "must be positive integer"),
        ("gradient_clipping", (int, float), lambda x: x > 0, "must be positive number"),
        ("validation_step_interval", int, lambda x: x > 0, "must be positive integer"),
        ("checkpoint_step_interval", int, lambda x: x > 0, "must be positive integer"),
        ("use_qk_norm", bool, lambda x: True, "must be boolean"),
        ("use_flex_attention", bool, lambda x: True, "must be boolean"),
        ("use_swiglu", bool, lambda x: True, "must be boolean"),
        ("tie_embeddings", bool, lambda x: True, "must be boolean"),
        ("compile_model", bool, lambda x: True, "must be boolean"),
        ("use_mixed_precision", bool, lambda x: True, "must be boolean"),
        ("use_efficient_attention", bool, lambda x: True, "must be boolean"),
        ("use_fused_kernels", bool, lambda x: True, "must be boolean"),
        ("use_wandb", bool, lambda x: True, "must be boolean"),
        ("device", str, lambda x: x in ["cuda", "cpu"], "must be 'cuda' or 'cpu'"),
    ]

    for field, expected_type, validator, error_msg in validations:
        if field in config:
            value = config[field]
            if not isinstance(value, expected_type):
                errors.append(f"{field}: expected {expected_type}, got {type(value).__name__}")
            elif not validator(value):
                errors.append(f"{field}: {error_msg}, got {value}")

    # Special validations
    if "d_model" in config and "num_heads" in config:
        if config["d_model"] % config["num_heads"] != 0:
            errors.append(f"d_model ({config['d_model']}) must be divisible by num_heads ({config['num_heads']})")

    if "min_learning_rate" in config and "max_learning_rate" in config:
        if config["min_learning_rate"] >= config["max_learning_rate"]:
            errors.append(
                f"min_learning_rate ({config['min_learning_rate']}) must be less than max_learning_rate ({config['max_learning_rate']})"
            )

    return errors


def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """Validate a single configuration file."""
    errors = []

    # Check file exists
    if not Path(config_path).exists():
        return False, [f"File not found: {config_path}"]

    # Load and parse JSON
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Check field coverage
    expected_fields = get_expected_fields()
    config_fields = set(config.keys())

    missing_fields = expected_fields - config_fields
    extra_fields = config_fields - expected_fields

    if extra_fields:
        errors.append(f"Unexpected fields (will be ignored): {sorted(extra_fields)}")

    # Validate field types and values
    validation_errors = validate_field_types_and_values(config)
    errors.extend(validation_errors)

    return len(validation_errors) == 0, errors


def compare_configs(config_paths: List[str]) -> None:
    """Compare multiple configuration files."""
    configs = {}

    for config_path in config_paths:
        if Path(config_path).exists():
            with open(config_path) as f:
                configs[Path(config_path).stem] = json.load(f)

    if len(configs) >= 2:
        print("\n=== CONFIG COMPARISON ===")

        # Key fields to compare
        key_fields = [
            "steps",
            "batch_size",
            "max_learning_rate",
            "warmup_iters",
            "cosine_cycle_iters",
            "gradient_accumulation_steps",
        ]

        config_names = list(configs.keys())
        print(f"Comparing: {' vs '.join(config_names)}")

        for field in key_fields:
            values = []
            for name in config_names:
                val = configs[name].get(field, "N/A")
                values.append(f"{name}={val}")
            print(f"  {field}: {', '.join(values)}")


def main():
    """Main validation function."""
    print("=== CONFIGURATION VALIDATION ===")

    # Configuration files to validate
    config_files = [
        "cs336_basics/scripts/configs/optimized_h100_config.json",
        "cs336_basics/scripts/configs/time_optimized_h100_config.json",
    ]

    all_valid = True

    for config_file in config_files:
        print(f"\n--- Validating {Path(config_file).name} ---")

        is_valid, errors = validate_config_file(config_file)

        if is_valid:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")
            all_valid = False

        # Show warnings for non-critical issues
        if errors and is_valid:
            print("⚠️  Warnings:")
            for error in errors:
                print(f"  - {error}")

    # Compare configurations
    compare_configs(config_files)

    print(f"\n=== SUMMARY ===")
    if all_valid:
        print("✅ All configuration files are valid!")
        return 0
    else:
        print("❌ Some configuration files have critical errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
