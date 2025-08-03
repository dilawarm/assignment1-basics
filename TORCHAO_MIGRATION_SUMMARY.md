# TorchAO Migration Summary

This document summarizes the changes made to ensure TorchAO is used consistently across the repository for FP8 training.

## Key Changes Made

### 1. Updated `train_h100.py`
- Added `--fp8_backend` flag with options: "native" or "torchao"
- Default backend is "native" for backward compatibility
- Added automatic TorchAO model conversion when `--fp8_backend torchao` is specified
- Updated error messages to recommend TorchAO instead of float8_experimental

### 2. Documentation Updates

#### README_TRAINING.md
- Moved TorchAO FP8 to the top as the "BEST Option"
- Clearly marked native FP8 as having "Limited Support"
- Reorganized quick start section to prioritize TorchAO

#### FP8_OPTIONS_GUIDE.md
- Added "Quick Answer: Use TorchAO" at the very top
- Updated to show TorchAO as Option #1 (recommended)
- Marked float8_experimental as "Legacy - DO NOT USE"
- Updated performance comparison table to highlight TorchAO
- Updated recommendation decision tree to prioritize TorchAO

#### NATIVE_FP8_GUIDE.md
- Added prominent warning at the top recommending TorchAO
- Clarified that native FP8 is kept mainly for educational purposes

#### TROUBLESHOOTING.md
- Updated all FP8 troubleshooting sections to recommend TorchAO first
- Changed error resolution steps to suggest `--fp8_backend torchao`

### 3. Code Documentation

Added warnings to all native FP8 implementation files:
- `cs336_basics/model/fp8_linear.py`
- `cs336_basics/model/attention_native_fp8.py`
- `cs336_basics/model/transformer_native_fp8.py`
- `test_fp8.py`

Each file now includes a note that TorchAO is recommended for production use.

### 4. Dependencies
- Confirmed `torchao>=0.12.0` is already in `pyproject.toml`

## Why TorchAO?

Based on the official PyTorch sources:
- float8_experimental has been **archived** and moved to pytorch/ao (TorchAO)
- TorchAO is PyTorch's official architecture optimization library
- It provides stable FP8 training with ~1.5x speedup over FP16
- Works seamlessly with torch.compile and FSDP
- Avoids the cuBLAS matrix layout issues of native PyTorch FP8

## Usage

The recommended way to use FP8 training is now:

```bash
# Install TorchAO
pip install torchao

# Run with TorchAO FP8
python train_h100.py --use_fp8 --fp8_backend torchao
```

## Backward Compatibility

The native FP8 implementation is retained for:
1. Educational purposes
2. Backward compatibility
3. Fallback option if TorchAO is not available

However, all documentation now clearly recommends TorchAO as the preferred solution.