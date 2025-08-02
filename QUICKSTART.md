# Quick Start Guide

## Installation

### Step 1: Install base dependencies
```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Step 2: Install H100 optimizations (optional but recommended)
```bash
# Easy method - run our installation script
./install_h100_optimizations.sh

# Or manually:
# 1. Install PyTorch first (if not already installed)
pip install torch

# 2. Install Flash Attention
pip install flash-attn --no-build-isolation

# 3. Install Transformer Engine (for FP8 on H100)
pip install transformer-engine
```

## Running the Training

### Quick test to verify everything works
```bash
# Test the model
python test_model.py

# Test data loading
python test_dataloader.py

# Run a quick demo
python demo_training.py
```

### Full training on H100
```bash
python train_h100.py
```

### Training with custom settings
```bash
python train_h100.py \
    --batch_size 16 \
    --learning_rate 4e-4 \
    --max_hours 1.5
```

### Running without H100 optimizations
If Flash Attention or Transformer Engine fail to install, the code will automatically fall back to standard implementations:

```bash
python train_h100.py --use_fp8 False --use_flash_attn False
```

## Expected Results

- **Training time**: 1.5 hours on H100
- **Training tokens**: ~4.9 billion tokens
- **Target validation loss**: < 3.0781
- **Expected validation loss**: 2.90-2.95

## Troubleshooting

### Flash Attention installation fails
```bash
# Install with limited parallel jobs
MAX_JOBS=1 pip install flash-attn --no-build-isolation
```

### Out of memory during training
- Reduce batch_size (default is 8)
- Increase gradient_accumulation_steps
- Ensure gradient_checkpointing is enabled (default: True)

### Warnings about missing optimizations
These warnings are normal if you don't have an H100 or haven't installed the optional packages:
- "Warning: Flash Attention not available. Using standard attention."
- "Warning: Transformer Engine not available. Using standard PyTorch."

The training will still work, just without the H100-specific optimizations.

## Data

The training uses preprocessed OpenWebText data located in:
- `data/encoded/owt_train_tokens.npy` (5.1GB)
- `data/encoded/owt_valid_tokens.npy` (127MB)

No additional data download is required.