# Model Size Optimization for 350M Parameters

## Current Issue

Your current configuration creates a **454M parameter model** instead of the target 350M:
```bash
--dim 1024 --n_layers 24 --n_heads 16 --intermediate_size 4096
```

## Recommended Configurations for ~350M Parameters

### Option 1: Reduce Hidden Dimension (Recommended)
```bash
--dim 896 --n_layers 24 --n_heads 14 --head_dim 64 --intermediate_size 3584
```
- Parameters: ~349M
- Maintains 24 layers for good expressiveness
- Slightly smaller hidden dimension

### Option 2: Reduce Layers
```bash
--dim 1024 --n_layers 18 --n_heads 16 --head_dim 64 --intermediate_size 4096
```
- Parameters: ~345M
- Keeps original dimensions but fewer layers
- May train faster but potentially less expressive

### Option 3: Balanced Reduction
```bash
--dim 960 --n_layers 20 --n_heads 15 --head_dim 64 --intermediate_size 3840
```
- Parameters: ~350M
- Balanced trade-off between depth and width

## Parameter Count Calculation

For a transformer with tied embeddings:
```
params ≈ vocab_size × dim + n_layers × (12 × dim² + 4 × dim × intermediate_size)
```

Where:
- Embeddings: 50,257 × dim
- Per layer: 
  - Attention (QKV + O): 4 × dim²
  - FFN: 2 × dim × intermediate_size
  - LayerNorms: 2 × dim (negligible)

## Performance Impact

- **Smaller models train faster**: Less compute per forward/backward pass
- **Memory efficiency**: More room for larger batch sizes
- **Target performance**: Should still achieve <3.0781 validation loss

## Quick Test

To verify the parameter count before training:
```python
from cs336_basics.model import TransformerLM

model = TransformerLM(
    vocab_size=50257,
    max_seq_len=1024,
    dim=896,  # Reduced from 1024
    n_layers=24,
    n_heads=14,  # Reduced from 16
    head_dim=64,
    intermediate_size=3584,  # Reduced from 4096
)

# This will print: Total parameters: 349,XXX,XXX (~349M)
```

## Training Command

For the optimized 350M model:
```bash
uv run python train_h100.py \
    --dim 896 \
    --n_heads 14 \
    --intermediate_size 3584 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --no_gradient_checkpointing
```

This configuration should:
- Train faster (fewer parameters)
- Use less memory
- Still achieve the target validation loss
- Potentially reach higher tokens/sec