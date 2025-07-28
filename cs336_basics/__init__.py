"""
A comprehensive implementation of Transformer language models from scratch.

This package provides all the components needed to train and use Transformer language models:
- Tokenization (BPE tokenizer)
- Neural network components (attention, layers, models)
- Training infrastructure (optimizers, schedulers, checkpointing)
- Text generation and decoding
- Experiment tracking and logging
"""

__version__ = "1.0.0"

# Data loading
from cs336_basics.data import get_batch

# Experiment tracking
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator

# Text generation
from cs336_basics.generation.decoding import (
    GreedyDecoding,
    MultinomialDecoding,
    TopPDecoding,
    compute_perplexity,
    decode_text,
    generate_completions,
)

# Loss functions
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.activations import SwiGLU, silu, softmax
from cs336_basics.nn.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding, scaled_dot_product_attention
from cs336_basics.nn.layers import Embedding, Linear, RMSNorm

# Core model components
from cs336_basics.nn.models import TransformerBlock, TransformerLM
from cs336_basics.tokenization.bpe import train_bpe

# Tokenization
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule

# Training components
from cs336_basics.training.optimizers import AdamW

__all__ = [
    # Core models
    "TransformerLM",
    "TransformerBlock",
    # Layers and components
    "Linear",
    "Embedding",
    "RMSNorm",
    "MultiHeadSelfAttention",
    "RotaryPositionalEmbedding",
    "scaled_dot_product_attention",
    "SwiGLU",
    "silu",
    "softmax",
    # Tokenization
    "Tokenizer",
    "train_bpe",
    # Training
    "AdamW",
    "cosine_learning_rate_schedule",
    "gradient_clipping",
    "save_checkpoint",
    "load_checkpoint",
    # Loss
    "cross_entropy",
    # Data
    "get_batch",
    # Generation
    "decode_text",
    "generate_completions",
    "compute_perplexity",
    "GreedyDecoding",
    "MultinomialDecoding",
    "TopPDecoding",
    # Experiments
    "ExperimentLogger",
    "TrainingIntegrator",
]
