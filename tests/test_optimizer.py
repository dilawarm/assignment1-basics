"""
Tests for the Muon optimizer and mixed optimizers.
"""

import math

import torch
import torch.nn as nn

from cs336_basics.training.muon_optimizer import Muon, MuonAdamHybrid
from cs336_basics.training.optimizers import Adam


class SimpleLinear(nn.Module):
    """Simple linear model for testing."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class SimpleTransformer(nn.Module):
    """Simple transformer-like model for testing parameter separation."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embeddings(x)
        residual = x
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return self.lm_head(x)


class TestMuon:
    """Test Muon optimizer."""

    def test_muon_initialization(self):
        """Test Muon optimizer initialization."""
        model = SimpleLinear(10, 20, 5)
        optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.9)

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["momentum"] == 0.9
