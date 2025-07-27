"""
Activation functions and modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.nn.layers import Linear


class CustomFFN(nn.Module):
    """
    Custom Feed-Forward Network.

    Uses the activation pattern: w2(max(w1(x), 0)^2)
    This outperformed SwiGLU and other GLU variants in the competition.

    Key improvements:
    - Squaring after ReLU provides better non-linearity
    - More stable training than SwiGLU
    - Better performance on OpenWebText
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the CustomFFN.

        Args:
            d_model: Input/output dimension (hidden dimension of the model)
            d_ff: Inner dimension of the feed-forward network. If None, defaults to
                  4 * d_model rounded to the nearest multiple of 64
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            d_ff = 4 * d_model
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """
        Apply CustomFFN transformation: w2(max(w1(x), 0)^2).

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of the same shape as input
        """
        hidden = self.w1(x)
        activated = torch.relu(hidden)
        activated = activated.clamp(max=20.0)
        squared = activated * activated
        squared = squared.clamp(max=400.0)
        output = self.w2(squared)
        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class SwiGLU(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.

    SwiGLU combines the SiLu (Swish) activation functions with a Gated Linear Unit (GLU).
    This is used in modern language models like LLaMA and provides better performance
    than traditional ReLU-based feed-forward networks.

    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    Formula: SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) ⊙ W3 x)
    where SiLU(x) = x · σ(x) = x / (1 + e^(-x))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the SwiGLU feed-forward network.

        Args:
            d_model: Input/output dimension (hidden dimension of the model)
            d_ff: Inner dimension of the feed-forward network. If None, defaults to
                  approximately 8/3 * d_model rounded to the nearest multiple of 64
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """
        Apply SwiGLU transformation to input.

        Args:
            x: Input tensor of shape (..., d_model) where ... can be any number
               of batch dimensions (e.g., batch_size, sequence_length, d_model)

        Returns:
            Output tensor of the same shape as input with SwiGLU applied
        """
        gate = self.w1(x)
        value = self.w3(x)

        silu_gate = silu(gate)
        gated_value = silu_gate * value

        output = self.w2(gated_value)
        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


FFN = CustomFFN


def silu(input: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    """
    Apply the SiLU (Sigmoid Linear Unit) activation function element-wise.

    SiLU, also known as Swish, is a smooth, non-monotonic activation function
    that has been shown to work well in deep neural networks.

    Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        input: Input tensor with arbitrary shape

    Returns:
        Output tensor with same shape as input, with SiLU applied element-wise
    """
    return input * torch.sigmoid(input)


def softmax(input: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """
    Apply softmax function to the specified dimension of the input tensor.

    This implementation uses the numerically stable version that subtracts the maximum
    value from the input before applying exponential to prevent overflow.

    Formula: softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        input: Input tensor with arbitrary shape
        dim: Dimension along which to apply softmax. The resulting tensor
             will have the same shape as input, but the specified dimension
             will be normalized to sum to 1.

    Returns:
        Output tensor with same shape as input, with softmax applied along
        the specified dimension
    """
    max_vals = torch.max(input, dim=dim, keepdim=True)[0]
    input_stable = input - max_vals

    exp_vals = torch.exp(input_stable)
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)

    softmax_output = exp_vals / sum_exp
    return softmax_output
