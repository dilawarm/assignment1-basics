"""
Muon Optimizer - Advanced optimizer for neural network training.

Based on the reference implementation from:
https://medium.com/@kyeg/building-the-muon-optimizer-in-pytorch-a-geometric-approach-to-neural-network-optimization-17f4601be548

Provides significant speedups for transformer training with proper geometric optimization.
"""

from typing import Callable, Iterator, Optional

import torch
from torch.optim.optimizer import Optimizer


def newton_schulz_orthogonalize(X: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    Apply Newton-Schulz iterations to approximate orthogonalization.

    This function applies the polynomial f(X) = (3X - X^3)/2 repeatedly to a normalized matrix,
    which gradually forces all singular values to 1 while preserving singular vectors.

    For non-square matrices, we work with the Gram matrix to ensure proper dimensions.

    Args:
        X (torch.Tensor): Input matrix to orthogonalize
        num_iters (int): Number of Newton-Schulz iterations

    Returns:
        torch.Tensor: Orthogonalized matrix
    """
    # Handle edge cases
    if X.numel() == 0:
        return X

    # First, normalize the input matrix to get spectral norm close to 1
    # We use Frobenius norm as a simple approximation for initialization
    norm = torch.norm(X, p="fro")
    if norm < 1e-8:
        return X  # Avoid division by zero

    X = X / norm

    # For square matrices, apply Newton-Schulz directly
    if X.shape[0] == X.shape[1]:
        # Apply Newton-Schulz iterations: f(X) = (3X - X^3)/2
        for _ in range(num_iters):
            X = (3 * X - torch.matmul(torch.matmul(X, X), X)) / 2
    else:
        # For non-square matrices, work with Gram matrix
        # This is a simplified approach that preserves the directional information
        if X.shape[0] > X.shape[1]:
            # Tall matrix: work with X.T @ X
            gram = torch.matmul(X.T, X)
            # Normalize gram matrix
            gram_norm = torch.norm(gram, p="fro")
            if gram_norm > 1e-8:
                gram = gram / gram_norm

                # Apply Newton-Schulz to gram matrix
                for _ in range(num_iters):
                    gram = (3 * gram - torch.matmul(torch.matmul(gram, gram), gram)) / 2

                # Reconstruct X using the orthogonalized gram matrix
                # This is an approximation that maintains the shape
                X = torch.matmul(X, gram)
        else:
            # Wide matrix: work with X @ X.T
            gram = torch.matmul(X, X.T)
            # Normalize gram matrix
            gram_norm = torch.norm(gram, p="fro")
            if gram_norm > 1e-8:
                gram = gram / gram_norm

                # Apply Newton-Schulz to gram matrix
                for _ in range(num_iters):
                    gram = (3 * gram - torch.matmul(torch.matmul(gram, gram), gram)) / 2

                # Reconstruct X using the orthogonalized gram matrix
                # This is an approximation that maintains the shape
                X = torch.matmul(gram, X)

    return X


class Muon(Optimizer):
    """
    Implements the Muon optimization algorithm for linear layers.

    Muon uses a geometric approach to optimization, specifically addressing
    how changes in weight matrices affect neural network behavior.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        ns_iters (int, optional): number of Newton-Schulz iterations (default: 5)
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        ns_iters: int = 5,
        momentum: float = 0.9,
        weight_decay: float = 0,
    ):
        defaults = dict(lr=lr, ns_iters=ns_iters, momentum=momentum, weight_decay=weight_decay)
        super(Muon, self).__init__(params, defaults)

    def get_dimension_scaling(self, shape: torch.Size) -> float:
        """
        Get the dimension scaling factor for RMS-to-RMS operator norm.

        Args:
            shape: Parameter shape

        Returns:
            Scaling factor sqrt(d_x * d_y)
        """
        if len(shape) >= 2:
            # For matrices: sqrt(input_dim * output_dim)
            d_in, d_out = shape[-1], shape[0]
            return (d_in * d_out) ** 0.5
        else:
            # For vectors: no scaling needed
            return 1.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss

        Returns:
            Optional[float]: Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            ns_iters = group["ns_iters"]
            momentum_factor = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize momentum buffer if needed
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    state["step"] = 0

                # Increment step count
                state["step"] += 1

                # Update momentum buffer
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(momentum_factor).add_(grad, alpha=1 - momentum_factor)

                # Apply Muon update based on parameter type
                if len(p.shape) >= 2:  # For matrices and higher-dimensional tensors
                    # Reshape to matrix for higher dimensions
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        p_flat = p.reshape(p.shape[0], -1)
                        momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        p_flat = p
                        momentum_flat = momentum_buffer

                    # Apply Newton-Schulz orthogonalization to momentum buffer
                    ortho_momentum = newton_schulz_orthogonalize(momentum_flat, ns_iters)

                    # Get dimension scaling: sqrt(d_x * d_y)
                    dim_scaling = self.get_dimension_scaling(original_shape)

                    # Calculate Frobenius norm of momentum buffer
                    buffer_norm = torch.norm(momentum_flat, p="fro")

                    if buffer_norm > 1e-8:
                        # Apply Muon update rule: W ← W - α · (sqrt(d_x * d_y) / |G|_F) · NS(G)
                        scaling = dim_scaling / buffer_norm
                        update = ortho_momentum * scaling

                        # Reshape back if needed
                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)

                        # Apply the update
                        p.add_(update, alpha=-lr)

                else:
                    # For non-matrix parameters (embeddings, biases), use standard momentum update
                    p.add_(momentum_buffer, alpha=-lr)

        return loss


class MuonAdamHybrid(Optimizer):
    """
    Hybrid optimizer that uses corrected Muon for linear layers and Adam for other parameters.

    This provides the best of both worlds - Muon's efficiency for linear transformations
    and Adam's stability for embeddings and other parameter types.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        newton_schulz_steps: int = 5,
    ):
        self.muon_params = []
        self.adam_params = []

        # Convert params to a list if it's an iterator
        if hasattr(params, "__iter__") and not isinstance(params, (list, tuple)):
            params = list(params)

        # Handle both individual parameters and parameter groups
        if len(params) > 0 and isinstance(params[0], dict):
            # Parameter groups format: [{"params": [p1, p2, ...]}, ...]
            for param_group in params:
                param_list = param_group["params"]
                for p in param_list:
                    if len(p.shape) >= 2 and min(p.shape) >= 16:
                        self.muon_params.append(p)
                    else:
                        self.adam_params.append(p)
        else:
            # Individual parameters format: [p1, p2, p3, ...]
            for p in params:
                if len(p.shape) >= 2 and min(p.shape) >= 16:
                    self.muon_params.append(p)
                else:
                    self.adam_params.append(p)

        if len(self.muon_params) == 0 and len(self.adam_params) == 0:
            raise ValueError("No parameters to optimize! Check if model parameters are being passed correctly.")

        # Use corrected Muon implementation
        self.muon = Muon(
            self.muon_params,
            lr=lr,
            momentum=muon_momentum,
            ns_iters=newton_schulz_steps,
        )

        self.adam = torch.optim.AdamW(
            self.adam_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        self.param_groups = self.muon.param_groups + self.adam.param_groups

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(self.param_groups, defaults)

    def step(self, closure=None):
        """Perform optimization step with both optimizers."""
        loss = None
        if closure is not None:
            loss = closure()

        if self.muon_params:
            self.muon.step()
        if self.adam_params:
            self.adam.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for both optimizers."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return combined state dict."""
        return {
            "muon": self.muon.state_dict(),
            "adam": self.adam.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
