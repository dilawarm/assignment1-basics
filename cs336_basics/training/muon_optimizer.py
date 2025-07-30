"""
Muon optimizer implementation for outlier-safe training.

Based on "Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models" (2025).
Implements Muon optimizer and MuonAdamHybrid for geometric optimization approach
that prevents activation outliers during training.
"""

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


def newton_schulz_iteration(G: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    Compute the matrix square root using Newton-Schulz iteration.

    For a positive definite matrix G, computes G^{-1/2} using:
    Y_{k+1} = Y_k * (3I - G * Y_k^2) / 2

    Args:
        G: Positive definite matrix
        num_iters: Number of Newton-Schulz iterations

    Returns:
        Approximation of G^{-1/2}
    """
    # Initialize Y_0 = I / ||G||_2
    spectral_norm = torch.norm(G, p=2)
    Y = torch.eye(G.size(0), device=G.device, dtype=G.dtype) / spectral_norm

    # Newton-Schulz iterations
    for _ in range(num_iters):
        # Y_{k+1} = Y_k * (3I - G * Y_k^2) / 2
        Y_squared = Y @ Y
        GY_squared = G @ Y_squared
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        Y = Y @ (3 * I - GY_squared) / 2

    return Y


class Muon(Optimizer):
    """
    Muon optimizer for geometric optimization approach.

    Muon uses a geometric approach to optimization, specifically addressing
    how changes in weight matrices affect neural network behavior.
    Eliminates privileged bases and prevents outlier formation.

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        ns_iters: Number of Newton-Schulz iterations (default: 5)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        ns_iters: int = 5,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not ns_iters >= 1:
            raise ValueError(f"Invalid ns_iters value: {ns_iters}")

        defaults = dict(
            lr=lr,
            ns_iters=ns_iters,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def get_dimension_scaling(self, shape: torch.Size) -> float:
        """
        Get dimension-based scaling factor for different parameter types.

        Args:
            shape: Parameter tensor shape

        Returns:
            Scaling factor
        """
        if len(shape) >= 2:
            # For matrices: scale by 1/sqrt(fan_in)
            fan_in = shape[1] if len(shape) == 2 else shape[1] * shape[2] * shape[3]
            return 1.0 / math.sqrt(fan_in)
        else:
            # For vectors: no scaling needed
            return 1.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

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
                if len(p.shape) >= 2:
                    # For matrices: apply geometric update
                    self._apply_matrix_update(p, momentum_buffer, lr, ns_iters)
                else:
                    # For vectors: apply standard update
                    scaling = self.get_dimension_scaling(p.shape)
                    p.add_(momentum_buffer, alpha=-lr * scaling)

        return loss

    def _apply_matrix_update(self, param: torch.Tensor, grad: torch.Tensor, lr: float, ns_iters: int):
        """
        Apply geometric matrix update using Muon approach.

        Args:
            param: Parameter matrix to update
            grad: Gradient matrix
            lr: Learning rate
            ns_iters: Number of Newton-Schulz iterations
        """
        # Get parameter matrix dimensions
        if len(param.shape) == 2:
            # Standard linear layer: (out_features, in_features)
            W = param
            G = grad
        else:
            # For higher-dimensional tensors, flatten to 2D
            original_shape = param.shape
            W = param.view(param.size(0), -1)
            G = grad.view(grad.size(0), -1)

        # Compute Gram matrix G^T @ G
        gram_matrix = G.T @ G

        # Add regularization for numerical stability
        reg_term = 1e-8 * torch.eye(gram_matrix.size(0), device=gram_matrix.device, dtype=gram_matrix.dtype)
        gram_matrix = gram_matrix + reg_term

        # Compute matrix square root using Newton-Schulz iteration
        try:
            gram_inv_sqrt = newton_schulz_iteration(gram_matrix, ns_iters)

            # Apply geometric update: W = W - lr * G * (G^T G)^{-1/2}
            update = G @ gram_inv_sqrt
            scaling = self.get_dimension_scaling(W.shape)
            W.add_(update, alpha=-lr * scaling)

        except Exception:
            # Fallback to standard update if geometric update fails
            scaling = self.get_dimension_scaling(W.shape)
            W.add_(G, alpha=-lr * scaling)

        # Reshape back to original shape if needed
        if len(param.shape) > 2:
            param.data = W.view(original_shape)


class MuonAdamHybrid(Optimizer):
    """
    Hybrid optimizer combining Muon geometric updates with Adam's adaptive learning rates.

    Uses Muon for matrix parameters (weights) and Adam for vector parameters (biases, norms).
    This provides outlier-safe training while maintaining adaptive optimization benefits.

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for Adam momentum (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        muon_momentum: Momentum factor for Muon updates (default: 0.95)
        ns_iters: Number of Newton-Schulz iterations (default: 5)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        muon_momentum: float = 0.95,
        ns_iters: int = 5,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= muon_momentum <= 1.0:
            raise ValueError(f"Invalid muon_momentum value: {muon_momentum}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            muon_momentum=muon_momentum,
            ns_iters=ns_iters,
        )

        super().__init__(params, defaults)

    def get_dimension_scaling(self, shape: torch.Size) -> float:
        """Get dimension-based scaling factor."""
        if len(shape) >= 2:
            fan_in = shape[1] if len(shape) == 2 else shape[1] * shape[2] * shape[3]
            return 1.0 / math.sqrt(fan_in)
        else:
            return 1.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Optional[float]: Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            muon_momentum = group["muon_momentum"]
            ns_iters = group["ns_iters"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if len(p.shape) >= 2:
                        # For matrices: use Muon momentum
                        state["muon_momentum_buffer"] = torch.zeros_like(grad)
                    else:
                        # For vectors: use Adam moments
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                if len(p.shape) >= 2:
                    # Matrix parameters: use Muon geometric update
                    self._apply_muon_update(p, grad, state, lr, muon_momentum, ns_iters)
                else:
                    # Vector parameters: use Adam update
                    self._apply_adam_update(p, grad, state, lr, beta1, beta2, eps)

        return loss

    def _apply_muon_update(
        self, param: torch.Tensor, grad: torch.Tensor, state: Dict[str, Any], lr: float, momentum: float, ns_iters: int
    ):
        """Apply Muon geometric update for matrix parameters."""
        # Update momentum buffer
        momentum_buffer = state["muon_momentum_buffer"]
        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

        # Get parameter matrix dimensions
        if len(param.shape) == 2:
            W = param
            G = momentum_buffer
        else:
            # For higher-dimensional tensors, flatten to 2D
            original_shape = param.shape
            W = param.view(param.size(0), -1)
            G = momentum_buffer.view(momentum_buffer.size(0), -1)

        # Compute Gram matrix G^T @ G
        gram_matrix = G.T @ G

        # Add regularization for numerical stability
        reg_term = 1e-8 * torch.eye(gram_matrix.size(0), device=gram_matrix.device, dtype=gram_matrix.dtype)
        gram_matrix = gram_matrix + reg_term

        # Apply geometric update
        try:
            gram_inv_sqrt = newton_schulz_iteration(gram_matrix, ns_iters)
            update = G @ gram_inv_sqrt
            scaling = self.get_dimension_scaling(W.shape)
            W.add_(update, alpha=-lr * scaling)
        except Exception:
            # Fallback to standard update
            scaling = self.get_dimension_scaling(W.shape)
            W.add_(G, alpha=-lr * scaling)

        # Reshape back if needed
        if len(param.shape) > 2:
            param.data = W.view(original_shape)

    def _apply_adam_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
    ):
        """Apply Adam update for vector parameters."""
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step = state["step"]

        # Exponential moving average of gradient values
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Exponential moving average of squared gradient values
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Compute update
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1

        # Apply update
        param.addcdiv_(exp_avg, denom, value=-step_size)


def create_muon_optimizer(
    parameters: Iterable[torch.nn.Parameter], optimizer_type: str = "hybrid", **kwargs
) -> Optimizer:
    """
    Create a Muon-based optimizer.

    Args:
        parameters: Model parameters
        optimizer_type: "muon" or "hybrid"
        **kwargs: Optimizer-specific arguments

    Returns:
        Muon optimizer instance
    """
    if optimizer_type.lower() == "muon":
        return Muon(parameters, **kwargs)
    elif optimizer_type.lower() == "hybrid":
        return MuonAdamHybrid(parameters, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Aliases for backward compatibility
MuonOptimizer = Muon
HybridOptimizer = MuonAdamHybrid
