"""
Enhanced optimizers for improved training performance.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.optim import Optimizer


class EnhancedAdam(Optimizer):
    """
    Enhanced Adam optimizer with improvements for transformer training.

    This implements several improvements:
    - Momentum scaling based on parameter dimension
    - Adaptive epsilon
    - Better bias correction
    - Support for parameter groups with different hyperparameters
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_scale: bool = True,
        adaptive_eps: bool = True,
    ) -> None:
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_scale=momentum_scale,
            adaptive_eps=adaptive_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Cast gradient to float32 for stability
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                    # Calculate effective dimensionality for momentum scaling
                    if group["momentum_scale"] and p.ndim >= 2:
                        # Use the geometric mean of dimensions as effective dimension
                        dims = p.shape
                        state["eff_dim"] = math.pow(math.prod(dims), 1.0 / len(dims))
                    else:
                        state["eff_dim"] = 1.0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                # Momentum scaling based on dimension
                if group["momentum_scale"]:
                    # Scale beta1 based on effective dimension
                    # Higher dimensional parameters get higher momentum
                    dim_scale = min(state["eff_dim"] / 512.0, 2.0)  # Cap at 2x
                    effective_beta1 = 1 - (1 - beta1) / dim_scale
                else:
                    effective_beta1 = beta1

                # Bias correction with numerical stability
                # Cap step to prevent overflow in bias correction
                # When beta^step becomes very small, it's effectively 0 anyway
                max_step = 1000  # After 1000 steps, bias correction is minimal
                capped_step = min(step, max_step)

                bias_correction1 = 1 - effective_beta1**capped_step
                bias_correction2 = 1 - beta2**capped_step

                # Update biased first moment estimate
                exp_avg.mul_(effective_beta1).add_(grad, alpha=1 - effective_beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the denominator
                if group["adaptive_eps"]:
                    # Scale epsilon based on gradient magnitude
                    grad_scale = exp_avg_sq.sqrt().mean().item()
                    adaptive_epsilon = group["eps"] * max(1.0, grad_scale)
                else:
                    adaptive_epsilon = group["eps"]

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(adaptive_epsilon)

                # Step size
                step_size = group["lr"] / bias_correction1

                # Update parameters
                if group["weight_decay"] != 0:
                    # Decoupled weight decay (AdamW style)
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Apply update
                if p.dtype in {torch.float16, torch.bfloat16}:
                    # Cast back to original dtype
                    p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class LionOptimizer(Optimizer):
    """
    Lion optimizer - a memory-efficient optimizer discovered through evolution.

    Lion achieves similar or better performance than Adam with less memory usage.
    Paper: "Symbolic Discovery of Optimization Algorithms"
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Cast gradient to float32 for stability
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Compute update direction: sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()

                # Update parameters
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.add_(update.to(p.dtype), alpha=-group["lr"])
                else:
                    p.add_(update, alpha=-group["lr"])

                # Update momentum
                exp_avg.mul_(beta2).add(grad, alpha=1 - beta2)

        return loss


class MuonOptimizer(Optimizer):
    """
    Muon optimizer - MomentUm Orthogonalized by Newton-schulz.

    Muon is specifically designed for 2D parameters (linear layers) in neural networks.
    It orthogonalizes the momentum using Newton-Schulz iterations to achieve better
    convergence properties.

    Paper: "Muon: An optimizer for hidden layers in neural networks"

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 0.02)
        momentum: momentum factor (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        ns_steps: number of Newton-Schulz iterations (default: 5)
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def newton_schulz_orthogonalize(self, G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations to orthogonalize a matrix.

        This uses optimized coefficients (3.4445, -4.7750, 2.0315) for faster convergence.
        """
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)

        # Use bfloat16 for efficiency
        X = G.to(torch.bfloat16)
        X = X / (X.norm() + eps)

        # Transpose if necessary for efficiency
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True

        # Newton-Schulz iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        # Transpose back if needed
        if transposed:
            X = X.T

        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Only apply Muon to 2D parameters
                if p.ndim != 2:
                    continue

                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state["momentum_buffer"]
                momentum = group["momentum"]

                # Add weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Update momentum buffer
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Apply Newton-Schulz orthogonalization
                if group["nesterov"]:
                    # Nesterov look-ahead
                    G = grad.add(buf, alpha=momentum)
                else:
                    G = buf

                # Orthogonalize
                G_ortho = self.newton_schulz_orthogonalize(G, steps=group["ns_steps"])

                # Scale by sqrt(d_in * d_out) / ||G||_F as per Muon formula
                d_in, d_out = p.shape
                scale = (d_in * d_out) ** 0.5 / (G.norm() + 1e-7)

                # Apply update
                p.add_(G_ortho, alpha=-group["lr"] * scale)

        return loss
