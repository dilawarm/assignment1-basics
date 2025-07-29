"""
Muon Optimizer - Advanced optimizer for neural network training.

Based on the latest research from Keller Jordan and Jeremy Bernstein.
Provides significant speedups for transformer training.
"""

import torch
from torch.optim.optimizer import Optimizer


@torch.compile
def zeroth_power_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration for computing the zeroth power (orthogonalization).

    Args:
        G: Input matrix to orthogonalize
        steps: Number of Newton-Schulz iterations
        eps: Small epsilon for numerical stability

    Returns:
        Orthogonalized matrix
    """
    assert len(G.shape) == 2, "Input must be a 2D matrix"

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)

    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer - A modern optimizer for neural network training.

    Combines momentum with orthogonalization for better training dynamics.
    Particularly effective for transformer models.

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 3e-4)
        momentum: Momentum factor (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        newton_schulz_steps: Number of Newton-Schulz iterations (default: 5)
        orthogonalize_every: Apply orthogonalization every N steps (default: 1)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        newton_schulz_steps: int = 5,
        orthogonalize_every: int = 1,
        eps: float = 1e-7,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0 < newton_schulz_steps:
            raise ValueError(f"Invalid newton_schulz_steps: {newton_schulz_steps}")
        if not 0 < orthogonalize_every:
            raise ValueError(f"Invalid orthogonalize_every: {orthogonalize_every}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            newton_schulz_steps=newton_schulz_steps,
            orthogonalize_every=orthogonalize_every,
            eps=eps,
        )
        super().__init__(params, defaults)

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

                grad = p.grad
                if grad.dtype in (torch.float16, torch.bfloat16):
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

                buf = state["momentum_buffer"]
                momentum = group["momentum"]

                state["step"] += 1

                if len(grad.shape) >= 2:
                    if state["step"] % group["orthogonalize_every"] == 0:
                        original_shape = grad.shape
                        if len(grad.shape) > 2:
                            grad = grad.view(grad.shape[0], -1)

                        grad = zeroth_power_via_newtonschulz5(
                            grad, steps=group["newton_schulz_steps"], eps=group["eps"]
                        )

                        if len(original_shape) > 2:
                            grad = grad.view(original_shape)

                buf.mul_(momentum).add_(grad)

                if group["nesterov"]:
                    update = grad + momentum * buf
                else:
                    update = buf

                p.add_(update, alpha=-group["lr"])

        return loss


class MuonAdamHybrid(Optimizer):
    """
    Hybrid optimizer that uses Muon for linear layers and Adam for other parameters.

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

        for param_group in params:
            if isinstance(param_group, dict):
                param_list = param_group["params"]
            else:
                param_list = param_group

            for p in param_list:
                if len(p.shape) >= 2 and min(p.shape) >= 16:
                    self.muon_params.append(p)
                else:
                    self.adam_params.append(p)

        self.muon = Muon(
            self.muon_params,
            lr=lr,
            momentum=muon_momentum,
            newton_schulz_steps=newton_schulz_steps,
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
        super().__init__([], defaults)

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
