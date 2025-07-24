"""Optimizers for training neural networks."""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer implementation.

    This implementation follows the algorithm described in "Decoupled Weight Decay Regularization"
    by Loshchilov and Hutter (2019).
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
            betas: Coefficients used for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay coefficient
        """
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group["lr"] / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Muon(Optimizer):
    """
    Muon optimizer implementation.

    Muon is a state-of-the-art optimizer that uses geometric principles and Newton-Schulz
    orthogonalization for faster convergence and automatic learning rate transfer.

    Based on "Muon: Fast, Accurate Neural-Network Training using Reparameterization and a Spectral Method"
    by Bernstein et al. (2024).
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 3e-3,
        momentum: float = 0.95,
        ns_iters: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize Muon optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate (typically higher than AdamW)
            momentum: Momentum factor for exponential moving average
            ns_iters: Number of Newton-Schulz iterations for orthogonalization
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 1 <= ns_iters <= 10:
            raise ValueError(f"Invalid ns_iters value: {ns_iters}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, momentum=momentum, ns_iters=ns_iters, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations to approximate orthogonalization.

        The Newton-Schulz method applies the polynomial f(X) = (3X - X^3)/2 iteratively
        to force all singular values to 1 while preserving singular vectors.

        Args:
            X: Input matrix to orthogonalize
            num_iters: Number of Newton-Schulz iterations

        Returns:
            Orthogonalized matrix
        """
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf detected in Newton-Schulz input, returning identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype)

        norm = torch.norm(X, p="fro")
        if norm < 1e-8:
            return X

        X = X / (norm + 1e-10)

        X = torch.clamp(X, min=-5.0, max=5.0)

        for i in range(num_iters):
            X_squared = torch.matmul(X, X.transpose(-2, -1))

            reg_strength = 1e-6
            X_squared = X_squared + reg_strength * torch.eye(X_squared.shape[-1], device=X.device, dtype=X.dtype)

            X_cubed = torch.matmul(X_squared, X)

            if torch.isnan(X_cubed).any() or torch.isinf(X_cubed).any():
                print(f"WARNING: NaN/Inf detected in Newton-Schulz iteration {i}, stopping early")
                break

            X_new = (3 * X - X_cubed) / 2

            X_new = torch.clamp(X_new, min=-10.0, max=10.0)

            if torch.norm(X_new - X, p="fro") < 1e-5:
                X = X_new
                break

            X = X_new

            if torch.norm(X, p="fro") > 20.0:
                print(f"WARNING: Newton-Schulz values becoming too large at iteration {i}, applying correction")
                X = X / torch.norm(X, p="fro") * 2.0

        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf in final Newton-Schulz result, returning scaled identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """
        Calculate the appropriate dimension scaling factor.

        For matrices (linear layers), this is sqrt(fan_in * fan_out).
        For other parameter types, we use appropriate heuristics.

        Args:
            shape: Shape of the parameter tensor

        Returns:
            Scaling factor
        """
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_in * fan_out)
        elif len(shape) == 1:
            return math.sqrt(shape[0])
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            return math.sqrt(c_in * c_out * k_h * k_w)
        else:
            return math.sqrt(torch.prod(torch.tensor(shape)).item())

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_iters = group["ns_iters"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(grad)

                momentum_buffer = state["momentum_buffer"]
                state["step"] += 1

                momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

                if len(p.shape) >= 2:
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        p_flat = p.reshape(p.shape[0], -1)
                        momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        p_flat = p
                        momentum_flat = momentum_buffer

                    ortho_momentum = self.newton_schulz_orthogonalize(momentum_flat, ns_iters)

                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_flat, p="fro")

                    if momentum_norm > eps:
                        scaling = dim_scaling / (momentum_norm + eps)
                        update = ortho_momentum * scaling

                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)

                        p.add_(update, alpha=-lr)
                else:
                    p.add_(momentum_buffer, alpha=-lr)

        return loss


class MixedOptimizer(Optimizer):
    """
    Mixed optimizer that uses different optimizers for different parameter types.

    This follows the approach:
    - Muon for most parameters (linear layers)
    - AdamW for embeddings, LM head, and 1-dimensional parameters
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        muon_lr: float = 3e-3,
        adamw_lr: float = 3e-3,
        embedding_lr: float = 4e-3,
        lm_head_lr: float = 2e-3,
        muon_momentum: float = 0.95,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        ns_iters: int = 5,
    ) -> None:
        """
        Initialize mixed optimizer.

        Args:
            params: Iterator of parameters to optimize
            muon_lr: Learning rate for Muon optimizer
            adamw_lr: Learning rate for AdamW optimizer
            embedding_lr: Learning rate for embedding parameters
            lm_head_lr: Learning rate for LM head parameters
            muon_momentum: Momentum for Muon optimizer
            adamw_betas: Beta parameters for AdamW optimizer
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            ns_iters: Number of Newton-Schulz iterations
        """
        defaults = dict(
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            embedding_lr=embedding_lr,
            lm_head_lr=lm_head_lr,
            muon_momentum=muon_momentum,
            adamw_betas=adamw_betas,
            weight_decay=weight_decay,
            eps=eps,
            ns_iters=ns_iters,
        )
        super().__init__(params, defaults)

        self.muon_params = []
        self.adamw_params = []
        self.embedding_params = []
        self.lm_head_params = []

    def categorize_parameter(self, name: str, param: torch.nn.Parameter) -> str:
        """
        Categorize parameter based on its name and properties.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            Category string: 'muon', 'adamw', 'embedding', or 'lm_head'
        """
        if "embedding" in name.lower() or "wte" in name.lower():
            return "embedding"
        elif "lm_head" in name.lower() or "final" in name.lower():
            return "lm_head"
        elif len(param.shape) == 1:
            return "adamw"
        else:
            return "muon"

    @torch.no_grad()
    def step(self, closure=None, param_names=None):
        """
        Perform a single optimization step with mixed optimizers and enhanced stability.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            param_names: Dictionary mapping parameters to their names
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print("WARNING: NaN/Inf gradients detected in MixedOptimizer, skipping parameter update")
                    continue

                param_name = param_names.get(p, "") if param_names else ""
                category = self.categorize_parameter(param_name, p)

                if category == "embedding":
                    lr = group.get("embedding_lr", group.get("adamw_lr", 1e-3))
                    optimizer_type = "adamw"
                elif category == "lm_head":
                    lr = group.get("lm_head_lr", group.get("adamw_lr", 1e-3))
                    optimizer_type = "adamw"
                elif category == "adamw":
                    lr = group.get("adamw_lr", 1e-3)
                    optimizer_type = "adamw"
                else:
                    lr = group.get("muon_lr", 1e-3)
                    optimizer_type = "muon"

                if lr <= 0 or not torch.isfinite(torch.tensor(lr)):
                    print(f"WARNING: Invalid learning rate {lr} for parameter {param_name}, using fallback")
                    lr = 1e-4

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                grad_norm = torch.norm(grad)
                if grad_norm > 10.0:
                    grad = grad * (10.0 / grad_norm)

                state = self.state[p]

                if optimizer_type == "muon":
                    self._apply_muon_step(p, grad, state, group, lr)
                else:
                    self._apply_adamw_step(p, grad, state, group, lr)

        return loss

    def _apply_muon_step(self, param, grad, state, group, lr):
        """Apply Muon optimization step with enhanced numerical stability."""
        if group["weight_decay"] > 0:
            param.mul_(1 - lr * group["weight_decay"])

        if len(state) == 0:
            state["step"] = 0
            state["momentum_buffer"] = torch.zeros_like(grad)

        momentum_buffer = state["momentum_buffer"]
        state["step"] += 1

        momentum = group.get("muon_momentum", 0.95)
        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

        if torch.isnan(momentum_buffer).any() or torch.isinf(momentum_buffer).any():
            print("WARNING: NaN/Inf in momentum buffer, resetting")
            momentum_buffer.zero_()

        if len(param.shape) >= 2:
            muon = Muon(
                [param], lr=lr, momentum=momentum, ns_iters=group.get("ns_iters", 5), eps=group.get("eps", 1e-8)
            )

            muon.state[param] = state

            original_shape = param.shape
            if len(param.shape) > 2:
                momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
            else:
                momentum_flat = momentum_buffer

            try:
                ortho_momentum = muon.newton_schulz_orthogonalize(momentum_flat, group.get("ns_iters", 5))
            except Exception as e:
                print(f"WARNING: Newton-Schulz failed: {e}, using scaled momentum")
                ortho_momentum = momentum_flat * 0.1

            dim_scaling = muon.get_dimension_scaling(original_shape)
            momentum_norm = torch.norm(momentum_flat, p="fro")

            eps = group.get("eps", 1e-8)
            if momentum_norm > eps:
                scaling = dim_scaling / (momentum_norm + eps)

                scaling = torch.clamp(scaling, min=1e-6, max=100.0)

                update = ortho_momentum * scaling

                if len(param.shape) > 2:
                    update = update.reshape(original_shape)

                update_norm = torch.norm(update)
                if update_norm > 1.0:
                    update = update * (1.0 / update_norm)

                param.add_(update, alpha=-lr)
        else:
            param.add_(momentum_buffer, alpha=-lr)

    def _apply_adamw_step(self, param, grad, state, group, lr):
        """Apply AdamW optimization step."""
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["adamw_betas"]

        state["step"] += 1
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

        if group["weight_decay"] > 0:
            param.mul_(1 - lr * group["weight_decay"])

        param.addcdiv_(exp_avg, denom, value=-step_size)
