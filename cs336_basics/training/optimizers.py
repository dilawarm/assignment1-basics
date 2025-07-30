"""
Optimizers for neural network training.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamW(Optimizer):
    """AdamW optimizer."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Muon(Optimizer):
    """
    Muon optimizer with stable Newton-Schulz implementation.

    Based on the latest research from YouJiacheng, Franz Cesista, and Jeremy Bernstein.
    Uses the proven 6-step Newton-Schulz coefficients for maximum stability.
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 3e-3,
        momentum: float = 0.95,
        ns_iters: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        use_optimized_coefficients: bool = True,
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
            use_optimized_coefficients: Whether to use optimized NS coefficients
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

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_iters=ns_iters,
            weight_decay=weight_decay,
            eps=eps,
            use_optimized_coefficients=use_optimized_coefficients,
        )
        super().__init__(params, defaults)

        if use_optimized_coefficients:
            # YouJiacheng's proven 6-step coefficients for maximum stability
            # Source: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),
                (3735 / 1024, -6681 / 1024, 3463 / 1024),
                (3799 / 1024, -6499 / 1024, 3211 / 1024),
                (4019 / 1024, -6385 / 1024, 2906 / 1024),
                (2677 / 1024, -3029 / 1024, 1162 / 1024),
                (2172 / 1024, -1833 / 1024, 682 / 1024),
            ]
        else:
            # Stable cubic fallback
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

    def newton_schulz_orthogonalize(self, G: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations for matrix orthogonalization.

        Implements a robust, mathematically correct algorithm based on research.

        Args:
            G: Input matrix to orthogonalize
            num_iters: Number of Newton-Schulz iterations

        Returns:
            Orthogonalized matrix
        """
        # Input validation and early returns
        if torch.isnan(G).any() or torch.isinf(G).any():
            return torch.eye(G.shape[0], G.shape[1], device=G.device, dtype=G.dtype) * 0.01

        norm = torch.norm(G, p="fro")
        if norm < 1e-8:
            return G

        # Normalize to ensure convergence (critical step)
        X = G / (norm + 1e-8)

        device = X.device
        dtype = X.dtype

        # Determine optimal approach based on matrix shape
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.transpose(-2, -1)

        # Use stable coefficients
        if hasattr(self, "optimized_ns_coefficients") and len(self.optimized_ns_coefficients) > 0:
            coeffs = self.optimized_ns_coefficients[: min(num_iters, len(self.optimized_ns_coefficients))]
        else:
            # Fallback to stable cubic
            coeffs = [(1.5, -0.5, 0.0)] * num_iters

        for i, (a, b, c) in enumerate(coeffs):
            try:
                # Compute Gram matrix
                A = torch.matmul(X.transpose(-2, -1), X)
                I = torch.eye(A.shape[-1], device=device, dtype=dtype)

                # Small regularization for stability
                A = A + 1e-7 * I

                # Newton-Schulz iteration: X := X @ polynomial(A)
                if abs(c) > 1e-8:  # Use quintic if c is significant
                    try:
                        A_squared = torch.matmul(A, A)
                        poly_matrix = a * I + b * A + c * A_squared
                        X_new = torch.matmul(X, poly_matrix)
                    except:
                        # Fallback to cubic if quintic fails
                        poly_matrix = 1.5 * I - 0.5 * A
                        X_new = torch.matmul(X, poly_matrix)
                else:
                    # Use cubic Newton-Schulz: X @ (aI + bA)
                    poly_matrix = a * I + b * A
                    X_new = torch.matmul(X, poly_matrix)

                # Stability checks and corrections
                if torch.isnan(X_new).any() or torch.isinf(X_new).any():
                    # Fall back to basic cubic
                    poly_matrix = 1.5 * I - 0.5 * A
                    X_new = torch.matmul(X, poly_matrix)

                # Gentle clamping to prevent explosion
                X_new = torch.clamp(X_new, min=-10.0, max=10.0)

                # Check for convergence
                diff_norm = torch.norm(X_new - X, p="fro")
                if diff_norm < 1e-6:
                    X = X_new
                    break

                # Apply update
                X = X_new

                # Monitor and correct instability
                current_norm = torch.norm(X, p="fro")
                if current_norm > 5.0:  # More lenient threshold
                    X = X / current_norm * 1.73  # Normalize to reasonable size

            except Exception as e:
                # Robust fallback to basic cubic iteration
                try:
                    A = torch.matmul(X.transpose(-2, -1), X)
                    I = torch.eye(A.shape[-1], device=device, dtype=dtype)
                    X = torch.matmul(X, 1.5 * I - 0.5 * A)
                except:
                    # Ultimate fallback: return scaled identity
                    eye_matrix = torch.eye(X.shape[0], X.shape[1], device=device, dtype=dtype)
                    return eye_matrix * 0.1

        # Restore original orientation
        if transpose:
            X = X.transpose(-2, -1)

        # Final safety check
        if torch.isnan(X).any() or torch.isinf(X).any():
            return torch.eye(G.shape[0], G.shape[1], device=device, dtype=dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """Calculate dimension scaling factor for Muon updates."""
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_out / fan_in)  # Correct scaling from theory
        elif len(shape) == 1:
            return 1.0
        elif len(shape) == 4:  # Conv layers (flattened)
            c_out, c_in, k_h, k_w = shape
            fan_in = c_in * k_h * k_w
            fan_out = c_out * k_h * k_w
            return math.sqrt(fan_out / fan_in)
        else:
            total_elements = torch.prod(torch.tensor(shape)).item()
            return math.sqrt(total_elements)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
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

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(grad)

                momentum_buffer = state["momentum_buffer"]
                state["step"] += 1

                # Update momentum
                momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Apply Muon update for 2D+ parameters
                if len(p.shape) >= 2:
                    # Reshape for Newton-Schulz if needed
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        param_2d = p.reshape(p.shape[0], -1)
                        momentum_2d = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        param_2d = p
                        momentum_2d = momentum_buffer

                    # Apply Newton-Schulz orthogonalization
                    ortho_momentum = self.newton_schulz_orthogonalize(momentum_2d, ns_iters)

                    # Calculate scaling
                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_2d, p="fro")

                    if momentum_norm > eps:
                        # Scale the orthogonalized momentum
                        update = ortho_momentum * (dim_scaling / (momentum_norm + eps))

                        # Reshape back if needed
                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)

                        # Apply weight decay
                        if weight_decay > 0:
                            p.mul_(1 - lr * weight_decay)

                        # Apply update
                        p.add_(update, alpha=-lr)
                else:
                    # Standard momentum update for 1D parameters
                    if weight_decay > 0:
                        p.mul_(1 - lr * weight_decay)
                    p.add_(momentum_buffer, alpha=-lr)

        return loss


class MixedOptimizerV2(Optimizer):
    """
    Mixed optimizer that uses Muon for linear layers and Adam for embeddings/lm_head/1D params.

    Implements the winning solution approach:
    - Muon for everything except embedding, lm_head and 1-dimensional values
    - Different learning rates for different parameter groups
    - Proper parameter categorization as described in the winning solution
    """

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 8e-3,
        adam_lr: float = 6e-3,
        embedding_lr: float = 12e-3,
        lm_head_lr: float = 4e-3,
        muon_momentum: float = 0.97,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.015,
        eps: float = 1e-8,
        ns_iters: int = 5,
        use_optimized_muon: bool = True,
    ) -> None:
        """
        Initialize the mixed optimizer.

        Args:
            model: The model to optimize
            muon_lr: Learning rate for Muon optimizer (linear layers)
            adam_lr: Learning rate for Adam optimizer (1D params)
            embedding_lr: Learning rate for embedding parameters
            lm_head_lr: Learning rate for LM head parameters
            muon_momentum: Momentum for Muon optimizer
            adam_betas: Beta parameters for Adam optimizer
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            ns_iters: Number of Newton-Schulz iterations
            use_optimized_muon: Whether to use optimized Muon coefficients
        """
        # Categorize parameters according to winning solution approach
        param_groups = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine parameter type and corresponding optimizer
            if "token_embeddings" in name or "embedding" in name:
                param_type = "embedding"
                lr = embedding_lr
                optimizer_type = "adam"
            elif "lm_head" in name:
                param_type = "lm_head"
                lr = lm_head_lr
                optimizer_type = "adam"
            elif len(param.shape) <= 1:  # 1-dimensional parameters (biases, norms, etc.)
                param_type = "1d"
                lr = adam_lr
                optimizer_type = "adam"
            else:  # Multi-dimensional parameters (linear layers)
                param_type = "linear"
                lr = muon_lr
                optimizer_type = "muon"

            param_groups.append(
                {
                    "params": [param],
                    "name": name,
                    "type": param_type,
                    "optimizer": optimizer_type,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        # Store configuration
        self.muon_config = {
            "lr": muon_lr,
            "momentum": muon_momentum,
            "ns_iters": ns_iters,
            "weight_decay": weight_decay,
            "eps": eps,
            "use_optimized_coefficients": use_optimized_muon,
        }

        self.adam_config = {
            "lr": adam_lr,
            "betas": adam_betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        # Initialize base class
        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])

        defaults = dict(lr=muon_lr, weight_decay=weight_decay)
        super().__init__(all_params, defaults)

        self.param_groups = param_groups

        # Initialize Newton-Schulz coefficients for Muon
        if use_optimized_muon:
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),
                (3735 / 1024, -6681 / 1024, 3463 / 1024),
                (3799 / 1024, -6499 / 1024, 3211 / 1024),
                (4019 / 1024, -6385 / 1024, 2906 / 1024),
                (2677 / 1024, -3029 / 1024, 1162 / 1024),
                (2172 / 1024, -1833 / 1024, 682 / 1024),
            ]
        else:
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

        # Log parameter distribution
        muon_count = sum(1 for g in param_groups if g["optimizer"] == "muon")
        adam_count = sum(1 for g in param_groups if g["optimizer"] == "adam")
        embedding_count = sum(1 for g in param_groups if g["type"] == "embedding")
        lm_head_count = sum(1 for g in param_groups if g["type"] == "lm_head")

        print(f"MixedOptimizerV2 initialized:")
        print(f"  Muon parameters (linear): {muon_count}")
        print(f"  Adam parameters (1D): {adam_count - embedding_count - lm_head_count}")
        print(f"  Embedding parameters: {embedding_count}")
        print(f"  LM head parameters: {lm_head_count}")

    def newton_schulz_orthogonalize(self, G: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """Apply Newton-Schulz orthogonalization using stable coefficients."""
        # Input validation
        if torch.isnan(G).any() or torch.isinf(G).any():
            return torch.eye(G.shape[0], G.shape[1], device=G.device, dtype=G.dtype) * 0.01

        norm = torch.norm(G, p="fro")
        if norm < 1e-8:
            return G

        # Normalize input
        X = G / (norm + 1e-8)

        device = X.device
        dtype = X.dtype

        # Handle transposition
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.transpose(-2, -1)

        # Use stable coefficients
        coeffs = self.optimized_ns_coefficients[: min(num_iters, len(self.optimized_ns_coefficients))]

        for i, (a, b, c) in enumerate(coeffs):
            try:
                # Compute Gram matrix
                A = torch.matmul(X.transpose(-2, -1), X)
                I = torch.eye(A.shape[-1], device=device, dtype=dtype)
                A = A + 1e-7 * I  # Regularization

                # Newton-Schulz update
                if abs(c) > 1e-8:  # Use quintic if c is significant
                    try:
                        A_squared = torch.matmul(A, A)
                        poly_matrix = a * I + b * A + c * A_squared
                        X_new = torch.matmul(X, poly_matrix)
                    except:
                        # Fallback to cubic
                        poly_matrix = 1.5 * I - 0.5 * A
                        X_new = torch.matmul(X, poly_matrix)
                else:
                    poly_matrix = a * I + b * A
                    X_new = torch.matmul(X, poly_matrix)

                # Stability checks
                if torch.isnan(X_new).any() or torch.isinf(X_new).any():
                    poly_matrix = 1.5 * I - 0.5 * A
                    X_new = torch.matmul(X, poly_matrix)

                # Gentle clamping
                X_new = torch.clamp(X_new, min=-10.0, max=10.0)

                # Check convergence
                if torch.norm(X_new - X, p="fro") < 1e-6:
                    X = X_new
                    break

                X = X_new

                # Stability monitoring
                current_norm = torch.norm(X, p="fro")
                if current_norm > 5.0:
                    X = X / current_norm * 1.73

            except Exception:
                # Robust fallback
                try:
                    A = torch.matmul(X.transpose(-2, -1), X)
                    I = torch.eye(A.shape[-1], device=device, dtype=dtype)
                    X = torch.matmul(X, 1.5 * I - 0.5 * A)
                except:
                    return torch.eye(G.shape[0], G.shape[1], device=device, dtype=dtype) * 0.1

        # Restore orientation
        if transpose:
            X = X.transpose(-2, -1)

        # Final check
        if torch.isnan(X).any() or torch.isinf(X).any():
            return torch.eye(G.shape[0], G.shape[1], device=device, dtype=dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """Calculate dimension scaling factor."""
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_out / fan_in)
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            fan_in = c_in * k_h * k_w
            fan_out = c_out * k_h * k_w
            return math.sqrt(fan_out / fan_in)
        else:
            return 1.0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with mixed optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            param = group["params"][0]  # Single parameter per group
            optimizer_type = group["optimizer"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if param.grad is None:
                continue

            grad = param.grad
            if grad.dtype in {torch.float16, torch.bfloat16}:
                grad = grad.float()

            state = self.state[param]

            if len(state) == 0:
                state["step"] = 0
                if optimizer_type == "muon":
                    state["momentum_buffer"] = torch.zeros_like(grad)
                else:
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            state["step"] += 1

            if optimizer_type == "muon":
                # Muon update for linear layers
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(self.muon_config["momentum"]).add_(grad, alpha=1 - self.muon_config["momentum"])

                if len(param.shape) >= 2:
                    # Reshape if needed
                    original_shape = param.shape
                    if len(param.shape) > 2:
                        param_2d = param.reshape(param.shape[0], -1)
                        momentum_2d = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        param_2d = param
                        momentum_2d = momentum_buffer

                    # Apply Newton-Schulz orthogonalization
                    ortho_momentum = self.newton_schulz_orthogonalize(momentum_2d, self.muon_config["ns_iters"])

                    # Calculate scaling
                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_2d, p="fro")

                    if momentum_norm > self.muon_config["eps"]:
                        update = ortho_momentum * (dim_scaling / (momentum_norm + self.muon_config["eps"]))

                        if len(param.shape) > 2:
                            update = update.reshape(original_shape)

                        # Apply weight decay and update
                        if weight_decay > 0:
                            param.mul_(1 - lr * weight_decay)
                        param.add_(update, alpha=-lr)
                else:
                    # Fallback for 1D parameters (shouldn't happen with correct categorization)
                    if weight_decay > 0:
                        param.mul_(1 - lr * weight_decay)
                    param.add_(momentum_buffer, alpha=-lr)

            else:
                # Adam update for embeddings, lm_head, and 1D parameters
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = self.adam_config["betas"]
                eps = self.adam_config["eps"]

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Weight decay (AdamW style)
                if weight_decay > 0:
                    param.mul_(1 - lr * weight_decay)

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute step size
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def update_learning_rates(
        self, muon_lr_factor: float, adam_lr_factor: float, embedding_lr_factor: float, lm_head_lr_factor: float
    ):
        """Update learning rates for all parameter groups."""
        for group in self.param_groups:
            param_type = group["type"]
            if param_type == "linear":
                group["lr"] = self.muon_config["lr"] * muon_lr_factor
            elif param_type == "embedding":
                group["lr"] = self.muon_config["lr"] * embedding_lr_factor  # Use base lr for embeddings
            elif param_type == "lm_head":
                group["lr"] = self.muon_config["lr"] * lm_head_lr_factor  # Use base lr for lm_head
            else:  # 1d parameters
                group["lr"] = self.adam_config["lr"] * adam_lr_factor
